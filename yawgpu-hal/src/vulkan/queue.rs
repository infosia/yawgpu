use super::*;

/// Stores vulkan queue data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct VulkanQueue {
    pub(super) inner: Arc<VulkanQueueInner>,
}

/// Holds shared state for the vulkan queue handle.
#[derive(Debug)]
pub(super) struct VulkanQueueInner {
    pub(super) device: Arc<VulkanDeviceInner>,
    pub(super) queue: vk::Queue,
    pub(super) queue_access: Mutex<()>,
    pub(super) retire: Mutex<RetireRing>,
    pub(super) submissions: Arc<Mutex<SubmissionTracker>>,
}

#[derive(Debug)]
pub(super) struct SubmissionTracker {
    last_issued: SubmissionIndex,
    completed: SubmissionIndex,
    fences: VecDeque<(SubmissionIndex, vk::Fence)>,
    pinned_fences: Vec<(vk::Fence, usize)>,
    pending_destroy: Vec<vk::Fence>,
}

impl SubmissionTracker {
    pub(super) fn new() -> Self {
        Self {
            last_issued: SubmissionIndex::NONE,
            completed: SubmissionIndex::NONE,
            fences: VecDeque::new(),
            pinned_fences: Vec::new(),
            pending_destroy: Vec::new(),
        }
    }

    pub(super) fn reserve(&mut self) -> Result<SubmissionIndex, HalError> {
        let next = self
            .last_issued
            .0
            .checked_add(1)
            .map(SubmissionIndex)
            .ok_or_else(|| HalError::QueueSubmissionFailed {
                backend: BACKEND,
                message: "submission index exhausted".to_string(),
            })?;
        self.last_issued = next;
        Ok(next)
    }

    pub(super) fn register_fence(&mut self, index: SubmissionIndex, fence: vk::Fence) {
        self.fences.push_back((index, fence));
    }

    pub(super) fn mark_completed(&mut self, index: SubmissionIndex) {
        self.completed = self.completed.max(index);
    }

    pub(super) fn pin_fence(&mut self, index: SubmissionIndex) -> Option<vk::Fence> {
        let fence = self
            .fences
            .iter()
            .find_map(|(mapped_index, fence)| (*mapped_index == index).then_some(*fence))?;
        if let Some((_, pin_count)) = self
            .pinned_fences
            .iter_mut()
            .find(|(pinned_fence, _)| *pinned_fence == fence)
        {
            *pin_count += 1;
        } else {
            self.pinned_fences.push((fence, 1));
        }
        Some(fence)
    }

    pub(super) fn remove_fence(&mut self, index: SubmissionIndex) -> Option<vk::Fence> {
        if let Some(position) = self
            .fences
            .iter()
            .position(|(mapped_index, _)| *mapped_index == index)
        {
            self.fences.remove(position).map(|(_, fence)| fence)
        } else {
            None
        }
    }

    pub(super) fn defer_destroy_if_pinned(&mut self, fence: vk::Fence) -> bool {
        if self
            .pinned_fences
            .iter()
            .any(|(pinned_fence, _)| *pinned_fence == fence)
        {
            debug_assert!(!self.pending_destroy.contains(&fence));
            self.pending_destroy.push(fence);
            true
        } else {
            false
        }
    }

    pub(super) fn unpin_fence(&mut self, fence: vk::Fence) -> bool {
        let Some(position) = self
            .pinned_fences
            .iter()
            .position(|(pinned_fence, _)| *pinned_fence == fence)
        else {
            debug_assert!(false, "unpinning a fence that is not pinned");
            return false;
        };
        if self.pinned_fences[position].1 > 1 {
            self.pinned_fences[position].1 -= 1;
            return false;
        }
        self.pinned_fences.swap_remove(position);
        if let Some(position) = self
            .pending_destroy
            .iter()
            .position(|pending_fence| *pending_fence == fence)
        {
            self.pending_destroy.swap_remove(position);
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct TrackedSubmission {
    pub(super) index: SubmissionIndex,
    pub(super) tracker: Arc<Mutex<SubmissionTracker>>,
}

impl Drop for VulkanQueueInner {
    fn drop(&mut self) {
        let mut retire = self
            .retire
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _ = retire.wait_all(&self.device.device);
    }
}

impl VulkanQueue {
    /// Submits an empty command buffer to flush the queue.
    pub fn submit_empty(&self) -> Result<SubmissionIndex, HalError> {
        let fence_info = vk::FenceCreateInfo::default();
        let mut fence = Some(
            unsafe { self.inner.device.device.create_fence(&fence_info, None) }
                .map_err(|error| queue_submission_error("vkCreateFence", error))?,
        );
        let submitted = (|| {
            let _queue_access = self
                .inner
                .queue_access
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut submissions = self
                .inner
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let index = submissions.reserve()?;
            let submitted_fence = fence.expect("empty submission fence should exist");
            unsafe {
                self.inner
                    .device
                    .device
                    .queue_submit(self.inner.queue, &[], submitted_fence)
                    .map_err(|error| queue_submission_error("vkQueueSubmit", error))?;
            }
            submissions.register_fence(index, submitted_fence);
            Ok((
                index,
                TrackedSubmission {
                    index,
                    tracker: Arc::clone(&self.inner.submissions),
                },
            ))
        })();
        let (index, tracked_submission) = match submitted {
            Ok(submitted) => submitted,
            Err(error) => {
                unsafe {
                    self.inner
                        .device
                        .device
                        .destroy_fence(fence.take().expect("empty submission fence"), None);
                }
                return Err(error);
            }
        };
        let submitted_fence = fence.take().expect("submitted empty fence should exist");
        {
            let mut retire = self
                .inner
                .retire
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            retire.retire_tracked(
                &self.inner.device.device,
                submitted_fence,
                Vec::new(),
                Vec::new(),
                true,
                Some(tracked_submission),
            )?;
        }
        self.wait_for_submission(index)?;
        Ok(index)
    }

    /// Waits until all submitted queue work has completed.
    pub fn wait_idle(&self) -> Result<(), HalError> {
        let _queue_access = self
            .inner
            .queue_access
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let completed_through = self
            .inner
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .last_issued;
        unsafe {
            self.inner
                .device
                .device
                .queue_wait_idle(self.inner.queue)
                .map_err(|error| queue_submission_error("vkQueueWaitIdle", error))?;
        }
        self.inner
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .mark_completed(completed_through);
        Ok(())
    }

    /// Returns the highest submission index proven complete without blocking.
    pub fn completed_submission_index(&self) -> Result<SubmissionIndex, HalError> {
        let mut submissions = self
            .inner
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for position in 0..submissions.fences.len() {
            let (index, fence) = submissions.fences[position];
            if index <= submissions.completed {
                continue;
            }
            match unsafe { self.inner.device.device.get_fence_status(fence) } {
                Ok(true) => submissions.mark_completed(index),
                Ok(false) => break,
                Err(error) => return Err(queue_submission_error("vkGetFenceStatus", error)),
            }
        }
        Ok(submissions.completed)
    }

    /// Blocks until the requested submission index has completed.
    pub fn wait_for_submission(&self, index: SubmissionIndex) -> Result<(), HalError> {
        let fence = {
            let mut submissions = self
                .inner
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if index <= submissions.completed {
                return Ok(());
            }
            submissions
                .pin_fence(index)
                .ok_or_else(|| HalError::QueueSubmissionFailed {
                    backend: BACKEND,
                    message: "submission index has not been issued".to_string(),
                })?
        };
        let wait_result = unsafe {
            self.inner
                .device
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|error| queue_submission_error("vkWaitForFences", error))
        };
        let destroy_fence = {
            let mut submissions = self
                .inner
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if wait_result.is_ok() {
                submissions.mark_completed(index);
            }
            // If the waiter finishes first, this only drops its pin and the
            // retire-slot cleanup later destroys the fence. If retirement
            // removed the mapping first, the fence is pending destruction and
            // the last waiter to unpin is instead responsible for destroying it.
            submissions.unpin_fence(fence)
        };
        if destroy_fence {
            unsafe {
                self.inner.device.device.destroy_fence(fence, None);
            }
        }
        wait_result
    }

    /// Records and submits the given buffer/texture copy operations.
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<SubmissionIndex, HalError> {
        if copies.is_empty() {
            return self.submit_empty();
        }
        submit_copies(&self.inner, copies)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;
    use crate::HalBuffer;

    #[test]
    fn submission_tracker_defers_pinned_fence_destruction_until_unpin() {
        use ash::vk::Handle;

        let mut tracker = SubmissionTracker::new();
        let index = tracker.reserve().expect("reserve submission index");
        let fence = vk::Fence::from_raw(1);
        tracker.register_fence(index, fence);

        assert_eq!(tracker.pin_fence(index), Some(fence));
        assert_eq!(tracker.remove_fence(index), Some(fence));
        assert!(tracker.defer_destroy_if_pinned(fence));
        assert_eq!(tracker.pending_destroy, [fence]);

        assert!(tracker.unpin_fence(fence));
        assert!(tracker.pinned_fences.is_empty());
        assert!(tracker.pending_destroy.is_empty());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_queue_submit_empty_completes() {
        vulkan_device()
            .queue()
            .submit_empty()
            .expect("submit empty queue work");
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_queue_completion_index_polls_and_waits_for_submission_fence() {
        let device = vulkan_device();
        let queue = device.queue();
        let source = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("Vulkan source buffer allocation should succeed");
        let destination = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("Vulkan destination buffer allocation should succeed");

        assert_eq!(
            queue
                .completed_submission_index()
                .expect("query initial completion"),
            SubmissionIndex::NONE
        );
        let submitted = queue
            .submit_copies(&[HalCopy::Buffer(HalBufferCopy {
                source: HalBuffer::Vulkan(source),
                source_offset: 0,
                destination: HalBuffer::Vulkan(destination),
                destination_offset: 0,
                size: 4,
            })])
            .expect("submit buffer copy");

        let polled = queue
            .completed_submission_index()
            .expect("poll submission fence");
        assert!(polled == SubmissionIndex::NONE || polled == submitted);
        queue
            .wait_for_submission(submitted)
            .expect("wait for submission fence");
        assert_eq!(
            queue
                .completed_submission_index()
                .expect("query waited completion"),
            submitted
        );
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_queue_submit_copies_accepts_buffer_copy() {
        let device = vulkan_device();
        let source = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("Vulkan source buffer allocation should succeed");
        let destination = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("Vulkan destination buffer allocation should succeed");
        source.write(0, &[1, 2, 3, 4]).expect("write source");
        device
            .queue()
            .submit_copies(&[HalCopy::Buffer(HalBufferCopy {
                source: HalBuffer::Vulkan(source),
                source_offset: 0,
                destination: HalBuffer::Vulkan(destination.clone()),
                destination_offset: 0,
                size: 4,
            })])
            .expect("submit buffer copy");
        // submit_copies is asynchronous (the retire ring waits fences only on
        // slot reuse), so the copy must be drained before reading back.
        device.queue().wait_idle().expect("wait idle");
        assert_eq!(
            destination.read(0, 4).expect("read destination"),
            [1, 2, 3, 4]
        );
    }
}
