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
    pub(super) retire: Mutex<SubmissionRetireQueue>,
    pub(super) recycler: Arc<Mutex<SubmitRecycler>>,
    pub(super) submissions: Arc<Mutex<SubmissionTracker>>,
}

const MAX_RECYCLED_SUBMIT_OBJECTS: usize = 16;

#[derive(Debug, Default)]
pub(super) struct SubmitRecycler {
    pools: Vec<vk::CommandPool>,
    fences: Vec<vk::Fence>,
}

impl SubmitRecycler {
    pub(super) fn take_pool(
        &mut self,
        device: &VulkanDeviceInner,
    ) -> Result<vk::CommandPool, HalError> {
        if let Some(pool) = self.pools.pop() {
            return Ok(pool);
        }
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.queue_family_index);
        unsafe { device.device.create_command_pool(&info, None) }
            .map_err(|error| queue_submission_error("vkCreateCommandPool", error))
    }

    pub(super) fn take_fence(&mut self, device: &ash::Device) -> Result<vk::Fence, HalError> {
        if let Some(fence) = self.fences.pop() {
            return Ok(fence);
        }
        unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|error| queue_submission_error("vkCreateFence", error))
    }

    pub(super) fn recycle_pool(&mut self, device: &ash::Device, pool: vk::CommandPool) {
        unsafe {
            if device
                .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())
                .is_ok()
                && self.pools.len() < MAX_RECYCLED_SUBMIT_OBJECTS
            {
                self.pools.push(pool);
            } else {
                device.destroy_command_pool(pool, None);
            }
        }
    }

    pub(super) fn recycle_fence(&mut self, device: &ash::Device, fence: vk::Fence) {
        unsafe {
            if device.reset_fences(&[fence]).is_ok()
                && self.fences.len() < MAX_RECYCLED_SUBMIT_OBJECTS
            {
                self.fences.push(fence);
            } else {
                device.destroy_fence(fence, None);
            }
        }
    }

    fn destroy_free(&mut self, device: &ash::Device) {
        unsafe {
            for pool in self.pools.drain(..) {
                device.destroy_command_pool(pool, None);
            }
            for fence in self.fences.drain(..) {
                device.destroy_fence(fence, None);
            }
        }
    }
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
    pub(super) recycler: Arc<Mutex<SubmitRecycler>>,
}

impl Drop for VulkanQueueInner {
    fn drop(&mut self) {
        let mut retire = self
            .retire
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _ = retire.wait_all(&self.device.device);
        self.recycler
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .destroy_free(&self.device.device);
    }
}

impl VulkanQueue {
    /// Submits an empty command buffer to flush the queue.
    pub fn submit_empty(&self) -> Result<SubmissionIndex, HalError> {
        let index = submit_copies(&self.inner, &[])?;
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
        drop(_queue_access);
        self.inner
            .retire
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .harvest(&self.inner.device.device)?;
        Ok(())
    }

    fn try_harvest(&self) {
        let mut retire = match self.inner.retire.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(error)) => error.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return,
        };
        // Retirement is opportunistic and must not change a poll or wait result.
        let _ = retire.harvest(&self.inner.device.device);
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
        let completed = submissions.completed;
        drop(submissions);
        self.try_harvest();
        Ok(completed)
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
                drop(submissions);
                self.try_harvest();
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
            // retirement later recycles the fence. If retirement
            // removed the mapping first, the fence is pending recycling and
            // the last waiter to unpin is instead responsible for recycling it.
            submissions.unpin_fence(fence)
        };
        if destroy_fence {
            self.inner
                .recycler
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .recycle_fence(&self.inner.device.device, fence);
        }
        wait_result?;
        self.try_harvest();
        Ok(())
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
    fn submission_tracker_multiple_pins_defer_recycling_until_last_unpin() {
        use ash::vk::Handle;
        let mut tracker = SubmissionTracker::new();
        let index = tracker.reserve().unwrap();
        let fence = vk::Fence::from_raw(1);
        tracker.register_fence(index, fence);
        assert_eq!(tracker.pin_fence(index), Some(fence));
        assert_eq!(tracker.pin_fence(index), Some(fence));
        assert_eq!(tracker.remove_fence(index), Some(fence));
        assert!(tracker.defer_destroy_if_pinned(fence));
        assert!(!tracker.unpin_fence(fence));
        assert_eq!(tracker.pending_destroy, [fence]);
        assert!(tracker.unpin_fence(fence));
        assert!(tracker.pending_destroy.is_empty());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_queue_poll_harvests_retained_resources() {
        let device = vulkan_device();
        let queue = device.queue();
        let buffer = device.create_buffer(4, HalBufferUsage::default()).unwrap();
        let inner = Arc::clone(buffer.inner.as_ref().unwrap());
        let copy = HalCopy::BufferClear(crate::HalBufferClear {
            buffer: HalBuffer::Vulkan(buffer),
            offset: 0,
            size: 4,
        });
        let baseline = Arc::strong_count(&inner);
        let index = submit_copies(&queue.inner, &[copy]).unwrap();
        // Raw wait leaves retirement exclusively to the public poll below.
        unsafe { device.inner.device.queue_wait_idle(queue.inner.queue) }.unwrap();
        assert_eq!(Arc::strong_count(&inner), baseline);
        assert_eq!(queue.completed_submission_index().unwrap(), index);
        assert_eq!(Arc::strong_count(&inner), baseline - 1);
        assert_eq!(queue.inner.recycler.lock().unwrap().pools.len(), 1);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_queue_poll_and_wait_skip_locked_retirement() {
        let device = vulkan_device();
        let queue = device.queue();
        let index = submit_copies(&queue.inner, &[]).unwrap();
        // Complete GPU work without updating the tracker or harvesting.
        unsafe { device.inner.device.queue_wait_idle(queue.inner.queue) }.unwrap();
        let retirement = queue.inner.retire.lock().unwrap();
        let worker_queue = queue.clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            // Exercise the post-wait harvest, poll harvest, and early-return
            // harvest while the main thread holds the retirement lock.
            worker_queue.wait_for_submission(index).unwrap();
            assert_eq!(worker_queue.completed_submission_index().unwrap(), index);
            worker_queue.wait_for_submission(index).unwrap();
            sender.send(()).unwrap();
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("poll and both wait paths must finish while retirement is locked");
        worker.join().unwrap();
        drop(retirement);
        queue.wait_idle().unwrap();
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_queue_wait_idle_harvests_retained_resources() {
        let device = vulkan_device();
        let queue = device.queue();
        let buffer = device.create_buffer(4, HalBufferUsage::default()).unwrap();
        let inner = Arc::clone(buffer.inner.as_ref().unwrap());
        let baseline = Arc::strong_count(&inner);
        let copy = HalCopy::BufferClear(crate::HalBufferClear {
            buffer: HalBuffer::Vulkan(buffer.clone()),
            offset: 0,
            size: 4,
        });
        submit_copies(&queue.inner, &[copy]).unwrap();
        assert_eq!(Arc::strong_count(&inner), baseline + 1);
        queue.wait_idle().unwrap();
        assert_eq!(Arc::strong_count(&inner), baseline);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_queue_reuses_command_pool_and_unsignalled_fence() {
        let device = vulkan_device();
        let queue = device.queue();
        let pool = {
            let mut recycler = queue.inner.recycler.lock().unwrap();
            let pool = recycler.take_pool(&device.inner).unwrap();
            recycler.recycle_pool(&device.inner.device, pool);
            pool
        };
        let first = submit_copies(&queue.inner, &[]).unwrap();
        let fence = queue.inner.submissions.lock().unwrap().fences[0].1;
        queue.wait_for_submission(first).unwrap();
        {
            let recycler = queue.inner.recycler.lock().unwrap();
            assert_eq!(recycler.pools, [pool]);
            assert_eq!(recycler.fences, [fence]);
            assert!(!unsafe { device.inner.device.get_fence_status(fence) }.unwrap());
        }
        let second = submit_copies(&queue.inner, &[]).unwrap();
        assert_eq!(queue.inner.submissions.lock().unwrap().fences[0].1, fence);
        assert!(queue.inner.recycler.lock().unwrap().pools.is_empty());
        queue.wait_for_submission(second).unwrap();
        assert_eq!(queue.inner.recycler.lock().unwrap().pools, [pool]);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_queue_pinned_fence_recycles_only_after_last_unpin() {
        let device = vulkan_device();
        let queue = device.queue();
        let index = submit_copies(&queue.inner, &[]).unwrap();
        let fence = {
            let mut tracker = queue.inner.submissions.lock().unwrap();
            let fence = tracker.pin_fence(index).unwrap();
            assert_eq!(tracker.pin_fence(index), Some(fence));
            fence
        };
        unsafe {
            device
                .inner
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
        }
        .unwrap();
        queue.completed_submission_index().unwrap();
        assert!(queue.inner.recycler.lock().unwrap().fences.is_empty());
        assert!(unsafe { device.inner.device.get_fence_status(fence) }.unwrap());
        {
            let mut tracker = queue.inner.submissions.lock().unwrap();
            assert!(!tracker.unpin_fence(fence));
            assert!(unsafe { device.inner.device.get_fence_status(fence) }.unwrap());
            assert!(tracker.unpin_fence(fence));
        }
        queue
            .inner
            .recycler
            .lock()
            .unwrap()
            .recycle_fence(&device.inner.device, fence);
        assert_eq!(queue.inner.recycler.lock().unwrap().fences, [fence]);
        assert!(!unsafe { device.inner.device.get_fence_status(fence) }.unwrap());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_submit_recycler_caps_free_lists() {
        let device = vulkan_device();
        let mut recycler = SubmitRecycler::default();
        let mut pools = Vec::new();
        let mut fences = Vec::new();
        for _ in 0..MAX_RECYCLED_SUBMIT_OBJECTS + 3 {
            pools.push(recycler.take_pool(&device.inner).unwrap());
            fences.push(recycler.take_fence(&device.inner.device).unwrap());
        }
        for pool in pools {
            recycler.recycle_pool(&device.inner.device, pool);
        }
        for fence in fences {
            recycler.recycle_fence(&device.inner.device, fence);
        }
        assert_eq!(recycler.pools.len(), MAX_RECYCLED_SUBMIT_OBJECTS);
        assert_eq!(recycler.fences.len(), MAX_RECYCLED_SUBMIT_OBJECTS);
        recycler.destroy_free(&device.inner.device);
        assert!(recycler.pools.is_empty());
        assert!(recycler.fences.is_empty());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_queue_recording_failure_recycles_pool() {
        let device = vulkan_device();
        let queue = device.queue();
        let buffer = device.create_buffer(4, HalBufferUsage::default()).unwrap();
        let copy = HalCopy::Buffer(HalBufferCopy {
            source: HalBuffer::Vulkan(buffer.clone()),
            source_offset: 0,
            destination: HalBuffer::Vulkan(buffer),
            destination_offset: 0,
            size: 8,
        });
        assert!(submit_copies(&queue.inner, &[copy]).is_err());
        assert_eq!(queue.inner.recycler.lock().unwrap().pools.len(), 1);
        assert!(queue.inner.submissions.lock().unwrap().fences.is_empty());
        let index = submit_copies(&queue.inner, &[]).unwrap();
        queue.wait_for_submission(index).unwrap();
        assert_eq!(queue.inner.recycler.lock().unwrap().pools.len(), 1);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_queue_failure_after_fence_allocation_recycles_objects() {
        let device = vulkan_device();
        let queue = device.queue();
        // Force a safe error after pool recording and fence acquisition,
        // immediately before vkQueueSubmit, without a device-loss injection.
        queue.inner.submissions.lock().unwrap().last_issued = SubmissionIndex(u64::MAX);
        assert!(submit_copies(&queue.inner, &[]).is_err());
        {
            let recycler = queue.inner.recycler.lock().unwrap();
            assert_eq!(recycler.pools.len(), 1);
            assert_eq!(recycler.fences.len(), 1);
            assert!(!unsafe { device.inner.device.get_fence_status(recycler.fences[0]) }.unwrap());
        }
        queue.inner.submissions.lock().unwrap().last_issued = SubmissionIndex::NONE;
        let index = submit_copies(&queue.inner, &[]).unwrap();
        queue.wait_for_submission(index).unwrap();
        let recycler = queue.inner.recycler.lock().unwrap();
        assert_eq!(recycler.pools.len(), 1);
        assert_eq!(recycler.fences.len(), 1);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_queue_retirement_releases_queue_access() {
        let device = vulkan_device();
        let queue = device.queue();
        let retirement = queue.inner.retire.lock().unwrap();
        let mut threads = Vec::new();
        let mut registered = false;
        let mut access_released = false;
        for target in 1..=2 {
            let inner = Arc::clone(&queue.inner);
            threads.push(std::thread::spawn(move || submit_copies(&inner, &[])));
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            registered = false;
            while std::time::Instant::now() < deadline {
                registered = queue.inner.submissions.lock().unwrap().last_issued.0 >= target;
                access_released = queue.inner.queue_access.try_lock().is_ok();
                if registered && access_released {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if !registered || !access_released {
                break;
            }
        }
        drop(retirement);
        for thread in threads {
            thread.join().unwrap().unwrap();
        }
        assert!(
            registered,
            "a second submit must reach vkQueueSubmit while retirement is locked"
        );
        assert!(access_released, "retirement must not retain queue_access");
        queue.wait_idle().unwrap();
    }

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
        // submit_copies is asynchronous, so drain the copy before reading back.
        device.queue().wait_idle().expect("wait idle");
        assert_eq!(
            destination.read(0, 4).expect("read destination"),
            [1, 2, 3, 4]
        );
    }
}
