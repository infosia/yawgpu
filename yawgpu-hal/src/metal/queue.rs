use super::*;

/// Stores metal queue data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalQueue {
    pub(super) submissions: Arc<Mutex<MetalSubmissionTracker>>,
    pub(super) inner: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(super) submission_lock: Arc<Mutex<()>>,
}

struct MetalTrackedSubmission {
    index: SubmissionIndex,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    _retained_copies: Vec<HalCopy>,
}

// SAFETY: Entries are published only after encoding has validated that every
// retained HalCopy resource is a thread-safe Metal wrapper, and only after the
// command buffer is committed. From then on yawgpu calls only Metal's
// synchronization-safe status, error, and wait methods on the command buffer.
unsafe impl Send for MetalTrackedSubmission {}

/// Tracks committed Metal command buffers and their Rust resource owners.
pub(super) struct MetalSubmissionTracker {
    last_issued: SubmissionIndex,
    completed: SubmissionIndex,
    command_buffers: VecDeque<MetalTrackedSubmission>,
    first_error: Option<(SubmissionIndex, String)>,
}

impl MetalSubmissionTracker {
    /// Creates an empty Metal submission timeline.
    pub(super) fn new() -> Self {
        Self {
            last_issued: SubmissionIndex::NONE,
            completed: SubmissionIndex::NONE,
            command_buffers: VecDeque::new(),
            first_error: None,
        }
    }

    fn reserve(&mut self) -> Result<SubmissionIndex, HalError> {
        let next = self
            .last_issued
            .0
            .checked_add(1)
            .map(SubmissionIndex)
            .ok_or_else(|| queue_submission_error("submission index exhausted"))?;
        self.last_issued = next;
        Ok(next)
    }

    fn register(
        &mut self,
        index: SubmissionIndex,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        retained_copies: Vec<HalCopy>,
    ) {
        self.command_buffers.push_back(MetalTrackedSubmission {
            index,
            command_buffer,
            _retained_copies: retained_copies,
        });
    }

    fn retire_terminal_prefix(&mut self) {
        while let Some(submission) = self.command_buffers.front() {
            let status = submission.command_buffer.status();
            if !command_buffer_status_is_terminal(status) {
                break;
            }
            let Some(submission) = self.command_buffers.pop_front() else {
                break;
            };
            self.completed = submission.index;
            if status == MTLCommandBufferStatus::Error && self.first_error.is_none() {
                let message = submission
                    .command_buffer
                    .error()
                    .map(|error| error.localizedDescription().to_string())
                    .unwrap_or_else(|| "Metal command buffer execution failed".to_owned());
                self.first_error = Some((submission.index, message));
            }
            // Dropping this entry releases both the retained command buffer
            // and every Rust HAL resource cloned from its HalCopy list.
            drop(submission);
        }
    }

    fn result_through(&self, index: SubmissionIndex) -> Result<(), HalError> {
        if let Some((failed_index, message)) = &self.first_error {
            if *failed_index <= index {
                return Err(queue_submission_error(format!(
                    "submission {} failed: {message}",
                    failed_index.0
                )));
            }
        }
        Ok(())
    }
}

impl Drop for MetalSubmissionTracker {
    fn drop(&mut self) {
        autoreleasepool(|_| {
            if let Some(submission) = self.command_buffers.back() {
                submission.command_buffer.waitUntilCompleted();
            }
            self.retire_terminal_prefix();
        });
    }
}

fn command_buffer_status_is_terminal(status: MTLCommandBufferStatus) -> bool {
    matches!(
        status,
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    )
}

impl std::fmt::Debug for MetalQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalQueue").finish_non_exhaustive()
    }
}

fn queue_submission_error(message: impl Into<String>) -> HalError {
    HalError::QueueSubmissionFailed {
        backend: BACKEND,
        message: message.into(),
    }
}

impl MetalQueue {
    /// Creates a new instance.
    pub fn new() -> Result<Self, HalError> {
        Ok(MetalDevice::new()?.queue().clone())
    }

    /// Submits an empty command buffer to flush the queue.
    pub fn submit_empty(&self) -> Result<SubmissionIndex, HalError> {
        autoreleasepool(|_| {
            let command_buffer = self.inner.commandBuffer().ok_or_else(|| {
                queue_submission_error("submit-empty command buffer creation returned nil")
            })?;
            self.commit_tracked(command_buffer, Vec::new())
        })
    }

    /// Returns the highest submission index proven complete without blocking.
    pub fn completed_submission_index(&self) -> Result<SubmissionIndex, HalError> {
        autoreleasepool(|_| {
            let mut submissions = self
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Metal command buffers committed to one queue finish in commit
            // order, so the first unfinished entry bounds the completed prefix.
            submissions.retire_terminal_prefix();
            let completed = submissions.completed;
            submissions.result_through(completed)?;
            Ok(completed)
        })
    }

    /// Blocks until the requested submission index has completed.
    pub fn wait_for_submission(&self, index: SubmissionIndex) -> Result<(), HalError> {
        let command_buffer = {
            let submissions = self
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if index <= submissions.completed {
                return submissions.result_through(index);
            }
            if index > submissions.last_issued {
                return Err(queue_submission_error(
                    "submission index has not been issued",
                ));
            }
            submissions
                .command_buffers
                .iter()
                .find_map(|submission| {
                    (submission.index == index).then(|| submission.command_buffer.clone())
                })
                .ok_or_else(|| queue_submission_error("submission command buffer is unavailable"))?
        };
        autoreleasepool(|_| {
            command_buffer.waitUntilCompleted();
            let mut submissions = self
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            submissions.retire_terminal_prefix();
            if submissions.completed < index {
                return Err(queue_submission_error(
                    "submission wait did not complete the queue timeline",
                ));
            }
            submissions.result_through(index)
        })
    }

    /// Waits until all submitted queue work has completed.
    pub fn wait_idle(&self) -> Result<(), HalError> {
        let _submission = self
            .submission_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let last_issued = self
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .last_issued;
        self.wait_for_submission(last_issued)
    }

    /// Records and submits the given buffer/texture copy operations.
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<SubmissionIndex, HalError> {
        if copies.is_empty() {
            return self.submit_empty();
        }

        autoreleasepool(|_| {
            let command_buffer = self.inner.commandBuffer().ok_or_else(|| {
                queue_submission_error("submit command buffer creation returned nil")
            })?;
            for copy in copies {
                match copy {
                    HalCopy::Buffer(copy) => {
                        let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                            queue_submission_error("buffer-copy blit encoder creation returned nil")
                        })?;
                        let result = encode_buffer_copy(&blit, copy);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::BufferClear(clear) => {
                        let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                            queue_submission_error(
                                "buffer-clear blit encoder creation returned nil",
                            )
                        })?;
                        let result = encode_buffer_clear(&blit, clear);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::ClearTexture(clear) => {
                        let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                            queue_submission_error(
                                "texture-clear blit encoder creation returned nil",
                            )
                        })?;
                        let result = encode_texture_clear(&blit, clear);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::ResolveQuerySet(resolve) => {
                        let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                            queue_submission_error(
                                "query-resolve blit encoder creation returned nil",
                            )
                        })?;
                        let result = encode_resolve_query_set(&blit, resolve);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::BufferToTexture(copy) => {
                        let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                            queue_submission_error(
                                "buffer-to-texture blit encoder creation returned nil",
                            )
                        })?;
                        let result = encode_buffer_to_texture(&blit, copy);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::TextureToBuffer(copy) => {
                        let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                            queue_submission_error(
                                "texture-to-buffer blit encoder creation returned nil",
                            )
                        })?;
                        let result = encode_texture_to_buffer(&blit, copy);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::TextureToTexture(copy) => {
                        let blit = command_buffer.blitCommandEncoder().ok_or_else(|| {
                            queue_submission_error(
                                "texture-to-texture blit encoder creation returned nil",
                            )
                        })?;
                        let result = encode_texture_to_texture(&blit, copy);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::ComputePass(pass) => {
                        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
                            queue_submission_error("compute command encoder creation returned nil")
                        })?;
                        let result = encode_compute_pass(&encoder, pass);
                        encoder.endEncoding();
                        result?;
                    }
                    HalCopy::RenderPassCommandStream(pass) => {
                        let descriptor = render_pass_command_stream_descriptor(pass)?;
                        let encoder = command_buffer
                            .renderCommandEncoderWithDescriptor(&descriptor)
                            .ok_or_else(|| {
                                queue_submission_error(
                                    "render command encoder creation returned nil",
                                )
                            })?;
                        let result = encode_render_pass_command_stream(&encoder, pass);
                        encoder.endEncoding();
                        result?;
                    }
                    #[cfg(feature = "tiled")]
                    HalCopy::SubpassRenderPass(pass) => {
                        let descriptor = subpass_render_pass_descriptor(pass)?;
                        let encoder = command_buffer
                            .renderCommandEncoderWithDescriptor(&descriptor)
                            .ok_or_else(|| {
                                queue_submission_error(
                                    "subpass render command encoder creation returned nil",
                                )
                            })?;
                        let result = encode_subpass_render_pass(&encoder, pass);
                        encoder.endEncoding();
                        result?;
                    }
                }
            }
            self.commit_tracked(command_buffer, copies.to_vec())
        })
    }

    fn commit_tracked(
        &self,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        retained_copies: Vec<HalCopy>,
    ) -> Result<SubmissionIndex, HalError> {
        let _submission = self
            .submission_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut submissions = self
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let index = submissions.reserve()?;
        command_buffer.commit();
        submissions.register(index, command_buffer, retained_copies);
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;
    use crate::HalBufferCopy;

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_new_constructs_queue() {
        MetalQueue::new().expect("create Metal queue");
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_submit_empty_is_tracked_until_waited() {
        let device = metal_device();
        let queue = device.queue();
        let submitted = queue.submit_empty().expect("submit empty queue work");
        assert_eq!(
            queue
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .command_buffers
                .len(),
            1
        );
        queue
            .wait_for_submission(submitted)
            .expect("wait for empty queue work");
        assert!(queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .command_buffers
            .is_empty());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_completion_poll_advances_and_evicts_finished_submissions() {
        let device = metal_device();
        let queue = device.queue();

        assert_eq!(
            queue
                .completed_submission_index()
                .expect("query initial completion"),
            SubmissionIndex::NONE
        );
        let first = queue.submit_empty().expect("submit first empty work");
        let second = queue.submit_empty().expect("submit second empty work");

        assert!(first < second);
        let second_command_buffer = queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .command_buffers
            .back()
            .map(|submission| submission.command_buffer.clone())
            .expect("second command buffer is tracked");
        autoreleasepool(|_| second_command_buffer.waitUntilCompleted());
        assert_eq!(
            queue
                .completed_submission_index()
                .expect("query completed submission"),
            second
        );
        assert!(queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .command_buffers
            .is_empty());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_wait_for_submission_advances_and_evicts_in_commit_order() {
        let device = metal_device();
        let queue = device.queue();
        let first = queue.submit_empty().expect("submit first empty work");
        let second = queue.submit_empty().expect("submit second empty work");

        queue
            .wait_for_submission(second)
            .expect("wait for second submission");
        let submissions = queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(first < second);
        assert_eq!(submissions.completed, second);
        assert!(submissions.command_buffers.is_empty());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_wait_idle_waits_for_and_evicts_all_submissions() {
        let device = metal_device();
        let queue = device.queue();
        let submitted = queue.submit_empty().expect("submit empty work");

        queue.wait_idle().expect("wait for queue idle");
        let submissions = queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(submissions.completed, submitted);
        assert!(submissions.command_buffers.is_empty());
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_command_buffer_error_status_is_terminal() {
        assert!(command_buffer_status_is_terminal(
            MTLCommandBufferStatus::Completed
        ));
        assert!(command_buffer_status_is_terminal(
            MTLCommandBufferStatus::Error
        ));
        assert!(!command_buffer_status_is_terminal(
            MTLCommandBufferStatus::Scheduled
        ));
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_submission_tracker_starts_empty() {
        let tracker = MetalSubmissionTracker::new();
        assert_eq!(tracker.last_issued, SubmissionIndex::NONE);
        assert_eq!(tracker.completed, SubmissionIndex::NONE);
        assert!(tracker.command_buffers.is_empty());
        assert!(tracker.first_error.is_none());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_submit_copies_accepts_buffer_copy() {
        let device = metal_device();
        let source = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("Metal source buffer allocation should succeed");
        let destination = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("Metal destination buffer allocation should succeed");
        source.write(0, &[1, 2, 3, 4]).expect("write source");
        let submitted = device
            .queue()
            .submit_copies(&[HalCopy::Buffer(HalBufferCopy {
                source: HalBuffer::Metal(source),
                source_offset: 0,
                destination: HalBuffer::Metal(destination.clone()),
                destination_offset: 0,
                size: 4,
            })])
            .expect("submit buffer copy");
        device
            .queue()
            .wait_for_submission(submitted)
            .expect("wait for buffer copy");
        assert_eq!(
            destination.read(0, 4).expect("read destination"),
            [1, 2, 3, 4]
        );
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_submit_copies_retains_hal_copies_until_completion() {
        let device = metal_device();
        let queue = device.queue();
        let source = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("create source buffer");
        let destination = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("create destination buffer");
        source.write(0, &[1, 2, 3, 4]).expect("write source");
        let copies = vec![HalCopy::Buffer(HalBufferCopy {
            source: HalBuffer::Metal(source),
            source_offset: 0,
            destination: HalBuffer::Metal(destination.clone()),
            destination_offset: 0,
            size: 4,
        })];

        let submitted = queue.submit_copies(&copies).expect("submit buffer copy");
        drop(copies);
        assert_eq!(
            queue
                .submissions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .command_buffers
                .front()
                .map(|submission| submission._retained_copies.len()),
            Some(1)
        );
        queue
            .wait_for_submission(submitted)
            .expect("wait for retained copy");
        assert!(queue
            .submissions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .command_buffers
            .is_empty());
        assert_eq!(
            destination.read(0, 4).expect("read destination"),
            [1, 2, 3, 4]
        );
    }
}
