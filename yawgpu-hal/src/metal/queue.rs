use super::*;

/// Stores metal queue data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalQueue {
    pub(super) inner: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(super) last_submission_index: Arc<AtomicU64>,
    pub(super) completed_submission_index: Arc<AtomicU64>,
    pub(super) submission_lock: Arc<Mutex<()>>,
}

impl std::fmt::Debug for MetalQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalQueue").finish_non_exhaustive()
    }
}

fn queue_submission_error(message: &'static str) -> HalError {
    HalError::QueueSubmissionFailed {
        backend: BACKEND,
        message: message.to_string(),
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
            let _submission = self
                .submission_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let index = crate::next_submission_index(&self.last_submission_index, BACKEND)?;
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            self.completed_submission_index
                .store(index.0, Ordering::Release);
            Ok(index)
        })
    }

    /// Returns the highest submission index proven complete without blocking.
    pub fn completed_submission_index(&self) -> Result<SubmissionIndex, HalError> {
        Ok(SubmissionIndex(
            self.completed_submission_index.load(Ordering::Acquire),
        ))
    }

    /// Blocks until the requested submission index has completed.
    pub fn wait_for_submission(&self, index: SubmissionIndex) -> Result<(), HalError> {
        if index <= self.completed_submission_index()? {
            Ok(())
        } else {
            Err(queue_submission_error(
                "submission index has not been issued",
            ))
        }
    }

    /// Waits until all submitted queue work has completed.
    pub fn wait_idle(&self) -> Result<(), HalError> {
        autoreleasepool(|_| {
            let command_buffer = self.inner.commandBuffer().ok_or_else(|| {
                queue_submission_error("wait-idle command buffer creation returned nil")
            })?;
            let _submission = self
                .submission_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            Ok(())
        })
    }

    /// Records and submits the given buffer/texture copy operations.
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<SubmissionIndex, HalError> {
        if copies.is_empty() {
            let _submission = self
                .submission_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let index = crate::next_submission_index(&self.last_submission_index, BACKEND)?;
            self.completed_submission_index
                .store(index.0, Ordering::Release);
            return Ok(index);
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
                    HalCopy::RenderPass(pass) => {
                        let descriptor = render_pass_descriptor(pass)?;
                        let encoder = command_buffer
                            .renderCommandEncoderWithDescriptor(&descriptor)
                            .ok_or_else(|| {
                                queue_submission_error(
                                    "render command encoder creation returned nil",
                                )
                            })?;
                        let result = encode_render_pass(&encoder, pass);
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
            let _submission = self
                .submission_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let index = crate::next_submission_index(&self.last_submission_index, BACKEND)?;
            command_buffer.commit();
            // Keep B1 behavior unchanged: Metal still blocks here. B2 moves
            // completion tracking to a command-buffer completion handler.
            command_buffer.waitUntilCompleted();
            self.completed_submission_index
                .store(index.0, Ordering::Release);
            Ok(index)
        })
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
    fn metal_queue_submit_empty_completes() {
        metal_device()
            .queue()
            .submit_empty()
            .expect("submit empty queue work");
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_queue_completion_index_tracks_synchronous_submissions() {
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
        assert_eq!(
            queue
                .completed_submission_index()
                .expect("query completed submission"),
            second
        );
        queue
            .wait_for_submission(second)
            .expect("wait for completed submission");
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
        device
            .queue()
            .submit_copies(&[HalCopy::Buffer(HalBufferCopy {
                source: HalBuffer::Metal(source),
                source_offset: 0,
                destination: HalBuffer::Metal(destination.clone()),
                destination_offset: 0,
                size: 4,
            })])
            .expect("submit buffer copy");
        assert_eq!(
            destination.read(0, 4).expect("read destination"),
            [1, 2, 3, 4]
        );
    }
}
