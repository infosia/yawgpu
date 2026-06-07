use super::*;

/// Stores metal queue data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalQueue {
    pub(super) inner: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl std::fmt::Debug for MetalQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalQueue").finish_non_exhaustive()
    }
}

impl MetalQueue {
    /// Creates a new instance.
    pub fn new() -> Result<Self, HalError> {
        Ok(MetalDevice::new()?.queue().clone())
    }

    /// Submits an empty command buffer to flush the queue.
    pub fn submit_empty(&self) -> Result<(), HalError> {
        self.wait_idle()
    }

    /// Waits until all submitted queue work has completed.
    pub fn wait_idle(&self) -> Result<(), HalError> {
        let command_buffer = self
            .inner
            .commandBuffer()
            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        Ok(())
    }

    /// Records and submits the given buffer/texture copy operations.
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
        if copies.is_empty() {
            return Ok(());
        }

        autoreleasepool(|_| {
            let command_buffer = self
                .inner
                .commandBuffer()
                .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
            for copy in copies {
                match copy {
                    HalCopy::Buffer(copy) => {
                        let blit = command_buffer
                            .blitCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        let result = encode_buffer_copy(&blit, copy);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::BufferClear(clear) => {
                        let blit = command_buffer
                            .blitCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        let result = encode_buffer_clear(&blit, clear);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::ResolveQuerySet(resolve) => {
                        let blit = command_buffer
                            .blitCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        let result = encode_resolve_query_set(&blit, resolve);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::BufferToTexture(copy) => {
                        let blit = command_buffer
                            .blitCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        let result = encode_buffer_to_texture(&blit, copy);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::TextureToBuffer(copy) => {
                        let blit = command_buffer
                            .blitCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        let result = encode_texture_to_buffer(&blit, copy);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::TextureToTexture(copy) => {
                        let blit = command_buffer
                            .blitCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        let result = encode_texture_to_texture(&blit, copy);
                        blit.endEncoding();
                        result?;
                    }
                    HalCopy::ComputePass(pass) => {
                        let encoder = command_buffer
                            .computeCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        let result = encode_compute_pass(&encoder, pass);
                        encoder.endEncoding();
                        result?;
                    }
                    HalCopy::RenderPass(pass) => {
                        let descriptor = render_pass_descriptor(pass)?;
                        let encoder = command_buffer
                            .renderCommandEncoderWithDescriptor(&descriptor)
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        let result = encode_render_pass(&encoder, pass);
                        encoder.endEncoding();
                        result?;
                    }
                    #[cfg(feature = "tiled")]
                    HalCopy::SubpassRenderPass(pass) => {
                        let descriptor = subpass_render_pass_descriptor(pass)?;
                        let encoder = command_buffer
                            .renderCommandEncoderWithDescriptor(&descriptor)
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        let result = encode_subpass_render_pass(&encoder, pass);
                        encoder.endEncoding();
                        result?;
                    }
                }
            }
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            Ok(())
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
    fn metal_queue_submit_copies_accepts_buffer_copy() {
        let device = metal_device();
        let source = device.create_buffer(4, HalBufferUsage::default());
        let destination = device.create_buffer(4, HalBufferUsage::default());
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
