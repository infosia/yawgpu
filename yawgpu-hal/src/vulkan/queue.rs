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
}

impl VulkanQueue {
    /// Submits an empty command buffer to flush the queue.
    pub fn submit_empty(&self) -> Result<(), HalError> {
        unsafe {
            self.inner
                .device
                .device
                .queue_submit(self.inner.queue, &[], vk::Fence::null())
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
            self.inner
                .device
                .device
                .queue_wait_idle(self.inner.queue)
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
        }
        Ok(())
    }

    /// Records and submits the given buffer/texture copy operations.
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
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
    fn vulkan_queue_submit_copies_accepts_buffer_copy() {
        let device = vulkan_device();
        let source = device.create_buffer(4);
        let destination = device.create_buffer(4);
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
        assert_eq!(
            destination.read(0, 4).expect("read destination"),
            [1, 2, 3, 4]
        );
    }
}
