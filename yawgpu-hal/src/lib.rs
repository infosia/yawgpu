#[cfg(feature = "noop")]
pub mod noop;

#[cfg(feature = "metal")]
mod metal {
    #[derive(Debug)]
    pub struct MetalInstance;
    #[derive(Debug)]
    pub struct MetalAdapter;
    #[derive(Debug)]
    pub struct MetalDevice;
    #[derive(Debug)]
    pub struct MetalQueue;
    #[derive(Debug)]
    pub struct MetalBuffer;
    #[derive(Debug)]
    pub struct MetalTexture;
}

#[cfg(feature = "vulkan")]
mod vulkan {
    #[derive(Debug)]
    pub struct VulkanInstance;
    #[derive(Debug)]
    pub struct VulkanAdapter;
    #[derive(Debug)]
    pub struct VulkanDevice;
    #[derive(Debug)]
    pub struct VulkanQueue;
    #[derive(Debug)]
    pub struct VulkanBuffer;
    #[derive(Debug)]
    pub struct VulkanTexture;
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HalError {
    #[error("HAL backend is unavailable: {backend}")]
    BackendUnavailable { backend: &'static str },
}

#[derive(Debug)]
#[non_exhaustive]
pub enum HalInstance {
    #[cfg(feature = "noop")]
    Noop(noop::NoopInstance),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanInstance),
    #[cfg(feature = "metal")]
    Metal(metal::MetalInstance),
}

impl HalInstance {
    #[cfg(feature = "noop")]
    #[must_use]
    pub fn new_noop() -> Self {
        Self::Noop(noop::NoopInstance::new())
    }

    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<HalAdapter> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(instance) => instance
                .enumerate_adapters()
                .into_iter()
                .map(HalAdapter::Noop)
                .collect(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => Vec::new(),
            #[cfg(feature = "metal")]
            Self::Metal(_) => Vec::new(),
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum HalAdapter {
    #[cfg(feature = "noop")]
    Noop(noop::NoopAdapter),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanAdapter),
    #[cfg(feature = "metal")]
    Metal(metal::MetalAdapter),
}

impl HalAdapter {
    pub fn create_device(&self) -> Result<HalDevice, HalError> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.create_device().map(HalDevice::Noop),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => Err(HalError::BackendUnavailable { backend: "vulkan" }),
            #[cfg(feature = "metal")]
            Self::Metal(_) => Err(HalError::BackendUnavailable { backend: "metal" }),
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum HalDevice {
    #[cfg(feature = "noop")]
    Noop(noop::NoopDevice),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanDevice),
    #[cfg(feature = "metal")]
    Metal(metal::MetalDevice),
}

impl HalDevice {
    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => device.allocation_count(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => 0,
            #[cfg(feature = "metal")]
            Self::Metal(_) => 0,
        }
    }

    #[must_use]
    pub fn queue(&self) -> HalQueue {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalQueue::Noop(device.queue().clone()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => HalQueue::Vulkan(vulkan::VulkanQueue),
            #[cfg(feature = "metal")]
            Self::Metal(_) => HalQueue::Metal(metal::MetalQueue),
        }
    }

    #[must_use]
    pub fn create_buffer(&self, size: u64) -> HalBuffer {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalBuffer::Noop(device.create_buffer(size)),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => HalBuffer::Vulkan(vulkan::VulkanBuffer),
            #[cfg(feature = "metal")]
            Self::Metal(_) => HalBuffer::Metal(metal::MetalBuffer),
        }
    }

    #[must_use]
    pub fn create_texture(&self) -> HalTexture {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalTexture::Noop(device.create_texture()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => HalTexture::Vulkan(vulkan::VulkanTexture),
            #[cfg(feature = "metal")]
            Self::Metal(_) => HalTexture::Metal(metal::MetalTexture),
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum HalQueue {
    #[cfg(feature = "noop")]
    Noop(noop::NoopQueue),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanQueue),
    #[cfg(feature = "metal")]
    Metal(metal::MetalQueue),
}

#[derive(Debug)]
#[non_exhaustive]
pub enum HalBuffer {
    #[cfg(feature = "noop")]
    Noop(noop::NoopBuffer),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanBuffer),
    #[cfg(feature = "metal")]
    Metal(metal::MetalBuffer),
}

#[derive(Debug)]
#[non_exhaustive]
pub enum HalTexture {
    #[cfg(feature = "noop")]
    Noop(noop::NoopTexture),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanTexture),
    #[cfg(feature = "metal")]
    Metal(metal::MetalTexture),
}

#[cfg(test)]
mod tests {
    use super::{HalError, HalInstance};

    #[test]
    fn noop_creates_device_with_zero_allocations() -> Result<(), HalError> {
        let instance = HalInstance::new_noop();
        let adapters = instance.enumerate_adapters();
        assert_eq!(adapters.len(), 1);

        let device = adapters[0].create_device()?;
        assert_eq!(device.allocation_count(), 0);

        Ok(())
    }
}
