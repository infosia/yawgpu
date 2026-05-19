#[cfg(feature = "noop")]
pub mod noop;

#[cfg(feature = "metal")]
pub mod metal;

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
    #[derive(Debug, Clone)]
    pub struct VulkanBuffer;
    #[derive(Debug, Clone)]
    pub struct VulkanTexture;
    #[derive(Debug, Clone)]
    pub struct VulkanSampler;
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HalError {
    #[error("HAL backend is unavailable: {backend}")]
    BackendUnavailable { backend: &'static str },
    #[error("HAL device creation failed: {backend}")]
    DeviceCreationFailed { backend: &'static str },
    #[error("HAL queue submission failed: {backend}")]
    QueueSubmissionFailed { backend: &'static str },
    #[error("HAL buffer operation failed: {backend}: {message}")]
    BufferOperationFailed {
        backend: &'static str,
        message: &'static str,
    },
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
            Self::Metal(instance) => instance
                .enumerate_adapters()
                .into_iter()
                .map(HalAdapter::Metal)
                .collect(),
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
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.name().to_owned(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => "yawgpu Vulkan Adapter".to_owned(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.name().to_owned(),
        }
    }

    #[must_use]
    pub fn backend(&self) -> HalBackend {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => HalBackend::Noop,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => HalBackend::Vulkan,
            #[cfg(feature = "metal")]
            Self::Metal(_) => HalBackend::Metal,
        }
    }

    pub fn create_device(&self) -> Result<HalDevice, HalError> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.create_device().map(HalDevice::Noop),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => Err(HalError::BackendUnavailable { backend: "vulkan" }),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.create_device().map(HalDevice::Metal),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalBackend {
    Noop,
    Vulkan,
    Metal,
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
            Self::Metal(device) => device.allocation_count(),
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
            Self::Metal(device) => HalQueue::Metal(device.queue().clone()),
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
            Self::Metal(device) => HalBuffer::Metal(device.create_buffer(size)),
        }
    }

    #[must_use]
    pub fn create_texture(&self, descriptor: &HalTextureDescriptor) -> HalTexture {
        #[cfg(not(feature = "metal"))]
        let _ = descriptor;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalTexture::Noop(device.create_texture()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => HalTexture::Vulkan(vulkan::VulkanTexture),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalTexture::Metal(device.create_texture(descriptor)),
        }
    }

    #[must_use]
    pub fn create_sampler(&self, descriptor: &HalSamplerDescriptor) -> HalSampler {
        #[cfg(not(feature = "metal"))]
        let _ = descriptor;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalSampler::Noop(device.create_sampler()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => HalSampler::Vulkan(vulkan::VulkanSampler),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalSampler::Metal(device.create_sampler(descriptor)),
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

impl HalQueue {
    pub fn submit_empty(&self) -> Result<(), HalError> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => Ok(()),
            #[cfg(feature = "metal")]
            Self::Metal(queue) => queue.submit_empty(),
        }
    }

    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
        #[cfg(not(feature = "metal"))]
        let _ = copies;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => Ok(()),
            #[cfg(feature = "metal")]
            Self::Metal(queue) => queue.submit_copies(copies),
        }
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalBuffer {
    #[cfg(feature = "noop")]
    Noop(noop::NoopBuffer),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanBuffer),
    #[cfg(feature = "metal")]
    Metal(metal::MetalBuffer),
}

impl HalBuffer {
    #[must_use]
    pub fn size(&self) -> u64 {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(buffer) => buffer.size(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => 0,
            #[cfg(feature = "metal")]
            Self::Metal(buffer) => buffer.size(),
        }
    }

    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), HalError> {
        #[cfg(not(feature = "metal"))]
        let _ = (offset, data);
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => Ok(()),
            #[cfg(feature = "metal")]
            Self::Metal(buffer) => buffer.write(offset, data),
        }
    }

    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        #[cfg(not(feature = "metal"))]
        let _ = offset;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => usize::try_from(len).map_or_else(
                |_| {
                    Err(HalError::BufferOperationFailed {
                        backend: "noop",
                        message: "read length is too large",
                    })
                },
                |len| Ok(vec![0; len]),
            ),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => usize::try_from(len).map_or_else(
                |_| {
                    Err(HalError::BufferOperationFailed {
                        backend: "vulkan",
                        message: "read length is too large",
                    })
                },
                |len| Ok(vec![0; len]),
            ),
            #[cfg(feature = "metal")]
            Self::Metal(buffer) => buffer.read(offset, len),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HalBufferCopy {
    pub source: HalBuffer,
    pub source_offset: u64,
    pub destination: HalBuffer,
    pub destination_offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone)]
pub enum HalCopy {
    Buffer(HalBufferCopy),
    BufferToTexture(HalBufferTextureCopy),
    TextureToBuffer(HalBufferTextureCopy),
    TextureToTexture(HalTextureCopy),
}

#[derive(Debug, Clone, Copy)]
pub struct HalBufferTextureLayout {
    pub offset: u64,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
}

#[derive(Debug, Clone)]
pub struct HalBufferTextureCopy {
    pub buffer: HalBuffer,
    pub buffer_layout: HalBufferTextureLayout,
    pub texture: HalTexture,
    pub mip_level: u32,
    pub origin: HalOrigin3d,
    pub extent: HalExtent3d,
}

#[derive(Debug, Clone)]
pub struct HalTextureCopy {
    pub source: HalTexture,
    pub source_mip_level: u32,
    pub source_origin: HalOrigin3d,
    pub destination: HalTexture,
    pub destination_mip_level: u32,
    pub destination_origin: HalOrigin3d,
    pub extent: HalExtent3d,
}

#[derive(Debug, Clone, Copy)]
pub struct HalOrigin3d {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct HalExtent3d {
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct HalTextureDescriptor {
    pub format: HalTextureFormat,
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub usage: HalTextureUsage,
}

#[derive(Debug, Clone, Copy)]
pub enum HalTextureFormat {
    R8Unorm,
    Rgba8Unorm,
    Bgra8Unorm,
    Unsupported,
}

#[derive(Debug, Clone, Copy)]
pub struct HalTextureUsage {
    pub copy_src: bool,
    pub copy_dst: bool,
    pub texture_binding: bool,
    pub storage_binding: bool,
    pub render_attachment: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct HalSamplerDescriptor {
    pub address_mode_u: HalAddressMode,
    pub address_mode_v: HalAddressMode,
    pub address_mode_w: HalAddressMode,
    pub mag_filter: HalFilterMode,
    pub min_filter: HalFilterMode,
    pub mipmap_filter: HalMipmapFilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare: Option<HalCompareFunction>,
    pub max_anisotropy: u16,
}

#[derive(Debug, Clone, Copy)]
pub enum HalAddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

#[derive(Debug, Clone, Copy)]
pub enum HalFilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy)]
pub enum HalMipmapFilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy)]
pub enum HalCompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalTexture {
    #[cfg(feature = "noop")]
    Noop(noop::NoopTexture),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanTexture),
    #[cfg(feature = "metal")]
    Metal(metal::MetalTexture),
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalSampler {
    #[cfg(feature = "noop")]
    Noop(noop::NoopSampler),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanSampler),
    #[cfg(feature = "metal")]
    Metal(metal::MetalSampler),
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
