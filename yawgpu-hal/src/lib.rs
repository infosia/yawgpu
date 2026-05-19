#[cfg(feature = "noop")]
pub mod noop;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "vulkan")]
pub mod vulkan;

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
    #[error("HAL shader compilation failed: {backend}: {message}")]
    ShaderCompilationFailed {
        backend: &'static str,
        message: String,
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
            Self::Vulkan(instance) => instance
                .enumerate_adapters()
                .into_iter()
                .map(HalAdapter::Vulkan)
                .collect(),
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
            Self::Vulkan(adapter) => adapter.name().to_owned(),
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
            Self::Vulkan(adapter) => adapter.create_device().map(HalDevice::Vulkan),
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

    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => device.allocation_count(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.allocation_count(),
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
            Self::Vulkan(device) => HalQueue::Vulkan(device.queue().clone()),
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
            Self::Vulkan(device) => HalBuffer::Vulkan(device.create_buffer(size)),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalBuffer::Metal(device.create_buffer(size)),
        }
    }

    #[must_use]
    pub fn create_texture(&self, descriptor: &HalTextureDescriptor) -> HalTexture {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = descriptor;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalTexture::Noop(device.create_texture()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => HalTexture::Vulkan(device.create_texture(descriptor)),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalTexture::Metal(device.create_texture(descriptor)),
        }
    }

    #[must_use]
    pub fn create_sampler(&self, descriptor: &HalSamplerDescriptor) -> HalSampler {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = descriptor;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalSampler::Noop(device.create_sampler()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => HalSampler::Vulkan(device.create_sampler(descriptor)),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalSampler::Metal(device.create_sampler(descriptor)),
        }
    }

    pub fn create_compute_pipeline(
        &self,
        msl_source: &str,
        entry_point: &str,
        workgroup_size: (u32, u32, u32),
    ) -> Result<HalComputePipeline, HalError> {
        #[cfg(not(feature = "metal"))]
        let _ = (msl_source, entry_point, workgroup_size);
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(HalComputePipeline::Noop),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => Ok(HalComputePipeline::Vulkan(vulkan::VulkanComputePipeline)),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device
                .create_compute_pipeline(msl_source, entry_point, workgroup_size)
                .map(HalComputePipeline::Metal),
        }
    }

    pub fn create_render_pipeline(
        &self,
        msl_source: &str,
        vertex_entry_point: &str,
        fragment_entry_point: &str,
        descriptor: &HalRenderPipelineDescriptor,
    ) -> Result<HalRenderPipeline, HalError> {
        #[cfg(not(feature = "metal"))]
        let _ = (
            msl_source,
            vertex_entry_point,
            fragment_entry_point,
            descriptor,
        );
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(HalRenderPipeline::Noop),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => Ok(HalRenderPipeline::Vulkan(vulkan::VulkanRenderPipeline)),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device
                .create_render_pipeline(
                    msl_source,
                    vertex_entry_point,
                    fragment_entry_point,
                    descriptor,
                )
                .map(HalRenderPipeline::Metal),
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
            Self::Vulkan(queue) => queue.submit_empty(),
            #[cfg(feature = "metal")]
            Self::Metal(queue) => queue.submit_empty(),
        }
    }

    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = copies;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(queue) => queue.submit_copies(copies),
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
            Self::Vulkan(buffer) => buffer.size(),
            #[cfg(feature = "metal")]
            Self::Metal(buffer) => buffer.size(),
        }
    }

    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = (offset, data);
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(buffer) => buffer.write(offset, data),
            #[cfg(feature = "metal")]
            Self::Metal(buffer) => buffer.write(offset, data),
        }
    }

    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
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
            Self::Vulkan(buffer) => buffer.read(offset, len),
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
    ComputePass(HalComputePass),
    RenderPass(HalRenderPass),
}

#[derive(Debug, Clone)]
pub struct HalComputePass {
    pub pipeline: HalComputePipeline,
    pub bind_buffers: Vec<HalBoundBuffer>,
    pub workgroups: (u32, u32, u32),
}

#[derive(Debug, Clone)]
pub struct HalBoundBuffer {
    pub metal_index: u32,
    pub buffer: HalBuffer,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub struct HalRenderPass {
    pub pipeline: HalRenderPipeline,
    pub color_target: HalRenderColorTarget,
    pub bind_buffers: Vec<HalBoundBuffer>,
    pub vertex_buffers: Vec<HalBoundBuffer>,
    pub draw: HalDraw,
}

#[derive(Debug, Clone)]
pub struct HalRenderColorTarget {
    pub texture: HalTexture,
    pub load_op: HalRenderLoadOp,
    pub store: bool,
    pub clear_color: [f64; 4],
}

#[derive(Debug, Clone, Copy)]
pub enum HalRenderLoadOp {
    Load,
    Clear,
}

#[derive(Debug, Clone, Copy)]
pub struct HalDraw {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[derive(Debug, Clone)]
pub struct HalRenderPipelineDescriptor {
    pub color_formats: Vec<HalTextureFormat>,
    pub vertex_buffers: Vec<HalVertexBufferLayout>,
    pub primitive_topology: HalPrimitiveTopology,
}

#[derive(Debug, Clone)]
pub struct HalVertexBufferLayout {
    pub array_stride: u64,
    pub step_mode: HalVertexStepMode,
    pub attributes: Vec<HalVertexAttribute>,
}

#[derive(Debug, Clone)]
pub struct HalVertexAttribute {
    pub format: HalVertexFormat,
    pub offset: u64,
    pub shader_location: u32,
    pub metal_buffer_index: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum HalVertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Unsupported,
}

#[derive(Debug, Clone, Copy)]
pub enum HalVertexStepMode {
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy)]
pub enum HalPrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
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

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalComputePipeline {
    #[cfg(feature = "noop")]
    Noop,
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanComputePipeline),
    #[cfg(feature = "metal")]
    Metal(metal::MetalComputePipeline),
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalRenderPipeline {
    #[cfg(feature = "noop")]
    Noop,
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanRenderPipeline),
    #[cfg(feature = "metal")]
    Metal(metal::MetalRenderPipeline),
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
