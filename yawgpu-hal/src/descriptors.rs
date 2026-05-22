use crate::{
    HalAddressMode, HalCompareFunction, HalFilterMode, HalMipmapFilterMode, HalPrimitiveTopology,
    HalTextureFormat, HalTextureUsage, HalVertexFormat, HalVertexStepMode,
};

/// Describes HAL render pipeline descriptor.
#[derive(Debug, Clone)]
pub struct HalRenderPipelineDescriptor {
    /// Color formats.
    pub color_formats: Vec<HalTextureFormat>,
    /// Vertex buffers.
    pub vertex_buffers: Vec<HalVertexBufferLayout>,
    /// Primitive topology.
    pub primitive_topology: HalPrimitiveTopology,
}

/// Stores layout metadata.
#[derive(Debug, Clone)]
pub struct HalVertexBufferLayout {
    /// Array stride.
    pub array_stride: u64,
    /// Step mode.
    pub step_mode: HalVertexStepMode,
    /// Attributes.
    pub attributes: Vec<HalVertexAttribute>,
}

/// Stores attribute metadata.
#[derive(Debug, Clone)]
pub struct HalVertexAttribute {
    /// Format.
    pub format: HalVertexFormat,
    /// Offset.
    pub offset: u64,
    /// Shader location.
    pub shader_location: u32,
    /// Metal buffer index.
    pub metal_buffer_index: u32,
}

/// Stores origin metadata.
#[derive(Debug, Clone, Copy)]
pub struct HalOrigin3d {
    /// X.
    pub x: u32,
    /// Y.
    pub y: u32,
    /// Z.
    pub z: u32,
}

/// Stores extent metadata.
#[derive(Debug, Clone, Copy)]
pub struct HalExtent3d {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Depth or array layers.
    pub depth_or_array_layers: u32,
}

/// Describes HAL texture descriptor.
#[derive(Debug, Clone, Copy)]
pub struct HalTextureDescriptor {
    /// Format.
    pub format: HalTextureFormat,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Depth or array layers.
    pub depth_or_array_layers: u32,
    /// Mip level count.
    pub mip_level_count: u32,
    /// Sample count.
    pub sample_count: u32,
    /// Usage.
    pub usage: HalTextureUsage,
}

/// Describes HAL transient attachment descriptor.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone, Copy)]
pub struct HalTransientAttachmentDescriptor {
    /// Format.
    pub format: HalTextureFormat,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Sample count.
    pub sample_count: u32,
}

/// Describes HAL sampler descriptor.
#[derive(Debug, Clone, Copy)]
pub struct HalSamplerDescriptor {
    /// Address mode u.
    pub address_mode_u: HalAddressMode,
    /// Address mode v.
    pub address_mode_v: HalAddressMode,
    /// Address mode w.
    pub address_mode_w: HalAddressMode,
    /// Mag filter.
    pub mag_filter: HalFilterMode,
    /// Min filter.
    pub min_filter: HalFilterMode,
    /// Mipmap filter.
    pub mipmap_filter: HalMipmapFilterMode,
    /// Lod min clamp.
    pub lod_min_clamp: f32,
    /// Lod max clamp.
    pub lod_max_clamp: f32,
    /// Compare.
    pub compare: Option<HalCompareFunction>,
    /// Max anisotropy.
    pub max_anisotropy: u16,
}
