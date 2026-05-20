use crate::{
    HalAddressMode, HalCompareFunction, HalFilterMode, HalMipmapFilterMode, HalPrimitiveTopology,
    HalTextureFormat, HalTextureUsage, HalVertexFormat, HalVertexStepMode,
};

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
