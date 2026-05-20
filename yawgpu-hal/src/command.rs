use crate::{
    HalBuffer, HalComputePipeline, HalExtent3d, HalOrigin3d, HalRenderPipeline, HalTexture,
};

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

#[derive(Debug, Clone, Copy)]
pub struct HalDescriptorBinding {
    pub group: u32,
    pub binding: u32,
    pub kind: HalBufferBindingKind,
}

#[derive(Debug, Clone, Copy)]
pub enum HalBufferBindingKind {
    Uniform,
    Storage,
}

#[derive(Debug, Clone)]
pub struct HalBoundBuffer {
    pub group: u32,
    pub binding: u32,
    pub metal_index: u32,
    pub buffer: HalBuffer,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone)]
pub struct HalRenderPass {
    pub pipeline: Option<HalRenderPipeline>,
    pub color_target: HalRenderColorTarget,
    pub bind_buffers: Vec<HalBoundBuffer>,
    pub vertex_buffers: Vec<HalBoundBuffer>,
    pub draw: Option<HalDraw>,
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
