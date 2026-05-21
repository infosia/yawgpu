use crate::{
    HalBuffer, HalComputePipeline, HalExtent3d, HalOrigin3d, HalRenderPipeline, HalTexture,
};

/// Wraps  for the selected backend.
#[derive(Debug, Clone)]
pub struct HalBufferCopy {
    /// Source.
    pub source: HalBuffer,
    /// Source offset.
    pub source_offset: u64,
    /// Destination.
    pub destination: HalBuffer,
    /// Destination offset.
    pub destination_offset: u64,
    /// Size.
    pub size: u64,
}

/// Enumerates HAL copy values.
#[derive(Debug, Clone)]
pub enum HalCopy {
    /// Buffer variant.
    Buffer(HalBufferCopy),
    /// Buffer to texture variant.
    BufferToTexture(HalBufferTextureCopy),
    /// Texture to buffer variant.
    TextureToBuffer(HalBufferTextureCopy),
    /// Texture to texture variant.
    TextureToTexture(HalTextureCopy),
    /// Compute pass variant.
    ComputePass(HalComputePass),
    /// Render pass variant.
    RenderPass(HalRenderPass),
}

/// Wraps  for the selected backend.
#[derive(Debug, Clone)]
pub struct HalComputePass {
    /// Pipeline.
    pub pipeline: HalComputePipeline,
    /// Bind buffers.
    pub bind_buffers: Vec<HalBoundBuffer>,
    /// Workgroups.
    pub workgroups: (u32, u32, u32),
}

/// Stores binding metadata.
#[derive(Debug, Clone, Copy)]
pub struct HalDescriptorBinding {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Kind.
    pub kind: HalBufferBindingKind,
}

/// Enumerates HAL buffer binding kind values.
#[derive(Debug, Clone, Copy)]
pub enum HalBufferBindingKind {
    /// Uniform variant.
    Uniform,
    /// Storage variant.
    Storage,
}

/// Wraps  for the selected backend.
#[derive(Debug, Clone)]
pub struct HalBoundBuffer {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Metal index.
    pub metal_index: u32,
    /// Buffer.
    pub buffer: HalBuffer,
    /// Offset.
    pub offset: u64,
    /// Size.
    pub size: u64,
}

/// Wraps  for the selected backend.
#[derive(Debug, Clone)]
pub struct HalRenderPass {
    /// Pipeline.
    pub pipeline: Option<HalRenderPipeline>,
    /// Color target.
    pub color_target: HalRenderColorTarget,
    /// Bind buffers.
    pub bind_buffers: Vec<HalBoundBuffer>,
    /// Vertex buffers.
    pub vertex_buffers: Vec<HalBoundBuffer>,
    /// Draw.
    pub draw: Option<HalDraw>,
}

/// Stores color metadata.
#[derive(Debug, Clone)]
pub struct HalRenderColorTarget {
    /// Texture.
    pub texture: HalTexture,
    /// Load op.
    pub load_op: HalRenderLoadOp,
    /// Store.
    pub store: bool,
    /// Clear color.
    pub clear_color: [f64; 4],
}

/// Enumerates HAL render load op values.
#[derive(Debug, Clone, Copy)]
pub enum HalRenderLoadOp {
    /// Load variant.
    Load,
    /// Clear variant.
    Clear,
}

/// Wraps  for the selected backend.
#[derive(Debug, Clone, Copy)]
pub struct HalDraw {
    /// Vertex count.
    pub vertex_count: u32,
    /// Instance count.
    pub instance_count: u32,
    /// First vertex.
    pub first_vertex: u32,
    /// First instance.
    pub first_instance: u32,
}

/// Stores layout metadata.
#[derive(Debug, Clone, Copy)]
pub struct HalBufferTextureLayout {
    /// Offset.
    pub offset: u64,
    /// Bytes per row.
    pub bytes_per_row: u32,
    /// Rows per image.
    pub rows_per_image: u32,
}

/// Wraps  for the selected backend.
#[derive(Debug, Clone)]
pub struct HalBufferTextureCopy {
    /// Buffer.
    pub buffer: HalBuffer,
    /// Buffer layout.
    pub buffer_layout: HalBufferTextureLayout,
    /// Texture.
    pub texture: HalTexture,
    /// Mip level.
    pub mip_level: u32,
    /// Origin.
    pub origin: HalOrigin3d,
    /// Extent.
    pub extent: HalExtent3d,
}

/// Wraps  for the selected backend.
#[derive(Debug, Clone)]
pub struct HalTextureCopy {
    /// Source.
    pub source: HalTexture,
    /// Source mip level.
    pub source_mip_level: u32,
    /// Source origin.
    pub source_origin: HalOrigin3d,
    /// Destination.
    pub destination: HalTexture,
    /// Destination mip level.
    pub destination_mip_level: u32,
    /// Destination origin.
    pub destination_origin: HalOrigin3d,
    /// Extent.
    pub extent: HalExtent3d,
}
