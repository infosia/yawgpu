#[cfg(feature = "tiled")]
use crate::HalError;
use crate::{
    HalBuffer, HalComputePipeline, HalExtent3d, HalOrigin3d, HalRenderPipeline, HalTexture,
};
#[cfg(feature = "tiled")]
use crate::{HalTextureFormat, HalTransientAttachment};

/// Stores Noop subpass render pass state.
#[cfg(all(feature = "noop", feature = "tiled"))]
#[derive(Debug, Clone)]
pub struct HalNoopSubpassRenderPass {
    active_subpass: u32,
}

#[cfg(all(feature = "noop", feature = "tiled"))]
impl HalNoopSubpassRenderPass {
    /// Creates a new Noop subpass render pass state.
    #[must_use]
    pub fn new() -> Self {
        Self { active_subpass: 0 }
    }

    /// Advances the active subpass index.
    pub fn next_subpass(&mut self) {
        self.active_subpass = self.active_subpass.saturating_add(1);
    }
}

#[cfg(all(feature = "noop", feature = "tiled"))]
impl Default for HalNoopSubpassRenderPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Enumerates HAL subpass render pass values.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalSubpassRenderPass {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop(HalNoopSubpassRenderPass),
    #[cfg(feature = "vulkan")]
    /// Vulkan placeholder variant.
    Vulkan,
    #[cfg(feature = "metal")]
    /// Metal placeholder variant.
    Metal,
}

#[cfg(feature = "tiled")]
impl HalSubpassRenderPass {
    /// Advances the backend subpass render pass.
    pub fn next_subpass(&mut self) -> Result<(), HalError> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(pass) => {
                pass.next_subpass();
                Ok(())
            }
            #[cfg(feature = "vulkan")]
            Self::Vulkan => Ok(()),
            #[cfg(feature = "metal")]
            Self::Metal => Ok(()),
        }
    }

    /// Ends the backend subpass render pass.
    pub fn end(self) -> Result<(), HalError> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan => Ok(()),
            #[cfg(feature = "metal")]
            Self::Metal => Ok(()),
        }
    }
}

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
    #[cfg(feature = "tiled")]
    /// Subpass render pass variant.
    SubpassRenderPass(HalSubpassRenderPassCommand),
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
    /// Input attachment variant (a subpass-local framebuffer read, wired from the
    /// pass layout's input-source mapping rather than a caller-bound resource).
    #[cfg(feature = "tiled")]
    InputAttachment,
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

/// Stores one subpass draw command for backend execution.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub struct HalSubpassDraw {
    /// Subpass index.
    pub subpass_index: u32,
    /// Pipeline.
    pub pipeline: HalRenderPipeline,
    /// Bind buffers.
    pub bind_buffers: Vec<HalBoundBuffer>,
    /// Vertex buffers.
    pub vertex_buffers: Vec<HalBoundBuffer>,
    /// Draw.
    pub draw: HalDraw,
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

/// Stores one subpass attachment layout.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HalSubpassAttachmentLayout {
    /// Format.
    pub format: HalTextureFormat,
    /// Sample count.
    pub sample_count: u32,
}

/// Stores one subpass input attachment mapping.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HalSubpassInputAttachment {
    /// Bind group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Source subpass.
    pub source_subpass: u32,
    /// Source attachment index, or `u32::MAX` for depth-stencil.
    pub source_attachment: u32,
}

/// Stores one subpass dependency kind.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HalSubpassDependencyType {
    /// Color to input.
    ColorToInput,
    /// Depth to input.
    DepthToInput,
    /// Color and depth to input.
    ColorDepthToInput,
}

/// Stores one subpass dependency.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HalSubpassDependency {
    /// Source subpass.
    pub src_subpass: u32,
    /// Destination subpass.
    pub dst_subpass: u32,
    /// Dependency kind.
    pub dependency_type: HalSubpassDependencyType,
    /// Whether dependency is region-local.
    pub by_region: bool,
}

/// Stores one subpass layout.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct HalSubpassLayout {
    /// Color attachment slot indices.
    pub color_attachment_indices: Vec<u32>,
    /// Whether the subpass uses depth-stencil.
    pub uses_depth_stencil: bool,
    /// Input attachment mappings.
    pub input_attachments: Vec<HalSubpassInputAttachment>,
}

/// Stores a subpass pass layout for backend execution.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct HalSubpassPassLayout {
    /// Color attachment slot layouts.
    pub color_attachments: Vec<HalSubpassAttachmentLayout>,
    /// Optional depth-stencil attachment slot layout.
    pub depth_stencil_attachment: Option<HalSubpassAttachmentLayout>,
    /// Subpass layouts.
    pub subpasses: Vec<HalSubpassLayout>,
    /// Subpass dependencies.
    pub dependencies: Vec<HalSubpassDependency>,
}

/// Enumerates subpass attachment resources.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub enum HalSubpassAttachmentResource {
    /// Persistent texture.
    Persistent {
        /// Texture.
        texture: HalTexture,
        /// Optional resolve target.
        resolve_target: Option<HalTexture>,
    },
    /// Transient attachment.
    Transient(HalTransientAttachment),
}

/// Stores one subpass color attachment binding.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub struct HalSubpassColorAttachment {
    /// Resource.
    pub resource: HalSubpassAttachmentResource,
    /// Load op.
    pub load_op: HalRenderLoadOp,
    /// Store.
    pub store: bool,
    /// Clear color.
    pub clear_color: [f64; 4],
}

/// Stores one subpass depth-stencil attachment binding.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub struct HalSubpassDepthStencilAttachment {
    /// Resource.
    pub resource: HalSubpassAttachmentResource,
    /// Depth load op.
    pub depth_load_op: HalRenderLoadOp,
    /// Depth store.
    pub depth_store: bool,
    /// Depth clear value.
    pub depth_clear_value: f32,
    /// Stencil load op.
    pub stencil_load_op: HalRenderLoadOp,
    /// Stencil store.
    pub stencil_store: bool,
    /// Stencil clear value.
    pub stencil_clear_value: u32,
}

/// Stores subpass render pass command data.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub struct HalSubpassRenderPassCommand {
    /// Layout.
    pub layout: HalSubpassPassLayout,
    /// Extent.
    pub extent: HalExtent3d,
    /// Color attachments by slot.
    pub color_attachments: Vec<HalSubpassColorAttachment>,
    /// Optional depth-stencil attachment.
    pub depth_stencil_attachment: Option<HalSubpassDepthStencilAttachment>,
    /// Draw commands.
    pub draws: Vec<HalSubpassDraw>,
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
