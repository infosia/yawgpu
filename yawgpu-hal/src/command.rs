#[cfg(feature = "tiled")]
use crate::HalError;
use crate::HalQuerySet;
use crate::{
    HalBuffer, HalComputePipeline, HalExtent3d, HalOrigin3d, HalRenderPipeline, HalSampler,
    HalTexture, HalTextureComponentSwizzle, HalTextureFormat,
};

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

    /// Returns the active subpass index.
    #[must_use]
    pub fn active_subpass(&self) -> u32 {
        self.active_subpass
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

/// Wraps buffer clear data for the selected backend.
#[derive(Debug, Clone)]
pub struct HalBufferClear {
    /// Buffer.
    pub buffer: HalBuffer,
    /// Offset.
    pub offset: u64,
    /// Size.
    pub size: u64,
}

/// Wraps texture clear data for the selected backend.
#[derive(Debug, Clone)]
pub struct HalTextureClear {
    /// Texture.
    pub texture: HalTexture,
    /// Texture format.
    pub format: HalTextureFormat,
    /// Aspect to clear.
    pub aspect: HalTextureAspect,
    /// Mip level.
    pub mip_level: u32,
    /// First array layer to clear.
    pub base_array_layer: u32,
    /// Number of array layers to clear.
    pub array_layer_count: u32,
}

/// Wraps query-set resolve data for the selected backend.
#[derive(Debug, Clone)]
pub struct HalResolveQuerySet {
    /// Source query set.
    pub query_set: HalQuerySet,
    /// First query index to resolve.
    pub first_query: u32,
    /// Number of query results to resolve.
    pub query_count: u32,
    /// Absolute query indices that were written by actual draws in this submission.
    pub written_queries: Vec<u32>,
    /// Destination buffer.
    pub destination: HalBuffer,
    /// Destination byte offset.
    pub destination_offset: u64,
}

/// Enumerates HAL copy values.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum HalCopy {
    /// Buffer variant.
    Buffer(HalBufferCopy),
    /// Buffer clear variant.
    BufferClear(HalBufferClear),
    /// Texture clear variant.
    ClearTexture(HalTextureClear),
    /// Buffer to texture variant.
    BufferToTexture(HalBufferTextureCopy),
    /// Texture to buffer variant.
    TextureToBuffer(HalBufferTextureCopy),
    /// Texture to texture variant.
    TextureToTexture(HalTextureCopy),
    /// Query-set resolve variant.
    ResolveQuerySet(HalResolveQuerySet),
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
    /// Bind textures.
    pub bind_textures: Vec<HalBoundTexture>,
    /// Bind samplers.
    pub bind_samplers: Vec<HalBoundSampler>,
    /// Bind external textures.
    pub bind_external_textures: Vec<HalBoundExternalTexture>,
    /// The pipeline's immediates block for this dispatch: user bytes
    /// `[0, layout.immediate_size)` (Block 94). Compute pipelines have no
    /// internal immediates today, so this is exactly the user prefix
    /// (`specs/blocks/94-immediates.md` "Immediates block layout").
    /// Delivery is backend-specific and lands in S2 (Metal) / S3 (Vulkan);
    /// Noop and (today) Metal/Vulkan/GLES all ignore this field.
    pub immediate_data: Vec<u8>,
    /// Dispatch command.
    pub dispatch: HalComputeDispatch,
}

/// Enumerates HAL compute dispatch execution values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalComputeDispatch {
    /// Direct dispatch.
    Direct {
        /// Workgroup counts.
        workgroups: (u32, u32, u32),
    },
    /// Indirect dispatch.
    Indirect {
        /// Buffer containing dispatch arguments.
        buffer: Box<HalBoundIndirectBuffer>,
    },
}

/// Stores binding metadata.
#[derive(Debug, Clone, Copy)]
pub struct HalDescriptorBinding {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Kind.
    pub kind: HalDescriptorBindingKind,
}

/// Enumerates HAL descriptor binding kind values.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum HalDescriptorBindingKind {
    /// Buffer variant.
    Buffer(HalBufferBindingKind),
    /// Sampled texture variant.
    Texture,
    /// Storage texture variant.
    StorageTexture {
        /// Storage texture access mode.
        access: HalStorageTextureAccess,
    },
    /// Sampler variant.
    Sampler,
    /// Input attachment self-read from the render pass color target at `color_slot`.
    InputAttachment {
        /// Color attachment slot read by this input attachment.
        color_slot: u32,
    },
}

/// Enumerates HAL storage texture access values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalStorageTextureAccess {
    /// Read-only storage texture access.
    ReadOnly,
    /// Write-only storage texture access.
    WriteOnly,
    /// Read-write storage texture access.
    ReadWrite,
}

/// Enumerates HAL buffer binding kind values.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
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
    /// Metal per-kind buffer-space slot (used by the compute encoder and as the
    /// fallback when `vertex_metal_index` / `fragment_metal_index` are both
    /// `None`).
    pub metal_index: u32,
    /// Metal per-kind buffer-space slot for the vertex stage of a render
    /// pipeline.  `None` for compute bindings and for bindings that are not
    /// visible to the vertex stage.
    pub vertex_metal_index: Option<u32>,
    /// Metal per-kind buffer-space slot for the fragment stage of a render
    /// pipeline.  `None` for compute bindings and for bindings that are not
    /// visible to the fragment stage.
    pub fragment_metal_index: Option<u32>,
    /// Buffer.
    pub buffer: HalBuffer,
    /// Offset.
    pub offset: u64,
    /// Size.
    pub size: u64,
}

/// Enumerates HAL texture view dimension values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalTextureViewDimension {
    /// One-dimensional texture view.
    D1,
    /// Two-dimensional texture view.
    D2,
    /// Two-dimensional array texture view.
    D2Array,
    /// Cube texture view.
    Cube,
    /// Cube array texture view.
    CubeArray,
    /// Three-dimensional texture view.
    D3,
}

/// Wraps a bound texture for the selected backend.
#[derive(Debug, Clone)]
pub struct HalBoundTexture {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Metal per-kind texture-space slot (used by the compute encoder and as
    /// the fallback when `vertex_metal_index` / `fragment_metal_index` are both
    /// `None`).
    pub metal_index: u32,
    /// Metal per-kind texture-space slot for the vertex stage of a render
    /// pipeline.  `None` for compute bindings and for bindings not visible to
    /// the vertex stage.
    pub vertex_metal_index: Option<u32>,
    /// Metal per-kind texture-space slot for the fragment stage of a render
    /// pipeline.  `None` for compute bindings and for bindings not visible to
    /// the fragment stage.
    pub fragment_metal_index: Option<u32>,
    /// Texture.
    pub texture: HalTexture,
    /// View format.
    pub format: HalTextureFormat,
    /// View dimension.
    pub dimension: HalTextureViewDimension,
    /// First mip level exposed by the view.
    pub base_mip_level: u32,
    /// Number of mip levels exposed by the view.
    pub mip_level_count: u32,
    /// First array layer exposed by the view.
    pub base_array_layer: u32,
    /// Number of array layers exposed by the view.
    pub array_layer_count: u32,
    /// Texture aspect exposed by the view.
    pub aspect: HalTextureAspect,
    /// Component swizzle exposed by the view.
    pub swizzle: HalTextureComponentSwizzle,
    /// Storage texture access mode when this binding is a storage texture.
    pub storage_access: Option<HalStorageTextureAccess>,
}

/// Wraps a bound sampler for the selected backend.
#[derive(Debug, Clone)]
pub struct HalBoundSampler {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Metal per-kind sampler-space slot (used by the compute encoder and as
    /// the fallback when `vertex_metal_index` / `fragment_metal_index` are both
    /// `None`).
    pub metal_index: u32,
    /// Metal per-kind sampler-space slot for the vertex stage of a render
    /// pipeline.  `None` for compute bindings and for bindings not visible to
    /// the vertex stage.
    pub vertex_metal_index: Option<u32>,
    /// Metal per-kind sampler-space slot for the fragment stage of a render
    /// pipeline.  `None` for compute bindings and for bindings not visible to
    /// the fragment stage.
    pub fragment_metal_index: Option<u32>,
    /// Sampler.
    pub sampler: HalSampler,
}

/// Wraps a bound external texture for the selected backend.
#[derive(Debug, Clone)]
pub struct HalBoundExternalTexture {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Plane0 texture.
    pub plane0: HalTexture,
    /// Plane1 texture, or plane0 again for single-plane external textures.
    pub plane1: HalTexture,
    /// Plane0 Metal texture-space slot.
    pub plane0_metal_index: u32,
    /// Plane1 Metal texture-space slot.
    pub plane1_metal_index: u32,
    /// Plane0 Metal texture-space slot for the vertex stage.
    pub plane0_vertex_metal_index: Option<u32>,
    /// Plane1 Metal texture-space slot for the vertex stage.
    pub plane1_vertex_metal_index: Option<u32>,
    /// Plane0 Metal texture-space slot for the fragment stage.
    pub plane0_fragment_metal_index: Option<u32>,
    /// Plane1 Metal texture-space slot for the fragment stage.
    pub plane1_fragment_metal_index: Option<u32>,
    /// Params buffer.
    pub params: HalBuffer,
    /// Params Metal buffer-space slot.
    pub params_metal_index: u32,
    /// Params Metal buffer-space slot for the vertex stage.
    pub params_vertex_metal_index: Option<u32>,
    /// Params Metal buffer-space slot for the fragment stage.
    pub params_fragment_metal_index: Option<u32>,
    /// Plane view format.
    pub format: HalTextureFormat,
    /// Plane view dimension.
    pub dimension: HalTextureViewDimension,
    /// Params buffer byte offset.
    pub params_offset: u64,
    /// Params buffer byte size.
    pub params_size: u64,
}

/// Wraps  for the selected backend.
#[derive(Debug, Clone)]
pub struct HalRenderPass {
    /// Pipeline.
    pub pipeline: Option<HalRenderPipeline>,
    /// Color targets in attachment slot order; `None` is an empty slot.
    pub color_targets: Vec<Option<HalRenderColorTarget>>,
    /// Color attachment slots read as framebuffer-fetch input attachments.
    pub framebuffer_fetch_color_slots: Vec<u32>,
    /// Optional depth-stencil attachment.
    pub depth_stencil_attachment: Option<HalRenderDepthStencilAttachment>,
    /// Bind buffers.
    pub bind_buffers: Vec<HalBoundBuffer>,
    /// Bind textures.
    pub bind_textures: Vec<HalBoundTexture>,
    /// Bind samplers.
    pub bind_samplers: Vec<HalBoundSampler>,
    /// Bind external textures.
    pub bind_external_textures: Vec<HalBoundExternalTexture>,
    /// Vertex buffers.
    pub vertex_buffers: Vec<HalBoundBuffer>,
    /// Optional index buffer.
    pub index_buffer: Option<Box<HalBoundIndexBuffer>>,
    /// Optional indirect draw buffer.
    pub indirect_buffer: Option<Box<HalBoundIndirectBuffer>>,
    /// Optional viewport override.
    pub viewport: Option<HalViewport>,
    /// Optional scissor rectangle override.
    pub scissor_rect: Option<HalScissorRect>,
    /// Render pass blend constant.
    pub blend_constant: [f32; 4],
    /// Render pass stencil reference.
    pub stencil_reference: u32,
    /// Optional occlusion query set for this render pass.
    pub occlusion_query_set: Option<HalQuerySet>,
    /// Optional active occlusion query index for this draw.
    pub occlusion_query_index: Option<u32>,
    /// Draw.
    pub draw: Option<HalDraw>,
    /// The pipeline's immediates block for this draw: user bytes
    /// `[0, layout.immediate_size)` today (Block 94). The fragment
    /// frag-depth-clamp internal constants (`ClampFragDepthArgs`,
    /// `dawn/native/ImmediatesLayout.h:47-50`) that follow the user prefix
    /// per `specs/blocks/94-immediates.md` "Immediates block layout" are a
    /// Metal-delivery concern threaded in S2, not carried here. Delivery is
    /// backend-specific and lands in S2 (Metal) / S3 (Vulkan); Noop and
    /// (today) Metal/Vulkan/GLES all ignore this field.
    pub immediate_data: Vec<u8>,
}

/// Stores HAL viewport state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalViewport {
    /// X origin.
    pub x: f32,
    /// Y origin.
    pub y: f32,
    /// Width.
    pub width: f32,
    /// Height.
    pub height: f32,
    /// Minimum depth.
    pub min_depth: f32,
    /// Maximum depth.
    pub max_depth: f32,
}

/// Stores HAL scissor rectangle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalScissorRect {
    /// X origin.
    pub x: u32,
    /// Y origin.
    pub y: u32,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
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
    /// Bind textures.
    pub bind_textures: Vec<HalBoundTexture>,
    /// Bind samplers.
    pub bind_samplers: Vec<HalBoundSampler>,
    /// Vertex buffers.
    pub vertex_buffers: Vec<HalBoundBuffer>,
    /// Optional viewport override.
    pub viewport: Option<HalViewport>,
    /// Optional scissor rectangle override.
    pub scissor_rect: Option<HalScissorRect>,
    /// Draw.
    pub draw: HalDraw,
}

/// Stores color metadata.
#[derive(Debug, Clone)]
pub struct HalRenderColorTarget {
    /// Texture.
    pub texture: HalTexture,
    /// Format of the WebGPU view bound as this color attachment; may reinterpret
    /// the underlying texture.
    pub view_format: HalTextureFormat,
    /// Optional resolve target texture.
    pub resolve_target: Option<HalTexture>,
    /// Format of the WebGPU view bound as the resolve target, when present.
    pub resolve_view_format: Option<HalTextureFormat>,
    /// Mip level the attachment view targets.
    pub mip_level: u32,
    /// Array layer the attachment view targets.
    pub array_layer: u32,
    /// Depth slice selected for 3D color attachments; zero for non-3D targets.
    pub depth_slice: u32,
    /// Mip level the resolve target view targets.
    pub resolve_mip_level: u32,
    /// Array layer the resolve target view targets.
    pub resolve_array_layer: u32,
    /// Load op.
    pub load_op: HalRenderLoadOp,
    /// Store.
    pub store: bool,
    /// Clear color.
    pub clear_color: [f64; 4],
}

/// Stores one regular render pass depth-stencil attachment binding.
#[derive(Debug, Clone)]
pub struct HalRenderDepthStencilAttachment {
    /// Texture.
    pub texture: HalTexture,
    /// Texture format.
    pub format: HalTextureFormat,
    /// Mip level the attachment view targets.
    pub mip_level: u32,
    /// Array layer the attachment view targets.
    pub array_layer: u32,
    /// Depth load op.
    pub depth_load_op: HalRenderLoadOp,
    /// Depth store.
    pub depth_store: bool,
    /// Depth clear value.
    pub depth_clear_value: f32,
    /// Depth read-only flag.
    pub depth_read_only: bool,
    /// Stencil load op.
    pub stencil_load_op: HalRenderLoadOp,
    /// Stencil store.
    pub stencil_store: bool,
    /// Stencil clear value.
    pub stencil_clear_value: u32,
    /// Stencil read-only flag.
    pub stencil_read_only: bool,
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
#[non_exhaustive]
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
#[non_exhaustive]
pub enum HalSubpassAttachmentResource {
    /// Persistent texture.
    Persistent {
        /// Texture.
        texture: HalTexture,
        /// Optional resolve target.
        resolve_target: Option<HalTexture>,
    },
    // TODO(tiled 2.4): transient attachment arm
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalIndexFormat {
    /// Unsigned 16-bit indices.
    Uint16,
    /// Unsigned 32-bit indices.
    Uint32,
}

/// Stores a bound index buffer for render draw execution.
#[derive(Debug, Clone)]
pub struct HalBoundIndexBuffer {
    /// Buffer.
    pub buffer: HalBuffer,
    /// Index format.
    pub format: HalIndexFormat,
    /// Offset.
    pub offset: u64,
    /// Size.
    pub size: u64,
}

/// Stores a bound indirect draw buffer for render draw execution.
#[derive(Debug, Clone)]
pub struct HalBoundIndirectBuffer {
    /// Buffer.
    pub buffer: HalBuffer,
    /// Offset.
    pub offset: u64,
}

/// Enumerates render draw execution values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalDraw {
    /// Direct non-indexed draw.
    Direct {
        /// Vertex count.
        vertex_count: u32,
        /// Instance count.
        instance_count: u32,
        /// First vertex.
        first_vertex: u32,
        /// First instance.
        first_instance: u32,
    },
    /// Direct indexed draw.
    Indexed {
        /// Index count.
        index_count: u32,
        /// Instance count.
        instance_count: u32,
        /// First index.
        first_index: u32,
        /// Base vertex.
        base_vertex: i32,
        /// First instance.
        first_instance: u32,
    },
    /// Indirect non-indexed draw.
    Indirect {
        /// Offset into the indirect buffer.
        offset: u64,
    },
    /// Indirect indexed draw.
    IndexedIndirect {
        /// Offset into the indirect buffer.
        offset: u64,
    },
}

#[cfg(test)]
mod external_texture_tests {
    use super::*;
    use crate::{
        HalBufferUsage, HalInstance, HalTextureDescriptor, HalTextureDimension, HalTextureFormat,
        HalTextureUsage,
    };

    fn noop_device() -> crate::HalDevice {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter");
        adapter.create_device().expect("Noop device")
    }

    fn noop_texture(device: &crate::HalDevice) -> crate::HalTexture {
        device
            .create_texture(&HalTextureDescriptor {
                dimension: HalTextureDimension::D2,
                format: HalTextureFormat::Rgba8Unorm,
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: HalTextureUsage {
                    copy_src: false,
                    copy_dst: false,
                    texture_binding: true,
                    storage_binding: false,
                    render_attachment: false,
                    transient: false,
                },
            })
            .expect("Noop texture")
    }

    #[test]
    fn external_texture_binding_struct_carries_planes_params_and_stage_slots() {
        let device = noop_device();
        let plane0 = noop_texture(&device);
        let plane1 = noop_texture(&device);
        let params = device
            .create_buffer(
                296,
                HalBufferUsage {
                    uniform: true,
                    ..HalBufferUsage::default()
                },
            )
            .expect("Noop params buffer");

        let binding = HalBoundExternalTexture {
            group: 1,
            binding: 2,
            plane0,
            plane1,
            plane0_metal_index: 4,
            plane1_metal_index: 5,
            plane0_vertex_metal_index: Some(10),
            plane1_vertex_metal_index: Some(11),
            plane0_fragment_metal_index: Some(20),
            plane1_fragment_metal_index: Some(21),
            params,
            params_metal_index: 7,
            params_vertex_metal_index: Some(12),
            params_fragment_metal_index: Some(22),
            format: HalTextureFormat::Rgba8Unorm,
            dimension: HalTextureViewDimension::D2,
            params_offset: 0,
            params_size: 296,
        };
        let cloned = binding.clone();

        assert_eq!(cloned.group, 1);
        assert_eq!(cloned.binding, 2);
        assert_eq!(cloned.plane0_metal_index, 4);
        assert_eq!(cloned.plane1_metal_index, 5);
        assert_eq!(cloned.plane0_vertex_metal_index, Some(10));
        assert_eq!(cloned.plane1_vertex_metal_index, Some(11));
        assert_eq!(cloned.plane0_fragment_metal_index, Some(20));
        assert_eq!(cloned.plane1_fragment_metal_index, Some(21));
        assert_eq!(cloned.params_metal_index, 7);
        assert_eq!(cloned.params_vertex_metal_index, Some(12));
        assert_eq!(cloned.params_fragment_metal_index, Some(22));
        assert_eq!(cloned.params_offset, 0);
        assert_eq!(cloned.params_size, 296);
    }
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
    /// Texture format (so backends can select the depth/stencil plane).
    pub format: HalTextureFormat,
    /// Aspect of the texture this copy targets.
    pub aspect: HalTextureAspect,
    /// Mip level.
    pub mip_level: u32,
    /// Origin.
    pub origin: HalOrigin3d,
    /// Extent.
    pub extent: HalExtent3d,
}

/// Selects which aspect of a texture a buffer⇄texture copy targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalTextureAspect {
    /// All aspects (color, or the single plane of a single-aspect format).
    All,
    /// Depth plane only.
    DepthOnly,
    /// Stencil plane only.
    StencilOnly,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        noop, HalBufferUsage, HalTextureDescriptor, HalTextureDimension, HalTextureFormat,
        HalTextureUsage,
    };

    fn depth_texture() -> HalTexture {
        let device = noop::NoopDevice::new();
        HalTexture::Noop(
            device
                .create_texture(&HalTextureDescriptor {
                    dimension: HalTextureDimension::D2,
                    format: HalTextureFormat::Depth32Float,
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                    mip_level_count: 1,
                    sample_count: 1,
                    usage: HalTextureUsage {
                        copy_src: false,
                        copy_dst: false,
                        texture_binding: false,
                        storage_binding: false,
                        render_attachment: true,
                        transient: false,
                    },
                })
                .expect("Noop texture allocation should succeed"),
        )
    }

    fn noop_buffer(size: u64) -> HalBuffer {
        let device = noop::NoopDevice::new();
        HalBuffer::Noop(
            device
                .create_buffer(size, HalBufferUsage::default())
                .expect("Noop buffer allocation should succeed"),
        )
    }

    #[test]
    fn hal_render_depth_stencil_attachment_constructs_and_round_trips_fields() {
        let texture = depth_texture();
        let attachment = HalRenderDepthStencilAttachment {
            texture,
            format: HalTextureFormat::Depth32Float,
            mip_level: 0,
            array_layer: 0,
            depth_load_op: HalRenderLoadOp::Clear,
            depth_store: true,
            depth_clear_value: 0.5,
            depth_read_only: false,
            stencil_load_op: HalRenderLoadOp::Load,
            stencil_store: false,
            stencil_clear_value: 7,
            stencil_read_only: true,
        };

        assert!(matches!(attachment.texture, HalTexture::Noop(_)));
        assert_eq!(attachment.format, HalTextureFormat::Depth32Float);
        assert!(matches!(attachment.depth_load_op, HalRenderLoadOp::Clear));
        assert!(attachment.depth_store);
        assert_eq!(attachment.depth_clear_value, 0.5);
        assert!(!attachment.depth_read_only);
        assert!(matches!(attachment.stencil_load_op, HalRenderLoadOp::Load));
        assert!(!attachment.stencil_store);
        assert_eq!(attachment.stencil_clear_value, 7);
        assert!(attachment.stencil_read_only);
    }

    #[test]
    fn hal_bound_texture_constructs_and_round_trips_view_fields() {
        let texture = depth_texture();
        let binding = HalBoundTexture {
            group: 1,
            binding: 2,
            metal_index: 3,
            vertex_metal_index: Some(1),
            fragment_metal_index: Some(2),
            texture,
            format: HalTextureFormat::Depth32Float,
            dimension: HalTextureViewDimension::D2,
            base_mip_level: 4,
            mip_level_count: 5,
            base_array_layer: 6,
            array_layer_count: 7,
            aspect: HalTextureAspect::DepthOnly,
            swizzle: HalTextureComponentSwizzle::default(),
            storage_access: Some(HalStorageTextureAccess::ReadOnly),
        };

        assert_eq!(binding.group, 1);
        assert_eq!(binding.binding, 2);
        assert_eq!(binding.metal_index, 3);
        assert_eq!(binding.vertex_metal_index, Some(1));
        assert_eq!(binding.fragment_metal_index, Some(2));
        assert!(matches!(binding.texture, HalTexture::Noop(_)));
        assert_eq!(binding.format, HalTextureFormat::Depth32Float);
        assert_eq!(binding.dimension, HalTextureViewDimension::D2);
        assert_eq!(binding.base_mip_level, 4);
        assert_eq!(binding.mip_level_count, 5);
        assert_eq!(binding.base_array_layer, 6);
        assert_eq!(binding.array_layer_count, 7);
        assert_eq!(binding.aspect, HalTextureAspect::DepthOnly);
        assert_eq!(binding.swizzle, HalTextureComponentSwizzle::default());
        assert_eq!(
            binding.storage_access,
            Some(HalStorageTextureAccess::ReadOnly)
        );
    }

    #[test]
    fn hal_draw_index_and_indirect_bindings_round_trip() {
        let index_buffer = HalBoundIndexBuffer {
            buffer: noop_buffer(32),
            format: HalIndexFormat::Uint16,
            offset: 4,
            size: 16,
        };
        let indirect_buffer = HalBoundIndirectBuffer {
            buffer: noop_buffer(32),
            offset: 8,
        };
        let indexed = HalDraw::Indexed {
            index_count: 3,
            instance_count: 2,
            first_index: 1,
            base_vertex: -1,
            first_instance: 4,
        };
        let indirect = HalDraw::IndexedIndirect { offset: 8 };

        assert!(matches!(index_buffer.buffer, HalBuffer::Noop(_)));
        assert_eq!(index_buffer.format, HalIndexFormat::Uint16);
        assert_eq!(index_buffer.offset, 4);
        assert_eq!(index_buffer.size, 16);
        assert!(matches!(indirect_buffer.buffer, HalBuffer::Noop(_)));
        assert_eq!(indirect_buffer.offset, 8);
        assert!(matches!(
            indexed,
            HalDraw::Indexed {
                index_count: 3,
                instance_count: 2,
                first_index: 1,
                base_vertex: -1,
                first_instance: 4,
            }
        ));
        assert!(matches!(indirect, HalDraw::IndexedIndirect { offset: 8 }));
    }
}
