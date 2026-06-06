#[cfg(feature = "tiled")]
use crate::HalError;
#[cfg(feature = "tiled")]
use crate::HalTransientAttachment;
use crate::{
    HalBuffer, HalComputePipeline, HalExtent3d, HalOrigin3d, HalRenderPipeline, HalSampler,
    HalTexture, HalTextureFormat,
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

/// Enumerates HAL copy values.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum HalCopy {
    /// Buffer variant.
    Buffer(HalBufferCopy),
    /// Buffer clear variant.
    BufferClear(HalBufferClear),
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
    /// Bind textures.
    pub bind_textures: Vec<HalBoundTexture>,
    /// Bind samplers.
    pub bind_samplers: Vec<HalBoundSampler>,
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
    /// Metal index.
    pub metal_index: u32,
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
    /// Metal index.
    pub metal_index: u32,
    /// Sampler.
    pub sampler: HalSampler,
}

/// Wraps  for the selected backend.
#[derive(Debug, Clone)]
pub struct HalRenderPass {
    /// Pipeline.
    pub pipeline: Option<HalRenderPipeline>,
    /// Color targets in attachment slot order.
    pub color_targets: Vec<HalRenderColorTarget>,
    /// Optional depth-stencil attachment.
    pub depth_stencil_attachment: Option<HalRenderDepthStencilAttachment>,
    /// Bind buffers.
    pub bind_buffers: Vec<HalBoundBuffer>,
    /// Bind textures.
    pub bind_textures: Vec<HalBoundTexture>,
    /// Bind samplers.
    pub bind_samplers: Vec<HalBoundSampler>,
    /// Vertex buffers.
    pub vertex_buffers: Vec<HalBoundBuffer>,
    /// Optional index buffer.
    pub index_buffer: Option<Box<HalBoundIndexBuffer>>,
    /// Optional indirect draw buffer.
    pub indirect_buffer: Option<Box<HalBoundIndirectBuffer>>,
    /// Render pass blend constant.
    pub blend_constant: [f32; 4],
    /// Render pass stencil reference.
    pub stencil_reference: u32,
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
    /// Bind textures.
    pub bind_textures: Vec<HalBoundTexture>,
    /// Bind samplers.
    pub bind_samplers: Vec<HalBoundSampler>,
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
    /// Optional resolve target texture.
    pub resolve_target: Option<HalTexture>,
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
        HalTexture::Noop(device.create_texture(&HalTextureDescriptor {
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
            },
        }))
    }

    fn noop_buffer(size: u64) -> HalBuffer {
        let device = noop::NoopDevice::new();
        HalBuffer::Noop(device.create_buffer(size, HalBufferUsage::default()))
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
            texture,
            format: HalTextureFormat::Depth32Float,
            dimension: HalTextureViewDimension::D2,
            base_mip_level: 4,
            mip_level_count: 5,
            base_array_layer: 6,
            array_layer_count: 7,
            aspect: HalTextureAspect::DepthOnly,
            storage_access: Some(HalStorageTextureAccess::ReadOnly),
        };

        assert_eq!(binding.group, 1);
        assert_eq!(binding.binding, 2);
        assert_eq!(binding.metal_index, 3);
        assert!(matches!(binding.texture, HalTexture::Noop(_)));
        assert_eq!(binding.format, HalTextureFormat::Depth32Float);
        assert_eq!(binding.dimension, HalTextureViewDimension::D2);
        assert_eq!(binding.base_mip_level, 4);
        assert_eq!(binding.mip_level_count, 5);
        assert_eq!(binding.base_array_layer, 6);
        assert_eq!(binding.array_layer_count, 7);
        assert_eq!(binding.aspect, HalTextureAspect::DepthOnly);
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
