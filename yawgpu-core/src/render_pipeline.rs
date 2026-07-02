use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use yawgpu_hal::{
    HalBackend, HalBlendComponent, HalBlendFactor, HalBlendOperation, HalBlendState,
    HalColorTargetState, HalCompareFunction, HalCullMode, HalDepthStencilState,
    HalDescriptorBinding, HalDescriptorBindingKind, HalDevice, HalFrontFace, HalMslImmediates,
    HalPrimitiveTopology, HalRenderPipeline, HalRenderPipelineDescriptor, HalShaderSource,
    HalStencilFaceState, HalStencilOperation, HalVertexAttribute, HalVertexBufferLayout,
    HalVertexFormat, HalVertexStepMode,
};

use crate::adapter::Feature;
use crate::bind_group_layout::*;
use crate::compute_pipeline::*;
use crate::device::FeatureSet;
use crate::format::*;
use crate::frontend;
use crate::limits::*;
use crate::pipeline_id::next_pipeline_id;
use crate::pipeline_layout::*;
use crate::sampler::*;
use crate::shader::*;
#[cfg(feature = "tiled")]
use crate::subpass::{
    compute_subpass_color_slots, hal_subpass_pass_layout, SubpassPassLayout,
    SubpassPassLayoutDescriptor, DEPTH_STENCIL_ATTACHMENT_INDEX,
};
use crate::texture::*;

/// Stores attachment signature data used by validation and backend submission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttachmentSignature {
    pub(crate) color_formats: Vec<Option<TextureFormat>>,
    pub(crate) depth_stencil_format: Option<TextureFormat>,
    pub(crate) sample_count: u32,
    pub(crate) depth_read_only: bool,
    pub(crate) stencil_read_only: bool,
}

impl AttachmentSignature {
    /// Returns whether a render bundle carrying this signature may execute in a
    /// render pass whose signature is `pass`. Color/depth-stencil formats and
    /// sample count must match exactly; for each aspect the pass marks
    /// read-only, the bundle must also be read-only — a read-write bundle cannot
    /// run in a read-only pass, but a read-only bundle may run in a read-write
    /// pass. Mirrors the WebGPU `executeBundles` rule (F-062), which is *not*
    /// whole-signature equality.
    pub(crate) fn bundle_compatible_with_pass(&self, pass: &AttachmentSignature) -> bool {
        self.color_formats == pass.color_formats
            && self.depth_stencil_format == pass.depth_stencil_format
            && self.sample_count == pass.sample_count
            && (!pass.depth_read_only || self.depth_read_only)
            && (!pass.stencil_read_only || self.stencil_read_only)
    }
}

/// Describes render pipeline descriptor.
#[derive(Debug, Clone)]
pub struct RenderPipelineDescriptor {
    /// Layout.
    pub layout: RenderPipelineLayout,
    /// Vertex.
    pub vertex: RenderPipelineVertexState,
    /// Primitive.
    pub primitive: PrimitiveState,
    /// Depth stencil.
    pub depth_stencil: Option<DepthStencilState>,
    /// Multisample.
    pub multisample: MultisampleState,
    /// Fragment.
    pub fragment: Option<RenderPipelineFragmentState>,
    /// Error.
    pub error: Option<String>,
}

/// Enumerates render pipeline layout values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RenderPipelineLayout {
    /// Auto variant.
    Auto,
    /// Explicit variant.
    Explicit(Arc<PipelineLayout>),
}

/// Tracks the lifecycle state for render pipeline vertex.
#[derive(Debug, Clone)]
pub struct RenderPipelineVertexState {
    /// Shader.
    pub shader: RenderPipelineShaderStage,
    /// Buffer count.
    pub buffer_count: usize,
    /// Buffers.
    pub buffers: Vec<VertexBufferLayout>,
}

/// Stores layout metadata.
#[derive(Debug, Clone)]
pub struct VertexBufferLayout {
    /// True when this slot is a used vertex buffer layout.
    ///
    /// False marks a WebGPU unused/gap slot: stepMode Undefined with no
    /// attributes. Its arrayStride is still creation-validated, but the slot
    /// requires no bound buffer and is not emitted to the HAL.
    pub used: bool,
    /// Array stride.
    pub array_stride: u64,
    /// Step mode.
    pub step_mode: VertexStepMode,
    /// Attributes.
    pub attributes: Vec<VertexAttribute>,
}

/// Enumerates vertex step mode values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexStepMode {
    /// Vertex variant.
    Vertex,
    /// Instance variant.
    Instance,
}

/// Stores attribute metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    /// Format.
    pub format: VertexFormat,
    /// Offset.
    pub offset: u64,
    /// Shader location.
    pub shader_location: u32,
}

/// Enumerates vertex format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexFormat(u32);

impl VertexFormat {
    /// Constant value for fn.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the raw.
    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }

    /// Returns this vertex format's byte size and output class.
    pub(crate) fn info(self) -> VertexFormatInfo {
        match self.0 {
            0x0000_0001 => VertexFormatInfo::new(1, FormatOutputClass::Uint),
            0x0000_0002 => VertexFormatInfo::new(2, FormatOutputClass::Uint),
            0x0000_0003 => VertexFormatInfo::new(4, FormatOutputClass::Uint),
            0x0000_0004 => VertexFormatInfo::new(1, FormatOutputClass::Sint),
            0x0000_0005 => VertexFormatInfo::new(2, FormatOutputClass::Sint),
            0x0000_0006 => VertexFormatInfo::new(4, FormatOutputClass::Sint),
            0x0000_0007 | 0x0000_000A => VertexFormatInfo::new(1, FormatOutputClass::Float),
            0x0000_0008 | 0x0000_000B => VertexFormatInfo::new(2, FormatOutputClass::Float),
            0x0000_0009 | 0x0000_000C => VertexFormatInfo::new(4, FormatOutputClass::Float),
            0x0000_000D => VertexFormatInfo::new(2, FormatOutputClass::Uint),
            0x0000_000E => VertexFormatInfo::new(4, FormatOutputClass::Uint),
            0x0000_000F => VertexFormatInfo::new(8, FormatOutputClass::Uint),
            0x0000_0010 => VertexFormatInfo::new(2, FormatOutputClass::Sint),
            0x0000_0011 => VertexFormatInfo::new(4, FormatOutputClass::Sint),
            0x0000_0012 => VertexFormatInfo::new(8, FormatOutputClass::Sint),
            0x0000_0013 | 0x0000_0016 | 0x0000_0019 => {
                VertexFormatInfo::new(2, FormatOutputClass::Float)
            }
            0x0000_0014 | 0x0000_0017 | 0x0000_001A => {
                VertexFormatInfo::new(4, FormatOutputClass::Float)
            }
            0x0000_0015 | 0x0000_0018 | 0x0000_001B => {
                VertexFormatInfo::new(8, FormatOutputClass::Float)
            }
            0x0000_001C => VertexFormatInfo::new(4, FormatOutputClass::Float),
            0x0000_001D => VertexFormatInfo::new(8, FormatOutputClass::Float),
            0x0000_001E => VertexFormatInfo::new(12, FormatOutputClass::Float),
            0x0000_001F => VertexFormatInfo::new(16, FormatOutputClass::Float),
            0x0000_0020 => VertexFormatInfo::new(4, FormatOutputClass::Uint),
            0x0000_0021 => VertexFormatInfo::new(8, FormatOutputClass::Uint),
            0x0000_0022 => VertexFormatInfo::new(12, FormatOutputClass::Uint),
            0x0000_0023 => VertexFormatInfo::new(16, FormatOutputClass::Uint),
            0x0000_0024 => VertexFormatInfo::new(4, FormatOutputClass::Sint),
            0x0000_0025 => VertexFormatInfo::new(8, FormatOutputClass::Sint),
            0x0000_0026 => VertexFormatInfo::new(12, FormatOutputClass::Sint),
            0x0000_0027 => VertexFormatInfo::new(16, FormatOutputClass::Sint),
            0x0000_0028 | 0x0000_0029 => VertexFormatInfo::new(4, FormatOutputClass::Float),
            // Keep unknown future values conservative instead of guessing a smaller footprint.
            _ => VertexFormatInfo::new(16, FormatOutputClass::Float),
        }
    }
}

impl From<u32> for VertexFormat {
    fn from(value: u32) -> Self {
        Self::from_raw(value)
    }
}

impl From<i32> for VertexFormat {
    fn from(value: i32) -> Self {
        Self::from_raw(value as u32)
    }
}

impl From<VertexFormat> for u32 {
    fn from(value: VertexFormat) -> Self {
        value.raw()
    }
}

impl From<VertexFormat> for i32 {
    fn from(value: VertexFormat) -> Self {
        value.raw() as i32
    }
}

/// Stores info metadata.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VertexFormatInfo {
    pub(crate) byte_size: u64,
    pub(crate) output_class: FormatOutputClass,
}

impl VertexFormatInfo {
    /// Constant value for fn.
    pub(crate) const fn new(byte_size: u64, output_class: FormatOutputClass) -> Self {
        Self {
            byte_size,
            output_class,
        }
    }
}

/// Tracks the lifecycle state for render pipeline fragment.
#[derive(Debug, Clone)]
pub struct RenderPipelineFragmentState {
    /// Shader.
    pub shader: RenderPipelineShaderStage,
    /// Target count.
    pub target_count: usize,
    /// Targets.
    pub targets: Vec<ColorTargetState>,
}

/// Stores render pipeline shader stage data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct RenderPipelineShaderStage {
    /// Module.
    pub module: Arc<ShaderModule>,
    /// Entry point.
    pub entry_point: Option<String>,
    /// Constants.
    pub constants: Vec<PipelineConstant>,
}

/// Stores color metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorTargetState {
    /// Format.
    pub format: TextureFormat,
    /// Blend.
    pub blend: Option<BlendState>,
    /// Write mask.
    pub write_mask: u64,
}

/// Stores blend state metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendState {
    /// Color component.
    pub color: BlendComponent,
    /// Alpha component.
    pub alpha: BlendComponent,
}

/// Stores blend component metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendComponent {
    /// Operation.
    pub operation: BlendOperation,
    /// Source factor.
    pub src_factor: BlendFactor,
    /// Destination factor.
    pub dst_factor: BlendFactor,
}

/// Enumerates blend operation values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BlendOperation {
    /// Add variant.
    Add,
    /// Subtract variant.
    Subtract,
    /// Reverse subtract variant.
    ReverseSubtract,
    /// Min variant.
    Min,
    /// Max variant.
    Max,
}

/// Enumerates blend factor values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BlendFactor {
    /// Zero variant.
    Zero,
    /// One variant.
    One,
    /// Source variant.
    Src,
    /// One minus source variant.
    OneMinusSrc,
    /// Source alpha variant.
    SrcAlpha,
    /// One minus source alpha variant.
    OneMinusSrcAlpha,
    /// Destination variant.
    Dst,
    /// One minus destination variant.
    OneMinusDst,
    /// Destination alpha variant.
    DstAlpha,
    /// One minus destination alpha variant.
    OneMinusDstAlpha,
    /// Source alpha saturated variant.
    SrcAlphaSaturated,
    /// Constant variant.
    Constant,
    /// One minus constant variant.
    OneMinusConstant,
    /// Source one variant.
    Src1,
    /// One minus source one variant.
    OneMinusSrc1,
    /// Source one alpha variant.
    Src1Alpha,
    /// One minus source one alpha variant.
    OneMinusSrc1Alpha,
}

/// Tracks the lifecycle state for primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveState {
    /// Topology.
    pub topology: PrimitiveTopology,
    /// Strip index format.
    pub strip_index_format: Option<IndexFormat>,
    /// Front-facing winding.
    pub front_face: FrontFace,
    /// Face culling mode.
    pub cull_mode: CullMode,
    /// Whether clip-space depth is unclipped.
    pub unclipped_depth: bool,
}

/// Enumerates front-facing winding values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrontFace {
    /// Counter-clockwise winding.
    Ccw,
    /// Clockwise winding.
    Cw,
}

/// Enumerates primitive culling values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CullMode {
    /// Disable face culling.
    None,
    /// Cull front-facing primitives.
    Front,
    /// Cull back-facing primitives.
    Back,
}

/// Enumerates primitive topology values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrimitiveTopology {
    /// Point list variant.
    PointList,
    /// Line list variant.
    LineList,
    /// Line strip variant.
    LineStrip,
    /// Triangle list variant.
    TriangleList,
    /// Triangle strip variant.
    TriangleStrip,
}

/// Enumerates index format values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IndexFormat {
    /// Uint16 variant.
    Uint16,
    /// Uint32 variant.
    Uint32,
}

/// Tracks the lifecycle state for depth stencil.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthStencilState {
    /// Format.
    pub format: TextureFormat,
    /// Depth write enabled.
    pub depth_write_enabled: Option<bool>,
    /// Depth compare.
    pub depth_compare: Option<CompareFunction>,
    /// Stencil front.
    pub stencil_front: StencilFaceState,
    /// Stencil back.
    pub stencil_back: StencilFaceState,
    /// Stencil read mask.
    pub stencil_read_mask: u32,
    /// Stencil write mask.
    pub stencil_write_mask: u32,
    /// Depth bias.
    pub depth_bias: i32,
    /// Depth bias slope scale.
    pub depth_bias_slope_scale: f32,
    /// Depth bias clamp.
    pub depth_bias_clamp: f32,
}

/// Tracks the lifecycle state for stencil face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilFaceState {
    /// Compare.
    pub compare: CompareFunction,
    /// Fail op.
    pub fail_op: StencilOperation,
    /// Depth fail op.
    pub depth_fail_op: StencilOperation,
    /// Pass op.
    pub pass_op: StencilOperation,
}

/// Enumerates stencil operation values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StencilOperation {
    /// Keep variant.
    Keep,
    /// Zero variant.
    Zero,
    /// Replace variant.
    Replace,
    /// Invert variant.
    Invert,
    /// Increment clamp variant.
    IncrementClamp,
    /// Decrement clamp variant.
    DecrementClamp,
    /// Increment wrap variant.
    IncrementWrap,
    /// Decrement wrap variant.
    DecrementWrap,
}

/// Tracks the lifecycle state for multisample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultisampleState {
    /// Count.
    pub count: u32,
    /// Mask.
    pub mask: u32,
    /// Alpha to coverage enabled.
    pub alpha_to_coverage_enabled: bool,
}

/// Stores render pipeline data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct RenderPipeline {
    pub(crate) inner: Arc<RenderPipelineInner>,
}

/// Holds shared state for the render pipeline handle.
#[derive(Debug)]
pub(crate) struct RenderPipelineInner {
    pub(crate) _layout: RenderPipelineLayout,
    pub(crate) _vertex: RenderPipelineVertexState,
    pub(crate) _primitive: PrimitiveState,
    pub(crate) _depth_stencil: Option<DepthStencilState>,
    pub(crate) _multisample: MultisampleState,
    pub(crate) _fragment: Option<RenderPipelineFragmentState>,
    pub(crate) vertex_entry_name: String,
    pub(crate) fragment_entry_name: Option<String>,
    pub(crate) metal_bindings: Vec<MetalBufferBinding>,
    pub(crate) vertex_buffer_bindings: Vec<MetalVertexBufferBinding>,
    pub(crate) hal: Option<HalRenderPipeline>,
    pub(crate) bind_group_layouts: Vec<Arc<BindGroupLayout>>,
    pub(crate) framebuffer_fetch_color_slots: Vec<u32>,
    #[cfg(feature = "tiled")]
    #[allow(dead_code)]
    pub(crate) subpass_compatibility: Option<SubpassPipelineCompatibility>,
    pub(crate) is_error: bool,
}

/// Describes subpass-pipeline compatibility metadata.
#[cfg(feature = "tiled")]
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct SubpassPipelineCompatibility {
    /// Compatible subpass pass layout.
    pub(crate) pass_layout: Arc<SubpassPassLayout>,
    /// Compatible subpass index.
    pub(crate) subpass_index: u32,
}

/// Describes a subpass render pipeline descriptor.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub struct SubpassRenderPipelineDescriptor {
    /// Base render pipeline descriptor.
    pub base: RenderPipelineDescriptor,
    /// Compatible subpass pass layout.
    pub pass_layout: Arc<SubpassPassLayout>,
    /// Compatible subpass index.
    pub subpass_index: u32,
    /// Descriptor error from FFI conversion.
    pub error: Option<String>,
}

/// Stores binding metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalVertexBufferBinding {
    pub(crate) slot: u32,
    pub(crate) metal_index: u32,
}

impl RenderPipeline {
    /// Creates a new instance.
    pub(crate) fn new(
        descriptor: RenderPipelineDescriptor,
        is_error: bool,
        limits: Limits,
        features: &FeatureSet,
        hal_device: Option<&HalDevice>,
    ) -> (Self, Option<String>) {
        let pipeline_id = next_pipeline_id();
        let resolved = if is_error {
            None
        } else {
            resolve_render_pipeline_descriptor_for_source(
                &descriptor,
                limits,
                features,
                None,
                pipeline_id,
            )
            .ok()
        };
        let (vertex_entry_name, fragment_entry_name, bind_group_layouts) =
            resolved.unwrap_or_else(|| {
                (
                    descriptor
                        .vertex
                        .shader
                        .entry_point
                        .clone()
                        .unwrap_or_default(),
                    descriptor
                        .fragment
                        .as_ref()
                        .and_then(|fragment| fragment.shader.entry_point.clone()),
                    Vec::new(),
                )
            });
        let metal_bindings = metal_buffer_binding_map(&bind_group_layouts);
        let vertex_buffer_bindings =
            metal_vertex_buffer_binding_map(&descriptor.vertex.buffers, &metal_bindings);
        let framebuffer_fetch_color_slots = render_pipeline_framebuffer_fetch_color_slots(
            &descriptor,
            fragment_entry_name.as_deref(),
        );
        let (hal, backend_error) = if is_error {
            (None, None)
        } else {
            create_hal_render_pipeline(
                hal_device,
                &descriptor,
                &vertex_entry_name,
                fragment_entry_name.as_deref(),
                &metal_bindings,
                &vertex_buffer_bindings,
                &bind_group_layouts,
            )
        };
        let is_error = is_error || backend_error.is_some();
        (
            Self {
                inner: Arc::new(RenderPipelineInner {
                    _layout: descriptor.layout,
                    _vertex: descriptor.vertex,
                    _primitive: descriptor.primitive,
                    _depth_stencil: descriptor.depth_stencil,
                    _multisample: descriptor.multisample,
                    _fragment: descriptor.fragment,
                    vertex_entry_name,
                    fragment_entry_name,
                    metal_bindings,
                    vertex_buffer_bindings,
                    hal,
                    bind_group_layouts,
                    framebuffer_fetch_color_slots,
                    #[cfg(feature = "tiled")]
                    subpass_compatibility: None,
                    is_error,
                }),
            },
            backend_error,
        )
    }

    /// Creates a new subpass-compatible render pipeline.
    #[cfg(feature = "tiled")]
    pub(crate) fn new_subpass(
        descriptor: SubpassRenderPipelineDescriptor,
        is_error: bool,
        limits: Limits,
        features: &FeatureSet,
        hal_device: Option<&HalDevice>,
    ) -> (Self, Option<String>) {
        let pipeline_id = next_pipeline_id();
        let compatibility = SubpassPipelineCompatibility {
            pass_layout: Arc::clone(&descriptor.pass_layout),
            subpass_index: descriptor.subpass_index,
        };
        let subpass_color_attachment_indices = descriptor
            .pass_layout
            .descriptor()
            .subpasses
            .get(descriptor.subpass_index as usize)
            .map(|subpass| subpass.color_attachment_indices.as_slice());
        let resolved = if is_error {
            None
        } else {
            resolve_render_pipeline_descriptor_for_source(
                &descriptor.base,
                limits,
                features,
                subpass_color_attachment_indices,
                pipeline_id,
            )
            .ok()
        };
        let (vertex_entry_name, fragment_entry_name, bind_group_layouts) =
            resolved.unwrap_or_else(|| {
                (
                    descriptor
                        .base
                        .vertex
                        .shader
                        .entry_point
                        .clone()
                        .unwrap_or_default(),
                    descriptor
                        .base
                        .fragment
                        .as_ref()
                        .and_then(|fragment| fragment.shader.entry_point.clone()),
                    Vec::new(),
                )
            });
        let metal_bindings = metal_buffer_binding_map(&bind_group_layouts);
        let vertex_buffer_bindings =
            metal_vertex_buffer_binding_map(&descriptor.base.vertex.buffers, &metal_bindings);
        let framebuffer_fetch_color_slots = render_pipeline_framebuffer_fetch_color_slots(
            &descriptor.base,
            fragment_entry_name.as_deref(),
        );
        let subpass_color_slots = compute_subpass_color_slots(
            descriptor.pass_layout.descriptor(),
            descriptor.subpass_index,
        );
        let multisampling_error = if is_error {
            None
        } else {
            validate_subpass_pipeline_multisampling(
                &descriptor.base,
                descriptor.pass_layout.descriptor(),
                descriptor.subpass_index,
                &bind_group_layouts,
            )
        };
        let (hal, backend_error) = if is_error || multisampling_error.is_some() {
            (None, multisampling_error)
        } else {
            create_hal_subpass_render_pipeline(
                hal_device,
                &descriptor.base,
                &vertex_entry_name,
                fragment_entry_name.as_deref(),
                &metal_bindings,
                &vertex_buffer_bindings,
                &bind_group_layouts,
                descriptor.pass_layout.descriptor(),
                descriptor.subpass_index,
                &subpass_color_slots,
            )
        };
        let is_error = is_error || backend_error.is_some();
        (
            Self {
                inner: Arc::new(RenderPipelineInner {
                    _layout: descriptor.base.layout,
                    _vertex: descriptor.base.vertex,
                    _primitive: descriptor.base.primitive,
                    _depth_stencil: descriptor.base.depth_stencil,
                    _multisample: descriptor.base.multisample,
                    _fragment: descriptor.base.fragment,
                    vertex_entry_name,
                    fragment_entry_name,
                    metal_bindings,
                    vertex_buffer_bindings,
                    hal,
                    bind_group_layouts,
                    framebuffer_fetch_color_slots,
                    subpass_compatibility: Some(compatibility),
                    is_error,
                }),
            },
            backend_error,
        )
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns vertex entry name.
    #[must_use]
    pub fn vertex_entry_name(&self) -> &str {
        &self.inner.vertex_entry_name
    }

    /// Returns fragment entry name.
    #[must_use]
    pub fn fragment_entry_name(&self) -> Option<&str> {
        self.inner.fragment_entry_name.as_deref()
    }

    /// Returns the bind group layouts.
    #[must_use]
    pub fn bind_group_layouts(&self) -> &[Arc<BindGroupLayout>] {
        &self.inner.bind_group_layouts
    }

    /// Returns the HAL.
    pub(crate) fn hal(&self) -> Option<HalRenderPipeline> {
        self.inner.hal.clone()
    }

    /// Returns the metal bindings.
    pub(crate) fn metal_bindings(&self) -> &[MetalBufferBinding] {
        &self.inner.metal_bindings
    }

    /// Returns vertex buffer bindings.
    pub(crate) fn vertex_buffer_bindings(&self) -> &[MetalVertexBufferBinding] {
        &self.inner.vertex_buffer_bindings
    }

    /// Returns required vertex buffer count.
    #[must_use]
    pub(crate) fn required_vertex_buffer_count(&self) -> usize {
        self.inner._vertex.buffer_count
    }

    /// Returns vertex buffer layouts.
    #[must_use]
    pub(crate) fn vertex_buffer_layouts(&self) -> &[VertexBufferLayout] {
        &self.inner._vertex.buffers
    }

    /// Returns primitive state.
    #[must_use]
    pub(crate) fn primitive_state(&self) -> PrimitiveState {
        self.inner._primitive
    }

    /// Returns the attachment signature used for render pass compatibility checks.
    #[must_use]
    pub(crate) fn attachment_signature(&self) -> AttachmentSignature {
        AttachmentSignature {
            color_formats: self
                .inner
                ._fragment
                .as_ref()
                .map(|fragment| {
                    fragment
                        .targets
                        .iter()
                        .map(|target| (!target.format.is_undefined()).then_some(target.format))
                        .collect()
                })
                .unwrap_or_default(),
            depth_stencil_format: self.inner._depth_stencil.map(|depth| depth.format),
            sample_count: self.inner._multisample.count,
            depth_read_only: false,
            stencil_read_only: false,
        }
    }

    /// Returns true when this pipeline can write depth.
    pub(crate) fn writes_depth(&self) -> bool {
        self.inner
            ._depth_stencil
            .is_some_and(|depth| depth.depth_write_enabled == Some(true))
    }

    /// Returns true when this pipeline can write stencil.
    ///
    /// A pipeline writes stencil only when the write mask is non-zero **and** at
    /// least one non-culled face has a stencil operation other than `Keep`. The
    /// write mask alone does not imply a write — a state whose every operation is
    /// `Keep` (e.g. a depth/stencil *test* with `passOp = Keep`) leaves stencil
    /// untouched and is compatible with a read-only stencil attachment. Mirrors
    /// wgpu's `StencilState::is_read_only`.
    pub(crate) fn writes_stencil(&self) -> bool {
        let Some(depth) = self.inner._depth_stencil else {
            return false;
        };
        if depth.stencil_write_mask == 0 {
            return false;
        }
        let cull = self.inner._primitive.cull_mode;
        let front_writes = cull != CullMode::Front && stencil_face_writes(depth.stencil_front);
        let back_writes = cull != CullMode::Back && stencil_face_writes(depth.stencil_back);
        front_writes || back_writes
    }

    /// Returns subpass compatibility metadata.
    #[cfg(feature = "tiled")]
    #[allow(dead_code)]
    pub(crate) fn subpass_compatibility(&self) -> Option<&SubpassPipelineCompatibility> {
        self.inner.subpass_compatibility.as_ref()
    }
}

/// Validates render pipeline descriptor and returns a descriptive error on failure.
pub(crate) fn validate_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    limits: Limits,
    features: &FeatureSet,
) -> Option<String> {
    resolve_render_pipeline_descriptor_for_source(descriptor, limits, features, None, 0).err()
}

/// Validates subpass render pipeline descriptor and returns a descriptive error on failure.
#[cfg(feature = "tiled")]
pub(crate) fn validate_subpass_render_pipeline_descriptor(
    descriptor: &SubpassRenderPipelineDescriptor,
    limits: Limits,
    features: &FeatureSet,
) -> Option<String> {
    let subpass_color_attachment_indices = descriptor
        .pass_layout
        .descriptor()
        .subpasses
        .get(descriptor.subpass_index as usize)
        .map(|subpass| subpass.color_attachment_indices.as_slice());
    let (vertex_entry, fragment_entry, _) = match resolve_render_pipeline_descriptor_for_source(
        &descriptor.base,
        limits,
        features,
        subpass_color_attachment_indices,
        0,
    ) {
        Ok(resolved) => resolved,
        Err(message) => return Some(message),
    };
    validate_subpass_pipeline_has_no_immediates(
        &descriptor.base,
        &vertex_entry,
        fragment_entry.as_deref(),
    )
    .err()
}

/// Rejects subpass pipelines whose shader stages statically use
/// `var<immediate>` data (Block 94 Phase Review MAJOR 2). The tiled subpass
/// vendor extension has no `SetImmediates` surface -- the subpass encoder
/// records no immediates scratch and the Metal subpass draw path
/// (`encode_subpass_render_pass`) therefore has no snapshot to deliver --
/// so such a pipeline would silently read zeroes. Never silently wrong:
/// fail pipeline creation deterministically instead, as a documented
/// tiled-feature limitation (the standard render-pass path fully supports
/// immediates).
#[cfg(feature = "tiled")]
fn validate_subpass_pipeline_has_no_immediates(
    descriptor: &RenderPipelineDescriptor,
    vertex_entry: &str,
    fragment_entry: Option<&str>,
) -> Result<(), String> {
    const MESSAGE: &str = "subpass render pipelines do not support immediate data (var<immediate>)";
    // Passthrough (unreflected) modules bypass Tint reflection entirely and
    // are outside-spec vendor input; only WGSL-reflected stages are checked.
    if let Some(vertex_module) = descriptor.vertex.shader.module.reflected() {
        if vertex_module.immediate_data_size(vertex_entry)? > 0 {
            return Err(MESSAGE.to_owned());
        }
    }
    if let (Some(fragment), Some(fragment_entry)) = (&descriptor.fragment, fragment_entry) {
        if let Some(fragment_module) = fragment.shader.module.reflected() {
            if fragment_module.immediate_data_size(fragment_entry)? > 0 {
                return Err(MESSAGE.to_owned());
            }
        }
    }
    Ok(())
}

/// Alias for resolved render pipeline parts.
pub(crate) type ResolvedRenderPipelineParts = (String, Option<String>, Vec<Arc<BindGroupLayout>>);

fn resolve_render_pipeline_descriptor_for_source(
    descriptor: &RenderPipelineDescriptor,
    limits: Limits,
    features: &FeatureSet,
    subpass_color_attachment_indices: Option<&[u32]>,
    pipeline_id: u64,
) -> Result<ResolvedRenderPipelineParts, String> {
    #[cfg(feature = "shader-passthrough")]
    if render_pipeline_uses_shader_passthrough(descriptor) {
        return resolve_shader_passthrough_render_pipeline_descriptor(descriptor);
    }
    resolve_render_pipeline_descriptor(
        descriptor,
        limits,
        features,
        subpass_color_attachment_indices,
        pipeline_id,
    )
}

#[cfg(feature = "shader-passthrough")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderPassthroughKind {
    Msl,
    Spirv,
}

#[cfg(feature = "shader-passthrough")]
fn render_stage_passthrough_kind(module: &ShaderModule) -> Option<RenderPassthroughKind> {
    if module.msl_passthrough().is_some() {
        Some(RenderPassthroughKind::Msl)
    } else if module.spirv_passthrough().is_some() {
        Some(RenderPassthroughKind::Spirv)
    } else {
        None
    }
}

#[cfg(feature = "shader-passthrough")]
fn render_pipeline_uses_shader_passthrough(descriptor: &RenderPipelineDescriptor) -> bool {
    render_stage_passthrough_kind(&descriptor.vertex.shader.module).is_some()
        || descriptor.fragment.as_ref().is_some_and(|fragment| {
            render_stage_passthrough_kind(&fragment.shader.module).is_some()
        })
}

/// Creates HAL render pipeline and reports validation errors through the owning device.
pub(crate) fn create_hal_render_pipeline(
    hal_device: Option<&HalDevice>,
    descriptor: &RenderPipelineDescriptor,
    vertex_entry_name: &str,
    fragment_entry_name: Option<&str>,
    metal_bindings: &[MetalBufferBinding],
    vertex_buffer_bindings: &[MetalVertexBufferBinding],
    bind_group_layouts: &[Arc<BindGroupLayout>],
) -> (Option<HalRenderPipeline>, Option<String>) {
    create_hal_render_pipeline_with_subpass_color_slots(
        hal_device,
        descriptor,
        vertex_entry_name,
        fragment_entry_name,
        metal_bindings,
        vertex_buffer_bindings,
        bind_group_layouts,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn create_hal_render_pipeline_with_subpass_color_slots(
    hal_device: Option<&HalDevice>,
    descriptor: &RenderPipelineDescriptor,
    vertex_entry_name: &str,
    fragment_entry_name: Option<&str>,
    metal_bindings: &[MetalBufferBinding],
    vertex_buffer_bindings: &[MetalVertexBufferBinding],
    bind_group_layouts: &[Arc<BindGroupLayout>],
    subpass_color_slots: &[((u32, u32), u32)],
) -> (Option<HalRenderPipeline>, Option<String>) {
    let Some(hal_device) = hal_device else {
        return (None, None);
    };
    if matches!(hal_device.backend(), HalBackend::Noop) {
        return (Some(HalRenderPipeline::Noop), None);
    }
    // Validate Metal slot ranges up front so the Metal compiler never sees an
    // out-of-range slot (Metal rejects these at compile-time with a cryptic
    // message that is hard to trace back to the binding layout).
    if matches!(hal_device.backend(), HalBackend::Metal) {
        if let Err(message) = validate_metal_slot_ranges(metal_bindings) {
            return (None, Some(message));
        }
    }
    if descriptor.fragment.is_none() && descriptor.depth_stencil.is_none() {
        return (
            None,
            Some(
                "real render pipeline requires a fragment stage or depth-stencil state".to_owned(),
            ),
        );
    }
    let (shader, vertex_entry_point, fragment_entry_point, mut descriptor_bindings) =
        match select_render_shader_source(
            hal_device.backend(),
            descriptor,
            vertex_entry_name,
            fragment_entry_name,
            metal_bindings,
            vertex_buffer_bindings,
            subpass_color_slots,
            hal_device.vulkan_memory_model(),
            bind_group_layouts,
        ) {
            Ok(selection) => selection,
            Err(message) => return (None, Some(message)),
        };
    if matches!(hal_device.backend(), HalBackend::Vulkan) && !subpass_color_slots.is_empty() {
        match input_attachment_hal_bindings(bind_group_layouts, subpass_color_slots) {
            Ok(input_attachment_bindings) => descriptor_bindings.extend(input_attachment_bindings),
            Err(message) => return (None, Some(message)),
        }
    }
    let mut hal_descriptor =
        match hal_render_pipeline_descriptor(descriptor, vertex_buffer_bindings) {
            Ok(descriptor) => descriptor,
            Err(message) => return (None, Some(message)),
        };
    // On Vulkan, the `@builtin(position)` pixel-center polyfill (applied when the
    // fragment reads position under sample-rate shading) reconstructs the depth
    // in NDC space and needs the viewport depth range delivered as a fragment
    // push constant. Flag it so the HAL declares the push-constant range and
    // writes the viewport min/max depth at draw time. Matches the SPIR-V
    // generation decision in `select_render_shader_source`.
    if matches!(hal_device.backend(), HalBackend::Vulkan) {
        if let (Some(fragment), Some(fragment_entry_name)) =
            (descriptor.fragment.as_ref(), fragment_entry_name)
        {
            if let Some(fragment_module) = fragment.shader.module.reflected() {
                hal_descriptor.needs_frag_depth_range_push_constant = fragment_module
                    .fragment_needs_pixel_center_polyfill(
                        fragment_entry_name,
                        descriptor.multisample.count,
                    );
            }
        }
        // Block 94 S3: the pipeline's effective user-immediate byte budget
        // (same resolution the SPIR-V codegen used above via
        // `select_render_shader_source`) sizes the user prefix of the
        // combined push-constant block; the depth-range pair -- when the
        // polyfill flag above is set -- sits directly after it.
        match render_pipeline_layout_immediate_size(
            descriptor,
            vertex_entry_name,
            fragment_entry_name,
        ) {
            Ok(size) => hal_descriptor.user_immediate_size = size,
            Err(message) => return (None, Some(message)),
        }
    }
    match hal_device.create_render_pipeline(
        shader,
        &vertex_entry_point,
        fragment_entry_point.as_deref(),
        &hal_descriptor,
        &descriptor_bindings,
    ) {
        Ok(pipeline) => (Some(pipeline), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

/// Creates HAL subpass render pipeline through the regular HAL render-pipeline path.
#[cfg(feature = "tiled")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn create_hal_subpass_render_pipeline(
    hal_device: Option<&HalDevice>,
    descriptor: &RenderPipelineDescriptor,
    vertex_entry_name: &str,
    fragment_entry_name: Option<&str>,
    metal_bindings: &[MetalBufferBinding],
    vertex_buffer_bindings: &[MetalVertexBufferBinding],
    bind_group_layouts: &[Arc<BindGroupLayout>],
    pass_layout: &SubpassPassLayoutDescriptor,
    subpass_index: u32,
    subpass_color_slots: &[((u32, u32), u32)],
) -> (Option<HalRenderPipeline>, Option<String>) {
    if descriptor.fragment.is_none() {
        return (
            None,
            Some("subpass render pipeline requires a fragment stage".to_owned()),
        );
    }
    let Some(hal_device) = hal_device else {
        return (None, None);
    };
    if matches!(hal_device.backend(), HalBackend::Noop) {
        return (Some(HalRenderPipeline::Noop), None);
    }
    match hal_device.backend() {
        HalBackend::Vulkan => create_hal_vulkan_subpass_render_pipeline(
            hal_device,
            descriptor,
            vertex_entry_name,
            fragment_entry_name,
            metal_bindings,
            vertex_buffer_bindings,
            bind_group_layouts,
            pass_layout,
            subpass_index,
            subpass_color_slots,
        ),
        _ => create_hal_non_vulkan_subpass_render_pipeline(
            hal_device,
            descriptor,
            vertex_entry_name,
            fragment_entry_name,
            metal_bindings,
            vertex_buffer_bindings,
            bind_group_layouts,
            pass_layout,
            subpass_index,
            subpass_color_slots,
        ),
    }
}

#[cfg(feature = "tiled")]
#[allow(clippy::too_many_arguments)]
fn create_hal_non_vulkan_subpass_render_pipeline(
    hal_device: &HalDevice,
    descriptor: &RenderPipelineDescriptor,
    vertex_entry_name: &str,
    fragment_entry_name: Option<&str>,
    metal_bindings: &[MetalBufferBinding],
    vertex_buffer_bindings: &[MetalVertexBufferBinding],
    bind_group_layouts: &[Arc<BindGroupLayout>],
    pass_layout: &SubpassPassLayoutDescriptor,
    subpass_index: u32,
    subpass_color_slots: &[((u32, u32), u32)],
) -> (Option<HalRenderPipeline>, Option<String>) {
    if matches!(hal_device.backend(), HalBackend::Metal) {
        if let Err(message) = validate_metal_slot_ranges(metal_bindings) {
            return (None, Some(message));
        }
    }
    let (shader, vertex_entry_point, fragment_entry_point, descriptor_bindings) =
        match select_render_shader_source(
            hal_device.backend(),
            descriptor,
            vertex_entry_name,
            fragment_entry_name,
            metal_bindings,
            vertex_buffer_bindings,
            subpass_color_slots,
            hal_device.vulkan_memory_model(),
            bind_group_layouts,
        ) {
            Ok(selection) => selection,
            Err(message) => return (None, Some(message)),
        };
    let hal_descriptor = match hal_subpass_render_pipeline_descriptor(
        descriptor,
        vertex_buffer_bindings,
        pass_layout,
        subpass_index,
        subpass_color_slots,
        hal_device.backend(),
    ) {
        Ok(descriptor) => descriptor,
        Err(message) => return (None, Some(message)),
    };
    match hal_device.create_render_pipeline(
        shader,
        &vertex_entry_point,
        fragment_entry_point.as_deref(),
        &hal_descriptor,
        &descriptor_bindings,
    ) {
        Ok(pipeline) => (Some(pipeline), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

#[cfg(feature = "tiled")]
#[allow(clippy::too_many_arguments)]
fn create_hal_vulkan_subpass_render_pipeline(
    hal_device: &HalDevice,
    descriptor: &RenderPipelineDescriptor,
    vertex_entry_name: &str,
    fragment_entry_name: Option<&str>,
    metal_bindings: &[MetalBufferBinding],
    vertex_buffer_bindings: &[MetalVertexBufferBinding],
    bind_group_layouts: &[Arc<BindGroupLayout>],
    pass_layout: &SubpassPassLayoutDescriptor,
    subpass_index: u32,
    subpass_color_slots: &[((u32, u32), u32)],
) -> (Option<HalRenderPipeline>, Option<String>) {
    let (shader, vertex_entry_point, fragment_entry_point, mut descriptor_bindings) =
        match select_render_shader_source(
            hal_device.backend(),
            descriptor,
            vertex_entry_name,
            fragment_entry_name,
            metal_bindings,
            vertex_buffer_bindings,
            subpass_color_slots,
            hal_device.vulkan_memory_model(),
            bind_group_layouts,
        ) {
            Ok(selection) => selection,
            Err(message) => return (None, Some(message)),
        };
    match input_attachment_hal_bindings(bind_group_layouts, subpass_color_slots) {
        Ok(input_attachment_bindings) => descriptor_bindings.extend(input_attachment_bindings),
        Err(message) => return (None, Some(message)),
    }
    let hal_descriptor = match hal_subpass_render_pipeline_descriptor(
        descriptor,
        vertex_buffer_bindings,
        pass_layout,
        subpass_index,
        subpass_color_slots,
        HalBackend::Vulkan,
    ) {
        Ok(descriptor) => descriptor,
        Err(message) => return (None, Some(message)),
    };
    let hal_pass_layout = hal_subpass_pass_layout(pass_layout);
    match hal_device.create_subpass_render_pipeline(
        shader,
        &vertex_entry_point,
        fragment_entry_point.as_deref(),
        &hal_descriptor,
        &descriptor_bindings,
        &hal_pass_layout,
        subpass_index,
    ) {
        Ok(pipeline) => (Some(pipeline), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

#[cfg(feature = "tiled")]
fn input_attachment_hal_bindings(
    bind_group_layouts: &[Arc<BindGroupLayout>],
    subpass_color_slots: &[((u32, u32), u32)],
) -> Result<Vec<HalDescriptorBinding>, String> {
    let mut bindings = Vec::new();
    for (group_index, layout) in bind_group_layouts.iter().enumerate() {
        let group = u32::try_from(group_index)
            .map_err(|_| "input attachment bind group index is too large".to_owned())?;
        for entry in layout.entries() {
            if matches!(entry.kind, Some(BindingLayoutKind::InputAttachment { .. })) {
                let color_slot = subpass_color_slots
                    .iter()
                    .find_map(|&((input_group, input_binding), source_attachment)| {
                        (input_group == group && input_binding == entry.binding)
                            .then_some(source_attachment)
                    })
                    .ok_or_else(|| {
                        "subpass input attachment binding is missing from subpass layout".to_owned()
                    })?;
                bindings.push(HalDescriptorBinding {
                    group,
                    binding: entry.binding,
                    kind: HalDescriptorBindingKind::InputAttachment { color_slot },
                });
            }
        }
    }
    Ok(bindings)
}

#[cfg(not(feature = "tiled"))]
fn input_attachment_hal_bindings(
    _bind_group_layouts: &[Arc<BindGroupLayout>],
    _subpass_color_slots: &[((u32, u32), u32)],
) -> Result<Vec<HalDescriptorBinding>, String> {
    Ok(Vec::new())
}

/// Returns `true` when any bind group layout declares a multisampled input
/// attachment, so the Vulkan fragment SPIR-V must emit multisampled
/// `SubpassData` (the 2-arg `inputAttachmentLoad(ia, sample_index)` overload).
#[cfg(feature = "tiled")]
fn pipeline_has_multisampled_input_attachment(bind_group_layouts: &[Arc<BindGroupLayout>]) -> bool {
    bind_group_layouts.iter().any(|layout| {
        layout.entries().iter().any(|entry| {
            matches!(
                entry.kind,
                Some(BindingLayoutKind::InputAttachment {
                    multisampled: true,
                    ..
                })
            )
        })
    })
}

#[cfg(not(feature = "tiled"))]
fn pipeline_has_multisampled_input_attachment(_b: &[Arc<BindGroupLayout>]) -> bool {
    false
}

/// Validates MSAA consistency between a subpass render pipeline, its bind group
/// layouts, and the subpass attachment sample counts. Returns `Some(message)`
/// on the first violation, `None` when consistent.
#[cfg(feature = "tiled")]
fn validate_subpass_pipeline_multisampling(
    base: &RenderPipelineDescriptor,
    pass_layout: &SubpassPassLayoutDescriptor,
    subpass_index: u32,
    bind_group_layouts: &[Arc<BindGroupLayout>],
) -> Option<String> {
    let subpass = pass_layout.subpasses.get(subpass_index as usize)?;

    // C-1 — each input attachment's multisampled flag must match the sample count
    // of the source attachment it reads.
    for input in &subpass.input_attachments {
        let Some(layout) = bind_group_layouts.get(input.group as usize) else {
            continue;
        };
        let Some(multisampled) = layout.entries().iter().find_map(|entry| {
            if entry.binding != input.binding {
                return None;
            }
            match entry.kind {
                Some(BindingLayoutKind::InputAttachment { multisampled, .. }) => Some(multisampled),
                _ => None,
            }
        }) else {
            continue;
        };
        let source_sample_count = if input.source_attachment == DEPTH_STENCIL_ATTACHMENT_INDEX {
            pass_layout
                .depth_stencil_attachment
                .as_ref()
                .map(|attachment| attachment.sample_count)
        } else {
            pass_layout
                .color_attachments
                .get(input.source_attachment as usize)
                .map(|attachment| attachment.sample_count)
        };
        let Some(source_sample_count) = source_sample_count else {
            continue;
        };
        if multisampled != (source_sample_count > 1) {
            return Some(
                "subpass input attachment multisampled flag must match its source attachment sample count"
                    .to_owned(),
            );
        }
    }

    // C-2 — every attachment written by the subpass must match the pipeline's
    // rasterization sample count.
    for &color_index in &subpass.color_attachment_indices {
        if let Some(attachment) = pass_layout.color_attachments.get(color_index as usize) {
            if attachment.sample_count != base.multisample.count {
                return Some(
                    "subpass render pipeline multisample count must match the subpass attachment sample count"
                        .to_owned(),
                );
            }
        }
    }
    if subpass.uses_depth_stencil {
        if let Some(attachment) = &pass_layout.depth_stencil_attachment {
            if attachment.sample_count != base.multisample.count {
                return Some(
                    "subpass render pipeline multisample count must match the subpass attachment sample count"
                        .to_owned(),
                );
            }
        }
    }

    None
}

/// Selects the HAL shader source for a render pipeline.
// Render-stage source selection legitimately needs backend, descriptor, both
// entry names, the Metal binding/vertex-buffer tables, and the
// Vulkan Memory Model flag (F-112) — grouping them into a struct would only add
// indirection for a single crate-private call path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_render_shader_source(
    backend: HalBackend,
    descriptor: &RenderPipelineDescriptor,
    vertex_entry_name: &str,
    fragment_entry_name: Option<&str>,
    metal_bindings: &[MetalBufferBinding],
    vertex_buffer_bindings: &[MetalVertexBufferBinding],
    subpass_color_slots: &[((u32, u32), u32)],
    vulkan_memory_model: bool,
    bind_group_layouts: &[Arc<BindGroupLayout>],
) -> Result<
    (
        HalShaderSource,
        String,
        Option<String>,
        Vec<HalDescriptorBinding>,
    ),
    String,
> {
    let fragment = descriptor.fragment.as_ref();
    let vertex_pipeline_constants = pipeline_constant_map(&descriptor.vertex.shader.constants);
    let fragment_pipeline_constants =
        fragment.map(|fragment| pipeline_constant_map(&fragment.shader.constants));
    match backend {
        HalBackend::Metal => {
            #[cfg(feature = "shader-passthrough")]
            if descriptor
                .vertex
                .shader
                .module
                .spirv_passthrough()
                .is_some()
                || fragment
                    .is_some_and(|fragment| fragment.shader.module.spirv_passthrough().is_some())
            {
                return Err("SPIR-V passthrough shader requires the Vulkan backend".to_owned());
            }
            #[cfg(feature = "shader-passthrough")]
            if let Some((vertex_source, _entries)) =
                descriptor.vertex.shader.module.msl_passthrough()
            {
                let fragment_source = match fragment {
                    Some(fragment) => {
                        let Some((source, _entries)) = fragment.shader.module.msl_passthrough()
                        else {
                            return Err(
                                "render shader passthrough requires all stages to be passthrough modules of the same kind"
                                    .to_owned(),
                            );
                        };
                        Some(source.to_owned())
                    }
                    None => None,
                };
                return Ok((
                    HalShaderSource::MslStages {
                        vertex: vertex_source.to_owned(),
                        fragment: fragment_source,
                    },
                    vertex_entry_name.to_owned(),
                    fragment_entry_name.map(str::to_owned),
                    Vec::new(),
                ));
            }
            #[cfg(feature = "shader-passthrough")]
            if fragment.is_some_and(|fragment| fragment.shader.module.msl_passthrough().is_some()) {
                return Err(
                    "render shader passthrough requires all stages to be passthrough modules of the same kind"
                        .to_owned(),
                );
            }
            let module =
                descriptor.vertex.shader.module.reflected().ok_or_else(|| {
                    "render pipeline requires a reflected shader module".to_owned()
                })?;
            // Build separate per-stage binding maps so each stage's codegen
            // receives the correct per-kind slot indices for its own index space.
            let msl_vertex_binding_map = frontend::MslBindingMap {
                resources: msl_stage_resource_bindings(metal_bindings, true),
            };
            let msl_fragment_binding_map = frontend::MslBindingMap {
                resources: msl_stage_resource_bindings(metal_bindings, false),
            };
            let msl_vertex_buffers =
                msl_vertex_buffer_bindings(&descriptor.vertex.buffers, vertex_buffer_bindings)?;
            let force_point_size =
                matches!(descriptor.primitive.topology, PrimitiveTopology::PointList);
            // The pipeline's effective user-immediate byte budget (Block 94,
            // 0..=64): the explicit layout's `immediate_size`, or -- for auto
            // layouts, mirroring Dawn `CreateDefault`
            // (`dawn/native/PipelineLayout.cpp:588-590,616`) -- the max of
            // the stages' reflected immediate usage. This is the boundary
            // after which the fragment frag-depth clamp range is appended
            // (see `yawgpu_tint::Program::generate_msl`'s doc comment).
            let user_immediate_size = render_pipeline_layout_immediate_size(
                descriptor,
                vertex_entry_name,
                fragment_entry_name,
            )?;
            let vertex = module.generate_render_vertex_msl(
                vertex_entry_name,
                &msl_vertex_binding_map,
                &msl_vertex_buffers,
                force_point_size,
                &vertex_pipeline_constants,
                user_immediate_size,
            )?;
            let fragment = match (
                fragment,
                fragment_entry_name,
                fragment_pipeline_constants.as_ref(),
            ) {
                (Some(fragment), Some(fragment_entry_name), Some(fragment_pipeline_constants)) => {
                    let fragment_module =
                        if Arc::ptr_eq(&descriptor.vertex.shader.module, &fragment.shader.module) {
                            module
                        } else {
                            fragment.shader.module.reflected().ok_or_else(|| {
                                "render pipeline requires a reflected fragment shader module"
                                    .to_owned()
                            })?
                        };
                    Some(fragment_module.generate_render_fragment_msl(
                        fragment_entry_name,
                        &msl_fragment_binding_map,
                        subpass_color_slots,
                        fragment_pipeline_constants,
                        descriptor.multisample.mask,
                        user_immediate_size,
                    )?)
                }
                (None, None, None) => None,
                _ => {
                    return Err(
                        "real render pipeline fragment state and entry point must match".to_owned(),
                    );
                }
            };
            // Collect metal indices in the same order as vertex_buffer_mappings
            // passed to shader codegen (i.e. the order of msl_vertex_buffers).
            // These are the metal buffer slot ids for the `buffer_sizeN` fields
            // appended after storage-array size fields in `_mslBufferSizes`.
            let vertex_buffer_metal_indices: Vec<u32> =
                msl_vertex_buffers.iter().map(|b| b.metal_index).collect();
            // Vertex never carries the frag-depth clamp -- its block is just
            // the user-immediate prefix when the vertex entry point uses any
            // immediates.
            let vertex_immediates = vertex
                .immediate_slot
                .map(|slot| HalMslImmediates::new(slot, user_immediate_size, None));
            // Fragment's block appends the 8-byte clamp range right after
            // the user prefix when this pipeline clamps frag_depth (Block
            // 94 "Immediates block layout"); `frag_depth_clamp_slot` and
            // `immediate_slot` are always the same value when both are
            // `Some` (same combined block, same Metal slot).
            let fragment_immediates = fragment.as_ref().and_then(|fragment| {
                fragment.immediate_slot.map(|slot| {
                    let frag_depth_clamp_offset = fragment
                        .frag_depth_clamp_slot
                        .is_some()
                        .then_some(user_immediate_size);
                    let block_size =
                        frag_depth_clamp_offset.map_or(user_immediate_size, |offset| offset + 8);
                    HalMslImmediates::new(slot, block_size, frag_depth_clamp_offset)
                })
            });
            Ok((
                HalShaderSource::MslStagesWithBufferSizes {
                    vertex: vertex.source,
                    fragment: fragment.as_ref().map(|fragment| fragment.source.clone()),
                    vertex_buffer_sizes_slot: vertex.buffer_sizes_slot,
                    vertex_buffer_size_bindings: hal_msl_buffer_size_bindings(
                        &vertex.buffer_size_bindings,
                    ),
                    fragment_buffer_sizes_slot: fragment
                        .as_ref()
                        .and_then(|fragment| fragment.buffer_sizes_slot),
                    fragment_buffer_size_bindings: hal_msl_buffer_size_bindings(
                        fragment
                            .as_ref()
                            .map(|fragment| fragment.buffer_size_bindings.as_slice())
                            .unwrap_or(&[]),
                    ),
                    vertex_immediates,
                    fragment_immediates,
                    vertex_buffer_metal_indices,
                },
                vertex.entry_point,
                fragment.map(|fragment| fragment.entry_point),
                Vec::new(),
            ))
        }
        HalBackend::Vulkan => {
            #[cfg(feature = "shader-passthrough")]
            if let Some(vertex_words) = descriptor.vertex.shader.module.spirv_passthrough() {
                let fragment_words = match fragment {
                    Some(fragment) => {
                        let Some(words) = fragment.shader.module.spirv_passthrough() else {
                            return Err(
                                "render shader passthrough requires all stages to be passthrough modules of the same kind"
                                    .to_owned(),
                            );
                        };
                        Some(words.to_vec())
                    }
                    None => None,
                };
                return Ok((
                    HalShaderSource::SpirVStages {
                        vertex: vertex_words.to_vec(),
                        fragment: fragment_words,
                    },
                    vertex_entry_name.to_owned(),
                    fragment_entry_name.map(str::to_owned),
                    hal_descriptor_bindings(metal_bindings),
                ));
            }
            #[cfg(feature = "shader-passthrough")]
            if render_stage_passthrough_kind(&descriptor.vertex.shader.module)
                == Some(RenderPassthroughKind::Msl)
                || fragment.is_some_and(|fragment| {
                    render_stage_passthrough_kind(&fragment.shader.module)
                        == Some(RenderPassthroughKind::Msl)
                })
            {
                return Err("MSL passthrough shader requires the Metal backend".to_owned());
            }
            #[cfg(feature = "shader-passthrough")]
            if fragment.is_some_and(|fragment| fragment.shader.module.spirv_passthrough().is_some())
            {
                return Err(
                    "render shader passthrough requires all stages to be passthrough modules of the same kind"
                        .to_owned(),
                );
            }
            if let Some(message) = vulkan_external_texture_rejection(metal_bindings) {
                return Err(message);
            }
            let vertex_module = descriptor.vertex.shader.module.reflected().ok_or_else(|| {
                "render pipeline requires a reflected vertex shader module".to_owned()
            })?;
            let framebuffer_fetch_descriptor_set =
                framebuffer_fetch_descriptor_set(bind_group_layouts)?;
            // `multisampled_input_attachment` is a module-wide Tint SPIR-V option,
            // so it must be identical for every stage generated from the module —
            // Tint validates the whole module's `inputAttachmentLoad` overloads
            // against it even when generating the vertex entry point. Compute it
            // once and pass it to both stages (a vertex/fragment pair that share a
            // module with a 2-arg `inputAttachmentLoad(ia, sample_index)` would
            // otherwise fail the vertex generation).
            let multisampled_input_attachment =
                pipeline_has_multisampled_input_attachment(bind_group_layouts);
            // Pixel-center polyfill for `@builtin(position)` under sample-rate
            // shading (see `ReflectedModule::fragment_needs_pixel_center_polyfill`).
            // The polyfill carries a `center_pos` inter-stage varying from the
            // vertex stage to the fragment stage, so the SAME free location must
            // be passed to both stages' SPIR-V generation. Computed once here,
            // before the vertex stage is generated. Mirrors Dawn's Vulkan backend.
            let polyfill_pixel_center = match (fragment, fragment_entry_name) {
                (Some(fragment), Some(fragment_entry_name)) => fragment
                    .shader
                    .module
                    .reflected()
                    .filter(|fragment_module| {
                        fragment_module.fragment_needs_pixel_center_polyfill(
                            fragment_entry_name,
                            descriptor.multisample.count,
                        )
                    })
                    .map(|_| vertex_module.free_inter_stage_location(vertex_entry_name)),
                _ => None,
            };
            // The pipeline's effective user-immediate byte budget (Block 94
            // S3, same resolution as the Metal arm): the internal depth-range
            // immediates (pixel-center polyfill) are rebased directly after
            // this region in the push-constant block, mirroring Dawn
            // (`dawn/native/vulkan/ShaderModuleVk.cpp:349-355`).
            let user_immediate_size = render_pipeline_layout_immediate_size(
                descriptor,
                vertex_entry_name,
                fragment_entry_name,
            )?;
            let vertex = vertex_module.generate_spirv(
                vertex_entry_name,
                frontend::ShaderStage::Vertex,
                &vertex_pipeline_constants,
                vulkan_memory_model,
                framebuffer_fetch_descriptor_set,
                multisampled_input_attachment,
                polyfill_pixel_center,
                user_immediate_size,
            )?;
            let fragment_color_slots = match (fragment, fragment_entry_name) {
                (Some(fragment), Some(fragment_entry_name)) => fragment
                    .shader
                    .module
                    .reflected()
                    .ok_or_else(|| {
                        "render pipeline requires a reflected fragment shader module".to_owned()
                    })?
                    .fragment_color_inputs(fragment_entry_name),
                _ => Vec::new(),
            };
            let fragment = match (fragment, fragment_entry_name) {
                (Some(fragment), Some(fragment_entry_name)) => {
                    let fragment_module = fragment.shader.module.reflected().ok_or_else(|| {
                        "render pipeline requires a reflected fragment shader module".to_owned()
                    })?;
                    Some(fragment_module.generate_spirv(
                        fragment_entry_name,
                        frontend::ShaderStage::Fragment,
                        fragment_pipeline_constants.as_ref().ok_or_else(|| {
                            "render pipeline fragment constants were not resolved".to_owned()
                        })?,
                        vulkan_memory_model,
                        framebuffer_fetch_descriptor_set,
                        multisampled_input_attachment,
                        polyfill_pixel_center,
                        user_immediate_size,
                    )?)
                }
                (None, None) => None,
                _ => {
                    return Err(
                        "real render pipeline fragment state and entry point must match".to_owned(),
                    );
                }
            };
            let mut descriptor_bindings = hal_descriptor_bindings(metal_bindings);
            descriptor_bindings.extend(fragment_color_slots.into_iter().map(|slot| {
                HalDescriptorBinding {
                    group: framebuffer_fetch_descriptor_set,
                    binding: slot,
                    kind: HalDescriptorBindingKind::InputAttachment { color_slot: slot },
                }
            }));
            Ok((
                HalShaderSource::SpirVStages { vertex, fragment },
                vertex_entry_name.to_owned(),
                fragment_entry_name.map(str::to_owned),
                descriptor_bindings,
            ))
        }
        #[cfg(feature = "gles")]
        HalBackend::Gles => {
            let vertex_module = descriptor.vertex.shader.module.reflected().ok_or_else(|| {
                "render pipeline requires a reflected vertex shader module".to_owned()
            })?;
            let vertex_glsl = vertex_module.generate_glsl(
                vertex_entry_name,
                frontend::ShaderStage::Vertex,
                &vertex_pipeline_constants,
            )?;
            let fragment_glsl = match (fragment, fragment_entry_name) {
                (Some(fragment), Some(fragment_entry_name)) => {
                    let fragment_module = fragment.shader.module.reflected().ok_or_else(|| {
                        "render pipeline requires a reflected fragment shader module".to_owned()
                    })?;
                    Some(
                        fragment_module
                            .generate_glsl(
                                fragment_entry_name,
                                frontend::ShaderStage::Fragment,
                                fragment_pipeline_constants.as_ref().ok_or_else(|| {
                                    "render pipeline fragment constants were not resolved"
                                        .to_owned()
                                })?,
                            )?
                            .source,
                    )
                }
                (None, None) => None,
                _ => {
                    return Err(
                        "real render pipeline fragment state and entry point must match".to_owned(),
                    );
                }
            };
            Ok((
                HalShaderSource::GlslStages {
                    vertex: vertex_glsl.source,
                    fragment: fragment_glsl,
                },
                vertex_entry_name.to_owned(),
                fragment_entry_name.map(str::to_owned),
                hal_descriptor_bindings(metal_bindings),
            ))
        }
        HalBackend::Noop => Err("Noop backend does not create HAL shader sources".to_owned()),
        _ => Err("unsupported backend does not create HAL shader sources".to_owned()),
    }
}

/// Returns MSL resource bindings projected to a single render stage.
///
/// `vertex = true` selects `vertex_metal_index`; `vertex = false` selects
/// `fragment_metal_index`.  Entries that have no index for the requested stage
/// are omitted, so only bindings that the stage actually uses are included.
/// When both per-stage indices are `None` (compute-style flat map) the flat
/// `metal_index` is used for both stages as a fallback.
pub(crate) fn msl_stage_resource_bindings(
    bindings: &[MetalBufferBinding],
    vertex: bool,
) -> Vec<frontend::MslResourceBinding> {
    bindings
        .iter()
        .filter_map(|binding| {
            // Choose the stage-specific index; fall back to flat metal_index when
            // no per-stage indices are stored (backwards compat, or compute maps).
            let metal_index = if vertex {
                match binding.vertex_metal_index {
                    Some(idx) => idx,
                    None if binding.fragment_metal_index.is_none() => binding.metal_index,
                    None => return None, // not visible to vertex stage
                }
            } else {
                match binding.fragment_metal_index {
                    Some(idx) => idx,
                    None if binding.vertex_metal_index.is_none() => binding.metal_index,
                    None => return None, // not visible to fragment stage
                }
            };
            let ext_params_buffer_slot = if vertex {
                binding.ext_params_vertex_buffer_slot
            } else {
                binding.ext_params_fragment_buffer_slot
            }
            .or(binding.ext_params_buffer_slot);
            Some(frontend::MslResourceBinding {
                group: binding.group,
                binding: binding.binding,
                metal_index,
                ext_params_buffer_slot,
                kind: match binding.kind {
                    MetalBindingKind::Buffer(_) => frontend::MslResourceBindingKind::Buffer,
                    MetalBindingKind::Texture | MetalBindingKind::StorageTexture { .. } => {
                        frontend::MslResourceBindingKind::Texture
                    }
                    MetalBindingKind::Sampler => frontend::MslResourceBindingKind::Sampler,
                    MetalBindingKind::ExternalTexture => {
                        frontend::MslResourceBindingKind::ExternalTexture
                    }
                },
            })
        })
        .collect()
}

/// Returns metal vertex buffer binding map.
///
/// Vertex buffers share the `[[buffer(N)]]` index space with bind-group
/// buffers in the vertex stage.  The start slot must therefore be placed
/// immediately after the last vertex-stage buffer-space slot used by the
/// bind-group layout (i.e. `vertex_stage_buffer_count(metal_bindings)`), not
/// after the total number of entries in the flat binding list.
pub(crate) fn metal_vertex_buffer_binding_map(
    layouts: &[VertexBufferLayout],
    metal_bindings: &[MetalBufferBinding],
) -> Vec<MetalVertexBufferBinding> {
    let start = vertex_stage_buffer_count(metal_bindings);
    layouts
        .iter()
        .enumerate()
        .filter(|(_, layout)| layout.used)
        .enumerate()
        .filter_map(|slot| {
            let (metal_slot, (slot, _)) = slot;
            Some(MetalVertexBufferBinding {
                slot: u32::try_from(slot).ok()?,
                metal_index: u32::try_from(start.checked_add(metal_slot)?).ok()?,
            })
        })
        .collect()
}

/// Returns msl vertex buffer bindings.
pub(crate) fn msl_vertex_buffer_bindings(
    layouts: &[VertexBufferLayout],
    bindings: &[MetalVertexBufferBinding],
) -> Result<Vec<frontend::MslVertexBufferBinding>, String> {
    layouts
        .iter()
        .filter(|layout| layout.used)
        .zip(bindings)
        .map(|(layout, binding)| {
            Ok(frontend::MslVertexBufferBinding {
                slot: binding.slot,
                metal_index: binding.metal_index,
                array_stride: layout.array_stride,
                step_mode: match layout.step_mode {
                    VertexStepMode::Vertex => frontend::MslVertexStepMode::Vertex,
                    VertexStepMode::Instance => frontend::MslVertexStepMode::Instance,
                },
                attributes: layout
                    .attributes
                    .iter()
                    .map(|attribute| {
                        Ok(frontend::MslVertexAttribute {
                            shader_location: attribute.shader_location,
                            offset: attribute.offset,
                            format: msl_vertex_format(attribute.format)?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            })
        })
        .collect()
}

/// Returns HAL render pipeline descriptor.
pub(crate) fn hal_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    bindings: &[MetalVertexBufferBinding],
) -> Result<HalRenderPipelineDescriptor, String> {
    let color_targets = descriptor
        .fragment
        .as_ref()
        .map(|fragment| {
            fragment
                .targets
                .iter()
                .map(|target| {
                    if target.format.is_undefined() {
                        Ok(None)
                    } else {
                        hal_color_target_state(target).map(Some)
                    }
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?
        .unwrap_or_default();
    let vertex_buffers = descriptor
        .vertex
        .buffers
        .iter()
        .filter(|layout| layout.used)
        .zip(bindings)
        .map(|(layout, binding)| {
            Ok(HalVertexBufferLayout {
                slot: binding.slot,
                array_stride: layout.array_stride,
                step_mode: match layout.step_mode {
                    VertexStepMode::Vertex => HalVertexStepMode::Vertex,
                    VertexStepMode::Instance => HalVertexStepMode::Instance,
                },
                attributes: layout
                    .attributes
                    .iter()
                    .map(|attribute| {
                        Ok(HalVertexAttribute {
                            format: hal_vertex_format(attribute.format),
                            offset: attribute.offset,
                            shader_location: attribute.shader_location,
                            metal_buffer_index: binding.metal_index,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(HalRenderPipelineDescriptor {
        sample_count: descriptor.multisample.count,
        sample_mask: descriptor.multisample.mask,
        alpha_to_coverage_enabled: descriptor.multisample.alpha_to_coverage_enabled,
        color_targets,
        depth_stencil: descriptor.depth_stencil.map(hal_depth_stencil_state),
        vertex_buffers,
        primitive_topology: hal_primitive_topology(descriptor.primitive.topology),
        front_face: hal_front_face(descriptor.primitive.front_face),
        cull_mode: hal_cull_mode(descriptor.primitive.cull_mode),
        unclipped_depth: descriptor.primitive.unclipped_depth,
        // Set by the caller once the shader stages are generated (Vulkan only).
        needs_frag_depth_range_push_constant: false,
        user_immediate_size: 0,
    })
}

#[cfg(feature = "tiled")]
fn hal_subpass_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    bindings: &[MetalVertexBufferBinding],
    pass_layout: &SubpassPassLayoutDescriptor,
    subpass_index: u32,
    subpass_color_slots: &[((u32, u32), u32)],
    backend: HalBackend,
) -> Result<HalRenderPipelineDescriptor, String> {
    let mut hal = hal_render_pipeline_descriptor(descriptor, bindings)?;
    let Some(fragment) = descriptor.fragment.as_ref() else {
        return Ok(hal);
    };
    let Some(subpass) = pass_layout.subpasses.get(subpass_index as usize) else {
        return Ok(hal);
    };
    let max_written_slot = subpass.color_attachment_indices.iter().copied().max();
    let max_input_slot = subpass_color_slots
        .iter()
        .map(|&(_, source_attachment)| source_attachment)
        .filter(|&source_attachment| source_attachment != DEPTH_STENCIL_ATTACHMENT_INDEX)
        .max();
    let Some(max_color_slot) = max_written_slot.max(max_input_slot) else {
        hal.color_targets.clear();
        return Ok(hal);
    };
    let span = usize::try_from(max_color_slot)
        .map_err(|_| "subpass color attachment slot is too large".to_owned())?
        .checked_add(1)
        .ok_or_else(|| "subpass color attachment slot count overflows".to_owned())?;
    let flat_targets = fragment.targets.len() > subpass.color_attachment_indices.len();
    let mut color_targets = vec![None; span];
    if flat_targets {
        for (slot, target) in hal.color_targets.iter().copied().enumerate().take(span) {
            color_targets[slot] = target;
        }
    } else {
        for (local_slot, &global_slot) in subpass.color_attachment_indices.iter().enumerate() {
            let global_slot = usize::try_from(global_slot)
                .map_err(|_| "subpass color attachment slot is too large".to_owned())?;
            if global_slot >= color_targets.len() {
                return Err("subpass color attachment slot exceeds derived target span".to_owned());
            }
            color_targets[global_slot] = hal.color_targets.get(local_slot).copied().flatten();
        }
    }
    if matches!(backend, HalBackend::Metal) {
        for &(_, source_attachment) in subpass_color_slots {
            if source_attachment == DEPTH_STENCIL_ATTACHMENT_INDEX {
                continue;
            }
            let slot = usize::try_from(source_attachment)
                .map_err(|_| "subpass input attachment slot is too large".to_owned())?;
            let attachment = pass_layout.color_attachments.get(slot).ok_or_else(|| {
                "subpass input attachment source color slot is out of range".to_owned()
            })?;
            if slot >= color_targets.len() {
                color_targets.resize(slot + 1, None);
            }
            color_targets[slot] = Some(HalColorTargetState {
                format: hal_texture_format(attachment.format),
                blend: None,
                write_mask: 0,
            });
        }
    }
    hal.color_targets = color_targets;
    Ok(hal)
}

fn hal_front_face(front_face: FrontFace) -> HalFrontFace {
    match front_face {
        FrontFace::Ccw => HalFrontFace::Ccw,
        FrontFace::Cw => HalFrontFace::Cw,
    }
}

fn hal_cull_mode(cull_mode: CullMode) -> HalCullMode {
    match cull_mode {
        CullMode::None => HalCullMode::None,
        CullMode::Front => HalCullMode::Front,
        CullMode::Back => HalCullMode::Back,
    }
}

fn hal_color_target_state(target: &ColorTargetState) -> Result<HalColorTargetState, String> {
    Ok(HalColorTargetState {
        format: hal_texture_format(target.format),
        blend: target.blend.map(hal_blend_state),
        write_mask: u32::try_from(target.write_mask)
            .map_err(|_| "color target write mask is too large for HAL".to_owned())?,
    })
}

fn hal_blend_state(state: BlendState) -> HalBlendState {
    HalBlendState {
        color: hal_blend_component(state.color),
        alpha: hal_blend_component(state.alpha),
    }
}

fn hal_blend_component(component: BlendComponent) -> HalBlendComponent {
    HalBlendComponent {
        operation: hal_blend_operation(component.operation),
        src_factor: hal_blend_factor(component.src_factor),
        dst_factor: hal_blend_factor(component.dst_factor),
    }
}

fn hal_blend_operation(operation: BlendOperation) -> HalBlendOperation {
    match operation {
        BlendOperation::Add => HalBlendOperation::Add,
        BlendOperation::Subtract => HalBlendOperation::Subtract,
        BlendOperation::ReverseSubtract => HalBlendOperation::ReverseSubtract,
        BlendOperation::Min => HalBlendOperation::Min,
        BlendOperation::Max => HalBlendOperation::Max,
    }
}

fn hal_blend_factor(factor: BlendFactor) -> HalBlendFactor {
    match factor {
        BlendFactor::Zero => HalBlendFactor::Zero,
        BlendFactor::One => HalBlendFactor::One,
        BlendFactor::Src => HalBlendFactor::Src,
        BlendFactor::OneMinusSrc => HalBlendFactor::OneMinusSrc,
        BlendFactor::SrcAlpha => HalBlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => HalBlendFactor::OneMinusSrcAlpha,
        BlendFactor::Dst => HalBlendFactor::Dst,
        BlendFactor::OneMinusDst => HalBlendFactor::OneMinusDst,
        BlendFactor::DstAlpha => HalBlendFactor::DstAlpha,
        BlendFactor::OneMinusDstAlpha => HalBlendFactor::OneMinusDstAlpha,
        BlendFactor::SrcAlphaSaturated => HalBlendFactor::SrcAlphaSaturated,
        BlendFactor::Constant => HalBlendFactor::Constant,
        BlendFactor::OneMinusConstant => HalBlendFactor::OneMinusConstant,
        BlendFactor::Src1 => HalBlendFactor::Src1,
        BlendFactor::OneMinusSrc1 => HalBlendFactor::OneMinusSrc1,
        BlendFactor::Src1Alpha => HalBlendFactor::Src1Alpha,
        BlendFactor::OneMinusSrc1Alpha => HalBlendFactor::OneMinusSrc1Alpha,
    }
}

fn hal_depth_stencil_state(depth_stencil: DepthStencilState) -> HalDepthStencilState {
    HalDepthStencilState {
        format: hal_texture_format(depth_stencil.format),
        depth_write_enabled: depth_stencil.depth_write_enabled.unwrap_or(false),
        depth_compare: depth_stencil
            .depth_compare
            .map(crate::sampler::hal_compare_function)
            .unwrap_or(HalCompareFunction::Always),
        stencil_front: hal_stencil_face_state(depth_stencil.stencil_front),
        stencil_back: hal_stencil_face_state(depth_stencil.stencil_back),
        stencil_read_mask: depth_stencil.stencil_read_mask,
        stencil_write_mask: depth_stencil.stencil_write_mask,
        depth_bias: depth_stencil.depth_bias,
        depth_bias_slope_scale: depth_stencil.depth_bias_slope_scale,
        depth_bias_clamp: depth_stencil.depth_bias_clamp,
    }
}

fn hal_stencil_face_state(face: StencilFaceState) -> HalStencilFaceState {
    HalStencilFaceState {
        compare: crate::sampler::hal_compare_function(face.compare),
        fail_op: hal_stencil_operation(face.fail_op),
        depth_fail_op: hal_stencil_operation(face.depth_fail_op),
        pass_op: hal_stencil_operation(face.pass_op),
    }
}

fn hal_stencil_operation(operation: StencilOperation) -> HalStencilOperation {
    match operation {
        StencilOperation::Keep => HalStencilOperation::Keep,
        StencilOperation::Zero => HalStencilOperation::Zero,
        StencilOperation::Replace => HalStencilOperation::Replace,
        StencilOperation::Invert => HalStencilOperation::Invert,
        StencilOperation::IncrementClamp => HalStencilOperation::IncrementClamp,
        StencilOperation::DecrementClamp => HalStencilOperation::DecrementClamp,
        StencilOperation::IncrementWrap => HalStencilOperation::IncrementWrap,
        StencilOperation::DecrementWrap => HalStencilOperation::DecrementWrap,
    }
}

/// Returns msl vertex format.
pub(crate) fn msl_vertex_format(format: VertexFormat) -> Result<frontend::MslVertexFormat, String> {
    match format.0 {
        0x0000_0001 => Ok(frontend::MslVertexFormat::Uint8),
        0x0000_0002 => Ok(frontend::MslVertexFormat::Uint8x2),
        0x0000_0003 => Ok(frontend::MslVertexFormat::Uint8x4),
        0x0000_0004 => Ok(frontend::MslVertexFormat::Sint8),
        0x0000_0005 => Ok(frontend::MslVertexFormat::Sint8x2),
        0x0000_0006 => Ok(frontend::MslVertexFormat::Sint8x4),
        0x0000_0007 => Ok(frontend::MslVertexFormat::Unorm8),
        0x0000_0008 => Ok(frontend::MslVertexFormat::Unorm8x2),
        0x0000_0009 => Ok(frontend::MslVertexFormat::Unorm8x4),
        0x0000_000A => Ok(frontend::MslVertexFormat::Snorm8),
        0x0000_000B => Ok(frontend::MslVertexFormat::Snorm8x2),
        0x0000_000C => Ok(frontend::MslVertexFormat::Snorm8x4),
        0x0000_000D => Ok(frontend::MslVertexFormat::Uint16),
        0x0000_000E => Ok(frontend::MslVertexFormat::Uint16x2),
        0x0000_000F => Ok(frontend::MslVertexFormat::Uint16x4),
        0x0000_0010 => Ok(frontend::MslVertexFormat::Sint16),
        0x0000_0011 => Ok(frontend::MslVertexFormat::Sint16x2),
        0x0000_0012 => Ok(frontend::MslVertexFormat::Sint16x4),
        0x0000_0013 => Ok(frontend::MslVertexFormat::Unorm16),
        0x0000_0014 => Ok(frontend::MslVertexFormat::Unorm16x2),
        0x0000_0015 => Ok(frontend::MslVertexFormat::Unorm16x4),
        0x0000_0016 => Ok(frontend::MslVertexFormat::Snorm16),
        0x0000_0017 => Ok(frontend::MslVertexFormat::Snorm16x2),
        0x0000_0018 => Ok(frontend::MslVertexFormat::Snorm16x4),
        0x0000_0019 => Ok(frontend::MslVertexFormat::Float16),
        0x0000_001A => Ok(frontend::MslVertexFormat::Float16x2),
        0x0000_001B => Ok(frontend::MslVertexFormat::Float16x4),
        0x0000_001C => Ok(frontend::MslVertexFormat::Float32),
        0x0000_001D => Ok(frontend::MslVertexFormat::Float32x2),
        0x0000_001E => Ok(frontend::MslVertexFormat::Float32x3),
        0x0000_001F => Ok(frontend::MslVertexFormat::Float32x4),
        0x0000_0020 => Ok(frontend::MslVertexFormat::Uint32),
        0x0000_0021 => Ok(frontend::MslVertexFormat::Uint32x2),
        0x0000_0022 => Ok(frontend::MslVertexFormat::Uint32x3),
        0x0000_0023 => Ok(frontend::MslVertexFormat::Uint32x4),
        0x0000_0024 => Ok(frontend::MslVertexFormat::Sint32),
        0x0000_0025 => Ok(frontend::MslVertexFormat::Sint32x2),
        0x0000_0026 => Ok(frontend::MslVertexFormat::Sint32x3),
        0x0000_0027 => Ok(frontend::MslVertexFormat::Sint32x4),
        0x0000_0028 => Ok(frontend::MslVertexFormat::Unorm10_10_10_2),
        0x0000_0029 => Ok(frontend::MslVertexFormat::Unorm8x4Bgra),
        _ => Err("unsupported Metal vertex format".to_owned()),
    }
}

/// Returns HAL vertex format.
pub(crate) fn hal_vertex_format(format: VertexFormat) -> HalVertexFormat {
    match format.0 {
        0x0000_0001 => HalVertexFormat::Uint8,
        0x0000_0002 => HalVertexFormat::Uint8x2,
        0x0000_0003 => HalVertexFormat::Uint8x4,
        0x0000_0004 => HalVertexFormat::Sint8,
        0x0000_0005 => HalVertexFormat::Sint8x2,
        0x0000_0006 => HalVertexFormat::Sint8x4,
        0x0000_0007 => HalVertexFormat::Unorm8,
        0x0000_0008 => HalVertexFormat::Unorm8x2,
        0x0000_0009 => HalVertexFormat::Unorm8x4,
        0x0000_000A => HalVertexFormat::Snorm8,
        0x0000_000B => HalVertexFormat::Snorm8x2,
        0x0000_000C => HalVertexFormat::Snorm8x4,
        0x0000_000D => HalVertexFormat::Uint16,
        0x0000_000E => HalVertexFormat::Uint16x2,
        0x0000_000F => HalVertexFormat::Uint16x4,
        0x0000_0010 => HalVertexFormat::Sint16,
        0x0000_0011 => HalVertexFormat::Sint16x2,
        0x0000_0012 => HalVertexFormat::Sint16x4,
        0x0000_0013 => HalVertexFormat::Unorm16,
        0x0000_0014 => HalVertexFormat::Unorm16x2,
        0x0000_0015 => HalVertexFormat::Unorm16x4,
        0x0000_0016 => HalVertexFormat::Snorm16,
        0x0000_0017 => HalVertexFormat::Snorm16x2,
        0x0000_0018 => HalVertexFormat::Snorm16x4,
        0x0000_0019 => HalVertexFormat::Float16,
        0x0000_001A => HalVertexFormat::Float16x2,
        0x0000_001B => HalVertexFormat::Float16x4,
        0x0000_001C => HalVertexFormat::Float32,
        0x0000_001D => HalVertexFormat::Float32x2,
        0x0000_001E => HalVertexFormat::Float32x3,
        0x0000_001F => HalVertexFormat::Float32x4,
        0x0000_0020 => HalVertexFormat::Uint32,
        0x0000_0021 => HalVertexFormat::Uint32x2,
        0x0000_0022 => HalVertexFormat::Uint32x3,
        0x0000_0023 => HalVertexFormat::Uint32x4,
        0x0000_0024 => HalVertexFormat::Sint32,
        0x0000_0025 => HalVertexFormat::Sint32x2,
        0x0000_0026 => HalVertexFormat::Sint32x3,
        0x0000_0027 => HalVertexFormat::Sint32x4,
        0x0000_0028 => HalVertexFormat::Unorm10_10_10_2,
        0x0000_0029 => HalVertexFormat::Unorm8x4Bgra,
        _ => HalVertexFormat::Unsupported,
    }
}

/// Returns HAL primitive topology.
pub(crate) fn hal_primitive_topology(topology: PrimitiveTopology) -> HalPrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => HalPrimitiveTopology::PointList,
        PrimitiveTopology::LineList => HalPrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip => HalPrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList => HalPrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip => HalPrimitiveTopology::TriangleStrip,
    }
}

/// Records resolve into the command stream.
pub(crate) fn resolve_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    limits: Limits,
    features: &FeatureSet,
    subpass_color_attachment_indices: Option<&[u32]>,
    pipeline_id: u64,
) -> Result<ResolvedRenderPipelineParts, String> {
    if let RenderPipelineLayout::Explicit(layout) = &descriptor.layout {
        if layout.is_error() {
            return Err("render pipeline layout must not be an error pipeline layout".to_owned());
        }
    }
    let vertex_entry = resolve_render_entry(
        &descriptor.vertex.shader,
        frontend::ShaderStage::Vertex,
        "vertex",
    )?;
    let fragment_entry = if let Some(fragment) = &descriptor.fragment {
        Some(resolve_render_entry(
            &fragment.shader,
            frontend::ShaderStage::Fragment,
            "fragment",
        )?)
    } else {
        None
    };

    validate_render_constants(&descriptor.vertex.shader)?;
    if let Some(fragment) = &descriptor.fragment {
        validate_render_constants(&fragment.shader)?;
    }
    validate_vertex_state(&descriptor.vertex, &vertex_entry, limits)?;
    validate_render_presence(descriptor)?;
    validate_primitive_state(descriptor.primitive, features)?;
    if let Some(depth_stencil) = descriptor.depth_stencil {
        validate_depth_bias_state(descriptor.primitive.topology, depth_stencil)?;
        validate_depth_stencil_aspects(depth_stencil, features)?;
    }
    validate_fragment_depth_output(descriptor, fragment_entry.as_deref(), features)?;
    validate_inter_stage_interface(descriptor, &vertex_entry, fragment_entry.as_deref(), limits)?;
    validate_color_targets(
        descriptor,
        fragment_entry.as_deref(),
        limits,
        features,
        subpass_color_attachment_indices,
    )?;
    validate_render_pipeline_layout(descriptor, &vertex_entry, fragment_entry.as_deref())?;
    let immediate_size_budget = resolve_render_pipeline_immediate_size(
        descriptor,
        &vertex_entry,
        fragment_entry.as_deref(),
        limits,
    )?;
    validate_render_stage_immediate_size(
        &descriptor.vertex.shader,
        &vertex_entry,
        immediate_size_budget,
    )?;
    if let (Some(fragment), Some(fragment_entry)) =
        (&descriptor.fragment, fragment_entry.as_deref())
    {
        validate_render_stage_immediate_size(
            &fragment.shader,
            fragment_entry,
            immediate_size_budget,
        )?;
    }
    validate_multisample_state(descriptor, fragment_entry.as_deref())?;
    let bind_group_layouts = effective_render_bind_group_layouts(
        descriptor,
        &vertex_entry,
        fragment_entry.as_deref(),
        limits,
        features,
        pipeline_id,
    )?;
    validate_bind_groups_plus_vertex_buffers(
        &bind_group_layouts,
        descriptor.vertex.buffer_count,
        limits,
    )?;

    Ok((vertex_entry, fragment_entry, bind_group_layouts))
}

/// Records resolve into the command stream for raw shader passthrough render.
#[cfg(feature = "shader-passthrough")]
pub(crate) fn resolve_shader_passthrough_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
) -> Result<ResolvedRenderPipelineParts, String> {
    if descriptor.vertex.shader.module.is_error()
        || descriptor
            .fragment
            .as_ref()
            .is_some_and(|fragment| fragment.shader.module.is_error())
    {
        return Err("render pipeline shader module must not be an error module".to_owned());
    }
    if !descriptor.vertex.shader.constants.is_empty()
        || descriptor
            .fragment
            .as_ref()
            .is_some_and(|fragment| !fragment.shader.constants.is_empty())
    {
        return Err(
            "pipeline-overridable constants are not supported with shader passthrough".to_owned(),
        );
    }
    let Some(kind) = render_stage_passthrough_kind(&descriptor.vertex.shader.module) else {
        return Err(
            "render shader passthrough requires all stages to be passthrough modules of the same kind"
                .to_owned(),
        );
    };
    if descriptor.fragment.as_ref().is_some_and(|fragment| {
        render_stage_passthrough_kind(&fragment.shader.module) != Some(kind)
    }) {
        return Err(
            "render shader passthrough requires all stages to be passthrough modules of the same kind"
                .to_owned(),
        );
    }
    let RenderPipelineLayout::Explicit(layout) = &descriptor.layout else {
        return Err("shader passthrough requires an explicit pipeline layout".to_owned());
    };
    if layout.is_error() {
        return Err("render pipeline layout must not be an error pipeline layout".to_owned());
    }
    validate_render_presence(descriptor)?;
    let Some(vertex_entry) = descriptor
        .vertex
        .shader
        .entry_point
        .as_deref()
        .filter(|entry| !entry.is_empty())
    else {
        return Err("shader passthrough vertex entry point is required".to_owned());
    };
    let fragment_entry = descriptor
        .fragment
        .as_ref()
        .map(|fragment| {
            fragment
                .shader
                .entry_point
                .as_deref()
                .filter(|entry| !entry.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| "shader passthrough fragment entry point is required".to_owned())
        })
        .transpose()?;

    Ok((
        vertex_entry.to_owned(),
        fragment_entry,
        layout.bind_group_layouts().to_vec(),
    ))
}

fn validate_bind_groups_plus_vertex_buffers(
    bind_group_layouts: &[Arc<BindGroupLayout>],
    vertex_buffer_count: usize,
    limits: Limits,
) -> Result<(), String> {
    let total = bind_group_layouts
        .len()
        .checked_add(vertex_buffer_count)
        .ok_or_else(|| {
            "render pipeline bind group plus vertex buffer count overflows".to_owned()
        })?;
    if total > limits.max_bind_groups_plus_vertex_buffers as usize {
        return Err(
            "render pipeline bind group plus vertex buffer count exceeds the device limit"
                .to_owned(),
        );
    }
    Ok(())
}

/// Validates vertex state and returns a descriptive error on failure.
pub(crate) fn validate_vertex_state(
    vertex: &RenderPipelineVertexState,
    vertex_entry: &str,
    limits: Limits,
) -> Result<(), String> {
    if vertex.buffer_count > limits.max_vertex_buffers as usize {
        return Err("render pipeline vertex buffer count exceeds the device limit".to_owned());
    }
    if vertex.buffers.len() != vertex.buffer_count {
        return Err("render pipeline vertex buffer count does not match buffers".to_owned());
    }

    let attribute_count = vertex
        .buffers
        .iter()
        .map(|buffer| buffer.attributes.len())
        .try_fold(0usize, |sum, count| {
            sum.checked_add(count)
                .ok_or_else(|| "render pipeline vertex attribute count overflows".to_owned())
        })?;
    if attribute_count > limits.max_vertex_attributes as usize {
        return Err("render pipeline vertex attribute count exceeds the device limit".to_owned());
    }

    let mut locations = BTreeSet::new();
    let mut attribute_classes = BTreeMap::new();
    for buffer in &vertex.buffers {
        if buffer.array_stride != 0 && buffer.array_stride % 4 != 0 {
            return Err(
                "render pipeline vertex buffer arrayStride must be a multiple of 4".to_owned(),
            );
        }
        if buffer.array_stride > u64::from(limits.max_vertex_buffer_array_stride) {
            return Err(
                "render pipeline vertex buffer arrayStride exceeds the device limit".to_owned(),
            );
        }

        for attribute in &buffer.attributes {
            let info = attribute.format.info();
            let alignment = info.byte_size.min(4);
            if attribute.offset % alignment != 0 {
                return Err(
                    "render pipeline vertex attribute offset is not properly aligned".to_owned(),
                );
            }
            let end = attribute
                .offset
                .checked_add(info.byte_size)
                .ok_or_else(|| {
                    "render pipeline vertex attribute byte range overflows".to_owned()
                })?;
            let upper_bound = if buffer.array_stride == 0 {
                u64::from(limits.max_vertex_buffer_array_stride)
            } else {
                buffer.array_stride
            };
            if end > upper_bound {
                return Err(
                    "render pipeline vertex attribute byte range exceeds the buffer arrayStride"
                        .to_owned(),
                );
            }
            if !locations.insert(attribute.shader_location) {
                return Err(
                    "render pipeline vertex attributes must not duplicate shaderLocation"
                        .to_owned(),
                );
            }
            if attribute.shader_location >= limits.max_vertex_attributes {
                return Err(
                    "render pipeline vertex attribute shaderLocation exceeds the device limit"
                        .to_owned(),
                );
            }
            attribute_classes.insert(attribute.shader_location, info.output_class);
        }
    }

    for (location, input) in vertex_inputs(vertex, vertex_entry)? {
        let Some(attribute_class) = attribute_classes.get(&location) else {
            return Err(
                "render pipeline vertex shader input has no matching vertex attribute".to_owned(),
            );
        };
        let input_class = match input.scalar {
            frontend::ReflectedTypeScalarClass::Float => FormatOutputClass::Float,
            frontend::ReflectedTypeScalarClass::Sint => FormatOutputClass::Sint,
            frontend::ReflectedTypeScalarClass::Uint => FormatOutputClass::Uint,
            frontend::ReflectedTypeScalarClass::Bool => {
                return Err("render pipeline vertex shader input type is incompatible".to_owned());
            }
        };
        if *attribute_class != input_class {
            return Err("render pipeline vertex shader input type is incompatible".to_owned());
        }
    }

    Ok(())
}

/// Returns vertex inputs.
pub(crate) fn vertex_inputs(
    vertex: &RenderPipelineVertexState,
    vertex_entry: &str,
) -> Result<BTreeMap<u32, frontend::ReflectedTypeClass>, String> {
    let Some(module) = vertex.shader.module.reflected() else {
        return Err("vertex module reflection failed".to_owned());
    };
    Ok(module
        .entry_point_io(vertex_entry)
        .map(|io| {
            io.inputs
                .into_iter()
                .map(|input| (input.location, input.ty))
                .collect()
        })
        .unwrap_or_default())
}

/// Records resolve into the command stream.
pub(crate) fn resolve_render_entry(
    stage: &RenderPipelineShaderStage,
    expected_stage: frontend::ShaderStage,
    label: &str,
) -> Result<String, String> {
    if stage.module.is_error() {
        return Err(format!(
            "render pipeline {label} shader module must not be an error module"
        ));
    }
    let Some(module) = stage.module.reflected() else {
        return Err(format!(
            "render pipeline {label} stage requires a reflected shader module"
        ));
    };
    let entries = module.entry_points();
    let matching_entries = entries
        .iter()
        .filter(|entry| entry.stage == expected_stage)
        .collect::<Vec<_>>();

    match stage.entry_point.as_deref() {
        None => match matching_entries.as_slice() {
            [entry] => Ok(entry.name.clone()),
            [] => Err(format!(
                "render pipeline {label} shader module has no matching entry point"
            )),
            _ => Err(format!(
                "render pipeline {label} entryPoint is required when multiple matching entries exist"
            )),
        },
        Some(name) => matching_entries
            .iter()
            .any(|entry| entry.name == name)
            .then(|| name.to_owned())
            .ok_or_else(|| {
                format!("render pipeline {label} entryPoint must name a matching entry point")
            }),
    }
}

/// Validates render presence and returns a descriptive error on failure.
pub(crate) fn validate_render_presence(
    descriptor: &RenderPipelineDescriptor,
) -> Result<(), String> {
    if descriptor.fragment.is_none() && descriptor.depth_stencil.is_none() {
        return Err("render pipeline requires a fragment state or depthStencil state".to_owned());
    }
    if descriptor
        .fragment
        .as_ref()
        .is_some_and(|fragment| fragment.target_count == 0)
        && descriptor.depth_stencil.is_none()
    {
        return Err("render pipeline requires a color target or depthStencil state".to_owned());
    }
    // A fragment state with zero color targets is allowed. Fragment color
    // outputs without matching targets are discarded when a depth-stencil state
    // makes the pipeline complete.
    Ok(())
}

/// Validates render constants and returns a descriptive error on failure.
pub(crate) fn validate_render_constants(stage: &RenderPipelineShaderStage) -> Result<(), String> {
    let Some(module) = stage.module.reflected() else {
        return Err("render pipeline stage requires a reflected shader module".to_owned());
    };
    resolve_pipeline_constants(&module.overrides(), &stage.constants)?;
    Ok(())
}

/// Validates primitive state and returns a descriptive error on failure.
pub(crate) fn validate_primitive_state(
    primitive: PrimitiveState,
    features: &FeatureSet,
) -> Result<(), String> {
    if primitive.unclipped_depth && !features.contains(&Feature::DepthClipControl) {
        return Err(
            "render pipeline unclippedDepth requires the depth-clip-control feature".to_owned(),
        );
    }
    if primitive.strip_index_format.is_some()
        && !matches!(
            primitive.topology,
            PrimitiveTopology::LineStrip | PrimitiveTopology::TriangleStrip
        )
    {
        return Err(
            "render pipeline stripIndexFormat requires a strip primitive topology".to_owned(),
        );
    }
    Ok(())
}

/// Validates depth bias state and returns a descriptive error on failure.
pub(crate) fn validate_depth_bias_state(
    topology: PrimitiveTopology,
    depth_stencil: DepthStencilState,
) -> Result<(), String> {
    if !depth_stencil.depth_bias_slope_scale.is_finite()
        || !depth_stencil.depth_bias_clamp.is_finite()
    {
        return Err("render pipeline depth bias values must be finite".to_owned());
    }

    let has_non_zero_bias = depth_stencil.depth_bias != 0
        || depth_stencil.depth_bias_slope_scale != 0.0
        || depth_stencil.depth_bias_clamp != 0.0;
    if has_non_zero_bias
        && !matches!(
            topology,
            PrimitiveTopology::TriangleList | PrimitiveTopology::TriangleStrip
        )
    {
        return Err("render pipeline non-zero depth bias requires triangle topology".to_owned());
    }
    Ok(())
}

/// Validates depth stencil aspects and returns a descriptive error on failure.
pub(crate) fn validate_depth_stencil_aspects(
    depth_stencil: DepthStencilState,
    features: &FeatureSet,
) -> Result<(), String> {
    let caps = depth_stencil.format.caps(features);
    let has_depth = caps.is_some_and(|caps| caps.aspects.depth);
    let has_stencil = caps.is_some_and(|caps| caps.aspects.stencil);

    if !has_depth && !has_stencil {
        return Err("render pipeline depthStencil format must have depth or stencil".to_owned());
    }

    let uses_depth_test = depth_stencil
        .depth_compare
        .is_some_and(|compare| compare != CompareFunction::Always);
    let uses_depth_write = depth_stencil.depth_write_enabled == Some(true);

    if (uses_depth_test || uses_depth_write) && !has_depth {
        return Err("render pipeline depth test or write requires a depth format".to_owned());
    }

    if has_depth {
        // `depthWriteEnabled` is always required for a depth format.
        if depth_stencil.depth_write_enabled.is_none() {
            return Err("render pipeline depth format requires depthWriteEnabled".to_owned());
        }
        // `depthCompare` is required only when the depth aspect is actually used:
        // depth is written, or a stencil face's depthFailOp (which consults the
        // depth test) is not `Keep`. When depth is neither written nor consulted,
        // `depthCompare` may be omitted (F-058 — yawgpu previously always required
        // it). Mirrors the WebGPU `GPUDepthStencilState` validation algorithm.
        let depth_aspect_used = depth_stencil.depth_write_enabled == Some(true)
            || depth_stencil.stencil_front.depth_fail_op != StencilOperation::Keep
            || depth_stencil.stencil_back.depth_fail_op != StencilOperation::Keep;
        if depth_aspect_used && depth_stencil.depth_compare.is_none() {
            return Err(
                "render pipeline depth format requires depthCompare when the depth aspect is used"
                    .to_owned(),
            );
        }
    }

    if depth_stencil_uses_stencil(depth_stencil) && !has_stencil {
        return Err("render pipeline stencil state requires a stencil format".to_owned());
    }

    Ok(())
}

/// Returns depth stencil uses stencil.
pub(crate) fn depth_stencil_uses_stencil(depth_stencil: DepthStencilState) -> bool {
    stencil_face_uses_stencil(depth_stencil.stencil_front)
        || stencil_face_uses_stencil(depth_stencil.stencil_back)
}

/// Returns stencil face uses stencil.
pub(crate) fn stencil_face_uses_stencil(face: StencilFaceState) -> bool {
    face.compare != CompareFunction::Always
        || face.fail_op != StencilOperation::Keep
        || face.depth_fail_op != StencilOperation::Keep
        || face.pass_op != StencilOperation::Keep
}

/// Returns true when a stencil face performs a stencil *write* — any of its
/// operations is not `Keep`. The compare function is a test (a read), not a
/// write, so it is deliberately excluded; see [`RenderPipeline::writes_stencil`].
pub(crate) fn stencil_face_writes(face: StencilFaceState) -> bool {
    face.fail_op != StencilOperation::Keep
        || face.depth_fail_op != StencilOperation::Keep
        || face.pass_op != StencilOperation::Keep
}

/// Validates fragment depth output and returns a descriptive error on failure.
pub(crate) fn validate_fragment_depth_output(
    descriptor: &RenderPipelineDescriptor,
    fragment_entry: Option<&str>,
    features: &FeatureSet,
) -> Result<(), String> {
    let Some(fragment) = &descriptor.fragment else {
        return Ok(());
    };
    let Some(entry_name) = fragment_entry else {
        return Ok(());
    };
    let Some(module) = fragment.shader.module.reflected() else {
        return Err("fragment module reflection failed".to_owned());
    };
    let outputs_frag_depth = module
        .fragment_builtins(entry_name)
        .is_some_and(|builtins| builtins.frag_depth);
    if outputs_frag_depth
        && !descriptor
            .depth_stencil
            .and_then(|state| state.format.caps(features))
            .is_some_and(|caps| caps.aspects.depth)
    {
        return Err("render pipeline frag_depth output requires a depth attachment".to_owned());
    }
    Ok(())
}

/// Validates color targets and returns a descriptive error on failure.
pub(crate) fn validate_color_targets(
    descriptor: &RenderPipelineDescriptor,
    fragment_entry: Option<&str>,
    limits: Limits,
    features: &FeatureSet,
    subpass_color_attachment_indices: Option<&[u32]>,
) -> Result<(), String> {
    let Some(fragment) = &descriptor.fragment else {
        return Ok(());
    };
    if fragment.targets.len() != fragment.target_count {
        return Err("render pipeline fragment target array must match targetCount".to_owned());
    }
    if fragment.target_count > limits.max_color_attachments as usize {
        return Err("render pipeline color target count exceeds the device limit".to_owned());
    }

    validate_fragment_color_inputs(fragment, fragment_entry)?;

    let outputs = fragment_outputs(fragment, fragment_entry)?;
    let fragment_writes_blend_src_1 = match fragment_entry {
        Some(entry_name) => fragment
            .shader
            .module
            .reflected()
            .ok_or_else(|| "fragment module reflection failed".to_owned())?
            .fragment_writes_blend_src_1(entry_name),
        None => false,
    };
    let mut color_formats = Vec::new();
    let mut has_alpha_to_coverage_target = false;
    for (index, target) in fragment.targets.iter().enumerate() {
        if target.format.is_undefined() {
            if target.blend.is_some() {
                return Err("render pipeline undefined color target must not have blend".to_owned());
            }
            continue;
        }

        if target.write_mask & !0xF != 0 {
            return Err("render pipeline color target writeMask has invalid bits".to_owned());
        }
        let caps = target
            .format
            .caps(features)
            .ok_or_else(|| "render pipeline color target format must be defined".to_owned())?;
        if !caps.renderable {
            return Err("render pipeline color target format must be renderable".to_owned());
        }
        if let Some(blend) = target.blend {
            validate_blend_state(blend)?;
            if blend_state_uses_dual_source(blend) {
                if !features.contains(&Feature::DualSourceBlending) {
                    return Err(
                        "render pipeline dual-source blend factors require the dual-source-blending feature"
                            .to_owned(),
                    );
                }
                if !fragment_writes_blend_src_1 {
                    return Err(
                        "render pipeline dual-source blend factors require the fragment shader to write a @blend_src(1) output"
                            .to_owned(),
                    );
                }
                if fragment.target_count != 1 || index != 0 {
                    return Err("dual-source blending requires a single color target".to_owned());
                }
            }
        }
        if target.blend.is_some() && !caps.is_blendable {
            return Err("render pipeline color target format must be blendable".to_owned());
        }
        if descriptor.multisample.alpha_to_coverage_enabled && caps.is_blendable && caps.has_alpha {
            has_alpha_to_coverage_target = true;
        }

        // Subpass pipelines accept both subpass-local and flat attachment-slot
        // `@location` conventions. The flat slot is supplied by the subpass's
        // `color_attachment_indices`; regular pipelines use the local index.
        let subpass_local = index as u32;
        let flat = subpass_color_attachment_indices
            .and_then(|indices| indices.get(index).copied())
            .unwrap_or(subpass_local);
        let output = outputs.get(&subpass_local).or_else(|| outputs.get(&flat));
        match output {
            Some(output) => {
                validate_fragment_output_compat(*output, caps)?;
                if target.blend.is_some_and(color_blend_uses_source_alpha) && output.components < 4
                {
                    return Err(
                        "render pipeline blend state requires a vec4 fragment output".to_owned(),
                    );
                }
            }
            None if target.write_mask != 0 => {
                return Err(
                    "render pipeline color target without shader output must use writeMask 0"
                        .to_owned(),
                );
            }
            None => {}
        }

        color_formats.push(target.format);
    }

    if descriptor.multisample.alpha_to_coverage_enabled && !has_alpha_to_coverage_target {
        return Err(
            "render pipeline alphaToCoverage requires an alpha blendable color target".to_owned(),
        );
    }
    let color_bytes = color_attachment_bytes_per_sample(color_formats)
        .ok_or_else(|| "render pipeline color target byte count overflows".to_owned())?;
    if color_bytes > limits.max_color_attachment_bytes_per_sample {
        return Err(
            "render pipeline color target bytes per sample exceed the device limit".to_owned(),
        );
    }

    Ok(())
}

/// Validates framebuffer-fetch color inputs against declared color targets.
pub(crate) fn validate_fragment_color_inputs(
    fragment: &RenderPipelineFragmentState,
    fragment_entry: Option<&str>,
) -> Result<(), String> {
    let Some(entry_name) = fragment_entry else {
        return Ok(());
    };
    let Some(module) = fragment.shader.module.reflected() else {
        return Err("fragment module reflection failed".to_owned());
    };
    for slot in module.fragment_color_inputs(entry_name) {
        let Some(target) = fragment.targets.get(slot as usize) else {
            return Err(
                "render pipeline framebuffer-fetch color input requires a declared color target"
                    .to_owned(),
            );
        };
        if target.format.is_undefined() {
            return Err(
                "render pipeline framebuffer-fetch color input requires a declared color target"
                    .to_owned(),
            );
        }
    }
    Ok(())
}

/// Returns the dedicated Vulkan descriptor set used for framebuffer-fetch inputs.
pub(crate) fn framebuffer_fetch_descriptor_set(
    bind_group_layouts: &[Arc<BindGroupLayout>],
) -> Result<u32, String> {
    u32::try_from(bind_group_layouts.len())
        .map_err(|_| "framebuffer-fetch descriptor set index is too large".to_owned())
}

fn render_pipeline_framebuffer_fetch_color_slots(
    descriptor: &RenderPipelineDescriptor,
    fragment_entry: Option<&str>,
) -> Vec<u32> {
    let Some(fragment_entry) = fragment_entry else {
        return Vec::new();
    };
    descriptor
        .fragment
        .as_ref()
        .and_then(|fragment| fragment.shader.module.reflected())
        .map(|module| module.fragment_color_inputs(fragment_entry))
        .unwrap_or_default()
}

pub(crate) fn color_attachment_bytes_per_sample(
    formats: impl IntoIterator<Item = TextureFormat>,
) -> Option<u32> {
    formats.into_iter().try_fold(0_u32, |bytes, format| {
        let Some((byte_cost, alignment)) = color_attachment_byte_cost_and_alignment(format) else {
            return Some(bytes);
        };
        let aligned = align_color_attachment_bytes(bytes, alignment)?;
        aligned.checked_add(byte_cost)
    })
}

fn align_color_attachment_bytes(bytes: u32, alignment: u32) -> Option<u32> {
    let addend = alignment.checked_sub(1)?;
    bytes.checked_add(addend).map(|value| value & !addend)
}

fn color_attachment_byte_cost_and_alignment(format: TextureFormat) -> Option<(u32, u32)> {
    match format.raw() {
        TextureFormat::R8_UNORM
        | TextureFormat::R8_UINT
        | TextureFormat::R8_SINT
        | TextureFormat::R8_SNORM => Some((1, 1)),
        TextureFormat::RG8_UNORM
        | TextureFormat::RG8_UINT
        | TextureFormat::RG8_SINT
        | TextureFormat::RG8_SNORM => Some((2, 1)),
        TextureFormat::RGBA8_UNORM
        | TextureFormat::RGBA8_UNORM_SRGB
        | TextureFormat::BGRA8_UNORM
        | TextureFormat::BGRA8_UNORM_SRGB => Some((8, 1)),
        TextureFormat::RGBA8_UINT | TextureFormat::RGBA8_SINT | TextureFormat::RGBA8_SNORM => {
            Some((4, 1))
        }
        TextureFormat::R16_UNORM
        | TextureFormat::R16_SNORM
        | TextureFormat::R16_UINT
        | TextureFormat::R16_SINT
        | TextureFormat::R16_FLOAT => Some((2, 2)),
        TextureFormat::RG16_UNORM
        | TextureFormat::RG16_SNORM
        | TextureFormat::RG16_UINT
        | TextureFormat::RG16_SINT
        | TextureFormat::RG16_FLOAT => Some((4, 2)),
        TextureFormat::RGBA16_UNORM => Some((8, 4)),
        TextureFormat::RGBA16_SNORM
        | TextureFormat::RGBA16_UINT
        | TextureFormat::RGBA16_SINT
        | TextureFormat::RGBA16_FLOAT => Some((8, 2)),
        TextureFormat::R32_UINT | TextureFormat::R32_SINT | TextureFormat::R32_FLOAT => {
            Some((4, 4))
        }
        TextureFormat::RG32_UINT | TextureFormat::RG32_SINT | TextureFormat::RG32_FLOAT => {
            Some((8, 4))
        }
        TextureFormat::RGBA32_UINT | TextureFormat::RGBA32_SINT | TextureFormat::RGBA32_FLOAT => {
            Some((16, 4))
        }
        TextureFormat::RGB10A2_UINT
        | TextureFormat::RGB10A2_UNORM
        | TextureFormat::RG11B10_UFLOAT => Some((8, 4)),
        _ => None,
    }
}

/// Validates blend state and returns a descriptive error on failure.
pub(crate) fn validate_blend_state(blend: BlendState) -> Result<(), String> {
    validate_blend_component(blend.color)?;
    validate_blend_component(blend.alpha)
}

fn blend_state_uses_dual_source(blend: BlendState) -> bool {
    blend_component_uses_dual_source(blend.color) || blend_component_uses_dual_source(blend.alpha)
}

fn blend_component_uses_dual_source(component: BlendComponent) -> bool {
    blend_factor_is_dual_source(component.src_factor)
        || blend_factor_is_dual_source(component.dst_factor)
}

fn blend_factor_is_dual_source(factor: BlendFactor) -> bool {
    matches!(
        factor,
        BlendFactor::Src1
            | BlendFactor::OneMinusSrc1
            | BlendFactor::Src1Alpha
            | BlendFactor::OneMinusSrc1Alpha
    )
}

fn validate_blend_component(component: BlendComponent) -> Result<(), String> {
    if matches!(
        component.operation,
        BlendOperation::Min | BlendOperation::Max
    ) && (component.src_factor != BlendFactor::One || component.dst_factor != BlendFactor::One)
    {
        return Err("render pipeline min/max blend operations require one factors".to_owned());
    }
    Ok(())
}

fn color_blend_uses_source_alpha(blend: BlendState) -> bool {
    blend_component_uses_source_alpha(blend.color)
}

fn blend_component_uses_source_alpha(component: BlendComponent) -> bool {
    blend_factor_uses_source_alpha(component.src_factor)
        || blend_factor_uses_source_alpha(component.dst_factor)
}

fn blend_factor_uses_source_alpha(factor: BlendFactor) -> bool {
    matches!(
        factor,
        BlendFactor::SrcAlpha
            | BlendFactor::OneMinusSrcAlpha
            | BlendFactor::SrcAlphaSaturated
            | BlendFactor::Src1Alpha
            | BlendFactor::OneMinusSrc1Alpha
    )
}

/// Returns fragment outputs.
pub(crate) fn fragment_outputs(
    fragment: &RenderPipelineFragmentState,
    fragment_entry: Option<&str>,
) -> Result<BTreeMap<u32, frontend::ReflectedTypeClass>, String> {
    let Some(entry_name) = fragment_entry else {
        return Ok(BTreeMap::new());
    };
    let Some(module) = fragment.shader.module.reflected() else {
        return Err("fragment module reflection failed".to_owned());
    };
    Ok(module
        .entry_point_io(entry_name)
        .map(|io| {
            io.outputs
                .into_iter()
                .map(|output| (output.location, output.ty))
                .collect()
        })
        .unwrap_or_default())
}

/// Validates fragment output compat and returns a descriptive error on failure.
pub(crate) fn validate_fragment_output_compat(
    output: frontend::ReflectedTypeClass,
    caps: FormatCaps,
) -> Result<(), String> {
    let Some(format_class) = caps.output_class else {
        return Err("render pipeline color target format has no output class".to_owned());
    };
    let output_class = match output.scalar {
        frontend::ReflectedTypeScalarClass::Float => FormatOutputClass::Float,
        frontend::ReflectedTypeScalarClass::Sint => FormatOutputClass::Sint,
        frontend::ReflectedTypeScalarClass::Uint => FormatOutputClass::Uint,
        frontend::ReflectedTypeScalarClass::Bool => {
            return Err("render pipeline fragment output type is incompatible".to_owned());
        }
    };
    if output_class != format_class || output.components < caps.color_components {
        return Err("render pipeline fragment output type is incompatible".to_owned());
    }
    Ok(())
}

/// Validates vertex-output to fragment-input interface.
pub(crate) fn validate_inter_stage_interface(
    descriptor: &RenderPipelineDescriptor,
    vertex_entry: &str,
    fragment_entry: Option<&str>,
    limits: Limits,
) -> Result<(), String> {
    let Some(fragment) = &descriptor.fragment else {
        return Ok(());
    };
    let Some(fragment_entry) = fragment_entry else {
        return Ok(());
    };
    let outputs = inter_stage_outputs(&descriptor.vertex, vertex_entry)?;
    let inputs = inter_stage_inputs(fragment, fragment_entry)?;
    let clip_distances_size = vertex_clip_distances_size(&descriptor.vertex, vertex_entry)?;
    validate_clip_distances_size(clip_distances_size)?;
    let clip_slots = clip_distances_size.div_ceil(4);
    // Fragment stage-input `@builtin`s (front_facing, sample_index, sample_mask,
    // …) also consume `maxInterStageShaderVariables` slots, so count them with
    // the user-defined inputs (F-063).
    let input_builtins = inter_stage_input_builtin_count(fragment, fragment_entry)?;
    validate_inter_stage_limits(&outputs, clip_slots, clip_slots, limits, "output")?;
    validate_inter_stage_limits(&inputs, input_builtins, 0, limits, "input")?;
    let point_list_slots = u32::from(matches!(
        descriptor.primitive.topology,
        PrimitiveTopology::PointList
    ));
    if outputs.len() as u64 + u64::from(clip_slots) + u64::from(point_list_slots)
        > u64::from(limits.max_inter_stage_shader_variables)
    {
        return Err(
            "render pipeline point-list inter-stage output count reaches the device limit"
                .to_owned(),
        );
    }

    for (location, input) in &inputs {
        let Some(output) = outputs.get(location) else {
            return Err("render pipeline fragment input has no matching vertex output".to_owned());
        };
        if output.ty != input.ty {
            return Err("render pipeline inter-stage types are incompatible".to_owned());
        }
        // Compare interpolation type and sampling after applying WebGPU defaults:
        // an unspecified interpolation defaults to `perspective`, and an
        // unspecified sampling defaults to `center` for perspective/linear. The
        // frontend only fills these defaults when the *interpolation* was unspecified, so
        // `@interpolate(perspective)` (sampling None) and `@interpolate(perspective,
        // center)` would otherwise compare unequal — a false reject (F-063).
        if effective_interpolation(output.interpolation)
            != effective_interpolation(input.interpolation)
        {
            return Err(
                "render pipeline inter-stage interpolation types are incompatible".to_owned(),
            );
        }
        if effective_sampling(output.interpolation, output.sampling)
            != effective_sampling(input.interpolation, input.sampling)
        {
            return Err(
                "render pipeline inter-stage interpolation sampling is incompatible".to_owned(),
            );
        }
    }
    Ok(())
}

/// Returns the effective inter-stage interpolation, defaulting an unspecified
/// interpolation to `perspective` (the WebGPU default).
fn effective_interpolation(
    interpolation: Option<frontend::ReflectedInterpolation>,
) -> frontend::ReflectedInterpolation {
    interpolation.unwrap_or(frontend::ReflectedInterpolation::Perspective)
}

/// Returns the effective inter-stage sampling. For perspective/linear
/// interpolation an unspecified sampling defaults to `center`; for flat (and
/// per-vertex) interpolation the sampling carries as-is.
fn effective_sampling(
    interpolation: Option<frontend::ReflectedInterpolation>,
    sampling: Option<frontend::ReflectedSampling>,
) -> Option<frontend::ReflectedSampling> {
    match effective_interpolation(interpolation) {
        frontend::ReflectedInterpolation::Perspective
        | frontend::ReflectedInterpolation::Linear => {
            Some(sampling.unwrap_or(frontend::ReflectedSampling::Center))
        }
        frontend::ReflectedInterpolation::Flat => sampling,
    }
}

fn validate_inter_stage_limits(
    locations: &BTreeMap<u32, frontend::ReflectedIoLocation>,
    extra_builtins: u32,
    reserved_top_locations: u32,
    limits: Limits,
    label: &str,
) -> Result<(), String> {
    if locations.len() as u64 + u64::from(extra_builtins)
        > u64::from(limits.max_inter_stage_shader_variables)
    {
        return Err(format!(
            "render pipeline inter-stage {label} count exceeds the device limit"
        ));
    }
    let max_location = limits
        .max_inter_stage_shader_variables
        .saturating_sub(reserved_top_locations);
    if locations.keys().any(|location| *location >= max_location) {
        return Err(format!(
            "render pipeline inter-stage {label} location exceeds the device limit"
        ));
    }
    Ok(())
}

fn validate_clip_distances_size(clip_distances_size: u32) -> Result<(), String> {
    if clip_distances_size > 8 {
        return Err(
            "render pipeline vertex clip-distances array size exceeds the device limit".to_owned(),
        );
    }
    Ok(())
}

fn inter_stage_outputs(
    vertex: &RenderPipelineVertexState,
    vertex_entry: &str,
) -> Result<BTreeMap<u32, frontend::ReflectedIoLocation>, String> {
    let Some(module) = vertex.shader.module.reflected() else {
        return Err("vertex module reflection failed".to_owned());
    };
    Ok(module
        .entry_point_io(vertex_entry)
        .map(|io| {
            io.outputs
                .into_iter()
                .map(|output| (output.location, output))
                .collect()
        })
        .unwrap_or_default())
}

fn inter_stage_inputs(
    fragment: &RenderPipelineFragmentState,
    fragment_entry: &str,
) -> Result<BTreeMap<u32, frontend::ReflectedIoLocation>, String> {
    let Some(module) = fragment.shader.module.reflected() else {
        return Err("fragment module reflection failed".to_owned());
    };
    Ok(module
        .entry_point_io(fragment_entry)
        .map(|io| {
            io.inputs
                .into_iter()
                .map(|input| (input.location, input))
                .collect()
        })
        .unwrap_or_default())
}

fn vertex_clip_distances_size(
    vertex: &RenderPipelineVertexState,
    vertex_entry: &str,
) -> Result<u32, String> {
    let Some(module) = vertex.shader.module.reflected() else {
        return Err("vertex module reflection failed".to_owned());
    };
    Ok(module.vertex_clip_distances_size(vertex_entry))
}

/// Returns the number of fragment stage-input `@builtin`s that consume an
/// inter-stage shader-variable slot (F-063).
fn inter_stage_input_builtin_count(
    fragment: &RenderPipelineFragmentState,
    fragment_entry: &str,
) -> Result<u32, String> {
    let Some(module) = fragment.shader.module.reflected() else {
        return Err("fragment module reflection failed".to_owned());
    };
    Ok(module
        .entry_point_io(fragment_entry)
        .map(|io| io.input_inter_stage_builtins)
        .unwrap_or(0))
}

/// Validates render pipeline layout and returns a descriptive error on failure.
pub(crate) fn validate_render_pipeline_layout(
    descriptor: &RenderPipelineDescriptor,
    vertex_entry: &str,
    fragment_entry: Option<&str>,
) -> Result<(), String> {
    let RenderPipelineLayout::Explicit(layout) = &descriptor.layout else {
        return Ok(());
    };
    if layout.is_error() {
        return Err("render pipeline layout must not be an error pipeline layout".to_owned());
    }

    let mut requirements = stage_resource_bindings(
        &descriptor.vertex.shader,
        vertex_entry,
        PipelineShaderStage::Vertex,
    )?;
    if let Some(fragment) = &descriptor.fragment {
        if let Some(fragment_entry) = fragment_entry {
            requirements.extend(stage_resource_bindings(
                &fragment.shader,
                fragment_entry,
                PipelineShaderStage::Fragment,
            )?);
        }
    }
    validate_pipeline_layout_stage_bindings(layout, &requirements)
}

/// Returns the pipeline's effective user-immediate budget in bytes (Blocks
/// 93/94) WITHOUT the auto-layout `maxImmediateSize` bound. An explicit
/// layout contributes its declared `immediateSize`. An auto (default)
/// layout mirrors Dawn's default-layout rule
/// (`dawn/native/PipelineLayout.cpp:549,588-590,616` --
/// `PipelineLayoutBase::CreateDefault` sets the synthesized descriptor's
/// `immediateSize` to the max of the stages' reflected immediate usage,
/// vertex and fragment sharing one immediate block).
///
/// Callers: `resolve_render_pipeline_immediate_size` (validation -- adds
/// the auto-arm `maxImmediateSize` bound Dawn applies when the default
/// descriptor goes through `CreatePipelineLayout`) and
/// `select_render_shader_source` (HAL codegen -- runs only for pipelines
/// that already passed validation, so the bound is not re-checked there).
pub(crate) fn render_pipeline_layout_immediate_size(
    descriptor: &RenderPipelineDescriptor,
    vertex_entry: &str,
    fragment_entry: Option<&str>,
) -> Result<u32, String> {
    match &descriptor.layout {
        RenderPipelineLayout::Explicit(layout) => Ok(layout.immediate_size()),
        RenderPipelineLayout::Auto => {
            let vertex_module = descriptor.vertex.shader.module.reflected().ok_or_else(|| {
                "render pipeline stage requires a reflected shader module".to_owned()
            })?;
            let mut size = vertex_module.immediate_data_size(vertex_entry)?;
            if let (Some(fragment), Some(fragment_entry)) = (&descriptor.fragment, fragment_entry) {
                let fragment_module = fragment.shader.module.reflected().ok_or_else(|| {
                    "render pipeline stage requires a reflected shader module".to_owned()
                })?;
                size = size.max(fragment_module.immediate_data_size(fragment_entry)?);
            }
            Ok(size)
        }
    }
}

/// Resolves the pipeline's effective user-immediate byte budget for
/// validation (Blocks 93/94): [`render_pipeline_layout_immediate_size`]
/// plus, for auto layouts, the `immediateSize <= maxImmediateSize` bound
/// Dawn applies because `CreateDefault`'s synthesized descriptor goes
/// through the ordinary validated `CreatePipelineLayout`
/// (`dawn/native/PipelineLayout.cpp:634` -> `:144-147`), with the same
/// error message `createPipelineLayout` uses
/// (`validate_pipeline_layout_descriptor`). Explicit layouts were already
/// bounds-checked at their own creation.
pub(crate) fn resolve_render_pipeline_immediate_size(
    descriptor: &RenderPipelineDescriptor,
    vertex_entry: &str,
    fragment_entry: Option<&str>,
    limits: Limits,
) -> Result<u32, String> {
    let size = render_pipeline_layout_immediate_size(descriptor, vertex_entry, fragment_entry)?;
    if matches!(descriptor.layout, RenderPipelineLayout::Auto) && size > limits.max_immediate_size {
        return Err("pipeline layout immediateSize exceeds the device limit".to_owned());
    }
    Ok(size)
}

/// Validates that `entry_point`'s reflected immediate data size fits within
/// `immediate_size_budget` (Block 93). Thin wrapper around
/// [`validate_stage_immediate_size`] that resolves `stage`'s reflected
/// module first, matching [`stage_resource_bindings`]'s shape.
pub(crate) fn validate_render_stage_immediate_size(
    stage: &RenderPipelineShaderStage,
    entry_point: &str,
    immediate_size_budget: u32,
) -> Result<(), String> {
    let Some(module) = stage.module.reflected() else {
        return Err("render pipeline stage requires a reflected shader module".to_owned());
    };
    validate_stage_immediate_size(module, entry_point, immediate_size_budget)
}

/// Returns effective render bind group layouts.
pub(crate) fn effective_render_bind_group_layouts(
    descriptor: &RenderPipelineDescriptor,
    vertex_entry: &str,
    fragment_entry: Option<&str>,
    limits: Limits,
    features: &FeatureSet,
    pipeline_id: u64,
) -> Result<Vec<Arc<BindGroupLayout>>, String> {
    match &descriptor.layout {
        RenderPipelineLayout::Explicit(layout) => Ok(layout.bind_group_layouts().to_vec()),
        RenderPipelineLayout::Auto => {
            let mut requirements = stage_resource_bindings(
                &descriptor.vertex.shader,
                vertex_entry,
                PipelineShaderStage::Vertex,
            )?;
            if let Some(fragment) = &descriptor.fragment {
                if let Some(fragment_entry) = fragment_entry {
                    requirements.extend(stage_resource_bindings(
                        &fragment.shader,
                        fragment_entry,
                        PipelineShaderStage::Fragment,
                    )?);
                }
            }
            // Storage-texture format/access support is validated uniformly by
            // the derived bind-group-layout validation (`bind_group_layout.rs`,
            // via `FormatCaps`), so no render-specific format gate is needed.
            derive_bind_group_layouts(requirements, limits, features, pipeline_id)
        }
    }
}

/// Returns stage resource bindings.
pub(crate) fn stage_resource_bindings(
    stage: &RenderPipelineShaderStage,
    entry_point: &str,
    pipeline_stage: PipelineShaderStage,
) -> Result<Vec<StageResourceBinding>, String> {
    let Some(module) = stage.module.reflected() else {
        return Err("render pipeline stage requires a reflected shader module".to_owned());
    };
    Ok(module
        .resource_bindings_for_entry(entry_point)?
        .into_iter()
        .map(|binding| StageResourceBinding {
            stage: pipeline_stage,
            binding,
        })
        .collect())
}

/// Validates multisample state and returns a descriptive error on failure.
pub(crate) fn validate_multisample_state(
    descriptor: &RenderPipelineDescriptor,
    fragment_entry: Option<&str>,
) -> Result<(), String> {
    let multisample = descriptor.multisample;
    if !matches!(multisample.count, 1 | 4) {
        return Err("render pipeline multisample count must be 1 or 4".to_owned());
    }
    if multisample.alpha_to_coverage_enabled && multisample.count != 4 {
        return Err("render pipeline alphaToCoverage requires multisample count 4".to_owned());
    }
    if multisample.alpha_to_coverage_enabled {
        if let (Some(fragment), Some(entry_name)) = (&descriptor.fragment, fragment_entry) {
            let module = fragment
                .shader
                .module
                .reflected()
                .ok_or_else(|| "fragment module reflection failed".to_owned())?;
            if module
                .fragment_builtins(entry_name)
                .is_some_and(|builtins| builtins.sample_mask)
            {
                return Err(
                    "render pipeline alphaToCoverage conflicts with fragment sample_mask output"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pass::bind_group_layouts_compatible;
    #[cfg(feature = "tiled")]
    use crate::subpass::{
        AttachmentLayout, SubpassDependency, SubpassDependencyType, SubpassInputAttachment,
        SubpassLayoutDesc, SubpassPassLayoutDescriptor,
    };
    use crate::test_helpers::*;
    #[cfg(feature = "shader-passthrough")]
    use crate::ShaderStage;
    use crate::{Device, ErrorFilter, TextureViewDimension};

    use std::sync::Arc;

    fn primitive_state(unclipped_depth: bool) -> PrimitiveState {
        PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            unclipped_depth,
        }
    }

    /// Block 93 render analog: a vertex entry point that statically touches
    /// a `var<immediate>` is rejected when the EXPLICIT pipeline layout
    /// reserves a smaller `immediateSize` (here 0) -- but the same shader
    /// under an AUTO layout creates successfully, because Dawn's default
    /// layout sizes its immediate budget to the max of the stages' own
    /// reflected usage (`dawn/native/PipelineLayout.cpp:588-590,616`),
    /// which trivially covers each stage. Fires identically on Noop
    /// (backend-independent core validation; Noop `maxImmediateSize` = 64).
    #[test]
    fn render_pipeline_rejects_vertex_entry_point_immediate_data_size_exceeding_layout_budget() {
        let device = noop_device();
        assert_eq!(device.limits().max_immediate_size, 64);
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r#"
requires immediate_address_space;

var<immediate> pc : vec4f;

@vertex
fn vs() -> @builtin(position) vec4<f32> {
  return pc;
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#
                .to_owned(),
            )),
        );

        let mut descriptor = render_pipeline_descriptor(module.clone());
        descriptor.layout = RenderPipelineLayout::Explicit(explicit_empty_render_layout(&device));
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_render_pipeline(descriptor);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("vertex stage touching an over-budget immediate should be scoped");
        assert!(pipeline.is_error());
        assert_eq!(
            scoped.message,
            "shader entry point immediate data size exceeds the pipeline layout's immediateSize"
        );

        // Auto layout: budget = max stage usage (16 bytes) <= maxImmediateSize,
        // so the same shader creates successfully (Dawn CreateDefault rule).
        device.push_error_scope(ErrorFilter::Validation);
        let auto_pipeline = device.create_render_pipeline(render_pipeline_descriptor(module));
        let scoped_auto = device.pop_error_scope().expect("scope should exist");
        assert!(
            scoped_auto.is_none(),
            "auto-layout pipeline within maxImmediateSize must not error: {scoped_auto:?}"
        );
        assert!(!auto_pipeline.is_error());
    }

    /// Fragment-stage counterpart of the vertex test above: only the
    /// fragment entry point touches the `var<immediate>` (the vertex entry
    /// does not), so this exercises the fragment branch of the Block 93
    /// immediate-size check in `resolve_render_pipeline_descriptor` --
    /// explicit budget-0 rejection plus auto-layout success.
    #[test]
    fn render_pipeline_rejects_fragment_entry_point_immediate_data_size_exceeding_layout_budget() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r#"
requires immediate_address_space;

var<immediate> pc : vec4f;

@vertex
fn vs() -> @builtin(position) vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return pc;
}
"#
                .to_owned(),
            )),
        );

        let mut descriptor = render_pipeline_descriptor(module.clone());
        descriptor.layout = RenderPipelineLayout::Explicit(explicit_empty_render_layout(&device));
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_render_pipeline(descriptor);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("fragment stage touching an over-budget immediate should be scoped");
        assert!(pipeline.is_error());
        assert_eq!(
            scoped.message,
            "shader entry point immediate data size exceeds the pipeline layout's immediateSize"
        );

        // Auto layout: fragment usage (16 bytes) sizes the default budget,
        // so the same shader creates successfully.
        device.push_error_scope(ErrorFilter::Validation);
        let auto_pipeline = device.create_render_pipeline(render_pipeline_descriptor(module));
        let scoped_auto = device.pop_error_scope().expect("scope should exist");
        assert!(
            scoped_auto.is_none(),
            "auto-layout pipeline within maxImmediateSize must not error: {scoped_auto:?}"
        );
        assert!(!auto_pipeline.is_error());
    }

    #[test]
    fn validate_primitive_state_gates_unclipped_depth_on_depth_clip_control_feature() {
        let empty = FeatureSet::new();
        assert_eq!(
            validate_primitive_state(primitive_state(false), &empty),
            Ok(())
        );
        assert_eq!(
            validate_primitive_state(primitive_state(true), &empty),
            Err(
                "render pipeline unclippedDepth requires the depth-clip-control feature".to_owned()
            )
        );

        let mut features = FeatureSet::new();
        features.insert(Feature::DepthClipControl);
        assert_eq!(
            validate_primitive_state(primitive_state(false), &features),
            Ok(())
        );
        assert_eq!(
            validate_primitive_state(primitive_state(true), &features),
            Ok(())
        );
    }

    #[test]
    fn vertex_format_from_raw_pins_known_zero_and_unknown_values() {
        let known_values = 1..=0x29;
        for raw in known_values {
            let format = VertexFormat::from_raw(raw);
            assert_eq!(format, VertexFormat::from_raw(raw));
            assert_eq!(format.raw(), raw);
            assert_eq!(VertexFormat::from(raw), format);
            assert_eq!(u32::from(format), raw);
        }

        let zero = VertexFormat::from_raw(0);
        let unknown = VertexFormat::from_raw(0xFFFF);
        assert_eq!(VertexFormat::from(0_i32), zero);
        assert_eq!(i32::from(unknown), 0xFFFF);
        assert_eq!(zero.raw(), 0);
        assert_eq!(unknown.raw(), 0xFFFF);
        assert_eq!(zero.info().byte_size, 16);
        assert_eq!(unknown.info().byte_size, 16);
        assert_eq!(zero.info().output_class, FormatOutputClass::Float);
        assert_eq!(unknown.info().output_class, FormatOutputClass::Float);
    }

    #[test]
    fn hal_vertex_format_maps_representative_full_set_formats() {
        let cases = [
            (0x0000_0003, HalVertexFormat::Uint8x4),
            (0x0000_0009, HalVertexFormat::Unorm8x4),
            (0x0000_001B, HalVertexFormat::Float16x4),
            (0x0000_0023, HalVertexFormat::Uint32x4),
            (0x0000_0028, HalVertexFormat::Unorm10_10_10_2),
            (0x0000_0029, HalVertexFormat::Unorm8x4Bgra),
        ];

        for (raw, expected) in cases {
            assert_eq!(hal_vertex_format(VertexFormat::from_raw(raw)), expected);
        }
        assert_eq!(
            hal_vertex_format(VertexFormat::from_raw(0xFFFF)),
            HalVertexFormat::Unsupported
        );
    }

    #[test]
    fn validate_depth_stencil_treats_always_as_inert_for_stencil_only_formats() {
        let mut state = DepthStencilState {
            format: TextureFormat::from_raw(TextureFormat::STENCIL8),
            depth_write_enabled: Some(false),
            depth_compare: Some(CompareFunction::Always),
            stencil_front: StencilFaceState {
                compare: CompareFunction::Always,
                fail_op: StencilOperation::Keep,
                depth_fail_op: StencilOperation::Keep,
                pass_op: StencilOperation::Keep,
            },
            stencil_back: StencilFaceState {
                compare: CompareFunction::Always,
                fail_op: StencilOperation::Keep,
                depth_fail_op: StencilOperation::Keep,
                pass_op: StencilOperation::Keep,
            },
            stencil_read_mask: u32::MAX,
            stencil_write_mask: u32::MAX,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        };

        assert_eq!(
            validate_depth_stencil_aspects(state, &FeatureSet::default()),
            Ok(())
        );
        state.depth_compare = Some(CompareFunction::Less);
        assert_eq!(
            validate_depth_stencil_aspects(state, &FeatureSet::default()),
            Err("render pipeline depth test or write requires a depth format".to_owned())
        );
    }

    #[test]
    fn default_depth_state_is_valid_for_depth_less_format_but_non_default_is_not() {
        let device = noop_device();
        let state = DepthStencilState {
            format: TextureFormat::from_raw(TextureFormat::STENCIL8),
            depth_write_enabled: Some(false),
            depth_compare: Some(CompareFunction::Always),
            stencil_front: default_stencil_face_state(),
            stencil_back: default_stencil_face_state(),
            stencil_read_mask: u32::MAX,
            stencil_write_mask: 0,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        };
        assert_eq!(
            validate_depth_stencil_aspects(state, &device.features()),
            Ok(())
        );

        let mut descriptor = render_pipeline_descriptor(render_shader_module(&device));
        descriptor.depth_stencil = Some(state);
        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &device.features()),
            None
        );
        assert!(!device.create_render_pipeline(descriptor).is_error());

        let mut non_default = state;
        non_default.depth_compare = Some(CompareFunction::Less);
        assert_eq!(
            validate_depth_stencil_aspects(non_default, &device.features()),
            Err("render pipeline depth test or write requires a depth format".to_owned())
        );
    }

    #[test]
    fn default_stencil_state_is_valid_for_stencil_less_format_but_non_default_is_not() {
        let device = noop_device();
        let state = DepthStencilState {
            format: depth32_float(),
            depth_write_enabled: Some(false),
            depth_compare: Some(CompareFunction::Always),
            stencil_front: default_stencil_face_state(),
            stencil_back: default_stencil_face_state(),
            stencil_read_mask: u32::MAX,
            stencil_write_mask: 0,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        };
        assert_eq!(
            validate_depth_stencil_aspects(state, &device.features()),
            Ok(())
        );

        let mut descriptor = render_pipeline_descriptor(render_shader_module(&device));
        descriptor.depth_stencil = Some(state);
        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &device.features()),
            None
        );
        assert!(!device.create_render_pipeline(descriptor).is_error());

        let mut non_default = state;
        non_default.stencil_front.pass_op = StencilOperation::Replace;
        assert_eq!(
            validate_depth_stencil_aspects(non_default, &device.features()),
            Err("render pipeline stencil state requires a stencil format".to_owned())
        );
    }

    #[test]
    fn stencil_face_writes_only_on_non_keep_ops() {
        // The compare function is a test, not a write: an all-`Keep` face never
        // writes stencil even with a non-`Always` compare. This is the F-055
        // false-reject — `writes_stencil` previously keyed only on a non-zero
        // write mask and wrongly rejected such a pipeline against a read-only
        // stencil attachment.
        let keep = StencilFaceState {
            compare: CompareFunction::LessEqual,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            pass_op: StencilOperation::Keep,
        };
        assert!(!stencil_face_writes(keep));

        let mut pass_replace = keep;
        pass_replace.pass_op = StencilOperation::Replace;
        assert!(stencil_face_writes(pass_replace));

        let mut fail_replace = keep;
        fail_replace.fail_op = StencilOperation::Replace;
        assert!(stencil_face_writes(fail_replace));

        let mut depth_fail_replace = keep;
        depth_fail_replace.depth_fail_op = StencilOperation::Replace;
        assert!(stencil_face_writes(depth_fail_replace));
    }

    #[test]
    fn depth_compare_is_optional_when_depth_aspect_is_unused() {
        let keep_face = StencilFaceState {
            compare: CompareFunction::Always,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            pass_op: StencilOperation::Keep,
        };
        let base = || DepthStencilState {
            format: depth32_float(),
            depth_write_enabled: Some(false),
            depth_compare: None,
            stencil_front: keep_face,
            stencil_back: keep_face,
            stencil_read_mask: u32::MAX,
            stencil_write_mask: u32::MAX,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        };
        let features = FeatureSet::default();

        // F-058: depth neither written nor consulted -> `depthCompare` is optional.
        assert_eq!(validate_depth_stencil_aspects(base(), &features), Ok(()));

        // `depthWriteEnabled` is always required for a depth format.
        let mut missing_write = base();
        missing_write.depth_write_enabled = None;
        assert!(validate_depth_stencil_aspects(missing_write, &features).is_err());

        // Depth written -> `depthCompare` becomes required.
        let mut writes_depth = base();
        writes_depth.depth_write_enabled = Some(true);
        assert!(validate_depth_stencil_aspects(writes_depth, &features).is_err());

        // A non-`Keep` stencil depthFailOp consults the depth test -> required.
        let mut depth_fail = base();
        depth_fail.stencil_front.depth_fail_op = StencilOperation::Replace;
        assert!(validate_depth_stencil_aspects(depth_fail, &features).is_err());

        // Supplying `depthCompare` satisfies the used cases.
        let mut writes_depth_ok = base();
        writes_depth_ok.depth_write_enabled = Some(true);
        writes_depth_ok.depth_compare = Some(CompareFunction::Always);
        assert_eq!(
            validate_depth_stencil_aspects(writes_depth_ok, &features),
            Ok(())
        );
    }

    fn depth_stencil_state() -> DepthStencilState {
        DepthStencilState {
            format: depth32_float(),
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil_front: StencilFaceState {
                compare: CompareFunction::Always,
                fail_op: StencilOperation::Keep,
                depth_fail_op: StencilOperation::Keep,
                pass_op: StencilOperation::Keep,
            },
            stencil_back: StencilFaceState {
                compare: CompareFunction::Always,
                fail_op: StencilOperation::Keep,
                depth_fail_op: StencilOperation::Keep,
                pass_op: StencilOperation::Keep,
            },
            stencil_read_mask: u32::MAX,
            stencil_write_mask: u32::MAX,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }
    }

    fn framebuffer_fetch_render_wgsl(slot: u32) -> String {
        format!(
            r#"
enable chromium_experimental_framebuffer_fetch;

@vertex
fn vs() -> @builtin(position) vec4<f32> {{
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}}

@fragment
fn fs(@color({slot}) prev: vec4<f32>) -> @location(0) vec4<f32> {{
  return prev;
}}
"#
        )
    }

    fn dual_source_device() -> Device {
        noop_adapter()
            .create_device(None, &[Feature::DualSourceBlending], "", "")
            .expect("Noop adapter should create dual-source-blending device")
    }

    fn dual_source_render_shader_module(device: &Device) -> Arc<ShaderModule> {
        Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r#"
enable dual_source_blending;

@vertex
fn vs() -> @builtin(position) vec4f {
  return vec4f();
}

struct Out {
  @location(0) @blend_src(0) a: vec4f,
  @location(0) @blend_src(1) b: vec4f,
}

@fragment
fn fs() -> Out {
  return Out(vec4f(), vec4f());
}
"#
                .to_owned(),
            )),
        )
    }

    fn clip_distances_device() -> Device {
        noop_adapter()
            .create_device(None, &[Feature::ClipDistances], "", "")
            .expect("Noop adapter should create clip-distances device")
    }

    fn clip_inter_stage_wgsl(user_locations: u32, clip_distances_size: u32) -> String {
        let outputs = (0..user_locations)
            .map(|location| format!("  @location({location}) v{location}: f32,\n"))
            .collect::<String>();
        let output_values = (0..user_locations)
            .map(|_| "    0.0,\n".to_owned())
            .collect::<String>();
        let inputs = (0..user_locations)
            .map(|location| format!("  @location({location}) v{location}: f32,\n"))
            .collect::<String>();
        let clip_values = (0..clip_distances_size)
            .map(|_| "0.0")
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            r#"
enable clip_distances;

struct VsOut {{
  @builtin(position) pos: vec4f,
  @builtin(clip_distances) clip: array<f32, {clip_distances_size}>,
{outputs}}}

@vertex
fn vs() -> VsOut {{
  return VsOut(
    vec4f(),
    array<f32, {clip_distances_size}>({clip_values}),
{output_values}  );
}}

struct FsIn {{
{inputs}}}

@fragment
fn fs(input: FsIn) -> @location(0) vec4f {{
  _ = input;
  return vec4f();
}}
"#
        )
    }

    fn clip_inter_stage_location_wgsl(location: u32, clip_distances_size: u32) -> String {
        let clip_field = if clip_distances_size == 0 {
            String::new()
        } else {
            format!("  @builtin(clip_distances) clip: array<f32, {clip_distances_size}>,\n")
        };
        let clip_value = if clip_distances_size == 0 {
            String::new()
        } else {
            let clip_values = (0..clip_distances_size)
                .map(|_| "0.0")
                .collect::<Vec<_>>()
                .join(", ");
            format!("    array<f32, {clip_distances_size}>({clip_values}),\n")
        };
        let clip_enable = if clip_distances_size == 0 {
            String::new()
        } else {
            "enable clip_distances;\n".to_owned()
        };
        format!(
            r#"
{clip_enable}
struct VsOut {{
  @builtin(position) pos: vec4f,
{clip_field}  @location({location}) value: f32,
}}

@vertex
fn vs() -> VsOut {{
  return VsOut(
    vec4f(),
{clip_value}    0.0,
  );
}}

struct FsIn {{
  @location({location}) value: f32,
}}

@fragment
fn fs(input: FsIn) -> @location(0) vec4f {{
  _ = input;
  return vec4f();
}}
"#
        )
    }

    #[cfg(feature = "tiled")]
    fn subpass_attachment_layout(format: TextureFormat) -> AttachmentLayout {
        AttachmentLayout {
            format,
            sample_count: 1,
        }
    }

    #[cfg(feature = "tiled")]
    fn valid_two_subpass_deferred_layout_descriptor() -> SubpassPassLayoutDescriptor {
        SubpassPassLayoutDescriptor {
            color_attachments: vec![
                subpass_attachment_layout(rgba8_unorm()),
                subpass_attachment_layout(rgba8_unorm()),
            ],
            depth_stencil_attachment: None,
            subpasses: vec![
                SubpassLayoutDesc {
                    color_attachment_indices: vec![0],
                    uses_depth_stencil: false,
                    input_attachments: Vec::new(),
                },
                SubpassLayoutDesc {
                    color_attachment_indices: vec![1],
                    uses_depth_stencil: false,
                    input_attachments: vec![SubpassInputAttachment {
                        group: 0,
                        binding: 0,
                        source_subpass: 0,
                        source_attachment: 0,
                    }],
                },
            ],
            dependencies: vec![SubpassDependency {
                src_subpass: 0,
                dst_subpass: 1,
                dependency_type: SubpassDependencyType::ColorToInput,
                by_region: true,
            }],
            error: None,
        }
    }

    #[cfg(feature = "tiled")]
    fn render_shader_module_with_fragment_location(
        device: &crate::device::Device,
        location: u32,
    ) -> Arc<ShaderModule> {
        Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(format!(
                "@vertex
             fn vs() -> @builtin(position) vec4<f32> {{
                 return vec4<f32>(0.0, 0.0, 0.0, 1.0);
             }}

             @fragment
             fn fs() -> @location({location}) vec4<f32> {{
                 return vec4<f32>(1.0, 0.0, 0.0, 1.0);
             }}"
            ))),
        )
    }

    #[cfg(feature = "tiled")]
    fn subpass_pipeline_descriptor_for_subpass_one(
        device: &crate::device::Device,
        pass_layout: Arc<SubpassPassLayout>,
        location: u32,
    ) -> SubpassRenderPipelineDescriptor {
        SubpassRenderPipelineDescriptor {
            base: render_pipeline_descriptor(render_shader_module_with_fragment_location(
                device, location,
            )),
            pass_layout,
            subpass_index: 1,
            error: None,
        }
    }

    fn default_stencil_face_state() -> StencilFaceState {
        StencilFaceState {
            compare: CompareFunction::Always,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            pass_op: StencilOperation::Keep,
        }
    }

    #[test]
    fn render_pipeline_validation_accepts_color_depth_and_vertex_only_depth_only() {
        let device = noop_device();
        let limits = device.limits();
        let features = device.features();
        let module = render_shader_module(&device);
        let mut color_depth = render_pipeline_descriptor(Arc::clone(&module));
        color_depth.depth_stencil = Some(depth_stencil_state());
        assert_eq!(
            validate_render_pipeline_descriptor(&color_depth, limits, &features),
            None
        );
        assert!(!device.create_render_pipeline(color_depth).is_error());

        let vertex_only = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@vertex fn vs() -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }"
                .to_owned(),
            )),
        );
        let mut depth_only = render_pipeline_descriptor(vertex_only);
        depth_only.fragment = None;
        depth_only.depth_stencil = Some(depth_stencil_state());
        assert_eq!(
            validate_render_pipeline_descriptor(&depth_only, limits, &features),
            None
        );
        assert!(!device.create_render_pipeline(depth_only).is_error());

        let mut invalid = render_pipeline_descriptor(module);
        invalid.fragment = None;
        invalid.depth_stencil = None;
        assert_eq!(
            validate_render_pipeline_descriptor(&invalid, limits, &features),
            Some("render pipeline requires a fragment state or depthStencil state".to_owned())
        );
    }

    #[test]
    fn validate_blend_state_rejects_min_max_non_one_factors() {
        let valid = BlendState {
            color: BlendComponent {
                operation: BlendOperation::Min,
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
            },
            alpha: BlendComponent {
                operation: BlendOperation::Add,
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::Zero,
            },
        };
        let invalid = BlendState {
            color: BlendComponent {
                operation: BlendOperation::Max,
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::One,
            },
            alpha: valid.alpha,
        };

        assert_eq!(validate_blend_state(valid), Ok(()));
        assert_eq!(
            validate_blend_state(invalid),
            Err("render pipeline min/max blend operations require one factors".to_owned())
        );
    }

    #[test]
    fn blend_factor_is_dual_source_identifies_src1_factors() {
        assert!(!blend_factor_is_dual_source(BlendFactor::Src));
        assert!(blend_factor_is_dual_source(BlendFactor::Src1));
        assert!(blend_factor_is_dual_source(BlendFactor::OneMinusSrc1));
        assert!(blend_factor_is_dual_source(BlendFactor::Src1Alpha));
        assert!(blend_factor_is_dual_source(BlendFactor::OneMinusSrc1Alpha));
    }

    #[test]
    fn hal_render_pipeline_descriptor_maps_color_target_blend_and_write_mask() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        let blend = BlendState {
            color: BlendComponent {
                operation: BlendOperation::ReverseSubtract,
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusConstant,
            },
            alpha: BlendComponent {
                operation: BlendOperation::Add,
                src_factor: BlendFactor::Constant,
                dst_factor: BlendFactor::OneMinusDstAlpha,
            },
        };
        descriptor.fragment.as_mut().expect("fragment").targets[0] = ColorTargetState {
            format: rgba8_unorm(),
            blend: Some(blend),
            write_mask: 0b0101,
        };

        let hal = hal_render_pipeline_descriptor(&descriptor, &[]).expect("HAL render descriptor");

        assert_eq!(hal.sample_count, 1);
        assert_eq!(hal.color_targets.len(), 1);
        let target = hal.color_targets[0].expect("color target");
        assert_eq!(target.format, hal_texture_format(rgba8_unorm()));
        assert_eq!(target.write_mask, 0b0101);
        let blend = target.blend.expect("blend");
        assert_eq!(blend.color.operation, HalBlendOperation::ReverseSubtract);
        assert_eq!(blend.color.src_factor, HalBlendFactor::SrcAlpha);
        assert_eq!(blend.color.dst_factor, HalBlendFactor::OneMinusConstant);
        assert_eq!(blend.alpha.src_factor, HalBlendFactor::Constant);
        assert_eq!(blend.alpha.dst_factor, HalBlendFactor::OneMinusDstAlpha);
    }

    #[test]
    fn hal_render_pipeline_descriptor_preserves_undefined_color_target_holes() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        let fragment = descriptor.fragment.as_mut().expect("fragment");
        fragment.target_count = 2;
        fragment.targets = vec![
            ColorTargetState {
                format: TextureFormat::from_raw(TextureFormat::UNDEFINED),
                blend: None,
                write_mask: 0,
            },
            ColorTargetState {
                format: rgba8_unorm(),
                blend: None,
                write_mask: 0xF,
            },
        ];

        let hal = hal_render_pipeline_descriptor(&descriptor, &[]).expect("HAL render descriptor");

        assert_eq!(hal.color_targets.len(), 2);
        assert_eq!(hal.color_targets[0], None);
        assert_eq!(
            hal.color_targets[1].map(|target| target.format),
            Some(hal_texture_format(rgba8_unorm()))
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn hal_subpass_render_pipeline_descriptor_scatters_written_targets_to_global_slots() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let descriptor = render_pipeline_descriptor(module);
        let pass_layout = SubpassPassLayoutDescriptor {
            color_attachments: vec![
                AttachmentLayout {
                    format: rgba8_unorm(),
                    sample_count: 1,
                },
                AttachmentLayout {
                    format: rgba8_unorm(),
                    sample_count: 1,
                },
            ],
            depth_stencil_attachment: None,
            subpasses: vec![SubpassLayoutDesc {
                color_attachment_indices: vec![1],
                uses_depth_stencil: false,
                input_attachments: vec![SubpassInputAttachment {
                    group: 0,
                    binding: 0,
                    source_subpass: 0,
                    source_attachment: 0,
                }],
            }],
            dependencies: Vec::new(),
            error: None,
        };

        let hal = hal_subpass_render_pipeline_descriptor(
            &descriptor,
            &[],
            &pass_layout,
            0,
            &[((0, 0), 0)],
            HalBackend::Vulkan,
        )
        .expect("HAL subpass descriptor");

        assert_eq!(hal.color_targets.len(), 2);
        assert_eq!(hal.color_targets[0], None);
        assert_eq!(
            hal.color_targets[1].map(|target| target.format),
            Some(hal_texture_format(rgba8_unorm()))
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn hal_subpass_render_pipeline_descriptor_fills_metal_input_color_targets() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let descriptor = render_pipeline_descriptor(module);
        let pass_layout = SubpassPassLayoutDescriptor {
            color_attachments: vec![
                AttachmentLayout {
                    format: rgba8_unorm(),
                    sample_count: 1,
                },
                AttachmentLayout {
                    format: rgba8_unorm(),
                    sample_count: 1,
                },
            ],
            depth_stencil_attachment: None,
            subpasses: vec![SubpassLayoutDesc {
                color_attachment_indices: vec![1],
                uses_depth_stencil: false,
                input_attachments: vec![SubpassInputAttachment {
                    group: 0,
                    binding: 0,
                    source_subpass: 0,
                    source_attachment: 0,
                }],
            }],
            dependencies: Vec::new(),
            error: None,
        };

        let hal = hal_subpass_render_pipeline_descriptor(
            &descriptor,
            &[],
            &pass_layout,
            0,
            &[((0, 0), 0)],
            HalBackend::Metal,
        )
        .expect("HAL subpass descriptor");

        assert_eq!(hal.color_targets.len(), 2);
        assert_eq!(
            hal.color_targets[0],
            Some(HalColorTargetState {
                format: hal_texture_format(rgba8_unorm()),
                blend: None,
                write_mask: 0,
            })
        );
        assert_eq!(
            hal.color_targets[1].map(|target| target.format),
            Some(hal_texture_format(rgba8_unorm()))
        );
    }

    #[test]
    fn hal_render_pipeline_descriptor_carries_multisample_count() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.multisample.count = 4;

        let pipeline = device.create_render_pipeline(descriptor.clone());
        assert!(!pipeline.is_error());

        let hal = hal_render_pipeline_descriptor(&descriptor, &[]).expect("HAL render descriptor");
        assert_eq!(hal.sample_count, 4);
    }

    #[test]
    fn vertex_buffer_binding_and_hal_descriptor_skip_unused_slots() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.vertex.buffer_count = 8;
        descriptor.vertex.buffers = vec![
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: true,
                array_stride: 8,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![VertexAttribute {
                    format: VertexFormat::from_raw(0x0000_001D),
                    offset: 0,
                    shader_location: 2,
                }],
            },
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: true,
                array_stride: 16,
                step_mode: VertexStepMode::Instance,
                attributes: vec![VertexAttribute {
                    format: VertexFormat::from_raw(0x0000_001D),
                    offset: 0,
                    shader_location: 6,
                }],
            },
        ];
        let bindings = metal_vertex_buffer_binding_map(&descriptor.vertex.buffers, &[]);
        assert_eq!(
            bindings
                .iter()
                .map(|binding| (binding.slot, binding.metal_index))
                .collect::<Vec<_>>(),
            vec![(1, 0), (7, 1)]
        );

        let msl = msl_vertex_buffer_bindings(&descriptor.vertex.buffers, &bindings)
            .expect("MSL vertex buffers");
        assert_eq!(msl.len(), 2);
        assert_eq!(msl[0].slot, 1);
        assert_eq!(msl[1].slot, 7);

        let hal =
            hal_render_pipeline_descriptor(&descriptor, &bindings).expect("HAL render descriptor");
        assert_eq!(hal.vertex_buffers.len(), 2);
        assert_eq!(hal.vertex_buffers[0].slot, 1);
        assert_eq!(hal.vertex_buffers[0].attributes[0].metal_buffer_index, 0);
        assert_eq!(hal.vertex_buffers[1].slot, 7);
        assert_eq!(hal.vertex_buffers[1].attributes[0].metal_buffer_index, 1);
    }

    #[test]
    fn unused_vertex_buffer_slots_still_validate_array_stride_limit() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.vertex.buffer_count = 1;
        descriptor.vertex.buffers = vec![VertexBufferLayout {
            used: false,
            array_stride: u64::from(device.limits().max_vertex_buffer_array_stride) + 4,
            step_mode: VertexStepMode::Vertex,
            attributes: Vec::new(),
        }];

        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &device.features()),
            Some("render pipeline vertex buffer arrayStride exceeds the device limit".to_owned())
        );
    }

    #[test]
    fn unused_vertex_buffer_slots_still_validate_array_stride_alignment() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.vertex.buffer_count = 1;
        descriptor.vertex.buffers = vec![VertexBufferLayout {
            used: false,
            array_stride: 2,
            step_mode: VertexStepMode::Vertex,
            attributes: Vec::new(),
        }];

        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &device.features()),
            Some("render pipeline vertex buffer arrayStride must be a multiple of 4".to_owned())
        );
    }

    #[test]
    fn rgb10a2unorm_alpha_to_coverage_pipeline_is_accepted() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.multisample.count = 4;
        descriptor.multisample.alpha_to_coverage_enabled = true;
        descriptor.fragment.as_mut().expect("fragment").targets[0].format =
            TextureFormat::from_raw(TextureFormat::RGB10A2_UNORM);

        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &device.features()),
            None
        );
        assert!(!device.create_render_pipeline(descriptor).is_error());
    }

    #[test]
    fn color_target_bytes_per_sample_uses_cts_alignment_formula() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        let fragment = descriptor.fragment.as_mut().expect("fragment");
        fragment.targets = vec![
            ColorTargetState {
                format: TextureFormat::from_raw(TextureFormat::R8_UNORM),
                blend: None,
                write_mask: 0,
            },
            ColorTargetState {
                format: TextureFormat::from_raw(TextureFormat::R32_FLOAT),
                blend: None,
                write_mask: 0,
            },
            ColorTargetState {
                format: rgba8_unorm(),
                blend: None,
                write_mask: 0,
            },
            ColorTargetState {
                format: TextureFormat::from_raw(TextureFormat::RGBA32_FLOAT),
                blend: None,
                write_mask: 0,
            },
            ColorTargetState {
                format: TextureFormat::from_raw(TextureFormat::R8_UNORM),
                blend: None,
                write_mask: 0,
            },
        ];
        fragment.target_count = fragment.targets.len();

        let mut tight_limits = device.limits();
        tight_limits.max_color_attachment_bytes_per_sample = 32;
        assert_eq!(
            color_attachment_bytes_per_sample(fragment.targets.iter().map(|target| target.format)),
            Some(33)
        );
        assert!(
            validate_render_pipeline_descriptor(&descriptor, tight_limits, &device.features())
                .is_some()
        );

        tight_limits.max_color_attachment_bytes_per_sample = 33;
        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, tight_limits, &device.features()),
            None
        );
    }

    #[test]
    fn fragment_color_output_without_target_is_valid() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        let fragment = descriptor.fragment.as_mut().expect("fragment");
        fragment.targets.clear();
        fragment.target_count = 0;
        descriptor.depth_stencil = Some(depth_stencil_state());

        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &device.features()),
            None
        );
        assert!(!device.create_render_pipeline(descriptor).is_error());
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn framebuffer_fetch_color_input_with_matching_target_is_valid() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(framebuffer_fetch_render_wgsl(0))),
        );
        let descriptor = render_pipeline_descriptor(module);

        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &device.features()),
            None
        );
        assert!(!device.create_render_pipeline(descriptor).is_error());
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn framebuffer_fetch_color_input_without_matching_target_is_invalid() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(framebuffer_fetch_render_wgsl(1))),
        );
        let descriptor = render_pipeline_descriptor(module);

        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &device.features()),
            Some(
                "render pipeline framebuffer-fetch color input requires a declared color target"
                    .to_owned()
            )
        );
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_render_pipeline(descriptor);
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid pipeline should be scoped");
        assert!(pipeline.is_error());
        assert_eq!(
            error.message,
            "render pipeline framebuffer-fetch color input requires a declared color target"
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_render_pipeline_records_compatibility_for_subpass_one() {
        let device = noop_device();
        let pass_layout = Arc::new(
            device.create_subpass_pass_layout(valid_two_subpass_deferred_layout_descriptor()),
        );
        let descriptor =
            subpass_pipeline_descriptor_for_subpass_one(&device, Arc::clone(&pass_layout), 0);

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_subpass_render_pipeline(descriptor);
        let scoped = device.pop_error_scope().expect("scope should exist");

        assert!(!pipeline.is_error());
        assert_eq!(scoped, None);
        let compatibility = pipeline
            .subpass_compatibility()
            .expect("subpass pipeline should record compatibility");
        assert!(compatibility.pass_layout.same(&pass_layout));
        assert_eq!(compatibility.subpass_index, 1);
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_render_pipeline_rejects_fragment_output_outside_subpass_colors() {
        let device = noop_device();
        let pass_layout = Arc::new(
            device.create_subpass_pass_layout(valid_two_subpass_deferred_layout_descriptor()),
        );
        let descriptor = subpass_pipeline_descriptor_for_subpass_one(&device, pass_layout, 3);

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_subpass_render_pipeline(descriptor);
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid subpass pipeline should be scoped");

        assert!(pipeline.is_error());
        assert_eq!(
            error.message,
            "render pipeline color target without shader output must use writeMask 0"
        );
    }

    /// Block 94 Phase Review MAJOR 2: the tiled subpass vendor extension has
    /// no `SetImmediates` surface, so a subpass pipeline whose fragment
    /// stage statically uses `var<immediate>` data is rejected at creation
    /// (deterministic validation error) instead of silently reading zeroes
    /// at draw time.
    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_render_pipeline_rejects_immediate_data_usage() {
        let device = noop_device();
        assert_eq!(device.limits().max_immediate_size, 64);
        let pass_layout = Arc::new(
            device.create_subpass_pass_layout(valid_two_subpass_deferred_layout_descriptor()),
        );
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r#"
requires immediate_address_space;

var<immediate> pc : vec4f;

@vertex
fn vs() -> @builtin(position) vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return pc;
}
"#
                .to_owned(),
            )),
        );
        let descriptor = SubpassRenderPipelineDescriptor {
            base: render_pipeline_descriptor(module),
            pass_layout,
            subpass_index: 1,
            error: None,
        };

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_subpass_render_pipeline(descriptor);
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("subpass pipeline using var<immediate> should be scoped");

        assert!(pipeline.is_error());
        assert_eq!(
            error.message,
            "subpass render pipelines do not support immediate data (var<immediate>)"
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn regular_render_pipeline_has_no_subpass_compatibility() {
        let device = noop_device();
        let pipeline = device
            .create_render_pipeline(render_pipeline_descriptor(render_shader_module(&device)));

        assert!(!pipeline.is_error());
        assert!(pipeline.subpass_compatibility().is_none());
    }

    #[cfg(not(feature = "tiled"))]
    #[test]
    fn framebuffer_fetch_shader_module_requires_tiled_feature() {
        let device = noop_device();

        device.push_error_scope(ErrorFilter::Validation);
        let module =
            device.create_shader_module(ShaderModuleSource::Wgsl(framebuffer_fetch_render_wgsl(0)));
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid shader should be scoped");

        assert!(module.is_error());
        assert!(error
            .message
            .contains("chromium_experimental_framebuffer_fetch"));
    }

    #[test]
    fn fragment_without_color_target_or_depth_stencil_is_invalid() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        let fragment = descriptor.fragment.as_mut().expect("fragment");
        fragment.targets.clear();
        fragment.target_count = 0;

        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &device.features()),
            Some("render pipeline requires a color target or depthStencil state".to_owned())
        );
        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_render_pipeline(descriptor);
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid pipeline should be scoped");
        assert!(pipeline.is_error());
        assert_eq!(
            error.message,
            "render pipeline requires a color target or depthStencil state"
        );
    }

    #[test]
    fn alpha_blend_source_alpha_does_not_require_vec4_fragment_output() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
             @fragment fn fs() -> @location(0) f32 { return 1.0; }"
                    .to_owned(),
            )),
        );
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.fragment.as_mut().expect("fragment").targets[0] = ColorTargetState {
            format: TextureFormat::from_raw(TextureFormat::R8_UNORM),
            blend: Some(BlendState {
                color: BlendComponent {
                    operation: BlendOperation::Add,
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::Zero,
                },
                alpha: BlendComponent {
                    operation: BlendOperation::Add,
                    src_factor: BlendFactor::SrcAlpha,
                    dst_factor: BlendFactor::Zero,
                },
            }),
            write_mask: 0xF,
        };

        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &device.features()),
            None
        );
        assert!(!device.create_render_pipeline(descriptor).is_error());
    }

    #[test]
    fn sixteen_bit_float_color_targets_with_blend_are_valid() {
        let device = noop_device();
        for (format, output) in [
            (TextureFormat::R16_FLOAT, "f32"),
            (TextureFormat::RG16_FLOAT, "vec2f"),
            (TextureFormat::RGBA16_FLOAT, "vec4f"),
        ] {
            let module = Arc::new(
                device.create_shader_module(ShaderModuleSource::Wgsl(format!(
                    "@vertex fn vs() -> @builtin(position) vec4f {{ return vec4f(); }}
                 @fragment fn fs() -> @location(0) {output} {{ return {output}(); }}"
                ))),
            );
            let mut descriptor = render_pipeline_descriptor(module);
            descriptor.fragment.as_mut().expect("fragment").targets[0] = ColorTargetState {
                format: TextureFormat::from_raw(format),
                blend: Some(BlendState {
                    color: BlendComponent {
                        operation: BlendOperation::Add,
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                    },
                    alpha: BlendComponent {
                        operation: BlendOperation::Add,
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                    },
                }),
                write_mask: 0xF,
            };

            assert_eq!(
                validate_render_pipeline_descriptor(
                    &descriptor,
                    device.limits(),
                    &device.features()
                ),
                None
            );
            assert!(!device.create_render_pipeline(descriptor).is_error());
        }
    }

    #[test]
    fn rgba32_float_color_target_with_blend_requires_float32_blendable_feature() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
                 @fragment fn fs() -> @location(0) vec4f { return vec4f(); }"
                    .to_owned(),
            )),
        );
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.fragment.as_mut().expect("fragment").targets[0] = ColorTargetState {
            format: TextureFormat::from_raw(TextureFormat::RGBA32_FLOAT),
            blend: Some(BlendState {
                color: BlendComponent {
                    operation: BlendOperation::Add,
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::Zero,
                },
                alpha: BlendComponent {
                    operation: BlendOperation::Add,
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::Zero,
                },
            }),
            write_mask: 0xF,
        };

        let empty = FeatureSet::new();
        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &empty),
            Some("render pipeline color target format must be blendable".to_owned())
        );

        let mut enabled = FeatureSet::new();
        enabled.insert(Feature::Float32Blendable);
        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &enabled),
            None
        );
    }

    #[test]
    fn dual_source_blend_factor_requires_feature_and_accepts_when_enabled() {
        let device = dual_source_device();
        let module = dual_source_render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.fragment.as_mut().expect("fragment").targets[0].blend = Some(BlendState {
            color: BlendComponent {
                operation: BlendOperation::Add,
                src_factor: BlendFactor::Src1,
                dst_factor: BlendFactor::OneMinusSrc1,
            },
            alpha: BlendComponent {
                operation: BlendOperation::Add,
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::Zero,
            },
        });

        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &FeatureSet::new()),
            Some(
                "render pipeline dual-source blend factors require the dual-source-blending feature"
                    .to_owned()
            )
        );

        let mut enabled = FeatureSet::new();
        enabled.insert(Feature::DualSourceBlending);
        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &enabled),
            None
        );
    }

    #[test]
    fn dual_source_blend_factor_requires_fragment_blend_src_1_output() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.fragment.as_mut().expect("fragment").targets[0].blend = Some(BlendState {
            color: BlendComponent {
                operation: BlendOperation::Add,
                src_factor: BlendFactor::Src1,
                dst_factor: BlendFactor::Zero,
            },
            alpha: BlendComponent {
                operation: BlendOperation::Add,
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::Zero,
            },
        });

        let mut enabled = FeatureSet::new();
        enabled.insert(Feature::DualSourceBlending);
        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &enabled),
            Some(
                "render pipeline dual-source blend factors require the fragment shader to write a @blend_src(1) output"
                    .to_owned()
            )
        );
    }

    #[test]
    fn dual_source_blending_requires_single_color_target() {
        let device = dual_source_device();
        let module = dual_source_render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        let fragment = descriptor.fragment.as_mut().expect("fragment");
        fragment.target_count = 2;
        fragment.targets = vec![
            ColorTargetState {
                format: rgba8_unorm(),
                blend: Some(BlendState {
                    color: BlendComponent {
                        operation: BlendOperation::Add,
                        src_factor: BlendFactor::Src1Alpha,
                        dst_factor: BlendFactor::OneMinusSrc1Alpha,
                    },
                    alpha: BlendComponent {
                        operation: BlendOperation::Add,
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                    },
                }),
                write_mask: 0xF,
            },
            ColorTargetState {
                format: rgba8_unorm(),
                blend: None,
                write_mask: 0xF,
            },
        ];

        let mut enabled = FeatureSet::new();
        enabled.insert(Feature::DualSourceBlending);
        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &enabled),
            Some("dual-source blending requires a single color target".to_owned())
        );
    }

    #[test]
    fn two_non_dual_source_color_targets_are_valid() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
                 struct Out {
                   @location(0) a: vec4f,
                   @location(1) b: vec4f,
                 }
                 @fragment fn fs() -> Out { return Out(vec4f(), vec4f()); }"
                    .to_owned(),
            )),
        );
        let mut descriptor = render_pipeline_descriptor(module);
        let fragment = descriptor.fragment.as_mut().expect("fragment");
        fragment.target_count = 2;
        fragment.targets = vec![
            ColorTargetState {
                format: rgba8_unorm(),
                blend: Some(BlendState {
                    color: BlendComponent {
                        operation: BlendOperation::Add,
                        src_factor: BlendFactor::SrcAlpha,
                        dst_factor: BlendFactor::OneMinusSrcAlpha,
                    },
                    alpha: BlendComponent {
                        operation: BlendOperation::Add,
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::Zero,
                    },
                }),
                write_mask: 0xF,
            },
            ColorTargetState {
                format: rgba8_unorm(),
                blend: None,
                write_mask: 0xF,
            },
        ];

        assert_eq!(
            validate_render_pipeline_descriptor(&descriptor, device.limits(), &FeatureSet::new()),
            None
        );
    }

    #[test]
    fn validate_inter_stage_interface_rejects_missing_fragment_input() {
        let device = noop_device();
        let vertex = Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(
            "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }".to_owned(),
        )));
        let fragment = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "struct In { @location(0) value: f32, }
             @fragment fn main(input: In) -> @location(0) vec4f {
                 _ = input;
                 return vec4f();
             }"
                .to_owned(),
            )),
        );
        let descriptor = render_pipeline_descriptor(vertex);
        let mut descriptor = descriptor;
        descriptor.fragment.as_mut().unwrap().shader.module = fragment;

        assert_eq!(
            validate_inter_stage_interface(&descriptor, "main", Some("main"), device.limits()),
            Err("render pipeline fragment input has no matching vertex output".to_owned())
        );
    }

    #[test]
    fn validate_inter_stage_interface_counts_clip_distance_slots_at_limit() {
        let device = clip_distances_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(clip_inter_stage_wgsl(14, 5))),
        );
        assert!(!module.is_error());
        let descriptor = render_pipeline_descriptor(module);

        assert_eq!(
            validate_inter_stage_interface(&descriptor, "vs", Some("fs"), device.limits()),
            Ok(())
        );
    }

    #[test]
    fn validate_inter_stage_interface_rejects_clip_distance_slot_overflow() {
        let device = clip_distances_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(clip_inter_stage_wgsl(15, 5))),
        );
        assert!(!module.is_error());
        let descriptor = render_pipeline_descriptor(module);

        assert_eq!(
            validate_inter_stage_interface(&descriptor, "vs", Some("fs"), device.limits()),
            Err("render pipeline inter-stage output count exceeds the device limit".to_owned())
        );
    }

    #[test]
    fn validate_inter_stage_interface_reserves_top_locations_for_clip_distances() {
        let device = clip_distances_device();
        let limits = device.limits();
        let top_location = limits.max_inter_stage_shader_variables - 1;
        let reserved_top_location = top_location - 1;

        let module = Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(
            clip_inter_stage_location_wgsl(top_location, 0),
        )));
        assert!(!module.is_error());
        let descriptor = render_pipeline_descriptor(module);
        assert_eq!(
            validate_inter_stage_interface(&descriptor, "vs", Some("fs"), limits),
            Ok(())
        );

        for clip_distances_size in 1..=4 {
            let module = Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(
                clip_inter_stage_location_wgsl(top_location, clip_distances_size),
            )));
            assert!(!module.is_error());
            let descriptor = render_pipeline_descriptor(module);
            assert_eq!(
                validate_inter_stage_interface(&descriptor, "vs", Some("fs"), limits),
                Err(
                    "render pipeline inter-stage output location exceeds the device limit"
                        .to_owned()
                )
            );
        }

        let module = Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(
            clip_inter_stage_location_wgsl(reserved_top_location, 1),
        )));
        assert!(!module.is_error());
        let descriptor = render_pipeline_descriptor(module);
        assert_eq!(
            validate_inter_stage_interface(&descriptor, "vs", Some("fs"), limits),
            Ok(())
        );
    }

    #[test]
    fn validate_clip_distances_size_rejects_over_spec_limit() {
        assert_eq!(
            validate_clip_distances_size(9),
            Err(
                "render pipeline vertex clip-distances array size exceeds the device limit"
                    .to_owned()
            )
        );
        assert_eq!(validate_clip_distances_size(8), Ok(()));
    }

    #[test]
    fn select_render_shader_source_metal_generates_per_stage_msl_for_separate_modules() {
        let device = noop_device();
        let vertex = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@vertex fn vs() -> @builtin(position) vec4<f32> {
                    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }"
                .to_owned(),
            )),
        );
        let fragment = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@fragment fn fs() -> @location(0) vec4<f32> {
                    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
                }"
                .to_owned(),
            )),
        );
        let mut descriptor = render_pipeline_descriptor(vertex);
        descriptor
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = fragment;

        let (source, vertex_entry, fragment_entry, bindings) = select_render_shader_source(
            HalBackend::Metal,
            &descriptor,
            "vs",
            Some("fs"),
            &[],
            &[],
            &[],
            false,
            &[],
        )
        .expect("separate WGSL modules should generate per-stage MSL");

        assert!(matches!(
            source,
            HalShaderSource::MslStagesWithBufferSizes { vertex, fragment, .. }
                if vertex.contains("vertex")
                    && fragment.as_ref().is_some_and(|fragment| fragment.contains("fragment"))
        ));
        let expected_vertex_entry = "tint_vs";
        let expected_fragment_entry = "tint_fs";
        assert_eq!(vertex_entry, expected_vertex_entry);
        assert_eq!(fragment_entry.as_deref(), Some(expected_fragment_entry));
        assert!(bindings.is_empty());
    }

    /// Block 94 S2: a pipeline layout with a non-zero `immediateSize`, whose
    /// fragment entry point both references the user `var<immediate>` and
    /// clamps `frag_depth`, threads `HalMslImmediates` all the way from
    /// Tint's reflection into `HalShaderSource::MslStagesWithBufferSizes`
    /// (mirrors how `fragment_frag_depth_clamp_slot` used to be threaded,
    /// generalized to the combined user+clamp block).
    #[test]
    fn select_render_shader_source_metal_threads_immediate_metadata_to_hal_shader_source() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r#"
requires immediate_address_space;

var<immediate> pc : vec4f;

@vertex
fn vs() -> @builtin(position) vec4<f32> {
  return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @builtin(frag_depth) f32 {
  return pc.x;
}
"#
                .to_owned(),
            )),
        );
        let mut descriptor = render_pipeline_descriptor(module);
        // `pc` is a vec4f (16 bytes); the pipeline layout reserves exactly
        // that much user-immediate budget.
        descriptor.layout = RenderPipelineLayout::Explicit(Arc::new(
            device.create_pipeline_layout(PipelineLayoutDescriptor {
                bind_group_layouts: Vec::new(),
                immediate_size: 16,
                error: None,
            }),
        ));

        let (source, ..) = select_render_shader_source(
            HalBackend::Metal,
            &descriptor,
            "vs",
            Some("fs"),
            &[],
            &[],
            &[],
            false,
            &[],
        )
        .expect("render pipeline with a user immediate should select Metal MSL");

        let HalShaderSource::MslStagesWithBufferSizes {
            vertex_immediates,
            fragment_immediates,
            ..
        } = source
        else {
            panic!("Metal backend must select MslStagesWithBufferSizes");
        };

        // The vertex entry point never references `pc`, so it carries no
        // immediates of its own.
        assert_eq!(vertex_immediates, None);
        // The fragment entry point references `pc` (16 bytes) AND clamps
        // frag_depth: one combined block, clamp appended right after the
        // full 16-byte layout-reserved user prefix (not colliding with it).
        let fragment_immediates =
            fragment_immediates.expect("fragment references pc and clamps frag_depth");
        assert_eq!(fragment_immediates.frag_depth_clamp_offset, Some(16));
        assert_eq!(fragment_immediates.block_size, 16 + 8);
    }

    #[cfg(feature = "shader-passthrough")]
    fn msl_render_module(
        device: &Device,
        source: &str,
        entries: Vec<MslEntryPoint>,
    ) -> Arc<ShaderModule> {
        Arc::new(
            device.create_shader_module(ShaderModuleSource::MslPassthrough {
                source: source.to_owned(),
                entries,
            }),
        )
    }

    #[cfg(feature = "shader-passthrough")]
    fn msl_render_entry(name: &str, stage: ShaderStage) -> MslEntryPoint {
        MslEntryPoint {
            name: name.to_owned(),
            stage,
            workgroup_size: [0, 0, 0],
        }
    }

    #[cfg(feature = "shader-passthrough")]
    fn spirv_render_module(device: &Device) -> Arc<ShaderModule> {
        Arc::new(
            device.create_shader_module(ShaderModuleSource::SpirvPassthrough(vec![
                0x0723_0203,
                0,
                0,
                0,
                0,
            ])),
        )
    }

    fn explicit_empty_render_layout(device: &Device) -> Arc<PipelineLayout> {
        Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: Vec::new(),
            immediate_size: 0,
            error: None,
        }))
    }

    #[cfg(feature = "shader-passthrough")]
    fn msl_render_passthrough_descriptor(device: &Device) -> RenderPipelineDescriptor {
        let vertex = msl_render_module(
            device,
            "vertex msl source",
            vec![msl_render_entry("vs", ShaderStage::Vertex)],
        );
        let fragment = msl_render_module(
            device,
            "fragment msl source",
            vec![msl_render_entry("fs", ShaderStage::Fragment)],
        );
        let mut descriptor = render_pipeline_descriptor(vertex);
        descriptor.layout = RenderPipelineLayout::Explicit(explicit_empty_render_layout(device));
        descriptor
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = fragment;
        descriptor
    }

    #[cfg(feature = "shader-passthrough")]
    fn spirv_render_passthrough_descriptor(device: &Device) -> RenderPipelineDescriptor {
        let vertex = spirv_render_module(device);
        let fragment = spirv_render_module(device);
        let mut descriptor = render_pipeline_descriptor(vertex);
        descriptor.layout = RenderPipelineLayout::Explicit(explicit_empty_render_layout(device));
        descriptor
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = fragment;
        descriptor
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn resolve_shader_passthrough_render_pipeline_uses_explicit_layout_and_entries() {
        let device = noop_device();
        let descriptor = msl_render_passthrough_descriptor(&device);

        let (vertex_entry, fragment_entry, layouts) =
            resolve_shader_passthrough_render_pipeline_descriptor(&descriptor)
                .expect("MSL render passthrough resolve");

        assert_eq!(vertex_entry, "vs");
        assert_eq!(fragment_entry.as_deref(), Some("fs"));
        assert_eq!(layouts.len(), 0);
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_passthrough_render_pipeline_builds_on_noop_with_explicit_layout() {
        let device = noop_device();
        let descriptor = msl_render_passthrough_descriptor(&device);

        let pipeline = device.create_render_pipeline(descriptor);

        assert!(!pipeline.is_error());
        assert_eq!(pipeline.vertex_entry_name(), "vs");
        assert_eq!(pipeline.fragment_entry_name(), Some("fs"));
        assert!(pipeline.bind_group_layouts().is_empty());
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_passthrough_render_pipeline_rejects_auto_layout() {
        let device = noop_device();
        let mut descriptor = msl_render_passthrough_descriptor(&device);
        descriptor.layout = RenderPipelineLayout::Auto;

        let err = validate_render_pipeline_descriptor(
            &descriptor,
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("auto layout should fail");

        assert_eq!(
            err,
            "shader passthrough requires an explicit pipeline layout"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_passthrough_render_pipeline_rejects_pipeline_constants() {
        let device = noop_device();
        let mut descriptor = msl_render_passthrough_descriptor(&device);
        descriptor.vertex.shader.constants.push(PipelineConstant {
            key: "x".to_owned(),
            value: 1.0,
        });

        let err = validate_render_pipeline_descriptor(
            &descriptor,
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("constants should fail");

        assert_eq!(
            err,
            "pipeline-overridable constants are not supported with shader passthrough"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_passthrough_render_pipeline_rejects_mixed_fragment_module() {
        let device = noop_device();
        let mut descriptor = msl_render_passthrough_descriptor(&device);
        descriptor
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(
            "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }".to_owned(),
        )));

        let err = validate_render_pipeline_descriptor(
            &descriptor,
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("mixed passthrough/reflected stages should fail");

        assert_eq!(
            err,
            "render shader passthrough requires all stages to be passthrough modules of the same kind"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_passthrough_render_pipeline_rejects_mixed_vertex_module() {
        let device = noop_device();
        let vertex = Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(); }".to_owned(),
        )));
        let fragment = msl_render_module(
            &device,
            "fragment msl source",
            vec![msl_render_entry("fs", ShaderStage::Fragment)],
        );
        let mut descriptor = render_pipeline_descriptor(vertex);
        descriptor.layout = RenderPipelineLayout::Explicit(explicit_empty_render_layout(&device));
        descriptor
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = fragment;

        let err = validate_render_pipeline_descriptor(
            &descriptor,
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("mixed reflected/passthrough stages should fail");

        assert_eq!(
            err,
            "render shader passthrough requires all stages to be passthrough modules of the same kind"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_passthrough_render_pipeline_rejects_vertex_only_without_depth_or_fragment() {
        let device = noop_device();
        let mut descriptor = msl_render_passthrough_descriptor(&device);
        descriptor.fragment = None;
        descriptor.depth_stencil = None;

        let err = validate_render_pipeline_descriptor(
            &descriptor,
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("vertex-only render pipeline should fail");

        assert_eq!(
            err,
            "render pipeline requires a fragment state or depthStencil state"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn spirv_passthrough_render_pipeline_builds_on_noop_with_explicit_layout() {
        let device = noop_device();
        let descriptor = spirv_render_passthrough_descriptor(&device);

        let pipeline = device.create_render_pipeline(descriptor);

        assert!(!pipeline.is_error());
        assert_eq!(pipeline.vertex_entry_name(), "vs");
        assert_eq!(pipeline.fragment_entry_name(), Some("fs"));
        assert!(pipeline.bind_group_layouts().is_empty());
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn spirv_passthrough_render_pipeline_rejects_auto_layout() {
        let device = noop_device();
        let mut descriptor = spirv_render_passthrough_descriptor(&device);
        descriptor.layout = RenderPipelineLayout::Auto;

        let err = validate_render_pipeline_descriptor(
            &descriptor,
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("auto layout should fail");

        assert_eq!(
            err,
            "shader passthrough requires an explicit pipeline layout"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn spirv_passthrough_render_pipeline_rejects_pipeline_constants() {
        let device = noop_device();
        let mut descriptor = spirv_render_passthrough_descriptor(&device);
        descriptor.vertex.shader.constants.push(PipelineConstant {
            key: "x".to_owned(),
            value: 1.0,
        });

        let err = validate_render_pipeline_descriptor(
            &descriptor,
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("constants should fail");

        assert_eq!(
            err,
            "pipeline-overridable constants are not supported with shader passthrough"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn spirv_passthrough_render_pipeline_rejects_msl_fragment_module() {
        let device = noop_device();
        let mut descriptor = spirv_render_passthrough_descriptor(&device);
        descriptor
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = msl_render_module(
            &device,
            "fragment msl source",
            vec![msl_render_entry("fs", ShaderStage::Fragment)],
        );

        let err = validate_render_pipeline_descriptor(
            &descriptor,
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("mixed SPIR-V/MSL passthrough stages should fail");

        assert_eq!(
            err,
            "render shader passthrough requires all stages to be passthrough modules of the same kind"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_passthrough_render_pipeline_rejects_spirv_fragment_module() {
        let device = noop_device();
        let mut descriptor = msl_render_passthrough_descriptor(&device);
        descriptor
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = spirv_render_module(&device);

        let err = validate_render_pipeline_descriptor(
            &descriptor,
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("mixed MSL/SPIR-V passthrough stages should fail");

        assert_eq!(
            err,
            "render shader passthrough requires all stages to be passthrough modules of the same kind"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn spirv_passthrough_render_pipeline_rejects_wgsl_fragment_module() {
        let device = noop_device();
        let mut descriptor = spirv_render_passthrough_descriptor(&device);
        descriptor
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(
            "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }".to_owned(),
        )));

        let err = validate_render_pipeline_descriptor(
            &descriptor,
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("mixed SPIR-V/reflected stages should fail");

        assert_eq!(
            err,
            "render shader passthrough requires all stages to be passthrough modules of the same kind"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn wgsl_render_pipeline_rejects_spirv_fragment_module() {
        let device = noop_device();
        let vertex = Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(); }".to_owned(),
        )));
        let fragment = spirv_render_module(&device);
        let mut descriptor = render_pipeline_descriptor(vertex);
        descriptor.layout = RenderPipelineLayout::Explicit(explicit_empty_render_layout(&device));
        descriptor
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = fragment;

        let err = validate_render_pipeline_descriptor(
            &descriptor,
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("mixed reflected/SPIR-V stages should fail");

        assert_eq!(
            err,
            "render shader passthrough requires all stages to be passthrough modules of the same kind"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn select_render_shader_source_metal_uses_msl_passthrough_stages_verbatim() {
        let device = noop_device();
        let descriptor = msl_render_passthrough_descriptor(&device);

        let (source, vertex_entry, fragment_entry, bindings) = select_render_shader_source(
            HalBackend::Metal,
            &descriptor,
            "vs",
            Some("fs"),
            &[],
            &[],
            &[],
            false,
            &[],
        )
        .expect("MSL passthrough should select Metal MSL stages");

        let HalShaderSource::MslStages { vertex, fragment } = source else {
            panic!("Metal should select plain MSL stages");
        };
        assert_eq!(vertex, "vertex msl source");
        assert_eq!(fragment.as_deref(), Some("fragment msl source"));
        assert_eq!(vertex_entry, "vs");
        assert_eq!(fragment_entry.as_deref(), Some("fs"));
        assert!(bindings.is_empty());
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn select_render_shader_source_rejects_msl_passthrough_on_vulkan() {
        let device = noop_device();
        let descriptor = msl_render_passthrough_descriptor(&device);

        let err = select_render_shader_source(
            HalBackend::Vulkan,
            &descriptor,
            "vs",
            Some("fs"),
            &[],
            &[],
            &[],
            false,
            &[],
        )
        .expect_err("MSL passthrough must reject Vulkan");

        assert_eq!(err, "MSL passthrough shader requires the Metal backend");
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn select_render_shader_source_vulkan_uses_spirv_passthrough_stages_verbatim() {
        let device = noop_device();
        let descriptor = spirv_render_passthrough_descriptor(&device);

        let (source, vertex_entry, fragment_entry, bindings) = select_render_shader_source(
            HalBackend::Vulkan,
            &descriptor,
            "vs",
            Some("fs"),
            &[],
            &[],
            &[],
            false,
            &[],
        )
        .expect("SPIR-V passthrough should select Vulkan SPIR-V stages");

        let HalShaderSource::SpirVStages { vertex, fragment } = source else {
            panic!("Vulkan should select SPIR-V stages");
        };
        assert_eq!(vertex, vec![0x0723_0203, 0, 0, 0, 0]);
        assert_eq!(fragment, Some(vec![0x0723_0203, 0, 0, 0, 0]));
        assert_eq!(vertex_entry, "vs");
        assert_eq!(fragment_entry.as_deref(), Some("fs"));
        assert!(bindings.is_empty());
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn select_render_shader_source_rejects_spirv_passthrough_on_metal() {
        let device = noop_device();
        let descriptor = spirv_render_passthrough_descriptor(&device);

        let err = select_render_shader_source(
            HalBackend::Metal,
            &descriptor,
            "vs",
            Some("fs"),
            &[],
            &[],
            &[],
            false,
            &[],
        )
        .expect_err("SPIR-V passthrough must reject Metal");

        assert_eq!(err, "SPIR-V passthrough shader requires the Vulkan backend");
    }

    #[cfg(feature = "gles")]
    #[test]
    fn select_render_shader_source_generates_gles_glsl_stages() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "struct VertexOut {
                 @builtin(position) position: vec4<f32>,
             }

             @vertex
             fn vs(@location(0) position: vec2<f32>) -> VertexOut {
                 var out: VertexOut;
                 out.position = vec4<f32>(position, 0.0, 1.0);
                 return out;
             }

             @fragment
             fn fs() -> @location(0) vec4<f32> {
                 return vec4<f32>(1.0, 0.0, 0.0, 1.0);
             }"
                .to_owned(),
            )),
        );
        let descriptor = render_pipeline_descriptor(module);

        let (source, vertex_entry, fragment_entry, bindings) = select_render_shader_source(
            HalBackend::Gles,
            &descriptor,
            "vs",
            Some("fs"),
            &[],
            &[],
            &[],
            false,
            &[],
        )
        .expect("WGSL should generate GLES GLSL stages");

        assert!(
            matches!(source, HalShaderSource::GlslStages { vertex, fragment }
                if vertex.contains("#version 310 es")
                    && fragment.as_ref().is_some_and(|fragment| fragment.contains("#version 310 es"))
                    && vertex.contains("void main()")
                    && fragment.as_ref().is_some_and(|fragment| fragment.contains("void main()")))
        );
        assert_eq!(vertex_entry, "vs");
        assert_eq!(fragment_entry.as_deref(), Some("fs"));
        assert!(bindings.is_empty());
    }

    #[test]
    fn auto_render_pipeline_layouts_are_exclusive_per_pipeline_on_noop() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "struct Uniforms {
                    value: vec4<f32>,
                };

                @group(2) @binding(0) var<uniform> u2: Uniforms;
                @group(3) @binding(0) var<uniform> u3: Uniforms;

                @vertex
                fn vs() -> @builtin(position) vec4<f32> {
                    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }

                @fragment
                fn fs() -> @location(0) vec4<f32> {
                    return u2.value + u3.value * 0.0;
                }"
                .to_owned(),
            )),
        );

        let first = device.create_render_pipeline(render_pipeline_descriptor(Arc::clone(&module)));
        let second = device.create_render_pipeline(render_pipeline_descriptor(module));

        assert!(!first.is_error());
        assert!(!second.is_error());
        assert_eq!(first.bind_group_layouts().len(), 4);
        assert_eq!(second.bind_group_layouts().len(), 4);

        let first_group_2 = &first.bind_group_layouts()[2];
        let first_group_3 = &first.bind_group_layouts()[3];
        let second_group_2 = &second.bind_group_layouts()[2];
        let first_id = first_group_2
            .exclusive_pipeline()
            .expect("auto layout should carry an exclusive pipeline id");

        assert_eq!(first_group_3.exclusive_pipeline(), Some(first_id));
        assert_ne!(second_group_2.exclusive_pipeline(), Some(first_id));
        assert!(bind_group_layouts_compatible(first_group_2, first_group_2));
        assert!(bind_group_layouts_compatible(first_group_2, first_group_3));
        assert!(!bind_group_layouts_compatible(
            first_group_2,
            second_group_2
        ));
    }

    #[test]
    fn external_texture_render_pipeline_auto_layout_and_explicit_compat() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@group(0) @binding(0) var tex: texture_external;
             @vertex
             fn vs() -> @builtin(position) vec4<f32> {
                 return vec4<f32>(0.0, 0.0, 0.0, 1.0);
             }
             @fragment
             fn fs() -> @location(0) vec4<f32> {
                 return textureLoad(tex, vec2<i32>(0, 0));
             }"
                .to_owned(),
            )),
        );

        let auto = device.create_render_pipeline(render_pipeline_descriptor(Arc::clone(&module)));
        assert!(!auto.is_error());
        assert_eq!(
            auto.bind_group_layouts()[0].entries()[0].kind,
            Some(BindingLayoutKind::ExternalTexture)
        );

        let mismatch_layout =
            Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: SHADER_STAGE_FRAGMENT,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::Texture {
                        sample_type: TextureSampleType::Float,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    }),
                }],
                error: None,
            }));
        let mismatch_pipeline_layout =
            Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
                bind_group_layouts: vec![mismatch_layout],
                immediate_size: 0,
                error: None,
            }));
        let mut mismatch = render_pipeline_descriptor(Arc::clone(&module));
        mismatch.layout = RenderPipelineLayout::Explicit(mismatch_pipeline_layout);
        device.push_error_scope(ErrorFilter::Validation);
        let mismatch_pipeline = device.create_render_pipeline(mismatch);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("external texture mismatch should be scoped");
        assert!(mismatch_pipeline.is_error());
        assert_eq!(
            scoped.message,
            "compute pipeline layout binding type is incompatible"
        );

        let external_layout =
            Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: SHADER_STAGE_FRAGMENT,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::ExternalTexture),
                }],
                error: None,
            }));
        let external_pipeline_layout =
            Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
                bind_group_layouts: vec![external_layout],
                immediate_size: 0,
                error: None,
            }));
        let mut matched = render_pipeline_descriptor(module);
        matched.layout = RenderPipelineLayout::Explicit(external_pipeline_layout);
        device.push_error_scope(ErrorFilter::Validation);
        let matched_pipeline = device.create_render_pipeline(matched);
        let scoped = device.pop_error_scope().expect("scope should exist");
        assert!(!matched_pipeline.is_error());
        assert_eq!(scoped, None);
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn input_attachment_render_pipeline_auto_layout_reflects_binding_kind() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "enable chromium_internal_input_attachments;

                @group(0) @binding(0) @input_attachment_index(0) var ia: input_attachment<f32>;

                @vertex
                fn vs() -> @builtin(position) vec4<f32> {
                    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }

                @fragment
                fn fs() -> @location(0) vec4<f32> {
                    return inputAttachmentLoad(ia);
                }"
                .to_owned(),
            )),
        );

        let pipeline = device.create_render_pipeline(render_pipeline_descriptor(module));

        assert!(!pipeline.is_error());
        assert_eq!(
            pipeline.bind_group_layouts()[0].entries()[0].kind,
            Some(BindingLayoutKind::InputAttachment {
                sample_type: TextureSampleType::Float,
                multisampled: false,
            })
        );
    }

    #[cfg(feature = "tiled")]
    fn input_attachment_layout(
        device: &crate::Device,
        binding: u32,
        multisampled: bool,
    ) -> Arc<BindGroupLayout> {
        Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding,
                visibility: SHADER_STAGE_FRAGMENT,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::InputAttachment {
                    sample_type: TextureSampleType::Float,
                    multisampled,
                }),
            }],
            error: None,
        }))
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn pipeline_has_multisampled_input_attachment_detects_only_multisampled_input() {
        let device = noop_device();
        let multisampled = input_attachment_layout(&device, 0, true);
        let single_sampled = input_attachment_layout(&device, 0, false);
        let buffer = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: SHADER_STAGE_FRAGMENT,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: 0,
                }),
            }],
            error: None,
        }));

        assert!(pipeline_has_multisampled_input_attachment(&[multisampled]));
        assert!(!pipeline_has_multisampled_input_attachment(&[
            single_sampled
        ]));
        assert!(!pipeline_has_multisampled_input_attachment(&[buffer]));
        assert!(!pipeline_has_multisampled_input_attachment(&[]));
    }

    #[cfg(feature = "tiled")]
    fn msaa_subpass_pass_layout(
        source_sample_count: u32,
        written_sample_count: u32,
    ) -> SubpassPassLayoutDescriptor {
        SubpassPassLayoutDescriptor {
            color_attachments: vec![
                AttachmentLayout {
                    format: rgba8_unorm(),
                    sample_count: source_sample_count,
                },
                AttachmentLayout {
                    format: rgba8_unorm(),
                    sample_count: written_sample_count,
                },
            ],
            depth_stencil_attachment: None,
            subpasses: vec![
                SubpassLayoutDesc {
                    color_attachment_indices: vec![0],
                    uses_depth_stencil: false,
                    input_attachments: Vec::new(),
                },
                SubpassLayoutDesc {
                    color_attachment_indices: vec![1],
                    uses_depth_stencil: false,
                    input_attachments: vec![SubpassInputAttachment {
                        group: 0,
                        binding: 0,
                        source_subpass: 0,
                        source_attachment: 0,
                    }],
                },
            ],
            dependencies: Vec::new(),
            error: None,
        }
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn validate_subpass_pipeline_multisampling_accepts_consistent_msaa() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
             @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
                    .to_owned(),
            )),
        );
        let mut base = render_pipeline_descriptor(module);
        base.multisample.count = 4;
        let layout = input_attachment_layout(&device, 0, true);

        assert_eq!(
            validate_subpass_pipeline_multisampling(
                &base,
                &msaa_subpass_pass_layout(4, 4),
                1,
                &[layout],
            ),
            None
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn validate_subpass_pipeline_multisampling_rejects_flag_mismatch_c1() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
             @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
                    .to_owned(),
            )),
        );
        let mut base = render_pipeline_descriptor(module);
        base.multisample.count = 4;
        // Source attachment is 4x but the layout declares single-sampled input.
        let layout = input_attachment_layout(&device, 0, false);

        assert_eq!(
            validate_subpass_pipeline_multisampling(
                &base,
                &msaa_subpass_pass_layout(4, 4),
                1,
                &[layout],
            ),
            Some(
                "subpass input attachment multisampled flag must match its source attachment sample count"
                    .to_owned()
            )
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn validate_subpass_pipeline_multisampling_rejects_count_mismatch_c2() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
             @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
                    .to_owned(),
            )),
        );
        let mut base = render_pipeline_descriptor(module);
        // Pipeline rasterizes single-sampled but the written attachment is 4x.
        base.multisample.count = 1;
        let layout = input_attachment_layout(&device, 0, true);

        assert_eq!(
            validate_subpass_pipeline_multisampling(
                &base,
                &msaa_subpass_pass_layout(4, 4),
                1,
                &[layout],
            ),
            Some(
                "subpass render pipeline multisample count must match the subpass attachment sample count"
                    .to_owned()
            )
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn input_attachment_hal_bindings_use_subpass_source_attachment_as_color_slot() {
        let device = noop_device();
        let group0 = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: SHADER_STAGE_FRAGMENT,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::InputAttachment {
                        sample_type: TextureSampleType::Float,
                        multisampled: false,
                    }),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: SHADER_STAGE_FRAGMENT,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: 0,
                    }),
                },
            ],
            error: None,
        }));
        let group1 = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 3,
                visibility: SHADER_STAGE_FRAGMENT,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::InputAttachment {
                    sample_type: TextureSampleType::Float,
                    multisampled: false,
                }),
            }],
            error: None,
        }));
        let subpass_color_slots = [((0, 0), 2), ((1, 3), 5)];

        let bindings =
            input_attachment_hal_bindings(&[group0, group1], &subpass_color_slots).unwrap();

        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings[0].group, 0);
        assert_eq!(bindings[0].binding, 0);
        assert!(matches!(
            bindings[0].kind,
            HalDescriptorBindingKind::InputAttachment { color_slot: 2 }
        ));
        assert_eq!(bindings[1].group, 1);
        assert_eq!(bindings[1].binding, 3);
        assert!(matches!(
            bindings[1].kind,
            HalDescriptorBindingKind::InputAttachment { color_slot: 5 }
        ));
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn input_attachment_hal_bindings_require_subpass_mapping() {
        let device = noop_device();
        let group0 = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: SHADER_STAGE_FRAGMENT,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::InputAttachment {
                    sample_type: TextureSampleType::Float,
                    multisampled: false,
                }),
            }],
            error: None,
        }));

        assert_eq!(
            input_attachment_hal_bindings(&[group0], &[])
                .expect_err("missing mapping must be rejected"),
            "subpass input attachment binding is missing from subpass layout"
        );
    }

    /// The Vulkan render path rejects external textures deterministically — the
    /// descriptor is valid WebGPU (no validation error) but the backend has no
    /// SPIR-V lowering, so `select_render_shader_source` returns a backend error
    /// before any code reaches a driver. This guards against the driver-dependent
    /// divergence the `e2e_vulkan_external_texture` regression observed (NVIDIA
    /// compiled the multiplanar-transformed SPIR-V, Mesa rejected it).
    #[test]
    fn select_render_shader_source_rejects_external_texture_on_vulkan() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@group(0) @binding(0) var tex: texture_external;
             @vertex
             fn vs() -> @builtin(position) vec4<f32> {
                 return vec4<f32>(0.0, 0.0, 0.0, 1.0);
             }
             @fragment
             fn fs() -> @location(0) vec4<f32> {
                 return textureLoad(tex, vec2<i32>(0, 0));
             }"
                .to_owned(),
            )),
        );
        let descriptor = render_pipeline_descriptor(module);
        let external = [MetalBufferBinding {
            group: 0,
            binding: 0,
            metal_index: 0,
            ext_params_buffer_slot: Some(1),
            ext_params_vertex_buffer_slot: None,
            ext_params_fragment_buffer_slot: Some(1),
            vertex_metal_index: None,
            fragment_metal_index: Some(0),
            kind: MetalBindingKind::ExternalTexture,
        }];
        let err = select_render_shader_source(
            HalBackend::Vulkan,
            &descriptor,
            "vs",
            Some("fs"),
            &external,
            &[],
            &[],
            false,
            &[],
        )
        .expect_err("Vulkan must reject external textures");
        assert_eq!(
            err,
            "external textures are not supported on the Vulkan backend"
        );
    }

    /// Regression B / F-081: a fragment-only `texture_external` render pipeline
    /// must have `ext_params_buffer_slot = Some(...)` in its `MetalBufferBinding`.
    ///
    /// Before the fix, the ExternalTexture arm in `metal_buffer_binding_map`
    /// returned `(vi_tex, fi_tex, vi_buf)` where `vi_buf` is `None` for
    /// fragment-only visibility.  That propagated as `ext_params_buffer_slot =
    /// None`, causing `msl_resources` to abort with "MSL external texture
    /// binding is missing params buffer slot" on both Metal and (superficially)
    /// Vulkan backends.
    ///
    /// This test reproduces the defect on Noop by inspecting the binding map
    /// directly, so it fails *before* the fix and passes after.
    #[test]
    fn fragment_only_external_texture_binding_has_ext_params_buffer_slot() {
        let device = noop_device();
        // Fragment-only external texture shader: vs does not reference `tex`,
        // so auto layout assigns SHADER_STAGE_FRAGMENT visibility to it.
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@group(0) @binding(0) var tex: texture_external;
             @vertex
             fn vs() -> @builtin(position) vec4<f32> {
                 return vec4<f32>(0.0, 0.0, 0.0, 1.0);
             }
             @fragment
             fn fs() -> @location(0) vec4<f32> {
                 return textureLoad(tex, vec2<i32>(0, 0));
             }"
                .to_owned(),
            )),
        );
        let pipeline = device.create_render_pipeline(render_pipeline_descriptor(module));
        assert!(!pipeline.is_error(), "pipeline must not be in error state");

        // The binding map must contain exactly one entry for the external texture.
        let bindings = pipeline.metal_bindings();
        let ext = bindings
            .iter()
            .find(|b| matches!(b.kind, MetalBindingKind::ExternalTexture))
            .expect("external texture binding must be present in the binding map");

        // Before the fix: ext_params_buffer_slot == None (vi_buf was used instead
        // of vi_buf.or(fi_buf)).  After the fix it must be Some.
        assert!(
            ext.ext_params_buffer_slot.is_some(),
            "fragment-only ExternalTexture must have a non-None ext_params_buffer_slot \
             (regression B / F-081); got None"
        );
        // Fragment-metal-index must be Some (the binding is visible to the
        // fragment stage) and vertex-metal-index must be None.
        assert!(
            ext.fragment_metal_index.is_some(),
            "fragment-only ExternalTexture must have fragment_metal_index = Some(_)"
        );
        assert_eq!(
            ext.vertex_metal_index, None,
            "fragment-only ExternalTexture must have vertex_metal_index = None"
        );
    }
    fn depth32_float() -> TextureFormat {
        TextureFormat::from_raw(0x30)
    }
    #[test]
    fn bundle_attachment_signature_compat_uses_readonly_implication() {
        // F-062: formats/sample-count match exactly, but read-only is an
        // implication — a read-write bundle cannot run in a read-only pass,
        // while a read-only bundle may run in a read-write pass.
        let sig = |depth_ro: bool, stencil_ro: bool| AttachmentSignature {
            color_formats: vec![Some(TextureFormat::from_raw(TextureFormat::RGBA8_UNORM))],
            depth_stencil_format: Some(TextureFormat::from_raw(
                TextureFormat::DEPTH24_PLUS_STENCIL8,
            )),
            sample_count: 1,
            depth_read_only: depth_ro,
            stencil_read_only: stencil_ro,
        };
        // Identical → compatible.
        assert!(sig(true, true).bundle_compatible_with_pass(&sig(true, true)));
        // Read-only bundle in a read-write pass → compatible.
        assert!(sig(true, true).bundle_compatible_with_pass(&sig(false, false)));
        // Read-write bundle in a read-only pass → incompatible.
        assert!(!sig(false, false).bundle_compatible_with_pass(&sig(true, true)));
        assert!(!sig(true, false).bundle_compatible_with_pass(&sig(true, true)));
        // Differing color formats → incompatible regardless of read-only.
        let other_color = AttachmentSignature {
            color_formats: vec![Some(TextureFormat::from_raw(TextureFormat::RG8_UNORM))],
            ..sig(false, false)
        };
        assert!(!other_color.bundle_compatible_with_pass(&sig(false, false)));
    }

    #[test]
    fn inter_stage_interpolation_defaults_are_applied() {
        use frontend::{ReflectedInterpolation as I, ReflectedSampling as S};
        // F-063: unspecified interpolation defaults to perspective; unspecified
        // sampling defaults to center for perspective/linear, so `perspective`
        // (None sampling) matches `perspective, center`.
        assert_eq!(effective_interpolation(None), I::Perspective);
        assert_eq!(effective_sampling(None, None), Some(S::Center));
        assert_eq!(
            effective_sampling(Some(I::Perspective), None),
            Some(S::Center)
        );
        assert_eq!(
            effective_sampling(Some(I::Perspective), Some(S::Center)),
            Some(S::Center)
        );
        assert_eq!(effective_sampling(Some(I::Linear), None), Some(S::Center));
        assert_ne!(
            effective_sampling(Some(I::Perspective), Some(S::Sample)),
            effective_sampling(Some(I::Perspective), None)
        );
        // Flat carries sampling as-is (no center default).
        assert_eq!(effective_sampling(Some(I::Flat), None), None);
    }

    // ---- F-080: unfilterable-float texture + filtering sampler must error (render) ----
    //
    // CTS: api,validation,non_filterable_texture:non_filterable_texture_with_filtering_sampler
    // (pipeline=render cases)
    //
    // The shader uses texture_2d<f32> + textureGather. The frontend reflects the texture as
    // Float.  The explicit BGL declares the texture entry as UnfilterableFloat.  The
    // F-061 rule allows UnfilterableFloat layout to accept a Float shader binding, but
    // an UnfilterableFloat texture combined with a Filtering sampler in the same pipeline
    // layout must still produce a validation error at createRenderPipeline time.

    /// WGSL for the render non_filterable test: texture_2d<f32> at @group(0) @binding(0),
    /// sampler at @group(group_ndx) @binding(1), used together in textureGather in the
    /// fragment entry point.
    fn non_filterable_render_wgsl(group_ndx: u32) -> String {
        format!(
            r"
@group(0) @binding(0) var t: texture_2d<f32>;
@group({group_ndx}) @binding(1) var s: sampler;

fn test() -> vec4<f32> {{
  return textureGather(0, t, s, vec2f(0.0));
}}

@vertex fn vs() -> @builtin(position) vec4f {{ return vec4f(0.0); }}
@fragment fn fs() -> @location(0) vec4f {{ return test(); }}
",
        )
    }

    /// Regression test (F-080 render): explicit BGL with `UnfilterableFloat` texture and
    /// `Filtering` sampler in the same group must error on createRenderPipeline.
    #[test]
    fn unfilterable_float_texture_with_filtering_sampler_rejects_render_pipeline() {
        let device = noop_device();

        let vis = SHADER_STAGE_FRAGMENT | SHADER_STAGE_VERTEX;
        let bgl = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vis,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::Texture {
                        sample_type: TextureSampleType::UnfilterableFloat,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    }),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vis,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::Sampler {
                        ty: SamplerBindingType::Filtering,
                    }),
                },
            ],
            error: None,
        }));
        let pipeline_layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::clone(&bgl)],
            immediate_size: 0,
            error: None,
        }));
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(non_filterable_render_wgsl(0))),
        );
        assert!(!module.is_error(), "shader module must compile");

        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.layout = RenderPipelineLayout::Explicit(pipeline_layout);

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_render_pipeline(descriptor);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("UnfilterableFloat+Filtering must produce a validation error (F-080 render)");
        assert!(pipeline.is_error());
        assert_eq!(
            scoped.message,
            "textureGather with a filtering sampler requires a filterable texture binding"
        );
    }

    /// Positive case (F-080 render): explicit BGL with `Float` texture and `Filtering`
    /// sampler must succeed.
    #[test]
    fn filterable_float_texture_with_filtering_sampler_accepts_render_pipeline() {
        let device = noop_device();

        let vis = SHADER_STAGE_FRAGMENT | SHADER_STAGE_VERTEX;
        let bgl = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vis,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::Texture {
                        sample_type: TextureSampleType::Float,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    }),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vis,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::Sampler {
                        ty: SamplerBindingType::Filtering,
                    }),
                },
            ],
            error: None,
        }));
        let pipeline_layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::clone(&bgl)],
            immediate_size: 0,
            error: None,
        }));
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(non_filterable_render_wgsl(0))),
        );
        assert!(!module.is_error(), "shader module must compile");

        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.layout = RenderPipelineLayout::Explicit(pipeline_layout);

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_render_pipeline(descriptor);
        let scoped = device.pop_error_scope().expect("scope should exist");
        assert!(
            !pipeline.is_error(),
            "Float+Filtering must succeed for render pipeline"
        );
        assert_eq!(scoped, None, "no validation error expected");
    }
}
