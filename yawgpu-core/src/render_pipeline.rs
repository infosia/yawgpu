use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use yawgpu_hal::{
    HalBackend, HalBlendComponent, HalBlendFactor, HalBlendOperation, HalBlendState,
    HalColorTargetState, HalCompareFunction, HalCullMode, HalDepthStencilState,
    HalDescriptorBinding, HalDevice, HalFrontFace, HalMslBufferSizeBinding, HalPrimitiveTopology,
    HalRenderPipeline, HalRenderPipelineDescriptor, HalShaderSource, HalStencilFaceState,
    HalStencilOperation, HalVertexAttribute, HalVertexBufferLayout, HalVertexFormat,
    HalVertexStepMode,
};
#[cfg(feature = "tiled")]
use yawgpu_hal::{
    HalBufferBindingKind, HalDescriptorBindingKind, HalSubpassAttachmentLayout,
    HalSubpassDependency, HalSubpassDependencyType, HalSubpassInputAttachment, HalSubpassLayout,
    HalSubpassPassLayout,
};

use crate::bind_group_layout::*;
use crate::compute_pipeline::*;
use crate::device::FeatureSet;
use crate::format::*;
use crate::limits::*;
use crate::pipeline_layout::*;
use crate::sampler::*;
use crate::shader::*;
use crate::shader_naga;
#[cfg(feature = "tiled")]
use crate::subpass::SubpassPassLayout;
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
    #[cfg(feature = "tiled")]
    pub(crate) subpass_compatibility: Option<SubpassPipelineCompatibility>,
    pub(crate) is_error: bool,
}

/// Describes subpass-pipeline compatibility metadata.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub(crate) struct SubpassPipelineCompatibility {
    pub(crate) pass_layout: Arc<SubpassPassLayout>,
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
        let resolved = if is_error {
            None
        } else {
            resolve_render_pipeline_descriptor(&descriptor, limits, features, None).ok()
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
            metal_vertex_buffer_binding_map(descriptor.vertex.buffer_count, &metal_bindings);
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
        let compatibility = SubpassPipelineCompatibility {
            pass_layout: Arc::clone(&descriptor.pass_layout),
            subpass_index: descriptor.subpass_index,
        };
        let resolved = if is_error {
            None
        } else {
            resolve_render_pipeline_descriptor(
                &descriptor.base,
                limits,
                features,
                Some(
                    &descriptor
                        .pass_layout
                        .descriptor()
                        .subpasses
                        .get(descriptor.subpass_index as usize)
                        .map(|s| s.color_attachment_indices.clone())
                        .unwrap_or_default(),
                ),
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
            metal_vertex_buffer_binding_map(descriptor.base.vertex.buffer_count, &metal_bindings);
        let (hal, backend_error) = if is_error {
            (None, None)
        } else {
            create_hal_subpass_render_pipeline(
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
    pub(crate) fn writes_stencil(&self) -> bool {
        self.inner
            ._depth_stencil
            .is_some_and(|depth| depth.stencil_write_mask != 0)
    }

    /// Returns subpass compatibility metadata.
    #[cfg(feature = "tiled")]
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
    resolve_render_pipeline_descriptor(descriptor, limits, features, None).err()
}

/// Alias for resolved render pipeline parts.
pub(crate) type ResolvedRenderPipelineParts = (String, Option<String>, Vec<Arc<BindGroupLayout>>);

#[cfg(feature = "shader-passthrough")]
fn render_pipeline_uses_msl(descriptor: &RenderPipelineDescriptor) -> bool {
    descriptor.vertex.shader.module.msl_passthrough().is_some()
        || descriptor
            .fragment
            .as_ref()
            .is_some_and(|fragment| fragment.shader.module.msl_passthrough().is_some())
}

/// Creates HAL render pipeline and reports validation errors through the owning device.
pub(crate) fn create_hal_render_pipeline(
    hal_device: Option<&HalDevice>,
    descriptor: &RenderPipelineDescriptor,
    vertex_entry_name: &str,
    fragment_entry_name: Option<&str>,
    metal_bindings: &[MetalBufferBinding],
    vertex_buffer_bindings: &[MetalVertexBufferBinding],
) -> (Option<HalRenderPipeline>, Option<String>) {
    let Some(hal_device) = hal_device else {
        return (None, None);
    };
    if matches!(hal_device.backend(), HalBackend::Noop) {
        return (Some(HalRenderPipeline::Noop), None);
    }
    if descriptor.fragment.is_none() && descriptor.depth_stencil.is_none() {
        return (
            None,
            Some(
                "real render pipeline requires a fragment stage or depth-stencil state".to_owned(),
            ),
        );
    }
    let (shader, vertex_entry_point, fragment_entry_point, descriptor_bindings) =
        match select_render_shader_source(
            hal_device.backend(),
            descriptor,
            vertex_entry_name,
            fragment_entry_name,
            metal_bindings,
            vertex_buffer_bindings,
            &[],
        ) {
            Ok(selection) => selection,
            Err(message) => return (None, Some(message)),
        };
    let hal_descriptor = match hal_render_pipeline_descriptor(descriptor, vertex_buffer_bindings) {
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

/// Creates HAL subpass render pipeline and reports validation errors through the owning device.
#[cfg(feature = "tiled")]
pub(crate) fn create_hal_subpass_render_pipeline(
    hal_device: Option<&HalDevice>,
    descriptor: &SubpassRenderPipelineDescriptor,
    vertex_entry_name: &str,
    fragment_entry_name: Option<&str>,
    metal_bindings: &[MetalBufferBinding],
    vertex_buffer_bindings: &[MetalVertexBufferBinding],
    bind_group_layouts: &[Arc<BindGroupLayout>],
) -> (Option<HalRenderPipeline>, Option<String>) {
    let Some(hal_device) = hal_device else {
        return (None, None);
    };
    if descriptor.base.multisample.count != 1 {
        return (
            None,
            Some("subpass render pipeline does not yet support multisample > 1".to_owned()),
        );
    }
    if matches!(hal_device.backend(), HalBackend::Noop) {
        return (Some(HalRenderPipeline::Noop), None);
    }
    if descriptor.base.fragment.is_none() {
        return (
            None,
            Some("subpass render pipeline requires a fragment stage".to_owned()),
        );
    }
    let Some(fragment_entry_name) = fragment_entry_name else {
        return (
            None,
            Some("subpass render pipeline requires a fragment entry point".to_owned()),
        );
    };
    // Build the Metal pass-local color-slot map from the pass layout's input
    // attachments for this subpass: each `subpass_input` shader binding
    // `(group, binding)` maps to its `source_attachment` color slot index in
    // the layout's color attachments. naga's MSL backend lowers
    // `subpassLoad(global)` to `[[color(N)]]` using this map; without it,
    // subpass inputs would silently read zero.
    let subpass_color_slots: Vec<((u32, u32), u32)> = descriptor
        .pass_layout
        .descriptor()
        .subpasses
        .get(descriptor.subpass_index as usize)
        .map(|subpass| {
            subpass
                .input_attachments
                .iter()
                .map(|input| ((input.group, input.binding), input.source_attachment))
                .collect()
        })
        .unwrap_or_default();
    let (shader, vertex_entry_point, fragment_entry_point, mut descriptor_bindings) =
        match select_render_shader_source(
            hal_device.backend(),
            &descriptor.base,
            vertex_entry_name,
            Some(fragment_entry_name),
            metal_bindings,
            vertex_buffer_bindings,
            &subpass_color_slots,
        ) {
            Ok(selection) => selection,
            Err(message) => return (None, Some(message)),
        };
    let Some(fragment_entry_point) = fragment_entry_point else {
        return (
            None,
            Some("subpass render pipeline requires a fragment entry point".to_owned()),
        );
    };
    // Vulkan reads subpass inputs through `INPUT_ATTACHMENT` descriptors wired
    // from the pass layout's input-source mapping; the Metal backend reads them
    // via the color-slot map supplied above, so it takes no extra descriptors here.
    if matches!(hal_device.backend(), HalBackend::Vulkan) {
        descriptor_bindings.extend(input_attachment_hal_bindings(bind_group_layouts));
    }
    let hal_descriptor =
        match hal_render_pipeline_descriptor(&descriptor.base, vertex_buffer_bindings) {
            Ok(descriptor) => descriptor,
            Err(message) => return (None, Some(message)),
        };
    let hal_pass_layout = hal_subpass_pass_layout(descriptor.pass_layout.descriptor());
    match hal_device.create_subpass_render_pipeline(
        shader,
        &vertex_entry_point,
        &fragment_entry_point,
        &hal_descriptor,
        &descriptor_bindings,
        &hal_pass_layout,
        descriptor.subpass_index,
    ) {
        Ok(pipeline) => (Some(pipeline), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

/// Builds HAL input-attachment descriptor bindings from the resolved bind group
/// layouts (Vulkan binds subpass inputs through `INPUT_ATTACHMENT` descriptors).
#[cfg(feature = "tiled")]
fn input_attachment_hal_bindings(
    bind_group_layouts: &[Arc<BindGroupLayout>],
) -> Vec<HalDescriptorBinding> {
    let mut bindings = Vec::new();
    for (group_index, layout) in bind_group_layouts.iter().enumerate() {
        let Ok(group) = u32::try_from(group_index) else {
            break;
        };
        for entry in layout.entries() {
            if matches!(entry.kind, Some(BindingLayoutKind::InputAttachment { .. })) {
                bindings.push(HalDescriptorBinding {
                    group,
                    binding: entry.binding,
                    kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::InputAttachment),
                });
            }
        }
    }
    bindings
}

#[cfg(feature = "tiled")]
fn hal_subpass_pass_layout(
    layout: &crate::subpass::SubpassPassLayoutDescriptor,
) -> HalSubpassPassLayout {
    HalSubpassPassLayout {
        color_attachments: layout
            .color_attachments
            .iter()
            .map(|attachment| HalSubpassAttachmentLayout {
                format: hal_texture_format(attachment.format),
                sample_count: attachment.sample_count,
            })
            .collect(),
        depth_stencil_attachment: layout.depth_stencil_attachment.map(|attachment| {
            HalSubpassAttachmentLayout {
                format: hal_texture_format(attachment.format),
                sample_count: attachment.sample_count,
            }
        }),
        subpasses: layout
            .subpasses
            .iter()
            .map(|subpass| HalSubpassLayout {
                color_attachment_indices: subpass.color_attachment_indices.clone(),
                uses_depth_stencil: subpass.uses_depth_stencil,
                input_attachments: subpass
                    .input_attachments
                    .iter()
                    .map(|input| HalSubpassInputAttachment {
                        group: input.group,
                        binding: input.binding,
                        source_subpass: input.source_subpass,
                        source_attachment: input.source_attachment,
                    })
                    .collect(),
            })
            .collect(),
        dependencies: layout
            .dependencies
            .iter()
            .map(|dependency| HalSubpassDependency {
                src_subpass: dependency.src_subpass,
                dst_subpass: dependency.dst_subpass,
                dependency_type: match dependency.dependency_type {
                    crate::subpass::SubpassDependencyType::ColorToInput => {
                        HalSubpassDependencyType::ColorToInput
                    }
                    crate::subpass::SubpassDependencyType::DepthToInput => {
                        HalSubpassDependencyType::DepthToInput
                    }
                    crate::subpass::SubpassDependencyType::ColorDepthToInput => {
                        HalSubpassDependencyType::ColorDepthToInput
                    }
                },
                by_region: dependency.by_region,
            })
            .collect(),
    }
}

/// Selects the HAL shader source for a render pipeline.
pub(crate) fn select_render_shader_source(
    backend: HalBackend,
    descriptor: &RenderPipelineDescriptor,
    vertex_entry_name: &str,
    fragment_entry_name: Option<&str>,
    metal_bindings: &[MetalBufferBinding],
    vertex_buffer_bindings: &[MetalVertexBufferBinding],
    subpass_color_slots: &[((u32, u32), u32)],
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
            if let Some((source, _)) = descriptor.vertex.shader.module.msl_passthrough() {
                let Some(fragment) = fragment else {
                    return Err(
                        "Metal passthrough render pipeline requires a fragment stage".to_owned(),
                    );
                };
                if !Arc::ptr_eq(&descriptor.vertex.shader.module, &fragment.shader.module) {
                    return Err(
                        "Metal render pipeline requires vertex and fragment entries in the same MSL module"
                            .to_owned(),
                    );
                }
                return Ok((
                    HalShaderSource::Msl(source.to_owned()),
                    vertex_entry_name.to_owned(),
                    fragment_entry_name.map(str::to_owned),
                    Vec::new(),
                ));
            }
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
                return Err("SPIR-V shader module cannot be used on the Metal backend".to_owned());
            }
            let module =
                descriptor.vertex.shader.module.reflected().ok_or_else(|| {
                    "render pipeline requires a reflected shader module".to_owned()
                })?;
            let msl_binding_map = shader_naga::MslBindingMap {
                resources: msl_resource_bindings(metal_bindings),
            };
            let msl_vertex_buffers =
                msl_vertex_buffer_bindings(&descriptor.vertex.buffers, vertex_buffer_bindings)?;
            let force_point_size =
                matches!(descriptor.primitive.topology, PrimitiveTopology::PointList);
            let vertex = module.generate_render_vertex_msl(
                vertex_entry_name,
                &msl_binding_map,
                &msl_vertex_buffers,
                force_point_size,
                &vertex_pipeline_constants,
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
                        &msl_binding_map,
                        subpass_color_slots,
                        fragment_pipeline_constants,
                    )?)
                }
                (None, None, None) => None,
                _ => {
                    return Err(
                        "real render pipeline fragment state and entry point must match".to_owned(),
                    );
                }
            };
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
                },
                vertex.entry_point,
                fragment.map(|fragment| fragment.entry_point),
                Vec::new(),
            ))
        }
        HalBackend::Vulkan => {
            #[cfg(feature = "shader-passthrough")]
            if descriptor.vertex.shader.module.msl_passthrough().is_some()
                || fragment
                    .is_some_and(|fragment| fragment.shader.module.msl_passthrough().is_some())
            {
                return Err("MSL shader module cannot be used on the Vulkan backend".to_owned());
            }
            #[cfg(feature = "shader-passthrough")]
            if let Some((vertex_words, _)) = descriptor.vertex.shader.module.spirv_passthrough() {
                let Some(fragment) = fragment else {
                    return Err(
                        "SPIR-V passthrough render pipeline requires a fragment stage".to_owned(),
                    );
                };
                let Some((fragment_words, _)) = fragment.shader.module.spirv_passthrough() else {
                    return Err(
                        "render pipeline cannot mix a SPIR-V passthrough module with a non-SPIR-V module"
                            .to_owned(),
                    );
                };
                return Ok((
                    HalShaderSource::SpirVStages {
                        vertex: vertex_words.to_vec(),
                        fragment: Some(fragment_words.to_vec()),
                    },
                    vertex_entry_name.to_owned(),
                    fragment_entry_name.map(str::to_owned),
                    hal_descriptor_bindings(metal_bindings),
                ));
            }
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
                return Err(
                    "render pipeline cannot mix a SPIR-V passthrough module with a non-SPIR-V module"
                        .to_owned(),
                );
            }
            let vertex_module = descriptor.vertex.shader.module.reflected().ok_or_else(|| {
                "render pipeline requires a reflected vertex shader module".to_owned()
            })?;
            let vertex = vertex_module.generate_spirv(
                vertex_entry_name,
                naga::ShaderStage::Vertex,
                &vertex_pipeline_constants,
            )?;
            let fragment = match (fragment, fragment_entry_name) {
                (Some(fragment), Some(fragment_entry_name)) => {
                    let fragment_module = fragment.shader.module.reflected().ok_or_else(|| {
                        "render pipeline requires a reflected fragment shader module".to_owned()
                    })?;
                    Some(fragment_module.generate_spirv(
                        fragment_entry_name,
                        naga::ShaderStage::Fragment,
                        fragment_pipeline_constants.as_ref().ok_or_else(|| {
                            "render pipeline fragment constants were not resolved".to_owned()
                        })?,
                    )?)
                }
                (None, None) => None,
                _ => {
                    return Err(
                        "real render pipeline fragment state and entry point must match".to_owned(),
                    );
                }
            };
            Ok((
                HalShaderSource::SpirVStages { vertex, fragment },
                vertex_entry_name.to_owned(),
                fragment_entry_name.map(str::to_owned),
                hal_descriptor_bindings(metal_bindings),
            ))
        }
        #[cfg(feature = "gles")]
        HalBackend::Gles => {
            #[cfg(feature = "shader-passthrough")]
            if descriptor.vertex.shader.module.msl_passthrough().is_some()
                || fragment.shader.module.msl_passthrough().is_some()
                || descriptor
                    .vertex
                    .shader
                    .module
                    .spirv_passthrough()
                    .is_some()
                || fragment
                    .is_some_and(|fragment| fragment.shader.module.spirv_passthrough().is_some())
            {
                return Err(
                    "passthrough shader modules cannot be used on the GLES backend".to_owned(),
                );
            }
            let vertex_module = descriptor.vertex.shader.module.reflected().ok_or_else(|| {
                "render pipeline requires a reflected vertex shader module".to_owned()
            })?;
            let vertex_glsl = vertex_module.generate_glsl(
                vertex_entry_name,
                naga::ShaderStage::Vertex,
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
                                naga::ShaderStage::Fragment,
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

fn hal_msl_buffer_size_bindings(
    bindings: &[shader_naga::MslBufferSizeBinding],
) -> Vec<HalMslBufferSizeBinding> {
    bindings
        .iter()
        .map(|binding| HalMslBufferSizeBinding::new(binding.group, binding.binding))
        .collect()
}

/// Returns metal vertex buffer binding map.
pub(crate) fn metal_vertex_buffer_binding_map(
    vertex_buffer_count: usize,
    metal_bindings: &[MetalBufferBinding],
) -> Vec<MetalVertexBufferBinding> {
    let start = metal_bindings.len();
    (0..vertex_buffer_count)
        .filter_map(|slot| {
            Some(MetalVertexBufferBinding {
                slot: u32::try_from(slot).ok()?,
                metal_index: u32::try_from(start.checked_add(slot)?).ok()?,
            })
        })
        .collect()
}

/// Returns msl vertex buffer bindings.
pub(crate) fn msl_vertex_buffer_bindings(
    layouts: &[VertexBufferLayout],
    bindings: &[MetalVertexBufferBinding],
) -> Result<Vec<shader_naga::MslVertexBufferBinding>, String> {
    layouts
        .iter()
        .zip(bindings)
        .map(|(layout, binding)| {
            Ok(shader_naga::MslVertexBufferBinding {
                slot: binding.slot,
                metal_index: binding.metal_index,
                array_stride: layout.array_stride,
                step_mode: match layout.step_mode {
                    VertexStepMode::Vertex => shader_naga::MslVertexStepMode::Vertex,
                    VertexStepMode::Instance => shader_naga::MslVertexStepMode::Instance,
                },
                attributes: layout
                    .attributes
                    .iter()
                    .map(|attribute| {
                        Ok(shader_naga::MslVertexAttribute {
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
                .map(hal_color_target_state)
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?
        .unwrap_or_default();
    let vertex_buffers = descriptor
        .vertex
        .buffers
        .iter()
        .zip(bindings)
        .map(|(layout, binding)| {
            Ok(HalVertexBufferLayout {
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
    })
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
pub(crate) fn msl_vertex_format(
    format: VertexFormat,
) -> Result<shader_naga::MslVertexFormat, String> {
    match format.0 {
        0x0000_0001 => Ok(shader_naga::MslVertexFormat::Uint8),
        0x0000_0002 => Ok(shader_naga::MslVertexFormat::Uint8x2),
        0x0000_0003 => Ok(shader_naga::MslVertexFormat::Uint8x4),
        0x0000_0004 => Ok(shader_naga::MslVertexFormat::Sint8),
        0x0000_0005 => Ok(shader_naga::MslVertexFormat::Sint8x2),
        0x0000_0006 => Ok(shader_naga::MslVertexFormat::Sint8x4),
        0x0000_0007 => Ok(shader_naga::MslVertexFormat::Unorm8),
        0x0000_0008 => Ok(shader_naga::MslVertexFormat::Unorm8x2),
        0x0000_0009 => Ok(shader_naga::MslVertexFormat::Unorm8x4),
        0x0000_000A => Ok(shader_naga::MslVertexFormat::Snorm8),
        0x0000_000B => Ok(shader_naga::MslVertexFormat::Snorm8x2),
        0x0000_000C => Ok(shader_naga::MslVertexFormat::Snorm8x4),
        0x0000_000D => Ok(shader_naga::MslVertexFormat::Uint16),
        0x0000_000E => Ok(shader_naga::MslVertexFormat::Uint16x2),
        0x0000_000F => Ok(shader_naga::MslVertexFormat::Uint16x4),
        0x0000_0010 => Ok(shader_naga::MslVertexFormat::Sint16),
        0x0000_0011 => Ok(shader_naga::MslVertexFormat::Sint16x2),
        0x0000_0012 => Ok(shader_naga::MslVertexFormat::Sint16x4),
        0x0000_0013 => Ok(shader_naga::MslVertexFormat::Unorm16),
        0x0000_0014 => Ok(shader_naga::MslVertexFormat::Unorm16x2),
        0x0000_0015 => Ok(shader_naga::MslVertexFormat::Unorm16x4),
        0x0000_0016 => Ok(shader_naga::MslVertexFormat::Snorm16),
        0x0000_0017 => Ok(shader_naga::MslVertexFormat::Snorm16x2),
        0x0000_0018 => Ok(shader_naga::MslVertexFormat::Snorm16x4),
        0x0000_0019 => Ok(shader_naga::MslVertexFormat::Float16),
        0x0000_001A => Ok(shader_naga::MslVertexFormat::Float16x2),
        0x0000_001B => Ok(shader_naga::MslVertexFormat::Float16x4),
        0x0000_001C => Ok(shader_naga::MslVertexFormat::Float32),
        0x0000_001D => Ok(shader_naga::MslVertexFormat::Float32x2),
        0x0000_001E => Ok(shader_naga::MslVertexFormat::Float32x3),
        0x0000_001F => Ok(shader_naga::MslVertexFormat::Float32x4),
        0x0000_0020 => Ok(shader_naga::MslVertexFormat::Uint32),
        0x0000_0021 => Ok(shader_naga::MslVertexFormat::Uint32x2),
        0x0000_0022 => Ok(shader_naga::MslVertexFormat::Uint32x3),
        0x0000_0023 => Ok(shader_naga::MslVertexFormat::Uint32x4),
        0x0000_0024 => Ok(shader_naga::MslVertexFormat::Sint32),
        0x0000_0025 => Ok(shader_naga::MslVertexFormat::Sint32x2),
        0x0000_0026 => Ok(shader_naga::MslVertexFormat::Sint32x3),
        0x0000_0027 => Ok(shader_naga::MslVertexFormat::Sint32x4),
        0x0000_0028 => Ok(shader_naga::MslVertexFormat::Unorm10_10_10_2),
        0x0000_0029 => Ok(shader_naga::MslVertexFormat::Unorm8x4Bgra),
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
) -> Result<ResolvedRenderPipelineParts, String> {
    if let RenderPipelineLayout::Explicit(layout) = &descriptor.layout {
        if layout.is_error() {
            return Err("render pipeline layout must not be an error pipeline layout".to_owned());
        }
    }
    #[cfg(feature = "shader-passthrough")]
    if render_pipeline_uses_msl(descriptor)
        && matches!(descriptor.layout, RenderPipelineLayout::Auto)
    {
        return Err("MSL shader module requires an explicit pipeline layout".to_owned());
    }

    let vertex_entry = resolve_render_entry(
        &descriptor.vertex.shader,
        shader_naga::ReflectedShaderStage::Vertex,
        "vertex",
    )?;
    let fragment_entry = if let Some(fragment) = &descriptor.fragment {
        Some(resolve_render_entry(
            &fragment.shader,
            shader_naga::ReflectedShaderStage::Fragment,
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
    validate_primitive_state(descriptor.primitive)?;
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
    validate_multisample_state(descriptor, fragment_entry.as_deref())?;
    let bind_group_layouts = effective_render_bind_group_layouts(
        descriptor,
        &vertex_entry,
        fragment_entry.as_deref(),
        limits,
        features,
    )?;
    validate_bind_groups_plus_vertex_buffers(
        &bind_group_layouts,
        descriptor.vertex.buffer_count,
        limits,
    )?;

    Ok((vertex_entry, fragment_entry, bind_group_layouts))
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
            shader_naga::ReflectedTypeScalarClass::Float => FormatOutputClass::Float,
            shader_naga::ReflectedTypeScalarClass::Sint => FormatOutputClass::Sint,
            shader_naga::ReflectedTypeScalarClass::Uint => FormatOutputClass::Uint,
            shader_naga::ReflectedTypeScalarClass::Bool => {
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
) -> Result<BTreeMap<u32, shader_naga::ReflectedTypeClass>, String> {
    #[cfg(feature = "shader-passthrough")]
    if vertex.shader.module.msl_passthrough().is_some() {
        return Ok(BTreeMap::new());
    }
    let Some(module) = vertex.shader.module.reflected() else {
        return Err("vertex module reflection failed".to_owned());
    };
    Ok(module
        .entry_point_io()
        .into_iter()
        .find(|io| io.entry_point == vertex_entry)
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
    expected_stage: shader_naga::ReflectedShaderStage,
    label: &str,
) -> Result<String, String> {
    if stage.module.is_error() {
        return Err(format!(
            "render pipeline {label} shader module must not be an error module"
        ));
    }
    #[cfg(feature = "shader-passthrough")]
    if let Some((_, reflection)) = stage.module.msl_passthrough() {
        let stage_bit = match expected_stage {
            shader_naga::ReflectedShaderStage::Vertex => SHADER_STAGE_VERTEX,
            shader_naga::ReflectedShaderStage::Fragment => SHADER_STAGE_FRAGMENT,
            shader_naga::ReflectedShaderStage::Compute => SHADER_STAGE_COMPUTE,
        };
        return resolve_msl_entry(reflection, stage_bit, stage.entry_point.as_deref(), label)
            .map(|entry| entry.name.clone());
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
    // A fragment state with zero colour targets is allowed here (a frag-depth-only
    // fragment is valid); `validate_color_targets` separately rejects a fragment
    // that writes a colour output with no matching target.
    Ok(())
}

/// Validates render constants and returns a descriptive error on failure.
pub(crate) fn validate_render_constants(stage: &RenderPipelineShaderStage) -> Result<(), String> {
    #[cfg(feature = "shader-passthrough")]
    if stage.module.msl_passthrough().is_some() {
        if stage.constants.is_empty() {
            return Ok(());
        }
        return Err("MSL shader module does not support pipeline constants".to_owned());
    }
    let Some(module) = stage.module.reflected() else {
        return Err("render pipeline stage requires a reflected shader module".to_owned());
    };
    resolve_pipeline_constants(&module.overrides(), &stage.constants)?;
    Ok(())
}

/// Validates primitive state and returns a descriptive error on failure.
pub(crate) fn validate_primitive_state(primitive: PrimitiveState) -> Result<(), String> {
    if primitive.unclipped_depth {
        return Err("render pipeline unclippedDepth is not supported".to_owned());
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

    if has_depth
        && (depth_stencil.depth_compare.is_none() || depth_stencil.depth_write_enabled.is_none())
    {
        return Err(
            "render pipeline depth format requires depthCompare and depthWriteEnabled".to_owned(),
        );
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
        || depth_stencil.stencil_read_mask != u32::MAX
        || depth_stencil.stencil_write_mask != u32::MAX
}

/// Returns stencil face uses stencil.
pub(crate) fn stencil_face_uses_stencil(face: StencilFaceState) -> bool {
    face.compare != CompareFunction::Always
        || face.fail_op != StencilOperation::Keep
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
    #[cfg(feature = "shader-passthrough")]
    if fragment.shader.module.msl_passthrough().is_some() {
        return Ok(());
    }
    let Some(entry_name) = fragment_entry else {
        return Ok(());
    };
    let Some(module) = fragment.shader.module.reflected() else {
        return Err("fragment module reflection failed".to_owned());
    };
    let outputs_frag_depth = module
        .fragment_builtins()
        .into_iter()
        .any(|builtins| builtins.entry_point == entry_name && builtins.frag_depth);
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

    let skip_shader_outputs = {
        #[cfg(feature = "shader-passthrough")]
        {
            fragment.shader.module.msl_passthrough().is_some()
        }
        #[cfg(not(feature = "shader-passthrough"))]
        {
            false
        }
    };
    let outputs = if skip_shader_outputs {
        BTreeMap::new()
    } else {
        fragment_outputs(fragment, fragment_entry)?
    };
    let mut color_bytes = 0_u32;
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
        }
        if target.blend.is_some() && !caps.is_blendable {
            return Err("render pipeline color target format must be blendable".to_owned());
        }
        if descriptor.multisample.alpha_to_coverage_enabled && caps.is_blendable && caps.has_alpha {
            has_alpha_to_coverage_target = true;
        }

        if !skip_shader_outputs {
            // Block 55 accepts both subpass-local and flat-slot `@location`
            // conventions for subpass pipelines. Vulkan remaps the
            // subpass-local index through VkRenderPass, while Metal's MSL
            // path emits the flat color slot directly; HAL routing decides
            // which convention is used at submission time.
            let subpass_local = index as u32;
            let flat = subpass_color_attachment_indices
                .and_then(|indices| indices.get(index).copied())
                .unwrap_or(subpass_local);
            let output = outputs.get(&subpass_local).or_else(|| outputs.get(&flat));
            match output {
                Some(output) => {
                    validate_fragment_output_compat(*output, caps)?;
                    if target.blend.is_some_and(blend_state_uses_source_alpha)
                        && output.components < 4
                    {
                        return Err(
                            "render pipeline blend state requires a vec4 fragment output"
                                .to_owned(),
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
        }

        color_bytes = color_bytes
            .checked_add(color_attachment_byte_cost(caps.texel_block_size))
            .ok_or_else(|| "render pipeline color target byte count overflows".to_owned())?;
    }

    // Every fragment colour output (`@location(N)`) must have a colour target.
    // The loop above only checks targets→outputs, so a fragment that writes a
    // colour with too few (or zero) targets — e.g. a colour fragment paired with
    // an empty `targets` list — would otherwise slip through. A frag-depth-only
    // fragment has no colour outputs, so zero targets stays valid. (Subpass
    // pipelines remap output locations through the pass layout, so skip them.)
    if !skip_shader_outputs && subpass_color_attachment_indices.is_none() {
        for &location in outputs.keys() {
            if location as usize >= fragment.target_count {
                return Err(
                    "render pipeline fragment color output requires a color target".to_owned(),
                );
            }
        }
    }

    if descriptor.multisample.alpha_to_coverage_enabled && !has_alpha_to_coverage_target {
        return Err(
            "render pipeline alphaToCoverage requires an alpha blendable color target".to_owned(),
        );
    }
    if color_bytes > limits.max_color_attachment_bytes_per_sample {
        return Err(
            "render pipeline color target bytes per sample exceed the device limit".to_owned(),
        );
    }

    Ok(())
}

pub(crate) fn color_attachment_byte_cost(byte_size: u32) -> u32 {
    byte_size.next_power_of_two()
}

/// Validates blend state and returns a descriptive error on failure.
pub(crate) fn validate_blend_state(blend: BlendState) -> Result<(), String> {
    validate_blend_component(blend.color)?;
    validate_blend_component(blend.alpha)
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

fn blend_state_uses_source_alpha(blend: BlendState) -> bool {
    blend_component_uses_source_alpha(blend.color) || blend_component_uses_source_alpha(blend.alpha)
}

fn blend_component_uses_source_alpha(component: BlendComponent) -> bool {
    blend_factor_uses_source_alpha(component.src_factor)
        || blend_factor_uses_source_alpha(component.dst_factor)
}

fn blend_factor_uses_source_alpha(factor: BlendFactor) -> bool {
    matches!(
        factor,
        BlendFactor::SrcAlpha | BlendFactor::OneMinusSrcAlpha | BlendFactor::SrcAlphaSaturated
    )
}

/// Returns fragment outputs.
pub(crate) fn fragment_outputs(
    fragment: &RenderPipelineFragmentState,
    fragment_entry: Option<&str>,
) -> Result<BTreeMap<u32, shader_naga::ReflectedTypeClass>, String> {
    let Some(entry_name) = fragment_entry else {
        return Ok(BTreeMap::new());
    };
    #[cfg(feature = "shader-passthrough")]
    if fragment.shader.module.msl_passthrough().is_some() {
        return Ok(BTreeMap::new());
    }
    let Some(module) = fragment.shader.module.reflected() else {
        return Err("fragment module reflection failed".to_owned());
    };
    Ok(module
        .entry_point_io()
        .into_iter()
        .find(|io| io.entry_point == entry_name)
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
    output: shader_naga::ReflectedTypeClass,
    caps: FormatCaps,
) -> Result<(), String> {
    let Some(format_class) = caps.output_class else {
        return Err("render pipeline color target format has no output class".to_owned());
    };
    let output_class = match output.scalar {
        shader_naga::ReflectedTypeScalarClass::Float => FormatOutputClass::Float,
        shader_naga::ReflectedTypeScalarClass::Sint => FormatOutputClass::Sint,
        shader_naga::ReflectedTypeScalarClass::Uint => FormatOutputClass::Uint,
        shader_naga::ReflectedTypeScalarClass::Bool => {
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
    #[cfg(feature = "shader-passthrough")]
    if descriptor.vertex.shader.module.msl_passthrough().is_some()
        || fragment.shader.module.msl_passthrough().is_some()
    {
        return Ok(());
    }

    let outputs = inter_stage_outputs(&descriptor.vertex, vertex_entry)?;
    let inputs = inter_stage_inputs(fragment, fragment_entry)?;
    validate_inter_stage_limits(&outputs, limits, "output")?;
    validate_inter_stage_limits(&inputs, limits, "input")?;
    if matches!(descriptor.primitive.topology, PrimitiveTopology::PointList)
        && outputs.len() >= limits.max_inter_stage_shader_variables as usize
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
        if output.interpolation != input.interpolation {
            return Err(
                "render pipeline inter-stage interpolation types are incompatible".to_owned(),
            );
        }
        if output.sampling != input.sampling {
            return Err(
                "render pipeline inter-stage interpolation sampling is incompatible".to_owned(),
            );
        }
    }
    Ok(())
}

fn validate_inter_stage_limits(
    locations: &BTreeMap<u32, shader_naga::ReflectedIoLocation>,
    limits: Limits,
    label: &str,
) -> Result<(), String> {
    if locations.len() > limits.max_inter_stage_shader_variables as usize {
        return Err(format!(
            "render pipeline inter-stage {label} count exceeds the device limit"
        ));
    }
    if locations
        .keys()
        .any(|location| *location >= limits.max_inter_stage_shader_variables)
    {
        return Err(format!(
            "render pipeline inter-stage {label} location exceeds the device limit"
        ));
    }
    Ok(())
}

fn inter_stage_outputs(
    vertex: &RenderPipelineVertexState,
    vertex_entry: &str,
) -> Result<BTreeMap<u32, shader_naga::ReflectedIoLocation>, String> {
    let Some(module) = vertex.shader.module.reflected() else {
        return Err("vertex module reflection failed".to_owned());
    };
    Ok(module
        .entry_point_io()
        .into_iter()
        .find(|io| io.entry_point == vertex_entry)
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
) -> Result<BTreeMap<u32, shader_naga::ReflectedIoLocation>, String> {
    let Some(module) = fragment.shader.module.reflected() else {
        return Err("fragment module reflection failed".to_owned());
    };
    Ok(module
        .entry_point_io()
        .into_iter()
        .find(|io| io.entry_point == fragment_entry)
        .map(|io| {
            io.inputs
                .into_iter()
                .map(|input| (input.location, input))
                .collect()
        })
        .unwrap_or_default())
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

/// Returns effective render bind group layouts.
pub(crate) fn effective_render_bind_group_layouts(
    descriptor: &RenderPipelineDescriptor,
    vertex_entry: &str,
    fragment_entry: Option<&str>,
    limits: Limits,
    features: &FeatureSet,
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
            validate_render_auto_layout_storage_textures(&requirements)?;
            derive_bind_group_layouts(requirements, limits, features)
        }
    }
}

fn validate_render_auto_layout_storage_textures(
    requirements: &[StageResourceBinding],
) -> Result<(), String> {
    for requirement in requirements {
        if !requirement.binding.statically_used {
            continue;
        }
        let shader_naga::ReflectedResourceBindingKind::StorageTexture { format, access, .. } =
            &requirement.binding.kind
        else {
            continue;
        };
        let format = reflected_storage_texture_format(format)?;
        let access = reflected_storage_texture_access(access);
        if requirement.stage == PipelineShaderStage::Fragment
            && access != StorageTextureAccess::ReadOnly
            && format == TextureFormat::from_raw(TextureFormat::RGBA8_SINT)
        {
            return Err(
                "render pipeline auto layout storage texture format/access is unsupported"
                    .to_owned(),
            );
        }
    }
    Ok(())
}

/// Returns stage resource bindings.
pub(crate) fn stage_resource_bindings(
    stage: &RenderPipelineShaderStage,
    entry_point: &str,
    pipeline_stage: PipelineShaderStage,
) -> Result<Vec<StageResourceBinding>, String> {
    #[cfg(feature = "shader-passthrough")]
    if stage.module.msl_passthrough().is_some() {
        return Ok(Vec::new());
    }
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
            #[cfg(feature = "shader-passthrough")]
            if fragment.shader.module.msl_passthrough().is_some() {
                return Ok(());
            }
            let module = fragment
                .shader
                .module
                .reflected()
                .ok_or_else(|| "fragment module reflection failed".to_owned())?;
            if module
                .fragment_builtins()
                .into_iter()
                .any(|builtins| builtins.entry_point == entry_name && builtins.sample_mask)
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
    #[cfg(feature = "tiled")]
    use crate::subpass::{
        AttachmentLayout, SubpassDependency, SubpassDependencyType, SubpassInputAttachment,
        SubpassLayoutDesc, SubpassPassLayoutDescriptor,
    };
    use crate::test_helpers::*;
    #[cfg(any(feature = "shader-passthrough", feature = "tiled"))]
    use crate::ErrorFilter;

    use std::sync::Arc;

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
        let target = hal.color_targets[0];
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
        )
        .expect("separate WGSL modules should generate per-stage MSL");

        assert!(matches!(
            source,
            HalShaderSource::MslStagesWithBufferSizes { vertex, fragment, .. }
                if vertex.contains("vertex")
                    && fragment.as_ref().is_some_and(|fragment| fragment.contains("fragment"))
        ));
        assert_eq!(vertex_entry, "vs");
        assert_eq!(fragment_entry.as_deref(), Some("fs"));
        assert!(bindings.is_empty());
    }

    #[cfg(feature = "shader-passthrough")]
    fn spirv_words(source: &str, entry_point: &str, stage: naga::ShaderStage) -> Vec<u32> {
        shader_naga::parse_and_validate_wgsl(source)
            .expect("test WGSL should validate")
            .generate_spirv(
                entry_point,
                stage,
                &naga::back::PipelineConstants::default(),
            )
            .expect("test WGSL should generate SPIR-V")
    }

    #[cfg(feature = "shader-passthrough")]
    fn msl_render_reflection() -> MslReflection {
        MslReflection {
            entry_points: vec![
                MslEntryPoint {
                    name: "vs".to_owned(),
                    stage: SHADER_STAGE_VERTEX,
                    workgroup_size: [1, 1, 1],
                },
                MslEntryPoint {
                    name: "fs".to_owned(),
                    stage: SHADER_STAGE_FRAGMENT,
                    workgroup_size: [1, 1, 1],
                },
            ],
        }
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn select_render_shader_source_covers_passthrough_backend_matrix() {
        let device = noop_device();
        let wgsl_module = render_shader_module(&device);
        let wgsl_descriptor = render_pipeline_descriptor(Arc::clone(&wgsl_module));

        let vertex_source = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }";
        let fragment_source =
            "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }";
        let vertex_words = spirv_words(vertex_source, "vs", naga::ShaderStage::Vertex);
        let fragment_words = spirv_words(fragment_source, "fs", naga::ShaderStage::Fragment);
        let vertex_spirv = Arc::new(device.create_shader_module_spirv(vertex_words.clone()));
        let fragment_spirv = Arc::new(device.create_shader_module_spirv(fragment_words.clone()));
        let mut spirv_descriptor = render_pipeline_descriptor(Arc::clone(&vertex_spirv));
        spirv_descriptor
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = Arc::clone(&fragment_spirv);

        let msl_source =
            "vertex float4 vs() { return float4(0); }\nfragment float4 fs() { return float4(1); }"
                .to_owned();
        let msl_module =
            Arc::new(device.create_shader_module_msl(msl_source.clone(), msl_render_reflection()));
        let msl_descriptor = render_pipeline_descriptor(Arc::clone(&msl_module));
        let mut mixed_vertex_spirv = render_pipeline_descriptor(Arc::clone(&vertex_spirv));
        mixed_vertex_spirv
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = Arc::clone(&wgsl_module);
        let mut mixed_fragment_spirv = render_pipeline_descriptor(Arc::clone(&wgsl_module));
        mixed_fragment_spirv
            .fragment
            .as_mut()
            .expect("fragment should exist")
            .shader
            .module = Arc::clone(&fragment_spirv);

        let (source, vertex_entry, fragment_entry, bindings) = select_render_shader_source(
            HalBackend::Vulkan,
            &wgsl_descriptor,
            "vs",
            Some("fs"),
            &[],
            &[],
            &[],
        )
        .expect("WGSL should generate Vulkan SPIR-V stages");
        assert!(
            matches!(source, HalShaderSource::SpirVStages { vertex, fragment } if !vertex.is_empty() && fragment.as_ref().is_some_and(|fragment| !fragment.is_empty()))
        );
        assert_eq!(vertex_entry, "vs");
        assert_eq!(fragment_entry.as_deref(), Some("fs"));
        assert!(bindings.is_empty());

        let (source, _, _, _) = select_render_shader_source(
            HalBackend::Vulkan,
            &spirv_descriptor,
            "vs",
            Some("fs"),
            &[],
            &[],
            &[],
        )
        .expect("SPIR-V passthrough should select Vulkan SPIR-V stages");
        assert!(matches!(
            source,
            HalShaderSource::SpirVStages { vertex, fragment }
                if vertex == vertex_words && fragment.as_deref() == Some(fragment_words.as_slice())
        ));
        assert_eq!(
            select_render_shader_source(
                HalBackend::Vulkan,
                &mixed_vertex_spirv,
                "vs",
                Some("fs"),
                &[],
                &[],
                &[],
            )
            .expect_err("mixed SPIR-V vertex and WGSL fragment must be rejected"),
            "render pipeline cannot mix a SPIR-V passthrough module with a non-SPIR-V module"
        );
        assert_eq!(
            select_render_shader_source(
                HalBackend::Vulkan,
                &mixed_fragment_spirv,
                "vs",
                Some("fs"),
                &[],
                &[],
                &[],
            )
            .expect_err("mixed WGSL vertex and SPIR-V fragment must be rejected"),
            "render pipeline cannot mix a SPIR-V passthrough module with a non-SPIR-V module"
        );

        let (source, vertex_entry, fragment_entry, _) = select_render_shader_source(
            HalBackend::Metal,
            &msl_descriptor,
            "vs",
            Some("fs"),
            &[],
            &[],
            &[],
        )
        .expect("MSL passthrough should select Metal MSL");
        assert!(matches!(source, HalShaderSource::Msl(selected) if selected == msl_source));
        assert_eq!(vertex_entry, "vs");
        assert_eq!(fragment_entry.as_deref(), Some("fs"));

        assert_eq!(
            select_render_shader_source(
                HalBackend::Metal,
                &spirv_descriptor,
                "vs",
                Some("fs"),
                &[],
                &[],
                &[]
            )
            .expect_err("SPIR-V must not run on Metal"),
            "SPIR-V shader module cannot be used on the Metal backend"
        );
        assert_eq!(
            select_render_shader_source(
                HalBackend::Vulkan,
                &msl_descriptor,
                "vs",
                Some("fs"),
                &[],
                &[],
                &[]
            )
            .expect_err("MSL must not run on Vulkan"),
            "MSL shader module cannot be used on the Vulkan backend"
        );
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

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_render_pipeline_requires_explicit_layout_on_noop() {
        let device = noop_device();
        let module = Arc::new(device.create_shader_module_msl(
            "vertex float4 vs() { return float4(0); }\nfragment float4 fs() { return float4(1); }"
                .to_owned(),
            msl_render_reflection(),
        ));

        device.push_error_scope(ErrorFilter::Validation);
        let auto = device.create_render_pipeline(render_pipeline_descriptor(Arc::clone(&module)));
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("auto MSL render pipeline should be scoped");
        assert!(auto.is_error());
        assert_eq!(
            scoped.message,
            "MSL shader module requires an explicit pipeline layout"
        );

        let explicit_layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: Vec::new(),
            immediate_size: 0,
            error: None,
        }));
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.layout = RenderPipelineLayout::Explicit(explicit_layout);

        device.push_error_scope(ErrorFilter::Validation);
        let explicit = device.create_render_pipeline(descriptor);
        let scoped = device.pop_error_scope().expect("scope should exist");
        assert!(!explicit.is_error());
        assert_eq!(explicit.vertex_entry_name(), "vs");
        assert_eq!(explicit.fragment_entry_name(), Some("fs"));
        assert_eq!(scoped, None);
    }

    #[cfg(feature = "tiled")]
    fn subpass_input_shader(sample: &str) -> String {
        format!(
            "@group(0) @binding(0) var s: subpass_input<{sample}>;
             @vertex
             fn vs() -> @builtin(position) vec4<f32> {{
                 return vec4<f32>(0.0, 0.0, 0.0, 1.0);
             }}
             @fragment
             fn fs() -> @location(0) vec4<f32> {{
                 let loaded = subpassLoad(s);
                 if loaded.x == {zero} {{
                     return vec4<f32>(1.0, 0.0, 0.0, 1.0);
                 }}
                 return vec4<f32>(0.0, 1.0, 0.0, 1.0);
             }}",
            zero = if sample == "f32" { "0.0" } else { "0" }
        )
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

    fn depth32_float() -> TextureFormat {
        TextureFormat::from_raw(0x30)
    }

    #[cfg(feature = "tiled")]
    fn subpass_attachment_layout(format: TextureFormat) -> AttachmentLayout {
        AttachmentLayout {
            format,
            sample_count: 1,
        }
    }

    #[cfg(feature = "tiled")]
    fn multi_color_depth_layout_descriptor() -> SubpassPassLayoutDescriptor {
        SubpassPassLayoutDescriptor {
            color_attachments: vec![
                subpass_attachment_layout(rgba8_unorm()),
                subpass_attachment_layout(rgba8_unorm()),
                subpass_attachment_layout(rgba8_unorm()),
            ],
            depth_stencil_attachment: Some(subpass_attachment_layout(depth32_float())),
            subpasses: vec![
                SubpassLayoutDesc {
                    color_attachment_indices: vec![0, 1],
                    uses_depth_stencil: true,
                    input_attachments: Vec::new(),
                },
                SubpassLayoutDesc {
                    color_attachment_indices: vec![2],
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
    fn multi_output_render_shader_module(device: &crate::device::Device) -> Arc<ShaderModule> {
        Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@vertex
                 fn vs() -> @builtin(position) vec4<f32> {
                     return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                 }

                 struct Output {
                     @location(0) a: vec4<f32>,
                     @location(1) b: vec4<f32>,
                 }

                 @fragment
                 fn fs() -> Output {
                     var out: Output;
                     out.a = vec4<f32>(1.0, 0.0, 0.0, 1.0);
                     out.b = vec4<f32>(0.0, 1.0, 0.0, 1.0);
                     return out;
                 }"
                .to_owned(),
            )),
        )
    }

    #[cfg(feature = "tiled")]
    fn subpass_pipeline_descriptor_for_layout(
        device: &crate::device::Device,
        layout: Arc<SubpassPassLayout>,
    ) -> SubpassRenderPipelineDescriptor {
        let module = multi_output_render_shader_module(device);
        let mut base = render_pipeline_descriptor(module);
        base.depth_stencil = Some(DepthStencilState {
            format: depth32_float(),
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Less),
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
        });
        if let Some(fragment) = &mut base.fragment {
            fragment.target_count = 2;
            fragment.targets = vec![
                ColorTargetState {
                    format: rgba8_unorm(),
                    blend: None,
                    write_mask: 0xF,
                },
                ColorTargetState {
                    format: rgba8_unorm(),
                    blend: None,
                    write_mask: 0xF,
                },
            ];
        }
        SubpassRenderPipelineDescriptor {
            base,
            pass_layout: layout,
            subpass_index: 0,
            error: None,
        }
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_pipeline_accepts_depth_stencil() {
        let device = noop_device();
        let layout =
            Arc::new(device.create_subpass_pass_layout(multi_color_depth_layout_descriptor()));
        let descriptor = subpass_pipeline_descriptor_for_layout(&device, layout);

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_subpass_render_pipeline(descriptor);
        let scoped = device.pop_error_scope().expect("scope should exist");

        assert!(!pipeline.is_error());
        assert_eq!(scoped, None);
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_pipeline_accepts_multiple_color_targets() {
        let device = noop_device();
        let layout =
            Arc::new(device.create_subpass_pass_layout(multi_color_depth_layout_descriptor()));
        let descriptor = subpass_pipeline_descriptor_for_layout(&device, layout);

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_subpass_render_pipeline(descriptor);
        let scoped = device.pop_error_scope().expect("scope should exist");

        assert!(!pipeline.is_error());
        assert_eq!(scoped, None);
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_pipeline_rejects_multisample_above_one() {
        let device = noop_device();
        let mut layout_descriptor = multi_color_depth_layout_descriptor();
        for attachment in &mut layout_descriptor.color_attachments {
            attachment.sample_count = 4;
        }
        if let Some(depth) = &mut layout_descriptor.depth_stencil_attachment {
            depth.sample_count = 4;
        }
        let layout = Arc::new(device.create_subpass_pass_layout(layout_descriptor));
        let mut descriptor = subpass_pipeline_descriptor_for_layout(&device, layout);
        descriptor.base.multisample.count = 4;

        device.push_error_scope(ErrorFilter::Internal);
        let pipeline = device.create_subpass_render_pipeline(descriptor);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("multisample rejection should be scoped");

        assert!(pipeline.is_error());
        assert_eq!(
            scoped.message,
            "subpass render pipeline does not yet support multisample > 1"
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn validate_color_targets_subpass_accepts_both_location_conventions() {
        let device = noop_device();
        let limits = device.limits();
        let subpass_slot_one = [1];

        let subpass_local =
            render_pipeline_descriptor(render_shader_module_with_fragment_location(&device, 0));
        resolve_render_pipeline_descriptor(
            &subpass_local,
            limits,
            &device.features(),
            Some(&subpass_slot_one),
        )
        .expect("subpass-local fragment output should be accepted");

        let flat_slot =
            render_pipeline_descriptor(render_shader_module_with_fragment_location(&device, 1));
        resolve_render_pipeline_descriptor(
            &flat_slot,
            limits,
            &device.features(),
            Some(&subpass_slot_one),
        )
        .expect("flat-slot fragment output should be accepted");

        let missing_slot =
            render_pipeline_descriptor(render_shader_module_with_fragment_location(&device, 3));
        assert_eq!(
            resolve_render_pipeline_descriptor(
                &missing_slot,
                limits,
                &device.features(),
                Some(&[2])
            )
            .expect_err("unmatched fragment output should require writeMask 0"),
            "render pipeline color target without shader output must use writeMask 0"
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_input_shader_generates_spirv_and_msl_status_is_known() {
        let module = shader_naga::parse_and_validate_wgsl(&subpass_input_shader("f32"))
            .expect("subpass input WGSL should validate");

        let spirv = module
            .generate_spirv(
                "fs",
                naga::ShaderStage::Fragment,
                &naga::back::PipelineConstants::default(),
            )
            .expect("subpass input fragment shader should generate SPIR-V");
        assert!(!spirv.is_empty());

        let msl = module
            .generate_render_msl(
                "vs",
                Some("fs"),
                &shader_naga::MslBindingMap {
                    resources: Vec::new(),
                },
                &[],
                &[((0, 0), 0)],
                false,
            )
            .expect("naga must lower subpass_input when subpass_color_slots is populated");
        assert!(msl.source.contains("[[color("));
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_input_explicit_layout_checks_sample_type() {
        let device = noop_device();
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(subpass_input_shader("i32"))),
        );
        let float_layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
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
        let sint_layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: SHADER_STAGE_FRAGMENT,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::InputAttachment {
                    sample_type: TextureSampleType::Sint,
                    multisampled: false,
                }),
            }],
            error: None,
        }));

        let float_pipeline_layout =
            Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
                bind_group_layouts: vec![float_layout],
                immediate_size: 0,
                error: None,
            }));
        let mut mismatch = render_pipeline_descriptor(Arc::clone(&module));
        mismatch.layout = RenderPipelineLayout::Explicit(float_pipeline_layout);

        device.push_error_scope(ErrorFilter::Validation);
        let mismatch_pipeline = device.create_render_pipeline(mismatch);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("mismatch should be scoped");
        assert!(mismatch_pipeline.is_error());
        assert_eq!(
            scoped.message,
            "pipeline layout binding kind is incompatible with the shader binding"
        );

        let sint_pipeline_layout =
            Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
                bind_group_layouts: vec![sint_layout],
                immediate_size: 0,
                error: None,
            }));
        let mut matched = render_pipeline_descriptor(module);
        matched.layout = RenderPipelineLayout::Explicit(sint_pipeline_layout);

        device.push_error_scope(ErrorFilter::Validation);
        let matched_pipeline = device.create_render_pipeline(matched);
        let scoped = device.pop_error_scope().expect("scope should exist");
        assert!(!matched_pipeline.is_error());
        assert_eq!(scoped, None);
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn input_attachment_hal_bindings_extracts_only_input_attachment_entries() {
        let device = noop_device();
        // Group 0 mixes an input attachment (binding 0) with a uniform (binding 1);
        // group 1 holds only a uniform. Only the input attachment must be emitted.
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

        let bindings = input_attachment_hal_bindings(&[group0, group1]);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].group, 0);
        assert_eq!(bindings[0].binding, 0);
        assert!(matches!(
            bindings[0].kind,
            HalDescriptorBindingKind::Buffer(HalBufferBindingKind::InputAttachment)
        ));
    }
}
