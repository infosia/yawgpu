use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use yawgpu_hal::{
    HalBackend, HalDescriptorBinding, HalDevice, HalPrimitiveTopology, HalRenderPipeline,
    HalRenderPipelineDescriptor, HalShaderSource, HalVertexAttribute, HalVertexBufferLayout,
    HalVertexFormat, HalVertexStepMode,
};

use crate::bind_group_layout::*;
use crate::compute_pipeline::*;
use crate::format::*;
use crate::limits::*;
use crate::pipeline_layout::*;
use crate::sampler::*;
use crate::shader::*;
use crate::shader_naga;
use crate::texture::*;

/// Stores attachment signature data used by validation and backend submission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttachmentSignature {
    pub(crate) color_formats: Vec<Option<TextureFormat>>,
    pub(crate) depth_stencil_format: Option<TextureFormat>,
    pub(crate) sample_count: u32,
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
    pub blend: bool,
    /// Write mask.
    pub write_mask: u64,
}

/// Tracks the lifecycle state for primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveState {
    /// Topology.
    pub topology: PrimitiveTopology,
    /// Strip index format.
    pub strip_index_format: Option<IndexFormat>,
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
    pub(crate) is_error: bool,
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
        hal_device: Option<&HalDevice>,
    ) -> (Self, Option<String>) {
        let resolved = if is_error {
            None
        } else {
            resolve_render_pipeline_descriptor(&descriptor, limits).ok()
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
        }
    }
}

/// Validates render pipeline descriptor and returns a descriptive error on failure.
pub(crate) fn validate_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    limits: Limits,
) -> Option<String> {
    resolve_render_pipeline_descriptor(descriptor, limits).err()
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
        return (None, None);
    }
    if descriptor.depth_stencil.is_some()
        || descriptor.multisample.count != 1
        || descriptor
            .fragment
            .as_ref()
            .map_or(0, |fragment| fragment.target_count)
            != 1
    {
        return (
            None,
            Some(
                "real render pipeline currently supports one single-sampled color target only"
                    .to_owned(),
            ),
        );
    }
    if descriptor.fragment.is_none() {
        return (
            None,
            Some("Metal render pipeline requires a fragment stage".to_owned()),
        );
    }
    let Some(fragment_entry_name) = fragment_entry_name else {
        return (
            None,
            Some("real render pipeline requires a fragment entry point".to_owned()),
        );
    };
    let (shader, vertex_entry_point, fragment_entry_point, descriptor_bindings) =
        match select_render_shader_source(
            hal_device.backend(),
            descriptor,
            vertex_entry_name,
            fragment_entry_name,
            metal_bindings,
            vertex_buffer_bindings,
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
        &fragment_entry_point,
        &hal_descriptor,
        &descriptor_bindings,
    ) {
        Ok(pipeline) => (Some(pipeline), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

/// Selects the HAL shader source for a render pipeline.
pub(crate) fn select_render_shader_source(
    backend: HalBackend,
    descriptor: &RenderPipelineDescriptor,
    vertex_entry_name: &str,
    fragment_entry_name: &str,
    metal_bindings: &[MetalBufferBinding],
    vertex_buffer_bindings: &[MetalVertexBufferBinding],
) -> Result<(HalShaderSource, String, String, Vec<HalDescriptorBinding>), String> {
    let fragment = descriptor
        .fragment
        .as_ref()
        .ok_or_else(|| "real render pipeline requires a fragment stage".to_owned())?;
    match backend {
        HalBackend::Metal => {
            #[cfg(feature = "shader-passthrough")]
            if let Some((source, _)) = descriptor.vertex.shader.module.msl_passthrough() {
                if !Arc::ptr_eq(&descriptor.vertex.shader.module, &fragment.shader.module) {
                    return Err(
                        "Metal render pipeline requires vertex and fragment entries in the same MSL module"
                            .to_owned(),
                    );
                }
                return Ok((
                    HalShaderSource::Msl(source.to_owned()),
                    vertex_entry_name.to_owned(),
                    fragment_entry_name.to_owned(),
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
                || fragment.shader.module.spirv_passthrough().is_some()
            {
                return Err("SPIR-V shader module cannot be used on the Metal backend".to_owned());
            }
            if !Arc::ptr_eq(&descriptor.vertex.shader.module, &fragment.shader.module) {
                return Err(
                    "Metal render pipeline requires vertex and fragment entries in the same WGSL module"
                        .to_owned(),
                );
            }
            let module =
                descriptor.vertex.shader.module.reflected().ok_or_else(|| {
                    "render pipeline requires a reflected shader module".to_owned()
                })?;
            let msl_binding_map = shader_naga::MslBindingMap {
                buffers: metal_bindings
                    .iter()
                    .map(|binding| shader_naga::MslBufferBinding {
                        group: binding.group,
                        binding: binding.binding,
                        metal_index: binding.metal_index,
                    })
                    .collect(),
            };
            let msl_vertex_buffers =
                msl_vertex_buffer_bindings(&descriptor.vertex.buffers, vertex_buffer_bindings)?;
            let generated = module.generate_render_msl(
                vertex_entry_name,
                fragment_entry_name,
                &msl_binding_map,
                &msl_vertex_buffers,
            )?;
            Ok((
                HalShaderSource::Msl(generated.source),
                generated.vertex_entry_point,
                generated.fragment_entry_point,
                Vec::new(),
            ))
        }
        HalBackend::Vulkan => {
            #[cfg(feature = "shader-passthrough")]
            if descriptor.vertex.shader.module.msl_passthrough().is_some()
                || fragment.shader.module.msl_passthrough().is_some()
            {
                return Err("MSL shader module cannot be used on the Vulkan backend".to_owned());
            }
            #[cfg(feature = "shader-passthrough")]
            if let (Some((vertex_words, _)), Some((fragment_words, _))) = (
                descriptor.vertex.shader.module.spirv_passthrough(),
                fragment.shader.module.spirv_passthrough(),
            ) {
                return Ok((
                    HalShaderSource::SpirVStages {
                        vertex: vertex_words.to_vec(),
                        fragment: fragment_words.to_vec(),
                    },
                    vertex_entry_name.to_owned(),
                    fragment_entry_name.to_owned(),
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
                || fragment.shader.module.spirv_passthrough().is_some()
            {
                return Err(
                    "render pipeline cannot mix a SPIR-V passthrough module with a non-SPIR-V module"
                        .to_owned(),
                );
            }
            let vertex_module = descriptor.vertex.shader.module.reflected().ok_or_else(|| {
                "render pipeline requires a reflected vertex shader module".to_owned()
            })?;
            let fragment_module = fragment.shader.module.reflected().ok_or_else(|| {
                "render pipeline requires a reflected fragment shader module".to_owned()
            })?;
            let vertex =
                vertex_module.generate_spirv(vertex_entry_name, naga::ShaderStage::Vertex)?;
            let fragment =
                fragment_module.generate_spirv(fragment_entry_name, naga::ShaderStage::Fragment)?;
            Ok((
                HalShaderSource::SpirVStages { vertex, fragment },
                vertex_entry_name.to_owned(),
                fragment_entry_name.to_owned(),
                hal_descriptor_bindings(metal_bindings),
            ))
        }
        HalBackend::Noop => Err("Noop backend does not create HAL shader sources".to_owned()),
        _ => Err("unsupported backend does not create HAL shader sources".to_owned()),
    }
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
    let color_formats = descriptor
        .fragment
        .as_ref()
        .map(|fragment| {
            fragment
                .targets
                .iter()
                .map(|target| hal_texture_format(target.format))
                .collect()
        })
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
        color_formats,
        vertex_buffers,
        primitive_topology: hal_primitive_topology(descriptor.primitive.topology),
    })
}

/// Returns msl vertex format.
pub(crate) fn msl_vertex_format(
    format: VertexFormat,
) -> Result<shader_naga::MslVertexFormat, String> {
    match format.0 {
        0x0000_001C => Ok(shader_naga::MslVertexFormat::Float32),
        0x0000_001D => Ok(shader_naga::MslVertexFormat::Float32x2),
        0x0000_001E => Ok(shader_naga::MslVertexFormat::Float32x3),
        0x0000_001F => Ok(shader_naga::MslVertexFormat::Float32x4),
        _ => Err("Metal render pipeline currently supports Float32 vertex formats only".to_owned()),
    }
}

/// Returns HAL vertex format.
pub(crate) fn hal_vertex_format(format: VertexFormat) -> HalVertexFormat {
    match format.0 {
        0x0000_001C => HalVertexFormat::Float32,
        0x0000_001D => HalVertexFormat::Float32x2,
        0x0000_001E => HalVertexFormat::Float32x3,
        0x0000_001F => HalVertexFormat::Float32x4,
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
        validate_depth_stencil_aspects(depth_stencil)?;
    }
    validate_fragment_depth_output(descriptor, fragment_entry.as_deref())?;
    validate_color_targets(descriptor, fragment_entry.as_deref(), limits)?;
    validate_render_pipeline_layout(descriptor, &vertex_entry, fragment_entry.as_deref())?;
    validate_multisample_state(descriptor, fragment_entry.as_deref())?;
    let bind_group_layouts = effective_render_bind_group_layouts(
        descriptor,
        &vertex_entry,
        fragment_entry.as_deref(),
        limits,
    )?;

    Ok((vertex_entry, fragment_entry, bind_group_layouts))
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
    if descriptor
        .fragment
        .as_ref()
        .is_some_and(|fragment| fragment.target_count == 0)
    {
        return Err("render pipeline fragment targetCount must be at least one".to_owned());
    }
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
) -> Result<(), String> {
    let caps = depth_stencil.format.caps();
    let has_depth = caps.is_some_and(|caps| caps.aspects.depth);
    let has_stencil = caps.is_some_and(|caps| caps.aspects.stencil);

    if (depth_stencil.depth_compare.is_some() || depth_stencil.depth_write_enabled == Some(true))
        && !has_depth
    {
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
            .and_then(|state| state.format.caps())
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
) -> Result<(), String> {
    let Some(fragment) = &descriptor.fragment else {
        return Ok(());
    };
    if fragment.targets.len() != fragment.target_count {
        return Err("render pipeline fragment target array must match targetCount".to_owned());
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
            if target.blend {
                return Err("render pipeline undefined color target must not have blend".to_owned());
            }
            continue;
        }

        let caps = target
            .format
            .caps()
            .ok_or_else(|| "render pipeline color target format must be defined".to_owned())?;
        if !caps.renderable {
            return Err("render pipeline color target format must be renderable".to_owned());
        }
        if target.blend && !caps.is_blendable {
            return Err("render pipeline color target format must be blendable".to_owned());
        }
        if descriptor.multisample.alpha_to_coverage_enabled && caps.is_blendable && caps.has_alpha {
            has_alpha_to_coverage_target = true;
        }

        if !skip_shader_outputs {
            match outputs.get(&(index as u32)) {
                Some(output) => validate_fragment_output_compat(*output, caps)?,
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
            .checked_add(caps.texel_block_size)
            .ok_or_else(|| "render pipeline color target byte count overflows".to_owned())?;
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
            derive_bind_group_layouts(requirements, limits)
        }
    }
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
    #[cfg(any(feature = "shader-passthrough", feature = "tiled"))]
    use crate::test_helpers::*;
    #[cfg(any(feature = "shader-passthrough", feature = "tiled"))]
    use crate::ErrorFilter;

    #[cfg(any(feature = "shader-passthrough", feature = "tiled"))]
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

    #[cfg(feature = "shader-passthrough")]
    fn spirv_words(source: &str, entry_point: &str, stage: naga::ShaderStage) -> Vec<u32> {
        shader_naga::parse_and_validate_wgsl(source)
            .expect("test WGSL should validate")
            .generate_spirv(entry_point, stage)
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

        let (source, vertex_entry, fragment_entry, bindings) =
            select_render_shader_source(HalBackend::Vulkan, &wgsl_descriptor, "vs", "fs", &[], &[])
                .expect("WGSL should generate Vulkan SPIR-V stages");
        assert!(
            matches!(source, HalShaderSource::SpirVStages { vertex, fragment } if !vertex.is_empty() && !fragment.is_empty())
        );
        assert_eq!(vertex_entry, "vs");
        assert_eq!(fragment_entry, "fs");
        assert!(bindings.is_empty());

        let (source, _, _, _) = select_render_shader_source(
            HalBackend::Vulkan,
            &spirv_descriptor,
            "vs",
            "fs",
            &[],
            &[],
        )
        .expect("SPIR-V passthrough should select Vulkan SPIR-V stages");
        assert!(matches!(
            source,
            HalShaderSource::SpirVStages { vertex, fragment }
                if vertex == vertex_words && fragment == fragment_words
        ));
        assert_eq!(
            select_render_shader_source(
                HalBackend::Vulkan,
                &mixed_vertex_spirv,
                "vs",
                "fs",
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
                "fs",
                &[],
                &[],
            )
            .expect_err("mixed WGSL vertex and SPIR-V fragment must be rejected"),
            "render pipeline cannot mix a SPIR-V passthrough module with a non-SPIR-V module"
        );

        let (source, vertex_entry, fragment_entry, _) =
            select_render_shader_source(HalBackend::Metal, &msl_descriptor, "vs", "fs", &[], &[])
                .expect("MSL passthrough should select Metal MSL");
        assert!(matches!(source, HalShaderSource::Msl(selected) if selected == msl_source));
        assert_eq!(vertex_entry, "vs");
        assert_eq!(fragment_entry, "fs");

        assert_eq!(
            select_render_shader_source(HalBackend::Metal, &spirv_descriptor, "vs", "fs", &[], &[])
                .expect_err("SPIR-V must not run on Metal"),
            "SPIR-V shader module cannot be used on the Metal backend"
        );
        assert_eq!(
            select_render_shader_source(HalBackend::Vulkan, &msl_descriptor, "vs", "fs", &[], &[])
                .expect_err("MSL must not run on Vulkan"),
            "MSL shader module cannot be used on the Vulkan backend"
        );
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
    #[test]
    fn subpass_input_shader_generates_spirv_and_msl_status_is_known() {
        let module = shader_naga::parse_and_validate_wgsl(&subpass_input_shader("f32"))
            .expect("subpass input WGSL should validate");

        let spirv = module
            .generate_spirv("fs", naga::ShaderStage::Fragment)
            .expect("subpass input fragment shader should generate SPIR-V");
        assert!(!spirv.is_empty());

        let msl = module.generate_render_msl(
            "vs",
            "fs",
            &shader_naga::MslBindingMap {
                buffers: Vec::new(),
            },
            &[],
        );
        if let Ok(msl) = msl {
            assert!(msl.source.contains("[[color("));
        } else {
            // The pinned naga MSL backend needs a subpass color-slot map for
            // global `subpass_input` values; B4 supplies that pass-local map.
        }
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
}
