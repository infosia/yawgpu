use std::cell::UnsafeCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{
    HalAdapter, HalAddressMode, HalBackend, HalBoundBuffer, HalBuffer, HalBufferBindingKind,
    HalBufferCopy, HalBufferTextureCopy, HalBufferTextureLayout, HalCompareFunction,
    HalComputePass, HalComputePipeline, HalCopy, HalDescriptorBinding, HalDevice, HalDraw,
    HalError, HalExtent3d, HalFilterMode, HalInstance, HalMipmapFilterMode, HalOrigin3d,
    HalPrimitiveTopology, HalQueue, HalRenderColorTarget, HalRenderLoadOp, HalRenderPass,
    HalRenderPipeline, HalRenderPipelineDescriptor, HalSampler, HalSamplerDescriptor,
    HalShaderSource, HalSurface, HalTexture, HalTextureCopy, HalTextureDescriptor,
    HalTextureFormat, HalTextureUsage, HalVertexAttribute, HalVertexBufferLayout, HalVertexFormat,
    HalVertexStepMode,
};

use crate::adapter::*;
use crate::bind_group::*;
use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::compute_pass::*;
use crate::compute_pipeline::*;
use crate::copy::*;
use crate::device::*;
use crate::error::*;
use crate::extent::*;
use crate::format::*;
use crate::future::*;
use crate::instance::*;
use crate::limits::*;
use crate::pass::*;
use crate::pipeline_layout::*;
use crate::query_set::*;
use crate::queue::*;
use crate::render_bundle::*;
use crate::render_pass::*;
use crate::sampler::*;
use crate::shader::*;
use crate::shader_naga;
use crate::texture::*;
use crate::texture_view::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AttachmentSignature {
    pub(crate) color_formats: Vec<Option<TextureFormat>>,
    pub(crate) depth_stencil_format: Option<TextureFormat>,
    pub(crate) sample_count: u32,
}

#[derive(Debug, Clone)]
pub struct RenderPipelineDescriptor {
    pub layout: RenderPipelineLayout,
    pub vertex: RenderPipelineVertexState,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    pub fragment: Option<RenderPipelineFragmentState>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RenderPipelineLayout {
    Auto,
    Explicit(Arc<PipelineLayout>),
}

#[derive(Debug, Clone)]
pub struct RenderPipelineVertexState {
    pub shader: RenderPipelineShaderStage,
    pub buffer_count: usize,
    pub buffers: Vec<VertexBufferLayout>,
}

#[derive(Debug, Clone)]
pub struct VertexBufferLayout {
    pub array_stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: Vec<VertexAttribute>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexStepMode {
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u64,
    pub shader_location: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexFormat(u32);

impl VertexFormat {
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }

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

#[derive(Debug, Clone, Copy)]
pub(crate) struct VertexFormatInfo {
    pub(crate) byte_size: u64,
    pub(crate) output_class: FormatOutputClass,
}

impl VertexFormatInfo {
    pub(crate) const fn new(byte_size: u64, output_class: FormatOutputClass) -> Self {
        Self {
            byte_size,
            output_class,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderPipelineFragmentState {
    pub shader: RenderPipelineShaderStage,
    pub target_count: usize,
    pub targets: Vec<ColorTargetState>,
}

#[derive(Debug, Clone)]
pub struct RenderPipelineShaderStage {
    pub module: Arc<ShaderModule>,
    pub entry_point: Option<String>,
    pub constants: Vec<PipelineConstant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: bool,
    pub write_mask: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub strip_index_format: Option<IndexFormat>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth_write_enabled: Option<bool>,
    pub depth_compare: Option<CompareFunction>,
    pub stencil_front: StencilFaceState,
    pub stencil_back: StencilFaceState,
    pub stencil_read_mask: u32,
    pub stencil_write_mask: u32,
    pub depth_bias: i32,
    pub depth_bias_slope_scale: f32,
    pub depth_bias_clamp: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilFaceState {
    pub compare: CompareFunction,
    pub fail_op: StencilOperation,
    pub depth_fail_op: StencilOperation,
    pub pass_op: StencilOperation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StencilOperation {
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultisampleState {
    pub count: u32,
    pub mask: u32,
    pub alpha_to_coverage_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct RenderPipeline {
    pub(crate) inner: Arc<RenderPipelineInner>,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalVertexBufferBinding {
    pub(crate) slot: u32,
    pub(crate) metal_index: u32,
}

impl RenderPipeline {
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

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub fn vertex_entry_name(&self) -> &str {
        &self.inner.vertex_entry_name
    }

    #[must_use]
    pub fn fragment_entry_name(&self) -> Option<&str> {
        self.inner.fragment_entry_name.as_deref()
    }

    #[must_use]
    pub fn bind_group_layouts(&self) -> &[Arc<BindGroupLayout>] {
        &self.inner.bind_group_layouts
    }

    pub(crate) fn hal(&self) -> Option<HalRenderPipeline> {
        self.inner.hal.clone()
    }

    pub(crate) fn metal_bindings(&self) -> &[MetalBufferBinding] {
        &self.inner.metal_bindings
    }

    pub(crate) fn vertex_buffer_bindings(&self) -> &[MetalVertexBufferBinding] {
        &self.inner.vertex_buffer_bindings
    }

    #[must_use]
    pub(crate) fn required_vertex_buffer_count(&self) -> usize {
        self.inner._vertex.buffer_count
    }

    #[must_use]
    pub(crate) fn vertex_buffer_layouts(&self) -> &[VertexBufferLayout] {
        &self.inner._vertex.buffers
    }

    #[must_use]
    pub(crate) fn primitive_state(&self) -> PrimitiveState {
        self.inner._primitive
    }

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

pub(crate) fn validate_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    limits: Limits,
) -> Option<String> {
    resolve_render_pipeline_descriptor(descriptor, limits).err()
}

pub(crate) type ResolvedRenderPipelineParts = (String, Option<String>, Vec<Arc<BindGroupLayout>>);

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
    let Some(fragment) = &descriptor.fragment else {
        return (
            None,
            Some("Metal render pipeline requires a fragment stage".to_owned()),
        );
    };
    let Some(fragment_entry_name) = fragment_entry_name else {
        return (
            None,
            Some("real render pipeline requires a fragment entry point".to_owned()),
        );
    };
    let (shader, vertex_entry_point, fragment_entry_point, descriptor_bindings) = match hal_device
        .backend()
    {
        HalBackend::Metal => {
            if !Arc::ptr_eq(&descriptor.vertex.shader.module, &fragment.shader.module) {
                return (
                    None,
                    Some(
                        "Metal render pipeline requires vertex and fragment entries in the same WGSL module"
                            .to_owned(),
                    ),
                );
            }
            let Some(module) = descriptor.vertex.shader.module.validated_wgsl() else {
                return (
                    None,
                    Some("render pipeline requires a valid WGSL shader module".to_owned()),
                );
            };
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
            let msl_vertex_buffers = match msl_vertex_buffer_bindings(
                &descriptor.vertex.buffers,
                vertex_buffer_bindings,
            ) {
                Ok(bindings) => bindings,
                Err(message) => return (None, Some(message)),
            };
            let generated = match module.generate_render_msl(
                vertex_entry_name,
                fragment_entry_name,
                &msl_binding_map,
                &msl_vertex_buffers,
            ) {
                Ok(generated) => generated,
                Err(message) => return (None, Some(message)),
            };
            (
                HalShaderSource::Msl(generated.source),
                generated.vertex_entry_point,
                generated.fragment_entry_point,
                Vec::new(),
            )
        }
        HalBackend::Vulkan => {
            let Some(vertex_module) = descriptor.vertex.shader.module.validated_wgsl() else {
                return (
                    None,
                    Some("render pipeline requires a valid WGSL vertex shader module".to_owned()),
                );
            };
            let Some(fragment_module) = fragment.shader.module.validated_wgsl() else {
                return (
                    None,
                    Some("render pipeline requires a valid WGSL fragment shader module".to_owned()),
                );
            };
            let vertex =
                match vertex_module.generate_spirv(vertex_entry_name, naga::ShaderStage::Vertex) {
                    Ok(spirv) => spirv,
                    Err(message) => return (None, Some(message)),
                };
            let fragment = match fragment_module
                .generate_spirv(fragment_entry_name, naga::ShaderStage::Fragment)
            {
                Ok(spirv) => spirv,
                Err(message) => return (None, Some(message)),
            };
            (
                HalShaderSource::SpirVStages { vertex, fragment },
                vertex_entry_name.to_owned(),
                fragment_entry_name.to_owned(),
                hal_descriptor_bindings(metal_bindings),
            )
        }
        _ => return (None, None),
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

pub(crate) fn hal_vertex_format(format: VertexFormat) -> HalVertexFormat {
    match format.0 {
        0x0000_001C => HalVertexFormat::Float32,
        0x0000_001D => HalVertexFormat::Float32x2,
        0x0000_001E => HalVertexFormat::Float32x3,
        0x0000_001F => HalVertexFormat::Float32x4,
        _ => HalVertexFormat::Unsupported,
    }
}

pub(crate) fn hal_primitive_topology(topology: PrimitiveTopology) -> HalPrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => HalPrimitiveTopology::PointList,
        PrimitiveTopology::LineList => HalPrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip => HalPrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList => HalPrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip => HalPrimitiveTopology::TriangleStrip,
    }
}

pub(crate) fn resolve_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    limits: Limits,
) -> Result<ResolvedRenderPipelineParts, String> {
    if let RenderPipelineLayout::Explicit(layout) = &descriptor.layout {
        if layout.is_error() {
            return Err("render pipeline layout must not be an error pipeline layout".to_owned());
        }
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

pub(crate) fn vertex_inputs(
    vertex: &RenderPipelineVertexState,
    vertex_entry: &str,
) -> Result<BTreeMap<u32, shader_naga::ReflectedTypeClass>, String> {
    let Some(module) = vertex.shader.module.validated_wgsl() else {
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
    let Some(module) = stage.module.validated_wgsl() else {
        return Err(format!(
            "render pipeline {label} stage requires a valid WGSL shader module"
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

pub(crate) fn validate_render_constants(stage: &RenderPipelineShaderStage) -> Result<(), String> {
    let Some(module) = stage.module.validated_wgsl() else {
        return Err("render pipeline stage requires a valid WGSL shader module".to_owned());
    };
    resolve_pipeline_constants(&module.overrides(), &stage.constants)?;
    Ok(())
}

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

pub(crate) fn depth_stencil_uses_stencil(depth_stencil: DepthStencilState) -> bool {
    stencil_face_uses_stencil(depth_stencil.stencil_front)
        || stencil_face_uses_stencil(depth_stencil.stencil_back)
        || depth_stencil.stencil_read_mask != u32::MAX
        || depth_stencil.stencil_write_mask != u32::MAX
}

pub(crate) fn stencil_face_uses_stencil(face: StencilFaceState) -> bool {
    face.compare != CompareFunction::Always
        || face.fail_op != StencilOperation::Keep
        || face.depth_fail_op != StencilOperation::Keep
        || face.pass_op != StencilOperation::Keep
}

pub(crate) fn validate_fragment_depth_output(
    descriptor: &RenderPipelineDescriptor,
    fragment_entry: Option<&str>,
) -> Result<(), String> {
    let Some(fragment) = &descriptor.fragment else {
        return Ok(());
    };
    let Some(entry_name) = fragment_entry else {
        return Ok(());
    };
    let Some(module) = fragment.shader.module.validated_wgsl() else {
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

    let outputs = fragment_outputs(fragment, fragment_entry)?;
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

pub(crate) fn fragment_outputs(
    fragment: &RenderPipelineFragmentState,
    fragment_entry: Option<&str>,
) -> Result<BTreeMap<u32, shader_naga::ReflectedTypeClass>, String> {
    let Some(entry_name) = fragment_entry else {
        return Ok(BTreeMap::new());
    };
    let Some(module) = fragment.shader.module.validated_wgsl() else {
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

pub(crate) fn stage_resource_bindings(
    stage: &RenderPipelineShaderStage,
    entry_point: &str,
    pipeline_stage: PipelineShaderStage,
) -> Result<Vec<StageResourceBinding>, String> {
    let Some(module) = stage.module.validated_wgsl() else {
        return Err("render pipeline stage requires a valid WGSL shader module".to_owned());
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
                .validated_wgsl()
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
    use crate::test_helpers::*;
    use crate::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
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
}
