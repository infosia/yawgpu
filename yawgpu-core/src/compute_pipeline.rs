use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use yawgpu_hal::{
    HalBackend, HalBufferBindingKind, HalComputePipeline, HalDescriptorBinding,
    HalDescriptorBindingKind, HalDevice, HalShaderSource,
};

use crate::bind_group_layout::*;
use crate::device::FeatureSet;
use crate::format::*;
use crate::limits::*;
use crate::pipeline_layout::*;
use crate::shader::*;
use crate::shader_naga;
use crate::texture_view::*;

/// Describes compute pipeline descriptor.
#[derive(Debug, Clone)]
pub struct ComputePipelineDescriptor {
    /// Layout.
    pub layout: ComputePipelineLayout,
    /// Shader module.
    pub shader_module: Arc<ShaderModule>,
    /// Entry point.
    pub entry_point: Option<String>,
    /// Constants.
    pub constants: Vec<PipelineConstant>,
    /// Error.
    pub error: Option<String>,
}

/// Enumerates compute pipeline layout values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ComputePipelineLayout {
    /// Auto variant.
    Auto,
    /// Explicit variant.
    Explicit(Arc<PipelineLayout>),
}

/// Stores pipeline constant data used by validation and backend submission.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineConstant {
    /// Key.
    pub key: String,
    /// Value.
    pub value: f64,
}

/// Stores compute pipeline data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct ComputePipeline {
    pub(crate) inner: Arc<ComputePipelineInner>,
}

/// Holds shared state for the compute pipeline handle.
#[derive(Debug)]
pub(crate) struct ComputePipelineInner {
    pub(crate) _layout: ComputePipelineLayout,
    pub(crate) _shader_module: Arc<ShaderModule>,
    pub(crate) entry_name: String,
    pub(crate) _bindings: Vec<shader_naga::ReflectedResourceBinding>,
    pub(crate) metal_bindings: Vec<MetalBufferBinding>,
    pub(crate) hal: Option<HalComputePipeline>,
    pub(crate) bind_group_layouts: Vec<Arc<BindGroupLayout>>,
    pub(crate) is_error: bool,
}

/// Stores resolved compute workgroup data used by validation and backend submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResolvedComputeWorkgroup {
    pub(crate) size: [u32; 3],
    pub(crate) storage_size: u64,
}

/// Stores binding metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalBufferBinding {
    pub(crate) group: u32,
    pub(crate) binding: u32,
    pub(crate) metal_index: u32,
    pub(crate) kind: MetalBindingKind,
}

/// Stores the shader resource kind for a resolved Metal binding slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalBindingKind {
    /// Buffer binding.
    Buffer(BufferBindingType),
    /// Sampled texture binding.
    Texture,
    /// Sampler binding.
    Sampler,
}

impl ComputePipeline {
    /// Creates a new instance.
    pub(crate) fn new(
        descriptor: ComputePipelineDescriptor,
        is_error: bool,
        limits: Limits,
        features: &FeatureSet,
        hal_device: Option<&HalDevice>,
    ) -> (Self, Option<String>) {
        let resolved = if is_error {
            None
        } else {
            resolve_compute_pipeline_descriptor(&descriptor, limits, features).ok()
        };
        let (entry_name, bindings, workgroup, bind_group_layouts) = resolved.unwrap_or_else(|| {
            (
                descriptor.entry_point.clone().unwrap_or_default(),
                Vec::new(),
                None,
                Vec::new(),
            )
        });
        let metal_bindings = metal_buffer_binding_map(&bind_group_layouts);
        let (hal, backend_error) = if is_error {
            (None, None)
        } else {
            create_hal_compute_pipeline(
                hal_device,
                &descriptor.shader_module,
                &entry_name,
                workgroup,
                &metal_bindings,
            )
        };
        let is_error = is_error || backend_error.is_some();
        (
            Self {
                inner: Arc::new(ComputePipelineInner {
                    _layout: descriptor.layout,
                    _shader_module: descriptor.shader_module,
                    entry_name,
                    _bindings: bindings,
                    metal_bindings,
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

    /// Returns the entry name.
    #[must_use]
    pub fn entry_name(&self) -> &str {
        &self.inner.entry_name
    }

    /// Returns the bind group layouts.
    #[must_use]
    pub fn bind_group_layouts(&self) -> &[Arc<BindGroupLayout>] {
        &self.inner.bind_group_layouts
    }

    /// Returns the HAL.
    pub(crate) fn hal(&self) -> Option<HalComputePipeline> {
        self.inner.hal.clone()
    }

    /// Returns the metal bindings.
    pub(crate) fn metal_bindings(&self) -> &[MetalBufferBinding] {
        &self.inner.metal_bindings
    }
}

/// Alias for resolved pipeline parts.
pub(crate) type ResolvedPipelineParts = (
    String,
    Vec<shader_naga::ReflectedResourceBinding>,
    Option<ResolvedComputeWorkgroup>,
    Vec<Arc<BindGroupLayout>>,
);

/// Creates HAL compute pipeline and reports validation errors through the owning device.
pub(crate) fn create_hal_compute_pipeline(
    hal_device: Option<&HalDevice>,
    shader_module: &ShaderModule,
    entry_name: &str,
    workgroup: Option<ResolvedComputeWorkgroup>,
    metal_bindings: &[MetalBufferBinding],
) -> (Option<HalComputePipeline>, Option<String>) {
    let Some(hal_device) = hal_device else {
        return (None, None);
    };
    if matches!(hal_device.backend(), HalBackend::Noop) {
        return (Some(HalComputePipeline::Noop), None);
    }
    let Some(workgroup) = workgroup else {
        return (
            None,
            Some("compute pipeline workgroup size reflection failed".to_owned()),
        );
    };
    let (shader, entry_point, descriptor_bindings) = match select_compute_shader_source(
        hal_device.backend(),
        shader_module,
        entry_name,
        metal_bindings,
    ) {
        Ok(selection) => selection,
        Err(message) => return (None, Some(message)),
    };
    match hal_device.create_compute_pipeline(
        shader,
        &entry_point,
        (workgroup.size[0], workgroup.size[1], workgroup.size[2]),
        &descriptor_bindings,
    ) {
        Ok(pipeline) => (Some(pipeline), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

/// Selects the HAL shader source for a compute pipeline.
pub(crate) fn select_compute_shader_source(
    backend: HalBackend,
    shader_module: &ShaderModule,
    entry_name: &str,
    metal_bindings: &[MetalBufferBinding],
) -> Result<(HalShaderSource, String, Vec<HalDescriptorBinding>), String> {
    match backend {
        HalBackend::Metal => {
            #[cfg(feature = "shader-passthrough")]
            if let Some((source, _)) = shader_module.msl_passthrough() {
                return Ok((
                    HalShaderSource::Msl(source.to_owned()),
                    entry_name.to_owned(),
                    Vec::new(),
                ));
            }
            #[cfg(feature = "shader-passthrough")]
            if shader_module.spirv_passthrough().is_some() {
                return Err("SPIR-V shader module cannot be used on the Metal backend".to_owned());
            }
            let module = shader_module
                .reflected()
                .ok_or_else(|| "compute pipeline requires a reflected shader module".to_owned())?;
            let msl_binding_map = shader_naga::MslBindingMap {
                resources: msl_resource_bindings(metal_bindings),
            };
            let generated = module.generate_msl(entry_name, &msl_binding_map)?;
            Ok((
                HalShaderSource::Msl(generated.source),
                generated.entry_point,
                Vec::new(),
            ))
        }
        HalBackend::Vulkan => {
            #[cfg(feature = "shader-passthrough")]
            if let Some((words, _)) = shader_module.spirv_passthrough() {
                return Ok((
                    HalShaderSource::SpirV(words.to_vec()),
                    entry_name.to_owned(),
                    hal_descriptor_bindings(metal_bindings),
                ));
            }
            #[cfg(feature = "shader-passthrough")]
            if shader_module.msl_passthrough().is_some() {
                return Err("MSL shader module cannot be used on the Vulkan backend".to_owned());
            }
            let module = shader_module
                .reflected()
                .ok_or_else(|| "compute pipeline requires a reflected shader module".to_owned())?;
            let spirv = module.generate_spirv(entry_name, naga::ShaderStage::Compute)?;
            Ok((
                HalShaderSource::SpirV(spirv),
                entry_name.to_owned(),
                hal_descriptor_bindings(metal_bindings),
            ))
        }
        #[cfg(feature = "gles")]
        HalBackend::Gles => {
            #[cfg(feature = "shader-passthrough")]
            if shader_module.spirv_passthrough().is_some()
                || shader_module.msl_passthrough().is_some()
            {
                return Err(
                    "passthrough shader modules cannot be used on the GLES backend".to_owned(),
                );
            }
            let module = shader_module
                .reflected()
                .ok_or_else(|| "compute pipeline requires a reflected shader module".to_owned())?;
            let generated = module.generate_glsl(entry_name, naga::ShaderStage::Compute)?;
            Ok((
                HalShaderSource::Glsl {
                    source: generated.source,
                    stage: yawgpu_hal::HalShaderStage::Compute,
                },
                generated.entry_point,
                hal_descriptor_bindings(metal_bindings),
            ))
        }
        HalBackend::Noop => Err("Noop backend does not create HAL shader sources".to_owned()),
        _ => Err("unsupported backend does not create HAL shader sources".to_owned()),
    }
}

/// Returns HAL descriptor bindings.
pub(crate) fn hal_descriptor_bindings(
    bindings: &[MetalBufferBinding],
) -> Vec<HalDescriptorBinding> {
    bindings
        .iter()
        .map(|binding| HalDescriptorBinding {
            group: binding.group,
            binding: binding.binding,
            kind: match binding.kind {
                MetalBindingKind::Buffer(BufferBindingType::Uniform) => {
                    HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Uniform)
                }
                MetalBindingKind::Buffer(
                    BufferBindingType::Storage | BufferBindingType::ReadOnlyStorage,
                ) => HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage),
                MetalBindingKind::Texture => HalDescriptorBindingKind::Texture,
                MetalBindingKind::Sampler => HalDescriptorBindingKind::Sampler,
            },
        })
        .collect()
}

/// Returns MSL resource bindings.
pub(crate) fn msl_resource_bindings(
    bindings: &[MetalBufferBinding],
) -> Vec<shader_naga::MslResourceBinding> {
    bindings
        .iter()
        .map(|binding| shader_naga::MslResourceBinding {
            group: binding.group,
            binding: binding.binding,
            metal_index: binding.metal_index,
            kind: match binding.kind {
                MetalBindingKind::Buffer(_) => shader_naga::MslResourceBindingKind::Buffer,
                MetalBindingKind::Texture => shader_naga::MslResourceBindingKind::Texture,
                MetalBindingKind::Sampler => shader_naga::MslResourceBindingKind::Sampler,
            },
        })
        .collect()
}

/// Returns metal buffer binding map.
pub(crate) fn metal_buffer_binding_map(
    layouts: &[Arc<BindGroupLayout>],
) -> Vec<MetalBufferBinding> {
    let mut bindings = Vec::new();
    let mut metal_index = 0u32;
    for (group_index, layout) in layouts.iter().enumerate() {
        let Ok(group) = u32::try_from(group_index) else {
            break;
        };
        for entry in layout.entries() {
            let kind = match entry.kind {
                Some(BindingLayoutKind::Buffer { ty, .. }) => MetalBindingKind::Buffer(ty),
                Some(BindingLayoutKind::Texture { .. }) => MetalBindingKind::Texture,
                Some(BindingLayoutKind::Sampler { .. }) => MetalBindingKind::Sampler,
                _ => continue,
            };
            bindings.push(MetalBufferBinding {
                group,
                binding: entry.binding,
                metal_index,
                kind,
            });
            metal_index = metal_index.saturating_add(1);
        }
    }
    bindings.sort_by_key(|binding| (binding.group, binding.binding));
    for (index, binding) in bindings.iter_mut().enumerate() {
        binding.metal_index = u32::try_from(index).unwrap_or(u32::MAX);
    }
    bindings
}

/// Validates compute pipeline descriptor and returns a descriptive error on failure.
pub(crate) fn validate_compute_pipeline_descriptor(
    descriptor: &ComputePipelineDescriptor,
    limits: Limits,
    features: &FeatureSet,
) -> Option<String> {
    resolve_compute_pipeline_descriptor(descriptor, limits, features).err()
}

/// Records resolve into the command stream.
pub(crate) fn resolve_compute_pipeline_descriptor(
    descriptor: &ComputePipelineDescriptor,
    limits: Limits,
    features: &FeatureSet,
) -> Result<ResolvedPipelineParts, String> {
    if descriptor.shader_module.is_error() {
        return Err("compute pipeline shader module must not be an error module".to_owned());
    }
    #[cfg(feature = "shader-passthrough")]
    if let Some((_, reflection)) = descriptor.shader_module.msl_passthrough() {
        let ComputePipelineLayout::Explicit(layout) = &descriptor.layout else {
            return Err("MSL shader module requires an explicit pipeline layout".to_owned());
        };
        if layout.is_error() {
            return Err("compute pipeline layout must not be an error pipeline layout".to_owned());
        }
        let entry = resolve_msl_entry(
            reflection,
            SHADER_STAGE_COMPUTE,
            descriptor.entry_point.as_deref(),
            "compute",
        )?;
        let workgroup = ResolvedComputeWorkgroup {
            size: entry.workgroup_size,
            storage_size: 0,
        };
        return Ok((
            entry.name.clone(),
            Vec::new(),
            Some(workgroup),
            layout.bind_group_layouts().to_vec(),
        ));
    }
    let Some(module) = descriptor.shader_module.reflected() else {
        return Err("compute pipeline requires a reflected shader module".to_owned());
    };
    let entry_name = resolve_compute_entry(module, descriptor.entry_point.as_deref())?;
    let overrides = module.overrides();
    let constants = resolve_pipeline_constants(&overrides, &descriptor.constants)?;
    let workgroup = resolve_compute_workgroup(module, &entry_name, &constants, limits)?;
    let bindings = module.resource_bindings_for_entry(&entry_name)?;
    validate_compute_pipeline_layout(&descriptor.layout, &bindings)?;
    let bind_group_layouts =
        effective_compute_bind_group_layouts(&descriptor.layout, &bindings, limits, features)?;
    Ok((entry_name, bindings, Some(workgroup), bind_group_layouts))
}

#[cfg(feature = "shader-passthrough")]
pub(crate) fn resolve_msl_entry<'a>(
    reflection: &'a MslReflection,
    stage: u64,
    entry_point: Option<&str>,
    label: &str,
) -> Result<&'a MslEntryPoint, String> {
    let matching = reflection
        .entry_points
        .iter()
        .filter(|entry| entry.stage == stage)
        .collect::<Vec<_>>();
    match entry_point {
        None => match matching.as_slice() {
            [entry] => Ok(*entry),
            [] => Err(format!(
                "{label} pipeline shader module has no matching MSL entry point"
            )),
            _ => Err(format!(
                "{label} pipeline entryPoint is required when multiple matching MSL entries exist"
            )),
        },
        Some(name) => matching
            .into_iter()
            .find(|entry| entry.name == name)
            .ok_or_else(|| {
                format!("{label} pipeline entryPoint must name a matching MSL entry point")
            }),
    }
}

/// Records resolve into the command stream.
pub(crate) fn resolve_compute_entry(
    module: &shader_naga::ReflectedModule,
    entry_point: Option<&str>,
) -> Result<String, String> {
    let entries = module.entry_points();
    let compute_entries = entries
        .iter()
        .filter(|entry| entry.stage == shader_naga::ReflectedShaderStage::Compute)
        .collect::<Vec<_>>();

    match entry_point {
        None => match compute_entries.as_slice() {
            [entry] => Ok(entry.name.clone()),
            [] => Err("compute pipeline shader module has no compute entry point".to_owned()),
            _ => Err(
                "compute pipeline entryPoint is required when multiple compute entries exist"
                    .to_owned(),
            ),
        },
        Some(name) => {
            if compute_entries.iter().any(|entry| entry.name == name) {
                Ok(name.to_owned())
            } else {
                Err("compute pipeline entryPoint must name a compute entry point".to_owned())
            }
        }
    }
}

/// Stores resolved override constant data used by validation and backend submission.
#[derive(Debug, Clone)]
pub(crate) struct ResolvedOverrideConstant {
    pub(crate) index: usize,
    pub(crate) key: String,
    pub(crate) value: f64,
}

/// Records resolve into the command stream.
pub(crate) fn resolve_pipeline_constants(
    overrides: &[shader_naga::ReflectedOverride],
    constants: &[PipelineConstant],
) -> Result<Vec<ResolvedOverrideConstant>, String> {
    let mut seen_keys = BTreeSet::new();
    let mut resolved = Vec::new();

    for constant in constants {
        if !seen_keys.insert(constant.key.as_str()) {
            return Err("pipeline constant keys must be unique".to_owned());
        }
        let index = resolve_pipeline_constant_key(overrides, &constant.key)?;
        validate_pipeline_constant_value(&overrides[index], constant.value)?;
        resolved.push(ResolvedOverrideConstant {
            index,
            key: constant.key.clone(),
            value: constant.value,
        });
    }

    for (index, override_) in overrides.iter().enumerate() {
        if !override_.has_default && !resolved.iter().any(|constant| constant.index == index) {
            return Err("pipeline constant is required for override without a default".to_owned());
        }
    }

    Ok(resolved)
}

/// Records resolve into the command stream.
pub(crate) fn resolve_pipeline_constant_key(
    overrides: &[shader_naga::ReflectedOverride],
    key: &str,
) -> Result<usize, String> {
    if let Ok(id) = key.parse::<u16>() {
        return overrides
            .iter()
            .position(|override_| override_.id == Some(id))
            .ok_or_else(|| "pipeline constant key does not match a shader override".to_owned());
    }

    if overrides
        .iter()
        .any(|override_| override_.id.is_some() && override_.name.as_deref() == Some(key))
    {
        return Err("pipeline constant key must use numeric id for @id overrides".to_owned());
    }

    overrides
        .iter()
        .position(|override_| override_.id.is_none() && override_.name.as_deref() == Some(key))
        .ok_or_else(|| "pipeline constant key does not match a shader override".to_owned())
}

/// Validates pipeline constant value and returns a descriptive error on failure.
pub(crate) fn validate_pipeline_constant_value(
    override_: &shader_naga::ReflectedOverride,
    value: f64,
) -> Result<(), String> {
    if !value.is_finite() {
        return Err("pipeline constant value must be finite".to_owned());
    }
    if override_.ty.components != 1 {
        return Err("pipeline override constants must be scalar".to_owned());
    }

    match override_.ty.scalar {
        shader_naga::ReflectedTypeScalarClass::Float => {
            let max = if override_.ty.width == 2 {
                65_504.0
            } else {
                f64::from(f32::MAX)
            };
            if value.abs() > max {
                return Err("pipeline constant value is outside the override type range".to_owned());
            }
        }
        shader_naga::ReflectedTypeScalarClass::Sint => {
            if value.fract() != 0.0 || value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
                return Err("pipeline constant value is outside the override type range".to_owned());
            }
        }
        shader_naga::ReflectedTypeScalarClass::Uint => {
            if value.fract() != 0.0 || value < 0.0 || value > f64::from(u32::MAX) {
                return Err("pipeline constant value is outside the override type range".to_owned());
            }
        }
        shader_naga::ReflectedTypeScalarClass::Bool => {}
    }
    Ok(())
}

/// Records resolve into the command stream.
pub(crate) fn resolve_compute_workgroup(
    module: &shader_naga::ReflectedModule,
    entry_name: &str,
    constants: &[ResolvedOverrideConstant],
    limits: Limits,
) -> Result<ResolvedComputeWorkgroup, String> {
    let pipeline_constants = constants
        .iter()
        .map(|constant| (constant.key.clone(), constant.value))
        .collect();
    let workgroup = module.resolved_compute_workgroup_size(entry_name, &pipeline_constants)?;
    let size = workgroup.literal_size;

    if size.contains(&0) {
        return Err("compute workgroup size must be at least one".to_owned());
    }
    if size[0] > limits.max_compute_workgroup_size_x {
        return Err("compute workgroup x size exceeds the device limit".to_owned());
    }
    if size[1] > limits.max_compute_workgroup_size_y {
        return Err("compute workgroup y size exceeds the device limit".to_owned());
    }
    if size[2] > limits.max_compute_workgroup_size_z {
        return Err("compute workgroup z size exceeds the device limit".to_owned());
    }
    let invocations = size[0]
        .checked_mul(size[1])
        .and_then(|xy| xy.checked_mul(size[2]))
        .ok_or_else(|| "compute workgroup invocation count overflows".to_owned())?;
    if invocations > limits.max_compute_invocations_per_workgroup {
        return Err("compute workgroup invocation count exceeds the device limit".to_owned());
    }
    if workgroup.workgroup_storage_size > u64::from(limits.max_compute_workgroup_storage_size) {
        return Err("compute workgroup storage size exceeds the device limit".to_owned());
    }

    Ok(ResolvedComputeWorkgroup {
        size,
        storage_size: workgroup.workgroup_storage_size,
    })
}

/// Validates compute pipeline layout and returns a descriptive error on failure.
pub(crate) fn validate_compute_pipeline_layout(
    layout: &ComputePipelineLayout,
    bindings: &[shader_naga::ReflectedResourceBinding],
) -> Result<(), String> {
    let ComputePipelineLayout::Explicit(layout) = layout else {
        return Ok(());
    };
    if layout.is_error() {
        return Err("compute pipeline layout must not be an error pipeline layout".to_owned());
    }
    let requirements = bindings
        .iter()
        .cloned()
        .map(|binding| StageResourceBinding {
            stage: PipelineShaderStage::Compute,
            binding,
        })
        .collect::<Vec<_>>();
    validate_pipeline_layout_stage_bindings(layout, &requirements)
}

/// Returns effective compute bind group layouts.
pub(crate) fn effective_compute_bind_group_layouts(
    layout: &ComputePipelineLayout,
    bindings: &[shader_naga::ReflectedResourceBinding],
    limits: Limits,
    features: &FeatureSet,
) -> Result<Vec<Arc<BindGroupLayout>>, String> {
    match layout {
        ComputePipelineLayout::Explicit(layout) => Ok(layout.bind_group_layouts().to_vec()),
        ComputePipelineLayout::Auto => derive_bind_group_layouts(
            bindings
                .iter()
                .cloned()
                .map(|binding| StageResourceBinding {
                    stage: PipelineShaderStage::Compute,
                    binding,
                }),
            limits,
            features,
        ),
    }
}

/// Stores binding metadata.
#[derive(Debug, Clone)]
pub(crate) struct StageResourceBinding {
    pub(crate) stage: PipelineShaderStage,
    pub(crate) binding: shader_naga::ReflectedResourceBinding,
}

/// Enumerates pipeline shader stage values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PipelineShaderStage {
    /// Vertex variant.
    Vertex,
    /// Fragment variant.
    Fragment,
    /// Compute variant.
    Compute,
}

/// Validates pipeline layout stage bindings and returns a descriptive error on failure.
pub(crate) fn validate_pipeline_layout_stage_bindings(
    layout: &PipelineLayout,
    requirements: &[StageResourceBinding],
) -> Result<(), String> {
    for requirement in requirements {
        let binding = &requirement.binding;
        if !binding.statically_used {
            continue;
        }
        let group = usize::try_from(binding.group)
            .map_err(|_| "shader binding group index is too large".to_owned())?;
        let Some(group_layout) = layout.bind_group_layouts().get(group) else {
            return Err("pipeline layout is missing a shader bind group".to_owned());
        };
        let Some(layout_entry) = group_layout
            .entries()
            .iter()
            .find(|entry| entry.binding == binding.binding)
        else {
            return Err("pipeline layout is missing a shader binding".to_owned());
        };
        if layout_entry.visibility & pipeline_stage_visibility_bit(requirement.stage) == 0 {
            return Err(
                "pipeline layout binding visibility does not include the shader stage".to_owned(),
            );
        }
        let Some(kind) = layout_entry.kind else {
            return Err("pipeline layout binding must be valid".to_owned());
        };
        validate_shader_binding_compat(binding, kind)?;
    }
    validate_non_filterable_gather_bindings(layout, requirements)?;

    Ok(())
}

fn validate_non_filterable_gather_bindings(
    layout: &PipelineLayout,
    requirements: &[StageResourceBinding],
) -> Result<(), String> {
    for requirement in requirements {
        let shader_naga::ReflectedResourceBindingKind::Texture {
            sampled,
            sample_kind,
            sample_usage: shader_naga::ReflectedTextureSampleUsage::Gather,
            ..
        } = requirement.binding.kind
        else {
            continue;
        };
        if reflected_texture_sample_type(
            sampled,
            sample_kind,
            shader_naga::ReflectedTextureSampleUsage::Gather,
        )? == TextureSampleType::Float
        {
            continue;
        }
        let visibility = pipeline_stage_visibility_bit(requirement.stage);
        if layout.bind_group_layouts().iter().any(|group| {
            group.entries().iter().any(|entry| {
                entry.visibility & visibility != 0
                    && matches!(
                        entry.kind,
                        Some(BindingLayoutKind::Sampler {
                            ty: SamplerBindingType::Filtering
                        })
                    )
            })
        }) {
            return Err(
                "textureGather with a filtering sampler requires a filterable texture binding"
                    .to_owned(),
            );
        }
    }
    Ok(())
}

/// Returns derive bind group layouts.
pub(crate) fn derive_bind_group_layouts<I>(
    requirements: I,
    limits: Limits,
    features: &FeatureSet,
) -> Result<Vec<Arc<BindGroupLayout>>, String>
where
    I: IntoIterator<Item = StageResourceBinding>,
{
    let mut groups = BTreeMap::<u32, BTreeMap<u32, BindGroupLayoutEntry>>::new();
    for requirement in requirements {
        let binding = requirement.binding;
        if !binding.statically_used {
            continue;
        }
        let group = groups.entry(binding.group).or_default();
        let visibility = pipeline_stage_visibility_bit(requirement.stage);
        let derived = reflected_bind_group_layout_entry(&binding, visibility)?;
        match group.get_mut(&binding.binding) {
            Some(existing) => merge_bind_group_layout_entry(existing, derived)?,
            None => {
                group.insert(binding.binding, derived);
            }
        }
    }

    let Some(max_group) = groups.keys().next_back().copied() else {
        return Ok(Vec::new());
    };
    let group_count = usize::try_from(max_group)
        .ok()
        .and_then(|group| group.checked_add(1))
        .ok_or_else(|| "pipeline bind group index is too large".to_owned())?;
    if group_count > limits.max_bind_groups as usize {
        return Err("pipeline auto layout bind group count exceeds the device limit".to_owned());
    }

    let mut layouts = Vec::with_capacity(group_count);
    for group_index in 0..=max_group {
        let entries = groups
            .remove(&group_index)
            .map(|entries| entries.into_values().collect::<Vec<_>>())
            .unwrap_or_default();
        if let Some(message) = crate::bind_group_layout::validate_bind_group_layout_descriptor(
            &entries, limits, features,
        ) {
            return Err(message);
        }
        layouts.push(Arc::new(BindGroupLayout::new(entries, false, true)));
    }
    Ok(layouts)
}

/// Returns reflected bind group layout entry.
pub(crate) fn reflected_bind_group_layout_entry(
    binding: &shader_naga::ReflectedResourceBinding,
    visibility: u64,
) -> Result<BindGroupLayoutEntry, String> {
    Ok(BindGroupLayoutEntry {
        binding: binding.binding,
        visibility,
        binding_array_size: 0,
        kind: Some(reflected_binding_layout_kind(binding)?),
    })
}

/// Returns reflected binding layout kind.
pub(crate) fn reflected_binding_layout_kind(
    binding: &shader_naga::ReflectedResourceBinding,
) -> Result<BindingLayoutKind, String> {
    match &binding.kind {
        shader_naga::ReflectedResourceBindingKind::Buffer(ty) => Ok(BindingLayoutKind::Buffer {
            ty: match ty {
                shader_naga::ReflectedBufferType::Uniform => BufferBindingType::Uniform,
                shader_naga::ReflectedBufferType::Storage => BufferBindingType::Storage,
                shader_naga::ReflectedBufferType::ReadOnlyStorage => {
                    BufferBindingType::ReadOnlyStorage
                }
            },
            has_dynamic_offset: false,
            min_binding_size: binding.min_binding_size,
        }),
        shader_naga::ReflectedResourceBindingKind::Sampler { comparison } => {
            Ok(BindingLayoutKind::Sampler {
                ty: if *comparison {
                    SamplerBindingType::Comparison
                } else {
                    SamplerBindingType::Filtering
                },
            })
        }
        shader_naga::ReflectedResourceBindingKind::Texture {
            sampled,
            sample_kind,
            sample_usage,
            view_dimension,
            multisampled,
        } => Ok(BindingLayoutKind::Texture {
            sample_type: reflected_texture_sample_type(*sampled, *sample_kind, *sample_usage)?,
            view_dimension: reflected_texture_view_dimension(*view_dimension),
            multisampled: *multisampled,
        }),
        #[cfg(feature = "tiled")]
        shader_naga::ReflectedResourceBindingKind::InputAttachment {
            sample_kind,
            multisampled,
        } => Ok(BindingLayoutKind::InputAttachment {
            sample_type: reflected_input_attachment_sample_type(*sample_kind)?,
            multisampled: *multisampled,
        }),
        shader_naga::ReflectedResourceBindingKind::StorageTexture {
            format,
            access,
            view_dimension,
        } => Ok(BindingLayoutKind::StorageTexture {
            access: reflected_storage_texture_access(access),
            format: reflected_storage_texture_format(format)?,
            view_dimension: reflected_texture_view_dimension(*view_dimension),
        }),
    }
}

/// Returns reflected texture sample type.
pub(crate) fn reflected_texture_sample_type(
    sampled: bool,
    sample_kind: Option<shader_naga::ReflectedTypeScalarClass>,
    sample_usage: shader_naga::ReflectedTextureSampleUsage,
) -> Result<TextureSampleType, String> {
    if !sampled {
        return Ok(TextureSampleType::Depth);
    }
    match sample_kind {
        Some(shader_naga::ReflectedTypeScalarClass::Float) => Ok(match sample_usage {
            shader_naga::ReflectedTextureSampleUsage::Sample
            | shader_naga::ReflectedTextureSampleUsage::Gather => TextureSampleType::Float,
            shader_naga::ReflectedTextureSampleUsage::Load => TextureSampleType::UnfilterableFloat,
        }),
        Some(shader_naga::ReflectedTypeScalarClass::Sint) => Ok(TextureSampleType::Sint),
        Some(shader_naga::ReflectedTypeScalarClass::Uint) => Ok(TextureSampleType::Uint),
        _ => Err("pipeline texture binding sample type is unsupported".to_owned()),
    }
}

/// Returns reflected input attachment sample type.
#[cfg(feature = "tiled")]
pub(crate) fn reflected_input_attachment_sample_type(
    sample_kind: shader_naga::ReflectedTypeScalarClass,
) -> Result<TextureSampleType, String> {
    match sample_kind {
        shader_naga::ReflectedTypeScalarClass::Float => Ok(TextureSampleType::Float),
        shader_naga::ReflectedTypeScalarClass::Sint => Ok(TextureSampleType::Sint),
        shader_naga::ReflectedTypeScalarClass::Uint => Ok(TextureSampleType::Uint),
        _ => Err("pipeline input attachment sample type is unsupported".to_owned()),
    }
}

/// Returns reflected texture view dimension.
pub(crate) fn reflected_texture_view_dimension(
    dimension: shader_naga::ReflectedTextureViewDimension,
) -> TextureViewDimension {
    match dimension {
        shader_naga::ReflectedTextureViewDimension::D1 => TextureViewDimension::D1,
        shader_naga::ReflectedTextureViewDimension::D2 => TextureViewDimension::D2,
        shader_naga::ReflectedTextureViewDimension::D2Array => TextureViewDimension::D2Array,
        shader_naga::ReflectedTextureViewDimension::Cube => TextureViewDimension::Cube,
        shader_naga::ReflectedTextureViewDimension::CubeArray => TextureViewDimension::CubeArray,
        shader_naga::ReflectedTextureViewDimension::D3 => TextureViewDimension::D3,
    }
}

/// Returns reflected storage texture access.
pub(crate) fn reflected_storage_texture_access(
    access: &shader_naga::ReflectedStorageTextureAccess,
) -> StorageTextureAccess {
    match (access.read, access.write) {
        (true, true) => StorageTextureAccess::ReadWrite,
        (true, false) => StorageTextureAccess::ReadOnly,
        _ => StorageTextureAccess::WriteOnly,
    }
}

/// Returns reflected storage texture format.
pub(crate) fn reflected_storage_texture_format(format: &str) -> Result<TextureFormat, String> {
    let raw = match format {
        "Rgba8Unorm" => 0x0000_0016,
        "Rgba8Snorm" => 0x0000_0018,
        "Rgba8Uint" => 0x0000_0019,
        "Rgba8Sint" => 0x0000_001A,
        "Rgba16Uint" => 0x0000_0026,
        "Rgba16Sint" => 0x0000_0027,
        "Rgba16Float" => 0x0000_0028,
        "R32Uint" => 0x0000_000F,
        "R32Sint" => 0x0000_0010,
        "R32Float" => 0x0000_000E,
        "Rg32Uint" => 0x0000_0022,
        "Rg32Sint" => 0x0000_0023,
        "Rg32Float" => 0x0000_0021,
        "Rgba32Uint" => 0x0000_002A,
        "Rgba32Sint" => 0x0000_002B,
        "Rgba32Float" => 0x0000_0029,
        _ => return Err("pipeline auto layout storage texture format is unsupported".to_owned()),
    };
    Ok(TextureFormat::from_raw(raw))
}

/// Returns merge bind group layout entry.
pub(crate) fn merge_bind_group_layout_entry(
    existing: &mut BindGroupLayoutEntry,
    incoming: BindGroupLayoutEntry,
) -> Result<(), String> {
    existing.visibility |= incoming.visibility;
    match (&mut existing.kind, incoming.kind) {
        (
            Some(BindingLayoutKind::Buffer {
                ty,
                min_binding_size,
                ..
            }),
            Some(BindingLayoutKind::Buffer {
                ty: incoming_ty,
                min_binding_size: incoming_min_binding_size,
                ..
            }),
        ) if *ty == incoming_ty => {
            *min_binding_size = (*min_binding_size).max(incoming_min_binding_size);
            Ok(())
        }
        (
            Some(BindingLayoutKind::Texture { sample_type, .. }),
            Some(BindingLayoutKind::Texture {
                sample_type: incoming_sample_type,
                ..
            }),
        ) if *sample_type == incoming_sample_type
            || matches!(
                (*sample_type, incoming_sample_type),
                (
                    TextureSampleType::Float,
                    TextureSampleType::UnfilterableFloat
                ) | (
                    TextureSampleType::UnfilterableFloat,
                    TextureSampleType::Float
                )
            ) =>
        {
            if incoming_sample_type == TextureSampleType::Float {
                *sample_type = TextureSampleType::Float;
            }
            Ok(())
        }
        (Some(existing_kind), Some(incoming_kind)) if *existing_kind == incoming_kind => Ok(()),
        _ => Err("pipeline auto layout has incompatible shader bindings".to_owned()),
    }
}

/// Returns pipeline stage visibility bit.
pub(crate) fn pipeline_stage_visibility_bit(stage: PipelineShaderStage) -> u64 {
    match stage {
        PipelineShaderStage::Vertex => 1,
        PipelineShaderStage::Fragment => 2,
        PipelineShaderStage::Compute => 4,
    }
}

/// Validates shader binding compat and returns a descriptive error on failure.
pub(crate) fn validate_shader_binding_compat(
    binding: &shader_naga::ReflectedResourceBinding,
    layout_kind: BindingLayoutKind,
) -> Result<(), String> {
    match (&binding.kind, layout_kind) {
        (
            shader_naga::ReflectedResourceBindingKind::Buffer(shader_ty),
            BindingLayoutKind::Buffer {
                ty,
                min_binding_size,
                ..
            },
        ) => {
            if !buffer_binding_types_compatible(*shader_ty, ty) {
                return Err(
                    "compute pipeline layout buffer binding type is incompatible".to_owned(),
                );
            }
            if min_binding_size != 0 && min_binding_size < binding.min_binding_size {
                return Err("compute pipeline layout buffer minBindingSize is too small".to_owned());
            }
            Ok(())
        }
        (
            shader_naga::ReflectedResourceBindingKind::Sampler { .. },
            BindingLayoutKind::Sampler { .. },
        )
        | (
            shader_naga::ReflectedResourceBindingKind::Texture { .. },
            BindingLayoutKind::Texture { .. },
        )
        | (
            shader_naga::ReflectedResourceBindingKind::StorageTexture { .. },
            BindingLayoutKind::StorageTexture { .. },
        ) => {
            let expected = reflected_binding_layout_kind(binding)?;
            if shader_binding_layout_kinds_compatible(expected, layout_kind) {
                Ok(())
            } else {
                Err(
                    "pipeline layout binding kind is incompatible with the shader binding"
                        .to_owned(),
                )
            }
        }
        #[cfg(feature = "tiled")]
        (
            shader_naga::ReflectedResourceBindingKind::InputAttachment { .. },
            BindingLayoutKind::InputAttachment { .. },
        ) => {
            let expected = reflected_binding_layout_kind(binding)?;
            if shader_binding_layout_kinds_compatible(expected, layout_kind) {
                Ok(())
            } else {
                Err(
                    "pipeline layout binding kind is incompatible with the shader binding"
                        .to_owned(),
                )
            }
        }
        _ => Err("compute pipeline layout binding type is incompatible".to_owned()),
    }
}

/// Returns shader binding layout kinds compatible.
pub(crate) fn shader_binding_layout_kinds_compatible(
    expected: BindingLayoutKind,
    actual: BindingLayoutKind,
) -> bool {
    match (expected, actual) {
        (
            BindingLayoutKind::Sampler { ty: expected },
            BindingLayoutKind::Sampler { ty: actual },
        ) => expected == actual,
        (
            BindingLayoutKind::Texture {
                sample_type,
                view_dimension,
                multisampled,
            },
            BindingLayoutKind::Texture {
                sample_type: actual_sample_type,
                view_dimension: actual_view_dimension,
                multisampled: actual_multisampled,
            },
        ) => {
            sample_type == actual_sample_type
                && view_dimension == actual_view_dimension
                && multisampled == actual_multisampled
        }
        (
            BindingLayoutKind::StorageTexture {
                access,
                format,
                view_dimension,
            },
            BindingLayoutKind::StorageTexture {
                access: actual_access,
                format: actual_format,
                view_dimension: actual_view_dimension,
            },
        ) => {
            access == actual_access
                && format == actual_format
                && view_dimension == actual_view_dimension
        }
        #[cfg(feature = "tiled")]
        (
            BindingLayoutKind::InputAttachment {
                sample_type,
                multisampled,
            },
            BindingLayoutKind::InputAttachment {
                sample_type: actual_sample_type,
                multisampled: actual_multisampled,
            },
        ) => sample_type == actual_sample_type && multisampled == actual_multisampled,
        _ => false,
    }
}

/// Returns buffer binding types compatible.
pub(crate) fn buffer_binding_types_compatible(
    shader_ty: shader_naga::ReflectedBufferType,
    layout_ty: BufferBindingType,
) -> bool {
    matches!(
        (shader_ty, layout_ty),
        (
            shader_naga::ReflectedBufferType::Uniform,
            BufferBindingType::Uniform
        ) | (
            shader_naga::ReflectedBufferType::Storage,
            BufferBindingType::Storage
        ) | (
            shader_naga::ReflectedBufferType::ReadOnlyStorage,
            BufferBindingType::ReadOnlyStorage
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;

    use std::sync::Arc;

    #[test]
    fn compute_pipeline_accessors_and_render_pipeline_accessors() {
        let device = noop_device();
        let compute = noop_compute_pipeline(&device);
        let render = noop_render_pipeline(&device);

        assert!(!compute.is_error());
        assert_eq!(compute.entry_name(), "cs");
        assert!(compute.bind_group_layouts().is_empty());
        assert!(!render.is_error());
        assert_eq!(render.vertex_entry_name(), "vs");
        assert_eq!(render.fragment_entry_name(), Some("fs"));
        assert!(render.bind_group_layouts().is_empty());

        let bad_shader = Arc::new(
            device.create_shader_module(ShaderModuleSource::Invalid("bad shader".to_owned())),
        );
        device.push_error_scope(ErrorFilter::Validation);
        let bad_compute = device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Auto,
            shader_module: bad_shader,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        });
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid compute pipeline should be scoped");
        assert!(bad_compute.is_error());
        assert_eq!(
            scoped.message,
            "compute pipeline shader module must not be an error module"
        );
    }

    #[test]
    fn shader_binding_compat_defers_unspecified_min_binding_size() {
        let binding = shader_naga::ReflectedResourceBinding {
            group: 0,
            binding: 0,
            kind: shader_naga::ReflectedResourceBindingKind::Buffer(
                shader_naga::ReflectedBufferType::Uniform,
            ),
            min_binding_size: 4,
            statically_used: true,
        };

        let layout_kind = |min_binding_size| BindingLayoutKind::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size,
        };

        assert_eq!(
            validate_shader_binding_compat(&binding, layout_kind(0)),
            Ok(())
        );
        assert_eq!(
            validate_shader_binding_compat(&binding, layout_kind(8)),
            Ok(())
        );
        assert_eq!(
            validate_shader_binding_compat(&binding, layout_kind(2)),
            Err("compute pipeline layout buffer minBindingSize is too small".to_owned())
        );
    }

    #[cfg(feature = "shader-passthrough")]
    fn spirv_words(source: &str, entry_point: &str, stage: naga::ShaderStage) -> Vec<u32> {
        shader_naga::parse_and_validate_wgsl(source)
            .expect("test WGSL should validate")
            .generate_spirv(entry_point, stage)
            .expect("test WGSL should generate SPIR-V")
    }

    #[cfg(feature = "shader-passthrough")]
    fn msl_compute_reflection() -> MslReflection {
        MslReflection {
            entry_points: vec![MslEntryPoint {
                name: "cs".to_owned(),
                stage: SHADER_STAGE_COMPUTE,
                workgroup_size: [2, 3, 4],
            }],
        }
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn select_compute_shader_source_covers_passthrough_backend_matrix() {
        let device = noop_device();
        let wgsl = device.create_shader_module(ShaderModuleSource::Wgsl(
            "@compute @workgroup_size(1) fn cs() {}".to_owned(),
        ));
        let words = spirv_words(
            "@compute @workgroup_size(1) fn cs() {}",
            "cs",
            naga::ShaderStage::Compute,
        );
        let spirv = device.create_shader_module_spirv(words.clone());
        let msl_source = "kernel void cs() {}".to_owned();
        let msl = device.create_shader_module_msl(msl_source.clone(), msl_compute_reflection());

        let (source, entry, bindings) =
            select_compute_shader_source(HalBackend::Vulkan, &wgsl, "cs", &[])
                .expect("WGSL should generate Vulkan SPIR-V");
        assert!(matches!(source, HalShaderSource::SpirV(words) if !words.is_empty()));
        assert_eq!(entry, "cs");
        assert!(bindings.is_empty());

        let (source, entry, _) =
            select_compute_shader_source(HalBackend::Vulkan, &spirv, "cs", &[])
                .expect("SPIR-V passthrough should select Vulkan SPIR-V");
        assert!(matches!(source, HalShaderSource::SpirV(selected) if selected == words));
        assert_eq!(entry, "cs");

        let (source, entry, _) = select_compute_shader_source(HalBackend::Metal, &msl, "cs", &[])
            .expect("MSL passthrough should select Metal MSL");
        assert!(matches!(source, HalShaderSource::Msl(selected) if selected == msl_source));
        assert_eq!(entry, "cs");

        assert_eq!(
            select_compute_shader_source(HalBackend::Metal, &spirv, "cs", &[])
                .expect_err("SPIR-V must not run on Metal"),
            "SPIR-V shader module cannot be used on the Metal backend"
        );
        assert_eq!(
            select_compute_shader_source(HalBackend::Vulkan, &msl, "cs", &[])
                .expect_err("MSL must not run on Vulkan"),
            "MSL shader module cannot be used on the Vulkan backend"
        );
    }

    #[cfg(feature = "gles")]
    #[test]
    fn select_compute_shader_source_generates_gles_glsl() {
        let device = noop_device();
        let wgsl = device.create_shader_module(ShaderModuleSource::Wgsl(
            "@compute @workgroup_size(2, 1, 1) fn cs() {}".to_owned(),
        ));

        let (source, entry, bindings) =
            select_compute_shader_source(HalBackend::Gles, &wgsl, "cs", &[])
                .expect("WGSL should generate GLES GLSL");

        let HalShaderSource::Glsl {
            source,
            stage: yawgpu_hal::HalShaderStage::Compute,
        } = source
        else {
            panic!("GLES should select compute GLSL");
        };
        assert_eq!(entry, "cs");
        assert!(bindings.is_empty());
        assert!(source.contains("#version 310 es"));
        assert!(source.contains("local_size_x = 2"));
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn spirv_compute_pipeline_auto_layout_resolves_on_noop() {
        let device = noop_device();
        let words = spirv_words(
            "@compute @workgroup_size(2, 3, 4) fn cs() {}",
            "cs",
            naga::ShaderStage::Compute,
        );
        let module = Arc::new(device.create_shader_module_spirv(words));

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Auto,
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        });
        let scoped = device.pop_error_scope().expect("scope should exist");

        assert!(!pipeline.is_error());
        assert_eq!(pipeline.entry_name(), "cs");
        assert!(pipeline.bind_group_layouts().is_empty());
        assert_eq!(scoped, None);
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_compute_pipeline_requires_explicit_layout_on_noop() {
        let device = noop_device();
        let module =
            Arc::new(device.create_shader_module_msl(
                "kernel void cs() {}".to_owned(),
                msl_compute_reflection(),
            ));

        device.push_error_scope(ErrorFilter::Validation);
        let auto = device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Auto,
            shader_module: Arc::clone(&module),
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        });
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("auto MSL pipeline should be scoped");
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
        device.push_error_scope(ErrorFilter::Validation);
        let explicit = device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(explicit_layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        });
        let scoped = device.pop_error_scope().expect("scope should exist");
        assert!(!explicit.is_error());
        assert_eq!(explicit.entry_name(), "cs");
        assert_eq!(scoped, None);
    }

    #[test]
    fn resolve_compute_workgroup_evaluates_override_expressions() {
        let module = shader_naga::parse_and_validate_wgsl(
            "override n: u32 = 4u;
             var<workgroup> scratch: array<u32, n * 2u>;
             @compute @workgroup_size(n + 1u, 1, 1)
             fn cs() { scratch[0] = 1u; }",
        )
        .expect("test WGSL should validate");
        let overrides = module.overrides();
        let constants = resolve_pipeline_constants(
            &overrides,
            &[PipelineConstant {
                key: "n".to_owned(),
                value: 8.0,
            }],
        )
        .expect("override should resolve");

        let resolved = resolve_compute_workgroup(&module, "cs", &constants, Limits::DEFAULT)
            .expect("override expressions should evaluate");

        assert_eq!(resolved.size, [9, 1, 1]);
        assert_eq!(resolved.storage_size, 64);
    }

    #[test]
    fn resolve_compute_workgroup_rejects_override_arithmetic_error() {
        let module = shader_naga::parse_and_validate_wgsl(
            "override n: u32;
             @compute @workgroup_size(1u / n, 1, 1)
             fn cs() {}",
        )
        .expect("override arithmetic is pipeline-time validation");
        let constants = resolve_pipeline_constants(
            &module.overrides(),
            &[PipelineConstant {
                key: "n".to_owned(),
                value: 0.0,
            }],
        )
        .expect("override constant should resolve");

        assert!(resolve_compute_workgroup(&module, "cs", &constants, Limits::DEFAULT).is_err());
    }
}
