//! Tint shader frontend skeleton for the feature-selected frontend facade.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::shader::{CompilationMessage, CompilationSeverity};
pub(crate) use crate::shader_types::*;

const NOT_IMPLEMENTED: &str = "shader_tint: not yet implemented (P2b/P2c)";

/// Stores reflected shader module data used by validation and backend submission.
#[derive(Debug)]
pub struct ReflectedModule {
    /// Tint program.
    pub program: yawgpu_tint::Program,
    /// Non-fatal compilation warnings.
    pub(crate) warnings: Vec<CompilationMessage>,
}

// SAFETY: Phase 2a does not call into the Tint program after parsing; the handle
// is stored only to preserve the future frontend shape. Later Tint slices must
// revisit this when reflection and code generation start using the handle.
unsafe impl Send for ReflectedModule {}

// SAFETY: See the `Send` impl above.
unsafe impl Sync for ReflectedModule {}

/// Returns parse and validate wgsl.
pub(crate) fn parse_and_validate_wgsl(src: &str) -> Result<ReflectedModule, String> {
    parse_and_validate_wgsl_gated(src, true)
}

/// Returns parse and validate wgsl using the supplied `shader-f16` gate.
pub(crate) fn parse_and_validate_wgsl_gated(
    src: &str,
    shader_f16: bool,
) -> Result<ReflectedModule, String> {
    let program = yawgpu_tint::Program::parse(src, shader_f16)?;
    let warnings = program
        .diagnostics()?
        .into_iter()
        .filter(|diagnostic| diagnostic.severity == yawgpu_tint::DiagnosticSeverity::Warning)
        .map(|diagnostic| CompilationMessage {
            severity: CompilationSeverity::Warning,
            message: diagnostic.message,
            line_num: 0,
            line_pos: 0,
            offset: 0,
            length: 0,
        })
        .collect();
    Ok(ReflectedModule { program, warnings })
}

impl ReflectedModule {
    /// Generates spirv for the validated shader module.
    pub(crate) fn generate_spirv(
        &self,
        entry_name: &str,
        _stage: ShaderStage,
        pipeline_constants: &PipelineConstants,
        unchecked_buffer_bounds: bool,
    ) -> Result<Vec<u32>, String> {
        self.program.generate_spirv(
            entry_name,
            &yawgpu_tint::Bindings::default(),
            &override_values(pipeline_constants),
            !unchecked_buffer_bounds,
        )
    }

    /// Generates GLSL ES for the validated shader module.
    #[cfg(feature = "gles")]
    pub(crate) fn generate_glsl(
        &self,
        entry_name: &str,
        _stage: ShaderStage,
        pipeline_constants: &PipelineConstants,
    ) -> Result<GeneratedGlsl, String> {
        let source = self.program.generate_glsl(
            entry_name,
            &yawgpu_tint::Bindings::default(),
            &override_values(pipeline_constants),
        )?;
        Ok(GeneratedGlsl {
            source,
            entry_point: entry_name.to_owned(),
        })
    }

    /// Generates msl for the validated shader module.
    pub(crate) fn generate_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        pipeline_constants: &PipelineConstants,
    ) -> Result<GeneratedMsl, String> {
        self.generate_stage_msl(entry_name, binding_map, pipeline_constants, true)
    }

    /// Generates render vertex MSL for a validated shader module.
    pub(crate) fn generate_render_vertex_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        vertex_buffers: &[MslVertexBufferBinding],
        force_point_size: bool,
        pipeline_constants: &PipelineConstants,
    ) -> Result<GeneratedMsl, String> {
        if !vertex_buffers.is_empty() || force_point_size {
            // TODO(phase-3): vertex layout / point size handled by the Metal HAL
            // under Tint's stage_in model.
        }
        self.generate_stage_msl(entry_name, binding_map, pipeline_constants, false)
    }

    /// Generates render fragment MSL for a validated shader module.
    pub(crate) fn generate_render_fragment_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        pipeline_constants: &PipelineConstants,
        sample_mask: u32,
    ) -> Result<GeneratedMsl, String> {
        if sample_mask != u32::MAX {
            // TODO(phase-3): sample mask handled by Tint's MSL writer and the
            // Metal pipeline state.
        }
        self.generate_stage_msl(entry_name, binding_map, pipeline_constants, false)
    }

    /// Generates render msl for the validated shader module.
    pub(crate) fn generate_render_msl(
        &self,
        _vertex_entry_name: &str,
        _fragment_entry_name: Option<&str>,
        _binding_map: &MslBindingMap,
        _vertex_buffers: &[MslVertexBufferBinding],
        _force_point_size: bool,
    ) -> Result<GeneratedRenderMsl, String> {
        // TODO(P2c.3): combined same-module render under Tint emits per-stage;
        // the Metal HAL should consume per-stage (MslStagesWithBufferSizes).
        Err(NOT_IMPLEMENTED.to_owned())
    }

    fn generate_stage_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        pipeline_constants: &PipelineConstants,
        disable_robustness: bool,
    ) -> Result<GeneratedMsl, String> {
        let bindings =
            tint_bindings_for_msl(binding_map, &self.resource_bindings_for_entry(entry_name)?)?;
        let output = self.program.generate_msl(
            entry_name,
            &bindings,
            &override_values(pipeline_constants),
            disable_robustness,
        )?;
        let buffer_size_bindings = if output.needs_storage_buffer_sizes {
            self.msl_buffer_size_bindings_for_entry(entry_name)?
        } else {
            Vec::new()
        };
        let buffer_sizes_slot = msl_buffer_sizes_slot(binding_map, &buffer_size_bindings)?;
        Ok(GeneratedMsl {
            source: output.source,
            entry_point: entry_name.to_owned(),
            buffer_sizes_slot,
            buffer_size_bindings,
            frag_depth_clamp_slot: None,
            // The current shim exposes the total workgroup size but not each
            // workgroup variable's byte size, so compute encoders receive an
            // empty per-argument allocation list until that metadata is exposed.
            workgroup_memory_sizes: Vec::new(),
        })
    }

    /// Returns entry points reflected by the validated shader module.
    pub(crate) fn entry_points(&self) -> Vec<ReflectedEntryPoint> {
        self.program
            .entry_points()
            .unwrap_or_default()
            .into_iter()
            .map(|entry| ReflectedEntryPoint {
                name: entry.name,
                stage: shader_stage(entry.stage),
            })
            .collect()
    }

    /// Returns compute workgroup size reflected by the validated shader module.
    pub(crate) fn compute_workgroup_size(
        &self,
        entry_point: &str,
    ) -> Result<Option<ReflectedWorkgroupSize>, String> {
        let entries = self.program.entry_points()?;
        let Some(entry) = entries.into_iter().find(|entry| {
            entry.name == entry_point && entry.stage == yawgpu_tint::PipelineStage::Compute
        }) else {
            return Ok(None);
        };
        let Some(literal_size) = entry.workgroup_size else {
            return Ok(None);
        };

        Ok(Some(ReflectedWorkgroupSize {
            entry_point: entry.name,
            literal_size,
            override_keys: [None, None, None],
            workgroup_storage_size: 0,
        }))
    }

    /// Returns compute workgroup size after resolving pipeline constants.
    pub(crate) fn resolved_compute_workgroup_size(
        &self,
        entry_point: &str,
        pipeline_constants: &PipelineConstants,
    ) -> Result<ReflectedWorkgroupSize, String> {
        let overrides = pipeline_constants
            .constants
            .iter()
            .map(|(name, value)| yawgpu_tint::OverrideValue {
                name: name.clone(),
                value: *value,
            })
            .collect::<Vec<_>>();
        let spirv = self.program.generate_spirv(
            entry_point,
            &yawgpu_tint::Bindings::default(),
            &overrides,
            true,
        )?;
        let literal_size = spirv_local_size(&spirv)
            .ok_or_else(|| "compute entry point workgroup size reflection failed".to_owned())?;
        Ok(ReflectedWorkgroupSize {
            entry_point: entry_point.to_owned(),
            literal_size,
            override_keys: [None, None, None],
            workgroup_storage_size: 0,
        })
    }

    /// Returns entry point io reflected by the validated shader module.
    pub(crate) fn entry_point_io(&self) -> Vec<ReflectedEntryPointIo> {
        self.program
            .entry_points()
            .unwrap_or_default()
            .into_iter()
            .map(|entry| {
                let is_compute = entry.stage == yawgpu_tint::PipelineStage::Compute;
                ReflectedEntryPointIo {
                    inputs: if is_compute {
                        Vec::new()
                    } else {
                        self.program
                            .entry_point_inputs(&entry.name)
                            .unwrap_or_default()
                            .into_iter()
                            .filter_map(reflected_io_location)
                            .collect()
                    },
                    outputs: if is_compute {
                        Vec::new()
                    } else {
                        self.program
                            .entry_point_outputs(&entry.name)
                            .unwrap_or_default()
                            .into_iter()
                            .filter_map(reflected_io_location)
                            .collect()
                    },
                    input_inter_stage_builtins: input_inter_stage_builtin_count(&entry),
                    entry_point: entry.name,
                }
            })
            .collect()
    }

    /// Returns resource bindings reflected by the validated shader module.
    pub(crate) fn resource_bindings(&self) -> Vec<ReflectedResourceBinding> {
        let mut seen = HashSet::new();
        self.program
            .entry_points()
            .unwrap_or_default()
            .into_iter()
            .flat_map(|entry| {
                self.resource_bindings_for_entry(&entry.name)
                    .unwrap_or_default()
            })
            .filter(|binding| seen.insert((binding.group, binding.binding)))
            .collect()
    }

    /// Returns resource bindings for entry reflected by the validated shader module.
    pub(crate) fn resource_bindings_for_entry(
        &self,
        entry_point: &str,
    ) -> Result<Vec<ReflectedResourceBinding>, String> {
        let entry_exists = self
            .program
            .entry_points()?
            .into_iter()
            .any(|entry| entry.name == entry_point);
        if !entry_exists {
            return Err("shader entry point was not found for resource reflection".to_owned());
        }

        self.program
            .resource_bindings(entry_point)?
            .into_iter()
            .map(reflected_resource_binding)
            .collect()
    }

    /// Returns storage buffer bindings that populate MSL `_mslBufferSizes`.
    pub(crate) fn msl_buffer_size_bindings_for_entry(
        &self,
        entry_point: &str,
    ) -> Result<Vec<MslBufferSizeBinding>, String> {
        Ok(self
            .resource_bindings_for_entry(entry_point)?
            .into_iter()
            .filter_map(|binding| match binding.kind {
                ReflectedResourceBindingKind::Buffer(
                    ReflectedBufferType::Storage | ReflectedBufferType::ReadOnlyStorage,
                ) => Some(MslBufferSizeBinding {
                    group: binding.group,
                    binding: binding.binding,
                }),
                _ => None,
            })
            .collect())
    }

    /// Returns fragment builtins reflected by the validated shader module.
    pub(crate) fn fragment_builtins(&self) -> Vec<ReflectedFragmentBuiltins> {
        self.program
            .entry_points()
            .unwrap_or_default()
            .into_iter()
            .filter(|entry| entry.stage == yawgpu_tint::PipelineStage::Fragment)
            .map(|entry| ReflectedFragmentBuiltins {
                entry_point: entry.name,
                frag_depth: entry.frag_depth_used,
                sample_mask: entry.sample_mask_used,
            })
            .collect()
    }

    /// Returns overrides reflected by the validated shader module.
    pub(crate) fn overrides(&self) -> Vec<ReflectedOverride> {
        self.program
            .overrides()
            .unwrap_or_default()
            .into_iter()
            .map(reflected_override)
            .collect()
    }
}

fn shader_stage(stage: yawgpu_tint::PipelineStage) -> ShaderStage {
    match stage {
        yawgpu_tint::PipelineStage::Vertex => ShaderStage::Vertex,
        yawgpu_tint::PipelineStage::Fragment => ShaderStage::Fragment,
        yawgpu_tint::PipelineStage::Compute => ShaderStage::Compute,
    }
}

fn override_values(constants: &PipelineConstants) -> Vec<yawgpu_tint::OverrideValue> {
    constants
        .constants
        .iter()
        .map(|(name, value)| yawgpu_tint::OverrideValue {
            name: name.clone(),
            value: *value,
        })
        .collect()
}

fn tint_bindings_for_msl(
    binding_map: &MslBindingMap,
    resource_bindings: &[ReflectedResourceBinding],
) -> Result<yawgpu_tint::Bindings, String> {
    let resources = resource_bindings
        .iter()
        .map(|binding| ((binding.group, binding.binding), &binding.kind))
        .collect::<HashMap<_, _>>();
    let mut bindings = yawgpu_tint::Bindings::default();
    for binding in &binding_map.resources {
        let remap = yawgpu_tint::BindingRemap {
            group: binding.group,
            binding: binding.binding,
            dst_group: 0,
            dst_binding: binding.metal_index,
        };
        match binding.kind {
            MslResourceBindingKind::Buffer => {
                match resources.get(&(binding.group, binding.binding)).copied() {
                    Some(ReflectedResourceBindingKind::Buffer(ReflectedBufferType::Uniform)) => {
                        bindings.uniform.push(remap);
                    }
                    Some(ReflectedResourceBindingKind::Buffer(
                        ReflectedBufferType::Storage | ReflectedBufferType::ReadOnlyStorage,
                    )) => {
                        bindings.storage.push(remap);
                    }
                    Some(_) => {
                        return Err(
                            "MSL buffer binding map entry does not match reflected resource kind"
                                .to_owned(),
                        );
                    }
                    None => {
                        return Err(
                            "MSL buffer binding map entry was not reflected for the entry point"
                                .to_owned(),
                        );
                    }
                }
            }
            MslResourceBindingKind::Texture => {
                match resources.get(&(binding.group, binding.binding)).copied() {
                    Some(ReflectedResourceBindingKind::StorageTexture { .. }) => {
                        bindings.storage_texture.push(remap);
                    }
                    Some(ReflectedResourceBindingKind::Texture { .. }) => {
                        bindings.texture.push(remap);
                    }
                    Some(_) => {
                        return Err(
                            "MSL texture binding map entry does not match reflected resource kind"
                                .to_owned(),
                        );
                    }
                    None => {
                        return Err(
                            "MSL texture binding map entry was not reflected for the entry point"
                                .to_owned(),
                        );
                    }
                }
            }
            MslResourceBindingKind::Sampler => {
                bindings.sampler.push(remap);
            }
            MslResourceBindingKind::ExternalTexture => {
                return Err("MSL external texture binding remap is not implemented".to_owned());
            }
        }
    }
    Ok(bindings)
}

fn msl_buffer_sizes_slot(
    binding_map: &MslBindingMap,
    buffer_size_bindings: &[MslBufferSizeBinding],
) -> Result<Option<u32>, String> {
    if buffer_size_bindings.is_empty() {
        return Ok(None);
    }
    let next_slot = binding_map
        .resources
        .iter()
        .filter_map(|binding| match binding.kind {
            MslResourceBindingKind::Buffer => Some(binding.metal_index),
            MslResourceBindingKind::ExternalTexture => binding.ext_params_buffer_slot,
            _ => None,
        })
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    if next_slot > u32::from(u8::MAX) {
        return Err("MSL generated buffer slot exceeds the supported slot range".to_owned());
    }
    Ok(Some(next_slot))
}

fn spirv_local_size(words: &[u32]) -> Option<[u32; 3]> {
    const OP_EXECUTION_MODE: u16 = 16;
    const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

    let mut offset = 5usize;
    while offset < words.len() {
        let instruction = words[offset];
        let word_count = usize::try_from(instruction >> 16).ok()?;
        let opcode = (instruction & 0xffff) as u16;
        if word_count == 0 || offset.checked_add(word_count)? > words.len() {
            return None;
        }
        if opcode == OP_EXECUTION_MODE
            && word_count >= 6
            && words[offset + 2] == EXECUTION_MODE_LOCAL_SIZE
        {
            return Some([words[offset + 3], words[offset + 4], words[offset + 5]]);
        }
        offset += word_count;
    }
    None
}

fn reflected_resource_binding(
    binding: yawgpu_tint::ResourceBinding,
) -> Result<ReflectedResourceBinding, String> {
    Ok(ReflectedResourceBinding {
        group: binding.group,
        binding: binding.binding,
        kind: resource_binding_kind(&binding)?,
        min_binding_size: binding.size,
        statically_used: true,
    })
}

fn resource_binding_kind(
    binding: &yawgpu_tint::ResourceBinding,
) -> Result<ReflectedResourceBindingKind, String> {
    match binding.resource_type {
        yawgpu_tint::ResourceType::UniformBuffer => Ok(ReflectedResourceBindingKind::Buffer(
            ReflectedBufferType::Uniform,
        )),
        yawgpu_tint::ResourceType::StorageBuffer => Ok(ReflectedResourceBindingKind::Buffer(
            ReflectedBufferType::Storage,
        )),
        yawgpu_tint::ResourceType::ReadOnlyStorageBuffer => Ok(
            ReflectedResourceBindingKind::Buffer(ReflectedBufferType::ReadOnlyStorage),
        ),
        yawgpu_tint::ResourceType::Sampler => Ok(ReflectedResourceBindingKind::Sampler {
            comparison: binding.sampler_type == yawgpu_tint::SamplerType::Comparison,
        }),
        yawgpu_tint::ResourceType::SampledTexture => Ok(texture_binding_kind(binding, false, true)),
        yawgpu_tint::ResourceType::MultisampledTexture => {
            Ok(texture_binding_kind(binding, true, true))
        }
        yawgpu_tint::ResourceType::DepthTexture => Ok(depth_texture_binding_kind(binding, false)),
        yawgpu_tint::ResourceType::DepthMultisampledTexture => {
            Ok(depth_texture_binding_kind(binding, true))
        }
        yawgpu_tint::ResourceType::WriteOnlyStorageTexture => Ok(storage_texture_binding_kind(
            binding,
            ReflectedStorageTextureAccess {
                read: false,
                write: true,
            },
        )?),
        yawgpu_tint::ResourceType::ReadOnlyStorageTexture => Ok(storage_texture_binding_kind(
            binding,
            ReflectedStorageTextureAccess {
                read: true,
                write: false,
            },
        )?),
        yawgpu_tint::ResourceType::ReadWriteStorageTexture => Ok(storage_texture_binding_kind(
            binding,
            ReflectedStorageTextureAccess {
                read: true,
                write: true,
            },
        )?),
        yawgpu_tint::ResourceType::ExternalTexture => {
            Ok(ReflectedResourceBindingKind::ExternalTexture)
        }
        yawgpu_tint::ResourceType::ReadOnlyTexelBuffer
        | yawgpu_tint::ResourceType::ReadWriteTexelBuffer
        | yawgpu_tint::ResourceType::InputAttachment => {
            Err("tint: unsupported reflected resource binding type".to_owned())
        }
    }
}

fn texture_binding_kind(
    binding: &yawgpu_tint::ResourceBinding,
    multisampled: bool,
    sampled: bool,
) -> ReflectedResourceBindingKind {
    ReflectedResourceBindingKind::Texture {
        sampled,
        sample_kind: sampled_kind(binding.sampled_kind),
        sample_usage: texture_sample_usage(binding.sample_usage),
        view_dimension: texture_view_dimension(binding.dim),
        multisampled,
    }
}

fn depth_texture_binding_kind(
    binding: &yawgpu_tint::ResourceBinding,
    multisampled: bool,
) -> ReflectedResourceBindingKind {
    ReflectedResourceBindingKind::Texture {
        sampled: false,
        sample_kind: None,
        sample_usage: texture_sample_usage(binding.sample_usage),
        view_dimension: texture_view_dimension(binding.dim),
        multisampled,
    }
}

fn texture_sample_usage(usage: yawgpu_tint::TextureSampleUsage) -> ReflectedTextureSampleUsage {
    match usage {
        yawgpu_tint::TextureSampleUsage::Load => ReflectedTextureSampleUsage::Load,
        yawgpu_tint::TextureSampleUsage::Sample => ReflectedTextureSampleUsage::Sample,
        yawgpu_tint::TextureSampleUsage::Gather => ReflectedTextureSampleUsage::Gather,
    }
}

fn storage_texture_binding_kind(
    binding: &yawgpu_tint::ResourceBinding,
    access: ReflectedStorageTextureAccess,
) -> Result<ReflectedResourceBindingKind, String> {
    Ok(ReflectedResourceBindingKind::StorageTexture {
        format: texel_format(binding.texel_format)?,
        access,
        view_dimension: texture_view_dimension(binding.dim),
    })
}

fn sampled_kind(kind: yawgpu_tint::SampledKind) -> Option<ReflectedTypeScalarClass> {
    match kind {
        yawgpu_tint::SampledKind::Float
        | yawgpu_tint::SampledKind::Filterable
        | yawgpu_tint::SampledKind::Unfilterable
        | yawgpu_tint::SampledKind::UnknownFilterable => Some(ReflectedTypeScalarClass::Float),
        yawgpu_tint::SampledKind::UInt => Some(ReflectedTypeScalarClass::Uint),
        yawgpu_tint::SampledKind::SInt => Some(ReflectedTypeScalarClass::Sint),
    }
}

fn texture_view_dimension(dim: yawgpu_tint::TextureDimension) -> ReflectedTextureViewDimension {
    match dim {
        yawgpu_tint::TextureDimension::D1 => ReflectedTextureViewDimension::D1,
        yawgpu_tint::TextureDimension::D2 => ReflectedTextureViewDimension::D2,
        yawgpu_tint::TextureDimension::D2Array => ReflectedTextureViewDimension::D2Array,
        yawgpu_tint::TextureDimension::D3 => ReflectedTextureViewDimension::D3,
        yawgpu_tint::TextureDimension::Cube => ReflectedTextureViewDimension::Cube,
        yawgpu_tint::TextureDimension::CubeArray => ReflectedTextureViewDimension::CubeArray,
        yawgpu_tint::TextureDimension::None => ReflectedTextureViewDimension::D2,
    }
}

fn reflected_io_location(variable: yawgpu_tint::StageVariable) -> Option<ReflectedIoLocation> {
    Some(ReflectedIoLocation {
        location: variable.location?,
        ty: reflected_stage_variable_type(variable.component_type, variable.composition_type)?,
        interpolation: reflected_interpolation(variable.interpolation_type),
        sampling: reflected_sampling(variable.interpolation_sampling),
    })
}

fn reflected_stage_variable_type(
    component: yawgpu_tint::ComponentType,
    composition: yawgpu_tint::CompositionType,
) -> Option<ReflectedTypeClass> {
    let (scalar, width) = match component {
        yawgpu_tint::ComponentType::F32 => (ReflectedTypeScalarClass::Float, 4),
        yawgpu_tint::ComponentType::U32 => (ReflectedTypeScalarClass::Uint, 4),
        yawgpu_tint::ComponentType::I32 => (ReflectedTypeScalarClass::Sint, 4),
        yawgpu_tint::ComponentType::F16 => (ReflectedTypeScalarClass::Float, 2),
        yawgpu_tint::ComponentType::Unknown => return None,
    };
    let components = match composition {
        yawgpu_tint::CompositionType::Scalar => 1,
        yawgpu_tint::CompositionType::Vec2 => 2,
        yawgpu_tint::CompositionType::Vec3 => 3,
        yawgpu_tint::CompositionType::Vec4 => 4,
        yawgpu_tint::CompositionType::Unknown => return None,
    };
    Some(ReflectedTypeClass {
        scalar,
        components,
        width,
    })
}

fn reflected_interpolation(
    interpolation: yawgpu_tint::InterpolationType,
) -> Option<ReflectedInterpolation> {
    match interpolation {
        yawgpu_tint::InterpolationType::Perspective => Some(ReflectedInterpolation::Perspective),
        yawgpu_tint::InterpolationType::Linear => Some(ReflectedInterpolation::Linear),
        yawgpu_tint::InterpolationType::Flat => Some(ReflectedInterpolation::Flat),
        yawgpu_tint::InterpolationType::Unknown => None,
    }
}

fn reflected_sampling(sampling: yawgpu_tint::InterpolationSampling) -> Option<ReflectedSampling> {
    match sampling {
        yawgpu_tint::InterpolationSampling::None | yawgpu_tint::InterpolationSampling::Unknown => {
            None
        }
        yawgpu_tint::InterpolationSampling::Center => Some(ReflectedSampling::Center),
        yawgpu_tint::InterpolationSampling::Centroid => Some(ReflectedSampling::Centroid),
        yawgpu_tint::InterpolationSampling::Sample => Some(ReflectedSampling::Sample),
        yawgpu_tint::InterpolationSampling::First => Some(ReflectedSampling::First),
        yawgpu_tint::InterpolationSampling::Either => Some(ReflectedSampling::Either),
    }
}

fn input_inter_stage_builtin_count(entry: &yawgpu_tint::EntryPoint) -> u32 {
    if entry.stage == yawgpu_tint::PipelineStage::Compute {
        return 0;
    }
    // WebGPU counts these stage-input builtins against
    // `maxInterStageShaderVariables`: front_facing, sample_index, input
    // sample_mask, primitive_index, subgroup_invocation_id, and subgroup_size.
    // Tint reflects each as an entry-point boolean; position and all other
    // builtins are intentionally excluded from this count.
    u32::from(entry.front_facing_used)
        + u32::from(entry.sample_index_used)
        + u32::from(entry.input_sample_mask_used)
        + u32::from(entry.primitive_index_used)
        + u32::from(entry.subgroup_invocation_id_used)
        + u32::from(entry.subgroup_size_used)
}

fn texel_format(format: yawgpu_tint::TexelFormat) -> Result<String, String> {
    let name = match format {
        yawgpu_tint::TexelFormat::R8Snorm => "R8Snorm",
        yawgpu_tint::TexelFormat::R8Uint => "R8Uint",
        yawgpu_tint::TexelFormat::R8Sint => "R8Sint",
        yawgpu_tint::TexelFormat::Rg8Unorm => "Rg8Unorm",
        yawgpu_tint::TexelFormat::Rg8Snorm => "Rg8Snorm",
        yawgpu_tint::TexelFormat::Rg8Uint => "Rg8Uint",
        yawgpu_tint::TexelFormat::Rg8Sint => "Rg8Sint",
        yawgpu_tint::TexelFormat::R16Unorm => "R16Unorm",
        yawgpu_tint::TexelFormat::R16Snorm => "R16Snorm",
        yawgpu_tint::TexelFormat::R16Uint => "R16Uint",
        yawgpu_tint::TexelFormat::R16Sint => "R16Sint",
        yawgpu_tint::TexelFormat::R16Float => "R16Float",
        yawgpu_tint::TexelFormat::Rg16Unorm => "Rg16Unorm",
        yawgpu_tint::TexelFormat::Rg16Snorm => "Rg16Snorm",
        yawgpu_tint::TexelFormat::Rg16Uint => "Rg16Uint",
        yawgpu_tint::TexelFormat::Rg16Sint => "Rg16Sint",
        yawgpu_tint::TexelFormat::Rg16Float => "Rg16Float",
        yawgpu_tint::TexelFormat::Bgra8Unorm => "Bgra8Unorm",
        yawgpu_tint::TexelFormat::Rgba8Unorm => "Rgba8Unorm",
        yawgpu_tint::TexelFormat::Rgba8Snorm => "Rgba8Snorm",
        yawgpu_tint::TexelFormat::Rgba8Uint => "Rgba8Uint",
        yawgpu_tint::TexelFormat::Rgba8Sint => "Rgba8Sint",
        yawgpu_tint::TexelFormat::Rgba16Unorm => "Rgba16Unorm",
        yawgpu_tint::TexelFormat::Rgba16Snorm => "Rgba16Snorm",
        yawgpu_tint::TexelFormat::Rgba16Uint => "Rgba16Uint",
        yawgpu_tint::TexelFormat::Rgba16Sint => "Rgba16Sint",
        yawgpu_tint::TexelFormat::Rgba16Float => "Rgba16Float",
        yawgpu_tint::TexelFormat::R32Uint => "R32Uint",
        yawgpu_tint::TexelFormat::R32Sint => "R32Sint",
        yawgpu_tint::TexelFormat::R32Float => "R32Float",
        yawgpu_tint::TexelFormat::Rg32Uint => "Rg32Uint",
        yawgpu_tint::TexelFormat::Rg32Sint => "Rg32Sint",
        yawgpu_tint::TexelFormat::Rg32Float => "Rg32Float",
        yawgpu_tint::TexelFormat::Rgba32Uint => "Rgba32Uint",
        yawgpu_tint::TexelFormat::Rgba32Sint => "Rgba32Sint",
        yawgpu_tint::TexelFormat::Rgba32Float => "Rgba32Float",
        yawgpu_tint::TexelFormat::R8Unorm => "R8Unorm",
        yawgpu_tint::TexelFormat::Rgb10A2Uint => "Rgb10A2Uint",
        yawgpu_tint::TexelFormat::Rgb10A2Unorm => "Rgb10A2Unorm",
        yawgpu_tint::TexelFormat::Rg11B10Ufloat => "Rg11B10Ufloat",
        yawgpu_tint::TexelFormat::None => {
            return Err("tint: storage texture has no texel format".to_owned());
        }
    };
    Ok(name.to_owned())
}

fn reflected_override(override_: yawgpu_tint::Override) -> ReflectedOverride {
    let ty = match override_.type_class {
        yawgpu_tint::OverrideType::Bool => ReflectedTypeClass {
            scalar: ReflectedTypeScalarClass::Bool,
            components: 1,
            width: 1,
        },
        yawgpu_tint::OverrideType::Float32 => ReflectedTypeClass {
            scalar: ReflectedTypeScalarClass::Float,
            components: 1,
            width: 4,
        },
        yawgpu_tint::OverrideType::Uint32 => ReflectedTypeClass {
            scalar: ReflectedTypeScalarClass::Uint,
            components: 1,
            width: 4,
        },
        yawgpu_tint::OverrideType::Int32 => ReflectedTypeClass {
            scalar: ReflectedTypeScalarClass::Sint,
            components: 1,
            width: 4,
        },
        yawgpu_tint::OverrideType::Float16 => ReflectedTypeClass {
            scalar: ReflectedTypeScalarClass::Float,
            components: 1,
            width: 2,
        },
    };
    let default_value = override_.has_default.then_some(match override_.type_class {
        yawgpu_tint::OverrideType::Bool => {
            ReflectedOverrideValue::Bool(override_.default_value != 0.0)
        }
        _ => ReflectedOverrideValue::Number(override_.default_value),
    });

    ReflectedOverride {
        name: (!override_.name.is_empty()).then_some(override_.name),
        // Only surface the id when `@id` is explicit, matching the default
        // frontend — yawgpu keys constants by numeric id only for `@id`
        // overrides, and Tint assigns an implicit id to every override.
        id: override_.has_explicit_id.then_some(override_.id),
        ty,
        has_default: override_.has_default,
        default_value,
    }
}

#[cfg(all(test, feature = "tint"))]
mod tests {
    use super::*;

    #[test]
    fn reflects_compute_workgroup_size_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
@compute @workgroup_size(8, 4, 1)
fn cs() {}
"#,
        )
        .unwrap();

        let reflected = module.compute_workgroup_size("cs").unwrap().unwrap();
        assert_eq!(reflected.literal_size, [8, 4, 1]);

        let resolved = module
            .resolved_compute_workgroup_size("cs", &PipelineConstants::default())
            .unwrap();
        assert_eq!(resolved.literal_size, [8, 4, 1]);
    }

    #[test]
    fn resolves_override_driven_workgroup_size_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
override x: u32 = 4;

@compute @workgroup_size(x, 2, 1)
fn cs() {}
"#,
        )
        .unwrap();
        let constants = PipelineConstants::from_iter([("x".to_owned(), 8.0)]);

        let resolved = module
            .resolved_compute_workgroup_size("cs", &constants)
            .unwrap();
        assert_eq!(resolved.literal_size, [8, 2, 1]);
    }

    #[test]
    fn reflects_resource_bindings_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
struct U {
  x: vec4<f32>,
}

struct S {
  x: array<u32>,
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> s: S;
@group(1) @binding(0) var<storage, read> ro: S;
@group(1) @binding(1) var tex: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;

@compute @workgroup_size(1)
fn cs() {
  let a = u.x.x;
  s.x[0] = ro.x[0] + u32(textureDimensions(tex).x) + u32(a);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return textureSample(tex, samp, vec2<f32>(0.5));
}
"#,
        )
        .unwrap();

        let compute = module.resource_bindings_for_entry("cs").unwrap();
        assert!(compute.iter().any(|binding| {
            binding.group == 0
                && binding.binding == 0
                && binding.kind
                    == ReflectedResourceBindingKind::Buffer(ReflectedBufferType::Uniform)
                && binding.statically_used
                && binding.min_binding_size > 0
        }));
        assert!(compute.iter().any(|binding| {
            binding.group == 0
                && binding.binding == 1
                && binding.kind
                    == ReflectedResourceBindingKind::Buffer(ReflectedBufferType::Storage)
        }));
        assert!(compute.iter().any(|binding| {
            binding.group == 1
                && binding.binding == 0
                && binding.kind
                    == ReflectedResourceBindingKind::Buffer(ReflectedBufferType::ReadOnlyStorage)
        }));
        assert!(compute.iter().any(|binding| {
            binding.group == 1
                && binding.binding == 1
                && binding.kind
                    == ReflectedResourceBindingKind::Texture {
                        sampled: true,
                        sample_kind: Some(ReflectedTypeScalarClass::Float),
                        sample_usage: ReflectedTextureSampleUsage::Load,
                        view_dimension: ReflectedTextureViewDimension::D2,
                        multisampled: false,
                    }
        }));

        let fragment = module.resource_bindings_for_entry("fs").unwrap();
        assert!(fragment.iter().any(|binding| {
            binding.group == 1
                && binding.binding == 2
                && binding.kind == ReflectedResourceBindingKind::Sampler { comparison: false }
        }));
    }

    #[test]
    fn reflects_texture_gather_usage_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

fn helper(t: texture_2d<f32>) -> vec4f {
  return textureGather(0, t, samp, vec2f(0.5));
}

@compute @workgroup_size(1)
fn cs() {
  _ = helper(tex);
}
"#,
        )
        .unwrap();

        let compute = module.resource_bindings_for_entry("cs").unwrap();
        assert!(compute.iter().any(|binding| {
            binding.group == 0
                && binding.binding == 0
                && binding.kind
                    == ReflectedResourceBindingKind::Texture {
                        sampled: true,
                        sample_kind: Some(ReflectedTypeScalarClass::Float),
                        sample_usage: ReflectedTextureSampleUsage::Gather,
                        view_dimension: ReflectedTextureViewDimension::D2,
                        multisampled: false,
                    }
        }));
    }

    #[test]
    fn reflects_vertex_fragment_io_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) value: f32,
  @location(1) @interpolate(flat) index: u32,
}

@vertex
fn vs(@location(0) value: f32, @location(1) @interpolate(flat) index: u32) -> VsOut {
  return VsOut(vec4f(0.0, 0.0, 0.0, 1.0), value, index);
}

@fragment
fn fs(
  @builtin(position) pos: vec4f,
  @builtin(front_facing) ff: bool,
  @builtin(sample_index) si: u32,
  @builtin(sample_mask) sm: u32,
  @location(0) value: f32,
  @location(1) @interpolate(flat) index: u32,
) -> @location(0) vec4f {
  _ = pos;
  _ = ff;
  _ = si;
  _ = sm;
  return vec4f(value + f32(index), 0.0, 0.0, 1.0);
}
"#,
        )
        .unwrap();

        let io = module.entry_point_io();
        let vs = io.iter().find(|entry| entry.entry_point == "vs").unwrap();
        assert_eq!(vs.inputs.len(), 2);
        let vs_value = vs.inputs.iter().find(|input| input.location == 0).unwrap();
        assert_eq!(vs_value.ty.scalar, ReflectedTypeScalarClass::Float);
        assert_eq!(vs_value.ty.components, 1);
        assert_eq!(vs_value.ty.width, 4);
        let vs_index = vs.inputs.iter().find(|input| input.location == 1).unwrap();
        assert_eq!(vs_index.ty.scalar, ReflectedTypeScalarClass::Uint);
        assert_eq!(vs_index.interpolation, Some(ReflectedInterpolation::Flat));
        assert_eq!(vs_index.sampling, Some(ReflectedSampling::First));
        assert_eq!(vs.outputs.len(), 2);
        assert_eq!(vs.input_inter_stage_builtins, 0);

        let fs = io.iter().find(|entry| entry.entry_point == "fs").unwrap();
        assert_eq!(fs.inputs.len(), 2);
        assert!(fs.outputs.iter().any(|output| {
            output.location == 0
                && output.ty.scalar == ReflectedTypeScalarClass::Float
                && output.ty.components == 4
        }));
        assert_eq!(fs.input_inter_stage_builtins, 3);
    }

    #[test]
    fn reflects_override_default_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
override x: f32 = 1.5;

@compute @workgroup_size(1)
fn cs() {}
"#,
        )
        .unwrap();

        let overrides = module.overrides();
        assert_eq!(overrides.len(), 1);
        assert_eq!(overrides[0].name.as_deref(), Some("x"));
        assert_eq!(overrides[0].ty.scalar, ReflectedTypeScalarClass::Float);
        assert_eq!(
            overrides[0].default_value,
            Some(ReflectedOverrideValue::Number(1.5))
        );
    }

    #[test]
    fn reflects_fragment_builtins_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
@fragment
fn fs() -> @builtin(frag_depth) f32 {
  return 0.5;
}
"#,
        )
        .unwrap();

        assert_eq!(
            module.fragment_builtins(),
            vec![ReflectedFragmentBuiltins {
                entry_point: "fs".to_owned(),
                frag_depth: true,
                sample_mask: false,
            }]
        );
    }

    #[test]
    fn generate_compute_msl_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
@compute @workgroup_size(1)
fn cs() {}
"#,
        )
        .unwrap();

        let generated = module
            .generate_msl(
                "cs",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &PipelineConstants::default(),
            )
            .unwrap();

        assert_eq!(generated.entry_point, "cs");
        assert!(generated.source.contains("kernel"));
        assert!(generated.source.contains("cs"));
        assert_eq!(generated.frag_depth_clamp_slot, None);
    }

    #[test]
    fn generate_spirv_from_tint_for_compute_and_render_entries() {
        let compute = parse_and_validate_wgsl(
            r#"
@compute @workgroup_size(1)
fn cs() {}
"#,
        )
        .unwrap()
        .generate_spirv(
            "cs",
            ShaderStage::Compute,
            &PipelineConstants::default(),
            false,
        )
        .unwrap();
        assert_eq!(compute.first().copied(), Some(0x0723_0203));

        let render = parse_and_validate_wgsl(
            r#"
struct VOut {
  @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs() -> VOut {
  var out: VOut;
  out.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  return out;
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0);
}
"#,
        )
        .unwrap();

        let vertex = render
            .generate_spirv(
                "vs",
                ShaderStage::Vertex,
                &PipelineConstants::default(),
                false,
            )
            .unwrap();
        let fragment = render
            .generate_spirv(
                "fs",
                ShaderStage::Fragment,
                &PipelineConstants::default(),
                false,
            )
            .unwrap();
        assert_eq!(vertex.first().copied(), Some(0x0723_0203));
        assert_eq!(fragment.first().copied(), Some(0x0723_0203));
    }

    #[cfg(feature = "gles")]
    #[test]
    fn generate_glsl_from_tint() {
        let generated = parse_and_validate_wgsl(
            r#"
@compute @workgroup_size(1)
fn cs() {}
"#,
        )
        .unwrap()
        .generate_glsl("cs", ShaderStage::Compute, &PipelineConstants::default())
        .unwrap();

        assert_eq!(generated.entry_point, "cs");
        assert!(generated.source.contains("#version 310 es"));
    }

    #[test]
    fn generate_msl_uses_metal_binding_indices_from_remap() {
        let module = parse_and_validate_wgsl(
            r#"
struct U {
  x: vec4<f32>,
}

struct S {
  x: u32,
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> s: S;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@compute @workgroup_size(1)
fn cs() {
  let dims = textureDimensions(tex);
  let sampled = textureSampleLevel(tex, samp, vec2<f32>(0.5), 0.0);
  s.x = u32(u.x.x + sampled.x) + u32(dims.x);
}
"#,
        )
        .unwrap();

        let generated = module
            .generate_msl(
                "cs",
                &MslBindingMap {
                    resources: vec![
                        MslResourceBinding {
                            group: 0,
                            binding: 0,
                            metal_index: 4,
                            ext_params_buffer_slot: None,
                            kind: MslResourceBindingKind::Buffer,
                        },
                        MslResourceBinding {
                            group: 0,
                            binding: 1,
                            metal_index: 7,
                            ext_params_buffer_slot: None,
                            kind: MslResourceBindingKind::Buffer,
                        },
                        MslResourceBinding {
                            group: 0,
                            binding: 2,
                            metal_index: 5,
                            ext_params_buffer_slot: None,
                            kind: MslResourceBindingKind::Texture,
                        },
                        MslResourceBinding {
                            group: 0,
                            binding: 3,
                            metal_index: 6,
                            ext_params_buffer_slot: None,
                            kind: MslResourceBindingKind::Sampler,
                        },
                    ],
                },
                &PipelineConstants::default(),
            )
            .unwrap();

        assert!(generated.source.contains("[[buffer(4)]]"));
        assert!(generated.source.contains("[[buffer(7)]]"));
        assert!(generated.source.contains("[[texture(5)]]"));
        assert!(generated.source.contains("[[sampler(6)]]"));
    }
}
