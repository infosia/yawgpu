//! Tint shader frontend — yawgpu's WGSL→{MSL, SPIR-V, GLSL} compiler and
//! reflection source, backed by Dawn's Tint via the `yawgpu-tint` shim. This is
//! the sole shader frontend (the `crate::frontend` alias points here), and the
//! render path emits per-stage shader sources for pipeline creation.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::shader::{CompilationMessage, CompilationSeverity};
pub(crate) use crate::shader_types::*;

/// Stores reflected shader module data used by validation and backend submission.
#[derive(Debug)]
pub struct ReflectedModule {
    /// Tint program.
    pub program: yawgpu_tint::Program,
    /// Non-fatal compilation warnings.
    pub(crate) warnings: Vec<CompilationMessage>,
}

// SAFETY: the wrapped `yawgpu_tint::Program` (an opaque Tint handle) is treated as
// immutable after parsing — reflection and codegen only read from it — so it is safe
// to send/share across threads.
unsafe impl Send for ReflectedModule {}

// SAFETY: See the `Send` impl above.
unsafe impl Sync for ReflectedModule {}

/// Returns parse and validate wgsl.
pub(crate) fn parse_and_validate_wgsl(src: &str) -> Result<ReflectedModule, String> {
    parse_and_validate_wgsl_gated(src, true, true, true, true, true)
}

/// Returns parse and validate wgsl using the supplied feature gates.
pub(crate) fn parse_and_validate_wgsl_gated(
    src: &str,
    shader_f16: bool,
    subgroups: bool,
    dual_source_blending: bool,
    clip_distances: bool,
    primitive_index: bool,
) -> Result<ReflectedModule, String> {
    let program = yawgpu_tint::Program::parse(
        src,
        shader_f16,
        subgroups,
        dual_source_blending,
        clip_distances,
        primitive_index,
        crate::SUPPORTED_WGSL_LANGUAGE_FEATURES,
    )?;
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
    ///
    /// `vulkan_memory_model` enables Tint's SPV_KHR_vulkan_memory_model output
    /// when the Vulkan backend enabled `VK_KHR_vulkan_memory_model` /
    /// `vulkanMemoryModel`. SPIR-V robustness stays enabled.
    ///
    /// `multisampled_input_attachment` makes Tint emit multisampled
    /// `SubpassData` input attachments (the 2-arg `inputAttachmentLoad(ia,
    /// sample_index)` overload) so per-sample MSAA subpass input works.
    pub(crate) fn generate_spirv(
        &self,
        entry_name: &str,
        _stage: ShaderStage,
        pipeline_constants: &PipelineConstants,
        vulkan_memory_model: bool,
        framebuffer_fetch_descriptor_set: u32,
        multisampled_input_attachment: bool,
    ) -> Result<Vec<u32>, String> {
        self.program.generate_spirv(
            entry_name,
            &yawgpu_tint::Bindings::default(),
            &override_values(pipeline_constants),
            true,
            vulkan_memory_model,
            framebuffer_fetch_descriptor_set,
            multisampled_input_attachment,
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
        // Robustness ENABLED (disable_robustness = false): WebGPU requires
        // out-of-bounds safety in compute shaders too.
        self.generate_stage_msl(
            entry_name,
            binding_map,
            pipeline_constants,
            &[],
            &[],
            false,
            false,
            0xFFFF_FFFF,
        )
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
        let vertex_buffers = tint_vertex_buffers(vertex_buffers)?;
        // `force_point_size` makes Tint emit `[[point_size]] = 1.0` (point-list
        // topology requires it on Metal).
        self.generate_stage_msl(
            entry_name,
            binding_map,
            pipeline_constants,
            &[],
            &vertex_buffers,
            false,
            force_point_size,
            0xFFFF_FFFF,
        )
    }

    /// Generates render fragment MSL for a validated shader module.
    pub(crate) fn generate_render_fragment_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        subpass_color_slots: &[((u32, u32), u32)],
        pipeline_constants: &PipelineConstants,
        sample_mask: u32,
    ) -> Result<GeneratedMsl, String> {
        self.generate_stage_msl(
            entry_name,
            binding_map,
            pipeline_constants,
            subpass_color_slots,
            &[],
            false,
            false,
            sample_mask,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_stage_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        pipeline_constants: &PipelineConstants,
        subpass_color_slots: &[((u32, u32), u32)],
        vertex_buffers: &[yawgpu_tint::VertexBuffer],
        disable_robustness: bool,
        emit_vertex_point_size: bool,
        fixed_sample_mask: u32,
    ) -> Result<GeneratedMsl, String> {
        let bindings = tint_bindings_for_msl(
            binding_map,
            &self.resource_bindings_for_entry(entry_name)?,
            subpass_color_slots,
        )?;
        let binding_buffer_sizes_slot = msl_buffer_sizes_slot(binding_map)?;
        let buffer_sizes_slot = if vertex_buffers.is_empty() {
            binding_buffer_sizes_slot
        } else {
            let max_vertex_metal_index = vertex_buffers
                .iter()
                .map(|buffer| buffer.metal_index)
                .max()
                .unwrap_or(0);
            binding_buffer_sizes_slot.max(max_vertex_metal_index.saturating_add(1))
        };
        if buffer_sizes_slot > u32::from(u8::MAX) {
            return Err("MSL generated buffer slot exceeds the supported slot range".to_owned());
        }
        let output = self.program.generate_msl(
            entry_name,
            &bindings,
            &override_values(pipeline_constants),
            buffer_sizes_slot,
            // The wrapper takes `robust` (robustness ENABLED), which is the
            // negation of this fn's `disable_robustness`.
            !disable_robustness,
            emit_vertex_point_size,
            vertex_buffers,
            fixed_sample_mask,
        )?;
        let buffer_size_bindings = output
            .buffer_size_bindings
            .into_iter()
            .map(|binding| MslBufferSizeBinding {
                group: binding.group,
                binding: binding.binding,
            })
            .collect::<Vec<_>>();
        Ok(GeneratedMsl {
            source: output.source,
            entry_point: output.entry_point,
            buffer_sizes_slot: (!buffer_size_bindings.is_empty() || !vertex_buffers.is_empty())
                .then_some(buffer_sizes_slot),
            buffer_size_bindings,
            frag_depth_clamp_slot: output.frag_depth_clamp_slot,
            workgroup_memory_sizes: output
                .workgroup_allocations
                .iter()
                .map(|&size| size.div_ceil(16) * 16)
                .collect(),
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
            workgroup_storage_size: self.program.workgroup_storage_size(&[])?,
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
            false,
            0,
            false,
        )?;
        let literal_size = spirv_local_size(&spirv)
            .ok_or_else(|| "compute entry point workgroup size reflection failed".to_owned())?;
        Ok(ReflectedWorkgroupSize {
            entry_point: entry_point.to_owned(),
            literal_size,
            override_keys: [None, None, None],
            workgroup_storage_size: self.program.workgroup_storage_size(&overrides)?,
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

    /// Returns the `@color(N)` framebuffer-fetch slots read by the given fragment entry point.
    ///
    /// Slots are ascending. Returns empty if the shader uses no framebuffer fetch.
    pub(crate) fn fragment_color_inputs(&self, entry_point: &str) -> Vec<u32> {
        let mut slots = self
            .program
            .entry_point_inputs(entry_point)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|variable| variable.color)
            .collect::<Vec<_>>();
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    /// Returns true when the fragment entry point writes a second dual-source
    /// blend output (`@location(0) @blend_src(1)`).
    pub(crate) fn fragment_writes_blend_src_1(&self, entry_point: &str) -> bool {
        self.program
            .entry_point_outputs(entry_point)
            .unwrap_or_default()
            .into_iter()
            .any(|variable| variable.blend_src == Some(1))
    }

    /// Returns the vertex clip-distances array size reflected for `entry_point`.
    pub(crate) fn vertex_clip_distances_size(&self, entry_point: &str) -> u32 {
        self.program
            .entry_points()
            .unwrap_or_default()
            .into_iter()
            .find(|entry| {
                entry.stage == yawgpu_tint::PipelineStage::Vertex && entry.name == entry_point
            })
            .and_then(|entry| entry.clip_distances_size)
            .unwrap_or(0)
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

fn tint_vertex_buffers(
    vertex_buffers: &[MslVertexBufferBinding],
) -> Result<Vec<yawgpu_tint::VertexBuffer>, String> {
    vertex_buffers
        .iter()
        .map(|buffer| {
            let array_stride = u32::try_from(buffer.array_stride)
                .map_err(|_| "MSL vertex buffer array stride exceeds u32".to_owned())?;
            let attributes = buffer
                .attributes
                .iter()
                .map(|attribute| {
                    let offset = u32::try_from(attribute.offset)
                        .map_err(|_| "MSL vertex attribute offset exceeds u32".to_owned())?;
                    Ok(yawgpu_tint::VertexAttribute {
                        format: tint_vertex_format(attribute.format),
                        offset,
                        shader_location: attribute.shader_location,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(yawgpu_tint::VertexBuffer {
                slot: buffer.slot,
                metal_index: buffer.metal_index,
                array_stride,
                step_mode: tint_vertex_step_mode(buffer.step_mode),
                attributes,
            })
        })
        .collect()
}

fn tint_vertex_step_mode(step_mode: MslVertexStepMode) -> yawgpu_tint::VertexStepMode {
    match step_mode {
        MslVertexStepMode::Vertex => yawgpu_tint::VertexStepMode::Vertex,
        MslVertexStepMode::Instance => yawgpu_tint::VertexStepMode::Instance,
    }
}

fn tint_vertex_format(format: MslVertexFormat) -> yawgpu_tint::VertexFormat {
    match format {
        MslVertexFormat::Uint8 => yawgpu_tint::VertexFormat::Uint8,
        MslVertexFormat::Uint8x2 => yawgpu_tint::VertexFormat::Uint8x2,
        MslVertexFormat::Uint8x4 => yawgpu_tint::VertexFormat::Uint8x4,
        MslVertexFormat::Sint8 => yawgpu_tint::VertexFormat::Sint8,
        MslVertexFormat::Sint8x2 => yawgpu_tint::VertexFormat::Sint8x2,
        MslVertexFormat::Sint8x4 => yawgpu_tint::VertexFormat::Sint8x4,
        MslVertexFormat::Unorm8 => yawgpu_tint::VertexFormat::Unorm8,
        MslVertexFormat::Unorm8x2 => yawgpu_tint::VertexFormat::Unorm8x2,
        MslVertexFormat::Unorm8x4 => yawgpu_tint::VertexFormat::Unorm8x4,
        MslVertexFormat::Snorm8 => yawgpu_tint::VertexFormat::Snorm8,
        MslVertexFormat::Snorm8x2 => yawgpu_tint::VertexFormat::Snorm8x2,
        MslVertexFormat::Snorm8x4 => yawgpu_tint::VertexFormat::Snorm8x4,
        MslVertexFormat::Uint16 => yawgpu_tint::VertexFormat::Uint16,
        MslVertexFormat::Uint16x2 => yawgpu_tint::VertexFormat::Uint16x2,
        MslVertexFormat::Uint16x4 => yawgpu_tint::VertexFormat::Uint16x4,
        MslVertexFormat::Sint16 => yawgpu_tint::VertexFormat::Sint16,
        MslVertexFormat::Sint16x2 => yawgpu_tint::VertexFormat::Sint16x2,
        MslVertexFormat::Sint16x4 => yawgpu_tint::VertexFormat::Sint16x4,
        MslVertexFormat::Unorm16 => yawgpu_tint::VertexFormat::Unorm16,
        MslVertexFormat::Unorm16x2 => yawgpu_tint::VertexFormat::Unorm16x2,
        MslVertexFormat::Unorm16x4 => yawgpu_tint::VertexFormat::Unorm16x4,
        MslVertexFormat::Snorm16 => yawgpu_tint::VertexFormat::Snorm16,
        MslVertexFormat::Snorm16x2 => yawgpu_tint::VertexFormat::Snorm16x2,
        MslVertexFormat::Snorm16x4 => yawgpu_tint::VertexFormat::Snorm16x4,
        MslVertexFormat::Float16 => yawgpu_tint::VertexFormat::Float16,
        MslVertexFormat::Float16x2 => yawgpu_tint::VertexFormat::Float16x2,
        MslVertexFormat::Float16x4 => yawgpu_tint::VertexFormat::Float16x4,
        MslVertexFormat::Float32 => yawgpu_tint::VertexFormat::Float32,
        MslVertexFormat::Float32x2 => yawgpu_tint::VertexFormat::Float32x2,
        MslVertexFormat::Float32x3 => yawgpu_tint::VertexFormat::Float32x3,
        MslVertexFormat::Float32x4 => yawgpu_tint::VertexFormat::Float32x4,
        MslVertexFormat::Uint32 => yawgpu_tint::VertexFormat::Uint32,
        MslVertexFormat::Uint32x2 => yawgpu_tint::VertexFormat::Uint32x2,
        MslVertexFormat::Uint32x3 => yawgpu_tint::VertexFormat::Uint32x3,
        MslVertexFormat::Uint32x4 => yawgpu_tint::VertexFormat::Uint32x4,
        MslVertexFormat::Sint32 => yawgpu_tint::VertexFormat::Sint32,
        MslVertexFormat::Sint32x2 => yawgpu_tint::VertexFormat::Sint32x2,
        MslVertexFormat::Sint32x3 => yawgpu_tint::VertexFormat::Sint32x3,
        MslVertexFormat::Sint32x4 => yawgpu_tint::VertexFormat::Sint32x4,
        MslVertexFormat::Unorm10_10_10_2 => yawgpu_tint::VertexFormat::Unorm10_10_10_2,
        MslVertexFormat::Unorm8x4Bgra => yawgpu_tint::VertexFormat::Unorm8x4Bgra,
    }
}

fn tint_bindings_for_msl(
    binding_map: &MslBindingMap,
    resource_bindings: &[ReflectedResourceBinding],
    subpass_color_slots: &[((u32, u32), u32)],
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
                    None => {}
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
                    None => {}
                }
            }
            MslResourceBindingKind::Sampler => {
                match resources.get(&(binding.group, binding.binding)).copied() {
                    Some(ReflectedResourceBindingKind::Sampler { .. }) => {
                        bindings.sampler.push(remap);
                    }
                    Some(_) => {
                        return Err(
                            "MSL sampler binding map entry does not match reflected resource kind"
                                .to_owned(),
                        );
                    }
                    None => {}
                }
            }
            MslResourceBindingKind::ExternalTexture => {
                match resources.get(&(binding.group, binding.binding)).copied() {
                    Some(ReflectedResourceBindingKind::ExternalTexture) => {
                        let params_slot = binding.ext_params_buffer_slot.ok_or_else(|| {
                            "MSL external texture binding is missing its params buffer slot"
                                .to_owned()
                        })?;
                        bindings
                            .external_texture
                            .push(yawgpu_tint::ExternalTextureRemap {
                                group: binding.group,
                                binding: binding.binding,
                                plane0_slot: binding.metal_index,
                                plane1_slot: binding.metal_index + 1,
                                params_slot,
                            });
                    }
                    Some(_) => {
                        return Err(
                            "MSL external texture binding map entry does not match reflected resource kind"
                                .to_owned(),
                        );
                    }
                    None => {}
                }
            }
        }
    }
    bindings.input_attachment_color_index = subpass_color_slots
        .iter()
        .map(
            |&((group, binding), color_slot)| yawgpu_tint::InputAttachmentColorIndex {
                group,
                binding,
                color_slot,
            },
        )
        .collect();
    Ok(bindings)
}

fn msl_buffer_sizes_slot(binding_map: &MslBindingMap) -> Result<u32, String> {
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
    Ok(next_slot)
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
        | yawgpu_tint::ResourceType::ReadWriteTexelBuffer => {
            Err("tint: unsupported reflected resource binding type".to_owned())
        }
        #[cfg(feature = "tiled")]
        yawgpu_tint::ResourceType::InputAttachment => Ok(input_attachment_binding_kind(binding)),
        #[cfg(not(feature = "tiled"))]
        yawgpu_tint::ResourceType::InputAttachment => {
            Err("tint: unsupported reflected resource binding type".to_owned())
        }
    }
}

#[cfg(feature = "tiled")]
fn input_attachment_binding_kind(
    binding: &yawgpu_tint::ResourceBinding,
) -> ReflectedResourceBindingKind {
    ReflectedResourceBindingKind::InputAttachment {
        sample_kind: sampled_kind(binding.sampled_kind),
        multisampled: binding.resource_type == yawgpu_tint::ResourceType::MultisampledTexture,
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
        yawgpu_tint::TexelFormat::Rgb10A2Uint => "Rgb10a2Uint",
        yawgpu_tint::TexelFormat::Rgb10A2Unorm => "Rgb10a2Unorm",
        yawgpu_tint::TexelFormat::Rg11B10Ufloat => "Rg11b10Ufloat",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texel_format_matches_reflected_storage_texture_format_names() {
        assert_eq!(
            texel_format(yawgpu_tint::TexelFormat::Rgb10A2Uint).unwrap(),
            "Rgb10a2Uint"
        );
        assert_eq!(
            texel_format(yawgpu_tint::TexelFormat::Rgb10A2Unorm).unwrap(),
            "Rgb10a2Unorm"
        );
        assert_eq!(
            texel_format(yawgpu_tint::TexelFormat::Rg11B10Ufloat).unwrap(),
            "Rg11b10Ufloat"
        );
    }

    #[test]
    fn reflects_compute_workgroup_size_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
var<workgroup> data: array<u32, 8>;

@compute @workgroup_size(8, 4, 1)
fn cs() {
  data[0] = 1u;
}
"#,
        )
        .unwrap();

        let reflected = module.compute_workgroup_size("cs").unwrap().unwrap();
        assert_eq!(reflected.literal_size, [8, 4, 1]);
        assert_eq!(reflected.workgroup_storage_size, 32);

        let resolved = module
            .resolved_compute_workgroup_size("cs", &PipelineConstants::default())
            .unwrap();
        assert_eq!(resolved.literal_size, [8, 4, 1]);
        assert_eq!(resolved.workgroup_storage_size, 32);
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

    #[cfg(feature = "tiled")]
    #[test]
    fn reflects_input_attachment_binding_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
enable chromium_internal_input_attachments;

@group(0) @binding(0) @input_attachment_index(0) var ia: input_attachment<f32>;

@fragment
fn fs() -> @location(0) vec4<f32> {
  return inputAttachmentLoad(ia);
}
"#,
        )
        .unwrap();

        let fragment = module.resource_bindings_for_entry("fs").unwrap();
        assert!(fragment.iter().any(|binding| {
            binding.group == 0
                && binding.binding == 0
                && binding.kind
                    == ReflectedResourceBindingKind::InputAttachment {
                        sample_kind: Some(ReflectedTypeScalarClass::Float),
                        multisampled: false,
                    }
        }));
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn generate_render_fragment_msl_maps_input_attachment_to_non_identity_color_slot() {
        let module = parse_and_validate_wgsl(
            r#"
enable chromium_internal_input_attachments;

@group(0) @binding(0) @input_attachment_index(0) var ia: input_attachment<f32>;

@fragment
fn fs() -> @location(0) vec4<f32> {
  return inputAttachmentLoad(ia);
}
"#,
        )
        .unwrap();

        let generated = module
            .generate_render_fragment_msl(
                "fs",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &[((0, 0), 1)],
                &PipelineConstants::default(),
                0xFFFF_FFFF,
            )
            .unwrap();

        assert!(
            generated
                .source
                .contains("tint_input_attachment_1 [[color(1)]]"),
            "MSL:\n{}",
            generated.source
        );
        assert!(
            !generated
                .source
                .contains("tint_input_attachment_1 [[color(0)]]"),
            "MSL:\n{}",
            generated.source
        );

        let err = module
            .generate_render_fragment_msl(
                "fs",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &[],
                &PipelineConstants::default(),
                0xFFFF_FFFF,
            )
            .expect_err("input attachment MSL generation should require a color-slot map");
        assert!(!err.is_empty());
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

    #[cfg(feature = "tiled")]
    #[test]
    fn reflects_fragment_color_inputs_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
enable chromium_experimental_framebuffer_fetch;

@fragment
fn fs(@color(1) prev1: vec4<f32>, @color(0) prev0: vec4<f32>) -> @location(0) vec4<f32> {
  return prev0 + prev1;
}
"#,
        )
        .unwrap();

        assert_eq!(module.fragment_color_inputs("fs"), vec![0, 1]);
    }

    #[test]
    fn reflects_fragment_blend_src_1_output_from_tint() {
        let module = parse_and_validate_wgsl_gated(
            r#"
enable dual_source_blending;

struct DualSourceOut {
  @location(0) @blend_src(0) a: vec4f,
  @location(0) @blend_src(1) b: vec4f,
}

@fragment
fn fs_dual() -> DualSourceOut {
  return DualSourceOut(vec4f(), vec4f());
}

@fragment
fn fs_plain() -> @location(0) vec4f {
  return vec4f();
}
"#,
            false,
            false,
            true,
            false,
            false,
        )
        .unwrap();

        assert!(module.fragment_writes_blend_src_1("fs_dual"));
        assert!(!module.fragment_writes_blend_src_1("fs_plain"));
    }

    #[test]
    fn parse_and_validate_wgsl_gates_clip_distances_extension() {
        let source = r#"
enable clip_distances;

struct Out {
  @builtin(position) pos: vec4f,
  @builtin(clip_distances) clip: array<f32, 1>,
}

@vertex
fn main() -> Out {
  return Out(vec4f(), array<f32, 1>(0.0));
}
"#;

        assert!(parse_and_validate_wgsl_gated(source, true, true, true, false, false).is_err());
        let module = parse_and_validate_wgsl_gated(source, true, true, true, true, false).unwrap();
        assert_eq!(module.vertex_clip_distances_size("main"), 1);
    }

    #[test]
    fn parse_and_validate_wgsl_gates_primitive_index_extension() {
        let source = r#"
enable primitive_index;

@fragment
fn main(@builtin(primitive_index) idx: u32) -> @location(0) vec4f {
  return vec4f(f32(idx), 0.0, 0.0, 1.0);
}
"#;

        assert!(parse_and_validate_wgsl_gated(source, true, true, true, true, false).is_err());
        assert!(parse_and_validate_wgsl_gated(source, true, true, true, true, true).is_ok());
    }

    #[test]
    fn generate_compute_msl_from_tint() {
        let module = parse_and_validate_wgsl(
            r#"
@compute @workgroup_size(1)
fn main() {}
"#,
        )
        .unwrap();

        let generated = module
            .generate_msl(
                "main",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &PipelineConstants::default(),
            )
            .unwrap();

        assert_eq!(generated.entry_point, "tint_main");
        assert!(generated.source.contains("kernel void tint_main"));
        assert_eq!(generated.frag_depth_clamp_slot, None);
    }

    #[test]
    fn generate_msl_carries_buffer_sizes_for_runtime_storage_array() {
        let module = parse_and_validate_wgsl(
            r#"
struct Data {
  values: array<u32>,
}

@group(0) @binding(0) var<storage, read_write> data: Data;

@compute @workgroup_size(1)
fn cs() {
  if (arrayLength(&data.values) > 0u) {
    data.values[0] = arrayLength(&data.values);
  }
}
"#,
        )
        .unwrap();

        let generated = module
            .generate_msl(
                "cs",
                &MslBindingMap {
                    resources: vec![MslResourceBinding {
                        group: 0,
                        binding: 0,
                        metal_index: 3,
                        ext_params_buffer_slot: None,
                        kind: MslResourceBindingKind::Buffer,
                    }],
                },
                &PipelineConstants::default(),
            )
            .unwrap();

        assert_eq!(generated.buffer_sizes_slot, Some(4));
        assert_eq!(
            generated.buffer_size_bindings,
            vec![MslBufferSizeBinding {
                group: 0,
                binding: 0,
            }]
        );
        assert!(generated
            .source
            .contains("tint_storage_buffer_sizes [[buffer(4)]]"));
    }

    #[test]
    fn generate_render_vertex_msl_reports_buffer_sizes_for_vertex_pulling() {
        let module = parse_and_validate_wgsl(
            r#"
struct VIn {
  @location(0) p: vec4<f32>,
}

@vertex
fn vs(i: VIn) -> @builtin(position) vec4<f32> {
  return i.p;
}
"#,
        )
        .unwrap();

        let vertex_metal_index = 3;
        let generated = module
            .generate_render_vertex_msl(
                "vs",
                &MslBindingMap {
                    resources: Vec::new(),
                },
                &[MslVertexBufferBinding {
                    slot: 0,
                    metal_index: vertex_metal_index,
                    array_stride: 16,
                    step_mode: MslVertexStepMode::Vertex,
                    attributes: vec![MslVertexAttribute {
                        shader_location: 0,
                        offset: 0,
                        format: MslVertexFormat::Float32x4,
                    }],
                }],
                false,
                &PipelineConstants::default(),
            )
            .unwrap();

        assert!(generated.buffer_sizes_slot.is_some());
        assert!(generated.buffer_sizes_slot.unwrap() > vertex_metal_index);
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
            0,
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
                0,
                false,
            )
            .unwrap();
        let fragment = render
            .generate_spirv(
                "fs",
                ShaderStage::Fragment,
                &PipelineConstants::default(),
                false,
                0,
                false,
            )
            .unwrap();
        assert_eq!(vertex.first().copied(), Some(0x0723_0203));
        assert_eq!(fragment.first().copied(), Some(0x0723_0203));
    }

    #[test]
    fn generate_spirv_keeps_robustness_with_vulkan_memory_model_toggle() {
        let module = parse_and_validate_wgsl(
            r#"
struct Data {
  values: array<u32>,
}

@group(0) @binding(0) var<storage, read_write> data: Data;

@compute @workgroup_size(1)
fn cs() {
  data.values[0] = data.values[0] + 1u;
}
"#,
        )
        .unwrap();

        for vulkan_memory_model in [false, true] {
            let spirv = module
                .generate_spirv(
                    "cs",
                    ShaderStage::Compute,
                    &PipelineConstants::default(),
                    vulkan_memory_model,
                    0,
                    false,
                )
                .unwrap();
            assert_eq!(spirv.first().copied(), Some(0x0723_0203));
        }
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

    #[test]
    fn generate_msl_skips_unused_layout_bindings() {
        let module = parse_and_validate_wgsl(
            r#"
struct U {
  x: u32,
}

struct Unused {
  x: u32,
}

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> unused: Unused;

@compute @workgroup_size(1)
fn cs() {
  _ = u.x;
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
                    ],
                },
                &PipelineConstants::default(),
            )
            .unwrap();

        assert!(generated.source.contains("[[buffer(4)]]"));
        assert!(!generated.source.contains("[[buffer(7)]]"));
    }

    #[test]
    fn supported_wgsl_language_features_allow_packed_4x8_and_linear_indexing() {
        if !yawgpu_tint::HAVE_TINT {
            return;
        }

        parse_and_validate_wgsl(
            r#"
requires packed_4x8_integer_dot_product;
requires linear_indexing;

@compute @workgroup_size(1)
fn cs(@builtin(global_invocation_index) index: u32) {
  let packed = dot4I8Packed(1u, 2u);
  _ = packed;
  _ = index;
}
"#,
        )
        .expect("packed_4x8 and linear_indexing are in yawgpu's supported WGSL set");
    }

    #[test]
    fn supported_wgsl_language_features_allow_subgroup_language_features() {
        if !yawgpu_tint::HAVE_TINT {
            return;
        }

        parse_and_validate_wgsl(
            r#"
requires subgroup_id;
requires subgroup_uniformity;

@compute @workgroup_size(1)
fn cs() {}
"#,
        )
        .expect("subgroup_id and subgroup_uniformity are in yawgpu's supported WGSL set");
    }
}
