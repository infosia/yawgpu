use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use yawgpu_hal::{
    HalBackend, HalBufferBindingKind, HalComputePipeline, HalDescriptorBinding,
    HalDescriptorBindingKind, HalDevice, HalMslBufferSizeBinding, HalShaderSource,
    HalStorageTextureAccess,
};

use crate::bind_group_layout::*;
use crate::device::FeatureSet;
use crate::format::*;
use crate::frontend;
use crate::limits::*;
use crate::pipeline_id::next_pipeline_id;
use crate::pipeline_layout::*;
use crate::shader::*;
use crate::texture_view::*;
#[cfg(feature = "shader-passthrough")]
use crate::ShaderStage;

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
    pub(crate) _bindings: Vec<frontend::ReflectedResourceBinding>,
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
    /// Per-kind Metal slot used for compute pipelines and as fallback for render
    /// pipelines when both `vertex_metal_index` and `fragment_metal_index` are
    /// `None` (i.e. the binding is visible to a single well-known stage).
    /// Buffer-space for `Buffer`; texture-space base for
    /// `Texture`/`StorageTexture`/`ExternalTexture`; sampler-space for `Sampler`.
    pub(crate) metal_index: u32,
    /// For `ExternalTexture` only: the buffer-space slot reserved for the
    /// external-texture params buffer.  `None` for all other binding kinds.
    pub(crate) ext_params_buffer_slot: Option<u32>,
    /// For render-pipeline `ExternalTexture` only: the vertex-stage
    /// buffer-space slot reserved for the external-texture params buffer.
    pub(crate) ext_params_vertex_buffer_slot: Option<u32>,
    /// For render-pipeline `ExternalTexture` only: the fragment-stage
    /// buffer-space slot reserved for the external-texture params buffer.
    pub(crate) ext_params_fragment_buffer_slot: Option<u32>,
    /// For render pipelines: per-kind slot in the vertex stage's index space.
    /// `None` when the binding is not visible to the vertex stage.
    pub(crate) vertex_metal_index: Option<u32>,
    /// For render pipelines: per-kind slot in the fragment stage's index space.
    /// `None` when the binding is not visible to the fragment stage.
    pub(crate) fragment_metal_index: Option<u32>,
    pub(crate) kind: MetalBindingKind,
}

/// Stores the shader resource kind for a resolved Metal binding slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetalBindingKind {
    /// Buffer binding.
    Buffer(BufferBindingType),
    /// Sampled texture binding.
    Texture,
    /// Storage texture binding.
    StorageTexture { access: StorageTextureAccess },
    /// Sampler binding.
    Sampler,
    /// External texture binding.
    ExternalTexture,
}

/// Returns the device-independent rejection message when any binding is an
/// external texture, which the Vulkan (SPIR-V) backend does not support.
///
/// yawgpu matches wgpu's posture: external textures are implemented on Metal
/// only. Tint *can* emit SPIR-V for a `texture_external` binding, but the
/// generated module relies on the multiplanar-external-texture transform's
/// expanded bindings (plane textures + params buffer) which yawgpu does not set
/// up for the Vulkan backend. Whether the resulting SPIR-V is accepted then
/// depends on the driver (NVIDIA compiles it, Mesa rejects it), so the rejection
/// must happen here — before any SPIR-V reaches a driver — to be deterministic
/// across GPUs. The descriptor itself is valid WebGPU, so this is a backend
/// (`Internal`) error, never a validation error.
pub(crate) fn vulkan_external_texture_rejection(
    metal_bindings: &[MetalBufferBinding],
) -> Option<String> {
    metal_bindings
        .iter()
        .any(|binding| matches!(binding.kind, MetalBindingKind::ExternalTexture))
        .then(|| "external textures are not supported on the Vulkan backend".to_owned())
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
        let pipeline_id = next_pipeline_id();
        let resolved = if is_error {
            None
        } else {
            resolve_compute_pipeline_descriptor_for_source(
                &descriptor,
                limits,
                features,
                pipeline_id,
            )
            .ok()
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
                &descriptor.constants,
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
    Vec<frontend::ReflectedResourceBinding>,
    Option<ResolvedComputeWorkgroup>,
    Vec<Arc<BindGroupLayout>>,
);

fn resolve_compute_pipeline_descriptor_for_source(
    descriptor: &ComputePipelineDescriptor,
    limits: Limits,
    features: &FeatureSet,
    pipeline_id: u64,
) -> Result<ResolvedPipelineParts, String> {
    #[cfg(feature = "shader-passthrough")]
    if descriptor.shader_module.spirv_passthrough().is_some() {
        return resolve_spirv_passthrough_compute_pipeline_descriptor(descriptor);
    }
    #[cfg(feature = "shader-passthrough")]
    if let Some((_source, entries)) = descriptor.shader_module.msl_passthrough() {
        return resolve_msl_passthrough_compute_pipeline_descriptor(descriptor, entries);
    }
    resolve_compute_pipeline_descriptor(descriptor, limits, features, pipeline_id)
}

/// Creates HAL compute pipeline and reports validation errors through the owning device.
pub(crate) fn create_hal_compute_pipeline(
    hal_device: Option<&HalDevice>,
    shader_module: &ShaderModule,
    entry_name: &str,
    constants: &[PipelineConstant],
    workgroup: Option<ResolvedComputeWorkgroup>,
    metal_bindings: &[MetalBufferBinding],
) -> (Option<HalComputePipeline>, Option<String>) {
    let Some(hal_device) = hal_device else {
        return (None, None);
    };
    if matches!(hal_device.backend(), HalBackend::Noop) {
        return (Some(HalComputePipeline::Noop), None);
    }
    // Validate Metal slot ranges up front so the Metal compiler never sees an
    // out-of-range slot (Metal rejects these at compile-time with a cryptic
    // message that is hard to trace back to the binding layout).
    if matches!(hal_device.backend(), HalBackend::Metal) {
        if let Err(message) = validate_metal_slot_ranges(metal_bindings) {
            return (None, Some(message));
        }
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
        constants,
        metal_bindings,
        hal_device.vulkan_memory_model(),
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
    constants: &[PipelineConstant],
    metal_bindings: &[MetalBufferBinding],
    vulkan_memory_model: bool,
) -> Result<(HalShaderSource, String, Vec<HalDescriptorBinding>), String> {
    let pipeline_constants = pipeline_constant_map(constants);
    match backend {
        HalBackend::Metal => {
            #[cfg(feature = "shader-passthrough")]
            if shader_module.spirv_passthrough().is_some() {
                return Err("SPIR-V passthrough shader requires the Vulkan backend".to_owned());
            }
            #[cfg(feature = "shader-passthrough")]
            if let Some((source, _entries)) = shader_module.msl_passthrough() {
                if !constants.is_empty() {
                    return Err(
                        "pipeline-overridable constants are not supported with shader passthrough"
                            .to_owned(),
                    );
                }
                return Ok((
                    HalShaderSource::Msl(source.to_owned()),
                    entry_name.to_owned(),
                    Vec::new(),
                ));
            }
            let module = shader_module
                .reflected()
                .ok_or_else(|| "compute pipeline requires a reflected shader module".to_owned())?;
            let msl_binding_map = frontend::MslBindingMap {
                resources: msl_resource_bindings(metal_bindings),
            };
            let generated =
                module.generate_msl(entry_name, &msl_binding_map, &pipeline_constants)?;
            Ok((
                HalShaderSource::MslWithBufferSizes {
                    source: generated.source,
                    buffer_sizes_slot: generated.buffer_sizes_slot,
                    buffer_size_bindings: hal_msl_buffer_size_bindings(
                        &generated.buffer_size_bindings,
                    ),
                    workgroup_memory_sizes: generated.workgroup_memory_sizes,
                },
                generated.entry_point,
                Vec::new(),
            ))
        }
        HalBackend::Vulkan => {
            #[cfg(feature = "shader-passthrough")]
            if let Some(words) = shader_module.spirv_passthrough() {
                if !constants.is_empty() {
                    return Err(
                        "pipeline-overridable constants are not supported with shader passthrough"
                            .to_owned(),
                    );
                }
                return Ok((
                    HalShaderSource::SpirV(words.to_vec()),
                    entry_name.to_owned(),
                    hal_descriptor_bindings(metal_bindings),
                ));
            }
            #[cfg(feature = "shader-passthrough")]
            if shader_module.msl_passthrough().is_some() {
                return Err("MSL passthrough shader requires the Metal backend".to_owned());
            }
            if let Some(message) = vulkan_external_texture_rejection(metal_bindings) {
                return Err(message);
            }
            let module = shader_module
                .reflected()
                .ok_or_else(|| "compute pipeline requires a reflected shader module".to_owned())?;
            let spirv = module.generate_spirv(
                entry_name,
                frontend::ShaderStage::Compute,
                &pipeline_constants,
                vulkan_memory_model,
                0,
                false,
            )?;
            Ok((
                HalShaderSource::SpirV(spirv),
                entry_name.to_owned(),
                hal_descriptor_bindings(metal_bindings),
            ))
        }
        #[cfg(feature = "gles")]
        HalBackend::Gles => {
            let module = shader_module
                .reflected()
                .ok_or_else(|| "compute pipeline requires a reflected shader module".to_owned())?;
            let generated = module.generate_glsl(
                entry_name,
                frontend::ShaderStage::Compute,
                &pipeline_constants,
            )?;
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

/// Returns frontend pipeline constants keyed the same way as WebGPU constant entries.
pub(crate) fn pipeline_constant_map(constants: &[PipelineConstant]) -> frontend::PipelineConstants {
    frontend::PipelineConstants::from_iter(
        constants
            .iter()
            .map(|constant| (constant.key.clone(), constant.value)),
    )
}

/// Returns HAL MSL buffer-size bindings.
pub(crate) fn hal_msl_buffer_size_bindings(
    bindings: &[frontend::MslBufferSizeBinding],
) -> Vec<HalMslBufferSizeBinding> {
    bindings
        .iter()
        .map(|binding| HalMslBufferSizeBinding::new(binding.group, binding.binding))
        .collect()
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
                MetalBindingKind::StorageTexture { access } => {
                    HalDescriptorBindingKind::StorageTexture {
                        access: hal_storage_texture_access(access),
                    }
                }
                MetalBindingKind::Sampler => HalDescriptorBindingKind::Sampler,
                MetalBindingKind::ExternalTexture => HalDescriptorBindingKind::Texture,
            },
        })
        .collect()
}

/// Converts a storage texture access mode into the corresponding HAL value.
pub(crate) fn hal_storage_texture_access(access: StorageTextureAccess) -> HalStorageTextureAccess {
    match access {
        StorageTextureAccess::ReadOnly => HalStorageTextureAccess::ReadOnly,
        StorageTextureAccess::WriteOnly => HalStorageTextureAccess::WriteOnly,
        StorageTextureAccess::ReadWrite => HalStorageTextureAccess::ReadWrite,
    }
}

/// Returns MSL resource bindings.
pub(crate) fn msl_resource_bindings(
    bindings: &[MetalBufferBinding],
) -> Vec<frontend::MslResourceBinding> {
    bindings
        .iter()
        .map(|binding| frontend::MslResourceBinding {
            group: binding.group,
            binding: binding.binding,
            metal_index: binding.metal_index,
            ext_params_buffer_slot: binding.ext_params_buffer_slot,
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
        .collect()
}

/// Returns metal buffer binding map.
///
/// For compute pipelines all entries are included in one map with per-kind
/// counters (buffer-space / texture-space / sampler-space are independent).
/// For render pipelines the `visibility` of each layout entry is used to build
/// per-stage per-kind counters; the flat `metal_index` field holds the
/// vertex-stage index (matching the legacy behaviour for single-stage entries)
/// and `vertex_metal_index`/`fragment_metal_index` carry the independent
/// per-stage slot when both stages are present.
pub(crate) fn metal_buffer_binding_map(
    layouts: &[Arc<BindGroupLayout>],
) -> Vec<MetalBufferBinding> {
    // Collect raw entries sorted by (group, binding).
    let mut raw: Vec<(u32, u32, u64, MetalBindingKind)> = Vec::new();
    for (group_index, layout) in layouts.iter().enumerate() {
        let Ok(group) = u32::try_from(group_index) else {
            break;
        };
        for entry in layout.entries() {
            let kind = match entry.kind {
                Some(BindingLayoutKind::Buffer { ty, .. }) => MetalBindingKind::Buffer(ty),
                Some(BindingLayoutKind::Texture { .. }) => MetalBindingKind::Texture,
                Some(BindingLayoutKind::StorageTexture { access, .. }) => {
                    MetalBindingKind::StorageTexture { access }
                }
                Some(BindingLayoutKind::Sampler { .. }) => MetalBindingKind::Sampler,
                Some(BindingLayoutKind::ExternalTexture) => MetalBindingKind::ExternalTexture,
                _ => continue,
            };
            raw.push((group, entry.binding, entry.visibility, kind));
        }
    }
    raw.sort_by_key(|&(group, binding, _, _)| (group, binding));

    // Determine whether this is a render layout (any entry has vertex or
    // fragment visibility) or a compute layout.
    let is_render = raw
        .iter()
        .any(|&(_, _, vis, _)| vis & (SHADER_STAGE_VERTEX | SHADER_STAGE_FRAGMENT) != 0);

    let mut bindings = Vec::with_capacity(raw.len());

    if is_render {
        // Per-stage per-kind counters.
        let mut vtx_buf = 0u32;
        let mut vtx_tex = 0u32;
        let mut vtx_smp = 0u32;
        let mut frag_buf = 0u32;
        let mut frag_tex = 0u32;
        let mut frag_smp = 0u32;

        for (group, binding, visibility, kind) in raw {
            let in_vtx = visibility & SHADER_STAGE_VERTEX != 0;
            let in_frag = visibility & SHADER_STAGE_FRAGMENT != 0;

            // Assign per-stage slots and advance counters.
            let (
                vertex_metal_index,
                fragment_metal_index,
                ext_params_buffer_slot,
                ext_params_vertex_buffer_slot,
                ext_params_fragment_buffer_slot,
            ) = match kind {
                MetalBindingKind::Buffer(_) => {
                    let vi = if in_vtx {
                        let s = vtx_buf;
                        vtx_buf = vtx_buf.saturating_add(1);
                        Some(s)
                    } else {
                        None
                    };
                    let fi = if in_frag {
                        let s = frag_buf;
                        frag_buf = frag_buf.saturating_add(1);
                        Some(s)
                    } else {
                        None
                    };
                    (vi, fi, None, None, None)
                }
                MetalBindingKind::Texture | MetalBindingKind::StorageTexture { .. } => {
                    let vi = if in_vtx {
                        let s = vtx_tex;
                        vtx_tex = vtx_tex.saturating_add(1);
                        Some(s)
                    } else {
                        None
                    };
                    let fi = if in_frag {
                        let s = frag_tex;
                        frag_tex = frag_tex.saturating_add(1);
                        Some(s)
                    } else {
                        None
                    };
                    (vi, fi, None, None, None)
                }
                MetalBindingKind::Sampler => {
                    let vi = if in_vtx {
                        let s = vtx_smp;
                        vtx_smp = vtx_smp.saturating_add(1);
                        Some(s)
                    } else {
                        None
                    };
                    let fi = if in_frag {
                        let s = frag_smp;
                        frag_smp = frag_smp.saturating_add(1);
                        Some(s)
                    } else {
                        None
                    };
                    (vi, fi, None, None, None)
                }
                MetalBindingKind::ExternalTexture => {
                    // 2 plane textures + 1 params buffer per stage.
                    let vi_tex = if in_vtx {
                        let s = vtx_tex;
                        vtx_tex = vtx_tex.saturating_add(2);
                        Some(s)
                    } else {
                        None
                    };
                    let fi_tex = if in_frag {
                        let s = frag_tex;
                        frag_tex = frag_tex.saturating_add(2);
                        Some(s)
                    } else {
                        None
                    };
                    let vi_buf = if in_vtx {
                        let s = vtx_buf;
                        vtx_buf = vtx_buf.saturating_add(1);
                        Some(s)
                    } else {
                        None
                    };
                    let fi_buf = if in_frag {
                        let s = frag_buf;
                        frag_buf = frag_buf.saturating_add(1);
                        Some(s)
                    } else {
                        None
                    };
                    // The flat slots keep compute/fallback callers working;
                    // render codegen and binding use the per-stage slots.
                    (vi_tex, fi_tex, vi_buf.or(fi_buf), vi_buf, fi_buf)
                }
            };

            // Flat `metal_index`: use vertex index when available, then fragment.
            let metal_index = vertex_metal_index.or(fragment_metal_index).unwrap_or(0);

            bindings.push(MetalBufferBinding {
                group,
                binding,
                metal_index,
                ext_params_buffer_slot,
                ext_params_vertex_buffer_slot,
                ext_params_fragment_buffer_slot,
                vertex_metal_index,
                fragment_metal_index,
                kind,
            });
        }
    } else {
        // Compute (or no-visibility) layout: one flat map with per-kind counters.
        let mut buf_idx = 0u32;
        let mut tex_idx = 0u32;
        let mut smp_idx = 0u32;

        for (group, binding, _, kind) in raw {
            let (metal_index, ext_params_buffer_slot) = match kind {
                MetalBindingKind::Buffer(_) => {
                    let s = buf_idx;
                    buf_idx = buf_idx.saturating_add(1);
                    (s, None)
                }
                MetalBindingKind::Texture | MetalBindingKind::StorageTexture { .. } => {
                    let s = tex_idx;
                    tex_idx = tex_idx.saturating_add(1);
                    (s, None)
                }
                MetalBindingKind::Sampler => {
                    let s = smp_idx;
                    smp_idx = smp_idx.saturating_add(1);
                    (s, None)
                }
                MetalBindingKind::ExternalTexture => {
                    // 2 consecutive texture slots + 1 buffer slot.
                    let tex_base = tex_idx;
                    tex_idx = tex_idx.saturating_add(2);
                    let buf_slot = buf_idx;
                    buf_idx = buf_idx.saturating_add(1);
                    (tex_base, Some(buf_slot))
                }
            };
            bindings.push(MetalBufferBinding {
                group,
                binding,
                metal_index,
                ext_params_buffer_slot,
                ext_params_vertex_buffer_slot: None,
                ext_params_fragment_buffer_slot: None,
                vertex_metal_index: None,
                fragment_metal_index: None,
                kind,
            });
        }
    }

    bindings
}

/// Returns the number of buffer-space slots consumed by the vertex stage of
/// a render pipeline binding map.  Used to place vertex-buffer slots
/// immediately after the bind-group buffer slots in the same `[[buffer(N)]]`
/// index space.
pub(crate) fn vertex_stage_buffer_count(metal_bindings: &[MetalBufferBinding]) -> usize {
    // Count distinct buffer-space slots used by the vertex stage.
    // For ExternalTexture the params buffer occupies the buffer space too.
    let max_slot = metal_bindings
        .iter()
        .filter_map(|b| {
            b.vertex_metal_index.and_then(|_| {
                // The vertex buffer-space slot is:
                //   - `vertex_metal_index` for Buffer bindings
                //   - `ext_params_vertex_buffer_slot` for ExternalTexture
                //     (vertex_metal_index is texture-space here)
                match b.kind {
                    MetalBindingKind::Buffer(_) => b.vertex_metal_index.map(|s| s + 1),
                    MetalBindingKind::ExternalTexture => {
                        b.ext_params_vertex_buffer_slot.map(|s| s + 1)
                    }
                    _ => None,
                }
            })
        })
        .max()
        .unwrap_or(0);
    usize::try_from(max_slot).unwrap_or(0)
}

/// Validates Metal slot assignments for a binding map and returns an error
/// if any slot index exceeds the hardware limit.
///
/// Metal limits: `[[buffer(N)]]` slots 0–30, `[[texture(N)]]` and
/// `[[sampler(N)]]` slots 0–15.
pub(crate) fn validate_metal_slot_ranges(
    metal_bindings: &[MetalBufferBinding],
) -> Result<(), String> {
    const MAX_BUFFER_SLOT: u32 = 30;
    // Metal's texture argument table has at least 31 entries (indices 0-30)
    // on every WebGPU-capable device; only the sampler table is capped at 16
    // entries (indices 0-15). Review fix: an earlier draft wrongly applied the
    // sampler cap to textures, rejecting valid max-bindings pipelines (F-077).
    const MAX_TEXTURE_SLOT: u32 = 30;
    const MAX_SAMPLER_SLOT: u32 = 15;

    for binding in metal_bindings {
        // Check flat + per-stage indices for each kind.
        let check_buf = |slot: u32| -> Result<(), String> {
            if slot > MAX_BUFFER_SLOT {
                Err(format!(
                    "Metal buffer slot {slot} exceeds the maximum allowed slot ({MAX_BUFFER_SLOT})"
                ))
            } else {
                Ok(())
            }
        };
        let check_tex = |slot: u32| -> Result<(), String> {
            if slot > MAX_TEXTURE_SLOT {
                Err(format!(
                    "Metal texture slot {slot} exceeds the maximum allowed slot ({MAX_TEXTURE_SLOT})"
                ))
            } else {
                Ok(())
            }
        };
        let check_smp = |slot: u32| -> Result<(), String> {
            if slot > MAX_SAMPLER_SLOT {
                Err(format!(
                    "Metal sampler slot {slot} exceeds the maximum allowed slot ({MAX_SAMPLER_SLOT})"
                ))
            } else {
                Ok(())
            }
        };

        match binding.kind {
            MetalBindingKind::Buffer(_) => {
                check_buf(binding.metal_index)?;
                if let Some(s) = binding.vertex_metal_index {
                    check_buf(s)?;
                }
                if let Some(s) = binding.fragment_metal_index {
                    check_buf(s)?;
                }
            }
            MetalBindingKind::Texture | MetalBindingKind::StorageTexture { .. } => {
                check_tex(binding.metal_index)?;
                if let Some(s) = binding.vertex_metal_index {
                    check_tex(s)?;
                }
                if let Some(s) = binding.fragment_metal_index {
                    check_tex(s)?;
                }
            }
            MetalBindingKind::Sampler => {
                check_smp(binding.metal_index)?;
                if let Some(s) = binding.vertex_metal_index {
                    check_smp(s)?;
                }
                if let Some(s) = binding.fragment_metal_index {
                    check_smp(s)?;
                }
            }
            MetalBindingKind::ExternalTexture => {
                // Planes are in texture-space; check base + 1 for all stages.
                let bases = [binding.metal_index]
                    .into_iter()
                    .chain(binding.vertex_metal_index)
                    .chain(binding.fragment_metal_index);
                for base in bases {
                    check_tex(base.saturating_add(1))?;
                }
                // Params buffer is in buffer-space.
                if let Some(s) = binding.ext_params_buffer_slot {
                    check_buf(s)?;
                }
                if let Some(s) = binding.ext_params_vertex_buffer_slot {
                    check_buf(s)?;
                }
                if let Some(s) = binding.ext_params_fragment_buffer_slot {
                    check_buf(s)?;
                }
            }
        }
    }
    Ok(())
}

/// Validates compute pipeline descriptor and returns a descriptive error on failure.
pub(crate) fn validate_compute_pipeline_descriptor(
    descriptor: &ComputePipelineDescriptor,
    limits: Limits,
    features: &FeatureSet,
) -> Option<String> {
    resolve_compute_pipeline_descriptor_for_source(descriptor, limits, features, 0).err()
}

/// Records resolve into the command stream.
pub(crate) fn resolve_compute_pipeline_descriptor(
    descriptor: &ComputePipelineDescriptor,
    limits: Limits,
    features: &FeatureSet,
    pipeline_id: u64,
) -> Result<ResolvedPipelineParts, String> {
    if descriptor.shader_module.is_error() {
        return Err("compute pipeline shader module must not be an error module".to_owned());
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
    let bind_group_layouts = effective_compute_bind_group_layouts(
        &descriptor.layout,
        &bindings,
        limits,
        features,
        pipeline_id,
    )?;
    Ok((entry_name, bindings, Some(workgroup), bind_group_layouts))
}

/// Records resolve into the command stream for raw MSL passthrough compute.
#[cfg(feature = "shader-passthrough")]
pub(crate) fn resolve_msl_passthrough_compute_pipeline_descriptor(
    descriptor: &ComputePipelineDescriptor,
    entries: &[MslEntryPoint],
) -> Result<ResolvedPipelineParts, String> {
    if descriptor.shader_module.is_error() {
        return Err("compute pipeline shader module must not be an error module".to_owned());
    }
    if !descriptor.constants.is_empty() {
        return Err(
            "pipeline-overridable constants are not supported with shader passthrough".to_owned(),
        );
    }
    let ComputePipelineLayout::Explicit(layout) = &descriptor.layout else {
        return Err("shader passthrough requires an explicit pipeline layout".to_owned());
    };
    if layout.is_error() {
        return Err("compute pipeline layout must not be an error pipeline layout".to_owned());
    }
    let Some(entry_name) = descriptor
        .entry_point
        .as_deref()
        .filter(|name| !name.is_empty())
    else {
        return Err(
            "MSL passthrough compute entry point not found or missing workgroup size".to_owned(),
        );
    };
    let Some(entry) = entries
        .iter()
        .find(|entry| entry.name == entry_name && entry.stage == ShaderStage::Compute)
    else {
        return Err(
            "MSL passthrough compute entry point not found or missing workgroup size".to_owned(),
        );
    };
    if entry.workgroup_size.contains(&0) {
        return Err(
            "MSL passthrough compute entry point not found or missing workgroup size".to_owned(),
        );
    }
    Ok((
        entry_name.to_owned(),
        Vec::new(),
        Some(ResolvedComputeWorkgroup {
            size: entry.workgroup_size,
            storage_size: 0,
        }),
        layout.bind_group_layouts().to_vec(),
    ))
}

/// Records resolve into the command stream for raw SPIR-V passthrough compute.
#[cfg(feature = "shader-passthrough")]
pub(crate) fn resolve_spirv_passthrough_compute_pipeline_descriptor(
    descriptor: &ComputePipelineDescriptor,
) -> Result<ResolvedPipelineParts, String> {
    if descriptor.shader_module.is_error() {
        return Err("compute pipeline shader module must not be an error module".to_owned());
    }
    if !descriptor.constants.is_empty() {
        return Err(
            "pipeline-overridable constants are not supported with shader passthrough".to_owned(),
        );
    }
    let ComputePipelineLayout::Explicit(layout) = &descriptor.layout else {
        return Err("shader passthrough requires an explicit pipeline layout".to_owned());
    };
    if layout.is_error() {
        return Err("compute pipeline layout must not be an error pipeline layout".to_owned());
    }
    let Some(entry_name) = descriptor
        .entry_point
        .as_deref()
        .filter(|name| !name.is_empty())
    else {
        return Err("SPIR-V passthrough compute entry point is required".to_owned());
    };
    Ok((
        entry_name.to_owned(),
        Vec::new(),
        Some(ResolvedComputeWorkgroup {
            size: [1, 1, 1],
            storage_size: 0,
        }),
        layout.bind_group_layouts().to_vec(),
    ))
}

/// Records resolve into the command stream.
pub(crate) fn resolve_compute_entry(
    module: &frontend::ReflectedModule,
    entry_point: Option<&str>,
) -> Result<String, String> {
    let entries = module.entry_points();
    let compute_entries = entries
        .iter()
        .filter(|entry| entry.stage == frontend::ReflectedShaderStage::Compute)
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
    overrides: &[frontend::ReflectedOverride],
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
    overrides: &[frontend::ReflectedOverride],
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
    override_: &frontend::ReflectedOverride,
    value: f64,
) -> Result<(), String> {
    if !value.is_finite() {
        return Err("pipeline constant value must be finite".to_owned());
    }
    if override_.ty.components != 1 {
        return Err("pipeline override constants must be scalar".to_owned());
    }

    match override_.ty.scalar {
        frontend::ReflectedTypeScalarClass::Float => {
            let max = if override_.ty.width == 2 {
                65_504.0
            } else {
                f64::from(f32::MAX)
            };
            if value.abs() > max {
                return Err("pipeline constant value is outside the override type range".to_owned());
            }
        }
        frontend::ReflectedTypeScalarClass::Sint => {
            if value.fract() != 0.0 || value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
                return Err("pipeline constant value is outside the override type range".to_owned());
            }
        }
        frontend::ReflectedTypeScalarClass::Uint => {
            if value.fract() != 0.0 || value < 0.0 || value > f64::from(u32::MAX) {
                return Err("pipeline constant value is outside the override type range".to_owned());
            }
        }
        frontend::ReflectedTypeScalarClass::Bool => {}
    }
    Ok(())
}

/// Records resolve into the command stream.
pub(crate) fn resolve_compute_workgroup(
    module: &frontend::ReflectedModule,
    entry_name: &str,
    constants: &[ResolvedOverrideConstant],
    limits: Limits,
) -> Result<ResolvedComputeWorkgroup, String> {
    let pipeline_constants = resolved_pipeline_constant_map(constants);
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

fn resolved_pipeline_constant_map(
    constants: &[ResolvedOverrideConstant],
) -> frontend::PipelineConstants {
    frontend::PipelineConstants::from_iter(
        constants
            .iter()
            .map(|constant| (constant.key.clone(), constant.value)),
    )
}

/// Validates compute pipeline layout and returns a descriptive error on failure.
pub(crate) fn validate_compute_pipeline_layout(
    layout: &ComputePipelineLayout,
    bindings: &[frontend::ReflectedResourceBinding],
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
    bindings: &[frontend::ReflectedResourceBinding],
    limits: Limits,
    features: &FeatureSet,
    pipeline_id: u64,
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
            pipeline_id,
        ),
    }
}

/// Stores binding metadata.
#[derive(Debug, Clone)]
pub(crate) struct StageResourceBinding {
    pub(crate) stage: PipelineShaderStage,
    pub(crate) binding: frontend::ReflectedResourceBinding,
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
        let frontend::ReflectedResourceBindingKind::Texture {
            sampled,
            sample_kind,
            sample_usage: frontend::ReflectedTextureSampleUsage::Gather,
            ..
        } = requirement.binding.kind
        else {
            continue;
        };
        let shader_sample_type = reflected_texture_sample_type(
            sampled,
            sample_kind,
            frontend::ReflectedTextureSampleUsage::Gather,
        )?;

        // The texture is filterable (and textureGather is legal with a
        // Filtering sampler) only when BOTH the shader-reflected type AND the
        // explicit layout entry agree it is filterable.  The WebGPU F-061 rule
        // lets an explicit `UnfilterableFloat` layout accept a shader-reflected
        // `Float` texture — so the shader alone is not authoritative here.
        // We must also check the layout entry: if the layout says
        // `UnfilterableFloat`, the texture is non-filterable regardless of what
        // the shader reflection produces.
        let layout_sample_type = {
            let binding = &requirement.binding;
            let group = usize::try_from(binding.group).ok();
            group
                .and_then(|g| layout.bind_group_layouts().get(g))
                .and_then(|group_layout| {
                    group_layout
                        .entries()
                        .iter()
                        .find(|entry| entry.binding == binding.binding)
                })
                .and_then(|entry| entry.kind)
                .and_then(|kind| {
                    if let BindingLayoutKind::Texture { sample_type, .. } = kind {
                        Some(sample_type)
                    } else {
                        None
                    }
                })
        };

        // Skip validation only when the texture is genuinely filterable: the
        // shader says Float AND the layout says Float (or there is no explicit
        // layout entry — auto-layout, which derives filterable from the shader).
        let is_filterable = shader_sample_type == TextureSampleType::Float
            && layout_sample_type != Some(TextureSampleType::UnfilterableFloat);
        if is_filterable {
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
    pipeline_id: u64,
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
        layouts.push(Arc::new(BindGroupLayout::new_auto(entries, pipeline_id)));
    }

    let mut stage_counts = [StageResourceCounts::default(); 3];
    for layout in &layouts {
        for entry in layout.entries() {
            let Some(kind) = entry.kind else {
                continue;
            };
            for stage in visible_stages(entry.visibility) {
                stage_counts[stage].add(kind);
            }
        }
    }

    for counts in stage_counts {
        if counts.sampled_textures > limits.max_sampled_textures_per_shader_stage {
            return Err(
                "pipeline auto layout uses too many sampled textures for one shader stage"
                    .to_owned(),
            );
        }
        if counts.samplers > limits.max_samplers_per_shader_stage {
            return Err(
                "pipeline auto layout uses too many samplers for one shader stage".to_owned(),
            );
        }
        if counts.uniform_buffers > limits.max_uniform_buffers_per_shader_stage {
            return Err(
                "pipeline auto layout uses too many uniform buffers for one shader stage"
                    .to_owned(),
            );
        }
        if counts.storage_textures > limits.max_storage_textures_per_shader_stage {
            return Err(
                "pipeline auto layout uses too many storage textures for one shader stage"
                    .to_owned(),
            );
        }
        if counts.storage_buffers > limits.max_storage_buffers_per_shader_stage {
            return Err(
                "pipeline auto layout uses too many storage buffers for one shader stage"
                    .to_owned(),
            );
        }
    }
    if stage_counts[0].storage_textures > limits.max_storage_textures_in_vertex_stage {
        return Err(
            "pipeline auto layout uses too many storage textures in the vertex stage".to_owned(),
        );
    }
    if stage_counts[1].storage_textures > limits.max_storage_textures_in_fragment_stage {
        return Err(
            "pipeline auto layout uses too many storage textures in the fragment stage".to_owned(),
        );
    }
    Ok(layouts)
}

/// Returns reflected bind group layout entry.
pub(crate) fn reflected_bind_group_layout_entry(
    binding: &frontend::ReflectedResourceBinding,
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
    binding: &frontend::ReflectedResourceBinding,
) -> Result<BindingLayoutKind, String> {
    match &binding.kind {
        frontend::ReflectedResourceBindingKind::Buffer(ty) => Ok(BindingLayoutKind::Buffer {
            ty: match ty {
                frontend::ReflectedBufferType::Uniform => BufferBindingType::Uniform,
                frontend::ReflectedBufferType::Storage => BufferBindingType::Storage,
                frontend::ReflectedBufferType::ReadOnlyStorage => {
                    BufferBindingType::ReadOnlyStorage
                }
            },
            has_dynamic_offset: false,
            min_binding_size: binding.min_binding_size,
        }),
        frontend::ReflectedResourceBindingKind::Sampler { comparison } => {
            Ok(BindingLayoutKind::Sampler {
                ty: if *comparison {
                    SamplerBindingType::Comparison
                } else {
                    SamplerBindingType::Filtering
                },
            })
        }
        frontend::ReflectedResourceBindingKind::Texture {
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
        frontend::ReflectedResourceBindingKind::StorageTexture {
            format,
            access,
            view_dimension,
        } => Ok(BindingLayoutKind::StorageTexture {
            access: reflected_storage_texture_access(access),
            format: reflected_storage_texture_format(format)?,
            view_dimension: reflected_texture_view_dimension(*view_dimension),
        }),
        #[cfg(feature = "tiled")]
        frontend::ReflectedResourceBindingKind::InputAttachment {
            sample_kind,
            multisampled,
        } => Ok(BindingLayoutKind::InputAttachment {
            sample_type: reflected_input_attachment_sample_type(*sample_kind)?,
            multisampled: *multisampled,
        }),
        frontend::ReflectedResourceBindingKind::ExternalTexture => {
            Ok(BindingLayoutKind::ExternalTexture)
        }
    }
}

#[cfg(feature = "tiled")]
fn reflected_input_attachment_sample_type(
    sample_kind: Option<frontend::ReflectedTypeScalarClass>,
) -> Result<TextureSampleType, String> {
    match sample_kind {
        Some(frontend::ReflectedTypeScalarClass::Float) => Ok(TextureSampleType::Float),
        Some(frontend::ReflectedTypeScalarClass::Sint) => Ok(TextureSampleType::Sint),
        Some(frontend::ReflectedTypeScalarClass::Uint) => Ok(TextureSampleType::Uint),
        Some(frontend::ReflectedTypeScalarClass::Bool) | None => {
            Err("pipeline auto layout input attachment sample type is unsupported".to_owned())
        }
    }
}

/// Returns reflected texture sample type.
pub(crate) fn reflected_texture_sample_type(
    sampled: bool,
    sample_kind: Option<frontend::ReflectedTypeScalarClass>,
    sample_usage: frontend::ReflectedTextureSampleUsage,
) -> Result<TextureSampleType, String> {
    if !sampled {
        return Ok(TextureSampleType::Depth);
    }
    match sample_kind {
        Some(frontend::ReflectedTypeScalarClass::Float) => Ok(match sample_usage {
            frontend::ReflectedTextureSampleUsage::Sample
            | frontend::ReflectedTextureSampleUsage::Gather => TextureSampleType::Float,
            frontend::ReflectedTextureSampleUsage::Load => TextureSampleType::UnfilterableFloat,
        }),
        Some(frontend::ReflectedTypeScalarClass::Sint) => Ok(TextureSampleType::Sint),
        Some(frontend::ReflectedTypeScalarClass::Uint) => Ok(TextureSampleType::Uint),
        _ => Err("pipeline texture binding sample type is unsupported".to_owned()),
    }
}

/// Returns reflected texture view dimension.
pub(crate) fn reflected_texture_view_dimension(
    dimension: frontend::ReflectedTextureViewDimension,
) -> TextureViewDimension {
    match dimension {
        frontend::ReflectedTextureViewDimension::D1 => TextureViewDimension::D1,
        frontend::ReflectedTextureViewDimension::D2 => TextureViewDimension::D2,
        frontend::ReflectedTextureViewDimension::D2Array => TextureViewDimension::D2Array,
        frontend::ReflectedTextureViewDimension::Cube => TextureViewDimension::Cube,
        frontend::ReflectedTextureViewDimension::CubeArray => TextureViewDimension::CubeArray,
        frontend::ReflectedTextureViewDimension::D3 => TextureViewDimension::D3,
    }
}

/// Returns reflected storage texture access.
pub(crate) fn reflected_storage_texture_access(
    access: &frontend::ReflectedStorageTextureAccess,
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
        // `texture-formats-tier1` storage formats — recognised here so the
        // shared `caps.storage_capable` check (feature-gated in
        // `FormatCaps::apply_feature_upgrades`) decides acceptance, instead of
        // rejecting them up front (F-059).
        "R8Unorm" => 0x0000_0001,
        "R8Snorm" => 0x0000_0002,
        "R8Uint" => 0x0000_0003,
        "R8Sint" => 0x0000_0004,
        "R16Uint" => 0x0000_0007,
        "R16Sint" => 0x0000_0008,
        "R16Float" => 0x0000_0009,
        "Rg8Unorm" => 0x0000_000A,
        "Rg8Snorm" => 0x0000_000B,
        "Rg8Uint" => 0x0000_000C,
        "Rg8Sint" => 0x0000_000D,
        "Rg16Uint" => 0x0000_0013,
        "Rg16Sint" => 0x0000_0014,
        "Rg16Float" => 0x0000_0015,
        "Bgra8Unorm" => 0x0000_001B,
        "Rgb10a2Uint" => 0x0000_001D,
        "Rgb10a2Unorm" => 0x0000_001E,
        "Rg11b10Ufloat" => 0x0000_001F,
        // 16-bit-norm storage formats — baseline-storage in WebGPU, gated only by
        // format availability; the shader frontend must accept baseline WebGPU
        // storage formats here.
        // to compile them (F-059).
        "R16Unorm" => 0x0000_0005,
        "R16Snorm" => 0x0000_0006,
        "Rg16Unorm" => 0x0000_0011,
        "Rg16Snorm" => 0x0000_0012,
        "Rgba16Unorm" => 0x0000_0024,
        "Rgba16Snorm" => 0x0000_0025,
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
    binding: &frontend::ReflectedResourceBinding,
    layout_kind: BindingLayoutKind,
) -> Result<(), String> {
    match (&binding.kind, layout_kind) {
        (
            frontend::ReflectedResourceBindingKind::Buffer(shader_ty),
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
            frontend::ReflectedResourceBindingKind::Sampler { .. },
            BindingLayoutKind::Sampler { .. },
        )
        | (
            frontend::ReflectedResourceBindingKind::Texture { .. },
            BindingLayoutKind::Texture { .. },
        )
        | (
            frontend::ReflectedResourceBindingKind::StorageTexture { .. },
            BindingLayoutKind::StorageTexture { .. },
        )
        | (
            frontend::ReflectedResourceBindingKind::ExternalTexture,
            BindingLayoutKind::ExternalTexture,
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
            frontend::ReflectedResourceBindingKind::InputAttachment { .. },
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
    // `expected` is the shader-reflected binding, `actual` is the explicit
    // pipeline-layout binding. Compatibility is NOT exact equality — it mirrors
    // the WebGPU shader↔layout rules (`doResourcesMatch`/`doSampleTypesMatch`/
    // `doAccessesMatch`): a float layout sample type accepts either float shader
    // type, a read-write layout access accepts read-write or write-only shader
    // access, and samplers match unless exactly one is a comparison sampler
    // (F-061). view dimension, multisampled and storage format must still match.
    match (expected, actual) {
        (BindingLayoutKind::Sampler { ty: shader }, BindingLayoutKind::Sampler { ty: layout }) => {
            sampler_types_compatible(layout, shader)
        }
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
            sample_types_compatible(actual_sample_type, sample_type)
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
            storage_accesses_compatible(actual_access, access)
                && format == actual_format
                && view_dimension == actual_view_dimension
        }
        (BindingLayoutKind::ExternalTexture, BindingLayoutKind::ExternalTexture) => true,
        #[cfg(feature = "tiled")]
        (
            BindingLayoutKind::InputAttachment { sample_type, .. },
            BindingLayoutKind::InputAttachment {
                sample_type: actual_sample_type,
                ..
            },
        ) => {
            // Compare only `sample_type`. The shader-reflected `multisampled`
            // for an input attachment is not meaningful under Tint (always
            // `false` — WGSL/`input_attachment<T>` cannot carry multisampled-ness;
            // it is a module-wide SPIR-V generate option). The explicit pipeline
            // layout is therefore the sole authority for input-attachment
            // multisampled-ness. Sample-count consistency against the pass layout
            // is still enforced separately by rule C-1 in
            // `validate_subpass_pipeline_multisampling`, so relaxing the
            // shader↔layout kind check loses no real validation.
            sample_types_compatible(actual_sample_type, sample_type)
        }
        _ => false,
    }
}

/// Mirrors WebGPU `doSampleTypesMatch` (shader↔layout): a float layout sample
/// type (filterable or unfilterable) accepts either float shader sample type;
/// every other sample type must match exactly.
fn sample_types_compatible(layout: TextureSampleType, shader: TextureSampleType) -> bool {
    match layout {
        TextureSampleType::Float | TextureSampleType::UnfilterableFloat => matches!(
            shader,
            TextureSampleType::Float | TextureSampleType::UnfilterableFloat
        ),
        other => other == shader,
    }
}

/// Mirrors WebGPU `doAccessesMatch`: a read-write layout storage access accepts
/// a read-write or write-only shader access; every other access must match
/// exactly.
fn storage_accesses_compatible(layout: StorageTextureAccess, shader: StorageTextureAccess) -> bool {
    match layout {
        StorageTextureAccess::ReadWrite => matches!(
            shader,
            StorageTextureAccess::ReadWrite | StorageTextureAccess::WriteOnly
        ),
        other => other == shader,
    }
}

/// Mirrors WebGPU sampler compatibility: the layout and shader samplers are
/// compatible when they share a type, or when neither is a comparison sampler.
fn sampler_types_compatible(layout: SamplerBindingType, shader: SamplerBindingType) -> bool {
    layout == shader
        || (layout != SamplerBindingType::Comparison && shader != SamplerBindingType::Comparison)
}

/// Returns buffer binding types compatible.
pub(crate) fn buffer_binding_types_compatible(
    shader_ty: frontend::ReflectedBufferType,
    layout_ty: BufferBindingType,
) -> bool {
    matches!(
        (shader_ty, layout_ty),
        (
            frontend::ReflectedBufferType::Uniform,
            BufferBindingType::Uniform
        ) | (
            frontend::ReflectedBufferType::Storage,
            BufferBindingType::Storage
        ) | (
            frontend::ReflectedBufferType::ReadOnlyStorage,
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
        let binding = frontend::ReflectedResourceBinding {
            group: 0,
            binding: 0,
            kind: frontend::ReflectedResourceBindingKind::Buffer(
                frontend::ReflectedBufferType::Uniform,
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

    #[test]
    fn external_texture_reflection_derives_exact_layout_compat() {
        let binding = frontend::ReflectedResourceBinding {
            group: 0,
            binding: 0,
            kind: frontend::ReflectedResourceBindingKind::ExternalTexture,
            min_binding_size: 0,
            statically_used: true,
        };

        assert_eq!(
            reflected_binding_layout_kind(&binding),
            Ok(BindingLayoutKind::ExternalTexture)
        );
        assert_eq!(
            validate_shader_binding_compat(&binding, BindingLayoutKind::ExternalTexture),
            Ok(())
        );
        assert_eq!(
            validate_shader_binding_compat(
                &binding,
                BindingLayoutKind::Texture {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            Err("compute pipeline layout binding type is incompatible".to_owned())
        );
    }

    fn reflected_texture_binding(group: u32, binding: u32) -> StageResourceBinding {
        StageResourceBinding {
            stage: PipelineShaderStage::Fragment,
            binding: frontend::ReflectedResourceBinding {
                group,
                binding,
                kind: frontend::ReflectedResourceBindingKind::Texture {
                    sampled: true,
                    sample_kind: Some(frontend::ReflectedTypeScalarClass::Float),
                    sample_usage: frontend::ReflectedTextureSampleUsage::Sample,
                    view_dimension: frontend::ReflectedTextureViewDimension::D2,
                    multisampled: false,
                },
                min_binding_size: 0,
                statically_used: true,
            },
        }
    }

    fn reflected_storage_texture_binding(group: u32, binding: u32) -> StageResourceBinding {
        StageResourceBinding {
            stage: PipelineShaderStage::Fragment,
            binding: frontend::ReflectedResourceBinding {
                group,
                binding,
                kind: frontend::ReflectedResourceBindingKind::StorageTexture {
                    format: "Rgba8Unorm".to_owned(),
                    access: frontend::ReflectedStorageTextureAccess {
                        read: false,
                        write: true,
                    },
                    view_dimension: frontend::ReflectedTextureViewDimension::D2,
                },
                min_binding_size: 0,
                statically_used: true,
            },
        }
    }

    #[test]
    fn derive_bind_group_layouts_rejects_aggregate_sampled_texture_stage_over_limit() {
        let limits = Limits {
            max_sampled_textures_per_shader_stage: 1,
            ..Limits::DEFAULT
        };
        let requirements = [
            reflected_texture_binding(0, 0),
            reflected_texture_binding(1, 0),
        ];

        assert_eq!(
            derive_bind_group_layouts(requirements, limits, &FeatureSet::default(), 1)
                .expect_err("aggregate sampled texture over-limit should reject auto layout"),
            "pipeline auto layout uses too many sampled textures for one shader stage"
        );
    }

    #[test]
    fn derive_bind_group_layouts_rejects_aggregate_fragment_storage_texture_over_limit() {
        let limits = Limits {
            max_storage_textures_per_shader_stage: 4,
            max_storage_textures_in_fragment_stage: 1,
            ..Limits::DEFAULT
        };
        let requirements = [
            reflected_storage_texture_binding(0, 0),
            reflected_storage_texture_binding(1, 0),
        ];

        assert_eq!(
            derive_bind_group_layouts(requirements, limits, &FeatureSet::default(), 1).expect_err(
                "aggregate fragment storage texture over-limit should reject auto layout"
            ),
            "pipeline auto layout uses too many storage textures in the fragment stage"
        );
    }

    #[test]
    fn derive_bind_group_layouts_accepts_aggregate_stage_counts_within_limits() {
        let requirements = [
            reflected_texture_binding(0, 0),
            reflected_storage_texture_binding(1, 0),
        ];
        let layouts = derive_bind_group_layouts(
            requirements,
            Limits {
                max_sampled_textures_per_shader_stage: 1,
                max_storage_textures_per_shader_stage: 1,
                max_storage_textures_in_fragment_stage: 1,
                ..Limits::DEFAULT
            },
            &FeatureSet::default(),
            1,
        )
        .expect("auto layout within aggregate stage limits should derive");

        assert_eq!(layouts.len(), 2);
        assert_eq!(layouts[0].entries().len(), 1);
        assert_eq!(layouts[1].entries().len(), 1);
    }

    #[cfg(feature = "gles")]
    #[test]
    fn select_compute_shader_source_generates_gles_glsl() {
        let device = noop_device();
        let wgsl = device.create_shader_module(ShaderModuleSource::Wgsl(
            "@compute @workgroup_size(2, 1, 1) fn cs() {}".to_owned(),
        ));

        let (source, entry, bindings) =
            select_compute_shader_source(HalBackend::Gles, &wgsl, "cs", &[], &[], false)
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

    #[test]
    fn vulkan_external_texture_rejection_detects_external_binding() {
        let plain = MetalBufferBinding {
            group: 0,
            binding: 0,
            metal_index: 0,
            ext_params_buffer_slot: None,
            ext_params_vertex_buffer_slot: None,
            ext_params_fragment_buffer_slot: None,
            vertex_metal_index: None,
            fragment_metal_index: None,
            kind: MetalBindingKind::Texture,
        };
        assert_eq!(vulkan_external_texture_rejection(&[plain]), None);

        let external = MetalBufferBinding {
            group: 0,
            binding: 0,
            metal_index: 0,
            ext_params_buffer_slot: Some(1),
            ext_params_vertex_buffer_slot: None,
            ext_params_fragment_buffer_slot: Some(1),
            vertex_metal_index: None,
            fragment_metal_index: Some(0),
            kind: MetalBindingKind::ExternalTexture,
        };
        assert_eq!(
            vulkan_external_texture_rejection(&[external]).as_deref(),
            Some("external textures are not supported on the Vulkan backend")
        );
    }

    #[test]
    fn select_compute_shader_source_rejects_external_texture_on_vulkan() {
        let device = noop_device();
        let wgsl = device.create_shader_module(ShaderModuleSource::Wgsl(
            "@compute @workgroup_size(1) fn cs() {}".to_owned(),
        ));
        let external = [MetalBufferBinding {
            group: 0,
            binding: 0,
            metal_index: 0,
            ext_params_buffer_slot: Some(1),
            ext_params_vertex_buffer_slot: None,
            ext_params_fragment_buffer_slot: Some(1),
            vertex_metal_index: None,
            fragment_metal_index: None,
            kind: MetalBindingKind::ExternalTexture,
        }];
        let err =
            select_compute_shader_source(HalBackend::Vulkan, &wgsl, "cs", &[], &external, false)
                .expect_err("Vulkan must reject external textures");
        assert_eq!(
            err,
            "external textures are not supported on the Vulkan backend"
        );
    }

    #[test]
    fn shader_layout_binding_kind_compatibility_follows_webgpu_rules() {
        // F-061 sample-type rule: a float layout (filterable/unfilterable) accepts
        // either float shader type; other types are exact.
        assert!(sample_types_compatible(
            TextureSampleType::Float,
            TextureSampleType::UnfilterableFloat
        ));
        assert!(sample_types_compatible(
            TextureSampleType::UnfilterableFloat,
            TextureSampleType::Float
        ));
        assert!(!sample_types_compatible(
            TextureSampleType::Float,
            TextureSampleType::Depth
        ));
        assert!(sample_types_compatible(
            TextureSampleType::Depth,
            TextureSampleType::Depth
        ));
        assert!(!sample_types_compatible(
            TextureSampleType::Uint,
            TextureSampleType::Sint
        ));

        // Access rule: a read-write layout accepts read-write or write-only.
        assert!(storage_accesses_compatible(
            StorageTextureAccess::ReadWrite,
            StorageTextureAccess::WriteOnly
        ));
        assert!(storage_accesses_compatible(
            StorageTextureAccess::ReadWrite,
            StorageTextureAccess::ReadWrite
        ));
        assert!(!storage_accesses_compatible(
            StorageTextureAccess::ReadOnly,
            StorageTextureAccess::WriteOnly
        ));
        assert!(!storage_accesses_compatible(
            StorageTextureAccess::WriteOnly,
            StorageTextureAccess::ReadWrite
        ));

        // Sampler rule: same type, or neither is comparison.
        assert!(sampler_types_compatible(
            SamplerBindingType::Filtering,
            SamplerBindingType::NonFiltering
        ));
        assert!(sampler_types_compatible(
            SamplerBindingType::Comparison,
            SamplerBindingType::Comparison
        ));
        assert!(!sampler_types_compatible(
            SamplerBindingType::Comparison,
            SamplerBindingType::Filtering
        ));
        assert!(!sampler_types_compatible(
            SamplerBindingType::Filtering,
            SamplerBindingType::Comparison
        ));
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn input_attachment_compat_ignores_multisampled_and_checks_sample_type() {
        // The shader-reflected `multisampled` for an input attachment is always
        // `false` under Tint, so the compat check must ignore `multisampled` and
        // compare only `sample_type`. C-1 owns the sample-count consistency.
        let ia = |sample_type, multisampled| BindingLayoutKind::InputAttachment {
            sample_type,
            multisampled,
        };

        // Reflection (Float, false) vs explicit layout (Float, true) — compatible.
        assert!(shader_binding_layout_kinds_compatible(
            ia(TextureSampleType::Float, false),
            ia(TextureSampleType::Float, true),
        ));
        // And the reverse.
        assert!(shader_binding_layout_kinds_compatible(
            ia(TextureSampleType::Float, true),
            ia(TextureSampleType::Float, false),
        ));
        // Matching sample types are compatible regardless of the two flags.
        for (a, b) in [(false, false), (false, true), (true, false), (true, true)] {
            assert!(shader_binding_layout_kinds_compatible(
                ia(TextureSampleType::Uint, a),
                ia(TextureSampleType::Uint, b),
            ));
        }
        // Incompatible only on sample type (shader Float vs layout Sint).
        assert!(!shader_binding_layout_kinds_compatible(
            ia(TextureSampleType::Float, false),
            ia(TextureSampleType::Sint, true),
        ));
    }

    /// Build a minimal single-group layout with one entry per kind descriptor.
    fn make_layout(
        _device: &crate::device::Device,
        entries: Vec<BindGroupLayoutEntry>,
    ) -> Arc<BindGroupLayout> {
        Arc::new(BindGroupLayout::new(entries, false, true))
    }

    fn buf_entry(binding: u32, vis: u64) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: vis,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: 0,
            }),
        }
    }

    fn tex_entry(binding: u32, vis: u64) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: vis,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::Texture {
                sample_type: TextureSampleType::Float,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            }),
        }
    }

    fn smp_entry(binding: u32, vis: u64) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: vis,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::Sampler {
                ty: SamplerBindingType::Filtering,
            }),
        }
    }

    fn ext_entry(binding: u32, vis: u64) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: vis,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::ExternalTexture),
        }
    }

    #[test]
    fn metal_binding_map_per_kind_counters_are_independent() {
        // Layout: buffer@0, texture@1, sampler@2, texture@3, sampler@4
        // Expected: buffer-space: buf@0→0; texture-space: tex@1→0, tex@3→1;
        //           sampler-space: smp@2→0, smp@4→1.
        let device = noop_device();
        let layout = make_layout(
            &device,
            vec![
                buf_entry(0, SHADER_STAGE_COMPUTE),
                tex_entry(1, SHADER_STAGE_COMPUTE),
                smp_entry(2, SHADER_STAGE_COMPUTE),
                tex_entry(3, SHADER_STAGE_COMPUTE),
                smp_entry(4, SHADER_STAGE_COMPUTE),
            ],
        );
        let bindings = metal_buffer_binding_map(&[layout]);

        // Find each binding by kind and slot.
        let buf = bindings.iter().find(|b| b.binding == 0).unwrap();
        let tex0 = bindings.iter().find(|b| b.binding == 1).unwrap();
        let smp0 = bindings.iter().find(|b| b.binding == 2).unwrap();
        let tex1 = bindings.iter().find(|b| b.binding == 3).unwrap();
        let smp1 = bindings.iter().find(|b| b.binding == 4).unwrap();

        assert_eq!(buf.metal_index, 0, "buffer slot");
        assert_eq!(tex0.metal_index, 0, "first texture slot");
        assert_eq!(smp0.metal_index, 0, "first sampler slot");
        assert_eq!(tex1.metal_index, 1, "second texture slot");
        assert_eq!(smp1.metal_index, 1, "second sampler slot");
    }

    #[test]
    fn metal_binding_map_external_texture_consumes_two_texture_and_one_buffer_slot() {
        // ExternalTexture occupies 2 consecutive texture slots + 1 buffer slot.
        // A plain buffer after it should start at buffer slot 1 (not 0).
        let device = noop_device();
        let layout = make_layout(
            &device,
            vec![
                ext_entry(0, SHADER_STAGE_COMPUTE),
                buf_entry(1, SHADER_STAGE_COMPUTE),
                tex_entry(2, SHADER_STAGE_COMPUTE),
            ],
        );
        let bindings = metal_buffer_binding_map(&[layout]);

        let ext = bindings.iter().find(|b| b.binding == 0).unwrap();
        let buf = bindings.iter().find(|b| b.binding == 1).unwrap();
        let tex = bindings.iter().find(|b| b.binding == 2).unwrap();

        // ExternalTexture: texture-space base slot 0, params buffer slot 0.
        assert_eq!(ext.metal_index, 0, "ext texture base slot");
        assert_eq!(
            ext.ext_params_buffer_slot,
            Some(0),
            "ext params buffer slot"
        );
        // The plain buffer follows in buffer-space: slot 1 (after the ext params slot).
        assert_eq!(buf.metal_index, 1, "buffer after external texture");
        // The plain texture follows in texture-space: slot 2 (after plane0/plane1).
        assert_eq!(tex.metal_index, 2, "texture after external texture");
    }

    #[test]
    fn metal_binding_map_external_texture_keeps_render_stage_slots_distinct() {
        let device = noop_device();
        let layout = make_layout(
            &device,
            vec![
                tex_entry(0, SHADER_STAGE_FRAGMENT),
                buf_entry(1, SHADER_STAGE_VERTEX),
                ext_entry(2, SHADER_STAGE_VERTEX | SHADER_STAGE_FRAGMENT),
            ],
        );
        let bindings = metal_buffer_binding_map(&[layout]);

        let ext = bindings.iter().find(|b| b.binding == 2).unwrap();

        assert_eq!(ext.vertex_metal_index, Some(0), "vertex plane0 slot");
        assert_eq!(ext.fragment_metal_index, Some(1), "fragment plane0 slot");
        assert_eq!(
            ext.ext_params_vertex_buffer_slot,
            Some(1),
            "vertex params slot"
        );
        assert_eq!(
            ext.ext_params_fragment_buffer_slot,
            Some(0),
            "fragment params slot"
        );
        assert_eq!(
            ext.ext_params_buffer_slot,
            Some(1),
            "flat fallback slot prefers vertex"
        );
    }

    #[test]
    fn metal_binding_map_per_stage_indices_are_independent_for_render_pipelines() {
        // Render layout:
        //   binding 0 — VERTEX only (buffer)
        //   binding 1 — FRAGMENT only (texture)
        //   binding 2 — BOTH (sampler)
        // Expected:
        //   binding 0: vertex_metal_index=Some(0), fragment_metal_index=None
        //   binding 1: vertex_metal_index=None, fragment_metal_index=Some(0)
        //   binding 2 (sampler): vertex_metal_index=Some(0), fragment_metal_index=Some(0)
        let device = noop_device();
        let layout = make_layout(
            &device,
            vec![
                buf_entry(0, SHADER_STAGE_VERTEX),
                tex_entry(1, SHADER_STAGE_FRAGMENT),
                smp_entry(2, SHADER_STAGE_VERTEX | SHADER_STAGE_FRAGMENT),
            ],
        );
        let bindings = metal_buffer_binding_map(&[layout]);

        let buf = bindings.iter().find(|b| b.binding == 0).unwrap();
        let tex = bindings.iter().find(|b| b.binding == 1).unwrap();
        let smp = bindings.iter().find(|b| b.binding == 2).unwrap();

        // VERTEX-only buffer.
        assert_eq!(buf.vertex_metal_index, Some(0));
        assert_eq!(buf.fragment_metal_index, None);

        // FRAGMENT-only texture.
        assert_eq!(tex.vertex_metal_index, None);
        assert_eq!(tex.fragment_metal_index, Some(0));

        // BOTH sampler — each stage has its own independent counter starting at 0.
        assert_eq!(smp.vertex_metal_index, Some(0));
        assert_eq!(smp.fragment_metal_index, Some(0));
    }

    #[test]
    fn metal_vertex_buffer_start_slot_equals_vertex_stage_buffer_count() {
        use crate::render_pipeline::metal_vertex_buffer_binding_map;
        // Two VERTEX-visible buffers → vertex buffer start slot must be 2.
        let device = noop_device();
        let layout = make_layout(
            &device,
            vec![
                buf_entry(0, SHADER_STAGE_VERTEX),
                buf_entry(1, SHADER_STAGE_VERTEX),
            ],
        );
        let bindings = metal_buffer_binding_map(&[layout]);
        let vb_bindings = metal_vertex_buffer_binding_map(
            &[crate::VertexBufferLayout {
                used: true,
                array_stride: 4,
                step_mode: crate::VertexStepMode::Vertex,
                attributes: Vec::new(),
            }],
            &bindings,
        );
        // The one vertex buffer should start at Metal buffer slot 2.
        assert_eq!(vb_bindings[0].metal_index, 2);
    }

    #[test]
    fn validate_metal_slot_ranges_rejects_sampler_past_limit() {
        // Build 16 sampler bindings (slots 0-15) — last one is slot 15 which is
        // still in range.  The 17th would be slot 16 which must be rejected.
        let device = noop_device();
        let entries: Vec<_> = (0..17)
            .map(|i| smp_entry(i, SHADER_STAGE_COMPUTE))
            .collect();
        let layout = make_layout(&device, entries);
        let bindings = metal_buffer_binding_map(&[layout]);

        let result = validate_metal_slot_ranges(&bindings);
        assert!(
            result.is_err(),
            "slot 16 must exceed the sampler limit of 15"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("sampler slot"),
            "error mentions sampler: {msg}"
        );
    }

    #[test]
    fn validate_metal_slot_ranges_accepts_valid_layout() {
        let device = noop_device();
        let layout = make_layout(
            &device,
            vec![
                buf_entry(0, SHADER_STAGE_COMPUTE),
                tex_entry(1, SHADER_STAGE_COMPUTE),
                smp_entry(2, SHADER_STAGE_COMPUTE),
            ],
        );
        let bindings = metal_buffer_binding_map(&[layout]);
        assert_eq!(validate_metal_slot_ranges(&bindings), Ok(()));
    }

    #[cfg(feature = "shader-passthrough")]
    fn valid_spirv_words() -> Vec<u32> {
        vec![0x0723_0203, 0, 0, 0, 0]
    }

    #[cfg(feature = "shader-passthrough")]
    fn spirv_passthrough_module(device: &Device) -> Arc<ShaderModule> {
        Arc::new(
            device.create_shader_module(ShaderModuleSource::SpirvPassthrough(valid_spirv_words())),
        )
    }

    #[cfg(feature = "shader-passthrough")]
    fn msl_passthrough_module(device: &Device, entries: Vec<MslEntryPoint>) -> Arc<ShaderModule> {
        Arc::new(
            device.create_shader_module(ShaderModuleSource::MslPassthrough {
                source: "kernel void cs() {}".to_owned(),
                entries,
            }),
        )
    }

    #[cfg(feature = "shader-passthrough")]
    fn msl_compute_entry(name: &str, workgroup_size: [u32; 3]) -> MslEntryPoint {
        MslEntryPoint {
            name: name.to_owned(),
            stage: ShaderStage::Compute,
            workgroup_size,
        }
    }

    #[cfg(feature = "shader-passthrough")]
    fn empty_pipeline_layout(device: &Device) -> Arc<PipelineLayout> {
        Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: Vec::new(),
            immediate_size: 0,
            error: None,
        }))
    }

    #[cfg(feature = "shader-passthrough")]
    fn spirv_passthrough_descriptor(
        shader_module: Arc<ShaderModule>,
        layout: ComputePipelineLayout,
    ) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            layout,
            shader_module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        }
    }

    #[cfg(feature = "shader-passthrough")]
    fn msl_passthrough_descriptor(
        shader_module: Arc<ShaderModule>,
        layout: ComputePipelineLayout,
    ) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            layout,
            shader_module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        }
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn spirv_passthrough_compute_pipeline_builds_on_noop_with_explicit_layout() {
        let device = noop_device();
        let module = spirv_passthrough_module(&device);
        let layout = empty_pipeline_layout(&device);

        let pipeline = device.create_compute_pipeline(spirv_passthrough_descriptor(
            module,
            ComputePipelineLayout::Explicit(layout),
        ));

        assert!(!pipeline.is_error());
        assert_eq!(pipeline.entry_name(), "cs");
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn resolve_spirv_passthrough_compute_pipeline_uses_explicit_layout_and_placeholder_workgroup() {
        let device = noop_device();
        let module = spirv_passthrough_module(&device);
        let layout = empty_pipeline_layout(&device);
        let descriptor = spirv_passthrough_descriptor(
            module,
            ComputePipelineLayout::Explicit(Arc::clone(&layout)),
        );

        let (entry, bindings, workgroup, layouts) =
            resolve_spirv_passthrough_compute_pipeline_descriptor(&descriptor)
                .expect("SPIR-V passthrough resolve");

        assert_eq!(entry, "cs");
        assert!(bindings.is_empty());
        assert_eq!(
            workgroup,
            Some(ResolvedComputeWorkgroup {
                size: [1, 1, 1],
                storage_size: 0,
            })
        );
        assert_eq!(layouts.len(), layout.bind_group_layouts().len());
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn select_compute_shader_source_rejects_spirv_passthrough_on_metal() {
        let device = noop_device();
        let module = spirv_passthrough_module(&device);

        let err = select_compute_shader_source(HalBackend::Metal, &module, "cs", &[], &[], false)
            .expect_err("SPIR-V passthrough must reject Metal");

        assert_eq!(err, "SPIR-V passthrough shader requires the Vulkan backend");
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn select_compute_shader_source_uses_spirv_passthrough_on_vulkan() {
        let device = noop_device();
        let module = spirv_passthrough_module(&device);
        let layout = make_layout(&device, vec![buf_entry(7, SHADER_STAGE_COMPUTE)]);
        let metal_bindings = metal_buffer_binding_map(&[layout]);

        let (source, entry, descriptor_bindings) = select_compute_shader_source(
            HalBackend::Vulkan,
            &module,
            "cs",
            &[],
            &metal_bindings,
            false,
        )
        .expect("SPIR-V passthrough should select Vulkan source");

        let HalShaderSource::SpirV(words) = source else {
            panic!("Vulkan should select SPIR-V passthrough words");
        };
        assert_eq!(words, valid_spirv_words());
        assert_eq!(entry, "cs");
        assert_eq!(descriptor_bindings.len(), 1);
        assert_eq!(descriptor_bindings[0].group, 0);
        assert_eq!(descriptor_bindings[0].binding, 7);
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn spirv_passthrough_compute_pipeline_rejects_auto_layout() {
        let device = noop_device();
        let module = spirv_passthrough_module(&device);

        let err = validate_compute_pipeline_descriptor(
            &spirv_passthrough_descriptor(module, ComputePipelineLayout::Auto),
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
    fn spirv_passthrough_compute_pipeline_rejects_pipeline_constants() {
        let device = noop_device();
        let module = spirv_passthrough_module(&device);
        let layout = ComputePipelineLayout::Explicit(empty_pipeline_layout(&device));
        let mut descriptor = spirv_passthrough_descriptor(module, layout);
        descriptor.constants.push(PipelineConstant {
            key: "x".to_owned(),
            value: 1.0,
        });

        let err = validate_compute_pipeline_descriptor(
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
    fn msl_passthrough_compute_pipeline_builds_on_noop_with_explicit_layout() {
        let device = noop_device();
        let module = msl_passthrough_module(&device, vec![msl_compute_entry("cs", [4, 1, 1])]);
        let layout = empty_pipeline_layout(&device);

        let pipeline = device.create_compute_pipeline(msl_passthrough_descriptor(
            module,
            ComputePipelineLayout::Explicit(layout),
        ));

        assert!(!pipeline.is_error());
        assert_eq!(pipeline.entry_name(), "cs");
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn select_compute_shader_source_rejects_msl_passthrough_on_vulkan() {
        let device = noop_device();
        let module = msl_passthrough_module(&device, vec![msl_compute_entry("cs", [1, 1, 1])]);

        let err = select_compute_shader_source(HalBackend::Vulkan, &module, "cs", &[], &[], false)
            .expect_err("MSL passthrough must reject Vulkan");

        assert_eq!(err, "MSL passthrough shader requires the Metal backend");
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_passthrough_compute_pipeline_rejects_auto_layout() {
        let device = noop_device();
        let module = msl_passthrough_module(&device, vec![msl_compute_entry("cs", [1, 1, 1])]);

        let err = validate_compute_pipeline_descriptor(
            &msl_passthrough_descriptor(module, ComputePipelineLayout::Auto),
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
    fn msl_passthrough_compute_pipeline_requires_matching_compute_workgroup() {
        let device = noop_device();
        let layout = ComputePipelineLayout::Explicit(empty_pipeline_layout(&device));
        let missing = msl_passthrough_module(
            &device,
            vec![MslEntryPoint {
                name: "vs".to_owned(),
                stage: ShaderStage::Vertex,
                workgroup_size: [0, 0, 0],
            }],
        );

        let err = validate_compute_pipeline_descriptor(
            &msl_passthrough_descriptor(missing, layout.clone()),
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("missing compute metadata should fail");
        assert_eq!(
            err,
            "MSL passthrough compute entry point not found or missing workgroup size"
        );

        let zero = Arc::new(ShaderModule::new(
            ShaderModuleSourceKind::MslPassthrough {
                source: "kernel void cs() {}".to_owned(),
                entries: vec![msl_compute_entry("cs", [1, 0, 1])],
            },
            None,
        ));
        let err = validate_compute_pipeline_descriptor(
            &msl_passthrough_descriptor(zero, layout),
            device.limits(),
            &FeatureSet::default(),
        )
        .expect("zero workgroup metadata should fail");
        assert_eq!(
            err,
            "MSL passthrough compute entry point not found or missing workgroup size"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_passthrough_compute_pipeline_rejects_pipeline_constants() {
        let device = noop_device();
        let module = msl_passthrough_module(&device, vec![msl_compute_entry("cs", [1, 1, 1])]);
        let layout = ComputePipelineLayout::Explicit(empty_pipeline_layout(&device));
        let mut descriptor = msl_passthrough_descriptor(module, layout);
        descriptor.constants.push(PipelineConstant {
            key: "x".to_owned(),
            value: 1.0,
        });

        let err = validate_compute_pipeline_descriptor(
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

    // ---- Regression A (F-078): explicit two-group compute pipeline must not error ----

    fn storage_bgl_entry(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: SHADER_STAGE_COMPUTE,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::Buffer {
                ty: BufferBindingType::Storage,
                has_dynamic_offset: false,
                min_binding_size: 0,
            }),
        }
    }

    fn uniform_bgl_entry(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: SHADER_STAGE_COMPUTE,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: 0,
            }),
        }
    }

    /// Regression A: explicit two-group compute pipeline (group 0 = storage buffer;
    /// group 1 = uniform + storage) must create without error on every backend
    /// including Noop.  This was broken by the d376a1b per-kind/per-stage binding-map
    /// rework when it misclassified a compute layout as a render layout (or vice
    /// versa), producing Metal slot validation failures even on Noop.
    #[test]
    fn explicit_two_group_compute_pipeline_creates_without_error() {
        // Shader matching the CTS `robust_access,linear_memory` shape exactly:
        // group(0) binding(0): storage read_write buffer
        // group(1) binding(0): uniform buffer (constants)
        // group(1) binding(1): storage read_write buffer (result)
        const WGSL: &str = "\
struct Constants { zero: u32 }\n\
struct Result { value: u32 }\n\
@group(0) @binding(0) var<storage, read_write> src: array<u32>;\n\
@group(1) @binding(0) var<uniform> constants: Constants;\n\
@group(1) @binding(1) var<storage, read_write> result: Result;\n\
@compute @workgroup_size(1)\n\
fn main() {\n\
  _ = constants.zero;\n\
  result.value = select(src[0], 0u, constants.zero == 0u);\n\
}\n";

        let device = noop_device();

        // BGL 0: one storage buffer binding with COMPUTE visibility.
        let bgl0 = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![storage_bgl_entry(0)],
            error: None,
        }));

        // BGL 1: uniform (binding 0) + storage read_write (binding 1), COMPUTE visibility.
        let bgl1 = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![uniform_bgl_entry(0), storage_bgl_entry(1)],
            error: None,
        }));

        let layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::clone(&bgl0), Arc::clone(&bgl1)],
            immediate_size: 0,
            error: None,
        }));

        let module =
            Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(WGSL.to_owned())));
        assert!(!module.is_error(), "shader module must compile");

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(Arc::clone(&layout)),
            shader_module: Arc::clone(&module),
            entry_point: Some("main".to_owned()),
            constants: Vec::new(),
            error: None,
        });
        let scoped = device.pop_error_scope().expect("scope should exist");
        assert!(!pipeline.is_error(), "pipeline must not be an error");
        assert_eq!(
            scoped, None,
            "no validation error expected for explicit two-group compute pipeline"
        );

        // Also verify the metal binding map assigns per-kind slots correctly:
        // Buffer indices must be sequential across groups (0, 1, 2).
        let bindings = pipeline.metal_bindings();
        let b00 = bindings
            .iter()
            .find(|b| b.group == 0 && b.binding == 0)
            .expect("group 0 binding 0 must be present");
        let b10 = bindings
            .iter()
            .find(|b| b.group == 1 && b.binding == 0)
            .expect("group 1 binding 0 must be present");
        let b11 = bindings
            .iter()
            .find(|b| b.group == 1 && b.binding == 1)
            .expect("group 1 binding 1 must be present");

        assert_eq!(b00.metal_index, 0, "group0/binding0 must be buffer slot 0");
        assert_eq!(b10.metal_index, 1, "group1/binding0 must be buffer slot 1");
        assert_eq!(b11.metal_index, 2, "group1/binding1 must be buffer slot 2");

        // All are compute-layout entries: per-stage indices must be None.
        assert_eq!(b00.vertex_metal_index, None);
        assert_eq!(b00.fragment_metal_index, None);
        assert_eq!(b10.vertex_metal_index, None);
        assert_eq!(b10.fragment_metal_index, None);
        assert_eq!(b11.vertex_metal_index, None);
        assert_eq!(b11.fragment_metal_index, None);
    }

    /// Regression A (empty-group variant): group 0 is EMPTY (no entries), group 1
    /// has uniform + storage.  The empty BGL is valid and must not cause an error.
    #[test]
    fn explicit_two_group_compute_pipeline_with_empty_group0_creates_without_error() {
        const WGSL: &str = "\
struct Constants { zero: u32 }\n\
struct Result { value: u32 }\n\
@group(1) @binding(0) var<uniform> constants: Constants;\n\
@group(1) @binding(1) var<storage, read_write> result: Result;\n\
@compute @workgroup_size(1)\n\
fn main() {\n\
  _ = constants.zero;\n\
  result.value = 0u;\n\
}\n";

        let device = noop_device();

        // BGL 0: intentionally empty.
        let bgl0 = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![],
            error: None,
        }));

        // BGL 1: uniform + storage.
        let bgl1 = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![uniform_bgl_entry(0), storage_bgl_entry(1)],
            error: None,
        }));

        let layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::clone(&bgl0), Arc::clone(&bgl1)],
            immediate_size: 0,
            error: None,
        }));

        let module =
            Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(WGSL.to_owned())));
        assert!(!module.is_error(), "shader module must compile");

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(Arc::clone(&layout)),
            shader_module: Arc::clone(&module),
            entry_point: Some("main".to_owned()),
            constants: Vec::new(),
            error: None,
        });
        let scoped = device.pop_error_scope().expect("scope should exist");
        assert!(
            !pipeline.is_error(),
            "pipeline with empty group 0 must not be an error"
        );
        assert_eq!(scoped, None, "no validation error expected");
    }

    // ---- F-080: unfilterable-float texture + filtering sampler must error ----
    //
    // CTS: api,validation,non_filterable_texture:non_filterable_texture_with_filtering_sampler
    //
    // The WGSL shader uses texture_2d<f32> (shader-reflected as Float) with
    // textureGather.  The explicit BGL declares the texture slot as
    // UnfilterableFloat.  The F-061 compat rule allows an UnfilterableFloat
    // layout to accept a Float shader binding — but the explicit
    // UnfilterableFloat combined with a Filtering sampler must still produce a
    // validation error.  Before the fix, the shader-Float early-exit in
    // validate_non_filterable_gather_bindings skipped the check, accepting the
    // invalid combination silently.

    /// Makes a BGL entry for a texture with the given sample type and visibility.
    fn tex_entry_with_sample_type(
        binding: u32,
        vis: u64,
        sample_type: TextureSampleType,
    ) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: vis,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::Texture {
                sample_type,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            }),
        }
    }

    /// Makes a BGL entry for a sampler with the given type and visibility.
    fn smp_entry_with_type(binding: u32, vis: u64, ty: SamplerBindingType) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: vis,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::Sampler { ty }),
        }
    }

    /// WGSL for the CTS non_filterable_texture test: texture_2d<f32> at
    /// @group(0) @binding(0), sampler at @group(group_ndx) @binding(1),
    /// used together in textureGather — all from a compute entry point.
    fn non_filterable_wgsl(group_ndx: u32) -> String {
        format!(
            r"
@group(0) @binding(0) var t: texture_2d<f32>;
@group({group_ndx}) @binding(1) var s: sampler;

fn test() {{
  _ = textureGather(0, t, s, vec2f(0.0));
}}

@compute @workgroup_size(1) fn cs() {{ test(); }}
",
        )
    }

    /// Regression test (F-080): explicit BGL `UnfilterableFloat` texture + `Filtering`
    /// sampler in the same group must produce a validation error on compute pipeline
    /// creation.
    #[test]
    fn unfilterable_float_texture_with_filtering_sampler_rejects_compute_pipeline() {
        let device = noop_device();

        let vis = SHADER_STAGE_COMPUTE;
        let bgl = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![
                tex_entry_with_sample_type(0, vis, TextureSampleType::UnfilterableFloat),
                smp_entry_with_type(1, vis, SamplerBindingType::Filtering),
            ],
            error: None,
        }));
        let layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::clone(&bgl)],
            immediate_size: 0,
            error: None,
        }));
        let module =
            Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(non_filterable_wgsl(0))));
        assert!(!module.is_error(), "shader module must compile");

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        });
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("unfilterable_float+filtering_sampler must produce a validation error (F-080)");
        assert!(pipeline.is_error());
        assert_eq!(
            scoped.message,
            "textureGather with a filtering sampler requires a filterable texture binding"
        );
    }

    /// Positive case (F-080): explicit BGL `Float` texture + `Filtering` sampler
    /// must succeed.
    #[test]
    fn filterable_float_texture_with_filtering_sampler_accepts_compute_pipeline() {
        let device = noop_device();

        let vis = SHADER_STAGE_COMPUTE;
        let bgl = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![
                tex_entry_with_sample_type(0, vis, TextureSampleType::Float),
                smp_entry_with_type(1, vis, SamplerBindingType::Filtering),
            ],
            error: None,
        }));
        let layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::clone(&bgl)],
            immediate_size: 0,
            error: None,
        }));
        let module =
            Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(non_filterable_wgsl(0))));
        assert!(!module.is_error(), "shader module must compile");

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        });
        let scoped = device.pop_error_scope().expect("scope should exist");
        assert!(!pipeline.is_error(), "Float+Filtering must succeed");
        assert_eq!(
            scoped, None,
            "no validation error expected for filterable Float"
        );
    }

    /// Cross-group variant (F-080, sameGroup=false): texture in group 0, sampler
    /// in group 1 — the rejection must still fire.
    #[test]
    fn unfilterable_float_texture_cross_group_filtering_sampler_rejects_compute_pipeline() {
        let device = noop_device();

        let vis = SHADER_STAGE_COMPUTE;
        let bgl0 = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![tex_entry_with_sample_type(
                0,
                vis,
                TextureSampleType::UnfilterableFloat,
            )],
            error: None,
        }));
        let bgl1 = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![smp_entry_with_type(1, vis, SamplerBindingType::Filtering)],
            error: None,
        }));
        let layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::clone(&bgl0), Arc::clone(&bgl1)],
            immediate_size: 0,
            error: None,
        }));
        // Sampler is at group 1, binding 1.
        let module =
            Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(non_filterable_wgsl(1))));
        assert!(!module.is_error(), "shader module must compile");

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        });
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("cross-group unfilterable+filtering must produce a validation error (F-080)");
        assert!(pipeline.is_error());
        assert_eq!(
            scoped.message,
            "textureGather with a filtering sampler requires a filterable texture binding"
        );
    }
}
