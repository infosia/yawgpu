//! Tint shader frontend skeleton for the feature-selected frontend facade.
#![allow(dead_code)]

use std::collections::HashSet;

use crate::shader::CompilationMessage;
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
    Ok(ReflectedModule {
        program: yawgpu_tint::Program::parse(src, shader_f16)?,
        warnings: Vec::new(),
    })
}

impl ReflectedModule {
    /// Generates spirv for the validated shader module.
    pub(crate) fn generate_spirv(
        &self,
        _entry_name: &str,
        _stage: ShaderStage,
        _pipeline_constants: &PipelineConstants,
        _unchecked_buffer_bounds: bool,
    ) -> Result<Vec<u32>, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Generates GLSL ES for the validated shader module.
    #[cfg(feature = "gles")]
    pub(crate) fn generate_glsl(
        &self,
        _entry_name: &str,
        _stage: ShaderStage,
        _pipeline_constants: &PipelineConstants,
    ) -> Result<GeneratedGlsl, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Generates msl for the validated shader module.
    pub(crate) fn generate_msl(
        &self,
        _entry_name: &str,
        _binding_map: &MslBindingMap,
        _pipeline_constants: &PipelineConstants,
    ) -> Result<GeneratedMsl, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Generates render vertex MSL for a validated shader module.
    pub(crate) fn generate_render_vertex_msl(
        &self,
        _entry_name: &str,
        _binding_map: &MslBindingMap,
        _vertex_buffers: &[MslVertexBufferBinding],
        _force_point_size: bool,
        _pipeline_constants: &PipelineConstants,
    ) -> Result<GeneratedMsl, String> {
        Err(NOT_IMPLEMENTED.to_owned())
    }

    /// Generates render fragment MSL for a validated shader module.
    pub(crate) fn generate_render_fragment_msl(
        &self,
        _entry_name: &str,
        _binding_map: &MslBindingMap,
        _pipeline_constants: &PipelineConstants,
        _sample_mask: u32,
    ) -> Result<GeneratedMsl, String> {
        Err(NOT_IMPLEMENTED.to_owned())
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
        Err(NOT_IMPLEMENTED.to_owned())
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
        // TODO(P2b.2): expose Tint entry point input/output variables in the shim.
        Vec::new()
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
        _entry_point: &str,
    ) -> Result<Vec<MslBufferSizeBinding>, String> {
        // TODO(P2c): derive this from Tint code generation metadata.
        Err(NOT_IMPLEMENTED.to_owned())
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
        sample_usage: ReflectedTextureSampleUsage::Sample,
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
        sample_usage: ReflectedTextureSampleUsage::Sample,
        view_dimension: texture_view_dimension(binding.dim),
        multisampled,
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
        id: Some(override_.id),
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
                        sample_usage: ReflectedTextureSampleUsage::Sample,
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
}
