use std::collections::BTreeSet;
use std::sync::Arc;

use crate::bind_group_layout::*;
use crate::shader_naga;
#[cfg(feature = "shader-passthrough")]
use crate::ReflectedModule;

pub(crate) const SHADER_STAGE_VERTEX: u64 = 1;
pub(crate) const SHADER_STAGE_FRAGMENT: u64 = 2;
pub(crate) const SHADER_STAGE_COMPUTE: u64 = 4;

/// Enumerates shader module source values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ShaderModuleSource {
    /// Wgsl variant.
    Wgsl(String),
    /// Spirv variant.
    Spirv(Vec<u32>),
    /// Msl variant.
    #[cfg(feature = "shader-passthrough")]
    Msl {
        /// Source.
        source: String,
        /// Reflection.
        reflection: MslReflection,
    },
    /// Invalid variant.
    Invalid(String),
}

/// Stores caller-supplied MSL reflection metadata.
#[cfg(feature = "shader-passthrough")]
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MslReflection {
    /// Entry points available in the MSL source.
    pub entry_points: Vec<MslEntryPoint>,
}

#[cfg(feature = "shader-passthrough")]
impl MslReflection {
    /// Creates caller-supplied MSL reflection metadata.
    #[must_use]
    pub fn new(entry_points: Vec<MslEntryPoint>) -> Self {
        Self { entry_points }
    }
}

/// Stores caller-supplied MSL entry point metadata.
#[cfg(feature = "shader-passthrough")]
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct MslEntryPoint {
    /// Entry point name.
    pub name: String,
    /// Shader stage bit. Exactly one of vertex (1), fragment (2), or compute (4).
    pub stage: u64,
    /// Compute workgroup size. Ignored for non-compute stages.
    pub workgroup_size: [u32; 3],
}

#[cfg(feature = "shader-passthrough")]
impl MslEntryPoint {
    /// Creates caller-supplied MSL entry-point metadata.
    #[must_use]
    pub fn new(name: String, stage: u64, workgroup_size: [u32; 3]) -> Self {
        Self {
            name,
            stage,
            workgroup_size,
        }
    }
}

/// Stores shader module data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct ShaderModule {
    pub(crate) inner: Arc<ShaderModuleInner>,
}

/// Holds shared state for the shader module handle.
#[derive(Debug)]
pub(crate) struct ShaderModuleInner {
    pub(crate) _source: ShaderModuleSourceKind,
    pub(crate) diagnostic: Option<String>,
    pub(crate) is_error: bool,
}

/// Enumerates shader module source kind values.
#[derive(Debug)]
pub(crate) enum ShaderModuleSourceKind {
    /// Wgsl variant.
    Wgsl {
        /// Source variant.
        _source: String,
        /// Reflected variant.
        reflected: Box<shader_naga::ReflectedModule>,
    },
    /// Spirv variant.
    #[cfg(feature = "shader-passthrough")]
    Spirv {
        /// Words variant.
        words: Vec<u32>,
        /// Reflected variant.
        reflected: Box<shader_naga::ReflectedModule>,
    },
    /// Msl variant.
    #[cfg(feature = "shader-passthrough")]
    Msl {
        /// Source variant.
        source: String,
        /// Reflection variant.
        reflection: MslReflection,
    },
    /// Invalid variant.
    Invalid,
}

impl ShaderModule {
    /// Creates a new instance.
    pub(crate) fn new(source: ShaderModuleSourceKind, diagnostic: Option<String>) -> Self {
        Self {
            inner: Arc::new(ShaderModuleInner {
                is_error: diagnostic.is_some(),
                _source: source,
                diagnostic,
            }),
        }
    }

    /// Constructs this object from wgsl.
    pub(crate) fn from_wgsl(source: String) -> Result<ShaderModuleSourceKind, String> {
        let reflected = shader_naga::parse_and_validate_wgsl(&source)?;
        validate_module_limits(&reflected.module)?;
        Ok(ShaderModuleSourceKind::Wgsl {
            _source: source,
            reflected: Box::new(reflected),
        })
    }

    /// Constructs this object from SPIR-V.
    #[cfg(feature = "shader-passthrough")]
    pub(crate) fn from_spirv(words: Vec<u32>) -> Result<ShaderModuleSourceKind, String> {
        if words.first().copied() != Some(0x0723_0203) {
            return Err("SPIR-V shader module must start with the SPIR-V magic number".to_owned());
        }
        let reflected = shader_naga::reflect_spirv(&words)?;
        validate_module_limits(&reflected.module)?;
        Ok(ShaderModuleSourceKind::Spirv {
            words,
            reflected: Box::new(reflected),
        })
    }

    /// Constructs this object from MSL and caller-supplied reflection metadata.
    #[cfg(feature = "shader-passthrough")]
    pub(crate) fn from_msl(
        source: String,
        reflection: MslReflection,
    ) -> Result<ShaderModuleSourceKind, String> {
        if source.is_empty() {
            return Err("MSL shader module source must not be empty".to_owned());
        }
        for entry_point in &reflection.entry_points {
            if !matches!(entry_point.stage, 1 | 2 | 4) {
                return Err("MSL entry point stage must set exactly one shader stage".to_owned());
            }
        }
        Ok(ShaderModuleSourceKind::Msl { source, reflection })
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Creates a validation diagnostic from the supplied message.
    #[must_use]
    pub fn diagnostic(&self) -> Option<&str> {
        self.inner.diagnostic.as_deref()
    }

    /// Returns shader reflection data when the module source provides it.
    #[must_use]
    pub(crate) fn reflected(&self) -> Option<&shader_naga::ReflectedModule> {
        match &self.inner._source {
            ShaderModuleSourceKind::Wgsl { reflected, .. } => Some(reflected),
            #[cfg(feature = "shader-passthrough")]
            ShaderModuleSourceKind::Spirv { reflected, .. } => Some(reflected),
            _ => None,
        }
    }

    /// Returns the SPIR-V passthrough words and reflection data.
    #[cfg(feature = "shader-passthrough")]
    #[must_use]
    pub fn spirv_passthrough(&self) -> Option<(&[u32], &ReflectedModule)> {
        match &self.inner._source {
            ShaderModuleSourceKind::Spirv { words, reflected } => Some((words, reflected)),
            _ => None,
        }
    }

    /// Returns the MSL passthrough source and reflection metadata.
    #[cfg(feature = "shader-passthrough")]
    #[must_use]
    pub fn msl_passthrough(&self) -> Option<(&str, &MslReflection)> {
        match &self.inner._source {
            ShaderModuleSourceKind::Msl { source, reflection } => Some((source, reflection)),
            _ => None,
        }
    }
}

/// Validates module limits and returns a descriptive error on failure.
pub(crate) fn validate_module_limits(module: &naga::Module) -> Result<(), String> {
    let mut ids = BTreeSet::new();
    for (_, override_) in module.overrides.iter() {
        if let Some(id) = override_.id {
            if !ids.insert(id) {
                return Err(format!("duplicate shader override id {id}"));
            }
        }
    }

    for (_, global) in module.global_variables.iter() {
        if let Some(binding) = global.binding {
            if binding.binding >= 1000 {
                return Err(format!(
                    "shader resource binding {} exceeds the maximum binding number",
                    binding.binding
                ));
            }
        }
    }

    Ok(())
}

/// Stores stage resource counts data used by validation and backend submission.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct StageResourceCounts {
    pub(crate) sampled_textures: u32,
    pub(crate) samplers: u32,
    pub(crate) storage_buffers: u32,
    pub(crate) storage_textures: u32,
    pub(crate) uniform_buffers: u32,
}

impl StageResourceCounts {
    /// Adds this value to the aggregate counts.
    pub(crate) fn add(&mut self, kind: BindingLayoutKind) {
        match kind {
            BindingLayoutKind::Buffer {
                ty: BufferBindingType::Uniform,
                ..
            } => self.uniform_buffers += 1,
            BindingLayoutKind::Buffer {
                ty: BufferBindingType::Storage | BufferBindingType::ReadOnlyStorage,
                ..
            } => self.storage_buffers += 1,
            BindingLayoutKind::Sampler { .. } => self.samplers += 1,
            BindingLayoutKind::Texture { .. } => self.sampled_textures += 1,
            #[cfg(feature = "tiled")]
            BindingLayoutKind::InputAttachment { .. } => self.sampled_textures += 1,
            BindingLayoutKind::StorageTexture { .. } => self.storage_textures += 1,
        }
    }
}

/// Returns visible stages.
pub(crate) fn visible_stages(visibility: u64) -> impl Iterator<Item = usize> {
    [
        SHADER_STAGE_VERTEX,
        SHADER_STAGE_FRAGMENT,
        SHADER_STAGE_COMPUTE,
    ]
    .into_iter()
    .enumerate()
    .filter_map(move |(index, bit)| (visibility & bit != 0).then_some(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;

    #[test]
    fn shader_module_accessors_pin_is_error_and_diagnostic() {
        let device = noop_device();

        let valid = device.create_shader_module(ShaderModuleSource::Wgsl(
            "@compute @workgroup_size(1) fn cs() {}".to_owned(),
        ));
        assert!(!valid.is_error());
        assert_eq!(valid.diagnostic(), None);

        device.push_error_scope(ErrorFilter::Validation);
        let invalid = device.create_shader_module(ShaderModuleSource::Wgsl("not wgsl".to_owned()));
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid shader should be scoped");

        assert!(invalid.is_error());
        assert!(invalid.diagnostic().is_some());
        assert!(scoped.message.contains("expected global item"));
    }

    #[cfg(feature = "shader-passthrough")]
    fn test_spirv_words() -> Vec<u32> {
        shader_naga::parse_and_validate_wgsl("@compute @workgroup_size(2, 3, 4) fn cs() {}")
            .expect("test WGSL should validate")
            .generate_spirv(
                "cs",
                naga::ShaderStage::Compute,
                &naga::back::PipelineConstants::default(),
            )
            .expect("test WGSL should generate SPIR-V")
    }

    #[cfg(feature = "shader-passthrough")]
    fn test_msl_reflection(stage: u64) -> MslReflection {
        MslReflection {
            entry_points: vec![MslEntryPoint {
                name: "cs".to_owned(),
                stage,
                workgroup_size: [2, 3, 4],
            }],
        }
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn create_shader_module_spirv_reflects_entry_point_and_reports_bad_input() {
        let device = noop_device();
        let words = test_spirv_words();

        let valid = device.create_shader_module_spirv(words.clone());
        assert!(!valid.is_error());
        let (stored_words, reflected) = valid
            .spirv_passthrough()
            .expect("SPIR-V module should retain passthrough data");
        assert_eq!(stored_words, words.as_slice());
        assert!(reflected.entry_points().iter().any(|entry| {
            entry.name == "cs" && entry.stage == shader_naga::ReflectedShaderStage::Compute
        }));
        assert_eq!(
            reflected
                .compute_workgroup_size("cs")
                .expect("workgroup reflection should succeed")
                .expect("compute workgroup should exist")
                .literal_size,
            [2, 3, 4]
        );

        for bad_words in [Vec::new(), vec![0, 1, 2, 3, 4]] {
            device.push_error_scope(ErrorFilter::Validation);
            let invalid = device.create_shader_module_spirv(bad_words);
            let scoped = device
                .pop_error_scope()
                .expect("scope should exist")
                .expect("invalid SPIR-V should be scoped");
            assert!(invalid.is_error());
            assert!(invalid.diagnostic().is_some());
            assert!(invalid.spirv_passthrough().is_none());
            assert!(scoped.message.contains("SPIR-V magic number"));
            drop(invalid);
        }
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn create_shader_module_msl_stores_source_and_reflection_on_noop() {
        let device = noop_device();
        let source = "kernel void cs() {}".to_owned();
        let reflection = test_msl_reflection(SHADER_STAGE_COMPUTE);

        let module = device.create_shader_module_msl(source.clone(), reflection.clone());
        assert!(!module.is_error());
        assert_eq!(module.diagnostic(), None);
        let (stored_source, stored_reflection) = module
            .msl_passthrough()
            .expect("MSL module should retain passthrough data");
        assert_eq!(stored_source, source);
        assert_eq!(stored_reflection, &reflection);
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn create_shader_module_msl_rejects_invalid_stage_bits_and_is_release_safe() {
        let device = noop_device();

        for stage in [0, SHADER_STAGE_VERTEX | SHADER_STAGE_FRAGMENT, 8] {
            device.push_error_scope(ErrorFilter::Validation);
            let invalid = device.create_shader_module_msl(
                "kernel void cs() {}".to_owned(),
                test_msl_reflection(stage),
            );
            let scoped = device
                .pop_error_scope()
                .expect("scope should exist")
                .expect("invalid MSL reflection should be scoped");
            assert!(invalid.is_error());
            assert!(invalid.diagnostic().is_some());
            assert!(invalid.msl_passthrough().is_none());
            assert!(scoped.message.contains("exactly one shader stage"));
            drop(invalid);
        }
    }
}
