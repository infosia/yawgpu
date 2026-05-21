use std::collections::BTreeSet;
use std::sync::Arc;

use crate::bind_group_layout::*;
use crate::shader_naga;

/// Enumerates shader module source values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ShaderModuleSource {
    /// Wgsl variant.
    Wgsl(String),
    /// Spirv variant.
    Spirv(Vec<u32>),
    /// Invalid variant.
    Invalid(String),
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
        /// Validated variant.
        validated: Box<shader_naga::ValidatedWgslModule>,
    },
    /// Spirv variant.
    Spirv {
        /// Words variant.
        _words: Vec<u32>,
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
        let validated = shader_naga::parse_and_validate_wgsl(&source)?;
        validate_wgsl_module_limits(&validated.module)?;
        Ok(ShaderModuleSourceKind::Wgsl {
            _source: source,
            validated: Box::new(validated),
        })
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

    /// Returns the validated WGSL module when validation succeeded.
    #[must_use]
    pub(crate) fn validated_wgsl(&self) -> Option<&shader_naga::ValidatedWgslModule> {
        match &self.inner._source {
            ShaderModuleSourceKind::Wgsl { validated, .. } => Some(validated),
            _ => None,
        }
    }
}

/// Validates wgsl module limits and returns a descriptive error on failure.
pub(crate) fn validate_wgsl_module_limits(module: &naga::Module) -> Result<(), String> {
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
            BindingLayoutKind::StorageTexture { .. } => self.storage_textures += 1,
        }
    }
}

/// Returns visible stages.
pub(crate) fn visible_stages(visibility: u64) -> impl Iterator<Item = usize> {
    /// Constant value for vertex.
    pub(crate) const VERTEX: u64 = 1;
    /// Constant value for fragment.
    pub(crate) const FRAGMENT: u64 = 2;
    /// Constant value for compute.
    pub(crate) const COMPUTE: u64 = 4;
    [VERTEX, FRAGMENT, COMPUTE]
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
}
