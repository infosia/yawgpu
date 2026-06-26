use std::collections::BTreeSet;
use std::sync::Arc;

use crate::bind_group_layout::*;
use crate::shader_naga;

/// Stores one shader compilation message for compilation-info callbacks.
#[derive(Clone, Debug)]
pub struct CompilationMessage {
    /// Message severity.
    pub severity: CompilationSeverity,
    /// Human-readable message.
    pub message: String,
    /// 1-based line number.
    pub line_num: u64,
    /// 1-based line position.
    pub line_pos: u64,
    /// 0-based byte offset.
    pub offset: u64,
    /// Byte length.
    pub length: u64,
}

/// Enumerates shader compilation message severities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompilationSeverity {
    /// Error message.
    Error,
    /// Warning message.
    Warning,
    /// Info message.
    Info,
}

pub(crate) const SHADER_STAGE_VERTEX: u64 = 1;
pub(crate) const SHADER_STAGE_FRAGMENT: u64 = 2;
pub(crate) const SHADER_STAGE_COMPUTE: u64 = 4;

/// Enumerates shader module source values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ShaderModuleSource {
    /// Wgsl variant.
    Wgsl(String),
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
    pub(crate) messages: Vec<CompilationMessage>,
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
    /// Invalid variant.
    Invalid,
}

impl ShaderModule {
    /// Creates a new instance.
    pub(crate) fn new(source: ShaderModuleSourceKind, diagnostic: Option<String>) -> Self {
        let messages = match &source {
            ShaderModuleSourceKind::Wgsl { reflected, .. } => reflected.warnings.clone(),
            _ => Vec::new(),
        };
        Self {
            inner: Arc::new(ShaderModuleInner {
                is_error: diagnostic.is_some(),
                _source: source,
                diagnostic,
                messages,
            }),
        }
    }

    /// Constructs this object from wgsl.
    pub(crate) fn from_wgsl(
        source: String,
        shader_f16: bool,
    ) -> Result<ShaderModuleSourceKind, String> {
        let reflected = shader_naga::parse_and_validate_wgsl_gated(&source, shader_f16)?;
        validate_module_limits(&reflected.module)?;
        Ok(ShaderModuleSourceKind::Wgsl {
            _source: source,
            reflected: Box::new(reflected),
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

    /// Returns non-fatal compilation messages.
    #[must_use]
    pub fn compilation_messages(&self) -> &[CompilationMessage] {
        &self.inner.messages
    }

    /// Returns shader reflection data when the module source provides it.
    #[must_use]
    pub(crate) fn reflected(&self) -> Option<&shader_naga::ReflectedModule> {
        match &self.inner._source {
            ShaderModuleSourceKind::Wgsl { reflected, .. } => Some(reflected),
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
            BindingLayoutKind::ExternalTexture => {
                self.sampled_textures += 4;
                self.samplers += 1;
                self.uniform_buffers += 1;
            }
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
        assert!(valid.compilation_messages().is_empty());

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

    #[test]
    fn shader_module_accessors_include_warning_messages() {
        let device = noop_device();
        let module = device.create_shader_module(ShaderModuleSource::Wgsl(
            "diagnostic(info, bogus_rule);\n@compute @workgroup_size(1) fn cs() {}".to_owned(),
        ));

        assert!(!module.is_error());
        assert_eq!(module.diagnostic(), None);
        assert_eq!(module.compilation_messages().len(), 1);
        assert_eq!(
            module.compilation_messages()[0].severity,
            CompilationSeverity::Warning
        );
    }

    #[test]
    fn shader_module_validation_allows_large_binding_numbers_on_noop() {
        let device = noop_device();

        let module = device.create_shader_module(ShaderModuleSource::Wgsl(
            "@group(0) @binding(1000) var<uniform> data: u32;\n\
             @compute @workgroup_size(1) fn cs() { _ = data; }"
                .to_owned(),
        ));

        assert!(!module.is_error());
        assert_eq!(module.diagnostic(), None);
    }

    #[test]
    fn shader_module_validation_rejects_duplicate_override_ids_on_noop() {
        let mut module = naga::Module::default();
        let ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar {
                    kind: naga::ScalarKind::Uint,
                    width: 4,
                }),
            },
            naga::Span::default(),
        );
        let override_ = naga::Override {
            name: None,
            id: Some(0),
            ty,
            init: None,
        };

        module
            .overrides
            .append(override_.clone(), naga::Span::default());
        module.overrides.append(override_, naga::Span::default());

        assert_eq!(
            validate_module_limits(&module),
            Err("duplicate shader override id 0".to_owned())
        );
    }

    #[test]
    fn shader_module_creation_gates_f16_usage_on_required_feature() {
        let source =
            "enable f16;\n@compute @workgroup_size(1) fn cs() { let x: f16 = 1.0h; _ = x; }";
        let device_without_f16 = noop_device();
        device_without_f16.push_error_scope(ErrorFilter::Validation);
        let invalid =
            device_without_f16.create_shader_module(ShaderModuleSource::Wgsl(source.to_owned()));
        let scoped = device_without_f16
            .pop_error_scope()
            .expect("scope should exist")
            .expect("f16 usage should be scoped");

        assert!(invalid.is_error());
        assert!(!scoped.message.is_empty());

        let adapter = noop_adapter();
        let device_with_f16 = adapter
            .create_device(None, &[Feature::ShaderF16], "", "")
            .expect("Noop adapter should create shader-f16 device");
        let valid =
            device_with_f16.create_shader_module(ShaderModuleSource::Wgsl(source.to_owned()));

        assert!(!valid.is_error());
        assert_eq!(valid.diagnostic(), None);
    }

    #[test]
    fn shader_module_creation_keeps_f16_packing_builtins_baseline() {
        let device = noop_device();
        let source = "@compute @workgroup_size(1) fn cs() { let x = pack2x16float(vec2<f32>(1.0, 2.0)); _ = unpack2x16float(x); }";

        let module = device.create_shader_module(ShaderModuleSource::Wgsl(source.to_owned()));

        assert!(!module.is_error());
        assert_eq!(module.diagnostic(), None);
    }
}
