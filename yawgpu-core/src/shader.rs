use std::sync::Arc;

use crate::bind_group_layout::*;
use crate::frontend;
#[cfg(feature = "shader-passthrough")]
use crate::ShaderStage;

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
    /// Raw SPIR-V words.
    #[cfg(feature = "shader-passthrough")]
    SpirvPassthrough(Vec<u32>),
    /// Raw MSL source and caller-provided entry metadata.
    #[cfg(feature = "shader-passthrough")]
    MslPassthrough {
        /// MSL source code.
        source: String,
        /// Entry points declared by the caller.
        entries: Vec<MslEntryPoint>,
    },
    /// Invalid variant.
    Invalid(String),
}

/// Describes one entry point in a raw MSL shader module.
#[cfg(feature = "shader-passthrough")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MslEntryPoint {
    /// Entry point function name.
    pub name: String,
    /// Shader stage for this entry point.
    pub stage: ShaderStage,
    /// Compute workgroup size for compute entries.
    pub workgroup_size: [u32; 3],
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
        reflected: Box<frontend::ReflectedModule>,
    },
    /// Raw SPIR-V words.
    #[cfg(feature = "shader-passthrough")]
    SpirvPassthrough {
        /// SPIR-V words.
        words: Vec<u32>,
    },
    /// Raw MSL source and caller-provided entry metadata.
    #[cfg(feature = "shader-passthrough")]
    MslPassthrough {
        /// MSL source code.
        source: String,
        /// Entry points declared by the caller.
        entries: Vec<MslEntryPoint>,
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
        subgroups: bool,
        dual_source_blending: bool,
        clip_distances: bool,
        primitive_index: bool,
    ) -> Result<ShaderModuleSourceKind, String> {
        let reflected = frontend::parse_and_validate_wgsl_gated(
            &source,
            shader_f16,
            subgroups,
            dual_source_blending,
            clip_distances,
            primitive_index,
        )?;
        Ok(ShaderModuleSourceKind::Wgsl {
            _source: source,
            reflected: Box::new(reflected),
        })
    }

    /// Constructs this object from raw SPIR-V passthrough words.
    #[cfg(feature = "shader-passthrough")]
    pub(crate) fn from_spirv(words: Vec<u32>) -> Result<ShaderModuleSourceKind, String> {
        const SPIRV_MAGIC: u32 = 0x0723_0203;
        if words.is_empty() {
            return Err("SPIR-V shader source must not be empty".to_owned());
        }
        if words[0] != SPIRV_MAGIC {
            return Err("SPIR-V shader source has an invalid magic word".to_owned());
        }
        Ok(ShaderModuleSourceKind::SpirvPassthrough { words })
    }

    /// Constructs this object from raw MSL passthrough source.
    #[cfg(feature = "shader-passthrough")]
    pub(crate) fn from_msl(
        source: String,
        entries: Vec<MslEntryPoint>,
    ) -> Result<ShaderModuleSourceKind, String> {
        if source.is_empty() {
            return Err("MSL shader source must not be empty".to_owned());
        }
        if entries
            .iter()
            .any(|entry| entry.stage == ShaderStage::Compute && entry.workgroup_size.contains(&0))
        {
            return Err("MSL compute entry point workgroup size must be at least one".to_owned());
        }
        Ok(ShaderModuleSourceKind::MslPassthrough { source, entries })
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
    pub(crate) fn reflected(&self) -> Option<&frontend::ReflectedModule> {
        match &self.inner._source {
            ShaderModuleSourceKind::Wgsl { reflected, .. } => Some(reflected),
            _ => None,
        }
    }

    /// Returns raw SPIR-V passthrough words when available.
    #[cfg(feature = "shader-passthrough")]
    #[must_use]
    pub fn spirv_passthrough(&self) -> Option<&[u32]> {
        match &self.inner._source {
            ShaderModuleSourceKind::SpirvPassthrough { words } => Some(words.as_slice()),
            _ => None,
        }
    }

    /// Returns raw MSL passthrough source and entry metadata when available.
    #[cfg(feature = "shader-passthrough")]
    #[must_use]
    pub fn msl_passthrough(&self) -> Option<(&str, &[MslEntryPoint])> {
        match &self.inner._source {
            ShaderModuleSourceKind::MslPassthrough { source, entries } => {
                Some((source.as_str(), entries.as_slice()))
            }
            _ => None,
        }
    }
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
        assert!(scoped.message.contains("unexpected token"));
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
    fn shader_module_creation_gates_subgroups_usage_on_required_feature() {
        let source = r#"
enable subgroups;

@compute @workgroup_size(1)
fn cs() {
  let x = subgroupAdd(1u);
  _ = x;
}
"#;
        let device_without_subgroups = noop_device();
        device_without_subgroups.push_error_scope(ErrorFilter::Validation);
        let invalid = device_without_subgroups
            .create_shader_module(ShaderModuleSource::Wgsl(source.to_owned()));
        let scoped = device_without_subgroups
            .pop_error_scope()
            .expect("scope should exist")
            .expect("subgroups usage should be scoped");

        assert!(invalid.is_error());
        assert!(!scoped.message.is_empty());

        let adapter = noop_adapter();
        let device_with_subgroups = adapter
            .create_device(None, &[Feature::Subgroups], "", "")
            .expect("Noop adapter should create subgroups device");
        let valid =
            device_with_subgroups.create_shader_module(ShaderModuleSource::Wgsl(source.to_owned()));

        assert!(!valid.is_error());
        assert_eq!(valid.diagnostic(), None);
    }

    #[test]
    fn shader_module_creation_gates_dual_source_blending_usage_on_required_feature() {
        let source = r#"
enable dual_source_blending;

struct Out {
  @blend_src(0) @location(0) a: vec4f,
  @blend_src(1) @location(0) b: vec4f,
}

@fragment
fn fs() -> Out {
  return Out(vec4f(), vec4f());
}
"#;
        let device_without_dual_source = noop_device();
        device_without_dual_source.push_error_scope(ErrorFilter::Validation);
        let invalid = device_without_dual_source
            .create_shader_module(ShaderModuleSource::Wgsl(source.to_owned()));
        let scoped = device_without_dual_source
            .pop_error_scope()
            .expect("scope should exist")
            .expect("dual-source blending usage should be scoped");

        assert!(invalid.is_error());
        assert!(!scoped.message.is_empty());

        let adapter = noop_adapter();
        let device_with_dual_source = adapter
            .create_device(None, &[Feature::DualSourceBlending], "", "")
            .expect("Noop adapter should create dual-source-blending device");
        let valid = device_with_dual_source
            .create_shader_module(ShaderModuleSource::Wgsl(source.to_owned()));

        assert!(!valid.is_error());
        assert_eq!(valid.diagnostic(), None);
    }

    #[test]
    fn shader_module_creation_gates_clip_distances_usage_on_required_feature() {
        let source = r#"
enable clip_distances;

struct Out {
  @builtin(position) pos: vec4f,
  @builtin(clip_distances) clip: array<f32, 1>,
}

@vertex
fn vs() -> Out {
  return Out(vec4f(), array<f32, 1>(0.0));
}
"#;
        let device_without_clip_distances = noop_device();
        device_without_clip_distances.push_error_scope(ErrorFilter::Validation);
        let invalid = device_without_clip_distances
            .create_shader_module(ShaderModuleSource::Wgsl(source.to_owned()));
        let scoped = device_without_clip_distances
            .pop_error_scope()
            .expect("scope should exist")
            .expect("clip-distances usage should be scoped");

        assert!(invalid.is_error());
        assert!(!scoped.message.is_empty());

        let adapter = noop_adapter();
        let device_with_clip_distances = adapter
            .create_device(None, &[Feature::ClipDistances], "", "")
            .expect("Noop adapter should create clip-distances device");
        let valid = device_with_clip_distances
            .create_shader_module(ShaderModuleSource::Wgsl(source.to_owned()));

        assert!(!valid.is_error());
        assert_eq!(valid.diagnostic(), None);
    }

    #[test]
    fn shader_module_creation_gates_primitive_index_usage_on_required_feature() {
        let source = r#"
enable primitive_index;

@fragment
fn fs(@builtin(primitive_index) idx: u32) -> @location(0) vec4f {
  return vec4f(f32(idx), 0.0, 0.0, 1.0);
}
"#;
        let device_without_primitive_index = noop_device();
        device_without_primitive_index.push_error_scope(ErrorFilter::Validation);
        let invalid = device_without_primitive_index
            .create_shader_module(ShaderModuleSource::Wgsl(source.to_owned()));
        let scoped = device_without_primitive_index
            .pop_error_scope()
            .expect("scope should exist")
            .expect("primitive-index usage should be scoped");

        assert!(invalid.is_error());
        assert!(!scoped.message.is_empty());

        let adapter = noop_adapter();
        let device_with_primitive_index = adapter
            .create_device(None, &[Feature::PrimitiveIndex], "", "")
            .expect("Noop adapter should create primitive-index device");
        let valid = device_with_primitive_index
            .create_shader_module(ShaderModuleSource::Wgsl(source.to_owned()));

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

    #[cfg(feature = "shader-passthrough")]
    fn valid_spirv_words() -> Vec<u32> {
        vec![0x0723_0203, 0, 0, 0, 0]
    }

    #[cfg(feature = "shader-passthrough")]
    fn msl_entry(name: &str, workgroup_size: [u32; 3]) -> MslEntryPoint {
        MslEntryPoint {
            name: name.to_owned(),
            stage: ShaderStage::Compute,
            workgroup_size,
        }
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn from_spirv_accepts_magic_prefixed_words() {
        let words = valid_spirv_words();

        match ShaderModule::from_spirv(words.clone()).expect("valid SPIR-V") {
            ShaderModuleSourceKind::SpirvPassthrough { words: stored } => {
                assert_eq!(stored, words);
            }
            other => panic!("unexpected source kind: {other:?}"),
        }
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn from_spirv_rejects_empty_source_and_bad_magic() {
        assert_eq!(
            ShaderModule::from_spirv(Vec::new()).expect_err("empty source should fail"),
            "SPIR-V shader source must not be empty"
        );
        assert_eq!(
            ShaderModule::from_spirv(vec![1, 2, 3]).expect_err("bad magic should fail"),
            "SPIR-V shader source has an invalid magic word"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn from_msl_accepts_source_and_entries() {
        let source = "kernel void cs() {}".to_owned();
        let entries = vec![msl_entry("cs", [1, 2, 3])];

        match ShaderModule::from_msl(source.clone(), entries.clone()).expect("valid MSL") {
            ShaderModuleSourceKind::MslPassthrough {
                source: stored,
                entries: stored_entries,
            } => {
                assert_eq!(stored, source);
                assert_eq!(stored_entries, entries);
            }
            other => panic!("unexpected source kind: {other:?}"),
        }
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn from_msl_rejects_empty_source_and_zero_compute_workgroup() {
        assert_eq!(
            ShaderModule::from_msl(String::new(), vec![msl_entry("cs", [1, 1, 1])])
                .expect_err("empty source should fail"),
            "MSL shader source must not be empty"
        );
        assert_eq!(
            ShaderModule::from_msl(
                "kernel void cs() {}".to_owned(),
                vec![msl_entry("cs", [1, 0, 1])]
            )
            .expect_err("zero workgroup component should fail"),
            "MSL compute entry point workgroup size must be at least one"
        );
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn spirv_passthrough_accessor_reports_only_spirv_modules() {
        let spirv = ShaderModule::new(
            ShaderModule::from_spirv(valid_spirv_words()).expect("valid SPIR-V"),
            None,
        );
        assert_eq!(
            spirv.spirv_passthrough().expect("SPIR-V accessor"),
            &[0x0723_0203, 0, 0, 0, 0]
        );

        let msl = ShaderModule::new(
            ShaderModule::from_msl(
                "kernel void cs() {}".to_owned(),
                vec![msl_entry("cs", [1, 1, 1])],
            )
            .expect("valid MSL"),
            None,
        );
        let device = noop_device();
        let wgsl = device.create_shader_module(ShaderModuleSource::Wgsl(
            "@compute @workgroup_size(1) fn cs() {}".to_owned(),
        ));
        let invalid = ShaderModule::new(ShaderModuleSourceKind::Invalid, Some("bad".to_owned()));
        assert!(msl.spirv_passthrough().is_none());
        assert!(wgsl.spirv_passthrough().is_none());
        assert!(invalid.spirv_passthrough().is_none());
    }

    #[cfg(feature = "shader-passthrough")]
    #[test]
    fn msl_passthrough_accessor_reports_only_msl_modules() {
        let msl = ShaderModule::new(
            ShaderModule::from_msl(
                "kernel void cs() {}".to_owned(),
                vec![msl_entry("cs", [1, 1, 1])],
            )
            .expect("valid MSL"),
            None,
        );
        let (source, entries) = msl.msl_passthrough().expect("MSL accessor");
        assert_eq!(source, "kernel void cs() {}");
        assert_eq!(entries[0].name, "cs");

        let device = noop_device();
        let wgsl = device.create_shader_module(ShaderModuleSource::Wgsl(
            "@compute @workgroup_size(1) fn cs() {}".to_owned(),
        ));
        let invalid = ShaderModule::new(ShaderModuleSourceKind::Invalid, Some("bad".to_owned()));
        assert!(wgsl.msl_passthrough().is_none());
        assert!(invalid.msl_passthrough().is_none());
    }
}
