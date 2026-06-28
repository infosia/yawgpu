//! Backend-neutral shader reflection and code-generation data shared by shader frontends.

use std::collections::HashMap;

/// Stores binding metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MslBindingMap {
    /// Resources.
    pub resources: Vec<MslResourceBinding>,
}

/// Stores one MSL buffer-size entry binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MslBufferSizeBinding {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
}

/// Stores MSL resource binding metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MslResourceBinding {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Per-kind Metal slot: buffer-space for `Buffer`, texture-space for
    /// `Texture` and `ExternalTexture` (planes base slot), sampler-space for
    /// `Sampler`.
    pub metal_index: u32,
    /// For `ExternalTexture` only: the buffer-space slot assigned to the params
    /// buffer that carries the external-texture metadata. For all other kinds
    /// this is `None`.
    pub ext_params_buffer_slot: Option<u32>,
    /// Kind.
    pub kind: MslResourceBindingKind,
}

/// Enumerates MSL resource binding kind values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MslResourceBindingKind {
    /// Buffer variant.
    Buffer,
    /// Texture variant.
    Texture,
    /// Sampler variant.
    Sampler,
    /// External texture variant.
    ExternalTexture,
}

/// Stores generated shader source for generated MSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GeneratedMsl {
    /// Source.
    pub source: String,
    /// Entry point.
    pub entry_point: String,
    /// Reserved MSL buffer slot for `_mslBufferSizes`.
    pub buffer_sizes_slot: Option<u32>,
    /// Bindings whose byte lengths populate `_mslBufferSizes`.
    pub buffer_size_bindings: Vec<MslBufferSizeBinding>,
    /// Reserved fragment immediate slot for the frag-depth clamp range.
    pub frag_depth_clamp_slot: Option<u32>,
    /// Per-index threadgroup allocation sizes from Tint's workgroup_allocations,
    /// rounded up to a multiple of 16 for Metal.
    pub workgroup_memory_sizes: Vec<u32>,
}

/// Stores generated shader source for generated GLSL.
#[cfg(feature = "gles")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GeneratedGlsl {
    /// Source.
    pub source: String,
    /// Entry point.
    pub entry_point: String,
}

/// Stores binding metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MslVertexBufferBinding {
    /// Slot.
    pub slot: u32,
    /// Metal index.
    pub metal_index: u32,
    /// Array stride.
    pub array_stride: u64,
    /// Step mode.
    pub step_mode: MslVertexStepMode,
    /// Attributes.
    pub attributes: Vec<MslVertexAttribute>,
}

/// Enumerates MSL vertex step mode values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MslVertexStepMode {
    /// Vertex variant.
    Vertex,
    /// Instance variant.
    Instance,
}

/// Stores attribute metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MslVertexAttribute {
    /// Shader location.
    pub shader_location: u32,
    /// Offset.
    pub offset: u64,
    /// Format.
    pub format: MslVertexFormat,
}

/// Enumerates MSL vertex format values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MslVertexFormat {
    /// Uint8 variant.
    Uint8,
    /// Uint8x2 variant.
    Uint8x2,
    /// Uint8x4 variant.
    Uint8x4,
    /// Sint8 variant.
    Sint8,
    /// Sint8x2 variant.
    Sint8x2,
    /// Sint8x4 variant.
    Sint8x4,
    /// Unorm8 variant.
    Unorm8,
    /// Unorm8x2 variant.
    Unorm8x2,
    /// Unorm8x4 variant.
    Unorm8x4,
    /// Snorm8 variant.
    Snorm8,
    /// Snorm8x2 variant.
    Snorm8x2,
    /// Snorm8x4 variant.
    Snorm8x4,
    /// Uint16 variant.
    Uint16,
    /// Uint16x2 variant.
    Uint16x2,
    /// Uint16x4 variant.
    Uint16x4,
    /// Sint16 variant.
    Sint16,
    /// Sint16x2 variant.
    Sint16x2,
    /// Sint16x4 variant.
    Sint16x4,
    /// Unorm16 variant.
    Unorm16,
    /// Unorm16x2 variant.
    Unorm16x2,
    /// Unorm16x4 variant.
    Unorm16x4,
    /// Snorm16 variant.
    Snorm16,
    /// Snorm16x2 variant.
    Snorm16x2,
    /// Snorm16x4 variant.
    Snorm16x4,
    /// Float16 variant.
    Float16,
    /// Float16x2 variant.
    Float16x2,
    /// Float16x4 variant.
    Float16x4,
    /// Float32 variant.
    Float32,
    /// Float32x2 variant.
    Float32x2,
    /// Float32x3 variant.
    Float32x3,
    /// Float32x4 variant.
    Float32x4,
    /// Uint32 variant.
    Uint32,
    /// Uint32x2 variant.
    Uint32x2,
    /// Uint32x3 variant.
    Uint32x3,
    /// Uint32x4 variant.
    Uint32x4,
    /// Sint32 variant.
    Sint32,
    /// Sint32x2 variant.
    Sint32x2,
    /// Sint32x3 variant.
    Sint32x3,
    /// Sint32x4 variant.
    Sint32x4,
    /// Unorm10_10_10_2 variant.
    Unorm10_10_10_2,
    /// Unorm8x4 BGRA variant.
    Unorm8x4Bgra,
}

/// Enumerates shader stage values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    /// Vertex variant.
    Vertex,
    /// Fragment variant.
    Fragment,
    /// Compute variant.
    Compute,
}

/// Backward-compatible reflected shader-stage name.
pub(crate) type ReflectedShaderStage = ShaderStage;

/// Stores pipeline override constants keyed the same way as WebGPU constant entries.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PipelineConstants {
    /// Constants keyed by override name or decimal override id.
    pub constants: HashMap<String, f64>,
}

impl PipelineConstants {
    /// Creates pipeline constants from an iterator of key-value pairs.
    pub(crate) fn from_iter(iter: impl IntoIterator<Item = (String, f64)>) -> Self {
        Self {
            constants: iter.into_iter().collect(),
        }
    }
}

/// Stores entry metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedEntryPoint {
    /// Name.
    pub name: String,
    /// Stage.
    pub stage: ShaderStage,
}

/// Enumerates reflected type scalar class values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedTypeScalarClass {
    /// Float variant.
    Float,
    /// Sint variant.
    Sint,
    /// Uint variant.
    Uint,
    /// Bool variant.
    Bool,
}

/// Stores reflection data for reflected type class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReflectedTypeClass {
    /// Scalar.
    pub scalar: ReflectedTypeScalarClass,
    /// Components.
    pub components: u8,
    /// Width.
    pub width: u8,
}

/// Stores reflection data for reflected io location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedIoLocation {
    /// Location.
    pub location: u32,
    /// Ty.
    pub ty: ReflectedTypeClass,
    /// Interpolation.
    pub interpolation: Option<ReflectedInterpolation>,
    /// Sampling.
    pub sampling: Option<ReflectedSampling>,
}

/// Enumerates reflected interpolation values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedInterpolation {
    /// Perspective variant.
    Perspective,
    /// Linear variant.
    Linear,
    /// Flat variant.
    Flat,
}

/// Enumerates reflected interpolation sampling values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedSampling {
    /// Center variant.
    Center,
    /// Centroid variant.
    Centroid,
    /// Sample variant.
    Sample,
    /// First variant.
    First,
    /// Either variant.
    Either,
}

/// Stores entry metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedEntryPointIo {
    /// Entry point.
    pub entry_point: String,
    /// Inputs.
    pub inputs: Vec<ReflectedIoLocation>,
    /// Outputs.
    pub outputs: Vec<ReflectedIoLocation>,
    /// Number of stage-input `@builtin`s that consume an inter-stage shader variable slot.
    pub input_inter_stage_builtins: u32,
}

/// Identifies reflected override key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedOverrideKey {
    /// Name.
    pub name: Option<String>,
    /// Id.
    pub id: Option<u16>,
}

/// Stores reflection data for reflected workgroup size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedWorkgroupSize {
    /// Entry point.
    pub entry_point: String,
    /// Literal size.
    pub literal_size: [u32; 3],
    /// Per-dimension override keys for `@workgroup_size(x, y, z)`.
    pub override_keys: [Option<ReflectedOverrideKey>; 3],
    /// Workgroup storage size.
    pub workgroup_storage_size: u64,
}

/// Enumerates reflected buffer type values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedBufferType {
    /// Uniform variant.
    Uniform,
    /// Storage variant.
    Storage,
    /// Read only storage variant.
    ReadOnlyStorage,
}

/// Enumerates reflected texture sample usage values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ReflectedTextureSampleUsage {
    /// Load variant.
    Load,
    /// Sample variant.
    Sample,
    /// Gather variant.
    Gather,
}

/// Stores reflection data for reflected storage texture access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedStorageTextureAccess {
    /// Read.
    pub read: bool,
    /// Write.
    pub write: bool,
}

/// Enumerates reflected resource binding kind values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReflectedResourceBindingKind {
    /// Buffer variant.
    Buffer(ReflectedBufferType),
    /// Sampler variant.
    Sampler {
        /// Comparison variant.
        comparison: bool,
    },
    /// Texture variant.
    Texture {
        /// Sampled variant.
        sampled: bool,
        /// Sample kind variant.
        sample_kind: Option<ReflectedTypeScalarClass>,
        /// Sample usage variant.
        sample_usage: ReflectedTextureSampleUsage,
        /// View dimension variant.
        view_dimension: ReflectedTextureViewDimension,
        /// Multisampled variant.
        multisampled: bool,
    },
    /// Storage texture variant.
    StorageTexture {
        /// Format variant.
        format: String,
        /// Access variant.
        access: ReflectedStorageTextureAccess,
        /// View dimension variant.
        view_dimension: ReflectedTextureViewDimension,
    },
    /// External texture variant.
    ExternalTexture,
}

/// Enumerates reflected texture view dimension values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReflectedTextureViewDimension {
    /// D1 variant.
    D1,
    /// D2 variant.
    D2,
    /// D2 array variant.
    D2Array,
    /// Cube variant.
    Cube,
    /// Cube array variant.
    CubeArray,
    /// D3 variant.
    D3,
}

/// Stores binding metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedResourceBinding {
    /// Group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Kind.
    pub kind: ReflectedResourceBindingKind,
    /// Min binding size.
    pub min_binding_size: u64,
    /// Statically used.
    pub statically_used: bool,
}

/// Stores reflection data for reflected fragment builtins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReflectedFragmentBuiltins {
    /// Entry point.
    pub entry_point: String,
    /// Frag depth.
    pub frag_depth: bool,
    /// Sample mask.
    pub sample_mask: bool,
}

/// Stores reflection data for reflected override.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReflectedOverride {
    /// Name.
    pub name: Option<String>,
    /// Id.
    pub id: Option<u16>,
    /// Ty.
    pub ty: ReflectedTypeClass,
    /// Has default.
    pub has_default: bool,
    /// Default value.
    pub default_value: Option<ReflectedOverrideValue>,
}

/// Enumerates reflected override value values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ReflectedOverrideValue {
    /// Number variant.
    Number(f64),
    /// Bool variant.
    Bool(bool),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_stage_public_variants_are_distinct() {
        assert_ne!(ShaderStage::Vertex, ShaderStage::Fragment);
        assert_ne!(ShaderStage::Fragment, ShaderStage::Compute);
        assert_eq!(format!("{:?}", ShaderStage::Compute), "Compute");
    }
}
