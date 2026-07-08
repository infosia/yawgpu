//! Rust bindings for Dawn's Tint shader compiler, driven through a small C++
//! shim (`shim/tint_shim.cpp`).
//!
//! When built without `YAWGPU_DAWN_DIR` set, the Tint backend is not linked
//! ([`HAVE_TINT`] is `false`) and the public API returns an unavailable error.
//!
//! # Module layout (refactor R5, `specs/tracking/tint-integration-refactor.md`)
//!
//! Every plain data type and enum below (`EntryPoint`, `ResourceBinding`,
//! `Bindings`, `VertexFormat`, ...) is declared exactly once, unconditionally
//! — these are the same shape regardless of whether Tint is linked. Only the
//! `Program` inherent methods, the FFI extern block, and the `RawXxx` mirror
//! structs used to cross the FFI boundary live behind `#[cfg(have_tint)]`
//! (module `real`) or `#[cfg(not(have_tint))]` (module `stub`), which provide
//! the same public method signatures with real-vs-unavailable bodies. Before
//! this refactor the whole public surface (~700 lines of identical data type
//! declarations) was duplicated verbatim between the two `cfg` arms.
#![warn(missing_docs)]

/// Whether this build links the Tint backend.
pub const HAVE_TINT: bool = cfg!(have_tint);

/// Error returned by yawgpu-tint's public API.
///
/// Every variant carries the exact message text this crate has always
/// produced (this type replaces a bare `Result<_, String>`), so `Display`
/// output — and therefore `.to_string()` — is byte-identical to the pre-R5
/// `String` errors. `yawgpu-core`'s `shader_tint.rs` boundary converts back
/// to `String` via `.map_err(|e| e.to_string())`, so WebGPU-facing error
/// message text (some of which CTS matches on) is unaffected by this type's
/// introduction.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TintError {
    /// WGSL parsing or module-level validation failed (from [`Program::parse`]).
    Parse(String),
    /// Code generation (MSL/SPIR-V/GLSL, or workgroup-size/storage-size
    /// resolution) failed.
    Codegen(String),
    /// Reflection (entry points, stage IO, resource bindings, overrides,
    /// diagnostics) failed or returned inconsistent data.
    Reflection(String),
    /// This build does not link Tint (`HAVE_TINT` is `false`).
    Unavailable,
}

impl std::fmt::Display for TintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TintError::Parse(msg) | TintError::Codegen(msg) | TintError::Reflection(msg) => {
                f.write_str(msg)
            }
            TintError::Unavailable => f.write_str(UNAVAILABLE),
        }
    }
}

impl std::error::Error for TintError {}

/// Message carried by [`TintError::Unavailable`] when this build does not
/// link Tint (`YAWGPU_DAWN_DIR` unset at build time).
const UNAVAILABLE: &str = "yawgpu-tint was built without Tint (YAWGPU_DAWN_DIR unset)";

/// A parsed and validated Tint program.
///
/// SAFETY (Send/Sync, refactor finding F3): see the `unsafe impl` blocks
/// below, gated on `have_tint` since a stub `Program` holds no pointer at
/// all and is trivially `Send + Sync`.
#[derive(Debug)]
pub struct Program {
    #[cfg(have_tint)]
    raw: *mut real::RawProgram,
}

// SAFETY: `Program::raw` points to a `tint::Program` owned exclusively by
// this `Program` (freed once, in `Drop`, never aliased). Every method that
// touches it either only reads from the underlying `tint::Program` (this
// paragraph) or serializes itself internally (the reflection accessors,
// below). Verified against this Dawn/Tint revision
// (`third_party/dawn`, submodule pin in the workspace root):
//   - `tint::Program` (`src/tint/lang/wgsl/program/program.h:78-154`) has no
//     `mutable` members and no lazily-computed/cached accessors: `Types()`,
//     `ASTNodes()`, `AST()`, `Sem()`, `Symbols()`, `Diagnostics()`,
//     `IsValid()` are all `const` reads of fields populated once by the
//     `Program(ProgramBuilder&&)` constructor (`program.cc:60-79`) before the
//     object is ever handed to this shim; nothing mutates them afterward.
//   - `sem::Info::Get`/`GetVal` (`src/tint/lang/wgsl/sem/info.h:85-101`) and
//     `sem::Module::DependencyOrderedDeclarations()`
//     (`src/tint/lang/wgsl/sem/module.h:55`) are plain `const` reads of
//     vectors filled during resolution, not lazily on access.
//     `SymbolTable::Get` (`src/tint/utils/symbol/symbol_table.h:78`) is
//     likewise `const` with no mutable state.
//   - `ProgramToLoweredIR` (`src/tint/lang/wgsl/reader/program_to_ir/
//     program_to_ir.cc`) — the entry point every codegen/`resolved_
//     workgroup_size` shim call runs (`lower_ir` in `tint_shim.cpp`) — never
//     touches `program_.Types()`/`program_.ASTNodes()`; its `CloneContext`
//     (`program_to_ir.cc:151-158`) treats the source `Program` as `src` only
//     and allocates exclusively into a fresh, per-call destination
//     `core::ir::Module` (`type::Manager::Get`/`constant::Manager::Get`,
//     both non-const, are called only on `ctx.dst.mgr`, never `ctx.src`).
//     `type::Type::Clone(CloneContext&) const` is itself `const`.
//   - Dawn corroborates this in production: `src/dawn/native/metal/
//     ShaderModuleMTL.mm:373-397` runs `ProgramToLoweredIR` then
//     `msl::writer::Generate` directly on a shared `const tint::Program`
//     (`ShaderModuleMTL.h`/`ShaderModule.h:112-117`, `const tint::Program
//     program;`) from Dawn's async pipeline-creation worker pool, with no
//     lock held across the call — the only mutex
//     (`MutexProtected<TintData> tintData`, `ShaderModule.h:466`) guards
//     acquiring the `Ref<TintProgram>` in `GetTintProgram()`
//     (`ShaderModule.cpp:2019-2041`) and is released before codegen runs.
//     Two concurrent Dawn pipeline-creation tasks compiling the same shader
//     module run Tint codegen on the identical `Program` concurrently,
//     unguarded — the oracle relies on exactly the read-only contract this
//     comment documents.
// The reflection accessors (`entry_points`, `resource_bindings`, ...) go
// through the C++ shim's `reflection_mutex`, which serializes
// `tint::inspector::Inspector` construction (a genuinely separate object
// built fresh per call, not proven read-only by the above) — see
// `tint_shim.cpp`'s `YawgpuTintProgram::reflection_mutex`. Override
// substitution (`generate_msl`, `generate_spirv`, `workgroup_storage_size`,
// `resolved_workgroup_size`, `generate_glsl` — all codegen paths, not
// `reflection_mutex`-guarded) used to be the one exception: it built an
// `Inspector` on the shared `Program` outside the mutex. As of the F3 fix
// (`make_override_config` in `tint_shim.cpp`), it instead reads
// `YawgpuTintProgram::overrides`, a `const`-after-construction vector filled
// once, single-threaded, by `yawgpu_tint_program_create` before the
// `Program` is ever shared across threads — so there is no longer any
// mutex-free `Inspector` construction anywhere in the shim, and every
// codegen path (including override substitution) now falls under the
// read-only `const tint::Program` contract proved above.
#[cfg(have_tint)]
unsafe impl Send for Program {}

// SAFETY: see the `Send` impl above; the same read-only contract makes
// concurrent shared access (`&Program` from multiple threads) sound too.
#[cfg(have_tint)]
unsafe impl Sync for Program {}

/// MSL generator output.
pub struct MslOutput {
    /// Generated MSL source.
    pub source: String,
    /// Generated MSL entry point name.
    pub entry_point: String,
    /// Whether the generated MSL needs a storage-buffer-size table.
    pub needs_storage_buffer_sizes: bool,
    /// Ordered storage bindings whose byte lengths populate the size table.
    pub buffer_size_bindings: Vec<MslBufferSizeBinding>,
    /// Per-index threadgroup memory allocation sizes (compute).
    pub workgroup_allocations: Vec<u32>,
    /// MSL buffer slot of the frag-depth clamp immediate block, if this fragment entry point writes frag_depth.
    pub frag_depth_clamp_slot: Option<u32>,
    /// MSL buffer slot of the combined immediates block (user immediates,
    /// plus the frag-depth clamp range when present), when this entry point
    /// uses any immediates (user `var<immediate>` data or the frag-depth
    /// clamp). `None` when the entry point uses no immediates at all.
    /// Always equal to `frag_depth_clamp_slot` when the latter is `Some`
    /// (same combined block, same slot).
    pub immediate_slot: Option<u32>,
}

/// GLSL generator output.
pub struct GlslOutput {
    /// Generated GLSL source.
    pub source: String,
    /// Combined texture/sampler uniforms emitted by Tint's GLSL writer.
    pub combined_samplers: Vec<CombinedSampler>,
    /// UBO slots emitted by Tint for texture metadata builtins. The binding
    /// points are post-remap, matching Tint's GLSL writer options.
    pub texture_metadata_slots: Vec<TextureMetadataSlot>,
    /// GLES uniform-buffer binding carrying texture metadata for robust
    /// `textureLoad` lowering, when Tint emitted one.
    pub texture_metadata_ubo_binding: Option<u32>,
}

/// A GLSL combined texture/sampler uniform and the WGSL bindings it represents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombinedSampler {
    /// GLSL sampler uniform name.
    pub glsl_uniform_name: String,
    /// WGSL bind group of the texture.
    pub texture_group: u32,
    /// WGSL binding number of the texture.
    pub texture_binding: u32,
    /// WGSL bind group of the sampler, or Tint's placeholder sampler sentinel
    /// when [`uses_placeholder_sampler`](Self::uses_placeholder_sampler) is true.
    pub sampler_group: u32,
    /// WGSL binding number of the sampler, or Tint's placeholder sampler
    /// sentinel binding when [`uses_placeholder_sampler`](Self::uses_placeholder_sampler) is true.
    pub sampler_binding: u32,
    /// Whether this uniform pairs the texture with Tint's placeholder sampler
    /// because WGSL used the texture without an explicit sampler.
    pub uses_placeholder_sampler: bool,
}

/// One Tint texture-metadata UBO slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureMetadataSlot {
    /// Slot index into Tint's packed u32 metadata UBO.
    pub offset: u32,
    /// Post-remap bind group of the queried texture.
    pub group: u32,
    /// Post-remap binding number of the queried texture.
    pub binding: u32,
}

/// A storage binding whose byte length is required by generated MSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MslBufferSizeBinding {
    /// Original WebGPU bind group.
    pub group: u32,
    /// Original WebGPU binding number.
    pub binding: u32,
}

/// A reflected entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPoint {
    /// Entry point name.
    pub name: String,
    /// Pipeline stage.
    pub stage: PipelineStage,
    /// Workgroup size when Tint can resolve it during reflection.
    pub workgroup_size: Option<[u32; 3]>,
    /// Whether the fragment entry point writes frag_depth.
    pub frag_depth_used: bool,
    /// Whether the entry point uses sample_mask.
    pub sample_mask_used: bool,
    /// Whether the entry point reads sample_mask as an input.
    pub input_sample_mask_used: bool,
    /// Whether the entry point reads front_facing.
    pub front_facing_used: bool,
    /// Whether the entry point reads sample_index.
    pub sample_index_used: bool,
    /// Whether the entry point reads primitive_index.
    pub primitive_index_used: bool,
    /// Whether the entry point reads subgroup_invocation_id.
    pub subgroup_invocation_id_used: bool,
    /// Whether the entry point reads subgroup_size.
    pub subgroup_size_used: bool,
    /// Whether the fragment entry point reads `@builtin(position)`
    /// (FragCoord). Drives the Vulkan pixel-center polyfill decision under
    /// sample-rate shading.
    pub frag_position_used: bool,
    /// Size of the vertex clip-distances builtin array, when present.
    pub clip_distances_size: Option<u32>,
}

/// A reflected entry point input or output variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageVariable {
    /// Location attribute, when the variable is location-based IO.
    pub location: Option<u32>,
    /// Color attribute, when the variable is framebuffer-fetch IO.
    pub color: Option<u32>,
    /// Blend source attribute, when the variable is a dual-source blending output.
    pub blend_src: Option<u32>,
    /// Scalar component type.
    pub component_type: ComponentType,
    /// Scalar or vector composition.
    pub composition_type: CompositionType,
    /// Interpolation type.
    pub interpolation_type: InterpolationType,
    /// Interpolation sampling.
    pub interpolation_sampling: InterpolationSampling,
}

/// A non-error diagnostic reported for a valid program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    /// Diagnostic message text.
    pub message: String,
    /// Diagnostic severity.
    pub severity: DiagnosticSeverity,
}

/// Non-error Tint diagnostic severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    /// Informational note.
    Note,
    /// Warning.
    Warning,
}

#[cfg(have_tint)]
impl DiagnosticSeverity {
    fn try_from_raw(raw: u8) -> Result<Self, TintError> {
        match raw {
            0 => Ok(Self::Note),
            1 => Ok(Self::Warning),
            _ => Err(TintError::Reflection(format!(
                "tint: unknown DiagnosticSeverity value {raw}"
            ))),
        }
    }
}

/// Pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Vertex shader stage.
    Vertex,
    /// Fragment shader stage.
    Fragment,
    /// Compute shader stage.
    Compute,
}

#[cfg(have_tint)]
impl PipelineStage {
    fn try_from_raw(raw: u8) -> Result<Self, TintError> {
        match raw {
            0 => Ok(Self::Vertex),
            1 => Ok(Self::Fragment),
            2 => Ok(Self::Compute),
            _ => Err(TintError::Reflection(format!(
                "tint: unknown pipeline stage {raw}"
            ))),
        }
    }
}

/// A reflected resource binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceBinding {
    /// Bind group.
    pub group: u32,
    /// Binding number within the bind group.
    pub binding: u32,
    /// Resource class.
    pub resource_type: ResourceType,
    /// Texture dimension.
    pub dim: TextureDimension,
    /// Sampled texture component kind.
    pub sampled_kind: SampledKind,
    /// Sampler kind.
    pub sampler_type: SamplerType,
    /// Storage texture texel format.
    pub texel_format: TexelFormat,
    /// Strongest sampled texture usage for the queried entry point.
    pub sample_usage: TextureSampleUsage,
    /// Static byte size reported by Tint, or zero when not applicable.
    pub size: u64,
    /// Binding array size when present.
    pub array_size: Option<u32>,
    /// Input-attachment index for input attachment resources; otherwise zero.
    pub input_attachment_index: u32,
}

/// Declares a raw-`u8`-backed reflection enum. The enum itself (and its
/// variants' doc comments) is unconditional/shared; `try_from_raw` — the
/// FFI-facing conversion, the only thing that needs the numeric mapping — is
/// gated to `have_tint` since it is only ever called while reading a shim
/// output struct.
macro_rules! raw_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident { $($(#[$vmeta:meta])* $variant:ident = $value:literal,)* }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name {
            $($(#[$vmeta])* $variant,)*
        }

        #[cfg(have_tint)]
        impl $name {
            fn try_from_raw(raw: u8) -> Result<Self, TintError> {
                match raw {
                    $($value => Ok(Self::$variant),)*
                    _ => Err(TintError::Reflection(format!(
                        "tint: unknown {} value {}",
                        stringify!($name),
                        raw
                    ))),
                }
            }
        }
    };
}

raw_enum! {
    /// Stage variable scalar component type.
    pub enum ComponentType {
        /// f32 component.
        F32 = 0,
        /// u32 component.
        U32 = 1,
        /// i32 component.
        I32 = 2,
        /// f16 component.
        F16 = 3,
        /// Unknown component type.
        Unknown = 4,
    }
}

raw_enum! {
    /// Stage variable scalar or vector composition.
    pub enum CompositionType {
        /// Scalar.
        Scalar = 0,
        /// Two-component vector.
        Vec2 = 1,
        /// Three-component vector.
        Vec3 = 2,
        /// Four-component vector.
        Vec4 = 3,
        /// Unknown composition.
        Unknown = 4,
    }
}

raw_enum! {
    /// Stage variable interpolation type.
    pub enum InterpolationType {
        /// Perspective interpolation.
        Perspective = 0,
        /// Linear interpolation.
        Linear = 1,
        /// Flat interpolation.
        Flat = 2,
        /// Unknown interpolation.
        Unknown = 3,
    }
}

raw_enum! {
    /// Stage variable interpolation sampling.
    pub enum InterpolationSampling {
        /// No sampling value.
        None = 0,
        /// Center sampling.
        Center = 1,
        /// Centroid sampling.
        Centroid = 2,
        /// Sample sampling.
        Sample = 3,
        /// First sampling.
        First = 4,
        /// Either sampling.
        Either = 5,
        /// Unknown sampling.
        Unknown = 6,
    }
}

raw_enum! {
    /// Sampled texture usage for an entry point.
    pub enum TextureSampleUsage {
        /// Texture is only loaded or queried.
        Load = 0,
        /// Texture is sampled.
        Sample = 1,
        /// Texture is gathered.
        Gather = 2,
    }
}

raw_enum! {
    /// Tint resource binding class.
    pub enum ResourceType {
        /// Uniform buffer.
        UniformBuffer = 0,
        /// Writable storage buffer.
        StorageBuffer = 1,
        /// Read-only storage buffer.
        ReadOnlyStorageBuffer = 2,
        /// Sampler.
        Sampler = 3,
        /// Sampled texture.
        SampledTexture = 4,
        /// Multisampled texture.
        MultisampledTexture = 5,
        /// Write-only storage texture.
        WriteOnlyStorageTexture = 6,
        /// Read-only storage texture.
        ReadOnlyStorageTexture = 7,
        /// Read-write storage texture.
        ReadWriteStorageTexture = 8,
        /// Depth texture.
        DepthTexture = 9,
        /// Depth multisampled texture.
        DepthMultisampledTexture = 10,
        /// External texture.
        ExternalTexture = 11,
        /// Read-only texel buffer.
        ReadOnlyTexelBuffer = 12,
        /// Read-write texel buffer.
        ReadWriteTexelBuffer = 13,
        /// Input attachment.
        InputAttachment = 14,
    }
}

raw_enum! {
    /// Texture dimension.
    pub enum TextureDimension {
        /// One-dimensional texture.
        D1 = 0,
        /// Two-dimensional texture.
        D2 = 1,
        /// Two-dimensional array texture.
        D2Array = 2,
        /// Three-dimensional texture.
        D3 = 3,
        /// Cube texture.
        Cube = 4,
        /// Cube array texture.
        CubeArray = 5,
        /// No texture dimension.
        None = 6,
    }
}

raw_enum! {
    /// Sampled texture component kind.
    pub enum SampledKind {
        /// Float component.
        Float = 0,
        /// Unsigned integer component.
        UInt = 1,
        /// Signed integer component.
        SInt = 2,
        /// Filterable float component.
        Filterable = 3,
        /// Unfilterable float component.
        Unfilterable = 4,
        /// Unknown filterability.
        UnknownFilterable = 5,
    }
}

raw_enum! {
    /// Sampler binding kind.
    pub enum SamplerType {
        /// Comparison sampler.
        Comparison = 0,
        /// Filtering sampler.
        Filtering = 1,
        /// Non-filtering sampler.
        NonFiltering = 2,
        /// Unknown filtering mode.
        UnknownFiltering = 3,
    }
}

raw_enum! {
    /// Storage texture texel format.
    pub enum TexelFormat {
        /// r8snorm.
        R8Snorm = 0,
        /// r8uint.
        R8Uint = 1,
        /// r8sint.
        R8Sint = 2,
        /// rg8unorm.
        Rg8Unorm = 3,
        /// rg8snorm.
        Rg8Snorm = 4,
        /// rg8uint.
        Rg8Uint = 5,
        /// rg8sint.
        Rg8Sint = 6,
        /// r16unorm.
        R16Unorm = 7,
        /// r16snorm.
        R16Snorm = 8,
        /// r16uint.
        R16Uint = 9,
        /// r16sint.
        R16Sint = 10,
        /// r16float.
        R16Float = 11,
        /// rg16unorm.
        Rg16Unorm = 12,
        /// rg16snorm.
        Rg16Snorm = 13,
        /// rg16uint.
        Rg16Uint = 14,
        /// rg16sint.
        Rg16Sint = 15,
        /// rg16float.
        Rg16Float = 16,
        /// bgra8unorm.
        Bgra8Unorm = 17,
        /// rgba8unorm.
        Rgba8Unorm = 18,
        /// rgba8snorm.
        Rgba8Snorm = 19,
        /// rgba8uint.
        Rgba8Uint = 20,
        /// rgba8sint.
        Rgba8Sint = 21,
        /// rgba16unorm.
        Rgba16Unorm = 22,
        /// rgba16snorm.
        Rgba16Snorm = 23,
        /// rgba16uint.
        Rgba16Uint = 24,
        /// rgba16sint.
        Rgba16Sint = 25,
        /// rgba16float.
        Rgba16Float = 26,
        /// r32uint.
        R32Uint = 27,
        /// r32sint.
        R32Sint = 28,
        /// r32float.
        R32Float = 29,
        /// rg32uint.
        Rg32Uint = 30,
        /// rg32sint.
        Rg32Sint = 31,
        /// rg32float.
        Rg32Float = 32,
        /// rgba32uint.
        Rgba32Uint = 33,
        /// rgba32sint.
        Rgba32Sint = 34,
        /// rgba32float.
        Rgba32Float = 35,
        /// r8unorm.
        R8Unorm = 36,
        /// rgb10a2uint.
        Rgb10A2Uint = 37,
        /// rgb10a2unorm.
        Rgb10A2Unorm = 38,
        /// rg11b10ufloat.
        Rg11B10Ufloat = 39,
        /// No texel format.
        None = 40,
    }
}

/// A reflected pipeline override.
#[derive(Debug, Clone, PartialEq)]
pub struct Override {
    /// Override name.
    pub name: String,
    /// Numeric override identifier (always assigned by Tint).
    pub id: u16,
    /// Whether the override has an explicit `@id(N)` attribute (vs an id Tint
    /// assigned implicitly). Callers applying WebGPU's "key by numeric id only
    /// for `@id` overrides" rule must consult this rather than `id` alone.
    pub has_explicit_id: bool,
    /// Override scalar type.
    pub type_class: OverrideType,
    /// Whether the override has a default initializer.
    pub has_default: bool,
    /// Reflected default value when Tint exposes it.
    pub default_value: f64,
}

raw_enum! {
    /// Override scalar type.
    pub enum OverrideType {
        /// Boolean override.
        Bool = 0,
        /// f32 override.
        Float32 = 1,
        /// u32 override.
        Uint32 = 2,
        /// i32 override.
        Int32 = 3,
        /// f16 override.
        Float16 = 4,
    }
}

/// A single binding remap from source group/binding to destination group/binding.
///
/// `#[repr(C)]` and passed directly to the FFI as `*const BindingRemap` (no
/// `RawBindingRemap` mirror struct or per-call copy — refactor finding F9);
/// its layout is pinned to the shim's `YawgpuTintBindingRemap` by the
/// `const _` assertions below and the mirrored `static_assert`s in
/// `tint_shim.cpp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BindingRemap {
    /// Source bind group.
    pub group: u32,
    /// Source binding.
    pub binding: u32,
    /// Destination bind group.
    pub dst_group: u32,
    /// Destination binding.
    pub dst_binding: u32,
}

// ABI drift guard: pins `BindingRemap`'s layout to `YawgpuTintBindingRemap`'s
// `static_assert`s in `tint_shim.cpp`. Checked unconditionally (pure Rust
// layout, no Tint link required), so a regression is caught even in a
// Tint-less build.
const _: () = {
    assert!(core::mem::size_of::<BindingRemap>() == 16);
    assert!(core::mem::offset_of!(BindingRemap, group) == 0);
    assert!(core::mem::offset_of!(BindingRemap, binding) == 4);
    assert!(core::mem::offset_of!(BindingRemap, dst_group) == 8);
    assert!(core::mem::offset_of!(BindingRemap, dst_binding) == 12);
};

/// A texture_external remap from the WGSL binding point to Metal texture and metadata slots.
///
/// `#[repr(C)]`; see [`BindingRemap`]'s doc comment for the direct-FFI
/// convention this follows (layout pinned to `YawgpuTintExternalTextureRemap`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ExternalTextureRemap {
    /// WGSL bind group of the texture_external.
    pub group: u32,
    /// WGSL binding of the texture_external.
    pub binding: u32,
    /// Metal texture slot for plane 0.
    pub plane0_slot: u32,
    /// Metal texture slot for plane 1.
    pub plane1_slot: u32,
    /// Metal buffer slot for the external-texture metadata UBO.
    pub params_slot: u32,
}

const _: () = {
    assert!(core::mem::size_of::<ExternalTextureRemap>() == 20);
    assert!(core::mem::offset_of!(ExternalTextureRemap, group) == 0);
    assert!(core::mem::offset_of!(ExternalTextureRemap, binding) == 4);
    assert!(core::mem::offset_of!(ExternalTextureRemap, plane0_slot) == 8);
    assert!(core::mem::offset_of!(ExternalTextureRemap, plane1_slot) == 12);
    assert!(core::mem::offset_of!(ExternalTextureRemap, params_slot) == 16);
};

/// An input_attachment mapping from the WGSL binding point to a Metal color slot.
///
/// `#[repr(C)]`; see [`BindingRemap`]'s doc comment for the direct-FFI
/// convention this follows (layout pinned to
/// `YawgpuTintInputAttachmentColorIndex`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct InputAttachmentColorIndex {
    /// WGSL bind group of the input attachment.
    pub group: u32,
    /// WGSL binding of the input attachment.
    pub binding: u32,
    /// Metal color slot used for `[[color(N)]]` lowering.
    pub color_slot: u32,
}

const _: () = {
    assert!(core::mem::size_of::<InputAttachmentColorIndex>() == 12);
    assert!(core::mem::offset_of!(InputAttachmentColorIndex, group) == 0);
    assert!(core::mem::offset_of!(InputAttachmentColorIndex, binding) == 4);
    assert!(core::mem::offset_of!(InputAttachmentColorIndex, color_slot) == 8);
};

/// Resource binding remap sets grouped by resource class.
#[derive(Debug, Clone, Default)]
pub struct Bindings {
    /// Uniform buffer binding remaps.
    pub uniform: Vec<BindingRemap>,
    /// Storage buffer binding remaps.
    pub storage: Vec<BindingRemap>,
    /// Sampled/depth/external texture binding remaps.
    pub texture: Vec<BindingRemap>,
    /// Storage texture binding remaps.
    pub storage_texture: Vec<BindingRemap>,
    /// Sampler binding remaps.
    pub sampler: Vec<BindingRemap>,
    /// External texture remaps to Metal plane and metadata slots.
    pub external_texture: Vec<ExternalTextureRemap>,
    /// Input attachment remaps to Metal color slots.
    pub input_attachment_color_index: Vec<InputAttachmentColorIndex>,
}

/// Vertex input attribute used by Tint vertex pulling.
///
/// `#[repr(C)]` and passed directly to the FFI as `*const VertexAttribute`
/// (no `RawVertexAttribute` mirror — see [`BindingRemap`]'s doc comment).
/// `format`'s `#[repr(u8)]` gives it the same 1-byte layout the shim expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct VertexAttribute {
    /// WebGPU vertex format.
    pub format: VertexFormat,
    /// Byte offset within the vertex buffer element.
    pub offset: u32,
    /// WGSL shader location.
    pub shader_location: u32,
}

const _: () = {
    assert!(core::mem::size_of::<VertexAttribute>() == 12);
    assert!(core::mem::offset_of!(VertexAttribute, format) == 0);
    assert!(core::mem::offset_of!(VertexAttribute, offset) == 4);
    assert!(core::mem::offset_of!(VertexAttribute, shader_location) == 8);
};

/// WebGPU vertex format used by Tint vertex pulling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VertexFormat {
    /// uint8.
    Uint8 = 0,
    /// uint8x2.
    Uint8x2 = 1,
    /// uint8x4.
    Uint8x4 = 2,
    /// sint8.
    Sint8 = 3,
    /// sint8x2.
    Sint8x2 = 4,
    /// sint8x4.
    Sint8x4 = 5,
    /// unorm8.
    Unorm8 = 6,
    /// unorm8x2.
    Unorm8x2 = 7,
    /// unorm8x4.
    Unorm8x4 = 8,
    /// snorm8.
    Snorm8 = 9,
    /// snorm8x2.
    Snorm8x2 = 10,
    /// snorm8x4.
    Snorm8x4 = 11,
    /// uint16.
    Uint16 = 12,
    /// uint16x2.
    Uint16x2 = 13,
    /// uint16x4.
    Uint16x4 = 14,
    /// sint16.
    Sint16 = 15,
    /// sint16x2.
    Sint16x2 = 16,
    /// sint16x4.
    Sint16x4 = 17,
    /// unorm16.
    Unorm16 = 18,
    /// unorm16x2.
    Unorm16x2 = 19,
    /// unorm16x4.
    Unorm16x4 = 20,
    /// snorm16.
    Snorm16 = 21,
    /// snorm16x2.
    Snorm16x2 = 22,
    /// snorm16x4.
    Snorm16x4 = 23,
    /// float16.
    Float16 = 24,
    /// float16x2.
    Float16x2 = 25,
    /// float16x4.
    Float16x4 = 26,
    /// float32.
    Float32 = 27,
    /// float32x2.
    Float32x2 = 28,
    /// float32x3.
    Float32x3 = 29,
    /// float32x4.
    Float32x4 = 30,
    /// uint32.
    Uint32 = 31,
    /// uint32x2.
    Uint32x2 = 32,
    /// uint32x3.
    Uint32x3 = 33,
    /// uint32x4.
    Uint32x4 = 34,
    /// sint32.
    Sint32 = 35,
    /// sint32x2.
    Sint32x2 = 36,
    /// sint32x3.
    Sint32x3 = 37,
    /// sint32x4.
    Sint32x4 = 38,
    /// unorm10-10-10-2.
    Unorm10_10_10_2 = 39,
    /// unorm8x4-bgra.
    Unorm8x4Bgra = 40,
}

/// Vertex-buffer stepping mode used by Tint vertex pulling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VertexStepMode {
    /// Per-vertex input.
    Vertex = 0,
    /// Per-instance input.
    Instance = 1,
}

/// Vertex buffer layout used by Tint vertex pulling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexBuffer {
    /// WebGPU vertex buffer slot.
    pub slot: u32,
    /// Metal buffer index where the HAL binds this vertex buffer.
    pub metal_index: u32,
    /// Byte stride between elements.
    pub array_stride: u32,
    /// Vertex or instance stepping.
    pub step_mode: VertexStepMode,
    /// Attributes supplied by this buffer.
    pub attributes: Vec<VertexAttribute>,
}

/// Pipeline override substitution value.
#[derive(Debug, Clone, PartialEq)]
pub struct OverrideValue {
    /// Override name or numeric ID encoded as decimal text.
    pub name: String,
    /// Value to substitute, converted by Tint to the declared override type.
    pub value: f64,
}

#[cfg(have_tint)]
mod real {
    //! FFI plumbing: the extern "C" block, `RawXxx` mirror structs still
    //! needed for output-direction reflection or Vec-to-ptr+len flattening,
    //! and the `Program` inherent impl. Everything declared here attaches to
    //! the crate-root shared types (`Program`, `EntryPoint`, `Bindings`, ...)
    //! regardless of this module's own privacy — inherent `impl` blocks are
    //! crate-global, so this module does not need to be `pub`.

    use super::*;
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_void};
    use std::ptr;
    use std::slice;

    #[repr(C)]
    pub(super) struct RawProgram(c_void);

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawEntryPoint {
        name: *const c_char,
        stage: u8,
        has_workgroup_size: bool,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
        frag_depth_used: bool,
        sample_mask_used: bool,
        input_sample_mask_used: bool,
        front_facing_used: bool,
        sample_index_used: bool,
        primitive_index_used: bool,
        subgroup_invocation_id_used: bool,
        subgroup_size_used: bool,
        frag_position_used: bool,
        has_clip_distances: bool,
        clip_distances_size: u32,
    }

    // ABI drift guard: mirrors the `static_assert`s on `YawgpuTintEntryPoint`
    // in tint_shim.cpp. If either side's field order/size changes without
    // updating the other, this fails to compile. See tint_shim.h's "Dawn rev
    // bump" checklist.
    const _: () = {
        assert!(core::mem::size_of::<RawEntryPoint>() == 40);
        assert!(core::mem::offset_of!(RawEntryPoint, name) == 0);
        assert!(core::mem::offset_of!(RawEntryPoint, stage) == 8);
        assert!(core::mem::offset_of!(RawEntryPoint, has_workgroup_size) == 9);
        assert!(core::mem::offset_of!(RawEntryPoint, wg_x) == 12);
        assert!(core::mem::offset_of!(RawEntryPoint, wg_y) == 16);
        assert!(core::mem::offset_of!(RawEntryPoint, wg_z) == 20);
        assert!(core::mem::offset_of!(RawEntryPoint, frag_depth_used) == 24);
        assert!(core::mem::offset_of!(RawEntryPoint, sample_mask_used) == 25);
        assert!(core::mem::offset_of!(RawEntryPoint, input_sample_mask_used) == 26);
        assert!(core::mem::offset_of!(RawEntryPoint, front_facing_used) == 27);
        assert!(core::mem::offset_of!(RawEntryPoint, sample_index_used) == 28);
        assert!(core::mem::offset_of!(RawEntryPoint, primitive_index_used) == 29);
        assert!(core::mem::offset_of!(RawEntryPoint, subgroup_invocation_id_used) == 30);
        assert!(core::mem::offset_of!(RawEntryPoint, subgroup_size_used) == 31);
        assert!(core::mem::offset_of!(RawEntryPoint, frag_position_used) == 32);
        assert!(core::mem::offset_of!(RawEntryPoint, has_clip_distances) == 33);
        assert!(core::mem::offset_of!(RawEntryPoint, clip_distances_size) == 36);
    };

    impl EntryPoint {
        fn try_from_raw(raw: RawEntryPoint) -> Result<Self, TintError> {
            Ok(Self {
                name: raw_string(raw.name),
                stage: PipelineStage::try_from_raw(raw.stage)?,
                workgroup_size: raw
                    .has_workgroup_size
                    .then_some([raw.wg_x, raw.wg_y, raw.wg_z]),
                frag_depth_used: raw.frag_depth_used,
                sample_mask_used: raw.sample_mask_used,
                input_sample_mask_used: raw.input_sample_mask_used,
                front_facing_used: raw.front_facing_used,
                sample_index_used: raw.sample_index_used,
                primitive_index_used: raw.primitive_index_used,
                subgroup_invocation_id_used: raw.subgroup_invocation_id_used,
                subgroup_size_used: raw.subgroup_size_used,
                frag_position_used: raw.frag_position_used,
                clip_distances_size: raw.has_clip_distances.then_some(raw.clip_distances_size),
            })
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawStageVariable {
        has_location: bool,
        location: u32,
        has_color: bool,
        color: u32,
        has_blend_src: bool,
        blend_src: u32,
        component_type: u8,
        composition_type: u8,
        interpolation_type: u8,
        interpolation_sampling: u8,
    }

    // ABI drift guard: mirrors `YawgpuTintStageVariable`'s static_asserts.
    const _: () = {
        assert!(core::mem::size_of::<RawStageVariable>() == 28);
        assert!(core::mem::offset_of!(RawStageVariable, has_location) == 0);
        assert!(core::mem::offset_of!(RawStageVariable, location) == 4);
        assert!(core::mem::offset_of!(RawStageVariable, has_color) == 8);
        assert!(core::mem::offset_of!(RawStageVariable, color) == 12);
        assert!(core::mem::offset_of!(RawStageVariable, has_blend_src) == 16);
        assert!(core::mem::offset_of!(RawStageVariable, blend_src) == 20);
        assert!(core::mem::offset_of!(RawStageVariable, component_type) == 24);
        assert!(core::mem::offset_of!(RawStageVariable, composition_type) == 25);
        assert!(core::mem::offset_of!(RawStageVariable, interpolation_type) == 26);
        assert!(core::mem::offset_of!(RawStageVariable, interpolation_sampling) == 27);
    };

    impl StageVariable {
        fn try_from_raw(raw: RawStageVariable) -> Result<Self, TintError> {
            Ok(Self {
                location: raw.has_location.then_some(raw.location),
                color: raw.has_color.then_some(raw.color),
                blend_src: raw.has_blend_src.then_some(raw.blend_src),
                component_type: ComponentType::try_from_raw(raw.component_type)?,
                composition_type: CompositionType::try_from_raw(raw.composition_type)?,
                interpolation_type: InterpolationType::try_from_raw(raw.interpolation_type)?,
                interpolation_sampling: InterpolationSampling::try_from_raw(
                    raw.interpolation_sampling,
                )?,
            })
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawDiagnostic {
        message: *const c_char,
        severity: u8,
    }

    // ABI drift guard: mirrors `YawgpuTintDiagnostic`'s static_asserts.
    const _: () = {
        assert!(core::mem::size_of::<RawDiagnostic>() == 16);
        assert!(core::mem::offset_of!(RawDiagnostic, message) == 0);
        assert!(core::mem::offset_of!(RawDiagnostic, severity) == 8);
    };

    impl Diagnostic {
        fn try_from_raw(raw: RawDiagnostic) -> Result<Self, TintError> {
            Ok(Self {
                message: raw_string(raw.message),
                severity: DiagnosticSeverity::try_from_raw(raw.severity)?,
            })
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawResourceBinding {
        group: u32,
        binding: u32,
        resource_type: u8,
        dim: u8,
        sampled_kind: u8,
        sampler_type: u8,
        texel_format: u8,
        sample_usage: u8,
        size: u64,
        has_array_size: bool,
        array_size: u32,
        input_attachment_index: u32,
    }

    // ABI drift guard: mirrors `YawgpuTintResourceBinding`'s static_asserts.
    const _: () = {
        assert!(core::mem::size_of::<RawResourceBinding>() == 40);
        assert!(core::mem::offset_of!(RawResourceBinding, group) == 0);
        assert!(core::mem::offset_of!(RawResourceBinding, binding) == 4);
        assert!(core::mem::offset_of!(RawResourceBinding, resource_type) == 8);
        assert!(core::mem::offset_of!(RawResourceBinding, dim) == 9);
        assert!(core::mem::offset_of!(RawResourceBinding, sampled_kind) == 10);
        assert!(core::mem::offset_of!(RawResourceBinding, sampler_type) == 11);
        assert!(core::mem::offset_of!(RawResourceBinding, texel_format) == 12);
        assert!(core::mem::offset_of!(RawResourceBinding, sample_usage) == 13);
        assert!(core::mem::offset_of!(RawResourceBinding, size) == 16);
        assert!(core::mem::offset_of!(RawResourceBinding, has_array_size) == 24);
        assert!(core::mem::offset_of!(RawResourceBinding, array_size) == 28);
        assert!(core::mem::offset_of!(RawResourceBinding, input_attachment_index) == 32);
    };

    impl ResourceBinding {
        fn try_from_raw(raw: RawResourceBinding) -> Result<Self, TintError> {
            Ok(Self {
                group: raw.group,
                binding: raw.binding,
                resource_type: ResourceType::try_from_raw(raw.resource_type)?,
                dim: TextureDimension::try_from_raw(raw.dim)?,
                sampled_kind: SampledKind::try_from_raw(raw.sampled_kind)?,
                sampler_type: SamplerType::try_from_raw(raw.sampler_type)?,
                texel_format: TexelFormat::try_from_raw(raw.texel_format)?,
                sample_usage: TextureSampleUsage::try_from_raw(raw.sample_usage)?,
                size: raw.size,
                array_size: raw.has_array_size.then_some(raw.array_size),
                input_attachment_index: raw.input_attachment_index,
            })
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawOverride {
        name: *const c_char,
        id: u16,
        has_explicit_id: bool,
        type_class: u8,
        has_default: bool,
        default_value: f64,
    }

    // ABI drift guard: mirrors `YawgpuTintOverride`'s static_asserts.
    const _: () = {
        assert!(core::mem::size_of::<RawOverride>() == 24);
        assert!(core::mem::offset_of!(RawOverride, name) == 0);
        assert!(core::mem::offset_of!(RawOverride, id) == 8);
        assert!(core::mem::offset_of!(RawOverride, has_explicit_id) == 10);
        assert!(core::mem::offset_of!(RawOverride, type_class) == 11);
        assert!(core::mem::offset_of!(RawOverride, has_default) == 12);
        assert!(core::mem::offset_of!(RawOverride, default_value) == 16);
    };

    impl Override {
        fn try_from_raw(raw: RawOverride) -> Result<Self, TintError> {
            Ok(Self {
                name: raw_string(raw.name),
                id: raw.id,
                has_explicit_id: raw.has_explicit_id,
                type_class: OverrideType::try_from_raw(raw.type_class)?,
                has_default: raw.has_default,
                default_value: raw.default_value,
            })
        }
    }

    #[repr(C)]
    struct RawBindings<'a> {
        uniform: *const BindingRemap,
        n_uniform: usize,
        storage: *const BindingRemap,
        n_storage: usize,
        texture: *const BindingRemap,
        n_texture: usize,
        storage_texture: *const BindingRemap,
        n_storage_texture: usize,
        sampler: *const BindingRemap,
        n_sampler: usize,
        external_texture: *const ExternalTextureRemap,
        n_external_texture: usize,
        input_attachment_color_index: *const InputAttachmentColorIndex,
        n_input_attachment_color_index: usize,
        // Ties this struct's lifetime to the borrowed `Bindings` (m2, code
        // review of the Tint-integration refactor): every pointer field
        // above points straight into `Bindings`'s own `Vec<BindingRemap>`
        // etc. (see the `as_raw` doc comment below), so the borrow must be
        // compiler-enforced the same way `RawVertexBuffers<'a>` enforces it
        // for `VertexBuffer`, not left as an unbounded raw-pointer bundle.
        // Zero-sized, so it does not perturb any of the `offset_of!`
        // asserts below (repr(C) never reserves space for a trailing ZST).
        _borrow: std::marker::PhantomData<&'a Bindings>,
    }

    // ABI drift guard: mirrors `YawgpuTintBindings`'s static_asserts. All
    // fields are pointer/`usize` pairs (8 bytes each on every yawgpu target),
    // so the size and offsets are platform-stable.
    const _: () = {
        assert!(core::mem::size_of::<RawBindings>() == 112);
        assert!(core::mem::offset_of!(RawBindings, uniform) == 0);
        assert!(core::mem::offset_of!(RawBindings, n_uniform) == 8);
        assert!(core::mem::offset_of!(RawBindings, storage) == 16);
        assert!(core::mem::offset_of!(RawBindings, n_storage) == 24);
        assert!(core::mem::offset_of!(RawBindings, texture) == 32);
        assert!(core::mem::offset_of!(RawBindings, n_texture) == 40);
        assert!(core::mem::offset_of!(RawBindings, storage_texture) == 48);
        assert!(core::mem::offset_of!(RawBindings, n_storage_texture) == 56);
        assert!(core::mem::offset_of!(RawBindings, sampler) == 64);
        assert!(core::mem::offset_of!(RawBindings, n_sampler) == 72);
        assert!(core::mem::offset_of!(RawBindings, external_texture) == 80);
        assert!(core::mem::offset_of!(RawBindings, n_external_texture) == 88);
        assert!(core::mem::offset_of!(RawBindings, input_attachment_color_index) == 96);
        assert!(core::mem::offset_of!(RawBindings, n_input_attachment_color_index) == 104);
    };

    impl Bindings {
        // Borrows straight from `self`'s own `Vec<BindingRemap>` etc. (all
        // direct-FFI PODs since refactor R5 — no `RawXxx` mirror, no per-call
        // copy). Valid for as long as the `&Bindings` caller passed in stays
        // borrowed, which every call site here does (built and used within a
        // single `Program` method body) — and, since m2, compiler-enforced
        // via `RawBindings<'a>`'s `PhantomData<&'a Bindings>` rather than
        // left as an implicit convention.
        fn as_raw(&self) -> RawBindings<'_> {
            RawBindings {
                uniform: self.uniform.as_ptr(),
                n_uniform: self.uniform.len(),
                storage: self.storage.as_ptr(),
                n_storage: self.storage.len(),
                texture: self.texture.as_ptr(),
                n_texture: self.texture.len(),
                storage_texture: self.storage_texture.as_ptr(),
                n_storage_texture: self.storage_texture.len(),
                sampler: self.sampler.as_ptr(),
                n_sampler: self.sampler.len(),
                external_texture: self.external_texture.as_ptr(),
                n_external_texture: self.external_texture.len(),
                input_attachment_color_index: self.input_attachment_color_index.as_ptr(),
                n_input_attachment_color_index: self.input_attachment_color_index.len(),
                _borrow: std::marker::PhantomData,
            }
        }
    }

    #[repr(C)]
    struct RawOverrideValue {
        name: *const c_char,
        value: f64,
    }

    // ABI drift guard: mirrors `YawgpuTintOverrideValue`'s static_asserts.
    const _: () = {
        assert!(core::mem::size_of::<RawOverrideValue>() == 16);
        assert!(core::mem::offset_of!(RawOverrideValue, name) == 0);
        assert!(core::mem::offset_of!(RawOverrideValue, value) == 8);
    };

    #[repr(C)]
    struct RawVertexBuffer {
        slot: u32,
        metal_index: u32,
        array_stride: u32,
        step_mode: u8,
        attributes: *const VertexAttribute,
        n_attributes: usize,
    }

    // ABI drift guard: mirrors `YawgpuTintVertexBuffer`'s static_asserts.
    const _: () = {
        assert!(core::mem::size_of::<RawVertexBuffer>() == 32);
        assert!(core::mem::offset_of!(RawVertexBuffer, slot) == 0);
        assert!(core::mem::offset_of!(RawVertexBuffer, metal_index) == 4);
        assert!(core::mem::offset_of!(RawVertexBuffer, array_stride) == 8);
        assert!(core::mem::offset_of!(RawVertexBuffer, step_mode) == 12);
        assert!(core::mem::offset_of!(RawVertexBuffer, attributes) == 16);
        assert!(core::mem::offset_of!(RawVertexBuffer, n_attributes) == 24);
    };

    #[repr(C)]
    struct RawMslOutput {
        msl: *mut c_char,
        entry_point: *mut c_char,
        needs_storage_buffer_sizes: bool,
        buffer_size_bindings: *mut u32,
        n_buffer_size_bindings: usize,
        workgroup_allocations: *mut u32,
        n_workgroup_allocations: usize,
        has_frag_depth_clamp: bool,
        frag_depth_clamp_slot: u32,
        uses_immediates: bool,
        immediate_slot: u32,
    }

    // ABI drift guard: mirrors `YawgpuTintMslOutput`'s static_asserts.
    const _: () = {
        assert!(core::mem::size_of::<RawMslOutput>() == 72);
        assert!(core::mem::offset_of!(RawMslOutput, msl) == 0);
        assert!(core::mem::offset_of!(RawMslOutput, entry_point) == 8);
        assert!(core::mem::offset_of!(RawMslOutput, needs_storage_buffer_sizes) == 16);
        assert!(core::mem::offset_of!(RawMslOutput, buffer_size_bindings) == 24);
        assert!(core::mem::offset_of!(RawMslOutput, n_buffer_size_bindings) == 32);
        assert!(core::mem::offset_of!(RawMslOutput, workgroup_allocations) == 40);
        assert!(core::mem::offset_of!(RawMslOutput, n_workgroup_allocations) == 48);
        assert!(core::mem::offset_of!(RawMslOutput, has_frag_depth_clamp) == 56);
        assert!(core::mem::offset_of!(RawMslOutput, frag_depth_clamp_slot) == 60);
        assert!(core::mem::offset_of!(RawMslOutput, uses_immediates) == 64);
        assert!(core::mem::offset_of!(RawMslOutput, immediate_slot) == 68);
    };

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawCombinedSampler {
        glsl_uniform_name: *mut c_char,
        texture_group: u32,
        texture_binding: u32,
        sampler_group: u32,
        sampler_binding: u32,
        uses_placeholder_sampler: bool,
    }

    const _: () = {
        assert!(core::mem::size_of::<RawCombinedSampler>() == 32);
        assert!(core::mem::offset_of!(RawCombinedSampler, glsl_uniform_name) == 0);
        assert!(core::mem::offset_of!(RawCombinedSampler, texture_group) == 8);
        assert!(core::mem::offset_of!(RawCombinedSampler, texture_binding) == 12);
        assert!(core::mem::offset_of!(RawCombinedSampler, sampler_group) == 16);
        assert!(core::mem::offset_of!(RawCombinedSampler, sampler_binding) == 20);
        assert!(core::mem::offset_of!(RawCombinedSampler, uses_placeholder_sampler) == 24);
    };

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawTextureMetadataSlot {
        offset: u32,
        group: u32,
        binding: u32,
    }

    const _: () = {
        assert!(core::mem::size_of::<RawTextureMetadataSlot>() == 12);
        assert!(core::mem::offset_of!(RawTextureMetadataSlot, offset) == 0);
        assert!(core::mem::offset_of!(RawTextureMetadataSlot, group) == 4);
        assert!(core::mem::offset_of!(RawTextureMetadataSlot, binding) == 8);
    };

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawGlslOutput {
        glsl: *mut c_char,
        combined_samplers: *mut RawCombinedSampler,
        n_combined_samplers: usize,
        texture_metadata_slots: *mut RawTextureMetadataSlot,
        n_texture_metadata_slots: usize,
        has_texture_metadata_ubo: bool,
        texture_metadata_ubo_binding: u32,
    }

    const _: () = {
        assert!(core::mem::size_of::<RawGlslOutput>() == 48);
        assert!(core::mem::offset_of!(RawGlslOutput, glsl) == 0);
        assert!(core::mem::offset_of!(RawGlslOutput, combined_samplers) == 8);
        assert!(core::mem::offset_of!(RawGlslOutput, n_combined_samplers) == 16);
        assert!(core::mem::offset_of!(RawGlslOutput, texture_metadata_slots) == 24);
        assert!(core::mem::offset_of!(RawGlslOutput, n_texture_metadata_slots) == 32);
        assert!(core::mem::offset_of!(RawGlslOutput, has_texture_metadata_ubo) == 40);
        assert!(core::mem::offset_of!(RawGlslOutput, texture_metadata_ubo_binding) == 44);
    };

    extern "C" {
        fn yawgpu_tint_initialize();
        fn yawgpu_tint_program_create(
            wgsl: *const c_char,
            wgsl_len: usize,
            shader_f16: bool,
            subgroups: bool,
            dual_source_blending: bool,
            clip_distances: bool,
            primitive_index: bool,
            allow_framebuffer_fetch: bool,
            lang_features: *const u32,
            n_lang_features: usize,
            err: *mut *mut c_char,
        ) -> *mut RawProgram;
        fn yawgpu_tint_program_destroy(program: *mut RawProgram);
        fn yawgpu_tint_entry_point_count(program: *const RawProgram) -> usize;
        fn yawgpu_tint_entry_point_get(
            program: *const RawProgram,
            i: usize,
            out: *mut RawEntryPoint,
        ) -> bool;
        fn yawgpu_tint_entry_point_input_count(
            program: *const RawProgram,
            ep: *const c_char,
        ) -> usize;
        fn yawgpu_tint_entry_point_input_get(
            program: *const RawProgram,
            ep: *const c_char,
            i: usize,
            out: *mut RawStageVariable,
        ) -> bool;
        fn yawgpu_tint_entry_point_output_count(
            program: *const RawProgram,
            ep: *const c_char,
        ) -> usize;
        fn yawgpu_tint_entry_point_output_get(
            program: *const RawProgram,
            ep: *const c_char,
            i: usize,
            out: *mut RawStageVariable,
        ) -> bool;
        fn yawgpu_tint_diagnostic_count(program: *const RawProgram) -> usize;
        fn yawgpu_tint_diagnostic_get(
            program: *const RawProgram,
            i: usize,
            out: *mut RawDiagnostic,
        ) -> bool;
        fn yawgpu_tint_resource_binding_count(
            program: *const RawProgram,
            ep: *const c_char,
        ) -> usize;
        fn yawgpu_tint_resource_binding_get(
            program: *const RawProgram,
            ep: *const c_char,
            i: usize,
            out: *mut RawResourceBinding,
        ) -> bool;
        fn yawgpu_tint_override_count(program: *const RawProgram) -> usize;
        fn yawgpu_tint_override_get(
            program: *const RawProgram,
            i: usize,
            out: *mut RawOverride,
        ) -> bool;
        fn yawgpu_tint_generate_msl(
            program: *const RawProgram,
            ep: *const c_char,
            bindings: *const RawBindings<'_>,
            ov: *const RawOverrideValue,
            n_ov: usize,
            buffer_sizes_slot: u32,
            disable_robustness: bool,
            emit_vertex_point_size: bool,
            vertex_buffers: *const RawVertexBuffer,
            n_vertex_buffers: usize,
            fixed_sample_mask: u32,
            user_immediate_size: u32,
            out: *mut RawMslOutput,
            err: *mut *mut c_char,
        ) -> bool;
        fn yawgpu_tint_generate_spirv(
            program: *const RawProgram,
            ep: *const c_char,
            bindings: *const RawBindings<'_>,
            ov: *const RawOverrideValue,
            n_ov: usize,
            disable_robustness: bool,
            use_vulkan_memory_model: bool,
            framebuffer_fetch_descriptor_set: u32,
            multisampled_input_attachment: bool,
            has_polyfill_pixel_center: bool,
            polyfill_pixel_center: u32,
            user_immediate_size: u32,
            words_out: *mut *mut u32,
            n_words_out: *mut usize,
            err: *mut *mut c_char,
        ) -> bool;
        fn yawgpu_tint_workgroup_storage_size(
            program: *const RawProgram,
            ov: *const RawOverrideValue,
            n_ov: usize,
            out: *mut u64,
            err: *mut *mut c_char,
        ) -> bool;
        fn yawgpu_tint_resolved_workgroup_size(
            program: *const RawProgram,
            ep: *const c_char,
            ov: *const RawOverrideValue,
            n_ov: usize,
            out: *mut u32,
            err: *mut *mut c_char,
        ) -> bool;
        fn yawgpu_tint_immediate_data_size(
            program: *const RawProgram,
            ep: *const c_char,
            out: *mut u32,
            err: *mut *mut c_char,
        ) -> bool;
        fn yawgpu_tint_immediate_data_used_slots(
            program: *const RawProgram,
            ep: *const c_char,
            out: *mut u64,
            err: *mut *mut c_char,
        ) -> bool;
        fn yawgpu_tint_generate_glsl(
            program: *const RawProgram,
            ep: *const c_char,
            bindings: *const RawBindings<'_>,
            ov: *const RawOverrideValue,
            n_ov: usize,
            has_first_instance_offset: bool,
            first_instance_offset: u32,
            out: *mut RawGlslOutput,
            err: *mut *mut c_char,
        ) -> bool;
        fn yawgpu_tint_string_free(s: *mut c_char);
        fn yawgpu_tint_u32_free(s: *mut u32);
        fn yawgpu_tint_glsl_output_free(out: *mut RawGlslOutput);
    }

    fn take_error(err: *mut c_char) -> String {
        if err.is_null() {
            return "tint: unknown error".to_owned();
        }
        // SAFETY: error strings are allocated by the shim and NUL-terminated.
        unsafe {
            let msg = CStr::from_ptr(err).to_string_lossy().into_owned();
            yawgpu_tint_string_free(err);
            msg
        }
    }

    fn cstring(name: &str, what: &str) -> Result<CString, String> {
        CString::new(name).map_err(|_| format!("{what} contains an interior NUL"))
    }

    fn raw_string(ptr: *const c_char) -> String {
        if ptr.is_null() {
            return String::new();
        }
        // SAFETY: reflection strings are borrowed from the live program handle.
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    /// Frees a shim-owned, possibly-null `*mut c_char` on drop unless
    /// [`StringGuard::into_string`] has already consumed it. Makes every
    /// generator's multi-pointer failure path leak-free by construction
    /// instead of hand-freeing 2-4 pointers per branch (refactor finding F9).
    struct StringGuard(*mut c_char);

    impl StringGuard {
        fn new(ptr: *mut c_char) -> Self {
            Self(ptr)
        }

        fn is_null(&self) -> bool {
            self.0.is_null()
        }

        /// Reads the shim buffer into an owned `String` and frees it.
        /// Returns an empty string for a null pointer (callers needing to
        /// distinguish that case check [`Self::is_null`] first).
        fn into_string(self) -> String {
            let ptr = self.0;
            // Ownership of `ptr` moves into the unsafe block below; skip
            // `Drop` so it is not freed twice.
            std::mem::forget(self);
            if ptr.is_null() {
                return String::new();
            }
            // SAFETY: `ptr` is a non-null, shim-owned, NUL-terminated buffer,
            // and this is the only place that reads+frees it (the `Drop`
            // impl below was just disarmed via `mem::forget`).
            unsafe {
                let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
                yawgpu_tint_string_free(ptr);
                s
            }
        }
    }

    impl Drop for StringGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: shim-owned buffer, freed at most once (here, or by
                // `into_string`, which disarms this impl via `mem::forget`).
                unsafe { yawgpu_tint_string_free(self.0) };
            }
        }
    }

    /// Frees a shim-owned, possibly-null `*mut u32` buffer on drop. See
    /// [`StringGuard`].
    struct U32Guard(*mut u32);

    impl U32Guard {
        fn new(ptr: *mut u32) -> Self {
            Self(ptr)
        }

        fn is_null(&self) -> bool {
            self.0.is_null()
        }

        /// Borrows `len` initialized words without transferring ownership;
        /// the buffer is freed when the guard drops. Returns an empty slice
        /// for a null pointer or zero length.
        fn as_slice(&self, len: usize) -> &[u32] {
            if self.0.is_null() || len == 0 {
                &[]
            } else {
                // SAFETY: caller-provided `len` matches the shim's reported
                // element count for this buffer.
                unsafe { slice::from_raw_parts(self.0, len) }
            }
        }
    }

    impl Drop for U32Guard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: shim-owned buffer, freed exactly once.
                unsafe { yawgpu_tint_u32_free(self.0) };
            }
        }
    }

    struct GlslOutputGuard(RawGlslOutput);

    impl GlslOutputGuard {
        fn new(raw: RawGlslOutput) -> Self {
            Self(raw)
        }

        fn is_glsl_null(&self) -> bool {
            self.0.glsl.is_null()
        }

        fn into_output(self) -> GlslOutput {
            let raw = self.0;
            std::mem::forget(self);
            let source = raw_string(raw.glsl);
            let combined_samplers =
                if raw.combined_samplers.is_null() || raw.n_combined_samplers == 0 {
                    Vec::new()
                } else {
                    // SAFETY: `combined_samplers` points to `n_combined_samplers`
                    // initialized shim-owned records until the output is freed.
                    unsafe { slice::from_raw_parts(raw.combined_samplers, raw.n_combined_samplers) }
                        .iter()
                        .map(|raw| CombinedSampler {
                            glsl_uniform_name: raw_string(raw.glsl_uniform_name),
                            texture_group: raw.texture_group,
                            texture_binding: raw.texture_binding,
                            sampler_group: raw.sampler_group,
                            sampler_binding: raw.sampler_binding,
                            uses_placeholder_sampler: raw.uses_placeholder_sampler,
                        })
                        .collect()
                };
            let texture_metadata_slots = if raw.texture_metadata_slots.is_null()
                || raw.n_texture_metadata_slots == 0
            {
                Vec::new()
            } else {
                // SAFETY: `texture_metadata_slots` points to
                // `n_texture_metadata_slots` initialized shim-owned records
                // until the output is freed.
                unsafe {
                    slice::from_raw_parts(raw.texture_metadata_slots, raw.n_texture_metadata_slots)
                }
                .iter()
                .map(|raw| TextureMetadataSlot {
                    offset: raw.offset,
                    group: raw.group,
                    binding: raw.binding,
                })
                .collect()
            };
            let mut raw = raw;
            // SAFETY: all pointed-to allocations are shim-owned and have been
            // copied into Rust-owned values above.
            unsafe { yawgpu_tint_glsl_output_free(&mut raw) };
            GlslOutput {
                source,
                combined_samplers,
                texture_metadata_slots,
                texture_metadata_ubo_binding: raw
                    .has_texture_metadata_ubo
                    .then_some(raw.texture_metadata_ubo_binding),
            }
        }
    }

    impl Drop for GlslOutputGuard {
        fn drop(&mut self) {
            // SAFETY: frees any shim-owned fields if `into_output` did not
            // already consume them.
            unsafe { yawgpu_tint_glsl_output_free(&mut self.0) };
        }
    }

    /// Reinterprets a `U32Guard`-owned buffer of `len` `(group, binding)`
    /// pairs (`2 * len` words) into [`MslBufferSizeBinding`]s. Split out of
    /// `Program::generate_msl` since it is the one output field needing more
    /// than a plain slice-to-`Vec` copy.
    fn buffer_size_bindings_from_guard(
        guard: &U32Guard,
        len: usize,
    ) -> Result<Vec<MslBufferSizeBinding>, TintError> {
        if len == 0 {
            return Ok(Vec::new());
        }
        if guard.is_null() {
            return Err(TintError::Codegen(
                "tint: MSL buffer size bindings pointer was NULL".to_owned(),
            ));
        }
        let Some(word_len) = len.checked_mul(2) else {
            return Err(TintError::Codegen(
                "tint: MSL buffer size bindings length overflowed".to_owned(),
            ));
        };
        Ok(guard
            .as_slice(word_len)
            .chunks_exact(2)
            .map(|pair| MslBufferSizeBinding {
                group: pair[0],
                binding: pair[1],
            })
            .collect())
    }

    /// Initializes the Tint runtime.
    pub fn initialize() {
        // SAFETY: the shim guards Tint initialization.
        unsafe { yawgpu_tint_initialize() }
    }

    impl Program {
        /// Parses WGSL into a Tint program.
        pub fn parse(
            wgsl: &str,
            shader_f16: bool,
            subgroups: bool,
            dual_source_blending: bool,
            clip_distances: bool,
            primitive_index: bool,
            language_features: &[u32],
        ) -> Result<Self, TintError> {
            initialize();
            let mut err = ptr::null_mut();
            // SAFETY: `wgsl` is valid for the call and the shim copies it using the length.
            // `language_features` is valid for the duration of the call.
            let raw = unsafe {
                yawgpu_tint_program_create(
                    wgsl.as_ptr().cast::<c_char>(),
                    wgsl.len(),
                    shader_f16,
                    subgroups,
                    dual_source_blending,
                    clip_distances,
                    primitive_index,
                    cfg!(feature = "tiled"),
                    language_features.as_ptr(),
                    language_features.len(),
                    &mut err,
                )
            };
            if raw.is_null() {
                Err(TintError::Parse(take_error(err)))
            } else {
                Ok(Self { raw })
            }
        }

        /// Returns reflected entry points.
        pub fn entry_points(&self) -> Result<Vec<EntryPoint>, TintError> {
            // SAFETY: `self.raw` is owned by this Program.
            let count = unsafe { yawgpu_tint_entry_point_count(self.raw) };
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let mut raw = RawEntryPoint {
                    name: ptr::null(),
                    stage: 0,
                    has_workgroup_size: false,
                    wg_x: 0,
                    wg_y: 0,
                    wg_z: 0,
                    frag_depth_used: false,
                    sample_mask_used: false,
                    input_sample_mask_used: false,
                    front_facing_used: false,
                    sample_index_used: false,
                    primitive_index_used: false,
                    subgroup_invocation_id_used: false,
                    subgroup_size_used: false,
                    frag_position_used: false,
                    has_clip_distances: false,
                    clip_distances_size: 0,
                };
                // SAFETY: `raw` points to valid writable memory.
                let ok = unsafe { yawgpu_tint_entry_point_get(self.raw, i, &mut raw) };
                if !ok {
                    return Err(TintError::Reflection(
                        "tint: failed to read entry point reflection".to_owned(),
                    ));
                }
                out.push(EntryPoint::try_from_raw(raw)?);
            }
            Ok(out)
        }

        /// Returns stage input variables used by `entry_point`.
        pub fn entry_point_inputs(
            &self,
            entry_point: &str,
        ) -> Result<Vec<StageVariable>, TintError> {
            self.entry_point_variables(entry_point, StageVariableDirection::Input)
        }

        /// Returns stage output variables used by `entry_point`.
        pub fn entry_point_outputs(
            &self,
            entry_point: &str,
        ) -> Result<Vec<StageVariable>, TintError> {
            self.entry_point_variables(entry_point, StageVariableDirection::Output)
        }

        /// Returns non-error diagnostics reported for this valid program.
        pub fn diagnostics(&self) -> Result<Vec<Diagnostic>, TintError> {
            // SAFETY: `self.raw` is owned by this Program.
            let count = unsafe { yawgpu_tint_diagnostic_count(self.raw) };
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let mut raw = RawDiagnostic {
                    message: ptr::null(),
                    severity: 0,
                };
                // SAFETY: `raw` points to valid writable memory.
                let ok = unsafe { yawgpu_tint_diagnostic_get(self.raw, i, &mut raw) };
                if !ok {
                    return Err(TintError::Reflection(
                        "tint: failed to read diagnostic".to_owned(),
                    ));
                }
                out.push(Diagnostic::try_from_raw(raw)?);
            }
            Ok(out)
        }

        fn entry_point_variables(
            &self,
            entry_point: &str,
            direction: StageVariableDirection,
        ) -> Result<Vec<StageVariable>, TintError> {
            let ep = cstring(entry_point, "entry point").map_err(TintError::Reflection)?;
            // SAFETY: pointers are valid for the duration of the call.
            let count = unsafe {
                match direction {
                    StageVariableDirection::Input => {
                        yawgpu_tint_entry_point_input_count(self.raw, ep.as_ptr())
                    }
                    StageVariableDirection::Output => {
                        yawgpu_tint_entry_point_output_count(self.raw, ep.as_ptr())
                    }
                }
            };
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let mut raw = RawStageVariable {
                    has_location: false,
                    location: 0,
                    has_color: false,
                    color: 0,
                    has_blend_src: false,
                    blend_src: 0,
                    component_type: 0,
                    composition_type: 0,
                    interpolation_type: 0,
                    interpolation_sampling: 0,
                };
                // SAFETY: pointers are valid for the duration of the call.
                let ok = unsafe {
                    match direction {
                        StageVariableDirection::Input => {
                            yawgpu_tint_entry_point_input_get(self.raw, ep.as_ptr(), i, &mut raw)
                        }
                        StageVariableDirection::Output => {
                            yawgpu_tint_entry_point_output_get(self.raw, ep.as_ptr(), i, &mut raw)
                        }
                    }
                };
                if !ok {
                    return Err(TintError::Reflection(
                        "tint: failed to read stage variable reflection".to_owned(),
                    ));
                }
                out.push(StageVariable::try_from_raw(raw)?);
            }
            Ok(out)
        }

        /// Returns resource bindings used by `entry_point`.
        pub fn resource_bindings(
            &self,
            entry_point: &str,
        ) -> Result<Vec<ResourceBinding>, TintError> {
            let ep = cstring(entry_point, "entry point").map_err(TintError::Reflection)?;
            // SAFETY: pointers are valid for the duration of the call.
            let count = unsafe { yawgpu_tint_resource_binding_count(self.raw, ep.as_ptr()) };
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let mut raw = RawResourceBinding {
                    group: 0,
                    binding: 0,
                    resource_type: 0,
                    dim: 0,
                    sampled_kind: 0,
                    sampler_type: 0,
                    texel_format: 0,
                    sample_usage: 0,
                    size: 0,
                    has_array_size: false,
                    array_size: 0,
                    input_attachment_index: 0,
                };
                // SAFETY: pointers are valid for the duration of the call.
                let ok =
                    unsafe { yawgpu_tint_resource_binding_get(self.raw, ep.as_ptr(), i, &mut raw) };
                if !ok {
                    return Err(TintError::Reflection(
                        "tint: failed to read resource binding reflection".to_owned(),
                    ));
                }
                out.push(ResourceBinding::try_from_raw(raw)?);
            }
            Ok(out)
        }

        /// Returns module override declarations.
        pub fn overrides(&self) -> Result<Vec<Override>, TintError> {
            // SAFETY: `self.raw` is owned by this Program.
            let count = unsafe { yawgpu_tint_override_count(self.raw) };
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let mut raw = RawOverride {
                    name: ptr::null(),
                    id: 0,
                    has_explicit_id: false,
                    type_class: 0,
                    has_default: false,
                    default_value: 0.0,
                };
                // SAFETY: `raw` points to valid writable memory.
                let ok = unsafe { yawgpu_tint_override_get(self.raw, i, &mut raw) };
                if !ok {
                    return Err(TintError::Reflection(
                        "tint: failed to read override reflection".to_owned(),
                    ));
                }
                out.push(Override::try_from_raw(raw)?);
            }
            Ok(out)
        }

        /// Generates MSL for `entry_point`.
        ///
        /// `user_immediate_size` is the owning pipeline layout's reserved
        /// user-immediate byte budget (`PipelineLayout.immediateSize`, 0..=64,
        /// Block 94) -- NOT this entry point's own smaller reflected
        /// `immediate_data_size`. It sets the byte boundary after which any
        /// pipeline-internal immediates (currently only the fragment
        /// frag-depth clamp range) are appended, mirroring Dawn's
        /// `GetImmediateByteOffsetInPipeline` (`ImmediatesLayout.h`), which
        /// places `ClampFragDepthArgs` after the full layout-reserved user
        /// region rather than after the entry point's own (possibly smaller)
        /// declared usage.
        #[allow(clippy::too_many_arguments)]
        pub fn generate_msl(
            &self,
            entry_point: &str,
            bindings: &Bindings,
            overrides: &[OverrideValue],
            buffer_sizes_slot: u32,
            robust: bool,
            emit_vertex_point_size: bool,
            vertex_buffers: &[VertexBuffer],
            fixed_sample_mask: u32,
            user_immediate_size: u32,
        ) -> Result<MslOutput, TintError> {
            let ep = cstring(entry_point, "entry point").map_err(TintError::Codegen)?;
            let raw_bindings = bindings.as_raw();
            let raw_overrides = RawOverrideValues::new(overrides)?;
            let raw_vertex_buffers = RawVertexBuffers::new(vertex_buffers);
            let mut out = RawMslOutput {
                msl: ptr::null_mut(),
                entry_point: ptr::null_mut(),
                needs_storage_buffer_sizes: false,
                buffer_size_bindings: ptr::null_mut(),
                n_buffer_size_bindings: 0,
                workgroup_allocations: ptr::null_mut(),
                n_workgroup_allocations: 0,
                has_frag_depth_clamp: false,
                frag_depth_clamp_slot: 0,
                uses_immediates: false,
                immediate_slot: 0,
            };
            let mut err = ptr::null_mut();
            // SAFETY: all pointers are valid for the duration of the call.
            let ok = unsafe {
                yawgpu_tint_generate_msl(
                    self.raw,
                    ep.as_ptr(),
                    &raw_bindings,
                    raw_overrides.as_ptr(),
                    raw_overrides.len(),
                    buffer_sizes_slot,
                    !robust,
                    emit_vertex_point_size,
                    raw_vertex_buffers.as_ptr(),
                    raw_vertex_buffers.len(),
                    fixed_sample_mask,
                    user_immediate_size,
                    &mut out,
                    &mut err,
                )
            };
            if !ok {
                return Err(TintError::Codegen(take_error(err)));
            }
            // Every allocated out-pointer is wrapped in a free-on-drop guard
            // immediately after the call succeeds, so every early return
            // below (the NULL checks, the `?` on `buffer_size_bindings_from_
            // guard`) stays leak-free without a hand-written free ladder per
            // branch (refactor finding F9).
            let msl_guard = StringGuard::new(out.msl);
            let entry_point_guard = StringGuard::new(out.entry_point);
            let buffer_size_bindings_guard = U32Guard::new(out.buffer_size_bindings);
            let workgroup_allocations_guard = U32Guard::new(out.workgroup_allocations);
            if msl_guard.is_null() {
                return Err(TintError::Codegen(
                    "tint: MSL generator returned NULL output".to_owned(),
                ));
            }
            if entry_point_guard.is_null() {
                return Err(TintError::Codegen(
                    "tint: MSL generator returned NULL entry point".to_owned(),
                ));
            }
            let workgroup_allocations = workgroup_allocations_guard
                .as_slice(out.n_workgroup_allocations)
                .to_vec();
            let buffer_size_bindings = buffer_size_bindings_from_guard(
                &buffer_size_bindings_guard,
                out.n_buffer_size_bindings,
            )?;
            let frag_depth_clamp_slot = out
                .has_frag_depth_clamp
                .then_some(out.frag_depth_clamp_slot);
            let immediate_slot = out.uses_immediates.then_some(out.immediate_slot);
            Ok(MslOutput {
                source: msl_guard.into_string(),
                entry_point: entry_point_guard.into_string(),
                needs_storage_buffer_sizes: out.needs_storage_buffer_sizes,
                buffer_size_bindings,
                workgroup_allocations,
                frag_depth_clamp_slot,
                immediate_slot,
            })
        }

        /// Generates SPIR-V words for `entry_point`.
        ///
        /// `user_immediate_size` is the owning pipeline layout's reserved
        /// user-immediate byte budget (Block 94, 0..=64) -- same meaning as
        /// in [`Self::generate_msl`]. It only affects output when
        /// `polyfill_pixel_center` is `Some`: the internal depth-range
        /// immediates are placed at byte offsets `{user_immediate_size,
        /// user_immediate_size + 4}` of the push-constant block, directly
        /// after the user region (mirrors Dawn
        /// `ShaderModuleVk.cpp:349-355`).
        #[allow(clippy::too_many_arguments)]
        pub fn generate_spirv(
            &self,
            entry_point: &str,
            bindings: &Bindings,
            overrides: &[OverrideValue],
            robust: bool,
            use_vulkan_memory_model: bool,
            framebuffer_fetch_descriptor_set: u32,
            multisampled_input_attachment: bool,
            polyfill_pixel_center: Option<u32>,
            user_immediate_size: u32,
        ) -> Result<Vec<u32>, TintError> {
            let ep = cstring(entry_point, "entry point").map_err(TintError::Codegen)?;
            let raw_bindings = bindings.as_raw();
            let raw_overrides = RawOverrideValues::new(overrides)?;
            let mut words = ptr::null_mut();
            let mut len = 0usize;
            let mut err = ptr::null_mut();
            // SAFETY: all pointers are valid for the duration of the call.
            let ok = unsafe {
                yawgpu_tint_generate_spirv(
                    self.raw,
                    ep.as_ptr(),
                    &raw_bindings,
                    raw_overrides.as_ptr(),
                    raw_overrides.len(),
                    !robust,
                    use_vulkan_memory_model,
                    framebuffer_fetch_descriptor_set,
                    multisampled_input_attachment,
                    polyfill_pixel_center.is_some(),
                    polyfill_pixel_center.unwrap_or(0),
                    user_immediate_size,
                    &mut words,
                    &mut len,
                    &mut err,
                )
            };
            if !ok {
                return Err(TintError::Codegen(take_error(err)));
            }
            // A successful generate of valid SPIR-V always yields a non-empty
            // module (>=5 header words); `U32Guard::as_slice` returns `&[]`
            // for a NULL pointer regardless, so the null case is handled
            // uniformly rather than as a special early return.
            let words_guard = U32Guard::new(words);
            Ok(words_guard.as_slice(len).to_vec())
        }

        /// Returns the module's total `var<workgroup>` storage size in bytes.
        pub fn workgroup_storage_size(
            &self,
            overrides: &[OverrideValue],
        ) -> Result<u64, TintError> {
            let raw_overrides = RawOverrideValues::new(overrides)?;
            let mut out = 0u64;
            let mut err = ptr::null_mut();
            // SAFETY: `out` and `err` point to valid writable memory.
            let ok = unsafe {
                yawgpu_tint_workgroup_storage_size(
                    self.raw,
                    raw_overrides.as_ptr(),
                    raw_overrides.len(),
                    &mut out,
                    &mut err,
                )
            };
            if !ok {
                return Err(TintError::Codegen(take_error(err)));
            }
            Ok(out)
        }

        /// Returns `entry_point`'s override-resolved `@workgroup_size` as
        /// `[x, y, z]`, without generating a full SPIR-V/MSL module (see F6 in
        /// `specs/tracking/tint-integration-refactor.md`). Fails if
        /// `entry_point` does not name an entry point in the module, an
        /// override in `overrides` is invalid/unknown, or the entry point's
        /// workgroup size does not resolve to constants (e.g. `entry_point`
        /// is not a compute entry point).
        pub fn resolved_workgroup_size(
            &self,
            entry_point: &str,
            overrides: &[OverrideValue],
        ) -> Result<[u32; 3], TintError> {
            let ep = cstring(entry_point, "entry point").map_err(TintError::Codegen)?;
            let raw_overrides = RawOverrideValues::new(overrides)?;
            let mut out = [0u32; 3];
            let mut err = ptr::null_mut();
            // SAFETY: `ep`, `out`, and `err` point to valid memory for the call.
            let ok = unsafe {
                yawgpu_tint_resolved_workgroup_size(
                    self.raw,
                    ep.as_ptr(),
                    raw_overrides.as_ptr(),
                    raw_overrides.len(),
                    out.as_mut_ptr(),
                    &mut err,
                )
            };
            if !ok {
                return Err(TintError::Codegen(take_error(err)));
            }
            Ok(out)
        }

        /// Returns `entry_point`'s immediate data size in bytes -- the total
        /// byte size of all `var<immediate>` (WGSL `immediate_address_space`
        /// language feature) globals it statically accesses, per Tint's
        /// `inspector::EntryPoint::immediate_data_size`. `0` means the entry
        /// point declares/uses no immediates (including modules that merely
        /// declare an unused `var<immediate>`). Fails if `entry_point` does
        /// not name an entry point in the module -- unlike
        /// [`Self::workgroup_storage_size`], that failure is never silently
        /// coerced to `Ok(0)`.
        pub fn immediate_data_size(&self, entry_point: &str) -> Result<u32, TintError> {
            let ep = cstring(entry_point, "entry point").map_err(TintError::Reflection)?;
            let mut out = 0u32;
            let mut err = ptr::null_mut();
            // SAFETY: `ep`, `out`, and `err` point to valid memory for the call.
            let ok = unsafe {
                yawgpu_tint_immediate_data_size(self.raw, ep.as_ptr(), &mut out, &mut err)
            };
            if !ok {
                return Err(TintError::Reflection(take_error(err)));
            }
            Ok(out)
        }

        /// Returns `entry_point`'s required immediate-data slots as a bitmask.
        ///
        /// Bit N corresponds to bytes `[4*N, 4*N+4)` of the user immediate
        /// block. The mask is Tint's `Inspector::GetImmediateBlockInfo`
        /// result: it includes non-padding words of any statically referenced
        /// `var<immediate>` variable and excludes struct/matrix padding words.
        /// Fails if `entry_point` does not name an entry point in the module.
        pub fn immediate_data_used_slots(&self, entry_point: &str) -> Result<u64, TintError> {
            let ep = cstring(entry_point, "entry point").map_err(TintError::Reflection)?;
            let mut out = 0u64;
            let mut err = ptr::null_mut();
            // SAFETY: `ep`, `out`, and `err` point to valid memory for the call.
            let ok = unsafe {
                yawgpu_tint_immediate_data_used_slots(self.raw, ep.as_ptr(), &mut out, &mut err)
            };
            if !ok {
                return Err(TintError::Reflection(take_error(err)));
            }
            Ok(out)
        }

        /// Generates GLSL ES 3.1 for `entry_point`.
        ///
        /// `first_instance_offset`, when `Some`, is passed through to Tint's
        /// `glsl::writer::Options::first_instance_offset` (a byte offset into
        /// Tint's internal "immediate data" map -- yawgpu-core always passes
        /// `Some(0)` for vertex stages, since GLES does not share that map
        /// with any other internal immediate today). This makes Tint rewrite
        /// `@builtin(instance_index)` to add a value it reads back from a
        /// plain array-typed uniform, `layout(location = 0) uniform uint
        /// tint_immediates[1]` -- NOT a named struct with a
        /// `tint_first_instance` member; query it as
        /// `glGetUniformLocation(program, "tint_immediates[0]")` (see
        /// `tint_shim.h` docs on `yawgpu_tint_generate_glsl`, and the
        /// `generate_glsl_first_instance_offset_only_applied_when_requested`
        /// test below, which pins the exact GLSL text). `None` leaves
        /// `instance_index` as raw `gl_InstanceID` with no immediate-data
        /// array emitted.
        pub fn generate_glsl(
            &self,
            entry_point: &str,
            bindings: &Bindings,
            overrides: &[OverrideValue],
            first_instance_offset: Option<u32>,
        ) -> Result<GlslOutput, TintError> {
            let ep = cstring(entry_point, "entry point").map_err(TintError::Codegen)?;
            let raw_bindings = bindings.as_raw();
            let raw_overrides = RawOverrideValues::new(overrides)?;
            let mut out = RawGlslOutput {
                glsl: ptr::null_mut(),
                combined_samplers: ptr::null_mut(),
                n_combined_samplers: 0,
                texture_metadata_slots: ptr::null_mut(),
                n_texture_metadata_slots: 0,
                has_texture_metadata_ubo: false,
                texture_metadata_ubo_binding: 0,
            };
            let mut err = ptr::null_mut();
            // SAFETY: all pointers are valid for the duration of the call.
            let ok = unsafe {
                yawgpu_tint_generate_glsl(
                    self.raw,
                    ep.as_ptr(),
                    &raw_bindings,
                    raw_overrides.as_ptr(),
                    raw_overrides.len(),
                    first_instance_offset.is_some(),
                    first_instance_offset.unwrap_or(0),
                    &mut out,
                    &mut err,
                )
            };
            if !ok {
                return Err(TintError::Codegen(take_error(err)));
            }
            let glsl_guard = GlslOutputGuard::new(out);
            if glsl_guard.is_glsl_null() {
                return Err(TintError::Codegen(
                    "tint: GLSL generator returned NULL output".to_owned(),
                ));
            }
            Ok(glsl_guard.into_output())
        }
    }

    impl Drop for Program {
        fn drop(&mut self) {
            // SAFETY: `raw` was returned by the shim and is owned by this Program.
            unsafe { yawgpu_tint_program_destroy(self.raw) }
        }
    }

    enum StageVariableDirection {
        Input,
        Output,
    }

    struct RawVertexBuffers<'a> {
        buffers: Vec<RawVertexBuffer>,
        // Ties this struct's lifetime to the borrowed `VertexBuffer` slice:
        // unlike the deleted `RawBindingsOwned` (which owned full copies and
        // so never actually needed its `PhantomData`), `buffers` here holds
        // raw pointers straight into each `VertexBuffer::attributes` `Vec`
        // (no per-call copy), so the borrow genuinely must outlive `Self`.
        _borrow: std::marker::PhantomData<&'a [VertexBuffer]>,
    }

    impl<'a> RawVertexBuffers<'a> {
        fn new(vertex_buffers: &'a [VertexBuffer]) -> Self {
            let buffers = vertex_buffers
                .iter()
                .map(|buffer| RawVertexBuffer {
                    slot: buffer.slot,
                    metal_index: buffer.metal_index,
                    array_stride: buffer.array_stride,
                    step_mode: buffer.step_mode as u8,
                    attributes: buffer.attributes.as_ptr(),
                    n_attributes: buffer.attributes.len(),
                })
                .collect();
            Self {
                buffers,
                _borrow: std::marker::PhantomData,
            }
        }

        fn as_ptr(&self) -> *const RawVertexBuffer {
            self.buffers.as_ptr()
        }

        fn len(&self) -> usize {
            self.buffers.len()
        }
    }

    struct RawOverrideValues {
        // Never read directly -- held only so the `CString` buffers whose
        // `.as_ptr()`s populate `values` stay alive for `Self`'s lifetime
        // (leading `_` documents this and silences the false-positive
        // `dead_code` lint, replacing the old `let _keep_names_alive = ...`
        // statement in `as_ptr` with a single, clearer mechanism).
        _names: Vec<CString>,
        values: Vec<RawOverrideValue>,
    }

    impl RawOverrideValues {
        fn new(overrides: &[OverrideValue]) -> Result<Self, TintError> {
            let names: Vec<CString> = overrides
                .iter()
                .map(|ov| cstring(&ov.name, "override name"))
                .collect::<Result<_, _>>()
                .map_err(TintError::Codegen)?;
            let values = names
                .iter()
                .zip(overrides)
                .map(|(name, ov)| RawOverrideValue {
                    name: name.as_ptr(),
                    value: ov.value,
                })
                .collect();
            Ok(Self {
                _names: names,
                values,
            })
        }

        fn as_ptr(&self) -> *const RawOverrideValue {
            self.values.as_ptr()
        }

        fn len(&self) -> usize {
            self.values.len()
        }
    }
}

#[cfg(not(have_tint))]
mod stub {
    use super::*;

    /// Initializes the Tint runtime.
    pub fn initialize() {}

    impl Program {
        /// Parses WGSL into a Tint program.
        pub fn parse(
            _wgsl: &str,
            _shader_f16: bool,
            _subgroups: bool,
            _dual_source_blending: bool,
            _clip_distances: bool,
            _primitive_index: bool,
            _language_features: &[u32],
        ) -> Result<Self, TintError> {
            Err(TintError::Unavailable)
        }

        /// Returns reflected entry points.
        pub fn entry_points(&self) -> Result<Vec<EntryPoint>, TintError> {
            Err(TintError::Unavailable)
        }

        /// Returns stage input variables used by `entry_point`.
        pub fn entry_point_inputs(
            &self,
            _entry_point: &str,
        ) -> Result<Vec<StageVariable>, TintError> {
            Err(TintError::Unavailable)
        }

        /// Returns stage output variables used by `entry_point`.
        pub fn entry_point_outputs(
            &self,
            _entry_point: &str,
        ) -> Result<Vec<StageVariable>, TintError> {
            Err(TintError::Unavailable)
        }

        /// Returns non-error diagnostics reported for this valid program.
        pub fn diagnostics(&self) -> Result<Vec<Diagnostic>, TintError> {
            Err(TintError::Unavailable)
        }

        /// Returns resource bindings used by `entry_point`.
        pub fn resource_bindings(
            &self,
            _entry_point: &str,
        ) -> Result<Vec<ResourceBinding>, TintError> {
            Err(TintError::Unavailable)
        }

        /// Returns module override declarations.
        pub fn overrides(&self) -> Result<Vec<Override>, TintError> {
            Err(TintError::Unavailable)
        }

        /// Generates MSL for `entry_point`.
        #[allow(clippy::too_many_arguments)]
        pub fn generate_msl(
            &self,
            _entry_point: &str,
            _bindings: &Bindings,
            _overrides: &[OverrideValue],
            _buffer_sizes_slot: u32,
            _robust: bool,
            _emit_vertex_point_size: bool,
            _vertex_buffers: &[VertexBuffer],
            _fixed_sample_mask: u32,
            _user_immediate_size: u32,
        ) -> Result<MslOutput, TintError> {
            Err(TintError::Unavailable)
        }

        /// Generates SPIR-V words for `entry_point`.
        #[allow(clippy::too_many_arguments)]
        pub fn generate_spirv(
            &self,
            _entry_point: &str,
            _bindings: &Bindings,
            _overrides: &[OverrideValue],
            _robust: bool,
            _use_vulkan_memory_model: bool,
            _framebuffer_fetch_descriptor_set: u32,
            _multisampled_input_attachment: bool,
            _polyfill_pixel_center: Option<u32>,
            _user_immediate_size: u32,
        ) -> Result<Vec<u32>, TintError> {
            Err(TintError::Unavailable)
        }

        /// Returns the module's total `var<workgroup>` storage size in bytes.
        pub fn workgroup_storage_size(
            &self,
            _overrides: &[OverrideValue],
        ) -> Result<u64, TintError> {
            Ok(0)
        }

        /// Returns `entry_point`'s override-resolved `@workgroup_size` as `[x, y, z]`.
        pub fn resolved_workgroup_size(
            &self,
            _entry_point: &str,
            _overrides: &[OverrideValue],
        ) -> Result<[u32; 3], TintError> {
            Err(TintError::Unavailable)
        }

        /// Returns `entry_point`'s immediate data size in bytes.
        pub fn immediate_data_size(&self, _entry_point: &str) -> Result<u32, TintError> {
            Err(TintError::Unavailable)
        }

        /// Returns `entry_point`'s required immediate-data slots as a bitmask.
        pub fn immediate_data_used_slots(&self, _entry_point: &str) -> Result<u64, TintError> {
            Err(TintError::Unavailable)
        }

        /// Generates GLSL ES 3.1 for `entry_point`.
        pub fn generate_glsl(
            &self,
            _entry_point: &str,
            _bindings: &Bindings,
            _overrides: &[OverrideValue],
            _first_instance_offset: Option<u32>,
        ) -> Result<GlslOutput, TintError> {
            Err(TintError::Unavailable)
        }
    }
}

#[cfg(have_tint)]
pub use real::initialize;
#[cfg(not(have_tint))]
pub use stub::initialize;

#[cfg(all(test, have_tint))]
mod tests {
    use super::*;

    fn compute_wgsl() -> &'static str {
        "@compute @workgroup_size(8, 1, 1) fn cs() {}"
    }

    fn render_wgsl() -> &'static str {
        r#"
struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  var p = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0));
  var out: VsOut;
  out.pos = vec4f(p[vi], 0.0, 1.0);
  out.uv = p[vi];
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  return vec4f(in.uv, 0.0, 1.0);
}
"#
    }

    fn framebuffer_fetch_wgsl() -> &'static str {
        r#"
enable chromium_experimental_framebuffer_fetch;

@fragment
fn fs(@color(0) prev: vec4<f32>) -> @location(0) vec4<f32> {
  return prev;
}
"#
    }

    fn input_attachment_wgsl() -> &'static str {
        r#"
enable chromium_internal_input_attachments;

@group(0) @binding(0) @input_attachment_index(0) var ia: input_attachment<f32>;

@fragment
fn fs() -> @location(0) vec4<f32> {
  return inputAttachmentLoad(ia);
}
"#
    }

    fn spirv_has_binding_decoration(words: &[u32], binding: u32) -> bool {
        words.windows(4).any(|w| {
            let opcode = w[0] & 0xffff;
            let word_count = w[0] >> 16;
            opcode == 71 && word_count == 4 && w[2] == 33 && w[3] == binding
        })
    }

    #[test]
    fn reflects_workgroup_storage_size() {
        let program = Program::parse(
            r#"
var<workgroup> data: array<u32, 8>;

@compute @workgroup_size(1)
fn cs() {
  data[0] = 1u;
}
"#,
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(program.workgroup_storage_size(&[]).unwrap(), 32);

        let program = Program::parse(
            "@compute @workgroup_size(1) fn cs() {}",
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(program.workgroup_storage_size(&[]).unwrap(), 0);

        let program = Program::parse(
            "@fragment fn fs() -> @location(0) vec4f { return vec4f(); }",
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(program.workgroup_storage_size(&[]).unwrap(), 0);

        let program = Program::parse(
            r#"
override n: u32 = 4;
var<workgroup> d: array<u32, n>;

@compute @workgroup_size(1)
fn main() {
  d[0] = 1u;
}
"#,
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(program.workgroup_storage_size(&[]).unwrap(), 16);
        assert_eq!(
            program
                .workgroup_storage_size(&[OverrideValue {
                    name: "n".into(),
                    value: 8.0,
                }])
                .unwrap(),
            32
        );
    }

    #[test]
    fn reflects_immediate_data_size() {
        let wgsl = r#"
requires immediate_address_space;

var<immediate> unused_imm : u32;
var<immediate> used_imm : vec4f;

@compute @workgroup_size(1)
fn uses_immediate() {
  let v = used_imm;
  _ = v;
}

@compute @workgroup_size(1)
fn no_immediate() {}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[11]).unwrap();
        // Entry point that statically accesses `used_imm` (a vec4f = 16 bytes);
        // `unused_imm` is declared but never touched by this entry point, so
        // it does not contribute to the size (matches Dawn's
        // ImmediateDataSizeTwoConstants inspector test).
        assert_eq!(program.immediate_data_size("uses_immediate").unwrap(), 16);
        assert_eq!(
            program.immediate_data_used_slots("uses_immediate").unwrap(),
            0b1111
        );
        // Entry point that touches no immediate-address-space variable at all.
        assert_eq!(program.immediate_data_size("no_immediate").unwrap(), 0);
        assert_eq!(
            program.immediate_data_used_slots("no_immediate").unwrap(),
            0
        );
        // A nonexistent entry point is a real error, never silently `Ok(0)`
        // (the flaw this query deliberately avoids -- see
        // `workgroup_storage_size`'s doc comment).
        assert!(program.immediate_data_size("does_not_exist").is_err());
        assert!(program.immediate_data_used_slots("does_not_exist").is_err());
    }

    #[test]
    fn reflects_immediate_data_used_slots_excluding_padding() {
        let wgsl = r#"
requires immediate_address_space;

struct S {
  a : u32,
  b : vec3<f32>,
}

var<immediate> params : S;

@compute @workgroup_size(1)
fn main() {
  _ = params;
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[11]).unwrap();

        assert_eq!(program.immediate_data_size("main").unwrap(), 32);
        assert_eq!(
            program.immediate_data_used_slots("main").unwrap(),
            0b0111_0001
        );
    }

    #[test]
    fn compute_generates_msl_spirv_glsl() {
        let program =
            Program::parse(compute_wgsl(), false, false, false, false, false, &[]).unwrap();
        let bindings = Bindings::default();
        let msl = program
            .generate_msl("cs", &bindings, &[], 0, true, false, &[], 0xFFFF_FFFF, 0)
            .unwrap();
        assert!(msl.source.contains("kernel"), "MSL:\n{}", msl.source);
        let spirv = program
            .generate_spirv("cs", &bindings, &[], true, false, 0, false, None, 0)
            .unwrap();
        assert_eq!(spirv.first().copied(), Some(0x0723_0203));
        let spirv_with_vulkan_memory_model = program
            .generate_spirv("cs", &bindings, &[], true, true, 0, false, None, 0)
            .unwrap();
        assert_eq!(
            spirv_with_vulkan_memory_model.first().copied(),
            Some(0x0723_0203)
        );
        let glsl = program
            .generate_glsl("cs", &bindings, &[], None)
            .unwrap()
            .source;
        assert!(glsl.contains("#version 310 es"), "GLSL:\n{glsl}");
    }

    #[test]
    fn generate_glsl_cube_array_texture_uses_es_320() {
        let wgsl = r#"
@group(0) @binding(0) var t: texture_cube_array<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn fs() -> @location(0) vec4f {
  return textureSample(t, s, vec3f(1.0, 0.0, 0.0), 0);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let glsl = program
            .generate_glsl("fs", &Bindings::default(), &[], None)
            .unwrap()
            .source;
        assert!(glsl.contains("#version 320 es"), "GLSL:\n{glsl}");
        assert!(glsl.contains("samplerCubeArray"), "GLSL:\n{glsl}");
    }

    fn instance_index_vertex_wgsl() -> &'static str {
        r#"
@vertex
fn vs(@builtin(instance_index) ii: u32, @builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  return vec4f(f32(ii), f32(vi), 0.0, 1.0);
}
"#
    }

    /// F2 drift guard (specs/tracking/tint-integration-refactor.md, slice
    /// R6): pins the GLES contract the yawgpu-hal GLES pipeline relies on.
    /// Without `first_instance_offset`, `instance_index` is raw
    /// `gl_InstanceID` with no immediate-data struct emitted at all --
    /// confirms that a HAL which never requests the offset (the naga-era
    /// `naga_vs_first_instance` uniform lookup this replaces) makes
    /// `firstInstance` silently no-op. With it, Tint emits a single
    /// `layout(location = 0) uniform uint tint_immediates[1]` array (NOT a
    /// named struct member -- Tint's internal immediate symbol name
    /// (`tint_first_instance`) never reaches the printed GLSL text; the
    /// raise pass indexes the immediate block by word offset only) and
    /// reads element `[0]` to offset `gl_InstanceID`. The HAL must look
    /// this uniform up by `glGetUniformLocation(program,
    /// "tint_immediates[0]")` and write with `glUniform1ui`, not query a
    /// UBO member by dotted name.
    #[test]
    fn generate_glsl_first_instance_offset_only_applied_when_requested() {
        let program = Program::parse(
            instance_index_vertex_wgsl(),
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let bindings = Bindings::default();

        let glsl_without = program
            .generate_glsl("vs", &bindings, &[], None)
            .unwrap()
            .source;
        assert!(
            glsl_without.contains("gl_InstanceID"),
            "GLSL:\n{glsl_without}"
        );
        assert!(
            !glsl_without.contains("tint_immediates"),
            "GLSL:\n{glsl_without}"
        );

        let glsl_with = program
            .generate_glsl("vs", &bindings, &[], Some(0))
            .unwrap()
            .source;
        // The immediate-data block is a plain array-typed uniform with an
        // explicit location, not a UBO -- HAL must look it up by
        // `glGetUniformLocation`, not `glGetUniformBlockIndex`.
        assert!(
            glsl_with.contains("layout(location = 0) uniform uint tint_immediates[1];"),
            "GLSL:\n{glsl_with}"
        );
        assert!(
            glsl_with.contains("tint_immediates[0"),
            "expected an indexed read of element 0: {glsl_with}"
        );
        assert!(
            glsl_with.contains("gl_InstanceID"),
            "instance_index is still sourced from gl_InstanceID, just offset: {glsl_with}"
        );
    }

    #[test]
    fn generate_glsl_sample_mask_output_emits_sample_variables_extension() {
        let wgsl = r#"
@fragment
fn fs() -> @builtin(sample_mask) u32 {
  return 1u;
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let glsl = program
            .generate_glsl("fs", &Bindings::default(), &[], None)
            .unwrap()
            .source;
        assert!(
            glsl.contains("#extension GL_OES_sample_variables"),
            "GLSL:\n{glsl}"
        );
        assert!(glsl.contains("gl_SampleMask"), "GLSL:\n{glsl}");
    }

    #[test]
    fn generate_glsl_sample_mask_input_emits_sample_variables_extension() {
        let wgsl = r#"
@fragment
fn fs(@builtin(sample_mask) mask_in: u32) -> @location(0) vec4f {
  if ((mask_in & 1u) != 0u) {
    return vec4f(1.0, 0.0, 0.0, 1.0);
  }
  return vec4f(0.0, 0.0, 0.0, 1.0);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let glsl = program
            .generate_glsl("fs", &Bindings::default(), &[], None)
            .unwrap()
            .source;
        assert!(
            glsl.contains("#extension GL_OES_sample_variables"),
            "GLSL:\n{glsl}"
        );
        assert!(glsl.contains("gl_SampleMaskIn"), "GLSL:\n{glsl}");
    }

    #[test]
    fn generate_glsl_multisampled_texture_load_uses_texel_fetch_sample_index() {
        let wgsl = r#"
@group(0) @binding(0) var t: texture_multisampled_2d<f32>;

@fragment
fn fs(@builtin(sample_index) sample_index: u32) -> @location(0) vec4f {
  return textureLoad(t, vec2i(0, 0), sample_index);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let glsl = program
            .generate_glsl("fs", &Bindings::default(), &[], None)
            .unwrap()
            .source;
        assert!(glsl.contains("sampler2DMS"), "GLSL:\n{glsl}");
        assert!(glsl.contains("texelFetch"), "GLSL:\n{glsl}");
        assert!(glsl.contains("gl_SampleID"), "GLSL:\n{glsl}");
    }

    // P2 depth raw-read: a `texture_depth_2d` sampled by a NON-comparison
    // builtin must lower to a plain `sampler2D` (raw depth read), NOT
    // `sampler2DShadow` (a ref-0 shadow compare that returns 0/1). The
    // shim-level Core-IR transform retypes the depth var to a sampled
    // `texture_2d<f32>` before the GLSL writer's raise runs.
    #[test]
    fn generate_glsl_depth_sample_level_reads_raw_not_shadow() {
        let wgsl = r#"
@group(0) @binding(0) var t: texture_depth_2d;
@group(0) @binding(1) var s: sampler;

@fragment
fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
  let d = textureSampleLevel(t, s, uv, 0u);
  return vec4f(d, d, d, 1.0);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let glsl = program
            .generate_glsl("fs", &Bindings::default(), &[], None)
            .unwrap()
            .source;
        assert!(glsl.contains("sampler2D "), "expected sampler2D: {glsl}");
        assert!(
            !glsl.contains("sampler2DShadow"),
            "must not shadow-compare: {glsl}"
        );
        // Raw sample -> `textureLod(...)`, scalar depth extracted via `.x`.
        assert!(
            glsl.contains("textureLod("),
            "expected textureLod(): {glsl}"
        );
        assert!(glsl.contains(").x"), "expected an .x swizzle: {glsl}");
    }

    // Depth `textureGather` (no component arg, returns vec4f) must lower to a
    // plain `textureGather` on a `sampler2D` with no `refz` comparison arg.
    #[test]
    fn generate_glsl_depth_gather_reads_raw_not_shadow() {
        let wgsl = r#"
@group(0) @binding(0) var t: texture_depth_2d;
@group(0) @binding(1) var s: sampler;

@fragment
fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureGather(t, s, uv);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let glsl = program
            .generate_glsl("fs", &Bindings::default(), &[], None)
            .unwrap()
            .source;
        assert!(glsl.contains("sampler2D "), "expected sampler2D: {glsl}");
        assert!(
            !glsl.contains("sampler2DShadow"),
            "must not shadow-compare: {glsl}"
        );
        assert!(
            glsl.contains("textureGather("),
            "expected textureGather(): {glsl}"
        );
    }

    // A depth texture used by a COMPARISON builtin (`textureSampleCompare`) is
    // out of scope for the raw-read rewrite and must keep the shadow lowering:
    // `sampler2DShadow`. Eligibility skips any var with a comparison use.
    #[test]
    fn generate_glsl_depth_sample_compare_stays_shadow() {
        let wgsl = r#"
@group(0) @binding(0) var t: texture_depth_2d;
@group(0) @binding(1) var s: sampler_comparison;

@fragment
fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
  let d = textureSampleCompare(t, s, uv, 0.5);
  return vec4f(d, d, d, 1.0);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let glsl = program
            .generate_glsl("fs", &Bindings::default(), &[], None)
            .unwrap()
            .source;
        assert!(
            glsl.contains("sampler2DShadow"),
            "comparison must stay shadow: {glsl}"
        );
    }

    // A single depth texture used by BOTH a comparison and a non-comparison
    // builtin is out of scope (mixed use). Eligibility skips it, so it keeps
    // the shadow lowering unchanged (documents the residual).
    #[test]
    fn generate_glsl_depth_mixed_compare_and_sample_stays_shadow() {
        let wgsl = r#"
@group(0) @binding(0) var t: texture_depth_2d;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var sc: sampler_comparison;

@fragment
fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
  let raw = textureSampleLevel(t, s, uv, 0u);
  let cmp = textureSampleCompareLevel(t, sc, uv, 0.5);
  return vec4f(raw, cmp, 0.0, 1.0);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let glsl = program
            .generate_glsl("fs", &Bindings::default(), &[], None)
            .unwrap()
            .source;
        assert!(
            glsl.contains("sampler2DShadow"),
            "mixed use must stay shadow (out of scope): {glsl}"
        );
    }

    #[test]
    fn generate_glsl_cts_readback_compute_variants_with_explicit_remaps() {
        for (texture_type, value_type, sample_count) in [
            ("texture_2d<f32>", "f32", 1u32),
            ("texture_depth_2d", "f32", 1),
            ("texture_2d<u32>", "u32", 1),
            ("texture_multisampled_2d<f32>", "f32", 4),
            ("texture_depth_multisampled_2d", "f32", 4),
            ("texture_multisampled_2d<u32>", "u32", 4),
        ] {
            let mut wgsl = format!(
                "@group(0) @binding(0) var src: {texture_type};\n\
                 @group(0) @binding(1) var<storage, read_write> dst: array<{value_type}>;\n\
                 @compute @workgroup_size(1) fn main() {{\n"
            );
            for sample in 0..sample_count {
                if texture_type.contains("multisampled") {
                    wgsl.push_str(&format!(
                        "  let v{sample} = textureLoad(src, vec2<i32>(0, 0), {sample});\n"
                    ));
                } else {
                    wgsl.push_str(&format!(
                        "  let v{sample} = textureLoad(src, vec2<i32>(0, 0), 0);\n"
                    ));
                }
                if texture_type == "texture_depth_2d"
                    || texture_type == "texture_depth_multisampled_2d"
                {
                    wgsl.push_str(&format!("  dst[{sample}] = v{sample};\n"));
                } else {
                    wgsl.push_str(&format!("  dst[{sample}] = v{sample}.r;\n"));
                }
            }
            wgsl.push_str("}\n");
            let program = Program::parse(&wgsl, false, false, false, false, false, &[]).unwrap();
            let mut bindings = Bindings::default();
            bindings.texture.push(BindingRemap {
                group: 0,
                binding: 0,
                dst_group: 0,
                dst_binding: 0,
            });
            bindings.storage.push(BindingRemap {
                group: 0,
                binding: 1,
                dst_group: 0,
                dst_binding: 1,
            });
            let output = program.generate_glsl("main", &bindings, &[], None).unwrap();
            assert!(
                output.source.contains("TintTextureUniformData"),
                "GLSL for {texture_type}:\n{}",
                output.source
            );
            assert_eq!(output.texture_metadata_ubo_binding, Some(0));
            assert_eq!(output.combined_samplers.len(), 1);
            assert!(output.combined_samplers[0].uses_placeholder_sampler);
            assert!(
                output.source.contains(if texture_type.contains("<u32>") {
                    if texture_type.contains("multisampled") {
                        "usampler2DMS"
                    } else {
                        "usampler2D"
                    }
                } else if texture_type.contains("multisampled") {
                    "sampler2DMS"
                } else {
                    "sampler2D"
                }),
                "GLSL for {texture_type}:\n{}",
                output.source
            );
        }
    }

    #[test]
    fn generate_glsl_texture_num_levels_without_sampler_reports_metadata_slot() {
        let wgsl = r#"
@group(2) @binding(7) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(1)
fn main() {
  dst[0] = textureNumLevels(src);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let mut bindings = Bindings::default();
        bindings.texture.push(BindingRemap {
            group: 2,
            binding: 7,
            dst_group: 0,
            dst_binding: 0,
        });
        bindings.storage.push(BindingRemap {
            group: 0,
            binding: 1,
            dst_group: 0,
            dst_binding: 1,
        });
        let output = program.generate_glsl("main", &bindings, &[], None).unwrap();
        assert!(
            output.source.contains("TintTextureUniformData"),
            "GLSL:\n{}",
            output.source
        );
        assert_eq!(output.texture_metadata_ubo_binding, Some(0));
        assert_eq!(
            output.texture_metadata_slots,
            vec![TextureMetadataSlot {
                offset: 0,
                group: 0,
                binding: 0,
            }]
        );
    }

    #[test]
    fn generate_glsl_metadata_slot_offset_follows_resolved_binding_across_stages() {
        // A vertex stage and a fragment stage each query textureNumLevels on a
        // DIFFERENT texture, plus a third texture shared by both. With the
        // per-resolved-binding offset scheme, each stage independently computes
        // offset == its resolved binding, so different textures get disjoint
        // offsets (no cross-stage collision) and the shared texture gets the
        // same offset in both stages. This mirrors Dawn's per-pipeline
        // EmulatedTextureBuiltinRegistrar and is what lets core's
        // merge_texture_metadata_slots agree instead of raising a conflict.
        let wgsl = r#"
@group(0) @binding(0) var vtex: texture_2d<f32>;
@group(0) @binding(1) var ftex: texture_2d<f32>;
@group(0) @binding(2) var shared_tex: texture_2d<f32>;

@vertex
fn vs() -> @builtin(position) vec4f {
  let a = textureNumLevels(vtex);
  let b = textureNumLevels(shared_tex);
  return vec4f(f32(a + b), 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4f {
  let a = textureNumLevels(ftex);
  let b = textureNumLevels(shared_tex);
  return vec4f(f32(a + b), 0.0, 0.0, 1.0);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let mut bindings = Bindings::default();
        // yawgpu's flat remap: each texture keeps its binding as the flat value
        // under group 0 (unique per texture across the pipeline).
        for b in 0..3u32 {
            bindings.texture.push(BindingRemap {
                group: 0,
                binding: b,
                dst_group: 0,
                dst_binding: b,
            });
        }

        let vs = program.generate_glsl("vs", &bindings, &[], None).unwrap();
        let fs = program.generate_glsl("fs", &bindings, &[], None).unwrap();

        // Every slot's offset equals its resolved binding, in both stages.
        for slot in vs.texture_metadata_slots.iter().chain(&fs.texture_metadata_slots) {
            assert_eq!(
                slot.offset, slot.binding,
                "slot offset must equal resolved binding: {slot:?}"
            );
            assert_eq!(slot.group, 0, "resolved group must be 0: {slot:?}");
        }

        let offset_for = |slots: &[TextureMetadataSlot], binding: u32| -> u32 {
            slots
                .iter()
                .find(|s| s.binding == binding)
                .unwrap_or_else(|| panic!("no metadata slot for binding {binding}"))
                .offset
        };

        // Different textures -> disjoint offsets across the two stages.
        let vtex_off = offset_for(&vs.texture_metadata_slots, 0);
        let ftex_off = offset_for(&fs.texture_metadata_slots, 1);
        assert_ne!(
            vtex_off, ftex_off,
            "vertex-only and fragment-only textures must not share a metadata offset"
        );

        // Shared texture -> identical offset in both stages.
        let shared_vs = offset_for(&vs.texture_metadata_slots, 2);
        let shared_fs = offset_for(&fs.texture_metadata_slots, 2);
        assert_eq!(
            shared_vs, shared_fs,
            "a texture shared by both stages must map to the same metadata offset"
        );
    }

    #[test]
    fn generate_glsl_returns_combined_sampler_mapping_for_texture_sample() {
        let wgsl = r#"
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn fs(@location(0) uv: vec2f) -> @location(0) vec4f {
  return textureSample(t, s, uv);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let output = program
            .generate_glsl("fs", &Bindings::default(), &[], None)
            .unwrap();

        assert_eq!(
            output.combined_samplers.len(),
            1,
            "GLSL:\n{}",
            output.source
        );
        let combined = &output.combined_samplers[0];
        assert_eq!(combined.texture_group, 0);
        assert_eq!(combined.texture_binding, 0);
        assert_eq!(combined.sampler_group, 0);
        assert_eq!(combined.sampler_binding, 1);
        assert!(!combined.uses_placeholder_sampler);
        assert!(
            output.source.contains(&format!(
                "uniform highp sampler2D {};",
                combined.glsl_uniform_name
            )),
            "GLSL:\n{}",
            output.source
        );
    }

    #[test]
    fn generate_glsl_returns_placeholder_combined_sampler_for_texture_load() {
        let wgsl = r#"
@group(0) @binding(0) var t: texture_2d<f32>;

@fragment
fn fs() -> @location(0) vec4f {
  return textureLoad(t, vec2i(0, 0), 0);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let output = program
            .generate_glsl("fs", &Bindings::default(), &[], None)
            .unwrap();

        assert_eq!(
            output.combined_samplers.len(),
            1,
            "GLSL:\n{}",
            output.source
        );
        let combined = &output.combined_samplers[0];
        assert_eq!(combined.texture_group, 0);
        assert_eq!(combined.texture_binding, 0);
        assert!(combined.uses_placeholder_sampler);
        assert!(
            output.source.contains(&format!(
                "uniform highp sampler2D {};",
                combined.glsl_uniform_name
            )),
            "GLSL:\n{}",
            output.source
        );
    }

    /// F2 drift guard: left as `Bindings::default()`, Tint's own
    /// `GenerateBindings` renumbers GLSL `layout(binding = N)` sequentially
    /// in declaration order -- it does NOT preserve non-sequential WGSL
    /// `@binding` numbers (here `@binding(3)` becomes `layout(binding = 1)`
    /// because it is the second buffer Tint sees). This is why
    /// `yawgpu-core`'s `tint_bindings_for_glsl` always supplies an explicit
    /// identity `BindingRemap` per buffer resource rather than relying on
    /// the default: `yawgpu-hal`'s GLES backend binds buffers with
    /// `glBindBufferRange` at the raw WGSL binding number
    /// (`HalDescriptorBinding::binding`), so leaving this on default would
    /// silently desync the GLSL text from the HAL's runtime bind calls for
    /// any non-sequential binding layout.
    #[test]
    fn generate_glsl_default_bindings_renumber_sequentially_not_identity() {
        let wgsl = r#"
struct Uniforms {
  scale: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(3) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn cs() {
  data[0] = u32(u.scale);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let bindings = Bindings::default();
        let glsl = program
            .generate_glsl("cs", &bindings, &[], None)
            .unwrap()
            .source;
        assert!(
            glsl.contains("layout(binding = 0, std140)\nuniform"),
            "GLSL:\n{glsl}"
        );
        // NOT `layout(binding = 3, ...)` -- Tint renumbers to the next free
        // slot (1), not the original WGSL binding.
        assert!(
            glsl.contains("layout(binding = 1, std430)\nbuffer"),
            "GLSL:\n{glsl}"
        );
        assert!(!glsl.contains("binding = 3"), "GLSL:\n{glsl}");
    }

    /// F2 drift guard: uniform/storage buffers get an explicit
    /// `layout(binding = N)` on the GLSL block (GLES 3.1 core, no
    /// extension needed) that the GL driver honors at link time with no
    /// `glUniformBlockBinding` / `glShaderStorageBlockBinding` remap
    /// required, PROVIDED Tint is given an explicit identity `BindingRemap`
    /// (the previous test shows the default auto-numbering does not
    /// preserve non-sequential WGSL binding numbers). This is why
    /// `block_binding_from_name`'s naga-era `_block_N`-suffix parsing was
    /// deleted outright rather than rewritten: Tint's block names carry an
    /// internal counter unrelated to the WGSL binding index (both blocks
    /// below are suffixed `_block_1` despite binding at 0 and 3), so
    /// running that old parser against Tint's real names would have
    /// silently mis-bound (or collided) buffers instead of leaving the
    /// already-correct explicit binding alone.
    #[test]
    fn generate_glsl_explicit_binding_remap_pins_exact_gl_binding() {
        let wgsl = r#"
struct Uniforms {
  scale: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(3) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn cs() {
  data[0] = u32(u.scale);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let bindings = Bindings {
            uniform: vec![BindingRemap {
                group: 0,
                binding: 0,
                dst_group: 0,
                dst_binding: 0,
            }],
            storage: vec![BindingRemap {
                group: 0,
                binding: 3,
                dst_group: 0,
                dst_binding: 3,
            }],
            ..Default::default()
        };
        let glsl = program
            .generate_glsl("cs", &bindings, &[], None)
            .unwrap()
            .source;
        assert!(
            glsl.contains("layout(binding = 0, std140)\nuniform u_block_1_ubo"),
            "GLSL:\n{glsl}"
        );
        assert!(
            glsl.contains("layout(binding = 3, std430)\nbuffer data_block_1_ssbo"),
            "GLSL:\n{glsl}"
        );
    }

    #[test]
    fn override_evaluation_error_returns_err_without_crashing() {
        let wgsl = r#"
override cu: u32 = 0u;
override cx: u32 = 1u / cu;

@compute @workgroup_size(1)
fn main() {
  _ = cx;
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let err = program
            .generate_spirv(
                "main",
                &Bindings::default(),
                &[],
                true,
                false,
                0,
                false,
                None,
                0,
            )
            .unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn compute_msl_array_length_returns_size_bindings() {
        let wgsl = r#"
struct Data {
  values: array<u32>,
}

@group(1) @binding(2) var<storage, read_write> data: Data;

@compute @workgroup_size(1)
fn cs() {
  if (arrayLength(&data.values) > 0u) {
    data.values[0] = arrayLength(&data.values);
  }
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let msl = program
            .generate_msl(
                "cs",
                &Bindings::default(),
                &[],
                9,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap();
        assert!(
            msl.source
                .contains("tint_storage_buffer_sizes [[buffer(9)]]"),
            "MSL:\n{}",
            msl.source
        );
        assert!(msl.needs_storage_buffer_sizes);
        assert_eq!(
            msl.buffer_size_bindings,
            vec![MslBufferSizeBinding {
                group: 1,
                binding: 2,
            }]
        );
    }

    #[test]
    fn compute_msl_returns_workgroup_allocations() {
        let wgsl = r#"
var<workgroup> wg: atomic<u32>;

@compute @workgroup_size(64)
fn cs() {
  atomicAdd(&wg, 1u);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let msl = program
            .generate_msl(
                "cs",
                &Bindings::default(),
                &[],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap();
        assert!(
            !msl.workgroup_allocations.is_empty(),
            "MSL:\n{}",
            msl.source
        );
        assert!(
            msl.workgroup_allocations[0] >= 4,
            "allocations: {:?}\nMSL:\n{}",
            msl.workgroup_allocations,
            msl.source
        );
    }

    #[test]
    fn render_stages_generate_msl_and_spirv() {
        let program =
            Program::parse(render_wgsl(), false, false, false, false, false, &[]).unwrap();
        let bindings = Bindings::default();
        for ep in ["vs", "fs"] {
            let msl = program
                .generate_msl(ep, &bindings, &[], 0, true, false, &[], 0xFFFF_FFFF, 0)
                .unwrap();
            assert!(!msl.source.is_empty());
            let spirv = program
                .generate_spirv(ep, &bindings, &[], true, false, 0, false, None, 0)
                .unwrap();
            assert_eq!(spirv.first().copied(), Some(0x0723_0203));
        }
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn framebuffer_fetch_reflects_color_and_generates_code() {
        let program = Program::parse(
            framebuffer_fetch_wgsl(),
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let inputs = program.entry_point_inputs("fs").unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].color, Some(0));

        let msl = program
            .generate_msl(
                "fs",
                &Bindings::default(),
                &[],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap();
        assert!(msl.source.contains("[[color(0)]]"), "MSL:\n{}", msl.source);

        let spirv = program
            .generate_spirv(
                "fs",
                &Bindings::default(),
                &[],
                true,
                false,
                0,
                false,
                None,
                0,
            )
            .unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv.first().copied(), Some(0x0723_0203));
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn input_attachment_reflects_index_and_generates_msl() {
        let program = Program::parse(
            input_attachment_wgsl(),
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let bindings = program.resource_bindings("fs").unwrap();
        let input_attachment = bindings
            .iter()
            .find(|binding| binding.resource_type == ResourceType::InputAttachment)
            .unwrap();
        assert_eq!(input_attachment.group, 0);
        assert_eq!(input_attachment.binding, 0);
        assert_eq!(input_attachment.input_attachment_index, 0);

        let msl = program
            .generate_msl(
                "fs",
                &Bindings {
                    input_attachment_color_index: vec![InputAttachmentColorIndex {
                        group: 0,
                        binding: 0,
                        color_slot: 0,
                    }],
                    ..Bindings::default()
                },
                &[],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap();
        assert!(msl.source.contains("[[color(0)]]"), "MSL:\n{}", msl.source);
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn input_attachment_msl_missing_color_slot_returns_err() {
        let program = Program::parse(
            input_attachment_wgsl(),
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let err = match program.generate_msl(
            "fs",
            &Bindings::default(),
            &[],
            0,
            true,
            false,
            &[],
            0xFFFF_FFFF,
            0,
        ) {
            Ok(_) => panic!("expected input attachment MSL generation to fail without color slot"),
            Err(err) => err,
        };
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn frag_position_used_reflection_and_pixel_center_polyfill() {
        // A fragment that reads @builtin(position) alongside a
        // @interpolate(_, sample) input (forces sample-rate shading), plus a
        // second fragment that does not read position.
        let program = Program::parse(
            r#"
struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(perspective, sample) uv: vec2<f32>,
};

@vertex
fn vs() -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  o.uv = vec2<f32>(0.0, 0.0);
  return o;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4<f32> {
  return vec4<f32>(v.pos.xy, v.uv);
}

@fragment
fn fs_no_pos(@location(0) @interpolate(perspective, sample) uv: vec2<f32>)
    -> @location(0) vec4<f32> {
  return vec4<f32>(uv, 0.0, 1.0);
}
"#,
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();

        // Reflection: frag_position_used tracks whether the fragment reads
        // @builtin(position).
        let entries = program.entry_points().unwrap();
        let fs = entries.iter().find(|e| e.name == "fs").unwrap();
        let fs_no_pos = entries.iter().find(|e| e.name == "fs_no_pos").unwrap();
        assert!(fs.frag_position_used, "fs reads @builtin(position)");
        assert!(
            !fs_no_pos.frag_position_used,
            "fs_no_pos does not read @builtin(position)"
        );

        // The pixel-center polyfill option must reach the SPIR-V writer and
        // change the emitted module (Tint reconstructs the pixel center from a
        // center-sampled interpolant at the given free location). Without the
        // option, FragCoord would leak the per-sample position under sample-rate
        // shading — the CTS-found bug this guards against.
        let without = program
            .generate_spirv(
                "fs",
                &Bindings::default(),
                &[],
                true,
                false,
                0,
                false,
                None,
                0,
            )
            .unwrap();
        let with = program
            .generate_spirv(
                "fs",
                &Bindings::default(),
                &[],
                true,
                false,
                0,
                false,
                Some(1),
                0,
            )
            .unwrap();
        assert_eq!(without.first().copied(), Some(0x0723_0203));
        assert_eq!(with.first().copied(), Some(0x0723_0203));
        assert_ne!(
            without, with,
            "polyfill_pixel_center must alter the emitted SPIR-V"
        );
    }

    /// Block 94 S3: a fragment that needs the pixel-center polyfill's
    /// internal depth-range immediates AND declares a user `var<immediate>`
    /// generates SPIR-V only when `user_immediate_size` rebases the
    /// depth-range offsets past the user region -- with the pre-S3 hardcoded
    /// `{0, 4}` shape (budget 0) Tint's `PrepareImmediateData` rejects the
    /// overlap outright, proving the threaded offset is load-bearing.
    /// Mirrors the MSL probe
    /// `fragment_msl_user_immediate_and_frag_depth_clamp_share_slot_with_non_colliding_offsets`.
    #[test]
    fn spirv_user_immediate_and_polyfill_depth_range_get_non_colliding_offsets() {
        let wgsl = r#"
requires immediate_address_space;

var<immediate> params : vec4f;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(perspective, sample) uv: vec2<f32>,
};

@vertex
fn vs() -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  o.uv = vec2<f32>(0.0, 0.0);
  return o;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4<f32> {
  return vec4<f32>(v.pos.xy, v.uv) + params;
}

@compute @workgroup_size(1)
fn cs() {
  let v = params;
  _ = v;
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[11]).unwrap();

        // `params` is a vec4f (16 bytes): with the matching layout budget the
        // depth-range pair lands at bytes {16, 20} and generation succeeds.
        let spirv = program
            .generate_spirv(
                "fs",
                &Bindings::default(),
                &[],
                true,
                false,
                0,
                false,
                Some(1),
                16,
            )
            .unwrap();
        assert_eq!(spirv.first().copied(), Some(0x0723_0203));

        // Budget 0 places the depth-range pair at {0, 4}, inside `params` --
        // Tint's PrepareImmediateData rejects the overlap with a hard error
        // rather than silently mislaying either block.
        let collision = program.generate_spirv(
            "fs",
            &Bindings::default(),
            &[],
            true,
            false,
            0,
            false,
            Some(1),
            0,
        );
        let Err(collision_err) = collision else {
            panic!(
                "a user_immediate_size (0) narrower than the entry's real \
                 var<immediate> usage (16 bytes) must be rejected as a \
                 depth-range/user-immediate offset collision"
            );
        };
        assert!(
            matches!(collision_err, TintError::Codegen(ref message) if message.contains("overlap")),
            "expected an offset-overlap codegen error, got: {collision_err:?}"
        );

        // A compute entry using the same user immediate has no internal
        // immediates: Tint places the user block at push-constant offset 0
        // regardless of the budget, and generation succeeds.
        let compute = program
            .generate_spirv(
                "cs",
                &Bindings::default(),
                &[],
                true,
                false,
                0,
                false,
                None,
                16,
            )
            .unwrap();
        assert_eq!(compute.first().copied(), Some(0x0723_0203));
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn input_attachment_generates_spirv() {
        let program = Program::parse(
            input_attachment_wgsl(),
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let spirv = program
            .generate_spirv(
                "fs",
                &Bindings::default(),
                &[],
                true,
                false,
                0,
                false,
                None,
                0,
            )
            .unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv.first().copied(), Some(0x0723_0203));
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn multisampled_input_attachment_flag_reaches_spirv_writer() {
        let program = Program::parse(
            input_attachment_wgsl(),
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let err = match program.generate_spirv(
            "fs",
            &Bindings::default(),
            &[],
            true,
            false,
            0,
            true,
            None,
            0,
        ) {
            Ok(_) => panic!("expected multisampled input attachment generation to fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("requires an explicit sample index"),
            "SPIR-V error:\n{err}"
        );
    }

    #[cfg(feature = "tiled")]
    fn multisampled_input_attachment_wgsl() -> &'static str {
        r#"
enable chromium_internal_input_attachments;

@group(0) @binding(0) @input_attachment_index(0) var ia: input_attachment<f32>;

@fragment
fn fs(@builtin(sample_index) s: u32) -> @location(0) vec4<f32> {
  return inputAttachmentLoad(ia, s);
}
"#
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn multisampled_input_attachment_generates_spirv() {
        let program = Program::parse(
            multisampled_input_attachment_wgsl(),
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let spirv = program
            .generate_spirv(
                "fs",
                &Bindings::default(),
                &[],
                true,
                false,
                0,
                true,
                None,
                0,
            )
            .unwrap();
        assert!(!spirv.is_empty());
        assert_eq!(spirv.first().copied(), Some(0x0723_0203));
    }

    /// A vertex + fragment pair sharing one module where the fragment uses the
    /// 2-arg `inputAttachmentLoad(ia, sample_index)`. `multisampled_input_attachment`
    /// is a MODULE-WIDE option: Tint validates the whole module's overloads even
    /// when generating the *vertex* entry point, so generating `"vs"` must be
    /// given the same `true` flag (else it fails). This guards the yawgpu-core
    /// invariant that both stages of a subpass pipeline receive the flag.
    #[cfg(feature = "tiled")]
    fn multisampled_input_attachment_vs_fs_wgsl() -> &'static str {
        r#"
enable chromium_internal_input_attachments;

@group(0) @binding(0) @input_attachment_index(0) var ia: input_attachment<f32>;

@vertex
fn vs() -> @builtin(position) vec4<f32> {
  return vec4<f32>(0.0);
}

@fragment
fn fs(@builtin(sample_index) s: u32) -> @location(0) vec4<f32> {
  return inputAttachmentLoad(ia, s);
}
"#
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn multisampled_input_attachment_vertex_stage_requires_module_wide_flag() {
        let program = Program::parse(
            multisampled_input_attachment_vs_fs_wgsl(),
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        // Generating the vertex entry with the module-wide flag TRUE succeeds …
        let vertex = program
            .generate_spirv(
                "vs",
                &Bindings::default(),
                &[],
                true,
                false,
                0,
                true,
                None,
                0,
            )
            .unwrap();
        assert_eq!(vertex.first().copied(), Some(0x0723_0203));
        // … while FALSE fails, because the module contains a 2-arg load. This is
        // why yawgpu-core must pass the flag to the vertex stage too, not just the
        // fragment.
        assert!(program
            .generate_spirv(
                "vs",
                &Bindings::default(),
                &[],
                true,
                false,
                0,
                false,
                None,
                0
            )
            .is_err());
    }

    #[cfg(not(feature = "tiled"))]
    #[test]
    fn framebuffer_fetch_enable_requires_tiled_feature() {
        let err = Program::parse(
            framebuffer_fetch_wgsl(),
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[cfg(not(feature = "tiled"))]
    #[test]
    fn input_attachment_enable_requires_tiled_feature() {
        let err = Program::parse(
            input_attachment_wgsl(),
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn framebuffer_fetch_requires_enable() {
        let err = Program::parse(
            r#"
@fragment
fn fs(@color(0) prev: vec4<f32>) -> @location(0) vec4<f32> {
  return prev;
}
"#,
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn vertex_msl_uses_vertex_pulling_when_configured() {
        let wgsl = r#"
struct VIn {
  @location(0) p: vec4f,
}

@vertex
fn vs(i: VIn) -> @builtin(position) vec4f {
  return i.p;
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let default_msl = program
            .generate_msl(
                "vs",
                &Bindings::default(),
                &[],
                1,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap()
            .source;
        let pulling_msl = program
            .generate_msl(
                "vs",
                &Bindings::default(),
                &[],
                1,
                true,
                false,
                &[VertexBuffer {
                    slot: 0,
                    metal_index: 0,
                    array_stride: 16,
                    step_mode: VertexStepMode::Vertex,
                    attributes: vec![VertexAttribute {
                        format: VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                0xFFFF_FFFF,
                0,
            )
            .unwrap()
            .source;

        assert!(default_msl.contains("stage_in"), "MSL:\n{default_msl}");
        assert!(!pulling_msl.contains("stage_in"), "MSL:\n{pulling_msl}");
        assert!(
            pulling_msl.contains("[[buffer(0)]]")
                && pulling_msl.contains("tint_storage_buffer_sizes"),
            "MSL:\n{pulling_msl}"
        );
        assert_ne!(pulling_msl, default_msl);
    }

    #[test]
    fn fragment_msl_fixed_sample_mask_affects_output() {
        let wgsl = r#"
@fragment
fn fs() -> @location(0) vec4f {
  return vec4f(1.0);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let default_msl = program
            .generate_msl(
                "fs",
                &Bindings::default(),
                &[],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap()
            .source;
        let masked_msl = program
            .generate_msl("fs", &Bindings::default(), &[], 0, true, false, &[], 0x1, 0)
            .unwrap()
            .source;

        assert_ne!(masked_msl, default_msl, "MSL:\n{masked_msl}");
    }

    #[test]
    fn fragment_msl_frag_depth_reports_clamp_slot() {
        let frag_depth_wgsl = r#"
@fragment
fn fs() -> @builtin(frag_depth) f32 {
  return 2.0;
}
"#;
        let program =
            Program::parse(frag_depth_wgsl, false, false, false, false, false, &[]).unwrap();
        let frag_depth_msl = program
            .generate_msl(
                "fs",
                &Bindings::default(),
                &[],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap();
        assert!(frag_depth_msl.frag_depth_clamp_slot.is_some());
        assert!(
            frag_depth_msl.source.contains("metal::clamp")
                || frag_depth_msl.source.contains("clamp("),
            "MSL:\n{}",
            frag_depth_msl.source
        );

        let color_wgsl = r#"
@fragment
fn fs() -> @location(0) vec4f {
  return vec4f(0.0);
}
"#;
        let program = Program::parse(color_wgsl, false, false, false, false, false, &[]).unwrap();
        let color_msl = program
            .generate_msl(
                "fs",
                &Bindings::default(),
                &[],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap();
        assert_eq!(color_msl.frag_depth_clamp_slot, None);
    }

    /// Block 94 S2: a compute entry point with a user `var<immediate>`
    /// reports `immediate_slot` and the generated MSL declares the
    /// combined immediate block at that exact Metal buffer slot.
    #[test]
    fn compute_msl_user_immediate_declares_block_at_reported_slot() {
        let wgsl = r#"
requires immediate_address_space;

var<immediate> params : vec4f;

@compute @workgroup_size(1)
fn cs() {
  let v = params;
  _ = v;
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[11]).unwrap();
        // `params` is a vec4f (16 bytes); the owning pipeline layout reserves
        // exactly that much user-immediate budget.
        let msl = program
            .generate_msl(
                "cs",
                &Bindings::default(),
                &[],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                16,
            )
            .unwrap();

        let slot = msl
            .immediate_slot
            .expect("compute entry using var<immediate> must report an immediate slot");
        // No frag-depth clamp exists on a compute entry point.
        assert_eq!(msl.frag_depth_clamp_slot, None);
        assert!(
            msl.source.contains(&format!("[[buffer({slot})]]")),
            "MSL:\n{}",
            msl.source
        );
    }

    /// Block 94 S2: a fragment entry point that both writes `frag_depth`
    /// (needing the internal clamp range) AND declares a user
    /// `var<immediate>` gets the SAME combined-block Metal slot for both
    /// (`immediate_slot == frag_depth_clamp_slot`), and the clamp range is
    /// rebased to sit after the full `user_immediate_size` budget rather
    /// than colliding with it at a hardcoded offset 0 (the pre-S2
    /// behaviour). Rebasing the clamp offset via `user_immediate_size`
    /// changes the generated MSL relative to the offset-0 baseline,
    /// confirming the parameter is actually threaded through to Tint's
    /// `depth_range_offsets`.
    #[test]
    fn fragment_msl_user_immediate_and_frag_depth_clamp_share_slot_with_non_colliding_offsets() {
        let wgsl = r#"
requires immediate_address_space;

var<immediate> params : vec4f;

@fragment
fn fs() -> @builtin(frag_depth) f32 {
  let v = params;
  return v.x;
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[11]).unwrap();
        // `params` is a vec4f (16 bytes): the clamp range must land at byte
        // offset 16, not overlap `params` at `[0, 16)`.
        let msl = program
            .generate_msl(
                "fs",
                &Bindings::default(),
                &[],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                16,
            )
            .unwrap();

        let immediate_slot = msl
            .immediate_slot
            .expect("fragment entry using var<immediate> must report an immediate slot");
        let clamp_slot = msl
            .frag_depth_clamp_slot
            .expect("fragment entry writing frag_depth must report a clamp slot");
        assert_eq!(
            immediate_slot, clamp_slot,
            "user immediates and the frag-depth clamp share one combined block/slot"
        );

        // Negative check: a `user_immediate_size` narrower than the entry's
        // real `var<immediate>` usage (16 bytes) would place the clamp range
        // AT or inside the user member -- exactly the collision the
        // rebasing exists to avoid (the pre-S2 code hardcoded offset 0,
        // which collided with any real user immediate). Tint's own
        // `PrepareImmediateData` (prepare_immediate_data.cc) rejects this
        // with a hard `Failure` rather than silently overlapping, and
        // `yawgpu_tint_generate_msl` surfaces that as `Err` here -- proving
        // the offset is genuinely load-bearing, not ignored.
        let collision_result = program.generate_msl(
            "fs",
            &Bindings::default(),
            &[],
            0,
            true,
            false,
            &[],
            0xFFFF_FFFF,
            0,
        );
        let Err(collision_err) = collision_result else {
            panic!(
                "a user_immediate_size (0) narrower than the entry's real \
                 var<immediate> usage (16 bytes) must be rejected as a \
                 clamp/user-immediate offset collision"
            );
        };
        assert!(
            matches!(collision_err, TintError::Codegen(ref message) if message.contains("overlap")),
            "expected an offset-overlap codegen error, got: {collision_err:?}"
        );
    }

    #[test]
    fn reflects_bindings_and_workgroup_size() {
        let wgsl = r#"
struct U { value: vec4f }
struct S { value: array<vec4f, 4> }
@group(0) @binding(0) var<uniform> u: U;
@group(1) @binding(2) var<storage, read_write> s: S;
@group(2) @binding(3) var t: texture_2d<f32>;
@group(2) @binding(4) var samp: sampler;

@compute @workgroup_size(8, 4, 1)
fn cs() {
  let c = textureSampleLevel(t, samp, vec2f(0.5), 0.0);
  s.value[0] = u.value + c;
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let entries = program.entry_points().unwrap();
        assert_eq!(entries[0].workgroup_size, Some([8, 4, 1]));
        let bindings = program.resource_bindings("cs").unwrap();
        assert!(bindings.iter().any(|b| {
            b.group == 0 && b.binding == 0 && b.resource_type == ResourceType::UniformBuffer
        }));
        assert!(bindings.iter().any(|b| {
            b.group == 1 && b.binding == 2 && b.resource_type == ResourceType::StorageBuffer
        }));
        assert!(bindings.iter().any(|b| {
            b.group == 2 && b.binding == 3 && b.resource_type == ResourceType::SampledTexture
        }));
        assert!(bindings
            .iter()
            .any(|b| b.group == 2 && b.binding == 4 && b.resource_type == ResourceType::Sampler));
    }

    #[test]
    fn exposes_non_error_diagnostics() {
        let program = Program::parse(
            "diagnostic(info, bogus_rule);\n@compute @workgroup_size(1) fn cs() {}",
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let diagnostics = program.diagnostics().unwrap();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, DiagnosticSeverity::Warning);
        assert!(diagnostics[0]
            .message
            .contains("unrecognized diagnostic rule"));
    }

    #[test]
    fn reflects_texture_sample_usage() {
        let wgsl = r#"
@group(0) @binding(0) var load_tex: texture_2d<f32>;
@group(0) @binding(1) var sample_tex: texture_2d<f32>;
@group(0) @binding(2) var gather_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

fn helper(t: texture_2d<f32>) -> vec4f {
  return textureGather(0, t, samp, vec2f(0.5));
}

@compute @workgroup_size(1)
fn cs() {
  _ = textureLoad(load_tex, vec2i(0), 0);
  _ = textureSampleLevel(sample_tex, samp, vec2f(0.5), 0.0);
  _ = helper(gather_tex);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let bindings = program.resource_bindings("cs").unwrap();
        let usage = |binding: u32| {
            bindings
                .iter()
                .find(|resource| resource.group == 0 && resource.binding == binding)
                .map(|resource| resource.sample_usage)
                .unwrap()
        };
        assert_eq!(usage(0), TextureSampleUsage::Load);
        assert_eq!(usage(1), TextureSampleUsage::Sample);
        assert_eq!(usage(2), TextureSampleUsage::Gather);
    }

    #[test]
    fn reflects_entry_point_stage_variables() {
        let wgsl = r#"
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
fn fs(@location(0) value: f32, @location(1) @interpolate(flat) index: u32) -> @location(0) vec4f {
  return vec4f(value + f32(index), 0.0, 0.0, 1.0);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();

        let inputs = program.entry_point_inputs("fs").unwrap();
        let value = inputs
            .iter()
            .find(|variable| variable.location == Some(0))
            .unwrap();
        assert_eq!(value.component_type, ComponentType::F32);
        assert_eq!(value.composition_type, CompositionType::Scalar);

        let index = inputs
            .iter()
            .find(|variable| variable.location == Some(1))
            .unwrap();
        assert_eq!(index.component_type, ComponentType::U32);
        assert_eq!(index.composition_type, CompositionType::Scalar);
        assert_eq!(index.interpolation_type, InterpolationType::Flat);
        assert_eq!(index.interpolation_sampling, InterpolationSampling::First);

        let outputs = program.entry_point_outputs("vs").unwrap();
        assert!(outputs.iter().any(|variable| {
            variable.location == Some(1)
                && variable.component_type == ComponentType::U32
                && variable.interpolation_type == InterpolationType::Flat
        }));
    }

    #[test]
    fn reflects_and_substitutes_overrides() {
        let wgsl = r#"
override x: u32 = 4;
@compute @workgroup_size(x, 1, 1)
fn cs() {}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let overrides = program.overrides().unwrap();
        assert_eq!(overrides.len(), 1);
        assert_eq!(overrides[0].name, "x");
        assert_eq!(overrides[0].type_class, OverrideType::Uint32);
        assert!(overrides[0].has_default);
        assert_eq!(overrides[0].default_value, 4.0);

        let bindings = Bindings::default();
        let a = program
            .generate_msl(
                "cs",
                &bindings,
                &[OverrideValue {
                    name: "x".to_owned(),
                    value: 2.0,
                }],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap()
            .source;
        let b = program
            .generate_msl(
                "cs",
                &bindings,
                &[OverrideValue {
                    name: "x".to_owned(),
                    value: 5.0,
                }],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap()
            .source;
        assert_ne!(a, b);
    }

    #[test]
    fn reflects_override_default_values() {
        let wgsl = r#"
override f: f32 = 1.5;
override b: bool = true;
override u: u32 = 7u;
override i: i32 = -3i;
@compute @workgroup_size(1) fn cs() {}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let overrides = program.overrides().unwrap();
        let default = |name: &str| {
            overrides
                .iter()
                .find(|override_| override_.name == name)
                .map(|override_| override_.default_value)
                .unwrap()
        };
        assert_eq!(default("f"), 1.5);
        assert_eq!(default("b"), 1.0);
        assert_eq!(default("u"), 7.0);
        assert_eq!(default("i"), -3.0);
    }

    #[test]
    fn external_texture_remap_generates_msl() {
        let wgsl = r#"
@group(0) @binding(0) var s: sampler;
@group(0) @binding(1) var t: texture_external;

@fragment
fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  return textureSampleBaseClampToEdge(t, s, p.xy);
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let bindings = Bindings {
            sampler: vec![BindingRemap {
                group: 0,
                binding: 0,
                dst_group: 0,
                dst_binding: 0,
            }],
            external_texture: vec![ExternalTextureRemap {
                group: 0,
                binding: 1,
                plane0_slot: 0,
                plane1_slot: 1,
                params_slot: 0,
            }],
            ..Bindings::default()
        };

        let msl = program
            .generate_msl("fs", &bindings, &[], 2, true, false, &[], 0xFFFF_FFFF, 0)
            .unwrap();
        assert!(!msl.source.is_empty(), "MSL was empty");
        assert!(msl.source.contains("sampler"), "MSL:\n{}", msl.source);
        assert!(
            msl.source.matches("texture2d").count() >= 2,
            "MSL:\n{}",
            msl.source
        );
    }

    #[test]
    fn binding_remap_changes_msl_and_spirv() {
        let wgsl = r#"
struct U { value: vec4f }
@group(0) @binding(0) var<uniform> u: U;
@compute @workgroup_size(1)
fn cs() { _ = u.value; }
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let default_bindings = Bindings::default();
        let remapped = Bindings {
            uniform: vec![BindingRemap {
                group: 0,
                binding: 0,
                dst_group: 0,
                dst_binding: 7,
            }],
            ..Bindings::default()
        };
        let default_msl = program
            .generate_msl(
                "cs",
                &default_bindings,
                &[],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap()
            .source;
        let remapped_msl = program
            .generate_msl("cs", &remapped, &[], 0, true, false, &[], 0xFFFF_FFFF, 0)
            .unwrap()
            .source;
        assert!(remapped_msl.contains("[[buffer(7)]]"), "{remapped_msl}");
        assert_ne!(default_msl, remapped_msl);

        let default_spv = program
            .generate_spirv("cs", &default_bindings, &[], true, false, 0, false, None, 0)
            .unwrap();
        let remapped_spv = program
            .generate_spirv("cs", &remapped, &[], true, false, 0, false, None, 0)
            .unwrap();
        assert_ne!(default_spv, remapped_spv);
        assert!(spirv_has_binding_decoration(&remapped_spv, 7));
    }

    #[test]
    fn packed_4x8_integer_dot_product_without_requires_matches_pinned_tint() {
        // NOTE: pinned-Tint gap -- packed_4x8 builtins are not gated on `requires`
        // in this Dawn revision; tracked as a CTS expectation.
        let program = Program::parse(
            "@compute @workgroup_size(1) fn m() { let v = dot4I8Packed(1u, 2u); }",
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        drop(program);
    }

    #[test]
    fn readonly_and_readwrite_storage_textures_requires_shipped_feature_parses() {
        let wgsl = r#"
requires readonly_and_readwrite_storage_textures;

@group(0) @binding(0)
var tex : texture_storage_2d<r32uint, read_write>;

@compute @workgroup_size(1)
fn cs() {
  textureStore(tex, vec2i(0, 0), vec4u(1u, 0u, 0u, 0u));
  let value = textureLoad(tex, vec2i(0, 0));
  _ = value;
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[1]).unwrap();
        drop(program);
    }

    #[test]
    fn unrestricted_pointer_parameters_requires_shipped_feature_parses() {
        let wgsl = r#"
requires unrestricted_pointer_parameters;

struct Data {
  value: u32,
}

fn read_value(data: ptr<function, Data>) -> u32 {
  return (*data).value;
}

@compute @workgroup_size(1)
fn cs() {
  var data = Data(42u);
  let value = read_value(&data);
  _ = value;
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[3]).unwrap();
        drop(program);
    }

    #[test]
    fn dual_source_blending_extension_is_gated_by_parse_option() {
        let wgsl = r#"
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
        assert!(Program::parse(wgsl, false, false, false, false, false, &[]).is_err());
        let program = Program::parse(wgsl, false, false, true, false, false, &[]).unwrap();
        let outputs = program.entry_point_outputs("fs").unwrap();
        assert!(outputs.iter().any(|variable| variable.blend_src == Some(0)));
        assert!(outputs.iter().any(|variable| variable.blend_src == Some(1)));
    }

    #[test]
    fn clip_distances_extension_is_gated_by_parse_option() {
        let wgsl = r#"
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

        assert!(Program::parse(wgsl, false, false, false, false, false, &[]).is_err());
        let program = Program::parse(wgsl, false, false, false, true, false, &[]).unwrap();
        let entry_points = program.entry_points().unwrap();
        let vertex = entry_points
            .iter()
            .find(|entry| entry.name == "vs")
            .expect("vertex entry point should be reflected");
        assert_eq!(vertex.clip_distances_size, Some(1));
    }

    #[test]
    fn primitive_index_extension_is_gated_by_parse_option() {
        let wgsl = r#"
enable primitive_index;

@fragment
fn fs(@builtin(primitive_index) idx: u32) -> @location(0) vec4f {
  return vec4f(f32(idx), 0.0, 0.0, 1.0);
}
"#;

        assert!(Program::parse(wgsl, false, false, false, false, false, &[]).is_err());
        let program = Program::parse(wgsl, false, false, false, false, true, &[]).unwrap();
        let entry_points = program.entry_points().unwrap();
        let fragment = entry_points
            .iter()
            .find(|entry| entry.name == "fs")
            .expect("fragment entry point should be reflected");
        assert!(fragment.primitive_index_used);
    }

    #[test]
    fn shader_f16_parses_and_generates() {
        let wgsl = r#"
enable f16;
@compute @workgroup_size(1)
fn cs() {
  let x: f16 = 1.0h;
  _ = x;
}
"#;
        let program = Program::parse(wgsl, true, false, false, false, false, &[]).unwrap();
        let msl = program
            .generate_msl(
                "cs",
                &Bindings::default(),
                &[],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap();
        assert!(msl.source.contains("kernel"));
    }

    #[test]
    fn subgroups_extension_is_gated_by_parse_option() {
        let wgsl = r#"
enable subgroups;

@compute @workgroup_size(1)
fn cs() {
  let x = subgroupAdd(1u);
  _ = x;
}
"#;
        assert!(Program::parse(wgsl, false, false, false, false, false, &[]).is_err());
        assert!(Program::parse(wgsl, false, true, false, false, false, &[]).is_ok());
    }

    #[test]
    fn invalid_wgsl_reports_error() {
        let err =
            Program::parse("this is not wgsl", false, false, false, false, false, &[]).unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn parse_then_generate_msl_smoke_test() {
        // Was `wgsl_to_msl(compute_wgsl(), "cs")` via the now-deleted legacy
        // free-fn wrapper (refactor R5, F9); inlined per its removal note.
        let program =
            Program::parse(compute_wgsl(), false, false, false, false, false, &[]).unwrap();
        let msl = program
            .generate_msl(
                "cs",
                &Bindings::default(),
                &[],
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap();
        assert!(msl.source.contains("kernel"));
    }

    // -----------------------------------------------------------------------
    // R4 byte-compare regression tests (specs/tracking/tint-integration-
    // refactor.md, F4/F6). Each `generate_msl`/`generate_spirv` call re-lowers
    // the Tint program to IR from scratch (`lower_ir` in tint_shim.cpp) --
    // this Tint revision has no whole-module IR clone API, and Dawn itself
    // (`ShaderModuleMTL.mm`, `ShaderModuleVk.cpp`, `ComputePipelineWGPU.cpp`,
    // ...) also re-runs `ProgramToLoweredIR` fresh at every shader-stage
    // compile rather than caching+cloning IR, so this shim intentionally
    // matches that behavior instead of inventing an unproven module-clone.
    // These tests pin the invariant that repeated codegen calls on the same
    // `Program`, and codegen on a freshly re-parsed `Program` for identical
    // WGSL, produce byte-identical output -- the property the (deferred) IR
    // cache would have had to preserve, and the property the direct
    // `resolved_workgroup_size` query below relies on when it supersedes the
    // deleted SPIR-V-generate-and-parse round trip.
    // -----------------------------------------------------------------------

    #[test]
    fn generate_msl_and_spirv_are_byte_identical_across_calls_runtime_array() {
        let wgsl = r#"
struct Data {
  values: array<u32>,
}

@group(1) @binding(2) var<storage, read_write> data: Data;

@compute @workgroup_size(1)
fn cs() {
  if (arrayLength(&data.values) > 0u) {
    data.values[0] = arrayLength(&data.values);
  }
}
"#;
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let fresh = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();

        let generate_msl = |program: &Program| {
            program
                .generate_msl(
                    "cs",
                    &Bindings::default(),
                    &[],
                    9,
                    true,
                    false,
                    &[],
                    0xFFFF_FFFF,
                    0,
                )
                .unwrap()
        };
        let first = generate_msl(&program);
        let second = generate_msl(&program);
        assert_eq!(first.source, second.source);
        assert_eq!(first.entry_point, second.entry_point);
        assert_eq!(
            first.needs_storage_buffer_sizes,
            second.needs_storage_buffer_sizes
        );
        assert_eq!(first.buffer_size_bindings, second.buffer_size_bindings);
        assert_eq!(first.workgroup_allocations, second.workgroup_allocations);
        assert_eq!(first.frag_depth_clamp_slot, second.frag_depth_clamp_slot);
        let third = generate_msl(&fresh);
        assert_eq!(first.source, third.source);
        assert_eq!(first.entry_point, third.entry_point);
        assert_eq!(first.buffer_size_bindings, third.buffer_size_bindings);
        assert_eq!(first.workgroup_allocations, third.workgroup_allocations);

        let generate_spirv = |program: &Program| {
            program
                .generate_spirv(
                    "cs",
                    &Bindings::default(),
                    &[],
                    true,
                    false,
                    0,
                    false,
                    None,
                    0,
                )
                .unwrap()
        };
        let spirv_first = generate_spirv(&program);
        let spirv_second = generate_spirv(&program);
        assert_eq!(spirv_first, spirv_second);
        let spirv_third = generate_spirv(&fresh);
        assert_eq!(spirv_first, spirv_third);
    }

    #[test]
    fn generate_msl_and_spirv_are_byte_identical_across_calls_render_vertex_pulling() {
        let wgsl = r#"
struct VIn {
  @location(0) p: vec4f,
}

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(i: VIn) -> VsOut {
  var out: VsOut;
  out.pos = i.p;
  out.uv = i.p.xy;
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  return vec4f(in.uv, 0.0, 1.0);
}
"#;
        let vertex_buffers = [VertexBuffer {
            slot: 0,
            metal_index: 0,
            array_stride: 16,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![VertexAttribute {
                format: VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            }],
        }];
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let fresh = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();

        let generate_vs_msl = |program: &Program| {
            program
                .generate_msl(
                    "vs",
                    &Bindings::default(),
                    &[],
                    1,
                    true,
                    false,
                    &vertex_buffers,
                    0xFFFF_FFFF,
                    0,
                )
                .unwrap()
        };
        let first = generate_vs_msl(&program);
        let second = generate_vs_msl(&program);
        assert_eq!(first.source, second.source);
        assert_eq!(first.buffer_size_bindings, second.buffer_size_bindings);
        assert!(!first.source.contains("stage_in"), "MSL:\n{}", first.source);
        let third = generate_vs_msl(&fresh);
        assert_eq!(first.source, third.source);

        for ep in ["vs", "fs"] {
            let generate_spirv = |program: &Program| {
                program
                    .generate_spirv(
                        ep,
                        &Bindings::default(),
                        &[],
                        true,
                        false,
                        0,
                        false,
                        None,
                        0,
                    )
                    .unwrap()
            };
            let a = generate_spirv(&program);
            let b = generate_spirv(&program);
            assert_eq!(a, b);
            let c = generate_spirv(&fresh);
            assert_eq!(a, c);
        }
    }

    #[test]
    fn generate_msl_is_byte_identical_across_calls_with_override_driven_workgroup_size() {
        let wgsl = r#"
override x: u32 = 4;

@compute @workgroup_size(x, 2, 1)
fn cs() {}
"#;
        let overrides = [OverrideValue {
            name: "x".into(),
            value: 8.0,
        }];
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let fresh = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();

        let generate = |program: &Program| {
            program
                .generate_msl(
                    "cs",
                    &Bindings::default(),
                    &overrides,
                    0,
                    true,
                    false,
                    &[],
                    0xFFFF_FFFF,
                    0,
                )
                .unwrap()
        };
        let first = generate(&program);
        let second = generate(&program);
        assert_eq!(first.source, second.source);
        // x=8, y=2, z=1 (override x=8 applied) -> total threads = 16; MSL
        // emits only the total, not the individual dimensions.
        assert!(
            first
                .source
                .contains("max_total_threads_per_threadgroup(16)"),
            "MSL:\n{}",
            first.source
        );
        let third = generate(&fresh);
        assert_eq!(first.source, third.source);
    }

    #[test]
    fn generate_msl_resolves_id_specified_override_referenced_by_name() {
        // Companion to
        // `resolved_workgroup_size_resolves_id_specified_override_referenced_by_name`,
        // covering the same name -> OverrideId map lookup (F3 fix) through
        // `generate_msl`'s override-substitution path instead.
        let wgsl = r#"
@id(3) override x: u32 = 4;

@compute @workgroup_size(x, 2, 1)
fn cs() {}
"#;
        let overrides = [OverrideValue {
            name: "x".into(),
            value: 8.0,
        }];
        let program = Program::parse(wgsl, false, false, false, false, false, &[]).unwrap();
        let output = program
            .generate_msl(
                "cs",
                &Bindings::default(),
                &overrides,
                0,
                true,
                false,
                &[],
                0xFFFF_FFFF,
                0,
            )
            .unwrap();
        // x=8, y=2, z=1 (override x=8 applied) -> total threads = 16.
        assert!(
            output
                .source
                .contains("max_total_threads_per_threadgroup(16)"),
            "MSL:\n{}",
            output.source
        );
    }

    #[test]
    fn generate_msl_and_spirv_are_byte_identical_across_calls_f16() {
        let wgsl = r#"
enable f16;
@compute @workgroup_size(1)
fn cs() {
  let x: f16 = 1.0h;
  _ = x;
}
"#;
        let program = Program::parse(wgsl, true, false, false, false, false, &[]).unwrap();
        let fresh = Program::parse(wgsl, true, false, false, false, false, &[]).unwrap();

        let generate_msl = |program: &Program| {
            program
                .generate_msl(
                    "cs",
                    &Bindings::default(),
                    &[],
                    0,
                    true,
                    false,
                    &[],
                    0xFFFF_FFFF,
                    0,
                )
                .unwrap()
        };
        let first = generate_msl(&program);
        let second = generate_msl(&program);
        assert_eq!(first.source, second.source);
        let third = generate_msl(&fresh);
        assert_eq!(first.source, third.source);

        let generate_spirv = |program: &Program| {
            program
                .generate_spirv(
                    "cs",
                    &Bindings::default(),
                    &[],
                    true,
                    false,
                    0,
                    false,
                    None,
                    0,
                )
                .unwrap()
        };
        let spirv_first = generate_spirv(&program);
        let spirv_second = generate_spirv(&program);
        assert_eq!(spirv_first, spirv_second);
        let spirv_third = generate_spirv(&fresh);
        assert_eq!(spirv_first, spirv_third);
    }

    // -----------------------------------------------------------------------
    // `resolved_workgroup_size` (F6): direct IR query replacing the deleted
    // SPIR-V-generate-and-parse round trip (`spirv_local_size` in
    // yawgpu-core's `shader_tint.rs`). Expected [x, y, z] values below are
    // reasoned directly from each WGSL source (or its override input), the
    // same values the deleted SPIR-V `OpExecutionMode LocalSize` round trip
    // used to produce.
    // -----------------------------------------------------------------------

    #[test]
    fn resolved_workgroup_size_matches_literal_reflection() {
        let program = Program::parse(
            "@compute @workgroup_size(8, 4, 1) fn cs() {}",
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(
            program.resolved_workgroup_size("cs", &[]).unwrap(),
            [8, 4, 1]
        );
    }

    #[test]
    fn resolved_workgroup_size_resolves_named_override_driven_size() {
        let program = Program::parse(
            r#"
override x: u32 = 4;

@compute @workgroup_size(x, 2, 1)
fn cs() {}
"#,
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        // No override supplied: falls back to the override's default (x=4).
        assert_eq!(
            program.resolved_workgroup_size("cs", &[]).unwrap(),
            [4, 2, 1]
        );
        // Override supplied by WGSL identifier name.
        assert_eq!(
            program
                .resolved_workgroup_size(
                    "cs",
                    &[OverrideValue {
                        name: "x".into(),
                        value: 8.0,
                    }]
                )
                .unwrap(),
            [8, 2, 1]
        );
    }

    #[test]
    fn resolved_workgroup_size_resolves_id_driven_override() {
        let program = Program::parse(
            r#"
@id(3) override x: u32 = 4;

@compute @workgroup_size(x)
fn cs() {}
"#,
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        // Override supplied by numeric @id, y/z default to 1.
        assert_eq!(
            program
                .resolved_workgroup_size(
                    "cs",
                    &[OverrideValue {
                        name: "3".into(),
                        value: 16.0,
                    }]
                )
                .unwrap(),
            [16, 1, 1]
        );
    }

    #[test]
    fn resolved_workgroup_size_resolves_id_specified_override_referenced_by_name() {
        // Same shader as `resolved_workgroup_size_resolves_id_driven_override`,
        // but the override is supplied by its WGSL identifier ("x") rather
        // than the numeric `@id` value. This forces resolution through the
        // name -> OverrideId map (`make_override_config`'s
        // `named_override_ids` lookup in tint_shim.cpp) instead of the
        // numeric-string fast path, verifying an `@id`-attributed override
        // is still discoverable by name (F3 fix: the map is now built from
        // `YawgpuTintProgram::overrides`, cached at program-creation time,
        // rather than a fresh `Inspector::GetNamedOverrideIds()` call).
        let program = Program::parse(
            r#"
@id(3) override x: u32 = 4;

@compute @workgroup_size(x)
fn cs() {}
"#,
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(
            program
                .resolved_workgroup_size(
                    "cs",
                    &[OverrideValue {
                        name: "x".into(),
                        value: 16.0,
                    }]
                )
                .unwrap(),
            [16, 1, 1]
        );
    }

    #[test]
    fn resolved_workgroup_size_errors_on_unknown_entry_point() {
        let program = Program::parse(
            "@compute @workgroup_size(1) fn cs() {}",
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let err = program.resolved_workgroup_size("missing", &[]).unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn resolved_workgroup_size_errors_on_unknown_override_name() {
        let program = Program::parse(
            r#"
override x: u32 = 4;

@compute @workgroup_size(x)
fn cs() {}
"#,
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let err = program
            .resolved_workgroup_size(
                "cs",
                &[OverrideValue {
                    name: "does_not_exist".into(),
                    value: 1.0,
                }],
            )
            .unwrap_err();
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn resolved_workgroup_size_is_entry_scoped_for_override_const_eval_errors() {
        // `cx`'s initializer const-evals to 1u / 0u once `cu`'s default is
        // substituted. Only the entry point that references `cx` may fail;
        // the sibling must resolve, because substitution runs after
        // SingleEntryPoint strips unreferenced module-scope declarations --
        // the same scoping the generate paths apply (writer Raise() runs
        // SingleEntryPoint, then SubstituteOverrides).
        let program = Program::parse(
            r#"
override cu: u32 = 0u;
override cx: u32 = 1u / cu;

@compute @workgroup_size(1)
fn main_pipe_error() {
  _ = cx;
}

@compute @workgroup_size(8, 4, 1)
fn main_ok() {}
"#,
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let err = program
            .resolved_workgroup_size("main_pipe_error", &[])
            .unwrap_err();
        assert!(!err.to_string().is_empty());
        assert_eq!(
            program.resolved_workgroup_size("main_ok", &[]).unwrap(),
            [8, 4, 1]
        );
    }

    #[test]
    fn resolved_workgroup_size_errors_on_non_compute_entry_point() {
        let program = Program::parse(
            "@fragment fn fs() -> @location(0) vec4f { return vec4f(); }",
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .unwrap();
        let err = program.resolved_workgroup_size("fs", &[]).unwrap_err();
        assert!(!err.to_string().is_empty());
    }
}
