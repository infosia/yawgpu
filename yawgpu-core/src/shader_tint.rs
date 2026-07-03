//! Tint shader frontend — yawgpu's WGSL→{MSL, SPIR-V, GLSL} compiler and
//! reflection source, backed by Dawn's Tint via the `yawgpu-tint` shim. This is
//! the sole shader frontend (the `crate::frontend` alias points here), and the
//! render path emits per-stage shader sources for pipeline creation.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::shader::{CompilationMessage, CompilationSeverity};
pub(crate) use crate::shader_types::*;

/// Canonical, order-insensitive, hashable form of `PipelineConstants`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CanonicalConstants(Vec<(String, u64)>);

impl From<&PipelineConstants> for CanonicalConstants {
    fn from(constants: &PipelineConstants) -> Self {
        let mut constants = constants
            .constants
            .iter()
            .map(|(name, value)| (name.clone(), value.to_bits()))
            .collect::<Vec<_>>();
        constants.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
        Self(constants)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MslCodegenKey {
    entry_point: String,
    binding_map: MslBindingMap,
    constants: CanonicalConstants,
    subpass_color_slots: Vec<((u32, u32), u32)>,
    vertex_buffers: Vec<MslVertexBufferBinding>,
    disable_robustness: bool,
    emit_vertex_point_size: bool,
    fixed_sample_mask: u32,
    user_immediate_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SpirvCodegenKey {
    entry_point: String,
    constants: CanonicalConstants,
    vulkan_memory_model: bool,
    framebuffer_fetch_descriptor_set: u32,
    multisampled_input_attachment: bool,
    polyfill_pixel_center: Option<u32>,
    user_immediate_size: u32,
}

#[cfg(feature = "gles")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GlslCodegenKey {
    entry_point: String,
    stage: ShaderStage,
    constants: CanonicalConstants,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct WorkgroupResolveKey {
    entry_point: String,
    constants: CanonicalConstants,
}

/// Stores reflected shader module data used by validation and backend submission.
///
/// Reflection (entry points, per-entry IO, per-entry resource bindings,
/// fragment builtins, overrides) is memoized lazily on first access: each
/// shader module is parsed once but reflected repeatedly by pipeline
/// validation (a render pipeline resolve queries the same entry point's IO
/// and resource bindings from several independent validators), and every
/// accessor previously re-crossed the Tint FFI boundary on each call — see
/// finding F5 in `specs/tracking/tint-integration-refactor.md`. The caches
/// are filled lazily (not at construction) so `createShaderModule` — which
/// never touches most of this data for shaders that are never used in a
/// pipeline — does not get more expensive.
#[derive(Debug)]
pub struct ReflectedModule {
    /// Tint program.
    pub program: yawgpu_tint::Program,
    /// Non-fatal compilation warnings.
    pub(crate) warnings: Vec<CompilationMessage>,
    /// Raw Tint entry-point reflection, memoized once. Backing store for
    /// [`Self::entry_points`], [`Self::entry_point_io`], and
    /// [`Self::fragment_builtins`], which all need per-entry fields (stage,
    /// `frag_depth_used`, …) that the simplified [`ReflectedEntryPoint`]
    /// does not carry.
    raw_entry_points: OnceLock<Vec<yawgpu_tint::EntryPoint>>,
    /// Memoized [`Self::entry_points`] result.
    entry_points_cache: OnceLock<Vec<ReflectedEntryPoint>>,
    /// Per-entry-point IO reflection, memoized by entry-point name on first
    /// [`Self::entry_point_io`] call for that name. `None` means the entry
    /// point does not exist (also memoized, so a repeated miss is free).
    entry_point_io_cache: Mutex<HashMap<String, Option<ReflectedEntryPointIo>>>,
    /// Per-entry-point resource-binding reflection, memoized by entry-point
    /// name on first [`Self::resource_bindings_for_entry`] call for that
    /// name. Caches the `Result` (including the "entry point not found"
    /// error) so a repeated call never re-crosses the FFI boundary.
    resource_bindings_cache: Mutex<HashMap<String, Result<Vec<ReflectedResourceBinding>, String>>>,
    /// Per-entry-point immediate data size (bytes), memoized by entry-point
    /// name on first [`Self::immediate_data_size`] call for that name.
    /// Caches the `Result` (including the "entry point not found" error) so
    /// a repeated call never re-crosses the FFI boundary.
    immediate_data_size_cache: Mutex<HashMap<String, Result<u32, String>>>,
    /// Per-entry-point required immediate-data slots, memoized by entry-point
    /// name on first [`Self::immediate_data_used_slots`] call for that name.
    /// Caches the `Result` (including the "entry point not found" error) so
    /// a repeated call never re-crosses the FFI boundary.
    immediate_data_used_slots_cache: Mutex<HashMap<String, Result<u64, String>>>,
    /// Per-module MSL codegen results. Values include memoized `Err`s because
    /// failures are deterministic for a module and key, so repeated failing
    /// pipeline creation must not re-cross Tint. Entries live as long as the
    /// module and are bounded by distinct pipeline configs ever created from
    /// it; if profiles demand a cap later, clear-all-at-N is the intended
    /// escape hatch. The lock is held across Tint codegen to dedup concurrent
    /// identical compiles; shared-program codegen is documented concurrent-safe,
    /// so this is policy rather than soundness. The only nested cache call is
    /// `resource_bindings_for_entry`, which uses a different mutex, and
    /// per-generator mutexes keep unrelated generators unserialized.
    msl_codegen_cache: Mutex<HashMap<MslCodegenKey, Result<GeneratedMsl, String>>>,
    /// Per-module SPIR-V codegen results. Values include memoized `Err`s
    /// because failures are deterministic for a module and key, so repeated
    /// failing pipeline creation must not re-cross Tint. Entries live as long
    /// as the module and are bounded by distinct pipeline configs ever created
    /// from it; if profiles demand a cap later, clear-all-at-N is the intended
    /// escape hatch. The lock is held across Tint codegen to dedup concurrent
    /// identical compiles; shared-program codegen is documented concurrent-safe,
    /// so this is policy rather than soundness. The miss branch nests no cache
    /// call at all, and
    /// per-generator mutexes keep unrelated generators unserialized.
    spirv_codegen_cache: Mutex<HashMap<SpirvCodegenKey, Result<Vec<u32>, String>>>,
    /// Per-module GLSL codegen results. Values include memoized `Err`s because
    /// failures are deterministic for a module and key, so repeated failing
    /// pipeline creation must not re-cross Tint. Entries live as long as the
    /// module and are bounded by distinct pipeline configs ever created from
    /// it; if profiles demand a cap later, clear-all-at-N is the intended
    /// escape hatch. The lock is held across Tint codegen to dedup concurrent
    /// identical compiles; shared-program codegen is documented concurrent-safe,
    /// so this is policy rather than soundness. The only nested cache call is
    /// `resource_bindings_for_entry`, which uses a different mutex, and
    /// per-generator mutexes keep unrelated generators unserialized.
    #[cfg(feature = "gles")]
    glsl_codegen_cache: Mutex<HashMap<GlslCodegenKey, Result<GeneratedGlsl, String>>>,
    /// Per-module workgroup-size resolve results. Values include memoized
    /// `Err`s because failures are deterministic for a module and key, so
    /// repeated failing pipeline creation must not re-cross Tint. Entries live
    /// as long as the module and are bounded by distinct pipeline configs ever
    /// created from it; if profiles demand a cap later, clear-all-at-N is the
    /// intended escape hatch. The lock is held across Tint resolve to dedup
    /// concurrent identical resolves; shared-program codegen/resolve is
    /// documented concurrent-safe, so this is policy rather than soundness.
    /// The miss branch nests `overrides()` / `compute_workgroup_size()` into
    /// `raw_entry_points()`, which is a `OnceLock`, and per-generator mutexes
    /// keep unrelated generators unserialized.
    workgroup_resolve_cache:
        Mutex<HashMap<WorkgroupResolveKey, Result<ReflectedWorkgroupSize, String>>>,
    /// Counts actual Tint codegen/resolve executions, excluding cache hits.
    codegen_misses: AtomicUsize,
    /// Memoized [`Self::fragment_builtins`] backing data (all fragment
    /// entry points), computed once.
    fragment_builtins_cache: OnceLock<Vec<ReflectedFragmentBuiltins>>,
    /// Memoized [`Self::overrides`] result.
    overrides_cache: OnceLock<Vec<ReflectedOverride>>,
}

// `ReflectedModule` is `Send + Sync` via ordinary auto-trait derivation, with
// no `unsafe impl` needed here (refactor finding F3,
// `specs/tracking/tint-integration-refactor.md`): `yawgpu_tint::Program`
// itself now carries `unsafe impl Send`/`Sync` (see its SAFETY comment in
// `yawgpu-tint/src/lib.rs`, which cites the Tint/Dawn source evidence this
// was previously asserted here without), and every other field is ordinary
// owned Rust data (`OnceLock`/`Mutex` around `String`/`Vec`/`HashMap`) that
// is already `Send + Sync` on its own. The `Mutex` around each cache still
// serializes concurrent *reflection* calls into `program` from the Rust
// side, matching the shim-side `reflection_mutex` (same finding) — that
// mutex is about avoiding redundant `Inspector` rebuilds (F5), not filling a
// soundness gap.

/// Returns parse and validate wgsl.
///
/// Test-only convenience wrapper around [`parse_and_validate_wgsl_gated`] with
/// every optional-feature gate enabled; production code always calls the
/// gated form directly so it can pass the device's actual feature set.
#[cfg(test)]
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
    )
    .map_err(|e| e.to_string())?;
    let warnings = program
        .diagnostics()
        .map_err(|e| e.to_string())?
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
    Ok(ReflectedModule {
        program,
        warnings,
        raw_entry_points: OnceLock::new(),
        entry_points_cache: OnceLock::new(),
        entry_point_io_cache: Mutex::new(HashMap::new()),
        resource_bindings_cache: Mutex::new(HashMap::new()),
        immediate_data_size_cache: Mutex::new(HashMap::new()),
        immediate_data_used_slots_cache: Mutex::new(HashMap::new()),
        msl_codegen_cache: Mutex::new(HashMap::new()),
        spirv_codegen_cache: Mutex::new(HashMap::new()),
        #[cfg(feature = "gles")]
        glsl_codegen_cache: Mutex::new(HashMap::new()),
        workgroup_resolve_cache: Mutex::new(HashMap::new()),
        codegen_misses: AtomicUsize::new(0),
        fragment_builtins_cache: OnceLock::new(),
        overrides_cache: OnceLock::new(),
    })
}

impl ReflectedModule {
    /// Counts actual Tint codegen/resolve executions, excluding cache hits.
    #[allow(dead_code)]
    pub(crate) fn codegen_miss_count(&self) -> usize {
        self.codegen_misses.load(Ordering::Relaxed)
    }

    /// Generates spirv for the validated shader module.
    ///
    /// `vulkan_memory_model` enables Tint's SPV_KHR_vulkan_memory_model output
    /// when the Vulkan backend enabled `VK_KHR_vulkan_memory_model` /
    /// `vulkanMemoryModel`. SPIR-V robustness stays enabled.
    ///
    /// `multisampled_input_attachment` makes Tint emit multisampled
    /// `SubpassData` input attachments (the 2-arg `inputAttachmentLoad(ia,
    /// sample_index)` overload) so per-sample MSAA subpass input works.
    ///
    /// `user_immediate_size` is the owning pipeline layout's reserved
    /// user-immediate byte budget (Block 94, 0..=64; see
    /// [`Self::generate_msl`]). It only affects output when
    /// `polyfill_pixel_center` is `Some`: the internal depth-range
    /// immediates land at push-constant byte offsets
    /// `{user_immediate_size, user_immediate_size + 4}`, directly after
    /// the user region.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_spirv(
        &self,
        entry_name: &str,
        _stage: ShaderStage,
        pipeline_constants: &PipelineConstants,
        vulkan_memory_model: bool,
        framebuffer_fetch_descriptor_set: u32,
        multisampled_input_attachment: bool,
        polyfill_pixel_center: Option<u32>,
        user_immediate_size: u32,
    ) -> Result<Vec<u32>, String> {
        let key = SpirvCodegenKey {
            entry_point: entry_name.to_owned(),
            constants: CanonicalConstants::from(pipeline_constants),
            vulkan_memory_model,
            framebuffer_fetch_descriptor_set,
            multisampled_input_attachment,
            polyfill_pixel_center,
            user_immediate_size,
        };
        let mut cache = self
            .spirv_codegen_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }
        self.codegen_misses.fetch_add(1, Ordering::Relaxed);
        let generated = self
            .program
            .generate_spirv(
                entry_name,
                &yawgpu_tint::Bindings::default(),
                &override_values(pipeline_constants),
                true,
                vulkan_memory_model,
                framebuffer_fetch_descriptor_set,
                multisampled_input_attachment,
                polyfill_pixel_center,
                user_immediate_size,
            )
            .map_err(|e| e.to_string());
        cache.insert(key, generated.clone());
        generated
    }

    /// Generates GLSL ES for the validated shader module.
    ///
    /// Vertex stages request Tint's `first_instance_offset` (offset `0`,
    /// GLES has no other internal immediate sharing that struct) so
    /// `@builtin(instance_index)` is offset by the WebGPU `firstInstance`
    /// draw parameter -- see `yawgpu_tint::Program::generate_glsl` and
    /// `tint_shim.h`'s `yawgpu_tint_generate_glsl` docs (F2,
    /// specs/tracking/tint-integration-refactor.md slice R6). Non-vertex
    /// stages never read `instance_index`, so they skip it.
    ///
    /// Buffer bindings get an explicit identity remap (see
    /// `tint_bindings_for_glsl`) so the GLSL `layout(binding = N)` always
    /// matches the raw WGSL binding number that `yawgpu-hal`'s GLES backend
    /// binds with `glBindBufferRange` -- Tint's default `GenerateBindings`
    /// only coincides with that when bindings happen to already be dense
    /// and sequential from 0.
    #[cfg(feature = "gles")]
    pub(crate) fn generate_glsl(
        &self,
        entry_name: &str,
        stage: ShaderStage,
        pipeline_constants: &PipelineConstants,
    ) -> Result<GeneratedGlsl, String> {
        let key = GlslCodegenKey {
            entry_point: entry_name.to_owned(),
            stage,
            constants: CanonicalConstants::from(pipeline_constants),
        };
        let mut cache = self
            .glsl_codegen_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }
        self.codegen_misses.fetch_add(1, Ordering::Relaxed);
        let generated = (|| {
            let first_instance_offset = matches!(stage, ShaderStage::Vertex).then_some(0);
            let bindings = tint_bindings_for_glsl(&self.resource_bindings_for_entry(entry_name)?);
            let source = self
                .program
                .generate_glsl(
                    entry_name,
                    &bindings,
                    &override_values(pipeline_constants),
                    first_instance_offset,
                )
                .map_err(|e| e.to_string())?;
            Ok(GeneratedGlsl {
                source,
                entry_point: entry_name.to_owned(),
            })
        })();
        cache.insert(key, generated.clone());
        generated
    }

    /// Generates msl for the validated shader module.
    ///
    /// `user_immediate_size` is the owning pipeline layout's reserved
    /// user-immediate byte budget (Block 94, `layout.immediate_size`,
    /// 0..=64); it sets where any pipeline-internal immediates would be
    /// appended after the user prefix (compute pipelines have none today,
    /// but the offset must still be layout-consistent, see
    /// `yawgpu_tint::Program::generate_msl`'s doc comment).
    pub(crate) fn generate_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        pipeline_constants: &PipelineConstants,
        user_immediate_size: u32,
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
            user_immediate_size,
        )
    }

    /// Generates render vertex MSL for a validated shader module.
    ///
    /// `user_immediate_size` -- see [`Self::generate_msl`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_render_vertex_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        vertex_buffers: &[MslVertexBufferBinding],
        force_point_size: bool,
        pipeline_constants: &PipelineConstants,
        user_immediate_size: u32,
    ) -> Result<GeneratedMsl, String> {
        // `force_point_size` makes Tint emit `[[point_size]] = 1.0` (point-list
        // topology requires it on Metal).
        self.generate_stage_msl(
            entry_name,
            binding_map,
            pipeline_constants,
            &[],
            vertex_buffers,
            false,
            force_point_size,
            0xFFFF_FFFF,
            user_immediate_size,
        )
    }

    /// Generates render fragment MSL for a validated shader module.
    ///
    /// `user_immediate_size` -- see [`Self::generate_msl`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_render_fragment_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        subpass_color_slots: &[((u32, u32), u32)],
        pipeline_constants: &PipelineConstants,
        sample_mask: u32,
        user_immediate_size: u32,
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
            user_immediate_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_stage_msl(
        &self,
        entry_name: &str,
        binding_map: &MslBindingMap,
        pipeline_constants: &PipelineConstants,
        subpass_color_slots: &[((u32, u32), u32)],
        vertex_buffers: &[MslVertexBufferBinding],
        disable_robustness: bool,
        emit_vertex_point_size: bool,
        fixed_sample_mask: u32,
        user_immediate_size: u32,
    ) -> Result<GeneratedMsl, String> {
        let key = MslCodegenKey {
            entry_point: entry_name.to_owned(),
            binding_map: binding_map.clone(),
            constants: CanonicalConstants::from(pipeline_constants),
            subpass_color_slots: subpass_color_slots.to_vec(),
            vertex_buffers: vertex_buffers.to_vec(),
            disable_robustness,
            emit_vertex_point_size,
            fixed_sample_mask,
            user_immediate_size,
        };
        let mut cache = self
            .msl_codegen_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }
        self.codegen_misses.fetch_add(1, Ordering::Relaxed);
        let generated = (|| {
            let tint_vertex_buffers = tint_vertex_buffers(vertex_buffers)?;
            let bindings = tint_bindings_for_msl(
                binding_map,
                &self.resource_bindings_for_entry(entry_name)?,
                subpass_color_slots,
            )?;
            let binding_buffer_sizes_slot = msl_buffer_sizes_slot(binding_map)?;
            let buffer_sizes_slot = if tint_vertex_buffers.is_empty() {
                binding_buffer_sizes_slot
            } else {
                let max_vertex_metal_index = tint_vertex_buffers
                    .iter()
                    .map(|buffer| buffer.metal_index)
                    .max()
                    .unwrap_or(0);
                binding_buffer_sizes_slot.max(max_vertex_metal_index.saturating_add(1))
            };
            if buffer_sizes_slot > u32::from(u8::MAX) {
                return Err("MSL generated buffer slot exceeds the supported slot range".to_owned());
            }
            let output = self
                .program
                .generate_msl(
                    entry_name,
                    &bindings,
                    &override_values(pipeline_constants),
                    buffer_sizes_slot,
                    // The wrapper takes `robust` (robustness ENABLED), which is the
                    // negation of this fn's `disable_robustness`.
                    !disable_robustness,
                    emit_vertex_point_size,
                    &tint_vertex_buffers,
                    fixed_sample_mask,
                    user_immediate_size,
                )
                .map_err(|e| e.to_string())?;
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
                buffer_sizes_slot: (!buffer_size_bindings.is_empty()
                    || !tint_vertex_buffers.is_empty())
                .then_some(buffer_sizes_slot),
                buffer_size_bindings,
                frag_depth_clamp_slot: output.frag_depth_clamp_slot,
                immediate_slot: output.immediate_slot,
                workgroup_memory_sizes: output
                    .workgroup_allocations
                    .iter()
                    .map(|&size| size.div_ceil(16) * 16)
                    .collect(),
            })
        })();
        cache.insert(key, generated.clone());
        generated
    }

    /// Returns the raw Tint entry-point reflection, memoized on first access.
    ///
    /// `unwrap_or_default()` (m4, code review of the Tint-integration
    /// refactor): `self.program.entry_points()` can only return `Err` in two
    /// cases, neither reachable here. (1) `TintError::Unavailable`, when
    /// this build does not link Tint (`HAVE_TINT` false) -- but then
    /// `yawgpu_tint::Program::parse` itself already returns that same error
    /// first, in `parse_and_validate_wgsl_gated` above, so a `ReflectedModule`
    /// is never constructed for that build in the first place. (2)
    /// `TintError::Reflection`, from the FFI accessor
    /// `yawgpu_tint_entry_point_get` returning `false` -- but that shim
    /// function (`tint_shim.cpp`) only ever reads out of the
    /// `YawgpuTintProgram::entry_points` vector, which is populated once,
    /// unconditionally, by `yawgpu_tint_program_create` at parse time (no
    /// `Inspector` re-construction, no scope for a fresh failure), and the
    /// Rust-side loop in [`yawgpu_tint::Program::entry_points`] only ever
    /// indexes `0..count` where `count` came from that same vector's
    /// `size()` -- so the bounds check inside the shim accessor can never
    /// trip. A widened error surface (`resource_bindings_for_entry`
    /// reporting the underlying reflection error instead of "entry point not
    /// found") would therefore never actually manifest; propagating a
    /// `Result` through `raw_entry_points` and every one of its ~10 call
    /// sites in this file was judged not worth the churn for an
    /// unreachable path. If this ever proves reachable (e.g. a future Tint
    /// upgrade changes the shim's failure modes), that is a bug to fix at
    /// the shim/FFI layer, not a signal to thread `Result` through this
    /// accessor.
    fn raw_entry_points(&self) -> &[yawgpu_tint::EntryPoint] {
        self.raw_entry_points
            .get_or_init(|| self.program.entry_points().unwrap_or_default())
    }

    /// Returns entry points reflected by the validated shader module.
    pub(crate) fn entry_points(&self) -> Vec<ReflectedEntryPoint> {
        self.entry_points_cache
            .get_or_init(|| {
                self.raw_entry_points()
                    .iter()
                    .map(|entry| ReflectedEntryPoint {
                        name: entry.name.clone(),
                        stage: shader_stage(entry.stage),
                    })
                    .collect()
            })
            .clone()
    }

    /// Returns compute workgroup size reflected by the validated shader
    /// module, or `None` if `entry_point` is not a compute entry point, or
    /// its `@workgroup_size` is not fully literal.
    ///
    /// This is the literal-size fast path: Tint's Inspector only populates
    /// `EntryPoint::workgroup_size` when every dimension resolved to a
    /// constant during semantic analysis (`sem::Function::WorkgroupSize()`
    /// returns `None` for a dimension that is any override-expression, even
    /// one with a default), so a `Some` here means the size has zero
    /// override dependence and needs no IR lowering at all — just the
    /// already-memoized [`Self::raw_entry_points`]. The private helper
    /// backing [`Self::resolved_compute_workgroup_size`] below (finding F6 in
    /// `specs/tracking/tint-integration-refactor.md`), taken only when the
    /// module declares no overrides at all — see the caller for why.
    fn compute_workgroup_size(
        &self,
        entry_point: &str,
    ) -> Result<Option<ReflectedWorkgroupSize>, String> {
        let Some(literal_size) = self
            .raw_entry_points()
            .iter()
            .find(|entry| {
                entry.name == entry_point && entry.stage == yawgpu_tint::PipelineStage::Compute
            })
            .and_then(|entry| entry.workgroup_size)
        else {
            return Ok(None);
        };

        Ok(Some(ReflectedWorkgroupSize {
            entry_point: entry_point.to_owned(),
            literal_size,
            workgroup_storage_size: self
                .program
                .workgroup_storage_size(&[])
                .map_err(|e| e.to_string())?,
        }))
    }

    /// Returns compute workgroup size after resolving pipeline constants.
    ///
    /// Short-circuits to the literal-size fast path
    /// ([`Self::compute_workgroup_size`]) only when the module declares no
    /// overrides at all; any override-bearing module goes through
    /// [`yawgpu_tint::Program::resolved_workgroup_size`], a direct IR query
    /// (lower + entry-scoped `SubstituteOverrides` + read the resolved size)
    /// that replaced generating a full SPIR-V module just to grep
    /// `OpExecutionMode LocalSize` back out of it (finding F6 in
    /// `specs/tracking/tint-integration-refactor.md`).
    ///
    /// The "no overrides declared" gate is load-bearing for WebGPU
    /// pipeline-creation validation, not just storage-size accuracy: an
    /// override whose initializer fails const-eval (e.g. `override bad: u32 =
    /// 1u / zero;` with `zero` defaulting to 0) must fail
    /// `createComputePipeline` for an entry point that references it as a
    /// *captured* validation error at resolve time — exactly where the old
    /// SPIR-V round trip surfaced it — not later at HAL shader compilation,
    /// where it would land in the device error sink as an *uncaptured* error
    /// (CTS `compute_pipeline:overrides,entry_point,validation_error`).
    /// Because the shim query substitutes entry-scoped (`SingleEntryPoint`
    /// before `SubstituteOverrides`, like the generate paths), a sibling
    /// entry point that does not reference the bad override still resolves
    /// successfully.
    pub(crate) fn resolved_compute_workgroup_size(
        &self,
        entry_point: &str,
        pipeline_constants: &PipelineConstants,
    ) -> Result<ReflectedWorkgroupSize, String> {
        let key = WorkgroupResolveKey {
            entry_point: entry_point.to_owned(),
            constants: CanonicalConstants::from(pipeline_constants),
        };
        let mut cache = self
            .workgroup_resolve_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }
        self.codegen_misses.fetch_add(1, Ordering::Relaxed);
        let resolved = (|| {
            if self.overrides().is_empty() {
                if let Some(reflected) = self.compute_workgroup_size(entry_point)? {
                    return Ok(reflected);
                }
            }
            let overrides = override_values(pipeline_constants);
            let literal_size = self
                .program
                .resolved_workgroup_size(entry_point, &overrides)
                .map_err(|e| e.to_string())?;
            Ok(ReflectedWorkgroupSize {
                entry_point: entry_point.to_owned(),
                literal_size,
                workgroup_storage_size: self
                    .program
                    .workgroup_storage_size(&overrides)
                    .map_err(|e| e.to_string())?,
            })
        })();
        cache.insert(key, resolved.clone());
        resolved
    }

    /// Returns entry point IO reflected by the validated shader module for a
    /// single `entry_point`, or `None` if no entry point with that name
    /// exists. Memoized per entry-point name: every render-pipeline
    /// validator that needs IO reflection (vertex inputs, inter-stage
    /// outputs/inputs, inter-stage builtin count, fragment outputs) looks up
    /// exactly one entry point, so this only ever reflects each entry point
    /// once regardless of how many validators ask for it.
    pub(crate) fn entry_point_io(&self, entry_point: &str) -> Option<ReflectedEntryPointIo> {
        // No panics in library code (CLAUDE.md): a poisoned lock here only
        // means some *other* thread panicked mid-insert; the cache map
        // itself is never left semantically inconsistent (inserts happen
        // only after `build_entry_point_io`/`build_resource_bindings_for_entry`
        // already returned a complete value, see below), so recovering the
        // inner guard and carrying on is safe.
        let mut cache = self
            .entry_point_io_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = cache.get(entry_point) {
            return cached.clone();
        }
        let computed = self.build_entry_point_io(entry_point);
        cache.insert(entry_point.to_owned(), computed.clone());
        computed
    }

    fn build_entry_point_io(&self, entry_point: &str) -> Option<ReflectedEntryPointIo> {
        let entry = self
            .raw_entry_points()
            .iter()
            .find(|entry| entry.name == entry_point)?;
        let is_compute = entry.stage == yawgpu_tint::PipelineStage::Compute;
        Some(ReflectedEntryPointIo {
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
            input_inter_stage_builtins: input_inter_stage_builtin_count(entry),
            entry_point: entry.name.clone(),
        })
    }

    /// Returns resource bindings for entry reflected by the validated shader module.
    ///
    /// Memoized per entry-point name (including the "entry point not found"
    /// error), so a repeated call for the same entry point never re-crosses
    /// the FFI boundary.
    pub(crate) fn resource_bindings_for_entry(
        &self,
        entry_point: &str,
    ) -> Result<Vec<ReflectedResourceBinding>, String> {
        // See the matching comment on `entry_point_io`'s lock above: a
        // poisoned lock does not imply an inconsistent cache, so recover
        // rather than panic (no panics in library code, CLAUDE.md).
        let mut cache = self
            .resource_bindings_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = cache.get(entry_point) {
            return cached.clone();
        }
        let computed = self.build_resource_bindings_for_entry(entry_point);
        cache.insert(entry_point.to_owned(), computed.clone());
        computed
    }

    fn build_resource_bindings_for_entry(
        &self,
        entry_point: &str,
    ) -> Result<Vec<ReflectedResourceBinding>, String> {
        // Tint's `GetResourceBindings` silently returns an empty vector for an
        // unknown entry point rather than erroring, so this existence check is
        // load-bearing for the "entry point not found" error surface below —
        // it must run before, not after, the FFI call. It reads the raw
        // entry-point cache (already filled at parse time on the shim side,
        // see `YawgpuTintProgram::entry_points`) rather than re-crossing the
        // FFI boundary with a fresh `entry_points()` call.
        let entry_exists = self
            .raw_entry_points()
            .iter()
            .any(|entry| entry.name == entry_point);
        if !entry_exists {
            return Err("shader entry point was not found for resource reflection".to_owned());
        }

        self.program
            .resource_bindings(entry_point)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(reflected_resource_binding)
            .collect()
    }

    /// Returns `entry_point`'s immediate data size in bytes -- the total
    /// byte size of all `var<immediate>` (WGSL `immediate_address_space`
    /// language feature) globals it statically accesses. `0` means the
    /// entry point declares/uses no immediates (including a module that
    /// merely declares one unused by the entry point). Errors if
    /// `entry_point` does not name an entry point in the module.
    ///
    /// Memoized per entry-point name (including the "entry point not found"
    /// error), so a repeated call for the same entry point never re-crosses
    /// the FFI boundary.
    pub(crate) fn immediate_data_size(&self, entry_point: &str) -> Result<u32, String> {
        // See the matching comment on `entry_point_io`'s lock above: a
        // poisoned lock does not imply an inconsistent cache, so recover
        // rather than panic (no panics in library code, CLAUDE.md).
        let mut cache = self
            .immediate_data_size_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = cache.get(entry_point) {
            return cached.clone();
        }
        let computed = self
            .program
            .immediate_data_size(entry_point)
            .map_err(|e| e.to_string());
        cache.insert(entry_point.to_owned(), computed.clone());
        computed
    }

    /// Returns `entry_point`'s required immediate-data slots as a bitmask.
    ///
    /// Bit N corresponds to bytes `[4*N, 4*N+4)` of the user immediate block.
    /// This is Tint's `Inspector::GetImmediateBlockInfo` result: non-padding
    /// words of statically referenced `var<immediate>` variables are set;
    /// struct/matrix padding words are not. Errors if `entry_point` does not
    /// name an entry point in the module.
    pub(crate) fn immediate_data_used_slots(&self, entry_point: &str) -> Result<u64, String> {
        // See the matching comment on `entry_point_io`'s lock above: a
        // poisoned lock does not imply an inconsistent cache, so recover
        // rather than panic (no panics in library code, CLAUDE.md).
        let mut cache = self
            .immediate_data_used_slots_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(cached) = cache.get(entry_point) {
            return cached.clone();
        }
        let computed = self
            .program
            .immediate_data_used_slots(entry_point)
            .map_err(|e| e.to_string());
        cache.insert(entry_point.to_owned(), computed.clone());
        computed
    }

    /// Returns fragment builtins reflected by the validated shader module for
    /// a single `entry_point`, or `None` if no fragment entry point with that
    /// name exists. Both call sites (`frag_depth` / `sample_mask` output
    /// checks) look up exactly one entry point.
    pub(crate) fn fragment_builtins(&self, entry_point: &str) -> Option<ReflectedFragmentBuiltins> {
        self.fragment_builtins_cache
            .get_or_init(|| {
                self.raw_entry_points()
                    .iter()
                    .filter(|entry| entry.stage == yawgpu_tint::PipelineStage::Fragment)
                    .map(|entry| ReflectedFragmentBuiltins {
                        entry_point: entry.name.clone(),
                        frag_depth: entry.frag_depth_used,
                        sample_mask: entry.sample_mask_used,
                    })
                    .collect::<Vec<_>>()
            })
            .iter()
            .find(|builtins| builtins.entry_point == entry_point)
            .cloned()
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

    /// Returns whether the given fragment entry point needs the Vulkan
    /// pixel-center polyfill for `@builtin(position)`.
    ///
    /// WebGPU requires `@builtin(position)` in a fragment shader to always be
    /// the pixel-center (fragment) coordinate, never a sample position. Under
    /// Vulkan sample-rate shading the SPIR-V `FragCoord` builtin instead
    /// reflects the covered sample's location, so it must be reconstructed.
    /// This mirrors Dawn's `RenderPipeline::NeedsPixelCenterPolyfill`
    /// (`UseSampleRateShading() && UsesFragPosition()`): the polyfill is needed
    /// only when the fragment reads `@builtin(position)` AND sample-rate shading
    /// is active — i.e. `sample_count > 1` and the fragment uses a
    /// `@interpolate(_, sample)` input, `@builtin(sample_index)`, or framebuffer
    /// fetch (all of which force per-sample fragment invocation).
    pub(crate) fn fragment_needs_pixel_center_polyfill(
        &self,
        entry_point: &str,
        sample_count: u32,
    ) -> bool {
        if sample_count <= 1 {
            return false;
        }
        let Some(entry) = self.raw_entry_points().iter().find(|entry| {
            entry.name == entry_point && entry.stage == yawgpu_tint::PipelineStage::Fragment
        }) else {
            return false;
        };
        if !entry.frag_position_used {
            return false;
        }
        let inputs = self
            .program
            .entry_point_inputs(entry_point)
            .unwrap_or_default();
        let uses_sample_interpolant = inputs.iter().any(|variable| {
            variable.interpolation_sampling == yawgpu_tint::InterpolationSampling::Sample
        });
        let uses_framebuffer_fetch = inputs.iter().any(|variable| variable.color.is_some());
        uses_sample_interpolant || entry.sample_index_used || uses_framebuffer_fetch
    }

    /// Returns the lowest inter-stage `@location` index not used by the given
    /// vertex entry point's outputs.
    ///
    /// The pixel-center polyfill adds a `center_pos` inter-stage varying carried
    /// from the vertex stage to the fragment stage, so it needs a free location.
    /// Fragment inputs are a subset of vertex outputs, so a location free in the
    /// vertex outputs is also free in the fragment inputs (matching Dawn, which
    /// scans the vertex stage's used inter-stage variables). The search is
    /// bounded by the output count, so a free index in `0..=len` always exists.
    pub(crate) fn free_inter_stage_location(&self, vertex_entry_point: &str) -> u32 {
        let used = self
            .program
            .entry_point_outputs(vertex_entry_point)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|variable| variable.location)
            .collect::<HashSet<u32>>();
        (0..=used.len() as u32)
            .find(|location| !used.contains(location))
            .unwrap_or(used.len() as u32)
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
        self.raw_entry_points()
            .iter()
            .find(|entry| {
                entry.stage == yawgpu_tint::PipelineStage::Vertex && entry.name == entry_point
            })
            .and_then(|entry| entry.clip_distances_size)
            .unwrap_or(0)
    }

    /// Returns overrides reflected by the validated shader module, memoized once.
    pub(crate) fn overrides(&self) -> Vec<ReflectedOverride> {
        self.overrides_cache
            .get_or_init(|| {
                self.program
                    .overrides()
                    .unwrap_or_default()
                    .into_iter()
                    .map(reflected_override)
                    .collect()
            })
            .clone()
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

/// Returns identity Tint binding remaps for the GLES GLSL writer.
///
/// GLES only supports plain uniform/storage buffers at bind group 0
/// (`yawgpu-hal/src/gles/queue.rs` rejects textures/samplers and non-zero
/// groups at runtime). Left as `Bindings::default()`, Tint's own
/// `GenerateBindings` auto-numbers GLSL `layout(binding = N)` sequentially
/// in shader declaration order, which only coincides with the WGSL
/// `@binding` number when bindings already happen to be dense and
/// sequential starting at 0 -- e.g. `@binding(0)` + `@binding(3)` gets
/// renumbered to `layout(binding = 0)` + `layout(binding = 1)`. Since
/// `yawgpu-hal`'s GLES backend always binds buffers with
/// `glBindBufferRange` at the raw WGSL binding number
/// (`HalDescriptorBinding::binding`), the GLSL text and the HAL's runtime
/// bind calls would silently disagree for any non-sequential binding
/// layout. An explicit group/binding -> same group/binding remap for every
/// buffer resource pins Tint's output to the WGSL numbers directly (F2,
/// specs/tracking/tint-integration-refactor.md slice R6). Non-buffer
/// resources are skipped: GLES rejects them before a generated shader
/// would ever run.
#[cfg(feature = "gles")]
fn tint_bindings_for_glsl(resource_bindings: &[ReflectedResourceBinding]) -> yawgpu_tint::Bindings {
    let mut bindings = yawgpu_tint::Bindings::default();
    for binding in resource_bindings {
        let remap = yawgpu_tint::BindingRemap {
            group: binding.group,
            binding: binding.binding,
            dst_group: binding.group,
            dst_binding: binding.binding,
        };
        match binding.kind {
            ReflectedResourceBindingKind::Buffer(ReflectedBufferType::Uniform) => {
                bindings.uniform.push(remap);
            }
            ReflectedResourceBindingKind::Buffer(
                ReflectedBufferType::Storage | ReflectedBufferType::ReadOnlyStorage,
            ) => {
                bindings.storage.push(remap);
            }
            _ => {}
        }
    }
    bindings
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

fn reflected_resource_binding(
    binding: yawgpu_tint::ResourceBinding,
) -> Result<ReflectedResourceBinding, String> {
    Ok(ReflectedResourceBinding {
        group: binding.group,
        binding: binding.binding,
        kind: resource_binding_kind(&binding)?,
        min_binding_size: binding.size,
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

    fn empty_msl_binding_map() -> MslBindingMap {
        MslBindingMap {
            resources: Vec::new(),
        }
    }

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
    fn generate_msl_memoizes_identical_requests() {
        let module = parse_and_validate_wgsl(
            r#"
@compute @workgroup_size(1)
fn cs() {}
"#,
        )
        .unwrap();
        let constants = PipelineConstants::default();
        let binding_map = empty_msl_binding_map();

        let first = module
            .generate_msl("cs", &binding_map, &constants, 0)
            .unwrap();
        let second = module
            .generate_msl("cs", &binding_map, &constants, 0)
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(module.codegen_miss_count(), 1);

        let _ = module
            .generate_msl("cs", &binding_map, &constants, 4)
            .unwrap();
        assert_eq!(module.codegen_miss_count(), 2);
    }

    #[test]
    fn generate_msl_key_is_constant_order_insensitive() {
        let module = parse_and_validate_wgsl(
            r#"
override x: u32 = 1;
override y: u32 = 2;

@compute @workgroup_size(1)
fn cs() {
  _ = x + y;
}
"#,
        )
        .unwrap();
        let binding_map = empty_msl_binding_map();
        let xy = PipelineConstants::from_iter([("x".to_owned(), 3.0), ("y".to_owned(), 4.0)]);
        let yx = PipelineConstants::from_iter([("y".to_owned(), 4.0), ("x".to_owned(), 3.0)]);

        let first = module.generate_msl("cs", &binding_map, &xy, 0).unwrap();
        let second = module.generate_msl("cs", &binding_map, &yx, 0).unwrap();
        assert_eq!(first, second);
        assert_eq!(module.codegen_miss_count(), 1);
    }

    #[test]
    fn generate_msl_distinguishes_constant_values() {
        let module = parse_and_validate_wgsl(
            r#"
override x: u32 = 1;

@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn cs() {
  data[0] = x;
}
"#,
        )
        .unwrap();
        let binding_map = MslBindingMap {
            resources: vec![MslResourceBinding {
                group: 0,
                binding: 0,
                metal_index: 0,
                ext_params_buffer_slot: None,
                kind: MslResourceBindingKind::Buffer,
            }],
        };
        let one = PipelineConstants::from_iter([("x".to_owned(), 1.0)]);
        let two = PipelineConstants::from_iter([("x".to_owned(), 2.0)]);

        let first = module.generate_msl("cs", &binding_map, &one, 0).unwrap();
        let second = module.generate_msl("cs", &binding_map, &two, 0).unwrap();
        assert_ne!(first.source, second.source);
        assert_eq!(module.codegen_miss_count(), 2);
    }

    #[test]
    fn generate_msl_memoizes_error_results() {
        let module = parse_and_validate_wgsl(
            r#"
@compute @workgroup_size(1)
fn cs() {}
"#,
        )
        .unwrap();
        let binding_map = empty_msl_binding_map();
        let constants = PipelineConstants::default();

        let first = module
            .generate_msl("missing", &binding_map, &constants, 0)
            .unwrap_err();
        let second = module
            .generate_msl("missing", &binding_map, &constants, 0)
            .unwrap_err();
        assert_eq!(first, second);
        assert_eq!(module.codegen_miss_count(), 1);
    }

    #[test]
    fn generate_render_vertex_msl_memoizes_with_vertex_buffers() {
        let module = parse_and_validate_wgsl(
            r#"
@vertex
fn vs(@location(0) p: vec4<f32>) -> @builtin(position) vec4<f32> {
  return p;
}
"#,
        )
        .unwrap();
        let constants = PipelineConstants::default();
        let binding_map = empty_msl_binding_map();
        let vertex_buffers = vec![MslVertexBufferBinding {
            slot: 0,
            metal_index: 2,
            array_stride: 16,
            step_mode: MslVertexStepMode::Vertex,
            attributes: vec![MslVertexAttribute {
                shader_location: 0,
                offset: 0,
                format: MslVertexFormat::Float32x4,
            }],
        }];

        let first = module
            .generate_render_vertex_msl("vs", &binding_map, &vertex_buffers, false, &constants, 0)
            .unwrap();
        let second = module
            .generate_render_vertex_msl("vs", &binding_map, &vertex_buffers, false, &constants, 0)
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(module.codegen_miss_count(), 1);

        let mut changed = vertex_buffers.clone();
        changed[0].attributes[0].offset = 4;
        let _ = module
            .generate_render_vertex_msl("vs", &binding_map, &changed, false, &constants, 0)
            .unwrap();
        assert_eq!(module.codegen_miss_count(), 2);
    }

    #[test]
    fn generate_render_fragment_msl_distinguishes_sample_mask() {
        let module = parse_and_validate_wgsl(
            r#"
@fragment
fn fs() -> @builtin(sample_mask) u32 {
  return 0xffffffffu;
}
"#,
        )
        .unwrap();
        let constants = PipelineConstants::default();
        let binding_map = empty_msl_binding_map();

        let all = module
            .generate_render_fragment_msl("fs", &binding_map, &[], &constants, 0xFFFF_FFFF, 0)
            .unwrap();
        let narrow = module
            .generate_render_fragment_msl("fs", &binding_map, &[], &constants, 0xF, 0)
            .unwrap();
        assert_ne!(all.source, narrow.source);
        assert_eq!(module.codegen_miss_count(), 2);

        let repeat = module
            .generate_render_fragment_msl("fs", &binding_map, &[], &constants, 0xF, 0)
            .unwrap();
        assert_eq!(narrow, repeat);
        assert_eq!(module.codegen_miss_count(), 2);
    }

    #[test]
    fn generate_spirv_memoizes_identical_requests() {
        let module = parse_and_validate_wgsl(
            r#"
@compute @workgroup_size(1)
fn cs() {}
"#,
        )
        .unwrap();
        let constants = PipelineConstants::default();

        let first = module
            .generate_spirv(
                "cs",
                ShaderStage::Compute,
                &constants,
                false,
                0,
                false,
                None,
                0,
            )
            .unwrap();
        let second = module
            .generate_spirv(
                "cs",
                ShaderStage::Compute,
                &constants,
                false,
                0,
                false,
                None,
                0,
            )
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(module.codegen_miss_count(), 1);

        let _ = module
            .generate_spirv(
                "cs",
                ShaderStage::Compute,
                &constants,
                true,
                0,
                false,
                None,
                0,
            )
            .unwrap();
        assert_eq!(module.codegen_miss_count(), 2);
    }

    #[test]
    fn resolved_compute_workgroup_size_memoizes_per_constants() {
        let module = parse_and_validate_wgsl(
            r#"
override x: u32 = 4;

@compute @workgroup_size(x, 2, 1)
fn cs() {}
"#,
        )
        .unwrap();
        let eight = PipelineConstants::from_iter([("x".to_owned(), 8.0)]);
        let sixteen = PipelineConstants::from_iter([("x".to_owned(), 16.0)]);

        let first = module
            .resolved_compute_workgroup_size("cs", &eight)
            .unwrap();
        let second = module
            .resolved_compute_workgroup_size("cs", &eight)
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.literal_size, [8, 2, 1]);
        assert_eq!(module.codegen_miss_count(), 1);

        let changed = module
            .resolved_compute_workgroup_size("cs", &sixteen)
            .unwrap();
        assert_eq!(changed.literal_size, [16, 2, 1]);
        assert_eq!(module.codegen_miss_count(), 2);
    }

    #[test]
    fn resolved_compute_workgroup_size_memoizes_const_eval_errors() {
        let module = parse_and_validate_wgsl(
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
        )
        .unwrap();
        let constants = PipelineConstants::default();

        let first = module
            .resolved_compute_workgroup_size("main_pipe_error", &constants)
            .unwrap_err();
        let second = module
            .resolved_compute_workgroup_size("main_pipe_error", &constants)
            .unwrap_err();
        assert_eq!(first, second);
        assert_eq!(module.codegen_miss_count(), 1);

        let ok = module
            .resolved_compute_workgroup_size("main_ok", &constants)
            .unwrap();
        assert_eq!(ok.literal_size, [8, 4, 1]);
        assert_eq!(module.codegen_miss_count(), 2);
    }

    #[cfg(feature = "gles")]
    #[test]
    fn generate_glsl_memoizes_identical_requests() {
        let module = parse_and_validate_wgsl(
            r#"
@compute @workgroup_size(1)
fn cs() {}
"#,
        )
        .unwrap();
        let constants = PipelineConstants::default();

        let first = module
            .generate_glsl("cs", ShaderStage::Compute, &constants)
            .unwrap();
        let second = module
            .generate_glsl("cs", ShaderStage::Compute, &constants)
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(module.codegen_miss_count(), 1);

        let _ = module
            .generate_glsl("cs", ShaderStage::Vertex, &constants)
            .unwrap();
        assert_eq!(module.codegen_miss_count(), 2);
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
    fn resolves_workgroup_storage_size_from_overrides_with_literal_workgroup_size() {
        // @workgroup_size is fully literal, but the module declares an
        // override (driving the `var<workgroup>` array length), so this must
        // NOT take the literal fast path -- overrides have to be resolved
        // (entry-scoped) at resolve time. Regression guard for the bug the
        // old dead-code fast path had (it always resolved
        // `workgroup_storage_size` with an empty override set, independent
        // of what the caller actually passed).
        let module = parse_and_validate_wgsl(
            r#"
override n: u32 = 4;
var<workgroup> data: array<u32, n>;

@compute @workgroup_size(1)
fn cs() {
  data[0] = 1u;
}
"#,
        )
        .unwrap();

        let default_resolved = module
            .resolved_compute_workgroup_size("cs", &PipelineConstants::default())
            .unwrap();
        assert_eq!(default_resolved.literal_size, [1, 1, 1]);
        assert_eq!(default_resolved.workgroup_storage_size, 16);

        let constants = PipelineConstants::from_iter([("n".to_owned(), 8.0)]);
        let overridden = module
            .resolved_compute_workgroup_size("cs", &constants)
            .unwrap();
        assert_eq!(overridden.literal_size, [1, 1, 1]);
        assert_eq!(overridden.workgroup_storage_size, 32);
    }

    #[test]
    fn resolved_workgroup_size_surfaces_entry_scoped_override_const_eval_errors() {
        // Mirrors CTS
        // `compute_pipeline:overrides,entry_point,validation_error`: an
        // override whose initializer fails const-evaluation (1u / 0u after
        // substituting `cu`'s default) must fail
        // `resolved_compute_workgroup_size` -- and therefore
        // createComputePipeline, as a captured validation error -- for the
        // entry point that references it, while a sibling entry point that
        // does not reference it must still resolve. Both entry points have
        // fully literal @workgroup_size, so this also pins that the literal
        // fast path is NOT taken when the module declares overrides.
        let module = parse_and_validate_wgsl(
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
        )
        .unwrap();

        let err = module
            .resolved_compute_workgroup_size("main_pipe_error", &PipelineConstants::default())
            .unwrap_err();
        assert!(!err.is_empty());

        let ok = module
            .resolved_compute_workgroup_size("main_ok", &PipelineConstants::default())
            .unwrap();
        assert_eq!(ok.literal_size, [8, 4, 1]);
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

        // A non-existent entry point errors rather than silently returning
        // an empty binding list; the error is memoized the same as a hit
        // (edge case for the per-entry cache added in the
        // tint-integration-refactor R3 slice).
        let err = module
            .resource_bindings_for_entry("does_not_exist")
            .expect_err("unknown entry point should error");
        assert_eq!(
            err,
            "shader entry point was not found for resource reflection"
        );
        let err_again = module
            .resource_bindings_for_entry("does_not_exist")
            .expect_err("unknown entry point should error on a cached lookup too");
        assert_eq!(err_again, err);
    }

    #[test]
    fn reflects_immediate_data_size() {
        let module = parse_and_validate_wgsl(
            r#"
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
"#,
        )
        .unwrap();

        assert_eq!(module.immediate_data_size("uses_immediate").unwrap(), 16);
        assert_eq!(
            module.immediate_data_used_slots("uses_immediate").unwrap(),
            0b1111
        );
        assert_eq!(module.immediate_data_size("no_immediate").unwrap(), 0);
        assert_eq!(module.immediate_data_used_slots("no_immediate").unwrap(), 0);

        // A non-existent entry point errors rather than silently returning 0;
        // the error is memoized the same as a hit.
        let err = module
            .immediate_data_size("does_not_exist")
            .expect_err("unknown entry point should error");
        let err_again = module
            .immediate_data_size("does_not_exist")
            .expect_err("unknown entry point should error on a cached lookup too");
        assert_eq!(err_again, err);
        let err = module
            .immediate_data_used_slots("does_not_exist")
            .expect_err("unknown entry point should error");
        let err_again = module
            .immediate_data_used_slots("does_not_exist")
            .expect_err("unknown entry point should error on a cached lookup too");
        assert_eq!(err_again, err);
    }

    #[test]
    fn reflects_immediate_data_used_slots_excluding_padding() {
        let module = parse_and_validate_wgsl(
            r#"
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
"#,
        )
        .unwrap();

        assert_eq!(module.immediate_data_size("main").unwrap(), 32);
        assert_eq!(
            module.immediate_data_used_slots("main").unwrap(),
            0b0111_0001
        );
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
                0,
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
                0,
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

        let vs = module.entry_point_io("vs").unwrap();
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

        let fs = module.entry_point_io("fs").unwrap();
        assert_eq!(fs.inputs.len(), 2);
        assert!(fs.outputs.iter().any(|output| {
            output.location == 0
                && output.ty.scalar == ReflectedTypeScalarClass::Float
                && output.ty.components == 4
        }));
        assert_eq!(fs.input_inter_stage_builtins, 3);

        // Missing entry point: `None`, memoized the same as a hit (edge case
        // for the per-entry cache added in the tint-integration-refactor R3
        // slice).
        assert_eq!(module.entry_point_io("does_not_exist"), None);
        assert_eq!(module.entry_point_io("does_not_exist"), None);
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
            module.fragment_builtins("fs"),
            Some(ReflectedFragmentBuiltins {
                entry_point: "fs".to_owned(),
                frag_depth: true,
                sample_mask: false,
            })
        );
        // Repeated lookup hits the memoized cache and a non-existent entry
        // point returns `None` rather than erroring.
        assert_eq!(
            module.fragment_builtins("fs"),
            Some(ReflectedFragmentBuiltins {
                entry_point: "fs".to_owned(),
                frag_depth: true,
                sample_mask: false,
            })
        );
        assert_eq!(module.fragment_builtins("does_not_exist"), None);
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
                0,
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
                0,
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
                0,
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
            None,
            0,
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
                None,
                0,
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
                None,
                0,
            )
            .unwrap();
        assert_eq!(vertex.first().copied(), Some(0x0723_0203));
        assert_eq!(fragment.first().copied(), Some(0x0723_0203));
    }

    #[test]
    fn fragment_pixel_center_polyfill_decision_and_free_location() {
        let module = parse_and_validate_wgsl(
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

// Reads @builtin(position) with a sample-interpolated input -> needs polyfill.
@fragment
fn fs(v: VOut) -> @location(0) vec4<f32> {
  return vec4<f32>(v.pos.xy, v.uv);
}

// Reads @builtin(position) but only a center-interpolated input -> no
// sample-rate shading, so no polyfill.
@fragment
fn fs_center(@builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>)
    -> @location(0) vec4<f32> {
  return vec4<f32>(pos.xy, uv);
}

// Sample-rate shading but does not read @builtin(position) -> no polyfill.
@fragment
fn fs_no_pos(@location(0) @interpolate(perspective, sample) uv: vec2<f32>)
    -> @location(0) vec4<f32> {
  return vec4<f32>(uv, 0.0, 1.0);
}
"#,
        )
        .unwrap();

        // Position + sample interpolant + multisample -> polyfill needed.
        assert!(module.fragment_needs_pixel_center_polyfill("fs", 4));
        // Single-sampled: never per-sample, so no polyfill regardless.
        assert!(!module.fragment_needs_pixel_center_polyfill("fs", 1));
        // Position but no sample-rate trigger -> no polyfill.
        assert!(!module.fragment_needs_pixel_center_polyfill("fs_center", 4));
        // Sample-rate shading but no @builtin(position) read -> no polyfill.
        assert!(!module.fragment_needs_pixel_center_polyfill("fs_no_pos", 4));

        // Vertex outputs occupy location 0, so location 1 is the first free
        // inter-stage location for the center_pos varying.
        assert_eq!(module.free_inter_stage_location("vs"), 1);
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
                    None,
                    0,
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

    /// F2 (specs/tracking/tint-integration-refactor.md, slice R6): only the
    /// vertex stage requests Tint's `first_instance_offset`, since only
    /// vertex shaders read `@builtin(instance_index)`.
    #[cfg(feature = "gles")]
    #[test]
    fn generate_glsl_requests_first_instance_offset_for_vertex_stage_only() {
        let vertex_glsl = parse_and_validate_wgsl(
            r#"
@vertex
fn vs(@builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  return vec4f(f32(ii), 0.0, 0.0, 1.0);
}
"#,
        )
        .unwrap()
        .generate_glsl("vs", ShaderStage::Vertex, &PipelineConstants::default())
        .unwrap();
        // Tint's internal `tint_first_instance` symbol never reaches the
        // printed GLSL text (see yawgpu-tint's
        // `generate_glsl_first_instance_offset_only_applied_when_requested`)
        // -- the offset shows up as a `tint_immediates` array read instead.
        assert!(
            vertex_glsl.source.contains("tint_immediates"),
            "GLSL:\n{}",
            vertex_glsl.source
        );

        let fragment_glsl = parse_and_validate_wgsl(
            r#"
@fragment
fn fs() -> @location(0) vec4f {
  return vec4f(0.0);
}
"#,
        )
        .unwrap()
        .generate_glsl("fs", ShaderStage::Fragment, &PipelineConstants::default())
        .unwrap();
        assert!(
            !fragment_glsl.source.contains("tint_immediates"),
            "GLSL:\n{}",
            fragment_glsl.source
        );
    }

    /// F2 (specs/tracking/tint-integration-refactor.md, slice R6): pins
    /// `tint_bindings_for_glsl`'s identity remap end-to-end through
    /// `ReflectedModule::generate_glsl`. Without the remap, Tint's default
    /// `GenerateBindings` would renumber the non-sequential `@binding(3)`
    /// storage buffer to `layout(binding = 1)` (see yawgpu-tint's
    /// `generate_glsl_default_bindings_renumber_sequentially_not_identity`),
    /// which would desync from `yawgpu-hal`'s GLES backend -- it always
    /// binds buffers with `glBindBufferRange` at the raw WGSL binding
    /// number.
    #[cfg(feature = "gles")]
    #[test]
    fn generate_glsl_pins_non_sequential_wgsl_binding_numbers() {
        let generated = parse_and_validate_wgsl(
            r#"
struct Uniforms {
  scale: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(3) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn cs() {
  data[0] = u32(u.scale);
}
"#,
        )
        .unwrap()
        .generate_glsl("cs", ShaderStage::Compute, &PipelineConstants::default())
        .unwrap();
        assert!(
            generated
                .source
                .contains("layout(binding = 0, std140)\nuniform"),
            "GLSL:\n{}",
            generated.source
        );
        assert!(
            generated
                .source
                .contains("layout(binding = 3, std430)\nbuffer"),
            "GLSL:\n{}",
            generated.source
        );
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
                0,
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
                0,
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
