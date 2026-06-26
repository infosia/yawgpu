# Tint migration plan — replace naga with Dawn's Tint as the shader frontend

**Status: PLANNING (2026-06-26).** Supersedes the "stay on naga (A)" decision in
`specs/tracking/tint-migration-spike.md`. The spike de-risked feasibility
(standalone CMake build 73 s / ~6 MB, Rust↔C shim, Metal runs Tint MSL, iOS +
Android cross-build all proven). The blocker that tilted the prior decision to A
was the cost of re-homing the TBDR `subpass_input` extension onto Tint. **That
blocker is removed: the user has decided to delete all naga-fork vendor
extensions** (see "Decisions" below), so migration scope no longer includes any
extension re-home.

## Decisions (user, 2026-06-26)

1. **Vendor extensions: delete all.** Drop the TBDR `subpass_input`/`subpassLoad`
   path, `shader-passthrough` (raw SPIR-V/MSL module creation), and the
   external-texture honest-rejection. Tint retains `input_attachment` primitives,
   so a future re-home stays *possible* but is explicitly out of scope here.
2. **GLES (Tier 2): keep, via Tint's `glsl/writer`** (ES 3.1 default). The ANGLE
   path survives.
3. **Build: source-build Tint from `build.rs`.** Vendor Dawn (pinned rev) and
   drive CMake from `build.rs`. No prebuilt-artifact distribution.

## Why this is much smaller than the spike feared

Two facts established while planning (2026-06-26):

- **The HAL boundary is already naga-free.** Every `naga` reference in
  `yawgpu-hal` is a comment or a test name — no `naga::*` types cross into the
  HAL. The HAL consumes already-generated MSL/SPIR-V/GLSL strings/bytes via
  `HalShaderSource` plus yawgpu-core-owned binding structs. (The spike's worry
  that `metal/{device,pipeline,encode}.rs` build `naga::back::msl::*` directly is
  **outdated** — that coupling was refactored away.) So the swap touches
  **yawgpu-core only**, not the HAL.
- **The naga surface is concentrated.** It lives in `yawgpu-core/src/shader_naga.rs`
  (~3562 LOC, the `ReflectedModule` + all `generate_*` + reflection wrapper types)
  with three callers: `shader.rs`, `compute_pipeline.rs`, `render_pipeline.rs`.

## Integration surface (replacement target)

`yawgpu-core/src/shader_naga.rs` owns:

- **Entry:** `parse_and_validate_wgsl[_gated]` → `ReflectedModule { module: naga::Module,
  info: naga::valid::ModuleInfo, warnings }`.
- **Codegen (per target):** `generate_spirv`, `generate_glsl` (gles),
  `generate_msl`, `generate_render_vertex_msl`, `generate_render_fragment_msl`,
  `generate_render_msl`.
- **Reflection:** `entry_points`, `compute_workgroup_size` /
  `resolved_compute_workgroup_size`, `entry_point_io`, `resource_bindings[_for_entry]`,
  `msl_buffer_size_bindings_for_entry`, `fragment_builtins`, `overrides`.
- **yawgpu-core-owned wrapper types** (these stay; only their *producer* changes):
  `MslResourceBinding`/`MslResourceBindingKind`/`MslBindingMap`, `GeneratedMsl`/
  `GeneratedGlsl`/`GeneratedRenderMsl`, `ReflectedEntryPoint`,
  `ReflectedEntryPointIo`, `ReflectedWorkgroupSize`, `ReflectedResourceBinding`/
  `ReflectedResourceBindingKind`, `ReflectedFragmentBuiltins`, `ReflectedOverride`.

The goal of Phase 2 is to **keep this public API byte-for-byte** (renamed module
`shader_tint.rs`, same type/fn names + signatures) so the three callers and the
HAL boundary are untouched. The naga-typed fields of `ReflectedModule`
(`naga::Module`, `naga::valid::ModuleInfo`) become opaque Tint handles.

## Tint API mapping (the producer swap)

Reference driver: `dawn/src/tint/cmd/tint/main.cc`. Build via Dawn's CMake
(`src/tint/BUILD.cmake`; the spike's 73 s build used exactly this — the API libs
are `tint_api` + `tint_api_helpers`, C++20).

| yawgpu need | naga today | Tint replacement |
|---|---|---|
| Parse + validate WGSL | `front::wgsl::parse_str_with_warnings` + `valid::Validator` | `tint::Initialize()` once; `wgsl::reader::Parse(file)` → `Program`; `inspector::Inspector(program)` for reflection; `ProgramToLoweredIR(program)` → IR for codegen |
| Entry points / stage / workgroup | naga module walk | `Inspector::GetEntryPoints()` → `EntryPoint{name,stage,workgroup_size,...}` |
| Resource bindings | naga reflection | `Inspector::GetResourceBindings(ep)` → `ResourceBinding{resource_type,bind_group,binding,size,array_size,dim,sampled_kind,...}` |
| Overrides / pipeline constants | naga overrides + `PipelineConstants` | `Inspector::Overrides()` + writer `SubstituteOverridesConfig` |
| MSL codegen w/ flat Metal indices | `back::msl` + `MslBindingMap` | `msl::writer::Generate(ir, Options{entry_point_name, bindings, use_argument_buffers=false, ...})` → `Output{msl, workgroup_info, needs_storage_buffer_sizes, workgroup_allocations}`. Feed yawgpu's flat `metal_index` via `Options.bindings` (`Bindings.{uniform,storage,texture,sampler,...}` remap maps). |
| SPIR-V codegen | `back::spv` + bounds-check policy | `spirv::writer::Generate(ir, Options{entry_point_name, bindings, spirv_version=kSpv13, extensions, disable_robustness=false})` → `Output{spirv}` |
| GLSL ES codegen (gles) | `back::glsl` | `glsl::writer::Generate(ir, Options{entry_point_name, version=ES 3.1, bindings})` → `Output{glsl}`; bindings via `glsl::writer::GenerateBindings` |

**Binding remap is the core reflection-wiring work.** yawgpu already computes flat
Metal `[[buffer/texture/sampler(N)]]` indices today (it builds `MslBindingMap`).
Phase 2 routes those same indices into Tint's `Options.bindings` (group,binding →
remapped point) with `use_argument_buffers=false`, and the SPIR-V/GLSL `(set,binding)`
via the same `Bindings` struct.

## Phases

### Phase 0 — Delete vendor extensions (naga-independent; do FIRST)

Shrinks the API surface *before* the Tint swap so Phase 2 targets less code.

- Delete TBDR/subpass (`render_pipeline.rs` subpass paths, `InputAttachment`
  binding kind, `e2e_*_tiled.rs`, vendor FFI SType `0x7000_0010`+, `tiled` feature).
- Delete shader-passthrough (`reflect_spirv*`, `ShaderModuleSource::{Spirv,Msl}`,
  `MslReflection`/`MslEntryPoint`, `e2e_*_shader_passthrough.rs`, FFI entry points,
  `shader-passthrough` feature, `naga/spv-in`).
- Delete external-texture honest-rejection (`module_has_external_texture`,
  `ExternalTexture` binding kinds, `e2e_vulkan_external_texture.rs`).
- Scope: ~1250 lines across ~20 files (deletion-only). Keep the `gles` feature.
- **Gate:** Noop workspace test green; CTS Metal + native Vulkan green surface
  unchanged (the deleted areas were vendor-only, not WebGPU-baseline). Update
  `DESIGN.md`/`SPEC.md`/`specs/blocks/{33,55,67}` + mobile-extension specs to mark
  the extensions removed. Phase Review (no open CRITICAL/MAJOR).

### Phase 1 — Tint build + C shim + Rust FFI

**Phase 1a — DONE (2026-06-26).** New crate `yawgpu-tint` proves the
build/link/FFI path end-to-end on this Mac. `build.rs` (cmake crate) builds a
minimal Tint (`add_subdirectory(dawn) EXCLUDE_FROM_ALL` + `tint_shim` target →
WGSL reader + inspector + MSL/SPIR-V/GLSL writers only; Dawn native backends /
HLSL / validators / tests / protobuf all off) from a Dawn checkout located via
`YAWGPU_DAWN_DIR`, links it into one `libtint_shim` dylib, and exposes a C ABI
(`shim/tint_shim.{h,cpp}`, C++20, no abort across FFI). `src/lib.rs` wraps it; a
smoke test compiles WGSL→MSL (asserts an MSL `kernel`) and exercises the error
path. **Graceful degradation:** with `YAWGPU_DAWN_DIR` unset the crate builds as
a stub (`cfg(have_tint)` off, functions return an "unavailable" error) so
`cargo build/test --workspace` keeps working without a C++ toolchain. Build of
Tint-from-source ≈ 1m20s cold (cached after). Remaining 1b/1c below.

**Phase 1b — TODO (full shim, on the proven 1a foundation).**
- `tint_shim.cpp` (C++20, C ABI): opaque `TintProgram*` handle (parse+validate,
  holds `Program`/IR), reflection getters (entry points, workgroup, resource
  bindings, overrides), and per-target `Generate` (MSL/SPIR-V/GLSL) taking
  yawgpu's binding-remap inputs. **No panics/aborts across FFI** — every Tint
  ICE/validation failure → a C error code + message (CLAUDE.md principle 3).
  Clone IR per writer (writers mutate the IR `core::ir::Module&`).
- Rust FFI wrapper (hand-written or bindgen) presenting safe `Result`-returning
  fns, replacing the 1a smoke surface.

**Phase 1b — DONE (2026-06-26).** Full C ABI + safe Rust `Program` (RAII) wrapper
landed: `program_create`/`destroy`, entry-point / resource-binding / override
reflection, and `generate_{msl,spirv,glsl}` with grouped binding remap (MSL flat
`GenerateBindings(_,_,true,true)`, SPIR-V grouped `(false,false)`, GLSL ES 3.1 +
`texture_builtins_from_uniform`), `SubstituteOverridesConfig` override values, and
robustness control. Fresh IR lowered per generate call. Shim wraps every path in
try/catch → C `false` + heap `*err` (no abort across FFI). 8 unit tests green
(MSL/SPIR-V/GLSL compute + render stages, reflection, workgroup, overrides+subst,
binding remap, f16, error path); clippy `-D warnings` + fmt clean; stub mode
preserved. **Known gaps to resolve in Phase 2:** (1) `Override.default_value` is
not exposed by this Tint Inspector revision (returned as `0.0`) — yawgpu's
pipeline-constant resolution needs override defaults, so Phase 2 must recover them
(AST/const-eval, or run substitute with no values). (2) Tint *internal compiler
errors* may `abort()` rather than throw; the shim catches `std::exception` but an
ICE could still take down the process — install a non-aborting Tint ICE reporter
before relying on the "no abort across FFI" guarantee under fuzz/adversarial WGSL.

**Phase 1c — TODO (vendoring).** Vendor Dawn at a **pinned rev** (git submodule;
`depot_tools` present at `~/Documents/workspace/bin/depot_tools` for the
`gclient`/`DEPS` third_party sync) so the Tint build no longer depends on an
external `YAWGPU_DAWN_DIR` checkout. The submodule add + `gclient sync` are
network ops the **user** runs via the `!` prompt. Record the rev.
- Rust FFI wrapper (hand-written or bindgen) presenting safe `Result`-returning fns.
- **Gate:** builds on macOS (Metal + Vulkan) and cross-builds iOS arm64 + Android
  arm64-v8a (already proven in spike). Smoke test: trivial WGSL → MSL + SPIR-V +
  GLSL, magic numbers / non-empty source asserted.

### Phase 2 — Reimplement the shader frontend on Tint (parity layer)

- New `yawgpu-core/src/shader_tint.rs` implementing the **same public API** as
  `shader_naga.rs` (same fn names/signatures, same wrapper types) over the Phase-1
  FFI. `ReflectedModule`'s naga fields → opaque Tint handle(s).
- Wire each mapping-table row. Critical sub-tasks: PipelineConstants →
  `SubstituteOverridesConfig`; flat Metal indices → `Options.bindings`;
  robustness/bounds-check parity (Tint robustness default ON); entry-point IO /
  fragment builtins (`frag_depth`/`sample_mask`) / workgroup / overrides via
  Inspector; runtime-sized storage array buffer-size reflection
  (`needs_storage_buffer_sizes` + `workgroup_allocations`).
- Port the inline `#[cfg(test)]` unit tests to assert against Tint output (CLAUDE.md
  principle 1). Swap `shader_naga` → `shader_tint` at the three call sites.
- **Gate:** Noop workspace test green; clippy `-D warnings`; missing_docs clean.

### Phase 3 — CTS re-verification on real GPU (the dominant cost)

- Rebuild webgpu-native-cts against Tint-backed yawgpu; run the whole green surface
  on Metal + MoltenVK + native Vulkan + (gles) ANGLE.
- **Expected wins:** a class of naga-divergence findings evaporates (Tint = the
  oracle's compiler) — uniformity (F-120), abstract-type edge cases (F-133/F-134/
  F-136), F-085-class. Re-measure and retire them in the ledger.
- **Watch for NEW divergences** from different MSL/SPIR-V (entry-point naming,
  argument-buffer-off layout, robustness emission, workgroup allocation). Triage
  each; root-cause in the shim/wiring, not by relaxing core validation.
- Update `specs/tracking/cts-coverage.md` + reference the external FINDINGS ledger.

### Phase 4 — Remove naga + cleanup

- Delete `shader_naga.rs`; drop the `naga` dependency from all Cargo manifests;
  retire the `../wgpu` naga-fork pin. Update CLAUDE.md/DESIGN.md/SPEC.md, the mobile
  cross-build notes, and memory. Final Phase Review.

## Open design questions (resolve in Phase 1/2, not blocking the plan)

1. **Shim handle lifecycle** — one opaque `TintProgram` reused for reflect +
   multi-target codegen (recommended: parse once) vs re-parse per call.
2. **IR cloning** — writers take a mutable IR ref and may mutate; clone per target
   inside the shim (confirm Tint's clone cost is negligible for our shaders).
3. **Override resolution timing** — resolve pipeline constants at codegen via
   `SubstituteOverridesConfig` vs pre-substitute; match yawgpu's current
   `resolved_compute_workgroup_size` semantics.
4. **Robustness parity** — naga's current bounds-check policy vs Tint
   `disable_robustness=false`; confirm no behavioral CTS regression.
5. **Dawn rev cadence** — Dawn is unversioned; pin deliberately, bump on purpose.

## Risks

- **CTS regressions from different codegen** — mitigated by Phase 3 full re-verify
  (the dominant effort, as the spike predicted).
- **Build cost** — C++20 + CMake-from-build.rs adds ~73 s cold (cacheable); a real
  contributor/CI hit. Document the `$OUT_DIR` cache + submodule fetch.
- **Permanent C++ dependency** tracking Dawn's release cycle (acceptable — yawgpu
  already links MoltenVK/objc2).
- **FFI safety** — the shim must never abort across the boundary; all Tint failure
  modes route to `Result`.

## Effort

Build/FFI/mobile = de-risked (spike). Now dominated by **Phase 2 reflection wiring
+ Phase 3 CTS re-verify**; Phase 0 deletion is mechanical. Removing the
extension re-home (the user's decision) cuts the spike's "~1.5–3 month" estimate
materially — the long pole is CTS re-verification of the green surface, not new
compiler code.
