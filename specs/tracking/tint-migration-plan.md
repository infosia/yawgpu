# Tint migration plan — replace naga with Dawn's Tint as the shader frontend

**Status: PLANNING (2026-06-26).** Supersedes the "stay on naga (A)" decision in
`specs/tracking/tint-migration-spike.md`. The spike de-risked feasibility
(standalone CMake build 73 s / ~6 MB, Rust↔C shim, Metal runs Tint MSL, iOS +
Android cross-build all proven). The blocker that tilted the prior decision to A
was the cost of re-homing the TBDR `subpass_input` extension onto Tint. **That
blocker is removed: the user has decided to delete all naga-fork vendor
extensions** (see "Decisions" below), so migration scope no longer includes any
extension re-home.

## Governing decision (user, 2026-06-26): align to Tint, not naga

**naga is being removed entirely, so adopt whatever model Tint is comfortable
with — do NOT force naga's representations/output onto Tint.** Where naga and Tint
diverge, prefer **Tint's** representation and adapt yawgpu-core / the HAL / the
tests to match (Tint is the CTS oracle's compiler, so its classification is the
spec-correct reference). Concretely: render MSL uses Tint's per-entry-point model
(per-stage modules; the Metal HAL adapts — e.g. Metal vertex descriptors instead of
naga-style vertex pulling — rather than us replaying naga's combined-module
transforms); texture filterability uses Tint's direct `kFilterable`/`kUnfilterable`;
diagnostics/warnings follow Tint, and naga-specific test expectations are updated to
Tint's output. This governs P2c.2, the divergence resolutions, and Phase 3.

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

**Phase 1c — DONE (2026-06-26).** Dawn vendored as a pinned submodule at
`third_party/dawn`, rev `c8f5ca3df8b3b2f0ced5afa3c765e15bd5b065f7` (branch
chromium/7914). `build.rs` now resolves the Dawn source via `resolve_dawn_dir()`:
`YAWGPU_DAWN_DIR` override first, else the vendored submodule — gated on a
deps-fetched marker (`third_party/abseil-cpp/CMakeLists.txt`) so an
initialized-but-unfetched submodule degrades to the stub instead of a hard CMake
failure. Verified: `cargo test/clippy -p yawgpu-tint` with **no env var** builds
Tint from the submodule and passes (8 tests, clippy `-D warnings` clean).

**One-time contributor setup** (the Dawn submodule's `third_party` deps — abseil,
SPIRV-Tools, etc. — are NOT yawgpu-tracked; they are fetched per-clone):
```
git submodule update --init third_party/dawn
cd third_party/dawn && python3 tools/fetch_dawn_dependencies.py
```
Without this, `yawgpu-tint` builds as a stub (Tint FFI unavailable) and the rest
of the workspace is unaffected.
- Rust FFI wrapper (hand-written or bindgen) presenting safe `Result`-returning fns.
- **Gate:** builds on macOS (Metal + Vulkan) and cross-builds iOS arm64 + Android
  arm64-v8a (already proven in spike). Smoke test: trivial WGSL → MSL + SPIR-V +
  GLSL, magic numbers / non-empty source asserted.

### Phase 2 — Reimplement the shader frontend on Tint (parity layer)

**Decisions (user, 2026-06-26):**
- **Transition = parallel-then-switch.** Build `shader_tint.rs` *alongside*
  `shader_naga.rs`, selected by a `tint` cargo feature. naga stays the default
  through Phase 3 so the same CTS case can be diffed naga-vs-Tint to triage
  divergences; flip the default to Tint once parity holds, then delete naga
  (Phase 4). Cost accepted: yawgpu-core depends on both naga and Tint during the
  transition.
- **Override default values = recover from the AST in the shim.** Extend
  `tint_shim` to read override initializer values from the Program AST/sem
  (the Inspector only exposes `is_initialized`), so yawgpu's pipeline-constant
  resolution (`compute_workgroup_size`, etc.) matches naga.

**Slicing:**
- **P2a — plumbing (pure-Rust refactor, no behavior change).** Add the
  `yawgpu-tint` dep + a `tint` feature to yawgpu-core. Extract the
  backend-independent reflection/codegen *data* types (`Reflected*`, `Generated*`,
  `Msl*` — currently defined in `shader_naga.rs`) into a neutral module both
  frontends produce. Introduce a feature-selected `shader_frontend` facade
  (default → `shader_naga`, unchanged). Gate: workspace test green (naga path);
  `--features tint` compiles with `shader_tint` stubbed.
- **P2a.0 — shim gap fixes (yawgpu-tint).** Override-default-from-AST recovery +
  a non-aborting Tint ICE reporter (so internal compiler errors route to the C
  error path, not `abort()`). Unblocks P2b reflection.
- **P2b — shader_tint reflection.** Entry points / workgroup / resource bindings /
  overrides (with default values) / fragment builtins → the shared types. Unit
  tests.
- **P2c — shader_tint codegen.** `generate_spirv` / `generate_msl` (+ render
  vertex/fragment variants) / `generate_glsl`, mapping PipelineConstants → override
  values, flat Metal indices → binding remap, robustness parity, runtime-sized
  storage buffer-size reflection. Unit tests asserting against Tint output.
- **P2d — switch + parity.** Port the inline `#[cfg(test)]` tests; make the three
  call sites use the facade; verify `--features tint` Noop workspace test green.
- **Gate (each slice):** Noop workspace test green; clippy `-D warnings`;
  missing_docs clean; both default (naga) and `--features tint` build.

**Phase 2 progress (2026-06-26):**
- **P2a DONE** (`6275d39`) — plumbing (dep+feature, `shader_types.rs`, facade,
  skeleton). Default unchanged.
- **P2a.0 DONE** (`8f7c8a6`) — shim recovers override default values from the AST
  (`sem::Variable::ConstantValue()`). ICE: not catchable (Tint `[[noreturn]]`),
  documented.
- **P2b DONE** (`757c370`) — `shader_tint` reflection computed **from Tint** (zero
  naga refs): entry_points / workgroup (+resolved) / resource_bindings /
  fragment_builtins / overrides. Deferred: `entry_point_io` (→ **P2b.2**, needs a
  shim extension to expose Tint `EntryPoint` IO variables) and
  `msl_buffer_size_bindings` (→ P2c). **An initial P2b attempt delegated reflection
  to a naga mirror — rejected and reverted; shader_tint must never depend on naga.**
  - **WIP state:** default path green (the standing gate). `cargo test
    --features tint --lib` is intentionally **red (293 pass / 58 fail)** — all 58
    are pipeline/render-pass creation tests that require codegen, which is still
    the P2c skeleton. Goes green as P2b.2 + P2c land.
- **P2c.1 DONE** (`7d09188`) — non-render codegen from Tint: `generate_spirv`,
  `generate_glsl` (gles), compute `generate_msl`; PipelineConstants→overrides;
  MslBindingMap→Tint per-class binding remap (buffers classified uniform/storage,
  textures sampled/storage via reflection); robustness via `disable_robustness`.
- **P2b.2 DONE** (`a414c77`) — `entry_point_io` from Tint (shim exposes
  inspector StageVariables: location/component+composition type/interpolation).
  This unblocked render-pipeline inter-stage validation: `--features tint --lib`
  **58 → 6 failures** (349 pass).

**Remaining `--features tint --lib` failures (6), now well-isolated:**
1. **render-MSL codegen (1 test, → P2c.2)** — `generate_render_{vertex,fragment,}_msl`
   still skeleton. Architectural question: naga emits ONE MSL module with both
   stages + Metal-specific transforms (vertex pulling, frag-depth clamp, sample
   mask); Tint emits per-entry-point. Decide per-stage modules vs combined, and
   whether to replicate the transforms via Tint options or adapt the Metal HAL.
   (Noop tests don't compile MSL, so only the explicit MSL-generation test fails
   on Noop; real-Metal render correctness is a Phase 3 item.)
2. **texture sample-type / filterability divergence (3 tests)** — yawgpu derives
   filterable-vs-unfilterable-float from `sample_usage` (Load/Sample/Gather, which
   naga computes by IR usage analysis); Tint reports filterability directly
   (`kFilterable`/`kUnfilterable`), and the P2b mapping collapsed both to plain
   `Float` + hardcoded `sample_usage: Sample`. Fix needs the shim to expose Tint's
   filterability/texture-usage and a reconciled mapping. (Phase-3-class semantic
   divergence.)
3. **diagnostic/warning parity (2 tests)** — Tint emits different
   warnings/diagnostics than naga.

**NEXT: P2c.2** (render-MSL — the one architectural decision left) + resolve the 5
semantic divergences (texture filterability ×3, diagnostics ×2) → **P2d** (flip the
feature default + full parity). Then Phase 3 (real-GPU CTS) / Phase 4 (remove naga).
Watch codex for shortcuts (it delegated reflection to naga once — caught + reverted).

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
