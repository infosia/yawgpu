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

- **P2-div DONE** (`f410f04`) — resolved the 5 semantic divergences by aligning to
  Tint: shim exposes non-error diagnostics → `shader_tint` surfaces Tint warnings;
  shim computes per-texture sample usage (Gather>Sample>Load) from Tint's sem call
  graph → F-080 textureGather validation works; "not wgsl" parse-error assertion
  cfg-split per frontend.
- **P2c.2 DONE** (`114604b`) — render-MSL codegen (Tint per-stage:
  generate_render_{vertex,fragment}_msl via per-stage `generate_msl`). Vertex
  pulling / frag-depth clamp / sample mask are NOT replayed (Tint uses `stage_in`;
  Metal HAL adapts in Phase 3 — marked TODO). `generate_render_msl` combined
  same-module stays a documented skeleton (P2c.3; no `--features tint` test needs it).

**🏁 MILESTONE (2026-06-26): `cargo test -p yawgpu-core --features tint --lib` is
FULLY GREEN (356/0).** The Tint frontend produces all reflection + codegen the
yawgpu-core lib suite exercises, entirely from Tint (`shader_tint.rs` has zero naga
refs). Default (naga) path unchanged throughout.

- **P2d DONE** (`e22da6c`) — `yawgpu` crate forwards a `tint` feature; integration +
  FFI suites run under Tint and reach parity. Fixed two diagnostic-wording asserts
  (cfg-split) and a real override-id divergence: naga sets `ReflectedOverride.id`
  only for explicit `@id(N)`, but Tint assigns an id to every override; yawgpu keys
  pipeline constants by numeric id only for `@id` overrides, so Tint's implicit ids
  made name-keyed override constants error → uncached error pipelines → caching
  tests failed. Shim now reports `has_explicit_id`
  (`ast::HasAttribute<ast::IdAttribute>`); shader_tint surfaces id only when explicit.

**🏁🏁 MILESTONE (2026-06-26): the ENTIRE Noop-testable surface is GREEN under
`--features tint`** — yawgpu-core lib (356/0) + yawgpu lib & integration (283/0),
all reflection + codegen from Tint, naga still default & untouched.

**NEXT:**
- **Phase 3** — real-GPU CTS under `--features tint` (Metal/MoltenVK/native-Vulkan/
  ANGLE). The e2e (`#[ignore]`) suites + CTS validate Tint's actual codegen on
  hardware; this is where the Metal HAL adapts to Tint's per-stage / `stage_in`
  render model (vertex pulling not replayed). Do this BEFORE flipping the default
  (flipping makes the real-GPU e2e use Tint codegen).

  **Phase 3 finding #1 (2026-06-26, Metal e2e compute under tint): Tint ICEs
  (aborts the process — `[[noreturn]]`) on any `arrayLength()` / runtime-sized
  storage array.** `tint/lang/msl/writer/raise/decompose_buffer.cc:111
  TINT_ASSERT(call->Func() != kArrayLength)` fires because the shim's
  `generate_msl` never sets `options.array_length_from_constants` (the
  `_mslBufferSizes` contract). `cmd/tint/main.cc:1116 GenerateArrayLengthFromConstants`
  sets `ubo_binding` (slot 30, == its `immediate_binding_point`) + a
  `bindpoint_to_size_index` mapping each non-fixed-footprint kStorage buffer the
  entry point references → a sequential index. **Fix (next slice):** the shim's
  `generate_msl` must set `array_length_from_constants` with `ubo_binding =
  yawgpu's computed buffer_sizes_slot` and `bindpoint_to_size_index` matching the
  ORDER yawgpu's Metal HAL fills `_mslBufferSizes` (i.e. `buffer_size_bindings[i]
  → i`); reconcile with the shim's hardcoded `immediate_binding_point = {0,30}`
  (from P1b — may itself collide with a real slot 30 binding). This is the
  shim↔Metal-HAL `_mslBufferSizes` contract; verify on real Metal. (Didn't show on
  Noop — Noop never generates MSL.) Note: the ICE confirms Tint aborts are real and
  uncatchable across FFI (P2a.0) — fixing the missing option avoids THIS abort, but
  fuzz/adversarial WGSL can still abort; a true guard would need out-of-process
  compilation or a Tint patch.

  **Phase 3 finding #1 RESOLVED** (`ec62ac7`): shim now sets
  `array_length_from_constants` (ubo_binding = buffer_sizes_slot; ordered size
  bindings returned for the HAL). Real Metal e2e no longer SIGTRAPs on arrayLength.

  **Phase 3 finding #2 (the active one): Tint MSL storage-buffer binding indices
  do NOT match yawgpu's Metal HAL binding** → wrong real-GPU results. Unmasked once
  the ICE was fixed (crash-masks-behavior): `e2e_metal_compute` under
  `--features metal,tint` now runs but 4/5 fail — e.g. `metal_compute_fills_storage_buffer`
  (a FIXED `array<u32,8>` storage buffer at `@group(0)@binding(0)`, NO arrayLength)
  reads back all zeros instead of `[0,1,4,…,49]`: the shader's writes land in a
  different Metal `[[buffer(N)]]` slot than where the HAL bound `out_data`. So
  yawgpu's `MslBindingMap` metal_index → the shim's binding remap → Tint's emitted
  `[[buffer(N)]]` is NOT round-tripping to the index the Metal HAL binds. **Next
  slice:** dump Tint's MSL for one failing shader, read its `[[buffer(N)]]`/
  `[[texture(N)]]` indices, compare to the HAL's bound indices (and check whether
  the new dynamic `immediate_binding_point` reserves a colliding slot / Tint flattens
  `(dst_group,dst_binding)` differently than expected), reconcile, re-verify on real
  Metal. This is THE core Metal binding contract for the Tint frontend; the
  SPIR-V/Vulkan path needs the analogous real-GPU check too.

  **Phase 3 finding #2 RESOLVED** (`d1e97c8`): Tint minifies the MSL entry point
  (`main`→`v`); the shim now sets `remapped_entry_point_name = "tint_"+ep` and
  returns it so the Metal HAL binds the right function. **Real Metal:
  e2e_metal_compute --features metal,tint 5/5** (was 4/5 all-zeros). Compute under
  Tint fully works on hardware.

  **Phase 3 finding #3 (active): render needs the Metal HAL to use a
  `MTLVertexDescriptor` (`stage_in`) instead of naga-style vertex pulling.**
  `e2e_metal_render --features metal,tint` real tests fail (no triangle — a vertex
  shader reads `@location(0) position: vec2<f32>` from a vertex buffer). Per the
  governing decision we do NOT replay naga vertex pulling; Tint emits vertex MSL
  expecting Metal `[[stage_in]]` attributes, so **yawgpu-hal/metal must build a
  `MTLVertexDescriptor` from the pipeline's vertex-buffer layouts** (attribute
  formats/offsets/strides/step-modes) and bind vertex buffers at the slots Tint
  expects, replacing vertex pulling for the Tint frontend. Largest remaining Phase 3
  chunk; then re-verify render on Metal, then MoltenVK/native-Vulkan, then flip.

  **Phase 3 finding #3 RESOLVED** (`e7eec7c` + `411f262`): the Metal HAL already had
  an `MTLVertexDescriptor` path but chose it by `HalShaderSource` *variant*; now
  `render_shader_uses_metal_vertex_descriptor` detects the model from the emitted
  vertex MSL source (`contains("[[stage_in]]")`) — Tint stage_in → descriptor,
  naga pulling → pulling (unchanged). Plus `emit_vertex_point_size` wired through the
  shim (point-list needs `[[point_size]]`). **🏁 Metal e2e at PARITY under Tint
  (real M2):** e2e_metal_{compute 5/5, render 3/3, point 1/1, f16 5/5, f114, f115,
  threading_audit 15/15} all green under `--features metal,tint`; naga default
  render unchanged (no regression).

  **Phase 3 — Vulkan/MoltenVK under `--features vulkan,tint` (M2, started):**
  e2e_vulkan_{compute 4/4, f16 5/5} GREEN — SPIR-V compute lines up (Tint keeps
  `main` in SPIR-V; binding remap works). **Finding #4 (active): render triangle
  winding / front-face.** Failures cluster on culling + winding-sensitive draws:
  `vulkan_cull_back_discards_back_facing_cw`, `vulkan_cull_back_keeps_front_facing_ccw`,
  `vulkan_render_uniform_color_triangle_readback` (no green pixel — likely the
  triangle is front-face-culled because its winding flipped), plus
  `vulkan_render_two_color_attachments_write_distinct_targets` (count 3≠1). Likely
  root cause: the Vulkan **clip-space / Y-flip / winding convention** differs between
  Tint's and naga's SPIR-V (naga applies a Vulkan position adjustment); under Tint
  the emitted winding is opposite, so the test's front_face/cull setup culls the
  triangle. **ROOT CAUSE CONFIRMED:** naga's SPIR-V default `WriterFlags` includes
  `ADJUST_COORDINATE_SPACE` (`../wgpu/naga/src/back/spv/mod.rs:1123`) → naga flips Y
  **in the shader** for Vulkan; yawgpu's Vulkan HAL viewport uses **positive height**
  (`yawgpu-hal/src/vulkan/encode.rs:1050-1053`, no flip), relying on naga's shader
  flip. Tint does NOT flip in-shader (Dawn uses a negative-height viewport) and the
  Tint SPIR-V writer has no clip-space option. So under Tint nothing flips Y → image
  inverted + winding reversed → front_face/cull tests fail. **Fix (aligned to Tint,
  a deliberate slice — touches the naga DEFAULT path + HAL, needs dual-frontend
  real-GPU re-verify):** make the Vulkan HAL flip Y via a **negative-height viewport**
  (`y += height; height = -height`, the wgpu/Dawn convention) AND drop
  `ADJUST_COORDINATE_SPACE` from naga's `generate_spirv` flags so naga also stops
  flipping in-shader — both frontends then emit un-flipped SPIR-V and the HAL flips
  once. Re-verify naga Vulkan e2e (must stay green) AND tint Vulkan e2e on MoltenVK,
  ideally native Vulkan too. Also triage `two_color_attachments` (count 3≠1) — may be
  the same winding issue or separate. Then optional GLES, flip the default, Phase 4.
  Combined `generate_render_msl` (P2c.3) still skeleton.

  **Finding #4 RESOLVED** (`cd7914d`): unified Vulkan clip-space on a negative-height
  viewport (HAL) + dropped naga's in-shader `ADJUST_COORDINATE_SPACE`. Real MoltenVK:
  naga DEFAULT unregressed, tint render/threading (incl. two_color_attachments) green.

  ### 🏁🏁🏁 Phase 3 MILESTONE: both Tier-1 backends at near-full parity under Tint (real M2 + MoltenVK)
  Comprehensive e2e sweep under `--features {metal,vulkan},tint`:
  - **Metal 79/80** (every e2e_metal_* binary).
  - **Vulkan/MoltenVK 51/52** (every e2e_vulkan_* binary).
  - naga DEFAULT path unregressed throughout.

  Six hardware bugs found+fixed, all masked on Noop (Noop never compiles backend
  shaders): arrayLength ICE, MSL entry minify, MSL stage_in vertex descriptor,
  point_size, array_length_from_constants, Vulkan Y-flip/winding.

  **Only 2 remaining e2e failures, both bounded:**
  1. `e2e_metal_depth::metal_readonly_depth_stencil_isolation` (F-055 isolation) —
     **PRE-EXISTING, NOT a Tint regression: it fails IDENTICALLY under naga default**
     (verified). The test's Submit 2 does a `CopyTextureToBuffer` with `DepthOnly`
     aspect on a `Depth24PlusStencil8` texture — `depth24plus` has no defined layout
     and is non-copyable to a buffer, so yawgpu-core correctly rejects it ("copy
     texture to buffer depth/stencil format does not support this copy aspect/usage")
     → error command buffer → all-zero readback. A stale/broken test unrelated to the
     migration; fix is in the test (sample the depth aspect like the READ stage does,
     don't direct-copy depth24plus) — a separate cleanup. **Conclusion: Metal under
     Tint has ZERO regressions vs naga — true full parity (both 79/80, same one
     pre-existing-broken test).**
  2. `e2e_vulkan_external_texture` — passes under naga, fails under Tint: the ONE
     genuine Tint difference, and it is **deferred by the governing decision**
     (external textures get reworked on Tint's native support rather than naga's
     honest-rejection). Expected until that rework.

  **Net: Tint has ZERO *unexpected* real-GPU regressions on either Tier-1 backend.**
  Metal = true parity (the single failure is pre-existing under naga); Vulkan = parity
  except the deliberately-deferred external textures. The parallel-then-switch
  "parallel" half is fully validated on hardware.
- **Flip default → Tint — DONE** (`7fda995`): `yawgpu-core` + `yawgpu` now
  `default = ["tint"]`; the default build/test/e2e use Tint. naga is the
  `--no-default-features` opt-out (kept until Phase 4). Verified: default workspace
  green, default e2e_metal/vulkan_compute green on HW, naga opt-out 349/0. **Tint has
  LANDED.** Consequence: the default build now needs the vendored Dawn submodule (P1c
  setup); without it yawgpu-tint is a stub. Doc rewrite (README "naga is the only
  GPU-stack dep", CLAUDE.md naga references, build-setup) is folded into Phase 4.
- **P2c.3** — combined same-module `generate_render_msl` (minor; no test needs it).
- **P2c.3** — `generate_render_msl` combined same-module (minor; no test needs it yet).
- **Phase 3** — real-GPU CTS (Metal/MoltenVK/native-Vulkan/ANGLE); the real render
  correctness + Metal HAL adaptation to Tint's per-stage / `stage_in` model.
- **Phase 4** — remove naga.

(Process note: codex delegated reflection to naga once — caught in review + reverted;
every shader_tint slice now verifies `grep naga` is empty.)

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

- **P4a — DONE** (`64fe785`): deleted `shader_naga.rs` (−3528 lines); collapsed the
  `crate::frontend` facade to `shader_tint` unconditionally; repointed all
  `shader_naga::` → `crate::frontend::`; dropped the naga dep (yawgpu-core), the naga
  dev-dep (yawgpu), and the `../wgpu` fork pin (workspace + Cargo.lock); removed the
  `tint` feature + `default=["tint"]` (yawgpu-tint is now a non-optional dep, the sole
  frontend); `gles` uses Tint's GLSL writer. `grep naga` across yawgpu-core/yawgpu
  (.rs/.toml) + Cargo.lock = none. **naga is GONE — Tint is the only frontend.**
  Verified: default workspace green; default Tier-1 e2e on real HW green (Metal
  compute 5/5 / render 3/3 / point 1/1; Vulkan render 5/5 / threading 14/14 / compute
  4/4, single-threaded — MoltenVK flakes under parallel test execution).
- **P4b — docs (in progress):** rewrite README ("only dep is naga" → Tint/Dawn +
  Dawn-submodule build setup), CLAUDE.md (naga-fork workflow obsolete; frontend is
  Tint), DESIGN.md/SPEC.md, shader spec blocks.
- **Remaining follow-ups (non-blocking):** external-texture rework on Tint's native
  support; combined `generate_render_msl` (P2c.3) skeleton; fix the stale
  `e2e_metal_depth::metal_readonly_depth_stencil_isolation` test (pre-existing, fails
  under old naga too — does a non-copyable depth24plus DepthOnly copy).

## 🎉 MIGRATION FUNCTIONALLY COMPLETE — naga→Tint
Tint is the sole shader frontend, the default build, and verified at parity with the
old naga frontend on both Tier-1 backends (Metal + Vulkan/MoltenVK) on real hardware.
`feature/tint` (main untouched) is ready for review/merge.

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

## Post-migration CTS regression audit (2026-06-27)

CTS smoke testing of `feature/tint` against the Dawn oracle (real Metal, via
webgpu-native-cts) found the "hardware parity" milestone was **premature**: three
WebGPU behaviors that naga implemented as MSL transforms were left as hardcoded
stubs in `shader_tint.rs::generate_stage_msl` and silently regressed. All fixed by
threading the equivalent Tint MSL-writer option through the shim (the **Metal HAL
needed no change** — it already consumes all three). See memory
`tint-transform-regressions`.

| Regression | Was | Stub | Tint fix | CTS verify (Metal) |
|---|---|---|---|---|
| Workgroup memory reads 0 | F-069 | `workgroup_memory_sizes: Vec::new()` | `Output::workgroup_allocations` → per-index threadgroup sizes (rounded to 16) | atomics 1445/0 (was 86 fail) |
| Pipeline sample mask dropped | F-051/F-053 | `sample_mask` arg ignored | `Options::fixed_sample_mask` | sample_mask 68/0 (was 32 fail) |
| frag_depth not clamped to viewport | F-045 | `frag_depth_clamp_slot: None` | `Options::depth_range_offsets={0,4}` at the immediate-block slot (`immediate_binding_point->binding`) | depth_clip_clamp 3/0 |

Also fixed in the same pass (not a regression — a Phase-0 deletion side-effect):
**`createTexture:new_usages:usage=48`** (`TransientAttachment` 0x20) — TBDR deletion
dropped the bit from `TextureUsage::ALL`; re-accepted as a plain render attachment
(no memoryless optimization), validated as transient⊕render-attachment only.

Verified clean afterward on real Metal: operation tree
(rendering/render_pass/render_pipeline/compute/compute_pipeline/vertex_state)
1069/0/0, all atomics 1445/0/0, createTexture usage 34/0/0; workspace test + clippy +
fmt green. CTS build recipe (Tint dylib link): memory `cts-tint-dylib-build`.

**Lesson:** when swapping a shader frontend, grep the new codegen path for
`None`/`Vec::new()`/`TODO` stubs and map each to a transform the old frontend had;
verify with the **targeted** CTS case (atomics `*_workgroup`, `sample_mask`,
`depth_clip_clamp`), never the "basic" case alone. Remaining known stub:
external-texture MSL remap (deferred — `f060-external-textures`).

## CTS Dawn-parity sweep (2026-06-27) — goal: yawgpu CTS == Dawn CTS

User goal: make yawgpu's CTS results **100% identical to Dawn's** (vendor extensions
incl. TBDR come *after*). Method: run the full `api,validation` + `api,operation`
tree (~196 file queries) under `cts --isolate --workers 8` vs the Dawn oracle on real
Metal, cluster failures by test, root-cause each to a single cause, fix, re-sweep.

**Baseline (post-regression-fixes, before this sweep): fail=615, crash=43.** Each
cluster traced to one root cause, all fixed (shim/core; HAL unchanged except the
vertex-stride one):

| Cluster | Cases | Root cause | Fix | Commit |
|---|---|---|---|---|
| layout/bind-group compat, buffer in-pass | ~352 | Tint reflects only statically-used bindings; yawgpu errored on unused layout bindings | skip unreflected entries in tint_bindings_for_msl | dea89ba |
| maxComputeWorkgroupStorageSize at_over | ~222 | workgroup_storage_size stubbed to 0 | surface tint::core::ir::GetWorkgroupInfo().storage_size | dea89ba |
| vertex_state type-match | 41 crash | arrayStride=0 → Metal MTLVertexDescriptor "no stride" abort | Constant step + align_4(max attr extent), mirror Dawn | ab5f94a |
| storage_texture format, texture_view write | ~20 | texel_format() emitted "Rgb10A2…/Rg11B10…" (caps) vs lowercase keys | align producer literals | 4264df3 |
| overrides,entry_point validation_error | 2 crash | **use-after-free**: shim parsed against a local Source::File; dangling on diagnostic format | YawgpuTintProgram owns unique_ptr<Source::File> | 1cc4ea6 |
| overrides,workgroup_size storage limit | 2 | storage size 0 for override-derived sizes | SubstituteOverrides before GetWorkgroupInfo | 190fd69 |

Plus external-texture multiplanar MSL (Slice A, 279f9a4) and the 3 transform
regressions (e4b0db1). **Pattern confirmed:** nearly every divergence was a Tint
reflection field stubbed to a default (`0`/`None`/`Vec::new()`) or a naga-era
assumption (all-bindings-reflected, vertex-pulling). Each fix verified on real Metal
before the next.

**Known-remaining (expected, not bugs):** `index_buffer_format_dirtying` — yawgpu is
*stricter* than Dawn's lenient oracle (in expectations/yawgpu.txt). **Not yet swept:**
the `shader,execution` tree (huge), and Vulkan/MoltenVK + GLES backends. Memory:
`tint-transform-regressions`. CTS run recipe: `cts-tint-dylib-build`.
