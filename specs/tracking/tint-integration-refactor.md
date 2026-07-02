# Tint integration refactor — remove the naga-compat shape

**Status: PLANNED (2026-07-02).** Follow-up to the completed naga→Tint migration
(`tint-migration-plan.md`). Phase 2 of that migration deliberately kept the old
`shader_naga.rs` public API **byte-for-byte** so the parallel-then-switch diffing
worked; naga is now deleted (P4a), so that compatibility constraint no longer buys
anything — but its cost remains baked into the frontend, the FFI crate, and the
shim. This plan removes the naga-era shape and re-centers the integration on
Tint's natural `Program → (Inspector | lowered IR → writer)` model.

**Ground rule:** every slice must be behavior-preserving at the WebGPU API level.
The CTS green surface (Metal byte-identical to Dawn; Vulkan api-trees at parity)
is the regression bar; slices that touch codegen inputs must byte-compare
generated MSL/SPIR-V before/after in unit tests where practical.

## Review findings (2026-07-02, three independent deep reads)

### A. What is already right (do not churn)

- **Parse-once is real.** WGSL crosses FFI exactly once per `createShaderModule`
  (`shader.rs:134` → `Program::parse`); every later operation reuses the opaque
  `Program` handle. No re-parse anywhere.
- **`shader_types.rs` is WebGPU-shaped, not naga-shaped.** No `Handle`/arena/
  `TypeInner` mirrors; flat binding/stage-oriented structs. Keep.
- **`HalShaderSource` text/bytecode payload** is a natural fit for Tint output.
- **`vertex_buffer_metal_indices` is LIVE, not residue** — it feeds the Tint
  vertex-pulling `_mslBufferSizes` contract (`render_pipeline.rs:1491-1513`;
  pulling re-enabled by the robust_access fix `ba46e12`). Only its doc comments
  still say "naga". Do NOT delete; relabel.
- Zero `naga` code references remain in yawgpu-core/yawgpu; yawgpu-hal has 15
  comment/test-name/string mentions only (see F7/F8).

### B. Findings, by class

**Correctness / latent-risk**

- **F1 — enum-ordinal ABI with no drift guard.** Reflection enums cross the shim
  as `u8` ordinals maintained by hand in 3–4 synchronized layers: header comments
  (`tint_shim.h:83-156`), C++ `static_cast<uint8_t>` fills (`tint_shim.cpp:460,
  599-608, 652`), Rust `raw_enum!` literals (`yawgpu-tint/src/lib.rs:1040-1362`),
  plus a fourth re-stringify hop for texel formats (`shader_tint.rs:1030-1077`).
  A Tint enum reorder (Dawn rev bump) silently corrupts reflection with no
  compile error. Likewise the FFI fn/struct signatures are hand-mirrored (no
  bindgen) across `tint_shim.h` / `lib.rs:177-290` / `tint_shim.cpp`.
- **F2 — GLES naga-name coupling (Tier 2, likely latent bug).** The GLES HAL
  looks up naga's GLSL conventions at runtime: uniform `"naga_vs_first_instance"`
  (`gles/pipeline.rs:411` — if absent, `first_instance` draw offsets silently
  no-op via `queue.rs:610`) and `_block_N` block-name suffix parsing
  (`gles/pipeline.rs:465-472`). The shim now emits Tint GLSL with
  `glsl::writer::GenerateBindings` (`tint_shim.cpp:1217-1241`), which does not
  produce naga's names. Needs output-level verification, then fix or delete.
- **F3 — asserted-not-enforced `Send + Sync`.** `unsafe impl Send/Sync for
  ReflectedModule` (`shader_tint.rs:24-27`) claims the Tint program is
  immutable-after-parse, but shim entry points construct fresh `Inspector`s and
  run IR lowering on the shared `const` program. Concurrent codegen on one
  module from two threads is unproven. Verify Tint's actual thread-safety
  contract or serialize per-program FFI calls with a mutex.

**Performance / architecture (the naga-era "stateless frontend" shape)**

- **F4 — every codegen call re-lowers the whole program.** `lower_ir`
  (`tint_shim.cpp:290-296`) runs `ProgramToLoweredIR` from scratch at every
  `generate_*` and `workgroup_storage_size` call (call sites cpp:903, 1084,
  1166, 1213). A render pipeline lowers the same module 3–5×.
- **F5 — reflection is uncached and O(N²) at the shim.** One
  `resolve_render_pipeline_descriptor` performs ~12–18 FFI reflection round
  trips: `entry_point_io()` (which reflects ALL entry points) ×~5
  (`render_pipeline.rs:2453, 2995, 3163, 3183, 3215`), `fragment_builtins()` ×2,
  per-entry resource reflection ×~6 (`resource_bindings_for_entry` even calls
  `entry_points()` first just for an existence check, `shader_tint.rs:378-391`).
  Below that, the shim's count/get accessor pairs rebuild an `Inspector` and
  re-run `GetResourceBindings` per index, and re-run the ~100-line
  `texture_sample_usages` sem-walk per element (`tint_shim.cpp:830-855`) —
  O(N²) per query.
- **F6 — workgroup size resolved by generating and parsing SPIR-V.**
  `resolved_compute_workgroup_size` (`shader_tint.rs:289-320`) runs full SPIR-V
  codegen and greps `OpExecutionMode LocalSize` back out (`spirv_local_size`,
  `shader_tint.rs:780`); Vulkan compute then generates SPIR-V again
  (`compute_pipeline.rs:388`). The literal path (`compute_workgroup_size:266`)
  already covers the no-override case.

**Dead code / naming residue (mechanical)**

- **F7 — dead naga-shaped fields and guards.**
  `ReflectedResourceBinding.statically_used` is always `true`
  (`shader_tint.rs:811`) making the `if !binding.statically_used { continue; }`
  guards unreachable (`compute_pipeline.rs:1232, 1354`);
  `ReflectedWorkgroupSize.override_keys` is always `[None;3]`
  (`shader_tint.rs:283, 316`) and `ReflectedOverrideKey` has no live producer;
  `ReflectedShaderStage` is a self-described backward-compat alias
  (`shader_types.rs:214-215`); `ShaderModuleSourceKind::Wgsl { _source }` keeps
  a never-read copy of the WGSL source (`shader.rs:94, 151`);
  `shader_tint.rs:5` carries a file-wide `#![allow(dead_code)]`; legacy
  `wgsl_to_msl` free fn used by one test (`yawgpu-tint/src/lib.rs:1758`).
- **F8 — stale "naga" comments/test names in yawgpu-hal** (15 sites; audit table
  in the review). All live logic (threadgroup sizing `metal/encode.rs:466-476`,
  `shaderDemoteToHelperInvocation` `vulkan/mod.rs:605`, Y-flip
  `vulkan/encode.rs:1987`) is frontend-agnostic — fix attribution only. Test
  names `render_shader_skips_vertex_descriptor_for_naga_vertex_pulling_msl`
  (`metal/pipeline.rs:627`) and
  `block_binding_from_name_extracts_naga_binding_suffix` (`gles/pipeline.rs:559`).
- **F9 — yawgpu-tint stub module duplicates the whole API.** ~700 lines
  (`lib.rs:1776-2487`) restate every public type/enum of the real impl
  (`lib.rs:11-1774`); only `Program`'s method bodies need the `cfg(have_tint)`
  gate. Also: `Raw*` mirror structs rebuilt per call (`Bindings::as_raw`
  allocates 7 Vecs per codegen, `lib.rs:1466-1513`), vestigial
  `PhantomData` lifetime (`lib.rs:1463`), no-op `_keep_*_alive` locals
  (`lib.rs:1708, 1749`), hand-freed error ladders duplicated on both sides
  (`lib.rs:618-645` / `tint_shim.cpp:1013-1044`), stringly-typed
  `Result<_, String>` everywhere.

## Refactor slices

Order chosen so safety nets land before shape changes, and mechanical cleanup
lands before the slices whose diffs would otherwise be noisy.

| Slice | Scope | Risk | Gate |
|---|---|---|---|
| **R1 — ABI drift guards** (F1) | C++ `static_assert(static_cast<int>(tint::…::kFoo) == N)` for every ordinal the shim exports; consider bindgen-from-header for the extern block; document the Dawn-rev-bump checklist. No behavior change. | low | build + Noop tests |
| **R2 — dead-shape sweep** (F7, F8) | Delete `statically_used` + dead guards, `override_keys`/`ReflectedOverrideKey`, `ReflectedShaderStage` alias, `_source` copy, `#![allow(dead_code)]` (fix what it hides), `wgsl_to_msl`; scrub/relabel the 15 HAL naga mentions (keep live logic; `vertex_buffer_metal_indices` stays, re-documented as the Tint pulling contract). | low | Noop workspace + clippy |
| **R3 — reflection: per-entry shape + memoization** (F5) | (a) shim: replace count/get pairs with one-shot array returns; hoist `Inspector`/`texture_sample_usages` out of per-index calls. (b) yawgpu-tint: per-entry queries. (c) core: `entry_point_io(name)`-shaped accessors + lazy `OnceLock` per-entry reflection cache on `ReflectedModule` serving all validators. | med | Noop + Metal CTS spot (render_pipeline, capability_checks) |
| **R4 — codegen: cached IR + workgroup-size API** (F4, F6) | Shim caches the pre-substitution lowered IR on the program handle (clone per writer — writers mutate IR); new shim call returning override-resolved workgroup size (kill the SPIR-V-generate-and-parse round trip + the Vulkan double-generate). | med | Noop + byte-compare MSL/SPIR-V unit tests + Metal/MoltenVK e2e |
| **R5 — yawgpu-tint crate hygiene** (F9, F3) | De-duplicate the stub (shared type module, cfg only on impls); `#[repr(C)]` public PODs passed directly (drop `Raw*` mirrors, PhantomData, keep-alive locals); RAII guard for the free-ladders; typed error enum; resolve the `Send/Sync` question (verify Tint thread-safety or add a per-program mutex). | med | Noop + clippy + miri-free unsafe review |
| **R6 — GLES Tint-name verification** (F2) | Unit-test the shim's actual GLSL output for first-instance + block naming; fix the HAL lookup (or delete the path if Tint GLSL carries explicit bindings). Tier 2: Noop/unit-level assertions + manual ANGLE when available. | low | gles feature build + unit tests |

Deliberately **not** planned: rewriting `shader_types.rs` (already WebGPU-shaped),
moving stage-interface validation into Tint (Rust-side matching is Dawn-like and
correct), pipeline-level codegen caching (pipelines are created once in
practice; revisit only if profiling says otherwise).

## Verification strategy

- Every slice: Noop workspace test + clippy `-D warnings` + feature-gated HAL
  unit tests (`--features metal/vulkan --lib`).
- R3/R4 (the only slices touching what Tint receives/produces): add unit tests
  that byte-compare `generate_msl`/`generate_spirv` output before/after for a
  representative shader set (compute + render + overrides + runtime-sized
  arrays), then real-GPU e2e (Metal + MoltenVK) and Metal CTS spot trees;
  full-tree re-sweep only at the end of the phase.
- Phase ends with the standard no-context Phase Review.

## Log

- 2026-07-02: three-way deep review (yawgpu-tint crate / core frontend / HAL
  consumption) completed; findings F1–F9 recorded; plan authored. Corrected one
  review claim: Metal vertex-pulling metadata is live under Tint, not naga
  residue.
