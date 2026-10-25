# Block 113 — bug-class audit: core / FFI

Status: **PLANNED (2026-10-25)**.
Source: `specs/tracking/class-audit-2026-10.md` (the IDs below are its
rows; each row states the defect, the Dawn behaviour and the triggering
call). Oracle: Dawn `third_party/dawn/src/dawn/native/` + `webgpu.h` +
the WebGPU spec. Every rule is Noop-testable; each fix ships with an
inline unit test reproducing the row's scenario (it must fail on the
pre-fix code).

| Slice | Rows | Files | Wave |
|---|---|---|---|
| S1 validation parity | D1, D2, D3, D5, D6, D7, D4/E4, E1, E3, E7, E8, E9 | `yawgpu-core` texture_view / texture / command_encoder / render_pipeline / compute_pipeline / bind_group_layout / shader_tint; `yawgpu-tint` (E1 per-entry-point override list) | A |
| S2 FFI edge values | E2, E5, E6, E10, E11, E12 | `yawgpu/src/conv/**`, `yawgpu/src/ffi/**` (non-concurrency parts) | A |
| S3 concurrency / lifetime | C1–C11, S5, S10, T6 | `yawgpu-core` queue / device / buffer / error; `yawgpu/src/ffi/**` | B (after S2) |
| S4 | Phase Review | | |

Rules (one per row; "match Dawn" means the exact Dawn function cited in
the ledger row):

- **D1** reject a texture view with `dimension != 2D` on a texture with
  `sampleCount > 1`.
- **D2** reject a resolve target whose view dimension is not 2D / 2DArray.
- **D3** match Dawn `ValidateCopySizeFitsIn…` / `copy within same texture`:
  a 3D texture copied onto itself within one mip is rejected; a 1D texture
  copied onto itself is rejected; the zero-size early return no longer
  skips the same-subresource rule.
- **D5** the depth/stencil "copy covers the whole subresource" rule applies
  to zero-size copies too (copy and writeTexture).
- **D6** `copyBufferToBuffer` with `source == destination` is rejected
  regardless of size.
- **D7** a single-aspect view's format must be the aspect format (match
  Dawn `ValidateCanViewTextureAs`); a defaulted view format resolves to the
  aspect format.
- **D4/E4** alphaToCoverage requires a fragment state and `targets[0]`
  present with an alpha channel; blendability is irrelevant.
- **E1** the "override without default must be provided" rule and the
  constant-key lookup use the overrides **referenced by the stage's entry
  point** (keys of other module overrides remain legal). Add a shim query
  for the entry point's override ids (Tint inspector entry-point
  `overrides`); keep the existing ABI drift guards consistent.
- **E3** reject unknown `WGPUVertexFormat` values at pipeline creation.
- **E7** i32 / u32 override values: range check only (truncate), per Dawn
  `IsDoubleValueRepresentable`.
- **E8** numeric constant keys match only the exact decimal spelling of the
  id; spelling variants are "not found".
- **E9** reject BGL entries with visibility bits outside
  VERTEX | FRAGMENT | COMPUTE.
- **E2** every enum mapper rejects out-of-range values with a validation
  error and an error object (sampler address / filter / mipmap filter,
  view dimension, texture aspect incl. copies / writeTexture, load / store
  op); `Undefined` keeps its documented default. Update the unit tests that
  asserted the silent `None`.
- **E5** non-UTF-8 WGSL → error shader module with compilation info.
- **E6** non-UTF-8 entry-point name → validation error, never "not given".
- **E10** null array + non-zero count → validation error (viewFormats,
  requiredFeatures → request fails), and `dynamicOffsets` / `ExecuteBundles`
  route to the encoder / device error sink instead of `assert!`.
- **E11** invalid callback mode → null future + no callback (Dawn
  `ValidateCallbackMode`); honour `forceFallbackAdapter` (no fallback
  adapter → null adapter / Unavailable); invalid `featureLevel` → error.
- **E12** `wgpuTextureGetTextureBindingViewDimension` returns `Undefined` in
  core mode; a chained value is validated.
- **C1** `write_buffer` participates in the submission lock like
  `write_texture` (validation of map state under it).
- **C2** `GetLostFuture`: check-and-register atomically with loss.
- **C3** device-lost callback receives null when the device has no external
  reference left (Dawn `HasExternalRef`).
- **C4** instance drop fires `CallbackCancelled` for every pending callback,
  exactly once; pending results must not keep the instance alive (break the
  Arc cycle, e.g. Weak or explicit cancel on last instance release).
- **C5** error scopes are per thread (`webgpu.h` "current thread's error
  scope stack"; Dawn keys by thread id).
- **C6** after device loss: mapAsync → Aborted, submit / writeBuffer /
  writeTexture are no-ops, validation errors are no longer dispatched,
  uncaptured callback cleared once the lost event completes (Dawn
  `ValidateIsAlive`, `Device.cpp`).
- **C7** `wait_idle` snapshots the submission index before waiting.
- **C8** `unmap` publishes Unmapped only after the host write-back.
- **C9** pending work-done callbacks complete with Success on device loss.
- **C10** `wgpuSurfaceConfigure`: no callback under surface locks; on
  configure failure the previous state is restored or cleared consistently.
- **C11** after `wgpuDeviceRelease` returns, the uncaptured callback is not
  running and never runs again.
- **S5** `wgpuSurfacePresent` flushes pending queue writes first.
- **S10** `onSubmittedWorkDone` flushes / accounts for pending writes.
- **T6** `writeTexture` repacks when `offset % 4 != 0` for depth / stencil
  formats (Vulkan requires 4-byte buffer offsets for DS images).

Gates: `cargo test --workspace`, strict clippy (all backend features),
fmt; Claude runs the Metal + Vulkan e2e suites (Vulkan under sync
validation) and webgpu-native-cts on Metal against a baseline taken before
the block (`api,validation,*`, `api,operation,*`) — no new failures.

## S4 — Phase Review (2026-10-25)

Fresh reviewer over `76a0714`, `b469aa5`, `7488716`: **0 CRITICAL / 1 MAJOR /
7 MINOR**. Gates before review: `cargo test --workspace`, strict clippy
(all backend combos) + fmt clean; Metal / Vulkan e2e green (Vulkan under
sync validation, only the pre-existing MoltenVK fail); webgpu-native-cts
Metal `api,validation` identical to the pre-block baseline (298847/0),
`api,operation` 228851/0 (+3 skips: `forceFallbackAdapter` now correctly
Unavailable on this host).

| ID | Finding | Disposition |
|---|---|---|
| M1 | C11 incomplete: `yawgpuDeviceCreateSubpassPassLayout`, `yawgpuDeviceCreateSubpassRenderPipeline`, `yawgpuDeviceCreateExternalTexture`, `wgpuDeviceDestroy` still borrow the device handle and use it after an error dispatch — an uncaptured callback that releases the last reference frees it (use-after-free) | fix: `clone_handle` everywhere a device entry point can dispatch; reentrant-release test through a vendor entry point and `wgpuDeviceDestroy` |
| m1 | per-thread error-scope / active-callback maps grow for every thread that ever dispatched (Dawn trims) | fix: insert only on push; remove empty entries |
| m2 | `wgpuInstanceRelease(NULL)` now UB (bare `Arc::from_raw`) | fix: borrow first, like the other releases |
| m3 | device-lost callback in `WaitAnyOnly` mode can never fire: `GetLostFuture` returns a separate callback-less future (Dawn returns the lost event's own future) | fix: return the creation-time lost future id |
| m4 | E12 validates the chained `TextureBindingViewDimension` even in core mode (Dawn reads it only without flexible texture views) | fix: match Dawn (ignore in core mode) |
| m5 | E1 per-entry-point override lookup not memoized (undoes the Tint reflection memoization) | fix: cache per entry point |
| m6 | test gaps: override reached only through another override's initializer; `map_load_op` / `map_store_op` invalid branch; null `viewFormats` conv test does not assert the error | fix: add the tests |
| m7 | non-UTF-8 pipeline-constant keys become `""` | fix: validation error, like E6 |
