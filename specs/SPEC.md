# SPEC.md — yawgpu top-level specification & roadmap

The implemented spec is `webgpu.h` (`webgpu-headers/webgpu.h`,
6766 lines). Behaviour is pinned by Dawn's validation tests
(`dawn/src/dawn/tests/unittests/validation`, 55 files), ported to
Rust. See `reference/dawn-test-mapping.md` for the per-file port plan.

## Execution model

Implementation is done by a **separate coding agent**; Claude plans,
reviews, and commits. See `reference/workflow.md` for roles, the per-slice
loop, and the task-handoff template. Each phase is decomposed into
self-contained task handoffs recorded in `tracking/phase-N.md`.

## Phased roadmap

Each phase: write `blocks/<area>.md`, emit task handoffs, the coding agent
ports the Dawn test(s) as failing Rust tests and implements minimally on
Noop, Claude reviews against acceptance criteria and commits, log in
`tracking/phase-N.md`. **Each phase then ends with a mandatory Phase
Review ("Clean Review Then Fix")** — a fresh no-context subagent emits
`CRITICAL`/`MAJOR`/`MINOR` findings, fixed in severity order; no phase is
COMPLETE with an open CRITICAL/MAJOR (see `reference/workflow.md`).

| Phase | Area | Dawn test files (port targets) | Exit criteria |
|---|---|---|---|
| **0** | Scaffold + FFI + harness | — | bindgen builds; Noop device; `assert_device_error!` works; `wgpuCreateInstance`/`Release` round-trip green; CI green |
| **1** | Instance/Adapter/Device + Future | `DeviceValidationTests`, `UnsafeAPIValidationTests`, adapter/device init | Async model (Future, callback-info, WaitAny, ProcessEvents) working on Noop |
| **2** | Buffer + Queue | `BufferValidationTests`, `QueueWriteBufferValidationTests`, `QueueSubmitValidationTests`, `QueueOnSubmittedWorkDoneValidationTests` | create/map/unmap/mappedAtCreation, usage rules, submit |
| **3** | Texture/View/Sampler | `TextureValidationTests`, `TextureViewValidationTests`, `SamplerValidationTests` | format/dimension/usage validation |
| **4** | Shader + BindGroup(Layout) + PipelineLayout | `ShaderModuleValidationTests`, `BindGroupValidationTests`, `BindGroupLayout*`, `PipelineLayout*` | Tint (Dawn's WGSL compiler) WGSL parse/validate; binding validation |
| **5** | Render/Compute pipeline | `RenderPipelineValidationTests`, `ComputeValidationTests`, `PipelineAndPassCompatibilityTests` | pipeline + layout/format compat |
| **6** | Command encoding / passes | `CommandBufferValidationTests`, `CopyCommandsValidationTests`, `RenderPassDescriptorValidationTests`, `ComputeDispatch*` | encoder/pass state machine |
| **7** | Real backends | Dawn `end2end` Basic/Compute/Copy (GPU-gated) | Noop→**Metal→Vulkan** bring-up; real draw/dispatch. *(Order reversed from Vulkan→Metal: dev platform is macOS — Metal native, no MoltenVK; see `blocks/60-real-backends.md` / `tracking/phase-7.md`.)* |
| **8** | Surface/Query/ErrorScope/DeviceLost | `QuerySetValidationTests`, `ErrorScopeValidationTests`, `DeviceLostValidationTests`, surface config | remaining API surface |
| **9** | Examples + real surface/presentation | Dawn `samples/` + wgpu-native `examples/` (C, webgpu.h) | C samples ported; SF3 real window→swapchain on Metal/Vulkan. *(Post-core, user-requested; see `blocks/80-examples.md` / `tracking/phase-9.md`.)* |
| **13** | Shader passthrough (vendor) | — | **REMOVED (Tint migration Phase 0, 2026-06-26.)** The `shader-passthrough` vendor extension (raw SPIR-V / MSL `WGPUShaderModule`) was deleted from yawgpu. Retained as historical record. *(See `blocks/33-shader-passthrough.md` / `tracking/tint-migration-plan.md`.)* |
| **14** | Tiled rendering / TBDR (vendor) | — | **REMOVED (Tint migration Phase 0, 2026-06-26.)** The `tiled` vendor extension (transient attachments, multi-subpass passes, framebuffer fetch, subpass pipelines) was deleted from yawgpu. Retained as historical record. *(See `blocks/55-tiled-rendering.md` / `tracking/tint-migration-plan.md`.)* |
| **15** | GLES backend — **Tier 2 / experimental** | Reuses Phase 7 e2e ports under `--features gles` (basic/buffer/copy/compute_dispatch/render); no new Dawn-derived tests | Android (native EGL) + Windows (ANGLE default, opt-in WGL fallback via `YAWGPU_GLES_BACKEND=wgl` or the `YaWGPUGlesContextBackend` chain entry); opt-in `gles` cargo feature (default off); webgpu.h paths that do not cleanly map to GLES 3.1 may be rejected at HAL (no core-rule relaxation); minimum e2e set verified end-to-end on host NVIDIA driver via WGL fallback (`examples/triangle` runs clean), unsupported areas catalogued in the block 67 mapping matrix. *(See `blocks/67-gles-backend.md` / `tracking/phase-15.md`.)* |
| **16** | WebGPU CTS conformance port (validation first) | WebGPU CTS (`gpuweb/cts`) `src/webgpu/api/validation/` (129 spec files / 704 `g.test()`); operation + shader CTS deferred | CTS validation cases re-expressed as Noop integration tests under `tests/cts/validation/`, CTS dir layout mirrored, reconciling the legacy Dawn ports; harness extended (`expect_no_validation_error`, cartesian subcases, feature/limit-gated device, future-poll). *(See `blocks/91-cts-conformance.md` / `tracking/cts-coverage.md` + `tracking/cts-phase-a.md`.)* |

> Phases 10–12 (unit-test coverage, modularization, Win32 surface) were
> post-core, user-requested, and are tracked in `tracking/` only.

## Out of scope

- **DirectX (D3D11 / D3D12) backends.** Permanently out of scope.
  (GLES is **Tier 2 / experimental**, brought up in Phase 15 — see
  `blocks/67-gles-backend.md`. Tier 1 = Vulkan + Metal. The HAL enum
  is open to additional variants but no D3D variant is planned.)
- Dawn `wire/` (dawn-wire IPC) — no yawgpu analog.
- Multi-threading correctness beyond what ported tests require (revisit later).

## Block specs

Per-area rules extracted from Dawn live in `blocks/`. Stubs created at
planning time; filled at the start of each phase.
