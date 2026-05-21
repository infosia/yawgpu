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
| **4** | Shader + BindGroup(Layout) + PipelineLayout | `ShaderModuleValidationTests`, `BindGroupValidationTests`, `BindGroupLayout*`, `PipelineLayout*` | naga (`wgpu/naga`) WGSL parse/validate; binding validation |
| **5** | Render/Compute pipeline | `RenderPipelineValidationTests`, `ComputeValidationTests`, `PipelineAndPassCompatibilityTests` | pipeline + layout/format compat |
| **6** | Command encoding / passes | `CommandBufferValidationTests`, `CopyCommandsValidationTests`, `RenderPassDescriptorValidationTests`, `ComputeDispatch*` | encoder/pass state machine |
| **7** | Real backends | Dawn `end2end` Basic/Compute/Copy (GPU-gated) | Noop→**Metal→Vulkan** bring-up; real draw/dispatch. *(Order reversed from Vulkan→Metal: dev platform is macOS — Metal native, no MoltenVK; see `blocks/60-real-backends.md` / `tracking/phase-7.md`.)* |
| **8** | Surface/Query/ErrorScope/DeviceLost | `QuerySetValidationTests`, `ErrorScopeValidationTests`, `DeviceLostValidationTests`, surface config | remaining API surface |
| **9** | Examples + real surface/presentation | Dawn `samples/` + wgpu-native `examples/` (C, webgpu.h) | C samples ported; SF3 real window→swapchain on Metal/Vulkan. *(Post-core, user-requested; see `blocks/80-examples.md` / `tracking/phase-9.md`.)* |
| **13** | Shader passthrough (vendor) | — (vendor extension; direct unit tests + GPU-gated e2e) | Create `WGPUShaderModule` from raw SPIR-V (Vulkan) / MSL (Metal), feature `shader-passthrough` (default off), no API breakage. *(See `blocks/33-shader-passthrough.md` / `tracking/phase-13.md`.)* |
| **14** | Tiled rendering / TBDR (vendor) | — (vendor extension; direct unit tests + GPU-gated e2e) | Transient attachments, multi-subpass passes, framebuffer fetch, subpass pipelines (+ tile-dispatch scaffold), feature `tiled` (default off), Vulkan+Metal only, no API breakage. *(See `blocks/55-tiled-rendering.md` / `tracking/phase-14.md`.)* |

> Phases 10–12 (unit-test coverage, modularization, Win32 surface) were
> post-core, user-requested, and are tracked in `tracking/` only.

## Out of scope

- **OpenGL / OpenGL ES and DirectX (D3D11/D3D12) backends.** Primary
  platforms are **Vulkan and Metal** only; no GL/D3D backend in the initial
  implementation (the HAL enum may gain variants later, but none planned).
- Dawn `wire/` (dawn-wire IPC) — no yawgpu analog.
- Multi-threading correctness beyond what ported tests require (revisit later).

## Block specs

Per-area rules extracted from Dawn live in `blocks/`. Stubs created at
planning time; filled at the start of each phase.
