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
`tracking/phase-N.md`.

| Phase | Area | Dawn test files (port targets) | Exit criteria |
|---|---|---|---|
| **0** | Scaffold + FFI + harness | — | bindgen builds; Noop device; `assert_device_error!` works; `wgpuCreateInstance`/`Release` round-trip green; CI green |
| **1** | Instance/Adapter/Device + Future | `DeviceValidationTests`, `UnsafeAPIValidationTests`, adapter/device init | Async model (Future, callback-info, WaitAny, ProcessEvents) working on Noop |
| **2** | Buffer + Queue | `BufferValidationTests`, `QueueWriteBufferValidationTests`, `QueueSubmitValidationTests`, `QueueOnSubmittedWorkDoneValidationTests` | create/map/unmap/mappedAtCreation, usage rules, submit |
| **3** | Texture/View/Sampler | `TextureValidationTests`, `TextureViewValidationTests`, `SamplerValidationTests` | format/dimension/usage validation |
| **4** | Shader + BindGroup(Layout) + PipelineLayout | `ShaderModuleValidationTests`, `BindGroupValidationTests`, `BindGroupLayout*`, `PipelineLayout*` | naga (`wgpu/naga`) WGSL parse/validate; binding validation |
| **5** | Render/Compute pipeline | `RenderPipelineValidationTests`, `ComputeValidationTests`, `PipelineAndPassCompatibilityTests` | pipeline + layout/format compat |
| **6** | Command encoding / passes | `CommandBufferValidationTests`, `CopyCommandsValidationTests`, `RenderPassDescriptorValidationTests`, `ComputeDispatch*` | encoder/pass state machine |
| **7** | Real backends | Dawn `end2end` Basic/Compute/Copy (GPU-gated) | Noop→Vulkan→Metal bring-up; real draw/dispatch |
| **8** | Surface/Query/ErrorScope/DeviceLost | `QuerySetValidationTests`, `ErrorScopeValidationTests`, `DeviceLostValidationTests`, surface config | remaining API surface |

## Out of scope

- **OpenGL / OpenGL ES and DirectX (D3D11/D3D12) backends.** Primary
  platforms are **Vulkan and Metal** only; no GL/D3D backend in the initial
  implementation (the HAL enum may gain variants later, but none planned).
- Dawn `wire/` (dawn-wire IPC) — no yawgpu analog.
- Multi-threading correctness beyond what ported tests require (revisit later).

## Block specs

Per-area rules extracted from Dawn live in `blocks/`. Stubs created at
planning time; filled at the start of each phase.
