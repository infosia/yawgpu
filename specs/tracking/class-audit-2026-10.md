# Bug-class audit — 2026-10-25

After Blocks 109–112 the user asked for a focused audit for **more
instances of the bug classes those blocks fixed**. Six fresh read-only
auditors, one per class, at `7e82a6d`; each was given the fixing commits as
worked examples. Claude re-verified the items marked **v** against the
code. IDs are referenced by Blocks 113–115.

Classes: **F** validated-but-ignored HAL fields · **S** GPU sync / object
lifetime · **C** concurrency / TOCTOU / callbacks · **E** spec edge values,
defaults, ordering · **T** test-masked / CTS-shaped code · **D** over- /
under-strict vs Dawn.

## Core / FFI → Block 113

| ID | Sev | Finding | |
|---|---|---|---|
| E1 | MAJOR | "override without default must be set" checked over the **whole module**, not the stage's entry point (`compute_pipeline.rs` `resolve_pipeline_constants`, callers pass `module.overrides()`); a VS+FS module with an FS-only override fails the vertex stage. Shim has no per-entry-point override list (Dawn `ShaderModule.cpp` `uninitializedOverrides`) | v |
| E2 | MAJOR | invalid enum values still silently defaulted: address / filter / mipmap filter mode, texture view dimension, texture aspect (`unwrap_or(All)` in copies / writeTexture), load / store op | v |
| E3 | MAJOR | invalid `WGPUVertexFormat` passes validation (fails later in the HAL) | |
| D4/E4 | MAJOR | alphaToCoverage: yawgpu requires *some* blendable target with alpha; Dawn requires `targets[0]` present with alpha (blendability irrelevant) and a fragment state | |
| D1 | MAJOR | view of a multisampled texture may be `2DArray` (Dawn: only `2D`) | |
| D2 | MAJOR | resolve target may be a 3D view (Dawn: 2D / 2DArray only) | |
| D3 | MAJOR | 3D texture copy onto itself within one mip accepted when z ranges disjoint (Dawn rejects; one subresource); 1D self copy and zero-size self copy accepted | |
| D5 | MINOR | zero-size partial depth/stencil copy skips the full-subresource rule | |
| D6 | MINOR | `copyBufferToBuffer(b, …, b, …, 0)` accepted (Dawn rejects source == destination unconditionally) | |
| D7 | MINOR | single-aspect view may keep the combined DS format; default view format ignores the aspect | |
| E5 | MINOR | non-UTF-8 WGSL becomes an empty valid module | |
| E6 | MINOR | non-UTF-8 entry-point name treated as "not given" | |
| E7 | MINOR | fractional values for i32/u32 overrides rejected (Dawn: range check only, truncation) | |
| E8 | MINOR | numeric constant keys parsed loosely (`"01"`, `"+1"` match `@id(1)`, dodge the duplicate check) | |
| E9 | MINOR | unknown BGL visibility bits not rejected | |
| E10 | MINOR | null array + non-zero count: `viewFormats`, `requiredFeatures` silently empty; `dynamicOffsets` / `ExecuteBundles` `assert!` aborts across `extern "C"` | v |
| E11 | MINOR | invalid `WGPUCallbackMode` (incl. 0) treated as WaitAnyOnly; `forceFallbackAdapter` ignored; invalid `featureLevel` → Core | |
| E12 | MINOR | `wgpuTextureGetTextureBindingViewDimension` returns computed / unvalidated values in core mode (Dawn: Undefined) | |
| C1 | MAJOR | `writeBuffer` races `mapAsync`: `write_buffer` does not take the submission lock, so a map can resolve without the write and the write later lands in a mapped buffer | v |
| C2 | MAJOR | `wgpuDeviceGetLostFuture` check-then-push race: future never completes | |
| C3 | MAJOR | device-lost callback may receive a freed `WGPUDevice` (Dawn passes null without an external ref) | |
| C4 | MAJOR | pending callbacks never get `CallbackCancelled` on instance drop; pending results hold Arc cycles (leak) | |
| C5 | MAJOR | error scopes are per-device, webgpu.h / Dawn: per-thread | |
| C6 | MAJOR | no ValidateIsAlive: mapAsync succeeds, submit / writeBuffer reach the HAL, errors keep dispatching after device loss | |
| C7 | MINOR | `Queue::wait_idle` reads the submission index after waiting → may recycle in-flight staging | |
| C8 | MINOR | `unmap` publishes Unmapped before the host write-back (host-fallback backends) | |
| C9 | MINOR | work-done callbacks report Error on device loss (Dawn: Success) | |
| C10 | MINOR | `wgpuSurfaceConfigure` dispatches the error callback while holding surface locks; clears state before the fallible configure | |
| C11 | MINOR | uncaptured callback may still run after `wgpuDeviceRelease` returns | |
| S5 | MAJOR | `wgpuSurfacePresent` does not flush pending queue writes (Dawn submits pending commands in present) | |
| S10 | MINOR | `onSubmittedWorkDone` ignores pending writes (no flush) | |
| T6 | MINOR | `writeTexture` into depth16unorm / stencil8 with a non-multiple-of-4 offset is not repacked; Vulkan requires 4-byte buffer offsets for DS images | |

## Vulkan → Block 114

| ID | Sev | Finding | |
|---|---|---|---|
| S2 | MAJOR | no barrier ever includes `UNIFORM_READ` (grep: zero uses) — `writeBuffer(uniforms)` → draw / dispatch has no visibility | v |
| S1 | MAJOR | no read → write (WAR) buffer barriers: render-pass vertex / index / indirect / uniform / storage reads and copy-source reads are never waited on before a later copy / fill / compute / storage write | |
| S3 | MAJOR | a submit waits on the swapchain acquire semaphore only if the swapchain image is a color / resolve target or copy endpoint — not for compute storage writes, lazy-init clears, bind-group use, or any tiled subpass (Dawn: the next submit always waits) | v |
| S6 | MAJOR (error path) | `pending.consumed = true` still set before `vkCreateFence` / `vkQueueSubmit`; a failure leaves present waiting on a never-signalled semaphore | v |
| S4 | MAJOR (tiled) | `subpass_dependencies` incoming src access still empty (Block 109 fixed only the regular path) | |
| F1 | MAJOR | 2D-array → 3D `copyTextureToTexture`: `extent.depth = 1` (source extent), only one slice copied (Dawn: depth = layer count) | v |
| F2 | MAJOR | `viewFormats` never reach the HAL; images lack `MUTABLE_FORMAT` + format list, so srgb views of unorm textures are invalid usage | v |
| T2 | MAJOR | swapchain ignores `viewFormats` (no mutable-format swapchain) | |
| T1 | MAJOR | swapchain built at `currentExtent`, ignoring the configured size while core uses the configured size (Dawn: configured size, blit when outside `[min,max]`) | v |
| T3 | MAJOR | acquire / present never report Outdated / Suboptimal; every failure → Error | |
| T4 | MAJOR | `shader-f16` advertised without Dawn's `shaderInt16` / `storageBuffer16BitAccess` gate; shim never clears `use_storage_input_output_16` when `storageInputOutput16` is unsupported | |
| S7 | MINOR | host-read buffers never get a `HOST_READ` barrier (coherent memory masks it) | |
| S9 | MINOR | present-ready semaphore indexed by sync slot, reused before the previous present's wait completes (index by image) | |
| T7 | MINOR | swapchain `pre_transform = currentTransform` (Dawn forces identity) | |

## Metal / GLES / tiled → Block 115

| ID | Sev | Finding | |
|---|---|---|---|
| T5 | MAJOR | Metal compressed-texture copies not clamped to the mip's real size (Vulkan clamps; Dawn `UtilsMetal.mm`) | |
| T8 | MINOR | Metal buffer↔texture copies never split rows wider than Metal's per-row limit; no `bytesPerImage = 0` / last-row handling | |
| S8 | MINOR | Metal `nextDrawable` without autoreleasepool | |
| F5 | MAJOR (T2) | GLES binds 3D storage textures as a single 2D layer (`full_layer_count` uses the 3D depth) | |
| F6 | MAJOR (T2) | GLES render attachments ignore `view_format` / `resolve_view_format` | |
| F8 | MINOR (T2) | GLES never clamps `frag_depth` to the viewport depth range (uncatalogued) | |
| F3 | MAJOR (tiled) | subpass attachments lose the view's base mip / layer (and format on Vulkan) | |
| F4 | MAJOR (tiled) | Vulkan ignores subpass color `resolve_target` | |
| F7 | MINOR (tiled) | core accepts subpass `setIndexBuffer` / `drawIndexed`, HAL has no index-buffer field | |

Checked clean by the auditors (one line each in their reports): regular
render-pass attachment subresources and view formats on Metal/Vulkan,
copies' aspect / origin / strides, samplers, pipeline state, draw inputs,
queries, immediates; GLES memory barriers; Metal tracked resources; Vulkan
retire ring / layout overlay; WHOLE_SIZE and other sentinels; per-entry-point
Tint reflection except overrides; 60+ CTS-motivated sites judged correct.
