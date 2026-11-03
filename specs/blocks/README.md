# Block specs

One file per API area, authored by **Claude** (the coding agent does not
edit `specs/`). Filled at the **start of its phase** by reading the
corresponding Dawn validation test(s) and extracting the rules under test.
Each rule gets an ID (R1, R2, …) so task handoffs can reference
"rules R1..Rn each exercised by a test" as acceptance criteria.
Format per block:

- **Surface**: `webgpu.h` functions/structs covered.
- **Rules**: enumerated validation rules (each traceable to a Dawn test case).
- **Async**: futures/callbacks involved, Noop completion behaviour.
- **Open questions**.

The bootstrap phases (Phases 0 to 16, tracked in `../SPEC.md`) are blocks 00, 10, 20,
30, 40, 50, 60, 70 (Phases 0 to 8), 33, 55, 36, and 91 (Phases 13 to 16). The other blocks are
one file per feature, campaign, or fix. Each carries its own `Status:` line and, where
it has one, links its `../tracking/<topic>.md` ledger. Add a row here when a block
file is created.

| File | Topic |
|---|---|
| `00-foundation.md` | Foundation (instance / adapter / device / future / error sink) |
| `10-buffer-queue.md` | Buffer + Queue |
| `20-texture-sampler.md` | Texture / TextureView / Sampler |
| `30-shader-binding.md` | ShaderModule / BindGroupLayout / BindGroup / PipelineLayout |
| `33-shader-passthrough.md` | Shader passthrough (SPIR-V / MSL) |
| `36-external-textures.md` | External textures (`texture_external`) |
| `40-pipeline.md` | Render / Compute Pipeline |
| `50-commands.md` | Command encoding & passes |
| `55-tiled-rendering.md` | Tiled rendering (TBDR mobile extension) |
| `60-real-backends.md` | Real backends (Phase 7) |
| `62-subgroups.md` | `subgroups` optional feature |
| `63-depth-clip-control.md` | `depth-clip-control` optional feature |
| `64-float32-blendable.md` | `float32-blendable` optional feature |
| `65-dual-source-blending.md` | `dual-source-blending` optional feature |
| `66-indirect-first-instance.md` | `indirect-first-instance` optional feature |
| `67-gles-backend.md` | GLES backend (Phase 15, Tier 2 / experimental) |
| `68-clip-distances.md` | `clip-distances` optional feature |
| `69-primitive-index.md` | `primitive-index` optional feature |
| `70-finalize.md` | Finalize: Query / ErrorScope / DeviceLost / Toggle / Surface / MultipleDevice |
| `71-texture-component-swizzle.md` | `texture-component-swizzle` optional feature |
| `72-texture-formats-tier2.md` | `texture-formats-tier2` advertisement is a device query |
| `73-texture-compression-vulkan.md` | Vulkan compressed-texture completion |
| `80-examples.md` | Examples + real surface/presentation (Phase 9) |
| `85-windows-surface.md` | Windows windowed examples (HWND surface + Win32 windowing) |
| `90-unit-tests.md` | Public-API unit-test policy (Phase 10+ ongoing) |
| `91-cts-conformance.md` | WebGPU CTS conformance porting (validation first) |
| `92-adapter-limits.md` | real hardware-queried adapter limits (Dawn/CTS parity) |
| `93-wgsl-language-features.md` | WGSL language features |
| `94-immediates.md` | Immediate data (`SetImmediates`, `maxImmediateSize > 0`) |
| `95-shader-compile-cache.md` | Shader-compile caching (FFI short-circuit + codegen memoization) |
| `96-queue-submission-perf.md` | Queue submission performance |
| `97-render-pass-command-stream.md` | Render pass command stream |
| `98-tint-build-integration.md` | Tint build integration (`yawgpu-tint/build.rs`) |
| `99-feature-advertisement-queries.md` | Feature advertisement follows Dawn's device queries |
| `100-encoder-write-buffer.md` | `wgpuCommandEncoderWriteBuffer` executes |
| `101-get-proc-address.md` | `wgpuGetProcAddress` |
| `102-timestamp-query.md` | `timestamp-query` executes (Metal + Vulkan) |
| `103-surface-capabilities.md` | Surface capabilities come from the HAL |
| `104-lazy-zero-init-stage2.md` | Lazy zero-initialization Stage 2 |
| `105-vulkan-subresource-layouts.md` | Vulkan image layouts are tracked per subresource |
| `106-native-vulkan-verifications.md` | Native-Vulkan verifications deferred from Blocks 68 / 71 / 94 (backlog D11, Vulkan items) |
| `107-vulkan-discard-epilogue-shared-layout.md` | Vulkan: drop the eager `storeOp: Discard` clear (backlog A8) and give sampled + storage-bound subresources one layout (backlog A9) |
| `108-subgroup-size-control.md` | `subgroup-size-control` on Vulkan (`@subgroup_size`) |
| `109-code-review-criticals.md` | whole-codebase review: CRITICAL fixes |
| `110-code-review-core-majors.md` | whole-codebase review: core / FFI validation MAJORs |
| `111-code-review-hal-majors.md` | whole-codebase review: HAL / Tint MAJORs |
| `112-code-review-submit-and-aliasing.md` | whole-codebase review: queue submit serialization + writable-binding aliasing |
| `113-class-audit-core-ffi.md` | bug-class audit: core / FFI |
| `114-class-audit-vulkan.md` | bug-class audit: Vulkan |
| `115-class-audit-metal-gles-tiled.md` | bug-class audit: Metal / GLES / tiled |
| `116-tint-single-lowering.md` | Tint: one IR lowering per resolved workgroup query |
| `117-vulkan-submit-recycling.md` | Vulkan: non-blocking retirement and recycled submit objects |
| `118-dawn-v20261030-bump.md` | Dawn bump to release v20261030.184100 |
| `119-cts-d116bcc-findings.md` | CTS d116bcc findings F-161 to F-164 |
| `120-tint-driver-workarounds.md` | Tint driver workarounds |
| `121-work-done-pending-writes.md` | `onSubmittedWorkDone` does not flush pending writes |
| `122-metal-blit-coalescing.md` | Metal: one blit encoder per run of copies |
| `123-metal-msl-math-mode.md` | Metal: relaxed MSL math mode, as Dawn |
| `124-metal-preserve-invariance.md` | Metal: `preserveInvariance` only for `@invariant`, as Dawn |
| `125-tint-option-parity.md` | Tint writer and pipeline flags: Dawn parity, part 1 |
| `126-vulkan-frag-depth-clamp.md` | Vulkan: clamp `frag_depth` in the shader, as Dawn |
| `127-vulkan-subgroup-parity.md` | Vulkan subgroup setup: Dawn parity |
| `128-vulkan-robustness2.md` | Vulkan robustness2 and pipeline robustness, as Dawn |
| `129-vulkan-buffer-zero-init.md` | Vulkan buffers read as zero |
| `130-dropped-pass-and-spontaneous-device-lost.md` | Dropped pass lowering and spontaneous device-lost delivery |
| `131-dawn-vendor-gated-tint-workarounds.md` | Dawn's vendor-gated Tint workarounds |
| `132-render-bind-groups-at-draw.md` | Render bind groups are resolved at draw time |
| `133-vulkan-spirv-1-4.md` | Vulkan SPIR-V 1.4, as Dawn |
| `134-vulkan-limits-parity.md` | Vulkan adapter limits: NVIDIA storage cap and Dawn's remaining mappings |
| `135-adapter-info-parity.md` | `WGPUAdapterInfo` reports the GPU, as Dawn |
| `136-msvc-tint-optimization.md` | Optimize the Tint shim on MSVC release builds |
| `137-spontaneous-ready-callbacks.md` | Spontaneous callbacks that are ready when registered |

Vendor-extension blocks (33, 55) are **not** Dawn ports — their rules are
exercised by direct unit tests (principle 1) + GPU-gated e2e, and they live
behind opt-in cargo features (`shader-passthrough`, `tiled`; default off).
External textures (36) are a **hybrid**: the binding model is core WebGPU
(always on, no feature gate), only `WGPUExternalTexture` *creation* is a vendor
extension (canonical `webgpu.h` leaves creation impl-defined).
