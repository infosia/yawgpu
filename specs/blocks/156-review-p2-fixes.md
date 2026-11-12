# Block 156 — Whole-code review 2026-11-13: P2 library fixes

Status: **PLANNED (2026-11-13)**.
Source: whole-code review at `92e38c7` (`REPORT-review.md`), findings R13, R14, and R11.
(R10 and R12 went into Block 151; R15, X01, and X02 are Claude-owned test and example
fixes; R16, GLES, is Block 157.)
Oracle: pinned Dawn — `native/Format.cpp` (`RG11B10Ufloat` baseline caps and the
`RG11B10UfloatRenderable` upgrade); `native/Device.cpp` (`CreateSampler` reports a backend
failure at creation).

## Rules

R1. **`rg11b10ufloat` is a core format (R13).** `TextureFormat::RG11B10_UFLOAT.caps()`
always returns its baseline capabilities: float color, 3 components, 4-byte texel,
filterable, blendable, copyable, sampled. Without `Rg11b10UfloatRenderable` it is not
renderable and not multisample-capable; with the feature, the existing upgrade
(`format.rs` ~773) makes it renderable and multisample-capable. Storage rules stay as
they are. Any table or test that encoded the old "unsupported without the feature"
behaviour is updated to the core behaviour.

R2. **An empty Vulkan submit does not wait (R14).** `VulkanQueue::submit_empty` returns the
submission index without `wait_for_submission`. Completion goes through the usual
retirement / poll / wait paths. The empty submission still orders after all earlier
queue work (one `vkQueueSubmit` on the same queue). The agent checks every caller of
`submit_empty` (core `Queue::submit` with no command buffers, on-submitted-work-done,
map paths) and reports any that relied on the wait.

R3. **Sampler creation reports backend failure at creation (R11).**
`HalDevice::create_sampler` returns `Result<HalSampler, HalError>`. Metal (`nil` sampler)
and Vulkan (`vkCreateSampler` error) return `HalError::OutOfMemory` for an
allocation failure (`VK_ERROR_OUT_OF_HOST_MEMORY`, `VK_ERROR_OUT_OF_DEVICE_MEMORY`,
Metal `nil`) and another `HalError` otherwise; they no longer store a missing native
sampler. GLES returns its creation error immediately where it can detect one at creation
and keeps its deferred path only where GL objects must be created on the context thread
(document which in the GLES mapping matrix, `67-gles-backend.md`, is Claude's job; the
agent reports it). `Device::create_sampler_with_error` maps the error through
`dispatch_hal_allocation_error` and returns an error sampler.

## Slices

S1: R1–R2. S2: R3.

## Verification

- Inline tests: R1 (`RG11B10_UFLOAT.caps(&empty)` is `Some`, sampled/filterable, not
  renderable; with the feature renderable + multisample; texture creation with
  `TEXTURE_BINDING | COPY_DST` succeeds without the feature and `RENDER_ATTACHMENT`
  fails); R2 (Vulkan-gated unit test or HAL `#[ignore]` test that `submit_empty` returns
  before completion is observed, if it can be made deterministic; otherwise review);
  R3 (Noop/injected HAL failure → `OutOfMemory`/`Internal` error, error sampler).
- Agent gates: `cargo test -p yawgpu-core --lib`, `cargo test -p yawgpu --lib`, the
  format-capability integration tests, clippy workspace, plus
  `--features metal`, `--features vulkan`, and `--features gles` clippy for S2.
- Claude: workspace test; feature-gated HAL tests; `e2e_vulkan_*` under the validation
  layer (S1), `e2e_metal_*` sampler suites (S2), and a CTS
  `webgpu:api,validation,texture,*` / `format_info` slice on Metal for R1.
