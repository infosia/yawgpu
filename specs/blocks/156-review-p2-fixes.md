# Block 156 — Whole-code review 2026-11-13: P2 library fixes

Status: **COMPLETE (2026-11-13)** apart from the Linux GLES hardware run of S3 (owed). Phase Review: 0 CRITICAL, 1 MAJOR (fixed in S3), 3 MINOR (fixed in S3).
Source: whole-code review at `92e38c7` (`specs/tracking/code-review-2026-11.md`), findings R13, R14, and R11.
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

R4. **GLES executes the newly exposed baseline `rg11b10ufloat` (Phase Review M1).** R1
lets a GLES device without `EXT_color_buffer_float` create a sampled/copyable
`rg11b10ufloat` texture. Its lazy clear and texture→buffer readback must not need a
color-renderable framebuffer: the format joins `texture_to_buffer_compute_encoding` (as
`rgb9e5ufloat` does), so the clear takes the zero-upload path (`R11F_G11F_B10F` / `RGB` /
`UNSIGNED_INT_10F_11F_11F_REV`, all-zero bits = 0.0) and readback packs the texel in the
compute shader. The FBO clear path also deletes its framebuffer when attaching a layer
fails (it returned with `?` and leaked it).

R5. **Phase Review MINORs.** (m1) Blendable is part of the `Rg11b10UfloatRenderable`
upgrade, not the baseline (Dawn `Format.cpp` baseline `Cap::None`). (m2) The GLES sampler
pre-creation error drain is capped (16 iterations). (m3) `map_sampler_error` documents why
`VK_ERROR_TOO_MANY_OBJECTS` is `Internal`. Block 157 m1: `submit_texture_clear_zero_upload`
checks `glGetError` once after the upload loop and returns a `HalError` on failure.

## Slices

S1: R1–R2. S2: R3. S3 (Phase Review fixes): R4, R5.

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

## Verification record

S1+S2: workspace test 1,476/0; `yawgpu-core --lib` 735/0 (the two R1 tests red before the
fix); clippy clean for `yawgpu-hal` with `metal`, `vulkan`, and `gles`. Real GPU on the
M2: `yawgpu-hal --features metal --lib --include-ignored` 267/0; every `e2e_metal_*`
170/0; `yawgpu-hal --features vulkan --lib --include-ignored` 460/0 on MoltenVK under the
Khronos layer, 0 VUIDs, including `vulkan_queue_submit_empty_returns_before_completion_is_observed`;
every `e2e_vulkan_*` 163 passed, 1 failed (the known `imageView2DOn3DImage` case, the
source of every VUID). R2 audit: no production caller of `submit_empty` relied on the
wait. R3 (GLES): EGL and WGL run sampler creation synchronously on the current context,
so the deferred error path was removed entirely; `GL_OUT_OF_MEMORY` maps to
`OutOfMemory`, other GL errors to `Internal`.

## Phase Review (2026-11-13, fresh-context reviewer over `f4b59a3`)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| M1 | MAJOR | R1 exposes baseline `rg11b10ufloat` on GLES without `EXT_color_buffer_float`, where its lazy clear (FBO path) and readback fail | S3: R4 |
| m1 | MINOR | Baseline is blendable; Dawn grants blendable only with the renderable feature | S3: R5 |
| m2 | MINOR | Unbounded GLES error-drain loop before sampler creation | S3: R5 |
| m3 | MINOR | `VK_ERROR_TOO_MANY_OBJECTS` maps to `Internal` without a comment | S3: R5 |

R2 (no production caller relied on the empty-submit wait) and R3 (no leaked native
samplers, error sampler semantics) were confirmed.

S3: `rg11b10ufloat` has a GLES compute encoding (R11G11B10 bit packing in GLSL, checked
exhaustively over finite channel values; the three generated shader variants pass
`glslangValidator`), so its lazy clear takes the zero-upload path and its readback the
compute path; the FBO clear deletes its framebuffer on an attach failure; blendable moved
into the renderable upgrade; the sampler error drain is capped at 16; the zero upload
checks `glGetError` after its loop. `yawgpu-core --lib` 737/0; `yawgpu-hal --features gles
--lib` 273/0 (hardware bodies skip without libEGL); Metal HAL incl. ignored 267/0. The new
GLES hardware test `submit_copies_rg11b10_clears_and_reads_back_via_compute` is owed to
the Linux EGL host.
