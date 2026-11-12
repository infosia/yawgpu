# Whole-code review — 2026-11-13

Reviewed revision: `92e38c7410ff8774717e54ed79c8512d29a8d902`.

This report records issues not already covered by the remaining-work inventory, with priority, estimated difficulty, source evidence, a triggering scenario, and a proposed correction. It contains **16 findings and 2 smaller example improvements**. These are static-review findings: no tests, builds, CTS runs, GPU probes, or reproduction programs were executed. No implementation changes were made.

## Scope and method

The review covers the complete bodies of tracked, first-party implementation files, inline and integration tests, shader sources, build scripts, manifests, and CI configuration. Review was divided among agents and checked against the tracked-file inventory; unsupported completion claims in intermediate coverage notes were withdrawn and the affected files reread.

Full-line review and inventory reconciliation are complete for all 328 files below, including the final reread of the three core render files. Coverage means that source bodies were inspected, not merely searched or counted; it does not imply that every defect has been found.

| Area | Files | Lines, including comments and tests |
|---|---:|---:|
| `yawgpu` — FFI, conversions, custom header, build files, integration tests | 181 | 97,771 |
| `yawgpu-core` | 36 | 65,236 |
| `yawgpu-hal` — common, Noop, Metal, Vulkan, GLES | 53 | 61,396 |
| `yawgpu-tint` — Rust wrapper, C++ shim, build support and tests | 8 | 13,097 |
| `yawgpu-test` | 2 | 645 |
| `examples` | 39 | 5,113 |
| `benches` | 5 | 1,120 |
| `tools` | 2 | 128 |
| Workspace manifest and CI | 2 | 63 |
| **Total** | **328** | **244,569** |

Excluded from full-line coverage: vendored `third_party` code, the upstream `webgpu.h`, `stb_image_write.h`, generated SPIR-V binaries, build output, dependency lockfile contents, and untracked local files. The small checked-in `gpu_info_data.rs` table is included. Repository prose was consulted for contracts, previous findings, and backlog overlap; it is not part of the code line count. Relevant parts of the pinned Dawn source were inspected as the repository's compatibility reference, without claiming a full review of Dawn.

The principal overlap reference is [backlog.md](backlog.md), including its historical closure notes and linked blocks, plus [the previous code review](code-review-2026-10.md). A31 is already closed by Block 150 despite its stale table row. A21 is explicitly deferred by user decision. Neither is proposed again.

## Priority and difficulty

P1 means a memory-safety problem, process termination, invalid native resource use, initialization failure, or a substantial validation omission. P2 means a narrower correctness, error-reporting, performance, or verification problem. Priority describes impact, not proof of exploitation or a measured frequency.

S = localized change, usually under a day; M = several days or changes across layers; L = a week or more. Estimates include the regression checks appropriate when implementing a fix, although none were run during this review. GLES remains last in recommended order, following the repository's stated prioritization.

| ID | Priority | Difficulty | Finding |
|---|---|---|---|
| R01 | P1 | S–M | FFI input handles can be freed by a reentrant error callback before their last use |
| R02 | P1 | S | Invalid texture descriptors still allocate initialization tracking from unchecked dimensions |
| R03 | P1 | S | A public safe Rust converter dereferences caller-controlled chain pointers |
| R04 | P1 | S | External texture planes are not checked against the creating device |
| R05 | P1 | M | Acquired surface textures remain usable after presentation or reconfiguration |
| R06 | P1 | M | Failed queued texture writes leave initialization marks committed |
| R07 | P1 | M | Vulkan cube views lack the necessary image flags and cube-array feature enablement |
| R08 | P1 | M | Subpass draws omit buffer-range and strip-index-format validation |
| R09 | P1 | S | Metal occlusion query allocation can succeed with a missing native buffer |
| R10 | P2 | S | External texture plane component counts are not validated |
| R11 | P2 | M | Sampler allocation errors are swallowed or deferred until use |
| R12 | P2 | S–M | Query-set allocation failures are reported as validation errors |
| R13 | P2 | S | `rg11b10ufloat` sampling/copy support is incorrectly gated on its renderability feature |
| R14 | P2 | S | Empty Vulkan submissions synchronously wait for GPU completion |
| R15 | P2 | S | Several floating-point readback checks accept NaN as a passing result |
| R16 | P2 | M | GLES lazy clears allocate a complete mip's zero data in one host allocation |

The most accessible high-priority starting points are R02, R03, R04, and R09. R01 is also localized but should be fixed consistently across all callback-producing entry points. R09 and R12 belong together so the newly detected allocation failure reaches the correct error scope.

## Resolution ledger

| ID | Block | State |
|---|---|---|
| R01 | 152 | fixed (`16965af`, `665fe1f`) |
| R02, R03, R04, R09, R10, R12 | 151 | fixed (`1855795`) |
| R05 | 153 S2 | fixed (`665fe1f`) |
| R06 | 153 S1 | fixed (`d0839ee`) |
| R07 | 155 | fixed (`26a12ae`) |
| R08 | 154 | fixed (`e9d97c0`) |
| R11, R13, R14 | 156 | fixed (`f4b59a3`) |
| R15 | — | fixed in the e2e tests (`36c8e3c`, Claude-owned): every listed check also rejects a non-finite value |
| R16 | 157 | fixed (`3328d0d`); Linux GLES hardware run owed |
| X01, X02 | — | fixed in `examples/` (`20f4362`, Claude-owned) |

## Findings

### R01 — Retain every input handle used after an error callback

**Evidence:** `yawgpu/src/ffi/texture.rs:46–74` borrows the texture, dispatches a validation error, then clones fields from the borrowed object. The same lifetime pattern occurs in `ffi/encoder.rs:26–157`, `ffi/bundle.rs:377–397`, `ffi/tiled.rs:141–165`, and the pipeline bind-group-layout error paths in `ffi/mod.rs:2885–2947`. `ffi/device.rs:429–455` retains the device but only borrows the bind-group layout across `create_bind_group`, which may dispatch an error.

**Trigger and impact:** register an uncaptured-error callback that releases the last external reference to the input texture, encoder, bundle encoder, pipeline, or layout. Call an operation with a descriptor that triggers validation. When the callback returns, the entry point still accesses the freed FFI wrapper. Retaining an underlying core object does not retain that wrapper. This is a use-after-free risk in a valid reentrant release pattern.

**Correction:** clone the input FFI handle before an operation that may dispatch, or capture all required owned fields before dispatch and never access the borrowed wrapper afterward. Apply the rule to secondary descriptor handles as well as the primary receiver.

**Backlog distinction:** Block 113 M1 fixed device-handle retention at device entry points. These are the remaining non-device receivers and the separately borrowed layout; they are not the open multithreading item B7.

### R02 — Do not allocate initialization state for invalid/error textures

**Evidence:** `yawgpu-core/src/device.rs:437–470` constructs `Texture::new` with the original descriptor even after validation fails, the device is lost, or native allocation fails. `texture.rs:261–285` enables initialization tracking from usage alone, ignoring `is_error`. `texture.rs:582–607` multiplies the unvalidated mip/layer counts and executes `vec![false; len]`.

**Trigger and impact:** submit a 2D, 1×1, RGBA8, `COPY_DST` descriptor whose mip and array-layer counts are both `u32::MAX`. On a 64-bit host their product fits `usize` but exceeds the allocation capacity bound. Validation reports the bad descriptor, then constructing its error object can panic. Through an `extern "C"` entry point this can abort the process. Less extreme invalid dimensions can request gigabytes of host memory. The lost-device path has the same allocation issue.

**Correction:** create error textures with disabled/empty initialization tracking. Preserve descriptor values needed by getters without allocating per-subresource state. Only size real tracking from validated dimensions.

**Backlog distinction:** no corresponding item; this is allocation caused by rejected input, not a requested increase in texture limits or a backend OOM stress test.

### R03 — The texture-view converter's safe signature is unsound

**Evidence:** public `yawgpu::conv::map_texture_view_descriptor` at `yawgpu/src/conv/descriptors.rs:583` accepts `Option<&WGPUTextureViewDescriptor>`, then calls the unsafe chain traversal at line 600. The helper at lines 396–405 follows `nextInChain` through `find_in_chain`.

**Trigger and impact:** safe Rust can construct a native descriptor containing an arbitrary non-null raw `nextInChain` pointer, then pass a reference to this safe function. Holding a valid reference to the descriptor does not prove that its pointed-to chain is valid. The function can therefore cause undefined behavior without an unsafe caller operation.

**Correction:** make this API unsafe with an explicit chain-lifetime/validity contract and update its callers, or expose a safe owned representation that cannot contain unchecked chain pointers. A null check alone cannot make arbitrary non-null pointers safe.

**Backlog distinction:** independent of invalid C handles, which the FFI already documents as unsafe caller input. The defect is the public safe Rust boundary.

### R04 — Reject external texture planes from another device

**Evidence:** `yawgpu/src/ffi/external_texture.rs:29–110` clones the plane-view handles but does not compare their `_device` with the creating device. `yawgpu-core/src/external_texture.rs:123–166,311–403` validates the plane properties and creates the parameters buffer, without establishing plane-device ownership. Later external-texture binding validation checks the owner of the external texture wrapper, not each plane's original device.

**Trigger and impact:** create a view on device A, import it as an external texture on device B, and bind that external texture on B. The wrapper and parameter buffer belong to B while its native image belongs to A. This can send foreign native resources to the backend, with validation errors, device loss, or incorrect execution instead of a WebGPU validation error at creation.

**Correction:** validate every supplied plane handle's device before constructing the external texture. This can be done at the FFI boundary where ownership is already retained.

**Backlog distinction:** A27/A28 concerned initialization and subresource views, and B3 concerns GLES execution support. This ownership omission affects the supported backends. Pinned Dawn explicitly validates both plane objects in `native/ExternalTexture.cpp:71,113`.

### R05 — Expire surface textures when their acquisition ends

**Evidence:** `yawgpu/src/ffi/surface.rs:124–128,206–210,365–371` clears the cached `current_texture` on configure, unconfigure, and present without destroying/expiring its core texture. `Texture::destroy` in `yawgpu-core/src/texture.rs` is a separate explicit state transition. An application-held texture/view or encoded command retains its own reference.

**Trigger and impact:** hold the acquired texture or a view, present or reconfigure the surface, then use the old object in a later submission. The old core object still passes destroyed-resource checks. Vulkan can consequently receive operations on an image that has been presented, recycled, or detached from the old swapchain. Metal likewise accepts use beyond the acquisition's intended lifetime.

**Correction:** explicitly expire the acquisition's core texture at the appropriate lifecycle transition, so existing views and later submissions also fail validation. Preserve ordering for pending writes and already submitted work; do not simply drop the native image before those operations finish. Include surface teardown in the lifetime audit.

**Backlog distinction:** A11 intentionally permits discarded contents at presentation. This finding concerns use of a previous acquisition after presentation, not the contents presented. Dawn's swapchain lifecycle destroys/detaches its current texture.

### R06 — Commit queued-write initialization only when the write survives submission

**Evidence:** both `Queue::write_texture` paths commit an `InitOverlay` before queuing the copy (`yawgpu-core/src/queue.rs:698–710,756–768`). `flush_pending_writes_locked` takes the pending batch at lines 1021–1038. `finish_pending_submission` returns an error at lines 1045–1060 without restoring the batch or its initialization state. Ordinary command-buffer submission commits its overlay only on successful submission at line 986.

**Trigger and impact:** write to an uninitialized texture, then encounter a HAL failure while flushing that batch. The batch is dropped but the destination remains marked initialized. A later read/use skips its required zero clear even though the write never reached the GPU, potentially exposing stale contents or yielding incorrect pixels. The error path does not itself make the device unusable, so subsequent use matters.

**Correction:** retain initialization changes with pending writes and commit them on a successful HAL submission. Any approach that marks early needs a precise rollback policy that also handles overlapping queued writes. Keep the submission lock's existing ordering guarantees.

**Backlog distinction:** separate from the completed lazy-initialization work and Block 121's pending-write flushing; those do not make failed queued-write initialization transactional.

### R07 — Create Vulkan images and devices capable of the cube views core accepts

**Evidence:** `yawgpu-hal/src/vulkan/texture.rs:160–174` creates only the 3D-attachment and mutable-format flags, never `CUBE_COMPATIBLE`. The image creation path uses these flags. `vulkan/pipeline.rs:2121–2129` nevertheless lowers accepted views to `CUBE`/`CUBE_ARRAY`. Logical-device feature construction at `vulkan/mod.rs:1190–1237` does not enable `image_cube_array`.

**Trigger and impact:** sample a square 2D texture with six layers through a cube view, or a compatible array through a cube-array view. These accepted API operations produce native views without the required image creation flag; cube arrays additionally require the corresponding enabled device feature. The result is invalid native usage or backend failure. Existing scalar 2D copy/render tests do not establish this path's correctness.

**Correction:** add cube compatibility to eligible square, single-sampled 2D array images; query/enable the cube-array feature consistently with exposed capabilities. Keep restrictions for ineligible images. Pinned Dawn does this in `vulkan/TextureVk.cpp:1516` and `vulkan/DeviceVk.cpp:575`.

**Backlog distinction:** D7 covers GLES cube attachments. This is ordinary Vulkan sampled cube support.

### R08 — Reuse ordinary draw validation in subpass draws

**Evidence:** `yawgpu-core/src/subpass.rs:834–860` checks pipeline bindings, required storage/uniform sizes, presence of used vertex slots, and presence of an index buffer. It discards the counts/offsets in `RenderDrawKind` apart from `is_indexed()`. The ordinary implementation in `pass.rs:677–714` additionally calls `validate_strip_index_format`, `validate_vertex_buffer_oob`, and `validate_index_buffer_oob`.

**Trigger and impact:** in a `tiled` pass, draw beyond the bound vertex or instance range, request indices beyond the bound index range, or bind a strip index format inconsistent with the pipeline. The equivalent regular draw is rejected, while the subpass path records it for HAL execution. This violates the intended validation contract and can result in invalid native draw accesses or wrong output.

**Correction:** share the range/strip validation logic, preserving the correct distinction between direct, indexed-direct, and indirect draws. Presence-only checks are insufficient.

**Backlog distinction:** Block 150 closed A31's pass-wide usage-scope conflicts. This is per-draw range and strip-format validation, not that usage-scope issue or E5's missing subpass features.

### R09 — Treat a missing Metal occlusion buffer as an allocation failure

**Evidence:** `yawgpu-hal/src/metal/query_set.rs:63–77` stores the optional result of `newBufferWithLength_options` directly in `MetalBuffer`, then returns `Ok`. The timestamp allocation in the same constructor already maps a native failure to `HalError::OutOfMemory`.

**Trigger and impact:** when Metal returns `nil` for the occlusion visibility buffer, query-set creation still succeeds. The invalid backing resource is discovered only when encoding or resolving queries, losing the creation-time allocation error and allowing an unusable query set into later commands.

**Correction:** convert `None` into `HalError::OutOfMemory` before constructing the buffer, and implement R12 so the C API reports that error with the correct kind.

**Backlog distinction:** no corresponding remaining item; this is a missing failure check in a supported allocation path.

### R10 — Validate external texture plane component counts

**Evidence:** `yawgpu-core/src/external_texture.rs:311–403` checks plane presence, dimensions, usage, and float/filterability, but never checks `FormatCaps::color_components` for the declared external texture format. Pinned Dawn's `native/ExternalTexture.cpp:85–135` requires four components for a single plane, or one and two for the two-plane case.

**Trigger and impact:** import an R8 or RG8 view as RGBA, or use an incorrectly sized component format for either NV12 plane. Creation succeeds and conversion consumes default/missing channels, producing wrong colors instead of rejecting the descriptor.

**Correction:** require four components for RGBA, one for NV12 plane 0, and two for NV12 plane 1, alongside the existing sample-type rules.

**Backlog distinction:** separate from A27/A28's plane initialization and view lowering; no remaining entry records this validation gap.

### R11 — Preserve sampler creation failures

**Evidence:** `yawgpu-hal/src/metal/device.rs:123–127` and `vulkan/device.rs:168–174` call `.ok()` on native sampler creation, returning a wrapper with no native sampler. GLES stores a deferred result in `gles/sampler.rs:12–58`. The common creation API in `yawgpu-hal/src/lib.rs` does not return a `Result`.

**Trigger and impact:** a native sampler allocation/creation failure still appears to be a successful WebGPU sampler creation. An error scope around creation receives nothing; the error surfaces later during command processing, with a different operation and classification. An unused failed sampler may never report its failure.

**Correction:** return a typed `Result` from HAL sampler creation through core, map allocation failures to `OutOfMemory` and other backend failures to `Internal`, and return a properly marked error sampler.

**Backlog distinction:** independent of GLES drop/context issues and of the existing allocation checks for buffers and textures.

### R12 — Keep the error kind when query-set creation fails

**Evidence:** `yawgpu-core/src/device.rs:192–230` converts HAL query-set failures to `Option<String>`. `yawgpu/src/ffi/device.rs:329–334` dispatches every such string as `ErrorKind::Validation`.

**Trigger and impact:** native query allocation fails with OOM or an internal/backend error. A valid descriptor is blamed as invalid, and an `OutOfMemory` error scope cannot catch the allocation failure. This also applies to failures already correctly reported by the timestamp constructor.

**Correction:** return a typed `DeviceError` (or equivalent error enum) from core creation, retaining the distinction between descriptor validation, OOM, and internal failure.

**Backlog distinction:** no existing entry covers this lossy conversion. R09 fixes detecting one failure; R12 fixes propagating all detected failures.

### R13 — Gate only RG11B10 renderability, not the whole format

**Evidence:** `yawgpu-core/src/format.rs:555–559` returns `None` for `RG11B10_UFLOAT` when `Rg11b10UfloatRenderable` is absent. A separate feature-dependent renderability upgrade already exists at lines 773–775. Pinned Dawn's `Format.cpp` defines the baseline format separately from its renderable-feature upgrade.

**Trigger and impact:** a device without the optional renderability feature creates an `rg11b10ufloat` texture for sampling and/or copies. Core rejects the format even though those usages do not require renderability. The premature `None` also prevents later capability upgrades from being applied.

**Correction:** expose the baseline float texture capabilities unconditionally; keep only render attachment/MSAA capabilities behind the renderability feature and preserve the independent storage-feature rules.

**Backlog distinction:** not B6's already-correct Tier-2 storage format list; this is an earlier feature gate that removes the entire format.

### R14 — Avoid waiting inside an empty Vulkan submit

**Evidence:** `yawgpu-hal/src/vulkan/queue.rs:224–227` submits an empty batch and immediately calls `wait_for_submission`. Core routes an empty `Queue::submit` through that path (`yawgpu-core/src/queue.rs:879–886`).

**Trigger and impact:** an application uses `wgpuQueueSubmit(queue, 0, ...)` as a flush while previous GPU work is outstanding. The API call blocks until the empty submission and all preceding queue work finish, turning a flush into a CPU/GPU synchronization point. It can cause stalls that ordinary nonempty submission benchmarks miss.

**Correction:** return the submission index without waiting and let existing retirement/poll/wait paths handle completion. Verify that the empty submit retains its flush ordering.

**Backlog distinction:** Block 117 made Vulkan submission nonblocking. E11 concerns native object caches, and E2 records Metal performance measurements; neither records this synchronous empty-submit path.

### R15 — Make floating-point assertions reject NaN

**Evidence:** examples include `yawgpu/tests/e2e_metal_depth.rs:157,291,789,1021,1722,1941`, `e2e_vulkan_depth.rs:237,1401,1577`, and both `e2e_*_immediates.rs:321`. They increment mismatch counts only when `(got - expected).abs() > tolerance`. A shader-side check in `e2e_metal_depth.rs:3828` uses the same pattern.

**Trigger and impact:** broken shader output or readback yields NaN. Its ordered greater-than comparison is false, so these checks treat the invalid result as matching. This weakens the regression evidence for depth and immediate-data paths. Checks written as positive `abs() <= tolerance` assertions do not have this particular defect.

**Correction:** explicitly reject non-finite results when finite values are expected, or assert the positive matching condition. Apply the equivalent rule to shader-side checks.

**Backlog distinction:** a verification correctness issue, separate from C4's CI/platform coverage. This finding was established from the predicates, not by running the tests.

### R16 — Bound GLES zero-upload staging memory

**Evidence:** `yawgpu-hal/src/gles/queue.rs:3292–3320`, `submit_texture_clear_zero_upload`, computes full mip width × height × layer count × bytes per pixel, then creates a single zero-filled `Vec<u8>`. Conversion and multiplication are checked, but allocation is not bounded or fallible. The core emits subresource clears, so even a one-layer clear can be large.

**Trigger and impact:** a large valid, non-renderable color texture needs lazy initialization. The backend allocates a full mip's additional host memory to clear it. This can be hundreds of megabytes, causing unnecessary pressure, process termination on allocation failure, and long stalls even when the GPU image itself fits.

**Correction:** upload zero rows/tiles from a bounded reusable staging block, and propagate staging allocation failures. Size work according to the supported format and layer layout.

**Backlog distinction:** A3 covers unsupported depth/stencil and MSAA clear operations. This concerns memory use of an implemented single-sample color clear. Kept last because GLES is experimental and already ranked below supported-backend work.

## Smaller example improvements

These are useful local fixes, but are lower priority than the library findings.

| ID | Difficulty | Evidence and correction |
|---|---|---|
| X01 | S | `examples/framework/framework.c:497–518`: `yawgpu_create_buffer_init` sets `mappedAtCreation` whenever size is positive, but unmaps only if `contents` is non-null. A positive-size/null-contents request returns a still-mapped buffer and later GPU use fails validation. Reject that input or unmap the created buffer independently of whether data was copied. |
| X02 | S | `examples/triangle_passthrough/main.c:565–571`: initialization failure calls `app_destroy` before testing `app.window`; `app_destroy` at lines 307–320 clears the entire struct. Therefore the failure branch always returns `EXIT_SUCCESS`, including real Metal/Vulkan failures. Even testing the window before cleanup would misclassify context/shader failures that occur before window creation. Return an explicit initialization outcome distinguishing unsupported-backend skip from failure. |

## Existing items and rejected candidates

- **A31:** already fixed by Block 150; R08 is the separate draw-validation omission.
- **A21:** explicitly deferred pending a real Intel Gen9 report; not proposed again.
- **GLES mixed comparison/non-comparison depth sampling:** still an execution limitation, but already documented in `specs/blocks/67-gles-backend.md:895–902` and the GLES sweep ledger. Removed from the new-finding list after checking those records.
- **B7 / GLES context and drop issues:** existing threading/context-lifetime scope; not relisted as new work.
- **A3, A7, B2, B3, C1–C4, D1–D13, E2–E5, E11, E15:** existing limitations, deferred work, or intentional behavior remain in their own ledger.
- **RGBA8 color-byte cost:** a suspected discrepancy was discarded; the attachment cost can be 8 despite a texel byte size of 4, matching the pinned Dawn table.
- **Vulkan coherent host-memory fallback:** discarded as a candidate after checking the required memory-type guarantee.
- **Command buffers consumed by invalid submit and views of destroyed textures:** not reported where the pinned Dawn behavior agrees with the implementation.

## Validation limits and next implementation checks

All triggering scenarios above are derived from code paths. They are not runtime reproductions. For fixes, prioritize reentrant-release cases (R01), extreme invalid descriptors and lost devices (R02), safe-API boundaries (R03), cross-device imports (R04), retained surface objects (R05), injected submission failure (R06), native Vulkan cube sampling (R07), direct/indexed subpass validation (R08), and allocation/error-scope injection (R09/R11/R12). Backend-dependent effects and performance magnitude require later hardware verification.

The report changes documentation only. The backlog itself and all implementation/test files are unchanged.
