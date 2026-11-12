# Block 153 — Transactional queued-write initialization; surface textures expire

Status: **S1 and S2 landed (2026-11-13)**; S3 (S1 Phase Review fixes) and the S2 Phase Review pending.
Source: whole-code review at `92e38c7` (`specs/tracking/code-review-2026-11.md`), findings R06 and R05.
Oracle: pinned Dawn — `native/SwapChain.cpp` (`SwapChainBase::Present` checks that the
backend destroyed the current texture, line 155; the destructor checks the same, line 88);
yawgpu Block 104 R5 for initialization-mark ordering.

## Problem

R06: `Queue::write_texture` (both the repack and the fast path, `queue.rs` ~698 and ~756)
commits its `InitOverlay` immediately, before the copy reaches the GPU. If the HAL submit
that later carries the pending batch fails (`flush_pending_writes_locked` →
`finish_pending_submission` error), the batch is dropped but the destination stays
marked initialized, so a later use skips its lazy zero clear. Command-buffer submission
already commits only after a successful HAL submit (Block 104 R5).

R05: present, configure, and unconfigure only clear the surface's cached
`current_texture` (`ffi/surface.rs`). An application-held `WGPUTexture`, a view of it,
or a command buffer recorded against it keeps passing the destroyed-resource checks, so
a later submit can still use a swapchain image that was presented or belongs to an old
swapchain.

## Rules

R1. **Queued-write marks are pending until their submit succeeds (R06).** The pending
write batch owns an `InitOverlay` of its own. `write_texture` plans its clears against the
committed state *plus* that pending overlay (so a second queued write to the same
subresource does not clear the first one's data — the F-138 inverse bug) and records its
marks into the pending overlay, not the texture. The pending overlay is taken together
with the batch's copies and chunks (`take_submission`) and:
- on a standalone flush (`flush_pending_writes_locked`, the empty-`submit` path, the
  validation-error and lowering-error paths in `submit`) it is committed only when the
  HAL submit of that batch succeeds, and dropped otherwise;
- on a command-buffer `submit`, it seeds the walk's `init_overlay`, because the queued
  copies run first in the same HAL submit; the walk's overlay is then committed or
  dropped as today.

A dropped overlay leaves the textures' committed state unchanged, so their next use
lazy-clears. The submission lock ordering is unchanged.

R2. **Readers of initialization state.** The coding agent lists every reader of a
texture's committed initialization state and confirms that, between a queued write and
its flush, each one either consults the pending overlay or cannot observe the
difference (report it). Any reader that would now see a stale "uninitialized" for a
subresource with a queued write must consult the pending overlay.

R3. **The acquired surface texture expires (R05).** The core texture backing the
surface's current texture is destroyed (`Texture::destroy`) when its acquisition ends:
- after `wgpuSurfacePresent` hands it to the HAL present (success or failure);
- in `wgpuSurfaceConfigure` and `wgpuSurfaceUnconfigure`, before the old swapchain
  state is replaced;
- when the surface is dropped while a texture is acquired.

Pending queue writes are flushed before present (as today); work already submitted keeps
its HAL references (Block 143), so destroying the core texture never frees an image the
GPU still uses. After expiry, using the old texture, a view of it, or a command buffer
that references it fails with the ordinary destroyed-texture validation errors. Calling
`wgpuTextureDestroy` on it remains allowed and is a no-op.

R4. **No HAL-level release from R3.** Taking the core texture's HAL handle on destroy
must not destroy or recycle the swapchain image itself — the HAL surface still owns it.
The agent confirms this for Metal (drawable), Vulkan (swapchain image), and GLES, and
reports how each backend's surface texture handle behaves on drop.

## Slices

S1: R1–R2 (`yawgpu-core/src/queue.rs`), with inline tests using the existing
`fail_next_pending_submit` hook: a failed pending flush leaves the destination
uninitialized (the next read-back path plans a clear); two queued writes to one
subresource plan no clear between them; a queued write followed by a command-buffer
submit that reads the texture plans no clear; a successful flush commits the marks.

S3 (Phase Review S1 fixes): A-m1, A-m2, A-m3.

S2: R3–R4 (`yawgpu/src/ffi/surface.rs`, `instance.rs` drop path), with inline FFI tests on
Noop where a surface texture can be simulated, otherwise a core-level test plus the
real-GPU checks below.

## Verification

- Agent gates: `cargo test -p yawgpu-core --lib`, `cargo test -p yawgpu --lib`, clippy
  workspace (+ `--features metal`, `--features vulkan` for S2).
- Claude: workspace test; for S1 `e2e_metal_*` / `e2e_vulkan_*` lazy-init suites
  (F-138 regression); for S2 a new real-GPU e2e test on Metal and MoltenVK: acquire,
  keep a view, present, then submit a pass that uses the view → validation error, no
  backend error; and every windowed C example with `--verify`.

## Verification record

S1 (with Block 151 m1): workspace test 1,465/0; `yawgpu-core --lib` 731/0 (the four S1
tests red before the fix); clippy and fmt clean. Real GPU on the M2: every `e2e_metal_*`
170/0 (including `e2e_metal_lazy_init` 7/7); every `e2e_vulkan_*` on MoltenVK under the
Khronos layer 163 passed, 1 failed — the known MoltenVK `imageView2DOn3DImage` case
(`vulkan_3d_attachment_discard_lazily_zeroes_the_whole_mip`), which is the source of every
VUID. R2 audit: `InitOverlay::is_initialized` is the only production reader of committed
init state outside `texture.rs`; every planning helper (copy, bound view, render and
subpass attachment, external-texture plane) reads through the pending or combined overlay.
A failed lowering walk flushes the queued writes with their own overlay only, so its
aborted discard marks never commit.

## Phase Review S1 (2026-11-13, fresh-context reviewer over `d0839ee`)

0 CRITICAL, 0 MAJOR, 3 MINOR. The reviewer traced every flush path (command-buffer,
validation-error, duplicate, lowering-error, empty, threshold, map, work-done, present,
device loss, queue drop) and confirmed marks commit only with a successful HAL submit,
a later `Discard` in the same submit wins, the lock order is unchanged, and the overlay's
texture handles form no cycle.

| # | Severity | Finding | Resolution |
|---|---|---|---|
| A-m1 | MINOR | No injected HAL failure on the empty-`submit` path or the command-buffer HAL submit while queued marks are pending | Fix slice S3: the `fail_next_pending_submit` hook also covers `submit(&[])` and the command-buffer submit; one failure test per path |
| A-m2 | MINOR | `queued_init_overlay.clone()` on every command-buffer submit, for the rare lowering-error path | S3: skip the clone when the queued overlay is empty |
| A-m3 | MINOR | Vague fast-path comment ("Share pending initialization with the repack path above.") | S3: state the rule (earlier queued marks suppress clears until submission) |

## Verification record — S2

Workspace test 1,476/0; `yawgpu --lib` 322/0, with `tiled` 326/0; clippy clean. Four
`acquired_texture_expires*` FFI tests (present success and failure, configure success and
HAL failure, unconfigure, surface drop) were red before the fix: each submits a command
buffer recorded against the old view and gets `queue submit cannot use a destroyed
texture`, and the next acquisition renders normally. Real GPU on the M2: `triangle`,
`hello_triangle`, and `surface_smoke` with `--verify` on Metal and on MoltenVK under the
Khronos layer, exit 0, 0 VUIDs. R4 audit: Metal drops a retained `MTLTexture` (the
surface keeps the drawable); Vulkan swapchain textures have `owns_image = false`; GLES
drops a clone of the configured back buffer — no expiry frees or recycles a swapchain
image. A real-GPU e2e of the expiry error itself needs a window and is covered by the
Noop tests instead.
