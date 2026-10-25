# Block 117 — Vulkan: non-blocking retirement and recycled submit objects

Status: **COMPLETE (2026-10-25)** — implementation and Phase Review fixes in one
commit.
Source: `specs/tracking/code-review-2026-10.md` ("Vulkan per-submit object
churn + retire waits under `queue_access` (perf)"); perf baseline
`specs/tracking/perf-dawn-baseline.md` (queue layer ~10× behind Dawn, blocking
submit). Oracle: Dawn `third_party/dawn/src/dawn/native/vulkan/`
(`DeviceVk.cpp` command-pool recycling `GetUnusedCommands` /
`RecycleCompletedCommands`, fence recycling `GetUnusedFence` /
`mUnusedFences`; `ResourceMemoryAllocator` / deferred deletion by serial — no
CPU wait on submit).

## Problem

Every `VulkanQueue::submit_copies` (the single submit path: copies, render and
compute passes, `writeBuffer` staging, `submit_empty`) in
`yawgpu-hal/src/vulkan/encode.rs` `submit_copies` /
`record_and_submit_copies`:

1. **creates a `VkCommandPool` and a `VkFence`** and destroys both when the
   submission retires (`surface.rs` `cleanup_frame`).
2. **retires through a 3-slot `RetireRing`** (`RETIRE_RING_SIZE`): when the
   next slot is occupied, `retire_tracked` blocks on that slot's fence with
   `vkWaitForFences(u64::MAX)`. So the 4th in-flight submit stalls the CPU
   until the 1st completes on the GPU.
3. does that wait **while holding `queue_access`** (and the retire lock), so
   every other thread's submit, `writeBuffer` / `writeTexture`, `wait_idle`
   and present transition is blocked behind one GPU wait.

## Rules

### R1 — retirement never waits under `queue_access`

`record_and_submit_copies` drops the `queue_access` guard (and the per-surface
state guards) before handing the submission to the retire queue. Recording,
`vkQueueSubmit`, layout commit and `register_fence` stay under
`queue_access` exactly as today.

### R2 — the queue's retire queue harvests instead of blocking

The queue-level retire ring (`VulkanQueueInner::retire`) becomes an unbounded
FIFO of in-flight frames:

- On every push, first harvest from the front every frame whose fence reports
  signalled (`vkGetFenceStatus`, non-blocking), running `cleanup_frame` on
  each; stop at the first unsignalled one.
- Back-pressure only above a high-water mark (`MAX_IN_FLIGHT_SUBMISSIONS`,
  64): while the queue holds more frames than that, wait on the **oldest**
  frame's fence and clean it up. This wait happens outside `queue_access`
  (R1).
- `completed_submission_index` and `wait_for_submission` also harvest
  signalled frames (so retained resources and recycled objects come back on
  poll / wait without a new submit). They must not take `queue_access`.
- `wait_all` (queue drop) keeps its semantics.
- A `vkGetFenceStatus` / `vkWaitForFences` error keeps the frame queued and is
  reported as today (no cleanup of possibly in-flight objects).

The per-surface `SurfacePendingState::retire` ring (present transitions,
swapchain-owned fences, Block 109 C2) is **out of scope** and keeps its
current behaviour; if the shared type is split, the surface keeps the
existing ring semantics and its unit tests unchanged.

### R3 — command pools are recycled

The queue owns a free list of `VkCommandPool`s (TRANSIENT, the queue family).
A submit takes a pool from the free list or creates one. When its frame
retires, the pool is reset (`vkResetCommandPool`, flags 0) and returned to the
free list instead of destroyed. The free list is capped (16); a pool retired
while the list is full is destroyed. All free pools are destroyed when the
queue is dropped (after `wait_all`). A pool whose submission failed before
`vkQueueSubmit` is reset and recycled (or destroyed) through the same path,
never leaked.

### R4 — fences are recycled

The queue owns a free list of unsignalled `VkFence`s. A submit takes one or
creates one. Wherever the code **destroys** a submission fence today it
recycles it instead: `vkResetFences`, then push to the free list (capped at
16; destroy when full):

- `cleanup_frame` when no waiter has it pinned;
- the last waiter's `unpin_fence` in `wait_for_submission` when retirement
  removed the mapping first (the existing pin / `pending_destroy` protocol is
  kept, only the final action changes from destroy to recycle).

A pinned fence must never be reset or reused. Fences from the per-surface
ring (`destroy_fence = false`, swapchain-owned) are untouched. Free fences are
destroyed when the queue is dropped. If `vkResetFences` fails, destroy the
fence instead of recycling it.

### Out of scope (follow-up, measure first)

Per-pass `VkRenderPass` / `VkFramebuffer` / transient `VkImageView` /
`VkDescriptorPool` creation inside recording. Dawn caches render passes and
uses framebuffer-less dynamic state differently; a cache needs its own
block. Record the post-block bench numbers so the next block starts from data.

## Tests

HAL unit tests (`#[cfg(test)]` in `yawgpu-hal/src/vulkan/**`, real device,
run by Claude with `--features vulkan --lib -- --ignored` on MoltenVK under
the Khronos layer, like the existing ring tests):

- R2: pushing more than `RETIRE_RING_SIZE` frames whose fences are never
  signalled returns immediately (the old ring would block forever; use a
  fence created unsignalled and never submitted, and assert the call returns);
  a signalled frame at the front is harvested on the next push; above the
  high-water mark the oldest frame is waited (use signalled fences so the test
  cannot hang).
- R2: `completed_submission_index` after a completed submit harvests the frame
  (retained resource `Arc` strong count drops).
- R3: two sequential submits with a harvest in between reuse the same
  `VkCommandPool` handle; the free list never exceeds its cap.
- R4: a retired, unpinned fence comes back unsignalled and is reused by the
  next submit; a fence pinned by a waiter when retirement runs is recycled
  only by the last unpin, never while pinned.
- R1: a submit from thread B completes while thread A's submission is still
  being retired / waited (e.g. thread A blocked in `wait_for_submission` on an
  unsignalled-by-design submission is not required — instead assert
  structurally that `retire` is called after the `queue_access` guard is
  dropped, via a test hook or by holding the retire lock in the test while
  another thread submits successfully).

Claude (real GPU): all `e2e_vulkan_*` on MoltenVK with `vulkan,tiled` under
the Khronos layer **with synchronization validation** — 0 `VUID` / 0
`SYNC-HAZARD` (only the pre-existing MoltenVK 3D-attachment failure); a new
`e2e_vulkan_*` stress case (hundreds of un-waited submits that each write and
read buffers, then one readback — exact values) and a two-thread submit case;
the `benches/` `submit/` + `queue/` + `frame/` cases on MoltenVK before and
after (numbers recorded here).

## Gates

`cargo test --workspace`; strict clippy (default / `vulkan` / `vulkan,tiled` /
`metal` / `gles`) + fmt; `cargo test -p yawgpu-hal --features vulkan --lib`
(+ `-- --ignored` by Claude on MoltenVK).

## Bench (MoltenVK on M2, `--reps 7`, min ns/op; before = `0915a5d`)

| case | before | after |
|---|---|---|
| `submit/empty` | 23656 | 17425 (−26%) |
| `submit/compute_1_dispatch` | 45359 | 24105 (−47%) |
| `submit/render_100_draws_wait` | 215690 | 192566 |
| `submit/compute_wait_idle` | 146261 | 150081 |
| `queue/write_buffer_4kb` | 622 | 645 |
| `queue/write_buffer_then_wait` | 124140 | 122456 |
| `frame/10writes_dispatch_submit_wait` | 162080 | 167288 |

Un-waited submits are where the CPU cost moved; the `_wait` cases are
dominated by GPU / MoltenVK latency and stay within noise. The remaining
~17 µs per empty submit is the next thing to profile (per-pass Vulkan object
creation is out of scope above).

## Verification (Claude, real GPU)

- HAL ignored tests on MoltenVK under the Khronos layer with synchronization
  validation (`--features vulkan,tiled --lib -- --ignored`): 70/70 after the
  Phase Review fixes, 0 VUID.
- All `e2e_vulkan_*` (`vulkan,tiled`) under synchronization validation:
  132 pass, 1 pre-existing MoltenVK fail
  (`vulkan_3d_attachment_discard_lazily_zeroes_the_whole_mip`, the only
  source of `VUID` lines), 0 `SYNC-HAZARD`. Includes the new
  `e2e_vulkan_submit_recycling` (300 un-waited submissions, two threads ×
  150) — exact sums. Re-run after the Phase Review fixes: identical.
- `cargo test --workspace` 1231/0; strict clippy default / `vulkan` /
  `vulkan,tiled` / `metal` / `gles` + fmt clean.

## Phase Review (2026-10-25)

Fresh reviewer: **0 CRITICAL / 0 MAJOR / 5 MINOR**. Lifetime (pools reset
only after their fence signalled and the frame was popped; fences recycled
only unmapped and unpinned), lock order (queue_access → surface states →
submissions / recycler; retire → submissions, retire → recycler; retire is
never taken under queue_access) and ordering (index-sorted insert, front-only
harvest; `completed` never passes incomplete work) were checked.

| ID | Finding | Disposition |
|---|---|---|
| m1 | `completed_submission_index` / `wait_for_submission` block on the retire mutex while another thread is in a back-pressure GPU wait | fixed: `try_lock`, harvest is opportunistic |
| m2 | `wait_idle` does not harvest; up to 64 frames of retained resources outlive it | fixed: harvest after `vkQueueWaitIdle`, outside `queue_access` |
| m3 | a `vkGetFenceStatus` error during harvest turns a successful submit into `Err` | accepted (same as the old ring; device-loss-class error) |
| m4 | pinned-fence test drives the last unpin by hand; high-water test calls `enforce_limit` directly | accepted: the last-unpin path is a single call site; the e2e stress case runs the real back-pressure path |
| m5 | queue frames may hold views / framebuffers of swapchain images across recreation for longer | accepted: destroying them after the image is legal and was already possible |
