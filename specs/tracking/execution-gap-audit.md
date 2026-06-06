# Execution-gap audit (Audit A) — "validated-but-not-executed"

Proactive sweep (2 parallel agents: core→HAL seam + backend submit/encode) for the F-034/F-042/F-025
family — an operation `yawgpu-core` accepts/records but which emits no (or a wrong) executable HAL command,
so it is a silent no-op / clobber on real hardware. Part of [[cts-failure-patterns]] (pattern 1a).

## Findings

| # | Site | Issue | Severity | Status |
|---|---|---|---|---|
| **1** | `yawgpu-core` regular render path (`render_pass.rs` records one `RenderPassCommand` per draw, each cloning the pass's original `load_op`/`store_op`; `queue.rs` replays each as a SEPARATE HAL render pass; both backends open a fresh encoder with `loadAction`/`storeAction` from those ops) | **A render pass with ≥2 draws re-cleared the attachment before every draw (`loadOp=Clear` → only the LAST draw survived) and/or discarded earlier output (`storeOp=Discard` → later draw loaded undefined; the common multi-draw + depth-discard case).** | **Tier-1 correctness (severe)** | **RESOLVED** |
| **2** | `yawgpu-hal/src/gles/queue.rs` `submit_render_pass` | GLES render path binds `bind_buffers`+`vertex_buffers` but never read `pass.bind_textures`/`bind_samplers` — a fragment shader sampling a texture drew with it unbound (garbage), no error. | Tier-2, uncatalogued silent skip | **RESOLVED** |

**Fix (commit pending push).** #1: `PassEncoderState::load_attachments_for_draw` forces `store_op = Store`
on every split pass (color + depth + stencil) so each pass's output persists, and downgrades
`load_op → Load` only after the first draw; the first draw keeps the user's `load_op` (Clear), the clear-only
`end()` path keeps the user's original ops. All draw sites + `execute_bundles` route through it. Forcing
`Store` is a safe superset (`Discard` only permits not-storing; yawgpu copies attachments back) AND is
load-bearing for MSAA resolve (intermediate passes must store the MSAA target so the final resolve
accumulates). The `tiled` subpass path was already correct (one HAL pass, many draws). #2: reject a render
pass carrying texture/sampler bindings with a catalogued `HalError` (mirrors the compute path).

**Verified real-GPU Metal + Vulkan/MoltenVK:** `*_two_draws_one_pass_accumulate` (clear red → draw left-half
green → draw right-half green → BOTH halves green) passes on both; round-1 of the fix (which kept the user's
`store_op` on the first pass) was caught in review for the `storeOp=Discard` case and corrected. No
regression in MRT / MSAA resolve / two-draws-same-storage / render-bundle / depthSlice / multi-pass-depth
e2e. Noop unit test asserts first command = Clear/Store, second = Load/Store, clear-only = Clear/Discard.
Clean Review: no CRITICAL/MAJOR.

## Deferred / documented (NOT new findings — noted for completeness)
- `wgpuCommandEncoderWriteBuffer` (`ffi/encoder.rs:280`) is a documented "P6.2 validation-only" stub that
  discards `data` — F-025's un-fixed sibling (exposed, silent no-op on real backends). Implement (stage data
  → buffer copy at submit) or return a device error in a later slice; don't leave it silently succeeding.
- `wgpuCommandEncoderWriteTimestamp`, `wgpuCommandEncoderResolveQuerySet`,
  `wgpuRenderPassEncoderBegin/EndOcclusionQuery` — occlusion/timestamp queries are documented-deferred
  (C34/C35 → P8 in `specs/blocks/50-commands.md`/`70-finalize.md`; GLES Tier-2 deferred). They currently
  validate+accept silently rather than erroring; revisit when P8 queries land.

## Fix (this slice): items 1 + 2
1. Regular render path must clear the attachments ONCE per render pass. Minimal fix (preserve the
   per-draw-HAL-pass architecture): the first emitted HAL pass for a render pass uses the user's color +
   depth + stencil `load_op`; every subsequent draw's `RenderPassCommand` downgrades color `load_op` →
   `Load` and depth/stencil load → `Load` (store stays `Store` so each pass's output persists for the next
   to load). Clear-only `end()` (no draws) keeps the real `load_op`. Track "first draw emitted" on the pass
   state. (Future optimization: coalesce consecutive draws into one HAL pass like the subpass path.)
2. GLES: reject a render pass carrying texture/sampler bindings with a `HalError` (mirror the compute path),
   catalogue it in `67-gles-backend.md`. (Implementing `glBindTexture`/`glBindSampler` is the better long
   fix but larger; HalError is the safe Tier-2 move and never relaxes core.)

Verify (Claude): `e2e_{metal,vulkan}_threading_audit.rs::*_two_draws_one_pass_accumulate` (both halves green)
on Metal + Vulkan/MoltenVK; no regression in existing rendering e2e/CTS.
