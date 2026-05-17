# Phase 3 Review — Clean Review Then Fix

Process: `../reference/workflow.md` → "Phase Review". Fresh no-context
reviewer over the cumulative Phase-3 diff (`e4d1a80..64850d1`) + block 20.
Result: **0 CRITICAL, 1 MAJOR, 3 MINOR**.

The reviewer confirmed otherwise sound: ABI signatures vs `webgpu.h`,
`max_texture_mips`, `subresource_size` (no-panic mip shift),
`required_bytes_in_texel_copy` == Dawn `ComputeRequiredBytesInCopy`
(overflow-guarded), non-256 queue `bytesPerRow`, Arc/refcount &
error-object handling, no disallowed panics/lossy casts.

## Triage + disposition

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| V1 | MAJOR | `Texture::is_view_format_compatible` accepts the sRGB pair **unconditionally** — Dawn/WebGPU only allow a non-identical view format if it is in the texture's `viewFormats` list. T31 not faithful; the ported test codifies the too-permissive behavior. | **FIX (codex)**: compat = `view==texture.format` OR in `view_formats`; drop the unconditional `srgb_pair()` branch. Correct `texture_view_validation.rs` (sRGB w/o viewFormats ⇒ error; sRGB *listed* ⇒ ok). Claude: clarify T31 wording in block 20. |
| V2 | MINOR | `validate_texture_view_descriptor` has dead `unwrap_or` defaults (caller always passes a fully resolved descriptor); misleading/unreachable. | **FIX (codex)**: take resolved non-`Option` values (or `expect` resolved). |
| V3 | MINOR | Handle keep-alive Arc set inconsistent: `WGPUBufferImpl` keeps device+instance; `WGPUTextureViewImpl` only `_texture`; `WGPUSamplerImpl` only `_device`. Latent footgun for real-backend Drop ordering. | **FIX (codex)**: standardize — view/sampler also hold device (+instance where the buffer does), matching the buffer handle. |
| V4 | MINOR | `FormatCaps` approximation (already a tracked note) **and** `texture_format_validation.rs` asserts `RGBA8Snorm.storage_capable == true`, which is NOT Dawn-accurate — the "sanity" test enshrines a wrong cap. | **FIX (codex)**: set `RGBA8Snorm` (and other snorm) `storage_capable=false` per Dawn; fix the sanity assertion so it does not present an inaccurate value. Keep the broader `*16/*32` refinement deferred → P4/P5 (tracked note stays). |

No findings dropped. Spec-side: Claude clarifies block 20 T31 (this
commit). All code fixes → coding agent (one handoff).

## Gate

Phase 3 cannot be marked COMPLETE while **V1 (MAJOR)** is open. V2–V4
fixed in the same pass. Re-run full gate after the fixes.

## Resolution log

**CLOSED** — all 4 findings resolved by codex, reviewed by Claude, gate
green (23 test binaries, `clippy --all-targets -D warnings` clean).

- **V1** FIXED: `srgb_pair()` removed; `is_view_format_compatible` =
  `view==texture.format || view_formats.contains(view)` only.
  `texture_view_validation` rewritten Dawn-faithful (sRGB w/o
  `viewFormats` ⇒ error; sRGB listed ⇒ ok; unrelated listed ⇒ ok).
- **V2** FIXED: `validate_texture_view_descriptor` takes a
  `ResolvedTextureViewDescriptor` (destructured non-`Option`); dead
  `unwrap_or` defaults removed.
- **V3** FIXED: `WGPUTextureImpl` +instance; `WGPUTextureViewImpl`
  +_device/_instance; `WGPUSamplerImpl` +_instance — owner set
  consistent with `WGPUBufferImpl`.
- **V4** FIXED: `RGBA8_SNORM`/`RGBA16_SNORM` `storage_capable=false`
  (Dawn `Format.cpp`); `populated_format_caps_match_dawn_sanity_checks`
  now asserts the accurate value. Broader `*16/*32` feature-gated caps
  refinement remains the tracked note → P4/P5.

Gate: no open CRITICAL/MAJOR. Phase 3 Review **CLOSED**. Commit:
`phase-3: phase review — 4 findings (1 MAJOR / 3 MINOR) fixed`.
