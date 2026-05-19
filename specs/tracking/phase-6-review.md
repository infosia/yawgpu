# Phase 6 Review — Clean Review Then Fix

Process: `../reference/workflow.md` → "Phase Review". Fresh no-context
reviewer over the cumulative Phase-6 implementation diff
(`fd074a0..1a6e0b8`, slices P6.1–P6.9) + block 50 + the documented
per-slice divergence notes. Result: **0 CRITICAL, 1 MAJOR, 5 MINOR**.

Reviewer verified sound (no finding): no `unwrap`/`expect`/`panic!`/
overflow reachable from the C ABI on hostile input in `yawgpu-core`
(all new copy-bounds / draw-OOB / dynamic-offset / indirect-args /
usage-scope / same-texture arithmetic uses `checked_*`/`saturating_*`
with explicit `Err`; the lone bare subtractions are guarded by a
preceding `offset > size` early-return); FFI `expect` confined to null
handles / null required descriptor pointers (wgpu-native parity);
`clone_handle` Arc refcounting in the new copy FFI is net-correct (no
leak / double-free); no unsound `unsafe`; spot-checked ☑ rules
(C9, C40, C46, C50–C52, C68/C72/C73, C75–C82) genuinely enforced with
non-vacuous ok/error test pairs; documented divergences match the code.

## Triage + disposition

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| K1 | MAJOR | Render/compute-pass debug-group **imbalance at End (C60/C61)** and **pop-underflow** were reported twice for one fault: the branch called `parent.record_first_error(..)` (→ surfaces at `wgpuCommandEncoderFinish`) **and** returned `Some(message)` (→ FFI `dispatch_optional_error` dispatches it immediately). Inconsistent with the deferred-error model used by every other pass command (`record_pass_command` records-only/`None`). The `debug_marker_validation.rs` pass/compute helpers hid it by omitting the post-command `assert!(errors().is_empty())` every sibling helper has. | **FIX (codex)**: `PassEncoderInner::{end (imbalance branch), pop_debug_group (empty-stack branch)}` record-only + return `None`; C3/C4 (double-end / after-parent-finish) immediate branches and encoder C63 / bundle path untouched. Strengthen the four pass/compute test helpers to assert no early error after commands and after End, error only at Finish (matching `pass_state_validation.rs`). |
| m1 | MINOR | `debug_marker_validation.rs` pass/compute helpers weaker than siblings (no deferred-timing assertion). | **FIX (codex)** — folded into K1's test hardening. |
| m2 | MINOR | `record_first_error` re-locks the encoder mutex while the pass-state lock is held (pass-lock-before-encoder-lock); no cycle demonstrable but ordering implicit. | **DEFER (logged)**: correctness-preserving; no demonstrated cycle. Logged in `phase-6.md` for a future lock-ordering comment. |
| m3 | MINOR | `set_scissor_rect` overflow-only, no attachment-size clamp. | **DEFER (logged)**: already a documented divergence (block 50 P6.5 note → P6.9/P7); code matches the note. |
| m4 | MINOR | `firstInstance` indirect feature gating accepted unconditionally. | **DEFER (logged)**: already a documented divergence (block 50 P6.6 note); no canonical webgpu.h toggle; code matches the note. |
| m5 | MINOR | ExecuteBundles-contributed resource usage not merged into the pass usage scope. | **DEFER (logged)**: already a documented divergence (block 50 P6.9 note); no ported test asserts anything false; code matches the note. |

No false positives. **Gate: Phase 6 cannot be COMPLETE while K1
(MAJOR) is open.** m2 is a new tracked MINOR; m3/m4/m5 are
pre-documented divergences whose code faithfully matches the notes.

## Resolution log

**CLOSED** — K1 fixed by codex, m1 folded into the K1 test hardening,
m2 accepted-as-tracked, m3/m4/m5 confirmed already-documented
divergences. Reviewed by Claude, gate green (42 binaries,
`cargo clippy --workspace --all-targets -D warnings` clean).

- **K1** FIXED: `PassEncoderInner::end()` (unbalanced-debug-groups
  branch) and `pop_debug_group()` (empty-stack branch) now
  `parent.record_first_error(message)` + return `None` — the single
  device error surfaces only at `wgpuCommandEncoderFinish`, like all
  other deferred pass errors. The `state.ended` (C3) /
  `parent.is_finished()` (C4/C85) early branches and `push_debug_
  group` / encoder C63 / the bundle path are unchanged. The four
  `debug_marker_validation.rs` pass/compute helpers now
  `assert!(test.errors().is_empty())` after `commands(..)` and after
  `End` and assert the error appears at `finish_error`; redundant
  `clear_errors()` removed. `debug_marker_validation` 14 passed
  (now actually pinning single, deferred error).
- **m1** FIXED with K1.
- **m2** ACCEPTED-AS-TRACKED: no demonstrated lock cycle (pass lock
  is always taken before the encoder lock on these paths); logged in
  `phase-6.md` as a follow-up to add a lock-ordering comment.
- **m3 / m4 / m5** ACCEPTED: pre-existing documented divergences in
  `blocks/50-commands.md` (P6.5 scissor / P6.6 firstInstance / P6.9
  ExecuteBundles-usage); reviewer confirmed the code matches each
  note; no further action.

Gate at close: `cargo test --workspace` 42 test binaries green on
Noop (incl. `command_encoder_lifecycle` 9, `command_buffer_copy` 6,
`command_texture_copy` 4, `render_pass_descriptor` 5,
`pass_state_validation` 9, `render_bundle_validation` 5,
`debug_marker_validation` 14, `resource_usage_tracking_validation` 8,
`queue_submit_validation` 5); `clippy --workspace --all-targets
-D warnings` clean. **Phase 6 COMPLETE.**
