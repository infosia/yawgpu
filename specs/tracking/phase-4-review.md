# Phase 4 Review — Clean Review Then Fix

Process: `../reference/workflow.md` → "Phase Review". Fresh no-context
reviewer over the cumulative Phase-4 diff (`395dd05..8acd977`) + block 30
+ the naga pin. Result: **0 CRITICAL, 0 MAJOR, 5 MINOR**.

Reviewer verified sound (no finding): naga git+rev pin exact &
reproducible (`Cargo.toml` rev + `Cargo.lock` `?rev=SHA#SHA`, no
committed `[patch]`, deps checksummed); `wgsl-in`-only minimal;
`parse_and_validate_wgsl` cannot panic; S3/S4 reflection non-vacuous;
`nextInChain` walk / SPIR-V `from_raw_parts` / chained `cast` match the
existing FFI contract; `PendingCallback::CompilationInfo` lifetime &
once-only sound; `clone_handle` refcount balanced (no leak/double-free);
`Device::same` correct `Arc::ptr_eq`; error-object propagation &
first-match-wins enforced; S13 `BindingNotUsed` sentinel correct; ABI
signatures match `webgpu.h`; ported S1–S37 tests non-vacuous;
S8/S35(pipeline)/S38/S39–S44 correctly Defer→P5.

## Triage + disposition

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| W1 | MINOR | `validate_bind_group_buffer` treats `size==0` as WHOLE_SIZE; only `WGPU_WHOLE_SIZE`(=u64::MAX) is the "to end" sentinel — `size==0` is a zero-length binding WebGPU rejects. Block 30 S27 wording also encoded this wrong. | **FIX (codex code+test) + Claude (spec)**: drop the `size == 0 ||`; let the `effective_size == 0` guard reject zero; add a test; correct block 30 S27 (done this commit). |
| W2 | MINOR | `conv::map_bind_group_entry` uses `resource.expect("present … recorded")` — a non-handle internal-invariant panic at the FFI boundary (unreachable, but CLAUDE.md allows only null-*handle* `expect`). | **FIX (codex)**: restructure so the `BindGroupResource` is produced without `Option`/`expect` (or fall back to `Invalid`). |
| W3 | MINOR | `pub enum ShaderModuleSource` lacks `#[non_exhaustive]` (sibling Phase-4 public enums have it; CLAUDE.md requires it on extensible public enums). | **FIX (codex)**: add `#[non_exhaustive]`. |
| W4 | MINOR | naga `Validator` uses `Capabilities::empty()` — minimal set; later phases will over-reject until widened. | **ACCEPT (tracked)**: intentional per block 30 "Open questions"; capability-expansion is a recorded P5+ task. No code change. |
| W5 | MINOR | `FormatCaps` catch-all (unknown format ⇒ plain color) now also feeds S18/S29; no Phase-4 gap (all tested formats have explicit entries). | **ACCEPT (tracked)**: the pre-existing carried `FormatCaps`-approximation note (block 20 / block 30) already covers this; refine at P5+. No new action. |

No findings dropped as false positives. Gate: no CRITICAL/MAJOR, so
Phase 4 may close; W1–W3 fixed in one codex pass for hygiene
(user-directed), W4/W5 accepted with logged rationale.

## Resolution log

**CLOSED** — W1–W3 fixed by codex, W4/W5 accepted-as-tracked, reviewed
by Claude, gate green (27 binaries, `clippy --all-targets -D warnings`
clean).

- **W1** FIXED: `validate_bind_group_buffer` now treats only
  `size == u64::MAX` as whole-size; explicit `size == 0` falls through
  to the `effective_size == 0 ⇒ error` guard. New assertion in
  `bind_group_validation.rs` (`buffer_binding(0, uniform, 0, 0)` ⇒
  device error). Block 30 S27 corrected.
- **W2** FIXED: `conv::map_bind_group_entry` builds
  `core::BindGroupResource` directly per branch (default
  `Invalid`, `present_count != 1 ⇒ Invalid`); the non-handle
  `expect` is gone — no panic on any C input.
- **W3** FIXED: `#[non_exhaustive]` added to
  `pub enum ShaderModuleSource`.
- **W4** ACCEPTED: naga `Capabilities::empty()` intentional; widening
  tracked in block 30 "Open questions" → P5+.
- **W5** ACCEPTED: `FormatCaps` catch-all is the pre-existing carried
  approximation note (block 20/30); no Phase-4 gap; refine P5+.

Gate: no CRITICAL/MAJOR. Phase 4 Review **CLOSED**. Commit:
`phase-4: phase review — 5 findings (0C/0M/5m; W1–W3 fixed, W4/W5
accepted)`.
