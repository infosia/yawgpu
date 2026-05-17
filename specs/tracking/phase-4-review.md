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

- _filled as fixes land_
