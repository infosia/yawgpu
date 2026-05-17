# Phase Review — consolidated P0–P2 (Clean Review Then Fix)

Retroactive: Phases 0–2 closed before the Phase Review gate existed
(introduced `77b5e73`). One consolidated fresh-no-context review over the
cumulative codebase (baseline..`ab36f18`). Process: `../reference/
workflow.md` → "Phase Review".

Reviewer: independent subagent, no session context. Result:
**1 CRITICAL, 5 MAJOR, 6 MINOR**.

## Triage + disposition

| ID | Sev | Finding | Disposition |
|---|---|---|---|
| C1 | CRITICAL | `requiredFeatures`/`requiredFeatureCount>0` + NULL ⇒ `expect` panic across C ABI (not a null-*handle*; CLAUDE.md forbids) — `yawgpu/src/lib.rs` `instance_has_timed_wait_any`, `required_features_from_descriptor` | **FIX (codex)**: treat null+count>0 as empty / TimedWaitAny-absent, never panic |
| M1 | MAJOR | `wgpuDeviceRelease` hand-rolled refcount + magic `==2` last-ref detection; breaks if outer Arc ever cloned elsewhere | **FIX (codex)**: robust last-ref detection, one shared implicit-destroy path with `Drop` |
| M2 | MAJOR | `wgpuDeviceGetQueue` mints a new `Arc<WGPUQueueImpl>` per call + double-Arc; default queue not handle-stable (impacts future R15) | **FIX (codex)**: cache one queue handle on the device, AddRef it; drop redundant `Arc::new` layer |
| M3 | MAJOR | `validate_buffer_descriptor` first-match-wins; B2/B3 only tested single-violation; order ≠ Dawn | **SPLIT**: Claude documents the first-match contract in block 10 (done); codex adds a multi-violation test asserting exactly-one-error holds |
| M4 | MAJOR | B13/B14 (map mode exact/unsupported bits) enforced only in FFI `conv`, not core `begin_map`; core untestable, 2nd caller bypasses | **FIX (codex)**: canonical map-mode validation in `yawgpu-core` + core unit test; conv delegates |
| M5 | MAJOR | `wgpuQueueSubmit` validates only top-level null; spec note says "arg validation" (overclaim) | **SPEC ONLY (Claude)**: tighten block 10 wording to "top-level pointer only; element/command validation → P6" (done). No code change (deferred stub, reviewer concurs) |
| m1 | MINOR | dead `_poll_only` param in core `wait_any` | **FIX (codex)**: remove the unused param |
| m2 | MINOR | `HostBuffer::new` `usize::try_from(size).unwrap_or(0)` silent truncation | **FIX (codex)**: `debug_assert!` + explicit (unreachable on valid input) |
| m3 | MINOR | `abort_pending_map` leaves `pending_map` populated (transient Unmapped+pending state) | **FIX (codex)**: doc-comment the invariant (behavior is correct: resolve drains it) |
| m4 | MINOR | empty `Drop` impls for Instance/Adapter/Queue (dead ceremony) | **FIX (codex)**: remove the no-op `Drop`s (keep Buffer/Device which do work) |
| m5 | MINOR | `validate_required_limits` ignores `self` (Noop supported==default) | **FIX (codex)**: code comment pointing to the block-00 design decision |
| m6 | MINOR | `_ => Force32`/`_Error` fallbacks on `#[non_exhaustive]` core enums silently swallow new variants | **FIX (codex)**: `// exhaustive as of <enum>` comments |

No findings dropped as false positives. The reviewer confirmed the
unsafe/FFI core is otherwise sound (Arc balance, `HostBuffer` no `&mut`
aliasing, no UAF in map last-ref/Drop path, callbacks fire once, ABI
matches `webgpu.h`).

## Gate

Phase Review is not closed until C1 + M1 + M2 + M3(test) + M4 are fixed
and the full gate (`cargo test --workspace` +
`clippy --all-targets -D warnings`) is green. MINORs m1–m6 fixed in the
same pass (cheap). Spec-side M3-doc / M5-wording done by Claude (this
commit).

## Resolution log

- _filled as fixes land_
