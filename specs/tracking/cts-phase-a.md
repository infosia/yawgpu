# CTS port — Phase A (scaffolding) tracking

Phase A lays the groundwork for the CTS validation port. Methodology:
`specs/blocks/91-cts-conformance.md`. Ledger:
`specs/tracking/cts-coverage.md`.

Phase A has two deliverables. The docs (this file, the block, the
ledger) are authored by **Claude** and already landed. The harness +
scaffolding below is a **coding-agent** task — dispatch the handoff,
review the diff, then Phase B (per-area porting) can begin.

## Status

- [x] `specs/blocks/91-cts-conformance.md` (methodology) — Claude
- [x] `specs/tracking/cts-coverage.md` (ledger, 129-row matrix) — Claude
- [x] `specs/tracking/cts-phase-a.md` (this handoff) — Claude
- [x] `yawgpu-test` harness extensions — coding agent (done; reviewed)
- [x] `tests/cts/` directory + first aggregator wiring — coding agent
      (done; `buffer/create` ported, 5/5 green on Noop)

**Review note (Claude, on completion):** the coding agent also added a
core validation rule to `yawgpu-core/src/buffer.rs` (reject unknown
buffer-usage bits — `"buffer usage contains unknown bits"`), which the
CTS `usage` case requires (`(usage & ~kAllBufferUsageBits) === 0`). This
was technically outside the Phase-A "flag, don't fix core" scope, but
the change is spec-correct, minimal, and unit-tested — accepted as a
genuine validation gap surfaced by the port. Recorded here as the kind
of core fix future slices should ideally route through a separate fix
handoff.

---

## Task: yawgpu-test — CTS harness extensions + tests/cts scaffolding

Goal: add the four harness primitives the CTS validation port needs, and
stand up the CTS-mirrored test directory with one worked aggregator, so
Phase B slices are pure case-porting with no infrastructure work.

Inputs to read:
- `specs/blocks/91-cts-conformance.md` (esp. "harness contract",
  "directory structure", "Mapping CTS constructs → Rust")
- `yawgpu-test/src/lib.rs` (existing `ValidationTest`, `assert_device_error!`,
  `real_backend_available`, `assert_current_device_error_after`)
- `yawgpu/tests/future_modes.rs` (existing future-polling logic to promote)
- `yawgpu/tests/buffer_creation_validation.rs` (existing port style)
- `specs/reference/naming-conventions.md`, `CLAUDE.md`

Produce (in `yawgpu-test`, with inline `#[cfg(test)] mod tests` per
principle 1 — every new `pub fn`/macro gets a direct unit test):

1. **`expect_no_validation_error`** — run a closure, assert the device
   error sink is empty afterward. Method on `ValidationTest` and/or a
   free fn mirroring `assert_current_device_error_after`. Positive
   counterpart to `assert_device_error!`.

2. **Cartesian-product helper** for subcase tables (CTS `u.combine()`).
   Either a small `pub fn cartesian` returning owned tuples/`Vec`, or a
   `combine!` macro. Must read cleanly at a 2–3 dimension call site:
   ```rust
   for (usage, mapped) in cartesian2(&USAGES, &[false, true]) { … }
   ```
   Keep it minimal — no proc-macro, no extra deps.

3. **Feature/limit-gated device builder** — `ValidationTest::with_features(&[WGPUFeatureName])`
   and `ValidationTest::with_limits(WGPULimits)` (builder or `new_*`
   constructors) so `capability_checks/*` can request a device at/over a
   limit and assert the create call routes to the error sink. Must work
   on Noop (use the default-advertised limit/feature set as the baseline).

4. **Future-poll helper** — drive a `WGPUFuture` to completion on Noop
   (for `mapAsync` / `createRenderPipelineAsync` / future-modelled
   `popErrorScope`). Promote the logic from `yawgpu/tests/future_modes.rs`
   into a reusable `yawgpu-test` fn; leave `future_modes.rs` working
   (call the new helper from it, or keep its local copy and note the
   shared one is canonical).

5. **`tests/cts/` scaffolding**: create the directory
   `yawgpu/tests/cts/validation/buffer/` and one **worked** spec file
   `create.rs` that ports `$CTS/src/webgpu/api/validation/buffer/create.spec.ts`
   — **all 5 cases** (`size`, `limit`, `usage`, `new_usages`,
   `createBuffer_invalid_and_oom`), each as its own `#[test]`. Port them
   independently: this is a self-contained CTS layer, so port every case
   even though `buffer_creation_validation.rs` overlaps — do **not**
   dedupe against it. You may read that file for the Rust idiom; a file
   header comment should cite the CTS source path. Add the aggregator
   binary `yawgpu/tests/cts_validation_buffer.rs`:
   ```rust
   #[path = "cts/validation/buffer/create.rs"] mod create;
   ```
   This both proves the `#[path]` aggregator pattern compiles under
   Cargo and serves as the template for every later area.

Out of scope: any other CTS area (Phase B), real backends, operation
tests, spec edits, commits, changing `yawgpu-core`/`yawgpu`/`yawgpu-hal`
production code beyond what `with_features`/`with_limits` strictly need
(if a device-creation path is missing, flag it — do not silently widen
core).

Acceptance criteria:
- [ ] all four harness primitives exist with inline unit tests; existing
      `yawgpu-test` surface unchanged in behaviour
- [ ] `tests/cts/validation/buffer/create.rs` ports all 5 CTS
      `buffer/create` cases as `#[test]` (independently, not deduped
      against the legacy test); header comment cites the CTS source path
- [ ] `cts_validation_buffer.rs` aggregator compiles and runs green on Noop
- [ ] `cargo test` green on Noop, no GPU; `cargo clippy --workspace
      --all-targets -- -D warnings` clean on this host; CLAUDE.md
      conventions met (`#![warn(missing_docs)]`, no panics in lib code)
- [ ] `specs/tracking/cts-coverage.md`: `buffer/create.spec.ts` flipped to
      `ported` with the Rust file recorded (Claude updates this on review)

Report back: files changed; harness API shape chosen (free fn vs method,
macro vs fn for cartesian); any CTS `buffer/create` case intentionally
deferred (+why); any core gap surfaced by `with_features`/`with_limits`.
