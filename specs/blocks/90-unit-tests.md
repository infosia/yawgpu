# Block 90 — Public-API unit-test policy (Phase 10+ ongoing)

This block defines the **going-forward unit-test policy** for
yawgpu. It supersedes the Phase 0–9 "TDD against Dawn" bootstrap
methodology as the primary testing rule. The Dawn-ported
integration tests in `yawgpu/tests/*.rs` and the real-GPU e2e
tests in `yawgpu/tests/e2e_*.rs` remain in the tree as
regression / spec-conformance layers, but new public API work
follows the rule defined here.

## Policy

**Every `pub fn` in `yawgpu`, `yawgpu-core`, and `yawgpu-hal` must
have a direct unit test.** "Direct" means an inline
`#[cfg(test)] mod tests` test in the same source file as the
`pub fn` (or a sibling module file in the same crate) that calls
the public fn explicitly and asserts on its behaviour or error.
Indirect coverage via integration tests does **not** satisfy the
rule.

A public-API-changing commit (new `pub fn`, signature change,
rename, deprecation) MUST update the unit test in the same
commit. Phase Review will flag a missing or stale unit test as
**MAJOR**.

## Definition of "public API"

| Crate | What counts | Test location |
|---|---|---|
| `yawgpu` (C FFI) | every `pub unsafe extern "C" fn wgpu*` and every `pub fn` / `pub const` exported from `yawgpu/src/lib.rs` or its modules (`conv.rs` etc.) | inline `#[cfg(test)] mod tests` in the same file |
| `yawgpu-core` | every `pub fn` reachable from outside the crate (i.e. not `pub(crate)` or stricter) | inline `#[cfg(test)] mod tests` in the same source file |
| `yawgpu-hal` | every `pub fn` in `lib.rs` (HAL enum dispatch) AND each backend's pub surface (`noop/`, `metal/`, `vulkan/`) | inline `#[cfg(test)] mod tests` in the same file; backend-only fns may gate the test on the corresponding `#[cfg(feature = "...")]` |

**Excluded from the rule** (do not require a unit test):
- `pub(crate)` / `pub(super)` / `pub(in path)` — restricted-
  visibility items are not public API.
- Derived impls (`Debug`, `Clone`, `Copy`, `Default`) — the
  derives' correctness is the language's responsibility.
- Public *types* with no `pub fn` (a `pub struct` with all
  private fields and no inherent impl). The constructors of
  such a type are covered when something else with a `pub fn`
  produces an instance and tests assert on observable
  behaviour through other `pub fn`s.
- Generated bindings (`include!("bindings.rs")` in
  `pub mod native`) — these are build artifacts.
- `Drop` impls — covered indirectly by the constructor's unit
  test plus leak / refcount assertions where reasonable.
- Trivial getters that simply return a private field (`#[inline]
  fn label(&self) -> &str { &self.label }`) MAY be covered by a
  single unit test against the constructor that asserts the
  getter reflects the constructor argument; an isolated test
  per getter is not required.

## Test shape

Each unit test should:
- Live in `#[cfg(test)] mod tests { use super::*; ... }` at the
  bottom of the source file, **or** in a sibling module
  (`#[cfg(test)] mod tests;` + `src/<file>/tests.rs`) when the
  file is over ~600 lines.
- Be named `<fn_name>_<scenario>` (e.g. `padded_bytes_per_row_
  aligns_to_256`, `wgpuBufferUnmap_returns_when_not_mapped`).
- Cover (a) the happy path, (b) at least one error or boundary
  case if the fn returns `Result` / can panic / has explicit
  validation, and (c) a roundtrip / inverse when relevant
  (e.g. `map_texture_format` ↔ `map_texture_format_back`).
- Be Noop-runnable wherever the fn is Noop-reachable. Real-GPU-
  only fns (Metal/Vulkan pub methods that require a live device)
  may gate the test on `#[ignore]` matching the existing e2e
  pattern; Claude runs those on the host.
- C FFI fns that take/return `WGPUSurface` / `WGPUDevice` etc.
  build the handle via the existing creation fn (also tested by
  this rule) and `Release` it at the end.

## Scope

This block governs Phase 10 (the catch-up pass to add unit tests
for the existing public API surface) and every subsequent phase.

- **P10.0** — this block + the historical-note edit in
  CLAUDE.md; CLAUDE.md core principle #1 changed from "TDD
  against Dawn" to "Every public API has a direct unit test"
  (done in the same commit as this block).
- **P10.1** — `yawgpu-hal` unit tests:
  - `noop/mod.rs` 14 pub fn
  - `lib.rs` 25 pub fn (HAL enum dispatch over Noop arm)
  - `metal/mod.rs` 25 pub fn (gated `#[cfg(feature="metal")]` +
    `#[ignore]`, real-GPU-run)
  - `vulkan/mod.rs` 22 pub fn (gated + `#[ignore]`, real-GPU-run)
- **P10.2** — `yawgpu/src/conv.rs` ~20 conversion fns +
  whatever other `pub fn` live in `yawgpu/src/*.rs` outside
  `lib.rs` (sibling modules).
- **P10.3** — `yawgpu-core` ~207 pub fn audit:
  - Triage `pub` → keep, narrow to `pub(crate)`, or test.
  - Add unit tests for everything that stays `pub`.
- **P10.4** — `yawgpu/src/lib.rs` 169 C FFI fns. Likely split
  into 3–5 sub-slices by area: instance/adapter/device, buffer
  /texture/sampler, command/pass/pipeline, query/error/surface.
- **Phase 10 Review** (mandatory) → COMPLETE.

## Exit criteria (Phase 10)

- Every `pub fn` defined above has at least one inline unit
  test asserting its behaviour. CI gate: a build-time check
  (script or `cargo deny`-style rule, TBD in P10.0) that fails
  if a `pub fn` exists without a sibling `#[test]` referencing
  it. (If the static check is too heavy for one phase, a
  `tracking/phase-10-coverage.md` table listing every pub fn ↔
  its test name is acceptable as an interim.)
- Noop `cargo test --workspace` remains green and grows in test
  count by ~the number of new unit tests; clippy clean.
- Real-GPU e2e (Phase-7) plus the new HAL backend unit tests
  remain green on the host.
- One commit per slice. Mandatory Phase 10 Review.
