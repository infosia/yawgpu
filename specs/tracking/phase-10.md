# Phase 10 — Public-API unit-test catch-up

Status: **in progress** (P10.0 active). Rules/plan:
`../blocks/90-unit-tests.md`. Roles/loop:
`../reference/workflow.md`.

Phase 10 closes the unit-test gap left by the Phase 0–9 "TDD
against Dawn" bootstrap methodology. Now that the C ABI, the
HAL backends (Metal+Vulkan via Phase 7 e2e), and the C example
ports (Phase 9) are real-GPU-verified, the going-forward rule is
**every `pub fn` has a direct unit test** — see
`../blocks/90-unit-tests.md` for policy + scope.

## Inventory (snapshot 2026-05-20)

| Crate / file | `pub fn` count | direct unit tests today |
|---|---:|---:|
| `yawgpu/src/lib.rs` (C FFI) | 169 | 2 |
| `yawgpu/src/conv.rs` + other | ~20 | 0 |
| `yawgpu-core/src/lib.rs` | 207 | 5 |
| `yawgpu-hal/src/lib.rs` | 25 | 1 |
| `yawgpu-hal/src/noop/mod.rs` | 14 | 0 |
| `yawgpu-hal/src/metal/mod.rs` | 25 | 0 |
| `yawgpu-hal/src/vulkan/mod.rs` | 22 | 0 |
| **Total** | **~482** | **8** |

(`pub fn` counts via `grep -cE "^\\s*pub (unsafe )?fn "`; the
`yawgpu-core` 207 figure includes some legitimately cross-crate-
internal items that may be narrowed to `pub(crate)` during P10.3
triage.)

## P10.0 — Policy + CLAUDE.md update *(in progress)*

Update CLAUDE.md core principle #1 from "TDD against Dawn" to
"Every public API has a direct unit test"; add a historical
note pointing to the Phase 0–9 bootstrap methodology. Create
`../blocks/90-unit-tests.md` (the policy block). Land both in
the same commit as `phase-10: P10.0 — public-API unit-test
policy + CLAUDE.md update`. No production code change; no test
count change.

**Verification:** Noop `cargo test --workspace` byte-for-byte
unchanged (58 binaries / 287 tests). Phase-7 e2e unchanged.
This slice only ships docs/specs.

## P10.1 — `yawgpu-hal` unit tests  *(after P10.0)*

Add inline `#[cfg(test)] mod tests` to each of the four HAL
source files. Tests live in the same file unless the file is
over ~600 lines, in which case a sibling `tests.rs` is fine.

- `noop/mod.rs` (14 pub fn): assert each Noop method's
  baseline behavior (return values, allocation count
  bookkeeping, no panic, no side effects). Noop is the **spec
  baseline** — these tests fix the contract that Metal/Vulkan
  must match.
- `lib.rs` (25 pub fn): cover HAL enum-dispatch fns on the
  Noop arm. The Metal/Vulkan arms are covered transitively by
  the backend tests + e2e suite.
- `metal/mod.rs` (25 pub fn): `#[cfg(feature="metal")]` +
  `#[ignore]` per fn (matches the e2e pattern). Test happy
  path + at least one error case where applicable. Claude
  runs the full ignored sweep on the host per slice.
- `vulkan/mod.rs` (22 pub fn): same pattern as metal.

May ship as one slice or split into noop+lib (P10.1a) and
metal+vulkan (P10.1b) depending on size.

## P10.2 — `yawgpu/src/conv.rs` + sibling files

`conv.rs` is mostly pure data-conversion code (C↔Rust enum
mapping, bitfield decoding) — easy to unit-test in isolation
with high signal-to-noise. Add tests for every public fn /
macro-generated fn. Also covers any `pub fn` in
`yawgpu/src/*.rs` outside `lib.rs`.

## P10.3 — `yawgpu-core` pub fn audit + tests

Two-step:
1. **Audit** — go through every `pub fn` in `yawgpu-core/src/
   lib.rs` and decide: keep `pub` (and test), narrow to
   `pub(crate)` (no test required), or delete (dead code).
   Most of the 207 figure is expected to narrow.
2. **Test** — add inline unit tests for everything that stays
   `pub`.

Likely ~50–80 fns will remain `pub` after audit; budget
accordingly.

## P10.4 — `yawgpu/src/lib.rs` (C FFI) unit tests

169 `pub unsafe extern "C" fn` to cover. Likely split into
4–5 sub-slices by API area:

- **P10.4a** — Instance / Adapter / Device / Queue creation
  +  Release / AddRef family.
- **P10.4b** — Buffer (create / map / unmap / write /
  GetMappedRange / GetConstMappedRange / Release / AddRef
  / Destroy).
- **P10.4c** — Texture / TextureView / Sampler / their
  Release+AddRef.
- **P10.4d** — Command encoder / RenderPass / ComputePass /
  Bundle / their Release+AddRef.
- **P10.4e** — Pipeline (Compute, Render, PipelineLayout,
  BindGroupLayout, BindGroup, ShaderModule) + Query +
  ErrorScope + Surface.

Test helper module in `yawgpu/tests/common/` (or inline) to
create a Noop instance/device for FFI tests without
boilerplate.

## Phase 10 Review *(mandatory, after P10.4)*

Same workflow as Phase 9: fresh no-context subagent reviews
cumulative diff, emits CRITICAL/MAJOR/MINOR, fix in severity
order, no open CRITICAL/MAJOR → Phase 10 COMPLETE.

## Exit criteria

- Every `pub fn` per the scope in `../blocks/90-unit-tests.md`
  has at least one direct unit test asserting its behaviour.
- Coverage table in `tracking/phase-10-coverage.md` (per-file
  `pub fn` ↔ test name) up to date.
- Noop `cargo test --workspace` green + clippy clean; Phase-7
  e2e + Phase-9 example runs unchanged (regression).
- One commit per slice. Mandatory Phase 10 Review logged in
  `tracking/phase-10-review.md`.
