# CLAUDE.md — yawgpu permanent development rules

These rules are inherited/adapted from `mgpu`'s conventions and apply to all
work in this repository. Read `DESIGN.md` and `specs/SPEC.md` before coding.

## Roles (read first)

Implementation is done by a **separate coding agent**. **Claude plans and
orchestrates** — it authors `specs/`, emits task handoffs, reviews the coding
agent's diffs against acceptance criteria, runs `cargo build`/`cargo test`,
and manages git (`init`/`add`/`commit`). Claude does not write production
code; the coding agent does not plan, edit `specs/`, change scope, or commit.
Full detail: `specs/reference/workflow.md`.

## Language

- **All repository documentation, specs, comments, and identifiers: English.**
- Conversation with the user (chat responses): Japanese.

## Core principles

1. **Every public API has a direct unit test.** Any `pub fn` in
   `yawgpu` (the C FFI), `yawgpu-core`, or `yawgpu-hal` must have an
   inline `#[cfg(test)] mod tests` test that exercises it directly
   (happy path + error / edge cases as relevant). New public API
   ships in the same commit as its unit test; renaming or changing
   a public signature updates the test. Scope and definitions are
   in `specs/blocks/90-unit-tests.md`. Integration tests
   (`yawgpu/tests/*.rs`) and real-GPU e2e tests
   (`yawgpu/tests/e2e_*.rs`) remain as regression / spec-conformance
   layers on top — they don't replace the unit test for the
   underlying public fn.
2. **Noop-first.** Every validation/integration test must pass on
   the Noop HAL backend with no GPU. Real-backend (Vulkan/Metal)
   work is gated and never required for CI.
3. **No panics in library code.** `yawgpu-core` and `yawgpu-hal`
   return `Result`; use `?`. The single exception is the **FFI
   boundary** in the `yawgpu` crate: invalid C handles/null where
   the spec forbids null may `expect(...)` (mirrors wgpu-native),
   but spec-level validation failures must route to the device
   error sink, not panic.
4. **Arc-based handles.** Every WebGPU object is `Arc<XxxImpl>`. C
   handles are `Arc::into_raw` / reconstructed; `wgpuXxxRelease`
   drops one ref, `wgpuXxxAddRef` clones. `Drop` releases backend
   resources.

### Historical note — Dawn TDD (Phases 0–9)

The project was bootstrapped Phases 0–9 by porting Dawn's
`dawn/src/dawn/tests/unittests/validation` tests as
integration tests against the C FFI (`yawgpu/tests/*.rs`), with
that suite as the executable spec for each API area. **That was
the bootstrap methodology, not a permanent principle.** The
ported tests remain in the tree as a spec-conformance regression
layer and continue to run on every Noop gate, but new public API
work follows principle 1 above (direct unit tests) — porting a
Dawn test is optional, useful when it materially closes a
spec-coverage gap.

## Code conventions

- `#[non_exhaustive]` on extensible public enums/structs.
- `#[must_use]` on builders and handle-producing fns.
- Colocate `Device::create_*` logic with the created type's module, not in
  one giant `device.rs` (mgpu convention).
- HAL is **static enum dispatch**, never `dyn Trait`:
  `enum HalDevice { Noop(..), Vulkan(..), Metal(..) }`, backends `cfg`-gated.
- C↔Rust conversions live in `yawgpu/src/conv.rs` (macro-driven, like
  wgpu-native's `conv.rs`).
- bindgen output is `include!`d into a `pub mod native { ... }`; never edit
  generated code.

## Workflow per API area

1. Write/extend `specs/blocks/<area>.md` — describe the new
   public API + its behaviour contract.
2. Write the **inline unit test** for the new public fn (Red).
3. Implement the public fn (Green).
4. Optionally add an integration test in `yawgpu/tests/<area>.rs`
   when the API spans multiple objects and the unit test cannot
   reach the cross-object interaction (or port the Dawn test if
   it materially closes a spec-conformance gap).
5. Verify on Noop; log in `specs/tracking/phase-N.md`.
6. Refactor for reuse/clarity before moving on.

**Every phase ends with a mandatory Phase Review ("Clean Review Then
Fix"):** a fresh no-context subagent reviews the phase's cumulative diff
and emits `CRITICAL`/`MAJOR`/`MINOR` findings; findings are fixed in
severity order; a phase cannot be COMPLETE with any open CRITICAL/MAJOR.
Full process: `specs/reference/workflow.md` → "Phase Review".

## Out of scope (initially)

- **GL / D3D backends.** Primary platforms are Vulkan and Metal only. Do not
  add or stub OpenGL/OpenGL ES or DirectX HAL variants.
- Dawn `wire/` tests — they validate dawn-wire IPC, which yawgpu has no
  analog for. The C ABI boundary is our equivalent boundary.
- Dawn `end2end` tests — deferred to Phase 7 (real backends), GPU-gated.

## Privacy / repo hygiene

- No credentials, signing material, or device-specific secrets committed.
- `.gitignore`: `target/`, `.claude/`, local test transcripts.
- Generated bindings are build artifacts (`$OUT_DIR`), not committed.
