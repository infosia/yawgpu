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

1. **TDD against Dawn.** Dawn's `dawn/src/dawn/tests/unittests/validation`
   is the executable spec. The cycle for every API area is:
   Red (port the Dawn test, it fails) → Green (minimal impl) → refactor.
   Never implement an API surface that has no ported test covering it.
2. **Noop-first.** Every validation test must pass on the Noop HAL backend
   with no GPU. Real-backend (Vulkan/Metal) work is gated and never required
   for CI.
3. **No panics in library code.** `yawgpu-core` and `yawgpu-hal` return
   `Result`; use `?`. The single exception is the **FFI boundary** in the
   `yawgpu` crate: invalid C handles/null where the spec forbids null may
   `expect(...)` (mirrors wgpu-native), but spec-level validation failures
   must route to the device error sink, not panic.
4. **Arc-based handles.** Every WebGPU object is `Arc<XxxImpl>`. C handles are
   `Arc::into_raw` / reconstructed; `wgpuXxxRelease` drops one ref,
   `wgpuXxxAddRef` clones. `Drop` releases backend resources.

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

1. Write/extend `specs/blocks/<area>.md` — extract rules from the Dawn test.
2. Port the Dawn test file to `yawgpu/tests/<area>_validation.rs` (Red).
3. Implement minimally in `yawgpu-core` + `yawgpu` (Green).
4. Verify on Noop; log in `specs/tracking/phase-N.md`.
5. Refactor for reuse/clarity before moving on.

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
