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

## Backend support tiers

| Tier | Backends | Meaning |
|---|---|---|
| **Tier 1 — Supported** | Vulkan, Metal | webgpu.h semantics fully mapped. Phase Review-clean implies "conformant on real GPU" within the slices brought up so far. API and behaviour changes follow normal SemVer discipline. |
| **Tier 2 — Experimental (best-effort)** | GLES (Android + Windows ANGLE) | Behind opt-in `gles` cargo feature; never in `default`. webgpu.h paths that cannot be cleanly mapped to GLES 3.1 may be **rejected at the HAL layer with `HalError`**, which `yawgpu-core` surfaces as a device error. Feature set and behaviour may change without SemVer guarantees. Real-GPU verification is manual on Windows ANGLE only. |

**Operational rule (Tier-independent core validation).** `yawgpu-core`
validation is identical regardless of the backend tier — a rule that
fires for Vulkan/Metal must fire identically for GLES. Tier 2 is
"best-effort" only at the **HAL execution** layer: the GLES backend may
return `HalError` for a validated WebGPU operation that has no clean
GLES 3.1 mapping. Such cases must be catalogued in
`specs/blocks/67-gles-backend.md` ("WebGPU × GLES mapping matrix"),
never silently widened in core. **Never relax a core rule to make a
Tier 2 backend pass.**

**D3D backends (D3D11 / D3D12) remain permanently out of scope.**

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

### CTS conformance — webgpu-native-cts

WebGPU CTS conformance (`api/validation` + `api/operation`) is verified
**externally** by
[webgpu-native-cts](https://github.com/infosia/webgpu-native-cts), which links
the `webgpu.h` C ABI directly and runs each case against a real GPU (Metal +
Vulkan/MoltenVK) with **Dawn** as the oracle — not by in-repo Rust ports. The
in-repo test layers are: the inline `#[cfg(test)]` unit tests (principle 1 — the
primary public-API coverage), the Dawn-ported `yawgpu/tests/*_validation.rs`
Noop regression tests, and the `yawgpu/tests/e2e_{metal,vulkan,gles}_*.rs`
real-GPU tests. A yawgpu divergence the CTS suite finds is root-caused and fixed
in the library (with its inline unit test), then re-confirmed on hardware; the
finding ledger is `specs/tracking/cts-coverage.md` and the suite's
`docs/FINDINGS.md`.

## Shader frontend — Tint

The WGSL→{MSL, SPIR-V, GLSL ES} compiler and reflection source is **Tint** (Dawn's
WGSL compiler), vendored as the pinned `third_party/dawn` git submodule and driven
from Rust through the `yawgpu-tint` crate's C++ shim. Since Tint is the same
compiler the CTS oracle (Dawn) uses, shader translation matches the oracle by
construction. The default build links Tint, so it **requires the Dawn submodule
initialized + its deps fetched** (`git submodule update --init third_party/dawn`,
then `tools/fetch_dawn_dependencies.py`; see `specs/reference/dependencies.md`);
without it `yawgpu-tint` is a non-functional stub. Shader-compiler issues are fixed
in the `yawgpu-tint` shim (or upstream Dawn/Tint) — **there is no longer a naga fork
to edit**. (Tint replaced the earlier naga frontend; the historical naga/`../wgpu`
fork workflow is obsolete.)

## Code conventions

- `#[non_exhaustive]` on extensible public enums/structs.
- `#[must_use]` on builders and handle-producing fns.
- Every public item (`pub fn`, `pub struct`/fields, `pub enum`/variants,
  `pub const`, `pub type`, `pub trait`) in `yawgpu`, `yawgpu-core`, and
  `yawgpu-hal` carries a `///` doc comment. This is enforced by
  `#![warn(missing_docs)]` at each crate root and escalated to an error by the
  `-D warnings` clippy gate. Generated `yawgpu::native` bindings are exempt via
  `#[allow(missing_docs)]`.
- Colocate `Device::create_*` logic with the created type's module, not in
  one giant `device.rs` (mgpu convention).
- HAL is **static enum dispatch**, never `dyn Trait`:
  `enum HalDevice { Noop(..), Vulkan(..), Metal(..), Gles(..) }`, backends
  `cfg`-gated. (`Gles` arm is Tier 2; see "Backend support tiers" above.)
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
5. Verify on Noop; log in the area's tracking doc
   (`specs/tracking/<topic>.md`). Per-phase `phase-N.md` logs are no
   longer written.
6. Refactor for reuse/clarity before moving on.

**Every phase ends with a mandatory Phase Review ("Clean Review Then
Fix"):** a fresh no-context subagent reviews the phase's cumulative diff
and emits `CRITICAL`/`MAJOR`/`MINOR` findings; findings are fixed in
severity order; a phase cannot be COMPLETE with any open CRITICAL/MAJOR.
Full process: `specs/reference/workflow.md` → "Phase Review".

## Out of scope (initially)

- **D3D backends (D3D11 / D3D12).** Permanently out of scope. (GLES is
  now Tier 2 / experimental — see "Backend support tiers" above and
  `specs/blocks/67-gles-backend.md` for the Android + Windows ANGLE
  bring-up plan.)
- Dawn `wire/` tests — they validate dawn-wire IPC, which yawgpu has no
  analog for. The C ABI boundary is our equivalent boundary.
- Dawn `end2end` tests — deferred to Phase 7 (real backends), GPU-gated.

## Privacy / repo hygiene

- No credentials, signing material, or device-specific secrets committed.
- `.gitignore`: `target/`, `.claude/`, local test transcripts.
- Generated bindings are build artifacts (`$OUT_DIR`), not committed.

## Tooling — sandbox

- **Avoid `dangerouslyDisableSandbox: true` whenever possible.** Prefer
  sandboxed Bash commands. Only disable when there is no alternative —
  e.g. real-GPU Metal e2e runs (the Bash sandbox blocks
  `MTLCopyAllDevices()` in spawned child processes; see
  [[claude-runs-real-gpu-tests]]) or a specific operation that has
  already been shown to fail under the sandbox in this session. Network
  ops (`git push`/`pull`) and other ad-hoc commands should be invoked
  by the user via the `!` prompt, not run by Claude with the sandbox
  disabled.
