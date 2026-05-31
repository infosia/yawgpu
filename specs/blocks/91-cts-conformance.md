# Block 91 — WebGPU CTS conformance porting (validation first)

This block defines how the **WebGPU Conformance Test Suite** (CTS,
the normative TypeScript suite `gpuweb/cts`) is ported onto the yawgpu
**C FFI** as Rust integration tests. It is the going-forward source for
*new* spec-conformance test work, layered on top of the Phase 0–9
Dawn-ported validation tests (see `90-unit-tests.md` for the
relationship between unit tests, Dawn ports, and this effort).

> **CTS path convention.** All `src/webgpu/…` paths in this block and in
> `tracking/cts-coverage.md` are **relative to the root of a local CTS
> checkout**, denoted `$CTS`. Set it to wherever `gpuweb/cts` is cloned
> on your machine; no fixed sibling location is assumed by these docs.

CTS is **not executed** against yawgpu — it targets a JavaScript
`navigator.gpu` implementation in a browser / Node. Instead, each
`g.test(...)` case is **read for intent and re-expressed** as a Rust
test driving `webgpu.h` through the `yawgpu` crate. The CTS file is the
executable spec; the Rust port is its conformance witness.

The live coverage matrix (every CTS `*.spec.ts` → mapped yawgpu test →
status) is `specs/tracking/cts-coverage.md`. This block is the
*methodology*; that file is the *ledger*.

## Scope

- **In scope now:** `src/webgpu/api/validation/` — 129 spec files /
  704 `g.test()` cases. These are error-path / rule-validation tests
  that run on the **Noop** HAL with no GPU, so they belong in CI and
  reuse the existing `yawgpu-test::ValidationTest` harness.
- **Deferred (later phases):**
  - `src/webgpu/api/operation/` — 72 spec files. Behavioural / readback
    tests; require a real GPU. Port as `e2e_*` (`#[ignore]`,
    Metal/Vulkan-gated) on top of a new `OperationTest` harness
    (CTS `GPUTest` analogue). Tracked separately when that phase opens.
  - `src/webgpu/shader/validation` + `shader/execution` — WGSL surface;
    large, tied to the shader-passthrough work; its own initiative.

## Out of scope (permanently, for this port)

These CTS areas have no C-ABI analogue and are **excluded**, flagged
`N/A` in the matrix with the reason:

- `web_platform/`, `canvas`, `external_texture`, `compat/`,
  `queue/copyToTexture/CopyExternalImageToTexture` — web/canvas/WebCodecs
  surfaces. yawgpu has no `HTMLCanvasElement` / `ImageBitmap` /
  `VideoFrame`.
- `idl/` — checks the JavaScript IDL surface; the C header surface is a
  separate concern, not a CTS port.
- `stress/`, `manual/`, `demo/` — not normative.
- Individual **web-only subcases** inside an otherwise-portable spec
  (e.g. `createBindGroup` `external_texture,*`, `state/device_lost`
  `importExternalTexture` / `copyExternalImageToTexture,*`,
  `texture_formats` `canvas_configuration*`) are dropped; the spec still
  reaches `ported` once its non-web cases are done, with the excluded
  subcases noted in the matrix (`ported*`).

D3D backends remain permanently out of scope (CLAUDE.md). GLES is
Tier 2 — a validation rule fires identically on every backend
(Tier-independent core validation), so these tests are **not**
backend-specific; they assert `yawgpu-core` behaviour on Noop.

## Directory structure (CTS-mirrored)

CTS directory layout is mirrored 1:1 under `yawgpu/tests/cts/`:

```
yawgpu/tests/cts/validation/
  buffer/        { create.rs, destroy.rs, mapping.rs }
  texture/       { create_texture.rs, create_view.rs, destroy.rs, ... }
  encoding/cmds/ { copy_buffer_to_buffer.rs, ... }
  capability_checks/limits/ { max_bind_groups.rs, ... }
  ...
```

Cargo only auto-discovers test **binaries** from files directly under
`yawgpu/tests/`. Subdirectories are not compiled on their own. So each
CTS second-level area gets **one thin aggregator binary** at the
`tests/` root that pulls its subtree in with `#[path]`:

```rust
// yawgpu/tests/cts_validation_buffer.rs
#[path = "cts/validation/buffer/create.rs"]   mod create;
#[path = "cts/validation/buffer/destroy.rs"]  mod destroy;
#[path = "cts/validation/buffer/mapping.rs"]  mod mapping;
```

This keeps the on-disk tree identical to CTS, gives one test binary per
area (≈13 for validation — parallel compile, bounded link cost), and
keeps shared helpers in the `yawgpu-test` crate reachable from all of
them. Aggregators are added per area as that area's slice is dispatched;
do not pre-create empty ones.

**File naming.** Rust module files use `snake_case`; the CTS spec name
is preserved as closely as Rust allows (`copyBufferToBuffer.spec.ts` →
`copy_buffer_to_buffer.rs`). Each `g.test('foo,bar')` case maps to a
`#[test] fn foo_bar()` (commas → underscores). A doc comment at the top
of each Rust file cites the CTS source path it ports.

## Mapping CTS constructs → Rust

| CTS construct | Rust port |
|---|---|
| `g.test('a,b').fn(t => …)` | `#[test] fn a_b()` in the mirrored file |
| `t.device`, `t.queue` | `ValidationTest::new()` → `.device()`, queue via FFI |
| `t.expectValidationError(() => …)` | `assert_device_error!{ … }` |
| "should succeed" / no error | `expect_no_validation_error` (new harness helper) |
| `t.expectGPUError('out-of-memory', …)` | error-scope helper + filter on error type |
| `await t.device.popErrorScope()` | error-sink inspection (`ValidationTest::errors`) |
| `await buffer.mapAsync(…)` etc. | Future-poll helper (drive `WGPUFuture` to completion) |
| `createRenderPipelineAsync` etc. | Future-poll helper |
| `u.combine('k', [v…]).combine(…)` | table-driven: cartesian-product helper → loop of subcases |
| device with feature/limit X | `ValidationTest::with_features(…)` / `with_limits(…)` (new) |
| `t.skip(…)` on missing feature | early `return` after `real_backend`/feature probe |

Subcases (`.params(u => u.combine(…))`) are ported as an in-test loop
over a `&[…]` table or via the cartesian-product helper, with each
iteration labelled (e.g. `eprintln!`/`assert!(…, "case {i}: …")`) so a
failure identifies the offending combination — the Rust test stays one
`#[test]` per `g.test()`.

## `yawgpu-test` harness contract (extensions needed)

Phase A adds these to the `yawgpu-test` crate (implemented by the
coding agent; see the Phase A handoff). Existing surface:
`ValidationTest`, `assert_device_error!`, `real_backend_available`,
`assert_current_device_error_after`.

New, required before bulk porting:

1. **`expect_no_validation_error`** — positive counterpart to
   `assert_device_error!`: run a closure, assert the error sink stayed
   empty (CTS's implicit "this should succeed").
2. **Cartesian-product helper** — replicate `u.combine()` so subcase
   tables are declared once and iterated, e.g.
   `for (a, b) in cartesian(&AS, &BS) { … }` or a `combine!` macro.
3. **Feature/limit-gated device builder** — `ValidationTest::with_features(&[…])`
   and `with_limits(WGPULimits)` so `capability_checks/*` can request a
   device at/over a limit and assert the create call errors. Must work
   on Noop (Noop advertises the default limit set).
4. **Future-poll helper** — drive a `WGPUFuture` to completion on Noop
   for async cases (`mapAsync`, `createRenderPipelineAsync`,
   `popErrorScope` where modelled as a future). Promote/consolidate the
   logic already in `yawgpu/tests/future_modes.rs`.

Harness changes ship with their own inline unit tests (principle 1).

## Per-area porting workflow (one slice = one CTS area)

**The CTS port is a self-contained conformance layer, counted
independently of the legacy Dawn-ported tests.** Every non-excluded
CTS `g.test()` gets its own Rust `#[test]` under `tests/cts/validation/…`,
**even when a legacy `yawgpu/tests/*.rs` file already exercises the same
rule** — duplication across the two layers is allowed and expected
*during the port*. The legacy Dawn tests stay as-is while porting; do not
delete, merge, or dedupe against them in a porting slice. The "related
legacy test" column in the matrix is **informational only** (a pointer to
prior art a porter may consult for the Rust idiom), never a reason to
skip a CTS case.

**Planned legacy cleanup (deferred — Phase E).** Once the CTS port has
superseded a legacy Dawn-derived test's coverage, the redundant legacy
test is **deleted** in a dedicated later cleanup phase (user decision
2026-05-29), *not* during a porting slice. The "related legacy test"
column is the worklist for that phase: for each `ported`/`ported*` row,
Phase E confirms the CTS file covers everything the listed legacy file
did (the legacy test may still hold cases the CTS port deferred, e.g.
feature-gated subcases — those must be retained or re-homed, not lost),
then removes the legacy file. Until Phase E runs, both layers coexist.

**A CTS test asserts the spec-correct expectation — never the current
(possibly buggy) behaviour.** If `yawgpu-core` does not yet enforce a
rule the CTS case checks, you have exactly two honest options: (a) **fix
the core rule** (minimal, unit-tested, flagged in the report), or (b)
**mark the case `#[ignore = "core does not yet enforce X; CTS expects
<error>"]`** and record the spec as `partial` in the ledger with the gap
enumerated. You must **never invert or weaken the assertion** to make the
test green (e.g. asserting success where the spec requires a validation
error, or reducing a case to a trivially-passing stub). A green test that
encodes non-spec behaviour is worse than no test — it certifies a bug as
conformant. This is the test-side corollary of "never relax a core rule"
(CLAUDE.md).

For each area in the matrix:

1. **Read the CTS spec(s)** for the area. Optionally glance at the
   related legacy test (matrix column) for the Rust idiom — but port
   every CTS case regardless of overlap.
2. **Port every case** into the mirrored `cts/validation/<area>/`
   files, one `#[test]` per `g.test()`, subcases as in-test tables.
3. **Green on Noop**, no GPU. Any case that can only be a real-GPU
   behavioural check (not a validation rule) is **deferred to the
   operation phase**, not silently dropped — note it in the matrix.
4. **Update `specs/tracking/cts-coverage.md`**: flip the spec's status
   `todo` → `ported`, record the Rust file.
5. Slice review (Claude) against acceptance criteria below.

A whole CTS area being `ported` means: every non-`N/A` `g.test()` in
its spec files has a corresponding Rust `#[test]` (subcases covered by
the in-test table), green on Noop.

## Completion report (`REPORT.md`)

When a slice is finished, the coding agent **writes its completion report
to `REPORT.md`** at the repo root (a fixed filename, `.gitignore`d like
`HANDOFF.md` — it is a working artifact, never committed). It does not
just reply in chat. This lets the reviewer (Claude) read deterministic
results instead of re-running and polling. The report must contain:

- **Verification** — the exact commands run and their **exit codes**.
  **The coding agent must NOT run `cargo test --workspace`** (it links 59+
  test binaries against naga and takes ~1 h in the codex sandbox — wasted
  effort). Instead the agent runs the cheaper, targeted gates:
  - lib unit tests: `cargo test -p yawgpu --lib` and `cargo test
    -p yawgpu-core --lib` (fast; catches most regressions — e.g. the FFI
    `ffi::tests` regressions a workspace run would also catch);
  - the specific touched integration binaries:
    `cargo test -p yawgpu --test <cts_validation_area>` (+ any legacy
    test it had to update);
  - `cargo clippy --workspace --all-targets -- -D warnings` (default +
    `--features tiled`), and `cargo build -p yawgpu-core --features tiled`.
  State pass/fail per command; paste the `test result:` summary lines;
  judge by exit code, not prose. **Claude runs the full `cargo test
  --workspace --release` (+ relevant features) during review** as the
  backstop — the agent does not. The backstop is **release** because
  optimization exposes UB that debug masks (e.g. a dangling-temporary
  test descriptor segfaulted only under release); release also builds/
  runs fast enough here. (Run a debug workspace pass too if a test could
  depend on debug-only behaviour such as overflow-check panics.)
- **Files changed** — production vs test, with a one-line why for each
  non-test file.
- **Per-spec case accounting** — ported / deferred / `N/A` counts, and
  which specific cases were deferred or excluded (+ reason).
- **Core gaps surfaced** — any `yawgpu-core` behaviour added or found
  missing (flagged, not silently widened).

Each handoff's "Report back" section points here; the reviewer reads
`REPORT.md` first.

## Acceptance criteria (per slice)

- [ ] every non-excluded `g.test()` in the area's spec file(s) has a
      corresponding `#[test]` (subcases enumerated in-test) under
      `tests/cts/validation/…` — overlap with a legacy Dawn test is fine
- [ ] excluded subcases are `N/A`-flagged with reason; deferred-to-
      operation cases noted — nothing silently dropped
- [ ] `cargo test` green on Noop, no GPU; no panics in
      `yawgpu-core`/`yawgpu-hal`; CLAUDE.md conventions met
- [ ] `specs/tracking/cts-coverage.md` status + Rust-file columns updated
- [ ] the area's aggregator binary `cts_validation_<area>.rs` wires in
      the new module files

## Phasing

- **Phase A — scaffolding** (this block + matrix + harness extensions +
  Cargo/dir wiring). Authored docs by Claude; harness + scaffolding by
  the coding agent.
- **Phase B — validation port**, area by area (see matrix grouping).
  Suggested order maximises early CI value and reuse:
  buffer → texture (createTexture/createView/createSampler/destroy) →
  image_copy → queue → render_pipeline (+ pipeline/compute_pipeline/
  shader_module/layout_shader_compat) → bind-group family
  (createBindGroup/Layout, getBindGroupLayout, createPipelineLayout) →
  render_pass → encoding (cmds/programmable/queries + dispatch/debug) →
  resource_usages → query_set → capability_checks (features → limits) →
  state/device_lost + error_scope.
  Validation port closes with a mandatory **Phase Review**.
- **Phase E — legacy Dawn-test cleanup** (deferred; user decision
  2026-05-29). After the validation port, delete the Dawn-derived
  `yawgpu/tests/*.rs` tests whose coverage the CTS port has superseded,
  using the ledger's "related legacy test" column as the worklist.
  Per row: confirm the CTS file covers everything the legacy file did
  (retain/re-home any feature-gated or deferred cases the CTS port did
  not take), then remove the legacy file. Runs as its own phase, not
  inside a porting slice.
- **Phase C — operation port** (deferred; real GPU, `OperationTest`).
- **Phase D — shader/WGSL** (deferred; separate initiative).

## Open questions

- Operation-phase `OperationTest` harness shape (readback / `expectContents`
  / tracked-resource lifetimes) — designed when Phase C opens.
- Whether `capability_checks/limits` (35 mechanical spec files) is worth
  one-`#[test]`-per-limit or a single table-driven generator over the
  limit set; decide at the B-capability_checks slice.
- `surface`-mapped limit subcases (`configure`/`getCurrentTexture` in
  `maxTextureDimension2D`) — map onto native surface or defer with the
  rest of surface validation.
