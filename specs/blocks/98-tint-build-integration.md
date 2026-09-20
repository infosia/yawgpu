# Block 98 — Tint build integration (`yawgpu-tint/build.rs`)

Post-core, user-requested. Not a Dawn port: this block specifies the
**build-time contract** of `yawgpu-tint/build.rs` — how it decides between
linking real Tint and degrading to the non-functional stub, and when that
decision must be re-evaluated.

Background: `specs/reference/dependencies.md` → "Crate dependencies" (Tint),
`README.md` → "Using it from C".

## Surface

Not a WebGPU surface. The observable artifacts are:

- The `have_tint` rustc cfg emitted by `yawgpu-tint/build.rs`.
- `yawgpu_tint::HAVE_TINT` (`pub const`, already exists) — the runtime-visible
  projection of that cfg.
- The `cargo:warning=` diagnostic emitted when the build degrades to the stub.

## Problem (observed 2026-09-20, Linux clean install)

`resolve_dawn_dir()` decides stub-vs-real by probing two files:

- `<workspace>/third_party/dawn/CMakeLists.txt` (submodule initialized)
- `<workspace>/third_party/dawn/third_party/abseil-cpp/CMakeLists.txt`
  (`fetch_dawn_dependencies.py` has run)

but `build.rs` emits `cargo:rerun-if-changed=` only for `shim/tint_shim.cpp`,
`shim/tint_shim.h`, `shim/CMakeLists.txt` and a set of env vars. **The inputs to
the decision are not among the rerun keys.** Consequences:

1. A `cargo build` run *before* the one-time Dawn setup completes caches the
   "stub" decision for that profile. Completing the submodule setup afterwards
   does not invalidate it — cargo never re-runs the build script, so the profile
   stays stubbed indefinitely.
2. The failure is silent: a single `cargo:warning` line, exit status 0, and a
   `libyawgpu.so` that links and loads but cannot compile any shader.
3. The decision is cached **per profile**, so `debug` and `release` can disagree.
   Observed exactly this: `cargo test --workspace` fully green on `debug`
   (1013 passed) while `cargo build -p yawgpu --release` — the command the
   README's "Using it from C" section tells users to run — silently produced a
   stubbed `libyawgpu.so` with no `libtint_shim.so` beside it.
4. Running the documented setup in the documented order is not sufficient to
   recover; nothing in the repo documents `cargo clean -p yawgpu-tint` as the fix.

The ordering that triggers it (build once, then initialize the submodule) is the
common path on a fresh clone, so this is a first-run trap, not an edge case.

## Rules

- **R1 — Rerun keys cover the decision inputs.** `build.rs` emits a
  `cargo:rerun-if-changed=` for every path `resolve_dawn_dir()` probes, so any
  change to the Dawn checkout's presence re-evaluates the stub-vs-real decision.
  The keys are emitted **unconditionally and before** the probe, so they are
  registered on the stub path too — that is the path that must recover.

- **R2 — Absent Dawn re-probes every build.** With the vendored checkout absent
  or incomplete, at least one rerun key names a non-existent path. Cargo treats a
  missing `rerun-if-changed` path as dirty, so the build script re-runs on every
  build and picks up the checkout the moment it appears. No `cargo clean` is
  required to go from stub to real.

  **Known limitation (measured 2026-09-20).** Cargo's freshness check is
  mtime-based: once the probe path *exists* again, the script re-runs only if
  that file's mtime is newer than the last build-script run. Restoring a probe
  file with an **older** mtime — a `mv` of a pre-existing tree, an archive
  extracted with preserved timestamps, a snapshot restore — leaves cargo
  considering the crate fresh, and the stub decision stands. The documented
  setup does not hit this (`git submodule update --init` and
  `fetch_dawn_dependencies.py` both write files with a current mtime), and R5's
  `cargo clean -p yawgpu-tint` is the escape hatch when it does. Verified
  empirically: with the path missing the script re-runs on *every* build (three
  consecutive builds, three distinct run timestamps); restored with a fresh
  mtime it re-runs and produces a real Tint build with no `cargo clean`;
  restored with an mtime 33 ms older than the previous run it does not re-run.

- **R3 — Present Dawn does not thrash.** With the checkout complete, every rerun
  key names an existing file, so the build script is **not** re-run on an
  otherwise-unchanged build. A no-op `cargo build` after a successful Tint build
  must not reconfigure or rebuild CMake. (R2 and R3 together: the re-probe cost is
  paid only while the setup is genuinely incomplete.)

- **R4 — Reverse transition detected.** Removing or de-initializing the submodule
  after a real build makes a probed path vanish, which is a `rerun-if-changed`
  change, so the next build re-evaluates and degrades to the stub rather than
  attempting to build against a missing tree.

- **R5 — Actionable degradation diagnostic.** The stub `cargo:warning` states
  (a) that Tint is unavailable and shader compilation will fail at run time,
  (b) the two setup commands (`git submodule update --init third_party/dawn`,
  `python3 tools/fetch_dawn_dependencies.py`), and (c) that a build cached from
  before the setup is cleared with `cargo clean -p yawgpu-tint`. Point (c) is
  retained even though R1/R2 make it unnecessary going forward, because a tree
  poisoned by the pre-fix build.rs still needs it once.

- **R6 — `YAWGPU_DAWN_DIR` override unchanged.** A non-empty `YAWGPU_DAWN_DIR`
  still short-circuits the vendored probe and is trusted as-is (no probing of its
  contents). Its existing `rerun-if-env-changed` key is unaffected. R1's keys are
  still emitted in this case — harmless, and it keeps the emission
  unconditional per R1.

## Unit tests (principle 1)

`resolve_dawn_dir()`'s predicate is currently inlined and untestable. Extract the
pure part as a helper taking the candidate root — e.g.
`fn dawn_probe_paths(root: &Path) -> [PathBuf; 2]` plus
`fn dawn_checkout_usable(root: &Path) -> bool`:

- `dawn_probe_paths` returns the two documented paths for a given root (R1).
- `dawn_checkout_usable` is `false` for an empty tempdir, `false` for a root with
  only `CMakeLists.txt` (initialized but unfetched), `true` for a root with both
  probe files (R2/R4), and tracks the transition in both directions.

### R7 — those tests must actually execute (corrected 2026-09-20)

**Correction.** This block originally asserted "Build-script unit tests run under
`cargo test -p yawgpu-tint`". That is false and the acceptance criterion built on
it was unsatisfiable: cargo compiles `build.rs` *without* `--test`, so a
`#[cfg(test)] mod tests` inside a build script is never built into a libtest
harness and never runs. `yawgpu-tint/Cargo.toml` declares no test target and
nothing includes `build.rs` into one. Consequence: the pre-existing
`android_abi_for_arch_maps_supported_targets` test **has never executed since it
was written** — it type-checks nothing and asserts nothing.

- **R7 — Build-script logic is covered by a test target that actually runs.**
  The pure, filesystem-only helpers (`dawn_probe_paths`, `dawn_checkout_usable`,
  `android_abi_for_arch`) move into a standalone module file at the crate root,
  e.g. `yawgpu-tint/build_probe.rs`, which is included with `#[path]` by **both**
  `build.rs` and a real test target `yawgpu-tint/tests/build_probe.rs`. Their
  unit tests move with them.

  This shape is chosen over adding a `[[test]]` target that includes `build.rs`
  wholesale: `build.rs` references the `cmake` crate, which is a
  **build-dependency** and therefore not linkable from a test target, so the
  wholesale approach would need `cfg` gates around `main()` and
  `copy_runtime_shim()`. Splitting the pure part out needs neither — `main()`,
  the CMake invocation and `copy_runtime_shim()` stay exactly as they are.

  After R7, `cargo test -p yawgpu-tint` executes these tests, and they are part
  of the `cargo test --workspace` Noop gate.

## Docs

- `README.md` → "Using it from C": after the submodule setup block, note that a
  `cargo build` run before the setup completes must be cleared with
  `cargo clean -p yawgpu-tint` (pre-fix trees only), and that
  `yawgpu_tint::HAVE_TINT` / the absence of `libtint_shim.{so,dylib,dll}` next to
  `libyawgpu.*` is how to tell a stubbed build apart.
- `specs/reference/dependencies.md`: record the rerun-key invariant (R1) next to
  the existing Tint build notes, so a future edit to `resolve_dawn_dir()` keeps
  the probe set and the rerun set in sync.

## Acceptance criteria

1. R1–R6 hold; R1's invariant is visible in `build.rs` (probe set and rerun set
   derived from one source, not two hand-maintained lists).
2. R7 holds: `cargo test -p yawgpu-tint` **runs** the probe-helper tests (they
   appear by name in its output, in a binary other than `unittests src/lib.rs`),
   and they pass. A deliberately broken assertion in one of them must make
   `cargo test -p yawgpu-tint` fail — verify this once, then revert it. Without
   that check the criterion cannot distinguish "passing" from "not running".
3. Manual gate — remove one probed path (hiding
   `third_party/dawn/third_party/abseil-cpp/CMakeLists.txt` is enough and far
   cheaper than de-initializing the submodule):
   `cargo build -p yawgpu --release` emits no `rustc-cfg=have_tint` and warns
   per R5 → restore the file **with a current mtime**, as a real fetch would →
   `cargo build -p yawgpu --release` **without any `cargo clean`** re-runs the
   script and produces `libtint_shim.so` beside `libyawgpu.so` with
   `rustc-cfg=have_tint`. Assert on the build script's `output` file, not on the
   presence of `libtint_shim.so` — a stub build leaves the previously built shim
   in place, so the file alone does not prove a real build.
4. R3 gate — a second `cargo build -p yawgpu --release` immediately after (3)
   is a no-op (no CMake reconfigure, no recompile).
5. Full Noop suite green: `cargo test --workspace`.
6. Clippy gate clean: `cargo clippy --workspace --all-targets -- -D warnings`.

## Open questions

- Whether to make a stubbed build a hard error behind an opt-in env var
  (`YAWGPU_REQUIRE_TINT=1`) for CI/packaging, so a release artifact can never
  ship stubbed. Deferred — not required by this block.
