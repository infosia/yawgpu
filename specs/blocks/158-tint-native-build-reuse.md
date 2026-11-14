# Block 158 — One Tint native build per configuration, shared across Cargo features and check/build

Status: **COMPLETE (2026-11-14)**. Phase Review: 0 CRITICAL, 0 MAJOR, 4 MINOR (m1, m2, m4 fixed in S2; m3 documented).
Source: gate latency audit (2026-11-14, item P1 "Reuse native Tint builds across
feature combinations").

## Problem

`yawgpu-tint/build.rs` builds the Tint shim with `cmake::Config`, whose build tree
is the build script's `OUT_DIR`. Cargo gives the build script a distinct `OUT_DIR`
for every distinct `yawgpu-tint` feature set (`tiled` on or off) and also for
`cargo check`/`clippy` versus `cargo build`/`test`. The native shim does not depend
on any Cargo feature (`build.rs` and `shim/CMakeLists.txt` read no
`CARGO_FEATURE_*`), so each of these units repeats the same C++ build.

Measured on the reference M2 (`target/debug`, 2026-11-14): four
`yawgpu-tint-*/out` trees of about 1.0 GB each, one per
{no features, `tiled`} × {check, build}. Each costs about 100 s of C++ compile
after a clean build, a shim change, or a Dawn bump.

## Rules

R1. **Shared native build directory.** The CMake build tree lives at
`<profile build dir>/yawgpu-tint-native-<key>`, where `<profile build dir>` is
`OUT_DIR`'s grandparent (`<target-dir>/<profile>/build`), and is passed with
`cmake::Config::out_dir`. Everything that reads the built shim
(`rustc-link-search`, `-rpath`, the Windows per-config subdirectories,
`copy_runtime_shim`) uses this directory instead of `OUT_DIR`. `OUT_DIR` keeps
holding only Cargo's own build-script output.

R2. **Key = native configuration only.** `<key>` is a 16-hex-digit FNV-1a 64 hash
of a canonical, ordered list of `name=value` entries describing everything that
selects a *different* native build:
- `TARGET`, `HOST`, `PROFILE`, `OPT_LEVEL`, `DEBUG`;
- the Dawn directory actually used (`YAWGPU_DAWN_DIR` override or the vendored
  path);
- the CMake defines this script sets: Apple deployment target (the resolved value),
  Android toolchain file, `ANDROID_ABI`, `ANDROID_PLATFORM`;
- compiler/CMake environment the `cmake`/`cc` crates read: `CC`, `CXX`, `CFLAGS`,
  `CXXFLAGS`, `CMAKE_GENERATOR`, `CMAKE_TOOLCHAIN_FILE`, and their `_<target>` /
  `_<target_with_underscores>` / `TARGET_` variants as the `cc` crate defines them;
  an unset variable contributes `name=` so set-versus-unset differs.

It does **not** include Cargo features or the check/build mode, nor source contents:
shim and Dawn source changes are rebuilt incrementally by CMake inside the same
tree (the existing `rerun-if-changed` lines stay as they are, so the build script
still re-runs and CMake still runs its incremental build). The hash and the
key-list construction are pure functions in `yawgpu-tint/build_probe.rs`, so the
existing `tests/build_probe.rs` target tests them.

R3. **Exclusive access.** Before configuring, the script opens
`<profile build dir>/yawgpu-tint-native-<key>.lock` and holds an exclusive
`std::fs::File::lock` until the CMake build and the runtime copies finish. Two
build-script runs in one Cargo invocation (for example a check unit and a build unit)
therefore cannot build the same tree at once. A lock failure is a build error with
the path in the message; there is no silent fallback to `OUT_DIR`.

R4. **Unchanged behaviour.** The stub decision, warnings, `have_tint` cfg,
`rerun-if-*` set, Android/Apple configuration, link directives and runtime copy
targets are unchanged apart from the directory they point at. Windows multi-config
subdirectories are still searched.

## Slices

S1: R1–R4 in `yawgpu-tint/build.rs` and `yawgpu-tint/build_probe.rs`, with tests in
`build_probe.rs`'s inline test module (run through the `tests/build_probe.rs` target):
- the hash is stable (a fixed input list gives a fixed, literal expected key);
- changing any single entry changes the key; reordering is impossible (the list
  builder emits a fixed order) and the builder's output is asserted literally for
  one input;
- unset versus empty-string environment values produce different keys;
- the shared directory path is derived from an `OUT_DIR` of the documented shape,
  and a too-shallow `OUT_DIR` is an error, not a panic.

S2 (Phase Review MINORs m1, m2, m4):
- m1: `cargo clean -p yawgpu-tint` does not remove `yawgpu-tint-native-*` (Cargo's
  `clean -p` matches only `<package>-<hash>` names). The stub warning in `build.rs`
  also names the native trees and how to remove them
  (`rm -rf target/*/build/yawgpu-tint-native-*`).
- m2: key entries encode set values explicitly (`name=<byte length>:<bytes>` from
  `OsStr::as_encoded_bytes`) instead of `{:?}`, so a toolchain change to `Debug`
  escaping cannot change the key. Unset stays `name=`.
- m4: add `SDKROOT`, `CROSS_COMPILE` and `CMAKE` (with the same target variants
  as the other variables) to the key list; add a test with a target triple that
  contains `.` (for example `thumbv8m.main-none-eabi`) asserting the cc versus
  cmake underscore difference.

Old trees are not deleted automatically: every change to a key input leaves the
previous tree (about 1 GB for debug) in place until `cargo clean` or a manual `rm`.

## Verification

- Agent gates (redirect output to a file; report exit codes):
  `cargo test -p yawgpu-tint` (lib + `build_probe`),
  `cargo clippy -p yawgpu-tint --all-targets -- -D warnings`,
  `cargo clippy -p yawgpu-tint --all-targets --features tiled -- -D warnings`.
- Claude, on review: from a state where the new directory does not yet exist,
  `cargo test -p yawgpu --test instance_smoke --no-run` builds Tint once; then
  `--features tiled` and a `cargo clippy` run do **not** recompile Tint (wall time,
  and exactly one `yawgpu-tint-native-*` tree); touching `shim/tint_shim.cpp`
  rebuilds incrementally in that tree; the workspace tests pass with real Tint
  (`have_tint`), and a real-Metal e2e target that compiles WGSL passes.

## Verification record

S1 (2026-11-14, M2): `cargo test -p yawgpu-tint` 129 + 18 build-probe tests pass;
clippy `-D warnings` default and `tiled`. One debug tree
(`yawgpu-tint-native-1cd7449f78c3b3cd`) served `cargo test`, default clippy, `tiled`
clippy, `yawgpu --features tiled` tests (6.9 s, no Tint rebuild) and
`clippy --features metal,tiled` (5.2 s). Touching `shim/tint_shim.cpp` rebuilt
incrementally in that tree (7.4 s). `cargo test --workspace` 1492/0 in debug and
release (one release tree); workspace clippy clean; real-Metal `e2e_metal_compute`,
`e2e_metal_render`, `e2e_metal_shader_parity` pass (real Tint, no stub).

S2 (2026-11-14, M2): `cargo test -p yawgpu-tint` 129 + 21 build-probe tests pass;
clippy default and `tiled` clean. The new key format produced one new tree, which the
sandboxed agent build, `clippy --features metal,tiled` (3.2 s, no Tint rebuild) and an
unsandboxed real-Metal run all reused. `cargo test --workspace` 1495/0; workspace
clippy clean; `e2e_metal_compute` and `e2e_metal_tiled` pass on real Metal.

## Phase Review (2026-11-14, fresh-context reviewer over the S1 diff)

0 CRITICAL, 0 MAJOR, 4 MINOR. The key's variable variants match cc 1.2.65 and
cmake 0.1.58; the lock has no early-return path; reuse under clippy is not
defeated by wrapper variables. MINORs: m1 `cargo clean -p` leaves the native tree;
m2 key depends on `OsStr` `Debug` formatting; m3 old trees accumulate (documented
above, no code change); m4 `SDKROOT`/`CROSS_COMPILE`/`CMAKE` missing from the key.
m1, m2 and m4 are S2.
