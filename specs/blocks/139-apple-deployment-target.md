# Block 139 — The Tint shim uses the Rust target's Apple deployment target

Status: **COMPLETE (2026-11-04)**. Phase Review: 0 CRITICAL, 0 MAJOR, 3 MINOR; see below.
Source: found while re-verifying the iOS cross-build on 2026-11-04; fixed by user request.

## Problem

On Apple targets, `libyawgpu.dylib` and `libtint_shim.dylib` declare different minimum OS
versions. Measured with `vtool -show-build` and `otool -l` on the M2, with Xcode 27.2 and
SDK 27.2, and with neither `MACOSX_DEPLOYMENT_TARGET` nor `IPHONEOS_DEPLOYMENT_TARGET`
set:

| Target | `libyawgpu.dylib` (rustc default) | `libtint_shim.dylib` (CMake) |
|---|---|---|
| `aarch64-apple-darwin` | 11.0 | 27.2 |
| `aarch64-apple-ios` | 10.0 | 27.2 |
| `aarch64-apple-ios-sim` | 14.0 | 27.2 |

The shim's C and C++ flags carry `-mmacosx-version-min=27.2` and the iOS equivalent.
With no environment variable set, the shim's CMake build falls back to the host SDK
version, while rustc uses the target's own default. The library therefore claims to
support macOS 11 and iOS 10, but its shim is compiled for the newest OS. A binary
built on an up-to-date Mac may then fail to load, or call missing symbols, on any
older OS.

Setting the variable by hand fixes it. With `MACOSX_DEPLOYMENT_TARGET=11.0`,
`IPHONEOS_DEPLOYMENT_TARGET=10.0` (device), or `IPHONEOS_DEPLOYMENT_TARGET=14.0`
(simulator), Tint compiles with no error or availability diagnostic, and the shim
reports the same minimum as `libyawgpu.dylib`.

## Rules

R1. **One deployment target per build.** When `yawgpu-tint/build.rs` builds the shim for
an Apple target (`CARGO_CFG_TARGET_OS` is `macos` or `ios`), it builds it for the same
minimum OS version that rustc uses for the Rust code of that target:
- the version in the target's environment variable when it is set
  (`MACOSX_DEPLOYMENT_TARGET` for macOS, `IPHONEOS_DEPLOYMENT_TARGET` for iOS and the
  iOS simulator);
- otherwise rustc's default for the target, as `rustc --print deployment-target
  --target <target>` reports it, run with the `RUSTC` that Cargo gives the build script.

Both the CMake configuration (`CMAKE_OSX_DEPLOYMENT_TARGET`) and the compiler flags that
the `cmake` crate passes must carry that version. The result is checked on the built
binaries, not on the flags.

R2. **Failure.** If rustc's output cannot be read or parsed, the build script prints a
`cargo:warning` and keeps today's behaviour. It does not fail the build.

R3. **Rebuild triggers.** The build script reruns when either environment variable
changes (`cargo:rerun-if-env-changed`).

R4. **Unchanged.** Other targets: Android (Block 98, `ANDROID_PLATFORM`), Linux, and
Windows. A deployment target that the user sets is used as given.

## Tests

- Unit (`yawgpu-tint` build-script helper, a pure function as `android_abi_for_arch`
  is): parse rustc's `MACOSX_DEPLOYMENT_TARGET=11.0` and
  `IPHONEOS_DEPLOYMENT_TARGET=14.0` output; reject empty, malformed, and wrong-variable
  output; the environment variable wins over rustc's default; choose the variable name
  from the target OS.
- Claude, on the M2 with the variables unset: `vtool -show-build` / `otool -l` show the
  same minimum for `libyawgpu.dylib` and `libtint_shim.dylib` on `aarch64-apple-darwin`
  (11.0), `aarch64-apple-ios` (10.0), and `aarch64-apple-ios-sim` (14.0). With
  `MACOSX_DEPLOYMENT_TARGET=13.0` set, both report 13.0.
- Claude: the iOS cross-builds and clippy as on 2026-11-04; Metal e2e; CTS
  `webgpu:shader,execution,*` on Metal. The shim is now compiled for an older OS, which
  can change library availability paths but not Tint's output.

## Out of scope

- tvOS, watchOS, visionOS, and Mac Catalyst (`*-apple-ios-macabi`). yawgpu does not build
  for them. Mac Catalyst has `CARGO_CFG_TARGET_OS=ios` but a macOS deployment target, so
  R1's iOS rule would not fit it.
- A minimum OS above rustc's default, for example to match Dawn's own build. Users can
  set the variable.

## Slices

- **S1 (coding agent):** R1-R4 with the unit tests.
- **S2 (Claude):** binary checks, cross-builds, e2e, CTS, Phase Review.

## Verification (2026-11-04)

Run on Apple M2, macOS 27.2, Xcode 27.2, SDK 27.2, with neither deployment-target
variable set unless stated.

| Gate | Result |
|---|---|
| Minimum OS, `libyawgpu.dylib` / `libtint_shim.dylib` | `aarch64-apple-darwin` 11.0 / 11.0; `aarch64-apple-ios` 10.0 / 10.0; `aarch64-apple-ios-sim` 14.0 / 14.0 (the shim was 27.2 on all three) |
| `MACOSX_DEPLOYMENT_TARGET=13.0` | 13.0 / 13.0 |
| iOS device and simulator, `metal` and `metal,tiled,shader-passthrough` | build with 0 warnings; clippy `-D warnings` clean |
| `cargo test --workspace` | 1,403 passed, 0 failed (`build_probe` 12 of 12) |
| fmt; clippy `-D warnings` on default, `metal`, `vulkan`, `gles` (host) | clean |
| Metal e2e and every other ignored `yawgpu` test, `metal,tiled` | 162 passed, 0 failed |
| webgpu-native-cts, Metal, `webgpu:shader,execution,*` | pass 853,787, skip 29,466, fail 0, crash 0; identical to the Block 130 baseline |

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | A comment named the cmake crate's patch version, but `Cargo.toml` allows any `0.1` | Reworded without the version |
| m2 | An explicit malformed value makes cc or rustc fail | Same before this block; R4 uses the value as given. No change |
| m3 | Mac Catalyst has `CARGO_CFG_TARGET_OS=ios` but a macOS deployment target | Out of scope, recorded |

