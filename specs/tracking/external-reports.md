# External bug reports — ledger

Findings reported by external clients (via the repo-root `HANDOFF.md`
drop, which is gitignored) that are not CTS findings (`F-xxx`, ledger
`specs/tracking/cts-coverage.md`) and not perf findings (`P-xxx`,
ledger `specs/tracking/perf-dawn-baseline.md`). Each entry records the
report, the root cause, the fix, and how it was verified.

## 2026-08-09 — surface configure rejects the header's zero sentinels

**Reported by:** subscript-gpu (windowed triangle example), against a
`4e961e6`-era tree, `metal` release build.

**Symptom.** A surface configured from `WGPU_SURFACE_CONFIGURATION_INIT`
(only device/format/usage/size set) never produces a texture:
`wgpuSurfaceConfigure` silently dispatches "surface configuration
present mode is not supported" to the device error sink, `configured`
stays `None`, and every `wgpuSurfaceGetCurrentTexture` reports
`Error` with a null texture.

**Root cause.** `surface_configuration_error`
(`yawgpu/src/ffi/mod.rs`) validated `presentMode` / `alphaMode` by
membership in `SURFACE_PRESENT_MODES = [Fifo]` /
`SURFACE_ALPHA_MODES = [Opaque]`, but the pinned `webgpu.h` defines
both zero values as valid configure inputs:
`WGPUPresentMode_Undefined` (0) "defaults to Fifo", and
`WGPUCompositeAlphaMode_Auto` (0) is "an alias for the first element"
of the reported `alphaModes`. The `INIT` macro sets exactly these two
zeros, so the header's own defaults were unconfigurable. Downstream
`hal_present_mode` already mapped `Undefined → Fifo`; only the
validation ahead of it was wrong.

**Fix.** Resolve the sentinels before the membership check
(`resolved_present_mode`: `Undefined → Fifo`; `resolved_alpha_mode`:
`Auto → SURFACE_ALPHA_MODES[0]`, i.e. first-capability alias, not a
hardcoded constant), store and forward the *resolved* modes.
Membership checks, error messages, and `wgpuSurfaceGetCapabilities`
(concrete modes only, never sentinels) are unchanged; explicit
unsupported modes (`Immediate`, `Mailbox`, `FifoRelaxed`,
`Premultiplied`, `Unpremultiplied`, `Inherit`, out-of-range) still
fail with the existing messages. Contract recorded in
`specs/blocks/70-finalize.md` → Surface → "Sentinel resolution".

**Verification.**

- Inline unit tests: resolver behaviour + `surface_configuration_error`
  accepts the sentinel pair and still rejects each explicit
  unsupported mode with the exact existing message.
- Noop integration (`yawgpu/tests/surface_validation.rs`): INIT-default
  configure dispatches no error and reaches the configured-Noop
  `Lost` boundary; explicit `Immediate` / `Premultiplied` still error.
- Real-Metal e2e (`yawgpu/tests/e2e_metal_surface.rs`, new): the
  report's repro as a test — standalone `CAMetalLayer`, configure with
  both sentinels, `getCurrentTexture` returns `SuccessOptimal` with a
  non-null texture, `wgpuSurfacePresent` succeeds; companion test
  keeps explicit unsupported modes erroring on real Metal. Full
  real-Metal e2e suite: 105 passed / 0 failed.
- CTS: the harness never calls `wgpuSurface*` (verified by grep over
  `webgpu-native-cts/src`), so the changed functions are unreachable
  from CTS; the Run-4 Metal tree set (8 `api,operation` subtrees,
  `--workers 6`) was still re-run as a before/after failure-set diff:
  `pass=175738 skip=47 warn=0 fail=0 crash=0` — identical to the
  baseline, failure set empty before and after.

## 2026-08-10 — `libyawgpu.so` cannot resolve its sibling `libtint_shim.so`

**Reported by:** external consumer (HANDOFF drop), against `96b380c`,
x86_64 Linux, release build, glibc loader.

**Symptom.** Any consumer process that loads `libyawgpu.so` as a shared
object fails at startup: `error while loading shared libraries:
libtint_shim.so: cannot open shared object file`. Workaround required
`LD_LIBRARY_PATH` on every developer machine and CI job. yawgpu's own
test suite never showed it — test binaries link `tint_shim` directly,
so their own `DT_RUNPATH` resolves it; the failure needs the shim as a
**transitive** `NEEDED` of the cdylib.

**Root cause.** `DT_RUNPATH` is not transitive: it resolves only the
`NEEDED` entries of the object carrying it, so the consumer's own
`RUNPATH` never applies to `libyawgpu.so`'s `NEEDED libtint_shim.so`.
`yawgpu/build.rs` handled Apple (`@rpath` install name; the shim's
`@loader_path` name) and `yawgpu-tint`'s `copy_runtime_shim` handled
Windows (image-directory DLL copy) and already places
`libtint_shim.so` beside the cdylib on ELF too — but no ELF image
pointed at that directory: `readelf -d target/release/libyawgpu.so`
showed `NEEDED libtint_shim.so` and no `RUNPATH` line. The
`yawgpu-tint/build.rs` rpath cannot fix it: `cargo:rustc-*link-arg`
does not propagate from a dependency's build script to the dependent's
cdylib link.

**Fix.** `yawgpu/build.rs`: beside the Apple `install_name` branch, an
ELF branch (`CARGO_CFG_TARGET_FAMILY == "unix"` and vendor not
`apple`, so Linux + Android; never Windows/wasm — target detection,
not host) emits `cargo:rustc-cdylib-link-arg=-Wl,-rpath,$ORIGIN`
(`$ORIGIN` written literally; cargo passes the arg to rustc with no
shell in between). The `yawgpu-tint/build.rs` rpath and the
Apple/Windows paths are untouched. Contract recorded in
`specs/reference/dependencies.md` → "ELF consumers — `$ORIGIN` rpath
on the cdylib".

**Verification.**

- New Linux-only integration tests (`yawgpu/tests/linkage_elf.rs`):
  `cdylib_carries_origin_runpath` parses the built cdylib's ELF64
  `.dynamic` section directly (no `readelf` dependency) and asserts a
  `$ORIGIN` component in `DT_RUNPATH`/`DT_RPATH`; it reads
  `<profile>/deps/libyawgpu.so` first because the profile-root file is
  an uplifted hardlink that test-only builds do not refresh.
  `consumer_loads_cdylib_without_ld_library_path` copies the cdylib
  (+ shim when present, absent in Dawn-less stub builds) into a fresh
  `CARGO_TARGET_TMPDIR` dir and loads it from a fresh python3 child
  process with `LD_LIBRARY_PATH` removed — a fresh process because the
  test binary itself links the shim, so an in-process `dlopen` would
  falsely succeed.
- Red-first at `96b380c`: both tests failed before the build.rs change
  (`no $ORIGIN component in DT_RUNPATH or DT_RPATH (found [])`;
  child `OSError: libtint_shim.so: cannot open shared object file`),
  and passed after.
- `readelf -d` on the rebuilt cdylib now reports
  `(RUNPATH) Library runpath: [$ORIGIN]`.
- Full Noop gate on Linux: `cargo test --workspace` 987 passed /
  0 failed; `cargo clippy --workspace --all-targets -- -D warnings`
  clean.
- macOS: unchanged by inspection (the new branch is unreachable for
  `vendor == apple`); the report's criterion "re-confirm the
  `@loader_path` install name on Apple" is deferred to the next run on
  a macOS host, as this fix was produced on a Linux-only host.
