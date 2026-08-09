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
