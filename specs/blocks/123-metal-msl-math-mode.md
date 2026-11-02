# Block 123 — Metal: relaxed MSL math mode, as Dawn

Status: **COMPLETE (2026-11-02)**. Phase Review: 0 CRITICAL, 0 MAJOR, 4 MINOR; m1 and m3
fixed, m2 recorded as R4a, m4 moved to backlog A16.
Source: backlog A15, found in Block 122 S2.
Oracle: Dawn at the pinned `third_party/dawn`: `native/metal/ShaderModuleMTL.mm`
(the MSL header prepended before compile, and the compile options), and Dawn commit
`58c1c685a8` ("MacOS use relaxed math where possible", crbug.com/425650181).

## Problem

Dawn prepends a header to the MSL it compiles. On macOS 15 / iOS 18 and later, that
header includes `#pragma METAL fp math_mode(relaxed)`, unless the module asks for
strict math. Relaxed math allows unsafe floating-point optimizations but keeps
infinities and NaNs. On older systems Dawn sets `fastMathEnabled = true` instead.

yawgpu prepends nothing, and passes no math option: compute compiles with no
`MTLCompileOptions`, and render with only `preserveInvariance`. Metal's default
behaves as fast math, which may assume no infinities or NaNs. Measured on Apple M2,
macOS 27.2, with `x = +inf` loaded at run time:

| Expression | yawgpu (default) | Dawn (relaxed) | IEEE |
|---|---|---|---|
| `x - x` | 0 | NaN | NaN |
| `x / x` | 1.0 | NaN | NaN |
| `x * 0.0` (literal) | 0 | NaN | NaN |
| `isnan(x - x)` | false | true | true |

WGSL lets an implementation assume that no infinity or NaN occurs during shader
execution, so neither output violates the spec, and the CTS does not see the
difference. It is still a divergence from the oracle that applications can observe.

## Rules

R1. Every MSL source produced by Tint is compiled with Dawn's header prepended. The
header is `"#ifdef __clang__\n#pragma clang diagnostic ignored \"-Wall\"\n#endif\n"`.
On macOS 15.0 / iOS 18.0 and later it is followed by
`"\n#pragma METAL fp math_mode(relaxed)\n"`. The text matches Dawn's
byte for byte.

R2. On older systems the math pragma is absent and the compile options keep Metal's
default, under which `fastMathEnabled` is true. That equals Dawn's
`fastMathEnabled = !strictMath` with strict math unset.

R3. R1 covers every pipeline whose MSL Tint generated: compute, render (single
library and per-stage libraries), tiled subpass pipelines, and internal pipelines
such as timestamp conversion and indirect-dispatch validation. MSL supplied through
`shader-passthrough` is compiled unchanged, because it is the application's own
source and Dawn has no MSL passthrough.

R4. The OS check runs once per device, as Block 120 does for its gate. The header is
built by a pure function of that result, unit-tested without a GPU.

R5. The Block 95 codegen keys do not change. The header is added in the HAL after
codegen, and the OS result is fixed for the process.

R4a. Only macOS and iOS are covered. Every other target gets `false`. That differs
from Dawn's `@available(macOS 15.0, iOS 18.0, *)`, which is true on tvOS and visionOS.
Mac Catalyst compiles as `target_os = "ios"` but may report the macOS version, so it
would compare macOS 15 against 18. yawgpu supports none of these targets. Revisit
this if one is added.

R6. yawgpu exposes no strict-math option, so R1 always uses `relaxed`. Dawn's
`strictMath` and its `safe` arm are out of scope until `webgpu.h` exposes one.

## Tests

- Unit (R1, R2, R4): the header function, with the OS gate on and off. The text equals
  Dawn's literal.
- Unit (R3): passthrough MSL is compiled without the header, and Tint MSL with it, at
  whatever seam the HAL exposes.
- Metal HAL `#[ignore]`: an MSL compute pipeline that computes `x - x`, `x / x`, and
  `x * 0.0` for `x = +inf` read from a buffer writes NaN bits. It must fail without
  the change.
- Real GPU e2e (Claude): the same three expressions through WGSL and the C ABI, read
  back as NaN.
- CTS on Metal, `webgpu:*`: `fail=2 crash=0`.

## Slices

- **S1 (coding agent):** R1-R5 with the unit tests and the HAL ignored test.
- **S2 (Claude):** e2e, red check, CTS, bench, Phase Review.

## Verification (2026-11-02)

Run on Apple M2, macOS 27.2.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,307 passed, 0 failed |
| fmt, clippy on default, `metal`, `metal,tiled`, `metal,shader-passthrough`, `vulkan`, `gles` | clean |
| Metal HAL ignored tests | 62 passed, including the NaN test |
| Metal e2e, `metal,tiled,shader-passthrough` | 151 passed, 0 failed |
| `e2e_metal_math_mode` red check | without the HAL change, `x - x` gives `0x00000000` and the test fails; with it, all three are NaN |
| webgpu-native-cts, Metal, `webgpu:*` | `fail=2 crash=0`; subcase counts unchanged |
| Bench | no case moved by more than 8%, except the noisy `frame/*` latency case |

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | OS thresholds inline and untested | `relaxed_math_min_os_version` with a unit test |
| m2 | tvOS, visionOS, and Mac Catalyst differ from Dawn's `*` arm | R4a; none is a supported target |
| m3 | The e2e did not skip below macOS 15 | Skips there, since R2 keeps fast math by design |
| m4 | `create_render_library` always sets `preserveInvariance`; Dawn sets it only for `@invariant` | Pre-existing; backlog A16 |

