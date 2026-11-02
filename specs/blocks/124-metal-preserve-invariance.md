# Block 124 — Metal: `preserveInvariance` only for `@invariant`, as Dawn

Status: **COMPLETE (2026-11-02)**. Phase Review: 0 CRITICAL, 0 MAJOR, 3 MINOR; all
resolved, see below.
Source: backlog A16, Block 123 Phase Review m4.
Oracle: Dawn at the pinned `third_party/dawn`: `native/metal/ShaderModuleMTL.mm`
(`hasInvariantAttribute`, from Tint's `has_invariant_attribute`, sets
`preserveInvariance`), `native/metal/RenderPipelineMTL.mm` (descriptor fields), and
Tint `lang/msl/writer/common/output.h`.

## Problem

`create_render_library` (`yawgpu-hal/src/metal/pipeline.rs`) compiles every render
library with `preserveInvariance = true`. Dawn sets it only when Tint reports
`has_invariant_attribute` for that MSL. Metal documents the option as compiling
vertex shaders conservatively so that position calculations are consistent. With it
always on, yawgpu gives up optimizations Dawn keeps for every vertex shader that
declares no `@invariant` output. Results and speed can differ from Dawn.

The yawgpu Tint shim does not return `has_invariant_attribute`, so the HAL cannot
make Dawn's decision today.

A survey of the Metal pipeline descriptor against `RenderPipelineMTL.mm` found one
more field Dawn sets and yawgpu does not: `inputPrimitiveTopology`, set from the
primitive topology class. WebGPU has no layered rendering, which is where Metal needs
it, so no result changes today. It is cheap to match. Dawn's other unset field,
`supportIndirectCommandBuffers`, depends on `MultiDrawIndirect`, which yawgpu does not
offer, so leaving it unset already matches Dawn. Compute pipelines set nothing Dawn
does not.

## Rules

R1. The Tint shim returns `has_invariant_attribute` for every MSL generation. Core
threads it, per generated MSL source, into the Tint MSL variants of `HalShaderSource`:
`MslWithBufferSizes`, and `MslStagesWithBufferSizes` per stage.

R2. A render library compiled from Tint MSL sets `preserveInvariance` only when that
library's MSL has the attribute. A library holding both stages sets it when either
stage has it. A per-stage library uses its own stage's flag. The decision is a pure
function, unit-tested.

R3. Shader-passthrough MSL keeps `preserveInvariance = true`. The application's own
`[[invariant]]` outputs must still be honoured, and Dawn has no passthrough to match.

R4. Compute compiles are unchanged: they set no compile options, and `@invariant` does
not apply to compute.

R5. Every Metal render pipeline sets `inputPrimitiveTopology` from its WebGPU
topology: point-list gives `Point`, line-list and line-strip give `Line`, and
triangle-list and triangle-strip give `Triangle`. This matches Dawn's
`MTLInputPrimitiveTopology(GetPrimitiveTopology())`. Tiled subpass pipelines follow
the same rule.

R6. The flag is part of codegen output, so the Block 95 codegen caches store it with
the MSL. No key changes.

## Tests

- Shim: a vertex entry with an `@invariant @builtin(position)` output reports the
  attribute, and one without it does not.
- Core: the flag reaches `HalShaderSource` for compute and for both render stages.
  A Block 95 cache hit returns the same flag.
- HAL unit: the `preserveInvariance` decision for combined and per-stage libraries,
  and for passthrough. The topology-class mapping for every topology.
- Metal HAL `#[ignore]`: render pipelines with and without `@invariant` compile and
  draw.
- Real GPU e2e (Claude): two pipelines that draw the same triangle, both with
  `@invariant` positions and different vertex and fragment code around them. The
  second pass uses depth compare `equal`, and every covered pixel passes. This is a
  smoke test: the unit tests pin the flag itself.
- CTS on Metal, `webgpu:*`: `fail=2 crash=0`. Bench `pipeline/render_unique`.

## Slices

- **S1 (coding agent):** R1-R6 with unit tests and the HAL ignored test.
- **S2 (Claude):** e2e, CTS, bench, Phase Review.

## Verification (2026-11-02)

Run on Apple M2, macOS 27.2.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,311 passed, 0 failed |
| fmt, clippy on default, `metal`, `metal,tiled`, `metal,shader-passthrough`, `vulkan`, `vulkan,tiled`, `gles` | clean |
| Metal HAL ignored tests | 63 passed, including the invariant render test |
| Metal e2e, `metal,tiled,shader-passthrough` | 152 passed, 0 failed |
| webgpu-native-cts, Metal, `webgpu:*` | `fail=2 crash=0`; subcase counts unchanged |
| Bench `pipeline/render_unique` | 137 µs to 140 µs (Dawn 147 µs), within noise |

Tint sets `has_invariant_attribute` only when it emits the attribute for that entry
point. A fragment entry whose input struct carries `@invariant @builtin(position)`
reports false, because the MSL printer does not emit the attribute on fragment
inputs. Dawn reads the same field, so the outcome matches.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| m1 | The e2e passes with or without the flag | Documented as a smoke test; unit tests pin the flag |
| m2 | No fragment-input case | Shim and core tests assert the flag equals the presence of `[[invariant]]` in that stage's MSL (false today) |
| m3 | The combined render arm has no production caller | Comment added; R2's contract kept |

