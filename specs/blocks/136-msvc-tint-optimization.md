# Block 136 — Optimize the Tint shim on MSVC release builds

Status: **COMPLETE (2026-11-03)**. Phase Review: 0 CRITICAL, 0 MAJOR, 2 MINOR; both fixed in
the spec.
Source: the `VkPipelineCache` measurement for backlog E3 (2026-11-03), by user request.
Related: Block 118 R9 (`NDEBUG` for non-Debug shim builds).

## Problem

On Windows, the release build of `yawgpu-tint` compiles Tint without optimization.
The `cmake` crate replaces `CMAKE_<LANG>_FLAGS_RELEASE` with its own flags when the
generator is Visual Studio. For the release shim that is `-nologo -MD -Brepro -W0`, so
CMake's default `/O2 /Ob2 /DNDEBUG` is gone. Every generated Release `.vcxproj` (for
example `dawn/src/tint/tint_lang_core.vcxproj`) has an empty `<Optimization>`, and MSVC's
default is no optimization. Block 118 restored `NDEBUG` only.

Measured on Windows 11, NVIDIA RTX 5060 Ti, driver 617.14, Vulkan, bench median, against
Dawn built from source with its own CMake:

| Case | yawgpu | Dawn | yawgpu / Dawn |
|---|---:|---:|---:|
| `shader/create_unique` | 432 µs | 47 µs | 9.2x |
| `pipeline/compute_unique` | 1,279 µs | 453 µs | 2.8x |
| `pipeline/render_unique` | 1,399 µs | 740 µs | 1.9x |

On the M2 the same cases are 0.85x to 1.15x of Dawn (`specs/tracking/perf-dawn-baseline.md`
Run 9). Shader-module creation runs only Tint, so the gap is in the shim build, not in
the Vulkan driver. With NVIDIA's disk cache off (`__GL_SHADER_DISK_CACHE=0`) the ratios
are the same.

## Rules

R1. For the MSVC compiler ID only (`$<COMPILE_LANG_AND_ID:C|CXX,MSVC>`), the shim's CMake adds
`/O2 /Ob2` to every target in the `Release` configuration and `/O2 /Ob1` in
`RelWithDebInfo`, before `add_subdirectory(${YAWGPU_DAWN_DIR} ...)` so the Dawn and Tint
targets get them too. `MinSizeRel` gets `/O1 /Ob1`. These are CMake's own defaults for
those configurations. `Debug` is unchanged. clang-cl reports the compiler ID `Clang`
(with `CMAKE_CXX_SIMULATE_ID` `MSVC`) and is not covered: yawgpu does not build with it.
The `cmake` crate replaces only the flags of the configuration it selects (`Release`, or
`RelWithDebInfo` for a release profile with debug info, or `MinSizeRel` for opt-level `s`
/ `z`); the entries for the other configurations then duplicate CMake's own defaults,
which is harmless.

R2. Non-MSVC toolchains are unchanged: their optimization flags come from the `cmake`
crate as today.

R3. The comment above the `NDEBUG` define is extended to say that the same replacement
drops the optimization flags, and that R1 restores them.

## Tests

- No unit test can observe a compiler flag. Verification is by inspection and timing
  (Claude):
  - after a clean rebuild of the shim, the Release `tint_lang_core.vcxproj` has
    `<Optimization>MaxSpeed</Optimization>`;
  - `cargo test --workspace` and the Vulkan e2e pass;
  - the bench cases above are re-measured against Dawn on the same host.

## Out of scope

- `VkPipelineCache` (E3): Dawn creates one per pipeline key from its blob cache, which is
  empty in standalone runs; the unique-pipeline cases above never hit it. Recorded in the
  backlog after this block.

## Slices

- **S1 (coding agent):** R1-R3.
- **S2 (Claude):** clean rebuild, inspection, tests, bench, Phase Review.

## Verification (2026-11-03)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan, after
`cargo clean -p yawgpu-tint --release` and a release rebuild.

The Release configuration of every generated project now optimizes:
`dawn/src/tint/tint_lang_core.vcxproj` shows `<Optimization>MaxSpeed</Optimization>`. The
Phase Review checked `tint_shim`, abseil, and SPIRV-Tools too (Release `MaxSpeed` /
`AnySuitable`), and Debug still `Disabled`.

Bench, median, `--reps 15`, against Dawn built from source with its own CMake
(`C:/Users/kiguc/workspace/C/dawn`, commit `20734e3dea`):

| Case | yawgpu before | yawgpu after | Dawn | after / Dawn |
|---|---:|---:|---:|---:|
| `shader/create_unique` | 431.8 µs | 35.8 µs | 43.7 µs | 0.82x |
| `pipeline/compute_unique` | 1,279.2 µs | 306.7 µs | 494.7 µs | 0.62x |
| `pipeline/render_unique` | 1,399.2 µs | 312.9 µs | 792.8 µs | 0.39x |
| `shader/create_cached` | 127.9 ns | 132.9 ns | 1,216.5 ns | not like-for-like |
| `pipeline/compute_cached` | 84.2 ns | 85.2 ns | 654.6 ns | not like-for-like |

Shader-module creation is 12x faster and pipeline creation 4x faster. Dawn's default
`EnableSpirvValidation` runs `spirv-val` on every module, which yawgpu does not, so part
of the remaining difference is that, not a yawgpu win in the compiler.

| Gate | Result |
|---|---|
| `cargo test --workspace` (Debug configuration, unchanged) | 1,375 passed, 0 failed |
| `cargo fmt --all -- --check` | clean |
| webgpu-native-cts (CTS `a5cbf67`), `webgpu:*`, optimized shim | `fail=4 crash=3`, all known |

| Area | pass | fail | crash | Block 134 run pass |
|---|---:|---:|---:|---:|
| `api,validation` | 246,666 | 0 | 0 | 246,666 |
| `api,operation` | 180,779 | 0 | 0 | 180,779 |
| `shader,execution` | 562,531 | 2 | 3 | 562,548 |
| `shader,validation` | 662,535 | 2 | 0 | 662,535 |

- The `shader,validation` failures are the Block 119 R9 cases.
- The `shader,execution` failures are `memory_model,weak:message_passing` and
  `weak:store_buffer` on `atomic_storage`, the intermittent driver behaviour recorded in
  Block 120; `memory_model,weak:*` then passed 12/12 in each of 3 runs.
- The crashes are F-165 (`bufferArrayView` / `bufferView` workgroup f16 aliasing), made
  more frequent by SPIR-V 1.4 (Block 133).
- The full suite took 29 minutes, as before.

## `VkPipelineCache` (backlog E3)

Measured before this block was found: the unique-pipeline cases never hit a pipeline
cache, and Dawn's per-key `VkPipelineCache` starts empty in standalone runs (no blob cache;
`VulkanMonolithicPipelineCache` is off by default). After this block yawgpu creates
pipelines faster than Dawn on this host. A `VkPipelineCache` is therefore not pursued; the
E3 entry records this.
