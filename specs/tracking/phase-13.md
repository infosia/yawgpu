# Phase 13 — Shader passthrough (SPIR-V / MSL)

Status: **PLANNED**. Rules/plan: `../blocks/33-shader-passthrough.md`.
Roles/loop: `../reference/workflow.md`.

**Vendor extension**, gated by cargo feature **`shader-passthrough`** (default
off). Lets C callers create `WGPUShaderModule`s from raw SPIR-V (Vulkan-only,
reflected via naga `spv-in`) or raw MSL (Metal-only, caller-supplied entry-point
metadata, explicit pipeline layout required). Purely additive: no existing
public signature/struct changes. Also re-routes the already-dead-ended standard
`WGPUShaderSourceSPIRV` chain to the same path.

**Gate (permanent):** `cargo test --workspace` + `cargo clippy --workspace
--all-targets -- -D warnings` green on **Noop**, run in **both** the default
config and `--features shader-passthrough` (and with `metal`/`vulkan` for the
e2e slice). `missing_docs` must hold in both. Feature-gated `pub fn`s carry
their unit tests under the same `#[cfg]`. Real-GPU e2e (`#[ignore]`) is run
**by Claude directly** on this Apple Silicon (Metal native; Vulkan via MoltenVK
with `$VULKAN_SDK` sourced) per the `claude-runs-real-gpu-tests` memory.
**Phase ends with the mandatory Phase Review** (`phase-13-review.md`).

Methodology: per CLAUDE.md principle 1 — each new `pub fn` ships with a direct
inline unit test (Red→Green); real-backend e2e is a regression layer on top.

## A0 — yawgpu.h header + feature wiring  *(☐ TODO)*

New `yawgpu.h` consolidating all `YaWGPU*` vendor declarations; absorb
`YaWGPUInstanceBackendSelect` + its SType from `yawgpu/src/lib.rs` and the
re-declaration in `examples/framework/framework.h`; examples switch to
`#include "yawgpu.h"`. Add cargo features `shader-passthrough` (→
`yawgpu-core/shader-passthrough` → naga `spv-in`), `tiled` (placeholder for
Phase 14), `mobile` umbrella; all default off. CMake forwards the feature and
defines `YAWGPU_HAS_SHADER_PASSTHROUGH`.
*Accept:* default + feature-on gates green; examples build unchanged
behaviourally; no symbol added yet beyond the moved backend-select.

## A1 — core data model + naga reflection  *(☐ TODO)*

`yawgpu-core`: generalize `ValidatedWgslModule` → `ReflectedModule`; enable
`spv-in` under the feature; real `Spirv{words,reflected}` + new
`Msl{source,reflection}` in `ShaderModuleSourceKind`;
`Device::create_shader_module_spirv/_msl` + inline unit tests.
*Accept:* rules SP1, SP5, MP1, MP5, CB1, CB2 each exercised by a unit test;
gates green both configs.

## A2 — pipeline wiring + backend-match  *(☐ TODO)*

Generalize `create_hal_compute_pipeline` / render equivalent: SPIR-V passthrough
→ words verbatim to Vulkan + reflected metadata; MSL passthrough → source
verbatim to Metal + supplied metadata; cross-backend use → device error; MSL +
`auto` layout → error.
*Accept:* SP3 (noop assertions), SP4, MP2, MP3, MP4 (noop) unit-tested.

## A3 — C FFI + standard-SPIRV re-route  *(☐ TODO)*

`yawgpu`: `yawgpuDeviceCreateShaderModuleSpirV/Msl` + `yawgpu.h` types +
`conv`; re-route standard `WGPUShaderSourceSPIRV` to the SPIR-V core path
(SP2); FFI unit tests.
*Accept:* SP2 + the FFI happy/error paths unit-tested; gates green both configs.

## A4 — real-backend e2e (`#[ignore]`)  *(☐ TODO)*

Vulkan SPIR-V compute + render (SP3); Metal MSL compute + render (MP4). Run by
Claude; log results here.

## A5 — Phase Review  *(☐ TODO)*

Fresh no-context subagent reviews the cumulative diff; CRITICAL/MAJOR/MINOR;
fix in severity order; cannot be COMPLETE with any open CRITICAL/MAJOR. Recorded
in `phase-13-review.md`.
