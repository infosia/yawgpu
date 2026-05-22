# Phase 13 — Shader passthrough (SPIR-V / MSL)

Status: **COMPLETE** (A0-A4 done + real-GPU verified; Phase 13 Review CLOSED —
0C/2M/4m, all MAJOR + kept MINOR fixed, see A5). Commits `phase-13: A0` →
`phase-13: phase review`. Rules/plan: `../blocks/33-shader-passthrough.md`.
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

## A0 — yawgpu.h header + feature wiring  *(☑ DONE)*

Done: `yawgpu/ffi/webgpu-headers/yawgpu.h` created (naming-convention header +
renamed `YaWGPUInstanceBackendSelect` / `YAWGPU_STYPE_INSTANCE_BACKEND_SELECT` /
`YAWGPU_INSTANCE_BACKEND_*` + empty `YAWGPU_HAS_{SHADER_PASSTHROUGH,TILED}`
guard blocks). Rust vendor symbols renamed in `lib.rs`/`ffi/mod.rs` (no stray
old names). Cargo features added: `yawgpu` {`shader-passthrough`,`tiled`,
`mobile`} → `yawgpu-core` {`shader-passthrough`=`naga/spv-in`,`tiled`} +
`yawgpu-hal` {`tiled`}; all default off. `examples/framework/framework.h`
now `#include "yawgpu.h"` (private re-decl removed); `framework.c` uses new
names. CMake gained `YAWGPU_EXTENSIONS` (appends cargo features + defines
`YAWGPU_HAS_*`; default empty = unchanged; backend `target-${FEATURE}` dir
scheme intact).
*Gate (Claude-run):* default `cargo test --workspace` + `clippy -D warnings`
green (0 failed); same green with `--features shader-passthrough` / `tiled` /
`mobile` (no `dead_code` from the unused `spv-in`); metal examples build clean
(C17-strict). All A0 acceptance criteria met.

New `yawgpu.h` consolidating all `YaWGPU*` vendor declarations; absorb
`YaWGPUInstanceBackendSelect` + its SType from `yawgpu/src/lib.rs` and the
re-declaration in `examples/framework/framework.h`; examples switch to
`#include "yawgpu.h"`. Add cargo features `shader-passthrough` (→
`yawgpu-core/shader-passthrough` → naga `spv-in`), `tiled` (placeholder for
Phase 14), `mobile` umbrella; all default off. CMake forwards the feature and
defines `YAWGPU_HAS_SHADER_PASSTHROUGH`.
*Accept:* default + feature-on gates green; examples build unchanged
behaviourally; no symbol added yet beyond the moved backend-select.

## A1 — core data model + naga reflection  *(☑ DONE)*

Done: `ValidatedWgslModule` → `ReflectedModule` (rename propagated to
shader.rs/compute_pipeline.rs/render_pipeline.rs; `validated_wgsl()` →
`reflected_wgsl()`). `shader_naga::reflect_spirv` (naga `spv-in` Frontend →
shared `validate_module`), feature-gated. `ShaderModuleSourceKind` real
`Spirv{words,reflected}` + new `Msl{source,reflection}` (both gated);
`from_spirv` (magic-number + limit check), `from_msl` (non-empty + exactly-one
stage bit), `spirv_passthrough`/`msl_passthrough` accessors; `MslReflection`/
`MslEntryPoint`. `Device::create_shader_module_spirv/_msl`; the `Msl` source
variant + the on/off `Spirv` arms in `create_shader_module` (feature-off →
"SPIR-V passthrough not enabled" error module).
*Gate (Claude-run):* DEFAULT (feature off) compiles without `spv-in` —
`cargo test --workspace` + `clippy -D warnings` green (0 failed); with
`--features shader-passthrough` clippy clean + the 3 new tests pass. No
dead_code (accessors read the new fields). Rules SP1/SP5/MP1/MP5/MP6/CB1/CB2
each exercised. All A1 acceptance criteria met.

## A2 — pipeline wiring + backend-match  *(☑ DONE)*

Done: `ShaderModule::reflected()` (Wgsl + Spirv) drives resolution so SPIR-V
reflects like WGSL (`auto` layout, workgroup, bindings). MSL branch requires an
explicit layout (auto → error) and takes entry/workgroup from metadata. The
backend×source rule is a pure helper — `select_{compute,render}_shader_source`
— so the cross-backend matrix is unit-testable without a real device: Vulkan
WGSL→generate_spirv, Vulkan SPIR-V→words verbatim, Metal WGSL→generate_msl,
Metal MSL→source verbatim, mismatches → SP4/MP2 errors.
*Gate (Claude-run):* default + `--features shader-passthrough` `cargo test
--workspace`/`clippy -D warnings` green; WGSL pipeline tests unchanged. A2
tests: `select_{compute,render}_shader_source_covers_passthrough_backend_matrix`
(SP4/MP2/MP4 incl. verbatim `selected == words`/`== msl_source`),
`spirv_compute_pipeline_auto_layout_resolves_on_noop` (SP3),
`msl_compute_pipeline_requires_explicit_layout_on_noop` (MP3). All A2
acceptance met.

## A3 — C FFI + standard-SPIRV re-route  *(☑ DONE)*

Done: `yawgpu.h` `YAWGPU_HAS_SHADER_PASSTHROUGH` block filled —
`YaWGPUShaderModuleSpirVDescriptor`/`YaWGPUMslEntryPoint`(std `WGPUShaderStage`)/
`YaWGPUShaderModuleMslDescriptor` + `YAWGPU_*_INIT` macros (webgpu.h style) +
the documented Metal binding-index mapping algorithm & worked MSL example.
Matching Rust `#[repr(C)]` mirrors in `lib.rs` (ABI-equal). FFI
`yawgpuDeviceCreateShaderModule{SpirV,Msl}` in `ffi/shader.rs` mirror
`wgpuDeviceCreateShaderModule`; `map_msl_entry_point` + `MslReflection::new`/
`MslEntryPoint::new` constructors. Standard `WGPUShaderSourceSPIRV` already
reaches the core path (A1) — covered by an FFI test.
*Gate (Claude-run):* default + `--features shader-passthrough` `cargo test
--workspace`/`clippy -D warnings` green; metal examples build clean with
`-DYAWGPU_EXTENSIONS=shader-passthrough` (yawgpu.h block compiles, C17-strict).
FFI tests: `yawgpu_spirv_shader_module_ffi_accepts_valid_words_and_errors_on_bad_input`
(SP1), `standard_spirv_shader_source_chain_reaches_spirv_core_path` (SP2),
`yawgpu_msl_shader_module_ffi_accepts_metadata_and_rejects_bad_stage_bits`
(MP1/MP6).
*MINOR (deferred to A5 review):* the feature-off "SPIR-V passthrough not enabled"
degrade exists in `device.rs` but has no direct asserting test.

## A4 — real-backend e2e (`#[ignore]`)  *(☑ DONE)*

Done: **(0)** finished the A0 vendor-symbol rename leak in `yawgpu/tests/` — all
10 `e2e_{metal,vulkan}_*.rs` still used the old `WGPUYawgpu*` names and failed
to compile under `--features metal`/`vulkan` (A0's gate omitted the
backend-feature `--tests` build; that's now added to the standing gate). **(1)**
new `e2e_{metal,vulkan}_shader_passthrough.rs` (`#[ignore]`, cfg-gated on
backend + `shader-passthrough`); SPIR-V words generated at test time from WGSL
via a `naga` dev-dependency (`wgsl-in,spv-out`).
*Build:* `cargo build -p yawgpu --tests` with `metal` / `vulkan` /
`metal,shader-passthrough` / `vulkan,shader-passthrough` all compile; default
Noop `cargo test --workspace` + `clippy -D warnings` green.
*Real-GPU runs (Claude, this Apple Silicon):*
- Metal `--features metal,shader-passthrough -- --ignored`:
  `metal_msl_compute_fills_storage_buffer`, `metal_msl_render_draws_constant_color_triangle`
  → **2 passed** (MP4 verified).
- Vulkan (MoltenVK, `$VULKAN_SDK` 1.3.296.0 sourced)
  `--features vulkan,shader-passthrough -- --ignored`:
  `vulkan_spirv_compute_fills_storage_buffer`, `vulkan_spirv_render_draws_constant_color_triangle`
  → **2 passed** (SP3 verified).

## A5 — Phase Review  *(☑ DONE — CLOSED)*

Clean Review (fresh no-context subagent, diff `c2fc3e6..d6fe762` + block 33 +
CLAUDE.md + naming-conventions): **6 findings — 0 CRITICAL / 2 MAJOR / 4 MINOR.**

Triage + resolution:
- **MAJOR 1 (fixed)** — `select_render_shader_source` Vulkan arm took the
  verbatim path only when *both* stages were SPIR-V; a SPIR-V+WGSL mix fell
  through to `generate_spirv`, re-emitting the SPIR-V (SP3 violation). Fix:
  reject any mixed SPIR-V-passthrough + non-SPIR-V render pipeline
  (render_pipeline.rs:718) + matrix tests for both mix directions.
- **MAJOR 2 (fixed)** — render-path MP3 was enforced but untested. Added
  `msl_render_pipeline_requires_explicit_layout_on_noop`.
- **MINOR 1 (fixed)** — `pub use ReflectedModule` is now
  `#[cfg(feature = "shader-passthrough")]` (no naga-internals leak in the
  default build).
- **MINOR 2 (fixed)** — restored the WGSL `_source` field name; dropped the
  `let _ = source;` hack.
- **MINOR 3 (deferred, rationale)** — vendor SpirV/Msl FFI entry points don't use
  the shader-module cache. Functionally correct; passthrough modules are
  typically created once, so dedupe value is low. Left as-is.
- **MINOR 4 (fixed, spec)** — empty-MSL rejection now recorded as rule MP7 in
  block 33.

*Final gate (Claude-run):* default + `--features shader-passthrough`
`cargo test --workspace` / `clippy -D warnings` green; `cargo build -p yawgpu
--tests` with metal / vulkan / `*,shader-passthrough` all compile. No open
CRITICAL/MAJOR → **Phase 13 COMPLETE**.
