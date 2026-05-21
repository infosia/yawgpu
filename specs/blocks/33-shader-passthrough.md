# Block 33 — Shader passthrough (SPIR-V / MSL)

Phase 13. **Vendor extension** (not a Dawn port): create `WGPUShaderModule`
objects from raw SPIR-V or raw MSL, bypassing the WGSL→naga translation that
blocks 30/40 rely on. Rules here are exercised by **direct unit tests**
(CLAUDE.md principle 1) plus real-backend `#[ignore]` e2e — not Dawn tests.
Status legend: ☐ todo ◐ partial ☑ done.

This block is gated by the cargo feature **`shader-passthrough`** (default
**off**; see "Feature gating"). It is purely additive: no existing public
signature or struct changes.

## Motivation

Engines that ship precompiled native shaders (a `.spv` blob for Vulkan, an
MSL string for Metal) want to hand those bytes to yawgpu directly rather than
authoring WGSL. yawgpu's pipeline path translates *lazily at pipeline-creation
time* from a validated naga module (`generate_spirv` for Vulkan,
`generate_msl` for Metal — see `yawgpu-core/src/{compute,render}_pipeline.rs`).
Passthrough means: keep the caller's bytes intact and feed them to the
**matching backend**, using reflection only to recover the metadata pipeline
creation needs (entry point, stage, workgroup size, resource bindings).

## Surface (yawgpu.h — new consolidated vendor header)

A0 introduces `yawgpu.h` as the single home for all `YaWGPU*` vendor
declarations. It absorbs the existing `YaWGPUInstanceBackendSelect` /
`YAWGPU_STYPE_INSTANCE_BACKEND_SELECT` (today hand-written in
`yawgpu/src/lib.rs` and re-declared in `examples/framework/framework.h`).
Examples switch to `#include "webgpu.h"` + `#include "yawgpu.h"`. Every vendor
descriptor ships a `YAWGPU_*_INIT` zero/sentinel initializer macro
(matching `webgpu.h` ergonomics).

**Naming convention** (documented at the top of `yawgpu.h`): the yawgpu-flavored
analog of webgpu.h's `wgpu*`/`WGPU*` asymmetry —
- functions: `yawgpu*` (e.g. `yawgpuDeviceCreateShaderModuleSpirV`)
- types / structs / enums / handles: `YaWGPU*` (e.g. `YaWGPUTransientAttachment`)
- constants / macros / SType: `YAWGPU_*` / `YAWGPU_STYPE_*`
- feature names: `YaWGPUFeatureName_*` (values of standard `WGPUFeatureName`)

Standard webgpu.h types (`WGPUDevice`, `WGPUTextureFormat`, `WGPUShaderStage`, …)
keep their `WGPU*` names; yawgpu functions take/return them directly.

SPIR-V (Vulkan-only; reflection automatic via naga `spv-in`):

```c
typedef struct YaWGPUShaderModuleSpirVDescriptor {
    WGPUChainedStruct const* nextInChain;
    WGPUStringView           label;
    uint32_t                 codeSize;   /* number of u32 words */
    uint32_t const*          code;       /* SPIR-V words */
} YaWGPUShaderModuleSpirVDescriptor;

WGPUShaderModule yawgpuDeviceCreateShaderModuleSpirV(
    WGPUDevice device,
    YaWGPUShaderModuleSpirVDescriptor const* descriptor);
```

MSL (Metal-only; caller supplies reflection; explicit pipeline layout required):

```c
typedef struct YaWGPUMslEntryPoint {
    WGPUStringView  name;
    WGPUShaderStage stage;            /* standard webgpu.h bitflag; exactly one of
                                         Vertex(1) / Fragment(2) / Compute(4) */
    uint32_t        workgroupSize[3]; /* compute only; ignored otherwise */
} YaWGPUMslEntryPoint;

typedef struct YaWGPUShaderModuleMslDescriptor {
    WGPUChainedStruct const*       nextInChain;
    WGPUStringView                 label;
    WGPUStringView                 code;            /* MSL source */
    size_t                         entryPointCount;
    YaWGPUMslEntryPoint const* entryPoints;
} YaWGPUShaderModuleMslDescriptor;

WGPUShaderModule yawgpuDeviceCreateShaderModuleMsl(
    WGPUDevice device,
    YaWGPUShaderModuleMslDescriptor const* descriptor);
```

The standard webgpu.h `WGPUShaderSourceSPIRV` chained struct (already parsed by
`map_shader_module_descriptor`, currently discarded) is re-routed to the same
SPIR-V core path — no new C declaration, no breaking change.

The produced handles are ordinary `WGPUShaderModule`s: `is_error`,
`diagnostic`, `wgpuShaderModuleGetCompilationInfo`, `AddRef`/`Release` all
behave as in block 30.

## Design decisions

- **Reflection split.**
  - **SPIR-V** is reflected by enabling naga's `spv-in` frontend
    (`yawgpu-core/Cargo.toml`, gated by `shader-passthrough`). The words are
    parsed into a `naga::Module` + `ModuleInfo` purely to drive reflection;
    the **original words are passed verbatim** to the Vulkan backend and are
    never re-emitted via `spv-out`.
  - **MSL** cannot be reflected (no MSL frontend exists in naga). The caller
    supplies entry-point name/stage/workgroup-size metadata; the source is
    passed verbatim to the Metal backend.
- **Reflection generalization.** `yawgpu-core/src/shader_naga.rs`'s
  `ValidatedWgslModule` becomes a source-agnostic `ReflectedModule` (naga
  `Module` + `ModuleInfo`) that both the WGSL frontend and the SPIR-V frontend
  produce, so existing accessors (`entry_points`, `compute_workgroup_size`,
  `resource_bindings`, …) work unchanged for SPIR-V.
- **Backend-specific modules.** A SPIR-V module is usable only on Vulkan, an
  MSL module only on Metal. Mismatch is a **pipeline-creation** error routed to
  the device error sink (the module create itself still succeeds, mirroring the
  reference: backend is only known at pipeline time).
- **MSL requires an explicit pipeline layout** (`layout != auto`). Binding
  reflection is unrecoverable from MSL, so shader-derived binding validation is
  skipped for MSL modules. SPIR-V keeps full `auto`-layout + binding validation.
- **MSL Metal binding-index mapping is a contract, documented exactly.** The
  caller's MSL must bake in metal `[[buffer(n)]]`/`[[texture(n)]]`/`[[sampler(n)]]`
  indices matching yawgpu's deterministic group/binding→index assignment. That
  assignment is derived from the explicit pipeline layout (the same algorithm
  the WGSL→MSL path uses in `compute_pipeline.rs`/`render_pipeline.rs` to build
  `MetalBufferBinding`). A3 ships, in `yawgpu.h`, the **exact mapping algorithm**
  plus a small worked MSL example (buffer / texture / sampler / storage) so
  authors can compute the indices at author time.
  - *Declined alternatives (recorded):* a binding-map field in the descriptor
    was declined per the prior scope decision (explicit layout, no binding map);
    a runtime `yawgpuGetMetalBindingIndex` helper was declined because the
    index depends on the full pipeline layout and authors need the value at
    author time, not runtime — the documented algorithm is the correct fix.
- `WGPUShaderStage` reuse: `YaWGPUMslEntryPoint.stage` is the standard
  `webgpu.h` bitflag; exactly one of Vertex/Fragment/Compute must be set.
- **MSL subpass-input *globals* unsupported** — irrelevant here, noted for
  cross-reference with block 55 (`@color(N)` entry-point form only).
- **Error-object model** identical to block 30: an invalid create (bad SPIR-V
  magic / parse failure / missing entry-point metadata) emits a device
  validation error and returns a `Release`-safe error `WGPUShaderModule` whose
  `is_error()` is true.
- **No panics** in core/HAL; FFI boundary may `expect` only on null handle
  where the spec forbids null.

## Core data model (`yawgpu-core/src/shader.rs`)

`ShaderModuleSourceKind` (today: `Wgsl{..}`, dead `Spirv{_words}`, `Invalid`)
becomes:

- `Wgsl { source, reflected: Box<ReflectedModule> }`
- `Spirv { words: Vec<u32>, reflected: Box<ReflectedModule> }`  *(real)*
- `Msl { source: String, reflection: MslReflection }`           *(new)*
- `Invalid`

where `MslReflection { entry_points: Vec<MslEntryPoint> }`,
`MslEntryPoint { name, stage, workgroup_size: [u32;3] }`.

New core APIs (each with an inline unit test):
- `Device::create_shader_module_spirv(words: &[u32]) -> ShaderModule`
- `Device::create_shader_module_msl(source: String, reflection: MslReflection) -> ShaderModule`
- accessors: `spirv_passthrough() -> Option<(&[u32], &ReflectedModule)>`,
  `msl_passthrough() -> Option<&MslReflection + &str>`.

Pipeline creation (`create_hal_compute_pipeline` / render equivalent)
generalizes the backend branch:
- Vulkan: WGSL→`generate_spirv`; SPIR-V passthrough→words verbatim + reflected
  entry/bindings; MSL passthrough→error.
- Metal: WGSL→`generate_msl`; MSL passthrough→source verbatim + supplied
  metadata; SPIR-V passthrough→error.

## Rules

### SPIR-V passthrough (P13.1)

- **SP1** `yawgpuDeviceCreateShaderModuleSpirV` ingests words and reflects
  them via naga `spv-in` into a `ReflectedModule`. Bad magic / parse failure /
  empty code ⇒ device error + error module. ☐ (UT)
- **SP2** the standard `WGPUShaderSourceSPIRV` chain (via
  `wgpuDeviceCreateShaderModule`) routes to the **same** core path (fixes the
  current dead-end). ☐ (UT)
- **SP3** at Vulkan pipeline creation the original words are passed **verbatim**
  (not re-emitted); entry point / workgroup / bindings come from reflection;
  `auto` layout is supported. ☐ (UT noop + e2e Vulkan)
- **SP4** a SPIR-V module used to create a pipeline on the **Metal** backend ⇒
  device error ("SPIR-V module cannot be used on the Metal backend"). ☐ (UT)
- **SP5** on Noop the create succeeds (reflection is backend-independent) and
  yields a valid module handle; no pipeline is built. ☐ (UT)

### MSL passthrough (P13.2)

- **MP1** `yawgpuDeviceCreateShaderModuleMsl` stores source + caller
  entry-point metadata; no naga involvement. Missing metadata for the stage a
  pipeline needs ⇒ error at pipeline creation. ☐ (UT)
- **MP2** an MSL module used on the **Vulkan** backend ⇒ device error. ☐ (UT)
- **MP3** an MSL module created against an **`auto`** pipeline layout ⇒ error;
  an explicit `WGPUPipelineLayout` is required. ☐ (UT)
- **MP4** at Metal pipeline creation the source is passed **verbatim**; entry
  point + workgroup size come from the supplied metadata. ☐ (UT noop + e2e Metal)
- **MP5** on Noop the create succeeds (metadata-only) and yields a valid module
  handle; no pipeline is built. ☐ (UT)
- **MP6** each `YaWGPUMslEntryPoint.stage` must have exactly one
  `WGPUShaderStage` bit set (Vertex/Fragment/Compute); zero or multiple ⇒ error. ☐ (UT)

### Common / handle behaviour (P13.3)

- **CB1** both passthrough handles are ordinary `WGPUShaderModule`s:
  `is_error`/`diagnostic`/`GetCompilationInfo`/`AddRef`/`Release` behave as in
  block 30. ☐ (UT)
- **CB2** error create returns a `Release`-safe error handle; first-match-wins
  error semantics. ☐ (UT)

## Async

`wgpuShaderModuleGetCompilationInfo` reuses block 30's future/callback
machinery; for a valid passthrough module it returns an empty/Info
`WGPUCompilationInfo`, for an error module ≥1 Error message. No new async
surface.

## Feature gating

- Cargo feature **`shader-passthrough`** on `yawgpu` forwards to
  `yawgpu-core/shader-passthrough`, which enables naga `spv-in`. Default off.
- When **off**: `yawgpuDeviceCreateShaderModule{SpirV,Msl}` are not
  compiled (link error if called); the standard `WGPUShaderSourceSPIRV` chain
  yields an error module ("SPIR-V passthrough not enabled"). `yawgpu.h` still
  declares the symbols and exposes a `YAWGPU_HAS_SHADER_PASSTHROUGH` macro
  (defined by the build / examples CMake) for `#ifdef` guards.
- Orthogonal to `metal`/`vulkan`: core ingestion + reflection compile without a
  backend; the real passthrough path only engages when the matching backend
  feature is also on.
- **Gates run in both configs**: `cargo test`/`clippy -D warnings`/
  `missing_docs` must pass with the feature off (default) **and** on. Each
  feature-gated `pub fn` carries its unit test under the same `#[cfg]`.

## Slices (1 handoff + 1 commit each)

- **A0** `yawgpu.h` new header + feature wiring (absorb backend-select; examples
  include migration). Gate green.
- **A1** core: `spv-in`, `ReflectedModule`, real `Spirv`/new `Msl` variants,
  `create_shader_module_spirv/_msl` + unit tests (SP1/SP5/MP1/MP5/CB1/CB2).
- **A2** pipeline wiring + backend-mismatch errors (SP3 noop/SP4/MP2/MP3/MP4 noop).
- **A3** C FFI vendor entry points + descriptors (`WGPUShaderStage` reuse) +
  `YAWGPU_*_INIT` macros + conv + standard-SPIRV re-route (SP2) + the
  documented Metal binding-index mapping algorithm & worked MSL example in
  `yawgpu.h` + FFI unit tests (incl. MP6).
- **A4** real-backend e2e (`#[ignore]`): Vulkan SPIR-V compute+render (SP3),
  Metal MSL compute+render (MP4).
- **A5** **Phase Review**.

## Open questions

- Whether to also accept a SPIR-V *entry-point override* in the descriptor
  (multi-entry-point modules) — default: reflect all, select by name at
  pipeline creation (same as WGSL).
