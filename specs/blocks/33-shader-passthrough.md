# Block 33 — Shader passthrough (SPIR-V / MSL)

> **REVIVED 2026-06-28 (post-Tint, simplified).** Originally removed 2026-06-26
> in Tint migration Phase 0; reinstated with a **reflection-free** design.
> A vendor extension lives *outside* the WebGPU spec by definition, so once a
> caller opts into passthrough they have left the spec's guarantees behind —
> "breaking WebGPU semantics" is therefore **not a risk** to weigh here, and
> "matching the Dawn CTS oracle" does not apply (passthrough is never exercised
> by CTS). This freed the design from the original naga-`spv-in` reflection
> machinery: we now pass the caller's bytes **verbatim** and recover every piece
> of pipeline metadata from the **explicit pipeline layout** + caller-supplied
> entry metadata, never from shader reflection.

**Vendor extension** (not a Dawn port): create `WGPUShaderModule` objects from
raw SPIR-V (Vulkan) or raw MSL (Metal), bypassing the WGSL→Tint translation that
blocks 30/40 rely on. Rules are exercised by **direct unit tests** (CLAUDE.md
principle 1) plus real-backend `#[ignore]` e2e — not Dawn tests.
Status legend: ☐ todo ◐ partial ☑ done.

Gated by the cargo feature **`shader-passthrough`** (default **off**; see
"Feature gating"). Purely additive: no existing public signature or struct
changes.

## Motivation

Engines that ship precompiled native shaders (a `.spv` blob for Vulkan, an MSL
string for Metal) want to hand those bytes to yawgpu directly rather than
authoring WGSL. yawgpu's normal path translates lazily at pipeline-creation time
from a Tint-reflected module (`generate_spirv` for Vulkan, `generate_msl` for
Metal — `yawgpu-core/src/{compute,render}_pipeline.rs`). Passthrough means: keep
the caller's bytes intact and feed them to the **matching backend**, taking
binding slots from the explicit pipeline layout and (for Metal compute) the
workgroup size from caller metadata.

## Surface (`yawgpu.h`)

### SPIR-V — reuse the standard chained struct (no new C declaration)

The standard `webgpu.h` already declares `WGPUShaderSourceSPIRV`
(`WGPUSType_ShaderSourceSPIRV = 0x1`) and the instance feature
`WGPUInstanceFeatureName_ShaderSourceSPIRV`. Today
`map_shader_module_descriptor` does not route it; this block wires it to the
SPIR-V passthrough core path. **No new vendor C declaration.** Vulkan-only.

```c
// standard webgpu.h — chained onto WGPUShaderModuleDescriptor.nextInChain
typedef struct WGPUShaderSourceSPIRV {
    WGPUChainedStruct chain;     // sType = WGPUSType_ShaderSourceSPIRV
    uint32_t          codeSize;  // number of u32 words
    uint32_t const*   code;      // SPIR-V words
} WGPUShaderSourceSPIRV;
```

### MSL — new vendor chained struct

MSL has no standard source struct, so yawgpu adds one. Metal-only. Compute
entries must declare their workgroup size (no reflection to recover it from):

```c
#define YAWGPU_STYPE_SHADER_SOURCE_MSL ((WGPUSType)0x70000004u)

typedef struct YaWGPUMslEntryPoint {
    WGPUStringView  name;
    WGPUShaderStage stage;             // exactly one of Vertex(1)/Fragment(2)/Compute(4)
    uint32_t        workgroupSize[3];  // compute only; ignored for vertex/fragment
} YaWGPUMslEntryPoint;

typedef struct YaWGPUShaderSourceMSL {
    WGPUChainedStruct          chain;  // sType = YAWGPU_STYPE_SHADER_SOURCE_MSL
    WGPUStringView             code;   // MSL source
    size_t                     entryPointCount;
    YaWGPUMslEntryPoint const* entryPoints;
} YaWGPUShaderSourceMSL;
```

Each ships a `YAWGPU_*_INIT` initializer macro. The produced handles are
ordinary `WGPUShaderModule`s (`is_error` / `diagnostic` /
`wgpuShaderModuleGetCompilationInfo` / `AddRef` / `Release` behave as in
block 30). No new functions: both flow through `wgpuDeviceCreateShaderModule`.

## Design decisions

- **No reflection, either source.** Both SPIR-V words and MSL source are passed
  **verbatim** to the matching backend; they are never parsed or re-emitted.
- **Explicit pipeline layout required for both** (`layout != auto`). With no
  reflection there is nothing to derive an auto-layout from, so an `auto` layout
  against a passthrough module is a pipeline-creation error. Binding slots come
  from the explicit layout via the existing maps:
  - **Metal**: `metal_buffer_binding_map(&bind_group_layouts)` — per-kind
    counters (`[[buffer(i)]]`/`[[texture(i)]]`/`[[sampler(i)]]`) assigned in
    `(group, binding)` ascending order. The caller's MSL **must** bake in these
    exact indices. The algorithm is documented verbatim in `yawgpu.h` with a
    worked example.
  - **Vulkan**: descriptor `set = group`, `binding = binding`. The caller's
    SPIR-V must decorate resources to match the explicit layout.
- **Backend-specific modules.** A SPIR-V module is usable only on Vulkan, an MSL
  module only on Metal. Mismatch is a **pipeline-creation** error routed to the
  device error sink (module create itself succeeds — the backend is only known
  at pipeline time). On **Noop**, pipeline creation succeeds with no compiled
  shader (Noop never compiles), keeping the path Noop-testable.
- **Metal compute workgroup size from metadata.** `HalComputePipeline` carries
  `workgroup_size`; the WGSL path fills it from reflection
  (`resolve_compute_workgroup`). Passthrough MSL fills it from the
  `YaWGPUMslEntryPoint` whose `name` matches the pipeline's `compute.entryPoint`.
  Missing/zero workgroup size ⇒ error. SPIR-V needs no metadata (LocalSize is
  baked into the words and consumed by the driver).
- **No pipeline-overridable constants in passthrough v1.** `constants` requires
  reflection; a non-empty `constants` array against a passthrough module ⇒
  error. (Re-home later if needed.)
- **Error-object model** identical to block 30: an invalid create (empty SPIR-V
  / not a multiple-of-4 / bad magic word `0x07230203`; empty MSL; a compute
  entry with no/zero workgroup size; an entry whose `stage` is not exactly one
  bit) emits a device validation error and returns a `Release`-safe error
  `WGPUShaderModule` with `is_error() == true`.
- **No panics** in core/HAL (`Result` + `?`); the FFI boundary may `expect` only
  on a null handle where the spec forbids null.

## Core data model (`yawgpu-core/src/shader.rs`)

`ShaderModuleSource` (today `Wgsl(String)`, `Invalid(String)`) gains, under
`#[cfg(feature = "shader-passthrough")]`:

- `SpirvPassthrough(Vec<u32>)`
- `MslPassthrough { source: String, entries: Vec<MslEntryPoint> }`

`ShaderModuleSourceKind` (today `Wgsl { _source, reflected }`, `Invalid`) gains:

- `SpirvPassthrough { words: Vec<u32> }`            *(no reflected module)*
- `MslPassthrough { source: String, entries: Vec<MslEntryPoint> }`

where `MslEntryPoint { name: String, stage: ShaderStage, workgroup_size: [u32;3] }`.

New core constructors (each with an inline unit test):
- `ShaderModule::from_spirv(words: Vec<u32>) -> Result<ShaderModuleSourceKind, String>`
- `ShaderModule::from_msl(source, entries) -> Result<ShaderModuleSourceKind, String>`
- accessors: `spirv_passthrough() -> Option<&[u32]>`,
  `msl_passthrough() -> Option<(&str, &[MslEntryPoint])>`.

Pipeline creation generalizes the backend branch (compute first, render later):
- Metal: WGSL→`generate_msl`; **MSL passthrough**→`HalShaderSource::Msl(source)`
  verbatim + layout-derived `metal_bindings` + metadata workgroup size;
  SPIR-V passthrough→error.
- Vulkan: WGSL→`generate_spirv`; **SPIR-V passthrough**→`HalShaderSource::SpirV`
  verbatim + layout-derived descriptor bindings; MSL passthrough→error.

## Rules

### MSL passthrough (P13.1) — Metal — **B1 DONE**

- **MP1** `YaWGPUShaderSourceMSL` stores source + entry metadata; no Tint
  involvement. Empty `code` ⇒ error module. ☑ (UT)
- **MP2** an MSL module used on the **Vulkan** backend ⇒ device error
  ("MSL passthrough shader requires the Metal backend"). ☑ (UT)
- **MP3** an MSL module against an **`auto`** pipeline layout ⇒ error; an
  explicit `WGPUPipelineLayout` is required. ☑ (UT)
- **MP4** at Metal pipeline creation the source is passed **verbatim**; the
  compute entry point + workgroup size come from the matching
  `YaWGPUMslEntryPoint`. A compute entry with no/zero workgroup size ⇒ error.
  ☑ (UT noop + e2e Metal — `e2e_metal_shader_passthrough.rs`)
- **MP5** on **Noop** the create succeeds and a pipeline builds with no compiled
  shader. ☑ (UT)
- **MP6** each `YaWGPUMslEntryPoint.stage` must have exactly one `WGPUShaderStage`
  bit (Vertex/Fragment/Compute); zero or multiple ⇒ error module. ☑ (UT — zero
  and multi-bit)
- **MP7** with the feature **off**, the `YAWGPU_STYPE_SHADER_SOURCE_MSL` chain
  yields an error module ("shader passthrough not enabled"). ☑ (UT)

### SPIR-V passthrough (P13.2) — Vulkan

- **SP1** the standard `WGPUShaderSourceSPIRV` chain routes to the SPIR-V
  passthrough core path. Empty / non-word-multiple / bad magic ⇒ error module.
  ☐ (UT)
- **SP2** at Vulkan pipeline creation the words are passed **verbatim** (never
  re-emitted); descriptor bindings come from the explicit layout; `auto` layout
  ⇒ error. ☐ (UT noop + e2e Vulkan)
- **SP3** a SPIR-V module used on the **Metal** backend ⇒ device error. ☐ (UT)
- **SP4** on **Noop** the create succeeds and a pipeline builds with no compiled
  shader. ☐ (UT)
- **SP5** with the feature **off**, the `WGPUShaderSourceSPIRV` chain yields an
  error module ("shader passthrough not enabled"). ☐ (UT)

### Common / handle behaviour (P13.3)

- **CB1** both passthrough handles are ordinary `WGPUShaderModule`s
  (`is_error`/`diagnostic`/`GetCompilationInfo`/`AddRef`/`Release` per block 30).
  ☑ (UT — MSL; SPIR-V with B2)
- **CB2** error create returns a `Release`-safe error handle; first-match-wins
  error semantics. ☑ (UT — MSL; SPIR-V with B2)

## Feature gating

- Cargo feature **`shader-passthrough`** on `yawgpu` forwards to
  `yawgpu-core/shader-passthrough`. Default **off**. Pulls **no extra
  dependency** (no reflection frontend) — the gate is an opt-in escape hatch,
  not a dependency switch.
- When **off**: the new core variants are `#[cfg]`-compiled out; the MSL chain
  (`YAWGPU_STYPE_SHADER_SOURCE_MSL`) and the standard SPIR-V chain both yield an
  error module. `yawgpu.h` still declares the MSL struct/sType so callers can
  `#ifdef YAWGPU_HAS_SHADER_PASSTHROUGH`.
- Orthogonal to `metal`/`vulkan`: core ingestion compiles without a backend; the
  real passthrough path only engages when the matching backend feature is on.
- **Gates run in both configs**: `cargo test` / `clippy -D warnings` /
  `missing_docs` must pass with the feature **off** (default) **and on**. Each
  feature-gated `pub fn` carries its unit test under the same `#[cfg]`.

## Slices (1 handoff + 1 commit each)

- **B1** ✅ **DONE** **Metal MSL compute, end-to-end.** Cargo feature wiring;
  core `MslPassthrough` variants + `from_msl` + accessor + unit tests
  (MP1/MP5/MP6/CB1/CB2); compute-pipeline Metal branch + backend-mismatch (MP2) +
  auto-layout reject (MP3) + workgroup-size threading (MP4 noop); `yawgpu.h`
  `YaWGPUShaderSourceMSL` + conv routing (MP7) + FFI unit test; real-GPU
  `e2e_metal_shader_passthrough.rs` compute (MP4).
- **B2** **Vulkan SPIR-V compute, end-to-end.** Core `SpirvPassthrough` + magic
  validation + accessor + unit tests (SP1/SP4/SP5); compute-pipeline Vulkan
  branch + mismatch (SP3) + auto-layout reject; conv routing of standard
  `WGPUShaderSourceSPIRV` (SP1); real-GPU `e2e_vulkan_shader_passthrough.rs`
  reusing the existing `compute_spirv()` fixture (SP2).
- **B3** **Render pipelines** (vertex+fragment) for both backends:
  `HalShaderSource::MslStages` / per-stage `SpirV`; e2e triangle→texture readback
  on Metal and Vulkan.
- **B4** **Docs + edge validation**: the Metal binding-index algorithm + worked
  MSL example in `yawgpu.h`; README "Shaders" note (vendor, off-by-default,
  unsafe); reject-matrix unit tests; spec rules ☑.
- **B5** **Phase Review** (fresh no-context subagent over the cumulative diff;
  fix CRITICAL/MAJOR before COMPLETE).

## Open questions

- Multi-entry-point SPIR-V: select by name at pipeline creation (as WGSL) —
  default yes, no descriptor override in v1.
- Whether to later gate SPIR-V passthrough behind the standard
  `WGPUInstanceFeatureName_ShaderSourceSPIRV` instance feature (spec-sanctioned
  opt-in) in addition to the cargo feature — deferred; not in v1.
