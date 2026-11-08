# Block 148 — Shader passthrough: Phase Review fixes and the `ShaderSourceSPIRV` instance feature

Status: **IN PROGRESS**.
Source: backlog E8, started by user request on 2026-11-08. Block 33 (shader passthrough)
never had its Phase Review (slice B5). This block runs it on the code at `531bd45`, fixes
what it found, and gates SPIR-V input behind the standard instance feature, the Block 33
open question.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/native/`) for
the instance feature: `Instance.cpp` (`kSupportedFeatures` lists `ShaderSourceSPIRV`;
`mInstanceFeatures` keeps the instance descriptor's `requiredFeatures` without
rejecting any) and `Device.cpp` (`ShaderSourceSPIRV` input fails with "SPIR-V is
disallowed." unless the instance has the feature). Passthrough itself has no oracle; it
is a vendor extension.

## B5 Phase Review findings (2026-11-08)

A fresh reviewer read the passthrough path at `531bd45`. No CRITICAL. Unit tests:
`yawgpu-core` 30 and `yawgpu` 6 passthrough tests with the feature on, 2 with it off,
all passing.

| ID | Finding | Rule |
|---|---|---|
| M1 | Render passthrough runs only `validate_render_presence`. Vertex state, primitive state, depth bias and depth-stencil aspects, color targets, multisample state, and the bind-groups-plus-vertex-buffers limit are skipped, so for example `multisample.count = 3` or a depth format as a color target reaches `vkCreateGraphicsPipelines` or Metal | R1 |
| M2 | `metal_buffer_binding_map` (`yawgpu-core/src/compute_pipeline.rs`) chooses per-stage (render) slot assignment when any layout entry is visible to the vertex or fragment stage, also for a compute pipeline. A compute-only entry then gets slot 0, and an entry visible to compute and fragment gets the fragment slot. Two compute bindings can share `[[buffer(0)]]`. The map also feeds Tint's MSL bindings, so WGSL compute pipelines on Metal are affected too, not only passthrough | R2 |
| m1 | A passthrough module on the wrong backend (MSL on Vulkan, SPIR-V on Metal, either on GLES) is reported as an internal error, not a validation error; on GLES the message is "requires a reflected shader module" | R3 |
| m2 | An MSL compute entry's workgroup size is only checked for being non-zero, not against the device limits | R4 |
| m3 | A passthrough layout with an external-texture entry on Vulkan builds descriptor bindings without the plane 1 and parameter expansion, so the shader sees plane 0 only and the bound plane 1 and parameters are dropped. The Metal slot ABI for external textures is documented | R5 |
| m4 | `YAWGPU_HAS_SHADER_PASSTHROUGH` is always defined, but `yawgpu.h` says it is gated | R7 |
| m5 | The header and README leave out the entry-point requirement, the rejected pipeline constants, the Vulkan `set`/`binding` mapping, the Vulkan negative-height viewport, the immediates ABI (Metal drops `setImmediates` for MSL passthrough), and the vertex-buffer slots for MSL. The Block 33 spec lists "not a multiple of 4" as an error though `codeSize` counts words | R6, R7 |
| m6 | Invalid UTF-8 MSL `code` is reported as empty; an invalid UTF-8 entry name silently becomes `""`; the `entryPoints` slice length is not checked for overflow | R8 |

## Rules

R1. **Render passthrough validates the descriptor.** A render passthrough pipeline runs
every render-pipeline validation that does not need shader reflection, with the same
messages and order as the WGSL path: vertex state (buffer count, stride, attribute
format, offset, duplicate locations, limits), primitive state, depth bias and
depth-stencil aspects, color targets (count, formats, blend, write mask, bytes per
sample), multisample state, and the bind-groups-plus-vertex-buffers limit. Checks that
need reflection (shader locations against the entry point, inter-stage interface,
fragment outputs against targets, `frag_depth`, `sample_mask`) are skipped. Split a
validator if it mixes both kinds; the WGSL path's behaviour does not change.

R2. **Compute slots are compute slots.** A compute pipeline's Metal slot map uses the
compute assignment (per-kind counters over every entry, in group and binding order)
regardless of the entries' visibility. A render pipeline keeps the per-stage
assignment. The caller says which, rather than the map inferring it from visibility.
This applies to WGSL and passthrough compute pipelines.

R3. **Backend mismatch is a validation error.** Creating a pipeline from a passthrough
module on a backend that cannot run it is a validation error, with the message
"MSL passthrough shader requires the Metal backend" or "SPIR-V passthrough shader
requires the Vulkan backend", for compute and render, sync and async. Noop keeps
accepting both.

R4. **Workgroup size within limits.** An MSL compute entry's workgroup size is checked
like a WGSL one: each dimension against `maxComputeWorkgroupSizeX/Y/Z` and the product
(computed without overflow) against `maxComputeInvocationsPerWorkgroup`. A failure is a
validation error at pipeline creation.

R5. **No external textures in Vulkan passthrough.** Creating a SPIR-V passthrough
pipeline whose layout has an external-texture entry is a validation error
("SPIR-V passthrough does not support external texture bindings"). MSL passthrough
keeps the documented Metal slot ABI.

R6. **No immediates in MSL passthrough.** Creating an MSL passthrough pipeline whose
layout has a non-zero `immediateSize` is a validation error, because Metal has no
binding for them. SPIR-V passthrough keeps its push-constant range (offset 0, the
layout's `immediateSize`, every stage) and it is documented.

R7. **Documentation matches the code.** `yawgpu.h` and the README passthrough section
state:
- `YAWGPU_HAS_SHADER_PASSTHROUGH` means the declarations exist; with the cargo feature
  off, creation returns an error module "shader passthrough not enabled";
- SPIR-V needs `WGPUInstanceFeatureName_ShaderSourceSPIRV` (R9);
- the entry point name is required and pipeline constants are rejected;
- Vulkan maps `group` to `set` and `binding` to `binding`, uses a negative-height
  viewport (WebGPU clip space), and has the push-constant range of R6;
- MSL vertex buffers use the slots the WGSL path uses (state the rule from the code);
- external textures (R5) and immediates (R6).
The Block 33 spec drops the "multiple of 4" error and records the B5 outcome.

R8. **Input decoding.** Invalid UTF-8 MSL `code` yields an error module "MSL shader
source must be valid UTF-8"; an invalid UTF-8 entry-point name yields an error module
instead of an empty name. The `entryPoints` slice is built only when
`entryPointCount * size_of::<YaWGPUMslEntryPoint>()` does not overflow `isize`;
otherwise an error module.

R9. **SPIR-V needs the instance feature.** With `shader-passthrough` on,
`wgpuHasInstanceFeature(WGPUInstanceFeatureName_ShaderSourceSPIRV)` returns true and
`wgpuGetInstanceFeatures` lists it; with the feature off, neither does. A
`WGPUShaderSourceSPIRV` module created on a device whose instance did not list the
feature in `requiredFeatures` is an error module with the message "SPIR-V is
disallowed." With the cargo feature off, the existing "shader passthrough not enabled"
error is unchanged. Requiring an unsupported instance feature still does not fail
instance creation (as Dawn). MSL input is not affected.

R10. **Unchanged.** WGSL pipelines apart from R2, the C API types, and Tint.

## Tests

- Unit (core, Noop): R1 one rejected case per validator group for render passthrough
  (for example sample count 3, a depth format color target, attribute offset past the
  stride, too many color targets), and a valid passthrough render pipeline still
  builds; R2 a compute layout with a compute-only buffer and a compute-and-fragment
  buffer gets slots 0 and 1, and a render layout keeps per-stage slots; R3 the error
  kind for each mismatch, sync and async; R4 each dimension and the product; R5; R6.
- Unit (FFI): R8 each case; R9 the instance feature report with the cargo feature on
  and off, SPIR-V with and without the required feature, and that MSL is unaffected.
- Claude, Metal e2e: a WGSL compute pipeline with an explicit layout whose binding 0 is
  a compute-only storage buffer and binding 1 a compute-and-fragment storage buffer
  writes distinct values to each; both read back correctly. Fails before R2. The MSL
  passthrough e2e (`e2e_metal_shader_passthrough.rs`) and the SPIR-V one under MoltenVK
  (`e2e_vulkan_shader_passthrough.rs`, which now requires the instance feature) pass.
- Claude: the full Metal e2e; CTS on Metal `webgpu:api,*` and
  `webgpu:shader,execution,*` must not regress (R2 touches WGSL compute slots).

## Out of scope

- A runtime query for whether passthrough is compiled in.
- Checking render entry names against MSL entry metadata.
- External textures in SPIR-V passthrough.

## Slices

- **S1 (coding agent):** R1-R10 with the unit tests.
- **S2 (Claude):** e2e, CTS, Phase Review.
