# Block 148 — Shader passthrough: Phase Review fixes and the `ShaderSourceSPIRV` instance feature

Status: **COMPLETE (2026-11-08)**. Phase Review: 0 CRITICAL, 2 MAJOR (both fixed), 8 MINOR;
six fixed, two moved to the backlog, see below.
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

R10. **Unchanged.** WGSL pipelines apart from R2 and R11, the C API types, and Tint.

R11. **Render passes do not bind compute-only entries.** In a render pipeline's Metal
binding map, an entry visible to neither the vertex nor the fragment stage has no slot,
and the Metal render encoding (pass and bundle) binds nothing for it. Today it gets
`metal_index` 0, and `per_stage_slots(None, None, 0)` binds it at slot 0 of both
stages, over the resource a shader reads there (found by the Phase Review; a Metal e2e
where a compute-only uniform overwrites the fragment's `[[buffer(0)]]` renders red
instead of green). Every other user of the flat `metal_index` fallback is checked to be
compute-only or to handle a missing slot.

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
- **S1b (coding agent):** R11 and the Phase Review MINORs.
- **S2 (Claude):** e2e, CTS, Phase Review.

## Verification (2026-11-08)

Run on Apple M2, macOS 27.2. Vulkan here is MoltenVK under the Khronos layer, which is
reference-only.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,441 passed, 0 failed; with `yawgpu/shader-passthrough` 1,490 passed, 0 failed |
| fmt; clippy `-D warnings` on default, `metal`, `vulkan`, `gles`, `metal,tiled`, `vulkan,tiled`, `shader-passthrough`, `metal,shader-passthrough`, `vulkan,shader-passthrough` | clean |
| HAL Metal lib and ignored tests | 201 passed; 66 passed |
| `e2e_metal_compute_binding_slots` (new) | 2 of 2. Against `92a6f21` (before this block) both fail: the compute pipeline reads back `[[0, 0], [0, 0], [0, 0]]` instead of `[[11, 5], [22, 7], [33, 12]]` (R2), and the render pipeline draws red, the compute-only uniform, instead of green (R11) |
| Metal e2e and every other ignored `yawgpu` test, `metal,tiled,shader-passthrough` | 178 passed, 0 failed |
| All `e2e_vulkan_*`, `vulkan,shader-passthrough`, MoltenVK, Khronos layer | 163 passed, 1 failed (the known `imageView2DOn3DImage` portability case, whose VUIDs are the only validation output); `e2e_vulkan_shader_passthrough` (now requires the instance feature) 2 of 2 and `e2e_vulkan_compute_binding_slots` 2 of 2 |
| `examples/triangle_passthrough` | Metal `--verify` passes. On MoltenVK the SPIR-V modules and pipeline are created without error; the run then stops at surface acquisition with `SuccessSuboptimal`, as the plain `triangle` example does on this host |
| webgpu-native-cts, Metal, `webgpu:api,*` | pass 589,574, skip 101,793, fail 0, crash 0, on the library after R11; identical to Block 145 |
| webgpu-native-cts, Metal, `webgpu:shader,execution,*` | pass 857,330, skip 25,860, fail 0, crash 0, xfail 63, xpass 9 (F-160), on the library after R11; pass + xfail identical to Block 146 |

### Windows native Vulkan (2026-11-08)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351, at yawgpu
`c4cbb5b`. The Khronos validation layer was on for the HAL ignored and e2e runs.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,441 passed, 0 failed |
| clippy `-D warnings`, default (workspace), `vulkan,tiled`, `vulkan,shader-passthrough` and `gles` (`yawgpu-core`, `yawgpu-hal`, `yawgpu`, `yawgpu-tint`), all targets | clean |
| Vulkan HAL lib tests, `vulkan,tiled` | 386 passed, 0 failed |
| Vulkan HAL ignored tests | 82 passed, 0 failed, 0 VUIDs |
| Vulkan e2e, every integration test, `vulkan,tiled,shader-passthrough`, `--ignored` | 178 passed, 0 failed, 0 VUIDs |

On Vulkan the Block 148 paths ran: `vulkan_spirv_passthrough_compute_doubles_storage_buffer`,
`vulkan_spirv_passthrough_render_draws_solid_color`,
`vulkan_compute_bindings_with_render_visibility_reach_their_buffers` and
`vulkan_render_ignores_compute_only_bindings` pass.

## Phase Review

Two fresh reviews: one after S1 (R1-R10) and one after S1b (R11 and the MINOR fixes).

| ID | Finding | Disposition |
|---|---|---|
| M1 | A Metal render pipeline bound a layout entry visible only to compute at slot 0 of both stages, over the resource the shader reads there (predates this block) | R11 added and fixed; Metal e2e |
| M2 | The examples framework created its instance without `ShaderSourceSPIRV`, so `triangle_passthrough` failed on Vulkan under R9 | The framework requests the feature |
| m1 | The async callback computed the backend-mismatch message separately from the error core recorded | The callback reports the recorded error |
| m2 | The async mismatch test injected its message instead of using the real path | Replaced by core and FFI tests of the real path |
| m3 | With both a WGSL and a SPIR-V chain and no instance feature, the error was "SPIR-V is disallowed." instead of the one-source error | The feature check runs on the mapped SPIR-V source only |
| m4 | WGSL accepts a depth-format color target with `writeMask 0` and no shader output; passthrough now rejects it (outside R10) | Backlog |
| m5 | Stale `#[allow(unused_mut)]`; an inaccurate `metal_buffer_binding_map` doc comment | Fixed |
| m6 | `yawgpu.h` did not say that a passthrough stage may only use entries visible to it | Documented |
| m7 | The render e2e had the compute-only entry only after the fragment entry and no bundle | Compute-only entries before and after; the right half is drawn through a render bundle |
| m8 | Tiled subpass draws do not bind external textures (reported, not verified; not caused by this block) | Backlog |
