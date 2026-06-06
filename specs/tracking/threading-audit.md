# Threading audit — "validated/accepted but not threaded to the HAL" (silently wrong)

A proactive audit (5 parallel trace agents, FFI → `yawgpu-core` → `yawgpu-hal` → Metal/Vulkan) for the
recurring pattern behind F-035 (`blend_constant`), F-038 (`stencil_reference`), and F-043 (`depthSlice`):
**a WebGPU field is parsed/validated in `yawgpu-core` but its value never reaches the HAL backend, so the
GPU uses a default and the result is silently wrong.** The CTS currently passes these because only the
default values are exercised on real hardware; non-default values would surface them (future CTS findings).

**Clean (no gap found):** the full `WGPUDepthStencilState` pipeline state (depthWrite/compare/bias/slope/
clamp, stencil ops/compare/masks, format) and the full `WGPUSamplerDescriptor` (all address modes incl.
`addressModeW`, filters, lod clamps, compare, maxAnisotropy). All draw/dispatch/binding/copy params
(firstVertex/firstInstance/baseVertex/firstIndex, dynamic offsets, vertex/index offsets, copy
offset/bytesPerRow/rowsPerImage/origin.z) are applied on both backends.

**Open findings — fixed in a single combined slice (user revised 2026-06-04; the groups below
are the per-group structure of that one HANDOFF, not separate phases):**

| Group | Field(s) | Symptom | Status |
|---|---|---|---|
| A | `primitive.cullMode`, `primitive.frontFace` | culling never happens / winding wrong (Vulkan hardcodes NONE/CCW; Metal never sets cull/winding) | OPEN |
| B | `setViewport`, `setScissorRect` | validated only, never applied; both backends hardcode full-attachment viewport (depth 0–1) + full scissor | OPEN |
| C | `depthReadOnly`, `stencilReadOnly` | carried to the HAL attachment but not consulted; read-only forces loadOp Clear → read-only depth/stencil is CLEARED instead of loaded+preserved | OPEN |
| D | `multisample.mask`, `multisample.alphaToCoverageEnabled` | parsed/validated, never threaded; sample mask + alpha-to-coverage ignored under MSAA | OPEN |
| E | `primitive.unclippedDepth` | Vulkan hardcodes `depth_clamp_enable(false)`, Metal no `setDepthClipMode`; also not feature-gate-rejected (`depth-clip-control`) | OPEN |
| F | `dispatchWorkgroupsIndirect` | validates + records the referenced buffer but records NO executable command → silent no-op (compute analog of the resolved F-034 indirect-draw gap) | OPEN |

All six groups are implemented in one HANDOFF / one diff, then one cycle: review → real-GPU verify (Metal
+ Vulkan/MoltenVK) → Clean Review → commit. GLES (Tier 2) applies the field or returns a catalogued
`HalError` (`specs/blocks/67-gles-backend.md`); never relax core.
