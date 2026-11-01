# Block 119 — CTS d116bcc findings F-161 to F-164

Status: **COMPLETE (2026-11-01)**. Phase Review: 0 CRITICAL, 0 MAJOR, 4 MINOR; m1-m3
fixed, m4 recorded as R5d. GLES behaviour is unverified on a real driver.
Source: user request (2026-11-01); webgpu-native-cts `docs/FINDINGS.md` F-161 to
F-164, from the sweep at CTS `d116bcc` on yawgpu `ea67149`. All four were
re-measured at `ea67149` before this block was written, and each reproduces.
Oracle: Dawn at the pinned `third_party/dawn`.

## Problem

| Finding | Failure | Measured on Metal |
|---|---|---|
| F-163 | `alphaToCoverageEnabled` with a non-blendable target reaches Metal, whose API validation aborts the process | crash |
| F-162a | no late binding-size check at dispatch when `minBindingSize` is 0 | 3 cases in `dispatch`, 3 in `min_buffer_binding_size` |
| F-162b | direct dispatch does not reject a linear-index overflow | 2 cases |
| F-162c | indirect dispatch does not skip a linear-index overflow | 2 cases |
| F-161 | `snorm10-10-10-2` vertex format not implemented | 364 + 113 subcases |
| F-164 | `swizzle_assignment`, `buffer_view`, `fragment_depth`, `unrestricted_aliasing` not reported | 4 cases, plus dependent skips |

## Rules

### F-163: alpha-to-coverage target format

R1. With `alphaToCoverageEnabled`, pipeline creation fails with a validation error
unless target 0 has a format with an alpha channel that is blendable. Mirrors
Dawn `RenderPipeline.cpp` `ValidateFragmentState`. The rule is in `yawgpu-core`,
so it fires identically on every backend.

### F-162a: late binding size

R2. Dispatch and draw validate each bound buffer against the size the pipeline's
shader requires, for entries whose layout `minBindingSize` is 0. The failure is a
validation error surfaced at encoder finish, like other pass validation. Mirrors
Dawn `CommandBufferStateTracker.cpp` (`mMinBufferSizes`).

R2a. A render pipeline's required size for a binding is the maximum across its
vertex and fragment stages, as in Dawn `PipelineBase`. Draws in a render pass,
subpass draws, and draws recorded in a render bundle are all validated. A bundle
draw is validated at encode time against the bundle's own bind groups, and
executing a bundle re-validates nothing, matching Dawn. The render-side gap was
hidden because all four `render,shader_required_buffer_size` CTS cases skip without
`buffer_view`.

### F-162b: direct dispatch linear-index range

R3. A compute pipeline uses linear indexing when its entry point uses
`global_invocation_index` or `workgroup_index`.

R4. For such a pipeline, a direct dispatch is a validation error if the total
exceeds `u32::MAX`, where the total is `X * Y * Z`, multiplied by the workgroup
invocation count only for `global_invocation_index`. Mirrors the overflow logic in
Dawn `ComputePassEncoder.cpp`, including the checks on partial products, so no
intermediate wraps.

### F-162c: indirect dispatch range (S2)

R5. An indirect dispatch whose counts are invalid does not execute and raises no
error. The counts are GPU data, so every indirect dispatch goes through a GPU
validation step, as in Dawn `ComputePassEncoder.cpp`
`TransformIndirectDispatchBuffer`. Invalid means either:
- any count exceeds `maxComputeWorkgroupsPerDimension`, on every pipeline
- the pipeline uses linear indexing and the total exceeds the R4 range

R5a. The step is built in `yawgpu-core` from the existing internal-compute pattern
(`Device::timestamp_conversion_pipeline`, `create_internal_bind_group`,
`timestamp_conversion_command`), so all backends share it:
- `dispatch_workgroups_indirect` records a validation `ComputePassCommand`, then the
  user's dispatch as `Indirect` on a validated-arguments buffer at offset 0.
- The validation shader is Dawn's, without `duplicateNumWorkgroups` and
  `enableValidation`. Parameters go through immediates, not a uniform buffer.
- The client buffer binding offset is aligned down to
  `minStorageBufferOffsetAlignment`, with the remainder passed as a word offset.
- The validated-arguments buffer is a device-owned internal 12-byte buffer with
  storage and indirect use. Failure to create the pipeline or buffer invalidates the
  encoder, as the timestamp path does.

- Internal pipeline layouts are exempt from `maxImmediateSize`, because GLES reports
  0. This changes no user-visible rule: the check still applies to every layout a
  user creates. The GLES HAL uploads compute immediates to `tint_immediates[N]`, as
  its render path does for `first_instance` and the depth range. User render
  immediates remain undelivered on GLES.
- A failed internal buffer creation is not cached, so a later dispatch retries.

R5b. User-visible state and validation do not change: usage-scope checks stay
against the client buffer, and the client buffer is still recorded as referenced.
The internal buffer is also recorded as referenced.

R5c. Indirect buffers are created so an internal storage binding is legal on every
backend. The storage write must be visible to the following indirect read on each
backend: a Vulkan barrier from compute write to indirect read, and on GLES
`glMemoryBarrier` with `GL_COMMAND_BARRIER_BIT`. Repeated dispatches in one pass must
not observe a stale value.

### F-161: `snorm10-10-10-2`

R6. `WGPUVertexFormat_Snorm10_10_10_2` (0x2A) is a valid vertex format with no
feature gate: 4 bytes, 4 components, float shader type, alignment 4.

R7. It is implemented on every path `unorm10-10-10-2` has: core tables, MSL vertex
pulling through Tint, and the Metal, Vulkan, and GLES HAL mappings. Mappings follow
Dawn: `MTLVertexFormatInt1010102Normalized`, `A2B10G10R10_SNORM_PACK32`, and GLES
`GL_INT_2_10_10_10_REV` normalized.

### F-164: WGSL language features

R8. `buffer_view` (0x0C) and `swizzle_assignment` (0x0D) are reported and allowed by
the Tint shim. Tint marks both as shipped, and Dawn reports them. Any backend
execution gap found afterwards is a finding, not a reason to widen core.

R9. `fragment_depth` stays unreported, matching Dawn: Tint marks it unsafe
experimental. `unrestricted_aliasing` stays unreported: `webgpu.h` has no enumerator
for it. Both `have_all_language_features` cases remain failing, as on Dawn (F-159).

R5d. Invalid counts now reach `vkCmdDispatchIndirect` as zeros, as in Dawn. The HAL
skips zero-sized direct dispatches because of a Mesa ANV Haswell wedge. Whether that
wedge also applies to indirect dispatch is unconfirmed.

## Verification (2026-11-01)

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,287 passed, 0 failed |
| Metal e2e | 143 passed; tiled 3 passed |
| Metal HAL ignored tests | 58 passed |
| Vulkan e2e under the Khronos layer | 133 passed, 1 failed (known `imageView2DOn3DImage`) |
| Vulkan indirect and threading-audit e2e under sync validation | 22 passed, 0 hazards, 0 VUIDs |
| fmt, clippy on 10 feature sets | clean |
| webgpu-native-cts, Metal, `webgpu:*`, CTS `d116bcc` | `fail=2 crash=0`, 0 regressions |

Metal subcases moved from pass 2,071,962 / skip 184,925 / fail 491 / crash 2 to pass
2,102,837 / skip 154,617 / fail 2 / crash 0. The 2 failures are the R9 cases, which
Dawn fails too. The full CTS run predates m1, which only changes when the internal
buffer is created; the affected dispatch and compute queries were re-run after it.

The full Vulkan e2e run predates the GLES change and m1. Neither touches a Vulkan
path; the indirect and threading-audit e2e were re-run after both.

### Linux native Vulkan and GLES (2026-11-01)

Run on yawgpu `9977e34` / Dawn pin `843d53377` / CTS `b7ac3ef`, Linux / NVIDIA RTX
5060 Ti, driver 615.71.09. This closes the native Vulkan and GLES gaps left by the
Metal host. Vulkan e2e ran with `--test-threads=1` and without the Khronos layer.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 0 failed |
| Vulkan e2e (`vulkan,tiled,shader-passthrough`) | 133 passed, 0 failed |
| GLES e2e, including `e2e_gles_indirect_dispatch` | 36 passed, 0 failed |
| `submit_compute_pass_uploads_immediates_to_tint_immediates` | passed on the EGL device |
| webgpu-native-cts, native Vulkan, per-area `webgpu:*` | `fail=2 crash=0` with expectations |
| webgpu-native-cts, GLES, per-area `webgpu:*` | `fail=28,810 crash=0`, raw |

The 2 Vulkan failures are the R9 cases. The F-161 to F-164 queries have no other
Vulkan failures. On GLES the indirect dispatch queries, `api,validation,dispatch`, and
`min_buffer_binding_size:compute` all pass. The GLES failures in the Block 119 areas
are existing Tier 2 gaps. `alpha_to_coverage` reports a color target that is not
color-renderable on GLES 3.1 (`HalError`), and the 38 `snorm10-10-10-2` failures are in
a vertex-format test that fails the same way for other formats, such as `float32`.

## Slices

- **S1 (coding agent):** R1-R4, R6-R9.
- **S2 (coding agent, then Claude):** R5.
- **S3 (Claude):** real-GPU e2e; CTS on Metal; Phase Review.
