# Over-strict validation audit (Audit B) — spec-legal ops false-rejected

Proactive sweep (2 parallel agents: create-validation + command/usage-scope validation) for the
F-005/F-009/F-039/F-042 family — `yawgpu-core` validation STRICTER than the WebGPU spec that false-rejects a
legal operation (then, correctly per spec, the error poisons the command buffer). Part of
[[cts-failure-patterns]] (pattern 2). The encoder-poisoning mechanism itself is spec-correct (a validation
error during encoding invalidates the command buffer) — only the over-strict rules are bugs.

## Findings

| # | Site | Issue | Severity | Status |
|---|---|---|---|---|
| **1** | `yawgpu-core/src/pass.rs:585` `validate_vertex_buffer_oob` | required size = `array_stride * (first+count)`; the WebGPU `draw`/`drawIndexed` rule is `(strideCount-1)*arrayStride + lastStride` (`lastStride = max(attr.offset + size(attr.format))`). yawgpu over-reports by `arrayStride - lastStride` on the last element, so a spec-minimum-sized vertex buffer (any layout where `arrayStride > lastStride` — padded/interleaved vertices) is **false-rejected**. wgpu + spec confirm; the matching CTS case `vertex_buffer_oob` (`tests/cts/.../render/draw.rs:289`) is already `#[ignore]`d for this exact reason. | over-strict (common) | **FIX (this slice)** |
| 2 | `yawgpu-core/src/bind_group.rs:463` `validate_bind_group_storage_texture` | rejects `array_layer_count() != 1` for storage-texture bindings; current WebGPU requires only `mipLevelCount == 1` for a storage view — a multi-layer `2d-array` storage binding is spec-legal (wgpu allows). Provenance: spec item S32 cited a since-relaxed spec line (same "spec moved on" class as F-016/F-018). Internally inconsistent: the BGL validator already admits `2d-array` storage layouts. Uncaught by CTS. | over-strict (medium) | DEFERRED — needs backend array-storage-view binding (F-041 was single-layer); removing the core check without backend support would turn a false-reject into a silent-wrong. Track as a dedicated slice. |
| 3 | `yawgpu-core/src/command_encoder.rs:1934` `validate_same_texture_copy` | blanket-rejects ALL same-texture same-mip `copyTextureToTexture` when `dimension == D3`; the spec keys 3D copy subresources by mip + z-slice range, so disjoint z-ranges are legal (wgpu returns Ok when `src.z >= dst.z+depth \|\| dst.z >= src.z+depth`). Uncaught by CTS (only 2D tested; upstream has a TODO). | over-strict (narrow) | DEFERRED — narrow + needs same-3D-texture-copy backend verification. Track as a dedicated slice. |

## Spot-checked and confirmed spec-CORRECT (not over-strict)
Compute per-dispatch usage scope (F-039), render write+write-across-draws (F-042), copy `bytesPerRow`
256-alignment (gated correctly: required for B2T/T2B, not `queue.writeTexture`), B2B/clearBuffer 4-byte
alignment + range OOB, dynamic-offset alignment+bounds, viewport/scissor bounds, multisampled+Float
sampleType reject, `unclippedDepth` reject (no `depth-clip-control` feature), depth-format
depthCompare/depthWrite requirement, MapRead/MapWrite usage exclusivity, buffer-range OOB, indexed/indirect
OOB skips — all match the spec / passing CTS.

## Fix (this slice): finding 1 only
`validate_vertex_buffer_oob`: compute `last_stride = max over layout.attributes of (attribute.offset +
vertex_format_byte_size(attribute.format))` (0 if no attributes), and
`required_size = (stride_count - 1) * array_stride + last_stride` for `stride_count >= 1` (no requirement
when `stride_count == 0`). Un-ignore the CTS `vertex_buffer_oob` test (it should pass). Findings 2 + 3 are
real but deferred (backend-spanning / narrow) — user to greenlight as separate slices.
