# Narrowing audit — "a passed parameter is reduced to a less-informative representation" (lossy)

A proactive audit (3 parallel trace agents, FFI → `yawgpu-core` → `yawgpu-hal` → Metal/Vulkan/GLES) for the
pattern where **a WebGPU parameter DOES flow through the conversion chain but is degraded en route into a
type/representation with fewer states, less range, or less precision — so a legal input silently produces
the wrong result.** This is the COMPLEMENT of the [[threading-audit]] ("validated but never threaded to the
HAL"): there the value never arrived; here it arrives degraded. Verified by Claude reading each site.

**Clean (no gap):** numeric narrowing is disciplined — every user offset/size/count crossing a width
boundary uses a checked `try_from(...).map_err(...HalError)` or is bounded by validation/limits below the
target range (0 reachable truncation bugs). Most enum mappings are 1:1; unsupported formats collapse to
`HalTextureFormat::Unsupported`/`HalVertexFormat::Unsupported` which every backend turns into a `HalError`
(not a silent wrong-GPU result); `*_Undefined → documented default` is spec-correct.

**Findings — fixed in a single combined slice (user-approved 2026-06-06):**

| # | Site | Pattern | Severity | Status |
|---|---|---|---|---|
| 1 | `vulkan/encode.rs` `render_pass_clear_values` (~1902) + `subpass_clear_values` (~1400) | color `clearValue` always written to the `float32` member of `VkClearColorValue` regardless of the attachment's numeric class; integer formats (`*_UINT`/`*_SINT`, all `.renderable()`) need the `uint32`/`int32` member + `as u32`/`as i32` (not `as f32`) → **garbage clear of integer render targets**. Metal is correct. | **Tier-1 correctness** | OPEN |
| 2 | `yawgpu/src/ffi/mod.rs:1031` `hal_present_mode` | `WGPUPresentMode_FifoRelaxed` silently collapses to `Fifo` (`HalPresentMode` has no `FifoRelaxed` variant; `_ => Fifo` swallows it) | minor | OPEN |
| 3 | `yawgpu/src/conv/bind.rs:309, 333` | `map_texture_view_dimension(...).unwrap_or(D2)` maps BOTH `Undefined` (correct default) AND an out-of-range/invalid value to `D2` → invalid `viewDimension` on a BGL entry silently accepted instead of erroring (sibling `access` field + the texture-view path correctly error on invalid) | minor (validation strictness) | OPEN |
| 4 | `yawgpu-hal/src/metal/encode.rs:415` | MSL buffer-size `u32::try_from(size).unwrap_or(u32::MAX)` saturates instead of erroring; unreachable today only because `Adapter::limits()` pins `max_buffer_size` to 256 MiB — would silently report a wrong runtime-array length if real adapter limits are ever wired in | latent (defensive) | OPEN |
| 5 | `yawgpu-core/src/subpass.rs:540` `set_scissor_rect` | (adjacent, not narrowing) subpass scissor checks only `checked_add` overflow, NOT containment in the attachment extent like the main `render_pass.rs` path (`validate_scissor_rect(render_extent, …)`) — strictness asymmetry | minor (validation) | OPEN |

All five fixed in one HANDOFF / one diff, then one cycle: review → real-GPU verify (Vulkan integer-clear +
Metal parity) → Clean Review → commit. GLES (Tier 2) applies where mappable or returns a catalogued
`HalError` (`specs/blocks/67-gles-backend.md`); never relax core.
