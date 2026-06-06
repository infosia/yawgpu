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

**Findings — RESOLVED in a single combined slice (user-approved 2026-06-06). Real-GPU verified on
Metal + Vulkan/MoltenVK; Clean Review clean (no CRITICAL/MAJOR):**

| # | Site | Fix | Severity | Status |
|---|---|---|---|---|
| 1 | `vulkan/encode.rs` `render_pass_clear_values` + `subpass_clear_values` | new `HalColorClearKind` + `HalTextureFormat::color_clear_kind()` classifier (`hal/format.rs`); `vulkan_color_clear_value(format, [f64;4])` picks the `float32`/`uint32`/`int32` `VkClearColorValue` member with `as f32`/`as u32`/`as i32`. Integer render targets now clear to the exact value. Metal untouched (was correct). | **Tier-1 correctness** | RESOLVED |
| 2 | `ffi/mod.rs` `hal_present_mode` | `HalPresentMode::FifoRelaxed` added; FFI maps it explicitly; Vulkan `create_swapchain` now queries supported present modes + `select_present_mode` picks `FIFO_RELAXED` if supported else `FIFO` (for ALL modes — also fixes the prior unconditional Immediate/Mailbox use); Metal/GLES map to vsync/Fifo explicitly | minor | RESOLVED |
| 3 | `conv/bind.rs` | `map_bgl_texture_view_dimension` distinguishes `Undefined`(→D2) from out-of-range(→`set_first_error`+reject); applied to texture + storage-texture BGL entries | minor (validation) | RESOLVED |
| 4 | `metal/encode.rs` MSL buffer-size | checked `u32::try_from(...).map_err(... HalError)` instead of `unwrap_or(u32::MAX)` saturation | latent (defensive) | RESOLVED |
| 5 | `subpass.rs` `set_scissor_rect` | now calls `validate_scissor_rect(Some(self.inner.extent), …)` to match the regular render-pass path | minor (validation) | RESOLVED |

Verification: `e2e_{metal,vulkan}_threading_audit.rs` gained `*_integer_render_target_clear_is_exact`
(clear an `R32Uint` target to `0xDEADBEEF`, read back exact) — green on Metal + Vulkan/MoltenVK
(non-tautological: the pre-fix float-union path would read a bit-reinterpreted value); the other 6 probes in
each file still pass. Noop/feature-gated unit tests for all 5 items; all clippy gates + fmt clean.

**Follow-up (Clean Review MINOR, forward-safety, not a current defect):** `HalTextureFormat::color_clear_kind`
classifies integer formats by explicit listing with a `_ => Float` catch-all, and `HalTextureFormat` is not
`#[non_exhaustive]` — a FUTURE integer color format added without updating the classifier would silently fall
into `Float` and re-introduce the integer-clear bug with no compile error. All current `*Uint`/`*Sint`
variants (incl. `Rgb10a2Uint`) are covered. Harden later by classifying the float formats explicitly (so a
new unlisted variant fails to compile) or a warning comment at the `_` arm.

GLES (Tier 2) unaffected (no integer-clear/present-mode regressions); core rules unchanged.
