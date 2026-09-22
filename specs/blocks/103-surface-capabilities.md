# Block 103 — Surface capabilities come from the HAL

Status: **COMPLETE (2026-09-22)** — `a2907ca`; Metal: inline + C-ABI e2e (Dawn's list, sRGB + Immediate + Premultiplied + CopySrc readback, Rgba16Float, rejection of unlisted modes, read-before-render zero); MoltenVK: e2e (driver-reported set deduplicated per format — a driver lists one entry per colour space; MoltenVK reports Fifo/Immediate, Opaque/Unpremultiplied/Inherit, usages incl. StorageBinding; read-before-render zero). The three Khronos-validation follow-ups (VUID 02662 view usage ⊄ image usage, VUID 05149 present semaphore destroyed while pending, VUID 01689 zero `currentExtent`) were fixed in `974a818`; the Phase Review with Block 104 (`251a4cc`, `c83bbbf`) fixed the iOS-unavailable `setDisplaySyncEnabled` (C1), the swapchain `TRANSFER_DST` for lazy zero-init (M2), the `Internal` error kind for capability-query failures (m2), the `Undefined`-format guard (m3) and the double device-idle wait (m4). Backlog item **B5** in
`specs/tracking/backlog.md`. Extends Block 70 "Surface" (descriptor
validation, sentinel resolution) and Block 85 (Win32 surface).

## Problem

`wgpuSurfaceGetCapabilities` returns compile-time constants
(`yawgpu/src/ffi/mod.rs`: formats `BGRA8Unorm`/`RGBA8Unorm`, present
mode `Fifo`, alpha `Opaque`, usage `RenderAttachment`) and
`wgpuSurfaceConfigure` validates against the same constants. Real
backends can do more, and Dawn advertises it:

| | Dawn Metal (`PhysicalDeviceMTL.mm` `GetSurfaceCapabilities`) | Dawn Vulkan (`PhysicalDeviceVk.cpp` `GetSurfaceCapabilities`) |
|---|---|---|
| usages | `RenderAttachment \| TextureBinding \| CopySrc \| CopyDst` | from `VkSurfaceCapabilitiesKHR.supportedUsageFlags` (`TRANSFER_SRC`→CopySrc, `TRANSFER_DST`→CopyDst, `COLOR_ATTACHMENT`→RenderAttachment, `SAMPLED`→TextureBinding, `STORAGE`→StorageBinding) |
| formats | `BGRA8Unorm`, `BGRA8UnormSrgb`, `RGBA16Float` (+ `RGB10A2Unorm` on macOS) | from `vkGetPhysicalDeviceSurfaceFormatsKHR`, keeping `R8G8B8A8_UNORM/SRGB`, `B8G8R8A8_UNORM/SRGB`, `A2B10G10R10_UNORM_PACK32`, `R16G16B16A16_SFLOAT` |
| present modes | `Fifo`, `Immediate`, `Mailbox` | from `vkGetPhysicalDeviceSurfacePresentModesKHR` (`FIFO`, `FIFO_RELAXED`, `MAILBOX`, `IMMEDIATE`) |
| alpha modes | `Opaque`, `Premultiplied` | from `supportedCompositeAlpha` (`OPAQUE`→Opaque, `PRE_MULTIPLIED`→Premultiplied, `POST_MULTIPLIED`→Unpremultiplied, `INHERIT`→Inherit) |

The Vulkan HAL already maps `Immediate` / `Mailbox` / `FifoRelaxed` in
`select_present_mode`; only the FFI membership check keeps them out.

## Behaviour contract

### R1 — HAL capability query

`yawgpu-hal` gains

```rust
#[non_exhaustive] pub struct HalSurfaceCapabilities {
    pub usages: HalTextureUsage,
    pub formats: Vec<HalTextureFormat>,        // preferred first
    pub present_modes: Vec<HalPresentMode>,    // Fifo always present
    pub alpha_modes: Vec<HalCompositeAlphaMode>, // first = the `Auto` alias
}
#[non_exhaustive] pub enum HalCompositeAlphaMode { Opaque, Premultiplied, Unpremultiplied, Inherit }
```

and `HalSurface::capabilities(&self, adapter: &HalAdapter) -> Result<HalSurfaceCapabilities, HalError>`
(static enum dispatch; a backend mismatch between surface and adapter is
a `HalError`):

- **Noop**: unchanged synthetic set — formats `[Bgra8Unorm, Rgba8Unorm]`,
  present `[Fifo]`, alpha `[Opaque]`, usage `render_attachment` — so
  every Noop integration test keeps its expectations.
- **Metal**: Dawn's list verbatim — usages `render_attachment |
  texture_binding | copy_src | copy_dst`; formats `[Bgra8Unorm,
  Bgra8UnormSrgb, Rgba16Float, Rgb10a2Unorm]` (the macOS set; iOS drops
  `Rgb10a2Unorm`); present `[Fifo, Immediate, Mailbox]`; alpha
  `[Opaque, Premultiplied]`.
- **Vulkan**: queried per surface exactly as Dawn's table above, order
  preserved from the driver; unknown formats / modes dropped; `Fifo`
  is always present per the Vulkan spec. `StorageBinding` is mapped when
  offered but yawgpu's swapchain images are created with the requested
  usage only — plus `TRANSFER_DST` whenever the surface offers it, so the
  Block 104 lazy zero-init of an acquired image can `vkCmdClearColorImage`
  it (the usage is internal; it is not reported as `CopyDst`).
- **GLES** (Tier 2): `[Rgba8Unorm, Bgra8Unorm]`, `[Fifo]`, `[Opaque]`,
  `render_attachment` (unchanged behaviour; catalogued in Block 67).

### R2 — FFI reports the HAL answer

`wgpuSurfaceGetCapabilities(surface, adapter, caps)` fills `caps` from
`HalSurface::capabilities(adapter.hal())`, mapping HAL → `webgpu.h`
enums (`hal_surface_format` inverse; present / alpha enums 1:1). The
arrays are heap-allocated per call with their real lengths and freed by
`wgpuSurfaceCapabilitiesFreeMembers` — the current fixed-size
`Box<[T; N]>` scheme is replaced by `Vec::into_boxed_slice` +
`slice::from_raw_parts_mut(ptr, count)` reconstruction using the counts
in the struct (the header guarantees `FreeMembers` receives the struct
`GetCapabilities` filled). An error surface or a HAL failure returns
`WGPUStatus_Error` with zeroed counts / null pointers. The
adapter/surface backend mismatch is `WGPUStatus_Error` too.

### R3 — Configure validates against the same capabilities

`surface_configuration_error` takes the capabilities (queried once at
configure time from the surface + the device's adapter) instead of the
constants:

- `format ∈ formats`, else `"surface configuration format is not supported"`;
- `usage != 0 && usage ⊆ usages`, else `"surface configuration usage is not supported"`;
- `resolved_present_mode(presentMode) ∈ present_modes`, else the existing
  present-mode message; `Undefined → Fifo` stays;
- `resolved_alpha_mode(alphaMode) ∈ alpha_modes`, else the existing
  alpha-mode message; `Auto → alpha_modes[0]` (first-capability alias,
  as Block 70 specifies);
- `viewFormats`: each must be the format itself or its sRGB / non-sRGB
  sibling (WebGPU `GPUCanvasConfiguration` rule; Dawn `ValidateSurfaceConfiguration`),
  else `"surface configuration view format is not compatible"`.

Every existing message string is kept verbatim. A `format` of `Undefined`
is always rejected with the format message (a HAL format with no
`webgpu.h` mapping must never make `Undefined` a member). A failure of the
HAL capability query itself at configure time (no HAL surface for the
adapter's backend, driver error) is dispatched as an **`Internal`** error,
not `Validation`; `wgpuSurfaceGetCapabilities` returns `WGPUStatus_Error`
for the same cases.

### R4 — The HAL receives the resolved modes

`HalSurfaceConfiguration` gains `alpha_mode: HalCompositeAlphaMode`
(constructor updated; `present_mode` already there).

- **Metal** `configure`: `layer.setPixelFormat` for every advertised
  format (the `map_texture_format` table already has them);
  `layer.setDisplaySyncEnabled(present_mode != Immediate)` **on macOS
  only** (`#[cfg(target_os = "macos")]`; the property is
  `API_UNAVAILABLE(ios, tvos, watchos, visionos)` and Dawn guards it with
  `DAWN_PLATFORM_IS(MACOS)` in `SwapChainMTL.mm` — on iOS `Immediate`
  behaves as `Fifo`; `Mailbox` behaves as `Fifo` on Metal, as in Dawn);
  `layer.setOpaque(alpha_mode != Premultiplied)`;
  `layer.setFramebufferOnly(false)` stays (needed for `copy_src` /
  `texture_binding`). The acquired `MetalTexture` reports the configured
  format and usage.
- **Vulkan** `create_swapchain`: `image_format` + `image_color_space`
  taken from the matching `VkSurfaceFormatKHR` entry (not hard-coded
  `SRGB_NONLINEAR` when the driver reports otherwise); `image_usage`
  = the requested usage mapped to `VkImageUsageFlags` (a requested bit
  the surface does not support is a validation error at the FFI, so the
  HAL may `HalError` if it still sees one) **plus `TRANSFER_DST` when
  `supportedUsageFlags` offers it** (lazy zero-init, Block 104 R6); the
  swapchain texture records its final usage so view creation and clears
  can intersect against it; `composite_alpha` from
  `alpha_mode`; `present_mode` = the requested mode exactly
  (`select_present_mode`'s FIFO fallback stays only for a driver that
  lies between the capability query and swapchain creation).
- **GLES**: unchanged; a non-`Opaque` alpha or non-`Fifo` present mode
  never reaches it because R3 rejects them.

### R5 — `wgpuSurfaceGetCurrentTexture` texture

The core `Texture` wrapping the acquired image carries the configured
`format`, `usage` and `viewFormats` (already threaded through
`SurfaceConfigurationState`); nothing else changes.

## Tests

- **HAL inline**: Noop `capabilities()` equals the synthetic set;
  Metal (`#[ignore]` real device) `capabilities()` equals Dawn's macOS
  list; Vulkan (`#[ignore]`, MoltenVK through a `CAMetalLayer` surface)
  `capabilities()` contains `Fifo`, `Opaque`, `Bgra8Unorm` and every
  format/mode it reports is one `vkGetPhysicalDeviceSurface*` reported
  (compare against a direct `ash` query in the test). Pure conversion
  fns (`vk::Format → HalTextureFormat`, `vk::PresentModeKHR →
  HalPresentMode`, composite-alpha bits → modes, usage flags → usage)
  unit-tested on Noop.
- **FFI inline**: `wgpuSurfaceGetCapabilities` round-trips arbitrary
  counts through `FreeMembers` without leaking (3 formats / 2 modes /
  4 alphas via a Noop-injected capability set if the FFI has a seam;
  otherwise the Noop set); error surface → `Error` + zeroed members;
  `surface_configuration_error` accepts every member of the capability
  set and rejects a non-member of each kind with the existing messages;
  view-format sibling rule.
- **Integration (Noop, `yawgpu/tests/surface_validation.rs`)**: existing
  expectations unchanged (Noop set is unchanged); add the view-format
  sibling rule cases.
- **Real-GPU e2e (Claude)**: extend `yawgpu/tests/e2e_metal_surface.rs`:
  capabilities equal Dawn's Metal list; configure with
  `Bgra8UnormSrgb` + `Immediate` + `Premultiplied` +
  `RenderAttachment | CopySrc` succeeds, `getCurrentTexture` returns
  `SuccessOptimal`, a clear + `copyTextureToBuffer` readback of the
  surface texture works (proves `copy_src`), `present` succeeds;
  `Rgba16Float` configure + present succeeds. New
  `yawgpu/tests/e2e_vulkan_surface.rs` (MoltenVK via `CAMetalLayer` +
  `VK_EXT_metal_surface`): capabilities agree with a direct `ash`
  query; configure with a driver-reported non-Fifo mode succeeds.
- **CTS**: not applicable (the harness never calls `wgpuSurface*`).

## Out of scope

- Linux / Android window surfaces (backlog C1–C3).
- `WGPUSurfaceCapabilities.nextInChain` extensions.
- Changing the Noop surface contract.
