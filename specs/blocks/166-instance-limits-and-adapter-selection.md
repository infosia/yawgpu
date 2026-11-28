# Block 166 — Instance limits and adapter-selection options are consumed

Status: **COMPLETE (2026-11-28)**. Phase Review: 0 CRITICAL, 0 MAJOR.
Source: parameter-propagation audit at `b772306`
(`specs/tracking/threading-audit.md`, "Propagation audit 2026-11"), candidates 05, 06, 10,
12.
Oracle: pinned Dawn `src/dawn/native/Instance.cpp` (`ReifyDefaultLimits`,
`ValidateLimits`, `APIGetInstanceLimits`, `APIWaitAny`, `EnumerateAdapters`) and
`Adapter.cpp` `SortAdapters`; `webgpu.h` `WGPURequestAdapterOptions.compatibleSurface`
("requires the adapter to be able to output to a particular surface. If this is not
possible, the request returns null").

## Problem

`wgpuCreateInstance` reads the required features and the backend chain but not
`requiredLimits`; `wgpuInstanceWaitAny` forwards any number of futures to a timed wait
although `wgpuGetInstanceLimits` advertises `timedWaitAnyMaxCount = 64`.
`wgpuInstanceRequestAdapter` reads `featureLevel`, `backendType` and
`forceFallbackAdapter`; `compatibleSurface`, `powerPreference` and the
`WGPURequestAdapterWebXROptions` chain are ignored, so a request can return an adapter that
cannot present to the required surface, and an XR request succeeds.

## Rules

R1. **Instance limits are negotiated.** With `requiredLimits` non-null,
`timedWaitAnyMaxCount` greater than the supported maximum (64) fails instance creation
(`wgpuCreateInstance` returns NULL). A value of 0 or anything up to 64 yields the
instance limit 64 (a maximum-class limit takes the better default). With
`requiredLimits` null the instance limit is 64. The instance stores its negotiated limit.

R2. **Timed waits honour the limit.** `wgpuInstanceWaitAny` with `timeout_ns > 0` and
`future_count` greater than the instance's `timedWaitAnyMaxCount` returns
`WGPUWaitStatus_Error` without waiting. The check follows the existing
timed-wait-feature check. A zero timeout has no count restriction. An empty wait
(`future_count == 0`) that passes both checks returns `WGPUWaitStatus_Success`.

R3. **`wgpuGetInstanceLimits` rejects a chain.** A non-null `nextInChain` on the output
struct returns `WGPUStatus_Error`.

R4. **WebXR requests are declined.** A `WGPURequestAdapterOptions` chain containing
`WGPURequestAdapterWebXROptions` (any `xrCompatible` value) completes with
`WGPURequestAdapterStatus_Unavailable` and a null adapter.

R5. **`compatibleSurface` filters.** With a non-null `compatibleSurface`, only adapters
that can present to that surface are eligible: the surface is not an error surface and
the surface-capability query for the (surface, adapter) pair succeeds — the same query
`wgpuSurfaceGetCapabilities` uses. No eligible adapter completes with `Unavailable` and
a null adapter.

R6. **`powerPreference` ranks, it does not filter.** Eligible adapters are stably sorted
by adapter-type rank before the first is taken: `HighPerformance` ranks discrete 0 and
integrated 1; `LowPower` ranks integrated 0 and discrete 1; `Undefined` ranks both 1;
CPU is always 2 and unknown 3. Equal ranks keep enumeration order.

R7. **Filter, then rank.** Selection is: enumerate at the feature level → keep fallback
adapters only when `forceFallbackAdapter` → keep the requested `backendType` → R5 → R6 →
first. Capability filtering and preference ranking are separate steps; each is a
function with its own unit test.

## Slices

S1: R1–R7 in `yawgpu/src/ffi/instance.rs` (and the instance impl it owns), inline tests.

## Verification

- Inline FFI tests:
  - R1: creation with `requiredLimits` 0, 64 and null succeeds; 65 returns NULL.
  - R2: with TimedWaitAny enabled, 64 ready futures with a positive timeout return
    `Success`; 65 return `Error`; 65 with a zero timeout are not rejected for the count.
  - R3: a chained `WGPUInstanceLimits` returns `Error`.
  - R4: the XR chain with `xrCompatible` true and false both deliver `Unavailable` and
    null; the same request without the chain delivers an adapter.
  - R5: a Noop error surface as `compatibleSurface` delivers `Unavailable`; a valid Noop
    surface delivers an adapter for which `wgpuSurfaceGetCapabilities` succeeds.
  - R6/R7: the ranking function over synthetic (type, order) lists — an
    integrated/discrete pair in both enumeration orders under each preference; CPU and
    unknown after both; stability on ties.
- Agent gates: `cargo test -p yawgpu --lib`,
  `cargo test -p yawgpu --test instance_smoke`, clippy.
- Claude: workspace test; real-GPU `e2e_metal_*` surface/instance tests. Multi-adapter
  hosts are needed to observe R5/R6 on hardware; this M2 has one adapter.

## Verification record

Shared gates for Blocks 163–167 (2026-11-28, M2): `cargo test --workspace` 1514/0, with
`yawgpu/vulkan,yawgpu/tiled` 1871/0; fmt clean; clippy clean for default,
`vulkan,tiled` and `metal,tiled`; `yawgpu-hal --lib` 395/0 (`vulkan,tiled`) and 218/0
(`metal,tiled`); all `e2e_metal_*` with `metal,tiled` 176/0; all `e2e_vulkan_*` with
`vulkan,tiled` on MoltenVK under the Khronos layer 174 passed, 1 failed (the known
`imageView2DOn3DImage` portability case; every VUID comes from that test).

`yawgpu --lib` 338/0; `instance_smoke` 5/0; `future_modes` 5/0.
`surface_smoke` on Metal requests its adapter, configures and reaches its frame loop.
R5 and R6 are observable only on a multi-adapter host; that run is owed.

### Windows native Vulkan (2026-11-28)

Run on Windows 11, NVIDIA GeForce RTX 5060 Ti, driver 617.14, Vulkan 1.4.351, at yawgpu
`b22601f`. The Khronos validation layer was on for the HAL ignored, e2e, and example runs.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,513 passed, 0 failed; with `yawgpu/vulkan,yawgpu/tiled` 1,870 passed, 0 failed |
| clippy `-D warnings`, default (workspace), `vulkan,tiled`, `vulkan,shader-passthrough` and `gles` (`yawgpu-core`, `yawgpu-hal`, `yawgpu`, `yawgpu-tint`), all targets | clean |
| Vulkan HAL lib tests, `vulkan,tiled` | 395 passed, 0 failed |
| Vulkan HAL ignored tests, `vulkan,tiled` | 83 passed, 0 failed, 0 VUIDs |
| Vulkan e2e (`e2e_vulkan_*`, `vulkan,tiled,shader-passthrough`, `--ignored`) | 177 passed, 0 failed, 0 VUIDs; `e2e_vulkan_tiled` 6 of 6 |
| C examples on Vulkan (`device_info`, `compute`, `capture`; windowed `hello_triangle`, `triangle`, `triangle_passthrough`, `surface_smoke`, `tiled_msaa`, `tiled_deferred` for 8 s each; `tiled_deferred --verify`) | no error, no VUID; every windowed example configures its surface and presents |

This host exposes a single Vulkan adapter, so R5 and R6 (multi-adapter selection) are not
observable here; that run stays owed.

### Linux native Vulkan, multi-adapter (2026-11-28)

Run on Linux with three Vulkan adapters (NVIDIA GeForce RTX 5060 Ti discrete, AMD RADV
integrated, llvmpipe CPU) at yawgpu `ff6913a`.
`e2e_vulkan_adapter_info` `request_adapter_ranks_by_power_preference` (the loader's own
device types are the oracle) passes: `HighPerformance` selects the discrete adapter,
`LowPower` the integrated one, `Undefined` the first enumerated GPU (the discrete one), and
`forceFallbackAdapter` the CPU adapter. R6 and R7 are observed on hardware.

R5, declining side: `request_adapter_declines_a_surface_no_adapter_can_present_to` (same
file, Linux only, at `cd143c9`) creates a surface from an Xlib source. The Vulkan HAL has
no Xlib/XCB/Wayland surface path (backlog C2), so the surface carries no `VkSurfaceKHR`
and the capability query fails for every adapter. With that surface as
`compatibleSurface` the request completes `Unavailable` with a null adapter under each
power preference, while the same request without it returns an adapter: the filter runs
against all three real adapters and removes them.

Still owed for R5: a host where the filter keeps some adapters and removes others — a
real presentable surface (Windows or macOS; Linux has none until backlog C2) with more
than one adapter. On the hosts run so far a real surface is accepted by the single
adapter (Windows, macOS), which shows the keeping side only.

## Phase Review

Fresh-agent review of Blocks 166–167: 0 CRITICAL, 0 MAJOR, 3 MINOR for this block.
- An empty wait returned `TimedOut` before the timed-wait checks; Dawn runs the checks
  and then returns `Success` — fixed (R2).
- No FFI-level test fails if `powerPreference` is not forwarded to the ranking —
  deferred: Noop enumerates one adapter, so the FFI cannot observe a ranking; the ranking
  and selection functions have direct tests.
- The `conv` test fixture builds `WGPUInstanceImpl` by literal — deferred: the
  constructor is private to `ffi`.
