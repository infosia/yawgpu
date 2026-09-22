# Block 99 — Feature advertisement follows Dawn's device queries

Status: **IMPLEMENTED (2026-09-22)** — HAL in `6783728`; real-device inline tests green on M2 Metal + MoltenVK; C-ABI e2e (`e2e_metal_feature_queries.rs`, `e2e_vulkan_feature_queries.rs`) green against direct Metal / `ash` oracles. CTS re-confirm pending (S4 of Block 102). Backlog item **A4** in
`specs/tracking/backlog.md`. Generalises Block 72 (which fixed
`texture-formats-tier1/2`) to the remaining `supports_*` entries that
still return a literal where Dawn queries the device.

## Problem

`HalAdapter::supports_*` is the single source for `Adapter::features()`
(`yawgpu-core/src/adapter.rs` → `add_hal_optional_features`). Several
Tier-1 entries are unconditional `true` although Dawn's physical-device
code gates them on a query. On hardware that lacks the capability the
feature is advertised, the core validation accepts the request, and the
failure surfaces late (driver error, wrong results, or — for
`timestamp-query` — nothing at all). The dev machines all have the
capability, so neither the e2e suite nor the CTS sees it.

The Dawn rules (pinned submodule, `third_party/dawn/src/dawn/native/`):

| Feature | Dawn Metal (`metal/PhysicalDeviceMTL.mm`) | Dawn Vulkan (`vulkan/PhysicalDeviceVk.cpp`) |
|---|---|---|
| `timestamp-query` | `IsGPUCounterSupported(device, MTLCommonCounterSetTimestamp, {MTLCommonCounterTimestamp})`: the device's `counterSets` contains a set named `timestamp` (case-insensitive) that lists the `timestamp` counter, **and** the device supports counter sampling at a command boundary (draw ∧ dispatch ∧ blit) **or** at the stage boundary | `properties.limits.timestampComputeAndGraphics == VK_TRUE` |
| `float32-filterable` | `[device supports32BitFloatFiltering]` (macOS 11 / iOS 14 SDK) | `R32_SFLOAT`, `R32G32_SFLOAT`, `R32G32B32A32_SFLOAT` optimal-tiling features all contain `SAMPLED_IMAGE_FILTER_LINEAR` |
| `depth32float-stencil8` | unconditional | `D32_SFLOAT_S8_UINT` optimal-tiling features contain `DEPTH_STENCIL_ATTACHMENT` |
| `rg11b10ufloat-renderable` | unconditional | `B10G11R11_UFLOAT_PACK32` optimal-tiling features contain `COLOR_ATTACHMENT ∣ COLOR_ATTACHMENT_BLEND` |
| `bgra8unorm-storage` | unconditional | `B8G8R8A8_UNORM` optimal-tiling features contain `STORAGE_IMAGE` |
| `texture-component-swizzle` | family query (already done) | unconditional |
| `depth-clip-control`, `float32-blendable`, `dual-source-blending`, `clip-distances`, `indirect-first-instance`, `shader-f16`, `texture-formats-tier1` | unconditional | already queried in yawgpu |

GLES: `DEPTH32F_STENCIL8` is a core ES 3.0 sized internal format, so the
GLES `supports_depth32float_stencil8() == true` is correct on the ES 3.1
baseline; Dawn's GL backend has no gate either. Unchanged.

## Behaviour contract

### R1 — Metal `supports_timestamp_query`

Returns `true` iff **both** hold:

- (a) `device.counterSets()` is `Some` and contains a set whose `name()`
  equals `MTLCommonCounterSetTimestamp` (compare case-insensitively, as
  Dawn does) and whose `counters()` contains a counter whose `name()`
  equals `MTLCommonCounterTimestamp` (case-insensitive);
- (b) `device.supportsCounterSampling(AtStageBoundary)` **or**
  (`AtDrawBoundary` ∧ `AtDispatchBoundary` ∧ `AtBlitBoundary`).

The two halves are separate `pub(super)` helpers so each is unit-testable:
`metal_device_has_timestamp_counter_set(&device) -> bool` and
`metal_device_supports_counter_sampling(&device) -> bool`
(the latter is the exact Dawn disjunction and is reused by Block 102 —
timestamp-query execution — to pick stage- vs command-boundary sampling).
Cache the result on `MetalAdapter` at construction (one query, like
`read_write_texture_tier`).

### R2 — Metal `supports_float32_filterable`

Returns `device.supports32BitFloatFiltering()`. Cached at construction.
No `Mac2` fallback branch: yawgpu's minimum deployment target already
includes the selector.

### R3 — Vulkan `supports_timestamp_query`

Returns `physical_device_properties().limits.timestamp_compute_and_graphics == vk::TRUE`.

### R4 — Vulkan format-property gates

Add a private helper
`optimal_tiling_features(&self, format: vk::Format) -> vk::FormatFeatureFlags`
(wraps `get_physical_device_format_properties`) and express:

- `supports_depth32float_stencil8`: `optimal(D32_SFLOAT_S8_UINT) ⊇ DEPTH_STENCIL_ATTACHMENT`
- `supports_rg11b10ufloat_renderable`: `optimal(B10G11R11_UFLOAT_PACK32) ⊇ COLOR_ATTACHMENT | COLOR_ATTACHMENT_BLEND`
- `supports_bgra8unorm_storage`: `optimal(B8G8R8A8_UNORM) ⊇ STORAGE_IMAGE`
- `supports_float32_filterable`: for each of `R32_SFLOAT`, `R32G32_SFLOAT`, `R32G32B32A32_SFLOAT`: `optimal(f) ⊇ SAMPLED_IMAGE_FILTER_LINEAR`

The set-membership decision for each rule is a pure function of the
flags (`fn rg11b10_renderable_from_flags(flags) -> bool` etc., or one
generic `flags.contains(required)` call site) so it is unit-testable on
Noop without a Vulkan device — same shape as
`vulkan_supports_shader_float16_requires_extension_and_feature`.

### R5 — Everything else unchanged

The unconditional Metal entries listed as "unconditional" in the table
stay literal, but each gets a doc comment stating that Dawn enables the
feature unconditionally on Metal (so the literal is a deliberate mirror,
not an omission). `VulkanAdapter::supports_texture_component_swizzle`
likewise. Noop and GLES tables unchanged. No core change: the
`Feature` insertion order in `add_hal_optional_features` and every
downstream validation rule are untouched.

### R6 — Device creation is consistent with advertisement

Nothing at `create_device` depends on these six entries today (no
`VkPhysicalDeviceFeatures` bit is involved; `timestampComputeAndGraphics`
is a limit, not a feature). Verify by reading, note in the report; do
not add enablement code.

## Tests

Per CLAUDE.md principle 1 every changed `pub fn` has an inline test.

- **Noop / pure-function tests (`cargo test -p yawgpu-hal --features vulkan --lib` compiles them; they must not need a device):**
  - each R4 predicate with flags that satisfy / do not satisfy the rule;
  - the R3 predicate with `vk::TRUE` / `vk::FALSE`;
  - the R1(b) disjunction as a pure function of the four boolean
    sampling-point answers: stage-only → true; draw+dispatch+blit → true;
    draw+dispatch without blit → false; none → false.
- **Real-GPU inline tests (`#[ignore = "manual real Metal backend test"]` / Vulkan analogue), run by Claude:**
  - Metal: `supports_timestamp_query()` equals a fresh recomputation from
    the device (same shape as
    `metal_adapter_texture_formats_tier2_matches_cached_device_query`);
    `supports_float32_filterable()` equals
    `device.supports32BitFloatFiltering()`.
  - Vulkan: each of the five gated entries equals a fresh
    `get_physical_device_format_properties` /
    `get_physical_device_properties` evaluation on the adapter's
    physical device.
- **Real-GPU e2e (`yawgpu/tests/e2e_metal_feature_queries.rs`,
  `yawgpu/tests/e2e_vulkan_feature_queries.rs`, Claude authors and runs):**
  `wgpuAdapterHasFeature` for each gated feature equals the oracle
  computed directly from Metal (`MTLCreateSystemDefaultDevice`) /
  Vulkan (`ash` instance + first physical device), mirroring
  `e2e_metal_texture_formats_tier2.rs`.
- **CTS:** run `webgpu:api,validation,capability_checks,features,*` and
  `webgpu:api,operation,adapter,requestDevice:*` on Metal and MoltenVK;
  expect byte-identical pass/skip counts to the pre-change baseline on
  this M2 (every gated feature is present here, so the advertised set
  must not move).

## Out of scope

- Implementing `timestamp-query` execution (Block 102 / backlog A1).
- Metal `ChromiumExperimentalTimestampQueryInsidePasses` (Dawn-only).
- GLES feature table (Tier 2, catalogued in Block 67).
