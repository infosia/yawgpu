# Block 66 — `indirect-first-instance` optional feature

Status: **COMPLETE** — all slices done, CTS-verified on real Metal (draw:arguments
540/180-skip → 720/0), Phase Review clean (no CRITICAL/MAJOR/MINOR). Owner:
Dawn-parity feature backfill.

The WebGPU `indirect-first-instance` optional feature
(`WGPUFeatureName_IndirectFirstInstance = 0x0A`) allows a **non-zero
`firstInstance`** in the arguments of `drawIndirect` / `drawIndexedIndirect`.
Without it, the `firstInstance` field of an indirect draw must be zero (an
app-side content-timeline constraint — the indirect buffer's contents are not
CPU-validatable).

This is the smallest remaining backfill: a pure **capability + device-feature
enable**. No shader changes, no new core validation, no limit interaction:
- The `firstInstance` for an indirect draw comes from the GPU indirect buffer,
  not a CPU argument, so there is nothing to validate at the API and nothing to
  thread through the HAL draw call.
- **Metal** honors `baseInstance` from the indirect buffer natively.
- **Vulkan** honors the indirect `firstInstance` only when the
  `drawIndirectFirstInstance` device feature is enabled; otherwise the driver
  treats it as 0.

CTS: no dedicated validation cases — verified by
`api,operation,rendering,draw:arguments` (currently **180 cases with
`first_instance!=0;indirect=true` skip** with reason "indirect-first-instance
feature is not supported"; advertising the feature un-skips them).

## Public API surface

No new C entry points:
- **`wgpuAdapterGetFeatures` / `wgpuAdapterHasFeature` / device equivalents**
  report `WGPUFeatureName_IndirectFirstInstance` when supported / requested.

### `yawgpu-core::Feature`

Add an `IndirectFirstInstance` variant (not `cfg`-gated); map it C↔Rust in
`yawgpu/src/conv/feature.rs`; advertise from `Adapter::features()` via
`add_indirect_first_instance_feature` consulting
`HalAdapter::supports_indirect_first_instance()`.

## Behaviour contract

### Advertisement (HAL capability query) — Dawn parity

`HalAdapter::supports_indirect_first_instance() -> bool`:

- **Metal** — always `true` (Dawn enables it unconditionally,
  `PhysicalDeviceMTL.mm:712`; Metal indirect draws honor `baseInstance`).
- **Vulkan** — `VkPhysicalDeviceFeatures::drawIndirectFirstInstance == VK_TRUE`
  (Dawn gates the same way, `PhysicalDeviceVk.cpp:315`).
- **Noop** — `true` (no execution; keeps enumeration tests deterministic).
- **GLES** — `false` (Tier 2).

### Vulkan device-feature enable

In `VulkanAdapter::create_device`, enable
`enabled_features.draw_indirect_first_instance = vk::TRUE` whenever
`supported_features.draw_indirect_first_instance == vk::TRUE` (mirrors
`independent_blend` / `depth_clamp`). Without it, a non-zero indirect
`firstInstance` is silently ignored by the driver.

### No core validation

There is no core rule to add: direct `draw`/`drawIndexed` already accept
`firstInstance` (it is a CPU argument and always allowed), and an indirect
draw's `firstInstance` lives in the GPU buffer, which WebGPU does not validate.
The feature purely advertises + unlocks the Vulkan capability. (Dawn defensively
zeroes the indirect `firstInstance` via a GPU validation-encoder pass when the
feature is off; yawgpu does not replicate that — an app that violates the
must-be-zero constraint gets undefined-but-non-crashing behavior, matching the
spec's content-timeline treatment. Document; revisit only if a conformance gap
appears.)

## Slices

1. **Feature plumbing + Vulkan device-feature enable (Noop + HAL cap).**
   `Feature::IndirectFirstInstance`, `add_indirect_first_instance_feature`,
   `HalAdapter::supports_indirect_first_instance` on all four backends, the
   Vulkan `drawIndirectFirstInstance` enable, `conv/feature.rs`, the FFI
   feature-count test. Inline unit tests. **Acceptance:** `cargo test
   --workspace` green on Noop.

2. **Real-GPU verification via CTS.** Confirm the previously-skipped
   `api,operation,rendering,draw:arguments` cases with `first_instance!=0;
   indirect=true` now run and pass on real Metal (and MoltenVK if it advertises
   `drawIndirectFirstInstance`). No new in-repo e2e needed — the CTS draw
   operation suite is the verification. **Acceptance:** those 180 cases pass
   (or are correctly gated on MoltenVK support).

3. **Docs + Phase Review.** README capability note; Block 66 finalization; the
   mandatory no-context Phase Review.

Tracking: `specs/tracking/indirect-first-instance.md`.
