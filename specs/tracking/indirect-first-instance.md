# Tracking — `indirect-first-instance` optional feature

Spec: [Block 66](../blocks/66-indirect-first-instance.md). Goal: Dawn parity for
`WGPUFeatureName_IndirectFirstInstance = 0x0A` on Tier-1 Metal + Vulkan.

## Status

| Slice | Scope | State |
|---|---|---|
| 1 | Feature plumbing + Vulkan device-feature enable (Noop + HAL cap) | **DONE** (2026-07-01) |
| 2 | Real-GPU verification via CTS draw:arguments | **DONE** (2026-07-01) |
| 3 | Docs + Phase Review | TODO |

**Slice 2 — CTS-verified on real Metal.** `api,operation,rendering,draw:arguments`
went from **pass=540 skip=180** → **pass=720 skip=0 fail=0**: the 180
`first_instance!=0;indirect=true` cases (previously skipped "indirect-first-instance
feature is not supported") now run and pass. No in-repo e2e needed.

## Key facts (verified 2026-07-01)

- **Smallest backfill:** pure capability + Vulkan device-feature enable. No
  shader, no core validation, no HAL draw-path change (indirect `firstInstance`
  comes from the GPU buffer; Metal honors `baseInstance` natively).
- Dawn parity: Metal unconditional (`PhysicalDeviceMTL.mm:712`); Vulkan gates on
  `VkPhysicalDeviceFeatures.drawIndirectFirstInstance` (`PhysicalDeviceVk.cpp:315`).
- CTS: no dedicated cases; `api,operation,rendering,draw:arguments` currently
  **skips 180** `first_instance!=0;indirect=true` cases ("indirect-first-instance
  feature is not supported"). Advertising un-skips them.
- Vulkan device-feature enable mirrors `independent_blend` (`vulkan/mod.rs:568/594`)
  / `depth_clamp`. ash field: `VkPhysicalDeviceFeatures.draw_indirect_first_instance`.
- Template: `depth-clip-control` / `float32-blendable` (capability-only), minus
  the validation gate.
