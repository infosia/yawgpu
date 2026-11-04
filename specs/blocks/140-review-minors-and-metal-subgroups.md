# Block 140 — Review MINORs and Metal subgroup parity

Status: **COMPLETE (2026-11-04)**. Phase Review: 0 CRITICAL, 1 MAJOR, 5 MINOR; all resolved,
see below.
Source: backlog E12 and E7, started by user request on 2026-11-04.
Oracle: Dawn at the pinned `third_party/dawn` (paths relative to `src/dawn/native/`):
`metal/PhysicalDeviceMTL.mm` (subgroup sizes at lines 322-330, `SubgroupSizeControl` at
lines 762-773), `metal/TextureMTL.mm` (storage mode, lines 236-242), and
`Adapter.cpp` (`APIGetInfo` subgroup sizes).

## Problem

### E12 — remaining review MINORs

E12 groups the MINORs of the 2026-10 review, which `specs/tracking/code-review-2026-10.md`
lists by name only, and Block 120 m5. Each was checked against the code at `e3a2df9`:

| Item | State at `e3a2df9` | Here |
|---|---|---|
| `wgpuInstanceWaitAny` busy-spin | Open. With a timeout, the loop calls `std::thread::yield_now` until a future completes or the timeout ends, so a waiting thread keeps one core busy (`yawgpu/src/ffi/instance.rs`, the `wait_any` loop) | R1 |
| Block 120 m5: `yawgpu-tint` workaround structs are not `#[non_exhaustive]` | Already done by Block 131 (`f3a63e9`): both structs are `#[non_exhaustive]`, and the construction sites outside the crate use `Default` and field assignment | R2 (no change) |
| `destroy()` does not free GPU memory | Open. `Buffer::destroy` clears the map state and the host bytes, but the HAL buffer lives until the last `Arc` drops. Dawn frees the allocation at destroy, after the GPU is done with it. It needs the HAL resource to become optional across core, so it gets its own block | New backlog A24 |
| Metal `StorageModeShared` textures | Open. Every single-sample, non-transient texture is `MTLStorageModeShared` (`yawgpu-hal/src/metal/texture.rs`). Dawn uses `Private`. On Apple GPUs that costs lossless compression; on Intel and AMD Macs some formats, depth among them, may not allow `Shared`. Changing it touches every CPU path into textures, so it gets its own block | New backlog A25 |
| Vulkan per-submit object churn | Fixed by Block 117 | — |
| Per-draw validation redundancy | A performance item, part of E2 | — |
| Tint shim exception safety | No action. Tint and the shim build with `-fno-exceptions`, so no C++ exception can cross the C ABI; an allocation failure aborts, as it does in Dawn | — |
| `build.rs` rerun triggers | Covered by Blocks 98 and 139 | — |

### E7 — Metal subgroups

- yawgpu reports a subgroup size range of 32 to 32 on every Metal device
  (`yawgpu-hal/src/metal/mod.rs`, `subgroup_size_range`). Dawn reports 32 to 32 on Apple
  GPUs and 4 to 64 on the others ("4 comes from the minimum derivative group", "a
  ballot is a uint64").
- yawgpu never advertises `subgroup-size-control` on Metal; a comment there says Dawn
  does not support it on Metal. That is wrong. Dawn enables it when the minimum and
  maximum subgroup sizes are equal, that is on Apple GPUs. A C probe on the M2 confirms
  it: Dawn's adapter lists feature `0x17` (`SubgroupSizeControl`); yawgpu's does not.
- The `shader-f16` review left two MINORs: the five 16-bit flags in `VulkanDeviceInner`
  are written and never read, and the float16 support is queried twice (adapter
  advertisement and device creation).

The same probe found two more standard features that Dawn lists on the M2 and yawgpu
does not: `TextureCompressionUnaligned` (`0x18`) and `AtomicVec2uMinMax` (`0x19`). They are
new backlog item B8, not this block.

## Rules

R1. **`wgpuInstanceWaitAny` does not busy-spin.** With a non-zero timeout and no
completed future, the waiting thread first yields for a short time, then sleeps. Each
sleep doubles, from about 50 µs up to 1 ms, and never runs past the deadline. Results,
statuses, the zero-timeout path, and `timedWaitAnyEnable` validation do not change.
Dawn blocks on the queue fence instead. yawgpu's HAL has no timed wait, so this
bounded backoff is the accepted difference: a wait that ends after the backoff has
started may return up to 1 ms late.

R2. **Workaround structs are `#[non_exhaustive]`.** Already true at `f3a63e9`; kept as a rule so it stays true. The Rust-side MSL and SPIR-V
workaround structs in `yawgpu-tint` get `#[non_exhaustive]`. Code outside the crate
builds them from `Default` and assigns fields. The `#[repr(C)]` layout, the ABI layout
assertions, and the generated code do not change.

R3. **Metal subgroup size range follows Dawn.** Apple GPUs report 32 to 32. Other Metal
GPUs report 4 to 64. "Apple" is the vendor id that `WGPUAdapterInfo` reports (Block 135),
so iOS, whose vendor id is 0 as in Dawn, reports 4 to 64 and no `subgroup-size-control`.
This covers `WGPUAdapterInfo`'s `subgroupMinSize` and `subgroupMaxSize`, and every other
use of the range.

R4. **`subgroup-size-control` on Metal.** When `subgroups` is supported and the minimum
and maximum sizes are equal, Metal advertises `subgroup-size-control`. Its caps are
min = max = that size. The caps' `max_compute_workgroup_subgroups` is the device's
maximum threads per threadgroup divided by that size, so it never rejects a pipeline
that Dawn accepts. A compute pipeline with `@subgroup_size(32)` compiles and runs.
Tint's MSL output needs no option for it, because the size is fixed. Core validation
(Block 108) applies unchanged.

R5. **`shader-f16` MINORs.** The 16-bit flags in `VulkanDeviceInner` that nothing reads
are removed, together with their `Debug` fields. Keep any flag that has a reader, and
name it in the report. The adapter's float16 support is queried once and reused at
device creation, if that does not need a wider change; otherwise the coding agent
reports why and leaves it.

R6. **Unchanged.** Every other backend and every other feature.

## Tests

- Unit (FFI, Noop): `WaitAny` with a timeout and a future that never completes returns
  `TimedOut` no earlier than the timeout and no later than the timeout plus a small
  bound. The sleep schedule is a pure function with its own test: it starts small,
  doubles, is capped at 1 ms, and never passes the deadline.
- Unit (`yawgpu-tint`, core): the structs are built through `Default` and field
  assignment; the existing layout and field-mapping tests still pass.
- Unit (HAL, Metal, pure): the range is 32/32 for Apple and 4/64 for others; the
  `subgroup-size-control` decision for equal and unequal ranges, and the caps values.
- Metal `#[ignore]` (Claude, M2): the adapter advertises `subgroup-size-control`, and the
  caps are 32/32.
- e2e (Claude): a Metal test where a `@subgroup_size(32)` compute pipeline writes
  `subgroup_size` and `subgroup_invocation_id` to a buffer, read back as 32 and
  0..31.
- Claude: the feature probe matches Dawn for `0x17`. CTS on Metal: the subgroup queries
  (`webgpu:shader,execution,expression,call,builtin,subgroup*`,
  `webgpu:api,validation,*subgroup*`), then `webgpu:api,*` and
  `webgpu:shader,execution,*` must not regress.

## Out of scope

- A24 (free GPU memory at `destroy`), A25 (Metal texture storage mode), B8
  (`TextureCompressionUnaligned`, `AtomicVec2uMinMax`).
- A timed HAL wait for `WaitAny`.

## Slices

- **S1 (coding agent):** R1-R6 with the unit tests.
- **S2 (Claude):** Metal ignored test and e2e, probe, CTS, Phase Review.

## Verification (2026-11-04)

Run on Apple M2, macOS 27.2. Vulkan here is MoltenVK under the Khronos layer, which is
reference-only.

| Gate | Result |
|---|---|
| `cargo test --workspace` | 1,405 passed, 0 failed |
| fmt; clippy `-D warnings` on default, `metal`, `vulkan`, `gles`, `metal,tiled`, `vulkan,tiled`, `shader-passthrough` (host, all targets), and `metal` for `aarch64-apple-ios` and `aarch64-apple-ios-sim` | clean |
| HAL `apple_gpu_advertises_fixed_subgroup_size_control`, M2 | range (32, 32); caps 32 / 32 / 32 |
| `e2e_metal_subgroup_size_control` (new) and `e2e_metal_subgroups` | 3 of 3 and 3 of 3. `@subgroup_size(32)` runs at width 32; each `subgroup_invocation_id` in `0..32` occurs once per subgroup |
| `e2e_vulkan_subgroup_size_control` (now a wrapper over the shared scenarios) and `e2e_vulkan_subgroups`, MoltenVK | 3 of 3 and 3 of 3, 0 VUIDs |
| Feature-list probe against Dawn `v20261030`, Metal | `SubgroupSizeControl` (`0x17`) now listed by both. The only standard features still listed by Dawn alone are `0x18` and `0x19` (B8) |
| Metal e2e and every other ignored `yawgpu` test, `metal,tiled` | 165 passed, 0 failed |
| webgpu-native-cts, Metal, `webgpu:api,*` | pass 586,612, skip 104,755, fail 0, crash 0 (Block 139: 586,611 / 104,756) |
| webgpu-native-cts, Metal, `webgpu:shader,execution,*` | pass 853,793, skip 29,460, fail 0, crash 0 (Block 139: 853,787 / 29,466) |

The CTS and the full e2e ran on the library before the Phase Review fixes. Those fixes
change iOS only (vendor 0), revert test-only edits, and widen a test bound.

## Phase Review

| ID | Finding | Disposition |
|---|---|---|
| M1 | The subgroup range used the device name on iOS, where `WGPUAdapterInfo` and Dawn use vendor 0, so iOS reported 32/32 and `subgroup-size-control` while Dawn reports 4/64 without it | Fixed: one `vendor_id()` helper feeds both; R3 amended |
| m1 | R2 was already done by Block 131; the diff only rewrote in-crate tests and added a module-wide clippy allow | Reverted; the spec row says so |
| m2 | The `WaitAny` timeout test allowed 50 ms of overshoot | 500 ms |
| m3 | The e2e did not read back `subgroup_invocation_id` as the Tests section asks | Added to the shared scenario |
| m4 | The Metal `subgroups` gate is `Apple6 || Metal3`; Dawn's is `Apple6 || Mac2`, and Dawn reports the range even without subgroups | Outside this block: backlog A26 |
| m5 | Block 108 and `tracking/subgroups.md` still said Metal never advertises the feature | Updated |

