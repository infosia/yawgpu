# HANDOFF — F-135: share the Vulkan `ash::Entry` process-wide (device-creation churn leak)

**Spec:** `specs/blocks/60-real-backends.md` → "CTS finding F-135 — Vulkan entry
loaded per `VulkanInstance` → device-creation churn leak (2026-06-23)". Read it
first; this handoff is the implementation contract.

**Upstream finding:** webgpu-native-cts `docs/FINDINGS.md` F-135 /
`specs/investigate-yawgpu-device-create-leak.md` (in the webgpu-native-cts repo).

## Problem (one line)

`VulkanInstance::new` calls `ash::Entry::load()` on every `WGPUInstance`, and the
entry is owned per-instance, so each instance lifecycle does a full
`LoadLibrary`/`FreeLibrary` of `vulkan-1.dll`. The Windows loader/ICD leaks
per-load state; after ~150 instance/device-creation cycles in one process
`wgpuRequestDevice` fails with `HAL device creation failed: vulkan` (or
`0xC0000005`).

## Required change (`yawgpu-hal/src/vulkan/mod.rs`)

1. **Process-global entry.** Add a `static VULKAN_ENTRY: OnceLock<ash::Entry>`
   (use `std::sync::OnceLock`). Load the entry exactly once:

   ```rust
   fn shared_entry() -> Result<&'static ash::Entry, HalError> {
       VULKAN_ENTRY
           .get_or_try_init(|| unsafe { ash::Entry::load() })
           .map_err(|_| HalError::BackendUnavailable { backend: BACKEND })
   }
   ```

   (`OnceLock::get_or_try_init` is stable as of the project's MSRV — if it is
   not, use `get_or_init` returning an `Option` and map `None`/poisoned to
   `BackendUnavailable`, or guard the load in a small helper. Do **not**
   introduce a new dependency for this.)

2. **`VulkanInstance::new`** uses `shared_entry()?` instead of
   `ash::Entry::load()`. Behaviour on load failure is unchanged
   (`HalError::BackendUnavailable { backend: "vulkan" }`).

3. **`VulkanInstanceInner._entry`** must reference the shared entry, NOT own a
   freshly-loaded one. Change the field type from `ash::Entry` to a
   `&'static ash::Entry` (cleanest, since the entry now lives for the process)
   — or `ash::Entry` cloned from the shared one **only if** clone does not
   re-`LoadLibrary` (verify against ash 0.38: `Entry: Clone` duplicates the
   function tables, not the library handle — confirm before relying on it). The
   `&'static` form is the safe default and avoids the question entirely. Update
   every read of `self.inner._entry` (e.g. the `metal_surface` / `win32_surface`
   loader construction at `mod.rs:110,150`) to the new type — these take
   `&ash::Entry`, so a `&'static ash::Entry` works directly.

4. **`impl Drop for VulkanInstanceInner`** keeps `self.instance.destroy_instance(None)`
   and must **no longer** drop/unload the entry. With the `&'static` field there is
   nothing to drop; the `vulkan-1.dll` handle intentionally stays resident for the
   process lifetime (matches wgpu-hal / Dawn).

5. No change to `create_device`, extension/feature selection, the minimum Vulkan
   version, or any other Drop impl.

## Tests

Add Vulkan-gated inline `#[cfg(test)]` unit tests in `vulkan/mod.rs` (gated like
the existing `vulkan_adapter_create_device_returns_zero_allocation_device` test,
which only runs when a real Vulkan device is present — keep that gating so Noop CI
is unaffected):

- **Shared entry identity:** two `VulkanInstance::new()` calls resolve to the same
  underlying entry (assert the `shared_entry()` pointer is stable, or that a second
  `new()` does not re-load — e.g. `std::ptr::eq` on the `&'static` entry).
- **Churn survives:** loop `N` (≥160, comfortably past the ~150 ceiling) times:
  `VulkanInstance::new()` → `enumerate_adapters()` → `adapter.create_device()` →
  drop all. Assert `create_device` succeeds on every iteration. This is the
  regression guard for F-135 — it must FAIL on the current `ash::Entry::load()`
  -per-instance code and PASS after the fix, on the Windows native-Vulkan host.

(Per CLAUDE.md principle 1, the new/changed public path ships with its unit test
in the same commit.)

## Standalone repro (verification, not committed)

Before/after proof decoupled from CTS, per the spec's acceptance criteria. A
throwaway is fine — either:

- a `#[ignore]` Vulkan e2e test under `yawgpu/tests/e2e_vulkan_*.rs` that runs the
  ×700 `createInstance→adapter→device→release` loop and asserts no failure, or
- a scratchpad `cl.exe`-linked C program against `yawgpu.dll` doing the same
  (need not be committed).

Expected: **before** the fix, first failure ≤ ~150 iterations; **after**, 700
iterations clean. Record the numbers in the eventual commit message.

## Acceptance

- [ ] `ash::Entry` loaded once per process; `VulkanInstanceInner` no longer owns a
      per-instance entry; instance `Drop` only destroys the `VkInstance`.
- [ ] Churn unit test (≥160 create/destroy cycles) passes on Windows native Vulkan;
      Noop CI unaffected (test is GPU-gated).
- [ ] `cargo build`/`cargo test` green on Noop (no real GPU required for CI).
- [ ] `cargo clippy -- -D warnings` clean (missing-docs etc.).
- [ ] Standalone ×700 churn repro: fail before, clean after — numbers recorded.
- [ ] Commit references `F-135`; matches existing message style
      (`fix(F-135): ...`).

## Out of scope

- CTS-side carry (`--isolate` for device-churning families) — lives in
  webgpu-native-cts, already authoritative; do not touch `expectations/`.
- `CTS_DEVICE_RECYCLE_INTERVAL` tuning — wrong layer, confirmed dead end.
- D3D / GLES paths.

---

## ROUND 2 — the entry fix is REAL but INSUFFICIENT (verified on Windows native Vulkan, 2026-06-23)

The `ash::Entry` process-wide share (commit `b71e59c`) landed and its HAL churn unit
test passes, BUT it does **not** close F-135 at the CTS level. End-to-end re-verify
(rebuilt `yawgpu.dll` deployed next to `cts.exe`):

- `api,validation,capability_checks,limits,maxStorageBuffersPerShaderStage:*`
  single-process, `CTS_DEVICE_RECYCLE_INTERVAL=0`, `--workers 1`: **still fail=318**,
  all `requestDevice failed: HAL device creation failed: vulkan` (unchanged from
  pre-fix). `--isolate` = fail=0.

There is a **second, distinct leak**. Bisected by subtest:
- `createBindGroupLayout` (device+limits, no pipeline) → **fail=0** (clean)
- `createPipelineLayout` → **fail=0** (clean)
- `createPipeline` → **fail=48** (`HAL device creation failed`, onset ~case #120)

So the remaining leak is **specific to the createPipeline path**, and it is a
**device-accumulation** leak (the error is *device* creation failing after ~120
pipeline-creating cases — a leaked pipeline/handle keeps its `Arc<core::Device>` alive,
so VkDevices accrue until creation fails).

Layer bisect (what it is NOT):
- **HAL is clean.** A HAL-level churn (`VulkanInstance::new → adapter → create_device →
  create_compute_pipeline → drop all`) ×200 PASSES. So `yawgpu-hal` device/pipeline Drop
  is correct — the leak is **above** the HAL, in `yawgpu/src/ffi` + `yawgpu-core`.
- **Not the pipeline caches.** `WGPUDeviceImpl::{compute,render}_pipeline_cache` hold
  `Weak<…>` (ffi/mod.rs:173-176) — they do not retain pipelines.
- **Not CTS cleanup.** CTS tracks + releases pipelines in `GpuTest::finalize()`
  (`webgpu-native-cts/src/common/harness.cpp:450`); device+BGL churn proves CTS device
  release works.
- **Not core retention.** `core::Device::create_compute_pipeline` returns the pipeline;
  it does not stash a strong ref (`yawgpu-core/src/device.rs:526`).

Leading hypothesis (UNCONFIRMED) — the **async** path. The failing subcases skew
`async=true` (3 of 4 fail-groups). `wgpuDeviceCreateComputePipelineAsync` registers a
`PendingCallback` on `WGPUInstanceImpl.pending_callbacks`; if a validation test never
pumps `wgpuInstanceProcessEvents`, those callbacks pile up. If a pending callback retains
the device (or forms an instance↔callback cycle), the device never drops → VkDevice leak.
Verify: (a) does the async create path hold the device/pipeline in the pending callback,
and is it dropped if the future is abandoned at instance teardown? (b) instrument a
strong-count / live-object leak counter on Noop across a create-async-pipeline →
release-pipeline → release-device sequence WITHOUT processEvents.

Next steps for whoever continues:
1. Add an **FFI-on-Vulkan** (not Noop) churn repro or a Noop refcount-leak assertion for
   the create-compute-pipeline (sync AND async) → release-pipeline → release-device
   sequence. This is the missing regression guard (the Noop `noop_chain()` tests can't
   exhaust a real VkDevice, and the HAL test is below the leak).
2. Find the leaked strong `Arc<core::Device>` (or `Arc<WGPUDeviceImpl>`) on the
   createPipeline FFI/core path; fix so `wgpuComputePipelineRelease` + `wgpuDeviceRelease`
   drop the device to 0. Re-verify the CTS repro goes 318 → 0.

Upstream tracking updated: webgpu-native-cts `docs/FINDINGS.md` F-135 +
`specs/investigate-yawgpu-device-create-leak.md` (entry fix recorded as partial).
