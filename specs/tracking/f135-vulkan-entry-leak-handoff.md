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
