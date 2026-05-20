# Phase 10 — Phase Review (mandatory)

Status: in progress. Per `../reference/workflow.md` ("Phase
Review"): a fresh no-context subagent reviewed the cumulative
Phase 10 diff (`0be8fca..HEAD` — from after the Phase 9
review-complete commit through the P10.4e C-FFI-100%-coverage
commit, covering P10.0/P10.1a/b + P10.2 + P10.3a/b/c/d/e/f +
P10.4a/b/c/d/e) and emitted findings; CRITICAL/MAJOR must be
fixed before Phase 10 is COMPLETE.

## Review headline

**Phase 10 review: 0 critical, 3 major, 4 minor.**

**Recomputed coverage stats** (independently verified):
- yawgpu-hal (Noop 14 + lib 25 + Metal 25 + Vulkan 22):
  86 pub fn, **85 tested directly + 1 deferred**
  (`VulkanSurface::configure` — see K2).
- yawgpu/src/conv.rs: 65 pub fn, **65 tested directly**;
  but two `pub unsafe fn`s panic on null without
  `#[should_panic]` coverage of the panic branch
  (see K3).
- yawgpu-core (post-narrow): 183 pub fn, **183 tested
  directly**. Audit/coverage label inconsistencies recorded
  as m4.
- yawgpu C FFI (`yawgpu/src/lib.rs`): 169
  `pub unsafe extern "C" fn` **all tested directly**, BUT
  there are 5 additional non-extern `pub fn` items
  (inherent-impl methods + `testing_*` helpers) that are
  **part of the public API surface and lack direct tests**
  (see K1).

Workspace gate (Noop): **492 passed / 0 failed / 10 ignored**
(58 binaries). Real-GPU `--ignored` sweeps: 47 Metal + 42
Vulkan all pass. Clippy clean under default + `--features
metal` + `--features vulkan`.

## MAJOR findings — must fix

### K1 — Five `pub` items in `yawgpu/src/lib.rs` lack direct unit tests
Block-90 scope explicitly covers "every `pub unsafe extern
"C" fn wgpu*` AND every `pub fn` / `pub const` exported from
`yawgpu/src/lib.rs` or its modules". Five non-extern pub fns
are public per Rust visibility and have no inline test that
directly calls them with an assertion:
- `WGPUDeviceImpl::set_uncaptured_error_callback`
  (`yawgpu/src/lib.rs:535`) — only forwarded to from
  `testing_set_uncaptured_error_callback`; no direct unit test.
- `WGPUDeviceImpl::dispatch_error`
  (`yawgpu/src/lib.rs:543`) — only forwarded to from
  `testing_dispatch_device_error`; no direct unit test.
- `testing_set_uncaptured_error_callback`
  (`yawgpu/src/lib.rs:5212`) — used only by integration
  tests in `yawgpu/tests/`; block-90 indirect coverage does
  **not** satisfy the rule.
- `testing_dispatch_device_error`
  (`yawgpu/src/lib.rs:5227`) — same as above.
- `testing_bind_group_layout_entry_visibility`
  (`yawgpu/src/lib.rs:5241`) — same as above.

The two siblings `testing_get_device_label` /
`testing_get_queue_label` ARE directly called by
`wgpuDeviceSetLabel_..._pin_noop_device` (file lines 6536,
6750) and satisfy the rule.

**Fix:** add 5 inline tests to
`yawgpu/src/lib.rs#[cfg(test)] mod tests`, build a Noop
device handle via `noop_chain` helper, exercise each fn
directly, assert observable side-effect. Add coverage rows
to `phase-10-coverage.md`.

### K2 — `VulkanSurface::configure` lacks a direct unit test
`yawgpu-hal/src/vulkan/mod.rs:383` (impl) +
`:3081-3093` (deferral comment) +
`phase-10-coverage.md:321` (coverage row).

Currently marked "(deferred — e2e-covered; see follow-up
note above)". Block-90 has **no carve-out for "deferred to
e2e"**. The deferral comment explains the SIGSEGV risk on a
synthesized `vk::SurfaceKHR::null()` and proposes a
defensive null-handle pre-check as the closure. That fix
was not implemented in Phase 10.

**Fix (minimal):** add the null-handle pre-check in
`VulkanSurface::configure` (return `Err(HalError::
SwapchainCreationFailed { backend: "vulkan", message:
"surface is null" })` when `self.surface == vk::SurfaceKHR
::null()`). Add the corresponding
`vulkan_surface_configure_errors_for_null_surface` test
(now safe, no UB), update the coverage row from "(deferred
...)" to the new test name.

### K3 — Missing `#[should_panic]` tests for null-handle FFI panic contracts
`yawgpu/src/conv.rs:28-34` (`release_handle`) +
`:42-49` (`add_ref_handle`).

Both are `pub unsafe fn` that unconditionally panic on null
input with `"{name} must not be null"` (per CLAUDE.md core
principle #3 FFI-boundary contract). Their existing tests
`release_handle_drops_owned_reference_once` and
`add_ref_handle_increments_refcount_for_later_release` only
exercise the happy path. The P10.2 handoff explicitly
listed null-panic contracts as expected; the sibling fns
`clone_handle` and `borrow_handle` got
`#[should_panic(expected = "... must not be null")]` tests
(lines 1871, 1877) but Release/AddRef did not.

**Fix:** add two `#[should_panic(expected = "... must not
be null")]` tests mirroring the existing
`clone_handle_null_panics_with_contract_message` shape.
Update `phase-10-coverage.md` conv.rs rows for
`release_handle` / `add_ref_handle` to list the new tests
alongside the happy-path tests.

## MINOR findings — deferred (fold into the same close-out commit or skip)

### m1 — `phase-10-coverage.md` Encoder+Pass heading says 48 but table has 49 rows
The C-FFI total 16 + 32 + 26 + 49 + 46 = 169 is correct;
just a stale label. **Fix:** change heading to `(49 pub fn)`.

### m2 — `phase-10-coverage.md` conv.rs heading says 66 but file has 65 pub fn
The 65-row table is correct. **Fix:** change heading to
`(65 pub fn)`.

### m3 — `pub const` exports have no coverage rows
5 `pub const` items in `yawgpu/src/lib.rs`
(`WGPU_STRLEN`, `WGPU_YAWGPU_INSTANCE_BACKEND_{NOOP,METAL,
VULKAN}`, `WGPU_STYPE_YAWGPU_INSTANCE_BACKEND_SELECT`) are
named by block-90 but not in the coverage table. They are
literal values; spirit of "trivial getters" exclusion
arguably applies, but the policy text says `pub const`
explicitly. **Fix:** either add a "public constants"
micro-section to `phase-10-coverage.md` (each is read by
the helpers / cross-section tests) or document the
trivial-constant exclusion at the top of the file.

### m4 — `phase-10-audit.md` "Surface / utilities: 2" overcounts by 1
Subgroup sums in the audit `Pipeline / Shader: 17` + audit
`Surface / utilities: 2` = 19, but `phase-10-coverage.md`
says `Pipeline / Shader: 18` + Surface absorbed = 18. The
truth is 18 + 1 = 19 (Pipeline/Shader has 18, Surface has
1 — `Instance::create_surface_from_metal_layer`).
**Fix:** update `phase-10-audit.md` "Kept Public
Distribution" to `Pipeline / Shader: 18` and
`Surface / utilities: 1`.

### m5 — `tracking/phase-10-review.md` was missing before this review
Not a Phase 10 production-code finding; doc-lifecycle. This
file (the one you are reading) closes the gap.

## Fix log

- K1 fixed in `yawgpu/src/lib.rs` and `specs/tracking/phase-10-coverage.md`:
  added direct inline tests
  `WGPUDeviceImpl_set_uncaptured_error_callback_records_callback_for_dispatch`,
  `WGPUDeviceImpl_dispatch_error_routes_to_uncaptured_callback`,
  `testing_set_uncaptured_error_callback_installs_callback_for_dispatch`,
  `testing_dispatch_device_error_routes_to_uncaptured_callback`, and
  `testing_bind_group_layout_entry_visibility_returns_entry_visibility_and_none`.
- K2 fixed in `yawgpu-hal/src/vulkan/mod.rs` and
  `specs/tracking/phase-10-coverage.md`: added the
  `VulkanSurface::configure` null-surface guard, restored `surface_config()`,
  and added `vulkan_surface_configure_errors_for_null_surface`.
- K3 fixed in `yawgpu/src/conv.rs` and
  `specs/tracking/phase-10-coverage.md`: added
  `release_handle_null_panics_with_contract_message` and
  `add_ref_handle_null_panics_with_contract_message`.
- m1/m2 fixed in `specs/tracking/phase-10-coverage.md`; m3 documented in the
  public-constants subsection; m4 fixed in `specs/tracking/phase-10-audit.md`.
- Verification: `cargo test --workspace` passed. Current workspace test-list
  count is 509 tests, +7 from the pre-fix current baseline of 502. `cargo test
  -p yawgpu-hal --features vulkan -- --ignored` discovered 22 ignored Vulkan
  tests, +1 from the reviewed 21-test baseline, but this sandbox host failed
  all Vulkan ignored tests before assertions with `VK_ERROR_INCOMPATIBLE_DRIVER`
  / `DeviceCreationFailed { backend: "vulkan" }`.

## Status

K1–K3 fixes are applied, including m1–m5 close-out notes. Noop/default
verification is green; the real Vulkan ignored sweep needs rerun on a host with
Vulkan/MoltenVK support.
