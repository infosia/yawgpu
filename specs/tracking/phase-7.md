# Phase 7 — Real backends

Status: **in progress** (P7.0 active). Rules/plan:
`../blocks/60-real-backends.md`. Roles/loop:
`../reference/workflow.md`.

**Roadmap divergence (approved):** SPEC roadmap lists Phase 7 as
"Vulkan→Metal"; we bring up **Metal first, then Vulkan** because the
dev platform is macOS (Metal native; no MoltenVK/Vulkan-SDK needed for
on-machine real-GPU verification). Vulkan (P7.6) reuses the identical
HAL contract. SPEC.md Phase-7 row annotated accordingly.

**Gate (permanent):** `cargo test --workspace` + `cargo clippy
--workspace --all-targets -- -D warnings` green on **Noop**
(real-backend code is build-only in CI). Per slice **also**: `cargo
build -p yawgpu --features metal` (later `--features vulkan`) +
clippy with the feature. **Real-GPU end2end** (`cargo test --features
metal -- --ignored`) is run **by Claude directly** — the Bash tool
executes on this Apple Silicon and the sandbox permits Metal device access
(confirmed P7.1) — and logged here per slice (no manual user step for
Metal). Vulkan (P7.6) via MoltenVK is also machine-runnable
(`$VULKAN_SDK` = `$VULKAN_SDK`; Apple Silicon enumerates).
**Phase ends with the mandatory Phase Review**
(`tracking/phase-7-review.md`).

Methodology shift vs Phases 0–6: not validation-rule porting —
execution bring-up verified by gated Dawn `end2end` Basic/Compute/Copy
ports. Validation stays in `yawgpu-core`; backends only execute
already-validated work; driver failure → `HalError` → device error,
never panic.

## P7.0 — Bring-up scaffolding + gating harness  *(☑ DONE)*

Done: `metal` crate (0.33.0, recorded in `dependencies.md`) wired as
an **optional** `yawgpu-hal` dep behind `metal = ["dep:metal"]`
(`default = ["noop"]` unchanged); `yawgpu` gained a `metal` feature
forwarding to `yawgpu-hal/metal`. Inline `metal` HAL placeholder
moved to `yawgpu-hal/src/metal/mod.rs` mirroring the Noop contract
(`MetalInstance/Adapter/Device/Queue/Buffer/Texture/Sampler`): every
fallible entry (`*::new`, `MetalAdapter::create_device`) returns
`HalError::BackendUnavailable`, `enumerate_adapters()` is empty (so
the `HalInstance::Metal` arm is unreachable), infallible creators are
allocation-counting no-ops; `use metal as _;` proves link with **zero
Objective-C/MTL calls**. `yawgpu-test` gained `RealBackend` +
`real_backend_available` (→ false in P7.0) + `real_backend_skip_
reason`; one `#[ignore]` `yawgpu/tests/e2e_metal_smoke.rs` asserting
unavailability (proves the harness shape). `wgpuCreateInstance`
backend *selection* intentionally deferred to P7.1 (nothing real to
select yet; Noop remains the only reachable backend). Gate: Noop
`cargo test --workspace` 43 binaries green + `clippy --workspace
--all-targets -D warnings` clean (smoke ignored, not run); `cargo
build -p yawgpu --features metal` + `clippy -p yawgpu --features
metal --all-targets -D warnings` clean; smoke passes on `--features
metal -- --ignored`. Committed `phase-7: P7.0`.

## P7.1 — Metal Instance/Adapter/Device/Queue  *(☑ DONE — real-GPU-verified)*

Done (codex + Noop/feature gate): `metal` module real for objects —
`MetalInstance::new` ok; `enumerate_adapters` via `metal::Device::
all()` (name from `device.name()`); `MetalAdapter::create_device`
builds a `metal::CommandQueue`; `MetalDevice` retains device+queue;
`MetalQueue::submit_empty` = new command buffer → `commit()` →
`wait_until_completed()`; buffer/texture/sampler stay P7.0 counter-
only stubs (`// P7.2/P7.3`). No panics (`system_default`/`all` →
`Option`→`HalError`). `HalError` gained `DeviceCreationFailed`/
`QueueSubmissionFailed`; `HalAdapter::{name,backend}` + `HalBackend`;
`HalQueue::submit_empty` (Noop/Vulkan = `Ok(())` no-op — Noop
byte-for-byte unchanged; Metal real). `core::Queue::submit` returns
`Option<DeviceError>`; **only zero-CB submits** call
`hal.submit_empty()` (`HalError`→`DeviceError::internal`); validation
path unchanged. `DeviceError::{validation,internal}` ctors. FFI:
yawgpu vendor `WGPUYawgpuInstanceBackendSelect` chained struct
(SType `0x7000_0001`, backend Noop=0/Metal=1/Vulkan=2);
`wgpuCreateInstance` selects Metal only when the struct requests it
**and** `cfg(feature="metal")` **and** ≥1 adapter — else exact
`new_noop()` fallback; `WGPUInstanceImpl::from_core`;
`wgpuAdapterGetInfo`/`wgpuAdapterInfoFreeMembers` (for the name
assertion); `dispatch_optional_device_error`. `yawgpu-test` gained an
optional `metal` feature; `real_backend_available(Metal)` probes
`metal::Device::system_default()`. Tests: `e2e_metal_basic.rs` (3:
adapter name, device+queue+empty-submit, default-instance-is-Noop) +
`e2e_metal_smoke.rs` updated to match the probe — all `#[ignore]` +
`cfg(feature="metal")` self-skip. Gate: Noop `cargo test --workspace`
44 binaries green + `clippy --workspace --all-targets -D warnings`
clean; `cargo build -p yawgpu --features metal` + `clippy -p yawgpu
--features metal --all-targets -D warnings` clean; e2e tests ignored
(not run — no GPU in codex/CI). Committed `phase-7: P7.1`.
Real-GPU verified by Claude directly (the Bash tool runs on this
Apple Silicon; the seatbelt sandbox permits Metal device access — no
manual user step needed for Metal slices).

### P7.1 real-GPU run log
- 2026-05-19, Apple Silicon, `cargo test -p yawgpu --features metal --test
  e2e_metal_basic --test e2e_metal_smoke -- --ignored`:
  **e2e_metal_basic 3/3 pass** (adapter name, device+queue+empty
  submit, default-instance-Noop) + **e2e_metal_smoke 1/1 pass**.
  P7.1 hardware-confirmed.

## P7.2 — Metal Buffer + writeBuffer/submit + B2B  *(NEXT — gated on P7.1 M2 verify)*
`MTLBuffer` alloc + map/staging + readback; B2B copy. Port
`BufferTests`/`CopyTests` buffer subset.

## P7.3 — Metal Texture/Sampler + B2T/T2B/T2T  *(after P7.2)*
`MTLTexture`/view/sampler; blit copies. Port `CopyTests` texture
subset.

## P7.4 — Metal Shader (naga→MSL) + compute dispatch  *(after P7.3)*
WGSL→MSL via naga MSL backend; compute pipeline + dispatch +
storage-buffer readback. Port `ComputeDispatchTests` (basic).

## P7.5 — Metal render pipeline + render pass draw  *(after P7.4)*
Render pipeline; render pass load/store + draw; color readback.
Port `BasicTests` render / minimal draw.

## P7.6 — Vulkan bring-up (mirror P7.1–P7.5)  *(after P7.5)*
`ash` + MoltenVK on macOS; naga→SPIR-V; same HAL contract; reuse the
ported end2end tests parametrized by backend feature.

## Phase 7 exit criteria

- Metal + Vulkan fill their HAL enum arms; `yawgpu-core` validation
  unchanged & still green on Noop; per-slice `--features` build +
  clippy clean.
- Ported `end2end` Basic/Compute/Copy pass on real GPU (Metal on this
  machine; Vulkan as available) — user-run, logged here per slice.
- One commit per slice (`phase-7: <slice> — <short>`).
- **Mandatory Phase 7 Review** before COMPLETE; logged in
  `tracking/phase-7-review.md`.
