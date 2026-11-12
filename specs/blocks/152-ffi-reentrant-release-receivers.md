# Block 152 — FFI receivers survive a reentrant release in the error callback

Status: **S1 landed (2026-11-13)**; Phase Review done; S2 (m1) pending.
Source: whole-code review at `92e38c7` (`specs/tracking/code-review-2026-11.md`), finding R01. It extends Block
113 M1 / audit rule C11, which covered device receivers only.

## Problem

An FFI entry point that borrows a handle (`borrow_handle`, no reference held) and then
reaches the device error sink can run the application's uncaptured-error callback
synchronously. If that callback releases the last external reference to the borrowed
object, the entry point keeps reading a freed `WGPU*Impl` after the dispatch returns.
Holding an `Arc` to the underlying core object does not keep the FFI wrapper alive.

Known sites (review R01): `wgpuTextureCreateView` (`ffi/texture.rs`), the command-encoder
entry points in `ffi/encoder.rs` (begin passes, copies, finish, …), the bundle-encoder
finish path (`ffi/bundle.rs`), the tiled entry points (`ffi/tiled.rs`), the pipeline
`GetBindGroupLayout` error paths (`ffi/mod.rs`), and `wgpuDeviceCreateBindGroup`'s
borrowed bind-group layout (`ffi/device.rs`).

## Rules

R1. **Retain before dispatch.** Every FFI entry point that can reach the device error sink
(`dispatch_error`, `dispatch_optional_error`, or a core call that dispatches) holds an
owned reference (`clone_handle`) to every handle it reads after that point: the receiver
and every secondary handle taken from the arguments or the descriptor. The alternative —
copy the needed owned fields out before the first possible dispatch and never touch the
borrowed wrapper afterwards — is also allowed. Entry points that cannot dispatch keep
`borrow_handle`.

R2. **Audit, not just the listed sites.** The coding agent audits every `borrow_handle`
in `yawgpu/src/ffi/**` against R1 and lists, in the report, each entry point changed and
each entry point left on `borrow_handle` because it cannot dispatch (grouped, not one line
per getter). Pass-encoder recording calls that only record an error into the encoder
(dispatch happens at `finish`) do not dispatch.

R3. **Hot paths stay cheap.** Changing a draw/dispatch/set-state recording call that does
not dispatch is out of scope.

R4. **No borrowed `&core::Device` is a live argument across the callback.** A dispatch
from a borrowed wrapper passes a `&core::Device` that points into an allocation the
wrapper owns; if the callback frees the wrapper, that reference is still a protected
function argument (formal Stacked/Tree Borrows UB, though nothing reads it). The FFI
dispatch helpers (`dispatch_optional_error`, `dispatch_optional_device_error`, and every
direct `<wrapper>.device.dispatch_error(..)` reached from a borrowed wrapper) clone the
`Arc<core::Device>` on the error branch only and dispatch through that owned clone. The
success path stays free of reference-count traffic (R3).

## Slices

S1: R1–R3 with tests.

S2 (Phase Review m1): R4.

## Verification

- Inline FFI unit tests modeled on
  `audit_c11_uncaptured_callback_can_release_its_device_reentrantly` (`ffi/mod.rs`): an
  uncaptured-error callback releases the receiver (texture, command encoder, render-bundle
  encoder, pipeline) or the secondary handle (the bind-group layout of
  `wgpuDeviceCreateBindGroup`), the call triggers a validation error, and the test
  checks that the callback fired, the returned object is an error object, and the
  released wrapper is freed afterwards (a `Weak` no longer upgrades). One `tiled` entry
  point gets the same test under `--features tiled`. Under plain `cargo test` a
  use-after-free is silent UB, so the structural audit in R2 is the primary evidence.
- Agent gates: `cargo test -p yawgpu --lib`, `cargo test -p yawgpu --features tiled --lib`,
  clippy with and without `tiled`.
- Claude: workspace test; optional AddressSanitizer run of the new tests on nightly.

## Verification record

S1: workspace test 1,459/0; `yawgpu --lib` 315/0; `yawgpu --features tiled --lib` 319/0;
clippy clean with and without `tiled`. Nine `audit_r01_*` tests were red before the fix.
The R2 audit changed 22 of 175 production `borrow_handle` sites (texture view, 14
command-encoder entry points, bundle finish, both `GetBindGroupLayout`s,
`CreateBindGroup`'s layout, `BeginSubpassRenderPass`, `ProcessEvents`, `WaitAny`); the
rest either cannot dispatch or dispatch only as their last wrapper access.

## Phase Review (2026-11-13, fresh-context reviewer over `16965af`)

0 CRITICAL, 0 MAJOR, 2 MINOR. The reviewer's independent audit of every production
`borrow_handle` found no wrapper read after a possible dispatch, and confirmed the nine
tests fail without the fix (the callback releases only while the entry point holds a
reference).

| # | Severity | Finding | Resolution |
|---|---|---|---|
| m1 | MINOR | Borrowed-receiver dispatch passes a `&core::Device` into the wrapper's allocation; freed during the callback while still a protected argument | S2: R4 |
| m2 | MINOR | Encoder debug-marker and copy entry points were retained although their dispatch is the last wrapper access | Accepted: retained defensively (not hot paths); the ledger above counts them among the 22 |
