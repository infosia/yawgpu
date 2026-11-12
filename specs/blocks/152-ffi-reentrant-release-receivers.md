# Block 152 — FFI receivers survive a reentrant release in the error callback

Status: **PLANNED (2026-11-13)**.
Source: whole-code review at `92e38c7` (`REPORT-review.md`), finding R01. It extends Block
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

## Slices

S1: R1–R3 with tests.

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
