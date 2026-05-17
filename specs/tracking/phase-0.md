# Phase 0 — Scaffold + FFI + test harness

Status: **planned, not started**

Phase 0 is split into 6 ordered task handoffs (T0.1–T0.6) for the coding
agent. Each is self-contained; Claude reviews and commits after each.
Roles & loop: `../reference/workflow.md`. Conventions: `../../CLAUDE.md`.

Decisions locked for Phase 0:
- `webgpu.h` is **vendored** into `yawgpu/ffi/webgpu-headers/` (copied from
  `webgpu-headers/`, wgpu-native style) so bindgen output is
  reproducible. Record the source commit/date in
  `../reference/dependencies.md`.
- naga is **not** introduced in Phase 0 (first needed Phase 4).
- A minimal GitHub Actions `cargo test` workflow IS added (Noop = no GPU).

---

## T0.1 — Workspace skeleton

Produce:
- `/Cargo.toml` — `[workspace] resolver = "2"`, members `yawgpu`,
  `yawgpu-core`, `yawgpu-hal`, `yawgpu-test`. `[workspace.package]`
  (edition 2021, `license = "MIT OR Apache-2.0"`). `[workspace.dependencies]`:
  `thiserror = "2"`, `parking_lot = "0.12"`, `bitflags = "2"`,
  `smallvec = "1"`, `raw-window-handle = "0.6"`.
- Empty crate skeletons: `yawgpu-core/{Cargo.toml,src/lib.rs}`,
  `yawgpu-hal/{Cargo.toml,src/lib.rs}`,
  `yawgpu-test/{Cargo.toml,src/lib.rs}`,
  `yawgpu/{Cargo.toml,src/lib.rs}` (yawgpu: `crate-type =
  ["cdylib","staticlib"]`, deps on the other three by path).
- `.gitignore`: `/target`, `.claude/`, `*.transcript`.
- `LICENSE-MIT`, `LICENSE-APACHE`.

Acceptance:
- [ ] `cargo build` succeeds (empty crates).
- [ ] `cargo metadata` lists all 4 members.

---

## T0.2 — Vendor header + bindgen

Produce:
- Copy `webgpu-headers/webgpu.h` →
  `yawgpu/ffi/webgpu-headers/webgpu.h`. Note source date/commit in the
  report.
- `yawgpu/build.rs`: `bindgen` over `ffi/webgpu-headers/webgpu.h` with:
  - `.allowlist_item("WGPU.*")`, `.allowlist_item("wgpu.*")`
  - `.prepend_enum_name(false)`, `.size_t_is_usize(true)`
  - `.ignore_functions()` (all `wgpu*` fns hand-written)
  - **opaque-handle rename trick**: for each `WGPUXxx` object typedef,
    `.blocklist_type("WGPUXxx")` + `.raw_line("pub type WGPUXxx = *const
    crate::WGPUXxxImpl;")` (enumerate all object handles from the header —
    Adapter, BindGroup, BindGroupLayout, Buffer, CommandBuffer,
    CommandEncoder, ComputePassEncoder, ComputePipeline, Device,
    Instance, PipelineLayout, QuerySet, Queue, RenderBundle,
    RenderBundleEncoder, RenderPassEncoder, RenderPipeline, Sampler,
    ShaderModule, Surface, Texture, TextureView).
  - output to `$OUT_DIR/bindings.rs`.
- `yawgpu/src/lib.rs`: `pub mod native { #![allow(...)] include!(concat!(
  env!("OUT_DIR"), "/bindings.rs")); }` and empty `WGPUXxxImpl` structs so
  the raw_line type aliases resolve (real fields added in T0.4/T0.5).
- `yawgpu/Cargo.toml`: `[build-dependencies] bindgen = "0.72"`.
- Update `../reference/dependencies.md` pinning section.

Acceptance:
- [ ] `cargo build -p yawgpu` succeeds; bindings generated.
- [ ] `native::WGPUBufferDescriptor` and friends exist; `WGPUBuffer`
      resolves to `*const WGPUBufferImpl`.

---

## T0.3 — yawgpu-hal Noop backend skeleton

Produce in `yawgpu-hal`:
- `lib.rs`: enum-dispatch types `HalInstance`, `HalAdapter`, `HalDevice`,
  `HalQueue` each `enum { Noop(noop::...) }` (Vulkan/Metal variants behind
  `#[cfg(feature = "vulkan"|"metal")]`, **not** implemented in Phase 0).
  Define `HalError` (thiserror).
- `noop/`: synthetic `NoopInstance::new()`, `enumerate_adapters` →
  one synthetic adapter, `create_device` → `NoopDevice` with an
  allocation counter (`AtomicU64`) for later buffer/texture tracking.
- `Cargo.toml`: features `noop` (default), `vulkan`, `metal`; no GPU deps.

Acceptance:
- [ ] `cargo test -p yawgpu-hal` green.
- [ ] A unit test creates instance→adapter→device on Noop and asserts the
      allocation counter is zero initially.

---

## T0.4 — yawgpu-core foundation

Produce in `yawgpu-core`:
- `Instance`, `Adapter`, `Device`, `Queue` as `Arc`-backed structs wrapping
  the matching `Hal*`. ID/registry not required yet (added Phase 1+).
- **Error sink**: per-`Device` uncaptured-error callback slot + an
  error-scope stack API (`push_error_scope`, `pop_error_scope`,
  `dispatch_error(kind, msg)` → routes to current scope or uncaptured
  callback). No panics — `Result`/sink only.
- Future registry stub: `FutureId`, `register`, `complete`, `poll_all`
  (Noop completes synchronously on poll). Enough for T0.6.

Acceptance:
- [ ] `cargo test -p yawgpu-core` green.
- [ ] Unit test: pushing an error scope, dispatching an error, popping it
      returns that error and does NOT hit the uncaptured callback.

---

## T0.5 — yawgpu FFI: instance/adapter/device/queue + handle lifetime

Produce in `yawgpu`:
- Real `WGPUInstanceImpl/AdapterImpl/DeviceImpl/QueueImpl` wrapping
  `yawgpu_core::*` (Arc). `Drop` releases.
- Hand-written `extern "C"`:
  - `wgpuCreateInstance`, `wgpuInstanceRelease`, `wgpuInstanceAddRef`
  - `wgpuInstanceRequestAdapter` (+ `WGPURequestAdapterCallbackInfo`),
    `wgpuAdapterRelease`/`AddRef`, `wgpuAdapterRequestDevice`
    (+ `WGPURequestDeviceCallbackInfo`)
  - `wgpuDeviceRelease`/`AddRef`, `wgpuDeviceGetQueue`,
    `wgpuQueueRelease`/`AddRef`
  - `wgpuInstanceProcessEvents`, `wgpuInstanceWaitAny`
- `conv.rs`: `WGPUStringView` ↔ `&str`/label helper; handle in/out via
  `Arc::into_raw`/`from_raw`/borrow-without-consume helpers (document the
  refcount contract; mirror wgpu-native).
- Convention: invalid/null handle where spec forbids null → `expect(...)`
  at FFI boundary (allowed exception per CLAUDE.md); spec validation
  failures → error sink.

Acceptance:
- [ ] `cargo build` produces cdylib + staticlib.
- [ ] No panics on valid input paths; AddRef/Release balance verified by a
      test (create, addref, release, release → core Arc dropped once).

---

## T0.6 — yawgpu-test harness + first TDD slice (Red→Green)

Produce:
- `yawgpu-test` crate:
  - `ValidationTest` fixture: builds a Noop instance→adapter→device with a
    captured uncaptured-error sink (errors pushed to a `Vec`).
  - `assert_device_error!(expr)` macro ≈ Dawn `ASSERT_DEVICE_ERROR`: runs
    `expr`, asserts exactly one device error was captured (optional
    substring matcher arg).
  - future/poll helper: `wait(future)` driving
    `wgpuInstanceProcessEvents`.
- `yawgpu/tests/instance_smoke.rs`:
  - `wgpuCreateInstance` → non-null → `wgpuInstanceRelease`.
  - Noop round-trip: `wgpuInstanceRequestAdapter` →
    `wgpuAdapterRequestDevice` → `wgpuDeviceGetQueue` → release all,
    callbacks fire via `wgpuInstanceProcessEvents`.
  - Negative: a deliberately injected device error is caught by
    `assert_device_error!` (proves the harness works).
- `.github/workflows/ci.yml`: `cargo build` + `cargo test` on
  ubuntu-latest (Noop, no GPU).

Acceptance:
- [ ] `cargo test` green locally and in CI on Noop with no GPU.
- [ ] `assert_device_error!` demonstrably passes on the injected error and
      would fail if no error occurred (include a `#[should_panic]` guard
      test).

---

## Phase 0 exit criteria

- All T0.1–T0.6 acceptance boxes checked.
- `cargo build` (cdylib+staticlib) and `cargo test` green on Noop, no GPU,
  in CI.
- `git init` done; one commit per task (`phase-0: <task> — <short>`).
- `../reference/dependencies.md` pinning section filled.
- `../reference/dawn-test-mapping.md`: `ValidationTest (base)` row → ☑.
