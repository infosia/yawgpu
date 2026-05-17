# Naming conventions

## C ABI (fixed by webgpu.h — do not deviate)

- Objects: opaque `WGPUXxx` (e.g. `WGPUDevice`, `WGPUBuffer`).
- Functions: `wgpuObjectMethod` (e.g. `wgpuDeviceCreateBuffer`,
  `wgpuBufferMapAsync`), object is first parameter.
- Lifetime: `wgpuXxxAddRef` / `wgpuXxxRelease`.
- Descriptors: `WGPUXxxDescriptor` with `nextInChain` extension chain.
- Async: `WGPUFuture` return + `WGPUXxxCallbackInfo { mode, callback,
  userdata1, userdata2 }`.

## Rust internals

- C ABI crate: `yawgpu`. Implementation handle types: `WGPUXxxImpl`
  (matches wgpu-native; the rename trick maps `WGPUXxx` → `*const WGPUXxxImpl`).
- Core types: plain Rust names in `yawgpu-core` (e.g. `Device`, `Buffer`,
  `BufferDescriptor`) — not `WGPU`-prefixed.
- HAL types: `Hal` prefix enums (`HalDevice`, `HalBuffer`); per-backend
  inner types `NoopDevice`, `VulkanDevice`, `MetalDevice`.
- Conversion fns in `conv.rs`: `map_<thing>` (e.g. `map_buffer_usage`),
  mirroring wgpu-native.
- Test fixtures/files: `tests/<area>_validation.rs` (port of Dawn
  `XxxValidationTests.cpp`).

## Crates

`yawgpu`, `yawgpu-core`, `yawgpu-hal`, `yawgpu-test`. Examples under
`examples/<name>` (kebab-case).
