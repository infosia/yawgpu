# yawgpu (Yet Another wgpu)

A from-scratch implementation of the **WebGPU C API** (`webgpu.h`) in Rust.

yawgpu lets native applications written in C, C++, or any language with a C
FFI talk to the GPU through the standard WebGPU interface — the same
`wgpuCreateInstance` / `wgpuDeviceCreateBuffer` / `wgpuQueueSubmit` surface
that browsers expose to WebAssembly — without a browser, a JavaScript engine,
or a web runtime.

## What makes it different

yawgpu implements the **entire** WebGPU stack itself:

- the **C ABI** (`webgpu.h` entry points, opaque handles, reference
  counting),
- **WebGPU semantics and validation** (descriptor checking, usage rules,
  state tracking, the error-scope model),
- **resource and lifetime management** (buffers, textures, pipelines,
  command encoding), and
- the **GPU backends** themselves — a hand-written hardware abstraction
  layer that talks to Metal and Vulkan directly.

The only third-party GPU-stack dependency is **naga**, used to translate
WGSL shaders into the platform shading languages (MSL for Metal, SPIR-V for
Vulkan). Everything else — validation, the object model, the backends — is
original code.

This is a deliberately different point in the design space from a thin C
shim layered over an existing Rust GPU engine: yawgpu owns the whole
pipeline from the C call down to the native graphics API, which keeps the
implementation legible and self-contained.

## Architecture

yawgpu is a small Cargo workspace of layered crates:

```
        C / C++ application
                │  webgpu.h  (standard WebGPU C ABI)
                ▼
┌───────────────────────────────────────────────┐
│ yawgpu        C ABI layer                       │  cdylib + staticlib + rlib
│               extern "C" entry points,          │
│               opaque Arc-based handles,          │
│               C↔Rust descriptor conversion       │
├───────────────────────────────────────────────┤
│ yawgpu-core   WebGPU semantics                   │  platform-independent
│               validation, object model,          │
│               resource lifetimes, error scopes   │
├───────────────────────────────────────────────┤
│ yawgpu-hal    hardware abstraction layer         │  enum dispatch (no dyn)
│               Noop · Metal · Vulkan              │
└───────────────────────────────────────────────┘
                │               │
              Metal           Vulkan
```

- **`yawgpu`** — the public crate. It exports the `webgpu.h` symbols as a C
  dynamic/static library and binds the canonical header with `bindgen`.
- **`yawgpu-core`** — the platform-independent heart: the WebGPU object
  model, descriptor validation, resource state tracking, the asynchronous
  map/submit/error-scope machinery. It has no knowledge of any specific GPU
  API.
- **`yawgpu-hal`** — the hardware abstraction layer. Backends are selected
  by static `enum` dispatch (never `dyn Trait`) and gated behind Cargo
  features, so a build only compiles the backends it needs.

## Backends

| Backend | Purpose | Notes |
|---|---|---|
| **Noop** | CPU-only reference backend | Always available; runs the full validation layer with no GPU. Ideal for CI and headless testing. |
| **Metal** | Apple platforms | Built with the `metal` feature via the `objc2` family. |
| **Vulkan** | Cross-platform | Built with the `vulkan` feature via `ash`; runs on MoltenVK on macOS. |

OpenGL/GLES and Direct3D are intentionally out of scope.

A backend is chosen at instance-creation time through a small vendor
extension chained onto `WGPUInstanceDescriptor` — applications that only ever
want validation can run entirely on Noop with no GPU present.

## Using it from C

Build the library and link against it with the vendored header:

```sh
# Noop-only (no GPU dependencies)
cargo build -p yawgpu --release

# with a real backend
cargo build -p yawgpu --release --features metal      # Apple
cargo build -p yawgpu --release --features vulkan     # Vulkan / MoltenVK
```

This produces `libyawgpu.{a,dylib,so}` (and a Windows `.dll`). Include
`yawgpu/ffi/webgpu-headers/webgpu.h`, link the library, and call the standard
`wgpu*` functions.

## Using it from Rust

The same entry points are available as a normal Rust crate (`rlib`); the
generated bindings are exposed under `yawgpu::native`, and the
`extern "C"` functions are callable directly.

## Examples

The `examples/` directory contains small **C programs** built with CMake that
link against `libyawgpu` and exercise the C ABI:

| Example | What it shows |
|---|---|
| `enumerate_adapters` | Listing adapters and their properties |
| `device_info` | Querying adapter/device limits and features |
| `compute` | A storage-buffer compute dispatch with readback |
| `capture` | Offscreen render → texture → buffer readback → PNG file |
| `surface_smoke` | Opening a window and presenting cleared frames |
| `triangle` | A classic windowed triangle (vertex-index shader) |
| `hello_triangle` | A windowed triangle fed from a vertex buffer |

```sh
brew install cmake glfw          # windowed examples need GLFW
cmake -S examples -B examples/build
cmake --build examples/build

# pick a backend at runtime
YAWGPU_BACKEND=metal  ./examples/build/triangle/triangle
YAWGPU_BACKEND=vulkan ./examples/build/compute/compute
```

The C sources are written to modern C17 (strict ISO, no compiler
extensions).

## Shaders

Shaders are authored in **WGSL** and compiled at pipeline-creation time by
naga into the backend's native language — Metal Shading Language for Metal,
SPIR-V for Vulkan.

## Quality

- **Validation-tested**: the WebGPU validation rules are exercised by an
  extensive suite that runs on the Noop backend with no GPU, so correctness
  checks need no hardware.
- **Unit-tested public API**: every public function across the three crates
  has a direct unit test.
- **Real-GPU end-to-end tests**: buffer/texture/compute/render paths are
  verified against live Metal and Vulkan devices.
- **Platform coverage**:
  - **macOS** — builds, unit tests, real-GPU end-to-end tests, and the C
    examples all verified (Metal and Vulkan/MoltenVK).
  - **Windows (MSVC)** — builds and passes the full unit-test suite; the C
    examples are not yet verified on this platform.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
