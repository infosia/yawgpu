# yawgpu (Yet Another wgpu)

A from-scratch implementation of the **WebGPU C API** (`webgpu.h`) in Rust.

yawgpu lets native applications written in C, C++, or any language with a C
FFI talk to the GPU through the standard WebGPU interface — the same
`wgpuCreateInstance` / `wgpuDeviceCreateBuffer` / `wgpuQueueSubmit` surface
that browsers expose to WebAssembly — without a browser, a JavaScript engine,
or a web runtime.

On top of the standard `webgpu.h`, yawgpu ships a small companion header
[`yawgpu.h`](yawgpu/ffi/webgpu-headers/yawgpu.h) that adds vendor extensions
for backend selection, precompiled-shader passthrough (SPIR-V / MSL), and
tile-based deferred rendering (subpasses, transient attachments, framebuffer
fetch). See **[Vendor extensions](#vendor-extensions-yawgpuh)** below.

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
                │  yawgpu.h  (vendor extensions)
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

A backend is chosen at instance-creation time through `YaWGPUInstanceBackendSelect`
(see below) — applications that only ever want validation can run entirely
on Noop with no GPU present.

## Vendor extensions (`yawgpu.h`)

`yawgpu.h` is a companion header that sits next to `webgpu.h` and exposes
the yawgpu-specific surface area. It follows a strict naming convention so
that vendor symbols never collide with the standard WebGPU C API:

| Kind | Prefix | Example |
|---|---|---|
| Functions | `yawgpu*` | `yawgpuDeviceCreateShaderModuleSpirV` |
| Types / structs / enums / handles | `YaWGPU*` | `YaWGPUTransientAttachment` |
| Constants / macros / SType tags | `YAWGPU_*` / `YAWGPU_STYPE_*` | `YAWGPU_STYPE_INSTANCE_BACKEND_SELECT` |
| Feature names | `YaWGPUFeatureName_*` | `YaWGPUFeatureName_MultiSubpass` |

Every descriptor ships a matching `YAWGPU_*_INIT` zero/sentinel initializer
macro, mirroring `webgpu.h` ergonomics. Each extension is gated by an
opt-in Cargo feature; the default build only exposes the standard WebGPU C
API plus backend selection.

| Cargo feature | What it adds | Header guard |
|---|---|---|
| *(none — always on)* | Backend selection via `YaWGPUInstanceBackendSelect` | — |
| `shader-passthrough` | Raw SPIR-V / MSL shader modules, bypassing WGSL→naga | `YAWGPU_HAS_SHADER_PASSTHROUGH` |
| `tiled` | TBDR primitives: subpasses, transient attachments, framebuffer fetch | `YAWGPU_HAS_TILED` |
| `mobile` | Aggregate of `shader-passthrough` + `tiled` for mobile GPU targets | both guards |

### Backend selection (always available)

Chain `YaWGPUInstanceBackendSelect` onto `WGPUInstanceDescriptor` to pick
which backend the instance will create:

```c
#include "webgpu.h"
#include "yawgpu.h"

YaWGPUInstanceBackendSelect sel = {
    .chain   = { .sType = YAWGPU_STYPE_INSTANCE_BACKEND_SELECT },
    .backend = YAWGPU_INSTANCE_BACKEND_METAL,   /* or _VULKAN, _NOOP */
};
WGPUInstanceDescriptor desc = { .nextInChain = &sel.chain };
WGPUInstance instance = wgpuCreateInstance(&desc);
```

The same effect is available via the `YAWGPU_BACKEND` environment variable
that the bundled C examples use.

### Shader passthrough — `shader-passthrough`

Engines that already ship native shader bytes (a `.spv` blob for Vulkan, an
MSL source string for Metal) can hand them straight to yawgpu rather than
authoring WGSL. The default pipeline path always goes through naga; this
extension keeps the caller's bytes intact and only uses naga's reflection
to recover the metadata pipeline creation needs.

```c
/* Vulkan: SPIR-V words go in; naga reflects entry points and bindings. */
YaWGPUShaderModuleSpirVDescriptor spv = YAWGPU_SHADER_MODULE_SPIRV_DESCRIPTOR_INIT;
spv.codeSize = word_count;
spv.code     = spirv_words;
WGPUShaderModule m = yawgpuDeviceCreateShaderModuleSpirV(device, &spv);

/* Metal: caller supplies MSL source + entry-point metadata. The pipeline
   layout drives the Metal binding-index mapping documented in yawgpu.h. */
YaWGPUMslEntryPoint entries[] = {
    { .name = { .data = "vs_main", .length = 7 }, .stage = WGPUShaderStage_Vertex   },
    { .name = { .data = "fs_main", .length = 7 }, .stage = WGPUShaderStage_Fragment },
};
YaWGPUShaderModuleMslDescriptor msl = YAWGPU_SHADER_MODULE_MSL_DESCRIPTOR_INIT;
msl.code            = (WGPUStringView){ msl_src, msl_src_len };
msl.entryPoints     = entries;
msl.entryPointCount = 2;
WGPUShaderModule m = yawgpuDeviceCreateShaderModuleMsl(device, &msl);
```

`examples/triangle_passthrough` exercises both paths on the same triangle
program.

### Tiled rendering — `tiled`

On tile-based deferred renderers (Apple GPUs; mobile Vulkan on Adreno /
Mali / PowerVR) a multi-pass G-buffer pipeline can keep intermediate
attachments **in tile memory** instead of round-tripping to system RAM.
WebGPU's core API has no concept of subpasses or transient attachments;
this extension exposes them as additive vendor entry points.

The shape is:

- **Capabilities** — `yawgpuAdapterGetTiledCapabilities` reports
  `maxSubpasses`, `maxSubpassColorAttachments`, `maxInputAttachments`, and
  an `estimatedTileMemoryBytes` hint. Three vendor feature names
  (`YaWGPUFeatureName_MultiSubpass`, `_TransientAttachments`,
  `_ShaderFramebufferFetch`) plug into the standard
  `wgpuAdapterHasFeature` / `WGPUDeviceDescriptor.requiredFeatures` flow.
- **Transient attachments** — `YaWGPUTransientAttachment` is a first-class
  Arc resource with no DRAM backing (`VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`
  + `LAZILY_ALLOCATED` on Vulkan; `MTLStorageModeMemoryless` on Metal).
  It is only legal as a slot inside a subpass render pass — never bound
  through a bind group.
- **Subpass pass layout** — a reusable `YaWGPUSubpassPassLayout`
  describes attachment formats, per-subpass usage, input-attachment source
  mapping, and dependencies once; both pipeline creation and pass begin
  reference it. This is the single source of truth for Vulkan
  `VkRenderPass` compatibility.
- **Subpass-aware render pipelines** — `yawgpuDeviceCreateSubpassRenderPipeline`
  wraps `WGPURenderPipelineDescriptor` with a `(passLayout, subpassIndex)`
  pair.
- **Subpass-input bindings** — chain `YaWGPUInputAttachmentBindingLayout`
  onto a bind-group-layout entry to mark a `(group, binding)` as an input
  attachment. The *resource* feeding it is wired automatically from the
  pass layout's source mapping; the caller never creates a bind group or
  view for it.
- **Subpass render pass encoder** — `yawgpuCommandEncoderBeginSubpassRenderPass`
  → record draws → `yawgpuSubpassRenderPassEncoderNextSubpass` → more
  draws → `…End`. Mirrors the WebGPU render-pass-encoder shape with a
  `nextSubpass` step in the middle.

`examples/tiled_deferred` records a two-subpass offscreen pass (G-buffer →
lighting) that reads the G-buffer through a subpass input from tile
memory, then copies the persistent output to a PNG.

## Using it from C

Build the library and link against it with the vendored headers:

```sh
# Noop-only (no GPU dependencies)
cargo build -p yawgpu --release

# with a real backend
cargo build -p yawgpu --release --features metal      # Apple
cargo build -p yawgpu --release --features vulkan     # Vulkan / MoltenVK

# with vendor extensions
cargo build -p yawgpu --release --features "vulkan tiled"
cargo build -p yawgpu --release --features "metal mobile"   # = shader-passthrough + tiled
```

This produces `libyawgpu.{a,dylib,so}` (and a Windows `.dll`). Include
`yawgpu/ffi/webgpu-headers/webgpu.h` (and `yawgpu.h` for the vendor
extensions), link the library, and call the standard `wgpu*` /
`yawgpu*` functions.

## Using it from Rust

The same entry points are available as a normal Rust crate (`rlib`); the
generated bindings are exposed under `yawgpu::native`, and the
`extern "C"` functions are callable directly.

## Examples

The `examples/` directory contains small **C programs** built with CMake that
link against `libyawgpu` and exercise both the standard `webgpu.h` and the
`yawgpu.h` vendor extensions:

| Example | What it shows | Requires |
|---|---|---|
| `enumerate_adapters` | Listing adapters and their properties | core |
| `device_info` | Querying adapter/device limits and features | core |
| `compute` | A storage-buffer compute dispatch with readback | core |
| `capture` | Offscreen render → texture → buffer readback → PNG file | core |
| `surface_smoke` | Opening a window and presenting cleared frames | core |
| `triangle` | A classic windowed triangle (vertex-index shader) | core |
| `hello_triangle` | A windowed triangle fed from a vertex buffer | core |
| `triangle_passthrough` | The same triangle driven from precompiled SPIR-V and MSL | `shader-passthrough` |
| `tiled_deferred` | Two-subpass G-buffer + lighting with a tile-memory input attachment | `tiled` (Metal / native Vulkan; not MoltenVK) |

Pick a backend at runtime with `YAWGPU_BACKEND`:

```sh
# macOS
brew install cmake glfw
cmake -S examples -B examples/build
cmake --build examples/build

YAWGPU_BACKEND=metal  ./examples/build/triangle/triangle
YAWGPU_BACKEND=vulkan ./examples/build/compute/compute

# extensions: pass YAWGPU_EXTENSIONS as a CMake cache string
cmake -S examples -B examples/build-tiled \
      -DYAWGPU_FEATURE=metal -DYAWGPU_EXTENSIONS="tiled"
cmake --build examples/build-tiled
```

On Windows (MSVC + Vulkan SDK):

```powershell
cmake -S examples -B examples/build -DYAWGPU_FEATURE=vulkan
cmake --build examples/build
$env:YAWGPU_BACKEND = "vulkan"
examples\build\triangle\Debug\triangle.exe
```

Windowed examples use GLFW on macOS and native Win32 on Windows. See
[`examples/README.md`](examples/README.md) for the full build matrix.

The C sources are written to modern C17 (strict ISO, no compiler
extensions).

## Shaders

By default, shaders are authored in **WGSL** and compiled at pipeline-creation
time by naga into the backend's native language — Metal Shading Language
for Metal, SPIR-V for Vulkan.

With the `shader-passthrough` extension, callers can also hand precompiled
SPIR-V or MSL straight to the backend; see
[Shader passthrough](#shader-passthrough--shader-passthrough) above.

## Quality

- **Validation-tested**: the WebGPU validation rules are exercised by an
  extensive suite that runs on the Noop backend with no GPU, so correctness
  checks need no hardware.
- **Unit-tested public API**: every public function across the three crates
  has a direct unit test.
- **Real-GPU end-to-end tests**: buffer/texture/compute/render paths are
  verified against live Metal and Vulkan devices, including the tiled
  two-subpass G-buffer path.
- **Platform coverage**:
  - **macOS** — builds, unit tests, real-GPU end-to-end tests, and the C
    examples all verified (Metal and Vulkan/MoltenVK; MoltenVK does not
    support the native subpass-input read path used by `tiled_deferred`).
  - **Windows (MSVC)** — builds and passes the full unit-test suite; the
    windowed and tiled C examples are runtime-verified against native
    Vulkan drivers.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
