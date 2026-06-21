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
│               Noop · Metal · Vulkan · GLES*      │
└───────────────────────────────────────────────┘
                │               │             │
              Metal           Vulkan      OpenGL ES*
                                          (experimental)
```

\* OpenGL ES is opt-in / Tier 2 — see Backends below.

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

| Backend | Tier | Notes |
|---|---|---|
| **Noop** | reference | CPU-only; always available. Runs the full validation layer with no GPU. Ideal for CI and headless testing. |
| **Metal** | 1 — supported | Apple platforms. Built with the `metal` feature via the `objc2` family. |
| **Vulkan** | 1 — supported | Cross-platform. Built with the `vulkan` feature via `ash`; targets **Vulkan 1.1+** (MoltenVK ≥ 1.1 on macOS, native drivers on Linux / Windows / Android). |
| **OpenGL ES** | 2 — experimental | Opt-in `gles` feature (never in default). Targets Android (native EGL) and Windows (ANGLE by default, host GL via opt-in `YAWGPU_GLES_BACKEND=wgl`). Best-effort: paths that do not cleanly map to GLES 3.1 are rejected at the HAL layer with `HalError`; `yawgpu.h` vendor extensions (`tiled`, `shader-passthrough`) are not implemented for GLES. |

Direct3D is intentionally out of scope.

The OpenGL ES backend goes through ANGLE by default on Windows
(`libEGL.dll` / `libGLESv2.dll` — yawgpu does not bundle ANGLE; the
caller must place an ES 3.1-capable build on the DLL search path or
set `YAWGPU_ANGLE_PATH=<dir>` to preload from). On systems where the
locally available ANGLE binary caps at ES 3.0 (Chromium / CEF builds
do), set `YAWGPU_GLES_BACKEND=wgl` to bypass ANGLE and use the host
GL driver via `WGL_EXT_create_context_es2_profile` — verified on
NVIDIA / AMD / Intel desktop drivers.

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

Chain `YaWGPUInstanceBackendSelect` onto `WGPUInstanceDescriptor` to pin
the instance to a single HAL backend at creation time:

```c
#include "webgpu.h"
#include "yawgpu.h"

YaWGPUInstanceBackendSelect sel = {
    .chain   = { .sType = YAWGPU_STYPE_INSTANCE_BACKEND_SELECT },
    .backend = YAWGPU_INSTANCE_BACKEND_METAL,   /* or _VULKAN, _GLES, _NOOP */
};
WGPUInstanceDescriptor desc = { .nextInChain = &sel.chain };
WGPUInstance instance = wgpuCreateInstance(&desc);
```

This is distinct from the standard `WGPURequestAdapterOptions.backendType`
hint, which filters adapters per-request after every backend's runtime is
already up. `YaWGPUInstanceBackendSelect` decides at `wgpuCreateInstance`
time, so:

- **Only the chosen backend's runtime is initialized.** `_NOOP` truly
  touches no GPU driver — useful for validation-only CI; `_METAL` skips
  the Vulkan ICD scan, and vice versa.
- **No silent multi-backend leakage.** `wgpuInstanceEnumerateAdapters`
  returns adapters from exactly one backend, which keeps e2e tests
  (`yawgpu/tests/e2e_metal_*.rs`, `e2e_vulkan_*.rs`, `e2e_gles_*.rs`)
  isolated and lets a single process compare backends side-by-side by
  holding two pinned instances at once.
- **Fully programmatic.** The library does not read `YAWGPU_BACKEND`
  itself — only the bundled examples' framework does, as a convenience
  for `./example` CLI invocations. Applications that want backend
  control without env-var coupling get a clean C API path.

If the requested backend isn't compiled in or isn't usable on the host,
adapter enumeration comes back empty so the caller sees the failure
immediately (no silent fallback to a different backend).

When the resolved instance backend is GLES, an independent chain entry
`YaWGPUGlesContextBackend` (sType `YAWGPU_STYPE_GLES_CONTEXT_BACKEND`)
can additionally pin the GLES context backend to EGL or WGL without
touching the `YAWGPU_GLES_BACKEND` environment variable:

```c
YaWGPUGlesContextBackend ctx = YAWGPU_GLES_CONTEXT_BACKEND_INIT;
ctx.contextBackend = YAWGPU_GLES_CONTEXT_BACKEND_WGL; /* or _EGL, or _DEFAULT */

YaWGPUInstanceBackendSelect sel = {
    .chain   = { .next = &ctx.chain, .sType = YAWGPU_STYPE_INSTANCE_BACKEND_SELECT },
    .backend = YAWGPU_INSTANCE_BACKEND_GLES,
};
WGPUInstanceDescriptor desc = { .nextInChain = &sel.chain };
WGPUInstance instance = wgpuCreateInstance(&desc);
```

Resolution order is: a non-default chain value wins; `..._DEFAULT` (or
no chain entry) defers to `YAWGPU_GLES_BACKEND`; otherwise the
default EGL path is taken. `YAWGPU_GLES_CONTEXT_BACKEND_WGL` is
Windows-only and falls back to EGL on non-Windows hosts. The entry
is ignored when the resolved instance backend is not GLES.

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
cargo build -p yawgpu --release --features gles       # Android / Windows ANGLE (Tier 2)

# with vendor extensions
cargo build -p yawgpu --release --features "vulkan tiled"
cargo build -p yawgpu --release --features "metal mobile"   # = shader-passthrough + tiled
```

This produces `libyawgpu.{a,dylib,so}` (and a Windows `.dll`). Include
`yawgpu/ffi/webgpu-headers/webgpu.h` (and `yawgpu.h` for the vendor
extensions), link the library, and call the standard `wgpu*` /
`yawgpu*` functions.

### Cross-building for Android

Both the Vulkan and OpenGL ES backends cross-build for
`aarch64-linux-android` (verified 2026-05-25 from a macOS arm64
host with NDK r30). Vulkan is the Tier 1 path on real Android
devices; GLES is the Tier 2 fallback.

```sh
rustup target add aarch64-linux-android

export ANDROID_NDK_HOME=/path/to/ndk           # e.g. ~/Library/Android/sdk/ndk/30.0.14904198
export NDK_BIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin"
export SYSROOT="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/sysroot"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$NDK_BIN/aarch64-linux-android24-clang"
export CC_aarch64_linux_android="$NDK_BIN/aarch64-linux-android24-clang"
export CXX_aarch64_linux_android="$NDK_BIN/aarch64-linux-android24-clang++"
export AR_aarch64_linux_android="$NDK_BIN/llvm-ar"
# Load-bearing: without this, build.rs's bindgen pass over webgpu.h
# can't find <math.h> and the build fails.
export BINDGEN_EXTRA_CLANG_ARGS_aarch64_linux_android="--target=aarch64-linux-android24 --sysroot=$SYSROOT"

# Vulkan (Tier 1) — ash dynamically loads libvulkan.so at runtime,
# which Android 7.0+ (API 24+) ships with the platform.
cargo build --release --target aarch64-linux-android -p yawgpu --features vulkan
cargo build --release --target aarch64-linux-android -p yawgpu --features "vulkan mobile"

# GLES (Tier 2)
cargo build --release --target aarch64-linux-android -p yawgpu --features gles
```

On Linux hosts replace `darwin-x86_64` with `linux-x86_64`. The
target API level (`24` above) is the Vulkan / GLES 3.1 floor;
raise it if a dependency demands a newer one.

### Cross-building for iOS

The Metal backend cross-builds for both real iOS devices and
the Apple-Silicon iOS simulator (verified 2026-05-25 from a
macOS arm64 host with Xcode 26.5 / iOS SDK 26.5). No extra
toolchain setup is required — the Apple `cc` / linker found
via `xcrun` handles sysroot resolution automatically, and
`build.rs`'s bindgen pass picks up the right Apple SDK on its
own.

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim

# Real iOS devices (arm64)
cargo build --release --target aarch64-apple-ios     -p yawgpu --features metal
cargo build --release --target aarch64-apple-ios     -p yawgpu --features "metal mobile"

# iOS Simulator on Apple Silicon
cargo build --release --target aarch64-apple-ios-sim -p yawgpu --features "metal mobile"
```

Produces `libyawgpu.{a,dylib,rlib}` under
`target/aarch64-apple-ios{,-sim}/release/`. The dylibs are
tagged `LC_VERSION_MIN_IPHONEOS` (device) and `LC_BUILD_VERSION
platform=iOSSimulator` (sim).

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
| `triangle` | A classic windowed RGB-gradient triangle (vertex-index shader) | core |
| `hello_triangle` | The same RGB-gradient triangle fed from an interleaved (position + color) vertex buffer | core |
| `triangle_passthrough` | The same gradient triangle driven from precompiled SPIR-V and MSL | `shader-passthrough` |
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

To run the windowed examples through the OpenGL ES backend on Windows
(opt-in / Tier 2 — requires either an ES 3.1-capable ANGLE on PATH or
the host GL driver via WGL):

```powershell
cmake -S examples -B examples/build-gles -DYAWGPU_FEATURE=gles
cmake --build examples/build-gles
$env:YAWGPU_BACKEND = "gles"
# Default: ANGLE (libEGL.dll on PATH). If the local ANGLE caps at ES 3.0,
# bypass it and use the host GL driver instead:
$env:YAWGPU_GLES_BACKEND = "wgl"
examples\build-gles\triangle\Debug\triangle.exe
```

Windowed examples use GLFW on macOS and native Win32 on Windows. See
[`examples/README.md`](examples/README.md) for the full build matrix.

The C sources are written to modern C17 (strict ISO, no compiler
extensions).

## Shaders

By default, shaders are authored in **WGSL** and compiled at pipeline-creation
time by naga into the backend's native language — Metal Shading Language
for Metal, SPIR-V for Vulkan.

16-bit floats are supported through the standard WebGPU **`shader-f16`**
optional feature: request `WGPUFeatureName_ShaderF16` in the device's
`requiredFeatures`, then use `enable f16;` in WGSL. Shaders that use `f16`
without the feature requested are rejected with a validation error. It is
advertised on Metal (native `half`) and on Vulkan when the device exposes
`shaderFloat16` (the backend also enables `VK_KHR_16bit_storage` so `f16`
works in storage/uniform buffers, not just arithmetic); it is not available
on the Tier-2 GLES backend.

With the `shader-passthrough` extension, callers can also hand precompiled
SPIR-V or MSL straight to the backend; see
[Shader passthrough](#shader-passthrough--shader-passthrough) above.

## Quality

- **Validation-tested**: the WebGPU validation rules are exercised by an
  extensive suite that runs on the Noop backend with no GPU, so correctness
  checks need no hardware.
- **CTS conformance**: yawgpu is verified case-by-case against the official
  [WebGPU Conformance Test Suite](https://github.com/gpuweb/cts) through
  [webgpu-native-cts](https://github.com/infosia/webgpu-native-cts), which ports
  the CTS onto the `webgpu.h` C ABI and runs each case against a real GPU
  (Metal and native Vulkan), with **Dawn** as the conformance oracle — see
  [Independent conformance](#independent-conformance--webgpu-native-cts) below.
- **Unit-tested public API**: every public function across the three crates
  has a direct unit test.
- **Real-GPU end-to-end tests**: buffer/texture/compute/render paths are
  verified against live Metal and Vulkan devices, including the tiled
  two-subpass G-buffer path. The Vulkan backend runs **validation-clean**
  under `VK_LAYER_KHRONOS_validation` (zero VUID violations across the
  full `--ignored` suite). The OpenGL ES backend (Tier 2) is verified
  end-to-end on a host NVIDIA driver via the WGL fallback
  (`YAWGPU_GLES_BACKEND=wgl`), covering buffer / texture / compute /
  render e2e suites plus the windowed `triangle` example.
- **Platform coverage**:
  - **macOS** — builds, unit tests, real-GPU end-to-end tests, and the C
    examples all verified (Metal and Vulkan/MoltenVK; MoltenVK does not
    support the native subpass-input read path used by `tiled_deferred`).
  - **Windows (MSVC)** — builds and passes the full unit-test suite; the
    windowed and tiled C examples are runtime-verified against native
    Vulkan drivers, and the windowed `triangle` example additionally
    runs through the OpenGL ES backend via the WGL fallback (host GL
    driver, opt-in).
  - **Linux (`x86_64-unknown-linux-gnu`)** — the CI host (`ubuntu-latest`):
    every push builds the workspace and runs the full unit + validation
    test suite (Noop backend) green, including the `tiled` feature gate
    (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)). The
    **Vulkan backend** is additionally verified **real-GPU on Linux**
    against the native ICD (`ash` loads `libvulkan.so` at runtime) — on
    Intel Iris Graphics 5100 (HSW GT3) / Mesa / Vulkan 1.2 the basic,
    buffer, texture, compute, render, depth, external-texture, OOM, and
    two-subpass `tiled` G-buffer end-to-end suites pass against the live
    driver. The handful of non-passes are bound to this particular GPU /
    driver rather than to yawgpu: ETC2 / ASTC compressed-texture
    roundtrips need compressed-format features this desktop GPU does not
    expose, `depthCompare=Equal` is sensitive to Mesa depth-invariance
    precision, and one SPIR-V shader-passthrough render case differs on
    this driver. X11 / Wayland windowed surface sources are currently
    recognized-but-inert, so windowed presentation is not yet wired.
  - **Android (`aarch64-linux-android`)** — both Vulkan and OpenGL ES
    backends cross-build from a macOS arm64 host with NDK r30 (see
    "Cross-building for Android" above), including the `mobile`
    extension combo (`shader-passthrough` + `tiled`). Real-device
    runtime verification is left to downstream integrators.
  - **iOS (`aarch64-apple-ios` + `aarch64-apple-ios-sim`)** — the
    Metal backend cross-builds for both real devices and the
    Apple-Silicon iOS Simulator from a macOS arm64 host with Xcode
    26.5 / iOS SDK 26.5, including the `mobile` extension combo
    (see "Cross-building for iOS" above). Real-device runtime
    verification is left to downstream integrators.

### Independent conformance — webgpu-native-cts

yawgpu is the **primary conformance subject** of
[**webgpu-native-cts**](https://github.com/infosia/webgpu-native-cts) — a C++20
suite that ports the upstream WebGPU CTS and links *directly* against the
`webgpu.h` C ABI (no JavaScript engine), running every case in its own
subprocess (`--isolate`) against a real GPU. **Dawn**, Google's C++ reference
implementation, is the **oracle** every result is judged against.

The suite covers the **entire `api` surface** — all 126 `api/validation` and 70
`api/operation` files — and yawgpu runs it **green on both Tier-1 backends, on
native hardware**:

| Backend (host) | `api` surface |
|---|---|
| **Metal** — Apple Silicon | matches Dawn — all cases pass |
| **Vulkan** — native (NVIDIA) | matches Dawn — all cases pass |

The lone exception is a single draw-validation case where yawgpu is
*stricter* than Dawn — it rejects a draw Dawn allows — a deliberate, documented
choice, not a defect.

Per-case results and any cross-backend differences are tracked in the suite's
[`docs/FINDINGS.md`](https://github.com/infosia/webgpu-native-cts/blob/main/docs/FINDINGS.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
