# yawgpu (Yet Another wgpu)

A from-scratch implementation of the **WebGPU C API** (`webgpu.h`) in Rust.

yawgpu lets native applications written in C, C++, or any language with a C
FFI talk to the GPU through the standard WebGPU interface — the same
`wgpuCreateInstance` / `wgpuDeviceCreateBuffer` / `wgpuQueueSubmit` surface
that browsers expose to WebAssembly — without a browser, a JavaScript engine,
or a web runtime.

On top of the standard `webgpu.h`, yawgpu ships a small companion header
[`yawgpu.h`](yawgpu/ffi/webgpu-headers/yawgpu.h) for backend selection and a
handful of yawgpu-specific calls. See
**[The `yawgpu.h` companion header](#the-yawgpuh-companion-header)** below.

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

The only third-party GPU-stack dependency is **Tint** — Dawn's WGSL compiler —
used to translate WGSL shaders into the platform shading languages (MSL, SPIR-V,
GLSL ES) and to reflect their interface. Everything else — validation, the
object model, the backends — is original code. Because Tint is also the compiler
the WebGPU CTS's reference implementation (Dawn) uses, yawgpu's shader
translation matches the conformance oracle **by construction**. yawgpu runs the
**entire ported WebGPU CTS** with **zero failures and zero crashes** on both
native Metal and native Vulkan (see
[Independent conformance](#independent-conformance--webgpu-native-cts)).

This is a deliberately different point in the design space from a thin C
shim layered over an existing Rust GPU engine: yawgpu owns the whole
pipeline from the C call down to the native graphics API, which keeps the
implementation legible and self-contained.

## Architecture

yawgpu is a small Cargo workspace of layered crates:

```
        C / C++ application
                │  webgpu.h  (standard WebGPU C ABI)
                │  yawgpu.h  (companion header: backend selection)
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
| **OpenGL ES** | 2 — experimental | Opt-in `gles` feature (never in default). Targets Android (EGL), Windows (ANGLE or the host GL driver), and desktop Linux (EGL). Best-effort: paths that do not cleanly map to GLES 3.1 are rejected with a device error. |

Direct3D is intentionally out of scope.

OpenGL ES platform notes:

- **Windows** — ANGLE by default. yawgpu does not bundle ANGLE: place an
  ES 3.1-capable `libEGL.dll` / `libGLESv2.dll` on the DLL search path or set
  `YAWGPU_ANGLE_PATH=<dir>`. If the available ANGLE caps at ES 3.0, set
  `YAWGPU_GLES_BACKEND=wgl` to use the host GL driver instead.
- **Linux** — system EGL, headless (windowed presentation is not wired). yawgpu
  opens the first usable hardware device through `EGL_PLATFORM_DEVICE_EXT`
  rather than the default display, which can silently resolve to a software
  rasterizer. Override with `YAWGPU_GLES_EGL_DEVICE` (`auto`, `default`,
  `software`, or an enumeration index).

A backend is chosen at instance-creation time through `YaWGPUInstanceBackendSelect`
(see below) — applications that only ever want validation can run entirely
on Noop with no GPU present.

## The `yawgpu.h` companion header

yawgpu is overwhelmingly the **standard** WebGPU C API — the goal is
Dawn-equivalent behaviour, not a divergent surface. `yawgpu.h` is a small
companion header that sits next to `webgpu.h` for the few things the standard
API has no place for: choosing a HAL backend at instance-creation time, plus a
couple of yawgpu-specific calls. It follows a strict naming convention so these
symbols never collide with the standard WebGPU C API:

| Kind | Prefix | Example |
|---|---|---|
| Functions | `yawgpu*` | `yawgpuDeviceCreateExternalTexture` |
| Types / structs / enums / handles | `YaWGPU*` | `YaWGPUInstanceBackendSelect` |
| Constants / macros / SType tags | `YAWGPU_*` / `YAWGPU_STYPE_*` | `YAWGPU_STYPE_INSTANCE_BACKEND_SELECT` |

Every descriptor ships a matching `YAWGPU_*_INIT` zero/sentinel initializer
macro, mirroring `webgpu.h` ergonomics. The default build exposes the
standard WebGPU C API plus backend selection.

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

Unlike the standard `WGPURequestAdapterOptions.backendType` hint, which only
filters adapters per request, this initializes **only the chosen backend's
runtime** — `_NOOP` touches no GPU driver at all — and adapter enumeration
returns adapters from exactly that backend. If the backend isn't compiled in or
isn't usable on the host, enumeration comes back empty; there is no silent
fallback. The library itself never reads `YAWGPU_BACKEND` — only the bundled
examples do.

For GLES, an additional chain entry `YaWGPUGlesContextBackend` pins the context
backend to EGL or WGL (Windows-only) programmatically, taking precedence over
the `YAWGPU_GLES_BACKEND` environment variable.

### External textures (Metal, experimental)

`yawgpuDeviceCreateExternalTexture` creates a `WGPUExternalTexture` from one
plane (RGBA) or two planes (NV12, with YUV→RGB conversion). It is experimental
and implemented on **Metal only**; on Vulkan, a pipeline that uses an external
texture is rejected deterministically with a device error.

### Tiled rendering (TBDR / multi-subpass) — `tiled`

On tile-based deferred renderers (Apple GPUs; mobile Vulkan) a multi-pass
pipeline such as a deferred G-buffer can keep intermediate attachments **in tile
memory** instead of round-tripping to system RAM. WebGPU's core API has no
concept of subpasses, so yawgpu exposes them as vendor entry points behind the
**`tiled` cargo feature** (default off; the `yawgpu.h` declarations are guarded
by `YAWGPU_HAS_TILED`), on both Metal and Vulkan:

- `yawgpuAdapterGetTiledCapabilities` and the `YaWGPUFeatureName_MultiSubpass`
  vendor feature report support and limits.
- A reusable `YaWGPUSubpassPassLayout` describes the attachments, subpasses, and
  input-attachment mapping; `yawgpuDeviceCreateSubpassRenderPipeline` builds a
  pipeline for one subpass of it.
- `yawgpuCommandEncoderBeginSubpassRenderPass` → draws →
  `yawgpuSubpassRenderPassEncoderNextSubpass` → draws → `…End` records the pass.
- Shaders read an earlier subpass's output through Tint's `input_attachment<T>`
  / `inputAttachmentLoad`; the resource is wired automatically from the pass
  layout.
- Attachments with the `TransientAttachment` usage are memoryless (on-tile, no
  DRAM backing).

The same C code and shaders run unchanged on both backends. See
`examples/tiled_deferred` and `examples/tiled_msaa`, and
[`yawgpu.h`](yawgpu/ffi/webgpu-headers/yawgpu.h) for the full API.

## Using it from C

**Prerequisite — the Tint shader compiler.** yawgpu's shader frontend is Tint,
built from a vendored Dawn checkout (a pinned git submodule), so the build needs a
C++20 toolchain + CMake and a one-time submodule setup:

```sh
git submodule update --init third_party/dawn
cd third_party/dawn && python3 tools/fetch_dawn_dependencies.py && cd ../..
```

(On Windows, invoke the fetch script with `python` if `python3` resolves to the
Microsoft Store stub.)

`yawgpu-tint/build.rs` then builds the minimal Tint libraries from source on the
first `cargo build` (cached afterwards). Without this setup the `yawgpu-tint`
crate compiles as a non-functional stub: it still links, but every shader
compilation fails at run time (`yawgpu_tint::HAVE_TINT` is `false`). Completing
the setup and re-running `cargo build` switches to the real compiler.

On Windows (MSVC) the Tint shim is a shared library, so an application shipping
the yawgpu `.dll` must distribute `tint_shim.dll` alongside it.

Build the library and link against it with the vendored headers:

```sh
# Noop-only (no GPU dependencies)
cargo build -p yawgpu --release

# with a real backend
cargo build -p yawgpu --release --features metal      # Apple
cargo build -p yawgpu --release --features vulkan     # Vulkan / MoltenVK
cargo build -p yawgpu --release --features gles       # Android / Windows ANGLE (Tier 2)
```

This produces `libyawgpu.{a,dylib,so}` (and a Windows `.dll`). Include
`yawgpu/ffi/webgpu-headers/webgpu.h` (and `yawgpu.h` for backend
selection), link the library, and call the standard `wgpu*` functions.

### Cross-building for Android

Both the Vulkan and OpenGL ES backends cross-build for
`aarch64-linux-android` with the Android NDK. Vulkan is the Tier 1 path on real
Android devices; GLES is the Tier 2 fallback. The build picks up the NDK's CMake
toolchain for Tint automatically — it only needs the NDK location
(`ANDROID_NDK_HOME`). `ANDROID_PLATFORM` defaults to `android-24`.

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

# GLES (Tier 2)
cargo build --release --target aarch64-linux-android -p yawgpu --features gles
```

On Linux hosts replace `darwin-x86_64` with `linux-x86_64`. The
target API level (`24` above) is the Vulkan / GLES 3.1 floor;
raise it if a dependency demands a newer one.

### Cross-building for iOS

The Metal backend cross-builds for both real iOS devices and the Apple-Silicon
iOS simulator from a macOS host with Xcode; no extra toolchain setup is
required.

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim

# Real iOS devices (arm64)
cargo build --release --target aarch64-apple-ios     -p yawgpu --features metal

# iOS Simulator on Apple Silicon
cargo build --release --target aarch64-apple-ios-sim -p yawgpu --features metal
```

## Using it from Rust

The same entry points are available as a normal Rust crate (`rlib`); the
generated bindings are exposed under `yawgpu::native`, and the
`extern "C"` functions are callable directly.

## Examples

The `examples/` directory contains small **C programs** built with CMake that
link against `libyawgpu` and exercise the standard `webgpu.h` API:

| Example | What it shows | Requires |
|---|---|---|
| `enumerate_adapters` | Listing adapters and their properties | core |
| `device_info` | Querying adapter/device limits and features | core |
| `compute` | A storage-buffer compute dispatch with readback | core |
| `capture` | Offscreen render → texture → buffer readback → PNG file | core |
| `surface_smoke` | Opening a window and presenting cleared frames | core |
| `triangle` | A classic windowed RGB-gradient triangle (vertex-index shader) | core |
| `hello_triangle` | The same RGB-gradient triangle fed from an interleaved (position + color) vertex buffer | core |
| `triangle_passthrough` | The same triangle fed **native bytecode** (SPIR-V / MSL) via the opt-in `shader-passthrough` feature | `-DYAWGPU_SHADER_PASSTHROUGH=ON` |
| `tiled_deferred` | Two-subpass deferred shading with a memoryless G-buffer read through input attachments ([`tiled` vendor extension](#tiled-rendering-tbdr--multi-subpass--tiled)) | `-DYAWGPU_TILED=ON` (Metal / Vulkan) |
| `tiled_msaa` | Per-sample MSAA subpass input with a custom in-shader resolve | `-DYAWGPU_TILED=ON` (Vulkan-only) |

Pick a backend at runtime with `YAWGPU_BACKEND`:

```sh
# macOS
brew install cmake glfw
cmake -S examples -B examples/build
cmake --build examples/build

YAWGPU_BACKEND=metal  ./examples/build/triangle/triangle
YAWGPU_BACKEND=vulkan ./examples/build/compute/compute
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

The `triangle_passthrough` example is **opt-in**: pass
`-DYAWGPU_SHADER_PASSTHROUGH=ON` at configure time (which also enables the
`shader-passthrough` cargo feature). It runs on Metal and Vulkan only and
self-skips elsewhere.

Windowed examples use GLFW on macOS and native Win32 on Windows. See
[`examples/README.md`](examples/README.md) for the full build matrix.

The C sources are written to modern C17 (strict ISO, no compiler
extensions).

## Shaders

By default, shaders are authored in **WGSL** and compiled at pipeline-creation
time by Tint into the backend's native language — Metal Shading Language
for Metal, SPIR-V for Vulkan, GLSL ES for GLES.

16-bit floats are supported through the standard WebGPU **`shader-f16`**
optional feature, and SIMD-lane collective operations through **`subgroups`**.
Like every optional feature, each must be requested in the device's
`requiredFeatures` (then `enable f16;` / `enable subgroups;` in WGSL); using one
without requesting it is a validation error.

The optional features yawgpu supports on the Tier-1 backends (none are available
on the Tier-2 GLES backend):

| Feature | What it enables |
|---|---|
| `shader-f16` | `f16` types in WGSL, including in storage/uniform buffers |
| `subgroups` | subgroup builtins (`subgroupAdd`, `subgroupBroadcast`, …); `WGPUAdapterInfo` reports the subgroup size range |
| `depth-clip-control` | `primitive.unclippedDepth` — clamp instead of clip at the near/far planes |
| `float32-blendable` | blend state on `r32float` / `rg32float` / `rgba32float` color targets |
| `dual-source-blending` | a second fragment color output (`@blend_src`) and the `src1` blend factors |
| `indirect-first-instance` | non-zero `firstInstance` in indirect draws |
| `clip-distances` | `@builtin(clip_distances)` user-defined clip planes |
| `primitive-index` | `@builtin(primitive_index)` in fragment shaders |
| `texture-component-swizzle` | per-view `r/g/b/a` component remapping on sampled texture views |

Each is advertised only when the underlying device supports it, matching Dawn.

### Native shader passthrough (vendor, opt-in, unsafe)

For engines that ship **precompiled native shaders**, the opt-in
`shader-passthrough` cargo feature (default **off**) lets you create a
`WGPUShaderModule` directly from raw **SPIR-V** (Vulkan) or raw **MSL** (Metal),
bypassing WGSL and Tint entirely:

- SPIR-V uses the standard `WGPUShaderSourceSPIRV` chain (Vulkan only); MSL uses
  the vendor `YaWGPUShaderSourceMSL` chain (Metal only). A module is rejected if
  used on the other backend.
- This is a **vendor escape hatch that leaves the WebGPU spec behind** — the
  bytes reach the driver verbatim with no validation or reflection, so it is
  inherently **unsafe** and the caller owns correctness. It is never exercised by
  the CTS.
- Because there is no reflection, an **explicit pipeline layout is required**
  (no `layout: "auto"`), binding slots are taken from that layout, and the
  caller's shader must match yawgpu's deterministic slot ABI (documented in
  [`yawgpu.h`](yawgpu/ffi/webgpu-headers/yawgpu.h)). Compute and render
  (vertex + fragment) pipelines are supported on both Tier-1 backends.

## Quality

- **Validation-tested**: the WebGPU validation rules are exercised by an
  extensive suite that runs on the Noop backend with no GPU, so correctness
  checks need no hardware.
- **CTS conformance**: yawgpu is verified case-by-case against the official
  [WebGPU Conformance Test Suite](https://github.com/gpuweb/cts) through
  [webgpu-native-cts](https://github.com/infosia/webgpu-native-cts), with
  **Dawn** as the conformance oracle. The full ported suite — about 2.1 million
  subcases — runs **`fail = 0`, `crash = 0`** on both native backends — see
  [Independent conformance](#independent-conformance--webgpu-native-cts) below.
- **Unit-tested public API**: every public function across the three crates
  has a direct unit test.
- **Real-GPU end-to-end tests**: buffer/texture/compute/render paths are
  verified against live Metal and Vulkan devices, and the Vulkan backend runs
  clean under `VK_LAYER_KHRONOS_validation`. The OpenGL ES backend (Tier 2) is
  verified end-to-end on real GPUs on Windows and Linux.
- **Platform coverage**:
  - **macOS** — builds, unit tests, real-GPU end-to-end tests, and the C
    examples all verified (Metal and Vulkan/MoltenVK).
  - **Windows (MSVC)** — builds and passes the full unit-test suite. The
    **Vulkan backend is verified real-GPU** against a native NVIDIA driver: the
    `e2e_vulkan_*` suite, the windowed C examples, and the **entire
    webgpu-native-cts ported suite** pass. The `triangle` example also runs
    through the OpenGL ES backend (host GL driver via WGL).
  - **Linux (`x86_64-unknown-linux-gnu`)** — the CI host: every push builds the
    workspace and runs the full unit + validation test suite on the Noop backend
    (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)). The **Vulkan
    backend is verified real-GPU** on NVIDIA (proprietary driver) and AMD
    (RADV / Mesa): the full `e2e_vulkan_*` suite and the C examples pass on
    both, and the **entire webgpu-native-cts ported suite** runs with zero open
    defects and zero crashes. The OpenGL ES backend (Tier 2) passes its
    `e2e_gles_*` suite on NVIDIA, AMD, and Mesa llvmpipe. Windowed presentation
    (X11 / Wayland) is not yet wired.
  - **Android (`aarch64-linux-android`)** — both Vulkan and OpenGL ES backends
    cross-build with the NDK. Real-device runtime verification is left to
    downstream integrators.
  - **iOS (`aarch64-apple-ios` + `aarch64-apple-ios-sim`)** — the Metal backend
    cross-builds for both real devices and the Apple-Silicon simulator.
    Real-device runtime verification is left to downstream integrators.

### Independent conformance — webgpu-native-cts

yawgpu is the **primary conformance subject** of
[**webgpu-native-cts**](https://github.com/infosia/webgpu-native-cts) — a C++20
suite that ports the upstream WebGPU CTS and links *directly* against the
`webgpu.h` C ABI (no JavaScript engine), running every case in its own
subprocess (`--isolate`) against a real GPU. **Dawn**, Google's C++ reference
implementation, is the **oracle** every result is judged against.

Across the **entire ported suite** — `api/validation`, `api/operation`,
`shader/validation`, and `shader/execution` — yawgpu runs on real hardware with
**zero open implementation defects**, matching the Dawn oracle. The tables below
are **raw** (no expectations applied), so documented non-defects carried as
`xfail` show in `fail` with a note rather than being masked.

**Native Metal** (macOS / Apple, Tint frontend), per-subcase:

| area | pass | skip | fail | crash |
|---|---:|---:|---:|---:|
| `api/validation` | 292,960 | 61,655 | 2† | 0 |
| `api/operation` | 228,600 | 993 | 0 | 0 |
| `shader/execution` | 822,209 | 22,377 | 0 | 0 |
| `shader/validation` | 646,773 | 20,369 | 0 | 0 |
| **total** | **1,990,542** | **105,394** | **2†** | **0** |

† The same 2 `draw,index_buffer_format_dirtying` cases the Dawn oracle rejects
identically — a CTS port quirk, not a yawgpu defect.

**Native Vulkan** (Windows 11 / NVIDIA RTX 5060 Ti, Tint frontend), per-subcase:

| area | pass | skip | fail | crash |
|---|---:|---:|---:|---:|
| `api/validation` | 235,547 | 119,070 | 4‡ | 0 |
| `api/operation` | 209,108 | 20,487 | 0 | 0 |
| `shader/execution` | 515,870 | 328,603 | 113‡ | 0 |
| `shader/validation` | 646,773 | 20,369 | 0 | 0 |
| **total** | **1,607,298** | **488,529** | **117‡** | **0** |

**Native Vulkan** (Linux / NVIDIA RTX 5060 Ti, Tint frontend), per-subcase:

| area | pass | skip | fail | crash |
|---|---:|---:|---:|---:|
| `api/validation` | 244,676 | 110,316 | 4‡ | 0 |
| `api/operation` | 209,360 | 20,233 | 0 | 0 |
| `shader/execution` | 531,304 | 313,170 | 113‡ | 0 |
| `shader/validation` | 646,773 | 20,369 | 0 | 0 |
| **total** | **1,632,113** | **464,088** | **117‡** | **0** |

‡ On both hosts, all 117 fails are documented non-defects, each cross-checked
against a Dawn-Vulkan oracle on the same GPU (e.g. the spec-in-flux per-sample
`sample_mask` builtin, and `external_texture` being Metal-only by design). They
are carried as `xfail`, so the suite exits `fail = 0`.

Shader conformance falls out **by construction**: yawgpu compiles WGSL with the
same Tint compiler Dawn uses, so the entire `shader/*` surface is
Dawn-equivalent with no separate shader code path to keep in sync. (MoltenVK on
macOS is a non-authoritative Vulkan path and is not part of these results.)

**Native GLES — Tier 2 / experimental**, per-subcase:

> **Not a conformance table.** GLES is opt-in and experimental; these are raw
> bring-up snapshots, and numbers may change without SemVer guarantees.

Linux / NVIDIA RTX 5060 Ti, OpenGL ES 3.2:

| area | pass | skip | fail | crash |
|---|---:|---:|---:|---:|
| `api/validation` | 203,412 | 151,323 | 257 | 0 |
| `api/operation` | 141,076 | 76,704 | 11,813 | 0 |
| `shader/execution` | 309,535 | 517,406 | 17,645 | 0 |
| `shader/validation` | 369,753 | 297,389 | 0 | 0 |
| **total** | **1,023,776** | **1,042,822** | **29,715** | **0** |

Linux / Mesa `crocus` on Intel Haswell:

| area | pass | skip | fail | crash |
|---|---:|---:|---:|---:|
| `api/validation`§ | 194,827 | 157,163 | 325 | 0 |
| `api/operation` | 149,727 | 76,698 | 3,036 | 0 |
| `shader/execution` | 315,602 | 516,424 | 2,900 | 0 |
| `shader/validation` | 369,753 | 297,389 | 0 | 0 |
| **total** | **1,029,909** | **1,047,674** | **6,261** | **0** |

§ 2 `api/validation` files are quarantined on this host because of a driver
defect that hangs the GPU.

The failures are dominated by catalogued Tier-2 boundaries — GLES hardware/spec
limits with no clean WebGPU mapping — listed in
[`specs/blocks/67-gles-backend.md`](specs/blocks/67-gles-backend.md).

Per-case results and cross-backend differences are tracked in the suite's
[`docs/FINDINGS.md`](https://github.com/infosia/webgpu-native-cts/blob/main/docs/FINDINGS.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
