# yawgpu-dawn

This crate supplies an immutable `DawnProcTable` for the repository's pinned Dawn revision. The default build is empty. Enable `dawn-compat` to build the static library and Rust accessor. The feature uses Python 3, the initialized Dawn submodule and its vendored dependencies, a host C11 compiler for the ABI guard, and a C++20 toolchain.

## Build the host library and consumer package

Run these commands from the repository root for a macOS host using Metal.

```sh
cargo build -p yawgpu-dawn --release --features dawn-compat,metal
export DAWN_API_DIR="$PWD/target/dawn-api"
python3 tools/dawn_compat/make_dawn_api_package.py --out "$DAWN_API_DIR"
```

Use `dawn-compat` alone for Noop or `dawn-compat,vulkan` for Vulkan. Link the host with `target/release/libyawgpu_dawn.a` (on MSVC, `yawgpu_dawn.lib`), the C++ runtime and yawgpu's native dependencies. The archive includes yawgpu's Rust implementation. The Tint build produces a separate runtime library. On macOS, link against `libtint_shim.dylib` and place it beside the loading image.

Include `yawgpu-dawn/include/yawgpu_dawn.h` in the host and call `yawgpuDawnGetProcTable()`. Rust callers can use `yawgpu_dawn::dawn_proc_table()`. The pointer stays valid for the life of the process. A C++ consumer compiles the package's two sources with its `include/` directory and installs the pointer with `dawnProcSetProcs(table)` before calling the WebGPU API. Generate the package and library from the same checkout. The consumer aborts on a version mismatch.

For a null descriptor or a chain without a backend selector, the table's `createInstance` defaults to Metal on Apple targets built with `metal`, Vulkan on other targets built with `vulkan`, and Noop otherwise. A host can change the process-wide default for subsequent instances with `yawgpuDawnSetDefaultBackend(backend)` or Rust's `yawgpu_dawn::set_default_backend(backend)`, using a `YAWGPU_INSTANCE_BACKEND_*` value. The setter is thread-safe. A `YaWGPUInstanceBackendSelect` anywhere in the descriptor chain takes precedence.

A null descriptor enables `TimedWaitAny`. An explicit descriptor enables it only when the feature is requested.

The table's `getProcAddress` returns itself for `wgpuGetProcAddress` and its own `createInstance` entry for `wgpuCreateInstance`. Loaders that resolve these names in succession use the same backend selection as direct table callers. Other names use yawgpu's lookup, which returns null for stubs.

Unsupported entry points print their name to stderr on first use. They return null handles or pointers, `WGPUStatus_Error` for status, or zero for other values. Void stubs return after logging. Set `YAWGPU_DAWN_STRICT=1` to abort at an unsupported call.

## ONNX Runtime external Dawn

In the same shell, run this command from the ONNX Runtime checkout to build its shared library with the generated package.

```sh
python3 tools/ci_build/build.py --config Release --use_webgpu --use_external_dawn \
  --build_shared_lib --skip_tests --target onnxruntime \
  --cmake_extra_defines onnxruntime_DAWN_PREBUILT_DIR="$DAWN_API_DIR"
```

ONNX Runtime compiles its own WebGPU dispatch functions. Link the yawgpu archive into the host and keep ONNX Runtime's dispatch functions private to the shared library. Both define `wgpu*` symbols.

Before creating a session, pass the table address as a decimal string in the session options and enable the WebGPU execution provider. The following function configures both with the ONNX Runtime C++ API.

```cpp
#include <cstdint>
#include <string>
#include <onnxruntime_cxx_api.h>
#include "yawgpu_dawn.h"

void configureWebGPU(Ort::SessionOptions& options) {
    const auto address = reinterpret_cast<std::uintptr_t>(yawgpuDawnGetProcTable());
    options.AddConfigEntry("ep.webgpuexecutionprovider.dawnProcTable",
                           std::to_string(address).c_str());
    options.AppendExecutionProvider("WebGPU", {});
}
```
