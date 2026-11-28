# yawgpu-dawn

This crate supplies an immutable `DawnProcTable` for the repository's pinned Dawn revision. The default build is empty. Enable `dawn-compat` to build the static library and Rust accessor. The feature uses Python 3, the initialized Dawn submodule and its vendored dependencies, and a C++20 toolchain.

## Build the host library and consumer package

Run these commands from the repository root for a macOS host using Metal.

```sh
CARGO_TARGET_DIR=target-b171a cargo build -p yawgpu-dawn --release --features dawn-compat,metal > /tmp/b171d.log 2>&1; echo EXIT=$?; tail -n 30 /tmp/b171d.log
python3 tools/dawn_compat/make_dawn_api_package.py --out /tmp/dawn-api
```

Use `dawn-compat` alone for Noop or `dawn-compat,vulkan` for Vulkan. Link the host with `target-b171a/release/libyawgpu_dawn.a` (on MSVC, `yawgpu_dawn.lib`), the C++ runtime and yawgpu's native dependencies. The archive includes yawgpu's Rust implementation. The Tint build produces a separate runtime library. On macOS, link against `libtint_shim.dylib` and place it beside the loading image.

Include `yawgpu-dawn/include/yawgpu_dawn.h` in the host and call `yawgpuDawnGetProcTable()`. Rust callers can use `yawgpu_dawn::dawn_proc_table()`. The pointer stays valid for the life of the process. A C++ consumer compiles the package's two sources with its `include/` directory and installs the pointer with `dawnProcSetProcs(table)` before calling the WebGPU API. Generate the package and library from the same checkout. The consumer aborts on a version mismatch.

Unsupported entry points print their name to stderr on first use. They return null handles or pointers, `WGPUStatus_Error` for status, or zero for other values. Void stubs return after logging. Set `YAWGPU_DAWN_STRICT=1` to abort at an unsupported call. Looking up a stub with the table's `getProcAddress` returns null.

## ONNX Runtime external Dawn

Run this command from the ONNX Runtime checkout to build its shared library with the generated package.

```sh
python3 tools/ci_build/build.py --config Release --use_webgpu --use_external_dawn \
  --build_shared_lib --skip_tests --target onnxruntime \
  --cmake_extra_defines onnxruntime_DAWN_PREBUILT_DIR=/tmp/dawn-api
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
