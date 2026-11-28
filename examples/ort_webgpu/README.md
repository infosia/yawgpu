# ort_webgpu — ONNX Runtime's WebGPU provider on yawgpu

A host that runs ONNX Runtime's WebGPU execution provider on yawgpu through
the Dawn compatibility layer (`yawgpu-dawn`). It is not part of the CMake
example build: it needs an ONNX Runtime checkout (`$ORT`) built against
yawgpu's Dawn API package. Verified with ONNX Runtime 1.32.0 on macOS / Metal
and on Windows / Vulkan.

## Build

```sh
# Run from the yawgpu checkout. $ORT is the ONNX Runtime checkout, $ORT_OUT its build dir.
DAWN_API_DIR="$PWD/target/dawn-api"

# 1. The Dawn API package ONNX Runtime compiles against.
python3 tools/dawn_compat/make_dawn_api_package.py --out "$DAWN_API_DIR"

# 2. ONNX Runtime in external Dawn mode (build.py needs Python 3.10 or later).
(cd "$ORT" && python3 tools/ci_build/build.py --build_dir "$ORT_OUT" --config Release \
    --use_webgpu --use_external_dawn --build_shared_lib --skip_tests --parallel \
    --compile_no_warning_as_error --target onnxruntime \
    --cmake_extra_defines onnxruntime_DAWN_PREBUILT_DIR="$DAWN_API_DIR")

# 3. yawgpu with the proc table, as a static library.
cargo rustc -p yawgpu-dawn --features dawn-compat,metal --release --lib --crate-type staticlib

# 4. The host.
ORT="$ORT" ORT_BUILD="$ORT_OUT/Release" examples/ort_webgpu/build.sh
```

The package, the proc table and the ABI of the linked yawgpu must come from the
same yawgpu checkout: ONNX Runtime aborts when the table's Dawn version differs
from the package's.

### Windows / Vulkan

Needs Visual Studio 2022 (C++ workload), Python 3.10 or later as `python`, and
Clang for the ABI guard that runs when `yawgpu-dawn` is built. PowerShell, from
the yawgpu checkout:

```powershell
$api = "$PWD\target\dawn-api"; $env:PYTHONUTF8 = "1"
python tools\dawn_compat\make_dawn_api_package.py --out $api

python $env:ORT\tools\ci_build\build.py --build_dir $env:ORT_OUT --config Release `
    --use_webgpu --use_external_dawn --build_shared_lib --skip_tests --parallel `
    --compile_no_warning_as_error --target onnxruntime `
    --cmake_extra_defines "onnxruntime_DAWN_PREBUILT_DIR=$($api -replace '\\','/')" `
    onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=ON onnxruntime_ENABLE_DAWN_BACKEND_D3D12=OFF

$env:YAWGPU_ABI_GUARD_CC = '"C:/Program Files/LLVM/bin/clang.exe"'
cargo rustc -p yawgpu-dawn --features dawn-compat,vulkan --release --lib --crate-type staticlib

$env:ORT_BUILD = "$env:ORT_OUT\Release\Release"
examples\ort_webgpu\build.ps1          # target\ort_webgpu\ort_webgpu.exe
```

The two backend defines make ONNX Runtime request a Vulkan adapter; its Windows
default is D3D12, which yawgpu does not have. Because that request is fixed at
build time, `--noop` does not work with a Windows build of ONNX Runtime. Add
`--skip_submodule_sync` when `$ORT` is not a git checkout.

## Run

```sh
target/ort_webgpu                    # Abs model; ONNX Runtime creates the device
target/ort_webgpu --host_device      # the host creates the instance and device
target/ort_webgpu --model m.onnx     # compare WebGPU with the CPU provider
target/ort_webgpu --noop             # Noop backend: session setup only
```

Every session runs with `session.disable_cpu_ep_fallback=1`, so a node the
WebGPU provider cannot take is an error rather than a silent CPU run.
`--model` supports models whose inputs are float32 tensors; a dynamic dimension
is taken as 1. Set `YAWGPU_DAWN_STRICT=1` to abort on the first Dawn entry point
yawgpu does not implement.

The host calls `yawgpuDawnSetDefaultBackend()` because the instance ONNX
Runtime creates through the proc table carries no yawgpu backend selector.
