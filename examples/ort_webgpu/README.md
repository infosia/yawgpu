# ort_webgpu — ONNX Runtime's WebGPU provider on yawgpu

A host that runs ONNX Runtime's WebGPU execution provider on yawgpu through
the Dawn compatibility layer (`yawgpu-dawn`). It is not part of the CMake
example build: it needs an ONNX Runtime checkout (`$ORT`) built against
yawgpu's Dawn API package. Verified with ONNX Runtime 1.32.0 on macOS / Metal.

## Build

```sh
# 1. The Dawn API package ONNX Runtime compiles against.
python3 tools/dawn_compat/make_dawn_api_package.py --out "$PWD/target/dawn-api"

# 2. ONNX Runtime in external Dawn mode (build.py needs Python 3.10 or later).
(cd "$ORT" && python3 tools/ci_build/build.py --build_dir "$ORT_OUT" --config Release \
    --use_webgpu --use_external_dawn --build_shared_lib --skip_tests --parallel \
    --compile_no_warning_as_error --target onnxruntime \
    --cmake_extra_defines onnxruntime_DAWN_PREBUILT_DIR="$PWD/target/dawn-api")

# 3. yawgpu with the proc table, as a static library.
cargo rustc -p yawgpu-dawn --features dawn-compat,metal --release --lib --crate-type staticlib

# 4. The host.
ORT="$ORT" ORT_BUILD="$ORT_OUT/Release" examples/ort_webgpu/build.sh
```

The package, the proc table and the ABI of the linked yawgpu must come from the
same yawgpu checkout: ONNX Runtime aborts when the table's Dawn version differs
from the package's.

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
