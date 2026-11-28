#!/bin/sh
# Builds the ONNX Runtime host against yawgpu's Dawn compatibility layer (macOS / Metal).
#
#   ORT=<onnxruntime checkout> ORT_BUILD=<its build dir>/Release \
#   YAWGPU_TARGET=<cargo target dir> OUT=<output binary> examples/ort_webgpu/build.sh
#
# Prerequisites:
#   python3 tools/dawn_compat/make_dawn_api_package.py --out <package>
#   ONNX Runtime built with --use_webgpu --use_external_dawn --build_shared_lib and
#     --cmake_extra_defines onnxruntime_DAWN_PREBUILT_DIR=<package>
#   cargo rustc -p yawgpu-dawn --features dawn-compat,metal --release --lib --crate-type staticlib
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
: "${ORT:?set ORT to the ONNX Runtime checkout}"
: "${ORT_BUILD:?set ORT_BUILD to the directory holding libonnxruntime.dylib}"
YAWGPU_TARGET=${YAWGPU_TARGET:-$ROOT/target}
OUT=${OUT:-$ROOT/target/ort_webgpu}

c++ -std=c++20 -O1 "$ROOT/examples/ort_webgpu/main.cc" \
  -I"$ORT/include/onnxruntime/core/session" \
  -I"$ROOT/yawgpu/ffi/webgpu-headers" -I"$ROOT/yawgpu-dawn/include" \
  "$YAWGPU_TARGET/release/libyawgpu_dawn.a" \
  -L"$YAWGPU_TARGET/release" -ltint_shim -L"$ORT_BUILD" -lonnxruntime \
  -Wl,-rpath,"$ORT_BUILD" \
  -framework QuartzCore -framework CoreVideo -framework CoreGraphics -framework Metal \
  -framework Foundation -framework CoreFoundation -liconv -lobjc \
  -o "$OUT"
# The Tint shim is loaded from next to the binary (@loader_path).
cp -f "$YAWGPU_TARGET/release/libtint_shim.dylib" "$(dirname "$OUT")/"
echo "built $OUT"
