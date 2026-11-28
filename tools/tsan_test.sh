#!/bin/sh
# Runs cargo tests under ThreadSanitizer with both the Rust crates and the C++ Tint shim
# instrumented (macOS, Apple Clang, nightly Rust with the rust-src component).
#
#   tools/tsan_test.sh -p yawgpu --test concurrent_device
#   tools/tsan_test.sh -p yawgpu-core --lib
#
# The arguments are passed to `cargo test`. A data race makes the test process exit
# non-zero after printing "WARNING: ThreadSanitizer: data race".
#
# How it is set up:
# - One TSan runtime for the whole process. `-fsanitize=thread` makes Clang link its own
#   runtime into the shim, and rustc links a bundled one into Rust binaries; two runtimes
#   in one process abort with "Interceptors are not working". `-Zexternal-clangrt` drops
#   rustc's, and the link arguments point Rust binaries at Clang's.
# - `-Zbuild-std` rebuilds the standard library instrumented; it needs network access the
#   first time to fetch the standard library's dependencies.
# - A dedicated target directory (`target-tsan`), so instrumented objects never mix with
#   a normal build. The Tint shim's native tree is keyed by the compiler flags; after
#   changing CFLAGS / CXXFLAGS by hand, remove
#   `target-tsan/*/debug/build/yawgpu-tint` to force a reconfigure.
# - `allocator_may_return_null=1`: tests that probe allocation failure request sizes the
#   TSan allocator otherwise aborts on.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUNTIME=$(clang -print-file-name=libclang_rt.tsan_osx_dynamic.dylib)
[ -f "$RUNTIME" ] || { echo "Clang's TSan runtime was not found: $RUNTIME" >&2; exit 1; }
HOST=$(rustc +nightly -vV | sed -n 's/^host: //p')

cd "$ROOT"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target-tsan}" \
RUSTFLAGS="-Zsanitizer=thread -Zexternal-clangrt -Clink-arg=$RUNTIME -Clink-arg=-Wl,-rpath,$(dirname "$RUNTIME")" \
CFLAGS="-fsanitize=thread -g" \
CXXFLAGS="-fsanitize=thread -g" \
TSAN_OPTIONS="${TSAN_OPTIONS:-allocator_may_return_null=1}" \
  exec cargo +nightly test -Zbuild-std --target "$HOST" "$@"
