//! Test target for the build script's pure probe helpers (Block 98 R7).
//!
//! `build_probe.rs` is included here with `#[path]`, exactly as `build.rs`
//! includes it, so its `#[cfg(test)] mod tests` is compiled into a real libtest
//! harness and actually runs under `cargo test -p yawgpu-tint`. Only the pure
//! helpers live in that module: it pulls in no build-dependency (`cmake` is not
//! linkable from a test target), so no `cfg` gating of `build.rs` is needed.

#[path = "../build_probe.rs"]
mod build_probe;
