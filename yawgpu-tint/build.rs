//! Builds the Tint C shim (and the minimal Tint libraries it links) from a Dawn
//! checkout, then emits the link directives so the `yawgpu-tint` crate can call
//! the shim over FFI.
//!
//! Phase 1: the Dawn source is located via the `YAWGPU_DAWN_DIR` environment
//! variable (a full Dawn checkout with `third_party` synced). When it is unset
//! the crate builds as a **stub** — the Tint FFI is `cfg`-gated out and the
//! public functions return an error — so `cargo build/test --workspace` keeps
//! working without a C++ toolchain or a Dawn checkout. Phase 1b will vendor Dawn
//! as a pinned submodule and make the Tint build unconditional.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=YAWGPU_DAWN_DIR");
    println!("cargo:rerun-if-changed=shim/tint_shim.cpp");
    println!("cargo:rerun-if-changed=shim/tint_shim.h");
    println!("cargo:rerun-if-changed=shim/CMakeLists.txt");
    // Declared so the `have_tint` cfg below does not trip the unexpected-cfg lint.
    println!("cargo:rustc-check-cfg=cfg(have_tint)");

    let Ok(dawn_dir) = env::var("YAWGPU_DAWN_DIR") else {
        println!(
            "cargo:warning=YAWGPU_DAWN_DIR not set; yawgpu-tint built as a stub \
             (Tint FFI unavailable). Set it to a Dawn checkout to enable Tint."
        );
        return;
    };

    // `build_target("tint_shim")` builds only the shim and its transitive Tint
    // dependencies (the Dawn subdirectory is added EXCLUDE_FROM_ALL in the shim
    // CMakeLists), keeping the build to the minimal reader+writers+inspector.
    let dst = cmake::Config::new("shim")
        .define("YAWGPU_DAWN_DIR", &dawn_dir)
        .build_target("tint_shim")
        .build();

    // The cmake crate places the configured build tree under `<OUT_DIR>/build`.
    let build_dir = dst.join("build");
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=dylib=tint_shim");
    // Locate libtint_shim at runtime (it is built next to the crate's artifacts).
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir.display());
    println!("cargo:rustc-cfg=have_tint");
}
