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
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=YAWGPU_DAWN_DIR");
    println!("cargo:rerun-if-changed=shim/tint_shim.cpp");
    println!("cargo:rerun-if-changed=shim/tint_shim.h");
    println!("cargo:rerun-if-changed=shim/CMakeLists.txt");
    // Declared so the `have_tint` cfg below does not trip the unexpected-cfg lint.
    println!("cargo:rustc-check-cfg=cfg(have_tint)");

    let Some(dawn_dir) = resolve_dawn_dir() else {
        println!(
            "cargo:warning=No Dawn checkout found; yawgpu-tint built as a stub \
             (Tint FFI unavailable). Initialize the third_party/dawn submodule \
             (and run its tools/fetch_dawn_dependencies.py), or set YAWGPU_DAWN_DIR."
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

    // Detect the *target* OS (not the host): `cfg!` in a build script reflects the
    // host, while `CARGO_CFG_TARGET_OS` is the platform the shim is built for.
    let target_windows = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if target_windows {
        // Multi-config MSVC generators (Visual Studio) place artifacts in a
        // per-config subdirectory (`build/Debug`, `build/Release`, ...), unlike
        // single-config generators (Ninja, Makefiles) which use `build/`. Add
        // every config subdir that exists to the link search path so the
        // import library `tint_shim.lib` is found regardless of generator.
        for config in ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"] {
            let dir = build_dir.join(config);
            if dir.is_dir() {
                println!("cargo:rustc-link-search=native={}", dir.display());
            }
        }
        // Windows has no rpath: a dependent loads `tint_shim.dll` from the
        // executable's directory (or PATH). Copy it next to the consuming
        // artifacts so `cargo test`/`cargo run` and the cdylib find it.
        copy_runtime_dll(&build_dir);
    } else {
        // Locate libtint_shim at runtime (it is built next to the crate's
        // artifacts). `-Wl,-rpath` is GNU/Clang linker syntax; MSVC rejects it.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir.display());
    }
    println!("cargo:rustc-cfg=have_tint");
}

/// Copies the built `tint_shim.dll` next to the Cargo target artifacts so it is
/// discoverable at run time on Windows (which resolves dependent DLLs from the
/// executable's directory, not via rpath). Failures are warnings, not errors:
/// linking still succeeds against the import library, and a missing runtime copy
/// only surfaces when an artifact that loads Tint is actually executed.
fn copy_runtime_dll(build_dir: &Path) {
    let candidates = [
        build_dir.join("tint_shim.dll"),
        build_dir.join("Debug").join("tint_shim.dll"),
        build_dir.join("Release").join("tint_shim.dll"),
        build_dir.join("RelWithDebInfo").join("tint_shim.dll"),
        build_dir.join("MinSizeRel").join("tint_shim.dll"),
    ];
    let Some(dll) = candidates.into_iter().find(|p| p.is_file()) else {
        println!(
            "cargo:warning=tint_shim.dll not found under {}; runtime loads of Tint may fail",
            build_dir.display()
        );
        return;
    };

    // OUT_DIR is `<target>/<profile>/build/<pkg>-<hash>/out`; the profile dir
    // (where test/example/cdylib artifacts and their `deps/` live) is 3 up.
    let Some(out_dir) = env::var_os("OUT_DIR").map(PathBuf::from) else {
        return;
    };
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };
    for dest_dir in [profile_dir.to_path_buf(), profile_dir.join("deps")] {
        if dest_dir.is_dir() {
            let dest = dest_dir.join("tint_shim.dll");
            if let Err(e) = std::fs::copy(&dll, &dest) {
                println!(
                    "cargo:warning=failed to copy tint_shim.dll to {}: {e}",
                    dest.display()
                );
            }
        }
    }
}

/// Locates a usable Dawn source tree: the explicit `YAWGPU_DAWN_DIR` override
/// first, otherwise the vendored `third_party/dawn` submodule — but only when
/// its dependencies have actually been fetched (abseil present), so an
/// initialized-but-unfetched submodule degrades to the stub instead of a hard
/// CMake failure.
fn resolve_dawn_dir() -> Option<PathBuf> {
    if let Ok(dir) = env::var("YAWGPU_DAWN_DIR") {
        if !dir.is_empty() {
            return Some(PathBuf::from(dir));
        }
    }

    let manifest = env::var("CARGO_MANIFEST_DIR").ok()?;
    let vendored = Path::new(&manifest)
        .parent()?
        .join("third_party")
        .join("dawn");
    let has_dawn = vendored.join("CMakeLists.txt").is_file();
    let deps_fetched = vendored
        .join("third_party")
        .join("abseil-cpp")
        .join("CMakeLists.txt")
        .is_file();
    if has_dawn && deps_fetched {
        Some(vendored)
    } else {
        None
    }
}
