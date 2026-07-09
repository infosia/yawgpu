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
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_HOME");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_ROOT");
    println!("cargo:rerun-if-env-changed=NDK_HOME");
    println!("cargo:rerun-if-env-changed=ANDROID_PLATFORM");
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
    let mut cfg = cmake::Config::new("shim");
    cfg.define("YAWGPU_DAWN_DIR", &dawn_dir)
        .build_target("tint_shim");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("android") {
        configure_android_cmake(&mut cfg);
    }
    let dst = cfg.build();

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
        copy_runtime_shim(&build_dir);
    } else {
        // Locate libtint_shim at runtime (it is built next to the crate's
        // artifacts). `-Wl,-rpath` is GNU/Clang linker syntax; MSVC rejects it.
        // Harmless and still useful for any image linked directly by this crate's
        // own targets; kept even though Apple builds resolve the shim via the
        // `@loader_path` install name below.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_dir.display());
        // On Apple targets the shim's install name is `@loader_path/...` (set in
        // shim/CMakeLists.txt), so dyld resolves it next to each loading image
        // and NO LONGER consults DYLD_FALLBACK_LIBRARY_PATH for it. Every dir that
        // hosts an image that loads the shim must therefore contain a copy — miss
        // one and that target breaks at run time only. Copy it next to the
        // consuming artifacts, exactly as on Windows.
        copy_runtime_shim(&build_dir);
    }
    println!("cargo:rustc-cfg=have_tint");
}

fn configure_android_cmake(cfg: &mut cmake::Config) {
    if let Some(toolchain) = android_ndk_toolchain_file() {
        cfg.define("CMAKE_TOOLCHAIN_FILE", toolchain);
    } else {
        println!(
            "cargo:warning=Android cross-compilation of Tint needs ANDROID_NDK_HOME \
             pointing at an NDK with build/cmake/android.toolchain.cmake \
             (ANDROID_NDK_ROOT and NDK_HOME are also checked)"
        );
    }

    match env::var("CARGO_CFG_TARGET_ARCH")
        .ok()
        .as_deref()
        .and_then(android_abi_for_arch)
    {
        Some(abi) => {
            cfg.define("ANDROID_ABI", abi);
        }
        None => {
            let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| "<unset>".into());
            println!(
                "cargo:warning=unsupported Android target arch `{arch}`; \
                 not defining ANDROID_ABI"
            );
        }
    }

    let platform = env::var("ANDROID_PLATFORM").unwrap_or_else(|_| "android-24".into());
    cfg.define("ANDROID_PLATFORM", platform);
}

fn android_ndk_toolchain_file() -> Option<PathBuf> {
    for var in ["ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "NDK_HOME"] {
        let Ok(root) = env::var(var) else {
            continue;
        };
        if root.is_empty() {
            continue;
        }

        let toolchain = PathBuf::from(root)
            .join("build")
            .join("cmake")
            .join("android.toolchain.cmake");
        if toolchain.is_file() {
            return Some(toolchain);
        }
    }
    None
}

fn android_abi_for_arch(arch: &str) -> Option<&'static str> {
    match arch {
        "aarch64" => Some("arm64-v8a"),
        "arm" => Some("armeabi-v7a"),
        "x86_64" => Some("x86_64"),
        "x86" => Some("x86"),
        "riscv64" => Some("riscv64"),
        _ => None,
    }
}

/// Copies the built Tint shim next to the Cargo target artifacts so it is
/// discoverable at run time. Needed on Windows (which resolves dependent DLLs
/// from the executable's directory, not via rpath) and on Apple targets (where
/// the shim's `@loader_path` install name makes dyld resolve it next to each
/// loading image, bypassing DYLD_FALLBACK_LIBRARY_PATH). The runtime file name
/// is chosen per *target* OS. Failures are warnings, not errors: linking still
/// succeeds, and a missing runtime copy only surfaces when an artifact that
/// loads Tint is actually executed.
fn copy_runtime_shim(build_dir: &Path) {
    let file_name = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => "tint_shim.dll",
        Ok("macos") | Ok("ios") => "libtint_shim.dylib",
        Ok("android") | Ok("linux") => "libtint_shim.so",
        _ => return,
    };

    // Single-config generators (Ninja, Makefiles) place the artifact in
    // `build/`; multi-config MSVC generators use a per-config subdir.
    let candidates = [
        build_dir.join(file_name),
        build_dir.join("Debug").join(file_name),
        build_dir.join("Release").join(file_name),
        build_dir.join("RelWithDebInfo").join(file_name),
        build_dir.join("MinSizeRel").join(file_name),
    ];
    let Some(shim) = candidates.into_iter().find(|p| p.is_file()) else {
        println!(
            "cargo:warning={file_name} not found under {}; runtime loads of Tint may fail",
            build_dir.display()
        );
        return;
    };

    // OUT_DIR is `<target>/<profile>/build/<pkg>-<hash>/out`; the profile dir
    // (where cdylib/test/example artifacts and their `deps/` live) is 3 up.
    let Some(out_dir) = env::var_os("OUT_DIR").map(PathBuf::from) else {
        return;
    };
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };
    // `<profile>` holds libyawgpu.dylib; `<profile>/deps` holds its real file and
    // every test binary; `<profile>/examples` holds example binaries. Each of
    // these can load the shim, so each needs its own copy.
    for dest_dir in [
        profile_dir.to_path_buf(),
        profile_dir.join("deps"),
        profile_dir.join("examples"),
    ] {
        if dest_dir.is_dir() {
            let dest = dest_dir.join(file_name);
            if let Err(e) = std::fs::copy(&shim, &dest) {
                println!(
                    "cargo:warning=failed to copy {file_name} to {}: {e}",
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

#[cfg(test)]
mod tests {
    use super::android_abi_for_arch;

    #[test]
    fn android_abi_for_arch_maps_supported_targets() {
        assert_eq!(android_abi_for_arch("aarch64"), Some("arm64-v8a"));
        assert_eq!(android_abi_for_arch("arm"), Some("armeabi-v7a"));
        assert_eq!(android_abi_for_arch("x86_64"), Some("x86_64"));
        assert_eq!(android_abi_for_arch("x86"), Some("x86"));
        assert_eq!(android_abi_for_arch("riscv64"), Some("riscv64"));
        assert_eq!(android_abi_for_arch("wasm32"), None);
    }
}
