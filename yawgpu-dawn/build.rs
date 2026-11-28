#[cfg(not(feature = "dawn-compat"))]
fn main() {}

#[cfg(feature = "dawn-compat")]
fn main() {
    use std::{env, path::PathBuf, process::Command};

    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    for path in [
        "src/proc_table.cpp",
        "include",
        "tests/smoke",
        "../tools/dawn_compat",
        "../yawgpu/ffi/webgpu-headers",
        "../third_party/dawn/include",
        "../third_party/dawn/generator",
        "../third_party/dawn/src/dawn/dawn.json",
        "../third_party/dawn/src/dawn/dawn_wire.json",
        "../third_party/dawn/src/dawn/dawn_version.json",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }
    assert!(Command::new("python3")
        .args(["-B", "../tools/dawn_compat/gen_proc_table.py", "--out"])
        .arg(&out)
        .status()
        .expect("run the Dawn generators")
        .success());

    let package = out.join("dawn-api");
    let mut cpp = cc::Build::new();
    cpp.cpp(true)
        .std("c++20")
        .include(package.join("include"))
        .include("include")
        .include(&out);
    cpp.clone()
        .file("src/proc_table.cpp")
        .compile("yawgpu_procs");

    // Only the integration test requests this archive via #[link]. Its private
    // dispatch symbols let both sides of the C ABI coexist in the test executable.
    let aliases = out.join("smoke_dispatch.h");
    if cpp.get_compiler().is_like_msvc() {
        cpp.flag(format!("/FI{}", aliases.display()));
    } else {
        cpp.flag("-include").flag(aliases);
    }
    let backend = cc::Build::new()
        .cargo_metadata(false)
        .include("../yawgpu/ffi/webgpu-headers")
        .file("tests/smoke/backend.c")
        .compile_intermediates();
    cpp.cargo_metadata(false)
        .objects(backend)
        .file(package.join("src/dawn_proc.cpp"))
        .file(package.join("src/dawn_thread_dispatch_proc.cpp"))
        .file("tests/smoke/table.cpp")
        .file("tests/smoke/consumer.cpp")
        .compile("dawn_smoke");
}
