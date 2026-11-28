use std::fs;
use std::path::Path;
use std::process::{Command, Output};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

fn run_guard(include: &Path) -> Output {
    Command::new("python3")
        .arg(Path::new(ROOT).join("tools/dawn_compat/abi_guard.py"))
        .arg("--yawgpu-include")
        .arg(include)
        .arg("--work")
        .arg(env!("CARGO_TARGET_TMPDIR"))
        .output()
        .expect("run the ABI guard with python3")
}

#[test]
fn guard_passes_at_the_current_pin() {
    let output = run_guard(&Path::new(ROOT).join("yawgpu/ffi/webgpu-headers"));
    assert!(output.status.success(), "{output:?}");
}

fn check_mutation(header: &str, before: &str, after: &str, expected: &[&str]) {
    let include = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap()
    ));
    fs::create_dir(&include).unwrap();
    for name in ["webgpu.h", "yawgpu.h"] {
        let path = Path::new(ROOT).join("yawgpu/ffi/webgpu-headers").join(name);
        let mut contents = fs::read_to_string(path).unwrap();
        if name == header {
            assert_eq!(contents.matches(before).count(), 1);
            contents = contents.replace(before, after);
        }
        fs::write(include.join(name), contents).unwrap();
    }
    let output = run_guard(&include);
    fs::remove_dir_all(include).unwrap();
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let stdout = std::str::from_utf8(&output.stdout).unwrap();
    assert_eq!(stdout.lines().collect::<Vec<_>>(), expected, "{output:?}");
}

#[test]
fn guard_reports_a_reordered_struct_member() {
    check_mutation(
        "yawgpu.h",
        "uint32_t M;\n    uint32_t N;",
        "uint32_t N;\n    uint32_t M;",
        &[
            "member WGPUSubgroupMatrixConfig.M",
            "member WGPUSubgroupMatrixConfig.N",
        ],
    );
}

#[test]
fn guard_reports_a_changed_enum_value() {
    check_mutation(
        "webgpu.h",
        "WGPUBlendOperation_Add = 0x00000001",
        "WGPUBlendOperation_Add = 0x0000000F",
        &["enum WGPUBlendOperation_Add"],
    );
}

#[test]
fn guard_reports_a_changed_function_signature() {
    check_mutation(
        "webgpu.h",
        "wgpuBufferGetSize(WGPUBuffer buffer)",
        "wgpuBufferGetSize(WGPUTexture buffer)",
        &["function wgpuBufferGetSize"],
    );
}
