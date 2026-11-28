use std::ffi::{c_char, CStr};
use std::process::{Command, Output};
use std::sync::Mutex;
use yawgpu_dawn::{dawn_proc_table, DawnProcTable};

#[link(name = "dawn_smoke", kind = "static")]
extern "C" {
    fn smokeCheckTable(table: *const DawnProcTable) -> bool;
    fn smokeCallStub() -> bool;
    fn smokeSetWrongVersion();
    fn smokeCompute(metal: i32, bytes: *mut u8, size: usize) -> *const c_char;
}

fn compute(metal: bool) -> [u8; 16] {
    // dawnProcSetProcs changes the consumer's process-wide dispatch table.
    static DISPATCH: Mutex<()> = Mutex::new(());
    let _guard = DISPATCH.lock().unwrap();
    let mut bytes = [0; 16];
    // SAFETY: bytes is writable for exactly size bytes throughout the synchronous call.
    let error = unsafe { smokeCompute(metal.into(), bytes.as_mut_ptr(), bytes.len()) };
    if !error.is_null() {
        // SAFETY: the helper returns a static, null-terminated diagnostic on failure.
        panic!("C++ smoke failed at {:?}", unsafe { CStr::from_ptr(error) });
    }
    bytes
}

#[test]
fn smoke_runs_on_noop() {
    compute(false);
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "manual real-backend test"]
fn smoke_computes_on_metal() {
    let bytes = compute(true);
    for (i, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(u32::from_le_bytes(*word), 0x1710_0000 + i as u32);
    }
}

// This test binary never sets the default backend.
#[cfg(any(
    not(any(feature = "metal", feature = "vulkan")),
    all(target_os = "macos", feature = "metal")
))]
fn default_adapter_backend() -> yawgpu::native::WGPUBackendType {
    extern "C" {
        fn smokeAdapterBackend(
            desc: *const yawgpu::native::WGPUInstanceDescriptor,
        ) -> yawgpu::native::WGPUBackendType;
    }
    // SAFETY: The table accepts a null descriptor.
    unsafe { smokeAdapterBackend(std::ptr::null()) }
}

#[test]
#[cfg(not(any(feature = "metal", feature = "vulkan")))]
fn default_without_backend_features_is_noop() {
    assert_eq!(
        default_adapter_backend(),
        yawgpu::native::WGPUBackendType_Null
    );
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "manual real-backend test"]
fn default_on_macos_with_metal_is_metal() {
    assert_eq!(
        default_adapter_backend(),
        yawgpu::native::WGPUBackendType_Metal
    );
}

#[test]
fn table_has_all_members_and_pinned_version() {
    // SAFETY: the accessor returns the static table inspected by this helper.
    assert!(unsafe { smokeCheckTable(dawn_proc_table()) });
}

const CHILD: &str = "YAWGPU_DAWN_SMOKE_CHILD";

#[test]
fn child_probe() {
    match std::env::var(CHILD).as_deref() {
        Ok("stub") => {
            // SAFETY: this helper calls the known status stub, which ignores its arguments.
            assert!(unsafe { smokeCallStub() });
            assert!(unsafe { smokeCallStub() });
        }
        Ok("version") => {
            // SAFETY: the helper passes a live table copy; Dawn checks its changed version.
            unsafe { smokeSetWrongVersion() };
        }
        _ => {}
    }
}

fn child(probe: &str, strict: bool) -> Output {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_probe", "--nocapture"])
        .env(CHILD, probe)
        .env("YAWGPU_DAWN_STRICT", if strict { "1" } else { "0" })
        .output()
        .unwrap()
}

fn assert_aborted(output: &Output) {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(output.status.signal(), Some(6), "{output:?}"); // SIGABRT
    }
    #[cfg(not(unix))]
    assert!(!output.status.success(), "{output:?}");
}

fn assert_one_stub_message(output: &Output) {
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).trim_end(),
        "yawgpu-dawn: unsupported wgpuAdapterGetFormatCapabilities",
        "{output:?}"
    );
}

#[test]
fn stub_reports_once_and_returns_error() {
    let output = child("stub", false);
    assert!(output.status.success(), "{output:?}");
    assert_one_stub_message(&output);
}

#[test]
fn strict_stub_reports_and_aborts() {
    let output = child("stub", true);
    assert_aborted(&output);
    assert_one_stub_message(&output);
}

#[test]
fn mismatched_version_aborts() {
    assert_aborted(&child("version", false));
}
