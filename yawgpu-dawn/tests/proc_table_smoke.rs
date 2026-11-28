use std::ffi::{c_char, CStr};
use std::process::{Command, Output};
use std::sync::Mutex;
use yawgpu_dawn::{dawn_proc_table, DawnProcTable};

#[link(name = "dawn_smoke", kind = "static")]
extern "C" {
    fn smokeCheckTable(table: *const DawnProcTable) -> bool;
    fn smokeCheckProcAddress() -> bool;
    fn smokeCallStubs() -> bool;
    fn smokeUnconfiguredCreateInstance() -> bool;
    fn smokeSetWrongVersion();
    fn smokeCompute(backend: u32, bytes: *mut u8, size: usize) -> *const c_char;
}

/// Runs the C++ consumer's compute pass on a `YAWGPU_INSTANCE_BACKEND_*` backend.
fn compute(backend: u32) -> [u8; 16] {
    // dawnProcSetProcs changes the consumer's process-wide dispatch table.
    static DISPATCH: Mutex<()> = Mutex::new(());
    let _guard = DISPATCH.lock().unwrap();
    let mut bytes = [0; 16];
    // SAFETY: bytes is writable for exactly size bytes throughout the synchronous call.
    let error = unsafe { smokeCompute(backend, bytes.as_mut_ptr(), bytes.len()) };
    if !error.is_null() {
        // SAFETY: the helper returns a static, null-terminated diagnostic on failure.
        panic!("C++ smoke failed at {:?}", unsafe { CStr::from_ptr(error) });
    }
    bytes
}

#[test]
fn smoke_runs_on_noop() {
    compute(yawgpu::YAWGPU_INSTANCE_BACKEND_NOOP);
}

#[cfg(any(all(target_os = "macos", feature = "metal"), feature = "vulkan"))]
fn assert_computed(bytes: [u8; 16]) {
    for (i, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(u32::from_le_bytes(*word), 0x1710_0000 + i as u32);
    }
}

#[test]
#[cfg(all(target_os = "macos", feature = "metal"))]
#[ignore = "manual real-backend test"]
fn smoke_computes_on_metal() {
    assert_computed(compute(yawgpu::YAWGPU_INSTANCE_BACKEND_METAL));
}

#[test]
#[cfg(feature = "vulkan")]
#[ignore = "manual real-backend test"]
fn smoke_computes_on_vulkan() {
    assert_computed(compute(yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN));
}

// This test binary never sets the default backend.
#[cfg(any(
    not(any(feature = "metal", feature = "vulkan")),
    all(target_os = "macos", feature = "metal"),
    all(not(target_vendor = "apple"), feature = "vulkan")
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
#[cfg(all(not(target_vendor = "apple"), feature = "vulkan"))]
#[ignore = "manual real-backend test"]
fn default_off_apple_with_vulkan_is_vulkan() {
    assert_eq!(
        default_adapter_backend(),
        yawgpu::native::WGPUBackendType_Vulkan
    );
}

#[test]
fn table_has_all_members_and_pinned_version() {
    // SAFETY: the accessor returns the static table inspected by this helper.
    assert!(unsafe { smokeCheckTable(dawn_proc_table()) });
}

#[test]
fn table_resolves_create_instance_and_forwards_other_names() {
    // SAFETY: the helper compares function pointers from the static table.
    assert!(unsafe { smokeCheckProcAddress() });
}

const CHILD: &str = "YAWGPU_DAWN_SMOKE_CHILD";

#[test]
fn child_probe() {
    match std::env::var(CHILD).as_deref() {
        Ok("stub") => {
            // SAFETY: the tested stubs ignore their arguments.
            assert!(unsafe { smokeCallStubs() });
            assert!(unsafe { smokeCallStubs() });
        }
        Ok("warning") => {
            // SAFETY: the child starts with Dawn's default dispatch table.
            assert!(unsafe { smokeUnconfiguredCreateInstance() });
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

const STUBS: [&str; 5] = [
    "wgpuAdapterGetFormatCapabilities",
    "wgpuDeviceTick",
    "wgpuAdapterGetInstance",
    "wgpuResourceTableGetSize",
    "wgpuSharedBufferMemoryIsDeviceLost",
];

fn assert_stub_messages(output: &Output, names: &[&str]) {
    let expected: Vec<_> = names
        .iter()
        .map(|name| format!("yawgpu-dawn: unsupported {name}"))
        .collect();
    let stderr = std::str::from_utf8(&output.stderr).unwrap();
    assert_eq!(stderr.lines().collect::<Vec<_>>(), expected);
}

#[test]
fn stubs_report_once_and_return_their_fallbacks() {
    let output = child("stub", false);
    assert!(output.status.success(), "{output:?}");
    assert_stub_messages(&output, &STUBS);
}

#[test]
fn strict_stub_reports_and_aborts() {
    let output = child("stub", true);
    assert_aborted(&output);
    assert_stub_messages(&output, &STUBS[..1]);
}

#[test]
fn mismatched_version_aborts() {
    assert_aborted(&child("version", false));
}

#[test]
fn standalone_create_instance_warning_ends_with_newline() {
    let output = child("warning", false);
    assert!(output.status.success(), "{output:?}");
    let stderr = std::str::from_utf8(&output.stderr).unwrap();
    assert!(stderr.contains("wgpuCreateInstance"), "{stderr}");
    assert!(stderr.ends_with('\n'), "{stderr:?}");
}
