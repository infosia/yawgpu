#![cfg(feature = "gles")]
//! F-153 regression: releasing the last GLES device reference from inside
//! `exit()` must not crash the process.
//!
//! The webgpu-native-cts holds its device in a C++ namespace-scope static, so
//! `wgpuDeviceRelease` runs from that object's static destructor — after the
//! EGL vendor driver's own exit handler has already `dlclose`d the driver.
//! Every GLES run therefore exited 139 (SIGSEGV) no matter what it ran.
//!
//! The shape is reproduced here without C++: an `atexit` handler registered
//! **before** the GL driver is loaded gets exactly the ordering a static
//! destructor does, because exit handlers run last-registered-first and the
//! driver registers its own handler when `eglInitialize` pulls it in. The
//! handler then does what `Device::lose` + the `Arc` drop do: drain the
//! queue, then drop the device.
//!
//! The assertion is the child process's exit status, so the test runs itself
//! as a child: a crash cannot be observed from inside the faulting process.
//!
//! The *manifestation* of the pre-fix crash depends on which thread last made
//! the context current, but the condition is the same in both cases. libtest
//! runs a test body on a worker thread, so the exit handler binds the context
//! from the main thread: `eglMakeCurrent` then has real work to do, fails
//! inside the unloaded vendor driver, and reports `EGL_SUCCESS` anyway, which
//! `khronos-egl` `unwrap()`s into a panic — SIGABRT. A C++ consumer whose
//! device was created on the main thread instead gets a no-op `eglMakeCurrent`
//! and faults one call later, in `glFinish` — SIGSEGV. (Verified: with the
//! same worker-thread device released from an `atexit` handler registered
//! *after* the driver loaded — i.e. before the unload — teardown succeeds and
//! the process exits 0, so the thread is not what breaks it.) The assertion is
//! therefore just "the child exited cleanly", which covers both.

use std::process::Command;
use std::sync::Mutex;

use yawgpu_hal::gles::GlesInstance;
use yawgpu_hal::{HalDevice, HalInstance, HalQueue};
use yawgpu_test::{real_backend_available, RealBackend};

/// Set on the child process to select the child half of this test.
const CHILD_ENV: &str = "YAWGPU_E2E_GLES_PROCESS_EXIT_CHILD";

/// This test's own name, used to re-run just it in the child.
const TEST_NAME: &str = "gles_device_released_during_process_exit_exits_cleanly";

unsafe extern "C" {
    /// C `atexit(3)`; see `yawgpu-hal/src/gles/exit_guard.rs` for why the
    /// registration *order* is the whole point of this test.
    fn atexit(callback: extern "C" fn()) -> core::ffi::c_int;
}

/// Holds the device across `main`'s return so it is released from the exit
/// handler below rather than from the test body — the C++ static-member
/// equivalent.
static PARKED_DEVICE: Mutex<Option<(HalQueue, HalDevice)>> = Mutex::new(None);

/// Released from `exit()`. Mirrors `Device::lose` (the queue drain) followed
/// by the final `Arc` drop (`EglDeviceState::drop`) — the two GL/EGL call
/// sites that faulted before the fix.
extern "C" fn release_parked_device() {
    // A panic escaping an exit handler aborts the process, which would look
    // exactly like the crash under test. Fail closed instead.
    let Ok(mut parked) = PARKED_DEVICE.lock() else {
        return;
    };
    if let Some((queue, device)) = parked.take() {
        let _ = queue.wait_idle();
        drop(device);
    }
}

/// The child half: arm the exit handler, create a GLES device, park it, and
/// return. The process then exits and the handler does the teardown.
fn run_child() {
    assert_eq!(
        // SAFETY: `release_parked_device` is a `'static` `extern "C" fn()`.
        // Registered here, before `GlesInstance::new()` loads the EGL vendor
        // driver, so it runs *after* the driver's own exit handler.
        unsafe { atexit(release_parked_device) },
        0,
        "atexit registration failed"
    );

    let instance = HalInstance::Gles(GlesInstance::new().expect("EGL init"));
    let adapter = instance
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("adapter");
    let device = adapter.create_device().expect("device");
    let queue = device.queue();
    // Prove the context is live before parking it, so a later fault cannot be
    // confused with a device that never worked.
    queue.submit_empty().expect("empty submit");

    *PARKED_DEVICE.lock().expect("park device") = Some((queue, device));
}

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn gles_device_released_during_process_exit_exits_cleanly() {
    if std::env::var_os(CHILD_ENV).is_some() {
        run_child();
        return;
    }

    assert!(
        real_backend_available(RealBackend::Gles),
        "GLES backend not available; install an ANGLE build with ES 3.1 support \
         (Chrome / Edge ANGLE caps at ES 3.0; see specs/blocks/67-gles-backend.md). \
         Without a real GLES adapter this test cannot verify the GLES execution path."
    );

    let exe = std::env::current_exe().expect("test binary path");
    let output = Command::new(exe)
        .env(CHILD_ENV, "1")
        .args([
            "--exact",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
            TEST_NAME,
        ])
        .output()
        .expect("spawn child test process");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "releasing a GLES device from an exit handler must not crash the \
         process; child status: {:?}\n--- child stdout ---\n{stdout}\
         \n--- child stderr ---\n{stderr}",
        output.status,
    );
    assert!(
        stdout.contains("1 passed"),
        "child did not report the parked-device test as passing\
         \n--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}",
    );
}
