//! Process-exit guard for the EGL path of the GLES backend (F-153).
//!
//! # The problem
//!
//! A WebGPU consumer written in C++ commonly holds its last device reference
//! in a namespace-scope static (the webgpu-native-cts `DeviceCache` is one),
//! so `wgpuDeviceRelease` is called from that object's **static destructor**,
//! i.e. from `__run_exit_handlers` inside `exit()`.
//!
//! On Linux/EGL that is too late to touch GL. The vendor EGL driver registers
//! its own exit handler when it is `dlopen`ed — which happens *during*
//! `eglInitialize`, and therefore *after* the executable's static-init
//! `__cxa_atexit` registrations. Exit handlers run last-registered-first, so
//! the driver's handler runs **before** the consumer's static destructor and
//! `dlclose`s the vendor implementation. Measured on NVIDIA 595.91.07:
//!
//! ```text
//! [main returns]
//!   calling fini: libnvidia-egl-wayland2.so.1
//!   calling fini: libnvidia-egl-gbm.so.1
//!   calling fini: libnvidia-egl-xcb.so.1
//!   calling fini: libnvidia-egl-xlib.so.1
//!   calling fini: libnvidia-allocator.so.1
//!   calling fini: libnvidia-eglcore.so.595.91.07   <- the GL implementation
//!   calling fini: libnvidia-gpucomp.so.595.91.07
//! [~DeviceCache -> wgpuDeviceRelease]
//! ```
//!
//! `libGLdispatch.so` (libglvnd's dispatch layer) and `libEGL_nvidia.so` stay
//! mapped, so every GL entry point still *resolves* — it just tail-jumps into
//! an address range that is no longer mapped. The result is a SIGSEGV in a
//! frameless driver address, first in `glFinish` (the `Device::lose` queue
//! drain) and then, if that call is removed, in `glDeleteSampler`
//! (`EglDeviceState::drop`).
//!
//! Nothing observable distinguishes this state: `eglGetCurrentContext` and
//! `eglGetCurrentDisplay` still return the live handles and `eglGetError`
//! still reports `EGL_SUCCESS`, because those queries are answered by the
//! still-mapped dispatch layer. So there is no handle-validity check that
//! could guard the teardown — the only thing that distinguishes "before the
//! driver's exit handler" from "after" is *when* we are.
//!
//! # The guard
//!
//! Register our own `atexit` handler at **device creation**, which is after
//! `eglInitialize` has pulled the vendor driver in, so our handler is
//! registered later than the driver's and therefore runs *earlier*. It sets a
//! latch; every GLES teardown path checks the latch and skips its GL/EGL work
//! once it is set. At that point the process is exiting and the driver has
//! already released everything it owns, so skipping is not a leak in any
//! observable sense.
//!
//! The latch is armed only from the EGL device-creation path, so a
//! WGL-only process never sets it and its behaviour is unchanged.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

unsafe extern "C" {
    /// C `atexit(3)`. Declared directly rather than pulled in through a new
    /// dependency: it is in every libc/CRT this backend can target, and the
    /// callback takes no argument, so the declaration is unambiguous.
    fn atexit(callback: extern "C" fn()) -> core::ffi::c_int;
}

/// Set once the process has entered `exit()` far enough that the EGL vendor
/// driver may already have unloaded itself.
static DRIVER_TEARDOWN_STARTED: AtomicBool = AtomicBool::new(false);

/// Guards the one-shot `atexit` registration.
static REGISTER_ONCE: Once = Once::new();

/// Sets a teardown latch.
///
/// Split out from the `atexit` callback so the latch protocol can be unit
/// tested against a local `AtomicBool`: the process-wide latch must never be
/// set by a test, because `yawgpu-hal`'s own GLES unit tests create real EGL
/// devices in the same process and read it when they tear those devices down.
fn set_latch(latch: &AtomicBool) {
    latch.store(true, Ordering::SeqCst);
}

/// Reads a teardown latch. See [`set_latch`] for why this is factored out.
fn read_latch(latch: &AtomicBool) -> bool {
    latch.load(Ordering::SeqCst)
}

/// The `atexit` callback. Runs on the thread that called `exit()`, before the
/// exit handlers registered earlier than it — which includes the consumer's
/// C++ static destructors, and excludes the EGL driver's own teardown handler.
extern "C" fn mark_driver_teardown_started() {
    set_latch(&DRIVER_TEARDOWN_STARTED);
}

/// Arms the process-exit guard, once per process.
///
/// Must be called only after the EGL vendor driver has been loaded (i.e. after
/// `eglInitialize` and context creation), so that this registration lands
/// *after* the driver's own exit handler and therefore runs *before* it.
pub(super) fn arm() {
    REGISTER_ONCE.call_once(|| {
        // SAFETY: `mark_driver_teardown_started` is an `extern "C" fn()` with
        // static lifetime and no captured state; it only stores into a
        // `'static` atomic, which is async-signal-safe enough for an exit
        // handler.
        let status = unsafe { atexit(mark_driver_teardown_started) };
        if status != 0 {
            eprintln!(
                "yawgpu-gles: atexit registration failed ({status}); \
                 releasing a device from a static destructor may crash"
            );
        }
    });
}

/// Whether the EGL vendor driver may already have unloaded itself.
///
/// Every GLES teardown path (`GlesQueue::wait_idle`, every resource `Drop`
/// routed through `with_current_context`, `EglDeviceState::drop` and
/// `GlesSurfaceInner::drop`) checks this and skips its GL/EGL work when it is
/// `true`.
pub(super) fn driver_teardown_started() -> bool {
    read_latch(&DRIVER_TEARDOWN_STARTED)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latch_starts_clear_then_sets_and_stays_set() {
        // The latch protocol, exercised on a local latch so the process-wide
        // one is untouched: clear until the exit callback runs, set
        // afterwards, and idempotent (an exit handler may not assume it runs
        // exactly once).
        let latch = AtomicBool::new(false);
        assert!(!read_latch(&latch));
        set_latch(&latch);
        assert!(read_latch(&latch));
        set_latch(&latch);
        assert!(read_latch(&latch));
    }

    #[test]
    fn arming_the_guard_is_idempotent_and_leaves_the_latch_clear() {
        // Arming only registers an `atexit` handler. It must not latch, or
        // every device created in this process would skip its GL teardown
        // from here on. Reading the process-wide latch is safe: nothing in a
        // test run can set it, because only `exit()` does.
        arm();
        arm();
        assert!(!driver_teardown_started());
    }
}
