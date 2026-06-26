//! Rust bindings for the Tint shader compiler (Dawn's WGSL frontend), driven
//! through a small C++ shim (`shim/tint_shim.cpp`).
//!
//! Phase 1 of the naga->Tint migration
//! (`specs/tracking/tint-migration-plan.md`). This currently exposes only a
//! smoke surface (WGSL -> MSL) that proves the build/link/FFI path against a
//! local Dawn checkout; the full reflection + multi-target codegen API lands in
//! Phase 1b.
//!
//! When built without `YAWGPU_DAWN_DIR` set, the Tint backend is not linked
//! ([`HAVE_TINT`] is `false`) and the public functions return an error instead
//! of calling into Tint, so the crate still compiles in the default workspace.
#![warn(missing_docs)]

/// Whether this build links the Tint backend. `true` only when the crate was
/// built with `YAWGPU_DAWN_DIR` pointing at a Dawn checkout.
pub const HAVE_TINT: bool = cfg!(have_tint);

#[cfg(have_tint)]
mod imp {
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;

    extern "C" {
        fn yawgpu_tint_initialize();
        fn yawgpu_tint_wgsl_to_msl(
            wgsl: *const c_char,
            entry_point: *const c_char,
            err: *mut *mut c_char,
        ) -> *mut c_char;
        fn yawgpu_tint_string_free(s: *mut c_char);
    }

    pub fn initialize() {
        // SAFETY: the shim guards against repeated initialization.
        unsafe { yawgpu_tint_initialize() }
    }

    pub fn wgsl_to_msl(wgsl: &str, entry_point: &str) -> Result<String, String> {
        let wgsl_c =
            CString::new(wgsl).map_err(|_| "wgsl source contains an interior NUL".to_owned())?;
        let ep_c = CString::new(entry_point)
            .map_err(|_| "entry point contains an interior NUL".to_owned())?;
        let mut err: *mut c_char = std::ptr::null_mut();

        // SAFETY: pointers are valid for the duration of the call; the shim
        // either returns a heap string we own (freed below) or NULL with `err`
        // set to a heap string we own.
        unsafe {
            let out = yawgpu_tint_wgsl_to_msl(wgsl_c.as_ptr(), ep_c.as_ptr(), &mut err);
            if out.is_null() {
                let msg = if err.is_null() {
                    "tint: unknown error".to_owned()
                } else {
                    let s = CStr::from_ptr(err).to_string_lossy().into_owned();
                    yawgpu_tint_string_free(err);
                    s
                };
                return Err(msg);
            }
            let msl = CStr::from_ptr(out).to_string_lossy().into_owned();
            yawgpu_tint_string_free(out);
            Ok(msl)
        }
    }
}

#[cfg(not(have_tint))]
mod imp {
    const UNAVAILABLE: &str = "yawgpu-tint was built without Tint (YAWGPU_DAWN_DIR unset)";

    pub fn initialize() {}

    pub fn wgsl_to_msl(_wgsl: &str, _entry_point: &str) -> Result<String, String> {
        Err(UNAVAILABLE.to_owned())
    }
}

/// Initializes the Tint runtime. Idempotent; safe to call repeatedly. A no-op
/// when the Tint backend is not linked ([`HAVE_TINT`] is `false`).
pub fn initialize() {
    imp::initialize()
}

/// Compiles a WGSL module to Metal Shading Language for the named entry point.
///
/// Returns the generated MSL source on success, or a diagnostic message on
/// failure (parse/validation/codegen error, or — when [`HAVE_TINT`] is `false` —
/// an "unavailable" error).
pub fn wgsl_to_msl(wgsl: &str, entry_point: &str) -> Result<String, String> {
    imp::wgsl_to_msl(wgsl, entry_point)
}

#[cfg(all(test, have_tint))]
mod tests {
    use super::*;

    #[test]
    fn smoke_compute_wgsl_to_msl() {
        initialize();
        let wgsl = "@compute @workgroup_size(8, 1, 1) fn cs() {}";
        let msl = wgsl_to_msl(wgsl, "cs").expect("tint should compile a trivial compute shader");
        assert!(!msl.is_empty(), "expected non-empty MSL");
        assert!(
            msl.contains("kernel"),
            "expected an MSL kernel in output, got:\n{msl}"
        );
    }

    #[test]
    fn invalid_wgsl_reports_error() {
        initialize();
        let err = wgsl_to_msl("this is not wgsl", "cs").unwrap_err();
        assert!(!err.is_empty(), "expected a diagnostic message");
    }
}
