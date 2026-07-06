use std::sync::Arc;

use khronos_egl as egl;
use std::ffi::c_void;
#[cfg(windows)]
use windows_sys::Win32::Foundation::HWND;

use super::adapter::GlesAdapter;
use super::egl::{self as gles_egl, EglConfig, EglDisplay, EglInstance};
use super::surface::GlesSurface;
use super::BACKEND;
use crate::HalError;

pub(super) enum GlesInstanceInner {
    Egl(Box<EglInstanceState>),
    #[cfg(windows)]
    Wgl(super::wgl::WglInstanceState),
}

pub(super) struct EglInstanceState {
    pub(super) egl: EglInstance,
    pub(super) display: EglDisplay,
}

// SAFETY: `GlesInstanceInner` only shares the dynamically loaded EGL function
// table / WGL loader state and initialized display handles. Device-level GL
// context use is serialized by `GlesDeviceInner::current_lock`.
unsafe impl Send for GlesInstanceInner {}
// SAFETY: See the `Send` impl; shared access does not mutate Rust-managed
// state, and calls that bind contexts are serialized at the device level.
unsafe impl Sync for GlesInstanceInner {}

// `EglInstanceState` deliberately has no `Drop` impl and never calls
// `eglTerminate`. EGL display handles are process-global:
// `eglGetDisplay(EGL_DEFAULT_DISPLAY)` returns the same display handle to
// every caller in the process, so terminating it on Drop would kill the
// display under any other live `GlesInstance` (every subsequent EGL call on
// it fails with `EGL_NOT_INITIALIZED`, and live GL contexts on a terminated
// display are undefined behaviour at the driver level). The display is left
// initialized for the process lifetime instead — wgpu-hal precedent.

/// Stores GLES instance data used by validation and backend submission.
pub struct GlesInstance {
    pub(super) inner: Arc<GlesInstanceInner>,
}

impl std::fmt::Debug for GlesInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesInstance").finish()
    }
}

impl GlesInstance {
    /// Creates a new GLES instance.
    pub fn new() -> Result<Self, HalError> {
        Self::new_with_choice(None)
    }

    /// Creates a new GLES instance with an optional context-backend override.
    pub fn new_with_choice(choice: Option<BackendChoice>) -> Result<Self, HalError> {
        match choice.unwrap_or_else(backend_from_env) {
            BackendChoice::Egl => Self::new_egl(),
            #[cfg(windows)]
            BackendChoice::Wgl => {
                let state = super::wgl::WglInstanceState::new()?;
                eprintln!("yawgpu-gles: using WGL backend (host OpenGL ES profile)");
                Ok(Self {
                    inner: Arc::new(GlesInstanceInner::Wgl(state)),
                })
            }
        }
    }

    /// Returns adapters exposed by this instance.
    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<GlesAdapter> {
        match self.inner.as_ref() {
            GlesInstanceInner::Egl(egl_state) => match choose_config(egl_state)
                .and_then(|config| GlesAdapter::new_egl(Arc::clone(&self.inner), config))
            {
                Ok(adapter) => vec![adapter],
                Err(err) => {
                    // Diagnostic: an empty enumeration is a legitimate
                    // spec-level "no adapter" outcome for callers, but on the
                    // EGL path it always stems from an EGL failure
                    // (`choose_config` prints the raw EGL error); make the
                    // mapping visible instead of silently returning empty.
                    eprintln!(
                        "yawgpu-gles: enumerate_adapters: choose_config failed ({err:?}); returning no adapters"
                    );
                    Vec::new()
                }
            },
            #[cfg(windows)]
            GlesInstanceInner::Wgl(_) => match GlesAdapter::new_wgl(Arc::clone(&self.inner)) {
                Ok(adapter) => vec![adapter],
                Err(err) => {
                    eprintln!(
                        "yawgpu-gles: enumerate_adapters: WGL limit probe failed ({err:?}); returning no adapters"
                    );
                    Vec::new()
                }
            },
        }
    }

    /// # Safety
    ///
    /// `hwnd` must be a valid Win32 window handle backed by an
    /// ANGLE-compatible window class for the lifetime of the resulting surface.
    pub unsafe fn create_surface_from_windows_hwnd(
        &self,
        hwnd: *mut c_void,
    ) -> Result<GlesSurface, HalError> {
        self.create_window_surface(hwnd)
    }

    /// # Safety
    ///
    /// `window` must be a valid `ANativeWindow*` for the lifetime of the
    /// resulting surface.
    pub unsafe fn create_surface_from_android_native_window(
        &self,
        window: *mut c_void,
    ) -> Result<GlesSurface, HalError> {
        let GlesInstanceInner::Egl(_) = self.inner.as_ref() else {
            return Err(HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "GLES Android surface requires the EGL backend",
            });
        };
        self.create_window_surface(window)
    }

    fn create_window_surface(&self, native: *mut c_void) -> Result<GlesSurface, HalError> {
        match self.inner.as_ref() {
            GlesInstanceInner::Egl(egl_state) => {
                let config = choose_config(egl_state)?;
                let surface = unsafe {
                    egl_state.egl.create_window_surface(
                        egl_state.display,
                        config,
                        native as _,
                        None,
                    )
                }
                .map_err(|_| HalError::SwapchainCreationFailed {
                    backend: BACKEND,
                    message: "eglCreateWindowSurface failed",
                })?;
                Ok(GlesSurface::from_egl_window(
                    Arc::clone(&self.inner),
                    surface,
                ))
            }
            #[cfg(windows)]
            GlesInstanceInner::Wgl(wgl_state) => {
                let surface = wgl_state.create_window_surface(native as HWND)?;
                Ok(GlesSurface::from_wgl_window(
                    Arc::clone(&self.inner),
                    surface,
                ))
            }
        }
    }

    fn new_egl() -> Result<Self, HalError> {
        let egl = gles_egl::load_egl()?;
        // get_and_initialize_display returns an EGLDisplay that has already
        // had eglInitialize called on it successfully (cascading through
        // ANGLE backends on Windows; default-display on other platforms).
        let display = gles_egl::get_and_initialize_display(&egl)
            .ok_or(HalError::BackendUnavailable { backend: BACKEND })?;
        egl.bind_api(egl::OPENGL_ES_API)
            .map_err(|_| HalError::BackendUnavailable { backend: BACKEND })?;

        Ok(Self {
            inner: Arc::new(GlesInstanceInner::Egl(Box::new(EglInstanceState {
                egl,
                display,
            }))),
        })
    }
}

fn choose_config(instance: &EglInstanceState) -> Result<EglConfig, HalError> {
    let attribs = [
        egl::SURFACE_TYPE,
        egl::PBUFFER_BIT,
        egl::RENDERABLE_TYPE,
        egl::OPENGL_ES3_BIT,
        egl::RED_SIZE,
        8,
        egl::GREEN_SIZE,
        8,
        egl::BLUE_SIZE,
        8,
        egl::ALPHA_SIZE,
        8,
        egl::NONE,
    ];
    instance
        .egl
        .choose_first_config(instance.display, &attribs)
        .map_err(|err| {
            // Diagnostic: surface the raw EGL error (e.g. EGL_NOT_INITIALIZED
            // when the display was terminated out from under us), consistent
            // with the other `yawgpu-gles:` bring-up failure paths.
            eprintln!("yawgpu-gles: eglChooseConfig failed: {err:?}");
            HalError::BackendUnavailable { backend: BACKEND }
        })?
        .ok_or(HalError::BackendUnavailable { backend: BACKEND })
}

/// Selects the GLES context backend used when creating a GLES instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    /// EGL context backend.
    Egl,
    /// Windows WGL context backend.
    #[cfg(windows)]
    Wgl,
}

pub(super) fn backend_from_env() -> BackendChoice {
    parse_backend(std::env::var("YAWGPU_GLES_BACKEND").ok().as_deref())
}

pub(super) fn parse_backend(value: Option<&str>) -> BackendChoice {
    match value {
        #[cfg(windows)]
        Some("wgl") => BackendChoice::Wgl,
        Some("egl") | Some("") | None => BackendChoice::Egl,
        Some(other) => {
            eprintln!("yawgpu-gles: unknown YAWGPU_GLES_BACKEND={other:?}; falling back to egl");
            BackendChoice::Egl
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_backend_defaults_to_egl() {
        assert_eq!(parse_backend(None), BackendChoice::Egl);
        assert_eq!(parse_backend(Some("")), BackendChoice::Egl);
        assert_eq!(parse_backend(Some("egl")), BackendChoice::Egl);
        assert_eq!(parse_backend(Some("unknown")), BackendChoice::Egl);
    }

    #[test]
    fn parse_backend_handles_wgl_by_platform() {
        #[cfg(windows)]
        assert_eq!(parse_backend(Some("wgl")), BackendChoice::Wgl);

        #[cfg(not(windows))]
        assert_eq!(parse_backend(Some("wgl")), BackendChoice::Egl);
    }

    #[test]
    fn new_with_choice_uses_explicit_egl_when_available() {
        let instance = match GlesInstance::new_with_choice(Some(BackendChoice::Egl)) {
            Ok(instance) => instance,
            Err(error) => {
                eprintln!("skipping GLES EGL constructor test; backend unavailable: {error:?}");
                return;
            }
        };

        assert!(matches!(instance.inner.as_ref(), GlesInstanceInner::Egl(_)));
    }

    #[test]
    fn dropping_one_instance_leaves_overlapping_instance_usable() {
        // Regression test for the process-global EGL display: dropping one
        // instance must never terminate the display under another live
        // instance (see the `EglInstanceState` comment above).
        let first = match GlesInstance::new() {
            Ok(instance) => instance,
            Err(error) => {
                eprintln!(
                    "skipping GLES overlapping-instance test; backend unavailable: {error:?}"
                );
                return;
            }
        };
        let second = match GlesInstance::new() {
            Ok(instance) => instance,
            Err(error) => {
                eprintln!(
                    "skipping GLES overlapping-instance test; backend unavailable: {error:?}"
                );
                return;
            }
        };

        drop(first);

        let adapters = second.enumerate_adapters();
        assert!(
            !adapters.is_empty(),
            "surviving instance must still enumerate adapters after the other instance is dropped"
        );
        adapters[0]
            .create_device()
            .expect("surviving instance must still create a device");
    }
}
