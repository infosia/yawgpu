use std::sync::Arc;

use khronos_egl as egl;
use std::ffi::c_void;

use super::adapter::GlesAdapter;
use super::egl::{self as gles_egl, EglConfig, EglDisplay, EglInstance};
use super::surface::GlesSurface;
use super::BACKEND;
use crate::HalError;

pub(super) struct GlesInstanceInner {
    pub(super) egl: EglInstance,
    pub(super) display: EglDisplay,
}

// SAFETY: `GlesInstanceInner` only shares the dynamically loaded EGL function
// table and an initialized display handle. Device-level GL/EGL context use is
// serialized by `GlesDeviceInner::current_lock`.
unsafe impl Send for GlesInstanceInner {}
// SAFETY: See the `Send` impl; shared access does not mutate Rust-managed
// state, and EGL calls that bind contexts are serialized at the device level.
unsafe impl Sync for GlesInstanceInner {}

impl Drop for GlesInstanceInner {
    fn drop(&mut self) {
        let _ = self.egl.terminate(self.display);
    }
}

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
        let egl = gles_egl::load_egl()?;
        // get_and_initialize_display returns an EGLDisplay that has already
        // had eglInitialize called on it successfully (cascading through
        // ANGLE backends on Windows; default-display on other platforms).
        let display = gles_egl::get_and_initialize_display(&egl)
            .ok_or(HalError::BackendUnavailable { backend: BACKEND })?;
        egl.bind_api(egl::OPENGL_ES_API)
            .map_err(|_| HalError::BackendUnavailable { backend: BACKEND })?;

        Ok(Self {
            inner: Arc::new(GlesInstanceInner { egl, display }),
        })
    }

    /// Returns adapters exposed by this instance.
    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<GlesAdapter> {
        choose_config(&self.inner)
            .map(|config| vec![GlesAdapter::new(Arc::clone(&self.inner), config)])
            .unwrap_or_default()
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
        self.create_window_surface(window)
    }

    fn create_window_surface(&self, native: *mut c_void) -> Result<GlesSurface, HalError> {
        let config = choose_config(&self.inner)?;
        let surface = unsafe {
            self.inner
                .egl
                .create_window_surface(self.inner.display, config, native as _, None)
        }
        .map_err(|_| HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "eglCreateWindowSurface failed",
        })?;
        Ok(GlesSurface::from_window_surface(
            Arc::clone(&self.inner),
            surface,
        ))
    }
}

fn choose_config(instance: &GlesInstanceInner) -> Result<EglConfig, HalError> {
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
        .map_err(|_| HalError::BackendUnavailable { backend: BACKEND })?
        .ok_or(HalError::BackendUnavailable { backend: BACKEND })
}
