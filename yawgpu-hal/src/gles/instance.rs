use std::sync::Arc;

use khronos_egl as egl;

use super::adapter::GlesAdapter;
use super::egl::{self as gles_egl, EglConfig, EglDisplay, EglInstance};
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
        let display = unsafe { egl.get_display(egl::DEFAULT_DISPLAY) }
            .ok_or(HalError::BackendUnavailable { backend: BACKEND })?;
        egl.initialize(display)
            .map_err(|_| HalError::BackendUnavailable { backend: BACKEND })?;
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
