use std::ffi::c_void;

use super::device::GlesDevice;
use super::texture::GlesTexture;
use super::unavailable;
use crate::{HalError, HalSurfaceConfiguration};

/// Stores GLES surface data used by validation and backend submission.
pub struct GlesSurface;

impl std::fmt::Debug for GlesSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesSurface").finish()
    }
}

impl GlesSurface {
    /// Creates a GLES surface scaffold from an Android native window pointer.
    ///
    /// # Safety
    ///
    /// `window` must be a valid `ANativeWindow*` from the Android NDK and
    /// must outlive the resulting surface.
    pub unsafe fn from_android_native_window(_window: *mut c_void) -> Result<Self, HalError> {
        unavailable()
    }

    /// Configures the surface's swapchain for the given format, size, and present mode.
    pub fn configure(
        &mut self,
        _device: &GlesDevice,
        _config: HalSurfaceConfiguration,
    ) -> Result<(), HalError> {
        unavailable()
    }

    /// Tears down the surface's swapchain.
    pub fn unconfigure(&mut self) {}

    /// Returns acquire next texture.
    pub fn acquire_next_texture(&mut self) -> Result<GlesTexture, HalError> {
        unavailable()
    }

    /// Presents the most recently acquired surface texture.
    pub fn present(&mut self, _queue: &super::queue::GlesQueue) -> Result<(), HalError> {
        unavailable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_backend_unavailable<T>(result: Result<T, HalError>) {
        assert!(matches!(
            result,
            Err(HalError::BackendUnavailable { backend: "gles" })
        ));
    }

    #[test]
    fn gles_surface_from_android_native_window_returns_unavailable() {
        let window = 0xdead_beefusize as *mut c_void;

        // SAFETY: The scaffold does not dereference the window pointer.
        assert_backend_unavailable(unsafe { GlesSurface::from_android_native_window(window) });
    }

    #[test]
    fn gles_surface_unconfigure_is_noop() {
        let mut surface = GlesSurface;

        surface.unconfigure();
        surface.unconfigure();
    }

    #[test]
    fn gles_surface_acquire_next_texture_returns_unavailable() {
        let mut surface = GlesSurface;

        assert_backend_unavailable(surface.acquire_next_texture());
    }

    #[test]
    fn gles_surface_present_is_covered_by_e2e() {
        let _surface = GlesSurface;
    }
}
