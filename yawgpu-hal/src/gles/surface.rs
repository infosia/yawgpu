use std::sync::Arc;

use glow::HasContext;
use parking_lot::Mutex;

use super::device::{EglDeviceState, GlesDevice, GlesDeviceInner};
use super::egl::EglSurface;
use super::instance::{EglInstanceState, GlesInstanceInner};
use super::texture::GlesTexture;
use super::BACKEND;
use crate::{
    HalError, HalPresentMode, HalSurfaceConfiguration, HalTextureDescriptor, HalTextureFormat,
    HalTextureUsage,
};

pub(super) struct GlesSurfaceInner {
    instance: Arc<GlesInstanceInner>,
    window_surface: EglSurface,
    state: Mutex<GlesSurfaceState>,
}

// SAFETY: The EGL window surface handle is only used while holding the
// configured device's make-current lock, and configured state is mutexed.
unsafe impl Send for GlesSurfaceInner {}
// SAFETY: See the `Send` impl; shared access is synchronized by the state
// mutex and the device make-current lock.
unsafe impl Sync for GlesSurfaceInner {}

impl Drop for GlesSurfaceInner {
    fn drop(&mut self) {
        if let GlesInstanceInner::Egl(egl_state) = self.instance.as_ref() {
            let _ = egl_state
                .egl
                .make_current(egl_state.display, None, None, None);
            let _ = egl_state
                .egl
                .destroy_surface(egl_state.display, self.window_surface);
        }
    }
}

#[derive(Default)]
struct GlesSurfaceState {
    configured: Option<ConfiguredSurface>,
}

struct ConfiguredSurface {
    device: Arc<GlesDeviceInner>,
    back_buffer: GlesTexture,
    width: u32,
    height: u32,
    swap_interval: i32,
}

/// Stores GLES surface data used by validation and backend submission.
pub struct GlesSurface {
    inner: Arc<GlesSurfaceInner>,
}

// SAFETY: EGL/GL operations are synchronized through the configured device's
// make-current mutex, and mutable surface state is protected by a mutex.
unsafe impl Send for GlesSurface {}
// SAFETY: See the `Send` impl; shared access is synchronized.
unsafe impl Sync for GlesSurface {}

impl std::fmt::Debug for GlesSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let configured = self.inner.state.lock().configured.is_some();
        f.debug_struct("GlesSurface")
            .field("configured", &configured)
            .finish()
    }
}

impl GlesSurface {
    pub(super) fn from_window_surface(
        instance: Arc<GlesInstanceInner>,
        window_surface: EglSurface,
    ) -> Self {
        Self {
            inner: Arc::new(GlesSurfaceInner {
                instance,
                window_surface,
                state: Mutex::new(GlesSurfaceState::default()),
            }),
        }
    }

    /// Configures the surface's swapchain for the given format, size, and present mode.
    pub fn configure(
        &mut self,
        device: &GlesDevice,
        config: HalSurfaceConfiguration,
    ) -> Result<(), HalError> {
        validate_config(config)?;
        let device = device.inner_clone();
        let back_buffer = GlesTexture::new(Arc::clone(&device), &back_buffer_descriptor(config));
        back_buffer
            .raw_or_err()
            .map_err(|_| HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "surface back-buffer allocation failed",
            })?;
        let swap_interval = swap_interval_for_present_mode(config.present_mode);
        set_swap_interval(
            &self.inner.instance,
            &device,
            self.inner.window_surface,
            swap_interval,
        )?;
        self.inner.state.lock().configured = Some(ConfiguredSurface {
            device,
            back_buffer,
            width: config.width,
            height: config.height,
            swap_interval,
        });
        Ok(())
    }

    /// Tears down the surface's swapchain.
    pub fn unconfigure(&mut self) {
        self.inner.state.lock().configured = None;
    }

    /// Returns acquire next texture.
    pub fn acquire_next_texture(&mut self) -> Result<GlesTexture, HalError> {
        self.inner
            .state
            .lock()
            .configured
            .as_ref()
            .map(|configured| configured.back_buffer.clone())
            .ok_or(HalError::AcquireFailed {
                backend: BACKEND,
                message: "surface is not configured",
            })
    }

    /// Presents the most recently acquired surface texture.
    pub fn present(&mut self, _queue: &super::queue::GlesQueue) -> Result<(), HalError> {
        let (device, back_buffer, width, height, _swap_interval) = {
            let state = self.inner.state.lock();
            let configured = state.configured.as_ref().ok_or(HalError::PresentFailed {
                backend: BACKEND,
                message: "surface is not configured",
            })?;
            (
                Arc::clone(&configured.device),
                configured.back_buffer.clone(),
                configured.width,
                configured.height,
                configured.swap_interval,
            )
        };
        let texture = back_buffer.raw_or_err()?;
        let width = i32::try_from(width).map_err(|_| HalError::PresentFailed {
            backend: BACKEND,
            message: "surface width exceeds GLES limit",
        })?;
        let height = i32::try_from(height).map_err(|_| HalError::PresentFailed {
            backend: BACKEND,
            message: "surface height exceeds GLES limit",
        })?;
        blit_and_swap(
            &self.inner.instance,
            &device,
            self.inner.window_surface,
            texture,
            width,
            height,
        )
    }
}

fn validate_config(config: HalSurfaceConfiguration) -> Result<(), HalError> {
    if config.format != HalTextureFormat::Rgba8Unorm {
        return Err(HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "GLES surfaces support only Rgba8Unorm format",
        });
    }
    if config.width == 0 || config.height == 0 {
        return Err(HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "surface width and height must be non-zero",
        });
    }
    Ok(())
}

fn back_buffer_descriptor(config: HalSurfaceConfiguration) -> HalTextureDescriptor {
    HalTextureDescriptor {
        format: HalTextureFormat::Rgba8Unorm,
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
        mip_level_count: 1,
        sample_count: 1,
        usage: HalTextureUsage {
            copy_src: true,
            copy_dst: false,
            texture_binding: false,
            storage_binding: false,
            render_attachment: true,
        },
    }
}

fn swap_interval_for_present_mode(mode: HalPresentMode) -> i32 {
    match mode {
        HalPresentMode::Fifo => 1,
        HalPresentMode::Immediate | HalPresentMode::Mailbox => 0,
    }
}

fn set_swap_interval(
    instance: &Arc<GlesInstanceInner>,
    device: &Arc<GlesDeviceInner>,
    window_surface: EglSurface,
    interval: i32,
) -> Result<(), HalError> {
    let _guard = device.current_lock_acquire();
    let egl_state = egl_instance(instance)?;
    let device_state = egl_device(device)?;
    egl_state
        .egl
        .make_current(
            egl_state.display,
            Some(window_surface),
            Some(window_surface),
            Some(device_state.context),
        )
        .map_err(|_| HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "eglMakeCurrent(window) failed",
        })?;
    let _restore = RestoreCurrent { instance, device };
    egl_state
        .egl
        .swap_interval(egl_state.display, interval)
        .map_err(|_| HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "eglSwapInterval failed",
        })
}

fn blit_and_swap(
    instance: &Arc<GlesInstanceInner>,
    device: &Arc<GlesDeviceInner>,
    window_surface: EglSurface,
    back_buffer_texture: glow::Texture,
    width: i32,
    height: i32,
) -> Result<(), HalError> {
    let _guard = device.current_lock_acquire();
    let egl_state = egl_instance(instance)?;
    let device_state = egl_device(device)?;
    egl_state
        .egl
        .make_current(
            egl_state.display,
            Some(window_surface),
            Some(window_surface),
            Some(device_state.context),
        )
        .map_err(|_| HalError::PresentFailed {
            backend: BACKEND,
            message: "eglMakeCurrent(window) failed",
        })?;
    let _restore = RestoreCurrent { instance, device };
    blit_back_buffer_to_window(&device_state.gl, back_buffer_texture, width, height)?;
    egl_state
        .egl
        .swap_buffers(egl_state.display, window_surface)
        .map_err(|_| HalError::PresentFailed {
            backend: BACKEND,
            message: "eglSwapBuffers failed",
        })
}

struct RestoreCurrent<'a> {
    instance: &'a Arc<GlesInstanceInner>,
    device: &'a Arc<GlesDeviceInner>,
}

impl Drop for RestoreCurrent<'_> {
    fn drop(&mut self) {
        let (Some(instance), Some(device)) = (
            egl_instance(self.instance).ok(),
            egl_device(self.device).ok(),
        ) else {
            return;
        };
        let _ = instance.egl.make_current(
            instance.display,
            Some(device.surface),
            Some(device.surface),
            Some(device.context),
        );
    }
}

fn egl_instance(instance: &Arc<GlesInstanceInner>) -> Result<&EglInstanceState, HalError> {
    let GlesInstanceInner::Egl(state) = instance.as_ref() else {
        return Err(HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "GLES surface is only available with the EGL backend",
        });
    };
    Ok(state)
}

fn egl_device(device: &Arc<GlesDeviceInner>) -> Result<&EglDeviceState, HalError> {
    device.egl_state().ok_or(HalError::SwapchainCreationFailed {
        backend: BACKEND,
        message: "GLES surface is only available with the EGL backend",
    })
}

fn blit_back_buffer_to_window(
    gl: &glow::Context,
    texture: glow::Texture,
    width: i32,
    height: i32,
) -> Result<(), HalError> {
    unsafe {
        let fbo = gl
            .create_framebuffer()
            .map_err(|_| HalError::PresentFailed {
                backend: BACKEND,
                message: "glCreateFramebuffer (present) failed",
            })?;
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(fbo));
        gl.framebuffer_texture_2d(
            glow::READ_FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(texture),
            0,
        );
        gl.read_buffer(glow::COLOR_ATTACHMENT0);
        if gl.check_framebuffer_status(glow::READ_FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            gl.delete_framebuffer(fbo);
            return Err(HalError::PresentFailed {
                backend: BACKEND,
                message: "framebuffer incomplete (present)",
            });
        }
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
        gl.blit_framebuffer(
            0,
            0,
            width,
            height,
            0,
            0,
            width,
            height,
            glow::COLOR_BUFFER_BIT,
            glow::NEAREST,
        );
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        gl.delete_framebuffer(fbo);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_interval_maps_present_modes() {
        assert_eq!(swap_interval_for_present_mode(HalPresentMode::Fifo), 1);
        assert_eq!(swap_interval_for_present_mode(HalPresentMode::Immediate), 0);
        assert_eq!(swap_interval_for_present_mode(HalPresentMode::Mailbox), 0);
    }

    #[test]
    fn validate_config_accepts_rgba8_non_zero_surface() {
        let config = HalSurfaceConfiguration::new(
            HalTextureFormat::Rgba8Unorm,
            HalTextureUsage {
                copy_src: false,
                copy_dst: false,
                texture_binding: false,
                storage_binding: false,
                render_attachment: true,
            },
            320,
            240,
            HalPresentMode::Fifo,
        );

        assert!(validate_config(config).is_ok());
    }

    #[test]
    fn validate_config_rejects_unsupported_format_and_zero_size() {
        let usage = HalTextureUsage {
            copy_src: false,
            copy_dst: false,
            texture_binding: false,
            storage_binding: false,
            render_attachment: true,
        };
        let format = HalSurfaceConfiguration::new(
            HalTextureFormat::Bgra8Unorm,
            usage,
            320,
            240,
            HalPresentMode::Fifo,
        );
        assert!(matches!(
            validate_config(format),
            Err(HalError::SwapchainCreationFailed {
                backend: "gles",
                message: "GLES surfaces support only Rgba8Unorm format",
            })
        ));

        let zero_size = HalSurfaceConfiguration::new(
            HalTextureFormat::Rgba8Unorm,
            usage,
            0,
            240,
            HalPresentMode::Fifo,
        );
        assert!(matches!(
            validate_config(zero_size),
            Err(HalError::SwapchainCreationFailed {
                backend: "gles",
                message: "surface width and height must be non-zero",
            })
        ));
    }
}
