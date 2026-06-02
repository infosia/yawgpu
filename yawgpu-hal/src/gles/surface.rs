use std::sync::Arc;

use glow::HasContext;
use parking_lot::Mutex;

use super::device::{EglDeviceState, GlesDevice, GlesDeviceInner};
use super::egl::EglSurface;
use super::instance::{EglInstanceState, GlesInstanceInner};
use super::texture::GlesTexture;
use super::BACKEND;
use crate::{
    HalError, HalPresentMode, HalSurfaceConfiguration, HalTextureDescriptor, HalTextureDimension,
    HalTextureFormat, HalTextureUsage,
};

pub(super) struct GlesSurfaceInner {
    state: Mutex<GlesSurfaceState>,
    kind: GlesSurfaceKind,
}

enum GlesSurfaceKind {
    Egl(EglSurfaceKind),
    #[cfg(windows)]
    Wgl(WglSurfaceKind),
}

struct EglSurfaceKind {
    instance: Arc<GlesInstanceInner>,
    window_surface: EglSurface,
}

#[cfg(windows)]
struct WglSurfaceKind {
    surface: super::wgl::WglSurfaceState,
}

// SAFETY: The EGL window surface handle is only used while holding the
// configured device's make-current lock, and configured state is mutexed.
unsafe impl Send for GlesSurfaceInner {}
// SAFETY: See the `Send` impl; shared access is synchronized by the state
// mutex and the device make-current lock.
unsafe impl Sync for GlesSurfaceInner {}

impl Drop for GlesSurfaceInner {
    fn drop(&mut self) {
        match &self.kind {
            GlesSurfaceKind::Egl(kind) => {
                if let GlesInstanceInner::Egl(egl_state) = kind.instance.as_ref() {
                    let _ = egl_state
                        .egl
                        .make_current(egl_state.display, None, None, None);
                    let _ = egl_state
                        .egl
                        .destroy_surface(egl_state.display, kind.window_surface);
                }
            }
            #[cfg(windows)]
            GlesSurfaceKind::Wgl(kind) => {
                // `release_surface_dc` is `wglMakeCurrent(NULL,NULL)` (per-thread)
                // + `ReleaseDC(hwnd, hdc)` (per-HDC). Neither races with device-
                // side GL operations on other threads, so no make-current lock
                // is needed — symmetric with the EGL arm above.
                super::wgl::release_surface_dc(&kind.surface);
            }
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
    pub(super) fn from_egl_window(
        instance: Arc<GlesInstanceInner>,
        window_surface: EglSurface,
    ) -> Self {
        Self {
            inner: Arc::new(GlesSurfaceInner {
                state: Mutex::new(GlesSurfaceState::default()),
                kind: GlesSurfaceKind::Egl(EglSurfaceKind {
                    instance,
                    window_surface,
                }),
            }),
        }
    }

    #[cfg(windows)]
    pub(super) fn from_wgl_window(
        _instance: Arc<GlesInstanceInner>,
        surface: super::wgl::WglSurfaceState,
    ) -> Self {
        Self {
            inner: Arc::new(GlesSurfaceInner {
                state: Mutex::new(GlesSurfaceState::default()),
                kind: GlesSurfaceKind::Wgl(WglSurfaceKind { surface }),
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
        set_swap_interval(&self.inner.kind, &device, swap_interval)?;
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
        blit_and_swap(&self.inner.kind, &device, texture, width, height)
    }
}

fn validate_config(config: HalSurfaceConfiguration) -> Result<(), HalError> {
    if !matches!(
        config.format,
        HalTextureFormat::Rgba8Unorm | HalTextureFormat::Bgra8Unorm
    ) {
        return Err(HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "GLES surfaces support only Rgba8Unorm and Bgra8Unorm formats",
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
        dimension: HalTextureDimension::D2,
        format: config.format,
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
    kind: &GlesSurfaceKind,
    device: &Arc<GlesDeviceInner>,
    interval: i32,
) -> Result<(), HalError> {
    let _guard = device.current_lock_acquire();
    match kind {
        GlesSurfaceKind::Egl(kind) => {
            let egl_state = egl_instance(&kind.instance)?;
            let device_state = egl_device(device)?;
            egl_state
                .egl
                .make_current(
                    egl_state.display,
                    Some(kind.window_surface),
                    Some(kind.window_surface),
                    Some(device_state.context),
                )
                .map_err(|_| HalError::SwapchainCreationFailed {
                    backend: BACKEND,
                    message: "eglMakeCurrent(window) failed",
                })?;
            let _restore = RestoreCurrent::Egl {
                instance: &kind.instance,
                device,
            };
            egl_state
                .egl
                .swap_interval(egl_state.display, interval)
                .map_err(|_| HalError::SwapchainCreationFailed {
                    backend: BACKEND,
                    message: "eglSwapInterval failed",
                })
        }
        #[cfg(windows)]
        GlesSurfaceKind::Wgl(kind) => {
            let device_state = wgl_device(device)?;
            device_state
                .make_current_on_hdc(kind.surface.hdc())
                .map_err(|_| HalError::SwapchainCreationFailed {
                    backend: BACKEND,
                    message: "wglMakeCurrent(window) failed",
                })?;
            let _restore = RestoreCurrent::Wgl { device };
            super::wgl::swap_interval(interval);
            Ok(())
        }
    }
}

fn blit_and_swap(
    kind: &GlesSurfaceKind,
    device: &Arc<GlesDeviceInner>,
    back_buffer_texture: glow::Texture,
    width: i32,
    height: i32,
) -> Result<(), HalError> {
    let _guard = device.current_lock_acquire();
    match kind {
        GlesSurfaceKind::Egl(kind) => {
            let egl_state = egl_instance(&kind.instance)?;
            let device_state = egl_device(device)?;
            egl_state
                .egl
                .make_current(
                    egl_state.display,
                    Some(kind.window_surface),
                    Some(kind.window_surface),
                    Some(device_state.context),
                )
                .map_err(|_| HalError::PresentFailed {
                    backend: BACKEND,
                    message: "eglMakeCurrent(window) failed",
                })?;
            let _restore = RestoreCurrent::Egl {
                instance: &kind.instance,
                device,
            };
            blit_back_buffer_to_window(&device_state.gl, back_buffer_texture, width, height)?;
            egl_state
                .egl
                .swap_buffers(egl_state.display, kind.window_surface)
                .map_err(|_| HalError::PresentFailed {
                    backend: BACKEND,
                    message: "eglSwapBuffers failed",
                })
        }
        #[cfg(windows)]
        GlesSurfaceKind::Wgl(kind) => {
            let device_state = wgl_device(device)?;
            device_state.make_current_on_hdc(kind.surface.hdc())?;
            let _restore = RestoreCurrent::Wgl { device };
            blit_back_buffer_to_window(device_state.gl(), back_buffer_texture, width, height)?;
            super::wgl::swap_buffers(kind.surface.hdc())
        }
    }
}

enum RestoreCurrent<'a> {
    Egl {
        instance: &'a Arc<GlesInstanceInner>,
        device: &'a Arc<GlesDeviceInner>,
    },
    #[cfg(windows)]
    Wgl { device: &'a Arc<GlesDeviceInner> },
}

impl Drop for RestoreCurrent<'_> {
    fn drop(&mut self) {
        match self {
            RestoreCurrent::Egl { instance, device } => {
                let (Some(instance), Some(device)) =
                    (egl_instance(instance).ok(), egl_device(device).ok())
                else {
                    return;
                };
                let _ = instance.egl.make_current(
                    instance.display,
                    Some(device.surface),
                    Some(device.surface),
                    Some(device.context),
                );
            }
            #[cfg(windows)]
            RestoreCurrent::Wgl { device } => {
                if let Ok(device) = wgl_device(device) {
                    device.restore_current();
                }
            }
        }
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

#[cfg(windows)]
fn wgl_device(device: &Arc<GlesDeviceInner>) -> Result<&super::wgl::WglDeviceState, HalError> {
    let GlesDeviceInner::Wgl(state) = device.as_ref() else {
        return Err(HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "GLES WGL surface requires a WGL device",
        });
    };
    Ok(state)
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
    fn validate_config_accepts_rgba8_and_bgra8_non_zero_surface() {
        for format in [HalTextureFormat::Rgba8Unorm, HalTextureFormat::Bgra8Unorm] {
            let config = HalSurfaceConfiguration::new(
                format,
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
            HalTextureFormat::Depth24Plus,
            usage,
            320,
            240,
            HalPresentMode::Fifo,
        );
        assert!(matches!(
            validate_config(format),
            Err(HalError::SwapchainCreationFailed {
                backend: "gles",
                message: "GLES surfaces support only Rgba8Unorm and Bgra8Unorm formats",
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
