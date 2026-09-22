use super::*;
use crate::{HalCompositeAlphaMode, HalSurfaceCapabilities, HalTextureDimension};

/// Stores metal surface data used by validation and backend submission.
#[derive(Debug)]
pub struct MetalSurface {
    pub(super) layer: Retained<CAMetalLayer>,
    pub(super) current_drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
    pub(super) config: Option<HalSurfaceConfiguration>,
    pub(super) device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
}

// SAFETY: `MetalSurface` owns a `Retained<CAMetalLayer>` plus an optional
// drawable / config snapshot. The `CAMetalLayer` reference itself can be
// moved across threads — `Retained<...>` performs `retain` / `release`
// atomically, and Apple permits `nextDrawable` (called from
// `acquire_next_texture`) off the main thread. The remaining configuration-
// mutating accessors (`setDrawableSize:`, `setPixelFormat:`,
// `setDevice:` — all driven from `MetalSurface::configure`) are
// main-thread-only per Apple's QuartzCore docs. yawgpu's examples and tests
// only invoke `configure` from the same thread that owns the window
// (`main` on macOS, where GLFW already requires that), so the
// main-thread invariant holds in practice. Sharing the surface across
// threads beyond what the examples exercise (e.g. driving `configure`
// from a worker) would violate that invariant; this loose `Send` /
// `Sync` matches the broader HAL convention and is documented here so a
// future Send/Sync audit can tighten it (e.g. wrap mutating ops in a
// main-thread runner) without missing the constraint.
unsafe impl Send for MetalSurface {}
unsafe impl Sync for MetalSurface {}

impl MetalSurface {
    /// Returns the formats and modes supported by CAMetalLayer.
    #[must_use]
    pub fn capabilities(&self) -> HalSurfaceCapabilities {
        HalSurfaceCapabilities {
            usages: HalTextureUsage {
                render_attachment: true,
                texture_binding: true,
                copy_src: true,
                copy_dst: true,
                storage_binding: false,
                transient: false,
            },
            formats: vec![
                HalTextureFormat::Bgra8Unorm,
                HalTextureFormat::Bgra8UnormSrgb,
                HalTextureFormat::Rgba16Float,
                #[cfg(target_os = "macos")]
                HalTextureFormat::Rgb10a2Unorm,
            ],
            present_modes: vec![
                HalPresentMode::Fifo,
                HalPresentMode::Immediate,
                HalPresentMode::Mailbox,
            ],
            alpha_modes: vec![
                HalCompositeAlphaMode::Opaque,
                HalCompositeAlphaMode::Premultiplied,
            ],
        }
    }

    /// # Safety
    ///
    /// `layer` must be a valid, non-dangling `CAMetalLayer` instance pointer.
    pub unsafe fn from_layer(layer: *mut c_void) -> Result<Self, HalError> {
        let layer = unsafe { Retained::retain(layer.cast::<CAMetalLayer>()) }.ok_or(
            HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "surface layer is null",
            },
        )?;
        Ok(Self {
            layer,
            current_drawable: None,
            config: None,
            device: None,
        })
    }

    /// Configures the surface's swapchain for the given format, size, and present mode.
    pub fn configure(
        &mut self,
        device: &MetalDevice,
        config: HalSurfaceConfiguration,
    ) -> Result<(), HalError> {
        let (pixel_format, _) = map_texture_format(config.format)?;
        self.layer.setDevice(Some(&device.device));
        self.layer.setPixelFormat(pixel_format);
        self.layer.setFramebufferOnly(false);
        self.layer.setDrawableSize(CGSize {
            width: f64::from(config.width),
            height: f64::from(config.height),
        });
        self.layer
            .setDisplaySyncEnabled(config.present_mode != HalPresentMode::Immediate);
        self.layer
            .setOpaque(config.alpha_mode != HalCompositeAlphaMode::Premultiplied);
        self.current_drawable = None;
        self.device = Some(device.device.clone());
        self.config = Some(config);
        Ok(())
    }

    /// Tears down the surface's swapchain.
    pub fn unconfigure(&mut self) {
        self.current_drawable = None;
        self.config = None;
        self.device = None;
    }

    /// Returns acquire next texture.
    pub fn acquire_next_texture(&mut self) -> Result<MetalTexture, HalError> {
        let config = self.config.ok_or(HalError::AcquireFailed {
            backend: BACKEND,
            message: "surface is not configured",
        })?;
        let drawable = self.layer.nextDrawable().ok_or(HalError::AcquireFailed {
            backend: BACKEND,
            message: "nextDrawable returned null",
        })?;
        let texture = drawable.texture();
        let device = self.device.clone().ok_or(HalError::AcquireFailed {
            backend: BACKEND,
            message: "surface is not configured",
        })?;
        self.current_drawable = Some(drawable);
        let (_, bytes_per_pixel) = map_texture_format(config.format)?;
        Ok(MetalTexture {
            inner: Some(texture),
            device,
            format: config.format,
            dimension: HalTextureDimension::D2,
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
            sample_count: 1,
            bytes_per_pixel,
        })
    }

    /// Submits a present command buffer for the most recently acquired drawable.
    pub fn present(&mut self, queue: &MetalQueue) -> Result<(), HalError> {
        let drawable = self
            .current_drawable
            .take()
            .ok_or(HalError::PresentFailed {
                backend: BACKEND,
                message: "no acquired drawable to present",
            })?;
        autoreleasepool(|_| {
            let command_buffer = queue.inner.commandBuffer().ok_or(HalError::PresentFailed {
                backend: BACKEND,
                message: "present command buffer creation failed",
            })?;
            let drawable_ref: &ProtocolObject<dyn MTLDrawable> =
                unsafe { &*((&*drawable as *const ProtocolObject<dyn CAMetalDrawable>).cast()) };
            command_buffer.presentDrawable(drawable_ref);
            command_buffer.commit();
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::super::*;
    use super::*;

    #[test]
    #[ignore = "manual real Metal backend test"]
    fn metal_surface_capabilities_and_resolved_modes() {
        let device = metal_device();
        let layer = metal_layer();
        let raw = (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>();
        let mut surface = unsafe { MetalSurface::from_layer(raw) }.unwrap();
        let caps = surface.capabilities();
        let expected = vec![
            HalTextureFormat::Bgra8Unorm,
            HalTextureFormat::Bgra8UnormSrgb,
            HalTextureFormat::Rgba16Float,
            #[cfg(target_os = "macos")]
            HalTextureFormat::Rgb10a2Unorm,
        ];
        assert_eq!(caps.formats, expected);
        assert_eq!(
            caps.present_modes,
            [
                HalPresentMode::Fifo,
                HalPresentMode::Immediate,
                HalPresentMode::Mailbox
            ]
        );
        assert_eq!(
            caps.alpha_modes,
            [
                HalCompositeAlphaMode::Opaque,
                HalCompositeAlphaMode::Premultiplied
            ]
        );
        assert!(
            caps.usages.render_attachment
                && caps.usages.texture_binding
                && caps.usages.copy_src
                && caps.usages.copy_dst
        );
        assert!(!caps.usages.storage_binding && !caps.usages.transient);
        for format in caps.formats {
            let mut config = surface_config();
            config.format = format;
            config.present_mode = HalPresentMode::Immediate;
            config.alpha_mode = HalCompositeAlphaMode::Premultiplied;
            surface.configure(&device, config).unwrap();
            assert!(!layer.displaySyncEnabled());
            assert!(!layer.isOpaque());
            assert!(!layer.framebufferOnly());
            assert_eq!(layer.pixelFormat(), map_texture_format(format).unwrap().0);
            config.present_mode = HalPresentMode::Mailbox;
            config.alpha_mode = HalCompositeAlphaMode::Opaque;
            surface.configure(&device, config).unwrap();
            assert!(layer.displaySyncEnabled());
            assert!(layer.isOpaque());
        }
        let surface = crate::HalSurface::Metal(surface);
        let noop = crate::HalInstance::new_noop()
            .enumerate_adapters()
            .remove(0);
        assert!(surface.capabilities(&noop).is_err());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_surface_from_layer_rejects_null_layer() {
        let error = unsafe { MetalSurface::from_layer(std::ptr::null_mut()) }
            .expect_err("null layer must fail");
        assert!(matches!(
            error,
            HalError::SwapchainCreationFailed {
                backend: "metal",
                message: "surface layer is null"
            }
        ));
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_surface_from_layer_wraps_cametal_layer() {
        let layer = metal_layer();
        let raw = (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>();
        let surface = unsafe { MetalSurface::from_layer(raw) }.expect("create Metal surface");
        assert!(surface.config.is_none());
        assert!(surface.current_drawable.is_none());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_surface_configure_stores_configuration() {
        let device = metal_device();
        let layer = metal_layer();
        let raw = (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>();
        let mut surface = unsafe { MetalSurface::from_layer(raw) }.expect("create Metal surface");
        let config = surface_config();
        surface
            .configure(&device, config)
            .expect("configure Metal surface");
        assert_eq!(surface.config.expect("stored config").width, 100);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_surface_unconfigure_clears_configuration() {
        let device = metal_device();
        let layer = metal_layer();
        let raw = (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>();
        let mut surface = unsafe { MetalSurface::from_layer(raw) }.expect("create Metal surface");
        surface
            .configure(&device, surface_config())
            .expect("configure Metal surface");
        surface.unconfigure();
        assert!(surface.config.is_none());
        assert!(surface.current_drawable.is_none());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_surface_acquire_next_texture_errors_when_unconfigured() {
        let layer = metal_layer();
        let raw = (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>();
        let mut surface = unsafe { MetalSurface::from_layer(raw) }.expect("create Metal surface");
        let error = surface
            .acquire_next_texture()
            .expect_err("unconfigured surface must fail");
        assert!(matches!(
            error,
            HalError::AcquireFailed {
                backend: "metal",
                message: "surface is not configured"
            }
        ));
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_surface_present_errors_without_acquired_drawable() {
        let device = metal_device();
        let layer = metal_layer();
        let raw = (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>();
        let mut surface = unsafe { MetalSurface::from_layer(raw) }.expect("create Metal surface");
        let error = surface
            .present(device.queue())
            .expect_err("surface without drawable must fail");
        assert!(matches!(
            error,
            HalError::PresentFailed {
                backend: "metal",
                message: "no acquired drawable to present"
            }
        ));
    }
}
