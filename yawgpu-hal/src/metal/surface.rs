use super::*;

#[derive(Debug)]
pub struct MetalSurface {
    pub(super) layer: Retained<CAMetalLayer>,
    pub(super) current_drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
    pub(super) config: Option<HalSurfaceConfiguration>,
}

unsafe impl Send for MetalSurface {}
unsafe impl Sync for MetalSurface {}

impl MetalSurface {
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
        })
    }

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
        let _ = config.usage;
        let _ = config.present_mode;
        self.current_drawable = None;
        self.config = Some(config);
        Ok(())
    }

    pub fn unconfigure(&mut self) {
        self.current_drawable = None;
        self.config = None;
    }

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
        self.current_drawable = Some(drawable);
        let (_, bytes_per_pixel) = map_texture_format(config.format)?;
        Ok(MetalTexture {
            inner: Some(texture),
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
            bytes_per_pixel,
        })
    }

    pub fn present(&mut self, queue: &MetalQueue) -> Result<(), HalError> {
        let drawable = self
            .current_drawable
            .take()
            .ok_or(HalError::PresentFailed {
                backend: BACKEND,
                message: "no acquired drawable to present",
            })?;
        let _ = queue;
        drawable.present();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::super::*;
    use super::*;

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
