use super::*;

/// Returns deterministic Noop surface capabilities.
///
/// # Safety
///
/// `surface` and `adapter` must be non-null live yawgpu handles.
/// `capabilities`, when non-null, must point to writable memory.
/// Returns WGPU surface get capabilities.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceGetCapabilities(
    surface: native::WGPUSurface,
    adapter: native::WGPUAdapter,
    capabilities: *mut native::WGPUSurfaceCapabilities,
) -> native::WGPUStatus {
    let surface = borrow_handle(surface, "WGPUSurface");
    let _adapter = borrow_handle(adapter, "WGPUAdapter");
    let Some(capabilities) = capabilities.as_mut() else {
        return native::WGPUStatus_Error;
    };
    if surface.is_error {
        return native::WGPUStatus_Error;
    }
    capabilities.nextInChain = std::ptr::null_mut();
    capabilities.usages = SURFACE_USAGES;
    capabilities.formatCount = SURFACE_FORMATS.len();
    capabilities.formats = Box::leak(Box::new(SURFACE_FORMATS)).as_ptr();
    capabilities.presentModeCount = SURFACE_PRESENT_MODES.len();
    capabilities.presentModes = Box::leak(Box::new(SURFACE_PRESENT_MODES)).as_ptr();
    capabilities.alphaModeCount = SURFACE_ALPHA_MODES.len();
    capabilities.alphaModes = Box::leak(Box::new(SURFACE_ALPHA_MODES)).as_ptr();
    native::WGPUStatus_Success
}

/// Frees arrays allocated by `wgpuSurfaceGetCapabilities`.
///
/// # Safety
///
/// Any non-null array member must have been returned by yawgpu.
/// Returns WGPU surface capabilities free members.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceCapabilitiesFreeMembers(
    capabilities: native::WGPUSurfaceCapabilities,
) {
    if !capabilities.formats.is_null() {
        drop(Box::from_raw(
            capabilities.formats as *mut [native::WGPUTextureFormat; SURFACE_FORMATS.len()],
        ));
    }
    if !capabilities.presentModes.is_null() {
        drop(Box::from_raw(
            capabilities.presentModes
                as *mut [native::WGPUPresentMode; SURFACE_PRESENT_MODES.len()],
        ));
    }
    if !capabilities.alphaModes.is_null() {
        drop(Box::from_raw(
            capabilities.alphaModes
                as *mut [native::WGPUCompositeAlphaMode; SURFACE_ALPHA_MODES.len()],
        ));
    }
}

/// Configures a surface after validating it against Noop capabilities.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle. `config`, when
/// non-null, must point to a valid `WGPUSurfaceConfiguration`.
/// Returns WGPU surface configure.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceConfigure(
    surface: native::WGPUSurface,
    config: *const native::WGPUSurfaceConfiguration,
) {
    let surface = borrow_handle(surface, "WGPUSurface");
    let Some(config) = config.as_ref() else {
        return;
    };
    if config.device.is_null() {
        return;
    }
    let device = borrow_handle(config.device, "WGPUDevice");
    if surface.is_error {
        device.dispatch_error(core::ErrorKind::Validation, "surface is invalid");
        return;
    }
    if device.core.is_lost() {
        device.dispatch_error(
            core::ErrorKind::Validation,
            "surface configuration device is lost",
        );
        return;
    }
    if let Some(message) = surface_configuration_error(device, config) {
        device.dispatch_error(core::ErrorKind::Validation, message);
        return;
    }
    let view_formats = if config.viewFormatCount == 0 {
        Vec::new()
    } else if config.viewFormats.is_null() {
        device.dispatch_error(
            core::ErrorKind::Validation,
            "surface configuration viewFormats pointer is null",
        );
        return;
    } else {
        std::slice::from_raw_parts(config.viewFormats, config.viewFormatCount).to_vec()
    };
    if let Some(hal) = surface
        .hal
        .lock()
        .expect("surface HAL lock is not poisoned")
        .as_mut()
    {
        let hal_config = HalSurfaceConfiguration::new(
            hal_surface_format(config.format),
            hal_surface_usage(config.usage),
            config.width,
            config.height,
            hal_present_mode(config.presentMode),
        );
        if let Err(error) = hal.configure(device.core.hal(), hal_config) {
            device.dispatch_error(core::ErrorKind::Internal, error.to_string());
            return;
        }
    }
    *surface
        .configured
        .lock()
        .expect("surface configuration lock is not poisoned") = Some(SurfaceConfigurationState {
        device: Arc::clone(&device.core),
        format: config.format,
        usage: config.usage,
        width: config.width,
        height: config.height,
        view_formats,
        _present_mode: config.presentMode,
        _alpha_mode: config.alphaMode,
    });
}

/// Clears any stored surface configuration.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle.
/// Returns WGPU surface unconfigure.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceUnconfigure(surface: native::WGPUSurface) {
    let surface = borrow_handle(surface, "WGPUSurface");
    *surface
        .configured
        .lock()
        .expect("surface configuration lock is not poisoned") = None;
    if let Some(hal) = surface
        .hal
        .lock()
        .expect("surface HAL lock is not poisoned")
        .as_mut()
    {
        hal.unconfigure();
    }
}

/// Gets the current surface texture.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle. `surface_texture`,
/// when non-null, must point to writable memory.
/// Returns WGPU surface get current texture.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceGetCurrentTexture(
    surface: native::WGPUSurface,
    surface_texture: *mut native::WGPUSurfaceTexture,
) {
    let surface = borrow_handle(surface, "WGPUSurface");
    let Some(surface_texture) = surface_texture.as_mut() else {
        return;
    };
    surface_texture.nextInChain = std::ptr::null_mut();
    surface_texture.texture = std::ptr::null();
    let config = surface
        .configured
        .lock()
        .expect("surface configuration lock is not poisoned")
        .clone();
    if surface.is_error || config.is_none() {
        surface_texture.status = native::WGPUSurfaceGetCurrentTextureStatus_Error;
        return;
    }
    let config = config.expect("surface configuration was checked");
    if config.device.is_lost() {
        surface_texture.status = native::WGPUSurfaceGetCurrentTextureStatus_Error;
        return;
    }
    if let Some(hal) = surface
        .hal
        .lock()
        .expect("surface HAL lock is not poisoned")
        .as_mut()
    {
        match hal.acquire_next_texture() {
            Ok(hal_texture) => {
                let descriptor = core::TextureDescriptor {
                    usage: map_texture_usage(config.usage),
                    dimension: core::TextureDimension::D2,
                    size: core::Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    format: crate::conv::map_texture_format(config.format),
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats: config
                        .view_formats
                        .iter()
                        .copied()
                        .map(crate::conv::map_texture_format)
                        .collect(),
                };
                let texture = Arc::new(WGPUTextureImpl {
                    core: Arc::new(core::Texture::from_hal(descriptor, hal_texture)),
                    device: Arc::clone(&config.device),
                    instance: Arc::clone(&surface._instance),
                });
                surface_texture.texture = arc_to_handle(texture);
                surface_texture.status = native::WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal;
                return;
            }
            Err(_) => {
                surface_texture.status = native::WGPUSurfaceGetCurrentTextureStatus_Error;
                return;
            }
        }
    }
    // Noop has no native window/backbuffer, so a valid configuration still
    // cannot produce a swapchain image. This is the recorded SF3 N/A boundary.
    surface_texture.status = native::WGPUSurfaceGetCurrentTextureStatus_Lost;
}

/// Presents the current surface texture. Noop has no presentation backend.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle.
/// Returns WGPU surface present.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfacePresent(surface: native::WGPUSurface) -> native::WGPUStatus {
    let surface = borrow_handle(surface, "WGPUSurface");
    if let Some(hal) = surface
        .hal
        .lock()
        .expect("surface HAL lock is not poisoned")
        .as_mut()
    {
        let config = surface
            .configured
            .lock()
            .expect("surface configuration lock is not poisoned")
            .clone();
        let Some(config) = config else {
            return native::WGPUStatus_Error;
        };
        if config.device.is_lost() {
            return native::WGPUStatus_Error;
        }
        if hal.present(config.device.queue().hal()).is_err() {
            return native::WGPUStatus_Error;
        }
    }
    native::WGPUStatus_Success
}

/// Sets the debug label for a surface.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle. `label` must point
/// to valid string data according to `WGPUStringView` when non-empty.
/// Returns WGPU surface set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceSetLabel(
    surface: native::WGPUSurface,
    label: native::WGPUStringView,
) {
    let surface = borrow_handle(surface, "WGPUSurface");
    *surface
        .label
        .lock()
        .expect("surface label lock is not poisoned") =
        label_from_string_view(label).unwrap_or_default();
}

/// Releases one owned reference to a surface handle.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle.
/// Returns WGPU surface release.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceRelease(surface: native::WGPUSurface) {
    release_handle(surface, "WGPUSurface");
}

/// Adds one owned reference to a surface handle.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle.
/// Returns WGPU surface add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceAddRef(surface: native::WGPUSurface) {
    add_ref_handle(surface, "WGPUSurface");
}
