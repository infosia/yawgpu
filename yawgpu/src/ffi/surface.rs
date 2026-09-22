use super::*;

/// Returns the HAL surface capabilities for the adapter.
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
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    let Some(capabilities) = capabilities.as_mut() else {
        return native::WGPUStatus_Error;
    };
    *capabilities = std::mem::zeroed();
    if surface.is_error {
        return native::WGPUStatus_Error;
    }
    let Ok(caps) = query_surface_capabilities(surface, adapter.core.hal()) else {
        return native::WGPUStatus_Error;
    };
    fill_surface_capabilities(capabilities, caps);
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
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            capabilities.formats.cast_mut(),
            capabilities.formatCount,
        )));
    }
    if !capabilities.presentModes.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            capabilities.presentModes.cast_mut(),
            capabilities.presentModeCount,
        )));
    }
    if !capabilities.alphaModes.is_null() {
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            capabilities.alphaModes.cast_mut(),
            capabilities.alphaModeCount,
        )));
    }
}

/// Configures a surface after validating it against HAL capabilities.
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
    let caps = match query_surface_capabilities(surface, device.adapter.hal()) {
        Ok(caps) => caps,
        Err(error) => {
            device.dispatch_error(core::ErrorKind::Validation, error.to_string());
            return;
        }
    };
    if let Some(message) = surface_configuration_error(device, config, &caps) {
        device.dispatch_error(core::ErrorKind::Validation, message);
        return;
    }
    let present_mode = resolved_present_mode(config.presentMode);
    let alpha_mode = resolved_alpha_mode(config.alphaMode, &caps);
    let view_formats = if config.viewFormatCount == 0 {
        Vec::new()
    } else if config.viewFormats.is_null() {
        device.dispatch_error(
            core::ErrorKind::Validation,
            "surface configuration viewFormats pointer is null",
        );
        return;
    } else {
        std::slice::from_raw_parts(config.viewFormats, config.viewFormatCount)
            .iter()
            .copied()
            .map(crate::conv::map_texture_format)
            .collect()
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
            hal_present_mode(present_mode),
            hal_alpha_mode(alpha_mode),
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
        _present_mode: present_mode,
        _alpha_mode: alpha_mode,
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
        .as_ref()
        .map(|config| {
            (
                Arc::clone(&config.device),
                config.usage,
                config.width,
                config.height,
                config.format,
                config.view_formats.clone(),
            )
        });
    if surface.is_error || config.is_none() {
        surface_texture.status = native::WGPUSurfaceGetCurrentTextureStatus_Error;
        return;
    }
    let (device, usage, width, height, format, view_formats) =
        config.expect("surface configuration was checked");
    if let Some(hal) = surface
        .hal
        .lock()
        .expect("surface HAL lock is not poisoned")
        .as_mut()
    {
        match hal.acquire_next_texture() {
            Ok(hal_texture) => {
                let descriptor = core::TextureDescriptor {
                    usage: map_texture_usage(usage),
                    dimension: core::TextureDimension::D2,
                    size: core::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    format: crate::conv::map_texture_format(format),
                    mip_level_count: 1,
                    sample_count: 1,
                    view_formats,
                };
                let texture = Arc::new(WGPUTextureImpl {
                    core: Arc::new(core::Texture::from_hal(descriptor, hal_texture)),
                    device,
                    instance: Arc::clone(&surface._instance),
                    label: Mutex::new(None),
                    binding_view_dimension: native::WGPUTextureViewDimension_Undefined,
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
        let device = surface
            .configured
            .lock()
            .expect("surface configuration lock is not poisoned")
            .as_ref()
            .map(|config| Arc::clone(&config.device));
        let Some(device) = device else {
            return native::WGPUStatus_Error;
        };
        if hal.present(device.queue().hal()).is_err() {
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
        .expect("surface label lock is not poisoned") = label_from_string_view(label);
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

fn query_surface_capabilities(
    surface: &WGPUSurfaceImpl,
    adapter: &yawgpu_hal::HalAdapter,
) -> Result<yawgpu_hal::HalSurfaceCapabilities, yawgpu_hal::HalError> {
    let hal = surface
        .hal
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    hal.as_ref()
        .unwrap_or(&HalSurface::Noop)
        .capabilities(adapter)
}

fn fill_surface_capabilities(
    output: &mut native::WGPUSurfaceCapabilities,
    caps: yawgpu_hal::HalSurfaceCapabilities,
) {
    let formats = caps
        .formats
        .into_iter()
        .map(native_surface_format)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let modes = caps
        .present_modes
        .into_iter()
        .map(native_present_mode)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let alphas = caps
        .alpha_modes
        .into_iter()
        .map(native_alpha_mode)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    output.usages = native_surface_usage(caps.usages);
    output.formatCount = formats.len();
    output.formats = Box::into_raw(formats).cast();
    output.presentModeCount = modes.len();
    output.presentModes = Box::into_raw(modes).cast();
    output.alphaModeCount = alphas.len();
    output.alphaModes = Box::into_raw(alphas).cast();
}

#[cfg(test)]
mod tests {
    use super::super::tests::{
        noop_chain, noop_surface_caps, release_handles, valid_surface_config,
    };
    use super::*;
    use yawgpu_hal::{HalCompositeAlphaMode as A, HalSurfaceCapabilities};

    fn varied_caps() -> HalSurfaceCapabilities {
        let mut caps = noop_surface_caps();
        caps.formats.push(HalTextureFormat::Bgra8UnormSrgb);
        caps.present_modes.push(HalPresentMode::Immediate);
        caps.alpha_modes = vec![A::Premultiplied, A::Opaque, A::Unpremultiplied, A::Inherit];
        caps.usages.copy_src = true;
        caps
    }

    #[test]
    fn surface_capabilities_variable_arrays_free_with_reported_counts() {
        unsafe {
            let mut output: native::WGPUSurfaceCapabilities = std::mem::zeroed();
            fill_surface_capabilities(&mut output, varied_caps());
            assert_eq!(
                (
                    output.formatCount,
                    output.presentModeCount,
                    output.alphaModeCount
                ),
                (3, 2, 4)
            );
            assert_eq!(
                std::slice::from_raw_parts(output.formats, 3),
                [
                    native::WGPUTextureFormat_BGRA8Unorm,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureFormat_BGRA8UnormSrgb
                ]
            );
            assert_eq!(
                std::slice::from_raw_parts(output.presentModes, 2),
                [
                    native::WGPUPresentMode_Fifo,
                    native::WGPUPresentMode_Immediate
                ]
            );
            assert_eq!(
                std::slice::from_raw_parts(output.alphaModes, 4),
                [
                    native::WGPUCompositeAlphaMode_Premultiplied,
                    native::WGPUCompositeAlphaMode_Opaque,
                    native::WGPUCompositeAlphaMode_Unpremultiplied,
                    native::WGPUCompositeAlphaMode_Inherit
                ]
            );
            wgpuSurfaceCapabilitiesFreeMembers(output);
            wgpuSurfaceCapabilitiesFreeMembers(std::mem::zeroed());
        }
    }

    #[test]
    fn surface_capabilities_error_surface_zeroes_members() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let surface = wgpuInstanceCreateSurface(instance, std::ptr::null());
            let mut output: native::WGPUSurfaceCapabilities = std::mem::zeroed();
            output.formatCount = 9;
            output.presentModeCount = 8;
            output.alphaModeCount = 7;
            output.usages = native::WGPUTextureUsage_CopySrc;
            output.formats = std::ptr::dangling();
            output.presentModes = std::ptr::dangling();
            output.alphaModes = std::ptr::dangling();
            assert_eq!(
                wgpuSurfaceGetCapabilities(surface, adapter, &mut output),
                native::WGPUStatus_Error
            );
            assert_eq!(
                (
                    output.formatCount,
                    output.presentModeCount,
                    output.alphaModeCount,
                    output.usages
                ),
                (0, 0, 0, 0)
            );
            assert!(
                output.formats.is_null()
                    && output.presentModes.is_null()
                    && output.alphaModes.is_null()
            );
            wgpuSurfaceCapabilitiesFreeMembers(output);
            wgpuSurfaceRelease(surface);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn surface_configuration_validates_capability_members_and_view_siblings() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let handle = borrow_handle(device, "WGPUDevice");
            let caps = varied_caps();
            let mut config = valid_surface_config(device);
            for format in &caps.formats {
                config.format = native_surface_format(*format);
                for mode in &caps.present_modes {
                    config.presentMode = native_present_mode(*mode);
                    for alpha in &caps.alpha_modes {
                        config.alphaMode = native_alpha_mode(*alpha);
                        config.usage = native_surface_usage(caps.usages);
                        assert_eq!(surface_configuration_error(handle, &config, &caps), None);
                    }
                }
            }
            config = valid_surface_config(device);
            config.alphaMode = native::WGPUCompositeAlphaMode_Auto;
            assert_eq!(
                resolved_alpha_mode(config.alphaMode, &caps),
                native::WGPUCompositeAlphaMode_Premultiplied
            );
            assert_eq!(surface_configuration_error(handle, &config, &caps), None);
            config.format = native::WGPUTextureFormat_RGBA16Float;
            assert_eq!(
                surface_configuration_error(handle, &config, &caps),
                Some("surface configuration format is not supported")
            );
            config = valid_surface_config(device);
            for usage in [
                0,
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_StorageBinding,
            ] {
                config.usage = usage;
                assert_eq!(
                    surface_configuration_error(handle, &config, &caps),
                    Some("surface configuration usage is not supported")
                );
            }
            config = valid_surface_config(device);
            config.presentMode = native::WGPUPresentMode_Mailbox;
            assert_eq!(
                surface_configuration_error(handle, &config, &caps),
                Some("surface configuration present mode is not supported")
            );
            config = valid_surface_config(device);
            config.alphaMode = native::WGPUCompositeAlphaMode_Force32;
            assert_eq!(
                surface_configuration_error(handle, &config, &caps),
                Some("surface configuration alpha mode is not supported")
            );
            config = valid_surface_config(device);
            for (format, view) in [
                (
                    native::WGPUTextureFormat_BGRA8Unorm,
                    native::WGPUTextureFormat_BGRA8UnormSrgb,
                ),
                (
                    native::WGPUTextureFormat_BGRA8UnormSrgb,
                    native::WGPUTextureFormat_BGRA8Unorm,
                ),
            ] {
                config.format = format;
                for view in [format, view] {
                    config.viewFormatCount = 1;
                    config.viewFormats = &view;
                    assert_eq!(surface_configuration_error(handle, &config, &caps), None);
                }
            }
            let invalid = native::WGPUTextureFormat_RGBA8Unorm;
            config.viewFormats = &invalid;
            assert_eq!(
                surface_configuration_error(handle, &config, &caps),
                Some("surface configuration view format is not compatible")
            );
            config.viewFormats = std::ptr::null();
            assert_eq!(
                surface_configuration_error(handle, &config, &caps),
                Some("surface configuration viewFormats pointer is null")
            );
            release_handles(instance, adapter, device);
        }
    }
}
