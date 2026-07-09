use super::*;
use crate::{
    YaWGPUExternalTextureDescriptor, YAWGPU_EXTERNAL_TEXTURE_FORMAT_NV12,
    YAWGPU_EXTERNAL_TEXTURE_FORMAT_RGBA, YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_180_DEGREES,
    YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_270_DEGREES,
    YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_90_DEGREES,
};

/// Creates an external texture on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `YaWGPUExternalTextureDescriptor`; non-null texture-view
/// handles inside it must be live yawgpu texture view handles.
/// Returns yawgpu device create external texture.
#[no_mangle]
pub unsafe extern "C" fn yawgpuDeviceCreateExternalTexture(
    device: native::WGPUDevice,
    descriptor: *const YaWGPUExternalTextureDescriptor,
) -> native::WGPUExternalTexture {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("YaWGPUExternalTextureDescriptor must not be null");
    if descriptor.plane0.is_null() {
        device.dispatch_error(
            core::ErrorKind::Validation,
            "external texture plane0 is required",
        );
        return std::ptr::null();
    }
    let Some((core_descriptor, plane_handles)) = map_external_texture_descriptor(descriptor) else {
        device.dispatch_error(
            core::ErrorKind::Validation,
            "external texture descriptor contains an invalid enum value",
        );
        return std::ptr::null();
    };

    match device.core.create_external_texture(core_descriptor) {
        Ok(texture) => arc_to_handle(Arc::new(WGPUExternalTextureImpl {
            _core: Arc::new(texture),
            _planes: plane_handles,
            _device: Arc::clone(&device.core),
            _instance: Arc::clone(&device.instance),
            label: Mutex::new(None),
        })),
        Err(error) => {
            device.dispatch_error(error.kind, error.message);
            std::ptr::null()
        }
    }
}

/// Sets the debug label for an external texture.
///
/// # Safety
///
/// `external_texture` must be a non-null live yawgpu external texture handle.
/// Returns WGPU external texture set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuExternalTextureSetLabel(
    external_texture: native::WGPUExternalTexture,
    label: native::WGPUStringView,
) {
    let external_texture = borrow_handle(external_texture, "WGPUExternalTexture");
    *external_texture.label.lock().expect("label lock must not poison") =
        label_from_string_view(label);
}

/// Adds one owned reference to an external texture handle.
///
/// # Safety
///
/// `external_texture` must be a non-null live yawgpu external texture handle.
/// Returns WGPU external texture add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuExternalTextureAddRef(external_texture: native::WGPUExternalTexture) {
    add_ref_handle(external_texture, "WGPUExternalTexture");
}

/// Releases one owned reference to an external texture handle.
///
/// # Safety
///
/// `external_texture` must be a non-null live yawgpu external texture handle.
/// Returns WGPU external texture release.
#[no_mangle]
pub unsafe extern "C" fn wgpuExternalTextureRelease(external_texture: native::WGPUExternalTexture) {
    release_handle(external_texture, "WGPUExternalTexture");
}

fn map_external_texture_descriptor(
    value: &YaWGPUExternalTextureDescriptor,
) -> Option<(
    core::ExternalTextureDescriptor,
    Vec<Arc<WGPUTextureViewImpl>>,
)> {
    let format = match value.format {
        YAWGPU_EXTERNAL_TEXTURE_FORMAT_RGBA => core::ExternalTextureFormat::Rgba,
        YAWGPU_EXTERNAL_TEXTURE_FORMAT_NV12 => core::ExternalTextureFormat::Nv12,
        _ => return None,
    };
    let rotation = match value.rotation {
        crate::YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_0_DEGREES => {
            core::ExternalTextureRotation::Rotate0
        }
        YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_90_DEGREES => {
            core::ExternalTextureRotation::Rotate90
        }
        YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_180_DEGREES => {
            core::ExternalTextureRotation::Rotate180
        }
        YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_270_DEGREES => {
            core::ExternalTextureRotation::Rotate270
        }
        _ => return None,
    };

    let plane0_handle = unsafe {
        clone_handle::<WGPUTextureViewImpl>(value.plane0, "YaWGPUExternalTextureDescriptor.plane0")
    };
    let plane0 = Arc::clone(&plane0_handle._core);
    let mut plane_handles = vec![plane0_handle];
    let plane1 = if value.plane1.is_null() {
        None
    } else {
        let plane1_handle = unsafe {
            clone_handle::<WGPUTextureViewImpl>(
                value.plane1,
                "YaWGPUExternalTextureDescriptor.plane1",
            )
        };
        let plane1 = Arc::clone(&plane1_handle._core);
        plane_handles.push(plane1_handle);
        Some(plane1)
    };

    Some((
        core::ExternalTextureDescriptor {
            plane0,
            plane1,
            format,
            crop_origin: core::Origin2d {
                x: value.cropOrigin.x,
                y: value.cropOrigin.y,
            },
            crop_size: core::Extent3d {
                width: value.cropSize.width,
                height: value.cropSize.height,
                depth_or_array_layers: 1,
            },
            apparent_size: core::Extent3d {
                width: value.apparentSize.width,
                height: value.apparentSize.height,
                depth_or_array_layers: 1,
            },
            do_yuv_to_rgb_conversion_only: value.doYuvToRgbConversionOnly != 0,
            yuv_to_rgb_conversion_matrix: Some(value.yuvToRgbConversionMatrix),
            src_transfer_function_parameters: value.srcTransferFunctionParameters,
            dst_transfer_function_parameters: value.dstTransferFunctionParameters,
            gamut_conversion_matrix: value.gamutConversionMatrix,
            mirrored: value.mirrored != 0,
            rotation,
        },
        plane_handles,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        YaWGPUExtent2D, YaWGPUOrigin2D, YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_0_DEGREES,
    };
    use std::collections::{BTreeMap, HashMap};

    fn instance_impl() -> Arc<WGPUInstanceImpl> {
        Arc::new(WGPUInstanceImpl {
            core: Arc::new(core::Instance::new_noop()),
            timed_wait_any_enabled: false,
            pending_callbacks: Mutex::new(BTreeMap::new()),
        })
    }

    fn device_impl() -> Arc<WGPUDeviceImpl> {
        let instance = instance_impl();
        let adapter = instance
            .core
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter");
        let device = adapter
            .create_device(None, &[], "device", "queue")
            .expect("Noop device");
        Arc::new(WGPUDeviceImpl {
            core: Arc::new(device),
            instance,
            adapter: Arc::new(adapter),
            device_lost_callback: DeviceLostCallbackInfo {
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: None,
                userdata1: 0,
                userdata2: 0,
            },
            device_lost_futures: Mutex::new(Vec::new()),
            default_queue: Mutex::new(None),
            shader_module_cache: Mutex::new(HashMap::new()),
            pipeline_layout_cache: Mutex::new(HashMap::new()),
            compute_pipeline_cache: Mutex::new(HashMap::new()),
            render_pipeline_cache: Mutex::new(HashMap::new()),
        })
    }

    fn texture_view_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUTextureView {
        let texture = Arc::new(device.core.create_texture(core::TextureDescriptor {
            usage: core::TextureUsage::TEXTURE_BINDING,
            dimension: core::TextureDimension::D2,
            size: core::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm.into(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let (view, error) = texture.create_view(core::TextureViewDescriptor {
            format: None,
            dimension: Some(core::TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            aspect: None,
            usage: Some(core::TextureUsage::TEXTURE_BINDING),
            swizzle: None,
        });
        assert_eq!(error, None);
        arc_to_handle(Arc::new(WGPUTextureViewImpl {
            _core: Arc::new(view),
            _texture: texture,
            _device: Arc::clone(&device.core),
            _instance: Arc::clone(&device.instance),
            label: Mutex::new(None),
        }))
    }

    fn descriptor(plane0: native::WGPUTextureView) -> YaWGPUExternalTextureDescriptor {
        YaWGPUExternalTextureDescriptor {
            plane0,
            plane1: std::ptr::null(),
            format: YAWGPU_EXTERNAL_TEXTURE_FORMAT_RGBA,
            cropOrigin: YaWGPUOrigin2D { x: 0, y: 0 },
            cropSize: YaWGPUExtent2D {
                width: 4,
                height: 4,
            },
            apparentSize: YaWGPUExtent2D {
                width: 4,
                height: 4,
            },
            doYuvToRgbConversionOnly: 1,
            yuvToRgbConversionMatrix: [
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            srcTransferFunctionParameters: [0.0; 7],
            dstTransferFunctionParameters: [0.0; 7],
            gamutConversionMatrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            mirrored: 0,
            rotation: YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_0_DEGREES,
        }
    }

    #[test]
    fn yawgpu_device_create_external_texture_rgba_noop_returns_non_null() {
        let device = device_impl();
        let device_handle = arc_to_handle(Arc::clone(&device));
        let plane0 = texture_view_handle(&device);
        let desc = descriptor(plane0);
        unsafe {
            wgpuDevicePushErrorScope(device_handle, native::WGPUErrorFilter_Validation);
            let external = yawgpuDeviceCreateExternalTexture(device_handle, &desc);
            assert!(!external.is_null());
            assert_eq!(device.core.pop_error_scope().expect("scope"), None);
            wgpuExternalTextureAddRef(external);
            wgpuExternalTextureRelease(external);
            wgpuExternalTextureRelease(external);
            wgpuTextureViewRelease(plane0);
            wgpuDeviceRelease(device_handle);
        }
    }

    #[test]
    fn yawgpu_device_create_external_texture_nv12_one_plane_reports_device_error() {
        let device = device_impl();
        let device_handle = arc_to_handle(Arc::clone(&device));
        let plane0 = texture_view_handle(&device);
        let mut desc = descriptor(plane0);
        desc.format = YAWGPU_EXTERNAL_TEXTURE_FORMAT_NV12;
        unsafe {
            wgpuDevicePushErrorScope(device_handle, native::WGPUErrorFilter_Validation);
            let external = yawgpuDeviceCreateExternalTexture(device_handle, &desc);
            assert!(external.is_null());
            let error = device
                .core
                .pop_error_scope()
                .expect("scope")
                .expect("error");
            assert_eq!(error.kind, core::ErrorKind::Validation);
            wgpuTextureViewRelease(plane0);
            wgpuDeviceRelease(device_handle);
        }
    }

    #[test]
    fn yawgpu_device_create_external_texture_null_plane0_reports_device_error() {
        let device = device_impl();
        let device_handle = arc_to_handle(Arc::clone(&device));
        let desc = descriptor(std::ptr::null_mut());
        unsafe {
            wgpuDevicePushErrorScope(device_handle, native::WGPUErrorFilter_Validation);
            let external = yawgpuDeviceCreateExternalTexture(device_handle, &desc);
            assert!(external.is_null());
            let error = device
                .core
                .pop_error_scope()
                .expect("scope")
                .expect("error");
            assert_eq!(error.kind, core::ErrorKind::Validation);
            assert_eq!(error.message, "external texture plane0 is required");
            wgpuDeviceRelease(device_handle);
        }
    }

    #[test]
    fn wgpu_external_texture_set_label_stores_string_view_cases() {
        let device = device_impl();
        let device_handle = arc_to_handle(Arc::clone(&device));
        let plane0 = texture_view_handle(&device);
        let desc = descriptor(plane0);
        unsafe {
            let external = yawgpuDeviceCreateExternalTexture(device_handle, &desc);
            assert!(!external.is_null());
            let handle = borrow_handle(external, "WGPUExternalTexture");

            wgpuExternalTextureSetLabel(external, crate::conv::string_view(b"a\0b"));
            assert_eq!(handle.label(), Some("a\0b".to_owned()));

            let strlen_bytes = b"strlen\0";
            wgpuExternalTextureSetLabel(
                external,
                native::WGPUStringView {
                    data: strlen_bytes.as_ptr().cast(),
                    length: crate::conv::WGPU_STRLEN,
                },
            );
            assert_eq!(handle.label(), Some("strlen".to_owned()));

            wgpuExternalTextureSetLabel(
                external,
                native::WGPUStringView {
                    data: std::ptr::null(),
                    length: crate::conv::WGPU_STRLEN,
                },
            );
            assert_eq!(handle.label(), None);

            wgpuExternalTextureSetLabel(
                external,
                native::WGPUStringView {
                    data: std::ptr::null(),
                    length: 0,
                },
            );
            assert_eq!(handle.label(), Some(String::new()));

            wgpuExternalTextureSetLabel(
                external,
                native::WGPUStringView {
                    data: b"ignored".as_ptr().cast(),
                    length: 0,
                },
            );
            assert_eq!(handle.label(), Some(String::new()));

            wgpuExternalTextureRelease(external);
            wgpuTextureViewRelease(plane0);
            wgpuDeviceRelease(device_handle);
        }
    }
}
