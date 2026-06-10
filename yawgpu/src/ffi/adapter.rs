use super::*;

/// Releases one owned reference to an adapter handle.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle.
/// Returns WGPU adapter release.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterRelease(adapter: native::WGPUAdapter) {
    release_handle(adapter, "WGPUAdapter");
}

/// Adds one owned reference to an adapter handle.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle.
/// Returns WGPU adapter add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterAddRef(adapter: native::WGPUAdapter) {
    add_ref_handle(adapter, "WGPUAdapter");
}

/// Gets the supported limits for an adapter.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle. `limits` must
/// point to writable `WGPULimits` storage.
/// Returns WGPU adapter get limits.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterGetLimits(
    adapter: native::WGPUAdapter,
    limits: *mut native::WGPULimits,
) -> native::WGPUStatus {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    let Some(limits) = limits.as_mut() else {
        return native::WGPUStatus_Error;
    };
    *limits = map_limits_to_native(adapter.core.limits());
    native::WGPUStatus_Success
}

/// Gets the supported features for an adapter.
///
/// The returned `features` array is allocated by yawgpu and must be released
/// with `wgpuSupportedFeaturesFreeMembers`.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle. `features` must
/// point to writable `WGPUSupportedFeatures` storage.
/// Returns WGPU adapter get features.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterGetFeatures(
    adapter: native::WGPUAdapter,
    features: *mut native::WGPUSupportedFeatures,
) {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    let features = features
        .as_mut()
        .expect("WGPUSupportedFeatures must not be null");
    *features = map_features_to_native(&adapter.core.features());
}

/// Returns whether the adapter supports `feature`.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle.
/// Returns WGPU adapter has feature.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterHasFeature(
    adapter: native::WGPUAdapter,
    feature: native::WGPUFeatureName,
) -> native::WGPUBool {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    native::WGPUBool::from(adapter.core.has_feature(map_feature(feature)))
}

/// Gets identifying information for an adapter.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle. `info` must point
/// to writable `WGPUAdapterInfo` storage. String members must be released with
/// `wgpuAdapterInfoFreeMembers`.
/// Returns WGPU adapter get info.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterGetInfo(
    adapter: native::WGPUAdapter,
    info: *mut native::WGPUAdapterInfo,
) -> native::WGPUStatus {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    let Some(info) = info.as_mut() else {
        return native::WGPUStatus_Error;
    };
    *info = adapter_info_from_core(&adapter.core);
    native::WGPUStatus_Success
}

/// Gets yawgpu tiled rendering capabilities for an adapter.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle. `capabilities`
/// must point to writable `YaWGPUTiledCapabilities` storage.
/// Returns yawgpu adapter get tiled capabilities.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuAdapterGetTiledCapabilities(
    adapter: native::WGPUAdapter,
    capabilities: *mut YaWGPUTiledCapabilities,
) -> native::WGPUStatus {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    let Some(capabilities) = capabilities.as_mut() else {
        return native::WGPUStatus_Error;
    };
    let next_in_chain = capabilities.nextInChain;
    let tiled = adapter.core.tiled_capabilities();
    *capabilities = YaWGPUTiledCapabilities {
        nextInChain: next_in_chain,
        maxSubpasses: tiled.max_subpasses,
        maxSubpassColorAttachments: tiled.max_subpass_color_attachments,
        maxInputAttachments: tiled.max_input_attachments,
        estimatedTileMemoryBytes: tiled.estimated_tile_memory_bytes,
    };
    native::WGPUStatus_Success
}

/// Frees string members allocated by `wgpuAdapterGetInfo`.
///
/// # Safety
///
/// Any non-null string member must have been returned by yawgpu.
/// Returns WGPU adapter info free members.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterInfoFreeMembers(info: native::WGPUAdapterInfo) {
    free_owned_string_view(info.vendor);
    free_owned_string_view(info.architecture);
    free_owned_string_view(info.device);
    free_owned_string_view(info.description);
}

/// Requests a device from an adapter.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle. `descriptor`, when
/// non-null, must point to a valid `WGPUDeviceDescriptor`.
/// Returns WGPU adapter request device.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterRequestDevice(
    adapter: native::WGPUAdapter,
    descriptor: *const native::WGPUDeviceDescriptor,
    callback_info: native::WGPURequestDeviceCallbackInfo,
) -> native::WGPUFuture {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    let required_limits = descriptor
        .as_ref()
        .and_then(|descriptor| descriptor.requiredLimits.as_ref())
        .map(map_limits);
    let required_features = descriptor
        .as_ref()
        .map(|descriptor| required_features_from_descriptor(descriptor))
        .unwrap_or_default();
    let label = descriptor
        .as_ref()
        .and_then(|descriptor| label_from_string_view(descriptor.label))
        .unwrap_or_default();
    let queue_label = descriptor
        .as_ref()
        .and_then(|descriptor| label_from_string_view(descriptor.defaultQueue.label))
        .unwrap_or_default();
    let device_lost_callback = descriptor
        .as_ref()
        .map(|descriptor| map_device_lost_callback_info(descriptor.deviceLostCallbackInfo))
        .unwrap_or(DeviceLostCallbackInfo {
            mode: 0,
            callback: None,
            userdata1: 0,
            userdata2: 0,
        });
    let uncaptured_error_callback = descriptor
        .as_ref()
        .map(|descriptor| {
            map_uncaptured_error_callback_info(descriptor.uncapturedErrorCallbackInfo)
        })
        .unwrap_or(UncapturedErrorCallbackInfo {
            callback: None,
            userdata1: 0,
            userdata2: 0,
        });
    let result = adapter
        .core
        .create_device(
            required_limits.as_ref(),
            &required_features,
            label,
            queue_label,
        )
        .map(|device| {
            let device_impl = Arc::new(WGPUDeviceImpl {
                core: Arc::new(device),
                instance: Arc::clone(&adapter.instance),
                device_lost_callback,
                device_lost_futures: Mutex::new(Vec::new()),
                default_queue: Mutex::new(None),
                shader_module_cache: Mutex::new(HashMap::new()),
                pipeline_layout_cache: Mutex::new(HashMap::new()),
                compute_pipeline_cache: Mutex::new(HashMap::new()),
                render_pipeline_cache: Mutex::new(HashMap::new()),
            });
            if let Some(callback) = uncaptured_error_callback.callback {
                let device_handle = Arc::as_ptr(&device_impl) as usize;
                let userdata1 = uncaptured_error_callback.userdata1;
                let userdata2 = uncaptured_error_callback.userdata2;
                device_impl.set_uncaptured_error_callback(Some(move |error: core::DeviceError| {
                    let device = device_handle as native::WGPUDevice;
                    callback(
                        &device,
                        map_error_type(error.kind),
                        string_view(error.message.as_bytes()),
                        userdata1 as *mut c_void,
                        userdata2 as *mut c_void,
                    );
                }));
            }
            device_impl
        })
        .map_err(|err| err.to_string());
    let failed = result.is_err();

    let future = adapter
        .instance
        .register_callback(PendingCallback::RequestDevice {
            mode: callback_info.mode,
            callback: callback_info.callback,
            result,
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        });

    if failed && device_lost_callback.has_callback() {
        adapter
            .instance
            .register_callback(PendingCallback::DeviceLost {
                mode: device_lost_callback.mode,
                callback: device_lost_callback.callback,
                device: 0,
                reason: core::DeviceLostReason::FailedCreation,
                userdata1: device_lost_callback.userdata1,
                userdata2: device_lost_callback.userdata2,
            });
    }

    future
}

/// Frees a feature array returned by `wgpuAdapterGetFeatures` or
/// `wgpuDeviceGetFeatures`.
///
/// # Safety
///
/// `supported_features.features`, when non-null, must be a pointer previously
/// returned by yawgpu from `wgpuAdapterGetFeatures` or
/// `wgpuDeviceGetFeatures`, paired with the same `featureCount`, and must not
/// be freed more than once.
/// Returns WGPU supported features free members.
#[no_mangle]
pub unsafe extern "C" fn wgpuSupportedFeaturesFreeMembers(
    supported_features: native::WGPUSupportedFeatures,
) {
    free_supported_features(supported_features);
}
