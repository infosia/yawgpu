use super::*;

/// Creates a new WebGPU instance.
///
/// # Safety
///
/// `descriptor`, when non-null, must point to a valid `WGPUInstanceDescriptor`.
/// Returns WGPU create instance.
#[no_mangle]
pub unsafe extern "C" fn wgpuCreateInstance(
    descriptor: *const native::WGPUInstanceDescriptor,
) -> native::WGPUInstance {
    let timed_wait_any_enabled = instance_has_timed_wait_any(descriptor);
    let instance = match instance_backend_selection(descriptor) {
        InstanceBackendSelection::Noop => WGPUInstanceImpl::new_noop(timed_wait_any_enabled),
        InstanceBackendSelection::Metal => {
            #[cfg(feature = "metal")]
            {
                match yawgpu_hal::metal::MetalInstance::new() {
                    Ok(instance) => {
                        let hal_instance = yawgpu_hal::HalInstance::Metal(instance);
                        if hal_instance.enumerate_adapters().is_empty() {
                            WGPUInstanceImpl::new_noop(timed_wait_any_enabled)
                        } else {
                            WGPUInstanceImpl::from_core(
                                core::Instance::from_hal(hal_instance),
                                timed_wait_any_enabled,
                            )
                        }
                    }
                    Err(_) => WGPUInstanceImpl::new_noop(timed_wait_any_enabled),
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                WGPUInstanceImpl::new_noop(timed_wait_any_enabled)
            }
        }
        InstanceBackendSelection::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                match yawgpu_hal::vulkan::VulkanInstance::new() {
                    Ok(instance) => {
                        let hal_instance = yawgpu_hal::HalInstance::Vulkan(instance);
                        if hal_instance.enumerate_adapters().is_empty() {
                            WGPUInstanceImpl::new_noop(timed_wait_any_enabled)
                        } else {
                            WGPUInstanceImpl::from_core(
                                core::Instance::from_hal(hal_instance),
                                timed_wait_any_enabled,
                            )
                        }
                    }
                    Err(_) => WGPUInstanceImpl::new_noop(timed_wait_any_enabled),
                }
            }
            #[cfg(not(feature = "vulkan"))]
            {
                WGPUInstanceImpl::new_noop(timed_wait_any_enabled)
            }
        }
    };
    arc_to_handle(instance)
}

/// Releases one owned reference to an instance handle.
///
/// # Safety
///
/// `instance` must be a non-null handle previously returned by yawgpu and not
/// already fully released.
/// Returns WGPU instance release.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceRelease(instance: native::WGPUInstance) {
    release_handle(instance, "WGPUInstance");
}

/// Adds one owned reference to an instance handle.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle.
/// Returns WGPU instance add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceAddRef(instance: native::WGPUInstance) {
    add_ref_handle(instance, "WGPUInstance");
}

/// Creates a synthetic Noop surface from a recognized surface-source chain.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle. `descriptor`,
/// when non-null, must point to a valid `WGPUSurfaceDescriptor`.
/// Returns WGPU instance create surface.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceCreateSurface(
    instance: native::WGPUInstance,
    descriptor: *const native::WGPUSurfaceDescriptor,
) -> native::WGPUSurface {
    let instance = clone_handle(instance, "WGPUInstance");
    let (label, is_error, hal) = if let Some(descriptor) = descriptor.as_ref() {
        let layer = find_metal_layer_source(descriptor.nextInChain);
        let mut real_surface_creation_failed =
            layer.is_some() && is_real_hal_instance(instance.core.hal());
        let mut hal = layer.and_then(|layer| {
            unsafe { instance.core.create_surface_from_metal_layer(layer) }
                .ok()
                .and_then(real_hal_surface)
        });
        if layer.is_none() {
            let hwnd_source = find_windows_hwnd_source(descriptor.nextInChain);
            real_surface_creation_failed =
                hwnd_source.is_some() && is_real_hal_instance(instance.core.hal());
            hal = hwnd_source.and_then(|(hinstance, hwnd)| {
                unsafe {
                    instance
                        .core
                        .create_surface_from_windows_hwnd(hinstance, hwnd)
                }
                .ok()
                .and_then(real_hal_surface)
            });
        }
        let surface_source_is_unsupported = !has_supported_surface_source(descriptor.nextInChain);
        real_surface_creation_failed = real_surface_creation_failed && hal.is_none();
        (
            label_from_string_view(descriptor.label).unwrap_or_default(),
            surface_source_is_unsupported || real_surface_creation_failed,
            hal,
        )
    } else {
        (String::new(), true, None)
    };
    arc_to_handle(Arc::new(WGPUSurfaceImpl {
        label: Mutex::new(label),
        configured: Mutex::new(None),
        hal: Mutex::new(hal),
        is_error,
        _instance: instance,
    }))
}

/// Requests a Noop adapter from an instance.
///
/// # Safety
///
/// `instance_handle` must be a non-null live yawgpu instance handle. `options`,
/// when non-null, must point to a valid `WGPURequestAdapterOptions`.
/// Returns WGPU instance request adapter.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceRequestAdapter(
    instance_handle: native::WGPUInstance,
    options: *const native::WGPURequestAdapterOptions,
    callback_info: native::WGPURequestAdapterCallbackInfo,
) -> native::WGPUFuture {
    let instance = borrow_handle(instance_handle, "WGPUInstance");
    let options_ref = options.as_ref();
    let feature_level = options_ref
        .map(|options| map_feature_level(options.featureLevel))
        .unwrap_or(core::FeatureLevel::Core);
    let backend_type = options_ref
        .map(|options| options.backendType)
        .unwrap_or(native::WGPUBackendType_Undefined);
    let Some(adapter) = select_request_adapter(instance, backend_type, feature_level) else {
        return instance.register_callback(PendingCallback::RequestAdapterError {
            mode: callback_info.mode,
            status: native::WGPURequestAdapterStatus_Unavailable,
            callback: callback_info.callback,
            message: "requested adapter backend is unavailable".to_owned(),
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        });
    };
    let adapter = Arc::new(WGPUAdapterImpl {
        core: Arc::new(adapter),
        instance: clone_handle(instance_handle, "WGPUInstance"),
    });

    instance.register_callback(PendingCallback::RequestAdapter {
        mode: callback_info.mode,
        callback: callback_info.callback,
        adapter,
        userdata1: callback_info.userdata1 as usize,
        userdata2: callback_info.userdata2 as usize,
    })
}

fn select_request_adapter(
    instance: &WGPUInstanceImpl,
    backend_type: native::WGPUBackendType,
    feature_level: core::FeatureLevel,
) -> Option<core::Adapter> {
    #[cfg(feature = "gles")]
    if backend_type == native::WGPUBackendType_OpenGLES {
        return instance.gles_core.as_ref().and_then(|core_instance| {
            core_instance
                .enumerate_adapters_with_feature_level(feature_level)
                .into_iter()
                .next()
        });
    }
    #[cfg(not(feature = "gles"))]
    if backend_type == native::WGPUBackendType_OpenGLES {
        return None;
    }

    instance
        .core
        .enumerate_adapters_with_feature_level(feature_level)
        .into_iter()
        .next()
}

/// Processes callbacks whose mode allows process-events delivery.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle.
/// Returns WGPU instance process events.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceProcessEvents(instance: native::WGPUInstance) {
    let instance = borrow_handle(instance, "WGPUInstance");
    instance.process_callbacks();
}

/// Waits for any listed future and fires callbacks for completed futures.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle. If
/// `future_count` is non-zero, `futures` must point to `future_count` valid
/// `WGPUFutureWaitInfo` entries.
/// Returns WGPU instance wait any.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceWaitAny(
    instance: native::WGPUInstance,
    future_count: usize,
    futures: *mut native::WGPUFutureWaitInfo,
    timeout_ns: u64,
) -> native::WGPUWaitStatus {
    let instance = borrow_handle(instance, "WGPUInstance");
    if future_count > 0 && futures.is_null() {
        return native::WGPUWaitStatus_Error;
    }
    if future_count == 0 {
        return native::WGPUWaitStatus_TimedOut;
    }
    if timeout_ns > 0 && !instance.timed_wait_any_enabled {
        return native::WGPUWaitStatus_Error;
    }

    let wait_infos = std::slice::from_raw_parts_mut(futures, future_count);
    let future_ids = wait_infos
        .iter()
        .map(|info| core::FutureId::from_raw(info.future.id))
        .collect::<Vec<_>>();
    let result = instance.wait_any(&future_ids);

    for info in wait_infos {
        let id = core::FutureId::from_raw(info.future.id);
        info.completed = u32::from(result.completed.contains(&id));
    }

    match result.status {
        core::WaitAnyStatus::Success => native::WGPUWaitStatus_Success,
        core::WaitAnyStatus::TimedOut => native::WGPUWaitStatus_TimedOut,
        core::WaitAnyStatus::Error => native::WGPUWaitStatus_Error,
        _ => native::WGPUWaitStatus_Error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_request_adapter_undefined_returns_primary_noop_adapter() {
        let instance = WGPUInstanceImpl::new_noop(false);

        let adapter = select_request_adapter(
            &instance,
            native::WGPUBackendType_Undefined,
            core::FeatureLevel::Core,
        );

        assert!(adapter.is_some());
    }

    #[test]
    #[cfg(feature = "gles")]
    fn select_request_adapter_opengles_with_no_side_instance_returns_none() {
        let instance = WGPUInstanceImpl {
            core: Arc::new(core::Instance::new_noop()),
            gles_core: None,
            timed_wait_any_enabled: false,
            pending_callbacks: Mutex::new(BTreeMap::new()),
        };

        let adapter = select_request_adapter(
            &instance,
            native::WGPUBackendType_OpenGLES,
            core::FeatureLevel::Core,
        );

        assert!(adapter.is_none());
    }

    #[test]
    #[cfg(not(feature = "gles"))]
    fn select_request_adapter_opengles_without_gles_feature_returns_none() {
        let instance = WGPUInstanceImpl::new_noop(false);

        let adapter = select_request_adapter(
            &instance,
            native::WGPUBackendType_OpenGLES,
            core::FeatureLevel::Core,
        );

        assert!(adapter.is_none());
    }
}
