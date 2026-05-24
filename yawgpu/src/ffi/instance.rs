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
        InstanceBackendSelection::Gles => {
            #[cfg(feature = "gles")]
            {
                match yawgpu_hal::gles::GlesInstance::new() {
                    Ok(instance) => {
                        let hal_instance = yawgpu_hal::HalInstance::Gles(instance);
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
            #[cfg(not(feature = "gles"))]
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
        let mut found_hwnd_source = false;
        if layer.is_none() {
            let hwnd_source = find_windows_hwnd_source(descriptor.nextInChain);
            found_hwnd_source = hwnd_source.is_some();
            real_surface_creation_failed =
                found_hwnd_source && is_real_hal_instance(instance.core.hal());
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
        if layer.is_none() && !found_hwnd_source && hal.is_none() {
            let android_source = find_android_native_window_source(descriptor.nextInChain);
            real_surface_creation_failed =
                android_source.is_some() && is_real_hal_instance(instance.core.hal());
            hal = android_source.and_then(|window| {
                unsafe {
                    instance
                        .core
                        .create_surface_from_android_native_window(window)
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
    let feature_level = options
        .as_ref()
        .map(|options| map_feature_level(options.featureLevel))
        .unwrap_or(core::FeatureLevel::Core);
    let adapter = instance
        .core
        .enumerate_adapters_with_feature_level(feature_level)
        .into_iter()
        .next()
        .expect("Noop instance must expose an adapter");
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
