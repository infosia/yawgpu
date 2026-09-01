use super::*;

const SUPPORTED_INSTANCE_FEATURES: &[native::WGPUInstanceFeatureName] =
    &[native::WGPUInstanceFeatureName_TimedWaitAny];

// Advertised for native parity; FutureRegistry::wait_any does not enforce it today.
pub(crate) const TIMED_WAIT_ANY_MAX_COUNT: usize = 64;

#[cfg(any(feature = "metal", feature = "vulkan", feature = "gles"))]
fn finish_real_instance(
    hal_instance: HalInstance,
    prefix: &str,
    backend: &str,
    timed_wait_any_enabled: bool,
) -> Option<Arc<WGPUInstanceImpl>> {
    if hal_instance.enumerate_adapters().is_empty() {
        eprintln!(
            "{prefix}: {backend} requested but enumerate_adapters returned no adapters; returning NULL instance"
        );
        return None;
    }
    Some(WGPUInstanceImpl::from_core(
        core::Instance::from_hal(hal_instance),
        timed_wait_any_enabled,
    ))
}

/// Creates a new WebGPU instance.
///
/// Resolves the HAL backend per the `YaWGPUInstanceBackendSelect` chain entry,
/// when present, following rules IB1-IB4 from
/// `specs/blocks/60-real-backends.md`:
///
/// - **IB1** No chain entry: returns a Noop instance.
/// - **IB2** Chain `backend = YAWGPU_INSTANCE_BACKEND_NOOP`: returns a Noop
///   instance.
/// - **IB3** Chain `backend = YAWGPU_INSTANCE_BACKEND_{METAL, VULKAN, GLES}`:
///   strict. Returns NULL when the matching cargo feature was not compiled in,
///   when the backend's `HalInstance::new` returns `Err`, or when
///   `enumerate_adapters()` returns empty. A best-effort diagnostic is written
///   to `stderr`.
/// - **IB4** Chain `backend` value outside the four constants above: returns
///   a Noop instance (lenient — newer-header constants stay forward-compatible
///   against older `yawgpu` builds).
///
/// # Safety
///
/// `descriptor`, when non-null, must point to a valid `WGPUInstanceDescriptor`.
/// Returns the new instance handle, or NULL when an IB3 strict failure occurs.
#[no_mangle]
pub unsafe extern "C" fn wgpuCreateInstance(
    descriptor: *const native::WGPUInstanceDescriptor,
) -> native::WGPUInstance {
    let timed_wait_any_enabled = instance_has_timed_wait_any(descriptor);
    let selection = instance_backend_selection(descriptor);
    let instance = match selection {
        // IB1 (no chain) and IB2 (chain == NOOP) / IB4 (chain == unknown).
        None | Some(InstanceBackendSelection::Noop) => {
            WGPUInstanceImpl::new_noop(timed_wait_any_enabled)
        }
        Some(InstanceBackendSelection::Metal) => {
            #[cfg(feature = "metal")]
            {
                match yawgpu_hal::metal::MetalInstance::new() {
                    Ok(instance) => {
                        let Some(instance) = finish_real_instance(
                            yawgpu_hal::HalInstance::Metal(instance),
                            "yawgpu-metal",
                            "YAWGPU_INSTANCE_BACKEND_METAL",
                            timed_wait_any_enabled,
                        ) else {
                            return std::ptr::null();
                        };
                        instance
                    }
                    Err(err) => {
                        eprintln!(
                            "yawgpu-metal: YAWGPU_INSTANCE_BACKEND_METAL requested but HalInstance::new failed ({err:?}); returning NULL instance"
                        );
                        return std::ptr::null();
                    }
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = timed_wait_any_enabled;
                eprintln!(
                    "yawgpu-metal: YAWGPU_INSTANCE_BACKEND_METAL requested but yawgpu was built without feature=metal; returning NULL instance"
                );
                return std::ptr::null();
            }
        }
        Some(InstanceBackendSelection::Vulkan) => {
            #[cfg(feature = "vulkan")]
            {
                match yawgpu_hal::vulkan::VulkanInstance::new() {
                    Ok(instance) => {
                        let Some(instance) = finish_real_instance(
                            yawgpu_hal::HalInstance::Vulkan(instance),
                            "yawgpu-vulkan",
                            "YAWGPU_INSTANCE_BACKEND_VULKAN",
                            timed_wait_any_enabled,
                        ) else {
                            return std::ptr::null();
                        };
                        instance
                    }
                    Err(err) => {
                        eprintln!(
                            "yawgpu-vulkan: YAWGPU_INSTANCE_BACKEND_VULKAN requested but HalInstance::new failed ({err:?}); returning NULL instance"
                        );
                        return std::ptr::null();
                    }
                }
            }
            #[cfg(not(feature = "vulkan"))]
            {
                let _ = timed_wait_any_enabled;
                eprintln!(
                    "yawgpu-vulkan: YAWGPU_INSTANCE_BACKEND_VULKAN requested but yawgpu was built without feature=vulkan; returning NULL instance"
                );
                return std::ptr::null();
            }
        }
        Some(InstanceBackendSelection::Gles) => {
            #[cfg(feature = "gles")]
            {
                let context_backend = gles_context_backend_choice(descriptor);
                match yawgpu_hal::gles::GlesInstance::new_with_choice(context_backend) {
                    Ok(instance) => {
                        let Some(instance) = finish_real_instance(
                            yawgpu_hal::HalInstance::Gles(instance),
                            "yawgpu-gles",
                            "YAWGPU_INSTANCE_BACKEND_GLES",
                            timed_wait_any_enabled,
                        ) else {
                            return std::ptr::null();
                        };
                        instance
                    }
                    Err(err) => {
                        eprintln!(
                            "yawgpu-gles: YAWGPU_INSTANCE_BACKEND_GLES requested but HalInstance::new failed ({err:?}); returning NULL instance"
                        );
                        return std::ptr::null();
                    }
                }
            }
            #[cfg(not(feature = "gles"))]
            {
                let _ = timed_wait_any_enabled;
                eprintln!(
                    "yawgpu-gles: YAWGPU_INSTANCE_BACKEND_GLES requested but yawgpu was built without feature=gles; returning NULL instance"
                );
                return std::ptr::null();
            }
        }
    };
    arc_to_handle(instance)
}

/// Gets the instance features supported by this library.
///
/// # Safety
///
/// `features` must point to writable `WGPUSupportedInstanceFeatures` storage.
/// Returns WGPU get instance features.
#[no_mangle]
pub unsafe extern "C" fn wgpuGetInstanceFeatures(
    features: *mut native::WGPUSupportedInstanceFeatures,
) {
    let features = features
        .as_mut()
        .expect("WGPUSupportedInstanceFeatures must not be null");
    let boxed = SUPPORTED_INSTANCE_FEATURES.to_vec().into_boxed_slice();
    features.featureCount = boxed.len();
    features.features = Box::into_raw(boxed).cast::<native::WGPUInstanceFeatureName>();
}

/// Gets the instance limits supported by this library.
///
/// # Safety
///
/// `limits` must be null or point to writable `WGPUInstanceLimits` storage.
/// Returns WGPU get instance limits.
#[no_mangle]
pub unsafe extern "C" fn wgpuGetInstanceLimits(
    limits: *mut native::WGPUInstanceLimits,
) -> native::WGPUStatus {
    let Some(limits) = limits.as_mut() else {
        return native::WGPUStatus_Error;
    };
    limits.timedWaitAnyMaxCount = TIMED_WAIT_ANY_MAX_COUNT;
    native::WGPUStatus_Success
}

/// Reports whether an instance feature is supported by this library.
///
/// # Safety
///
/// This function does not dereference pointers.
/// Returns WGPU has instance feature.
#[no_mangle]
pub unsafe extern "C" fn wgpuHasInstanceFeature(
    feature: native::WGPUInstanceFeatureName,
) -> native::WGPUBool {
    u32::from(SUPPORTED_INSTANCE_FEATURES.contains(&feature))
}

/// Frees feature members returned by `wgpuGetInstanceFeatures`.
///
/// # Safety
///
/// `supported_features.features`, when non-null, must be a pointer previously
/// returned by yawgpu from `wgpuGetInstanceFeatures`, paired with the same
/// `featureCount`, and must not be freed more than once.
/// Returns WGPU supported instance features free members.
#[no_mangle]
pub unsafe extern "C" fn wgpuSupportedInstanceFeaturesFreeMembers(
    supported_features: native::WGPUSupportedInstanceFeatures,
) {
    if supported_features.features.is_null() || supported_features.featureCount == 0 {
        return;
    }
    let slice = std::ptr::slice_from_raw_parts_mut(
        supported_features.features.cast_mut(),
        supported_features.featureCount,
    );
    drop(Box::from_raw(slice));
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

/// Gets the WGSL language features supported by this instance.
///
/// The returned `features` array is allocated by yawgpu and must be released
/// with `wgpuSupportedWGSLLanguageFeaturesFreeMembers`.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle. `features` must
/// point to writable `WGPUSupportedWGSLLanguageFeatures` storage.
/// Returns WGPU instance get WGSL language features.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceGetWGSLLanguageFeatures(
    instance: native::WGPUInstance,
    features: *mut native::WGPUSupportedWGSLLanguageFeatures,
) {
    let _instance = borrow_handle(instance, "WGPUInstance");
    let features = features
        .as_mut()
        .expect("WGPUSupportedWGSLLanguageFeatures must not be null");
    let feature_values = core::SUPPORTED_WGSL_LANGUAGE_FEATURES
        .iter()
        .copied()
        .map(|feature| feature as native::WGPUWGSLLanguageFeatureName)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let feature_count = feature_values.len();
    let feature_values = Box::into_raw(feature_values);
    *features = native::WGPUSupportedWGSLLanguageFeatures {
        featureCount: feature_count,
        features: feature_values.cast(),
    };
}

/// Returns whether this instance supports a WGSL language feature.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle.
/// Returns WGPU instance has WGSL language feature.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceHasWGSLLanguageFeature(
    instance: native::WGPUInstance,
    feature: native::WGPUWGSLLanguageFeatureName,
) -> native::WGPUBool {
    let _instance = borrow_handle(instance, "WGPUInstance");
    // Compare via the native alias cast (as in `wgpuInstanceGetWGSLLanguageFeatures`
    // above) rather than `contains(&feature)`: the core list is `&[u32]` but the
    // `WGPUWGSLLanguageFeatureName` alias is `i32` on MSVC and `u32` on clang, so a
    // direct `&u32` vs `&i32` comparison fails to compile on Windows.
    let supported = core::SUPPORTED_WGSL_LANGUAGE_FEATURES
        .iter()
        .any(|&f| f as native::WGPUWGSLLanguageFeatureName == feature);
    native::WGPUBool::from(supported)
}

/// Frees a WGSL language feature array returned by `wgpuInstanceGetWGSLLanguageFeatures`.
///
/// # Safety
///
/// `supported_features.features`, when non-null, must be a pointer previously
/// returned by yawgpu from `wgpuInstanceGetWGSLLanguageFeatures`, paired with
/// the same `featureCount`, and must not be freed more than once.
/// Returns WGPU supported WGSL language features free members.
#[no_mangle]
pub unsafe extern "C" fn wgpuSupportedWGSLLanguageFeaturesFreeMembers(
    supported_features: native::WGPUSupportedWGSLLanguageFeatures,
) {
    if supported_features.features.is_null() {
        return;
    }
    let slice = std::ptr::slice_from_raw_parts_mut(
        supported_features.features.cast_mut(),
        supported_features.featureCount,
    );
    drop(Box::from_raw(slice));
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
            label_from_string_view(descriptor.label),
            surface_source_is_unsupported || real_surface_creation_failed,
            hal,
        )
    } else {
        (None, true, None)
    };
    arc_to_handle(Arc::new(WGPUSurfaceImpl {
        label: Mutex::new(label),
        configured: Mutex::new(None),
        hal: Mutex::new(hal),
        is_error,
        _instance: instance,
    }))
}

/// Requests an adapter from an instance.
///
/// When the instance exposes no adapters (a legitimate spec-level outcome on
/// real backends — e.g. GLES display loss), or when no adapter matches the
/// requested backend type, the callback is delivered with
/// `WGPURequestAdapterStatus_Unavailable` and a null adapter; this never
/// panics.
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
    let requested_backend_type = options
        .as_ref()
        .map(|options| options.backendType)
        .unwrap_or(native::WGPUBackendType_Undefined);
    let result = select_request_adapter(
        instance
            .core
            .enumerate_adapters_with_feature_level(feature_level),
        requested_backend_type,
        |adapter| backend_type_from_hal(adapter.backend()),
    )
    .map(|adapter| {
        Arc::new(WGPUAdapterImpl {
            core: Arc::new(adapter),
            instance: clone_handle(instance_handle, "WGPUInstance"),
        })
    });

    instance.register_callback(PendingCallback::RequestAdapter {
        mode: callback_info.mode,
        callback: callback_info.callback,
        result,
        userdata1: callback_info.userdata1 as usize,
        userdata2: callback_info.userdata2 as usize,
    })
}

/// Selects the adapter delivered by `wgpuInstanceRequestAdapter` from an
/// instance's enumeration.
///
/// Returns the first adapter whose mapped backend matches
/// `requested_backend_type`. An undefined request preserves the original
/// first-adapter behavior. An empty enumeration or a request with no matching
/// adapter returns the failure message delivered with
/// `WGPURequestAdapterStatus_Unavailable` and a null adapter.
fn select_request_adapter(
    adapters: Vec<core::Adapter>,
    requested_backend_type: native::WGPUBackendType,
    mut adapter_backend_type: impl FnMut(&core::Adapter) -> native::WGPUBackendType,
) -> Result<core::Adapter, String> {
    let mut adapters = adapters.into_iter();
    let first_adapter = adapters
        .next()
        .ok_or_else(|| String::from("no adapters available"))?;
    let instance_backend = first_adapter.backend();

    if requested_backend_type == native::WGPUBackendType_Undefined
        || adapter_backend_type(&first_adapter) == requested_backend_type
    {
        return Ok(first_adapter);
    }

    adapters
        .find(|adapter| adapter_backend_type(adapter) == requested_backend_type)
        .ok_or_else(|| {
            format!(
                "no adapter with backendType {} (instance backend: {instance_backend:?})",
                backend_type_name(requested_backend_type)
            )
        })
}

/// Returns the unprefixed `webgpu.h` constant name for a backend type.
fn backend_type_name(backend_type: native::WGPUBackendType) -> String {
    let name = match backend_type {
        native::WGPUBackendType_Undefined => "Undefined",
        native::WGPUBackendType_Null => "Null",
        native::WGPUBackendType_WebGPU => "WebGPU",
        native::WGPUBackendType_D3D11 => "D3D11",
        native::WGPUBackendType_D3D12 => "D3D12",
        native::WGPUBackendType_Metal => "Metal",
        native::WGPUBackendType_Vulkan => "Vulkan",
        native::WGPUBackendType_OpenGL => "OpenGL",
        native::WGPUBackendType_OpenGLES => "OpenGLES",
        native::WGPUBackendType_Force32 => "Force32",
        value => return format!("Unknown({value:#x})"),
    };
    String::from(name)
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
    let timeout = std::time::Duration::from_nanos(timeout_ns);
    let started = std::time::Instant::now();
    let mut callbacks_to_fire = Vec::new();
    let result = loop {
        let result = instance.wait_any(&future_ids, &mut callbacks_to_fire);
        if result.status != core::WaitAnyStatus::TimedOut
            || timeout_ns == 0
            || started.elapsed() >= timeout
        {
            break result;
        }
        std::thread::yield_now();
    };

    let completed = result.completed.iter().copied().collect::<HashSet<_>>();
    for info in wait_infos {
        let id = core::FutureId::from_raw(info.future.id);
        info.completed = u32::from(completed.contains(&id));
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

    #[derive(Default)]
    struct RequestAdapterState {
        fired: u32,
        status: native::WGPURequestAdapterStatus,
        adapter: native::WGPUAdapter,
        message: String,
    }

    unsafe extern "C" fn request_adapter_callback(
        status: native::WGPURequestAdapterStatus,
        adapter: native::WGPUAdapter,
        message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        let state = &mut *(userdata1 as *mut RequestAdapterState);
        state.fired += 1;
        state.status = status;
        state.adapter = adapter;
        state.message = string_view_to_str(message).unwrap_or_default().to_owned();
    }

    fn request_adapter_options(
        backend_type: native::WGPUBackendType,
    ) -> native::WGPURequestAdapterOptions {
        native::WGPURequestAdapterOptions {
            nextInChain: std::ptr::null_mut(),
            featureLevel: native::WGPUFeatureLevel_Undefined,
            powerPreference: native::WGPUPowerPreference_Undefined,
            forceFallbackAdapter: 0,
            backendType: backend_type,
            compatibleSurface: std::ptr::null(),
        }
    }

    unsafe fn request_adapter(
        instance: native::WGPUInstance,
        options: *const native::WGPURequestAdapterOptions,
    ) -> RequestAdapterState {
        let mut state = RequestAdapterState::default();
        let callback_info = native::WGPURequestAdapterCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(request_adapter_callback),
            userdata1: (&mut state as *mut RequestAdapterState).cast(),
            userdata2: std::ptr::null_mut(),
        };

        let future = wgpuInstanceRequestAdapter(instance, options, callback_info);
        assert_ne!(future.id, 0);
        wgpuInstanceProcessEvents(instance);
        assert_eq!(state.fired, 1);
        state
    }

    fn noop_adapters(count: usize) -> Vec<core::Adapter> {
        (0..count)
            .flat_map(|_| core::Instance::new_noop().enumerate_adapters())
            .collect()
    }

    #[test]
    fn wgpu_instance_get_wgsl_language_features_reports_canonical_set() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());
            let mut features = native::WGPUSupportedWGSLLanguageFeatures {
                featureCount: 0,
                features: std::ptr::null(),
            };

            wgpuInstanceGetWGSLLanguageFeatures(instance, &mut features);

            assert_eq!(
                features.featureCount,
                core::SUPPORTED_WGSL_LANGUAGE_FEATURES.len()
            );
            let values = std::slice::from_raw_parts(features.features, features.featureCount);
            // The returned slice is `WGPUWGSLLanguageFeatureName` (i32 on MSVC, u32
            // on clang); cast the core `&[u32]` list to the alias before comparing.
            let expected: Vec<native::WGPUWGSLLanguageFeatureName> =
                core::SUPPORTED_WGSL_LANGUAGE_FEATURES
                    .iter()
                    .map(|&f| f as native::WGPUWGSLLanguageFeatureName)
                    .collect();
            assert_eq!(values, expected.as_slice());
            assert_eq!(
                wgpuInstanceHasWGSLLanguageFeature(
                    instance,
                    native::WGPUWGSLLanguageFeatureName_Packed4x8IntegerDotProduct,
                ),
                native::WGPUBool::from(true)
            );
            assert_eq!(
                wgpuInstanceHasWGSLLanguageFeature(
                    instance,
                    native::WGPUWGSLLanguageFeatureName_SubgroupId,
                ),
                native::WGPUBool::from(true)
            );
            assert_eq!(
                wgpuInstanceHasWGSLLanguageFeature(
                    instance,
                    native::WGPUWGSLLanguageFeatureName_SubgroupUniformity,
                ),
                native::WGPUBool::from(true)
            );
            assert_eq!(
                wgpuInstanceHasWGSLLanguageFeature(
                    instance,
                    native::WGPUWGSLLanguageFeatureName_ImmediateAddressSpace,
                ),
                native::WGPUBool::from(true)
            );
            assert!(values.contains(&native::WGPUWGSLLanguageFeatureName_ImmediateAddressSpace));

            wgpuSupportedWGSLLanguageFeaturesFreeMembers(features);
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn select_request_adapter_returns_first_matching_adapter() {
        let adapters = noop_adapters(2);

        let result = select_request_adapter(adapters, native::WGPUBackendType_Null, |adapter| {
            backend_type_from_hal(adapter.backend())
        });

        assert_eq!(
            result.map(|adapter| adapter.backend()),
            Ok(yawgpu_hal::HalBackend::Noop)
        );
    }

    #[test]
    fn select_request_adapter_returns_second_matching_adapter() {
        let adapters = noop_adapters(2);
        let mut mapped_count = 0;

        let result = select_request_adapter(adapters, native::WGPUBackendType_Null, |adapter| {
            mapped_count += 1;
            if mapped_count == 1 {
                native::WGPUBackendType_Vulkan
            } else {
                backend_type_from_hal(adapter.backend())
            }
        });

        assert_eq!(mapped_count, 2);
        assert_eq!(
            result.map(|adapter| adapter.backend()),
            Ok(yawgpu_hal::HalBackend::Noop)
        );
    }

    #[test]
    fn select_request_adapter_reports_unavailable_when_no_adapter_matches() {
        let result = select_request_adapter(
            noop_adapters(2),
            native::WGPUBackendType_Vulkan,
            |adapter| backend_type_from_hal(adapter.backend()),
        );

        assert_eq!(
            result.err().as_deref(),
            Some("no adapter with backendType Vulkan (instance backend: Noop)")
        );
    }

    #[test]
    fn select_request_adapter_reports_unavailable_message_on_empty_enumeration() {
        let result = select_request_adapter(Vec::new(), native::WGPUBackendType_Null, |adapter| {
            backend_type_from_hal(adapter.backend())
        });

        assert_eq!(result.err().as_deref(), Some("no adapters available"));
    }

    #[test]
    fn wgpu_instance_request_adapter_null_options_succeeds() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());

            let state = request_adapter(instance, std::ptr::null());

            assert_eq!(state.status, native::WGPURequestAdapterStatus_Success);
            assert!(!state.adapter.is_null());
            assert_eq!(state.message, "");
            wgpuAdapterRelease(state.adapter);
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpu_instance_request_adapter_undefined_backend_succeeds() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());
            let options = request_adapter_options(native::WGPUBackendType_Undefined);

            let state = request_adapter(instance, &options);

            assert_eq!(state.status, native::WGPURequestAdapterStatus_Success);
            assert!(!state.adapter.is_null());
            assert_eq!(state.message, "");
            wgpuAdapterRelease(state.adapter);
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpu_instance_request_adapter_null_backend_succeeds_and_reports_null() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());
            let options = request_adapter_options(native::WGPUBackendType_Null);

            let state = request_adapter(instance, &options);

            assert_eq!(state.status, native::WGPURequestAdapterStatus_Success);
            assert!(!state.adapter.is_null());
            let mut info = std::mem::zeroed();
            assert_eq!(
                wgpuAdapterGetInfo(state.adapter, &mut info),
                native::WGPUStatus_Success
            );
            assert_eq!(info.backendType, native::WGPUBackendType_Null);
            wgpuAdapterInfoFreeMembers(info);
            wgpuAdapterRelease(state.adapter);
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpu_instance_request_adapter_vulkan_backend_is_unavailable() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());
            let options = request_adapter_options(native::WGPUBackendType_Vulkan);

            let state = request_adapter(instance, &options);

            assert_eq!(state.status, native::WGPURequestAdapterStatus_Unavailable);
            assert!(state.adapter.is_null());
            assert_eq!(
                state.message,
                "no adapter with backendType Vulkan (instance backend: Noop)"
            );
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpu_instance_request_adapter_d3d12_backend_is_unavailable() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());
            let options = request_adapter_options(native::WGPUBackendType_D3D12);

            let state = request_adapter(instance, &options);

            assert_eq!(state.status, native::WGPURequestAdapterStatus_Unavailable);
            assert!(state.adapter.is_null());
            assert_eq!(
                state.message,
                "no adapter with backendType D3D12 (instance backend: Noop)"
            );
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpu_supported_wgsl_language_features_free_members_accepts_empty_features() {
        unsafe {
            wgpuSupportedWGSLLanguageFeaturesFreeMembers(
                native::WGPUSupportedWGSLLanguageFeatures {
                    featureCount: 0,
                    features: std::ptr::null(),
                },
            );
        }
    }
}
