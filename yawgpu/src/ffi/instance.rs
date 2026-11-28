use super::*;

const SUPPORTED_INSTANCE_FEATURES: &[native::WGPUInstanceFeatureName] = &[
    native::WGPUInstanceFeatureName_TimedWaitAny,
    #[cfg(feature = "shader-passthrough")]
    native::WGPUInstanceFeatureName_ShaderSourceSPIRV,
];

/// Default and supported maximum number of futures in a timed wait.
pub(crate) const TIMED_WAIT_ANY_MAX_COUNT: usize = 64;

#[cfg(any(feature = "metal", feature = "vulkan", feature = "gles"))]
fn finish_real_instance(
    hal_instance: HalInstance,
    prefix: &str,
    backend: &str,
) -> Option<core::Instance> {
    if hal_instance.enumerate_adapters().is_empty() {
        eprintln!(
            "{prefix}: {backend} requested but enumerate_adapters returned no adapters; returning NULL instance"
        );
        return None;
    }
    Some(core::Instance::from_hal(hal_instance))
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
/// Returns NULL for unsupported required limits, a missing required-feature
/// array, or an unavailable requested backend.
#[no_mangle]
pub unsafe extern "C" fn wgpuCreateInstance(
    descriptor: *const native::WGPUInstanceDescriptor,
) -> native::WGPUInstance {
    if let Some(descriptor) = descriptor.as_ref() {
        if descriptor.requiredFeatureCount > 0 && descriptor.requiredFeatures.is_null() {
            return std::ptr::null();
        }
        if descriptor
            .requiredLimits
            .as_ref()
            .is_some_and(|limits| limits.timedWaitAnyMaxCount > TIMED_WAIT_ANY_MAX_COUNT)
        {
            return std::ptr::null();
        }
    }
    let timed_wait_any_enabled = instance_has_timed_wait_any(descriptor);
    let shader_source_spirv_enabled = instance_has_shader_source_spirv(descriptor);
    let selection = instance_backend_selection(descriptor);
    let instance = match selection {
        // IB1 (no chain) and IB2 (chain == NOOP) / IB4 (chain == unknown).
        None | Some(InstanceBackendSelection::Noop) => core::Instance::new_noop(),
        Some(InstanceBackendSelection::Metal) => {
            #[cfg(feature = "metal")]
            {
                match yawgpu_hal::metal::MetalInstance::new() {
                    Ok(instance) => {
                        let Some(instance) = finish_real_instance(
                            yawgpu_hal::HalInstance::Metal(instance),
                            "yawgpu-metal",
                            "YAWGPU_INSTANCE_BACKEND_METAL",
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
                eprintln!(
                    "yawgpu-gles: YAWGPU_INSTANCE_BACKEND_GLES requested but yawgpu was built without feature=gles; returning NULL instance"
                );
                return std::ptr::null();
            }
        }
    };
    // Every supported request receives the better default maximum.
    arc_to_handle(WGPUInstanceImpl::from_core(
        instance,
        timed_wait_any_enabled,
        TIMED_WAIT_ANY_MAX_COUNT,
        shader_source_spirv_enabled,
    ))
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
    if !limits.nextInChain.is_null() {
        return native::WGPUStatus_Error;
    }
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
    let _ = borrow_handle(instance, "WGPUInstance");
    let owned = Arc::from_raw(instance);
    owned
        .external_refs
        .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    owned.release_use_ref();
}

/// Adds one owned reference to an instance handle.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle.
/// Returns WGPU instance add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceAddRef(instance: native::WGPUInstance) {
    let instance_impl = borrow_handle(instance, "WGPUInstance");
    instance_impl.add_use_ref();
    instance_impl
        .external_refs
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    let descriptor = descriptor.as_ref();
    let label = descriptor.and_then(|desc| label_from_string_view(desc.label));
    let descriptor = descriptor.filter(|desc| !has_surface_color_management(desc.nextInChain));
    let (is_error, hal) = if let Some(descriptor) = descriptor {
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
            surface_source_is_unsupported || real_surface_creation_failed,
            hal,
        )
    } else {
        (true, None)
    };
    arc_to_handle(Arc::new(WGPUSurfaceImpl {
        label: Mutex::new(label),
        configured: Mutex::new(None),
        current_texture: Mutex::new(None),
        #[cfg(test)]
        acquire_texture_override: Mutex::new(None),
        #[cfg(test)]
        configure_error_override: Mutex::new(None),
        #[cfg(test)]
        present_error_override: Mutex::new(None),
        #[cfg(test)]
        pending_flush_error_override: Mutex::new(None),
        hal: Mutex::new(hal),
        is_error,
        _instance: instance,
    }))
}

/// Requests an adapter from an instance.
///
/// The callback receives the highest-ranked eligible adapter. A WebXR request
/// or an empty selection completes with `WGPURequestAdapterStatus_Unavailable`
/// and a null adapter.
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
    if !valid_callback_mode(callback_info.mode) {
        return native::WGPUFuture { id: 0 };
    }
    let instance = clone_handle(instance_handle, "WGPUInstance");
    let options = options.as_ref();
    let mut options_error = None;
    let feature_level = options
        .map(|options| map_feature_level(options.featureLevel, &mut options_error))
        .unwrap_or(core::FeatureLevel::Core);
    let webxr = options.is_some_and(|options| {
        find_in_chain::<native::WGPURequestAdapterWebXROptions>(
            options.nextInChain,
            native::WGPUSType_RequestAdapterWebXROptions,
        )
        .is_some()
    });
    let mut error_status = native::WGPURequestAdapterStatus_Unavailable;
    let result = if webxr {
        Err("RequestAdapterWebXROptions unsupported".to_owned())
    } else if let Some(error) = options_error {
        error_status = native::WGPURequestAdapterStatus_Error;
        Err(error)
    } else {
        let surface = options.and_then(|options| options.compatibleSurface.as_ref());
        let force_fallback = options.is_some_and(|options| options.forceFallbackAdapter != 0);
        select_request_adapter(
            instance
                .core
                .enumerate_adapters_with_feature_level(feature_level)
                .into_iter()
                .filter(|adapter| !force_fallback || adapter.hal().is_fallback())
                .collect(),
            options.map_or(native::WGPUBackendType_Undefined, |options| {
                options.backendType
            }),
            options.map_or(native::WGPUPowerPreference_Undefined, |options| {
                options.powerPreference
            }),
            core::Adapter::backend,
            |adapter| surface.is_none_or(|surface| adapter_supports_surface(adapter, surface)),
            |adapter| adapter.info().adapter_type,
        )
    }
    .map(|adapter| {
        Arc::new(WGPUAdapterImpl {
            external_refs: std::sync::atomic::AtomicUsize::new(0),
            core: Arc::new(adapter),
            instance: clone_handle(instance_handle, "WGPUInstance"),
        })
    });

    instance.register_callback(PendingCallback::RequestAdapter {
        error_status,
        mode: callback_info.mode,
        callback: callback_info.callback,
        result,
        userdata1: callback_info.userdata1 as usize,
        userdata2: callback_info.userdata2 as usize,
    })
}

/// Filters the enumeration by backend and presentation support before ranking it.
fn select_request_adapter<T>(
    mut adapters: Vec<T>,
    requested_backend_type: native::WGPUBackendType,
    preference: native::WGPUPowerPreference,
    backend: impl Fn(&T) -> yawgpu_hal::HalBackend,
    supports_surface: impl FnMut(&T) -> bool,
    adapter_type: impl Fn(&T) -> yawgpu_hal::HalAdapterType,
) -> Result<T, String> {
    let instance_backend = adapters
        .first()
        .map(&backend)
        .ok_or_else(|| String::from("no adapters available"))?;
    if requested_backend_type != native::WGPUBackendType_Undefined {
        adapters
            .retain(|adapter| backend_type_from_hal(backend(adapter)) == requested_backend_type);
        if adapters.is_empty() {
            return Err(format!(
                "no adapter with backendType {} (instance backend: {instance_backend:?})",
                backend_type_name(requested_backend_type)
            ));
        }
    }
    adapters.retain(supports_surface);
    sort_adapters_by_preference(&mut adapters, preference, adapter_type);
    adapters
        .into_iter()
        .next()
        .ok_or_else(|| "no adapter supports the requested surface".to_owned())
}

fn adapter_supports_surface(adapter: &core::Adapter, surface: &WGPUSurfaceImpl) -> bool {
    !surface.is_error && surface::query_surface_capabilities(surface, adapter.hal()).is_ok()
}

fn adapter_type_rank(
    kind: yawgpu_hal::HalAdapterType,
    preference: native::WGPUPowerPreference,
) -> u8 {
    use yawgpu_hal::HalAdapterType::{Cpu, Discrete, Integrated};
    match (kind, preference) {
        (Discrete, native::WGPUPowerPreference_HighPerformance)
        | (Integrated, native::WGPUPowerPreference_LowPower) => 0,
        (Discrete | Integrated, _) => 1,
        (Cpu, _) => 2,
        _ => 3,
    }
}

fn sort_adapters_by_preference<T>(
    adapters: &mut [T],
    preference: native::WGPUPowerPreference,
    adapter_type: impl Fn(&T) -> yawgpu_hal::HalAdapterType,
) {
    adapters.sort_by_cached_key(|adapter| adapter_type_rank(adapter_type(adapter), preference));
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
    // Polling submission gates can dispatch an uncaptured error reentrantly.
    let instance = clone_handle(instance, "WGPUInstance");
    instance.process_callbacks();
}

/// Yields for four polls, then doubles sleeps from 50 microseconds to 1 millisecond.
/// The caller supplies the time remaining so a sleep never exceeds the timeout.
fn wait_any_backoff(attempt: u32, remaining: std::time::Duration) -> std::time::Duration {
    if attempt < 4 {
        return std::time::Duration::ZERO;
    }
    std::time::Duration::from_micros((50u64 << (attempt - 4).min(5)).min(1000)).min(remaining)
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
    // Polling submission gates can dispatch an uncaptured error reentrantly.
    let instance = clone_handle(instance, "WGPUInstance");
    if future_count > 0 && futures.is_null() {
        return native::WGPUWaitStatus_Error;
    }
    if timeout_ns > 0 && !instance.timed_wait_any_enabled {
        return native::WGPUWaitStatus_Error;
    }
    if timeout_ns > 0 && future_count > instance.timed_wait_any_max_count {
        return native::WGPUWaitStatus_Error;
    }
    if future_count == 0 {
        return native::WGPUWaitStatus_Success;
    }

    let wait_infos = std::slice::from_raw_parts_mut(futures, future_count);
    let future_ids = wait_infos
        .iter()
        .map(|info| core::FutureId::from_raw(info.future.id))
        .collect::<Vec<_>>();
    let timeout = std::time::Duration::from_nanos(timeout_ns);
    let started = std::time::Instant::now();
    let mut attempt = 0u32;
    let result = loop {
        let result = instance.wait_any(&future_ids);
        if result.status != core::WaitAnyStatus::TimedOut
            || timeout_ns == 0
            || started.elapsed() >= timeout
        {
            break result;
        }
        let sleep = wait_any_backoff(attempt, timeout.saturating_sub(started.elapsed()));
        if sleep.is_zero() {
            std::thread::yield_now();
        } else {
            std::thread::sleep(sleep);
        }
        attempt = attempt.saturating_add(1);
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

    #[test]
    fn wait_any_backoff_yields_doubles_caps_and_respects_remaining_time() {
        use std::time::Duration;
        let remaining = Duration::from_secs(1);
        for attempt in 0..4 {
            assert_eq!(wait_any_backoff(attempt, remaining), Duration::ZERO);
        }
        for (attempt, micros) in (4..).zip([50, 100, 200, 400, 800, 1000, 1000]) {
            assert_eq!(
                wait_any_backoff(attempt, remaining),
                Duration::from_micros(micros)
            );
        }
        assert_eq!(
            wait_any_backoff(u32::MAX, remaining),
            Duration::from_millis(1)
        );
        for attempt in [4, 5, 9, u32::MAX] {
            for remaining in [
                Duration::ZERO,
                Duration::from_nanos(1),
                Duration::from_micros(25),
            ] {
                assert_eq!(wait_any_backoff(attempt, remaining), remaining);
            }
        }
    }

    #[test]
    fn wait_any_pending_future_times_out_with_bounded_delay() {
        use std::time::{Duration, Instant};
        let instance = WGPUInstanceImpl::from_core(
            core::Instance::new_noop(),
            true,
            TIMED_WAIT_ANY_MAX_COUNT,
            false,
        );
        let future = instance
            .core
            .future_registry()
            .register(core::FutureCallbackMode::WaitAnyOnly);
        let handle = Arc::into_raw(instance);
        let mut info = native::WGPUFutureWaitInfo {
            future: native::WGPUFuture { id: future.get() },
            completed: 1,
        };
        let timeout = Duration::from_millis(10);
        let started = Instant::now();
        let status =
            unsafe { wgpuInstanceWaitAny(handle, 1, &mut info, timeout.as_nanos() as u64) };
        let elapsed = started.elapsed();
        unsafe { wgpuInstanceRelease(handle) };
        assert_eq!(status, native::WGPUWaitStatus_TimedOut);
        assert_eq!(info.completed, 0);
        assert!(elapsed >= timeout, "elapsed: {elapsed:?}");
        assert!(
            elapsed <= timeout + Duration::from_millis(500),
            "elapsed: {elapsed:?}"
        );
    }

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
    fn create_instance_negotiates_required_timed_wait_count() {
        unsafe {
            for required in [None, Some(0), Some(1), Some(64), Some(65)] {
                let limits = required.map(|count| native::WGPUInstanceLimits {
                    nextInChain: std::ptr::null_mut(),
                    timedWaitAnyMaxCount: count,
                });
                let descriptor = native::WGPUInstanceDescriptor {
                    requiredLimits: limits.as_ref().map_or(std::ptr::null(), std::ptr::from_ref),
                    ..std::mem::zeroed()
                };
                let instance = wgpuCreateInstance(&descriptor);
                assert_eq!(instance.is_null(), required == Some(65));
                if let Some(instance_impl) = instance.as_ref() {
                    assert_eq!(instance_impl.timed_wait_any_max_count, 64);
                    wgpuInstanceRelease(instance);
                }
            }
        }
    }

    fn ready_waits<const N: usize>(instance: &WGPUInstanceImpl) -> [native::WGPUFutureWaitInfo; N] {
        let registry = instance.core.future_registry();
        std::array::from_fn(|_| {
            let id = registry.register(core::FutureCallbackMode::WaitAnyOnly);
            registry.complete(id);
            native::WGPUFutureWaitInfo {
                future: native::WGPUFuture { id: id.get() },
                completed: 0,
            }
        })
    }

    #[test]
    fn wait_any_limits_only_positive_timeouts() {
        unsafe {
            let feature = native::WGPUInstanceFeatureName_TimedWaitAny;
            let descriptor = native::WGPUInstanceDescriptor {
                requiredFeatureCount: 1,
                requiredFeatures: &feature,
                ..std::mem::zeroed()
            };
            let instance = wgpuCreateInstance(&descriptor);
            assert_eq!(
                wgpuInstanceWaitAny(instance, 0, std::ptr::null_mut(), 1),
                native::WGPUWaitStatus_Success
            );
            let mut waits = ready_waits::<65>(borrow_handle(instance, "WGPUInstance"));
            assert_eq!(
                wgpuInstanceWaitAny(instance, 65, waits.as_mut_ptr(), 1),
                native::WGPUWaitStatus_Error
            );
            assert!(waits.iter().all(|info| info.completed == 0));
            assert_eq!(
                wgpuInstanceWaitAny(instance, 64, waits.as_mut_ptr(), 1),
                native::WGPUWaitStatus_Success
            );
            assert!(waits[..64].iter().all(|info| info.completed != 0));
            assert_eq!(
                wgpuInstanceWaitAny(instance, 65, waits.as_mut_ptr(), 0),
                native::WGPUWaitStatus_Success
            );
            assert!(waits.iter().all(|info| info.completed != 0));
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wait_any_reads_the_limit_stored_on_the_instance() {
        unsafe {
            let instance = WGPUInstanceImpl::from_core(core::Instance::new_noop(), true, 1, false);
            let mut waits = ready_waits::<2>(&instance);
            let handle = arc_to_handle(instance);
            assert_eq!(
                wgpuInstanceWaitAny(handle, waits.len(), waits.as_mut_ptr(), 1),
                native::WGPUWaitStatus_Error
            );
            wgpuInstanceRelease(handle);
        }
    }

    #[test]
    fn get_instance_limits_rejects_chained_output() {
        unsafe {
            let mut chain = std::mem::zeroed();
            let mut limits = native::WGPUInstanceLimits {
                nextInChain: &mut chain,
                timedWaitAnyMaxCount: 7,
            };
            assert_eq!(wgpuGetInstanceLimits(&mut limits), native::WGPUStatus_Error);
            assert_eq!(limits.timedWaitAnyMaxCount, 7);
        }
    }

    #[test]
    fn create_instance_and_request_adapter_accept_dawn_toggles() {
        unsafe {
            let mut toggles = native::WGPUDawnTogglesDescriptor {
                chain: native::WGPUChainedStruct {
                    next: std::ptr::null_mut(),
                    sType: native::WGPUSType_DawnTogglesDescriptor,
                },
                ..std::mem::zeroed()
            };
            let descriptor = native::WGPUInstanceDescriptor {
                nextInChain: &mut toggles.chain,
                ..std::mem::zeroed()
            };
            let instance = wgpuCreateInstance(&descriptor);
            assert!(!instance.is_null());

            let enabled = [
                c"use_dxc".as_ptr(),
                c"allow_unsafe_apis".as_ptr(),
                c"decompose_uniform_buffers".as_ptr(),
                c"use_vulkan_memory_model".as_ptr(),
                c"vulkan_enable_f16_on_nvidia".as_ptr(),
            ];
            toggles.enabledToggleCount = enabled.len();
            toggles.enabledToggles = enabled.as_ptr();
            let mut options = request_adapter_options(native::WGPUBackendType_Null);
            options.nextInChain = &mut toggles.chain;
            let state = request_adapter(instance, &options);
            assert_eq!(state.status, native::WGPURequestAdapterStatus_Success);
            assert!(!state.adapter.is_null());
            wgpuAdapterRelease(state.adapter);

            let mut xr = native::WGPURequestAdapterWebXROptions {
                chain: native::WGPUChainedStruct {
                    next: std::ptr::null_mut(),
                    sType: native::WGPUSType_RequestAdapterWebXROptions,
                },
                xrCompatible: 1,
            };
            toggles.chain.next = &mut xr.chain;
            options.nextInChain = &mut toggles.chain;
            let state = request_adapter(instance, &options);
            assert_eq!(state.status, native::WGPURequestAdapterStatus_Unavailable);
            assert!(state.adapter.is_null());
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn request_adapter_declines_webxr_anywhere_in_chain() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());
            let mut options = request_adapter_options(native::WGPUBackendType_Null);
            let state = request_adapter(instance, &options);
            assert_eq!(state.status, native::WGPURequestAdapterStatus_Success);
            assert!(!state.adapter.is_null());
            wgpuAdapterRelease(state.adapter);
            for xr_compatible in [0, 1] {
                let mut xr = native::WGPURequestAdapterWebXROptions {
                    chain: native::WGPUChainedStruct {
                        next: std::ptr::null_mut(),
                        sType: native::WGPUSType_RequestAdapterWebXROptions,
                    },
                    xrCompatible: xr_compatible,
                };
                let mut head = native::WGPUChainedStruct {
                    next: &mut xr.chain,
                    sType: native::WGPUSType_Force32,
                };
                options.nextInChain = &mut head;
                let state = request_adapter(instance, &options);
                assert_eq!(state.status, native::WGPURequestAdapterStatus_Unavailable);
                assert!(state.adapter.is_null());
            }
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn request_adapter_filters_by_surface_capabilities() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());
            let error_surface = wgpuInstanceCreateSurface(instance, std::ptr::null());
            let valid_surface = super::super::tests::create_noop_surface(instance);
            let adapter = noop_adapters(1).pop().unwrap();
            let mut options = request_adapter_options(native::WGPUBackendType_Null);
            options.powerPreference = native::WGPUPowerPreference_HighPerformance;
            for (surface, supported) in [(error_surface, false), (valid_surface, true)] {
                assert!(!surface.is_null());
                assert_eq!(adapter_supports_surface(&adapter, &*surface), supported);
                options.compatibleSurface = surface;
                let state = request_adapter(instance, &options);
                if supported {
                    assert_eq!(state.status, native::WGPURequestAdapterStatus_Success);
                    assert!(!state.adapter.is_null());
                    let mut caps = std::mem::zeroed();
                    assert_eq!(
                        wgpuSurfaceGetCapabilities(surface, state.adapter, &mut caps),
                        native::WGPUStatus_Success
                    );
                    wgpuSurfaceCapabilitiesFreeMembers(caps);
                    wgpuAdapterRelease(state.adapter);
                } else {
                    assert_eq!(state.status, native::WGPURequestAdapterStatus_Unavailable);
                    assert!(state.adapter.is_null());
                }
                wgpuSurfaceRelease(surface);
            }
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn power_preference_ranks_gpu_types_and_preserves_ties() {
        use yawgpu_hal::HalAdapterType::{Cpu, Discrete, Integrated, Unknown};
        for (preference, discrete_rank, integrated_rank) in [
            (native::WGPUPowerPreference_HighPerformance, 0, 1),
            (native::WGPUPowerPreference_LowPower, 1, 0),
            (native::WGPUPowerPreference_Undefined, 1, 1),
            (0xCAFE, 1, 1),
        ] {
            assert_eq!(adapter_type_rank(Discrete, preference), discrete_rank);
            assert_eq!(adapter_type_rank(Integrated, preference), integrated_rank);
            assert_eq!(adapter_type_rank(Cpu, preference), 2);
            assert_eq!(adapter_type_rank(Unknown, preference), 3);
            for (first, second) in [(Discrete, Integrated), (Integrated, Discrete)] {
                let mut adapters = [(Unknown, 0), (first, 1), (Cpu, 2), (second, 3), (first, 4)];
                sort_adapters_by_preference(&mut adapters, preference, |&(kind, _)| kind);
                let expected = match preference {
                    native::WGPUPowerPreference_HighPerformance if first == Integrated => {
                        [3, 1, 4, 2, 0]
                    }
                    native::WGPUPowerPreference_LowPower if first == Discrete => [3, 1, 4, 2, 0],
                    native::WGPUPowerPreference_Undefined | 0xCAFE => [1, 3, 4, 2, 0],
                    _ => [1, 4, 3, 2, 0],
                };
                assert_eq!(adapters.map(|(_, index)| index), expected);
            }
        }
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
            // Block 119 R8: reported through both entry points.
            for reported in [
                native::WGPUWGSLLanguageFeatureName_BufferView,
                native::WGPUWGSLLanguageFeatureName_SwizzleAssignment,
            ] {
                assert!(values.contains(&reported));
                assert_eq!(
                    wgpuInstanceHasWGSLLanguageFeature(instance, reported),
                    native::WGPUBool::from(true)
                );
            }
            // Block 119 R9: `fragment_depth` and unknown values stay unreported.
            for unreported in [
                native::WGPUWGSLLanguageFeatureName_FragmentDepth,
                0xFFFF as native::WGPUWGSLLanguageFeatureName,
            ] {
                assert!(!values.contains(&unreported));
                assert_eq!(
                    wgpuInstanceHasWGSLLanguageFeature(instance, unreported),
                    native::WGPUBool::from(false)
                );
            }

            wgpuSupportedWGSLLanguageFeaturesFreeMembers(features);
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn select_request_adapter_filters_before_ranking_and_returns_first_tie() {
        use yawgpu_hal::HalAdapterType::{Discrete, Integrated};
        use yawgpu_hal::HalBackend::{Noop, Vulkan};
        let candidates = vec![
            (Vulkan, true, Discrete, 0),
            (Noop, false, Discrete, 1),
            (Noop, true, Integrated, 2),
            (Noop, true, Discrete, 3),
            (Noop, true, Discrete, 4),
        ];
        let result = select_request_adapter(
            candidates,
            native::WGPUBackendType_Null,
            native::WGPUPowerPreference_HighPerformance,
            |candidate| candidate.0,
            |&(backend, supported, _, _)| {
                assert_eq!(backend, Noop);
                supported
            },
            |&(_, supported, kind, _)| {
                assert!(supported);
                kind
            },
        );
        assert_eq!(result, Ok((Noop, true, Discrete, 3)));
    }

    #[test]
    fn select_request_adapter_reports_unavailable_when_no_adapter_matches() {
        let result = select_request_adapter(
            noop_adapters(2),
            native::WGPUBackendType_Vulkan,
            native::WGPUPowerPreference_Undefined,
            core::Adapter::backend,
            |_| true,
            |adapter| adapter.info().adapter_type,
        );

        assert_eq!(
            result.err().as_deref(),
            Some("no adapter with backendType Vulkan (instance backend: Noop)")
        );
    }

    #[test]
    fn select_request_adapter_reports_unavailable_message_on_empty_enumeration() {
        let result = select_request_adapter(
            Vec::new(),
            native::WGPUBackendType_Null,
            native::WGPUPowerPreference_Undefined,
            core::Adapter::backend,
            |_| true,
            |adapter| adapter.info().adapter_type,
        );

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
