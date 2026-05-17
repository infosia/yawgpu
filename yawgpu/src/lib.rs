pub mod conv;

use std::collections::BTreeMap;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu_core as core;

use crate::conv::{
    add_ref_handle, arc_to_handle, borrow_handle, clone_handle, free_supported_features,
    label_from_string_view, map_device_lost_callback_info, map_device_lost_reason, map_feature,
    map_feature_level, map_features_to_native, map_limits, map_limits_to_native, release_handle,
    string_view, DeviceLostCallbackInfo,
};

pub struct WGPUAdapterImpl {
    core: Arc<core::Adapter>,
    instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUDeviceImpl {
    core: Arc<core::Device>,
    instance: Arc<WGPUInstanceImpl>,
    device_lost_callback: DeviceLostCallbackInfo,
}

pub struct WGPUInstanceImpl {
    core: Arc<core::Instance>,
    timed_wait_any_enabled: bool,
    pending_callbacks: Mutex<BTreeMap<u64, PendingCallback>>,
}

pub struct WGPUQueueImpl {
    core: Arc<core::Queue>,
}

macro_rules! declare_empty_impl_handles {
    ($($name:ident),* $(,)?) => {
        $(
            pub struct $name;
        )*
    };
}

declare_empty_impl_handles!(
    WGPUBindGroupImpl,
    WGPUBindGroupLayoutImpl,
    WGPUBufferImpl,
    WGPUCommandBufferImpl,
    WGPUCommandEncoderImpl,
    WGPUComputePassEncoderImpl,
    WGPUComputePipelineImpl,
    WGPUPipelineLayoutImpl,
    WGPUQuerySetImpl,
    WGPURenderBundleImpl,
    WGPURenderBundleEncoderImpl,
    WGPURenderPassEncoderImpl,
    WGPURenderPipelineImpl,
    WGPUSamplerImpl,
    WGPUShaderModuleImpl,
    WGPUSurfaceImpl,
    WGPUTextureImpl,
    WGPUTextureViewImpl,
);

impl WGPUInstanceImpl {
    fn new_noop(timed_wait_any_enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            core: Arc::new(core::Instance::new_noop()),
            timed_wait_any_enabled,
            pending_callbacks: Mutex::new(BTreeMap::new()),
        })
    }

    fn register_callback(&self, callback: PendingCallback) -> native::WGPUFuture {
        let future = self
            .core
            .future_registry()
            .register(callback.callback_mode());
        self.core.future_registry().complete(future);
        self.pending_callbacks
            .lock()
            .expect("pending callback lock is not poisoned")
            .insert(future.get(), callback);
        native::WGPUFuture { id: future.get() }
    }

    fn process_callbacks(&self) -> usize {
        let ready = self.core.future_registry().process_events();
        let mut callbacks = self
            .pending_callbacks
            .lock()
            .expect("pending callback lock is not poisoned");
        let callbacks_to_fire = ready
            .into_iter()
            .filter_map(|id| callbacks.remove(&id.get()))
            .collect::<Vec<_>>();
        drop(callbacks);

        let count = callbacks_to_fire.len();
        for callback in callbacks_to_fire {
            unsafe {
                callback.fire();
            }
        }
        count
    }

    fn wait_any(&self, future_ids: &[core::FutureId]) -> core::WaitAnyResult {
        let result = self.core.future_registry().wait_any(future_ids, true);

        let mut callbacks = self
            .pending_callbacks
            .lock()
            .expect("pending callback lock is not poisoned");
        let callbacks_to_fire = result
            .callbacks_to_fire
            .iter()
            .filter_map(|id| callbacks.remove(&id.get()))
            .collect::<Vec<_>>();
        drop(callbacks);

        for callback in callbacks_to_fire {
            unsafe {
                callback.fire();
            }
        }

        result
    }
}

impl Drop for WGPUInstanceImpl {
    fn drop(&mut self) {}
}

impl Drop for WGPUAdapterImpl {
    fn drop(&mut self) {}
}

impl Drop for WGPUDeviceImpl {
    fn drop(&mut self) {
        self.schedule_device_lost(std::ptr::null(), core::DeviceLostReason::Destroyed);
    }
}

impl Drop for WGPUQueueImpl {
    fn drop(&mut self) {}
}

impl WGPUDeviceImpl {
    fn schedule_device_lost(
        &self,
        device: native::WGPUDevice,
        reason: core::DeviceLostReason,
    ) -> Option<native::WGPUFuture> {
        let reason = self.core.lose(reason)?;
        if !self.device_lost_callback.has_callback() {
            return None;
        }
        Some(
            self.instance
                .register_callback(PendingCallback::DeviceLost {
                    mode: self.device_lost_callback.mode,
                    callback: self.device_lost_callback.callback,
                    device: device as usize,
                    reason,
                    userdata1: self.device_lost_callback.userdata1,
                    userdata2: self.device_lost_callback.userdata2,
                }),
        )
    }

    #[doc(hidden)]
    pub fn set_uncaptured_error_callback<F>(&self, callback: Option<F>)
    where
        F: Fn(core::DeviceError) + Send + Sync + 'static,
    {
        self.core.set_uncaptured_error_callback(callback);
    }

    #[doc(hidden)]
    pub fn dispatch_error(&self, kind: core::ErrorKind, message: impl Into<String>) {
        self.core.dispatch_error(kind, message);
    }
}

enum PendingCallback {
    RequestAdapter {
        mode: native::WGPUCallbackMode,
        callback: native::WGPURequestAdapterCallback,
        adapter: Arc<WGPUAdapterImpl>,
        userdata1: usize,
        userdata2: usize,
    },
    RequestDevice {
        mode: native::WGPUCallbackMode,
        callback: native::WGPURequestDeviceCallback,
        result: Result<Arc<WGPUDeviceImpl>, String>,
        userdata1: usize,
        userdata2: usize,
    },
    DeviceLost {
        mode: native::WGPUCallbackMode,
        callback: native::WGPUDeviceLostCallback,
        device: usize,
        reason: core::DeviceLostReason,
        userdata1: usize,
        userdata2: usize,
    },
}

impl PendingCallback {
    fn callback_mode(&self) -> core::FutureCallbackMode {
        let mode = match self {
            Self::RequestAdapter { mode, .. }
            | Self::RequestDevice { mode, .. }
            | Self::DeviceLost { mode, .. } => *mode,
        };
        match mode {
            native::WGPUCallbackMode_AllowProcessEvents => {
                core::FutureCallbackMode::AllowProcessEvents
            }
            native::WGPUCallbackMode_AllowSpontaneous => core::FutureCallbackMode::AllowSpontaneous,
            _ => core::FutureCallbackMode::WaitAnyOnly,
        }
    }

    unsafe fn fire(self) {
        match self {
            Self::RequestAdapter {
                callback,
                adapter,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    callback(
                        native::WGPURequestAdapterStatus_Success,
                        arc_to_handle(adapter),
                        string_view(b""),
                        userdata1 as *mut c_void,
                        userdata2 as *mut c_void,
                    );
                }
            }
            Self::RequestDevice {
                callback,
                result,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    match result {
                        Ok(device) => callback(
                            native::WGPURequestDeviceStatus_Success,
                            arc_to_handle(device),
                            string_view(b""),
                            userdata1 as *mut c_void,
                            userdata2 as *mut c_void,
                        ),
                        Err(message) => callback(
                            native::WGPURequestDeviceStatus_Error,
                            std::ptr::null(),
                            string_view(message.as_bytes()),
                            userdata1 as *mut c_void,
                            userdata2 as *mut c_void,
                        ),
                    }
                }
            }
            Self::DeviceLost {
                callback,
                device,
                reason,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    let device = device as native::WGPUDevice;
                    callback(
                        &device,
                        map_device_lost_reason(reason),
                        string_view(device_lost_message(reason).as_bytes()),
                        userdata1 as *mut c_void,
                        userdata2 as *mut c_void,
                    );
                }
            }
        }
    }
}

fn device_lost_message(reason: core::DeviceLostReason) -> &'static str {
    match reason {
        core::DeviceLostReason::Destroyed => "Device was destroyed",
        core::DeviceLostReason::FailedCreation => "Device creation failed",
        core::DeviceLostReason::CallbackCancelled => "Device lost callback was cancelled",
        core::DeviceLostReason::Unknown => "Device was lost",
        _ => "Device was lost",
    }
}

pub mod native {
    #![allow(
        dead_code,
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        improper_ctypes
    )]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// Creates a new Noop-backed WebGPU instance.
///
/// # Safety
///
/// `descriptor`, when non-null, must point to a valid `WGPUInstanceDescriptor`.
#[no_mangle]
pub unsafe extern "C" fn wgpuCreateInstance(
    descriptor: *const native::WGPUInstanceDescriptor,
) -> native::WGPUInstance {
    arc_to_handle(WGPUInstanceImpl::new_noop(instance_has_timed_wait_any(
        descriptor,
    )))
}

/// Releases one owned reference to an instance handle.
///
/// # Safety
///
/// `instance` must be a non-null handle previously returned by yawgpu and not
/// already fully released.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceRelease(instance: native::WGPUInstance) {
    release_handle(instance, "WGPUInstance");
}

/// Adds one owned reference to an instance handle.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceAddRef(instance: native::WGPUInstance) {
    add_ref_handle(instance, "WGPUInstance");
}

/// Requests a Noop adapter from an instance.
///
/// # Safety
///
/// `instance_handle` must be a non-null live yawgpu instance handle. `options`,
/// when non-null, must point to a valid `WGPURequestAdapterOptions`.
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

/// Releases one owned reference to an adapter handle.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterRelease(adapter: native::WGPUAdapter) {
    release_handle(adapter, "WGPUAdapter");
}

/// Adds one owned reference to an adapter handle.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle.
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
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterHasFeature(
    adapter: native::WGPUAdapter,
    feature: native::WGPUFeatureName,
) -> native::WGPUBool {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    native::WGPUBool::from(adapter.core.has_feature(map_feature(feature)))
}

/// Requests a device from an adapter.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle. `descriptor`, when
/// non-null, must point to a valid `WGPUDeviceDescriptor`.
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
    let result = adapter
        .core
        .create_device(
            required_limits.as_ref(),
            &required_features,
            label,
            queue_label,
        )
        .map(|device| {
            Arc::new(WGPUDeviceImpl {
                core: Arc::new(device),
                instance: Arc::clone(&adapter.instance),
                device_lost_callback,
            })
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

/// Releases one owned reference to a device handle.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceRelease(device: native::WGPUDevice) {
    let device = device
        .as_ref()
        .map(|_| device)
        .unwrap_or_else(|| panic!("WGPUDevice must not be null"));
    Arc::increment_strong_count(device);
    let borrowed = Arc::from_raw(device);
    if Arc::strong_count(&borrowed) == 2 {
        borrowed.schedule_device_lost(std::ptr::null(), core::DeviceLostReason::Destroyed);
    }
    drop(borrowed);
    drop(Arc::from_raw(device));
}

/// Adds one owned reference to a device handle.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceAddRef(device: native::WGPUDevice) {
    add_ref_handle(device, "WGPUDevice");
}

/// Destroys a device and fires its device-lost callback once.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceDestroy(device: native::WGPUDevice) {
    let device_impl = borrow_handle(device, "WGPUDevice");
    device_impl.schedule_device_lost(device, core::DeviceLostReason::Destroyed);
}

/// Sets the debug label for a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `label` must point
/// to valid string data according to `WGPUStringView` when non-empty.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceSetLabel(
    device: native::WGPUDevice,
    label: native::WGPUStringView,
) {
    let device = borrow_handle(device, "WGPUDevice");
    let label = label_from_string_view(label).unwrap_or_default();
    device.core.set_label(&label);
}

/// Gets the effective limits for a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `limits` must point
/// to writable `WGPULimits` storage.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetLimits(
    device: native::WGPUDevice,
    limits: *mut native::WGPULimits,
) -> native::WGPUStatus {
    let device = borrow_handle(device, "WGPUDevice");
    let Some(limits) = limits.as_mut() else {
        return native::WGPUStatus_Error;
    };
    *limits = map_limits_to_native(device.core.limits());
    native::WGPUStatus_Success
}

/// Gets the resolved features for a device.
///
/// The returned `features` array is allocated by yawgpu and must be released
/// with `wgpuSupportedFeaturesFreeMembers`.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `features` must
/// point to writable `WGPUSupportedFeatures` storage.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetFeatures(
    device: native::WGPUDevice,
    features: *mut native::WGPUSupportedFeatures,
) {
    let device = borrow_handle(device, "WGPUDevice");
    let features = features
        .as_mut()
        .expect("WGPUSupportedFeatures must not be null");
    *features = map_features_to_native(&device.core.features());
}

/// Returns whether the device has `feature` enabled.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceHasFeature(
    device: native::WGPUDevice,
    feature: native::WGPUFeatureName,
) -> native::WGPUBool {
    let device = borrow_handle(device, "WGPUDevice");
    native::WGPUBool::from(device.core.has_feature(map_feature(feature)))
}

/// Gets the default queue for a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetQueue(device: native::WGPUDevice) -> native::WGPUQueue {
    let device = borrow_handle(device, "WGPUDevice");
    let queue = Arc::new(WGPUQueueImpl {
        core: Arc::new(device.core.queue()),
    });
    arc_to_handle(queue)
}

/// Releases one owned reference to a queue handle.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueRelease(queue: native::WGPUQueue) {
    release_handle(queue, "WGPUQueue");
}

/// Adds one owned reference to a queue handle.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueAddRef(queue: native::WGPUQueue) {
    add_ref_handle(queue, "WGPUQueue");
}

/// Sets the debug label for a queue.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle. `label` must point to
/// valid string data according to `WGPUStringView` when non-empty.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueSetLabel(
    queue: native::WGPUQueue,
    label: native::WGPUStringView,
) {
    let queue = borrow_handle(queue, "WGPUQueue");
    let label = label_from_string_view(label).unwrap_or_default();
    queue.core.set_label(&label);
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
#[no_mangle]
pub unsafe extern "C" fn wgpuSupportedFeaturesFreeMembers(
    supported_features: native::WGPUSupportedFeatures,
) {
    free_supported_features(supported_features);
}

/// Processes callbacks whose mode allows process-events delivery.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle.
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

unsafe fn instance_has_timed_wait_any(descriptor: *const native::WGPUInstanceDescriptor) -> bool {
    let Some(descriptor) = descriptor.as_ref() else {
        return true;
    };
    if descriptor.requiredFeatureCount == 0 {
        return false;
    }
    let features = descriptor
        .requiredFeatures
        .as_ref()
        .map(|_| {
            std::slice::from_raw_parts(descriptor.requiredFeatures, descriptor.requiredFeatureCount)
        })
        .expect("WGPUInstanceDescriptor requiredFeatures must not be null");
    features.contains(&native::WGPUInstanceFeatureName_TimedWaitAny)
}

unsafe fn required_features_from_descriptor(
    descriptor: &native::WGPUDeviceDescriptor,
) -> Vec<core::Feature> {
    if descriptor.requiredFeatureCount == 0 {
        return Vec::new();
    }
    let features = descriptor
        .requiredFeatures
        .as_ref()
        .map(|_| {
            std::slice::from_raw_parts(descriptor.requiredFeatures, descriptor.requiredFeatureCount)
        })
        .expect("WGPUDeviceDescriptor requiredFeatures must not be null");
    features.iter().copied().map(map_feature).collect()
}

/// Installs a Rust-side uncaptured-error callback for test harnesses.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[doc(hidden)]
pub unsafe fn testing_set_uncaptured_error_callback<F>(
    device: native::WGPUDevice,
    callback: Option<F>,
) where
    F: Fn(core::DeviceError) + Send + Sync + 'static,
{
    borrow_handle(device, "WGPUDevice").set_uncaptured_error_callback(callback);
}

/// Dispatches a Rust-side device error for test harnesses.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[doc(hidden)]
pub unsafe fn testing_dispatch_device_error(
    device: native::WGPUDevice,
    kind: core::ErrorKind,
    message: impl Into<String>,
) {
    borrow_handle(device, "WGPUDevice").dispatch_error(kind, message);
}

/// Returns the device label for validation tests.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[doc(hidden)]
pub unsafe fn testing_get_device_label(device: native::WGPUDevice) -> String {
    borrow_handle(device, "WGPUDevice").core.label()
}

/// Returns the queue label for validation tests.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle.
#[doc(hidden)]
pub unsafe fn testing_get_queue_label(queue: native::WGPUQueue) -> String {
    borrow_handle(queue, "WGPUQueue").core.label()
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn request_adapter_callback(
        status: native::WGPURequestAdapterStatus,
        adapter: native::WGPUAdapter,
        _message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        assert_eq!(status, native::WGPURequestAdapterStatus_Success);
        *(userdata1 as *mut native::WGPUAdapter) = adapter;
    }

    unsafe extern "C" fn request_device_callback(
        status: native::WGPURequestDeviceStatus,
        device: native::WGPUDevice,
        _message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        assert_eq!(status, native::WGPURequestDeviceStatus_Success);
        *(userdata1 as *mut native::WGPUDevice) = device;
    }

    #[test]
    fn instance_add_ref_release_balances_core_arc() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());
            let core = Arc::clone(&borrow_handle(instance, "WGPUInstance").core);
            assert_eq!(Arc::strong_count(&core), 2);

            wgpuInstanceAddRef(instance);
            assert_eq!(Arc::strong_count(&core), 2);

            wgpuInstanceRelease(instance);
            assert_eq!(Arc::strong_count(&core), 2);

            wgpuInstanceRelease(instance);
            assert_eq!(Arc::strong_count(&core), 1);
        }
    }

    #[test]
    fn noop_request_adapter_request_device_process_events_round_trip() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());
            let mut adapter: native::WGPUAdapter = std::ptr::null();

            let adapter_callback_info = native::WGPURequestAdapterCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(request_adapter_callback),
                userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let future =
                wgpuInstanceRequestAdapter(instance, std::ptr::null(), adapter_callback_info);
            assert_ne!(future.id, 0);
            assert!(adapter.is_null());

            wgpuInstanceProcessEvents(instance);
            assert!(!adapter.is_null());

            let mut device: native::WGPUDevice = std::ptr::null();
            let device_callback_info = native::WGPURequestDeviceCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(request_device_callback),
                userdata1: (&mut device as *mut native::WGPUDevice).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let future = wgpuAdapterRequestDevice(adapter, std::ptr::null(), device_callback_info);
            assert_ne!(future.id, 0);
            assert!(device.is_null());

            wgpuInstanceProcessEvents(instance);
            assert!(!device.is_null());

            let queue = wgpuDeviceGetQueue(device);
            assert!(!queue.is_null());

            wgpuQueueRelease(queue);
            wgpuDeviceRelease(device);
            wgpuAdapterRelease(adapter);
            wgpuInstanceRelease(instance);
        }
    }
}
