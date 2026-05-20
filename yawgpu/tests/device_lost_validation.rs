use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_test::{wait, ValidationTest};

#[derive(Default)]
struct Recorder {
    events: Vec<Event>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    RequestDevice(native::WGPURequestDeviceStatus),
    DeviceLost(native::WGPUDeviceLostReason, bool),
    MapAsync(native::WGPUMapAsyncStatus),
    QueueWorkDone(native::WGPUQueueWorkDoneStatus),
    PopErrorScope(native::WGPUPopErrorScopeStatus, native::WGPUErrorType),
}

#[test]
fn request_device_failure_device_lost_allow_spontaneous_orders_after_request() {
    unsafe {
        let fixture = Fixture::new();
        let mut recorder = Recorder::default();
        let mut limits = undefined_limits();
        limits.maxBindGroups = 5;
        let descriptor = failing_device_descriptor(
            native::WGPUCallbackMode_AllowSpontaneous,
            &mut recorder,
            &limits,
        );
        let callback_info = request_device_callback_info(&mut recorder);

        let _future = yawgpu::wgpuAdapterRequestDevice(fixture.adapter, &descriptor, callback_info);
        yawgpu::wgpuInstanceProcessEvents(fixture.instance);

        assert_eq!(
            recorder.events,
            vec![
                Event::RequestDevice(native::WGPURequestDeviceStatus_Error),
                Event::DeviceLost(native::WGPUDeviceLostReason_FailedCreation, true),
            ]
        );
    }
}

#[test]
fn request_device_failure_device_lost_allow_process_events_waits_for_process_events() {
    unsafe {
        let fixture = Fixture::new();
        let mut recorder = Recorder::default();
        let mut limits = undefined_limits();
        limits.maxBindGroups = 5;
        let descriptor = failing_device_descriptor(
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut recorder,
            &limits,
        );
        let callback_info = request_device_callback_info(&mut recorder);

        let future = yawgpu::wgpuAdapterRequestDevice(fixture.adapter, &descriptor, callback_info);
        wait_one(fixture.instance, future);

        assert_eq!(
            recorder.events,
            vec![Event::RequestDevice(native::WGPURequestDeviceStatus_Error)]
        );

        yawgpu::wgpuInstanceProcessEvents(fixture.instance);
        assert_eq!(
            recorder.events,
            vec![
                Event::RequestDevice(native::WGPURequestDeviceStatus_Error),
                Event::DeviceLost(native::WGPUDeviceLostReason_FailedCreation, true),
            ]
        );
    }
}

#[test]
fn device_destroy_fires_device_lost_destroyed_once_and_process_events_is_safe() {
    unsafe {
        let fixture = Fixture::new();
        let mut recorder = Recorder::default();
        let device = request_device_with_lost_callback(
            fixture.instance,
            fixture.adapter,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut recorder,
        );

        yawgpu::wgpuDeviceDestroy(device);
        yawgpu::wgpuDeviceDestroy(device);
        yawgpu::wgpuInstanceProcessEvents(fixture.instance);
        yawgpu::wgpuInstanceProcessEvents(fixture.instance);

        assert_eq!(
            recorder.events,
            vec![Event::DeviceLost(
                native::WGPUDeviceLostReason_Destroyed,
                false
            )]
        );

        yawgpu::wgpuDeviceRelease(device);
    }
}

#[test]
fn final_device_release_implicitly_destroys_and_fires_device_lost_once() {
    unsafe {
        let fixture = Fixture::new();
        let mut recorder = Recorder::default();
        let device = request_device_with_lost_callback(
            fixture.instance,
            fixture.adapter,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut recorder,
        );

        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuInstanceProcessEvents(fixture.instance);
        yawgpu::wgpuInstanceProcessEvents(fixture.instance);

        assert_eq!(
            recorder.events,
            vec![Event::DeviceLost(
                native::WGPUDeviceLostReason_Destroyed,
                true
            )]
        );
    }
}

#[test]
fn get_lost_future_completes_once_on_destroy_and_after_already_lost() {
    unsafe {
        let fixture = Fixture::new();
        let mut recorder = Recorder::default();
        let device = request_device_with_lost_callback(
            fixture.instance,
            fixture.adapter,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut recorder,
        );

        let future = yawgpu::wgpuDeviceGetLostFuture(device);
        assert_wait_pending(fixture.instance, future);

        yawgpu::wgpuDeviceDestroy(device);
        wait_one(fixture.instance, future);
        wait_one(fixture.instance, future);
        assert!(recorder.events.is_empty());

        yawgpu::wgpuInstanceProcessEvents(fixture.instance);
        yawgpu::wgpuInstanceProcessEvents(fixture.instance);
        assert_eq!(
            recorder.events,
            vec![Event::DeviceLost(
                native::WGPUDeviceLostReason_Destroyed,
                false
            )]
        );

        let already_lost = yawgpu::wgpuDeviceGetLostFuture(device);
        wait_one(fixture.instance, already_lost);
        yawgpu::wgpuDeviceRelease(device);
    }
}

#[test]
fn create_paths_after_device_loss_return_handles_without_new_device_errors() {
    let test = ValidationTest::new();
    unsafe {
        yawgpu::wgpuDeviceDestroy(test.device());
        test.clear_errors();

        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst);
        let sampler = yawgpu::wgpuDeviceCreateSampler(test.device(), std::ptr::null());
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);

        assert!(!buffer.is_null());
        assert!(!sampler.is_null());
        assert!(!query_set.is_null());
        assert!(test.errors().is_empty());

        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuSamplerRelease(sampler);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn pending_map_queue_and_pop_error_scope_callbacks_resolve_on_device_loss() {
    let test = ValidationTest::new();
    unsafe {
        let events = Arc::new(Mutex::new(Vec::new()));
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead);
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());

        let map_events = Arc::clone(&events);
        let map_info = map_callback_info(&map_events);
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, 16, map_info);

        let queue_events = Arc::clone(&events);
        let queue_info = queue_work_done_callback_info(&queue_events);
        yawgpu::wgpuQueueOnSubmittedWorkDone(queue, queue_info);

        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_Validation);

        yawgpu::wgpuDeviceDestroy(test.device());
        yawgpu::wgpuInstanceProcessEvents(test.instance());

        let pop_events = Arc::clone(&events);
        let pop_info = pop_error_scope_callback_info(&pop_events);
        let pop_future = yawgpu::wgpuDevicePopErrorScope(test.device(), pop_info);
        wait(test.instance(), pop_future);

        let events = events.lock().expect("events lock").clone();
        assert!(events.contains(&Event::MapAsync(native::WGPUMapAsyncStatus_Aborted)));
        assert!(events.contains(&Event::QueueWorkDone(native::WGPUQueueWorkDoneStatus_Error)));
        assert!(events.contains(&Event::PopErrorScope(
            native::WGPUPopErrorScopeStatus_Success,
            native::WGPUErrorType_NoError
        )));

        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuQueueRelease(queue);
    }
}

struct Fixture {
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
}

impl Fixture {
    unsafe fn new() -> Self {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        assert!(!instance.is_null());

        let mut adapter: native::WGPUAdapter = std::ptr::null();
        let callback_info = native::WGPURequestAdapterCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(request_adapter_callback),
            userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future = yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info);
        wait(instance, future);
        assert!(!adapter.is_null());

        Self { instance, adapter }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        unsafe {
            yawgpu::wgpuAdapterRelease(self.adapter);
            yawgpu::wgpuInstanceRelease(self.instance);
        }
    }
}

unsafe fn request_device_with_lost_callback(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    lost_mode: native::WGPUCallbackMode,
    recorder: &mut Recorder,
) -> native::WGPUDevice {
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: string_view_init(),
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
        defaultQueue: queue_descriptor_init(),
        deviceLostCallbackInfo: device_lost_callback_info(lost_mode, recorder),
        uncapturedErrorCallbackInfo: unsafe { std::mem::zeroed() },
    };
    let mut device: native::WGPUDevice = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_success_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, &descriptor, callback_info);
    wait(instance, future);
    assert!(!device.is_null());
    device
}

fn failing_device_descriptor(
    lost_mode: native::WGPUCallbackMode,
    recorder: &mut Recorder,
    limits: &native::WGPULimits,
) -> native::WGPUDeviceDescriptor {
    native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: string_view_init(),
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: limits,
        defaultQueue: queue_descriptor_init(),
        deviceLostCallbackInfo: device_lost_callback_info(lost_mode, recorder),
        uncapturedErrorCallbackInfo: unsafe { std::mem::zeroed() },
    }
}

fn request_device_callback_info(recorder: &mut Recorder) -> native::WGPURequestDeviceCallbackInfo {
    native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_error_callback),
        userdata1: (recorder as *mut Recorder).cast(),
        userdata2: std::ptr::null_mut(),
    }
}

fn device_lost_callback_info(
    mode: native::WGPUCallbackMode,
    recorder: &mut Recorder,
) -> native::WGPUDeviceLostCallbackInfo {
    native::WGPUDeviceLostCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode,
        callback: Some(device_lost_callback),
        userdata1: (recorder as *mut Recorder).cast(),
        userdata2: std::ptr::null_mut(),
    }
}

unsafe fn wait_one(instance: native::WGPUInstance, future: native::WGPUFuture) {
    let mut wait_info = native::WGPUFutureWaitInfo {
        future,
        completed: 0,
    };
    let status = yawgpu::wgpuInstanceWaitAny(instance, 1, &mut wait_info, 0);
    assert_eq!(status, native::WGPUWaitStatus_Success);
    assert_eq!(wait_info.completed, 1);
}

unsafe fn assert_wait_pending(instance: native::WGPUInstance, future: native::WGPUFuture) {
    let mut wait_info = native::WGPUFutureWaitInfo {
        future,
        completed: 0,
    };
    let status = yawgpu::wgpuInstanceWaitAny(instance, 1, &mut wait_info, 0);
    assert_eq!(status, native::WGPUWaitStatus_TimedOut);
    assert_eq!(wait_info.completed, 0);
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: string_view_init(),
        usage,
        size,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

unsafe fn create_query_set(
    device: native::WGPUDevice,
    query_type: native::WGPUQueryType,
    count: u32,
) -> native::WGPUQuerySet {
    let descriptor = native::WGPUQuerySetDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: string_view_init(),
        type_: query_type,
        count,
    };
    let query_set = yawgpu::wgpuDeviceCreateQuerySet(device, &descriptor);
    assert!(!query_set.is_null());
    query_set
}

fn map_callback_info(events: &Arc<Mutex<Vec<Event>>>) -> native::WGPUBufferMapCallbackInfo {
    native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (events as *const Arc<Mutex<Vec<Event>>>).cast_mut().cast(),
        userdata2: std::ptr::null_mut(),
    }
}

fn queue_work_done_callback_info(
    events: &Arc<Mutex<Vec<Event>>>,
) -> native::WGPUQueueWorkDoneCallbackInfo {
    native::WGPUQueueWorkDoneCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(queue_work_done_callback),
        userdata1: (events as *const Arc<Mutex<Vec<Event>>>).cast_mut().cast(),
        userdata2: std::ptr::null_mut(),
    }
}

fn pop_error_scope_callback_info(
    events: &Arc<Mutex<Vec<Event>>>,
) -> native::WGPUPopErrorScopeCallbackInfo {
    native::WGPUPopErrorScopeCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(pop_error_scope_callback),
        userdata1: (events as *const Arc<Mutex<Vec<Event>>>).cast_mut().cast(),
        userdata2: std::ptr::null_mut(),
    }
}

fn string_view_init() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

fn queue_descriptor_init() -> native::WGPUQueueDescriptor {
    native::WGPUQueueDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: string_view_init(),
    }
}

fn undefined_limits() -> native::WGPULimits {
    native::WGPULimits {
        nextInChain: std::ptr::null_mut(),
        maxTextureDimension1D: native::WGPU_LIMIT_U32_UNDEFINED,
        maxTextureDimension2D: native::WGPU_LIMIT_U32_UNDEFINED,
        maxTextureDimension3D: native::WGPU_LIMIT_U32_UNDEFINED,
        maxTextureArrayLayers: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBindGroups: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBindGroupsPlusVertexBuffers: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBindingsPerBindGroup: native::WGPU_LIMIT_U32_UNDEFINED,
        maxDynamicUniformBuffersPerPipelineLayout: native::WGPU_LIMIT_U32_UNDEFINED,
        maxDynamicStorageBuffersPerPipelineLayout: native::WGPU_LIMIT_U32_UNDEFINED,
        maxSampledTexturesPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxSamplersPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxStorageBuffersPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxStorageTexturesPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxUniformBuffersPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxUniformBufferBindingSize: native::WGPU_LIMIT_U64_UNDEFINED,
        maxStorageBufferBindingSize: native::WGPU_LIMIT_U64_UNDEFINED,
        minUniformBufferOffsetAlignment: native::WGPU_LIMIT_U32_UNDEFINED,
        minStorageBufferOffsetAlignment: native::WGPU_LIMIT_U32_UNDEFINED,
        maxVertexBuffers: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBufferSize: native::WGPU_LIMIT_U64_UNDEFINED,
        maxVertexAttributes: native::WGPU_LIMIT_U32_UNDEFINED,
        maxVertexBufferArrayStride: native::WGPU_LIMIT_U32_UNDEFINED,
        maxInterStageShaderVariables: native::WGPU_LIMIT_U32_UNDEFINED,
        maxColorAttachments: native::WGPU_LIMIT_U32_UNDEFINED,
        maxColorAttachmentBytesPerSample: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupStorageSize: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeInvocationsPerWorkgroup: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupSizeX: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupSizeY: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupSizeZ: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupsPerDimension: native::WGPU_LIMIT_U32_UNDEFINED,
        maxImmediateSize: native::WGPU_LIMIT_U32_UNDEFINED,
    }
}

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

unsafe extern "C" fn request_device_success_callback(
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestDeviceStatus_Success);
    *(userdata1 as *mut native::WGPUDevice) = device;
}

unsafe extern "C" fn request_device_error_callback(
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestDeviceStatus_Error);
    assert!(device.is_null());
    assert!(!string_view_to_string(message).is_empty());
    (*(userdata1 as *mut Recorder))
        .events
        .push(Event::RequestDevice(status));
}

unsafe extern "C" fn device_lost_callback(
    device: *const native::WGPUDevice,
    reason: native::WGPUDeviceLostReason,
    message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert!(!device.is_null());
    assert!(!string_view_to_string(message).is_empty());
    (*(userdata1 as *mut Recorder))
        .events
        .push(Event::DeviceLost(reason, (*device).is_null()));
}

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let events = &*(userdata1 as *const Arc<Mutex<Vec<Event>>>);
    events
        .lock()
        .expect("events lock")
        .push(Event::MapAsync(status));
}

unsafe extern "C" fn queue_work_done_callback(
    status: native::WGPUQueueWorkDoneStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let events = &*(userdata1 as *const Arc<Mutex<Vec<Event>>>);
    events
        .lock()
        .expect("events lock")
        .push(Event::QueueWorkDone(status));
}

unsafe extern "C" fn pop_error_scope_callback(
    status: native::WGPUPopErrorScopeStatus,
    error_type: native::WGPUErrorType,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let events = &*(userdata1 as *const Arc<Mutex<Vec<Event>>>);
    events
        .lock()
        .expect("events lock")
        .push(Event::PopErrorScope(status, error_type));
}

unsafe fn string_view_to_string(value: native::WGPUStringView) -> String {
    if value.data.is_null() {
        return String::new();
    }
    let bytes = if value.length == usize::MAX {
        std::ffi::CStr::from_ptr(value.data).to_bytes()
    } else {
        std::slice::from_raw_parts(value.data.cast::<u8>(), value.length)
    };
    String::from_utf8_lossy(bytes).into_owned()
}
