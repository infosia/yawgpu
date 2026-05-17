use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::wait;

#[derive(Default)]
struct Recorder {
    events: Vec<Event>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    RequestDevice(native::WGPURequestDeviceStatus),
    DeviceLost(native::WGPUDeviceLostReason, bool),
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
        maxUniformBufferBindingSize: native::WGPU_LIMIT_U64_UNDEFINED as u64,
        maxStorageBufferBindingSize: native::WGPU_LIMIT_U64_UNDEFINED as u64,
        minUniformBufferOffsetAlignment: native::WGPU_LIMIT_U32_UNDEFINED,
        minStorageBufferOffsetAlignment: native::WGPU_LIMIT_U32_UNDEFINED,
        maxVertexBuffers: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBufferSize: native::WGPU_LIMIT_U64_UNDEFINED as u64,
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
