use std::os::raw::c_void;
#[cfg(feature = "vulkan")]
use std::sync::{Arc, Mutex};

use yawgpu::native;
#[cfg(feature = "vulkan")]
use yawgpu::{
    WGPUYawgpuInstanceBackendSelect, WGPU_STYPE_YAWGPU_INSTANCE_BACKEND_SELECT,
    WGPU_YAWGPU_INSTANCE_BACKEND_VULKAN,
};
use yawgpu_test::wait;
#[cfg(feature = "vulkan")]
use yawgpu_test::{real_backend_skip_reason, RealBackend};

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_request_adapter_reports_non_empty_name() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        assert!(!adapter.is_null());

        let mut info = zeroed_adapter_info();
        assert_eq!(
            yawgpu::wgpuAdapterGetInfo(adapter, &mut info),
            native::WGPUStatus_Success
        );
        assert_eq!(info.backendType, native::WGPUBackendType_Vulkan);
        assert!(!string_view_to_string(info.device).is_empty());
        yawgpu::wgpuAdapterInfoFreeMembers(info);

        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_request_device_get_queue_and_empty_submit() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        assert!(!device.is_null());

        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );

        let queue = yawgpu::wgpuDeviceGetQueue(device);
        assert!(!queue.is_null());
        yawgpu::wgpuQueueSubmit(queue, 0, std::ptr::null());
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn default_instance_still_selects_noop() {
    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        assert!(!instance.is_null());
        let adapter = request_adapter(instance);

        let mut info = zeroed_adapter_info();
        assert_eq!(
            yawgpu::wgpuAdapterGetInfo(adapter, &mut info),
            native::WGPUStatus_Success
        );
        assert_eq!(info.backendType, native::WGPUBackendType_Null);
        assert!(!string_view_to_string(info.device).is_empty());
        yawgpu::wgpuAdapterInfoFreeMembers(info);

        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[cfg(feature = "vulkan")]
unsafe fn create_vulkan_instance() -> native::WGPUInstance {
    let mut backend = WGPUYawgpuInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: WGPU_STYPE_YAWGPU_INSTANCE_BACKEND_SELECT,
        },
        backend: WGPU_YAWGPU_INSTANCE_BACKEND_VULKAN,
    };
    let descriptor = native::WGPUInstanceDescriptor {
        nextInChain: (&mut backend.chain) as *mut native::WGPUChainedStruct,
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
    };
    let instance = yawgpu::wgpuCreateInstance(&descriptor);
    assert!(!instance.is_null());
    instance
}

unsafe fn request_adapter(instance: native::WGPUInstance) -> native::WGPUAdapter {
    let mut adapter = std::ptr::null();
    let callback_info = native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_adapter_callback),
        userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info);
    wait(instance, future);
    adapter
}

#[cfg(feature = "vulkan")]
unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let mut device = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), callback_info);
    wait(instance, future);
    device
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

#[cfg(feature = "vulkan")]
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

fn zeroed_adapter_info() -> native::WGPUAdapterInfo {
    native::WGPUAdapterInfo {
        nextInChain: std::ptr::null_mut(),
        vendor: empty_string_view(),
        architecture: empty_string_view(),
        device: empty_string_view(),
        description: empty_string_view(),
        backendType: native::WGPUBackendType_Undefined,
        adapterType: native::WGPUAdapterType_Unknown,
        vendorID: 0,
        deviceID: 0,
        subgroupMinSize: 0,
        subgroupMaxSize: 0,
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

unsafe fn string_view_to_string(value: native::WGPUStringView) -> String {
    if value.data.is_null() {
        return String::new();
    }
    let bytes = std::slice::from_raw_parts(value.data.cast::<u8>(), value.length);
    String::from_utf8_lossy(bytes).into_owned()
}
