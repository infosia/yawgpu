use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{assert_device_error, wait, ValidationTest};

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
fn create_instance_returns_non_null_handle() {
    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        assert!(!instance.is_null());
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
fn noop_adapter_device_queue_round_trip() {
    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        assert!(!instance.is_null());

        let mut adapter: native::WGPUAdapter = std::ptr::null();
        let adapter_callback_info = native::WGPURequestAdapterCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(request_adapter_callback),
            userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future =
            yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), adapter_callback_info);
        wait(instance, future);
        assert!(!adapter.is_null());

        let mut device: native::WGPUDevice = std::ptr::null();
        let device_callback_info = native::WGPURequestDeviceCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(request_device_callback),
            userdata1: (&mut device as *mut native::WGPUDevice).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future =
            yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), device_callback_info);
        wait(instance, future);
        assert!(!device.is_null());

        let queue = yawgpu::wgpuDeviceGetQueue(device);
        assert!(!queue.is_null());

        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
fn assert_device_error_catches_injected_error() {
    let test = ValidationTest::new();
    assert_device_error!(
        test.inject_device_error("injected validation error"),
        "injected"
    );
}

#[test]
#[should_panic(expected = "expected exactly one device error")]
fn assert_device_error_panics_when_no_error_occurs() {
    let _test = ValidationTest::new();
    assert_device_error!({});
}
