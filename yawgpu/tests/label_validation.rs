use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::wait;

#[test]
fn device_label_defaults_from_descriptor_and_set_label_updates_it() {
    unsafe {
        let fixture = Fixture::new();

        let device = request_device(fixture.instance, fixture.adapter, None, None);
        assert_eq!(yawgpu::testing_get_device_label(device), "");
        yawgpu::wgpuDeviceRelease(device);

        let device = request_device(fixture.instance, fixture.adapter, Some("dev"), None);
        assert_eq!(yawgpu::testing_get_device_label(device), "dev");

        yawgpu::wgpuDeviceSetLabel(device, string_view("dev2"));
        assert_eq!(yawgpu::testing_get_device_label(device), "dev2");

        yawgpu::wgpuDeviceRelease(device);
    }
}

#[test]
fn queue_label_defaults_from_descriptor_set_label_updates_shared_queue() {
    unsafe {
        let fixture = Fixture::new();

        let device = request_device(fixture.instance, fixture.adapter, None, None);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        assert_eq!(yawgpu::testing_get_queue_label(queue), "");
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);

        let device = request_device(fixture.instance, fixture.adapter, None, Some("q"));
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        assert_eq!(yawgpu::testing_get_queue_label(queue), "q");

        yawgpu::wgpuQueueSetLabel(queue, string_view("q2"));
        assert_eq!(yawgpu::testing_get_queue_label(queue), "q2");

        let second_queue = yawgpu::wgpuDeviceGetQueue(device);
        assert_eq!(yawgpu::testing_get_queue_label(second_queue), "q2");
        assert_eq!(queue, second_queue);

        yawgpu::wgpuQueueRelease(second_queue);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
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

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    device_label: Option<&str>,
    queue_label: Option<&str>,
) -> native::WGPUDevice {
    let mut device: native::WGPUDevice = std::ptr::null();
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: device_label
            .map(string_view)
            .unwrap_or_else(string_view_init),
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: queue_label
                .map(string_view)
                .unwrap_or_else(string_view_init),
        },
        deviceLostCallbackInfo: unsafe { std::mem::zeroed() },
        uncapturedErrorCallbackInfo: unsafe { std::mem::zeroed() },
    };
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, &descriptor, callback_info);
    wait(instance, future);
    assert!(!device.is_null());
    device
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

fn string_view_init() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
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
