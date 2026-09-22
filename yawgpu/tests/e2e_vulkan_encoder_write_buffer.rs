//! Real-Vulkan e2e for Block 100 (`specs/blocks/100-encoder-write-buffer.md`):
//! `wgpuCommandEncoderWriteBuffer` actually writes on a real backend, in
//! command order, including the oversized (> 64 KiB) staging path.

#![cfg(feature = "vulkan")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const BACKEND: RealBackend = RealBackend::Vulkan;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_encoder_write_buffer_writes_bytes_at_offset_and_leaves_the_rest_zero() {
    if real_backend_skip_reason(BACKEND).is_some() {
        return;
    }
    unsafe {
        let session = Session::new();
        let buffer = create_buffer(
            session.device,
            32,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let mut data: Vec<u8> = (1..=16).collect();

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(session.device, std::ptr::null());
        yawgpu::wgpuCommandEncoderWriteBuffer(encoder, buffer, 4, data.as_ptr().cast(), 16);
        // The bytes were snapshotted at encode time (R1): mutating the source
        // afterwards must not change what lands in the buffer.
        data.fill(0xEE);
        session.submit(encoder);

        let bytes = read_buffer(session.instance, buffer, 0, 32);
        let mut expected = vec![0u8; 32];
        expected[4..20].copy_from_slice(&(1..=16).collect::<Vec<u8>>());
        assert_eq!(bytes, expected);
        session.assert_no_errors();

        yawgpu::wgpuBufferRelease(buffer);
        session.release();
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_encoder_write_buffer_then_copy_in_one_encoder_orders_the_write_first() {
    if real_backend_skip_reason(BACKEND).is_some() {
        return;
    }
    unsafe {
        let session = Session::new();
        let a = create_buffer(
            session.device,
            16,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
        );
        let b = create_buffer(
            session.device,
            16,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let data: Vec<u8> = (100..116).collect();

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(session.device, std::ptr::null());
        yawgpu::wgpuCommandEncoderWriteBuffer(encoder, a, 0, data.as_ptr().cast(), 16);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, a, 0, b, 0, 16);
        session.submit(encoder);

        assert_eq!(read_buffer(session.instance, b, 0, 16), data);
        session.assert_no_errors();

        yawgpu::wgpuBufferRelease(a);
        yawgpu::wgpuBufferRelease(b);
        session.release();
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_encoder_write_buffer_larger_than_a_staging_chunk_round_trips() {
    if real_backend_skip_reason(BACKEND).is_some() {
        return;
    }
    // 100 KiB: larger than the 64 KiB standard staging chunk, so this takes
    // the dedicated oversized-staging path.
    const SIZE: usize = 100 * 1024;
    unsafe {
        let session = Session::new();
        let buffer = create_buffer(
            session.device,
            SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let data: Vec<u8> = (0..SIZE).map(|i| (i % 251) as u8).collect();

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(session.device, std::ptr::null());
        yawgpu::wgpuCommandEncoderWriteBuffer(encoder, buffer, 0, data.as_ptr().cast(), SIZE);
        session.submit(encoder);

        assert_eq!(read_buffer(session.instance, buffer, 0, SIZE), data);
        session.assert_no_errors();

        yawgpu::wgpuBufferRelease(buffer);
        session.release();
    }
}

struct Session {
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    errors: Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
}

impl Session {
    unsafe fn new() -> Self {
        let instance = create_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        Self {
            instance,
            adapter,
            device,
            queue,
            errors,
        }
    }

    unsafe fn submit(&self, encoder: native::WGPUCommandEncoder) {
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        assert!(!command_buffer.is_null());
        yawgpu::wgpuQueueSubmit(self.queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }

    fn assert_no_errors(&self) {
        let errors = self.errors.lock().expect("error lock");
        assert!(errors.is_empty(), "unexpected device errors: {errors:?}");
    }

    unsafe fn release(self) {
        yawgpu::wgpuQueueRelease(self.queue);
        yawgpu::wgpuDeviceRelease(self.device);
        yawgpu::wgpuAdapterRelease(self.adapter);
        yawgpu::wgpuInstanceRelease(self.instance);
    }
}

unsafe fn read_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    offset: usize,
    len: usize,
) -> Vec<u8> {
    let mut status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, offset, len, callback_info);
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, offset, len);
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

unsafe fn create_instance() -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: YAWGPU_INSTANCE_BACKEND_VULKAN,
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
    adapter
}

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let mut device: native::WGPUDevice = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), callback_info);
    wait(instance, future);
    assert!(!device.is_null());
    device
}

unsafe fn install_error_capture(
    device: native::WGPUDevice,
) -> Arc<Mutex<Vec<yawgpu_core::DeviceError>>> {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let captured_errors = Arc::clone(&errors);
    yawgpu::testing_set_uncaptured_error_callback(
        device,
        Some(move |error| captured_errors.lock().expect("error lock").push(error)),
    );
    errors
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

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
