//! Real-Metal regression for F-065 out-of-memory classification.
//!
//! A texture descriptor that is *valid* WebGPU (every dimension within the
//! device's advertised limits) but whose backing allocation the GPU cannot
//! satisfy must surface as a **`GPUOutOfMemoryError`** (`ErrorKind::OutOfMemory`),
//! never as a validation error and never as a panic. yawgpu previously emitted
//! only `Validation`/`Internal` errors and never `OutOfMemory`.
//!
//! The descriptor used here is a 4096 x 4096 x 256 `rgba32float` 2D-array
//! texture (= 64 GiB). 4096 == `maxTextureDimension2D` and 256 ==
//! `maxTextureArrayLayers` (the yawgpu default limits, which the Metal adapter
//! also advertises), so core validation accepts it and it reaches the Metal
//! HAL, where `newTextureWithDescriptor` returns nil → mapped to
//! `HalError::OutOfMemory` → `ErrorKind::OutOfMemory`.

#![cfg(feature = "metal")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
#[cfg(feature = "metal")]
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::wait;
#[cfg(feature = "metal")]
use yawgpu_test::{real_backend_skip_reason, RealBackend};

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_oversized_texture_reports_out_of_memory_without_panic() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        // Valid WebGPU descriptor (all dims within the advertised limits), but
        // a 64 GiB allocation no GPU can satisfy. Must NOT raise a validation
        // error; the Metal allocation failure is routed as GPUOutOfMemoryError.
        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_CopyDst,
            dimension: native::WGPUTextureDimension_2D,
            size: native::WGPUExtent3D {
                width: 4096,
                height: 4096,
                depthOrArrayLayers: 256,
            },
            format: native::WGPUTextureFormat_RGBA32Float,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };

        // Reaching past this call at all proves the Metal HAL did not panic on
        // the nil texture (it returns Err(OutOfMemory) instead).
        let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);

        let captured = errors.lock().expect("error lock");
        assert!(
            captured
                .iter()
                .any(|e| e.kind == yawgpu_core::ErrorKind::OutOfMemory),
            "expected a GPUOutOfMemoryError for the oversized Metal texture, got: {captured:?}"
        );
        assert!(
            captured
                .iter()
                .all(|e| e.kind != yawgpu_core::ErrorKind::Validation),
            "oversized-but-valid descriptor must not raise a validation error: {captured:?}"
        );
        drop(captured);

        if !texture.is_null() {
            yawgpu::wgpuTextureRelease(texture);
        }
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn create_metal_instance() -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: YAWGPU_INSTANCE_BACKEND_METAL,
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

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
