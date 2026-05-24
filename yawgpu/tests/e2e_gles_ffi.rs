#![cfg(feature = "gles")]

use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{real_backend_available, wait, RealBackend};

#[derive(Default)]
struct RequestAdapterState {
    fired: u32,
    status: native::WGPURequestAdapterStatus,
    adapter: native::WGPUAdapter,
}

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn gles_ffi_request_adapter_returns_gles_when_backend_opengles_requested() {
    if !real_backend_available(RealBackend::Gles) {
        eprintln!("skip: no GLES adapter");
        return;
    }

    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        let adapter = request_adapter_with_backend(instance, native::WGPUBackendType_OpenGLES);
        assert!(!adapter.is_null());

        let mut info = zeroed_adapter_info();
        assert_eq!(
            yawgpu::wgpuAdapterGetInfo(adapter, &mut info),
            native::WGPUStatus_Success
        );
        assert_eq!(info.backendType, native::WGPUBackendType_OpenGLES);
        yawgpu::wgpuAdapterInfoFreeMembers(info);

        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn gles_ffi_request_adapter_returns_noop_when_backend_undefined() {
    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        let adapter = request_adapter(instance, std::ptr::null());
        assert!(!adapter.is_null());

        let mut info = zeroed_adapter_info();
        assert_eq!(
            yawgpu::wgpuAdapterGetInfo(adapter, &mut info),
            native::WGPUStatus_Success
        );
        assert_eq!(info.backendType, native::WGPUBackendType_Null);
        yawgpu::wgpuAdapterInfoFreeMembers(info);

        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn gles_ffi_request_adapter_errors_when_gles_unavailable() {
    if real_backend_available(RealBackend::Gles) {
        eprintln!("skip: GLES is available; cannot exercise unavailable path");
        return;
    }

    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        let options = request_adapter_options(native::WGPUBackendType_OpenGLES);
        let mut state = RequestAdapterState::default();
        let callback_info = request_adapter_callback_info(&mut state);
        let future = yawgpu::wgpuInstanceRequestAdapter(instance, &options, callback_info);
        wait(instance, future);

        assert_eq!(state.fired, 1);
        assert_eq!(state.status, native::WGPURequestAdapterStatus_Unavailable);
        assert!(state.adapter.is_null());

        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn request_adapter_with_backend(
    instance: native::WGPUInstance,
    backend_type: native::WGPUBackendType,
) -> native::WGPUAdapter {
    let options = request_adapter_options(backend_type);
    request_adapter(instance, &options)
}

unsafe fn request_adapter(
    instance: native::WGPUInstance,
    options: *const native::WGPURequestAdapterOptions,
) -> native::WGPUAdapter {
    let mut state = RequestAdapterState::default();
    let callback_info = request_adapter_callback_info(&mut state);
    let future = yawgpu::wgpuInstanceRequestAdapter(instance, options, callback_info);
    wait(instance, future);
    assert_eq!(state.fired, 1);
    assert_eq!(state.status, native::WGPURequestAdapterStatus_Success);
    state.adapter
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

fn request_adapter_callback_info(
    state: &mut RequestAdapterState,
) -> native::WGPURequestAdapterCallbackInfo {
    native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_adapter_callback),
        userdata1: (state as *mut RequestAdapterState).cast(),
        userdata2: std::ptr::null_mut(),
    }
}

unsafe extern "C" fn request_adapter_callback(
    status: native::WGPURequestAdapterStatus,
    adapter: native::WGPUAdapter,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let state = &mut *(userdata1 as *mut RequestAdapterState);
    state.fired += 1;
    state.status = status;
    state.adapter = adapter;
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
