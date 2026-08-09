//! Real-Metal surface e2e: configuring with the `webgpu.h` INIT zero
//! sentinels (`WGPUPresentMode_Undefined`, `WGPUCompositeAlphaMode_Auto`)
//! must succeed and acquire a texture from a standalone `CAMetalLayer`
//! (externally reported 2026-08-09; contract in
//! `specs/blocks/70-finalize.md` → Surface → Sentinel resolution).

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use objc2_quartz_core::CAMetalLayer;
use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

#[test]
#[ignore = "manual real-backend test"]
fn metal_surface_configure_with_init_sentinels_acquires_texture() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        assert!(!device.is_null());

        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );

        let layer = CAMetalLayer::layer();
        let surface = create_surface_from_layer(
            instance,
            (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>(),
        );

        // The repro's configuration: capabilities-reported format, render
        // attachment usage, nonzero size, and both modes left at the INIT
        // zero sentinels.
        let config = native::WGPUSurfaceConfiguration {
            nextInChain: std::ptr::null_mut(),
            device,
            format: native::WGPUTextureFormat_BGRA8Unorm,
            usage: native::WGPUTextureUsage_RenderAttachment,
            width: 64,
            height: 64,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
            alphaMode: native::WGPUCompositeAlphaMode_Auto,
            presentMode: native::WGPUPresentMode_Undefined,
        };
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected errors: {:?}",
            errors.lock().expect("error lock")
        );

        let mut surface_texture = native::WGPUSurfaceTexture {
            nextInChain: std::ptr::null_mut(),
            texture: std::ptr::null(),
            status: 0,
        };
        yawgpu::wgpuSurfaceGetCurrentTexture(surface, &mut surface_texture);
        assert_eq!(
            surface_texture.status,
            native::WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal
        );
        assert!(!surface_texture.texture.is_null());
        assert_eq!(
            yawgpu::wgpuSurfacePresent(surface),
            native::WGPUStatus_Success
        );
        yawgpu::wgpuTextureRelease(surface_texture.texture.cast_mut());

        yawgpu::wgpuSurfaceUnconfigure(surface);
        yawgpu::wgpuSurfaceRelease(surface);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_surface_configure_still_rejects_explicit_unsupported_modes() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);

        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );

        let layer = CAMetalLayer::layer();
        let surface = create_surface_from_layer(
            instance,
            (&*layer as *const CAMetalLayer).cast_mut().cast::<c_void>(),
        );

        let mut config = native::WGPUSurfaceConfiguration {
            nextInChain: std::ptr::null_mut(),
            device,
            format: native::WGPUTextureFormat_BGRA8Unorm,
            usage: native::WGPUTextureUsage_RenderAttachment,
            width: 64,
            height: 64,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
            alphaMode: native::WGPUCompositeAlphaMode_Opaque,
            presentMode: native::WGPUPresentMode_Immediate,
        };
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        assert_eq!(errors.lock().expect("error lock").len(), 1);

        config.presentMode = native::WGPUPresentMode_Fifo;
        config.alphaMode = native::WGPUCompositeAlphaMode_Premultiplied;
        yawgpu::wgpuSurfaceConfigure(surface, &config);
        assert_eq!(errors.lock().expect("error lock").len(), 2);

        let mut surface_texture = native::WGPUSurfaceTexture {
            nextInChain: std::ptr::null_mut(),
            texture: std::ptr::null(),
            status: 0,
        };
        yawgpu::wgpuSurfaceGetCurrentTexture(surface, &mut surface_texture);
        assert_eq!(
            surface_texture.status,
            native::WGPUSurfaceGetCurrentTextureStatus_Error
        );
        assert!(surface_texture.texture.is_null());

        yawgpu::wgpuSurfaceRelease(surface);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn create_surface_from_layer(
    instance: native::WGPUInstance,
    layer: *mut c_void,
) -> native::WGPUSurface {
    let mut source = native::WGPUSurfaceSourceMetalLayer {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_SurfaceSourceMetalLayer,
        },
        layer,
    };
    let descriptor = native::WGPUSurfaceDescriptor {
        nextInChain: (&mut source.chain) as *mut _,
        label: empty_string_view(),
    };
    let surface = yawgpu::wgpuInstanceCreateSurface(instance, &descriptor);
    assert!(!surface.is_null());
    surface
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
