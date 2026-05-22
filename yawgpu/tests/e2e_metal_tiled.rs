#![cfg(all(feature = "metal", feature = "tiled"))]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::{
    native, YaWGPUInstanceBackendSelect, YaWGPUTiledCapabilities, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

#[test]
#[ignore = "manual real-backend test"]
fn metal_tiled_features_and_capabilities_are_advertised() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        assert!(!adapter.is_null());

        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(adapter, yawgpu::YaWGPUFeatureName_MultiSubpass),
            1
        );
        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(adapter, yawgpu::YaWGPUFeatureName_TransientAttachments,),
            1
        );
        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(
                adapter,
                yawgpu::YaWGPUFeatureName_ShaderFramebufferFetch,
            ),
            1
        );
        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(
                adapter,
                yawgpu::YaWGPUFeatureName_ProgrammableTileDispatch,
            ),
            1
        );

        let mut capabilities = zeroed_tiled_capabilities();
        assert_eq!(
            yawgpu::yawgpuAdapterGetTiledCapabilities(adapter, &mut capabilities),
            native::WGPUStatus_Success
        );
        assert!(capabilities.maxSubpasses > 0);
        assert!(capabilities.maxSubpassColorAttachments > 0);

        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_explicit_transient_attachment_allocates_without_device_error() {
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
        let descriptor = yawgpu::YaWGPUTransientAttachmentDescriptor {
            nextInChain: std::ptr::null(),
            label: native::WGPUStringView {
                data: std::ptr::null(),
                length: 0,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sizeMode: yawgpu::YaWGPUTransientSizeMode_Explicit,
            width: 16,
            height: 16,
            sampleCount: 1,
        };

        let attachment = yawgpu::yawgpuDeviceCreateTransientAttachment(device, &descriptor);
        assert!(!attachment.is_null());
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::yawgpuTransientAttachmentRelease(attachment);
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
    yawgpu::wgpuCreateInstance(&descriptor)
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

fn zeroed_tiled_capabilities() -> YaWGPUTiledCapabilities {
    YaWGPUTiledCapabilities {
        nextInChain: std::ptr::null(),
        maxSubpasses: 0,
        maxSubpassColorAttachments: 0,
        maxInputAttachments: 0,
        estimatedTileMemoryBytes: 0,
    }
}
