#![cfg(all(feature = "vulkan", feature = "tiled"))]

use std::os::raw::c_void;

use yawgpu::{
    native, YaWGPUInstanceBackendSelect, YaWGPUTiledCapabilities, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_tiled_features_and_capabilities_are_advertised() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
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

unsafe fn create_vulkan_instance() -> native::WGPUInstance {
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

fn zeroed_tiled_capabilities() -> YaWGPUTiledCapabilities {
    YaWGPUTiledCapabilities {
        nextInChain: std::ptr::null(),
        maxSubpasses: 0,
        maxSubpassColorAttachments: 0,
        maxInputAttachments: 0,
        estimatedTileMemoryBytes: 0,
    }
}
