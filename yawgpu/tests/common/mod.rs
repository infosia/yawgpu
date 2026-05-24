#![cfg(feature = "gles")]

use yawgpu::{
    native, YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_GLES,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};

pub unsafe fn create_gles_instance() -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: YAWGPU_INSTANCE_BACKEND_GLES,
    };
    let descriptor = native::WGPUInstanceDescriptor {
        nextInChain: (&mut backend.chain) as *mut native::WGPUChainedStruct,
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
    };
    let instance = unsafe { yawgpu::wgpuCreateInstance(&descriptor) };
    assert!(!instance.is_null());
    instance
}
