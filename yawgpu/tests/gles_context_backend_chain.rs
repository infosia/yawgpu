#[cfg(all(feature = "gles", not(windows)))]
use yawgpu::YAWGPU_GLES_CONTEXT_BACKEND_WGL;
#[cfg(feature = "gles")]
use yawgpu::{
    native, YaWGPUGlesContextBackend, YaWGPUInstanceBackendSelect,
    YAWGPU_GLES_CONTEXT_BACKEND_DEFAULT, YAWGPU_GLES_CONTEXT_BACKEND_EGL,
    YAWGPU_INSTANCE_BACKEND_GLES, YAWGPU_STYPE_GLES_CONTEXT_BACKEND,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};

#[test]
fn gles_context_backend_chain_creation_smoke() {
    #[cfg(not(feature = "gles"))]
    {
        eprintln!(
            "skipping GLES context backend chain test; yawgpu was built without feature=gles"
        );
    }

    #[cfg(feature = "gles")]
    unsafe {
        if !yawgpu_test::real_backend_available(yawgpu_test::RealBackend::Gles) {
            eprintln!("skipping GLES context backend chain test; no GLES adapter is available");
            return;
        }

        create_and_release_gles_instance(YAWGPU_GLES_CONTEXT_BACKEND_DEFAULT);
        create_and_release_gles_instance(YAWGPU_GLES_CONTEXT_BACKEND_EGL);

        #[cfg(not(windows))]
        create_and_release_gles_instance(YAWGPU_GLES_CONTEXT_BACKEND_WGL);
    }
}

#[cfg(feature = "gles")]
unsafe fn create_and_release_gles_instance(context_backend: u32) {
    let mut context = YaWGPUGlesContextBackend {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_GLES_CONTEXT_BACKEND,
        },
        contextBackend: context_backend,
    };
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: (&mut context.chain) as *mut native::WGPUChainedStruct,
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

    let instance = yawgpu::wgpuCreateInstance(&descriptor);
    assert!(!instance.is_null());
    yawgpu::wgpuInstanceRelease(instance);
}
