//! Real-Vulkan regression for F-060 external textures.
//!
//! yawgpu matches wgpu's posture: external textures are implemented on Metal
//! only. The Vulkan backend has no SPIR-V lowering for `texture_external`
//! (neither does naga's SPIR-V backend nor wgpu-hal/vulkan), so a render
//! pipeline whose shader samples an external texture cannot be compiled.
//!
//! The contract verified here is the honest one the user requested: the
//! descriptor is *valid* WebGPU (no validation error — `success=true` in the
//! CTS), but the Vulkan backend rejects code generation with a **clean
//! `GPUInternalError`**, never a panic. This is strictly better than wgpu,
//! which `unimplemented!()`-panics on the same path.

#![cfg(feature = "vulkan")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
#[cfg(feature = "vulkan")]
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::wait;
#[cfg(feature = "vulkan")]
use yawgpu_test::{real_backend_skip_reason, RealBackend};

// Verbatim upstream WGSL from the CTS external_texture test
// (api,validation,render_pipeline,misc:external_texture): the fragment entry
// samples a `texture_external`, so naga emits an `ImageClass::External` image
// op that the SPIR-V backend cannot lower.
const EXTERNAL_TEXTURE_SHADER: &str = "@vertex\n\
fn vertexMain() -> @builtin(position) vec4f {\n\
  return vec4f(1);\n\
}\n\
\n\
@group(0) @binding(0) var myTexture: texture_external;\n\
\n\
@fragment\n\
fn fragmentMain() -> @location(0) vec4f {\n\
  let result = textureLoad(myTexture, vec2u(1, 1));\n\
  return vec4f(1);\n\
}\n";

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_external_texture_pipeline_reports_internal_error_without_panic() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let module = create_wgsl_module(device, EXTERNAL_TEXTURE_SHADER);

        // layout: 'auto' — the bind group layout is reflected from the shader,
        // mirroring the CTS test. Core validation accepts the external-texture
        // binding (the descriptor is valid WebGPU), so this must NOT raise a
        // validation error. The Vulkan backend then fails SPIR-V code
        // generation and routes a GPUInternalError instead of panicking.
        let color_target = native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_RGBA8Unorm,
            blend: std::ptr::null(),
            writeMask: native::WGPUColorWriteMask_All,
        };
        let fragment = native::WGPUFragmentState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("fragmentMain"),
            constantCount: 0,
            constants: std::ptr::null(),
            targetCount: 1,
            targets: &color_target,
        };
        let descriptor = native::WGPURenderPipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null(),
            vertex: native::WGPUVertexState {
                nextInChain: std::ptr::null_mut(),
                module,
                entryPoint: string_view("vertexMain"),
                constantCount: 0,
                constants: std::ptr::null(),
                bufferCount: 0,
                buffers: std::ptr::null(),
            },
            primitive: primitive_state(),
            depthStencil: std::ptr::null(),
            multisample: multisample_state(),
            fragment: &fragment,
        };

        // Reaching this line at all proves there was no panic in naga's SPIR-V
        // backend (which `unimplemented!()`s on `ImageClass::External`).
        let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);

        let captured = errors.lock().expect("error lock");
        let internal = captured
            .iter()
            .find(|e| e.kind == yawgpu_core::ErrorKind::Internal)
            .unwrap_or_else(|| {
                panic!("expected a GPUInternalError for the Vulkan external-texture pipeline, got: {captured:?}")
            });
        assert!(
            internal.message.contains("external textures are not supported on the Vulkan backend"),
            "unexpected internal error message: {}",
            internal.message
        );
        // No *validation* error should be raised — the descriptor is valid WebGPU.
        assert!(
            captured
                .iter()
                .all(|e| e.kind != yawgpu_core::ErrorKind::Validation),
            "unexpected validation error: {captured:?}"
        );
        drop(captured);

        if !pipeline.is_null() {
            yawgpu::wgpuRenderPipelineRelease(pipeline);
        }
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

fn primitive_state() -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        nextInChain: std::ptr::null_mut(),
        topology: native::WGPUPrimitiveTopology_TriangleList,
        stripIndexFormat: native::WGPUIndexFormat_Undefined,
        frontFace: native::WGPUFrontFace_Undefined,
        cullMode: native::WGPUCullMode_Undefined,
        unclippedDepth: 0,
    }
}

fn multisample_state() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 1,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
}

unsafe fn create_wgsl_module(device: native::WGPUDevice, source: &str) -> native::WGPUShaderModule {
    let mut wgsl = native::WGPUShaderSourceWGSL {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ShaderSourceWGSL,
        },
        code: string_view(source),
    };
    let descriptor = native::WGPUShaderModuleDescriptor {
        nextInChain: (&mut wgsl.chain) as *mut _,
        label: empty_string_view(),
    };
    let module = yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor);
    assert!(!module.is_null());
    module
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

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
