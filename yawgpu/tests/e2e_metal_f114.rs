//! F-114 repro: `textureSampleGrad` on 3D / cube textures errors on Metal.
//!
//! Creates a compute pipeline whose shader does `textureSampleGrad` against a
//! 2D (baseline), 3D, and cube texture, and prints the first device error
//! captured for each. 2D is expected to compile; 3D/cube are the suspected
//! failures. Manual real-backend diagnostic, not a regression test.
//!
//! Run:
//!   cargo test -p yawgpu --features metal --test e2e_metal_f114 \
//!     f114_texture_sample_grad_dims -- --ignored --exact --nocapture

#![cfg(feature = "metal")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, RealBackend};

fn grad_shader(tex_type: &str, coord: &str) -> String {
    format!(
        "@group(0) @binding(0) var t: {tex_type};\n\
         @group(0) @binding(1) var s: sampler;\n\
         @group(0) @binding(2) var<storage, read_write> out: vec4<f32>;\n\
         @compute @workgroup_size(1)\n\
         fn main() {{\n\
         \x20 out = textureSampleGrad(t, s, {coord}, {coord}, {coord});\n\
         }}\n"
    )
}

#[test]
#[ignore = "manual real-backend F-114 repro"]
fn f114_texture_sample_grad_dims() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        eprintln!("f114: skipped (no real Metal device)");
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        for (label, tex_type, coord) in [
            ("2d  ", "texture_2d<f32>", "vec2<f32>(0.5)"),
            ("3d  ", "texture_3d<f32>", "vec3<f32>(0.5)"),
            ("cube", "texture_cube<f32>", "vec3<f32>(0.5)"),
        ] {
            errors.lock().expect("lock").clear();
            let source = grad_shader(tex_type, coord);
            let module = create_wgsl_module(device, &source);
            let pipeline = make_compute_pipeline(device, module);
            let errs = errors.lock().expect("lock");
            eprintln!(
                "F-114 [{label}] {tex_type}: pipeline_null={} errors={}",
                pipeline.is_null(),
                errs.len()
            );
            for e in errs.iter() {
                eprintln!("    error: {e:?}");
            }
            drop(errs);
            if !pipeline.is_null() {
                yawgpu::wgpuComputePipelineRelease(pipeline);
            }
            yawgpu::wgpuShaderModuleRelease(module);
        }

        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn make_compute_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPUComputePipeline {
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("main"),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor)
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
    yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor)
}

unsafe fn install_error_capture(
    device: native::WGPUDevice,
) -> Arc<Mutex<Vec<yawgpu_core::DeviceError>>> {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&errors);
    yawgpu::testing_set_uncaptured_error_callback(
        device,
        Some(move |error| captured.lock().expect("lock").push(error)),
    );
    errors
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
    yawgpu_test::wait(instance, future);
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
    yawgpu_test::wait(instance, future);
    assert!(!device.is_null());
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

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}
