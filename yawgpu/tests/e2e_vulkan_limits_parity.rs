//! Real-Vulkan e2e for Block 134 (`specs/blocks/134-vulkan-limits-parity.md`).
//!
//! Vulkan has no `maxColorAttachmentBytesPerSample`; Dawn reports
//! `32 * maxColorAttachments` so the limit never binds (R2). A render pipeline with
//! four `rgba32float` targets (64 bytes per sample) must be created without a device
//! error, and the adapter must report the Block 134 values.

#![cfg(feature = "vulkan")]

#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;

use std::os::raw::c_void;

use indirect_dispatch::{
    create_instance, create_wgsl_module, empty_string_view, install_error_capture, request_adapter,
    string_view,
};
use yawgpu::native;
use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const TARGETS: usize = 4;

const SHADER: &str = r#"
struct Out {
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
    @location(2) c: vec4<f32>,
    @location(3) d: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

@fragment
fn fs() -> Out {
    return Out(vec4<f32>(1.0), vec4<f32>(2.0), vec4<f32>(3.0), vec4<f32>(4.0));
}
"#;

#[test]
#[ignore = "manual real-backend test"]
fn four_rgba32float_targets_are_accepted() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_instance(YAWGPU_INSTANCE_BACKEND_VULKAN);
        let adapter = request_adapter(instance);

        let mut adapter_limits: native::WGPULimits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuAdapterGetLimits(adapter, &mut adapter_limits),
            native::WGPUStatus_Success
        );
        eprintln!(
            "adapter maxColorAttachments={}, maxColorAttachmentBytesPerSample={}, maxStorageBufferBindingSize={}",
            adapter_limits.maxColorAttachments,
            adapter_limits.maxColorAttachmentBytesPerSample,
            adapter_limits.maxStorageBufferBindingSize,
        );
        assert!(adapter_limits.maxColorAttachmentBytesPerSample >= 64);

        // R1: on NVIDIA the storage binding size stays below 2 GiB, where some drivers'
        // `OpArrayLength` returns 0 (crbug.com/435684920).
        let mut info: native::WGPUAdapterInfo = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuAdapterGetInfo(adapter, &mut info),
            native::WGPUStatus_Success
        );
        eprintln!("adapter vendorID={:#x}", info.vendorID);
        if info.vendorID == 0x10de {
            assert!(adapter_limits.maxStorageBufferBindingSize <= (1 << 31) - 4);
        }
        // R1, every vendor: within maxBufferSize and a multiple of 4.
        assert!(adapter_limits.maxStorageBufferBindingSize <= adapter_limits.maxBufferSize);
        assert_eq!(adapter_limits.maxStorageBufferBindingSize % 4, 0);
        yawgpu::wgpuAdapterInfoFreeMembers(info);

        // Request every adapter limit, so the device validates against the adapter's
        // `maxColorAttachmentBytesPerSample` rather than the WebGPU default of 32.
        let device = request_device_with_limits(instance, adapter, &adapter_limits);
        let errors = install_error_capture(device);

        let mut device_limits: native::WGPULimits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(device, &mut device_limits),
            native::WGPUStatus_Success
        );
        eprintln!(
            "device maxColorAttachmentBytesPerSample={}",
            device_limits.maxColorAttachmentBytesPerSample
        );
        assert_eq!(
            device_limits.maxColorAttachmentBytesPerSample,
            adapter_limits.maxColorAttachmentBytesPerSample
        );

        let pipeline = create_four_target_pipeline(device);
        assert!(!pipeline.is_null());
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuDeviceRelease(device);

        // Control: a device with the WebGPU default limits (32 bytes per sample) rejects
        // the same pipeline, so the acceptance above is due to the requested limit.
        // An adapter is consumed by its first device request, so take a fresh one.
        let default_adapter = request_adapter(instance);
        let default_device =
            request_device_with_limits_ptr(instance, default_adapter, std::ptr::null());
        let default_errors = install_error_capture(default_device);
        let rejected = create_four_target_pipeline(default_device);
        let messages = default_errors.lock().expect("error lock").clone();
        eprintln!("default-limits device errors: {messages:?}");
        assert_eq!(messages.len(), 1, "{messages:?}");
        if !rejected.is_null() {
            yawgpu::wgpuRenderPipelineRelease(rejected);
        }
        yawgpu::wgpuDeviceRelease(default_device);
        yawgpu::wgpuAdapterRelease(default_adapter);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn create_four_target_pipeline(device: native::WGPUDevice) -> native::WGPURenderPipeline {
    let module = create_wgsl_module(device, SHADER);
    let target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA32Float,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let targets = [target; TARGETS];
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: TARGETS,
        targets: targets.as_ptr(),
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(
        device,
        &native::WGPURenderPipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null_mut(),
            vertex: native::WGPUVertexState {
                nextInChain: std::ptr::null_mut(),
                module,
                entryPoint: string_view("vs"),
                constantCount: 0,
                constants: std::ptr::null(),
                bufferCount: 0,
                buffers: std::ptr::null(),
            },
            primitive: native::WGPUPrimitiveState {
                nextInChain: std::ptr::null_mut(),
                topology: native::WGPUPrimitiveTopology_TriangleList,
                stripIndexFormat: native::WGPUIndexFormat_Undefined,
                frontFace: native::WGPUFrontFace_CCW,
                cullMode: native::WGPUCullMode_None,
                unclippedDepth: 0,
            },
            depthStencil: std::ptr::null(),
            multisample: native::WGPUMultisampleState {
                nextInChain: std::ptr::null_mut(),
                count: 1,
                mask: u32::MAX,
                alphaToCoverageEnabled: 0,
            },
            fragment: &fragment,
        },
    );
    yawgpu::wgpuShaderModuleRelease(module);
    pipeline
}

unsafe fn request_device_with_limits(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    limits: &native::WGPULimits,
) -> native::WGPUDevice {
    request_device_with_limits_ptr(instance, adapter, limits)
}

unsafe fn request_device_with_limits_ptr(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    limits: *const native::WGPULimits,
) -> native::WGPUDevice {
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: limits,
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        },
        deviceLostCallbackInfo: std::mem::zeroed(),
        uncapturedErrorCallbackInfo: std::mem::zeroed(),
    };
    let mut device: native::WGPUDevice = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, &descriptor, callback_info);
    wait(instance, future);
    assert!(
        !device.is_null(),
        "device request with the adapter's limits failed"
    );
    device
}

unsafe extern "C" fn request_device_callback(
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    if status == native::WGPURequestDeviceStatus_Success {
        *(userdata1 as *mut native::WGPUDevice) = device;
    }
}
