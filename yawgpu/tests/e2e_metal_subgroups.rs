//! Real-GPU verification of the WebGPU `subgroups` feature on Metal.
//!
//! Proves the feature end-to-end on hardware (not just Noop validation):
//! - the Metal adapter advertises `WGPUFeatureName_Subgroups` and reports a
//!   non-zero `WGPUAdapterInfo::subgroupMinSize` / `subgroupMaxSize`;
//! - a device that requested `subgroups` runs an `enable subgroups;` compute
//!   shader that reads the `@builtin(subgroup_size)` and performs an arithmetic
//!   subgroup op (`subgroupAdd`) — readback-verified (exercises MSL SIMD-group
//!   intrinsics + the subgroup builtins);
//! - a device that did NOT request the feature rejects a subgroups-using shader
//!   at module creation (the Block 62 gate fires on a real backend too).
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

/// One workgroup of this many invocations, all active. Chosen as a multiple of
/// every plausible subgroup width (4/8/16/32/64) so every subgroup is *full*
/// and `subgroupAdd(1u)` deterministically equals the runtime subgroup size.
const WG: usize = 64;

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_adapter_advertises_subgroups() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        assert!(
            yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_Subgroups) != 0,
            "Metal adapter must advertise subgroups"
        );
        let info = adapter_info(adapter);
        assert!(
            info.subgroupMinSize >= 1 && info.subgroupMaxSize >= info.subgroupMinSize,
            "subgroup size range must be sane, got {}..={}",
            info.subgroupMinSize,
            info.subgroupMaxSize
        );
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_subgroup_size_and_reduction() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let info = adapter_info(adapter);
        let device = request_device_with_subgroups(instance, adapter);
        assert!(
            yawgpu::wgpuDeviceHasFeature(device, native::WGPUFeatureName_Subgroups) != 0,
            "device must report subgroups after requesting it"
        );
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // sizes[i] = @builtin(subgroup_size); sums[i] = subgroupAdd(1u). For a
        // fully-active workgroup whose size is a multiple of the subgroup width,
        // every subgroup is full so subgroupAdd(1u) == subgroup_size for all i.
        let shader = r#"
enable subgroups;
struct Out { sizes: array<u32, 64>, sums: array<u32, 64> }
@group(0) @binding(0) var<storage, read_write> out: Out;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32,
        @builtin(subgroup_size) sg_size: u32) {
    out.sizes[li] = sg_size;
    out.sums[li] = subgroupAdd(1u);
}
"#;
        let out_size = (WG * 2 * 4) as u64; // sizes[64] + sums[64], u32 each
        let output = create_buffer_sized(
            device,
            out_size,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let readback = create_buffer_sized(
            device,
            out_size,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let module = create_wgsl_module(device, shader);
        let pipeline = create_compute_pipeline(device, module);
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let bind_group = create_single_bind_group(device, layout, output, out_size);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, out_size);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let bytes = read_buffer(instance, readback, out_size as usize);
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let sizes = &words[0..WG];
        let sums = &words[WG..2 * WG];

        let s = sizes[0];
        assert!(s >= 1, "runtime subgroup size must be >= 1, got {s}");
        assert!(
            s >= info.subgroupMinSize && s <= info.subgroupMaxSize,
            "runtime subgroup size {s} must fall in adapter range {}..={}",
            info.subgroupMinSize,
            info.subgroupMaxSize
        );
        assert!(
            sizes.iter().all(|&v| v == s),
            "subgroup_size must be uniform across the workgroup: {sizes:?}"
        );
        assert!(
            (WG as u32).is_multiple_of(s),
            "test assumes a full-subgroup workgroup; WG {WG} not a multiple of subgroup size {s}"
        );
        assert!(
            sums.iter().all(|&v| v == s),
            "subgroupAdd(1u) must equal the (full) subgroup size {s}: {sums:?}"
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "subgroup compute path raised a device error"
        );

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(output);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_subgroups_shader_rejected_without_feature() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        // Device requests NO features → subgroups not enabled.
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let shader = "enable subgroups;\n@compute @workgroup_size(1)\nfn cs() { let x = subgroupAdd(1u); _ = x; }";
        let module = create_wgsl_module_allow_error(device, shader);

        // Without the subgroups feature the Block 62 gate rejects the shader: the
        // create routes a validation error to the device error sink and returns
        // an error shader-module handle (non-null, Release-safe).
        assert!(!module.is_null());
        assert!(
            !errors.lock().expect("error lock").is_empty(),
            "rejected subgroups shader must raise a validation device error"
        );

        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// ---- helpers ----

#[cfg(feature = "metal")]
unsafe fn adapter_info(adapter: native::WGPUAdapter) -> native::WGPUAdapterInfo {
    let mut info: native::WGPUAdapterInfo = std::mem::zeroed();
    assert_eq!(
        yawgpu::wgpuAdapterGetInfo(adapter, &mut info),
        native::WGPUStatus_Success
    );
    info
}

#[cfg(feature = "metal")]
unsafe fn create_buffer_sized(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

#[cfg(feature = "metal")]
unsafe fn create_compute_pipeline(
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
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
unsafe fn create_single_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    buffer: native::WGPUBuffer,
    size: u64,
) -> native::WGPUBindGroup {
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer,
        offset: 0,
        size,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    };
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: 1,
        entries: &entry,
    };
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!bind_group.is_null());
    bind_group
}

#[cfg(feature = "metal")]
unsafe fn read_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    len: usize,
) -> Vec<u8> {
    let mut status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, len, callback_info);
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, len);
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

unsafe fn create_wgsl_module(device: native::WGPUDevice, source: &str) -> native::WGPUShaderModule {
    let module = create_wgsl_module_allow_error(device, source);
    assert!(!module.is_null());
    module
}

unsafe fn create_wgsl_module_allow_error(
    device: native::WGPUDevice,
    source: &str,
) -> native::WGPUShaderModule {
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

#[cfg(feature = "metal")]
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
    request_device_inner(instance, adapter, std::ptr::null())
}

#[cfg(feature = "metal")]
unsafe fn request_device_with_subgroups(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let features = [native::WGPUFeatureName_Subgroups];
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        requiredFeatureCount: features.len(),
        requiredFeatures: features.as_ptr(),
        requiredLimits: std::ptr::null(),
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        },
        deviceLostCallbackInfo: std::mem::zeroed(),
        uncapturedErrorCallbackInfo: std::mem::zeroed(),
    };
    request_device_inner(instance, adapter, &descriptor)
}

unsafe fn request_device_inner(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    descriptor: *const native::WGPUDeviceDescriptor,
) -> native::WGPUDevice {
    let mut device: native::WGPUDevice = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, descriptor, callback_info);
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

#[cfg(feature = "metal")]
unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
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
