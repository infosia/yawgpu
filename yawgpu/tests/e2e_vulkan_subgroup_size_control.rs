//! Real-GPU verification of the WebGPU `subgroup-size-control` feature on
//! Vulkan (Block 108).
//!
//! Proves on hardware (not just Noop validation):
//! - the Vulkan adapter advertises `WGPUFeatureName_SubgroupSizeControl` when
//!   the driver exposes `VK_EXT_subgroup_size_control` with
//!   `subgroupSizeControl` + `computeFullSubgroups`, and a device requesting
//!   only that feature also reports `Subgroups` (the Dawn implication);
//! - for every power-of-two `S` in `[subgroupMinSize, subgroupMaxSize]`, a
//!   `@subgroup_size(S)` pipeline runs with exactly that width: every
//!   invocation reads `@builtin(subgroup_size) == S` and
//!   `subgroupAdd(1u) == S` (all invocations active, i.e. the
//!   `REQUIRE_FULL_SUBGROUPS` + `RequiredSubgroupSize` lowering took effect);
//! - the core pipeline rules (Block 108 R4) fire as validation errors on a
//!   real device: x not a multiple of `S`, `S` out of range, and a
//!   non-power-of-two override-driven `S`.
//!
//! Rule 3 (`maxComputeWorkgroupSubgroups`) is Noop-tested only: on the
//! verified hosts `maxComputeInvocationsPerWorkgroup / min_size` never exceeds
//! the driver's subgroup cap, so the rule is unreachable behind the
//! workgroup-size limit check.
//!
//! Run under `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; the expected
//! validation output is empty.
#![cfg(feature = "vulkan")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::wait;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

/// WebGPU default `maxComputeInvocationsPerWorkgroup`; the devices here request
/// no limits, so workgroup shapes stay within it.
const DEFAULT_MAX_INVOCATIONS: u32 = 256;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_adapter_advertises_subgroup_size_control() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        if !adapter_has_size_control(adapter) {
            eprintln!("skip: adapter does not expose subgroup-size-control");
            release_adapter_instance(adapter, instance);
            return;
        }
        assert!(
            yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_Subgroups) != 0,
            "subgroup-size-control implies an adapter with subgroups"
        );
        let info = adapter_info(adapter);
        eprintln!(
            "subgroup size range: {}..={}",
            info.subgroupMinSize, info.subgroupMaxSize
        );
        assert!(info.subgroupMinSize.is_power_of_two());
        assert!(info.subgroupMaxSize.is_power_of_two());
        assert!(info.subgroupMinSize >= 4 && info.subgroupMaxSize <= 128);
        assert!(info.subgroupMinSize <= info.subgroupMaxSize);

        // Requesting only subgroup-size-control enables subgroups too.
        let device = request_device_with(
            instance,
            adapter,
            &[native::WGPUFeatureName_SubgroupSizeControl],
        );
        assert!(
            yawgpu::wgpuDeviceHasFeature(device, native::WGPUFeatureName_SubgroupSizeControl) != 0
        );
        assert!(
            yawgpu::wgpuDeviceHasFeature(device, native::WGPUFeatureName_Subgroups) != 0,
            "subgroup-size-control must implicitly enable subgroups"
        );
        yawgpu::wgpuDeviceRelease(device);
        release_adapter_instance(adapter, instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_subgroup_size_attribute_pins_runtime_width() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        if !adapter_has_size_control(adapter) {
            eprintln!("skip: adapter does not expose subgroup-size-control");
            release_adapter_instance(adapter, instance);
            return;
        }
        let info = adapter_info(adapter);
        let device = request_device_with(
            instance,
            adapter,
            &[native::WGPUFeatureName_SubgroupSizeControl],
        );
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let mut ran = 0;
        let mut size = info.subgroupMinSize;
        while size <= info.subgroupMaxSize {
            for per_workgroup in [1u32, 2, 4] {
                let wgx = size * per_workgroup;
                if wgx > DEFAULT_MAX_INVOCATIONS {
                    continue;
                }
                run_pinned_width_case(instance, device, queue, size, wgx);
                ran += 1;
            }
            size *= 2;
        }
        assert!(ran > 0, "no (size, workgroup) combination fit the limits");
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "pinned-width pipelines raised device errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        release_adapter_instance(adapter, instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_subgroup_size_validation_errors() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        if !adapter_has_size_control(adapter) {
            eprintln!("skip: adapter does not expose subgroup-size-control");
            release_adapter_instance(adapter, instance);
            return;
        }
        let info = adapter_info(adapter);
        let device = request_device_with(
            instance,
            adapter,
            &[native::WGPUFeatureName_SubgroupSizeControl],
        );
        let errors = install_error_capture(device);
        let min = info.subgroupMinSize;
        let max = info.subgroupMaxSize;

        // Rule 1: workgroup x not a multiple of S (x = S + S/2).
        let rule1 = format!(
            "enable subgroups;\nenable subgroup_size_control;\n\
             @compute @workgroup_size({}) @subgroup_size({min})\nfn main() {{}}\n",
            min + min / 2
        );
        expect_pipeline_error(device, &errors, &rule1, &[], "not a multiple");

        // Rule 2: S above the adapter maximum (still a power of two).
        if max * 2 <= DEFAULT_MAX_INVOCATIONS {
            let rule2 = format!(
                "enable subgroups;\nenable subgroup_size_control;\n\
                 @compute @workgroup_size({0}) @subgroup_size({0})\nfn main() {{}}\n",
                max * 2
            );
            expect_pipeline_error(device, &errors, &rule2, &[], "allowed range");
        }

        // Rule 4: an override-driven S that is not a power of two. Tint does
        // not check this after override substitution; core must.
        let rule4 = "enable subgroups;\nenable subgroup_size_control;\n\
                     override sg: u32 = 32u;\n\
                     @compute @workgroup_size(96) @subgroup_size(sg)\nfn main() {}\n";
        expect_pipeline_error(device, &errors, rule4, &[("sg", 24.0)], "power of two");

        yawgpu::wgpuDeviceRelease(device);
        release_adapter_instance(adapter, instance);
    }
}

// ---- cases ----

unsafe fn run_pinned_width_case(
    instance: native::WGPUInstance,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    size: u32,
    wgx: u32,
) {
    // sizes[i] = @builtin(subgroup_size); sums[i] = subgroupAdd(1u).
    let shader = format!(
        "enable subgroups;\nenable subgroup_size_control;\n\
         @group(0) @binding(0) var<storage, read_write> out: array<u32>;\n\
         @compute @workgroup_size({wgx}) @subgroup_size({size})\n\
         fn main(@builtin(local_invocation_index) li: u32,\n\
                 @builtin(subgroup_size) sg_size: u32) {{\n\
             out[li] = sg_size;\n\
             out[{wgx}u + li] = subgroupAdd(1u);\n\
         }}\n"
    );
    let out_size = u64::from(wgx) * 2 * 4;
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
    let module = create_wgsl_module(device, &shader);
    let pipeline = create_compute_pipeline(device, module, &[]);
    assert!(!pipeline.is_null());
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_ne_bytes(*c))
        .collect();
    let wg = wgx as usize;
    let sizes = &words[0..wg];
    let sums = &words[wg..2 * wg];
    assert!(
        sizes.iter().all(|&v| v == size),
        "@subgroup_size({size}) wgx={wgx}: subgroup_size builtin must equal {size}: {sizes:?}"
    );
    assert!(
        sums.iter().all(|&v| v == size),
        "@subgroup_size({size}) wgx={wgx}: subgroupAdd(1u) must equal {size} \
         (all invocations active): {sums:?}"
    );

    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuBufferRelease(readback);
    yawgpu::wgpuBufferRelease(output);
}

unsafe fn expect_pipeline_error(
    device: native::WGPUDevice,
    errors: &Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
    shader: &str,
    constants: &[(&str, f64)],
    needle: &str,
) {
    errors.lock().expect("error lock").clear();
    let module = create_wgsl_module(device, shader);
    assert!(
        errors.lock().expect("error lock").is_empty(),
        "shader module itself must be valid: {shader}"
    );
    let pipeline = create_compute_pipeline(device, module, constants);
    let captured = errors.lock().expect("error lock");
    assert_eq!(
        captured.len(),
        1,
        "expected exactly one pipeline validation error for:\n{shader}\ngot {captured:?}"
    );
    let message = format!("{:?}", captured[0]);
    assert!(
        message.contains(needle),
        "error for:\n{shader}\nmust mention {needle:?}, got {message}"
    );
    drop(captured);
    if !pipeline.is_null() {
        yawgpu::wgpuComputePipelineRelease(pipeline);
    }
    yawgpu::wgpuShaderModuleRelease(module);
}

// ---- helpers ----

unsafe fn adapter_has_size_control(adapter: native::WGPUAdapter) -> bool {
    yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_SubgroupSizeControl) != 0
}

unsafe fn release_adapter_instance(adapter: native::WGPUAdapter, instance: native::WGPUInstance) {
    yawgpu::wgpuAdapterRelease(adapter);
    yawgpu::wgpuInstanceRelease(instance);
}

unsafe fn adapter_info(adapter: native::WGPUAdapter) -> native::WGPUAdapterInfo {
    let mut info: native::WGPUAdapterInfo = std::mem::zeroed();
    assert_eq!(
        yawgpu::wgpuAdapterGetInfo(adapter, &mut info),
        native::WGPUStatus_Success
    );
    info
}

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

unsafe fn create_compute_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    constants: &[(&str, f64)],
) -> native::WGPUComputePipeline {
    let entries: Vec<native::WGPUConstantEntry> = constants
        .iter()
        .map(|(key, value)| native::WGPUConstantEntry {
            nextInChain: std::ptr::null_mut(),
            key: string_view(key),
            value: *value,
        })
        .collect();
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("main"),
            constantCount: entries.len(),
            constants: if entries.is_empty() {
                std::ptr::null()
            } else {
                entries.as_ptr()
            },
        },
    };
    yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor)
}

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

unsafe fn request_device_with(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    features: &[native::WGPUFeatureName],
) -> native::WGPUDevice {
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
