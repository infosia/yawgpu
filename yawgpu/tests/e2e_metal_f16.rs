//! Real-GPU verification of the WebGPU `shader-f16` feature on Metal.
//!
//! Proves the feature end-to-end on hardware (not just Noop validation):
//! - the Metal adapter advertises `WGPUFeatureName_ShaderF16`;
//! - a device that requested `shader-f16` runs an `enable f16;` compute shader
//!   that reads f16 from a storage buffer, does f16 arithmetic, and writes f16
//!   back — readback-verified (exercises naga MSL `half` + f16 buffer I/O);
//! - a device that did NOT request the feature rejects an f16-using shader at
//!   module creation (the S12 gate fires on a real backend too).
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

const ELEMENTS: usize = 8;

/// Decodes an IEEE-754 binary16 bit pattern to f32 (test-side oracle).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if (bits >> 15) & 1 == 1 { -1.0 } else { 1.0 };
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    match exp {
        0 => sign * f32::from(mant) * 2f32.powi(-24),
        0x1f => {
            if mant == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + f32::from(mant) / 1024.0) * 2f32.powi(i32::from(exp) - 15),
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_adapter_advertises_shader_f16() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        assert!(
            yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_ShaderF16) != 0,
            "Metal adapter must advertise shader-f16"
        );
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_f16_compute_doubles_values() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device_with_f16(instance, adapter);
        assert!(
            yawgpu::wgpuDeviceHasFeature(device, native::WGPUFeatureName_ShaderF16) != 0,
            "device must report shader-f16 after requesting it"
        );
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // u32 input → f16 arithmetic → f16 storage output. Doubling small
        // integers is exact in binary16, so the expected output is exact.
        let shader = r#"
enable f16;
struct InData { values: array<u32, 8> }
struct OutData { values: array<f16, 8> }
@group(0) @binding(0) var<storage, read> in_data: InData;
@group(0) @binding(1) var<storage, read_write> out_data: OutData;
@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < 8u) {
        out_data.values[i] = f16(in_data.values[i]) * 2.0h;
    }
}
"#;
        let input: Vec<u32> = (1..=ELEMENTS as u32).collect();
        let input_buffer = create_buffer_sized(
            device,
            (ELEMENTS * 4) as u64,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopyDst,
        );
        yawgpu::wgpuQueueWriteBuffer(
            queue,
            input_buffer,
            0,
            input.as_ptr().cast(),
            std::mem::size_of_val(input.as_slice()),
        );
        let out_size = (ELEMENTS * 2) as u64; // array<f16, 8> = 16 bytes
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

        // Module creation must not error; the device-error sink is asserted
        // empty below, which covers a rejected (error) module.
        let module = create_wgsl_module(device, shader);
        let pipeline = create_compute_pipeline(device, module);
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let bind_group = create_io_bind_group(device, layout, input_buffer, output, out_size);

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
        let actual: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_ne_bytes([c[0], c[1]])))
            .collect();
        let expected: Vec<f32> = (1..=ELEMENTS as u32).map(|v| (v * 2) as f32).collect();
        assert_eq!(actual, expected, "f16 compute doubling mismatch");
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "f16 compute path raised a device error"
        );

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(output);
        yawgpu::wgpuBufferRelease(input_buffer);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_f16_bitcast_roundtrip_with_arith() {
    // F-121 regression: bitcast involving f16 (width/size-changing). Before the
    // naga fix, `bitcast<vec2<f16>>(u32)` lowered to `as_type<float>` (scalar
    // f32), so the subsequent f16 arithmetic failed to compile → error pipeline
    // → "queue submit cannot use an error command buffer". This shader does
    // u32 → vec2<f16> → +f16 → u32, which only compiles/runs when the bitcast
    // target type is correct, and the value verifies the half2 interpretation.
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device_with_f16(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let shader = r#"
enable f16;
@group(0) @binding(0) var<storage, read> inp: u32;
@group(0) @binding(1) var<storage, read_write> out: u32;
@compute @workgroup_size(1)
fn main() {
    let h = bitcast<vec2<f16>>(inp);
    let s = h + vec2<f16>(1.0h, 3.0h);
    out = bitcast<u32>(s);
}
"#;
        // inp = 0xC0003C00 → (1.0h, -2.0h); +(1.0h, 3.0h) → (2.0h, 1.0h);
        // bitcast<u32> → (0x3C00 << 16) | 0x4000 = 0x3C004000 = 1006649344.
        let input: u32 = 0xC000_3C00;
        let expected: u32 = 0x3C00_4000;

        let input_buffer = create_buffer_sized(
            device,
            4,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopyDst,
        );
        yawgpu::wgpuQueueWriteBuffer(queue, input_buffer, 0, (&input as *const u32).cast(), 4);
        let output = create_buffer_sized(
            device,
            4,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let readback = create_buffer_sized(
            device,
            4,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let module = create_wgsl_module(device, shader);
        let pipeline = create_compute_pipeline(device, module);
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let in_entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 0,
            buffer: input_buffer,
            offset: 0,
            size: 4,
            sampler: std::ptr::null(),
            textureView: std::ptr::null(),
        };
        let out_entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 1,
            buffer: output,
            offset: 0,
            size: 4,
            sampler: std::ptr::null(),
            textureView: std::ptr::null(),
        };
        let entries = [in_entry, out_entry];
        let bg_desc = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout,
            entryCount: entries.len(),
            entries: entries.as_ptr(),
        };
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &bg_desc);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, 4);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let bytes = read_buffer(instance, readback, 4);
        let actual = u32::from_ne_bytes(bytes[0..4].try_into().expect("four bytes"));
        assert_eq!(
            actual, expected,
            "f16 bitcast round-trip+arith mismatch (got {actual:#x}, want {expected:#x})"
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "f16 bitcast path raised a device error"
        );

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(output);
        yawgpu::wgpuBufferRelease(input_buffer);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_const_struct_with_matrix_compiles() {
    // Regression: a module-scope `const` struct that contains a matrix member
    // made naga's MSL backend emit a `constant`-address-space global whose
    // aggregate matrix initializer Metal rejects ("cannot have global
    // constructors (llvm.global_ctors) in program_source") → error pipeline →
    // "queue submit cannot use an error command buffer". (Not f16-specific; the
    // f16 member here mirrors the CTS access,structure,index:const case that
    // surfaced it.) The naga fix inlines such consts instead of emitting the
    // global; this shader must now compile and read back member_0 = 7.
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device_with_f16(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let shader = r#"
enable f16;
@group(0) @binding(0) var<storage, read_write> output : i32;
struct MyStruct { member_0 : i32, member_1 : f16, member_2 : vec4i, member_3 : mat3x2f, };
const S = MyStruct(7i, 1.0h, vec4i(2i, 2i, 2i, 2i), mat3x2f(3f, 3f, 3f, 3f, 3f, 3f));
@compute @workgroup_size(1)
fn main() { output = S.member_0; }
"#;
        let output = create_buffer_sized(
            device,
            4,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let readback = create_buffer_sized(
            device,
            4,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let module = create_wgsl_module(device, shader);
        let pipeline = create_compute_pipeline(device, module);
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 0,
            buffer: output,
            offset: 0,
            size: 4,
            sampler: std::ptr::null(),
            textureView: std::ptr::null(),
        };
        let bg_desc = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout,
            entryCount: 1,
            entries: &entry,
        };
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &bg_desc);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, 4);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let bytes = read_buffer(instance, readback, 4);
        let actual = i32::from_ne_bytes(bytes[0..4].try_into().expect("four bytes"));
        assert_eq!(actual, 7, "const struct member_0 readback");
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "const struct with matrix raised a device error (global-constructors regression)"
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
fn metal_f16_shader_rejected_without_feature() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        // Device requests NO features → shader-f16 not enabled.
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let shader =
            "enable f16;\n@compute @workgroup_size(1) fn cs() { let x: f16 = 1.0h; _ = x; }";
        let module = create_wgsl_module_allow_error(device, shader);

        // Without the shader-f16 feature the S12 gate rejects f16 usage: the
        // create routes a validation error to the device error sink and returns
        // an error shader-module handle (non-null, Release-safe).
        assert!(!module.is_null());
        assert!(
            !errors.lock().expect("error lock").is_empty(),
            "rejected f16 shader must raise a validation device error"
        );

        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// ---- helpers ----

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
unsafe fn create_io_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    input: native::WGPUBuffer,
    output: native::WGPUBuffer,
    out_size: u64,
) -> native::WGPUBindGroup {
    let in_entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer: input,
        offset: 0,
        size: (ELEMENTS * 4) as u64,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    };
    let out_entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 1,
        buffer: output,
        offset: 0,
        size: out_size,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    };
    let entries = [in_entry, out_entry];
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
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
unsafe fn request_device_with_f16(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let features = [native::WGPUFeatureName_ShaderF16];
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
