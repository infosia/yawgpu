//! Real-Metal e2e for Block 123 (`specs/blocks/123-metal-msl-math-mode.md`): Tint
//! MSL compiles under Dawn's relaxed math mode, which keeps infinities and NaNs. With
//! Metal's default (fast) math, `x - x`, `x / x`, and `x * 0.0` fold to finite values
//! for `x = +inf`; under relaxed math they are NaN, as on Dawn.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;

use indirect_dispatch::{
    create_buffer, create_instance, create_wgsl_module, empty_string_view, install_error_capture,
    read_buffer, request_adapter, request_device, string_view,
};
use yawgpu::native;
use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32, 1>;
@group(0) @binding(1) var<storage, read_write> output: array<u32, 3>;

@compute @workgroup_size(1)
fn main() {
    let x = input[0];
    output[0] = bitcast<u32>(x - x);
    output[1] = bitcast<u32>(x / x);
    output[2] = bitcast<u32>(x * 0.0);
}
"#;

fn is_nan_bits(bits: u32) -> bool {
    f32::from_bits(bits).is_nan()
}

/// Relaxed math needs macOS 15 (R1); before it the default fast math applies by
/// design (R2), so the NaN results are not expected there.
fn macos_at_least_15() -> bool {
    let output = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .expect("sw_vers");
    let version = String::from_utf8_lossy(&output.stdout);
    let major: u32 = version
        .trim()
        .split('.')
        .next()
        .and_then(|major| major.parse().ok())
        .expect("macOS major version");
    major >= 15
}

#[test]
#[ignore = "manual real-backend test"]
fn infinity_arithmetic_keeps_nan_like_dawn() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() || !macos_at_least_15() {
        return;
    }
    unsafe {
        let instance = create_instance(YAWGPU_INSTANCE_BACKEND_METAL);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let input = create_buffer(
            device,
            4,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopyDst,
        );
        let infinity = f32::INFINITY.to_le_bytes();
        yawgpu::wgpuQueueWriteBuffer(queue, input, 0, infinity.as_ptr().cast(), 4);
        let output = create_buffer(
            device,
            16,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let readback = create_buffer(
            device,
            16,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );

        let module = create_wgsl_module(device, SHADER);
        let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(
            device,
            &native::WGPUComputePipelineDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout: std::ptr::null_mut(),
                compute: native::WGPUComputeState {
                    nextInChain: std::ptr::null_mut(),
                    module,
                    entryPoint: string_view("main"),
                    constantCount: 0,
                    constants: std::ptr::null(),
                },
            },
        );
        assert!(!pipeline.is_null());
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let entries = [
            native::WGPUBindGroupEntry {
                nextInChain: std::ptr::null_mut(),
                binding: 0,
                buffer: input,
                offset: 0,
                size: 4,
                sampler: std::ptr::null_mut(),
                textureView: std::ptr::null_mut(),
            },
            native::WGPUBindGroupEntry {
                nextInChain: std::ptr::null_mut(),
                binding: 1,
                buffer: output,
                offset: 0,
                size: 12,
                sampler: std::ptr::null_mut(),
                textureView: std::ptr::null_mut(),
            },
        ];
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(
            device,
            &native::WGPUBindGroupDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                layout,
                entryCount: entries.len(),
                entries: entries.as_ptr(),
            },
        );
        assert!(!bind_group.is_null());

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, 16);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let bytes = read_buffer(instance, readback, 12);
        let words: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        for (expression, bits) in ["x - x", "x / x", "x * 0.0"].iter().zip(&words) {
            assert!(
                is_nan_bits(*bits),
                "{expression} with x = +inf gave {bits:#010x}, expected NaN"
            );
        }
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(output);
        yawgpu::wgpuBufferRelease(input);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}
