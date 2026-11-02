//! Shared body of the Block 120 e2e (`specs/blocks/120-tint-driver-workarounds.md`),
//! included by `e2e_metal_shader_workarounds.rs` and
//! `e2e_vulkan_shader_workarounds.rs` through `#[path]`. No development host matches
//! a workaround gate, so each scenario runs twice: as selected for this host, then in
//! a child process with `YAWGPU_FORCE_SHADER_WORKAROUNDS` set. The results must not
//! change. The render scenario binds nothing, so on Metal `fix_u32_div_mod` is the
//! only reason either stage has an immediate block (Block 120 R7).

use std::process::Command;

use yawgpu::native;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

use super::indirect_dispatch::{
    create_buffer, create_instance, create_wgsl_module, empty_string_view, install_error_capture,
    read_buffer, request_adapter, request_device, string_view,
};

const CHILD_ENV: &str = "YAWGPU_E2E_SHADER_WORKAROUNDS_CHILD";
const FORCE_ENV: &str = "YAWGPU_FORCE_SHADER_WORKAROUNDS";

const COMPUTE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<u32, 4>;
@group(0) @binding(1) var<storage, read_write> output: array<u32, 8>;
var<workgroup> shared_value: atomic<u32>;

@compute @workgroup_size(1)
fn main() {
    let a = input[0];
    let b = input[1];
    let index = input[2];
    let zero = input[3];

    output[0] = a / b;
    output[1] = a % b;
    let divided = vec2<u32>(a, a + 1u) % vec2<u32>(b);
    output[2] = divided.x;
    output[3] = divided.y;

    var flags = vec3<bool>(false, false, false);
    flags[index] = true;
    output[4] = select(0u, 1u, flags.x) + select(0u, 2u, flags.y) + select(0u, 4u, flags.z);

    atomicStore(&shared_value, a);
    output[5] = atomicLoad(&shared_value);

    output[6] = select(10u, 20u, zero == 0u);
    output[7] = select(10u, 20u, a == 0u);
}
"#;

const COMPUTE_INPUT: [u32; 4] = [0x1_0004, 3, 1, 0];
const COMPUTE_EXPECTED: [u32; 8] = [21_846, 2, 2, 0, 2, 0x1_0004, 20, 10];

const RENDER_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[(index + 3u) % 3u], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<u32> {
    let value = u32(position.x) + 0x10004u;
    return vec4<u32>(value % 3u, (value / 3u) & 255u, 0u, 255u);
}
"#;

const RENDER_EXPECTED: [u8; 4] = [2, 0x56, 0, 255];

/// Runs `scenario` as selected for this host, then forced in a child process.
fn run_both_ways(backend: RealBackend, forced: &str, test_name: &str, scenario: impl Fn()) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    scenario();
    if std::env::var_os(CHILD_ENV).is_some() {
        return;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let output = Command::new(exe)
        .env(CHILD_ENV, "1")
        .env(FORCE_ENV, forced)
        .args([
            "--exact",
            "--ignored",
            "--test-threads=1",
            "--nocapture",
            test_name,
        ])
        .output()
        .expect("spawn child test process");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "forced run failed: {:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        output.status,
    );
}

pub fn compute(backend: RealBackend, select: u32, forced: &str, test_name: &str) {
    run_both_ways(backend, forced, test_name, || unsafe {
        run_compute(select);
    });
}

pub fn render(backend: RealBackend, select: u32, forced: &str, test_name: &str) {
    run_both_ways(backend, forced, test_name, || unsafe {
        run_render(select);
    });
}

unsafe fn run_compute(select: u32) {
    let instance = create_instance(select);
    let adapter = request_adapter(instance);
    let device = request_device(instance, adapter, &[]);
    let errors = install_error_capture(device);
    let queue = yawgpu::wgpuDeviceGetQueue(device);

    let input_bytes: Vec<u8> = COMPUTE_INPUT
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let input = create_buffer(
        device,
        16,
        native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopyDst,
    );
    yawgpu::wgpuQueueWriteBuffer(queue, input, 0, input_bytes.as_ptr().cast(), 16);
    let output = create_buffer(
        device,
        32,
        native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
    );
    let readback = create_buffer(
        device,
        32,
        native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
    );

    let module = create_wgsl_module(device, COMPUTE_SHADER);
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

    let entry = |binding, buffer, size| native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer,
        offset: 0,
        size,
        sampler: std::ptr::null_mut(),
        textureView: std::ptr::null_mut(),
    };
    let entries = [entry(0, input, 16), entry(1, output, 32)];
    let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
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
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, 32);
    submit(queue, encoder);

    let bytes = read_buffer(instance, readback, 32);
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect();
    assert_eq!(words, COMPUTE_EXPECTED);
    assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");

    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    for buffer in [input, output, readback] {
        yawgpu::wgpuBufferRelease(buffer);
    }
    yawgpu::wgpuQueueRelease(queue);
    yawgpu::wgpuDeviceRelease(device);
    yawgpu::wgpuAdapterRelease(adapter);
    yawgpu::wgpuInstanceRelease(instance);
}

unsafe fn run_render(select: u32) {
    let instance = create_instance(select);
    let adapter = request_adapter(instance);
    let device = request_device(instance, adapter, &[]);
    let errors = install_error_capture(device);
    let queue = yawgpu::wgpuDeviceGetQueue(device);

    let extent = native::WGPUExtent3D {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(
        device,
        &native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            dimension: native::WGPUTextureDimension_2D,
            size: extent,
            format: native::WGPUTextureFormat_RGBA8Uint,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        },
    );
    assert!(!texture.is_null());
    let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
    let readback = create_buffer(
        device,
        4,
        native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
    );

    let module = create_wgsl_module(device, RENDER_SHADER);
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Uint,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &color_target,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(
        device,
        &native::WGPURenderPipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null(),
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
                frontFace: native::WGPUFrontFace_Undefined,
                cullMode: native::WGPUCullMode_Undefined,
                unclippedDepth: 0,
            },
            depthStencil: std::ptr::null(),
            multisample: native::WGPUMultisampleState {
                nextInChain: std::ptr::null_mut(),
                count: 1,
                mask: 0xFFFF_FFFF,
                alphaToCoverageEnabled: 0,
            },
            fragment: &fragment,
        },
    );
    assert!(!pipeline.is_null());

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let attachment = native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        },
    };
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(
        encoder,
        &native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 1,
            colorAttachments: &attachment,
            depthStencilAttachment: std::ptr::null(),
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        },
    );
    yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
        encoder,
        &native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        },
        &native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: 256,
                rowsPerImage: 1,
            },
            buffer: readback,
        },
        &extent,
    );
    submit(queue, encoder);

    assert_eq!(read_buffer(instance, readback, 4), RENDER_EXPECTED);
    assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");

    yawgpu::wgpuRenderPipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuBufferRelease(readback);
    yawgpu::wgpuTextureViewRelease(view);
    yawgpu::wgpuTextureRelease(texture);
    yawgpu::wgpuQueueRelease(queue);
    yawgpu::wgpuDeviceRelease(device);
    yawgpu::wgpuAdapterRelease(adapter);
    yawgpu::wgpuInstanceRelease(instance);
}

unsafe fn submit(queue: native::WGPUQueue, encoder: native::WGPUCommandEncoder) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}
