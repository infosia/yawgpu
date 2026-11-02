//! Real-Metal e2e for Block 124 (`specs/blocks/124-metal-preserve-invariance.md`):
//! `preserveInvariance` is now set only for MSL that declares an `@invariant`
//! output. Two pipelines draw the same triangle with `@invariant` positions but
//! different vertex and fragment code around them; the second pass uses depth
//! compare `equal`, so every covered pixel passes only if both computed the same
//! depth. This is a smoke and regression test that invariant pipelines still compile
//! and agree. It cannot prove the flag reaches `MTLCompileOptions`: on Apple GPUs the
//! two shaders likely agree even without it. The core and HAL unit tests pin the flag.

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

const SIZE: u32 = 64;
const BYTES_PER_ROW: u32 = SIZE * 4;

// Both entry points share the position math; `vs_b` adds unrelated work and an
// extra output, which gives the compiler room to schedule the math differently.
const SHADER: &str = r#"
fn corner(index: u32) -> vec3<f32> {
    var corners = array<vec3<f32>, 3>(
        vec3<f32>(-0.83, -0.91, 0.137),
        vec3<f32>(0.97, -0.41, 0.711),
        vec3<f32>(-0.29, 0.89, 0.423),
    );
    return corners[index];
}

fn transform(p: vec3<f32>) -> vec4<f32> {
    let s = vec3<f32>(0.977, 1.013, 0.871);
    let t = vec3<f32>(0.011, -0.007, 0.053);
    let q = p * s + t;
    return vec4<f32>(q.x * 1.0001, q.y * 0.9999, q.z * 0.913 + q.x * 0.017, 1.0);
}

struct OutA {
    @invariant @builtin(position) position: vec4<f32>,
}

struct OutB {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) shade: f32,
}

@vertex
fn vs_a(@builtin(vertex_index) index: u32) -> OutA {
    return OutA(transform(corner(index)));
}

@vertex
fn vs_b(@builtin(vertex_index) index: u32) -> OutB {
    let p = corner(index);
    let shade = fract(sin(p.x * 12.9898 + p.y * 78.233) * 43758.547);
    return OutB(transform(p), shade);
}

@fragment
fn fs_a() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_b(input: OutB) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, input.shade * 0.0, 1.0);
}
"#;

#[test]
#[ignore = "manual real-backend test"]
fn invariant_positions_pass_depth_equal_across_pipelines() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_instance(YAWGPU_INSTANCE_BACKEND_METAL);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let module = create_wgsl_module(device, SHADER);

        let color = create_texture(
            device,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        );
        let depth = create_texture(
            device,
            native::WGPUTextureFormat_Depth32Float,
            native::WGPUTextureUsage_RenderAttachment,
        );
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

        let first = create_pipeline(
            device,
            module,
            ("vs_a", "fs_a"),
            native::WGPUCompareFunction_Less,
            true,
        );
        let second = create_pipeline(
            device,
            module,
            ("vs_b", "fs_b"),
            native::WGPUCompareFunction_Equal,
            false,
        );

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: color_view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null_mut(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        };
        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 1.0,
            depthReadOnly: 0,
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: 0,
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 1,
            colorAttachments: &color_attachment,
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null_mut(),
            timestampWrites: std::ptr::null(),
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, first);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, second);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        let readback = create_buffer(
            device,
            u64::from(BYTES_PER_ROW * SIZE),
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        let source = native::WGPUTexelCopyTextureInfo {
            texture: color,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let destination = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: BYTES_PER_ROW,
                rowsPerImage: SIZE,
            },
            buffer: readback,
        };
        let extent = native::WGPUExtent3D {
            width: SIZE,
            height: SIZE,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &extent);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let pixels = read_buffer(instance, readback, (BYTES_PER_ROW * SIZE) as usize);
        let (mut covered, mut red) = (0, 0);
        for pixel in pixels.as_chunks::<4>().0 {
            match pixel {
                [0, 0, 0, 255] => {}
                [0, 255, 0, 255] => covered += 1,
                [255, 0, 0, 255] => red += 1,
                other => panic!("unexpected pixel {other:?}"),
            }
        }
        assert!(covered > 500, "triangle covered only {covered} pixels");
        assert_eq!(
            red, 0,
            "{red} pixels failed depth equal: the two pipelines disagreed on depth"
        );
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderPipelineRelease(second);
        yawgpu::wgpuRenderPipelineRelease(first);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    format: native::WGPUTextureFormat,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: SIZE,
            height: SIZE,
            depthOrArrayLayers: 1,
        },
        format,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

unsafe fn create_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    (vertex_entry, fragment_entry): (&str, &str),
    depth_compare: native::WGPUCompareFunction,
    depth_write: bool,
) -> native::WGPURenderPipeline {
    let target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view(fragment_entry),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &target,
    };
    let stencil_face = native::WGPUStencilFaceState {
        compare: native::WGPUCompareFunction_Always,
        failOp: native::WGPUStencilOperation_Keep,
        depthFailOp: native::WGPUStencilOperation_Keep,
        passOp: native::WGPUStencilOperation_Keep,
    };
    let depth_stencil = native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32Float,
        depthWriteEnabled: if depth_write {
            native::WGPUOptionalBool_True
        } else {
            native::WGPUOptionalBool_False
        },
        depthCompare: depth_compare,
        stencilFront: stencil_face,
        stencilBack: stencil_face,
        stencilReadMask: 0xff,
        stencilWriteMask: 0xff,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null_mut(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view(vertex_entry),
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
        depthStencil: &depth_stencil,
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: 1,
            mask: u32::MAX,
            alphaToCoverageEnabled: 0,
        },
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}
