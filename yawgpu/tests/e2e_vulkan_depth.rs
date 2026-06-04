//! Real-GPU Vulkan diagnosis for F-031 on the Vulkan backend: the depth render
//! path (`@builtin(frag_depth)` into a depth attachment) and the depth aspect of
//! `copyTextureToBuffer`. The CTS `copyTextureToTexture:copy_depth_stencil` test
//! couples a depth render, a t2t copy, and a depthCompare=Equal re-render; these
//! probes split the depth RENDER + depth readback apart from the copy so a
//! failure localises to a single stage. Gated on the `vulkan` feature (runs on
//! MoltenVK on macOS).
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

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const BYTES_PER_ROW: u32 = 256;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;
const DEPTH_VALUE: f32 = 0.5;

// Full-screen quad whose fragment writes a CONSTANT frag_depth. Isolates the
// depth render + depth-write from any sampling.
const FRAG_DEPTH_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.0, 1.0);
}
@fragment
fn fs() -> @builtin(frag_depth) f32 {
    return 0.5;
}
"#;

/// Isolation — render a CONSTANT `frag_depth` into a `Depth32Float` texture, then
/// read the depth aspect back via `copyTextureToBuffer(DepthOnly)`. If this reads
/// zeros, the Vulkan depth render path (not the copy) is broken.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_depth_render_constant_then_t2b() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let depth = create_depth_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let module = create_wgsl_module(device, FRAG_DEPTH_SHADER);
        let pipeline = create_depth_pipeline(device, module);
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_depth_t2b(encoder, depth, readback);
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        assert!(
            depths.iter().all(|d| (d - DEPTH_VALUE).abs() < 1e-5),
            "vulkan constant frag_depth not written: {depths:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// Full-screen quad whose fragment SAMPLES a bound `r32float` per fragment and
// writes it as `frag_depth` — the exact CTS depth staging.
const SAMPLE_FRAG_DEPTH_SHADER: &str = r#"
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.0, 1.0);
}
@fragment
fn fs(@builtin(position) coord: vec4<f32>) -> @builtin(frag_depth) f32 {
    return textureLoad(input_tex, vec2<i32>(coord.xy), 0).x;
}
"#;

/// Isolation — the exact CTS depth staging on Vulkan: SAMPLE an `r32float` per
/// fragment and write it as `frag_depth` into a `Depth32Float` texture, then read
/// the depth aspect back PER TEXEL. The constant-frag_depth probe passes, so this
/// pins whether sampled-texture binding inside a render pass works on Vulkan.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_sampled_frag_depth_per_texel() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let input = create_r32float_input(device);
        let mut floats = vec![0u8; READBACK_SIZE];
        let expected_depth =
            |col: usize, row: usize| (col + row * WIDTH as usize + 1) as f32 / 32.0;
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                floats[off..off + 4].copy_from_slice(&expected_depth(col, row).to_le_bytes());
            }
        }
        let input_dst = native::WGPUTexelCopyTextureInfo {
            texture: input,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let input_layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &input_dst,
            floats.as_ptr().cast(),
            floats.len(),
            &input_layout,
            &texture_extent(),
        );

        let depth = create_depth_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let module = create_wgsl_module(device, SAMPLE_FRAG_DEPTH_SHADER);
        let pipeline = create_depth_pipeline(device, module);
        let input_view = yawgpu::wgpuTextureCreateView(input, std::ptr::null());
        let bind_group = create_texture_bind_group(device, pipeline, input_view);
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_depth_t2b(encoder, depth, readback);
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        let mut mismatches = Vec::new();
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let got = depths[row * WIDTH as usize + col];
                let want = expected_depth(col, row);
                if (got - want).abs() > 1e-5 {
                    mismatches.push((col, row, want, got));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "vulkan sampled frag_depth not written per texel (col,row,want,got): {mismatches:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureViewRelease(input_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuTextureRelease(input);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// VERTEX-ONLY depth write: a full-screen quad whose vertex `position.z` is the
// depth, with NO fragment shader — exactly how the CTS copy_depth_stencil t2t
// test stages depth (`createDepthCopyPipeline(..., includeFragment=false)`).
const VERTEX_ONLY_DEPTH_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec3<f32>, 6>(
        vec3<f32>(-1.0, -1.0, 0.5), vec3<f32>( 1.0, -1.0, 0.5), vec3<f32>(-1.0,  1.0, 0.5),
        vec3<f32>(-1.0,  1.0, 0.5), vec3<f32>( 1.0, -1.0, 0.5), vec3<f32>( 1.0,  1.0, 0.5));
    return vec4<f32>(pos[idx], 1.0);
}
"#;

/// Isolation — a VERTEX-ONLY depth pipeline (no fragment shader) writes depth via
/// `position.z`, then reads the depth aspect back. The frag_depth probes pass, so
/// this pins whether a vertex-only depth pipeline writes depth on Vulkan — the
/// exact staging the CTS `copy_depth_stencil` t2t test uses.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_vertex_only_depth_write_then_t2b() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let depth = create_depth_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let module = create_wgsl_module(device, VERTEX_ONLY_DEPTH_SHADER);
        let pipeline = create_vertex_only_depth_pipeline(device, module);
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());

        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_depth_t2b(encoder, depth, readback);
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        assert!(
            depths.iter().all(|d| (d - DEPTH_VALUE).abs() < 1e-5),
            "vulkan vertex-only depth not written (position.z): {depths:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[cfg(feature = "vulkan")]
unsafe fn create_vertex_only_depth_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let depth_stencil = native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32Float,
        depthWriteEnabled: native::WGPUOptionalBool_True,
        depthCompare: native::WGPUCompareFunction_Always,
        stencilFront: stencil_face(),
        stencilBack: stencil_face(),
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
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
        primitive: primitive_state(),
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: std::ptr::null(),
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

// Green fragment + constant-depth vertex for the depthCompare=Equal verify that
// mirrors the CTS `verifyDepthAspect` stage.
const GREEN_EQUAL_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec3<f32>, 6>(
        vec3<f32>(-1.0, -1.0, 0.5), vec3<f32>( 1.0, -1.0, 0.5), vec3<f32>(-1.0,  1.0, 0.5),
        vec3<f32>(-1.0,  1.0, 0.5), vec3<f32>( 1.0, -1.0, 0.5), vec3<f32>( 1.0,  1.0, 0.5));
    return vec4<f32>(pos[idx], 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
"#;

/// Isolation — the CTS `verifyDepthAspect` mechanism: write depth=0.5, then
/// re-render a quad at z=0.5 with `depthCompare=Equal` + a green fragment and read
/// the COLOUR back. Green means the equal depth-test passed. The render/t2b/t2t
/// probes pass, so this pins whether reading a depth texture as a `depthCompare`
/// attachment works on Vulkan.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_depth_equal_verify_renders_green() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let depth = create_depth_texture(device);
        let color = create_color_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let write_module = create_wgsl_module(device, FRAG_DEPTH_SHADER);
        let write_pipeline = create_depth_pipeline(device, write_module);
        let verify_module = create_wgsl_module(device, GREEN_EQUAL_SHADER);
        let verify_pipeline = create_depth_equal_pipeline(device, verify_module);
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());

        // Pass 1: write depth = 0.5 (Always, depthWrite).
        let write_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let write_pass = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &write_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &write_pass);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, write_pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        // Pass 2: equal verify — green where depth == 0.5.
        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: color_view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        };
        let verify_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Load,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let attachments = [color_attachment];
        let verify_pass = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: attachments.len(),
            colorAttachments: attachments.as_ptr(),
            depthStencilAttachment: &verify_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &verify_pass);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, verify_pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        record_color_t2b(encoder, color, readback);
        submit_encoder(queue, encoder);

        let pixels = read_buffer(instance, readback, 0, READBACK_SIZE);
        let green = pixels[0] == 0 && pixels[1] == 255 && pixels[2] == 0 && pixels[3] == 255;
        assert!(
            green,
            "vulkan depthCompare=Equal did not pass (first pixel): {:?}",
            &pixels[0..4]
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuRenderPipelineRelease(verify_pipeline);
        yawgpu::wgpuRenderPipelineRelease(write_pipeline);
        yawgpu::wgpuShaderModuleRelease(verify_module);
        yawgpu::wgpuShaderModuleRelease(write_module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Isolation — the per-ARRAY-LAYER render-attachment path the CTS uses
/// (`createLayerView(baseArrayLayer + layer)`): on a 2-layer depth + 2-layer
/// colour texture, write depth=0.5 into **layer 1** via a single-layer view, then
/// `depthCompare=Equal`-verify **layer 1** (green) via single-layer views, and read
/// layer 1's colour. The mip0/layer0 equal probe passes, so this pins whether the
/// Vulkan render attachment honours a view's `baseArrayLayer`.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_depth_equal_verify_layer1_renders_green() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    const LAYER: u32 = 1;

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let depth = create_layered_depth_texture(device, 2);
        let color = create_layered_color_texture(device, 2);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let write_module = create_wgsl_module(device, FRAG_DEPTH_SHADER);
        let write_pipeline = create_depth_pipeline(device, write_module);
        let verify_module = create_wgsl_module(device, GREEN_EQUAL_SHADER);
        let verify_pipeline = create_depth_equal_pipeline(device, verify_module);
        let depth_view =
            create_layer_view(device, depth, native::WGPUTextureFormat_Depth32Float, LAYER);
        let color_view =
            create_layer_view(device, color, native::WGPUTextureFormat_RGBA8Unorm, LAYER);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());

        // Pass 1: write depth = 0.5 into layer 1.
        let write_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let write_pass = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &write_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &write_pass);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, write_pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        // Pass 2: equal-verify layer 1 — green where depth == 0.5, RED clear elsewhere.
        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: color_view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        };
        let verify_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Load,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let attachments = [color_attachment];
        let verify_pass = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: attachments.len(),
            colorAttachments: attachments.as_ptr(),
            depthStencilAttachment: &verify_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &verify_pass);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, verify_pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        // Read layer 1 of the colour texture.
        let t2b_src = native::WGPUTexelCopyTextureInfo {
            texture: color,
            mipLevel: 0,
            origin: native::WGPUOrigin3D {
                x: 0,
                y: 0,
                z: LAYER,
            },
            aspect: native::WGPUTextureAspect_All,
        };
        let t2b_dst = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: BYTES_PER_ROW,
                rowsPerImage: HEIGHT,
            },
            buffer: readback,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &t2b_src,
            &t2b_dst,
            &texture_extent(),
        );
        submit_encoder(queue, encoder);

        let pixels = read_buffer(instance, readback, 0, READBACK_SIZE);
        let green = pixels[0] == 0 && pixels[1] == 255 && pixels[2] == 0 && pixels[3] == 255;
        assert!(
            green,
            "vulkan layer-1 depthCompare=Equal did not pass (first pixel): {:?}",
            &pixels[0..4]
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuRenderPipelineRelease(verify_pipeline);
        yawgpu::wgpuRenderPipelineRelease(write_pipeline);
        yawgpu::wgpuShaderModuleRelease(verify_module);
        yawgpu::wgpuShaderModuleRelease(write_module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[cfg(feature = "vulkan")]
unsafe fn create_layered_depth_texture(
    device: native::WGPUDevice,
    layers: u32,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment
            | native::WGPUTextureUsage_CopySrc
            | native::WGPUTextureUsage_CopyDst,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: WIDTH,
            height: HEIGHT,
            depthOrArrayLayers: layers,
        },
        format: native::WGPUTextureFormat_Depth32Float,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

#[cfg(feature = "vulkan")]
unsafe fn create_layered_color_texture(
    device: native::WGPUDevice,
    layers: u32,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: WIDTH,
            height: HEIGHT,
            depthOrArrayLayers: layers,
        },
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

#[cfg(feature = "vulkan")]
unsafe fn create_layer_view(
    device: native::WGPUDevice,
    texture: native::WGPUTexture,
    format: native::WGPUTextureFormat,
    layer: u32,
) -> native::WGPUTextureView {
    create_subresource_view(device, texture, format, 0, layer)
}

#[cfg(feature = "vulkan")]
unsafe fn create_subresource_view(
    device: native::WGPUDevice,
    texture: native::WGPUTexture,
    format: native::WGPUTextureFormat,
    mip: u32,
    layer: u32,
) -> native::WGPUTextureView {
    let _ = device;
    let descriptor = native::WGPUTextureViewDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        format,
        dimension: native::WGPUTextureViewDimension_2D,
        baseMipLevel: mip,
        mipLevelCount: 1,
        baseArrayLayer: layer,
        arrayLayerCount: 1,
        aspect: native::WGPUTextureAspect_All,
        usage: native::WGPUTextureUsage_None,
    };
    let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
    assert!(!view.is_null());
    view
}

/// Isolation — `copyTextureToTexture` INTO a non-zero destination mip level: render
/// constant depth=0.5 into a 4×4 source, t2t it into **mip 2** of a 16×16 3-mip
/// destination (mip 2 = 4×4), then read dst mip 2 back. The same-mip (0→0) t2t and
/// the mip-2 render both pass — this pins t2t into a non-zero dst mip.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_depth_t2t_into_mip2() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    const BASE: u32 = 16;
    const MIP: u32 = 2; // 16 >> 2 = 4 == WIDTH/HEIGHT

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // 4x4 source (1 mip), rendered with constant depth.
        let source = create_depth_texture(device);
        // 16x16 destination with 3 mips; mip 2 is 4x4.
        let dst_desc = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment
                | native::WGPUTextureUsage_CopySrc
                | native::WGPUTextureUsage_CopyDst,
            dimension: native::WGPUTextureDimension_2D,
            size: native::WGPUExtent3D {
                width: BASE,
                height: BASE,
                depthOrArrayLayers: 1,
            },
            format: native::WGPUTextureFormat_Depth32Float,
            mipLevelCount: 3,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let destination = yawgpu::wgpuDeviceCreateTexture(device, &dst_desc);
        assert!(!destination.is_null());
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let module = create_wgsl_module(device, FRAG_DEPTH_SHADER);
        let pipeline = create_depth_pipeline(device, module);
        let source_view = yawgpu::wgpuTextureCreateView(source, std::ptr::null());

        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: source_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        // t2t: source mip 0 (4x4) -> destination mip 2 (4x4).
        let t2t_src = native::WGPUTexelCopyTextureInfo {
            texture: source,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let t2t_dst = native::WGPUTexelCopyTextureInfo {
            texture: destination,
            mipLevel: MIP,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToTexture(
            encoder,
            &t2t_src,
            &t2t_dst,
            &texture_extent(),
        );

        // Read dst mip 2.
        let t2b_src = native::WGPUTexelCopyTextureInfo {
            texture: destination,
            mipLevel: MIP,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_DepthOnly,
        };
        let t2b_dst = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: BYTES_PER_ROW,
                rowsPerImage: HEIGHT,
            },
            buffer: readback,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &t2b_src,
            &t2b_dst,
            &texture_extent(),
        );
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        assert!(
            depths.iter().all(|d| (d - DEPTH_VALUE).abs() < 1e-5),
            "vulkan depth not preserved through t2t into dst mip 2: {depths:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuTextureViewRelease(source_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(destination);
        yawgpu::wgpuTextureRelease(source);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// GRADIENT depth via vertex position.z (corners 0.5/0.0/1.0), matching the CTS
// kDepthVertexShader (copyLayer=0 → 0.5+0.2*sin(0)=0.5). Vertex-only (staging).
const GRADIENT_DEPTH_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec3<f32>, 6>(
        vec3<f32>(-1.0,  1.0, 0.5), vec3<f32>(-1.0, -1.0, 0.0), vec3<f32>( 1.0,  1.0, 1.0),
        vec3<f32>(-1.0, -1.0, 0.0), vec3<f32>( 1.0,  1.0, 1.0), vec3<f32>( 1.0, -1.0, 0.5));
    return vec4<f32>(pos[idx], 1.0);
}
"#;

// Same gradient vertices + a green fragment, for the depthCompare=Equal verify.
const GRADIENT_GREEN_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec3<f32>, 6>(
        vec3<f32>(-1.0,  1.0, 0.5), vec3<f32>(-1.0, -1.0, 0.0), vec3<f32>( 1.0,  1.0, 1.0),
        vec3<f32>(-1.0, -1.0, 0.0), vec3<f32>( 1.0,  1.0, 1.0), vec3<f32>( 1.0, -1.0, 0.5));
    return vec4<f32>(pos[idx], 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
"#;

/// Isolation — the CTS uses a depth GRADIENT (varying vertex `z`) and renders the
/// staging (vertex-only) and the `depthCompare=Equal` verify with DIFFERENT
/// pipelines. On Metal this needed `preserveInvariance` so the interpolated depth
/// is bit-identical between the two pipelines. This probe writes the gradient
/// then equal-verifies it (green where depth matches) — if Vulkan's SPIR-V lacks
/// position invariance, the gradient mismatches and pixels stay RED.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_depth_gradient_equal_verify_renders_green() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let depth = create_depth_texture(device);
        let color = create_color_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let write_module = create_wgsl_module(device, GRADIENT_DEPTH_SHADER);
        let write_pipeline = create_vertex_only_depth_pipeline(device, write_module);
        let verify_module = create_wgsl_module(device, GRADIENT_GREEN_SHADER);
        let verify_pipeline = create_depth_equal_pipeline(device, verify_module);
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());

        let write_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let write_pass = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &write_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &write_pass);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, write_pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: color_view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        };
        let verify_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Load,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let attachments = [color_attachment];
        let verify_pass = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: attachments.len(),
            colorAttachments: attachments.as_ptr(),
            depthStencilAttachment: &verify_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &verify_pass);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, verify_pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        record_color_t2b(encoder, color, readback);
        submit_encoder(queue, encoder);

        let pixels = read_buffer(instance, readback, 0, READBACK_SIZE);
        let mut red_pixels = 0;
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                if pixels[off] != 0 || pixels[off + 1] != 255 {
                    red_pixels += 1;
                }
            }
        }
        assert_eq!(
            red_pixels,
            0,
            "vulkan gradient depthCompare=Equal mismatched {red_pixels} pixels (first: {:?})",
            &pixels[0..4]
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuRenderPipelineRelease(verify_pipeline);
        yawgpu::wgpuRenderPipelineRelease(write_pipeline);
        yawgpu::wgpuShaderModuleRelease(verify_module);
        yawgpu::wgpuShaderModuleRelease(write_module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Isolation — the suspected round-4 gap: stage a depth GRADIENT into **mip 2**
/// (depth-only pass, no colour attachment to supply the right extent) then
/// `depthCompare=Equal`-verify mip 2. A constant-depth mip-2 render passes because
/// it is position-independent; a GRADIENT exposes a wrong framebuffer extent
/// (base size instead of mip-2 size) — the gradient lands at the wrong pixels in
/// the mip-2 region, so Equal fails and pixels stay RED.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_depth_gradient_mip2_equal_renders_green() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    const BASE: u32 = 16;
    const MIP: u32 = 2; // 16 >> 2 = 4 == WIDTH/HEIGHT

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let depth_desc = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            dimension: native::WGPUTextureDimension_2D,
            size: native::WGPUExtent3D {
                width: BASE,
                height: BASE,
                depthOrArrayLayers: 1,
            },
            format: native::WGPUTextureFormat_Depth32Float,
            mipLevelCount: 3,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let depth = yawgpu::wgpuDeviceCreateTexture(device, &depth_desc);
        assert!(!depth.is_null());
        let color = create_color_texture(device); // 4x4, mip 0 == mip-2 size
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let write_module = create_wgsl_module(device, GRADIENT_DEPTH_SHADER);
        let write_pipeline = create_vertex_only_depth_pipeline(device, write_module);
        let verify_module = create_wgsl_module(device, GRADIENT_GREEN_SHADER);
        let verify_pipeline = create_depth_equal_pipeline(device, verify_module);
        let depth_view = create_subresource_view(
            device,
            depth,
            native::WGPUTextureFormat_Depth32Float,
            MIP,
            0,
        );
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());

        // Pass 1: stage the gradient into depth mip 2 (depth-only).
        let write_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let write_pass = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &write_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &write_pass);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, write_pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        // Pass 2: equal-verify mip 2 (colour 4x4 supplies the correct extent here).
        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: color_view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        };
        let verify_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Load,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let attachments = [color_attachment];
        let verify_pass = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: attachments.len(),
            colorAttachments: attachments.as_ptr(),
            depthStencilAttachment: &verify_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &verify_pass);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, verify_pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        record_color_t2b(encoder, color, readback);
        submit_encoder(queue, encoder);

        let pixels = read_buffer(instance, readback, 0, READBACK_SIZE);
        let mut red_pixels = 0;
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                if pixels[off] != 0 || pixels[off + 1] != 255 {
                    red_pixels += 1;
                }
            }
        }
        assert_eq!(
            red_pixels,
            0,
            "vulkan gradient mip-2 equal mismatched {red_pixels} pixels (first: {:?})",
            &pixels[0..4]
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuRenderPipelineRelease(verify_pipeline);
        yawgpu::wgpuRenderPipelineRelease(write_pipeline);
        yawgpu::wgpuShaderModuleRelease(verify_module);
        yawgpu::wgpuShaderModuleRelease(write_module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Isolation — render a constant `frag_depth` into **mip level 2** of a 16×16,
/// 3-mip `Depth32Float` texture (mip 2 = 4×4) via a mip-2 view, then read mip 2's
/// depth aspect back. The layer-1 probe passes, so this pins whether a non-zero
/// **mip** render attachment is handled (the framebuffer extent must be the
/// mip-level size, not the base).
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_depth_render_mip2_then_t2b() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    const BASE: u32 = 16;
    const MIP: u32 = 2;
    const MIP_W: u32 = BASE >> MIP; // 4
    const MIP_H: u32 = BASE >> MIP; // 4

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            dimension: native::WGPUTextureDimension_2D,
            size: native::WGPUExtent3D {
                width: BASE,
                height: BASE,
                depthOrArrayLayers: 1,
            },
            format: native::WGPUTextureFormat_Depth32Float,
            mipLevelCount: 3,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        };
        let depth = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
        assert!(!depth.is_null());
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let module = create_wgsl_module(device, FRAG_DEPTH_SHADER);
        let pipeline = create_depth_pipeline(device, module);
        let depth_view = create_subresource_view(
            device,
            depth,
            native::WGPUTextureFormat_Depth32Float,
            MIP,
            0,
        );

        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        let t2b_src = native::WGPUTexelCopyTextureInfo {
            texture: depth,
            mipLevel: MIP,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_DepthOnly,
        };
        let t2b_dst = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: BYTES_PER_ROW,
                rowsPerImage: MIP_H,
            },
            buffer: readback,
        };
        let mip_extent = native::WGPUExtent3D {
            width: MIP_W,
            height: MIP_H,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &t2b_src, &t2b_dst, &mip_extent);
        submit_encoder(queue, encoder);

        let mapped = read_buffer(instance, readback, 0, READBACK_SIZE);
        let mut mismatches = Vec::new();
        for row in 0..MIP_H as usize {
            for col in 0..MIP_W as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                let got = f32::from_le_bytes([
                    mapped[off],
                    mapped[off + 1],
                    mapped[off + 2],
                    mapped[off + 3],
                ]);
                if (got - DEPTH_VALUE).abs() > 1e-5 {
                    mismatches.push((col, row, got));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "vulkan mip-2 depth render wrong (col,row,got): {mismatches:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[cfg(feature = "vulkan")]
unsafe fn create_color_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        dimension: native::WGPUTextureDimension_2D,
        size: texture_extent(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

#[cfg(feature = "vulkan")]
unsafe fn create_depth_equal_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let depth_stencil = native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32Float,
        depthWriteEnabled: native::WGPUOptionalBool_False,
        depthCompare: native::WGPUCompareFunction_Equal,
        stencilFront: stencil_face(),
        stencilBack: stencil_face(),
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    };
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
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
    let descriptor = native::WGPURenderPipelineDescriptor {
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
        primitive: primitive_state(),
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "vulkan")]
unsafe fn record_color_t2b(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
) {
    let source = native::WGPUTexelCopyTextureInfo {
        texture: source,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let destination = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        },
        buffer: destination,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
        encoder,
        &source,
        &destination,
        &texture_extent(),
    );
}

#[cfg(feature = "vulkan")]
unsafe fn create_r32float_input(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
        dimension: native::WGPUTextureDimension_2D,
        size: texture_extent(),
        format: native::WGPUTextureFormat_R32Float,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

#[cfg(feature = "vulkan")]
unsafe fn create_texture_bind_group(
    device: native::WGPUDevice,
    pipeline: native::WGPURenderPipeline,
    view: native::WGPUTextureView,
) -> native::WGPUBindGroup {
    let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer: std::ptr::null(),
        offset: 0,
        size: 0,
        sampler: std::ptr::null(),
        textureView: view,
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
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    bind_group
}

/// Isolation — the F-031 t2t path itself: render a constant `frag_depth` into a
/// source `Depth32Float`, `copyTextureToTexture` the depth aspect to a
/// destination, then read the destination's depth aspect back. Render+t2b are
/// proven by the other probes, so this pins the **depth t2t copy** on Vulkan.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_depth_copy_texture_to_texture_then_t2b() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let source = create_depth_texture(device);
        let destination = create_depth_texture(device);
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let module = create_wgsl_module(device, FRAG_DEPTH_SHADER);
        let pipeline = create_depth_pipeline(device, module);
        let source_view = yawgpu::wgpuTextureCreateView(source, std::ptr::null());

        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: source_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        // copyTextureToTexture, depth aspect (origin/whole-subresource).
        let t2t_src = native::WGPUTexelCopyTextureInfo {
            texture: source,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let t2t_dst = native::WGPUTexelCopyTextureInfo {
            texture: destination,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToTexture(
            encoder,
            &t2t_src,
            &t2t_dst,
            &texture_extent(),
        );
        record_depth_t2b(encoder, destination, readback);
        submit_encoder(queue, encoder);

        let depths = read_depth_buffer(instance, readback);
        assert!(
            depths.iter().all(|d| (d - DEPTH_VALUE).abs() < 1e-5),
            "vulkan depth not preserved through copyTextureToTexture: {depths:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuTextureViewRelease(source_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(destination);
        yawgpu::wgpuTextureRelease(source);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[cfg(feature = "vulkan")]
unsafe fn create_depth_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment
            | native::WGPUTextureUsage_CopySrc
            | native::WGPUTextureUsage_CopyDst,
        dimension: native::WGPUTextureDimension_2D,
        size: texture_extent(),
        format: native::WGPUTextureFormat_Depth32Float,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

#[cfg(feature = "vulkan")]
unsafe fn create_depth_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let depth_stencil = native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32Float,
        depthWriteEnabled: native::WGPUOptionalBool_True,
        depthCompare: native::WGPUCompareFunction_Always,
        stencilFront: stencil_face(),
        stencilBack: stencil_face(),
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 0,
        targets: std::ptr::null(),
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
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
        primitive: primitive_state(),
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "vulkan")]
unsafe fn record_depth_t2b(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
) {
    let source = native::WGPUTexelCopyTextureInfo {
        texture: source,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_DepthOnly,
    };
    let destination = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        },
        buffer: destination,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
        encoder,
        &source,
        &destination,
        &texture_extent(),
    );
}

#[cfg(feature = "vulkan")]
unsafe fn read_depth_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
) -> Vec<f32> {
    let mapped = read_buffer(instance, buffer, 0, READBACK_SIZE);
    let mut depths = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for row in 0..HEIGHT as usize {
        let row_offset = row * BYTES_PER_ROW as usize;
        for col in 0..WIDTH as usize {
            let off = row_offset + col * 4;
            depths.push(f32::from_le_bytes([
                mapped[off],
                mapped[off + 1],
                mapped[off + 2],
                mapped[off + 3],
            ]));
        }
    }
    depths
}

fn stencil_face() -> native::WGPUStencilFaceState {
    native::WGPUStencilFaceState {
        compare: native::WGPUCompareFunction_Always,
        failOp: native::WGPUStencilOperation_Keep,
        depthFailOp: native::WGPUStencilOperation_Keep,
        passOp: native::WGPUStencilOperation_Keep,
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

fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
    }
}

#[cfg(feature = "vulkan")]
unsafe fn submit_encoder(queue: native::WGPUQueue, encoder: native::WGPUCommandEncoder) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

#[cfg(feature = "vulkan")]
unsafe fn create_buffer(
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

#[cfg(feature = "vulkan")]
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

#[cfg(feature = "vulkan")]
unsafe fn read_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    offset: u64,
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
    let future = yawgpu::wgpuBufferMapAsync(
        buffer,
        native::WGPUMapMode_Read,
        usize::try_from(offset).expect("test offset fits in usize"),
        len,
        callback_info,
    );
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);

    let ptr = yawgpu::wgpuBufferGetConstMappedRange(
        buffer,
        usize::try_from(offset).expect("test offset fits in usize"),
        len,
    );
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

#[cfg(feature = "vulkan")]
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

#[cfg(feature = "vulkan")]
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

#[cfg(feature = "vulkan")]
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

#[cfg(feature = "vulkan")]
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
