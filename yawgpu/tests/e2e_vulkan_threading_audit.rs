//! Real-GPU Vulkan/MoltenVK verification for the threading audit: WebGPU state
//! fields that `yawgpu-core` validated but never threaded to the HAL, so the GPU
//! used a default and the result was silently wrong. This is the Vulkan mirror of
//! `e2e_metal_threading_audit` and asserts the SAME WebGPU pipelines produce the
//! SAME GPU effect on both Tier-1 backends. Each probe sets a NON-default value
//! and reads back the effect that only appears once the field reaches the backend:
//!
//! * `primitive.cullMode` + `primitive.frontFace` — a CCW full-screen triangle is
//!   drawn with `cull=Back, frontFace=CCW` (front → kept) and culled with
//!   `cull=Back, frontFace=CW` (now back → discarded). Identical to the Metal
//!   result pins WebGPU↔Vulkan Y/winding parity (Vulkan previously hard-coded
//!   `cull=NONE, front=CCW`).
//! * `setScissorRect` / `setViewport` — restrict rendering to a sub-rectangle; the
//!   excluded corner keeps the clear colour.
//! * `depthReadOnly` — a read-only depth pass must LOAD+preserve the pre-rendered
//!   depth, not clear it (the pre-fix mapping forced loadOp=Clear).
//! * `dispatchWorkgroupsIndirect` — was a validation-only no-op (recorded no HAL
//!   command); the indirect compute must now actually run its storage write.
//!
//! Gated on the `vulkan` feature; `#[ignore]`d (manual real-backend run).
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

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const BYTES_PER_ROW: u32 = 256;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;
const GREEN: [u8; 4] = [0, 255, 0, 255];
const RED: [u8; 4] = [255, 0, 0, 255];

// Oversized triangle covering the whole framebuffer. In WebGPU NDC (Y up) the
// vertex order (-1,-1),(3,-1),(-1,3) is counter-clockwise, so frontFace=CCW makes
// it front-facing. The fragment writes storage=1 (so culling => storage stays 0)
// and outputs green.
const TRI_SHADER: &str = r#"
struct Result { value: u32 }
@group(0) @binding(0) var<storage, read_write> result: Result;
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(pos[idx], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    result.value = 1u;
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
"#;

// Full-screen triangle at constant depth z=0.4, no storage; green fragment.
const DEPTH_TRI_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(pos[idx], 0.4, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
"#;

const COMPUTE_SHADER: &str = r#"
struct Result { value: u32 }
@group(0) @binding(0) var<storage, read_write> result: Result;
@compute @workgroup_size(1)
fn main() {
    result.value = 1u;
}
"#;

/// `cull=Back, frontFace=CCW`: the CCW triangle is front-facing → kept → green.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_cull_back_keeps_front_facing_ccw() {
    let (storage, pixel) = run_cull(native::WGPUFrontFace_CCW);
    assert_eq!(storage, 1, "front-facing draw was wrongly culled");
    assert_eq!(pixel, GREEN, "front-facing draw did not rasterize green");
}

/// `cull=Back, frontFace=CW`: the same CCW triangle is now back-facing → culled.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_cull_back_discards_back_facing_cw() {
    let (storage, pixel) = run_cull(native::WGPUFrontFace_CW);
    assert_eq!(
        storage, 0,
        "back-facing draw was not culled (cull not threaded)"
    );
    assert_eq!(pixel, RED, "culled draw still rasterized");
}

/// `setScissorRect` must clip rendering to the rectangle.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_scissor_rect_restricts_rendering() {
    let pixels = run_region(Region::Scissor);
    assert_eq!(
        read_pixel(&pixels, 0, 0),
        RED,
        "pixel outside scissor was drawn"
    );
    assert_eq!(
        read_pixel(&pixels, 3, 3),
        GREEN,
        "pixel inside scissor was not drawn"
    );
}

/// `setViewport` must remap NDC into the viewport sub-rectangle.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_viewport_restricts_rendering() {
    let pixels = run_region(Region::Viewport);
    assert_eq!(
        read_pixel(&pixels, 0, 0),
        RED,
        "pixel outside viewport was drawn"
    );
    assert_eq!(
        read_pixel(&pixels, 3, 3),
        GREEN,
        "pixel inside viewport was not drawn"
    );
}

/// A read-only depth pass must preserve the pre-rendered depth (0.5), so a
/// `depthCompare=Less` draw at z=0.4 passes (green). The pre-fix bug cleared the
/// read-only depth to 0.0, which would fail the test (red).
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_read_only_depth_preserves_contents() {
    let pixel = run_read_only_depth();
    assert_eq!(
        pixel, GREEN,
        "read-only depth was cleared instead of preserved (z=0.4 < preserved 0.5 should pass)"
    );
}

/// `dispatchWorkgroupsIndirect` must execute the storage write (was a no-op).
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_dispatch_workgroups_indirect_executes() {
    let value = run_indirect_dispatch();
    assert_eq!(
        value, 1,
        "indirect dispatch did not run (no HAL command recorded)"
    );
}

// ---- runners ----------------------------------------------------------------

fn run_cull(front_face: native::WGPUFrontFace) -> (u32, [u8; 4]) {
    let mut out = (0u32, RED);
    with_device(|ctx| unsafe {
        let storage = ctx.storage_buffer();
        let color = ctx.color_texture();
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());
        let module = create_wgsl_module(ctx.device, TRI_SHADER);
        let pipeline =
            create_raster_pipeline(ctx.device, module, front_face, native::WGPUCullMode_Back);
        let bind_group = create_storage_bind_group(ctx.device, pipeline, storage);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = begin_color_pass(encoder, color_view);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        let storage_readback = ctx.readback_buffer(4);
        let color_readback = ctx.readback_buffer(READBACK_SIZE as u64);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, storage_readback, 0, 4);
        copy_color(encoder, color, color_readback);
        ctx.submit(encoder);

        let s = read_buffer(ctx.instance, storage_readback, 4);
        let pixels = read_buffer(ctx.instance, color_readback, READBACK_SIZE);
        out = (
            u32::from_le_bytes([s[0], s[1], s[2], s[3]]),
            read_pixel(&pixels, 0, 0),
        );

        yawgpu::wgpuBufferRelease(color_readback);
        yawgpu::wgpuBufferRelease(storage_readback);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuBufferRelease(storage);
    });
    out
}

#[derive(Clone, Copy)]
enum Region {
    Scissor,
    Viewport,
}

fn run_region(region: Region) -> Vec<u8> {
    let mut out = Vec::new();
    with_device(|ctx| unsafe {
        let storage = ctx.storage_buffer();
        let color = ctx.color_texture();
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());
        let module = create_wgsl_module(ctx.device, TRI_SHADER);
        let pipeline = create_raster_pipeline(
            ctx.device,
            module,
            native::WGPUFrontFace_CCW,
            native::WGPUCullMode_None,
        );
        let bind_group = create_storage_bind_group(ctx.device, pipeline, storage);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = begin_color_pass(encoder, color_view);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        // Restrict to the bottom-right 2x2 quadrant (x>=2, y>=2).
        match region {
            Region::Scissor => yawgpu::wgpuRenderPassEncoderSetScissorRect(pass, 2, 2, 2, 2),
            Region::Viewport => {
                yawgpu::wgpuRenderPassEncoderSetViewport(pass, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0)
            }
        }
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        let color_readback = ctx.readback_buffer(READBACK_SIZE as u64);
        copy_color(encoder, color, color_readback);
        ctx.submit(encoder);
        out = read_buffer(ctx.instance, color_readback, READBACK_SIZE);

        yawgpu::wgpuBufferRelease(color_readback);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuBufferRelease(storage);
    });
    out
}

fn run_read_only_depth() -> [u8; 4] {
    let mut out = RED;
    with_device(|ctx| unsafe {
        let color = ctx.color_texture();
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());
        let depth = ctx.depth_texture();
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());
        let module = create_wgsl_module(ctx.device, DEPTH_TRI_SHADER);
        let pipeline = create_depth_test_pipeline(ctx.device, module);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        // Pass 1: clear depth to 0.5 and store it (no color, no draws).
        let prime = begin_depth_only_clear_pass(encoder, depth_view, 0.5);
        yawgpu::wgpuRenderPassEncoderEnd(prime);
        yawgpu::wgpuRenderPassEncoderRelease(prime);
        // Pass 2: read-only depth + depthCompare=Less at z=0.4 -> passes iff 0.5 preserved.
        let pass = begin_read_only_depth_pass(encoder, color_view, depth_view);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        let color_readback = ctx.readback_buffer(READBACK_SIZE as u64);
        copy_color(encoder, color, color_readback);
        ctx.submit(encoder);
        let pixels = read_buffer(ctx.instance, color_readback, READBACK_SIZE);
        out = read_pixel(&pixels, 0, 0);

        yawgpu::wgpuBufferRelease(color_readback);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureRelease(color);
    });
    out
}

fn run_indirect_dispatch() -> u32 {
    let mut out = 0u32;
    with_device(|ctx| unsafe {
        let storage = ctx.storage_buffer();
        let indirect_args: [u32; 3] = [1, 1, 1];
        let indirect = create_buffer(
            ctx.device,
            std::mem::size_of_val(&indirect_args) as u64,
            native::WGPUBufferUsage_Indirect | native::WGPUBufferUsage_CopyDst,
        );
        yawgpu::wgpuQueueWriteBuffer(
            ctx.queue,
            indirect,
            0,
            indirect_args.as_ptr().cast(),
            std::mem::size_of_val(&indirect_args),
        );
        let module = create_wgsl_module(ctx.device, COMPUTE_SHADER);
        let pipeline = create_compute_pipeline(ctx.device, module);
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let bind_group = create_storage_bind_group_with_layout(ctx.device, layout, storage);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, indirect, 0);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);

        let readback = ctx.readback_buffer(4);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, readback, 0, 4);
        ctx.submit(encoder);
        let s = read_buffer(ctx.instance, readback, 4);
        out = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(indirect);
        yawgpu::wgpuBufferRelease(storage);
    });
    out
}

// ---- device context + shared helpers ----------------------------------------

struct Ctx {
    instance: native::WGPUInstance,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    errors: Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
}

impl Ctx {
    unsafe fn storage_buffer(&self) -> native::WGPUBuffer {
        let buffer = create_buffer(
            self.device,
            4,
            native::WGPUBufferUsage_Storage
                | native::WGPUBufferUsage_CopySrc
                | native::WGPUBufferUsage_CopyDst,
        );
        let zero = [0u8; 4];
        yawgpu::wgpuQueueWriteBuffer(self.queue, buffer, 0, zero.as_ptr().cast(), zero.len());
        buffer
    }

    unsafe fn color_texture(&self) -> native::WGPUTexture {
        create_texture(
            self.device,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        )
    }

    unsafe fn depth_texture(&self) -> native::WGPUTexture {
        create_texture(
            self.device,
            native::WGPUTextureFormat_Depth32Float,
            native::WGPUTextureUsage_RenderAttachment,
        )
    }

    unsafe fn readback_buffer(&self, size: u64) -> native::WGPUBuffer {
        create_buffer(
            self.device,
            size,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        )
    }

    unsafe fn submit(&self, encoder: native::WGPUCommandEncoder) {
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(self.queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        assert!(
            self.errors.lock().expect("error lock").is_empty(),
            "device errors: {:?}",
            self.errors.lock().expect("error lock")
        );
    }
}

fn with_device(body: impl FnOnce(&Ctx)) {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let ctx = Ctx {
            instance,
            device,
            queue,
            errors,
        };
        body(&ctx);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

fn read_pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let o = (y * BYTES_PER_ROW + x * 4) as usize;
    [pixels[o], pixels[o + 1], pixels[o + 2], pixels[o + 3]]
}

unsafe fn begin_color_pass(
    encoder: native::WGPUCommandEncoder,
    view: native::WGPUTextureView,
) -> native::WGPURenderPassEncoder {
    let color_attachment = native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
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
    let attachments = [color_attachment];
    let descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor)
}

unsafe fn begin_depth_only_clear_pass(
    encoder: native::WGPUCommandEncoder,
    depth_view: native::WGPUTextureView,
    clear: f32,
) -> native::WGPURenderPassEncoder {
    let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
        nextInChain: std::ptr::null_mut(),
        view: depth_view,
        depthLoadOp: native::WGPULoadOp_Clear,
        depthStoreOp: native::WGPUStoreOp_Store,
        depthClearValue: clear,
        depthReadOnly: false.into(),
        stencilLoadOp: native::WGPULoadOp_Undefined,
        stencilStoreOp: native::WGPUStoreOp_Undefined,
        stencilClearValue: 0,
        stencilReadOnly: false.into(),
    };
    let descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: 0,
        colorAttachments: std::ptr::null(),
        depthStencilAttachment: &depth_attachment,
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor)
}

unsafe fn begin_read_only_depth_pass(
    encoder: native::WGPUCommandEncoder,
    color_view: native::WGPUTextureView,
    depth_view: native::WGPUTextureView,
) -> native::WGPURenderPassEncoder {
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
    let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
        nextInChain: std::ptr::null_mut(),
        view: depth_view,
        depthLoadOp: native::WGPULoadOp_Undefined,
        depthStoreOp: native::WGPUStoreOp_Undefined,
        depthClearValue: 0.0,
        depthReadOnly: true.into(),
        stencilLoadOp: native::WGPULoadOp_Undefined,
        stencilStoreOp: native::WGPUStoreOp_Undefined,
        stencilClearValue: 0,
        stencilReadOnly: false.into(),
    };
    let attachments = [color_attachment];
    let descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: &depth_attachment,
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor)
}

unsafe fn copy_color(
    encoder: native::WGPUCommandEncoder,
    color: native::WGPUTexture,
    readback: native::WGPUBuffer,
) {
    let src = native::WGPUTexelCopyTextureInfo {
        texture: color,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let dst = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        },
        buffer: readback,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &texture_extent());
}

unsafe fn create_raster_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    front_face: native::WGPUFrontFace,
    cull_mode: native::WGPUCullMode,
) -> native::WGPURenderPipeline {
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
        primitive: native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: front_face,
            cullMode: cull_mode,
            unclippedDepth: 0,
        },
        depthStencil: std::ptr::null(),
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

unsafe fn create_depth_test_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
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
    let depth_stencil = native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32Float,
        depthWriteEnabled: native::WGPUOptionalBool_False,
        depthCompare: native::WGPUCompareFunction_Less,
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
        primitive: native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: native::WGPUFrontFace_CCW,
            cullMode: native::WGPUCullMode_None,
            unclippedDepth: 0,
        },
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

fn stencil_face() -> native::WGPUStencilFaceState {
    native::WGPUStencilFaceState {
        compare: native::WGPUCompareFunction_Always,
        failOp: native::WGPUStencilOperation_Keep,
        depthFailOp: native::WGPUStencilOperation_Keep,
        passOp: native::WGPUStencilOperation_Keep,
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

unsafe fn create_storage_bind_group(
    device: native::WGPUDevice,
    pipeline: native::WGPURenderPipeline,
    storage: native::WGPUBuffer,
) -> native::WGPUBindGroup {
    let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
    let bind_group = create_storage_bind_group_with_layout(device, layout, storage);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    bind_group
}

unsafe fn create_storage_bind_group_with_layout(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    storage: native::WGPUBuffer,
) -> native::WGPUBindGroup {
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer: storage,
        offset: 0,
        size: 4,
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
        size: texture_extent(),
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

fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
    }
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
