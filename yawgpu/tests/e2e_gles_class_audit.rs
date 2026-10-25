#![cfg(feature = "gles")]
//! Real-GPU GLES coverage for the GLES rows of Block 111 (R8–R11,
//! `specs/blocks/111-code-review-hal-majors.md`) and Block 115 (F5, F6, F8,
//! `specs/blocks/115-class-audit-metal-gles-tiled.md`). Each case is shaped
//! so the pre-fix behaviour produces different pixels, not just an error.

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_test::{real_backend_available, wait, RealBackend};

mod common;
use common::create_gles_instance;

const BYTES_PER_ROW: u32 = 256;
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];

const SHADER: &str = r#"
@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[i], 0.0, 1.0);
}

// Covers the right half of the target (clip x in [0, 1]).
@vertex
fn vs_right(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array(
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs_red() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_half_red() -> @location(0) vec4<f32> {
    return vec4<f32>(0.5, 0.0, 0.0, 1.0);
}
"#;

/// R8: the default viewport of a pass rendering into mip 1 is the mip's
/// extent. With the mip-0 extent the right-half quad lands outside the
/// 8x8 attachment and nothing is drawn.
#[test]
#[ignore = "manual real-backend test"]
fn gles_render_into_mip_one_uses_mip_extent_viewport() {
    with_gles_device(|ctx| unsafe {
        let texture = create_texture(
            ctx.device,
            &texture_descriptor(
                native::WGPUTextureFormat_RGBA8Unorm,
                extent(16, 16, 1),
                2,
                1,
                native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            ),
        );
        let view = create_view(texture, native::WGPUTextureFormat_Undefined, 1);
        let module = create_wgsl_module(ctx.device, SHADER);
        let pipeline = create_pipeline(
            ctx.device,
            module,
            &raster("vs_right", "fs_red", native::WGPUTextureFormat_RGBA8Unorm),
        );
        let readback = create_readback(ctx.device, 8);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let attachments = [color_attachment(
            view,
            std::ptr::null(),
            [0.0, 0.0, 1.0, 1.0],
        )];
        let pass = begin_pass(encoder, &attachments, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        end_pass(pass);
        record_t2b(encoder, texture, 1, extent(8, 8, 1), readback);
        ctx.submit(encoder);

        let bytes = read_buffer(ctx.instance, readback, readback_len(8));
        assert_halves(&bytes, 8, 8, BLUE, RED);

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    });
}

/// R8: an 8x8 MSAA attachment resolves into mip 1 (8x8) of a 16x16 texture.
/// With the resolve texture's mip-0 extent as the blit destination rect the
/// blit magnifies the source's left (blue) half over the whole mip.
#[test]
#[ignore = "manual real-backend test"]
fn gles_msaa_resolve_into_mip_one_uses_mip_extent_rect() {
    with_gles_device(|ctx| unsafe {
        let msaa = create_texture(
            ctx.device,
            &texture_descriptor(
                native::WGPUTextureFormat_RGBA8Unorm,
                extent(8, 8, 1),
                1,
                4,
                native::WGPUTextureUsage_RenderAttachment,
            ),
        );
        let msaa_view = yawgpu::wgpuTextureCreateView(msaa, std::ptr::null());
        let resolve = create_texture(
            ctx.device,
            &texture_descriptor(
                native::WGPUTextureFormat_RGBA8Unorm,
                extent(16, 16, 1),
                2,
                1,
                native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            ),
        );
        let resolve_view = create_view(resolve, native::WGPUTextureFormat_Undefined, 1);
        let module = create_wgsl_module(ctx.device, SHADER);
        let mut desc = raster("vs_right", "fs_red", native::WGPUTextureFormat_RGBA8Unorm);
        desc.samples = 4;
        let pipeline = create_pipeline(ctx.device, module, &desc);
        let readback = create_readback(ctx.device, 8);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let mut attachment = color_attachment(msaa_view, resolve_view, [0.0, 0.0, 1.0, 1.0]);
        attachment.storeOp = native::WGPUStoreOp_Discard;
        let pass = begin_pass(encoder, &[attachment], std::ptr::null());
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        end_pass(pass);
        record_t2b(encoder, resolve, 1, extent(8, 8, 1), readback);
        ctx.submit(encoder);

        let bytes = read_buffer(ctx.instance, readback, readback_len(8));
        assert_halves(&bytes, 8, 8, BLUE, RED);

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(resolve_view);
        yawgpu::wgpuTextureRelease(resolve);
        yawgpu::wgpuTextureViewRelease(msaa_view);
        yawgpu::wgpuTextureRelease(msaa);
    });
}

/// R9: the scissor rect of the last draw must not clip the resolve blit.
/// A clipped blit leaves everything outside the 4x4 scissor unresolved.
#[test]
#[ignore = "manual real-backend test"]
fn gles_msaa_resolve_is_not_clipped_by_draw_scissor() {
    with_gles_device(|ctx| unsafe {
        let msaa = create_texture(
            ctx.device,
            &texture_descriptor(
                native::WGPUTextureFormat_RGBA8Unorm,
                extent(16, 16, 1),
                1,
                4,
                native::WGPUTextureUsage_RenderAttachment,
            ),
        );
        let msaa_view = yawgpu::wgpuTextureCreateView(msaa, std::ptr::null());
        let resolve = create_texture(
            ctx.device,
            &texture_descriptor(
                native::WGPUTextureFormat_RGBA8Unorm,
                extent(16, 16, 1),
                1,
                1,
                native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            ),
        );
        let resolve_view = yawgpu::wgpuTextureCreateView(resolve, std::ptr::null());
        let module = create_wgsl_module(ctx.device, SHADER);
        let mut desc = raster("vs_full", "fs_red", native::WGPUTextureFormat_RGBA8Unorm);
        desc.samples = 4;
        let pipeline = create_pipeline(ctx.device, module, &desc);
        let readback = create_readback(ctx.device, 16);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let mut attachment = color_attachment(msaa_view, resolve_view, [0.0, 0.0, 1.0, 1.0]);
        attachment.storeOp = native::WGPUStoreOp_Discard;
        let pass = begin_pass(encoder, &[attachment], std::ptr::null());
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, 4, 4);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        end_pass(pass);
        record_t2b(encoder, resolve, 0, extent(16, 16, 1), readback);
        ctx.submit(encoder);

        let bytes = read_buffer(ctx.instance, readback, readback_len(16));
        assert_scissored(&bytes, 16, 4, RED, BLUE);

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(resolve_view);
        yawgpu::wgpuTextureRelease(resolve);
        yawgpu::wgpuTextureViewRelease(msaa_view);
        yawgpu::wgpuTextureRelease(msaa);
    });
}

/// R9: a scissor rect set in one pass must not leak into the next pass's
/// load-op clear (GL clears obey the scissor test). Regression guard only:
/// pass begin, resolve, lazy-init clears and present each disable the
/// scissor test themselves, so removing the `RenderPassCleanup` reset alone
/// does not turn this red (checked on NVIDIA EGL, 2026-10-25).
#[test]
#[ignore = "manual real-backend test"]
fn gles_scissor_does_not_clip_next_pass_clear() {
    with_gles_device(|ctx| unsafe {
        let usage = native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc;
        let format = native::WGPUTextureFormat_RGBA8Unorm;
        let first = create_texture(
            ctx.device,
            &texture_descriptor(format, extent(16, 16, 1), 1, 1, usage),
        );
        let second = create_texture(
            ctx.device,
            &texture_descriptor(format, extent(16, 16, 1), 1, 1, usage),
        );
        let first_view = yawgpu::wgpuTextureCreateView(first, std::ptr::null());
        let second_view = yawgpu::wgpuTextureCreateView(second, std::ptr::null());
        let module = create_wgsl_module(ctx.device, SHADER);
        let pipeline = create_pipeline(ctx.device, module, &raster("vs_full", "fs_red", format));
        let first_readback = create_readback(ctx.device, 16);
        let second_readback = create_readback(ctx.device, 16);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = begin_pass(
            encoder,
            &[color_attachment(
                first_view,
                std::ptr::null(),
                [0.0, 0.0, 1.0, 1.0],
            )],
            std::ptr::null(),
        );
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, 4, 4);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        end_pass(pass);
        let pass = begin_pass(
            encoder,
            &[color_attachment(
                second_view,
                std::ptr::null(),
                [0.0, 1.0, 0.0, 1.0],
            )],
            std::ptr::null(),
        );
        end_pass(pass);
        record_t2b(encoder, first, 0, extent(16, 16, 1), first_readback);
        record_t2b(encoder, second, 0, extent(16, 16, 1), second_readback);
        ctx.submit(encoder);

        let first_bytes = read_buffer(ctx.instance, first_readback, readback_len(16));
        assert_scissored(&first_bytes, 16, 4, RED, BLUE);
        let second_bytes = read_buffer(ctx.instance, second_readback, readback_len(16));
        assert_scissored(&second_bytes, 16, 0, GREEN, GREEN);

        yawgpu::wgpuBufferRelease(second_readback);
        yawgpu::wgpuBufferRelease(first_readback);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(second_view);
        yawgpu::wgpuTextureViewRelease(first_view);
        yawgpu::wgpuTextureRelease(second);
        yawgpu::wgpuTextureRelease(first);
    });
}

/// R10: a `0xFFFF` index restarts a uint16 triangle strip. The vertex
/// buffer really holds 65536 vertices, with vertex 0xFFFF at the top
/// centre, so without restart the bridging triangles (2,3,X), (3,X,4),
/// (X,4,5) visibly fill the gap between the two quads.
#[test]
#[ignore = "manual real-backend test"]
fn gles_uint16_triangle_strip_restarts_at_all_ones_index() {
    with_gles_device(|ctx| unsafe {
        let shader = r#"
@vertex
fn vs(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 0.0, 1.0);
}

@fragment
fn fs_red() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;
        let mut vertices = vec![0.0f32; 65536 * 2];
        #[rustfmt::skip]
        let quads: [f32; 16] = [
            -1.0, -1.0, -1.0, 1.0, -0.5, -1.0, -0.5, 1.0, // left quad, v0..v3
             0.5, -1.0,  0.5, 1.0,  1.0, -1.0,  1.0, 1.0, // right quad, v4..v7
        ];
        vertices[..16].copy_from_slice(&quads);
        vertices[65535 * 2..].copy_from_slice(&[0.0, 1.0]);
        let vertex_size = std::mem::size_of_val(vertices.as_slice()) as u64;
        let vertex = create_buffer(
            ctx.device,
            vertex_size,
            native::WGPUBufferUsage_Vertex | native::WGPUBufferUsage_CopyDst,
        );
        yawgpu::wgpuQueueWriteBuffer(
            ctx.queue,
            vertex,
            0,
            vertices.as_ptr().cast(),
            vertex_size as usize,
        );
        // 9 indices, padded to a 4-byte multiple for writeBuffer.
        let indices: [u16; 10] = [0, 1, 2, 3, 0xFFFF, 4, 5, 6, 7, 0];
        let index = create_buffer(
            ctx.device,
            20,
            native::WGPUBufferUsage_Index | native::WGPUBufferUsage_CopyDst,
        );
        yawgpu::wgpuQueueWriteBuffer(ctx.queue, index, 0, indices.as_ptr().cast(), 20);

        let format = native::WGPUTextureFormat_RGBA8Unorm;
        let target = create_texture(
            ctx.device,
            &texture_descriptor(
                format,
                extent(16, 16, 1),
                1,
                1,
                native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            ),
        );
        let view = yawgpu::wgpuTextureCreateView(target, std::ptr::null());
        let module = create_wgsl_module(ctx.device, shader);
        let attribute = native::WGPUVertexAttribute {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUVertexFormat_Float32x2,
            offset: 0,
            shaderLocation: 0,
        };
        let layout = native::WGPUVertexBufferLayout {
            nextInChain: std::ptr::null_mut(),
            arrayStride: 8,
            stepMode: native::WGPUVertexStepMode_Vertex,
            attributeCount: 1,
            attributes: &attribute,
        };
        let mut desc = raster("vs", "fs_red", format);
        desc.primitive.topology = native::WGPUPrimitiveTopology_TriangleStrip;
        desc.primitive.stripIndexFormat = native::WGPUIndexFormat_Uint16;
        desc.buffers = std::slice::from_ref(&layout);
        let pipeline = create_pipeline(ctx.device, module, &desc);
        let readback = create_readback(ctx.device, 16);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = begin_pass(
            encoder,
            &[color_attachment(
                view,
                std::ptr::null(),
                [0.0, 0.0, 1.0, 1.0],
            )],
            std::ptr::null(),
        );
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 0, vertex_size);
        yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
            pass,
            index,
            native::WGPUIndexFormat_Uint16,
            0,
            20,
        );
        yawgpu::wgpuRenderPassEncoderDrawIndexed(pass, 9, 1, 0, 0, 0);
        end_pass(pass);
        record_t2b(encoder, target, 0, extent(16, 16, 1), readback);
        ctx.submit(encoder);

        let bytes = read_buffer(ctx.instance, readback, readback_len(16));
        for y in 0..16 {
            for x in 0..16 {
                let expected = if (4..12).contains(&x) { BLUE } else { RED };
                assert_eq!(
                    pixel(&bytes, 16, x, y, 0),
                    expected,
                    "texel ({x}, {y}): the strip must restart at 0xFFFF"
                );
            }
        }

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(target);
        yawgpu::wgpuBufferRelease(index);
        yawgpu::wgpuBufferRelease(vertex);
    });
}

/// R11: a buffer->texture copy into the stencil aspect of a combined
/// depth-stencil texture is valid WebGPU but has no clean GLES mapping; the
/// HAL rejects it, surfaced as an Internal (not Validation) device error.
#[test]
#[ignore = "manual real-backend test"]
fn gles_stencil_aspect_upload_into_depth24plus_stencil8_is_a_hal_error() {
    with_gles_device(|ctx| unsafe {
        let texture = create_texture(
            ctx.device,
            &texture_descriptor(
                native::WGPUTextureFormat_Depth24PlusStencil8,
                extent(16, 16, 1),
                1,
                1,
                native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopyDst,
            ),
        );
        let source = create_buffer(
            ctx.device,
            readback_len(16) as u64,
            native::WGPUBufferUsage_CopySrc,
        );
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let destination = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_StencilOnly,
        };
        yawgpu::wgpuCommandEncoderCopyBufferToTexture(
            encoder,
            &buffer_copy_info(source, 16),
            &destination,
            &extent(16, 16, 1),
        );
        let commands = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        ctx.assert_clean();
        yawgpu::wgpuQueueSubmit(ctx.queue, 1, &commands);
        yawgpu::wgpuCommandBufferRelease(commands);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let errors = std::mem::take(&mut *ctx.errors.lock().expect("error lock"));
        assert!(
            errors
                .iter()
                .any(|error| error.kind == yawgpu_core::ErrorKind::Internal
                    && error.message.contains("stencil")),
            "expected the GLES stencil-upload HalError, got {errors:?}"
        );
        assert!(
            errors
                .iter()
                .all(|error| error.kind != yawgpu_core::ErrorKind::Validation),
            "the copy is valid WebGPU; core must not reject it: {errors:?}"
        );

        yawgpu::wgpuBufferRelease(source);
        yawgpu::wgpuTextureRelease(texture);
    });
}

/// R11 (allowed side): `Stencil8` uploads still work. Left half of the
/// stencil is uploaded as 5, right half as 0; an `Equal 5` stencil test
/// then draws red only over the left half.
#[test]
#[ignore = "manual real-backend test"]
fn gles_stencil8_upload_drives_stencil_test() {
    with_gles_device(|ctx| unsafe {
        let stencil = create_texture(
            ctx.device,
            &texture_descriptor(
                native::WGPUTextureFormat_Stencil8,
                extent(16, 16, 1),
                1,
                1,
                native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopyDst,
            ),
        );
        let mut values = vec![0u8; readback_len(16)];
        for row in values.chunks_mut(BYTES_PER_ROW as usize) {
            row[..8].fill(5);
        }
        let destination = native::WGPUTexelCopyTextureInfo {
            texture: stencil,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let layout = buffer_copy_info(std::ptr::null(), 16).layout;
        yawgpu::wgpuQueueWriteTexture(
            ctx.queue,
            &destination,
            values.as_ptr().cast(),
            values.len(),
            &layout,
            &extent(16, 16, 1),
        );
        let stencil_view = yawgpu::wgpuTextureCreateView(stencil, std::ptr::null());

        let format = native::WGPUTextureFormat_RGBA8Unorm;
        let color = create_texture(
            ctx.device,
            &texture_descriptor(
                format,
                extent(16, 16, 1),
                1,
                1,
                native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            ),
        );
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());
        let module = create_wgsl_module(ctx.device, SHADER);
        let equal = native::WGPUStencilFaceState {
            compare: native::WGPUCompareFunction_Equal,
            failOp: native::WGPUStencilOperation_Keep,
            depthFailOp: native::WGPUStencilOperation_Keep,
            passOp: native::WGPUStencilOperation_Keep,
        };
        let depth_stencil = native::WGPUDepthStencilState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_Stencil8,
            depthWriteEnabled: native::WGPUOptionalBool_Undefined,
            depthCompare: native::WGPUCompareFunction_Undefined,
            stencilFront: equal,
            stencilBack: equal,
            stencilReadMask: 0xFF,
            stencilWriteMask: 0,
            depthBias: 0,
            depthBiasSlopeScale: 0.0,
            depthBiasClamp: 0.0,
        };
        let mut desc = raster("vs_full", "fs_red", format);
        desc.depth_stencil = &depth_stencil;
        let pipeline = create_pipeline(ctx.device, module, &desc);
        let readback = create_readback(ctx.device, 16);

        let stencil_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: stencil_view,
            depthLoadOp: native::WGPULoadOp_Undefined,
            depthStoreOp: native::WGPUStoreOp_Undefined,
            depthClearValue: f32::NAN,
            depthReadOnly: 0,
            stencilLoadOp: native::WGPULoadOp_Load,
            stencilStoreOp: native::WGPUStoreOp_Store,
            stencilClearValue: 0,
            stencilReadOnly: 0,
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = begin_pass(
            encoder,
            &[color_attachment(
                color_view,
                std::ptr::null(),
                [0.0, 0.0, 1.0, 1.0],
            )],
            &stencil_attachment,
        );
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetStencilReference(pass, 5);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        end_pass(pass);
        record_t2b(encoder, color, 0, extent(16, 16, 1), readback);
        ctx.submit(encoder);

        let bytes = read_buffer(ctx.instance, readback, readback_len(16));
        assert_halves(&bytes, 16, 16, RED, BLUE);

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuTextureViewRelease(stencil_view);
        yawgpu::wgpuTextureRelease(stencil);
    });
}

/// F5: a compute shader writes every texel of a 4x4x4 `texture_storage_3d`.
/// The pre-fix binding (`layered = FALSE`, layer 0) is undefined per the GL
/// spec for an `image3D`; NVIDIA's EGL driver still writes every slice, so
/// this does not go red there (checked 2026-10-25); the binding itself is
/// covered by the HAL unit test.
#[test]
#[ignore = "manual real-backend test"]
fn gles_3d_storage_texture_writes_every_depth_slice() {
    with_gles_device(|ctx| unsafe {
        let shader = r#"
@group(0) @binding(0) var volume: texture_storage_3d<rgba8unorm, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    textureStore(volume, id, vec4<f32>(vec3<f32>(id + vec3<u32>(1u)) / 255.0, 1.0));
}
"#;
        let mut descriptor = texture_descriptor(
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 4),
            1,
            1,
            native::WGPUTextureUsage_StorageBinding | native::WGPUTextureUsage_CopySrc,
        );
        descriptor.dimension = native::WGPUTextureDimension_3D;
        let texture = create_texture(ctx.device, &descriptor);
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        let module = create_wgsl_module(ctx.device, shader);
        let pipeline_descriptor = native::WGPUComputePipelineDescriptor {
            compute: native::WGPUComputeState {
                module,
                entryPoint: string_view("main"),
                ..std::mem::zeroed()
            },
            ..std::mem::zeroed()
        };
        let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(ctx.device, &pipeline_descriptor);
        assert!(!pipeline.is_null());
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let entry = native::WGPUBindGroupEntry {
            binding: 0,
            textureView: view,
            ..std::mem::zeroed()
        };
        let group_descriptor = native::WGPUBindGroupDescriptor {
            layout,
            entryCount: 1,
            entries: &entry,
            ..std::mem::zeroed()
        };
        let group = yawgpu::wgpuDeviceCreateBindGroup(ctx.device, &group_descriptor);
        assert!(!group.is_null());
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        let readback_size = BYTES_PER_ROW as usize * 4 * 4;
        let readback = create_buffer(
            ctx.device,
            readback_size as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        record_t2b(encoder, texture, 0, extent(4, 4, 4), readback);
        ctx.submit(encoder);

        let bytes = read_buffer(ctx.instance, readback, readback_size);
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    assert_eq!(
                        pixel(&bytes, 4, x, y, z),
                        [x as u8 + 1, y as u8 + 1, z as u8 + 1, 255],
                        "texel ({x}, {y}, {z})"
                    );
                }
            }
        }

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBindGroupRelease(group);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    });
}

/// F6: rendering through an `rgba8unorm-srgb` view of an `rgba8unorm`
/// texture encodes to sRGB, for both the load-op clear (left half) and a
/// draw (right half). Attaching the raw unorm texture would store 128.
#[test]
#[ignore = "manual real-backend test"]
fn gles_srgb_view_of_unorm_texture_renders_srgb_encoded() {
    with_gles_device(|ctx| unsafe {
        let view_formats = [native::WGPUTextureFormat_RGBA8UnormSrgb];
        let mut descriptor = texture_descriptor(
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(16, 16, 1),
            1,
            1,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        );
        descriptor.viewFormatCount = view_formats.len();
        descriptor.viewFormats = view_formats.as_ptr();
        let texture = create_texture(ctx.device, &descriptor);
        let view = create_view(texture, native::WGPUTextureFormat_RGBA8UnormSrgb, 0);
        let module = create_wgsl_module(ctx.device, SHADER);
        let pipeline = create_pipeline(
            ctx.device,
            module,
            &raster(
                "vs_right",
                "fs_half_red",
                native::WGPUTextureFormat_RGBA8UnormSrgb,
            ),
        );
        let readback = create_readback(ctx.device, 16);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = begin_pass(
            encoder,
            &[color_attachment(
                view,
                std::ptr::null(),
                [0.5, 0.5, 0.5, 1.0],
            )],
            std::ptr::null(),
        );
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        end_pass(pass);
        record_t2b(encoder, texture, 0, extent(16, 16, 1), readback);
        ctx.submit(encoder);

        let bytes = read_buffer(ctx.instance, readback, readback_len(16));
        for y in 0..16 {
            for x in 0..16 {
                let expected = if x < 8 {
                    [188, 188, 188, 255]
                } else {
                    [188, 0, 0, 255]
                };
                assert_approx(pixel(&bytes, 16, x, y, 0), expected, 1, x, y);
            }
        }

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    });
}

/// F6: an `rgba8unorm-srgb` MSAA attachment resolves into an `rgba8unorm-srgb`
/// view of an `rgba8unorm` texture; the resolved bytes stay sRGB-encoded.
/// Blitting into the raw unorm texture would linearize them to 128.
#[test]
#[ignore = "manual real-backend test"]
fn gles_msaa_resolve_into_srgb_view_of_unorm_texture_keeps_encoding() {
    with_gles_device(|ctx| unsafe {
        let srgb = native::WGPUTextureFormat_RGBA8UnormSrgb;
        let msaa = create_texture(
            ctx.device,
            &texture_descriptor(
                srgb,
                extent(16, 16, 1),
                1,
                4,
                native::WGPUTextureUsage_RenderAttachment,
            ),
        );
        let msaa_view = yawgpu::wgpuTextureCreateView(msaa, std::ptr::null());
        let view_formats = [srgb];
        let mut descriptor = texture_descriptor(
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(16, 16, 1),
            1,
            1,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        );
        descriptor.viewFormatCount = view_formats.len();
        descriptor.viewFormats = view_formats.as_ptr();
        let resolve = create_texture(ctx.device, &descriptor);
        let resolve_view = create_view(resolve, srgb, 0);
        let readback = create_readback(ctx.device, 16);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let mut attachment = color_attachment(msaa_view, resolve_view, [0.5, 0.5, 0.5, 1.0]);
        attachment.storeOp = native::WGPUStoreOp_Discard;
        let pass = begin_pass(encoder, &[attachment], std::ptr::null());
        end_pass(pass);
        record_t2b(encoder, resolve, 0, extent(16, 16, 1), readback);
        ctx.submit(encoder);

        let bytes = read_buffer(ctx.instance, readback, readback_len(16));
        for y in 0..16 {
            for x in 0..16 {
                assert_approx(pixel(&bytes, 16, x, y, 0), [188, 188, 188, 255], 1, x, y);
            }
        }

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureViewRelease(resolve_view);
        yawgpu::wgpuTextureRelease(resolve);
        yawgpu::wgpuTextureViewRelease(msaa_view);
        yawgpu::wgpuTextureRelease(msaa);
    });
}

/// F8: `frag_depth` is clamped to the viewport depth range. The first draw
/// (viewport depth [0, 0.5]) writes `frag_depth = 0.9`, which must store
/// 0.5; the second (viewport [0, 1], depth 0.7, `Less`) then fails the
/// depth test and red survives. Unclamped, 0.9 is stored and green wins.
#[test]
#[ignore = "manual real-backend test"]
fn gles_frag_depth_is_clamped_to_viewport_depth_range() {
    with_gles_device(|ctx| unsafe {
        let shader = r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[i], 0.7, 1.0);
}

struct DepthOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_far_red() -> DepthOut {
    return DepthOut(vec4<f32>(1.0, 0.0, 0.0, 1.0), 0.9);
}

@fragment
fn fs_green() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
"#;
        let format = native::WGPUTextureFormat_RGBA8Unorm;
        let color = create_texture(
            ctx.device,
            &texture_descriptor(
                format,
                extent(16, 16, 1),
                1,
                1,
                native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            ),
        );
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());
        let depth = create_texture(
            ctx.device,
            &texture_descriptor(
                native::WGPUTextureFormat_Depth32Float,
                extent(16, 16, 1),
                1,
                1,
                native::WGPUTextureUsage_RenderAttachment,
            ),
        );
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());
        let module = create_wgsl_module(ctx.device, shader);
        let always = depth_state(native::WGPUCompareFunction_Always);
        let less = depth_state(native::WGPUCompareFunction_Less);
        let mut clamped = raster("vs", "fs_far_red", format);
        clamped.depth_stencil = &always;
        let clamped = create_pipeline(ctx.device, module, &clamped);
        let mut tested = raster("vs", "fs_green", format);
        tested.depth_stencil = &less;
        let tested = create_pipeline(ctx.device, module, &tested);
        let readback = create_readback(ctx.device, 16);

        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Discard,
            depthClearValue: 1.0,
            depthReadOnly: 0,
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: 0,
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
        let pass = begin_pass(
            encoder,
            &[color_attachment(
                color_view,
                std::ptr::null(),
                [0.0, 0.0, 1.0, 1.0],
            )],
            &depth_attachment,
        );
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, clamped);
        yawgpu::wgpuRenderPassEncoderSetViewport(pass, 0.0, 0.0, 16.0, 16.0, 0.0, 0.5);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, tested);
        yawgpu::wgpuRenderPassEncoderSetViewport(pass, 0.0, 0.0, 16.0, 16.0, 0.0, 1.0);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        end_pass(pass);
        record_t2b(encoder, color, 0, extent(16, 16, 1), readback);
        ctx.submit(encoder);

        let bytes = read_buffer(ctx.instance, readback, readback_len(16));
        assert_scissored(&bytes, 16, 0, RED, RED);

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderPipelineRelease(tested);
        yawgpu::wgpuRenderPipelineRelease(clamped);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureRelease(color);
    });
}

/// Asserts the left half of a `width` x `height` readback is `left` and the
/// right half `right`.
fn assert_halves(bytes: &[u8], width: u32, height: u32, left: [u8; 4], right: [u8; 4]) {
    for y in 0..height {
        for x in 0..width {
            let expected = if x < width / 2 { left } else { right };
            assert_eq!(pixel(bytes, height, x, y, 0), expected, "texel ({x}, {y})");
        }
    }
}

/// Asserts the top-left `scissor` x `scissor` square of a `size` x `size`
/// readback is `inside` and every other texel `outside`.
fn assert_scissored(bytes: &[u8], size: u32, scissor: u32, inside: [u8; 4], outside: [u8; 4]) {
    for y in 0..size {
        for x in 0..size {
            let expected = if x < scissor && y < scissor {
                inside
            } else {
                outside
            };
            assert_eq!(pixel(bytes, size, x, y, 0), expected, "texel ({x}, {y})");
        }
    }
}

fn assert_approx(actual: [u8; 4], expected: [u8; 4], tolerance: u8, x: u32, y: u32) {
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.abs_diff(expected) <= tolerance),
        "texel ({x}, {y}): expected ~{expected:?}, got {actual:?}"
    );
}

/// Reads one RGBA8 texel from a readback laid out with `BYTES_PER_ROW` and
/// `rows_per_image` rows per depth slice.
fn pixel(bytes: &[u8], rows_per_image: u32, x: u32, y: u32, z: u32) -> [u8; 4] {
    let offset = ((z * rows_per_image + y) * BYTES_PER_ROW + x * 4) as usize;
    bytes[offset..offset + 4].try_into().expect("4-byte texel")
}

fn readback_len(height: u32) -> usize {
    BYTES_PER_ROW as usize * height as usize
}

/// GLES device fixture with captured validation and backend errors.
struct Ctx {
    instance: native::WGPUInstance,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    errors: Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
}

impl Ctx {
    fn assert_clean(&self) {
        let errors = self.errors.lock().expect("error lock");
        assert!(errors.is_empty(), "device errors: {errors:?}");
    }

    unsafe fn submit(&self, encoder: native::WGPUCommandEncoder) {
        let commands = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        self.assert_clean();
        yawgpu::wgpuQueueSubmit(self.queue, 1, &commands);
        yawgpu::wgpuCommandBufferRelease(commands);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        self.assert_clean();
    }
}

fn with_gles_device(body: impl FnOnce(&Ctx)) {
    assert!(
        real_backend_available(RealBackend::Gles),
        "GLES backend not available; install an ANGLE build with ES 3.1 support \
         (Chrome / Edge ANGLE caps at ES 3.0; see specs/blocks/67-gles-backend.md). \
         Without a real GLES adapter this test cannot verify the GLES execution path."
    );
    unsafe {
        let instance = create_gles_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured.lock().expect("error lock").push(error)),
        );
        let ctx = Ctx {
            instance,
            device,
            queue: yawgpu::wgpuDeviceGetQueue(device),
            errors,
        };
        body(&ctx);
        ctx.assert_clean();
        yawgpu::wgpuQueueRelease(ctx.queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Render pipeline inputs; `raster` fills the defaults (one RGBA target,
/// no vertex buffers, triangle list, single-sampled, no depth/stencil).
struct Raster<'a> {
    vs: &'a str,
    fs: &'a str,
    format: native::WGPUTextureFormat,
    samples: u32,
    primitive: native::WGPUPrimitiveState,
    buffers: &'a [native::WGPUVertexBufferLayout],
    depth_stencil: *const native::WGPUDepthStencilState,
}

fn raster<'a>(vs: &'a str, fs: &'a str, format: native::WGPUTextureFormat) -> Raster<'a> {
    Raster {
        vs,
        fs,
        format,
        samples: 1,
        primitive: native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: native::WGPUFrontFace_Undefined,
            cullMode: native::WGPUCullMode_Undefined,
            unclippedDepth: 0,
        },
        buffers: &[],
        depth_stencil: std::ptr::null(),
    }
}

unsafe fn create_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    desc: &Raster<'_>,
) -> native::WGPURenderPipeline {
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: desc.format,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view(desc.fs),
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
            entryPoint: string_view(desc.vs),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: desc.buffers.len(),
            buffers: if desc.buffers.is_empty() {
                std::ptr::null()
            } else {
                desc.buffers.as_ptr()
            },
        },
        primitive: desc.primitive,
        depthStencil: desc.depth_stencil,
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: desc.samples,
            mask: 0xFFFF_FFFF,
            alphaToCoverageEnabled: 0,
        },
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

fn depth_state(compare: native::WGPUCompareFunction) -> native::WGPUDepthStencilState {
    let keep_always = native::WGPUStencilFaceState {
        compare: native::WGPUCompareFunction_Always,
        failOp: native::WGPUStencilOperation_Keep,
        depthFailOp: native::WGPUStencilOperation_Keep,
        passOp: native::WGPUStencilOperation_Keep,
    };
    native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32Float,
        depthWriteEnabled: native::WGPUOptionalBool_True,
        depthCompare: compare,
        stencilFront: keep_always,
        stencilBack: keep_always,
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    }
}

fn color_attachment(
    view: native::WGPUTextureView,
    resolve_target: native::WGPUTextureView,
    clear: [f64; 4],
) -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: resolve_target,
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: clear[0],
            g: clear[1],
            b: clear[2],
            a: clear[3],
        },
    }
}

unsafe fn begin_pass(
    encoder: native::WGPUCommandEncoder,
    attachments: &[native::WGPURenderPassColorAttachment],
    depth_stencil: *const native::WGPURenderPassDepthStencilAttachment,
) -> native::WGPURenderPassEncoder {
    let descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: depth_stencil,
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    pass
}

unsafe fn end_pass(pass: native::WGPURenderPassEncoder) {
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
}

fn texture_descriptor(
    format: native::WGPUTextureFormat,
    size: native::WGPUExtent3D,
    mip_level_count: u32,
    sample_count: u32,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size,
        format,
        mipLevelCount: mip_level_count,
        sampleCount: sample_count,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    }
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    descriptor: &native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    let texture = yawgpu::wgpuDeviceCreateTexture(device, descriptor);
    assert!(!texture.is_null());
    texture
}

/// Creates a single-mip view at `base_mip_level`; `format` Undefined keeps
/// the texture's format.
unsafe fn create_view(
    texture: native::WGPUTexture,
    format: native::WGPUTextureFormat,
    base_mip_level: u32,
) -> native::WGPUTextureView {
    let descriptor = native::WGPUTextureViewDescriptor {
        format,
        dimension: native::WGPUTextureViewDimension_2D,
        baseMipLevel: base_mip_level,
        mipLevelCount: 1,
        baseArrayLayer: 0,
        arrayLayerCount: 1,
        aspect: native::WGPUTextureAspect_All,
        ..std::mem::zeroed()
    };
    let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
    assert!(!view.is_null());
    view
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

unsafe fn create_readback(device: native::WGPUDevice, height: u32) -> native::WGPUBuffer {
    create_buffer(
        device,
        readback_len(height) as u64,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    )
}

fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width,
        height,
        depthOrArrayLayers: depth_or_array_layers,
    }
}

fn buffer_copy_info(
    buffer: native::WGPUBuffer,
    rows_per_image: u32,
) -> native::WGPUTexelCopyBufferInfo {
    native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: rows_per_image,
        },
        buffer,
    }
}

unsafe fn record_t2b(
    encoder: native::WGPUCommandEncoder,
    texture: native::WGPUTexture,
    mip_level: u32,
    size: native::WGPUExtent3D,
    buffer: native::WGPUBuffer,
) {
    let source = native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: mip_level,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
        encoder,
        &source,
        &buffer_copy_info(buffer, size.height),
        &size,
    );
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
