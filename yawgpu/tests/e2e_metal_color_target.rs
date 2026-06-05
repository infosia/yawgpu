//! Real-GPU Metal regression for F-035: the render pipeline must honor each color
//! target's `writeMask` and `blend`, plus the render-pass blend constant. Before
//! F-035 yawgpu dropped `writeMask`/`blend` (only the format reached the HAL) and
//! `setBlendConstant` was a stub, so the raw fragment output was written to every
//! channel. Each probe renders a white fragment over a known clear and reads the
//! first pixel back. The gap was HAL-agnostic, so a Metal probe exercises the
//! shared pipeline color-target translation + a Tier-1 backend; the CTS
//! `color_target_state` port covers Metal and Vulkan. Gated on `metal`.
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

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const BYTES_PER_ROW: u32 = 256;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;

// Full-screen quad; the fragment outputs white (1,1,1,1).
const WHITE_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#;

// F-040 slice 1: two color attachments; fragment writes distinct colors to
// each — location 0 red, location 1 green.
const MRT_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.0, 1.0);
}
struct FragmentOut {
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
}
@fragment
fn fs() -> FragmentOut {
    return FragmentOut(vec4<f32>(1.0, 0.0, 0.0, 1.0), vec4<f32>(0.0, 1.0, 0.0, 1.0));
}
"#;

/// `writeMask = Red`: clear to black-opaque, draw white — only the red channel is
/// written, so the pixel reads `[255, 0, 0, 255]` (not `[255,255,255,255]`).
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_color_write_mask_gates_channels() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let (instance, adapter, device, queue, errors) = setup();
        let module = create_wgsl_module(device, WHITE_SHADER);
        let pipeline = create_pipeline(
            device,
            module,
            native::WGPUColorWriteMask_Red,
            std::ptr::null(),
        );
        let pixel = render_first_pixel(
            instance,
            device,
            queue,
            pipeline,
            native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            None,
        );
        assert_eq!(
            pixel,
            [255, 0, 0, 255],
            "writeMask=Red did not gate the green/blue channels"
        );
        assert!(errors.lock().expect("error lock").is_empty());
        teardown(module, pipeline, queue, device, adapter, instance);
    }
}

/// `blend` with `srcFactor=constant`: `result.rgb = src.rgb * blendConstant.rgb`.
/// White source × constant `0.5` ⇒ `[128, 128, 128, 255]` (not full-scale).
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_blend_constant_scales_output() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let (instance, adapter, device, queue, errors) = setup();
        let module = create_wgsl_module(device, WHITE_SHADER);
        // color: src*constant + dst*0 ; alpha: src*1 + dst*0.
        let blend = native::WGPUBlendState {
            color: native::WGPUBlendComponent {
                operation: native::WGPUBlendOperation_Add,
                srcFactor: native::WGPUBlendFactor_Constant,
                dstFactor: native::WGPUBlendFactor_Zero,
            },
            alpha: native::WGPUBlendComponent {
                operation: native::WGPUBlendOperation_Add,
                srcFactor: native::WGPUBlendFactor_One,
                dstFactor: native::WGPUBlendFactor_Zero,
            },
        };
        let pipeline = create_pipeline(device, module, native::WGPUColorWriteMask_All, &blend);
        let pixel = render_first_pixel(
            instance,
            device,
            queue,
            pipeline,
            native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            },
            Some(native::WGPUColor {
                r: 0.5,
                g: 0.5,
                b: 0.5,
                a: 1.0,
            }),
        );
        for (channel, value) in pixel.iter().take(3).enumerate() {
            assert!(
                value.abs_diff(128) <= 1,
                "blend src*constant not applied on channel {channel}: got {value} (pixel {pixel:?})"
            );
        }
        assert_eq!(pixel[3], 255, "alpha (src*1) wrong: {pixel:?}");
        assert!(errors.lock().expect("error lock").is_empty());
        teardown(module, pipeline, queue, device, adapter, instance);
    }
}

/// F-040 slice 1: a render pass with TWO color attachments and a two-target
/// pipeline must write each attachment independently — attachment 0 reads red,
/// attachment 1 reads green. Before slice 1 only one color target was supported.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_two_color_attachments_write_distinct_targets() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let (instance, adapter, device, queue, errors) = setup();
        let module = create_wgsl_module(device, MRT_SHADER);
        let pipeline = create_two_target_pipeline(device, module);
        let (pixel_a, pixel_b) = render_two_color_attachments(instance, device, queue, pipeline);
        assert_eq!(pixel_a, [255, 0, 0, 255], "attachment 0 should be red");
        assert_eq!(pixel_b, [0, 255, 0, 255], "attachment 1 should be green");
        assert!(errors.lock().expect("error lock").is_empty());
        teardown(module, pipeline, queue, device, adapter, instance);
    }
}

#[cfg(feature = "metal")]
unsafe fn render_two_color_attachments(
    instance: native::WGPUInstance,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    pipeline: native::WGPURenderPipeline,
) -> ([u8; 4], [u8; 4]) {
    let color_a = create_color_texture(device);
    let color_b = create_color_texture(device);
    let readback_a = create_buffer(
        device,
        READBACK_SIZE as u64,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let readback_b = create_buffer(
        device,
        READBACK_SIZE as u64,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let view_a = yawgpu::wgpuTextureCreateView(color_a, std::ptr::null());
    let view_b = yawgpu::wgpuTextureCreateView(color_b, std::ptr::null());
    let make_attachment = |view| native::WGPURenderPassColorAttachment {
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
            a: 1.0,
        },
    };
    let attachments = [make_attachment(view_a), make_attachment(view_b)];
    let pass_descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
    yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    for (texture, buffer) in [(color_a, readback_a), (color_b, readback_b)] {
        let src = native::WGPUTexelCopyTextureInfo {
            texture,
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
            buffer,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &texture_extent());
    }
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);

    let pixels_a = read_buffer(instance, readback_a, 0, READBACK_SIZE);
    let pixels_b = read_buffer(instance, readback_b, 0, READBACK_SIZE);
    let result = (
        [pixels_a[0], pixels_a[1], pixels_a[2], pixels_a[3]],
        [pixels_b[0], pixels_b[1], pixels_b[2], pixels_b[3]],
    );
    yawgpu::wgpuTextureViewRelease(view_b);
    yawgpu::wgpuTextureViewRelease(view_a);
    yawgpu::wgpuBufferRelease(readback_b);
    yawgpu::wgpuBufferRelease(readback_a);
    yawgpu::wgpuTextureRelease(color_b);
    yawgpu::wgpuTextureRelease(color_a);
    result
}

#[cfg(feature = "metal")]
unsafe fn create_two_target_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let targets = [color_target, color_target];
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: targets.len(),
        targets: targets.as_ptr(),
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
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

/// F-040 slice 2: a `sampleCount=4` MSAA color attachment with a single-sample
/// `resolveTarget` must resolve the multisampled output into the resolve target.
/// The triangle covers the whole attachment with white, so every sample is white
/// and the resolved pixel is white. Before slice 2 the resolve target was never
/// written (read back the cleared value).
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_msaa_resolve_writes_resolve_target() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let (instance, adapter, device, queue, errors) = setup();
        let module = create_wgsl_module(device, WHITE_SHADER);
        let pipeline = create_msaa_pipeline(device, module, 4);

        let msaa = create_msaa_color_texture(device, 4);
        let resolve = create_color_texture(device); // sampleCount=1, RenderAttachment | CopySrc
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let msaa_view = yawgpu::wgpuTextureCreateView(msaa, std::ptr::null());
        let resolve_view = yawgpu::wgpuTextureCreateView(resolve, std::ptr::null());

        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: msaa_view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: resolve_view,
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        };
        let attachments = [color_attachment];
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: attachments.len(),
            colorAttachments: attachments.as_ptr(),
            depthStencilAttachment: std::ptr::null(),
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        // The resolve target is single-sample, so it is copyable; the MSAA texture is not.
        let src = native::WGPUTexelCopyTextureInfo {
            texture: resolve,
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
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let pixels = read_buffer(instance, readback, 0, READBACK_SIZE);
        let pixel = [pixels[0], pixels[1], pixels[2], pixels[3]];
        assert_eq!(
            pixel,
            [255, 255, 255, 255],
            "MSAA resolve did not write the resolve target (got {pixel:?})"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuTextureViewRelease(resolve_view);
        yawgpu::wgpuTextureViewRelease(msaa_view);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(resolve);
        yawgpu::wgpuTextureRelease(msaa);
        teardown(module, pipeline, queue, device, adapter, instance);
    }
}

#[cfg(feature = "metal")]
unsafe fn create_msaa_color_texture(
    device: native::WGPUDevice,
    sample_count: u32,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment,
        dimension: native::WGPUTextureDimension_2D,
        size: texture_extent(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: sample_count,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

#[cfg(feature = "metal")]
unsafe fn create_msaa_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    sample_count: u32,
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
            frontFace: native::WGPUFrontFace_Undefined,
            cullMode: native::WGPUCullMode_Undefined,
            unclippedDepth: 0,
        },
        depthStencil: std::ptr::null(),
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: sample_count,
            mask: 0xFFFF_FFFF,
            alphaToCoverageEnabled: 0,
        },
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
unsafe fn render_first_pixel(
    instance: native::WGPUInstance,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    pipeline: native::WGPURenderPipeline,
    clear: native::WGPUColor,
    blend_constant: Option<native::WGPUColor>,
) -> [u8; 4] {
    let color = create_color_texture(device);
    let readback = create_buffer(
        device,
        READBACK_SIZE as u64,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());
    let color_attachment = native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view: color_view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: clear,
    };
    let attachments = [color_attachment];
    let pass_descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
    yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    if let Some(c) = blend_constant {
        yawgpu::wgpuRenderPassEncoderSetBlendConstant(pass, &c);
    }
    yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    let color_src = native::WGPUTexelCopyTextureInfo {
        texture: color,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let color_dst = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        },
        buffer: readback,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
        encoder,
        &color_src,
        &color_dst,
        &texture_extent(),
    );
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);

    let pixels = read_buffer(instance, readback, 0, READBACK_SIZE);
    let result = [pixels[0], pixels[1], pixels[2], pixels[3]];
    yawgpu::wgpuTextureViewRelease(color_view);
    yawgpu::wgpuBufferRelease(readback);
    yawgpu::wgpuTextureRelease(color);
    result
}

#[cfg(feature = "metal")]
unsafe fn create_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    write_mask: native::WGPUColorWriteMask,
    blend: *const native::WGPUBlendState,
) -> native::WGPURenderPipeline {
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend,
        writeMask: write_mask,
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
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
type Harness = (
    native::WGPUInstance,
    native::WGPUAdapter,
    native::WGPUDevice,
    native::WGPUQueue,
    Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
);

#[cfg(feature = "metal")]
unsafe fn setup() -> Harness {
    let instance = create_metal_instance();
    let adapter = request_adapter(instance);
    let device = request_device(instance, adapter);
    let errors = install_error_capture(device);
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    (instance, adapter, device, queue, errors)
}

#[cfg(feature = "metal")]
unsafe fn teardown(
    module: native::WGPUShaderModule,
    pipeline: native::WGPURenderPipeline,
    queue: native::WGPUQueue,
    device: native::WGPUDevice,
    adapter: native::WGPUAdapter,
    instance: native::WGPUInstance,
) {
    yawgpu::wgpuRenderPipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuQueueRelease(queue);
    yawgpu::wgpuDeviceRelease(device);
    yawgpu::wgpuAdapterRelease(adapter);
    yawgpu::wgpuInstanceRelease(instance);
}

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
    }
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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
