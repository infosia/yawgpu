//! Ports `$CTS/src/webgpu/api/validation/encoding/encoder_open_state.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_compute_pass, begin_render_pass, color_attachment, create_buffer,
    create_compute_pipeline, create_encoder, create_query_set, create_render_pipeline, create_view,
    empty_string_view, extent, finish_ok, release_view, render_pass_descriptor,
};

#[test]
fn non_pass_commands() {
    let test = ValidationTest::new();
    unsafe {
        for command in [
            "beginComputePass",
            "beginRenderPass",
            "clearBuffer",
            "copyBufferToBuffer",
            "copyBufferToTexture",
            "copyTextureToBuffer",
            "copyTextureToTexture",
            "insertDebugMarker",
            "pushDebugGroup",
            "popDebugGroup",
            "resolveQuerySet",
        ] {
            for finish_before_command in [false, true] {
                let src_buffer = create_buffer(
                    test.device(),
                    256,
                    native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                );
                let dst_buffer = create_buffer(
                    test.device(),
                    256,
                    native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_QueryResolve,
                );
                let src_texture = create_view(
                    test.device(),
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureUsage_CopySrc,
                    1,
                );
                let dst_texture = create_view(
                    test.device(),
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureUsage_CopyDst,
                    1,
                );
                let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
                let encoder = create_encoder(test.device());
                let finished = finish_before_command.then(|| finish_ok(&test, encoder));

                expect_device_error(&test, finish_before_command, || {
                    run_encoder_command(
                        encoder,
                        command,
                        src_buffer,
                        dst_buffer,
                        src_texture.texture,
                        dst_texture.texture,
                        query_set,
                    );
                });

                if let Some(command_buffer) = finished {
                    yawgpu::wgpuCommandBufferRelease(command_buffer);
                }
                yawgpu::wgpuCommandEncoderRelease(encoder);
                yawgpu::wgpuQuerySetRelease(query_set);
                release_view(dst_texture);
                release_view(src_texture);
                yawgpu::wgpuBufferRelease(dst_buffer);
                yawgpu::wgpuBufferRelease(src_buffer);
            }
        }
    }
}

#[test]
fn render_pass_commands() {
    let test = ValidationTest::new();
    unsafe {
        for command in [
            "draw",
            "drawIndexed",
            "drawIndirect",
            "drawIndexedIndirect",
            "setIndexBuffer",
            "setBindGroup",
            "setVertexBuffer",
            "setPipeline",
            "setViewport",
            "setScissorRect",
            "setBlendConstant",
            "setStencilReference",
            "beginOcclusionQuery",
            "endOcclusionQuery",
            "executeBundles",
            "pushDebugGroup",
            "popDebugGroup",
            "insertDebugMarker",
        ] {
            for finish_before_command in ["no", "pass", "encoder"] {
                let color = create_view(
                    test.device(),
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureUsage_RenderAttachment,
                    1,
                );
                let attachment = color_attachment(color.view);
                let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
                let mut descriptor = render_pass_descriptor(&[attachment], None);
                descriptor.occlusionQuerySet = query_set;
                let encoder = create_encoder(test.device());
                let pass = begin_render_pass(encoder, &descriptor);
                let buffer = create_buffer(
                    test.device(),
                    256,
                    native::WGPUBufferUsage_Indirect
                        | native::WGPUBufferUsage_Vertex
                        | native::WGPUBufferUsage_Index,
                );
                let pipeline = create_render_pipeline(&test);

                if finish_before_command != "no" {
                    yawgpu::wgpuRenderPassEncoderEnd(pass);
                }
                let finished =
                    (finish_before_command == "encoder").then(|| finish_ok(&test, encoder));

                expect_device_error(&test, finish_before_command != "no", || {
                    run_render_pass_command(pass, command, buffer, pipeline);
                });

                if let Some(command_buffer) = finished {
                    yawgpu::wgpuCommandBufferRelease(command_buffer);
                }
                yawgpu::wgpuRenderPipelineRelease(pipeline);
                yawgpu::wgpuBufferRelease(buffer);
                yawgpu::wgpuRenderPassEncoderRelease(pass);
                yawgpu::wgpuCommandEncoderRelease(encoder);
                yawgpu::wgpuQuerySetRelease(query_set);
                release_view(color);
            }
        }
    }
}

#[test]
fn render_bundle_commands() {
    let test = ValidationTest::new();
    unsafe {
        for command in [
            "draw",
            "drawIndexed",
            "drawIndexedIndirect",
            "drawIndirect",
            "setPipeline",
            "setBindGroup",
            "setIndexBuffer",
            "setVertexBuffer",
            "pushDebugGroup",
            "popDebugGroup",
            "insertDebugMarker",
        ] {
            for finish_before_command in [false, true] {
                let formats = [native::WGPUTextureFormat_RGBA8Unorm];
                let descriptor = native::WGPURenderBundleEncoderDescriptor {
                    nextInChain: std::ptr::null_mut(),
                    label: empty_string_view(),
                    colorFormatCount: formats.len(),
                    colorFormats: formats.as_ptr(),
                    depthStencilFormat: native::WGPUTextureFormat_Undefined,
                    sampleCount: 1,
                    depthReadOnly: 0,
                    stencilReadOnly: 0,
                };
                let encoder =
                    yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
                assert!(!encoder.is_null());
                let bundle = finish_before_command
                    .then(|| yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null()));
                let buffer = create_buffer(
                    test.device(),
                    256,
                    native::WGPUBufferUsage_Indirect
                        | native::WGPUBufferUsage_Vertex
                        | native::WGPUBufferUsage_Index,
                );
                let pipeline = create_render_pipeline(&test);

                expect_device_error(&test, finish_before_command, || {
                    run_render_bundle_command(encoder, command, buffer, pipeline);
                });

                yawgpu::wgpuRenderPipelineRelease(pipeline);
                yawgpu::wgpuBufferRelease(buffer);
                if let Some(bundle) = bundle {
                    yawgpu::wgpuRenderBundleRelease(bundle);
                }
                yawgpu::wgpuRenderBundleEncoderRelease(encoder);
            }
        }
    }
}

#[test]
fn compute_pass_commands() {
    let test = ValidationTest::new();
    unsafe {
        for command in [
            "setBindGroup",
            "setPipeline",
            "dispatchWorkgroups",
            "dispatchWorkgroupsIndirect",
            "pushDebugGroup",
            "popDebugGroup",
            "insertDebugMarker",
        ] {
            for finish_before_command in ["no", "pass", "encoder"] {
                let encoder = create_encoder(test.device());
                let pass = begin_compute_pass(encoder, None);
                let indirect = create_buffer(test.device(), 256, native::WGPUBufferUsage_Indirect);
                let pipeline = create_compute_pipeline(&test);

                if finish_before_command != "no" {
                    yawgpu::wgpuComputePassEncoderEnd(pass);
                }
                let finished =
                    (finish_before_command == "encoder").then(|| finish_ok(&test, encoder));

                expect_device_error(&test, finish_before_command != "no", || {
                    run_compute_pass_command(pass, command, indirect, pipeline);
                });

                if let Some(command_buffer) = finished {
                    yawgpu::wgpuCommandBufferRelease(command_buffer);
                }
                yawgpu::wgpuComputePipelineRelease(pipeline);
                yawgpu::wgpuBufferRelease(indirect);
                yawgpu::wgpuComputePassEncoderRelease(pass);
                yawgpu::wgpuCommandEncoderRelease(encoder);
            }
        }
    }
}

unsafe fn run_encoder_command(
    encoder: native::WGPUCommandEncoder,
    command: &str,
    src_buffer: native::WGPUBuffer,
    dst_buffer: native::WGPUBuffer,
    src_texture: native::WGPUTexture,
    dst_texture: native::WGPUTexture,
    query_set: native::WGPUQuerySet,
) {
    let texture_copy_src = native::WGPUTexelCopyTextureInfo {
        texture: src_texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let texture_copy_dst = native::WGPUTexelCopyTextureInfo {
        texture: dst_texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let buffer_copy = native::WGPUTexelCopyBufferInfo {
        buffer: dst_buffer,
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: native::WGPU_COPY_STRIDE_UNDEFINED,
            rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
        },
    };
    let copy_size = extent(1, 1, 1);
    match command {
        "beginComputePass" => {
            let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
            yawgpu::wgpuComputePassEncoderRelease(pass);
        }
        "beginRenderPass" => {
            let descriptor = render_pass_descriptor(&[], None);
            let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
        }
        "clearBuffer" => yawgpu::wgpuCommandEncoderClearBuffer(encoder, dst_buffer, 0, 16),
        "copyBufferToBuffer" => {
            yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, src_buffer, 0, dst_buffer, 0, 0);
        }
        "copyBufferToTexture" => {
            yawgpu::wgpuCommandEncoderCopyBufferToTexture(
                encoder,
                &native::WGPUTexelCopyBufferInfo {
                    buffer: src_buffer,
                    layout: buffer_copy.layout,
                },
                &texture_copy_dst,
                &copy_size,
            );
        }
        "copyTextureToBuffer" => {
            yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
                encoder,
                &texture_copy_src,
                &buffer_copy,
                &copy_size,
            );
        }
        "copyTextureToTexture" => {
            yawgpu::wgpuCommandEncoderCopyTextureToTexture(
                encoder,
                &texture_copy_src,
                &texture_copy_dst,
                &copy_size,
            );
        }
        "insertDebugMarker" => {
            yawgpu::wgpuCommandEncoderInsertDebugMarker(encoder, empty_string_view());
        }
        "pushDebugGroup" => yawgpu::wgpuCommandEncoderPushDebugGroup(encoder, empty_string_view()),
        "popDebugGroup" => yawgpu::wgpuCommandEncoderPopDebugGroup(encoder),
        "resolveQuerySet" => {
            yawgpu::wgpuCommandEncoderResolveQuerySet(encoder, query_set, 0, 1, dst_buffer, 0);
        }
        _ => unreachable!(),
    }
}

unsafe fn run_render_pass_command(
    pass: native::WGPURenderPassEncoder,
    command: &str,
    buffer: native::WGPUBuffer,
    pipeline: native::WGPURenderPipeline,
) {
    match command {
        "draw" => yawgpu::wgpuRenderPassEncoderDraw(pass, 1, 1, 0, 0),
        "drawIndexed" => yawgpu::wgpuRenderPassEncoderDrawIndexed(pass, 1, 1, 0, 0, 0),
        "drawIndirect" => yawgpu::wgpuRenderPassEncoderDrawIndirect(pass, buffer, 0),
        "drawIndexedIndirect" => {
            yawgpu::wgpuRenderPassEncoderDrawIndexedIndirect(pass, buffer, 0);
        }
        "setIndexBuffer" => {
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                buffer,
                native::WGPUIndexFormat_Uint32,
                0,
                16,
            );
        }
        "setBindGroup" => {
            yawgpu::wgpuRenderPassEncoderSetBindGroup(
                pass,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
            );
        }
        "setVertexBuffer" => yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 1, buffer, 0, 16),
        "setPipeline" => yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline),
        "setViewport" => {
            yawgpu::wgpuRenderPassEncoderSetViewport(pass, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)
        }
        "setScissorRect" => yawgpu::wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, 0, 0),
        "setBlendConstant" => {
            let color = native::WGPUColor {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            };
            yawgpu::wgpuRenderPassEncoderSetBlendConstant(pass, &color);
        }
        "setStencilReference" => yawgpu::wgpuRenderPassEncoderSetStencilReference(pass, 0),
        "beginOcclusionQuery" => yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0),
        "endOcclusionQuery" => yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass),
        "executeBundles" => yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 0, std::ptr::null()),
        "pushDebugGroup" => yawgpu::wgpuRenderPassEncoderPushDebugGroup(pass, empty_string_view()),
        "popDebugGroup" => yawgpu::wgpuRenderPassEncoderPopDebugGroup(pass),
        "insertDebugMarker" => {
            yawgpu::wgpuRenderPassEncoderInsertDebugMarker(pass, empty_string_view());
        }
        _ => unreachable!(),
    }
}

unsafe fn run_render_bundle_command(
    encoder: native::WGPURenderBundleEncoder,
    command: &str,
    buffer: native::WGPUBuffer,
    pipeline: native::WGPURenderPipeline,
) {
    match command {
        "draw" => yawgpu::wgpuRenderBundleEncoderDraw(encoder, 1, 1, 0, 0),
        "drawIndexed" => yawgpu::wgpuRenderBundleEncoderDrawIndexed(encoder, 1, 1, 0, 0, 0),
        "drawIndexedIndirect" => {
            yawgpu::wgpuRenderBundleEncoderDrawIndexedIndirect(encoder, buffer, 0);
        }
        "drawIndirect" => yawgpu::wgpuRenderBundleEncoderDrawIndirect(encoder, buffer, 0),
        "setPipeline" => yawgpu::wgpuRenderBundleEncoderSetPipeline(encoder, pipeline),
        "setBindGroup" => {
            yawgpu::wgpuRenderBundleEncoderSetBindGroup(
                encoder,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
            );
        }
        "setIndexBuffer" => {
            yawgpu::wgpuRenderBundleEncoderSetIndexBuffer(
                encoder,
                buffer,
                native::WGPUIndexFormat_Uint32,
                0,
                16,
            );
        }
        "setVertexBuffer" => {
            yawgpu::wgpuRenderBundleEncoderSetVertexBuffer(encoder, 1, buffer, 0, 16);
        }
        "pushDebugGroup" => {
            yawgpu::wgpuRenderBundleEncoderPushDebugGroup(encoder, empty_string_view());
        }
        "popDebugGroup" => yawgpu::wgpuRenderBundleEncoderPopDebugGroup(encoder),
        "insertDebugMarker" => {
            yawgpu::wgpuRenderBundleEncoderInsertDebugMarker(encoder, empty_string_view());
        }
        _ => unreachable!(),
    }
}

unsafe fn run_compute_pass_command(
    pass: native::WGPUComputePassEncoder,
    command: &str,
    indirect: native::WGPUBuffer,
    pipeline: native::WGPUComputePipeline,
) {
    match command {
        "setBindGroup" => {
            yawgpu::wgpuComputePassEncoderSetBindGroup(
                pass,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
            );
        }
        "setPipeline" => yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline),
        "dispatchWorkgroups" => yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 0, 1, 1),
        "dispatchWorkgroupsIndirect" => {
            yawgpu::wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, indirect, 0);
        }
        "pushDebugGroup" => yawgpu::wgpuComputePassEncoderPushDebugGroup(pass, empty_string_view()),
        "popDebugGroup" => yawgpu::wgpuComputePassEncoderPopDebugGroup(pass),
        "insertDebugMarker" => {
            yawgpu::wgpuComputePassEncoderInsertDebugMarker(pass, empty_string_view());
        }
        _ => unreachable!(),
    }
}

fn expect_device_error<F>(test: &ValidationTest, should_error: bool, action: F)
where
    F: FnOnce(),
{
    if should_error {
        test.assert_device_error_after(action, None);
    } else {
        test.clear_errors();
        action();
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );
    }
}
