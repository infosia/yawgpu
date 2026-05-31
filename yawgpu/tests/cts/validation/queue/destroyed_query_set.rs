//! CTS port of `webgpu/api/validation/queue/destroyed/query_set.spec.ts`.
//!
//! Timestamp subcases use `timestamp-query`; they run through
//! `ValidationTest::with_features` because Noop advertises the feature only
//! when requested.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use super::common::*;

#[test]
fn begin_occlusion_query() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for state in ["valid", "destroyed"] {
            let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
            let color = create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
            let color_view = create_texture_view(color);
            let color_attachment = color_attachment(color_view);
            let color_attachments = [color_attachment];
            let descriptor = render_pass_descriptor(&color_attachments, query_set);
            let encoder = create_encoder(test.device());
            let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
            yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
            yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
            yawgpu::wgpuRenderPassEncoderEnd(pass);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
            let cb = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
            yawgpu::wgpuCommandEncoderRelease(encoder);
            if state == "destroyed" {
                yawgpu::wgpuQuerySetDestroy(query_set);
            }
            expect_submit(&test, q, &[cb], state == "valid");
            yawgpu::wgpuCommandBufferRelease(cb);
            yawgpu::wgpuTextureViewRelease(color_view);
            yawgpu::wgpuTextureRelease(color);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn unused_occlusion_query() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for state in ["valid", "destroyed"] {
            let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
            let color = create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
            let color_view = create_texture_view(color);
            let color_attachment = color_attachment(color_view);
            let color_attachments = [color_attachment];
            let descriptor = render_pass_descriptor(&color_attachments, query_set);
            let encoder = create_encoder(test.device());
            let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
            yawgpu::wgpuRenderPassEncoderEnd(pass);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
            let cb = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
            yawgpu::wgpuCommandEncoderRelease(encoder);
            if state == "destroyed" {
                yawgpu::wgpuQuerySetDestroy(query_set);
            }
            expect_submit(&test, q, &[cb], state == "valid");
            yawgpu::wgpuCommandBufferRelease(cb);
            yawgpu::wgpuTextureViewRelease(color_view);
            yawgpu::wgpuTextureRelease(color);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn timestamps() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    unsafe {
        let q = queue(test.device());
        for state in ["valid", "destroyed"] {
            let query_set = create_query_set(test.device(), native::WGPUQueryType_Timestamp, 2);

            let compute_writes = native::WGPUPassTimestampWrites {
                nextInChain: std::ptr::null_mut(),
                querySet: query_set,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: native::WGPU_QUERY_SET_INDEX_UNDEFINED,
            };
            let compute_descriptor = native::WGPUComputePassDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                timestampWrites: &compute_writes,
            };
            let encoder = create_encoder(test.device());
            let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, &compute_descriptor);
            yawgpu::wgpuComputePassEncoderEnd(pass);
            yawgpu::wgpuComputePassEncoderRelease(pass);
            let compute_cb = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
            yawgpu::wgpuCommandEncoderRelease(encoder);

            let color = create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
            let color_view = create_texture_view(color);
            let color_attachment = color_attachment(color_view);
            let render_writes = native::WGPUPassTimestampWrites {
                nextInChain: std::ptr::null_mut(),
                querySet: query_set,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: native::WGPU_QUERY_SET_INDEX_UNDEFINED,
            };
            let descriptor = native::WGPURenderPassDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                colorAttachmentCount: 1,
                colorAttachments: &color_attachment,
                depthStencilAttachment: std::ptr::null(),
                occlusionQuerySet: std::ptr::null(),
                timestampWrites: &render_writes,
            };
            let encoder = create_encoder(test.device());
            let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
            yawgpu::wgpuRenderPassEncoderEnd(pass);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
            let render_cb = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
            yawgpu::wgpuCommandEncoderRelease(encoder);

            if state == "destroyed" {
                yawgpu::wgpuQuerySetDestroy(query_set);
            }
            expect_submit(&test, q, &[compute_cb], state == "valid");
            expect_submit(&test, q, &[render_cb], state == "valid");

            yawgpu::wgpuCommandBufferRelease(render_cb);
            yawgpu::wgpuCommandBufferRelease(compute_cb);
            yawgpu::wgpuTextureViewRelease(color_view);
            yawgpu::wgpuTextureRelease(color);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn resolve_query_set() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for state in ["valid", "destroyed"] {
            let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
            let buffer = create_buffer(
                test.device(),
                8,
                native::WGPUBufferUsage_QueryResolve,
                false,
            );
            let cb = encode_resolve_query_set(test.device(), query_set, buffer);
            if state == "destroyed" {
                yawgpu::wgpuQuerySetDestroy(query_set);
            }
            expect_submit(&test, q, &[cb], state == "valid");
            yawgpu::wgpuCommandBufferRelease(cb);
            yawgpu::wgpuBufferRelease(buffer);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}
