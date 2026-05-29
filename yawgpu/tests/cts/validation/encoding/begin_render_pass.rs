//! Ports `$CTS/src/webgpu/api/validation/encoding/beginRenderPass.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_render_pass, color_attachment_with_resolve, create_encoder, create_query_set,
    create_view, depth_stencil_attachment, expect_finish, release_view, render_pass_descriptor,
    timestamp_writes,
};

#[test]
#[ignore = "core does not yet validate render pass color attachment view/resolveTarget device ownership; CTS expects mismatched-device attachments to fail"]
fn color_attachments_device_mismatch() {
    let test = ValidationTest::new();
    let foreign = ValidationTest::new();
    unsafe {
        for (view0_mismatch, target0_mismatch, view1_mismatch, target1_mismatch) in [
            (false, false, false, false),
            (false, true, false, true),
            (true, false, true, false),
            (false, false, false, true),
        ] {
            let view0_device = if view0_mismatch {
                foreign.device()
            } else {
                test.device()
            };
            let target0_device = if target0_mismatch {
                foreign.device()
            } else {
                test.device()
            };
            let view1_device = if view1_mismatch {
                foreign.device()
            } else {
                test.device()
            };
            let target1_device = if target1_mismatch {
                foreign.device()
            } else {
                test.device()
            };

            let view0 = create_view(
                view0_device,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureUsage_RenderAttachment,
                4,
            );
            let target0 = create_view(
                target0_device,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureUsage_RenderAttachment,
                1,
            );
            let view1 = create_view(
                view1_device,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureUsage_RenderAttachment,
                4,
            );
            let target1 = create_view(
                target1_device,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureUsage_RenderAttachment,
                1,
            );
            let attachments = [
                color_attachment_with_resolve(view0.view, target0.view),
                color_attachment_with_resolve(view1.view, target1.view),
            ];
            let descriptor = render_pass_descriptor(&attachments, None);
            let mismatched =
                view0_mismatch || target0_mismatch || view1_mismatch || target1_mismatch;

            finish_render_pass(&test, &descriptor, !mismatched);

            release_view(target1);
            release_view(view1);
            release_view(target0);
            release_view(view0);
        }
    }
}

#[test]
#[ignore = "core does not yet validate render pass depth-stencil attachment view device ownership; CTS expects mismatched-device attachments to fail"]
fn depth_stencil_attachment_device_mismatch() {
    let test = ValidationTest::new();
    let foreign = ValidationTest::new();
    unsafe {
        for mismatched in [false, true] {
            let depth = create_view(
                if mismatched {
                    foreign.device()
                } else {
                    test.device()
                },
                native::WGPUTextureFormat_Depth24PlusStencil8,
                native::WGPUTextureUsage_RenderAttachment,
                1,
            );
            let depth_attachment = depth_stencil_attachment(depth.view);
            let descriptor = render_pass_descriptor(&[], Some(&depth_attachment));

            finish_render_pass(&test, &descriptor, !mismatched);

            release_view(depth);
        }
    }
}

#[test]
#[ignore = "core does not yet validate render pass occlusionQuerySet device ownership; CTS expects a mismatched-device query set to fail"]
fn occlusion_query_set_device_mismatch() {
    let test = ValidationTest::new();
    let foreign = ValidationTest::new();
    unsafe {
        for mismatched in [false, true] {
            let query_set = create_query_set(
                if mismatched {
                    foreign.device()
                } else {
                    test.device()
                },
                native::WGPUQueryType_Occlusion,
                1,
            );
            let color = create_view(
                test.device(),
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureUsage_RenderAttachment,
                1,
            );
            let attachment = crate::common::color_attachment(color.view);
            let mut descriptor = render_pass_descriptor(&[attachment], None);
            descriptor.occlusionQuerySet = query_set;

            finish_render_pass(&test, &descriptor, !mismatched);

            release_view(color);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[test]
#[ignore = "core does not yet validate render pass timestampWrites querySet device ownership; CTS expects a mismatched-device query set to fail"]
fn timestamp_query_set_device_mismatch() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    let foreign = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);
    unsafe {
        for mismatched in [false, true] {
            let query_set = create_query_set(
                if mismatched {
                    foreign.device()
                } else {
                    test.device()
                },
                native::WGPUQueryType_Timestamp,
                1,
            );
            let writes = timestamp_writes(query_set, 0, native::WGPU_QUERY_SET_INDEX_UNDEFINED);
            let color = create_view(
                test.device(),
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureUsage_RenderAttachment,
                1,
            );
            let attachment = crate::common::color_attachment(color.view);
            let mut descriptor = render_pass_descriptor(&[attachment], None);
            descriptor.timestampWrites = &writes;

            finish_render_pass(&test, &descriptor, !mismatched);

            release_view(color);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

unsafe fn finish_render_pass(
    test: &ValidationTest,
    descriptor: &native::WGPURenderPassDescriptor,
    success: bool,
) {
    let encoder = create_encoder(test.device());
    test.clear_errors();
    let pass = begin_render_pass(encoder, descriptor);
    assert!(
        test.errors().is_empty(),
        "beginRenderPass should defer descriptor validation to finish: {:?}",
        test.errors()
    );
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = expect_finish(test, encoder, success);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}
