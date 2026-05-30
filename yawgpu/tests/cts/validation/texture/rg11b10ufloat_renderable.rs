use yawgpu::native;

use crate::{common, feature_common};

#[test]
#[ignore = "core texture format capabilities are not yet keyed to rg11b10ufloat-renderable"]
fn create_texture() {
    feature_common::assert_noop_advertises_feature(native::WGPUFeatureName_RG11B10UfloatRenderable);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_RG11B10UfloatRenderable);
    unsafe {
        let texture = feature_common::assert_texture_ok(
            &test,
            feature_common::texture_descriptor(
                native::WGPUTextureFormat_RG11B10Ufloat,
                native::WGPUTextureUsage_RenderAttachment,
            ),
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
#[ignore = "core render pass format capabilities are not yet keyed to rg11b10ufloat-renderable"]
fn begin_render_pass_single_sampled() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_RG11B10UfloatRenderable);
    unsafe {
        let texture = feature_common::assert_texture_ok(
            &test,
            feature_common::texture_descriptor(
                native::WGPUTextureFormat_RG11B10Ufloat,
                native::WGPUTextureUsage_RenderAttachment,
            ),
        );
        let view = common::create_texture_view(texture);
        let attachment = common::color_attachment(view);
        let descriptor = common::render_pass_descriptor(&attachment);
        let encoder = common::create_command_encoder(test.device());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        assert!(!pass.is_null());
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        let commands = common::finish_command_encoder(encoder);
        yawgpu::wgpuCommandBufferRelease(commands);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
#[ignore = "core multisample/resolve capabilities are not yet keyed to rg11b10ufloat-renderable"]
fn begin_render_pass_msaa_and_resolve() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_RG11B10UfloatRenderable);
    unsafe {
        let msaa = feature_common::assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                sampleCount: 4,
                ..feature_common::texture_descriptor(
                    native::WGPUTextureFormat_RG11B10Ufloat,
                    native::WGPUTextureUsage_RenderAttachment,
                )
            },
        );
        yawgpu::wgpuTextureRelease(msaa);
    }
}

#[test]
#[ignore = "core render bundle format capabilities are not yet keyed to rg11b10ufloat-renderable"]
fn begin_render_bundle_encoder() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_RG11B10UfloatRenderable);
    unsafe {
        let encoder = feature_common::assert_render_bundle_encoder_ok(
            &test,
            native::WGPUTextureFormat_RG11B10Ufloat,
            native::WGPUTextureFormat_Undefined,
        );
        yawgpu::wgpuRenderBundleEncoderRelease(encoder);
    }
}

#[test]
#[ignore = "core render pipeline format capabilities are not yet keyed to rg11b10ufloat-renderable"]
fn create_render_pipeline() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_RG11B10UfloatRenderable);
    unsafe {
        let pipeline = feature_common::assert_color_target_pipeline_ok(
            &test,
            native::WGPUTextureFormat_RG11B10Ufloat,
        );
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}
