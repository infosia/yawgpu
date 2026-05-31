use yawgpu::native;

use crate::feature_common;

#[test]
fn enables_rg11b10ufloat_renderable() {
    feature_common::assert_noop_advertises_feature(native::WGPUFeatureName_TextureFormatsTier1);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureFormatsTier1);
    feature_common::assert_device_has_feature(&test, native::WGPUFeatureName_TextureFormatsTier1);
    feature_common::assert_device_has_feature(
        &test,
        native::WGPUFeatureName_RG11B10UfloatRenderable,
    );
}

#[test]
fn texture_usage_render_attachment() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureFormatsTier1);
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
fn texture_usage_multisample() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureFormatsTier1);
    unsafe {
        let texture = feature_common::assert_texture_ok(
            &test,
            native::WGPUTextureDescriptor {
                sampleCount: 4,
                ..feature_common::texture_descriptor(
                    native::WGPUTextureFormat_RG11B10Ufloat,
                    native::WGPUTextureUsage_RenderAttachment,
                )
            },
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn texture_usage_storage_binding() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureFormatsTier1);
    unsafe {
        for &format in feature_common::tier1_storage_formats() {
            let texture = feature_common::assert_texture_ok(
                &test,
                feature_common::texture_descriptor(format, native::WGPUTextureUsage_StorageBinding),
            );
            yawgpu::wgpuTextureRelease(texture);
        }
    }
}

#[test]
fn render_pipeline_color_target() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureFormatsTier1);
    unsafe {
        let pipeline = feature_common::assert_color_target_pipeline_ok(
            &test,
            native::WGPUTextureFormat_RG11B10Ufloat,
        );
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[test]
fn render_pass_resolvable() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureFormatsTier1);
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
fn bind_group_layout_storage_texture() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureFormatsTier1);
    unsafe {
        let layout = feature_common::assert_storage_texture_bgl_ok(
            &test,
            native::WGPUTextureFormat_R8Unorm,
            native::WGPUStorageTextureAccess_WriteOnly,
        );
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}

#[test]
fn pipeline_auto_layout_storage_texture() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureFormatsTier1);
    unsafe {
        let layout = feature_common::assert_storage_texture_bgl_ok(
            &test,
            native::WGPUTextureFormat_R8Unorm,
            native::WGPUStorageTextureAccess_WriteOnly,
        );
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}
