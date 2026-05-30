use yawgpu::native;

use crate::feature_common;

#[test]
fn enables_rg11b10ufloat_renderable_and_texture_formats_tier1() {
    feature_common::assert_noop_advertises_feature(native::WGPUFeatureName_TextureFormatsTier2);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureFormatsTier2);
    feature_common::assert_device_has_feature(&test, native::WGPUFeatureName_TextureFormatsTier2);
    feature_common::assert_device_has_feature(&test, native::WGPUFeatureName_TextureFormatsTier1);
    feature_common::assert_device_has_feature(
        &test,
        native::WGPUFeatureName_RG11B10UfloatRenderable,
    );
}

#[test]
#[ignore = "core read-write storage texture format capabilities are not yet keyed to texture-formats-tier2"]
fn bind_group_layout_storage_binding_read_write_access() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureFormatsTier2);
    unsafe {
        for &format in feature_common::tier2_read_write_formats() {
            let layout = feature_common::assert_storage_texture_bgl_ok(
                &test,
                format,
                native::WGPUStorageTextureAccess_ReadWrite,
            );
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

#[test]
#[ignore = "auto layout storage texture inference is not yet covered for texture-formats-tier2"]
fn pipeline_auto_layout_storage_texture() {
    let test = feature_common::test_with_feature(native::WGPUFeatureName_TextureFormatsTier2);
    unsafe {
        let layout = feature_common::assert_storage_texture_bgl_ok(
            &test,
            native::WGPUTextureFormat_RGBA32Float,
            native::WGPUStorageTextureAccess_ReadWrite,
        );
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}
