use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::feature_common;

#[test]
#[ignore = "core does not yet gate optional texture formats by device feature"]
fn texture_descriptor() {
    let test = ValidationTest::new();
    unsafe {
        for &format in feature_common::optional_formats() {
            feature_common::assert_texture_error(
                &test,
                feature_common::texture_descriptor(format, native::WGPUTextureUsage_TextureBinding),
            );
        }
    }
}

#[test]
#[ignore = "core does not yet gate optional texture view formats by device feature"]
fn texture_descriptor_view_formats() {
    let test = ValidationTest::new();
    unsafe {
        feature_common::assert_texture_view_format_error(
            &test,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_RGBA8UnormSrgb,
        );
    }
}

#[test]
#[ignore = "core does not yet gate optional texture view formats by device feature"]
fn texture_view_descriptor() {
    let test = ValidationTest::new();
    unsafe {
        feature_common::assert_texture_view_format_error(
            &test,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_RGBA8UnormSrgb,
        );
    }
}

#[test]
#[ignore = "Noop does not advertise texture-compression-bc-sliced-3d"]
fn texture_compression_bc_sliced_3d() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_TextureCompressionBCSliced3D);
    let test =
        feature_common::test_with_feature(native::WGPUFeatureName_TextureCompressionBCSliced3D);
    unsafe {
        let texture = feature_common::assert_texture_ok(
            &test,
            feature_common::texture_descriptor_3d(
                native::WGPUTextureFormat_BC1RGBAUnorm,
                native::WGPUTextureUsage_TextureBinding,
            ),
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
#[ignore = "Noop does not advertise texture-compression-astc-sliced-3d"]
fn texture_compression_astc_sliced_3d() {
    feature_common::assert_noop_lacks_feature(
        native::WGPUFeatureName_TextureCompressionASTCSliced3D,
    );
    let test =
        feature_common::test_with_feature(native::WGPUFeatureName_TextureCompressionASTCSliced3D);
    unsafe {
        let texture = feature_common::assert_texture_ok(
            &test,
            feature_common::texture_descriptor_3d(
                native::WGPUTextureFormat_ASTC4x4Unorm,
                native::WGPUTextureUsage_TextureBinding,
            ),
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
#[ignore = "native Noop has no canvas/surface fixture"]
fn canvas_configuration() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_TextureCompressionBC);
}

#[test]
#[ignore = "native Noop has no canvas/surface fixture"]
fn canvas_configuration_view_formats() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_TextureCompressionBC);
}

#[test]
#[ignore = "core does not yet gate storage texture formats by device feature"]
fn storage_texture_binding_layout() {
    let test = ValidationTest::new();
    unsafe {
        feature_common::assert_storage_texture_bgl_error(
            &test,
            native::WGPUTextureFormat_RGBA8UnormSrgb,
            native::WGPUStorageTextureAccess_WriteOnly,
        );
    }
}

#[test]
#[ignore = "core does not yet gate color target formats by device feature"]
fn color_target_state() {
    let test = ValidationTest::new();
    unsafe {
        feature_common::assert_color_target_pipeline_error(
            &test,
            native::WGPUTextureFormat_RG11B10Ufloat,
        );
    }
}

#[test]
#[ignore = "core does not yet gate depth-stencil formats by device feature"]
fn depth_stencil_state() {
    let test = ValidationTest::new();
    unsafe {
        feature_common::assert_render_bundle_encoder_error(
            &test,
            native::WGPUTextureFormat_Undefined,
            native::WGPUTextureFormat_Depth32FloatStencil8,
        );
    }
}

#[test]
#[ignore = "core does not yet gate render bundle color formats by device feature"]
fn render_bundle_encoder_descriptor_color_format() {
    let test = ValidationTest::new();
    unsafe {
        feature_common::assert_render_bundle_encoder_error(
            &test,
            native::WGPUTextureFormat_RG11B10Ufloat,
            native::WGPUTextureFormat_Undefined,
        );
    }
}

#[test]
#[ignore = "core does not yet gate render bundle depth-stencil formats by device feature"]
fn render_bundle_encoder_descriptor_depth_stencil_format() {
    let test = ValidationTest::new();
    unsafe {
        feature_common::assert_render_bundle_encoder_error(
            &test,
            native::WGPUTextureFormat_Undefined,
            native::WGPUTextureFormat_Depth32FloatStencil8,
        );
    }
}

#[test]
fn check_capability_guarantees() {
    assert!(feature_common::adapter_has_feature(
        native::WGPUFeatureName_TextureFormatsTier2
    ));
    assert!(feature_common::adapter_has_feature(
        native::WGPUFeatureName_TextureFormatsTier1
    ));
    assert!(feature_common::adapter_has_feature(
        native::WGPUFeatureName_RG11B10UfloatRenderable
    ));
    assert!(!feature_common::adapter_has_feature(
        native::WGPUFeatureName_TextureCompressionBC
    ));
    assert!(!feature_common::adapter_has_feature(
        native::WGPUFeatureName_TextureCompressionASTC
    ));
    assert!(!feature_common::adapter_has_feature(
        native::WGPUFeatureName_Depth32FloatStencil8
    ));
}
