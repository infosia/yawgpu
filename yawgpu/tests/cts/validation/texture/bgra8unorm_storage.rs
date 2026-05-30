use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::feature_common;

#[test]
#[ignore = "Noop does not advertise bgra8unorm-storage"]
fn create_texture() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_BGRA8UnormStorage);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_BGRA8UnormStorage);
    unsafe {
        let texture = feature_common::assert_texture_ok(
            &test,
            feature_common::texture_descriptor(
                native::WGPUTextureFormat_BGRA8Unorm,
                native::WGPUTextureUsage_StorageBinding,
            ),
        );
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
#[ignore = "Noop does not advertise bgra8unorm-storage"]
fn create_bind_group_layout() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_BGRA8UnormStorage);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_BGRA8UnormStorage);
    unsafe {
        let layout = feature_common::assert_storage_texture_bgl_ok(
            &test,
            native::WGPUTextureFormat_BGRA8Unorm,
            native::WGPUStorageTextureAccess_WriteOnly,
        );
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}

#[test]
#[ignore = "native Noop has no canvas/surface fixture"]
fn configure_storage_usage_on_canvas_context_without_bgra8unorm_storage() {
    let test = ValidationTest::new();
    assert_eq!(
        unsafe {
            yawgpu::wgpuDeviceHasFeature(test.device(), native::WGPUFeatureName_BGRA8UnormStorage)
        },
        0
    );
}

#[test]
#[ignore = "native Noop has no canvas/surface fixture and lacks bgra8unorm-storage"]
fn configure_storage_usage_on_canvas_context_with_bgra8unorm_storage() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_BGRA8UnormStorage);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_BGRA8UnormStorage);
    feature_common::assert_device_has_feature(&test, native::WGPUFeatureName_BGRA8UnormStorage);
}
