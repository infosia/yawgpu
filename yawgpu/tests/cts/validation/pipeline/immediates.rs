use yawgpu_test::ValidationTest;

use crate::common::{
    assert_compute_pipeline_error, assert_compute_pipeline_ok, create_pipeline_layout,
    device_limits,
};

// CTS: pipeline_creation_immediate_size_mismatch.
// The C header exposes pipeline layout `immediateSize`; yawgpu currently models
// no shader-side immediate declarations, so this port covers only the API size
// validation observable at pipeline-layout creation and reports the shader
// mismatch portion as N/A.
#[test]
fn pipeline_creation_immediate_size_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let valid_layout = create_pipeline_layout(test.device(), &[], limits.maxImmediateSize);
        assert!(!valid_layout.is_null());
        for is_async in [false, true] {
            assert_compute_pipeline_ok(
                &test,
                is_async,
                "@compute @workgroup_size(1) fn main() {}",
                Some("main"),
                &[],
                Some(valid_layout),
            );
        }
        yawgpu::wgpuPipelineLayoutRelease(valid_layout);

        let mut invalid_layout = std::ptr::null();
        yawgpu_test::assert_current_device_error_after(
            || {
                invalid_layout =
                    create_pipeline_layout(test.device(), &[], limits.maxImmediateSize + 1);
            },
            None,
        );
        assert!(!invalid_layout.is_null());
        for is_async in [false, true] {
            assert_compute_pipeline_error(
                &test,
                is_async,
                "@compute @workgroup_size(1) fn main() {}",
                Some("main"),
                &[],
                Some(invalid_layout),
            );
        }
        yawgpu::wgpuPipelineLayoutRelease(invalid_layout);
    }
}
