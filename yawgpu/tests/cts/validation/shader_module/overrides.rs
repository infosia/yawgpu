use yawgpu_test::ValidationTest;

use crate::common::{
    assert_compute_pipeline_error, assert_compute_pipeline_ok, constant, create_wgsl_module,
};

#[test]
fn id_conflict() {
    let test = ValidationTest::new();
    unsafe {
        let source = "
@id(0) override c0: u32 = 1u;
@id(0) override c1: u32 = 2u;
@compute @workgroup_size(1) fn main() {
  _ = c0 + c1;
}";
        let mut module = std::ptr::null();
        yawgpu_test::assert_current_device_error_after(
            || {
                module = create_wgsl_module(test.device(), source);
            },
            None,
        );
        assert!(!module.is_null());
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

#[test]
fn name_conflict() {
    let test = ValidationTest::new();
    unsafe {
        let source = "
override c0: u32 = 1u;
@id(1) override c1: u32 = 2u;
@compute @workgroup_size(1) fn main() {
  _ = c0 + c1;
}";
        for is_async in [false, true] {
            assert_compute_pipeline_ok(
                &test,
                is_async,
                source,
                Some("main"),
                &[constant("c0", 3.0), constant("1", 4.0)],
                None,
            );
            assert_compute_pipeline_error(
                &test,
                is_async,
                source,
                Some("main"),
                &[constant("c1", 4.0)],
                None,
            );
        }
    }
}
