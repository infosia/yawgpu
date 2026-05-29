use yawgpu_test::ValidationTest;

use crate::common::{
    assert_compute_pipeline_error, assert_compute_pipeline_ok, assert_render_pipeline_error,
    assert_render_pipeline_ok, depth_state, FragmentInput,
};

#[test]
fn compute() {
    let test = ValidationTest::new();
    unsafe {
        let compute = "@compute @workgroup_size(1) fn main() {}";
        let vertex = "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }";
        let fragment = "@fragment fn main() -> @location(0) vec4f { return vec4f(); }";
        for is_async in [false, true] {
            assert_compute_pipeline_ok(&test, is_async, compute, Some("main"), &[], None);
            assert_compute_pipeline_error(&test, is_async, vertex, Some("main"), &[], None);
            assert_compute_pipeline_error(&test, is_async, fragment, Some("main"), &[], None);
            assert_compute_pipeline_error(&test, is_async, compute, Some("missing"), &[], None);
        }
    }
}

#[test]
fn vertex() {
    let test = ValidationTest::new();
    unsafe {
        let vertex = "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }";
        let compute = "@compute @workgroup_size(1) fn main() {}";
        let fragment = "@fragment fn main() -> @location(0) vec4f { return vec4f(); }";
        for is_async in [false, true] {
            assert_render_pipeline_ok(
                &test,
                is_async,
                vertex,
                Some("main"),
                None,
                None,
                Some(depth_state()),
            );
            assert_render_pipeline_error(
                &test,
                is_async,
                compute,
                Some("main"),
                None,
                None,
                Some(depth_state()),
            );
            assert_render_pipeline_error(
                &test,
                is_async,
                fragment,
                Some("main"),
                None,
                None,
                Some(depth_state()),
            );
            assert_render_pipeline_error(
                &test,
                is_async,
                vertex,
                Some("missing"),
                None,
                None,
                Some(depth_state()),
            );
        }
    }
}

#[test]
fn fragment() {
    let test = ValidationTest::new();
    unsafe {
        let vertex = "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }";
        let fragment = "@fragment fn main() -> @location(0) vec4f { return vec4f(); }";
        let compute = "@compute @workgroup_size(1) fn main() {}";
        let wrong_vertex = "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }";
        for is_async in [false, true] {
            assert_render_pipeline_ok(
                &test,
                is_async,
                vertex,
                Some("vs"),
                Some(FragmentInput::new(fragment, Some("main"), 1)),
                None,
                None,
            );
            assert_render_pipeline_error(
                &test,
                is_async,
                vertex,
                Some("vs"),
                Some(FragmentInput::new(compute, Some("main"), 1)),
                None,
                None,
            );
            assert_render_pipeline_error(
                &test,
                is_async,
                vertex,
                Some("vs"),
                Some(FragmentInput::new(wrong_vertex, Some("main"), 1)),
                None,
                None,
            );
            assert_render_pipeline_error(
                &test,
                is_async,
                vertex,
                Some("vs"),
                Some(FragmentInput::new(fragment, Some("missing"), 1)),
                None,
                None,
            );
        }
    }
}

#[test]
fn compute_undefined_entry_point_and_extra_stage() {
    let test = ValidationTest::new();
    unsafe {
        for is_async in [false, true] {
            assert_compute_pipeline_ok(
                &test,
                is_async,
                "@compute @workgroup_size(1) fn main() {}
                 @vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }",
                None,
                &[],
                None,
            );
            assert_compute_pipeline_error(
                &test,
                is_async,
                "@compute @workgroup_size(1) fn a() {}
                 @compute @workgroup_size(1) fn b() {}
                 @vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }",
                None,
                &[],
                None,
            );
        }
    }
}

#[test]
fn vertex_undefined_entry_point_and_extra_stage() {
    let test = ValidationTest::new();
    unsafe {
        for is_async in [false, true] {
            assert_render_pipeline_ok(
                &test,
                is_async,
                "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }
                 @compute @workgroup_size(1) fn cs() {}",
                None,
                None,
                None,
                Some(depth_state()),
            );
            assert_render_pipeline_error(
                &test,
                is_async,
                "@vertex fn a() -> @builtin(position) vec4f { return vec4f(); }
                 @vertex fn b() -> @builtin(position) vec4f { return vec4f(); }
                 @compute @workgroup_size(1) fn cs() {}",
                None,
                None,
                None,
                Some(depth_state()),
            );
        }
    }
}

#[test]
fn fragment_undefined_entry_point_and_extra_stage() {
    let test = ValidationTest::new();
    unsafe {
        let vertex = "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }";
        for is_async in [false, true] {
            assert_render_pipeline_ok(
                &test,
                is_async,
                vertex,
                Some("vs"),
                Some(FragmentInput::new(
                    "@fragment fn main() -> @location(0) vec4f { return vec4f(); }
                     @compute @workgroup_size(1) fn cs() {}",
                    None,
                    1,
                )),
                None,
                None,
            );
            assert_render_pipeline_error(
                &test,
                is_async,
                vertex,
                Some("vs"),
                Some(FragmentInput::new(
                    "@fragment fn a() -> @location(0) vec4f { return vec4f(); }
                     @fragment fn b() -> @location(0) vec4f { return vec4f(); }
                     @compute @workgroup_size(1) fn cs() {}",
                    None,
                    1,
                )),
                None,
                None,
            );
        }
    }
}
