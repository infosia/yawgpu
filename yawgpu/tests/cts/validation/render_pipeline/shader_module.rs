//! CTS port of `webgpu/api/validation/render_pipeline/shader_module.spec.ts`.

use yawgpu_test::ValidationTest;

use crate::common::{create_wgsl_module, request_device};
use crate::render_common::{
    expect_render_pipeline, RenderPipelineCase, FRAGMENT_COLOR, VERTEX_NO_INPUT,
};

#[test]
fn device_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let own_vertex = create_wgsl_module(test.device(), VERTEX_NO_INPUT);
        let own_fragment = create_wgsl_module(test.device(), FRAGMENT_COLOR);
        let other_vertex = create_wgsl_module(other, VERTEX_NO_INPUT);
        let other_fragment = create_wgsl_module(other, FRAGMENT_COLOR);
        for is_async in [false, true] {
            expect_render_pipeline(
                &test,
                is_async,
                true,
                RenderPipelineCase {
                    vertex_module: Some(own_vertex),
                    fragment_module: Some(own_fragment),
                    ..Default::default()
                },
            );
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    vertex_module: Some(other_vertex),
                    fragment_module: Some(own_fragment),
                    ..Default::default()
                },
            );
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    vertex_module: Some(own_vertex),
                    fragment_module: Some(other_fragment),
                    ..Default::default()
                },
            );
        }
        yawgpu::wgpuShaderModuleRelease(other_fragment);
        yawgpu::wgpuShaderModuleRelease(other_vertex);
        yawgpu::wgpuShaderModuleRelease(own_fragment);
        yawgpu::wgpuShaderModuleRelease(own_vertex);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn invalid_vertex() {
    let test = ValidationTest::new();
    unsafe {
        let mut invalid = std::ptr::null();
        yawgpu_test::assert_current_device_error_after(
            || {
                invalid = create_wgsl_module(test.device(), "not wgsl @@@");
            },
            None,
        );
        assert!(!invalid.is_null());
        test.clear_errors();
        for is_async in [false, true] {
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    vertex_module: Some(invalid),
                    ..Default::default()
                },
            );
            expect_render_pipeline(&test, is_async, true, RenderPipelineCase::default());
        }
        yawgpu::wgpuShaderModuleRelease(invalid);
    }
}

#[test]
fn invalid_fragment() {
    let test = ValidationTest::new();
    unsafe {
        let mut invalid = std::ptr::null();
        yawgpu_test::assert_current_device_error_after(
            || {
                invalid = create_wgsl_module(test.device(), "not wgsl @@@");
            },
            None,
        );
        assert!(!invalid.is_null());
        test.clear_errors();
        for is_async in [false, true] {
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    fragment_module: Some(invalid),
                    ..Default::default()
                },
            );
            expect_render_pipeline(&test, is_async, true, RenderPipelineCase::default());
        }
        yawgpu::wgpuShaderModuleRelease(invalid);
    }
}
