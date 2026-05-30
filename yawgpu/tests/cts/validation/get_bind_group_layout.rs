//! CTS: src/webgpu/api/validation/getBindGroupLayout.spec.ts
//!
//! The `unique_js_object,*` CTS cases assert JavaScript wrapper identity and
//! expando-property behavior. The C ABI has ref-counted handles instead, so
//! those tests are adapted to assert repeated in-range calls return valid
//! handles without device errors.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::bind_group_common::{expect_bind_group_layout, release_pipeline_layout};
use crate::common::{
    color_target, create_pipeline_layout, create_wgsl_module, device_limits, empty_string_view,
    uniform_layout,
};

#[test]
fn index_range_explicit_layout() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let bgl = expect_bind_group_layout(&test, true, &[]);
        let layout = create_pipeline_layout(test.device(), &[bgl], 0);
        let pipeline = create_render_pipeline(&test, Some(layout), FRAGMENT_EMPTY);
        for index in 0..=limits.maxBindGroups + 1 {
            expect_get_render_bind_group_layout(
                &test,
                pipeline,
                index,
                index < limits.maxBindGroups,
            );
        }
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        release_pipeline_layout(layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
    }
}

#[test]
fn index_range_auto_layout() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let pipeline = create_render_pipeline(&test, None, FRAGMENT_UNIFORM);
        for index in 0..=limits.maxBindGroups + 1 {
            expect_get_render_bind_group_layout(
                &test,
                pipeline,
                index,
                index < limits.maxBindGroups,
            );
        }
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[test]
fn unique_js_object_auto_layout() {
    let test = ValidationTest::new();
    unsafe {
        let pipeline = create_render_pipeline(&test, None, FRAGMENT_UNIFORM);
        expect_repeated_get_render_bind_group_layout(&test, pipeline, 0);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[test]
fn unique_js_object_explicit_layout() {
    let test = ValidationTest::new();
    unsafe {
        let bgl = expect_bind_group_layout(
            &test,
            true,
            &[uniform_layout(0, native::WGPUShaderStage_Fragment, 4)],
        );
        let layout = create_pipeline_layout(test.device(), &[bgl], 0);
        let pipeline = create_render_pipeline(&test, Some(layout), FRAGMENT_UNIFORM);
        expect_repeated_get_render_bind_group_layout(&test, pipeline, 0);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        release_pipeline_layout(layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
    }
}

const VERTEX: &str = "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }";
const FRAGMENT_EMPTY: &str = "@fragment fn main() -> @location(0) vec4f { return vec4f(); }";
const FRAGMENT_UNIFORM: &str = "
@group(0) @binding(0) var<uniform> binding: f32;
@fragment fn main() -> @location(0) vec4f {
    _ = binding;
    return vec4f();
}";

unsafe fn expect_repeated_get_render_bind_group_layout(
    test: &ValidationTest,
    pipeline: native::WGPURenderPipeline,
    index: u32,
) {
    unsafe {
        test.clear_errors();
        let first = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, index);
        assert!(!first.is_null());
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );
        test.clear_errors();
        let second = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, index);
        assert!(!second.is_null());
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );
        yawgpu::wgpuBindGroupLayoutRelease(second);
        yawgpu::wgpuBindGroupLayoutRelease(first);
    }
}

unsafe fn expect_get_render_bind_group_layout(
    test: &ValidationTest,
    pipeline: native::WGPURenderPipeline,
    index: u32,
    success: bool,
) {
    unsafe {
        if success {
            test.clear_errors();
            let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, index);
            assert!(!layout.is_null());
            assert!(
                test.errors().is_empty(),
                "unexpected errors: {:?}",
                test.errors()
            );
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        } else {
            let mut layout = std::ptr::null();
            test.assert_device_error_after(
                || {
                    layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, index);
                },
                None,
            );
            assert!(!layout.is_null());
            yawgpu::wgpuBindGroupLayoutRelease(layout);
        }
    }
}

unsafe fn create_render_pipeline(
    test: &ValidationTest,
    layout: Option<native::WGPUPipelineLayout>,
    fragment_source: &str,
) -> native::WGPURenderPipeline {
    let vertex_module = unsafe { create_wgsl_module(test.device(), VERTEX) };
    let fragment_module = unsafe { create_wgsl_module(test.device(), fragment_source) };
    let target = color_target();
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment_module,
        entryPoint: empty_string_view(),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: layout.unwrap_or(std::ptr::null()),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex_module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: native::WGPUFrontFace_Undefined,
            cullMode: native::WGPUCullMode_Undefined,
            unclippedDepth: 0,
        },
        depthStencil: std::ptr::null(),
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: 1,
            mask: 0xFFFF_FFFF,
            alphaToCoverageEnabled: 0,
        },
        fragment: &fragment,
    };
    let pipeline = unsafe { yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor) };
    unsafe {
        yawgpu::wgpuShaderModuleRelease(fragment_module);
        yawgpu::wgpuShaderModuleRelease(vertex_module);
    }
    pipeline
}
