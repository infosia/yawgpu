//! CTS: src/webgpu/api/validation/render_pipeline/overrides.spec.ts

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    assert_render_pipeline_descriptor, color_target, constant, create_wgsl_module,
    empty_string_view, string_view, PipelineConstantInput,
};
use crate::render_common::{default_multisample, default_primitive};

const VERTEX_BASIC: &str = "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }";
const FRAGMENT_BASIC: &str = "@fragment fn main() -> @location(0) vec4f { return vec4f(); }";

const VERTEX_IDENTIFIERS: &str = "
override x: f32 = 0.0;
override y: f32 = 0.0;
@id(1) override z: f32 = 0.0;
@id(1000) override w: f32 = 1.0;
@vertex fn main() -> @builtin(position) vec4f {
    return vec4f(x, y, z, w);
}";

const FRAGMENT_IDENTIFIERS: &str = "
override r: f32 = 0.0;
override g: f32 = 0.0;
@id(1) override b: f32 = 0.0;
@id(1000) override a: f32 = 0.0;
@fragment fn main() -> @location(0) vec4f {
    return vec4f(r, g, b, a);
}";

const VERTEX_UNINITIALIZED: &str = "
override x: f32;
override y: f32 = 0.0;
override z: f32;
override w: f32 = 1.0;
@vertex fn main() -> @builtin(position) vec4f {
    return vec4f(x, y, z, w);
}";

const FRAGMENT_UNINITIALIZED: &str = "
override r: f32;
override g: f32 = 0.0;
override b: f32;
override a: f32 = 0.0;
@fragment fn main() -> @location(0) vec4f {
    return vec4f(r, g, b, a);
}";

const VERTEX_VALUES: &str = "
override cb: bool = false;
override cu: u32 = 0u;
override ci: i32 = 0;
override cf: f32 = 0.0;
@vertex fn main() -> @builtin(position) vec4f {
    _ = cb;
    _ = cu;
    _ = ci;
    _ = cf;
    return vec4f();
}";

const FRAGMENT_VALUES: &str = "
override cb: bool = false;
override cu: u32 = 0u;
override ci: i32 = 0;
override cf: f32 = 0.0;
@fragment fn main() -> @location(0) vec4f {
    _ = cb;
    _ = cu;
    _ = ci;
    _ = cf;
    return vec4f();
}";

const VERTEX_F16: &str = "
enable f16;
override cf16: f16 = 0.0h;
@vertex fn main() -> @builtin(position) vec4f {
    _ = cf16;
    return vec4f();
}";

const FRAGMENT_F16: &str = "
enable f16;
override cf16: f16 = 0.0h;
@fragment fn main() -> @location(0) vec4f {
    _ = cf16;
    return vec4f();
}";

#[test]
fn identifier_vertex() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (vec![], true),
            (vec![constant("x", 1.0), constant("y", 1.0)], true),
            (
                vec![
                    constant("x", 1.0),
                    constant("y", 1.0),
                    constant("1", 1.0),
                    constant("1000", 1.0),
                ],
                true,
            ),
            (vec![constant("x\0", 1.0)], false),
            (vec![constant("xxx", 1.0)], false),
            (vec![constant("1", 1.0)], true),
            (vec![constant("2", 1.0)], false),
            (vec![constant("z", 1.0)], false),
            (vec![constant("w", 1.0)], false),
        ];
        for (constants, success) in cases {
            for is_async in [false, true] {
                expect_render_pipeline_with_constants(
                    &test,
                    is_async,
                    success,
                    VERTEX_IDENTIFIERS,
                    &constants,
                    FRAGMENT_BASIC,
                    &[],
                );
            }
        }
    }
}

#[test]
fn identifier_fragment() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (vec![], true),
            (vec![constant("r", 1.0), constant("g", 1.0)], true),
            (
                vec![
                    constant("r", 1.0),
                    constant("g", 1.0),
                    constant("1", 1.0),
                    constant("1000", 1.0),
                ],
                true,
            ),
            (vec![constant("r\0", 1.0)], false),
            (vec![constant("xxx", 1.0)], false),
            (vec![constant("1", 1.0)], true),
            (vec![constant("2", 1.0)], false),
            (vec![constant("b", 1.0)], false),
            (vec![constant("a", 1.0)], false),
        ];
        for (constants, success) in cases {
            for is_async in [false, true] {
                expect_render_pipeline_with_constants(
                    &test,
                    is_async,
                    success,
                    VERTEX_BASIC,
                    &[],
                    FRAGMENT_IDENTIFIERS,
                    &constants,
                );
            }
        }
    }
}

#[test]
fn uninitialized_vertex() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (vec![], false),
            (vec![constant("x", 1.0), constant("y", 1.0)], false),
            (vec![constant("x", 1.0), constant("z", 1.0)], true),
            (
                vec![
                    constant("x", 1.0),
                    constant("y", 1.0),
                    constant("z", 1.0),
                    constant("w", 1.0),
                ],
                true,
            ),
        ];
        for (constants, success) in cases {
            for is_async in [false, true] {
                expect_render_pipeline_with_constants(
                    &test,
                    is_async,
                    success,
                    VERTEX_UNINITIALIZED,
                    &constants,
                    FRAGMENT_BASIC,
                    &[],
                );
            }
        }
    }
}

#[test]
fn uninitialized_fragment() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (vec![], false),
            (vec![constant("r", 1.0), constant("g", 1.0)], false),
            (vec![constant("r", 1.0), constant("b", 1.0)], true),
            (
                vec![
                    constant("r", 1.0),
                    constant("g", 1.0),
                    constant("b", 1.0),
                    constant("a", 1.0),
                ],
                true,
            ),
        ];
        for (constants, success) in cases {
            for is_async in [false, true] {
                expect_render_pipeline_with_constants(
                    &test,
                    is_async,
                    success,
                    VERTEX_BASIC,
                    &[],
                    FRAGMENT_UNINITIALIZED,
                    &constants,
                );
            }
        }
    }
}

#[test]
fn value_type_error_vertex() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (1.0, true),
            (f64::NAN, false),
            (f64::INFINITY, false),
            (f64::NEG_INFINITY, false),
        ];
        for (value, success) in cases {
            for is_async in [false, true] {
                expect_render_pipeline_with_constants(
                    &test,
                    is_async,
                    success,
                    VERTEX_VALUES,
                    &[constant("cf", value)],
                    FRAGMENT_BASIC,
                    &[],
                );
            }
        }
    }
}

#[test]
fn value_type_error_fragment() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (1.0, true),
            (f64::NAN, false),
            (f64::INFINITY, false),
            (f64::NEG_INFINITY, false),
        ];
        for (value, success) in cases {
            for is_async in [false, true] {
                expect_render_pipeline_with_constants(
                    &test,
                    is_async,
                    success,
                    VERTEX_BASIC,
                    &[],
                    FRAGMENT_VALUES,
                    &[constant("cf", value)],
                );
            }
        }
    }
}

#[test]
fn value_validation_error_vertex() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (constant("cu", 0.0), true),
            (constant("cu", -1.0), false),
            (constant("cu", f64::from(u32::MAX)), true),
            (constant("cu", f64::from(u32::MAX) + 1.0), false),
            (constant("ci", f64::from(i32::MIN)), true),
            (constant("ci", f64::from(i32::MIN) - 1.0), false),
            (constant("ci", f64::from(i32::MAX)), true),
            (constant("ci", f64::from(i32::MAX) + 1.0), false),
            (constant("cf", f64::from(f32::MAX)), true),
            (constant("cf", f64::from(f32::MAX) * 2.0), false),
            (constant("cb", f64::MAX), true),
        ];
        for (constant, success) in cases {
            for is_async in [false, true] {
                expect_render_pipeline_with_constants(
                    &test,
                    is_async,
                    success,
                    VERTEX_VALUES,
                    &[constant],
                    FRAGMENT_BASIC,
                    &[],
                );
            }
        }
    }
}

#[test]
fn value_validation_error_fragment() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (constant("cu", 0.0), true),
            (constant("cu", -1.0), false),
            (constant("cu", f64::from(u32::MAX)), true),
            (constant("cu", f64::from(u32::MAX) + 1.0), false),
            (constant("ci", f64::from(i32::MIN)), true),
            (constant("ci", f64::from(i32::MIN) - 1.0), false),
            (constant("ci", f64::from(i32::MAX)), true),
            (constant("ci", f64::from(i32::MAX) + 1.0), false),
            (constant("cf", f64::from(f32::MAX)), true),
            (constant("cf", f64::from(f32::MAX) * 2.0), false),
            (constant("cb", f64::MAX), true),
        ];
        for (constant, success) in cases {
            for is_async in [false, true] {
                expect_render_pipeline_with_constants(
                    &test,
                    is_async,
                    success,
                    VERTEX_BASIC,
                    &[],
                    FRAGMENT_VALUES,
                    &[constant],
                );
            }
        }
    }
}

#[test]
#[ignore = "Noop does not advertise shader-f16; CTS expects f16 override values outside the f16 range to fail when the feature is enabled"]
fn value_validation_error_f16_vertex() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_ShaderF16]);
    unsafe {
        for (value, success) in [
            (65_504.0, true),
            (65_505.0, false),
            (f64::from(f32::MAX), false),
        ] {
            for is_async in [false, true] {
                expect_render_pipeline_with_constants(
                    &test,
                    is_async,
                    success,
                    VERTEX_F16,
                    &[constant("cf16", value)],
                    FRAGMENT_BASIC,
                    &[],
                );
            }
        }
    }
}

#[test]
#[ignore = "Noop does not advertise shader-f16; CTS expects f16 override values outside the f16 range to fail when the feature is enabled"]
fn value_validation_error_f16_fragment() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_ShaderF16]);
    unsafe {
        for (value, success) in [
            (65_504.0, true),
            (65_505.0, false),
            (f64::from(f32::MAX), false),
        ] {
            for is_async in [false, true] {
                expect_render_pipeline_with_constants(
                    &test,
                    is_async,
                    success,
                    VERTEX_BASIC,
                    &[],
                    FRAGMENT_F16,
                    &[constant("cf16", value)],
                );
            }
        }
    }
}

unsafe fn expect_render_pipeline_with_constants(
    test: &ValidationTest,
    is_async: bool,
    success: bool,
    vertex_source: &str,
    vertex_constants: &[PipelineConstantInput<'_>],
    fragment_source: &str,
    fragment_constants: &[PipelineConstantInput<'_>],
) {
    let vertex_module = unsafe { create_wgsl_module(test.device(), vertex_source) };
    let fragment_module = unsafe { create_wgsl_module(test.device(), fragment_source) };
    let vertex_entries = native_constants(vertex_constants);
    let fragment_entries = native_constants(fragment_constants);
    let target = color_target();
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment_module,
        entryPoint: empty_string_view(),
        constantCount: fragment_entries.len(),
        constants: fragment_entries.as_ptr(),
        targetCount: 1,
        targets: &target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex_module,
            entryPoint: empty_string_view(),
            constantCount: vertex_entries.len(),
            constants: vertex_entries.as_ptr(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: default_primitive(),
        depthStencil: std::ptr::null(),
        multisample: default_multisample(),
        fragment: &fragment,
    };
    unsafe {
        assert_render_pipeline_descriptor(test, is_async, success, &descriptor);
        yawgpu::wgpuShaderModuleRelease(fragment_module);
        yawgpu::wgpuShaderModuleRelease(vertex_module);
    }
}

fn native_constants(constants: &[PipelineConstantInput<'_>]) -> Vec<native::WGPUConstantEntry> {
    constants
        .iter()
        .map(|constant| native::WGPUConstantEntry {
            nextInChain: std::ptr::null_mut(),
            key: string_view(constant.key),
            value: constant.value,
        })
        .collect()
}
