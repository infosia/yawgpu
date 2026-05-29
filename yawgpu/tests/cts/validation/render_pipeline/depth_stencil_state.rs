//! CTS: src/webgpu/api/validation/render_pipeline/depth_stencil_state.spec.ts

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::render_common::{
    default_primitive, default_stencil_face, depth_stencil_state, expect_render_pipeline,
    RenderPipelineCase,
};

const FRAGMENT_COLOR_AND_DEPTH: &str = "
struct FragmentOut {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}
@fragment fn main() -> FragmentOut {
    var out: FragmentOut;
    out.color = vec4f(0.0, 1.0, 0.0, 1.0);
    out.depth = 0.5;
    return out;
}";

#[test]
#[ignore = "core treats depthCompare=always as depth-test use for stencil-only formats; CTS expects stencil-only depthStencil.format to be valid when depth state is otherwise inert"]
fn format() {
    let test = ValidationTest::new();
    unsafe {
        for (format, success) in [
            (native::WGPUTextureFormat_Depth24Plus, true),
            (native::WGPUTextureFormat_Depth24PlusStencil8, true),
            (native::WGPUTextureFormat_Stencil8, true),
            (native::WGPUTextureFormat_RGBA8Unorm, false),
        ] {
            let depth = depth_stencil_state(
                format,
                native::WGPUOptionalBool_False,
                native::WGPUCompareFunction_Always,
            );
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        depth_stencil_state: Some(&depth),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
fn depth_compare_optional() {
    let test = ValidationTest::new();
    unsafe {
        let mut stencil_depth_fail = default_stencil_face();
        stencil_depth_fail.depthFailOp = native::WGPUStencilOperation_Zero;
        let cases = [
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUCompareFunction_Always,
                native::WGPUOptionalBool_False,
                default_stencil_face(),
                true,
            ),
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUCompareFunction_Undefined,
                native::WGPUOptionalBool_False,
                default_stencil_face(),
                false,
            ),
            (
                native::WGPUTextureFormat_Stencil8,
                native::WGPUCompareFunction_Undefined,
                native::WGPUOptionalBool_Undefined,
                default_stencil_face(),
                true,
            ),
            (
                native::WGPUTextureFormat_Depth24PlusStencil8,
                native::WGPUCompareFunction_Undefined,
                native::WGPUOptionalBool_False,
                stencil_depth_fail,
                false,
            ),
        ];
        for (format, compare, write, stencil_front, success) in cases {
            let mut depth = depth_stencil_state(format, write, compare);
            depth.stencilFront = stencil_front;
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        depth_stencil_state: Some(&depth),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
#[ignore = "core treats depthCompare=always as depth-test use for stencil-only formats; CTS expects depthWriteEnabled to be optional for stencil-only formats"]
fn depth_write_enabled_optional() {
    let test = ValidationTest::new();
    unsafe {
        for (format, success) in [
            (native::WGPUTextureFormat_Depth24Plus, false),
            (native::WGPUTextureFormat_Depth24PlusStencil8, false),
            (native::WGPUTextureFormat_Stencil8, true),
        ] {
            let depth = depth_stencil_state(
                format,
                native::WGPUOptionalBool_Undefined,
                native::WGPUCompareFunction_Always,
            );
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        depth_stencil_state: Some(&depth),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
#[ignore = "core treats depthCompare=always as depth-test use for stencil-only formats; CTS expects compare=always not to enable depth testing on stencil-only formats"]
fn depth_test() {
    let test = ValidationTest::new();
    unsafe {
        for (format, compare, success) in [
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUCompareFunction_Less,
                true,
            ),
            (
                native::WGPUTextureFormat_Stencil8,
                native::WGPUCompareFunction_Always,
                true,
            ),
            (
                native::WGPUTextureFormat_Stencil8,
                native::WGPUCompareFunction_Less,
                false,
            ),
        ] {
            let depth = depth_stencil_state(format, native::WGPUOptionalBool_False, compare);
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        depth_stencil_state: Some(&depth),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
#[ignore = "core treats depthCompare=always as depth-test use for stencil-only formats; CTS expects depthWriteEnabled=false to be valid on stencil-only formats"]
fn depth_write() {
    let test = ValidationTest::new();
    unsafe {
        for (format, write, success) in [
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUOptionalBool_True,
                true,
            ),
            (
                native::WGPUTextureFormat_Stencil8,
                native::WGPUOptionalBool_False,
                true,
            ),
            (
                native::WGPUTextureFormat_Stencil8,
                native::WGPUOptionalBool_True,
                false,
            ),
        ] {
            let depth = depth_stencil_state(format, write, native::WGPUCompareFunction_Always);
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        depth_stencil_state: Some(&depth),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
fn depth_write_frag_depth() {
    let test = ValidationTest::new();
    unsafe {
        let depth = depth_stencil_state(
            native::WGPUTextureFormat_Depth24Plus,
            native::WGPUOptionalBool_True,
            native::WGPUCompareFunction_Always,
        );
        for is_async in [false, true] {
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    fragment_source: Some(FRAGMENT_COLOR_AND_DEPTH),
                    ..Default::default()
                },
            );
            expect_render_pipeline(
                &test,
                is_async,
                true,
                RenderPipelineCase {
                    fragment_source: Some(FRAGMENT_COLOR_AND_DEPTH),
                    depth_stencil_state: Some(&depth),
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn depth_bias() {
    let test = ValidationTest::new();
    unsafe {
        let mut depth = depth_stencil_state(
            native::WGPUTextureFormat_Depth24Plus,
            native::WGPUOptionalBool_True,
            native::WGPUCompareFunction_LessEqual,
        );
        let mut point = default_primitive();
        point.topology = native::WGPUPrimitiveTopology_PointList;
        let mut triangle = default_primitive();
        triangle.topology = native::WGPUPrimitiveTopology_TriangleList;
        for is_async in [false, true] {
            depth.depthBiasSlopeScale = 0.0;
            depth.depthBiasClamp = 0.0;
            depth.depthBias = 1;
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    primitive: Some(point),
                    depth_stencil_state: Some(&depth),
                    ..Default::default()
                },
            );
            expect_render_pipeline(
                &test,
                is_async,
                true,
                RenderPipelineCase {
                    primitive: Some(triangle),
                    depth_stencil_state: Some(&depth),
                    ..Default::default()
                },
            );
            depth.depthBias = 0;
            depth.depthBiasSlopeScale = f32::NAN;
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    primitive: Some(triangle),
                    depth_stencil_state: Some(&depth),
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
#[ignore = "core treats depthCompare=always as depth-test use for stencil-only formats; CTS expects stencil tests to be valid on stencil-only formats"]
fn stencil_test() {
    let test = ValidationTest::new();
    unsafe {
        for (format, compare, success) in [
            (
                native::WGPUTextureFormat_Stencil8,
                native::WGPUCompareFunction_Less,
                true,
            ),
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUCompareFunction_Always,
                true,
            ),
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUCompareFunction_Less,
                false,
            ),
        ] {
            let mut depth = depth_stencil_state(
                format,
                if format == native::WGPUTextureFormat_Depth24Plus {
                    native::WGPUOptionalBool_False
                } else {
                    native::WGPUOptionalBool_Undefined
                },
                native::WGPUCompareFunction_Always,
            );
            depth.stencilFront.compare = compare;
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        depth_stencil_state: Some(&depth),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
#[ignore = "core treats depthCompare=always as depth-test use for stencil-only formats; CTS expects stencil writes to be valid on stencil-only formats"]
fn stencil_write() {
    let test = ValidationTest::new();
    unsafe {
        for (format, op, success) in [
            (
                native::WGPUTextureFormat_Stencil8,
                native::WGPUStencilOperation_Replace,
                true,
            ),
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUStencilOperation_Keep,
                true,
            ),
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUStencilOperation_Replace,
                false,
            ),
        ] {
            let mut depth = depth_stencil_state(
                format,
                if format == native::WGPUTextureFormat_Depth24Plus {
                    native::WGPUOptionalBool_False
                } else {
                    native::WGPUOptionalBool_Undefined
                },
                native::WGPUCompareFunction_Always,
            );
            depth.stencilFront.passOp = op;
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        depth_stencil_state: Some(&depth),
                        ..Default::default()
                    },
                );
            }
        }
    }
}
