//! CTS: src/webgpu/api/validation/render_pipeline/fragment_state.spec.ts

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::device_limits;
use crate::render_common::{
    blend_component, blend_state, color_target_with, default_blend_component, depth_stencil_state,
    expect_render_pipeline, RenderPipelineCase, FRAGMENT_COLOR,
};

const FRAGMENT_EMPTY: &str = "@fragment fn main() {}";
const FRAGMENT_U32: &str = "@fragment fn main() -> @location(0) vec4u { return vec4u(); }";
const FRAGMENT_F32_ONE: &str = "@fragment fn main() -> @location(0) f32 { return 1.0; }";
const FRAGMENT_F32_TWO: &str = "@fragment fn main() -> @location(0) vec2f { return vec2f(); }";

#[test]
fn color_target_exists() {
    let test = ValidationTest::new();
    unsafe {
        for is_async in [false, true] {
            expect_render_pipeline(&test, is_async, true, RenderPipelineCase::default());
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    fragment_has_target: false,
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn targets_format_is_color_format() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (native::WGPUTextureFormat_RGBA8Unorm, FRAGMENT_COLOR, true),
            (native::WGPUTextureFormat_Depth24Plus, FRAGMENT_COLOR, false),
            (native::WGPUTextureFormat_Stencil8, FRAGMENT_COLOR, false),
            (native::WGPUTextureFormat_RGBA8Uint, FRAGMENT_U32, true),
        ];
        for (format, fragment_source, success) in cases {
            let target = color_target_with(format, None, native::WGPUColorWriteMask_All);
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        fragment_source: Some(fragment_source),
                        fragment_targets: Some(std::slice::from_ref(&target)),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
fn targets_format_renderable() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (native::WGPUTextureFormat_RGBA8Unorm, true),
            (native::WGPUTextureFormat_RGBA8Snorm, false),
            (native::WGPUTextureFormat_RGB9E5Ufloat, false),
        ];
        for (format, success) in cases {
            let target = color_target_with(format, None, native::WGPUColorWriteMask_All);
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        fragment_targets: Some(std::slice::from_ref(&target)),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
fn limits_max_color_attachments() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for count in [limits.maxColorAttachments, limits.maxColorAttachments + 1] {
            let targets = (0..count)
                .map(|_| {
                    color_target_with(
                        native::WGPUTextureFormat_RG8Unorm,
                        None,
                        native::WGPUColorWriteMask_None,
                    )
                })
                .collect::<Vec<_>>();
            let depth = depth_stencil_state(
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUOptionalBool_True,
                native::WGPUCompareFunction_Always,
            );
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    count <= limits.maxColorAttachments,
                    RenderPipelineCase {
                        fragment_targets: Some(&targets),
                        depth_stencil_state: Some(&depth),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
fn limits_max_color_attachment_bytes_per_sample_aligned() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        let cases = [
            (native::WGPUTextureFormat_RGBA8Unorm, 1, true),
            (
                native::WGPUTextureFormat_RGBA32Float,
                1,
                16 <= limits.maxColorAttachmentBytesPerSample,
            ),
            (
                native::WGPUTextureFormat_RGBA32Float,
                3,
                48 <= limits.maxColorAttachmentBytesPerSample,
            ),
        ];
        for (format, count, success) in cases {
            let targets = (0..count)
                .map(|_| color_target_with(format, None, native::WGPUColorWriteMask_None))
                .collect::<Vec<_>>();
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        fragment_targets: Some(&targets),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
fn limits_max_color_attachment_bytes_per_sample_unaligned() {
    let test = ValidationTest::new();
    unsafe {
        let formats = [
            native::WGPUTextureFormat_R8Unorm,
            native::WGPUTextureFormat_R32Float,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_RGBA32Float,
            native::WGPUTextureFormat_R8Unorm,
        ];
        let targets = formats
            .into_iter()
            .map(|format| color_target_with(format, None, native::WGPUColorWriteMask_None))
            .collect::<Vec<_>>();
        for is_async in [false, true] {
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    fragment_targets: Some(&targets),
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
fn targets_format_filterable() {
    let test = ValidationTest::new();
    unsafe {
        let blend = blend_state(default_blend_component(), default_blend_component());
        let cases = [
            (native::WGPUTextureFormat_RGBA8Unorm, false, true),
            (native::WGPUTextureFormat_RGBA8Unorm, true, true),
            (native::WGPUTextureFormat_RGBA32Float, false, true),
            (native::WGPUTextureFormat_RGBA32Float, true, false),
        ];
        for (format, has_blend, success) in cases {
            let target = color_target_with(
                format,
                has_blend.then_some(&blend),
                native::WGPUColorWriteMask_All,
            );
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        fragment_targets: Some(std::slice::from_ref(&target)),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
fn targets_blend() {
    let test = ValidationTest::new();
    unsafe {
        let valid = blend_state(
            blend_component(
                native::WGPUBlendOperation_Min,
                native::WGPUBlendFactor_One,
                native::WGPUBlendFactor_One,
            ),
            default_blend_component(),
        );
        let invalid = blend_state(
            blend_component(
                native::WGPUBlendOperation_Min,
                native::WGPUBlendFactor_SrcAlpha,
                native::WGPUBlendFactor_One,
            ),
            default_blend_component(),
        );
        for (blend, success) in [(&valid, true), (&invalid, false)] {
            let target = color_target_with(
                native::WGPUTextureFormat_RGBA8Unorm,
                Some(blend),
                native::WGPUColorWriteMask_All,
            );
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        fragment_targets: Some(std::slice::from_ref(&target)),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
fn targets_write_mask() {
    let test = ValidationTest::new();
    unsafe {
        for (write_mask, success) in [(0, true), (0xF, true), (0x10, false), (0x8000_0001, false)] {
            let target = color_target_with(native::WGPUTextureFormat_RGBA8Unorm, None, write_mask);
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        fragment_targets: Some(std::slice::from_ref(&target)),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
fn pipeline_output_targets() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [
            (
                native::WGPUTextureFormat_RGBA8Unorm,
                FRAGMENT_COLOR,
                native::WGPUColorWriteMask_All,
                true,
            ),
            (
                native::WGPUTextureFormat_RGBA8Uint,
                FRAGMENT_COLOR,
                native::WGPUColorWriteMask_All,
                false,
            ),
            (
                native::WGPUTextureFormat_RGBA8Unorm,
                FRAGMENT_U32,
                native::WGPUColorWriteMask_All,
                false,
            ),
            (
                native::WGPUTextureFormat_RGBA8Unorm,
                FRAGMENT_EMPTY,
                native::WGPUColorWriteMask_None,
                true,
            ),
            (
                native::WGPUTextureFormat_RGBA8Unorm,
                FRAGMENT_EMPTY,
                native::WGPUColorWriteMask_All,
                false,
            ),
            (
                native::WGPUTextureFormat_RGBA8Unorm,
                FRAGMENT_F32_TWO,
                native::WGPUColorWriteMask_All,
                false,
            ),
        ];
        let depth = depth_stencil_state(
            native::WGPUTextureFormat_Depth24Plus,
            native::WGPUOptionalBool_False,
            native::WGPUCompareFunction_Always,
        );
        for (format, fragment_source, write_mask, success) in cases {
            let target = color_target_with(format, None, write_mask);
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        fragment_source: Some(fragment_source),
                        fragment_targets: Some(std::slice::from_ref(&target)),
                        depth_stencil_state: Some(&depth),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
fn pipeline_output_targets_blend() {
    let test = ValidationTest::new();
    unsafe {
        let blend = blend_state(
            blend_component(
                native::WGPUBlendOperation_Add,
                native::WGPUBlendFactor_SrcAlpha,
                native::WGPUBlendFactor_Zero,
            ),
            default_blend_component(),
        );
        for (fragment_source, success) in [(FRAGMENT_COLOR, true), (FRAGMENT_F32_ONE, false)] {
            let target = color_target_with(
                native::WGPUTextureFormat_R8Unorm,
                Some(&blend),
                native::WGPUColorWriteMask_All,
            );
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        fragment_source: Some(fragment_source),
                        fragment_targets: Some(std::slice::from_ref(&target)),
                        ..Default::default()
                    },
                );
            }
        }
    }
}

#[test]
#[ignore = "Noop does not advertise dual-source-blending, and core does not yet validate dual-source color target count; CTS expects src1 factors with more than one target to fail"]
fn dual_source_blending_color_target_count() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_DualSourceBlending]);
    unsafe {
        let blend = blend_state(
            blend_component(
                native::WGPUBlendOperation_Add,
                native::WGPUBlendFactor_Src1,
                native::WGPUBlendFactor_OneMinusSrc1,
            ),
            default_blend_component(),
        );
        for (count, success) in [(1, true), (2, false)] {
            let targets = (0..count)
                .map(|_| {
                    color_target_with(
                        native::WGPUTextureFormat_RGBA8Unorm,
                        Some(&blend),
                        native::WGPUColorWriteMask_All,
                    )
                })
                .collect::<Vec<_>>();
            expect_render_pipeline(
                &test,
                false,
                success,
                RenderPipelineCase {
                    fragment_targets: Some(&targets),
                    ..Default::default()
                },
            );
        }
    }
}

#[test]
#[ignore = "Noop does not advertise dual-source-blending, and core does not yet validate that src1 blend factors require fragment dual-source output; CTS expects missing @blend_src to fail"]
fn dual_source_blending_use_blend_src() {
    let test = ValidationTest::with_features(&[native::WGPUFeatureName_DualSourceBlending]);
    unsafe {
        let blend = blend_state(
            blend_component(
                native::WGPUBlendOperation_Add,
                native::WGPUBlendFactor_Src1,
                native::WGPUBlendFactor_Zero,
            ),
            default_blend_component(),
        );
        let target = color_target_with(
            native::WGPUTextureFormat_RGBA8Unorm,
            Some(&blend),
            native::WGPUColorWriteMask_All,
        );
        expect_render_pipeline(
            &test,
            false,
            false,
            RenderPipelineCase {
                fragment_targets: Some(std::slice::from_ref(&target)),
                ..Default::default()
            },
        );
    }
}
