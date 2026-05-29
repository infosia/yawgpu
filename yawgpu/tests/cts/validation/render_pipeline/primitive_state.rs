//! CTS port of `webgpu/api/validation/render_pipeline/primitive_state.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::render_common::{default_primitive, expect_render_pipeline, RenderPipelineCase};

#[test]
fn strip_index_format() {
    let test = ValidationTest::new();
    unsafe {
        for topology in [
            native::WGPUPrimitiveTopology_Undefined,
            native::WGPUPrimitiveTopology_PointList,
            native::WGPUPrimitiveTopology_LineList,
            native::WGPUPrimitiveTopology_LineStrip,
            native::WGPUPrimitiveTopology_TriangleList,
            native::WGPUPrimitiveTopology_TriangleStrip,
        ] {
            for strip_index_format in [
                native::WGPUIndexFormat_Undefined,
                native::WGPUIndexFormat_Uint16,
                native::WGPUIndexFormat_Uint32,
            ] {
                let mut primitive = default_primitive();
                primitive.topology = topology;
                primitive.stripIndexFormat = strip_index_format;
                let success = topology == native::WGPUPrimitiveTopology_LineStrip
                    || topology == native::WGPUPrimitiveTopology_TriangleStrip
                    || strip_index_format == native::WGPUIndexFormat_Undefined;
                for is_async in [false, true] {
                    expect_render_pipeline(
                        &test,
                        is_async,
                        success,
                        RenderPipelineCase {
                            primitive: Some(primitive),
                            ..Default::default()
                        },
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "core does not yet require depth-clip-control for primitive.unclippedDepth; CTS expects unclippedDepth=true without the feature to fail"]
fn unclipped_depth() {
    let test = ValidationTest::new();
    unsafe {
        let mut primitive = default_primitive();
        primitive.unclippedDepth = 0;
        for is_async in [false, true] {
            expect_render_pipeline(
                &test,
                is_async,
                true,
                RenderPipelineCase {
                    primitive: Some(primitive),
                    ..Default::default()
                },
            );
        }

        primitive.unclippedDepth = 1;
        for is_async in [false, true] {
            expect_render_pipeline(
                &test,
                is_async,
                false,
                RenderPipelineCase {
                    primitive: Some(primitive),
                    ..Default::default()
                },
            );
        }

        if yawgpu::wgpuAdapterHasFeature(test.adapter(), native::WGPUFeatureName_DepthClipControl)
            != 0
        {
            let feature_test =
                ValidationTest::with_features(&[native::WGPUFeatureName_DepthClipControl]);
            for is_async in [false, true] {
                expect_render_pipeline(
                    &feature_test,
                    is_async,
                    true,
                    RenderPipelineCase {
                        primitive: Some(primitive),
                        ..Default::default()
                    },
                );
            }
        }
    }
}
