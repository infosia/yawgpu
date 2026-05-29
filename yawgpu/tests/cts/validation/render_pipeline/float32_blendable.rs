//! CTS: src/webgpu/api/validation/render_pipeline/float32_blendable.spec.ts

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::render_common::{
    blend_state, color_target_with, default_blend_component, expect_render_pipeline,
    RenderPipelineCase,
};

#[test]
fn create_render_pipeline() {
    let test = ValidationTest::new();
    unsafe {
        let blend = blend_state(default_blend_component(), default_blend_component());
        let cases = [
            (native::WGPUTextureFormat_R32Float, false, true),
            (native::WGPUTextureFormat_RG32Float, false, true),
            (native::WGPUTextureFormat_RGBA32Float, false, true),
            (native::WGPUTextureFormat_R32Float, true, false),
            (native::WGPUTextureFormat_RG32Float, true, false),
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
