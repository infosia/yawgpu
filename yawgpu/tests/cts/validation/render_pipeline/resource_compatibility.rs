//! CTS: src/webgpu/api/validation/render_pipeline/resource_compatibility.spec.ts

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{create_bind_group_layout, create_pipeline_layout, uniform_layout};
use crate::render_common::{expect_render_pipeline, RenderPipelineCase, FRAGMENT_COLOR};

#[test]
#[ignore = "core does not yet reject every CTS render pipeline layout/shader resource mismatch; CTS expects explicit layout resources to match shader resources by binding type, visibility, and binding number"]
fn resource_compatibility() {
    let test = ValidationTest::new();
    unsafe {
        let vertex_resource = "
@group(0) @binding(0) var<uniform> u: vec4f;
@vertex fn main() -> @builtin(position) vec4f {
    return u;
}";
        let cases = [
            (
                vec![uniform_layout(0, native::WGPUShaderStage_Vertex, 16)],
                true,
            ),
            (vec![], false),
            (
                vec![uniform_layout(0, native::WGPUShaderStage_Fragment, 16)],
                false,
            ),
            (
                vec![uniform_layout(1, native::WGPUShaderStage_Vertex, 16)],
                false,
            ),
        ];
        for (entries, success) in cases {
            let bgl = create_bind_group_layout(test.device(), &entries);
            let layout = create_pipeline_layout(test.device(), &[bgl], 0);
            for is_async in [false, true] {
                expect_render_pipeline(
                    &test,
                    is_async,
                    success,
                    RenderPipelineCase {
                        vertex_source: vertex_resource,
                        fragment_source: Some(FRAGMENT_COLOR),
                        layout: Some(layout),
                        ..Default::default()
                    },
                );
            }
            yawgpu::wgpuPipelineLayoutRelease(layout);
            yawgpu::wgpuBindGroupLayoutRelease(bgl);
        }
    }
}
