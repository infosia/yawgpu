use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::{common, feature_common};

#[test]
#[ignore = "Noop does not advertise clip-distances"]
fn create_render_pipeline_at_over() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_ClipDistances);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_ClipDistances);
    unsafe {
        let module = common::create_wgsl_module(test.device(), CLIP_SHADER);
        let fragment = common::create_wgsl_module(test.device(), FRAGMENT_SHADER);
        let pipeline = common::create_render_pipeline(test.device(), module, fragment, "fs");
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[test]
#[ignore = "Noop does not advertise clip-distances"]
fn create_render_pipeline_max_vertex_output_location() {
    feature_common::assert_noop_lacks_feature(native::WGPUFeatureName_ClipDistances);
    let test = feature_common::test_with_feature(native::WGPUFeatureName_ClipDistances);
    unsafe {
        let module = common::create_wgsl_module(test.device(), CLIP_SHADER_WITH_LOCATION_OUTPUT);
        let fragment = common::create_wgsl_module(test.device(), FRAGMENT_SHADER);
        let pipeline = common::create_render_pipeline(test.device(), module, fragment, "fs");
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

#[allow(dead_code)]
fn _default_test_keeps_import_live(_: &ValidationTest) {}

const CLIP_SHADER: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4f,
    @builtin(clip_distances) clip: array<f32, 1>,
}

@vertex
fn main() -> VertexOut {
    var out: VertexOut;
    out.position = vec4f(0.0, 0.0, 0.0, 1.0);
    out.clip[0] = 1.0;
    return out;
}
"#;

const CLIP_SHADER_WITH_LOCATION_OUTPUT: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4f,
    @builtin(clip_distances) clip: array<f32, 1>,
    @location(15) value: vec4f,
}

@vertex
fn main() -> VertexOut {
    var out: VertexOut;
    out.position = vec4f(0.0, 0.0, 0.0, 1.0);
    out.clip[0] = 1.0;
    out.value = vec4f(0.0);
    return out;
}
"#;

const FRAGMENT_SHADER: &str = r#"
@fragment
fn fs() -> @location(0) vec4f {
    return vec4f(0.0);
}
"#;
