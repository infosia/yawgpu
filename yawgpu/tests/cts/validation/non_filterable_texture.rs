//! CTS: src/webgpu/api/validation/non_filterable_texture.spec.ts

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::bind_group_common::{
    expect_bind_group_layout, expect_pipeline_layout, release_bind_group_layouts,
    release_pipeline_layout, sampler_layout_typed, texture_layout,
};
use crate::common::{
    assert_compute_pipeline_error, assert_compute_pipeline_ok, assert_render_pipeline_error,
    assert_render_pipeline_ok, FragmentInput,
};

#[test]
fn non_filterable_texture_with_filtering_sampler() {
    let test = ValidationTest::new();
    unsafe {
        for (pipeline, is_async, sample_type, same_group, success) in [
            (
                "compute",
                false,
                native::WGPUTextureSampleType_Float,
                true,
                true,
            ),
            (
                "compute",
                true,
                native::WGPUTextureSampleType_Uint,
                true,
                false,
            ),
            (
                "render",
                false,
                native::WGPUTextureSampleType_Sint,
                false,
                false,
            ),
            (
                "render",
                true,
                native::WGPUTextureSampleType_UnfilterableFloat,
                false,
                false,
            ),
            (
                "compute",
                false,
                native::WGPUTextureSampleType_Depth,
                true,
                false,
            ),
        ] {
            let texture_entry = texture_layout(
                0,
                native::WGPUShaderStage_Compute | native::WGPUShaderStage_Fragment,
                sample_type,
                native::WGPUTextureViewDimension_2D,
                false,
            );
            let sampler_entry = sampler_layout_typed(
                1,
                native::WGPUShaderStage_Compute | native::WGPUShaderStage_Fragment,
                native::WGPUSamplerBindingType_Filtering,
            );
            let (group0_entries, group1) = if same_group {
                (vec![texture_entry, sampler_entry], None)
            } else {
                (vec![texture_entry], Some(sampler_entry))
            };
            let group0 = expect_bind_group_layout(&test, true, &group0_entries);
            let group1 = group1.map(|entry| expect_bind_group_layout(&test, true, &[entry]));
            let layouts = group1.map_or_else(|| vec![group0], |layout| vec![group0, layout]);
            let pipeline_layout = expect_pipeline_layout(&test, true, &layouts, 0);
            let source = texture_gather_source(sample_type, same_group);

            match (pipeline, success) {
                ("compute", true) => assert_compute_pipeline_ok(
                    &test,
                    is_async,
                    &source,
                    Some("cs"),
                    &[],
                    Some(pipeline_layout),
                ),
                ("compute", false) => assert_compute_pipeline_error(
                    &test,
                    is_async,
                    &source,
                    Some("cs"),
                    &[],
                    Some(pipeline_layout),
                ),
                ("render", true) => assert_render_pipeline_ok(
                    &test,
                    is_async,
                    &source,
                    Some("vs"),
                    Some(FragmentInput::new(&source, Some("fs"), 1)),
                    Some(pipeline_layout),
                    None,
                ),
                ("render", false) => assert_render_pipeline_error(
                    &test,
                    is_async,
                    &source,
                    Some("vs"),
                    Some(FragmentInput::new(&source, Some("fs"), 1)),
                    Some(pipeline_layout),
                    None,
                ),
                _ => unreachable!("unknown pipeline case"),
            }

            release_pipeline_layout(pipeline_layout);
            release_bind_group_layouts(&layouts);
        }
    }
}

fn texture_gather_source(sample_type: native::WGPUTextureSampleType, same_group: bool) -> String {
    let (texture_type, component) = match sample_type {
        native::WGPUTextureSampleType_Sint => ("texture_2d<i32>", "0, "),
        native::WGPUTextureSampleType_Uint => ("texture_2d<u32>", "0, "),
        native::WGPUTextureSampleType_Depth => ("texture_depth_2d", ""),
        _ => ("texture_2d<f32>", "0, "),
    };
    let sampler_group = if same_group { 0 } else { 1 };
    format!(
        r#"
@group(0) @binding(0) var t: {texture_type};
@group({sampler_group}) @binding(1) var s: sampler;

fn test() {{
    _ = textureGather({component}t, s, vec2f(0));
}}

@compute @workgroup_size(1) fn cs() {{
    test();
}}

@vertex fn vs() -> @builtin(position) vec4f {{
    return vec4f(0);
}}

@fragment fn fs() -> @location(0) vec4f {{
    test();
    return vec4f(0);
}}
"#
    )
}
