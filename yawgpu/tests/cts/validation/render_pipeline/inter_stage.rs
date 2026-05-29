//! CTS port of `webgpu/api/validation/render_pipeline/inter_stage.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::device_limits;
use crate::render_common::{
    default_primitive, expect_render_pipeline, inter_stage_fragment, inter_stage_vertex,
    RenderPipelineCase,
};

#[test]
#[ignore = "core does not yet validate vertex-output/fragment-input location equality at createRenderPipeline; CTS expects missing matching locations to fail"]
fn location_mismatch() {
    let cases = [
        (
            vec!["@location(0) __: f32"],
            vec!["@location(0) __: f32"],
            true,
        ),
        (
            vec!["@location(0) __: f32"],
            vec!["@location(1) __: f32"],
            false,
        ),
        (
            vec!["@location(1) __: f32"],
            vec!["@location(0) __: f32"],
            false,
        ),
        (
            vec!["@location(0) __: f32", "@location(1) __: f32"],
            vec!["@location(1) __: f32", "@location(0) __: f32"],
            true,
        ),
    ];
    run_inter_stage_cases(&cases);
}

#[test]
fn location_superset() {
    let cases = [(
        vec!["@location(0) __: f32", "@location(1) __: f32"],
        vec!["@location(1) __: f32"],
        true,
    )];
    run_inter_stage_cases(&cases);
}

#[test]
#[ignore = "core does not yet reject fragment inputs not produced by the vertex stage at createRenderPipeline; CTS expects validation error"]
fn location_subset() {
    let cases = [(
        vec!["@location(0) __: f32"],
        vec!["@location(0) __: f32", "@location(1) __: f32"],
        false,
    )];
    run_inter_stage_cases(&cases);
}

#[test]
#[ignore = "core does not yet validate inter-stage type compatibility at createRenderPipeline; CTS expects mismatched types to fail"]
fn type_() {
    let cases = [
        (
            vec!["@location(0) __: f32"],
            vec!["@location(0) __: f32"],
            true,
        ),
        (
            vec!["@location(0) __: i32"],
            vec!["@location(0) __: f32"],
            false,
        ),
        (
            vec!["@location(0) __: vec2<f32>"],
            vec!["@location(0) __: vec2<f32>"],
            true,
        ),
        (
            vec!["@location(0) __: vec3<f32>"],
            vec!["@location(0) __: vec2<f32>"],
            false,
        ),
    ];
    run_inter_stage_cases(&cases);
}

#[test]
#[ignore = "core does not yet validate inter-stage interpolation type compatibility at createRenderPipeline; CTS expects mismatches to fail"]
fn interpolation_type() {
    let cases = [
        (
            vec!["@location(0) __: f32"],
            vec!["@location(0) __: f32"],
            true,
        ),
        (
            vec!["@location(0) @interpolate(linear) __: f32"],
            vec!["@location(0) @interpolate(linear) __: f32"],
            true,
        ),
        (
            vec!["@location(0) @interpolate(linear) __: f32"],
            vec!["@location(0) @interpolate(perspective) __: f32"],
            false,
        ),
        (
            vec!["@location(0) @interpolate(flat, either) __: i32"],
            vec!["@location(0) @interpolate(flat, either) __: i32"],
            true,
        ),
    ];
    run_inter_stage_cases(&cases);
}

#[test]
#[ignore = "core does not yet validate inter-stage interpolation sampling compatibility at createRenderPipeline; CTS expects mismatches to fail"]
fn interpolation_sampling() {
    let cases = [
        (
            vec!["@location(0) @interpolate(perspective, center) __: f32"],
            vec!["@location(0) @interpolate(perspective, center) __: f32"],
            true,
        ),
        (
            vec!["@location(0) @interpolate(perspective, center) __: f32"],
            vec!["@location(0) @interpolate(perspective, sample) __: f32"],
            false,
        ),
        (
            vec!["@location(0) @interpolate(perspective, centroid) __: f32"],
            vec!["@location(0) @interpolate(perspective) __: f32"],
            false,
        ),
    ];
    run_inter_stage_cases(&cases);
}

#[test]
#[ignore = "core does not yet enforce maxInterStageShaderVariables location limits at createRenderPipeline; CTS expects location >= limit to fail"]
fn max_shader_variable_location() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for (location, success) in [
            (limits.maxInterStageShaderVariables - 1, true),
            (limits.maxInterStageShaderVariables, false),
        ] {
            let output = format!("@location({location}) __: f32");
            let input = format!("@location({location}) __: f32");
            expect_inter_stage(&test, false, &[&output], &[&input], success, None);
            expect_inter_stage(&test, true, &[&output], &[&input], success, None);
        }
    }
}

#[test]
#[ignore = "core does not yet enforce maxInterStageShaderVariables output-count limits at createRenderPipeline; CTS expects over-limit output counts to fail"]
fn max_variables_count_output() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for (count, topology, success) in [
            (
                limits.maxInterStageShaderVariables,
                native::WGPUPrimitiveTopology_TriangleList,
                true,
            ),
            (
                limits.maxInterStageShaderVariables + 1,
                native::WGPUPrimitiveTopology_TriangleList,
                false,
            ),
            (
                limits.maxInterStageShaderVariables,
                native::WGPUPrimitiveTopology_PointList,
                false,
            ),
        ] {
            let outputs = location_fields(count);
            let inputs = location_fields(count);
            let refs_out = outputs.iter().map(String::as_str).collect::<Vec<_>>();
            let refs_in = inputs.iter().map(String::as_str).collect::<Vec<_>>();
            let mut primitive = default_primitive();
            primitive.topology = topology;
            for is_async in [false, true] {
                expect_inter_stage(
                    &test,
                    is_async,
                    &refs_out,
                    &refs_in,
                    success,
                    Some(primitive),
                );
            }
        }
    }
}

#[test]
#[ignore = "core does not yet enforce maxInterStageShaderVariables input-count limits at createRenderPipeline; CTS expects over-limit input counts to fail"]
fn max_variables_count_input() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for (count, success) in [
            (limits.maxInterStageShaderVariables, true),
            (limits.maxInterStageShaderVariables + 1, false),
        ] {
            let outputs = location_fields(count);
            let inputs = location_fields(count);
            let refs_out = outputs.iter().map(String::as_str).collect::<Vec<_>>();
            let refs_in = inputs.iter().map(String::as_str).collect::<Vec<_>>();
            for is_async in [false, true] {
                expect_inter_stage(&test, is_async, &refs_out, &refs_in, success, None);
            }
        }
    }
}

fn run_inter_stage_cases(cases: &[(Vec<&str>, Vec<&str>, bool)]) {
    let test = ValidationTest::new();
    unsafe {
        for (outputs, inputs, success) in cases {
            for is_async in [false, true] {
                expect_inter_stage(&test, is_async, outputs, inputs, *success, None);
            }
        }
    }
}

unsafe fn expect_inter_stage(
    test: &ValidationTest,
    is_async: bool,
    outputs: &[&str],
    inputs: &[&str],
    success: bool,
    primitive: Option<native::WGPUPrimitiveState>,
) {
    let vertex = inter_stage_vertex(outputs);
    let fragment = inter_stage_fragment(inputs);
    unsafe {
        expect_render_pipeline(
            test,
            is_async,
            success,
            RenderPipelineCase {
                vertex_source: &vertex,
                fragment_source: Some(&fragment),
                primitive,
                ..Default::default()
            },
        );
    }
}

fn location_fields(count: u32) -> Vec<String> {
    (0..count)
        .map(|i| format!("@location({i}) __: vec4<f32>"))
        .collect()
}
