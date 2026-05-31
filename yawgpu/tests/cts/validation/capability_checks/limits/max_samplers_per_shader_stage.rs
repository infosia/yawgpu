//! CTS port of `webgpu/api/validation/capability_checks/limits/maxSamplersPerShaderStage.spec.ts`.

use crate::common;

#[test]
fn create_bind_group_layout_at_over() {
    unsafe { common::assert_max_samplers_bgl_at_over() };
}

#[test]
fn create_pipeline_layout_at_over() {
    unsafe { common::assert_max_samplers_create_pipeline_layout_at_over() };
}

#[test]
#[ignore = "createPipeline sampler shader resource matrix is not yet active"]
fn create_pipeline_at_over() {
    unsafe { common::assert_max_samplers_bgl_at_over() };
}
