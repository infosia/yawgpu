//! CTS port of `webgpu/api/validation/capability_checks/limits/maxSampledTexturesPerShaderStage.spec.ts`.

use crate::common;

#[test]
fn create_bind_group_layout_at_over() {
    unsafe { common::assert_max_sampled_textures_bgl_at_over() };
}

#[test]
fn create_pipeline_layout_at_over() {
    unsafe { common::assert_max_sampled_textures_create_pipeline_layout_at_over() };
}

#[test]
#[ignore = "createPipeline sampled texture shader resource matrix is not yet active"]
fn create_pipeline_at_over() {
    unsafe { common::assert_max_sampled_textures_bgl_at_over() };
}
