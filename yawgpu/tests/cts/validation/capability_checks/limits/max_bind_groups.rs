//! CTS port of `webgpu/api/validation/capability_checks/limits/maxBindGroups.spec.ts`.

use crate::common;

#[test]
fn create_pipeline_layout_at_over() {
    unsafe { common::assert_max_bind_groups_create_pipeline_layout_at_over() };
}

#[test]
#[ignore = "createPipeline maxBindGroups shader/layout matrix needs a dedicated pipeline creator; createPipelineLayout coverage is active"]
fn create_pipeline_at_over() {
    unsafe { common::assert_max_bind_groups_create_pipeline_layout_at_over() };
}

#[test]
#[ignore = "setBindGroup maxBindGroups pass-encoder matrix needs active command encoding coverage"]
fn set_bind_group_at_over() {
    unsafe { common::assert_max_bind_groups_create_pipeline_layout_at_over() };
}

#[test]
#[ignore = "maxBindGroupsPlusVertexBuffers relationship validation is not yet implemented in device creation"]
fn validate_max_bind_groups_plus_vertex_buffers() {
    unsafe { common::assert_required_limits_are_not_lowered_to_requested_values() };
}
