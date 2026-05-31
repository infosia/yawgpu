//! CTS port of `webgpu/api/validation/capability_checks/limits/maxDynamicUniformBuffersPerPipelineLayout.spec.ts`.

use crate::common;

#[test]
fn create_bind_group_layout_at_over() {
    unsafe { common::assert_max_dynamic_uniform_bgl_at_over() };
}

#[test]
fn create_pipeline_layout_at_over() {
    unsafe { common::assert_max_dynamic_uniform_create_pipeline_layout_at_over() };
}
