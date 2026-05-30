//! CTS port of `webgpu/api/validation/capability_checks/limits/maxBindingsPerBindGroup.spec.ts`.

use crate::common;

#[test]
fn create_bind_group_layout_at_over() {
    unsafe { common::assert_max_bindings_per_bind_group_create_bind_group_layout_at_over() };
}

#[test]
#[ignore = "createPipeline maxBindingsPerBindGroup shader resource matrix is not yet active"]
fn create_pipeline_at_over() {
    unsafe { common::assert_max_bindings_per_bind_group_create_bind_group_layout_at_over() };
}

#[test]
#[ignore = "maxBindingsPerBindGroup required-limit relationship validation is not yet implemented"]
fn validate() {
    unsafe { common::assert_required_limits_are_not_lowered_to_requested_values() };
}
