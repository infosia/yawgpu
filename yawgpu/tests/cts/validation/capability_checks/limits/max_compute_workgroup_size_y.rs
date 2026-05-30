//! CTS port of `webgpu/api/validation/capability_checks/limits/maxComputeWorkgroupSizeY.spec.ts`.

use crate::common;

#[test]
fn create_compute_pipeline_at_over() {
    unsafe { common::assert_max_compute_workgroup_size_y_create_compute_pipeline_at_over() };
}

#[test]
fn validate_max_compute_invocations_per_workgroup() {
    unsafe { common::assert_required_limits_are_not_lowered_to_requested_values() };
}
