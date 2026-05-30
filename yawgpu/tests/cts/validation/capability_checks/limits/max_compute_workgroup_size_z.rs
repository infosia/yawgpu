//! CTS port of `webgpu/api/validation/capability_checks/limits/maxComputeWorkgroupSizeZ.spec.ts`.

use crate::common;

#[test]
fn create_compute_pipeline_at_over() {
    unsafe { common::assert_max_compute_workgroup_size_z_create_compute_pipeline_at_over() };
}

#[test]
#[ignore = "maxComputeInvocationsPerWorkgroup relationship validation at device request is not yet implemented"]
fn validate_max_compute_invocations_per_workgroup() {
    unsafe { common::assert_required_limits_are_not_lowered_to_requested_values() };
}
