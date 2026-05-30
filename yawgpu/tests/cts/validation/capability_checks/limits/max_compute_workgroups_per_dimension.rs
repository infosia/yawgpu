//! CTS port of `webgpu/api/validation/capability_checks/limits/maxComputeWorkgroupsPerDimension.spec.ts`.

use crate::common;

#[test]
fn dispatch_workgroups_at_over() {
    unsafe { common::assert_max_compute_workgroups_dispatch_at_over() };
}

#[test]
#[ignore = "required-limit relationship validation for maxComputeWorkgroupsPerDimension is not yet implemented"]
fn validate() {
    unsafe { common::assert_required_limits_are_not_lowered_to_requested_values() };
}
