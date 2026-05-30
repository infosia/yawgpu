//! CTS port of `webgpu/api/validation/capability_checks/limits/maxComputeInvocationsPerWorkgroup.spec.ts`.

use crate::common;

#[test]
fn create_compute_pipeline_at_over() {
    unsafe { common::assert_max_compute_invocations_create_compute_pipeline_at_over() };
}
