//! CTS port of `webgpu/api/validation/capability_checks/limits/maxComputeWorkgroupStorageSize.spec.ts`.

use crate::common;

#[test]
fn create_compute_pipeline_at_over() {
    unsafe { common::assert_max_compute_workgroup_storage_create_compute_pipeline_at_over() };
}
