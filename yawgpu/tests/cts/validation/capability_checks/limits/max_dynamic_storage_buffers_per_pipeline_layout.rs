//! CTS port of `webgpu/api/validation/capability_checks/limits/maxDynamicStorageBuffersPerPipelineLayout.spec.ts`.

use crate::common;

#[test]
fn create_bind_group_layout_at_over() {
    unsafe { common::assert_max_dynamic_storage_bgl_at_over() };
}

#[test]
#[ignore = "pipeline layout aggregation for dynamic storage buffers is covered by BGL limit but not active as a separate CTS creator"]
fn create_pipeline_layout_at_over() {
    unsafe { common::assert_max_dynamic_storage_bgl_at_over() };
}
