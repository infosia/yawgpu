//! CTS port of `webgpu/api/validation/capability_checks/limits/maxVertexBufferArrayStride.spec.ts`.

use crate::common;

#[test]
fn create_render_pipeline_at_over() {
    unsafe { common::assert_max_vertex_buffer_array_stride_create_render_pipeline_at_over() };
}

#[test]
#[ignore = "required-limit validation for maxVertexBufferArrayStride is not yet active"]
fn validate() {
    unsafe { common::assert_required_limits_are_not_lowered_to_requested_values() };
}
