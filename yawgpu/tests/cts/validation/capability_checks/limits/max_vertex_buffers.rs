//! CTS port of `webgpu/api/validation/capability_checks/limits/maxVertexBuffers.spec.ts`.

use crate::common;

#[test]
fn create_render_pipeline_at_over() {
    unsafe { common::assert_max_vertex_buffers_create_render_pipeline_at_over() };
}

#[test]
#[ignore = "setVertexBuffer maxVertexBuffers command matrix is not yet active"]
fn set_vertex_buffer_at_over() {
    unsafe { common::assert_max_vertex_buffers_create_render_pipeline_at_over() };
}

#[test]
#[ignore = "maxBindGroupsPlusVertexBuffers relationship validation is not yet active"]
fn validate_max_bind_groups_plus_vertex_buffers() {
    unsafe { common::assert_required_limits_are_not_lowered_to_requested_values() };
}
