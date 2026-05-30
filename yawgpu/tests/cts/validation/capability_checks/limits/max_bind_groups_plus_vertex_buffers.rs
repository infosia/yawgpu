//! CTS port of `webgpu/api/validation/capability_checks/limits/maxBindGroupsPlusVertexBuffers.spec.ts`.

use crate::common;

#[test]
fn create_render_pipeline_at_over() {
    unsafe { common::assert_max_vertex_buffers_create_render_pipeline_at_over() };
}

#[test]
#[ignore = "draw-time maxBindGroupsPlusVertexBuffers relationship validation is not implemented as an active command check"]
fn draw_at_over() {
    unsafe { common::assert_max_vertex_buffers_create_render_pipeline_at_over() };
}
