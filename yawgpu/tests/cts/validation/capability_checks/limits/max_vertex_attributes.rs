//! CTS port of `webgpu/api/validation/capability_checks/limits/maxVertexAttributes.spec.ts`.

use crate::common;

#[test]
fn create_render_pipeline_at_over() {
    unsafe { common::assert_max_vertex_attributes_create_render_pipeline_at_over() };
}
