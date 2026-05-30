//! CTS port of `webgpu/api/validation/capability_checks/limits/maxInterStageShaderVariables.spec.ts`.

use crate::common;

#[test]
#[ignore = "inter-stage shader variable counting is not yet exposed by a dedicated active CTS helper"]
fn create_render_pipeline_at_over() {
    unsafe { common::assert_max_vertex_attributes_create_render_pipeline_at_over() };
}
