//! CTS port of `webgpu/api/validation/capability_checks/limits/maxStorageBuffersInVertexStage.spec.ts`.

use yawgpu::native;

use crate::common;

#[test]
fn create_bind_group_layout_at_over() {
    unsafe { common::assert_max_storage_buffers_bgl_at_over(native::WGPUShaderStage_Vertex) };
}

#[test]
fn create_pipeline_layout_at_over() {
    unsafe {
        common::assert_max_storage_buffers_create_pipeline_layout_at_over(
            native::WGPUShaderStage_Vertex,
        )
    };
}

#[test]
#[ignore = "createPipeline vertex storage buffer shader resource matrix is not yet active"]
fn create_pipeline_at_over() {
    unsafe { common::assert_max_storage_buffers_bgl_at_over(native::WGPUShaderStage_Vertex) };
}
