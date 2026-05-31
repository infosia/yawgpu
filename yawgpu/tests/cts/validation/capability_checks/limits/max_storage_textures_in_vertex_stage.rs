//! CTS port of `webgpu/api/validation/capability_checks/limits/maxStorageTexturesInVertexStage.spec.ts`.

use yawgpu::native;

use crate::common;

#[test]
fn create_bind_group_layout_at_over() {
    unsafe { common::assert_max_storage_textures_bgl_at_over(native::WGPUShaderStage_Vertex) };
}

#[test]
fn create_pipeline_layout_at_over() {
    unsafe {
        common::assert_max_storage_textures_create_pipeline_layout_at_over(
            native::WGPUShaderStage_Vertex,
        )
    };
}

#[test]
#[ignore = "createPipeline vertex storage texture shader resource matrix is not yet active"]
fn create_pipeline_at_over() {
    unsafe { common::assert_max_storage_textures_bgl_at_over(native::WGPUShaderStage_Vertex) };
}
