//! CTS port of `webgpu/api/validation/capability_checks/limits/maxStorageTexturesPerShaderStage.spec.ts`.

use yawgpu::native;

use crate::common;

#[test]
fn create_bind_group_layout_at_over() {
    unsafe { common::assert_max_storage_textures_bgl_at_over(native::WGPUShaderStage_Compute) };
}

#[test]
#[ignore = "pipeline layout resource aggregation for storage textures is not yet active as a separate CTS creator"]
fn create_pipeline_layout_at_over() {
    unsafe { common::assert_max_storage_textures_bgl_at_over(native::WGPUShaderStage_Compute) };
}

#[test]
#[ignore = "createPipeline storage texture shader resource matrix is not yet active"]
fn create_pipeline_at_over() {
    unsafe { common::assert_max_storage_textures_bgl_at_over(native::WGPUShaderStage_Compute) };
}
