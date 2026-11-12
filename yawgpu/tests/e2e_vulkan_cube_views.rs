//! Real-Vulkan e2e for Block 155 (`specs/blocks/155-vulkan-cube-compatible-images.md`);
//! the scenario lives in `common/cube_views.rs`.

#![cfg(feature = "vulkan")]

#[path = "common/cube_views.rs"]
mod cube_views;
#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;

use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_cube_view_samples_each_face() {
    cube_views::cube_view_samples_each_face(RealBackend::Vulkan, YAWGPU_INSTANCE_BACKEND_VULKAN);
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_cube_array_view_samples_each_face() {
    cube_views::cube_array_view_samples_each_face(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}
