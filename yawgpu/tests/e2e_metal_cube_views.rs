//! Real-Metal e2e for Block 155 (`specs/blocks/155-vulkan-cube-compatible-images.md`);
//! the scenario lives in `common/cube_views.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/cube_views.rs"]
mod cube_views;
#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn metal_cube_view_samples_each_face() {
    cube_views::cube_view_samples_each_face(RealBackend::Metal, YAWGPU_INSTANCE_BACKEND_METAL);
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_cube_array_view_samples_each_face() {
    cube_views::cube_array_view_samples_each_face(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}
