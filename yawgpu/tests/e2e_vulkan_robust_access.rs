//! Real-Vulkan e2e for Block 128; the scenarios live in `common/robust_access.rs`.

#![cfg(feature = "vulkan")]

#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;
#[path = "common/robust_access.rs"]
mod robust_access;

use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn storage_runtime_array_out_of_bounds() {
    robust_access::storage_runtime_array_out_of_bounds(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn storage_texture_out_of_bounds() {
    robust_access::storage_texture_out_of_bounds(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn texture_load_out_of_bounds() {
    robust_access::texture_load_out_of_bounds(RealBackend::Vulkan, YAWGPU_INSTANCE_BACKEND_VULKAN);
}
