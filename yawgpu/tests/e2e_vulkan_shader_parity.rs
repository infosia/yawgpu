//! Real-Vulkan e2e for Blocks 125 and 126; the scenarios live in
//! `common/shader_parity.rs`.

#![cfg(feature = "vulkan")]

#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;
#[path = "common/shader_parity.rs"]
mod shader_parity;

use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn packed_dot_products() {
    shader_parity::packed_dot_products(RealBackend::Vulkan, YAWGPU_INSTANCE_BACKEND_VULKAN);
}

#[test]
#[ignore = "manual real-backend test"]
fn frag_depth_is_clamped_to_viewport() {
    shader_parity::frag_depth_is_clamped_to_viewport(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}
