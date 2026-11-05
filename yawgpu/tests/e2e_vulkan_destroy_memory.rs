//! Real-Vulkan e2e for `destroy()` releasing GPU memory (Block 143); the scenarios live in
//! `common/destroy_memory.rs`. Run under the Khronos validation layer: destroying a
//! resource a submission still uses must not produce validation output.

#![cfg(feature = "vulkan")]

#[path = "common/destroy_memory.rs"]
mod destroy_memory;

use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_destroy_releases_memory() {
    // No portable allocated-size query; this runs the loop for the validation layer.
    destroy_memory::destroy_releases_memory(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
        None,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_destroy_while_in_flight() {
    destroy_memory::destroy_while_in_flight(RealBackend::Vulkan, YAWGPU_INSTANCE_BACKEND_VULKAN);
}
