//! Real-Vulkan e2e for Block 129 (`specs/blocks/129-vulkan-buffer-zero-init.md`);
//! the scenarios live in `common/buffer_zero_init.rs`.

#![cfg(feature = "vulkan")]

#[path = "common/buffer_zero_init.rs"]
mod buffer_zero_init;
#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;

use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn fresh_buffers_map_as_zero() {
    buffer_zero_init::fresh_buffers_map_as_zero(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn fresh_buffers_mapped_at_creation_are_zero() {
    buffer_zero_init::fresh_buffers_mapped_at_creation_are_zero(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}
