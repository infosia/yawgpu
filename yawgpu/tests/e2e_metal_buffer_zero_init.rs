//! Real-Metal e2e for Block 129 (`specs/blocks/129-vulkan-buffer-zero-init.md`);
//! the scenarios live in `common/buffer_zero_init.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/buffer_zero_init.rs"]
mod buffer_zero_init;
#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn fresh_buffers_map_as_zero() {
    buffer_zero_init::fresh_buffers_map_as_zero(RealBackend::Metal, YAWGPU_INSTANCE_BACKEND_METAL);
}

#[test]
#[ignore = "manual real-backend test"]
fn fresh_buffers_mapped_at_creation_are_zero() {
    buffer_zero_init::fresh_buffers_mapped_at_creation_are_zero(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}
