//! Real-Metal e2e for `destroy()` releasing GPU memory (Block 143); the scenarios live in
//! `common/destroy_memory.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/destroy_memory.rs"]
mod destroy_memory;

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

/// Bytes allocated on the system default Metal device, the device yawgpu picks on a
/// single-GPU Mac.
fn metal_allocated_size() -> u64 {
    let device = MTLCreateSystemDefaultDevice().expect("system default Metal device");
    device.currentAllocatedSize() as u64
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_destroy_releases_memory() {
    destroy_memory::destroy_releases_memory(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        Some(metal_allocated_size),
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_destroy_while_in_flight() {
    destroy_memory::destroy_while_in_flight(RealBackend::Metal, YAWGPU_INSTANCE_BACKEND_METAL);
}
