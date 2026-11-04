//! Real-Vulkan e2e for `atomic-vec2u-min-max` and `texture-compression-unaligned`
//! (Block 141); the scenarios live in `common/dawn_features.rs`.

#![cfg(feature = "vulkan")]

#[path = "common/dawn_features.rs"]
mod dawn_features;

use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_adapter_advertises_dawn_features() {
    // Every Vulkan adapter has texture-compression-unaligned; atomic-vec2u-min-max needs
    // 64-bit buffer atomics.
    dawn_features::adapter_advertises_features(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
        true,
        false,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_atomic_vec2u_min_max_readback() {
    dawn_features::atomic_vec2u_min_max_readback(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_unaligned_bc_texture_round_trip() {
    dawn_features::unaligned_bc_texture_round_trip(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_unaligned_bc_texture_lazy_zero_init() {
    dawn_features::unaligned_bc_texture_lazy_zero_init(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}
