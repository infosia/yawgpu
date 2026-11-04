//! Real-Metal e2e for `atomic-vec2u-min-max` and `texture-compression-unaligned`
//! (Block 141); the scenarios live in `common/dawn_features.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/dawn_features.rs"]
mod dawn_features;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn metal_adapter_advertises_dawn_features() {
    // Every Metal adapter has texture-compression-unaligned. Apple Silicon Macs from the
    // M2 on (Mac2 with Apple8, or Apple9) have atomic-vec2u-min-max; the M1 does not.
    dawn_features::adapter_advertises_features(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        true,
        false,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_atomic_vec2u_min_max_readback() {
    dawn_features::atomic_vec2u_min_max_readback(RealBackend::Metal, YAWGPU_INSTANCE_BACKEND_METAL);
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_unaligned_bc_texture_round_trip() {
    dawn_features::unaligned_bc_texture_round_trip(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_unaligned_bc_texture_lazy_zero_init() {
    dawn_features::unaligned_bc_texture_lazy_zero_init(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}
