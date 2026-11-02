//! Real-Metal e2e for Block 128; the scenarios live in `common/robust_access.rs`.
//! Metal keeps Tint's full robustness; these runs pin the shared expectations.

#![cfg(feature = "metal")]

#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;
#[path = "common/robust_access.rs"]
mod robust_access;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn storage_runtime_array_out_of_bounds() {
    robust_access::storage_runtime_array_out_of_bounds(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn storage_texture_out_of_bounds() {
    robust_access::storage_texture_out_of_bounds(RealBackend::Metal, YAWGPU_INSTANCE_BACKEND_METAL);
}

#[test]
#[ignore = "manual real-backend test"]
fn texture_load_out_of_bounds() {
    robust_access::texture_load_out_of_bounds(RealBackend::Metal, YAWGPU_INSTANCE_BACKEND_METAL);
}
