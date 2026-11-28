//! Real-Metal e2e for `chromium-experimental-subgroup-matrix` (Block 170); the scenarios
//! live in `common/subgroup_matrix.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/subgroup_matrix.rs"]
mod subgroup_matrix;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn metal_adapter_advertises_subgroup_matrix() {
    // Apple Silicon Macs have an Apple7-or-later GPU.
    subgroup_matrix::adapter_advertises_subgroup_matrix(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        cfg!(target_arch = "aarch64"),
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_matrix_multiply_matches_reference() {
    subgroup_matrix::matrix_multiply_matches_reference(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_matrix_scalar_arithmetic_matches_reference() {
    subgroup_matrix::matrix_scalar_arithmetic_matches_reference(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_subgroup_matrix_validation_errors() {
    subgroup_matrix::subgroup_matrix_validation_errors(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}
