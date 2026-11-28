//! Real-Metal e2e for `chromium-experimental-subgroup-matrix` (Block 170); the scenarios
//! live in `common/subgroup_matrix.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/subgroup_matrix.rs"]
mod subgroup_matrix;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

/// Apple Silicon Macs have an Apple7-or-later GPU, so the feature must be advertised there.
fn required() -> bool {
    cfg!(target_arch = "aarch64")
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_adapter_advertises_subgroup_matrix() {
    subgroup_matrix::adapter_advertises_subgroup_matrix(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        required(),
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_matrix_multiply_matches_reference() {
    subgroup_matrix::matrix_multiply_matches_reference(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        required(),
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_matrix_scalar_arithmetic_matches_reference() {
    subgroup_matrix::matrix_scalar_arithmetic_matches_reference(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        required(),
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_matrix_store_with_offset() {
    subgroup_matrix::matrix_store_with_offset(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        required(),
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_out_of_bounds_access_leaves_unbound_data_intact() {
    subgroup_matrix::out_of_bounds_access_leaves_unbound_data_intact(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        required(),
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_matrix_constructor_fills_every_element() {
    subgroup_matrix::matrix_constructor_fills_every_element(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        required(),
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_subgroup_matrix_validation_errors() {
    subgroup_matrix::subgroup_matrix_validation_errors(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        required(),
    );
}
