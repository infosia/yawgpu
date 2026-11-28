//! Real-Vulkan e2e for `chromium-experimental-subgroup-matrix` (Block 170); the scenarios
//! live in `common/subgroup_matrix.rs`.
//!
//! The feature needs `VK_KHR_cooperative_matrix`, which MoltenVK does not expose, so
//! these run on native Windows and Linux drivers. A driver without it skips. Set
//! `YAWGPU_REQUIRE_SUBGROUP_MATRIX=1` on a host known to support it, so that a lost
//! advertisement fails instead of skipping. The advertisement test prints every
//! configuration the adapter reports (`--nocapture`).

#![cfg(feature = "vulkan")]

#[path = "common/subgroup_matrix.rs"]
mod subgroup_matrix;

use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_adapter_advertises_subgroup_matrix() {
    subgroup_matrix::adapter_advertises_subgroup_matrix(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
        std::env::var_os("YAWGPU_REQUIRE_SUBGROUP_MATRIX").is_some(),
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_matrix_multiply_matches_reference() {
    subgroup_matrix::matrix_multiply_matches_reference(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_matrix_scalar_arithmetic_matches_reference() {
    subgroup_matrix::matrix_scalar_arithmetic_matches_reference(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_matrix_store_with_offset() {
    subgroup_matrix::matrix_store_with_offset(RealBackend::Vulkan, YAWGPU_INSTANCE_BACKEND_VULKAN);
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_out_of_bounds_access_leaves_unbound_data_intact() {
    subgroup_matrix::out_of_bounds_access_leaves_unbound_data_intact(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_matrix_constructor_fills_every_element() {
    subgroup_matrix::matrix_constructor_fills_every_element(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_subgroup_matrix_validation_errors() {
    subgroup_matrix::subgroup_matrix_validation_errors(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}
