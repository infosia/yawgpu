//! Real-Metal e2e for `subgroup-size-control` (Blocks 108 and 140); the scenarios live in
//! `common/subgroup_size_control.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/subgroup_size_control.rs"]
mod subgroup_size_control;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn metal_adapter_advertises_subgroup_size_control() {
    // Apple Silicon Macs have an Apple GPU, whose subgroup size is fixed at 32.
    subgroup_size_control::adapter_advertises_subgroup_size_control(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        cfg!(target_arch = "aarch64"),
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_subgroup_size_attribute_pins_runtime_width() {
    subgroup_size_control::subgroup_size_attribute_pins_runtime_width(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_subgroup_size_validation_errors() {
    subgroup_size_control::subgroup_size_validation_errors(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}
