//! Real-Vulkan e2e for `subgroup-size-control` (Blocks 108 and 140); the scenarios live in
//! `common/subgroup_size_control.rs`.

#![cfg(feature = "vulkan")]

#[path = "common/subgroup_size_control.rs"]
mod subgroup_size_control;

use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_adapter_advertises_subgroup_size_control() {
    subgroup_size_control::adapter_advertises_subgroup_size_control(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
        false,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_subgroup_size_attribute_pins_runtime_width() {
    subgroup_size_control::subgroup_size_attribute_pins_runtime_width(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_subgroup_size_validation_errors() {
    subgroup_size_control::subgroup_size_validation_errors(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}
