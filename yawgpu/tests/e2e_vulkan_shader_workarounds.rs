//! Real-Vulkan e2e for Block 120 (`specs/blocks/120-tint-driver-workarounds.md`);
//! the scenarios live in `common/shader_workarounds.rs`.

#![cfg(feature = "vulkan")]

#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;
#[path = "common/shader_workarounds.rs"]
mod shader_workarounds;

use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::RealBackend;

const FORCED: &str = "polyfill_bool_vec_dynamic_store,replace_workgroup_atomic_store_with_exchange,replace_unsigned_compare_zero";

#[test]
#[ignore = "manual real-backend test"]
fn compute_results_match_with_workarounds_forced() {
    shader_workarounds::compute(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
        FORCED,
        "compute_results_match_with_workarounds_forced",
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn render_results_match_with_workarounds_forced() {
    shader_workarounds::render(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
        FORCED,
        "render_results_match_with_workarounds_forced",
    );
}
