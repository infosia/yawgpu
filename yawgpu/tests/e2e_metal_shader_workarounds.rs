//! Real-Metal e2e for Block 120 (`specs/blocks/120-tint-driver-workarounds.md`);
//! the scenarios live in `common/shader_workarounds.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;
#[path = "common/shader_workarounds.rs"]
mod shader_workarounds;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

const FORCED: &str = "fix_u32_div_mod,polyfill_bool_vec_dynamic_store";

#[test]
#[ignore = "manual real-backend test"]
fn compute_results_match_with_workarounds_forced() {
    shader_workarounds::compute(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        FORCED,
        "compute_results_match_with_workarounds_forced",
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn render_results_match_with_workarounds_forced() {
    shader_workarounds::render(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
        FORCED,
        "render_results_match_with_workarounds_forced",
    );
}
