//! Real-Metal e2e for Block 148 R2 (`specs/blocks/148-shader-passthrough-review.md`);
//! the scenario lives in `common/compute_binding_slots.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/compute_binding_slots.rs"]
mod compute_binding_slots;
#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn metal_compute_bindings_with_render_visibility_reach_their_buffers() {
    compute_binding_slots::compute_bindings_with_render_visibility_reach_their_buffers(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_render_ignores_compute_only_bindings() {
    compute_binding_slots::render_ignores_compute_only_bindings(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}
