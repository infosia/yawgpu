//! Real-Metal e2e for Block 132 (`specs/blocks/132-render-bind-groups-at-draw.md`);
//! the scenario lives in `common/render_bind_groups_at_draw.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;
#[path = "common/render_bind_groups_at_draw.rs"]
mod render_bind_groups_at_draw;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn group_set_under_another_pipeline_draws_with_the_next() {
    render_bind_groups_at_draw::group_set_under_another_pipeline_draws_with_the_next(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}
