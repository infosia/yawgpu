//! Real-Vulkan e2e for Block 121 (`specs/blocks/121-work-done-pending-writes.md`);
//! the scenarios live in `common/work_done_writes.rs`.

#![cfg(feature = "vulkan")]

#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;
#[path = "common/work_done_writes.rs"]
mod work_done_writes;

use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn write_then_work_done_then_copy() {
    work_done_writes::write_then_work_done_then_copy(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn write_then_work_done_then_map() {
    work_done_writes::write_then_work_done_then_map(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn writes_across_the_threshold_keep_queue_order() {
    work_done_writes::writes_across_the_threshold_keep_queue_order(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}
