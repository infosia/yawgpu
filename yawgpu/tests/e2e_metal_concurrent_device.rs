//! Real-Metal run of the shared-device concurrency workload (Block 171 R3); the body
//! lives in `common/concurrent_device.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/concurrent_device.rs"]
mod concurrent_device;

#[test]
#[ignore = "manual real-backend test"]
fn metal_shared_device() {
    if yawgpu_test::real_backend_skip_reason(yawgpu_test::RealBackend::Metal).is_some() {
        return;
    }
    concurrent_device::run(yawgpu::YAWGPU_INSTANCE_BACKEND_METAL);
}
