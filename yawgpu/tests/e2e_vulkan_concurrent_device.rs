//! Real-Vulkan run of the shared-device concurrency workload (Block 171 R3); the body
//! lives in `common/concurrent_device.rs`.

#![cfg(feature = "vulkan")]

#[path = "common/concurrent_device.rs"]
mod concurrent_device;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_shared_device() {
    if yawgpu_test::real_backend_skip_reason(yawgpu_test::RealBackend::Vulkan).is_some() {
        return;
    }
    concurrent_device::run(yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN);
}
