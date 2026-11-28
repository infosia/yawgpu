#[path = "common/concurrent_device.rs"]
mod concurrent_device;

#[test]
fn shared_noop_device() {
    concurrent_device::run(yawgpu::YAWGPU_INSTANCE_BACKEND_NOOP);
}
