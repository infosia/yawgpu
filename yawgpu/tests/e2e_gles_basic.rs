#![cfg(feature = "gles")]

use yawgpu_hal::gles::GlesInstance;
use yawgpu_hal::{HalBackend, HalInstance};
use yawgpu_test::{real_backend_available, RealBackend};

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn gles_adapter_name_is_present() {
    if !real_backend_available(RealBackend::Gles) {
        eprintln!("skip: no GLES adapter");
        return;
    }
    let instance = HalInstance::Gles(GlesInstance::new().expect("EGL init"));
    let adapter = instance
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("one adapter");

    assert!(!adapter.name().is_empty());
    assert_eq!(adapter.backend(), HalBackend::Gles);
}

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn gles_device_queue_submits_empty() {
    if !real_backend_available(RealBackend::Gles) {
        eprintln!("skip: no GLES adapter");
        return;
    }
    let instance = HalInstance::Gles(GlesInstance::new().expect("EGL init"));
    let adapter = instance
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("adapter");
    let device = adapter.create_device().expect("device");
    let queue = device.queue();

    queue.submit_empty().expect("empty submit");
}

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn gles_device_reports_zero_allocations_at_creation() {
    if !real_backend_available(RealBackend::Gles) {
        eprintln!("skip: no GLES adapter");
        return;
    }
    let instance = HalInstance::Gles(GlesInstance::new().expect("EGL init"));
    let adapter = instance
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("adapter");
    let device = adapter.create_device().expect("device");

    assert_eq!(device.allocation_count(), 0);
}
