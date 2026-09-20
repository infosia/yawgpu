#![cfg(feature = "gles")]

use yawgpu_hal::gles::GlesInstance;
use yawgpu_hal::{HalBackend, HalInstance};
use yawgpu_test::{real_backend_available, RealBackend};

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn gles_adapter_name_is_present() {
    assert!(
        real_backend_available(RealBackend::Gles),
        "GLES backend not available; install an ANGLE build with ES 3.1 support \
         (Chrome / Edge ANGLE caps at ES 3.0; see specs/blocks/67-gles-backend.md). \
         Without a real GLES adapter this test cannot verify the GLES execution path."
    );
    let instance = HalInstance::Gles(GlesInstance::new().expect("EGL init"));
    let adapter = instance
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("one adapter");

    // L3 / D4: the name must carry the driver's own strings, not a constant —
    // a constant name is exactly what hid the llvmpipe fallback. The shape is
    // `yawgpu GLES Adapter (<backend>) — <GL_RENDERER> / <GL_VERSION>`, so
    // split it apart and check both halves rather than looking for a substring
    // a hard-coded name could also contain. `unknown` is the degraded
    // placeholder a driver-less name falls back to; on a real GPU neither
    // segment may be it.
    let name = adapter.name();
    assert!(!name.is_empty());
    let (prefix, driver) = name
        .split_once(" — ")
        .unwrap_or_else(|| panic!("adapter name should carry the driver strings, got {name:?}"));
    assert!(
        prefix.starts_with("yawgpu GLES Adapter ("),
        "unexpected adapter name prefix, got {name:?}"
    );
    let (renderer, version) = driver.split_once(" / ").unwrap_or_else(|| {
        panic!("adapter name should carry GL_RENDERER / GL_VERSION, got {name:?}")
    });
    assert!(
        !renderer.trim().is_empty() && renderer != "unknown",
        "adapter name should carry a real GL_RENDERER, got {name:?}"
    );
    assert!(
        version.contains("OpenGL ES"),
        "adapter name should carry GL_VERSION, got {name:?}"
    );
    assert_eq!(adapter.backend(), HalBackend::Gles);
}

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn gles_device_queue_submits_empty() {
    assert!(
        real_backend_available(RealBackend::Gles),
        "GLES backend not available; install an ANGLE build with ES 3.1 support \
         (Chrome / Edge ANGLE caps at ES 3.0; see specs/blocks/67-gles-backend.md). \
         Without a real GLES adapter this test cannot verify the GLES execution path."
    );
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
    assert!(
        real_backend_available(RealBackend::Gles),
        "GLES backend not available; install an ANGLE build with ES 3.1 support \
         (Chrome / Edge ANGLE caps at ES 3.0; see specs/blocks/67-gles-backend.md). \
         Without a real GLES adapter this test cannot verify the GLES execution path."
    );
    let instance = HalInstance::Gles(GlesInstance::new().expect("EGL init"));
    let adapter = instance
        .enumerate_adapters()
        .into_iter()
        .next()
        .expect("adapter");
    let device = adapter.create_device().expect("device");

    assert_eq!(device.allocation_count(), 0);
}
