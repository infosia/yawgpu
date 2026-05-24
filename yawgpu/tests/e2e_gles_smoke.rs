#![cfg(feature = "gles")]

use yawgpu_test::{real_backend_available, real_backend_skip_reason, RealBackend};

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn gles_backend_gate_matches_probe() {
    let available = real_backend_available(RealBackend::Gles);
    assert_eq!(
        real_backend_skip_reason(RealBackend::Gles).is_none(),
        available,
    );
}
