use yawgpu_test::{real_backend_available, real_backend_skip_reason, RealBackend};

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn metal_backend_gate_matches_probe() {
    let available = real_backend_available(RealBackend::Metal);
    assert_eq!(
        real_backend_skip_reason(RealBackend::Metal).is_none(),
        available
    );
}
