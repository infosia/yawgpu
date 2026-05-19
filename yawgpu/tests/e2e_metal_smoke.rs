use yawgpu_test::{real_backend_available, real_backend_skip_reason, RealBackend};

#[test]
#[ignore = "real-backend smoke tests are manually run with backend features"]
fn metal_backend_gate_reports_unavailable_until_p7_1() {
    assert!(!real_backend_available(RealBackend::Metal));
    assert!(real_backend_skip_reason(RealBackend::Metal)
        .expect("P7.0 metal backend should be unavailable")
        .contains("metal backend is unavailable"));
}
