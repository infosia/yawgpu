//! Real-Metal e2e for Block 119 R5 (`specs/blocks/119-cts-d116bcc-findings.md`);
//! the scenarios live in `common/indirect_dispatch.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn valid_counts_pass_through() {
    indirect_dispatch::valid_counts_pass_through(RealBackend::Metal, YAWGPU_INSTANCE_BACKEND_METAL);
}

#[test]
#[ignore = "manual real-backend test"]
fn zero_counts_run_nothing_without_error() {
    indirect_dispatch::zero_counts_run_nothing_without_error(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn over_limit_counts_skip_on_every_pipeline() {
    indirect_dispatch::over_limit_counts_skip_on_every_pipeline(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn linear_index_overflow_skips() {
    indirect_dispatch::linear_index_overflow_skips(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn unaligned_client_offsets_read_the_right_words() {
    indirect_dispatch::unaligned_client_offsets_read_the_right_words(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn repeated_dispatches_in_one_pass_see_no_stale_value() {
    indirect_dispatch::repeated_dispatches_in_one_pass_see_no_stale_value(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn pass_state_survives_for_a_following_direct_dispatch() {
    indirect_dispatch::pass_state_survives_for_a_following_direct_dispatch(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn arguments_written_on_the_gpu_are_read_after_the_write() {
    indirect_dispatch::arguments_written_on_the_gpu_are_read_after_the_write(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}
