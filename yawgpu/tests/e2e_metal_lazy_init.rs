//! Real-Metal e2e for Block 104 (`specs/blocks/104-lazy-zero-init-stage2.md`);
//! the scenarios live in `common/lazy_init.rs` and run here against the
//! Metal backend.

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/lazy_init.rs"]
mod lazy_init;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn lazy_init_sampled_read_of_uninitialized_mip_is_zero_and_canary_mips_survive() {
    lazy_init::lazy_init_sampled_read_of_uninitialized_mip_is_zero_and_canary_mips_survive(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn lazy_init_depth_and_stencil_copies_read_zero() {
    lazy_init::lazy_init_depth_and_stencil_copies_read_zero(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn lazy_init_non_renderable_depth_stencil_copies_read_zero() {
    lazy_init::lazy_init_non_renderable_depth_stencil_copies_read_zero(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn lazy_init_multisampled_load_attachment_resolves_to_zero() {
    lazy_init::lazy_init_multisampled_load_attachment_resolves_to_zero(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn lazy_init_compressed_mip_copies_zero_bytes_and_canary_mip_survives() {
    lazy_init::lazy_init_compressed_mip_copies_zero_bytes_and_canary_mip_survives(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn lazy_init_store_op_discard_makes_the_next_read_zero() {
    lazy_init::lazy_init_store_op_discard_makes_the_next_read_zero(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn lazy_init_partial_write_then_sampled_read_keeps_written_texels() {
    lazy_init::lazy_init_partial_write_then_sampled_read_keeps_written_texels(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}
