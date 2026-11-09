//! Real-Metal e2e for external textures in subpass draws; the scenarios live in
//! `common/tiled_external_texture.rs`.
//!
//! Run manually on a real GPU:
//!   cargo test -p yawgpu --features metal,tiled --test e2e_metal_tiled_external_texture -- --ignored

#![cfg(all(feature = "metal", feature = "tiled"))]

#[path = "common/tiled_external_texture.rs"]
mod tiled_external_texture;

use yawgpu::YAWGPU_INSTANCE_BACKEND_METAL;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn metal_subpass_samples_external_texture() {
    tiled_external_texture::subpass_samples_external_texture(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_subpass_reads_input_attachment_and_external_texture() {
    tiled_external_texture::subpass_reads_input_attachment_and_external_texture(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_subpass_unused_external_texture_keeps_buffer_sizes() {
    tiled_external_texture::subpass_unused_external_texture_keeps_buffer_sizes(
        RealBackend::Metal,
        YAWGPU_INSTANCE_BACKEND_METAL,
    );
}
