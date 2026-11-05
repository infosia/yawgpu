//! Real-Vulkan external-texture e2e; the scenarios live in `common/external_texture.rs`.
//!
//! Run manually on a real GPU:
//!   cargo test -p yawgpu --features vulkan --test e2e_vulkan_external_texture -- --ignored --nocapture

#![cfg(feature = "vulkan")]

#[path = "common/external_texture.rs"]
mod external_texture;

use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::RealBackend;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_external_texture_rgba_passthrough_round_trips() {
    external_texture::external_texture_rgba_passthrough_round_trips(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_compute_unreferenced_external_texture_keeps_buffer_sizes() {
    external_texture::compute_unreferenced_external_texture_keeps_buffer_sizes(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_external_texture_nv12_two_plane_round_trips() {
    external_texture::external_texture_nv12_two_plane_round_trips(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_external_texture_discarded_planes_read_zero() {
    external_texture::external_texture_discarded_planes_read_zero(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_external_texture_samples_plane_view_subresource() {
    external_texture::external_texture_samples_plane_view_subresource(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_external_textures_in_a_sparse_group_through_a_bundle() {
    external_texture::external_textures_in_a_sparse_group_through_a_bundle(
        RealBackend::Vulkan,
        YAWGPU_INSTANCE_BACKEND_VULKAN,
    );
}
