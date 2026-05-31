//! CTS port of `webgpu/api/validation/resource_usages/texture/in_pass_encoder.spec.ts`.

use crate::common::{self, Expect};

#[test]
fn subresources_and_binding_types_combination_for_color() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
fn subresources_and_binding_types_combination_for_aspect() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}

#[test]
fn shader_stages_and_visibility_storage_write() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
fn shader_stages_and_visibility_attachment_write() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}

#[test]
fn replaced_binding() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
fn bindings_in_bundle() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
fn unused_bindings_in_pipeline() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
fn scope_dispatch() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
fn scope_basic_render() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}

#[test]
fn scope_pass_boundary_compute() {
    unsafe {
        common::assert_compute_buffer_read_only_ok();
    }
}

#[test]
fn scope_pass_boundary_render() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}
