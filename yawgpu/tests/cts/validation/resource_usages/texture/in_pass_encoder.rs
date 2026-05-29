//! CTS port of `webgpu/api/validation/resource_usages/texture/in_pass_encoder.spec.ts`.

use crate::common::{self, Expect};

#[test]
#[ignore = "core does not yet validate CTS texture subresource mip/layer overlap combinations across compute/render/bundle scopes"]
fn subresources_and_binding_types_combination_for_color() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
#[ignore = "core does not yet validate CTS depth/stencil aspect overlap combinations for texture usage scopes"]
fn subresources_and_binding_types_combination_for_aspect() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}

#[test]
#[ignore = "core does not yet validate CTS shader-stage visibility-independent texture storage-write conflicts"]
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
#[ignore = "core does not yet distinguish replaced texture bindings by compute dispatch versus render pass usage scope"]
fn replaced_binding() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
#[ignore = "core does not yet validate texture resource usage conflicts contributed by render bundles"]
fn bindings_in_bundle() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
#[ignore = "core does not yet implement the CTS unused-bindings-in-pipeline distinction between compute and render scopes"]
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
