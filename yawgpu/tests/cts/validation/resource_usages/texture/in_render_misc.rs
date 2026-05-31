//! CTS port of `webgpu/api/validation/resource_usages/texture/in_render_misc.spec.ts`.

use crate::common::{self, Expect};

#[test]
fn subresources_set_bind_group_on_same_index_color_texture() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
fn subresources_set_bind_group_on_same_index_depth_stencil_texture() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}

#[test]
fn subresources_set_unused_bind_group() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
fn subresources_texture_usages_in_copy_and_render_pass() {
    unsafe {
        common::assert_copy_and_render_texture_ok();
    }
}

#[test]
fn subresources_texture_view_usages() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}
