//! CTS port of `webgpu/api/validation/resource_usages/texture/in_render_misc.spec.ts`.

use crate::common::{self, Expect};

#[test]
#[ignore = "core does not yet validate the CTS setBindGroup same-index color texture replacement matrix"]
fn subresources_set_bind_group_on_same_index_color_texture() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
#[ignore = "core does not yet validate the CTS setBindGroup same-index depth/stencil texture replacement matrix"]
fn subresources_set_bind_group_on_same_index_depth_stencil_texture() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}

#[test]
#[ignore = "core does not yet validate unused texture bind groups in render scopes with the CTS compute/render distinction"]
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
#[ignore = "core does not yet apply texture view usage overrides to every CTS binding and attachment validation path"]
fn subresources_texture_view_usages() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}
