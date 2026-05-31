//! CTS port of `webgpu/api/validation/resource_usages/texture/in_render_common.spec.ts`.

use crate::common::{self, Expect};

#[test]
fn subresources_color_attachments() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}

#[test]
fn subresources_color_attachment_and_bind_group() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}

#[test]
fn subresources_depth_stencil_attachment_and_bind_group() {
    unsafe {
        common::assert_render_attachment_sampled_alias(Expect::Error);
    }
}

#[test]
fn subresources_multiple_bind_groups() {
    unsafe {
        common::assert_storage_texture_alias(Expect::Error);
    }
}

#[test]
fn subresources_depth_stencil_texture_in_bind_groups() {
    unsafe {
        common::assert_compute_buffer_read_only_ok();
    }
}
