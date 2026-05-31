//! CTS port of `webgpu/api/validation/resource_usages/buffer/in_pass_misc.spec.ts`.

use crate::common::{self, Expect};

#[test]
fn subresources_reset_buffer_usage_before_dispatch() {
    unsafe {
        common::assert_compute_buffer_read_only_ok();
    }
}

#[test]
fn subresources_reset_buffer_usage_before_draw() {
    unsafe {
        common::assert_render_buffer_read_write_alias(Expect::Error);
    }
}

#[test]
fn subresources_buffer_usages_in_copy_and_pass() {
    unsafe {
        common::assert_copy_and_pass_buffer_ok();
    }
}
