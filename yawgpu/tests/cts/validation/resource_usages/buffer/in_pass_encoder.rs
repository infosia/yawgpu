//! CTS port of `webgpu/api/validation/resource_usages/buffer/in_pass_encoder.spec.ts`.

use crate::common::{self, Expect};

#[test]
fn subresources_buffer_usage_in_one_compute_pass_with_no_dispatch() {
    unsafe {
        common::assert_compute_buffer_read_only_ok();
    }
}

#[test]
fn subresources_buffer_usage_in_one_compute_pass_with_one_dispatch() {
    unsafe {
        common::assert_compute_buffer_alias(Expect::Error);
    }
}

#[test]
fn subresources_buffer_usage_in_compute_pass_with_two_dispatches() {
    unsafe {
        common::assert_compute_buffer_read_only_ok();
    }
}

#[test]
fn subresources_buffer_usage_in_one_render_pass_with_no_draw() {
    unsafe {
        common::assert_render_buffer_read_write_alias(Expect::Error);
    }
}

#[test]
fn subresources_buffer_usage_in_one_render_pass_with_one_draw() {
    unsafe {
        common::assert_render_buffer_read_write_alias(Expect::Error);
    }
}

#[test]
fn subresources_buffer_usage_in_one_render_pass_with_two_draws() {
    unsafe {
        common::assert_render_buffer_read_write_alias(Expect::Error);
    }
}
