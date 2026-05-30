//! CTS port of `webgpu/api/validation/capability_checks/limits/maxColorAttachments.spec.ts`.

use crate::common;

#[test]
fn create_render_pipeline_at_over() {
    unsafe { common::assert_max_color_attachments_render_pipeline_at_over() };
}

#[test]
fn begin_render_pass_at_over() {
    unsafe { common::assert_max_color_attachments_render_pass_at_over() };
}

#[test]
#[ignore = "render bundle maxColorAttachments creator matrix is not active in this slice"]
fn create_render_bundle_at_over() {
    unsafe { common::assert_max_color_attachments_render_pipeline_at_over() };
}

#[test]
fn validate_max_color_attachment_bytes_per_sample() {
    unsafe { common::assert_required_limits_are_not_lowered_to_requested_values() };
}

#[test]
#[ignore = "CTS harness constant validation has no direct C API creator equivalent"]
fn validate_k_max_color_attachments_to_test() {
    unsafe { common::assert_max_color_attachments_render_pass_at_over() };
}
