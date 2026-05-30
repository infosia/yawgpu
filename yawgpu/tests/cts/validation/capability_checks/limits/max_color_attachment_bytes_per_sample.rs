//! CTS port of `webgpu/api/validation/capability_checks/limits/maxColorAttachmentBytesPerSample.spec.ts`.

use crate::common;

#[test]
#[ignore = "bytes-per-sample over-limit matrix needs format selection coverage beyond color attachment count"]
fn create_render_pipeline_at_over() {
    unsafe { common::assert_max_color_attachments_render_pipeline_at_over() };
}

#[test]
#[ignore = "bytes-per-sample over-limit matrix needs render-pass format selection coverage"]
fn begin_render_pass_at_over() {
    unsafe { common::assert_max_color_attachments_render_pass_at_over() };
}

#[test]
#[ignore = "render bundle bytes-per-sample creator matrix is not active in this slice"]
fn create_render_bundle_at_over() {
    unsafe { common::assert_max_color_attachments_render_pipeline_at_over() };
}
