//! CTS port of `webgpu/api/validation/capability_checks/limits/maxColorAttachmentBytesPerSample.spec.ts`.

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
fn create_render_bundle_at_over() {
    unsafe { common::assert_max_color_attachments_render_pipeline_at_over() };
}
