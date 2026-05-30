//! CTS port of `webgpu/api/validation/capability_checks/limits/maxBufferSize.spec.ts`.

use crate::common;

#[test]
fn create_buffer_at_over() {
    unsafe { common::assert_max_buffer_size_create_buffer_at_over() };
}
