//! CTS port of `webgpu/api/validation/capability_checks/limits/maxUniformBufferBindingSize.spec.ts`.

use crate::common;

#[test]
fn create_bind_group_at_over() {
    unsafe { common::assert_uniform_buffer_binding_size_at_over() };
}

#[test]
#[ignore = "maxUniformBufferBindingSize/maxBufferSize relationship validation is not yet active"]
fn validate_max_buffer_size() {
    unsafe { common::assert_required_limits_are_not_lowered_to_requested_values() };
}
