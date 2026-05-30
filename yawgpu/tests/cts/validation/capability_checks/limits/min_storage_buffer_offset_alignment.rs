//! CTS port of `webgpu/api/validation/capability_checks/limits/minStorageBufferOffsetAlignment.spec.ts`.

use crate::common;

#[test]
fn create_bind_group_at_over() {
    unsafe { common::assert_min_storage_buffer_offset_alignment_at_over() };
}

#[test]
#[ignore = "setBindGroup dynamic offset alignment matrix is not yet active"]
fn set_bind_group_at_over() {
    unsafe { common::assert_min_storage_buffer_offset_alignment_at_over() };
}

#[test]
#[ignore = "minimum-limit power-of-two device-request relationship validation is not yet active"]
fn validate_power_of_2() {
    unsafe { common::assert_required_limits_are_not_lowered_to_requested_values() };
}

#[test]
#[ignore = "minimum-limit >=32 device-request relationship validation is not yet active"]
fn validate_greater_than_or_equal_to_32() {
    unsafe { common::assert_required_limits_are_not_lowered_to_requested_values() };
}
