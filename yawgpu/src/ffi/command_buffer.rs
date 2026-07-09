use super::*;

/// Sets a command buffer label.
///
/// # Safety
///
/// `command_buffer` must be a non-null live yawgpu command buffer handle.
/// `label` must point to valid string data according to `WGPUStringView` when
/// non-empty.
/// Returns WGPU command buffer set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandBufferSetLabel(
    command_buffer: native::WGPUCommandBuffer,
    label: native::WGPUStringView,
) {
    let command_buffer = borrow_handle(command_buffer, "WGPUCommandBuffer");
    *command_buffer
        .label
        .lock()
        .expect("label lock must not poison") = label_from_string_view(label);
}

/// Releases one owned reference to a command buffer handle.
///
/// # Safety
///
/// `command_buffer` must be a non-null live yawgpu command buffer handle.
/// Returns WGPU command buffer release.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandBufferRelease(command_buffer: native::WGPUCommandBuffer) {
    release_handle(command_buffer, "WGPUCommandBuffer");
}

/// Adds one owned reference to a command buffer handle.
///
/// # Safety
///
/// `command_buffer` must be a non-null live yawgpu command buffer handle.
/// Returns WGPU command buffer add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandBufferAddRef(command_buffer: native::WGPUCommandBuffer) {
    add_ref_handle(command_buffer, "WGPUCommandBuffer");
}
