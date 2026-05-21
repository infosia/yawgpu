use super::*;

/// Releases one owned reference to a command buffer handle.
///
/// # Safety
///
/// `command_buffer` must be a non-null live yawgpu command buffer handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandBufferRelease(command_buffer: native::WGPUCommandBuffer) {
    release_handle(command_buffer, "WGPUCommandBuffer");
}

/// Adds one owned reference to a command buffer handle.
///
/// # Safety
///
/// `command_buffer` must be a non-null live yawgpu command buffer handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandBufferAddRef(command_buffer: native::WGPUCommandBuffer) {
    add_ref_handle(command_buffer, "WGPUCommandBuffer");
}
