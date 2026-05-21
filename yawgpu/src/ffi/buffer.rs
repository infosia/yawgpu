use super::*;

/// Destroys a buffer. This operation is idempotent.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
/// Returns WGPU buffer destroy.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferDestroy(buffer: native::WGPUBuffer) {
    borrow_handle(buffer, "WGPUBuffer").core.destroy();
}

/// Unmaps a buffer. This is safe on unmapped, destroyed, and error buffers.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
/// Returns WGPU buffer unmap.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferUnmap(buffer: native::WGPUBuffer) {
    let buffer = borrow_handle(buffer, "WGPUBuffer");
    dispatch_optional_device_error(&buffer.device, buffer.core.unmap());
}

/// Asynchronously maps a buffer range.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle. `callback_info`
/// userdata pointers must remain valid until the callback fires.
/// Returns WGPU buffer map async.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferMapAsync(
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
    callback_info: native::WGPUBufferMapCallbackInfo,
) -> native::WGPUFuture {
    let buffer = borrow_handle(buffer, "WGPUBuffer");
    let map_result = validate_map_async(buffer, mode, offset, size);

    let pending = match map_result {
        Ok((mode, offset, size)) => match buffer.core.begin_map(mode, offset, size) {
            Ok(()) => PendingCallback::BufferMap {
                mode: callback_info.mode,
                callback: callback_info.callback,
                device: Arc::clone(&buffer.device),
                buffer: Some((*buffer.core).clone()),
                status: core::MapAsyncStatus::Success,
                userdata1: callback_info.userdata1 as usize,
                userdata2: callback_info.userdata2 as usize,
            },
            Err(message) => {
                buffer
                    .device
                    .dispatch_error(core::ErrorKind::Validation, message);
                PendingCallback::BufferMap {
                    mode: callback_info.mode,
                    callback: callback_info.callback,
                    device: Arc::clone(&buffer.device),
                    buffer: None,
                    status: core::MapAsyncStatus::Error,
                    userdata1: callback_info.userdata1 as usize,
                    userdata2: callback_info.userdata2 as usize,
                }
            }
        },
        Err(message) => {
            buffer
                .device
                .dispatch_error(core::ErrorKind::Validation, message);
            PendingCallback::BufferMap {
                mode: callback_info.mode,
                callback: callback_info.callback,
                device: Arc::clone(&buffer.device),
                buffer: None,
                status: core::MapAsyncStatus::Error,
                userdata1: callback_info.userdata1 as usize,
                userdata2: callback_info.userdata2 as usize,
            }
        }
    };

    buffer.instance.register_callback(pending)
}

/// Returns a mutable pointer to a mapped buffer range, or null on misuse.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle. The returned pointer
/// is valid only while the buffer remains mapped.
/// Returns WGPU buffer get mapped range.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetMappedRange(
    buffer: native::WGPUBuffer,
    offset: usize,
    size: usize,
) -> *mut c_void {
    mapped_range_ptr(buffer, false, offset, size).map_or(std::ptr::null_mut(), |ptr| ptr.cast())
}

/// Returns a const pointer to a mapped buffer range, or null on misuse.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle. The returned pointer
/// is valid only while the buffer remains mapped.
/// Returns WGPU buffer get const mapped range.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetConstMappedRange(
    buffer: native::WGPUBuffer,
    offset: usize,
    size: usize,
) -> *const c_void {
    mapped_range_ptr(buffer, true, offset, size)
        .map_or(std::ptr::null(), |ptr| ptr.cast_const().cast())
}

/// Returns the buffer map state.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
/// Maintains WGPU buffer get map state state.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetMapState(
    buffer: native::WGPUBuffer,
) -> native::WGPUBufferMapState {
    map_buffer_map_state(borrow_handle(buffer, "WGPUBuffer").core.map_state())
}

/// Returns the descriptor size reflected by the buffer.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
/// Returns WGPU buffer get size.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetSize(buffer: native::WGPUBuffer) -> u64 {
    borrow_handle(buffer, "WGPUBuffer").core.size()
}

/// Returns the descriptor usage reflected by the buffer.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
/// Returns WGPU buffer get usage.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetUsage(buffer: native::WGPUBuffer) -> native::WGPUBufferUsage {
    map_buffer_usage_to_native(borrow_handle(buffer, "WGPUBuffer").core.usage())
}

/// Releases one owned reference to a buffer handle.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
/// Returns WGPU buffer release.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferRelease(buffer: native::WGPUBuffer) {
    release_handle(buffer, "WGPUBuffer");
}

/// Adds one owned reference to a buffer handle.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
/// Returns WGPU buffer add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferAddRef(buffer: native::WGPUBuffer) {
    add_ref_handle(buffer, "WGPUBuffer");
}
