use super::*;

/// Releases one owned reference to a queue handle.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle.
/// Returns WGPU queue release.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueRelease(queue: native::WGPUQueue) {
    release_handle(queue, "WGPUQueue");
}

/// Adds one owned reference to a queue handle.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle.
/// Returns WGPU queue add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueAddRef(queue: native::WGPUQueue) {
    add_ref_handle(queue, "WGPUQueue");
}

/// Sets the debug label for a queue.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle. `label` must point to
/// valid string data according to `WGPUStringView` when non-empty.
/// Returns WGPU queue set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueSetLabel(
    queue: native::WGPUQueue,
    label: native::WGPUStringView,
) {
    let queue = borrow_handle(queue, "WGPUQueue");
    let label = label_from_string_view(label).unwrap_or_default();
    queue.core.set_label(&label);
}

/// Schedules a callback once all submitted queue work is done.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle. `callback_info`
/// userdata pointers must remain valid until the callback fires.
/// Returns WGPU queue on submitted work done.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueOnSubmittedWorkDone(
    queue: native::WGPUQueue,
    callback_info: native::WGPUQueueWorkDoneCallbackInfo,
) -> native::WGPUFuture {
    let queue = borrow_handle(queue, "WGPUQueue");
    queue
        .instance
        .register_callback(PendingCallback::QueueWorkDone {
            mode: callback_info.mode,
            callback: callback_info.callback,
            device: Arc::clone(&queue.device),
            status: core::QueueWorkDoneStatus::Success,
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        })
}

/// Submits command buffers to a queue. Phase 2 validates only null arguments.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle. If `command_count` is
/// non-zero, `commands` must be non-null.
/// Returns WGPU queue submit.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueSubmit(
    queue: native::WGPUQueue,
    command_count: usize,
    commands: *const native::WGPUCommandBuffer,
) {
    let queue = borrow_handle(queue, "WGPUQueue");
    if command_count > 0 && commands.is_null() {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue submit commands must not be null when commandCount is non-zero",
        );
        return;
    }
    let commands = if command_count == 0 {
        Some(Vec::new())
    } else {
        std::slice::from_raw_parts(commands, command_count)
            .iter()
            .map(|command| {
                let command = clone_handle::<WGPUCommandBufferImpl>(*command, "WGPUCommandBuffer");
                if !command._device.same(&queue.device) {
                    queue.device.dispatch_error(
                        core::ErrorKind::Validation,
                        "command buffer must belong to the queue device",
                    );
                    None
                } else {
                    Some(Arc::clone(&command.core))
                }
            })
            .collect::<Option<Vec<_>>>()
    };
    let Some(commands) = commands else {
        return;
    };
    dispatch_optional_device_error(&queue.device, queue.core.submit(&commands));
}

/// Writes CPU data into a buffer through the queue.
///
/// # Safety
///
/// `queue` and `buffer` must be non-null live yawgpu handles. `data` must
/// point to `size` bytes when `size` is non-zero.
/// Returns WGPU queue write buffer.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueWriteBuffer(
    queue: native::WGPUQueue,
    buffer: native::WGPUBuffer,
    buffer_offset: u64,
    data: *const c_void,
    size: usize,
) {
    let queue = borrow_handle(queue, "WGPUQueue");
    let buffer = borrow_handle(buffer, "WGPUBuffer");
    if !buffer.device.same(&queue.device) {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue write buffer target must belong to the queue device",
        );
        return;
    }
    if size > 0 && data.is_null() {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue write data must not be null when size is non-zero",
        );
        return;
    }
    let data = std::slice::from_raw_parts(data.cast::<u8>(), size);
    dispatch_optional_device_error(
        &queue.device,
        queue.core.write_buffer(&buffer.core, buffer_offset, data),
    );
}

/// Validates a queue texture write. Noop does not copy bytes.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle. `destination`,
/// `data_layout`, and `write_size` must be non-null pointers to valid WebGPU
/// structs. `destination.texture` must be a non-null live yawgpu texture
/// handle. `data` is not read by the Noop validation implementation.
/// Returns WGPU queue write texture.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueWriteTexture(
    queue: native::WGPUQueue,
    destination: *const native::WGPUTexelCopyTextureInfo,
    _data: *const c_void,
    data_size: usize,
    data_layout: *const native::WGPUTexelCopyBufferLayout,
    write_size: *const native::WGPUExtent3D,
) {
    let queue = borrow_handle(queue, "WGPUQueue");
    let Some(destination) = destination.as_ref() else {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue write texture destination must not be null",
        );
        return;
    };
    let Some(data_layout) = data_layout.as_ref() else {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue write texture dataLayout must not be null",
        );
        return;
    };
    let Some(write_size) = write_size.as_ref() else {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue write texture writeSize must not be null",
        );
        return;
    };
    let data_size = match u64::try_from(data_size) {
        Ok(size) => size,
        Err(_) => {
            queue.device.dispatch_error(
                core::ErrorKind::Validation,
                "queue write texture dataSize is too large",
            );
            return;
        }
    };
    let texture = borrow_handle(destination.texture, "WGPUTexture");
    let aspect = map_texture_aspect(destination.aspect).unwrap_or(core::TextureAspect::All);

    if let Err(message) = texture.core.validate_queue_write(
        destination.mipLevel,
        map_origin_3d(destination.origin),
        map_extent_3d(*write_size),
        aspect,
        map_texel_copy_buffer_layout(*data_layout),
        data_size,
    ) {
        queue
            .device
            .dispatch_error(core::ErrorKind::Validation, message);
    }
}
