use super::*;

/// Begins a render pass.
///
/// # Safety
///
/// `command_encoder` and `descriptor` must be non-null live yawgpu pointers.
/// Returns WGPU command encoder begin render pass.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderBeginRenderPass(
    command_encoder: native::WGPUCommandEncoder,
    descriptor: *const native::WGPURenderPassDescriptor,
) -> native::WGPURenderPassEncoder {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPURenderPassDescriptor must not be null");
    let descriptor =
        map_render_pass_descriptor(descriptor, encoder.device.limits().max_color_attachments);
    let (pass, error) = encoder.core.begin_render_pass(&descriptor);
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(WGPURenderPassEncoderImpl {
        core: Arc::new(pass),
        device: Arc::clone(&encoder.device),
        _parent: Arc::clone(&encoder.core),
        _instance: Arc::clone(&encoder.instance),
    }))
}

/// Begins a compute pass. The descriptor is nullable by `webgpu.h`; P6.1
/// tracks lifecycle only.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
/// Returns WGPU command encoder begin compute pass.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderBeginComputePass(
    command_encoder: native::WGPUCommandEncoder,
    _descriptor: *const native::WGPUComputePassDescriptor,
) -> native::WGPUComputePassEncoder {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let (pass, error) = encoder.core.begin_compute_pass();
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(WGPUComputePassEncoderImpl {
        core: Arc::new(pass),
        device: Arc::clone(&encoder.device),
        _parent: Arc::clone(&encoder.core),
        _instance: Arc::clone(&encoder.instance),
    }))
}

/// Begins a subpass render pass.
///
/// # Safety
///
/// `command_encoder` and `descriptor` must be non-null live yawgpu pointers.
/// Returns yawgpu command encoder begin subpass render pass.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuCommandEncoderBeginSubpassRenderPass(
    command_encoder: native::WGPUCommandEncoder,
    descriptor: *const YaWGPUSubpassRenderPassDescriptor,
) -> crate::YaWGPUSubpassRenderPassEncoder {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let descriptor = descriptor
        .as_ref()
        .expect("YaWGPUSubpassRenderPassDescriptor must not be null");
    let descriptor = map_subpass_render_pass_descriptor(descriptor);
    let layout = Arc::clone(&descriptor.pass_layout);
    let (pass, error) = encoder
        .core
        .begin_subpass_render_pass(&encoder.device, descriptor);
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(YaWGPUSubpassRenderPassEncoderImpl {
        core: Arc::new(pass),
        device: Arc::clone(&encoder.device),
        _parent: Arc::clone(&encoder.core),
        _layout: layout,
        _instance: Arc::clone(&encoder.instance),
    }))
}

/// Finishes command encoding into a command buffer.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
/// `descriptor` may be null; P6.1 stores no command buffer descriptor fields.
/// Returns WGPU command encoder finish.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderFinish(
    command_encoder: native::WGPUCommandEncoder,
    _descriptor: *const native::WGPUCommandBufferDescriptor,
) -> native::WGPUCommandBuffer {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let (command_buffer, error) = encoder.core.finish();
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(WGPUCommandBufferImpl {
        core: Arc::new(command_buffer),
        _device: Arc::clone(&encoder.device),
        _instance: Arc::clone(&encoder.instance),
    }))
}

/// Inserts an encoder debug marker.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
/// Returns WGPU command encoder insert debug marker.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderInsertDebugMarker(
    command_encoder: native::WGPUCommandEncoder,
    _marker_label: native::WGPUStringView,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.insert_debug_marker());
}

/// Pushes an encoder debug group.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
/// Returns WGPU command encoder push debug group.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderPushDebugGroup(
    command_encoder: native::WGPUCommandEncoder,
    _group_label: native::WGPUStringView,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.push_debug_group());
}

/// Pops an encoder debug group.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
/// Returns WGPU command encoder pop debug group.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderPopDebugGroup(
    command_encoder: native::WGPUCommandEncoder,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.pop_debug_group());
}

/// Records a buffer-to-buffer copy command.
///
/// # Safety
///
/// `command_encoder`, `source`, and `destination` must be non-null live yawgpu
/// handles.
/// Returns WGPU command encoder copy buffer to buffer.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyBufferToBuffer(
    command_encoder: native::WGPUCommandEncoder,
    source: native::WGPUBuffer,
    source_offset: u64,
    destination: native::WGPUBuffer,
    destination_offset: u64,
    size: u64,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let source = clone_handle(source, "WGPUBuffer");
    let destination = clone_handle(destination, "WGPUBuffer");
    if !source.device.same(&encoder.device) || !destination.device.same(&encoder.device) {
        dispatch_optional_error(
            &encoder.device,
            encoder
                .core
                .record_validation_error("copy buffers must belong to the command encoder device"),
        );
        return;
    }
    dispatch_optional_error(
        &encoder.device,
        encoder.core.copy_buffer_to_buffer(
            Arc::clone(&source.core),
            source_offset,
            Arc::clone(&destination.core),
            destination_offset,
            size,
        ),
    );
}

/// Records a buffer clear command.
///
/// # Safety
///
/// `command_encoder` and `buffer` must be non-null live yawgpu handles.
/// Returns WGPU command encoder clear buffer.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderClearBuffer(
    command_encoder: native::WGPUCommandEncoder,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let buffer = clone_handle(buffer, "WGPUBuffer");
    dispatch_optional_error(
        &encoder.device,
        encoder
            .core
            .clear_buffer(Arc::clone(&buffer.core), offset, size),
    );
}

/// Records a host-to-buffer write command. Noop validation does not consume
/// the `data` bytes.
///
/// # Safety
///
/// `command_encoder` and `buffer` must be non-null live yawgpu handles. `data`
/// is not read by this P6.2 validation implementation.
/// Returns WGPU command encoder write buffer.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderWriteBuffer(
    command_encoder: native::WGPUCommandEncoder,
    buffer: native::WGPUBuffer,
    buffer_offset: u64,
    _data: *const c_void,
    size: usize,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let buffer = clone_handle(buffer, "WGPUBuffer");
    let size = match u64::try_from(size) {
        Ok(size) => size,
        Err(_) => {
            dispatch_optional_error(
                &encoder.device,
                Some("command encoder write buffer size is too large".to_owned()),
            );
            return;
        }
    };
    dispatch_optional_error(
        &encoder.device,
        encoder
            .core
            .write_buffer(Arc::clone(&buffer.core), buffer_offset, size),
    );
}

/// Records a timestamp write command.
///
/// # Safety
///
/// `command_encoder` and `query_set` must be non-null live yawgpu handles.
/// Returns WGPU command encoder write timestamp.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderWriteTimestamp(
    command_encoder: native::WGPUCommandEncoder,
    query_set: native::WGPUQuerySet,
    query_index: u32,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let query_set = clone_handle(query_set, "WGPUQuerySet");
    dispatch_optional_error(
        &encoder.device,
        encoder
            .core
            .write_timestamp(Arc::clone(&query_set.core), query_index),
    );
}

/// Records a query set resolve command.
///
/// # Safety
///
/// `command_encoder`, `query_set`, and `destination` must be non-null live
/// yawgpu handles.
/// Returns WGPU command encoder resolve query set.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderResolveQuerySet(
    command_encoder: native::WGPUCommandEncoder,
    query_set: native::WGPUQuerySet,
    first_query: u32,
    query_count: u32,
    destination: native::WGPUBuffer,
    destination_offset: u64,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let query_set = clone_handle(query_set, "WGPUQuerySet");
    let destination = clone_handle(destination, "WGPUBuffer");
    dispatch_optional_error(
        &encoder.device,
        encoder.core.resolve_query_set(
            Arc::clone(&query_set.core),
            first_query,
            query_count,
            Arc::clone(&destination.core),
            destination_offset,
        ),
    );
}

/// Records a buffer-to-texture copy command.
///
/// # Safety
///
/// `command_encoder`, `source`, `destination`, and `copy_size` must be
/// non-null. Nested buffer and texture handles must be non-null live yawgpu
/// handles.
/// Returns WGPU command encoder copy buffer to texture.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyBufferToTexture(
    command_encoder: native::WGPUCommandEncoder,
    source: *const native::WGPUTexelCopyBufferInfo,
    destination: *const native::WGPUTexelCopyTextureInfo,
    copy_size: *const native::WGPUExtent3D,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let source = source
        .as_ref()
        .expect("wgpuCommandEncoderCopyBufferToTexture source must not be null");
    let destination = destination
        .as_ref()
        .expect("wgpuCommandEncoderCopyBufferToTexture destination must not be null");
    let copy_size = copy_size
        .as_ref()
        .expect("wgpuCommandEncoderCopyBufferToTexture copySize must not be null");
    let source_buffer = clone_handle(source.buffer, "WGPUBuffer");
    let destination_texture = clone_handle(destination.texture, "WGPUTexture");
    let (destination_mip_level, destination_origin, destination_aspect) =
        map_texel_copy_texture_info_parts(destination);

    dispatch_optional_error(
        &encoder.device,
        encoder.core.copy_buffer_to_texture(
            core::TexelCopyBufferInfo {
                buffer: Arc::clone(&source_buffer.core),
                layout: map_texel_copy_buffer_layout(source.layout),
            },
            core::TexelCopyTextureInfo {
                texture: Arc::clone(&destination_texture.core),
                mip_level: destination_mip_level,
                origin: destination_origin,
                aspect: destination_aspect,
            },
            map_extent_3d(*copy_size),
        ),
    );
}

/// Records a texture-to-buffer copy command.
///
/// # Safety
///
/// `command_encoder`, `source`, `destination`, and `copy_size` must be
/// non-null. Nested texture and buffer handles must be non-null live yawgpu
/// handles.
/// Returns WGPU command encoder copy texture to buffer.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyTextureToBuffer(
    command_encoder: native::WGPUCommandEncoder,
    source: *const native::WGPUTexelCopyTextureInfo,
    destination: *const native::WGPUTexelCopyBufferInfo,
    copy_size: *const native::WGPUExtent3D,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let source = source
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToBuffer source must not be null");
    let destination = destination
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToBuffer destination must not be null");
    let copy_size = copy_size
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToBuffer copySize must not be null");
    let source_texture = clone_handle(source.texture, "WGPUTexture");
    let destination_buffer = clone_handle(destination.buffer, "WGPUBuffer");
    let (source_mip_level, source_origin, source_aspect) =
        map_texel_copy_texture_info_parts(source);

    dispatch_optional_error(
        &encoder.device,
        encoder.core.copy_texture_to_buffer(
            core::TexelCopyTextureInfo {
                texture: Arc::clone(&source_texture.core),
                mip_level: source_mip_level,
                origin: source_origin,
                aspect: source_aspect,
            },
            core::TexelCopyBufferInfo {
                buffer: Arc::clone(&destination_buffer.core),
                layout: map_texel_copy_buffer_layout(destination.layout),
            },
            map_extent_3d(*copy_size),
        ),
    );
}

/// Records a texture-to-texture copy command.
///
/// # Safety
///
/// `command_encoder`, `source`, `destination`, and `copy_size` must be
/// non-null. Nested texture handles must be non-null live yawgpu handles.
/// Returns WGPU command encoder copy texture to texture.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyTextureToTexture(
    command_encoder: native::WGPUCommandEncoder,
    source: *const native::WGPUTexelCopyTextureInfo,
    destination: *const native::WGPUTexelCopyTextureInfo,
    copy_size: *const native::WGPUExtent3D,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let source = source
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToTexture source must not be null");
    let destination = destination
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToTexture destination must not be null");
    let copy_size = copy_size
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToTexture copySize must not be null");
    let source_texture = clone_handle(source.texture, "WGPUTexture");
    let destination_texture = clone_handle(destination.texture, "WGPUTexture");
    let (source_mip_level, source_origin, source_aspect) =
        map_texel_copy_texture_info_parts(source);
    let (destination_mip_level, destination_origin, destination_aspect) =
        map_texel_copy_texture_info_parts(destination);

    dispatch_optional_error(
        &encoder.device,
        encoder.core.copy_texture_to_texture(
            core::TexelCopyTextureInfo {
                texture: Arc::clone(&source_texture.core),
                mip_level: source_mip_level,
                origin: source_origin,
                aspect: source_aspect,
            },
            core::TexelCopyTextureInfo {
                texture: Arc::clone(&destination_texture.core),
                mip_level: destination_mip_level,
                origin: destination_origin,
                aspect: destination_aspect,
            },
            map_extent_3d(*copy_size),
        ),
    );
}

/// Releases one owned reference to a command encoder handle.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
/// Returns WGPU command encoder release.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderRelease(command_encoder: native::WGPUCommandEncoder) {
    release_handle(command_encoder, "WGPUCommandEncoder");
}

/// Adds one owned reference to a command encoder handle.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
/// Returns WGPU command encoder add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderAddRef(command_encoder: native::WGPUCommandEncoder) {
    add_ref_handle(command_encoder, "WGPUCommandEncoder");
}
