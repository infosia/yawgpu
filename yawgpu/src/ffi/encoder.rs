use super::*;
use crate::conv::map_render_pass_timestamp_writes;
use yawgpu_core::validate_compute_pass_timestamp_writes;

wgpu_handle_exports!(
    refcount_and_label:
    WGPUCommandEncoderImpl,
    native::WGPUCommandEncoder,
    "WGPUCommandEncoder",
    wgpuCommandEncoderAddRef,
    wgpuCommandEncoderRelease,
    wgpuCommandEncoderSetLabel
);

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
    let native_descriptor = descriptor
        .as_ref()
        .expect("WGPURenderPassDescriptor must not be null");
    let descriptor = map_render_pass_descriptor(
        native_descriptor,
        encoder.device.limits().max_color_attachments,
    );
    let (pass, error) = encoder.core.begin_render_pass(&descriptor);
    dispatch_optional_error(&encoder.device, error);
    if let Some(message) =
        validate_render_pass_descriptor_devices(native_descriptor, &encoder.device)
    {
        dispatch_optional_error(&encoder.device, pass.record_validation_error(message));
    }
    arc_to_handle(Arc::new(WGPURenderPassEncoderImpl {
        core: Arc::new(pass),
        device: Arc::clone(&encoder.device),
        _parent: Arc::clone(&encoder.core),
        _instance: Arc::clone(&encoder.instance),
        label: Mutex::new(label_from_string_view(native_descriptor.label)),
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
    descriptor: *const native::WGPUComputePassDescriptor,
) -> native::WGPUComputePassEncoder {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let mut mapped_writes = None;
    if let Some(timestamp_writes) = descriptor
        .as_ref()
        .and_then(|descriptor| descriptor.timestampWrites.as_ref())
    {
        let query_set = borrow_handle(timestamp_writes.querySet, "WGPUQuerySet");
        if !query_set._device.same(&encoder.device) {
            dispatch_optional_error(
                &encoder.device,
                encoder.core.record_validation_error(
                    "compute pass timestamp query set must belong to the command encoder device",
                ),
            );
        } else {
            let timestamp_writes = map_render_pass_timestamp_writes(timestamp_writes);
            if let Err(message) = validate_compute_pass_timestamp_writes(&timestamp_writes) {
                dispatch_optional_error(
                    &encoder.device,
                    encoder.core.record_validation_error(message),
                );
            } else {
                mapped_writes = Some(timestamp_writes);
            }
        }
    }
    let (pass, error) = encoder.core.begin_compute_pass(mapped_writes);
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(WGPUComputePassEncoderImpl {
        core: Arc::new(pass),
        device: Arc::clone(&encoder.device),
        _parent: Arc::clone(&encoder.core),
        _instance: Arc::clone(&encoder.instance),
        label: Mutex::new(
            descriptor
                .as_ref()
                .and_then(|descriptor| label_from_string_view(descriptor.label)),
        ),
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
    descriptor: *const native::WGPUCommandBufferDescriptor,
) -> native::WGPUCommandBuffer {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let (command_buffer, error) = encoder.core.finish();
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(WGPUCommandBufferImpl {
        core: Arc::new(command_buffer),
        _device: Arc::clone(&encoder.device),
        _instance: Arc::clone(&encoder.instance),
        label: Mutex::new(
            descriptor
                .as_ref()
                .and_then(|descriptor| label_from_string_view(descriptor.label)),
        ),
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
    let source = borrow_handle(source, "WGPUBuffer");
    let destination = borrow_handle(destination, "WGPUBuffer");
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
    let buffer = borrow_handle(buffer, "WGPUBuffer");
    if !buffer.device.same(&encoder.device) {
        dispatch_optional_error(
            &encoder.device,
            encoder
                .core
                .record_validation_error("clear buffer must belong to the command encoder device"),
        );
        return;
    }
    let size = if size == native::WGPU_WHOLE_SIZE {
        buffer.core.size().saturating_sub(offset)
    } else {
        size
    };
    dispatch_optional_error(
        &encoder.device,
        encoder
            .core
            .clear_buffer(Arc::clone(&buffer.core), offset, size),
    );
}

/// Records a host-to-buffer write command (Block 100).
///
/// The `size` bytes at `data` are copied into the command buffer at encode
/// time, so the caller may free or modify `data` as soon as this returns;
/// the write executes at submit, ordered with the other commands of the
/// command buffer. A null `data` with a non-zero `size` is a validation error
/// routed to the device error sink; null with `size == 0` is a validated
/// no-op.
///
/// # Safety
///
/// `command_encoder` and `buffer` must be non-null live yawgpu handles. When
/// `size > 0`, `data` must be null or point to at least `size` readable
/// bytes.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderWriteBuffer(
    command_encoder: native::WGPUCommandEncoder,
    buffer: native::WGPUBuffer,
    buffer_offset: u64,
    data: *const c_void,
    size: usize,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let buffer = borrow_handle(buffer, "WGPUBuffer");
    if u64::try_from(size).is_err() {
        dispatch_optional_error(
            &encoder.device,
            Some("command encoder write buffer size is too large".to_owned()),
        );
        return;
    }
    let data = if size == 0 {
        &[][..]
    } else if data.is_null() {
        dispatch_optional_error(
            &encoder.device,
            Some("command encoder write buffer data must not be null".to_owned()),
        );
        return;
    } else {
        // Safety: the caller guarantees `size` readable bytes at the non-null
        // `data`; core copies them before returning.
        std::slice::from_raw_parts(data.cast::<u8>(), size)
    };
    dispatch_optional_error(
        &encoder.device,
        encoder
            .core
            .write_buffer(Arc::clone(&buffer.core), buffer_offset, data),
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
    let query_set = borrow_handle(query_set, "WGPUQuerySet");
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
    let query_set = borrow_handle(query_set, "WGPUQuerySet");
    let destination = borrow_handle(destination, "WGPUBuffer");
    if !query_set._device.same(&encoder.device) || !destination.device.same(&encoder.device) {
        dispatch_optional_error(
            &encoder.device,
            encoder.core.record_validation_error(
                "query set and destination buffer must belong to the command encoder device",
            ),
        );
        return;
    }
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
    let source_buffer = borrow_handle(source.buffer, "WGPUBuffer");
    let destination_texture = borrow_handle(destination.texture, "WGPUTexture");
    if !destination_texture.device.same(&encoder.device) {
        dispatch_optional_error(
            &encoder.device,
            encoder.core.record_validation_error(
                "copy buffer to texture destination texture must belong to the command encoder device",
            ),
        );
        return;
    }
    let (destination_mip_level, destination_origin, destination_aspect) =
        map_texel_copy_texture_info_parts(destination);

    dispatch_optional_error(
        &encoder.device,
        encoder.core.copy_buffer_to_texture(
            core::TexelCopyBufferInfo {
                buffer: Arc::clone(&source_buffer.core),
                device: Some((*source_buffer.device).clone()),
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
    let source_texture = borrow_handle(source.texture, "WGPUTexture");
    let destination_buffer = borrow_handle(destination.buffer, "WGPUBuffer");
    if !source_texture.device.same(&encoder.device) {
        dispatch_optional_error(
            &encoder.device,
            encoder.core.record_validation_error(
                "copy texture to buffer source texture must belong to the command encoder device",
            ),
        );
        return;
    }
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
                device: Some((*destination_buffer.device).clone()),
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
    let source_texture = borrow_handle(source.texture, "WGPUTexture");
    let destination_texture = borrow_handle(destination.texture, "WGPUTexture");
    if !source_texture.device.same(&encoder.device)
        || !destination_texture.device.same(&encoder.device)
    {
        dispatch_optional_error(
            &encoder.device,
            encoder
                .core
                .record_validation_error("copy textures must belong to the command encoder device"),
        );
        return;
    }
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

fn validate_render_pass_descriptor_devices(
    descriptor: &native::WGPURenderPassDescriptor,
    device: &core::Device,
) -> Option<String> {
    let attachments = if descriptor.colorAttachmentCount == 0 {
        &[][..]
    } else {
        unsafe {
            std::slice::from_raw_parts(descriptor.colorAttachments, descriptor.colorAttachmentCount)
        }
    };
    for attachment in attachments {
        if !attachment.view.is_null() && !unsafe { view_belongs_to_device(attachment.view, device) }
        {
            return Some(
                "render pass color attachment view must belong to the command encoder device"
                    .to_owned(),
            );
        }
        if !attachment.resolveTarget.is_null()
            && !unsafe { view_belongs_to_device(attachment.resolveTarget, device) }
        {
            return Some(
                "render pass resolve target must belong to the command encoder device".to_owned(),
            );
        }
    }
    if let Some(depth_stencil) = unsafe { descriptor.depthStencilAttachment.as_ref() } {
        if !depth_stencil.view.is_null()
            && !unsafe { view_belongs_to_device(depth_stencil.view, device) }
        {
            return Some(
                "render pass depth-stencil attachment view must belong to the command encoder device"
                    .to_owned(),
            );
        }
    }
    if !descriptor.occlusionQuerySet.is_null() {
        let query_set = unsafe {
            borrow_handle::<WGPUQuerySetImpl>(descriptor.occlusionQuerySet, "WGPUQuerySet")
        };
        if !query_set._device.same(device) {
            return Some(
                "render pass occlusion query set must belong to the command encoder device"
                    .to_owned(),
            );
        }
    }
    if let Some(timestamp_writes) = unsafe { descriptor.timestampWrites.as_ref() } {
        let query_set =
            unsafe { borrow_handle::<WGPUQuerySetImpl>(timestamp_writes.querySet, "WGPUQuerySet") };
        if !query_set._device.same(device) {
            return Some(
                "render pass timestamp query set must belong to the command encoder device"
                    .to_owned(),
            );
        }
    }
    None
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use super::*;
    use std::collections::BTreeMap;

    fn device_impl() -> Arc<WGPUDeviceImpl> {
        device_impl_with_features(&[])
    }

    fn device_impl_with_features(features: &[core::Feature]) -> Arc<WGPUDeviceImpl> {
        let instance = Arc::new(WGPUInstanceImpl {
            core: Arc::new(core::Instance::new_noop()),
            timed_wait_any_enabled: false,
            pending_callbacks: Mutex::new(BTreeMap::new()),
        });
        let adapter = instance
            .core
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter");
        let device = adapter
            .create_device(None, features, "device", "queue")
            .expect("Noop device");
        Arc::new(WGPUDeviceImpl {
            core: Arc::new(device),
            instance,
            adapter: Arc::new(adapter),
            device_lost_callback: DeviceLostCallbackInfo {
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: None,
                userdata1: 0,
                userdata2: 0,
            },
            device_lost_futures: Mutex::new(Vec::new()),
            default_queue: Mutex::new(None),
            shader_module_cache: ObjectCache::new(),
            pipeline_layout_cache: ObjectCache::new(),
            compute_pipeline_cache: ObjectCache::new(),
            render_pipeline_cache: ObjectCache::new(),
        })
    }

    #[test]
    fn wgpuCommandEncoderBeginComputePass_with_timestamp_writes_records_begin_before_and_end_after()
    {
        let device = device_impl_with_features(&[core::Feature::TimestampQuery]);
        let handle = arc_to_handle(device.clone());
        unsafe {
            let descriptor = native::WGPUQuerySetDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: std::mem::zeroed(),
                type_: native::WGPUQueryType_Timestamp,
                count: 2,
            };
            let set = wgpuDeviceCreateQuerySet(handle, &descriptor);
            let encoder = wgpuDeviceCreateCommandEncoder(handle, std::ptr::null());
            let writes = native::WGPUPassTimestampWrites {
                nextInChain: std::ptr::null_mut(),
                querySet: set,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: 1,
            };
            let descriptor = native::WGPUComputePassDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: std::mem::zeroed(),
                timestampWrites: &writes,
            };
            let pass = wgpuCommandEncoderBeginComputePass(encoder, &descriptor);
            let shader = Arc::new(device.core.create_shader_module(
                core::ShaderModuleSource::Wgsl(
                    "@compute @workgroup_size(1) fn main() {}".to_owned(),
                ),
            ));
            let pipeline = Arc::new(device.core.create_compute_pipeline(
                core::ComputePipelineDescriptor {
                    layout: core::ComputePipelineLayout::Auto,
                    shader_module: shader,
                    entry_point: None,
                    constants: Vec::new(),
                    error: None,
                },
            ));
            let core_pass = &borrow_handle(pass, "WGPUComputePassEncoder").core;
            assert_eq!(core_pass.set_pipeline(pipeline), None);
            assert_eq!(
                core_pass.dispatch_workgroups(1, 1, 1, device.core.limits()),
                None
            );
            wgpuComputePassEncoderEnd(pass);
            let command = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!borrow_handle(command, "WGPUCommandBuffer").core.is_error());
            let queue = wgpuDeviceGetQueue(handle);
            wgpuQueueSubmit(queue, 1, &command);
            let queue_core = device.core.queue();
            let yawgpu_hal::HalQueue::Noop(hal) = queue_core.hal() else {
                panic!("expected Noop")
            };
            assert!(
                matches!(hal.submitted_copies().as_slice(), [yawgpu_hal::HalCopy::WriteTimestamp(b), yawgpu_hal::HalCopy::ComputePass(_), yawgpu_hal::HalCopy::WriteTimestamp(e)] if b.query_index == 0 && e.query_index == 1)
            );
            wgpuComputePassEncoderRelease(pass);
            wgpuCommandBufferRelease(command);
            wgpuCommandEncoderRelease(encoder);
            wgpuQuerySetRelease(set);
            wgpuQueueRelease(queue);
            wgpuDeviceRelease(handle);
        }
    }

    #[test]
    fn wgpuCommandEncoderBeginComputePass_timestamp_writes_reject_device_mismatch() {
        let device = device_impl_with_features(&[core::Feature::TimestampQuery]);
        let other = device_impl_with_features(&[core::Feature::TimestampQuery]);
        let handle = arc_to_handle(device.clone());
        let other_handle = arc_to_handle(other);
        unsafe {
            let descriptor = native::WGPUQuerySetDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: std::mem::zeroed(),
                type_: native::WGPUQueryType_Timestamp,
                count: 2,
            };
            let set = wgpuDeviceCreateQuerySet(other_handle, &descriptor);
            let encoder = wgpuDeviceCreateCommandEncoder(handle, std::ptr::null());
            let writes = native::WGPUPassTimestampWrites {
                nextInChain: std::ptr::null_mut(),
                querySet: set,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: 1,
            };
            let descriptor = native::WGPUComputePassDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: std::mem::zeroed(),
                timestampWrites: &writes,
            };
            let pass = wgpuCommandEncoderBeginComputePass(encoder, &descriptor);
            wgpuComputePassEncoderEnd(pass);
            let (_, error) = borrow_handle(encoder, "WGPUCommandEncoder").core.finish();
            assert_eq!(
                error.as_deref(),
                Some("compute pass timestamp query set must belong to the command encoder device")
            );
            wgpuComputePassEncoderRelease(pass);
            wgpuCommandEncoderRelease(encoder);
            wgpuQuerySetRelease(set);
            wgpuDeviceRelease(other_handle);
            wgpuDeviceRelease(handle);
        }
    }

    unsafe fn copy_dst_buffer(device: native::WGPUDevice, size: u64) -> native::WGPUBuffer {
        let descriptor = native::WGPUBufferDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: native::WGPUStringView {
                data: std::ptr::null(),
                length: 0,
            },
            usage: native::WGPUBufferUsage_CopyDst,
            size,
            mappedAtCreation: 0,
        };
        let buffer = wgpuDeviceCreateBuffer(device, &descriptor);
        assert!(!buffer.is_null());
        buffer
    }

    /// Block 100 R2: null `data` with a non-zero `size` is reported through
    /// the device error sink without dereferencing, and nothing is recorded
    /// on the encoder (finish still succeeds).
    #[test]
    fn wgpuCommandEncoderWriteBuffer_rejects_null_data_with_nonzero_size() {
        let device = device_impl();
        let device_handle = arc_to_handle(Arc::clone(&device));
        unsafe {
            let buffer = copy_dst_buffer(device_handle, 16);
            let encoder = wgpuDeviceCreateCommandEncoder(device_handle, std::ptr::null());

            wgpuDevicePushErrorScope(device_handle, native::WGPUErrorFilter_Validation);
            wgpuCommandEncoderWriteBuffer(encoder, buffer, 0, std::ptr::null(), 4);
            let error = device
                .core
                .pop_error_scope()
                .expect("scope")
                .expect("null data must be a validation error");
            assert_eq!(error.kind, core::ErrorKind::Validation);
            assert_eq!(
                error.message,
                "command encoder write buffer data must not be null"
            );

            // The rejected call recorded nothing: the encoder finishes clean.
            wgpuDevicePushErrorScope(device_handle, native::WGPUErrorFilter_Validation);
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());
            assert_eq!(device.core.pop_error_scope().expect("scope"), None);

            wgpuCommandBufferRelease(command_buffer);
            wgpuCommandEncoderRelease(encoder);
            wgpuBufferRelease(buffer);
            wgpuDeviceRelease(device_handle);
        }
    }

    /// Block 100 R2: null `data` with `size == 0` is a validated no-op with
    /// no error, and a non-null write still validates its range.
    #[test]
    fn wgpuCommandEncoderWriteBuffer_accepts_null_data_with_zero_size() {
        let device = device_impl();
        let device_handle = arc_to_handle(Arc::clone(&device));
        unsafe {
            let buffer = copy_dst_buffer(device_handle, 16);
            let encoder = wgpuDeviceCreateCommandEncoder(device_handle, std::ptr::null());

            wgpuDevicePushErrorScope(device_handle, native::WGPUErrorFilter_Validation);
            wgpuCommandEncoderWriteBuffer(encoder, buffer, 8, std::ptr::null(), 0);
            let bytes = [1_u8, 2, 3, 4];
            wgpuCommandEncoderWriteBuffer(encoder, buffer, 4, bytes.as_ptr().cast(), bytes.len());
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());
            assert_eq!(device.core.pop_error_scope().expect("scope"), None);

            // Zero size past the end is still range-validated (encoder error
            // surfaces at finish).
            let invalid = wgpuDeviceCreateCommandEncoder(device_handle, std::ptr::null());
            wgpuCommandEncoderWriteBuffer(invalid, buffer, 20, std::ptr::null(), 0);
            wgpuDevicePushErrorScope(device_handle, native::WGPUErrorFilter_Validation);
            let invalid_buffer = wgpuCommandEncoderFinish(invalid, std::ptr::null());
            let error = device
                .core
                .pop_error_scope()
                .expect("scope")
                .expect("out-of-range zero write must fail at finish");
            assert!(
                error
                    .message
                    .contains("command encoder write buffer range exceeds buffer size"),
                "{}",
                error.message
            );

            wgpuCommandBufferRelease(invalid_buffer);
            wgpuCommandEncoderRelease(invalid);
            wgpuCommandBufferRelease(command_buffer);
            wgpuCommandEncoderRelease(encoder);
            wgpuBufferRelease(buffer);
            wgpuDeviceRelease(device_handle);
        }
    }
}
