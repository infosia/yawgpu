use super::*;

/// Sets the render pipeline for a render bundle encoder.
///
/// # Safety
///
/// `render_bundle_encoder` and `pipeline` must be non-null live yawgpu handles.
/// Returns WGPU render bundle encoder set pipeline.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetPipeline(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    pipeline: native::WGPURenderPipeline,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let pipeline = clone_handle(pipeline, "WGPURenderPipeline");
    if !pipeline._device.same(&encoder.device) {
        dispatch_optional_error(
            &encoder.device,
            encoder.core.record_validation_error(
                "render pipeline must belong to the render bundle encoder device",
            ),
        );
        return;
    }
    dispatch_optional_error(
        &encoder.device,
        encoder.core.set_pipeline(Arc::clone(&pipeline._core)),
    );
}

/// Sets or clears a render bundle bind group.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// `group` may be null. `dynamic_offsets` must point to `dynamic_offset_count`
/// elements when the count is non-zero.
/// Returns WGPU render bundle encoder set bind group.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetBindGroup(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    group_index: u32,
    group: native::WGPUBindGroup,
    dynamic_offset_count: usize,
    dynamic_offsets: *const u32,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let group =
        (!group.is_null()).then(|| clone_handle::<WGPUBindGroupImpl>(group, "WGPUBindGroup"));
    if let Some(group) = group.as_ref() {
        if !group._device.same(&encoder.device) {
            dispatch_optional_error(
                &encoder.device,
                encoder.core.record_validation_error(
                    "bind group must belong to the render bundle encoder device",
                ),
            );
            return;
        }
    }
    let offsets = dynamic_offsets_slice(dynamic_offset_count, dynamic_offsets);
    dispatch_optional_error(
        &encoder.device,
        encoder.core.set_bind_group(
            group_index,
            group.map(|group| Arc::clone(&group._core)),
            offsets,
            encoder.device.limits(),
        ),
    );
}

/// Overwrites part of the render bundle's own user-immediates scratch.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// `data` must point to `size` bytes when `size` is non-zero (mirrors
/// `wgpuQueueWriteBuffer`'s null/size contract).
/// Returns WGPU render bundle encoder set immediates.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetImmediates(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    offset: u32,
    data: *const c_void,
    size: usize,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    if size > 0 && data.is_null() {
        encoder.device.dispatch_error(
            core::ErrorKind::Validation,
            "render bundle set immediates data must not be null when size is non-zero",
        );
        return;
    }
    let data = std::slice::from_raw_parts(data.cast::<u8>(), size);
    dispatch_optional_error(
        &encoder.device,
        encoder
            .core
            .set_immediates(offset, data, encoder.device.limits()),
    );
}

/// Sets or clears a render bundle vertex buffer.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// `buffer` may be null.
/// Returns WGPU render bundle encoder set vertex buffer.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetVertexBuffer(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    slot: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let buffer = (!buffer.is_null()).then(|| clone_handle::<WGPUBufferImpl>(buffer, "WGPUBuffer"));
    if let Some(buffer) = buffer.as_ref() {
        if !buffer.device.same(&encoder.device) {
            dispatch_optional_error(
                &encoder.device,
                encoder.core.record_validation_error(
                    "vertex buffer must belong to the render bundle encoder device",
                ),
            );
            return;
        }
    }
    dispatch_optional_error(
        &encoder.device,
        encoder.core.set_vertex_buffer(
            slot,
            buffer.map(|buffer| Arc::clone(&buffer.core)),
            offset,
            size,
            encoder.device.limits(),
        ),
    );
}

/// Sets a render bundle index buffer.
///
/// # Safety
///
/// `render_bundle_encoder` and `buffer` must be non-null live yawgpu handles.
/// Returns WGPU render bundle encoder set index buffer.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetIndexBuffer(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    buffer: native::WGPUBuffer,
    format: native::WGPUIndexFormat,
    offset: u64,
    size: u64,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let buffer = clone_handle::<WGPUBufferImpl>(buffer, "WGPUBuffer");
    if !buffer.device.same(&encoder.device) {
        dispatch_optional_error(
            &encoder.device,
            encoder.core.record_validation_error(
                "index buffer must belong to the render bundle encoder device",
            ),
        );
        return;
    }
    dispatch_optional_error(
        &encoder.device,
        encoder.core.set_index_buffer(
            Arc::clone(&buffer.core),
            map_index_format(format),
            offset,
            size,
        ),
    );
}

/// Records a non-indexed draw in a render bundle.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// Returns WGPU render bundle encoder draw.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDraw(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    dispatch_optional_error(
        &encoder.device,
        encoder.core.draw(
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
            encoder.device.limits(),
        ),
    );
}

/// Records an indexed draw in a render bundle.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// Returns WGPU render bundle encoder draw indexed.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDrawIndexed(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    dispatch_optional_error(
        &encoder.device,
        encoder.core.draw_indexed(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
            encoder.device.limits(),
        ),
    );
}

/// Records an indirect non-indexed draw in a render bundle.
///
/// # Safety
///
/// `render_bundle_encoder` and `indirect_buffer` must be non-null live yawgpu handles.
/// Returns WGPU render bundle encoder draw indirect.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDrawIndirect(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let indirect_buffer = clone_handle::<WGPUBufferImpl>(indirect_buffer, "WGPUBuffer");
    if !indirect_buffer.device.same(&encoder.device) {
        dispatch_optional_error(
            &encoder.device,
            encoder.core.record_validation_error(
                "indirect buffer must belong to the render bundle encoder device",
            ),
        );
        return;
    }
    dispatch_optional_error(
        &encoder.device,
        encoder.core.draw_indirect(
            Arc::clone(&indirect_buffer.core),
            indirect_offset,
            encoder.device.limits(),
        ),
    );
}

/// Records an indirect indexed draw in a render bundle.
///
/// # Safety
///
/// `render_bundle_encoder` and `indirect_buffer` must be non-null live yawgpu handles.
/// Returns WGPU render bundle encoder draw indexed indirect.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDrawIndexedIndirect(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let indirect_buffer = clone_handle::<WGPUBufferImpl>(indirect_buffer, "WGPUBuffer");
    if !indirect_buffer.device.same(&encoder.device) {
        dispatch_optional_error(
            &encoder.device,
            encoder.core.record_validation_error(
                "indirect buffer must belong to the render bundle encoder device",
            ),
        );
        return;
    }
    dispatch_optional_error(
        &encoder.device,
        encoder.core.draw_indexed_indirect(
            Arc::clone(&indirect_buffer.core),
            indirect_offset,
            encoder.device.limits(),
        ),
    );
}

/// Inserts a render bundle debug marker.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// Returns WGPU render bundle encoder insert debug marker.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderInsertDebugMarker(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    _marker_label: native::WGPUStringView,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.insert_debug_marker());
}

/// Pushes a render bundle debug group.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// Returns WGPU render bundle encoder push debug group.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderPushDebugGroup(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    _group_label: native::WGPUStringView,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.push_debug_group());
}

/// Pops a render bundle debug group.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// Returns WGPU render bundle encoder pop debug group.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderPopDebugGroup(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.pop_debug_group());
}

/// Finishes a render bundle encoder.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// Returns WGPU render bundle encoder finish.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderFinish(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    _descriptor: *const native::WGPURenderBundleDescriptor,
) -> native::WGPURenderBundle {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let (bundle, error) = encoder.core.finish();
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(WGPURenderBundleImpl {
        core: Arc::new(bundle),
        _device: Arc::clone(&encoder.device),
        _instance: Arc::clone(&encoder._instance),
    }))
}

/// Releases one owned reference to a render bundle encoder handle.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// Returns WGPU render bundle encoder release.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderRelease(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
) {
    release_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
}

/// Adds one owned reference to a render bundle encoder handle.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// Returns WGPU render bundle encoder add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderAddRef(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
) {
    add_ref_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
}

/// Releases one owned reference to a render bundle handle.
///
/// # Safety
///
/// `render_bundle` must be a non-null live yawgpu render bundle handle.
/// Returns WGPU render bundle release.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleRelease(render_bundle: native::WGPURenderBundle) {
    release_handle(render_bundle, "WGPURenderBundle");
}

/// Adds one owned reference to a render bundle handle.
///
/// # Safety
///
/// `render_bundle` must be a non-null live yawgpu render bundle handle.
/// Returns WGPU render bundle add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleAddRef(render_bundle: native::WGPURenderBundle) {
    add_ref_handle(render_bundle, "WGPURenderBundle");
}
