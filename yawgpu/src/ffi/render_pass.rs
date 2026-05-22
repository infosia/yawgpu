use super::*;

/// Ends a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder end.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderEnd(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.end());
}

/// Advances a subpass render pass to the next subpass.
///
/// # Safety
///
/// `encoder` must be a non-null live yawgpu subpass render pass encoder.
/// Returns yawgpu subpass render pass encoder next subpass.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderNextSubpass(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.next_subpass());
}

/// Ends a subpass render pass.
///
/// # Safety
///
/// `encoder` must be a non-null live yawgpu subpass render pass encoder.
/// Returns yawgpu subpass render pass encoder end.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderEnd(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.end());
}

/// Sets a subpass render pipeline.
///
/// # Safety
///
/// `encoder` and `pipeline` must be non-null live yawgpu handles.
/// Returns yawgpu subpass render pass encoder set pipeline.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderSetPipeline(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
    pipeline: native::WGPURenderPipeline,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    let pipeline = clone_handle(pipeline, "WGPURenderPipeline");
    if !pipeline._device.same(&pass.device) {
        dispatch_optional_error(
            &pass.device,
            pass.core.record_validation_error(
                "render pipeline must belong to the subpass render pass device",
            ),
        );
        return;
    }
    dispatch_optional_error(
        &pass.device,
        pass.core.set_pipeline(Arc::clone(&pipeline._core)),
    );
}

/// Sets or clears a subpass render pass bind group.
///
/// # Safety
///
/// `encoder` must be non-null. `group` may be null to clear the slot.
/// `dynamic_offsets` must point to `dynamic_offset_count` elements when count is
/// non-zero.
/// Returns yawgpu subpass render pass encoder set bind group.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderSetBindGroup(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
    group_index: u32,
    group: native::WGPUBindGroup,
    dynamic_offset_count: usize,
    dynamic_offsets: *const u32,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    let group =
        (!group.is_null()).then(|| clone_handle::<WGPUBindGroupImpl>(group, "WGPUBindGroup"));
    if let Some(group) = group.as_ref() {
        if !group._device.same(&pass.device) {
            dispatch_optional_error(
                &pass.device,
                pass.core.record_validation_error(
                    "bind group must belong to the subpass render pass device",
                ),
            );
            return;
        }
    }
    let offsets = dynamic_offsets_slice(dynamic_offset_count, dynamic_offsets);
    dispatch_optional_error(
        &pass.device,
        pass.core.set_bind_group(
            group_index,
            group.map(|group| Arc::clone(&group._core)),
            offsets,
        ),
    );
}

/// Sets or clears a subpass render pass vertex buffer.
///
/// # Safety
///
/// `encoder` must be non-null. `buffer` may be null to clear the slot.
/// Returns yawgpu subpass render pass encoder set vertex buffer.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderSetVertexBuffer(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
    slot: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    let buffer = (!buffer.is_null()).then(|| clone_handle::<WGPUBufferImpl>(buffer, "WGPUBuffer"));
    if let Some(buffer) = buffer.as_ref() {
        if !buffer.device.same(&pass.device) {
            dispatch_optional_error(
                &pass.device,
                pass.core.record_validation_error(
                    "vertex buffer must belong to the subpass render pass device",
                ),
            );
            return;
        }
    }
    dispatch_optional_error(
        &pass.device,
        pass.core.set_vertex_buffer(
            slot,
            buffer.map(|buffer| Arc::clone(&buffer.core)),
            offset,
            size,
            pass.device.limits(),
        ),
    );
}

/// Sets a subpass render pass index buffer.
///
/// # Safety
///
/// `encoder` and `buffer` must be non-null live yawgpu handles.
/// Returns yawgpu subpass render pass encoder set index buffer.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderSetIndexBuffer(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
    buffer: native::WGPUBuffer,
    format: native::WGPUIndexFormat,
    offset: u64,
    size: u64,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    let buffer = clone_handle::<WGPUBufferImpl>(buffer, "WGPUBuffer");
    if !buffer.device.same(&pass.device) {
        dispatch_optional_error(
            &pass.device,
            pass.core.record_validation_error(
                "index buffer must belong to the subpass render pass device",
            ),
        );
        return;
    }
    dispatch_optional_error(
        &pass.device,
        pass.core.set_index_buffer(
            Arc::clone(&buffer.core),
            map_index_format(format),
            offset,
            size,
        ),
    );
}

/// Records a subpass non-indexed draw.
///
/// # Safety
///
/// `encoder` must be a non-null live yawgpu subpass render pass encoder.
/// Returns yawgpu subpass render pass encoder draw.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderDraw(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core.draw(
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
            pass.device.limits(),
        ),
    );
}

/// Records a subpass indexed draw.
///
/// # Safety
///
/// `encoder` must be a non-null live yawgpu subpass render pass encoder.
/// Returns yawgpu subpass render pass encoder draw indexed.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderDrawIndexed(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core.draw_indexed(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
            pass.device.limits(),
        ),
    );
}

/// Records a programmable transient tile dispatch.
///
/// # Safety
///
/// `encoder` and `descriptor` must be non-null live yawgpu pointers.
/// Returns yawgpu subpass render pass encoder dispatch transient.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderDispatchTransient(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
    descriptor: *const YaWGPUTransientDispatchDescriptor,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    let descriptor = descriptor
        .as_ref()
        .expect("YaWGPUTransientDispatchDescriptor must not be null");
    let descriptor = map_transient_dispatch_descriptor(descriptor);
    dispatch_optional_error(&pass.device, pass.core.dispatch_transient(descriptor));
}

/// Sets the subpass viewport.
///
/// # Safety
///
/// `encoder` must be a non-null live yawgpu subpass render pass encoder.
/// Returns yawgpu subpass render pass encoder set viewport.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderSetViewport(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core
            .set_viewport(x, y, width, height, min_depth, max_depth),
    );
}

/// Sets the subpass scissor rectangle.
///
/// # Safety
///
/// `encoder` must be a non-null live yawgpu subpass render pass encoder.
/// Returns yawgpu subpass render pass encoder set scissor rect.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderSetScissorRect(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core.set_scissor_rect(x, y, width, height),
    );
}

/// Adds one owned reference to a subpass render pass encoder.
///
/// # Safety
///
/// `encoder` must be a non-null live yawgpu subpass render pass encoder.
/// Returns yawgpu subpass render pass encoder add ref.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderAddRef(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
) {
    add_ref_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
}

/// Releases one owned reference to a subpass render pass encoder.
///
/// # Safety
///
/// `encoder` must be a non-null live yawgpu subpass render pass encoder.
/// Returns yawgpu subpass render pass encoder release.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderRelease(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
) {
    release_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
}

/// Adds one owned reference to a subpass pass layout.
///
/// # Safety
///
/// `layout` must be a non-null live yawgpu subpass pass layout.
/// Returns yawgpu subpass pass layout add ref.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassPassLayoutAddRef(layout: crate::YaWGPUSubpassPassLayout) {
    add_ref_handle(layout, "YaWGPUSubpassPassLayout");
}

/// Releases one owned reference to a subpass pass layout.
///
/// # Safety
///
/// `layout` must be a non-null live yawgpu subpass pass layout.
/// Returns yawgpu subpass pass layout release.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassPassLayoutRelease(layout: crate::YaWGPUSubpassPassLayout) {
    release_handle(layout, "YaWGPUSubpassPassLayout");
}

/// Begins an occlusion query in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder begin occlusion query.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderBeginOcclusionQuery(
    render_pass_encoder: native::WGPURenderPassEncoder,
    query_index: u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.begin_occlusion_query(query_index));
}

/// Ends the current occlusion query in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder end occlusion query.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderEndOcclusionQuery(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.end_occlusion_query());
}

/// Inserts a render pass debug marker.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder insert debug marker.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderInsertDebugMarker(
    render_pass_encoder: native::WGPURenderPassEncoder,
    _marker_label: native::WGPUStringView,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.insert_debug_marker());
}

/// Pushes a render pass debug group.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder push debug group.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderPushDebugGroup(
    render_pass_encoder: native::WGPURenderPassEncoder,
    _group_label: native::WGPUStringView,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.push_debug_group());
}

/// Pops a render pass debug group.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder pop debug group.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderPopDebugGroup(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.pop_debug_group());
}

/// Sets the render pipeline for a render pass.
///
/// # Safety
///
/// `render_pass_encoder` and `pipeline` must be non-null live yawgpu handles.
/// Returns WGPU render pass encoder set pipeline.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetPipeline(
    render_pass_encoder: native::WGPURenderPassEncoder,
    pipeline: native::WGPURenderPipeline,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let pipeline = clone_handle(pipeline, "WGPURenderPipeline");
    if !pipeline._device.same(&pass.device) {
        dispatch_optional_error(
            &pass.device,
            pass.core
                .record_validation_error("render pipeline must belong to the render pass device"),
        );
        return;
    }
    dispatch_optional_error(
        &pass.device,
        pass.core.set_pipeline(Arc::clone(&pipeline._core)),
    );
}

/// Sets or clears a render pass bind group.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// `group` may be null to clear the slot. `dynamic_offsets` must point to
/// `dynamic_offset_count` elements when the count is non-zero.
/// Returns WGPU render pass encoder set bind group.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetBindGroup(
    render_pass_encoder: native::WGPURenderPassEncoder,
    group_index: u32,
    group: native::WGPUBindGroup,
    dynamic_offset_count: usize,
    dynamic_offsets: *const u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let group =
        (!group.is_null()).then(|| clone_handle::<WGPUBindGroupImpl>(group, "WGPUBindGroup"));
    if let Some(group) = group.as_ref() {
        if !group._device.same(&pass.device) {
            dispatch_optional_error(
                &pass.device,
                pass.core
                    .record_validation_error("bind group must belong to the render pass device"),
            );
            return;
        }
    }
    let offsets = dynamic_offsets_slice(dynamic_offset_count, dynamic_offsets);
    dispatch_optional_error(
        &pass.device,
        pass.core.set_bind_group(
            group_index,
            group.map(|group| Arc::clone(&group._core)),
            offsets,
        ),
    );
}

/// Sets or clears a render pass vertex buffer.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// `buffer` may be null to clear the slot.
/// Returns WGPU render pass encoder set vertex buffer.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetVertexBuffer(
    render_pass_encoder: native::WGPURenderPassEncoder,
    slot: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let buffer = (!buffer.is_null()).then(|| clone_handle::<WGPUBufferImpl>(buffer, "WGPUBuffer"));
    if let Some(buffer) = buffer.as_ref() {
        if !buffer.device.same(&pass.device) {
            dispatch_optional_error(
                &pass.device,
                pass.core
                    .record_validation_error("vertex buffer must belong to the render pass device"),
            );
            return;
        }
    }
    dispatch_optional_error(
        &pass.device,
        pass.core.set_vertex_buffer(
            slot,
            buffer.map(|buffer| Arc::clone(&buffer.core)),
            offset,
            size,
            pass.device.limits(),
        ),
    );
}

/// Sets the render pass index buffer.
///
/// # Safety
///
/// `render_pass_encoder` and `buffer` must be non-null live yawgpu handles.
/// Returns WGPU render pass encoder set index buffer.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetIndexBuffer(
    render_pass_encoder: native::WGPURenderPassEncoder,
    buffer: native::WGPUBuffer,
    format: native::WGPUIndexFormat,
    offset: u64,
    size: u64,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let buffer = clone_handle::<WGPUBufferImpl>(buffer, "WGPUBuffer");
    if !buffer.device.same(&pass.device) {
        dispatch_optional_error(
            &pass.device,
            pass.core
                .record_validation_error("index buffer must belong to the render pass device"),
        );
        return;
    }
    dispatch_optional_error(
        &pass.device,
        pass.core.set_index_buffer(
            Arc::clone(&buffer.core),
            map_index_format(format),
            offset,
            size,
        ),
    );
}

/// Records a non-indexed draw in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder draw.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDraw(
    render_pass_encoder: native::WGPURenderPassEncoder,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core.draw(
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
            pass.device.limits(),
        ),
    );
}

/// Records an indexed draw in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder draw indexed.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDrawIndexed(
    render_pass_encoder: native::WGPURenderPassEncoder,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core.draw_indexed(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
            pass.device.limits(),
        ),
    );
}

/// Records an indirect non-indexed draw in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` and `indirect_buffer` must be non-null live yawgpu handles.
/// Returns WGPU render pass encoder draw indirect.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDrawIndirect(
    render_pass_encoder: native::WGPURenderPassEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let indirect_buffer = clone_handle::<WGPUBufferImpl>(indirect_buffer, "WGPUBuffer");
    dispatch_optional_error(
        &pass.device,
        pass.core.draw_indirect(
            Arc::clone(&indirect_buffer.core),
            indirect_offset,
            pass.device.limits(),
        ),
    );
}

/// Records an indirect indexed draw in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` and `indirect_buffer` must be non-null live yawgpu handles.
/// Returns WGPU render pass encoder draw indexed indirect.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDrawIndexedIndirect(
    render_pass_encoder: native::WGPURenderPassEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let indirect_buffer = clone_handle::<WGPUBufferImpl>(indirect_buffer, "WGPUBuffer");
    dispatch_optional_error(
        &pass.device,
        pass.core.draw_indexed_indirect(
            Arc::clone(&indirect_buffer.core),
            indirect_offset,
            pass.device.limits(),
        ),
    );
}

/// Sets the render pass viewport.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder set viewport.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetViewport(
    render_pass_encoder: native::WGPURenderPassEncoder,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core
            .set_viewport(x, y, width, height, min_depth, max_depth),
    );
}

/// Sets the render pass scissor rectangle.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder set scissor rect.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetScissorRect(
    render_pass_encoder: native::WGPURenderPassEncoder,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core.set_scissor_rect(x, y, width, height),
    );
}

/// Sets the render pass blend constant.
///
/// # Safety
///
/// `render_pass_encoder` and `color` must be non-null live pointers.
/// Returns WGPU render pass encoder set blend constant.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetBlendConstant(
    render_pass_encoder: native::WGPURenderPassEncoder,
    color: *const native::WGPUColor,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let color = color
        .as_ref()
        .expect("WGPUColor for SetBlendConstant must not be null");
    dispatch_optional_error(
        &pass.device,
        pass.core.set_blend_constant(map_color(*color)),
    );
}

/// Sets the render pass stencil reference.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder set stencil reference.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetStencilReference(
    render_pass_encoder: native::WGPURenderPassEncoder,
    reference: u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.set_stencil_reference(reference));
}

/// Executes render bundles in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// `bundles` must point to `bundle_count` live render bundle handles when the
/// count is non-zero.
/// Returns WGPU render pass encoder execute bundles.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderExecuteBundles(
    render_pass_encoder: native::WGPURenderPassEncoder,
    bundle_count: usize,
    bundles: *const native::WGPURenderBundle,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let bundle_handles = render_bundle_slice(bundle_count, bundles);
    if bundle_handles
        .iter()
        .any(|bundle| !bundle._device.same(&pass.device))
    {
        dispatch_optional_error(
            &pass.device,
            pass.core
                .record_validation_error("render bundle must belong to the render pass device"),
        );
        return;
    }
    let bundles = bundle_handles
        .iter()
        .map(|bundle| Arc::clone(&bundle.core))
        .collect::<Vec<_>>();
    dispatch_optional_error(&pass.device, pass.core.execute_bundles(&bundles));
}

/// Releases one owned reference to a render pass encoder handle.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder release.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderRelease(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    release_handle(render_pass_encoder, "WGPURenderPassEncoder");
}

/// Adds one owned reference to a render pass encoder handle.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// Returns WGPU render pass encoder add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderAddRef(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    add_ref_handle(render_pass_encoder, "WGPURenderPassEncoder");
}
