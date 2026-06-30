use super::*;
use crate::{
    YaWGPUSubpassPassLayoutDescriptor, YaWGPUSubpassRenderPassDescriptor,
    YaWGPUSubpassRenderPipelineDescriptor, YaWGPUTiledCapabilities,
};

/// Gets yawgpu tiled rendering capabilities for an adapter.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle. `capabilities`
/// must point to writable `YaWGPUTiledCapabilities` storage.
/// Returns yawgpu adapter get tiled capabilities.
#[no_mangle]
pub unsafe extern "C" fn yawgpuAdapterGetTiledCapabilities(
    adapter: native::WGPUAdapter,
    capabilities: *mut YaWGPUTiledCapabilities,
) -> native::WGPUStatus {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    let Some(capabilities) = capabilities.as_mut() else {
        return native::WGPUStatus_Error;
    };
    let next_in_chain = capabilities.nextInChain;
    let tiled = adapter.core.tiled_capabilities();
    *capabilities = YaWGPUTiledCapabilities {
        nextInChain: next_in_chain,
        maxSubpasses: tiled.max_subpasses,
        maxSubpassColorAttachments: tiled.max_subpass_color_attachments,
        maxInputAttachments: tiled.max_input_attachments,
        estimatedTileMemoryBytes: tiled.estimated_tile_memory_bytes,
    };
    native::WGPUStatus_Success
}

/// Creates a subpass pass layout on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `YaWGPUSubpassPassLayoutDescriptor`.
/// Returns yawgpu device create subpass pass layout.
#[no_mangle]
pub unsafe extern "C" fn yawgpuDeviceCreateSubpassPassLayout(
    device: native::WGPUDevice,
    descriptor: *const YaWGPUSubpassPassLayoutDescriptor,
) -> crate::YaWGPUSubpassPassLayout {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("YaWGPUSubpassPassLayoutDescriptor must not be null");
    let layout = device
        .core
        .create_subpass_pass_layout(map_subpass_pass_layout_descriptor(descriptor));
    arc_to_handle(Arc::new(YaWGPUSubpassPassLayoutImpl {
        _core: Arc::new(layout),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Adds one owned reference to a subpass pass layout handle.
///
/// # Safety
///
/// `layout` must be a non-null live yawgpu subpass pass layout handle.
/// Returns yawgpu subpass pass layout add ref.
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassPassLayoutAddRef(layout: crate::YaWGPUSubpassPassLayout) {
    add_ref_handle(layout, "YaWGPUSubpassPassLayout");
}

/// Releases one owned reference to a subpass pass layout handle.
///
/// # Safety
///
/// `layout` must be a non-null live yawgpu subpass pass layout handle.
/// Returns yawgpu subpass pass layout release.
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassPassLayoutRelease(layout: crate::YaWGPUSubpassPassLayout) {
    release_handle(layout, "YaWGPUSubpassPassLayout");
}

/// Creates a subpass-compatible render pipeline on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `YaWGPUSubpassRenderPipelineDescriptor`; its base
/// descriptor follows the `WGPURenderPipelineDescriptor` pointer contract.
/// Returns yawgpu device create subpass render pipeline.
#[no_mangle]
pub unsafe extern "C" fn yawgpuDeviceCreateSubpassRenderPipeline(
    device: native::WGPUDevice,
    descriptor: *const YaWGPUSubpassRenderPipelineDescriptor,
) -> native::WGPURenderPipeline {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("YaWGPUSubpassRenderPipelineDescriptor must not be null");
    let device_error = validate_subpass_render_pipeline_devices(device, descriptor);
    let mut descriptor = map_subpass_render_pipeline_descriptor(descriptor);
    if descriptor.error.is_none() {
        descriptor.error = device_error;
    }
    let pipeline = device.core.create_subpass_render_pipeline(descriptor);
    arc_to_handle(Arc::new(WGPURenderPipelineImpl {
        _core: Arc::new(pipeline),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
        bind_group_layout_handles: Mutex::new(Vec::new()),
    }))
}

unsafe fn validate_subpass_render_pipeline_devices(
    device: &WGPUDeviceImpl,
    descriptor: &YaWGPUSubpassRenderPipelineDescriptor,
) -> Option<String> {
    let pass_layout = clone_handle::<YaWGPUSubpassPassLayoutImpl>(
        descriptor.passLayout,
        "YaWGPUSubpassPassLayout",
    );
    if !pass_layout._device.same(&device.core) {
        return Some("subpass render pipeline pass layout must belong to the same device".into());
    }
    validate_render_pipeline_devices(device, &descriptor.base)
}

/// Begins a subpass render pass.
///
/// # Safety
///
/// `command_encoder` and `descriptor` must be non-null live yawgpu handles.
/// Returns yawgpu command encoder begin subpass render pass.
#[no_mangle]
pub unsafe extern "C" fn yawgpuCommandEncoderBeginSubpassRenderPass(
    command_encoder: native::WGPUCommandEncoder,
    descriptor: *const YaWGPUSubpassRenderPassDescriptor,
) -> crate::YaWGPUSubpassRenderPassEncoder {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let native_descriptor = descriptor
        .as_ref()
        .expect("YaWGPUSubpassRenderPassDescriptor must not be null");
    let mut descriptor = map_subpass_render_pass_descriptor(native_descriptor);
    if descriptor.error.is_none() {
        descriptor.error =
            validate_subpass_render_pass_descriptor_devices(native_descriptor, &encoder.device);
    }
    let (pass, error) = encoder
        .core
        .begin_subpass_render_pass(&encoder.device, descriptor);
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(YaWGPUSubpassRenderPassEncoderImpl {
        core: Arc::new(pass),
        device: Arc::clone(&encoder.device),
        _parent: Arc::clone(&encoder.core),
        _instance: Arc::clone(&encoder.instance),
    }))
}

/// Advances a subpass render pass to the next subpass.
///
/// # Safety
///
/// `encoder` must be a non-null live yawgpu subpass render pass encoder.
/// Returns yawgpu subpass render pass encoder next subpass.
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
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderSetPipeline(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
    pipeline: native::WGPURenderPipeline,
) {
    let pass = borrow_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
    let pipeline = clone_handle::<WGPURenderPipelineImpl>(pipeline, "WGPURenderPipeline");
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
/// non-zero. Returns yawgpu subpass render pass encoder set bind group.
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
            pass.device.limits(),
        ),
    );
}

/// Sets or clears a subpass render pass vertex buffer.
///
/// # Safety
///
/// `encoder` must be non-null. `buffer` may be null to clear the slot.
/// Returns yawgpu subpass render pass encoder set vertex buffer.
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

/// Sets the subpass viewport.
///
/// # Safety
///
/// `encoder` must be a non-null live yawgpu subpass render pass encoder.
/// Returns yawgpu subpass render pass encoder set viewport.
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
#[no_mangle]
pub unsafe extern "C" fn yawgpuSubpassRenderPassEncoderRelease(
    encoder: crate::YaWGPUSubpassRenderPassEncoder,
) {
    release_handle(encoder, "YaWGPUSubpassRenderPassEncoder");
}

unsafe fn validate_subpass_render_pass_descriptor_devices(
    descriptor: &YaWGPUSubpassRenderPassDescriptor,
    device: &core::Device,
) -> Option<String> {
    let pass_layout = clone_handle::<YaWGPUSubpassPassLayoutImpl>(
        descriptor.passLayout,
        "YaWGPUSubpassPassLayout",
    );
    if !pass_layout._device.same(device) {
        return Some("subpass render pass layout must belong to the command encoder device".into());
    }
    let attachments = if descriptor.colorAttachmentCount == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(descriptor.colorAttachments, descriptor.colorAttachmentCount)
    };
    for attachment in attachments {
        if !attachment.view.is_null() {
            let view = clone_handle::<WGPUTextureViewImpl>(attachment.view, "WGPUTextureView");
            if !view._device.same(device) {
                return Some(
                    "subpass color attachment view must belong to the command encoder device"
                        .into(),
                );
            }
        }
        if !attachment.resolveTarget.is_null() {
            let target =
                clone_handle::<WGPUTextureViewImpl>(attachment.resolveTarget, "WGPUTextureView");
            if !target._device.same(device) {
                return Some(
                    "subpass resolve target must belong to the command encoder device".into(),
                );
            }
        }
    }
    if let Some(depth_stencil) = descriptor.depthStencilAttachment.as_ref() {
        if !depth_stencil.view.is_null() {
            let view = clone_handle::<WGPUTextureViewImpl>(depth_stencil.view, "WGPUTextureView");
            if !view._device.same(device) {
                return Some(
                    "subpass depth-stencil attachment view must belong to the command encoder device"
                        .into(),
                );
            }
        }
    }
    None
}
