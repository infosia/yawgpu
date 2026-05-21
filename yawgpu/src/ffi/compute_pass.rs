use super::*;

/// Ends a compute pass.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderEnd(
    compute_pass_encoder: native::WGPUComputePassEncoder,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    dispatch_optional_error(&pass.device, pass.core.end());
}

/// Inserts a compute pass debug marker.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderInsertDebugMarker(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    _marker_label: native::WGPUStringView,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    dispatch_optional_error(&pass.device, pass.core.insert_debug_marker());
}

/// Pushes a compute pass debug group.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderPushDebugGroup(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    _group_label: native::WGPUStringView,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    dispatch_optional_error(&pass.device, pass.core.push_debug_group());
}

/// Pops a compute pass debug group.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderPopDebugGroup(
    compute_pass_encoder: native::WGPUComputePassEncoder,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    dispatch_optional_error(&pass.device, pass.core.pop_debug_group());
}

/// Sets the compute pipeline for a compute pass.
///
/// # Safety
///
/// `compute_pass_encoder` and `pipeline` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderSetPipeline(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    pipeline: native::WGPUComputePipeline,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    let pipeline = clone_handle(pipeline, "WGPUComputePipeline");
    if !pipeline._device.same(&pass.device) {
        dispatch_optional_error(
            &pass.device,
            pass.core
                .record_validation_error("compute pipeline must belong to the compute pass device"),
        );
        return;
    }
    dispatch_optional_error(
        &pass.device,
        pass.core.set_pipeline(Arc::clone(&pipeline._core)),
    );
}

/// Sets or clears a compute pass bind group.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
/// `group` may be null to clear the slot. `dynamic_offsets` must point to
/// `dynamic_offset_count` elements when the count is non-zero.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderSetBindGroup(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    group_index: u32,
    group: native::WGPUBindGroup,
    dynamic_offset_count: usize,
    dynamic_offsets: *const u32,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    let group =
        (!group.is_null()).then(|| clone_handle::<WGPUBindGroupImpl>(group, "WGPUBindGroup"));
    if let Some(group) = group.as_ref() {
        if !group._device.same(&pass.device) {
            dispatch_optional_error(
                &pass.device,
                pass.core
                    .record_validation_error("bind group must belong to the compute pass device"),
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

/// Records a compute dispatch.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderDispatchWorkgroups(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    workgroup_count_x: u32,
    workgroup_count_y: u32,
    workgroup_count_z: u32,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core.dispatch_workgroups(
            workgroup_count_x,
            workgroup_count_y,
            workgroup_count_z,
            pass.device.limits(),
        ),
    );
}

/// Records an indirect compute dispatch.
///
/// # Safety
///
/// `compute_pass_encoder` and `indirect_buffer` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderDispatchWorkgroupsIndirect(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    let indirect_buffer = clone_handle::<WGPUBufferImpl>(indirect_buffer, "WGPUBuffer");
    dispatch_optional_error(
        &pass.device,
        pass.core.dispatch_workgroups_indirect(
            Arc::clone(&indirect_buffer.core),
            indirect_offset,
            pass.device.limits(),
        ),
    );
}

/// Releases one owned reference to a compute pass encoder handle.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderRelease(
    compute_pass_encoder: native::WGPUComputePassEncoder,
) {
    release_handle(compute_pass_encoder, "WGPUComputePassEncoder");
}

/// Adds one owned reference to a compute pass encoder handle.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderAddRef(
    compute_pass_encoder: native::WGPUComputePassEncoder,
) {
    add_ref_handle(compute_pass_encoder, "WGPUComputePassEncoder");
}
