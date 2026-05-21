use super::*;

/// Gets a compute pipeline bind group layout.
///
/// # Safety
///
/// `compute_pipeline` must be a non-null live yawgpu compute pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineGetBindGroupLayout(
    compute_pipeline: native::WGPUComputePipeline,
    group_index: u32,
) -> native::WGPUBindGroupLayout {
    let pipeline = borrow_handle(compute_pipeline, "WGPUComputePipeline");
    get_pipeline_bind_group_layout(
        pipeline._core.bind_group_layouts(),
        &pipeline._device,
        &pipeline._instance,
        &pipeline.bind_group_layout_handles,
        group_index,
    )
}

/// Gets a render pipeline bind group layout.
///
/// # Safety
///
/// `render_pipeline` must be a non-null live yawgpu render pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineGetBindGroupLayout(
    render_pipeline: native::WGPURenderPipeline,
    group_index: u32,
) -> native::WGPUBindGroupLayout {
    let pipeline = borrow_handle(render_pipeline, "WGPURenderPipeline");
    get_pipeline_bind_group_layout(
        pipeline._core.bind_group_layouts(),
        &pipeline._device,
        &pipeline._instance,
        &pipeline.bind_group_layout_handles,
        group_index,
    )
}

/// Releases one owned reference to a compute pipeline handle.
///
/// # Safety
///
/// `compute_pipeline` must be a non-null live yawgpu compute pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineRelease(compute_pipeline: native::WGPUComputePipeline) {
    release_handle(compute_pipeline, "WGPUComputePipeline");
}

/// Adds one owned reference to a compute pipeline handle.
///
/// # Safety
///
/// `compute_pipeline` must be a non-null live yawgpu compute pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineAddRef(compute_pipeline: native::WGPUComputePipeline) {
    add_ref_handle(compute_pipeline, "WGPUComputePipeline");
}

/// Releases one owned reference to a render pipeline handle.
///
/// # Safety
///
/// `render_pipeline` must be a non-null live yawgpu render pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineRelease(render_pipeline: native::WGPURenderPipeline) {
    release_handle(render_pipeline, "WGPURenderPipeline");
}

/// Adds one owned reference to a render pipeline handle.
///
/// # Safety
///
/// `render_pipeline` must be a non-null live yawgpu render pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineAddRef(render_pipeline: native::WGPURenderPipeline) {
    add_ref_handle(render_pipeline, "WGPURenderPipeline");
}
