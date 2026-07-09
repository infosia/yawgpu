use super::*;

/// Sets a compute pipeline label.
///
/// # Safety
///
/// `compute_pipeline` must be a non-null live yawgpu compute pipeline handle.
/// `label` must point to valid string data according to `WGPUStringView` when
/// non-empty.
/// Returns WGPU compute pipeline set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineSetLabel(
    compute_pipeline: native::WGPUComputePipeline,
    label: native::WGPUStringView,
) {
    let compute_pipeline = borrow_handle(compute_pipeline, "WGPUComputePipeline");
    *compute_pipeline
        .label
        .lock()
        .expect("label lock must not poison") = label_from_string_view(label);
}

/// Sets a pipeline layout label.
///
/// # Safety
///
/// `pipeline_layout` must be a non-null live yawgpu pipeline layout handle.
/// `label` must point to valid string data according to `WGPUStringView` when
/// non-empty.
/// Returns WGPU pipeline layout set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuPipelineLayoutSetLabel(
    pipeline_layout: native::WGPUPipelineLayout,
    label: native::WGPUStringView,
) {
    let pipeline_layout = borrow_handle(pipeline_layout, "WGPUPipelineLayout");
    *pipeline_layout
        .label
        .lock()
        .expect("label lock must not poison") = label_from_string_view(label);
}

/// Sets a render pipeline label.
///
/// # Safety
///
/// `render_pipeline` must be a non-null live yawgpu render pipeline handle.
/// `label` must point to valid string data according to `WGPUStringView` when
/// non-empty.
/// Returns WGPU render pipeline set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineSetLabel(
    render_pipeline: native::WGPURenderPipeline,
    label: native::WGPUStringView,
) {
    let render_pipeline = borrow_handle(render_pipeline, "WGPURenderPipeline");
    *render_pipeline
        .label
        .lock()
        .expect("label lock must not poison") = label_from_string_view(label);
}

/// Gets a compute pipeline bind group layout.
///
/// # Safety
///
/// `compute_pipeline` must be a non-null live yawgpu compute pipeline handle.
/// Returns WGPU compute pipeline get bind group layout.
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
/// Returns WGPU render pipeline get bind group layout.
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
/// Returns WGPU compute pipeline release.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineRelease(compute_pipeline: native::WGPUComputePipeline) {
    release_handle(compute_pipeline, "WGPUComputePipeline");
}

/// Adds one owned reference to a compute pipeline handle.
///
/// # Safety
///
/// `compute_pipeline` must be a non-null live yawgpu compute pipeline handle.
/// Returns WGPU compute pipeline add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineAddRef(compute_pipeline: native::WGPUComputePipeline) {
    add_ref_handle(compute_pipeline, "WGPUComputePipeline");
}

/// Releases one owned reference to a render pipeline handle.
///
/// # Safety
///
/// `render_pipeline` must be a non-null live yawgpu render pipeline handle.
/// Returns WGPU render pipeline release.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineRelease(render_pipeline: native::WGPURenderPipeline) {
    release_handle(render_pipeline, "WGPURenderPipeline");
}

/// Adds one owned reference to a render pipeline handle.
///
/// # Safety
///
/// `render_pipeline` must be a non-null live yawgpu render pipeline handle.
/// Returns WGPU render pipeline add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineAddRef(render_pipeline: native::WGPURenderPipeline) {
    add_ref_handle(render_pipeline, "WGPURenderPipeline");
}
