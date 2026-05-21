use super::*;

/// Releases one owned reference to a bind group layout handle.
///
/// # Safety
///
/// `bind_group_layout` must be a non-null live yawgpu bind group layout handle.
/// Returns WGPU bind group layout release.
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupLayoutRelease(
    bind_group_layout: native::WGPUBindGroupLayout,
) {
    release_handle(bind_group_layout, "WGPUBindGroupLayout");
}

/// Adds one owned reference to a bind group layout handle.
///
/// # Safety
///
/// `bind_group_layout` must be a non-null live yawgpu bind group layout handle.
/// Returns WGPU bind group layout add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupLayoutAddRef(bind_group_layout: native::WGPUBindGroupLayout) {
    add_ref_handle(bind_group_layout, "WGPUBindGroupLayout");
}

/// Releases one owned reference to a bind group handle.
///
/// # Safety
///
/// `bind_group` must be a non-null live yawgpu bind group handle.
/// Returns WGPU bind group release.
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupRelease(bind_group: native::WGPUBindGroup) {
    release_handle(bind_group, "WGPUBindGroup");
}

/// Adds one owned reference to a bind group handle.
///
/// # Safety
///
/// `bind_group` must be a non-null live yawgpu bind group handle.
/// Returns WGPU bind group add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupAddRef(bind_group: native::WGPUBindGroup) {
    add_ref_handle(bind_group, "WGPUBindGroup");
}

/// Releases one owned reference to a pipeline layout handle.
///
/// # Safety
///
/// `pipeline_layout` must be a non-null live yawgpu pipeline layout handle.
/// Returns WGPU pipeline layout release.
#[no_mangle]
pub unsafe extern "C" fn wgpuPipelineLayoutRelease(pipeline_layout: native::WGPUPipelineLayout) {
    release_handle(pipeline_layout, "WGPUPipelineLayout");
}

/// Adds one owned reference to a pipeline layout handle.
///
/// # Safety
///
/// `pipeline_layout` must be a non-null live yawgpu pipeline layout handle.
/// Returns WGPU pipeline layout add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuPipelineLayoutAddRef(pipeline_layout: native::WGPUPipelineLayout) {
    add_ref_handle(pipeline_layout, "WGPUPipelineLayout");
}
