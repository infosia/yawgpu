use super::*;

/// Sets a bind group label.
///
/// # Safety
///
/// `bind_group` must be a non-null live yawgpu bind group handle. `label` must
/// point to valid string data according to `WGPUStringView` when non-empty.
/// Returns WGPU bind group set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupSetLabel(
    bind_group: native::WGPUBindGroup,
    label: native::WGPUStringView,
) {
    let bind_group = borrow_handle(bind_group, "WGPUBindGroup");
    *bind_group.label.lock().expect("label lock must not poison") = label_from_string_view(label);
}

/// Sets a bind group layout label.
///
/// # Safety
///
/// `bind_group_layout` must be a non-null live yawgpu bind group layout handle.
/// `label` must point to valid string data according to `WGPUStringView` when
/// non-empty.
/// Returns WGPU bind group layout set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupLayoutSetLabel(
    bind_group_layout: native::WGPUBindGroupLayout,
    label: native::WGPUStringView,
) {
    let bind_group_layout = borrow_handle(bind_group_layout, "WGPUBindGroupLayout");
    *bind_group_layout
        .label
        .lock()
        .expect("label lock must not poison") = label_from_string_view(label);
}

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
