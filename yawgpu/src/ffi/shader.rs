use super::*;

/// Sets a shader module label.
///
/// # Safety
///
/// `shader_module` must be a non-null live yawgpu shader module handle.
/// `label` must point to valid string data according to `WGPUStringView` when
/// non-empty.
/// Returns WGPU shader module set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuShaderModuleSetLabel(
    shader_module: native::WGPUShaderModule,
    label: native::WGPUStringView,
) {
    let shader_module = borrow_handle(shader_module, "WGPUShaderModule");
    *shader_module
        .label
        .lock()
        .expect("label lock must not poison") = label_from_string_view(label);
}

/// Requests compilation information for a shader module.
///
/// # Safety
///
/// `shader_module` must be a non-null live yawgpu shader module handle.
/// Returns WGPU shader module get compilation info.
#[no_mangle]
pub unsafe extern "C" fn wgpuShaderModuleGetCompilationInfo(
    shader_module: native::WGPUShaderModule,
    callback_info: native::WGPUCompilationInfoCallbackInfo,
) -> native::WGPUFuture {
    let shader_module = borrow_handle(shader_module, "WGPUShaderModule");
    shader_module
        ._instance
        .register_callback(PendingCallback::CompilationInfo {
            mode: callback_info.mode,
            callback: callback_info.callback,
            shader_module: Arc::clone(&shader_module._core),
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        })
}

/// Releases one owned reference to a shader module handle.
///
/// # Safety
///
/// `shader_module` must be a non-null live yawgpu shader module handle.
/// Returns WGPU shader module release.
#[no_mangle]
pub unsafe extern "C" fn wgpuShaderModuleRelease(shader_module: native::WGPUShaderModule) {
    release_handle(shader_module, "WGPUShaderModule");
}

/// Adds one owned reference to a shader module handle.
///
/// # Safety
///
/// `shader_module` must be a non-null live yawgpu shader module handle.
/// Returns WGPU shader module add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuShaderModuleAddRef(shader_module: native::WGPUShaderModule) {
    add_ref_handle(shader_module, "WGPUShaderModule");
}
