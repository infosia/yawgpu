use super::*;

wgpu_handle_exports!(
    refcount_and_label:
    WGPUShaderModuleImpl,
    native::WGPUShaderModule,
    "WGPUShaderModule",
    wgpuShaderModuleAddRef,
    wgpuShaderModuleRelease,
    wgpuShaderModuleSetLabel
);

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
