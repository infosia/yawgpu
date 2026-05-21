use super::*;

/// Creates a shader module from raw SPIR-V words.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `YaWGPUShaderModuleSpirVDescriptor`. When `codeSize` is
/// non-zero, `code` must point to at least `codeSize` SPIR-V words.
/// Returns WGPU device create shader module SPIR-V.
#[cfg(feature = "shader-passthrough")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuDeviceCreateShaderModuleSpirV(
    device: native::WGPUDevice,
    descriptor: *const YaWGPUShaderModuleSpirVDescriptor,
) -> native::WGPUShaderModule {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("YaWGPUShaderModuleSpirVDescriptor must not be null");
    let words = if descriptor.codeSize == 0 || descriptor.code.is_null() {
        Vec::new()
    } else {
        std::slice::from_raw_parts(descriptor.code, descriptor.codeSize as usize).to_vec()
    };
    let shader_module = device.core.create_shader_module_spirv(words);
    arc_to_handle(Arc::new(WGPUShaderModuleImpl {
        _core: Arc::new(shader_module),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Creates a shader module from raw MSL source and entry-point metadata.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `YaWGPUShaderModuleMslDescriptor`. When `entryPointCount`
/// is non-zero, `entryPoints` must point to at least `entryPointCount` entries.
/// String views must be valid for their declared lengths.
/// Returns WGPU device create shader module MSL.
#[cfg(feature = "shader-passthrough")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuDeviceCreateShaderModuleMsl(
    device: native::WGPUDevice,
    descriptor: *const YaWGPUShaderModuleMslDescriptor,
) -> native::WGPUShaderModule {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("YaWGPUShaderModuleMslDescriptor must not be null");
    let source = string_view_to_str(descriptor.code).map_or_else(String::new, ToOwned::to_owned);
    let entry_points = if descriptor.entryPointCount == 0 || descriptor.entryPoints.is_null() {
        Vec::new()
    } else {
        std::slice::from_raw_parts(descriptor.entryPoints, descriptor.entryPointCount)
            .iter()
            .map(map_msl_entry_point)
            .collect()
    };
    let reflection = core::MslReflection::new(entry_points);
    let shader_module = device.core.create_shader_module_msl(source, reflection);
    arc_to_handle(Arc::new(WGPUShaderModuleImpl {
        _core: Arc::new(shader_module),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

#[cfg(feature = "shader-passthrough")]
fn map_msl_entry_point(entry: &YaWGPUMslEntryPoint) -> core::MslEntryPoint {
    core::MslEntryPoint::new(
        unsafe { string_view_to_str(entry.name) }.map_or_else(String::new, ToOwned::to_owned),
        entry.stage,
        entry.workgroupSize,
    )
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
