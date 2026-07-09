use super::*;

/// Sets a texture label.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle. `label` must point
/// to valid string data according to `WGPUStringView` when non-empty.
/// Returns WGPU texture set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureSetLabel(
    texture: native::WGPUTexture,
    label: native::WGPUStringView,
) {
    let texture = borrow_handle(texture, "WGPUTexture");
    *texture.label.lock().expect("label lock must not poison") = label_from_string_view(label);
}

/// Sets a texture view label.
///
/// # Safety
///
/// `texture_view` must be a non-null live yawgpu texture view handle. `label`
/// must point to valid string data according to `WGPUStringView` when
/// non-empty.
/// Returns WGPU texture view set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureViewSetLabel(
    texture_view: native::WGPUTextureView,
    label: native::WGPUStringView,
) {
    let texture_view = borrow_handle(texture_view, "WGPUTextureView");
    *texture_view
        .label
        .lock()
        .expect("label lock must not poison") = label_from_string_view(label);
}

/// Destroys a texture. This operation is idempotent.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture destroy.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureDestroy(texture: native::WGPUTexture) {
    borrow_handle(texture, "WGPUTexture").core.destroy();
}

/// Creates a view over a texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle. `descriptor`,
/// when non-null, must point to a valid `WGPUTextureViewDescriptor`.
/// Returns WGPU texture create view.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureCreateView(
    texture: native::WGPUTexture,
    descriptor: *const native::WGPUTextureViewDescriptor,
) -> native::WGPUTextureView {
    let texture = borrow_handle(texture, "WGPUTexture");
    let native_descriptor = descriptor.as_ref();
    let label = native_descriptor.and_then(|descriptor| label_from_string_view(descriptor.label));
    let descriptor = map_texture_view_descriptor(native_descriptor);
    let (view, error) = texture.core.create_view(descriptor);
    if let Some(message) = error {
        texture
            .device
            .dispatch_error(core::ErrorKind::Validation, message);
    }
    arc_to_handle(Arc::new(WGPUTextureViewImpl {
        _core: Arc::new(view),
        _texture: Arc::clone(&texture.core),
        _device: Arc::clone(&texture.device),
        _instance: Arc::clone(&texture.instance),
        label: Mutex::new(label),
    }))
}

/// Returns the descriptor format reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture get format.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetFormat(
    texture: native::WGPUTexture,
) -> native::WGPUTextureFormat {
    map_texture_format_to_native(borrow_handle(texture, "WGPUTexture").core.format())
}

/// Returns the descriptor dimension reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture get dimension.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetDimension(
    texture: native::WGPUTexture,
) -> native::WGPUTextureDimension {
    map_texture_dimension_to_native(borrow_handle(texture, "WGPUTexture").core.dimension())
}

/// Returns the texture binding view dimension.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture get texture binding view dimension.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetTextureBindingViewDimension(
    texture: native::WGPUTexture,
) -> native::WGPUTextureViewDimension {
    let texture = borrow_handle(texture, "WGPUTexture");
    if texture.core.is_error() {
        return native::WGPUTextureViewDimension_Undefined;
    }
    if texture.binding_view_dimension != native::WGPUTextureViewDimension_Undefined {
        return texture.binding_view_dimension;
    }
    match texture.core.dimension() {
        core::TextureDimension::D1 => native::WGPUTextureViewDimension_1D,
        core::TextureDimension::D2 if texture.core.size().depth_or_array_layers == 1 => {
            native::WGPUTextureViewDimension_2D
        }
        core::TextureDimension::D2 => native::WGPUTextureViewDimension_2DArray,
        core::TextureDimension::D3 => native::WGPUTextureViewDimension_3D,
        _ => native::WGPUTextureViewDimension_Undefined,
    }
}

/// Returns the descriptor width reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture get width.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetWidth(texture: native::WGPUTexture) -> u32 {
    borrow_handle(texture, "WGPUTexture").core.size().width
}

/// Returns the descriptor height reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture get height.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetHeight(texture: native::WGPUTexture) -> u32 {
    borrow_handle(texture, "WGPUTexture").core.size().height
}

/// Returns the descriptor depth/array-layer count reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture get depth or array layers.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetDepthOrArrayLayers(texture: native::WGPUTexture) -> u32 {
    borrow_handle(texture, "WGPUTexture")
        .core
        .size()
        .depth_or_array_layers
}

/// Returns the descriptor mip level count reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture get mip level count.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetMipLevelCount(texture: native::WGPUTexture) -> u32 {
    borrow_handle(texture, "WGPUTexture").core.mip_level_count()
}

/// Returns the descriptor sample count reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture get sample count.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetSampleCount(texture: native::WGPUTexture) -> u32 {
    borrow_handle(texture, "WGPUTexture").core.sample_count()
}

/// Returns the descriptor usage reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture get usage.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetUsage(
    texture: native::WGPUTexture,
) -> native::WGPUTextureUsage {
    map_texture_usage_to_native(borrow_handle(texture, "WGPUTexture").core.usage())
}

/// Releases one owned reference to a texture handle.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture release.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureRelease(texture: native::WGPUTexture) {
    release_handle(texture, "WGPUTexture");
}

/// Adds one owned reference to a texture handle.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
/// Returns WGPU texture add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureAddRef(texture: native::WGPUTexture) {
    add_ref_handle(texture, "WGPUTexture");
}

/// Releases one owned reference to a texture view handle.
///
/// # Safety
///
/// `texture_view` must be a non-null live yawgpu texture view handle.
/// Returns WGPU texture view release.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureViewRelease(texture_view: native::WGPUTextureView) {
    release_handle(texture_view, "WGPUTextureView");
}

/// Adds one owned reference to a texture view handle.
///
/// # Safety
///
/// `texture_view` must be a non-null live yawgpu texture view handle.
/// Returns WGPU texture view add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureViewAddRef(texture_view: native::WGPUTextureView) {
    add_ref_handle(texture_view, "WGPUTextureView");
}
