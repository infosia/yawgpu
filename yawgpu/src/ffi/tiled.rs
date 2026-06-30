use super::*;
use crate::{
    YaWGPUSubpassPassLayoutDescriptor, YaWGPUSubpassRenderPipelineDescriptor,
    YaWGPUTiledCapabilities,
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
