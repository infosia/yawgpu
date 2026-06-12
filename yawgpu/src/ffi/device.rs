use super::*;

/// Releases one owned reference to a device handle.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
/// Returns WGPU device release.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceRelease(device: native::WGPUDevice) {
    let device = device
        .as_ref()
        .map(|_| device)
        .unwrap_or_else(|| panic!("WGPUDevice must not be null"));
    let owned = Arc::from_raw(device);
    if Arc::strong_count(&owned) == 1 {
        owned.implicit_destroy_on_last_release();
    }
}

/// Adds one owned reference to a device handle.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
/// Returns WGPU device add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceAddRef(device: native::WGPUDevice) {
    add_ref_handle(device, "WGPUDevice");
}

/// Destroys a device and fires its device-lost callback once.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
/// Returns WGPU device destroy.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceDestroy(device: native::WGPUDevice) {
    let device_impl = borrow_handle(device, "WGPUDevice");
    device_impl.schedule_device_lost(device, core::DeviceLostReason::Destroyed);
}

/// Returns a future that completes when the device is lost.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
/// Returns WGPU device get lost future.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetLostFuture(device: native::WGPUDevice) -> native::WGPUFuture {
    let device_impl = borrow_handle(device, "WGPUDevice");
    device_impl.get_lost_future(device)
}

/// Pushes a device error scope for matching errors.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
/// Returns WGPU device push error scope.
#[no_mangle]
pub unsafe extern "C" fn wgpuDevicePushErrorScope(
    device: native::WGPUDevice,
    filter: native::WGPUErrorFilter,
) {
    let device = borrow_handle(device, "WGPUDevice");
    let Some(filter) = map_error_filter(filter) else {
        device.dispatch_error(core::ErrorKind::Validation, "error scope filter is invalid");
        return;
    };
    device.core.push_error_scope(filter);
}

/// Pops the innermost device error scope and resolves through the callback future.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `callback_info`
/// userdata pointers must remain valid until the callback fires.
/// Returns WGPU device pop error scope.
#[no_mangle]
pub unsafe extern "C" fn wgpuDevicePopErrorScope(
    device: native::WGPUDevice,
    callback_info: native::WGPUPopErrorScopeCallbackInfo,
) -> native::WGPUFuture {
    let device = borrow_handle(device, "WGPUDevice");
    let (status, error, message) = if device.core.is_lost() {
        (map_pop_error_scope_status_success(), None, String::new())
    } else {
        match device.core.pop_error_scope() {
            Ok(error) => (map_pop_error_scope_status_success(), error, String::new()),
            Err(core::PopErrorScopeError::EmptyStack) => (
                map_pop_error_scope_status_error(),
                None,
                "No error scopes are open".to_owned(),
            ),
            Err(_) => (
                map_pop_error_scope_status_error(),
                None,
                "Pop error scope failed".to_owned(),
            ),
        }
    };
    device
        .instance
        .register_callback(PendingCallback::PopErrorScope {
            mode: callback_info.mode,
            callback: callback_info.callback,
            status,
            error,
            message,
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        })
}

/// Sets the debug label for a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `label` must point
/// to valid string data according to `WGPUStringView` when non-empty.
/// Returns WGPU device set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceSetLabel(
    device: native::WGPUDevice,
    label: native::WGPUStringView,
) {
    let device = borrow_handle(device, "WGPUDevice");
    let label = label_from_string_view(label).unwrap_or_default();
    device.core.set_label(&label);
}

/// Creates a buffer on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUBufferDescriptor`.
/// Returns WGPU device create buffer.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBuffer(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUBufferDescriptor,
) -> native::WGPUBuffer {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUBufferDescriptor must not be null");
    let buffer = device.core.create_buffer(map_buffer_descriptor(descriptor));
    arc_to_handle(Arc::new(WGPUBufferImpl {
        core: Arc::new(buffer),
        device: Arc::clone(&device.core),
        instance: Arc::clone(&device.instance),
    }))
}

/// Creates a texture on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUTextureDescriptor`.
/// Returns WGPU device create texture.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateTexture(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUTextureDescriptor must not be null");
    let texture = device
        .core
        .create_texture(map_texture_descriptor(descriptor));
    arc_to_handle(Arc::new(WGPUTextureImpl {
        core: Arc::new(texture),
        device: Arc::clone(&device.core),
        instance: Arc::clone(&device.instance),
    }))
}

/// Creates a transient attachment on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `YaWGPUTransientAttachmentDescriptor`.
/// Returns yawgpu device create transient attachment.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuDeviceCreateTransientAttachment(
    device: native::WGPUDevice,
    descriptor: *const YaWGPUTransientAttachmentDescriptor,
) -> crate::YaWGPUTransientAttachment {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("YaWGPUTransientAttachmentDescriptor must not be null");
    let attachment = device
        .core
        .create_transient_attachment(map_transient_attachment_descriptor(descriptor));
    arc_to_handle(Arc::new(YaWGPUTransientAttachmentImpl {
        _core: Arc::new(attachment),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Creates a subpass pass layout on a device.
///
/// # Safety
///
/// `device` and `descriptor` must be non-null live yawgpu pointers.
/// Returns yawgpu device create subpass pass layout.
#[cfg(feature = "tiled")]
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

/// Creates a subpass-compatible render pipeline on a device.
///
/// # Safety
///
/// `device` and `descriptor` must be non-null live yawgpu pointers.
/// Returns yawgpu device create subpass render pipeline.
#[cfg(feature = "tiled")]
#[no_mangle]
pub unsafe extern "C" fn yawgpuDeviceCreateSubpassRenderPipeline(
    device: native::WGPUDevice,
    descriptor: *const YaWGPUSubpassRenderPipelineDescriptor,
) -> native::WGPURenderPipeline {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("YaWGPUSubpassRenderPipelineDescriptor must not be null");
    let pipeline = device
        .core
        .create_subpass_render_pipeline(map_subpass_render_pipeline_descriptor(descriptor));
    arc_to_handle(Arc::new(WGPURenderPipelineImpl {
        _core: Arc::new(pipeline),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
        bind_group_layout_handles: Mutex::new(Vec::new()),
    }))
}

/// Creates a sampler on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor`, when
/// non-null, must point to a valid `WGPUSamplerDescriptor`.
/// Returns WGPU device create sampler.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateSampler(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUSamplerDescriptor,
) -> native::WGPUSampler {
    let device = borrow_handle(device, "WGPUDevice");
    let sampler = device
        .core
        .create_sampler(map_sampler_descriptor(descriptor.as_ref()));
    arc_to_handle(Arc::new(WGPUSamplerImpl {
        _core: Arc::new(sampler),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Creates a query set on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUQuerySetDescriptor`.
/// Returns WGPU device create query set.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateQuerySet(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUQuerySetDescriptor,
) -> native::WGPUQuerySet {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUQuerySetDescriptor must not be null");
    let (query_set, error) = device
        .core
        .create_query_set(map_query_set_descriptor(descriptor));
    if let Some(message) = error {
        device.dispatch_error(core::ErrorKind::Validation, message);
    }
    arc_to_handle(Arc::new(WGPUQuerySetImpl {
        core: Arc::new(query_set),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Creates a shader module on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUShaderModuleDescriptor` and its extension chain must
/// contain exactly one recognized shader source.
/// Returns WGPU device create shader module.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateShaderModule(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUShaderModuleDescriptor,
) -> native::WGPUShaderModule {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUShaderModuleDescriptor must not be null");
    let source = map_shader_module_descriptor(descriptor);
    let key = shader_module_cache_key(&source);
    let shader_module = device.core.create_shader_module(source);
    let handle = Arc::new(WGPUShaderModuleImpl {
        _core: Arc::new(shader_module),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    });
    let handle = if !handle._core.is_error() {
        if let Some(key) = key {
            cache_handle(&device.shader_module_cache, key, handle)
        } else {
            handle
        }
    } else {
        handle
    };
    arc_to_handle(handle)
}

/// Creates a bind group layout on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUBindGroupLayoutDescriptor`.
/// Returns WGPU device create bind group layout.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBindGroupLayout(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUBindGroupLayoutDescriptor,
) -> native::WGPUBindGroupLayout {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUBindGroupLayoutDescriptor must not be null");
    let layout = device
        .core
        .create_bind_group_layout(map_bind_group_layout_descriptor(descriptor));
    arc_to_handle(Arc::new(WGPUBindGroupLayoutImpl {
        _core: Arc::new(layout),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Creates a bind group on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUBindGroupDescriptor`. `descriptor.layout` must be a
/// non-null live yawgpu bind group layout handle. `descriptor.entries`, when
/// non-null and `entryCount > 0`, must point to valid bind group entries.
/// Returns WGPU device create bind group.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBindGroup(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUBindGroupDescriptor,
) -> native::WGPUBindGroup {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUBindGroupDescriptor must not be null");
    let layout = clone_handle(descriptor.layout, "WGPUBindGroupLayout");
    let mut entries = map_bind_group_entries(descriptor);
    if !layout._device.same(&device.core) {
        entries.push(core::BindGroupEntry {
            binding: u32::MAX,
            resource: core::BindGroupResource::Invalid(
                "bind group layout must belong to the bind group device".to_owned(),
            ),
        });
    }
    let bind_group = device
        .core
        .create_bind_group(Arc::clone(&layout._core), entries);
    arc_to_handle(Arc::new(WGPUBindGroupImpl {
        _core: Arc::new(bind_group),
        _layout: Arc::clone(&layout._core),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Creates a pipeline layout on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUPipelineLayoutDescriptor`. Its `bindGroupLayouts`
/// array may be null only when `bindGroupLayoutCount` is zero.
/// Returns WGPU device create pipeline layout.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreatePipelineLayout(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUPipelineLayoutDescriptor,
) -> native::WGPUPipelineLayout {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUPipelineLayoutDescriptor must not be null");
    let key = pipeline_layout_cache_key(descriptor);
    let device_error = validate_pipeline_layout_devices(device, descriptor);
    let mut descriptor = map_pipeline_layout_descriptor(descriptor);
    if descriptor.error.is_none() {
        descriptor.error = device_error;
    }
    let pipeline_layout = device.core.create_pipeline_layout(descriptor);
    let handle = Arc::new(WGPUPipelineLayoutImpl {
        _core: Arc::new(pipeline_layout),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    });
    let handle = if !handle._core.is_error() {
        if let Some(key) = key {
            cache_handle(&device.pipeline_layout_cache, key, handle)
        } else {
            handle
        }
    } else {
        handle
    };
    arc_to_handle(handle)
}

/// Creates a compute pipeline on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUComputePipelineDescriptor`. `descriptor.compute.module`
/// must be a non-null live yawgpu shader module handle. `descriptor.layout`
/// may be null to request automatic layout.
/// Returns WGPU device create compute pipeline.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateComputePipeline(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUComputePipelineDescriptor,
) -> native::WGPUComputePipeline {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUComputePipelineDescriptor must not be null");
    arc_to_handle(create_compute_pipeline_handle(device, descriptor, true))
}

/// Creates a compute pipeline asynchronously on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUComputePipelineDescriptor`. The callback info follows
/// the `webgpu.h` callback contract.
/// Returns WGPU device create compute pipeline async.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateComputePipelineAsync(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUComputePipelineDescriptor,
    callback_info: native::WGPUCreateComputePipelineAsyncCallbackInfo,
) -> native::WGPUFuture {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUComputePipelineDescriptor must not be null");
    let pipeline = create_compute_pipeline_handle(device, descriptor, false);
    device
        .instance
        .register_callback(PendingCallback::CreateComputePipelineAsync {
            mode: callback_info.mode,
            callback: callback_info.callback,
            pipeline,
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        })
}

/// Creates a render pipeline on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPURenderPipelineDescriptor`. `descriptor.vertex.module`
/// and optional `descriptor.fragment.module` must be non-null live yawgpu
/// shader module handles. `descriptor.layout`, `depthStencil`, and `fragment`
/// may be null where allowed by WebGPU.
/// Returns WGPU device create render pipeline.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateRenderPipeline(
    device: native::WGPUDevice,
    descriptor: *const native::WGPURenderPipelineDescriptor,
) -> native::WGPURenderPipeline {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPURenderPipelineDescriptor must not be null");
    arc_to_handle(create_render_pipeline_handle(device, descriptor, true))
}

/// Creates a render pipeline asynchronously on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPURenderPipelineDescriptor`. The callback info follows
/// the `webgpu.h` callback contract.
/// Returns WGPU device create render pipeline async.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateRenderPipelineAsync(
    device: native::WGPUDevice,
    descriptor: *const native::WGPURenderPipelineDescriptor,
    callback_info: native::WGPUCreateRenderPipelineAsyncCallbackInfo,
) -> native::WGPUFuture {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPURenderPipelineDescriptor must not be null");
    let pipeline = create_render_pipeline_handle(device, descriptor, false);
    device
        .instance
        .register_callback(PendingCallback::CreateRenderPipelineAsync {
            mode: callback_info.mode,
            callback: callback_info.callback,
            pipeline,
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        })
}

/// Creates a command encoder on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` may be
/// null; P6.1 stores no command encoder descriptor fields.
/// Returns WGPU device create command encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateCommandEncoder(
    device: native::WGPUDevice,
    _descriptor: *const native::WGPUCommandEncoderDescriptor,
) -> native::WGPUCommandEncoder {
    let device = borrow_handle(device, "WGPUDevice");
    arc_to_handle(Arc::new(WGPUCommandEncoderImpl {
        core: Arc::new(device.core.create_command_encoder()),
        device: Arc::clone(&device.core),
        instance: Arc::clone(&device.instance),
    }))
}

/// Creates a render bundle encoder on a device.
///
/// # Safety
///
/// `device` and `descriptor` must be non-null live yawgpu pointers.
/// Returns WGPU device create render bundle encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateRenderBundleEncoder(
    device: native::WGPUDevice,
    descriptor: *const native::WGPURenderBundleEncoderDescriptor,
) -> native::WGPURenderBundleEncoder {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPURenderBundleEncoderDescriptor must not be null");
    let descriptor = map_render_bundle_encoder_descriptor(
        descriptor,
        device.core.limits().max_color_attachments,
    );
    let (encoder, error) =
        core::RenderBundleEncoder::new(descriptor, device.core.limits(), device.core.features());
    dispatch_optional_error(&device.core, error);
    arc_to_handle(Arc::new(WGPURenderBundleEncoderImpl {
        core: Arc::new(encoder),
        device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Gets the effective limits for a device.
///
/// If `limits.nextInChain` points to a `WGPUCompatibilityModeLimits` node
/// (identified by `WGPUSType_CompatibilityModeLimits`) the per-stage limits
/// (maxStorageBuffersIn{Vertex,Fragment}Stage,
/// maxStorageTexturesIn{Vertex,Fragment}Stage) are written into that struct.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `limits` must point
/// to writable `WGPULimits` storage. When `limits.nextInChain` is non-null it
/// must be a valid linked list of `WGPUChainedStruct` nodes.
/// Returns WGPU device get limits.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetLimits(
    device: native::WGPUDevice,
    limits: *mut native::WGPULimits,
) -> native::WGPUStatus {
    let device = borrow_handle(device, "WGPUDevice");
    let Some(limits) = limits.as_mut() else {
        return native::WGPUStatus_Error;
    };
    // Preserve the caller-supplied chain pointer before overwriting the struct.
    let caller_chain = limits.nextInChain;
    let core_limits = device.core.limits();
    *limits = map_limits_to_native(core_limits);
    limits.nextInChain = caller_chain;
    // Populate any WGPUCompatibilityModeLimits node the caller attached.
    fill_compat_limits_chain(caller_chain, core_limits);
    native::WGPUStatus_Success
}

/// Gets the resolved features for a device.
///
/// The returned `features` array is allocated by yawgpu and must be released
/// with `wgpuSupportedFeaturesFreeMembers`.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `features` must
/// point to writable `WGPUSupportedFeatures` storage.
/// Returns WGPU device get features.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetFeatures(
    device: native::WGPUDevice,
    features: *mut native::WGPUSupportedFeatures,
) {
    let device = borrow_handle(device, "WGPUDevice");
    let features = features
        .as_mut()
        .expect("WGPUSupportedFeatures must not be null");
    *features = map_features_to_native(&device.core.features());
}

/// Returns whether the device has `feature` enabled.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
/// Returns WGPU device has feature.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceHasFeature(
    device: native::WGPUDevice,
    feature: native::WGPUFeatureName,
) -> native::WGPUBool {
    let device = borrow_handle(device, "WGPUDevice");
    native::WGPUBool::from(device.core.has_feature(map_feature(feature)))
}

/// Gets the default queue for a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
/// Returns WGPU device get queue.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetQueue(device: native::WGPUDevice) -> native::WGPUQueue {
    let device = borrow_handle(device, "WGPUDevice");
    arc_to_handle(device.default_queue())
}
