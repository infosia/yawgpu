use std::ffi::CStr;
use std::sync::Arc;

use crate::native;
use crate::{
    WGPUBindGroupLayoutImpl, WGPUBufferImpl, WGPUPipelineLayoutImpl, WGPUQuerySetImpl,
    WGPUSamplerImpl, WGPUShaderModuleImpl, WGPUTextureViewImpl,
};
use yawgpu_core as core;

pub const WGPU_STRLEN: usize = usize::MAX;

/// Handle refcount contract:
/// - create/request functions return one owned C reference (+1) via `Arc::into_raw`.
/// - `wgpuXxxAddRef` borrows the handle, clones the `Arc`, and leaks that clone (+1).
/// - `wgpuXxxRelease` reconstructs one `Arc` with `Arc::from_raw` and drops it (-1).
#[must_use]
pub fn arc_to_handle<T>(value: Arc<T>) -> *const T {
    Arc::into_raw(value)
}

/// Drops one owned C reference for a yawgpu handle.
///
/// # Safety
///
/// `handle` must be a non-null pointer returned by `Arc::into_raw` for `T`.
/// It must represent one currently owned C reference.
pub unsafe fn release_handle<T>(handle: *const T, name: &str) {
    let handle = handle
        .as_ref()
        .map(|_| handle)
        .unwrap_or_else(|| panic!("{name} must not be null"));
    drop(Arc::from_raw(handle));
}

/// Clones one C handle reference without consuming the incoming handle.
///
/// # Safety
///
/// `handle` must be a non-null live pointer returned by `Arc::into_raw` for
/// `T`.
pub unsafe fn add_ref_handle<T>(handle: *const T, name: &str) {
    let handle = handle
        .as_ref()
        .map(|_| handle)
        .unwrap_or_else(|| panic!("{name} must not be null"));
    Arc::increment_strong_count(handle);
}

#[must_use]
/// Clones a C handle into a Rust `Arc`.
///
/// # Safety
///
/// `handle` must be a non-null live pointer returned by `Arc::into_raw` for
/// `T`.
pub unsafe fn clone_handle<T>(handle: *const T, name: &str) -> Arc<T> {
    let handle = handle
        .as_ref()
        .map(|_| handle)
        .unwrap_or_else(|| panic!("{name} must not be null"));
    Arc::increment_strong_count(handle);
    Arc::from_raw(handle)
}

/// Borrows a C handle without changing its reference count.
///
/// # Safety
///
/// `handle` must be a non-null live pointer returned by `Arc::into_raw` for
/// `T`, and the returned borrow must not outlive the owned C reference.
pub unsafe fn borrow_handle<'a, T>(handle: *const T, name: &str) -> &'a T {
    handle
        .as_ref()
        .unwrap_or_else(|| panic!("{name} must not be null"))
}

#[must_use]
pub fn string_view(data: &[u8]) -> native::WGPUStringView {
    native::WGPUStringView {
        data: data.as_ptr().cast(),
        length: data.len(),
    }
}

#[must_use]
/// Converts a `WGPUStringView` to UTF-8 text.
///
/// # Safety
///
/// `value.data`, when non-null, must point to a valid byte buffer for
/// `value.length` bytes, or to a valid NUL-terminated C string when
/// `value.length == WGPU_STRLEN`.
pub unsafe fn string_view_to_str<'a>(value: native::WGPUStringView) -> Option<&'a str> {
    if value.data.is_null() {
        return None;
    }

    let bytes = if value.length == WGPU_STRLEN {
        CStr::from_ptr(value.data).to_bytes()
    } else {
        std::slice::from_raw_parts(value.data.cast::<u8>(), value.length)
    };

    std::str::from_utf8(bytes).ok()
}

#[must_use]
/// Converts a label string view to an owned string.
///
/// # Safety
///
/// Same requirements as [`string_view_to_str`].
pub unsafe fn label_from_string_view(value: native::WGPUStringView) -> Option<String> {
    string_view_to_str(value).map(ToOwned::to_owned)
}

/// Converts a shader module descriptor chain to a core shader source.
///
/// # Safety
///
/// `descriptor.nextInChain` must be either null or a valid linked list of
/// `WGPUChainedStruct` nodes. Recognized shader-source nodes must point to
/// valid `WGPUShaderSourceWGSL` or `WGPUShaderSourceSPIRV` storage. WGSL
/// string data and SPIR-V word data must be valid for their declared lengths.
#[must_use]
pub unsafe fn map_shader_module_descriptor(
    descriptor: &native::WGPUShaderModuleDescriptor,
) -> core::ShaderModuleSource {
    let mut source = None;
    let mut chain = descriptor.nextInChain;

    while let Some(node) = chain.as_ref() {
        match node.sType {
            native::WGPUSType_ShaderSourceWGSL => {
                if source.is_some() {
                    return core::ShaderModuleSource::Invalid(
                        "shader module descriptor must contain exactly one shader source"
                            .to_owned(),
                    );
                }
                let Some(wgsl) = chain.cast::<native::WGPUShaderSourceWGSL>().as_ref() else {
                    return core::ShaderModuleSource::Invalid(
                        "WGSL shader source chain node must be valid".to_owned(),
                    );
                };
                let code =
                    string_view_to_str(wgsl.code).map_or_else(String::new, ToOwned::to_owned);
                source = Some(core::ShaderModuleSource::Wgsl(code));
            }
            native::WGPUSType_ShaderSourceSPIRV => {
                if source.is_some() {
                    return core::ShaderModuleSource::Invalid(
                        "shader module descriptor must contain exactly one shader source"
                            .to_owned(),
                    );
                }
                let Some(spirv) = chain.cast::<native::WGPUShaderSourceSPIRV>().as_ref() else {
                    return core::ShaderModuleSource::Invalid(
                        "SPIR-V shader source chain node must be valid".to_owned(),
                    );
                };
                if spirv.codeSize > 0 && spirv.code.is_null() {
                    return core::ShaderModuleSource::Invalid(
                        "SPIR-V shader source code must not be null when codeSize is non-zero"
                            .to_owned(),
                    );
                }
                let words = if spirv.codeSize == 0 {
                    Vec::new()
                } else {
                    std::slice::from_raw_parts(spirv.code, spirv.codeSize as usize).to_vec()
                };
                source = Some(core::ShaderModuleSource::Spirv(words));
            }
            _ => {}
        }

        chain = node.next;
    }

    source.unwrap_or_else(|| {
        core::ShaderModuleSource::Invalid(
            "shader module descriptor must contain exactly one shader source".to_owned(),
        )
    })
}

/// Converts a bind group layout descriptor to the core representation.
///
/// # Safety
///
/// `descriptor.entries`, when non-null and `entryCount > 0`, must point to
/// `entryCount` valid `WGPUBindGroupLayoutEntry` values.
#[must_use]
pub unsafe fn map_bind_group_layout_descriptor(
    descriptor: &native::WGPUBindGroupLayoutDescriptor,
) -> core::BindGroupLayoutDescriptor {
    if descriptor.entryCount > 0 && descriptor.entries.is_null() {
        return core::BindGroupLayoutDescriptor {
            entries: Vec::new(),
            error: Some("bind group layout entries must not be null".to_owned()),
        };
    }

    let mut error = None;
    let entries = if descriptor.entryCount == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(descriptor.entries, descriptor.entryCount)
            .iter()
            .map(|entry| map_bind_group_layout_entry(entry, &mut error))
            .collect()
    };

    core::BindGroupLayoutDescriptor { entries, error }
}

/// Converts bind group entries to the core representation.
///
/// # Safety
///
/// `descriptor.entries`, when non-null and `entryCount > 0`, must point to
/// `entryCount` valid `WGPUBindGroupEntry` values. Any non-null resource
/// handles must be live yawgpu handles of the matching type.
#[must_use]
pub unsafe fn map_bind_group_entries(
    descriptor: &native::WGPUBindGroupDescriptor,
) -> Vec<core::BindGroupEntry> {
    if descriptor.entryCount > 0 && descriptor.entries.is_null() {
        return vec![core::BindGroupEntry {
            binding: 0,
            resource: core::BindGroupResource::Invalid(
                "bind group entries must not be null".to_owned(),
            ),
        }];
    }

    if descriptor.entryCount == 0 {
        return Vec::new();
    }

    std::slice::from_raw_parts(descriptor.entries, descriptor.entryCount)
        .iter()
        .map(|entry| map_bind_group_entry(entry))
        .collect()
}

unsafe fn map_bind_group_entry(entry: &native::WGPUBindGroupEntry) -> core::BindGroupEntry {
    let mut present_count = 0;
    let mut resource = core::BindGroupResource::Invalid(
        "bind group entry must set exactly one resource".to_owned(),
    );

    if !entry.buffer.is_null() {
        present_count += 1;
        let buffer = clone_handle::<WGPUBufferImpl>(entry.buffer, "WGPUBuffer");
        resource = core::BindGroupResource::Buffer {
            buffer: Arc::clone(&buffer.core),
            device: Arc::clone(&buffer.device),
            offset: entry.offset,
            size: entry.size,
        };
    }
    if !entry.sampler.is_null() {
        present_count += 1;
        let sampler = clone_handle::<WGPUSamplerImpl>(entry.sampler, "WGPUSampler");
        resource = core::BindGroupResource::Sampler {
            sampler: Arc::clone(&sampler._core),
            device: Arc::clone(&sampler._device),
        };
    }
    if !entry.textureView.is_null() {
        present_count += 1;
        let texture_view =
            clone_handle::<WGPUTextureViewImpl>(entry.textureView, "WGPUTextureView");
        resource = core::BindGroupResource::TextureView {
            texture_view: Arc::clone(&texture_view._core),
            device: Arc::clone(&texture_view._device),
        };
    }

    if present_count != 1 {
        resource = core::BindGroupResource::Invalid(
            "bind group entry must set exactly one resource".to_owned(),
        );
    }

    core::BindGroupEntry {
        binding: entry.binding,
        resource,
    }
}

/// Converts a pipeline layout descriptor to the core representation.
///
/// # Safety
///
/// `descriptor.bindGroupLayouts`, when non-null and
/// `bindGroupLayoutCount > 0`, must point to `bindGroupLayoutCount`
/// `WGPUBindGroupLayout` slots. Non-null slots must be live yawgpu handles.
#[must_use]
pub unsafe fn map_pipeline_layout_descriptor(
    descriptor: &native::WGPUPipelineLayoutDescriptor,
) -> core::PipelineLayoutDescriptor {
    let mut error = None;
    let bind_group_layouts = if descriptor.bindGroupLayoutCount == 0 {
        Vec::new()
    } else if descriptor.bindGroupLayouts.is_null() {
        set_first_error(
            &mut error,
            "pipeline layout bindGroupLayouts must not be null when count is non-zero",
        );
        Vec::new()
    } else {
        std::slice::from_raw_parts(descriptor.bindGroupLayouts, descriptor.bindGroupLayoutCount)
            .iter()
            .filter_map(|layout| {
                if layout.is_null() {
                    set_first_error(
                        &mut error,
                        "pipeline layout bindGroupLayouts elements must not be null",
                    );
                    None
                } else {
                    let layout =
                        clone_handle::<WGPUBindGroupLayoutImpl>(*layout, "WGPUBindGroupLayout");
                    Some(Arc::clone(&layout._core))
                }
            })
            .collect()
    };

    core::PipelineLayoutDescriptor {
        bind_group_layouts,
        immediate_size: descriptor.immediateSize,
        error,
    }
}

/// Converts a compute pipeline descriptor to the core representation.
///
/// # Safety
///
/// `descriptor.compute.module` must be a non-null live yawgpu shader module.
/// `descriptor.layout`, when non-null, must be a live yawgpu pipeline layout.
/// `compute.constants`, when non-null and `constantCount > 0`, must point to
/// `constantCount` valid `WGPUConstantEntry` values.
#[must_use]
pub unsafe fn map_compute_pipeline_descriptor(
    descriptor: &native::WGPUComputePipelineDescriptor,
) -> core::ComputePipelineDescriptor {
    let mut error = None;
    let compute = &descriptor.compute;
    let shader_module = clone_handle::<WGPUShaderModuleImpl>(compute.module, "WGPUShaderModule");
    let layout = if descriptor.layout.is_null() {
        core::ComputePipelineLayout::Auto
    } else {
        let layout =
            clone_handle::<WGPUPipelineLayoutImpl>(descriptor.layout, "WGPUPipelineLayout");
        core::ComputePipelineLayout::Explicit(Arc::clone(&layout._core))
    };
    let entry_point = string_view_to_str(compute.entryPoint).map(ToOwned::to_owned);
    let constants = map_pipeline_constants(
        compute.constantCount,
        compute.constants,
        "compute pipeline",
        &mut error,
    );

    core::ComputePipelineDescriptor {
        layout,
        shader_module: Arc::clone(&shader_module._core),
        entry_point,
        constants,
        error,
    }
}

/// Converts a render pipeline descriptor to the core representation.
///
/// # Safety
///
/// `descriptor.vertex.module` and optional `fragment.module` must be non-null
/// live yawgpu shader modules. `descriptor.layout`, when non-null, must be a
/// live yawgpu pipeline layout. Optional pointer arrays must be valid for their
/// declared counts.
#[must_use]
pub unsafe fn map_render_pipeline_descriptor(
    descriptor: &native::WGPURenderPipelineDescriptor,
) -> core::RenderPipelineDescriptor {
    let mut error = None;
    let vertex_module =
        clone_handle::<WGPUShaderModuleImpl>(descriptor.vertex.module, "WGPUShaderModule");
    let layout = if descriptor.layout.is_null() {
        core::RenderPipelineLayout::Auto
    } else {
        let layout =
            clone_handle::<WGPUPipelineLayoutImpl>(descriptor.layout, "WGPUPipelineLayout");
        core::RenderPipelineLayout::Explicit(Arc::clone(&layout._core))
    };

    let vertex = core::RenderPipelineVertexState {
        shader: core::RenderPipelineShaderStage {
            module: Arc::clone(&vertex_module._core),
            entry_point: string_view_to_str(descriptor.vertex.entryPoint).map(ToOwned::to_owned),
            constants: map_pipeline_constants(
                descriptor.vertex.constantCount,
                descriptor.vertex.constants,
                "render pipeline vertex",
                &mut error,
            ),
        },
        buffer_count: descriptor.vertex.bufferCount,
        buffers: map_vertex_buffers(&descriptor.vertex, &mut error),
    };

    let fragment = if let Some(fragment) = descriptor.fragment.as_ref() {
        let fragment_module =
            clone_handle::<WGPUShaderModuleImpl>(fragment.module, "WGPUShaderModule");
        if fragment.targetCount > 0 && fragment.targets.is_null() {
            set_first_error(
                &mut error,
                "render pipeline fragment targets must not be null when count is non-zero",
            );
        }
        Some(core::RenderPipelineFragmentState {
            shader: core::RenderPipelineShaderStage {
                module: Arc::clone(&fragment_module._core),
                entry_point: string_view_to_str(fragment.entryPoint).map(ToOwned::to_owned),
                constants: map_pipeline_constants(
                    fragment.constantCount,
                    fragment.constants,
                    "render pipeline fragment",
                    &mut error,
                ),
            },
            target_count: fragment.targetCount,
            targets: map_color_targets(fragment, &mut error),
        })
    } else {
        None
    };

    core::RenderPipelineDescriptor {
        layout,
        vertex,
        primitive: map_primitive_state(descriptor.primitive, &mut error),
        depth_stencil: descriptor
            .depthStencil
            .as_ref()
            .map(map_depth_stencil_state),
        multisample: map_multisample_state(descriptor.multisample),
        fragment,
        error,
    }
}

unsafe fn map_vertex_buffers(
    vertex: &native::WGPUVertexState,
    error: &mut Option<String>,
) -> Vec<core::VertexBufferLayout> {
    if vertex.bufferCount == 0 {
        return Vec::new();
    }
    if vertex.buffers.is_null() {
        set_first_error(
            error,
            "render pipeline vertex buffers must not be null when count is non-zero",
        );
        return Vec::new();
    }

    std::slice::from_raw_parts(vertex.buffers, vertex.bufferCount)
        .iter()
        .map(|buffer| {
            let attributes = map_vertex_attributes(buffer, error);
            core::VertexBufferLayout {
                array_stride: buffer.arrayStride,
                step_mode: map_vertex_step_mode(buffer.stepMode, error),
                attributes,
            }
        })
        .collect()
}

unsafe fn map_vertex_attributes(
    buffer: &native::WGPUVertexBufferLayout,
    error: &mut Option<String>,
) -> Vec<core::VertexAttribute> {
    if buffer.attributeCount == 0 {
        return Vec::new();
    }
    if buffer.attributes.is_null() {
        set_first_error(
            error,
            "render pipeline vertex attributes must not be null when count is non-zero",
        );
        return Vec::new();
    }

    std::slice::from_raw_parts(buffer.attributes, buffer.attributeCount)
        .iter()
        .map(|attribute| core::VertexAttribute {
            format: map_vertex_format(attribute.format),
            offset: attribute.offset,
            shader_location: attribute.shaderLocation,
        })
        .collect()
}

#[must_use]
pub fn map_vertex_format(value: native::WGPUVertexFormat) -> core::VertexFormat {
    value.into()
}

#[must_use]
pub fn map_vertex_format_to_native(value: core::VertexFormat) -> native::WGPUVertexFormat {
    value.into()
}

fn map_vertex_step_mode(
    value: native::WGPUVertexStepMode,
    error: &mut Option<String>,
) -> core::VertexStepMode {
    match value {
        native::WGPUVertexStepMode_Undefined | native::WGPUVertexStepMode_Vertex => {
            core::VertexStepMode::Vertex
        }
        native::WGPUVertexStepMode_Instance => core::VertexStepMode::Instance,
        _ => {
            set_first_error(error, "render pipeline vertex stepMode is invalid");
            core::VertexStepMode::Vertex
        }
    }
}

unsafe fn map_pipeline_constants(
    count: usize,
    entries: *const native::WGPUConstantEntry,
    label: &str,
    error: &mut Option<String>,
) -> Vec<core::PipelineConstant> {
    if count == 0 {
        return Vec::new();
    }
    if entries.is_null() {
        set_first_error(
            error,
            &format!("{label} constants must not be null when count is non-zero"),
        );
        return Vec::new();
    }
    std::slice::from_raw_parts(entries, count)
        .iter()
        .map(|entry| {
            let key = string_view_to_str(entry.key).unwrap_or_default().to_owned();
            core::PipelineConstant {
                key,
                value: entry.value,
            }
        })
        .collect()
}

fn map_primitive_state(
    state: native::WGPUPrimitiveState,
    error: &mut Option<String>,
) -> core::PrimitiveState {
    core::PrimitiveState {
        topology: match state.topology {
            native::WGPUPrimitiveTopology_Undefined
            | native::WGPUPrimitiveTopology_TriangleList => core::PrimitiveTopology::TriangleList,
            native::WGPUPrimitiveTopology_PointList => core::PrimitiveTopology::PointList,
            native::WGPUPrimitiveTopology_LineList => core::PrimitiveTopology::LineList,
            native::WGPUPrimitiveTopology_LineStrip => core::PrimitiveTopology::LineStrip,
            native::WGPUPrimitiveTopology_TriangleStrip => core::PrimitiveTopology::TriangleStrip,
            _ => {
                set_first_error(error, "invalid primitive topology");
                core::PrimitiveTopology::TriangleList
            }
        },
        strip_index_format: match state.stripIndexFormat {
            native::WGPUIndexFormat_Undefined => None,
            native::WGPUIndexFormat_Uint16 => Some(core::IndexFormat::Uint16),
            native::WGPUIndexFormat_Uint32 => Some(core::IndexFormat::Uint32),
            _ => {
                set_first_error(error, "invalid strip index format");
                None
            }
        },
    }
}

fn map_depth_stencil_state(state: &native::WGPUDepthStencilState) -> core::DepthStencilState {
    core::DepthStencilState {
        format: map_texture_format(state.format),
        depth_write_enabled: map_optional_bool(state.depthWriteEnabled),
        depth_compare: map_compare_function(state.depthCompare),
        stencil_front: map_stencil_face_state(state.stencilFront),
        stencil_back: map_stencil_face_state(state.stencilBack),
        stencil_read_mask: state.stencilReadMask,
        stencil_write_mask: state.stencilWriteMask,
        depth_bias: state.depthBias,
        depth_bias_slope_scale: state.depthBiasSlopeScale,
        depth_bias_clamp: state.depthBiasClamp,
    }
}

fn map_multisample_state(state: native::WGPUMultisampleState) -> core::MultisampleState {
    core::MultisampleState {
        count: state.count,
        mask: state.mask,
        alpha_to_coverage_enabled: state.alphaToCoverageEnabled != 0,
    }
}

fn map_color_targets(
    fragment: &native::WGPUFragmentState,
    error: &mut Option<String>,
) -> Vec<core::ColorTargetState> {
    if fragment.targetCount == 0 {
        return Vec::new();
    }
    if fragment.targets.is_null() {
        set_first_error(
            error,
            "render pipeline fragment targets must not be null when count is non-zero",
        );
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(fragment.targets, fragment.targetCount) }
        .iter()
        .map(|target| core::ColorTargetState {
            format: map_texture_format(target.format),
            blend: !target.blend.is_null(),
            write_mask: target.writeMask,
        })
        .collect()
}

fn map_optional_bool(value: native::WGPUOptionalBool) -> Option<bool> {
    match value {
        native::WGPUOptionalBool_False => Some(false),
        native::WGPUOptionalBool_True => Some(true),
        _ => None,
    }
}

fn map_stencil_face_state(value: native::WGPUStencilFaceState) -> core::StencilFaceState {
    core::StencilFaceState {
        compare: map_compare_function(value.compare).unwrap_or(core::CompareFunction::Always),
        fail_op: map_stencil_operation(value.failOp),
        depth_fail_op: map_stencil_operation(value.depthFailOp),
        pass_op: map_stencil_operation(value.passOp),
    }
}

fn map_stencil_operation(value: native::WGPUStencilOperation) -> core::StencilOperation {
    match value {
        native::WGPUStencilOperation_Zero => core::StencilOperation::Zero,
        native::WGPUStencilOperation_Replace => core::StencilOperation::Replace,
        native::WGPUStencilOperation_Invert => core::StencilOperation::Invert,
        native::WGPUStencilOperation_IncrementClamp => core::StencilOperation::IncrementClamp,
        native::WGPUStencilOperation_DecrementClamp => core::StencilOperation::DecrementClamp,
        native::WGPUStencilOperation_IncrementWrap => core::StencilOperation::IncrementWrap,
        native::WGPUStencilOperation_DecrementWrap => core::StencilOperation::DecrementWrap,
        _ => core::StencilOperation::Keep,
    }
}

fn map_bind_group_layout_entry(
    entry: &native::WGPUBindGroupLayoutEntry,
    error: &mut Option<String>,
) -> core::BindGroupLayoutEntry {
    let mut present_count = 0;
    let mut kind = None;

    if entry.buffer.type_ != native::WGPUBufferBindingType_BindingNotUsed {
        present_count += 1;
        kind = map_buffer_binding_layout(entry.buffer, error);
    }
    if entry.sampler.type_ != native::WGPUSamplerBindingType_BindingNotUsed {
        present_count += 1;
        kind = map_sampler_binding_layout(entry.sampler, error);
    }
    if entry.texture.sampleType != native::WGPUTextureSampleType_BindingNotUsed {
        present_count += 1;
        kind = map_texture_binding_layout(entry.texture, error);
    }
    if entry.storageTexture.access != native::WGPUStorageTextureAccess_BindingNotUsed {
        present_count += 1;
        kind = map_storage_texture_binding_layout(entry.storageTexture, error);
    }

    if present_count != 1 && error.is_none() {
        *error = Some("bind group layout entry must set exactly one binding layout".to_owned());
        kind = None;
    }

    core::BindGroupLayoutEntry {
        binding: entry.binding,
        visibility: entry.visibility,
        binding_array_size: entry.bindingArraySize,
        kind,
    }
}

fn map_buffer_binding_layout(
    layout: native::WGPUBufferBindingLayout,
    error: &mut Option<String>,
) -> Option<core::BindingLayoutKind> {
    let ty = match layout.type_ {
        native::WGPUBufferBindingType_Undefined | native::WGPUBufferBindingType_Uniform => {
            core::BufferBindingType::Uniform
        }
        native::WGPUBufferBindingType_Storage => core::BufferBindingType::Storage,
        native::WGPUBufferBindingType_ReadOnlyStorage => core::BufferBindingType::ReadOnlyStorage,
        _ => {
            set_first_error(error, "invalid buffer binding type");
            return None;
        }
    };
    Some(core::BindingLayoutKind::Buffer {
        ty,
        has_dynamic_offset: layout.hasDynamicOffset != 0,
        min_binding_size: layout.minBindingSize,
    })
}

fn map_sampler_binding_layout(
    layout: native::WGPUSamplerBindingLayout,
    error: &mut Option<String>,
) -> Option<core::BindingLayoutKind> {
    let ty = match layout.type_ {
        native::WGPUSamplerBindingType_Undefined | native::WGPUSamplerBindingType_Filtering => {
            core::SamplerBindingType::Filtering
        }
        native::WGPUSamplerBindingType_NonFiltering => core::SamplerBindingType::NonFiltering,
        native::WGPUSamplerBindingType_Comparison => core::SamplerBindingType::Comparison,
        _ => {
            set_first_error(error, "invalid sampler binding type");
            return None;
        }
    };
    Some(core::BindingLayoutKind::Sampler { ty })
}

fn map_texture_binding_layout(
    layout: native::WGPUTextureBindingLayout,
    error: &mut Option<String>,
) -> Option<core::BindingLayoutKind> {
    let sample_type = match layout.sampleType {
        native::WGPUTextureSampleType_Undefined | native::WGPUTextureSampleType_Float => {
            core::TextureSampleType::Float
        }
        native::WGPUTextureSampleType_UnfilterableFloat => {
            core::TextureSampleType::UnfilterableFloat
        }
        native::WGPUTextureSampleType_Depth => core::TextureSampleType::Depth,
        native::WGPUTextureSampleType_Sint => core::TextureSampleType::Sint,
        native::WGPUTextureSampleType_Uint => core::TextureSampleType::Uint,
        _ => {
            set_first_error(error, "invalid texture sample type");
            return None;
        }
    };
    Some(core::BindingLayoutKind::Texture {
        sample_type,
        view_dimension: map_texture_view_dimension(layout.viewDimension)
            .unwrap_or(core::TextureViewDimension::D2),
        multisampled: layout.multisampled != 0,
    })
}

fn map_storage_texture_binding_layout(
    layout: native::WGPUStorageTextureBindingLayout,
    error: &mut Option<String>,
) -> Option<core::BindingLayoutKind> {
    let access = match layout.access {
        native::WGPUStorageTextureAccess_Undefined | native::WGPUStorageTextureAccess_WriteOnly => {
            core::StorageTextureAccess::WriteOnly
        }
        native::WGPUStorageTextureAccess_ReadOnly => core::StorageTextureAccess::ReadOnly,
        native::WGPUStorageTextureAccess_ReadWrite => core::StorageTextureAccess::ReadWrite,
        _ => {
            set_first_error(error, "invalid storage texture access");
            return None;
        }
    };
    Some(core::BindingLayoutKind::StorageTexture {
        access,
        format: map_texture_format(layout.format),
        view_dimension: map_texture_view_dimension(layout.viewDimension)
            .unwrap_or(core::TextureViewDimension::D2),
    })
}

fn set_first_error(error: &mut Option<String>, message: &str) {
    if error.is_none() {
        *error = Some(message.to_owned());
    }
}

#[must_use]
pub fn map_feature(value: native::WGPUFeatureName) -> core::Feature {
    match value {
        native::WGPUFeatureName_CoreFeaturesAndLimits => core::Feature::CoreFeaturesAndLimits,
        native::WGPUFeatureName_RG11B10UfloatRenderable => core::Feature::Rg11b10UfloatRenderable,
        native::WGPUFeatureName_TimestampQuery => core::Feature::TimestampQuery,
        native::WGPUFeatureName_TextureFormatsTier1 => core::Feature::TextureFormatsTier1,
        native::WGPUFeatureName_TextureFormatsTier2 => core::Feature::TextureFormatsTier2,
        other => core::Feature::Other(other),
    }
}

#[must_use]
pub fn map_feature_to_native(value: core::Feature) -> native::WGPUFeatureName {
    match value {
        core::Feature::CoreFeaturesAndLimits => native::WGPUFeatureName_CoreFeaturesAndLimits,
        core::Feature::Rg11b10UfloatRenderable => native::WGPUFeatureName_RG11B10UfloatRenderable,
        core::Feature::TimestampQuery => native::WGPUFeatureName_TimestampQuery,
        core::Feature::TextureFormatsTier1 => native::WGPUFeatureName_TextureFormatsTier1,
        core::Feature::TextureFormatsTier2 => native::WGPUFeatureName_TextureFormatsTier2,
        core::Feature::Other(value) => value,
        // exhaustive as of core::Feature @ 2026-05-17
        _ => native::WGPUFeatureName_Force32,
    }
}

#[must_use]
/// Converts a query set descriptor to the core representation.
///
/// # Safety
///
/// `descriptor.label`, when non-null, must point to a valid WebGPU string view.
pub unsafe fn map_query_set_descriptor(
    descriptor: &native::WGPUQuerySetDescriptor,
) -> core::QuerySetDescriptor {
    core::QuerySetDescriptor {
        label: label_from_string_view(descriptor.label).unwrap_or_default(),
        kind: map_query_type(descriptor.type_),
        count: descriptor.count,
    }
}

#[must_use]
pub fn map_query_type(value: native::WGPUQueryType) -> core::QueryType {
    value.into()
}

#[must_use]
pub fn map_query_type_to_native(value: core::QueryType) -> native::WGPUQueryType {
    value.into()
}

#[must_use]
pub fn map_feature_level(value: native::WGPUFeatureLevel) -> core::FeatureLevel {
    match value {
        native::WGPUFeatureLevel_Compatibility => core::FeatureLevel::Compatibility,
        _ => core::FeatureLevel::Core,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceLostCallbackInfo {
    pub mode: native::WGPUCallbackMode,
    pub callback: native::WGPUDeviceLostCallback,
    pub userdata1: usize,
    pub userdata2: usize,
}

impl DeviceLostCallbackInfo {
    #[must_use]
    pub fn has_callback(self) -> bool {
        self.callback.is_some()
    }
}

#[must_use]
pub fn map_device_lost_callback_info(
    value: native::WGPUDeviceLostCallbackInfo,
) -> DeviceLostCallbackInfo {
    DeviceLostCallbackInfo {
        mode: value.mode,
        callback: value.callback,
        userdata1: value.userdata1 as usize,
        userdata2: value.userdata2 as usize,
    }
}

#[must_use]
pub fn map_device_lost_reason(reason: core::DeviceLostReason) -> native::WGPUDeviceLostReason {
    match reason {
        core::DeviceLostReason::Unknown => native::WGPUDeviceLostReason_Unknown,
        core::DeviceLostReason::Destroyed => native::WGPUDeviceLostReason_Destroyed,
        core::DeviceLostReason::CallbackCancelled => native::WGPUDeviceLostReason_CallbackCancelled,
        core::DeviceLostReason::FailedCreation => native::WGPUDeviceLostReason_FailedCreation,
        // exhaustive as of core::DeviceLostReason @ 2026-05-17
        _ => native::WGPUDeviceLostReason_Unknown,
    }
}

#[must_use]
pub fn map_error_filter(value: native::WGPUErrorFilter) -> Option<core::ErrorFilter> {
    match value {
        native::WGPUErrorFilter_Validation => Some(core::ErrorFilter::Validation),
        native::WGPUErrorFilter_OutOfMemory => Some(core::ErrorFilter::OutOfMemory),
        native::WGPUErrorFilter_Internal => Some(core::ErrorFilter::Internal),
        _ => None,
    }
}

#[must_use]
pub fn map_error_type(kind: core::ErrorKind) -> native::WGPUErrorType {
    match kind {
        core::ErrorKind::Validation => native::WGPUErrorType_Validation,
        core::ErrorKind::OutOfMemory => native::WGPUErrorType_OutOfMemory,
        core::ErrorKind::Internal => native::WGPUErrorType_Internal,
        _ => native::WGPUErrorType_Unknown,
    }
}

#[must_use]
pub fn map_pop_error_scope_status_error() -> native::WGPUPopErrorScopeStatus {
    native::WGPUPopErrorScopeStatus_Error
}

#[must_use]
pub fn map_pop_error_scope_status_success() -> native::WGPUPopErrorScopeStatus {
    native::WGPUPopErrorScopeStatus_Success
}

#[must_use]
pub fn map_buffer_usage(value: native::WGPUBufferUsage) -> core::BufferUsage {
    core::BufferUsage::from_bits_retain(value)
}

#[must_use]
pub fn map_buffer_usage_to_native(value: core::BufferUsage) -> native::WGPUBufferUsage {
    value.bits()
}

#[must_use]
pub fn map_buffer_map_state(value: core::BufferMapState) -> native::WGPUBufferMapState {
    match value {
        core::BufferMapState::Unmapped => native::WGPUBufferMapState_Unmapped,
        core::BufferMapState::Pending => native::WGPUBufferMapState_Pending,
        core::BufferMapState::Mapped => native::WGPUBufferMapState_Mapped,
        // exhaustive as of core::BufferMapState @ 2026-05-17
        _ => native::WGPUBufferMapState_Force32,
    }
}

#[must_use]
pub fn map_map_async_status(value: core::MapAsyncStatus) -> native::WGPUMapAsyncStatus {
    match value {
        core::MapAsyncStatus::Success => native::WGPUMapAsyncStatus_Success,
        core::MapAsyncStatus::CallbackCancelled => native::WGPUMapAsyncStatus_CallbackCancelled,
        core::MapAsyncStatus::Error => native::WGPUMapAsyncStatus_Error,
        core::MapAsyncStatus::Aborted => native::WGPUMapAsyncStatus_Aborted,
        // exhaustive as of core::MapAsyncStatus @ 2026-05-17
        _ => native::WGPUMapAsyncStatus_Error,
    }
}

#[must_use]
pub fn map_queue_work_done_status(
    value: core::QueueWorkDoneStatus,
) -> native::WGPUQueueWorkDoneStatus {
    match value {
        core::QueueWorkDoneStatus::Success => native::WGPUQueueWorkDoneStatus_Success,
        core::QueueWorkDoneStatus::CallbackCancelled => {
            native::WGPUQueueWorkDoneStatus_CallbackCancelled
        }
        core::QueueWorkDoneStatus::Error => native::WGPUQueueWorkDoneStatus_Error,
        // exhaustive as of core::QueueWorkDoneStatus @ 2026-05-17
        _ => native::WGPUQueueWorkDoneStatus_Error,
    }
}

#[must_use]
pub fn map_compilation_info_request_status_success() -> native::WGPUCompilationInfoRequestStatus {
    native::WGPUCompilationInfoRequestStatus_Success
}

#[must_use]
pub fn map_compilation_message_type_error() -> native::WGPUCompilationMessageType {
    native::WGPUCompilationMessageType_Error
}

pub fn map_map_mode(value: native::WGPUMapMode) -> Result<core::MapMode, &'static str> {
    let bits = u32::try_from(value).map_err(|_| "map mode has unsupported bits")?;
    core::MapMode::from_bits(bits)
}

#[must_use]
pub fn map_buffer_descriptor(value: &native::WGPUBufferDescriptor) -> core::BufferDescriptor {
    core::BufferDescriptor {
        usage: map_buffer_usage(value.usage),
        size: value.size,
        mapped_at_creation: value.mappedAtCreation != 0,
    }
}

#[must_use]
pub fn map_address_mode(value: native::WGPUAddressMode) -> Option<core::AddressMode> {
    match value {
        native::WGPUAddressMode_Undefined => None,
        native::WGPUAddressMode_ClampToEdge => Some(core::AddressMode::ClampToEdge),
        native::WGPUAddressMode_Repeat => Some(core::AddressMode::Repeat),
        native::WGPUAddressMode_MirrorRepeat => Some(core::AddressMode::MirrorRepeat),
        _ => None,
    }
}

#[must_use]
pub fn map_filter_mode(value: native::WGPUFilterMode) -> Option<core::FilterMode> {
    match value {
        native::WGPUFilterMode_Undefined => None,
        native::WGPUFilterMode_Nearest => Some(core::FilterMode::Nearest),
        native::WGPUFilterMode_Linear => Some(core::FilterMode::Linear),
        _ => None,
    }
}

#[must_use]
pub fn map_mipmap_filter_mode(
    value: native::WGPUMipmapFilterMode,
) -> Option<core::MipmapFilterMode> {
    match value {
        native::WGPUMipmapFilterMode_Undefined => None,
        native::WGPUMipmapFilterMode_Nearest => Some(core::MipmapFilterMode::Nearest),
        native::WGPUMipmapFilterMode_Linear => Some(core::MipmapFilterMode::Linear),
        _ => None,
    }
}

#[must_use]
pub fn map_compare_function(value: native::WGPUCompareFunction) -> Option<core::CompareFunction> {
    match value {
        native::WGPUCompareFunction_Undefined => None,
        native::WGPUCompareFunction_Never => Some(core::CompareFunction::Never),
        native::WGPUCompareFunction_Less => Some(core::CompareFunction::Less),
        native::WGPUCompareFunction_Equal => Some(core::CompareFunction::Equal),
        native::WGPUCompareFunction_LessEqual => Some(core::CompareFunction::LessEqual),
        native::WGPUCompareFunction_Greater => Some(core::CompareFunction::Greater),
        native::WGPUCompareFunction_NotEqual => Some(core::CompareFunction::NotEqual),
        native::WGPUCompareFunction_GreaterEqual => Some(core::CompareFunction::GreaterEqual),
        native::WGPUCompareFunction_Always => Some(core::CompareFunction::Always),
        _ => None,
    }
}

#[must_use]
pub fn map_sampler_descriptor(
    value: Option<&native::WGPUSamplerDescriptor>,
) -> core::SamplerDescriptor {
    let Some(value) = value else {
        return core::SamplerDescriptor::default();
    };
    core::SamplerDescriptor {
        address_mode_u: map_address_mode(value.addressModeU),
        address_mode_v: map_address_mode(value.addressModeV),
        address_mode_w: map_address_mode(value.addressModeW),
        mag_filter: map_filter_mode(value.magFilter),
        min_filter: map_filter_mode(value.minFilter),
        mipmap_filter: map_mipmap_filter_mode(value.mipmapFilter),
        lod_min_clamp: value.lodMinClamp,
        lod_max_clamp: value.lodMaxClamp,
        compare: map_compare_function(value.compare),
        max_anisotropy: value.maxAnisotropy,
    }
}

#[must_use]
pub fn map_texture_usage(value: native::WGPUTextureUsage) -> core::TextureUsage {
    core::TextureUsage::from_bits_retain(value)
}

#[must_use]
pub fn map_texture_usage_to_native(value: core::TextureUsage) -> native::WGPUTextureUsage {
    value.bits()
}

#[must_use]
pub fn map_texture_dimension(value: native::WGPUTextureDimension) -> core::TextureDimension {
    match value {
        native::WGPUTextureDimension_1D => core::TextureDimension::D1,
        native::WGPUTextureDimension_3D => core::TextureDimension::D3,
        _ => core::TextureDimension::D2,
    }
}

#[must_use]
pub fn map_texture_dimension_to_native(
    value: core::TextureDimension,
) -> native::WGPUTextureDimension {
    match value {
        core::TextureDimension::D1 => native::WGPUTextureDimension_1D,
        core::TextureDimension::D2 => native::WGPUTextureDimension_2D,
        core::TextureDimension::D3 => native::WGPUTextureDimension_3D,
        // exhaustive as of core::TextureDimension @ 2026-05-17
        _ => native::WGPUTextureDimension_2D,
    }
}

#[must_use]
pub fn map_texture_format(value: native::WGPUTextureFormat) -> core::TextureFormat {
    value.into()
}

#[must_use]
pub fn map_texture_format_to_native(value: core::TextureFormat) -> native::WGPUTextureFormat {
    value.into()
}

#[must_use]
pub fn map_extent_3d(value: native::WGPUExtent3D) -> core::Extent3d {
    core::Extent3d {
        width: value.width,
        height: value.height,
        depth_or_array_layers: value.depthOrArrayLayers,
    }
}

#[must_use]
pub fn map_origin_3d(value: native::WGPUOrigin3D) -> core::Origin3d {
    core::Origin3d {
        x: value.x,
        y: value.y,
        z: value.z,
    }
}

#[must_use]
pub fn map_texel_copy_buffer_layout(
    value: native::WGPUTexelCopyBufferLayout,
) -> core::TexelCopyBufferLayout {
    core::TexelCopyBufferLayout {
        offset: value.offset,
        bytes_per_row: if value.bytesPerRow == native::WGPU_COPY_STRIDE_UNDEFINED {
            None
        } else {
            Some(value.bytesPerRow)
        },
        rows_per_image: if value.rowsPerImage == native::WGPU_COPY_STRIDE_UNDEFINED {
            None
        } else {
            Some(value.rowsPerImage)
        },
    }
}

#[must_use]
pub fn map_texel_copy_texture_info_parts(
    value: &native::WGPUTexelCopyTextureInfo,
) -> (u32, core::Origin3d, core::TextureAspect) {
    (
        value.mipLevel,
        map_origin_3d(value.origin),
        map_texture_aspect(value.aspect).unwrap_or(core::TextureAspect::All),
    )
}

/// Converts a render pass descriptor to core data.
///
/// # Safety
///
/// Nested non-null texture view handles must be live yawgpu handles. Null
/// color attachment views are decoded as sparse holes.
pub unsafe fn map_render_pass_descriptor(
    value: &native::WGPURenderPassDescriptor,
    max_color_attachments: u32,
) -> core::RenderPassDescriptor {
    let color_attachment_count = value
        .colorAttachmentCount
        .min(max_color_attachments as usize + 1);
    let color_attachments = if color_attachment_count == 0 || value.colorAttachments.is_null() {
        vec![None; color_attachment_count]
    } else {
        std::slice::from_raw_parts(value.colorAttachments, color_attachment_count)
            .iter()
            .map(|attachment| map_render_pass_color_attachment(attachment))
            .collect()
    };
    let depth_stencil_attachment = value
        .depthStencilAttachment
        .as_ref()
        .map(|attachment| map_render_pass_depth_stencil_attachment(attachment));
    let occlusion_query_set = if value.occlusionQuerySet.is_null() {
        None
    } else {
        Some(
            (*clone_handle::<WGPUQuerySetImpl>(value.occlusionQuerySet, "WGPUQuerySet").core)
                .clone(),
        )
    };
    let timestamp_writes = value
        .timestampWrites
        .as_ref()
        .map(|timestamp_writes| map_render_pass_timestamp_writes(timestamp_writes));

    core::RenderPassDescriptor {
        max_color_attachments,
        color_attachments,
        depth_stencil_attachment,
        occlusion_query_set,
        timestamp_writes,
    }
}

/// Maps a render bundle encoder descriptor.
///
/// # Safety
///
/// `colorFormats` must point to `colorFormatCount` elements when the count is
/// non-zero.
pub unsafe fn map_render_bundle_encoder_descriptor(
    value: &native::WGPURenderBundleEncoderDescriptor,
    max_color_attachments: u32,
) -> core::RenderBundleEncoderDescriptor {
    let color_format_count = value
        .colorFormatCount
        .min(max_color_attachments as usize + 1);
    let color_formats = if color_format_count == 0 || value.colorFormats.is_null() {
        vec![None; color_format_count]
    } else {
        std::slice::from_raw_parts(value.colorFormats, color_format_count)
            .iter()
            .copied()
            .map(|format| {
                (format != native::WGPUTextureFormat_Undefined)
                    .then_some(map_texture_format(format))
            })
            .collect()
    };
    core::RenderBundleEncoderDescriptor {
        max_color_attachments,
        color_formats,
        depth_stencil_format: (value.depthStencilFormat != native::WGPUTextureFormat_Undefined)
            .then_some(map_texture_format(value.depthStencilFormat)),
        sample_count: value.sampleCount,
        depth_read_only: value.depthReadOnly != 0,
        stencil_read_only: value.stencilReadOnly != 0,
    }
}

unsafe fn map_render_pass_color_attachment(
    value: &native::WGPURenderPassColorAttachment,
) -> Option<core::RenderPassColorAttachment> {
    if value.view.is_null() {
        return None;
    }
    let view = clone_handle::<WGPUTextureViewImpl>(value.view, "WGPUTextureView");
    let resolve_target = if value.resolveTarget.is_null() {
        None
    } else {
        Some(Arc::clone(
            &clone_handle::<WGPUTextureViewImpl>(value.resolveTarget, "WGPUTextureView")._core,
        ))
    };

    Some(core::RenderPassColorAttachment {
        view: Arc::clone(&view._core),
        resolve_target,
        load_op: map_load_op(value.loadOp),
        store_op: map_store_op(value.storeOp),
        clear_value: map_color(value.clearValue),
    })
}

unsafe fn map_render_pass_depth_stencil_attachment(
    value: &native::WGPURenderPassDepthStencilAttachment,
) -> core::RenderPassDepthStencilAttachment {
    let view = clone_handle::<WGPUTextureViewImpl>(value.view, "WGPUTextureView");
    core::RenderPassDepthStencilAttachment {
        view: Arc::clone(&view._core),
        depth_load_op: map_load_op(value.depthLoadOp),
        depth_store_op: map_store_op(value.depthStoreOp),
        depth_clear_value: value.depthClearValue,
        stencil_load_op: map_load_op(value.stencilLoadOp),
        stencil_store_op: map_store_op(value.stencilStoreOp),
    }
}

unsafe fn map_render_pass_timestamp_writes(
    value: &native::WGPUPassTimestampWrites,
) -> core::RenderPassTimestampWrites {
    let query_set = clone_handle::<WGPUQuerySetImpl>(value.querySet, "WGPUQuerySet");
    core::RenderPassTimestampWrites {
        query_set: (*query_set.core).clone(),
        beginning_index: map_query_index(value.beginningOfPassWriteIndex),
        end_index: map_query_index(value.endOfPassWriteIndex),
    }
}

#[must_use]
pub fn map_query_index(value: u32) -> Option<u32> {
    (value != native::WGPU_QUERY_SET_INDEX_UNDEFINED).then_some(value)
}

#[must_use]
pub fn map_load_op(value: native::WGPULoadOp) -> core::LoadOp {
    match value {
        native::WGPULoadOp_Load => core::LoadOp::Load,
        native::WGPULoadOp_Clear => core::LoadOp::Clear,
        _ => core::LoadOp::Undefined,
    }
}

#[must_use]
pub fn map_store_op(value: native::WGPUStoreOp) -> core::StoreOp {
    match value {
        native::WGPUStoreOp_Store => core::StoreOp::Store,
        native::WGPUStoreOp_Discard => core::StoreOp::Discard,
        _ => core::StoreOp::Undefined,
    }
}

#[must_use]
pub fn map_color(value: native::WGPUColor) -> core::Color {
    core::Color {
        r: value.r,
        g: value.g,
        b: value.b,
        a: value.a,
    }
}

#[must_use]
/// Converts a texture descriptor to the core representation.
///
/// # Safety
///
/// `value.viewFormats`, when non-null and `viewFormatCount > 0`, must point
/// to `viewFormatCount` valid `WGPUTextureFormat` entries.
pub unsafe fn map_texture_descriptor(
    value: &native::WGPUTextureDescriptor,
) -> core::TextureDescriptor {
    let view_formats = if value.viewFormatCount == 0 || value.viewFormats.is_null() {
        Vec::new()
    } else {
        std::slice::from_raw_parts(value.viewFormats, value.viewFormatCount)
            .iter()
            .copied()
            .map(map_texture_format)
            .collect()
    };
    core::TextureDescriptor {
        usage: map_texture_usage(value.usage),
        dimension: map_texture_dimension(value.dimension),
        size: map_extent_3d(value.size),
        format: map_texture_format(value.format),
        mip_level_count: value.mipLevelCount,
        sample_count: value.sampleCount,
        view_formats,
    }
}

#[must_use]
pub fn map_texture_view_dimension(
    value: native::WGPUTextureViewDimension,
) -> Option<core::TextureViewDimension> {
    match value {
        native::WGPUTextureViewDimension_Undefined => None,
        native::WGPUTextureViewDimension_1D => Some(core::TextureViewDimension::D1),
        native::WGPUTextureViewDimension_2D => Some(core::TextureViewDimension::D2),
        native::WGPUTextureViewDimension_2DArray => Some(core::TextureViewDimension::D2Array),
        native::WGPUTextureViewDimension_Cube => Some(core::TextureViewDimension::Cube),
        native::WGPUTextureViewDimension_CubeArray => Some(core::TextureViewDimension::CubeArray),
        native::WGPUTextureViewDimension_3D => Some(core::TextureViewDimension::D3),
        _ => None,
    }
}

#[must_use]
pub fn map_texture_aspect(value: native::WGPUTextureAspect) -> Option<core::TextureAspect> {
    match value {
        native::WGPUTextureAspect_Undefined => None,
        native::WGPUTextureAspect_All => Some(core::TextureAspect::All),
        native::WGPUTextureAspect_DepthOnly => Some(core::TextureAspect::DepthOnly),
        native::WGPUTextureAspect_StencilOnly => Some(core::TextureAspect::StencilOnly),
        _ => None,
    }
}

#[must_use]
pub fn map_texture_view_descriptor(
    value: Option<&native::WGPUTextureViewDescriptor>,
) -> core::TextureViewDescriptor {
    let Some(value) = value else {
        return core::TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
        };
    };
    core::TextureViewDescriptor {
        format: if value.format == native::WGPUTextureFormat_Undefined {
            None
        } else {
            Some(map_texture_format(value.format))
        },
        dimension: map_texture_view_dimension(value.dimension),
        base_mip_level: value.baseMipLevel,
        mip_level_count: if value.mipLevelCount == native::WGPU_MIP_LEVEL_COUNT_UNDEFINED {
            None
        } else {
            Some(value.mipLevelCount)
        },
        base_array_layer: value.baseArrayLayer,
        array_layer_count: if value.arrayLayerCount == native::WGPU_ARRAY_LAYER_COUNT_UNDEFINED {
            None
        } else {
            Some(value.arrayLayerCount)
        },
        aspect: map_texture_aspect(value.aspect),
    }
}

pub fn map_features_to_native(features: &core::FeatureSet) -> native::WGPUSupportedFeatures {
    let features = features
        .iter()
        .copied()
        .map(map_feature_to_native)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let feature_count = features.len();
    let features = Box::into_raw(features);

    native::WGPUSupportedFeatures {
        featureCount: feature_count,
        features: features.cast(),
    }
}

/// Frees the feature array allocated by `map_features_to_native`.
///
/// # Safety
///
/// `features.features`, when non-null, must be a pointer previously returned
/// by `map_features_to_native` with the same `featureCount`.
pub unsafe fn free_supported_features(features: native::WGPUSupportedFeatures) {
    if features.features.is_null() {
        return;
    }
    let slice =
        std::ptr::slice_from_raw_parts_mut(features.features.cast_mut(), features.featureCount);
    drop(Box::from_raw(slice));
}

#[must_use]
pub fn map_limits_to_native(limits: core::Limits) -> native::WGPULimits {
    native::WGPULimits {
        nextInChain: std::ptr::null_mut(),
        maxTextureDimension1D: limits.max_texture_dimension_1d,
        maxTextureDimension2D: limits.max_texture_dimension_2d,
        maxTextureDimension3D: limits.max_texture_dimension_3d,
        maxTextureArrayLayers: limits.max_texture_array_layers,
        maxBindGroups: limits.max_bind_groups,
        maxBindGroupsPlusVertexBuffers: limits.max_bind_groups_plus_vertex_buffers,
        maxBindingsPerBindGroup: limits.max_bindings_per_bind_group,
        maxDynamicUniformBuffersPerPipelineLayout: limits
            .max_dynamic_uniform_buffers_per_pipeline_layout,
        maxDynamicStorageBuffersPerPipelineLayout: limits
            .max_dynamic_storage_buffers_per_pipeline_layout,
        maxSampledTexturesPerShaderStage: limits.max_sampled_textures_per_shader_stage,
        maxSamplersPerShaderStage: limits.max_samplers_per_shader_stage,
        maxStorageBuffersPerShaderStage: limits.max_storage_buffers_per_shader_stage,
        maxStorageTexturesPerShaderStage: limits.max_storage_textures_per_shader_stage,
        maxUniformBuffersPerShaderStage: limits.max_uniform_buffers_per_shader_stage,
        maxUniformBufferBindingSize: limits.max_uniform_buffer_binding_size,
        maxStorageBufferBindingSize: limits.max_storage_buffer_binding_size,
        minUniformBufferOffsetAlignment: limits.min_uniform_buffer_offset_alignment,
        minStorageBufferOffsetAlignment: limits.min_storage_buffer_offset_alignment,
        maxVertexBuffers: limits.max_vertex_buffers,
        maxBufferSize: limits.max_buffer_size,
        maxVertexAttributes: limits.max_vertex_attributes,
        maxVertexBufferArrayStride: limits.max_vertex_buffer_array_stride,
        maxInterStageShaderVariables: limits.max_inter_stage_shader_variables,
        maxColorAttachments: limits.max_color_attachments,
        maxColorAttachmentBytesPerSample: limits.max_color_attachment_bytes_per_sample,
        maxComputeWorkgroupStorageSize: limits.max_compute_workgroup_storage_size,
        maxComputeInvocationsPerWorkgroup: limits.max_compute_invocations_per_workgroup,
        maxComputeWorkgroupSizeX: limits.max_compute_workgroup_size_x,
        maxComputeWorkgroupSizeY: limits.max_compute_workgroup_size_y,
        maxComputeWorkgroupSizeZ: limits.max_compute_workgroup_size_z,
        maxComputeWorkgroupsPerDimension: limits.max_compute_workgroups_per_dimension,
        maxImmediateSize: limits.max_immediate_size,
    }
}

#[must_use]
pub fn map_limits(value: &native::WGPULimits) -> core::Limits {
    let default = core::Limits::DEFAULT;
    let mut limits = default;
    limits.max_texture_dimension_1d = limit_u32(
        value.maxTextureDimension1D,
        default.max_texture_dimension_1d,
    );
    limits.max_texture_dimension_2d = limit_u32(
        value.maxTextureDimension2D,
        default.max_texture_dimension_2d,
    );
    limits.max_texture_dimension_3d = limit_u32(
        value.maxTextureDimension3D,
        default.max_texture_dimension_3d,
    );
    limits.max_texture_array_layers = limit_u32(
        value.maxTextureArrayLayers,
        default.max_texture_array_layers,
    );
    limits.max_bind_groups = limit_u32(value.maxBindGroups, default.max_bind_groups);
    limits.max_bind_groups_plus_vertex_buffers = limit_u32(
        value.maxBindGroupsPlusVertexBuffers,
        default.max_bind_groups_plus_vertex_buffers,
    );
    limits.max_bindings_per_bind_group = limit_u32(
        value.maxBindingsPerBindGroup,
        default.max_bindings_per_bind_group,
    );
    limits.max_dynamic_uniform_buffers_per_pipeline_layout = limit_u32(
        value.maxDynamicUniformBuffersPerPipelineLayout,
        default.max_dynamic_uniform_buffers_per_pipeline_layout,
    );
    limits.max_dynamic_storage_buffers_per_pipeline_layout = limit_u32(
        value.maxDynamicStorageBuffersPerPipelineLayout,
        default.max_dynamic_storage_buffers_per_pipeline_layout,
    );
    limits.max_sampled_textures_per_shader_stage = limit_u32(
        value.maxSampledTexturesPerShaderStage,
        default.max_sampled_textures_per_shader_stage,
    );
    limits.max_samplers_per_shader_stage = limit_u32(
        value.maxSamplersPerShaderStage,
        default.max_samplers_per_shader_stage,
    );
    limits.max_storage_buffers_per_shader_stage = limit_u32(
        value.maxStorageBuffersPerShaderStage,
        default.max_storage_buffers_per_shader_stage,
    );
    limits.max_storage_textures_per_shader_stage = limit_u32(
        value.maxStorageTexturesPerShaderStage,
        default.max_storage_textures_per_shader_stage,
    );
    limits.max_uniform_buffers_per_shader_stage = limit_u32(
        value.maxUniformBuffersPerShaderStage,
        default.max_uniform_buffers_per_shader_stage,
    );
    limits.max_uniform_buffer_binding_size = limit_u64(
        value.maxUniformBufferBindingSize,
        default.max_uniform_buffer_binding_size,
    );
    limits.max_storage_buffer_binding_size = limit_u64(
        value.maxStorageBufferBindingSize,
        default.max_storage_buffer_binding_size,
    );
    limits.min_uniform_buffer_offset_alignment = limit_u32(
        value.minUniformBufferOffsetAlignment,
        default.min_uniform_buffer_offset_alignment,
    );
    limits.min_storage_buffer_offset_alignment = limit_u32(
        value.minStorageBufferOffsetAlignment,
        default.min_storage_buffer_offset_alignment,
    );
    limits.max_vertex_buffers = limit_u32(value.maxVertexBuffers, default.max_vertex_buffers);
    limits.max_buffer_size = limit_u64(value.maxBufferSize, default.max_buffer_size);
    limits.max_vertex_attributes =
        limit_u32(value.maxVertexAttributes, default.max_vertex_attributes);
    limits.max_vertex_buffer_array_stride = limit_u32(
        value.maxVertexBufferArrayStride,
        default.max_vertex_buffer_array_stride,
    );
    limits.max_inter_stage_shader_variables = limit_u32(
        value.maxInterStageShaderVariables,
        default.max_inter_stage_shader_variables,
    );
    limits.max_color_attachments =
        limit_u32(value.maxColorAttachments, default.max_color_attachments);
    limits.max_color_attachment_bytes_per_sample = limit_u32(
        value.maxColorAttachmentBytesPerSample,
        default.max_color_attachment_bytes_per_sample,
    );
    limits.max_compute_workgroup_storage_size = limit_u32(
        value.maxComputeWorkgroupStorageSize,
        default.max_compute_workgroup_storage_size,
    );
    limits.max_compute_invocations_per_workgroup = limit_u32(
        value.maxComputeInvocationsPerWorkgroup,
        default.max_compute_invocations_per_workgroup,
    );
    limits.max_compute_workgroup_size_x = limit_u32(
        value.maxComputeWorkgroupSizeX,
        default.max_compute_workgroup_size_x,
    );
    limits.max_compute_workgroup_size_y = limit_u32(
        value.maxComputeWorkgroupSizeY,
        default.max_compute_workgroup_size_y,
    );
    limits.max_compute_workgroup_size_z = limit_u32(
        value.maxComputeWorkgroupSizeZ,
        default.max_compute_workgroup_size_z,
    );
    limits.max_compute_workgroups_per_dimension = limit_u32(
        value.maxComputeWorkgroupsPerDimension,
        default.max_compute_workgroups_per_dimension,
    );
    limits.max_immediate_size = limit_u32(value.maxImmediateSize, default.max_immediate_size);
    limits
}

#[must_use]
fn limit_u32(value: u32, default: u32) -> u32 {
    if value == native::WGPU_LIMIT_U32_UNDEFINED {
        default
    } else {
        value
    }
}

#[must_use]
fn limit_u64(value: u64, default: u64) -> u64 {
    if value == native::WGPU_LIMIT_U64_UNDEFINED {
        default
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        WGPUBindGroupLayoutImpl, WGPUBufferImpl, WGPUDeviceImpl, WGPUInstanceImpl,
        WGPUPipelineLayoutImpl, WGPUShaderModuleImpl, WGPUTextureImpl, WGPUTextureViewImpl,
    };
    use std::collections::{BTreeMap, HashSet};
    use std::ffi::{c_void, CString};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn empty_string_view() -> native::WGPUStringView {
        native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        }
    }

    fn instance_impl() -> Arc<WGPUInstanceImpl> {
        Arc::new(WGPUInstanceImpl {
            core: Arc::new(core::Instance::new_noop()),
            timed_wait_any_enabled: false,
            pending_callbacks: Mutex::new(BTreeMap::new()),
        })
    }

    fn device_impl() -> Arc<WGPUDeviceImpl> {
        let instance = instance_impl();
        let adapter = instance
            .core
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter");
        let device = adapter
            .create_device(None, &[], "device", "queue")
            .expect("Noop device");
        Arc::new(WGPUDeviceImpl {
            core: Arc::new(device),
            instance,
            device_lost_callback: DeviceLostCallbackInfo {
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: None,
                userdata1: 0,
                userdata2: 0,
            },
            device_lost_futures: Mutex::new(Vec::new()),
            default_queue: Mutex::new(None),
            shader_module_cache: Mutex::new(std::collections::HashMap::new()),
            pipeline_layout_cache: Mutex::new(std::collections::HashMap::new()),
            compute_pipeline_cache: Mutex::new(std::collections::HashMap::new()),
            render_pipeline_cache: Mutex::new(std::collections::HashMap::new()),
        })
    }

    fn buffer_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUBuffer {
        let buffer = device.core.create_buffer(core::BufferDescriptor {
            usage: core::BufferUsage::COPY_SRC | core::BufferUsage::COPY_DST,
            size: 64,
            mapped_at_creation: false,
        });
        arc_to_handle(Arc::new(WGPUBufferImpl {
            core: Arc::new(buffer),
            device: Arc::clone(&device.core),
            instance: Arc::clone(&device.instance),
        }))
    }

    fn shader_module_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUShaderModule {
        let shader = device
            .core
            .create_shader_module(core::ShaderModuleSource::Spirv(Vec::new()));
        arc_to_handle(Arc::new(WGPUShaderModuleImpl {
            _core: Arc::new(shader),
            _device: Arc::clone(&device.core),
            _instance: Arc::clone(&device.instance),
        }))
    }

    fn bind_group_layout_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUBindGroupLayout {
        let layout = device
            .core
            .create_bind_group_layout(core::BindGroupLayoutDescriptor {
                entries: vec![core::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: native::WGPUShaderStage_Compute,
                    binding_array_size: 0,
                    kind: Some(core::BindingLayoutKind::Buffer {
                        ty: core::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: 4,
                    }),
                }],
                error: None,
            });
        arc_to_handle(Arc::new(WGPUBindGroupLayoutImpl {
            _core: Arc::new(layout),
            _device: Arc::clone(&device.core),
            _instance: Arc::clone(&device.instance),
        }))
    }

    fn pipeline_layout_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUPipelineLayout {
        let layout = device
            .core
            .create_pipeline_layout(core::PipelineLayoutDescriptor {
                bind_group_layouts: Vec::new(),
                immediate_size: 0,
                error: None,
            });
        arc_to_handle(Arc::new(WGPUPipelineLayoutImpl {
            _core: Arc::new(layout),
            _device: Arc::clone(&device.core),
            _instance: Arc::clone(&device.instance),
        }))
    }

    fn texture_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUTexture {
        let texture = device.core.create_texture(core::TextureDescriptor {
            usage: core::TextureUsage::TEXTURE_BINDING | core::TextureUsage::RENDER_ATTACHMENT,
            dimension: core::TextureDimension::D2,
            size: core::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm.into(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        arc_to_handle(Arc::new(WGPUTextureImpl {
            core: Arc::new(texture),
            device: Arc::clone(&device.core),
            instance: Arc::clone(&device.instance),
        }))
    }

    fn texture_view_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUTextureView {
        let texture = Arc::new(device.core.create_texture(core::TextureDescriptor {
            usage: core::TextureUsage::TEXTURE_BINDING | core::TextureUsage::RENDER_ATTACHMENT,
            dimension: core::TextureDimension::D2,
            size: core::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm.into(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let (view, _) = texture.create_view(core::TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
        });
        arc_to_handle(Arc::new(WGPUTextureViewImpl {
            _core: Arc::new(view),
            _texture: texture,
            _device: Arc::clone(&device.core),
            _instance: Arc::clone(&device.instance),
        }))
    }

    #[derive(Debug)]
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn arc_to_handle_round_trips_with_clone_handle_refcount_math() {
        let value = Arc::new(7_u32);
        let handle = arc_to_handle(Arc::clone(&value));
        assert_eq!(Arc::strong_count(&value), 2);
        let cloned = unsafe { clone_handle(handle, "u32") };
        assert_eq!(*cloned, 7);
        assert_eq!(Arc::strong_count(&value), 3);
        drop(cloned);
        assert_eq!(Arc::strong_count(&value), 2);
        unsafe { release_handle(handle, "u32") };
        assert_eq!(Arc::strong_count(&value), 1);
    }

    #[test]
    fn release_handle_drops_owned_reference_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let handle = arc_to_handle(Arc::new(DropCounter(Arc::clone(&drops))));
        unsafe { release_handle(handle, "DropCounter") };
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn add_ref_handle_increments_refcount_for_later_release() {
        let value = Arc::new(11_u32);
        let handle = arc_to_handle(Arc::clone(&value));
        unsafe { add_ref_handle(handle, "u32") };
        assert_eq!(Arc::strong_count(&value), 3);
        unsafe { release_handle(handle, "u32") };
        assert_eq!(Arc::strong_count(&value), 2);
        unsafe { release_handle(handle, "u32") };
        assert_eq!(Arc::strong_count(&value), 1);
    }

    #[test]
    #[should_panic(expected = "WGPUInstance must not be null")]
    fn release_handle_null_panics_with_contract_message() {
        unsafe {
            release_handle::<core::Instance>(std::ptr::null(), "WGPUInstance");
        }
    }

    #[test]
    #[should_panic(expected = "WGPUInstance must not be null")]
    fn add_ref_handle_null_panics_with_contract_message() {
        unsafe {
            add_ref_handle::<core::Instance>(std::ptr::null(), "WGPUInstance");
        }
    }

    #[test]
    fn clone_handle_leaves_original_handle_valid() {
        let value = Arc::new(13_u32);
        let handle = arc_to_handle(Arc::clone(&value));
        let cloned = unsafe { clone_handle(handle, "u32") };
        assert_eq!(unsafe { *borrow_handle::<u32>(handle, "u32") }, 13);
        drop(cloned);
        unsafe { release_handle(handle, "u32") };
    }

    #[test]
    fn borrow_handle_returns_reference_without_consuming_arc() {
        let value = Arc::new(17_u32);
        let handle = arc_to_handle(Arc::clone(&value));
        let borrowed = unsafe { borrow_handle::<u32>(handle, "u32") };
        assert_eq!(*borrowed, 17);
        assert!(std::ptr::eq(borrowed, Arc::as_ptr(&value)));
        assert_eq!(Arc::strong_count(&value), 2);
        unsafe { release_handle(handle, "u32") };
    }

    #[test]
    #[should_panic(expected = "WGPUBuffer must not be null")]
    fn clone_handle_null_panics_with_contract_message() {
        let _ = unsafe { clone_handle::<WGPUBufferImpl>(std::ptr::null(), "WGPUBuffer") };
    }

    #[test]
    #[should_panic(expected = "WGPUBuffer must not be null")]
    fn borrow_handle_null_panics_with_contract_message() {
        let _ = unsafe { borrow_handle::<WGPUBufferImpl>(std::ptr::null(), "WGPUBuffer") };
    }

    #[test]
    fn string_view_round_trips_data_and_empty_slice() {
        let view = string_view(b"hello");
        assert_eq!(view.length, 5);
        assert_eq!(unsafe { string_view_to_str(view) }, Some("hello"));
        let empty = string_view(b"");
        assert_eq!(empty.length, 0);
    }

    #[test]
    fn string_view_to_str_handles_explicit_strlen_and_null_data() {
        let direct = string_view(b"abc");
        assert_eq!(unsafe { string_view_to_str(direct) }, Some("abc"));
        let c_string = CString::new("auto").expect("CString");
        let auto = native::WGPUStringView {
            data: c_string.as_ptr(),
            length: WGPU_STRLEN,
        };
        assert_eq!(unsafe { string_view_to_str(auto) }, Some("auto"));
        assert_eq!(unsafe { string_view_to_str(empty_string_view()) }, None);
    }

    #[test]
    fn label_from_string_view_returns_owned_label_or_none() {
        assert_eq!(
            unsafe { label_from_string_view(string_view(b"label")) },
            Some("label".to_owned())
        );
        assert_eq!(unsafe { label_from_string_view(empty_string_view()) }, None);
    }

    #[test]
    fn map_feature_round_trips_defined_and_other_variants() {
        for value in [
            native::WGPUFeatureName_CoreFeaturesAndLimits,
            native::WGPUFeatureName_RG11B10UfloatRenderable,
            native::WGPUFeatureName_TimestampQuery,
            native::WGPUFeatureName_TextureFormatsTier1,
            native::WGPUFeatureName_TextureFormatsTier2,
            0xCAFE,
        ] {
            assert_eq!(map_feature_to_native(map_feature(value)), value);
        }
    }

    #[test]
    fn map_query_type_round_trips_defined_and_unknown_variants() {
        for value in [
            native::WGPUQueryType_Occlusion,
            native::WGPUQueryType_Timestamp,
            0xCAFE,
        ] {
            assert_eq!(map_query_type_to_native(map_query_type(value)), value);
        }
    }

    #[test]
    fn from_native_query_type_round_trips_known_and_unknown_variants() {
        let known = core::QueryType::from(native::WGPUQueryType_Occlusion);
        assert_eq!(known, core::QueryType::Occlusion);
        assert_eq!(
            Into::<native::WGPUQueryType>::into(known),
            native::WGPUQueryType_Occlusion
        );

        let unknown_native = 0xFFFF_u32 as native::WGPUQueryType;
        let unknown = core::QueryType::from(unknown_native);
        assert_eq!(unknown, core::QueryType::Unknown(0xFFFF));
        assert_eq!(Into::<native::WGPUQueryType>::into(unknown), unknown_native);
    }

    #[test]
    fn map_buffer_usage_round_trips_bitmask() {
        let usage = native::WGPUBufferUsage_MapRead
            | native::WGPUBufferUsage_CopyDst
            | native::WGPUBufferUsage_Uniform
            | 0x8000_0000;
        assert_eq!(map_buffer_usage_to_native(map_buffer_usage(usage)), usage);
    }

    #[test]
    fn map_texture_usage_round_trips_bitmask() {
        let usage = native::WGPUTextureUsage_CopySrc
            | native::WGPUTextureUsage_TextureBinding
            | native::WGPUTextureUsage_RenderAttachment
            | 0x8000_0000;
        assert_eq!(map_texture_usage_to_native(map_texture_usage(usage)), usage);
    }

    #[test]
    fn map_texture_dimension_round_trips_defined_variants() {
        for (native_value, core_value) in [
            (native::WGPUTextureDimension_1D, core::TextureDimension::D1),
            (native::WGPUTextureDimension_2D, core::TextureDimension::D2),
            (native::WGPUTextureDimension_3D, core::TextureDimension::D3),
        ] {
            assert_eq!(map_texture_dimension(native_value), core_value);
            assert_eq!(map_texture_dimension_to_native(core_value), native_value);
        }
        assert_eq!(
            map_texture_dimension(native::WGPUTextureDimension_Undefined),
            core::TextureDimension::D2
        );
    }

    #[test]
    fn map_texture_format_round_trips_defined_and_unknown_raw_values() {
        for value in [
            native::WGPUTextureFormat_Undefined,
            native::WGPUTextureFormat_R8Unorm,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_BGRA8Unorm,
            0xCAFE,
        ] {
            assert_eq!(
                map_texture_format_to_native(map_texture_format(value)),
                value
            );
        }
    }

    #[test]
    fn from_native_texture_format_round_trips_known_and_unknown_variants() {
        let known = core::TextureFormat::from(native::WGPUTextureFormat_RGBA8Unorm);
        assert_eq!(known, core::TextureFormat::from_raw(0x16));
        assert_eq!(
            Into::<native::WGPUTextureFormat>::into(known),
            native::WGPUTextureFormat_RGBA8Unorm
        );

        let unknown_native = 0xFFFF_u32 as native::WGPUTextureFormat;
        let unknown = core::TextureFormat::from(unknown_native);
        assert_eq!(unknown.raw(), 0xFFFF);
        assert_eq!(
            Into::<native::WGPUTextureFormat>::into(unknown),
            unknown_native
        );
    }

    #[test]
    fn from_native_vertex_format_round_trips_known_and_unknown_variants() {
        let known = core::VertexFormat::from(native::WGPUVertexFormat_Float32x2);
        assert_eq!(known, core::VertexFormat::from_raw(0x1D));
        assert_eq!(map_vertex_format(native::WGPUVertexFormat_Float32x2), known);
        assert_eq!(
            Into::<native::WGPUVertexFormat>::into(known),
            native::WGPUVertexFormat_Float32x2
        );
        assert_eq!(
            map_vertex_format_to_native(known),
            native::WGPUVertexFormat_Float32x2
        );

        let unknown_native = 0xFFFF_u32 as native::WGPUVertexFormat;
        let unknown = core::VertexFormat::from(unknown_native);
        assert_eq!(unknown.raw(), 0xFFFF);
        assert_eq!(
            Into::<native::WGPUVertexFormat>::into(unknown),
            unknown_native
        );
    }

    #[test]
    fn map_feature_level_maps_compatibility_and_default_core() {
        assert_eq!(
            map_feature_level(native::WGPUFeatureLevel_Compatibility),
            core::FeatureLevel::Compatibility
        );
        assert_eq!(
            map_feature_level(native::WGPUFeatureLevel_Core),
            core::FeatureLevel::Core
        );
        assert_eq!(
            map_feature_level(native::WGPUFeatureLevel_Undefined),
            core::FeatureLevel::Core
        );
    }

    #[test]
    fn map_device_lost_reason_maps_every_core_variant() {
        assert_eq!(
            map_device_lost_reason(core::DeviceLostReason::Unknown),
            native::WGPUDeviceLostReason_Unknown
        );
        assert_eq!(
            map_device_lost_reason(core::DeviceLostReason::Destroyed),
            native::WGPUDeviceLostReason_Destroyed
        );
        assert_eq!(
            map_device_lost_reason(core::DeviceLostReason::CallbackCancelled),
            native::WGPUDeviceLostReason_CallbackCancelled
        );
        assert_eq!(
            map_device_lost_reason(core::DeviceLostReason::FailedCreation),
            native::WGPUDeviceLostReason_FailedCreation
        );
    }

    #[test]
    fn map_error_filter_maps_known_values_and_rejects_unknown() {
        assert_eq!(
            map_error_filter(native::WGPUErrorFilter_Validation),
            Some(core::ErrorFilter::Validation)
        );
        assert_eq!(
            map_error_filter(native::WGPUErrorFilter_OutOfMemory),
            Some(core::ErrorFilter::OutOfMemory)
        );
        assert_eq!(
            map_error_filter(native::WGPUErrorFilter_Internal),
            Some(core::ErrorFilter::Internal)
        );
        assert_eq!(map_error_filter(0xCAFE), None);
    }

    #[test]
    fn map_error_type_maps_every_core_variant() {
        assert_eq!(
            map_error_type(core::ErrorKind::Validation),
            native::WGPUErrorType_Validation
        );
        assert_eq!(
            map_error_type(core::ErrorKind::OutOfMemory),
            native::WGPUErrorType_OutOfMemory
        );
        assert_eq!(
            map_error_type(core::ErrorKind::Internal),
            native::WGPUErrorType_Internal
        );
    }

    #[test]
    fn map_pop_error_scope_status_error_returns_error() {
        assert_eq!(
            map_pop_error_scope_status_error(),
            native::WGPUPopErrorScopeStatus_Error
        );
    }

    #[test]
    fn map_pop_error_scope_status_success_returns_success() {
        assert_eq!(
            map_pop_error_scope_status_success(),
            native::WGPUPopErrorScopeStatus_Success
        );
    }

    #[test]
    fn map_buffer_map_state_maps_every_core_variant() {
        assert_eq!(
            map_buffer_map_state(core::BufferMapState::Unmapped),
            native::WGPUBufferMapState_Unmapped
        );
        assert_eq!(
            map_buffer_map_state(core::BufferMapState::Pending),
            native::WGPUBufferMapState_Pending
        );
        assert_eq!(
            map_buffer_map_state(core::BufferMapState::Mapped),
            native::WGPUBufferMapState_Mapped
        );
    }

    #[test]
    fn map_map_async_status_maps_every_core_variant() {
        assert_eq!(
            map_map_async_status(core::MapAsyncStatus::Success),
            native::WGPUMapAsyncStatus_Success
        );
        assert_eq!(
            map_map_async_status(core::MapAsyncStatus::CallbackCancelled),
            native::WGPUMapAsyncStatus_CallbackCancelled
        );
        assert_eq!(
            map_map_async_status(core::MapAsyncStatus::Error),
            native::WGPUMapAsyncStatus_Error
        );
        assert_eq!(
            map_map_async_status(core::MapAsyncStatus::Aborted),
            native::WGPUMapAsyncStatus_Aborted
        );
    }

    #[test]
    fn map_queue_work_done_status_maps_every_core_variant() {
        assert_eq!(
            map_queue_work_done_status(core::QueueWorkDoneStatus::Success),
            native::WGPUQueueWorkDoneStatus_Success
        );
        assert_eq!(
            map_queue_work_done_status(core::QueueWorkDoneStatus::CallbackCancelled),
            native::WGPUQueueWorkDoneStatus_CallbackCancelled
        );
        assert_eq!(
            map_queue_work_done_status(core::QueueWorkDoneStatus::Error),
            native::WGPUQueueWorkDoneStatus_Error
        );
    }

    #[test]
    fn map_compilation_info_request_status_success_returns_success() {
        assert_eq!(
            map_compilation_info_request_status_success(),
            native::WGPUCompilationInfoRequestStatus_Success
        );
    }

    #[test]
    fn map_compilation_message_type_error_returns_error() {
        assert_eq!(
            map_compilation_message_type_error(),
            native::WGPUCompilationMessageType_Error
        );
    }

    #[test]
    fn map_map_mode_accepts_single_modes_and_rejects_invalid_combinations() {
        assert_eq!(
            map_map_mode(native::WGPUMapMode_Read),
            Ok(core::MapMode::Read)
        );
        assert_eq!(
            map_map_mode(native::WGPUMapMode_Write),
            Ok(core::MapMode::Write)
        );
        assert!(map_map_mode(native::WGPUMapMode_Read | native::WGPUMapMode_Write).is_err());
        assert!(map_map_mode(native::WGPUMapMode_None).is_err());
        assert!(map_map_mode(0x8000_0000).is_err());
    }

    #[test]
    fn map_address_mode_maps_known_values_and_rejects_unknown() {
        assert_eq!(map_address_mode(native::WGPUAddressMode_Undefined), None);
        assert_eq!(
            map_address_mode(native::WGPUAddressMode_ClampToEdge),
            Some(core::AddressMode::ClampToEdge)
        );
        assert_eq!(
            map_address_mode(native::WGPUAddressMode_Repeat),
            Some(core::AddressMode::Repeat)
        );
        assert_eq!(
            map_address_mode(native::WGPUAddressMode_MirrorRepeat),
            Some(core::AddressMode::MirrorRepeat)
        );
        assert_eq!(map_address_mode(0xCAFE), None);
    }

    #[test]
    fn map_filter_mode_maps_known_values_and_rejects_unknown() {
        assert_eq!(map_filter_mode(native::WGPUFilterMode_Undefined), None);
        assert_eq!(
            map_filter_mode(native::WGPUFilterMode_Nearest),
            Some(core::FilterMode::Nearest)
        );
        assert_eq!(
            map_filter_mode(native::WGPUFilterMode_Linear),
            Some(core::FilterMode::Linear)
        );
        assert_eq!(map_filter_mode(0xCAFE), None);
    }

    #[test]
    fn map_mipmap_filter_mode_maps_known_values_and_rejects_unknown() {
        assert_eq!(
            map_mipmap_filter_mode(native::WGPUMipmapFilterMode_Undefined),
            None
        );
        assert_eq!(
            map_mipmap_filter_mode(native::WGPUMipmapFilterMode_Nearest),
            Some(core::MipmapFilterMode::Nearest)
        );
        assert_eq!(
            map_mipmap_filter_mode(native::WGPUMipmapFilterMode_Linear),
            Some(core::MipmapFilterMode::Linear)
        );
        assert_eq!(map_mipmap_filter_mode(0xCAFE), None);
    }

    #[test]
    fn map_compare_function_maps_known_values_and_rejects_undefined() {
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Undefined),
            None
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Never),
            Some(core::CompareFunction::Never)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Less),
            Some(core::CompareFunction::Less)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Equal),
            Some(core::CompareFunction::Equal)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_LessEqual),
            Some(core::CompareFunction::LessEqual)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Greater),
            Some(core::CompareFunction::Greater)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_NotEqual),
            Some(core::CompareFunction::NotEqual)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_GreaterEqual),
            Some(core::CompareFunction::GreaterEqual)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Always),
            Some(core::CompareFunction::Always)
        );
        assert_eq!(map_compare_function(0xCAFE), None);
    }

    #[test]
    fn map_texture_view_dimension_maps_known_values_and_rejects_unknown() {
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_Undefined),
            None
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_1D),
            Some(core::TextureViewDimension::D1)
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_2D),
            Some(core::TextureViewDimension::D2)
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_2DArray),
            Some(core::TextureViewDimension::D2Array)
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_Cube),
            Some(core::TextureViewDimension::Cube)
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_CubeArray),
            Some(core::TextureViewDimension::CubeArray)
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_3D),
            Some(core::TextureViewDimension::D3)
        );
        assert_eq!(map_texture_view_dimension(0xCAFE), None);
    }

    #[test]
    fn map_texture_aspect_maps_known_values_and_rejects_undefined() {
        assert_eq!(
            map_texture_aspect(native::WGPUTextureAspect_Undefined),
            None
        );
        assert_eq!(
            map_texture_aspect(native::WGPUTextureAspect_All),
            Some(core::TextureAspect::All)
        );
        assert_eq!(
            map_texture_aspect(native::WGPUTextureAspect_DepthOnly),
            Some(core::TextureAspect::DepthOnly)
        );
        assert_eq!(
            map_texture_aspect(native::WGPUTextureAspect_StencilOnly),
            Some(core::TextureAspect::StencilOnly)
        );
        assert_eq!(map_texture_aspect(0xCAFE), None);
    }

    #[test]
    fn map_load_op_maps_defined_values_and_undefined_fallback() {
        assert_eq!(map_load_op(native::WGPULoadOp_Load), core::LoadOp::Load);
        assert_eq!(map_load_op(native::WGPULoadOp_Clear), core::LoadOp::Clear);
        assert_eq!(
            map_load_op(native::WGPULoadOp_Undefined),
            core::LoadOp::Undefined
        );
    }

    #[test]
    fn map_store_op_maps_defined_values_and_undefined_fallback() {
        assert_eq!(
            map_store_op(native::WGPUStoreOp_Store),
            core::StoreOp::Store
        );
        assert_eq!(
            map_store_op(native::WGPUStoreOp_Discard),
            core::StoreOp::Discard
        );
        assert_eq!(
            map_store_op(native::WGPUStoreOp_Undefined),
            core::StoreOp::Undefined
        );
    }

    #[test]
    fn map_query_index_maps_defined_values_and_undefined_to_none() {
        assert_eq!(map_query_index(3), Some(3));
        assert_eq!(
            map_query_index(native::WGPU_QUERY_SET_INDEX_UNDEFINED),
            None
        );
    }

    #[test]
    fn has_callback_detects_present_and_absent_device_lost_callbacks() {
        unsafe extern "C" fn callback(
            _device: *const native::WGPUDevice,
            _reason: native::WGPUDeviceLostReason,
            _message: native::WGPUStringView,
            _userdata1: *mut c_void,
            _userdata2: *mut c_void,
        ) {
        }

        let with_callback = DeviceLostCallbackInfo {
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(callback),
            userdata1: 1,
            userdata2: 2,
        };
        let without_callback = DeviceLostCallbackInfo {
            callback: None,
            ..with_callback
        };
        assert!(with_callback.has_callback());
        assert!(!without_callback.has_callback());
    }

    #[test]
    fn map_buffer_descriptor_round_trips_fields() {
        let descriptor = native::WGPUBufferDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: string_view(b"buffer"),
            usage: native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_Uniform,
            size: 4096,
            mappedAtCreation: 1,
        };
        let mapped = map_buffer_descriptor(&descriptor);
        assert_eq!(mapped.usage.bits(), descriptor.usage);
        assert_eq!(mapped.size, 4096);
        assert!(mapped.mapped_at_creation);
    }

    #[test]
    fn map_sampler_descriptor_round_trips_fields_with_undefined_compare() {
        let descriptor = native::WGPUSamplerDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            addressModeU: native::WGPUAddressMode_ClampToEdge,
            addressModeV: native::WGPUAddressMode_Repeat,
            addressModeW: native::WGPUAddressMode_MirrorRepeat,
            magFilter: native::WGPUFilterMode_Linear,
            minFilter: native::WGPUFilterMode_Nearest,
            mipmapFilter: native::WGPUMipmapFilterMode_Linear,
            lodMinClamp: 1.0,
            lodMaxClamp: 8.0,
            compare: native::WGPUCompareFunction_Undefined,
            maxAnisotropy: 4,
        };
        let mapped = map_sampler_descriptor(Some(&descriptor));
        assert_eq!(mapped.address_mode_u, Some(core::AddressMode::ClampToEdge));
        assert_eq!(mapped.address_mode_v, Some(core::AddressMode::Repeat));
        assert_eq!(mapped.address_mode_w, Some(core::AddressMode::MirrorRepeat));
        assert_eq!(mapped.mag_filter, Some(core::FilterMode::Linear));
        assert_eq!(mapped.min_filter, Some(core::FilterMode::Nearest));
        assert_eq!(mapped.mipmap_filter, Some(core::MipmapFilterMode::Linear));
        assert_eq!(mapped.lod_min_clamp, 1.0);
        assert_eq!(mapped.lod_max_clamp, 8.0);
        assert_eq!(mapped.compare, None);
        assert_eq!(mapped.max_anisotropy, 4);
    }

    #[test]
    fn map_extent_3d_round_trips_fields() {
        let mapped = map_extent_3d(native::WGPUExtent3D {
            width: 1,
            height: 2,
            depthOrArrayLayers: 3,
        });
        assert_eq!(
            mapped,
            core::Extent3d {
                width: 1,
                height: 2,
                depth_or_array_layers: 3
            }
        );
    }

    #[test]
    fn map_origin_3d_round_trips_fields() {
        assert_eq!(
            map_origin_3d(native::WGPUOrigin3D { x: 4, y: 5, z: 6 }),
            core::Origin3d { x: 4, y: 5, z: 6 }
        );
    }

    #[test]
    fn map_color_round_trips_float_bits_including_nan() {
        let nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let mapped = map_color(native::WGPUColor {
            r: 1.0,
            g: -2.0,
            b: nan,
            a: 4.0,
        });
        assert_eq!(mapped.r.to_bits(), 1.0f64.to_bits());
        assert_eq!(mapped.g.to_bits(), (-2.0f64).to_bits());
        assert_eq!(mapped.b.to_bits(), nan.to_bits());
        assert_eq!(mapped.a.to_bits(), 4.0f64.to_bits());
    }

    #[test]
    fn map_texel_copy_buffer_layout_round_trips_fields_and_undefined_strides() {
        let mapped = map_texel_copy_buffer_layout(native::WGPUTexelCopyBufferLayout {
            offset: 64,
            bytesPerRow: 256,
            rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
        });
        assert_eq!(mapped.offset, 64);
        assert_eq!(mapped.bytes_per_row, Some(256));
        assert_eq!(mapped.rows_per_image, None);
    }

    #[test]
    fn map_texel_copy_texture_info_parts_round_trips_fields() {
        let device = device_impl();
        let texture = texture_handle(&device);
        let value = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: 2,
            origin: native::WGPUOrigin3D { x: 1, y: 2, z: 3 },
            aspect: native::WGPUTextureAspect_DepthOnly,
        };
        let (mip_level, origin, aspect) = map_texel_copy_texture_info_parts(&value);
        assert_eq!(mip_level, 2);
        assert_eq!(origin, core::Origin3d { x: 1, y: 2, z: 3 });
        assert_eq!(aspect, core::TextureAspect::DepthOnly);
        unsafe { release_handle(texture, "WGPUTexture") };
    }

    fn distinct_limits() -> native::WGPULimits {
        native::WGPULimits {
            nextInChain: std::ptr::null_mut(),
            maxTextureDimension1D: 101,
            maxTextureDimension2D: 102,
            maxTextureDimension3D: 103,
            maxTextureArrayLayers: 104,
            maxBindGroups: 105,
            maxBindGroupsPlusVertexBuffers: 106,
            maxBindingsPerBindGroup: 107,
            maxDynamicUniformBuffersPerPipelineLayout: 108,
            maxDynamicStorageBuffersPerPipelineLayout: 109,
            maxSampledTexturesPerShaderStage: 110,
            maxSamplersPerShaderStage: 111,
            maxStorageBuffersPerShaderStage: 112,
            maxStorageTexturesPerShaderStage: 113,
            maxUniformBuffersPerShaderStage: 114,
            maxUniformBufferBindingSize: 115,
            maxStorageBufferBindingSize: 116,
            minUniformBufferOffsetAlignment: 117,
            minStorageBufferOffsetAlignment: 118,
            maxVertexBuffers: 119,
            maxBufferSize: 120,
            maxVertexAttributes: 121,
            maxVertexBufferArrayStride: 122,
            maxInterStageShaderVariables: 123,
            maxColorAttachments: 124,
            maxColorAttachmentBytesPerSample: 125,
            maxComputeWorkgroupStorageSize: 126,
            maxComputeInvocationsPerWorkgroup: 127,
            maxComputeWorkgroupSizeX: 128,
            maxComputeWorkgroupSizeY: 129,
            maxComputeWorkgroupSizeZ: 130,
            maxComputeWorkgroupsPerDimension: 131,
            maxImmediateSize: 132,
        }
    }

    #[test]
    fn map_limits_round_trips_every_field_from_native() {
        let mapped = map_limits(&distinct_limits());
        assert_eq!(mapped.max_texture_dimension_1d, 101);
        assert_eq!(mapped.max_texture_dimension_2d, 102);
        assert_eq!(mapped.max_texture_dimension_3d, 103);
        assert_eq!(mapped.max_texture_array_layers, 104);
        assert_eq!(mapped.max_bind_groups, 105);
        assert_eq!(mapped.max_bind_groups_plus_vertex_buffers, 106);
        assert_eq!(mapped.max_bindings_per_bind_group, 107);
        assert_eq!(mapped.max_dynamic_uniform_buffers_per_pipeline_layout, 108);
        assert_eq!(mapped.max_dynamic_storage_buffers_per_pipeline_layout, 109);
        assert_eq!(mapped.max_sampled_textures_per_shader_stage, 110);
        assert_eq!(mapped.max_samplers_per_shader_stage, 111);
        assert_eq!(mapped.max_storage_buffers_per_shader_stage, 112);
        assert_eq!(mapped.max_storage_textures_per_shader_stage, 113);
        assert_eq!(mapped.max_uniform_buffers_per_shader_stage, 114);
        assert_eq!(mapped.max_uniform_buffer_binding_size, 115);
        assert_eq!(mapped.max_storage_buffer_binding_size, 116);
        assert_eq!(mapped.min_uniform_buffer_offset_alignment, 117);
        assert_eq!(mapped.min_storage_buffer_offset_alignment, 118);
        assert_eq!(mapped.max_vertex_buffers, 119);
        assert_eq!(mapped.max_buffer_size, 120);
        assert_eq!(mapped.max_vertex_attributes, 121);
        assert_eq!(mapped.max_vertex_buffer_array_stride, 122);
        assert_eq!(mapped.max_inter_stage_shader_variables, 123);
        assert_eq!(mapped.max_color_attachments, 124);
        assert_eq!(mapped.max_color_attachment_bytes_per_sample, 125);
        assert_eq!(mapped.max_compute_workgroup_storage_size, 126);
        assert_eq!(mapped.max_compute_invocations_per_workgroup, 127);
        assert_eq!(mapped.max_compute_workgroup_size_x, 128);
        assert_eq!(mapped.max_compute_workgroup_size_y, 129);
        assert_eq!(mapped.max_compute_workgroup_size_z, 130);
        assert_eq!(mapped.max_compute_workgroups_per_dimension, 131);
        assert_eq!(mapped.max_immediate_size, 132);
    }

    #[test]
    fn map_limits_to_native_round_trips_through_map_limits() {
        let limits = map_limits(&distinct_limits());
        let native = map_limits_to_native(limits);
        assert_eq!(map_limits(&native), limits);
    }

    #[test]
    fn map_features_to_native_allocates_feature_array_and_free_supported_features_releases_it() {
        let features = [
            core::Feature::TimestampQuery,
            core::Feature::TextureFormatsTier1,
        ]
        .into_iter()
        .collect::<core::FeatureSet>();
        let native_features = map_features_to_native(&features);
        assert_eq!(native_features.featureCount, 2);
        let slice = unsafe {
            std::slice::from_raw_parts(native_features.features, native_features.featureCount)
        };
        let found = slice.iter().copied().collect::<HashSet<_>>();
        assert!(found.contains(&native::WGPUFeatureName_TimestampQuery));
        assert!(found.contains(&native::WGPUFeatureName_TextureFormatsTier1));
        unsafe { free_supported_features(native_features) };
    }

    #[test]
    fn free_supported_features_accepts_null_feature_array() {
        unsafe {
            free_supported_features(native::WGPUSupportedFeatures {
                featureCount: 0,
                features: std::ptr::null(),
            })
        };
    }

    #[test]
    fn map_shader_module_descriptor_decodes_wgsl_source_and_missing_source_error() {
        let mut wgsl = native::WGPUShaderSourceWGSL {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_ShaderSourceWGSL,
            },
            code: string_view(b"@compute @workgroup_size(1) fn main() {}"),
        };
        let descriptor = native::WGPUShaderModuleDescriptor {
            nextInChain: (&mut wgsl.chain) as *mut native::WGPUChainedStruct,
            label: empty_string_view(),
        };
        match unsafe { map_shader_module_descriptor(&descriptor) } {
            core::ShaderModuleSource::Wgsl(source) => assert!(source.contains("fn main")),
            other => panic!("unexpected source: {other:?}"),
        }
        let missing = native::WGPUShaderModuleDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        };
        match unsafe { map_shader_module_descriptor(&missing) } {
            core::ShaderModuleSource::Invalid(message) => {
                assert!(message.contains("exactly one shader source"));
            }
            other => panic!("unexpected source: {other:?}"),
        }
    }

    fn buffer_binding_layout() -> native::WGPUBufferBindingLayout {
        native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_Uniform,
            hasDynamicOffset: 1,
            minBindingSize: 16,
        }
    }

    fn unused_sampler_layout() -> native::WGPUSamplerBindingLayout {
        native::WGPUSamplerBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUSamplerBindingType_BindingNotUsed,
        }
    }

    fn unused_texture_layout() -> native::WGPUTextureBindingLayout {
        native::WGPUTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            sampleType: native::WGPUTextureSampleType_BindingNotUsed,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
            multisampled: 0,
        }
    }

    fn unused_storage_texture_layout() -> native::WGPUStorageTextureBindingLayout {
        native::WGPUStorageTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            access: native::WGPUStorageTextureAccess_BindingNotUsed,
            format: native::WGPUTextureFormat_Undefined,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
        }
    }

    #[test]
    fn map_bind_group_layout_descriptor_decodes_buffer_entry_and_null_entries_error() {
        let entry = native::WGPUBindGroupLayoutEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 7,
            visibility: native::WGPUShaderStage_Vertex | native::WGPUShaderStage_Compute,
            bindingArraySize: 2,
            buffer: buffer_binding_layout(),
            sampler: unused_sampler_layout(),
            texture: unused_texture_layout(),
            storageTexture: unused_storage_texture_layout(),
        };
        let descriptor = native::WGPUBindGroupLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            entryCount: 1,
            entries: &entry,
        };
        let mapped = unsafe { map_bind_group_layout_descriptor(&descriptor) };
        assert_eq!(mapped.error, None);
        assert_eq!(mapped.entries.len(), 1);
        assert_eq!(mapped.entries[0].binding, 7);
        assert_eq!(mapped.entries[0].visibility, entry.visibility);
        assert!(matches!(
            mapped.entries[0].kind,
            Some(core::BindingLayoutKind::Buffer {
                ty: core::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: 16
            })
        ));

        let invalid = native::WGPUBindGroupLayoutDescriptor {
            entryCount: 1,
            entries: std::ptr::null(),
            ..descriptor
        };
        assert!(unsafe { map_bind_group_layout_descriptor(&invalid) }
            .error
            .expect("error")
            .contains("must not be null"));
    }

    #[test]
    fn map_bind_group_entries_decodes_buffer_entry_and_null_entries_error() {
        let device = device_impl();
        let buffer = buffer_handle(&device);
        let entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 3,
            buffer,
            offset: 4,
            size: 8,
            sampler: std::ptr::null(),
            textureView: std::ptr::null(),
        };
        let descriptor = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null(),
            entryCount: 1,
            entries: &entry,
        };
        let mapped = unsafe { map_bind_group_entries(&descriptor) };
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].binding, 3);
        assert!(matches!(
            mapped[0].resource,
            core::BindGroupResource::Buffer {
                offset: 4,
                size: 8,
                ..
            }
        ));

        let invalid = native::WGPUBindGroupDescriptor {
            entryCount: 1,
            entries: std::ptr::null(),
            ..descriptor
        };
        assert!(matches!(
            unsafe { map_bind_group_entries(&invalid) }[0].resource,
            core::BindGroupResource::Invalid(_)
        ));
        unsafe { release_handle(buffer, "WGPUBuffer") };
    }

    #[test]
    fn map_pipeline_layout_descriptor_decodes_layouts_and_null_array_error() {
        let device = device_impl();
        let layout = bind_group_layout_handle(&device);
        let layouts = [layout];
        let descriptor = native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            bindGroupLayoutCount: 1,
            bindGroupLayouts: layouts.as_ptr(),
            immediateSize: 32,
        };
        let mapped = unsafe { map_pipeline_layout_descriptor(&descriptor) };
        assert_eq!(mapped.bind_group_layouts.len(), 1);
        assert_eq!(mapped.immediate_size, 32);
        assert_eq!(mapped.error, None);

        let invalid = native::WGPUPipelineLayoutDescriptor {
            bindGroupLayouts: std::ptr::null(),
            ..descriptor
        };
        assert!(unsafe { map_pipeline_layout_descriptor(&invalid) }
            .error
            .expect("error")
            .contains("must not be null"));
        unsafe { release_handle(layout, "WGPUBindGroupLayout") };
    }

    #[test]
    #[should_panic(expected = "WGPUShaderModule must not be null")]
    fn map_compute_pipeline_descriptor_null_module_panics() {
        let descriptor = native::WGPUComputePipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null(),
            compute: native::WGPUComputeState {
                nextInChain: std::ptr::null_mut(),
                module: std::ptr::null(),
                entryPoint: string_view(b"main"),
                constantCount: 0,
                constants: std::ptr::null(),
            },
        };
        let _ = unsafe { map_compute_pipeline_descriptor(&descriptor) };
    }

    #[test]
    fn map_compute_pipeline_descriptor_decodes_module_entry_layout_and_constants() {
        let device = device_impl();
        let shader = shader_module_handle(&device);
        let layout = pipeline_layout_handle(&device);
        let constant = native::WGPUConstantEntry {
            nextInChain: std::ptr::null_mut(),
            key: string_view(b"X"),
            value: 2.5,
        };
        let descriptor = native::WGPUComputePipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout,
            compute: native::WGPUComputeState {
                nextInChain: std::ptr::null_mut(),
                module: shader,
                entryPoint: string_view(b"main"),
                constantCount: 1,
                constants: &constant,
            },
        };
        let mapped = unsafe { map_compute_pipeline_descriptor(&descriptor) };
        assert!(matches!(
            mapped.layout,
            core::ComputePipelineLayout::Explicit(_)
        ));
        assert_eq!(mapped.entry_point.as_deref(), Some("main"));
        assert_eq!(mapped.constants.len(), 1);
        assert_eq!(mapped.constants[0].key, "X");
        assert_eq!(mapped.constants[0].value, 2.5);
        assert_eq!(mapped.error, None);
        unsafe {
            release_handle(shader, "WGPUShaderModule");
            release_handle(layout, "WGPUPipelineLayout");
        }
    }

    fn primitive_state() -> native::WGPUPrimitiveState {
        native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: native::WGPUFrontFace_CCW,
            cullMode: native::WGPUCullMode_None,
            unclippedDepth: 0,
        }
    }

    #[test]
    fn map_render_pipeline_descriptor_decodes_vertex_fragment_and_error_path() {
        let device = device_impl();
        let shader = shader_module_handle(&device);
        let attribute = native::WGPUVertexAttribute {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUVertexFormat_Float32x2,
            offset: 0,
            shaderLocation: 1,
        };
        let vertex_buffer = native::WGPUVertexBufferLayout {
            nextInChain: std::ptr::null_mut(),
            stepMode: native::WGPUVertexStepMode_Vertex,
            arrayStride: 8,
            attributeCount: 1,
            attributes: &attribute,
        };
        let color_target = native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_RGBA8Unorm,
            blend: std::ptr::null(),
            writeMask: native::WGPUColorWriteMask_All,
        };
        let fragment = native::WGPUFragmentState {
            nextInChain: std::ptr::null_mut(),
            module: shader,
            entryPoint: string_view(b"fs_main"),
            constantCount: 0,
            constants: std::ptr::null(),
            targetCount: 1,
            targets: &color_target,
        };
        let descriptor = native::WGPURenderPipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null(),
            vertex: native::WGPUVertexState {
                nextInChain: std::ptr::null_mut(),
                module: shader,
                entryPoint: string_view(b"vs_main"),
                constantCount: 0,
                constants: std::ptr::null(),
                bufferCount: 1,
                buffers: &vertex_buffer,
            },
            primitive: primitive_state(),
            depthStencil: std::ptr::null(),
            multisample: native::WGPUMultisampleState {
                nextInChain: std::ptr::null_mut(),
                count: 1,
                mask: u32::MAX,
                alphaToCoverageEnabled: 0,
            },
            fragment: &fragment,
        };
        let mapped = unsafe { map_render_pipeline_descriptor(&descriptor) };
        assert_eq!(mapped.vertex.shader.entry_point.as_deref(), Some("vs_main"));
        assert_eq!(mapped.vertex.buffer_count, 1);
        assert_eq!(mapped.vertex.buffers[0].array_stride, 8);
        assert_eq!(mapped.fragment.as_ref().expect("fragment").target_count, 1);
        assert_eq!(mapped.error, None);

        let invalid_vertex = native::WGPUVertexState {
            bufferCount: 1,
            buffers: std::ptr::null(),
            ..descriptor.vertex
        };
        let invalid = native::WGPURenderPipelineDescriptor {
            vertex: invalid_vertex,
            ..descriptor
        };
        assert!(unsafe { map_render_pipeline_descriptor(&invalid) }
            .error
            .expect("error")
            .contains("vertex buffers"));
        unsafe { release_handle(shader, "WGPUShaderModule") };
    }

    #[test]
    fn map_query_set_descriptor_decodes_type_count_label() {
        let descriptor = native::WGPUQuerySetDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: string_view(b"query-set"),
            type_: native::WGPUQueryType_Timestamp,
            count: 4,
        };
        let mapped = unsafe { map_query_set_descriptor(&descriptor) };
        assert_eq!(mapped.label, "query-set");
        assert_eq!(mapped.kind, core::QueryType::Timestamp);
        assert_eq!(mapped.count, 4);
    }

    #[test]
    fn map_render_pass_descriptor_decodes_color_attachment_and_sparse_null_view() {
        let device = device_impl();
        let view = texture_view_handle(&device);
        let attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 0.4,
            },
        };
        let descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 1,
            colorAttachments: &attachment,
            depthStencilAttachment: std::ptr::null(),
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let mapped = unsafe { map_render_pass_descriptor(&descriptor, 4) };
        let color = mapped.color_attachments[0].as_ref().expect("color");
        assert_eq!(color.load_op, core::LoadOp::Clear);
        assert_eq!(color.store_op, core::StoreOp::Store);
        assert_eq!(color.clear_value.g, 0.2);

        let null_attachment = native::WGPURenderPassColorAttachment {
            view: std::ptr::null(),
            ..attachment
        };
        let null_descriptor = native::WGPURenderPassDescriptor {
            colorAttachments: &null_attachment,
            ..descriptor
        };
        assert!(
            unsafe { map_render_pass_descriptor(&null_descriptor, 4) }.color_attachments[0]
                .is_none()
        );
        unsafe { release_handle(view, "WGPUTextureView") };
    }

    #[test]
    fn map_render_bundle_encoder_descriptor_decodes_formats_and_null_format_array() {
        let formats = [
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_Undefined,
        ];
        let descriptor = native::WGPURenderBundleEncoderDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorFormatCount: 2,
            colorFormats: formats.as_ptr(),
            depthStencilFormat: native::WGPUTextureFormat_Depth24Plus,
            sampleCount: 4,
            depthReadOnly: 1,
            stencilReadOnly: 0,
        };
        let mapped = unsafe { map_render_bundle_encoder_descriptor(&descriptor, 4) };
        assert_eq!(mapped.color_formats.len(), 2);
        assert_eq!(
            mapped.color_formats[0],
            Some(native::WGPUTextureFormat_RGBA8Unorm.into())
        );
        assert_eq!(mapped.color_formats[1], None);
        assert_eq!(
            mapped.depth_stencil_format,
            Some(native::WGPUTextureFormat_Depth24Plus.into())
        );
        assert_eq!(mapped.sample_count, 4);
        assert!(mapped.depth_read_only);

        let null_descriptor = native::WGPURenderBundleEncoderDescriptor {
            colorFormats: std::ptr::null(),
            ..descriptor
        };
        assert_eq!(
            unsafe { map_render_bundle_encoder_descriptor(&null_descriptor, 4) }.color_formats,
            vec![None, None]
        );
    }

    #[test]
    fn map_texture_descriptor_decodes_usage_format_dimension_size_and_view_formats() {
        let view_formats = [native::WGPUTextureFormat_RGBA8UnormSrgb];
        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_TextureBinding
                | native::WGPUTextureUsage_RenderAttachment,
            dimension: native::WGPUTextureDimension_3D,
            size: native::WGPUExtent3D {
                width: 8,
                height: 9,
                depthOrArrayLayers: 10,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm,
            mipLevelCount: 3,
            sampleCount: 1,
            viewFormatCount: 1,
            viewFormats: view_formats.as_ptr(),
        };
        let mapped = unsafe { map_texture_descriptor(&descriptor) };
        assert_eq!(mapped.usage.bits(), descriptor.usage);
        assert_eq!(mapped.dimension, core::TextureDimension::D3);
        assert_eq!(mapped.size.width, 8);
        assert_eq!(mapped.format, native::WGPUTextureFormat_RGBA8Unorm.into());
        assert_eq!(mapped.view_formats.len(), 1);

        let null_view_formats = native::WGPUTextureDescriptor {
            viewFormats: std::ptr::null(),
            ..descriptor
        };
        assert!(unsafe { map_texture_descriptor(&null_view_formats) }
            .view_formats
            .is_empty());
    }

    #[test]
    fn map_texture_view_descriptor_decodes_fields_and_none_defaults() {
        let descriptor = native::WGPUTextureViewDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            format: native::WGPUTextureFormat_RGBA8Unorm,
            dimension: native::WGPUTextureViewDimension_2DArray,
            baseMipLevel: 2,
            mipLevelCount: 3,
            baseArrayLayer: 4,
            arrayLayerCount: 5,
            aspect: native::WGPUTextureAspect_All,
            usage: native::WGPUTextureUsage_TextureBinding,
        };
        let mapped = map_texture_view_descriptor(Some(&descriptor));
        assert_eq!(
            mapped.format,
            Some(native::WGPUTextureFormat_RGBA8Unorm.into())
        );
        assert_eq!(mapped.dimension, Some(core::TextureViewDimension::D2Array));
        assert_eq!(mapped.base_mip_level, 2);
        assert_eq!(mapped.mip_level_count, Some(3));
        assert_eq!(mapped.base_array_layer, 4);
        assert_eq!(mapped.array_layer_count, Some(5));
        assert_eq!(mapped.aspect, Some(core::TextureAspect::All));

        let defaulted = map_texture_view_descriptor(None);
        assert_eq!(defaulted.format, None);
        assert_eq!(defaulted.dimension, None);
        assert_eq!(defaulted.mip_level_count, None);
        assert_eq!(defaulted.array_layer_count, None);
    }

    #[test]
    fn map_device_lost_callback_info_round_trips_present_and_absent_callback() {
        unsafe extern "C" fn callback(
            _device: *const native::WGPUDevice,
            _reason: native::WGPUDeviceLostReason,
            _message: native::WGPUStringView,
            _userdata1: *mut c_void,
            _userdata2: *mut c_void,
        ) {
        }

        let native_info = native::WGPUDeviceLostCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_WaitAnyOnly,
            callback: Some(callback),
            userdata1: 0x1234usize as *mut c_void,
            userdata2: 0x5678usize as *mut c_void,
        };
        let mapped = map_device_lost_callback_info(native_info);
        assert_eq!(mapped.mode, native::WGPUCallbackMode_WaitAnyOnly);
        assert!(mapped.has_callback());
        assert_eq!(mapped.userdata1, 0x1234);
        assert_eq!(mapped.userdata2, 0x5678);

        let absent = map_device_lost_callback_info(native::WGPUDeviceLostCallbackInfo {
            callback: None,
            ..native_info
        });
        assert!(!absent.has_callback());
    }
}
