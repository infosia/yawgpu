use super::*;

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

/// Converts a compute pipeline descriptor to the core representation.
///
/// # Safety
///
/// `descriptor.compute.module` must be a non-null live yawgpu shader module.
/// `descriptor.layout`, when non-null, must be a live yawgpu pipeline layout.
/// `compute.constants`, when non-null and `constantCount > 0`, must point to
/// `constantCount` valid `WGPUConstantEntry` values.
/// Converts compute pipeline descriptor into the corresponding yawgpu representation.
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
/// Converts render pipeline descriptor into the corresponding yawgpu representation.
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
