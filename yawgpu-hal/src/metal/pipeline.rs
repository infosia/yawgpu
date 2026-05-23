use super::*;

/// Stores metal compute pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalComputePipeline {
    pub(super) inner: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) workgroup_size: (u32, u32, u32),
}

unsafe impl Send for MetalComputePipeline {}
unsafe impl Sync for MetalComputePipeline {}

impl std::fmt::Debug for MetalComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalComputePipeline")
            .field("workgroup_size", &self.workgroup_size)
            .finish()
    }
}

/// Stores metal render pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalRenderPipeline {
    pub(super) inner: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(super) depth_stencil_state: Option<Retained<ProtocolObject<dyn MTLDepthStencilState>>>,
    pub(super) primitive_topology: HalPrimitiveTopology,
}

unsafe impl Send for MetalRenderPipeline {}
unsafe impl Sync for MetalRenderPipeline {}

impl std::fmt::Debug for MetalRenderPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalRenderPipeline")
            .field("primitive_topology", &self.primitive_topology)
            .field(
                "has_depth_stencil_state",
                &self.depth_stencil_state.is_some(),
            )
            .finish()
    }
}

/// Creates compute pipeline and reports validation errors through the owning device.
pub(super) fn create_compute_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    msl_source: &str,
    entry_point: &str,
    workgroup_size: (u32, u32, u32),
) -> Result<MetalComputePipeline, HalError> {
    let source = NSString::from_str(msl_source);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|error| shader_error(error.localizedDescription().to_string()))?;
    let function = library
        .newFunctionWithName(&NSString::from_str(entry_point))
        .ok_or_else(|| shader_error(format!("compute function '{entry_point}' not found")))?;
    let inner = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| shader_error(error.localizedDescription().to_string()))?;
    Ok(MetalComputePipeline {
        inner,
        workgroup_size,
    })
}

/// Creates render pipeline and reports validation errors through the owning device.
pub(super) fn create_render_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    msl_source: &str,
    vertex_entry_point: &str,
    fragment_entry_point: &str,
    descriptor: &HalRenderPipelineDescriptor,
) -> Result<MetalRenderPipeline, HalError> {
    if descriptor.color_formats.is_empty() {
        return Err(shader_error(
            "render pipeline requires a color target".to_owned(),
        ));
    }
    let source = NSString::from_str(msl_source);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|error| shader_error(error.localizedDescription().to_string()))?;
    let vertex_function = library
        .newFunctionWithName(&NSString::from_str(vertex_entry_point))
        .ok_or_else(|| shader_error(format!("vertex function '{vertex_entry_point}' not found")))?;
    let fragment_function = library
        .newFunctionWithName(&NSString::from_str(fragment_entry_point))
        .ok_or_else(|| {
            shader_error(format!(
                "fragment function '{fragment_entry_point}' not found"
            ))
        })?;
    let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
    pipeline_descriptor.setVertexFunction(Some(&vertex_function));
    pipeline_descriptor.setFragmentFunction(Some(&fragment_function));
    // Each `color_formats[i]` populates `MTLRenderPipelineDescriptor.colorAttachments[i].pixelFormat`,
    // so the MTL pipeline's color-attachment layout matches the encoder slot-for-slot.
    // For subpass pipelines this carries every layout slot's format (including
    // ones the current subpass doesn't write to) — the fragment shader's
    // `[[color(N)]]` outputs naturally land in the right MTL slot.
    let color_attachments = pipeline_descriptor.colorAttachments();
    for (i, &color_format) in descriptor.color_formats.iter().enumerate() {
        let (pixel_format, _) = map_texture_format(color_format)?;
        let attach = unsafe { color_attachments.objectAtIndexedSubscript(i) };
        attach.setPixelFormat(pixel_format);
    }
    if let Some(depth_stencil) = descriptor.depth_stencil {
        let (pixel_format, _) = map_texture_format(depth_stencil.format)?;
        if format_has_depth_aspect(depth_stencil.format) {
            pipeline_descriptor.setDepthAttachmentPixelFormat(pixel_format);
        }
        if format_has_stencil_aspect(depth_stencil.format) {
            pipeline_descriptor.setStencilAttachmentPixelFormat(pixel_format);
        }
    }
    let vertex_descriptor = MTLVertexDescriptor::new();
    for buffer in &descriptor.vertex_buffers {
        let metal_index = buffer
            .attributes
            .first()
            .map(|attribute| attribute.metal_buffer_index)
            .unwrap_or(0);
        let layouts = vertex_descriptor.layouts();
        let layout = unsafe { layouts.objectAtIndexedSubscript(to_ns(u64::from(metal_index))?) };
        unsafe {
            layout.setStride(to_ns(buffer.array_stride)?);
            layout.setStepRate(1);
        }
        layout.setStepFunction(match buffer.step_mode {
            HalVertexStepMode::Vertex => MTLVertexStepFunction::PerVertex,
            HalVertexStepMode::Instance => MTLVertexStepFunction::PerInstance,
        });
        for attribute in &buffer.attributes {
            let attributes = vertex_descriptor.attributes();
            let attr = unsafe {
                attributes.objectAtIndexedSubscript(to_ns(u64::from(attribute.shader_location))?)
            };
            attr.setFormat(map_vertex_format(attribute.format)?);
            unsafe {
                attr.setOffset(to_ns(attribute.offset)?);
                attr.setBufferIndex(to_ns(u64::from(attribute.metal_buffer_index))?);
            }
        }
    }
    pipeline_descriptor.setVertexDescriptor(Some(&vertex_descriptor));
    let inner = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
        .map_err(|error| shader_error(error.localizedDescription().to_string()))?;
    // Every render pipeline carries an `MTLDepthStencilState`. When the
    // public descriptor opts out of depth-stencil (Option::None) we still
    // bind a no-op state (depthCompare=Always, depthWrite=false, no stencil)
    // so the encoder doesn't inherit a previous pipeline's depth test/write
    // against a shared depth attachment. Without this, multi-subpass passes
    // where one subpass uses depth and a later subpass doesn't would fail
    // depth-test for the later subpass's draws (the lighting/composite
    // fullscreen triangles in tiled_deferred specifically lose to the
    // gbuffer subpass's previously written depth values).
    let depth_stencil_state = match descriptor.depth_stencil {
        Some(depth_stencil) => Some(create_depth_stencil_state(device, depth_stencil)?),
        None => Some(create_noop_depth_stencil_state(device)),
    };
    Ok(MetalRenderPipeline {
        inner,
        depth_stencil_state,
        primitive_topology: descriptor.primitive_topology,
    })
}

fn create_noop_depth_stencil_state(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Retained<ProtocolObject<dyn MTLDepthStencilState>> {
    let descriptor = MTLDepthStencilDescriptor::new();
    descriptor.setDepthCompareFunction(MTLCompareFunction::Always);
    descriptor.setDepthWriteEnabled(false);
    device
        .newDepthStencilStateWithDescriptor(&descriptor)
        .expect("no-op MTLDepthStencilState creation cannot fail")
}

fn create_depth_stencil_state(
    device: &ProtocolObject<dyn MTLDevice>,
    depth_stencil: HalDepthStencilState,
) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>, HalError> {
    let descriptor = MTLDepthStencilDescriptor::new();
    descriptor.setDepthCompareFunction(map_compare_function(depth_stencil.depth_compare));
    descriptor.setDepthWriteEnabled(depth_stencil.depth_write_enabled);
    if format_has_stencil_aspect(depth_stencil.format) {
        let front = create_stencil_descriptor(
            depth_stencil.stencil_front,
            depth_stencil.stencil_read_mask,
            depth_stencil.stencil_write_mask,
        );
        let back = create_stencil_descriptor(
            depth_stencil.stencil_back,
            depth_stencil.stencil_read_mask,
            depth_stencil.stencil_write_mask,
        );
        descriptor.setFrontFaceStencil(Some(&front));
        descriptor.setBackFaceStencil(Some(&back));
    }
    device
        .newDepthStencilStateWithDescriptor(&descriptor)
        .ok_or_else(|| shader_error("depth stencil state creation failed".to_owned()))
}

fn create_stencil_descriptor(
    face: HalStencilFaceState,
    read_mask: u32,
    write_mask: u32,
) -> Retained<MTLStencilDescriptor> {
    let descriptor = MTLStencilDescriptor::new();
    descriptor.setStencilCompareFunction(map_compare_function(face.compare));
    descriptor.setStencilFailureOperation(map_stencil_operation(face.fail_op));
    descriptor.setDepthFailureOperation(map_stencil_operation(face.depth_fail_op));
    descriptor.setDepthStencilPassOperation(map_stencil_operation(face.pass_op));
    descriptor.setReadMask(read_mask);
    descriptor.setWriteMask(write_mask);
    descriptor
}

fn map_stencil_operation(operation: HalStencilOperation) -> MTLStencilOperation {
    match operation {
        HalStencilOperation::Keep => MTLStencilOperation::Keep,
        HalStencilOperation::Zero => MTLStencilOperation::Zero,
        HalStencilOperation::Replace => MTLStencilOperation::Replace,
        HalStencilOperation::Invert => MTLStencilOperation::Invert,
        HalStencilOperation::IncrementClamp => MTLStencilOperation::IncrementClamp,
        HalStencilOperation::DecrementClamp => MTLStencilOperation::DecrementClamp,
        HalStencilOperation::IncrementWrap => MTLStencilOperation::IncrementWrap,
        HalStencilOperation::DecrementWrap => MTLStencilOperation::DecrementWrap,
    }
}

fn format_has_depth_aspect(format: HalTextureFormat) -> bool {
    matches!(
        format,
        HalTextureFormat::Depth16Unorm
            | HalTextureFormat::Depth24Plus
            | HalTextureFormat::Depth24PlusStencil8
            | HalTextureFormat::Depth32Float
            | HalTextureFormat::Depth32FloatStencil8
    )
}

fn format_has_stencil_aspect(format: HalTextureFormat) -> bool {
    matches!(
        format,
        HalTextureFormat::Stencil8
            | HalTextureFormat::Depth24PlusStencil8
            | HalTextureFormat::Depth32FloatStencil8
    )
}
