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
    pub(super) primitive_topology: HalPrimitiveTopology,
}

unsafe impl Send for MetalRenderPipeline {}
unsafe impl Sync for MetalRenderPipeline {}

impl std::fmt::Debug for MetalRenderPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalRenderPipeline")
            .field("primitive_topology", &self.primitive_topology)
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
    let color_format = descriptor
        .color_formats
        .first()
        .copied()
        .ok_or_else(|| shader_error("render pipeline requires a color target".to_owned()))?;
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
    let (pixel_format, _) = map_texture_format(color_format)?;
    let color_attachments = pipeline_descriptor.colorAttachments();
    let color = unsafe { color_attachments.objectAtIndexedSubscript(0) };
    color.setPixelFormat(pixel_format);
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
    Ok(MetalRenderPipeline {
        inner,
        primitive_topology: descriptor.primitive_topology,
    })
}
