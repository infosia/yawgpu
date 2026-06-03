use super::*;
use crate::format::{format_has_depth_aspect, format_has_stencil_aspect};

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
///
/// `depth_stencil_state` is always populated: when the public descriptor opts
/// out of depth-stencil, `create_render_pipeline` synthesizes a no-op state
/// (`Always`, no write, no stencil) so the encoder never silently inherits a
/// previous pipeline's depth-stencil state across draws.
#[derive(Clone)]
pub struct MetalRenderPipeline {
    pub(super) inner: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(super) depth_stencil_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    pub(super) primitive_topology: HalPrimitiveTopology,
    pub(super) depth_bias: i32,
    pub(super) depth_bias_slope_scale: f32,
    pub(super) depth_bias_clamp: f32,
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
    shader: HalShaderSource,
    vertex_entry_point: &str,
    fragment_entry_point: Option<&str>,
    descriptor: &HalRenderPipelineDescriptor,
) -> Result<MetalRenderPipeline, HalError> {
    if descriptor.color_formats.is_empty() && descriptor.depth_stencil.is_none() {
        return Err(shader_error(
            "render pipeline requires a color target or depth-stencil state".to_owned(),
        ));
    }
    let (vertex_function, fragment_function) =
        create_render_functions(device, shader, vertex_entry_point, fragment_entry_point)?;
    let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
    pipeline_descriptor.setVertexFunction(Some(&vertex_function));
    pipeline_descriptor.setFragmentFunction(fragment_function.as_deref());
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
    let (depth_bias, depth_bias_slope_scale, depth_bias_clamp) = descriptor
        .depth_stencil
        .map(|depth_stencil| {
            (
                depth_stencil.depth_bias,
                depth_stencil.depth_bias_slope_scale,
                depth_stencil.depth_bias_clamp,
            )
        })
        .unwrap_or((0, 0.0, 0.0));
    let depth_stencil_state = match descriptor.depth_stencil {
        Some(depth_stencil) => create_depth_stencil_state(device, depth_stencil)?,
        None => create_noop_depth_stencil_state(device)?,
    };
    Ok(MetalRenderPipeline {
        inner,
        depth_stencil_state,
        primitive_topology: descriptor.primitive_topology,
        depth_bias,
        depth_bias_slope_scale,
        depth_bias_clamp,
    })
}

fn create_render_functions(
    device: &ProtocolObject<dyn MTLDevice>,
    shader: HalShaderSource,
    vertex_entry_point: &str,
    fragment_entry_point: Option<&str>,
) -> Result<
    (
        Retained<ProtocolObject<dyn MTLFunction>>,
        Option<Retained<ProtocolObject<dyn MTLFunction>>>,
    ),
    HalError,
> {
    match shader {
        HalShaderSource::Msl(source) => {
            let library = create_render_library(device, &source)?;
            let vertex_function = render_function(&library, vertex_entry_point, "vertex")?;
            let fragment_function = fragment_entry_point
                .map(|entry| render_function(&library, entry, "fragment"))
                .transpose()?;
            Ok((vertex_function, fragment_function))
        }
        HalShaderSource::MslStages { vertex, fragment } => {
            let vertex_library = create_render_library(device, &vertex)?;
            let vertex_function = render_function(&vertex_library, vertex_entry_point, "vertex")?;
            let fragment_function = match (fragment, fragment_entry_point) {
                (Some(fragment), Some(fragment_entry_point)) => {
                    let fragment_library = create_render_library(device, &fragment)?;
                    Some(render_function(
                        &fragment_library,
                        fragment_entry_point,
                        "fragment",
                    )?)
                }
                (None, None) => None,
                _ => {
                    return Err(shader_error(
                        "Metal render pipeline fragment source and entry point must match"
                            .to_owned(),
                    ));
                }
            };
            Ok((vertex_function, fragment_function))
        }
        _ => Err(shader_error(
            "Metal render pipeline requires MSL render shader source".to_owned(),
        )),
    }
}

fn render_function(
    library: &ProtocolObject<dyn MTLLibrary>,
    entry_point: &str,
    stage: &str,
) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, HalError> {
    library
        .newFunctionWithName(&NSString::from_str(entry_point))
        .ok_or_else(|| shader_error(format!("{stage} function '{entry_point}' not found")))
}

fn create_render_library(
    device: &ProtocolObject<dyn MTLDevice>,
    msl_source: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, HalError> {
    let source = NSString::from_str(msl_source);
    let options = MTLCompileOptions::new();
    options.setPreserveInvariance(true);
    device
        .newLibraryWithSource_options_error(&source, Some(&options))
        .map_err(|error| shader_error(error.localizedDescription().to_string()))
}

fn create_noop_depth_stencil_state(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>, HalError> {
    let descriptor = MTLDepthStencilDescriptor::new();
    descriptor.setDepthCompareFunction(MTLCompareFunction::Always);
    descriptor.setDepthWriteEnabled(false);
    device
        .newDepthStencilStateWithDescriptor(&descriptor)
        .ok_or_else(|| shader_error("no-op MTLDepthStencilState creation failed".to_owned()))
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
