use super::*;
use crate::format::{format_has_depth_aspect, format_has_stencil_aspect};
use crate::metal::format::vertex_format_byte_size;

/// Stores metal compute pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalComputePipeline {
    pub(super) inner: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) workgroup_size: (u32, u32, u32),
    pub(super) buffer_sizes_slot: Option<u32>,
    pub(super) buffer_size_bindings: Vec<HalMslBufferSizeBinding>,
    /// Per-argument threadgroup memory allocation sizes (bytes, rounded to a
    /// multiple of 16). Mirrors `wg_memory_sizes` in wgpu-hal/metal/device.rs.
    /// The compute encoder calls `setThreadgroupMemoryLength:atIndex:` for each
    /// entry in this vec before every dispatch.
    pub(super) workgroup_memory_sizes: Vec<u32>,
    /// Compute-stage immediates delivery metadata (Block 94 S2). `None` when
    /// the compute entry point uses no `var<immediate>` data.
    pub(super) immediates: Option<HalMslImmediates>,
}

unsafe impl Send for MetalComputePipeline {}
unsafe impl Sync for MetalComputePipeline {}

impl std::fmt::Debug for MetalComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalComputePipeline")
            .field("workgroup_size", &self.workgroup_size)
            .field("buffer_sizes_slot", &self.buffer_sizes_slot)
            .field("workgroup_memory_sizes", &self.workgroup_memory_sizes)
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
    pub(super) front_face: HalFrontFace,
    pub(super) cull_mode: HalCullMode,
    pub(super) unclipped_depth: bool,
    pub(super) depth_bias: i32,
    pub(super) depth_bias_slope_scale: f32,
    pub(super) depth_bias_clamp: f32,
    pub(super) vertex_buffer_sizes_slot: Option<u32>,
    pub(super) vertex_buffer_size_bindings: Vec<HalMslBufferSizeBinding>,
    pub(super) fragment_buffer_sizes_slot: Option<u32>,
    pub(super) fragment_buffer_size_bindings: Vec<HalMslBufferSizeBinding>,
    /// Vertex-stage immediates delivery metadata (Block 94 S2). `None` when
    /// the vertex entry point uses no `var<immediate>` data.
    pub(super) vertex_immediates: Option<HalMslImmediates>,
    /// Fragment-stage immediates delivery metadata (Block 94 S2); also
    /// covers the frag-depth clamp range when this pipeline clamps
    /// frag_depth. `None` when the fragment entry point uses no immediates
    /// and does not clamp frag_depth.
    pub(super) fragment_immediates: Option<HalMslImmediates>,
    /// Metal buffer indices for vertex buffers in `vertex_buffer_mappings` order.
    /// These correspond to the `buffer_sizeN` fields that Tint's MSL codegen appends
    /// after the storage-array size fields inside `_mslBufferSizes`.  The encoder
    /// fills these slots with effective vertex-buffer byte sizes before every draw
    /// so that Tint's vertex-pulling OOB guards compare against real values.
    pub(super) vertex_buffer_metal_indices: Vec<u32>,
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
#[allow(clippy::too_many_arguments)]
pub(super) fn create_compute_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    msl_source: &str,
    entry_point: &str,
    workgroup_size: (u32, u32, u32),
    buffer_sizes_slot: Option<u32>,
    buffer_size_bindings: Vec<HalMslBufferSizeBinding>,
    workgroup_memory_sizes: Vec<u32>,
    immediates: Option<HalMslImmediates>,
) -> Result<MetalComputePipeline, HalError> {
    // Wrap the compile in an autoreleasepool so the autoreleased temporaries
    // produced by the objc calls (NSString, the transient library/function)
    // are reclaimed when this returns rather than accumulating on a thread with
    // no drained pool. The returned pipeline state is held by an objc2
    // `Retained` (explicit +1), so it survives the pool drain. Mirrors
    // `MetalQueue::submit_copies` / `wait_idle`.
    autoreleasepool(|_| {
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
            buffer_sizes_slot,
            buffer_size_bindings,
            workgroup_memory_sizes,
            immediates,
        })
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
    // Wrap the compile in an autoreleasepool so the autoreleased temporaries
    // produced by the objc calls (NSString, transient libraries/functions, the
    // MTLRenderPipelineDescriptor and its sub-descriptors) are reclaimed when
    // this returns rather than accumulating on a thread with no drained pool.
    // The returned pipeline state and depth-stencil state are held by objc2
    // `Retained` (explicit +1), so they survive the pool drain. Mirrors
    // `MetalQueue::submit_copies` / `wait_idle`.
    autoreleasepool(|_| {
        if !descriptor.color_targets.iter().any(Option::is_some)
            && descriptor.depth_stencil.is_none()
        {
            return Err(shader_error(
                "render pipeline requires a color target or depth-stencil state".to_owned(),
            ));
        }
        // Metal has no pipeline sample-mask API; yawgpu-core bakes the mask into MSL.
        let size_metadata = render_size_metadata(&shader);
        let use_vertex_descriptor = render_shader_uses_metal_vertex_descriptor(&shader);
        let (vertex_function, fragment_function) =
            create_render_functions(device, shader, vertex_entry_point, fragment_entry_point)?;
        let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
        pipeline_descriptor.setVertexFunction(Some(&vertex_function));
        pipeline_descriptor.setFragmentFunction(fragment_function.as_deref());
        pipeline_descriptor.setRasterSampleCount(to_ns(u64::from(descriptor.sample_count))?);
        pipeline_descriptor.setAlphaToCoverageEnabled(descriptor.alpha_to_coverage_enabled);
        // Each color target populates `MTLRenderPipelineDescriptor.colorAttachments[i]`,
        // so the MTL pipeline's color-attachment layout matches the encoder slot-for-slot.
        let color_attachments = pipeline_descriptor.colorAttachments();
        for (i, color_target) in descriptor.color_targets.iter().copied().enumerate() {
            let Some(color_target) = color_target else {
                continue;
            };
            let (pixel_format, _) = map_texture_format(color_target.format)?;
            let attach = unsafe { color_attachments.objectAtIndexedSubscript(i) };
            attach.setPixelFormat(pixel_format);
            set_color_attachment_state(&attach, color_target);
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
        if use_vertex_descriptor {
            let vertex_descriptor = MTLVertexDescriptor::new();
            for buffer in &descriptor.vertex_buffers {
                let metal_index = buffer
                    .attributes
                    .first()
                    .map(|attribute| attribute.metal_buffer_index)
                    .unwrap_or(0);
                let layouts = vertex_descriptor.layouts();
                let layout =
                    unsafe { layouts.objectAtIndexedSubscript(to_ns(u64::from(metal_index))?) };
                if buffer.array_stride == 0 {
                    // WebGPU allows arrayStride=0 (all vertices read the same
                    // element). Metal requires a non-zero stride when an
                    // attribute references the buffer, so synthesize one.
                    let max_extent = buffer
                        .attributes
                        .iter()
                        .map(|attribute| {
                            attribute.offset + vertex_format_byte_size(attribute.format)
                        })
                        .max()
                        .unwrap_or(0);
                    let stride = max_extent.div_ceil(4) * 4;
                    unsafe {
                        layout.setStride(to_ns(stride.max(4))?);
                        layout.setStepRate(0);
                    }
                    layout.setStepFunction(MTLVertexStepFunction::Constant);
                } else {
                    unsafe {
                        layout.setStride(to_ns(buffer.array_stride)?);
                        layout.setStepRate(1);
                    }
                    layout.setStepFunction(match buffer.step_mode {
                        HalVertexStepMode::Vertex => MTLVertexStepFunction::PerVertex,
                        HalVertexStepMode::Instance => MTLVertexStepFunction::PerInstance,
                    });
                }
                for attribute in &buffer.attributes {
                    let attributes = vertex_descriptor.attributes();
                    let attr = unsafe {
                        attributes
                            .objectAtIndexedSubscript(to_ns(u64::from(attribute.shader_location))?)
                    };
                    attr.setFormat(map_vertex_format(attribute.format)?);
                    unsafe {
                        attr.setOffset(to_ns(attribute.offset)?);
                        attr.setBufferIndex(to_ns(u64::from(attribute.metal_buffer_index))?);
                    }
                }
            }
            pipeline_descriptor.setVertexDescriptor(Some(&vertex_descriptor));
        }
        let inner = device
            .newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
            .map_err(|error| shader_error(error.localizedDescription().to_string()))?;
        // Every render pipeline carries an `MTLDepthStencilState`. When the
        // public descriptor opts out of depth-stencil (Option::None) we still
        // bind a no-op state (depthCompare=Always, depthWrite=false, no stencil)
        // so the encoder doesn't inherit a previous pipeline's depth test/write.
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
            front_face: descriptor.front_face,
            cull_mode: descriptor.cull_mode,
            unclipped_depth: descriptor.unclipped_depth,
            depth_bias,
            depth_bias_slope_scale,
            depth_bias_clamp,
            vertex_buffer_sizes_slot: size_metadata.vertex_slot,
            vertex_buffer_size_bindings: size_metadata.vertex_bindings,
            fragment_buffer_sizes_slot: size_metadata.fragment_slot,
            fragment_buffer_size_bindings: size_metadata.fragment_bindings,
            vertex_immediates: size_metadata.vertex_immediates,
            fragment_immediates: size_metadata.fragment_immediates,
            vertex_buffer_metal_indices: size_metadata.vertex_buffer_metal_indices,
        })
    })
}

fn set_color_attachment_state(
    attachment: &MTLRenderPipelineColorAttachmentDescriptor,
    target: HalColorTargetState,
) {
    attachment.setWriteMask(mtl_color_write_mask(target.write_mask));
    if let Some(blend) = target.blend {
        attachment.setBlendingEnabled(true);
        attachment.setSourceRGBBlendFactor(mtl_blend_factor(blend.color.src_factor, false));
        attachment.setDestinationRGBBlendFactor(mtl_blend_factor(blend.color.dst_factor, false));
        attachment.setRgbBlendOperation(mtl_blend_operation(blend.color.operation));
        attachment.setSourceAlphaBlendFactor(mtl_blend_factor(blend.alpha.src_factor, true));
        attachment.setDestinationAlphaBlendFactor(mtl_blend_factor(blend.alpha.dst_factor, true));
        attachment.setAlphaBlendOperation(mtl_blend_operation(blend.alpha.operation));
    } else {
        attachment.setBlendingEnabled(false);
    }
}

fn mtl_color_write_mask(write_mask: u32) -> MTLColorWriteMask {
    let mut mask = MTLColorWriteMask::empty();
    if write_mask & 0x1 != 0 {
        mask |= MTLColorWriteMask::Red;
    }
    if write_mask & 0x2 != 0 {
        mask |= MTLColorWriteMask::Green;
    }
    if write_mask & 0x4 != 0 {
        mask |= MTLColorWriteMask::Blue;
    }
    if write_mask & 0x8 != 0 {
        mask |= MTLColorWriteMask::Alpha;
    }
    mask
}

fn mtl_blend_operation(operation: HalBlendOperation) -> MTLBlendOperation {
    match operation {
        HalBlendOperation::Add => MTLBlendOperation::Add,
        HalBlendOperation::Subtract => MTLBlendOperation::Subtract,
        HalBlendOperation::ReverseSubtract => MTLBlendOperation::ReverseSubtract,
        HalBlendOperation::Min => MTLBlendOperation::Min,
        HalBlendOperation::Max => MTLBlendOperation::Max,
    }
}

fn mtl_blend_factor(factor: HalBlendFactor, alpha: bool) -> MTLBlendFactor {
    match factor {
        HalBlendFactor::Zero => MTLBlendFactor::Zero,
        HalBlendFactor::One => MTLBlendFactor::One,
        HalBlendFactor::Src => {
            if alpha {
                MTLBlendFactor::SourceAlpha
            } else {
                MTLBlendFactor::SourceColor
            }
        }
        HalBlendFactor::OneMinusSrc => {
            if alpha {
                MTLBlendFactor::OneMinusSourceAlpha
            } else {
                MTLBlendFactor::OneMinusSourceColor
            }
        }
        HalBlendFactor::SrcAlpha => MTLBlendFactor::SourceAlpha,
        HalBlendFactor::OneMinusSrcAlpha => MTLBlendFactor::OneMinusSourceAlpha,
        HalBlendFactor::Dst => {
            if alpha {
                MTLBlendFactor::DestinationAlpha
            } else {
                MTLBlendFactor::DestinationColor
            }
        }
        HalBlendFactor::OneMinusDst => {
            if alpha {
                MTLBlendFactor::OneMinusDestinationAlpha
            } else {
                MTLBlendFactor::OneMinusDestinationColor
            }
        }
        HalBlendFactor::DstAlpha => MTLBlendFactor::DestinationAlpha,
        HalBlendFactor::OneMinusDstAlpha => MTLBlendFactor::OneMinusDestinationAlpha,
        HalBlendFactor::SrcAlphaSaturated => MTLBlendFactor::SourceAlphaSaturated,
        HalBlendFactor::Constant => {
            if alpha {
                MTLBlendFactor::BlendAlpha
            } else {
                MTLBlendFactor::BlendColor
            }
        }
        HalBlendFactor::OneMinusConstant => {
            if alpha {
                MTLBlendFactor::OneMinusBlendAlpha
            } else {
                MTLBlendFactor::OneMinusBlendColor
            }
        }
        HalBlendFactor::Src1 => {
            if alpha {
                MTLBlendFactor::Source1Alpha
            } else {
                MTLBlendFactor::Source1Color
            }
        }
        HalBlendFactor::OneMinusSrc1 => {
            if alpha {
                MTLBlendFactor::OneMinusSource1Alpha
            } else {
                MTLBlendFactor::OneMinusSource1Color
            }
        }
        HalBlendFactor::Src1Alpha => MTLBlendFactor::Source1Alpha,
        HalBlendFactor::OneMinusSrc1Alpha => MTLBlendFactor::OneMinusSource1Alpha,
    }
}

/// A compiled vertex function plus an optional fragment function (absent for a
/// vertex-only / depth-only pipeline).
type RenderFunctions = (
    Retained<ProtocolObject<dyn MTLFunction>>,
    Option<Retained<ProtocolObject<dyn MTLFunction>>>,
);

struct RenderSizeMetadata {
    vertex_slot: Option<u32>,
    vertex_bindings: Vec<HalMslBufferSizeBinding>,
    fragment_slot: Option<u32>,
    fragment_bindings: Vec<HalMslBufferSizeBinding>,
    vertex_immediates: Option<HalMslImmediates>,
    fragment_immediates: Option<HalMslImmediates>,
    /// Metal buffer indices for vertex buffers in vertex_buffer_mappings order.
    vertex_buffer_metal_indices: Vec<u32>,
}

fn render_size_metadata(shader: &HalShaderSource) -> RenderSizeMetadata {
    match shader {
        HalShaderSource::MslStagesWithBufferSizes {
            vertex_buffer_sizes_slot,
            vertex_buffer_size_bindings,
            fragment_buffer_sizes_slot,
            fragment_buffer_size_bindings,
            vertex_immediates,
            fragment_immediates,
            vertex_buffer_metal_indices,
            ..
        } => RenderSizeMetadata {
            vertex_slot: *vertex_buffer_sizes_slot,
            vertex_bindings: vertex_buffer_size_bindings.clone(),
            fragment_slot: *fragment_buffer_sizes_slot,
            fragment_bindings: fragment_buffer_size_bindings.clone(),
            vertex_immediates: *vertex_immediates,
            fragment_immediates: *fragment_immediates,
            vertex_buffer_metal_indices: vertex_buffer_metal_indices.clone(),
        },
        _ => RenderSizeMetadata {
            vertex_slot: None,
            vertex_bindings: Vec::new(),
            fragment_slot: None,
            fragment_bindings: Vec::new(),
            vertex_immediates: None,
            fragment_immediates: None,
            vertex_buffer_metal_indices: Vec::new(),
        },
    }
}

fn render_shader_uses_metal_vertex_descriptor(shader: &HalShaderSource) -> bool {
    // A vertex shader that declares Metal `[[stage_in]]` input needs an
    // `MTLVertexDescriptor` mapping its `[[attribute(N)]]`s to vertex buffers. Tint
    // emits this in its default MSL mode. Tint's vertex-pulling MSL instead reads
    // vertex data directly from bound buffers (no `[[stage_in]]`) and must NOT get a
    // descriptor — so detect the model from the emitted source, not the variant.
    let vertex_source = match shader {
        HalShaderSource::Msl(source) | HalShaderSource::MslWithBufferSizes { source, .. } => {
            Some(source.as_str())
        }
        HalShaderSource::MslStages { vertex, .. }
        | HalShaderSource::MslStagesWithBufferSizes { vertex, .. } => Some(vertex.as_str()),
        _ => None,
    };
    vertex_source.is_some_and(|source| source.contains("[[stage_in]]"))
}

fn create_render_functions(
    device: &ProtocolObject<dyn MTLDevice>,
    shader: HalShaderSource,
    vertex_entry_point: &str,
    fragment_entry_point: Option<&str>,
) -> Result<RenderFunctions, HalError> {
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
        HalShaderSource::MslWithBufferSizes { source, .. } => {
            let library = create_render_library(device, &source)?;
            let vertex_function = render_function(&library, vertex_entry_point, "vertex")?;
            let fragment_function = fragment_entry_point
                .map(|entry| render_function(&library, entry, "fragment"))
                .transpose()?;
            Ok((vertex_function, fragment_function))
        }
        HalShaderSource::MslStagesWithBufferSizes {
            vertex, fragment, ..
        } => {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_shader_uses_vertex_descriptor_for_stage_in_msl() {
        // The Tint frontend emits `[[stage_in]]` vertex input, which needs an
        // MTLVertexDescriptor regardless of the source variant.
        let shader = HalShaderSource::MslStagesWithBufferSizes {
            vertex: "vertex Out v(In in [[stage_in]]) { return {}; }".to_owned(),
            fragment: None,
            vertex_buffer_sizes_slot: None,
            vertex_buffer_size_bindings: Vec::new(),
            fragment_buffer_sizes_slot: None,
            fragment_buffer_size_bindings: Vec::new(),
            vertex_immediates: None,
            fragment_immediates: None,
            vertex_buffer_metal_indices: Vec::new(),
        };
        assert!(render_shader_uses_metal_vertex_descriptor(&shader));

        // A plain MSL with no `[[stage_in]]` (e.g. no vertex inputs) must not get
        // a descriptor.
        assert!(!render_shader_uses_metal_vertex_descriptor(
            &HalShaderSource::Msl(String::new())
        ));
    }

    #[test]
    fn render_shader_skips_vertex_descriptor_for_vertex_pulling_msl() {
        let shader = HalShaderSource::MslStagesWithBufferSizes {
            vertex: String::new(),
            fragment: None,
            vertex_buffer_sizes_slot: None,
            vertex_buffer_size_bindings: Vec::new(),
            fragment_buffer_sizes_slot: None,
            fragment_buffer_size_bindings: Vec::new(),
            vertex_immediates: None,
            fragment_immediates: None,
            vertex_buffer_metal_indices: Vec::new(),
        };

        assert!(!render_shader_uses_metal_vertex_descriptor(&shader));
    }

    #[test]
    fn render_shader_skips_vertex_descriptor_for_stage_msl_sources() {
        let shader = HalShaderSource::MslStages {
            vertex: String::new(),
            fragment: None,
        };

        assert!(!render_shader_uses_metal_vertex_descriptor(&shader));
    }
}
