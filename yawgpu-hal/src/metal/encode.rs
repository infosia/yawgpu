use super::*;

/// Records encode into the command stream.
pub(super) fn encode_buffer_copy(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
    copy: &crate::HalBufferCopy,
) -> Result<(), HalError> {
    let HalBuffer::Metal(source) = &copy.source else {
        return Err(buffer_error("source buffer is not Metal-backed"));
    };
    let HalBuffer::Metal(destination) = &copy.destination else {
        return Err(buffer_error("destination buffer is not Metal-backed"));
    };
    source.validate_range(copy.source_offset, copy.size)?;
    destination.validate_range(copy.destination_offset, copy.size)?;
    unsafe {
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            source.inner()?,
            to_ns(copy.source_offset)?,
            destination.inner()?,
            to_ns(copy.destination_offset)?,
            to_ns(copy.size)?,
        );
    }
    Ok(())
}

/// Records encode into the command stream.
pub(super) fn encode_buffer_to_texture(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &copy.buffer else {
        return Err(buffer_error("buffer is not Metal-backed"));
    };
    let HalTexture::Metal(texture) = &copy.texture else {
        return Err(texture_error("texture is not Metal-backed"));
    };
    texture.validate_origin_extent(copy.origin, copy.extent)?;
    validate_buffer_texture_range(buffer, copy)?;
    unsafe {
        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            buffer.inner()?,
            to_ns(copy.buffer_layout.offset)?,
            to_ns(u64::from(copy.buffer_layout.bytes_per_row))?,
            buffer_texture_bytes_per_image(copy)?,
            to_mtl_size(copy.extent)?,
            texture.inner()?,
            to_ns(u64::from(copy.origin.z))?,
            to_ns(u64::from(copy.mip_level))?,
            to_mtl_origin(copy.origin.x, copy.origin.y, 0)?,
        );
    }
    Ok(())
}

/// Records encode into the command stream.
pub(super) fn encode_texture_to_buffer(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &copy.buffer else {
        return Err(buffer_error("buffer is not Metal-backed"));
    };
    let HalTexture::Metal(texture) = &copy.texture else {
        return Err(texture_error("texture is not Metal-backed"));
    };
    texture.validate_origin_extent(copy.origin, copy.extent)?;
    validate_buffer_texture_range(buffer, copy)?;
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            texture.inner()?,
            to_ns(u64::from(copy.origin.z))?,
            to_ns(u64::from(copy.mip_level))?,
            to_mtl_origin(copy.origin.x, copy.origin.y, 0)?,
            to_mtl_size(copy.extent)?,
            buffer.inner()?,
            to_ns(copy.buffer_layout.offset)?,
            to_ns(u64::from(copy.buffer_layout.bytes_per_row))?,
            buffer_texture_bytes_per_image(copy)?,
        );
    }
    Ok(())
}

/// Records encode into the command stream.
pub(super) fn encode_texture_to_texture(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
    copy: &HalTextureCopy,
) -> Result<(), HalError> {
    let HalTexture::Metal(source) = &copy.source else {
        return Err(texture_error("source texture is not Metal-backed"));
    };
    let HalTexture::Metal(destination) = &copy.destination else {
        return Err(texture_error("destination texture is not Metal-backed"));
    };
    source.validate_origin_extent(copy.source_origin, copy.extent)?;
    destination.validate_origin_extent(copy.destination_origin, copy.extent)?;
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            source.inner()?,
            to_ns(u64::from(copy.source_origin.z))?,
            to_ns(u64::from(copy.source_mip_level))?,
            to_mtl_origin(copy.source_origin.x, copy.source_origin.y, 0)?,
            to_mtl_size(copy.extent)?,
            destination.inner()?,
            to_ns(u64::from(copy.destination_origin.z))?,
            to_ns(u64::from(copy.destination_mip_level))?,
            to_mtl_origin(copy.destination_origin.x, copy.destination_origin.y, 0)?,
        );
    }
    Ok(())
}

/// Records encode into the command stream.
pub(super) fn encode_compute_pass(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pass: &HalComputePass,
) -> Result<(), HalError> {
    let crate::HalComputePipeline::Metal(pipeline) = &pass.pipeline else {
        return Err(shader_error(
            "compute pipeline is not Metal-backed".to_owned(),
        ));
    };
    encoder.setComputePipelineState(&pipeline.inner);
    for binding in &pass.bind_buffers {
        encode_compute_buffer(encoder, binding)?;
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        to_mtl_dispatch_size(pass.workgroups)?,
        to_mtl_workgroup_size(pipeline.workgroup_size)?,
    );
    Ok(())
}

fn encode_compute_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    binding: &HalBoundBuffer,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &binding.buffer else {
        return Err(buffer_error("compute buffer is not Metal-backed"));
    };
    if binding.offset > buffer.size() {
        return Err(buffer_error("compute buffer offset exceeds buffer size"));
    }
    unsafe {
        encoder.setBuffer_offset_atIndex(
            Some(buffer.inner()?),
            to_ns(binding.offset)?,
            to_ns(u64::from(binding.metal_index))?,
        );
    }
    Ok(())
}

/// Returns render pass descriptor.
pub(super) fn render_pass_descriptor(
    pass: &HalRenderPass,
) -> Result<Retained<MTLRenderPassDescriptor>, HalError> {
    let HalTexture::Metal(texture) = &pass.color_target.texture else {
        return Err(texture_error("render target is not Metal-backed"));
    };
    let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
    let color_attachments = descriptor.colorAttachments();
    let color = unsafe { color_attachments.objectAtIndexedSubscript(0) };
    color.setTexture(Some(texture.inner()?));
    color.setLoadAction(match pass.color_target.load_op {
        HalRenderLoadOp::Load => MTLLoadAction::Load,
        HalRenderLoadOp::Clear => MTLLoadAction::Clear,
    });
    color.setStoreAction(if pass.color_target.store {
        MTLStoreAction::Store
    } else {
        MTLStoreAction::DontCare
    });
    let [r, g, b, a] = pass.color_target.clear_color;
    color.setClearColor(MTLClearColor {
        red: r,
        green: g,
        blue: b,
        alpha: a,
    });
    Ok(descriptor)
}

/// Returns whether a memoryless footprint fits within a tile memory budget.
#[cfg(feature = "tiled")]
#[must_use]
pub fn tile_memory_fits_budget(bytes_per_pixel: u64, sample_count: u32, budget: u64) -> bool {
    bytes_per_pixel.saturating_mul(u64::from(sample_count)) <= budget
}

/// Returns subpass render pass descriptor.
#[cfg(feature = "tiled")]
pub(super) fn subpass_render_pass_descriptor(
    pass: &HalSubpassRenderPassCommand,
) -> Result<Retained<MTLRenderPassDescriptor>, HalError> {
    let memoryless_bytes = subpass_memoryless_bytes_per_pixel(pass)?;
    let sample_count = subpass_memoryless_sample_count(pass);
    let budget = metal_tile_memory_budget_bytes();
    if !tile_memory_fits_budget(memoryless_bytes, sample_count, budget) {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "subpass memoryless attachments exceed tile memory budget",
        });
    }
    let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
    if pass.layout.subpasses.is_empty() {
        return Err(texture_error(
            "subpass render pass requires at least one subpass",
        ));
    }
    let color_attachments = descriptor.colorAttachments();
    let attachment_indices = subpass_color_attachment_indices(pass);
    for attachment_index in attachment_indices {
        let slot = to_ns(u64::from(attachment_index))?;
        let binding = pass
            .color_attachments
            .get(attachment_index as usize)
            .ok_or_else(|| texture_error("subpass color attachment binding missing"))?;
        let color = unsafe { color_attachments.objectAtIndexedSubscript(slot) };
        color.setTexture(Some(subpass_attachment_texture(&binding.resource)?));
        color.setLoadAction(mtl_load_action(binding.load_op));
        color.setStoreAction(if binding.store {
            MTLStoreAction::Store
        } else {
            MTLStoreAction::DontCare
        });
        let [r, g, b, a] = binding.clear_color;
        color.setClearColor(MTLClearColor {
            red: r,
            green: g,
            blue: b,
            alpha: a,
        });
    }
    if pass
        .layout
        .subpasses
        .iter()
        .any(|subpass| subpass.uses_depth_stencil)
    {
        if let Some(depth) = &pass.depth_stencil_attachment {
            let depth_attachment = descriptor.depthAttachment();
            depth_attachment.setTexture(Some(subpass_attachment_texture(&depth.resource)?));
            depth_attachment.setLoadAction(mtl_load_action(depth.depth_load_op));
            depth_attachment.setStoreAction(if depth.depth_store {
                MTLStoreAction::Store
            } else {
                MTLStoreAction::DontCare
            });
            depth_attachment.setClearDepth(f64::from(depth.depth_clear_value));
            let stencil_attachment = descriptor.stencilAttachment();
            stencil_attachment.setTexture(Some(subpass_attachment_texture(&depth.resource)?));
            stencil_attachment.setLoadAction(mtl_load_action(depth.stencil_load_op));
            stencil_attachment.setStoreAction(if depth.stencil_store {
                MTLStoreAction::Store
            } else {
                MTLStoreAction::DontCare
            });
            stencil_attachment.setClearStencil(depth.stencil_clear_value);
        }
    }
    Ok(descriptor)
}

#[cfg(feature = "tiled")]
fn subpass_color_attachment_indices(pass: &HalSubpassRenderPassCommand) -> Vec<u32> {
    let mut indices = pass
        .layout
        .subpasses
        .iter()
        .flat_map(|subpass| subpass.color_attachment_indices.iter().copied())
        .collect::<Vec<_>>();
    indices.sort_unstable();
    indices.dedup();
    indices
}

#[cfg(feature = "tiled")]
fn subpass_attachment_texture(
    resource: &HalSubpassAttachmentResource,
) -> Result<&ProtocolObject<dyn MTLTextureTrait>, HalError> {
    match resource {
        HalSubpassAttachmentResource::Persistent { texture, .. } => {
            let HalTexture::Metal(texture) = texture else {
                return Err(texture_error("subpass attachment is not Metal-backed"));
            };
            texture.inner()
        }
        HalSubpassAttachmentResource::Transient(attachment) => {
            let HalTransientAttachment::Metal(attachment) = attachment else {
                return Err(texture_error("subpass transient is not Metal-backed"));
            };
            Ok(&attachment._inner)
        }
    }
}

#[cfg(feature = "tiled")]
fn subpass_memoryless_bytes_per_pixel(pass: &HalSubpassRenderPassCommand) -> Result<u64, HalError> {
    let mut total = 0_u64;
    for attachment in &pass.color_attachments {
        if let HalSubpassAttachmentResource::Transient(HalTransientAttachment::Metal(transient)) =
            &attachment.resource
        {
            if transient._memoryless {
                total = total
                    .checked_add(u64::from(format_bytes_per_pixel(transient._format)?))
                    .ok_or_else(|| texture_error("subpass tile memory footprint overflows"))?;
            }
        }
    }
    if let Some(depth) = &pass.depth_stencil_attachment {
        if let HalSubpassAttachmentResource::Transient(HalTransientAttachment::Metal(transient)) =
            &depth.resource
        {
            if transient._memoryless {
                total = total
                    .checked_add(u64::from(format_bytes_per_pixel(transient._format)?))
                    .ok_or_else(|| texture_error("subpass tile memory footprint overflows"))?;
            }
        }
    }
    Ok(total)
}

#[cfg(feature = "tiled")]
fn subpass_memoryless_sample_count(pass: &HalSubpassRenderPassCommand) -> u32 {
    let mut sample_count = 1;
    for attachment in &pass.color_attachments {
        if let HalSubpassAttachmentResource::Transient(HalTransientAttachment::Metal(transient)) =
            &attachment.resource
        {
            if transient._memoryless {
                sample_count = sample_count.max(transient._sample_count);
            }
        }
    }
    if let Some(depth) = &pass.depth_stencil_attachment {
        if let HalSubpassAttachmentResource::Transient(HalTransientAttachment::Metal(transient)) =
            &depth.resource
        {
            if transient._memoryless {
                sample_count = sample_count.max(transient._sample_count);
            }
        }
    }
    sample_count
}

#[cfg(feature = "tiled")]
fn metal_tile_memory_budget_bytes() -> u64 {
    256 * 1024
}

#[cfg(feature = "tiled")]
fn format_bytes_per_pixel(format: HalTextureFormat) -> Result<u32, HalError> {
    map_texture_format(format).map(|(_, bytes)| bytes)
}

#[cfg(feature = "tiled")]
fn mtl_load_action(load_op: HalRenderLoadOp) -> MTLLoadAction {
    match load_op {
        HalRenderLoadOp::Load => MTLLoadAction::Load,
        HalRenderLoadOp::Clear => MTLLoadAction::Clear,
    }
}

/// Records encode into the command stream.
pub(super) fn encode_render_pass(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    pass: &HalRenderPass,
) -> Result<(), HalError> {
    let (Some(pipeline), Some(draw)) = (&pass.pipeline, pass.draw) else {
        return Ok(());
    };
    let crate::HalRenderPipeline::Metal(pipeline) = pipeline else {
        return Err(shader_error(
            "render pipeline is not Metal-backed".to_owned(),
        ));
    };
    encoder.setRenderPipelineState(&pipeline.inner);
    for binding in &pass.bind_buffers {
        encode_render_bind_buffer(encoder, binding)?;
    }
    for binding in &pass.vertex_buffers {
        encode_render_vertex_buffer(encoder, binding)?;
    }
    draw_primitives(encoder, pipeline.primitive_topology, draw)?;
    Ok(())
}

/// Records subpass encode into the command stream.
#[cfg(feature = "tiled")]
pub(super) fn encode_subpass_render_pass(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    pass: &HalSubpassRenderPassCommand,
) -> Result<(), HalError> {
    for draw in &pass.draws {
        encode_subpass_draw(encoder, draw)?;
    }
    Ok(())
}

#[cfg(feature = "tiled")]
fn encode_subpass_draw(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    draw: &HalSubpassDraw,
) -> Result<(), HalError> {
    let HalRenderPipeline::Metal(pipeline) = &draw.pipeline else {
        return Err(shader_error(
            "subpass render pipeline is not Metal-backed".to_owned(),
        ));
    };
    encoder.setRenderPipelineState(&pipeline.inner);
    for binding in &draw.bind_buffers {
        encode_render_bind_buffer(encoder, binding)?;
    }
    for binding in &draw.vertex_buffers {
        encode_render_vertex_buffer(encoder, binding)?;
    }
    draw_primitives(encoder, pipeline.primitive_topology, draw.draw)?;
    Ok(())
}

#[cfg(all(test, feature = "tiled"))]
mod tiled_tests {
    use super::*;
    use crate::HalSubpassLayout;

    #[test]
    fn tile_memory_budget_check_accepts_equal_and_rejects_over_budget() {
        assert!(tile_memory_fits_budget(1024, 4, 4096));
        assert!(!tile_memory_fits_budget(1025, 4, 4096));
    }

    #[test]
    fn subpass_color_attachment_indices_returns_union_across_subpasses() {
        let pass = HalSubpassRenderPassCommand {
            layout: HalSubpassPassLayout {
                color_attachments: Vec::new(),
                depth_stencil_attachment: None,
                subpasses: vec![
                    HalSubpassLayout {
                        color_attachment_indices: vec![1, 0],
                        uses_depth_stencil: false,
                        input_attachments: Vec::new(),
                    },
                    HalSubpassLayout {
                        color_attachment_indices: vec![2, 1],
                        uses_depth_stencil: true,
                        input_attachments: Vec::new(),
                    },
                ],
                dependencies: Vec::new(),
            },
            extent: HalExtent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
            draws: Vec::new(),
        };

        assert_eq!(subpass_color_attachment_indices(&pass), vec![0, 1, 2]);
    }
}

fn encode_render_bind_buffer(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    binding: &HalBoundBuffer,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &binding.buffer else {
        return Err(buffer_error("render bind buffer is not Metal-backed"));
    };
    if binding.offset > buffer.size() {
        return Err(buffer_error(
            "render bind buffer offset exceeds buffer size",
        ));
    }
    let index = to_ns(u64::from(binding.metal_index))?;
    let offset = to_ns(binding.offset)?;
    unsafe {
        encoder.setVertexBuffer_offset_atIndex(Some(buffer.inner()?), offset, index);
        encoder.setFragmentBuffer_offset_atIndex(Some(buffer.inner()?), offset, index);
    }
    Ok(())
}

fn encode_render_vertex_buffer(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    binding: &HalBoundBuffer,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &binding.buffer else {
        return Err(buffer_error("render vertex buffer is not Metal-backed"));
    };
    if binding.offset > buffer.size() {
        return Err(buffer_error(
            "render vertex buffer offset exceeds buffer size",
        ));
    }
    unsafe {
        encoder.setVertexBuffer_offset_atIndex(
            Some(buffer.inner()?),
            to_ns(binding.offset)?,
            to_ns(u64::from(binding.metal_index))?,
        );
    }
    Ok(())
}

fn draw_primitives(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    topology: HalPrimitiveTopology,
    draw: HalDraw,
) -> Result<(), HalError> {
    unsafe {
        encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
            map_primitive_topology(topology),
            to_ns(u64::from(draw.first_vertex))?,
            to_ns(u64::from(draw.vertex_count))?,
            to_ns(u64::from(draw.instance_count))?,
            to_ns(u64::from(draw.first_instance))?,
        );
    }
    Ok(())
}
