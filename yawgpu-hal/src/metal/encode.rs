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
