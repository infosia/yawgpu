use super::*;
use crate::format::{format_has_depth_aspect, format_has_stencil_aspect};
use crate::{HalTextureAspect, HalTextureDimension, HalTextureViewDimension};

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

/// Records buffer clear encode into the command stream.
pub(super) fn encode_buffer_clear(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
    clear: &HalBufferClear,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &clear.buffer else {
        return Err(buffer_error("buffer is not Metal-backed"));
    };
    buffer.validate_range(clear.offset, clear.size)?;
    blit.fillBuffer_range_value(
        buffer.inner()?,
        NSRange::new(to_ns(clear.offset)?, to_ns(clear.size)?),
        0,
    );
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
        let bytes_per_image = buffer_texture_bytes_per_image(copy)?;
        match texture.dimension {
            HalTextureDimension::D3 => {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    buffer.inner()?,
                    to_ns(copy.buffer_layout.offset)?,
                    to_ns(u64::from(copy.buffer_layout.bytes_per_row))?,
                    bytes_per_image,
                    to_mtl_size(copy.extent)?,
                    texture.inner()?,
                    0,
                    to_ns(u64::from(copy.mip_level))?,
                    to_mtl_origin(copy.origin.x, copy.origin.y, copy.origin.z)?,
                );
            }
            HalTextureDimension::D1 | HalTextureDimension::D2 => {
                let size = to_mtl_size(HalExtent3d {
                    depth_or_array_layers: 1,
                    ..copy.extent
                })?;
                let option = packed_depth_stencil_blit_option(copy.format, copy.aspect);
                let bytes_per_row = to_ns(u64::from(copy.buffer_layout.bytes_per_row))?;
                let level = to_ns(u64::from(copy.mip_level))?;
                let origin = to_mtl_origin(copy.origin.x, copy.origin.y, 0)?;
                for layer in 0..copy.extent.depth_or_array_layers {
                    let source_offset =
                        layer_buffer_offset(copy.buffer_layout.offset, bytes_per_image, layer)?;
                    let dst_slice = to_ns(u64::from(copy.origin.z + layer))?;
                    if option == MTLBlitOption::empty() {
                        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                            buffer.inner()?,
                            source_offset,
                            bytes_per_row,
                            bytes_per_image,
                            size,
                            texture.inner()?,
                            dst_slice,
                            level,
                            origin,
                        );
                    } else {
                        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                            buffer.inner()?,
                            source_offset,
                            bytes_per_row,
                            bytes_per_image,
                            size,
                            texture.inner()?,
                            dst_slice,
                            level,
                            origin,
                            option,
                        );
                    }
                }
            }
        }
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
        let bytes_per_image = buffer_texture_bytes_per_image(copy)?;
        match texture.dimension {
            HalTextureDimension::D3 => {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                    texture.inner()?,
                    0,
                    to_ns(u64::from(copy.mip_level))?,
                    to_mtl_origin(copy.origin.x, copy.origin.y, copy.origin.z)?,
                    to_mtl_size(copy.extent)?,
                    buffer.inner()?,
                    to_ns(copy.buffer_layout.offset)?,
                    to_ns(u64::from(copy.buffer_layout.bytes_per_row))?,
                    bytes_per_image,
                );
            }
            HalTextureDimension::D1 | HalTextureDimension::D2 => {
                let size = to_mtl_size(HalExtent3d {
                    depth_or_array_layers: 1,
                    ..copy.extent
                })?;
                let option = packed_depth_stencil_blit_option(copy.format, copy.aspect);
                let bytes_per_row = to_ns(u64::from(copy.buffer_layout.bytes_per_row))?;
                let level = to_ns(u64::from(copy.mip_level))?;
                let origin = to_mtl_origin(copy.origin.x, copy.origin.y, 0)?;
                for layer in 0..copy.extent.depth_or_array_layers {
                    let dst_offset =
                        layer_buffer_offset(copy.buffer_layout.offset, bytes_per_image, layer)?;
                    let src_slice = to_ns(u64::from(copy.origin.z + layer))?;
                    if option == MTLBlitOption::empty() {
                        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                            texture.inner()?,
                            src_slice,
                            level,
                            origin,
                            size,
                            buffer.inner()?,
                            dst_offset,
                            bytes_per_row,
                            bytes_per_image,
                        );
                    } else {
                        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                            texture.inner()?,
                            src_slice,
                            level,
                            origin,
                            size,
                            buffer.inner()?,
                            dst_offset,
                            bytes_per_row,
                            bytes_per_image,
                            option,
                        );
                    }
                }
            }
        }
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
        if matches!(source.dimension, HalTextureDimension::D3)
            || matches!(destination.dimension, HalTextureDimension::D3)
        {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                source.inner()?,
                0,
                to_ns(u64::from(copy.source_mip_level))?,
                to_mtl_origin(copy.source_origin.x, copy.source_origin.y, copy.source_origin.z)?,
                to_mtl_size(copy.extent)?,
                destination.inner()?,
                0,
                to_ns(u64::from(copy.destination_mip_level))?,
                to_mtl_origin(
                    copy.destination_origin.x,
                    copy.destination_origin.y,
                    copy.destination_origin.z,
                )?,
            );
        } else {
            let size = to_mtl_size(HalExtent3d {
                depth_or_array_layers: 1,
                ..copy.extent
            })?;
            for layer in 0..copy.extent.depth_or_array_layers {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    source.inner()?,
                    to_ns(u64::from(copy.source_origin.z + layer))?,
                    to_ns(u64::from(copy.source_mip_level))?,
                    to_mtl_origin(copy.source_origin.x, copy.source_origin.y, 0)?,
                    size,
                    destination.inner()?,
                    to_ns(u64::from(copy.destination_origin.z + layer))?,
                    to_ns(u64::from(copy.destination_mip_level))?,
                    to_mtl_origin(copy.destination_origin.x, copy.destination_origin.y, 0)?,
                );
            }
        }
    }
    Ok(())
}

/// Returns the `MTLBlitOption` needed to extract a single plane of a *packed*
/// depth+stencil texture in a buffer⇄texture copy. Single-aspect formats (pure
/// depth, pure stencil, colour) need no option — Metal copies their only plane.
fn packed_depth_stencil_blit_option(
    format: HalTextureFormat,
    aspect: HalTextureAspect,
) -> MTLBlitOption {
    if format_has_depth_aspect(format) && format_has_stencil_aspect(format) {
        match aspect {
            HalTextureAspect::DepthOnly => MTLBlitOption::DepthFromDepthStencil,
            HalTextureAspect::StencilOnly => MTLBlitOption::StencilFromDepthStencil,
            HalTextureAspect::All => MTLBlitOption::empty(),
        }
    } else {
        MTLBlitOption::empty()
    }
}

fn layer_buffer_offset(
    base_offset: u64,
    bytes_per_image: usize,
    layer: u32,
) -> Result<usize, HalError> {
    let bytes_per_image = u64::try_from(bytes_per_image)
        .map_err(|_| buffer_error("buffer texture image is too large"))?;
    let offset = u64::from(layer)
        .checked_mul(bytes_per_image)
        .and_then(|offset| base_offset.checked_add(offset))
        .ok_or_else(|| buffer_error("buffer texture image offset overflows"))?;
    to_ns(offset)
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
    for binding in &pass.bind_textures {
        encode_compute_texture(encoder, binding)?;
    }
    for binding in &pass.bind_samplers {
        encode_compute_sampler(encoder, binding)?;
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        to_mtl_dispatch_size(pass.workgroups)?,
        to_mtl_workgroup_size(pipeline.workgroup_size)?,
    );
    Ok(())
}

fn encode_compute_texture(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    binding: &HalBoundTexture,
) -> Result<(), HalError> {
    let HalTexture::Metal(texture) = &binding.texture else {
        return Err(texture_error("compute texture is not Metal-backed"));
    };
    let view = metal_texture_view(texture, binding)?;
    unsafe {
        encoder.setTexture_atIndex(Some(view.as_ref()), to_ns(u64::from(binding.metal_index))?);
    }
    Ok(())
}

fn encode_compute_sampler(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    binding: &HalBoundSampler,
) -> Result<(), HalError> {
    let HalSampler::Metal(sampler) = &binding.sampler else {
        return Err(texture_error("compute sampler is not Metal-backed"));
    };
    let sampler = sampler
        ._inner
        .as_deref()
        .ok_or_else(|| texture_error("sampler allocation failed"))?;
    unsafe {
        encoder.setSamplerState_atIndex(Some(sampler), to_ns(u64::from(binding.metal_index))?);
    }
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
    let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
    for (index, color_target) in pass.color_targets.iter().enumerate() {
        let HalTexture::Metal(texture) = &color_target.texture else {
            return Err(texture_error("render target is not Metal-backed"));
        };
        let color_attachments = descriptor.colorAttachments();
        let color = unsafe { color_attachments.objectAtIndexedSubscript(to_ns(index as u64)?) };
        color.setTexture(Some(texture.inner()?));
        color.setLevel(to_ns(u64::from(color_target.mip_level))?);
        color.setSlice(to_ns(u64::from(color_target.array_layer))?);
        color.setLoadAction(mtl_load_action(color_target.load_op));
        color.setStoreAction(mtl_store_action(color_target.store));
        let [r, g, b, a] = color_target.clear_color;
        color.setClearColor(MTLClearColor {
            red: r,
            green: g,
            blue: b,
            alpha: a,
        });
    }
    if let Some(depth_stencil) = &pass.depth_stencil_attachment {
        let HalTexture::Metal(texture) = &depth_stencil.texture else {
            return Err(texture_error(
                "depth-stencil attachment is not Metal-backed",
            ));
        };
        if format_has_depth_aspect(depth_stencil.format) {
            let depth_attachment = descriptor.depthAttachment();
            depth_attachment.setTexture(Some(texture.inner()?));
            depth_attachment.setLevel(to_ns(u64::from(depth_stencil.mip_level))?);
            depth_attachment.setSlice(to_ns(u64::from(depth_stencil.array_layer))?);
            depth_attachment.setLoadAction(mtl_load_action(depth_stencil.depth_load_op));
            depth_attachment.setStoreAction(mtl_store_action(depth_stencil.depth_store));
            depth_attachment.setClearDepth(f64::from(depth_stencil.depth_clear_value));
        }
        if format_has_stencil_aspect(depth_stencil.format) {
            let stencil_attachment = descriptor.stencilAttachment();
            stencil_attachment.setTexture(Some(texture.inner()?));
            stencil_attachment.setLevel(to_ns(u64::from(depth_stencil.mip_level))?);
            stencil_attachment.setSlice(to_ns(u64::from(depth_stencil.array_layer))?);
            stencil_attachment.setLoadAction(mtl_load_action(depth_stencil.stencil_load_op));
            stencil_attachment.setStoreAction(mtl_store_action(depth_stencil.stencil_store));
            stencil_attachment.setClearStencil(depth_stencil.stencil_clear_value);
        }
    }
    Ok(descriptor)
}

fn mtl_load_action(load_op: HalRenderLoadOp) -> MTLLoadAction {
    match load_op {
        HalRenderLoadOp::Load => MTLLoadAction::Load,
        HalRenderLoadOp::Clear => MTLLoadAction::Clear,
    }
}

fn mtl_store_action(store: bool) -> MTLStoreAction {
    if store {
        MTLStoreAction::Store
    } else {
        MTLStoreAction::DontCare
    }
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
    if let Some(layout_depth_stencil) = &pass.layout.depth_stencil_attachment {
        if pass
            .layout
            .subpasses
            .iter()
            .any(|subpass| subpass.uses_depth_stencil)
        {
            if let Some(depth) = &pass.depth_stencil_attachment {
                // Gate each side on the format's aspect: binding a depth-only
                // texture (e.g. Depth32Float) to stencilAttachment.setTexture
                // makes Metal silently reject the entire render pass.
                let format = layout_depth_stencil.format;
                if format_has_depth_aspect(format) {
                    let depth_attachment = descriptor.depthAttachment();
                    depth_attachment.setTexture(Some(subpass_attachment_texture(&depth.resource)?));
                    depth_attachment.setLoadAction(mtl_load_action(depth.depth_load_op));
                    depth_attachment.setStoreAction(if depth.depth_store {
                        MTLStoreAction::Store
                    } else {
                        MTLStoreAction::DontCare
                    });
                    depth_attachment.setClearDepth(f64::from(depth.depth_clear_value));
                }
                if format_has_stencil_aspect(format) {
                    let stencil_attachment = descriptor.stencilAttachment();
                    stencil_attachment
                        .setTexture(Some(subpass_attachment_texture(&depth.resource)?));
                    stencil_attachment.setLoadAction(mtl_load_action(depth.stencil_load_op));
                    stencil_attachment.setStoreAction(if depth.stencil_store {
                        MTLStoreAction::Store
                    } else {
                        MTLStoreAction::DontCare
                    });
                    stencil_attachment.setClearStencil(depth.stencil_clear_value);
                }
            }
        }
    }
    Ok(descriptor)
}

// Returns the union of color-attachment flat slots referenced across all
// subpasses, sorted + deduplicated. The MTLRenderPassDescriptor gets one
// `colorAttachments[i]` entry per slot ever used by the pass — a slot
// shared between subpasses (e.g. read in one, written in another) appears
// only once.
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
    encoder.setDepthStencilState(Some(&pipeline.depth_stencil_state));
    encoder.setDepthBias_slopeScale_clamp(
        pipeline.depth_bias as f32,
        pipeline.depth_bias_slope_scale,
        pipeline.depth_bias_clamp,
    );
    encoder.setBlendColorRed_green_blue_alpha(
        pass.blend_constant[0],
        pass.blend_constant[1],
        pass.blend_constant[2],
        pass.blend_constant[3],
    );
    encoder.setStencilReferenceValue(pass.stencil_reference);
    for binding in &pass.bind_buffers {
        encode_render_bind_buffer(encoder, binding)?;
    }
    for binding in &pass.bind_textures {
        encode_render_bind_texture(encoder, binding)?;
    }
    for binding in &pass.bind_samplers {
        encode_render_bind_sampler(encoder, binding)?;
    }
    for binding in &pass.vertex_buffers {
        encode_render_vertex_buffer(encoder, binding)?;
    }
    encode_render_draw(encoder, pass, pipeline.primitive_topology, draw)?;
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
    encoder.setDepthStencilState(Some(&pipeline.depth_stencil_state));
    encoder.setDepthBias_slopeScale_clamp(
        pipeline.depth_bias as f32,
        pipeline.depth_bias_slope_scale,
        pipeline.depth_bias_clamp,
    );
    for binding in &draw.bind_buffers {
        encode_render_bind_buffer(encoder, binding)?;
    }
    for binding in &draw.bind_textures {
        encode_render_bind_texture(encoder, binding)?;
    }
    for binding in &draw.bind_samplers {
        encode_render_bind_sampler(encoder, binding)?;
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

fn encode_render_bind_texture(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    binding: &HalBoundTexture,
) -> Result<(), HalError> {
    let HalTexture::Metal(texture) = &binding.texture else {
        return Err(texture_error("render bind texture is not Metal-backed"));
    };
    let view = metal_texture_view(texture, binding)?;
    let index = to_ns(u64::from(binding.metal_index))?;
    unsafe {
        encoder.setVertexTexture_atIndex(Some(view.as_ref()), index);
        encoder.setFragmentTexture_atIndex(Some(view.as_ref()), index);
    }
    Ok(())
}

fn metal_texture_view(
    texture: &MetalTexture,
    binding: &HalBoundTexture,
) -> Result<Retained<ProtocolObject<dyn MTLTextureTrait>>, HalError> {
    let (pixel_format, _) = map_texture_format(binding.format)?;
    let texture_type = metal_texture_view_type(binding.dimension);
    let level_range = NSRange::new(
        to_ns(u64::from(binding.base_mip_level))?,
        to_ns(u64::from(binding.mip_level_count))?,
    );
    let slice_range = NSRange::new(
        to_ns(u64::from(binding.base_array_layer))?,
        to_ns(u64::from(binding.array_layer_count))?,
    );
    unsafe {
        texture
            .inner()?
            .newTextureViewWithPixelFormat_textureType_levels_slices(
                pixel_format,
                texture_type,
                level_range,
                slice_range,
            )
            .ok_or_else(|| texture_error("texture view allocation failed"))
    }
}

fn metal_texture_view_type(dimension: HalTextureViewDimension) -> MTLTextureType {
    match dimension {
        HalTextureViewDimension::D1 => MTLTextureType::Type1D,
        HalTextureViewDimension::D2 => MTLTextureType::Type2D,
        HalTextureViewDimension::D2Array => MTLTextureType::Type2DArray,
        HalTextureViewDimension::Cube => MTLTextureType::TypeCube,
        HalTextureViewDimension::CubeArray => MTLTextureType::TypeCubeArray,
        HalTextureViewDimension::D3 => MTLTextureType::Type3D,
    }
}

fn encode_render_bind_sampler(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    binding: &HalBoundSampler,
) -> Result<(), HalError> {
    let HalSampler::Metal(sampler) = &binding.sampler else {
        return Err(texture_error("render bind sampler is not Metal-backed"));
    };
    let sampler = sampler
        ._inner
        .as_deref()
        .ok_or_else(|| texture_error("sampler allocation failed"))?;
    let index = to_ns(u64::from(binding.metal_index))?;
    unsafe {
        encoder.setVertexSamplerState_atIndex(Some(sampler), index);
        encoder.setFragmentSamplerState_atIndex(Some(sampler), index);
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

fn encode_render_draw(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    pass: &HalRenderPass,
    topology: HalPrimitiveTopology,
    draw: HalDraw,
) -> Result<(), HalError> {
    match draw {
        HalDraw::Direct {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        } => unsafe {
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                map_primitive_topology(topology),
                to_ns(u64::from(first_vertex))?,
                to_ns(u64::from(vertex_count))?,
                to_ns(u64::from(instance_count))?,
                to_ns(u64::from(first_instance))?,
            );
        },
        HalDraw::Indexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        } => {
            let (buffer, index_type, index_offset) = metal_index_buffer(pass, first_index)?;
            unsafe {
                encoder.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                    map_primitive_topology(topology),
                    to_ns(u64::from(index_count))?,
                    index_type,
                    buffer,
                    to_ns(index_offset)?,
                    to_ns(u64::from(instance_count))?,
                    base_vertex as isize,
                    to_ns(u64::from(first_instance))?,
                );
            }
        }
        HalDraw::Indirect { offset } => {
            let buffer = metal_indirect_buffer(pass)?;
            unsafe {
                encoder.drawPrimitives_indirectBuffer_indirectBufferOffset(
                    map_primitive_topology(topology),
                    buffer,
                    to_ns(offset)?,
                );
            }
        }
        HalDraw::IndexedIndirect { offset } => {
            let (index_buffer, index_type, index_offset) = metal_index_buffer(pass, 0)?;
            let indirect_buffer = metal_indirect_buffer(pass)?;
            unsafe {
                encoder.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                    map_primitive_topology(topology),
                    index_type,
                    index_buffer,
                    to_ns(index_offset)?,
                    indirect_buffer,
                    to_ns(offset)?,
                );
            }
        }
    }
    Ok(())
}

fn metal_index_buffer(
    pass: &HalRenderPass,
    first_index: u32,
) -> Result<(&ProtocolObject<dyn MTLBufferTrait>, MTLIndexType, u64), HalError> {
    let bound = pass
        .index_buffer
        .as_ref()
        .ok_or_else(|| buffer_error("render index buffer is missing"))?;
    let HalBuffer::Metal(buffer) = &bound.buffer else {
        return Err(buffer_error("render index buffer is not Metal-backed"));
    };
    let index_size = match bound.format {
        HalIndexFormat::Uint16 => 2,
        HalIndexFormat::Uint32 => 4,
    };
    let index_offset = bound
        .offset
        .checked_add(u64::from(first_index) * index_size)
        .ok_or_else(|| buffer_error("render index buffer offset overflows"))?;
    Ok((
        buffer.inner()?,
        metal_index_type(bound.format),
        index_offset,
    ))
}

fn metal_indirect_buffer(
    pass: &HalRenderPass,
) -> Result<&ProtocolObject<dyn MTLBufferTrait>, HalError> {
    let bound = pass
        .indirect_buffer
        .as_ref()
        .ok_or_else(|| buffer_error("render indirect buffer is missing"))?;
    let HalBuffer::Metal(buffer) = &bound.buffer else {
        return Err(buffer_error("render indirect buffer is not Metal-backed"));
    };
    buffer.inner()
}

fn metal_index_type(format: HalIndexFormat) -> MTLIndexType {
    match format {
        HalIndexFormat::Uint16 => MTLIndexType::UInt16,
        HalIndexFormat::Uint32 => MTLIndexType::UInt32,
    }
}
