use super::*;
use crate::format::{format_has_depth_aspect, format_has_stencil_aspect};
use crate::{
    HalBoundIndexBuffer, HalBoundIndirectBuffer, HalComponentSwizzle, HalScissorRect,
    HalTextureAspect, HalTextureComponentSwizzle, HalTextureDimension, HalTextureViewDimension,
    HalViewport,
};

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

/// Records texture clear encode into the command stream.
pub(super) fn encode_texture_clear(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
    clear: &HalTextureClear,
) -> Result<(), HalError> {
    let HalTexture::Metal(texture) = &clear.texture else {
        return Err(texture_error("texture is not Metal-backed"));
    };
    if clear.array_layer_count == 0 {
        return Ok(());
    }
    let (width, height, depth) = mip_texture_extent(texture, clear.mip_level)?;
    let bytes_per_row = u64::from(width)
        .checked_mul(u64::from(texture.bytes_per_pixel))
        .ok_or_else(|| texture_error("texture clear row bytes overflow"))?;
    let bytes_per_image = bytes_per_row
        .checked_mul(u64::from(height))
        .ok_or_else(|| texture_error("texture clear image bytes overflow"))?;
    let image_count = match texture.dimension {
        HalTextureDimension::D3 => depth,
        HalTextureDimension::D1 | HalTextureDimension::D2 => clear.array_layer_count,
    };
    let byte_count = bytes_per_image
        .checked_mul(u64::from(image_count))
        .ok_or_else(|| texture_error("texture clear byte count overflows"))?;
    if byte_count == 0 {
        return Ok(());
    }
    let byte_count_ns = to_ns(byte_count)?;
    let zero_buffer = texture
        .device
        .newBufferWithLength_options(byte_count_ns, MTLResourceOptions::StorageModeShared)
        .ok_or(HalError::OutOfMemory {
            backend: BACKEND,
            resource: "texture clear staging buffer",
        })?;
    unsafe {
        std::ptr::write_bytes(
            zero_buffer.contents().cast::<u8>().as_ptr(),
            0,
            byte_count_ns,
        );
        let level = to_ns(u64::from(clear.mip_level))?;
        let size = to_mtl_size(HalExtent3d {
            width,
            height,
            depth_or_array_layers: match texture.dimension {
                HalTextureDimension::D3 => depth,
                HalTextureDimension::D1 | HalTextureDimension::D2 => 1,
            },
        })?;
        match texture.dimension {
            HalTextureDimension::D3 => {
                blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &zero_buffer,
                    0,
                    to_ns(bytes_per_row)?,
                    to_ns(bytes_per_image)?,
                    size,
                    texture.inner()?,
                    0,
                    level,
                    to_mtl_origin(0, 0, 0)?,
                );
            }
            HalTextureDimension::D1 | HalTextureDimension::D2 => {
                for layer in 0..clear.array_layer_count {
                    let source_offset = layer_buffer_offset(0, to_ns(bytes_per_image)?, layer)?;
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &zero_buffer,
                        source_offset,
                        to_ns(bytes_per_row)?,
                        to_ns(bytes_per_image)?,
                        size,
                        texture.inner()?,
                        to_ns(u64::from(clear.base_array_layer + layer))?,
                        level,
                        to_mtl_origin(0, 0, 0)?,
                    );
                }
            }
        }
    }
    Ok(())
}

fn mip_texture_extent(texture: &MetalTexture, mip_level: u32) -> Result<(u32, u32, u32), HalError> {
    let mip = |value: u32| value.checked_shr(mip_level).unwrap_or(0).max(1);
    Ok((
        mip(texture.width),
        match texture.dimension {
            HalTextureDimension::D1 => 1,
            HalTextureDimension::D2 | HalTextureDimension::D3 => mip(texture.height),
        },
        match texture.dimension {
            HalTextureDimension::D3 => mip(texture.depth_or_array_layers),
            HalTextureDimension::D1 | HalTextureDimension::D2 => 1,
        },
    ))
}

/// Records query-set resolve encode into the command stream.
pub(super) fn encode_resolve_query_set(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
    resolve: &HalResolveQuerySet,
) -> Result<(), HalError> {
    let HalQuerySet::Metal(query_set) = &resolve.query_set else {
        return Err(buffer_error("query set is not Metal-backed"));
    };
    let HalBuffer::Metal(destination) = &resolve.destination else {
        return Err(buffer_error(
            "query resolve destination is not Metal-backed",
        ));
    };
    let source_offset = u64::from(resolve.first_query)
        .checked_mul(8)
        .ok_or_else(|| buffer_error("query resolve source offset overflows"))?;
    let size = u64::from(resolve.query_count)
        .checked_mul(8)
        .ok_or_else(|| buffer_error("query resolve byte count overflows"))?;
    destination.validate_range(resolve.destination_offset, size)?;
    query_set.buffer.validate_range(source_offset, size)?;
    for &query_index in &resolve.written_queries {
        if query_index < resolve.first_query {
            return Err(buffer_error("written query precedes resolve range"));
        }
        let relative_index = query_index - resolve.first_query;
        if relative_index >= resolve.query_count {
            return Err(buffer_error("written query exceeds resolve range"));
        }
        let source_offset = u64::from(query_index)
            .checked_mul(8)
            .ok_or_else(|| buffer_error("query resolve source offset overflows"))?;
        query_set.buffer.validate_range(source_offset, 8)?;
    }
    if size == 0 {
        return Ok(());
    }
    unsafe {
        let destination_buffer = destination.inner()?;
        // MTLBlitCommandEncoder preserves command order here: the zero-fill
        // completes before the overlapping per-query blit copies.
        blit.fillBuffer_range_value(
            destination_buffer,
            NSRange::new(to_ns(resolve.destination_offset)?, to_ns(size)?),
            0,
        );
        for &query_index in &resolve.written_queries {
            let source_offset = u64::from(query_index)
                .checked_mul(8)
                .ok_or_else(|| buffer_error("query resolve source offset overflows"))?;
            let destination_offset = resolve
                .destination_offset
                .checked_add(u64::from(query_index - resolve.first_query) * 8)
                .ok_or_else(|| buffer_error("query resolve destination offset overflows"))?;
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                query_set.buffer()?,
                to_ns(source_offset)?,
                destination_buffer,
                to_ns(destination_offset)?,
                8,
            );
        }
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
    // Allocate threadgroup (workgroup) memory slots required by var<workgroup>
    // globals emitted by Tint as [[threadgroup(N)]] arguments. Metal requires
    // each slot to be explicitly sized before dispatch; without this every slot
    // reads as zero. Mirrors wgpu-hal/src/metal/command.rs set_compute_pipeline
    // (lines 1756-1773). The sizes are already rounded to a multiple of 16 by
    // collect_workgroup_memory_sizes in yawgpu-core.
    for (index, &size) in pipeline.workgroup_memory_sizes.iter().enumerate() {
        unsafe {
            encoder.setThreadgroupMemoryLength_atIndex(size as usize, index);
        }
    }
    for binding in &pass.bind_buffers {
        encode_compute_buffer(encoder, binding)?;
    }
    encode_compute_buffer_sizes(encoder, pipeline, &pass.bind_buffers)?;
    encode_compute_immediates(encoder, pipeline, &pass.immediate_data)?;
    for binding in &pass.bind_textures {
        encode_compute_texture(encoder, binding)?;
    }
    for binding in &pass.bind_external_textures {
        encode_compute_external_texture(encoder, binding)?;
    }
    for binding in &pass.bind_samplers {
        encode_compute_sampler(encoder, binding)?;
    }
    match &pass.dispatch {
        HalComputeDispatch::Direct { workgroups } => {
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                to_mtl_dispatch_size(*workgroups)?,
                to_mtl_workgroup_size(pipeline.workgroup_size)?,
            );
        }
        HalComputeDispatch::Indirect { buffer } => {
            let HalBuffer::Metal(indirect_buffer) = &buffer.buffer else {
                return Err(buffer_error("compute indirect buffer is not Metal-backed"));
            };
            unsafe {
                encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                    indirect_buffer.inner()?,
                    to_ns(buffer.offset)?,
                    to_mtl_workgroup_size(pipeline.workgroup_size)?,
                );
            }
        }
    }
    Ok(())
}

fn encode_compute_buffer_sizes(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MetalComputePipeline,
    buffers: &[HalBoundBuffer],
) -> Result<(), HalError> {
    let Some(slot) = pipeline.buffer_sizes_slot else {
        return Ok(());
    };
    let sizes = msl_buffer_sizes(&pipeline.buffer_size_bindings, buffers)?;
    if sizes.is_empty() {
        return Ok(());
    }
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new(sizes.as_ptr().cast_mut().cast())
                .ok_or_else(|| buffer_error("MSL buffer sizes data is missing"))?,
            sizes.len() * std::mem::size_of::<u32>(),
            to_ns(u64::from(slot))?,
        );
    }
    Ok(())
}

/// Composes the combined Metal immediates block for one shader stage
/// (Block 94 S2) via the backend-shared
/// [`crate::immediates::compose_immediates_block`]: user immediate bytes
/// first, the fragment frag-depth clamp range (from the pass viewport,
/// defaulting to the full `[0.0, 1.0]` range) at
/// `immediates.frag_depth_clamp_offset` when present. Mirrors Dawn's
/// `RenderImmediates`/`ComputeImmediates` layout (`ImmediatesLayoutMTL.h`).
fn compose_metal_immediates_block(
    user_data: &[u8],
    immediates: HalMslImmediates,
    viewport: Option<HalViewport>,
) -> Vec<u8> {
    let range = viewport.map_or([0.0f32, 1.0], |viewport| {
        [viewport.min_depth, viewport.max_depth]
    });
    crate::immediates::compose_immediates_block(
        user_data,
        immediates.block_size,
        immediates.frag_depth_clamp_offset,
        range,
    )
}

fn encode_compute_immediates(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &MetalComputePipeline,
    immediate_data: &[u8],
) -> Result<(), HalError> {
    let Some(immediates) = pipeline.immediates else {
        return Ok(());
    };
    let block = compose_metal_immediates_block(immediate_data, immediates, None);
    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new(block.as_ptr().cast_mut().cast())
                .ok_or_else(|| buffer_error("compute immediates block data is missing"))?,
            block.len(),
            to_ns(u64::from(immediates.slot))?,
        );
    }
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

fn encode_compute_external_texture(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    binding: &HalBoundExternalTexture,
) -> Result<(), HalError> {
    let HalTexture::Metal(plane0) = &binding.plane0 else {
        return Err(texture_error(
            "compute external texture plane0 is not Metal-backed",
        ));
    };
    let HalTexture::Metal(plane1) = &binding.plane1 else {
        return Err(texture_error(
            "compute external texture plane1 is not Metal-backed",
        ));
    };
    let HalBuffer::Metal(params) = &binding.params else {
        return Err(buffer_error(
            "compute external texture params buffer is not Metal-backed",
        ));
    };
    if binding.params_offset > params.size() {
        return Err(buffer_error(
            "compute external texture params offset exceeds buffer size",
        ));
    }
    unsafe {
        encoder.setTexture_atIndex(
            Some(plane0.inner()?),
            to_ns(u64::from(binding.plane0_metal_index))?,
        );
        encoder.setTexture_atIndex(
            Some(plane1.inner()?),
            to_ns(u64::from(binding.plane1_metal_index))?,
        );
        encoder.setBuffer_offset_atIndex(
            Some(params.inner()?),
            to_ns(binding.params_offset)?,
            to_ns(u64::from(binding.params_metal_index))?,
        );
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

fn msl_buffer_sizes(
    size_bindings: &[HalMslBufferSizeBinding],
    buffers: &[HalBoundBuffer],
) -> Result<Vec<u32>, HalError> {
    size_bindings
        .iter()
        .map(|size_binding| {
            let Some(bound) = buffers.iter().find(|bound| {
                bound.group == size_binding.group && bound.binding == size_binding.binding
            }) else {
                return Ok(0);
            };
            let size = bound_buffer_size(bound)?;
            msl_buffer_size_u32(size)
        })
        .collect()
}

fn msl_buffer_size_u32(size: u64) -> Result<u32, HalError> {
    u32::try_from(size).map_err(|_| buffer_error("MSL buffer size exceeds u32"))
}

fn bound_buffer_size(bound: &HalBoundBuffer) -> Result<u64, HalError> {
    let HalBuffer::Metal(buffer) = &bound.buffer else {
        return Err(buffer_error("MSL buffer-size binding is not Metal-backed"));
    };
    if bound.offset > buffer.size() {
        return Err(buffer_error(
            "MSL buffer-size binding offset exceeds buffer size",
        ));
    }
    if bound.size == u64::MAX {
        buffer
            .size()
            .checked_sub(bound.offset)
            .ok_or_else(|| buffer_error("MSL buffer-size binding range exceeds buffer size"))
    } else {
        Ok(bound.size)
    }
}

/// Returns render pass descriptor.
pub(super) fn render_pass_descriptor(
    pass: &HalRenderPass,
) -> Result<Retained<MTLRenderPassDescriptor>, HalError> {
    let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
    for (index, color_target) in pass.color_targets.iter().enumerate() {
        let Some(color_target) = color_target else {
            continue;
        };
        let HalTexture::Metal(texture) = &color_target.texture else {
            return Err(texture_error("render target is not Metal-backed"));
        };
        let color_attachments = descriptor.colorAttachments();
        let color = unsafe { color_attachments.objectAtIndexedSubscript(to_ns(index as u64)?) };
        color.setTexture(Some(texture.inner()?));
        color.setLevel(to_ns(u64::from(color_target.mip_level))?);
        if texture.dimension == HalTextureDimension::D3 {
            color.setSlice(0);
            color.setDepthPlane(to_ns(u64::from(color_target.depth_slice))?);
        } else {
            color.setSlice(to_ns(u64::from(color_target.array_layer))?);
            color.setDepthPlane(0);
        }
        color.setLoadAction(mtl_load_action(color_target.load_op));
        if let Some(resolve_target) = &color_target.resolve_target {
            let HalTexture::Metal(resolve_texture) = resolve_target else {
                return Err(texture_error("resolve target is not Metal-backed"));
            };
            color.setResolveTexture(Some(resolve_texture.inner()?));
            color.setResolveLevel(to_ns(u64::from(color_target.resolve_mip_level))?);
            color.setResolveSlice(to_ns(u64::from(color_target.resolve_array_layer))?);
            color.setStoreAction(if color_target.store {
                MTLStoreAction::StoreAndMultisampleResolve
            } else {
                MTLStoreAction::MultisampleResolve
            });
        } else {
            color.setStoreAction(mtl_store_action(color_target.store));
        }
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
    match &pass.occlusion_query_set {
        Some(HalQuerySet::Metal(query_set)) => {
            descriptor.setVisibilityResultBuffer(Some(query_set.buffer()?));
        }
        Some(_) => return Err(buffer_error("occlusion query set is not Metal-backed")),
        None => {}
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

/// Returns subpass render pass descriptor.
#[cfg(feature = "tiled")]
pub(super) fn subpass_render_pass_descriptor(
    pass: &HalSubpassRenderPassCommand,
) -> Result<Retained<MTLRenderPassDescriptor>, HalError> {
    let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
    if pass.layout.subpasses.is_empty() {
        return Err(texture_error(
            "subpass render pass requires at least one subpass",
        ));
    }
    let color_attachments = descriptor.colorAttachments();
    for attachment_index in subpass_color_attachment_indices(pass) {
        let slot = to_ns(u64::from(attachment_index))?;
        let binding = pass
            .color_attachments
            .get(attachment_index as usize)
            .ok_or_else(|| texture_error("subpass color attachment binding missing"))?;
        let color = unsafe { color_attachments.objectAtIndexedSubscript(slot) };
        color.setTexture(Some(subpass_attachment_texture(&binding.resource)?));
        color.setLoadAction(mtl_load_action(binding.load_op));
        color.setStoreAction(mtl_store_action(binding.store));
        let [r, g, b, a] = binding.clear_color;
        color.setClearColor(MTLClearColor {
            red: r,
            green: g,
            blue: b,
            alpha: a,
        });
    }
    if let Some(layout_depth_stencil) = &pass.layout.depth_stencil_attachment {
        let uses_depth_stencil = pass
            .layout
            .subpasses
            .iter()
            .any(|subpass| subpass.uses_depth_stencil);
        if uses_depth_stencil {
            if let Some(depth) = &pass.depth_stencil_attachment {
                let format = layout_depth_stencil.format;
                if format_has_depth_aspect(format) {
                    let depth_attachment = descriptor.depthAttachment();
                    depth_attachment.setTexture(Some(subpass_attachment_texture(&depth.resource)?));
                    depth_attachment.setLoadAction(mtl_load_action(depth.depth_load_op));
                    depth_attachment.setStoreAction(mtl_store_action(depth.depth_store));
                    depth_attachment.setClearDepth(f64::from(depth.depth_clear_value));
                }
                if format_has_stencil_aspect(format) {
                    let stencil_attachment = descriptor.stencilAttachment();
                    stencil_attachment
                        .setTexture(Some(subpass_attachment_texture(&depth.resource)?));
                    stencil_attachment.setLoadAction(mtl_load_action(depth.stencil_load_op));
                    stencil_attachment.setStoreAction(mtl_store_action(depth.stencil_store));
                    stencil_attachment.setClearStencil(depth.stencil_clear_value);
                }
            }
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
    #[allow(unreachable_patterns)]
    match resource {
        HalSubpassAttachmentResource::Persistent { texture, .. } => {
            let HalTexture::Metal(texture) = texture else {
                return Err(texture_error("subpass attachment is not Metal-backed"));
            };
            texture.inner()
        }
        _ => Err(texture_error(
            "transient subpass attachments not yet supported",
        )),
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
    let state = RenderEncodeState {
        pipeline,
        bind_buffers: &pass.bind_buffers,
        bind_textures: &pass.bind_textures,
        bind_external_textures: &pass.bind_external_textures,
        bind_samplers: &pass.bind_samplers,
        vertex_buffers: &pass.vertex_buffers,
        index_buffer: pass.index_buffer.as_deref(),
        indirect_buffer: pass.indirect_buffer.as_deref(),
        viewport: pass.viewport,
        scissor_rect: pass.scissor_rect,
        blend_constant: pass.blend_constant,
        stencil_reference: pass.stencil_reference,
        occlusion_query_index: pass.occlusion_query_index,
        draw,
        immediate_data: &pass.immediate_data,
    };
    encode_render_state(encoder, &state)
}

/// Records subpass encode into the command stream.
#[cfg(feature = "tiled")]
pub(super) fn encode_subpass_render_pass(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    pass: &HalSubpassRenderPassCommand,
) -> Result<(), HalError> {
    for draw in &pass.draws {
        let crate::HalRenderPipeline::Metal(pipeline) = &draw.pipeline else {
            return Err(shader_error(
                "subpass render pipeline is not Metal-backed".to_owned(),
            ));
        };
        let state = RenderEncodeState {
            pipeline,
            bind_buffers: &draw.bind_buffers,
            bind_textures: &draw.bind_textures,
            bind_external_textures: &[],
            bind_samplers: &draw.bind_samplers,
            vertex_buffers: &draw.vertex_buffers,
            index_buffer: None,
            indirect_buffer: None,
            viewport: draw.viewport,
            scissor_rect: draw.scissor_rect,
            blend_constant: [0.0, 0.0, 0.0, 0.0],
            stencil_reference: 0,
            occlusion_query_index: None,
            draw: draw.draw,
            // The tiled subpass vendor extension has no `SetImmediates`
            // surface (the subpass encoder records no immediates scratch),
            // and core validation rejects subpass pipelines whose stages
            // use `var<immediate>` data at creation
            // (`validate_subpass_pipeline_has_no_immediates`, Block 94
            // Phase Review MAJOR 2) -- so no pipeline reaching this path
            // reads user immediates and the empty snapshot is never
            // observable. Only the internal frag-depth clamp range (which
            // reads no user bytes) can still be composed and delivered.
            immediate_data: &[],
        };
        encode_render_state(encoder, &state)?;
    }
    Ok(())
}

struct RenderEncodeState<'a> {
    pipeline: &'a MetalRenderPipeline,
    bind_buffers: &'a [HalBoundBuffer],
    bind_textures: &'a [HalBoundTexture],
    bind_external_textures: &'a [HalBoundExternalTexture],
    bind_samplers: &'a [HalBoundSampler],
    vertex_buffers: &'a [HalBoundBuffer],
    index_buffer: Option<&'a HalBoundIndexBuffer>,
    indirect_buffer: Option<&'a HalBoundIndirectBuffer>,
    viewport: Option<HalViewport>,
    scissor_rect: Option<HalScissorRect>,
    blend_constant: [f32; 4],
    stencil_reference: u32,
    occlusion_query_index: Option<u32>,
    draw: HalDraw,
    /// The pass's immediates scratch snapshot for this draw (Block 94).
    immediate_data: &'a [u8],
}

fn encode_render_state(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    state: &RenderEncodeState<'_>,
) -> Result<(), HalError> {
    let pipeline = state.pipeline;
    encoder.setRenderPipelineState(&pipeline.inner);
    encoder.setDepthStencilState(Some(&pipeline.depth_stencil_state));
    encoder.setFrontFacingWinding(mtl_front_face(pipeline.front_face));
    encoder.setCullMode(mtl_cull_mode(pipeline.cull_mode));
    encoder.setDepthClipMode(if pipeline.unclipped_depth {
        MTLDepthClipMode::Clamp
    } else {
        MTLDepthClipMode::Clip
    });
    if let Some(viewport) = state.viewport {
        encoder.setViewport(MTLViewport {
            originX: f64::from(viewport.x),
            originY: f64::from(viewport.y),
            width: f64::from(viewport.width),
            height: f64::from(viewport.height),
            znear: f64::from(viewport.min_depth),
            zfar: f64::from(viewport.max_depth),
        });
    }
    if let Some(rect) = state.scissor_rect {
        encoder.setScissorRect(MTLScissorRect {
            x: to_ns(u64::from(rect.x))?,
            y: to_ns(u64::from(rect.y))?,
            width: to_ns(u64::from(rect.width))?,
            height: to_ns(u64::from(rect.height))?,
        });
    }
    encoder.setDepthBias_slopeScale_clamp(
        pipeline.depth_bias as f32,
        pipeline.depth_bias_slope_scale,
        pipeline.depth_bias_clamp,
    );
    encoder.setBlendColorRed_green_blue_alpha(
        state.blend_constant[0],
        state.blend_constant[1],
        state.blend_constant[2],
        state.blend_constant[3],
    );
    encoder.setStencilReferenceValue(state.stencil_reference);
    if let Some(query_index) = state.occlusion_query_index {
        encoder.setVisibilityResultMode_offset(
            MTLVisibilityResultMode::Counting,
            to_ns(u64::from(query_index) * 8)?,
        );
    } else {
        encoder.setVisibilityResultMode_offset(MTLVisibilityResultMode::Disabled, 0);
    }
    for binding in state.bind_buffers {
        encode_render_bind_buffer(encoder, binding)?;
    }
    encode_render_buffer_sizes(encoder, pipeline, state.bind_buffers, state.vertex_buffers)?;
    encode_render_immediates(encoder, pipeline, state.immediate_data, state.viewport)?;
    for binding in state.bind_textures {
        encode_render_bind_texture(encoder, binding)?;
    }
    for binding in state.bind_external_textures {
        encode_render_bind_external_texture(encoder, binding)?;
    }
    for binding in state.bind_samplers {
        encode_render_bind_sampler(encoder, binding)?;
    }
    for binding in state.vertex_buffers {
        encode_render_vertex_buffer(encoder, binding)?;
    }
    encode_render_draw(encoder, state, pipeline.primitive_topology, state.draw)
}

fn mtl_front_face(front_face: HalFrontFace) -> MTLWinding {
    match front_face {
        HalFrontFace::Ccw => MTLWinding::CounterClockwise,
        HalFrontFace::Cw => MTLWinding::Clockwise,
    }
}

fn mtl_cull_mode(cull_mode: HalCullMode) -> MTLCullMode {
    match cull_mode {
        HalCullMode::None => MTLCullMode::None,
        HalCullMode::Front => MTLCullMode::Front,
        HalCullMode::Back => MTLCullMode::Back,
    }
}

/// Composes the full vertex-stage `_mslBufferSizes` array.
///
/// Layout Tint emits:
///   [storage-array sizes …] [buffer_sizeN per vertex_buffer_metal_indices entry]
///
/// `bind_buffers` supplies the bind-group buffers (storage-array entries).
/// `vertex_buffers` supplies the vertex-attribute buffers; each entry in
/// `vertex_buffer_metal_indices` is looked up by `metal_index`.  The effective
/// size is `buffer.size − bind_offset`, saturating to 0; a missing binding
/// yields 0.  All sizes are saturating-cast to `u32`.
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
fn compose_vertex_stage_sizes(
    storage_bindings: &[HalMslBufferSizeBinding],
    bind_buffers: &[HalBoundBuffer],
    vertex_buffer_metal_indices: &[u32],
    vertex_buffers: &[HalBoundBuffer],
) -> Result<Vec<u32>, HalError> {
    // Storage-array sizes first.
    let mut sizes = msl_buffer_sizes(storage_bindings, bind_buffers)?;
    // Vertex buffer sizes appended in vertex_buffer_mappings order.
    for &metal_index in vertex_buffer_metal_indices {
        let effective_size = vertex_buffers
            .iter()
            .find(|vb| vb.metal_index == metal_index)
            .map(|vb| {
                let HalBuffer::Metal(buffer) = &vb.buffer else {
                    return 0u64;
                };
                buffer.size().saturating_sub(vb.offset)
            })
            .unwrap_or(0);
        sizes.push(u32::try_from(effective_size).unwrap_or(u32::MAX));
    }
    Ok(sizes)
}

/// Composes the vertex-stage `_mslBufferSizes` array and writes it via
/// `setVertexBytes`, then composes the fragment-stage array and writes it via
/// `setFragmentBytes`.
///
/// The vertex-stage slot is forced when vertex buffers exist, so `sizes` will
/// be non-empty in that case.  See [`compose_vertex_stage_sizes`] for the exact
/// layout.
fn encode_render_buffer_sizes(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    pipeline: &MetalRenderPipeline,
    bind_buffers: &[HalBoundBuffer],
    vertex_buffers: &[HalBoundBuffer],
) -> Result<(), HalError> {
    if let Some(slot) = pipeline.vertex_buffer_sizes_slot {
        let sizes = compose_vertex_stage_sizes(
            &pipeline.vertex_buffer_size_bindings,
            bind_buffers,
            &pipeline.vertex_buffer_metal_indices,
            vertex_buffers,
        )?;
        if !sizes.is_empty() {
            unsafe {
                encoder.setVertexBytes_length_atIndex(
                    NonNull::new(sizes.as_ptr().cast_mut().cast())
                        .ok_or_else(|| buffer_error("MSL vertex buffer sizes data is missing"))?,
                    sizes.len() * std::mem::size_of::<u32>(),
                    to_ns(u64::from(slot))?,
                );
            }
        }
    }
    if let Some(slot) = pipeline.fragment_buffer_sizes_slot {
        let sizes = msl_buffer_sizes(&pipeline.fragment_buffer_size_bindings, bind_buffers)?;
        if !sizes.is_empty() {
            unsafe {
                encoder.setFragmentBytes_length_atIndex(
                    NonNull::new(sizes.as_ptr().cast_mut().cast())
                        .ok_or_else(|| buffer_error("MSL fragment buffer sizes data is missing"))?,
                    sizes.len() * std::mem::size_of::<u32>(),
                    to_ns(u64::from(slot))?,
                );
            }
        }
    }
    Ok(())
}

/// Delivers the vertex- and/or fragment-stage combined immediates block for
/// one draw (Block 94 S2). Delivered unconditionally per draw (no dirty
/// tracking, mirroring the pre-S2 clamp-only delivery this absorbs) at each
/// stage's own Metal slot; a stage with no immediates (`pipeline.
/// {vertex,fragment}_immediates == None`) is skipped entirely. Replaces the
/// old `encode_render_frag_depth_clamp`, which only ever delivered the bare
/// 8-byte clamp range at the fragment stage -- clamp-only pipelines still
/// get exactly that (the composed block is just `[0, 8)` = the clamp range
/// when there is no user-immediate prefix).
fn encode_render_immediates(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    pipeline: &MetalRenderPipeline,
    immediate_data: &[u8],
    viewport: Option<HalViewport>,
) -> Result<(), HalError> {
    if let Some(immediates) = pipeline.vertex_immediates {
        // Vertex stage never carries the frag-depth clamp.
        let block = compose_metal_immediates_block(immediate_data, immediates, None);
        unsafe {
            encoder.setVertexBytes_length_atIndex(
                NonNull::new(block.as_ptr().cast_mut().cast())
                    .ok_or_else(|| buffer_error("vertex immediates block data is missing"))?,
                block.len(),
                to_ns(u64::from(immediates.slot))?,
            );
        }
    }
    if let Some(immediates) = pipeline.fragment_immediates {
        let block = compose_metal_immediates_block(immediate_data, immediates, viewport);
        unsafe {
            encoder.setFragmentBytes_length_atIndex(
                NonNull::new(block.as_ptr().cast_mut().cast())
                    .ok_or_else(|| buffer_error("fragment immediates block data is missing"))?,
                block.len(),
                to_ns(u64::from(immediates.slot))?,
            );
        }
    }
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
    let offset = to_ns(binding.offset)?;
    // Per-stage binding: use stage-specific slots when available, otherwise
    // fall back to the flat metal_index for both stages (backwards compat).
    if let Some(vtx) = binding.vertex_metal_index {
        let vtx_index = to_ns(u64::from(vtx))?;
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(buffer.inner()?), offset, vtx_index);
        }
    } else if binding.fragment_metal_index.is_none() {
        // No per-stage info: bind to both stages at the flat index (compute
        // code path reuses this function for bind-group buffers in render
        // pipelines that predate per-stage maps; also handles Noop).
        let index = to_ns(u64::from(binding.metal_index))?;
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(buffer.inner()?), offset, index);
        }
    }
    if let Some(frag) = binding.fragment_metal_index {
        let frag_index = to_ns(u64::from(frag))?;
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(Some(buffer.inner()?), offset, frag_index);
        }
    } else if binding.vertex_metal_index.is_none() {
        // Symmetric fallback for fragment stage.
        let index = to_ns(u64::from(binding.metal_index))?;
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(Some(buffer.inner()?), offset, index);
        }
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
    // Per-stage binding: use stage-specific texture slots when available.
    if let Some(vtx) = binding.vertex_metal_index {
        let vtx_index = to_ns(u64::from(vtx))?;
        unsafe {
            encoder.setVertexTexture_atIndex(Some(view.as_ref()), vtx_index);
        }
    } else if binding.fragment_metal_index.is_none() {
        let index = to_ns(u64::from(binding.metal_index))?;
        unsafe {
            encoder.setVertexTexture_atIndex(Some(view.as_ref()), index);
        }
    }
    if let Some(frag) = binding.fragment_metal_index {
        let frag_index = to_ns(u64::from(frag))?;
        unsafe {
            encoder.setFragmentTexture_atIndex(Some(view.as_ref()), frag_index);
        }
    } else if binding.vertex_metal_index.is_none() {
        let index = to_ns(u64::from(binding.metal_index))?;
        unsafe {
            encoder.setFragmentTexture_atIndex(Some(view.as_ref()), index);
        }
    }
    Ok(())
}

fn encode_render_bind_external_texture(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    binding: &HalBoundExternalTexture,
) -> Result<(), HalError> {
    let HalTexture::Metal(plane0) = &binding.plane0 else {
        return Err(texture_error(
            "render external texture plane0 is not Metal-backed",
        ));
    };
    let HalTexture::Metal(plane1) = &binding.plane1 else {
        return Err(texture_error(
            "render external texture plane1 is not Metal-backed",
        ));
    };
    let HalBuffer::Metal(params) = &binding.params else {
        return Err(buffer_error(
            "render external texture params buffer is not Metal-backed",
        ));
    };
    if binding.params_offset > params.size() {
        return Err(buffer_error(
            "render external texture params offset exceeds buffer size",
        ));
    }
    let offset = to_ns(binding.params_offset)?;

    if let Some(vtx) = binding.plane0_vertex_metal_index {
        unsafe {
            encoder.setVertexTexture_atIndex(Some(plane0.inner()?), to_ns(u64::from(vtx))?);
            encoder.setVertexTexture_atIndex(
                Some(plane1.inner()?),
                to_ns(u64::from(binding.plane1_vertex_metal_index.ok_or_else(
                    || texture_error("render external texture plane1 vertex slot is missing"),
                )?))?,
            );
        }
    } else if binding.plane0_fragment_metal_index.is_none() {
        unsafe {
            encoder.setVertexTexture_atIndex(
                Some(plane0.inner()?),
                to_ns(u64::from(binding.plane0_metal_index))?,
            );
            encoder.setVertexTexture_atIndex(
                Some(plane1.inner()?),
                to_ns(u64::from(binding.plane1_metal_index))?,
            );
        }
    }
    if let Some(frag) = binding.plane0_fragment_metal_index {
        unsafe {
            encoder.setFragmentTexture_atIndex(Some(plane0.inner()?), to_ns(u64::from(frag))?);
            encoder.setFragmentTexture_atIndex(
                Some(plane1.inner()?),
                to_ns(u64::from(binding.plane1_fragment_metal_index.ok_or_else(
                    || texture_error("render external texture plane1 fragment slot is missing"),
                )?))?,
            );
        }
    } else if binding.plane0_vertex_metal_index.is_none() {
        unsafe {
            encoder.setFragmentTexture_atIndex(
                Some(plane0.inner()?),
                to_ns(u64::from(binding.plane0_metal_index))?,
            );
            encoder.setFragmentTexture_atIndex(
                Some(plane1.inner()?),
                to_ns(u64::from(binding.plane1_metal_index))?,
            );
        }
    }

    if let Some(vtx) = binding.params_vertex_metal_index {
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(
                Some(params.inner()?),
                offset,
                to_ns(u64::from(vtx))?,
            );
        }
    } else if binding.params_fragment_metal_index.is_none() {
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(
                Some(params.inner()?),
                offset,
                to_ns(u64::from(binding.params_metal_index))?,
            );
        }
    }
    if let Some(frag) = binding.params_fragment_metal_index {
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(
                Some(params.inner()?),
                offset,
                to_ns(u64::from(frag))?,
            );
        }
    } else if binding.params_vertex_metal_index.is_none() {
        unsafe {
            encoder.setFragmentBuffer_offset_atIndex(
                Some(params.inner()?),
                offset,
                to_ns(u64::from(binding.params_metal_index))?,
            );
        }
    }
    Ok(())
}

fn metal_texture_view(
    texture: &MetalTexture,
    binding: &HalBoundTexture,
) -> Result<Retained<ProtocolObject<dyn MTLTextureTrait>>, HalError> {
    let pixel_format =
        map_sampled_view_format_for_texture(texture.format, binding.format, binding.aspect)?;
    let texture_type =
        if binding.dimension == HalTextureViewDimension::D2 && texture.sample_count > 1 {
            MTLTextureType::Type2DMultisample
        } else {
            metal_texture_view_type(binding.dimension)
        };
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
            .newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                pixel_format,
                texture_type,
                level_range,
                slice_range,
                metal_texture_swizzle_channels(&binding.swizzle),
            )
            .ok_or_else(|| texture_error("texture view allocation failed"))
    }
}

fn metal_texture_swizzle_channels(
    swizzle: &HalTextureComponentSwizzle,
) -> MTLTextureSwizzleChannels {
    fn component(component: HalComponentSwizzle) -> MTLTextureSwizzle {
        match component {
            HalComponentSwizzle::Zero => MTLTextureSwizzle::Zero,
            HalComponentSwizzle::One => MTLTextureSwizzle::One,
            HalComponentSwizzle::R => MTLTextureSwizzle::Red,
            HalComponentSwizzle::G => MTLTextureSwizzle::Green,
            HalComponentSwizzle::B => MTLTextureSwizzle::Blue,
            HalComponentSwizzle::A => MTLTextureSwizzle::Alpha,
        }
    }

    MTLTextureSwizzleChannels {
        red: component(swizzle.r),
        green: component(swizzle.g),
        blue: component(swizzle.b),
        alpha: component(swizzle.a),
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
    // Per-stage binding: use stage-specific sampler slots when available.
    if let Some(vtx) = binding.vertex_metal_index {
        let vtx_index = to_ns(u64::from(vtx))?;
        unsafe {
            encoder.setVertexSamplerState_atIndex(Some(sampler), vtx_index);
        }
    } else if binding.fragment_metal_index.is_none() {
        let index = to_ns(u64::from(binding.metal_index))?;
        unsafe {
            encoder.setVertexSamplerState_atIndex(Some(sampler), index);
        }
    }
    if let Some(frag) = binding.fragment_metal_index {
        let frag_index = to_ns(u64::from(frag))?;
        unsafe {
            encoder.setFragmentSamplerState_atIndex(Some(sampler), frag_index);
        }
    } else if binding.vertex_metal_index.is_none() {
        let index = to_ns(u64::from(binding.metal_index))?;
        unsafe {
            encoder.setFragmentSamplerState_atIndex(Some(sampler), index);
        }
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
    state: &RenderEncodeState<'_>,
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
            let (buffer, index_type, index_offset) =
                metal_index_buffer(state.index_buffer, first_index)?;
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
            let buffer = metal_indirect_buffer(state.indirect_buffer)?;
            unsafe {
                encoder.drawPrimitives_indirectBuffer_indirectBufferOffset(
                    map_primitive_topology(topology),
                    buffer,
                    to_ns(offset)?,
                );
            }
        }
        HalDraw::IndexedIndirect { offset } => {
            let (index_buffer, index_type, index_offset) =
                metal_index_buffer(state.index_buffer, 0)?;
            let indirect_buffer = metal_indirect_buffer(state.indirect_buffer)?;
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
    bound: Option<&HalBoundIndexBuffer>,
    first_index: u32,
) -> Result<(&ProtocolObject<dyn MTLBufferTrait>, MTLIndexType, u64), HalError> {
    let bound = bound.ok_or_else(|| buffer_error("render index buffer is missing"))?;
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
    bound: Option<&HalBoundIndirectBuffer>,
) -> Result<&ProtocolObject<dyn MTLBufferTrait>, HalError> {
    let bound = bound.ok_or_else(|| buffer_error("render indirect buffer is missing"))?;
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

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;
    #[cfg(feature = "tiled")]
    use crate::{
        HalSubpassAttachmentLayout, HalSubpassColorAttachment, HalSubpassInputAttachment,
        HalSubpassLayout, HalSubpassPassLayout, HalSubpassRenderPassCommand,
    };

    /// Constructs a minimal `MetalBuffer` stub usable in unit tests.
    /// `inner` is `None` so GPU calls will fail, but `size()` works.
    fn make_metal_buffer(size: u64) -> MetalBuffer {
        MetalBuffer {
            inner: None,
            mapped_ptr: None,
            size,
        }
    }

    /// Constructs a `HalBoundBuffer` backed by a Metal buffer stub.
    fn make_vertex_bound_buffer(metal_index: u32, size: u64, offset: u64) -> HalBoundBuffer {
        HalBoundBuffer {
            group: 0,
            binding: metal_index,
            metal_index,
            vertex_metal_index: None,
            fragment_metal_index: None,
            buffer: HalBuffer::Metal(make_metal_buffer(size)),
            offset,
            size: u64::MAX,
        }
    }

    /// `compose_vertex_stage_sizes` places storage-array sizes first, then vertex
    /// buffer effective sizes (buffer.size - offset) in metal_index order.
    #[test]
    fn compose_vertex_stage_sizes_orders_storage_then_vertex_and_subtracts_offset() {
        // No storage-array bindings; two vertex buffers at metal slots 5 and 8.
        let storage_bindings: Vec<HalMslBufferSizeBinding> = Vec::new();
        let bind_buffers: Vec<HalBoundBuffer> = Vec::new();
        let vertex_buffer_metal_indices = vec![5u32, 8u32];
        // slot 5: size=256, offset=16 → effective=240
        // slot 8: size=1024, offset=0  → effective=1024
        let vertex_buffers = vec![
            make_vertex_bound_buffer(5, 256, 16),
            make_vertex_bound_buffer(8, 1024, 0),
        ];

        let sizes = compose_vertex_stage_sizes(
            &storage_bindings,
            &bind_buffers,
            &vertex_buffer_metal_indices,
            &vertex_buffers,
        )
        .expect("compose must succeed");

        // Expected: [240, 1024] (no storage entries).
        assert_eq!(sizes, vec![240u32, 1024u32]);
    }

    /// Missing vertex-buffer binding (no matching metal_index) contributes 0.
    #[test]
    fn compose_vertex_stage_sizes_missing_binding_contributes_zero() {
        let sizes = compose_vertex_stage_sizes(
            &[],
            &[],
            &[3u32], // no vertex buffer at slot 3
            &[],
        )
        .expect("compose must succeed");

        assert_eq!(sizes, vec![0u32]);
    }

    /// Vertex buffer sizes are appended AFTER any storage-array sizes.
    #[test]
    fn compose_vertex_stage_sizes_storage_entries_precede_vertex_entries() {
        // One storage-array binding with a Noop buffer (size=0 via msl_buffer_sizes fallback path).
        // The vertex buffer contributes 64.
        // Noop buffers return 0 from msl_buffer_sizes because bound_buffer_size rejects them.
        // Use a single vertex buffer slot only to verify ordering structure.
        let vertex_buffer_metal_indices = vec![2u32];
        let vertex_buffers = vec![make_vertex_bound_buffer(2, 64, 0)];
        // Storage binding references group=0,binding=99 which has no matching entry in bind_buffers
        // → msl_buffer_sizes returns 0 for it.
        let storage_bindings = vec![HalMslBufferSizeBinding::new(0, 99)];
        let bind_buffers: Vec<HalBoundBuffer> = Vec::new();

        let sizes = compose_vertex_stage_sizes(
            &storage_bindings,
            &bind_buffers,
            &vertex_buffer_metal_indices,
            &vertex_buffers,
        )
        .expect("compose must succeed");

        // Two entries: [storage_size(0), vertex_size(64)].
        assert_eq!(sizes.len(), 2);
        assert_eq!(sizes[0], 0u32); // storage entry (unbound → 0)
        assert_eq!(sizes[1], 64u32); // vertex entry
    }

    /// Effective size saturates at u32::MAX for a very large buffer.
    #[test]
    fn compose_vertex_stage_sizes_clamps_large_buffer_to_u32_max() {
        let vertex_buffer_metal_indices = vec![0u32];
        // Buffer larger than u32::MAX.
        let vertex_buffers = vec![make_vertex_bound_buffer(0, u64::from(u32::MAX) + 1, 0)];

        let sizes =
            compose_vertex_stage_sizes(&[], &[], &vertex_buffer_metal_indices, &vertex_buffers)
                .expect("compose must succeed");

        assert_eq!(sizes, vec![u32::MAX]);
    }

    #[test]
    fn msl_buffer_size_u32_rejects_overflow() {
        assert_eq!(
            msl_buffer_size_u32(u32::MAX as u64).expect("u32 max should fit"),
            u32::MAX
        );
        let error =
            msl_buffer_size_u32(u64::from(u32::MAX) + 1).expect_err("overflow must be rejected");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "metal",
                message: "MSL buffer size exceeds u32"
            }
        ));
    }

    #[test]
    fn metal_texture_swizzle_channels_maps_texture_component_swizzle() {
        let channels = metal_texture_swizzle_channels(&HalTextureComponentSwizzle {
            r: HalComponentSwizzle::Zero,
            g: HalComponentSwizzle::One,
            b: HalComponentSwizzle::B,
            a: HalComponentSwizzle::A,
        });

        assert_eq!(channels.red, MTLTextureSwizzle::Zero);
        assert_eq!(channels.green, MTLTextureSwizzle::One);
        assert_eq!(channels.blue, MTLTextureSwizzle::Blue);
        assert_eq!(channels.alpha, MTLTextureSwizzle::Alpha);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    fn metal_texture_view_uses_multisample_type_for_msaa_d2_source() {
        let device = metal_device();
        let mut descriptor = texture_descriptor();
        descriptor.sample_count = 4;
        let texture = device
            .create_texture(&descriptor)
            .expect("Metal texture allocation should succeed");
        let binding = HalBoundTexture {
            group: 0,
            binding: 0,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: None,
            texture: HalTexture::Metal(texture.clone()),
            format: descriptor.format,
            dimension: HalTextureViewDimension::D2,
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
            aspect: HalTextureAspect::All,
            swizzle: HalTextureComponentSwizzle::default(),
            storage_access: None,
        };

        let view = metal_texture_view(&texture, &binding)
            .expect("multisampled D2 texture view should allocate");
        assert_eq!(view.textureType(), MTLTextureType::Type2DMultisample);
    }

    #[cfg(feature = "tiled")]
    fn subpass_attachment(texture: HalTexture) -> HalSubpassColorAttachment {
        HalSubpassColorAttachment {
            resource: HalSubpassAttachmentResource::Persistent {
                texture,
                resolve_target: None,
            },
            load_op: HalRenderLoadOp::Clear,
            store: true,
            clear_color: [0.25, 0.5, 0.75, 1.0],
        }
    }

    #[cfg(feature = "tiled")]
    fn two_subpass_shared_color_command(
        color_attachments: Vec<HalSubpassColorAttachment>,
    ) -> HalSubpassRenderPassCommand {
        HalSubpassRenderPassCommand {
            layout: HalSubpassPassLayout {
                color_attachments: vec![HalSubpassAttachmentLayout {
                    format: HalTextureFormat::Rgba8Unorm,
                    sample_count: 1,
                }],
                depth_stencil_attachment: None,
                subpasses: vec![
                    HalSubpassLayout {
                        color_attachment_indices: vec![0],
                        uses_depth_stencil: false,
                        input_attachments: Vec::new(),
                    },
                    HalSubpassLayout {
                        color_attachment_indices: vec![0],
                        uses_depth_stencil: false,
                        input_attachments: vec![HalSubpassInputAttachment {
                            group: 0,
                            binding: 0,
                            source_subpass: 0,
                            source_attachment: 0,
                        }],
                    },
                ],
                dependencies: Vec::new(),
            },
            extent: HalExtent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            color_attachments,
            depth_stencil_attachment: None,
            draws: Vec::new(),
        }
    }

    #[test]
    #[cfg(feature = "tiled")]
    fn subpass_color_attachment_indices_dedupes_shared_color_slot() {
        let command = two_subpass_shared_color_command(Vec::new());

        assert_eq!(subpass_color_attachment_indices(&command), vec![0]);
    }

    #[test]
    #[cfg(all(feature = "noop", feature = "tiled"))]
    fn subpass_render_pass_descriptor_rejects_unsupported_attachment_resource() {
        let texture = crate::noop::NoopDevice::new()
            .create_texture(&texture_descriptor())
            .expect("Noop texture allocation should succeed");
        let command =
            two_subpass_shared_color_command(vec![subpass_attachment(HalTexture::Noop(texture))]);

        let error = subpass_render_pass_descriptor(&command)
            .expect_err("unsupported attachment resource should be rejected");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "metal",
                message: "subpass attachment is not Metal-backed"
            }
        ));
    }

    /// User-only block (no frag-depth clamp): the composed block is exactly
    /// `block_size` bytes copied from the front of `user_data`.
    #[test]
    fn compose_metal_immediates_block_copies_user_prefix_when_no_clamp() {
        let user_data: Vec<u8> = (0..64u8).collect();
        let immediates = HalMslImmediates::new(30, 16, None);

        let block = compose_metal_immediates_block(&user_data, immediates, None);

        assert_eq!(block, user_data[0..16]);
    }

    /// User + clamp: user bytes occupy `[0, frag_depth_clamp_offset)`, the
    /// clamp range (from the viewport) lands at
    /// `[frag_depth_clamp_offset, frag_depth_clamp_offset + 8)`, and no user
    /// bytes beyond the clamp offset leak into the block (Dawn
    /// `RenderImmediates` layout: user prefix, then `ClampFragDepthArgs`).
    #[test]
    fn compose_metal_immediates_block_appends_clamp_range_after_user_prefix() {
        let user_data: Vec<u8> = (0..64u8).collect();
        let immediates = HalMslImmediates::new(30, 24, Some(16));
        let viewport = HalViewport {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            min_depth: 0.25,
            max_depth: 0.75,
        };

        let block = compose_metal_immediates_block(&user_data, immediates, Some(viewport));

        assert_eq!(block.len(), 24);
        assert_eq!(&block[0..16], &user_data[0..16]);
        assert_eq!(&block[16..20], &0.25f32.to_ne_bytes());
        assert_eq!(&block[20..24], &0.75f32.to_ne_bytes());
    }

    /// A clamp-only pipeline (no user immediates reserved,
    /// `frag_depth_clamp_offset == Some(0)`) composes exactly the bare
    /// 8-byte clamp range -- the pre-Block-94 behaviour
    /// `encode_render_frag_depth_clamp` used to deliver on its own.
    #[test]
    fn compose_metal_immediates_block_clamp_only_pipeline_yields_bare_range() {
        let immediates = HalMslImmediates::new(30, 8, Some(0));
        let viewport = HalViewport {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            min_depth: 0.1,
            max_depth: 0.9,
        };

        let block = compose_metal_immediates_block(&[], immediates, Some(viewport));

        assert_eq!(block.len(), 8);
        assert_eq!(&block[0..4], &0.1f32.to_ne_bytes());
        assert_eq!(&block[4..8], &0.9f32.to_ne_bytes());
    }

    /// No viewport override at draw time falls back to the full depth range
    /// `[0.0, 1.0]`, matching the pre-S2 `encode_render_frag_depth_clamp`
    /// default.
    #[test]
    fn compose_metal_immediates_block_defaults_clamp_range_without_viewport() {
        let immediates = HalMslImmediates::new(30, 8, Some(0));

        let block = compose_metal_immediates_block(&[], immediates, None);

        assert_eq!(&block[0..4], &0.0f32.to_ne_bytes());
        assert_eq!(&block[4..8], &1.0f32.to_ne_bytes());
    }
}
