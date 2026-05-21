use super::*;

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
pub fn map_buffer_descriptor(value: &native::WGPUBufferDescriptor) -> core::BufferDescriptor {
    core::BufferDescriptor {
        usage: map_buffer_usage(value.usage),
        size: value.size,
        mapped_at_creation: value.mappedAtCreation != 0,
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
