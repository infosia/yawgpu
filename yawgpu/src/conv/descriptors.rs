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

/// Converts buffer descriptor into the corresponding yawgpu representation.
#[must_use]
pub fn map_buffer_descriptor(value: &native::WGPUBufferDescriptor) -> core::BufferDescriptor {
    core::BufferDescriptor {
        usage: map_buffer_usage(value.usage),
        size: value.size,
        mapped_at_creation: value.mappedAtCreation != 0,
    }
}

/// Converts sampler descriptor into the corresponding yawgpu representation.
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

/// Converts extent 3d into the corresponding yawgpu representation.
#[must_use]
pub fn map_extent_3d(value: native::WGPUExtent3D) -> core::Extent3d {
    core::Extent3d {
        width: value.width,
        height: value.height,
        depth_or_array_layers: value.depthOrArrayLayers,
    }
}

/// Converts origin 3d into the corresponding yawgpu representation.
#[must_use]
pub fn map_origin_3d(value: native::WGPUOrigin3D) -> core::Origin3d {
    core::Origin3d {
        x: value.x,
        y: value.y,
        z: value.z,
    }
}

/// Converts texel copy buffer layout into the corresponding yawgpu representation.
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

/// Converts texel copy texture info parts into the corresponding yawgpu representation.
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
    let max_draw_count = render_pass_max_draw_count(value.nextInChain);

    core::RenderPassDescriptor {
        max_color_attachments,
        color_attachments,
        depth_stencil_attachment,
        occlusion_query_set,
        timestamp_writes,
        max_draw_count,
    }
}

unsafe fn render_pass_max_draw_count(mut chain: *const native::WGPUChainedStruct) -> u64 {
    const DEFAULT_MAX_DRAW_COUNT: u64 = 50_000_000;
    while let Some(node) = unsafe { chain.as_ref() } {
        if node.sType == native::WGPUSType_RenderPassMaxDrawCount {
            let max_draw_count = unsafe {
                &*(node as *const native::WGPUChainedStruct
                    as *const native::WGPURenderPassMaxDrawCount)
            };
            return max_draw_count.maxDrawCount;
        }
        chain = node.next;
    }
    DEFAULT_MAX_DRAW_COUNT
}

unsafe fn texture_component_swizzle(
    mut chain: *const native::WGPUChainedStruct,
) -> Option<core::TextureComponentSwizzle> {
    while let Some(node) = unsafe { chain.as_ref() } {
        if node.sType == native::WGPUSType_TextureComponentSwizzleDescriptor {
            let descriptor = unsafe {
                &*(node as *const native::WGPUChainedStruct
                    as *const native::WGPUTextureComponentSwizzleDescriptor)
            };
            return Some(map_texture_component_swizzle(descriptor.swizzle));
        }
        chain = node.next;
    }
    None
}

fn map_component_swizzle(
    value: native::WGPUComponentSwizzle,
    default: core::ComponentSwizzle,
) -> core::ComponentSwizzle {
    match value {
        native::WGPUComponentSwizzle_Undefined => default,
        native::WGPUComponentSwizzle_Zero => core::ComponentSwizzle::Zero,
        native::WGPUComponentSwizzle_One => core::ComponentSwizzle::One,
        native::WGPUComponentSwizzle_R => core::ComponentSwizzle::R,
        native::WGPUComponentSwizzle_G => core::ComponentSwizzle::G,
        native::WGPUComponentSwizzle_B => core::ComponentSwizzle::B,
        native::WGPUComponentSwizzle_A => core::ComponentSwizzle::A,
        _ => core::ComponentSwizzle::Zero,
    }
}

fn map_texture_component_swizzle(
    value: native::WGPUTextureComponentSwizzle,
) -> core::TextureComponentSwizzle {
    core::TextureComponentSwizzle {
        r: map_component_swizzle(value.r, core::ComponentSwizzle::R),
        g: map_component_swizzle(value.g, core::ComponentSwizzle::G),
        b: map_component_swizzle(value.b, core::ComponentSwizzle::B),
        a: map_component_swizzle(value.a, core::ComponentSwizzle::A),
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
        depth_slice: (value.depthSlice != native::WGPU_DEPTH_SLICE_UNDEFINED)
            .then_some(value.depthSlice),
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
        depth_read_only: value.depthReadOnly != 0,
        stencil_load_op: map_load_op(value.stencilLoadOp),
        stencil_store_op: map_store_op(value.stencilStoreOp),
        stencil_clear_value: value.stencilClearValue,
        stencil_read_only: value.stencilReadOnly != 0,
    }
}

pub(crate) unsafe fn map_render_pass_timestamp_writes(
    value: &native::WGPUPassTimestampWrites,
) -> core::RenderPassTimestampWrites {
    let query_set = clone_handle::<WGPUQuerySetImpl>(value.querySet, "WGPUQuerySet");
    core::RenderPassTimestampWrites {
        query_set: (*query_set.core).clone(),
        beginning_index: map_query_index(value.beginningOfPassWriteIndex),
        end_index: map_query_index(value.endOfPassWriteIndex),
    }
}

/// Converts color into the corresponding yawgpu representation.
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

/// Converts a transient attachment descriptor to the core representation.
#[cfg(feature = "tiled")]
#[must_use]
pub fn map_transient_attachment_descriptor(
    value: &crate::YaWGPUTransientAttachmentDescriptor,
) -> core::TransientAttachmentDescriptor {
    core::TransientAttachmentDescriptor {
        format: map_texture_format(value.format),
        size: match value.sizeMode {
            crate::YaWGPUTransientSizeMode_Explicit => core::TransientSizeMode::Explicit {
                width: value.width,
                height: value.height,
            },
            _ => core::TransientSizeMode::MatchTarget,
        },
        sample_count: value.sampleCount,
    }
}

/// Converts a subpass pass layout descriptor to the core representation.
///
/// # Safety
///
/// Any non-null pointer/count pairs in `descriptor` must point to live arrays
/// of at least the declared count for the duration of this call.
#[cfg(feature = "tiled")]
#[must_use]
pub unsafe fn map_subpass_pass_layout_descriptor(
    descriptor: &crate::YaWGPUSubpassPassLayoutDescriptor,
) -> core::SubpassPassLayoutDescriptor {
    let mut error = None;
    let color_attachments = slice_or_error(
        descriptor.colorAttachments,
        descriptor.colorAttachmentCount,
        "subpass pass layout colorAttachments must not be null when count is non-zero",
        &mut error,
    )
    .iter()
    .map(|attachment| core::AttachmentLayout {
        format: map_texture_format(attachment.format),
        sample_count: attachment.sampleCount,
    })
    .collect();
    let depth_stencil_attachment =
        descriptor
            .depthStencilAttachment
            .as_ref()
            .map(|layout| core::AttachmentLayout {
                format: map_texture_format(layout.format),
                sample_count: layout.sampleCount,
            });
    let subpasses = slice_or_error(
        descriptor.subpasses,
        descriptor.subpassCount,
        "subpass pass layout subpasses must not be null when count is non-zero",
        &mut error,
    )
    .iter()
    .map(|subpass| core::SubpassLayoutDesc {
        color_attachment_indices: slice_or_error(
            subpass.colorAttachmentIndices,
            subpass.colorAttachmentIndexCount,
            "subpass colorAttachmentIndices must not be null when count is non-zero",
            &mut error,
        )
        .to_vec(),
        uses_depth_stencil: subpass.usesDepthStencil != 0,
        input_attachments: slice_or_error(
            subpass.inputAttachments,
            subpass.inputAttachmentCount,
            "subpass inputAttachments must not be null when count is non-zero",
            &mut error,
        )
        .iter()
        .map(|input| core::SubpassInputAttachment {
            group: input.group,
            binding: input.binding,
            source_subpass: input.sourceSubpass,
            source_attachment: input.sourceAttachment,
        })
        .collect(),
    })
    .collect();
    let dependencies = slice_or_error(
        descriptor.dependencies,
        descriptor.dependencyCount,
        "subpass pass layout dependencies must not be null when count is non-zero",
        &mut error,
    )
    .iter()
    .map(|dependency| core::SubpassDependency {
        src_subpass: dependency.srcSubpass,
        dst_subpass: dependency.dstSubpass,
        dependency_type: match dependency.dependencyType {
            crate::YaWGPUSubpassDependencyType_DepthToInput => {
                core::SubpassDependencyType::DepthToInput
            }
            crate::YaWGPUSubpassDependencyType_ColorDepthToInput => {
                core::SubpassDependencyType::ColorDepthToInput
            }
            _ => core::SubpassDependencyType::ColorToInput,
        },
        by_region: dependency.byRegion != 0,
    })
    .collect();
    core::SubpassPassLayoutDescriptor {
        color_attachments,
        depth_stencil_attachment,
        subpasses,
        dependencies,
        error,
    }
}

/// Converts a subpass render pass descriptor to the core representation.
///
/// # Safety
///
/// `descriptor.passLayout` and every non-null resource handle must be live C
/// handles returned by yawgpu. Any non-null pointer/count pairs must point to
/// live arrays of at least the declared count for the duration of this call.
#[cfg(feature = "tiled")]
#[must_use]
pub unsafe fn map_subpass_render_pass_descriptor(
    descriptor: &crate::YaWGPUSubpassRenderPassDescriptor,
) -> core::SubpassRenderPassDescriptor {
    let mut error = None;
    let pass_layout = clone_handle::<crate::YaWGPUSubpassPassLayoutImpl>(
        descriptor.passLayout,
        "YaWGPUSubpassPassLayout",
    );
    let color_attachments = slice_or_error(
        descriptor.colorAttachments,
        descriptor.colorAttachmentCount,
        "subpass render pass colorAttachments must not be null when count is non-zero",
        &mut error,
    )
    .iter()
    .map(|attachment| map_color_attachment_binding(attachment, &mut error))
    .collect();
    let depth_stencil_attachment = descriptor
        .depthStencilAttachment
        .as_ref()
        .map(|attachment| map_depth_stencil_attachment_binding(attachment, &mut error));
    core::SubpassRenderPassDescriptor {
        pass_layout: Arc::clone(&pass_layout._core),
        extent: map_extent_3d(descriptor.extent),
        color_attachments,
        depth_stencil_attachment,
        error,
    }
}

#[cfg(feature = "tiled")]
fn map_color_attachment_binding(
    attachment: &crate::YaWGPUColorAttachmentBinding,
    error: &mut Option<String>,
) -> core::SubpassColorAttachmentBinding {
    core::SubpassColorAttachmentBinding {
        resource: map_subpass_attachment_resource(
            attachment.kind,
            attachment.view,
            attachment.resolveTarget,
            attachment.transient,
            error,
        ),
        load_op: map_load_op(attachment.loadOp),
        store_op: map_store_op(attachment.storeOp),
        clear_value: map_color(attachment.clearValue),
    }
}

#[cfg(feature = "tiled")]
fn map_depth_stencil_attachment_binding(
    attachment: &crate::YaWGPUDepthStencilAttachmentBinding,
    error: &mut Option<String>,
) -> core::SubpassDepthStencilAttachmentBinding {
    core::SubpassDepthStencilAttachmentBinding {
        resource: map_subpass_attachment_resource(
            attachment.kind,
            attachment.view,
            std::ptr::null(),
            attachment.transient,
            error,
        ),
        depth_load_op: map_load_op(attachment.depthLoadOp),
        depth_store_op: map_store_op(attachment.depthStoreOp),
        depth_clear_value: attachment.depthClearValue,
        stencil_load_op: map_load_op(attachment.stencilLoadOp),
        stencil_store_op: map_store_op(attachment.stencilStoreOp),
        stencil_clear_value: attachment.stencilClearValue,
    }
}

#[cfg(feature = "tiled")]
fn map_subpass_attachment_resource(
    kind: crate::YaWGPUSubpassAttachmentKind,
    view: native::WGPUTextureView,
    resolve_target: native::WGPUTextureView,
    transient: crate::YaWGPUTransientAttachment,
    error: &mut Option<String>,
) -> core::SubpassAttachmentResource {
    if kind == crate::YaWGPUSubpassAttachmentKind_Transient {
        if transient.is_null() || !view.is_null() || !resolve_target.is_null() {
            set_first_error(
                error,
                "transient subpass attachment must set only transient",
            );
        }
        let transient = unsafe {
            clone_handle::<crate::YaWGPUTransientAttachmentImpl>(
                transient,
                "YaWGPUTransientAttachment",
            )
        };
        return core::SubpassAttachmentResource::Transient(Arc::clone(&transient._core));
    }
    if view.is_null() || !transient.is_null() {
        set_first_error(
            error,
            "persistent subpass attachment must set only view resources",
        );
    }
    let view = unsafe { clone_handle::<crate::WGPUTextureViewImpl>(view, "WGPUTextureView") };
    let resolve_target = (!resolve_target.is_null()).then(|| unsafe {
        clone_handle::<crate::WGPUTextureViewImpl>(resolve_target, "WGPUTextureView")
    });
    core::SubpassAttachmentResource::Persistent {
        view: Arc::clone(&view._core),
        resolve_target: resolve_target.map(|view| Arc::clone(&view._core)),
    }
}

#[cfg(feature = "tiled")]
unsafe fn slice_or_error<'a, T>(
    ptr: *const T,
    count: usize,
    message: &str,
    error: &mut Option<String>,
) -> &'a [T] {
    if count == 0 {
        &[]
    } else if ptr.is_null() {
        set_first_error(error, message);
        &[]
    } else {
        std::slice::from_raw_parts(ptr, count)
    }
}

/// Converts texture view descriptor into the corresponding yawgpu representation.
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
            usage: None,
            swizzle: None,
        };
    };
    let swizzle = unsafe { texture_component_swizzle(value.nextInChain) };
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
        usage: if value.usage == native::WGPUTextureUsage_None {
            None
        } else {
            Some(map_texture_usage(value.usage))
        },
        swizzle,
    }
}

#[cfg(all(test, feature = "tiled"))]
mod tests {
    use super::*;

    fn empty_string_view() -> native::WGPUStringView {
        native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        }
    }

    fn minimal_descriptor(
        color: &crate::YaWGPUAttachmentLayout,
        subpass: &crate::YaWGPUSubpassLayoutDesc,
        depth_stencil: *const crate::YaWGPUAttachmentLayout,
    ) -> crate::YaWGPUSubpassPassLayoutDescriptor {
        crate::YaWGPUSubpassPassLayoutDescriptor {
            nextInChain: std::ptr::null(),
            label: empty_string_view(),
            colorAttachments: color,
            colorAttachmentCount: 1,
            depthStencilAttachment: depth_stencil,
            subpasses: subpass,
            subpassCount: 1,
            dependencies: std::ptr::null(),
            dependencyCount: 0,
        }
    }

    #[test]
    fn subpass_pass_layout_descriptor_depth_stencil_null_maps_to_none() {
        let color = crate::YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sampleCount: 1,
        };
        let color_index: u32 = 0;
        let subpass = crate::YaWGPUSubpassLayoutDesc {
            colorAttachmentIndices: &color_index,
            colorAttachmentIndexCount: 1,
            usesDepthStencil: 0,
            inputAttachments: std::ptr::null(),
            inputAttachmentCount: 0,
        };
        let descriptor = minimal_descriptor(&color, &subpass, std::ptr::null());

        let mapped = unsafe { map_subpass_pass_layout_descriptor(&descriptor) };

        assert!(mapped.depth_stencil_attachment.is_none());
        assert!(mapped.error.is_none());
        assert_eq!(mapped.color_attachments.len(), 1);
        assert_eq!(mapped.color_attachments[0].sample_count, 1);
    }

    #[test]
    fn subpass_pass_layout_descriptor_depth_stencil_non_null_maps_to_some() {
        let color = crate::YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sampleCount: 4,
        };
        let color_index: u32 = 0;
        let subpass = crate::YaWGPUSubpassLayoutDesc {
            colorAttachmentIndices: &color_index,
            colorAttachmentIndexCount: 1,
            usesDepthStencil: 1,
            inputAttachments: std::ptr::null(),
            inputAttachmentCount: 0,
        };
        let depth_stencil = crate::YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_Depth32Float,
            sampleCount: 4,
        };
        let descriptor = minimal_descriptor(&color, &subpass, &depth_stencil);

        let mapped = unsafe { map_subpass_pass_layout_descriptor(&descriptor) };

        let ds = mapped
            .depth_stencil_attachment
            .as_ref()
            .expect("non-null depth-stencil pointer should map to Some");
        assert_eq!(ds.format, native::WGPUTextureFormat_Depth32Float.into());
        assert_eq!(ds.sample_count, 4);
        assert!(mapped.error.is_none());
    }
}
