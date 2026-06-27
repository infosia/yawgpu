use super::*;

/// Converts a bind group layout descriptor to the core representation.
///
/// # Safety
///
/// `descriptor.entries`, when non-null and `entryCount > 0`, must point to
/// `entryCount` valid `WGPUBindGroupLayoutEntry` values.
/// Converts bind group layout descriptor into the corresponding yawgpu representation.
#[must_use]
pub unsafe fn map_bind_group_layout_descriptor(
    descriptor: &native::WGPUBindGroupLayoutDescriptor,
) -> core::BindGroupLayoutDescriptor {
    if descriptor.entryCount > 0 && descriptor.entries.is_null() {
        return core::BindGroupLayoutDescriptor {
            entries: Vec::new(),
            error: Some("bind group layout entries must not be null".to_owned()),
        };
    }

    let mut error = None;
    let entries = if descriptor.entryCount == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(descriptor.entries, descriptor.entryCount)
            .iter()
            .map(|entry| map_bind_group_layout_entry(entry, &mut error))
            .collect()
    };

    core::BindGroupLayoutDescriptor { entries, error }
}

/// Converts bind group entries to the core representation.
///
/// # Safety
///
/// `descriptor.entries`, when non-null and `entryCount > 0`, must point to
/// `entryCount` valid `WGPUBindGroupEntry` values. Any non-null resource
/// handles must be live yawgpu handles of the matching type.
/// Converts bind group entries into the corresponding yawgpu representation.
#[must_use]
pub unsafe fn map_bind_group_entries(
    descriptor: &native::WGPUBindGroupDescriptor,
) -> Vec<core::BindGroupEntry> {
    if descriptor.entryCount > 0 && descriptor.entries.is_null() {
        return vec![core::BindGroupEntry {
            binding: 0,
            resource: core::BindGroupResource::Invalid(
                "bind group entries must not be null".to_owned(),
            ),
        }];
    }

    if descriptor.entryCount == 0 {
        return Vec::new();
    }

    std::slice::from_raw_parts(descriptor.entries, descriptor.entryCount)
        .iter()
        .map(|entry| map_bind_group_entry(entry))
        .collect()
}

unsafe fn map_bind_group_entry(entry: &native::WGPUBindGroupEntry) -> core::BindGroupEntry {
    let mut present_count = 0;
    let mut resource = core::BindGroupResource::Invalid(
        "bind group entry must set exactly one resource".to_owned(),
    );

    if !entry.buffer.is_null() {
        present_count += 1;
        let buffer = clone_handle::<WGPUBufferImpl>(entry.buffer, "WGPUBuffer");
        resource = core::BindGroupResource::Buffer {
            buffer: Arc::clone(&buffer.core),
            device: Arc::clone(&buffer.device),
            offset: entry.offset,
            size: entry.size,
        };
    }
    if !entry.sampler.is_null() {
        present_count += 1;
        let sampler = clone_handle::<WGPUSamplerImpl>(entry.sampler, "WGPUSampler");
        resource = core::BindGroupResource::Sampler {
            sampler: Arc::clone(&sampler._core),
            device: Arc::clone(&sampler._device),
        };
    }
    if !entry.textureView.is_null() {
        present_count += 1;
        let texture_view =
            clone_handle::<WGPUTextureViewImpl>(entry.textureView, "WGPUTextureView");
        resource = core::BindGroupResource::TextureView {
            texture_view: Arc::clone(&texture_view._core),
            device: Arc::clone(&texture_view._device),
        };
    }
    if let Some(external_texture_entry) = external_texture_binding_entry(entry.nextInChain) {
        present_count += 1;
        if external_texture_entry.externalTexture.is_null() {
            resource = core::BindGroupResource::Invalid(
                "external texture bind group entry must not be null".to_owned(),
            );
        } else {
            let external_texture = clone_handle::<WGPUExternalTextureImpl>(
                external_texture_entry.externalTexture,
                "WGPUExternalTexture",
            );
            resource = core::BindGroupResource::ExternalTexture {
                external_texture: Arc::clone(&external_texture._core),
                device: Arc::clone(&external_texture._device),
            };
        }
    }

    if present_count != 1 {
        resource = core::BindGroupResource::Invalid(
            "bind group entry must set exactly one resource".to_owned(),
        );
    }

    core::BindGroupEntry {
        binding: entry.binding,
        resource,
    }
}

/// Converts a pipeline layout descriptor to the core representation.
///
/// # Safety
///
/// `descriptor.bindGroupLayouts`, when non-null and
/// `bindGroupLayoutCount > 0`, must point to `bindGroupLayoutCount`
/// `WGPUBindGroupLayout` slots. Non-null slots must be live yawgpu handles.
/// Converts pipeline layout descriptor into the corresponding yawgpu representation.
#[must_use]
pub unsafe fn map_pipeline_layout_descriptor(
    descriptor: &native::WGPUPipelineLayoutDescriptor,
) -> core::PipelineLayoutDescriptor {
    let mut error = None;
    let bind_group_layouts = if descriptor.bindGroupLayoutCount == 0 {
        Vec::new()
    } else if descriptor.bindGroupLayouts.is_null() {
        set_first_error(
            &mut error,
            "pipeline layout bindGroupLayouts must not be null when count is non-zero",
        );
        Vec::new()
    } else {
        std::slice::from_raw_parts(descriptor.bindGroupLayouts, descriptor.bindGroupLayoutCount)
            .iter()
            .map(|layout| {
                if layout.is_null() {
                    Arc::new(core::BindGroupLayout::empty_unused())
                } else {
                    let layout =
                        clone_handle::<WGPUBindGroupLayoutImpl>(*layout, "WGPUBindGroupLayout");
                    Arc::clone(&layout._core)
                }
            })
            .collect()
    };

    core::PipelineLayoutDescriptor {
        bind_group_layouts,
        immediate_size: descriptor.immediateSize,
        error,
    }
}

fn map_bind_group_layout_entry(
    entry: &native::WGPUBindGroupLayoutEntry,
    error: &mut Option<String>,
) -> core::BindGroupLayoutEntry {
    let mut present_count = 0;
    let mut kind = None;

    if unsafe { external_texture_binding_layout(entry.nextInChain) }.is_some() {
        present_count += 1;
        if !standard_binding_layout_fields_empty(entry) {
            set_first_error(
                error,
                "external texture binding layout must not set standard binding layout fields",
            );
        }
        kind = Some(core::BindingLayoutKind::ExternalTexture);
    }

    if entry.buffer.type_ != native::WGPUBufferBindingType_BindingNotUsed {
        present_count += 1;
        kind = map_buffer_binding_layout(entry.buffer, error);
    }
    if entry.sampler.type_ != native::WGPUSamplerBindingType_BindingNotUsed {
        present_count += 1;
        kind = map_sampler_binding_layout(entry.sampler, error);
    }
    if entry.texture.sampleType != native::WGPUTextureSampleType_BindingNotUsed {
        present_count += 1;
        kind = map_texture_binding_layout(entry.texture, error);
    }
    if entry.storageTexture.access != native::WGPUStorageTextureAccess_BindingNotUsed {
        present_count += 1;
        kind = map_storage_texture_binding_layout(entry.storageTexture, error);
    }

    if present_count != 1 && error.is_none() {
        *error = Some("bind group layout entry must set exactly one binding layout".to_owned());
        kind = None;
    }

    core::BindGroupLayoutEntry {
        binding: entry.binding,
        visibility: entry.visibility,
        binding_array_size: entry.bindingArraySize,
        kind,
    }
}
unsafe fn external_texture_binding_layout<'a>(
    mut chain: *const native::WGPUChainedStruct,
) -> Option<&'a native::WGPUExternalTextureBindingLayout> {
    while let Some(node) = chain.as_ref() {
        if node.sType == native::WGPUSType_ExternalTextureBindingLayout {
            return Some(
                &*(node as *const native::WGPUChainedStruct
                    as *const native::WGPUExternalTextureBindingLayout),
            );
        }
        chain = node.next;
    }
    None
}

unsafe fn external_texture_binding_entry<'a>(
    mut chain: *const native::WGPUChainedStruct,
) -> Option<&'a native::WGPUExternalTextureBindingEntry> {
    while let Some(node) = chain.as_ref() {
        if node.sType == native::WGPUSType_ExternalTextureBindingEntry {
            return Some(
                &*(node as *const native::WGPUChainedStruct
                    as *const native::WGPUExternalTextureBindingEntry),
            );
        }
        chain = node.next;
    }
    None
}

fn standard_binding_layout_fields_empty(entry: &native::WGPUBindGroupLayoutEntry) -> bool {
    entry.buffer.type_ == native::WGPUBufferBindingType_BindingNotUsed
        && entry.sampler.type_ == native::WGPUSamplerBindingType_BindingNotUsed
        && entry.texture.sampleType == native::WGPUTextureSampleType_BindingNotUsed
        && entry.storageTexture.access == native::WGPUStorageTextureAccess_BindingNotUsed
}
fn map_buffer_binding_layout(
    layout: native::WGPUBufferBindingLayout,
    error: &mut Option<String>,
) -> Option<core::BindingLayoutKind> {
    let ty = match layout.type_ {
        native::WGPUBufferBindingType_Undefined | native::WGPUBufferBindingType_Uniform => {
            core::BufferBindingType::Uniform
        }
        native::WGPUBufferBindingType_Storage => core::BufferBindingType::Storage,
        native::WGPUBufferBindingType_ReadOnlyStorage => core::BufferBindingType::ReadOnlyStorage,
        _ => {
            set_first_error(error, "invalid buffer binding type");
            return None;
        }
    };
    Some(core::BindingLayoutKind::Buffer {
        ty,
        has_dynamic_offset: layout.hasDynamicOffset != 0,
        min_binding_size: layout.minBindingSize,
    })
}

fn map_sampler_binding_layout(
    layout: native::WGPUSamplerBindingLayout,
    error: &mut Option<String>,
) -> Option<core::BindingLayoutKind> {
    let ty = match layout.type_ {
        native::WGPUSamplerBindingType_Undefined | native::WGPUSamplerBindingType_Filtering => {
            core::SamplerBindingType::Filtering
        }
        native::WGPUSamplerBindingType_NonFiltering => core::SamplerBindingType::NonFiltering,
        native::WGPUSamplerBindingType_Comparison => core::SamplerBindingType::Comparison,
        _ => {
            set_first_error(error, "invalid sampler binding type");
            return None;
        }
    };
    Some(core::BindingLayoutKind::Sampler { ty })
}

fn map_texture_binding_layout(
    layout: native::WGPUTextureBindingLayout,
    error: &mut Option<String>,
) -> Option<core::BindingLayoutKind> {
    let sample_type = match layout.sampleType {
        native::WGPUTextureSampleType_Undefined | native::WGPUTextureSampleType_Float => {
            core::TextureSampleType::Float
        }
        native::WGPUTextureSampleType_UnfilterableFloat => {
            core::TextureSampleType::UnfilterableFloat
        }
        native::WGPUTextureSampleType_Depth => core::TextureSampleType::Depth,
        native::WGPUTextureSampleType_Sint => core::TextureSampleType::Sint,
        native::WGPUTextureSampleType_Uint => core::TextureSampleType::Uint,
        _ => {
            set_first_error(error, "invalid texture sample type");
            return None;
        }
    };
    Some(core::BindingLayoutKind::Texture {
        sample_type,
        view_dimension: map_bgl_texture_view_dimension(layout.viewDimension, error)?,
        multisampled: layout.multisampled != 0,
    })
}

fn map_storage_texture_binding_layout(
    layout: native::WGPUStorageTextureBindingLayout,
    error: &mut Option<String>,
) -> Option<core::BindingLayoutKind> {
    let access = match layout.access {
        native::WGPUStorageTextureAccess_Undefined | native::WGPUStorageTextureAccess_WriteOnly => {
            core::StorageTextureAccess::WriteOnly
        }
        native::WGPUStorageTextureAccess_ReadOnly => core::StorageTextureAccess::ReadOnly,
        native::WGPUStorageTextureAccess_ReadWrite => core::StorageTextureAccess::ReadWrite,
        _ => {
            set_first_error(error, "invalid storage texture access");
            return None;
        }
    };
    Some(core::BindingLayoutKind::StorageTexture {
        access,
        format: map_texture_format(layout.format),
        view_dimension: map_bgl_texture_view_dimension(layout.viewDimension, error)?,
    })
}

fn map_bgl_texture_view_dimension(
    value: native::WGPUTextureViewDimension,
    error: &mut Option<String>,
) -> Option<core::TextureViewDimension> {
    match value {
        native::WGPUTextureViewDimension_Undefined => Some(core::TextureViewDimension::D2),
        native::WGPUTextureViewDimension_1D => Some(core::TextureViewDimension::D1),
        native::WGPUTextureViewDimension_2D => Some(core::TextureViewDimension::D2),
        native::WGPUTextureViewDimension_2DArray => Some(core::TextureViewDimension::D2Array),
        native::WGPUTextureViewDimension_Cube => Some(core::TextureViewDimension::Cube),
        native::WGPUTextureViewDimension_CubeArray => Some(core::TextureViewDimension::CubeArray),
        native::WGPUTextureViewDimension_3D => Some(core::TextureViewDimension::D3),
        _ => {
            set_first_error(error, "invalid texture view dimension");
            None
        }
    }
}

// `as u32` / `as native::WGPUFeatureName` are required on Windows MSVC where
// `native::WGPUFeatureName` resolves to `c_int = i32`. On macOS clang it is
// `c_uint = u32`, so the cast becomes a no-op that clippy flags as
// `unnecessary_cast`; the lint is silenced here because the cast is the
// cross-platform-correct expression.
