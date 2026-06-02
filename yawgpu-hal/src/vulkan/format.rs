use super::*;

/// Converts texture format into the corresponding yawgpu representation.
pub(super) fn map_texture_format(format: HalTextureFormat) -> Result<(vk::Format, u32), HalError> {
    match format {
        HalTextureFormat::R8Unorm => Ok((vk::Format::R8_UNORM, 1)),
        HalTextureFormat::R8Snorm => Ok((vk::Format::R8_SNORM, 1)),
        HalTextureFormat::R8Uint => Ok((vk::Format::R8_UINT, 1)),
        HalTextureFormat::R8Sint => Ok((vk::Format::R8_SINT, 1)),
        HalTextureFormat::R16Unorm => Ok((vk::Format::R16_UNORM, 2)),
        HalTextureFormat::R16Snorm => Ok((vk::Format::R16_SNORM, 2)),
        HalTextureFormat::R16Uint => Ok((vk::Format::R16_UINT, 2)),
        HalTextureFormat::R16Sint => Ok((vk::Format::R16_SINT, 2)),
        HalTextureFormat::R16Float => Ok((vk::Format::R16_SFLOAT, 2)),
        HalTextureFormat::Rg8Unorm => Ok((vk::Format::R8G8_UNORM, 2)),
        HalTextureFormat::Rg8Snorm => Ok((vk::Format::R8G8_SNORM, 2)),
        HalTextureFormat::Rg8Uint => Ok((vk::Format::R8G8_UINT, 2)),
        HalTextureFormat::Rg8Sint => Ok((vk::Format::R8G8_SINT, 2)),
        HalTextureFormat::Rg16Unorm => Ok((vk::Format::R16G16_UNORM, 4)),
        HalTextureFormat::Rg16Snorm => Ok((vk::Format::R16G16_SNORM, 4)),
        HalTextureFormat::Rg16Uint => Ok((vk::Format::R16G16_UINT, 4)),
        HalTextureFormat::Rg16Sint => Ok((vk::Format::R16G16_SINT, 4)),
        HalTextureFormat::Rg16Float => Ok((vk::Format::R16G16_SFLOAT, 4)),
        HalTextureFormat::R32Uint => Ok((vk::Format::R32_UINT, 4)),
        HalTextureFormat::R32Sint => Ok((vk::Format::R32_SINT, 4)),
        HalTextureFormat::R32Float => Ok((vk::Format::R32_SFLOAT, 4)),
        HalTextureFormat::Rg32Uint => Ok((vk::Format::R32G32_UINT, 8)),
        HalTextureFormat::Rg32Sint => Ok((vk::Format::R32G32_SINT, 8)),
        HalTextureFormat::Rg32Float => Ok((vk::Format::R32G32_SFLOAT, 8)),
        HalTextureFormat::Rgba8Unorm => Ok((vk::Format::R8G8B8A8_UNORM, 4)),
        HalTextureFormat::Rgba8UnormSrgb => Ok((vk::Format::R8G8B8A8_SRGB, 4)),
        HalTextureFormat::Rgba8Snorm => Ok((vk::Format::R8G8B8A8_SNORM, 4)),
        HalTextureFormat::Rgba8Uint => Ok((vk::Format::R8G8B8A8_UINT, 4)),
        HalTextureFormat::Rgba8Sint => Ok((vk::Format::R8G8B8A8_SINT, 4)),
        HalTextureFormat::Bgra8Unorm => Ok((vk::Format::B8G8R8A8_UNORM, 4)),
        HalTextureFormat::Bgra8UnormSrgb => Ok((vk::Format::B8G8R8A8_SRGB, 4)),
        HalTextureFormat::Rgb10a2Uint => Ok((vk::Format::A2B10G10R10_UINT_PACK32, 4)),
        HalTextureFormat::Rgb10a2Unorm => Ok((vk::Format::A2B10G10R10_UNORM_PACK32, 4)),
        HalTextureFormat::Rg11b10Ufloat => Ok((vk::Format::B10G11R11_UFLOAT_PACK32, 4)),
        HalTextureFormat::Rgb9e5Ufloat => Ok((vk::Format::E5B9G9R9_UFLOAT_PACK32, 4)),
        HalTextureFormat::Rgba16Unorm => Ok((vk::Format::R16G16B16A16_UNORM, 8)),
        HalTextureFormat::Rgba16Snorm => Ok((vk::Format::R16G16B16A16_SNORM, 8)),
        HalTextureFormat::Rgba16Uint => Ok((vk::Format::R16G16B16A16_UINT, 8)),
        HalTextureFormat::Rgba16Sint => Ok((vk::Format::R16G16B16A16_SINT, 8)),
        HalTextureFormat::Rgba16Float => Ok((vk::Format::R16G16B16A16_SFLOAT, 8)),
        HalTextureFormat::Rgba32Uint => Ok((vk::Format::R32G32B32A32_UINT, 16)),
        HalTextureFormat::Rgba32Sint => Ok((vk::Format::R32G32B32A32_SINT, 16)),
        HalTextureFormat::Rgba32Float => Ok((vk::Format::R32G32B32A32_SFLOAT, 16)),
        HalTextureFormat::Stencil8 => Ok((vk::Format::S8_UINT, 1)),
        HalTextureFormat::Depth16Unorm => Ok((vk::Format::D16_UNORM, 2)),
        HalTextureFormat::Depth24Plus => Ok((vk::Format::D32_SFLOAT, 4)),
        HalTextureFormat::Depth24PlusStencil8 => Ok((vk::Format::D32_SFLOAT_S8_UINT, 5)),
        HalTextureFormat::Depth32Float => Ok((vk::Format::D32_SFLOAT, 4)),
        HalTextureFormat::Depth32FloatStencil8 => Ok((vk::Format::D32_SFLOAT_S8_UINT, 5)),
        HalTextureFormat::Unsupported => Err(texture_error("unsupported texture format")),
    }
}

/// Converts texture usage into the corresponding yawgpu representation.
pub(super) fn map_texture_usage(usage: HalTextureUsage) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    if usage.texture_binding {
        flags |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.storage_binding {
        flags |= vk::ImageUsageFlags::STORAGE;
    }
    if usage.render_attachment {
        flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        // With the tiled extension a render target can also be read back as a
        // subpass input attachment, which requires `INPUT_ATTACHMENT` usage.
        #[cfg(feature = "tiled")]
        {
            flags |= vk::ImageUsageFlags::INPUT_ATTACHMENT;
        }
    }
    flags
}

/// Converts buffer usage into the corresponding Vulkan representation.
pub(super) fn map_buffer_usage(usage: HalBufferUsage) -> vk::BufferUsageFlags {
    // TRANSFER_SRC | TRANSFER_DST are always set because yawgpu uses staging
    // buffers for mapAtCreation and writeBuffer regardless of the caller's
    // declared usage. See specs/tracking/vulkan-buffer-texture-usage-vuids.md § F1.
    let mut flags = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    if usage.index {
        flags |= vk::BufferUsageFlags::INDEX_BUFFER;
    }
    if usage.vertex {
        flags |= vk::BufferUsageFlags::VERTEX_BUFFER;
    }
    if usage.uniform {
        flags |= vk::BufferUsageFlags::UNIFORM_BUFFER;
    }
    if usage.storage {
        flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
    }
    if usage.indirect {
        flags |= vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    // query_resolve is a transfer-dst write; the bit is already on the
    // baseline above. map_read / map_write are host-memory properties and
    // have no Vulkan buffer-usage equivalent.
    flags
}

/// Converts vertex format into the corresponding yawgpu representation.
pub(super) fn map_vertex_format(format: HalVertexFormat) -> Result<vk::Format, HalError> {
    match format {
        HalVertexFormat::Float32 => Ok(vk::Format::R32_SFLOAT),
        HalVertexFormat::Float32x2 => Ok(vk::Format::R32G32_SFLOAT),
        HalVertexFormat::Float32x3 => Ok(vk::Format::R32G32B32_SFLOAT),
        HalVertexFormat::Float32x4 => Ok(vk::Format::R32G32B32A32_SFLOAT),
        HalVertexFormat::Unsupported => Err(shader_error("unsupported vertex format")),
    }
}

/// Converts primitive topology into the corresponding yawgpu representation.
pub(super) fn map_primitive_topology(topology: HalPrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        HalPrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
        HalPrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        HalPrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        HalPrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        HalPrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
}

/// Converts address mode into the corresponding yawgpu representation.
pub(super) fn map_address_mode(mode: HalAddressMode) -> vk::SamplerAddressMode {
    match mode {
        HalAddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        HalAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        HalAddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

/// Converts filter mode into the corresponding yawgpu representation.
pub(super) fn map_filter_mode(mode: HalFilterMode) -> vk::Filter {
    match mode {
        HalFilterMode::Nearest => vk::Filter::NEAREST,
        HalFilterMode::Linear => vk::Filter::LINEAR,
    }
}

/// Converts mipmap filter mode into the corresponding yawgpu representation.
pub(super) fn map_mipmap_filter_mode(mode: HalMipmapFilterMode) -> vk::SamplerMipmapMode {
    match mode {
        HalMipmapFilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        HalMipmapFilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

/// Converts compare function into the corresponding yawgpu representation.
pub(super) fn map_compare_function(compare: HalCompareFunction) -> vk::CompareOp {
    match compare {
        HalCompareFunction::Never => vk::CompareOp::NEVER,
        HalCompareFunction::Less => vk::CompareOp::LESS,
        HalCompareFunction::Equal => vk::CompareOp::EQUAL,
        HalCompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        HalCompareFunction::Greater => vk::CompareOp::GREATER,
        HalCompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        HalCompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        HalCompareFunction::Always => vk::CompareOp::ALWAYS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_buffer_usage_always_sets_transfer_src_and_transfer_dst() {
        let flags = map_buffer_usage(HalBufferUsage::default());

        assert!(flags.contains(vk::BufferUsageFlags::TRANSFER_SRC));
        assert!(flags.contains(vk::BufferUsageFlags::TRANSFER_DST));
    }

    #[test]
    fn map_texture_format_maps_uncompressed_color_formats() {
        let cases = [
            (HalTextureFormat::R8Unorm, vk::Format::R8_UNORM, 1),
            (HalTextureFormat::R8Snorm, vk::Format::R8_SNORM, 1),
            (HalTextureFormat::R8Uint, vk::Format::R8_UINT, 1),
            (HalTextureFormat::R8Sint, vk::Format::R8_SINT, 1),
            (HalTextureFormat::R16Unorm, vk::Format::R16_UNORM, 2),
            (HalTextureFormat::R16Snorm, vk::Format::R16_SNORM, 2),
            (HalTextureFormat::R16Uint, vk::Format::R16_UINT, 2),
            (HalTextureFormat::R16Sint, vk::Format::R16_SINT, 2),
            (HalTextureFormat::R16Float, vk::Format::R16_SFLOAT, 2),
            (HalTextureFormat::Rg8Unorm, vk::Format::R8G8_UNORM, 2),
            (HalTextureFormat::Rg8Snorm, vk::Format::R8G8_SNORM, 2),
            (HalTextureFormat::Rg8Uint, vk::Format::R8G8_UINT, 2),
            (HalTextureFormat::Rg8Sint, vk::Format::R8G8_SINT, 2),
            (HalTextureFormat::Rg16Unorm, vk::Format::R16G16_UNORM, 4),
            (HalTextureFormat::Rg16Snorm, vk::Format::R16G16_SNORM, 4),
            (HalTextureFormat::Rg16Uint, vk::Format::R16G16_UINT, 4),
            (HalTextureFormat::Rg16Sint, vk::Format::R16G16_SINT, 4),
            (HalTextureFormat::Rg16Float, vk::Format::R16G16_SFLOAT, 4),
            (HalTextureFormat::R32Uint, vk::Format::R32_UINT, 4),
            (HalTextureFormat::R32Sint, vk::Format::R32_SINT, 4),
            (HalTextureFormat::R32Float, vk::Format::R32_SFLOAT, 4),
            (HalTextureFormat::Rg32Uint, vk::Format::R32G32_UINT, 8),
            (HalTextureFormat::Rg32Sint, vk::Format::R32G32_SINT, 8),
            (HalTextureFormat::Rg32Float, vk::Format::R32G32_SFLOAT, 8),
            (HalTextureFormat::Rgba8Unorm, vk::Format::R8G8B8A8_UNORM, 4),
            (
                HalTextureFormat::Rgba8UnormSrgb,
                vk::Format::R8G8B8A8_SRGB,
                4,
            ),
            (HalTextureFormat::Rgba8Snorm, vk::Format::R8G8B8A8_SNORM, 4),
            (HalTextureFormat::Rgba8Uint, vk::Format::R8G8B8A8_UINT, 4),
            (HalTextureFormat::Rgba8Sint, vk::Format::R8G8B8A8_SINT, 4),
            (HalTextureFormat::Bgra8Unorm, vk::Format::B8G8R8A8_UNORM, 4),
            (
                HalTextureFormat::Bgra8UnormSrgb,
                vk::Format::B8G8R8A8_SRGB,
                4,
            ),
            (
                HalTextureFormat::Rgb10a2Uint,
                vk::Format::A2B10G10R10_UINT_PACK32,
                4,
            ),
            (
                HalTextureFormat::Rgb10a2Unorm,
                vk::Format::A2B10G10R10_UNORM_PACK32,
                4,
            ),
            (
                HalTextureFormat::Rg11b10Ufloat,
                vk::Format::B10G11R11_UFLOAT_PACK32,
                4,
            ),
            (
                HalTextureFormat::Rgb9e5Ufloat,
                vk::Format::E5B9G9R9_UFLOAT_PACK32,
                4,
            ),
            (
                HalTextureFormat::Rgba16Unorm,
                vk::Format::R16G16B16A16_UNORM,
                8,
            ),
            (
                HalTextureFormat::Rgba16Snorm,
                vk::Format::R16G16B16A16_SNORM,
                8,
            ),
            (
                HalTextureFormat::Rgba16Uint,
                vk::Format::R16G16B16A16_UINT,
                8,
            ),
            (
                HalTextureFormat::Rgba16Sint,
                vk::Format::R16G16B16A16_SINT,
                8,
            ),
            (
                HalTextureFormat::Rgba16Float,
                vk::Format::R16G16B16A16_SFLOAT,
                8,
            ),
            (
                HalTextureFormat::Rgba32Uint,
                vk::Format::R32G32B32A32_UINT,
                16,
            ),
            (
                HalTextureFormat::Rgba32Sint,
                vk::Format::R32G32B32A32_SINT,
                16,
            ),
            (
                HalTextureFormat::Rgba32Float,
                vk::Format::R32G32B32A32_SFLOAT,
                16,
            ),
        ];

        for (hal, vk, bytes_per_pixel) in cases {
            assert_eq!(
                map_texture_format(hal).expect("format supported"),
                (vk, bytes_per_pixel),
                "{hal:?}"
            );
        }
    }

    #[test]
    fn map_buffer_usage_sets_vertex_when_requested() {
        let usage = HalBufferUsage {
            vertex: true,
            ..HalBufferUsage::default()
        };

        assert!(map_buffer_usage(usage).contains(vk::BufferUsageFlags::VERTEX_BUFFER));
    }

    #[test]
    fn map_buffer_usage_sets_uniform_storage_index_indirect() {
        let cases = [
            (
                HalBufferUsage {
                    uniform: true,
                    ..HalBufferUsage::default()
                },
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            ),
            (
                HalBufferUsage {
                    storage: true,
                    ..HalBufferUsage::default()
                },
                vk::BufferUsageFlags::STORAGE_BUFFER,
            ),
            (
                HalBufferUsage {
                    index: true,
                    ..HalBufferUsage::default()
                },
                vk::BufferUsageFlags::INDEX_BUFFER,
            ),
            (
                HalBufferUsage {
                    indirect: true,
                    ..HalBufferUsage::default()
                },
                vk::BufferUsageFlags::INDIRECT_BUFFER,
            ),
        ];

        for (usage, expected) in cases {
            assert!(map_buffer_usage(usage).contains(expected));
        }
    }
}
