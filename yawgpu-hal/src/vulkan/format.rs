use super::*;

pub(super) fn map_texture_format(format: HalTextureFormat) -> Result<(vk::Format, u32), HalError> {
    match format {
        HalTextureFormat::R8Unorm => Ok((vk::Format::R8_UNORM, 1)),
        HalTextureFormat::Rgba8Unorm => Ok((vk::Format::R8G8B8A8_UNORM, 4)),
        HalTextureFormat::Bgra8Unorm => Ok((vk::Format::B8G8R8A8_UNORM, 4)),
        HalTextureFormat::Unsupported => Err(texture_error("unsupported texture format")),
    }
}

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
    }
    flags
}

pub(super) fn map_vertex_format(format: HalVertexFormat) -> Result<vk::Format, HalError> {
    match format {
        HalVertexFormat::Float32 => Ok(vk::Format::R32_SFLOAT),
        HalVertexFormat::Float32x2 => Ok(vk::Format::R32G32_SFLOAT),
        HalVertexFormat::Float32x3 => Ok(vk::Format::R32G32B32_SFLOAT),
        HalVertexFormat::Float32x4 => Ok(vk::Format::R32G32B32A32_SFLOAT),
        HalVertexFormat::Unsupported => Err(shader_error("unsupported vertex format")),
    }
}

pub(super) fn map_primitive_topology(topology: HalPrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        HalPrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
        HalPrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        HalPrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        HalPrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        HalPrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
}

pub(super) fn map_address_mode(mode: HalAddressMode) -> vk::SamplerAddressMode {
    match mode {
        HalAddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        HalAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        HalAddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

pub(super) fn map_filter_mode(mode: HalFilterMode) -> vk::Filter {
    match mode {
        HalFilterMode::Nearest => vk::Filter::NEAREST,
        HalFilterMode::Linear => vk::Filter::LINEAR,
    }
}

pub(super) fn map_mipmap_filter_mode(mode: HalMipmapFilterMode) -> vk::SamplerMipmapMode {
    match mode {
        HalMipmapFilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        HalMipmapFilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

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
