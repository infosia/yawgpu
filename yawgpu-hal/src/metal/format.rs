use super::*;

/// Converts this value into ns.
pub(super) fn to_ns(value: u64) -> Result<usize, HalError> {
    usize::try_from(value).map_err(|_| buffer_error("value is too large"))
}

/// Returns buffer error.
pub(super) fn buffer_error(message: &'static str) -> HalError {
    HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    }
}

/// Validates buffer texture range and returns a descriptive error on failure.
pub(super) fn validate_buffer_texture_range(
    buffer: &MetalBuffer,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let rows = u64::from(copy.extent.height.saturating_sub(1));
    let last_row = rows
        .checked_mul(u64::from(copy.buffer_layout.bytes_per_row))
        .ok_or_else(|| buffer_error("buffer texture row range overflows"))?;
    let images = u64::from(copy.extent.depth_or_array_layers.saturating_sub(1));
    let last_image = images
        .checked_mul(u64::from(copy.buffer_layout.bytes_per_row))
        .and_then(|bytes| bytes.checked_mul(u64::from(copy.buffer_layout.rows_per_image)))
        .ok_or_else(|| buffer_error("buffer texture image range overflows"))?;
    let row_bytes = u64::from(copy.extent.width)
        .checked_mul(u64::from(texture_bytes_per_pixel(copy)?))
        .ok_or_else(|| buffer_error("buffer texture row bytes overflow"))?;
    let required = copy
        .buffer_layout
        .offset
        .checked_add(last_image)
        .and_then(|offset| offset.checked_add(last_row))
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or_else(|| buffer_error("buffer texture range overflows"))?;
    if required > buffer.size() {
        return Err(buffer_error("buffer texture range exceeds buffer size"));
    }
    Ok(())
}

/// Returns the per-texel byte size of the *aspect* being copied. For a single
/// aspect of a depth/stencil format this is the plane's size (stencil = 1 byte;
/// depth = 2 or 4) rather than the whole texel — buffer⇄texture copies move one
/// plane at a time, so the buffer is laid out at the aspect's stride.
pub(super) fn texture_bytes_per_pixel(copy: &HalBufferTextureCopy) -> Result<u32, HalError> {
    let HalTexture::Metal(texture) = &copy.texture else {
        return Err(texture_error("texture is not Metal-backed"));
    };
    match copy.aspect {
        crate::HalTextureAspect::StencilOnly => Ok(1),
        crate::HalTextureAspect::DepthOnly => Ok(match copy.format {
            crate::HalTextureFormat::Depth16Unorm => 2,
            crate::HalTextureFormat::Depth32Float
            | crate::HalTextureFormat::Depth32FloatStencil8 => 4,
            _ => full_bytes_per_pixel(texture)?,
        }),
        crate::HalTextureAspect::All => full_bytes_per_pixel(texture),
    }
}

fn full_bytes_per_pixel(texture: &MetalTexture) -> Result<u32, HalError> {
    if texture.bytes_per_pixel == 0 {
        return Err(texture_error("unsupported texture format"));
    }
    Ok(texture.bytes_per_pixel)
}

/// Returns buffer texture bytes per image.
pub(super) fn buffer_texture_bytes_per_image(
    copy: &HalBufferTextureCopy,
) -> Result<usize, HalError> {
    let bytes = u64::from(copy.buffer_layout.bytes_per_row)
        .checked_mul(u64::from(copy.buffer_layout.rows_per_image))
        .ok_or_else(|| buffer_error("buffer texture bytes per image overflows"))?;
    to_ns(bytes)
}

/// Converts this value into mtl origin.
pub(super) fn to_mtl_origin(x: u32, y: u32, z: u32) -> Result<MTLOrigin, HalError> {
    Ok(MTLOrigin {
        x: to_ns(u64::from(x))?,
        y: to_ns(u64::from(y))?,
        z: to_ns(u64::from(z))?,
    })
}

/// Converts this value into mtl size.
pub(super) fn to_mtl_size(extent: HalExtent3d) -> Result<MTLSize, HalError> {
    Ok(MTLSize {
        width: to_ns(u64::from(extent.width))?,
        height: to_ns(u64::from(extent.height))?,
        depth: to_ns(u64::from(extent.depth_or_array_layers))?,
    })
}

/// Converts this value into mtl dispatch size.
pub(super) fn to_mtl_dispatch_size(size: (u32, u32, u32)) -> Result<MTLSize, HalError> {
    Ok(MTLSize {
        width: to_ns(u64::from(size.0))?,
        height: to_ns(u64::from(size.1))?,
        depth: to_ns(u64::from(size.2))?,
    })
}

/// Converts this value into mtl workgroup size.
pub(super) fn to_mtl_workgroup_size(size: (u32, u32, u32)) -> Result<MTLSize, HalError> {
    to_mtl_dispatch_size(size)
}

/// Converts texture format into the corresponding yawgpu representation.
pub(super) fn map_texture_format(
    format: HalTextureFormat,
) -> Result<(MTLPixelFormat, u32), HalError> {
    match format {
        HalTextureFormat::R8Unorm => Ok((MTLPixelFormat::R8Unorm, 1)),
        HalTextureFormat::R8Snorm => Ok((MTLPixelFormat::R8Snorm, 1)),
        HalTextureFormat::R8Uint => Ok((MTLPixelFormat::R8Uint, 1)),
        HalTextureFormat::R8Sint => Ok((MTLPixelFormat::R8Sint, 1)),
        HalTextureFormat::R16Unorm => Ok((MTLPixelFormat::R16Unorm, 2)),
        HalTextureFormat::R16Snorm => Ok((MTLPixelFormat::R16Snorm, 2)),
        HalTextureFormat::R16Uint => Ok((MTLPixelFormat::R16Uint, 2)),
        HalTextureFormat::R16Sint => Ok((MTLPixelFormat::R16Sint, 2)),
        HalTextureFormat::R16Float => Ok((MTLPixelFormat::R16Float, 2)),
        HalTextureFormat::Rg8Unorm => Ok((MTLPixelFormat::RG8Unorm, 2)),
        HalTextureFormat::Rg8Snorm => Ok((MTLPixelFormat::RG8Snorm, 2)),
        HalTextureFormat::Rg8Uint => Ok((MTLPixelFormat::RG8Uint, 2)),
        HalTextureFormat::Rg8Sint => Ok((MTLPixelFormat::RG8Sint, 2)),
        HalTextureFormat::Rg16Unorm => Ok((MTLPixelFormat::RG16Unorm, 4)),
        HalTextureFormat::Rg16Snorm => Ok((MTLPixelFormat::RG16Snorm, 4)),
        HalTextureFormat::Rg16Uint => Ok((MTLPixelFormat::RG16Uint, 4)),
        HalTextureFormat::Rg16Sint => Ok((MTLPixelFormat::RG16Sint, 4)),
        HalTextureFormat::Rg16Float => Ok((MTLPixelFormat::RG16Float, 4)),
        HalTextureFormat::R32Uint => Ok((MTLPixelFormat::R32Uint, 4)),
        HalTextureFormat::R32Sint => Ok((MTLPixelFormat::R32Sint, 4)),
        HalTextureFormat::R32Float => Ok((MTLPixelFormat::R32Float, 4)),
        HalTextureFormat::Rg32Uint => Ok((MTLPixelFormat::RG32Uint, 8)),
        HalTextureFormat::Rg32Sint => Ok((MTLPixelFormat::RG32Sint, 8)),
        HalTextureFormat::Rg32Float => Ok((MTLPixelFormat::RG32Float, 8)),
        HalTextureFormat::Rgba8Unorm => Ok((MTLPixelFormat::RGBA8Unorm, 4)),
        HalTextureFormat::Rgba8UnormSrgb => Ok((MTLPixelFormat::RGBA8Unorm_sRGB, 4)),
        HalTextureFormat::Rgba8Snorm => Ok((MTLPixelFormat::RGBA8Snorm, 4)),
        HalTextureFormat::Rgba8Uint => Ok((MTLPixelFormat::RGBA8Uint, 4)),
        HalTextureFormat::Rgba8Sint => Ok((MTLPixelFormat::RGBA8Sint, 4)),
        HalTextureFormat::Bgra8Unorm => Ok((MTLPixelFormat::BGRA8Unorm, 4)),
        HalTextureFormat::Bgra8UnormSrgb => Ok((MTLPixelFormat::BGRA8Unorm_sRGB, 4)),
        HalTextureFormat::Rgb10a2Uint => Ok((MTLPixelFormat::RGB10A2Uint, 4)),
        HalTextureFormat::Rgb10a2Unorm => Ok((MTLPixelFormat::RGB10A2Unorm, 4)),
        HalTextureFormat::Rg11b10Ufloat => Ok((MTLPixelFormat::RG11B10Float, 4)),
        HalTextureFormat::Rgb9e5Ufloat => Ok((MTLPixelFormat::RGB9E5Float, 4)),
        HalTextureFormat::Rgba16Unorm => Ok((MTLPixelFormat::RGBA16Unorm, 8)),
        HalTextureFormat::Rgba16Snorm => Ok((MTLPixelFormat::RGBA16Snorm, 8)),
        HalTextureFormat::Rgba16Uint => Ok((MTLPixelFormat::RGBA16Uint, 8)),
        HalTextureFormat::Rgba16Sint => Ok((MTLPixelFormat::RGBA16Sint, 8)),
        HalTextureFormat::Rgba16Float => Ok((MTLPixelFormat::RGBA16Float, 8)),
        HalTextureFormat::Rgba32Uint => Ok((MTLPixelFormat::RGBA32Uint, 16)),
        HalTextureFormat::Rgba32Sint => Ok((MTLPixelFormat::RGBA32Sint, 16)),
        HalTextureFormat::Rgba32Float => Ok((MTLPixelFormat::RGBA32Float, 16)),
        HalTextureFormat::Stencil8 => Ok((MTLPixelFormat::Stencil8, 1)),
        HalTextureFormat::Depth16Unorm => Ok((MTLPixelFormat::Depth16Unorm, 2)),
        HalTextureFormat::Depth24Plus => Ok((MTLPixelFormat::Depth32Float, 4)),
        HalTextureFormat::Depth24PlusStencil8 => Ok((MTLPixelFormat::Depth32Float_Stencil8, 5)),
        HalTextureFormat::Depth32Float => Ok((MTLPixelFormat::Depth32Float, 4)),
        HalTextureFormat::Depth32FloatStencil8 => Ok((MTLPixelFormat::Depth32Float_Stencil8, 5)),
        HalTextureFormat::Unsupported => Err(texture_error("unsupported texture format")),
    }
}

/// Converts texture usage into the corresponding yawgpu representation.
pub(super) fn map_texture_usage(usage: HalTextureUsage) -> MTLTextureUsage {
    let mut metal_usage = MTLTextureUsage::Unknown;
    if usage.copy_src || usage.texture_binding || usage.storage_binding {
        metal_usage |= MTLTextureUsage::ShaderRead;
    }
    if usage.copy_dst || usage.storage_binding {
        metal_usage |= MTLTextureUsage::ShaderWrite;
    }
    if usage.render_attachment {
        metal_usage |= MTLTextureUsage::RenderTarget;
    }
    metal_usage
}

/// Converts address mode into the corresponding yawgpu representation.
pub(super) fn map_address_mode(mode: HalAddressMode) -> MTLSamplerAddressMode {
    match mode {
        HalAddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
        HalAddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        HalAddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
    }
}

/// Converts filter mode into the corresponding yawgpu representation.
pub(super) fn map_filter_mode(mode: HalFilterMode) -> MTLSamplerMinMagFilter {
    match mode {
        HalFilterMode::Nearest => MTLSamplerMinMagFilter::Nearest,
        HalFilterMode::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

/// Converts mipmap filter mode into the corresponding yawgpu representation.
pub(super) fn map_mipmap_filter_mode(mode: HalMipmapFilterMode) -> MTLSamplerMipFilter {
    match mode {
        HalMipmapFilterMode::Nearest => MTLSamplerMipFilter::Nearest,
        HalMipmapFilterMode::Linear => MTLSamplerMipFilter::Linear,
    }
}

/// Converts compare function into the corresponding yawgpu representation.
pub(super) fn map_compare_function(compare: HalCompareFunction) -> MTLCompareFunction {
    match compare {
        HalCompareFunction::Never => MTLCompareFunction::Never,
        HalCompareFunction::Less => MTLCompareFunction::Less,
        HalCompareFunction::Equal => MTLCompareFunction::Equal,
        HalCompareFunction::LessEqual => MTLCompareFunction::LessEqual,
        HalCompareFunction::Greater => MTLCompareFunction::Greater,
        HalCompareFunction::NotEqual => MTLCompareFunction::NotEqual,
        HalCompareFunction::GreaterEqual => MTLCompareFunction::GreaterEqual,
        HalCompareFunction::Always => MTLCompareFunction::Always,
    }
}

/// Converts vertex format into the corresponding yawgpu representation.
pub(super) fn map_vertex_format(format: HalVertexFormat) -> Result<MTLVertexFormat, HalError> {
    match format {
        HalVertexFormat::Float32 => Ok(MTLVertexFormat::Float),
        HalVertexFormat::Float32x2 => Ok(MTLVertexFormat::Float2),
        HalVertexFormat::Float32x3 => Ok(MTLVertexFormat::Float3),
        HalVertexFormat::Float32x4 => Ok(MTLVertexFormat::Float4),
        HalVertexFormat::Unsupported => Err(shader_error(
            "unsupported vertex format for Metal".to_owned(),
        )),
    }
}

/// Converts primitive topology into the corresponding yawgpu representation.
pub(super) fn map_primitive_topology(topology: HalPrimitiveTopology) -> MTLPrimitiveType {
    match topology {
        HalPrimitiveTopology::PointList => MTLPrimitiveType::Point,
        HalPrimitiveTopology::LineList => MTLPrimitiveType::Line,
        HalPrimitiveTopology::LineStrip => MTLPrimitiveType::LineStrip,
        HalPrimitiveTopology::TriangleList => MTLPrimitiveType::Triangle,
        HalPrimitiveTopology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
    }
}

/// Returns texture error.
pub(super) fn texture_error(message: &'static str) -> HalError {
    HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    }
}

/// Returns shader error.
pub(super) fn shader_error(message: String) -> HalError {
    HalError::ShaderCompilationFailed {
        backend: BACKEND,
        message,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "metal")]
    fn map_texture_format_maps_uncompressed_color_formats() {
        let cases = [
            (HalTextureFormat::R8Unorm, MTLPixelFormat::R8Unorm, 1),
            (HalTextureFormat::R8Snorm, MTLPixelFormat::R8Snorm, 1),
            (HalTextureFormat::R8Uint, MTLPixelFormat::R8Uint, 1),
            (HalTextureFormat::R8Sint, MTLPixelFormat::R8Sint, 1),
            (HalTextureFormat::R16Unorm, MTLPixelFormat::R16Unorm, 2),
            (HalTextureFormat::R16Snorm, MTLPixelFormat::R16Snorm, 2),
            (HalTextureFormat::R16Uint, MTLPixelFormat::R16Uint, 2),
            (HalTextureFormat::R16Sint, MTLPixelFormat::R16Sint, 2),
            (HalTextureFormat::R16Float, MTLPixelFormat::R16Float, 2),
            (HalTextureFormat::Rg8Unorm, MTLPixelFormat::RG8Unorm, 2),
            (HalTextureFormat::Rg8Snorm, MTLPixelFormat::RG8Snorm, 2),
            (HalTextureFormat::Rg8Uint, MTLPixelFormat::RG8Uint, 2),
            (HalTextureFormat::Rg8Sint, MTLPixelFormat::RG8Sint, 2),
            (HalTextureFormat::Rg16Unorm, MTLPixelFormat::RG16Unorm, 4),
            (HalTextureFormat::Rg16Snorm, MTLPixelFormat::RG16Snorm, 4),
            (HalTextureFormat::Rg16Uint, MTLPixelFormat::RG16Uint, 4),
            (HalTextureFormat::Rg16Sint, MTLPixelFormat::RG16Sint, 4),
            (HalTextureFormat::Rg16Float, MTLPixelFormat::RG16Float, 4),
            (HalTextureFormat::R32Uint, MTLPixelFormat::R32Uint, 4),
            (HalTextureFormat::R32Sint, MTLPixelFormat::R32Sint, 4),
            (HalTextureFormat::R32Float, MTLPixelFormat::R32Float, 4),
            (HalTextureFormat::Rg32Uint, MTLPixelFormat::RG32Uint, 8),
            (HalTextureFormat::Rg32Sint, MTLPixelFormat::RG32Sint, 8),
            (HalTextureFormat::Rg32Float, MTLPixelFormat::RG32Float, 8),
            (HalTextureFormat::Rgba8Unorm, MTLPixelFormat::RGBA8Unorm, 4),
            (
                HalTextureFormat::Rgba8UnormSrgb,
                MTLPixelFormat::RGBA8Unorm_sRGB,
                4,
            ),
            (HalTextureFormat::Rgba8Snorm, MTLPixelFormat::RGBA8Snorm, 4),
            (HalTextureFormat::Rgba8Uint, MTLPixelFormat::RGBA8Uint, 4),
            (HalTextureFormat::Rgba8Sint, MTLPixelFormat::RGBA8Sint, 4),
            (HalTextureFormat::Bgra8Unorm, MTLPixelFormat::BGRA8Unorm, 4),
            (
                HalTextureFormat::Bgra8UnormSrgb,
                MTLPixelFormat::BGRA8Unorm_sRGB,
                4,
            ),
            (
                HalTextureFormat::Rgb10a2Uint,
                MTLPixelFormat::RGB10A2Uint,
                4,
            ),
            (
                HalTextureFormat::Rgb10a2Unorm,
                MTLPixelFormat::RGB10A2Unorm,
                4,
            ),
            (
                HalTextureFormat::Rg11b10Ufloat,
                MTLPixelFormat::RG11B10Float,
                4,
            ),
            (
                HalTextureFormat::Rgb9e5Ufloat,
                MTLPixelFormat::RGB9E5Float,
                4,
            ),
            (
                HalTextureFormat::Rgba16Unorm,
                MTLPixelFormat::RGBA16Unorm,
                8,
            ),
            (
                HalTextureFormat::Rgba16Snorm,
                MTLPixelFormat::RGBA16Snorm,
                8,
            ),
            (HalTextureFormat::Rgba16Uint, MTLPixelFormat::RGBA16Uint, 8),
            (HalTextureFormat::Rgba16Sint, MTLPixelFormat::RGBA16Sint, 8),
            (
                HalTextureFormat::Rgba16Float,
                MTLPixelFormat::RGBA16Float,
                8,
            ),
            (HalTextureFormat::Rgba32Uint, MTLPixelFormat::RGBA32Uint, 16),
            (HalTextureFormat::Rgba32Sint, MTLPixelFormat::RGBA32Sint, 16),
            (
                HalTextureFormat::Rgba32Float,
                MTLPixelFormat::RGBA32Float,
                16,
            ),
        ];

        for (hal, metal, bytes_per_pixel) in cases {
            assert_eq!(
                map_texture_format(hal).expect("format supported"),
                (metal, bytes_per_pixel),
                "{hal:?}"
            );
        }
    }
}
