use super::BACKEND;
use crate::{HalError, HalPrimitiveTopology, HalTextureFormat, HalVertexFormat};

#[derive(Clone, Copy, Debug)]
pub(super) struct GlesFormat {
    pub(super) internal: u32,
    pub(super) format: u32,
    pub(super) ty: u32,
    pub(super) bytes_per_pixel: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GlesVertexFormat {
    pub(super) components: i32,
    pub(super) ty: u32,
    pub(super) normalized: bool,
}

pub(super) fn map_texture_format(format: HalTextureFormat) -> Result<GlesFormat, HalError> {
    match format {
        HalTextureFormat::R8Unorm => Ok(gles_format(glow::R8, glow::RED, glow::UNSIGNED_BYTE, 1)),
        HalTextureFormat::R8Snorm => Ok(gles_format(glow::R8_SNORM, glow::RED, glow::BYTE, 1)),
        HalTextureFormat::R8Uint => Ok(gles_format(
            glow::R8UI,
            glow::RED_INTEGER,
            glow::UNSIGNED_BYTE,
            1,
        )),
        HalTextureFormat::R8Sint => Ok(gles_format(glow::R8I, glow::RED_INTEGER, glow::BYTE, 1)),
        HalTextureFormat::R16Unorm => {
            Ok(gles_format(glow::R16, glow::RED, glow::UNSIGNED_SHORT, 2))
        }
        HalTextureFormat::R16Snorm => Ok(gles_format(glow::R16_SNORM, glow::RED, glow::SHORT, 2)),
        HalTextureFormat::R16Uint => Ok(gles_format(
            glow::R16UI,
            glow::RED_INTEGER,
            glow::UNSIGNED_SHORT,
            2,
        )),
        HalTextureFormat::R16Sint => Ok(gles_format(glow::R16I, glow::RED_INTEGER, glow::SHORT, 2)),
        HalTextureFormat::R16Float => Ok(gles_format(glow::R16F, glow::RED, glow::HALF_FLOAT, 2)),
        HalTextureFormat::Rg8Unorm => Ok(gles_format(glow::RG8, glow::RG, glow::UNSIGNED_BYTE, 2)),
        HalTextureFormat::Rg8Snorm => Ok(gles_format(glow::RG8_SNORM, glow::RG, glow::BYTE, 2)),
        HalTextureFormat::Rg8Uint => Ok(gles_format(
            glow::RG8UI,
            glow::RG_INTEGER,
            glow::UNSIGNED_BYTE,
            2,
        )),
        HalTextureFormat::Rg8Sint => Ok(gles_format(glow::RG8I, glow::RG_INTEGER, glow::BYTE, 2)),
        HalTextureFormat::Rg16Unorm => {
            Ok(gles_format(glow::RG16, glow::RG, glow::UNSIGNED_SHORT, 4))
        }
        HalTextureFormat::Rg16Snorm => Ok(gles_format(glow::RG16_SNORM, glow::RG, glow::SHORT, 4)),
        HalTextureFormat::Rg16Uint => Ok(gles_format(
            glow::RG16UI,
            glow::RG_INTEGER,
            glow::UNSIGNED_SHORT,
            4,
        )),
        HalTextureFormat::Rg16Sint => {
            Ok(gles_format(glow::RG16I, glow::RG_INTEGER, glow::SHORT, 4))
        }
        HalTextureFormat::Rg16Float => Ok(gles_format(glow::RG16F, glow::RG, glow::HALF_FLOAT, 4)),
        HalTextureFormat::R32Uint => Ok(gles_format(
            glow::R32UI,
            glow::RED_INTEGER,
            glow::UNSIGNED_INT,
            4,
        )),
        HalTextureFormat::R32Sint => Ok(gles_format(glow::R32I, glow::RED_INTEGER, glow::INT, 4)),
        HalTextureFormat::R32Float => Ok(gles_format(glow::R32F, glow::RED, glow::FLOAT, 4)),
        HalTextureFormat::Rg32Uint => Ok(gles_format(
            glow::RG32UI,
            glow::RG_INTEGER,
            glow::UNSIGNED_INT,
            8,
        )),
        HalTextureFormat::Rg32Sint => Ok(gles_format(glow::RG32I, glow::RG_INTEGER, glow::INT, 8)),
        HalTextureFormat::Rg32Float => Ok(gles_format(glow::RG32F, glow::RG, glow::FLOAT, 8)),
        HalTextureFormat::Rgba8Unorm => Ok(rgba8_unorm()),
        HalTextureFormat::Rgba8UnormSrgb => Ok(gles_format(
            glow::SRGB8_ALPHA8,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            4,
        )),
        HalTextureFormat::Rgba8Snorm => {
            Ok(gles_format(glow::RGBA8_SNORM, glow::RGBA, glow::BYTE, 4))
        }
        HalTextureFormat::Rgba8Uint => Ok(rgba8_uint()),
        HalTextureFormat::Rgba8Sint => {
            Ok(gles_format(glow::RGBA8I, glow::RGBA_INTEGER, glow::BYTE, 4))
        }
        HalTextureFormat::Bgra8Unorm => {
            Ok(gles_format(glow::RGBA8, glow::BGRA, glow::UNSIGNED_BYTE, 4))
        }
        HalTextureFormat::Bgra8UnormSrgb => Ok(gles_format(
            glow::SRGB8_ALPHA8,
            glow::BGRA,
            glow::UNSIGNED_BYTE,
            4,
        )),
        HalTextureFormat::Rgb10a2Uint => Ok(gles_format(
            glow::RGB10_A2UI,
            glow::RGBA_INTEGER,
            glow::UNSIGNED_INT_2_10_10_10_REV,
            4,
        )),
        HalTextureFormat::Rgb10a2Unorm => Ok(gles_format(
            glow::RGB10_A2,
            glow::RGBA,
            glow::UNSIGNED_INT_2_10_10_10_REV,
            4,
        )),
        HalTextureFormat::Rg11b10Ufloat => Ok(gles_format(
            glow::R11F_G11F_B10F,
            glow::RGB,
            glow::UNSIGNED_INT_10F_11F_11F_REV,
            4,
        )),
        HalTextureFormat::Rgb9e5Ufloat => Ok(gles_format(
            glow::RGB9_E5,
            glow::RGB,
            glow::UNSIGNED_INT_5_9_9_9_REV,
            4,
        )),
        HalTextureFormat::Rgba16Unorm => Ok(gles_format(
            glow::RGBA16,
            glow::RGBA,
            glow::UNSIGNED_SHORT,
            8,
        )),
        HalTextureFormat::Rgba16Snorm => {
            Ok(gles_format(glow::RGBA16_SNORM, glow::RGBA, glow::SHORT, 8))
        }
        HalTextureFormat::Rgba16Uint => Ok(gles_format(
            glow::RGBA16UI,
            glow::RGBA_INTEGER,
            glow::UNSIGNED_SHORT,
            8,
        )),
        HalTextureFormat::Rgba16Sint => Ok(gles_format(
            glow::RGBA16I,
            glow::RGBA_INTEGER,
            glow::SHORT,
            8,
        )),
        HalTextureFormat::Rgba16Float => {
            Ok(gles_format(glow::RGBA16F, glow::RGBA, glow::HALF_FLOAT, 8))
        }
        HalTextureFormat::Rgba32Uint => Ok(gles_format(
            glow::RGBA32UI,
            glow::RGBA_INTEGER,
            glow::UNSIGNED_INT,
            16,
        )),
        HalTextureFormat::Rgba32Sint => Ok(gles_format(
            glow::RGBA32I,
            glow::RGBA_INTEGER,
            glow::INT,
            16,
        )),
        HalTextureFormat::Rgba32Float => {
            Ok(gles_format(glow::RGBA32F, glow::RGBA, glow::FLOAT, 16))
        }
        HalTextureFormat::Bc1RgbaUnorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_S3TC_DXT1_EXT,
            glow::RGBA,
            0,
            8,
        )),
        HalTextureFormat::Bc1RgbaUnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,
            glow::RGBA,
            0,
            8,
        )),
        HalTextureFormat::Bc2RgbaUnorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_S3TC_DXT3_EXT,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Bc2RgbaUnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Bc3RgbaUnorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_S3TC_DXT5_EXT,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Bc3RgbaUnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Bc4RUnorm => Ok(gles_format(glow::COMPRESSED_RED_RGTC1, glow::RED, 0, 8)),
        HalTextureFormat::Bc4RSnorm => Ok(gles_format(
            glow::COMPRESSED_SIGNED_RED_RGTC1,
            glow::RED,
            0,
            8,
        )),
        HalTextureFormat::Bc5RgUnorm => Ok(gles_format(glow::COMPRESSED_RG_RGTC2, glow::RG, 0, 16)),
        HalTextureFormat::Bc5RgSnorm => Ok(gles_format(
            glow::COMPRESSED_SIGNED_RG_RGTC2,
            glow::RG,
            0,
            16,
        )),
        HalTextureFormat::Bc6hRgbUfloat => Ok(gles_format(
            glow::COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT,
            glow::RGB,
            0,
            16,
        )),
        HalTextureFormat::Bc6hRgbFloat => Ok(gles_format(
            glow::COMPRESSED_RGB_BPTC_SIGNED_FLOAT,
            glow::RGB,
            0,
            16,
        )),
        HalTextureFormat::Bc7RgbaUnorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_BPTC_UNORM,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Bc7RgbaUnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB_ALPHA_BPTC_UNORM,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Etc2Rgb8Unorm => {
            Ok(gles_format(glow::COMPRESSED_RGB8_ETC2, glow::RGB, 0, 8))
        }
        HalTextureFormat::Etc2Rgb8UnormSrgb => {
            Ok(gles_format(glow::COMPRESSED_SRGB8_ETC2, glow::RGB, 0, 8))
        }
        HalTextureFormat::Etc2Rgb8a1Unorm => Ok(gles_format(
            glow::COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,
            glow::RGBA,
            0,
            8,
        )),
        HalTextureFormat::Etc2Rgb8a1UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2,
            glow::RGBA,
            0,
            8,
        )),
        HalTextureFormat::Etc2Rgba8Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA8_ETC2_EAC,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Etc2Rgba8UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::EacR11Unorm => Ok(gles_format(glow::COMPRESSED_R11_EAC, glow::RED, 0, 8)),
        HalTextureFormat::EacR11Snorm => Ok(gles_format(
            glow::COMPRESSED_SIGNED_R11_EAC,
            glow::RED,
            0,
            8,
        )),
        HalTextureFormat::EacRg11Unorm => {
            Ok(gles_format(glow::COMPRESSED_RG11_EAC, glow::RG, 0, 16))
        }
        HalTextureFormat::EacRg11Snorm => Ok(gles_format(
            glow::COMPRESSED_SIGNED_RG11_EAC,
            glow::RG,
            0,
            16,
        )),
        HalTextureFormat::Astc4x4Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_4x4_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc4x4UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc5x4Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_5x4_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc5x4UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc5x5Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_5x5_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc5x5UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc6x5Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_6x5_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc6x5UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc6x6Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_6x6_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc6x6UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc8x5Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_8x5_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc8x5UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc8x6Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_8x6_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc8x6UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc8x8Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_8x8_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc8x8UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc10x5Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_10x5_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc10x5UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc10x6Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_10x6_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc10x6UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc10x8Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_10x8_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc10x8UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc10x10Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_10x10_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc10x10UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc12x10Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_12x10_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc12x10UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc12x12Unorm => Ok(gles_format(
            glow::COMPRESSED_RGBA_ASTC_12x12_KHR,
            glow::RGBA,
            0,
            16,
        )),
        HalTextureFormat::Astc12x12UnormSrgb => Ok(gles_format(
            glow::COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR,
            glow::RGBA,
            0,
            16,
        )),
        _ => Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture format not supported on GLES (P15.3)",
        }),
    }
}

pub(super) fn map_vertex_format(format: HalVertexFormat) -> Result<GlesVertexFormat, HalError> {
    match format {
        HalVertexFormat::Float32 => Ok(GlesVertexFormat {
            components: 1,
            ty: glow::FLOAT,
            normalized: false,
        }),
        HalVertexFormat::Float32x2 => Ok(GlesVertexFormat {
            components: 2,
            ty: glow::FLOAT,
            normalized: false,
        }),
        HalVertexFormat::Float32x3 => Ok(GlesVertexFormat {
            components: 3,
            ty: glow::FLOAT,
            normalized: false,
        }),
        HalVertexFormat::Float32x4 => Ok(GlesVertexFormat {
            components: 4,
            ty: glow::FLOAT,
            normalized: false,
        }),
        HalVertexFormat::Unsupported => Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "Unsupported vertex format requested",
        }),
    }
}

pub(super) fn map_primitive_topology(topology: HalPrimitiveTopology) -> u32 {
    match topology {
        HalPrimitiveTopology::PointList => glow::POINTS,
        HalPrimitiveTopology::LineList => glow::LINES,
        HalPrimitiveTopology::LineStrip => glow::LINE_STRIP,
        HalPrimitiveTopology::TriangleList => glow::TRIANGLES,
        HalPrimitiveTopology::TriangleStrip => glow::TRIANGLE_STRIP,
    }
}

pub(super) fn fallback_format() -> GlesFormat {
    rgba8_unorm()
}

fn rgba8_unorm() -> GlesFormat {
    gles_format(glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE, 4)
}

fn rgba8_uint() -> GlesFormat {
    gles_format(glow::RGBA8UI, glow::RGBA_INTEGER, glow::UNSIGNED_BYTE, 4)
}

fn gles_format(internal: u32, format: u32, ty: u32, bytes_per_pixel: u32) -> GlesFormat {
    GlesFormat {
        internal,
        format,
        ty,
        bytes_per_pixel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_texture_format_maps_uncompressed_color_formats() {
        let cases = [
            (
                HalTextureFormat::R8Unorm,
                glow::R8,
                glow::RED,
                glow::UNSIGNED_BYTE,
                1,
            ),
            (
                HalTextureFormat::R8Snorm,
                glow::R8_SNORM,
                glow::RED,
                glow::BYTE,
                1,
            ),
            (
                HalTextureFormat::R8Uint,
                glow::R8UI,
                glow::RED_INTEGER,
                glow::UNSIGNED_BYTE,
                1,
            ),
            (
                HalTextureFormat::R8Sint,
                glow::R8I,
                glow::RED_INTEGER,
                glow::BYTE,
                1,
            ),
            (
                HalTextureFormat::R16Unorm,
                glow::R16,
                glow::RED,
                glow::UNSIGNED_SHORT,
                2,
            ),
            (
                HalTextureFormat::R16Snorm,
                glow::R16_SNORM,
                glow::RED,
                glow::SHORT,
                2,
            ),
            (
                HalTextureFormat::R16Uint,
                glow::R16UI,
                glow::RED_INTEGER,
                glow::UNSIGNED_SHORT,
                2,
            ),
            (
                HalTextureFormat::R16Sint,
                glow::R16I,
                glow::RED_INTEGER,
                glow::SHORT,
                2,
            ),
            (
                HalTextureFormat::R16Float,
                glow::R16F,
                glow::RED,
                glow::HALF_FLOAT,
                2,
            ),
            (
                HalTextureFormat::Rg8Unorm,
                glow::RG8,
                glow::RG,
                glow::UNSIGNED_BYTE,
                2,
            ),
            (
                HalTextureFormat::Rg8Snorm,
                glow::RG8_SNORM,
                glow::RG,
                glow::BYTE,
                2,
            ),
            (
                HalTextureFormat::Rg8Uint,
                glow::RG8UI,
                glow::RG_INTEGER,
                glow::UNSIGNED_BYTE,
                2,
            ),
            (
                HalTextureFormat::Rg8Sint,
                glow::RG8I,
                glow::RG_INTEGER,
                glow::BYTE,
                2,
            ),
            (
                HalTextureFormat::Rg16Unorm,
                glow::RG16,
                glow::RG,
                glow::UNSIGNED_SHORT,
                4,
            ),
            (
                HalTextureFormat::Rg16Snorm,
                glow::RG16_SNORM,
                glow::RG,
                glow::SHORT,
                4,
            ),
            (
                HalTextureFormat::Rg16Uint,
                glow::RG16UI,
                glow::RG_INTEGER,
                glow::UNSIGNED_SHORT,
                4,
            ),
            (
                HalTextureFormat::Rg16Sint,
                glow::RG16I,
                glow::RG_INTEGER,
                glow::SHORT,
                4,
            ),
            (
                HalTextureFormat::Rg16Float,
                glow::RG16F,
                glow::RG,
                glow::HALF_FLOAT,
                4,
            ),
            (
                HalTextureFormat::R32Uint,
                glow::R32UI,
                glow::RED_INTEGER,
                glow::UNSIGNED_INT,
                4,
            ),
            (
                HalTextureFormat::R32Sint,
                glow::R32I,
                glow::RED_INTEGER,
                glow::INT,
                4,
            ),
            (
                HalTextureFormat::R32Float,
                glow::R32F,
                glow::RED,
                glow::FLOAT,
                4,
            ),
            (
                HalTextureFormat::Rg32Uint,
                glow::RG32UI,
                glow::RG_INTEGER,
                glow::UNSIGNED_INT,
                8,
            ),
            (
                HalTextureFormat::Rg32Sint,
                glow::RG32I,
                glow::RG_INTEGER,
                glow::INT,
                8,
            ),
            (
                HalTextureFormat::Rg32Float,
                glow::RG32F,
                glow::RG,
                glow::FLOAT,
                8,
            ),
            (
                HalTextureFormat::Rgba8Unorm,
                glow::RGBA8,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                4,
            ),
            (
                HalTextureFormat::Rgba8UnormSrgb,
                glow::SRGB8_ALPHA8,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                4,
            ),
            (
                HalTextureFormat::Rgba8Snorm,
                glow::RGBA8_SNORM,
                glow::RGBA,
                glow::BYTE,
                4,
            ),
            (
                HalTextureFormat::Rgba8Uint,
                glow::RGBA8UI,
                glow::RGBA_INTEGER,
                glow::UNSIGNED_BYTE,
                4,
            ),
            (
                HalTextureFormat::Rgba8Sint,
                glow::RGBA8I,
                glow::RGBA_INTEGER,
                glow::BYTE,
                4,
            ),
            (
                HalTextureFormat::Bgra8Unorm,
                glow::RGBA8,
                glow::BGRA,
                glow::UNSIGNED_BYTE,
                4,
            ),
            (
                HalTextureFormat::Bgra8UnormSrgb,
                glow::SRGB8_ALPHA8,
                glow::BGRA,
                glow::UNSIGNED_BYTE,
                4,
            ),
            (
                HalTextureFormat::Rgb10a2Uint,
                glow::RGB10_A2UI,
                glow::RGBA_INTEGER,
                glow::UNSIGNED_INT_2_10_10_10_REV,
                4,
            ),
            (
                HalTextureFormat::Rgb10a2Unorm,
                glow::RGB10_A2,
                glow::RGBA,
                glow::UNSIGNED_INT_2_10_10_10_REV,
                4,
            ),
            (
                HalTextureFormat::Rg11b10Ufloat,
                glow::R11F_G11F_B10F,
                glow::RGB,
                glow::UNSIGNED_INT_10F_11F_11F_REV,
                4,
            ),
            (
                HalTextureFormat::Rgb9e5Ufloat,
                glow::RGB9_E5,
                glow::RGB,
                glow::UNSIGNED_INT_5_9_9_9_REV,
                4,
            ),
            (
                HalTextureFormat::Rgba16Unorm,
                glow::RGBA16,
                glow::RGBA,
                glow::UNSIGNED_SHORT,
                8,
            ),
            (
                HalTextureFormat::Rgba16Snorm,
                glow::RGBA16_SNORM,
                glow::RGBA,
                glow::SHORT,
                8,
            ),
            (
                HalTextureFormat::Rgba16Uint,
                glow::RGBA16UI,
                glow::RGBA_INTEGER,
                glow::UNSIGNED_SHORT,
                8,
            ),
            (
                HalTextureFormat::Rgba16Sint,
                glow::RGBA16I,
                glow::RGBA_INTEGER,
                glow::SHORT,
                8,
            ),
            (
                HalTextureFormat::Rgba16Float,
                glow::RGBA16F,
                glow::RGBA,
                glow::HALF_FLOAT,
                8,
            ),
            (
                HalTextureFormat::Rgba32Uint,
                glow::RGBA32UI,
                glow::RGBA_INTEGER,
                glow::UNSIGNED_INT,
                16,
            ),
            (
                HalTextureFormat::Rgba32Sint,
                glow::RGBA32I,
                glow::RGBA_INTEGER,
                glow::INT,
                16,
            ),
            (
                HalTextureFormat::Rgba32Float,
                glow::RGBA32F,
                glow::RGBA,
                glow::FLOAT,
                16,
            ),
        ];

        for (texture_format, internal, external, ty, bytes_per_pixel) in cases {
            let format = map_texture_format(texture_format).expect("format supported");
            assert_eq!(format.internal, internal, "{texture_format:?}");
            assert_eq!(format.format, external, "{texture_format:?}");
            assert_eq!(format.ty, ty, "{texture_format:?}");
            assert_eq!(
                format.bytes_per_pixel, bytes_per_pixel,
                "{texture_format:?}"
            );
        }
    }

    #[test]
    fn map_texture_format_unsupported_returns_error() {
        let error = map_texture_format(HalTextureFormat::Depth32Float)
            .expect_err("unsupported format must error");

        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "texture format not supported on GLES (P15.3)",
            }
        ));
    }

    #[test]
    fn map_vertex_format_table() {
        let float = map_vertex_format(HalVertexFormat::Float32).expect("Float32 supported");
        assert_eq!(float.components, 1);
        assert_eq!(float.ty, glow::FLOAT);
        assert!(!float.normalized);

        assert_eq!(
            map_vertex_format(HalVertexFormat::Float32x2)
                .expect("Float32x2 supported")
                .components,
            2
        );
        assert_eq!(
            map_vertex_format(HalVertexFormat::Float32x3)
                .expect("Float32x3 supported")
                .components,
            3
        );
        assert_eq!(
            map_vertex_format(HalVertexFormat::Float32x4)
                .expect("Float32x4 supported")
                .components,
            4
        );

        let error = map_vertex_format(HalVertexFormat::Unsupported)
            .expect_err("unsupported vertex format must error");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "Unsupported vertex format requested",
            }
        ));
    }

    #[test]
    fn map_primitive_topology_table() {
        assert_eq!(
            map_primitive_topology(HalPrimitiveTopology::PointList),
            glow::POINTS
        );
        assert_eq!(
            map_primitive_topology(HalPrimitiveTopology::LineList),
            glow::LINES
        );
        assert_eq!(
            map_primitive_topology(HalPrimitiveTopology::LineStrip),
            glow::LINE_STRIP
        );
        assert_eq!(
            map_primitive_topology(HalPrimitiveTopology::TriangleList),
            glow::TRIANGLES
        );
        assert_eq!(
            map_primitive_topology(HalPrimitiveTopology::TriangleStrip),
            glow::TRIANGLE_STRIP
        );
    }
}
