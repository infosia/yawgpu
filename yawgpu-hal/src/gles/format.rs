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
    pub(super) integer: bool,
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

/// Returns whether `format` is color-renderable on core GLES 3.1 (usable as a
/// render-pipeline color target / FBO color attachment without extensions).
///
/// This covers the GLES 3.1 core color-renderable internal formats that have
/// WebGPU equivalents: the 8-bit unorm family (R8, RG8, RGBA8, SRGB8_ALPHA8;
/// `Bgra8Unorm` rides the existing RGBA8-internal special case in
/// `map_texture_format`), RGB10_A2, and every integer (uint/sint) format
/// including RGB10_A2UI. Float formats (`R16Float`/`R32Float`/`Rg11b10Ufloat`/
/// `Rgba16Float`/`Rgba32Float`, ...) are deliberately excluded: core GLES 3.1
/// does not make them color-renderable — that requires
/// `EXT_color_buffer_float` (or `EXT_color_buffer_half_float`) gating, which
/// [`is_color_renderable_with`] layers on top of this core set (T-G12).
pub(super) fn is_color_renderable(format: HalTextureFormat) -> bool {
    matches!(
        format,
        // 8-bit unorm family (RGBA8-class internal formats).
        HalTextureFormat::R8Unorm
            | HalTextureFormat::Rg8Unorm
            | HalTextureFormat::Rgba8Unorm
            | HalTextureFormat::Rgba8UnormSrgb
            | HalTextureFormat::Bgra8Unorm
            // Packed 10-10-10-2 unorm.
            | HalTextureFormat::Rgb10a2Unorm
            // Unsigned/signed integer formats (all core color-renderable).
            | HalTextureFormat::R8Uint
            | HalTextureFormat::R8Sint
            | HalTextureFormat::R16Uint
            | HalTextureFormat::R16Sint
            | HalTextureFormat::R32Uint
            | HalTextureFormat::R32Sint
            | HalTextureFormat::Rg8Uint
            | HalTextureFormat::Rg8Sint
            | HalTextureFormat::Rg16Uint
            | HalTextureFormat::Rg16Sint
            | HalTextureFormat::Rg32Uint
            | HalTextureFormat::Rg32Sint
            | HalTextureFormat::Rgba8Uint
            | HalTextureFormat::Rgba8Sint
            | HalTextureFormat::Rgba16Uint
            | HalTextureFormat::Rgba16Sint
            | HalTextureFormat::Rgba32Uint
            | HalTextureFormat::Rgba32Sint
            | HalTextureFormat::Rgb10a2Uint
    )
}

/// Extension-gated color-renderability capabilities detected once at device
/// creation (T-G12), widening the core GLES 3.1 color-renderable set covered
/// by [`is_color_renderable`]. Neither extension is core in any GLES version
/// (3.2 included), so both stay pure extension-string checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct GlesColorRenderCaps {
    /// `GL_EXT_color_buffer_float` is advertised: makes R16F/RG16F/RGBA16F,
    /// R32F/RG32F/RGBA32F, and R11F_G11F_B10F color-renderable.
    pub(super) color_buffer_float: bool,
    /// `GL_EXT_color_buffer_half_float` is advertised: makes only the 16-bit
    /// float formats (R16F/RG16F/RGBA16F) color-renderable; the 32-bit float
    /// formats and R11F_G11F_B10F still require `GL_EXT_color_buffer_float`.
    pub(super) color_buffer_half_float: bool,
}

/// Returns whether `format` is color-renderable on a device with the given
/// extension caps (T-G12): the core GLES 3.1 set ([`is_color_renderable`]),
/// plus the float16 formats when either `EXT_color_buffer_float` or
/// `EXT_color_buffer_half_float` is present, plus the float32 formats and
/// `Rg11b10Ufloat` when `EXT_color_buffer_float` is present. With both caps
/// `false` this is identical to [`is_color_renderable`].
pub(super) fn is_color_renderable_with(
    format: HalTextureFormat,
    caps: GlesColorRenderCaps,
) -> bool {
    if is_color_renderable(format) {
        return true;
    }
    match format {
        // 16-bit float: renderable under either extension.
        HalTextureFormat::R16Float
        | HalTextureFormat::Rg16Float
        | HalTextureFormat::Rgba16Float => caps.color_buffer_float || caps.color_buffer_half_float,
        // 32-bit float and packed 11-11-10 float: EXT_color_buffer_float only.
        HalTextureFormat::R32Float
        | HalTextureFormat::Rg32Float
        | HalTextureFormat::Rgba32Float
        | HalTextureFormat::Rg11b10Ufloat => caps.color_buffer_float,
        _ => false,
    }
}

/// Component class a GLES color clear (and any other component-class-sensitive
/// path) must dispatch on for a given color-attachment format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GlesClearKind {
    /// Float / normalized components: `glClearColor` + `glClear` semantics.
    Float,
    /// Unsigned-integer components: `glClearBufferuiv` semantics.
    Uint,
    /// Signed-integer components: `glClearBufferiv` semantics.
    Sint,
}

/// Classifies a color format's component class from its GL upload triplet:
/// a `*_INTEGER` external format marks an integer attachment, and the
/// signedness of its `ty` picks `Sint` vs `Uint`. Formats that do not map on
/// GLES (or are non-integer) classify as `Float`, preserving the existing
/// `glClearColor` path.
pub(super) fn color_clear_kind(format: HalTextureFormat) -> GlesClearKind {
    let Ok(gles) = map_texture_format(format) else {
        return GlesClearKind::Float;
    };
    if !matches!(
        gles.format,
        glow::RED_INTEGER | glow::RG_INTEGER | glow::RGBA_INTEGER
    ) {
        return GlesClearKind::Float;
    }
    match gles.ty {
        glow::BYTE | glow::SHORT | glow::INT => GlesClearKind::Sint,
        _ => GlesClearKind::Uint,
    }
}

pub(super) fn map_vertex_format(format: HalVertexFormat) -> Result<GlesVertexFormat, HalError> {
    match format {
        HalVertexFormat::Uint8 => Ok(gles_vertex_format(1, glow::UNSIGNED_BYTE, false, true)),
        HalVertexFormat::Uint8x2 => Ok(gles_vertex_format(2, glow::UNSIGNED_BYTE, false, true)),
        HalVertexFormat::Uint8x4 => Ok(gles_vertex_format(4, glow::UNSIGNED_BYTE, false, true)),
        HalVertexFormat::Sint8 => Ok(gles_vertex_format(1, glow::BYTE, false, true)),
        HalVertexFormat::Sint8x2 => Ok(gles_vertex_format(2, glow::BYTE, false, true)),
        HalVertexFormat::Sint8x4 => Ok(gles_vertex_format(4, glow::BYTE, false, true)),
        HalVertexFormat::Unorm8 => Ok(gles_vertex_format(1, glow::UNSIGNED_BYTE, true, false)),
        HalVertexFormat::Unorm8x2 => Ok(gles_vertex_format(2, glow::UNSIGNED_BYTE, true, false)),
        HalVertexFormat::Unorm8x4 => Ok(gles_vertex_format(4, glow::UNSIGNED_BYTE, true, false)),
        HalVertexFormat::Snorm8 => Ok(gles_vertex_format(1, glow::BYTE, true, false)),
        HalVertexFormat::Snorm8x2 => Ok(gles_vertex_format(2, glow::BYTE, true, false)),
        HalVertexFormat::Snorm8x4 => Ok(gles_vertex_format(4, glow::BYTE, true, false)),
        HalVertexFormat::Uint16 => Ok(gles_vertex_format(1, glow::UNSIGNED_SHORT, false, true)),
        HalVertexFormat::Uint16x2 => Ok(gles_vertex_format(2, glow::UNSIGNED_SHORT, false, true)),
        HalVertexFormat::Uint16x4 => Ok(gles_vertex_format(4, glow::UNSIGNED_SHORT, false, true)),
        HalVertexFormat::Sint16 => Ok(gles_vertex_format(1, glow::SHORT, false, true)),
        HalVertexFormat::Sint16x2 => Ok(gles_vertex_format(2, glow::SHORT, false, true)),
        HalVertexFormat::Sint16x4 => Ok(gles_vertex_format(4, glow::SHORT, false, true)),
        HalVertexFormat::Unorm16 => Ok(gles_vertex_format(1, glow::UNSIGNED_SHORT, true, false)),
        HalVertexFormat::Unorm16x2 => Ok(gles_vertex_format(2, glow::UNSIGNED_SHORT, true, false)),
        HalVertexFormat::Unorm16x4 => Ok(gles_vertex_format(4, glow::UNSIGNED_SHORT, true, false)),
        HalVertexFormat::Snorm16 => Ok(gles_vertex_format(1, glow::SHORT, true, false)),
        HalVertexFormat::Snorm16x2 => Ok(gles_vertex_format(2, glow::SHORT, true, false)),
        HalVertexFormat::Snorm16x4 => Ok(gles_vertex_format(4, glow::SHORT, true, false)),
        HalVertexFormat::Float16 => Ok(gles_vertex_format(1, glow::HALF_FLOAT, false, false)),
        HalVertexFormat::Float16x2 => Ok(gles_vertex_format(2, glow::HALF_FLOAT, false, false)),
        HalVertexFormat::Float16x4 => Ok(gles_vertex_format(4, glow::HALF_FLOAT, false, false)),
        HalVertexFormat::Float32 => Ok(gles_vertex_format(1, glow::FLOAT, false, false)),
        HalVertexFormat::Float32x2 => Ok(gles_vertex_format(2, glow::FLOAT, false, false)),
        HalVertexFormat::Float32x3 => Ok(gles_vertex_format(3, glow::FLOAT, false, false)),
        HalVertexFormat::Float32x4 => Ok(gles_vertex_format(4, glow::FLOAT, false, false)),
        HalVertexFormat::Uint32 => Ok(gles_vertex_format(1, glow::UNSIGNED_INT, false, true)),
        HalVertexFormat::Uint32x2 => Ok(gles_vertex_format(2, glow::UNSIGNED_INT, false, true)),
        HalVertexFormat::Uint32x3 => Ok(gles_vertex_format(3, glow::UNSIGNED_INT, false, true)),
        HalVertexFormat::Uint32x4 => Ok(gles_vertex_format(4, glow::UNSIGNED_INT, false, true)),
        HalVertexFormat::Sint32 => Ok(gles_vertex_format(1, glow::INT, false, true)),
        HalVertexFormat::Sint32x2 => Ok(gles_vertex_format(2, glow::INT, false, true)),
        HalVertexFormat::Sint32x3 => Ok(gles_vertex_format(3, glow::INT, false, true)),
        HalVertexFormat::Sint32x4 => Ok(gles_vertex_format(4, glow::INT, false, true)),
        HalVertexFormat::Unorm10_10_10_2 => Ok(gles_vertex_format(
            4,
            glow::UNSIGNED_INT_2_10_10_10_REV,
            true,
            false,
        )),
        HalVertexFormat::Unorm8x4Bgra => Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "Unorm8x4Bgra vertex format is not supported on GLES",
        }),
        HalVertexFormat::Unsupported => Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "Unsupported vertex format requested",
        }),
    }
}

fn gles_vertex_format(
    components: i32,
    ty: u32,
    normalized: bool,
    integer: bool,
) -> GlesVertexFormat {
    GlesVertexFormat {
        components,
        ty,
        normalized,
        integer,
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
    fn is_color_renderable_table() {
        // GLES 3.1 core color-renderable formats with WebGPU equivalents.
        let renderable = [
            HalTextureFormat::R8Unorm,
            HalTextureFormat::Rg8Unorm,
            HalTextureFormat::Rgba8Unorm,
            HalTextureFormat::Rgba8UnormSrgb,
            HalTextureFormat::Bgra8Unorm,
            HalTextureFormat::Rgb10a2Unorm,
            HalTextureFormat::R8Uint,
            HalTextureFormat::R8Sint,
            HalTextureFormat::R16Uint,
            HalTextureFormat::R16Sint,
            HalTextureFormat::R32Uint,
            HalTextureFormat::R32Sint,
            HalTextureFormat::Rg8Uint,
            HalTextureFormat::Rg8Sint,
            HalTextureFormat::Rg16Uint,
            HalTextureFormat::Rg16Sint,
            HalTextureFormat::Rg32Uint,
            HalTextureFormat::Rg32Sint,
            HalTextureFormat::Rgba8Uint,
            HalTextureFormat::Rgba8Sint,
            HalTextureFormat::Rgba16Uint,
            HalTextureFormat::Rgba16Sint,
            HalTextureFormat::Rgba32Uint,
            HalTextureFormat::Rgba32Sint,
            HalTextureFormat::Rgb10a2Uint,
        ];
        for format in renderable {
            assert!(is_color_renderable(format), "{format:?}");
        }

        // Snorm, float (EXT_color_buffer_float-gated; see
        // `is_color_renderable_with`), depth, and compressed formats are not
        // color-renderable on core GLES 3.1.
        let not_renderable = [
            HalTextureFormat::R8Snorm,
            HalTextureFormat::Rgba8Snorm,
            HalTextureFormat::R16Float,
            HalTextureFormat::R32Float,
            HalTextureFormat::Rg11b10Ufloat,
            HalTextureFormat::Rgb9e5Ufloat,
            HalTextureFormat::Rgba16Float,
            HalTextureFormat::Rgba32Float,
            HalTextureFormat::Depth32Float,
            HalTextureFormat::Bc1RgbaUnorm,
        ];
        for format in not_renderable {
            assert!(!is_color_renderable(format), "{format:?}");
        }
    }

    #[test]
    fn is_color_renderable_with_gates_float_formats_on_extension_caps() {
        // T-G12: all four cap combinations across representative formats.
        let no_caps = GlesColorRenderCaps::default();
        let float_only = GlesColorRenderCaps {
            color_buffer_float: true,
            color_buffer_half_float: false,
        };
        let half_float_only = GlesColorRenderCaps {
            color_buffer_float: false,
            color_buffer_half_float: true,
        };
        let both = GlesColorRenderCaps {
            color_buffer_float: true,
            color_buffer_half_float: true,
        };

        let float16 = [
            HalTextureFormat::R16Float,
            HalTextureFormat::Rg16Float,
            HalTextureFormat::Rgba16Float,
        ];
        let float32 = [
            HalTextureFormat::R32Float,
            HalTextureFormat::Rg32Float,
            HalTextureFormat::Rgba32Float,
            HalTextureFormat::Rg11b10Ufloat,
        ];
        // Core-set representatives stay renderable, and never-renderable
        // formats stay rejected, under every cap combination.
        let core = [
            HalTextureFormat::R8Unorm,
            HalTextureFormat::Rgba8UnormSrgb,
            HalTextureFormat::Bgra8Unorm,
            HalTextureFormat::Rgb10a2Unorm,
            HalTextureFormat::R32Uint,
            HalTextureFormat::Rgba16Sint,
        ];
        let never = [
            HalTextureFormat::R8Snorm,
            HalTextureFormat::Rgba8Snorm,
            HalTextureFormat::Rgb9e5Ufloat,
            HalTextureFormat::Depth32Float,
            HalTextureFormat::Bc1RgbaUnorm,
        ];

        for caps in [no_caps, float_only, half_float_only, both] {
            for format in core {
                assert!(
                    is_color_renderable_with(format, caps),
                    "{format:?} {caps:?}"
                );
            }
            for format in never {
                assert!(
                    !is_color_renderable_with(format, caps),
                    "{format:?} {caps:?}"
                );
            }
            // float16: either extension unlocks renderability.
            for format in float16 {
                assert_eq!(
                    is_color_renderable_with(format, caps),
                    caps.color_buffer_float || caps.color_buffer_half_float,
                    "{format:?} {caps:?}"
                );
            }
            // float32 + Rg11b10Ufloat: EXT_color_buffer_float only.
            for format in float32 {
                assert_eq!(
                    is_color_renderable_with(format, caps),
                    caps.color_buffer_float,
                    "{format:?} {caps:?}"
                );
            }
        }

        // With both caps false the predicate degenerates to the core set.
        for format in float16.iter().chain(&float32).chain(&core).chain(&never) {
            assert_eq!(
                is_color_renderable_with(*format, no_caps),
                is_color_renderable(*format),
                "{format:?}"
            );
        }
    }

    #[test]
    fn color_clear_kind_classifies_component_classes() {
        let uint = [
            HalTextureFormat::R8Uint,
            HalTextureFormat::R16Uint,
            HalTextureFormat::R32Uint,
            HalTextureFormat::Rg8Uint,
            HalTextureFormat::Rg16Uint,
            HalTextureFormat::Rg32Uint,
            HalTextureFormat::Rgba8Uint,
            HalTextureFormat::Rgba16Uint,
            HalTextureFormat::Rgba32Uint,
            HalTextureFormat::Rgb10a2Uint,
        ];
        for format in uint {
            assert_eq!(color_clear_kind(format), GlesClearKind::Uint, "{format:?}");
        }

        let sint = [
            HalTextureFormat::R8Sint,
            HalTextureFormat::R16Sint,
            HalTextureFormat::R32Sint,
            HalTextureFormat::Rg8Sint,
            HalTextureFormat::Rg16Sint,
            HalTextureFormat::Rg32Sint,
            HalTextureFormat::Rgba8Sint,
            HalTextureFormat::Rgba16Sint,
            HalTextureFormat::Rgba32Sint,
        ];
        for format in sint {
            assert_eq!(color_clear_kind(format), GlesClearKind::Sint, "{format:?}");
        }

        let float = [
            HalTextureFormat::Rgba8Unorm,
            HalTextureFormat::Bgra8Unorm,
            HalTextureFormat::Rgb10a2Unorm,
            HalTextureFormat::R16Float,
            HalTextureFormat::Rgba32Float,
            // Unmapped formats fall back to the float clear path.
            HalTextureFormat::Depth32Float,
        ];
        for format in float {
            assert_eq!(color_clear_kind(format), GlesClearKind::Float, "{format:?}");
        }
    }

    #[test]
    fn map_vertex_format_table() {
        let float = map_vertex_format(HalVertexFormat::Float32).expect("Float32 supported");
        assert_eq!(float.components, 1);
        assert_eq!(float.ty, glow::FLOAT);
        assert!(!float.normalized);
        assert!(!float.integer);

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

        let uint = map_vertex_format(HalVertexFormat::Uint8x4).expect("Uint8x4 supported");
        assert_eq!(uint.components, 4);
        assert_eq!(uint.ty, glow::UNSIGNED_BYTE);
        assert!(!uint.normalized);
        assert!(uint.integer);

        let unorm = map_vertex_format(HalVertexFormat::Unorm8x4).expect("Unorm8x4 supported");
        assert_eq!(unorm.components, 4);
        assert_eq!(unorm.ty, glow::UNSIGNED_BYTE);
        assert!(unorm.normalized);
        assert!(!unorm.integer);

        let half = map_vertex_format(HalVertexFormat::Float16x4).expect("Float16x4 supported");
        assert_eq!(half.components, 4);
        assert_eq!(half.ty, glow::HALF_FLOAT);
        assert!(!half.normalized);
        assert!(!half.integer);

        let packed =
            map_vertex_format(HalVertexFormat::Unorm10_10_10_2).expect("Unorm10_10_10_2 supported");
        assert_eq!(packed.components, 4);
        assert_eq!(packed.ty, glow::UNSIGNED_INT_2_10_10_10_REV);
        assert!(packed.normalized);
        assert!(!packed.integer);

        let bgra_error = map_vertex_format(HalVertexFormat::Unorm8x4Bgra)
            .expect_err("BGRA vertex format must error on GLES");
        assert!(matches!(
            bgra_error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "Unorm8x4Bgra vertex format is not supported on GLES",
            }
        ));

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
