/// Enumerates HAL vertex format values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalVertexFormat {
    /// Uint8 variant.
    Uint8,
    /// Uint8x2 variant.
    Uint8x2,
    /// Uint8x4 variant.
    Uint8x4,
    /// Sint8 variant.
    Sint8,
    /// Sint8x2 variant.
    Sint8x2,
    /// Sint8x4 variant.
    Sint8x4,
    /// Unorm8 variant.
    Unorm8,
    /// Unorm8x2 variant.
    Unorm8x2,
    /// Unorm8x4 variant.
    Unorm8x4,
    /// Snorm8 variant.
    Snorm8,
    /// Snorm8x2 variant.
    Snorm8x2,
    /// Snorm8x4 variant.
    Snorm8x4,
    /// Uint16 variant.
    Uint16,
    /// Uint16x2 variant.
    Uint16x2,
    /// Uint16x4 variant.
    Uint16x4,
    /// Sint16 variant.
    Sint16,
    /// Sint16x2 variant.
    Sint16x2,
    /// Sint16x4 variant.
    Sint16x4,
    /// Unorm16 variant.
    Unorm16,
    /// Unorm16x2 variant.
    Unorm16x2,
    /// Unorm16x4 variant.
    Unorm16x4,
    /// Snorm16 variant.
    Snorm16,
    /// Snorm16x2 variant.
    Snorm16x2,
    /// Snorm16x4 variant.
    Snorm16x4,
    /// Float16 variant.
    Float16,
    /// Float16x2 variant.
    Float16x2,
    /// Float16x4 variant.
    Float16x4,
    /// Float32 variant.
    Float32,
    /// Float32x2 variant.
    Float32x2,
    /// Float32x3 variant.
    Float32x3,
    /// Float32x4 variant.
    Float32x4,
    /// Uint32 variant.
    Uint32,
    /// Uint32x2 variant.
    Uint32x2,
    /// Uint32x3 variant.
    Uint32x3,
    /// Uint32x4 variant.
    Uint32x4,
    /// Sint32 variant.
    Sint32,
    /// Sint32x2 variant.
    Sint32x2,
    /// Sint32x3 variant.
    Sint32x3,
    /// Sint32x4 variant.
    Sint32x4,
    /// Unorm10_10_10_2 variant.
    Unorm10_10_10_2,
    /// Unorm8x4 BGRA variant.
    Unorm8x4Bgra,
    /// Unsupported variant.
    Unsupported,
}

/// Enumerates HAL vertex step mode values.
#[derive(Debug, Clone, Copy)]
pub enum HalVertexStepMode {
    /// Vertex variant.
    Vertex,
    /// Instance variant.
    Instance,
}

/// Enumerates HAL primitive topology values.
#[derive(Debug, Clone, Copy)]
pub enum HalPrimitiveTopology {
    /// Point list variant.
    PointList,
    /// Line list variant.
    LineList,
    /// Line strip variant.
    LineStrip,
    /// Triangle list variant.
    TriangleList,
    /// Triangle strip variant.
    TriangleStrip,
}

/// Enumerates HAL texture format values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HalTextureFormat {
    /// R8 unorm variant.
    R8Unorm,
    /// R8 snorm variant.
    R8Snorm,
    /// R8 uint variant.
    R8Uint,
    /// R8 sint variant.
    R8Sint,
    /// R16 unorm variant.
    R16Unorm,
    /// R16 snorm variant.
    R16Snorm,
    /// R16 uint variant.
    R16Uint,
    /// R16 sint variant.
    R16Sint,
    /// R16 float variant.
    R16Float,
    /// Rg8 unorm variant.
    Rg8Unorm,
    /// Rg8 snorm variant.
    Rg8Snorm,
    /// Rg8 uint variant.
    Rg8Uint,
    /// Rg8 sint variant.
    Rg8Sint,
    /// Rg16 unorm variant.
    Rg16Unorm,
    /// Rg16 snorm variant.
    Rg16Snorm,
    /// Rg16 uint variant.
    Rg16Uint,
    /// Rg16 sint variant.
    Rg16Sint,
    /// Rg16 float variant.
    Rg16Float,
    /// R32 uint variant.
    R32Uint,
    /// R32 sint variant.
    R32Sint,
    /// R32 float variant.
    R32Float,
    /// Rg32 uint variant.
    Rg32Uint,
    /// Rg32 sint variant.
    Rg32Sint,
    /// Rg32 float variant.
    Rg32Float,
    /// Rgba8 unorm variant.
    Rgba8Unorm,
    /// Rgba8 unorm srgb variant.
    Rgba8UnormSrgb,
    /// Rgba8 snorm variant.
    Rgba8Snorm,
    /// Rgba8 uint variant.
    Rgba8Uint,
    /// Rgba8 sint variant.
    Rgba8Sint,
    /// Bgra8 unorm variant.
    Bgra8Unorm,
    /// Bgra8 unorm srgb variant.
    Bgra8UnormSrgb,
    /// Rgb10a2 uint variant.
    Rgb10a2Uint,
    /// Rgb10a2 unorm variant.
    Rgb10a2Unorm,
    /// Rg11b10 ufloat variant.
    Rg11b10Ufloat,
    /// Rgb9e5 ufloat variant.
    Rgb9e5Ufloat,
    /// Rgba16 unorm variant.
    Rgba16Unorm,
    /// Rgba16 snorm variant.
    Rgba16Snorm,
    /// Rgba16 uint variant.
    Rgba16Uint,
    /// Rgba16 sint variant.
    Rgba16Sint,
    /// Rgba16 float variant.
    Rgba16Float,
    /// Rgba32 uint variant.
    Rgba32Uint,
    /// Rgba32 sint variant.
    Rgba32Sint,
    /// Rgba32 float variant.
    Rgba32Float,
    /// Stencil8 variant.
    Stencil8,
    /// Depth16 unorm variant.
    Depth16Unorm,
    /// Depth24 plus variant.
    Depth24Plus,
    /// Depth24 plus stencil8 variant.
    Depth24PlusStencil8,
    /// Depth32 float variant.
    Depth32Float,
    /// Depth32 float stencil8 variant.
    Depth32FloatStencil8,
    /// BC1 RGBA unorm block-compressed variant.
    Bc1RgbaUnorm,
    /// BC1 RGBA unorm srgb block-compressed variant.
    Bc1RgbaUnormSrgb,
    /// BC2 RGBA unorm block-compressed variant.
    Bc2RgbaUnorm,
    /// BC2 RGBA unorm srgb block-compressed variant.
    Bc2RgbaUnormSrgb,
    /// BC3 RGBA unorm block-compressed variant.
    Bc3RgbaUnorm,
    /// BC3 RGBA unorm srgb block-compressed variant.
    Bc3RgbaUnormSrgb,
    /// BC4 R unorm block-compressed variant.
    Bc4RUnorm,
    /// BC4 R snorm block-compressed variant.
    Bc4RSnorm,
    /// BC5 RG unorm block-compressed variant.
    Bc5RgUnorm,
    /// BC5 RG snorm block-compressed variant.
    Bc5RgSnorm,
    /// BC6H RGB ufloat block-compressed variant.
    Bc6hRgbUfloat,
    /// BC6H RGB float block-compressed variant.
    Bc6hRgbFloat,
    /// BC7 RGBA unorm block-compressed variant.
    Bc7RgbaUnorm,
    /// BC7 RGBA unorm srgb block-compressed variant.
    Bc7RgbaUnormSrgb,
    /// ETC2 RGB8 unorm block-compressed variant.
    Etc2Rgb8Unorm,
    /// ETC2 RGB8 unorm srgb block-compressed variant.
    Etc2Rgb8UnormSrgb,
    /// ETC2 RGB8A1 unorm block-compressed variant.
    Etc2Rgb8a1Unorm,
    /// ETC2 RGB8A1 unorm srgb block-compressed variant.
    Etc2Rgb8a1UnormSrgb,
    /// ETC2 RGBA8 unorm block-compressed variant.
    Etc2Rgba8Unorm,
    /// ETC2 RGBA8 unorm srgb block-compressed variant.
    Etc2Rgba8UnormSrgb,
    /// EAC R11 unorm block-compressed variant.
    EacR11Unorm,
    /// EAC R11 snorm block-compressed variant.
    EacR11Snorm,
    /// EAC RG11 unorm block-compressed variant.
    EacRg11Unorm,
    /// EAC RG11 snorm block-compressed variant.
    EacRg11Snorm,
    /// ASTC 4x4 unorm block-compressed variant.
    Astc4x4Unorm,
    /// ASTC 4x4 unorm srgb block-compressed variant.
    Astc4x4UnormSrgb,
    /// ASTC 5x4 unorm block-compressed variant.
    Astc5x4Unorm,
    /// ASTC 5x4 unorm srgb block-compressed variant.
    Astc5x4UnormSrgb,
    /// ASTC 5x5 unorm block-compressed variant.
    Astc5x5Unorm,
    /// ASTC 5x5 unorm srgb block-compressed variant.
    Astc5x5UnormSrgb,
    /// ASTC 6x5 unorm block-compressed variant.
    Astc6x5Unorm,
    /// ASTC 6x5 unorm srgb block-compressed variant.
    Astc6x5UnormSrgb,
    /// ASTC 6x6 unorm block-compressed variant.
    Astc6x6Unorm,
    /// ASTC 6x6 unorm srgb block-compressed variant.
    Astc6x6UnormSrgb,
    /// ASTC 8x5 unorm block-compressed variant.
    Astc8x5Unorm,
    /// ASTC 8x5 unorm srgb block-compressed variant.
    Astc8x5UnormSrgb,
    /// ASTC 8x6 unorm block-compressed variant.
    Astc8x6Unorm,
    /// ASTC 8x6 unorm srgb block-compressed variant.
    Astc8x6UnormSrgb,
    /// ASTC 8x8 unorm block-compressed variant.
    Astc8x8Unorm,
    /// ASTC 8x8 unorm srgb block-compressed variant.
    Astc8x8UnormSrgb,
    /// ASTC 10x5 unorm block-compressed variant.
    Astc10x5Unorm,
    /// ASTC 10x5 unorm srgb block-compressed variant.
    Astc10x5UnormSrgb,
    /// ASTC 10x6 unorm block-compressed variant.
    Astc10x6Unorm,
    /// ASTC 10x6 unorm srgb block-compressed variant.
    Astc10x6UnormSrgb,
    /// ASTC 10x8 unorm block-compressed variant.
    Astc10x8Unorm,
    /// ASTC 10x8 unorm srgb block-compressed variant.
    Astc10x8UnormSrgb,
    /// ASTC 10x10 unorm block-compressed variant.
    Astc10x10Unorm,
    /// ASTC 10x10 unorm srgb block-compressed variant.
    Astc10x10UnormSrgb,
    /// ASTC 12x10 unorm block-compressed variant.
    Astc12x10Unorm,
    /// ASTC 12x10 unorm srgb block-compressed variant.
    Astc12x10UnormSrgb,
    /// ASTC 12x12 unorm block-compressed variant.
    Astc12x12Unorm,
    /// ASTC 12x12 unorm srgb block-compressed variant.
    Astc12x12UnormSrgb,
    /// Unsupported variant.
    Unsupported,
}

/// Enumerates the clear-value numeric class for color attachments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalColorClearKind {
    /// Floating-point or normalized color clear.
    Float,
    /// Unsigned integer color clear.
    Uint,
    /// Signed integer color clear.
    Sint,
}

impl HalTextureFormat {
    /// Returns the color clear-value numeric class for this texture format.
    #[must_use]
    pub fn color_clear_kind(self) -> HalColorClearKind {
        match self {
            Self::R8Uint
            | Self::R16Uint
            | Self::Rg8Uint
            | Self::Rg16Uint
            | Self::R32Uint
            | Self::Rg32Uint
            | Self::Rgba8Uint
            | Self::Rgb10a2Uint
            | Self::Rgba16Uint
            | Self::Rgba32Uint => HalColorClearKind::Uint,
            Self::R8Sint
            | Self::R16Sint
            | Self::Rg8Sint
            | Self::Rg16Sint
            | Self::R32Sint
            | Self::Rg32Sint
            | Self::Rgba8Sint
            | Self::Rgba16Sint
            | Self::Rgba32Sint => HalColorClearKind::Sint,
            _ => HalColorClearKind::Float,
        }
    }

    /// Returns compressed texture block information as `(bytes, width, height)`.
    #[must_use]
    pub fn compressed_block_info(self) -> Option<(u32, u32, u32)> {
        let info = match self {
            Self::Bc1RgbaUnorm
            | Self::Bc1RgbaUnormSrgb
            | Self::Bc4RUnorm
            | Self::Bc4RSnorm
            | Self::Etc2Rgb8Unorm
            | Self::Etc2Rgb8UnormSrgb
            | Self::Etc2Rgb8a1Unorm
            | Self::Etc2Rgb8a1UnormSrgb
            | Self::EacR11Unorm
            | Self::EacR11Snorm => (8, 4, 4),
            Self::Bc2RgbaUnorm
            | Self::Bc2RgbaUnormSrgb
            | Self::Bc3RgbaUnorm
            | Self::Bc3RgbaUnormSrgb
            | Self::Bc5RgUnorm
            | Self::Bc5RgSnorm
            | Self::Bc6hRgbUfloat
            | Self::Bc6hRgbFloat
            | Self::Bc7RgbaUnorm
            | Self::Bc7RgbaUnormSrgb
            | Self::Etc2Rgba8Unorm
            | Self::Etc2Rgba8UnormSrgb
            | Self::EacRg11Unorm
            | Self::EacRg11Snorm
            | Self::Astc4x4Unorm
            | Self::Astc4x4UnormSrgb => (16, 4, 4),
            Self::Astc5x4Unorm | Self::Astc5x4UnormSrgb => (16, 5, 4),
            Self::Astc5x5Unorm | Self::Astc5x5UnormSrgb => (16, 5, 5),
            Self::Astc6x5Unorm | Self::Astc6x5UnormSrgb => (16, 6, 5),
            Self::Astc6x6Unorm | Self::Astc6x6UnormSrgb => (16, 6, 6),
            Self::Astc8x5Unorm | Self::Astc8x5UnormSrgb => (16, 8, 5),
            Self::Astc8x6Unorm | Self::Astc8x6UnormSrgb => (16, 8, 6),
            Self::Astc8x8Unorm | Self::Astc8x8UnormSrgb => (16, 8, 8),
            Self::Astc10x5Unorm | Self::Astc10x5UnormSrgb => (16, 10, 5),
            Self::Astc10x6Unorm | Self::Astc10x6UnormSrgb => (16, 10, 6),
            Self::Astc10x8Unorm | Self::Astc10x8UnormSrgb => (16, 10, 8),
            Self::Astc10x10Unorm | Self::Astc10x10UnormSrgb => (16, 10, 10),
            Self::Astc12x10Unorm | Self::Astc12x10UnormSrgb => (16, 12, 10),
            Self::Astc12x12Unorm | Self::Astc12x12UnormSrgb => (16, 12, 12),
            _ => return None,
        };
        Some(info)
    }

    /// Returns the texture block byte size for copies and allocation metadata.
    #[must_use]
    pub fn texel_block_size(self, uncompressed_bytes_per_pixel: u32) -> u32 {
        self.compressed_block_info()
            .map_or(uncompressed_bytes_per_pixel, |(bytes, _, _)| bytes)
    }
}

/// Enumerates HAL texture usage.
#[derive(Debug, Clone, Copy)]
pub struct HalTextureUsage {
    /// Copy src.
    pub copy_src: bool,
    /// Copy dst.
    pub copy_dst: bool,
    /// Texture binding.
    pub texture_binding: bool,
    /// Storage binding.
    pub storage_binding: bool,
    /// Render attachment.
    pub render_attachment: bool,
    /// Transient attachment.
    pub transient: bool,
}

/// Enumerates HAL buffer usage.
#[derive(Debug, Clone, Copy, Default)]
pub struct HalBufferUsage {
    /// MAP_READ.
    pub map_read: bool,
    /// MAP_WRITE.
    pub map_write: bool,
    /// Copy src.
    pub copy_src: bool,
    /// Copy dst.
    pub copy_dst: bool,
    /// Index buffer.
    pub index: bool,
    /// Vertex buffer.
    pub vertex: bool,
    /// Uniform buffer.
    pub uniform: bool,
    /// Storage buffer.
    pub storage: bool,
    /// Indirect buffer.
    pub indirect: bool,
    /// Query resolve destination.
    pub query_resolve: bool,
}

/// Enumerates HAL address mode values.
#[derive(Debug, Clone, Copy)]
pub enum HalAddressMode {
    /// Clamp to edge variant.
    ClampToEdge,
    /// Repeat variant.
    Repeat,
    /// Mirror repeat variant.
    MirrorRepeat,
}

/// Enumerates HAL filter mode values.
#[derive(Debug, Clone, Copy)]
pub enum HalFilterMode {
    /// Nearest variant.
    Nearest,
    /// Linear variant.
    Linear,
}

/// Enumerates HAL mipmap filter mode values.
#[derive(Debug, Clone, Copy)]
pub enum HalMipmapFilterMode {
    /// Nearest variant.
    Nearest,
    /// Linear variant.
    Linear,
}

/// Enumerates HAL compare function values.
#[derive(Debug, Clone, Copy)]
pub enum HalCompareFunction {
    /// Never variant.
    Never,
    /// Less variant.
    Less,
    /// Equal variant.
    Equal,
    /// Less equal variant.
    LessEqual,
    /// Greater variant.
    Greater,
    /// Not equal variant.
    NotEqual,
    /// Greater equal variant.
    GreaterEqual,
    /// Always variant.
    Always,
}

/// Enumerates HAL stencil operation values.
#[derive(Debug, Clone, Copy)]
pub enum HalStencilOperation {
    /// Keep variant.
    Keep,
    /// Zero variant.
    Zero,
    /// Replace variant.
    Replace,
    /// Invert variant.
    Invert,
    /// Increment clamp variant.
    IncrementClamp,
    /// Decrement clamp variant.
    DecrementClamp,
    /// Increment wrap variant.
    IncrementWrap,
    /// Decrement wrap variant.
    DecrementWrap,
}

#[allow(dead_code)]
pub(crate) fn format_has_depth_aspect(format: HalTextureFormat) -> bool {
    matches!(
        format,
        HalTextureFormat::Depth16Unorm
            | HalTextureFormat::Depth24Plus
            | HalTextureFormat::Depth24PlusStencil8
            | HalTextureFormat::Depth32Float
            | HalTextureFormat::Depth32FloatStencil8
    )
}

#[allow(dead_code)]
pub(crate) fn format_has_stencil_aspect(format: HalTextureFormat) -> bool {
    matches!(
        format,
        HalTextureFormat::Stencil8
            | HalTextureFormat::Depth24PlusStencil8
            | HalTextureFormat::Depth32FloatStencil8
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNCOMPRESSED_COLOR_FORMATS: &[HalTextureFormat] = &[
        HalTextureFormat::R8Unorm,
        HalTextureFormat::R8Snorm,
        HalTextureFormat::R8Uint,
        HalTextureFormat::R8Sint,
        HalTextureFormat::R16Unorm,
        HalTextureFormat::R16Snorm,
        HalTextureFormat::R16Uint,
        HalTextureFormat::R16Sint,
        HalTextureFormat::R16Float,
        HalTextureFormat::Rg8Unorm,
        HalTextureFormat::Rg8Snorm,
        HalTextureFormat::Rg8Uint,
        HalTextureFormat::Rg8Sint,
        HalTextureFormat::Rg16Unorm,
        HalTextureFormat::Rg16Snorm,
        HalTextureFormat::Rg16Uint,
        HalTextureFormat::Rg16Sint,
        HalTextureFormat::Rg16Float,
        HalTextureFormat::R32Uint,
        HalTextureFormat::R32Sint,
        HalTextureFormat::R32Float,
        HalTextureFormat::Rg32Uint,
        HalTextureFormat::Rg32Sint,
        HalTextureFormat::Rg32Float,
        HalTextureFormat::Rgba8Unorm,
        HalTextureFormat::Rgba8UnormSrgb,
        HalTextureFormat::Rgba8Snorm,
        HalTextureFormat::Rgba8Uint,
        HalTextureFormat::Rgba8Sint,
        HalTextureFormat::Bgra8Unorm,
        HalTextureFormat::Bgra8UnormSrgb,
        HalTextureFormat::Rgb10a2Uint,
        HalTextureFormat::Rgb10a2Unorm,
        HalTextureFormat::Rg11b10Ufloat,
        HalTextureFormat::Rgb9e5Ufloat,
        HalTextureFormat::Rgba16Unorm,
        HalTextureFormat::Rgba16Snorm,
        HalTextureFormat::Rgba16Uint,
        HalTextureFormat::Rgba16Sint,
        HalTextureFormat::Rgba16Float,
        HalTextureFormat::Rgba32Uint,
        HalTextureFormat::Rgba32Sint,
        HalTextureFormat::Rgba32Float,
    ];

    #[test]
    fn hal_buffer_usage_default_is_all_false() {
        let usage = HalBufferUsage::default();

        assert!(!usage.map_read);
        assert!(!usage.map_write);
        assert!(!usage.copy_src);
        assert!(!usage.copy_dst);
        assert!(!usage.index);
        assert!(!usage.vertex);
        assert!(!usage.uniform);
        assert!(!usage.storage);
        assert!(!usage.indirect);
        assert!(!usage.query_resolve);
    }

    #[test]
    fn color_clear_kind_classifies_integer_formats() {
        assert_eq!(
            HalTextureFormat::R32Uint.color_clear_kind(),
            HalColorClearKind::Uint
        );
        assert_eq!(
            HalTextureFormat::Rgba8Uint.color_clear_kind(),
            HalColorClearKind::Uint
        );
        assert_eq!(
            HalTextureFormat::R32Sint.color_clear_kind(),
            HalColorClearKind::Sint
        );
        assert_eq!(
            HalTextureFormat::Rgba16Sint.color_clear_kind(),
            HalColorClearKind::Sint
        );
        assert_eq!(
            HalTextureFormat::Rgba8Unorm.color_clear_kind(),
            HalColorClearKind::Float
        );
        assert_eq!(
            HalTextureFormat::Rgba32Float.color_clear_kind(),
            HalColorClearKind::Float
        );
    }

    #[test]
    fn hal_stencil_operation_variants_are_constructible() {
        let operations = [
            HalStencilOperation::Keep,
            HalStencilOperation::Zero,
            HalStencilOperation::Replace,
            HalStencilOperation::Invert,
            HalStencilOperation::IncrementClamp,
            HalStencilOperation::DecrementClamp,
            HalStencilOperation::IncrementWrap,
            HalStencilOperation::DecrementWrap,
        ];

        assert!(matches!(operations[0], HalStencilOperation::Keep));
        assert!(matches!(operations[1], HalStencilOperation::Zero));
        assert!(matches!(operations[2], HalStencilOperation::Replace));
        assert!(matches!(operations[3], HalStencilOperation::Invert));
        assert!(matches!(operations[4], HalStencilOperation::IncrementClamp));
        assert!(matches!(operations[5], HalStencilOperation::DecrementClamp));
        assert!(matches!(operations[6], HalStencilOperation::IncrementWrap));
        assert!(matches!(operations[7], HalStencilOperation::DecrementWrap));
    }

    #[test]
    fn format_has_depth_aspect_covers_relevant_formats() {
        assert!(format_has_depth_aspect(HalTextureFormat::Depth16Unorm));
        assert!(format_has_depth_aspect(HalTextureFormat::Depth24Plus));
        assert!(format_has_depth_aspect(
            HalTextureFormat::Depth24PlusStencil8
        ));
        assert!(format_has_depth_aspect(HalTextureFormat::Depth32Float));
        assert!(format_has_depth_aspect(
            HalTextureFormat::Depth32FloatStencil8
        ));
        assert!(!format_has_depth_aspect(HalTextureFormat::Stencil8));
        for format in UNCOMPRESSED_COLOR_FORMATS {
            assert!(!format_has_depth_aspect(*format), "{format:?}");
        }
    }

    #[test]
    fn format_has_stencil_aspect_covers_relevant_formats() {
        assert!(format_has_stencil_aspect(HalTextureFormat::Stencil8));
        assert!(format_has_stencil_aspect(
            HalTextureFormat::Depth24PlusStencil8
        ));
        assert!(format_has_stencil_aspect(
            HalTextureFormat::Depth32FloatStencil8
        ));
        assert!(!format_has_stencil_aspect(HalTextureFormat::Depth16Unorm));
        assert!(!format_has_stencil_aspect(HalTextureFormat::Depth24Plus));
        assert!(!format_has_stencil_aspect(HalTextureFormat::Depth32Float));
        for format in UNCOMPRESSED_COLOR_FORMATS {
            assert!(!format_has_stencil_aspect(*format), "{format:?}");
        }
    }
}
