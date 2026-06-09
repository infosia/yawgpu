use crate::adapter::Feature;
use crate::device::FeatureSet;

/// Enumerates texture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureFormat(u32);

impl TextureFormat {
    /// Constant value for undefined.
    pub(crate) const UNDEFINED: u32 = 0x00;
    /// Constant value for r8 unorm.
    pub(crate) const R8_UNORM: u32 = 0x01;
    /// Constant value for r8 snorm.
    pub(crate) const R8_SNORM: u32 = 0x02;
    /// Constant value for r8 uint.
    pub(crate) const R8_UINT: u32 = 0x03;
    /// Constant value for r8 sint.
    pub(crate) const R8_SINT: u32 = 0x04;
    /// Constant value for r16 unorm.
    pub(crate) const R16_UNORM: u32 = 0x05;
    /// Constant value for r16 snorm.
    pub(crate) const R16_SNORM: u32 = 0x06;
    /// Constant value for r16 uint.
    pub(crate) const R16_UINT: u32 = 0x07;
    /// Constant value for r16 sint.
    pub(crate) const R16_SINT: u32 = 0x08;
    /// Constant value for r16 float.
    pub(crate) const R16_FLOAT: u32 = 0x09;
    /// Constant value for rg8 unorm.
    pub(crate) const RG8_UNORM: u32 = 0x0A;
    /// Constant value for rg8 snorm.
    pub(crate) const RG8_SNORM: u32 = 0x0B;
    /// Constant value for rg8 uint.
    pub(crate) const RG8_UINT: u32 = 0x0C;
    /// Constant value for rg8 sint.
    pub(crate) const RG8_SINT: u32 = 0x0D;
    /// Constant value for rg16 unorm.
    pub(crate) const RG16_UNORM: u32 = 0x11;
    /// Constant value for rg16 snorm.
    pub(crate) const RG16_SNORM: u32 = 0x12;
    /// Constant value for rg16 uint.
    pub(crate) const RG16_UINT: u32 = 0x13;
    /// Constant value for rg16 sint.
    pub(crate) const RG16_SINT: u32 = 0x14;
    /// Constant value for rg16 float.
    pub(crate) const RG16_FLOAT: u32 = 0x15;
    /// Constant value for r32 float.
    pub(crate) const R32_FLOAT: u32 = 0x0E;
    /// Constant value for r32 uint.
    pub(crate) const R32_UINT: u32 = 0x0F;
    /// Constant value for r32 sint.
    pub(crate) const R32_SINT: u32 = 0x10;
    /// Constant value for rgba8 unorm.
    pub(crate) const RGBA8_UNORM: u32 = 0x16;
    /// Constant value for rgba8 unorm srgb.
    pub(crate) const RGBA8_UNORM_SRGB: u32 = 0x17;
    /// Constant value for rgba8 snorm.
    pub(crate) const RGBA8_SNORM: u32 = 0x18;
    /// Constant value for rgba8 uint.
    pub(crate) const RGBA8_UINT: u32 = 0x19;
    /// Constant value for rgba8 sint.
    pub(crate) const RGBA8_SINT: u32 = 0x1A;
    /// Constant value for rg11 b10 ufloat.
    pub(crate) const RG11B10_UFLOAT: u32 = 0x1F;
    /// Constant value for rgb9 e5 ufloat.
    pub(crate) const RGB9E5_UFLOAT: u32 = 0x20;
    /// Constant value for rg32 float.
    pub(crate) const RG32_FLOAT: u32 = 0x21;
    /// Constant value for rg32 uint.
    pub(crate) const RG32_UINT: u32 = 0x22;
    /// Constant value for rg32 sint.
    pub(crate) const RG32_SINT: u32 = 0x23;
    /// Constant value for rgba16 unorm.
    pub(crate) const RGBA16_UNORM: u32 = 0x24;
    /// Constant value for rgba16 snorm.
    pub(crate) const RGBA16_SNORM: u32 = 0x25;
    /// Constant value for rgba16 uint.
    pub(crate) const RGBA16_UINT: u32 = 0x26;
    /// Constant value for rgba16 sint.
    pub(crate) const RGBA16_SINT: u32 = 0x27;
    /// Constant value for rgba16 float.
    pub(crate) const RGBA16_FLOAT: u32 = 0x28;
    /// Constant value for rgba32 float.
    pub(crate) const RGBA32_FLOAT: u32 = 0x29;
    /// Constant value for rgba32 uint.
    pub(crate) const RGBA32_UINT: u32 = 0x2A;
    /// Constant value for rgba32 sint.
    pub(crate) const RGBA32_SINT: u32 = 0x2B;
    /// Constant value for stencil8.
    pub(crate) const STENCIL8: u32 = 0x2C;
    /// Constant value for depth16 unorm.
    pub(crate) const DEPTH16_UNORM: u32 = 0x2D;
    /// Constant value for depth24 plus.
    pub(crate) const DEPTH24_PLUS: u32 = 0x2E;
    /// Constant value for depth24 plus stencil8.
    pub(crate) const DEPTH24_PLUS_STENCIL8: u32 = 0x2F;
    /// Constant value for depth32 float.
    pub(crate) const DEPTH32_FLOAT: u32 = 0x30;
    /// Constant value for depth32 float stencil8.
    pub(crate) const DEPTH32_FLOAT_STENCIL8: u32 = 0x31;
    /// Constant value for bc1 rgba unorm.
    pub(crate) const BC1_RGBA_UNORM: u32 = 0x32;
    /// Constant value for bc1 rgba unorm srgb.
    pub(crate) const BC1_RGBA_UNORM_SRGB: u32 = 0x33;
    /// Constant value for bc2 rgba unorm.
    pub(crate) const BC2_RGBA_UNORM: u32 = 0x34;
    /// Constant value for bc2 rgba unorm srgb.
    pub(crate) const BC2_RGBA_UNORM_SRGB: u32 = 0x35;
    /// Constant value for bc3 rgba unorm.
    pub(crate) const BC3_RGBA_UNORM: u32 = 0x36;
    /// Constant value for bc3 rgba unorm srgb.
    pub(crate) const BC3_RGBA_UNORM_SRGB: u32 = 0x37;
    /// Constant value for bc4 r unorm.
    pub(crate) const BC4_R_UNORM: u32 = 0x38;
    /// Constant value for bc4 r snorm.
    pub(crate) const BC4_R_SNORM: u32 = 0x39;
    /// Constant value for bc5 rg unorm.
    pub(crate) const BC5_RG_UNORM: u32 = 0x3A;
    /// Constant value for bc5 rg snorm.
    pub(crate) const BC5_RG_SNORM: u32 = 0x3B;
    /// Constant value for bc6h rgb ufloat.
    pub(crate) const BC6H_RGB_UFLOAT: u32 = 0x3C;
    /// Constant value for bc6h rgb float.
    pub(crate) const BC6H_RGB_FLOAT: u32 = 0x3D;
    /// Constant value for bgra8 unorm.
    pub(crate) const BGRA8_UNORM: u32 = 0x1B;
    /// Constant value for bgra8 unorm srgb.
    pub(crate) const BGRA8_UNORM_SRGB: u32 = 0x1C;
    /// Constant value for rgb10 a2 uint.
    pub(crate) const RGB10A2_UINT: u32 = 0x1D;
    /// Constant value for rgb10 a2 unorm.
    pub(crate) const RGB10A2_UNORM: u32 = 0x1E;
    /// Constant value for bc7 rgba unorm.
    pub(crate) const BC7_RGBA_UNORM: u32 = 0x3E;
    /// Constant value for bc7 rgba unorm srgb.
    pub(crate) const BC7_RGBA_UNORM_SRGB: u32 = 0x3F;
    /// Constant value for etc2 rgb8 unorm.
    pub(crate) const ETC2_RGB8_UNORM: u32 = 0x40;
    /// Constant value for etc2 rgb8 unorm srgb.
    pub(crate) const ETC2_RGB8_UNORM_SRGB: u32 = 0x41;
    /// Constant value for etc2 rgb8a1 unorm.
    pub(crate) const ETC2_RGB8A1_UNORM: u32 = 0x42;
    /// Constant value for etc2 rgb8a1 unorm srgb.
    pub(crate) const ETC2_RGB8A1_UNORM_SRGB: u32 = 0x43;
    /// Constant value for etc2 rgba8 unorm.
    pub(crate) const ETC2_RGBA8_UNORM: u32 = 0x44;
    /// Constant value for etc2 rgba8 unorm srgb.
    pub(crate) const ETC2_RGBA8_UNORM_SRGB: u32 = 0x45;
    /// Constant value for eac r11 unorm.
    pub(crate) const EAC_R11_UNORM: u32 = 0x46;
    /// Constant value for eac r11 snorm.
    pub(crate) const EAC_R11_SNORM: u32 = 0x47;
    /// Constant value for eac rg11 unorm.
    pub(crate) const EAC_RG11_UNORM: u32 = 0x48;
    /// Constant value for eac rg11 snorm.
    pub(crate) const EAC_RG11_SNORM: u32 = 0x49;
    /// Constant value for astc4x4 unorm.
    pub(crate) const ASTC4X4_UNORM: u32 = 0x4A;
    /// Constant value for astc4x4 unorm srgb.
    pub(crate) const ASTC4X4_UNORM_SRGB: u32 = 0x4B;
    /// Constant value for astc5x4 unorm.
    pub(crate) const ASTC5X4_UNORM: u32 = 0x4C;
    /// Constant value for astc5x4 unorm srgb.
    pub(crate) const ASTC5X4_UNORM_SRGB: u32 = 0x4D;
    /// Constant value for astc5x5 unorm.
    pub(crate) const ASTC5X5_UNORM: u32 = 0x4E;
    /// Constant value for astc5x5 unorm srgb.
    pub(crate) const ASTC5X5_UNORM_SRGB: u32 = 0x4F;
    /// Constant value for astc6x5 unorm.
    pub(crate) const ASTC6X5_UNORM: u32 = 0x50;
    /// Constant value for astc6x5 unorm srgb.
    pub(crate) const ASTC6X5_UNORM_SRGB: u32 = 0x51;
    /// Constant value for astc6x6 unorm.
    pub(crate) const ASTC6X6_UNORM: u32 = 0x52;
    /// Constant value for astc6x6 unorm srgb.
    pub(crate) const ASTC6X6_UNORM_SRGB: u32 = 0x53;
    /// Constant value for astc8x5 unorm.
    pub(crate) const ASTC8X5_UNORM: u32 = 0x54;
    /// Constant value for astc8x5 unorm srgb.
    pub(crate) const ASTC8X5_UNORM_SRGB: u32 = 0x55;
    /// Constant value for astc8x6 unorm.
    pub(crate) const ASTC8X6_UNORM: u32 = 0x56;
    /// Constant value for astc8x6 unorm srgb.
    pub(crate) const ASTC8X6_UNORM_SRGB: u32 = 0x57;
    /// Constant value for astc8x8 unorm.
    pub(crate) const ASTC8X8_UNORM: u32 = 0x58;
    /// Constant value for astc8x8 unorm srgb.
    pub(crate) const ASTC8X8_UNORM_SRGB: u32 = 0x59;
    /// Constant value for astc10x5 unorm.
    pub(crate) const ASTC10X5_UNORM: u32 = 0x5A;
    /// Constant value for astc10x5 unorm srgb.
    pub(crate) const ASTC10X5_UNORM_SRGB: u32 = 0x5B;
    /// Constant value for astc10x6 unorm.
    pub(crate) const ASTC10X6_UNORM: u32 = 0x5C;
    /// Constant value for astc10x6 unorm srgb.
    pub(crate) const ASTC10X6_UNORM_SRGB: u32 = 0x5D;
    /// Constant value for astc10x8 unorm.
    pub(crate) const ASTC10X8_UNORM: u32 = 0x5E;
    /// Constant value for astc10x8 unorm srgb.
    pub(crate) const ASTC10X8_UNORM_SRGB: u32 = 0x5F;
    /// Constant value for astc10x10 unorm.
    pub(crate) const ASTC10X10_UNORM: u32 = 0x60;
    /// Constant value for astc10x10 unorm srgb.
    pub(crate) const ASTC10X10_UNORM_SRGB: u32 = 0x61;
    /// Constant value for astc12x10 unorm.
    pub(crate) const ASTC12X10_UNORM: u32 = 0x62;
    /// Constant value for astc12x10 unorm srgb.
    pub(crate) const ASTC12X10_UNORM_SRGB: u32 = 0x63;
    /// Constant value for astc12x12 unorm.
    pub(crate) const ASTC12X12_UNORM: u32 = 0x64;
    /// Constant value for astc12x12 unorm srgb.
    pub(crate) const ASTC12X12_UNORM_SRGB: u32 = 0x65;

    /// Constructs this object from raw.
    #[must_use]
    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the raw.
    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }

    /// Returns true when this object is undefined.
    #[must_use]
    pub(crate) fn is_undefined(self) -> bool {
        self.0 == TextureFormat::UNDEFINED
    }

    /// Returns the caps.
    #[must_use]
    pub fn caps(self, features: &FeatureSet) -> Option<FormatCaps> {
        if self.is_undefined() {
            return None;
        }

        let mut caps = match self.0 {
            TextureFormat::R8_UNORM => FormatCaps::float_color(1, 1)
                .blendable()
                .renderable()
                .multisample(),
            TextureFormat::R8_SNORM => FormatCaps::float_color(1, 1).blendable(),
            TextureFormat::R8_UINT => FormatCaps::uint_color(1, 1).renderable().multisample(),
            TextureFormat::R8_SINT => FormatCaps::sint_color(1, 1).renderable().multisample(),
            TextureFormat::R16_UNORM => {
                if !features.contains(&Feature::TextureFormatsTier1) {
                    return None;
                }
                FormatCaps::float_color(2, 1)
                    .blendable()
                    .renderable()
                    .multisample()
                    .storage()
            }
            TextureFormat::R16_SNORM => {
                if !features.contains(&Feature::TextureFormatsTier1) {
                    return None;
                }
                FormatCaps::float_color(2, 1)
                    .blendable()
                    .renderable()
                    .multisample()
                    .storage()
            }
            TextureFormat::R16_UINT => FormatCaps::uint_color(2, 1).renderable().multisample(),
            TextureFormat::R16_SINT => FormatCaps::sint_color(2, 1).renderable().multisample(),
            TextureFormat::R16_FLOAT => FormatCaps::float_color(2, 1).renderable().multisample(),
            TextureFormat::RG8_UNORM => FormatCaps::float_color(2, 2)
                .blendable()
                .renderable()
                .multisample(),
            TextureFormat::RG8_SNORM => FormatCaps::float_color(2, 2).blendable(),
            TextureFormat::RG8_UINT => FormatCaps::uint_color(2, 2).renderable().multisample(),
            TextureFormat::RG8_SINT => FormatCaps::sint_color(2, 2).renderable().multisample(),
            TextureFormat::RG16_UNORM => {
                if !features.contains(&Feature::TextureFormatsTier1) {
                    return None;
                }
                FormatCaps::float_color(4, 2)
                    .blendable()
                    .renderable()
                    .multisample()
                    .storage()
            }
            TextureFormat::RG16_SNORM => {
                if !features.contains(&Feature::TextureFormatsTier1) {
                    return None;
                }
                FormatCaps::float_color(4, 2)
                    .blendable()
                    .renderable()
                    .multisample()
                    .storage()
            }
            TextureFormat::RG16_UINT => FormatCaps::uint_color(4, 2).renderable().multisample(),
            TextureFormat::RG16_SINT => FormatCaps::sint_color(4, 2).renderable().multisample(),
            TextureFormat::RG16_FLOAT => FormatCaps::float_color(4, 2).renderable().multisample(),
            TextureFormat::R32_FLOAT => FormatCaps::float_color(4, 1)
                .renderable()
                .multisample()
                .storage()
                .read_write_storage(),
            TextureFormat::R32_UINT => FormatCaps::uint_color(4, 1)
                .renderable()
                .storage()
                .read_write_storage(),
            TextureFormat::R32_SINT => FormatCaps::sint_color(4, 1)
                .renderable()
                .storage()
                .read_write_storage(),
            TextureFormat::RGBA8_UNORM => FormatCaps::float_color(4, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample()
                .storage(),
            TextureFormat::RGBA8_UNORM_SRGB => FormatCaps::float_color(4, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample(),
            TextureFormat::BGRA8_UNORM | TextureFormat::BGRA8_UNORM_SRGB => {
                FormatCaps::float_color(4, 4)
                    .alpha()
                    .blendable()
                    .renderable()
                    .multisample()
            }
            TextureFormat::RGBA8_SNORM => {
                FormatCaps::float_color(4, 4).alpha().blendable().storage()
            }
            TextureFormat::RGBA8_UINT => FormatCaps::uint_color(4, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            TextureFormat::RGBA8_SINT => FormatCaps::sint_color(4, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            TextureFormat::RGB10A2_UINT => FormatCaps::uint_color(4, 4).renderable().multisample(),
            TextureFormat::RGB10A2_UNORM => FormatCaps::float_color(4, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample(),
            TextureFormat::RG11B10_UFLOAT => {
                if !features.contains(&Feature::Rg11b10UfloatRenderable) {
                    return None;
                }
                FormatCaps::float_color(4, 3).blendable()
            }
            TextureFormat::RGB9E5_UFLOAT => FormatCaps::float_color(4, 3).blendable(),
            TextureFormat::RG32_FLOAT => FormatCaps::float_color(8, 2).renderable().storage(),
            TextureFormat::RG32_UINT => FormatCaps::uint_color(8, 2).renderable().storage(),
            TextureFormat::RG32_SINT => FormatCaps::sint_color(8, 2).renderable().storage(),
            TextureFormat::RGBA16_UNORM => FormatCaps::float_color(8, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample()
                .storage(),
            TextureFormat::RGBA16_SNORM => FormatCaps::float_color(8, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample()
                .storage(),
            TextureFormat::RGBA16_UINT => FormatCaps::uint_color(8, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            TextureFormat::RGBA16_SINT => FormatCaps::sint_color(8, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            TextureFormat::RGBA16_FLOAT => FormatCaps::float_color(8, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample()
                .storage(),
            TextureFormat::RGBA32_FLOAT => FormatCaps::float_color(16, 4)
                .alpha()
                .renderable()
                .storage(),
            TextureFormat::RGBA32_UINT => {
                FormatCaps::uint_color(16, 4).alpha().renderable().storage()
            }
            TextureFormat::RGBA32_SINT => {
                FormatCaps::sint_color(16, 4).alpha().renderable().storage()
            }
            TextureFormat::STENCIL8 => FormatCaps::stencil(1).renderable().multisample(),
            TextureFormat::DEPTH16_UNORM => FormatCaps::depth(2).renderable().multisample(),
            TextureFormat::DEPTH24_PLUS => FormatCaps::depth(4).renderable().multisample(),
            TextureFormat::DEPTH24_PLUS_STENCIL8 => {
                FormatCaps::depth_stencil(4).renderable().multisample()
            }
            TextureFormat::DEPTH32_FLOAT => FormatCaps::depth(4).renderable().multisample(),
            TextureFormat::DEPTH32_FLOAT_STENCIL8 => {
                if !features.contains(&Feature::Depth32FloatStencil8) {
                    return None;
                }
                FormatCaps::depth_stencil(5).renderable().multisample()
            }
            TextureFormat::BC1_RGBA_UNORM | TextureFormat::BC1_RGBA_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionBc) {
                    return None;
                }
                FormatCaps::compressed_color(8, 4, 4)
            }
            TextureFormat::BC2_RGBA_UNORM
            | TextureFormat::BC2_RGBA_UNORM_SRGB
            | TextureFormat::BC3_RGBA_UNORM
            | TextureFormat::BC3_RGBA_UNORM_SRGB
            | TextureFormat::BC5_RG_UNORM
            | TextureFormat::BC5_RG_SNORM
            | TextureFormat::BC6H_RGB_UFLOAT
            | TextureFormat::BC6H_RGB_FLOAT
            | TextureFormat::BC7_RGBA_UNORM
            | TextureFormat::BC7_RGBA_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionBc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 4, 4)
            }
            TextureFormat::BC4_R_UNORM | TextureFormat::BC4_R_SNORM => {
                if !features.contains(&Feature::TextureCompressionBc) {
                    return None;
                }
                FormatCaps::compressed_color(8, 4, 4)
            }
            TextureFormat::ETC2_RGB8_UNORM
            | TextureFormat::ETC2_RGB8_UNORM_SRGB
            | TextureFormat::ETC2_RGB8A1_UNORM
            | TextureFormat::ETC2_RGB8A1_UNORM_SRGB
            | TextureFormat::EAC_R11_UNORM
            | TextureFormat::EAC_R11_SNORM => {
                if !features.contains(&Feature::TextureCompressionEtc2) {
                    return None;
                }
                FormatCaps::compressed_color(8, 4, 4)
            }
            TextureFormat::ETC2_RGBA8_UNORM
            | TextureFormat::ETC2_RGBA8_UNORM_SRGB
            | TextureFormat::EAC_RG11_UNORM
            | TextureFormat::EAC_RG11_SNORM => {
                if !features.contains(&Feature::TextureCompressionEtc2) {
                    return None;
                }
                FormatCaps::compressed_color(16, 4, 4)
            }
            TextureFormat::ASTC4X4_UNORM | TextureFormat::ASTC4X4_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 4, 4)
            }
            TextureFormat::ASTC5X4_UNORM | TextureFormat::ASTC5X4_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 5, 4)
            }
            TextureFormat::ASTC5X5_UNORM | TextureFormat::ASTC5X5_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 5, 5)
            }
            TextureFormat::ASTC6X5_UNORM | TextureFormat::ASTC6X5_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 6, 5)
            }
            TextureFormat::ASTC6X6_UNORM | TextureFormat::ASTC6X6_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 6, 6)
            }
            TextureFormat::ASTC8X5_UNORM | TextureFormat::ASTC8X5_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 8, 5)
            }
            TextureFormat::ASTC8X6_UNORM | TextureFormat::ASTC8X6_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 8, 6)
            }
            TextureFormat::ASTC8X8_UNORM | TextureFormat::ASTC8X8_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 8, 8)
            }
            TextureFormat::ASTC10X5_UNORM | TextureFormat::ASTC10X5_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 10, 5)
            }
            TextureFormat::ASTC10X6_UNORM | TextureFormat::ASTC10X6_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 10, 6)
            }
            TextureFormat::ASTC10X8_UNORM | TextureFormat::ASTC10X8_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 10, 8)
            }
            TextureFormat::ASTC10X10_UNORM | TextureFormat::ASTC10X10_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 10, 10)
            }
            TextureFormat::ASTC12X10_UNORM | TextureFormat::ASTC12X10_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 12, 10)
            }
            TextureFormat::ASTC12X12_UNORM | TextureFormat::ASTC12X12_UNORM_SRGB => {
                if !features.contains(&Feature::TextureCompressionAstc) {
                    return None;
                }
                FormatCaps::compressed_color(16, 12, 12)
            }
            // Unknown defined formats are unsupported until explicitly modeled.
            _ => return None,
        };
        self.apply_feature_upgrades(features, &mut caps);
        Some(caps)
    }

    fn apply_feature_upgrades(self, features: &FeatureSet, caps: &mut FormatCaps) {
        if caps.output_class == Some(FormatOutputClass::Float)
            && !matches!(
                self.0,
                TextureFormat::R32_FLOAT | TextureFormat::RG32_FLOAT | TextureFormat::RGBA32_FLOAT
            )
        {
            caps.filterable = true;
        }
        if self.0 == TextureFormat::RG11B10_UFLOAT
            && features.contains(&Feature::Rg11b10UfloatRenderable)
        {
            caps.renderable = true;
            caps.multisample_capable = true;
        }
        if features.contains(&Feature::TextureFormatsTier1) {
            match self.0 {
                TextureFormat::R8_UNORM
                | TextureFormat::R8_SNORM
                | TextureFormat::R8_UINT
                | TextureFormat::R8_SINT
                | TextureFormat::RG8_UNORM
                | TextureFormat::RG8_SNORM
                | TextureFormat::RG8_UINT
                | TextureFormat::RG8_SINT
                | TextureFormat::R16_UINT
                | TextureFormat::R16_SINT
                | TextureFormat::R16_FLOAT
                | TextureFormat::RG16_UINT
                | TextureFormat::RG16_SINT
                | TextureFormat::RG16_FLOAT
                | TextureFormat::RGB10A2_UINT
                | TextureFormat::RGB10A2_UNORM
                | TextureFormat::RG11B10_UFLOAT => {
                    caps.storage_capable = true;
                    caps.storage_read_only_capable = true;
                }
                _ => {}
            }
        }
        if self.0 == TextureFormat::BGRA8_UNORM && features.contains(&Feature::Bgra8UnormStorage) {
            // `bgra8unorm` storage is write-only — NOT read-only-capable (the one
            // WebGPU format with that asymmetry); leave `storage_read_only_capable`.
            caps.storage_capable = true;
        }
        if features.contains(&Feature::TextureFormatsTier1) {
            match self.0 {
                TextureFormat::R8_SNORM | TextureFormat::RG8_SNORM | TextureFormat::RGBA8_SNORM => {
                    caps.renderable = true;
                    caps.multisample_capable = true;
                }
                _ => {}
            }
        }
        if features.contains(&Feature::TextureFormatsTier2) {
            // `texture-formats-tier2` grants read-write storage to this set
            // (mirrors the WebGPU CTS `kTextureFormatsTier2EnablesStorageReadWrite`).
            // R32{uint,sint,float} already have read-write storage at baseline.
            match self.0 {
                TextureFormat::R8_UNORM
                | TextureFormat::R8_UINT
                | TextureFormat::R8_SINT
                | TextureFormat::RGBA8_UNORM
                | TextureFormat::RGBA8_UINT
                | TextureFormat::RGBA8_SINT
                | TextureFormat::R16_UINT
                | TextureFormat::R16_SINT
                | TextureFormat::R16_FLOAT
                | TextureFormat::RGBA16_UINT
                | TextureFormat::RGBA16_SINT
                | TextureFormat::RGBA16_FLOAT
                | TextureFormat::RGBA32_UINT
                | TextureFormat::RGBA32_SINT
                | TextureFormat::RGBA32_FLOAT => {
                    caps.read_write_storage_capable = true;
                }
                _ => {}
            }
        }
        if matches!(
            self.0,
            TextureFormat::R32_FLOAT | TextureFormat::RG32_FLOAT | TextureFormat::RGBA32_FLOAT
        ) && features.contains(&Feature::Float32Filterable)
        {
            caps.filterable = true;
        }
    }

    /// Returns true when the format belongs to a BC compressed family.
    #[must_use]
    pub(crate) fn is_bc_compressed(self) -> bool {
        matches!(
            self.0,
            TextureFormat::BC1_RGBA_UNORM
                | TextureFormat::BC1_RGBA_UNORM_SRGB
                | TextureFormat::BC2_RGBA_UNORM
                | TextureFormat::BC2_RGBA_UNORM_SRGB
                | TextureFormat::BC3_RGBA_UNORM
                | TextureFormat::BC3_RGBA_UNORM_SRGB
                | TextureFormat::BC4_R_UNORM
                | TextureFormat::BC4_R_SNORM
                | TextureFormat::BC5_RG_UNORM
                | TextureFormat::BC5_RG_SNORM
                | TextureFormat::BC6H_RGB_UFLOAT
                | TextureFormat::BC6H_RGB_FLOAT
                | TextureFormat::BC7_RGBA_UNORM
                | TextureFormat::BC7_RGBA_UNORM_SRGB
        )
    }

    /// Returns true when the format belongs to an ASTC compressed family.
    #[must_use]
    pub(crate) fn is_astc_compressed(self) -> bool {
        matches!(
            self.0,
            TextureFormat::ASTC4X4_UNORM..=TextureFormat::ASTC12X12_UNORM_SRGB
        )
    }

    /// Returns true when the format belongs to an ETC2/EAC compressed family.
    #[must_use]
    pub(crate) fn is_etc2_compressed(self) -> bool {
        matches!(
            self.0,
            TextureFormat::ETC2_RGB8_UNORM
                | TextureFormat::ETC2_RGB8_UNORM_SRGB
                | TextureFormat::ETC2_RGB8A1_UNORM
                | TextureFormat::ETC2_RGB8A1_UNORM_SRGB
                | TextureFormat::ETC2_RGBA8_UNORM
                | TextureFormat::ETC2_RGBA8_UNORM_SRGB
                | TextureFormat::EAC_R11_UNORM
                | TextureFormat::EAC_R11_SNORM
                | TextureFormat::EAC_RG11_UNORM
                | TextureFormat::EAC_RG11_SNORM
        )
    }

    /// Returns the sRGB and linear variants for this texture format.
    #[must_use]
    pub(crate) fn srgb_pair(self) -> Option<Self> {
        let pair = match self.0 {
            TextureFormat::RGBA8_UNORM => TextureFormat::RGBA8_UNORM_SRGB,
            TextureFormat::RGBA8_UNORM_SRGB => TextureFormat::RGBA8_UNORM,
            TextureFormat::BGRA8_UNORM => TextureFormat::BGRA8_UNORM_SRGB,
            TextureFormat::BGRA8_UNORM_SRGB => TextureFormat::BGRA8_UNORM,
            TextureFormat::BC1_RGBA_UNORM => TextureFormat::BC1_RGBA_UNORM_SRGB,
            TextureFormat::BC1_RGBA_UNORM_SRGB => TextureFormat::BC1_RGBA_UNORM,
            TextureFormat::BC2_RGBA_UNORM => TextureFormat::BC2_RGBA_UNORM_SRGB,
            TextureFormat::BC2_RGBA_UNORM_SRGB => TextureFormat::BC2_RGBA_UNORM,
            TextureFormat::BC3_RGBA_UNORM => TextureFormat::BC3_RGBA_UNORM_SRGB,
            TextureFormat::BC3_RGBA_UNORM_SRGB => TextureFormat::BC3_RGBA_UNORM,
            TextureFormat::BC7_RGBA_UNORM => TextureFormat::BC7_RGBA_UNORM_SRGB,
            TextureFormat::BC7_RGBA_UNORM_SRGB => TextureFormat::BC7_RGBA_UNORM,
            TextureFormat::ETC2_RGB8_UNORM => TextureFormat::ETC2_RGB8_UNORM_SRGB,
            TextureFormat::ETC2_RGB8_UNORM_SRGB => TextureFormat::ETC2_RGB8_UNORM,
            TextureFormat::ETC2_RGB8A1_UNORM => TextureFormat::ETC2_RGB8A1_UNORM_SRGB,
            TextureFormat::ETC2_RGB8A1_UNORM_SRGB => TextureFormat::ETC2_RGB8A1_UNORM,
            TextureFormat::ETC2_RGBA8_UNORM => TextureFormat::ETC2_RGBA8_UNORM_SRGB,
            TextureFormat::ETC2_RGBA8_UNORM_SRGB => TextureFormat::ETC2_RGBA8_UNORM,
            TextureFormat::ASTC4X4_UNORM => TextureFormat::ASTC4X4_UNORM_SRGB,
            TextureFormat::ASTC4X4_UNORM_SRGB => TextureFormat::ASTC4X4_UNORM,
            TextureFormat::ASTC5X4_UNORM => TextureFormat::ASTC5X4_UNORM_SRGB,
            TextureFormat::ASTC5X4_UNORM_SRGB => TextureFormat::ASTC5X4_UNORM,
            TextureFormat::ASTC5X5_UNORM => TextureFormat::ASTC5X5_UNORM_SRGB,
            TextureFormat::ASTC5X5_UNORM_SRGB => TextureFormat::ASTC5X5_UNORM,
            TextureFormat::ASTC6X5_UNORM => TextureFormat::ASTC6X5_UNORM_SRGB,
            TextureFormat::ASTC6X5_UNORM_SRGB => TextureFormat::ASTC6X5_UNORM,
            TextureFormat::ASTC6X6_UNORM => TextureFormat::ASTC6X6_UNORM_SRGB,
            TextureFormat::ASTC6X6_UNORM_SRGB => TextureFormat::ASTC6X6_UNORM,
            TextureFormat::ASTC8X5_UNORM => TextureFormat::ASTC8X5_UNORM_SRGB,
            TextureFormat::ASTC8X5_UNORM_SRGB => TextureFormat::ASTC8X5_UNORM,
            TextureFormat::ASTC8X6_UNORM => TextureFormat::ASTC8X6_UNORM_SRGB,
            TextureFormat::ASTC8X6_UNORM_SRGB => TextureFormat::ASTC8X6_UNORM,
            TextureFormat::ASTC8X8_UNORM => TextureFormat::ASTC8X8_UNORM_SRGB,
            TextureFormat::ASTC8X8_UNORM_SRGB => TextureFormat::ASTC8X8_UNORM,
            TextureFormat::ASTC10X5_UNORM => TextureFormat::ASTC10X5_UNORM_SRGB,
            TextureFormat::ASTC10X5_UNORM_SRGB => TextureFormat::ASTC10X5_UNORM,
            TextureFormat::ASTC10X6_UNORM => TextureFormat::ASTC10X6_UNORM_SRGB,
            TextureFormat::ASTC10X6_UNORM_SRGB => TextureFormat::ASTC10X6_UNORM,
            TextureFormat::ASTC10X8_UNORM => TextureFormat::ASTC10X8_UNORM_SRGB,
            TextureFormat::ASTC10X8_UNORM_SRGB => TextureFormat::ASTC10X8_UNORM,
            TextureFormat::ASTC10X10_UNORM => TextureFormat::ASTC10X10_UNORM_SRGB,
            TextureFormat::ASTC10X10_UNORM_SRGB => TextureFormat::ASTC10X10_UNORM,
            TextureFormat::ASTC12X10_UNORM => TextureFormat::ASTC12X10_UNORM_SRGB,
            TextureFormat::ASTC12X10_UNORM_SRGB => TextureFormat::ASTC12X10_UNORM,
            TextureFormat::ASTC12X12_UNORM => TextureFormat::ASTC12X12_UNORM_SRGB,
            TextureFormat::ASTC12X12_UNORM_SRGB => TextureFormat::ASTC12X12_UNORM,
            _ => return None,
        };
        Some(Self(pair))
    }
}

impl From<u32> for TextureFormat {
    fn from(value: u32) -> Self {
        TextureFormat::from_raw(value)
    }
}

impl From<i32> for TextureFormat {
    fn from(value: i32) -> Self {
        TextureFormat::from_raw(value as u32)
    }
}

impl From<TextureFormat> for u32 {
    fn from(value: TextureFormat) -> Self {
        value.raw()
    }
}

impl From<TextureFormat> for i32 {
    fn from(value: TextureFormat) -> Self {
        value.raw() as i32
    }
}

/// Stores format aspects data used by validation and backend submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatAspects {
    /// Color.
    pub color: bool,
    /// Depth.
    pub depth: bool,
    /// Stencil.
    pub stencil: bool,
}

/// Stores format caps data used by validation and backend submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatCaps {
    /// Aspects.
    pub aspects: FormatAspects,
    /// Renderable.
    pub renderable: bool,
    /// Multisample capable.
    pub multisample_capable: bool,
    /// Storage capable (write-only storage; also the texture `STORAGE_BINDING`
    /// usage). A format may be write-only-storage capable yet not read-only —
    /// the only such WebGPU format is `bgra8unorm` — so read-only access is
    /// gated by [`Self::storage_read_only_capable`], not this flag.
    pub storage_capable: bool,
    /// Read-only storage capable. Equals [`Self::storage_capable`] for every
    /// format except `bgra8unorm` (write-only-storage but not read-only).
    pub storage_read_only_capable: bool,
    /// Read-write storage capable.
    pub read_write_storage_capable: bool,
    /// Filterable as a float sampled texture.
    pub filterable: bool,
    /// Output class.
    pub output_class: Option<FormatOutputClass>,
    /// Color components.
    pub color_components: u8,
    /// Is blendable.
    pub is_blendable: bool,
    /// Has alpha.
    pub has_alpha: bool,
    /// Is compressed.
    pub is_compressed: bool,
    /// Texel block size.
    pub texel_block_size: u32,
    /// Block w.
    pub block_w: u32,
    /// Block h.
    pub block_h: u32,
}

/// Enumerates format output class values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatOutputClass {
    /// Float variant.
    Float,
    /// Sint variant.
    Sint,
    /// Uint variant.
    Uint,
}

impl FormatCaps {
    /// Constant value for fn.
    pub(crate) const fn float_color(texel_block_size: u32, color_components: u8) -> Self {
        Self::color(texel_block_size, color_components, FormatOutputClass::Float)
    }

    /// Constant value for fn.
    pub(crate) const fn sint_color(texel_block_size: u32, color_components: u8) -> Self {
        Self::color(texel_block_size, color_components, FormatOutputClass::Sint)
    }

    /// Constant value for fn.
    pub(crate) const fn uint_color(texel_block_size: u32, color_components: u8) -> Self {
        Self::color(texel_block_size, color_components, FormatOutputClass::Uint)
    }

    /// Constant value for fn.
    pub(crate) const fn color(
        texel_block_size: u32,
        color_components: u8,
        output_class: FormatOutputClass,
    ) -> Self {
        Self::new(
            FormatAspects {
                color: true,
                depth: false,
                stencil: false,
            },
            texel_block_size,
            1,
            1,
            false,
            Some(output_class),
            color_components,
        )
    }

    /// Constant value for fn.
    pub(crate) const fn depth(texel_block_size: u32) -> Self {
        Self::new(
            FormatAspects {
                color: false,
                depth: true,
                stencil: false,
            },
            texel_block_size,
            1,
            1,
            false,
            None,
            0,
        )
    }

    /// Constant value for fn.
    pub(crate) const fn stencil(texel_block_size: u32) -> Self {
        Self::new(
            FormatAspects {
                color: false,
                depth: false,
                stencil: true,
            },
            texel_block_size,
            1,
            1,
            false,
            None,
            0,
        )
    }

    /// Constant value for fn.
    pub(crate) const fn depth_stencil(texel_block_size: u32) -> Self {
        Self::new(
            FormatAspects {
                color: false,
                depth: true,
                stencil: true,
            },
            texel_block_size,
            1,
            1,
            false,
            None,
            0,
        )
    }

    /// Constant value for fn.
    pub(crate) const fn compressed_color(
        texel_block_size: u32,
        block_w: u32,
        block_h: u32,
    ) -> Self {
        Self::new(
            FormatAspects {
                color: true,
                depth: false,
                stencil: false,
            },
            texel_block_size,
            block_w,
            block_h,
            true,
            Some(FormatOutputClass::Float),
            4,
        )
    }

    /// Constant value for fn.
    pub(crate) const fn new(
        aspects: FormatAspects,
        texel_block_size: u32,
        block_w: u32,
        block_h: u32,
        is_compressed: bool,
        output_class: Option<FormatOutputClass>,
        color_components: u8,
    ) -> Self {
        Self {
            aspects,
            renderable: false,
            multisample_capable: false,
            storage_capable: false,
            storage_read_only_capable: false,
            read_write_storage_capable: false,
            filterable: false,
            output_class,
            color_components,
            is_blendable: false,
            has_alpha: false,
            is_compressed,
            texel_block_size,
            block_w,
            block_h,
        }
    }

    /// Constant value for fn.
    pub(crate) const fn renderable(mut self) -> Self {
        self.renderable = true;
        self
    }

    /// Constant value for fn.
    pub(crate) const fn multisample(mut self) -> Self {
        self.multisample_capable = true;
        self
    }

    /// Constant value for fn. Marks the format as both write-only and read-only
    /// storage capable (the common case; `bgra8unorm`'s write-only-only storage
    /// is set directly in `apply_feature_upgrades`, not via this builder).
    pub(crate) const fn storage(mut self) -> Self {
        self.storage_capable = true;
        self.storage_read_only_capable = true;
        self
    }

    /// Constant value for fn.
    pub(crate) const fn read_write_storage(mut self) -> Self {
        self.read_write_storage_capable = true;
        self
    }

    /// Constant value for fn.
    pub(crate) const fn blendable(mut self) -> Self {
        self.is_blendable = true;
        self
    }

    /// Constant value for fn.
    pub(crate) const fn alpha(mut self) -> Self {
        self.has_alpha = true;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_format_from_raw_raw_and_caps_pin_rgba8_unorm_and_undefined() {
        let format = TextureFormat::from_raw(0x0000_0016);
        let features = FeatureSet::new();

        assert_eq!(format.raw(), 0x0000_0016);
        assert_eq!(TextureFormat::from(0x0000_0016_u32), format);
        assert_eq!(TextureFormat::from(0x0000_0016_i32), format);
        assert_eq!(u32::from(format), 0x0000_0016);
        assert_eq!(i32::from(format), 0x0000_0016);

        let caps = format.caps(&features).expect("RGBA8Unorm caps");
        assert_eq!(
            caps.aspects,
            FormatAspects {
                color: true,
                depth: false,
                stencil: false,
            }
        );
        assert_eq!(caps.texel_block_size, 4);
        assert_eq!(caps.block_w, 1);
        assert_eq!(caps.block_h, 1);
        assert_eq!(caps.output_class, Some(FormatOutputClass::Float));
        assert_eq!(caps.color_components, 4);
        assert!(caps.renderable);
        assert!(caps.multisample_capable);
        assert!(caps.storage_capable);
        assert!(!caps.read_write_storage_capable);
        assert!(caps.filterable);
        assert!(caps.is_blendable);
        assert!(caps.has_alpha);
        assert!(!caps.is_compressed);

        assert_eq!(TextureFormat::from_raw(0).caps(&features), None);
    }

    #[test]
    fn r32_storage_formats_are_read_write_without_features() {
        let no_features = FeatureSet::new();

        for raw in [
            TextureFormat::R32_UINT,
            TextureFormat::R32_SINT,
            TextureFormat::R32_FLOAT,
        ] {
            assert!(
                TextureFormat::from_raw(raw)
                    .caps(&no_features)
                    .is_some_and(|caps| caps.read_write_storage_capable),
                "{raw:#x} must support read-write storage without feature gates"
            );
        }

        assert!(!TextureFormat::from_raw(TextureFormat::RGBA8_UNORM)
            .caps(&no_features)
            .is_some_and(|caps| caps.read_write_storage_capable));
    }

    #[test]
    fn texture_format_caps_are_feature_gated() {
        let no_features = FeatureSet::new();
        let mut features = FeatureSet::new();
        features.insert(Feature::TextureCompressionBc);
        features.insert(Feature::Depth32FloatStencil8);
        features.insert(Feature::Rg11b10UfloatRenderable);
        features.insert(Feature::Bgra8UnormStorage);
        features.insert(Feature::Float32Filterable);
        features.insert(Feature::TextureFormatsTier1);
        features.insert(Feature::TextureFormatsTier2);

        assert_eq!(
            TextureFormat::from_raw(TextureFormat::BC1_RGBA_UNORM).caps(&no_features),
            None
        );
        assert!(TextureFormat::from_raw(TextureFormat::BC1_RGBA_UNORM)
            .caps(&features)
            .is_some());
        assert_eq!(
            TextureFormat::from_raw(TextureFormat::DEPTH32_FLOAT_STENCIL8).caps(&no_features),
            None
        );
        assert!(TextureFormat::from_raw(TextureFormat::RG11B10_UFLOAT)
            .caps(&features)
            .is_some_and(|caps| caps.renderable && caps.multisample_capable));
        assert!(TextureFormat::from_raw(TextureFormat::BGRA8_UNORM)
            .caps(&features)
            .is_some_and(|caps| caps.storage_capable));
        assert!(TextureFormat::from_raw(TextureFormat::R32_FLOAT)
            .caps(&features)
            .is_some_and(|caps| caps.filterable && caps.read_write_storage_capable));
    }

    #[test]
    fn texture_format_caps_cover_cts_create_texture_findings() {
        let no_features = FeatureSet::new();
        let mut tier1 = FeatureSet::new();
        tier1.insert(Feature::TextureFormatsTier1);

        for raw in [
            TextureFormat::R16_UINT,
            TextureFormat::R16_SINT,
            TextureFormat::R16_FLOAT,
            TextureFormat::RG16_UINT,
            TextureFormat::RG16_SINT,
            TextureFormat::RG16_FLOAT,
            TextureFormat::RGB10A2_UINT,
            TextureFormat::RGB10A2_UNORM,
        ] {
            let caps = TextureFormat::from_raw(raw)
                .caps(&no_features)
                .expect("core color format should have caps");
            assert!(caps.renderable);
            assert!(caps.multisample_capable);
        }

        for raw in [
            TextureFormat::R16_UNORM,
            TextureFormat::R16_SNORM,
            TextureFormat::RG16_UNORM,
            TextureFormat::RG16_SNORM,
        ] {
            assert_eq!(TextureFormat::from_raw(raw).caps(&no_features), None);
            let caps = TextureFormat::from_raw(raw)
                .caps(&tier1)
                .expect("tier1 color format should have caps");
            assert!(caps.renderable);
            assert!(caps.multisample_capable);
            assert!(caps.storage_capable);
        }

        assert!(
            !TextureFormat::from_raw(TextureFormat::R32_UINT)
                .caps(&no_features)
                .expect("R32Uint caps")
                .multisample_capable
        );
        assert!(
            !TextureFormat::from_raw(TextureFormat::R32_SINT)
                .caps(&no_features)
                .expect("R32Sint caps")
                .multisample_capable
        );
        assert!(
            !TextureFormat::from_raw(TextureFormat::RGBA8_SNORM)
                .caps(&no_features)
                .expect("RGBA8Snorm caps")
                .renderable
        );
        assert!(
            TextureFormat::from_raw(TextureFormat::RGBA8_SNORM)
                .caps(&no_features)
                .expect("RGBA8Snorm caps")
                .storage_capable
        );
        assert!(
            TextureFormat::from_raw(TextureFormat::RGBA8_SNORM)
                .caps(&tier1)
                .expect("RGBA8Snorm caps")
                .renderable
        );
        assert!(
            TextureFormat::from_raw(TextureFormat::RGBA8_SNORM)
                .caps(&tier1)
                .expect("RGBA8Snorm caps")
                .multisample_capable
        );
        assert!(
            TextureFormat::from_raw(TextureFormat::RGBA8_SNORM)
                .caps(&tier1)
                .expect("RGBA8Snorm caps")
                .storage_capable
        );
    }

    #[test]
    fn storage_access_caps_match_webgpu_tables() {
        // F-059: storage-format capability must distinguish write-only / read-only
        // / read-write per the WebGPU tables.
        let mut all = FeatureSet::new();
        all.insert(Feature::TextureFormatsTier1);
        all.insert(Feature::TextureFormatsTier2);
        all.insert(Feature::Bgra8UnormStorage);

        // bgra8unorm: write-only storage capable, but NOT read-only or read-write.
        let bgra = TextureFormat::from_raw(TextureFormat::BGRA8_UNORM)
            .caps(&all)
            .expect("bgra8unorm caps");
        assert!(bgra.storage_capable, "bgra8unorm is write-only storage");
        assert!(
            !bgra.storage_read_only_capable,
            "bgra8unorm is NOT read-only storage"
        );
        assert!(!bgra.read_write_storage_capable);

        // tier1 storage formats are both write-only and read-only capable.
        let r8 = TextureFormat::from_raw(TextureFormat::R8_UNORM)
            .caps(&all)
            .expect("r8unorm caps");
        assert!(r8.storage_capable && r8.storage_read_only_capable);

        // `texture-formats-tier2` read-write set (sample members) + a non-member.
        for raw in [
            TextureFormat::RGBA8_UNORM,
            TextureFormat::RGBA16_UINT,
            TextureFormat::RGBA32_SINT,
            TextureFormat::R8_UINT,
        ] {
            assert!(
                TextureFormat::from_raw(raw)
                    .caps(&all)
                    .is_some_and(|c| c.read_write_storage_capable),
                "{raw:#x} must be read-write storage under tier2"
            );
        }
        // rg32* is storage-capable but NOT in the tier2 read-write set.
        assert!(
            !TextureFormat::from_raw(TextureFormat::RG32_UINT)
                .caps(&all)
                .expect("rg32uint caps")
                .read_write_storage_capable
        );
    }

    #[test]
    fn srgb_pairs_cover_compressed_view_compatibility_pairs() {
        for (linear, srgb) in [
            (
                TextureFormat::BC2_RGBA_UNORM,
                TextureFormat::BC2_RGBA_UNORM_SRGB,
            ),
            (
                TextureFormat::BC3_RGBA_UNORM,
                TextureFormat::BC3_RGBA_UNORM_SRGB,
            ),
            (
                TextureFormat::ETC2_RGB8_UNORM,
                TextureFormat::ETC2_RGB8_UNORM_SRGB,
            ),
            (
                TextureFormat::ETC2_RGB8A1_UNORM,
                TextureFormat::ETC2_RGB8A1_UNORM_SRGB,
            ),
            (
                TextureFormat::ETC2_RGBA8_UNORM,
                TextureFormat::ETC2_RGBA8_UNORM_SRGB,
            ),
            (
                TextureFormat::ASTC4X4_UNORM,
                TextureFormat::ASTC4X4_UNORM_SRGB,
            ),
            (
                TextureFormat::ASTC12X12_UNORM,
                TextureFormat::ASTC12X12_UNORM_SRGB,
            ),
        ] {
            assert_eq!(
                TextureFormat::from_raw(linear).srgb_pair(),
                Some(TextureFormat::from_raw(srgb))
            );
            assert_eq!(
                TextureFormat::from_raw(srgb).srgb_pair(),
                Some(TextureFormat::from_raw(linear))
            );
        }
    }
}
