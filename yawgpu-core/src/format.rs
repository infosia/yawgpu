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
    /// Constant value for rg8 unorm.
    pub(crate) const RG8_UNORM: u32 = 0x0A;
    /// Constant value for rg8 snorm.
    pub(crate) const RG8_SNORM: u32 = 0x0B;
    /// Constant value for rg8 uint.
    pub(crate) const RG8_UINT: u32 = 0x0C;
    /// Constant value for rg8 sint.
    pub(crate) const RG8_SINT: u32 = 0x0D;
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
    /// Constant value for bgra8 unorm.
    pub(crate) const BGRA8_UNORM: u32 = 0x1B;
    /// Constant value for bgra8 unorm srgb.
    pub(crate) const BGRA8_UNORM_SRGB: u32 = 0x1C;
    /// Constant value for bc7 rgba unorm.
    pub(crate) const BC7_RGBA_UNORM: u32 = 0x3E;
    /// Constant value for bc7 rgba unorm srgb.
    pub(crate) const BC7_RGBA_UNORM_SRGB: u32 = 0x3F;

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
        self.0 == Self::UNDEFINED
    }

    /// Returns the caps.
    #[must_use]
    pub fn caps(self) -> Option<FormatCaps> {
        if self.is_undefined() {
            return None;
        }

        let caps = match self.0 {
            Self::R8_UNORM => FormatCaps::float_color(1, 1)
                .blendable()
                .renderable()
                .multisample(),
            Self::R8_SNORM => FormatCaps::float_color(1, 1).blendable(),
            Self::R8_UINT => FormatCaps::uint_color(1, 1).renderable().multisample(),
            Self::R8_SINT => FormatCaps::sint_color(1, 1).renderable().multisample(),
            Self::RG8_UNORM => FormatCaps::float_color(2, 2)
                .blendable()
                .renderable()
                .multisample(),
            Self::RG8_SNORM => FormatCaps::float_color(2, 2).blendable(),
            Self::RG8_UINT => FormatCaps::uint_color(2, 2).renderable().multisample(),
            Self::RG8_SINT => FormatCaps::sint_color(2, 2).renderable().multisample(),
            Self::R32_FLOAT => FormatCaps::float_color(4, 1)
                .renderable()
                .multisample()
                .storage(),
            Self::R32_UINT => FormatCaps::uint_color(4, 1)
                .renderable()
                .multisample()
                .storage(),
            Self::R32_SINT => FormatCaps::sint_color(4, 1)
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA8_UNORM => FormatCaps::float_color(4, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA8_UNORM_SRGB => FormatCaps::float_color(4, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample(),
            Self::BGRA8_UNORM | Self::BGRA8_UNORM_SRGB => FormatCaps::float_color(4, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample(),
            // snorm formats are NOT storage-capable (Dawn `Format.cpp`).
            Self::RGBA8_SNORM => FormatCaps::float_color(4, 4).alpha().blendable(),
            Self::RGBA8_UINT => FormatCaps::uint_color(4, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA8_SINT => FormatCaps::sint_color(4, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            Self::RG11B10_UFLOAT | Self::RGB9E5_UFLOAT => FormatCaps::float_color(4, 3).blendable(),
            Self::RG32_FLOAT => FormatCaps::float_color(8, 2).renderable().storage(),
            Self::RG32_UINT => FormatCaps::uint_color(8, 2).renderable().storage(),
            Self::RG32_SINT => FormatCaps::sint_color(8, 2).renderable().storage(),
            Self::RGBA16_UNORM => FormatCaps::float_color(8, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample()
                .storage(),
            // snorm formats are NOT storage-capable (Dawn `Format.cpp`); the
            // remaining `*16` renderable/multisample approximation stays a
            // tracked note (block 20 → P4/P5).
            Self::RGBA16_SNORM => FormatCaps::float_color(8, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample(),
            Self::RGBA16_UINT => FormatCaps::uint_color(8, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA16_SINT => FormatCaps::sint_color(8, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA16_FLOAT => FormatCaps::float_color(8, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA32_FLOAT => FormatCaps::float_color(16, 4)
                .alpha()
                .renderable()
                .storage(),
            Self::RGBA32_UINT => FormatCaps::uint_color(16, 4).alpha().renderable().storage(),
            Self::RGBA32_SINT => FormatCaps::sint_color(16, 4).alpha().renderable().storage(),
            Self::STENCIL8 => FormatCaps::stencil(1).renderable().multisample(),
            Self::DEPTH16_UNORM => FormatCaps::depth(2).renderable().multisample(),
            Self::DEPTH24_PLUS => FormatCaps::depth(4).renderable().multisample(),
            Self::DEPTH24_PLUS_STENCIL8 => FormatCaps::depth_stencil(4).renderable().multisample(),
            Self::DEPTH32_FLOAT => FormatCaps::depth(4).renderable().multisample(),
            Self::DEPTH32_FLOAT_STENCIL8 => FormatCaps::depth_stencil(5).renderable().multisample(),
            Self::BC1_RGBA_UNORM | Self::BC1_RGBA_UNORM_SRGB => {
                FormatCaps::compressed_color(8, 4, 4)
            }
            Self::BC7_RGBA_UNORM | Self::BC7_RGBA_UNORM_SRGB => {
                FormatCaps::compressed_color(16, 4, 4)
            }
            // Unknown defined formats are unsupported until explicitly modeled.
            _ => return None,
        };
        Some(caps)
    }

    /// Returns the sRGB and linear variants for this texture format.
    #[must_use]
    pub(crate) fn srgb_pair(self) -> Option<Self> {
        let pair = match self.0 {
            Self::RGBA8_UNORM => Self::RGBA8_UNORM_SRGB,
            Self::RGBA8_UNORM_SRGB => Self::RGBA8_UNORM,
            Self::BGRA8_UNORM => Self::BGRA8_UNORM_SRGB,
            Self::BGRA8_UNORM_SRGB => Self::BGRA8_UNORM,
            Self::BC1_RGBA_UNORM => Self::BC1_RGBA_UNORM_SRGB,
            Self::BC1_RGBA_UNORM_SRGB => Self::BC1_RGBA_UNORM,
            Self::BC7_RGBA_UNORM => Self::BC7_RGBA_UNORM_SRGB,
            Self::BC7_RGBA_UNORM_SRGB => Self::BC7_RGBA_UNORM,
            _ => return None,
        };
        Some(Self(pair))
    }
}

impl From<u32> for TextureFormat {
    fn from(value: u32) -> Self {
        Self::from_raw(value)
    }
}

impl From<i32> for TextureFormat {
    fn from(value: i32) -> Self {
        Self::from_raw(value as u32)
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
    /// Storage capable.
    pub storage_capable: bool,
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

    /// Constant value for fn.
    pub(crate) const fn storage(mut self) -> Self {
        self.storage_capable = true;
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

        assert_eq!(format.raw(), 0x0000_0016);
        assert_eq!(TextureFormat::from(0x0000_0016_u32), format);
        assert_eq!(TextureFormat::from(0x0000_0016_i32), format);
        assert_eq!(u32::from(format), 0x0000_0016);
        assert_eq!(i32::from(format), 0x0000_0016);

        let caps = format.caps().expect("RGBA8Unorm caps");
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
        assert!(caps.is_blendable);
        assert!(caps.has_alpha);
        assert!(!caps.is_compressed);

        assert_eq!(TextureFormat::from_raw(0).caps(), None);
    }
}
