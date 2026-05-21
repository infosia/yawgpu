use std::cell::UnsafeCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{
    HalAdapter, HalAddressMode, HalBackend, HalBoundBuffer, HalBuffer, HalBufferBindingKind,
    HalBufferCopy, HalBufferTextureCopy, HalBufferTextureLayout, HalCompareFunction,
    HalComputePass, HalComputePipeline, HalCopy, HalDescriptorBinding, HalDevice, HalDraw,
    HalError, HalExtent3d, HalFilterMode, HalInstance, HalMipmapFilterMode, HalOrigin3d,
    HalPrimitiveTopology, HalQueue, HalRenderColorTarget, HalRenderLoadOp, HalRenderPass,
    HalRenderPipeline, HalRenderPipelineDescriptor, HalSampler, HalSamplerDescriptor,
    HalShaderSource, HalSurface, HalTexture, HalTextureCopy, HalTextureDescriptor,
    HalTextureFormat, HalTextureUsage, HalVertexAttribute, HalVertexBufferLayout, HalVertexFormat,
    HalVertexStepMode,
};

use crate::adapter::*;
use crate::bind_group::*;
use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::compute_pass::*;
use crate::compute_pipeline::*;
use crate::copy::*;
use crate::device::*;
use crate::error::*;
use crate::extent::*;
use crate::future::*;
use crate::instance::*;
use crate::limits::*;
use crate::pass::*;
use crate::pipeline_layout::*;
use crate::query_set::*;
use crate::queue::*;
use crate::render_bundle::*;
use crate::render_pass::*;
use crate::render_pipeline::*;
use crate::sampler::*;
use crate::shader::*;
use crate::shader_naga;
use crate::texture::*;
use crate::texture_view::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureFormat(u32);

impl TextureFormat {
    pub(crate) const UNDEFINED: u32 = 0x00;
    pub(crate) const R8_UNORM: u32 = 0x01;
    pub(crate) const R8_SNORM: u32 = 0x02;
    pub(crate) const R8_UINT: u32 = 0x03;
    pub(crate) const R8_SINT: u32 = 0x04;
    pub(crate) const RG8_UNORM: u32 = 0x0A;
    pub(crate) const RG8_SNORM: u32 = 0x0B;
    pub(crate) const RG8_UINT: u32 = 0x0C;
    pub(crate) const RG8_SINT: u32 = 0x0D;
    pub(crate) const R32_FLOAT: u32 = 0x0E;
    pub(crate) const R32_UINT: u32 = 0x0F;
    pub(crate) const R32_SINT: u32 = 0x10;
    pub(crate) const RGBA8_UNORM: u32 = 0x16;
    pub(crate) const RGBA8_UNORM_SRGB: u32 = 0x17;
    pub(crate) const RGBA8_SNORM: u32 = 0x18;
    pub(crate) const RGBA8_UINT: u32 = 0x19;
    pub(crate) const RGBA8_SINT: u32 = 0x1A;
    pub(crate) const RG11B10_UFLOAT: u32 = 0x1F;
    pub(crate) const RGB9E5_UFLOAT: u32 = 0x20;
    pub(crate) const RG32_FLOAT: u32 = 0x21;
    pub(crate) const RG32_UINT: u32 = 0x22;
    pub(crate) const RG32_SINT: u32 = 0x23;
    pub(crate) const RGBA16_UNORM: u32 = 0x24;
    pub(crate) const RGBA16_SNORM: u32 = 0x25;
    pub(crate) const RGBA16_UINT: u32 = 0x26;
    pub(crate) const RGBA16_SINT: u32 = 0x27;
    pub(crate) const RGBA16_FLOAT: u32 = 0x28;
    pub(crate) const RGBA32_FLOAT: u32 = 0x29;
    pub(crate) const RGBA32_UINT: u32 = 0x2A;
    pub(crate) const RGBA32_SINT: u32 = 0x2B;
    pub(crate) const STENCIL8: u32 = 0x2C;
    pub(crate) const DEPTH16_UNORM: u32 = 0x2D;
    pub(crate) const DEPTH24_PLUS: u32 = 0x2E;
    pub(crate) const DEPTH24_PLUS_STENCIL8: u32 = 0x2F;
    pub(crate) const DEPTH32_FLOAT: u32 = 0x30;
    pub(crate) const DEPTH32_FLOAT_STENCIL8: u32 = 0x31;
    pub(crate) const BC1_RGBA_UNORM: u32 = 0x32;
    pub(crate) const BC1_RGBA_UNORM_SRGB: u32 = 0x33;
    pub(crate) const BGRA8_UNORM: u32 = 0x1B;
    pub(crate) const BGRA8_UNORM_SRGB: u32 = 0x1C;
    pub(crate) const BC7_RGBA_UNORM: u32 = 0x3E;
    pub(crate) const BC7_RGBA_UNORM_SRGB: u32 = 0x3F;

    #[must_use]
    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }

    #[must_use]
    pub(crate) fn is_undefined(self) -> bool {
        self.0 == Self::UNDEFINED
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatAspects {
    pub color: bool,
    pub depth: bool,
    pub stencil: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatCaps {
    pub aspects: FormatAspects,
    pub renderable: bool,
    pub multisample_capable: bool,
    pub storage_capable: bool,
    pub output_class: Option<FormatOutputClass>,
    pub color_components: u8,
    pub is_blendable: bool,
    pub has_alpha: bool,
    pub is_compressed: bool,
    pub texel_block_size: u32,
    pub block_w: u32,
    pub block_h: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatOutputClass {
    Float,
    Sint,
    Uint,
}

impl FormatCaps {
    pub(crate) const fn float_color(texel_block_size: u32, color_components: u8) -> Self {
        Self::color(texel_block_size, color_components, FormatOutputClass::Float)
    }

    pub(crate) const fn sint_color(texel_block_size: u32, color_components: u8) -> Self {
        Self::color(texel_block_size, color_components, FormatOutputClass::Sint)
    }

    pub(crate) const fn uint_color(texel_block_size: u32, color_components: u8) -> Self {
        Self::color(texel_block_size, color_components, FormatOutputClass::Uint)
    }

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

    pub(crate) const fn renderable(mut self) -> Self {
        self.renderable = true;
        self
    }

    pub(crate) const fn multisample(mut self) -> Self {
        self.multisample_capable = true;
        self
    }

    pub(crate) const fn storage(mut self) -> Self {
        self.storage_capable = true;
        self
    }

    pub(crate) const fn blendable(mut self) -> Self {
        self.is_blendable = true;
        self
    }

    pub(crate) const fn alpha(mut self) -> Self {
        self.has_alpha = true;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexFormat(u32);

impl VertexFormat {
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }

    pub(crate) fn info(self) -> VertexFormatInfo {
        match self.0 {
            0x0000_0001 => VertexFormatInfo::new(1, FormatOutputClass::Uint),
            0x0000_0002 => VertexFormatInfo::new(2, FormatOutputClass::Uint),
            0x0000_0003 => VertexFormatInfo::new(4, FormatOutputClass::Uint),
            0x0000_0004 => VertexFormatInfo::new(1, FormatOutputClass::Sint),
            0x0000_0005 => VertexFormatInfo::new(2, FormatOutputClass::Sint),
            0x0000_0006 => VertexFormatInfo::new(4, FormatOutputClass::Sint),
            0x0000_0007 | 0x0000_000A => VertexFormatInfo::new(1, FormatOutputClass::Float),
            0x0000_0008 | 0x0000_000B => VertexFormatInfo::new(2, FormatOutputClass::Float),
            0x0000_0009 | 0x0000_000C => VertexFormatInfo::new(4, FormatOutputClass::Float),
            0x0000_000D => VertexFormatInfo::new(2, FormatOutputClass::Uint),
            0x0000_000E => VertexFormatInfo::new(4, FormatOutputClass::Uint),
            0x0000_000F => VertexFormatInfo::new(8, FormatOutputClass::Uint),
            0x0000_0010 => VertexFormatInfo::new(2, FormatOutputClass::Sint),
            0x0000_0011 => VertexFormatInfo::new(4, FormatOutputClass::Sint),
            0x0000_0012 => VertexFormatInfo::new(8, FormatOutputClass::Sint),
            0x0000_0013 | 0x0000_0016 | 0x0000_0019 => {
                VertexFormatInfo::new(2, FormatOutputClass::Float)
            }
            0x0000_0014 | 0x0000_0017 | 0x0000_001A => {
                VertexFormatInfo::new(4, FormatOutputClass::Float)
            }
            0x0000_0015 | 0x0000_0018 | 0x0000_001B => {
                VertexFormatInfo::new(8, FormatOutputClass::Float)
            }
            0x0000_001C => VertexFormatInfo::new(4, FormatOutputClass::Float),
            0x0000_001D => VertexFormatInfo::new(8, FormatOutputClass::Float),
            0x0000_001E => VertexFormatInfo::new(12, FormatOutputClass::Float),
            0x0000_001F => VertexFormatInfo::new(16, FormatOutputClass::Float),
            0x0000_0020 => VertexFormatInfo::new(4, FormatOutputClass::Uint),
            0x0000_0021 => VertexFormatInfo::new(8, FormatOutputClass::Uint),
            0x0000_0022 => VertexFormatInfo::new(12, FormatOutputClass::Uint),
            0x0000_0023 => VertexFormatInfo::new(16, FormatOutputClass::Uint),
            0x0000_0024 => VertexFormatInfo::new(4, FormatOutputClass::Sint),
            0x0000_0025 => VertexFormatInfo::new(8, FormatOutputClass::Sint),
            0x0000_0026 => VertexFormatInfo::new(12, FormatOutputClass::Sint),
            0x0000_0027 => VertexFormatInfo::new(16, FormatOutputClass::Sint),
            0x0000_0028 | 0x0000_0029 => VertexFormatInfo::new(4, FormatOutputClass::Float),
            // Keep unknown future values conservative instead of guessing a smaller footprint.
            _ => VertexFormatInfo::new(16, FormatOutputClass::Float),
        }
    }
}

impl From<u32> for VertexFormat {
    fn from(value: u32) -> Self {
        Self::from_raw(value)
    }
}

impl From<i32> for VertexFormat {
    fn from(value: i32) -> Self {
        Self::from_raw(value as u32)
    }
}

impl From<VertexFormat> for u32 {
    fn from(value: VertexFormat) -> Self {
        value.raw()
    }
}

impl From<VertexFormat> for i32 {
    fn from(value: VertexFormat) -> Self {
        value.raw() as i32
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct VertexFormatInfo {
    pub(crate) byte_size: u64,
    pub(crate) output_class: FormatOutputClass,
}

impl VertexFormatInfo {
    pub(crate) const fn new(byte_size: u64, output_class: FormatOutputClass) -> Self {
        Self {
            byte_size,
            output_class,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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
