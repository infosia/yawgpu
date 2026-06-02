use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{HalTexture, HalTextureDescriptor, HalTextureFormat, HalTextureUsage};

use crate::copy::*;
use crate::device::FeatureSet;
use crate::extent::*;
use crate::format::*;
use crate::limits::*;
use crate::texture_view::*;

/// Enumerates texture usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureUsage(u64);

impl TextureUsage {
    /// Constant value for none.
    pub const NONE: Self = Self(0);
    /// Constant value for copy src.
    pub const COPY_SRC: Self = Self(1);
    /// Constant value for copy dst.
    pub const COPY_DST: Self = Self(2);
    /// Constant value for texture binding.
    pub const TEXTURE_BINDING: Self = Self(4);
    /// Constant value for storage binding.
    pub const STORAGE_BINDING: Self = Self(8);
    /// Constant value for render attachment.
    pub const RENDER_ATTACHMENT: Self = Self(16);
    /// Constant value for transient attachment.
    pub const TRANSIENT_ATTACHMENT: Self = Self(32);
    /// Mask of all known texture usage bits.
    pub const ALL: Self = Self(
        Self::COPY_SRC.0
            | Self::COPY_DST.0
            | Self::TEXTURE_BINDING.0
            | Self::STORAGE_BINDING.0
            | Self::RENDER_ATTACHMENT.0
            | Self::TRANSIENT_ATTACHMENT.0,
    );

    /// Constructs this object from bits retain.
    #[must_use]
    pub fn from_bits_retain(bits: u64) -> Self {
        Self(bits)
    }

    /// Returns the raw usage bitmask.
    #[must_use]
    pub fn bits(self) -> u64 {
        self.0
    }

    /// Returns whether every bit set in `other` is also set in `self`.
    #[must_use]
    pub(crate) fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for TextureUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Enumerates texture dimension values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureDimension {
    /// D1 variant.
    D1,
    /// D2 variant.
    D2,
    /// D3 variant.
    D3,
}

/// Describes texture descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureDescriptor {
    /// Usage.
    pub usage: TextureUsage,
    /// Dimension.
    pub dimension: TextureDimension,
    /// Size.
    pub size: Extent3d,
    /// Format.
    pub format: TextureFormat,
    /// Mip level count.
    pub mip_level_count: u32,
    /// Sample count.
    pub sample_count: u32,
    /// View formats.
    pub view_formats: Vec<TextureFormat>,
}

/// Stores texture data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct Texture {
    pub(crate) inner: Arc<TextureInner>,
}

/// Holds shared state for the texture handle.
#[derive(Debug)]
pub(crate) struct TextureInner {
    pub(crate) hal: Option<HalTexture>,
    pub(crate) usage: TextureUsage,
    pub(crate) dimension: TextureDimension,
    pub(crate) size: Extent3d,
    pub(crate) format: TextureFormat,
    pub(crate) features: FeatureSet,
    pub(crate) mip_level_count: u32,
    pub(crate) sample_count: u32,
    pub(crate) view_formats: Vec<TextureFormat>,
    pub(crate) state: Mutex<TextureState>,
}

/// Tracks the lifecycle state for texture.
#[derive(Debug)]
pub(crate) struct TextureState {
    pub(crate) is_error: bool,
    pub(crate) is_destroyed: bool,
}

impl Texture {
    /// Creates a new instance.
    pub(crate) fn new(
        descriptor: TextureDescriptor,
        hal: Option<HalTexture>,
        is_error: bool,
        features: FeatureSet,
    ) -> Self {
        Self {
            inner: Arc::new(TextureInner {
                hal,
                usage: descriptor.usage,
                dimension: descriptor.dimension,
                size: descriptor.size,
                format: descriptor.format,
                features,
                mip_level_count: descriptor.mip_level_count,
                sample_count: descriptor.sample_count,
                view_formats: descriptor.view_formats,
                state: Mutex::new(TextureState {
                    is_error,
                    is_destroyed: false,
                }),
            }),
        }
    }

    /// Constructs this object from the backend HAL object.
    #[must_use]
    pub fn from_hal(descriptor: TextureDescriptor, hal: HalTexture) -> Self {
        Self::new(descriptor, Some(hal), false, FeatureSet::new())
    }

    /// Returns the usage.
    #[must_use]
    pub fn usage(&self) -> TextureUsage {
        self.inner.usage
    }

    /// Returns the texture's dimensionality (1D / 2D / 3D).
    #[must_use]
    pub fn dimension(&self) -> TextureDimension {
        self.inner.dimension
    }

    /// Returns the size.
    #[must_use]
    pub fn size(&self) -> Extent3d {
        self.inner.size
    }

    /// Returns the format.
    #[must_use]
    pub fn format(&self) -> TextureFormat {
        self.inner.format
    }

    /// Returns the texture format capabilities for this texture's device features.
    #[must_use]
    pub(crate) fn format_caps(&self) -> Option<FormatCaps> {
        self.inner.format.caps(&self.inner.features)
    }

    /// Returns capabilities for a view format using this texture's device features.
    #[must_use]
    pub(crate) fn view_format_caps(&self, format: TextureFormat) -> Option<FormatCaps> {
        format.caps(&self.inner.features)
    }

    /// Returns mip level count.
    #[must_use]
    pub fn mip_level_count(&self) -> u32 {
        self.inner.mip_level_count
    }

    /// Returns sample count.
    #[must_use]
    pub fn sample_count(&self) -> u32 {
        self.inner.sample_count
    }

    /// Returns view formats.
    #[must_use]
    pub(crate) fn view_formats(&self) -> &[TextureFormat] {
        &self.inner.view_formats
    }

    /// A view format is compatible only when it equals the texture's format
    /// or is explicitly listed in the texture's `viewFormats`. There is no
    /// implicit sRGB-counterpart allowance — that mirrors Dawn
    /// `Texture.cpp` `ValidateCanViewTextureAs`.
    /// Returns true when this object is view format compatible.
    #[must_use]
    pub(crate) fn is_view_format_compatible(&self, view_format: TextureFormat) -> bool {
        view_format == self.format() || self.view_formats().contains(&view_format)
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.state.lock().is_error
    }

    /// Returns true when this object is destroyed.
    #[must_use]
    pub(crate) fn is_destroyed(&self) -> bool {
        self.inner.state.lock().is_destroyed
    }

    /// Returns true when both handles share the same backing object.
    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Returns the HAL.
    pub(crate) fn hal(&self) -> Option<HalTexture> {
        self.inner.hal.clone()
    }

    /// Destroys this object and releases backend resources.
    pub fn destroy(&self) {
        self.inner.state.lock().is_destroyed = true;
    }

    /// Creates a view of this texture, resolving defaulted fields against the texture.
    #[must_use]
    pub fn create_view(
        &self,
        descriptor: TextureViewDescriptor,
    ) -> (TextureView, Option<&'static str>) {
        let resolved = self.resolve_view_descriptor(descriptor);
        let error = if self.is_error() {
            Some("cannot create a view from an error texture")
        } else {
            validate_texture_view_descriptor(self, &resolved)
        };
        let is_error = error.is_some();
        (TextureView::new(self.clone(), resolved, is_error), error)
    }

    /// Fills in the defaulted fields of a texture-view descriptor against this texture.
    pub(crate) fn resolve_view_descriptor(
        &self,
        descriptor: TextureViewDescriptor,
    ) -> ResolvedTextureViewDescriptor {
        let base_mip_level = descriptor.base_mip_level;
        let base_array_layer = descriptor.base_array_layer;
        let mip_level_count = descriptor
            .mip_level_count
            .unwrap_or_else(|| self.mip_level_count().saturating_sub(base_mip_level));
        let dimension = descriptor
            .dimension
            .unwrap_or_else(|| match self.dimension() {
                TextureDimension::D1 => TextureViewDimension::D1,
                TextureDimension::D3 => TextureViewDimension::D3,
                TextureDimension::D2 if self.size().depth_or_array_layers == 1 => {
                    TextureViewDimension::D2
                }
                TextureDimension::D2 => TextureViewDimension::D2Array,
            });
        let array_layer_count = descriptor
            .array_layer_count
            .unwrap_or_else(|| match dimension {
                TextureViewDimension::D1 | TextureViewDimension::D2 | TextureViewDimension::D3 => 1,
                TextureViewDimension::Cube => 6,
                TextureViewDimension::D2Array | TextureViewDimension::CubeArray => self
                    .size()
                    .depth_or_array_layers
                    .saturating_sub(base_array_layer),
            });

        ResolvedTextureViewDescriptor {
            format: descriptor.format.unwrap_or_else(|| self.format()),
            dimension,
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            aspect: descriptor.aspect.unwrap_or(TextureAspect::All),
            usage: descriptor.usage.unwrap_or_else(|| self.usage()),
        }
    }

    /// Validates queue write and returns a descriptive error on failure.
    pub fn validate_queue_write(
        &self,
        mip_level: u32,
        origin: Origin3d,
        write_size: Extent3d,
        aspect: TextureAspect,
        layout: TexelCopyBufferLayout,
        data_size: u64,
    ) -> Result<(), String> {
        validate_queue_write_texture(
            self, mip_level, origin, write_size, aspect, layout, data_size,
        )
    }
}

/// Validates texture descriptor and returns a descriptive error on failure.
pub(crate) fn validate_texture_descriptor(
    descriptor: &TextureDescriptor,
    limits: Limits,
    features: &FeatureSet,
) -> Option<&'static str> {
    let usage = descriptor.usage;
    let size = descriptor.size;
    let multisampled = descriptor.sample_count > 1;

    if usage.bits() == 0 {
        return Some("texture usage must be non-zero");
    }
    if usage.bits() & !TextureUsage::ALL.bits() != 0 {
        return Some("texture usage contains unknown bits");
    }
    if usage.contains(TextureUsage::TRANSIENT_ATTACHMENT)
        && usage.bits()
            != (TextureUsage::RENDER_ATTACHMENT | TextureUsage::TRANSIENT_ATTACHMENT).bits()
    {
        return Some(
            "TransientAttachment texture usage requires exactly RenderAttachment and TransientAttachment",
        );
    }
    if descriptor.sample_count != 1 && descriptor.sample_count != 4 {
        return Some("texture sample count must be 1 or 4");
    }
    if multisampled && descriptor.mip_level_count != 1 {
        return Some("multisampled textures must have exactly one mip level");
    }
    if multisampled && descriptor.dimension != TextureDimension::D2 {
        return Some("multisampled textures must be 2D");
    }
    if multisampled && size.depth_or_array_layers != 1 {
        return Some("multisampled textures must have one array layer");
    }
    if multisampled && usage.contains(TextureUsage::STORAGE_BINDING) {
        return Some("multisampled textures cannot use StorageBinding");
    }
    if multisampled && !usage.contains(TextureUsage::RENDER_ATTACHMENT) {
        return Some("multisampled textures must use RenderAttachment");
    }
    if descriptor.mip_level_count == 0 {
        return Some("texture mipLevelCount must be at least 1");
    }
    if descriptor.mip_level_count > max_texture_mips(size, descriptor.dimension) {
        return Some("texture mipLevelCount exceeds the texture size");
    }
    if descriptor.dimension == TextureDimension::D1 && descriptor.mip_level_count != 1 {
        return Some("1D textures must have exactly one mip level");
    }
    if descriptor.dimension == TextureDimension::D2
        && size.depth_or_array_layers > limits.max_texture_array_layers
    {
        return Some("texture array layers exceed device limit");
    }
    match descriptor.dimension {
        TextureDimension::D1 => {
            if size.width == 0 || size.width > limits.max_texture_dimension_1d {
                return Some("1D texture width is out of range");
            }
            if size.height != 1 {
                return Some("1D texture height must be 1");
            }
            if size.depth_or_array_layers != 1 {
                return Some("1D texture depthOrArrayLayers must be 1");
            }
        }
        TextureDimension::D2 => {
            if size.width == 0 || size.width > limits.max_texture_dimension_2d {
                return Some("2D texture width is out of range");
            }
            if size.height == 0 || size.height > limits.max_texture_dimension_2d {
                return Some("2D texture height is out of range");
            }
            if size.depth_or_array_layers == 0 {
                return Some("2D texture depthOrArrayLayers must be at least 1");
            }
        }
        TextureDimension::D3 => {
            if size.width == 0 || size.width > limits.max_texture_dimension_3d {
                return Some("3D texture width is out of range");
            }
            if size.height == 0 || size.height > limits.max_texture_dimension_3d {
                return Some("3D texture height is out of range");
            }
            if size.depth_or_array_layers == 0
                || size.depth_or_array_layers > limits.max_texture_dimension_3d
            {
                return Some("3D texture depth is out of range");
            }
        }
    }
    if usage.contains(TextureUsage::RENDER_ATTACHMENT)
        && descriptor.dimension == TextureDimension::D1
    {
        return Some("RenderAttachment textures must not be 1D");
    }
    if usage.contains(TextureUsage::TRANSIENT_ATTACHMENT)
        && descriptor.dimension != TextureDimension::D2
    {
        return Some("TransientAttachment textures must be 2D");
    }
    let Some(format_caps) = descriptor.format.caps(features) else {
        return Some("texture format must not be Undefined");
    };
    if format_caps.is_compressed {
        if !size.width.is_multiple_of(format_caps.block_w)
            || !size.height.is_multiple_of(format_caps.block_h)
        {
            return Some("compressed texture size must be block-aligned");
        }
        if descriptor.dimension == TextureDimension::D1 {
            return Some("compressed textures must not be 1D");
        }
        if descriptor.dimension == TextureDimension::D3 {
            if descriptor.format.is_bc_compressed() {
                if !features.contains(&crate::adapter::Feature::TextureCompressionBcSliced3d) {
                    return Some(
                        "3D BC compressed textures require texture-compression-bc-sliced-3d",
                    );
                }
            } else if descriptor.format.is_astc_compressed() {
                if !features.contains(&crate::adapter::Feature::TextureCompressionAstcSliced3d) {
                    return Some(
                        "3D ASTC compressed textures require texture-compression-astc-sliced-3d",
                    );
                }
            } else if descriptor.format.is_etc2_compressed() {
                return Some("ETC2/EAC compressed textures must be 2D");
            } else {
                return Some("compressed textures must be 2D unless sliced-3d is supported");
            }
        }
    }
    for view_format in &descriptor.view_formats {
        if view_format.caps(features).is_none() {
            return Some("texture viewFormats must not contain Undefined");
        }
        if !texture_view_format_compatible(descriptor.format, *view_format) {
            return Some("texture viewFormats must be compatible with the texture format");
        }
    }
    if multisampled && !format_caps.multisample_capable {
        return Some("multisampled texture format must support multisampling");
    }
    if (format_caps.aspects.depth || format_caps.aspects.stencil)
        && descriptor.dimension != TextureDimension::D2
    {
        return Some("depth/stencil texture formats must be 2D");
    }
    if usage.contains(TextureUsage::RENDER_ATTACHMENT) && !format_caps.renderable {
        return Some("RenderAttachment texture format must be renderable");
    }
    if usage.contains(TextureUsage::STORAGE_BINDING) && !format_caps.storage_capable {
        return Some("StorageBinding texture format must support storage usage");
    }
    None
}

fn texture_view_format_compatible(format: TextureFormat, view_format: TextureFormat) -> bool {
    format == view_format || format.srgb_pair() == Some(view_format)
}

/// Validates queue write texture and returns a descriptive error on failure.
pub(crate) fn validate_queue_write_texture(
    texture: &Texture,
    mip_level: u32,
    origin: Origin3d,
    write_size: Extent3d,
    aspect: TextureAspect,
    layout: TexelCopyBufferLayout,
    data_size: u64,
) -> Result<(), String> {
    if !texture.usage().contains(TextureUsage::COPY_DST) {
        return Err("queue texture write destination usage must include CopyDst".to_owned());
    }
    if texture.is_error() || texture.is_destroyed() {
        return Err("queue texture write destination must be a valid live texture".to_owned());
    }
    if texture.sample_count() != 1 {
        return Err("queue texture write destination sampleCount must be one".to_owned());
    }
    if mip_level >= texture.mip_level_count() {
        return Err("queue texture write mipLevel is out of range".to_owned());
    }

    let Some(format_caps) = texture.format_caps() else {
        return Err("queue texture write format must not be Undefined".to_owned());
    };
    match aspect {
        TextureAspect::All => {}
        TextureAspect::DepthOnly if !format_caps.aspects.depth => {
            return Err("DepthOnly texture writes require a depth format".to_owned());
        }
        TextureAspect::StencilOnly if !format_caps.aspects.stencil => {
            return Err("StencilOnly texture writes require a stencil format".to_owned());
        }
        TextureAspect::DepthOnly | TextureAspect::StencilOnly => {}
    }

    let subresource = texture.subresource_size(mip_level);
    let empty_write =
        write_size.width == 0 || write_size.height == 0 || write_size.depth_or_array_layers == 0;
    if origin
        .x
        .checked_add(write_size.width)
        .is_none_or(|end| end > subresource.width)
        || origin
            .y
            .checked_add(write_size.height)
            .is_none_or(|end| end > subresource.height)
        || origin
            .z
            .checked_add(write_size.depth_or_array_layers)
            .is_none_or(|end| end > subresource.depth_or_array_layers)
    {
        return Err("queue texture write range exceeds the texture subresource".to_owned());
    }
    if texture.dimension() == TextureDimension::D2
        && !empty_write
        && write_size.depth_or_array_layers != 1
    {
        return Err(
            "queue texture writes to 2D textures require depthOrArrayLayers to be one".to_owned(),
        );
    }
    if !origin.x.is_multiple_of(format_caps.block_w)
        || !origin.y.is_multiple_of(format_caps.block_h)
    {
        return Err("queue texture write origin must be texel block aligned".to_owned());
    }
    if !write_size.width.is_multiple_of(format_caps.block_w)
        || !write_size.height.is_multiple_of(format_caps.block_h)
    {
        return Err("queue texture write size must be texel block aligned".to_owned());
    }
    if (format_caps.aspects.depth || format_caps.aspects.stencil)
        && !empty_write
        && (!crate::command_encoder::origin_is_zero(origin) || write_size != subresource)
    {
        return Err(
            "queue texture write depth/stencil copies must cover the full subresource".to_owned(),
        );
    }

    let required_bytes = validate_texel_copy_layout(
        format_caps,
        aspect,
        write_size,
        layout,
        "queue texture write",
        false,
    )?;
    let required_end = layout
        .offset
        .checked_add(required_bytes)
        .ok_or("queue texture write data range overflows")?;
    if required_end > data_size {
        return Err("queue texture write dataSize is too small".to_owned());
    }

    Ok(())
}

impl Texture {
    /// Returns subresource size.
    pub(crate) fn subresource_size(&self, mip_level: u32) -> Extent3d {
        let size = self.size();
        let mip = |value: u32| value.checked_shr(mip_level).unwrap_or(0).max(1);
        Extent3d {
            width: mip(size.width),
            height: match self.dimension() {
                TextureDimension::D1 => 1,
                TextureDimension::D2 | TextureDimension::D3 => mip(size.height),
            },
            depth_or_array_layers: match self.dimension() {
                TextureDimension::D1 => 1,
                TextureDimension::D2 => size.depth_or_array_layers,
                TextureDimension::D3 => mip(size.depth_or_array_layers),
            },
        }
    }
}

/// Validates texel copy layout and returns a descriptive error on failure.
pub(crate) fn validate_texel_copy_layout(
    format_caps: FormatCaps,
    aspect: TextureAspect,
    write_size: Extent3d,
    layout: TexelCopyBufferLayout,
    label: &str,
    require_bytes_per_row_alignment: bool,
) -> Result<u64, String> {
    let width_blocks = div_ceil_u32(write_size.width, format_caps.block_w);
    let height_blocks = div_ceil_u32(write_size.height, format_caps.block_h);
    let depth = write_size.depth_or_array_layers;
    let block_size = texel_copy_block_size(format_caps, aspect);
    let last_row_bytes = u64::from(width_blocks)
        .checked_mul(u64::from(block_size))
        .ok_or_else(|| format!("{label} row byte size overflows"))?;

    if let Some(bytes_per_row) = layout.bytes_per_row {
        if require_bytes_per_row_alignment && !bytes_per_row.is_multiple_of(256) {
            return Err(format!("{label} bytesPerRow must be 256-byte aligned"));
        }
        if u64::from(bytes_per_row) < last_row_bytes {
            return Err(format!("{label} bytesPerRow is too small"));
        }
    } else if height_blocks > 1 || depth > 1 {
        return Err(format!(
            "{label} bytesPerRow is required for multi-row copies"
        ));
    }

    if let Some(rows_per_image) = layout.rows_per_image {
        if rows_per_image < height_blocks {
            return Err(format!("{label} rowsPerImage is too small"));
        }
    } else if depth > 1 {
        return Err(format!(
            "{label} rowsPerImage is required for multi-image copies"
        ));
    }

    required_bytes_in_texel_copy(
        layout.bytes_per_row,
        layout.rows_per_image,
        height_blocks,
        depth,
        last_row_bytes,
        label,
    )
}

/// Returns required bytes in texel copy.
pub(crate) fn required_bytes_in_texel_copy(
    bytes_per_row: Option<u32>,
    rows_per_image: Option<u32>,
    height_blocks: u32,
    depth: u32,
    last_row_bytes: u64,
    label: &str,
) -> Result<u64, String> {
    if last_row_bytes == 0 || height_blocks == 0 || depth == 0 {
        return Ok(0);
    }

    let bytes_per_row = u64::from(bytes_per_row.unwrap_or(0));
    let rows_per_image = u64::from(rows_per_image.unwrap_or(height_blocks));
    let image_offset_rows = rows_per_image
        .checked_mul(u64::from(depth.saturating_sub(1)))
        .ok_or_else(|| format!("{label} required byte size overflows"))?;
    let row_offset_rows = u64::from(height_blocks.saturating_sub(1));
    let offset_rows = image_offset_rows
        .checked_add(row_offset_rows)
        .ok_or_else(|| format!("{label} required byte size overflows"))?;
    let offset_bytes = bytes_per_row
        .checked_mul(offset_rows)
        .ok_or_else(|| format!("{label} required byte size overflows"))?;
    offset_bytes
        .checked_add(last_row_bytes)
        .ok_or_else(|| format!("{label} required byte size overflows"))
}

/// Returns texel copy block size.
pub(crate) fn texel_copy_block_size(format_caps: FormatCaps, aspect: TextureAspect) -> u32 {
    if aspect == TextureAspect::StencilOnly {
        1
    } else {
        format_caps.texel_block_size
    }
}

/// Returns div ceil u32.
pub(crate) fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    if value == 0 {
        0
    } else {
        u64::from(value).div_ceil(u64::from(divisor)) as u32
    }
}

/// Returns HAL texture descriptor.
pub(crate) fn hal_texture_descriptor(descriptor: &TextureDescriptor) -> HalTextureDescriptor {
    HalTextureDescriptor {
        format: hal_texture_format(descriptor.format),
        width: descriptor.size.width,
        height: descriptor.size.height,
        depth_or_array_layers: descriptor.size.depth_or_array_layers,
        mip_level_count: descriptor.mip_level_count,
        sample_count: descriptor.sample_count,
        usage: hal_texture_usage(descriptor.usage),
    }
}

/// Returns HAL texture format.
pub(crate) fn hal_texture_format(format: TextureFormat) -> HalTextureFormat {
    match format.raw() {
        TextureFormat::R8_UNORM => HalTextureFormat::R8Unorm,
        TextureFormat::R8_SNORM => HalTextureFormat::R8Snorm,
        TextureFormat::R8_UINT => HalTextureFormat::R8Uint,
        TextureFormat::R8_SINT => HalTextureFormat::R8Sint,
        TextureFormat::R16_UNORM => HalTextureFormat::R16Unorm,
        TextureFormat::R16_SNORM => HalTextureFormat::R16Snorm,
        TextureFormat::R16_UINT => HalTextureFormat::R16Uint,
        TextureFormat::R16_SINT => HalTextureFormat::R16Sint,
        TextureFormat::R16_FLOAT => HalTextureFormat::R16Float,
        TextureFormat::RG8_UNORM => HalTextureFormat::Rg8Unorm,
        TextureFormat::RG8_SNORM => HalTextureFormat::Rg8Snorm,
        TextureFormat::RG8_UINT => HalTextureFormat::Rg8Uint,
        TextureFormat::RG8_SINT => HalTextureFormat::Rg8Sint,
        TextureFormat::RG16_UNORM => HalTextureFormat::Rg16Unorm,
        TextureFormat::RG16_SNORM => HalTextureFormat::Rg16Snorm,
        TextureFormat::RG16_UINT => HalTextureFormat::Rg16Uint,
        TextureFormat::RG16_SINT => HalTextureFormat::Rg16Sint,
        TextureFormat::RG16_FLOAT => HalTextureFormat::Rg16Float,
        TextureFormat::R32_UINT => HalTextureFormat::R32Uint,
        TextureFormat::R32_SINT => HalTextureFormat::R32Sint,
        TextureFormat::R32_FLOAT => HalTextureFormat::R32Float,
        TextureFormat::RG32_UINT => HalTextureFormat::Rg32Uint,
        TextureFormat::RG32_SINT => HalTextureFormat::Rg32Sint,
        TextureFormat::RG32_FLOAT => HalTextureFormat::Rg32Float,
        TextureFormat::RGBA8_UNORM => HalTextureFormat::Rgba8Unorm,
        TextureFormat::RGBA8_UNORM_SRGB => HalTextureFormat::Rgba8UnormSrgb,
        TextureFormat::RGBA8_SNORM => HalTextureFormat::Rgba8Snorm,
        TextureFormat::RGBA8_UINT => HalTextureFormat::Rgba8Uint,
        TextureFormat::RGBA8_SINT => HalTextureFormat::Rgba8Sint,
        TextureFormat::BGRA8_UNORM => HalTextureFormat::Bgra8Unorm,
        TextureFormat::BGRA8_UNORM_SRGB => HalTextureFormat::Bgra8UnormSrgb,
        TextureFormat::RGB10A2_UINT => HalTextureFormat::Rgb10a2Uint,
        TextureFormat::RGB10A2_UNORM => HalTextureFormat::Rgb10a2Unorm,
        TextureFormat::RG11B10_UFLOAT => HalTextureFormat::Rg11b10Ufloat,
        TextureFormat::RGB9E5_UFLOAT => HalTextureFormat::Rgb9e5Ufloat,
        TextureFormat::RGBA16_UNORM => HalTextureFormat::Rgba16Unorm,
        TextureFormat::RGBA16_SNORM => HalTextureFormat::Rgba16Snorm,
        TextureFormat::RGBA16_UINT => HalTextureFormat::Rgba16Uint,
        TextureFormat::RGBA16_SINT => HalTextureFormat::Rgba16Sint,
        TextureFormat::RGBA16_FLOAT => HalTextureFormat::Rgba16Float,
        TextureFormat::RGBA32_UINT => HalTextureFormat::Rgba32Uint,
        TextureFormat::RGBA32_SINT => HalTextureFormat::Rgba32Sint,
        TextureFormat::RGBA32_FLOAT => HalTextureFormat::Rgba32Float,
        TextureFormat::STENCIL8 => HalTextureFormat::Stencil8,
        TextureFormat::DEPTH16_UNORM => HalTextureFormat::Depth16Unorm,
        TextureFormat::DEPTH24_PLUS => HalTextureFormat::Depth24Plus,
        TextureFormat::DEPTH24_PLUS_STENCIL8 => HalTextureFormat::Depth24PlusStencil8,
        TextureFormat::DEPTH32_FLOAT => HalTextureFormat::Depth32Float,
        TextureFormat::DEPTH32_FLOAT_STENCIL8 => HalTextureFormat::Depth32FloatStencil8,
        _ => HalTextureFormat::Unsupported,
    }
}

/// Returns HAL texture usage.
pub(crate) fn hal_texture_usage(usage: TextureUsage) -> HalTextureUsage {
    HalTextureUsage {
        copy_src: usage.contains(TextureUsage::COPY_SRC),
        copy_dst: usage.contains(TextureUsage::COPY_DST),
        texture_binding: usage.contains(TextureUsage::TEXTURE_BINDING),
        storage_binding: usage.contains(TextureUsage::STORAGE_BINDING),
        render_attachment: usage.contains(TextureUsage::RENDER_ATTACHMENT),
    }
}

/// Returns max texture mips.
pub(crate) fn max_texture_mips(size: Extent3d, dimension: TextureDimension) -> u32 {
    let mut max_extent = size.width;
    if matches!(dimension, TextureDimension::D2 | TextureDimension::D3) {
        max_extent = max_extent.max(size.height);
    }
    if dimension == TextureDimension::D3 {
        max_extent = max_extent.max(size.depth_or_array_layers);
    }

    let mut levels = 0;
    while max_extent > 0 {
        levels += 1;
        max_extent /= 2;
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;

    #[test]
    fn texture_usage_from_bits_retain_round_trips_known_and_unknown_bits() {
        let raw = (TextureUsage::COPY_SRC | TextureUsage::RENDER_ATTACHMENT).bits() | (1_u64 << 40);
        let usage = TextureUsage::from_bits_retain(raw);

        assert_eq!(usage.bits(), raw);
    }

    #[test]
    fn hal_texture_format_maps_uncompressed_color_formats() {
        let cases = [
            (TextureFormat::R8_UNORM, HalTextureFormat::R8Unorm),
            (TextureFormat::R8_SNORM, HalTextureFormat::R8Snorm),
            (TextureFormat::R8_UINT, HalTextureFormat::R8Uint),
            (TextureFormat::R8_SINT, HalTextureFormat::R8Sint),
            (TextureFormat::R16_UNORM, HalTextureFormat::R16Unorm),
            (TextureFormat::R16_SNORM, HalTextureFormat::R16Snorm),
            (TextureFormat::R16_UINT, HalTextureFormat::R16Uint),
            (TextureFormat::R16_SINT, HalTextureFormat::R16Sint),
            (TextureFormat::R16_FLOAT, HalTextureFormat::R16Float),
            (TextureFormat::RG8_UNORM, HalTextureFormat::Rg8Unorm),
            (TextureFormat::RG8_SNORM, HalTextureFormat::Rg8Snorm),
            (TextureFormat::RG8_UINT, HalTextureFormat::Rg8Uint),
            (TextureFormat::RG8_SINT, HalTextureFormat::Rg8Sint),
            (TextureFormat::RG16_UNORM, HalTextureFormat::Rg16Unorm),
            (TextureFormat::RG16_SNORM, HalTextureFormat::Rg16Snorm),
            (TextureFormat::RG16_UINT, HalTextureFormat::Rg16Uint),
            (TextureFormat::RG16_SINT, HalTextureFormat::Rg16Sint),
            (TextureFormat::RG16_FLOAT, HalTextureFormat::Rg16Float),
            (TextureFormat::R32_UINT, HalTextureFormat::R32Uint),
            (TextureFormat::R32_SINT, HalTextureFormat::R32Sint),
            (TextureFormat::R32_FLOAT, HalTextureFormat::R32Float),
            (TextureFormat::RG32_UINT, HalTextureFormat::Rg32Uint),
            (TextureFormat::RG32_SINT, HalTextureFormat::Rg32Sint),
            (TextureFormat::RG32_FLOAT, HalTextureFormat::Rg32Float),
            (TextureFormat::RGBA8_UNORM, HalTextureFormat::Rgba8Unorm),
            (
                TextureFormat::RGBA8_UNORM_SRGB,
                HalTextureFormat::Rgba8UnormSrgb,
            ),
            (TextureFormat::RGBA8_SNORM, HalTextureFormat::Rgba8Snorm),
            (TextureFormat::RGBA8_UINT, HalTextureFormat::Rgba8Uint),
            (TextureFormat::RGBA8_SINT, HalTextureFormat::Rgba8Sint),
            (TextureFormat::BGRA8_UNORM, HalTextureFormat::Bgra8Unorm),
            (
                TextureFormat::BGRA8_UNORM_SRGB,
                HalTextureFormat::Bgra8UnormSrgb,
            ),
            (TextureFormat::RGB10A2_UINT, HalTextureFormat::Rgb10a2Uint),
            (TextureFormat::RGB10A2_UNORM, HalTextureFormat::Rgb10a2Unorm),
            (
                TextureFormat::RG11B10_UFLOAT,
                HalTextureFormat::Rg11b10Ufloat,
            ),
            (TextureFormat::RGB9E5_UFLOAT, HalTextureFormat::Rgb9e5Ufloat),
            (TextureFormat::RGBA16_UNORM, HalTextureFormat::Rgba16Unorm),
            (TextureFormat::RGBA16_SNORM, HalTextureFormat::Rgba16Snorm),
            (TextureFormat::RGBA16_UINT, HalTextureFormat::Rgba16Uint),
            (TextureFormat::RGBA16_SINT, HalTextureFormat::Rgba16Sint),
            (TextureFormat::RGBA16_FLOAT, HalTextureFormat::Rgba16Float),
            (TextureFormat::RGBA32_UINT, HalTextureFormat::Rgba32Uint),
            (TextureFormat::RGBA32_SINT, HalTextureFormat::Rgba32Sint),
            (TextureFormat::RGBA32_FLOAT, HalTextureFormat::Rgba32Float),
        ];

        for (raw, expected) in cases {
            assert_eq!(hal_texture_format(TextureFormat::from_raw(raw)), expected);
        }
    }

    #[test]
    fn validate_texture_descriptor_rejects_unknown_usage_bits() {
        let mut descriptor = texture_descriptor_4x4();
        descriptor.usage =
            TextureUsage::from_bits_retain(TextureUsage::TEXTURE_BINDING.bits() | (1_u64 << 40));

        assert_eq!(
            validate_texture_descriptor(&descriptor, Limits::DEFAULT, &FeatureSet::new()),
            Some("texture usage contains unknown bits")
        );
    }

    #[test]
    fn validate_texture_descriptor_rejects_invalid_transient_usage_combinations() {
        let mut descriptor = texture_descriptor_4x4();
        descriptor.usage = TextureUsage::TRANSIENT_ATTACHMENT;
        assert_eq!(
            validate_texture_descriptor(&descriptor, Limits::DEFAULT, &FeatureSet::new()),
            Some(
                "TransientAttachment texture usage requires exactly RenderAttachment and TransientAttachment"
            )
        );

        descriptor.usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::TRANSIENT_ATTACHMENT;
        assert_eq!(
            validate_texture_descriptor(&descriptor, Limits::DEFAULT, &FeatureSet::new()),
            None
        );

        descriptor.dimension = TextureDimension::D3;
        descriptor.size.depth_or_array_layers = 4;
        assert_eq!(
            validate_texture_descriptor(&descriptor, Limits::DEFAULT, &FeatureSet::new()),
            Some("TransientAttachment textures must be 2D")
        );
    }

    #[test]
    fn validate_texture_descriptor_allows_3d_render_attachments_but_not_1d() {
        let mut descriptor = texture_descriptor_4x4();
        descriptor.usage = TextureUsage::RENDER_ATTACHMENT;
        descriptor.dimension = TextureDimension::D3;
        descriptor.size.depth_or_array_layers = 4;
        assert_eq!(
            validate_texture_descriptor(&descriptor, Limits::DEFAULT, &FeatureSet::new()),
            None
        );

        descriptor.dimension = TextureDimension::D1;
        descriptor.size.height = 1;
        descriptor.size.depth_or_array_layers = 1;
        assert_eq!(
            validate_texture_descriptor(&descriptor, Limits::DEFAULT, &FeatureSet::new()),
            Some("RenderAttachment textures must not be 1D")
        );
    }

    #[test]
    fn validate_texture_descriptor_rejects_incompatible_view_formats_and_etc2_3d() {
        let mut descriptor = texture_descriptor_4x4();
        descriptor.view_formats = vec![TextureFormat::from_raw(TextureFormat::RGBA8_UNORM_SRGB)];
        assert_eq!(
            validate_texture_descriptor(&descriptor, Limits::DEFAULT, &FeatureSet::new()),
            None
        );

        descriptor.view_formats = vec![TextureFormat::from_raw(TextureFormat::R8_UNORM)];
        assert_eq!(
            validate_texture_descriptor(&descriptor, Limits::DEFAULT, &FeatureSet::new()),
            Some("texture viewFormats must be compatible with the texture format")
        );

        let mut features = FeatureSet::new();
        features.insert(Feature::TextureCompressionEtc2);
        descriptor = texture_descriptor_4x4();
        descriptor.format = TextureFormat::from_raw(TextureFormat::ETC2_RGBA8_UNORM);
        descriptor.dimension = TextureDimension::D3;
        descriptor.size = Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 4,
        };
        assert_eq!(
            validate_texture_descriptor(&descriptor, Limits::DEFAULT, &features),
            Some("ETC2/EAC compressed textures must be 2D")
        );
    }

    #[test]
    fn validate_texture_descriptor_rejects_compressed_block_misaligned_size() {
        let mut features = FeatureSet::new();
        features.insert(Feature::TextureCompressionBc);

        let mut descriptor = texture_descriptor_4x4();
        descriptor.format = TextureFormat::from_raw(TextureFormat::BC1_RGBA_UNORM);
        descriptor.size = Extent3d {
            width: 5,
            height: 4,
            depth_or_array_layers: 1,
        };

        assert_eq!(
            validate_texture_descriptor(&descriptor, Limits::DEFAULT, &features),
            Some("compressed texture size must be block-aligned")
        );
    }

    #[test]
    fn resolve_view_descriptor_defaults_array_layers_from_resolved_view_dimension() {
        let texture = noop_device().create_texture(TextureDescriptor {
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 12,
            },
            ..layered_mipped_texture_descriptor()
        });

        let resolved = texture.resolve_view_descriptor(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 3,
            array_layer_count: None,
            aspect: None,
            usage: None,
        });
        assert_eq!(resolved.dimension, TextureViewDimension::D2);
        assert_eq!(resolved.array_layer_count, 1);

        let resolved = texture.resolve_view_descriptor(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::Cube),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: None,
        });
        assert_eq!(resolved.dimension, TextureViewDimension::Cube);
        assert_eq!(resolved.array_layer_count, 6);

        let resolved = texture.resolve_view_descriptor(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D2Array),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 5,
            array_layer_count: None,
            aspect: None,
            usage: None,
        });
        assert_eq!(resolved.dimension, TextureViewDimension::D2Array);
        assert_eq!(resolved.array_layer_count, 7);

        let resolved = texture.resolve_view_descriptor(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::CubeArray),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 6,
            array_layer_count: None,
            aspect: None,
            usage: None,
        });
        assert_eq!(resolved.dimension, TextureViewDimension::CubeArray);
        assert_eq!(resolved.array_layer_count, 6);
    }

    #[test]
    fn texture_from_hal_and_descriptor_accessors_round_trip() {
        let descriptor = texture_descriptor_4x4();
        let texture = Texture::from_hal(
            descriptor.clone(),
            yawgpu_hal::HalTexture::Noop(yawgpu_hal::noop::NoopTexture),
        );

        assert_eq!(texture.usage(), descriptor.usage);
        assert_eq!(texture.dimension(), descriptor.dimension);
        assert_eq!(texture.size(), descriptor.size);
        assert_eq!(texture.format(), descriptor.format);
        assert_eq!(texture.mip_level_count(), descriptor.mip_level_count);
        assert_eq!(texture.sample_count(), descriptor.sample_count);
        assert!(!texture.is_error());
    }

    #[test]
    fn texture_is_error_same_destroy_create_view_and_validate_queue_write() {
        let texture = noop_texture();
        let other = noop_texture();
        let clone = texture.clone();

        assert!(texture.same(&clone));
        assert!(!texture.same(&other));

        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: None,
        });
        assert_eq!(error, None);
        assert!(!view.is_error());
        assert_eq!(view.format(), texture.format());

        assert_eq!(
            texture.validate_queue_write(
                0,
                Origin3d { x: 0, y: 0, z: 0 },
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                TextureAspect::All,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
                4,
            ),
            Ok(())
        );
        assert_eq!(
            texture.validate_queue_write(
                0,
                Origin3d { x: 4, y: 0, z: 0 },
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                TextureAspect::All,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
                4,
            ),
            Err("queue texture write range exceeds the texture subresource".to_owned())
        );

        texture.destroy();
        texture.destroy();
        assert_eq!(
            texture.validate_queue_write(
                0,
                Origin3d { x: 0, y: 0, z: 0 },
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                TextureAspect::All,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
                4,
            ),
            Err("queue texture write destination must be a valid live texture".to_owned())
        );
    }

    #[test]
    fn texture_error_texture_reports_is_error_and_error_view() {
        let device = noop_device();
        let mut invalid = texture_descriptor_4x4();
        invalid.size.width = 0;

        device.push_error_scope(ErrorFilter::Validation);
        let texture = device.create_texture(invalid);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid texture should be scoped");
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: None,
        });

        assert_eq!(scoped.kind, ErrorKind::Validation);
        assert_eq!(scoped.message, "2D texture width is out of range");
        assert!(texture.is_error());
        assert!(view.is_error());
        assert_eq!(error, Some("cannot create a view from an error texture"));
    }
}
