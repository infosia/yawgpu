use std::sync::Arc;

use yawgpu_hal::{HalExtent3d, HalOrigin3d};

use crate::buffer::*;
use crate::device::Device;
use crate::extent::*;
use crate::format::*;
use crate::texture::*;
use crate::texture_view::*;

/// Stores layout metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TexelCopyBufferLayout {
    /// Offset.
    pub offset: u64,
    /// Bytes per row.
    pub bytes_per_row: Option<u32>,
    /// Rows per image.
    pub rows_per_image: Option<u32>,
}

/// Stores info metadata.
#[derive(Debug, Clone)]
pub struct TexelCopyBufferInfo {
    /// Buffer.
    pub buffer: Arc<Buffer>,
    /// Device that owns the buffer, when known by the API boundary.
    pub device: Option<Device>,
    /// Layout.
    pub layout: TexelCopyBufferLayout,
}

/// Stores info metadata.
#[derive(Debug, Clone)]
pub struct TexelCopyTextureInfo {
    /// Texture.
    pub texture: Arc<Texture>,
    /// Mip level.
    pub mip_level: u32,
    /// Origin.
    pub origin: Origin3d,
    /// Aspect.
    pub aspect: TextureAspect,
}

/// Enumerates load op values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOp {
    /// Undefined variant.
    Undefined,
    /// Load variant.
    Load,
    /// Clear variant.
    Clear,
}

/// Enumerates store op values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOp {
    /// Undefined variant.
    Undefined,
    /// Store variant.
    Store,
    /// Discard variant.
    Discard,
}

/// Stores color metadata.
#[derive(Debug, Clone, Copy)]
pub struct Color {
    /// R.
    pub r: f64,
    /// G.
    pub g: f64,
    /// B.
    pub b: f64,
    /// A.
    pub a: f64,
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
    if depth == 0 {
        return Ok(0);
    }

    let bytes_per_row = u64::from(bytes_per_row.unwrap_or(0));
    let rows_per_image = u64::from(rows_per_image.unwrap_or(height_blocks));
    let mut required = 0u64;
    if depth > 1 {
        let image_bytes = bytes_per_row
            .checked_mul(rows_per_image)
            .and_then(|n| n.checked_mul(u64::from(depth - 1)))
            .ok_or_else(|| format!("{label} required byte size overflows"))?;
        required = required
            .checked_add(image_bytes)
            .ok_or_else(|| format!("{label} required byte size overflows"))?;
    }
    if height_blocks > 0 {
        let row_bytes = bytes_per_row
            .checked_mul(u64::from(height_blocks - 1))
            .ok_or_else(|| format!("{label} required byte size overflows"))?;
        required = required
            .checked_add(row_bytes)
            .and_then(|n| n.checked_add(last_row_bytes))
            .ok_or_else(|| format!("{label} required byte size overflows"))?;
    }
    Ok(required)
}

#[cfg(test)]
mod required_bytes_in_texel_copy_tests {
    use super::*;

    #[test]
    fn required_bytes_width_blocks_zero_keeps_stride_bytes() {
        let required = required_bytes_in_texel_copy(Some(256), Some(4), 4, 5, 0, "zero-width copy")
            .expect("required bytes should not overflow");

        assert_eq!(required, 4864);
    }

    #[test]
    fn required_bytes_height_blocks_zero_keeps_inter_image_bytes() {
        let required =
            required_bytes_in_texel_copy(Some(256), Some(4), 0, 5, 12, "zero-height copy")
                .expect("required bytes should not overflow");

        assert_eq!(required, 4096);
    }

    #[test]
    fn required_bytes_depth_zero_is_zero() {
        let required =
            required_bytes_in_texel_copy(Some(256), Some(4), 3, 0, 12, "zero-depth copy")
                .expect("zero-depth copy should not overflow");

        assert_eq!(required, 0);
    }

    #[test]
    fn required_bytes_multi_row_multi_image_uses_spec_formula() {
        let required =
            required_bytes_in_texel_copy(Some(256), Some(4), 3, 2, 12, "multi-image copy")
                .expect("required bytes should not overflow");

        assert_eq!(required, 1548);
    }
}

/// Returns the per-texel byte size of the *aspect* being copied to/from a buffer.
///
/// A single aspect of a depth/stencil format is laid out in the buffer at that
/// plane's stride, not the whole format's block size:
/// - `StencilOnly`: the stencil plane is always 1 byte.
/// - `DepthOnly` of a *packed* depth+stencil format: the depth plane is the
///   block size minus the 1-byte stencil plane (e.g. `depth32float-stencil8`
///   has `texel_block_size == 5`, so its depth plane is 4 bytes). Depth-only
///   formats already report their depth plane as the block size.
/// - Otherwise: the whole-format block size.
pub(crate) fn texel_copy_block_size(format_caps: FormatCaps, aspect: TextureAspect) -> u32 {
    match aspect {
        TextureAspect::StencilOnly => 1,
        TextureAspect::DepthOnly if format_caps.aspects.depth && format_caps.aspects.stencil => {
            format_caps.texel_block_size.saturating_sub(1)
        }
        _ => format_caps.texel_block_size,
    }
}

/// Returns true when a depth/stencil texture format supports the copy direction and aspect.
pub(crate) fn depth_stencil_copy_allowed(
    format: TextureFormat,
    aspect: TextureAspect,
    writing_texture: bool,
) -> bool {
    match format.raw() {
        TextureFormat::STENCIL8 => {
            matches!(aspect, TextureAspect::All | TextureAspect::StencilOnly)
        }
        TextureFormat::DEPTH16_UNORM => {
            matches!(aspect, TextureAspect::All | TextureAspect::DepthOnly)
        }
        TextureFormat::DEPTH32_FLOAT => {
            !writing_texture && matches!(aspect, TextureAspect::All | TextureAspect::DepthOnly)
        }
        TextureFormat::DEPTH24_PLUS => false,
        TextureFormat::DEPTH24_PLUS_STENCIL8 => aspect == TextureAspect::StencilOnly,
        TextureFormat::DEPTH32_FLOAT_STENCIL8 => {
            if writing_texture {
                aspect == TextureAspect::StencilOnly
            } else {
                matches!(
                    aspect,
                    TextureAspect::DepthOnly | TextureAspect::StencilOnly
                )
            }
        }
        _ => true,
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

/// Returns HAL origin.
pub(crate) fn hal_origin(origin: Origin3d) -> HalOrigin3d {
    HalOrigin3d {
        x: origin.x,
        y: origin.y,
        z: origin.z,
    }
}

/// Returns HAL extent.
pub(crate) fn hal_extent(extent: Extent3d) -> HalExtent3d {
    HalExtent3d {
        width: extent.width,
        height: extent.height,
        depth_or_array_layers: extent.depth_or_array_layers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_aspect_of_packed_format_uses_depth_plane_size() {
        // depth32float-stencil8: 5-byte block = 4-byte depth plane + 1-byte stencil.
        let caps = FormatCaps::depth_stencil(5);
        assert_eq!(texel_copy_block_size(caps, TextureAspect::DepthOnly), 4);
        assert_eq!(texel_copy_block_size(caps, TextureAspect::StencilOnly), 1);
        assert_eq!(texel_copy_block_size(caps, TextureAspect::All), 5);
    }

    #[test]
    fn depth_only_format_block_size_equals_block() {
        // For a non-packed depth format the depth plane is the whole block.
        assert_eq!(
            texel_copy_block_size(FormatCaps::depth(4), TextureAspect::DepthOnly),
            4
        );
        assert_eq!(
            texel_copy_block_size(FormatCaps::depth(2), TextureAspect::DepthOnly),
            2
        );
    }

    #[test]
    fn stencil_only_is_always_one_byte() {
        assert_eq!(
            texel_copy_block_size(FormatCaps::stencil(1), TextureAspect::StencilOnly),
            1
        );
        assert_eq!(
            texel_copy_block_size(FormatCaps::depth_stencil(5), TextureAspect::StencilOnly),
            1
        );
    }

    #[test]
    fn depth_stencil_copy_allowed_matches_webgpu_table() {
        let stencil8 = TextureFormat::from_raw(TextureFormat::STENCIL8);
        assert!(depth_stencil_copy_allowed(
            stencil8,
            TextureAspect::All,
            true
        ));
        assert!(depth_stencil_copy_allowed(
            stencil8,
            TextureAspect::StencilOnly,
            false
        ));
        assert!(!depth_stencil_copy_allowed(
            stencil8,
            TextureAspect::DepthOnly,
            true
        ));

        let depth16 = TextureFormat::from_raw(TextureFormat::DEPTH16_UNORM);
        assert!(depth_stencil_copy_allowed(
            depth16,
            TextureAspect::DepthOnly,
            true
        ));
        assert!(depth_stencil_copy_allowed(
            depth16,
            TextureAspect::All,
            false
        ));
        assert!(!depth_stencil_copy_allowed(
            depth16,
            TextureAspect::StencilOnly,
            false
        ));

        let depth32 = TextureFormat::from_raw(TextureFormat::DEPTH32_FLOAT);
        assert!(!depth_stencil_copy_allowed(
            depth32,
            TextureAspect::DepthOnly,
            true
        ));
        assert!(depth_stencil_copy_allowed(
            depth32,
            TextureAspect::DepthOnly,
            false
        ));
        assert!(depth_stencil_copy_allowed(
            depth32,
            TextureAspect::All,
            false
        ));
        assert!(!depth_stencil_copy_allowed(
            depth32,
            TextureAspect::StencilOnly,
            false
        ));

        let depth24_plus = TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS);
        for aspect in [
            TextureAspect::All,
            TextureAspect::DepthOnly,
            TextureAspect::StencilOnly,
        ] {
            assert!(!depth_stencil_copy_allowed(depth24_plus, aspect, true));
            assert!(!depth_stencil_copy_allowed(depth24_plus, aspect, false));
        }

        let depth24_stencil8 = TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS_STENCIL8);
        assert!(!depth_stencil_copy_allowed(
            depth24_stencil8,
            TextureAspect::DepthOnly,
            false
        ));
        assert!(depth_stencil_copy_allowed(
            depth24_stencil8,
            TextureAspect::StencilOnly,
            true
        ));
        assert!(!depth_stencil_copy_allowed(
            depth24_stencil8,
            TextureAspect::All,
            false
        ));

        let depth32_stencil8 = TextureFormat::from_raw(TextureFormat::DEPTH32_FLOAT_STENCIL8);
        assert!(!depth_stencil_copy_allowed(
            depth32_stencil8,
            TextureAspect::DepthOnly,
            true
        ));
        assert!(depth_stencil_copy_allowed(
            depth32_stencil8,
            TextureAspect::DepthOnly,
            false
        ));
        assert!(depth_stencil_copy_allowed(
            depth32_stencil8,
            TextureAspect::StencilOnly,
            true
        ));
        assert!(depth_stencil_copy_allowed(
            depth32_stencil8,
            TextureAspect::StencilOnly,
            false
        ));
        assert!(!depth_stencil_copy_allowed(
            depth32_stencil8,
            TextureAspect::All,
            false
        ));

        let color = TextureFormat::from_raw(TextureFormat::R8_UNORM);
        assert!(depth_stencil_copy_allowed(
            color,
            TextureAspect::DepthOnly,
            true
        ));
    }

    #[test]
    fn tight_buffer_accepted_for_packed_depth_aspect_copy() {
        // A 3×3 depth-aspect copy of depth32float-stencil8 with bytesPerRow=256:
        // required = (3-1)*256 + 3*4 = 524 (the depth plane is 4 bytes/texel, NOT
        // the whole-format 5-byte block which would over-report 527 and wrongly
        // reject a tight 524-byte buffer). Regression guard for F-032.
        let caps = FormatCaps::depth_stencil(5);
        let layout = TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(256),
            rows_per_image: Some(3),
        };
        let size = Extent3d {
            width: 3,
            height: 3,
            depth_or_array_layers: 1,
        };
        let required =
            validate_texel_copy_layout(caps, TextureAspect::DepthOnly, size, layout, "test", true)
                .expect("tight depth-aspect copy layout should validate");
        assert_eq!(required, 524);
    }
}
