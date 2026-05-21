use std::sync::Arc;

use yawgpu_hal::{HalExtent3d, HalOrigin3d};

use crate::buffer::*;
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
