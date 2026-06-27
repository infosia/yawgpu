use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{
    HalBoundBuffer, HalBoundExternalTexture, HalBoundIndexBuffer, HalBoundIndirectBuffer,
    HalBoundSampler, HalBoundTexture, HalBufferClear, HalBufferCopy, HalBufferTextureCopy,
    HalBufferTextureLayout, HalBufferUsage, HalComputeDispatch, HalComputePass, HalCopy, HalDevice,
    HalDraw, HalIndexFormat, HalQueue, HalRenderColorTarget, HalRenderDepthStencilAttachment,
    HalRenderLoadOp, HalRenderPass, HalResolveQuerySet, HalScissorRect, HalTextureAspect,
    HalTextureClear, HalTextureCopy, HalTextureViewDimension, HalViewport,
};

use crate::bind_group::*;
use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::compute_pipeline::*;
use crate::copy::*;
use crate::error::*;
use crate::extent::*;
use crate::pass::*;
use crate::query_set::QuerySet;
use crate::render_pipeline::*;
use crate::texture::hal_texture_format;
use crate::texture::*;
use crate::texture_view::{TextureAspect, TextureView, TextureViewDimension};

/// Stores queue data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct Queue {
    pub(crate) inner: Arc<QueueInner>,
}

/// Holds shared state for the queue handle.
#[derive(Debug)]
pub(crate) struct QueueInner {
    pub(crate) hal: HalQueue,
    pub(crate) label: Mutex<String>,
}

/// Describes a queue buffer write operation.
#[derive(Debug, Clone, Copy)]
pub struct QueueBufferWrite<'a> {
    /// Device used to allocate the temporary staging buffer.
    pub device: &'a HalDevice,
    /// Destination buffer.
    pub buffer: &'a Buffer,
    /// Byte offset into the destination buffer.
    pub offset: u64,
    /// Source bytes.
    pub data: &'a [u8],
}

/// Describes a queue texture write operation.
#[derive(Debug, Clone, Copy)]
pub struct QueueTextureWrite<'a> {
    /// Device used to allocate the temporary staging buffer.
    pub device: &'a HalDevice,
    /// Destination texture.
    pub texture: &'a Texture,
    /// Destination mip level.
    pub mip_level: u32,
    /// Destination origin.
    pub origin: Origin3d,
    /// Write extent.
    pub write_size: Extent3d,
    /// Destination aspect.
    pub aspect: TextureAspect,
    /// Source data layout.
    pub layout: TexelCopyBufferLayout,
    /// Source bytes.
    pub data: &'a [u8],
}

impl Queue {
    /// Constructs this object from the backend HAL object.
    #[must_use]
    pub fn from_hal(hal: HalQueue, label: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(QueueInner {
                hal,
                label: Mutex::new(label.into()),
            }),
        }
    }

    /// Returns the HAL.
    #[must_use]
    pub fn hal(&self) -> &HalQueue {
        &self.inner.hal
    }

    /// Sets label on this object or encoder.
    pub fn set_label(&self, label: &str) {
        *self.inner.label.lock() = label.to_owned();
    }

    /// Returns the label.
    #[must_use]
    pub fn label(&self) -> String {
        self.inner.label.lock().clone()
    }

    /// Writes `data` into the buffer at `offset` using a staging buffer copy.
    ///
    /// The write is ordered after all previously submitted queue work: a
    /// temporary `copy_src` staging buffer is allocated, the host data is
    /// written into it, and then a `HalCopy::Buffer` is submitted via
    /// `submit_copies`.  This matches the WebGPU queue-timeline ordering
    /// guarantee and avoids the race on Vulkan where a direct host write
    /// into the destination buffer can be observed by still-executing prior
    /// submits (CTS finding F-074).
    ///
    /// Empty writes (zero-length `data`) are validated and then skip
    /// staging allocation, matching the no-op behaviour of the former direct
    /// path.
    pub fn write_buffer(&self, write: QueueBufferWrite<'_>) -> Option<DeviceError> {
        let QueueBufferWrite {
            device,
            buffer,
            offset,
            data,
        } = write;

        // Validate size fits in u64 first (mirrors write_texture).
        let size = match u64::try_from(data.len()) {
            Ok(s) => s,
            Err(_) => {
                return Some(DeviceError::validation(
                    "queue write buffer size is too large",
                ));
            }
        };

        // Run all core validation (error/destroyed/mapped/usage/alignment/bounds).
        if let Err(message) = buffer.validate_queue_write(offset, size) {
            return Some(DeviceError::validation(message));
        }

        // Zero-length write: validated above, nothing to copy.
        if data.is_empty() {
            return None;
        }

        // Allocate a staging buffer with copy_src semantics.  The host writes
        // the data into fresh memory with no ordering constraint; the GPU copy
        // that follows is sequenced after prior submits by the queue.
        let staging = match device.create_buffer(
            size,
            HalBufferUsage {
                copy_src: true,
                ..HalBufferUsage::default()
            },
        ) {
            Ok(buf) => buf,
            Err(error) => {
                let kind = match error {
                    yawgpu_hal::HalError::OutOfMemory { .. } => ErrorKind::OutOfMemory,
                    _ => ErrorKind::Internal,
                };
                return Some(DeviceError::new(kind, error.to_string()));
            }
        };

        if let Err(error) = staging.write(0, data) {
            return Some(DeviceError::internal(error.to_string()));
        }

        // Retrieve the HAL handle for the destination buffer.  A None here
        // means the buffer is an error buffer, which validate_queue_write
        // would have caught above; treat it as internal if somehow reached.
        let Some(destination) = buffer.hal() else {
            return Some(DeviceError::internal(
                "queue write buffer destination has no HAL buffer",
            ));
        };

        let copy = HalCopy::Buffer(HalBufferCopy {
            source: staging,
            source_offset: 0,
            destination,
            destination_offset: offset,
            size,
        });

        self.inner
            .hal
            .submit_copies(&[copy])
            .err()
            .map(|error| DeviceError::internal(error.to_string()))
    }

    /// Waits until all submitted queue work has completed.
    pub fn wait_idle(&self) -> Option<DeviceError> {
        self.inner
            .hal
            .wait_idle()
            .err()
            .map(|error| DeviceError::internal(error.to_string()))
    }

    /// Writes `data` into the texture through the backend copy path.
    pub fn write_texture(&self, write: QueueTextureWrite<'_>) -> Option<DeviceError> {
        let QueueTextureWrite {
            device,
            texture,
            mip_level,
            origin,
            write_size,
            aspect,
            layout,
            data,
        } = write;
        let data_size = match u64::try_from(data.len()) {
            Ok(size) => size,
            Err(_) => {
                return Some(DeviceError::validation(
                    "queue write texture dataSize is too large",
                ))
            }
        };
        if let Err(message) =
            texture.validate_queue_write(mip_level, origin, write_size, aspect, layout, data_size)
        {
            return Some(DeviceError::validation(message));
        }
        if extent_is_empty(write_size) {
            return None;
        }
        let destination_texture = texture;

        // Determine if the caller's layout needs repacking into a tightly-packed
        // staging buffer. Vulkan's bufferRowLength is expressed in texels, so a
        // bytes_per_row that is not a multiple of the per-aspect block size cannot
        // be represented. Similarly, bufferOffset must be texel-aligned, so a
        // layout.offset that is not a multiple of the block size is also repacked.
        // When either condition holds, we copy each row individually into a new
        // staging buffer with offset=0 and a tight (block-aligned) stride.
        let Some(format_caps) = texture.format_caps() else {
            return Some(DeviceError::internal(
                "queue write texture format is unsupported",
            ));
        };
        let block_size = u64::from(crate::copy::texel_copy_block_size(format_caps, aspect));
        let width_blocks = u64::from(crate::copy::div_ceil_u32(
            write_size.width,
            format_caps.block_w,
        ));
        let height_blocks = u64::from(crate::copy::div_ceil_u32(
            write_size.height,
            format_caps.block_h,
        ));
        let depth = u64::from(write_size.depth_or_array_layers);
        // Tight row bytes: width_blocks * block_size (always fits in u32 per
        // validation, but use u64 throughout for checked arithmetic).
        let tight_row_bytes = width_blocks.saturating_mul(block_size);

        let needs_repack = layout
            .bytes_per_row
            .is_some_and(|bpr| u64::from(bpr) % block_size != 0)
            || (block_size > 1 && layout.offset % block_size != 0);

        if needs_repack {
            // src_bytes_per_row is guaranteed Some by validation (multi-row implies
            // bytes_per_row is required).
            let src_bytes_per_row = u64::from(layout.bytes_per_row.unwrap_or(0));
            let src_rows_per_image =
                u64::from(layout.rows_per_image.unwrap_or(height_blocks as u32));

            let repacked = match repack_texel_rows(
                data,
                layout.offset,
                src_bytes_per_row,
                src_rows_per_image,
                tight_row_bytes,
                height_blocks,
                depth,
            ) {
                Ok(buf) => buf,
                Err(message) => {
                    return Some(DeviceError::internal(message));
                }
            };

            let repacked_size = match u64::try_from(repacked.len()) {
                Ok(n) => n,
                Err(_) => {
                    return Some(DeviceError::internal(
                        "queue write texture repack buffer size overflows",
                    ))
                }
            };
            let staging = match device.create_buffer(
                repacked_size,
                HalBufferUsage {
                    copy_src: true,
                    ..HalBufferUsage::default()
                },
            ) {
                Ok(buffer) => buffer,
                Err(error) => {
                    let kind = match error {
                        yawgpu_hal::HalError::OutOfMemory { .. } => ErrorKind::OutOfMemory,
                        _ => ErrorKind::Internal,
                    };
                    return Some(DeviceError::new(kind, error.to_string()));
                }
            };
            if let Err(error) = staging.write(0, &repacked) {
                return Some(DeviceError::internal(error.to_string()));
            }

            // Build a packed HAL layout: offset=0, tight stride, height_blocks rows.
            let packed_bytes_per_row = match u32::try_from(tight_row_bytes) {
                Ok(n) => n,
                Err(_) => {
                    return Some(DeviceError::internal(
                        "queue write texture packed bytes_per_row overflows u32",
                    ))
                }
            };
            let packed_rows_per_image = match u32::try_from(height_blocks) {
                Ok(n) => n,
                Err(_) => {
                    return Some(DeviceError::internal(
                        "queue write texture packed rows_per_image overflows u32",
                    ))
                }
            };
            let buffer_layout = HalBufferTextureLayout {
                offset: 0,
                bytes_per_row: packed_bytes_per_row,
                rows_per_image: packed_rows_per_image,
            };

            let format = hal_texture_format(texture.format());
            let Some(texture) = texture.hal() else {
                return Some(DeviceError::internal(
                    "queue write texture has no HAL texture",
                ));
            };
            let copy = HalCopy::BufferToTexture(HalBufferTextureCopy {
                buffer: staging,
                buffer_layout,
                texture,
                format,
                aspect: hal_texture_aspect(aspect),
                mip_level,
                origin: hal_origin(origin),
                extent: hal_extent(write_size),
            });
            let mut copies = Vec::new();
            append_texture_write_init_clears(
                &mut copies,
                destination_texture,
                mip_level,
                origin,
                write_size,
                aspect,
            );
            copies.push(copy);
            return self
                .inner
                .hal
                .submit_copies(&copies)
                .err()
                .map(|error| DeviceError::internal(error.to_string()));
        }

        // Fast path: layout is already backend-representable; copy verbatim.
        let staging = match device.create_buffer(
            data_size,
            HalBufferUsage {
                copy_src: true,
                ..HalBufferUsage::default()
            },
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                let kind = match error {
                    yawgpu_hal::HalError::OutOfMemory { .. } => ErrorKind::OutOfMemory,
                    _ => ErrorKind::Internal,
                };
                return Some(DeviceError::new(kind, error.to_string()));
            }
        };
        if let Err(error) = staging.write(0, data) {
            return Some(DeviceError::internal(error.to_string()));
        }
        let Some(buffer_layout) = hal_buffer_texture_layout(layout, texture, write_size) else {
            return Some(DeviceError::internal(
                "queue write texture format is unsupported",
            ));
        };
        let format = hal_texture_format(texture.format());
        let Some(texture) = texture.hal() else {
            return Some(DeviceError::internal(
                "queue write texture has no HAL texture",
            ));
        };
        let copy = HalCopy::BufferToTexture(HalBufferTextureCopy {
            buffer: staging,
            buffer_layout,
            texture,
            format,
            aspect: hal_texture_aspect(aspect),
            mip_level,
            origin: hal_origin(origin),
            extent: hal_extent(write_size),
        });
        let mut copies = Vec::new();
        append_texture_write_init_clears(
            &mut copies,
            destination_texture,
            mip_level,
            origin,
            write_size,
            aspect,
        );
        copies.push(copy);
        self.inner
            .hal
            .submit_copies(&copies)
            .err()
            .map(|error| DeviceError::internal(error.to_string()))
    }

    /// Submits command buffers to the queue after validating each is non-error and not already submitted.
    pub fn submit(&self, command_buffers: &[Arc<CommandBuffer>]) -> Option<DeviceError> {
        let mut validation_error = None;
        for (index, command_buffer) in command_buffers.iter().enumerate() {
            if command_buffer.is_error() {
                validation_error = Some("queue submit cannot use an error command buffer");
                break;
            }
            if command_buffer.is_submitted() {
                validation_error = Some("command buffer cannot be submitted more than once");
                break;
            }
            if command_buffers[..index]
                .iter()
                .any(|previous| previous.same(command_buffer))
            {
                validation_error = Some("command buffer cannot be submitted more than once");
                break;
            }
            for buffer in command_buffer.referenced_buffers() {
                if buffer.map_state() != BufferMapState::Unmapped {
                    validation_error = Some("queue submit cannot use a mapped buffer");
                    break;
                }
                if buffer.is_destroyed() {
                    validation_error = Some("queue submit cannot use a destroyed buffer");
                    break;
                }
            }
            if validation_error.is_some() {
                break;
            }
            for texture in command_buffer.referenced_textures() {
                if texture.is_destroyed() {
                    validation_error = Some("queue submit cannot use a destroyed texture");
                    break;
                }
            }
            if validation_error.is_some() {
                break;
            }
            for query_set in command_buffer.referenced_query_sets() {
                if query_set.is_destroyed() {
                    validation_error = Some("queue submit cannot use a destroyed query set");
                    break;
                }
            }
            if validation_error.is_some() {
                break;
            }
            for texture in command_buffer_referenced_textures(command_buffer) {
                if texture.is_destroyed() {
                    validation_error = Some("queue submit cannot use a destroyed texture");
                    break;
                }
            }
            if validation_error.is_some() {
                break;
            }
        }
        for command_buffer in command_buffers {
            if let Err(message) = command_buffer.mark_submitted() {
                return Some(DeviceError::validation(message));
            }
        }
        if let Some(message) = validation_error {
            return Some(DeviceError::validation(message));
        }
        if command_buffers.is_empty() {
            if let Err(error) = self.inner.hal.submit_empty() {
                return Some(DeviceError::internal(error.to_string()));
            }
            return None;
        }
        let mut copies = Vec::new();
        let all_ops: Vec<_> = command_buffers
            .iter()
            .flat_map(|command_buffer| command_buffer.command_ops().iter())
            .collect();
        for (op_index, op) in all_ops.iter().enumerate() {
            append_hal_command_execution(&mut copies, op, &all_ops[..=op_index]);
        }
        if let Err(error) = self.inner.hal.submit_copies(&copies) {
            return Some(DeviceError::internal(error.to_string()));
        }
        None
    }
}

fn command_buffer_referenced_textures(command_buffer: &CommandBuffer) -> Vec<Texture> {
    let mut textures = Vec::new();
    for op in command_buffer.command_ops() {
        match op {
            CommandExecution::TextureCopy(copy) => match copy {
                TextureCopyCommand::BufferToTexture { destination, .. } => {
                    textures.push((*destination.texture).clone())
                }
                TextureCopyCommand::TextureToBuffer { source, .. } => {
                    textures.push((*source.texture).clone());
                }
                TextureCopyCommand::TextureToTexture {
                    source,
                    destination,
                    ..
                } => {
                    textures.push((*source.texture).clone());
                    textures.push((*destination.texture).clone());
                }
            },
            CommandExecution::RenderPass(pass) => textures.extend(pass.attachment_textures.clone()),
            CommandExecution::BufferCopy(_)
            | CommandExecution::BufferClear(_)
            | CommandExecution::ResolveQuerySet(_)
            | CommandExecution::ComputePass(_) => {}
        }
    }
    textures
}

/// Repacks a caller-strided texel-copy source into a tightly-packed buffer.
///
/// The caller's buffer may have an arbitrary `bytes_per_row` (no alignment rule
/// for `wgpuQueueWriteTexture`). Vulkan's `bufferRowLength` is expressed in
/// texels, so a stride that is not a multiple of the per-aspect block size is
/// not representable. This function copies each row individually from the source
/// into a new `Vec<u8>` with stride exactly `row_bytes`, discarding any
/// inter-row and inter-image padding.
///
/// All source ranges are guaranteed in-bounds by the preceding call to
/// `validate_queue_write_texture`, but we use checked slicing for defence in
/// depth rather than panicking.
pub(crate) fn repack_texel_rows(
    data: &[u8],
    src_offset: u64,
    src_bytes_per_row: u64,
    src_rows_per_image: u64,
    row_bytes: u64,
    height_blocks: u64,
    depth: u64,
) -> Result<Vec<u8>, String> {
    let total = row_bytes
        .checked_mul(height_blocks)
        .and_then(|n| n.checked_mul(depth))
        .ok_or_else(|| "repack_texel_rows: output size overflows u64".to_owned())?;
    let total_usize = usize::try_from(total)
        .map_err(|_| "repack_texel_rows: output size overflows usize".to_owned())?;
    let row_bytes_usize = usize::try_from(row_bytes)
        .map_err(|_| "repack_texel_rows: row_bytes overflows usize".to_owned())?;

    let mut out = vec![0u8; total_usize];
    for d in 0..depth {
        for r in 0..height_blocks {
            let row_index = d
                .checked_mul(src_rows_per_image)
                .and_then(|n| n.checked_add(r))
                .ok_or_else(|| "repack_texel_rows: source offset overflows u64".to_owned())?;
            let src_start = row_index
                .checked_mul(src_bytes_per_row)
                .and_then(|n| n.checked_add(src_offset))
                .ok_or_else(|| "repack_texel_rows: source offset overflows u64".to_owned())?;
            let src_start_usize = usize::try_from(src_start)
                .map_err(|_| "repack_texel_rows: source offset overflows usize".to_owned())?;
            let src_end_usize = src_start_usize
                .checked_add(row_bytes_usize)
                .ok_or_else(|| "repack_texel_rows: source end overflows usize".to_owned())?;
            let src_row = data
                .get(src_start_usize..src_end_usize)
                .ok_or_else(|| "repack_texel_rows: source slice out of bounds".to_owned())?;

            let dst_start = (d * height_blocks + r) * row_bytes;
            let dst_start_usize = usize::try_from(dst_start)
                .map_err(|_| "repack_texel_rows: destination offset overflows usize".to_owned())?;
            let dst_end_usize = dst_start_usize
                .checked_add(row_bytes_usize)
                .ok_or_else(|| "repack_texel_rows: destination end overflows usize".to_owned())?;
            out[dst_start_usize..dst_end_usize].copy_from_slice(src_row);
        }
    }
    Ok(out)
}

fn hal_buffer_texture_layout(
    layout: TexelCopyBufferLayout,
    texture: &Texture,
    copy_size: Extent3d,
) -> Option<HalBufferTextureLayout> {
    let format_caps = texture.format_caps()?;
    let width_blocks = crate::copy::div_ceil_u32(copy_size.width, format_caps.block_w);
    let height_blocks = crate::copy::div_ceil_u32(copy_size.height, format_caps.block_h);
    let row_bytes = width_blocks.checked_mul(format_caps.texel_block_size)?;
    Some(HalBufferTextureLayout {
        offset: layout.offset,
        bytes_per_row: layout.bytes_per_row.unwrap_or(row_bytes),
        rows_per_image: layout.rows_per_image.unwrap_or(height_blocks),
    })
}

/// Returns HAL command execution.
#[cfg(test)]
pub(crate) fn hal_command_execution(op: &CommandExecution) -> Option<HalCopy> {
    hal_command_execution_with_ops(op, &[op])
}

#[cfg(test)]
fn hal_command_execution_with_ops(
    op: &CommandExecution,
    command_ops: &[&CommandExecution],
) -> Option<HalCopy> {
    let mut copies = Vec::new();
    append_hal_command_execution(&mut copies, op, command_ops);
    copies.into_iter().next()
}

fn append_hal_command_execution(
    copies: &mut Vec<HalCopy>,
    op: &CommandExecution,
    command_ops: &[&CommandExecution],
) {
    match op {
        CommandExecution::BufferCopy(copy) => {
            if copy.size == 0 {
                return;
            }
            let Some(source) = copy.source.hal() else {
                return;
            };
            let Some(destination) = copy.destination.hal() else {
                return;
            };
            copies.push(HalCopy::Buffer(HalBufferCopy {
                source,
                source_offset: copy.source_offset,
                destination,
                destination_offset: copy.destination_offset,
                size: copy.size,
            }));
        }
        CommandExecution::BufferClear(clear) => {
            if clear.size == 0 {
                return;
            }
            let Some(buffer) = clear.buffer.hal() else {
                return;
            };
            copies.push(HalCopy::BufferClear(HalBufferClear {
                buffer,
                offset: clear.offset,
                size: clear.size,
            }));
        }
        CommandExecution::ResolveQuerySet(resolve) => {
            let Some(query_set) = resolve.query_set.hal() else {
                return;
            };
            let Some(destination) = resolve.destination.hal() else {
                return;
            };
            copies.push(HalCopy::ResolveQuerySet(HalResolveQuerySet {
                query_set,
                first_query: resolve.first_query,
                query_count: resolve.query_count,
                written_queries: resolve_written_occlusion_queries(resolve, command_ops),
                destination,
                destination_offset: resolve.destination_offset,
            }));
        }
        CommandExecution::TextureCopy(copy) => append_texture_copy_execution(copies, copy),
        CommandExecution::ComputePass(pass) => {
            append_writable_storage_texture_init_clears(copies, &pass.bind_groups);
            if let Some(copy) = hal_compute_pass_execution(pass) {
                copies.push(copy);
            }
        }
        CommandExecution::RenderPass(pass) => {
            append_writable_storage_texture_init_clears(copies, &pass.bind_groups);
            append_render_pass_color_attachment_init_clears(copies, pass);
            if let Some(copy) = hal_render_pass_execution(pass) {
                copies.push(copy);
            }
        }
    }
}

fn resolve_written_occlusion_queries(
    resolve: &ResolveQuerySetCommand,
    command_ops: &[&CommandExecution],
) -> Vec<u32> {
    let Some(end_query) = resolve.first_query.checked_add(resolve.query_count) else {
        return Vec::new();
    };
    command_ops
        .iter()
        .filter_map(|op| {
            let CommandExecution::RenderPass(pass) = *op else {
                return None;
            };
            let Some(query_set) = &pass.occlusion_query_set else {
                return None;
            };
            if !query_set.same(&resolve.query_set) {
                return None;
            }
            let query_index = pass.occlusion_query_index?;
            (resolve.first_query <= query_index && query_index < end_query).then_some(query_index)
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Returns HAL texture copy execution.
pub(crate) fn hal_texture_copy_execution(copy: &TextureCopyCommand) -> Option<HalCopy> {
    match copy {
        TextureCopyCommand::BufferToTexture {
            source,
            destination,
            copy_size,
        } => {
            if extent_is_empty(*copy_size) {
                return None;
            }
            let buffer = source.buffer.hal()?;
            let texture = destination.texture.hal()?;
            let buffer_layout =
                hal_buffer_texture_layout(source.layout, &destination.texture, *copy_size)?;
            Some(HalCopy::BufferToTexture(HalBufferTextureCopy {
                buffer,
                buffer_layout,
                texture,
                format: hal_texture_format(destination.texture.format()),
                aspect: hal_texture_aspect(destination.aspect),
                mip_level: destination.mip_level,
                origin: hal_origin(destination.origin),
                extent: hal_extent(*copy_size),
            }))
        }
        TextureCopyCommand::TextureToBuffer {
            source,
            destination,
            copy_size,
        } => {
            if extent_is_empty(*copy_size) {
                return None;
            }
            let buffer = destination.buffer.hal()?;
            let texture = source.texture.hal()?;
            let buffer_layout =
                hal_buffer_texture_layout(destination.layout, &source.texture, *copy_size)?;
            Some(HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer,
                buffer_layout,
                texture,
                format: hal_texture_format(source.texture.format()),
                aspect: hal_texture_aspect(source.aspect),
                mip_level: source.mip_level,
                origin: hal_origin(source.origin),
                extent: hal_extent(*copy_size),
            }))
        }
        TextureCopyCommand::TextureToTexture {
            source,
            destination,
            copy_size,
        } => {
            if extent_is_empty(*copy_size) {
                return None;
            }
            let source_texture = source.texture.hal()?;
            let destination_texture = destination.texture.hal()?;
            Some(HalCopy::TextureToTexture(HalTextureCopy {
                source: source_texture,
                source_mip_level: source.mip_level,
                source_origin: hal_origin(source.origin),
                destination: destination_texture,
                destination_mip_level: destination.mip_level,
                destination_origin: hal_origin(destination.origin),
                extent: hal_extent(*copy_size),
            }))
        }
    }
}

fn append_texture_copy_execution(copies: &mut Vec<HalCopy>, copy: &TextureCopyCommand) {
    match copy {
        TextureCopyCommand::BufferToTexture {
            destination,
            copy_size,
            ..
        } => {
            append_texture_write_init_clears(
                copies,
                &destination.texture,
                destination.mip_level,
                destination.origin,
                *copy_size,
                destination.aspect,
            );
        }
        TextureCopyCommand::TextureToBuffer {
            source, copy_size, ..
        } => {
            append_texture_read_init_clears(
                copies,
                &source.texture,
                source.mip_level,
                source.origin,
                *copy_size,
                source.aspect,
            );
        }
        TextureCopyCommand::TextureToTexture {
            source,
            destination,
            copy_size,
        } => {
            append_texture_read_init_clears(
                copies,
                &source.texture,
                source.mip_level,
                source.origin,
                *copy_size,
                source.aspect,
            );
            append_texture_write_init_clears(
                copies,
                &destination.texture,
                destination.mip_level,
                destination.origin,
                *copy_size,
                destination.aspect,
            );
        }
    }
    if let Some(copy) = hal_texture_copy_execution(copy) {
        copies.push(copy);
    }
}

fn append_texture_read_init_clears(
    copies: &mut Vec<HalCopy>,
    texture: &Texture,
    mip_level: u32,
    origin: Origin3d,
    copy_size: Extent3d,
    aspect: TextureAspect,
) {
    if extent_is_empty(copy_size) || !texture_init_clear_supported(texture) {
        return;
    }
    for subresource in texture.copy_subresources(mip_level, origin, copy_size) {
        if texture.is_initialized(subresource.mip_level, subresource.array_layer) {
            continue;
        }
        append_texture_zero_clear(
            copies,
            texture,
            subresource.mip_level,
            subresource.array_layer,
            aspect,
        );
        texture.mark_initialized(subresource.mip_level, subresource.array_layer);
    }
}

fn append_texture_write_init_clears(
    copies: &mut Vec<HalCopy>,
    texture: &Texture,
    mip_level: u32,
    origin: Origin3d,
    copy_size: Extent3d,
    aspect: TextureAspect,
) {
    if extent_is_empty(copy_size) || !texture_init_clear_supported(texture) {
        return;
    }
    for subresource in texture.copy_subresources(mip_level, origin, copy_size) {
        if !subresource.covers_full_subresource
            && !texture.is_initialized(subresource.mip_level, subresource.array_layer)
        {
            append_texture_zero_clear(
                copies,
                texture,
                subresource.mip_level,
                subresource.array_layer,
                aspect,
            );
        }
        texture.mark_initialized(subresource.mip_level, subresource.array_layer);
    }
}

fn append_texture_zero_clear(
    copies: &mut Vec<HalCopy>,
    texture: &Texture,
    mip_level: u32,
    array_layer: u32,
    aspect: TextureAspect,
) {
    let Some(texture_hal) = texture.hal() else {
        return;
    };
    copies.push(HalCopy::ClearTexture(HalTextureClear {
        texture: texture_hal,
        format: hal_texture_format(texture.format()),
        aspect: hal_texture_aspect(aspect),
        mip_level,
        base_array_layer: array_layer,
        array_layer_count: 1,
    }));
}

fn texture_init_clear_supported(texture: &Texture) -> bool {
    texture.is_lazy_init_eligible()
}

fn append_writable_storage_texture_init_clears(
    copies: &mut Vec<HalCopy>,
    bind_groups: &BTreeMap<u32, BoundBindGroup>,
) {
    // TODO(stage2): sampled texture bindings that read uninitialized textures
    // should inject clears before the pass, matching copy and storage writes.
    for bound in bind_groups.values() {
        let layout_entries = bound.group.layout().entries();
        for layout_entry in layout_entries {
            let Some(BindingLayoutKind::StorageTexture { access, .. }) = layout_entry.kind else {
                continue;
            };
            if access == StorageTextureAccess::ReadOnly {
                continue;
            }
            let Some(entry) = bound
                .group
                .entries()
                .iter()
                .find(|entry| entry.binding == layout_entry.binding)
            else {
                continue;
            };
            let BindGroupResource::TextureView { texture_view, .. } = &entry.resource else {
                continue;
            };
            append_texture_view_init_clears(copies, texture_view);
        }
    }
}

fn append_texture_view_init_clears(copies: &mut Vec<HalCopy>, texture_view: &TextureView) {
    let texture = texture_view.texture();
    if !texture.is_lazy_init_eligible() {
        return;
    }
    let mip_start = texture_view.base_mip_level();
    let mip_end = mip_start.saturating_add(texture_view.mip_level_count());
    for mip_level in mip_start..mip_end {
        match texture.dimension() {
            TextureDimension::D3 => {
                append_texture_subresource_init_clear_if_needed(copies, &texture, mip_level, 0);
            }
            TextureDimension::D1 | TextureDimension::D2 => {
                let layer_start = texture_view.base_array_layer();
                let layer_end = layer_start.saturating_add(texture_view.array_layer_count());
                for array_layer in layer_start..layer_end {
                    append_texture_subresource_init_clear_if_needed(
                        copies,
                        &texture,
                        mip_level,
                        array_layer,
                    );
                }
            }
        }
    }
}

fn append_texture_subresource_init_clear_if_needed(
    copies: &mut Vec<HalCopy>,
    texture: &Texture,
    mip_level: u32,
    array_layer: u32,
) {
    if !texture.is_initialized(mip_level, array_layer) {
        append_texture_zero_clear(copies, texture, mip_level, array_layer, TextureAspect::All);
    }
    texture.mark_initialized(mip_level, array_layer);
}

fn append_render_pass_color_attachment_init_clears(
    copies: &mut Vec<HalCopy>,
    pass: &RenderPassCommand,
) {
    for attachment in pass.color_attachments.iter().flatten() {
        append_render_attachment_init_clear_if_needed(
            copies,
            &attachment.texture,
            attachment.mip_level,
            attachment.array_layer,
        );
        if let Some(resolve_target) = &attachment.resolve_target {
            append_render_attachment_init_clear_if_needed(
                copies,
                resolve_target,
                attachment.resolve_mip_level,
                attachment.resolve_array_layer,
            );
        }
    }
}

fn append_render_attachment_init_clear_if_needed(
    copies: &mut Vec<HalCopy>,
    texture: &Texture,
    mip_level: u32,
    array_layer: u32,
) {
    if !texture.is_lazy_init_eligible() {
        return;
    }
    let tracked_layer = match texture.dimension() {
        TextureDimension::D3 => 0,
        TextureDimension::D1 | TextureDimension::D2 => array_layer,
    };
    append_texture_subresource_init_clear_if_needed(copies, texture, mip_level, tracked_layer);
}

fn extent_is_empty(extent: Extent3d) -> bool {
    extent.width == 0 || extent.height == 0 || extent.depth_or_array_layers == 0
}

/// Returns HAL compute pass execution.
pub(crate) fn hal_compute_pass_execution(pass: &ComputePassCommand) -> Option<HalCopy> {
    let pipeline = pass.pipeline.hal()?;
    let bindings = hal_bind_resources(
        pass.pipeline.bind_group_layouts(),
        pass.pipeline.metal_bindings(),
        &pass.bind_groups,
    )?;
    Some(HalCopy::ComputePass(HalComputePass {
        pipeline,
        bind_buffers: bindings.buffers,
        bind_textures: bindings.textures,
        bind_samplers: bindings.samplers,
        bind_external_textures: bindings.external_textures,
        dispatch: hal_compute_dispatch(&pass.dispatch)?,
    }))
}

fn hal_compute_dispatch(dispatch: &ComputeDispatch) -> Option<HalComputeDispatch> {
    match dispatch {
        ComputeDispatch::Direct { workgroups } => Some(HalComputeDispatch::Direct {
            workgroups: *workgroups,
        }),
        ComputeDispatch::Indirect { buffer, offset } => Some(HalComputeDispatch::Indirect {
            buffer: Box::new(HalBoundIndirectBuffer {
                buffer: buffer.hal()?,
                offset: *offset,
            }),
        }),
    }
}

/// Returns HAL render pass execution.
pub(crate) fn hal_render_pass_execution(pass: &RenderPassCommand) -> Option<HalCopy> {
    let (
        pipeline,
        bind_buffers,
        bind_textures,
        bind_samplers,
        bind_external_textures,
        vertex_buffers,
        index_buffer,
        indirect_buffer,
        draw,
    ) = if let (Some(pipeline), Some(draw)) = (&pass.pipeline, pass.draw) {
        let bindings = hal_bind_resources(
            pipeline.bind_group_layouts(),
            pipeline.metal_bindings(),
            &pass.bind_groups,
        )?;
        let mut vertex_buffers = Vec::new();
        for binding in pipeline.vertex_buffer_bindings() {
            let bound = pass.vertex_buffers.get(&binding.slot)?;
            vertex_buffers.push(HalBoundBuffer {
                group: 0,
                binding: binding.slot,
                metal_index: binding.metal_index,
                vertex_metal_index: None,
                fragment_metal_index: None,
                buffer: bound.buffer.hal()?,
                offset: bound.offset,
                size: bound.size,
            });
        }
        let index_buffer = match &pass.index_buffer {
            Some(bound) => Some(Box::new(hal_bound_index_buffer(bound)?)),
            None => None,
        };
        let indirect_buffer = match &pass.indirect_buffer {
            Some(bound) => Some(Box::new(hal_bound_indirect_buffer(bound)?)),
            None => None,
        };
        (
            Some(pipeline.hal()?),
            bindings.buffers,
            bindings.textures,
            bindings.samplers,
            bindings.external_textures,
            vertex_buffers,
            index_buffer,
            indirect_buffer,
            Some(hal_draw(draw)),
        )
    } else {
        (
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            None,
            None,
            None,
        )
    };
    Some(HalCopy::RenderPass(HalRenderPass {
        pipeline,
        color_targets: hal_render_color_targets(&pass.color_attachments)?,
        depth_stencil_attachment: hal_render_depth_stencil_attachment(
            pass.depth_stencil_attachment.as_ref(),
        )?,
        bind_buffers,
        bind_textures,
        bind_samplers,
        bind_external_textures,
        vertex_buffers,
        index_buffer,
        indirect_buffer,
        viewport: pass.viewport.map(hal_viewport),
        scissor_rect: pass.scissor_rect.map(hal_scissor_rect),
        blend_constant: pass.blend_constant,
        stencil_reference: pass.stencil_reference & 0xFF,
        occlusion_query_set: pass.occlusion_query_set.as_ref().and_then(QuerySet::hal),
        occlusion_query_index: pass.occlusion_query_index,
        draw,
    }))
}

fn hal_viewport(viewport: Viewport) -> HalViewport {
    HalViewport {
        x: viewport.x,
        y: viewport.y,
        width: viewport.width,
        height: viewport.height,
        min_depth: viewport.min_depth,
        max_depth: viewport.max_depth,
    }
}

fn hal_scissor_rect(rect: ScissorRect) -> HalScissorRect {
    HalScissorRect {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
    }
}

fn hal_draw(draw: RenderDrawExecution) -> HalDraw {
    match draw {
        RenderDrawExecution::Direct {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        } => HalDraw::Direct {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        },
        RenderDrawExecution::Indexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        } => HalDraw::Indexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        },
        RenderDrawExecution::Indirect { offset } => HalDraw::Indirect { offset },
        RenderDrawExecution::IndexedIndirect { offset } => HalDraw::IndexedIndirect { offset },
    }
}

fn hal_bound_index_buffer(bound: &BoundIndexBuffer) -> Option<HalBoundIndexBuffer> {
    Some(HalBoundIndexBuffer {
        buffer: bound.buffer.hal()?,
        format: hal_index_format(bound.format),
        offset: bound.offset,
        size: bound.size,
    })
}

fn hal_bound_indirect_buffer(bound: &BoundIndirectBuffer) -> Option<HalBoundIndirectBuffer> {
    Some(HalBoundIndirectBuffer {
        buffer: bound.buffer.hal()?,
        offset: bound.offset,
    })
}

fn hal_index_format(format: IndexFormat) -> HalIndexFormat {
    match format {
        IndexFormat::Uint16 => HalIndexFormat::Uint16,
        IndexFormat::Uint32 => HalIndexFormat::Uint32,
    }
}

fn hal_render_color_targets(
    attachments: &[Option<RenderPassColorExecution>],
) -> Option<Vec<Option<HalRenderColorTarget>>> {
    attachments
        .iter()
        .map(|attachment| match attachment {
            Some(attachment) => Some(Some(HalRenderColorTarget {
                texture: attachment.texture.hal()?,
                view_format: hal_texture_format(attachment.view_format),
                resolve_target: match &attachment.resolve_target {
                    Some(texture) => Some(texture.hal()?),
                    None => None,
                },
                resolve_view_format: attachment.resolve_view_format.map(hal_texture_format),
                mip_level: attachment.mip_level,
                array_layer: attachment.array_layer,
                depth_slice: attachment.depth_slice,
                resolve_mip_level: attachment.resolve_mip_level,
                resolve_array_layer: attachment.resolve_array_layer,
                load_op: hal_render_load_op(attachment.load_op),
                store: matches!(attachment.store_op, StoreOp::Store),
                clear_color: [
                    attachment.clear_value.r,
                    attachment.clear_value.g,
                    attachment.clear_value.b,
                    attachment.clear_value.a,
                ],
            })),
            None => Some(None),
        })
        .collect()
}

fn hal_render_depth_stencil_attachment(
    attachment: Option<&RenderPassDepthStencilExecution>,
) -> Option<Option<HalRenderDepthStencilAttachment>> {
    match attachment {
        None => Some(None),
        Some(attachment) => Some(Some(HalRenderDepthStencilAttachment {
            texture: attachment.texture.hal()?,
            format: hal_texture_format(attachment.format),
            mip_level: attachment.mip_level,
            array_layer: attachment.array_layer,
            depth_load_op: if attachment.depth_read_only {
                HalRenderLoadOp::Load
            } else {
                hal_render_load_op(attachment.depth_load_op)
            },
            depth_store: attachment.depth_read_only
                || matches!(attachment.depth_store_op, StoreOp::Store),
            depth_clear_value: attachment.depth_clear_value,
            depth_read_only: attachment.depth_read_only,
            stencil_load_op: if attachment.stencil_read_only {
                HalRenderLoadOp::Load
            } else {
                hal_render_load_op(attachment.stencil_load_op)
            },
            stencil_store: attachment.stencil_read_only
                || matches!(attachment.stencil_store_op, StoreOp::Store),
            stencil_clear_value: attachment.stencil_clear_value,
            stencil_read_only: attachment.stencil_read_only,
        })),
    }
}

fn hal_render_load_op(load_op: LoadOp) -> HalRenderLoadOp {
    match load_op {
        LoadOp::Load => HalRenderLoadOp::Load,
        LoadOp::Clear | LoadOp::Undefined => HalRenderLoadOp::Clear,
    }
}

fn hal_texture_aspect(aspect: TextureAspect) -> HalTextureAspect {
    match aspect {
        TextureAspect::All => HalTextureAspect::All,
        TextureAspect::DepthOnly => HalTextureAspect::DepthOnly,
        TextureAspect::StencilOnly => HalTextureAspect::StencilOnly,
    }
}

fn hal_texture_view_dimension(dimension: TextureViewDimension) -> HalTextureViewDimension {
    match dimension {
        TextureViewDimension::D1 => HalTextureViewDimension::D1,
        TextureViewDimension::D2 => HalTextureViewDimension::D2,
        TextureViewDimension::D2Array => HalTextureViewDimension::D2Array,
        TextureViewDimension::Cube => HalTextureViewDimension::Cube,
        TextureViewDimension::CubeArray => HalTextureViewDimension::CubeArray,
        TextureViewDimension::D3 => HalTextureViewDimension::D3,
    }
}

#[derive(Debug, Default)]
pub(crate) struct HalBoundResources {
    pub(crate) buffers: Vec<HalBoundBuffer>,
    pub(crate) textures: Vec<HalBoundTexture>,
    pub(crate) samplers: Vec<HalBoundSampler>,
    pub(crate) external_textures: Vec<HalBoundExternalTexture>,
}

/// Returns HAL bound shader resources.
pub(crate) fn hal_bind_resources(
    layouts: &[Arc<BindGroupLayout>],
    metal_bindings: &[MetalBufferBinding],
    bind_groups: &BTreeMap<u32, BoundBindGroup>,
) -> Option<HalBoundResources> {
    let mut resources = HalBoundResources::default();
    for binding in metal_bindings {
        let bound = bind_groups.get(&binding.group)?;
        let entry = bound
            .group
            .entries()
            .iter()
            .find(|entry| entry.binding == binding.binding)?;
        match (binding.kind, &entry.resource) {
            (
                MetalBindingKind::Buffer(_),
                BindGroupResource::Buffer {
                    buffer,
                    offset,
                    size,
                    ..
                },
            ) => {
                let dynamic_offset = dynamic_offset_for_binding(
                    layouts,
                    binding.group,
                    binding.binding,
                    &bound.dynamic_offsets,
                )?;
                let offset = offset.checked_add(dynamic_offset)?;
                resources.buffers.push(HalBoundBuffer {
                    group: binding.group,
                    binding: binding.binding,
                    metal_index: binding.metal_index,
                    vertex_metal_index: binding.vertex_metal_index,
                    fragment_metal_index: binding.fragment_metal_index,
                    buffer: buffer.hal()?,
                    offset,
                    size: *size,
                });
            }
            (MetalBindingKind::Texture, BindGroupResource::TextureView { texture_view, .. }) => {
                resources.textures.push(HalBoundTexture {
                    group: binding.group,
                    binding: binding.binding,
                    metal_index: binding.metal_index,
                    vertex_metal_index: binding.vertex_metal_index,
                    fragment_metal_index: binding.fragment_metal_index,
                    texture: texture_view.texture().hal()?,
                    format: hal_texture_format(texture_view.format()),
                    dimension: hal_texture_view_dimension(texture_view.dimension()),
                    base_mip_level: texture_view.base_mip_level(),
                    mip_level_count: texture_view.mip_level_count(),
                    base_array_layer: texture_view.base_array_layer(),
                    array_layer_count: texture_view.array_layer_count(),
                    aspect: hal_texture_aspect(texture_view.aspect()),
                    storage_access: None,
                });
            }
            (
                MetalBindingKind::StorageTexture { access },
                BindGroupResource::TextureView { texture_view, .. },
            ) => {
                resources.textures.push(HalBoundTexture {
                    group: binding.group,
                    binding: binding.binding,
                    metal_index: binding.metal_index,
                    vertex_metal_index: binding.vertex_metal_index,
                    fragment_metal_index: binding.fragment_metal_index,
                    texture: texture_view.texture().hal()?,
                    format: hal_texture_format(texture_view.format()),
                    dimension: hal_texture_view_dimension(texture_view.dimension()),
                    base_mip_level: texture_view.base_mip_level(),
                    mip_level_count: texture_view.mip_level_count(),
                    base_array_layer: texture_view.base_array_layer(),
                    array_layer_count: texture_view.array_layer_count(),
                    aspect: hal_texture_aspect(texture_view.aspect()),
                    storage_access: Some(hal_storage_texture_access(access)),
                });
            }
            (MetalBindingKind::Sampler, BindGroupResource::Sampler { sampler, .. }) => {
                resources.samplers.push(HalBoundSampler {
                    group: binding.group,
                    binding: binding.binding,
                    metal_index: binding.metal_index,
                    vertex_metal_index: binding.vertex_metal_index,
                    fragment_metal_index: binding.fragment_metal_index,
                    sampler: sampler.hal()?,
                });
            }
            (
                MetalBindingKind::ExternalTexture,
                BindGroupResource::ExternalTexture {
                    external_texture, ..
                },
            ) => {
                let inner = external_texture.inner();
                let plane0 = inner.planes.first()?;
                let plane1 = inner.planes.get(1).unwrap_or(plane0);
                let params_slot = binding.ext_params_buffer_slot?;
                let params_vertex_metal_index = match binding.vertex_metal_index {
                    Some(_) => Some(binding.ext_params_vertex_buffer_slot?),
                    None => None,
                };
                let params_fragment_metal_index = match binding.fragment_metal_index {
                    Some(_) => Some(binding.ext_params_fragment_buffer_slot?),
                    None => None,
                };
                let plane1_metal_index = binding.metal_index.checked_add(1)?;
                let plane1_vertex_metal_index = match binding.vertex_metal_index {
                    Some(slot) => Some(slot.checked_add(1)?),
                    None => None,
                };
                let plane1_fragment_metal_index = match binding.fragment_metal_index {
                    Some(slot) => Some(slot.checked_add(1)?),
                    None => None,
                };
                resources.external_textures.push(HalBoundExternalTexture {
                    group: binding.group,
                    binding: binding.binding,
                    plane0: plane0.texture().hal()?,
                    plane1: plane1.texture().hal()?,
                    plane0_metal_index: binding.metal_index,
                    plane1_metal_index,
                    plane0_vertex_metal_index: binding.vertex_metal_index,
                    plane1_vertex_metal_index,
                    plane0_fragment_metal_index: binding.fragment_metal_index,
                    plane1_fragment_metal_index,
                    params: inner.params.hal()?,
                    params_metal_index: params_slot,
                    params_vertex_metal_index,
                    params_fragment_metal_index,
                    format: hal_texture_format(plane0.format()),
                    dimension: hal_texture_view_dimension(plane0.dimension()),
                    params_offset: 0,
                    params_size: inner.params.size(),
                });
            }
            _ => return None,
        }
    }
    Some(resources)
}

/// Returns dynamic offset for binding.
pub(crate) fn dynamic_offset_for_binding(
    layouts: &[Arc<BindGroupLayout>],
    group: u32,
    binding: u32,
    dynamic_offsets: &[u32],
) -> Option<u64> {
    let layout = layouts.get(usize::try_from(group).ok()?)?;
    let mut dynamic_index = 0usize;
    for entry in layout.entries() {
        let is_dynamic = matches!(
            entry.kind,
            Some(BindingLayoutKind::Buffer {
                has_dynamic_offset: true,
                ..
            })
        );
        if entry.binding == binding {
            return if is_dynamic {
                dynamic_offsets.get(dynamic_index).copied().map(u64::from)
            } else {
                Some(0)
            };
        }
        if is_dynamic {
            dynamic_index = dynamic_index.checked_add(1)?;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::{SHADER_STAGE_COMPUTE, SHADER_STAGE_FRAGMENT, SHADER_STAGE_VERTEX};
    use crate::test_helpers::*;
    use crate::*;
    use yawgpu_hal::HalStorageTextureAccess;

    fn depth32_float() -> TextureFormat {
        TextureFormat::from_raw(0x30)
    }

    fn noop_depth_view(device: &Device) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: depth32_float(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);
        Arc::new(view)
    }

    fn sampled_external_texture_view(device: &Device) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::TEXTURE_BINDING,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);
        Arc::new(view)
    }

    fn rgba_external_texture(device: &Device) -> Arc<ExternalTexture> {
        Arc::new(
            device
                .create_external_texture(ExternalTextureDescriptor {
                    plane0: sampled_external_texture_view(device),
                    plane1: None,
                    format: ExternalTextureFormat::Rgba,
                    crop_origin: Origin2d { x: 0, y: 0 },
                    crop_size: Extent3d {
                        width: 4,
                        height: 4,
                        depth_or_array_layers: 1,
                    },
                    apparent_size: Extent3d {
                        width: 4,
                        height: 4,
                        depth_or_array_layers: 1,
                    },
                    do_yuv_to_rgb_conversion_only: true,
                    yuv_to_rgb_conversion_matrix: None,
                    src_transfer_function_parameters: [0.0; 7],
                    dst_transfer_function_parameters: [0.0; 7],
                    gamut_conversion_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    mirrored: false,
                    rotation: ExternalTextureRotation::Rotate0,
                })
                .expect("external texture"),
        )
    }

    fn depth_only_render_pass_descriptor(view: Arc<TextureView>) -> RenderPassDescriptor {
        RenderPassDescriptor {
            max_color_attachments: Limits::DEFAULT.max_color_attachments,
            color_attachments: Vec::new(),
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view,
                depth_load_op: LoadOp::Clear,
                depth_store_op: StoreOp::Store,
                depth_clear_value: 0.5,
                depth_read_only: false,
                stencil_load_op: LoadOp::Undefined,
                stencil_store_op: StoreOp::Undefined,
                stencil_clear_value: 0,
                stencil_read_only: false,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            max_draw_count: 50_000_000,
        }
    }

    fn depth_only_pipeline(device: &Device) -> Arc<RenderPipeline> {
        let module = Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(
            "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }"
                .to_owned(),
        )));
        Arc::new(device.create_render_pipeline(RenderPipelineDescriptor {
            layout: RenderPipelineLayout::Auto,
            vertex: RenderPipelineVertexState {
                shader: RenderPipelineShaderStage {
                    module,
                    entry_point: Some("vs".to_owned()),
                    constants: Vec::new(),
                },
                buffer_count: 0,
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
            },
            depth_stencil: Some(DepthStencilState {
                format: depth32_float(),
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Always),
                stencil_front: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail_op: StencilOperation::Keep,
                    depth_fail_op: StencilOperation::Keep,
                    pass_op: StencilOperation::Keep,
                },
                stencil_back: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail_op: StencilOperation::Keep,
                    depth_fail_op: StencilOperation::Keep,
                    pass_op: StencilOperation::Keep,
                },
                stencil_read_mask: u32::MAX,
                stencil_write_mask: u32::MAX,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            multisample: MultisampleState {
                count: 1,
                mask: u32::MAX,
                alpha_to_coverage_enabled: false,
            },
            fragment: None,
            error: None,
        }))
    }

    fn sparse_color_pipeline(device: &Device, real_slot: u32) -> Arc<RenderPipeline> {
        let module = Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(format!(
            "@vertex fn vs() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0, 0.0, 0.0, 1.0); }}
@fragment fn fs() -> @location({real_slot}) vec4<f32> {{ return vec4<f32>(1.0, 0.0, 0.0, 1.0); }}"
        ))));
        let mut targets = vec![
            ColorTargetState {
                format: TextureFormat::from_raw(TextureFormat::UNDEFINED),
                blend: None,
                write_mask: 0,
            },
            ColorTargetState {
                format: TextureFormat::from_raw(TextureFormat::UNDEFINED),
                blend: None,
                write_mask: 0,
            },
        ];
        targets[real_slot as usize] = ColorTargetState {
            format: rgba8_unorm(),
            blend: None,
            write_mask: 0xF,
        };
        Arc::new(device.create_render_pipeline(RenderPipelineDescriptor {
            layout: RenderPipelineLayout::Auto,
            vertex: RenderPipelineVertexState {
                shader: RenderPipelineShaderStage {
                    module: module.clone(),
                    entry_point: Some("vs".to_owned()),
                    constants: Vec::new(),
                },
                buffer_count: 0,
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: u32::MAX,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(RenderPipelineFragmentState {
                shader: RenderPipelineShaderStage {
                    module,
                    entry_point: Some("fs".to_owned()),
                    constants: Vec::new(),
                },
                target_count: 2,
                targets,
            }),
            error: None,
        }))
    }

    fn sparse_render_pass_descriptor(
        real_slot: usize,
        view: Arc<TextureView>,
    ) -> RenderPassDescriptor {
        let attachment = RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_value: Color {
                r: 0.25,
                g: 0.5,
                b: 0.75,
                a: 1.0,
            },
        };
        let mut color_attachments = vec![None, None];
        color_attachments[real_slot] = Some(attachment);
        RenderPassDescriptor {
            max_color_attachments: Limits::DEFAULT.max_color_attachments,
            color_attachments,
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            max_draw_count: 50_000_000,
        }
    }

    fn sampled_texture_bind_group_layout(device: &Device, visibility: u64) -> Arc<BindGroupLayout> {
        Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::Texture {
                        sample_type: TextureSampleType::Float,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    }),
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::Sampler {
                        ty: SamplerBindingType::Filtering,
                    }),
                },
            ],
            error: None,
        }))
    }

    fn sampled_texture_bind_group(device: &Device, layout: Arc<BindGroupLayout>) -> Arc<BindGroup> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 3,
            },
            ..valid_texture_descriptor()
        });
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 1,
            array_layer_count: Some(1),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);
        let sampler = device.create_sampler(SamplerDescriptor::default());
        Arc::new(device.create_bind_group(
            layout,
            vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindGroupResource::TextureView {
                        texture_view: Arc::new(view),
                        device: Arc::new(device.clone()),
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindGroupResource::Sampler {
                        sampler: Arc::new(sampler),
                        device: Arc::new(device.clone()),
                    },
                },
            ],
        ))
    }

    fn storage_texture_bind_group_layout(device: &Device) -> Arc<BindGroupLayout> {
        Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: SHADER_STAGE_COMPUTE,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::StorageTexture {
                    access: StorageTextureAccess::ReadOnly,
                    format: rgba8_unorm(),
                    view_dimension: TextureViewDimension::D2,
                }),
            }],
            error: None,
        }))
    }

    fn storage_texture_bind_group(device: &Device, layout: Arc<BindGroupLayout>) -> Arc<BindGroup> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::STORAGE_BINDING | TextureUsage::COPY_DST,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            ..valid_texture_descriptor()
        });
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(1),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);
        Arc::new(device.create_bind_group(
            layout,
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::TextureView {
                    texture_view: Arc::new(view),
                    device: Arc::new(device.clone()),
                },
            }],
        ))
    }

    fn explicit_pipeline_layout(
        device: &Device,
        layout: Arc<BindGroupLayout>,
    ) -> Arc<PipelineLayout> {
        Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![layout],
            immediate_size: 0,
            error: None,
        }))
    }

    fn sampled_compute_pipeline(
        device: &Device,
        layout: Arc<PipelineLayout>,
    ) -> Arc<ComputePipeline> {
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@compute @workgroup_size(1)
fn cs() {
    let loaded = textureLoad(tex, vec2<i32>(0, 0), 0);
    let sampled = textureSampleLevel(tex, samp, vec2<f32>(0.5, 0.5), 0.0);
    _ = loaded + sampled;
}
"
                .to_owned(),
            )),
        );
        Arc::new(device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        }))
    }

    fn storage_texture_compute_pipeline(
        device: &Device,
        layout: Arc<PipelineLayout>,
    ) -> Arc<ComputePipeline> {
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, read>;

@compute @workgroup_size(1)
fn cs() {
    _ = textureLoad(tex, vec2<i32>(0, 0));
}
"
                .to_owned(),
            )),
        );
        Arc::new(device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        }))
    }

    fn write_storage_texture_compute_pipeline(
        device: &Device,
        layout: Arc<PipelineLayout>,
    ) -> Arc<ComputePipeline> {
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1)
fn cs() {
    textureStore(tex, vec2<i32>(0, 0), vec4<f32>(0.4, 0.0, 0.0, 1.0));
}
"
                .to_owned(),
            )),
        );
        Arc::new(device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        }))
    }

    fn sampled_render_pipeline(
        device: &Device,
        layout: Arc<PipelineLayout>,
    ) -> Arc<RenderPipeline> {
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@vertex
fn vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return textureSample(tex, samp, vec2<f32>(0.5, 0.5));
}
"
                .to_owned(),
            )),
        );
        Arc::new(device.create_render_pipeline(RenderPipelineDescriptor {
            layout: RenderPipelineLayout::Explicit(layout),
            vertex: RenderPipelineVertexState {
                shader: RenderPipelineShaderStage {
                    module: module.clone(),
                    entry_point: Some("vs".to_owned()),
                    constants: Vec::new(),
                },
                buffer_count: 0,
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: u32::MAX,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(RenderPipelineFragmentState {
                shader: RenderPipelineShaderStage {
                    module,
                    entry_point: Some("fs".to_owned()),
                    constants: Vec::new(),
                },
                target_count: 1,
                targets: vec![ColorTargetState {
                    format: rgba8_unorm(),
                    blend: None,
                    write_mask: 0xF,
                }],
            }),
            error: None,
        }))
    }

    #[test]
    fn queue_from_hal_hal_label_and_set_label_round_trip() {
        let queue = Queue::from_hal(hal_noop_queue(), "initial");

        assert!(matches!(queue.hal(), yawgpu_hal::HalQueue::Noop(_)));
        assert_eq!(queue.label(), "initial");
        queue.set_label("renamed");
        assert_eq!(queue.label(), "renamed");
    }

    #[test]
    fn queue_write_buffer_and_submit_empty_succeed() {
        let device = noop_device();
        let queue = device.queue();
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        });

        assert_eq!(
            queue.write_buffer(QueueBufferWrite {
                device: device.hal(),
                buffer: &buffer,
                offset: 0,
                data: &[1, 2, 3, 4],
            }),
            None
        );
        assert_eq!(queue.submit(&[]), None);
    }

    #[test]
    fn hal_bind_resources_lowers_external_texture_planes_and_params() {
        let device = noop_device();
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 3,
                visibility: SHADER_STAGE_VERTEX | SHADER_STAGE_FRAGMENT,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::ExternalTexture),
            }],
            error: None,
        }));
        let external_texture = rgba_external_texture(&device);
        let bind_group = Arc::new(device.create_bind_group(
            Arc::clone(&layout),
            vec![BindGroupEntry {
                binding: 3,
                resource: BindGroupResource::ExternalTexture {
                    external_texture,
                    device: Arc::new(device.clone()),
                },
            }],
        ));
        assert!(!bind_group.is_error());

        let mut bind_groups = BTreeMap::new();
        bind_groups.insert(
            0,
            BoundBindGroup {
                group: bind_group,
                dynamic_offsets: Vec::new(),
            },
        );
        let resources = hal_bind_resources(
            &[layout],
            &[MetalBufferBinding {
                group: 0,
                binding: 3,
                metal_index: 4,
                ext_params_buffer_slot: Some(7),
                ext_params_vertex_buffer_slot: Some(12),
                ext_params_fragment_buffer_slot: Some(22),
                vertex_metal_index: Some(10),
                fragment_metal_index: Some(20),
                kind: MetalBindingKind::ExternalTexture,
            }],
            &bind_groups,
        )
        .expect("HAL resources");

        assert!(resources.buffers.is_empty());
        assert!(resources.textures.is_empty());
        assert!(resources.samplers.is_empty());
        assert_eq!(resources.external_textures.len(), 1);
        let binding = &resources.external_textures[0];
        assert_eq!(binding.group, 0);
        assert_eq!(binding.binding, 3);
        assert_eq!(binding.plane0_metal_index, 4);
        assert_eq!(binding.plane1_metal_index, 5);
        assert_eq!(binding.plane0_vertex_metal_index, Some(10));
        assert_eq!(binding.plane1_vertex_metal_index, Some(11));
        assert_eq!(binding.plane0_fragment_metal_index, Some(20));
        assert_eq!(binding.plane1_fragment_metal_index, Some(21));
        assert_eq!(binding.params_metal_index, 7);
        assert_eq!(binding.params_vertex_metal_index, Some(12));
        assert_eq!(binding.params_fragment_metal_index, Some(22));
        assert_eq!(binding.params_offset, 0);
        assert_eq!(binding.params_size, ExternalTextureParams::SIZE as u64);
    }

    #[test]
    fn noop_compute_pass_records_texture_and_sampler_bindings() {
        let device = noop_device();
        let layout = sampled_texture_bind_group_layout(&device, SHADER_STAGE_COMPUTE);
        let bind_group = sampled_texture_bind_group(&device, layout.clone());
        let pipeline_layout = explicit_pipeline_layout(&device, layout);
        let pipeline = sampled_compute_pipeline(&device, pipeline_layout);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert_eq!(device.queue().submit(&[Arc::new(command_buffer)]), None);

        let submitted = match device.queue().hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => Vec::new(),
        };
        // With per-kind counters texture-space and sampler-space are
        // independent, so both the sampled texture (binding=0) and the sampler
        // (binding=1) get slot 0 in their respective Metal index spaces.
        assert!(matches!(
            submitted.as_slice(),
            [HalCopy::ComputePass(pass)]
                if pass.bind_textures.len() == 1
                    && pass.bind_textures[0].group == 0
                    && pass.bind_textures[0].binding == 0
                    && pass.bind_textures[0].metal_index == 0
                    && pass.bind_textures[0].format == yawgpu_hal::HalTextureFormat::Rgba8Unorm
                    && pass.bind_textures[0].dimension == HalTextureViewDimension::D2
                    && pass.bind_textures[0].base_mip_level == 0
                    && pass.bind_textures[0].mip_level_count == 1
                    && pass.bind_textures[0].base_array_layer == 1
                    && pass.bind_textures[0].array_layer_count == 1
                    && pass.bind_textures[0].aspect == HalTextureAspect::All
                    && pass.bind_textures[0].storage_access.is_none()
                    && pass.bind_samplers.len() == 1
                    && pass.bind_samplers[0].group == 0
                    && pass.bind_samplers[0].binding == 1
                    && pass.bind_samplers[0].metal_index == 0
        ));
    }

    #[test]
    fn noop_compute_pass_records_storage_texture_binding() {
        let device = noop_device();
        let layout = storage_texture_bind_group_layout(&device);
        let bind_group = storage_texture_bind_group(&device, layout.clone());
        let pipeline_layout = explicit_pipeline_layout(&device, layout);
        let pipeline = storage_texture_compute_pipeline(&device, pipeline_layout);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert_eq!(device.queue().submit(&[Arc::new(command_buffer)]), None);

        let submitted = match device.queue().hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => Vec::new(),
        };
        assert!(matches!(
            submitted.as_slice(),
            [HalCopy::ComputePass(pass)]
                if pass.bind_textures.len() == 1
                    && pass.bind_textures[0].group == 0
                    && pass.bind_textures[0].binding == 0
                    && pass.bind_textures[0].metal_index == 0
                    && pass.bind_textures[0].format == yawgpu_hal::HalTextureFormat::Rgba8Unorm
                    && pass.bind_textures[0].dimension == HalTextureViewDimension::D2
                    && pass.bind_textures[0].storage_access == Some(HalStorageTextureAccess::ReadOnly)
                    && pass.bind_samplers.is_empty()
        ));
    }

    #[test]
    fn writable_storage_texture_pass_clears_before_pass_and_not_before_following_read() {
        let device = noop_device();
        let queue = device.queue();
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: SHADER_STAGE_COMPUTE,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: rgba8_unorm(),
                    view_dimension: TextureViewDimension::D2,
                }),
            }],
            error: None,
        }));
        let texture = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::STORAGE_BINDING | TextureUsage::COPY_SRC,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let (view, view_error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(1),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(view_error, None);
        let bind_group = Arc::new(device.create_bind_group(
            Arc::clone(&layout),
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::TextureView {
                    texture_view: Arc::new(view),
                    device: Arc::new(device.clone()),
                },
            }],
        ));
        let pipeline_layout = explicit_pipeline_layout(&device, layout);
        let pipeline = write_storage_texture_compute_pipeline(&device, pipeline_layout);
        let readback = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        }));
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.end(), None);
        assert_eq!(
            encoder.copy_texture_to_buffer(
                TexelCopyTextureInfo {
                    texture: Arc::clone(&texture),
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                TexelCopyBufferInfo {
                    buffer: readback,
                    device: None,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: None,
                        rows_per_image: None,
                    },
                },
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            ),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);

        assert_eq!(queue.submit(&[Arc::new(command_buffer)]), None);

        let submitted = match queue.hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => Vec::new(),
        };
        assert!(matches!(
            submitted.as_slice(),
            [
                HalCopy::ClearTexture(clear),
                HalCopy::ComputePass(_),
                HalCopy::TextureToBuffer(_)
            ] if clear.mip_level == 0 && clear.base_array_layer == 0
        ));
        assert!(texture.is_initialized(0, 0));
    }

    #[test]
    fn noop_render_pass_records_texture_and_sampler_bindings() {
        let device = noop_device();
        let layout =
            sampled_texture_bind_group_layout(&device, SHADER_STAGE_VERTEX | SHADER_STAGE_FRAGMENT);
        let bind_group = sampled_texture_bind_group(&device, layout.clone());
        let pipeline_layout = explicit_pipeline_layout(&device, layout);
        let pipeline = sampled_render_pipeline(&device, pipeline_layout);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert_eq!(device.queue().submit(&[Arc::new(command_buffer)]), None);

        let submitted = match device.queue().hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => Vec::new(),
        };
        // With per-kind per-stage counters both stages share the same Metal
        // index spaces.  The sampled texture (binding=0) gets texture-space
        // slot 0 and the sampler (binding=1) gets sampler-space slot 0; the
        // flat `metal_index` is the vertex-stage value in both cases.
        assert!(matches!(
            submitted.as_slice(),
            [HalCopy::ClearTexture(_), HalCopy::RenderPass(pass)]
                if pass.bind_textures.len() == 1
                    && pass.bind_textures[0].group == 0
                    && pass.bind_textures[0].binding == 0
                    && pass.bind_textures[0].metal_index == 0
                    && pass.bind_textures[0].format == yawgpu_hal::HalTextureFormat::Rgba8Unorm
                    && pass.bind_textures[0].dimension == HalTextureViewDimension::D2
                    && pass.bind_textures[0].base_mip_level == 0
                    && pass.bind_textures[0].mip_level_count == 1
                    && pass.bind_textures[0].base_array_layer == 1
                    && pass.bind_textures[0].array_layer_count == 1
                    && pass.bind_textures[0].aspect == HalTextureAspect::All
                    && pass.bind_textures[0].storage_access.is_none()
                    && pass.bind_samplers.len() == 1
                    && pass.bind_samplers[0].group == 0
                    && pass.bind_samplers[0].binding == 1
                    && pass.bind_samplers[0].metal_index == 0
        ));
    }

    fn submitted_render_pass_stencil_reference(reference: u32) -> u32 {
        let device = noop_device();
        let view = noop_render_attachment(&device);
        let pipeline = Arc::new(
            device
                .create_render_pipeline(render_pipeline_descriptor(render_shader_module(&device))),
        );
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.set_stencil_reference(reference), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, finish_error) = encoder.finish();
        assert_eq!(finish_error, None);
        assert_eq!(device.queue().submit(&[Arc::new(command_buffer)]), None);

        let submitted = match device.queue().hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => Vec::new(),
        };
        submitted
            .iter()
            .find_map(|copy| match copy {
                HalCopy::RenderPass(pass) => Some(pass.stencil_reference),
                _ => None,
            })
            .expect("render pass should be submitted")
    }

    #[test]
    fn render_pass_submit_masks_stencil_reference_to_stencil_bit_width() {
        assert_eq!(submitted_render_pass_stencil_reference(258), 2);
        assert_eq!(submitted_render_pass_stencil_reference(7), 7);
    }

    #[test]
    fn occlusion_query_draw_and_resolve_reach_noop_hal() {
        let device = noop_device();
        let view = noop_render_attachment(&device);
        let pipeline = Arc::new(
            device
                .create_render_pipeline(render_pipeline_descriptor(render_shader_module(&device))),
        );
        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "occlusion".to_owned(),
            kind: QueryType::Occlusion,
            count: 1,
        });
        assert_eq!(error, None);
        let query_set = Arc::new(query_set);
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::QUERY_RESOLVE,
            size: 256,
            mapped_at_creation: false,
        }));
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&noop_render_pass_descriptor(
            view,
            Some((*query_set).clone()),
        ));
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.begin_occlusion_query(0), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end_occlusion_query(), None);
        assert_eq!(pass.end(), None);
        assert_eq!(
            encoder.resolve_query_set(Arc::clone(&query_set), 0, 1, Arc::clone(&destination), 0),
            None
        );
        let (command_buffer, finish_error) = encoder.finish();
        assert_eq!(finish_error, None);
        assert!(matches!(
            command_buffer.command_ops(),
            [CommandExecution::RenderPass(pass), CommandExecution::ResolveQuerySet(resolve)]
                if pass.occlusion_query_index == Some(0)
                    && pass.occlusion_query_set.as_ref().is_some_and(|set| set.same(&query_set))
                    && resolve.query_set.same(&query_set)
                    && resolve.destination.same(&destination)
                    && resolve.first_query == 0
                    && resolve.query_count == 1
                    && resolve.destination_offset == 0
        ));

        assert_eq!(device.queue().submit(&[Arc::new(command_buffer)]), None);
        let submitted = match device.queue().hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => Vec::new(),
        };
        let mut saw_render_query_index = false;
        let mut resolve_destination = None;
        let mut written_queries = None;
        for copy in submitted {
            match copy {
                HalCopy::RenderPass(pass) => {
                    saw_render_query_index |=
                        pass.occlusion_query_index == Some(0) && pass.occlusion_query_set.is_some();
                }
                HalCopy::ResolveQuerySet(resolve) => {
                    written_queries = Some(resolve.written_queries);
                    resolve_destination = Some(resolve.destination);
                }
                _ => {}
            }
        }
        assert!(saw_render_query_index);
        assert_eq!(written_queries.as_deref(), Some(&[0][..]));
        let destination = resolve_destination.expect("query resolve should be submitted");
        assert_eq!(destination.read(0, 8).expect("read resolved bytes"), [0; 8]);
    }

    #[test]
    fn empty_occlusion_query_resolve_records_no_written_queries() {
        let device = noop_device();
        let view = noop_render_attachment(&device);
        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "occlusion".to_owned(),
            kind: QueryType::Occlusion,
            count: 1,
        });
        assert_eq!(error, None);
        let query_set = Arc::new(query_set);
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::QUERY_RESOLVE,
            size: 256,
            mapped_at_creation: false,
        }));
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&noop_render_pass_descriptor(
            view,
            Some((*query_set).clone()),
        ));
        assert_eq!(begin_error, None);
        assert_eq!(pass.begin_occlusion_query(0), None);
        assert_eq!(pass.end_occlusion_query(), None);
        assert_eq!(pass.end(), None);
        assert_eq!(
            encoder.resolve_query_set(Arc::clone(&query_set), 0, 1, Arc::clone(&destination), 0),
            None
        );
        let (command_buffer, finish_error) = encoder.finish();
        assert_eq!(finish_error, None);
        assert!(matches!(
            command_buffer.command_ops(),
            [CommandExecution::RenderPass(pass), CommandExecution::ResolveQuerySet(resolve)]
                if pass.occlusion_query_set.is_none()
                    && pass.occlusion_query_index.is_none()
                    && pass.draw.is_none()
                    && resolve.query_set.same(&query_set)
                    && resolve.destination.same(&destination)
                    && resolve.first_query == 0
                    && resolve.query_count == 1
                    && resolve.destination_offset == 0
        ));

        assert_eq!(device.queue().submit(&[Arc::new(command_buffer)]), None);
        let submitted = match device.queue().hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => Vec::new(),
        };
        let written_queries = submitted.into_iter().find_map(|copy| match copy {
            HalCopy::ResolveQuerySet(resolve) => Some(resolve.written_queries),
            _ => None,
        });
        assert_eq!(written_queries, Some(Vec::new()));
    }

    #[test]
    fn cross_command_buffer_occlusion_resolve_records_written_query() {
        let device = noop_device();
        let view = noop_render_attachment(&device);
        let pipeline = Arc::new(
            device
                .create_render_pipeline(render_pipeline_descriptor(render_shader_module(&device))),
        );
        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "occlusion".to_owned(),
            kind: QueryType::Occlusion,
            count: 1,
        });
        assert_eq!(error, None);
        let query_set = Arc::new(query_set);
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::QUERY_RESOLVE,
            size: 256,
            mapped_at_creation: false,
        }));

        let render_encoder = device.create_command_encoder();
        let (pass, begin_error) = render_encoder.begin_render_pass(&noop_render_pass_descriptor(
            view,
            Some((*query_set).clone()),
        ));
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.begin_occlusion_query(0), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end_occlusion_query(), None);
        assert_eq!(pass.end(), None);
        let (render_command_buffer, render_finish_error) = render_encoder.finish();
        assert_eq!(render_finish_error, None);

        let resolve_encoder = device.create_command_encoder();
        assert_eq!(
            resolve_encoder.resolve_query_set(
                Arc::clone(&query_set),
                0,
                1,
                Arc::clone(&destination),
                0,
            ),
            None
        );
        let (resolve_command_buffer, resolve_finish_error) = resolve_encoder.finish();
        assert_eq!(resolve_finish_error, None);

        assert_eq!(
            device.queue().submit(&[
                Arc::new(render_command_buffer),
                Arc::new(resolve_command_buffer),
            ]),
            None
        );
        let submitted = match device.queue().hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => Vec::new(),
        };
        let written_queries = submitted.into_iter().find_map(|copy| match copy {
            HalCopy::ResolveQuerySet(resolve) => Some(resolve.written_queries),
            _ => None,
        });
        assert_eq!(written_queries.as_deref(), Some(&[0][..]));
    }

    #[test]
    fn depth_only_render_pass_draw_records_render_pass_command() {
        let device = noop_device();
        let depth_view = noop_depth_view(&device);
        let pipeline = depth_only_pipeline(&device);
        let encoder = device.create_command_encoder();
        let (pass, error) =
            encoder.begin_render_pass(&depth_only_render_pass_descriptor(depth_view));
        assert_eq!(error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);

        assert!(matches!(
            command_buffer.command_ops(),
            [CommandExecution::RenderPass(pass)]
                if pass.color_attachments.is_empty()
                    && pass.depth_stencil_attachment.is_some()
                    && pass.draw.is_some()
        ));
    }

    #[test]
    fn depth_only_render_pass_submit_records_depth_stencil_hal_attachment() {
        let device = noop_device();
        let queue = device.queue();
        let depth_view = noop_depth_view(&device);
        let encoder = device.create_command_encoder();
        let (pass, error) =
            encoder.begin_render_pass(&depth_only_render_pass_descriptor(depth_view));
        assert_eq!(error, None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);

        assert_eq!(queue.submit(&[Arc::new(command_buffer)]), None);

        let submitted = match queue.hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => Vec::new(),
        };
        let pass = submitted
            .iter()
            .find_map(|copy| match copy {
                HalCopy::RenderPass(pass) => Some(pass),
                _ => None,
            })
            .expect("depth-only pass should submit a render pass");
        assert!(pass.color_targets.is_empty());
        let attachment = pass
            .depth_stencil_attachment
            .as_ref()
            .expect("render pass should carry depth-stencil attachment");
        assert_eq!(
            attachment.format,
            yawgpu_hal::HalTextureFormat::Depth32Float
        );
        assert!((attachment.depth_clear_value - 0.5).abs() < f32::EPSILON);
        assert!(pass.draw.is_none());
    }

    #[test]
    fn sparse_render_pass_submit_preserves_color_attachment_holes() {
        for real_slot in [0_usize, 1] {
            let device = noop_device();
            let queue = device.queue();
            let view = noop_render_attachment(&device);
            let pipeline = sparse_color_pipeline(&device, real_slot as u32);
            let encoder = device.create_command_encoder();
            let (pass, error) =
                encoder.begin_render_pass(&sparse_render_pass_descriptor(real_slot, view));
            assert_eq!(error, None);
            assert_eq!(pass.set_pipeline(pipeline), None);
            assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
            assert_eq!(pass.end(), None);
            let (command_buffer, error) = encoder.finish();
            assert_eq!(error, None);

            assert_eq!(queue.submit(&[Arc::new(command_buffer)]), None);

            let submitted = match queue.hal() {
                HalQueue::Noop(queue) => queue.submitted_copies(),
                _ => Vec::new(),
            };
            let pass = submitted
                .iter()
                .find_map(|copy| match copy {
                    HalCopy::RenderPass(pass) => Some(pass),
                    _ => None,
                })
                .expect("sparse color pass should submit a render pass");

            assert_eq!(pass.color_targets.len(), 2);
            assert!(pass.color_targets[real_slot].is_some());
            assert!(pass.color_targets[1 - real_slot].is_none());
            assert!(pass.draw.is_some());
        }
    }

    #[test]
    fn queue_wait_idle_noop_returns_ok() {
        let device = noop_device();
        let queue = device.queue();

        assert_eq!(queue.wait_idle(), None);
    }

    #[test]
    fn queue_write_buffer_then_map_read_resolves_after_wait_idle() {
        let device = noop_device();
        let queue = device.queue();
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::MAP_READ | BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        });

        assert_eq!(
            queue.write_buffer(QueueBufferWrite {
                device: device.hal(),
                buffer: &buffer,
                offset: 0,
                data: &[1, 2, 3, 4],
            }),
            None
        );
        assert_eq!(buffer.begin_map(MapMode::Read, 0, 4), Ok(()));
        assert_eq!(queue.wait_idle(), None);
        assert_eq!(
            buffer.resolve_pending_map_with_gpu_completion(|| true),
            MapAsyncStatus::Success
        );
    }

    // F-074: write_buffer staging-copy tests ---------------------------------

    /// Verifies that `write_buffer` via the staged-copy path makes data
    /// visible on Noop when the destination buffer is mapped for reading.
    ///
    /// The Noop `submit_copies` now executes `HalCopy::Buffer` eagerly, so
    /// the bytes written into the staging buffer appear in the destination
    /// before the map-read's `resolve_pending_map` reads them back.
    #[test]
    fn queue_write_buffer_staged_copy_data_visible_on_map_read() {
        let device = noop_device();
        let queue = device.queue();
        // 16 bytes; write 8 bytes at offset 8 (8-byte aligned, within bounds).
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::MAP_READ | BufferUsage::COPY_DST,
            size: 16,
            mapped_at_creation: false,
        });

        // Write [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80] at offset 8.
        let write_data: [u8; 8] = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80];
        assert_eq!(
            queue.write_buffer(QueueBufferWrite {
                device: device.hal(),
                buffer: &buffer,
                offset: 8,
                data: &write_data,
            }),
            None
        );

        // Map the full buffer for reading; the staged copy must already have
        // landed in the destination buffer storage.
        assert_eq!(buffer.begin_map(MapMode::Read, 0, 16), Ok(()));
        assert_eq!(queue.wait_idle(), None);
        let status = buffer.resolve_pending_map_with_gpu_completion(|| true);
        assert_eq!(status, MapAsyncStatus::Success);

        // getMappedRange requires 8-byte-aligned offset; read from offset 8.
        let ptr = buffer
            .mapped_range(true, 8, Some(8))
            .expect("mapped_range must succeed after resolved read map");
        // Safety: Noop host buffer; pointer is valid for `size` bytes.
        let read: Vec<u8> = unsafe { std::slice::from_raw_parts(ptr, 8).to_vec() };
        assert_eq!(read, write_data);

        assert_eq!(buffer.unmap(), None);
    }

    /// `write_buffer` with an out-of-bounds range must return a validation
    /// error and must NOT submit any copies.
    #[test]
    fn queue_write_buffer_oob_offset_returns_validation_error_and_no_copy() {
        let device = noop_device();
        let queue = device.queue();
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 8,
            mapped_at_creation: false,
        });

        // offset=8, data=[0,0,0,0] → end=12 > size=8 → OOB.
        let err = queue.write_buffer(QueueBufferWrite {
            device: device.hal(),
            buffer: &buffer,
            offset: 8,
            data: &[0, 0, 0, 0],
        });
        assert!(err.is_some());
        assert_eq!(
            err.unwrap().kind,
            ErrorKind::Validation,
            "OOB write must be a Validation error"
        );

        // No HalCopy should have been submitted.
        let submitted = match queue.hal() {
            HalQueue::Noop(q) => q.submitted_copies(),
            _ => Vec::new(),
        };
        assert!(
            submitted.is_empty(),
            "no copies must be submitted on validation failure"
        );
    }

    /// `write_buffer` on an error buffer must return a validation error and
    /// must NOT submit any copies.
    #[test]
    fn queue_write_buffer_error_buffer_returns_validation_error_and_no_copy() {
        let device = noop_device();
        let queue = device.queue();
        device.push_error_scope(ErrorFilter::Validation);
        let error_buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::NONE,
            size: 4,
            mapped_at_creation: false,
        });
        let _scope_error = device.pop_error_scope();

        let err = queue.write_buffer(QueueBufferWrite {
            device: device.hal(),
            buffer: &error_buffer,
            offset: 0,
            data: &[1, 2, 3, 4],
        });
        assert!(err.is_some());
        assert_eq!(
            err.unwrap().kind,
            ErrorKind::Validation,
            "write to error buffer must be a Validation error"
        );

        let submitted = match queue.hal() {
            HalQueue::Noop(q) => q.submitted_copies(),
            _ => Vec::new(),
        };
        assert!(
            submitted.is_empty(),
            "no copies must be submitted on validation failure"
        );
    }

    /// `write_buffer` with an empty slice must return `None` without submitting
    /// any copies (zero-length writes are a no-op after validation).
    #[test]
    fn queue_write_buffer_empty_data_is_noop_after_validation() {
        let device = noop_device();
        let queue = device.queue();
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 8,
            mapped_at_creation: false,
        });

        assert_eq!(
            queue.write_buffer(QueueBufferWrite {
                device: device.hal(),
                buffer: &buffer,
                offset: 0,
                data: &[],
            }),
            None,
            "empty write must succeed"
        );

        let submitted = match queue.hal() {
            HalQueue::Noop(q) => q.submitted_copies(),
            _ => Vec::new(),
        };
        assert!(
            submitted.is_empty(),
            "empty write must not submit any copies"
        );
    }

    /// Verifies that `write_buffer` submits a `HalCopy::Buffer` on the Noop
    /// queue, confirming the staging-copy dispatch path is taken.
    #[test]
    fn queue_write_buffer_submits_buffer_copy_to_hal() {
        let device = noop_device();
        let queue = device.queue();
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 8,
            mapped_at_creation: false,
        });

        assert_eq!(
            queue.write_buffer(QueueBufferWrite {
                device: device.hal(),
                buffer: &buffer,
                offset: 0,
                data: &[1, 2, 3, 4, 5, 6, 7, 8],
            }),
            None
        );

        let submitted = match queue.hal() {
            HalQueue::Noop(q) => q.submitted_copies(),
            _ => Vec::new(),
        };
        assert!(
            matches!(submitted.as_slice(), [HalCopy::Buffer(copy)] if copy.size == 8),
            "write_buffer must submit exactly one HalCopy::Buffer of the correct size"
        );
    }

    #[test]
    fn queue_write_texture_valid_call_submits_buffer_to_texture_copy() {
        let device = noop_device();
        let queue = device.queue();
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let layout = TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(16),
            rows_per_image: Some(4),
        };
        let data = [7_u8; 64];

        assert_eq!(
            queue.write_texture(QueueTextureWrite {
                device: device.hal(),
                texture: &texture,
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: 0 },
                write_size: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                aspect: TextureAspect::All,
                layout,
                data: &data,
            }),
            None
        );

        let submitted = match queue.hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => panic!("expected Noop queue"),
        };
        assert!(matches!(
            submitted.as_slice(),
            [HalCopy::BufferToTexture(copy)]
                if copy.mip_level == 0
                    && copy.origin.x == 0
                    && copy.origin.y == 0
                    && copy.origin.z == 0
                    && copy.extent.width == 4
                    && copy.extent.height == 4
                    && copy.extent.depth_or_array_layers == 1
                    && copy.buffer_layout.offset == 0
                    && copy.buffer_layout.bytes_per_row == 16
                    && copy.buffer_layout.rows_per_image == 4
        ));
    }

    #[test]
    fn queue_write_texture_partial_uninitialized_submits_clear_before_copy() {
        let device = noop_device();
        let queue = device.queue();
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let data = [7_u8; 16];

        assert_eq!(
            queue.write_texture(QueueTextureWrite {
                device: device.hal(),
                texture: &texture,
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: 0 },
                write_size: Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
                aspect: TextureAspect::All,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: Some(2),
                },
                data: &data,
            }),
            None
        );

        let submitted = match queue.hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => panic!("expected Noop queue"),
        };
        assert!(matches!(
            submitted.as_slice(),
            [HalCopy::ClearTexture(clear), HalCopy::BufferToTexture(copy)]
                if clear.mip_level == 0
                    && clear.base_array_layer == 0
                    && clear.array_layer_count == 1
                    && copy.mip_level == 0
                    && copy.extent.width == 2
                    && copy.extent.height == 2
        ));
        assert!(texture.is_initialized(0, 0));
    }

    #[test]
    fn depth_texture_copy_execution_does_not_submit_lazy_init_clear() {
        let device = noop_device();
        let source = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_SRC,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: depth32_float(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let destination = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: depth32_float(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let copy = TextureCopyCommand::TextureToTexture {
            source: TexelCopyTextureInfo {
                texture: Arc::clone(&source),
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: 0 },
                aspect: TextureAspect::DepthOnly,
            },
            destination: TexelCopyTextureInfo {
                texture: Arc::clone(&destination),
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: 0 },
                aspect: TextureAspect::DepthOnly,
            },
            copy_size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        };
        let mut copies = Vec::new();

        append_texture_copy_execution(&mut copies, &copy);

        assert!(matches!(copies.as_slice(), [HalCopy::TextureToTexture(_)]));
        assert!(!source.is_initialized(0, 0));
        assert!(!destination.is_initialized(0, 0));
    }

    #[test]
    fn render_pass_load_color_attachment_clears_before_pass_and_not_before_following_read() {
        let device = noop_device();
        let queue = device.queue();
        let texture = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let (view, view_error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(view_error, None);
        let readback = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        }));
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&RenderPassDescriptor {
            max_color_attachments: Limits::DEFAULT.max_color_attachments,
            color_attachments: vec![Some(RenderPassColorAttachment {
                view: Arc::new(view),
                depth_slice: None,
                resolve_target: None,
                load_op: LoadOp::Load,
                store_op: StoreOp::Store,
                clear_value: Color {
                    r: 0.0,
                    g: 1.0,
                    b: 0.0,
                    a: 1.0,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            max_draw_count: 50_000_000,
        });
        assert_eq!(begin_error, None);
        assert_eq!(pass.end(), None);
        assert_eq!(
            encoder.copy_texture_to_buffer(
                TexelCopyTextureInfo {
                    texture: Arc::clone(&texture),
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                TexelCopyBufferInfo {
                    buffer: Arc::clone(&readback),
                    device: None,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: None,
                        rows_per_image: None,
                    },
                },
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            ),
            None
        );
        let (command_buffer, finish_error) = encoder.finish();
        assert_eq!(finish_error, None);

        assert_eq!(queue.submit(&[Arc::new(command_buffer)]), None);

        let submitted = match queue.hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => panic!("expected Noop queue"),
        };
        assert!(matches!(
            submitted.as_slice(),
            [
                HalCopy::ClearTexture(clear),
                HalCopy::RenderPass(_),
                HalCopy::TextureToBuffer(_)
            ] if clear.mip_level == 0 && clear.base_array_layer == 0
        ));
        assert!(texture.is_initialized(0, 0));
    }

    #[test]
    fn render_pass_3d_color_attachment_clears_whole_mip_before_pass() {
        let device = noop_device();
        let queue = device.queue();
        let texture = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            dimension: TextureDimension::D3,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 4,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let (view, view_error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D3),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(view_error, None);
        let readback = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 1024,
            mapped_at_creation: false,
        }));
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&RenderPassDescriptor {
            max_color_attachments: Limits::DEFAULT.max_color_attachments,
            color_attachments: vec![Some(RenderPassColorAttachment {
                view: Arc::new(view),
                depth_slice: Some(2),
                resolve_target: None,
                load_op: LoadOp::Load,
                store_op: StoreOp::Store,
                clear_value: Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            max_draw_count: 50_000_000,
        });
        assert_eq!(begin_error, None);
        assert_eq!(pass.end(), None);
        assert_eq!(
            encoder.copy_texture_to_buffer(
                TexelCopyTextureInfo {
                    texture: Arc::clone(&texture),
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                TexelCopyBufferInfo {
                    buffer: readback,
                    device: None,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 4,
                },
            ),
            None
        );
        let (command_buffer, finish_error) = encoder.finish();
        assert_eq!(finish_error, None);

        assert_eq!(queue.submit(&[Arc::new(command_buffer)]), None);

        let submitted = match queue.hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => panic!("expected Noop queue"),
        };
        assert!(matches!(
            submitted.as_slice(),
            [
                HalCopy::ClearTexture(clear),
                HalCopy::RenderPass(_),
                HalCopy::TextureToBuffer(_)
            ] if clear.mip_level == 0
                && clear.base_array_layer == 0
                && clear.array_layer_count == 1
        ));
        assert!(texture.is_initialized(0, 0));
    }

    #[test]
    fn queue_write_texture_invalid_call_returns_validation_error() {
        let device = noop_device();
        let queue = device.queue();
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_SRC,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let error = queue
            .write_texture(QueueTextureWrite {
                device: device.hal(),
                texture: &texture,
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: 0 },
                write_size: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                aspect: TextureAspect::All,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(16),
                    rows_per_image: Some(4),
                },
                data: &[0_u8; 64],
            })
            .expect("missing validation error");

        assert_eq!(error.kind, ErrorKind::Validation);
        let submitted = match queue.hal() {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            _ => panic!("expected Noop queue"),
        };
        assert!(submitted.is_empty());
    }

    #[test]
    fn zero_size_buffer_copy_is_not_emitted_to_hal_and_submits_as_noop() {
        let device = noop_device();
        let queue = device.queue();
        let source = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 4,
            mapped_at_creation: false,
        }));
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        }));
        let copy = BufferCopyCommand {
            source: Arc::clone(&source),
            source_offset: 0,
            destination: Arc::clone(&destination),
            destination_offset: 0,
            size: 0,
        };

        assert!(hal_command_execution(&CommandExecution::BufferCopy(copy)).is_none());

        let encoder = device.create_command_encoder();
        assert_eq!(
            encoder.copy_buffer_to_buffer(Arc::clone(&source), 0, Arc::clone(&destination), 0, 0),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert_eq!(queue.submit(&[Arc::new(command_buffer)]), None);
    }

    #[test]
    fn zero_size_clear_buffer_submits_as_noop() {
        let device = noop_device();
        let queue = device.queue();
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        }));

        let encoder = device.create_command_encoder();
        assert_eq!(encoder.clear_buffer(Arc::clone(&buffer), 0, 0), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(matches!(
            command_buffer.command_ops(),
            [CommandExecution::BufferClear(clear)] if clear.buffer.same(&buffer) && clear.size == 0
        ));
        assert!(hal_command_execution(&command_buffer.command_ops()[0]).is_none());
        assert_eq!(queue.submit(&[Arc::new(command_buffer)]), None);
    }

    #[test]
    fn clear_buffer_execution_maps_to_hal_buffer_clear() {
        let device = noop_device();
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 16,
            mapped_at_creation: false,
        }));
        let clear = BufferClearCommand {
            buffer,
            offset: 4,
            size: 8,
        };

        assert!(matches!(
            hal_command_execution(&CommandExecution::BufferClear(clear)),
            Some(HalCopy::BufferClear(clear)) if clear.offset == 4 && clear.size == 8
        ));
    }

    // --- repack_texel_rows unit tests ---

    #[test]
    fn repack_texel_rows_two_rows_depth_one_strips_padding() {
        // 2 rows, depth=1, row_bytes=8, src stride=257 (non-block-aligned).
        // Source layout: [row0(8 bytes) | padding(249 bytes)] [row1(8 bytes)]
        // Total source consumed: offset(0) + 257 + 8 = 265 bytes.
        let mut data = vec![0u8; 265];
        // Fill row0 with 0x01, row1 with 0x02.
        data[..8].fill(0x01);
        data[257..265].fill(0x02);
        let result = repack_texel_rows(
            &data, /*src_offset=*/ 0, /*src_bytes_per_row=*/ 257,
            /*src_rows_per_image=*/ 2, /*row_bytes=*/ 8, /*height_blocks=*/ 2,
            /*depth=*/ 1,
        )
        .expect("repack should succeed");

        assert_eq!(result.len(), 16);
        assert_eq!(&result[..8], &[0x01; 8]);
        assert_eq!(&result[8..], &[0x02; 8]);
    }

    #[test]
    fn repack_texel_rows_depth_two_skips_image_padding() {
        // 2 images, 2 rows each, row_bytes=4, src stride=8, rows_per_image=3
        // (i.e. one row of padding between images).
        // Image 0: row0=[0x01;4] + pad(4), row1=[0x02;4] + pad(4), image-pad-row=[0xFF;4] + pad(4)
        // Image 1: row0=[0x03;4] + pad(4), row1=[0x04;4] + pad(4)
        // Total source: offset(0) + 3*8 + 2*8 = 40 bytes.
        let mut data = vec![0u8; 40];
        // Image 0 row 0
        data[0..4].fill(0x01);
        // Image 0 row 1
        data[8..12].fill(0x02);
        // Image 0 padding row (bytes 16..24) left as 0.
        // Image 1 row 0 starts at offset 3*8=24
        data[24..28].fill(0x03);
        // Image 1 row 1 at offset 24+8=32
        data[32..36].fill(0x04);

        let result = repack_texel_rows(
            &data, /*src_offset=*/ 0, /*src_bytes_per_row=*/ 8,
            /*src_rows_per_image=*/ 3, /*row_bytes=*/ 4, /*height_blocks=*/ 2,
            /*depth=*/ 2,
        )
        .expect("repack should succeed");

        // Output: image0 row0 + image0 row1 + image1 row0 + image1 row1 = 4*4 = 16 bytes.
        assert_eq!(result.len(), 16);
        assert_eq!(&result[0..4], &[0x01; 4]);
        assert_eq!(&result[4..8], &[0x02; 4]);
        assert_eq!(&result[8..12], &[0x03; 4]);
        assert_eq!(&result[12..16], &[0x04; 4]);
    }

    #[test]
    fn repack_texel_rows_adds_source_offset_after_stride() {
        let mut data = vec![0u8; 17];
        data[1..17].copy_from_slice(&[0xA5; 16]);

        let result = repack_texel_rows(
            &data, /*src_offset=*/ 1, /*src_bytes_per_row=*/ 256,
            /*src_rows_per_image=*/ 1, /*row_bytes=*/ 16, /*height_blocks=*/ 1,
            /*depth=*/ 1,
        )
        .expect("repack should read from offset one, not offset times stride");

        assert_eq!(result, [0xA5; 16]);
    }

    #[test]
    fn repack_texel_rows_source_offset_combines_with_image_and_row_index() {
        let mut data = vec![0u8; 43];
        data[3..7].fill(0x01);
        data[11..15].fill(0x02);
        data[27..31].fill(0x03);
        data[35..39].fill(0x04);

        let result = repack_texel_rows(
            &data, /*src_offset=*/ 3, /*src_bytes_per_row=*/ 8,
            /*src_rows_per_image=*/ 3, /*row_bytes=*/ 4, /*height_blocks=*/ 2,
            /*depth=*/ 2,
        )
        .expect("repack should combine image, row, and source offset");

        assert_eq!(result.len(), 16);
        assert_eq!(&result[0..4], &[0x01; 4]);
        assert_eq!(&result[4..8], &[0x02; 4]);
        assert_eq!(&result[8..12], &[0x03; 4]);
        assert_eq!(&result[12..16], &[0x04; 4]);
    }

    // --- write_texture with unaligned bytes_per_row integration tests ---

    /// Creates a 2x2 rgba8unorm COPY_DST texture on the noop device.
    fn noop_2x2_rgba8_texture(device: &Device) -> Texture {
        device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        })
    }

    #[test]
    fn write_texture_unaligned_bytes_per_row_succeeds_on_noop() {
        // rgba8unorm: block_size=4, so bytes_per_row=257 is not 4-aligned.
        // 2 rows need bytes_per_row specified. Source has stride 257 and 8 final bytes.
        // Data must satisfy: offset(0) + (height_blocks-1)*bpr + row_bytes
        //   = 0 + 1*257 + 8 = 265 bytes.
        let device = noop_device();
        let queue = device.queue();
        let texture = noop_2x2_rgba8_texture(&device);
        let data = vec![7_u8; 265];

        let error = queue.write_texture(QueueTextureWrite {
            device: device.hal(),
            texture: &texture,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            write_size: Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            aspect: TextureAspect::All,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(257),
                rows_per_image: None,
            },
            data: &data,
        });
        // Must succeed: no error on noop backend.
        assert_eq!(error, None);

        // On the noop HAL the submitted copy should carry the packed layout:
        // offset=0, bytes_per_row=8 (2 texels * 4 bytes), rows_per_image=2.
        let submitted = match queue.hal() {
            HalQueue::Noop(q) => q.submitted_copies(),
            _ => panic!("expected Noop queue"),
        };
        assert!(matches!(
            submitted.as_slice(),
            [HalCopy::BufferToTexture(copy)]
                if copy.buffer_layout.offset == 0
                    && copy.buffer_layout.bytes_per_row == 8
                    && copy.buffer_layout.rows_per_image == 2
        ));
    }

    #[test]
    fn write_texture_aligned_bytes_per_row_uses_verbatim_path() {
        // bytes_per_row=16 is 4-aligned (block_size=4): fast path, layout unchanged.
        let device = noop_device();
        let queue = device.queue();
        let texture = noop_2x2_rgba8_texture(&device);
        let data = vec![5_u8; 32]; // offset(0) + 1*16 + 8 = 24; 32 ≥ 24.

        let error = queue.write_texture(QueueTextureWrite {
            device: device.hal(),
            texture: &texture,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            write_size: Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            aspect: TextureAspect::All,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: None,
            },
            data: &data,
        });
        assert_eq!(error, None);

        let submitted = match queue.hal() {
            HalQueue::Noop(q) => q.submitted_copies(),
            _ => panic!("expected Noop queue"),
        };
        // Verbatim path: layout forwarded as given.
        assert!(matches!(
            submitted.as_slice(),
            [HalCopy::BufferToTexture(copy)]
                if copy.buffer_layout.offset == 0
                    && copy.buffer_layout.bytes_per_row == 16
                    && copy.buffer_layout.rows_per_image == 2
        ));
    }

    // --- F-079: destroyed bind-group resources / timestamp query sets must error at submit ----

    /// A bind group containing a destroyed uniform buffer must not cause an error at
    /// create_bind_group, set_bind_group, or finish(). The validation error must fire
    /// exclusively at queue.submit (WebGPU spec §17.3 "Queue submit validation").
    ///
    /// Mirrors CTS api,validation,encoding,cmds,setBindGroup:state_and_binding_index
    /// with state="destroyed", resourceType="buffer", encoderType="compute pass".
    #[test]
    fn destroyed_bind_group_buffer_errors_at_submit_not_at_encode() {
        let device = noop_device();

        // Create a uniform-buffer bind group layout.
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: SHADER_STAGE_COMPUTE,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: 0,
                }),
            }],
            error: None,
        }));

        // Create a uniform buffer and immediately destroy it before the bind group is created.
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::UNIFORM,
            size: 16,
            mapped_at_creation: false,
        }));
        buffer.destroy();
        assert!(
            buffer.is_destroyed(),
            "buffer must be destroyed before bind group creation"
        );

        // create_bind_group with a destroyed buffer must succeed (not an error bind group).
        let bind_group = Arc::new(device.create_bind_group(
            Arc::clone(&layout),
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::Buffer {
                    buffer: Arc::clone(&buffer),
                    device: Arc::new(device.clone()),
                    offset: 0,
                    size: u64::MAX,
                },
            }],
        ));
        assert!(
            !bind_group.is_error(),
            "bind group with destroyed buffer must not be an error bind group"
        );

        // set_bind_group on the compute pass must not produce an error.
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None, "begin_compute_pass must succeed");
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None,
            "set_bind_group with destroyed-buffer bind group must not error during encoding"
        );
        assert_eq!(pass.end(), None);

        // finish() must succeed (no error).
        let (command_buffer, finish_error) = encoder.finish();
        assert_eq!(
            finish_error, None,
            "finish must succeed when destroyed buffer is in bind group"
        );
        assert!(!command_buffer.is_error());

        // submit() must fail with the destroyed-buffer validation error.
        let submit_error = device
            .queue()
            .submit(&[Arc::new(command_buffer)])
            .expect("submit must fail when bind group references a destroyed buffer");
        assert_eq!(
            submit_error.message, "queue submit cannot use a destroyed buffer",
            "submit error message must identify the destroyed buffer"
        );
    }

    /// A bind group containing a destroyed sampled texture must not cause an error at
    /// create_bind_group, set_bind_group, or finish(). The validation error must fire
    /// exclusively at queue.submit (WebGPU spec §17.3 "Queue submit validation").
    ///
    /// Mirrors CTS api,validation,encoding,cmds,setBindGroup:state_and_binding_index
    /// with state="destroyed", resourceType="texture", encoderType="compute pass".
    #[test]
    fn destroyed_bind_group_texture_errors_at_submit_not_at_encode() {
        let device = noop_device();

        // Create a sampled-texture bind group layout.
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: SHADER_STAGE_COMPUTE,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::Texture {
                    sample_type: TextureSampleType::UnfilterableFloat,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                }),
            }],
            error: None,
        }));

        // Create a sampled texture, get a view, then destroy the underlying texture.
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let (view, view_error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(view_error, None);
        // Destroy the texture before creating the bind group (mirrors CTS destroyAfterCreate=true).
        texture.destroy();
        assert!(
            texture.is_destroyed(),
            "texture must be destroyed before bind group creation"
        );

        // create_bind_group with a destroyed texture must succeed.
        let bind_group = Arc::new(device.create_bind_group(
            Arc::clone(&layout),
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::TextureView {
                    texture_view: Arc::new(view),
                    device: Arc::new(device.clone()),
                },
            }],
        ));
        assert!(
            !bind_group.is_error(),
            "bind group with destroyed texture must not be an error bind group"
        );

        // set_bind_group on the compute pass must not produce an error.
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None,
            "set_bind_group with destroyed-texture bind group must not error during encoding"
        );
        assert_eq!(pass.end(), None);

        // finish() must succeed.
        let (command_buffer, finish_error) = encoder.finish();
        assert_eq!(
            finish_error, None,
            "finish must succeed when destroyed texture is in bind group"
        );
        assert!(!command_buffer.is_error());

        // submit() must fail with the destroyed-texture validation error.
        let submit_error = device
            .queue()
            .submit(&[Arc::new(command_buffer)])
            .expect("submit must fail when bind group references a destroyed texture");
        assert_eq!(
            submit_error.message, "queue submit cannot use a destroyed texture",
            "submit error message must identify the destroyed texture"
        );
    }

    /// A render pass with a destroyed timestamp-query-set in timestampWrites must not
    /// cause an error at begin_render_pass or finish(). The validation error must fire
    /// exclusively at queue.submit (WebGPU spec §17.3 "Queue submit validation").
    ///
    /// Mirrors CTS api,validation,queue,destroyed,query_set:timestamps with
    /// querySetState="destroyed" (render-pass sub-case).
    #[test]
    fn destroyed_timestamp_query_set_in_render_pass_errors_at_submit_not_at_encode() {
        // Timestamp query requires the TimestampQuery feature.
        let mut features = FeatureSet::new();
        features.insert(crate::Feature::TimestampQuery);
        let device = crate::Device::from_hal(hal_noop_device(), Limits::DEFAULT, features, "", "");

        let (query_set, qs_error) = device.create_query_set(QuerySetDescriptor {
            label: "ts".to_owned(),
            kind: QueryType::Timestamp,
            count: 2,
        });
        assert_eq!(qs_error, None);
        // Destroy the query set before using it in the render pass.
        query_set.destroy();
        assert!(query_set.is_destroyed());

        // Build a minimal render pass with the destroyed timestamp query set.
        let view = noop_render_attachment(&device);
        let descriptor = RenderPassDescriptor {
            max_color_attachments: Limits::DEFAULT.max_color_attachments,
            color_attachments: vec![Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_value: Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: Some(RenderPassTimestampWrites {
                query_set: query_set.clone(),
                beginning_index: Some(0),
                end_index: Some(1),
            }),
            max_draw_count: 50_000_000,
        };

        let encoder = device.create_command_encoder();
        // begin_render_pass must succeed (no error from the destroyed query set at encode time).
        let (pass, begin_error) = encoder.begin_render_pass(&descriptor);
        assert_eq!(
            begin_error, None,
            "begin_render_pass must succeed when timestamp query set is destroyed"
        );
        assert_eq!(pass.end(), None);

        // finish() must succeed.
        let (command_buffer, finish_error) = encoder.finish();
        assert_eq!(
            finish_error, None,
            "finish must succeed when timestamp query set is destroyed"
        );
        assert!(!command_buffer.is_error());

        // submit() must fail.
        let submit_error = device
            .queue()
            .submit(&[Arc::new(command_buffer)])
            .expect("submit must fail when timestamp query set is destroyed");
        assert_eq!(
            submit_error.message, "queue submit cannot use a destroyed query set",
            "submit error message must identify the destroyed query set"
        );
    }
}
