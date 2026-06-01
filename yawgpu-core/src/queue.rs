use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{
    HalBoundBuffer, HalBufferClear, HalBufferCopy, HalBufferTextureCopy, HalBufferTextureLayout,
    HalComputePass, HalCopy, HalDraw, HalQueue, HalRenderColorTarget, HalRenderLoadOp,
    HalRenderPass, HalTextureCopy,
};
#[cfg(feature = "tiled")]
use yawgpu_hal::{
    HalSubpassAttachmentLayout, HalSubpassAttachmentResource, HalSubpassColorAttachment,
    HalSubpassDependency, HalSubpassDependencyType, HalSubpassDepthStencilAttachment,
    HalSubpassDraw, HalSubpassInputAttachment, HalSubpassLayout, HalSubpassPassLayout,
    HalSubpassRenderPassCommand,
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
#[cfg(feature = "tiled")]
use crate::subpass::*;
#[cfg(feature = "tiled")]
use crate::texture::hal_texture_format;
use crate::texture::*;

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

    /// Writes `data` into the buffer at `offset` directly from the queue.
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Option<DeviceError> {
        buffer.write_from_queue(offset, data)
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
        for command_buffer in command_buffers {
            for op in command_buffer.command_ops() {
                if let Some(copy) = hal_command_execution(op) {
                    copies.push(copy);
                }
            }
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
            #[cfg(feature = "tiled")]
            CommandExecution::SubpassRenderPass(pass) => {
                for attachment in &pass.color_attachments {
                    push_subpass_resource_textures(&mut textures, &attachment.resource);
                }
                if let Some(attachment) = &pass.depth_stencil_attachment {
                    push_subpass_resource_textures(&mut textures, &attachment.resource);
                }
            }
            CommandExecution::BufferCopy(_)
            | CommandExecution::BufferClear(_)
            | CommandExecution::ComputePass(_) => {}
        }
    }
    textures
}

// Persistent subpass attachments are backed by user texture views whose
// textures may be destroyed; transient attachments are not user textures and
// cannot be destroyed, so they are skipped.
#[cfg(feature = "tiled")]
fn push_subpass_resource_textures(
    textures: &mut Vec<Texture>,
    resource: &SubpassAttachmentResource,
) {
    if let SubpassAttachmentResource::Persistent {
        view,
        resolve_target,
    } = resource
    {
        textures.push(view.texture());
        if let Some(resolve_target) = resolve_target {
            textures.push(resolve_target.texture());
        }
    }
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
pub(crate) fn hal_command_execution(op: &CommandExecution) -> Option<HalCopy> {
    match op {
        CommandExecution::BufferCopy(copy) => {
            if copy.size == 0 {
                return None;
            }
            let source = copy.source.hal()?;
            let destination = copy.destination.hal()?;
            Some(HalCopy::Buffer(HalBufferCopy {
                source,
                source_offset: copy.source_offset,
                destination,
                destination_offset: copy.destination_offset,
                size: copy.size,
            }))
        }
        CommandExecution::BufferClear(clear) => {
            if clear.size == 0 {
                return None;
            }
            let buffer = clear.buffer.hal()?;
            Some(HalCopy::BufferClear(HalBufferClear {
                buffer,
                offset: clear.offset,
                size: clear.size,
            }))
        }
        CommandExecution::TextureCopy(copy) => hal_texture_copy_execution(copy),
        CommandExecution::ComputePass(pass) => hal_compute_pass_execution(pass),
        CommandExecution::RenderPass(pass) => hal_render_pass_execution(pass),
        #[cfg(feature = "tiled")]
        CommandExecution::SubpassRenderPass(pass) => hal_subpass_render_pass_execution(pass),
    }
}

#[cfg(feature = "tiled")]
fn hal_subpass_render_pass_execution(pass: &SubpassRenderPassCommand) -> Option<HalCopy> {
    Some(HalCopy::SubpassRenderPass(HalSubpassRenderPassCommand {
        layout: hal_subpass_pass_layout(pass.layout.descriptor()),
        extent: hal_extent(pass.extent),
        color_attachments: pass
            .color_attachments
            .iter()
            .map(hal_subpass_color_attachment)
            .collect::<Option<Vec<_>>>()?,
        depth_stencil_attachment: match &pass.depth_stencil_attachment {
            Some(attachment) => Some(hal_subpass_depth_stencil_attachment(attachment)?),
            None => None,
        },
        draws: pass
            .draws
            .iter()
            .map(hal_subpass_draw_execution)
            .collect::<Option<Vec<_>>>()?,
    }))
}

#[cfg(feature = "tiled")]
fn hal_subpass_draw_execution(draw: &SubpassDrawExecution) -> Option<HalSubpassDraw> {
    let bind_buffers = hal_bind_buffers(
        draw.pipeline.bind_group_layouts(),
        draw.pipeline.metal_bindings(),
        &draw.bind_groups,
    )?;
    let mut vertex_buffers = Vec::new();
    for binding in draw.pipeline.vertex_buffer_bindings() {
        let bound = draw.vertex_buffers.get(&binding.slot)?;
        vertex_buffers.push(HalBoundBuffer {
            group: 0,
            binding: binding.slot,
            metal_index: binding.metal_index,
            buffer: bound.buffer.hal()?,
            offset: bound.offset,
            size: bound.size,
        });
    }
    Some(HalSubpassDraw {
        subpass_index: draw.subpass_index,
        pipeline: draw.pipeline.hal()?,
        bind_buffers,
        vertex_buffers,
        draw: HalDraw {
            vertex_count: draw.draw.vertex_count,
            instance_count: draw.draw.instance_count,
            first_vertex: draw.draw.first_vertex,
            first_instance: draw.draw.first_instance,
        },
    })
}

#[cfg(feature = "tiled")]
fn hal_subpass_pass_layout(layout: &SubpassPassLayoutDescriptor) -> HalSubpassPassLayout {
    HalSubpassPassLayout {
        color_attachments: layout
            .color_attachments
            .iter()
            .map(|attachment| HalSubpassAttachmentLayout {
                format: hal_texture_format(attachment.format),
                sample_count: attachment.sample_count,
            })
            .collect(),
        depth_stencil_attachment: layout.depth_stencil_attachment.map(|attachment| {
            HalSubpassAttachmentLayout {
                format: hal_texture_format(attachment.format),
                sample_count: attachment.sample_count,
            }
        }),
        subpasses: layout
            .subpasses
            .iter()
            .map(|subpass| HalSubpassLayout {
                color_attachment_indices: subpass.color_attachment_indices.clone(),
                uses_depth_stencil: subpass.uses_depth_stencil,
                input_attachments: subpass
                    .input_attachments
                    .iter()
                    .map(|input| HalSubpassInputAttachment {
                        group: input.group,
                        binding: input.binding,
                        source_subpass: input.source_subpass,
                        source_attachment: input.source_attachment,
                    })
                    .collect(),
            })
            .collect(),
        dependencies: layout
            .dependencies
            .iter()
            .map(|dependency| HalSubpassDependency {
                src_subpass: dependency.src_subpass,
                dst_subpass: dependency.dst_subpass,
                dependency_type: match dependency.dependency_type {
                    SubpassDependencyType::ColorToInput => HalSubpassDependencyType::ColorToInput,
                    SubpassDependencyType::DepthToInput => HalSubpassDependencyType::DepthToInput,
                    SubpassDependencyType::ColorDepthToInput => {
                        HalSubpassDependencyType::ColorDepthToInput
                    }
                },
                by_region: dependency.by_region,
            })
            .collect(),
    }
}

#[cfg(feature = "tiled")]
fn hal_subpass_color_attachment(
    attachment: &SubpassColorAttachmentBinding,
) -> Option<HalSubpassColorAttachment> {
    Some(HalSubpassColorAttachment {
        resource: hal_subpass_attachment_resource(&attachment.resource)?,
        load_op: hal_load_op(attachment.load_op),
        store: matches!(attachment.store_op, StoreOp::Store),
        clear_color: [
            attachment.clear_value.r,
            attachment.clear_value.g,
            attachment.clear_value.b,
            attachment.clear_value.a,
        ],
    })
}

#[cfg(feature = "tiled")]
fn hal_subpass_depth_stencil_attachment(
    attachment: &SubpassDepthStencilAttachmentBinding,
) -> Option<HalSubpassDepthStencilAttachment> {
    Some(HalSubpassDepthStencilAttachment {
        resource: hal_subpass_attachment_resource(&attachment.resource)?,
        depth_load_op: hal_load_op(attachment.depth_load_op),
        depth_store: matches!(attachment.depth_store_op, StoreOp::Store),
        depth_clear_value: attachment.depth_clear_value,
        stencil_load_op: hal_load_op(attachment.stencil_load_op),
        stencil_store: matches!(attachment.stencil_store_op, StoreOp::Store),
        stencil_clear_value: attachment.stencil_clear_value,
    })
}

#[cfg(feature = "tiled")]
fn hal_subpass_attachment_resource(
    resource: &SubpassAttachmentResource,
) -> Option<HalSubpassAttachmentResource> {
    match resource {
        SubpassAttachmentResource::Persistent {
            view,
            resolve_target,
        } => Some(HalSubpassAttachmentResource::Persistent {
            texture: view.texture().hal()?,
            resolve_target: match resolve_target {
                Some(view) => Some(view.texture().hal()?),
                None => None,
            },
        }),
        SubpassAttachmentResource::Transient(attachment) => {
            Some(HalSubpassAttachmentResource::Transient(attachment.hal()?))
        }
    }
}

#[cfg(feature = "tiled")]
fn hal_load_op(load_op: LoadOp) -> HalRenderLoadOp {
    match load_op {
        LoadOp::Load => HalRenderLoadOp::Load,
        LoadOp::Clear | LoadOp::Undefined => HalRenderLoadOp::Clear,
    }
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

fn extent_is_empty(extent: Extent3d) -> bool {
    extent.width == 0 || extent.height == 0 || extent.depth_or_array_layers == 0
}

/// Returns HAL compute pass execution.
pub(crate) fn hal_compute_pass_execution(pass: &ComputePassCommand) -> Option<HalCopy> {
    let pipeline = pass.pipeline.hal()?;
    let mut bind_buffers = Vec::new();
    for binding in pass.pipeline.metal_bindings() {
        let bound = pass.bind_groups.get(&binding.group)?;
        let entry = bound
            .group
            .entries()
            .iter()
            .find(|entry| entry.binding == binding.binding)?;
        let BindGroupResource::Buffer {
            buffer,
            offset,
            size,
            ..
        } = &entry.resource
        else {
            return None;
        };
        let dynamic_offset = dynamic_offset_for_binding(
            pass.pipeline.bind_group_layouts(),
            binding.group,
            binding.binding,
            &bound.dynamic_offsets,
        )?;
        let offset = offset.checked_add(dynamic_offset)?;
        bind_buffers.push(HalBoundBuffer {
            group: binding.group,
            binding: binding.binding,
            metal_index: binding.metal_index,
            buffer: buffer.hal()?,
            offset,
            size: *size,
        });
    }
    Some(HalCopy::ComputePass(HalComputePass {
        pipeline,
        bind_buffers,
        workgroups: pass.workgroups,
    }))
}

/// Returns HAL render pass execution.
pub(crate) fn hal_render_pass_execution(pass: &RenderPassCommand) -> Option<HalCopy> {
    let (pipeline, bind_buffers, vertex_buffers, draw) =
        if let (Some(pipeline), Some(draw)) = (&pass.pipeline, pass.draw) {
            let bind_buffers = hal_bind_buffers(
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
                    buffer: bound.buffer.hal()?,
                    offset: bound.offset,
                    size: bound.size,
                });
            }
            (
                Some(pipeline.hal()?),
                bind_buffers,
                vertex_buffers,
                Some(HalDraw {
                    vertex_count: draw.vertex_count,
                    instance_count: draw.instance_count,
                    first_vertex: draw.first_vertex,
                    first_instance: draw.first_instance,
                }),
            )
        } else {
            (None, Vec::new(), Vec::new(), None)
        };
    Some(HalCopy::RenderPass(HalRenderPass {
        pipeline,
        color_target: HalRenderColorTarget {
            texture: pass.color_attachment.texture.hal()?,
            load_op: match pass.color_attachment.load_op {
                LoadOp::Load => HalRenderLoadOp::Load,
                LoadOp::Clear | LoadOp::Undefined => HalRenderLoadOp::Clear,
            },
            store: matches!(pass.color_attachment.store_op, StoreOp::Store),
            clear_color: [
                pass.color_attachment.clear_value.r,
                pass.color_attachment.clear_value.g,
                pass.color_attachment.clear_value.b,
                pass.color_attachment.clear_value.a,
            ],
        },
        bind_buffers,
        vertex_buffers,
        draw,
    }))
}

/// Returns HAL bind buffers.
pub(crate) fn hal_bind_buffers(
    layouts: &[Arc<BindGroupLayout>],
    metal_bindings: &[MetalBufferBinding],
    bind_groups: &BTreeMap<u32, BoundBindGroup>,
) -> Option<Vec<HalBoundBuffer>> {
    let mut bind_buffers = Vec::new();
    for binding in metal_bindings {
        let bound = bind_groups.get(&binding.group)?;
        let entry = bound
            .group
            .entries()
            .iter()
            .find(|entry| entry.binding == binding.binding)?;
        let BindGroupResource::Buffer {
            buffer,
            offset,
            size,
            ..
        } = &entry.resource
        else {
            return None;
        };
        let dynamic_offset = dynamic_offset_for_binding(
            layouts,
            binding.group,
            binding.binding,
            &bound.dynamic_offsets,
        )?;
        let offset = offset.checked_add(dynamic_offset)?;
        bind_buffers.push(HalBoundBuffer {
            group: binding.group,
            binding: binding.binding,
            metal_index: binding.metal_index,
            buffer: buffer.hal()?,
            offset,
            size: *size,
        });
    }
    Some(bind_buffers)
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
    use crate::test_helpers::*;

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

        assert_eq!(queue.write_buffer(&buffer, 0, &[1, 2, 3, 4]), None);
        assert_eq!(queue.submit(&[]), None);
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
}
