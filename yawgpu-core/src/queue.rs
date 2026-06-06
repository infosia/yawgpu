use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{
    HalBoundBuffer, HalBoundIndexBuffer, HalBoundIndirectBuffer, HalBoundSampler, HalBoundTexture,
    HalBufferClear, HalBufferCopy, HalBufferTextureCopy, HalBufferTextureLayout, HalBufferUsage,
    HalComputePass, HalCopy, HalDevice, HalDraw, HalIndexFormat, HalQueue, HalRenderColorTarget,
    HalRenderDepthStencilAttachment, HalRenderLoadOp, HalRenderPass, HalTextureAspect,
    HalTextureCopy, HalTextureViewDimension,
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
use crate::render_pipeline::*;
#[cfg(feature = "tiled")]
use crate::subpass::*;
use crate::texture::hal_texture_format;
use crate::texture::*;
use crate::texture_view::{TextureAspect, TextureViewDimension};

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

    /// Writes `data` into the buffer at `offset` directly from the queue.
    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Option<DeviceError> {
        buffer.write_from_queue(offset, data)
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
        let staging = device.create_buffer(
            data_size,
            HalBufferUsage {
                copy_src: true,
                ..HalBufferUsage::default()
            },
        );
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
        self.inner
            .hal
            .submit_copies(&[copy])
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
    let bindings = hal_bind_resources(
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
    let RenderDrawExecution::Direct {
        vertex_count,
        instance_count,
        first_vertex,
        first_instance,
    } = draw.draw
    else {
        return None;
    };
    Some(HalSubpassDraw {
        subpass_index: draw.subpass_index,
        pipeline: draw.pipeline.hal()?,
        bind_buffers: bindings.buffers,
        bind_textures: bindings.textures,
        bind_samplers: bindings.samplers,
        vertex_buffers,
        draw: HalDraw::Direct {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
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
        workgroups: pass.workgroups,
    }))
}

/// Returns HAL render pass execution.
pub(crate) fn hal_render_pass_execution(pass: &RenderPassCommand) -> Option<HalCopy> {
    let (
        pipeline,
        bind_buffers,
        bind_textures,
        bind_samplers,
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
        vertex_buffers,
        index_buffer,
        indirect_buffer,
        blend_constant: pass.blend_constant,
        stencil_reference: pass.stencil_reference,
        draw,
    }))
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
    attachments: &[RenderPassColorExecution],
) -> Option<Vec<HalRenderColorTarget>> {
    attachments
        .iter()
        .map(|attachment| {
            Some(HalRenderColorTarget {
                texture: attachment.texture.hal()?,
                resolve_target: match &attachment.resolve_target {
                    Some(texture) => Some(texture.hal()?),
                    None => None,
                },
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
            })
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
            depth_load_op: hal_render_load_op(attachment.depth_load_op),
            depth_store: matches!(attachment.depth_store_op, StoreOp::Store),
            depth_clear_value: attachment.depth_clear_value,
            depth_read_only: attachment.depth_read_only,
            stencil_load_op: hal_render_load_op(attachment.stencil_load_op),
            stencil_store: matches!(attachment.stencil_store_op, StoreOp::Store),
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
                    sampler: sampler.hal()?,
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
        });
        assert_eq!(error, None);
        Arc::new(view)
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

        assert_eq!(queue.write_buffer(&buffer, 0, &[1, 2, 3, 4]), None);
        assert_eq!(queue.submit(&[]), None);
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
                    && pass.bind_samplers[0].metal_index == 1
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
        assert!(matches!(
            submitted.as_slice(),
            [HalCopy::RenderPass(pass)]
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
                    && pass.bind_samplers[0].metal_index == 1
        ));
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

        assert_eq!(queue.write_buffer(&buffer, 0, &[1, 2, 3, 4]), None);
        assert_eq!(buffer.begin_map(MapMode::Read, 0, 4), Ok(()));
        assert_eq!(queue.wait_idle(), None);
        assert_eq!(
            buffer.resolve_pending_map_with_gpu_completion(|| true),
            MapAsyncStatus::Success
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
}
