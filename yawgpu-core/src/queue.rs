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
use crate::format::*;
use crate::future::*;
use crate::instance::*;
use crate::limits::*;
use crate::pass::*;
use crate::pipeline_layout::*;
use crate::query_set::*;
use crate::render_bundle::*;
use crate::render_pass::*;
use crate::render_pipeline::*;
use crate::sampler::*;
use crate::shader::*;
use crate::shader_naga;
use crate::texture::*;
use crate::texture_view::*;

#[derive(Debug, Clone)]
pub struct Queue {
    pub(crate) inner: Arc<QueueInner>,
}

#[derive(Debug)]
pub(crate) struct QueueInner {
    pub(crate) hal: HalQueue,
    pub(crate) label: Mutex<String>,
}

impl Queue {
    #[must_use]
    pub fn from_hal(hal: HalQueue, label: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(QueueInner {
                hal,
                label: Mutex::new(label.into()),
            }),
        }
    }

    #[must_use]
    pub fn hal(&self) -> &HalQueue {
        &self.inner.hal
    }

    pub fn set_label(&self, label: &str) {
        *self.inner.label.lock() = label.to_owned();
    }

    #[must_use]
    pub fn label(&self) -> String {
        self.inner.label.lock().clone()
    }

    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Option<DeviceError> {
        buffer.write_from_queue(offset, data)
    }

    pub fn submit(&self, command_buffers: &[Arc<CommandBuffer>]) -> Option<DeviceError> {
        for (index, command_buffer) in command_buffers.iter().enumerate() {
            if command_buffer.is_error() {
                return Some(DeviceError::validation(
                    "queue submit cannot use an error command buffer",
                ));
            }
            if command_buffer.is_submitted() {
                return Some(DeviceError::validation(
                    "command buffer cannot be submitted more than once",
                ));
            }
            if command_buffers[..index]
                .iter()
                .any(|previous| previous.same(command_buffer))
            {
                return Some(DeviceError::validation(
                    "command buffer cannot be submitted more than once",
                ));
            }
            for buffer in command_buffer.referenced_buffers() {
                if buffer.map_state() != BufferMapState::Unmapped {
                    return Some(DeviceError::validation(
                        "queue submit cannot use a mapped buffer",
                    ));
                }
                if buffer.is_destroyed() {
                    return Some(DeviceError::validation(
                        "queue submit cannot use a destroyed buffer",
                    ));
                }
            }
        }
        for command_buffer in command_buffers {
            if let Err(message) = command_buffer.mark_submitted() {
                return Some(DeviceError::validation(message));
            }
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

fn hal_buffer_texture_layout(
    layout: TexelCopyBufferLayout,
    texture: &Texture,
    copy_size: Extent3d,
) -> Option<HalBufferTextureLayout> {
    let format_caps = texture.format().caps()?;
    let width_blocks = crate::copy::div_ceil_u32(copy_size.width, format_caps.block_w);
    let height_blocks = crate::copy::div_ceil_u32(copy_size.height, format_caps.block_h);
    let row_bytes = width_blocks.checked_mul(format_caps.texel_block_size)?;
    Some(HalBufferTextureLayout {
        offset: layout.offset,
        bytes_per_row: layout.bytes_per_row.unwrap_or(row_bytes),
        rows_per_image: layout.rows_per_image.unwrap_or(height_blocks),
    })
}

pub(crate) fn hal_command_execution(op: &CommandExecution) -> Option<HalCopy> {
    match op {
        CommandExecution::BufferCopy(copy) => {
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
        CommandExecution::TextureCopy(copy) => hal_texture_copy_execution(copy),
        CommandExecution::ComputePass(pass) => hal_compute_pass_execution(pass),
        CommandExecution::RenderPass(pass) => hal_render_pass_execution(pass),
    }
}

pub(crate) fn hal_texture_copy_execution(copy: &TextureCopyCommand) -> Option<HalCopy> {
    match copy {
        TextureCopyCommand::BufferToTexture {
            source,
            destination,
            copy_size,
        } => {
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
    use crate::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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
}
