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
use crate::queue::*;
use crate::render_pass::*;
use crate::render_pipeline::*;
use crate::sampler::*;
use crate::shader::*;
use crate::shader_naga;
use crate::texture::*;
use crate::texture_view::*;

#[derive(Debug, Clone)]
pub struct RenderBundleEncoderDescriptor {
    pub max_color_attachments: u32,
    pub color_formats: Vec<Option<TextureFormat>>,
    pub depth_stencil_format: Option<TextureFormat>,
    pub sample_count: u32,
    pub depth_read_only: bool,
    pub stencil_read_only: bool,
}

#[derive(Debug, Clone)]
pub struct RenderBundleEncoder {
    pub(crate) inner: Arc<RenderBundleEncoderInner>,
}

#[derive(Debug, Clone)]
pub struct RenderBundle {
    pub(crate) inner: Arc<RenderBundleInner>,
}

#[derive(Debug)]
pub(crate) struct RenderBundleEncoderInner {
    pub(crate) descriptor: RenderBundleEncoderDescriptor,
    pub(crate) state: Mutex<RenderBundleEncoderState>,
}

#[derive(Debug)]
pub(crate) struct RenderBundleEncoderState {
    pub(crate) lifecycle: RenderBundleEncoderLifecycle,
    pub(crate) first_error: Option<String>,
    pub(crate) pass_state: PassEncoderState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderBundleEncoderLifecycle {
    Recording,
    Errored,
    Finished,
}

#[derive(Debug)]
pub(crate) struct RenderBundleInner {
    pub(crate) is_error: bool,
    pub(crate) attachment_signature: AttachmentSignature,
}

impl RenderBundleEncoder {
    #[must_use]
    pub fn new(
        descriptor: RenderBundleEncoderDescriptor,
        limits: Limits,
    ) -> (Self, Option<String>) {
        let descriptor_error = validate_render_bundle_encoder_descriptor(&descriptor, limits).err();
        let attachment_signature = descriptor.attachment_signature();
        (
            Self {
                inner: Arc::new(RenderBundleEncoderInner {
                    descriptor,
                    state: Mutex::new(RenderBundleEncoderState {
                        lifecycle: if descriptor_error.is_some() {
                            RenderBundleEncoderLifecycle::Errored
                        } else {
                            RenderBundleEncoderLifecycle::Recording
                        },
                        first_error: None,
                        pass_state: PassEncoderState::new(
                            Some(attachment_signature),
                            Vec::new(),
                            None,
                            None,
                        ),
                    }),
                }),
            },
            descriptor_error,
        )
    }

    pub fn finish(&self) -> (RenderBundle, Option<String>) {
        let mut state = self.inner.state.lock();
        match state.lifecycle {
            RenderBundleEncoderLifecycle::Errored => {
                state.lifecycle = RenderBundleEncoderLifecycle::Finished;
                return (
                    RenderBundle::new(self.inner.descriptor.attachment_signature(), true),
                    None,
                );
            }
            RenderBundleEncoderLifecycle::Finished => {
                return (
                    RenderBundle::new(self.inner.descriptor.attachment_signature(), true),
                    Some("render bundle encoder cannot be finished more than once".to_owned()),
                );
            }
            RenderBundleEncoderLifecycle::Recording => {}
        }
        state.lifecycle = RenderBundleEncoderLifecycle::Finished;
        if state.first_error.is_none() && state.pass_state.debug_group_depth != 0 {
            record_first_error_option(
                &mut state.first_error,
                "render bundle debug group stack is unbalanced".to_owned(),
            );
        }
        let error = state.first_error.clone();
        (
            RenderBundle::new(
                self.inner.descriptor.attachment_signature(),
                error.is_some(),
            ),
            error,
        )
    }

    pub fn insert_debug_marker(&self) -> Option<String> {
        self.record_bundle_command(|_| Ok(()))
    }

    pub fn push_debug_group(&self) -> Option<String> {
        self.record_bundle_command(|state| {
            state.debug_group_depth = state.debug_group_depth.saturating_add(1);
            Ok(())
        })
    }

    pub fn pop_debug_group(&self) -> Option<String> {
        self.record_bundle_command(|state| {
            if state.debug_group_depth == 0 {
                Err("render bundle debug group stack is empty".to_owned())
            } else {
                state.debug_group_depth -= 1;
                Ok(())
            }
        })
    }

    pub fn set_pipeline(&self, pipeline: Arc<RenderPipeline>) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_bundle_pipeline(&self.inner.descriptor, &pipeline)?;
            state.render_pipeline = Some(pipeline);
            Ok(())
        })
    }

    pub fn set_bind_group(
        &self,
        index: u32,
        group: Option<Arc<BindGroup>>,
        dynamic_offsets: Vec<u32>,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            if let Some(group) = group {
                state.bind_groups.insert(
                    index,
                    BoundBindGroup {
                        group,
                        dynamic_offsets,
                    },
                );
            } else {
                state.bind_groups.remove(&index);
            }
            Ok(())
        })
    }

    pub fn set_vertex_buffer(
        &self,
        slot: u32,
        buffer: Option<Arc<Buffer>>,
        offset: u64,
        size: u64,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_vertex_buffer_slot(slot, limits)?;
            if let Some(buffer) = buffer {
                let size = validate_set_vertex_buffer(&buffer, offset, size)?;
                state.vertex_buffers.insert(
                    slot,
                    BoundVertexBuffer {
                        buffer,
                        offset,
                        size,
                    },
                );
            } else {
                validate_clear_vertex_buffer(offset, size)?;
                state.vertex_buffers.remove(&slot);
            }
            Ok(())
        })
    }

    pub fn set_index_buffer(
        &self,
        buffer: Arc<Buffer>,
        format: Option<IndexFormat>,
        offset: u64,
        size: u64,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            let format = format.ok_or_else(|| "render pass index format is invalid".to_owned())?;
            let size = validate_set_index_buffer(&buffer, format, offset, size)?;
            state.index_buffer = Some(BoundIndexBuffer {
                buffer,
                format,
                offset,
                size,
            });
            Ok(())
        })
    }

    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_draw_state(
                state,
                RenderDrawKind::Direct {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                },
                limits,
            )
        })
    }

    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        _base_vertex: i32,
        first_instance: u32,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_draw_state(
                state,
                RenderDrawKind::IndexedDirect {
                    index_count,
                    instance_count,
                    first_index,
                    first_instance,
                },
                limits,
            )
        })
    }

    pub fn draw_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::Indirect, limits)?;
            validate_indirect_buffer(&indirect_buffer, indirect_offset, 16, "draw indirect")
        })
    }

    pub fn draw_indexed_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::IndexedIndirect, limits)?;
            validate_indirect_buffer(
                &indirect_buffer,
                indirect_offset,
                20,
                "draw indexed indirect",
            )
        })
    }

    pub(crate) fn record_bundle_command<F>(&self, command: F) -> Option<String>
    where
        F: FnOnce(&mut PassEncoderState) -> Result<(), String>,
    {
        let mut state = self.inner.state.lock();
        match state.lifecycle {
            RenderBundleEncoderLifecycle::Recording => {}
            RenderBundleEncoderLifecycle::Errored => return None,
            RenderBundleEncoderLifecycle::Finished => {
                return Some("render bundle encoder cannot record after finish".to_owned());
            }
        }
        if let Err(message) = command(&mut state.pass_state) {
            record_first_error_option(&mut state.first_error, message);
        }
        None
    }
}

impl RenderBundle {
    pub(crate) fn new(attachment_signature: AttachmentSignature, is_error: bool) -> Self {
        Self {
            inner: Arc::new(RenderBundleInner {
                is_error,
                attachment_signature,
            }),
        }
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub(crate) fn attachment_signature(&self) -> &AttachmentSignature {
        &self.inner.attachment_signature
    }
}

impl RenderBundleEncoderDescriptor {
    pub(crate) fn attachment_signature(&self) -> AttachmentSignature {
        AttachmentSignature {
            color_formats: self.color_formats.clone(),
            depth_stencil_format: self.depth_stencil_format,
            sample_count: self.sample_count,
        }
    }
}

pub(crate) fn validate_render_bundle_encoder_descriptor(
    descriptor: &RenderBundleEncoderDescriptor,
    _limits: Limits,
) -> Result<(), String> {
    if descriptor.color_formats.len() > descriptor.max_color_attachments as usize {
        return Err("render bundle colorFormatCount exceeds the device limit".to_owned());
    }
    if descriptor.sample_count != 1 && descriptor.sample_count != 4 {
        return Err("render bundle sampleCount must be 1 or 4".to_owned());
    }

    let mut has_attachment = descriptor.depth_stencil_format.is_some();
    for color_format in descriptor.color_formats.iter().flatten().copied() {
        has_attachment = true;
        let Some(caps) = color_format.caps() else {
            return Err("render bundle color format must be defined".to_owned());
        };
        if !caps.aspects.color || !caps.renderable {
            return Err("render bundle color format must be color-renderable".to_owned());
        }
    }
    if let Some(depth_format) = descriptor.depth_stencil_format {
        let Some(caps) = depth_format.caps() else {
            return Err("render bundle depthStencilFormat must be defined".to_owned());
        };
        if !caps.aspects.depth && !caps.aspects.stencil {
            return Err(
                "render bundle depthStencilFormat must have depth or stencil aspect".to_owned(),
            );
        }
    }
    if !has_attachment {
        return Err("render bundle requires at least one attachment format".to_owned());
    }
    Ok(())
}

pub(crate) fn validate_render_bundle_pipeline(
    descriptor: &RenderBundleEncoderDescriptor,
    pipeline: &RenderPipeline,
) -> Result<(), String> {
    if pipeline.is_error() {
        return Err("render bundle requires a valid render pipeline".to_owned());
    }
    if pipeline.attachment_signature() != descriptor.attachment_signature() {
        return Err("render bundle pipeline attachment signature is incompatible".to_owned());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let bind_group = empty_bind_group(&device);
        let vertex_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::VERTEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let index_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::INDEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let (bundle_encoder, error) =
            RenderBundleEncoder::new(render_bundle_encoder_descriptor(), device.limits());
        assert_eq!(error, None);

        assert_eq!(bundle_encoder.insert_debug_marker(), None);
        assert_eq!(bundle_encoder.push_debug_group(), None);
        assert_eq!(bundle_encoder.pop_debug_group(), None);
        assert_eq!(bundle_encoder.set_pipeline(pipeline), None);
        assert_eq!(
            bundle_encoder.set_bind_group(0, Some(bind_group), Vec::new()),
            None
        );
        assert_eq!(
            bundle_encoder.set_vertex_buffer(0, Some(vertex_buffer), 0, 16, device.limits()),
            None
        );
        assert_eq!(
            bundle_encoder.set_index_buffer(index_buffer, Some(IndexFormat::Uint16), 0, 16),
            None
        );
        assert_eq!(bundle_encoder.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(
            bundle_encoder.draw_indexed(3, 1, 0, 0, 0, device.limits()),
            None
        );
        let (bundle, error) = bundle_encoder.finish();
        assert_eq!(error, None);
        assert!(!bundle.is_error());
    }

    #[test]
    fn render_bundle_encoder_indirect_draws() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let index_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::INDEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let indirect = noop_indirect_buffer(&device);
        let (bundle_encoder, error) =
            RenderBundleEncoder::new(render_bundle_encoder_descriptor(), device.limits());
        assert_eq!(error, None);

        assert_eq!(bundle_encoder.set_pipeline(pipeline), None);
        assert_eq!(
            bundle_encoder.set_index_buffer(index_buffer, Some(IndexFormat::Uint16), 0, 16),
            None
        );
        assert_eq!(
            bundle_encoder.draw_indirect(indirect.clone(), 0, device.limits()),
            None
        );
        assert_eq!(
            bundle_encoder.draw_indexed_indirect(indirect, 0, device.limits()),
            None
        );
        let (bundle, error) = bundle_encoder.finish();
        assert_eq!(error, None);
        assert!(!bundle.is_error());
    }
}
