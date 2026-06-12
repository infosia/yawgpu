use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::bind_group::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::device::FeatureSet;
use crate::format::*;
use crate::limits::*;
use crate::pass::*;
use crate::render_pipeline::*;
use crate::texture::*;

/// Describes render bundle encoder descriptor.
#[derive(Debug, Clone)]
pub struct RenderBundleEncoderDescriptor {
    /// Max color attachments.
    pub max_color_attachments: u32,
    /// Color formats.
    pub color_formats: Vec<Option<TextureFormat>>,
    /// Depth stencil format.
    pub depth_stencil_format: Option<TextureFormat>,
    /// Sample count.
    pub sample_count: u32,
    /// Depth read only.
    pub depth_read_only: bool,
    /// Stencil read only.
    pub stencil_read_only: bool,
}

/// Records commands for the RenderBundleEncoder.
#[derive(Debug, Clone)]
pub struct RenderBundleEncoder {
    pub(crate) inner: Arc<RenderBundleEncoderInner>,
}

/// Stores render bundle data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct RenderBundle {
    pub(crate) inner: Arc<RenderBundleInner>,
}

/// Holds shared state for the render bundle encoder handle.
#[derive(Debug)]
pub(crate) struct RenderBundleEncoderInner {
    pub(crate) descriptor: RenderBundleEncoderDescriptor,
    pub(crate) state: Mutex<RenderBundleEncoderState>,
}

/// Tracks the lifecycle state for render bundle encoder.
#[derive(Debug)]
pub(crate) struct RenderBundleEncoderState {
    pub(crate) lifecycle: RenderBundleEncoderLifecycle,
    pub(crate) first_error: Option<String>,
    pub(crate) pass_state: PassEncoderState,
    pub(crate) draws: Vec<RenderBundleDraw>,
}

/// Enumerates render bundle encoder lifecycle values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderBundleEncoderLifecycle {
    /// Recording variant.
    Recording,
    /// Errored variant.
    Errored,
    /// Finished variant.
    Finished,
}

/// Holds shared state for the render bundle handle.
#[derive(Debug)]
pub(crate) struct RenderBundleInner {
    pub(crate) is_error: bool,
    pub(crate) attachment_signature: AttachmentSignature,
    pub(crate) referenced_buffers: Vec<Arc<Buffer>>,
    pub(crate) referenced_textures: Vec<Texture>,
    pub(crate) buffer_uses: Vec<BufferScopeUse>,
    pub(crate) texture_uses: Vec<TextureScopeUse>,
    pub(crate) draws: Vec<RenderBundleDraw>,
}

/// Stores one render bundle draw with the state snapshotted at record time.
#[derive(Debug, Clone)]
pub(crate) struct RenderBundleDraw {
    pub(crate) pipeline: Arc<RenderPipeline>,
    pub(crate) bind_groups: BTreeMap<u32, BoundBindGroup>,
    pub(crate) vertex_buffers: BTreeMap<u32, BoundVertexBuffer>,
    pub(crate) index_buffer: Option<BoundIndexBuffer>,
    pub(crate) indirect_buffer: Option<BoundIndirectBuffer>,
    pub(crate) draw: RenderDrawExecution,
}

impl RenderBundleEncoder {
    /// Creates a new instance.
    #[must_use]
    pub fn new(
        descriptor: RenderBundleEncoderDescriptor,
        limits: Limits,
        features: FeatureSet,
    ) -> (Self, Option<String>) {
        let descriptor_error =
            validate_render_bundle_encoder_descriptor(&descriptor, limits, &features).err();
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
                            limits,
                            PassEncoderInit {
                                attachment_signature: Some(attachment_signature),
                                render_extent: None,
                                attachment_textures: Vec::new(),
                                render_color_attachments: Vec::new(),
                                render_depth_stencil_attachment: None,
                                occlusion_query_set: None,
                                max_draw_count: u64::MAX,
                            },
                        ),
                        draws: Vec::new(),
                    }),
                }),
            },
            descriptor_error,
        )
    }

    /// Finishes recording and returns the completed object.
    pub fn finish(&self) -> (RenderBundle, Option<String>) {
        let mut state = self.inner.state.lock();
        match state.lifecycle {
            RenderBundleEncoderLifecycle::Errored => {
                state.lifecycle = RenderBundleEncoderLifecycle::Finished;
                return (
                    RenderBundle::new(
                        self.inner.descriptor.attachment_signature(),
                        true,
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    ),
                    None,
                );
            }
            RenderBundleEncoderLifecycle::Finished => {
                return (
                    RenderBundle::new(
                        self.inner.descriptor.attachment_signature(),
                        true,
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    ),
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
        let referenced_buffers = render_bundle_referenced_buffers(&state.pass_state, &state.draws);
        let referenced_textures =
            render_bundle_referenced_textures(&state.pass_state, &state.draws);
        let buffer_uses = state.pass_state.scope_buffer_uses.clone();
        let texture_uses = state.pass_state.scope_texture_uses.clone();
        let draws = state.draws.clone();
        (
            RenderBundle::new(
                self.inner.descriptor.attachment_signature(),
                error.is_some(),
                referenced_buffers,
                referenced_textures,
                buffer_uses,
                texture_uses,
                draws,
            ),
            error,
        )
    }

    /// Records a debug marker into the bundle.
    pub fn insert_debug_marker(&self) -> Option<String> {
        self.record_bundle_command(|_| Ok(None))
    }

    /// Opens a debug group in the bundle.
    pub fn push_debug_group(&self) -> Option<String> {
        self.record_bundle_command(|state| {
            state.debug_group_depth = state.debug_group_depth.saturating_add(1);
            Ok(None)
        })
    }

    /// Closes the most recently opened debug group in the bundle.
    pub fn pop_debug_group(&self) -> Option<String> {
        self.record_bundle_command(|state| {
            if state.debug_group_depth == 0 {
                Err("render bundle debug group stack is empty".to_owned())
            } else {
                state.debug_group_depth -= 1;
                Ok(None)
            }
        })
    }

    /// Sets pipeline on this object or encoder.
    pub fn set_pipeline(&self, pipeline: Arc<RenderPipeline>) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_bundle_pipeline(&self.inner.descriptor, &pipeline)?;
            state.render_pipeline = Some(pipeline);
            Ok(None)
        })
    }

    /// Records a validation error against this render bundle encoder.
    pub fn record_validation_error(&self, message: impl Into<String>) -> Option<String> {
        let message = message.into();
        self.record_bundle_command(|_| Err(message))
    }

    /// Sets bind group on this object or encoder.
    pub fn set_bind_group(
        &self,
        index: u32,
        group: Option<Arc<BindGroup>>,
        dynamic_offsets: Vec<u32>,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_set_bind_group(index, group.as_deref(), &dynamic_offsets, limits)?;
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
            Ok(None)
        })
    }

    /// Sets vertex buffer on this object or encoder.
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
            Ok(None)
        })
    }

    /// Sets index buffer on this object or encoder.
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
            Ok(None)
        })
    }

    /// Records a draw command.
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
            )?;
            let pipeline = Arc::clone(
                state
                    .render_pipeline
                    .as_ref()
                    .ok_or_else(|| "render bundle requires a render pipeline".to_owned())?,
            );
            let bind_group_layouts = pipeline.bind_group_layouts().to_vec();
            record_pipeline_usage_scope(state, &bind_group_layouts, &[])?;
            Ok(Some(render_bundle_draw_snapshot(
                state,
                pipeline,
                None,
                RenderDrawExecution::Direct {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                },
            )))
        })
    }

    /// Records an indexed draw into the bundle after validation.
    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
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
            )?;
            let pipeline = Arc::clone(
                state
                    .render_pipeline
                    .as_ref()
                    .ok_or_else(|| "render bundle requires a render pipeline".to_owned())?,
            );
            let bind_group_layouts = pipeline.bind_group_layouts().to_vec();
            record_pipeline_usage_scope(state, &bind_group_layouts, &[])?;
            Ok(Some(render_bundle_draw_snapshot(
                state,
                pipeline,
                None,
                RenderDrawExecution::Indexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                },
            )))
        })
    }

    /// Records an indirect draw into the bundle after validation.
    pub fn draw_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::Indirect, limits)?;
            let pipeline = Arc::clone(
                state
                    .render_pipeline
                    .as_ref()
                    .ok_or_else(|| "render bundle requires a render pipeline".to_owned())?,
            );
            let bind_group_layouts = pipeline.bind_group_layouts().to_vec();
            record_pipeline_usage_scope(state, &bind_group_layouts, &[])?;
            validate_indirect_buffer(&indirect_buffer, indirect_offset, 16, "draw indirect")?;
            state
                .command_referenced_buffers
                .push(Arc::clone(&indirect_buffer));
            Ok(Some(render_bundle_draw_snapshot(
                state,
                pipeline,
                Some(BoundIndirectBuffer {
                    buffer: indirect_buffer,
                    offset: indirect_offset,
                }),
                RenderDrawExecution::Indirect {
                    offset: indirect_offset,
                },
            )))
        })
    }

    /// Records an indexed indirect draw into the bundle after validation.
    pub fn draw_indexed_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::IndexedIndirect, limits)?;
            let pipeline = Arc::clone(
                state
                    .render_pipeline
                    .as_ref()
                    .ok_or_else(|| "render bundle requires a render pipeline".to_owned())?,
            );
            let bind_group_layouts = pipeline.bind_group_layouts().to_vec();
            record_pipeline_usage_scope(state, &bind_group_layouts, &[])?;
            validate_indirect_buffer(
                &indirect_buffer,
                indirect_offset,
                20,
                "draw indexed indirect",
            )?;
            state
                .command_referenced_buffers
                .push(Arc::clone(&indirect_buffer));
            Ok(Some(render_bundle_draw_snapshot(
                state,
                pipeline,
                Some(BoundIndirectBuffer {
                    buffer: indirect_buffer,
                    offset: indirect_offset,
                }),
                RenderDrawExecution::IndexedIndirect {
                    offset: indirect_offset,
                },
            )))
        })
    }

    /// Records a command into the bundle, validating encoder state first.
    pub(crate) fn record_bundle_command<F>(&self, command: F) -> Option<String>
    where
        F: FnOnce(&mut PassEncoderState) -> Result<Option<RenderBundleDraw>, String>,
    {
        let mut state = self.inner.state.lock();
        match state.lifecycle {
            RenderBundleEncoderLifecycle::Recording => {}
            RenderBundleEncoderLifecycle::Errored => return None,
            RenderBundleEncoderLifecycle::Finished => {
                return Some("render bundle encoder cannot record after finish".to_owned());
            }
        }
        match command(&mut state.pass_state) {
            Ok(Some(draw)) => state.draws.push(draw),
            Ok(None) => {}
            Err(message) => record_first_error_option(&mut state.first_error, message),
        }
        None
    }
}

impl RenderBundle {
    /// Creates a new instance.
    pub(crate) fn new(
        attachment_signature: AttachmentSignature,
        is_error: bool,
        referenced_buffers: Vec<Arc<Buffer>>,
        referenced_textures: Vec<Texture>,
        buffer_uses: Vec<BufferScopeUse>,
        texture_uses: Vec<TextureScopeUse>,
        draws: Vec<RenderBundleDraw>,
    ) -> Self {
        Self {
            inner: Arc::new(RenderBundleInner {
                is_error,
                attachment_signature,
                referenced_buffers,
                referenced_textures,
                buffer_uses,
                texture_uses,
                draws,
            }),
        }
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns the attachment signature used for render pass compatibility checks.
    #[must_use]
    pub(crate) fn attachment_signature(&self) -> &AttachmentSignature {
        &self.inner.attachment_signature
    }

    /// Returns buffers referenced by this bundle.
    pub(crate) fn referenced_buffers(&self) -> &[Arc<Buffer>] {
        &self.inner.referenced_buffers
    }

    /// Returns textures referenced by this bundle.
    pub(crate) fn referenced_textures(&self) -> &[Texture] {
        &self.inner.referenced_textures
    }

    pub(crate) fn buffer_uses(&self) -> &[BufferScopeUse] {
        &self.inner.buffer_uses
    }

    pub(crate) fn texture_uses(&self) -> &[TextureScopeUse] {
        &self.inner.texture_uses
    }

    pub(crate) fn draws(&self) -> &[RenderBundleDraw] {
        &self.inner.draws
    }
}

fn render_bundle_draw_snapshot(
    state: &PassEncoderState,
    pipeline: Arc<RenderPipeline>,
    indirect_buffer: Option<BoundIndirectBuffer>,
    draw: RenderDrawExecution,
) -> RenderBundleDraw {
    RenderBundleDraw {
        pipeline,
        bind_groups: state.bind_groups.clone(),
        vertex_buffers: state.vertex_buffers.clone(),
        index_buffer: state.index_buffer.clone(),
        indirect_buffer,
        draw,
    }
}

fn render_bundle_referenced_buffers(
    state: &PassEncoderState,
    draws: &[RenderBundleDraw],
) -> Vec<Arc<Buffer>> {
    let mut buffers = Vec::new();
    for bound in state.bind_groups.values() {
        buffers.extend(bind_group_buffer_resources(&bound.group));
    }
    for bound in state.vertex_buffers.values() {
        buffers.push(Arc::clone(&bound.buffer));
    }
    if let Some(bound) = &state.index_buffer {
        buffers.push(Arc::clone(&bound.buffer));
    }
    buffers.extend(state.command_referenced_buffers.iter().cloned());
    for draw in draws {
        for bound in draw.bind_groups.values() {
            buffers.extend(bind_group_buffer_resources(&bound.group));
        }
        for bound in draw.vertex_buffers.values() {
            buffers.push(Arc::clone(&bound.buffer));
        }
        if let Some(bound) = &draw.index_buffer {
            buffers.push(Arc::clone(&bound.buffer));
        }
        if let Some(bound) = &draw.indirect_buffer {
            buffers.push(Arc::clone(&bound.buffer));
        }
    }
    buffers
}

fn render_bundle_referenced_textures(
    state: &PassEncoderState,
    draws: &[RenderBundleDraw],
) -> Vec<Texture> {
    let mut textures = Vec::new();
    for bound in state.bind_groups.values() {
        textures.extend(bind_group_texture_resources(&bound.group));
    }
    for draw in draws {
        for bound in draw.bind_groups.values() {
            textures.extend(bind_group_texture_resources(&bound.group));
        }
    }
    textures
}

impl RenderBundleEncoderDescriptor {
    /// Returns the attachment signature used for render pass compatibility checks.
    pub(crate) fn attachment_signature(&self) -> AttachmentSignature {
        AttachmentSignature {
            color_formats: self.color_formats.clone(),
            depth_stencil_format: self.depth_stencil_format,
            sample_count: self.sample_count,
            depth_read_only: self.depth_read_only,
            stencil_read_only: self.stencil_read_only,
        }
    }
}

/// Validates render bundle encoder descriptor and returns a descriptive error on failure.
pub(crate) fn validate_render_bundle_encoder_descriptor(
    descriptor: &RenderBundleEncoderDescriptor,
    limits: Limits,
    features: &FeatureSet,
) -> Result<(), String> {
    if descriptor.color_formats.len() > descriptor.max_color_attachments as usize {
        return Err("render bundle colorFormatCount exceeds the device limit".to_owned());
    }
    if descriptor.sample_count != 1 && descriptor.sample_count != 4 {
        return Err("render bundle sampleCount must be 1 or 4".to_owned());
    }

    let mut has_attachment = descriptor.depth_stencil_format.is_some();
    let mut color_byte_formats = Vec::new();
    for color_format in descriptor.color_formats.iter().flatten().copied() {
        has_attachment = true;
        let Some(caps) = color_format.caps(features) else {
            return Err("render bundle color format must be defined".to_owned());
        };
        if !caps.aspects.color || !caps.renderable {
            return Err("render bundle color format must be color-renderable".to_owned());
        }
        color_byte_formats.push(color_format);
    }
    let color_bytes = color_attachment_bytes_per_sample(color_byte_formats)
        .ok_or_else(|| "render bundle color format byte count overflows".to_owned())?;
    if color_bytes > limits.max_color_attachment_bytes_per_sample {
        return Err(
            "render bundle color attachment bytes per sample exceed the device limit".to_owned(),
        );
    }
    if let Some(depth_format) = descriptor.depth_stencil_format {
        let Some(caps) = depth_format.caps(features) else {
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

/// Validates render bundle pipeline and returns a descriptive error on failure.
pub(crate) fn validate_render_bundle_pipeline(
    descriptor: &RenderBundleEncoderDescriptor,
    pipeline: &RenderPipeline,
) -> Result<(), String> {
    if pipeline.is_error() {
        return Err("render bundle requires a valid render pipeline".to_owned());
    }
    let descriptor_signature = descriptor.attachment_signature();
    let pipeline_signature = pipeline.attachment_signature();
    if descriptor_signature.color_formats != pipeline_signature.color_formats
        || descriptor_signature.depth_stencil_format != pipeline_signature.depth_stencil_format
        || descriptor_signature.sample_count != pipeline_signature.sample_count
    {
        return Err("render bundle pipeline attachment signature is incompatible".to_owned());
    }
    if descriptor.depth_read_only && pipeline.writes_depth() {
        return Err(
            "render bundle read-only depth attachment is incompatible with depth writes".to_owned(),
        );
    }
    if descriptor.stencil_read_only && pipeline.writes_stencil() {
        return Err(
            "render bundle read-only stencil attachment is incompatible with stencil writes"
                .to_owned(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

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
        let (bundle_encoder, error) = RenderBundleEncoder::new(
            render_bundle_encoder_descriptor(),
            device.limits(),
            device.features(),
        );
        assert_eq!(error, None);

        assert_eq!(bundle_encoder.insert_debug_marker(), None);
        assert_eq!(bundle_encoder.push_debug_group(), None);
        assert_eq!(bundle_encoder.pop_debug_group(), None);
        assert_eq!(bundle_encoder.set_pipeline(pipeline), None);
        assert_eq!(
            bundle_encoder.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
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
    fn render_bundle_encoder_rejects_attachment_byte_limit_overflow() {
        let device = noop_device();
        let mut descriptor = render_bundle_encoder_descriptor();
        descriptor.color_formats = vec![Some(TextureFormat::from_raw(TextureFormat::RGBA32_FLOAT))];
        let mut limits = device.limits();
        limits.max_color_attachment_bytes_per_sample = 15;

        let (_encoder, error) = RenderBundleEncoder::new(descriptor, limits, device.features());
        assert_eq!(
            error,
            Some(
                "render bundle color attachment bytes per sample exceed the device limit"
                    .to_owned()
            )
        );
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
        let (bundle_encoder, error) = RenderBundleEncoder::new(
            render_bundle_encoder_descriptor(),
            device.limits(),
            device.features(),
        );
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

    #[test]
    fn render_bundle_encoder_records_validation_error_for_finish() {
        let device = noop_device();
        let (bundle_encoder, error) = RenderBundleEncoder::new(
            render_bundle_encoder_descriptor(),
            device.limits(),
            device.features(),
        );
        assert_eq!(error, None);

        assert_eq!(
            bundle_encoder.record_validation_error("bundle device mismatch"),
            None
        );
        let (bundle, error) = bundle_encoder.finish();

        assert_eq!(error.as_deref(), Some("bundle device mismatch"));
        assert!(bundle.is_error());
    }

    #[test]
    fn render_bundle_encoder_rejects_invalid_bind_group_state() {
        let device = noop_device();
        let bind_group = empty_bind_group(&device);
        let (bundle_encoder, error) = RenderBundleEncoder::new(
            render_bundle_encoder_descriptor(),
            device.limits(),
            device.features(),
        );
        assert_eq!(error, None);

        assert_eq!(
            bundle_encoder.set_bind_group(
                device.limits().max_bind_groups,
                Some(bind_group),
                Vec::new(),
                device.limits()
            ),
            None
        );
        let (bundle, error) = bundle_encoder.finish();

        assert!(bundle.is_error());
        assert_eq!(
            error,
            Some("bind group index exceeds the device limit".to_owned())
        );

        let (bundle_encoder, error) = RenderBundleEncoder::new(
            render_bundle_encoder_descriptor(),
            device.limits(),
            device.features(),
        );
        assert_eq!(error, None);

        assert_eq!(
            bundle_encoder.set_bind_group(0, None, vec![0], device.limits()),
            None
        );
        let (bundle, error) = bundle_encoder.finish();

        assert!(bundle.is_error());
        assert_eq!(
            error,
            Some("clearing a bind group must not include dynamic offsets".to_owned())
        );
    }
}
