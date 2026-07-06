use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::bind_group::*;
use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::compute_pipeline::*;
use crate::copy::{LoadOp, StoreOp};
use crate::extent::*;
use crate::limits::*;
use crate::query_set::*;
use crate::render_pipeline::*;
use crate::texture::*;
use crate::texture_view::{TextureAspect, TextureView};

/// Holds shared state for the pass encoder handle.
#[derive(Debug)]
pub(crate) struct PassEncoderInner {
    pub(crate) parent: CommandEncoder,
    pub(crate) token: PassToken,
    pub(crate) state: Mutex<PassEncoderState>,
}

/// Tracks the lifecycle state for pass encoder.
#[derive(Debug)]
pub(crate) struct PassEncoderState {
    pub(crate) ended: bool,
    pub(crate) debug_group_depth: u32,
    pub(crate) render_pipeline: Option<Arc<RenderPipeline>>,
    pub(crate) compute_pipeline: Option<Arc<ComputePipeline>>,
    pub(crate) limits: Limits,
    pub(crate) bind_groups: BTreeMap<u32, BoundBindGroup>,
    pub(crate) vertex_buffers: BTreeMap<u32, BoundVertexBuffer>,
    pub(crate) index_buffer: Option<BoundIndexBuffer>,
    pub(crate) viewport: Option<Viewport>,
    pub(crate) scissor_rect: Option<ScissorRect>,
    pub(crate) attachment_signature: Option<AttachmentSignature>,
    pub(crate) render_extent: Option<Extent3d>,
    pub(crate) attachment_textures: Vec<Texture>,
    pub(crate) attachment_texture_uses: Vec<TextureScopeUse>,
    pub(crate) render_color_attachments: Vec<Option<RenderPassColorExecution>>,
    pub(crate) render_depth_stencil_attachment: Option<RenderPassDepthStencilExecution>,
    pub(crate) render_pass_recorded: bool,
    pub(crate) blend_constant: [f32; 4],
    pub(crate) stencil_reference: u32,
    pub(crate) occlusion_query_set: Option<QuerySet>,
    pub(crate) open_occlusion_query: Option<u32>,
    pub(crate) used_occlusion_queries: BTreeSet<u32>,
    pub(crate) command_referenced_buffers: Vec<Arc<Buffer>>,
    pub(crate) scope_buffer_uses: Vec<BufferScopeUse>,
    pub(crate) scope_texture_uses: Vec<TextureScopeUse>,
    pub(crate) draw_count: u64,
    pub(crate) max_draw_count: u64,
    pub(crate) immediate_data: Vec<u8>,
    pub(crate) immediate_data_written: ImmediateWrittenMask,
}

/// The absolute ceiling of the per-pass user-immediates scratch, in bytes:
/// Dawn's `kMaxImmediateDataBytes` (`dawn/common/Constants.h:58`). Every
/// backend's `Limits::max_immediate_size` is expected to stay `<=` this
/// value (Noop is the first backend to reach it, Block 94 slice S1); the
/// scratch buffer is always allocated at this fixed size so `SetImmediates`
/// never needs to grow it mid-pass.
pub(crate) const MAX_IMMEDIATE_DATA_BYTES: usize = 64;

/// Per-4-byte-word "written by an explicit `SetImmediates`" bitmask over the
/// user-immediates scratch: bit `i` covers bytes `[4 * i, 4 * i + 4)`.
/// Word granularity is exact because `validate_set_immediates` enforces
/// 4-byte alignment on both `offset` and `size` (Dawn's own `ImmediateMask`
/// is the same idea at `kImmediateElementByteSize` granularity,
/// `dawn/native/ImmediatesLayout.h:61-67`). Render bundles use this to
/// replay Dawn's shared-tracker semantics: a bundle overlays only the byte
/// ranges it explicitly wrote onto the outer pass's scratch, inheriting the
/// rest (see `RenderPassEncoder::execute_bundles`).
pub(crate) type ImmediateWrittenMask = u16;

// The mask must have one bit per 4-byte word of the scratch.
const _: () = assert!(MAX_IMMEDIATE_DATA_BYTES / 4 <= ImmediateWrittenMask::BITS as usize);

/// Groups pass encoder creation metadata.
#[derive(Debug)]
pub(crate) struct PassEncoderInit {
    pub(crate) attachment_signature: Option<AttachmentSignature>,
    pub(crate) render_extent: Option<Extent3d>,
    pub(crate) attachment_textures: Vec<Texture>,
    pub(crate) render_color_attachments: Vec<Option<RenderPassColorExecution>>,
    pub(crate) render_depth_stencil_attachment: Option<RenderPassDepthStencilExecution>,
    pub(crate) occlusion_query_set: Option<QuerySet>,
    pub(crate) max_draw_count: u64,
}

impl PassEncoderState {
    /// Creates a new instance.
    pub(crate) fn new(limits: Limits, init: PassEncoderInit) -> Self {
        Self {
            ended: false,
            debug_group_depth: 0,
            render_pipeline: None,
            compute_pipeline: None,
            limits,
            bind_groups: BTreeMap::new(),
            vertex_buffers: BTreeMap::new(),
            index_buffer: None,
            viewport: None,
            scissor_rect: None,
            attachment_signature: init.attachment_signature,
            render_extent: init.render_extent,
            attachment_textures: init.attachment_textures,
            attachment_texture_uses: Vec::new(),
            render_color_attachments: init.render_color_attachments,
            render_depth_stencil_attachment: init.render_depth_stencil_attachment,
            render_pass_recorded: false,
            blend_constant: [0.0; 4],
            stencil_reference: 0,
            occlusion_query_set: init.occlusion_query_set,
            open_occlusion_query: None,
            used_occlusion_queries: BTreeSet::new(),
            command_referenced_buffers: Vec::new(),
            scope_buffer_uses: Vec::new(),
            scope_texture_uses: Vec::new(),
            draw_count: 0,
            max_draw_count: init.max_draw_count,
            // Dawn: `ImmediateDataContent::mData` is zero-initialized
            // (`dawn/native/ImmediatesTracker.h:70`) -- the per-pass
            // user-immediates scratch starts at zero and is never reset for
            // the lifetime of the pass; only `SetImmediates` overwrites it,
            // byte-range by byte-range.
            immediate_data: vec![0u8; MAX_IMMEDIATE_DATA_BYTES],
            immediate_data_written: 0,
        }
    }

    /// Resets the tracked render-pipeline / vertex / index state for this pass.
    pub(crate) fn clear_render_state(&mut self) {
        self.render_pipeline = None;
        self.bind_groups.clear();
        self.vertex_buffers.clear();
        self.index_buffer = None;
    }

    /// Sets attachment texture usages for render-pass scope validation.
    pub(crate) fn set_attachment_texture_uses(
        &mut self,
        uses: Vec<TextureScopeUse>,
    ) -> Result<(), String> {
        let mut scoped_texture_uses = self.scope_texture_uses.clone();
        scoped_texture_uses.extend(uses.iter().cloned());
        validate_texture_usage_scope_lenient(&scoped_texture_uses)?;
        self.attachment_texture_uses = uses.clone();
        self.scope_texture_uses.extend(uses);
        Ok(())
    }

    pub(crate) fn load_attachments_for_draw(
        &mut self,
    ) -> (
        Vec<Option<RenderPassColorExecution>>,
        Option<RenderPassDepthStencilExecution>,
    ) {
        let mut color_attachments = self.render_color_attachments.clone();
        let mut depth_stencil_attachment = self.render_depth_stencil_attachment.clone();
        for attachment in color_attachments.iter_mut().flatten() {
            attachment.store_op = StoreOp::Store;
        }
        if let Some(attachment) = &mut depth_stencil_attachment {
            attachment.depth_store_op = StoreOp::Store;
            attachment.stencil_store_op = StoreOp::Store;
        }
        if self.render_pass_recorded {
            for attachment in color_attachments.iter_mut().flatten() {
                attachment.load_op = LoadOp::Load;
            }
            if let Some(attachment) = &mut depth_stencil_attachment {
                attachment.depth_load_op = LoadOp::Load;
                attachment.stencil_load_op = LoadOp::Load;
            }
        }
        self.render_pass_recorded = true;
        (color_attachments, depth_stencil_attachment)
    }
}

/// Stores bound bind group data used by validation and backend submission.
#[derive(Debug, Clone)]
pub(crate) struct BoundBindGroup {
    pub(crate) group: Arc<BindGroup>,
    pub(crate) dynamic_offsets: Vec<u32>,
}

/// Stores bound vertex buffer data used by validation and backend submission.
#[derive(Debug, Clone)]
pub(crate) struct BoundVertexBuffer {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

/// Stores bound index buffer data used by validation and backend submission.
#[derive(Debug, Clone)]
pub(crate) struct BoundIndexBuffer {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) format: IndexFormat,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

impl PassEncoderInner {
    /// Creates a new instance.
    pub(crate) fn new(parent: CommandEncoder, token: PassToken, init: PassEncoderInit) -> Self {
        let limits = parent.inner.limits;
        Self {
            parent,
            token,
            state: Mutex::new(PassEncoderState::new(limits, init)),
        }
    }

    /// Ends recording for this pass or encoder.
    pub(crate) fn end(&self) -> Option<String> {
        let mut state = self.state.lock();
        if state.ended {
            let message = "pass encoder cannot be ended more than once".to_owned();
            return Some(message);
        }
        if self.parent.is_finished() {
            let message = "pass encoder cannot be used after parent encoder finish".to_owned();
            return Some(message);
        }
        if !self.parent.is_open_pass(self.token) {
            let message = "pass encoder is not the active pass".to_owned();
            return Some(message);
        }
        state.ended = true;
        let unbalanced_debug_groups = state.debug_group_depth != 0;
        let open_occlusion_query = state.open_occlusion_query.is_some();
        let draw_count_exceeded = state.draw_count > state.max_draw_count;
        let restore_store_ops = if state.render_pass_recorded && state.draw_count > 0 {
            Some((
                state
                    .render_color_attachments
                    .iter()
                    .map(|attachment| attachment.as_ref().map(|attachment| attachment.store_op))
                    .collect::<Vec<_>>(),
                state
                    .render_depth_stencil_attachment
                    .as_ref()
                    .map(|attachment| attachment.depth_store_op),
                state
                    .render_depth_stencil_attachment
                    .as_ref()
                    .map(|attachment| attachment.stencil_store_op),
            ))
        } else {
            None
        };
        let render_pass_command = if !state.render_pass_recorded
            && (!state.render_color_attachments.is_empty()
                || state.render_depth_stencil_attachment.is_some())
        {
            state.render_pass_recorded = true;
            Some(RenderPassCommand {
                pipeline: state.render_pipeline.clone(),
                color_attachments: state.render_color_attachments.clone(),
                depth_stencil_attachment: state.render_depth_stencil_attachment.clone(),
                attachment_textures: state.attachment_textures.clone(),
                bind_groups: state.bind_groups.clone(),
                vertex_buffers: state.vertex_buffers.clone(),
                index_buffer: state.index_buffer.clone(),
                indirect_buffer: None,
                viewport: state.viewport,
                scissor_rect: state.scissor_rect,
                blend_constant: state.blend_constant,
                stencil_reference: state.stencil_reference,
                occlusion_query_set: None,
                occlusion_query_index: None,
                draw: None,
                immediate_data: state.immediate_data.clone(),
            })
        } else {
            None
        };
        drop(state);

        if let Some(command) = render_pass_command {
            self.parent.record_render_pass(command);
        }
        if let Some((color_store_ops, depth_store_op, stencil_store_op)) = restore_store_ops {
            self.parent.patch_last_render_pass_store_ops(
                &color_store_ops,
                depth_store_op,
                stencil_store_op,
            );
        }
        self.parent.end_pass(self.token);
        if unbalanced_debug_groups {
            let message = "pass encoder debug group stack is unbalanced".to_owned();
            self.parent.record_first_error(message);
            None
        } else if open_occlusion_query {
            self.parent
                .record_first_error("render pass occlusion query is still open");
            None
        } else if draw_count_exceeded {
            self.parent
                .record_first_error("render pass draw count exceeds maxDrawCount");
            None
        } else {
            None
        }
    }

    /// Records a debug marker against the shared pass state.
    pub(crate) fn insert_debug_marker(&self) -> Option<String> {
        self.record_pass_command(|_| Ok(()))
    }

    /// Opens a debug group in the shared pass state.
    pub(crate) fn push_debug_group(&self) -> Option<String> {
        self.record_pass_command(|state| {
            state.debug_group_depth = state.debug_group_depth.saturating_add(1);
            Ok(())
        })
    }

    /// Closes the most recently opened debug group in the shared pass state.
    pub(crate) fn pop_debug_group(&self) -> Option<String> {
        self.record_pass_command(|state| {
            if state.debug_group_depth == 0 {
                Err("pass encoder debug group stack is empty".to_owned())
            } else {
                state.debug_group_depth -= 1;
                Ok(())
            }
        })
    }

    /// Returns an immediate validation error for commands that cannot be deferred.
    fn pass_command_immediate_error(&self) -> Option<String> {
        if self.parent.is_finished() {
            Some("pass encoder cannot be used after parent encoder finish".to_owned())
        } else {
            None
        }
    }

    /// Records a pass command after handling pass and parent encoder lifecycle.
    pub(crate) fn record_pass_command<F>(&self, command: F) -> Option<String>
    where
        F: FnOnce(&mut PassEncoderState) -> Result<(), String>,
    {
        if let Some(message) = self.pass_command_immediate_error() {
            return Some(message);
        }
        let ended = { self.state.lock().ended };
        if ended {
            self.parent
                .record_first_error("pass encoder cannot be used after end");
            return None;
        }
        let mut state = self.state.lock();
        if let Err(message) = command(&mut state) {
            self.parent.record_first_error(message);
        }
        None
    }
}

/// Enumerates render draw kind values.
#[derive(Debug, Clone, Copy)]
pub(crate) enum RenderDrawKind {
    /// Direct variant.
    Direct {
        /// Vertex count variant.
        vertex_count: u32,
        /// Instance count variant.
        instance_count: u32,
        /// First vertex variant.
        first_vertex: u32,
        /// First instance variant.
        first_instance: u32,
    },
    /// Indexed direct variant.
    IndexedDirect {
        /// Index count variant.
        index_count: u32,
        /// Instance count variant.
        instance_count: u32,
        /// First index variant.
        first_index: u32,
        /// First instance variant.
        first_instance: u32,
    },
    /// Indirect variant.
    Indirect,
    /// Indexed indirect variant.
    IndexedIndirect,
}

/// Validates render draw state and returns a descriptive error on failure.
pub(crate) fn validate_render_draw_state(
    state: &PassEncoderState,
    kind: RenderDrawKind,
    limits: Limits,
) -> Result<(), String> {
    let pipeline = validate_render_draw_base_state(state, limits, kind.is_indexed())?;
    validate_strip_index_format(pipeline, state, kind.is_indexed())?;
    match kind {
        RenderDrawKind::Direct {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        } => validate_vertex_buffer_oob(
            pipeline,
            state,
            Some((first_vertex, vertex_count)),
            first_instance,
            instance_count,
        ),
        RenderDrawKind::IndexedDirect {
            index_count,
            instance_count,
            first_index,
            first_instance,
        } => {
            validate_index_buffer_oob(state, first_index, index_count)?;
            validate_vertex_buffer_oob(pipeline, state, None, first_instance, instance_count)
        }
        RenderDrawKind::Indirect | RenderDrawKind::IndexedIndirect => Ok(()),
    }
}

impl RenderDrawKind {
    /// Returns true when this object is indexed.
    pub(crate) fn is_indexed(self) -> bool {
        matches!(
            self,
            RenderDrawKind::IndexedDirect { .. } | RenderDrawKind::IndexedIndirect
        )
    }
}

/// Validates render draw base state and returns a descriptive error on failure.
pub(crate) fn validate_render_draw_base_state(
    state: &PassEncoderState,
    limits: Limits,
    indexed: bool,
) -> Result<&Arc<RenderPipeline>, String> {
    let Some(pipeline) = &state.render_pipeline else {
        return Err("render pass draw requires a render pipeline".to_owned());
    };
    if pipeline.is_error() {
        return Err("render pass draw requires a valid render pipeline".to_owned());
    }
    validate_draw_time_bind_groups_plus_vertex_buffers(state, limits)?;
    validate_pipeline_bind_groups(pipeline.bind_group_layouts(), &state.bind_groups, limits)?;
    debug_assert_eq!(
        pipeline.vertex_buffer_layouts().len(),
        pipeline.required_vertex_buffer_count()
    );
    for (slot, layout) in pipeline.vertex_buffer_layouts().iter().enumerate() {
        if !layout.used {
            continue;
        }
        let slot = u32::try_from(slot)
            .map_err(|_| "render pipeline vertex buffer slot is too large".to_owned())?;
        if !state.vertex_buffers.contains_key(&slot) {
            return Err(
                "render pass draw requires all declared vertex buffers to be set".to_owned(),
            );
        }
    }
    if indexed && state.index_buffer.is_none() {
        return Err("render pass indexed draw requires an index buffer".to_owned());
    }
    validate_required_immediate_data(
        pipeline.immediate_required_mask(),
        state.immediate_data_written,
    )?;
    Ok(pipeline)
}

fn validate_draw_time_bind_groups_plus_vertex_buffers(
    state: &PassEncoderState,
    limits: Limits,
) -> Result<(), String> {
    let bind_group_count = state
        .bind_groups
        .keys()
        .next_back()
        .copied()
        .map_or(0, |index| index + 1);
    let vertex_buffer_count = state
        .vertex_buffers
        .keys()
        .next_back()
        .copied()
        .map_or(0, |slot| slot + 1);
    let total = bind_group_count
        .checked_add(vertex_buffer_count)
        .ok_or_else(|| {
            "render pass draw bind group plus vertex buffer count overflows".to_owned()
        })?;
    if total > limits.max_bind_groups_plus_vertex_buffers {
        return Err(
            "render pass draw bind group plus vertex buffer count exceeds the device limit"
                .to_owned(),
        );
    }
    Ok(())
}

/// Validates set index buffer and returns a descriptive error on failure.
pub(crate) fn validate_set_index_buffer(
    buffer: &Buffer,
    format: IndexFormat,
    offset: u64,
    size: u64,
) -> Result<u64, String> {
    if buffer.is_error() {
        return Err("render pass index buffer must not be an error buffer".to_owned());
    }
    if !buffer.usage().contains(BufferUsage::INDEX) {
        return Err("render pass index buffer requires Index usage".to_owned());
    }
    let format_size = index_format_size(format);
    if !offset.is_multiple_of(format_size) {
        return Err("render pass index buffer offset is not aligned".to_owned());
    }
    resolve_buffer_binding_size(
        offset,
        size,
        buffer.size(),
        "render pass index buffer range",
    )
}

/// Validates set vertex buffer and returns a descriptive error on failure.
pub(crate) fn validate_set_vertex_buffer(
    buffer: &Buffer,
    offset: u64,
    size: u64,
) -> Result<u64, String> {
    if buffer.is_error() {
        return Err("render pass vertex buffer must not be an error buffer".to_owned());
    }
    if !buffer.usage().contains(BufferUsage::VERTEX) {
        return Err("render pass vertex buffer requires Vertex usage".to_owned());
    }
    if !offset.is_multiple_of(4) {
        return Err("render pass vertex buffer offset must be 4-byte aligned".to_owned());
    }
    resolve_buffer_binding_size(
        offset,
        size,
        buffer.size(),
        "render pass vertex buffer range",
    )
}

/// Validates vertex buffer slot and returns a descriptive error on failure.
pub(crate) fn validate_vertex_buffer_slot(slot: u32, limits: Limits) -> Result<(), String> {
    if slot >= limits.max_vertex_buffers {
        return Err("render pass vertex buffer slot exceeds the device limit".to_owned());
    }
    Ok(())
}

/// Validates clear vertex buffer and returns a descriptive error on failure.
pub(crate) fn validate_clear_vertex_buffer(offset: u64, size: u64) -> Result<(), String> {
    if offset != 0 || size != 0 {
        return Err("render pass null vertex buffer requires zero offset and size".to_owned());
    }
    Ok(())
}

/// Records resolve into the command stream.
pub(crate) fn resolve_buffer_binding_size(
    offset: u64,
    size: u64,
    buffer_size: u64,
    label: &str,
) -> Result<u64, String> {
    if offset > buffer_size {
        return Err(format!("{label} exceeds buffer size"));
    }
    let resolved_size = if size == u64::MAX {
        buffer_size - offset
    } else {
        size
    };
    validate_buffer_range(offset, resolved_size, buffer_size, label)?;
    Ok(resolved_size)
}

/// Validates strip index format and returns a descriptive error on failure.
pub(crate) fn validate_strip_index_format(
    pipeline: &RenderPipeline,
    state: &PassEncoderState,
    indexed: bool,
) -> Result<(), String> {
    if !indexed {
        return Ok(());
    }
    let primitive = pipeline.primitive_state();
    if !matches!(
        primitive.topology,
        PrimitiveTopology::LineStrip | PrimitiveTopology::TriangleStrip
    ) {
        return Ok(());
    }
    let Some(strip_format) = primitive.strip_index_format else {
        return Err("render pass strip indexed draw requires pipeline stripIndexFormat".to_owned());
    };
    let index_buffer = state
        .index_buffer
        .as_ref()
        .ok_or_else(|| "render pass indexed draw requires an index buffer".to_owned())?;
    if index_buffer.format != strip_format {
        return Err(
            "render pass index buffer format must match pipeline stripIndexFormat".to_owned(),
        );
    }
    Ok(())
}

/// Validates vertex buffer oob and returns a descriptive error on failure.
pub(crate) fn validate_vertex_buffer_oob(
    pipeline: &RenderPipeline,
    state: &PassEncoderState,
    vertex_draw: Option<(u32, u32)>,
    first_instance: u32,
    instance_count: u32,
) -> Result<(), String> {
    for (slot, layout) in pipeline.vertex_buffer_layouts().iter().enumerate() {
        if !layout.used {
            continue;
        }
        let stride_count = match layout.step_mode {
            VertexStepMode::Vertex => {
                let Some((first_vertex, vertex_count)) = vertex_draw else {
                    continue;
                };
                first_vertex
                    .checked_add(vertex_count)
                    .ok_or_else(|| "render pass draw vertex count overflows".to_owned())?
            }
            VertexStepMode::Instance => first_instance
                .checked_add(instance_count)
                .ok_or_else(|| "render pass draw instance count overflows".to_owned())?,
        };
        if stride_count == 0 {
            continue;
        }
        let last_stride = layout
            .attributes
            .iter()
            .map(|attribute| {
                attribute
                    .offset
                    .checked_add(attribute.format.info().byte_size)
                    .ok_or_else(|| "render pass vertex buffer required size overflows".to_owned())
            })
            .max()
            .transpose()?
            .unwrap_or(0);
        let required_size = u64::from(stride_count - 1)
            .checked_mul(layout.array_stride)
            .and_then(|size| size.checked_add(last_stride))
            .ok_or_else(|| "render pass vertex buffer required size overflows".to_owned())?;
        let slot = u32::try_from(slot)
            .map_err(|_| "render pipeline vertex buffer slot is too large".to_owned())?;
        let bound = state.vertex_buffers.get(&slot).ok_or_else(|| {
            "render pass draw requires all declared vertex buffers to be set".to_owned()
        })?;
        let required_end = bound
            .offset
            .checked_add(required_size)
            .ok_or_else(|| "render pass vertex buffer required range overflows".to_owned())?;
        let bound_end = bound
            .offset
            .checked_add(bound.size)
            .ok_or_else(|| "render pass vertex buffer bound range overflows".to_owned())?;
        if required_end > bound_end || required_end > bound.buffer.size() {
            return Err("render pass draw vertex buffer range exceeds the bound buffer".to_owned());
        }
    }
    Ok(())
}

/// Validates index buffer oob and returns a descriptive error on failure.
pub(crate) fn validate_index_buffer_oob(
    state: &PassEncoderState,
    first_index: u32,
    index_count: u32,
) -> Result<(), String> {
    let index_buffer = state
        .index_buffer
        .as_ref()
        .ok_or_else(|| "render pass indexed draw requires an index buffer".to_owned())?;
    let required_indices = first_index
        .checked_add(index_count)
        .ok_or_else(|| "render pass indexed draw index count overflows".to_owned())?;
    let required_size = u64::from(required_indices)
        .checked_mul(index_format_size(index_buffer.format))
        .ok_or_else(|| "render pass indexed draw index buffer size overflows".to_owned())?;
    let required_end = index_buffer
        .offset
        .checked_add(required_size)
        .ok_or_else(|| "render pass indexed draw index buffer range overflows".to_owned())?;
    let bound_end = index_buffer
        .offset
        .checked_add(index_buffer.size)
        .ok_or_else(|| "render pass indexed draw index buffer bound range overflows".to_owned())?;
    if required_end > bound_end || required_end > index_buffer.buffer.size() {
        return Err(
            "render pass indexed draw index buffer range exceeds the bound buffer".to_owned(),
        );
    }
    Ok(())
}

/// Validates indirect buffer and returns a descriptive error on failure.
pub(crate) fn validate_indirect_buffer(
    buffer: &Buffer,
    indirect_offset: u64,
    args_size: u64,
    label: &str,
) -> Result<(), String> {
    if buffer.is_error() {
        return Err(format!("{label} buffer must not be an error buffer"));
    }
    if !buffer.usage().contains(BufferUsage::INDIRECT) {
        return Err(format!("{label} buffer requires Indirect usage"));
    }
    if !indirect_offset.is_multiple_of(4) {
        return Err(format!("{label} offset must be 4-byte aligned"));
    }
    validate_buffer_range(indirect_offset, args_size, buffer.size(), label)
}

const fn index_format_size(format: IndexFormat) -> u64 {
    match format {
        IndexFormat::Uint16 => 2,
        IndexFormat::Uint32 => 4,
    }
}

/// Enumerates resource access values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResourceAccess {
    /// Read variant.
    Read,
    /// Write variant.
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextureAccess {
    Read,
    WriteOnlyStorage,
    ReadWriteStorage,
    AttachmentWrite,
}

impl TextureAccess {
    fn compatible_in_render_scope(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Read, Self::Read)
                | (Self::WriteOnlyStorage, Self::WriteOnlyStorage)
                | (Self::ReadWriteStorage, Self::ReadWriteStorage)
        )
    }

    fn compatible_in_compute_scope(self, other: Self) -> bool {
        matches!((self, other), (Self::Read, Self::Read))
    }
}

/// Stores buffer scope use data used by validation and backend submission.
#[derive(Debug, Clone)]
pub(crate) struct BufferScopeUse {
    pub(crate) buffer: Arc<Buffer>,
    #[allow(dead_code)]
    pub(crate) offset: u64,
    #[allow(dead_code)]
    pub(crate) size: u64,
    pub(crate) access: ResourceAccess,
}

/// Stores texture scope use data used by validation and backend submission.
#[derive(Debug, Clone)]
pub(crate) struct TextureScopeUse {
    pub(crate) texture: Texture,
    pub(crate) base_mip_level: u32,
    pub(crate) mip_level_count: u32,
    pub(crate) base_array_layer: u32,
    pub(crate) array_layer_count: u32,
    pub(crate) depth_slice: Option<u32>,
    pub(crate) aspects: TextureAspectMask,
    pub(crate) access: TextureAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextureAspectMask(u8);

impl TextureAspectMask {
    pub(crate) const COLOR: Self = Self(1);
    pub(crate) const DEPTH: Self = Self(2);
    pub(crate) const STENCIL: Self = Self(4);

    fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Validates usage scope and returns a descriptive error on failure.
pub(crate) fn validate_usage_scope(
    required_layouts: &[Arc<BindGroupLayout>],
    bound_groups: &BTreeMap<u32, BoundBindGroup>,
    attachment_uses: Option<&[TextureScopeUse]>,
) -> Result<(), String> {
    let mut buffer_uses = Vec::new();
    let mut texture_uses = Vec::new();

    for (index, layout) in required_layouts.iter().enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| "pipeline bind group index is too large".to_owned())?;
        let Some(bound) = bound_groups.get(&index) else {
            continue;
        };
        collect_bind_group_usage(layout, bound, &mut buffer_uses, &mut texture_uses)?;
    }

    if let Some(attachment_uses) = attachment_uses {
        texture_uses.extend_from_slice(attachment_uses);
    }
    validate_resource_usage_scope(&buffer_uses, &texture_uses)
}

pub(crate) fn collect_pipeline_usage_scope(
    required_layouts: &[Arc<BindGroupLayout>],
    bound_groups: &BTreeMap<u32, BoundBindGroup>,
) -> Result<(Vec<BufferScopeUse>, Vec<TextureScopeUse>), String> {
    let mut buffer_uses = Vec::new();
    let mut texture_uses = Vec::new();

    for (index, layout) in required_layouts.iter().enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| "pipeline bind group index is too large".to_owned())?;
        let Some(bound) = bound_groups.get(&index) else {
            continue;
        };
        collect_bind_group_usage(layout, bound, &mut buffer_uses, &mut texture_uses)?;
    }

    Ok((buffer_uses, texture_uses))
}

pub(crate) fn record_pipeline_usage_scope(
    state: &mut PassEncoderState,
    required_layouts: &[Arc<BindGroupLayout>],
    _attachment_uses: &[TextureScopeUse],
) -> Result<(), String> {
    let _ = collect_pipeline_usage_scope(required_layouts, &state.bind_groups)?;
    Ok(())
}

pub(crate) fn record_bind_group_usage_scope(
    state: &mut PassEncoderState,
    bound: &BoundBindGroup,
) -> Result<(), String> {
    let mut buffer_uses = Vec::new();
    let mut texture_uses = Vec::new();
    collect_bind_group_usage(
        bound.group.layout(),
        bound,
        &mut buffer_uses,
        &mut texture_uses,
    )?;
    record_resource_usage_scope_uses(state, buffer_uses, texture_uses)
}

pub(crate) fn record_buffer_usage_scope_use(
    state: &mut PassEncoderState,
    buffer_use: BufferScopeUse,
) -> Result<(), String> {
    record_buffer_usage_scope_uses(state, vec![buffer_use])
}

fn record_buffer_usage_scope_uses(
    state: &mut PassEncoderState,
    buffer_uses: Vec<BufferScopeUse>,
) -> Result<(), String> {
    record_resource_usage_scope_uses(state, buffer_uses, Vec::new())
}

fn record_resource_usage_scope_uses(
    state: &mut PassEncoderState,
    buffer_uses: Vec<BufferScopeUse>,
    texture_uses: Vec<TextureScopeUse>,
) -> Result<(), String> {
    let mut scoped_buffer_uses = state.scope_buffer_uses.clone();
    scoped_buffer_uses.extend(buffer_uses.iter().cloned());
    validate_buffer_usage_scope_lenient(&scoped_buffer_uses)?;
    let mut scoped_texture_uses = state.scope_texture_uses.clone();
    scoped_texture_uses.extend(texture_uses.iter().cloned());
    validate_texture_usage_scope_lenient(&scoped_texture_uses)?;
    state.scope_buffer_uses.extend(buffer_uses);
    state.scope_texture_uses.extend(texture_uses);
    Ok(())
}

pub(crate) fn validate_resource_usage_scope(
    buffer_uses: &[BufferScopeUse],
    texture_uses: &[TextureScopeUse],
) -> Result<(), String> {
    validate_buffer_usage_scope(buffer_uses)?;
    validate_texture_usage_scope(texture_uses)
}

#[allow(dead_code)]
pub(crate) fn validate_resource_usage_scope_lenient(
    buffer_uses: &[BufferScopeUse],
    texture_uses: &[TextureScopeUse],
) -> Result<(), String> {
    validate_buffer_usage_scope_lenient(buffer_uses)?;
    validate_texture_usage_scope_lenient(texture_uses)
}

/// Returns collect bind group usage.
pub(crate) fn collect_bind_group_usage(
    layout: &BindGroupLayout,
    bound: &BoundBindGroup,
    buffer_uses: &mut Vec<BufferScopeUse>,
    texture_uses: &mut Vec<TextureScopeUse>,
) -> Result<(), String> {
    let layout_entries = layout
        .entries()
        .iter()
        .map(|entry| (entry.binding, entry))
        .collect::<BTreeMap<_, _>>();
    let dynamic_entries = layout
        .entries()
        .iter()
        .filter(|entry| {
            matches!(
                entry.kind,
                Some(BindingLayoutKind::Buffer {
                    has_dynamic_offset: true,
                    ..
                })
            )
        })
        .map(|entry| entry.binding)
        .collect::<Vec<_>>();

    for entry in bound.group.entries() {
        let Some(layout_entry) = layout_entries.get(&entry.binding).copied() else {
            continue;
        };
        let Some(kind) = layout_entry.kind else {
            continue;
        };
        match (&entry.resource, kind) {
            (
                BindGroupResource::Buffer {
                    buffer,
                    offset,
                    size,
                    ..
                },
                BindingLayoutKind::Buffer { .. },
            ) => {
                let access = match kind {
                    BindingLayoutKind::Buffer {
                        ty: BufferBindingType::Uniform | BufferBindingType::ReadOnlyStorage,
                        ..
                    } => ResourceAccess::Read,
                    BindingLayoutKind::Buffer {
                        ty: BufferBindingType::Storage,
                        ..
                    } => ResourceAccess::Write,
                    _ => unreachable!("buffer resource must have buffer binding layout"),
                };
                let dynamic_offset = dynamic_entries
                    .iter()
                    .position(|binding| *binding == entry.binding)
                    .and_then(|dynamic_index| bound.dynamic_offsets.get(dynamic_index))
                    .copied()
                    .unwrap_or(0);
                let offset = offset
                    .checked_add(u64::from(dynamic_offset))
                    .ok_or_else(|| "usage scope buffer offset overflows".to_owned())?;
                let size = if *size == u64::MAX {
                    buffer.size().saturating_sub(offset)
                } else {
                    size.saturating_sub(u64::from(dynamic_offset))
                };
                buffer_uses.push(BufferScopeUse {
                    buffer: Arc::clone(buffer),
                    offset,
                    size,
                    access,
                });
            }
            (
                BindGroupResource::TextureView { texture_view, .. },
                BindingLayoutKind::Texture { .. } | BindingLayoutKind::StorageTexture { .. },
            ) => {
                let access = match kind {
                    BindingLayoutKind::Texture { .. }
                    | BindingLayoutKind::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        ..
                    } => TextureAccess::Read,
                    BindingLayoutKind::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        ..
                    } => TextureAccess::WriteOnlyStorage,
                    BindingLayoutKind::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        ..
                    } => TextureAccess::ReadWriteStorage,
                    _ => unreachable!("texture resource must have texture binding layout"),
                };
                texture_uses.push(texture_scope_use(texture_view, access));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Validates buffer usage scope and returns a descriptive error on failure.
pub(crate) fn validate_buffer_usage_scope(buffer_uses: &[BufferScopeUse]) -> Result<(), String> {
    for (index, current) in buffer_uses.iter().enumerate() {
        for previous in &buffer_uses[..index] {
            if !current.buffer.same(&previous.buffer) {
                continue;
            }
            if current.access == ResourceAccess::Write || previous.access == ResourceAccess::Write {
                return Err(
                    "usage scope cannot read and write or write the same buffer range twice"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

/// Validates buffer usage scope allowing write/write overlap across render draws.
pub(crate) fn validate_buffer_usage_scope_lenient(
    buffer_uses: &[BufferScopeUse],
) -> Result<(), String> {
    for (index, current) in buffer_uses.iter().enumerate() {
        for previous in &buffer_uses[..index] {
            if !current.buffer.same(&previous.buffer) {
                continue;
            }
            if current.access != previous.access {
                return Err(
                    "usage scope cannot read and write or write the same buffer range twice"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

/// Validates strict texture usage scope and returns a descriptive error on failure.
pub(crate) fn validate_texture_usage_scope(texture_uses: &[TextureScopeUse]) -> Result<(), String> {
    for (index, current) in texture_uses.iter().enumerate() {
        for previous in &texture_uses[..index] {
            if !current.texture.same(&previous.texture)
                || !texture_subresource_ranges_overlap(current, previous)
                || !current.aspects.intersects(previous.aspects)
            {
                continue;
            }
            if !current.access.compatible_in_compute_scope(previous.access) {
                return Err(
                    "usage scope cannot read and write or write the same texture subresource twice"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

/// Validates texture usage scope allowing same-kind storage writes in render scopes.
#[allow(dead_code)]
pub(crate) fn validate_texture_usage_scope_lenient(
    texture_uses: &[TextureScopeUse],
) -> Result<(), String> {
    for (index, current) in texture_uses.iter().enumerate() {
        for previous in &texture_uses[..index] {
            if !current.texture.same(&previous.texture)
                || !texture_subresource_ranges_overlap(current, previous)
                || !current.aspects.intersects(previous.aspects)
            {
                continue;
            }
            if !current.access.compatible_in_render_scope(previous.access) {
                return Err(
                    "usage scope cannot read and write or write the same texture subresource twice"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn texture_scope_use(
    texture_view: &TextureView,
    access: TextureAccess,
) -> TextureScopeUse {
    TextureScopeUse {
        texture: texture_view.texture(),
        base_mip_level: texture_view.base_mip_level(),
        mip_level_count: texture_view.mip_level_count(),
        base_array_layer: texture_view.base_array_layer(),
        array_layer_count: texture_view.array_layer_count(),
        depth_slice: None,
        aspects: texture_view_aspect_mask(texture_view),
        access,
    }
}

pub(crate) fn texture_attachment_scope_use(
    texture_view: &TextureView,
    access: TextureAccess,
    aspects: TextureAspectMask,
    depth_slice: Option<u32>,
) -> TextureScopeUse {
    TextureScopeUse {
        texture: texture_view.texture(),
        base_mip_level: texture_view.base_mip_level(),
        mip_level_count: texture_view.mip_level_count(),
        base_array_layer: texture_view.base_array_layer(),
        array_layer_count: texture_view.array_layer_count(),
        depth_slice,
        aspects,
        access,
    }
}

pub(crate) fn texture_view_aspect_mask(texture_view: &TextureView) -> TextureAspectMask {
    match texture_view.aspect() {
        TextureAspect::DepthOnly => TextureAspectMask::DEPTH,
        TextureAspect::StencilOnly => TextureAspectMask::STENCIL,
        TextureAspect::All => texture_format_aspect_mask(texture_view),
    }
}

fn texture_format_aspect_mask(texture_view: &TextureView) -> TextureAspectMask {
    let caps = texture_view
        .texture()
        .view_format_caps(texture_view.format())
        .or_else(|| texture_view.texture().format_caps());
    let Some(caps) = caps else {
        return TextureAspectMask::COLOR;
    };
    let mut mask = TextureAspectMask(0);
    if caps.aspects.color {
        mask = mask.union(TextureAspectMask::COLOR);
    }
    if caps.aspects.depth {
        mask = mask.union(TextureAspectMask::DEPTH);
    }
    if caps.aspects.stencil {
        mask = mask.union(TextureAspectMask::STENCIL);
    }
    mask
}

pub(crate) fn texture_subresource_ranges_overlap(a: &TextureScopeUse, b: &TextureScopeUse) -> bool {
    ranges_overlap(
        a.base_mip_level,
        a.mip_level_count,
        b.base_mip_level,
        b.mip_level_count,
    ) && ranges_overlap(
        a.base_array_layer,
        a.array_layer_count,
        b.base_array_layer,
        b.array_layer_count,
    ) && depth_slices_overlap(a.depth_slice, b.depth_slice)
}

fn depth_slices_overlap(a: Option<u32>, b: Option<u32>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a == b,
        _ => true,
    }
}

fn ranges_overlap(a_start: u32, a_count: u32, b_start: u32, b_count: u32) -> bool {
    let a_end = a_start.saturating_add(a_count);
    let b_end = b_start.saturating_add(b_count);
    a_start < b_end && b_start < a_end
}

/// Returns buffer ranges overlap.
#[cfg(test)]
pub(crate) fn buffer_ranges_overlap(a: &BufferScopeUse, b: &BufferScopeUse) -> bool {
    let a_end = a.offset.saturating_add(a.size);
    let b_end = b.offset.saturating_add(b.size);
    a.offset < b_end && b.offset < a_end
}

/// Returns bind group buffer resources.
pub(crate) fn bind_group_buffer_resources(group: &BindGroup) -> Vec<Arc<Buffer>> {
    group
        .entries()
        .iter()
        .filter_map(|entry| match &entry.resource {
            BindGroupResource::Buffer { buffer, .. } => Some(Arc::clone(buffer)),
            _ => None,
        })
        .collect()
}

/// Returns bind group texture resources.
pub(crate) fn bind_group_texture_resources(group: &BindGroup) -> Vec<Texture> {
    group
        .entries()
        .iter()
        .filter_map(|entry| match &entry.resource {
            BindGroupResource::TextureView { texture_view, .. } => Some(texture_view.texture()),
            _ => None,
        })
        .collect()
}

/// Validates pipeline bind groups and returns a descriptive error on failure.
pub(crate) fn validate_pipeline_bind_groups(
    required_layouts: &[Arc<BindGroupLayout>],
    bound_groups: &BTreeMap<u32, BoundBindGroup>,
    limits: Limits,
) -> Result<(), String> {
    for (index, required_layout) in required_layouts.iter().enumerate() {
        if pipeline_bind_group_layout_is_implicitly_satisfied(required_layout) {
            continue;
        }
        let index = u32::try_from(index)
            .map_err(|_| "pipeline bind group index is too large".to_owned())?;
        let Some(bound) = bound_groups.get(&index) else {
            return Err("pipeline requires a missing bind group".to_owned());
        };
        validate_bound_bind_group(required_layout, bound, limits)?;
    }
    Ok(())
}

fn pipeline_bind_group_layout_is_implicitly_satisfied(layout: &BindGroupLayout) -> bool {
    if layout.entries().is_empty() {
        return true;
    }

    #[cfg(feature = "tiled")]
    {
        layout
            .entries()
            .iter()
            .all(|entry| matches!(entry.kind, Some(BindingLayoutKind::InputAttachment { .. })))
    }

    #[cfg(not(feature = "tiled"))]
    {
        let _ = layout;
        false
    }
}

/// Validates a single bound bind group against its required layout.
fn validate_bound_bind_group(
    required_layout: &Arc<BindGroupLayout>,
    bound: &BoundBindGroup,
    limits: Limits,
) -> Result<(), String> {
    if bound.group.is_error() {
        return Err("pipeline cannot use an error bind group".to_owned());
    }
    if !bind_group_layouts_compatible(required_layout, bound.group.layout()) {
        return Err("pipeline bind group layout is incompatible".to_owned());
    }
    validate_dynamic_offsets(
        required_layout,
        &bound.group,
        &bound.dynamic_offsets,
        limits,
    )
}

/// Returns bind group layouts compatible.
pub(crate) fn bind_group_layouts_compatible(
    required: &Arc<BindGroupLayout>,
    actual: &Arc<BindGroupLayout>,
) -> bool {
    if required.exclusive_pipeline() != actual.exclusive_pipeline() {
        return false;
    }
    if required.entries().len() != actual.entries().len() {
        return false;
    }
    let required_entries = required
        .entries()
        .iter()
        .map(|entry| (entry.binding, entry))
        .collect::<BTreeMap<_, _>>();
    let actual_entries = actual
        .entries()
        .iter()
        .map(|entry| (entry.binding, entry))
        .collect::<BTreeMap<_, _>>();
    required_entries == actual_entries
}

/// Validates a bind group command before storing it in encoder state.
pub(crate) fn validate_set_bind_group(
    index: u32,
    group: Option<&BindGroup>,
    dynamic_offsets: &[u32],
    limits: Limits,
) -> Result<(), String> {
    if index >= limits.max_bind_groups {
        return Err("bind group index exceeds the device limit".to_owned());
    }
    let Some(group) = group else {
        if dynamic_offsets.is_empty() {
            return Ok(());
        }
        return Err("clearing a bind group must not include dynamic offsets".to_owned());
    };
    if group.is_error() {
        return Err(if let Some(message) = group.error_message() {
            format!("cannot set an error bind group: {message}")
        } else {
            "cannot set an error bind group".to_owned()
        });
    }
    validate_dynamic_offsets(group.layout(), group, dynamic_offsets, limits)
}

/// Validates dynamic offsets and returns a descriptive error on failure.
pub(crate) fn validate_dynamic_offsets(
    layout: &BindGroupLayout,
    group: &BindGroup,
    dynamic_offsets: &[u32],
    limits: Limits,
) -> Result<(), String> {
    let dynamic_entries = layout
        .entries()
        .iter()
        .filter(|entry| {
            matches!(
                entry.kind,
                Some(BindingLayoutKind::Buffer {
                    has_dynamic_offset: true,
                    ..
                })
            )
        })
        .collect::<Vec<_>>();
    if dynamic_offsets.len() != dynamic_entries.len() {
        return Err("bind group dynamic offset count is invalid".to_owned());
    }

    for (layout_entry, dynamic_offset) in dynamic_entries.into_iter().zip(dynamic_offsets.iter()) {
        let Some(group_entry) = group
            .entries()
            .iter()
            .find(|entry| entry.binding == layout_entry.binding)
        else {
            return Err("bind group dynamic offset binding is missing".to_owned());
        };
        let Some(BindingLayoutKind::Buffer { ty, .. }) = layout_entry.kind else {
            continue;
        };
        let alignment = match ty {
            BufferBindingType::Uniform => limits.min_uniform_buffer_offset_alignment,
            BufferBindingType::Storage | BufferBindingType::ReadOnlyStorage => {
                limits.min_storage_buffer_offset_alignment
            }
        };
        if *dynamic_offset % alignment != 0 {
            return Err("bind group dynamic offset is not aligned".to_owned());
        }
        let BindGroupResource::Buffer {
            buffer,
            offset,
            size,
            ..
        } = &group_entry.resource
        else {
            return Err("bind group dynamic offset requires a buffer binding".to_owned());
        };
        let dynamic_offset = u64::from(*dynamic_offset);
        let base = offset
            .checked_add(dynamic_offset)
            .ok_or_else(|| "bind group dynamic offset range overflows".to_owned())?;
        if base > buffer.size() {
            return Err("bind group dynamic offset exceeds buffer size".to_owned());
        }
        if *size != u64::MAX {
            let end = base
                .checked_add(*size)
                .ok_or_else(|| "bind group dynamic offset range overflows".to_owned())?;
            if end > buffer.size() {
                return Err("bind group dynamic offset exceeds buffer size".to_owned());
            }
        }
    }

    Ok(())
}

/// Validates viewport and returns a descriptive error on failure.
pub(crate) fn validate_viewport(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
) -> Result<(), String> {
    if ![x, y, width, height, min_depth, max_depth]
        .into_iter()
        .all(f32::is_finite)
    {
        return Err("render pass viewport values must be finite".to_owned());
    }
    if width < 0.0 || height < 0.0 {
        return Err("render pass viewport width and height must be non-negative".to_owned());
    }
    if !(0.0..=1.0).contains(&min_depth)
        || !(0.0..=1.0).contains(&max_depth)
        || min_depth > max_depth
    {
        return Err("render pass viewport depth range is invalid".to_owned());
    }
    Ok(())
}

/// Validates viewport bounds against the device's maximum viewport dimensions.
pub(crate) fn validate_viewport_bounds(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    limits: Limits,
) -> Result<(), String> {
    let max = limits.max_texture_dimension_2d as f32;
    if width > max || height > max {
        return Err("render pass viewport size exceeds device bounds".to_owned());
    }
    let max_bounds = max * 2.0;
    if x < -max_bounds
        || y < -max_bounds
        || x + width > max_bounds - 1.0
        || y + height > max_bounds - 1.0
    {
        return Err("render pass viewport rectangle exceeds device bounds".to_owned());
    }
    Ok(())
}

/// Validates scissor rectangle containment in the active render attachment.
pub(crate) fn validate_scissor_rect(
    render_extent: Option<Extent3d>,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let Some(extent) = render_extent else {
        return Ok(());
    };
    let end_x = x
        .checked_add(width)
        .ok_or_else(|| "render pass scissor rectangle width overflows".to_owned())?;
    let end_y = y
        .checked_add(height)
        .ok_or_else(|| "render pass scissor rectangle height overflows".to_owned())?;
    if end_x > extent.width || end_y > extent.height {
        return Err("render pass scissor rectangle exceeds attachment size".to_owned());
    }
    Ok(())
}

/// Validates a `SetImmediates(offset, size)` command (Block 94), shared by
/// the compute pass, render pass, and render bundle encoders -- exactly
/// like [`validate_set_bind_group`] is shared across the same three
/// encoders. Authoritative source: Dawn's
/// `ProgrammableEncoder::ValidateSetImmediates`
/// (`dawn/native/ProgrammableEncoder.cpp:128-146`): 4-byte alignment on
/// both `offset` and `size`, then an overflow-safe range check against
/// `limits.max_immediate_size` -- the *device's* effective limit (mirrors
/// `GetDevice()->GetLimits()` there), not the adapter's supported limit.
/// Dawn checks `offset > max` then `size > max - offset` instead of
/// `offset + size > max` specifically to avoid a `u32` overflow on the
/// addition; this mirrors that structure.
///
/// Additionally clamps `limits.max_immediate_size` to
/// [`MAX_IMMEDIATE_DATA_BYTES`] (Dawn's own absolute ceiling on every
/// backend's advertised limit) so [`record_set_immediates`]'s scratch-buffer
/// write can never go out of bounds even if a `Limits` value were somehow
/// misconfigured above that ceiling.
pub(crate) fn validate_set_immediates(
    offset: u32,
    size: u32,
    limits: Limits,
) -> Result<(), String> {
    if !offset.is_multiple_of(4) {
        return Err("immediates offset must be 4-byte aligned".to_owned());
    }
    if !size.is_multiple_of(4) {
        return Err("immediates size must be 4-byte aligned".to_owned());
    }
    let max_immediate_size = limits
        .max_immediate_size
        .min(MAX_IMMEDIATE_DATA_BYTES as u32);
    if offset > max_immediate_size {
        return Err("immediates offset exceeds the device limit".to_owned());
    }
    if size > max_immediate_size - offset {
        return Err("immediates offset plus size exceeds the device limit".to_owned());
    }
    Ok(())
}

/// Overwrites `state.immediate_data[offset, offset + data.len())` after
/// validation. Mirrors Dawn's `UserImmediatesTrackerBase::SetImmediates`
/// (`dawn/native/ImmediatesTracker.h:81-87`): a byte-range overwrite into
/// the pass-scoped scratch buffer described on
/// [`MAX_IMMEDIATE_DATA_BYTES`] -- bytes outside `[offset, offset +
/// data.len())` are left untouched (whatever was last written there, or
/// zero from pass begin).
///
/// Callers must call [`validate_set_immediates`] first and skip this call
/// entirely when `data` is empty (Dawn: `ComputePassEncoder::APISetImmediates`
/// / `RenderEncoderBase::APISetImmediates` both validate unconditionally but
/// then early-return on `size == 0` without recording a command --
/// `dawn/native/ComputePassEncoder.cpp:562-565`,
/// `dawn/native/RenderEncoderBase.cpp:754-757`). This function relies on
/// that invariant (`offset + data.len() <= limits.max_immediate_size <=
/// MAX_IMMEDIATE_DATA_BYTES`) for the slice bounds and never panics as long
/// as it holds.
pub(crate) fn record_set_immediates(state: &mut PassEncoderState, offset: u32, data: &[u8]) {
    let start = offset as usize;
    let end = start + data.len();
    state.immediate_data[start..end].copy_from_slice(data);
    state.immediate_data_written |= written_words_mask(start, data.len());
}

/// Returns the [`ImmediateWrittenMask`] covering the byte range
/// `[offset, offset + len)` of the user-immediates scratch. Both values
/// must be 4-byte aligned and in-bounds (guaranteed by
/// [`validate_set_immediates`]); same bit math as Dawn's
/// `GetImmediateBlockBits` (`dawn/native/ImmediatesLayout.h:61-67`).
pub(crate) fn written_words_mask(offset: usize, len: usize) -> ImmediateWrittenMask {
    let first_word = offset / 4;
    let word_count = len / 4;
    // Compute in u32: `word_count` can be the full 16 words, where
    // `1u16 << 16` would overflow.
    (((1u32 << word_count) - 1) << first_word) as ImmediateWrittenMask
}

/// Validates that every required user-immediate word was explicitly written.
pub(crate) fn validate_required_immediate_data(
    required: ImmediateWrittenMask,
    written: ImmediateWrittenMask,
) -> Result<(), String> {
    let missing = required & !written;
    if missing == 0 {
        return Ok(());
    }
    let first_missing_offset = missing.trailing_zeros() * 4;
    Err(format!(
        "Required immediate data at offset {first_missing_offset} was not set."
    ))
}

/// Copies the 4-byte words of `src` flagged in `written` over `dest`,
/// leaving unflagged words of `dest` untouched. Both slices must be
/// [`MAX_IMMEDIATE_DATA_BYTES`] long. This implements Dawn's shared-tracker
/// bundle-replay semantics as an overlay: a render bundle contributes only
/// the byte ranges its own `SetImmediates` calls wrote, inheriting the rest
/// from the destination (the outer pass scratch).
pub(crate) fn overlay_written_immediates(
    dest: &mut [u8],
    src: &[u8],
    written: ImmediateWrittenMask,
) {
    for word in 0..(MAX_IMMEDIATE_DATA_BYTES / 4) {
        if written & (1 << word) != 0 {
            let start = word * 4;
            dest[start..start + 4].copy_from_slice(&src[start..start + 4]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{
        empty_bind_group, noop_device, noop_render_attachment, noop_render_pass_descriptor,
        noop_render_pipeline, noop_texture, render_pipeline_descriptor, render_shader_module,
    };
    use crate::Device;

    use std::sync::Arc;

    #[test]
    fn validate_required_immediate_data_requires_subset_of_written_words() {
        assert_eq!(validate_required_immediate_data(0b0111, 0b0111), Ok(()));
        assert_eq!(validate_required_immediate_data(0b0111, 0b1111), Ok(()));
        assert_eq!(validate_required_immediate_data(0, 0), Ok(()));

        assert_eq!(
            validate_required_immediate_data(0b0111, 0b0101),
            Err("Required immediate data at offset 4 was not set.".to_owned())
        );
        assert_eq!(
            validate_required_immediate_data(0b1000, 0),
            Err("Required immediate data at offset 12 was not set.".to_owned())
        );
    }

    fn texture_use(
        texture: &Texture,
        base_mip_level: u32,
        mip_level_count: u32,
        base_array_layer: u32,
        array_layer_count: u32,
        aspects: TextureAspectMask,
        access: TextureAccess,
    ) -> TextureScopeUse {
        TextureScopeUse {
            texture: texture.clone(),
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            depth_slice: None,
            aspects,
            access,
        }
    }

    fn uniform_layout_entry(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: 4,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: 16,
            }),
        }
    }

    #[cfg(feature = "tiled")]
    fn input_attachment_layout_entry(binding: u32) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: 4,
            binding_array_size: 0,
            kind: Some(BindingLayoutKind::InputAttachment {
                sample_type: TextureSampleType::Float,
                multisampled: false,
            }),
        }
    }

    fn pass_state_with_pipeline(
        device: &Device,
        pipeline: Arc<RenderPipeline>,
    ) -> PassEncoderState {
        let mut state = PassEncoderState::new(
            device.limits(),
            PassEncoderInit {
                attachment_signature: None,
                render_extent: None,
                attachment_textures: Vec::new(),
                render_color_attachments: Vec::new(),
                render_depth_stencil_attachment: None,
                occlusion_query_set: None,
                max_draw_count: u64::MAX,
            },
        );
        state.render_pipeline = Some(pipeline);
        state
    }

    fn vertex_buffer_binding(buffer: Arc<Buffer>, size: u64) -> BoundVertexBuffer {
        BoundVertexBuffer {
            buffer,
            offset: 0,
            size,
        }
    }

    fn sparse_vertex_pipeline(device: &Device) -> Arc<RenderPipeline> {
        let module = render_shader_module(device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.vertex.buffer_count = 8;
        descriptor.vertex.buffers = vec![
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: true,
                array_stride: 8,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![VertexAttribute {
                    format: VertexFormat::from_raw(0x0000_001D),
                    offset: 0,
                    shader_location: 2,
                }],
            },
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: false,
                array_stride: 0,
                step_mode: VertexStepMode::Vertex,
                attributes: Vec::new(),
            },
            VertexBufferLayout {
                used: true,
                array_stride: 8,
                step_mode: VertexStepMode::Instance,
                attributes: vec![VertexAttribute {
                    format: VertexFormat::from_raw(0x0000_001D),
                    offset: 0,
                    shader_location: 6,
                }],
            },
        ];
        Arc::new(device.create_render_pipeline(descriptor))
    }

    fn contiguous_vertex_pipeline(device: &Device) -> Arc<RenderPipeline> {
        let module = render_shader_module(device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.vertex.buffer_count = 2;
        descriptor.vertex.buffers = vec![
            VertexBufferLayout {
                used: true,
                array_stride: 8,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![VertexAttribute {
                    format: VertexFormat::from_raw(0x0000_001D),
                    offset: 0,
                    shader_location: 0,
                }],
            },
            VertexBufferLayout {
                used: true,
                array_stride: 8,
                step_mode: VertexStepMode::Vertex,
                attributes: vec![VertexAttribute {
                    format: VertexFormat::from_raw(0x0000_001D),
                    offset: 0,
                    shader_location: 1,
                }],
            },
        ];
        Arc::new(device.create_render_pipeline(descriptor))
    }

    fn buffer_scope_use(
        buffer: Arc<Buffer>,
        offset: u64,
        size: u64,
        access: ResourceAccess,
    ) -> BufferScopeUse {
        BufferScopeUse {
            buffer,
            offset,
            size,
            access,
        }
    }

    #[test]
    fn validate_buffer_usage_scope_uses_whole_buffer_access() {
        let device = noop_device();
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::UNIFORM,
            size: 64,
            mapped_at_creation: false,
        }));
        let other = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::UNIFORM,
            size: 64,
            mapped_at_creation: false,
        }));

        assert_eq!(
            validate_buffer_usage_scope(&[
                buffer_scope_use(buffer.clone(), 0, 16, ResourceAccess::Write),
                buffer_scope_use(buffer.clone(), 32, 16, ResourceAccess::Read),
            ]),
            Err(
                "usage scope cannot read and write or write the same buffer range twice".to_owned()
            )
        );
        assert!(!buffer_ranges_overlap(
            &buffer_scope_use(buffer.clone(), 0, 16, ResourceAccess::Write),
            &buffer_scope_use(buffer.clone(), 32, 16, ResourceAccess::Read),
        ));
        assert_eq!(
            validate_buffer_usage_scope(&[
                buffer_scope_use(buffer.clone(), 0, 16, ResourceAccess::Read),
                buffer_scope_use(buffer.clone(), 32, 16, ResourceAccess::Read),
            ]),
            Ok(())
        );
        assert_eq!(
            validate_buffer_usage_scope(&[
                buffer_scope_use(buffer, 0, 16, ResourceAccess::Write),
                buffer_scope_use(other, 0, 16, ResourceAccess::Read),
            ]),
            Ok(())
        );
    }

    #[test]
    fn validate_buffer_usage_scope_lenient_uses_whole_buffer_access_policy() {
        let device = noop_device();
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::UNIFORM,
            size: 64,
            mapped_at_creation: false,
        }));

        assert_eq!(
            validate_buffer_usage_scope_lenient(&[
                buffer_scope_use(buffer.clone(), 0, 16, ResourceAccess::Write),
                buffer_scope_use(buffer.clone(), 32, 16, ResourceAccess::Read),
            ]),
            Err(
                "usage scope cannot read and write or write the same buffer range twice".to_owned()
            )
        );
        assert_eq!(
            validate_buffer_usage_scope_lenient(&[
                buffer_scope_use(buffer.clone(), 0, 16, ResourceAccess::Write),
                buffer_scope_use(buffer, 32, 16, ResourceAccess::Write),
            ]),
            Ok(())
        );
    }

    #[test]
    fn record_buffer_usage_scope_use_accumulates_vertex_and_index_reads() {
        let device = noop_device();
        let pipeline = contiguous_vertex_pipeline(&device);
        let storage_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::VERTEX | BufferUsage::INDEX,
            size: 64,
            mapped_at_creation: false,
        }));
        let mut state = pass_state_with_pipeline(&device, pipeline.clone());
        state
            .vertex_buffers
            .insert(0, vertex_buffer_binding(Arc::clone(&storage_buffer), 16));
        state.index_buffer = Some(BoundIndexBuffer {
            buffer: Arc::clone(&storage_buffer),
            format: IndexFormat::Uint16,
            offset: 32,
            size: 16,
        });

        let vertex_buffer = state
            .vertex_buffers
            .get(&0)
            .expect("vertex buffer should be set")
            .clone();
        record_buffer_usage_scope_use(
            &mut state,
            BufferScopeUse {
                buffer: vertex_buffer.buffer,
                offset: vertex_buffer.offset,
                size: vertex_buffer.size,
                access: ResourceAccess::Read,
            },
        )
        .expect("vertex buffer read use should be valid");
        let index_buffer = state
            .index_buffer
            .as_ref()
            .expect("index buffer should be set")
            .clone();
        record_buffer_usage_scope_use(
            &mut state,
            BufferScopeUse {
                buffer: index_buffer.buffer,
                offset: index_buffer.offset,
                size: index_buffer.size,
                access: ResourceAccess::Read,
            },
        )
        .expect("index buffer read use should be valid");
        assert_eq!(state.scope_buffer_uses.len(), 2);
    }

    #[test]
    fn bind_group_layout_compatibility_uses_exclusive_pipeline_and_structure() {
        let entries = vec![uniform_layout_entry(0)];
        let explicit_a = Arc::new(BindGroupLayout::new(entries.clone(), false, false));
        let explicit_b = Arc::new(BindGroupLayout::new(entries.clone(), false, false));
        let auto_pipeline_a_group_0 = Arc::new(BindGroupLayout::new_auto(entries.clone(), 7));
        let auto_pipeline_a_group_1 = Arc::new(BindGroupLayout::new_auto(entries.clone(), 7));
        let auto_pipeline_b_group_0 = Arc::new(BindGroupLayout::new_auto(entries, 8));

        assert!(bind_group_layouts_compatible(&explicit_a, &explicit_b));
        assert!(bind_group_layouts_compatible(
            &auto_pipeline_a_group_0,
            &auto_pipeline_a_group_1
        ));
        assert!(!bind_group_layouts_compatible(
            &auto_pipeline_a_group_0,
            &auto_pipeline_b_group_0
        ));
        assert!(!bind_group_layouts_compatible(
            &explicit_a,
            &auto_pipeline_a_group_0
        ));
    }

    #[test]
    fn pass_command_after_end_defers_until_parent_finish_on_noop() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.end(), None);
        assert_eq!(pass.set_pipeline(Arc::clone(&pipeline)), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("pass encoder cannot be used after end".to_owned())
        );
        assert_eq!(
            pass.set_pipeline(pipeline),
            Some("pass encoder cannot be used after parent encoder finish".to_owned())
        );
    }

    #[test]
    fn texture_usage_scope_tracks_mip_layer_and_aspect_overlap() {
        let texture = noop_texture();
        let write_mip0 = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::WriteOnlyStorage,
        );
        let read_mip1 = texture_use(
            &texture,
            1,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::Read,
        );
        let read_layer1 = texture_use(
            &texture,
            0,
            1,
            1,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::Read,
        );
        let read_stencil = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::STENCIL,
            TextureAccess::Read,
        );

        assert_eq!(
            validate_texture_usage_scope(&[
                write_mip0.clone(),
                read_mip1,
                read_layer1,
                read_stencil
            ]),
            Ok(())
        );

        let read_same_subresource = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::Read,
        );
        assert_eq!(
            validate_texture_usage_scope(&[write_mip0, read_same_subresource]),
            Err(
                "usage scope cannot read and write or write the same texture subresource twice"
                    .to_owned()
            )
        );
    }

    #[test]
    fn texture_usage_scope_tracks_3d_attachment_depth_slices() {
        let texture = noop_texture();
        let mut write_slice0 = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::AttachmentWrite,
        );
        write_slice0.depth_slice = Some(0);
        let mut write_slice1 = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::AttachmentWrite,
        );
        write_slice1.depth_slice = Some(1);
        assert_eq!(
            validate_texture_usage_scope(&[write_slice0.clone(), write_slice1]),
            Ok(())
        );

        let mut write_same_slice = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::AttachmentWrite,
        );
        write_same_slice.depth_slice = Some(0);
        let expected = Err(
            "usage scope cannot read and write or write the same texture subresource twice"
                .to_owned(),
        );
        assert_eq!(
            validate_texture_usage_scope(&[write_slice0.clone(), write_same_slice]),
            expected
        );

        let read_whole_range = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::Read,
        );
        assert_eq!(
            validate_texture_usage_scope(&[write_slice0, read_whole_range]),
            Err(
                "usage scope cannot read and write or write the same texture subresource twice"
                    .to_owned()
            )
        );
    }

    #[test]
    fn texture_usage_scope_uses_texture_access_compatibility() {
        let texture = noop_texture();
        let read_a = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::Read,
        );
        let read_b = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::Read,
        );
        assert_eq!(
            validate_texture_usage_scope(&[read_a.clone(), read_b]),
            Ok(())
        );

        let storage_write_a = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::WriteOnlyStorage,
        );
        let storage_write_b = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::WriteOnlyStorage,
        );
        assert_eq!(
            validate_texture_usage_scope(&[storage_write_a.clone(), storage_write_b.clone()]),
            Err(
                "usage scope cannot read and write or write the same texture subresource twice"
                    .to_owned()
            )
        );
        let readwrite_storage_a = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::ReadWriteStorage,
        );
        let readwrite_storage_b = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::ReadWriteStorage,
        );
        assert_eq!(
            validate_texture_usage_scope(&[
                readwrite_storage_a.clone(),
                readwrite_storage_b.clone()
            ]),
            Err(
                "usage scope cannot read and write or write the same texture subresource twice"
                    .to_owned()
            )
        );
        assert_eq!(
            validate_texture_usage_scope_lenient(&[storage_write_a.clone(), storage_write_b]),
            Ok(())
        );
        assert_eq!(
            validate_texture_usage_scope_lenient(&[
                readwrite_storage_a.clone(),
                readwrite_storage_b
            ]),
            Ok(())
        );

        assert_eq!(
            validate_texture_usage_scope(&[read_a, storage_write_a.clone()]),
            Err(
                "usage scope cannot read and write or write the same texture subresource twice"
                    .to_owned()
            )
        );
        assert_eq!(
            validate_texture_usage_scope(&[storage_write_a.clone(), readwrite_storage_a]),
            Err(
                "usage scope cannot read and write or write the same texture subresource twice"
                    .to_owned()
            )
        );

        let attachment_write_a = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::AttachmentWrite,
        );
        let attachment_write_b = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            TextureAccess::AttachmentWrite,
        );
        assert_eq!(
            validate_texture_usage_scope(&[storage_write_a, attachment_write_a.clone()]),
            Err(
                "usage scope cannot read and write or write the same texture subresource twice"
                    .to_owned()
            )
        );
        assert_eq!(
            validate_texture_usage_scope(&[attachment_write_a, attachment_write_b]),
            Err(
                "usage scope cannot read and write or write the same texture subresource twice"
                    .to_owned()
            )
        );
    }

    #[test]
    fn viewport_bounds_use_double_limit_rectangle_and_single_limit_size() {
        let limits = Limits::DEFAULT;
        let max = limits.max_texture_dimension_2d as f32;
        let double_max = max * 2.0;

        assert_eq!(validate_viewport_bounds(1.0, 0.0, max, max, limits), Ok(()));
        assert_eq!(
            validate_viewport_bounds(max, 0.0, max, max, limits),
            Err("render pass viewport rectangle exceeds device bounds".to_owned())
        );
        assert_eq!(
            validate_viewport_bounds(0.0, 0.0, max + 1.0, 1.0, limits),
            Err("render pass viewport size exceeds device bounds".to_owned())
        );
        assert_eq!(
            validate_viewport_bounds(-double_max, 0.0, max, max, limits),
            Ok(())
        );
        assert_eq!(
            validate_viewport_bounds(-double_max - 1.0, 0.0, max, max, limits),
            Err("render pass viewport rectangle exceeds device bounds".to_owned())
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn validate_pipeline_bind_groups_skips_input_attachment_only_layout() {
        let device = noop_device();
        let input_only_layout =
            Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
                entries: vec![input_attachment_layout_entry(0)],
                error: None,
            }));
        let mixed_layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![input_attachment_layout_entry(0), uniform_layout_entry(1)],
            error: None,
        }));
        let bound_groups = BTreeMap::new();

        assert_eq!(
            validate_pipeline_bind_groups(&[input_only_layout], &bound_groups, device.limits()),
            Ok(())
        );
        assert_eq!(
            validate_pipeline_bind_groups(&[mixed_layout], &bound_groups, device.limits()),
            Err("pipeline requires a missing bind group".to_owned())
        );
    }

    #[test]
    fn draw_time_bind_groups_plus_vertex_buffers_is_checked_at_limit() {
        let device = noop_device();
        let mut limits = device.limits();
        limits.max_bind_groups_plus_vertex_buffers = 3;
        let bind_group = empty_bind_group(&device);
        let vertex_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::VERTEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let mut state = PassEncoderState::new(
            limits,
            PassEncoderInit {
                attachment_signature: None,
                render_extent: None,
                attachment_textures: Vec::new(),
                render_color_attachments: Vec::new(),
                render_depth_stencil_attachment: None,
                occlusion_query_set: None,
                max_draw_count: u64::MAX,
            },
        );
        state.bind_groups.insert(
            1,
            BoundBindGroup {
                group: bind_group,
                dynamic_offsets: Vec::new(),
            },
        );
        state.vertex_buffers.insert(
            0,
            BoundVertexBuffer {
                buffer: vertex_buffer.clone(),
                offset: 0,
                size: 16,
            },
        );
        assert_eq!(
            validate_draw_time_bind_groups_plus_vertex_buffers(&state, limits),
            Ok(())
        );

        state.vertex_buffers.insert(
            1,
            BoundVertexBuffer {
                buffer: vertex_buffer,
                offset: 0,
                size: 16,
            },
        );
        assert_eq!(
            validate_draw_time_bind_groups_plus_vertex_buffers(&state, limits),
            Err(
                "render pass draw bind group plus vertex buffer count exceeds the device limit"
                    .to_owned()
            )
        );
    }

    #[test]
    fn sparse_vertex_buffer_slots_require_only_used_bindings() {
        let device = noop_device();
        let pipeline = sparse_vertex_pipeline(&device);
        assert!(!pipeline.is_error());
        let vertex_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::VERTEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let instance_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::VERTEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let mut state = pass_state_with_pipeline(&device, Arc::clone(&pipeline));
        state
            .vertex_buffers
            .insert(1, vertex_buffer_binding(vertex_buffer, 16));
        state
            .vertex_buffers
            .insert(7, vertex_buffer_binding(instance_buffer, 16));

        assert_eq!(
            validate_render_draw_base_state(&state, device.limits(), false).map(|_| ()),
            Ok(())
        );
        assert_eq!(
            validate_vertex_buffer_oob(&pipeline, &state, Some((0, 2)), 0, 2),
            Ok(())
        );
    }

    #[test]
    fn contiguous_vertex_buffer_slots_still_require_each_binding() {
        let device = noop_device();
        let pipeline = contiguous_vertex_pipeline(&device);
        assert!(!pipeline.is_error());
        let vertex_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::VERTEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let mut state = pass_state_with_pipeline(&device, pipeline);
        state
            .vertex_buffers
            .insert(0, vertex_buffer_binding(vertex_buffer, 16));

        assert_eq!(
            validate_render_draw_base_state(&state, device.limits(), false)
                .map(|_| ())
                .expect_err("missing contiguous vertex slot"),
            "render pass draw requires all declared vertex buffers to be set"
        );
    }
}
