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
}

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
    pub(crate) fn set_attachment_texture_uses(&mut self, uses: Vec<TextureScopeUse>) {
        self.attachment_texture_uses = uses;
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
        self.pass_command_guard().err()
    }

    /// Opens a debug group in the shared pass state.
    pub(crate) fn push_debug_group(&self) -> Option<String> {
        if let Err(message) = self.pass_command_guard() {
            return Some(message);
        }
        let mut state = self.state.lock();
        state.debug_group_depth = state.debug_group_depth.saturating_add(1);
        None
    }

    /// Closes the most recently opened debug group in the shared pass state.
    pub(crate) fn pop_debug_group(&self) -> Option<String> {
        if let Err(message) = self.pass_command_guard() {
            return Some(message);
        }
        let mut state = self.state.lock();
        if state.debug_group_depth == 0 {
            let message = "pass encoder debug group stack is empty".to_owned();
            self.parent.record_first_error(message);
            None
        } else {
            state.debug_group_depth -= 1;
            None
        }
    }

    /// Records a pass command after confirming the pass is still open.
    pub(crate) fn record_pass_command<F>(&self, command: F) -> Option<String>
    where
        F: FnOnce(&mut PassEncoderState) -> Result<(), String>,
    {
        if let Err(message) = self.pass_command_guard() {
            return Some(message);
        }
        let mut state = self.state.lock();
        if let Err(message) = command(&mut state) {
            self.parent.record_first_error(message);
        }
        None
    }

    /// Begins a command guard for this pass.
    pub(crate) fn pass_command_guard(&self) -> Result<(), String> {
        if self.parent.is_finished() {
            let message = "pass encoder cannot be used after parent encoder finish".to_owned();
            return Err(message);
        }
        if self.state.lock().ended {
            let message = "pass encoder cannot be used after end".to_owned();
            return Err(message);
        }
        Ok(())
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
    validate_usage_scope(
        pipeline.bind_group_layouts(),
        &state.bind_groups,
        Some(&state.attachment_texture_uses),
    )?;
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
    for slot in 0..pipeline.required_vertex_buffer_count() {
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
        if layout.array_stride == 0 {
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

/// Stores buffer scope use data used by validation and backend submission.
#[derive(Debug, Clone)]
pub(crate) struct BufferScopeUse {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) offset: u64,
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
    pub(crate) access: ResourceAccess,
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
    attachment_uses: &[TextureScopeUse],
) -> Result<(), String> {
    let (buffer_uses, texture_uses) =
        collect_pipeline_usage_scope(required_layouts, &state.bind_groups)?;
    let mut current_texture_uses = texture_uses.clone();
    current_texture_uses.extend_from_slice(attachment_uses);
    validate_resource_usage_scope(&buffer_uses, &current_texture_uses)?;
    let mut scoped_buffer_uses = state.scope_buffer_uses.clone();
    scoped_buffer_uses.extend(buffer_uses.iter().cloned());
    let mut scoped_texture_uses = state.scope_texture_uses.clone();
    scoped_texture_uses.extend(texture_uses.iter().cloned());
    scoped_texture_uses.extend_from_slice(attachment_uses);
    validate_resource_usage_scope_lenient(&scoped_buffer_uses, &scoped_texture_uses)?;
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
        let access = match kind {
            BindingLayoutKind::Buffer {
                ty: BufferBindingType::Uniform | BufferBindingType::ReadOnlyStorage,
                ..
            }
            | BindingLayoutKind::Texture { .. }
            | BindingLayoutKind::StorageTexture {
                access: StorageTextureAccess::ReadOnly,
                ..
            } => Some(ResourceAccess::Read),
            BindingLayoutKind::Buffer {
                ty: BufferBindingType::Storage,
                ..
            }
            | BindingLayoutKind::StorageTexture {
                access: StorageTextureAccess::WriteOnly | StorageTextureAccess::ReadWrite,
                ..
            } => Some(ResourceAccess::Write),
            #[cfg(feature = "tiled")]
            BindingLayoutKind::InputAttachment { .. } => None,
            BindingLayoutKind::Sampler { .. } => None,
        };
        let Some(access) = access else {
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
            ) => texture_uses.push(texture_scope_use(texture_view, access)),
            _ => {}
        }
    }
    Ok(())
}

/// Validates buffer usage scope and returns a descriptive error on failure.
pub(crate) fn validate_buffer_usage_scope(buffer_uses: &[BufferScopeUse]) -> Result<(), String> {
    for (index, current) in buffer_uses.iter().enumerate() {
        for previous in &buffer_uses[..index] {
            if !current.buffer.same(&previous.buffer) || !buffer_ranges_overlap(current, previous) {
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
            if !current.buffer.same(&previous.buffer) || !buffer_ranges_overlap(current, previous) {
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

/// Validates texture usage scope and returns a descriptive error on failure.
pub(crate) fn validate_texture_usage_scope(texture_uses: &[TextureScopeUse]) -> Result<(), String> {
    for (index, current) in texture_uses.iter().enumerate() {
        for previous in &texture_uses[..index] {
            if !current.texture.same(&previous.texture)
                || !texture_subresource_ranges_overlap(current, previous)
                || !current.aspects.intersects(previous.aspects)
            {
                continue;
            }
            if current.access == ResourceAccess::Write || previous.access == ResourceAccess::Write {
                return Err(
                    "usage scope cannot read and write or write the same texture subresource twice"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

/// Validates texture usage scope allowing write/write overlap across render draws.
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
            if current.access != previous.access {
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
    access: ResourceAccess,
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
    access: ResourceAccess,
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
        if required_layout.entries().is_empty() {
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

/// Validates subpass pipeline bind groups, treating input-attachment-only groups
/// as auto-wired by the subpass pass.
///
/// A subpass-input binding is supplied from the pass layout's input-source
/// mapping rather than a caller-bound bind group, so a required layout whose
/// entries are all input attachments is satisfied without a `set_bind_group`
/// call. Any group that mixes input attachments with other resources still
/// requires a caller-bound group (the non-input bindings are not auto-wired).
#[cfg(feature = "tiled")]
pub(crate) fn validate_subpass_pipeline_bind_groups(
    required_layouts: &[Arc<BindGroupLayout>],
    bound_groups: &BTreeMap<u32, BoundBindGroup>,
    limits: Limits,
) -> Result<(), String> {
    for (index, required_layout) in required_layouts.iter().enumerate() {
        if required_layout.entries().is_empty() {
            continue;
        }
        let index = u32::try_from(index)
            .map_err(|_| "pipeline bind group index is too large".to_owned())?;
        let Some(bound) = bound_groups.get(&index) else {
            if bind_group_layout_is_input_attachment_only(required_layout) {
                continue;
            }
            return Err("pipeline requires a missing bind group".to_owned());
        };
        validate_bound_bind_group(required_layout, bound, limits)?;
    }
    Ok(())
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

/// Returns whether every entry in the layout is an input attachment.
///
/// Such a group is wired automatically by the subpass pass, so the pipeline does
/// not need a caller-bound bind group for it. An empty layout is not treated as
/// input-attachment-only (it carries no auto-wired binding to satisfy).
#[cfg(feature = "tiled")]
fn bind_group_layout_is_input_attachment_only(layout: &BindGroupLayout) -> bool {
    let entries = layout.entries();
    !entries.is_empty()
        && entries
            .iter()
            .all(|entry| matches!(entry.kind, Some(BindingLayoutKind::InputAttachment { .. })))
}

/// Returns bind group layouts compatible.
pub(crate) fn bind_group_layouts_compatible(
    required: &Arc<BindGroupLayout>,
    actual: &Arc<BindGroupLayout>,
) -> bool {
    if required.is_default() || actual.is_default() {
        return required.same(actual);
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
        return Err("cannot set an error bind group".to_owned());
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
    if x < -max || y < -max || x + width > max || y + height > max {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{empty_bind_group, noop_device, noop_texture};

    use std::sync::Arc;

    fn texture_use(
        texture: &Texture,
        base_mip_level: u32,
        mip_level_count: u32,
        base_array_layer: u32,
        array_layer_count: u32,
        aspects: TextureAspectMask,
        access: ResourceAccess,
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
            ResourceAccess::Write,
        );
        let read_mip1 = texture_use(
            &texture,
            1,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            ResourceAccess::Read,
        );
        let read_layer1 = texture_use(
            &texture,
            0,
            1,
            1,
            1,
            TextureAspectMask::COLOR,
            ResourceAccess::Read,
        );
        let read_stencil = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::STENCIL,
            ResourceAccess::Read,
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
            ResourceAccess::Read,
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
            ResourceAccess::Write,
        );
        write_slice0.depth_slice = Some(0);
        let mut write_slice1 = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            ResourceAccess::Write,
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
            ResourceAccess::Write,
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
            ResourceAccess::Read,
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
    fn texture_usage_scope_allows_read_only_overlap_but_rejects_write_write() {
        let texture = noop_texture();
        let read_a = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            ResourceAccess::Read,
        );
        let read_b = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            ResourceAccess::Read,
        );
        assert_eq!(
            validate_texture_usage_scope(&[read_a.clone(), read_b]),
            Ok(())
        );

        let write = texture_use(
            &texture,
            0,
            1,
            0,
            1,
            TextureAspectMask::COLOR,
            ResourceAccess::Write,
        );
        assert_eq!(
            validate_texture_usage_scope(&[read_a, write]),
            Err(
                "usage scope cannot read and write or write the same texture subresource twice"
                    .to_owned()
            )
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
}
