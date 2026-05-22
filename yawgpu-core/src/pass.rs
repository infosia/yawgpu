use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::bind_group::*;
use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::compute_pipeline::*;
use crate::limits::*;
use crate::query_set::*;
use crate::render_pipeline::*;
use crate::texture::*;

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
    pub(crate) bind_groups: BTreeMap<u32, BoundBindGroup>,
    pub(crate) vertex_buffers: BTreeMap<u32, BoundVertexBuffer>,
    pub(crate) index_buffer: Option<BoundIndexBuffer>,
    pub(crate) attachment_signature: Option<AttachmentSignature>,
    pub(crate) attachment_textures: Vec<Texture>,
    pub(crate) render_color_attachment: Option<RenderPassColorExecution>,
    pub(crate) render_pass_recorded: bool,
    pub(crate) occlusion_query_set: Option<QuerySet>,
    pub(crate) open_occlusion_query: Option<u32>,
    pub(crate) used_occlusion_queries: BTreeSet<u32>,
}

impl PassEncoderState {
    /// Creates a new instance.
    pub(crate) fn new(
        attachment_signature: Option<AttachmentSignature>,
        attachment_textures: Vec<Texture>,
        render_color_attachment: Option<RenderPassColorExecution>,
        occlusion_query_set: Option<QuerySet>,
    ) -> Self {
        Self {
            ended: false,
            debug_group_depth: 0,
            render_pipeline: None,
            compute_pipeline: None,
            bind_groups: BTreeMap::new(),
            vertex_buffers: BTreeMap::new(),
            index_buffer: None,
            attachment_signature,
            attachment_textures,
            render_color_attachment,
            render_pass_recorded: false,
            occlusion_query_set,
            open_occlusion_query: None,
            used_occlusion_queries: BTreeSet::new(),
        }
    }

    /// Resets the tracked render-pipeline / vertex / index state for this pass.
    pub(crate) fn clear_render_state(&mut self) {
        self.render_pipeline = None;
        self.bind_groups.clear();
        self.vertex_buffers.clear();
        self.index_buffer = None;
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
    pub(crate) fn new(
        parent: CommandEncoder,
        token: PassToken,
        attachment_signature: Option<AttachmentSignature>,
        attachment_textures: Vec<Texture>,
        render_color_attachment: Option<RenderPassColorExecution>,
        occlusion_query_set: Option<QuerySet>,
    ) -> Self {
        Self {
            parent,
            token,
            state: Mutex::new(PassEncoderState::new(
                attachment_signature,
                attachment_textures,
                render_color_attachment,
                occlusion_query_set,
            )),
        }
    }

    /// Ends recording for this pass or encoder.
    pub(crate) fn end(&self) -> Option<String> {
        let mut state = self.state.lock();
        if state.ended {
            let message = "pass encoder cannot be ended more than once".to_owned();
            self.parent.record_first_error(message.clone());
            return Some(message);
        }
        if self.parent.is_finished() {
            let message = "pass encoder cannot be used after parent encoder finish".to_owned();
            self.parent.record_first_error(message.clone());
            return Some(message);
        }
        state.ended = true;
        let unbalanced_debug_groups = state.debug_group_depth != 0;
        let open_occlusion_query = state.open_occlusion_query.is_some();
        let render_pass_command = if !state.render_pass_recorded {
            state
                .render_color_attachment
                .clone()
                .map(|color_attachment| {
                    state.render_pass_recorded = true;
                    RenderPassCommand {
                        pipeline: state.render_pipeline.clone(),
                        color_attachment,
                        bind_groups: state.bind_groups.clone(),
                        vertex_buffers: state.vertex_buffers.clone(),
                        draw: None,
                    }
                })
        } else {
            None
        };
        drop(state);

        if let Some(command) = render_pass_command {
            self.parent.record_render_pass(command);
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
            self.parent.record_first_error(message.clone());
            return Err(message);
        }
        if self.state.lock().ended {
            let message = "pass encoder cannot be used after end".to_owned();
            self.parent.record_first_error(message.clone());
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
        Some(&state.attachment_textures),
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
    if buffer.is_destroyed() {
        return Err("render pass index buffer must not be destroyed".to_owned());
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
    if buffer.is_destroyed() {
        return Err("render pass vertex buffer must not be destroyed".to_owned());
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
        let required_size = layout
            .array_stride
            .checked_mul(u64::from(stride_count))
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
    if buffer.is_destroyed() {
        return Err(format!("{label} buffer must not be destroyed"));
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
#[derive(Debug)]
pub(crate) struct BufferScopeUse {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) access: ResourceAccess,
}

/// Stores texture scope use data used by validation and backend submission.
#[derive(Debug)]
pub(crate) struct TextureScopeUse {
    pub(crate) texture: Texture,
    pub(crate) access: ResourceAccess,
}

/// Validates usage scope and returns a descriptive error on failure.
pub(crate) fn validate_usage_scope(
    required_layouts: &[Arc<BindGroupLayout>],
    bound_groups: &BTreeMap<u32, BoundBindGroup>,
    attachment_textures: Option<&[Texture]>,
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

    validate_buffer_usage_scope(&buffer_uses)?;
    validate_texture_usage_scope(&texture_uses)?;
    if let Some(attachment_textures) = attachment_textures {
        for texture_use in &texture_uses {
            if attachment_textures
                .iter()
                .any(|attachment| attachment.same(&texture_use.texture))
            {
                return Err(
                    "render pass attachment texture cannot be used through a bind group".to_owned(),
                );
            }
        }
    }
    Ok(())
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
            ) => texture_uses.push(TextureScopeUse {
                texture: texture_view.texture(),
                access,
            }),
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

/// Validates texture usage scope and returns a descriptive error on failure.
pub(crate) fn validate_texture_usage_scope(texture_uses: &[TextureScopeUse]) -> Result<(), String> {
    for (index, current) in texture_uses.iter().enumerate() {
        for previous in &texture_uses[..index] {
            if !current.texture.same(&previous.texture) {
                continue;
            }
            if current.access == ResourceAccess::Write || previous.access == ResourceAccess::Write {
                return Err(
                    "usage scope cannot read and write or write the same texture twice".to_owned(),
                );
            }
        }
    }
    Ok(())
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

/// Validates pipeline bind groups and returns a descriptive error on failure.
pub(crate) fn validate_pipeline_bind_groups(
    required_layouts: &[Arc<BindGroupLayout>],
    bound_groups: &BTreeMap<u32, BoundBindGroup>,
    limits: Limits,
) -> Result<(), String> {
    for (index, required_layout) in required_layouts.iter().enumerate() {
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
    validate_dynamic_offsets(required_layout, &bound.group, &bound.dynamic_offsets, limits)
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
    required.entries() == actual.entries()
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
        if *size != u64::MAX && dynamic_offset > *size {
            return Err("bind group dynamic offset exceeds binding size".to_owned());
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
