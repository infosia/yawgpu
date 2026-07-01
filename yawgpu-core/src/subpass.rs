use std::sync::Arc;

#[cfg(feature = "tiled")]
use parking_lot::Mutex;
#[cfg(feature = "tiled")]
use yawgpu_hal::{
    HalSubpassAttachmentLayout, HalSubpassDependency, HalSubpassDependencyType,
    HalSubpassInputAttachment, HalSubpassLayout, HalSubpassPassLayout, HalSubpassRenderPass,
};

use crate::adapter::{tiled_features_supported, TiledCapabilities};
#[cfg(feature = "tiled")]
use crate::bind_group::*;
#[cfg(feature = "tiled")]
use crate::buffer::*;
#[cfg(feature = "tiled")]
use crate::command_encoder::*;
#[cfg(feature = "tiled")]
use crate::copy::*;
use crate::device::Device;
use crate::error::ErrorKind;
#[cfg(feature = "tiled")]
use crate::extent::Extent3d;
use crate::format::TextureFormat;
#[cfg(feature = "tiled")]
use crate::limits::Limits;
#[cfg(feature = "tiled")]
use crate::pass::*;
#[cfg(feature = "tiled")]
use crate::render_pipeline::*;
#[cfg(feature = "tiled")]
use crate::texture::{hal_texture_format, TextureDimension, TextureUsage};
#[cfg(feature = "tiled")]
use crate::texture_view::{TextureView, TextureViewDimension};

/// Sentinel source attachment index for the depth-stencil attachment.
pub const DEPTH_STENCIL_ATTACHMENT_INDEX: u32 = u32::MAX;

/// Describes one subpass attachment slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttachmentLayout {
    /// Format.
    pub format: TextureFormat,
    /// Sample count.
    pub sample_count: u32,
}

/// Describes one input attachment source mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubpassInputAttachment {
    /// Bind group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Source subpass.
    pub source_subpass: u32,
    /// Source attachment index, or `DEPTH_STENCIL_ATTACHMENT_INDEX`.
    pub source_attachment: u32,
}

/// Enumerates subpass dependency kind values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SubpassDependencyType {
    /// Color to input variant.
    ColorToInput,
    /// Depth to input variant.
    DepthToInput,
    /// Color and depth to input variant.
    ColorDepthToInput,
}

/// Describes one subpass dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubpassDependency {
    /// Source subpass.
    pub src_subpass: u32,
    /// Destination subpass.
    pub dst_subpass: u32,
    /// Dependency kind.
    pub dependency_type: SubpassDependencyType,
    /// Whether dependency is region-local.
    pub by_region: bool,
}

/// Describes one subpass in a pass layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubpassLayoutDesc {
    /// Color attachment slot indices used by this subpass.
    pub color_attachment_indices: Vec<u32>,
    /// Whether this subpass uses the depth-stencil slot.
    pub uses_depth_stencil: bool,
    /// Input attachment mappings for this subpass.
    pub input_attachments: Vec<SubpassInputAttachment>,
}

/// Describes a subpass pass layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubpassPassLayoutDescriptor {
    /// Color attachment slot layouts.
    pub color_attachments: Vec<AttachmentLayout>,
    /// Optional depth-stencil attachment layout.
    pub depth_stencil_attachment: Option<AttachmentLayout>,
    /// Subpasses.
    pub subpasses: Vec<SubpassLayoutDesc>,
    /// Dependencies.
    pub dependencies: Vec<SubpassDependency>,
    /// Descriptor error from FFI conversion.
    pub error: Option<String>,
}

/// Maps each input attachment of `subpass_index` to its Metal `[[color(N)]]`
/// slot: key = the input's `(group, binding)`, value = the source color
/// attachment index it reads (`SubpassInputAttachment::source_attachment`).
pub(crate) fn compute_subpass_color_slots(
    layout: &SubpassPassLayoutDescriptor,
    subpass_index: u32,
) -> Vec<((u32, u32), u32)> {
    layout
        .subpasses
        .get(subpass_index as usize)
        .map(|subpass| {
            subpass
                .input_attachments
                .iter()
                .map(|input| ((input.group, input.binding), input.source_attachment))
                .collect()
        })
        .unwrap_or_default()
}

/// Stores a reusable subpass pass layout.
#[derive(Debug, Clone)]
pub struct SubpassPassLayout {
    inner: Arc<SubpassPassLayoutInner>,
}

/// Holds shared subpass pass layout state.
#[derive(Debug)]
pub(crate) struct SubpassPassLayoutInner {
    pub(crate) descriptor: SubpassPassLayoutDescriptor,
    pub(crate) is_error: bool,
}

impl SubpassPassLayout {
    /// Creates a new layout.
    #[must_use]
    pub(crate) fn new(descriptor: SubpassPassLayoutDescriptor, is_error: bool) -> Self {
        Self {
            inner: Arc::new(SubpassPassLayoutInner {
                descriptor,
                is_error,
            }),
        }
    }

    /// Returns the descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &SubpassPassLayoutDescriptor {
        &self.inner.descriptor
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns true when both handles share the same backing object.
    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

/// Converts a subpass pass layout descriptor to the HAL representation.
#[cfg(feature = "tiled")]
pub(crate) fn hal_subpass_pass_layout(
    layout: &SubpassPassLayoutDescriptor,
) -> HalSubpassPassLayout {
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

/// Enumerates subpass attachment resources.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub enum SubpassAttachmentResource {
    /// Persistent texture view.
    Persistent {
        /// View.
        view: Arc<TextureView>,
        /// Resolve target.
        resolve_target: Option<Arc<TextureView>>,
    },
    // TODO(tiled 2.4): Transient arm
}

/// Describes one color attachment binding.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub struct SubpassColorAttachmentBinding {
    /// Resource.
    pub resource: SubpassAttachmentResource,
    /// Load op.
    pub load_op: LoadOp,
    /// Store op.
    pub store_op: StoreOp,
    /// Clear value.
    pub clear_value: Color,
}

/// Describes one depth-stencil attachment binding.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub struct SubpassDepthStencilAttachmentBinding {
    /// Resource.
    pub resource: SubpassAttachmentResource,
    /// Depth load op.
    pub depth_load_op: LoadOp,
    /// Depth store op.
    pub depth_store_op: StoreOp,
    /// Depth clear value.
    pub depth_clear_value: f32,
    /// Stencil load op.
    pub stencil_load_op: LoadOp,
    /// Stencil store op.
    pub stencil_store_op: StoreOp,
    /// Stencil clear value.
    pub stencil_clear_value: u32,
}

/// Describes a subpass render pass begin operation.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub struct SubpassRenderPassDescriptor {
    /// Pass layout.
    pub pass_layout: Arc<SubpassPassLayout>,
    /// Pass extent.
    pub extent: Extent3d,
    /// Color attachments by slot.
    pub color_attachments: Vec<SubpassColorAttachmentBinding>,
    /// Depth-stencil attachment.
    pub depth_stencil_attachment: Option<SubpassDepthStencilAttachmentBinding>,
    /// Descriptor error from FFI conversion.
    pub error: Option<String>,
}

/// Stores the data needed to replay one subpass draw.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub(crate) struct SubpassDrawExecution {
    pub(crate) subpass_index: u32,
    pub(crate) pipeline: Arc<RenderPipeline>,
    pub(crate) bind_groups: std::collections::BTreeMap<u32, BoundBindGroup>,
    pub(crate) vertex_buffers: std::collections::BTreeMap<u32, BoundVertexBuffer>,
    #[allow(dead_code)]
    pub(crate) index_buffer: Option<BoundIndexBuffer>,
    pub(crate) viewport: Option<Viewport>,
    pub(crate) scissor_rect: Option<ScissorRect>,
    pub(crate) draw: RenderDrawExecution,
}

#[cfg(feature = "tiled")]
#[derive(Debug, Default)]
struct SubpassDrawState {
    render_pipeline: Option<Arc<RenderPipeline>>,
    bind_groups: std::collections::BTreeMap<u32, BoundBindGroup>,
    vertex_buffers: std::collections::BTreeMap<u32, BoundVertexBuffer>,
    index_buffer: Option<BoundIndexBuffer>,
    viewport: Option<Viewport>,
    scissor_rect: Option<ScissorRect>,
    draws: Vec<SubpassDrawExecution>,
}

/// Stores subpass render pass state.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub struct SubpassRenderPass {
    inner: Arc<SubpassRenderPassInner>,
}

/// Holds subpass render pass shared state.
#[cfg(feature = "tiled")]
#[derive(Debug)]
pub(crate) struct SubpassRenderPassInner {
    encoder: CommandEncoder,
    token: PassToken,
    layout: Arc<SubpassPassLayout>,
    extent: Extent3d,
    color_attachments: Vec<SubpassColorAttachmentBinding>,
    depth_stencil_attachment: Option<SubpassDepthStencilAttachmentBinding>,
    hal: Mutex<Option<HalSubpassRenderPass>>,
    active_subpass: Mutex<u32>,
    draw_state: Mutex<SubpassDrawState>,
    ended: Mutex<bool>,
    is_error: bool,
}

#[cfg(feature = "tiled")]
impl SubpassRenderPass {
    /// Creates a new subpass render pass.
    pub(crate) fn new(
        encoder: CommandEncoder,
        token: PassToken,
        descriptor: SubpassRenderPassDescriptor,
        hal: Option<HalSubpassRenderPass>,
        is_error: bool,
    ) -> Self {
        Self {
            inner: Arc::new(SubpassRenderPassInner {
                encoder,
                token,
                layout: Arc::clone(&descriptor.pass_layout),
                extent: descriptor.extent,
                color_attachments: descriptor.color_attachments,
                depth_stencil_attachment: descriptor.depth_stencil_attachment,
                hal: Mutex::new(hal),
                active_subpass: Mutex::new(0),
                draw_state: Mutex::new(SubpassDrawState::default()),
                ended: Mutex::new(false),
                is_error,
            }),
        }
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns the active subpass index.
    #[must_use]
    pub fn active_subpass(&self) -> u32 {
        *self.inner.active_subpass.lock()
    }

    /// Returns retained color attachment count.
    #[must_use]
    pub fn color_attachment_count(&self) -> usize {
        self.inner.color_attachments.len()
    }

    /// Returns true when a depth-stencil attachment is retained.
    #[must_use]
    pub fn has_depth_stencil_attachment(&self) -> bool {
        self.inner.depth_stencil_attachment.is_some()
    }

    /// Advances to the next subpass.
    pub fn next_subpass(&self) -> Option<String> {
        if *self.inner.ended.lock() {
            return Some("subpass render pass has already ended".to_owned());
        }
        if self.is_error() {
            return None;
        }
        let subpass_count = self.inner.layout.descriptor().subpasses.len() as u32;
        let mut active = self.inner.active_subpass.lock();
        if active.saturating_add(1) >= subpass_count {
            self.inner
                .encoder
                .record_first_error("subpass render pass cannot advance past the last subpass");
            return None;
        }
        if let Some(pass) = self.inner.hal.lock().as_mut() {
            if let Err(error) = pass.next_subpass() {
                let message = error.to_string();
                self.inner.encoder.record_first_error(message.clone());
                return Some(message);
            }
        }
        *active += 1;
        let mut draw_state = self.inner.draw_state.lock();
        draw_state.render_pipeline = None;
        draw_state.bind_groups.clear();
        draw_state.vertex_buffers.clear();
        draw_state.index_buffer = None;
        None
    }

    /// Ends the subpass render pass.
    pub fn end(&self) -> Option<String> {
        self.finish_pass(false)
    }

    /// Sets pipeline on this subpass render pass.
    pub fn set_pipeline(&self, pipeline: Arc<RenderPipeline>) -> Option<String> {
        self.record_draw_command(|state, active_subpass| {
            validate_subpass_pipeline_compatible(&self.inner.layout, active_subpass, &pipeline)?;
            state.render_pipeline = Some(pipeline);
            Ok(())
        })
    }

    /// Records a validation error against the subpass render pass.
    pub fn record_validation_error(&self, message: impl Into<String>) -> Option<String> {
        self.record_draw_command(|_, _| Err(message.into()))
    }

    /// Sets or clears a bind group.
    pub fn set_bind_group(
        &self,
        index: u32,
        group: Option<Arc<BindGroup>>,
        dynamic_offsets: Vec<u32>,
        limits: Limits,
    ) -> Option<String> {
        self.record_draw_command(|state, _| {
            validate_set_bind_group(index, group.as_deref(), &dynamic_offsets, limits)?;
            if let Some(group) = group {
                self.inner
                    .encoder
                    .record_referenced_buffers(bind_group_buffer_resources(&group));
                self.inner
                    .encoder
                    .record_referenced_textures(bind_group_texture_resources(&group));
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

    /// Sets or clears a vertex buffer.
    pub fn set_vertex_buffer(
        &self,
        slot: u32,
        buffer: Option<Arc<Buffer>>,
        offset: u64,
        size: u64,
        limits: Limits,
    ) -> Option<String> {
        self.record_draw_command(|state, _| {
            validate_vertex_buffer_slot(slot, limits)?;
            if let Some(buffer) = buffer {
                let size = validate_set_vertex_buffer(&buffer, offset, size)?;
                self.inner
                    .encoder
                    .record_referenced_buffer(Arc::clone(&buffer));
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

    /// Sets the index buffer.
    pub fn set_index_buffer(
        &self,
        buffer: Arc<Buffer>,
        format: Option<IndexFormat>,
        offset: u64,
        size: u64,
    ) -> Option<String> {
        self.record_draw_command(|state, _| {
            let format =
                format.ok_or_else(|| "subpass render pass index format is invalid".to_owned())?;
            let size = validate_set_index_buffer(&buffer, format, offset, size)?;
            self.inner
                .encoder
                .record_referenced_buffer(Arc::clone(&buffer));
            state.index_buffer = Some(BoundIndexBuffer {
                buffer,
                format,
                offset,
                size,
            });
            Ok(())
        })
    }

    /// Records a non-indexed draw.
    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
        limits: Limits,
    ) -> Option<String> {
        self.record_subpass_draw(
            RenderDrawKind::Direct {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            },
            RenderDrawExecution::Direct {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            },
            limits,
        )
    }

    /// Records an indexed draw.
    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
        limits: Limits,
    ) -> Option<String> {
        self.record_subpass_draw(
            RenderDrawKind::IndexedDirect {
                index_count,
                instance_count,
                first_index,
                first_instance,
            },
            RenderDrawExecution::Indexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            },
            limits,
        )
    }

    /// Sets viewport on this subpass render pass.
    pub fn set_viewport(
        &self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Option<String> {
        self.record_draw_command(|state, _| {
            validate_viewport(x, y, width, height, min_depth, max_depth)?;
            validate_viewport_bounds(x, y, width, height, self.inner.encoder.inner.limits)?;
            state.viewport = Some(Viewport {
                x,
                y,
                width,
                height,
                min_depth,
                max_depth,
            });
            Ok(())
        })
    }

    /// Sets scissor rect on this subpass render pass.
    pub fn set_scissor_rect(&self, x: u32, y: u32, width: u32, height: u32) -> Option<String> {
        self.record_draw_command(|state, _| {
            x.checked_add(width).ok_or_else(|| {
                "subpass render pass scissor rectangle width overflows".to_owned()
            })?;
            y.checked_add(height).ok_or_else(|| {
                "subpass render pass scissor rectangle height overflows".to_owned()
            })?;
            validate_scissor_rect(Some(self.inner.extent), x, y, width, height)?;
            state.scissor_rect = Some(ScissorRect {
                x,
                y,
                width,
                height,
            });
            Ok(())
        })
    }

    fn finish_pass(&self, from_drop: bool) -> Option<String> {
        let mut ended = self.inner.ended.lock();
        if *ended {
            return None;
        }
        *ended = true;
        if let Some(pass) = self.inner.hal.lock().take() {
            if let Err(error) = pass.end() {
                let message = error.to_string();
                self.inner.encoder.record_first_error(message.clone());
                self.inner.encoder.end_pass(self.inner.token);
                return (!from_drop).then_some(message);
            }
        }
        if !self.is_error() {
            self.inner
                .encoder
                .record_subpass_render_pass(SubpassRenderPassCommand {
                    layout: Arc::clone(&self.inner.layout),
                    extent: self.inner.extent,
                    color_attachments: self.inner.color_attachments.clone(),
                    depth_stencil_attachment: self.inner.depth_stencil_attachment.clone(),
                    draws: self.inner.draw_state.lock().draws.clone(),
                });
        }
        self.inner.encoder.end_pass(self.inner.token);
        None
    }

    fn record_subpass_draw(
        &self,
        kind: RenderDrawKind,
        draw: RenderDrawExecution,
        limits: Limits,
    ) -> Option<String> {
        self.record_draw_command(|state, active_subpass| {
            let pipeline =
                Arc::clone(state.render_pipeline.as_ref().ok_or_else(|| {
                    "subpass render pass draw requires a render pipeline".to_owned()
                })?);
            validate_subpass_pipeline_compatible(&self.inner.layout, active_subpass, &pipeline)?;
            validate_subpass_draw_state(&pipeline, state, kind, limits)?;
            state.draws.push(SubpassDrawExecution {
                subpass_index: active_subpass,
                pipeline,
                bind_groups: state.bind_groups.clone(),
                vertex_buffers: state.vertex_buffers.clone(),
                index_buffer: state.index_buffer.clone(),
                viewport: state.viewport,
                scissor_rect: state.scissor_rect,
                draw,
            });
            Ok(())
        })
    }

    fn record_draw_command<F>(&self, command: F) -> Option<String>
    where
        F: FnOnce(&mut SubpassDrawState, u32) -> Result<(), String>,
    {
        if *self.inner.ended.lock() {
            let message = "subpass render pass has already ended".to_owned();
            self.inner.encoder.record_first_error(message.clone());
            return Some(message);
        }
        if self.inner.encoder.is_finished() {
            let message =
                "subpass render pass cannot be used after parent encoder finish".to_owned();
            self.inner.encoder.record_first_error(message.clone());
            return Some(message);
        }
        if !self.inner.encoder.is_open_pass(self.inner.token) {
            let message = "subpass render pass is not the active pass".to_owned();
            self.inner.encoder.record_first_error(message.clone());
            return Some(message);
        }
        if self.is_error() {
            return None;
        }
        let active_subpass = self.active_subpass();
        let mut state = self.inner.draw_state.lock();
        if let Err(message) = command(&mut state, active_subpass) {
            self.inner.encoder.record_first_error(message);
        }
        None
    }
}

#[cfg(feature = "tiled")]
impl Drop for SubpassRenderPassInner {
    fn drop(&mut self) {
        if !*self.ended.lock() {
            let pass = SubpassRenderPass {
                inner: Arc::new(Self {
                    encoder: self.encoder.clone(),
                    token: self.token,
                    layout: Arc::clone(&self.layout),
                    extent: self.extent,
                    color_attachments: self.color_attachments.clone(),
                    depth_stencil_attachment: self.depth_stencil_attachment.clone(),
                    hal: Mutex::new(self.hal.lock().take()),
                    active_subpass: Mutex::new(*self.active_subpass.lock()),
                    draw_state: Mutex::new(std::mem::take(&mut *self.draw_state.lock())),
                    ended: Mutex::new(false),
                    is_error: self.is_error,
                }),
            };
            let _ = pass.finish_pass(true);
        }
    }
}

#[cfg(feature = "tiled")]
fn validate_subpass_draw_state(
    pipeline: &RenderPipeline,
    state: &SubpassDrawState,
    kind: RenderDrawKind,
    limits: Limits,
) -> Result<(), String> {
    validate_subpass_bind_group_and_vertex_limits(state, limits)?;
    validate_pipeline_bind_groups(pipeline.bind_group_layouts(), &state.bind_groups, limits)?;
    for (slot, layout) in pipeline.vertex_buffer_layouts().iter().enumerate() {
        if !layout.used {
            continue;
        }
        let slot = u32::try_from(slot)
            .map_err(|_| "render pipeline vertex buffer slot is too large".to_owned())?;
        if !state.vertex_buffers.contains_key(&slot) {
            return Err(
                "subpass render pass draw requires all declared vertex buffers to be set"
                    .to_owned(),
            );
        }
    }
    if kind.is_indexed() && state.index_buffer.is_none() {
        return Err("subpass render pass indexed draw requires an index buffer".to_owned());
    }
    Ok(())
}

#[cfg(feature = "tiled")]
fn validate_subpass_bind_group_and_vertex_limits(
    state: &SubpassDrawState,
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
            "subpass render pass draw bind group plus vertex buffer count overflows".to_owned()
        })?;
    if total > limits.max_bind_groups_plus_vertex_buffers {
        return Err(
            "subpass render pass draw bind group plus vertex buffer count exceeds the device limit"
                .to_owned(),
        );
    }
    Ok(())
}

/// Validates subpass pipeline compatibility.
#[cfg(feature = "tiled")]
pub(crate) fn validate_subpass_pipeline_compatible(
    pass_layout: &Arc<SubpassPassLayout>,
    subpass_index: u32,
    pipeline: &RenderPipeline,
) -> Result<(), String> {
    if pipeline.is_error() {
        return Err("subpass render pass requires a valid render pipeline".to_owned());
    }
    let Some(compatibility) = pipeline.subpass_compatibility() else {
        return Err("subpass render pass requires a subpass render pipeline".to_owned());
    };
    if !compatibility.pass_layout.same(pass_layout) || compatibility.subpass_index != subpass_index
    {
        return Err("subpass render pipeline is not compatible with the active subpass".to_owned());
    }
    Ok(())
}

/// Validates a subpass pass layout descriptor.
pub(crate) fn validate_subpass_pass_layout_descriptor(
    device: &Device,
    descriptor: &SubpassPassLayoutDescriptor,
) -> Option<String> {
    if let Some(error) = &descriptor.error {
        return Some(error.clone());
    }
    let caps = tiled_capabilities_for_device(device);
    validate_subpass_pass_layout_descriptor_with_caps(descriptor, caps)
}

fn tiled_capabilities_for_device(device: &Device) -> TiledCapabilities {
    if !tiled_features_supported(device.hal().backend()) {
        return TiledCapabilities {
            max_subpasses: 0,
            max_subpass_color_attachments: 0,
            max_input_attachments: 0,
            estimated_tile_memory_bytes: 0,
        };
    }
    let limits = device.limits();
    TiledCapabilities {
        max_subpasses: 4,
        max_subpass_color_attachments: limits.max_color_attachments,
        max_input_attachments: limits.max_color_attachments,
        estimated_tile_memory_bytes: 256 * 1024,
    }
}

fn validate_subpass_pass_layout_descriptor_with_caps(
    descriptor: &SubpassPassLayoutDescriptor,
    caps: TiledCapabilities,
) -> Option<String> {
    let enforce_caps = caps.max_subpasses != 0
        || caps.max_subpass_color_attachments != 0
        || caps.max_input_attachments != 0;
    if descriptor.subpasses.is_empty() {
        return Some("subpass pass layout requires at least one subpass".to_owned());
    }
    // Noop advertises zero tiled capabilities but still accepts subpass objects
    // so validation and lifecycle tests remain GPU-independent.
    if enforce_caps && descriptor.subpasses.len() > caps.max_subpasses as usize {
        return Some("subpass count exceeds tiled capabilities".to_owned());
    }
    for (subpass_index, subpass) in descriptor.subpasses.iter().enumerate() {
        if enforce_caps
            && subpass.color_attachment_indices.len() > caps.max_subpass_color_attachments as usize
        {
            return Some("subpass color attachment count exceeds tiled capabilities".to_owned());
        }
        if enforce_caps && subpass.input_attachments.len() > caps.max_input_attachments as usize {
            return Some("subpass input attachment count exceeds tiled capabilities".to_owned());
        }
        for &color_index in &subpass.color_attachment_indices {
            if color_index as usize >= descriptor.color_attachments.len() {
                return Some("subpass color attachment index is out of range".to_owned());
            }
        }
        // C-3 — every attachment written by a subpass (its color attachments plus
        // the depth-stencil, when used) must share a single sample count.
        let mut written_sample_counts = subpass
            .color_attachment_indices
            .iter()
            .filter_map(|&color_index| descriptor.color_attachments.get(color_index as usize))
            .map(|attachment| attachment.sample_count)
            .chain(
                subpass
                    .uses_depth_stencil
                    .then_some(descriptor.depth_stencil_attachment.as_ref())
                    .flatten()
                    .map(|attachment| attachment.sample_count),
            );
        if let Some(first) = written_sample_counts.next() {
            if written_sample_counts.any(|count| count != first) {
                return Some("subpass attachments must share a single sample count".to_owned());
            }
        }
        for input in &subpass.input_attachments {
            if input.source_subpass >= subpass_index as u32 {
                return Some(
                    "subpass input sourceSubpass must refer to a prior subpass".to_owned(),
                );
            }
            if input.source_attachment == DEPTH_STENCIL_ATTACHMENT_INDEX {
                if descriptor.depth_stencil_attachment.is_none() {
                    return Some(
                        "subpass input depth source requires a depth-stencil attachment".to_owned(),
                    );
                }
            } else if input.source_attachment as usize >= descriptor.color_attachments.len() {
                return Some("subpass input sourceAttachment is out of range".to_owned());
            }
        }
    }
    for dependency in &descriptor.dependencies {
        if dependency.src_subpass as usize >= descriptor.subpasses.len()
            || dependency.dst_subpass as usize >= descriptor.subpasses.len()
        {
            return Some("subpass dependency index is out of range".to_owned());
        }
    }
    None
}

/// Validates a subpass render pass descriptor.
#[cfg(feature = "tiled")]
pub(crate) fn validate_subpass_render_pass_descriptor(
    descriptor: &SubpassRenderPassDescriptor,
    features: &crate::device::FeatureSet,
) -> Option<String> {
    if let Some(error) = &descriptor.error {
        return Some(error.clone());
    }
    if descriptor.pass_layout.is_error() {
        return Some("subpass render pass layout must not be an error layout".to_owned());
    }
    let layout = descriptor.pass_layout.descriptor();
    if descriptor.extent.width == 0
        || descriptor.extent.height == 0
        || descriptor.extent.depth_or_array_layers != 1
    {
        return Some("subpass render pass extent must be non-zero 2D".to_owned());
    }
    if descriptor.color_attachments.len() != layout.color_attachments.len() {
        return Some("subpass render pass color attachment count must match layout".to_owned());
    }
    if descriptor.depth_stencil_attachment.is_some() != layout.depth_stencil_attachment.is_some() {
        return Some("subpass render pass depth-stencil attachment must match layout".to_owned());
    }
    for subpass in &layout.subpasses {
        for &index in &subpass.color_attachment_indices {
            if descriptor.color_attachments.get(index as usize).is_none() {
                return Some("subpass render pass missing used color attachment".to_owned());
            }
        }
    }
    for (index, attachment) in descriptor.color_attachments.iter().enumerate() {
        if let Err(message) = validate_subpass_color_attachment(
            attachment,
            layout.color_attachments.get(index),
            descriptor.extent,
            features,
        ) {
            return Some(message);
        }
    }
    if let Some(attachment) = &descriptor.depth_stencil_attachment {
        if let Err(message) = validate_subpass_depth_stencil_attachment(
            attachment,
            layout.depth_stencil_attachment.as_ref(),
            descriptor.extent,
            features,
        ) {
            return Some(message);
        }
    }
    None
}

/// Resolves retained attachment resources for a subpass render pass.
#[cfg(feature = "tiled")]
pub(crate) fn resolve_subpass_render_pass_resources(
    device: &Device,
    descriptor: &SubpassRenderPassDescriptor,
) -> Result<(), String> {
    if !tiled_features_supported(device.hal().backend())
        && !matches!(device.hal().backend(), yawgpu_hal::HalBackend::Noop)
    {
        return Err("subpass render pass backend is unsupported".to_owned());
    }
    for attachment in &descriptor.color_attachments {
        validate_subpass_resource(&attachment.resource)?;
    }
    if let Some(attachment) = &descriptor.depth_stencil_attachment {
        validate_subpass_resource(&attachment.resource)?;
    }
    Ok(())
}

#[cfg(feature = "tiled")]
fn validate_subpass_color_attachment(
    attachment: &SubpassColorAttachmentBinding,
    layout: Option<&AttachmentLayout>,
    extent: Extent3d,
    features: &crate::device::FeatureSet,
) -> Result<(), String> {
    let Some(layout) = layout else {
        return Err("subpass render pass color attachment count must match layout".to_owned());
    };
    let SubpassAttachmentResource::Persistent {
        view,
        resolve_target,
    } = &attachment.resource;
    let Some(format_caps) = view.format().caps(features) else {
        return Err("subpass render pass color attachment format must be supported".to_owned());
    };
    if !view.usage().contains(TextureUsage::RENDER_ATTACHMENT) {
        return Err(
            "subpass render pass color attachment requires RenderAttachment usage".to_owned(),
        );
    }
    if !format_caps.aspects.color || !format_caps.renderable {
        return Err(
            "subpass render pass color attachment format must be color-renderable".to_owned(),
        );
    }
    if view.format() != layout.format || view.texture().sample_count() != layout.sample_count {
        return Err("subpass render pass color attachment must match layout".to_owned());
    }
    validate_subpass_attachment_view(view, extent, "subpass render pass color attachment")?;
    if attachment.load_op == LoadOp::Undefined {
        return Err("subpass render pass color attachment loadOp must be set".to_owned());
    }
    if attachment.store_op == StoreOp::Undefined {
        return Err("subpass render pass color attachment storeOp must be set".to_owned());
    }
    if attachment.load_op == LoadOp::Clear
        && ![
            attachment.clear_value.r,
            attachment.clear_value.g,
            attachment.clear_value.b,
            attachment.clear_value.a,
        ]
        .into_iter()
        .all(f64::is_finite)
    {
        return Err("subpass render pass color clearValue components must be finite".to_owned());
    }
    if let Some(resolve_target) = resolve_target {
        validate_resolve_target(view, resolve_target)
            .map_err(|message| message.replace("render pass", "subpass render pass"))?;
    }
    Ok(())
}

#[cfg(feature = "tiled")]
fn validate_subpass_depth_stencil_attachment(
    attachment: &SubpassDepthStencilAttachmentBinding,
    layout: Option<&AttachmentLayout>,
    extent: Extent3d,
    features: &crate::device::FeatureSet,
) -> Result<(), String> {
    let Some(layout) = layout else {
        return Err("subpass render pass depth-stencil attachment must match layout".to_owned());
    };
    let SubpassAttachmentResource::Persistent { view, .. } = &attachment.resource;
    let Some(format_caps) = view.format().caps(features) else {
        return Err(
            "subpass render pass depth-stencil attachment format must be supported".to_owned(),
        );
    };
    if !view.usage().contains(TextureUsage::RENDER_ATTACHMENT) {
        return Err(
            "subpass render pass depth-stencil attachment requires RenderAttachment usage"
                .to_owned(),
        );
    }
    if !format_caps.aspects.depth && !format_caps.aspects.stencil {
        return Err(
            "subpass render pass depth-stencil attachment format must have depth or stencil aspect"
                .to_owned(),
        );
    }
    if view.format() != layout.format || view.texture().sample_count() != layout.sample_count {
        return Err("subpass render pass depth-stencil attachment must match layout".to_owned());
    }
    validate_subpass_attachment_view(view, extent, "subpass render pass depth-stencil attachment")?;
    if format_caps.aspects.depth {
        if attachment.depth_load_op == LoadOp::Undefined {
            return Err("subpass render pass depth loadOp must be set".to_owned());
        }
        if attachment.depth_store_op == StoreOp::Undefined {
            return Err("subpass render pass depth storeOp must be set".to_owned());
        }
        if attachment.depth_load_op == LoadOp::Clear
            && (!attachment.depth_clear_value.is_finite()
                || !(0.0..=1.0).contains(&attachment.depth_clear_value))
        {
            return Err(
                "subpass render pass depth clear value must be finite and in [0, 1]".to_owned(),
            );
        }
    } else if attachment.depth_load_op != LoadOp::Undefined
        || attachment.depth_store_op != StoreOp::Undefined
    {
        return Err(
            "subpass render pass non-depth attachment must not set depth load/store ops".to_owned(),
        );
    }
    if format_caps.aspects.stencil {
        if attachment.stencil_load_op == LoadOp::Undefined {
            return Err("subpass render pass stencil loadOp must be set".to_owned());
        }
        if attachment.stencil_store_op == StoreOp::Undefined {
            return Err("subpass render pass stencil storeOp must be set".to_owned());
        }
    } else if attachment.stencil_load_op != LoadOp::Undefined
        || attachment.stencil_store_op != StoreOp::Undefined
    {
        return Err(
            "subpass render pass non-stencil attachment must not set stencil load/store ops"
                .to_owned(),
        );
    }
    Ok(())
}

#[cfg(feature = "tiled")]
fn validate_subpass_attachment_view(
    view: &TextureView,
    extent: Extent3d,
    label: &str,
) -> Result<(), String> {
    if view.is_error() {
        return Err(format!("{label} view must not be an error view"));
    }
    if view.dimension() != TextureViewDimension::D2 {
        return Err(format!("{label} view dimension must be 2D"));
    }
    if view.texture().dimension() != TextureDimension::D2 {
        return Err(format!("{label} texture dimension must be 2D"));
    }
    if view.array_layer_count() != 1 {
        return Err(format!("{label} view arrayLayerCount must be one"));
    }
    if view.mip_level_count() != 1 {
        return Err(format!("{label} view mipLevelCount must be one"));
    }
    let view_extent = view.render_extent();
    if view_extent.width != extent.width || view_extent.height != extent.height {
        return Err("subpass render pass attachments must match extent".to_owned());
    }
    Ok(())
}

#[cfg(feature = "tiled")]
fn validate_subpass_resource(resource: &SubpassAttachmentResource) -> Result<(), String> {
    let SubpassAttachmentResource::Persistent {
        view,
        resolve_target,
    } = resource;
    if view.is_error() {
        return Err("subpass render pass cannot use an error texture view".to_owned());
    }
    if let Some(resolve_target) = resolve_target {
        if resolve_target.is_error() {
            return Err("subpass render pass cannot use an error resolve target".to_owned());
        }
    }
    Ok(())
}

impl Device {
    /// Creates a subpass pass layout.
    #[must_use]
    pub fn create_subpass_pass_layout(
        &self,
        descriptor: SubpassPassLayoutDescriptor,
    ) -> SubpassPassLayout {
        if self.is_lost() {
            return SubpassPassLayout::new(descriptor, true);
        }
        let error = validate_subpass_pass_layout_descriptor(self, &descriptor);
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        SubpassPassLayout::new(descriptor, is_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{
        noop_device, render_pipeline_descriptor, render_shader_module, rgba8_unorm,
    };
    use crate::*;

    fn attachment_layout() -> AttachmentLayout {
        AttachmentLayout {
            format: rgba8_unorm(),
            sample_count: 1,
        }
    }

    fn depth_attachment_layout() -> AttachmentLayout {
        AttachmentLayout {
            format: TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS),
            sample_count: 1,
        }
    }

    fn caps() -> TiledCapabilities {
        TiledCapabilities {
            max_subpasses: 4,
            max_subpass_color_attachments: 4,
            max_input_attachments: 4,
            estimated_tile_memory_bytes: 256 * 1024,
        }
    }

    fn zero_caps() -> TiledCapabilities {
        TiledCapabilities {
            max_subpasses: 0,
            max_subpass_color_attachments: 0,
            max_input_attachments: 0,
            estimated_tile_memory_bytes: 0,
        }
    }

    fn valid_two_subpass_deferred_layout() -> SubpassPassLayoutDescriptor {
        SubpassPassLayoutDescriptor {
            color_attachments: vec![attachment_layout(), attachment_layout()],
            depth_stencil_attachment: None,
            subpasses: vec![
                SubpassLayoutDesc {
                    color_attachment_indices: vec![0],
                    uses_depth_stencil: false,
                    input_attachments: Vec::new(),
                },
                SubpassLayoutDesc {
                    color_attachment_indices: vec![1],
                    uses_depth_stencil: false,
                    input_attachments: vec![SubpassInputAttachment {
                        group: 0,
                        binding: 0,
                        source_subpass: 0,
                        source_attachment: 0,
                    }],
                },
            ],
            dependencies: vec![SubpassDependency {
                src_subpass: 0,
                dst_subpass: 1,
                dependency_type: SubpassDependencyType::ColorToInput,
                by_region: true,
            }],
            error: None,
        }
    }

    fn render_attachment_view(device: &Device) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT,
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

    fn persistent_color(device: &Device) -> SubpassColorAttachmentBinding {
        SubpassColorAttachmentBinding {
            resource: SubpassAttachmentResource::Persistent {
                view: render_attachment_view(device),
                resolve_target: None,
            },
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            clear_value: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        }
    }

    fn subpass_descriptor(
        layout: Arc<SubpassPassLayout>,
        device: &Device,
    ) -> SubpassRenderPassDescriptor {
        SubpassRenderPassDescriptor {
            pass_layout: layout,
            extent: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            color_attachments: vec![persistent_color(device), persistent_color(device)],
            depth_stencil_attachment: None,
            error: None,
        }
    }

    fn subpass_pipeline(
        device: &Device,
        layout: Arc<SubpassPassLayout>,
        subpass_index: u32,
    ) -> Arc<RenderPipeline> {
        Arc::new(
            device.create_subpass_render_pipeline(SubpassRenderPipelineDescriptor {
                base: render_pipeline_descriptor(render_shader_module(device)),
                pass_layout: layout,
                subpass_index,
                error: None,
            }),
        )
    }

    #[test]
    fn subpass_pass_layout_rejects_empty_subpasses() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses.clear();

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass pass layout requires at least one subpass".to_owned())
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_pass_layout_rejects_mixed_sample_counts_within_subpass() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        // The first subpass now writes two attachments with differing sample
        // counts, which violates C-3.
        descriptor.color_attachments[1].sample_count = 4;
        descriptor.subpasses[0].color_attachment_indices = vec![0, 1];

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass attachments must share a single sample count".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_color_attachment_index_out_of_range() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses[0].color_attachment_indices = vec![2];

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass color attachment index is out of range".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_input_source_attachment_out_of_range() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses[1].input_attachments[0].source_attachment = 2;

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass input sourceAttachment is out of range".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_input_source_subpass_that_is_not_prior() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses[1].input_attachments[0].source_subpass = 1;

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass input sourceSubpass must refer to a prior subpass".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_depth_input_without_depth_stencil_attachment() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses[1].input_attachments[0].source_attachment =
            DEPTH_STENCIL_ATTACHMENT_INDEX;

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass input depth source requires a depth-stencil attachment".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_accepts_depth_input_with_depth_stencil_attachment() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.depth_stencil_attachment = Some(depth_attachment_layout());
        descriptor.subpasses[0].uses_depth_stencil = true;
        descriptor.subpasses[1].input_attachments[0].source_attachment =
            DEPTH_STENCIL_ATTACHMENT_INDEX;
        descriptor.dependencies[0].dependency_type = SubpassDependencyType::DepthToInput;

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            None
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_dependency_index_out_of_range() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.dependencies[0].dst_subpass = 2;

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(&descriptor, caps()),
            Some("subpass dependency index is out of range".to_owned())
        );
    }

    #[test]
    fn subpass_pass_layout_accepts_valid_two_subpass_deferred_layout() {
        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(
                &valid_two_subpass_deferred_layout(),
                caps(),
            ),
            None
        );
    }

    #[test]
    fn subpass_pass_layout_zero_caps_still_validate_well_formed_layout() {
        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(
                &valid_two_subpass_deferred_layout(),
                zero_caps(),
            ),
            None
        );

        let device = noop_device();
        assert_eq!(
            validate_subpass_pass_layout_descriptor(&device, &valid_two_subpass_deferred_layout()),
            None
        );
    }

    #[test]
    fn create_subpass_pass_layout_returns_error_layout_and_sinks_validation_message() {
        let device = noop_device();

        device.push_error_scope(crate::error::ErrorFilter::Validation);
        let valid = device.create_subpass_pass_layout(valid_two_subpass_deferred_layout());
        let scoped = device.pop_error_scope().expect("scope should exist");

        assert!(!valid.is_error());
        assert_eq!(scoped, None);

        let mut invalid = valid_two_subpass_deferred_layout();
        invalid.subpasses.clear();
        device.push_error_scope(crate::error::ErrorFilter::Validation);
        let error_layout = device.create_subpass_pass_layout(invalid);
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid layout should be scoped");

        assert!(error_layout.is_error());
        assert_eq!(
            error.message,
            "subpass pass layout requires at least one subpass"
        );
    }

    #[test]
    fn subpass_pass_layout_rejects_capability_overflow_when_caps_are_enforced() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses.push(SubpassLayoutDesc {
            color_attachment_indices: Vec::new(),
            uses_depth_stencil: false,
            input_attachments: Vec::new(),
        });

        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(
                &descriptor,
                TiledCapabilities {
                    max_subpasses: 2,
                    max_subpass_color_attachments: 4,
                    max_input_attachments: 4,
                    estimated_tile_memory_bytes: 0,
                },
            ),
            Some("subpass count exceeds tiled capabilities".to_owned())
        );
    }

    #[test]
    fn compute_subpass_color_slots_maps_input_binding_to_source_attachment_one() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.subpasses[1].input_attachments[0].source_attachment = 1;

        assert_eq!(
            compute_subpass_color_slots(&descriptor, 1),
            vec![((0, 0), 1)]
        );
    }

    #[test]
    fn compute_subpass_color_slots_maps_input_binding_to_source_attachment_two() {
        let mut descriptor = valid_two_subpass_deferred_layout();
        descriptor.color_attachments.push(attachment_layout());
        descriptor.subpasses[1].input_attachments[0].source_attachment = 2;

        assert_eq!(
            compute_subpass_color_slots(&descriptor, 1),
            vec![((0, 0), 2)]
        );
    }

    #[test]
    fn compute_subpass_color_slots_returns_empty_for_out_of_range_subpass() {
        let descriptor = valid_two_subpass_deferred_layout();

        assert_eq!(compute_subpass_color_slots(&descriptor, 2), Vec::new());
    }

    #[test]
    fn subpass_render_pass_noop_lifecycle_records_and_submits() {
        let device = noop_device();
        let layout =
            Arc::new(device.create_subpass_pass_layout(valid_two_subpass_deferred_layout()));
        let pipeline0 = subpass_pipeline(&device, Arc::clone(&layout), 0);
        let pipeline1 = subpass_pipeline(&device, Arc::clone(&layout), 1);
        assert!(!pipeline0.is_error());
        assert!(!pipeline1.is_error());

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder
            .begin_subpass_render_pass(&device, subpass_descriptor(Arc::clone(&layout), &device));
        assert_eq!(begin_error, None);
        assert!(!pass.is_error());
        assert_eq!(pass.set_pipeline(pipeline0), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.next_subpass(), None);
        assert_eq!(pass.set_pipeline(pipeline1), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, finish_error) = encoder.finish();
        assert_eq!(finish_error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(device.queue().submit(&[Arc::new(command_buffer)]), None);
    }

    #[test]
    fn subpass_render_pass_next_subpass_past_last_records_error() {
        let device = noop_device();
        let layout =
            Arc::new(device.create_subpass_pass_layout(valid_two_subpass_deferred_layout()));
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_subpass_render_pass(&device, subpass_descriptor(layout, &device));
        assert_eq!(begin_error, None);
        assert_eq!(pass.next_subpass(), None);
        assert_eq!(pass.next_subpass(), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, finish_error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            finish_error,
            Some("subpass render pass cannot advance past the last subpass".to_owned())
        );
    }

    #[test]
    fn subpass_render_pass_requires_first_encoder_operation() {
        let device = noop_device();
        let layout =
            Arc::new(device.create_subpass_pass_layout(valid_two_subpass_deferred_layout()));
        let encoder = device.create_command_encoder();
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        }));
        assert_eq!(
            encoder.copy_buffer_to_buffer(Arc::clone(&buffer), 0, Arc::clone(&buffer), 0, 0),
            None
        );
        let (pass, begin_error) =
            encoder.begin_subpass_render_pass(&device, subpass_descriptor(layout, &device));
        assert_eq!(begin_error, None);
        assert!(pass.is_error());
        drop(pass);
        let (command_buffer, finish_error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            finish_error,
            Some("subpass render pass must be the first command encoder operation".to_owned())
        );
    }

    #[test]
    fn subpass_render_pass_rejects_incompatible_subpass_pipeline() {
        let device = noop_device();
        let layout =
            Arc::new(device.create_subpass_pass_layout(valid_two_subpass_deferred_layout()));
        let wrong_pipeline = subpass_pipeline(&device, Arc::clone(&layout), 1);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_subpass_render_pass(&device, subpass_descriptor(layout, &device));
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(wrong_pipeline), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, finish_error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            finish_error,
            Some("subpass render pipeline is not compatible with the active subpass".to_owned())
        );
    }

    #[test]
    fn subpass_render_pass_descriptor_validation_records_error_pass() {
        let device = noop_device();
        let layout =
            Arc::new(device.create_subpass_pass_layout(valid_two_subpass_deferred_layout()));

        for descriptor in [
            SubpassRenderPassDescriptor {
                extent: Extent3d {
                    width: 0,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                ..subpass_descriptor(Arc::clone(&layout), &device)
            },
            SubpassRenderPassDescriptor {
                color_attachments: vec![persistent_color(&device)],
                ..subpass_descriptor(Arc::clone(&layout), &device)
            },
            SubpassRenderPassDescriptor {
                depth_stencil_attachment: Some(SubpassDepthStencilAttachmentBinding {
                    resource: SubpassAttachmentResource::Persistent {
                        view: render_attachment_view(&device),
                        resolve_target: None,
                    },
                    depth_load_op: LoadOp::Clear,
                    depth_store_op: StoreOp::Store,
                    depth_clear_value: 1.0,
                    stencil_load_op: LoadOp::Undefined,
                    stencil_store_op: StoreOp::Undefined,
                    stencil_clear_value: 0,
                }),
                ..subpass_descriptor(Arc::clone(&layout), &device)
            },
        ] {
            let encoder = device.create_command_encoder();
            let (pass, begin_error) = encoder.begin_subpass_render_pass(&device, descriptor);
            assert_eq!(begin_error, None);
            assert!(pass.is_error());
            drop(pass);
            let (command_buffer, finish_error) = encoder.finish();
            assert!(command_buffer.is_error());
            assert!(finish_error.is_some());
        }
    }
}
