use std::sync::Arc;

use parking_lot::Mutex;

use yawgpu_hal::{HalDevice, HalSubpassRenderPass};
#[cfg(feature = "tiled")]
use yawgpu_hal::{
    HalSubpassAttachmentLayout, HalSubpassDependency, HalSubpassDependencyType,
    HalSubpassInputAttachment, HalSubpassLayout, HalSubpassPassLayout,
};

use crate::adapter::{tiled_features_supported, TiledCapabilities};
use crate::bind_group::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::copy::*;
use crate::device::{Device, FeatureSet};
use crate::error::ErrorKind;
use crate::extent::*;
use crate::format::*;
use crate::limits::*;
use crate::pass::*;
use crate::render_pipeline::*;
#[cfg(feature = "tiled")]
use crate::texture::hal_texture_format;
use crate::texture_view::*;
use crate::transient_attachment::*;

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
#[derive(Debug, Clone)]
pub enum SubpassAttachmentResource {
    /// Persistent texture view.
    Persistent {
        /// View.
        view: Arc<TextureView>,
        /// Resolve target.
        resolve_target: Option<Arc<TextureView>>,
    },
    /// Transient attachment.
    Transient(Arc<TransientAttachment>),
}

/// Describes one color attachment binding.
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
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct SubpassDrawExecution {
    pub(crate) subpass_index: u32,
    pub(crate) pipeline: Arc<RenderPipeline>,
    pub(crate) bind_groups: std::collections::BTreeMap<u32, BoundBindGroup>,
    pub(crate) vertex_buffers: std::collections::BTreeMap<u32, BoundVertexBuffer>,
    pub(crate) index_buffer: Option<BoundIndexBuffer>,
    pub(crate) viewport: Option<Viewport>,
    pub(crate) scissor_rect: Option<ScissorRect>,
    pub(crate) draw: RenderDrawExecution,
}

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
#[derive(Debug, Clone)]
pub struct SubpassRenderPass {
    inner: Arc<SubpassRenderPassInner>,
}

/// Holds subpass render pass shared state.
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
                return Some(message);
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
    ) -> Option<String> {
        self.record_draw_command(|state, _| {
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
        _base_vertex: i32,
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
            RenderDrawExecution::Direct {
                vertex_count: index_count,
                instance_count,
                first_vertex: first_index,
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

    fn record_subpass_draw(
        &self,
        kind: RenderDrawKind,
        draw: RenderDrawExecution,
        limits: Limits,
    ) -> Option<String> {
        self.record_draw_command(|state, active_subpass| {
            let pipeline = state
                .render_pipeline
                .as_ref()
                .ok_or_else(|| "subpass render pass draw requires a render pipeline".to_owned())?;
            validate_subpass_pipeline_compatible(&self.inner.layout, active_subpass, pipeline)?;
            validate_subpass_pipeline_bind_groups(
                pipeline.bind_group_layouts(),
                &state.bind_groups,
                limits,
            )?;
            validate_subpass_vertex_state(pipeline, state, kind)?;
            state.draws.push(SubpassDrawExecution {
                subpass_index: active_subpass,
                pipeline: Arc::clone(pipeline),
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

impl Drop for SubpassRenderPassInner {
    fn drop(&mut self) {
        if !*self.ended.lock() {
            if let Some(pass) = self.hal.lock().take() {
                let _ = pass.end();
            }
            if !self.is_error {
                self.encoder
                    .record_subpass_render_pass(SubpassRenderPassCommand {
                        layout: Arc::clone(&self.layout),
                        extent: self.extent,
                        color_attachments: self.color_attachments.clone(),
                        depth_stencil_attachment: self.depth_stencil_attachment.clone(),
                        draws: self.draw_state.lock().draws.clone(),
                    });
            }
            self.encoder.end_pass(self.token);
        }
    }
}

fn validate_subpass_vertex_state(
    pipeline: &RenderPipeline,
    state: &SubpassDrawState,
    kind: RenderDrawKind,
) -> Result<(), String> {
    if pipeline.is_error() {
        return Err("subpass render pass draw requires a valid render pipeline".to_owned());
    }
    for slot in 0..pipeline.required_vertex_buffer_count() {
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

fn validate_subpass_render_pipeline_descriptor(
    descriptor: &SubpassRenderPipelineDescriptor,
    limits: Limits,
    features: &FeatureSet,
) -> Option<String> {
    if let Some(error) = &descriptor.error {
        return Some(error.clone());
    }
    if let Some(error) = &descriptor.base.error {
        return Some(error.clone());
    }
    if descriptor.pass_layout.is_error() {
        return Some("subpass render pipeline layout must not be an error layout".to_owned());
    }
    let layout = descriptor.pass_layout.descriptor();
    let Some(subpass) = layout.subpasses.get(descriptor.subpass_index as usize) else {
        return Some("subpass render pipeline subpassIndex is out of range".to_owned());
    };
    let fragment_targets = descriptor
        .base
        .fragment
        .as_ref()
        .map_or(&[][..], |fragment| fragment.targets.as_slice());
    if fragment_targets.len() != subpass.color_attachment_indices.len() {
        return Some("subpass render pipeline color target count must match subpass".to_owned());
    }
    for (target, &attachment_index) in fragment_targets
        .iter()
        .zip(subpass.color_attachment_indices.iter())
    {
        let Some(attachment) = layout.color_attachments.get(attachment_index as usize) else {
            return Some(
                "subpass render pipeline color attachment index is out of range".to_owned(),
            );
        };
        if target.format != attachment.format {
            return Some(
                "subpass render pipeline color target format must match layout".to_owned(),
            );
        }
        if descriptor.base.multisample.count != attachment.sample_count {
            return Some("subpass render pipeline sample count must match layout".to_owned());
        }
    }
    match (
        &descriptor.base.depth_stencil,
        &layout.depth_stencil_attachment,
    ) {
        (Some(depth), Some(attachment)) if depth.format != attachment.format => {
            return Some(
                "subpass render pipeline depth-stencil format must match layout".to_owned(),
            );
        }
        (Some(_), None) => {
            return Some(
                "subpass render pipeline depth-stencil target is not in layout".to_owned(),
            );
        }
        _ => {}
    }
    // Reuse `resolve_render_pipeline_descriptor` directly so we can pass the
    // subpass's color-attachment indices (so fragment `@location(N)` is
    // validated against the flat MTL slot index, not the subpass-relative one).
    crate::render_pipeline::resolve_render_pipeline_descriptor(
        &descriptor.base,
        limits,
        features,
        Some(&subpass.color_attachment_indices),
        0,
    )
    .err()
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
pub(crate) fn validate_subpass_render_pass_descriptor(
    descriptor: &SubpassRenderPassDescriptor,
) -> Option<String> {
    if let Some(error) = &descriptor.error {
        return Some(error.clone());
    }
    let layout = descriptor.pass_layout.descriptor();
    if descriptor.pass_layout.is_error() {
        return Some("subpass render pass layout must not be an error layout".to_owned());
    }
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
    None
}

/// Resolves match-target transients and validates all retained attachment resources.
pub(crate) fn resolve_subpass_render_pass_resources(
    _device: &Device,
    hal: &HalDevice,
    descriptor: &SubpassRenderPassDescriptor,
) -> Result<(), String> {
    if !tiled_features_supported(hal.backend())
        && !matches!(hal.backend(), yawgpu_hal::HalBackend::Noop)
    {
        return Err("subpass render pass backend is unsupported".to_owned());
    }
    for attachment in &descriptor.color_attachments {
        resolve_resource(hal, &attachment.resource, descriptor.extent)?;
    }
    if let Some(depth) = &descriptor.depth_stencil_attachment {
        resolve_resource(hal, &depth.resource, descriptor.extent)?;
    }
    Ok(())
}

fn resolve_resource(
    hal: &HalDevice,
    resource: &SubpassAttachmentResource,
    extent: Extent3d,
) -> Result<(), String> {
    match resource {
        SubpassAttachmentResource::Persistent {
            view,
            resolve_target,
        } => {
            if view.is_error() {
                return Err("subpass render pass cannot use an error texture view".to_owned());
            }
            if let Some(resolve_target) = resolve_target {
                if resolve_target.is_error() {
                    return Err("subpass render pass cannot use an error resolve target".to_owned());
                }
            }
        }
        SubpassAttachmentResource::Transient(attachment) => {
            if attachment.is_error() {
                return Err(
                    "subpass render pass cannot use an error transient attachment".to_owned(),
                );
            }
            if matches!(attachment.descriptor().size, TransientSizeMode::MatchTarget) {
                let descriptor = attachment.descriptor();
                attachment.validate_match_target_extent(extent.width, extent.height)?;
                let hal_descriptor =
                    hal_transient_attachment_descriptor(&descriptor, extent.width, extent.height);
                let hal_attachment = hal
                    .create_transient_attachment(&hal_descriptor)
                    .map_err(|error| error.to_string())?;
                attachment.set_hal_for_match_target(hal_attachment);
            }
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

    /// Creates a subpass-compatible render pipeline.
    #[must_use]
    pub fn create_subpass_render_pipeline(
        &self,
        descriptor: SubpassRenderPipelineDescriptor,
    ) -> RenderPipeline {
        if self.is_lost() {
            return RenderPipeline::new_subpass(
                descriptor,
                true,
                self.limits(),
                &self.inner.features,
                None,
            )
            .0;
        }
        let error = validate_subpass_render_pipeline_descriptor(
            &descriptor,
            self.limits(),
            &self.inner.features,
        );
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        let (pipeline, backend_error) = RenderPipeline::new_subpass(
            descriptor,
            is_error,
            self.limits(),
            &self.inner.features,
            Some(self.hal()),
        );
        if let Some(message) = backend_error {
            self.dispatch_error(ErrorKind::Internal, message);
        }
        pipeline
    }

    /// Creates a subpass-compatible render pipeline without routing validation errors.
    #[must_use]
    pub fn create_subpass_render_pipeline_without_error_dispatch(
        &self,
        descriptor: SubpassRenderPipelineDescriptor,
    ) -> RenderPipeline {
        if self.is_lost() {
            return RenderPipeline::new_subpass(
                descriptor,
                true,
                self.limits(),
                &self.inner.features,
                None,
            )
            .0;
        }
        let error = validate_subpass_render_pipeline_descriptor(
            &descriptor,
            self.limits(),
            &self.inner.features,
        );
        RenderPipeline::new_subpass(
            descriptor,
            error.is_some(),
            self.limits(),
            &self.inner.features,
            Some(self.hal()),
        )
        .0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;

    use std::sync::Arc;

    fn attachment_layout() -> AttachmentLayout {
        AttachmentLayout {
            format: rgba8_unorm(),
            sample_count: 1,
        }
    }

    fn two_subpass_layout_descriptor() -> SubpassPassLayoutDescriptor {
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
            clear_value: black(),
        }
    }

    fn black() -> Color {
        Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        }
    }

    fn subpass_pipeline_descriptor(
        device: &Device,
        layout: Arc<SubpassPassLayout>,
        subpass_index: u32,
    ) -> SubpassRenderPipelineDescriptor {
        SubpassRenderPipelineDescriptor {
            base: render_pipeline_descriptor(render_shader_module(device)),
            pass_layout: layout,
            subpass_index,
            error: None,
        }
    }

    #[test]
    fn subpass_pass_layout_validates_inputs_and_counts() {
        let device = noop_device();

        let valid = device.create_subpass_pass_layout(two_subpass_layout_descriptor());
        assert!(!valid.is_error());
        assert_eq!(valid.descriptor().subpasses.len(), 2);

        device.push_error_scope(ErrorFilter::Validation);
        let mut bad_source = two_subpass_layout_descriptor();
        bad_source.subpasses[1].input_attachments[0].source_subpass = 1;
        let layout = device.create_subpass_pass_layout(bad_source);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("bad source should be scoped");
        assert!(layout.is_error());
        assert_eq!(
            scoped.message,
            "subpass input sourceSubpass must refer to a prior subpass"
        );

        let mut too_many = two_subpass_layout_descriptor();
        too_many.subpasses = vec![
            SubpassLayoutDesc {
                color_attachment_indices: Vec::new(),
                uses_depth_stencil: false,
                input_attachments: Vec::new(),
            };
            5
        ];
        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(
                &too_many,
                TiledCapabilities {
                    max_subpasses: 4,
                    max_subpass_color_attachments: 4,
                    max_input_attachments: 4,
                    estimated_tile_memory_bytes: 0,
                },
            ),
            Some("subpass count exceeds tiled capabilities".to_owned())
        );
        assert_eq!(
            validate_subpass_pass_layout_descriptor_with_caps(
                &too_many,
                TiledCapabilities {
                    max_subpasses: 0,
                    max_subpass_color_attachments: 0,
                    max_input_attachments: 0,
                    estimated_tile_memory_bytes: 0,
                },
            ),
            None
        );
    }

    #[test]
    fn subpass_render_pass_requires_first_encoder_operation() {
        let device = noop_device();
        let layout = Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));
        let encoder = device.create_command_encoder();
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        }));
        encoder.copy_buffer_to_buffer(Arc::clone(&buffer), 0, Arc::clone(&buffer), 0, 0);

        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: Arc::clone(&layout),
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device), persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert!(pass.is_error());
        drop(pass);
        let (command_buffer, _) = encoder.finish();
        assert!(command_buffer.is_error());

        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: layout,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device), persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert!(!pass.is_error());
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn subpass_render_pass_validates_attachment_consistency() {
        let device = noop_device();
        let layout = Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));
        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: layout,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert!(pass.is_error());
        drop(pass);
        let (command_buffer, _) = encoder.finish();
        assert!(command_buffer.is_error());
    }

    #[test]
    fn subpass_render_pass_lifecycle_retains_resources_and_resolves_match_target() {
        let device = noop_device();
        let layout = Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));
        let transient = Arc::new(device.create_transient_attachment(
            TransientAttachmentDescriptor {
                format: rgba8_unorm(),
                size: TransientSizeMode::MatchTarget,
                sample_count: 1,
            },
        ));
        assert!(transient.hal().is_none());
        let before_count = Arc::strong_count(&transient);
        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: Arc::clone(&layout),
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![
                    persistent_color(&device),
                    SubpassColorAttachmentBinding {
                        resource: SubpassAttachmentResource::Transient(Arc::clone(&transient)),
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::Discard,
                        clear_value: black(),
                    },
                ],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert!(!pass.is_error());
        assert!(transient.hal().is_some());
        assert!(Arc::strong_count(&transient) > before_count);
        assert_eq!(pass.color_attachment_count(), 2);
        assert!(!pass.has_depth_stencil_attachment());
        assert_eq!(pass.active_subpass(), 0);
        assert_eq!(pass.next_subpass(), None);
        assert_eq!(pass.active_subpass(), 1);
        assert_eq!(pass.end(), None);

        let encoder = device.create_command_encoder();
        let (same_extent_pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: Arc::clone(&layout),
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![
                    persistent_color(&device),
                    SubpassColorAttachmentBinding {
                        resource: SubpassAttachmentResource::Transient(Arc::clone(&transient)),
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::Discard,
                        clear_value: black(),
                    },
                ],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert!(!same_extent_pass.is_error());
        assert_eq!(same_extent_pass.end(), None);

        let encoder = device.create_command_encoder();
        let (different_extent_pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: Arc::clone(&layout),
                extent: Extent3d {
                    width: 8,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![
                    persistent_color(&device),
                    SubpassColorAttachmentBinding {
                        resource: SubpassAttachmentResource::Transient(Arc::clone(&transient)),
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::Discard,
                        clear_value: black(),
                    },
                ],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert!(different_extent_pass.is_error());
        drop(different_extent_pass);
        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some(
                "match-target transient attachment was already resolved with a different extent"
                    .to_owned()
            )
        );

        let encoder = device.create_command_encoder();
        let (drop_pass, _) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: layout,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device), persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        drop(drop_pass);
    }

    #[test]
    fn subpass_render_pipeline_validates_layout_formats_and_subpass_match() {
        let device = noop_device();
        let layout = Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));

        device.push_error_scope(ErrorFilter::Validation);
        let ok = device.create_subpass_render_pipeline(subpass_pipeline_descriptor(
            &device,
            Arc::clone(&layout),
            0,
        ));
        assert!(!ok.is_error());
        assert_eq!(device.pop_error_scope().expect("scope should exist"), None);

        device.push_error_scope(ErrorFilter::Validation);
        let mut mismatch = subpass_pipeline_descriptor(&device, Arc::clone(&layout), 0);
        mismatch.base.fragment.as_mut().unwrap().targets[0].format =
            TextureFormat::from_raw(TextureFormat::BGRA8_UNORM);
        let bad = device.create_subpass_render_pipeline(mismatch);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("mismatch should be scoped");
        assert!(bad.is_error());
        assert_eq!(
            scoped.message,
            "subpass render pipeline color target format must match layout"
        );

        device.push_error_scope(ErrorFilter::Validation);
        let out_of_range = device.create_subpass_render_pipeline(subpass_pipeline_descriptor(
            &device,
            Arc::clone(&layout),
            2,
        ));
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("out-of-range should be scoped");
        assert!(out_of_range.is_error());
        assert_eq!(
            scoped.message,
            "subpass render pipeline subpassIndex is out of range"
        );

        device.push_error_scope(ErrorFilter::Validation);
        let mut base_error = subpass_pipeline_descriptor(&device, Arc::clone(&layout), 0);
        base_error.base.error = Some("base render pipeline conversion failed".to_owned());
        let bad = device.create_subpass_render_pipeline(base_error);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("base descriptor error should be scoped");
        assert!(bad.is_error());
        assert_eq!(scoped.message, "base render pipeline conversion failed");

        let wrong_subpass = Arc::new(
            device.create_subpass_render_pipeline(subpass_pipeline_descriptor(&device, layout, 1)),
        );
        let pass_layout =
            Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));
        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device), persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert_eq!(pass.set_pipeline(wrong_subpass), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("subpass render pipeline is not compatible with the active subpass".to_owned())
        );
    }

    #[test]
    fn subpass_render_pass_draw_records_active_subpass_and_resources() {
        let device = noop_device();
        let layout = Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));
        let pipeline = Arc::new(device.create_subpass_render_pipeline(
            subpass_pipeline_descriptor(&device, Arc::clone(&layout), 0),
        ));
        assert!(!pipeline.is_error());
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::VERTEX,
            size: 16,
            mapped_at_creation: false,
        }));

        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: layout,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device), persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_vertex_buffer(0, Some(Arc::clone(&buffer)), 0, 16, device.limits()),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.referenced_buffers().len(), 1);
        let CommandExecution::SubpassRenderPass(command) = &command_buffer.command_ops()[0] else {
            panic!("expected subpass render pass command");
        };
        assert_eq!(command.draws.len(), 1);
        assert_eq!(command.draws[0].subpass_index, 0);
        let RenderDrawExecution::Direct { vertex_count, .. } = command.draws[0].draw else {
            panic!("expected direct subpass draw");
        };
        assert_eq!(vertex_count, 3);
    }

    #[test]
    fn subpass_render_pass_draw_threads_viewport_and_scissor_to_hal() {
        let device = noop_device();
        let layout = Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));
        let pipeline = Arc::new(device.create_subpass_render_pipeline(
            subpass_pipeline_descriptor(&device, Arc::clone(&layout), 0),
        ));
        assert!(!pipeline.is_error());
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::VERTEX,
            size: 16,
            mapped_at_creation: false,
        }));

        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: layout,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device), persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_vertex_buffer(0, Some(Arc::clone(&buffer)), 0, 16, device.limits()),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.set_viewport(1.0, 2.0, 3.0, 4.0, 0.25, 0.75), None);
        assert_eq!(pass.set_scissor_rect(1, 1, 2, 2), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        let hal_copy = crate::queue::hal_command_execution(&command_buffer.command_ops()[0])
            .expect("subpass command should map to HAL");
        let yawgpu_hal::HalCopy::SubpassRenderPass(pass) = hal_copy else {
            panic!("expected HAL subpass render pass");
        };
        assert_eq!(pass.draws.len(), 2);
        assert_eq!(pass.draws[0].viewport, None);
        assert_eq!(pass.draws[0].scissor_rect, None);
        let viewport = pass.draws[1].viewport.expect("viewport should be threaded");
        assert_eq!(viewport.x, 1.0);
        assert_eq!(viewport.y, 2.0);
        assert_eq!(viewport.width, 3.0);
        assert_eq!(viewport.height, 4.0);
        assert_eq!(viewport.min_depth, 0.25);
        assert_eq!(viewport.max_depth, 0.75);
        let scissor = pass.draws[1]
            .scissor_rect
            .expect("scissor rect should be threaded");
        assert_eq!(scissor.x, 1);
        assert_eq!(scissor.y, 1);
        assert_eq!(scissor.width, 2);
        assert_eq!(scissor.height, 2);
    }

    #[test]
    fn subpass_render_pass_rejects_scissor_outside_extent() {
        let device = noop_device();
        let layout = Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));
        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: layout,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device), persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        // Like the regular render pass, an out-of-extent scissor is deferred to the
        // encoder error sink: `set_scissor_rect` returns `None` and the error surfaces
        // at `finish()` (see `record_draw_command` / the `render_pass` OOB-scissor test).
        assert_eq!(pass.set_scissor_rect(3, 0, 2, 1), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("render pass scissor rectangle exceeds attachment size".to_owned())
        );
    }

    #[test]
    fn subpass_render_pass_draw_auto_wires_input_attachment_bind_group() {
        // A subpass-input binding is supplied by the pass from its input-source
        // mapping, so drawing a pipeline whose only bind group is the input
        // attachment must succeed without a caller `set_bind_group` call.
        let device = noop_device();
        let layout = Arc::new(device.create_subpass_pass_layout(two_subpass_layout_descriptor()));

        let writer = Arc::new(
            device.create_subpass_render_pipeline(subpass_pipeline_descriptor(
                &device,
                Arc::clone(&layout),
                0,
            )),
        );
        assert!(!writer.is_error());

        let reader_module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@group(0) @binding(0) var gbuffer: subpass_input<f32>;
             @vertex fn vs() -> @builtin(position) vec4<f32> {
                 return vec4<f32>(0.0, 0.0, 0.0, 1.0);
             }
             @fragment fn fs() -> @location(0) vec4<f32> {
                 let loaded = subpassLoad(gbuffer);
                 return vec4<f32>(loaded.g, loaded.r, loaded.b, 1.0);
             }"
                .to_owned(),
            )),
        );
        let reader = Arc::new(device.create_subpass_render_pipeline(
            SubpassRenderPipelineDescriptor {
                base: render_pipeline_descriptor(reader_module),
                pass_layout: Arc::clone(&layout),
                subpass_index: 1,
                error: None,
            },
        ));
        assert!(!reader.is_error());
        // The reader pipeline's auto layout carries exactly the input-attachment group.
        assert_eq!(reader.bind_group_layouts().len(), 1);

        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_subpass_render_pass(
            &device,
            SubpassRenderPassDescriptor {
                pass_layout: layout,
                extent: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                color_attachments: vec![persistent_color(&device), persistent_color(&device)],
                depth_stencil_attachment: None,
                error: None,
            },
        );
        assert_eq!(error, None);
        assert_eq!(pass.set_pipeline(writer), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.next_subpass(), None);
        assert_eq!(pass.set_pipeline(reader), None);
        // No `set_bind_group` for group 0: the input attachment is auto-wired by
        // the subpass pass, so the draw must still validate.
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }
}
