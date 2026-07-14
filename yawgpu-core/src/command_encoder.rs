use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::buffer::*;
use crate::compute_pass::*;
use crate::compute_pipeline::*;
use crate::copy::*;
use crate::device::{Device, FeatureSet};
use crate::extent::*;
use crate::format::*;
use crate::limits::*;
use crate::pass::*;
use crate::query_set::*;
use crate::render_pass::*;
use crate::render_pipeline::*;
#[cfg(feature = "tiled")]
use crate::subpass::*;
use crate::texture::*;
use crate::texture_view::*;

/// Records commands for the CommandEncoder.
#[derive(Debug, Clone)]
pub struct CommandEncoder {
    pub(crate) inner: Arc<CommandEncoderInner>,
}

/// Holds shared state for the command encoder handle.
#[derive(Debug)]
pub(crate) struct CommandEncoderInner {
    pub(crate) device: Option<Device>,
    pub(crate) features: FeatureSet,
    pub(crate) limits: Limits,
    pub(crate) state: Mutex<CommandEncoderState>,
}

/// Tracks the lifecycle state for command encoder.
#[derive(Debug)]
pub(crate) struct CommandEncoderState {
    pub(crate) lifecycle: CommandEncoderLifecycle,
    pub(crate) open_pass: Option<PassToken>,
    pub(crate) next_pass_id: u64,
    pub(crate) first_error: Option<String>,
    pub(crate) debug_group_depth: u32,
    pub(crate) has_recorded_command: bool,
    pub(crate) referenced_buffers: Vec<Arc<Buffer>>,
    pub(crate) referenced_textures: Vec<Texture>,
    pub(crate) referenced_query_sets: Vec<QuerySet>,
    pub(crate) command_ops: Vec<CommandExecution>,
}

/// Enumerates command encoder lifecycle values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandEncoderLifecycle {
    /// Recording variant.
    Recording,
    /// Finished variant.
    Finished,
}

/// Enumerates pass kind values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PassKind {
    /// Render variant.
    Render,
    /// Compute variant.
    Compute,
}

/// Stores pass token data used by validation and backend submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PassToken {
    pub(crate) kind: PassKind,
    pub(crate) id: u64,
}

/// Stores command buffer data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct CommandBuffer {
    pub(crate) inner: Arc<CommandBufferInner>,
}

/// Holds shared state for the command buffer handle.
#[derive(Debug)]
pub(crate) struct CommandBufferInner {
    pub(crate) is_error: bool,
    pub(crate) error_message: Option<String>,
    pub(crate) referenced_buffers: Vec<Arc<Buffer>>,
    pub(crate) referenced_textures: Vec<Texture>,
    pub(crate) referenced_query_sets: Vec<QuerySet>,
    pub(crate) command_ops: Vec<CommandExecution>,
    pub(crate) submitted: Mutex<bool>,
}

/// Stores the data needed to replay a BufferCopyCommand.
#[derive(Debug, Clone)]
pub(crate) struct BufferCopyCommand {
    pub(crate) source: Arc<Buffer>,
    pub(crate) source_offset: u64,
    pub(crate) destination: Arc<Buffer>,
    pub(crate) destination_offset: u64,
    pub(crate) size: u64,
}

/// Stores the data needed to replay a BufferClearCommand.
#[derive(Debug, Clone)]
pub(crate) struct BufferClearCommand {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

/// Enumerates texture copy command values.
#[derive(Debug, Clone)]
pub(crate) enum TextureCopyCommand {
    /// Buffer to texture variant.
    BufferToTexture {
        /// Source variant.
        source: TexelCopyBufferInfo,
        /// Destination variant.
        destination: TexelCopyTextureInfo,
        /// Copy size variant.
        copy_size: Extent3d,
    },
    /// Texture to buffer variant.
    TextureToBuffer {
        /// Source variant.
        source: TexelCopyTextureInfo,
        /// Destination variant.
        destination: TexelCopyBufferInfo,
        /// Copy size variant.
        copy_size: Extent3d,
    },
    /// Texture to texture variant.
    TextureToTexture {
        /// Source variant.
        source: TexelCopyTextureInfo,
        /// Destination variant.
        destination: TexelCopyTextureInfo,
        /// Copy size variant.
        copy_size: Extent3d,
    },
}

/// Enumerates command execution values.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum CommandExecution {
    /// Buffer copy variant.
    BufferCopy(BufferCopyCommand),
    /// Buffer clear variant.
    BufferClear(BufferClearCommand),
    /// Texture copy variant.
    TextureCopy(TextureCopyCommand),
    /// Query-set resolve variant.
    ResolveQuerySet(ResolveQuerySetCommand),
    /// Compute pass variant.
    ComputePass(ComputePassCommand),
    /// Render pass variant.
    RenderPass(RenderPassCommand),
    #[cfg(feature = "tiled")]
    /// Subpass render pass variant.
    SubpassRenderPass(SubpassRenderPassCommand),
}

/// Stores the data needed to replay a ComputePassCommand.
#[derive(Debug, Clone)]
pub(crate) struct ComputePassCommand {
    pub(crate) pipeline: Arc<ComputePipeline>,
    pub(crate) bind_groups: BTreeMap<u32, BoundBindGroup>,
    pub(crate) dispatch: ComputeDispatch,
    /// Snapshot of the pass's user-immediates scratch (Block 94) at dispatch
    /// time -- always [`crate::pass::MAX_IMMEDIATE_DATA_BYTES`] bytes long.
    /// Threaded to the HAL as `HalComputePass::immediate_data`; the
    /// pipeline's actual budget is its pipeline layout's `immediate_size`
    /// bytes of this prefix (Block 93).
    pub(crate) immediate_data: Vec<u8>,
}

/// Enumerates compute dispatch execution values.
#[derive(Debug, Clone)]
pub(crate) enum ComputeDispatch {
    /// Direct dispatch variant.
    Direct {
        /// Workgroup counts.
        workgroups: (u32, u32, u32),
    },
    /// Indirect dispatch variant.
    Indirect {
        /// Buffer containing dispatch arguments.
        buffer: Arc<Buffer>,
        /// Byte offset of dispatch arguments.
        offset: u64,
    },
}

/// Stores the data needed to replay a RenderPassCommand.
#[derive(Debug, Clone)]
pub(crate) struct RenderPassCommand {
    pub(crate) pipeline: Option<Arc<RenderPipeline>>,
    pub(crate) color_attachments: Vec<Option<RenderPassColorExecution>>,
    pub(crate) depth_stencil_attachment: Option<RenderPassDepthStencilExecution>,
    pub(crate) attachment_textures: Vec<Texture>,
    pub(crate) bind_groups: BTreeMap<u32, BoundBindGroup>,
    pub(crate) vertex_buffers: BTreeMap<u32, BoundVertexBuffer>,
    pub(crate) index_buffer: Option<BoundIndexBuffer>,
    pub(crate) indirect_buffer: Option<BoundIndirectBuffer>,
    pub(crate) viewport: Option<Viewport>,
    pub(crate) scissor_rect: Option<ScissorRect>,
    pub(crate) blend_constant: [f32; 4],
    pub(crate) stencil_reference: u32,
    pub(crate) occlusion_query_set: Option<QuerySet>,
    pub(crate) occlusion_query_index: Option<u32>,
    pub(crate) draw: Option<RenderDrawExecution>,
    /// Snapshot of the pass's user-immediates scratch (Block 94) at draw
    /// time -- always [`crate::pass::MAX_IMMEDIATE_DATA_BYTES`] bytes long.
    /// Threaded to the HAL as `HalRenderPass::immediate_data`.
    pub(crate) immediate_data: Vec<u8>,
}

/// Stores the data needed to replay a subpass render pass.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub(crate) struct SubpassRenderPassCommand {
    pub(crate) layout: Arc<SubpassPassLayout>,
    pub(crate) extent: Extent3d,
    pub(crate) color_attachments: Vec<SubpassColorAttachmentBinding>,
    pub(crate) depth_stencil_attachment: Option<SubpassDepthStencilAttachmentBinding>,
    pub(crate) draws: Vec<SubpassDrawExecution>,
}

/// Stores the data needed to replay a query-set resolve.
#[derive(Debug, Clone)]
pub(crate) struct ResolveQuerySetCommand {
    pub(crate) query_set: Arc<QuerySet>,
    pub(crate) first_query: u32,
    pub(crate) query_count: u32,
    pub(crate) destination: Arc<Buffer>,
    pub(crate) destination_offset: u64,
}

/// Stores viewport state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Viewport {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) width: f32,
    pub(crate) height: f32,
    pub(crate) min_depth: f32,
    pub(crate) max_depth: f32,
}

/// Stores scissor rectangle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScissorRect {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Stores color metadata.
#[derive(Debug, Clone)]
pub(crate) struct RenderPassColorExecution {
    pub(crate) texture: Texture,
    pub(crate) view_format: TextureFormat,
    pub(crate) resolve_target: Option<Texture>,
    pub(crate) resolve_view_format: Option<TextureFormat>,
    pub(crate) mip_level: u32,
    pub(crate) array_layer: u32,
    pub(crate) depth_slice: u32,
    pub(crate) resolve_mip_level: u32,
    pub(crate) resolve_array_layer: u32,
    pub(crate) load_op: LoadOp,
    pub(crate) store_op: StoreOp,
    pub(crate) clear_value: Color,
}

/// Stores depth-stencil metadata.
#[derive(Debug, Clone)]
pub(crate) struct RenderPassDepthStencilExecution {
    pub(crate) texture: Texture,
    pub(crate) format: TextureFormat,
    pub(crate) mip_level: u32,
    pub(crate) array_layer: u32,
    pub(crate) depth_load_op: LoadOp,
    pub(crate) depth_store_op: StoreOp,
    pub(crate) depth_clear_value: f32,
    pub(crate) depth_read_only: bool,
    pub(crate) stencil_load_op: LoadOp,
    pub(crate) stencil_store_op: StoreOp,
    pub(crate) stencil_clear_value: u32,
    pub(crate) stencil_read_only: bool,
}

/// Stores render draw execution data used by validation and backend submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RenderDrawExecution {
    Direct {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    Indexed {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    },
    Indirect {
        offset: u64,
    },
    IndexedIndirect {
        offset: u64,
    },
}

/// Stores an indirect draw buffer binding.
#[derive(Debug, Clone)]
pub(crate) struct BoundIndirectBuffer {
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) offset: u64,
}

impl CommandEncoder {
    /// Creates a new instance.
    pub(crate) fn new(device: Option<Device>, features: FeatureSet, limits: Limits) -> Self {
        Self {
            inner: Arc::new(CommandEncoderInner {
                device,
                features,
                limits,
                state: Mutex::new(CommandEncoderState {
                    lifecycle: CommandEncoderLifecycle::Recording,
                    open_pass: None,
                    next_pass_id: 0,
                    first_error: None,
                    debug_group_depth: 0,
                    has_recorded_command: false,
                    referenced_buffers: Vec::new(),
                    referenced_textures: Vec::new(),
                    referenced_query_sets: Vec::new(),
                    command_ops: Vec::new(),
                }),
            }),
        }
    }

    /// Creates an error-state instance.
    pub(crate) fn new_error(message: impl Into<String>) -> Self {
        let encoder = Self::new(None, FeatureSet::new(), Limits::DEFAULT);
        encoder.record_first_error(message);
        encoder
    }

    /// Begins a render pass on this encoder and returns its render-pass encoder.
    #[must_use]
    pub fn begin_render_pass(
        &self,
        descriptor: &RenderPassDescriptor,
    ) -> (RenderPassEncoder, Option<String>) {
        let (token, immediate_error) = self.begin_pass(PassKind::Render);
        let attachment_signature =
            render_pass_attachment_signature(descriptor, &self.inner.features, self.inner.limits)
                .ok();
        if immediate_error.is_none() {
            if let Err(message) =
                validate_render_pass_descriptor(descriptor, &self.inner.features, self.inner.limits)
            {
                self.record_first_error(message);
            } else {
                self.record_referenced_textures(render_pass_attachment_textures(descriptor));
                self.record_referenced_query_sets(render_pass_query_sets(descriptor));
            }
        }
        (
            RenderPassEncoder {
                inner: {
                    let inner = Arc::new(PassEncoderInner::new(
                        self.clone(),
                        token,
                        PassEncoderInit {
                            attachment_signature,
                            render_extent: render_pass_extent(descriptor),
                            attachment_textures: render_pass_attachment_textures(descriptor),
                            render_color_attachments: render_pass_color_executions(descriptor),
                            render_depth_stencil_attachment: render_pass_depth_stencil_execution(
                                descriptor,
                            ),
                            occlusion_query_set: descriptor.occlusion_query_set.clone(),
                            max_draw_count: descriptor.max_draw_count,
                        },
                    ));
                    if let Err(message) = inner
                        .state
                        .lock()
                        .set_attachment_texture_uses(render_pass_attachment_scope_uses(descriptor))
                    {
                        self.record_first_error(message);
                    }
                    inner
                },
            },
            immediate_error,
        )
    }

    /// Begins a tiled subpass render pass.
    #[cfg(feature = "tiled")]
    #[must_use]
    pub fn begin_subpass_render_pass(
        &self,
        device: &Device,
        descriptor: SubpassRenderPassDescriptor,
    ) -> (SubpassRenderPass, Option<String>) {
        let mut state = self.inner.state.lock();
        let token = PassToken {
            kind: PassKind::Render,
            id: state.next_pass_id,
        };
        state.next_pass_id = state.next_pass_id.saturating_add(1);
        if state.lifecycle != CommandEncoderLifecycle::Recording {
            return (
                SubpassRenderPass::new(self.clone(), token, descriptor, None, true),
                Some("command encoder cannot record after finish".to_owned()),
            );
        }
        if state.open_pass.is_some() {
            record_first_error_locked(
                &mut state,
                "command encoder cannot begin a pass while another pass is open",
            );
            return (
                SubpassRenderPass::new(self.clone(), token, descriptor, None, true),
                None,
            );
        }
        if state.has_recorded_command {
            record_first_error_locked(
                &mut state,
                "subpass render pass must be the first command encoder operation",
            );
            return (
                SubpassRenderPass::new(self.clone(), token, descriptor, None, true),
                None,
            );
        }

        let validation_error =
            validate_subpass_render_pass_descriptor(&descriptor, &self.inner.features)
                .or_else(|| resolve_subpass_render_pass_resources(device, &descriptor).err());
        let mut hal = None;
        let mut is_error = validation_error.is_some();
        if let Some(message) = validation_error {
            record_first_error_locked(&mut state, message);
        } else {
            match device.hal().begin_subpass_render_pass() {
                Ok(pass) => hal = Some(pass),
                Err(error) => {
                    is_error = true;
                    record_first_error_locked(&mut state, error.to_string());
                }
            }
        }
        state.has_recorded_command = true;
        state.open_pass = Some(token);
        (
            SubpassRenderPass::new(self.clone(), token, descriptor, hal, is_error),
            None,
        )
    }

    /// Begins a compute pass on this encoder and returns its compute-pass encoder.
    #[must_use]
    pub fn begin_compute_pass(&self) -> (ComputePassEncoder, Option<String>) {
        let (token, immediate_error) = self.begin_pass(PassKind::Compute);
        (
            ComputePassEncoder {
                inner: Arc::new(PassEncoderInner::new(
                    self.clone(),
                    token,
                    PassEncoderInit {
                        attachment_signature: None,
                        render_extent: None,
                        attachment_textures: Vec::new(),
                        render_color_attachments: Vec::new(),
                        render_depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        max_draw_count: u64::MAX,
                    },
                )),
            },
            immediate_error,
        )
    }

    /// Begins a pass, locking the encoder until the pass ends.
    pub(crate) fn begin_pass(&self, kind: PassKind) -> (PassToken, Option<String>) {
        let mut state = self.inner.state.lock();
        let token = PassToken {
            kind,
            id: state.next_pass_id,
        };
        state.next_pass_id = state.next_pass_id.saturating_add(1);

        if state.lifecycle != CommandEncoderLifecycle::Recording {
            return (
                token,
                Some("command encoder cannot record after finish".to_owned()),
            );
        }

        if state.open_pass.is_some() {
            record_first_error_locked(
                &mut state,
                "command encoder cannot begin a pass while another pass is open",
            );
            return (token, None);
        }

        state.open_pass = Some(token);
        state.has_recorded_command = true;
        (token, None)
    }

    /// Records a single debug marker into the command stream.
    pub fn insert_debug_marker(&self) -> Option<String> {
        self.record_encoder_command()
    }

    /// Records a validation error against the encoder, marking it unusable.
    pub fn record_validation_error(&self, message: impl Into<String>) -> Option<String> {
        self.record_buffer_command(Vec::new(), None, None, None, || Err(message.into()))
    }

    /// Records a buffer-to-buffer copy after validating the ranges and usages.
    pub fn copy_buffer_to_buffer(
        &self,
        source: Arc<Buffer>,
        source_offset: u64,
        destination: Arc<Buffer>,
        destination_offset: u64,
        size: u64,
    ) -> Option<String> {
        let copy = BufferCopyCommand {
            source: Arc::clone(&source),
            source_offset,
            destination: Arc::clone(&destination),
            destination_offset,
            size,
        };
        self.record_buffer_command(
            vec![Arc::clone(&source), Arc::clone(&destination)],
            Some(copy),
            None,
            None,
            || {
                validate_copy_buffer_to_buffer(
                    &source,
                    source_offset,
                    &destination,
                    destination_offset,
                    size,
                )
            },
        )
    }

    /// Records a buffer clear (zero-fill) over the given range after validation.
    pub fn clear_buffer(&self, buffer: Arc<Buffer>, offset: u64, size: u64) -> Option<String> {
        let clear = BufferClearCommand {
            buffer: Arc::clone(&buffer),
            offset,
            size,
        };
        self.record_buffer_command(vec![Arc::clone(&buffer)], None, Some(clear), None, || {
            validate_clear_buffer(&buffer, offset, size)
        })
    }

    /// Records an encoder-side buffer write over the given range after validation.
    pub fn write_buffer(&self, buffer: Arc<Buffer>, offset: u64, size: u64) -> Option<String> {
        self.record_buffer_command(vec![Arc::clone(&buffer)], None, None, None, || {
            validate_encoder_write_buffer(&buffer, offset, size)
        })
    }

    /// Records a timestamp write into the query set at `query_index`.
    pub fn write_timestamp(&self, query_set: Arc<QuerySet>, query_index: u32) -> Option<String> {
        self.record_referenced_query_set((*query_set).clone());
        self.record_buffer_command(Vec::new(), None, None, None, || {
            validate_timestamp_query_set(&query_set, "write timestamp")?;
            validate_query_index(&query_set, query_index, "write timestamp query index")
        })
    }

    /// Records resolution of a query-set range into a destination buffer.
    pub fn resolve_query_set(
        &self,
        query_set: Arc<QuerySet>,
        first_query: u32,
        query_count: u32,
        destination: Arc<Buffer>,
        destination_offset: u64,
    ) -> Option<String> {
        if let Err(message) = self.record_command_guard() {
            let mut state = self.inner.state.lock();
            if state.lifecycle == CommandEncoderLifecycle::Recording {
                record_first_error_locked(&mut state, message);
                return None;
            }
            return Some(message);
        }

        if let Err(message) = validate_resolve_query_set(
            &query_set,
            first_query,
            query_count,
            &destination,
            destination_offset,
        ) {
            self.record_first_error(message);
        } else {
            self.record_referenced_query_set((*query_set).clone());
            let mut state = self.inner.state.lock();
            state.has_recorded_command = true;
            state.referenced_buffers.push(Arc::clone(&destination));
            state
                .command_ops
                .push(CommandExecution::ResolveQuerySet(ResolveQuerySetCommand {
                    query_set,
                    first_query,
                    query_count,
                    destination,
                    destination_offset,
                }));
        }
        None
    }

    /// Records a buffer-to-texture copy after validating layout, sizes, and usages.
    pub fn copy_buffer_to_texture(
        &self,
        source: TexelCopyBufferInfo,
        destination: TexelCopyTextureInfo,
        copy_size: Extent3d,
    ) -> Option<String> {
        let copy = TextureCopyCommand::BufferToTexture {
            source: source.clone(),
            destination: destination.clone(),
            copy_size,
        };
        self.record_buffer_command(
            vec![Arc::clone(&source.buffer)],
            None,
            None,
            Some(copy),
            || {
                validate_buffer_texture_copy(
                    source,
                    BufferUsage::COPY_SRC,
                    destination,
                    TextureUsage::COPY_DST,
                    copy_size,
                    "copy buffer to texture",
                    self.inner.device.as_ref(),
                )
            },
        )
    }

    /// Records a texture-to-buffer copy after validating layout, sizes, and usages.
    pub fn copy_texture_to_buffer(
        &self,
        source: TexelCopyTextureInfo,
        destination: TexelCopyBufferInfo,
        copy_size: Extent3d,
    ) -> Option<String> {
        let copy = TextureCopyCommand::TextureToBuffer {
            source: source.clone(),
            destination: destination.clone(),
            copy_size,
        };
        self.record_buffer_command(
            vec![Arc::clone(&destination.buffer)],
            None,
            None,
            Some(copy),
            || {
                validate_buffer_texture_copy(
                    destination,
                    BufferUsage::COPY_DST,
                    source,
                    TextureUsage::COPY_SRC,
                    copy_size,
                    "copy texture to buffer",
                    self.inner.device.as_ref(),
                )
            },
        )
    }

    /// Records a texture-to-texture copy after validating the regions and usages.
    pub fn copy_texture_to_texture(
        &self,
        source: TexelCopyTextureInfo,
        destination: TexelCopyTextureInfo,
        copy_size: Extent3d,
    ) -> Option<String> {
        let copy = TextureCopyCommand::TextureToTexture {
            source: source.clone(),
            destination: destination.clone(),
            copy_size,
        };
        self.record_buffer_command(Vec::new(), None, None, Some(copy), || {
            validate_texture_to_texture_copy(source, destination, copy_size)
        })
    }

    /// Opens a debug group in the command stream.
    pub fn push_debug_group(&self) -> Option<String> {
        let mut state = self.inner.state.lock();
        if state.lifecycle != CommandEncoderLifecycle::Recording {
            return Some("command encoder cannot record after finish".to_owned());
        }
        if state.open_pass.is_some() {
            record_first_error_locked(
                &mut state,
                "command encoder command cannot be recorded while a pass is open",
            );
            return None;
        }
        state.debug_group_depth = state.debug_group_depth.saturating_add(1);
        state.has_recorded_command = true;
        None
    }

    /// Closes the most recently opened debug group.
    pub fn pop_debug_group(&self) -> Option<String> {
        let mut state = self.inner.state.lock();
        if state.lifecycle != CommandEncoderLifecycle::Recording {
            return Some("command encoder cannot record after finish".to_owned());
        }
        if state.open_pass.is_some() {
            record_first_error_locked(
                &mut state,
                "command encoder command cannot be recorded while a pass is open",
            );
            return None;
        }
        if state.debug_group_depth == 0 {
            record_first_error_locked(&mut state, "command encoder debug group stack is empty");
        } else {
            state.debug_group_depth -= 1;
        }
        None
    }

    /// Returns an error if the encoder is finished or otherwise unusable for recording.
    pub(crate) fn record_command_guard(&self) -> Result<(), String> {
        let state = self.inner.state.lock();
        if state.lifecycle != CommandEncoderLifecycle::Recording {
            return Err("command encoder cannot record after finish".to_owned());
        }
        if state.open_pass.is_some() {
            return Err(
                "command encoder command cannot be recorded while a pass is open".to_owned(),
            );
        }
        Ok(())
    }

    /// Validates the encoder is recordable and notes that a command was recorded.
    pub(crate) fn record_encoder_command(&self) -> Option<String> {
        match self.record_command_guard() {
            Ok(()) => {
                self.inner.state.lock().has_recorded_command = true;
                None
            }
            Err(message) => {
                let mut state = self.inner.state.lock();
                if state.lifecycle == CommandEncoderLifecycle::Recording {
                    record_first_error_locked(&mut state, message);
                    None
                } else {
                    Some(message)
                }
            }
        }
    }

    /// Records a command that references a buffer, validating the encoder and tracking the buffer.
    pub(crate) fn record_buffer_command<F>(
        &self,
        referenced_buffers: Vec<Arc<Buffer>>,
        buffer_copy: Option<BufferCopyCommand>,
        buffer_clear: Option<BufferClearCommand>,
        texture_copy: Option<TextureCopyCommand>,
        validate: F,
    ) -> Option<String>
    where
        F: FnOnce() -> Result<(), String>,
    {
        if let Err(message) = self.record_command_guard() {
            let mut state = self.inner.state.lock();
            if state.lifecycle == CommandEncoderLifecycle::Recording {
                record_first_error_locked(&mut state, message);
                return None;
            }
            return Some(message);
        }

        if let Err(message) = validate() {
            self.record_first_error(message);
        } else {
            let mut state = self.inner.state.lock();
            state.has_recorded_command = true;
            state.referenced_buffers.extend(referenced_buffers);
            if let Some(copy) = buffer_copy {
                state
                    .command_ops
                    .push(CommandExecution::BufferCopy(copy.clone()));
            }
            if let Some(clear) = buffer_clear {
                state
                    .command_ops
                    .push(CommandExecution::BufferClear(clear.clone()));
            }
            if let Some(copy) = texture_copy {
                state
                    .command_ops
                    .push(CommandExecution::TextureCopy(copy.clone()));
            }
        }
        None
    }

    /// Finishes recording and returns the completed object.
    #[must_use]
    pub fn finish(&self) -> (CommandBuffer, Option<String>) {
        let mut state = self.inner.state.lock();
        if state.lifecycle != CommandEncoderLifecycle::Recording {
            let message = "command encoder cannot be finished more than once".to_owned();
            return (CommandBuffer::new_error(message.clone()), Some(message));
        }
        state.lifecycle = CommandEncoderLifecycle::Finished;

        let finish_error = state
            .first_error
            .clone()
            .or_else(|| {
                state
                    .open_pass
                    .is_some()
                    .then(|| "command encoder cannot finish while a pass is open".to_owned())
            })
            .or_else(|| {
                (state.debug_group_depth != 0)
                    .then(|| "command encoder debug group stack is unbalanced".to_owned())
            });
        let referenced_buffers = if finish_error.is_some() {
            Vec::new()
        } else {
            std::mem::take(&mut state.referenced_buffers)
        };
        let referenced_textures = if finish_error.is_some() {
            Vec::new()
        } else {
            std::mem::take(&mut state.referenced_textures)
        };
        let referenced_query_sets = if finish_error.is_some() {
            Vec::new()
        } else {
            std::mem::take(&mut state.referenced_query_sets)
        };
        let command_ops = if finish_error.is_some() {
            Vec::new()
        } else {
            std::mem::take(&mut state.command_ops)
        };
        (
            CommandBuffer::new(
                finish_error.clone(),
                referenced_buffers,
                referenced_textures,
                referenced_query_sets,
                command_ops,
            ),
            finish_error,
        )
    }

    /// Ends pass recording.
    pub(crate) fn end_pass(&self, token: PassToken) {
        let mut state = self.inner.state.lock();
        if state.open_pass == Some(token) {
            state.open_pass = None;
        }
    }

    /// Returns true when `token` is the currently open pass.
    pub(crate) fn is_open_pass(&self, token: PassToken) -> bool {
        self.inner.state.lock().open_pass == Some(token)
    }

    /// Stores `message` as the encoder's first error if none has been recorded yet.
    pub(crate) fn record_first_error(&self, message: impl Into<String>) {
        let mut state = self.inner.state.lock();
        record_first_error_locked(&mut state, message);
    }

    /// Tracks a buffer referenced by the encoder so it stays alive until submit.
    pub(crate) fn record_referenced_buffer(&self, buffer: Arc<Buffer>) {
        self.inner.state.lock().referenced_buffers.push(buffer);
    }

    /// Tracks several buffers referenced by the encoder.
    pub(crate) fn record_referenced_buffers(&self, buffers: Vec<Arc<Buffer>>) {
        self.inner.state.lock().referenced_buffers.extend(buffers);
    }

    /// Tracks several textures referenced by the encoder.
    pub(crate) fn record_referenced_textures(&self, textures: Vec<Texture>) {
        self.inner.state.lock().referenced_textures.extend(textures);
    }

    /// Tracks a query set referenced by the encoder.
    pub(crate) fn record_referenced_query_set(&self, query_set: QuerySet) {
        self.inner
            .state
            .lock()
            .referenced_query_sets
            .push(query_set);
    }

    /// Tracks several query sets referenced by the encoder.
    pub(crate) fn record_referenced_query_sets(&self, query_sets: Vec<QuerySet>) {
        self.inner
            .state
            .lock()
            .referenced_query_sets
            .extend(query_sets);
    }

    /// Appends a finished compute-pass command to the encoder's command list.
    pub(crate) fn record_compute_pass(&self, command: ComputePassCommand) {
        self.inner
            .state
            .lock()
            .command_ops
            .push(CommandExecution::ComputePass(command));
    }

    /// Appends a finished render-pass command to the encoder's command list.
    pub(crate) fn record_render_pass(&self, command: RenderPassCommand) {
        self.inner
            .state
            .lock()
            .command_ops
            .push(CommandExecution::RenderPass(command));
    }

    /// Appends a finished subpass-render-pass command to the encoder's command list.
    #[cfg(feature = "tiled")]
    pub(crate) fn record_subpass_render_pass(&self, command: SubpassRenderPassCommand) {
        self.inner
            .state
            .lock()
            .command_ops
            .push(CommandExecution::SubpassRenderPass(command));
    }

    /// Restores user store ops on the most recently recorded render-pass command.
    pub(crate) fn patch_last_render_pass_store_ops(
        &self,
        color_store_ops: &[Option<StoreOp>],
        depth_store_op: Option<StoreOp>,
        stencil_store_op: Option<StoreOp>,
    ) {
        let mut state = self.inner.state.lock();
        let Some(CommandExecution::RenderPass(command)) = state.command_ops.last_mut() else {
            return;
        };
        for (attachment, store_op) in command
            .color_attachments
            .iter_mut()
            .zip(color_store_ops.iter().copied())
        {
            if let (Some(attachment), Some(store_op)) = (attachment, store_op) {
                attachment.store_op = store_op;
            }
        }
        if let Some(attachment) = &mut command.depth_stencil_attachment {
            if let Some(store_op) = depth_store_op {
                attachment.depth_store_op = store_op;
            }
            if let Some(store_op) = stencil_store_op {
                attachment.stencil_store_op = store_op;
            }
        }
    }

    /// Returns true when this object is finished.
    pub(crate) fn is_finished(&self) -> bool {
        self.inner.state.lock().lifecycle == CommandEncoderLifecycle::Finished
    }
}

/// Validates copy buffer to buffer and returns a descriptive error on failure.
pub(crate) fn validate_copy_buffer_to_buffer(
    source: &Buffer,
    source_offset: u64,
    destination: &Buffer,
    destination_offset: u64,
    size: u64,
) -> Result<(), String> {
    if source.is_error() || destination.is_error() {
        return Err("copy buffer command cannot use an error buffer".to_owned());
    }
    if !source.usage().contains(BufferUsage::COPY_SRC) {
        return Err("copy source buffer must have CopySrc usage".to_owned());
    }
    if !destination.usage().contains(BufferUsage::COPY_DST) {
        return Err("copy destination buffer must have CopyDst usage".to_owned());
    }
    if !source_offset.is_multiple_of(4) {
        return Err("copy source offset must be 4-byte aligned".to_owned());
    }
    if !destination_offset.is_multiple_of(4) {
        return Err("copy destination offset must be 4-byte aligned".to_owned());
    }
    if !size.is_multiple_of(4) {
        return Err("copy size must be 4-byte aligned".to_owned());
    }
    validate_buffer_range(source_offset, size, source.size(), "copy source range")?;
    validate_buffer_range(
        destination_offset,
        size,
        destination.size(),
        "copy destination range",
    )?;
    if size > 0 && source.same(destination) {
        return Err("copy source and destination ranges must not use the same buffer".to_owned());
    }
    Ok(())
}

/// Validates clear buffer and returns a descriptive error on failure.
pub(crate) fn validate_clear_buffer(buffer: &Buffer, offset: u64, size: u64) -> Result<(), String> {
    if buffer.is_error() {
        return Err("clear buffer command cannot use an error buffer".to_owned());
    }
    if !buffer.usage().contains(BufferUsage::COPY_DST) {
        return Err("clear buffer requires CopyDst usage".to_owned());
    }
    if offset > buffer.size() {
        return Err("clear buffer offset exceeds buffer size".to_owned());
    }
    let resolved_size = if size == u64::MAX {
        buffer.size() - offset
    } else {
        size
    };
    if !offset.is_multiple_of(4) {
        return Err("clear buffer offset must be 4-byte aligned".to_owned());
    }
    if !resolved_size.is_multiple_of(4) {
        return Err("clear buffer size must be 4-byte aligned".to_owned());
    }
    validate_buffer_range(offset, resolved_size, buffer.size(), "clear buffer range")
}

/// Validates encoder write buffer and returns a descriptive error on failure.
pub(crate) fn validate_encoder_write_buffer(
    buffer: &Buffer,
    offset: u64,
    size: u64,
) -> Result<(), String> {
    if buffer.is_error() {
        return Err("command encoder write buffer cannot use an error buffer".to_owned());
    }
    if !buffer.usage().contains(BufferUsage::COPY_DST) {
        return Err("command encoder write buffer requires CopyDst usage".to_owned());
    }
    if !offset.is_multiple_of(4) {
        return Err("command encoder write buffer offset must be 4-byte aligned".to_owned());
    }
    if !size.is_multiple_of(4) {
        return Err("command encoder write buffer size must be 4-byte aligned".to_owned());
    }
    validate_buffer_range(
        offset,
        size,
        buffer.size(),
        "command encoder write buffer range",
    )
}

/// Validates render pass descriptor and returns a descriptive error on failure.
pub(crate) fn validate_render_pass_descriptor(
    descriptor: &RenderPassDescriptor,
    features: &FeatureSet,
    limits: Limits,
) -> Result<(), String> {
    render_pass_attachment_signature(descriptor, features, limits)?;
    if let Some(query_set) = &descriptor.occlusion_query_set {
        validate_occlusion_query_set(query_set, "render pass occlusion query set")?;
    }
    if let Some(timestamp_writes) = &descriptor.timestamp_writes {
        validate_render_pass_timestamp_writes(timestamp_writes)?;
    }
    Ok(())
}

/// Validates render pass timestamp writes and returns a descriptive error on failure.
pub(crate) fn validate_render_pass_timestamp_writes(
    timestamp_writes: &RenderPassTimestampWrites,
) -> Result<(), String> {
    validate_timestamp_query_set(
        &timestamp_writes.query_set,
        "render pass timestamp writes query set",
    )?;
    if timestamp_writes.beginning_index.is_none() && timestamp_writes.end_index.is_none() {
        return Err("render pass timestamp writes requires at least one query index".to_owned());
    }
    if let Some(index) = timestamp_writes.beginning_index {
        validate_query_index(
            &timestamp_writes.query_set,
            index,
            "render pass beginning timestamp query index",
        )?;
    }
    if let Some(index) = timestamp_writes.end_index {
        validate_query_index(
            &timestamp_writes.query_set,
            index,
            "render pass end timestamp query index",
        )?;
    }
    if timestamp_writes.beginning_index == timestamp_writes.end_index {
        return Err("render pass timestamp write indices must be distinct".to_owned());
    }
    Ok(())
}

/// Validates compute pass timestamp writes and returns a descriptive error on failure.
pub fn validate_compute_pass_timestamp_writes(
    timestamp_writes: &RenderPassTimestampWrites,
) -> Result<(), String> {
    validate_timestamp_query_set(
        &timestamp_writes.query_set,
        "compute pass timestamp writes query set",
    )?;
    if timestamp_writes.beginning_index.is_none() && timestamp_writes.end_index.is_none() {
        return Err("compute pass timestamp writes requires at least one query index".to_owned());
    }
    if let Some(index) = timestamp_writes.beginning_index {
        validate_query_index(
            &timestamp_writes.query_set,
            index,
            "compute pass beginning timestamp query index",
        )?;
    }
    if let Some(index) = timestamp_writes.end_index {
        validate_query_index(
            &timestamp_writes.query_set,
            index,
            "compute pass end timestamp query index",
        )?;
    }
    if timestamp_writes.beginning_index == timestamp_writes.end_index {
        return Err("compute pass timestamp write indices must be distinct".to_owned());
    }
    Ok(())
}

/// Validates occlusion query set and returns a descriptive error on failure.
pub(crate) fn validate_occlusion_query_set(
    query_set: &QuerySet,
    usage: &str,
) -> Result<(), String> {
    validate_query_set_alive(query_set, usage)?;
    if query_set.kind() != QueryType::Occlusion {
        return Err(format!("{usage} requires an occlusion query set"));
    }
    Ok(())
}

/// Validates timestamp query set and returns a descriptive error on failure.
pub(crate) fn validate_timestamp_query_set(
    query_set: &QuerySet,
    usage: &str,
) -> Result<(), String> {
    validate_query_set_alive(query_set, usage)?;
    // Destroyed-query-set check is intentionally deferred to queue.submit time
    // (WebGPU spec §17.3 "Queue submit validation"). The query set may be
    // destroyed after recording; it is tracked in CommandBuffer::referenced_query_sets
    // (via record_referenced_query_sets in begin_render_pass / write_timestamp) and
    // queue.submit validates every referenced query set is not destroyed before executing.
    if query_set.kind() != QueryType::Timestamp {
        return Err(format!("{usage} requires a timestamp query set"));
    }
    Ok(())
}

/// Validates query set alive and returns a descriptive error on failure.
pub(crate) fn validate_query_set_alive(query_set: &QuerySet, usage: &str) -> Result<(), String> {
    if query_set.is_error() {
        return Err(format!("{usage} cannot use an error query set"));
    }
    Ok(())
}

/// Validates query index and returns a descriptive error on failure.
pub(crate) fn validate_query_index(
    query_set: &QuerySet,
    index: u32,
    name: &str,
) -> Result<(), String> {
    if index >= query_set.count() {
        return Err(format!("{name} exceeds query set count"));
    }
    Ok(())
}

/// Validates resolve query set and returns a descriptive error on failure.
pub(crate) fn validate_resolve_query_set(
    query_set: &QuerySet,
    first_query: u32,
    query_count: u32,
    destination: &Buffer,
    destination_offset: u64,
) -> Result<(), String> {
    validate_query_set_alive(query_set, "resolve query set")?;
    if query_count == 0 {
        return Err("resolve query count must be greater than zero".to_owned());
    }
    let end_query = first_query
        .checked_add(query_count)
        .ok_or_else(|| "resolve query range overflows".to_owned())?;
    if end_query > query_set.count() {
        return Err("resolve query range exceeds query set count".to_owned());
    }
    if destination.is_error() {
        return Err("resolve query set cannot use an error destination buffer".to_owned());
    }
    if !destination.usage().contains(BufferUsage::QUERY_RESOLVE) {
        return Err("resolve query set destination requires QueryResolve usage".to_owned());
    }
    if !destination_offset.is_multiple_of(256) {
        return Err("resolve query set destination offset must be 256-byte aligned".to_owned());
    }
    let byte_count = u64::from(query_count)
        .checked_mul(8)
        .ok_or_else(|| "resolve query byte count overflows".to_owned())?;
    validate_buffer_range(
        destination_offset,
        byte_count,
        destination.size(),
        "resolve query destination range",
    )
}

/// Returns render pass attachment signature.
pub(crate) fn render_pass_attachment_signature(
    descriptor: &RenderPassDescriptor,
    features: &FeatureSet,
    limits: Limits,
) -> Result<AttachmentSignature, String> {
    if descriptor.color_attachments.len() > descriptor.max_color_attachments as usize {
        return Err("render pass colorAttachmentCount exceeds the device limit".to_owned());
    }

    let mut has_attachment = false;
    let mut render_extent = None;
    let mut sample_count = None;
    let mut color_formats = Vec::with_capacity(descriptor.color_attachments.len());
    let mut color_subresources = BTreeSet::new();
    let mut color_byte_formats = Vec::new();

    for attachment in &descriptor.color_attachments {
        if let Some(attachment) = attachment {
            has_attachment = true;
            validate_color_attachment(attachment, features)?;
            attachment
                .view
                .texture()
                .view_format_caps(attachment.view.format())
                .ok_or_else(|| {
                    "render pass color attachment format must be supported".to_owned()
                })?;
            color_byte_formats.push(attachment.view.format());
            validate_color_attachment_depth_slice(attachment)?;
            validate_single_mip_attachment_view(&attachment.view, "render pass color attachment")?;
            validate_color_attachment_overlap(attachment, &mut color_subresources)?;
            validate_render_attachment_common(
                &attachment.view,
                &mut render_extent,
                &mut sample_count,
                "render pass color attachment",
            )?;
            if let Some(resolve_target) = &attachment.resolve_target {
                validate_resolve_target(&attachment.view, resolve_target)?;
                validate_single_mip_attachment_view(resolve_target, "render pass resolveTarget")?;
            }
            color_formats.push(Some(attachment.view.format()));
        } else {
            color_formats.push(None);
        }
    }

    let mut depth_stencil_format = None;
    let mut depth_read_only = false;
    let mut stencil_read_only = false;
    if let Some(attachment) = &descriptor.depth_stencil_attachment {
        has_attachment = true;
        validate_depth_stencil_attachment(attachment, features)?;
        validate_single_mip_attachment_view(
            &attachment.view,
            "render pass depth-stencil attachment",
        )?;
        depth_stencil_format = Some(attachment.view.format());
        depth_read_only = attachment.depth_read_only;
        stencil_read_only = attachment.stencil_read_only;
        validate_render_attachment_common(
            &attachment.view,
            &mut render_extent,
            &mut sample_count,
            "render pass depth-stencil attachment",
        )?;
    }

    if !has_attachment {
        return Err("render pass requires at least one attachment".to_owned());
    }
    let color_bytes = color_attachment_bytes_per_sample(color_byte_formats)
        .ok_or_else(|| "render pass color attachment byte count overflows".to_owned())?;
    if color_bytes > limits.max_color_attachment_bytes_per_sample {
        return Err(
            "render pass color attachment bytes per sample exceed the device limit".to_owned(),
        );
    }
    Ok(AttachmentSignature {
        color_formats,
        depth_stencil_format,
        sample_count: sample_count.unwrap_or(1),
        depth_read_only,
        stencil_read_only,
    })
}

/// Returns render pass attachment textures.
pub(crate) fn render_pass_attachment_textures(descriptor: &RenderPassDescriptor) -> Vec<Texture> {
    let mut textures = Vec::new();
    for attachment in descriptor.color_attachments.iter().flatten() {
        textures.push(attachment.view.texture());
        if let Some(resolve_target) = &attachment.resolve_target {
            textures.push(resolve_target.texture());
        }
    }
    if let Some(attachment) = &descriptor.depth_stencil_attachment {
        textures.push(attachment.view.texture());
    }
    textures
}

pub(crate) fn render_pass_attachment_scope_uses(
    descriptor: &RenderPassDescriptor,
) -> Vec<TextureScopeUse> {
    let mut uses = Vec::new();
    for attachment in descriptor.color_attachments.iter().flatten() {
        uses.push(texture_attachment_scope_use(
            &attachment.view,
            TextureAccess::AttachmentWrite,
            TextureAspectMask::COLOR,
            attachment.depth_slice,
        ));
        if let Some(resolve_target) = &attachment.resolve_target {
            uses.push(texture_attachment_scope_use(
                resolve_target,
                TextureAccess::AttachmentWrite,
                TextureAspectMask::COLOR,
                None,
            ));
        }
    }
    if let Some(attachment) = &descriptor.depth_stencil_attachment {
        let view = &attachment.view;
        let Some(caps) = view.texture().view_format_caps(view.format()) else {
            return uses;
        };
        if caps.aspects.depth {
            uses.push(texture_attachment_scope_use(
                view,
                if attachment.depth_read_only {
                    TextureAccess::Read
                } else {
                    TextureAccess::AttachmentWrite
                },
                TextureAspectMask::DEPTH,
                None,
            ));
        }
        if caps.aspects.stencil {
            uses.push(texture_attachment_scope_use(
                view,
                if attachment.stencil_read_only {
                    TextureAccess::Read
                } else {
                    TextureAccess::AttachmentWrite
                },
                TextureAspectMask::STENCIL,
                None,
            ));
        }
    }
    uses
}

/// Returns render pass query sets.
pub(crate) fn render_pass_query_sets(descriptor: &RenderPassDescriptor) -> Vec<QuerySet> {
    let mut query_sets = Vec::new();
    if let Some(query_set) = &descriptor.occlusion_query_set {
        query_sets.push(query_set.clone());
    }
    if let Some(timestamp_writes) = &descriptor.timestamp_writes {
        query_sets.push(timestamp_writes.query_set.clone());
    }
    query_sets
}

/// Returns render pass color executions.
pub(crate) fn render_pass_color_executions(
    descriptor: &RenderPassDescriptor,
) -> Vec<Option<RenderPassColorExecution>> {
    descriptor
        .color_attachments
        .iter()
        .map(|attachment| {
            attachment.as_ref().map(|attachment| {
                let resolve_target = attachment.resolve_target.as_ref();
                RenderPassColorExecution {
                    texture: attachment.view.texture(),
                    view_format: attachment.view.format(),
                    resolve_target: resolve_target.map(|view| view.texture()),
                    resolve_view_format: resolve_target.map(|view| view.format()),
                    mip_level: attachment.view.base_mip_level(),
                    array_layer: attachment.view.base_array_layer(),
                    depth_slice: attachment.depth_slice.unwrap_or(0),
                    resolve_mip_level: resolve_target.map_or(0, |view| view.base_mip_level()),
                    resolve_array_layer: resolve_target.map_or(0, |view| view.base_array_layer()),
                    load_op: attachment.load_op,
                    store_op: attachment.store_op,
                    clear_value: attachment.clear_value,
                }
            })
        })
        .collect()
}

/// Returns render pass depth-stencil execution.
pub(crate) fn render_pass_depth_stencil_execution(
    descriptor: &RenderPassDescriptor,
) -> Option<RenderPassDepthStencilExecution> {
    descriptor
        .depth_stencil_attachment
        .as_ref()
        .map(|attachment| RenderPassDepthStencilExecution {
            texture: attachment.view.texture(),
            format: attachment.view.format(),
            mip_level: attachment.view.base_mip_level(),
            array_layer: attachment.view.base_array_layer(),
            depth_load_op: attachment.depth_load_op,
            depth_store_op: attachment.depth_store_op,
            depth_clear_value: attachment.depth_clear_value,
            depth_read_only: attachment.depth_read_only,
            stencil_load_op: attachment.stencil_load_op,
            stencil_store_op: attachment.stencil_store_op,
            stencil_clear_value: attachment.stencil_clear_value,
            stencil_read_only: attachment.stencil_read_only,
        })
}

/// Returns the render pass attachment extent used by dynamic state validation.
pub(crate) fn render_pass_extent(descriptor: &RenderPassDescriptor) -> Option<Extent3d> {
    descriptor
        .color_attachments
        .iter()
        .flatten()
        .map(|attachment| attachment.view.render_extent())
        .next()
        .or_else(|| {
            descriptor
                .depth_stencil_attachment
                .as_ref()
                .map(|attachment| attachment.view.render_extent())
        })
}

/// Validates color attachment and returns a descriptive error on failure.
pub(crate) fn validate_color_attachment(
    attachment: &RenderPassColorAttachment,
    features: &FeatureSet,
) -> Result<(), String> {
    let Some(format_caps) = attachment.view.format().caps(features) else {
        return Err("render pass color attachment format must be supported".to_owned());
    };
    if !attachment
        .view
        .usage()
        .contains(TextureUsage::RENDER_ATTACHMENT)
    {
        return Err("render pass color attachment requires RenderAttachment usage".to_owned());
    }
    if !format_caps.aspects.color || !format_caps.renderable {
        return Err("render pass color attachment format must be color-renderable".to_owned());
    }
    if attachment.load_op == LoadOp::Undefined {
        return Err("render pass color attachment loadOp must be set".to_owned());
    }
    if attachment.store_op == StoreOp::Undefined {
        return Err("render pass color attachment storeOp must be set".to_owned());
    }
    if attachment
        .view
        .usage()
        .contains(TextureUsage::TRANSIENT_ATTACHMENT)
    {
        if attachment.load_op != LoadOp::Clear {
            return Err("render pass transient color attachment loadOp must be clear".to_owned());
        }
        if attachment.store_op != StoreOp::Discard {
            return Err(
                "render pass transient color attachment storeOp must be discard".to_owned(),
            );
        }
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
        return Err("render pass color clearValue components must be finite".to_owned());
    }
    Ok(())
}

/// Validates color attachment depthSlice rules.
pub(crate) fn validate_color_attachment_depth_slice(
    attachment: &RenderPassColorAttachment,
) -> Result<(), String> {
    let texture = attachment.view.texture();
    match texture.dimension() {
        TextureDimension::D3 => {
            let Some(depth_slice) = attachment.depth_slice else {
                return Err("render pass 3D color attachment requires depthSlice".to_owned());
            };
            let subresource = texture.subresource_size(attachment.view.base_mip_level());
            if depth_slice >= subresource.depth_or_array_layers {
                return Err("render pass color attachment depthSlice is out of bounds".to_owned());
            }
            Ok(())
        }
        TextureDimension::D1 | TextureDimension::D2 => {
            if attachment.depth_slice.is_some() {
                return Err(
                    "render pass non-3D color attachment must not set depthSlice".to_owned(),
                );
            }
            Ok(())
        }
    }
}

fn validate_color_attachment_overlap(
    attachment: &RenderPassColorAttachment,
    color_subresources: &mut BTreeSet<(usize, u32, Option<u32>)>,
) -> Result<(), String> {
    let texture = attachment.view.texture();
    if texture.dimension() != TextureDimension::D3 {
        return Ok(());
    }
    let texture_id = Arc::as_ptr(&texture.inner) as usize;
    let key = (
        texture_id,
        attachment.view.base_mip_level(),
        attachment.depth_slice,
    );
    if !color_subresources.insert(key) {
        return Err("render pass color attachments overlap the same subresource".to_owned());
    }
    Ok(())
}

fn validate_single_mip_attachment_view(view: &TextureView, label: &str) -> Result<(), String> {
    if view.mip_level_count() != 1 {
        return Err(format!("{label} view mipLevelCount must be one"));
    }
    Ok(())
}

/// Validates depth stencil attachment and returns a descriptive error on failure.
pub(crate) fn validate_depth_stencil_attachment(
    attachment: &RenderPassDepthStencilAttachment,
    features: &FeatureSet,
) -> Result<(), String> {
    let Some(format_caps) = attachment.view.format().caps(features) else {
        return Err("render pass depth-stencil attachment format must be supported".to_owned());
    };
    if !attachment
        .view
        .usage()
        .contains(TextureUsage::RENDER_ATTACHMENT)
    {
        return Err(
            "render pass depth-stencil attachment requires RenderAttachment usage".to_owned(),
        );
    }
    if !format_caps.aspects.depth && !format_caps.aspects.stencil {
        return Err(
            "render pass depth-stencil attachment format must have depth or stencil aspect"
                .to_owned(),
        );
    }
    if format_caps.aspects.depth {
        if attachment.depth_read_only {
            if attachment.depth_load_op != LoadOp::Undefined
                || attachment.depth_store_op != StoreOp::Undefined
            {
                return Err(
                    "render pass read-only depth attachment must not set depth load/store ops"
                        .to_owned(),
                );
            }
        } else if attachment.depth_load_op == LoadOp::Undefined {
            return Err("render pass depth loadOp must be set".to_owned());
        } else if attachment.depth_store_op == StoreOp::Undefined {
            return Err("render pass depth storeOp must be set".to_owned());
        }
        if attachment
            .view
            .usage()
            .contains(TextureUsage::TRANSIENT_ATTACHMENT)
        {
            if attachment.depth_load_op != LoadOp::Clear {
                return Err(
                    "render pass transient depth attachment depthLoadOp must be clear".to_owned(),
                );
            }
            if attachment.depth_store_op != StoreOp::Discard {
                return Err(
                    "render pass transient depth attachment depthStoreOp must be discard"
                        .to_owned(),
                );
            }
        }
        if attachment.depth_load_op == LoadOp::Clear
            && (!attachment.depth_clear_value.is_finite()
                || !(0.0..=1.0).contains(&attachment.depth_clear_value))
        {
            return Err("render pass depth clear value must be finite and in [0, 1]".to_owned());
        }
    } else if attachment.depth_load_op != LoadOp::Undefined
        || attachment.depth_store_op != StoreOp::Undefined
    {
        return Err(
            "render pass non-depth attachment must not set depth load/store ops".to_owned(),
        );
    }
    if format_caps.aspects.stencil {
        if attachment.stencil_read_only {
            if attachment.stencil_load_op != LoadOp::Undefined
                || attachment.stencil_store_op != StoreOp::Undefined
            {
                return Err(
                    "render pass read-only stencil attachment must not set stencil load/store ops"
                        .to_owned(),
                );
            }
        } else if attachment.stencil_load_op == LoadOp::Undefined {
            return Err("render pass stencil loadOp must be set".to_owned());
        } else if attachment.stencil_store_op == StoreOp::Undefined {
            return Err("render pass stencil storeOp must be set".to_owned());
        }
        if attachment
            .view
            .usage()
            .contains(TextureUsage::TRANSIENT_ATTACHMENT)
        {
            if attachment.stencil_load_op != LoadOp::Clear {
                return Err(
                    "render pass transient stencil attachment stencilLoadOp must be clear"
                        .to_owned(),
                );
            }
            if attachment.stencil_store_op != StoreOp::Discard {
                return Err(
                    "render pass transient stencil attachment stencilStoreOp must be discard"
                        .to_owned(),
                );
            }
        }
    } else if attachment.stencil_load_op != LoadOp::Undefined
        || attachment.stencil_store_op != StoreOp::Undefined
    {
        return Err(
            "render pass non-stencil attachment must not set stencil load/store ops".to_owned(),
        );
    }
    Ok(())
}

/// Validates render attachment common and returns a descriptive error on failure.
pub(crate) fn validate_render_attachment_common(
    view: &TextureView,
    render_extent: &mut Option<(u32, u32)>,
    sample_count: &mut Option<u32>,
    label: &str,
) -> Result<(), String> {
    if view.is_error() {
        return Err(format!("{label} view must not be an error view"));
    }
    if !view.swizzle().is_identity() {
        return Err(format!(
            "{label} must not use a component-swizzled texture view"
        ));
    }
    if view.dimension() != TextureViewDimension::D3 && view.array_layer_count() != 1 {
        return Err(format!("{label} view arrayLayerCount must be one"));
    }
    let extent = view.render_extent();
    let size = (extent.width, extent.height);
    if let Some(expected) = *render_extent {
        if expected != size {
            return Err("render pass attachments must have matching sizes".to_owned());
        }
    } else {
        *render_extent = Some(size);
    }

    let view_sample_count = view.texture().sample_count();
    if let Some(expected) = *sample_count {
        if expected != view_sample_count {
            return Err("render pass attachments must have matching sample counts".to_owned());
        }
    } else {
        *sample_count = Some(view_sample_count);
    }
    Ok(())
}

/// Validates resolve target and returns a descriptive error on failure.
pub(crate) fn validate_resolve_target(
    color_view: &TextureView,
    resolve_target: &TextureView,
) -> Result<(), String> {
    let color_texture = color_view.texture();
    let resolve_texture = resolve_target.texture();
    if color_texture.sample_count() <= 1 {
        return Err(
            "render pass resolveTarget requires a multisampled color attachment".to_owned(),
        );
    }
    if resolve_target.is_error() {
        return Err("render pass resolveTarget view must not be an error view".to_owned());
    }
    if !resolve_target.swizzle().is_identity() {
        return Err(
            "render pass resolveTarget must not use a component-swizzled texture view".to_owned(),
        );
    }
    if !resolve_texture
        .usage()
        .contains(TextureUsage::RENDER_ATTACHMENT)
    {
        return Err("render pass resolveTarget requires RenderAttachment usage".to_owned());
    }
    if resolve_texture.sample_count() != 1 {
        return Err("render pass resolveTarget sampleCount must be one".to_owned());
    }
    if color_view.format() != resolve_target.format() {
        return Err("render pass resolveTarget format must match the color attachment".to_owned());
    }
    let Some(caps) = color_texture.view_format_caps(color_view.format()) else {
        return Err("render pass resolveTarget format must be supported".to_owned());
    };
    if caps.output_class != Some(FormatOutputClass::Float) || !color_view.format().is_resolvable() {
        return Err("render pass resolveTarget format must be resolvable".to_owned());
    }
    if resolve_target.array_layer_count() != 1 {
        return Err("render pass resolveTarget view arrayLayerCount must be one".to_owned());
    }
    if color_view.render_extent() != resolve_target.render_extent() {
        return Err("render pass resolveTarget size must match the color attachment".to_owned());
    }
    Ok(())
}

/// Validates buffer texture copy and returns a descriptive error on failure.
pub(crate) fn validate_buffer_texture_copy(
    buffer_copy: TexelCopyBufferInfo,
    required_buffer_usage: BufferUsage,
    texture_copy: TexelCopyTextureInfo,
    required_texture_usage: TextureUsage,
    copy_size: Extent3d,
    label: &str,
    encoder_device: Option<&Device>,
) -> Result<(), String> {
    if let (Some(encoder_device), Some(buffer_device)) =
        (encoder_device, buffer_copy.device.as_ref())
    {
        if !encoder_device.same(buffer_device) {
            return Err(format!("{label} buffer must belong to the same device"));
        }
    }
    let buffer = buffer_copy.buffer;
    let texture = texture_copy.texture;
    if buffer.is_error() || texture.is_error() {
        return Err(format!("{label} cannot use an error resource"));
    }
    if !buffer.usage().contains(required_buffer_usage) {
        return Err(format!("{label} buffer has invalid usage"));
    }
    if !texture.usage().contains(required_texture_usage) {
        return Err(format!("{label} texture has invalid usage"));
    }
    if texture.sample_count() != 1 {
        return Err(format!("{label} texture sampleCount must be one"));
    }

    let format_caps = validate_texture_copy_subresource(
        &texture,
        texture_copy.mip_level,
        texture_copy.origin,
        copy_size,
        texture_copy.aspect,
        label,
        true,
    )?;
    if format_caps.aspects.depth
        && format_caps.aspects.stencil
        && texture_copy.aspect == TextureAspect::All
    {
        return Err(format!(
            "{label} of a combined depth-stencil format requires a single aspect"
        ));
    }
    let writing_texture = required_texture_usage == TextureUsage::COPY_DST;
    if (format_caps.aspects.depth || format_caps.aspects.stencil)
        && !crate::copy::depth_stencil_copy_allowed(
            texture.format(),
            texture_copy.aspect,
            writing_texture,
        )
    {
        return Err(format!(
            "{label} depth/stencil format does not support this copy aspect/usage"
        ));
    }
    let offset_alignment = if format_caps.aspects.depth || format_caps.aspects.stencil {
        4
    } else {
        u64::from(crate::copy::texel_copy_block_size(
            format_caps,
            texture_copy.aspect,
        ))
    };
    if !buffer_copy.layout.offset.is_multiple_of(offset_alignment) {
        return Err(format!(
            "{label} buffer offset must be a multiple of the texel block size"
        ));
    }
    let required_bytes = crate::copy::validate_texel_copy_layout(
        format_caps,
        texture_copy.aspect,
        copy_size,
        buffer_copy.layout,
        label,
        true,
    )?;
    validate_buffer_range(
        buffer_copy.layout.offset,
        required_bytes,
        buffer.size(),
        label,
    )
}

/// Validates texture to texture copy and returns a descriptive error on failure.
pub(crate) fn validate_texture_to_texture_copy(
    source_copy: TexelCopyTextureInfo,
    destination_copy: TexelCopyTextureInfo,
    copy_size: Extent3d,
) -> Result<(), String> {
    let source = source_copy.texture;
    let destination = destination_copy.texture;
    if source.is_error() || destination.is_error() {
        return Err("copy texture to texture cannot use an error texture".to_owned());
    }
    if !source.usage().contains(TextureUsage::COPY_SRC) {
        return Err("copy texture source must have CopySrc usage".to_owned());
    }
    if !destination.usage().contains(TextureUsage::COPY_DST) {
        return Err("copy texture destination must have CopyDst usage".to_owned());
    }
    if !texture_formats_copy_compatible(source.format(), destination.format()) {
        return Err("copy texture formats are not copy-compatible".to_owned());
    }
    if source.sample_count() != destination.sample_count() {
        return Err("copy texture sample counts must match".to_owned());
    }

    let source_caps = validate_texture_copy_subresource(
        &source,
        source_copy.mip_level,
        source_copy.origin,
        copy_size,
        source_copy.aspect,
        "copy texture source",
        false,
    )?;
    let destination_caps = validate_texture_copy_subresource(
        &destination,
        destination_copy.mip_level,
        destination_copy.origin,
        copy_size,
        destination_copy.aspect,
        "copy texture destination",
        false,
    )?;
    if source_caps.aspects.depth
        && source_caps.aspects.stencil
        && source_copy.aspect != TextureAspect::All
    {
        return Err(
            "copy texture to texture combined depth-stencil source requires All aspect".to_owned(),
        );
    }
    if destination_caps.aspects.depth
        && destination_caps.aspects.stencil
        && destination_copy.aspect != TextureAspect::All
    {
        return Err(
            "copy texture to texture combined depth-stencil destination requires All aspect"
                .to_owned(),
        );
    }
    if source.sample_count() > 1
        && (!origin_is_zero(source_copy.origin)
            || !origin_is_zero(destination_copy.origin)
            || copy_size != source.subresource_size(source_copy.mip_level)
            || copy_size != destination.subresource_size(destination_copy.mip_level))
    {
        return Err("copy texture multisampled copies must cover the full subresource".to_owned());
    }
    if source.same(&destination) {
        validate_same_texture_copy(
            &source,
            source_copy.mip_level,
            source_copy.origin,
            destination_copy.mip_level,
            destination_copy.origin,
            copy_size,
        )?;
    }

    Ok(())
}

/// Validates texture copy subresource and returns a descriptive error on failure.
pub(crate) fn validate_texture_copy_subresource(
    texture: &Texture,
    mip_level: u32,
    origin: Origin3d,
    copy_size: Extent3d,
    aspect: TextureAspect,
    label: &str,
    require_2d_single_layer: bool,
) -> Result<FormatCaps, String> {
    if mip_level >= texture.mip_level_count() {
        return Err(format!("{label} mipLevel is out of range"));
    }

    let Some(format_caps) = texture.format_caps() else {
        return Err(format!("{label} format must not be Undefined"));
    };
    validate_copy_aspect(format_caps, aspect, label)?;

    let subresource = texture.subresource_size(mip_level);
    let physical_width = div_ceil_u32(subresource.width, format_caps.block_w)
        .checked_mul(format_caps.block_w)
        .ok_or_else(|| format!("{label} subresource width overflows"))?;
    let physical_height = div_ceil_u32(subresource.height, format_caps.block_h)
        .checked_mul(format_caps.block_h)
        .ok_or_else(|| format!("{label} subresource height overflows"))?;
    let empty_copy =
        copy_size.width == 0 || copy_size.height == 0 || copy_size.depth_or_array_layers == 0;
    if origin
        .x
        .checked_add(copy_size.width)
        .is_none_or(|end| end > physical_width)
        || origin
            .y
            .checked_add(copy_size.height)
            .is_none_or(|end| end > physical_height)
        || origin
            .z
            .checked_add(copy_size.depth_or_array_layers)
            .is_none_or(|end| end > subresource.depth_or_array_layers)
    {
        return Err(format!("{label} range exceeds the texture subresource"));
    }
    if require_2d_single_layer
        && texture.dimension() == TextureDimension::D2
        && texture.size().depth_or_array_layers == 1
        && !empty_copy
        && copy_size.depth_or_array_layers != 1
    {
        return Err(format!(
            "{label} 2D copies require depthOrArrayLayers to be one"
        ));
    }
    if !origin.x.is_multiple_of(format_caps.block_w)
        || !origin.y.is_multiple_of(format_caps.block_h)
    {
        return Err(format!("{label} origin must be texel block aligned"));
    }
    if !copy_size.width.is_multiple_of(format_caps.block_w)
        || !copy_size.height.is_multiple_of(format_caps.block_h)
    {
        return Err(format!("{label} copy size must be texel block aligned"));
    }
    // The depth or stencil aspect can only be copied as a whole 2D subresource:
    // full mip width/height at a zero x/y origin. A range of array layers
    // (non-zero `origin.z` / `copy_size.depth_or_array_layers > 1`) is allowed —
    // each layer is its own 2D subresource — and is bounds-checked above. This
    // matches WebGPU for buffer copies and texture-to-texture copies alike.
    if (format_caps.aspects.depth || format_caps.aspects.stencil)
        && !empty_copy
        && (origin.x != 0
            || origin.y != 0
            || copy_size.width != subresource.width
            || copy_size.height != subresource.height)
    {
        return Err(format!(
            "{label} depth/stencil copies must cover the full 2D subresource"
        ));
    }

    Ok(format_caps)
}

/// Validates copy aspect and returns a descriptive error on failure.
pub(crate) fn validate_copy_aspect(
    format_caps: FormatCaps,
    aspect: TextureAspect,
    label: &str,
) -> Result<(), String> {
    match aspect {
        TextureAspect::All => Ok(()),
        TextureAspect::DepthOnly if format_caps.aspects.depth => Ok(()),
        TextureAspect::StencilOnly if format_caps.aspects.stencil => Ok(()),
        TextureAspect::DepthOnly => {
            Err(format!("{label} DepthOnly aspect requires a depth format"))
        }
        TextureAspect::StencilOnly => Err(format!(
            "{label} StencilOnly aspect requires a stencil format"
        )),
    }
}

/// Returns texture formats copy compatible.
pub(crate) fn texture_formats_copy_compatible(
    source: TextureFormat,
    destination: TextureFormat,
) -> bool {
    source == destination || source.srgb_pair() == Some(destination)
}

/// Returns origin is zero.
pub(crate) fn origin_is_zero(origin: Origin3d) -> bool {
    origin.x == 0 && origin.y == 0 && origin.z == 0
}

/// Validates same texture copy and returns a descriptive error on failure.
pub(crate) fn validate_same_texture_copy(
    texture: &Texture,
    source_mip_level: u32,
    source_origin: Origin3d,
    destination_mip_level: u32,
    destination_origin: Origin3d,
    copy_size: Extent3d,
) -> Result<(), String> {
    if copy_size.width == 0 || copy_size.height == 0 || copy_size.depth_or_array_layers == 0 {
        return Ok(());
    }
    if source_mip_level != destination_mip_level {
        return Ok(());
    }
    let source_end = source_origin
        .z
        .saturating_add(copy_size.depth_or_array_layers);
    let destination_end = destination_origin
        .z
        .saturating_add(copy_size.depth_or_array_layers);
    if texture.dimension() == TextureDimension::D3 {
        if source_origin.z < destination_end && destination_origin.z < source_end {
            return Err(
                "copy texture to texture same-texture 3D z ranges must not overlap".to_owned(),
            );
        }
        return Ok(());
    }

    if source_origin.z < destination_end && destination_origin.z < source_end {
        return Err(
            "copy texture to texture same-texture array layers must not overlap".to_owned(),
        );
    }
    Ok(())
}

/// Validates buffer range and returns a descriptive error on failure.
pub(crate) fn validate_buffer_range(
    offset: u64,
    size: u64,
    buffer_size: u64,
    label: &str,
) -> Result<(), String> {
    let Some(end) = offset.checked_add(size) else {
        return Err(format!("{label} overflows"));
    };
    if offset > buffer_size || end > buffer_size {
        return Err(format!("{label} exceeds buffer size"));
    }
    Ok(())
}

/// Returns record first error locked.
pub(crate) fn record_first_error_locked(
    state: &mut CommandEncoderState,
    message: impl Into<String>,
) {
    if state.first_error.is_none() {
        state.first_error = Some(message.into());
    }
}

/// Returns record first error option.
pub(crate) fn record_first_error_option(
    first_error: &mut Option<String>,
    message: impl Into<String>,
) {
    if first_error.is_none() {
        *first_error = Some(message.into());
    }
}

impl CommandBuffer {
    /// Creates a new instance.
    pub(crate) fn new(
        error_message: Option<String>,
        referenced_buffers: Vec<Arc<Buffer>>,
        referenced_textures: Vec<Texture>,
        referenced_query_sets: Vec<QuerySet>,
        command_ops: Vec<CommandExecution>,
    ) -> Self {
        Self {
            inner: Arc::new(CommandBufferInner {
                is_error: error_message.is_some(),
                error_message,
                referenced_buffers,
                referenced_textures,
                referenced_query_sets,
                command_ops,
                submitted: Mutex::new(false),
            }),
        }
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns the diagnostic message that made this command buffer an error object.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.inner.error_message.as_deref()
    }

    /// Creates an error command buffer with no recorded resources.
    pub(crate) fn new_error(message: String) -> Self {
        Self::new(
            Some(message),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    /// Returns the referenced buffers.
    pub(crate) fn referenced_buffers(&self) -> &[Arc<Buffer>] {
        &self.inner.referenced_buffers
    }

    /// Returns the referenced textures.
    pub(crate) fn referenced_textures(&self) -> &[Texture] {
        &self.inner.referenced_textures
    }

    /// Returns the referenced query sets.
    pub(crate) fn referenced_query_sets(&self) -> &[QuerySet] {
        &self.inner.referenced_query_sets
    }

    /// Returns the command ops.
    pub(crate) fn command_ops(&self) -> &[CommandExecution] {
        &self.inner.command_ops
    }

    /// Marks the produced command buffer as submitted, erroring if it was already submitted.
    pub(crate) fn mark_submitted(&self) -> Result<(), String> {
        let mut submitted = self.inner.submitted.lock();
        if *submitted {
            Err("command buffer cannot be submitted more than once".to_owned())
        } else {
            *submitted = true;
            Ok(())
        }
    }

    /// Returns true when this object is submitted.
    pub(crate) fn is_submitted(&self) -> bool {
        *self.inner.submitted.lock()
    }

    /// Returns true when both handles share the same backing object.
    pub(crate) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

    use std::sync::Arc;

    #[test]
    fn command_encoder_create_finish_idempotent_and_command_buffer_is_error_false() {
        let encoder = noop_device().create_command_encoder();

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert!(command_buffer.command_ops().is_empty());

        let (second, error) = encoder.finish();
        assert!(second.is_error());
        assert_eq!(
            error,
            Some("command encoder cannot be finished more than once".to_owned())
        );
    }

    #[test]
    fn command_encoder_debug_markers_and_validation_error() {
        let encoder = noop_device().create_command_encoder();

        assert_eq!(encoder.push_debug_group(), None);
        assert_eq!(encoder.insert_debug_marker(), None);
        assert_eq!(encoder.pop_debug_group(), None);
        assert_eq!(
            encoder.record_validation_error("forced encoder validation"),
            None
        );

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(error, Some("forced encoder validation".to_owned()));
    }

    #[test]
    fn render_pass_validation_checks_depth_slice_and_attachment_byte_limit() {
        let device = noop_device();
        let texture = Texture::new(
            TextureDescriptor {
                usage: TextureUsage::RENDER_ATTACHMENT,
                dimension: TextureDimension::D3,
                size: Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 4,
                },
                format: rgba8_unorm(),
                mip_level_count: 1,
                sample_count: 1,
                view_formats: Vec::new(),
            },
            None,
            false,
            FeatureSet::new(),
        );
        let view = TextureView::new(
            texture,
            ResolvedTextureViewDescriptor {
                format: rgba8_unorm(),
                dimension: TextureViewDimension::D3,
                base_mip_level: 0,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                aspect: TextureAspect::All,
                usage: TextureUsage::RENDER_ATTACHMENT,
                swizzle: TextureComponentSwizzle::default(),
            },
            false,
            None,
        );
        let mut descriptor = noop_render_pass_descriptor(Arc::new(view), None);
        let features = device.features();

        assert!(validate_render_pass_descriptor(&descriptor, &features, device.limits()).is_err());
        descriptor.color_attachments[0]
            .as_mut()
            .unwrap()
            .depth_slice = Some(4);
        assert!(validate_render_pass_descriptor(&descriptor, &features, device.limits()).is_err());
        descriptor.color_attachments[0]
            .as_mut()
            .unwrap()
            .depth_slice = Some(3);
        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &features, device.limits()),
            Ok(())
        );

        let mut tight_limits = device.limits();
        tight_limits.max_color_attachment_bytes_per_sample = 3;
        assert!(validate_render_pass_descriptor(&descriptor, &features, tight_limits).is_err());
    }

    #[test]
    fn render_pass_depth_stencil_read_only_and_absent_aspect_ops_match_cts_matrix() {
        let device = noop_device();
        let depth_stencil_view = render_attachment_view_with_format(
            &device,
            TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS_STENCIL8),
            1,
        );
        let mut descriptor = RenderPassDescriptor {
            max_color_attachments: device.limits().max_color_attachments,
            color_attachments: Vec::new(),
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_stencil_view,
                depth_load_op: LoadOp::Undefined,
                depth_store_op: StoreOp::Undefined,
                depth_clear_value: 0.0,
                depth_read_only: true,
                stencil_load_op: LoadOp::Clear,
                stencil_store_op: StoreOp::Discard,
                stencil_clear_value: 0,
                stencil_read_only: false,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            max_draw_count: 50_000_000,
        };
        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Ok(())
        );

        descriptor
            .depth_stencil_attachment
            .as_mut()
            .expect("depth-stencil attachment")
            .depth_load_op = LoadOp::Load;
        assert!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits())
                .is_err()
        );

        let depth_only_view = render_attachment_view_with_format(
            &device,
            TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS),
            1,
        );
        {
            let depth_only = descriptor
                .depth_stencil_attachment
                .as_mut()
                .expect("depth-stencil attachment");
            depth_only.view = depth_only_view;
            depth_only.depth_read_only = false;
            depth_only.depth_load_op = LoadOp::Clear;
            depth_only.depth_store_op = StoreOp::Discard;
            depth_only.stencil_load_op = LoadOp::Undefined;
            depth_only.stencil_store_op = StoreOp::Undefined;
        }
        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Ok(())
        );
        descriptor
            .depth_stencil_attachment
            .as_mut()
            .expect("depth-stencil attachment")
            .stencil_store_op = StoreOp::Store;
        assert!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits())
                .is_err()
        );
    }

    #[test]
    fn render_pass_color_attachment_byte_limit_sums_with_cts_alignment() {
        let device = noop_device();
        let formats = [
            TextureFormat::from_raw(TextureFormat::R8_UNORM),
            TextureFormat::from_raw(TextureFormat::R32_FLOAT),
            TextureFormat::from_raw(TextureFormat::RGBA8_UNORM),
            TextureFormat::from_raw(TextureFormat::RGBA32_FLOAT),
            TextureFormat::from_raw(TextureFormat::R8_UNORM),
        ];
        let mut descriptor = RenderPassDescriptor {
            max_color_attachments: device.limits().max_color_attachments,
            color_attachments: formats
                .iter()
                .copied()
                .map(|format| {
                    Some(RenderPassColorAttachment {
                        view: render_attachment_view_with_format(&device, format, 1),
                        depth_slice: None,
                        resolve_target: None,
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::Store,
                        clear_value: Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        },
                    })
                })
                .collect(),
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            max_draw_count: 50_000_000,
        };
        let mut limits = device.limits();
        limits.max_color_attachment_bytes_per_sample = 33;
        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), limits),
            Ok(())
        );
        limits.max_color_attachment_bytes_per_sample = 32;
        assert!(validate_render_pass_descriptor(&descriptor, &device.features(), limits).is_err());

        descriptor.color_attachments.pop();
        limits.max_color_attachment_bytes_per_sample = 32;
        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), limits),
            Ok(())
        );
    }

    #[test]
    fn render_pass_resolve_target_rejects_non_resolvable_snorm_format() {
        let device = noop_adapter()
            .create_device(None, &[crate::Feature::TextureFormatsTier1], "", "")
            .expect("Noop device should support texture-formats-tier1");
        let mut descriptor = noop_render_pass_descriptor(
            render_attachment_view_with_format(
                &device,
                TextureFormat::from_raw(TextureFormat::RGBA16_SNORM),
                4,
            ),
            None,
        );
        let snorm_resolve = render_attachment_view_with_format(
            &device,
            TextureFormat::from_raw(TextureFormat::RGBA16_SNORM),
            1,
        );
        descriptor.color_attachments[0]
            .as_mut()
            .expect("color attachment")
            .resolve_target = Some(snorm_resolve);
        assert!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits())
                .is_err()
        );

        descriptor.color_attachments[0]
            .as_mut()
            .expect("color attachment")
            .view = render_attachment_view_with_format(
            &device,
            TextureFormat::from_raw(TextureFormat::RGBA16_FLOAT),
            4,
        );
        let float_resolve = render_attachment_view_with_format(
            &device,
            TextureFormat::from_raw(TextureFormat::RGBA16_FLOAT),
            1,
        );
        descriptor.color_attachments[0]
            .as_mut()
            .expect("color attachment")
            .resolve_target = Some(float_resolve);
        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Ok(())
        );
    }

    #[test]
    fn render_attachment_validation_uses_view_usage_override() {
        let device = noop_device();
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
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
            usage: Some(TextureUsage::TEXTURE_BINDING),
            swizzle: None,
        });
        assert_eq!(error, None);
        let descriptor = noop_render_pass_descriptor(Arc::new(view), None);

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Err("render pass color attachment requires RenderAttachment usage".to_owned())
        );
    }

    #[test]
    fn render_pass_transient_color_attachment_requires_clear_and_discard() {
        let device = noop_device();
        let view = transient_render_attachment_view_with_format(&device, rgba8_unorm());
        let mut descriptor = noop_render_pass_descriptor(view, None);
        let attachment = descriptor.color_attachments[0]
            .as_mut()
            .expect("color attachment");
        attachment.load_op = LoadOp::Load;
        attachment.store_op = StoreOp::Discard;

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Err("render pass transient color attachment loadOp must be clear".to_owned())
        );

        let attachment = descriptor.color_attachments[0]
            .as_mut()
            .expect("color attachment");
        attachment.load_op = LoadOp::Clear;
        attachment.store_op = StoreOp::Store;

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Err("render pass transient color attachment storeOp must be discard".to_owned())
        );

        descriptor.color_attachments[0]
            .as_mut()
            .expect("color attachment")
            .store_op = StoreOp::Discard;

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Ok(())
        );
    }

    #[test]
    fn render_pass_transient_depth_stencil_attachment_requires_clear_and_discard() {
        let device = noop_device();
        let view = transient_render_attachment_view_with_format(
            &device,
            TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS_STENCIL8),
        );
        let mut descriptor = RenderPassDescriptor {
            max_color_attachments: device.limits().max_color_attachments,
            color_attachments: Vec::new(),
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view,
                depth_load_op: LoadOp::Load,
                depth_store_op: StoreOp::Discard,
                depth_clear_value: 0.0,
                depth_read_only: false,
                stencil_load_op: LoadOp::Clear,
                stencil_store_op: StoreOp::Discard,
                stencil_clear_value: 0,
                stencil_read_only: false,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            max_draw_count: 50_000_000,
        };

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Err("render pass transient depth attachment depthLoadOp must be clear".to_owned())
        );

        let attachment = descriptor
            .depth_stencil_attachment
            .as_mut()
            .expect("depth-stencil attachment");
        attachment.depth_load_op = LoadOp::Clear;
        attachment.depth_store_op = StoreOp::Store;

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Err("render pass transient depth attachment depthStoreOp must be discard".to_owned())
        );

        let attachment = descriptor
            .depth_stencil_attachment
            .as_mut()
            .expect("depth-stencil attachment");
        attachment.depth_store_op = StoreOp::Discard;
        attachment.stencil_load_op = LoadOp::Load;

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Err("render pass transient stencil attachment stencilLoadOp must be clear".to_owned())
        );

        let attachment = descriptor
            .depth_stencil_attachment
            .as_mut()
            .expect("depth-stencil attachment");
        attachment.stencil_load_op = LoadOp::Clear;
        attachment.stencil_store_op = StoreOp::Store;

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Err(
                "render pass transient stencil attachment stencilStoreOp must be discard"
                    .to_owned()
            )
        );

        descriptor
            .depth_stencil_attachment
            .as_mut()
            .expect("depth-stencil attachment")
            .stencil_store_op = StoreOp::Discard;

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Ok(())
        );
    }

    #[test]
    fn render_pass_transient_read_only_depth_attachment_is_rejected() {
        let device = noop_device();
        let view = transient_render_attachment_view_with_format(
            &device,
            TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS),
        );
        let descriptor = RenderPassDescriptor {
            max_color_attachments: device.limits().max_color_attachments,
            color_attachments: Vec::new(),
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view,
                depth_load_op: LoadOp::Undefined,
                depth_store_op: StoreOp::Undefined,
                depth_clear_value: 0.0,
                depth_read_only: true,
                stencil_load_op: LoadOp::Undefined,
                stencil_store_op: StoreOp::Undefined,
                stencil_clear_value: 0,
                stencil_read_only: true,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            max_draw_count: 50_000_000,
        };

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Err("render pass transient depth attachment depthLoadOp must be clear".to_owned())
        );
    }

    #[test]
    fn render_pass_color_attachment_rejects_component_swizzled_view() {
        let device = noop_adapter()
            .create_device(None, &[crate::Feature::TextureComponentSwizzle], "", "")
            .expect("Noop adapter should support texture component swizzle");
        let view = swizzled_render_attachment_view(&device, rgba8_unorm(), 1);
        let descriptor = noop_render_pass_descriptor(view, None);

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Err(
                "render pass color attachment must not use a component-swizzled texture view"
                    .to_owned()
            )
        );
    }

    #[test]
    fn render_pass_depth_stencil_attachment_rejects_component_swizzled_view() {
        let device = noop_adapter()
            .create_device(None, &[crate::Feature::TextureComponentSwizzle], "", "")
            .expect("Noop adapter should support texture component swizzle");
        let view = swizzled_render_attachment_view(
            &device,
            TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS),
            1,
        );
        let descriptor = RenderPassDescriptor {
            max_color_attachments: device.limits().max_color_attachments,
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
                stencil_read_only: true,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            max_draw_count: 50_000_000,
        };

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Err(
                "render pass depth-stencil attachment must not use a component-swizzled texture view"
                    .to_owned()
            )
        );
    }

    #[test]
    fn render_pass_resolve_target_rejects_component_swizzled_view() {
        let device = noop_adapter()
            .create_device(None, &[crate::Feature::TextureComponentSwizzle], "", "")
            .expect("Noop adapter should support texture component swizzle");
        let color = render_attachment_view_with_format(&device, rgba8_unorm(), 4);
        let resolve = swizzled_render_attachment_view(&device, rgba8_unorm(), 1);
        let mut descriptor = noop_render_pass_descriptor(color, None);
        descriptor.color_attachments[0]
            .as_mut()
            .expect("color attachment")
            .resolve_target = Some(resolve);

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Err(
                "render pass resolveTarget must not use a component-swizzled texture view"
                    .to_owned()
            )
        );
    }

    #[test]
    fn render_pass_identity_swizzle_attachment_is_valid() {
        let device = noop_adapter()
            .create_device(None, &[crate::Feature::TextureComponentSwizzle], "", "")
            .expect("Noop adapter should support texture component swizzle");
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
            swizzle: Some(TextureComponentSwizzle::default()),
        });
        assert_eq!(error, None);
        let descriptor = noop_render_pass_descriptor(Arc::new(view), None);

        assert_eq!(
            validate_render_pass_descriptor(&descriptor, &device.features(), device.limits()),
            Ok(())
        );
    }

    #[test]
    fn render_pass_color_executions_preserve_reinterpreted_view_formats() {
        let device = noop_device();
        let texture_format = TextureFormat::from_raw(TextureFormat::RGBA8_UNORM_SRGB);
        let view_format = rgba8_unorm();
        let view = reinterpreted_render_attachment_view(&device, texture_format, view_format, 1);
        let descriptor = noop_render_pass_descriptor(view, None);

        let executions = render_pass_color_executions(&descriptor);
        let execution = executions[0].as_ref().expect("color execution");
        assert_eq!(execution.texture.format(), texture_format);
        assert_eq!(execution.view_format, view_format);
        assert_eq!(execution.resolve_view_format, None);

        let color_view =
            reinterpreted_render_attachment_view(&device, texture_format, view_format, 4);
        let resolve_view =
            reinterpreted_render_attachment_view(&device, texture_format, view_format, 1);
        let mut descriptor = noop_render_pass_descriptor(color_view, None);
        descriptor.color_attachments[0]
            .as_mut()
            .expect("color attachment")
            .resolve_target = Some(resolve_view);

        let executions = render_pass_color_executions(&descriptor);
        let execution = executions[0].as_ref().expect("color execution");
        assert_eq!(execution.texture.format(), texture_format);
        assert_eq!(execution.view_format, view_format);
        assert_eq!(
            execution
                .resolve_target
                .as_ref()
                .expect("resolve target")
                .format(),
            texture_format
        );
        assert_eq!(execution.resolve_view_format, Some(view_format));
    }

    fn reinterpreted_render_attachment_view(
        device: &Device,
        texture_format: TextureFormat,
        view_format: TextureFormat,
        sample_count: u32,
    ) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: texture_format,
            mip_level_count: 1,
            sample_count,
            view_formats: vec![view_format],
        });
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: Some(view_format),
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

    fn render_attachment_view_with_format(
        device: &Device,
        format: TextureFormat,
        sample_count: u32,
    ) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format,
            mip_level_count: 1,
            sample_count,
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

    fn transient_render_attachment_view_with_format(
        device: &Device,
        format: TextureFormat,
    ) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TRANSIENT_ATTACHMENT,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format,
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

    fn swizzled_render_attachment_view(
        device: &Device,
        format: TextureFormat,
        sample_count: u32,
    ) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format,
            mip_level_count: 1,
            sample_count,
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
            swizzle: Some(TextureComponentSwizzle {
                r: ComponentSwizzle::G,
                ..TextureComponentSwizzle::default()
            }),
        });
        assert_eq!(error, None);
        Arc::new(view)
    }

    #[test]
    fn command_encoder_buffer_copies_clear_and_write_validate_offsets() {
        let device = noop_device();
        let source = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 32,
            mapped_at_creation: false,
        }));
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 32,
            mapped_at_creation: false,
        }));
        let encoder = device.create_command_encoder();

        assert_eq!(
            encoder.copy_buffer_to_buffer(source.clone(), 0, destination.clone(), 0, 16),
            None
        );
        assert_eq!(encoder.clear_buffer(destination.clone(), 0, 16), None);
        assert_eq!(encoder.write_buffer(destination.clone(), 0, 16), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 2);
        assert!(matches!(
            &command_buffer.command_ops()[0],
            CommandExecution::BufferCopy(copy)
                if copy.source.same(&source)
                    && copy.source_offset == 0
                    && copy.destination.same(&destination)
                    && copy.destination_offset == 0
                    && copy.size == 16
        ));
        assert!(matches!(
            &command_buffer.command_ops()[1],
            CommandExecution::BufferClear(clear)
                if clear.buffer.same(&destination) && clear.offset == 0 && clear.size == 16
        ));

        let invalid = device.create_command_encoder();
        assert_eq!(
            invalid.copy_buffer_to_buffer(source, 2, destination, 0, 4),
            None
        );
        let (command_buffer, error) = invalid.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("copy source offset must be 4-byte aligned".to_owned())
        );

        let missing_copy_dst = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 32,
            mapped_at_creation: false,
        }));
        let invalid = device.create_command_encoder();
        assert_eq!(invalid.clear_buffer(missing_copy_dst, 0, 4), None);
        let (command_buffer, error) = invalid.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("clear buffer requires CopyDst usage".to_owned())
        );
    }

    #[test]
    fn destroyed_recorded_buffers_fail_at_submit_not_finish() {
        let device = noop_device();
        let source = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 16,
            mapped_at_creation: false,
        }));
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 16,
            mapped_at_creation: false,
        }));
        destination.destroy();
        let encoder = device.create_command_encoder();

        assert_eq!(
            encoder.copy_buffer_to_buffer(source, 0, destination, 0, 16),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());

        let error = device
            .queue()
            .submit(&[Arc::new(command_buffer)])
            .expect("destroyed buffer should fail at submit");
        assert_eq!(error.message, "queue submit cannot use a destroyed buffer");
    }

    #[test]
    fn error_recorded_buffers_still_fail_at_finish() {
        let device = noop_device();
        let source = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::NONE,
            size: 16,
            mapped_at_creation: false,
        }));
        assert!(source.is_error());
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 16,
            mapped_at_creation: false,
        }));
        let encoder = device.create_command_encoder();

        assert_eq!(
            encoder.copy_buffer_to_buffer(source, 0, destination, 0, 16),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("copy buffer command cannot use an error buffer".to_owned())
        );
    }

    #[test]
    fn destroyed_query_set_resolve_fails_at_submit_not_finish() {
        let device = noop_device();
        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "queries".to_owned(),
            kind: QueryType::Occlusion,
            count: 2,
        });
        assert_eq!(error, None);
        let query_set = Arc::new(query_set);
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::QUERY_RESOLVE,
            size: 16,
            mapped_at_creation: false,
        }));
        query_set.destroy();
        let encoder = device.create_command_encoder();

        assert_eq!(
            encoder.resolve_query_set(query_set, 0, 1, destination, 0),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());

        let error = device
            .queue()
            .submit(&[Arc::new(command_buffer)])
            .expect("destroyed resolve resources should fail at submit");
        assert_eq!(
            error.message,
            "queue submit cannot use a destroyed query set"
        );
    }

    #[test]
    fn destroyed_resolve_destination_fails_at_submit_not_finish() {
        let device = noop_device();
        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "queries".to_owned(),
            kind: QueryType::Occlusion,
            count: 2,
        });
        assert_eq!(error, None);
        let query_set = Arc::new(query_set);
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::QUERY_RESOLVE,
            size: 16,
            mapped_at_creation: false,
        }));
        destination.destroy();
        let encoder = device.create_command_encoder();

        assert_eq!(
            encoder.resolve_query_set(query_set, 0, 1, destination, 0),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());

        let error = device
            .queue()
            .submit(&[Arc::new(command_buffer)])
            .expect("destroyed resolve resources should fail at submit");
        assert_eq!(error.message, "queue submit cannot use a destroyed buffer");
    }

    #[test]
    fn command_encoder_texture_copies_record_copy_commands() {
        let device = noop_device();
        let texture_a = Arc::new(device.create_texture(texture_descriptor_4x4()));
        let texture_b = Arc::new(device.create_texture(texture_descriptor_4x4()));
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            size: 1024,
            mapped_at_creation: false,
        }));
        let layout = TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(256),
            rows_per_image: None,
        };
        let size = Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        };
        let texture_info_a = TexelCopyTextureInfo {
            texture: texture_a,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::All,
        };
        let texture_info_b = TexelCopyTextureInfo {
            texture: texture_b,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::All,
        };
        let buffer_info = TexelCopyBufferInfo {
            buffer,
            device: None,
            layout,
        };
        let encoder = device.create_command_encoder();

        assert_eq!(
            encoder.copy_buffer_to_texture(buffer_info.clone(), texture_info_a.clone(), size),
            None
        );
        assert_eq!(
            encoder.copy_texture_to_buffer(texture_info_a.clone(), buffer_info, size),
            None
        );
        assert_eq!(
            encoder.copy_texture_to_texture(texture_info_a, texture_info_b, size),
            None
        );

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 3);
    }

    #[test]
    fn command_encoder_accepts_block_aligned_compressed_buffer_texture_copies() {
        let device = noop_adapter()
            .create_device(
                None,
                &[crate::adapter::Feature::TextureCompressionBc],
                "",
                "",
            )
            .expect("Noop compressed texture device");
        let texture = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::from_raw(TextureFormat::BC1_RGBA_UNORM),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            size: 1024,
            mapped_at_creation: false,
        }));
        let buffer_info = TexelCopyBufferInfo {
            buffer,
            device: None,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        };
        let texture_info = TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::All,
        };
        let size = Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        };
        let encoder = device.create_command_encoder();

        assert_eq!(
            encoder.copy_buffer_to_texture(buffer_info.clone(), texture_info.clone(), size),
            None
        );
        assert_eq!(
            encoder.copy_texture_to_buffer(texture_info, buffer_info, size),
            None
        );
    }

    #[test]
    fn validate_texture_copy_subresource_uses_physical_compressed_mip_bounds() {
        let device = noop_adapter()
            .create_device(
                None,
                &[crate::adapter::Feature::TextureCompressionBc],
                "",
                "",
            )
            .expect("Noop compressed texture device");
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 60,
                height: 8,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::from_raw(TextureFormat::BC7_RGBA_UNORM),
            mip_level_count: 2,
            sample_count: 1,
            view_formats: Vec::new(),
        });

        assert_eq!(
            validate_texture_copy_subresource(
                &texture,
                1,
                Origin3d { x: 0, y: 0, z: 0 },
                Extent3d {
                    width: 32,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                TextureAspect::All,
                "copy texture",
                true,
            ),
            Ok(texture.format_caps().expect("BC7 caps should be available"))
        );
        assert_eq!(
            validate_texture_copy_subresource(
                &texture,
                1,
                Origin3d { x: 0, y: 0, z: 0 },
                Extent3d {
                    width: 36,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                TextureAspect::All,
                "copy texture",
                true,
            ),
            Err("copy texture range exceeds the texture subresource".to_owned())
        );
    }

    #[test]
    fn copy_texture_to_texture_allows_same_3d_texture_disjoint_z_ranges_only() {
        let device = noop_device();
        let texture_3d = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            dimension: TextureDimension::D3,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 2,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let copy_size = Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        };

        let encoder = device.create_command_encoder();
        assert_eq!(
            encoder.copy_texture_to_texture(
                TexelCopyTextureInfo {
                    texture: texture_3d.clone(),
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                TexelCopyTextureInfo {
                    texture: texture_3d.clone(),
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 1 },
                    aspect: TextureAspect::All,
                },
                copy_size,
            ),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());

        let encoder = device.create_command_encoder();
        assert_eq!(
            encoder.copy_texture_to_texture(
                TexelCopyTextureInfo {
                    texture: texture_3d.clone(),
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                TexelCopyTextureInfo {
                    texture: texture_3d,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                copy_size,
            ),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("copy texture to texture same-texture 3D z ranges must not overlap".to_owned())
        );

        let texture_2d = Arc::new(device.create_texture(texture_descriptor_4x4()));
        let encoder = device.create_command_encoder();
        assert_eq!(
            encoder.copy_texture_to_texture(
                TexelCopyTextureInfo {
                    texture: texture_2d.clone(),
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                TexelCopyTextureInfo {
                    texture: texture_2d,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                copy_size,
            ),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("copy texture to texture same-texture array layers must not overlap".to_owned())
        );
    }

    #[test]
    fn texture_to_texture_copy_validates_aspects_independently() {
        let device = noop_device();
        let source = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_SRC,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::from_raw(TextureFormat::DEPTH32_FLOAT),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let destination = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::from_raw(TextureFormat::DEPTH32_FLOAT),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let size = Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        };
        let source_info = TexelCopyTextureInfo {
            texture: source.clone(),
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::DepthOnly,
        };
        let destination_info = TexelCopyTextureInfo {
            texture: destination.clone(),
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::All,
        };
        assert_eq!(
            validate_texture_to_texture_copy(source_info, destination_info, size),
            Ok(())
        );

        let invalid_source = TexelCopyTextureInfo {
            texture: source,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::StencilOnly,
        };
        let invalid_destination = TexelCopyTextureInfo {
            texture: destination,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::All,
        };
        assert_eq!(
            validate_texture_to_texture_copy(invalid_source, invalid_destination, size),
            Err("copy texture source StencilOnly aspect requires a stencil format".to_owned())
        );

        let combined_source = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_SRC,
            dimension: TextureDimension::D2,
            size,
            format: TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS_STENCIL8),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let combined_destination = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size,
            format: TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS_STENCIL8),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let combined_source_info = TexelCopyTextureInfo {
            texture: combined_source.clone(),
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::All,
        };
        let combined_destination_info = TexelCopyTextureInfo {
            texture: combined_destination.clone(),
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::All,
        };
        assert_eq!(
            validate_texture_to_texture_copy(
                combined_source_info.clone(),
                combined_destination_info.clone(),
                size,
            ),
            Ok(())
        );

        let depth_only_combined_source = TexelCopyTextureInfo {
            aspect: TextureAspect::DepthOnly,
            ..combined_source_info.clone()
        };
        assert_eq!(
            validate_texture_to_texture_copy(
                depth_only_combined_source,
                combined_destination_info.clone(),
                size,
            ),
            Err(
                "copy texture to texture combined depth-stencil source requires All aspect"
                    .to_owned()
            )
        );

        let stencil_only_combined_destination = TexelCopyTextureInfo {
            aspect: TextureAspect::StencilOnly,
            ..combined_destination_info
        };
        assert_eq!(
            validate_texture_to_texture_copy(
                combined_source_info,
                stencil_only_combined_destination,
                size,
            ),
            Err(
                "copy texture to texture combined depth-stencil destination requires All aspect"
                    .to_owned()
            )
        );
    }

    #[test]
    fn buffer_texture_copy_rejects_combined_depth_stencil_all_aspect() {
        let device = noop_device();
        let size = Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_SRC | TextureUsage::RENDER_ATTACHMENT,
            dimension: TextureDimension::D2,
            size,
            format: TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS_STENCIL8),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 256,
            mapped_at_creation: false,
        }));
        let buffer_info = TexelCopyBufferInfo {
            buffer,
            device: Some(device.clone()),
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        };
        let texture_info = |aspect| TexelCopyTextureInfo {
            texture: Arc::clone(&texture),
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect,
        };
        let encoder = device.create_command_encoder();

        assert_eq!(
            encoder.copy_texture_to_buffer(
                texture_info(TextureAspect::All),
                buffer_info.clone(),
                size
            ),
            None
        );
        let (_, error) = encoder.finish();
        assert_eq!(
            error,
            Some(
                "copy texture to buffer of a combined depth-stencil format requires a single aspect"
                    .to_owned()
            )
        );

        let encoder = device.create_command_encoder();
        assert_eq!(
            encoder.copy_texture_to_buffer(
                texture_info(TextureAspect::StencilOnly),
                buffer_info,
                size
            ),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn buffer_texture_copy_accepts_color_offset_aligned_to_block_size() {
        let device = noop_device();
        let texture = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::from_raw(TextureFormat::R8_UNORM),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 1024,
            mapped_at_creation: false,
        }));

        assert_eq!(
            validate_buffer_texture_copy(
                TexelCopyBufferInfo {
                    buffer,
                    device: Some(device.clone()),
                    layout: TexelCopyBufferLayout {
                        offset: 1,
                        bytes_per_row: Some(256),
                        rows_per_image: None,
                    },
                },
                BufferUsage::COPY_SRC,
                TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::All,
                },
                TextureUsage::COPY_DST,
                Extent3d {
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                },
                "copy buffer to texture",
                Some(&device),
            ),
            Ok(())
        );
    }

    #[test]
    fn buffer_texture_copy_rejects_depth_offset_not_four_byte_aligned() {
        let device = noop_device();
        let texture = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::from_raw(TextureFormat::DEPTH16_UNORM),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 16,
            mapped_at_creation: false,
        }));

        assert_eq!(
            validate_buffer_texture_copy(
                TexelCopyBufferInfo {
                    buffer,
                    device: Some(device.clone()),
                    layout: TexelCopyBufferLayout {
                        offset: 1,
                        bytes_per_row: None,
                        rows_per_image: None,
                    },
                },
                BufferUsage::COPY_SRC,
                TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::DepthOnly,
                },
                TextureUsage::COPY_DST,
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                "copy buffer to texture",
                Some(&device),
            ),
            Err(
                "copy buffer to texture buffer offset must be a multiple of the texel block size"
                    .to_owned()
            )
        );
    }

    #[test]
    fn buffer_texture_copy_rejects_unsupported_depth_stencil_copy_aspect_usage() {
        let device = noop_device();
        let buffer_src = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 16,
            mapped_at_creation: false,
        }));
        let buffer_dst = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 16,
            mapped_at_creation: false,
        }));
        let layout = TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: None,
            rows_per_image: None,
        };
        let size = Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let depth24 = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size,
            format: TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let depth32 = Arc::new(device.create_texture(TextureDescriptor {
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size,
            format: TextureFormat::from_raw(TextureFormat::DEPTH32_FLOAT),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));

        assert_eq!(
            validate_buffer_texture_copy(
                TexelCopyBufferInfo {
                    buffer: Arc::clone(&buffer_src),
                    device: Some(device.clone()),
                    layout,
                },
                BufferUsage::COPY_SRC,
                TexelCopyTextureInfo {
                    texture: depth24,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::DepthOnly,
                },
                TextureUsage::COPY_DST,
                size,
                "copy buffer to texture",
                Some(&device),
            ),
            Err(
                "copy buffer to texture depth/stencil format does not support this copy aspect/usage"
                    .to_owned()
            )
        );
        assert_eq!(
            validate_buffer_texture_copy(
                TexelCopyBufferInfo {
                    buffer: buffer_src,
                    device: Some(device.clone()),
                    layout,
                },
                BufferUsage::COPY_SRC,
                TexelCopyTextureInfo {
                    texture: Arc::clone(&depth32),
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::DepthOnly,
                },
                TextureUsage::COPY_DST,
                size,
                "copy buffer to texture",
                Some(&device),
            ),
            Err(
                "copy buffer to texture depth/stencil format does not support this copy aspect/usage"
                    .to_owned()
            )
        );
        assert_eq!(
            validate_buffer_texture_copy(
                TexelCopyBufferInfo {
                    buffer: buffer_dst,
                    device: Some(device.clone()),
                    layout,
                },
                BufferUsage::COPY_DST,
                TexelCopyTextureInfo {
                    texture: depth32,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: 0 },
                    aspect: TextureAspect::DepthOnly,
                },
                TextureUsage::COPY_SRC,
                size,
                "copy texture to buffer",
                Some(&device),
            ),
            Ok(())
        );
    }

    #[test]
    fn buffer_texture_copy_rejects_buffer_from_different_device() {
        let device_a = noop_adapter()
            .create_device(None, &[], "", "")
            .expect("first Noop device");
        let device_b = noop_adapter()
            .create_device(None, &[], "", "")
            .expect("second Noop device");
        let texture = Arc::new(device_a.create_texture(texture_descriptor_4x4()));
        let other_buffer = Arc::new(device_b.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 1024,
            mapped_at_creation: false,
        }));
        let same_device_buffer = Arc::new(device_a.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 1024,
            mapped_at_creation: false,
        }));
        let texture_info = TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::All,
        };
        let layout = TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(256),
            rows_per_image: None,
        };
        let size = Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        };

        let encoder = device_a.create_command_encoder();
        assert_eq!(
            encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: other_buffer,
                    device: Some(device_b),
                    layout,
                },
                texture_info.clone(),
                size,
            ),
            None
        );
        let (_, error) = encoder.finish();
        assert_eq!(
            error,
            Some("copy buffer to texture buffer must belong to the same device".to_owned())
        );

        let encoder = device_a.create_command_encoder();
        assert_eq!(
            encoder.copy_buffer_to_texture(
                TexelCopyBufferInfo {
                    buffer: same_device_buffer,
                    device: Some(device_a),
                    layout,
                },
                texture_info,
                size,
            ),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn command_encoder_query_and_timestamps_pin_validation_and_resolve() {
        let device = noop_device();
        let (timestamp_query, _) = device.create_query_set(QuerySetDescriptor {
            label: "bad timestamp".to_owned(),
            kind: QueryType::Timestamp,
            count: 2,
        });
        let timestamp_encoder = device.create_command_encoder();
        assert_eq!(
            timestamp_encoder.write_timestamp(Arc::new(timestamp_query), 0),
            None
        );
        let (command_buffer, error) = timestamp_encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("write timestamp cannot use an error query set".to_owned())
        );

        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "occlusion".to_owned(),
            kind: QueryType::Occlusion,
            count: 2,
        });
        assert_eq!(error, None);
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::QUERY_RESOLVE,
            size: 256,
            mapped_at_creation: false,
        }));
        let encoder = device.create_command_encoder();
        assert_eq!(
            encoder.resolve_query_set(Arc::new(query_set), 0, 2, destination, 0),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn compute_pass_timestamp_writes_validate_indices() {
        let mut features = FeatureSet::new();
        features.insert(crate::Feature::TimestampQuery);
        let device = crate::Device::from_hal(hal_noop_device(), Limits::DEFAULT, features, "", "");
        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "timestamps".to_owned(),
            kind: QueryType::Timestamp,
            count: 2,
        });
        assert_eq!(error, None);
        let writes = |beginning_index, end_index| RenderPassTimestampWrites {
            query_set: query_set.clone(),
            beginning_index,
            end_index,
        };

        assert_eq!(
            validate_compute_pass_timestamp_writes(&writes(None, None)),
            Err("compute pass timestamp writes requires at least one query index".to_owned())
        );
        assert_eq!(
            validate_compute_pass_timestamp_writes(&writes(Some(1), Some(1))),
            Err("compute pass timestamp write indices must be distinct".to_owned())
        );
        assert_eq!(
            validate_compute_pass_timestamp_writes(&writes(Some(2), None)),
            Err("compute pass beginning timestamp query index exceeds query set count".to_owned())
        );
        assert_eq!(
            validate_compute_pass_timestamp_writes(&writes(Some(0), None)),
            Ok(())
        );
        assert_eq!(
            validate_compute_pass_timestamp_writes(&writes(Some(0), Some(1))),
            Ok(())
        );
    }
}
