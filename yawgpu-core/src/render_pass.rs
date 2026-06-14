use std::sync::Arc;

use crate::bind_group::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::command_encoder::{validate_occlusion_query_set, validate_query_index};
use crate::copy::*;
use crate::limits::*;
use crate::pass::*;
use crate::query_set::*;
use crate::render_bundle::*;
use crate::render_pipeline::*;
use crate::texture_view::*;

/// Describes render pass descriptor.
#[derive(Debug, Clone)]
pub struct RenderPassDescriptor {
    /// Max color attachments.
    pub max_color_attachments: u32,
    /// Color attachments.
    pub color_attachments: Vec<Option<RenderPassColorAttachment>>,
    /// Depth stencil attachment.
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment>,
    /// Occlusion query set.
    pub occlusion_query_set: Option<QuerySet>,
    /// Timestamp writes.
    pub timestamp_writes: Option<RenderPassTimestampWrites>,
    /// Maximum draw calls allowed in this pass.
    pub max_draw_count: u64,
}

/// Stores render pass timestamp writes data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct RenderPassTimestampWrites {
    /// Query set.
    pub query_set: QuerySet,
    /// Beginning index.
    pub beginning_index: Option<u32>,
    /// End index.
    pub end_index: Option<u32>,
}

/// Stores color metadata.
#[derive(Debug, Clone)]
pub struct RenderPassColorAttachment {
    /// View.
    pub view: Arc<TextureView>,
    /// Depth slice for 3D color attachments.
    pub depth_slice: Option<u32>,
    /// Resolve target.
    pub resolve_target: Option<Arc<TextureView>>,
    /// Load op.
    pub load_op: LoadOp,
    /// Store op.
    pub store_op: StoreOp,
    /// Clear value.
    pub clear_value: Color,
}

/// Stores render pass depth stencil attachment data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct RenderPassDepthStencilAttachment {
    /// View.
    pub view: Arc<TextureView>,
    /// Depth load op.
    pub depth_load_op: LoadOp,
    /// Depth store op.
    pub depth_store_op: StoreOp,
    /// Depth clear value.
    pub depth_clear_value: f32,
    /// Depth read only.
    pub depth_read_only: bool,
    /// Stencil load op.
    pub stencil_load_op: LoadOp,
    /// Stencil store op.
    pub stencil_store_op: StoreOp,
    /// Stencil clear value.
    pub stencil_clear_value: u32,
    /// Stencil read only.
    pub stencil_read_only: bool,
}

/// Records commands for the RenderPassEncoder.
#[derive(Debug, Clone)]
pub struct RenderPassEncoder {
    pub(crate) inner: Arc<PassEncoderInner>,
}

fn validate_pipeline_attachment_compatibility(
    state: &PassEncoderState,
    pipeline: &RenderPipeline,
) -> Result<(), String> {
    let Some(pass_signature) = &state.attachment_signature else {
        return Ok(());
    };
    let pipeline_signature = pipeline.attachment_signature();
    if pass_signature.color_formats != pipeline_signature.color_formats
        || pass_signature.depth_stencil_format != pipeline_signature.depth_stencil_format
        || pass_signature.sample_count != pipeline_signature.sample_count
    {
        return Err("render pass pipeline attachment signature is incompatible".to_owned());
    }
    if pass_signature.depth_read_only && pipeline.writes_depth() {
        return Err(
            "render pass read-only depth attachment is incompatible with depth writes".to_owned(),
        );
    }
    if pass_signature.stencil_read_only && pipeline.writes_stencil() {
        return Err(
            "render pass read-only stencil attachment is incompatible with stencil writes"
                .to_owned(),
        );
    }
    Ok(())
}

impl RenderPassEncoder {
    /// Ends recording for this pass or encoder.
    pub fn end(&self) -> Option<String> {
        self.inner.end()
    }

    /// Records a debug marker within the render pass.
    pub fn insert_debug_marker(&self) -> Option<String> {
        self.inner.insert_debug_marker()
    }

    /// Opens a debug group within the render pass.
    pub fn push_debug_group(&self) -> Option<String> {
        self.inner.push_debug_group()
    }

    /// Closes the most recently opened debug group in the render pass.
    pub fn pop_debug_group(&self) -> Option<String> {
        self.inner.pop_debug_group()
    }

    /// Sets pipeline on this object or encoder.
    pub fn set_pipeline(&self, pipeline: Arc<RenderPipeline>) -> Option<String> {
        self.inner.record_pass_command(|state| {
            if pipeline.is_error() {
                return Err("render pass requires a valid render pipeline".to_owned());
            }
            validate_pipeline_attachment_compatibility(state, &pipeline)?;
            state.render_pipeline = Some(pipeline);
            Ok(())
        })
    }

    /// Records a validation error against the render pass.
    pub fn record_validation_error(&self, message: impl Into<String>) -> Option<String> {
        self.inner.record_pass_command(|_| Err(message.into()))
    }

    /// Sets bind group on this object or encoder.
    pub fn set_bind_group(
        &self,
        index: u32,
        group: Option<Arc<BindGroup>>,
        dynamic_offsets: Vec<u32>,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_set_bind_group(index, group.as_deref(), &dynamic_offsets, limits)?;
            if let Some(group) = group {
                self.inner
                    .parent
                    .record_referenced_buffers(bind_group_buffer_resources(&group));
                self.inner
                    .parent
                    .record_referenced_textures(bind_group_texture_resources(&group));
                let bound = BoundBindGroup {
                    group,
                    dynamic_offsets,
                };
                record_bind_group_buffer_usage_scope(state, &bound)?;
                state.bind_groups.insert(index, bound);
            } else {
                state.bind_groups.remove(&index);
            }
            Ok(())
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
        self.inner.record_pass_command(|state| {
            validate_vertex_buffer_slot(slot, limits)?;
            if let Some(buffer) = buffer {
                let size = validate_set_vertex_buffer(&buffer, offset, size)?;
                self.inner
                    .parent
                    .record_referenced_buffer(Arc::clone(&buffer));
                record_buffer_usage_scope_use(
                    state,
                    BufferScopeUse {
                        buffer: Arc::clone(&buffer),
                        offset,
                        size,
                        access: ResourceAccess::Read,
                    },
                )?;
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

    /// Sets index buffer on this object or encoder.
    pub fn set_index_buffer(
        &self,
        buffer: Arc<Buffer>,
        format: Option<IndexFormat>,
        offset: u64,
        size: u64,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            let format = format.ok_or_else(|| "render pass index format is invalid".to_owned())?;
            let size = validate_set_index_buffer(&buffer, format, offset, size)?;
            self.inner
                .parent
                .record_referenced_buffer(Arc::clone(&buffer));
            record_buffer_usage_scope_use(
                state,
                BufferScopeUse {
                    buffer: Arc::clone(&buffer),
                    offset,
                    size,
                    access: ResourceAccess::Read,
                },
            )?;
            state.index_buffer = Some(BoundIndexBuffer {
                buffer,
                format,
                offset,
                size,
            });
            Ok(())
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
        self.inner.record_pass_command(|state| {
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
                    .ok_or_else(|| "render pass requires a render pipeline".to_owned())?,
            );
            let bind_group_layouts = pipeline.bind_group_layouts().to_vec();
            let attachment_uses = state.attachment_texture_uses.clone();
            record_pipeline_usage_scope(state, &bind_group_layouts, &attachment_uses)?;
            state.draw_count = state.draw_count.saturating_add(1);
            let (color_attachments, depth_stencil_attachment) = state.load_attachments_for_draw();
            if color_attachments.is_empty() && depth_stencil_attachment.is_none() {
                return Err("render pass requires at least one attachment".to_owned());
            }
            self.inner.parent.record_render_pass(RenderPassCommand {
                pipeline: Some(pipeline),
                color_attachments,
                depth_stencil_attachment,
                attachment_textures: state.attachment_textures.clone(),
                bind_groups: state.bind_groups.clone(),
                vertex_buffers: state.vertex_buffers.clone(),
                index_buffer: state.index_buffer.clone(),
                indirect_buffer: None,
                viewport: state.viewport,
                scissor_rect: state.scissor_rect,
                blend_constant: state.blend_constant,
                stencil_reference: state.stencil_reference,
                occlusion_query_set: state.occlusion_query_set.clone(),
                occlusion_query_index: state.open_occlusion_query,
                draw: Some(RenderDrawExecution::Direct {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                }),
            });
            Ok(())
        })
    }

    /// Records an indexed draw after validating the bound pipeline and buffers.
    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
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
                    .ok_or_else(|| "render pass requires a render pipeline".to_owned())?,
            );
            let bind_group_layouts = pipeline.bind_group_layouts().to_vec();
            let attachment_uses = state.attachment_texture_uses.clone();
            record_pipeline_usage_scope(state, &bind_group_layouts, &attachment_uses)?;
            state.draw_count = state.draw_count.saturating_add(1);
            let (color_attachments, depth_stencil_attachment) = state.load_attachments_for_draw();
            if color_attachments.is_empty() && depth_stencil_attachment.is_none() {
                return Err("render pass requires at least one attachment".to_owned());
            }
            let index_buffer = state
                .index_buffer
                .clone()
                .ok_or_else(|| "render pass requires an index buffer".to_owned())?;
            self.inner.parent.record_render_pass(RenderPassCommand {
                pipeline: Some(pipeline),
                color_attachments,
                depth_stencil_attachment,
                attachment_textures: state.attachment_textures.clone(),
                bind_groups: state.bind_groups.clone(),
                vertex_buffers: state.vertex_buffers.clone(),
                index_buffer: Some(index_buffer),
                indirect_buffer: None,
                viewport: state.viewport,
                scissor_rect: state.scissor_rect,
                blend_constant: state.blend_constant,
                stencil_reference: state.stencil_reference,
                occlusion_query_set: state.occlusion_query_set.clone(),
                occlusion_query_index: state.open_occlusion_query,
                draw: Some(RenderDrawExecution::Indexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                }),
            });
            Ok(())
        })
    }

    /// Records an indirect draw sourced from a buffer after validation.
    pub fn draw_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::Indirect, limits)?;
            let pipeline = Arc::clone(
                state
                    .render_pipeline
                    .as_ref()
                    .ok_or_else(|| "render pass requires a render pipeline".to_owned())?,
            );
            let bind_group_layouts = pipeline.bind_group_layouts().to_vec();
            let attachment_uses = state.attachment_texture_uses.clone();
            validate_indirect_buffer(&indirect_buffer, indirect_offset, 16, "draw indirect")?;
            record_buffer_usage_scope_use(
                state,
                BufferScopeUse {
                    buffer: Arc::clone(&indirect_buffer),
                    offset: indirect_offset,
                    size: 16,
                    access: ResourceAccess::Read,
                },
            )?;
            record_pipeline_usage_scope(state, &bind_group_layouts, &attachment_uses)?;
            self.inner
                .parent
                .record_referenced_buffer(Arc::clone(&indirect_buffer));
            state.draw_count = state.draw_count.saturating_add(1);
            let (color_attachments, depth_stencil_attachment) = state.load_attachments_for_draw();
            if color_attachments.is_empty() && depth_stencil_attachment.is_none() {
                return Err("render pass requires at least one attachment".to_owned());
            }
            self.inner.parent.record_render_pass(RenderPassCommand {
                pipeline: Some(pipeline),
                color_attachments,
                depth_stencil_attachment,
                attachment_textures: state.attachment_textures.clone(),
                bind_groups: state.bind_groups.clone(),
                vertex_buffers: state.vertex_buffers.clone(),
                index_buffer: state.index_buffer.clone(),
                indirect_buffer: Some(BoundIndirectBuffer {
                    buffer: indirect_buffer,
                    offset: indirect_offset,
                }),
                viewport: state.viewport,
                scissor_rect: state.scissor_rect,
                blend_constant: state.blend_constant,
                stencil_reference: state.stencil_reference,
                occlusion_query_set: state.occlusion_query_set.clone(),
                occlusion_query_index: state.open_occlusion_query,
                draw: Some(RenderDrawExecution::Indirect {
                    offset: indirect_offset,
                }),
            });
            Ok(())
        })
    }

    /// Records an indexed indirect draw sourced from a buffer after validation.
    pub fn draw_indexed_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::IndexedIndirect, limits)?;
            let pipeline = Arc::clone(
                state
                    .render_pipeline
                    .as_ref()
                    .ok_or_else(|| "render pass requires a render pipeline".to_owned())?,
            );
            let bind_group_layouts = pipeline.bind_group_layouts().to_vec();
            let attachment_uses = state.attachment_texture_uses.clone();
            validate_indirect_buffer(
                &indirect_buffer,
                indirect_offset,
                20,
                "draw indexed indirect",
            )?;
            record_buffer_usage_scope_use(
                state,
                BufferScopeUse {
                    buffer: Arc::clone(&indirect_buffer),
                    offset: indirect_offset,
                    size: 20,
                    access: ResourceAccess::Read,
                },
            )?;
            record_pipeline_usage_scope(state, &bind_group_layouts, &attachment_uses)?;
            self.inner
                .parent
                .record_referenced_buffer(Arc::clone(&indirect_buffer));
            state.draw_count = state.draw_count.saturating_add(1);
            let (color_attachments, depth_stencil_attachment) = state.load_attachments_for_draw();
            if color_attachments.is_empty() && depth_stencil_attachment.is_none() {
                return Err("render pass requires at least one attachment".to_owned());
            }
            let index_buffer = state
                .index_buffer
                .clone()
                .ok_or_else(|| "render pass requires an index buffer".to_owned())?;
            self.inner.parent.record_render_pass(RenderPassCommand {
                pipeline: Some(pipeline),
                color_attachments,
                depth_stencil_attachment,
                attachment_textures: state.attachment_textures.clone(),
                bind_groups: state.bind_groups.clone(),
                vertex_buffers: state.vertex_buffers.clone(),
                index_buffer: Some(index_buffer),
                indirect_buffer: Some(BoundIndirectBuffer {
                    buffer: indirect_buffer,
                    offset: indirect_offset,
                }),
                viewport: state.viewport,
                scissor_rect: state.scissor_rect,
                blend_constant: state.blend_constant,
                stencil_reference: state.stencil_reference,
                occlusion_query_set: state.occlusion_query_set.clone(),
                occlusion_query_index: state.open_occlusion_query,
                draw: Some(RenderDrawExecution::IndexedIndirect {
                    offset: indirect_offset,
                }),
            });
            Ok(())
        })
    }

    /// Sets viewport on this object or encoder.
    pub fn set_viewport(
        &self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_viewport(x, y, width, height, min_depth, max_depth)?;
            validate_viewport_bounds(x, y, width, height, state.limits)?;
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

    /// Sets scissor rect on this object or encoder.
    pub fn set_scissor_rect(&self, x: u32, y: u32, width: u32, height: u32) -> Option<String> {
        self.inner.record_pass_command(|state| {
            x.checked_add(width)
                .ok_or_else(|| "render pass scissor rectangle width overflows".to_owned())?;
            y.checked_add(height)
                .ok_or_else(|| "render pass scissor rectangle height overflows".to_owned())?;
            validate_scissor_rect(state.render_extent, x, y, width, height)?;
            state.scissor_rect = Some(ScissorRect {
                x,
                y,
                width,
                height,
            });
            Ok(())
        })
    }

    /// Sets blend constant on this object or encoder.
    pub fn set_blend_constant(&self, color: Color) -> Option<String> {
        self.inner.record_pass_command(|state| {
            let components = [color.r, color.g, color.b, color.a];
            if !components.into_iter().all(f64::is_finite) {
                return Err("render pass blend constant components must be finite".to_owned());
            }
            state.blend_constant = components.map(|component| component as f32);
            Ok(())
        })
    }

    /// Sets stencil reference on this object or encoder.
    pub fn set_stencil_reference(&self, reference: u32) -> Option<String> {
        self.inner.record_pass_command(|state| {
            state.stencil_reference = reference;
            Ok(())
        })
    }

    /// Replays the given render bundles into this pass after validation.
    pub fn execute_bundles(&self, bundles: &[Arc<RenderBundle>]) -> Option<String> {
        self.inner.record_pass_command(|state| {
            let pass_signature = state
                .attachment_signature
                .as_ref()
                .ok_or_else(|| "render pass has no attachment signature".to_owned())?
                .clone();
            for bundle in bundles {
                if bundle.is_error() {
                    return Err("render pass cannot execute an error render bundle".to_owned());
                }
                if !bundle
                    .attachment_signature()
                    .bundle_compatible_with_pass(&pass_signature)
                {
                    return Err(
                        "render bundle attachment signature is incompatible with the render pass"
                            .to_owned(),
                    );
                }
                self.inner
                    .parent
                    .record_referenced_buffers(bundle.referenced_buffers().to_vec());
                self.inner
                    .parent
                    .record_referenced_textures(bundle.referenced_textures().to_vec());
                let mut scoped_buffer_uses = state.scope_buffer_uses.clone();
                scoped_buffer_uses.extend_from_slice(bundle.buffer_uses());
                let mut scoped_texture_uses = state.scope_texture_uses.clone();
                scoped_texture_uses.extend_from_slice(bundle.texture_uses());
                scoped_texture_uses.extend_from_slice(&state.attachment_texture_uses);
                validate_resource_usage_scope_lenient(&scoped_buffer_uses, &scoped_texture_uses)?;
                state
                    .scope_buffer_uses
                    .extend_from_slice(bundle.buffer_uses());
                state
                    .scope_texture_uses
                    .extend_from_slice(bundle.texture_uses());
                for draw in bundle.draws() {
                    state.draw_count = state.draw_count.saturating_add(1);
                    let (color_attachments, depth_stencil_attachment) =
                        state.load_attachments_for_draw();
                    self.inner.parent.record_render_pass(RenderPassCommand {
                        pipeline: Some(Arc::clone(&draw.pipeline)),
                        color_attachments,
                        depth_stencil_attachment,
                        attachment_textures: state.attachment_textures.clone(),
                        bind_groups: draw.bind_groups.clone(),
                        vertex_buffers: draw.vertex_buffers.clone(),
                        index_buffer: draw.index_buffer.clone(),
                        indirect_buffer: draw.indirect_buffer.clone(),
                        viewport: state.viewport,
                        scissor_rect: state.scissor_rect,
                        blend_constant: state.blend_constant,
                        stencil_reference: state.stencil_reference,
                        occlusion_query_set: state.occlusion_query_set.clone(),
                        occlusion_query_index: state.open_occlusion_query,
                        draw: Some(draw.draw),
                    });
                }
            }
            state.clear_render_state();
            Ok(())
        })
    }

    /// Begins an occlusion query at `query_index`, counting samples for the draws that follow.
    pub fn begin_occlusion_query(&self, query_index: u32) -> Option<String> {
        self.inner.record_pass_command(|state| {
            let query_set = state
                .occlusion_query_set
                .as_ref()
                .ok_or_else(|| "render pass has no occlusion query set".to_owned())?;
            validate_occlusion_query_set(query_set, "render pass occlusion query")?;
            validate_query_index(query_set, query_index, "occlusion query index")?;
            if state.open_occlusion_query.is_some() {
                return Err("render pass occlusion query is already open".to_owned());
            }
            if !state.used_occlusion_queries.insert(query_index) {
                return Err("render pass occlusion query index was already used".to_owned());
            }
            state.open_occlusion_query = Some(query_index);
            Ok(())
        })
    }

    /// Ends occlusion query recording.
    pub fn end_occlusion_query(&self) -> Option<String> {
        self.inner.record_pass_command(|state| {
            if state.open_occlusion_query.take().is_none() {
                return Err("render pass has no open occlusion query".to_owned());
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::{
        BindGroupLayout, ColorTargetState, CompareFunction, Device, Extent3d, MultisampleState,
        PrimitiveState, PrimitiveTopology, RenderPipelineDescriptor, RenderPipelineFragmentState,
        RenderPipelineLayout, RenderPipelineShaderStage, RenderPipelineVertexState,
        ShaderModuleSource, TextureDescriptor, TextureDimension, TextureFormat, TextureUsage,
        VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode,
    };

    use std::sync::Arc;

    #[test]
    fn render_pass_encoder_lifecycle_and_debug_markers() {
        let device = noop_device();
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.push_debug_group(), None);
        assert_eq!(pass.insert_debug_marker(), None);
        assert_eq!(pass.pop_debug_group(), None);
        assert_eq!(
            pass.record_validation_error("forced render pass error"),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(error, Some("forced render pass error".to_owned()));
    }

    #[test]
    fn render_pass_encoder_set_pipeline_bind_group_buffers_and_draw() {
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
        assert_eq!(
            pass.set_vertex_buffer(0, Some(vertex_buffer), 0, 16, device.limits()),
            None
        );
        assert_eq!(
            pass.set_index_buffer(index_buffer, Some(IndexFormat::Uint16), 0, 16),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 1);
        let CommandExecution::RenderPass(command) = &command_buffer.command_ops()[0] else {
            panic!("expected render pass command");
        };
        assert!(matches!(
            command.draw,
            Some(RenderDrawExecution::Direct {
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            })
        ));
        assert_eq!(command.blend_constant, [0.0; 4]);
        assert_eq!(command.stencil_reference, 0);
        assert_eq!(
            command.color_attachments[0]
                .as_ref()
                .expect("color attachment")
                .depth_slice,
            0
        );
    }

    #[test]
    fn render_pass_vertex_buffer_oob_uses_last_stride() {
        let device = noop_device();
        let pipeline = padded_vertex_render_pipeline(&device);

        let error = draw_with_padded_vertex_buffer_size(&device, pipeline.clone(), 40);
        assert_eq!(error, None);

        let error = draw_with_padded_vertex_buffer_size(&device, pipeline, 39);
        assert_eq!(
            error,
            Some("render pass draw vertex buffer range exceeds the bound buffer".to_owned())
        );
    }

    #[test]
    fn render_pass_zero_stride_vertex_buffer_oob_uses_one_element() {
        let device = noop_device();
        let pipeline = zero_stride_vertex_render_pipeline(&device);

        let error = draw_with_padded_vertex_buffer_size(&device, pipeline.clone(), 8);
        assert_eq!(error, None);

        let error = draw_with_padded_vertex_buffer_size(&device, pipeline, 7);
        assert_eq!(
            error,
            Some("render pass draw vertex buffer range exceeds the bound buffer".to_owned())
        );
    }

    #[test]
    fn render_pass_encoder_allows_storage_buffer_write_write_across_draws() {
        let device = noop_device();
        let pipeline = storage_write_render_pipeline(&device);
        let layout = pipeline.bind_group_layouts()[0].clone();
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE,
            size: 16,
            mapped_at_creation: false,
        }));
        let bind_group_a = buffer_bind_group(&device, layout.clone(), buffer.clone(), 0);
        let bind_group_b = buffer_bind_group(&device, layout, buffer, 0);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group_a), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group_b), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 2);
        for command in command_buffer.command_ops() {
            let CommandExecution::RenderPass(command) = command else {
                panic!("expected render pass command");
            };
            assert!(matches!(
                command.draw,
                Some(RenderDrawExecution::Direct {
                    vertex_count: 3,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                })
            ));
        }
    }

    #[test]
    fn render_pass_encoder_rejects_cross_draw_storage_write_uniform_read_alias() {
        let device = noop_device();
        let write_pipeline = storage_write_render_pipeline(&device);
        let read_pipeline = uniform_read_render_pipeline(&device);
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::UNIFORM,
            size: 16,
            mapped_at_creation: false,
        }));
        let write_bind_group = buffer_bind_group(
            &device,
            write_pipeline.bind_group_layouts()[0].clone(),
            buffer.clone(),
            0,
        );
        let read_bind_group = buffer_bind_group(
            &device,
            read_pipeline.bind_group_layouts()[0].clone(),
            buffer,
            0,
        );
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(write_pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(write_bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.set_pipeline(read_pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(read_bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some(
                "usage scope cannot read and write or write the same buffer range twice".to_owned()
            )
        );
    }

    #[test]
    fn render_pass_encoder_rejects_same_draw_storage_write_uniform_read_alias() {
        let device = noop_device();
        let pipeline = storage_write_uniform_read_render_pipeline(&device);
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::UNIFORM,
            size: 16,
            mapped_at_creation: false,
        }));
        let bind_group =
            aliasing_buffer_bind_group(&device, pipeline.bind_group_layouts()[0].clone(), buffer);
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
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some(
                "usage scope cannot read and write or write the same buffer range twice".to_owned()
            )
        );
    }

    #[test]
    fn render_pass_encoder_rejects_vertex_buffer_storage_write_alias() {
        let device = noop_device();
        let pipeline = storage_write_vertex_render_pipeline(&device);
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::VERTEX,
            size: 32,
            mapped_at_creation: false,
        }));
        let bind_group = buffer_bind_group(
            &device,
            pipeline.bind_group_layouts()[0].clone(),
            buffer.clone(),
            0,
        );
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
        assert_eq!(
            pass.set_vertex_buffer(0, Some(buffer), 0, 32, device.limits()),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some(
                "usage scope cannot read and write or write the same buffer range twice".to_owned()
            )
        );
    }

    #[test]
    fn render_pass_encoder_allows_vertex_buffer_uniform_read_alias() {
        let device = noop_device();
        let pipeline = uniform_read_vertex_render_pipeline(&device);
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::UNIFORM | BufferUsage::VERTEX,
            size: 32,
            mapped_at_creation: false,
        }));
        let bind_group = buffer_bind_group(
            &device,
            pipeline.bind_group_layouts()[0].clone(),
            buffer.clone(),
            0,
        );
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
        assert_eq!(
            pass.set_vertex_buffer(0, Some(buffer), 0, 32, device.limits()),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn render_pass_encoder_rejects_index_buffer_storage_write_alias() {
        let device = noop_device();
        let pipeline = storage_write_render_pipeline(&device);
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::INDEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let bind_group = buffer_bind_group(
            &device,
            pipeline.bind_group_layouts()[0].clone(),
            buffer.clone(),
            0,
        );
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
        assert_eq!(
            pass.set_index_buffer(buffer, Some(IndexFormat::Uint16), 0, 16),
            None
        );
        assert_eq!(pass.draw_indexed(3, 1, 0, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some(
                "usage scope cannot read and write or write the same buffer range twice".to_owned()
            )
        );
    }

    #[test]
    fn render_pass_encoder_rejects_set_time_storage_vertex_alias_without_draw() {
        let device = noop_device();
        let pipeline = storage_write_render_pipeline(&device);
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::VERTEX,
            size: 32,
            mapped_at_creation: false,
        }));
        let bind_group = buffer_bind_group(
            &device,
            pipeline.bind_group_layouts()[0].clone(),
            buffer.clone(),
            0,
        );
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(
            pass.set_vertex_buffer(0, Some(buffer), 0, 32, device.limits()),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some(
                "usage scope cannot read and write or write the same buffer range twice".to_owned()
            )
        );
    }

    #[test]
    fn render_pass_encoder_rebinding_storage_still_counts_before_vertex_use() {
        let device = noop_device();
        let pipeline = storage_write_render_pipeline(&device);
        let buffer_a = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::VERTEX,
            size: 32,
            mapped_at_creation: false,
        }));
        let buffer_b = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE,
            size: 32,
            mapped_at_creation: false,
        }));
        let bind_group_a = buffer_bind_group(
            &device,
            pipeline.bind_group_layouts()[0].clone(),
            buffer_a.clone(),
            0,
        );
        let bind_group_b = buffer_bind_group(
            &device,
            pipeline.bind_group_layouts()[0].clone(),
            buffer_b,
            0,
        );
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(
            pass.set_bind_group(0, Some(bind_group_a), Vec::new(), device.limits()),
            None
        );
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group_b), Vec::new(), device.limits()),
            None
        );
        assert_eq!(
            pass.set_vertex_buffer(0, Some(buffer_a), 0, 32, device.limits()),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some(
                "usage scope cannot read and write or write the same buffer range twice".to_owned()
            )
        );
    }

    #[test]
    fn render_pass_encoder_execute_bundles_replays_bundle_draws() {
        let device = noop_device();
        let pipeline = storage_write_render_pipeline(&device);
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE,
            size: 16,
            mapped_at_creation: false,
        }));
        let bundle = storage_write_render_bundle(&device, pipeline, buffer, 2);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(
            pass.set_blend_constant(Color {
                r: 0.25,
                g: 0.5,
                b: 0.75,
                a: 1.0,
            }),
            None
        );
        assert_eq!(pass.set_stencil_reference(13), None);
        assert_eq!(pass.execute_bundles(&[bundle]), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 2);
        for command in command_buffer.command_ops() {
            assert_direct_render_pass_command(command, [0.25, 0.5, 0.75, 1.0], 13);
        }
    }

    #[test]
    fn render_pass_encoder_records_inline_draw_and_bundle_draw() {
        let device = noop_device();
        let pipeline = storage_write_render_pipeline(&device);
        let layout = pipeline.bind_group_layouts()[0].clone();
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE,
            size: 16,
            mapped_at_creation: false,
        }));
        let inline_bind_group = buffer_bind_group(&device, layout, buffer.clone(), 0);
        let bundle = storage_write_render_bundle(&device, pipeline.clone(), buffer, 1);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(inline_bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.execute_bundles(&[bundle]), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 2);
        for command in command_buffer.command_ops() {
            assert_direct_render_pass_command(command, [0.0; 4], 0);
        }
    }

    #[test]
    fn render_pass_encoder_loads_after_first_draw_and_clear_only_keeps_clear() {
        let device = noop_device();
        let pipeline = depth_stencil_render_pipeline(&device);
        let view_a = noop_render_attachment(&device);
        let view_b = noop_render_attachment(&device);
        let depth_stencil_view = depth24_plus_stencil8_attachment(&device);
        let mut descriptor = noop_render_pass_descriptor(view_a, None);
        let mut attachment_b = descriptor.color_attachments[0]
            .clone()
            .expect("base descriptor should have a color attachment");
        attachment_b.view = view_b;
        descriptor.color_attachments.push(Some(attachment_b));
        for attachment in descriptor.color_attachments.iter_mut().flatten() {
            attachment.store_op = StoreOp::Discard;
        }
        descriptor.depth_stencil_attachment = Some(RenderPassDepthStencilAttachment {
            view: depth_stencil_view,
            depth_load_op: LoadOp::Clear,
            depth_store_op: StoreOp::Discard,
            depth_clear_value: 1.0,
            depth_read_only: false,
            stencil_load_op: LoadOp::Clear,
            stencil_store_op: StoreOp::Discard,
            stencil_clear_value: 7,
            stencil_read_only: false,
        });

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&descriptor);
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 2);
        let CommandExecution::RenderPass(first) = &command_buffer.command_ops()[0] else {
            panic!("expected first render pass command");
        };
        let CommandExecution::RenderPass(second) = &command_buffer.command_ops()[1] else {
            panic!("expected second render pass command");
        };
        assert_render_pass_attachment_ops(first, LoadOp::Clear, StoreOp::Store);
        assert_render_pass_attachment_ops(second, LoadOp::Load, StoreOp::Discard);

        let clear_encoder = device.create_command_encoder();
        let (clear_pass, begin_error) = clear_encoder.begin_render_pass(&descriptor);
        assert_eq!(begin_error, None);
        assert_eq!(clear_pass.end(), None);
        let (clear_command_buffer, error) = clear_encoder.finish();
        assert_eq!(error, None);
        assert_eq!(clear_command_buffer.command_ops().len(), 1);
        let CommandExecution::RenderPass(clear_only) = &clear_command_buffer.command_ops()[0]
        else {
            panic!("expected clear-only render pass command");
        };
        assert!(clear_only.draw.is_none());
        assert_render_pass_attachment_ops(clear_only, LoadOp::Clear, StoreOp::Discard);
    }

    #[test]
    fn render_pass_encoder_final_draw_preserves_user_color_store_op() {
        for store_op in [StoreOp::Store, StoreOp::Discard] {
            let device = noop_device();
            let pipeline = noop_render_pipeline(&device);
            let view = noop_render_attachment(&device);
            let mut descriptor = noop_render_pass_descriptor(view, None);
            descriptor.color_attachments[0]
                .as_mut()
                .expect("color attachment")
                .store_op = store_op;

            let encoder = device.create_command_encoder();
            let (pass, begin_error) = encoder.begin_render_pass(&descriptor);
            assert_eq!(begin_error, None);
            assert_eq!(pass.set_pipeline(pipeline), None);
            assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
            assert_eq!(pass.end(), None);

            let (command_buffer, error) = encoder.finish();
            assert_eq!(error, None);
            assert_eq!(command_buffer.command_ops().len(), 1);
            let CommandExecution::RenderPass(command) = &command_buffer.command_ops()[0] else {
                panic!("expected render pass command");
            };
            let color = command.color_attachments[0]
                .as_ref()
                .expect("color attachment");
            assert_eq!(color.store_op, store_op);
        }
    }

    #[test]
    fn render_pass_encoder_two_draws_force_intermediate_store_and_preserve_final_store_op() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let view = noop_render_attachment(&device);
        let mut descriptor = noop_render_pass_descriptor(view, None);
        descriptor.color_attachments[0]
            .as_mut()
            .expect("color attachment")
            .store_op = StoreOp::Discard;

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&descriptor);
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert_eq!(command_buffer.command_ops().len(), 2);
        let CommandExecution::RenderPass(first) = &command_buffer.command_ops()[0] else {
            panic!("expected first render pass command");
        };
        let CommandExecution::RenderPass(last) = &command_buffer.command_ops()[1] else {
            panic!("expected last render pass command");
        };
        assert_eq!(
            first.color_attachments[0]
                .as_ref()
                .expect("color attachment")
                .store_op,
            StoreOp::Store
        );
        assert_eq!(
            last.color_attachments[0]
                .as_ref()
                .expect("color attachment")
                .store_op,
            StoreOp::Discard
        );
    }

    #[test]
    fn render_pass_encoder_execute_bundles_clears_render_state() {
        let device = noop_device();
        let pipeline = storage_write_render_pipeline(&device);
        let layout = pipeline.bind_group_layouts()[0].clone();
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE,
            size: 16,
            mapped_at_creation: false,
        }));
        let bind_group = buffer_bind_group(&device, layout, buffer.clone(), 0);
        let bundle = storage_write_render_bundle(&device, pipeline.clone(), buffer, 1);
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
        assert_eq!(pass.execute_bundles(&[bundle]), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("render pass draw requires a render pipeline".to_owned())
        );
    }

    #[test]
    fn render_pass_encoder_records_two_color_attachments() {
        let device = noop_device();
        let pipeline = two_color_render_pipeline(&device);
        let view_a = noop_render_attachment(&device);
        let view_b = noop_render_attachment(&device);
        let mut descriptor = noop_render_pass_descriptor(view_a, None);
        let mut attachment_b = descriptor.color_attachments[0]
            .clone()
            .expect("base descriptor should have a color attachment");
        attachment_b.view = view_b;
        descriptor.color_attachments.push(Some(attachment_b));

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&descriptor);
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 1);
        let CommandExecution::RenderPass(command) = &command_buffer.command_ops()[0] else {
            panic!("expected render pass command");
        };
        assert_eq!(command.color_attachments.len(), 2);
        assert!(command.depth_stencil_attachment.is_none());
        assert!(command.draw.is_some());
    }

    #[test]
    fn render_pass_encoder_records_3d_color_attachment_depth_slice() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let view = render_attachment_3d_view(&device);
        let mut descriptor = noop_render_pass_descriptor(view, None);
        descriptor.color_attachments[0]
            .as_mut()
            .expect("color attachment")
            .depth_slice = Some(1);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&descriptor);
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        let CommandExecution::RenderPass(command) = &command_buffer.command_ops()[0] else {
            panic!("expected render pass command");
        };
        assert_eq!(
            command.color_attachments[0]
                .as_ref()
                .expect("color attachment")
                .depth_slice,
            1
        );
    }

    fn buffer_bind_group(
        device: &crate::device::Device,
        layout: Arc<BindGroupLayout>,
        buffer: Arc<Buffer>,
        binding: u32,
    ) -> Arc<BindGroup> {
        Arc::new(device.create_bind_group(
            layout,
            vec![BindGroupEntry {
                binding,
                resource: BindGroupResource::Buffer {
                    buffer,
                    device: Arc::new(device.clone()),
                    offset: 0,
                    size: 16,
                },
            }],
        ))
    }

    fn aliasing_buffer_bind_group(
        device: &crate::device::Device,
        layout: Arc<BindGroupLayout>,
        buffer: Arc<Buffer>,
    ) -> Arc<BindGroup> {
        Arc::new(device.create_bind_group(
            layout,
            vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindGroupResource::Buffer {
                        buffer: buffer.clone(),
                        device: Arc::new(device.clone()),
                        offset: 0,
                        size: 16,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindGroupResource::Buffer {
                        buffer,
                        device: Arc::new(device.clone()),
                        offset: 0,
                        size: 16,
                    },
                },
            ],
        ))
    }

    fn storage_write_render_bundle(
        device: &crate::device::Device,
        pipeline: Arc<RenderPipeline>,
        buffer: Arc<Buffer>,
        draw_count: usize,
    ) -> Arc<RenderBundle> {
        let layout = pipeline.bind_group_layouts()[0].clone();
        let (bundle_encoder, error) = RenderBundleEncoder::new(
            render_bundle_encoder_descriptor(),
            device.limits(),
            device.features(),
        );
        assert_eq!(error, None);
        assert_eq!(bundle_encoder.set_pipeline(pipeline), None);
        for _ in 0..draw_count {
            let bind_group = buffer_bind_group(device, layout.clone(), buffer.clone(), 0);
            assert_eq!(
                bundle_encoder.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
                None
            );
            assert_eq!(bundle_encoder.draw(3, 1, 0, 0, device.limits()), None);
        }
        let (bundle, error) = bundle_encoder.finish();
        assert_eq!(error, None);
        assert!(!bundle.is_error());
        Arc::new(bundle)
    }

    fn assert_direct_render_pass_command(
        command: &CommandExecution,
        blend_constant: [f32; 4],
        stencil_reference: u32,
    ) {
        let CommandExecution::RenderPass(command) = command else {
            panic!("expected render pass command");
        };
        assert_eq!(command.color_attachments.len(), 1);
        assert!(command.depth_stencil_attachment.is_none());
        assert_eq!(command.blend_constant, blend_constant);
        assert_eq!(command.stencil_reference, stencil_reference);
        assert!(matches!(
            command.draw,
            Some(RenderDrawExecution::Direct {
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 0,
            })
        ));
    }

    #[test]
    fn render_pass_encoder_rejects_color_attachment_count_mismatch() {
        let device = noop_device();
        let pipeline = two_color_render_pipeline(&device);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("render pass pipeline attachment signature is incompatible".to_owned())
        );
    }

    #[test]
    fn render_pass_pipeline_depth_format_must_match_exactly() {
        let device = noop_device();
        let color = noop_render_attachment(&device);
        let depth = depth24_plus_stencil8_attachment(&device);
        let mut descriptor = noop_render_pass_descriptor(color, None);
        descriptor.depth_stencil_attachment = Some(depth_stencil_attachment(depth, false, false));

        let matching = depth_format_render_pipeline(
            &device,
            Some(TextureFormat::from_raw(
                TextureFormat::DEPTH24_PLUS_STENCIL8,
            )),
            false,
            false,
        );
        let no_depth = noop_render_pipeline(&device);
        let mismatched = depth_format_render_pipeline(
            &device,
            Some(TextureFormat::from_raw(TextureFormat::DEPTH32_FLOAT)),
            false,
            false,
        );

        assert_render_pass_pipeline_finish_ok(&device, descriptor.clone(), matching);
        assert_render_pass_pipeline_finish_error(&device, descriptor.clone(), no_depth);
        assert_render_pass_pipeline_finish_error(&device, descriptor, mismatched);

        let color_only = noop_render_pass_descriptor(noop_render_attachment(&device), None);
        let pipeline_with_depth = depth_format_render_pipeline(
            &device,
            Some(TextureFormat::from_raw(
                TextureFormat::DEPTH24_PLUS_STENCIL8,
            )),
            false,
            false,
        );
        assert_render_pass_pipeline_finish_error(&device, color_only, pipeline_with_depth);
    }

    #[test]
    fn render_pass_pipeline_writes_are_incompatible_with_read_only_aspects() {
        let device = noop_device();
        let color = noop_render_attachment(&device);
        let depth = depth24_plus_stencil8_attachment(&device);
        let mut descriptor = noop_render_pass_descriptor(color, None);
        descriptor.depth_stencil_attachment = Some(depth_stencil_attachment(depth, true, false));

        let depth_writer = depth_format_render_pipeline(
            &device,
            Some(TextureFormat::from_raw(
                TextureFormat::DEPTH24_PLUS_STENCIL8,
            )),
            true,
            false,
        );
        assert_render_pass_pipeline_finish_error(&device, descriptor.clone(), depth_writer);

        {
            let attachment = descriptor
                .depth_stencil_attachment
                .as_mut()
                .expect("depth-stencil attachment");
            attachment.depth_read_only = false;
            attachment.depth_load_op = LoadOp::Clear;
            attachment.depth_store_op = StoreOp::Discard;
        }
        let depth_writer = depth_format_render_pipeline(
            &device,
            Some(TextureFormat::from_raw(
                TextureFormat::DEPTH24_PLUS_STENCIL8,
            )),
            true,
            false,
        );
        assert_render_pass_pipeline_finish_ok(&device, descriptor.clone(), depth_writer);

        descriptor
            .depth_stencil_attachment
            .as_mut()
            .expect("depth-stencil attachment")
            .stencil_read_only = true;
        descriptor
            .depth_stencil_attachment
            .as_mut()
            .expect("depth-stencil attachment")
            .stencil_load_op = LoadOp::Undefined;
        descriptor
            .depth_stencil_attachment
            .as_mut()
            .expect("depth-stencil attachment")
            .stencil_store_op = StoreOp::Undefined;
        let stencil_writer = depth_format_render_pipeline(
            &device,
            Some(TextureFormat::from_raw(
                TextureFormat::DEPTH24_PLUS_STENCIL8,
            )),
            false,
            true,
        );
        assert_render_pass_pipeline_finish_error(&device, descriptor.clone(), stencil_writer);

        let stencil_reader = depth_format_render_pipeline(
            &device,
            Some(TextureFormat::from_raw(
                TextureFormat::DEPTH24_PLUS_STENCIL8,
            )),
            false,
            false,
        );
        assert_render_pass_pipeline_finish_ok(&device, descriptor, stencil_reader);
    }

    #[test]
    fn render_pass_encoder_records_resolve_target() {
        let device = noop_device();
        let pipeline = one_color_render_pipeline(&device, 4);
        let color = render_attachment_view(&device, 4);
        let resolve = render_attachment_view(&device, 1);
        let mut descriptor = noop_render_pass_descriptor(color, None);
        descriptor.color_attachments[0]
            .as_mut()
            .expect("color attachment")
            .resolve_target = Some(resolve);

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&descriptor);
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        let CommandExecution::RenderPass(command) = &command_buffer.command_ops()[0] else {
            panic!("expected render pass command");
        };
        assert_eq!(command.color_attachments.len(), 1);
        let color = command.color_attachments[0]
            .as_ref()
            .expect("color attachment");
        assert!(color.resolve_target.is_some());
        assert_eq!(color.resolve_mip_level, 0);
        assert_eq!(color.resolve_array_layer, 0);
    }

    fn render_attachment_view(
        device: &crate::device::Device,
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
            format: rgba8_unorm(),
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

    fn depth_stencil_attachment(
        view: Arc<TextureView>,
        depth_read_only: bool,
        stencil_read_only: bool,
    ) -> RenderPassDepthStencilAttachment {
        RenderPassDepthStencilAttachment {
            view,
            depth_load_op: if depth_read_only {
                LoadOp::Undefined
            } else {
                LoadOp::Clear
            },
            depth_store_op: if depth_read_only {
                StoreOp::Undefined
            } else {
                StoreOp::Discard
            },
            depth_clear_value: 0.0,
            depth_read_only,
            stencil_load_op: if stencil_read_only {
                LoadOp::Undefined
            } else {
                LoadOp::Clear
            },
            stencil_store_op: if stencil_read_only {
                StoreOp::Undefined
            } else {
                StoreOp::Discard
            },
            stencil_clear_value: 0,
            stencil_read_only,
        }
    }

    fn depth_format_render_pipeline(
        device: &Device,
        depth_format: Option<TextureFormat>,
        depth_write: bool,
        stencil_write: bool,
    ) -> Arc<RenderPipeline> {
        let module = render_shader_module(device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.depth_stencil = depth_format.map(|format| DepthStencilState {
            format,
            depth_write_enabled: Some(depth_write),
            depth_compare: (depth_write || stencil_write).then_some(CompareFunction::Always),
            stencil_front: stencil_face_state(stencil_write),
            stencil_back: stencil_face_state(stencil_write),
            stencil_read_mask: u32::MAX,
            stencil_write_mask: u32::MAX,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        });
        Arc::new(device.create_render_pipeline(descriptor))
    }

    fn stencil_face_state(write: bool) -> StencilFaceState {
        StencilFaceState {
            compare: CompareFunction::Always,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            pass_op: if write {
                StencilOperation::Replace
            } else {
                StencilOperation::Keep
            },
        }
    }

    fn assert_render_pass_pipeline_finish_ok(
        device: &Device,
        descriptor: RenderPassDescriptor,
        pipeline: Arc<RenderPipeline>,
    ) {
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&descriptor);
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    fn assert_render_pass_pipeline_finish_error(
        device: &Device,
        descriptor: RenderPassDescriptor,
        pipeline: Arc<RenderPipeline>,
    ) {
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&descriptor);
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert!(error.is_some());
    }

    fn render_attachment_3d_view(device: &crate::device::Device) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
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

    fn one_color_render_pipeline(
        device: &crate::device::Device,
        sample_count: u32,
    ) -> Arc<RenderPipeline> {
        let mut descriptor = render_pipeline_descriptor(render_shader_module(device));
        descriptor.multisample.count = sample_count;
        Arc::new(device.create_render_pipeline(descriptor))
    }

    fn padded_vertex_render_pipeline(device: &Device) -> Arc<RenderPipeline> {
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@vertex
fn vs(@location(0) position: vec2f) -> @builtin(position) vec4f {
    return vec4f(position, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
"
                .to_owned(),
            )),
        );
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.vertex.buffer_count = 1;
        descriptor.vertex.buffers = vec![VertexBufferLayout {
            used: true,
            array_stride: 32,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![VertexAttribute {
                format: VertexFormat::from_raw(0x0000_001D),
                offset: 0,
                shader_location: 0,
            }],
        }];
        Arc::new(device.create_render_pipeline(descriptor))
    }

    fn zero_stride_vertex_render_pipeline(device: &Device) -> Arc<RenderPipeline> {
        let mut descriptor = render_pipeline_descriptor(render_shader_module(device));
        descriptor.vertex.buffer_count = 1;
        descriptor.vertex.buffers = vec![VertexBufferLayout {
            used: true,
            array_stride: 0,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![VertexAttribute {
                format: VertexFormat::from_raw(0x0000_001D),
                offset: 0,
                shader_location: 0,
            }],
        }];
        Arc::new(device.create_render_pipeline(descriptor))
    }

    fn draw_with_padded_vertex_buffer_size(
        device: &Device,
        pipeline: Arc<RenderPipeline>,
        vertex_buffer_size: u64,
    ) -> Option<String> {
        let vertex_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::VERTEX,
            size: vertex_buffer_size,
            mapped_at_creation: false,
        }));
        let view = noop_render_attachment(device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_vertex_buffer(
                0,
                Some(vertex_buffer),
                0,
                vertex_buffer_size,
                device.limits()
            ),
            None
        );
        assert_eq!(pass.draw(2, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(command_buffer.is_error(), error.is_some());
        error
    }

    fn two_color_render_pipeline(device: &crate::device::Device) -> Arc<RenderPipeline> {
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@vertex
fn vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

struct FragmentOutput {
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
}

@fragment
fn fs() -> FragmentOutput {
    var output: FragmentOutput;
    output.a = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    output.b = vec4<f32>(0.0, 1.0, 0.0, 1.0);
    return output;
}
"
                .to_owned(),
            )),
        );
        let mut descriptor = RenderPipelineDescriptor {
            layout: RenderPipelineLayout::Auto,
            vertex: RenderPipelineVertexState {
                shader: RenderPipelineShaderStage {
                    module: Arc::clone(&module),
                    entry_point: Some("vs".to_owned()),
                    constants: Vec::new(),
                },
                buffer_count: 0,
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
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
                target_count: 2,
                targets: Vec::new(),
            }),
            error: None,
        };
        if let Some(fragment) = &mut descriptor.fragment {
            fragment.targets = vec![
                ColorTargetState {
                    format: rgba8_unorm(),
                    blend: None,
                    write_mask: 0xF,
                },
                ColorTargetState {
                    format: rgba8_unorm(),
                    blend: None,
                    write_mask: 0xF,
                },
            ];
        }
        Arc::new(device.create_render_pipeline(descriptor))
    }

    fn storage_write_render_pipeline(device: &crate::device::Device) -> Arc<RenderPipeline> {
        render_pipeline_from_shader(
            device,
            r"
@group(0) @binding(0) var<storage, read_write> output_buffer: array<u32>;

@vertex
fn vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    output_buffer[0] = 1u;
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
",
        )
    }

    fn uniform_read_render_pipeline(device: &crate::device::Device) -> Arc<RenderPipeline> {
        render_pipeline_from_shader(
            device,
            r"
struct Params {
    value: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;

@vertex
fn vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    let value = f32(params.value.x);
    return vec4<f32>(value, 0.0, 0.0, 1.0);
}
",
        )
    }

    fn storage_write_vertex_render_pipeline(device: &crate::device::Device) -> Arc<RenderPipeline> {
        vertex_buffer_render_pipeline_from_shader(
            device,
            r"
@group(0) @binding(0) var<storage, read_write> output_buffer: array<u32>;

@vertex
fn vs(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    output_buffer[0] = 1u;
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
",
        )
    }

    fn uniform_read_vertex_render_pipeline(device: &crate::device::Device) -> Arc<RenderPipeline> {
        vertex_buffer_render_pipeline_from_shader(
            device,
            r"
struct Params {
    value: vec4<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;

@vertex
fn vs(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    let x = position.x + f32(params.value.x) * 0.0;
    return vec4<f32>(x, position.y, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
",
        )
    }

    fn storage_write_uniform_read_render_pipeline(
        device: &crate::device::Device,
    ) -> Arc<RenderPipeline> {
        render_pipeline_from_shader(
            device,
            r"
struct Params {
    value: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> output_buffer: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@vertex
fn vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    output_buffer[0] = params.value.x;
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
",
        )
    }

    fn render_pipeline_from_shader(
        device: &crate::device::Device,
        source: &str,
    ) -> Arc<RenderPipeline> {
        let module =
            Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(source.to_owned())));
        Arc::new(device.create_render_pipeline(render_pipeline_descriptor(module)))
    }

    fn vertex_buffer_render_pipeline_from_shader(
        device: &crate::device::Device,
        source: &str,
    ) -> Arc<RenderPipeline> {
        let module =
            Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(source.to_owned())));
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.vertex.buffer_count = 1;
        descriptor.vertex.buffers = vec![VertexBufferLayout {
            used: true,
            array_stride: 8,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![VertexAttribute {
                format: VertexFormat::from_raw(0x0000_001D),
                offset: 0,
                shader_location: 0,
            }],
        }];
        Arc::new(device.create_render_pipeline(descriptor))
    }

    #[test]
    fn render_pass_encoder_set_blend_constant_records_draw_constant() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_blend_constant(Color {
                r: 0.25,
                g: 0.5,
                b: 0.75,
                a: 1.0,
            }),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        let CommandExecution::RenderPass(command) = &command_buffer.command_ops()[0] else {
            panic!("expected render pass command");
        };
        assert_eq!(command.blend_constant, [0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn render_pass_encoder_set_stencil_reference_records_draw_reference() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.set_stencil_reference(37), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        let CommandExecution::RenderPass(command) = &command_buffer.command_ops()[0] else {
            panic!("expected render pass command");
        };
        assert_eq!(command.stencil_reference, 37);
    }

    #[test]
    fn render_pass_encoder_enforces_max_draw_count_at_finish() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let view = noop_render_attachment(&device);
        let mut descriptor = noop_render_pass_descriptor(view, None);
        descriptor.max_draw_count = 1;
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&descriptor);
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("render pass draw count exceeds maxDrawCount".to_owned())
        );

        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let view = noop_render_attachment(&device);
        let mut descriptor = noop_render_pass_descriptor(view, None);
        descriptor.max_draw_count = 1;
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_render_pass(&descriptor);
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 1);
    }

    #[test]
    fn render_pass_encoder_indexed_draw_requires_index_buffer() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.draw_indexed(3, 1, 0, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(
            error,
            Some("render pass indexed draw requires an index buffer".to_owned())
        );
        assert!(command_buffer.is_error());
    }

    #[test]
    fn render_pass_encoder_indexed_and_indirect_draws() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let index_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::INDEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let indirect = noop_indirect_buffer(&device);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_index_buffer(index_buffer, Some(IndexFormat::Uint16), 0, 16),
            None
        );
        assert_eq!(pass.draw_indexed(3, 1, 0, -2, 0, device.limits()), None);
        assert_eq!(
            pass.draw_indirect(indirect.clone(), 0, device.limits()),
            None
        );
        assert_eq!(
            pass.draw_indexed_indirect(indirect, 0, device.limits()),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 3);
        let CommandExecution::RenderPass(indexed) = &command_buffer.command_ops()[0] else {
            panic!("expected indexed render pass command");
        };
        assert!(indexed.index_buffer.is_some());
        assert!(indexed.indirect_buffer.is_none());
        assert!(matches!(
            indexed.draw,
            Some(RenderDrawExecution::Indexed {
                index_count: 3,
                instance_count: 1,
                first_index: 0,
                base_vertex: -2,
                first_instance: 0,
            })
        ));
        let CommandExecution::RenderPass(indirect_draw) = &command_buffer.command_ops()[1] else {
            panic!("expected indirect render pass command");
        };
        assert!(indirect_draw.indirect_buffer.is_some());
        assert!(matches!(
            indirect_draw.draw,
            Some(RenderDrawExecution::Indirect { offset: 0 })
        ));
        let CommandExecution::RenderPass(indexed_indirect) = &command_buffer.command_ops()[2]
        else {
            panic!("expected indexed indirect render pass command");
        };
        assert!(indexed_indirect.index_buffer.is_some());
        assert!(indexed_indirect.indirect_buffer.is_some());
        assert!(matches!(
            indexed_indirect.draw,
            Some(RenderDrawExecution::IndexedIndirect { offset: 0 })
        ));
    }

    #[test]
    fn render_pass_encoder_state_setters_occlusion_query_and_execute_bundles() {
        let device = noop_device();
        let view = noop_render_attachment(&device);
        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "occlusion".to_owned(),
            kind: QueryType::Occlusion,
            count: 2,
        });
        assert_eq!(error, None);
        let (bundle_encoder, error) = RenderBundleEncoder::new(
            render_bundle_encoder_descriptor(),
            device.limits(),
            device.features(),
        );
        assert_eq!(error, None);
        let (bundle, error) = bundle_encoder.finish();
        assert_eq!(error, None);
        assert!(!bundle.is_error());
        let bundle = Arc::new(bundle);

        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, Some(query_set)));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_viewport(0.0, 0.0, 4.0, 4.0, 0.0, 1.0), None);
        assert_eq!(pass.set_scissor_rect(0, 0, 4, 4), None);
        assert_eq!(
            pass.set_blend_constant(Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            }),
            None
        );
        assert_eq!(pass.set_stencil_reference(1), None);
        assert_eq!(pass.begin_occlusion_query(0), None);
        assert_eq!(pass.end_occlusion_query(), None);
        assert_eq!(pass.execute_bundles(&[bundle]), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn render_pass_encoder_rejects_error_pipeline_at_set_pipeline() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.error = Some("forced render pipeline error".to_owned());
        let pipeline = Arc::new(device.create_render_pipeline(descriptor));
        assert!(pipeline.is_error());

        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("render pass requires a valid render pipeline".to_owned())
        );
    }

    #[test]
    fn render_pass_encoder_rejects_bind_group_index_at_set_bind_group() {
        let device = noop_device();
        let bind_group = empty_bind_group(&device);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(
            pass.set_bind_group(
                device.limits().max_bind_groups,
                Some(bind_group),
                Vec::new(),
                device.limits()
            ),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("bind group index exceeds the device limit".to_owned())
        );
    }

    #[test]
    fn render_pass_encoder_rejects_viewport_and_scissor_out_of_bounds() {
        let device = noop_device();
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        let max = device.limits().max_texture_dimension_2d as f32;
        assert_eq!(pass.set_viewport(max * 2.0, 0.0, 1.0, 1.0, 0.0, 1.0), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("render pass viewport rectangle exceeds device bounds".to_owned())
        );

        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_scissor_rect(1, 0, 4, 4), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("render pass scissor rectangle exceeds attachment size".to_owned())
        );
    }

    fn depth24_plus_stencil8() -> TextureFormat {
        TextureFormat::from_raw(0x0000_002F)
    }

    fn depth24_plus_stencil8_attachment(device: &Device) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: depth24_plus_stencil8(),
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

    fn depth_stencil_render_pipeline(device: &Device) -> Arc<RenderPipeline> {
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@vertex
fn vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

struct FragmentOutput {
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
}

@fragment
fn fs() -> FragmentOutput {
    var output: FragmentOutput;
    output.a = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    output.b = vec4<f32>(0.0, 1.0, 0.0, 1.0);
    return output;
}
"
                .to_owned(),
            )),
        );
        let mut descriptor = RenderPipelineDescriptor {
            layout: RenderPipelineLayout::Auto,
            vertex: RenderPipelineVertexState {
                shader: RenderPipelineShaderStage {
                    module: Arc::clone(&module),
                    entry_point: Some("vs".to_owned()),
                    constants: Vec::new(),
                },
                buffer_count: 0,
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::None,
                unclipped_depth: false,
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
                target_count: 2,
                targets: vec![
                    ColorTargetState {
                        format: rgba8_unorm(),
                        blend: None,
                        write_mask: 0xF,
                    },
                    ColorTargetState {
                        format: rgba8_unorm(),
                        blend: None,
                        write_mask: 0xF,
                    },
                ],
            }),
            error: None,
        };
        descriptor.depth_stencil = Some(DepthStencilState {
            format: depth24_plus_stencil8(),
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
        });
        Arc::new(device.create_render_pipeline(descriptor))
    }

    fn assert_render_pass_attachment_ops(
        command: &RenderPassCommand,
        load_op: LoadOp,
        store_op: StoreOp,
    ) {
        assert_eq!(command.color_attachments.len(), 2);
        for color_attachment in &command.color_attachments {
            let color_attachment = color_attachment.as_ref().expect("color attachment");
            assert_eq!(color_attachment.load_op, load_op);
            assert_eq!(color_attachment.store_op, store_op);
        }
        let depth_stencil = command
            .depth_stencil_attachment
            .as_ref()
            .expect("depth-stencil attachment");
        assert_eq!(depth_stencil.depth_load_op, load_op);
        assert_eq!(depth_stencil.depth_store_op, store_op);
        assert_eq!(depth_stencil.stencil_load_op, load_op);
        assert_eq!(depth_stencil.stencil_store_op, store_op);
    }
}
