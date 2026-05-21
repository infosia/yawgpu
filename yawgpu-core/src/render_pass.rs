use std::sync::Arc;

use crate::bind_group::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::copy::*;
use crate::limits::*;
use crate::pass::*;
use crate::query_set::*;
use crate::render_bundle::*;
use crate::render_pipeline::*;
use crate::texture_view::*;

#[derive(Debug, Clone)]
pub struct RenderPassDescriptor {
    pub max_color_attachments: u32,
    pub color_attachments: Vec<Option<RenderPassColorAttachment>>,
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment>,
    pub occlusion_query_set: Option<QuerySet>,
    pub timestamp_writes: Option<RenderPassTimestampWrites>,
}

#[derive(Debug, Clone)]
pub struct RenderPassTimestampWrites {
    pub query_set: QuerySet,
    pub beginning_index: Option<u32>,
    pub end_index: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct RenderPassColorAttachment {
    pub view: Arc<TextureView>,
    pub resolve_target: Option<Arc<TextureView>>,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_value: Color,
}

#[derive(Debug, Clone)]
pub struct RenderPassDepthStencilAttachment {
    pub view: Arc<TextureView>,
    pub depth_load_op: LoadOp,
    pub depth_store_op: StoreOp,
    pub depth_clear_value: f32,
    pub stencil_load_op: LoadOp,
    pub stencil_store_op: StoreOp,
}

#[derive(Debug, Clone)]
pub struct RenderPassEncoder {
    pub(crate) inner: Arc<PassEncoderInner>,
}

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

pub(crate) fn validate_query_set_alive(query_set: &QuerySet, usage: &str) -> Result<(), String> {
    if query_set.is_error() {
        return Err(format!("{usage} cannot use an error query set"));
    }
    if query_set.is_destroyed() {
        return Err(format!("{usage} cannot use a destroyed query set"));
    }
    Ok(())
}

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

impl RenderPassEncoder {
    pub fn end(&self) -> Option<String> {
        self.inner.end()
    }

    pub fn insert_debug_marker(&self) -> Option<String> {
        self.inner.insert_debug_marker()
    }

    pub fn push_debug_group(&self) -> Option<String> {
        self.inner.push_debug_group()
    }

    pub fn pop_debug_group(&self) -> Option<String> {
        self.inner.pop_debug_group()
    }

    pub fn set_pipeline(&self, pipeline: Arc<RenderPipeline>) -> Option<String> {
        self.inner.record_pass_command(|state| {
            state.render_pipeline = Some(pipeline);
            Ok(())
        })
    }

    pub fn record_validation_error(&self, message: impl Into<String>) -> Option<String> {
        self.inner.record_pass_command(|_| Err(message.into()))
    }

    pub fn set_bind_group(
        &self,
        index: u32,
        group: Option<Arc<BindGroup>>,
        dynamic_offsets: Vec<u32>,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            if let Some(group) = group {
                self.inner
                    .parent
                    .record_referenced_buffers(bind_group_buffer_resources(&group));
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
        self.inner.record_pass_command(|state| {
            validate_vertex_buffer_slot(slot, limits)?;
            if let Some(buffer) = buffer {
                let size = validate_set_vertex_buffer(&buffer, offset, size)?;
                self.inner
                    .parent
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
            let pipeline = state
                .render_pipeline
                .as_ref()
                .ok_or_else(|| "render pass requires a render pipeline".to_owned())?;
            let color_attachment = state
                .render_color_attachment
                .clone()
                .ok_or_else(|| "render pass requires a color attachment".to_owned())?;
            self.inner.parent.record_render_pass(RenderPassCommand {
                pipeline: Some(Arc::clone(pipeline)),
                color_attachment,
                bind_groups: state.bind_groups.clone(),
                vertex_buffers: state.vertex_buffers.clone(),
                draw: Some(RenderDrawExecution {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                }),
            });
            state.render_pass_recorded = true;
            Ok(())
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
            )
        })
    }

    pub fn draw_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::Indirect, limits)?;
            validate_indirect_buffer(&indirect_buffer, indirect_offset, 16, "draw indirect")?;
            self.inner.parent.record_referenced_buffer(indirect_buffer);
            Ok(())
        })
    }

    pub fn draw_indexed_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::IndexedIndirect, limits)?;
            validate_indirect_buffer(
                &indirect_buffer,
                indirect_offset,
                20,
                "draw indexed indirect",
            )?;
            self.inner.parent.record_referenced_buffer(indirect_buffer);
            Ok(())
        })
    }

    pub fn set_viewport(
        &self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Option<String> {
        self.inner
            .record_pass_command(|_| validate_viewport(x, y, width, height, min_depth, max_depth))
    }

    pub fn set_scissor_rect(&self, x: u32, y: u32, width: u32, height: u32) -> Option<String> {
        self.inner.record_pass_command(|_| {
            x.checked_add(width)
                .ok_or_else(|| "render pass scissor rectangle width overflows".to_owned())?;
            y.checked_add(height)
                .ok_or_else(|| "render pass scissor rectangle height overflows".to_owned())?;
            Ok(())
        })
    }

    pub fn set_blend_constant(&self, color: Color) -> Option<String> {
        self.inner.record_pass_command(|_| {
            if [color.r, color.g, color.b, color.a]
                .into_iter()
                .all(f64::is_finite)
            {
                Ok(())
            } else {
                Err("render pass blend constant components must be finite".to_owned())
            }
        })
    }

    pub fn set_stencil_reference(&self, _reference: u32) -> Option<String> {
        self.inner.record_pass_command(|_| Ok(()))
    }

    pub fn execute_bundles(&self, bundles: &[Arc<RenderBundle>]) -> Option<String> {
        self.inner.record_pass_command(|state| {
            let pass_signature = state
                .attachment_signature
                .as_ref()
                .ok_or_else(|| "render pass has no attachment signature".to_owned())?;
            for bundle in bundles {
                if bundle.is_error() {
                    return Err("render pass cannot execute an error render bundle".to_owned());
                }
                if bundle.attachment_signature() != pass_signature {
                    return Err(
                        "render bundle attachment signature is incompatible with the render pass"
                            .to_owned(),
                    );
                }
            }
            state.clear_render_state();
            Ok(())
        })
    }

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
        assert_eq!(pass.set_bind_group(0, Some(bind_group), Vec::new()), None);
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
        assert_eq!(pass.draw_indexed(3, 1, 0, 0, 0, device.limits()), None);
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
        let (bundle_encoder, error) =
            RenderBundleEncoder::new(render_bundle_encoder_descriptor(), device.limits());
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
}
