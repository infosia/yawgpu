use std::sync::Arc;

use crate::bind_group::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::compute_pipeline::*;
use crate::limits::*;
use crate::pass::*;

/// Records commands for the ComputePassEncoder.
#[derive(Debug, Clone)]
pub struct ComputePassEncoder {
    pub(crate) inner: Arc<PassEncoderInner>,
}

impl ComputePassEncoder {
    /// Ends recording for this pass or encoder.
    pub fn end(&self) -> Option<String> {
        self.inner.end()
    }

    /// Records a debug marker within the compute pass.
    pub fn insert_debug_marker(&self) -> Option<String> {
        self.inner.insert_debug_marker()
    }

    /// Opens a debug group within the compute pass.
    pub fn push_debug_group(&self) -> Option<String> {
        self.inner.push_debug_group()
    }

    /// Closes the most recently opened debug group in the compute pass.
    pub fn pop_debug_group(&self) -> Option<String> {
        self.inner.pop_debug_group()
    }

    /// Sets pipeline on this object or encoder.
    pub fn set_pipeline(&self, pipeline: Arc<ComputePipeline>) -> Option<String> {
        self.inner.record_pass_command(|state| {
            if pipeline.is_error() {
                return Err("compute pass requires a valid compute pipeline".to_owned());
            }
            state.compute_pipeline = Some(pipeline);
            Ok(())
        })
    }

    /// Records a validation error against the compute pass.
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

    /// Records a workgroup dispatch after validating the bound pipeline and limits.
    pub fn dispatch_workgroups(&self, x: u32, y: u32, z: u32, limits: Limits) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_compute_dispatch_state(state, limits)?;
            if x > limits.max_compute_workgroups_per_dimension
                || y > limits.max_compute_workgroups_per_dimension
                || z > limits.max_compute_workgroups_per_dimension
            {
                return Err("compute dispatch workgroup count exceeds the device limit".to_owned());
            }
            let pipeline = Arc::clone(
                state
                    .compute_pipeline
                    .as_ref()
                    .ok_or_else(|| "compute dispatch requires a compute pipeline".to_owned())?,
            );
            self.inner.parent.record_compute_pass(ComputePassCommand {
                pipeline,
                bind_groups: state.bind_groups.clone(),
                workgroups: (x, y, z),
            });
            Ok(())
        })
    }

    /// Records an indirect workgroup dispatch sourced from a buffer after validation.
    pub fn dispatch_workgroups_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_compute_dispatch_state(state, limits)?;
            validate_indirect_buffer(
                &indirect_buffer,
                indirect_offset,
                12,
                "dispatch workgroups indirect",
            )?;
            self.inner.parent.record_referenced_buffer(indirect_buffer);
            Ok(())
        })
    }
}

/// Validates compute dispatch state and returns a descriptive error on failure.
pub(crate) fn validate_compute_dispatch_state(
    state: &PassEncoderState,
    limits: Limits,
) -> Result<(), String> {
    let Some(pipeline) = &state.compute_pipeline else {
        return Err("compute dispatch requires a compute pipeline".to_owned());
    };
    if pipeline.is_error() {
        return Err("compute dispatch requires a valid compute pipeline".to_owned());
    }
    validate_pipeline_bind_groups(pipeline.bind_group_layouts(), &state.bind_groups, limits)?;
    validate_usage_scope(pipeline.bind_group_layouts(), &state.bind_groups, None)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::shader::SHADER_STAGE_COMPUTE;
    use crate::test_helpers::*;
    use crate::{
        BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindGroupResource,
        BindingLayoutKind, BufferBindingType, ComputePipelineLayout, PipelineLayoutDescriptor,
        ShaderModuleSource,
    };

    #[test]
    fn compute_pass_encoder_lifecycle_and_debug_markers() {
        let encoder = noop_device().create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.push_debug_group(), None);
        assert_eq!(pass.insert_debug_marker(), None);
        assert_eq!(pass.pop_debug_group(), None);
        assert_eq!(
            pass.record_validation_error("forced compute pass error"),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(error, Some("forced compute pass error".to_owned()));
    }

    #[test]
    fn compute_pass_double_end_does_not_poison_parent_encoder() {
        let encoder = noop_device().create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.end(), None);
        assert_eq!(
            pass.end(),
            Some("pass encoder cannot be ended more than once".to_owned())
        );

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn compute_pass_encoder_pipeline_bind_group_and_dispatch() {
        let device = noop_device();
        let pipeline = noop_compute_pipeline(&device);
        let bind_group = empty_bind_group(&device);
        let indirect = noop_indirect_buffer(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(
            pass.dispatch_workgroups_indirect(indirect, 0, device.limits()),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 1);
    }

    #[test]
    fn compute_pass_direct_dispatches_have_separate_usage_scopes() {
        let device = noop_device();
        let bind_group_layout =
            Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: SHADER_STAGE_COMPUTE,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::Buffer {
                        ty: BufferBindingType::Storage,
                        has_dynamic_offset: false,
                        min_binding_size: 4,
                    }),
                }],
                error: None,
            }));
        let pipeline_layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::clone(&bind_group_layout)],
            immediate_size: 0,
            error: None,
        }));
        let storage = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE,
            size: 4,
            mapped_at_creation: false,
        }));
        let bind_group = Arc::new(device.create_bind_group(
            bind_group_layout,
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::Buffer {
                    buffer: storage,
                    device: Arc::new(device.clone()),
                    offset: 0,
                    size: 4,
                },
            }],
        ));
        assert!(!bind_group.is_error());
        let pipeline_a = storage_compute_pipeline(&device, Arc::clone(&pipeline_layout));
        let pipeline_b = storage_compute_pipeline(&device, pipeline_layout);
        assert!(!pipeline_a.is_error());
        assert!(!pipeline_b.is_error());

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline_a), None);
        assert_eq!(
            pass.set_bind_group(
                0,
                Some(Arc::clone(&bind_group)),
                Vec::new(),
                device.limits()
            ),
            None
        );
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.set_pipeline(pipeline_b), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 2);
        assert!(command_buffer
            .command_ops()
            .iter()
            .all(|op| matches!(op, CommandExecution::ComputePass(_))));
    }

    fn storage_compute_pipeline(
        device: &crate::device::Device,
        layout: Arc<crate::pipeline_layout::PipelineLayout>,
    ) -> Arc<ComputePipeline> {
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@group(0) @binding(0) var<storage, read_write> values: array<u32>;

@compute @workgroup_size(1)
fn cs() {
    values[0] = 1u;
}
"
                .to_owned(),
            )),
        );
        Arc::new(device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        }))
    }

    #[test]
    fn compute_pass_encoder_rejects_error_pipeline_at_set_pipeline() {
        let device = noop_device();
        let module = compute_shader_module(&device);
        let mut descriptor = compute_pipeline_descriptor(module);
        descriptor.error = Some("forced compute pipeline error".to_owned());
        let pipeline = Arc::new(device.create_compute_pipeline(descriptor));
        assert!(pipeline.is_error());

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("compute pass requires a valid compute pipeline".to_owned())
        );
    }

    #[test]
    fn compute_pass_encoder_rejects_invalid_bind_group_index_and_dynamic_offsets() {
        let device = noop_device();
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: SHADER_STAGE_COMPUTE,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: 4,
                }),
            }],
            error: None,
        }));
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::UNIFORM,
            size: 260,
            mapped_at_creation: false,
        }));
        let bind_group = Arc::new(device.create_bind_group(
            layout,
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::Buffer {
                    buffer,
                    device: Arc::new(device.clone()),
                    offset: 0,
                    size: 4,
                },
            }],
        ));
        assert!(!bind_group.is_error());

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(
            pass.set_bind_group(
                device.limits().max_bind_groups,
                Some(Arc::clone(&bind_group)),
                vec![256],
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

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), vec![32], device.limits()),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("bind group dynamic offset is not aligned".to_owned())
        );
    }
}
