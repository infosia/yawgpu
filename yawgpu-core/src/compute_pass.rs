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
                dispatch: ComputeDispatch::Direct {
                    workgroups: (x, y, z),
                },
                immediate_data: state.immediate_data.clone(),
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
            self.inner
                .parent
                .record_referenced_buffer(Arc::clone(&indirect_buffer));
            let pipeline = Arc::clone(
                state
                    .compute_pipeline
                    .as_ref()
                    .ok_or_else(|| "compute dispatch requires a compute pipeline".to_owned())?,
            );
            let (mut buffer_uses, texture_uses) =
                collect_pipeline_usage_scope(pipeline.bind_group_layouts(), &state.bind_groups)?;
            buffer_uses.push(BufferScopeUse {
                buffer: Arc::clone(&indirect_buffer),
                offset: indirect_offset,
                size: 12,
                access: ResourceAccess::Read,
            });
            validate_resource_usage_scope(&buffer_uses, &texture_uses)?;
            self.inner.parent.record_compute_pass(ComputePassCommand {
                pipeline,
                bind_groups: state.bind_groups.clone(),
                dispatch: ComputeDispatch::Indirect {
                    buffer: indirect_buffer,
                    offset: indirect_offset,
                },
                immediate_data: state.immediate_data.clone(),
            });
            Ok(())
        })
    }

    /// Overwrites `[offset, offset + data.len())` of the pass's user-immediates
    /// scratch (Block 94). Mirrors the placement/state/error conventions of
    /// [`Self::set_bind_group`]: validation failures route to
    /// [`PassEncoderInner::record_pass_command`] as a captured validation
    /// error that invalidates the encoder, never a panic. Contents persist
    /// across pipeline changes within the pass (Dawn:
    /// `dawn/native/ImmediatesTracker.h:81-87`). A `size == 0` write is
    /// validated (offset alignment/bounds still apply) but otherwise a no-op,
    /// matching Dawn's `ComputePassEncoder::APISetImmediates`
    /// (`dawn/native/ComputePassEncoder.cpp:554-578`).
    pub fn set_immediates(&self, offset: u32, data: &[u8], limits: Limits) -> Option<String> {
        self.inner.record_pass_command(|state| {
            let size =
                u32::try_from(data.len()).map_err(|_| "immediates size is too large".to_owned())?;
            validate_set_immediates(offset, size, limits)?;
            if size == 0 {
                return Ok(());
            }
            record_set_immediates(state, offset, data);
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
    validate_usage_scope(pipeline.bind_group_layouts(), &state.bind_groups, None)?;
    validate_required_immediate_data(
        pipeline.immediate_required_mask(),
        state.immediate_data_written,
    )
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::shader::SHADER_STAGE_COMPUTE;
    use crate::test_helpers::*;
    use crate::{
        BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindGroupResource,
        BindingLayoutKind, BufferBindingType, ComputePipeline, ComputePipelineLayout, Device,
        Extent3d, PipelineLayoutDescriptor, ShaderModuleSource, StorageTextureAccess,
        TextureDescriptor, TextureDimension, TextureUsage, TextureViewDescriptor,
        TextureViewDimension,
    };

    fn immediate_compute_pipeline(device: &Device, wgsl: &str) -> Arc<ComputePipeline> {
        let module =
            Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(wgsl.to_owned())));
        assert!(!module.is_error(), "shader module must compile");
        let pipeline =
            Arc::new(device.create_compute_pipeline(compute_pipeline_descriptor(module)));
        assert!(!pipeline.is_error(), "compute pipeline must be valid");
        pipeline
    }

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
        let indirect_for_dispatch = Arc::clone(&indirect);
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
        assert_eq!(command_buffer.command_ops().len(), 2);
        let CommandExecution::ComputePass(command) = &command_buffer.command_ops()[0] else {
            panic!("expected direct compute pass command");
        };
        assert!(matches!(
            command.dispatch,
            ComputeDispatch::Direct {
                workgroups: (1, 1, 1)
            }
        ));
        let CommandExecution::ComputePass(command) = &command_buffer.command_ops()[1] else {
            panic!("expected indirect compute pass command");
        };
        match &command.dispatch {
            ComputeDispatch::Indirect { buffer, offset } => {
                assert!(Arc::ptr_eq(buffer, &indirect_for_dispatch));
                assert_eq!(*offset, 0);
            }
            ComputeDispatch::Direct { .. } => panic!("expected indirect dispatch"),
        }
    }

    /// WebGPU: `dispatchWorkgroups` with any workgroup count of zero is valid
    /// and does nothing. Core must accept and record it without a validation
    /// error -- skipping the empty dispatch is the backends' job, never a
    /// core-side rejection.
    #[test]
    fn compute_pass_dispatch_workgroups_accepts_zero_workgroup_counts() {
        let device = noop_device();
        let pipeline = noop_compute_pipeline(&device);
        let bind_group = empty_bind_group(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.dispatch_workgroups(0, 1, 1, device.limits()), None);
        assert_eq!(pass.dispatch_workgroups(1, 0, 1, device.limits()), None);
        assert_eq!(pass.dispatch_workgroups(1, 1, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 3);
        for (op, expected) in
            command_buffer
                .command_ops()
                .iter()
                .zip([(0, 1, 1), (1, 0, 1), (1, 1, 0)])
        {
            let CommandExecution::ComputePass(command) = op else {
                panic!("expected direct compute pass command");
            };
            let ComputeDispatch::Direct { workgroups } = command.dispatch else {
                panic!("expected direct dispatch");
            };
            assert_eq!(workgroups, expected);
        }
    }

    /// Block 94 S1 happy path: `SetImmediates` within the Noop device's
    /// `max_immediate_size` (64) records into the dispatch's immediates
    /// snapshot, and contents persist across a `set_pipeline` swap
    /// (Dawn: `dawn/native/ImmediatesTracker.h:81-87` -- the scratch is
    /// pass-scoped, not pipeline-scoped).
    #[test]
    fn compute_pass_set_immediates_happy_path_and_persists_across_set_pipeline() {
        let device = noop_device();
        assert_eq!(device.limits().max_immediate_size, 64);
        let pipeline_a = noop_compute_pipeline(&device);
        let pipeline_b = noop_compute_pipeline(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline_a), None);
        assert_eq!(pass.set_immediates(0, &[1, 2, 3, 4], device.limits()), None);
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        // Swap pipelines: the immediates scratch must survive.
        assert_eq!(pass.set_pipeline(pipeline_b), None);
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 2);
        for op in command_buffer.command_ops() {
            let CommandExecution::ComputePass(command) = op else {
                panic!("expected compute pass command");
            };
            assert_eq!(command.immediate_data.len(), 64);
            assert_eq!(&command.immediate_data[0..4], &[1, 2, 3, 4]);
            assert!(command.immediate_data[4..].iter().all(|byte| *byte == 0));
        }
    }

    /// Block 94 S1: each `ValidateSetImmediates` rule
    /// (`dawn/native/ProgrammableEncoder.cpp:128-146`) fires as a captured
    /// validation error that invalidates the encoder, never a panic.
    #[test]
    fn compute_pass_set_immediates_rejects_unaligned_offset_size_and_out_of_limit() {
        let device = noop_device();

        let unaligned_offset = || {
            let encoder = device.create_command_encoder();
            let (pass, _) = encoder.begin_compute_pass();
            assert_eq!(pass.set_immediates(1, &[1, 2, 3, 4], device.limits()), None);
            assert_eq!(pass.end(), None);
            encoder.finish().1
        };
        assert_eq!(
            unaligned_offset(),
            Some("immediates offset must be 4-byte aligned".to_owned())
        );

        let unaligned_size = || {
            let encoder = device.create_command_encoder();
            let (pass, _) = encoder.begin_compute_pass();
            assert_eq!(pass.set_immediates(0, &[1, 2, 3], device.limits()), None);
            assert_eq!(pass.end(), None);
            encoder.finish().1
        };
        assert_eq!(
            unaligned_size(),
            Some("immediates size must be 4-byte aligned".to_owned())
        );

        let offset_out_of_limit = || {
            let encoder = device.create_command_encoder();
            let (pass, _) = encoder.begin_compute_pass();
            assert_eq!(pass.set_immediates(68, &[], device.limits()), None);
            assert_eq!(pass.end(), None);
            encoder.finish().1
        };
        assert_eq!(
            offset_out_of_limit(),
            Some("immediates offset exceeds the device limit".to_owned())
        );

        let range_out_of_limit = || {
            let encoder = device.create_command_encoder();
            let (pass, _) = encoder.begin_compute_pass();
            assert_eq!(pass.set_immediates(60, &[0; 8], device.limits()), None);
            assert_eq!(pass.end(), None);
            encoder.finish().1
        };
        assert_eq!(
            range_out_of_limit(),
            Some("immediates offset plus size exceeds the device limit".to_owned())
        );
    }

    #[test]
    fn compute_pass_dispatch_requires_all_required_immediate_slots() {
        let device = noop_device();
        let pipeline = immediate_compute_pipeline(
            &device,
            r#"
requires immediate_address_space;
var<immediate> params : vec4u;

@compute @workgroup_size(1)
fn cs() {
  _ = params;
}
"#,
        );

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.set_immediates(0, &[1, 2, 3, 4], device.limits()), None);
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("Required immediate data at offset 4 was not set.".to_owned())
        );
    }

    #[test]
    fn compute_pass_dispatch_accepts_full_split_and_overprovisioned_immediates() {
        let device = noop_device();
        let vec4_pipeline = immediate_compute_pipeline(
            &device,
            r#"
requires immediate_address_space;
var<immediate> params : vec4u;

@compute @workgroup_size(1)
fn cs() {
  _ = params;
}
"#,
        );
        let scalar_pipeline = immediate_compute_pipeline(
            &device,
            r#"
requires immediate_address_space;
var<immediate> param : u32;

@compute @workgroup_size(1)
fn cs() {
  _ = param;
}
"#,
        );

        let dispatch_with_writes = |pipeline: Arc<ComputePipeline>, writes: Vec<(u32, Vec<u8>)>| {
            let encoder = device.create_command_encoder();
            let (pass, begin_error) = encoder.begin_compute_pass();
            assert_eq!(begin_error, None);
            assert_eq!(pass.set_pipeline(pipeline), None);
            for (offset, data) in writes {
                assert_eq!(pass.set_immediates(offset, &data, device.limits()), None);
            }
            assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
            assert_eq!(pass.end(), None);
            encoder.finish()
        };

        let (command_buffer, error) =
            dispatch_with_writes(Arc::clone(&vec4_pipeline), vec![(0, vec![1; 16])]);
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());

        let (command_buffer, error) = dispatch_with_writes(
            Arc::clone(&vec4_pipeline),
            vec![(0, vec![1; 8]), (8, vec![2; 8])],
        );
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());

        let (command_buffer, error) = dispatch_with_writes(scalar_pipeline, vec![(0, vec![3; 16])]);
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn compute_pass_dispatch_does_not_require_declared_but_unused_immediate() {
        let device = noop_device();
        let pipeline = immediate_compute_pipeline(
            &device,
            r#"
requires immediate_address_space;
var<immediate> unused_param : vec4u;

@compute @workgroup_size(1)
fn cs() {}
"#,
        );

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn compute_pass_dispatch_does_not_require_immediate_padding_slots() {
        let device = noop_device();
        let pipeline = immediate_compute_pipeline(
            &device,
            r#"
requires immediate_address_space;

struct Params {
  a : u32,
  b : vec3<f32>,
}

var<immediate> params : Params;

@compute @workgroup_size(1)
fn cs() {
  _ = params;
}
"#,
        );

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.set_immediates(0, &[1; 4], device.limits()), None);
        assert_eq!(pass.set_immediates(16, &[2; 12], device.limits()), None);
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    /// Block 94 S1: a `size == 0` write is validated (offset alignment/bounds
    /// still apply, per Dawn's `ValidateSetImmediates` running before the
    /// `size == 0` early return -- `dawn/native/ComputePassEncoder.cpp:559-565`)
    /// but is otherwise a no-op: it neither errors on a valid offset nor
    /// mutates the scratch.
    #[test]
    fn compute_pass_set_immediates_zero_size_is_validated_but_a_noop() {
        let device = noop_device();
        let pipeline = noop_compute_pipeline(&device);

        // Valid offset, zero size: succeeds and leaves the scratch untouched.
        let encoder = device.create_command_encoder();
        let (pass, _) = encoder.begin_compute_pass();
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.set_immediates(0, &[], device.limits()), None);
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        let CommandExecution::ComputePass(command) = &command_buffer.command_ops()[0] else {
            panic!("expected compute pass command");
        };
        assert!(command.immediate_data.iter().all(|byte| *byte == 0));

        // Invalid (unaligned) offset with zero size: still validated and
        // rejected, matching Dawn's unconditional `ValidateSetImmediates`.
        let encoder = device.create_command_encoder();
        let (pass, _) = encoder.begin_compute_pass();
        assert_eq!(pass.set_immediates(1, &[], device.limits()), None);
        assert_eq!(pass.end(), None);
        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("immediates offset must be 4-byte aligned".to_owned())
        );
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

    #[test]
    fn compute_pass_indirect_buffer_conflicts_with_storage_write() {
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
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::INDIRECT,
            size: 16,
            mapped_at_creation: false,
        }));
        let bind_group = Arc::new(device.create_bind_group(
            bind_group_layout,
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::Buffer {
                    buffer: Arc::clone(&buffer),
                    device: Arc::new(device.clone()),
                    offset: 0,
                    size: 16,
                },
            }],
        ));
        let pipeline = storage_compute_pipeline(&device, pipeline_layout);

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(
            pass.dispatch_workgroups_indirect(buffer, 0, device.limits()),
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
    fn compute_pass_read_only_buffer_uses_do_not_accumulate_across_dispatches() {
        let device = noop_device();
        let bind_group_layout =
            Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: SHADER_STAGE_COMPUTE,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::Buffer {
                        ty: BufferBindingType::Uniform,
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
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::UNIFORM,
            size: 16,
            mapped_at_creation: false,
        }));
        let bind_group = Arc::new(device.create_bind_group(
            bind_group_layout,
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::Buffer {
                    buffer,
                    device: Arc::new(device.clone()),
                    offset: 0,
                    size: 16,
                },
            }],
        ));
        let pipeline_a = uniform_compute_pipeline(&device, Arc::clone(&pipeline_layout));
        let pipeline_b = uniform_compute_pipeline(&device, pipeline_layout);

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
    }

    #[test]
    fn compute_pass_storage_texture_uses_do_not_accumulate_across_dispatches() {
        let device = noop_device();
        let bind_group_layout =
            Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: SHADER_STAGE_COMPUTE,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: rgba8_unorm(),
                        view_dimension: TextureViewDimension::D2,
                    }),
                }],
                error: None,
            }));
        let pipeline_layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::clone(&bind_group_layout)],
            immediate_size: 0,
            error: None,
        }));
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::STORAGE_BINDING,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);
        let bind_group = Arc::new(device.create_bind_group(
            bind_group_layout,
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::TextureView {
                    texture_view: Arc::new(view),
                    device: Arc::new(device.clone()),
                },
            }],
        ));
        let pipeline_a = storage_texture_compute_pipeline(&device, Arc::clone(&pipeline_layout));
        let pipeline_b = storage_texture_compute_pipeline(&device, pipeline_layout);

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
    }

    #[test]
    fn compute_pass_rejects_same_storage_texture_write_kind_in_one_dispatch() {
        let device = noop_device();
        let bind_group_layout =
            Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
                entries: vec![
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: SHADER_STAGE_COMPUTE,
                        binding_array_size: 0,
                        kind: Some(BindingLayoutKind::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: rgba8_unorm(),
                            view_dimension: TextureViewDimension::D2,
                        }),
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: SHADER_STAGE_COMPUTE,
                        binding_array_size: 0,
                        kind: Some(BindingLayoutKind::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: rgba8_unorm(),
                            view_dimension: TextureViewDimension::D2,
                        }),
                    },
                ],
                error: None,
            }));
        let pipeline_layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::clone(&bind_group_layout)],
            immediate_size: 0,
            error: None,
        }));
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::STORAGE_BINDING,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);
        let view = Arc::new(view);
        let bind_group = Arc::new(device.create_bind_group(
            bind_group_layout,
            vec![
                BindGroupEntry {
                    binding: 0,
                    resource: BindGroupResource::TextureView {
                        texture_view: Arc::clone(&view),
                        device: Arc::new(device.clone()),
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindGroupResource::TextureView {
                        texture_view: view,
                        device: Arc::new(device.clone()),
                    },
                },
            ],
        ));
        let pipeline = aliasing_storage_textures_compute_pipeline(&device, pipeline_layout);

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some(
                "usage scope cannot read and write or write the same texture subresource twice"
                    .to_owned()
            )
        );
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

    fn storage_texture_compute_pipeline(
        device: &crate::device::Device,
        layout: Arc<crate::pipeline_layout::PipelineLayout>,
    ) -> Arc<ComputePipeline> {
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1)
fn cs() {
    textureStore(output_texture, vec2<i32>(0, 0), vec4<f32>(1.0, 0.0, 0.0, 1.0));
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

    fn aliasing_storage_textures_compute_pipeline(
        device: &crate::device::Device,
        layout: Arc<crate::pipeline_layout::PipelineLayout>,
    ) -> Arc<ComputePipeline> {
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@group(0) @binding(0) var output_texture_a: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var output_texture_b: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1)
fn cs() {
    textureStore(output_texture_a, vec2<i32>(0, 0), vec4<f32>(1.0, 0.0, 0.0, 1.0));
    textureStore(output_texture_b, vec2<i32>(0, 0), vec4<f32>(0.0, 1.0, 0.0, 1.0));
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

    fn uniform_compute_pipeline(
        device: &crate::device::Device,
        layout: Arc<crate::pipeline_layout::PipelineLayout>,
    ) -> Arc<ComputePipeline> {
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
struct Params {
    value: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(1)
fn cs() {
    let value = params.value;
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
