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
            state.set_compute_pipeline(pipeline);
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
                state.set_bind_group(
                    index,
                    BoundBindGroup {
                        group,
                        dynamic_offsets,
                    },
                );
            } else {
                state.clear_bind_group(index);
            }
            Ok(())
        })
    }

    /// Records a workgroup dispatch after validating the bound pipeline and limits.
    pub fn dispatch_workgroups(&self, x: u32, y: u32, z: u32, limits: Limits) -> Option<String> {
        self.inner.record_pass_command(|state| {
            let (pipeline, _) = validate_compute_dispatch_state(state, limits)?;
            if x > limits.max_compute_workgroups_per_dimension
                || y > limits.max_compute_workgroups_per_dimension
                || z > limits.max_compute_workgroups_per_dimension
            {
                return Err("compute dispatch workgroup count exceeds the device limit".to_owned());
            }
            if let Some(invocations) = pipeline.linear_index_invocations() {
                validate_linear_index_range(invocations, x, y, z)?;
            }
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
            let (pipeline, buffer_uses) = validate_compute_dispatch_state(state, limits)?;
            validate_indirect_buffer(
                &indirect_buffer,
                indirect_offset,
                12,
                "dispatch workgroups indirect",
            )?;
            let indirect_use = BufferScopeUse {
                buffer: Arc::clone(&indirect_buffer),
                offset: indirect_offset,
                size: 12,
                access: ResourceAccess::Read,
            };
            validate_extra_buffer_uses_against_scope(
                &buffer_uses,
                std::slice::from_ref(&indirect_use),
            )?;
            self.inner
                .parent
                .record_referenced_buffer(Arc::clone(&indirect_buffer));
            // The counts are GPU data, so they are checked on the GPU. Usage
            // scopes above stay against the client buffer; the validation command
            // is internal and carries its own pipeline, bind group and immediates,
            // so the pass's state is untouched.
            let (validation, validated_buffer) =
                self.inner.parent.dispatch_indirect_validation_command(
                    indirect_buffer,
                    indirect_offset,
                    pipeline.linear_index_invocations(),
                    limits,
                )?;
            self.inner
                .parent
                .record_referenced_buffer(Arc::clone(&validated_buffer));
            self.inner.parent.record_compute_pass(validation);
            self.inner.parent.record_compute_pass(ComputePassCommand {
                pipeline,
                bind_groups: state.bind_groups.clone(),
                dispatch: ComputeDispatch::Indirect {
                    buffer: validated_buffer,
                    offset: 0,
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
    state: &mut PassEncoderState,
    limits: Limits,
) -> Result<(Arc<ComputePipeline>, Vec<BufferScopeUse>), String> {
    let Some(pipeline) = state.compute_pipeline.clone() else {
        return Err("compute dispatch requires a compute pipeline".to_owned());
    };
    if pipeline.is_error() {
        return Err("compute dispatch requires a valid compute pipeline".to_owned());
    }
    validate_pipeline_bind_groups_memoized(
        state,
        pipeline.bind_group_layouts(),
        pipeline.required_buffer_sizes(),
        limits,
    )?;
    let (buffer_uses, texture_uses) =
        collect_usage_scope(pipeline.bind_group_layouts(), &state.bind_groups)?;
    validate_resource_usage_scope(&buffer_uses, &texture_uses)?;
    validate_required_immediate_data(
        pipeline.immediate_required_mask(),
        state.immediate_data_written,
    )?;
    Ok((pipeline, buffer_uses))
}

/// Rejects a direct dispatch whose linear index would not fit in a `u32`.
/// `workgroup_invocations` is 1 for `workgroup_index`, which numbers workgroups.
/// Mirrors Dawn `ComputePassEncoder::APIDispatchWorkgroups`, which also rejects
/// on either partial product, so a zero count does not excuse the other factor.
fn validate_linear_index_range(
    workgroup_invocations: u64,
    x: u32,
    y: u32,
    z: u32,
) -> Result<(), String> {
    let max = u64::from(u32::MAX);
    let invocations_and_x = workgroup_invocations.checked_mul(u64::from(x));
    let y_and_z = u64::from(y) * u64::from(z);
    let in_range = invocations_and_x.is_some_and(|invocations_and_x| {
        invocations_and_x <= max
            && y_and_z <= max
            // Both factors are at most `u32::MAX`, so this cannot wrap.
            && invocations_and_x * y_and_z <= max
    });
    if in_range {
        Ok(())
    } else {
        Err(format!(
            "compute dispatch ({x}, {y}, {z}) with {workgroup_invocations} indexed invocations per workgroup exceeds the unsigned 32-bit linear index range"
        ))
    }
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
    fn compute_dispatch_storage_buffer_aliasing_and_dynamic_offsets() {
        let device = noop_device();
        let layout = Arc::new(
            device.create_bind_group_layout(BindGroupLayoutDescriptor {
                entries: [0, 3]
                    .into_iter()
                    .map(|binding| BindGroupLayoutEntry {
                        binding,
                        visibility: SHADER_STAGE_COMPUTE,
                        binding_array_size: 0,
                        kind: Some(BindingLayoutKind::Buffer {
                            ty: BufferBindingType::Storage,
                            has_dynamic_offset: true,
                            min_binding_size: 4,
                        }),
                    })
                    .collect(),
                error: None,
            }),
        );
        let pipeline_layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![layout.clone()],
            immediate_size: 0,
            error: None,
        }));
        let pipeline = storage_compute_pipeline(&device, pipeline_layout);
        assert!(!pipeline.is_error());
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE,
            size: 1024,
            mapped_at_creation: false,
        }));
        let group = Arc::new(
            device.create_bind_group(
                layout,
                [0, 3]
                    .into_iter()
                    .map(|binding| BindGroupEntry {
                        binding,
                        resource: BindGroupResource::Buffer {
                            buffer: buffer.clone(),
                            device: Arc::new(device.clone()),
                            offset: 0,
                            size: 256,
                        },
                    })
                    .collect(),
            ),
        );
        assert!(!group.is_error());
        for indirect in [None, Some(false), Some(true)] {
            let encoder = device.create_command_encoder();
            let (pass, error) = encoder.begin_compute_pass(None);
            assert_eq!(error, None);
            assert_eq!(pass.set_pipeline(pipeline.clone()), None);
            assert_eq!(
                pass.set_bind_group(0, Some(group.clone()), vec![0, 256], device.limits()),
                None
            );
            assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
            if indirect.is_none() {
                assert_eq!(pass.end(), None);
                let (commands, error) = encoder.finish();
                assert_eq!(error, None);
                assert!(!commands.is_error());
                continue;
            }
            // Same group identity, new offsets: the previously valid ranges now overlap.
            assert_eq!(
                pass.set_bind_group(0, Some(group.clone()), vec![256, 256], device.limits()),
                None
            );
            if indirect == Some(true) {
                assert_eq!(
                    pass.dispatch_workgroups_indirect(
                        noop_indirect_buffer(&device),
                        0,
                        device.limits()
                    ),
                    None
                );
            } else {
                assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
            }
            assert_eq!(pass.end(), None);
            let (commands, error) = encoder.finish();
            assert!(commands.is_error());
            assert_eq!(
                error.as_deref(),
                Some(
                    "writable storage buffer binding aliasing: group 0 binding 0 range [256, 512) overlaps group 0 binding 3 range [256, 512)"
                )
            );
        }
    }

    /// Read-only storage is excluded from writable binding alias checks, but
    /// still conflicts with writable storage in a dispatch's usage scope.
    #[test]
    fn compute_dispatch_read_only_and_writable_storage_share_buffer_usage_error() {
        let device = noop_device();
        let layout = Arc::new(
            device.create_bind_group_layout(BindGroupLayoutDescriptor {
                entries: [
                    BufferBindingType::ReadOnlyStorage,
                    BufferBindingType::Storage,
                ]
                .into_iter()
                .enumerate()
                .map(|(binding, ty)| BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: SHADER_STAGE_COMPUTE,
                    binding_array_size: 0,
                    kind: Some(BindingLayoutKind::Buffer {
                        ty,
                        has_dynamic_offset: false,
                        min_binding_size: 4,
                    }),
                })
                .collect(),
                error: None,
            }),
        );
        assert!(!layout.is_error());
        let pipeline_layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![Arc::clone(&layout)],
            immediate_size: 0,
            error: None,
        }));
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1)
fn cs() { output[0] = input[0]; }
"
                .to_owned(),
            )),
        );
        assert!(!module.is_error());
        let pipeline = Arc::new(device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(pipeline_layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        }));
        assert!(!pipeline.is_error());
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE,
            size: 256,
            mapped_at_creation: false,
        }));
        let group = Arc::new(
            device.create_bind_group(
                layout,
                (0..2)
                    .map(|binding| BindGroupEntry {
                        binding,
                        resource: BindGroupResource::Buffer {
                            buffer: Arc::clone(&buffer),
                            device: Arc::new(device.clone()),
                            offset: 0,
                            size: 256,
                        },
                    })
                    .collect(),
            ),
        );
        assert!(!group.is_error());
        for indirect in [false, true] {
            let encoder = device.create_command_encoder();
            let (pass, error) = encoder.begin_compute_pass(None);
            assert_eq!(error, None);
            assert_eq!(pass.set_pipeline(Arc::clone(&pipeline)), None);
            assert_eq!(
                pass.set_bind_group(0, Some(Arc::clone(&group)), vec![], device.limits()),
                None
            );
            if indirect {
                assert_eq!(
                    pass.dispatch_workgroups_indirect(
                        noop_indirect_buffer(&device),
                        0,
                        device.limits()
                    ),
                    None
                );
            } else {
                assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
            }
            assert_eq!(pass.end(), None);
            let (commands, error) = encoder.finish();
            assert!(commands.is_error());
            assert_eq!(
                error.as_deref(),
                Some("usage scope cannot read and write or write the same buffer range twice")
            );
        }
    }

    #[test]
    fn compute_pass_encoder_lifecycle_and_debug_markers() {
        let encoder = noop_device().create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
        assert_eq!(command_buffer.command_ops().len(), 3);
        let CommandExecution::ComputePass(command) = &command_buffer.command_ops()[0] else {
            panic!("expected direct compute pass command");
        };
        assert!(matches!(
            command.dispatch,
            ComputeDispatch::Direct {
                workgroups: (1, 1, 1)
            }
        ));
        // Block 119 R5a: the client buffer feeds the validation command, and the
        // user's dispatch reads the validated copy.
        let CommandExecution::ComputePass(command) = &command_buffer.command_ops()[1] else {
            panic!("expected indirect validation command");
        };
        assert!(Arc::ptr_eq(
            &validation_binding(command, 0).0,
            &indirect_for_dispatch
        ));
        let CommandExecution::ComputePass(command) = &command_buffer.command_ops()[2] else {
            panic!("expected indirect compute pass command");
        };
        match &command.dispatch {
            ComputeDispatch::Indirect { buffer, offset } => {
                assert!(Arc::ptr_eq(
                    buffer,
                    &device.dispatch_indirect_buffer().unwrap()
                ));
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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

    fn required_size_group(
        device: &Device,
        layout_min_binding_size: u64,
        ty: BufferBindingType,
        buffer_size: u64,
        offset: u64,
        size: u64,
    ) -> (Arc<ComputePipeline>, Arc<BindGroup>) {
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: SHADER_STAGE_COMPUTE,
                binding_array_size: 0,
                kind: Some(BindingLayoutKind::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: layout_min_binding_size,
                }),
            }],
            error: None,
        }));
        let pipeline_layout = Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![layout.clone()],
            immediate_size: 0,
            error: None,
        }));
        let declaration = match ty {
            BufferBindingType::Uniform => "var<uniform>",
            BufferBindingType::Storage => "var<storage, read_write>",
            _ => "var<storage, read>",
        };
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(format!(
                "@group(0) @binding(0) {declaration} data: array<vec4u, 2>;\n\
             @compute @workgroup_size(1) fn cs() {{ _ = data[0]; }}"
            ))),
        );
        let pipeline = Arc::new(device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(pipeline_layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        }));
        assert!(!pipeline.is_error());
        let usage = match ty {
            BufferBindingType::Uniform => BufferUsage::UNIFORM,
            _ => BufferUsage::STORAGE,
        };
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage,
            size: buffer_size,
            mapped_at_creation: false,
        }));
        let group = Arc::new(device.create_bind_group(
            layout,
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::Buffer {
                    buffer,
                    device: Arc::new(device.clone()),
                    offset,
                    size,
                },
            }],
        ));
        assert!(!group.is_error());
        (pipeline, group)
    }

    fn dispatch_error(
        device: &Device,
        pipeline: Arc<ComputePipeline>,
        group: Option<Arc<BindGroup>>,
        indirect: bool,
        counts: (u32, u32, u32),
        limits: Limits,
    ) -> Option<String> {
        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_compute_pass(None);
        assert_eq!(error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        if let Some(group) = group {
            assert_eq!(
                pass.set_bind_group(0, Some(group), Vec::new(), limits),
                None
            );
        }
        if indirect {
            assert_eq!(
                pass.dispatch_workgroups_indirect(noop_indirect_buffer(device), 0, limits),
                None
            );
        } else {
            assert_eq!(
                pass.dispatch_workgroups(counts.0, counts.1, counts.2, limits),
                None
            );
        }
        assert_eq!(pass.end(), None);
        let (commands, error) = encoder.finish();
        assert_eq!(commands.is_error(), error.is_some());
        error
    }

    /// Block 119 R2: with `minBindingSize` 0, the shader-required size is checked
    /// at dispatch for each buffer type and surfaces at encoder finish.
    #[test]
    fn compute_dispatch_validates_shader_required_buffer_size() {
        let device = noop_device();
        let too_small = "buffer bound with size 28 at group 0 binding 0 is too small: the pipeline requires at least 32 bytes";
        for ty in [
            BufferBindingType::Uniform,
            BufferBindingType::Storage,
            BufferBindingType::ReadOnlyStorage,
        ] {
            let dispatch = |buffer_size, indirect| {
                let (pipeline, group) =
                    required_size_group(&device, 0, ty, buffer_size, 0, u64::MAX);
                dispatch_error(
                    &device,
                    pipeline,
                    Some(group),
                    indirect,
                    (1, 1, 1),
                    device.limits(),
                )
            };
            assert_eq!(dispatch(28, false).as_deref(), Some(too_small));
            assert_eq!(dispatch(32, false), None);
            assert_eq!(dispatch(36, false), None);
            // Indirect dispatch shares the dispatch-state validation.
            assert_eq!(dispatch(28, true).as_deref(), Some(too_small));
            assert_eq!(dispatch(32, true), None);
        }
    }

    /// Block 119 R2: the checked size is the bound range, not the buffer's size.
    #[test]
    fn compute_dispatch_required_buffer_size_measures_the_bound_range() {
        let device = noop_device();
        let dispatch = |buffer_size, offset, size| {
            let (pipeline, group) = required_size_group(
                &device,
                0,
                BufferBindingType::Storage,
                buffer_size,
                offset,
                size,
            );
            dispatch_error(
                &device,
                pipeline,
                Some(group),
                false,
                (1, 1, 1),
                device.limits(),
            )
        };

        assert!(dispatch(1024, 0, 28).is_some());
        assert_eq!(dispatch(1024, 0, 32), None);
        // Offset 256 leaves 28 and 32 bytes of the whole-size binding.
        assert!(dispatch(284, 256, u64::MAX).is_some());
        assert_eq!(dispatch(288, 256, u64::MAX), None);
    }

    /// Block 119 R2: a nonzero layout `minBindingSize` is checked at pipeline and
    /// bind group creation instead, so dispatch adds nothing for it.
    #[test]
    fn compute_dispatch_skips_required_size_for_nonzero_min_binding_size() {
        let device = noop_device();
        let (pipeline, group) =
            required_size_group(&device, 32, BufferBindingType::Storage, 32, 0, u64::MAX);

        assert!(pipeline.required_buffer_sizes().is_empty());
        assert_eq!(
            dispatch_error(
                &device,
                pipeline,
                Some(group),
                false,
                (1, 1, 1),
                device.limits()
            ),
            None
        );
    }

    fn linear_index_pipeline(device: &Device, builtin: Option<&str>) -> Arc<ComputePipeline> {
        let parameter =
            builtin.map_or_else(String::new, |name| format!("@builtin({name}) index: u32"));
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(format!(
                "requires linear_indexing;\n\
             @compute @workgroup_size(3, 5, 17) fn cs({parameter}) {{}}"
            ))),
        );
        let pipeline = Arc::new(device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Auto,
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        }));
        assert!(!pipeline.is_error());
        pipeline
    }

    fn linear_index_error(
        device: &Device,
        pipeline: &Arc<ComputePipeline>,
        counts: (u32, u32, u32),
    ) -> Option<String> {
        let limits = Limits {
            max_compute_workgroups_per_dimension: u32::MAX,
            ..device.limits()
        };
        dispatch_error(device, pipeline.clone(), None, false, counts, limits)
    }

    /// Block 119 R4: 255 invocations per workgroup; 255 * 257 * 65537 is exactly
    /// `u32::MAX`, and the workgroup size counts for this builtin.
    #[test]
    fn compute_dispatch_global_invocation_index_range() {
        if !yawgpu_tint::HAVE_TINT {
            return;
        }
        let device = noop_device();
        let pipeline = linear_index_pipeline(&device, Some("global_invocation_index"));

        assert_eq!(pipeline.linear_index_invocations(), Some(255));
        assert_eq!(
            linear_index_error(&device, &pipeline, (65537, 257, 1)),
            None
        );
        assert_eq!(
            linear_index_error(&device, &pipeline, (257, 65537, 1)),
            None
        );
        assert_eq!(
            linear_index_error(&device, &pipeline, (1, 257, 65537)),
            None
        );
        // One workgroup over: 255 * 16843010 = u32::MAX + 255.
        assert_eq!(
            linear_index_error(&device, &pipeline, (16843010, 1, 1)).as_deref(),
            Some(
                "compute dispatch (16843010, 1, 1) with 255 indexed invocations per workgroup exceeds the unsigned 32-bit linear index range"
            )
        );
        assert!(linear_index_error(&device, &pipeline, (1, 16843010, 1)).is_some());
        assert!(linear_index_error(&device, &pipeline, (1, 1, 16843010)).is_some());
    }

    /// Block 119 R4: `workgroup_index` numbers workgroups, so the workgroup size
    /// does not count: a dispatch that would overflow with it is accepted.
    #[test]
    fn compute_dispatch_workgroup_index_range() {
        if !yawgpu_tint::HAVE_TINT {
            return;
        }
        let device = noop_device();
        let pipeline = linear_index_pipeline(&device, Some("workgroup_index"));

        assert_eq!(pipeline.linear_index_invocations(), Some(1));
        assert_eq!(
            linear_index_error(&device, &pipeline, (65537, 65535, 1)),
            None
        );
        assert_eq!(
            linear_index_error(&device, &pipeline, (u32::MAX, 1, 1)),
            None
        );
        assert!(linear_index_error(&device, &pipeline, (65536, 65536, 1)).is_some());
        assert!(linear_index_error(&device, &pipeline, (1, 65536, 65536)).is_some());
    }

    /// Block 119 R4: a pipeline using neither builtin has no range to enforce.
    #[test]
    fn compute_dispatch_without_linear_indexing_ignores_the_range() {
        let device = noop_device();
        let pipeline = linear_index_pipeline(&device, None);

        assert_eq!(pipeline.linear_index_invocations(), None);
        for counts in [(65537, 257, 1), (16843010, 1, 1), (65536, 65536, 1)] {
            assert_eq!(linear_index_error(&device, &pipeline, counts), None);
        }
    }

    /// Block 119 R4: mirrors Dawn's partial-product checks, so no intermediate
    /// wraps at the extremes, and a zero count does not excuse the other factor.
    #[test]
    fn linear_index_range_checks_partial_products() {
        let error =
            |invocations, x, y, z| validate_linear_index_range(invocations, x, y, z).is_err();

        assert!(!error(1, u32::MAX, 1, 1));
        assert!(!error(1, 1, u32::MAX, 1));
        assert!(error(2, u32::MAX, 1, 1));
        assert!(error(1, u32::MAX, u32::MAX, u32::MAX));
        // A wrapped 2^32 * 2^32 product would read as 0.
        assert!(error(1, 65536, 65536, 65536));
        assert!(error(u64::MAX, 2, 1, 1));
        assert!(error(u64::MAX, u32::MAX, u32::MAX, u32::MAX));
        // Partial products are checked even though these totals are 0.
        assert!(error(1, 0, 65536, 65536));
        assert!(error(65536, 65536, 0, 1));
        assert!(!error(1, 0, 65535, 65535));
    }

    fn indirect_client_buffer(device: &Device, size: u64) -> Arc<Buffer> {
        Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::INDIRECT,
            size,
            mapped_at_creation: false,
        }))
    }

    /// Records one indirect dispatch per offset and returns the finished buffer.
    fn indirect_commands(
        device: &Device,
        pipeline: Arc<ComputePipeline>,
        client: &Arc<Buffer>,
        offsets: &[u64],
    ) -> crate::command_encoder::CommandBuffer {
        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_compute_pass(None);
        assert_eq!(error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        for &offset in offsets {
            assert_eq!(
                pass.dispatch_workgroups_indirect(client.clone(), offset, device.limits()),
                None
            );
        }
        assert_eq!(pass.end(), None);
        let (commands, error) = encoder.finish();
        assert_eq!(error, None);
        commands
    }

    fn validation_immediates(command: &ComputePassCommand) -> [u32; 4] {
        assert_eq!(command.immediate_data.len(), MAX_IMMEDIATE_DATA_BYTES);
        let words = command.immediate_data.as_chunks::<4>().0;
        std::array::from_fn(|index| u32::from_le_bytes(words[index]))
    }

    fn validation_binding(command: &ComputePassCommand, binding: u32) -> (Arc<Buffer>, u64, u64) {
        let group = &command.bind_groups[&0].group;
        let entry = group
            .entries()
            .iter()
            .find(|entry| entry.binding == binding)
            .unwrap();
        let BindGroupResource::Buffer {
            buffer,
            offset,
            size,
            ..
        } = &entry.resource
        else {
            panic!("expected a buffer binding");
        };
        (buffer.clone(), *offset, *size)
    }

    fn compute_commands(commands: &[CommandExecution]) -> Vec<&ComputePassCommand> {
        commands
            .iter()
            .map(|command| {
                let CommandExecution::ComputePass(pass) = command else {
                    panic!("expected a compute pass command");
                };
                pass
            })
            .collect()
    }

    /// Block 119 R5/R5a: a validation command runs first, then the user's dispatch
    /// reads the validated-arguments buffer at offset 0. The linear-indexing
    /// immediates follow the builtin the user pipeline uses.
    #[test]
    fn indirect_dispatch_records_validation_then_indirect_on_internal_buffer() {
        if !yawgpu_tint::HAVE_TINT {
            return;
        }
        let device = noop_device();
        let max = device.limits().max_compute_workgroups_per_dimension;
        // The pipelines use workgroup_size(3, 5, 17): 255 invocations.
        for (builtin, linear_indexing, overflow_value) in [
            (None, 0, u32::MAX),
            (Some("global_invocation_index"), 1, u32::MAX / 255),
            (Some("workgroup_index"), 1, u32::MAX),
        ] {
            let pipeline = linear_index_pipeline(&device, builtin);
            let client = indirect_client_buffer(&device, 12);
            let buffer = indirect_commands(&device, pipeline.clone(), &client, &[0]);
            let validated = device.dispatch_indirect_buffer().unwrap();

            let [validation, dispatch] = compute_commands(buffer.command_ops())[..] else {
                panic!("expected a validation command and a dispatch");
            };
            assert!(Arc::ptr_eq(
                &validation.pipeline,
                &device.dispatch_indirect_pipeline().unwrap()
            ));
            assert!(matches!(
                validation.dispatch,
                ComputeDispatch::Direct {
                    workgroups: (1, 1, 1)
                }
            ));
            assert_eq!(
                validation_immediates(validation),
                [max, 0, linear_indexing, overflow_value],
                "{builtin:?}"
            );
            let (source, offset, size) = validation_binding(validation, 0);
            assert!(Arc::ptr_eq(&source, &client));
            assert_eq!((offset, size), (0, 12));
            let (target, offset, size) = validation_binding(validation, 1);
            assert!(Arc::ptr_eq(&target, &validated));
            assert_eq!((offset, size), (0, 12));

            assert!(Arc::ptr_eq(&dispatch.pipeline, &pipeline));
            let ComputeDispatch::Indirect { buffer, offset } = &dispatch.dispatch else {
                panic!("expected an indirect dispatch");
            };
            assert!(Arc::ptr_eq(buffer, &validated));
            assert_eq!(*offset, 0);
        }
    }

    /// Block 119 R5a: the binding offset is aligned down to
    /// `minStorageBufferOffsetAlignment` (256 on Noop) and the rest becomes a word
    /// offset, with the binding grown to still reach the arguments.
    #[test]
    fn indirect_dispatch_aligns_client_binding_offset() {
        let device = noop_device();
        let client = indirect_client_buffer(&device, 1024);

        for (client_offset, binding_offset, binding_size, words) in [
            (0, 0, 12, 0),
            (4, 0, 16, 1),
            (20, 0, 32, 5),
            (252, 0, 264, 63),
            (256, 256, 12, 0),
            (276, 256, 32, 5),
        ] {
            let buffer = indirect_commands(
                &device,
                noop_compute_pipeline(&device),
                &client,
                &[client_offset],
            );
            let commands = compute_commands(buffer.command_ops());
            let (_, offset, size) = validation_binding(commands[0], 0);
            assert_eq!((offset, size), (binding_offset, binding_size));
            assert_eq!(validation_immediates(commands[0])[1], words);
            // The validated buffer is always read from its start.
            assert!(matches!(
                commands[1].dispatch,
                ComputeDispatch::Indirect { offset: 0, .. }
            ));
        }
    }

    /// Block 119 R5c: every dispatch in a pass gets its own validation command, so
    /// none of them reads an earlier dispatch's arguments.
    #[test]
    fn repeated_indirect_dispatches_each_revalidate() {
        let device = noop_device();
        let client = indirect_client_buffer(&device, 1024);
        let pipeline = noop_compute_pipeline(&device);

        let buffer = indirect_commands(&device, pipeline, &client, &[0, 12, 24]);
        let commands = compute_commands(buffer.command_ops());

        assert_eq!(commands.len(), 6);
        for (pair, word) in commands.as_chunks::<2>().0.iter().zip([0, 3, 6]) {
            assert!(matches!(pair[0].dispatch, ComputeDispatch::Direct { .. }));
            assert_eq!(validation_immediates(pair[0])[1], word);
            assert!(matches!(pair[1].dispatch, ComputeDispatch::Indirect { .. }));
        }
    }

    /// Block 119 R5b: the validation command leaves the pass's state alone, so the
    /// user's dispatch keeps its own immediates and the next dispatch still works.
    #[test]
    fn indirect_dispatch_leaves_user_state_untouched() {
        let device = noop_device();
        let pipeline = noop_compute_pipeline(&device);
        let bind_group = empty_bind_group(&device);
        let client = indirect_client_buffer(&device, 12);
        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_compute_pass(None);
        assert_eq!(error, None);
        assert_eq!(pass.set_pipeline(pipeline.clone()), None);
        assert_eq!(
            pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.set_immediates(0, &[7, 0, 0, 0], device.limits()), None);
        assert_eq!(
            pass.dispatch_workgroups_indirect(client, 0, device.limits()),
            None
        );
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(pass.end(), None);
        let (buffer, error) = encoder.finish();
        assert_eq!(error, None);

        let commands = compute_commands(buffer.command_ops());
        assert_eq!(commands.len(), 3);
        for command in &commands[1..] {
            assert!(Arc::ptr_eq(&command.pipeline, &pipeline));
            assert_eq!(command.immediate_data[0], 7);
        }
    }

    /// Block 119 R5a: indirect dispatch works on a device reporting
    /// `maxImmediateSize` 0, as GLES does, and still delivers its immediates.
    #[test]
    fn indirect_dispatch_records_validation_with_zero_immediate_limit() {
        let device = noop_device_with_limits(Limits {
            max_immediate_size: 0,
            ..Limits::DEFAULT
        });
        let client = indirect_client_buffer(&device, 12);
        let buffer = indirect_commands(&device, noop_compute_pipeline(&device), &client, &[0]);

        let commands = compute_commands(buffer.command_ops());
        assert_eq!(commands.len(), 2);
        assert_eq!(
            validation_immediates(commands[0])[0],
            Limits::DEFAULT.max_compute_workgroups_per_dimension
        );
    }

    /// Block 119 R5a: failing to create either internal object invalidates the
    /// encoder and records nothing, as the timestamp path does.
    #[test]
    fn indirect_dispatch_internal_creation_failure_invalidates_encoder() {
        let dispatch = |device: &Device| {
            let client = indirect_client_buffer(device, 12);
            let encoder = device.create_command_encoder();
            let (pass, error) = encoder.begin_compute_pass(None);
            assert_eq!(error, None);
            assert_eq!(pass.set_pipeline(noop_compute_pipeline(device)), None);
            assert_eq!(
                pass.dispatch_workgroups_indirect(client, 0, device.limits()),
                None
            );
            assert_eq!(pass.end(), None);
            encoder.finish()
        };

        let device = noop_device();
        device
            .inner
            .dispatch_indirect_pipeline
            .set(Err("injected pipeline failure".to_owned()))
            .unwrap();
        let (buffer, error) = dispatch(&device);
        assert_eq!(error.as_deref(), Some("injected pipeline failure"));
        assert!(buffer.is_error());
        assert!(buffer.command_ops().is_empty());

        let device = noop_device();
        *device.inner.dispatch_indirect_buffer_failure.lock() =
            Some("injected buffer failure".to_owned());
        let (buffer, error) = dispatch(&device);
        assert_eq!(error.as_deref(), Some("injected buffer failure"));
        assert!(buffer.is_error());
        assert!(buffer.command_ops().is_empty());

        // Block 119 R5a: a failed buffer creation is not cached, so a later
        // dispatch on the same device retries and succeeds.
        assert!(device.inner.dispatch_indirect_buffer.get().is_none());
        let (buffer, error) = dispatch(&device);
        assert_eq!(error, None);
        assert_eq!(buffer.command_ops().len(), 2);
        assert!(device.inner.dispatch_indirect_buffer.get().is_some());
    }

    /// Block 119 R5b: the usage scope still sees the client buffer, and the check
    /// runs before any internal object is created.
    #[test]
    fn indirect_dispatch_usage_conflict_on_client_buffer_still_errors() {
        let device = noop_device();
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
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
            bind_group_layouts: vec![layout.clone()],
            immediate_size: 0,
            error: None,
        }));
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> data: array<u32>;
             @compute @workgroup_size(1) fn cs() { data[0] = 1u; }"
                    .to_owned(),
            )),
        );
        let pipeline = Arc::new(device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(pipeline_layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        }));
        let client = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::STORAGE | BufferUsage::INDIRECT,
            size: 64,
            mapped_at_creation: false,
        }));
        let group = Arc::new(device.create_bind_group(
            layout,
            vec![BindGroupEntry {
                binding: 0,
                resource: BindGroupResource::Buffer {
                    buffer: client.clone(),
                    device: Arc::new(device.clone()),
                    offset: 0,
                    size: 64,
                },
            }],
        ));
        let encoder = device.create_command_encoder();
        let (pass, error) = encoder.begin_compute_pass(None);
        assert_eq!(error, None);
        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_bind_group(0, Some(group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(
            pass.dispatch_workgroups_indirect(client, 0, device.limits()),
            None
        );
        assert_eq!(pass.end(), None);

        let (buffer, error) = encoder.finish();
        assert_eq!(
            error.as_deref(),
            Some("usage scope cannot read and write or write the same buffer range twice")
        );
        assert!(buffer.is_error());
        assert!(device.inner.dispatch_indirect_pipeline.get().is_none());
    }

    /// Block 119 R5b: both the client and the validated buffer stay referenced, so
    /// each stays alive and a destroyed client buffer is still caught at submit.
    #[test]
    fn indirect_dispatch_references_client_and_internal_buffers() {
        let device = noop_device();
        let client = indirect_client_buffer(&device, 12);
        let buffer = indirect_commands(&device, noop_compute_pipeline(&device), &client, &[0]);
        let validated = device.dispatch_indirect_buffer().unwrap();

        let referenced = &buffer.inner.referenced_buffers;
        assert!(referenced.iter().any(|buffer| Arc::ptr_eq(buffer, &client)));
        assert!(referenced
            .iter()
            .any(|buffer| Arc::ptr_eq(buffer, &validated)));

        client.destroy();
        assert!(device.queue().submit(&[Arc::new(buffer)]).is_some());
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
            let (pass, _) = encoder.begin_compute_pass(None);
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
            let (pass, _) = encoder.begin_compute_pass(None);
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
            let (pass, _) = encoder.begin_compute_pass(None);
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
            let (pass, _) = encoder.begin_compute_pass(None);
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
            let (pass, begin_error) = encoder.begin_compute_pass(None);
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
        let (pass, _) = encoder.begin_compute_pass(None);
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
        let (pass, _) = encoder.begin_compute_pass(None);
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
    fn compute_pass_indirect_validates_dispatch_state_before_indirect_buffer() {
        let device = noop_device();
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 16,
            mapped_at_creation: false,
        }));
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass(None);
        assert_eq!(begin_error, None);

        assert_eq!(
            pass.dispatch_workgroups_indirect(buffer, 0, device.limits()),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("compute dispatch requires a compute pipeline".to_owned())
        );
    }

    #[test]
    fn compute_pass_indirect_usage_conflict_stays_after_required_immediates() {
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
            immediate_size: 16,
            error: None,
        }));
        let module = Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r#"
requires immediate_address_space;
@group(0) @binding(0) var<storage, read_write> values: array<u32>;
var<immediate> params : vec4u;

@compute @workgroup_size(1)
fn cs() {
    values[0] = params.x;
}
"#
                .to_owned(),
            )),
        );
        let pipeline = Arc::new(device.create_compute_pipeline(ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Explicit(pipeline_layout),
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
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
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
            Some("Required immediate data at offset 0 was not set.".to_owned())
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
        let pipeline_a = storage_texture_compute_pipeline(
            &device,
            Arc::clone(&pipeline_layout),
            StorageTextureAccess::WriteOnly,
        );
        let pipeline_b = storage_texture_compute_pipeline(
            &device,
            pipeline_layout,
            StorageTextureAccess::WriteOnly,
        );

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
    fn compute_pass_storage_texture_aliasing_tracks_mips() {
        for second_mip in [0, 1] {
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
            let pipeline_layout =
                Arc::new(device.create_pipeline_layout(PipelineLayoutDescriptor {
                    bind_group_layouts: vec![Arc::clone(&bind_group_layout)],
                    immediate_size: 0,
                    error: None,
                }));
            let texture = device.create_texture(TextureDescriptor {
                usage: TextureUsage::STORAGE_BINDING,
                dimension: TextureDimension::D2,
                size: Extent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
                format: rgba8_unorm(),
                mip_level_count: 2,
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
            let (other_view, error) = texture.create_view(TextureViewDescriptor {
                base_mip_level: second_mip,
                mip_level_count: Some(1),
                format: None,
                dimension: Some(TextureViewDimension::D2),
                base_array_layer: 0,
                array_layer_count: Some(1),
                aspect: None,
                usage: None,
                swizzle: None,
            });
            assert_eq!(error, None);
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
                            texture_view: Arc::new(other_view),
                            device: Arc::new(device.clone()),
                        },
                    },
                ],
            ));
            let pipeline = aliasing_storage_textures_compute_pipeline(&device, pipeline_layout);

            let encoder = device.create_command_encoder();
            let (pass, begin_error) = encoder.begin_compute_pass(None);
            assert_eq!(begin_error, None);

            assert_eq!(pass.set_pipeline(pipeline), None);
            assert_eq!(
                pass.set_bind_group(0, Some(bind_group), Vec::new(), device.limits()),
                None
            );
            assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
            assert_eq!(pass.end(), None);

            let (command_buffer, error) = encoder.finish();
            if second_mip == 0 {
                assert!(command_buffer.is_error());
                let error = error.expect("overlapping storage bindings must fail");
                assert!(error.contains("writable storage texture binding aliasing"));
                assert!(error.contains("group 0 binding 0"));
                assert!(error.contains("group 0 binding 1"));
            } else {
                assert_eq!(error, None);
                assert!(!command_buffer.is_error());
            }
        }
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
    fn compute_pass_encoder_error_bind_group_includes_creation_reason() {
        let device = noop_device();
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
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
        let group = Arc::new(device.create_bind_group(layout, Vec::new()));
        assert!(group.is_error());
        assert_eq!(
            group.error_message(),
            Some("bind group entry count must match bind group layout")
        );

        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass(None);
        assert_eq!(begin_error, None);
        assert_eq!(
            pass.set_bind_group(0, Some(group), Vec::new(), device.limits()),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some(
                "cannot set an error bind group: bind group entry count must match bind group layout"
                    .to_owned()
            )
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
        let (pass, begin_error) = encoder.begin_compute_pass(None);
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
