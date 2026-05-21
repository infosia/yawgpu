use std::cell::UnsafeCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{
    HalAdapter, HalAddressMode, HalBackend, HalBoundBuffer, HalBuffer, HalBufferBindingKind,
    HalBufferCopy, HalBufferTextureCopy, HalBufferTextureLayout, HalCompareFunction,
    HalComputePass, HalComputePipeline, HalCopy, HalDescriptorBinding, HalDevice, HalDraw,
    HalError, HalExtent3d, HalFilterMode, HalInstance, HalMipmapFilterMode, HalOrigin3d,
    HalPrimitiveTopology, HalQueue, HalRenderColorTarget, HalRenderLoadOp, HalRenderPass,
    HalRenderPipeline, HalRenderPipelineDescriptor, HalSampler, HalSamplerDescriptor,
    HalShaderSource, HalSurface, HalTexture, HalTextureCopy, HalTextureDescriptor,
    HalTextureFormat, HalTextureUsage, HalVertexAttribute, HalVertexBufferLayout, HalVertexFormat,
    HalVertexStepMode,
};

use crate::adapter::*;
use crate::bind_group::*;
use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::compute_pipeline::*;
use crate::copy::*;
use crate::device::*;
use crate::error::*;
use crate::extent::*;
use crate::format::*;
use crate::future::*;
use crate::instance::*;
use crate::limits::*;
use crate::pass::*;
use crate::pipeline_layout::*;
use crate::query_set::*;
use crate::queue::*;
use crate::render_bundle::*;
use crate::render_pass::*;
use crate::render_pipeline::*;
use crate::sampler::*;
use crate::shader::*;
use crate::shader_naga;
use crate::texture::*;
use crate::texture_view::*;

#[derive(Debug, Clone)]
pub struct ComputePassEncoder {
    pub(crate) inner: Arc<PassEncoderInner>,
}

impl ComputePassEncoder {
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

    pub fn set_pipeline(&self, pipeline: Arc<ComputePipeline>) -> Option<String> {
        self.inner.record_pass_command(|state| {
            state.compute_pipeline = Some(pipeline);
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

    pub fn dispatch_workgroups(&self, x: u32, y: u32, z: u32, limits: Limits) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_compute_dispatch_state(state, limits)?;
            if x > limits.max_compute_workgroups_per_dimension
                || y > limits.max_compute_workgroups_per_dimension
                || z > limits.max_compute_workgroups_per_dimension
            {
                return Err("compute dispatch workgroup count exceeds the device limit".to_owned());
            }
            let pipeline = state
                .compute_pipeline
                .as_ref()
                .ok_or_else(|| "compute dispatch requires a compute pipeline".to_owned())?;
            self.inner.parent.record_compute_pass(ComputePassCommand {
                pipeline: Arc::clone(pipeline),
                bind_groups: state.bind_groups.clone(),
                workgroups: (x, y, z),
            });
            Ok(())
        })
    }

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
    use crate::test_helpers::*;
    use crate::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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
    fn compute_pass_encoder_pipeline_bind_group_and_dispatch() {
        let device = noop_device();
        let pipeline = noop_compute_pipeline(&device);
        let bind_group = empty_bind_group(&device);
        let indirect = noop_indirect_buffer(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.set_bind_group(0, Some(bind_group), Vec::new()), None);
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
}
