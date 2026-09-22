use super::*;
use crate::format::{format_has_depth_aspect, format_has_stencil_aspect};
#[cfg(feature = "tiled")]
use crate::{
    HalDescriptorBindingKind, HalSubpassAttachmentLayout, HalSubpassAttachmentResource,
    HalSubpassDependencyType, HalSubpassDepthStencilAttachment, HalSubpassDraw,
    HalSubpassPassLayout, HalSubpassRenderPassCommand,
};
use crate::{
    HalRenderColorTarget, HalRenderDepthStencilAttachment, HalTextureAspect, HalTextureClear,
    HalTextureDimension,
};
use std::collections::HashSet;

/// Records submit into the command stream.
pub(super) fn submit_copies(
    queue: &VulkanQueueInner,
    copies: &[HalCopy],
) -> Result<SubmissionIndex, HalError> {
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(queue.device.queue_family_index);
    let command_pool = unsafe {
        queue
            .device
            .device
            .create_command_pool(&command_pool_info, None)
    }
    .map_err(|error| queue_submission_error("vkCreateCommandPool", error))?;
    record_and_submit_copies(queue, command_pool, copies)
}

/// Returns record and submit copies.
pub(super) fn record_and_submit_copies(
    queue: &VulkanQueueInner,
    command_pool: vk::CommandPool,
    copies: &[HalCopy],
) -> Result<SubmissionIndex, HalError> {
    let mut temporary_resources = Vec::new();
    let mut descriptor_pools = Vec::new();
    let mut framebuffers = Vec::new();
    let mut image_views = Vec::new();
    let mut render_passes = Vec::new();
    let mut fence = None;
    let mut command_pool_cleanup = Some(command_pool);
    let surface_pending = find_surface_pending(copies);
    let result = (|| {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers =
            unsafe { queue.device.device.allocate_command_buffers(&allocate_info) }
                .map_err(|error| queue_submission_error("vkAllocateCommandBuffers", error))?;
        let Some(&command_buffer) = command_buffers.first() else {
            return Err(HalError::QueueSubmissionFailed {
                backend: BACKEND,
                message: "command buffer allocation returned no buffers".to_string(),
            });
        };
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            queue
                .device
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|error| queue_submission_error("vkBeginCommandBuffer", error))?;
        }
        for copy in copies {
            match copy {
                HalCopy::Buffer(copy) => {
                    encode_buffer_copy(&queue.device.device, command_buffer, copy)?;
                }
                HalCopy::BufferClear(clear) => {
                    encode_buffer_clear(&queue.device.device, command_buffer, clear)?;
                }
                HalCopy::ClearTexture(clear) => {
                    encode_texture_clear(
                        &queue.device.device,
                        command_buffer,
                        clear,
                        &mut temporary_resources,
                    )?;
                }
                HalCopy::WriteTimestamp(write) => {
                    encode_write_timestamp(&queue.device.device, command_buffer, write)?;
                }
                HalCopy::ResolveQuerySet(resolve) => {
                    encode_resolve_query_set(&queue.device.device, command_buffer, resolve)?;
                }
                HalCopy::BufferToTexture(copy) => {
                    encode_buffer_to_texture(&queue.device.device, command_buffer, copy)?;
                }
                HalCopy::TextureToBuffer(copy) => {
                    encode_texture_to_buffer(&queue.device.device, command_buffer, copy)?;
                }
                HalCopy::TextureToTexture(copy) => {
                    encode_texture_to_texture(
                        &queue.device.device,
                        command_buffer,
                        copy,
                        &mut temporary_resources,
                    )?;
                }
                HalCopy::ComputePass(pass) => {
                    let temps = encode_compute_pass(&queue.device.device, command_buffer, pass)?;
                    if let Some(pool) = temps.descriptor_pool {
                        descriptor_pools.push(pool);
                    }
                    image_views.extend(temps.image_views);
                }
                HalCopy::RenderPassCommandStream(pass) => {
                    let temps =
                        encode_render_pass_command_stream(&queue.device, command_buffer, pass)?;
                    descriptor_pools.extend(temps.descriptor_pools);
                    framebuffers.push(temps.framebuffer);
                    image_views.extend(temps.image_views);
                    if let Some(render_pass) = temps.render_pass {
                        render_passes.push(render_pass);
                    }
                }
                #[cfg(feature = "tiled")]
                HalCopy::SubpassRenderPass(pass) => {
                    let temps = encode_subpass_render_pass(&queue.device, command_buffer, pass)?;
                    descriptor_pools.extend(temps.descriptor_pools);
                    framebuffers.push(temps.framebuffer);
                    image_views.extend(temps.image_views);
                    if let Some(render_pass) = temps.render_pass {
                        render_passes.push(render_pass);
                    }
                }
            }
        }
        unsafe {
            queue
                .device
                .device
                .end_command_buffer(command_buffer)
                .map_err(|error| queue_submission_error("vkEndCommandBuffer", error))?;
            let command_buffers = [command_buffer];
            let mut wait_semaphores = Vec::new();
            let mut wait_stages = Vec::new();
            let mut signal_semaphores = Vec::new();
            let mut surface_retire = None;
            if let Some(pending_state) = surface_pending.as_ref() {
                let mut state = pending_state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(pending) = state.pending_acquire.as_mut() {
                    if !pending.consumed {
                        wait_semaphores.push(pending.acquired_sem);
                        wait_stages.push(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
                        signal_semaphores.push(pending.render_finished_sem);
                        pending.consumed = true;
                        surface_retire = Some(Arc::clone(pending_state));
                    }
                }
            }
            let fence_info = vk::FenceCreateInfo::default();
            let created_fence = queue
                .device
                .device
                .create_fence(&fence_info, None)
                .map_err(|error| queue_submission_error("vkCreateFence", error))?;
            fence = Some(created_fence);
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            let (submission_index, tracked_submission) = {
                // Vulkan queues require externally synchronized submission.
                // Queue access serializes all host queue operations, while
                // holding the timeline lock through vkQueueSubmit makes index
                // order match Vulkan queue order.
                let _queue_access = queue
                    .queue_access
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let mut submissions = queue
                    .submissions
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let submission_index = submissions.reserve()?;
                queue
                    .device
                    .device
                    .queue_submit(queue.queue, &[submit_info], created_fence)
                    .map_err(|error| queue_submission_error("vkQueueSubmit", error))?;
                submissions.register_fence(submission_index, created_fence);
                (
                    submission_index,
                    TrackedSubmission {
                        index: submission_index,
                        tracker: Arc::clone(&queue.submissions),
                    },
                )
            };
            fence = None;
            let retire_fence = created_fence;
            let mut retained = collect_retained_resources(copies);
            retained.append(&mut temporary_resources);
            let cleanup = retire_ops(
                command_pool,
                std::mem::take(&mut descriptor_pools),
                std::mem::take(&mut framebuffers),
                std::mem::take(&mut image_views),
                std::mem::take(&mut render_passes),
            );
            command_pool_cleanup = None;
            if let Some(pending_state) = surface_retire {
                let mut pending_state = pending_state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                pending_state.retire.retire_tracked(
                    &queue.device.device,
                    retire_fence,
                    cleanup,
                    retained,
                    true,
                    Some(tracked_submission),
                )?;
            } else {
                let mut retire = queue
                    .retire
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                retire.retire_tracked(
                    &queue.device.device,
                    retire_fence,
                    cleanup,
                    retained,
                    true,
                    Some(tracked_submission),
                )?;
            }
            Ok(submission_index)
        }
    })();
    if result.is_err() {
        unsafe {
            if let Some(fence) = fence.take() {
                queue.device.device.destroy_fence(fence, None);
            }
            if let Some(command_pool) = command_pool_cleanup.take() {
                cleanup_retire_ops(
                    &queue.device.device,
                    retire_ops(
                        command_pool,
                        descriptor_pools,
                        framebuffers,
                        image_views,
                        render_passes,
                    ),
                );
            }
        }
    }
    result
}

/// Returns transition swapchain image to present.
pub(super) fn transition_swapchain_image_to_present(
    queue: &VulkanQueue,
    texture: &VulkanTexture,
    pending_state: Arc<Mutex<SurfacePendingState>>,
    wait_semaphore: vk::Semaphore,
    signal_semaphore: vk::Semaphore,
    fence: vk::Fence,
) -> Result<(), HalError> {
    let inner = texture.inner()?;
    let command_pool = {
        let mut state = pending_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(command_pool) = state.transition_command_pool {
            command_pool
        } else {
            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue.inner.device.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let command_pool = unsafe {
                queue
                    .inner
                    .device
                    .device
                    .create_command_pool(&command_pool_info, None)
            }
            .map_err(|_| HalError::PresentFailed {
                backend: BACKEND,
                message: "command pool creation failed",
            })?;
            state.transition_command_pool = Some(command_pool);
            command_pool
        }
    };
    let result = (|| {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers = unsafe {
            queue
                .inner
                .device
                .device
                .allocate_command_buffers(&allocate_info)
        }
        .map_err(|_| HalError::PresentFailed {
            backend: BACKEND,
            message: "command buffer allocation failed",
        })?;
        let Some(&command_buffer) = command_buffers.first() else {
            return Err(HalError::PresentFailed {
                backend: BACKEND,
                message: "command buffer allocation failed",
            });
        };
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            queue
                .inner
                .device
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|_| HalError::PresentFailed {
                    backend: BACKEND,
                    message: "command buffer begin failed",
                })?;
        }
        transition_image(
            &queue.inner.device.device,
            command_buffer,
            inner,
            vk::ImageLayout::PRESENT_SRC_KHR,
            IMAGE_LAYOUT_PRESENT,
        )?;
        unsafe {
            queue
                .inner
                .device
                .device
                .end_command_buffer(command_buffer)
                .map_err(|_| HalError::PresentFailed {
                    backend: BACKEND,
                    message: "command buffer end failed",
                })?;
            let command_buffers = [command_buffer];
            let wait_semaphores = [wait_semaphore];
            let wait_stages = [vk::PipelineStageFlags::BOTTOM_OF_PIPE];
            let signal_semaphores = [signal_semaphore];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            {
                let _queue_access = queue
                    .inner
                    .queue_access
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                queue
                    .inner
                    .device
                    .device
                    .queue_submit(queue.inner.queue, &[submit_info], fence)
                    .map_err(|_| HalError::PresentFailed {
                        backend: BACKEND,
                        message: "queue submit failed",
                    })?;
            }
            pending_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .retire
                .retire(
                    &queue.inner.device.device,
                    fence,
                    vec![RetireOp::CommandBuffer {
                        pool: command_pool,
                        buffer: command_buffer,
                    }],
                    Vec::new(),
                    false,
                )
                .map_err(|_| HalError::PresentFailed {
                    backend: BACKEND,
                    message: "transition retire registration failed",
                })?;
        }
        Ok(())
    })();
    result
}

fn retire_ops(
    command_pool: vk::CommandPool,
    descriptor_pools: Vec<vk::DescriptorPool>,
    framebuffers: Vec<vk::Framebuffer>,
    image_views: Vec<vk::ImageView>,
    render_passes: Vec<vk::RenderPass>,
) -> Vec<RetireOp> {
    let mut cleanup = Vec::new();
    cleanup.push(RetireOp::CommandPool(command_pool));
    cleanup.extend(descriptor_pools.into_iter().map(RetireOp::DescriptorPool));
    cleanup.extend(framebuffers.into_iter().map(RetireOp::Framebuffer));
    cleanup.extend(image_views.into_iter().map(RetireOp::ImageView));
    cleanup.extend(render_passes.into_iter().map(RetireOp::RenderPass));
    cleanup
}

fn collect_retained_resources(copies: &[HalCopy]) -> Vec<RetainedResource> {
    let mut retained = RetainedResources::default();
    for copy in copies {
        retain_copy_resources(copy, &mut retained);
    }
    retained.into_vec()
}

#[derive(Default)]
struct RetainedResources {
    resources: Vec<RetainedResource>,
    seen: HashSet<(RetainedResourceKind, usize)>,
}

impl RetainedResources {
    fn retain(&mut self, kind: RetainedResourceKind, ptr: usize, resource: RetainedResource) {
        if self.seen.insert((kind, ptr)) {
            self.resources.push(resource);
        }
    }

    fn into_vec(self) -> Vec<RetainedResource> {
        self.resources
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RetainedResourceKind {
    Buffer,
    Texture,
    Sampler,
    QuerySet,
    ComputePipeline,
    RenderPipeline,
}

fn retain_copy_resources(copy: &HalCopy, retained: &mut RetainedResources) {
    match copy {
        HalCopy::Buffer(copy) => {
            retain_hal_buffer(&copy.source, retained);
            retain_hal_buffer(&copy.destination, retained);
        }
        HalCopy::BufferClear(clear) => retain_hal_buffer(&clear.buffer, retained),
        HalCopy::ClearTexture(clear) => retain_hal_texture(&clear.texture, retained),
        HalCopy::WriteTimestamp(write) => retain_hal_query_set(&write.query_set, retained),
        HalCopy::ResolveQuerySet(resolve) => {
            retain_hal_query_set(&resolve.query_set, retained);
            retain_hal_buffer(&resolve.destination, retained);
        }
        HalCopy::BufferToTexture(copy) | HalCopy::TextureToBuffer(copy) => {
            retain_hal_buffer(&copy.buffer, retained);
            retain_hal_texture(&copy.texture, retained);
        }
        HalCopy::TextureToTexture(copy) => {
            retain_hal_texture(&copy.source, retained);
            retain_hal_texture(&copy.destination, retained);
        }
        HalCopy::ComputePass(pass) => {
            retain_hal_compute_pipeline(&pass.pipeline, retained);
            for bound in &pass.bind_buffers {
                retain_hal_buffer(&bound.buffer, retained);
            }
            for bound in &pass.bind_textures {
                retain_hal_texture(&bound.texture, retained);
            }
            for bound in &pass.bind_samplers {
                retain_hal_sampler(&bound.sampler, retained);
            }
            for bound in &pass.bind_external_textures {
                retain_hal_external_texture(bound, retained);
            }
            if let HalComputeDispatch::Indirect { buffer } = &pass.dispatch {
                retain_hal_buffer(&buffer.buffer, retained);
            }
        }
        HalCopy::RenderPassCommandStream(pass) => {
            if let Some(query_set) = &pass.occlusion_query_set {
                retain_hal_query_set(query_set, retained);
            }
            for color_target in pass.color_targets.iter().flatten() {
                retain_hal_texture(&color_target.texture, retained);
                if let Some(resolve_target) = &color_target.resolve_target {
                    retain_hal_texture(resolve_target, retained);
                }
            }
            if let Some(depth_stencil_attachment) = &pass.depth_stencil_attachment {
                retain_hal_texture(&depth_stencil_attachment.texture, retained);
            }
            retain_render_stream_command_resources(&pass.commands, retained);
        }
        #[cfg(feature = "tiled")]
        HalCopy::SubpassRenderPass(pass) => retain_subpass_resources(pass, retained),
    }
}

fn retain_render_stream_command_resources(
    commands: &[HalRenderPassCommand],
    retained: &mut RetainedResources,
) {
    for command in commands {
        match command {
            HalRenderPassCommand::SetPipeline(pipeline) => {
                retain_hal_render_pipeline(pipeline, retained);
            }
            HalRenderPassCommand::SetBindGroup {
                buffers,
                textures,
                samplers,
                external_textures,
                ..
            } => {
                for bound in buffers {
                    retain_hal_buffer(&bound.buffer, retained);
                }
                for bound in textures {
                    retain_hal_texture(&bound.texture, retained);
                }
                for bound in samplers {
                    retain_hal_sampler(&bound.sampler, retained);
                }
                for bound in external_textures {
                    retain_hal_external_texture(bound, retained);
                }
            }
            HalRenderPassCommand::SetVertexBuffer {
                buffer: Some(bound),
                ..
            } => retain_hal_buffer(&bound.buffer, retained),
            HalRenderPassCommand::SetIndexBuffer(bound) => {
                retain_hal_buffer(&bound.buffer, retained);
            }
            HalRenderPassCommand::DrawIndirect { indirect_buffer }
            | HalRenderPassCommand::DrawIndexedIndirect { indirect_buffer } => {
                retain_hal_buffer(&indirect_buffer.buffer, retained);
            }
            HalRenderPassCommand::ExecuteRenderBundle(bundle) => {
                retain_render_stream_command_resources(&bundle.commands, retained);
            }
            _ => {}
        }
    }
}

fn retain_hal_buffer(buffer: &crate::HalBuffer, retained: &mut RetainedResources) {
    let crate::HalBuffer::Vulkan(buffer) = buffer else {
        return;
    };
    if let Some(inner) = &buffer.inner {
        retained.retain(
            RetainedResourceKind::Buffer,
            Arc::as_ptr(inner) as usize,
            RetainedResource::Buffer {
                _inner: Arc::clone(inner),
            },
        );
    }
}

fn retain_hal_texture(texture: &HalTexture, retained: &mut RetainedResources) {
    let HalTexture::Vulkan(texture) = texture else {
        return;
    };
    if let Some(inner) = &texture.inner {
        retained.retain(
            RetainedResourceKind::Texture,
            Arc::as_ptr(inner) as usize,
            RetainedResource::Texture {
                _inner: Arc::clone(inner),
            },
        );
    }
}

fn retain_hal_sampler(sampler: &HalSampler, retained: &mut RetainedResources) {
    let HalSampler::Vulkan(sampler) = sampler else {
        return;
    };
    if let Some(inner) = &sampler._inner {
        retained.retain(
            RetainedResourceKind::Sampler,
            Arc::as_ptr(inner) as usize,
            RetainedResource::Sampler {
                _inner: Arc::clone(inner),
            },
        );
    }
}

fn retain_hal_query_set(query_set: &HalQuerySet, retained: &mut RetainedResources) {
    let HalQuerySet::Vulkan(query_set) = query_set else {
        return;
    };
    retained.retain(
        RetainedResourceKind::QuerySet,
        Arc::as_ptr(&query_set.inner) as usize,
        RetainedResource::QuerySet {
            _inner: Arc::clone(&query_set.inner),
        },
    );
}

fn retain_hal_compute_pipeline(
    pipeline: &crate::HalComputePipeline,
    retained: &mut RetainedResources,
) {
    let crate::HalComputePipeline::Vulkan(pipeline) = pipeline else {
        return;
    };
    retained.retain(
        RetainedResourceKind::ComputePipeline,
        Arc::as_ptr(&pipeline.inner) as usize,
        RetainedResource::ComputePipeline {
            _inner: Arc::clone(&pipeline.inner),
        },
    );
}

fn retain_hal_render_pipeline(
    pipeline: &crate::HalRenderPipeline,
    retained: &mut RetainedResources,
) {
    let crate::HalRenderPipeline::Vulkan(pipeline) = pipeline else {
        return;
    };
    retained.retain(
        RetainedResourceKind::RenderPipeline,
        Arc::as_ptr(&pipeline.inner) as usize,
        RetainedResource::RenderPipeline {
            _inner: Arc::clone(&pipeline.inner),
        },
    );
}

fn retain_hal_external_texture(
    texture: &crate::HalBoundExternalTexture,
    retained: &mut RetainedResources,
) {
    retain_hal_texture(&texture.plane0, retained);
    retain_hal_texture(&texture.plane1, retained);
    retain_hal_buffer(&texture.params, retained);
}

#[cfg(feature = "tiled")]
fn retain_subpass_resources(pass: &HalSubpassRenderPassCommand, retained: &mut RetainedResources) {
    for attachment in &pass.color_attachments {
        retain_subpass_attachment_resource(&attachment.resource, retained);
    }
    if let Some(attachment) = &pass.depth_stencil_attachment {
        retain_subpass_attachment_resource(&attachment.resource, retained);
    }
    for draw in &pass.draws {
        retain_hal_render_pipeline(&draw.pipeline, retained);
        for bound in &draw.bind_buffers {
            retain_hal_buffer(&bound.buffer, retained);
        }
        for bound in &draw.bind_textures {
            retain_hal_texture(&bound.texture, retained);
        }
        for bound in &draw.bind_samplers {
            retain_hal_sampler(&bound.sampler, retained);
        }
        for bound in &draw.vertex_buffers {
            retain_hal_buffer(&bound.buffer, retained);
        }
    }
}

#[cfg(feature = "tiled")]
fn retain_subpass_attachment_resource(
    resource: &HalSubpassAttachmentResource,
    retained: &mut RetainedResources,
) {
    match resource {
        HalSubpassAttachmentResource::Persistent {
            texture,
            resolve_target,
        } => {
            retain_hal_texture(texture, retained);
            if let Some(resolve_target) = resolve_target {
                retain_hal_texture(resolve_target, retained);
            }
        }
    }
}

fn find_surface_pending(copies: &[HalCopy]) -> Option<Arc<Mutex<SurfacePendingState>>> {
    copies.iter().find_map(surface_pending_from_copy)
}

fn surface_pending_from_copy(copy: &HalCopy) -> Option<Arc<Mutex<SurfacePendingState>>> {
    match copy {
        HalCopy::Buffer(_)
        | HalCopy::BufferClear(_)
        | HalCopy::ClearTexture(_)
        | HalCopy::WriteTimestamp(_)
        | HalCopy::ResolveQuerySet(_)
        | HalCopy::ComputePass(_) => None,
        #[cfg(feature = "tiled")]
        HalCopy::SubpassRenderPass(pass) => surface_pending_from_subpass(pass),
        HalCopy::BufferToTexture(copy) | HalCopy::TextureToBuffer(copy) => {
            surface_pending_from_hal_texture(&copy.texture)
        }
        HalCopy::TextureToTexture(copy) => surface_pending_from_hal_texture(&copy.source)
            .or_else(|| surface_pending_from_hal_texture(&copy.destination)),
        HalCopy::RenderPassCommandStream(pass) => {
            pass.color_targets.iter().flatten().find_map(|target| {
                surface_pending_from_hal_texture(&target.texture).or_else(|| {
                    target
                        .resolve_target
                        .as_ref()
                        .and_then(surface_pending_from_hal_texture)
                })
            })
        }
    }
}

fn surface_pending_from_hal_texture(
    texture: &HalTexture,
) -> Option<Arc<Mutex<SurfacePendingState>>> {
    let HalTexture::Vulkan(texture) = texture else {
        return None;
    };
    texture.surface_pending.as_ref().map(Arc::clone)
}

#[cfg(feature = "tiled")]
fn surface_pending_from_subpass(
    _pass: &HalSubpassRenderPassCommand,
) -> Option<Arc<Mutex<SurfacePendingState>>> {
    None
}

/// Records encode into the command stream.
pub(super) fn encode_buffer_copy(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    copy: &HalBufferCopy,
) -> Result<(), HalError> {
    let crate::HalBuffer::Vulkan(source) = &copy.source else {
        return Err(buffer_error("source buffer is not Vulkan-backed"));
    };
    let crate::HalBuffer::Vulkan(destination) = &copy.destination else {
        return Err(buffer_error("destination buffer is not Vulkan-backed"));
    };
    source.validate_range(copy.source_offset, copy.size)?;
    destination.validate_range(copy.destination_offset, copy.size)?;
    if copy.size == 0 {
        return Ok(());
    }
    let source = source.inner()?;
    let destination = destination.inner()?;
    let region = vk::BufferCopy::default()
        .src_offset(copy.source_offset)
        .dst_offset(copy.destination_offset)
        .size(copy.size);
    unsafe {
        device.cmd_copy_buffer(command_buffer, source.buffer, destination.buffer, &[region]);
    }
    transfer_to_compute_barrier(device, command_buffer);
    Ok(())
}

/// Records buffer clear encode into the command stream.
pub(super) fn encode_buffer_clear(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    clear: &HalBufferClear,
) -> Result<(), HalError> {
    let crate::HalBuffer::Vulkan(buffer) = &clear.buffer else {
        return Err(buffer_error("buffer is not Vulkan-backed"));
    };
    buffer.validate_range(clear.offset, clear.size)?;
    if clear.size == 0 {
        return Ok(());
    }
    let buffer = buffer.inner()?;
    unsafe {
        device.cmd_fill_buffer(command_buffer, buffer.buffer, clear.offset, clear.size, 0);
    }
    transfer_to_compute_barrier(device, command_buffer);
    Ok(())
}

/// Selects how a `HalCopy::ClearTexture` is executed on Vulkan (Block 104 R6).
///
/// `vkCmdClearColorImage` is only defined for uncompressed color images, so the
/// depth / stencil and block-compressed subresources that Stage 2 now asks the
/// HAL to zero take the two other paths Dawn's `Texture::ClearTexture`
/// (`TextureVk.cpp`) uses for them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClearKind {
    /// Uncompressed color, any sample count: `vkCmdClearColorImage`, which is
    /// valid on a multisampled image.
    Color,
    /// Depth and/or stencil planes: `vkCmdClearDepthStencilImage` over this
    /// aspect mask. The mask is empty when the requested aspect does not exist
    /// on the format, in which case nothing is cleared.
    DepthStencil(vk::ImageAspectFlags),
    /// Block-compressed color: a zero-filled transient buffer copied in with
    /// `vkCmdCopyBufferToImage`, because a compressed image rejects both image
    /// clear commands.
    Compressed {
        /// Bytes occupied by one compressed block.
        block_bytes: u32,
        /// Block width in texels.
        block_width: u32,
        /// Block height in texels.
        block_height: u32,
    },
}

/// Returns the clear path for `format` restricted to the requested `aspect`.
fn clear_kind(
    format: HalTextureFormat,
    aspect: HalTextureAspect,
    usage: vk::ImageUsageFlags,
) -> Result<ClearKind, HalError> {
    if !usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
        return Err(texture_error(
            "texture clear requires TRANSFER_DST image usage",
        ));
    }
    let has_depth = format_has_depth_aspect(format);
    let has_stencil = format_has_stencil_aspect(format);
    if has_depth || has_stencil {
        let mut aspects = vk::ImageAspectFlags::empty();
        if has_depth && matches!(aspect, HalTextureAspect::All | HalTextureAspect::DepthOnly) {
            aspects |= vk::ImageAspectFlags::DEPTH;
        }
        if has_stencil
            && matches!(
                aspect,
                HalTextureAspect::All | HalTextureAspect::StencilOnly
            )
        {
            aspects |= vk::ImageAspectFlags::STENCIL;
        }
        return Ok(ClearKind::DepthStencil(aspects));
    }
    Ok(match format.compressed_block_info() {
        Some((block_bytes, block_width, block_height)) => ClearKind::Compressed {
            block_bytes,
            block_width,
            block_height,
        },
        None => ClearKind::Color,
    })
}

/// Returns `(bytes_per_row, bytes_per_image)` of the zeroed staging bytes one
/// `width` x `height` texel subresource of a `block` format needs.
///
/// `block` is `(bytes, width, height)` as
/// [`HalTextureFormat::compressed_block_info`] reports it. Both results count
/// whole blocks, so a mip edge that does not fill its last block still gets a
/// full block row.
fn compressed_clear_layout(
    width: u32,
    height: u32,
    block: (u32, u32, u32),
) -> Result<(u64, u64), HalError> {
    let (block_bytes, block_width, block_height) = block;
    if block_width == 0 || block_height == 0 {
        return Err(texture_error("compressed clear block size is zero"));
    }
    let bytes_per_row = u64::from(width.div_ceil(block_width))
        .checked_mul(u64::from(block_bytes))
        .ok_or_else(|| texture_error("compressed clear row size overflows"))?;
    let bytes_per_image = bytes_per_row
        .checked_mul(u64::from(height.div_ceil(block_height)))
        .ok_or_else(|| texture_error("compressed clear image size overflows"))?;
    Ok((bytes_per_row, bytes_per_image))
}

/// Rounds one mip axis up to a whole block grid (the physical mip size).
fn block_aligned_extent(extent: u32, block: u32) -> Result<u32, HalError> {
    if block == 0 {
        return Err(texture_error("compressed clear block size is zero"));
    }
    extent
        .div_ceil(block)
        .checked_mul(block)
        .ok_or_else(|| texture_error("compressed clear physical extent overflows"))
}

/// Returns the mip/layer rectangle touched by a texture clear.
fn clear_subresource_range(
    dimension: HalTextureDimension,
    clear: &HalTextureClear,
) -> SubresourceRange {
    copy_subresource_range(
        dimension,
        clear.mip_level,
        clear.base_array_layer,
        clear.array_layer_count,
    )
}

/// Returns the mip/layer rectangle touched by a copy; 3D slices share layer zero.
fn copy_subresource_range(
    dimension: HalTextureDimension,
    mip_level: u32,
    z: u32,
    depth_or_array_layers: u32,
) -> SubresourceRange {
    let (base_array_layer, array_layer_count) = match dimension {
        HalTextureDimension::D3 => (0, 1),
        HalTextureDimension::D1 | HalTextureDimension::D2 => (z, depth_or_array_layers),
    };
    SubresourceRange {
        base_mip_level: mip_level,
        mip_level_count: 1,
        base_array_layer,
        array_layer_count,
    }
}

/// Bounds two validated image subresource rectangles.
fn union_subresource_range(a: SubresourceRange, b: SubresourceRange) -> SubresourceRange {
    let base_mip_level = a.base_mip_level.min(b.base_mip_level);
    let base_array_layer = a.base_array_layer.min(b.base_array_layer);
    SubresourceRange {
        base_mip_level,
        mip_level_count: a
            .base_mip_level
            .saturating_add(a.mip_level_count)
            .max(b.base_mip_level.saturating_add(b.mip_level_count))
            - base_mip_level,
        base_array_layer,
        array_layer_count: a
            .base_array_layer
            .saturating_add(a.array_layer_count)
            .max(b.base_array_layer.saturating_add(b.array_layer_count))
            - base_array_layer,
    }
}

/// Returns the subresource range one `HalTextureClear` covers.
///
/// A 3D texture has a single array layer, so the whole mip — every depth
/// slice — is the clear unit, as in Stage 1.
fn texture_clear_subresource_range(
    texture: &VulkanTexture,
    clear: &HalTextureClear,
    aspect: vk::ImageAspectFlags,
) -> vk::ImageSubresourceRange {
    let range = clear_subresource_range(texture.dimension, clear);
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .base_mip_level(range.base_mip_level)
        .level_count(range.mip_level_count)
        .base_array_layer(range.base_array_layer)
        .layer_count(range.array_layer_count)
}

/// Orders the clear's transfer write against the transfer write or read that
/// follows it in the same submission (F-138).
fn texture_clear_write_after_write_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ);
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

/// Records texture clear encode into the command stream.
///
/// Block 104 R6: the subresources are zeroed for **any** format and sample
/// count. Uncompressed color (including multisampled) uses
/// `vkCmdClearColorImage`; a depth and/or stencil aspect uses
/// `vkCmdClearDepthStencilImage` with `{depth: 0.0, stencil: 0}`; a
/// block-compressed image is filled from a zeroed transient buffer, which the
/// submission retirement ring keeps alive through `temporary_resources`.
///
/// Layout transitions cover the clear's mip and layers and every image aspect,
/// even when the clear itself touches only one depth/stencil plane.
pub(super) fn encode_texture_clear(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    clear: &HalTextureClear,
    temporary_resources: &mut Vec<RetainedResource>,
) -> Result<(), HalError> {
    let crate::HalTexture::Vulkan(texture) = &clear.texture else {
        return Err(texture_error("texture is not Vulkan-backed"));
    };
    validate_mip_level(texture, clear.mip_level)?;
    let texture_inner = texture.inner()?;
    match clear_kind(clear.format, clear.aspect, texture_inner.usage)? {
        ClearKind::Color => {
            let range =
                texture_clear_subresource_range(texture, clear, vk::ImageAspectFlags::COLOR);
            transition_image_range(
                device,
                command_buffer,
                texture_inner,
                clear_subresource_range(texture.dimension, clear),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                IMAGE_LAYOUT_TRANSFER_DST,
            )?;
            let value = unsafe { vulkan_color_clear_value(clear.format, [0.0; 4]).color };
            unsafe {
                device.cmd_clear_color_image(
                    command_buffer,
                    texture_inner.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &value,
                    &[range],
                );
            }
            texture_clear_write_after_write_barrier(
                device,
                command_buffer,
                texture_inner.image,
                range,
            );
        }
        ClearKind::DepthStencil(aspects) => {
            if aspects.is_empty() {
                return Ok(());
            }
            let range = texture_clear_subresource_range(texture, clear, aspects);
            transition_image_range(
                device,
                command_buffer,
                texture_inner,
                clear_subresource_range(texture.dimension, clear),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                IMAGE_LAYOUT_TRANSFER_DST,
            )?;
            unsafe {
                device.cmd_clear_depth_stencil_image(
                    command_buffer,
                    texture_inner.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                    &[range],
                );
            }
            // The clear itself is aspect-scoped, but a layout barrier on a
            // combined depth-stencil image must name both aspects
            // (VUID-VkImageMemoryBarrier-image-03320).
            texture_clear_write_after_write_barrier(
                device,
                command_buffer,
                texture_inner.image,
                vk::ImageSubresourceRange {
                    aspect_mask: texture_inner.aspect_flags,
                    ..range
                },
            );
        }
        ClearKind::Compressed {
            block_bytes,
            block_width,
            block_height,
        } => encode_compressed_texture_clear(
            device,
            command_buffer,
            clear,
            texture,
            texture_inner,
            (block_bytes, block_width, block_height),
            temporary_resources,
        )?,
    }
    Ok(())
}

/// Zeroes a block-compressed subresource with a buffer-to-image copy of a
/// zero-filled transient buffer, the path Dawn takes for a format both
/// `vkCmdClearColorImage` and `vkCmdClearDepthStencilImage` reject.
///
/// The staging buffer is zeroed on the GPU with `vkCmdFillBuffer` so no host
/// allocation scales with the subresource, and its size is rounded up to the
/// four bytes that command requires. One `VkBufferImageCopy` per array layer
/// (per depth slice for a 3D texture) keeps every `bufferOffset` a multiple of
/// both four and the block size.
fn encode_compressed_texture_clear(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    clear: &HalTextureClear,
    texture: &VulkanTexture,
    texture_inner: &VulkanTextureInner,
    block: (u32, u32, u32),
    temporary_resources: &mut Vec<RetainedResource>,
) -> Result<(), HalError> {
    let (block_bytes, block_width, block_height) = block;
    let (mip_width, mip_height) = mip_extent(texture.width, texture.height, clear.mip_level);
    let (bytes_per_row, bytes_per_image) = compressed_clear_layout(mip_width, mip_height, block)?;
    // Vulkan copies whole blocks: the region covers the block-rounded physical
    // mip, and `buffer_image_copy` clamps `imageExtent` back to the logical mip
    // size at the edge.
    let physical_width = block_aligned_extent(mip_width, block_width)?;
    let physical_height = block_aligned_extent(mip_height, block_height)?;
    let (base_slice, slice_count) = match texture.dimension {
        HalTextureDimension::D3 => (
            0,
            texture
                .depth_or_array_layers
                .checked_shr(clear.mip_level)
                .unwrap_or(0)
                .max(1),
        ),
        HalTextureDimension::D1 | HalTextureDimension::D2 => {
            (clear.base_array_layer, clear.array_layer_count)
        }
    };
    if slice_count == 0 {
        return Ok(());
    }
    let size = bytes_per_image
        .checked_mul(u64::from(slice_count))
        .and_then(|size| size.checked_next_multiple_of(4))
        .ok_or_else(|| texture_error("compressed clear buffer size overflows"))?;
    let bytes_per_row = u32::try_from(bytes_per_row)
        .map_err(|_| texture_error("compressed clear bytes per row is too large"))?;
    let rows_per_image = mip_height.div_ceil(block_height);
    let zeros = Arc::new(create_buffer(
        Arc::clone(&texture_inner.device),
        size,
        HalBufferUsage {
            copy_src: true,
            copy_dst: true,
            ..Default::default()
        },
    )?);
    let zeros_handle = zeros.buffer;
    // The submission retirement ring owns the allocation through fence
    // completion, as it does for the temporary buffer of a compressed
    // texture-to-texture copy. On recording errors the caller drops it after
    // destroying the unsubmitted command pool.
    temporary_resources.push(RetainedResource::Buffer {
        _inner: Arc::clone(&zeros),
    });
    let staging = VulkanBuffer {
        inner: Some(zeros),
        size,
    };
    let mut regions = Vec::new();
    for slice in 0..slice_count {
        let offset = bytes_per_image
            .checked_mul(u64::from(slice))
            .ok_or_else(|| texture_error("compressed clear buffer offset overflows"))?;
        let z = base_slice
            .checked_add(slice)
            .ok_or_else(|| texture_error("compressed clear layer range overflows"))?;
        let copy = HalBufferTextureCopy {
            buffer: HalBuffer::Vulkan(staging.clone()),
            buffer_layout: crate::HalBufferTextureLayout {
                offset,
                bytes_per_row,
                rows_per_image,
            },
            texture: clear.texture.clone(),
            format: clear.format,
            aspect: HalTextureAspect::All,
            mip_level: clear.mip_level,
            origin: crate::HalOrigin3d { x: 0, y: 0, z },
            extent: HalExtent3d {
                width: physical_width,
                height: physical_height,
                depth_or_array_layers: 1,
            },
        };
        regions.push(buffer_image_copy(
            &copy,
            texture,
            block_bytes,
            vk::ImageAspectFlags::COLOR,
        )?);
    }
    transition_image_range(
        device,
        command_buffer,
        texture_inner,
        clear_subresource_range(texture.dimension, clear),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_DST,
    )?;
    unsafe {
        device.cmd_fill_buffer(command_buffer, zeros_handle, 0, size, 0);
        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(zeros_handle)
            .offset(0)
            .size(size);
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
        device.cmd_copy_buffer_to_image(
            command_buffer,
            zeros_handle,
            texture_inner.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );
    }
    texture_clear_write_after_write_barrier(
        device,
        command_buffer,
        texture_inner.image,
        texture_clear_subresource_range(texture, clear, vk::ImageAspectFlags::COLOR),
    );
    Ok(())
}

/// Records a raw timestamp write into the command stream (Block 102 R3).
///
/// Mirrors Dawn's `RecordWriteTimestampCmd` (`CommandBufferVk.cpp`) at top
/// level: a query must be reset before it is written again, and
/// `vkCmdResetQueryPool` cannot be recorded inside a render pass instance —
/// which holds here because `HalCopy` commands are always recorded at the top
/// level of the command buffer. The stamp itself is taken at `ALL_COMMANDS`,
/// the stage Dawn uses for a standalone `writeTimestamp`.
pub(super) fn encode_write_timestamp(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    write: &HalWriteTimestamp,
) -> Result<(), HalError> {
    let HalQuerySet::Vulkan(query_set) = &write.query_set else {
        return Err(buffer_error("query set is not Vulkan-backed"));
    };
    if query_set.kind() != HalQueryKind::Timestamp {
        return Err(buffer_error(
            "timestamp write requires a timestamp query set",
        ));
    }
    query_set.validate_query(write.query_index)?;
    unsafe {
        device.cmd_reset_query_pool(command_buffer, query_set.pool(), write.query_index, 1);
        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            query_set.pool(),
            write.query_index,
        );
    }
    Ok(())
}

/// Records query-set resolve encode into the command stream.
pub(super) fn encode_resolve_query_set(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    resolve: &HalResolveQuerySet,
) -> Result<(), HalError> {
    let HalQuerySet::Vulkan(query_set) = &resolve.query_set else {
        return Err(buffer_error("query set is not Vulkan-backed"));
    };
    let HalBuffer::Vulkan(destination) = &resolve.destination else {
        return Err(buffer_error(
            "query resolve destination is not Vulkan-backed",
        ));
    };
    let byte_count = u64::from(resolve.query_count)
        .checked_mul(8)
        .ok_or_else(|| buffer_error("query resolve byte count overflows"))?;
    destination.validate_range(resolve.destination_offset, byte_count)?;
    query_set.validate_range(resolve.first_query, resolve.query_count)?;
    for &query_index in &resolve.written_queries {
        if query_index < resolve.first_query {
            return Err(buffer_error("written query precedes resolve range"));
        }
        let relative_index = query_index - resolve.first_query;
        if relative_index >= resolve.query_count {
            return Err(buffer_error("written query exceeds resolve range"));
        }
        query_set.validate_query(query_index)?;
    }
    if byte_count == 0 {
        return Ok(());
    }
    let destination_buffer = destination.inner()?.buffer;
    unsafe {
        device.cmd_fill_buffer(
            command_buffer,
            destination_buffer,
            resolve.destination_offset,
            byte_count,
            0,
        );
        query_resolve_fill_to_copy_barrier(
            device,
            command_buffer,
            destination_buffer,
            resolve.destination_offset,
            byte_count,
        );
        for &query_index in &resolve.written_queries {
            let destination_offset = resolve
                .destination_offset
                .checked_add(u64::from(query_index - resolve.first_query) * 8)
                .ok_or_else(|| buffer_error("query resolve destination offset overflows"))?;
            device.cmd_copy_query_pool_results(
                command_buffer,
                query_set.pool(),
                query_index,
                1,
                destination_buffer,
                destination_offset,
                8,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            );
        }
        if query_resolve_needs_shader_barrier(query_set.kind()) {
            query_resolve_copy_to_shader_barrier(
                device,
                command_buffer,
                destination_buffer,
                resolve.destination_offset,
                byte_count,
            );
        }
    }
    Ok(())
}

fn query_resolve_needs_shader_barrier(kind: HalQueryKind) -> bool {
    kind == HalQueryKind::Timestamp
}

fn query_resolve_fill_to_copy_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(size);
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}

/// Makes the resolved query results visible to the compute pass that converts
/// them (Block 102 R3).
///
/// `vkCmdCopyQueryPoolResults` is a transfer write, and for a timestamp set
/// core follows the resolve with the timestamp-to-nanoseconds conversion
/// dispatch reading the very same range as a storage buffer. The range barrier
/// pairs the copies with those shader accesses; the compute pass's own global
/// barrier covers the same hazard more coarsely, so this one keeps the
/// dependency scoped to the resolved bytes.
fn query_resolve_copy_to_shader_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
) {
    let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(size);
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[barrier],
            &[],
        );
    }
}

/// Records encode into the command stream.
pub(super) fn encode_buffer_to_texture(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let crate::HalBuffer::Vulkan(buffer) = &copy.buffer else {
        return Err(buffer_error("buffer is not Vulkan-backed"));
    };
    let crate::HalTexture::Vulkan(texture) = &copy.texture else {
        return Err(texture_error("texture is not Vulkan-backed"));
    };
    validate_mip_level(texture, copy.mip_level)?;
    texture.validate_origin_extent(copy.origin, copy.extent)?;
    validate_buffer_texture_range(buffer, copy)?;
    let buffer = buffer.inner()?;
    let texture_inner = texture.inner()?;
    let aspect = buffer_texture_copy_aspect_flags(copy.format, copy.aspect);
    let region = buffer_image_copy(copy, texture, texture_bytes_per_pixel(copy)?, aspect)?;
    // The layout barrier covers the copied mip/layers and both depth/stencil
    // aspects. Only the copy region above may narrow the aspect selection.
    transition_image_range(
        device,
        command_buffer,
        texture_inner,
        copy_subresource_range(
            texture.dimension,
            copy.mip_level,
            copy.origin.z,
            copy.extent.depth_or_array_layers,
        ),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_DST,
    )?;
    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer.buffer,
            texture_inner.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    Ok(())
}

/// Records encode into the command stream.
pub(super) fn encode_texture_to_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let crate::HalBuffer::Vulkan(buffer) = &copy.buffer else {
        return Err(buffer_error("buffer is not Vulkan-backed"));
    };
    let crate::HalTexture::Vulkan(texture) = &copy.texture else {
        return Err(texture_error("texture is not Vulkan-backed"));
    };
    validate_mip_level(texture, copy.mip_level)?;
    texture.validate_origin_extent(copy.origin, copy.extent)?;
    validate_buffer_texture_range(buffer, copy)?;
    let buffer = buffer.inner()?;
    let texture_inner = texture.inner()?;
    let aspect = buffer_texture_copy_aspect_flags(copy.format, copy.aspect);
    let region = buffer_image_copy(copy, texture, texture_bytes_per_pixel(copy)?, aspect)?;
    // The layout barrier covers the copied mip/layers and both depth/stencil
    // aspects. Only the copy region above may narrow the aspect selection.
    transition_image_range(
        device,
        command_buffer,
        texture_inner,
        copy_subresource_range(
            texture.dimension,
            copy.mip_level,
            copy.origin.z,
            copy.extent.depth_or_array_layers,
        ),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_SRC,
    )?;
    unsafe {
        device.cmd_copy_image_to_buffer(
            command_buffer,
            texture_inner.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer.buffer,
            &[region],
        );
    }
    transfer_to_compute_barrier(device, command_buffer);
    Ok(())
}

/// Records encode into the command stream.
pub(super) fn encode_texture_to_texture(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    copy: &HalTextureCopy,
    temporary_resources: &mut Vec<RetainedResource>,
) -> Result<(), HalError> {
    let crate::HalTexture::Vulkan(source) = &copy.source else {
        return Err(texture_error("source texture is not Vulkan-backed"));
    };
    let crate::HalTexture::Vulkan(destination) = &copy.destination else {
        return Err(texture_error("destination texture is not Vulkan-backed"));
    };
    validate_mip_level(source, copy.source_mip_level)?;
    validate_mip_level(destination, copy.destination_mip_level)?;
    source.validate_origin_extent(copy.source_origin, copy.extent)?;
    destination.validate_origin_extent(copy.destination_origin, copy.extent)?;
    let source_inner = source.inner()?;
    let destination_inner = destination.inner()?;
    let aspect = copy_format_aspect_flags(source.format);
    let source_range = copy_subresource_range(
        source.dimension,
        copy.source_mip_level,
        copy.source_origin.z,
        copy.extent.depth_or_array_layers,
    );
    let destination_range = copy_subresource_range(
        destination.dimension,
        copy.destination_mip_level,
        copy.destination_origin.z,
        copy.extent.depth_or_array_layers,
    );
    let source_extent = compressed_copy_extent(
        source,
        copy.source_mip_level,
        copy.source_origin,
        copy.extent,
    )?;
    let destination_extent = compressed_copy_extent(
        destination,
        copy.destination_mip_level,
        copy.destination_origin,
        copy.extent,
    )?;
    if compressed_copy_needs_temporary_buffer(
        source_extent,
        destination_extent,
        source.format.compressed_block_info().is_some(),
    ) {
        let temporary_copy = compressed_temporary_copy(source, destination, copy)?;
        let buffer = Arc::new(create_buffer(
            Arc::clone(&source_inner.device),
            temporary_copy.size,
            HalBufferUsage {
                copy_src: true,
                copy_dst: true,
                ..Default::default()
            },
        )?);
        let buffer_handle = buffer.buffer;
        // The existing submission retirement ring owns the allocation through
        // fence completion, including surface submissions. On recording errors,
        // the caller drops it after destroying the unsubmitted command pool.
        temporary_resources.push(RetainedResource::Buffer { _inner: buffer });
        transition_image_range(
            device,
            command_buffer,
            source_inner,
            source_range,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            IMAGE_LAYOUT_TRANSFER_SRC,
        )?;
        unsafe {
            device.cmd_copy_image_to_buffer(
                command_buffer,
                source_inner.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer_handle,
                &[temporary_copy.source],
            );
            let barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(buffer_handle)
                .offset(0)
                .size(temporary_copy.size);
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
        // Transition the destination after the read, including shared images.
        transition_image_range(
            device,
            command_buffer,
            destination_inner,
            destination_range,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            IMAGE_LAYOUT_TRANSFER_DST,
        )?;
        unsafe {
            device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer_handle,
                destination_inner.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[temporary_copy.destination],
            );
        }
        return Ok(());
    }
    let image_extent = source_extent;
    let region = vk::ImageCopy::default()
        .src_subresource(texture_copy_subresource_layers(
            aspect,
            source.dimension,
            copy.source_mip_level,
            copy.source_origin.z,
            copy.extent.depth_or_array_layers,
        ))
        .src_offset(texture_copy_offset(
            source.dimension,
            copy.source_origin.x,
            copy.source_origin.y,
            copy.source_origin.z,
        )?)
        .dst_subresource(texture_copy_subresource_layers(
            aspect,
            destination.dimension,
            copy.destination_mip_level,
            copy.destination_origin.z,
            copy.extent.depth_or_array_layers,
        ))
        .dst_offset(texture_copy_offset(
            destination.dimension,
            copy.destination_origin.x,
            copy.destination_origin.y,
            copy.destination_origin.z,
        )?)
        .extent(image_extent);
    let same_image = source_inner.image == destination_inner.image;
    let (source_layout, destination_layout) =
        texture_copy_layouts(same_image, region.src_subresource, region.dst_subresource);
    if same_image && source_layout == vk::ImageLayout::GENERAL {
        transition_image_range(
            device,
            command_buffer,
            source_inner,
            union_subresource_range(source_range, destination_range),
            vk::ImageLayout::GENERAL,
            IMAGE_LAYOUT_GENERAL,
        )?;
    } else {
        transition_image_range(
            device,
            command_buffer,
            source_inner,
            source_range,
            source_layout,
            IMAGE_LAYOUT_TRANSFER_SRC,
        )?;
        transition_image_range(
            device,
            command_buffer,
            destination_inner,
            destination_range,
            destination_layout,
            IMAGE_LAYOUT_TRANSFER_DST,
        )?;
    }
    unsafe {
        device.cmd_copy_image(
            command_buffer,
            source_inner.image,
            source_layout,
            destination_inner.image,
            destination_layout,
            &[region],
        );
    }
    Ok(())
}

// 3D depth slices all occupy array layer zero of the same mip.
fn texture_copy_layouts(
    same_image: bool,
    source: vk::ImageSubresourceLayers,
    destination: vk::ImageSubresourceLayers,
) -> (vk::ImageLayout, vk::ImageLayout) {
    let layers_overlap = u64::from(source.base_array_layer)
        < u64::from(destination.base_array_layer) + u64::from(destination.layer_count)
        && u64::from(destination.base_array_layer)
            < u64::from(source.base_array_layer) + u64::from(source.layer_count);
    if same_image && source.mip_level == destination.mip_level && layers_overlap {
        (vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL)
    } else {
        (
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )
    }
}

/// Records encode into the command stream.
pub(super) fn encode_compute_pass(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pass: &HalComputePass,
) -> Result<ComputePassTemps, HalError> {
    let crate::HalComputePipeline::Vulkan(pipeline) = &pass.pipeline else {
        return Err(shader_error("compute pipeline is not Vulkan-backed"));
    };
    let descriptor_pool = create_compute_descriptor_pool(device, pipeline)?;
    let descriptor_sets = if let Some(pool) = descriptor_pool {
        match allocate_descriptor_sets(device, pool, &pipeline.inner.descriptor_set_layouts) {
            Ok(sets) => sets,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
                return Err(error);
            }
        }
    } else {
        Vec::new()
    };
    let image_views = match update_compute_descriptor_sets(device, pipeline, pass, &descriptor_sets)
    {
        Ok(image_views) => image_views,
        Err(error) => {
            if let Some(pool) = descriptor_pool {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
            }
            return Err(error);
        }
    };
    transition_sampled_textures(device, command_buffer, &pass.bind_textures, &[])?;
    transition_storage_textures(device, command_buffer, &pass.bind_textures, &[])?;
    record_memory_barrier(
        device,
        command_buffer,
        compute_pass_pre_dispatch_barrier_scopes(),
    );
    unsafe {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.inner.pipeline,
        );
        if !descriptor_sets.is_empty() {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.inner.pipeline_layout,
                0,
                &descriptor_sets,
                &[],
            );
        }
        // Deliver the user immediates push-constant block (Block 94 S3).
        // Compute pipelines have no internal immediates, so the block is
        // exactly the pass's user prefix; the pipeline layout declared a
        // matching compute-stage range.
        if let Some(immediates) = pipeline.inner.immediates {
            let block = crate::immediates::compose_immediates_block(
                &pass.immediate_data,
                immediates.block_size,
                immediates.depth_range_offset,
                [0.0, 1.0],
            );
            device.cmd_push_constants(
                command_buffer,
                pipeline.inner.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &block,
            );
        }
        match &pass.dispatch {
            HalComputeDispatch::Direct { workgroups } => {
                // WebGPU: a dispatch with any zero workgroup count does
                // nothing, so skip the API call. This also works around a
                // Mesa ANV (Haswell) driver bug where a zero-dimension
                // vkCmdDispatch hard-wedges the GPU. Indirect dispatches
                // cannot be pre-checked CPU-side and are left as-is.
                if workgroups.0 != 0 && workgroups.1 != 0 && workgroups.2 != 0 {
                    device.cmd_dispatch(command_buffer, workgroups.0, workgroups.1, workgroups.2);
                }
            }
            HalComputeDispatch::Indirect { buffer } => {
                let HalBuffer::Vulkan(indirect_buffer) = &buffer.buffer else {
                    return Err(buffer_error("compute indirect buffer is not Vulkan-backed"));
                };
                device.cmd_dispatch_indirect(
                    command_buffer,
                    indirect_buffer.inner()?.buffer,
                    buffer.offset,
                );
            }
        }
    }
    record_memory_barrier(
        device,
        command_buffer,
        compute_pass_post_dispatch_barrier_scopes(),
    );
    Ok(ComputePassTemps {
        descriptor_pool,
        image_views,
    })
}

/// Stage and access scopes of a global (`VkMemoryBarrier`) dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MemoryBarrierScopes {
    src_stages: vk::PipelineStageFlags,
    src_access: vk::AccessFlags,
    dst_stages: vk::PipelineStageFlags,
    dst_access: vk::AccessFlags,
}

/// Records a global memory barrier with the given scopes.
fn record_memory_barrier(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    scopes: MemoryBarrierScopes,
) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(scopes.src_access)
        .dst_access_mask(scopes.dst_access);
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            scopes.src_stages,
            scopes.dst_stages,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// Scopes of the barrier recorded before a compute pass's dispatch
/// (Block 102 R3).
///
/// Vulkan orders nothing between a transfer write and a later shader access,
/// so a dispatch that consumes buffer data produced earlier in the same
/// command buffer needs an explicit dependency — the timestamp resolve
/// (`vkCmdFillBuffer` + `vkCmdCopyQueryPoolResults`) followed by the
/// nanosecond conversion dispatch over the same range is exactly that shape.
/// The source scope covers every write from the transfer and compute stages
/// (`MEMORY_WRITE` subsumes `TRANSFER_WRITE`), so a preceding copy, fill,
/// query resolve or dispatch is included. The destination scope keeps the
/// wider buffer-read mask on top of R3's `SHADER_READ | SHADER_WRITE` so that
/// index, indirect and copy-source reads recorded after this pass still
/// observe those earlier transfer writes (F-106).
fn compute_pass_pre_dispatch_barrier_scopes() -> MemoryBarrierScopes {
    MemoryBarrierScopes {
        src_stages: vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
        src_access: vk::AccessFlags::MEMORY_WRITE,
        dst_stages: vk::PipelineStageFlags::COMPUTE_SHADER
            | buffer_write_read_barrier_dst_stage_mask(),
        dst_access: vk::AccessFlags::SHADER_READ
            | vk::AccessFlags::SHADER_WRITE
            | buffer_write_read_barrier_dst_access_mask(),
    }
}

/// Scopes of the barrier recorded after a compute pass's dispatch
/// (Block 102 R3).
///
/// The dispatch's storage-buffer writes are made available to everything that
/// can consume them next: a following transfer (the conversion pass is
/// normally followed by a copy out of the resolve destination), another
/// dispatch, a draw reading index/indirect/vertex data (F-106), and the host
/// once the submission's fence is signalled.
fn compute_pass_post_dispatch_barrier_scopes() -> MemoryBarrierScopes {
    MemoryBarrierScopes {
        src_stages: vk::PipelineStageFlags::COMPUTE_SHADER,
        src_access: vk::AccessFlags::SHADER_WRITE,
        dst_stages: vk::PipelineStageFlags::TRANSFER
            | vk::PipelineStageFlags::COMPUTE_SHADER
            | vk::PipelineStageFlags::HOST
            | buffer_write_read_barrier_dst_stage_mask(),
        dst_access: vk::AccessFlags::MEMORY_READ
            | vk::AccessFlags::MEMORY_WRITE
            | buffer_write_read_barrier_dst_access_mask(),
    }
}

/// Returns the single attachment mip/layer; 3D slices share image array layer zero.
fn attachment_subresource_range_of(
    dimension: HalTextureDimension,
    mip_level: u32,
    array_layer: u32,
) -> SubresourceRange {
    SubresourceRange {
        base_mip_level: mip_level,
        mip_level_count: 1,
        base_array_layer: match dimension {
            HalTextureDimension::D3 => 0,
            HalTextureDimension::D1 | HalTextureDimension::D2 => array_layer,
        },
        array_layer_count: 1,
    }
}

/// Returns the exact mip/layer rectangle exposed by a bound texture view.
fn bound_view_subresource_range(bound: &HalBoundTexture) -> SubresourceRange {
    SubresourceRange {
        base_mip_level: bound.base_mip_level,
        mip_level_count: bound.mip_level_count,
        base_array_layer: bound.base_array_layer,
        array_layer_count: bound.array_layer_count,
    }
}

/// Subtracts exclusions, merging adjacent layers and then identical mip run lists.
fn subtract_subresource_ranges(
    bound: SubresourceRange,
    exclusions: &[SubresourceRange],
) -> Vec<SubresourceRange> {
    coalesce_subresources(bound, |mip, layer| {
        (!exclusions.iter().any(|range| {
            mip.checked_sub(range.base_mip_level)
                .is_some_and(|offset| offset < range.mip_level_count)
                && layer
                    .checked_sub(range.base_array_layer)
                    .is_some_and(|offset| offset < range.array_layer_count)
        }))
        .then_some(())
    })
    .into_iter()
    .map(|(range, ())| range)
    .collect()
}

/// Transitions storage views to GENERAL outside this pass's attachment subresources.
/// Attachment intersections stay skipped, preserving Block 105 "Known limitations".
fn transition_storage_textures(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    textures: &[HalBoundTexture],
    attachments: &[(vk::Image, SubresourceRange)],
) -> Result<(), HalError> {
    for bound in textures
        .iter()
        .filter(|bound| bound.storage_access.is_some())
    {
        let crate::HalTexture::Vulkan(texture) = &bound.texture else {
            return Err(texture_error("storage texture is not Vulkan-backed"));
        };
        let inner = texture.inner()?;
        let exclusions: Vec<_> = attachments
            .iter()
            .filter_map(|(image, range)| (*image == inner.image).then_some(*range))
            .collect();
        for range in subtract_subresource_ranges(bound_view_subresource_range(bound), &exclusions) {
            transition_image_range(
                device,
                command_buffer,
                inner,
                range,
                vk::ImageLayout::GENERAL,
                IMAGE_LAYOUT_GENERAL,
            )?;
        }
    }
    Ok(())
}

/// Transitions sampled views to SHADER_READ_ONLY_OPTIMAL outside attachments.
/// A read-only depth-stencil attachment sampled in the same pass stays skipped
/// (Block 105 "Known limitations"); other mips/layers of its image still transition.
fn transition_sampled_textures(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    textures: &[HalBoundTexture],
    attachments: &[(vk::Image, SubresourceRange)],
) -> Result<(), HalError> {
    for bound in textures
        .iter()
        .filter(|bound| bound.storage_access.is_none())
    {
        let crate::HalTexture::Vulkan(texture) = &bound.texture else {
            return Err(texture_error("sampled texture is not Vulkan-backed"));
        };
        let inner = texture.inner()?;
        let exclusions: Vec<_> = attachments
            .iter()
            .filter_map(|(image, range)| (*image == inner.image).then_some(*range))
            .collect();
        for range in subtract_subresource_ranges(bound_view_subresource_range(bound), &exclusions) {
            transition_image_range(
                device,
                command_buffer,
                inner,
                range,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                IMAGE_LAYOUT_SHADER_READ_ONLY,
            )?;
        }
    }
    Ok(())
}

/// Stores compute pass temps data used by backend submission cleanup.
pub(super) struct ComputePassTemps {
    descriptor_pool: Option<vk::DescriptorPool>,
    image_views: Vec<vk::ImageView>,
}

/// Stores render pass temps data used by validation and backend submission.
pub(super) struct RenderPassTemps {
    descriptor_pools: Vec<vk::DescriptorPool>,
    framebuffer: vk::Framebuffer,
    image_views: Vec<vk::ImageView>,
    render_pass: Option<vk::RenderPass>,
}

/// Records a tiled subpass render pass into the command stream.
#[cfg(feature = "tiled")]
pub(super) fn encode_subpass_render_pass(
    device: &VulkanDeviceInner,
    command_buffer: vk::CommandBuffer,
    pass: &HalSubpassRenderPassCommand,
) -> Result<RenderPassTemps, HalError> {
    if pass.layout.subpasses.is_empty() {
        return Err(shader_error(
            "subpass render pass requires at least one subpass",
        ));
    }
    let render_pass = cached_subpass_render_pass(device, pass)?;
    let (views, persistent_textures, depth_stencil_texture) = subpass_attachment_views(pass)?;
    for (slot, texture) in persistent_textures.iter() {
        let is_input_source = pass.layout.subpasses.iter().any(|subpass| {
            subpass
                .input_attachments
                .iter()
                .any(|input| input.source_attachment == *slot)
        });
        let (layout, layout_id) = if is_input_source {
            (vk::ImageLayout::GENERAL, IMAGE_LAYOUT_GENERAL)
        } else {
            (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                IMAGE_LAYOUT_COLOR_ATTACHMENT,
            )
        };
        transition_image(
            &device.device,
            command_buffer,
            texture.inner()?,
            layout,
            layout_id,
        )?;
    }
    if let Some(texture) = depth_stencil_texture {
        transition_image(
            &device.device,
            command_buffer,
            texture.inner()?,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT,
        )?;
    }
    let mut attachments = Vec::new();
    for resource in pass
        .color_attachments
        .iter()
        .map(|attachment| &attachment.resource)
        .chain(
            pass.depth_stencil_attachment
                .iter()
                .map(|attachment| &attachment.resource),
        )
    {
        let HalSubpassAttachmentResource::Persistent {
            texture,
            resolve_target,
        } = resource;
        for texture in std::iter::once(texture).chain(resolve_target.iter()) {
            let crate::HalTexture::Vulkan(texture) = texture else {
                return Err(texture_error("subpass attachment is not Vulkan-backed"));
            };
            let inner = texture.inner()?;
            attachments.push((inner.image, inner.layouts.whole()));
        }
    }
    let bound = subpass_bound_textures(pass);
    // All barriers precede the pass and exclude its whole-image attachment views.
    transition_sampled_textures(&device.device, command_buffer, &bound, &attachments)?;
    // Storage follows sampled so GENERAL wins for a view bound both ways.
    transition_storage_textures(&device.device, command_buffer, &bound, &attachments)?;
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&views)
        .width(pass.extent.width)
        .height(pass.extent.height)
        .layers(1);
    let framebuffer = unsafe { device.device.create_framebuffer(&framebuffer_info, None) }
        .map_err(|_| shader_error("subpass framebuffer creation failed"))?;
    let clear_values = subpass_clear_values(pass);
    let render_area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: pass.extent.width,
            height: pass.extent.height,
        },
    };
    let begin_info = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);
    unsafe {
        device.device.cmd_begin_render_pass(
            command_buffer,
            &begin_info,
            vk::SubpassContents::INLINE,
        );
    }
    let mut descriptor_pools = Vec::new();
    let mut image_views = Vec::new();
    for subpass_index in 0..pass.layout.subpasses.len() {
        for draw in pass
            .draws
            .iter()
            .filter(|draw| draw.subpass_index as usize == subpass_index)
        {
            let temps = encode_subpass_draw(&device.device, command_buffer, pass, draw, &views)?;
            if let Some(pool) = temps.descriptor_pool {
                descriptor_pools.push(pool);
            }
            image_views.extend(temps.image_views);
        }
        if subpass_index + 1 < pass.layout.subpasses.len() {
            unsafe {
                device
                    .device
                    .cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);
            }
        }
    }
    unsafe {
        device.device.cmd_end_render_pass(command_buffer);
    }
    for (_, texture) in persistent_textures {
        let inner = texture.inner()?;
        inner.layouts.set(
            inner.layouts.whole(),
            subpass_color_tracked_layout(texture.transient),
        )?;
    }
    Ok(RenderPassTemps {
        descriptor_pools,
        framebuffer,
        image_views,
        render_pass: None,
    })
}

/// Collects bound textures in subpass execution order, retaining repeated bindings.
#[cfg(feature = "tiled")]
fn subpass_bound_textures(pass: &HalSubpassRenderPassCommand) -> Vec<HalBoundTexture> {
    (0..pass.layout.subpasses.len())
        .flat_map(|subpass_index| {
            pass.draws
                .iter()
                .filter(move |draw| draw.subpass_index as usize == subpass_index)
                .flat_map(|draw| draw.bind_textures.iter().cloned())
        })
        .collect()
}

#[cfg(feature = "tiled")]
struct SubpassDrawTemps {
    descriptor_pool: Option<vk::DescriptorPool>,
    image_views: Vec<vk::ImageView>,
}

/// Whole-image views, indexed color textures, and the separate depth-stencil texture.
#[cfg(feature = "tiled")]
type SubpassAttachmentViews = (
    Vec<vk::ImageView>,
    Vec<(u32, VulkanTexture)>,
    Option<VulkanTexture>,
);

#[cfg(feature = "tiled")]
fn cached_subpass_render_pass(
    device: &VulkanDeviceInner,
    pass: &HalSubpassRenderPassCommand,
) -> Result<vk::RenderPass, HalError> {
    let key = pass.layout.clone();
    if let Ok(cache) = device.subpass_render_pass_cache.lock() {
        if let Some(&render_pass) = cache.get(&key) {
            return Ok(render_pass);
        }
    }
    let render_pass = create_subpass_render_pass(&device.device, pass)?;
    match device.subpass_render_pass_cache.lock() {
        Ok(mut cache) => {
            let entry = cache.entry(key).or_insert(render_pass);
            if *entry != render_pass {
                unsafe {
                    device.device.destroy_render_pass(render_pass, None);
                }
            }
            Ok(*entry)
        }
        Err(_) => Ok(render_pass),
    }
}

#[cfg(feature = "tiled")]
fn encode_subpass_draw(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pass: &HalSubpassRenderPassCommand,
    draw: &HalSubpassDraw,
    attachment_views: &[vk::ImageView],
) -> Result<SubpassDrawTemps, HalError> {
    let crate::HalRenderPipeline::Vulkan(pipeline) = &draw.pipeline else {
        return Err(shader_error("subpass render pipeline is not Vulkan-backed"));
    };
    let descriptor_pool = create_render_descriptor_pool(device, pipeline)?;
    let descriptor_sets = if let Some(pool) = descriptor_pool {
        match allocate_descriptor_sets(device, pool, &pipeline.inner.descriptor_set_layouts) {
            Ok(sets) => sets,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
                return Err(error);
            }
        }
    } else {
        Vec::new()
    };
    let image_views = match update_subpass_descriptor_sets(
        device,
        pipeline,
        pass,
        draw,
        &descriptor_sets,
        attachment_views,
    ) {
        Ok(image_views) => image_views,
        Err(error) => {
            if let Some(pool) = descriptor_pool {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
            }
            return Err(error);
        }
    };
    unsafe {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.inner.pipeline,
        );
        let viewport = draw.viewport.map_or(
            vk::Viewport {
                x: 0.0,
                y: pass.extent.height as f32,
                width: pass.extent.width as f32,
                height: -(pass.extent.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            },
            |viewport| vk::Viewport {
                x: viewport.x,
                y: viewport.y + viewport.height,
                width: viewport.width,
                height: -viewport.height,
                min_depth: viewport.min_depth,
                max_depth: viewport.max_depth,
            },
        );
        let scissor = draw.scissor_rect.map_or(
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: pass.extent.width,
                    height: pass.extent.height,
                },
            },
            |rect| vk::Rect2D {
                offset: vk::Offset2D {
                    x: rect.x as i32,
                    y: rect.y as i32,
                },
                extent: vk::Extent2D {
                    width: rect.width,
                    height: rect.height,
                },
            },
        );
        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
        device.cmd_set_scissor(command_buffer, 0, &[scissor]);
        device.cmd_set_blend_constants(command_buffer, &[0.0, 0.0, 0.0, 0.0]);
        device.cmd_set_stencil_reference(command_buffer, vk::StencilFaceFlags::FRONT_AND_BACK, 0);
        bind_render_descriptor_sets(device, command_buffer, pipeline, &descriptor_sets);
    }
    bind_subpass_vertex_buffers(device, command_buffer, draw)?;
    encode_subpass_draw_call(device, command_buffer, draw)?;
    Ok(SubpassDrawTemps {
        descriptor_pool,
        image_views,
    })
}

#[cfg(feature = "tiled")]
fn update_subpass_descriptor_sets(
    device: &ash::Device,
    pipeline: &VulkanRenderPipeline,
    pass: &HalSubpassRenderPassCommand,
    draw: &HalSubpassDraw,
    descriptor_sets: &[vk::DescriptorSet],
    attachment_views: &[vk::ImageView],
) -> Result<Vec<vk::ImageView>, HalError> {
    if pipeline.inner.descriptor_bindings.is_empty() {
        return Ok(Vec::new());
    }
    let mut buffer_infos = Vec::new();
    let mut image_infos = Vec::new();
    let mut image_views = Vec::new();
    let mut write_specs = Vec::new();
    let result = (|| {
        {
            let mut scratch = DescriptorUpdateScratch {
                device,
                buffer_infos: &mut buffer_infos,
                image_infos: &mut image_infos,
                image_views: &mut image_views,
            };
            for descriptor in &pipeline.inner.descriptor_bindings {
                let info = match descriptor.kind {
                    HalDescriptorBindingKind::InputAttachment { color_slot } => {
                        let subpass_inputs = pass
                            .layout
                            .subpasses
                            .get(draw.subpass_index as usize)
                            .map(|subpass| subpass.input_attachments.as_slice())
                            .unwrap_or(&[]);
                        let input = subpass_inputs
                            .iter()
                            .find(|input| {
                                input.group == descriptor.group
                                    && input.binding == descriptor.binding
                                    && input.source_attachment == color_slot
                            })
                            .ok_or_else(|| {
                                shader_error("subpass input attachment mapping is missing")
                            })?;
                        let view = attachment_views
                            .get(input.source_attachment as usize)
                            .copied()
                            .ok_or_else(|| {
                                shader_error("subpass input attachment view is missing")
                            })?;
                        scratch.image_infos.push(
                            vk::DescriptorImageInfo::default()
                                .image_view(view)
                                .image_layout(vk::ImageLayout::GENERAL),
                        );
                        DescriptorInfo::Image(scratch.image_infos.len() - 1)
                    }
                    _ => descriptor_info(
                        descriptor,
                        &draw.bind_buffers,
                        &draw.bind_textures,
                        &draw.bind_samplers,
                        &mut scratch,
                        "render",
                    )?,
                };
                write_specs.push((
                    info,
                    descriptor.group,
                    descriptor.binding,
                    descriptor_type(descriptor.kind),
                ));
            }
        }
        let writes = write_specs
            .iter()
            .map(|(info, group, binding, descriptor_type)| {
                let group = usize::try_from(*group)
                    .map_err(|_| shader_error("descriptor group index is too large"))?;
                let descriptor_set = descriptor_sets
                    .get(group)
                    .copied()
                    .ok_or_else(|| shader_error("descriptor set is missing"))?;
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(*binding)
                    .descriptor_type(*descriptor_type);
                Ok(match info {
                    DescriptorInfo::Buffer(index) => {
                        write.buffer_info(std::slice::from_ref(&buffer_infos[*index]))
                    }
                    DescriptorInfo::Image(index) => {
                        write.image_info(std::slice::from_ref(&image_infos[*index]))
                    }
                })
            })
            .collect::<Result<Vec<_>, HalError>>()?;
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    })();
    if let Err(error) = result {
        destroy_descriptor_image_views(device, &image_views);
        return Err(error);
    }
    Ok(image_views)
}

#[cfg(feature = "tiled")]
fn bind_subpass_vertex_buffers(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    draw: &HalSubpassDraw,
) -> Result<(), HalError> {
    for bound in &draw.vertex_buffers {
        let crate::HalBuffer::Vulkan(buffer) = &bound.buffer else {
            return Err(buffer_error("subpass vertex buffer is not Vulkan-backed"));
        };
        validate_bound_buffer_range(bound)?;
        let buffers = [buffer.inner()?.buffer];
        let offsets = [bound.offset];
        unsafe {
            device.cmd_bind_vertex_buffers(command_buffer, bound.binding, &buffers, &offsets);
        }
    }
    Ok(())
}

#[cfg(feature = "tiled")]
fn encode_subpass_draw_call(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    draw: &HalSubpassDraw,
) -> Result<(), HalError> {
    match draw.draw {
        HalDraw::Direct {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        } => unsafe {
            device.cmd_draw(
                command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
            Ok(())
        },
        _ => Err(shader_error("subpass draw supports only direct draws")),
    }
}

/// Collects attachment views while keeping depth-stencil out of the color layout path.
#[cfg(feature = "tiled")]
fn subpass_attachment_views(
    pass: &HalSubpassRenderPassCommand,
) -> Result<SubpassAttachmentViews, HalError> {
    let mut views = Vec::new();
    let mut persistent_textures = Vec::new();
    for (slot, attachment) in pass.color_attachments.iter().enumerate() {
        let (view, texture) = subpass_attachment_view(&attachment.resource)?;
        views.push(view);
        let slot =
            u32::try_from(slot).map_err(|_| texture_error("subpass color slot is too large"))?;
        persistent_textures.push((slot, texture));
    }
    let depth_stencil_texture = if let Some(depth) = &pass.depth_stencil_attachment {
        let (view, texture) = subpass_attachment_view(&depth.resource)?;
        views.push(view);
        Some(texture)
    } else {
        None
    };
    Ok((views, persistent_textures, depth_stencil_texture))
}

/// Returns whether the bound Vulkan texture is transient (memoryless), used to
/// choose the color attachment's render-pass `finalLayout`. Falls back to
/// non-transient when the resource is not a Vulkan-backed persistent texture,
/// matching the pre-existing default rather than erroring here.
#[cfg(feature = "tiled")]
fn subpass_binding_transient(resource: &HalSubpassAttachmentResource) -> bool {
    match resource {
        HalSubpassAttachmentResource::Persistent { texture, .. } => match texture {
            crate::HalTexture::Vulkan(texture) => texture.transient,
            _ => false,
        },
    }
}

#[cfg(feature = "tiled")]
fn subpass_attachment_view(
    resource: &HalSubpassAttachmentResource,
) -> Result<(vk::ImageView, VulkanTexture), HalError> {
    match resource {
        HalSubpassAttachmentResource::Persistent { texture, .. } => {
            let crate::HalTexture::Vulkan(texture) = texture else {
                return Err(texture_error("subpass attachment is not Vulkan-backed"));
            };
            Ok((texture.inner()?.view, texture.clone()))
        }
    }
}

#[cfg(feature = "tiled")]
fn create_subpass_render_pass(
    device: &ash::Device,
    pass: &HalSubpassRenderPassCommand,
) -> Result<vk::RenderPass, HalError> {
    let mut attachments = Vec::new();
    for (index, layout) in pass.layout.color_attachments.iter().enumerate() {
        let binding = pass
            .color_attachments
            .get(index)
            .ok_or_else(|| shader_error("subpass color attachment binding missing"))?;
        let (format, _) = map_texture_format(layout.format)?;
        let used_as_input = pass.layout.subpasses.iter().any(|subpass| {
            subpass
                .input_attachments
                .iter()
                .any(|input| input.source_attachment as usize == index)
        });
        let transient = subpass_binding_transient(&binding.resource);
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk_sample_count(layout.sample_count)?)
                .load_op(vk_load_op(binding.load_op))
                .store_op(vk_store_op(binding.store))
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(render_color_attachment_layout(used_as_input))
                .final_layout(subpass_color_final_layout(transient)),
        );
    }
    if let Some(layout) = pass.layout.depth_stencil_attachment {
        let binding = pass
            .depth_stencil_attachment
            .as_ref()
            .ok_or_else(|| shader_error("subpass depth-stencil attachment binding missing"))?;
        attachments.push(vk_depth_stencil_attachment_description(layout, binding)?);
    }
    create_subpass_render_pass_with_attachments(device, &pass.layout, &attachments)
}

#[cfg(feature = "tiled")]
pub(super) fn create_subpass_render_pass_for_layout(
    device: &ash::Device,
    layout: &HalSubpassPassLayout,
) -> Result<vk::RenderPass, HalError> {
    let mut attachments = Vec::new();
    for (index, attachment) in layout.color_attachments.iter().enumerate() {
        let (format, _) = map_texture_format(attachment.format)?;
        let used_as_input = layout.subpasses.iter().any(|subpass| {
            subpass
                .input_attachments
                .iter()
                .any(|input| input.source_attachment as usize == index)
        });
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk_sample_count(attachment.sample_count)?)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(render_color_attachment_layout(used_as_input))
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        );
    }
    if let Some(attachment) = layout.depth_stencil_attachment {
        let (format, _) = map_texture_format(attachment.format)?;
        let has_depth = format_has_depth_aspect(attachment.format);
        let has_stencil = format_has_stencil_aspect(attachment.format);
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk_sample_count(attachment.sample_count)?)
                .load_op(if has_depth {
                    vk::AttachmentLoadOp::CLEAR
                } else {
                    vk::AttachmentLoadOp::DONT_CARE
                })
                .store_op(if has_depth {
                    vk::AttachmentStoreOp::STORE
                } else {
                    vk::AttachmentStoreOp::DONT_CARE
                })
                .stencil_load_op(if has_stencil {
                    vk::AttachmentLoadOp::CLEAR
                } else {
                    vk::AttachmentLoadOp::DONT_CARE
                })
                .stencil_store_op(if has_stencil {
                    vk::AttachmentStoreOp::STORE
                } else {
                    vk::AttachmentStoreOp::DONT_CARE
                })
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
    }
    create_subpass_render_pass_with_attachments(device, layout, &attachments)
}

#[cfg(feature = "tiled")]
fn create_subpass_render_pass_with_attachments(
    device: &ash::Device,
    layout: &HalSubpassPassLayout,
    attachments: &[vk::AttachmentDescription],
) -> Result<vk::RenderPass, HalError> {
    let depth_index = u32::try_from(layout.color_attachments.len())
        .map_err(|_| shader_error("subpass depth attachment index is too large"))?;
    let color_refs = layout
        .subpasses
        .iter()
        .map(|subpass| {
            let max_written_slot = subpass.color_attachment_indices.iter().copied().max();
            let max_input_slot = subpass
                .input_attachments
                .iter()
                .map(|input| input.source_attachment)
                .filter(|&source_attachment| source_attachment != u32::MAX)
                .max();
            let Some(max_color_slot) = max_written_slot.max(max_input_slot) else {
                return Vec::new();
            };
            (0..=max_color_slot)
                .map(|attachment| {
                    if subpass.color_attachment_indices.contains(&attachment) {
                        let layout = if subpass
                            .input_attachments
                            .iter()
                            .any(|input| input.source_attachment == attachment)
                        {
                            vk::ImageLayout::GENERAL
                        } else {
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        };
                        vk::AttachmentReference::default()
                            .attachment(attachment)
                            .layout(layout)
                    } else {
                        vk::AttachmentReference::default()
                            .attachment(vk::ATTACHMENT_UNUSED)
                            .layout(vk::ImageLayout::UNDEFINED)
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let input_refs = layout
        .subpasses
        .iter()
        .map(|subpass| {
            subpass
                .input_attachments
                .iter()
                .map(|input| {
                    vk::AttachmentReference::default()
                        .attachment(if input.source_attachment == u32::MAX {
                            depth_index
                        } else {
                            input.source_attachment
                        })
                        .layout(if input.source_attachment == u32::MAX {
                            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        } else {
                            vk::ImageLayout::GENERAL
                        })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let depth_refs = layout
        .subpasses
        .iter()
        .map(|subpass| {
            subpass.uses_depth_stencil.then(|| {
                vk::AttachmentReference::default()
                    .attachment(depth_index)
                    .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            })
        })
        .collect::<Vec<_>>();
    let mut subpasses = Vec::new();
    for index in 0..layout.subpasses.len() {
        let mut description = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs[index])
            .input_attachments(&input_refs[index]);
        if let Some(depth_ref) = depth_refs[index].as_ref() {
            description = description.depth_stencil_attachment(depth_ref);
        }
        subpasses.push(description);
    }
    let dependencies = subpass_dependencies(layout);
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    unsafe { device.create_render_pass(&render_pass_info, None) }
        .map_err(|_| shader_error("subpass render pass creation failed"))
}

#[cfg(feature = "tiled")]
fn subpass_dependencies(layout: &HalSubpassPassLayout) -> Vec<vk::SubpassDependency> {
    let mut dependencies = vec![vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(render_attachment_stage_flags())
        .dst_stage_mask(render_attachment_stage_flags())
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(render_attachment_access_flags())];
    dependencies.extend(layout.dependencies.iter().map(|dependency| {
        let (src_stage, src_access, dst_stage, dst_access) = match dependency.dependency_type {
            HalSubpassDependencyType::ColorToInput => (
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::INPUT_ATTACHMENT_READ,
            ),
            HalSubpassDependencyType::DepthToInput => (
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::INPUT_ATTACHMENT_READ,
            ),
            HalSubpassDependencyType::ColorDepthToInput => (
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::INPUT_ATTACHMENT_READ,
            ),
        };
        vk::SubpassDependency::default()
            .src_subpass(dependency.src_subpass)
            .dst_subpass(dependency.dst_subpass)
            .src_stage_mask(src_stage)
            .dst_stage_mask(dst_stage)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .dependency_flags(if dependency.by_region {
                vk::DependencyFlags::BY_REGION
            } else {
                vk::DependencyFlags::empty()
            })
    }));
    dependencies.push(
        vk::SubpassDependency::default()
            .src_subpass(layout.subpasses.len().saturating_sub(1) as u32)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(render_attachment_stage_flags())
            .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .src_access_mask(render_attachment_access_flags())
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
    );
    dependencies
}

#[cfg(feature = "tiled")]
fn subpass_clear_values(pass: &HalSubpassRenderPassCommand) -> Vec<vk::ClearValue> {
    let mut values = pass
        .color_attachments
        .iter()
        .enumerate()
        .map(|(index, attachment)| {
            let format = pass
                .layout
                .color_attachments
                .get(index)
                .map_or(HalTextureFormat::Unsupported, |layout| layout.format);
            vulkan_color_clear_value(format, attachment.clear_color)
        })
        .collect::<Vec<_>>();
    if let Some(depth) = &pass.depth_stencil_attachment {
        values.push(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: depth.depth_clear_value,
                stencil: depth.stencil_clear_value,
            },
        });
    }
    values
}

#[cfg(feature = "tiled")]
fn vk_depth_stencil_attachment_description(
    layout: HalSubpassAttachmentLayout,
    binding: &HalSubpassDepthStencilAttachment,
) -> Result<vk::AttachmentDescription, HalError> {
    let (format, _) = map_texture_format(layout.format)?;
    let has_depth = format_has_depth_aspect(layout.format);
    let has_stencil = format_has_stencil_aspect(layout.format);
    Ok(vk::AttachmentDescription::default()
        .format(format)
        .samples(vk_sample_count(layout.sample_count)?)
        .load_op(if has_depth {
            vk_load_op(binding.depth_load_op)
        } else {
            vk::AttachmentLoadOp::DONT_CARE
        })
        .store_op(if has_depth && binding.depth_store {
            vk::AttachmentStoreOp::STORE
        } else {
            vk::AttachmentStoreOp::DONT_CARE
        })
        .stencil_load_op(if has_stencil {
            vk_load_op(binding.stencil_load_op)
        } else {
            vk::AttachmentLoadOp::DONT_CARE
        })
        .stencil_store_op(if has_stencil && binding.stencil_store {
            vk::AttachmentStoreOp::STORE
        } else {
            vk::AttachmentStoreOp::DONT_CARE
        })
        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL))
}

fn vk_load_op(load_op: HalRenderLoadOp) -> vk::AttachmentLoadOp {
    match load_op {
        HalRenderLoadOp::Load => vk::AttachmentLoadOp::LOAD,
        HalRenderLoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
    }
}

fn vk_store_op(store: bool) -> vk::AttachmentStoreOp {
    if store {
        vk::AttachmentStoreOp::STORE
    } else {
        vk::AttachmentStoreOp::DONT_CARE
    }
}
fn vk_sample_count(sample_count: u32) -> Result<vk::SampleCountFlags, HalError> {
    match sample_count {
        1 => Ok(vk::SampleCountFlags::TYPE_1),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        _ => Err(texture_error("unsupported render pass sample count")),
    }
}

pub(super) fn encode_render_pass_command_stream(
    device: &VulkanDeviceInner,
    command_buffer: vk::CommandBuffer,
    pass: &HalRenderPassCommandStream,
) -> Result<RenderPassTemps, HalError> {
    let mut bind_textures = Vec::new();
    collect_vulkan_stream_textures(&pass.commands, &mut bind_textures);
    encode_render_pass_impl(device, command_buffer, pass, &bind_textures)
}

fn encode_render_pass_impl(
    device: &VulkanDeviceInner,
    command_buffer: vk::CommandBuffer,
    pass: &HalRenderPassCommandStream,
    bind_textures: &[HalBoundTexture],
) -> Result<RenderPassTemps, HalError> {
    let vk_device = &device.device;
    let color_textures = vulkan_render_color_textures(pass)?;
    let resolve_textures = vulkan_render_resolve_textures(pass)?;
    let depth_stencil_texture = vulkan_render_depth_stencil_texture(pass)?;
    if !color_textures.iter().any(Option::is_some) && depth_stencil_texture.is_none() {
        return Err(shader_error("Vulkan render pass requires an attachment"));
    }
    let query_set = vulkan_stream_query_set(pass)?;
    if let Some(query_set) = query_set {
        for query_index in vulkan_stream_query_indices(&pass.commands) {
            query_set.validate_query(query_index)?;
            unsafe {
                vk_device.cmd_reset_query_pool(command_buffer, query_set.pool(), query_index, 1);
            }
        }
    }
    let mut attachments = Vec::new();
    for (slot, texture) in color_textures
        .iter()
        .enumerate()
        .filter_map(|(slot, texture)| texture.map(|texture| (slot, texture)))
    {
        let framebuffer_fetch = pass
            .framebuffer_fetch_color_slots
            .iter()
            .any(|&fetch_slot| usize::try_from(fetch_slot).ok() == Some(slot));
        let (layout, layout_id) = if framebuffer_fetch {
            (vk::ImageLayout::GENERAL, IMAGE_LAYOUT_GENERAL)
        } else {
            (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                IMAGE_LAYOUT_COLOR_ATTACHMENT,
            )
        };
        let target = pass
            .color_targets
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| texture_error("color attachment target is missing"))?;
        let range = attachment_subresource_range_of(
            texture.dimension,
            target.mip_level,
            target.array_layer,
        );
        attachments.push((texture.inner()?.image, range));
        transition_image_range(
            vk_device,
            command_buffer,
            texture.inner()?,
            range,
            layout,
            layout_id,
        )?;
    }
    for (texture, target) in resolve_textures
        .iter()
        .copied()
        .zip(&pass.color_targets)
        .filter_map(|(texture, target)| texture.zip(target.as_ref()))
    {
        let range = attachment_subresource_range_of(
            texture.dimension,
            target.resolve_mip_level,
            target.resolve_array_layer,
        );
        attachments.push((texture.inner()?.image, range));
        transition_image_range(
            vk_device,
            command_buffer,
            texture.inner()?,
            range,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            IMAGE_LAYOUT_COLOR_ATTACHMENT,
        )?;
    }
    if let (Some(texture), Some(attachment)) =
        (depth_stencil_texture, &pass.depth_stencil_attachment)
    {
        let range = attachment_subresource_range_of(
            texture.dimension,
            attachment.mip_level,
            attachment.array_layer,
        );
        attachments.push((texture.inner()?.image, range));
        transition_image_range(
            vk_device,
            command_buffer,
            texture.inner()?,
            range,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT,
        )?;
    }
    // Barriers precede the pass and exclude only attached subresources.
    transition_sampled_textures(vk_device, command_buffer, bind_textures, &attachments)?;
    // Storage follows sampled so GENERAL wins for a view bound both ways.
    transition_storage_textures(vk_device, command_buffer, bind_textures, &attachments)?;
    let color_formats = render_pass_color_formats(&pass.color_targets);
    let resolve_formats = render_pass_resolve_formats(&pass.color_targets)?;
    let render_pass = create_render_pass_for_targets(
        vk_device,
        &color_formats,
        &resolve_formats,
        &pass.color_targets,
        pass.depth_stencil_attachment.as_ref(),
        &pass.framebuffer_fetch_color_slots,
    )?;
    let temporary_render_pass = Some(render_pass);
    let color_attachments: Vec<_> = color_textures
        .iter()
        .copied()
        .zip(pass.color_targets.iter())
        .map(|(texture, target)| texture.zip(target.as_ref()))
        .collect();
    let resolve_attachments: Vec<_> = resolve_textures
        .iter()
        .zip(pass.color_targets.iter())
        .filter_map(|(texture, target)| {
            texture.and_then(|texture| target.as_ref().map(|target| (texture, target)))
        })
        .collect();
    let depth_stencil_attachment =
        depth_stencil_texture.zip(pass.depth_stencil_attachment.as_ref());
    let framebuffer_resources = create_framebuffer(
        vk_device,
        render_pass,
        &color_attachments,
        &resolve_attachments,
        depth_stencil_attachment,
    )?;
    let framebuffer = framebuffer_resources.framebuffer;
    let image_views = framebuffer_resources.image_views;
    let color_attachment_views = framebuffer_resources.color_attachment_views;
    let clear_values = render_pass_clear_values(pass);
    let (width, height) =
        render_pass_extent_from_targets(&color_attachments, depth_stencil_attachment)?;
    let render_area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D { width, height },
    };
    let begin_info = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);
    unsafe {
        vk_device.cmd_begin_render_pass(command_buffer, &begin_info, vk::SubpassContents::INLINE);
    }
    let mut stream_temps = VulkanRenderStreamTemps::default();
    let mut state = VulkanRenderStreamState::new(render_area, width, height, query_set);
    encode_vulkan_render_commands(
        device,
        command_buffer,
        &pass.commands,
        &color_attachment_views,
        &pass.color_targets,
        pass.depth_stencil_attachment.as_ref(),
        &mut state,
        &mut stream_temps,
    )?;
    unsafe {
        vk_device.cmd_end_render_pass(command_buffer);
    }
    for (texture, target) in color_attachments.iter().flatten() {
        texture.inner()?.layouts.set(
            attachment_subresource_range_of(
                texture.dimension,
                target.mip_level,
                target.array_layer,
            ),
            IMAGE_LAYOUT_TRANSFER_SRC,
        )?;
    }
    for (texture, target) in &resolve_attachments {
        texture.inner()?.layouts.set(
            attachment_subresource_range_of(
                texture.dimension,
                target.resolve_mip_level,
                target.resolve_array_layer,
            ),
            IMAGE_LAYOUT_TRANSFER_SRC,
        )?;
    }
    for (texture, target) in color_textures
        .iter()
        .copied()
        .zip(pass.color_targets.iter())
        .filter_map(|(texture, target)| texture.zip(target.as_ref()))
        .filter(|(_, target)| !target.store)
    {
        let inner = texture.inner()?;
        let attachment_range = attachment_subresource_range_of(
            texture.dimension,
            target.mip_level,
            target.array_layer,
        );
        transition_image_range(
            vk_device,
            command_buffer,
            inner,
            attachment_range,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            IMAGE_LAYOUT_TRANSFER_DST,
        )?;
        let clear_value = unsafe { vulkan_color_clear_value(target.view_format, [0.0; 4]).color };
        unsafe {
            vk_device.cmd_clear_color_image(
                command_buffer,
                inner.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_value,
                &[color_attachment_subresource_range(texture, target)],
            );
        }
        transition_image_range(
            vk_device,
            command_buffer,
            inner,
            attachment_range,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            IMAGE_LAYOUT_TRANSFER_SRC,
        )?;
    }
    if let (Some(texture), Some(attachment)) =
        (depth_stencil_texture, &pass.depth_stencil_attachment)
    {
        let discarded_aspects = discarded_depth_stencil_aspects(
            attachment.depth_store,
            attachment.stencil_store,
            attachment.format,
        );
        if !discarded_aspects.is_empty() {
            let inner = texture.inner()?;
            let attachment_range = attachment_subresource_range_of(
                texture.dimension,
                attachment.mip_level,
                attachment.array_layer,
            );
            transition_image_range(
                vk_device,
                command_buffer,
                inner,
                attachment_range,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                IMAGE_LAYOUT_TRANSFER_DST,
            )?;
            let mut range = depth_stencil_attachment_subresource_range(attachment);
            range.aspect_mask = discarded_aspects;
            unsafe {
                vk_device.cmd_clear_depth_stencil_image(
                    command_buffer,
                    inner.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                    &[range],
                );
            }
            transition_image_range(
                vk_device,
                command_buffer,
                inner,
                attachment_range,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT,
            )?;
        }
    }
    Ok(RenderPassTemps {
        descriptor_pools: stream_temps.descriptor_pools,
        framebuffer,
        image_views: image_views
            .into_iter()
            .chain(stream_temps.image_views)
            .collect(),
        render_pass: temporary_render_pass,
    })
}

fn collect_vulkan_stream_textures(
    commands: &[HalRenderPassCommand],
    textures: &mut Vec<HalBoundTexture>,
) {
    for command in commands {
        match command {
            HalRenderPassCommand::SetBindGroup {
                textures: command_textures,
                ..
            } => {
                textures.extend(command_textures.iter().cloned());
            }
            HalRenderPassCommand::ExecuteRenderBundle(bundle) => {
                collect_vulkan_stream_textures(&bundle.commands, textures);
            }
            _ => {}
        }
    }
}

fn vulkan_stream_query_set(
    pass: &HalRenderPassCommandStream,
) -> Result<Option<&VulkanQuerySet>, HalError> {
    match &pass.occlusion_query_set {
        Some(HalQuerySet::Vulkan(query_set)) => Ok(Some(query_set)),
        Some(_) => Err(buffer_error("occlusion query set is not Vulkan-backed")),
        None => Ok(None),
    }
}

fn vulkan_stream_query_indices(commands: &[HalRenderPassCommand]) -> Vec<u32> {
    let mut indices = Vec::new();
    for command in commands {
        match command {
            HalRenderPassCommand::BeginOcclusionQuery { index } => indices.push(*index),
            HalRenderPassCommand::ExecuteRenderBundle(bundle) => {
                indices.extend(vulkan_stream_query_indices(&bundle.commands));
            }
            _ => {}
        }
    }
    indices
}

#[derive(Default)]
struct VulkanRenderStreamTemps {
    descriptor_pools: Vec<vk::DescriptorPool>,
    image_views: Vec<vk::ImageView>,
}

struct VulkanRenderStreamState<'a> {
    pipeline: Option<crate::HalRenderPipeline>,
    bind_buffers: Vec<HalBoundBuffer>,
    bind_textures: Vec<HalBoundTexture>,
    bind_samplers: Vec<HalBoundSampler>,
    vertex_buffers: Vec<HalBoundBuffer>,
    index_buffer: Option<HalBoundIndexBuffer>,
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    immediate_data: Vec<u8>,
    query_set: Option<&'a VulkanQuerySet>,
    active_query_index: Option<u32>,
    descriptors_dirty: bool,
    initialized: bool,
}

impl<'a> VulkanRenderStreamState<'a> {
    fn new(
        render_area: vk::Rect2D,
        width: u32,
        height: u32,
        query_set: Option<&'a VulkanQuerySet>,
    ) -> Self {
        Self {
            pipeline: None,
            bind_buffers: Vec::new(),
            bind_textures: Vec::new(),
            bind_samplers: Vec::new(),
            vertex_buffers: Vec::new(),
            index_buffer: None,
            viewport: vk::Viewport {
                x: 0.0,
                y: height as f32,
                width: width as f32,
                height: -(height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            },
            scissor: render_area,
            immediate_data: vec![0; 64],
            query_set,
            active_query_index: None,
            descriptors_dirty: true,
            initialized: false,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_vulkan_render_commands(
    device: &VulkanDeviceInner,
    command_buffer: vk::CommandBuffer,
    commands: &[HalRenderPassCommand],
    color_attachment_views: &[Option<vk::ImageView>],
    _color_targets: &[Option<HalRenderColorTarget>],
    _depth_stencil_attachment: Option<&HalRenderDepthStencilAttachment>,
    state: &mut VulkanRenderStreamState<'_>,
    temps: &mut VulkanRenderStreamTemps,
) -> Result<(), HalError> {
    let vk_device = &device.device;
    if !state.initialized {
        unsafe {
            vk_device.cmd_set_viewport(command_buffer, 0, &[state.viewport]);
            vk_device.cmd_set_scissor(command_buffer, 0, &[state.scissor]);
            vk_device.cmd_set_blend_constants(command_buffer, &[0.0; 4]);
            vk_device.cmd_set_stencil_reference(
                command_buffer,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                0,
            );
        }
        state.initialized = true;
    }
    for command in commands {
        match command {
            HalRenderPassCommand::SetPipeline(pipeline) => {
                let crate::HalRenderPipeline::Vulkan(pipeline) = pipeline else {
                    return Err(shader_error("render pipeline is not Vulkan-backed"));
                };
                unsafe {
                    vk_device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.inner.pipeline,
                    );
                }
                state.pipeline = Some(crate::HalRenderPipeline::Vulkan(pipeline.clone()));
                state.descriptors_dirty = true;
            }
            HalRenderPassCommand::SetBindGroup {
                index,
                buffers,
                textures,
                samplers,
                ..
            } => {
                state.bind_buffers.retain(|binding| binding.group != *index);
                state
                    .bind_textures
                    .retain(|binding| binding.group != *index);
                state
                    .bind_samplers
                    .retain(|binding| binding.group != *index);
                state.bind_buffers.extend(buffers.iter().cloned());
                state.bind_textures.extend(textures.iter().cloned());
                state.bind_samplers.extend(samplers.iter().cloned());
                state.descriptors_dirty = true;
            }
            HalRenderPassCommand::SetVertexBuffer { slot, buffer } => {
                state
                    .vertex_buffers
                    .retain(|binding| binding.binding != *slot);
                if let Some(bound) = buffer {
                    let crate::HalBuffer::Vulkan(buffer) = &bound.buffer else {
                        return Err(buffer_error("vertex buffer is not Vulkan-backed"));
                    };
                    validate_bound_buffer_range(bound)?;
                    unsafe {
                        vk_device.cmd_bind_vertex_buffers(
                            command_buffer,
                            *slot,
                            &[buffer.inner()?.buffer],
                            &[bound.offset],
                        );
                    }
                    state.vertex_buffers.push(bound.clone());
                }
            }
            HalRenderPassCommand::SetIndexBuffer(bound) => {
                let crate::HalBuffer::Vulkan(buffer) = &bound.buffer else {
                    return Err(buffer_error("render index buffer is not Vulkan-backed"));
                };
                buffer.validate_range(bound.offset, bound.size)?;
                unsafe {
                    vk_device.cmd_bind_index_buffer(
                        command_buffer,
                        buffer.inner()?.buffer,
                        bound.offset,
                        vk_index_type(bound.format),
                    );
                }
                state.index_buffer = Some(bound.clone());
            }
            HalRenderPassCommand::SetViewport(viewport) => {
                state.viewport = vk::Viewport {
                    x: viewport.x,
                    y: viewport.y + viewport.height,
                    width: viewport.width,
                    height: -viewport.height,
                    min_depth: viewport.min_depth,
                    max_depth: viewport.max_depth,
                };
                unsafe {
                    vk_device.cmd_set_viewport(command_buffer, 0, &[state.viewport]);
                }
            }
            HalRenderPassCommand::SetScissorRect(rect) => {
                state.scissor = vk::Rect2D {
                    offset: vk::Offset2D {
                        x: rect.x as i32,
                        y: rect.y as i32,
                    },
                    extent: vk::Extent2D {
                        width: rect.width,
                        height: rect.height,
                    },
                };
                unsafe {
                    vk_device.cmd_set_scissor(command_buffer, 0, &[state.scissor]);
                }
            }
            HalRenderPassCommand::SetBlendConstant(color) => unsafe {
                vk_device.cmd_set_blend_constants(command_buffer, color);
            },
            HalRenderPassCommand::SetStencilReference(reference) => unsafe {
                vk_device.cmd_set_stencil_reference(
                    command_buffer,
                    vk::StencilFaceFlags::FRONT_AND_BACK,
                    *reference,
                );
            },
            HalRenderPassCommand::SetImmediates { offset, data } => {
                let start = usize::try_from(*offset)
                    .map_err(|_| buffer_error("render immediates offset exceeds usize"))?;
                let end = start
                    .checked_add(data.len())
                    .ok_or_else(|| buffer_error("render immediates range overflows"))?;
                state
                    .immediate_data
                    .get_mut(start..end)
                    .ok_or_else(|| buffer_error("render immediates range exceeds scratch"))?
                    .copy_from_slice(data);
            }
            HalRenderPassCommand::BeginOcclusionQuery { index } => {
                let query_set = state
                    .query_set
                    .ok_or_else(|| buffer_error("active occlusion query has no query set"))?;
                unsafe {
                    vk_device.cmd_begin_query(
                        command_buffer,
                        query_set.pool(),
                        *index,
                        if device.occlusion_query_precise {
                            vk::QueryControlFlags::PRECISE
                        } else {
                            vk::QueryControlFlags::empty()
                        },
                    );
                }
                state.active_query_index = Some(*index);
            }
            HalRenderPassCommand::EndOcclusionQuery => {
                let query_set = state
                    .query_set
                    .ok_or_else(|| buffer_error("active occlusion query has no query set"))?;
                let Some(index) = state.active_query_index.take() else {
                    return Err(buffer_error("render occlusion query end has no begin"));
                };
                unsafe {
                    vk_device.cmd_end_query(command_buffer, query_set.pool(), index);
                }
            }
            HalRenderPassCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => {
                prepare_vulkan_render_draw(
                    vk_device,
                    command_buffer,
                    color_attachment_views,
                    state,
                    temps,
                )?;
                unsafe {
                    vk_device.cmd_draw(
                        command_buffer,
                        *vertex_count,
                        *instance_count,
                        *first_vertex,
                        *first_instance,
                    );
                }
            }
            HalRenderPassCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => {
                if state.index_buffer.is_none() {
                    return Err(buffer_error("render index buffer is missing"));
                }
                prepare_vulkan_render_draw(
                    vk_device,
                    command_buffer,
                    color_attachment_views,
                    state,
                    temps,
                )?;
                unsafe {
                    vk_device.cmd_draw_indexed(
                        command_buffer,
                        *index_count,
                        *instance_count,
                        *first_index,
                        *base_vertex,
                        *first_instance,
                    );
                }
            }
            HalRenderPassCommand::DrawIndirect { indirect_buffer } => {
                prepare_vulkan_render_draw(
                    vk_device,
                    command_buffer,
                    color_attachment_views,
                    state,
                    temps,
                )?;
                let crate::HalBuffer::Vulkan(buffer) = &indirect_buffer.buffer else {
                    return Err(buffer_error("render indirect buffer is not Vulkan-backed"));
                };
                unsafe {
                    vk_device.cmd_draw_indirect(
                        command_buffer,
                        buffer.inner()?.buffer,
                        indirect_buffer.offset,
                        1,
                        16,
                    );
                }
            }
            HalRenderPassCommand::DrawIndexedIndirect { indirect_buffer } => {
                if state.index_buffer.is_none() {
                    return Err(buffer_error("render index buffer is missing"));
                }
                prepare_vulkan_render_draw(
                    vk_device,
                    command_buffer,
                    color_attachment_views,
                    state,
                    temps,
                )?;
                let crate::HalBuffer::Vulkan(buffer) = &indirect_buffer.buffer else {
                    return Err(buffer_error("render indirect buffer is not Vulkan-backed"));
                };
                unsafe {
                    vk_device.cmd_draw_indexed_indirect(
                        command_buffer,
                        buffer.inner()?.buffer,
                        indirect_buffer.offset,
                        1,
                        20,
                    );
                }
            }
            HalRenderPassCommand::ExecuteRenderBundle(bundle) => {
                encode_vulkan_render_commands(
                    device,
                    command_buffer,
                    &bundle.commands,
                    color_attachment_views,
                    _color_targets,
                    _depth_stencil_attachment,
                    state,
                    temps,
                )?;
                state.pipeline = None;
                state.bind_buffers.clear();
                state.bind_textures.clear();
                state.bind_samplers.clear();
                state.vertex_buffers.clear();
                state.index_buffer = None;
                state.descriptors_dirty = true;
            }
        }
    }
    Ok(())
}

fn prepare_vulkan_render_draw(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    color_attachment_views: &[Option<vk::ImageView>],
    state: &mut VulkanRenderStreamState<'_>,
    temps: &mut VulkanRenderStreamTemps,
) -> Result<(), HalError> {
    let Some(crate::HalRenderPipeline::Vulkan(pipeline)) = &state.pipeline else {
        return Err(shader_error("render draw has no Vulkan pipeline"));
    };
    if state.descriptors_dirty {
        let descriptor_pool = create_render_descriptor_pool(device, pipeline)?;
        let descriptor_sets = if let Some(pool) = descriptor_pool {
            match allocate_descriptor_sets(device, pool, &pipeline.inner.descriptor_set_layouts) {
                Ok(sets) => sets,
                Err(error) => {
                    unsafe {
                        device.destroy_descriptor_pool(pool, None);
                    }
                    return Err(error);
                }
            }
        } else {
            Vec::new()
        };
        let image_views = match update_render_descriptor_sets(
            device,
            pipeline,
            &state.bind_buffers,
            &state.bind_textures,
            &state.bind_samplers,
            color_attachment_views,
            &descriptor_sets,
        ) {
            Ok(views) => views,
            Err(error) => {
                if let Some(pool) = descriptor_pool {
                    unsafe {
                        device.destroy_descriptor_pool(pool, None);
                    }
                }
                return Err(error);
            }
        };
        bind_render_descriptor_sets(device, command_buffer, pipeline, &descriptor_sets);
        if let Some(pool) = descriptor_pool {
            temps.descriptor_pools.push(pool);
        }
        temps.image_views.extend(image_views);
        state.descriptors_dirty = false;
    }
    if let Some(immediates) = pipeline.inner.immediates {
        let block = crate::immediates::compose_immediates_block(
            &state.immediate_data,
            immediates.block_size,
            immediates.depth_range_offset,
            [state.viewport.min_depth, state.viewport.max_depth],
        );
        unsafe {
            device.cmd_push_constants(
                command_buffer,
                pipeline.inner.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &block,
            );
        }
    }
    Ok(())
}

fn vk_index_type(format: HalIndexFormat) -> vk::IndexType {
    match format {
        HalIndexFormat::Uint16 => vk::IndexType::UINT16,
        HalIndexFormat::Uint32 => vk::IndexType::UINT32,
    }
}

fn vulkan_render_color_textures(
    pass: &HalRenderPassCommandStream,
) -> Result<Vec<Option<&VulkanTexture>>, HalError> {
    pass.color_targets
        .iter()
        .map(|target| match target {
            Some(target) => match &target.texture {
                crate::HalTexture::Vulkan(texture) => Ok(Some(texture)),
                _ => Err(texture_error("render target is not Vulkan-backed")),
            },
            None => Ok(None),
        })
        .collect()
}

fn vulkan_render_resolve_textures(
    pass: &HalRenderPassCommandStream,
) -> Result<Vec<Option<&VulkanTexture>>, HalError> {
    pass.color_targets
        .iter()
        .map(|target| {
            target
                .as_ref()
                .and_then(|target| {
                    target.resolve_target.as_ref().map(|texture| match texture {
                        crate::HalTexture::Vulkan(texture) => Ok(texture),
                        _ => Err(texture_error("resolve target is not Vulkan-backed")),
                    })
                })
                .transpose()
        })
        .collect()
}

fn vulkan_render_depth_stencil_texture(
    pass: &HalRenderPassCommandStream,
) -> Result<Option<&VulkanTexture>, HalError> {
    pass.depth_stencil_attachment
        .as_ref()
        .map(|attachment| match &attachment.texture {
            crate::HalTexture::Vulkan(texture) => Ok(texture),
            _ => Err(texture_error(
                "depth-stencil attachment is not Vulkan-backed",
            )),
        })
        .transpose()
}

fn render_pass_color_formats(
    color_targets: &[Option<HalRenderColorTarget>],
) -> Vec<Option<HalTextureFormat>> {
    color_targets
        .iter()
        .map(|target| target.as_ref().map(|target| target.view_format))
        .collect()
}

fn render_pass_resolve_formats(
    color_targets: &[Option<HalRenderColorTarget>],
) -> Result<Vec<Option<HalTextureFormat>>, HalError> {
    color_targets
        .iter()
        .map(|target| {
            let Some(target) = target else {
                return Ok(None);
            };
            if target.resolve_target.is_none() {
                return Ok(None);
            }
            target
                .resolve_view_format
                .map(Some)
                .ok_or_else(|| shader_error("resolve target view format is missing"))
        })
        .collect()
}

fn render_pass_clear_values(pass: &HalRenderPassCommandStream) -> Vec<vk::ClearValue> {
    let mut clear_values = Vec::new();
    for color in &pass.color_targets {
        let Some(color) = color else {
            continue;
        };
        clear_values.push(vulkan_color_clear_value(
            color.view_format,
            color.clear_color,
        ));
    }
    for _ in pass
        .color_targets
        .iter()
        .flatten()
        .filter(|target| target.resolve_target.is_some())
    {
        clear_values.push(vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0; 4] },
        });
    }
    if let Some(depth_stencil) = &pass.depth_stencil_attachment {
        clear_values.push(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: depth_stencil.depth_clear_value,
                stencil: depth_stencil.stencil_clear_value,
            },
        });
    }
    clear_values
}

fn vulkan_color_clear_value(format: HalTextureFormat, color: [f64; 4]) -> vk::ClearValue {
    match format.color_clear_kind() {
        HalColorClearKind::Float => vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [
                    color[0] as f32,
                    color[1] as f32,
                    color[2] as f32,
                    color[3] as f32,
                ],
            },
        },
        HalColorClearKind::Uint => vk::ClearValue {
            color: vk::ClearColorValue {
                uint32: [
                    color[0] as u32,
                    color[1] as u32,
                    color[2] as u32,
                    color[3] as u32,
                ],
            },
        },
        HalColorClearKind::Sint => vk::ClearValue {
            color: vk::ClearColorValue {
                int32: [
                    color[0] as i32,
                    color[1] as i32,
                    color[2] as i32,
                    color[3] as i32,
                ],
            },
        },
    }
}

fn render_pass_extent_from_targets(
    color_attachments: &[Option<(&VulkanTexture, &HalRenderColorTarget)>],
    depth_stencil_attachment: Option<(&VulkanTexture, &HalRenderDepthStencilAttachment)>,
) -> Result<(u32, u32), HalError> {
    if let Some((texture, target)) = color_attachments.iter().flatten().next() {
        return Ok(mip_extent(texture.width, texture.height, target.mip_level));
    }
    if let Some((texture, attachment)) = depth_stencil_attachment {
        return Ok(mip_extent(
            texture.width,
            texture.height,
            attachment.mip_level,
        ));
    }
    Err(shader_error("render pass requires an attachment"))
}

fn mip_extent(width: u32, height: u32, mip_level: u32) -> (u32, u32) {
    (
        width.checked_shr(mip_level).unwrap_or(0).max(1),
        height.checked_shr(mip_level).unwrap_or(0).max(1),
    )
}

fn depth_stencil_aspect_flags(format: HalTextureFormat) -> vk::ImageAspectFlags {
    let mut flags = vk::ImageAspectFlags::empty();
    if format_has_depth_aspect(format) {
        flags |= vk::ImageAspectFlags::DEPTH;
    }
    if format_has_stencil_aspect(format) {
        flags |= vk::ImageAspectFlags::STENCIL;
    }
    flags
}

fn discarded_depth_stencil_aspects(
    depth_store: bool,
    stencil_store: bool,
    format: HalTextureFormat,
) -> vk::ImageAspectFlags {
    let mut flags = vk::ImageAspectFlags::empty();
    if !depth_store {
        flags |= vk::ImageAspectFlags::DEPTH;
    }
    if !stencil_store {
        flags |= vk::ImageAspectFlags::STENCIL;
    }
    flags & depth_stencil_aspect_flags(format)
}

fn copy_format_aspect_flags(format: HalTextureFormat) -> vk::ImageAspectFlags {
    // A whole-format copy covers the format's full aspect mask, which is the
    // same rule as the image's own aspect mask.
    image_aspect_flags(format)
}

fn buffer_texture_copy_aspect_flags(
    format: HalTextureFormat,
    aspect: HalTextureAspect,
) -> vk::ImageAspectFlags {
    match aspect {
        HalTextureAspect::All => copy_format_aspect_flags(format),
        HalTextureAspect::DepthOnly => vk::ImageAspectFlags::DEPTH,
        HalTextureAspect::StencilOnly => vk::ImageAspectFlags::STENCIL,
    }
}

fn create_render_pass_for_targets(
    device: &ash::Device,
    color_formats: &[Option<HalTextureFormat>],
    resolve_formats: &[Option<HalTextureFormat>],
    color_targets: &[Option<HalRenderColorTarget>],
    depth_stencil: Option<&HalRenderDepthStencilAttachment>,
    framebuffer_fetch_color_slots: &[u32],
) -> Result<vk::RenderPass, HalError> {
    if color_formats.len() != color_targets.len() {
        return Err(shader_error("render pass color target count mismatch"));
    }
    if resolve_formats.len() != color_targets.len() {
        return Err(shader_error("render pass resolve target count mismatch"));
    }
    if !color_targets.iter().any(Option::is_some) && depth_stencil.is_none() {
        return Err(shader_error("render pass requires an attachment"));
    }
    let mut attachments = Vec::new();
    let mut color_references = Vec::new();
    for (slot, (color_format, color_target)) in
        color_formats.iter().copied().zip(color_targets).enumerate()
    {
        let (Some(color_format), Some(color_target)) = (color_format, color_target) else {
            color_references.push(
                vk::AttachmentReference::default()
                    .attachment(vk::ATTACHMENT_UNUSED)
                    .layout(vk::ImageLayout::UNDEFINED),
            );
            continue;
        };
        let color_slot =
            u32::try_from(slot).map_err(|_| shader_error("color attachment slot is too large"))?;
        let framebuffer_fetch = framebuffer_fetch_color_slots.contains(&color_slot);
        let index = u32::try_from(attachments.len())
            .map_err(|_| shader_error("color attachment index is too large"))?;
        attachments.push(vk_color_attachment_description(
            color_format,
            color_target,
            framebuffer_fetch,
        )?);
        color_references.push(
            vk::AttachmentReference::default()
                .attachment(index)
                .layout(render_color_attachment_layout(framebuffer_fetch)),
        );
    }
    let color_target_present = color_targets
        .iter()
        .map(Option::is_some)
        .collect::<Vec<_>>();
    let input_references = input_attachment_references(
        framebuffer_fetch_color_slots,
        &color_target_present,
        color_references.as_slice(),
    )?;
    let mut resolve_references = Vec::new();
    for (resolve_format, color_target) in resolve_formats.iter().copied().zip(color_targets) {
        if let (Some(resolve_format), Some(color_target)) = (resolve_format, color_target) {
            let index = u32::try_from(attachments.len())
                .map_err(|_| shader_error("resolve attachment index is too large"))?;
            attachments.push(vk_resolve_attachment_description(
                resolve_format,
                color_target,
            )?);
            resolve_references.push(
                vk::AttachmentReference::default()
                    .attachment(index)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            );
        } else {
            resolve_references.push(
                vk::AttachmentReference::default()
                    .attachment(vk::ATTACHMENT_UNUSED)
                    .layout(vk::ImageLayout::UNDEFINED),
            );
        }
    }
    let depth_reference = if let Some(depth_stencil) = depth_stencil {
        let index = u32::try_from(attachments.len())
            .map_err(|_| shader_error("depth attachment index is too large"))?;
        attachments.push(vk_render_depth_stencil_attachment_description(
            depth_stencil,
        )?);
        Some(
            vk::AttachmentReference::default()
                .attachment(index)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        )
    } else {
        None
    };
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_references);
    let subpass = if input_references.is_empty() {
        subpass
    } else {
        subpass.input_attachments(&input_references)
    };
    let subpass = if resolve_references
        .iter()
        .any(|reference| reference.attachment != vk::ATTACHMENT_UNUSED)
    {
        subpass.resolve_attachments(&resolve_references)
    } else {
        subpass
    };
    let subpass = if let Some(depth_reference) = depth_reference.as_ref() {
        subpass.depth_stencil_attachment(depth_reference)
    } else {
        subpass
    };
    let subpasses = [subpass];
    let dependencies = render_pass_dependencies(!input_references.is_empty());
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    unsafe { device.create_render_pass(&render_pass_info, None) }
        .map_err(|_| shader_error("render pass creation failed"))
}

fn vk_color_attachment_description(
    format: HalTextureFormat,
    target: &HalRenderColorTarget,
    framebuffer_fetch: bool,
) -> Result<vk::AttachmentDescription, HalError> {
    let (format, _) = map_texture_format(format)?;
    let crate::HalTexture::Vulkan(texture) = &target.texture else {
        return Err(texture_error("render target is not Vulkan-backed"));
    };
    Ok(vk::AttachmentDescription::default()
        .format(format)
        .samples(vk_sample_count(texture.sample_count)?)
        .load_op(vk_load_op(target.load_op))
        .store_op(vk_store_op(target.store))
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(if framebuffer_fetch {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        })
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL))
}

pub(super) fn input_attachment_references(
    framebuffer_fetch_color_slots: &[u32],
    color_target_present: &[bool],
    color_references: &[vk::AttachmentReference],
) -> Result<Vec<vk::AttachmentReference>, HalError> {
    let mut max_slot = None;
    for &slot in framebuffer_fetch_color_slots {
        let index = usize::try_from(slot)
            .map_err(|_| shader_error("input attachment slot is too large"))?;
        if !color_target_present.get(index).copied().unwrap_or(false) {
            return Err(shader_error("input attachment color target is missing"));
        }
        max_slot = Some(max_slot.map_or(slot, |max: u32| max.max(slot)));
    }
    let Some(max_slot) = max_slot else {
        return Ok(Vec::new());
    };
    let len = usize::try_from(max_slot)
        .ok()
        .and_then(|slot| slot.checked_add(1))
        .ok_or_else(|| shader_error("input attachment slot is too large"))?;
    let mut refs = vec![
        vk::AttachmentReference::default()
            .attachment(vk::ATTACHMENT_UNUSED)
            .layout(vk::ImageLayout::UNDEFINED);
        len
    ];
    for &slot in framebuffer_fetch_color_slots {
        let index = usize::try_from(slot)
            .map_err(|_| shader_error("input attachment slot is too large"))?;
        let color_ref = color_references
            .get(index)
            .ok_or_else(|| shader_error("input attachment color reference is missing"))?;
        refs[index] = vk::AttachmentReference::default()
            .attachment(color_ref.attachment)
            .layout(vk::ImageLayout::GENERAL);
    }
    Ok(refs)
}

pub(super) fn render_color_attachment_layout(framebuffer_fetch: bool) -> vk::ImageLayout {
    if framebuffer_fetch {
        vk::ImageLayout::GENERAL
    } else {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    }
}

/// Returns the render-pass `finalLayout` for a subpass color attachment.
///
/// Non-transient attachments carry `TRANSFER_SRC` usage and end in
/// `TRANSFER_SRC_OPTIMAL` so the post-pass copy needs no barrier. Transient
/// attachments lack `TRANSFER_SRC` usage, so they must end in
/// `COLOR_ATTACHMENT_OPTIMAL` to satisfy
/// VUID-vkCmdBeginRenderPass-initialLayout-00898.
#[cfg(feature = "tiled")]
fn subpass_color_final_layout(transient: bool) -> vk::ImageLayout {
    if transient {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
    }
}

/// Returns the tracked layout state stored after a subpass render pass for a
/// color attachment, consistent with [`subpass_color_final_layout`].
#[cfg(feature = "tiled")]
fn subpass_color_tracked_layout(transient: bool) -> u8 {
    if transient {
        IMAGE_LAYOUT_COLOR_ATTACHMENT
    } else {
        IMAGE_LAYOUT_TRANSFER_SRC
    }
}

pub(super) fn render_pass_dependencies(framebuffer_fetch: bool) -> Vec<vk::SubpassDependency> {
    let dependency_in = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(render_attachment_stage_flags())
        .dst_stage_mask(render_attachment_stage_flags())
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(render_attachment_access_flags());
    let dependency_out = vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(render_attachment_stage_flags())
        .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
        .src_access_mask(render_attachment_access_flags())
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    if framebuffer_fetch {
        vec![
            dependency_in,
            framebuffer_fetch_self_dependency(),
            dependency_out,
        ]
    } else {
        vec![dependency_in, dependency_out]
    }
}

fn render_attachment_stage_flags() -> vk::PipelineStageFlags {
    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
}

fn render_attachment_access_flags() -> vk::AccessFlags {
    vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
}

fn framebuffer_fetch_self_dependency() -> vk::SubpassDependency {
    vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::INPUT_ATTACHMENT_READ)
        .dependency_flags(vk::DependencyFlags::BY_REGION)
}

fn vk_resolve_attachment_description(
    format: HalTextureFormat,
    target: &HalRenderColorTarget,
) -> Result<vk::AttachmentDescription, HalError> {
    let (format, _) = map_texture_format(format)?;
    Ok(vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk_store_op(target.store || target.resolve_target.is_some()))
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL))
}

fn vk_render_depth_stencil_attachment_description(
    attachment: &HalRenderDepthStencilAttachment,
) -> Result<vk::AttachmentDescription, HalError> {
    let (format, _) = map_texture_format(attachment.format)?;
    let crate::HalTexture::Vulkan(texture) = &attachment.texture else {
        return Err(texture_error(
            "depth-stencil attachment is not Vulkan-backed",
        ));
    };
    let has_depth = format_has_depth_aspect(attachment.format);
    let has_stencil = format_has_stencil_aspect(attachment.format);
    Ok(vk::AttachmentDescription::default()
        .format(format)
        .samples(vk_sample_count(texture.sample_count)?)
        .load_op(if has_depth {
            vk_load_op(attachment.depth_load_op)
        } else {
            vk::AttachmentLoadOp::DONT_CARE
        })
        .store_op(if has_depth {
            vk_store_op(attachment.depth_store)
        } else {
            vk::AttachmentStoreOp::DONT_CARE
        })
        .stencil_load_op(if has_stencil {
            vk_load_op(attachment.stencil_load_op)
        } else {
            vk::AttachmentLoadOp::DONT_CARE
        })
        .stencil_store_op(if has_stencil {
            vk_store_op(attachment.stencil_store)
        } else {
            vk::AttachmentStoreOp::DONT_CARE
        })
        .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL))
}

/// Stores framebuffer resources created for one Vulkan render pass.
pub(super) struct FramebufferResources {
    framebuffer: vk::Framebuffer,
    image_views: Vec<vk::ImageView>,
    color_attachment_views: Vec<Option<vk::ImageView>>,
}

/// Creates framebuffer and reports validation errors through the owning device.
pub(super) fn create_framebuffer(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    color_attachments: &[Option<(&VulkanTexture, &HalRenderColorTarget)>],
    resolve_attachments: &[(&VulkanTexture, &HalRenderColorTarget)],
    depth_stencil_attachment: Option<(&VulkanTexture, &HalRenderDepthStencilAttachment)>,
) -> Result<FramebufferResources, HalError> {
    let mut attachments = Vec::new();
    let mut color_attachment_views = Vec::with_capacity(color_attachments.len());
    for (texture, target) in color_attachments.iter().flatten() {
        match create_color_attachment_image_view(device, texture, target) {
            Ok(view) => {
                attachments.push(view);
                color_attachment_views.push(Some(view));
            }
            Err(error) => {
                destroy_image_views(device, &attachments);
                return Err(error);
            }
        }
    }
    if color_attachment_views.len() != color_attachments.len() {
        color_attachment_views.clear();
        let mut color_iter = attachments.iter().copied();
        for attachment in color_attachments {
            color_attachment_views.push(attachment.as_ref().and_then(|_| color_iter.next()));
        }
    }
    for (texture, target) in resolve_attachments {
        match create_resolve_attachment_image_view(device, texture, target) {
            Ok(view) => attachments.push(view),
            Err(error) => {
                destroy_image_views(device, &attachments);
                return Err(error);
            }
        }
    }
    if let Some((texture, attachment)) = depth_stencil_attachment {
        match create_depth_stencil_attachment_image_view(device, texture, attachment) {
            Ok(view) => attachments.push(view),
            Err(error) => {
                destroy_image_views(device, &attachments);
                return Err(error);
            }
        }
    }
    let (width, height) =
        render_pass_extent_from_targets(color_attachments, depth_stencil_attachment)?;
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(width)
        .height(height)
        .layers(1);
    let framebuffer = match unsafe { device.create_framebuffer(&framebuffer_info, None) } {
        Ok(framebuffer) => framebuffer,
        Err(_) => {
            destroy_image_views(device, &attachments);
            return Err(shader_error("framebuffer creation failed"));
        }
    };
    Ok(FramebufferResources {
        framebuffer,
        image_views: attachments,
        color_attachment_views,
    })
}

fn create_color_attachment_image_view(
    device: &ash::Device,
    texture: &VulkanTexture,
    target: &HalRenderColorTarget,
) -> Result<vk::ImageView, HalError> {
    let (format, _) = map_texture_format(target.view_format)?;
    create_attachment_image_view(
        device,
        texture.inner()?,
        format,
        color_attachment_subresource_range(texture, target),
        color_attachment_image_view_usage(),
    )
}

fn create_resolve_attachment_image_view(
    device: &ash::Device,
    texture: &VulkanTexture,
    target: &HalRenderColorTarget,
) -> Result<vk::ImageView, HalError> {
    let view_format = target.resolve_view_format.unwrap_or(texture.format);
    let (format, _) = map_texture_format(view_format)?;
    create_attachment_image_view(
        device,
        texture.inner()?,
        format,
        resolve_attachment_subresource_range(target),
        color_attachment_image_view_usage(),
    )
}

fn create_depth_stencil_attachment_image_view(
    device: &ash::Device,
    texture: &VulkanTexture,
    attachment: &HalRenderDepthStencilAttachment,
) -> Result<vk::ImageView, HalError> {
    let (format, _) = map_texture_format(attachment.format)?;
    create_attachment_image_view(
        device,
        texture.inner()?,
        format,
        depth_stencil_attachment_subresource_range(attachment),
        depth_stencil_attachment_image_view_usage(),
    )
}

fn create_attachment_image_view(
    device: &ash::Device,
    texture: &VulkanTextureInner,
    format: vk::Format,
    subresource_range: vk::ImageSubresourceRange,
    usage: vk::ImageUsageFlags,
) -> Result<vk::ImageView, HalError> {
    // Restrict views to image usage, including swapchain images without
    // INPUT_ATTACHMENT (VUID-VkImageViewCreateInfo-pNext-02662).
    let usage = usage & texture.usage;
    let mut view_usage_info = vk::ImageViewUsageCreateInfo::default().usage(usage);
    let mut view_info = vk::ImageViewCreateInfo::default()
        .image(texture.image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(subresource_range);
    if usage != texture.usage {
        view_info = view_info.push_next(&mut view_usage_info);
    }
    unsafe { device.create_image_view(&view_info, None) }
        .map_err(|_| shader_error("attachment image view creation failed"))
}

fn color_attachment_image_view_usage() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT
}

fn depth_stencil_attachment_image_view_usage() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
}

fn color_attachment_subresource_range(
    texture: &VulkanTexture,
    target: &HalRenderColorTarget,
) -> vk::ImageSubresourceRange {
    let layer = match texture.dimension {
        HalTextureDimension::D3 => target.depth_slice,
        HalTextureDimension::D1 | HalTextureDimension::D2 => target.array_layer,
    };
    attachment_subresource_range(vk::ImageAspectFlags::COLOR, target.mip_level, layer)
}

fn resolve_attachment_subresource_range(
    target: &HalRenderColorTarget,
) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(target.resolve_mip_level)
        .level_count(1)
        .base_array_layer(target.resolve_array_layer)
        .layer_count(1)
}

fn depth_stencil_attachment_subresource_range(
    attachment: &HalRenderDepthStencilAttachment,
) -> vk::ImageSubresourceRange {
    attachment_subresource_range(
        depth_stencil_aspect_flags(attachment.format),
        attachment.mip_level,
        attachment.array_layer,
    )
}

fn attachment_subresource_range(
    aspect: vk::ImageAspectFlags,
    mip_level: u32,
    array_layer: u32,
) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .base_mip_level(mip_level)
        .level_count(1)
        .base_array_layer(array_layer)
        .layer_count(1)
}

fn destroy_image_views(device: &ash::Device, views: &[vk::ImageView]) {
    unsafe {
        for &view in views {
            device.destroy_image_view(view, None);
        }
    }
}

/// Validates buffer texture range and returns a descriptive error on failure.
pub(super) fn validate_buffer_texture_range(
    buffer: &VulkanBuffer,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    crate::format::validate_buffer_texture_range(
        BACKEND,
        buffer.size(),
        copy,
        texture_bytes_per_pixel(copy)?,
    )
}

/// Returns texture bytes per pixel.
pub(super) fn texture_bytes_per_pixel(copy: &HalBufferTextureCopy) -> Result<u32, HalError> {
    let crate::HalTexture::Vulkan(texture) = &copy.texture else {
        return Err(texture_error("texture is not Vulkan-backed"));
    };
    crate::format::aspect_bytes_per_pixel(
        BACKEND,
        copy.format,
        copy.aspect,
        texture.bytes_per_pixel,
    )
}

/// Returns buffer image copy.
pub(super) fn buffer_image_copy(
    copy: &HalBufferTextureCopy,
    texture: &VulkanTexture,
    bytes_per_pixel: u32,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::BufferImageCopy, HalError> {
    // Vulkan derives the slice stride from bufferRowLength and
    // bufferImageHeight.  The tightly-packed shortcut is only valid for a
    // single block-row within a single slice.
    let height_in_blocks = div_ceil_u32(copy.extent.height, texture_block_height(copy));
    let buffer_row_length = if height_in_blocks <= 1 && copy.extent.depth_or_array_layers <= 1 {
        0
    } else {
        let row_length = buffer_row_length(copy.buffer_layout.bytes_per_row, bytes_per_pixel)?;
        row_length
            .checked_mul(texture_block_width(copy))
            .ok_or_else(|| buffer_error("buffer texture row length overflows"))?
    };
    let buffer_image_height = copy
        .buffer_layout
        .rows_per_image
        .checked_mul(texture_block_height(copy))
        .ok_or_else(|| buffer_error("buffer texture image height overflows"))?;
    Ok(vk::BufferImageCopy::default()
        .buffer_offset(copy.buffer_layout.offset)
        .buffer_row_length(buffer_row_length)
        .buffer_image_height(buffer_image_height)
        .image_subresource(texture_copy_subresource_layers(
            aspect,
            texture.dimension,
            copy.mip_level,
            copy.origin.z,
            copy.extent.depth_or_array_layers,
        ))
        .image_offset(texture_copy_offset(
            texture.dimension,
            copy.origin.x,
            copy.origin.y,
            copy.origin.z,
        )?)
        .image_extent(compressed_copy_extent(
            texture,
            copy.mip_level,
            copy.origin,
            copy.extent,
        )?))
}

/// Validates mip level and returns a descriptive error on failure.
pub(super) fn validate_mip_level(texture: &VulkanTexture, mip_level: u32) -> Result<(), HalError> {
    if mip_level >= texture.inner()?.mip_level_count {
        return Err(texture_error("texture mip level exceeds texture mip count"));
    }
    Ok(())
}

fn texture_copy_subresource_layers(
    aspect: vk::ImageAspectFlags,
    dimension: HalTextureDimension,
    mip_level: u32,
    z: u32,
    depth_or_array_layers: u32,
) -> vk::ImageSubresourceLayers {
    match dimension {
        HalTextureDimension::D3 => image_subresource_layers(aspect, mip_level, 0, 1),
        HalTextureDimension::D1 | HalTextureDimension::D2 => {
            image_subresource_layers(aspect, mip_level, z, depth_or_array_layers)
        }
    }
}

fn texture_copy_offset(
    dimension: HalTextureDimension,
    x: u32,
    y: u32,
    z: u32,
) -> Result<vk::Offset3D, HalError> {
    match dimension {
        HalTextureDimension::D3 => to_image_offset(x, y, z),
        HalTextureDimension::D1 | HalTextureDimension::D2 => to_image_offset(x, y, 0),
    }
}

// WebGPU counts complete compressed blocks at a mip edge. Vulkan instead
// requires imageExtent to end at the logical mip size, even for partial blocks.
// Buffer strides still use the original block dimensions in buffer_image_copy.
fn compressed_copy_extent(
    texture: &VulkanTexture,
    mip_level: u32,
    origin: crate::HalOrigin3d,
    extent: HalExtent3d,
) -> Result<vk::Extent3D, HalError> {
    let Some((_, block_width, block_height)) = texture.format.compressed_block_info() else {
        return Ok(texture_copy_extent(texture.dimension, extent));
    };
    let axis_extent = |base: u32, start: u32, count: u32, block: u32| {
        let logical = base
            .checked_shr(mip_level)
            .ok_or_else(|| texture_error("compressed copy mip level is too large"))?
            .max(1);
        let physical = logical
            .div_ceil(block)
            .checked_mul(block)
            .ok_or_else(|| texture_error("compressed copy physical extent overflows"))?;
        let end = start
            .checked_add(count)
            .ok_or_else(|| texture_error("compressed copy range overflows"))?;
        if start >= logical
            || end > physical
            || count == 0
            || !start.is_multiple_of(block)
            || !count.is_multiple_of(block)
        {
            return Err(texture_error(
                "compressed copy range exceeds or misaligns the physical mip",
            ));
        }
        Ok(count.min(logical - start))
    };
    Ok(texture_copy_extent(
        texture.dimension,
        HalExtent3d {
            width: axis_extent(texture.width, origin.x, extent.width, block_width)?,
            height: axis_extent(texture.height, origin.y, extent.height, block_height)?,
            ..extent
        },
    ))
}

fn compressed_copy_needs_temporary_buffer(
    source: vk::Extent3D,
    destination: vk::Extent3D,
    is_compressed: bool,
) -> bool {
    is_compressed && source != destination
}

struct CompressedTemporaryCopy {
    size: u64,
    source: vk::BufferImageCopy,
    destination: vk::BufferImageCopy,
}

fn compressed_temporary_copy(
    source: &VulkanTexture,
    destination: &VulkanTexture,
    copy: &HalTextureCopy,
) -> Result<CompressedTemporaryCopy, HalError> {
    let (block_bytes, block_width, block_height) = source
        .format
        .compressed_block_info()
        .ok_or_else(|| texture_error("temporary compressed copy requires a compressed format"))?;
    let extent = copy.extent;
    let size = u64::from(extent.width / block_width)
        .checked_mul(u64::from(extent.height / block_height))
        .and_then(|size| size.checked_mul(u64::from(extent.depth_or_array_layers)))
        .and_then(|size| size.checked_mul(u64::from(block_bytes)))
        .ok_or_else(|| buffer_error("temporary compressed copy size overflows"))?;
    let region = |texture: &VulkanTexture, mip, origin: crate::HalOrigin3d| {
        Ok(vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(extent.width)
            .buffer_image_height(extent.height)
            .image_subresource(texture_copy_subresource_layers(
                copy_format_aspect_flags(texture.format),
                texture.dimension,
                mip,
                origin.z,
                extent.depth_or_array_layers,
            ))
            .image_offset(texture_copy_offset(
                texture.dimension,
                origin.x,
                origin.y,
                origin.z,
            )?)
            .image_extent(compressed_copy_extent(texture, mip, origin, extent)?))
    };
    Ok(CompressedTemporaryCopy {
        size,
        source: region(source, copy.source_mip_level, copy.source_origin)?,
        destination: region(
            destination,
            copy.destination_mip_level,
            copy.destination_origin,
        )?,
    })
}

fn texture_copy_extent(dimension: HalTextureDimension, extent: HalExtent3d) -> vk::Extent3D {
    match dimension {
        HalTextureDimension::D3 => to_image_extent(extent),
        HalTextureDimension::D1 | HalTextureDimension::D2 => to_image_extent(HalExtent3d {
            depth_or_array_layers: 1,
            ..extent
        }),
    }
}

/// Returns buffer row length.
pub(super) fn buffer_row_length(bytes_per_row: u32, bytes_per_pixel: u32) -> Result<u32, HalError> {
    if bytes_per_row == 0 {
        return Ok(0);
    }
    if bytes_per_pixel == 0 || !bytes_per_row.is_multiple_of(bytes_per_pixel) {
        return Err(buffer_error(
            "buffer texture bytes per row is not texel-aligned",
        ));
    }
    Ok(bytes_per_row / bytes_per_pixel)
}

fn texture_block_info(copy: &HalBufferTextureCopy) -> (u32, u32, u32) {
    copy.format.compressed_block_info().unwrap_or((1, 1, 1))
}

fn texture_block_width(copy: &HalBufferTextureCopy) -> u32 {
    texture_block_info(copy).1
}

fn texture_block_height(copy: &HalBufferTextureCopy) -> u32 {
    texture_block_info(copy).2
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

/// Converts this value into image offset.
pub(super) fn to_image_offset(x: u32, y: u32, z: u32) -> Result<vk::Offset3D, HalError> {
    Ok(vk::Offset3D {
        x: i32::try_from(x).map_err(|_| texture_error("texture x offset is too large"))?,
        y: i32::try_from(y).map_err(|_| texture_error("texture y offset is too large"))?,
        z: i32::try_from(z).map_err(|_| texture_error("texture z offset is too large"))?,
    })
}

/// Converts this value into image extent.
pub(super) fn to_image_extent(extent: HalExtent3d) -> vk::Extent3D {
    vk::Extent3D {
        width: extent.width,
        height: extent.height,
        depth: extent.depth_or_array_layers,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn copy_ranges_select_array_layers_and_collapse_3d_slices() {
        for dimension in [HalTextureDimension::D1, HalTextureDimension::D2] {
            assert_eq!(
                copy_subresource_range(dimension, 1, 2, 3),
                SubresourceRange {
                    base_mip_level: 1,
                    mip_level_count: 1,
                    base_array_layer: 2,
                    array_layer_count: 3
                }
            );
        }
        assert_eq!(
            copy_subresource_range(HalTextureDimension::D3, 2, 5, 4),
            SubresourceRange {
                base_mip_level: 2,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1
            }
        );
    }

    #[test]
    fn clear_ranges_select_mip_layers_and_collapse_3d_slices() {
        let texture = VulkanTexture {
            inner: None,
            swapchain: None,
            surface_pending: None,
            dimension: HalTextureDimension::D2,
            width: 4,
            height: 4,
            depth_or_array_layers: 8,
            sample_count: 1,
            bytes_per_pixel: 4,
            format: HalTextureFormat::Rgba8Unorm,
            transient: false,
        };
        let clear = HalTextureClear {
            texture: HalTexture::Vulkan(texture),
            format: HalTextureFormat::Rgba8Unorm,
            aspect: HalTextureAspect::All,
            mip_level: 2,
            base_array_layer: 3,
            array_layer_count: 4,
        };
        for dimension in [HalTextureDimension::D1, HalTextureDimension::D2] {
            assert_eq!(
                clear_subresource_range(dimension, &clear),
                SubresourceRange {
                    base_mip_level: 2,
                    mip_level_count: 1,
                    base_array_layer: 3,
                    array_layer_count: 4
                }
            );
        }
        assert_eq!(
            clear_subresource_range(HalTextureDimension::D3, &clear),
            SubresourceRange {
                base_mip_level: 2,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1
            }
        );
    }

    #[test]
    fn union_ranges_bounds_both_mips_and_layers() {
        let a = SubresourceRange {
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 2,
        };
        let b = SubresourceRange {
            base_mip_level: 1,
            mip_level_count: 1,
            base_array_layer: 1,
            array_layer_count: 2,
        };
        let expected = SubresourceRange {
            base_mip_level: 0,
            mip_level_count: 2,
            base_array_layer: 0,
            array_layer_count: 3,
        };
        assert_eq!(union_subresource_range(a, b), expected);
        assert_eq!(union_subresource_range(b, a), expected);
        assert_eq!(union_subresource_range(a, a), a);
    }

    #[test]
    fn combined_depth_stencil_copy_aspects_preserve_whole_image_barrier_aspects() {
        for format in [
            HalTextureFormat::Depth24PlusStencil8,
            HalTextureFormat::Depth32FloatStencil8,
        ] {
            let barrier_aspects = image_aspect_flags(format);
            assert_eq!(
                barrier_aspects,
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            );
            for (aspect, expected) in [
                (HalTextureAspect::DepthOnly, vk::ImageAspectFlags::DEPTH),
                (HalTextureAspect::StencilOnly, vk::ImageAspectFlags::STENCIL),
            ] {
                let copy_aspects = buffer_texture_copy_aspect_flags(format, aspect);
                assert_eq!(copy_aspects, expected);
                assert_ne!(copy_aspects, barrier_aspects);
                let runs = [LayoutRun {
                    range: copy_subresource_range(HalTextureDimension::D2, 1, 2, 3),
                    old_state: IMAGE_LAYOUT_TRANSFER_DST,
                }];
                let (barriers, _) = layout_barriers(
                    vk::Image::null(),
                    barrier_aspects,
                    &runs,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                assert_eq!(barriers.len(), 1);
                assert_eq!(barriers[0].subresource_range.aspect_mask, barrier_aspects);
            }
        }
    }

    #[test]
    fn query_resolve_shader_barrier_is_timestamp_only() {
        assert!(super::query_resolve_needs_shader_barrier(
            crate::HalQueryKind::Timestamp
        ));
        assert!(!super::query_resolve_needs_shader_barrier(
            crate::HalQueryKind::Occlusion
        ));
    }

    #[cfg(feature = "vulkan")]
    use super::super::test_helpers::{
        compute_spirv, sampler_descriptor, texture_descriptor, vulkan_device,
    };
    use super::*;
    use crate::{noop, HalBufferTextureLayout, HalOrigin3d, HalTextureDescriptor, HalTextureUsage};
    #[cfg(feature = "vulkan")]
    use crate::{
        HalBoundExternalTexture, HalBoundIndexBuffer, HalBoundIndirectBuffer, HalBoundSampler,
        HalBoundTexture, HalBuffer, HalBufferUsage, HalComputeDispatch, HalComputePass,
        HalComputePipeline, HalIndexFormat, HalRenderPassCommand, HalRenderPassCommandStream,
        HalSampler, HalShaderSource, HalTexture, HalTextureComponentSwizzle,
        HalTextureViewDimension,
    };
    #[cfg(feature = "tiled")]
    use crate::{
        HalSubpassAttachmentLayout, HalSubpassAttachmentResource, HalSubpassColorAttachment,
        HalSubpassDependency, HalSubpassDepthStencilAttachment, HalSubpassInputAttachment,
        HalSubpassLayout, HalSubpassPassLayout,
    };

    #[test]
    fn compressed_copy_extent_clamps_mip_edges_and_preserves_depth_and_pitches() {
        let mut texture =
            dummy_vulkan_texture(HalTextureDimension::D3, HalTextureFormat::Bc1RgbaUnorm);
        texture.width = 12;
        texture.height = 12;
        let origin = HalOrigin3d { x: 0, y: 0, z: 1 };
        for (mip, physical, logical) in [(0, 12, 12), (1, 8, 6), (2, 4, 3), (3, 4, 1)] {
            let extent = HalExtent3d {
                width: physical,
                height: physical,
                depth_or_array_layers: 2,
            };
            assert_eq!(
                compressed_copy_extent(&texture, mip, origin, extent).unwrap(),
                vk::Extent3D {
                    width: logical,
                    height: logical,
                    depth: 2
                }
            );
        }
        let edge_origin = HalOrigin3d { x: 4, y: 4, z: 1 };
        let extent = HalExtent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 2,
        };
        assert_eq!(
            compressed_copy_extent(&texture, 1, edge_origin, extent).unwrap(),
            vk::Extent3D {
                width: 2,
                height: 2,
                depth: 2
            }
        );
        let (mut copy, _) = make_copy(256, 3, 8, 8, 2);
        copy.format = texture.format;
        copy.mip_level = 1;
        let region = buffer_image_copy(&copy, &texture, 8, vk::ImageAspectFlags::COLOR).unwrap();
        assert_eq!(
            region.image_extent,
            vk::Extent3D {
                width: 6,
                height: 6,
                depth: 2
            }
        );
        assert_eq!(region.buffer_row_length, 128);
        assert_eq!(region.buffer_image_height, 12);
        texture.format = HalTextureFormat::Rgba8Unorm;
        assert_eq!(
            compressed_copy_extent(&texture, 1, edge_origin, extent).unwrap(),
            vk::Extent3D {
                width: 4,
                height: 4,
                depth: 2
            }
        );
    }

    #[test]
    fn compressed_copy_extent_rejects_invalid_ranges_without_panicking() {
        let mut texture =
            dummy_vulkan_texture(HalTextureDimension::D2, HalTextureFormat::Bc1RgbaUnorm);
        let origin = HalOrigin3d { x: 0, y: 0, z: 0 };
        let extent = HalExtent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        };
        assert!(compressed_copy_extent(&texture, 32, origin, extent).is_err());
        assert!(
            compressed_copy_extent(&texture, 0, HalOrigin3d { x: 4, ..origin }, extent).is_err()
        );
        assert!(
            compressed_copy_extent(&texture, 0, HalOrigin3d { x: 1, ..origin }, extent).is_err()
        );
        assert!(
            compressed_copy_extent(&texture, 0, origin, HalExtent3d { width: 8, ..extent })
                .is_err()
        );
        assert!(compressed_copy_extent(
            &texture,
            0,
            HalOrigin3d {
                x: u32::MAX,
                ..origin
            },
            extent
        )
        .is_err());
        texture.width = u32::MAX;
        assert!(compressed_copy_extent(&texture, 0, origin, extent).is_err());
    }

    #[test]
    fn compressed_copy_needs_temporary_buffer_only_for_mismatched_compressed_extents() {
        let source = vk::Extent3D {
            width: 16,
            height: 16,
            depth: 1,
        };
        let destination = vk::Extent3D {
            width: 15,
            height: 15,
            depth: 1,
        };
        assert!(compressed_copy_needs_temporary_buffer(
            source,
            destination,
            true
        ));
        assert!(compressed_copy_needs_temporary_buffer(
            destination,
            source,
            true
        ));
        assert!(!compressed_copy_needs_temporary_buffer(
            source, source, true
        ));
        assert!(!compressed_copy_needs_temporary_buffer(
            source,
            destination,
            false
        ));
    }

    #[test]
    fn compressed_temporary_copy_preserves_blocks_mip_edges_and_layers() {
        for dimension in [HalTextureDimension::D2, HalTextureDimension::D3] {
            for layers in [1, 3] {
                let mut source = dummy_vulkan_texture(dimension, HalTextureFormat::Bc1RgbaUnorm);
                source.width = 16;
                source.height = 16;
                let mut destination = source.clone();
                destination.width = 60;
                destination.height = 60;
                let copy = HalTextureCopy {
                    source: HalTexture::Vulkan(source.clone()),
                    destination: HalTexture::Vulkan(destination.clone()),
                    source_mip_level: 0,
                    destination_mip_level: 2,
                    source_origin: HalOrigin3d { x: 0, y: 0, z: 1 },
                    destination_origin: HalOrigin3d { x: 0, y: 0, z: 2 },
                    extent: HalExtent3d {
                        width: 16,
                        height: 16,
                        depth_or_array_layers: layers,
                    },
                };
                let temporary = compressed_temporary_copy(&source, &destination, &copy).unwrap();
                assert_eq!(temporary.size, 128 * u64::from(layers));
                for (region, logical, mip, z) in [
                    (temporary.source, 16, 0, 1),
                    (temporary.destination, 15, 2, 2),
                ] {
                    assert_eq!(region.buffer_offset, 0);
                    assert_eq!(region.buffer_row_length, 16);
                    assert_eq!(region.buffer_image_height, 16);
                    assert_eq!(region.image_extent.width, logical);
                    assert_eq!(region.image_extent.height, logical);
                    assert_eq!(region.image_subresource.mip_level, mip);
                    if dimension == HalTextureDimension::D3 {
                        assert_eq!(region.image_extent.depth, layers);
                        assert_eq!(region.image_subresource.layer_count, 1);
                        assert_eq!(region.image_subresource.base_array_layer, 0);
                        assert_eq!(region.image_offset.z, z as i32);
                    } else {
                        assert_eq!(region.image_extent.depth, 1);
                        assert_eq!(region.image_subresource.layer_count, layers);
                        assert_eq!(region.image_subresource.base_array_layer, z);
                        assert_eq!(region.image_offset.z, 0);
                    }
                }
                let mut overflow = copy.clone();
                overflow.extent = HalExtent3d {
                    width: u32::MAX - 3,
                    height: u32::MAX - 3,
                    depth_or_array_layers: u32::MAX,
                };
                assert!(compressed_temporary_copy(&source, &destination, &overflow).is_err());
                let mut invalid = copy.clone();
                invalid.extent.width = 15;
                assert!(compressed_temporary_copy(&source, &destination, &invalid).is_err());
            }
        }
    }

    #[test]
    fn texture_copy_layouts_require_general_only_for_shared_subresources() {
        let layers = |mip, base, count| {
            image_subresource_layers(vk::ImageAspectFlags::COLOR, mip, base, count)
        };
        let transfer = (
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        let general = (vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL);
        assert_eq!(
            texture_copy_layouts(false, layers(0, 0, 2), layers(0, 0, 2)),
            transfer
        );
        assert_eq!(
            texture_copy_layouts(true, layers(0, 0, 2), layers(1, 0, 2)),
            transfer
        );
        assert_eq!(
            texture_copy_layouts(true, layers(0, 0, 2), layers(0, 2, 2)),
            transfer
        );
        assert_eq!(
            texture_copy_layouts(true, layers(0, 2, 2), layers(0, 0, 2)),
            transfer
        );
        assert_eq!(
            texture_copy_layouts(true, layers(0, 0, 2), layers(0, 1, 2)),
            general
        );
        assert_eq!(
            texture_copy_layouts(true, layers(0, 1, 2), layers(0, 0, 2)),
            general
        );
        // Widen the endpoint arithmetic so even maximal layer indices cannot panic.
        assert_eq!(
            texture_copy_layouts(true, layers(0, u32::MAX, 1), layers(0, u32::MAX, 1)),
            general
        );
        let slice = |z| {
            texture_copy_subresource_layers(
                vk::ImageAspectFlags::COLOR,
                HalTextureDimension::D3,
                0,
                z,
                1,
            )
        };
        assert_eq!(texture_copy_layouts(true, slice(0), slice(2)), general);
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_color_final_layout_uses_color_attachment_for_transient() {
        assert_eq!(
            subpass_color_final_layout(true),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            subpass_color_final_layout(false),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        );
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_color_tracked_layout_matches_final_layout_choice() {
        assert_eq!(
            subpass_color_tracked_layout(true),
            IMAGE_LAYOUT_COLOR_ATTACHMENT
        );
        assert_eq!(
            subpass_color_tracked_layout(false),
            IMAGE_LAYOUT_TRANSFER_SRC
        );
    }

    /// Builds a pure mip/layer rectangle for exclusion tests.
    fn subresource_range(mip: u32, mips: u32, layer: u32, layers: u32) -> SubresourceRange {
        SubresourceRange {
            base_mip_level: mip,
            mip_level_count: mips,
            base_array_layer: layer,
            array_layer_count: layers,
        }
    }

    /// Attachment ranges select one mip and layer, with 3D slices sharing layer zero.
    #[test]
    fn attachment_subresource_range_of_scopes_dimensions() {
        for dimension in [HalTextureDimension::D1, HalTextureDimension::D2] {
            assert_eq!(
                attachment_subresource_range_of(dimension, 2, 3),
                subresource_range(2, 1, 3, 1)
            );
        }
        assert_eq!(
            attachment_subresource_range_of(HalTextureDimension::D3, 2, 3),
            subresource_range(2, 1, 0, 1)
        );
        assert_eq!(
            attachment_subresource_range_of(HalTextureDimension::D2, 0, 0),
            subresource_range(0, 1, 0, 1)
        );
    }

    /// View ranges preserve both nonzero bases and multi-subresource counts.
    #[test]
    fn bound_view_subresource_range_preserves_view_rectangle() {
        let mut bound = bound_texture(dummy_texture(HalTextureFormat::Rgba8Unorm));
        assert_eq!(
            bound_view_subresource_range(&bound),
            subresource_range(0, 1, 0, 1)
        );
        bound.base_mip_level = 2;
        bound.mip_level_count = 3;
        bound.base_array_layer = 4;
        bound.array_layer_count = 5;
        assert_eq!(
            bound_view_subresource_range(&bound),
            subresource_range(2, 3, 4, 5)
        );
    }

    /// A mip-zero attachment excludes only that mip from a two-mip view.
    #[test]
    fn subtract_subresource_ranges_excludes_attachment_mip() {
        assert_eq!(
            subtract_subresource_ranges(
                subresource_range(0, 2, 0, 1),
                &[subresource_range(0, 1, 0, 1)]
            ),
            vec![subresource_range(1, 1, 0, 1)]
        );
    }

    /// Fully contained views have no remaining transition rectangles.
    #[test]
    fn subtract_subresource_ranges_contained_is_empty() {
        assert!(subtract_subresource_ranges(
            subresource_range(1, 1, 2, 1),
            &[subresource_range(0, 3, 0, 4)]
        )
        .is_empty());
    }

    /// Disjoint and absent exclusions leave the complete view unchanged.
    #[test]
    fn subtract_subresource_ranges_ignores_disjoint_and_absent_exclusions() {
        let bound = subresource_range(1, 2, 2, 3);
        for exclusions in [
            vec![],
            vec![subresource_range(0, 1, 2, 3)],
            vec![subresource_range(1, 2, 8, 1)],
            vec![subresource_range(9, 2, 9, 2)],
        ] {
            assert_eq!(subtract_subresource_ranges(bound, &exclusions), vec![bound]);
        }
    }

    /// Removing an interior layer leaves two non-overlapping layer runs.
    #[test]
    fn subtract_subresource_ranges_splits_layers() {
        assert_eq!(
            subtract_subresource_ranges(
                subresource_range(0, 1, 0, 4),
                &[subresource_range(0, 1, 1, 1)]
            ),
            vec![subresource_range(0, 1, 0, 1), subresource_range(0, 1, 2, 2)]
        );
    }

    /// Overlapping exclusions coalesce surviving identical run lists across mips.
    #[test]
    fn subtract_subresource_ranges_coalesces_multiple_exclusions() {
        assert_eq!(
            subtract_subresource_ranges(
                subresource_range(2, 3, 1, 6),
                &[subresource_range(2, 3, 2, 2), subresource_range(2, 3, 3, 2)]
            ),
            vec![subresource_range(2, 3, 1, 1), subresource_range(2, 3, 5, 2)]
        );
    }

    fn dummy_texture(format: HalTextureFormat) -> HalTexture {
        let device = noop::NoopDevice::new();
        HalTexture::Noop(
            device
                .create_texture(&HalTextureDescriptor {
                    dimension: HalTextureDimension::D2,
                    format,
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                    mip_level_count: 1,
                    sample_count: 1,
                    usage: HalTextureUsage {
                        copy_src: false,
                        copy_dst: false,
                        texture_binding: false,
                        storage_binding: false,
                        render_attachment: true,
                        transient: false,
                    },
                })
                .expect("Noop texture allocation should succeed"),
        )
    }

    fn dummy_vulkan_texture(
        dimension: HalTextureDimension,
        format: HalTextureFormat,
    ) -> VulkanTexture {
        VulkanTexture {
            inner: None,
            swapchain: None,
            surface_pending: None,
            dimension,
            width: 4,
            height: 4,
            depth_or_array_layers: 8,
            sample_count: 1,
            bytes_per_pixel: 4,
            format,
            transient: false,
        }
    }

    #[cfg(feature = "vulkan")]
    fn bound_texture(texture: HalTexture) -> HalBoundTexture {
        HalBoundTexture {
            group: 0,
            binding: 0,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: None,
            texture,
            format: HalTextureFormat::Rgba8Unorm,
            dimension: HalTextureViewDimension::D2,
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
            aspect: HalTextureAspect::All,
            swizzle: HalTextureComponentSwizzle::default(),
            storage_access: None,
        }
    }

    #[cfg(feature = "vulkan")]
    fn bound_sampler(sampler: HalSampler) -> HalBoundSampler {
        HalBoundSampler {
            group: 0,
            binding: 1,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: None,
            sampler,
        }
    }

    #[cfg(feature = "vulkan")]
    fn bound_external_texture(
        plane0: HalTexture,
        plane1: HalTexture,
        params: HalBuffer,
    ) -> HalBoundExternalTexture {
        HalBoundExternalTexture {
            group: 0,
            binding: 2,
            plane0,
            plane1,
            plane0_metal_index: 0,
            plane1_metal_index: 1,
            plane0_vertex_metal_index: None,
            plane1_vertex_metal_index: None,
            plane0_fragment_metal_index: None,
            plane1_fragment_metal_index: None,
            params,
            params_metal_index: 0,
            params_vertex_metal_index: None,
            params_fragment_metal_index: None,
            format: HalTextureFormat::Rgba8Unorm,
            dimension: HalTextureViewDimension::D2,
            params_offset: 0,
            params_size: 16,
        }
    }

    #[cfg(feature = "vulkan")]
    fn retained_buffer_count(
        retained: &[RetainedResource],
        target: &Arc<VulkanBufferInner>,
    ) -> usize {
        retained
            .iter()
            .filter(|resource| {
                matches!(
                    resource,
                    RetainedResource::Buffer { _inner: inner } if Arc::ptr_eq(inner, target)
                )
            })
            .count()
    }

    #[cfg(feature = "vulkan")]
    fn retained_texture_count(
        retained: &[RetainedResource],
        target: &Arc<VulkanTextureInner>,
    ) -> usize {
        retained
            .iter()
            .filter(|resource| {
                matches!(
                    resource,
                    RetainedResource::Texture { _inner: inner } if Arc::ptr_eq(inner, target)
                )
            })
            .count()
    }

    #[cfg(feature = "vulkan")]
    fn retained_sampler_count(
        retained: &[RetainedResource],
        target: &Arc<VulkanSamplerInner>,
    ) -> usize {
        retained
            .iter()
            .filter(|resource| {
                matches!(
                    resource,
                    RetainedResource::Sampler { _inner: inner } if Arc::ptr_eq(inner, target)
                )
            })
            .count()
    }

    #[cfg(feature = "vulkan")]
    fn retained_compute_pipeline_count(
        retained: &[RetainedResource],
        target: &Arc<VulkanComputePipelineInner>,
    ) -> usize {
        retained
            .iter()
            .filter(|resource| {
                matches!(
                    resource,
                    RetainedResource::ComputePipeline { _inner: inner } if Arc::ptr_eq(inner, target)
                )
            })
            .count()
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn collect_retained_resources_covers_render_pass_submit_inputs() {
        let device = vulkan_device();
        let index = device
            .create_buffer(16, HalBufferUsage::default())
            .expect("create index buffer");
        let indirect = device
            .create_buffer(16, HalBufferUsage::default())
            .expect("create indirect buffer");
        let params = device
            .create_buffer(16, HalBufferUsage::default())
            .expect("create external texture params buffer");
        let texture = device
            .create_texture(&texture_descriptor())
            .expect("create bound texture");
        let external_plane0 = device
            .create_texture(&texture_descriptor())
            .expect("create external plane0 texture");
        let external_plane1 = device
            .create_texture(&texture_descriptor())
            .expect("create external plane1 texture");
        let sampler = device.create_sampler(&sampler_descriptor());
        let index_inner = Arc::clone(index.inner.as_ref().expect("index buffer inner"));
        let indirect_inner = Arc::clone(indirect.inner.as_ref().expect("indirect buffer inner"));
        let params_inner = Arc::clone(params.inner.as_ref().expect("params buffer inner"));
        let texture_inner = Arc::clone(texture.inner.as_ref().expect("texture inner"));
        let external_plane0_inner = Arc::clone(
            external_plane0
                .inner
                .as_ref()
                .expect("plane0 texture inner"),
        );
        let external_plane1_inner = Arc::clone(
            external_plane1
                .inner
                .as_ref()
                .expect("plane1 texture inner"),
        );
        let sampler_inner = Arc::clone(sampler._inner.as_ref().expect("sampler inner"));
        let pass = HalRenderPassCommandStream {
            color_targets: Vec::new(),
            framebuffer_fetch_color_slots: Vec::new(),
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            commands: vec![
                HalRenderPassCommand::SetBindGroup {
                    index: 0,
                    buffers: Vec::new(),
                    textures: vec![bound_texture(HalTexture::Vulkan(texture))],
                    samplers: vec![bound_sampler(HalSampler::Vulkan(sampler))],
                    external_textures: vec![bound_external_texture(
                        HalTexture::Vulkan(external_plane0),
                        HalTexture::Vulkan(external_plane1),
                        HalBuffer::Vulkan(params),
                    )],
                },
                HalRenderPassCommand::SetIndexBuffer(HalBoundIndexBuffer {
                    buffer: HalBuffer::Vulkan(index),
                    format: HalIndexFormat::Uint16,
                    offset: 0,
                    size: 16,
                }),
                HalRenderPassCommand::DrawIndexedIndirect {
                    indirect_buffer: HalBoundIndirectBuffer {
                        buffer: HalBuffer::Vulkan(indirect),
                        offset: 0,
                    },
                },
            ],
        };

        let retained = collect_retained_resources(&[HalCopy::RenderPassCommandStream(pass)]);

        assert_eq!(retained_buffer_count(&retained, &index_inner), 1);
        assert_eq!(retained_buffer_count(&retained, &indirect_inner), 1);
        assert_eq!(retained_buffer_count(&retained, &params_inner), 1);
        assert_eq!(retained_texture_count(&retained, &texture_inner), 1);
        assert_eq!(retained_texture_count(&retained, &external_plane0_inner), 1);
        assert_eq!(retained_texture_count(&retained, &external_plane1_inner), 1);
        assert_eq!(retained_sampler_count(&retained, &sampler_inner), 1);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn collect_retained_resources_covers_compute_pass_submit_inputs() {
        let device = vulkan_device();
        let pipeline = device
            .create_compute_pipeline(
                HalShaderSource::SpirV(compute_spirv()),
                "main",
                (1, 1, 1),
                &[],
                0,
            )
            .expect("create compute pipeline");
        let indirect = device
            .create_buffer(16, HalBufferUsage::default())
            .expect("create indirect buffer");
        let params = device
            .create_buffer(16, HalBufferUsage::default())
            .expect("create external texture params buffer");
        let texture = device
            .create_texture(&texture_descriptor())
            .expect("create bound texture");
        let external_plane0 = device
            .create_texture(&texture_descriptor())
            .expect("create external plane0 texture");
        let external_plane1 = device
            .create_texture(&texture_descriptor())
            .expect("create external plane1 texture");
        let sampler = device.create_sampler(&sampler_descriptor());
        let pipeline_inner = Arc::clone(&pipeline.inner);
        let indirect_inner = Arc::clone(indirect.inner.as_ref().expect("indirect buffer inner"));
        let params_inner = Arc::clone(params.inner.as_ref().expect("params buffer inner"));
        let texture_inner = Arc::clone(texture.inner.as_ref().expect("texture inner"));
        let external_plane0_inner = Arc::clone(
            external_plane0
                .inner
                .as_ref()
                .expect("plane0 texture inner"),
        );
        let external_plane1_inner = Arc::clone(
            external_plane1
                .inner
                .as_ref()
                .expect("plane1 texture inner"),
        );
        let sampler_inner = Arc::clone(sampler._inner.as_ref().expect("sampler inner"));
        let pass = HalComputePass {
            pipeline: HalComputePipeline::Vulkan(pipeline),
            bind_buffers: Vec::new(),
            bind_textures: vec![bound_texture(HalTexture::Vulkan(texture))],
            bind_samplers: vec![bound_sampler(HalSampler::Vulkan(sampler))],
            bind_external_textures: vec![bound_external_texture(
                HalTexture::Vulkan(external_plane0),
                HalTexture::Vulkan(external_plane1),
                HalBuffer::Vulkan(params),
            )],
            immediate_data: Vec::new(),
            dispatch: HalComputeDispatch::Indirect {
                buffer: Box::new(HalBoundIndirectBuffer {
                    buffer: HalBuffer::Vulkan(indirect),
                    offset: 0,
                }),
            },
        };

        let retained = collect_retained_resources(&[HalCopy::ComputePass(pass)]);

        assert_eq!(
            retained_compute_pipeline_count(&retained, &pipeline_inner),
            1
        );
        assert_eq!(retained_buffer_count(&retained, &indirect_inner), 1);
        assert_eq!(retained_buffer_count(&retained, &params_inner), 1);
        assert_eq!(retained_texture_count(&retained, &texture_inner), 1);
        assert_eq!(retained_texture_count(&retained, &external_plane0_inner), 1);
        assert_eq!(retained_texture_count(&retained, &external_plane1_inner), 1);
        assert_eq!(retained_sampler_count(&retained, &sampler_inner), 1);
    }

    /// Block 102 R3: the barriers recorded around a compute-pass dispatch
    /// order transfer writes (query resolve, copies, fills) against the
    /// dispatch's storage accesses, and the dispatch's writes against later
    /// transfers, dispatches, draws and the host. Pure flag math, no GPU.
    #[test]
    fn compute_pass_barrier_scopes_order_transfer_and_compute_buffer_hazards() {
        let pre = compute_pass_pre_dispatch_barrier_scopes();
        assert!(pre
            .src_stages
            .contains(vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(pre.src_access.contains(vk::AccessFlags::MEMORY_WRITE));
        assert!(pre
            .dst_stages
            .contains(vk::PipelineStageFlags::COMPUTE_SHADER));
        assert!(pre
            .dst_access
            .contains(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE));
        // The wider buffer-read destination scope (F-106) is kept on top of R3.
        assert!(pre
            .dst_stages
            .contains(buffer_write_read_barrier_dst_stage_mask()));
        assert!(pre
            .dst_access
            .contains(buffer_write_read_barrier_dst_access_mask()));

        let post = compute_pass_post_dispatch_barrier_scopes();
        assert_eq!(post.src_stages, vk::PipelineStageFlags::COMPUTE_SHADER);
        assert_eq!(post.src_access, vk::AccessFlags::SHADER_WRITE);
        assert!(post.dst_stages.contains(
            vk::PipelineStageFlags::TRANSFER
                | vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::HOST
        ));
        assert!(post
            .dst_access
            .contains(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE));
        assert!(post
            .dst_stages
            .contains(buffer_write_read_barrier_dst_stage_mask()));
    }

    /// Block 102 R3: two timestamps written around GPU work resolve to raw
    /// ticks that are both non-zero and strictly increasing.
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_write_timestamp_then_resolve_yields_increasing_ticks() {
        let device = vulkan_device();
        let query_set = device
            .create_query_set(HalQueryKind::Timestamp, 2)
            .expect("create timestamp query set");
        let destination = device
            .create_buffer(16, query_resolve_usage())
            .expect("create resolve destination");
        let readback = device
            .create_buffer(16, readback_usage())
            .expect("create readback buffer");
        let busy = device
            .create_buffer(1024 * 1024, HalBufferUsage::default())
            .expect("create busy-work buffer");

        device
            .queue()
            .submit_copies(&[
                HalCopy::WriteTimestamp(HalWriteTimestamp {
                    query_set: HalQuerySet::Vulkan(query_set.clone()),
                    query_index: 0,
                }),
                // Keep the GPU busy so the two stamps cannot collapse onto the
                // same tick.
                HalCopy::BufferClear(HalBufferClear {
                    buffer: HalBuffer::Vulkan(busy),
                    offset: 0,
                    size: 1024 * 1024,
                }),
                HalCopy::WriteTimestamp(HalWriteTimestamp {
                    query_set: HalQuerySet::Vulkan(query_set.clone()),
                    query_index: 1,
                }),
                HalCopy::ResolveQuerySet(HalResolveQuerySet {
                    query_set: HalQuerySet::Vulkan(query_set),
                    first_query: 0,
                    query_count: 2,
                    written_queries: vec![0, 1],
                    destination: HalBuffer::Vulkan(destination.clone()),
                    destination_offset: 0,
                }),
                HalCopy::Buffer(HalBufferCopy {
                    source: HalBuffer::Vulkan(destination),
                    source_offset: 0,
                    destination: HalBuffer::Vulkan(readback.clone()),
                    destination_offset: 0,
                    size: 16,
                }),
            ])
            .expect("submit timestamp writes and resolve");
        device.queue().wait_idle().expect("wait idle");

        let ticks = read_query_results(&readback, 2);
        assert!(
            ticks[0] > 0,
            "first timestamp should be non-zero: {ticks:?}"
        );
        assert!(
            ticks[1] > ticks[0],
            "second timestamp should be later: {ticks:?}"
        );
    }

    /// Block 102 R3: slots never written in the submission resolve to zero
    /// while the written ones carry their ticks.
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_resolve_timestamp_query_set_zero_fills_unwritten_slots() {
        let device = vulkan_device();
        let query_set = device
            .create_query_set(HalQueryKind::Timestamp, 2)
            .expect("create timestamp query set");
        let destination = device
            .create_buffer(16, query_resolve_usage())
            .expect("create resolve destination");
        let readback = device
            .create_buffer(16, readback_usage())
            .expect("create readback buffer");

        device
            .queue()
            .submit_copies(&[
                HalCopy::WriteTimestamp(HalWriteTimestamp {
                    query_set: HalQuerySet::Vulkan(query_set.clone()),
                    query_index: 1,
                }),
                HalCopy::ResolveQuerySet(HalResolveQuerySet {
                    query_set: HalQuerySet::Vulkan(query_set),
                    first_query: 0,
                    query_count: 2,
                    written_queries: vec![1],
                    destination: HalBuffer::Vulkan(destination.clone()),
                    destination_offset: 0,
                }),
                HalCopy::Buffer(HalBufferCopy {
                    source: HalBuffer::Vulkan(destination),
                    source_offset: 0,
                    destination: HalBuffer::Vulkan(readback.clone()),
                    destination_offset: 0,
                    size: 16,
                }),
            ])
            .expect("submit timestamp write and resolve");
        device.queue().wait_idle().expect("wait idle");

        let ticks = read_query_results(&readback, 2);
        assert_eq!(ticks[0], 0, "unwritten slot should resolve to zero");
        assert!(ticks[1] > 0, "written slot should carry ticks: {ticks:?}");
    }

    /// Block 102 R3: a timestamp write keeps its query set — and therefore the
    /// query pool — alive until the submission retires.
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn collect_retained_resources_retains_write_timestamp_query_set() {
        let device = vulkan_device();
        let query_set = device
            .create_query_set(HalQueryKind::Timestamp, 2)
            .expect("create timestamp query set");
        let query_set_inner = Arc::clone(&query_set.inner);

        let retained = collect_retained_resources(&[HalCopy::WriteTimestamp(HalWriteTimestamp {
            query_set: HalQuerySet::Vulkan(query_set),
            query_index: 0,
        })]);

        assert_eq!(
            retained
                .iter()
                .filter(|resource| matches!(
                    resource,
                    RetainedResource::QuerySet { _inner: inner }
                        if Arc::ptr_eq(inner, &query_set_inner)
                ))
                .count(),
            1
        );
    }

    /// Returns the usage of a timestamp-resolve destination: written by the
    /// resolve copy, read by the conversion pass, copied out for readback.
    #[cfg(feature = "vulkan")]
    fn query_resolve_usage() -> HalBufferUsage {
        HalBufferUsage {
            copy_src: true,
            copy_dst: true,
            query_resolve: true,
            ..HalBufferUsage::default()
        }
    }

    /// Returns the usage of a host-readable readback buffer.
    #[cfg(feature = "vulkan")]
    fn readback_usage() -> HalBufferUsage {
        HalBufferUsage {
            copy_dst: true,
            map_read: true,
            ..HalBufferUsage::default()
        }
    }

    /// Reads `count` resolved 64-bit query results back from a buffer.
    #[cfg(feature = "vulkan")]
    fn read_query_results(buffer: &VulkanBuffer, count: u64) -> Vec<u64> {
        let bytes = buffer
            .read(0, count * 8)
            .expect("read resolved query results");
        bytes
            .chunks_exact(8)
            .map(|chunk| {
                let mut value = [0; 8];
                value.copy_from_slice(chunk);
                u64::from_le_bytes(value)
            })
            .collect()
    }

    #[test]
    fn color_clear_value_uses_format_numeric_class() {
        let float_clear =
            vulkan_color_clear_value(HalTextureFormat::Rgba8Unorm, [1.25, 2.5, 3.75, 4.0]);
        let uint_clear =
            vulkan_color_clear_value(HalTextureFormat::R32Uint, [1.0, 255.0, 65_535.0, 7.0]);
        let sint_clear =
            vulkan_color_clear_value(HalTextureFormat::R32Sint, [-1.0, 2.0, -3.0, 4.0]);

        unsafe {
            assert_eq!(float_clear.color.float32, [1.25, 2.5, 3.75, 4.0]);
            assert_eq!(uint_clear.color.uint32, [1, 255, 65_535, 7]);
            assert_eq!(sint_clear.color.int32, [-1, 2, -3, 4]);
        }
    }

    /// Builds a two-subpass command for GPU-independent binding collection tests.
    #[cfg(feature = "tiled")]
    fn bound_texture_subpass_fixture() -> HalSubpassRenderPassCommand {
        HalSubpassRenderPassCommand {
            layout: HalSubpassPassLayout {
                color_attachments: Vec::new(),
                depth_stencil_attachment: None,
                subpasses: vec![
                    HalSubpassLayout {
                        color_attachment_indices: Vec::new(),
                        uses_depth_stencil: false,
                        input_attachments: Vec::new(),
                    };
                    2
                ],
                dependencies: Vec::new(),
            },
            extent: HalExtent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
            draws: Vec::new(),
        }
    }

    /// Every subpass and draw contributes its bindings in execution order.
    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_bound_textures_collects_all_draws_in_subpass_order() {
        let mut pass = bound_texture_subpass_fixture();
        let texture = dummy_texture(HalTextureFormat::Rgba8Unorm);
        for (subpass_index, binding) in [(1, 2), (0, 0), (1, 3), (0, 1)] {
            let mut bound = bound_texture(texture.clone());
            bound.binding = binding;
            pass.draws.push(HalSubpassDraw {
                subpass_index,
                pipeline: crate::HalRenderPipeline::Noop,
                bind_buffers: Vec::new(),
                bind_textures: vec![bound],
                bind_samplers: Vec::new(),
                vertex_buffers: Vec::new(),
                viewport: None,
                scissor_rect: None,
                draw: HalDraw::Direct {
                    vertex_count: 3,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
            });
        }
        assert_eq!(
            subpass_bound_textures(&pass)
                .iter()
                .map(|bound| bound.binding)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    /// Subpasses without draws contribute no bound textures.
    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_bound_textures_is_empty_without_draws() {
        assert!(subpass_bound_textures(&bound_texture_subpass_fixture()).is_empty());
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_dependencies_map_color_to_input_by_region() {
        let pass = HalSubpassRenderPassCommand {
            layout: HalSubpassPassLayout {
                color_attachments: vec![HalSubpassAttachmentLayout {
                    format: HalTextureFormat::Rgba8Unorm,
                    sample_count: 1,
                }],
                depth_stencil_attachment: None,
                subpasses: vec![
                    HalSubpassLayout {
                        color_attachment_indices: vec![0],
                        uses_depth_stencil: false,
                        input_attachments: Vec::new(),
                    },
                    HalSubpassLayout {
                        color_attachment_indices: vec![0],
                        uses_depth_stencil: false,
                        input_attachments: vec![HalSubpassInputAttachment {
                            group: 0,
                            binding: 0,
                            source_subpass: 0,
                            source_attachment: 0,
                        }],
                    },
                ],
                dependencies: vec![HalSubpassDependency {
                    src_subpass: 0,
                    dst_subpass: 1,
                    dependency_type: HalSubpassDependencyType::ColorToInput,
                    by_region: true,
                }],
            },
            extent: HalExtent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            color_attachments: vec![HalSubpassColorAttachment {
                resource: HalSubpassAttachmentResource::Persistent {
                    texture: dummy_texture(HalTextureFormat::Rgba8Unorm),
                    resolve_target: None,
                },
                load_op: HalRenderLoadOp::Clear,
                store: true,
                clear_color: [0.0, 0.0, 0.0, 1.0],
            }],
            depth_stencil_attachment: None,
            draws: Vec::new(),
        };

        let dependencies = subpass_dependencies(&pass.layout);
        let dependency = dependencies
            .iter()
            .find(|dependency| dependency.src_subpass == 0 && dependency.dst_subpass == 1)
            .expect("layout dependency");

        assert_eq!(
            dependency.src_stage_mask,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        );
        assert_eq!(
            dependency.dst_stage_mask,
            vk::PipelineStageFlags::FRAGMENT_SHADER
        );
        assert_eq!(
            dependency.src_access_mask,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
        );
        assert_eq!(
            dependency.dst_access_mask,
            vk::AccessFlags::INPUT_ATTACHMENT_READ
        );
        assert_eq!(dependency.dependency_flags, vk::DependencyFlags::BY_REGION);
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn subpass_clear_values_follow_attachment_order() {
        let pass = HalSubpassRenderPassCommand {
            layout: HalSubpassPassLayout {
                color_attachments: vec![HalSubpassAttachmentLayout {
                    format: HalTextureFormat::Rgba8Unorm,
                    sample_count: 1,
                }],
                depth_stencil_attachment: Some(HalSubpassAttachmentLayout {
                    format: HalTextureFormat::Depth24PlusStencil8,
                    sample_count: 1,
                }),
                subpasses: vec![HalSubpassLayout {
                    color_attachment_indices: vec![0],
                    uses_depth_stencil: true,
                    input_attachments: Vec::new(),
                }],
                dependencies: Vec::new(),
            },
            extent: HalExtent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            color_attachments: vec![HalSubpassColorAttachment {
                resource: HalSubpassAttachmentResource::Persistent {
                    texture: dummy_texture(HalTextureFormat::Rgba8Unorm),
                    resolve_target: None,
                },
                load_op: HalRenderLoadOp::Clear,
                store: true,
                clear_color: [0.25, 0.5, 0.75, 1.0],
            }],
            depth_stencil_attachment: Some(HalSubpassDepthStencilAttachment {
                resource: HalSubpassAttachmentResource::Persistent {
                    texture: dummy_texture(HalTextureFormat::Depth24PlusStencil8),
                    resolve_target: None,
                },
                depth_load_op: HalRenderLoadOp::Clear,
                depth_store: true,
                depth_clear_value: 0.5,
                stencil_load_op: HalRenderLoadOp::Clear,
                stencil_store: true,
                stencil_clear_value: 3,
            }),
            draws: Vec::new(),
        };

        let values = subpass_clear_values(&pass);

        assert_eq!(values.len(), 2);
        unsafe {
            assert_eq!(values[0].color.float32, [0.25, 0.5, 0.75, 1.0]);
            assert_eq!(values[1].depth_stencil.depth, 0.5);
            assert_eq!(values[1].depth_stencil.stencil, 3);
        }
    }

    #[test]
    fn copy_format_aspect_flags_uses_color_fallback_and_depth_stencil_planes() {
        assert_eq!(
            copy_format_aspect_flags(HalTextureFormat::Rgba8Unorm),
            vk::ImageAspectFlags::COLOR
        );
        assert_eq!(
            copy_format_aspect_flags(HalTextureFormat::Depth32Float),
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            copy_format_aspect_flags(HalTextureFormat::Stencil8),
            vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(
            copy_format_aspect_flags(HalTextureFormat::Depth24PlusStencil8),
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(
            copy_format_aspect_flags(HalTextureFormat::Depth32FloatStencil8),
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
    }

    #[test]
    fn discarded_depth_stencil_aspects_intersects_store_ops_with_format_planes() {
        assert_eq!(
            discarded_depth_stencil_aspects(false, true, HalTextureFormat::Depth32Float),
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            discarded_depth_stencil_aspects(true, false, HalTextureFormat::Depth32Float),
            vk::ImageAspectFlags::empty()
        );
        assert_eq!(
            discarded_depth_stencil_aspects(true, false, HalTextureFormat::Stencil8),
            vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(
            discarded_depth_stencil_aspects(false, true, HalTextureFormat::Depth32FloatStencil8),
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            discarded_depth_stencil_aspects(true, true, HalTextureFormat::Depth32FloatStencil8),
            vk::ImageAspectFlags::empty()
        );
    }

    #[test]
    fn buffer_texture_copy_aspect_flags_honors_requested_aspect() {
        assert_eq!(
            buffer_texture_copy_aspect_flags(
                HalTextureFormat::Depth32FloatStencil8,
                HalTextureAspect::All
            ),
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(
            buffer_texture_copy_aspect_flags(
                HalTextureFormat::Depth32FloatStencil8,
                HalTextureAspect::DepthOnly,
            ),
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            buffer_texture_copy_aspect_flags(
                HalTextureFormat::Depth32FloatStencil8,
                HalTextureAspect::StencilOnly,
            ),
            vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(
            buffer_texture_copy_aspect_flags(HalTextureFormat::Rgba8Unorm, HalTextureAspect::All),
            vk::ImageAspectFlags::COLOR
        );
    }

    #[test]
    fn buffer_image_copy_converts_compressed_layout_from_blocks_to_texels() {
        let device = noop::NoopDevice::new();
        let mut texture =
            dummy_vulkan_texture(HalTextureDimension::D2, HalTextureFormat::Bc1RgbaUnorm);
        texture.bytes_per_pixel = 8;
        texture.height = 8;
        let copy = HalBufferTextureCopy {
            buffer: HalBuffer::Noop(
                device
                    .create_buffer(
                        1024,
                        crate::HalBufferUsage {
                            map_read: false,
                            map_write: false,
                            copy_src: true,
                            copy_dst: true,
                            index: false,
                            vertex: false,
                            uniform: false,
                            storage: false,
                            indirect: false,
                            query_resolve: false,
                        },
                    )
                    .expect("Noop buffer allocation should succeed"),
            ),
            buffer_layout: HalBufferTextureLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
            texture: HalTexture::Vulkan(texture.clone()),
            format: HalTextureFormat::Bc1RgbaUnorm,
            aspect: HalTextureAspect::All,
            mip_level: 0,
            origin: HalOrigin3d { x: 0, y: 0, z: 0 },
            extent: HalExtent3d {
                width: 4,
                height: 8,
                depth_or_array_layers: 2,
            },
        };

        let region = buffer_image_copy(
            &copy,
            &texture,
            texture_bytes_per_pixel(&copy).expect("compressed block byte size"),
            vk::ImageAspectFlags::COLOR,
        )
        .expect("compressed buffer image copy");

        assert_eq!(region.buffer_row_length, 128);
        assert_eq!(region.buffer_image_height, 8);
    }

    // Helper shared by the single-row and multi-row buffer_image_copy tests.
    fn make_copy(
        bytes_per_row: u32,
        rows_per_image: u32,
        width: u32,
        height: u32,
        depth_or_array_layers: u32,
    ) -> (HalBufferTextureCopy, VulkanTexture) {
        let device = noop::NoopDevice::new();
        let texture = dummy_vulkan_texture(HalTextureDimension::D2, HalTextureFormat::Rgba8Unorm);
        let copy = HalBufferTextureCopy {
            buffer: HalBuffer::Noop(
                device
                    .create_buffer(
                        65536,
                        crate::HalBufferUsage {
                            map_read: false,
                            map_write: false,
                            copy_src: true,
                            copy_dst: true,
                            index: false,
                            vertex: false,
                            uniform: false,
                            storage: false,
                            indirect: false,
                            query_resolve: false,
                        },
                    )
                    .expect("Noop buffer allocation should succeed"),
            ),
            buffer_layout: HalBufferTextureLayout {
                offset: 0,
                bytes_per_row,
                rows_per_image,
            },
            texture: HalTexture::Vulkan(texture.clone()),
            format: HalTextureFormat::Rgba8Unorm,
            aspect: HalTextureAspect::All,
            mip_level: 0,
            origin: HalOrigin3d { x: 0, y: 0, z: 0 },
            extent: HalExtent3d {
                width,
                height,
                depth_or_array_layers,
            },
        };
        (copy, texture)
    }

    /// A single-row copy with a non-texel-aligned bytesPerRow (257 bytes for a
    /// 4-byte/texel rgba8unorm texture) must succeed and yield bufferRowLength
    /// == 0 (tightly packed).  WebGPU allows arbitrary bytesPerRow when the
    /// copy height is ≤ one block-row; Vulkan ignores bufferRowLength in that
    /// case.
    #[test]
    fn buffer_image_copy_single_row_non_aligned_bytes_per_row_yields_zero_row_length() {
        // 257 is not divisible by 4 (rgba8unorm bytes-per-pixel).
        let (copy, texture) = make_copy(257, 0, 4, 1, 1);
        let region = buffer_image_copy(
            &copy,
            &texture,
            4, // rgba8unorm bytes_per_pixel
            vk::ImageAspectFlags::COLOR,
        )
        .expect("single-row non-aligned copy must not error");
        assert_eq!(
            region.buffer_row_length, 0,
            "single-row copy must use tightly-packed (0) bufferRowLength"
        );
    }

    /// A single-row multi-slice 3D copy must keep the row length so Vulkan
    /// computes the correct per-slice stride from bufferImageHeight *
    /// bufferRowLength.
    #[test]
    fn buffer_image_copy_single_row_multi_slice_computes_row_length() {
        let device = noop::NoopDevice::new();
        let texture = dummy_vulkan_texture(HalTextureDimension::D3, HalTextureFormat::Rgba8Unorm);
        let copy = HalBufferTextureCopy {
            buffer: HalBuffer::Noop(
                device
                    .create_buffer(
                        65536,
                        crate::HalBufferUsage {
                            map_read: false,
                            map_write: false,
                            copy_src: true,
                            copy_dst: true,
                            index: false,
                            vertex: false,
                            uniform: false,
                            storage: false,
                            indirect: false,
                            query_resolve: false,
                        },
                    )
                    .expect("Noop buffer allocation should succeed"),
            ),
            buffer_layout: HalBufferTextureLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 1,
            },
            texture: HalTexture::Vulkan(texture.clone()),
            format: HalTextureFormat::Rgba8Unorm,
            aspect: HalTextureAspect::All,
            mip_level: 0,
            origin: HalOrigin3d { x: 0, y: 0, z: 0 },
            extent: HalExtent3d {
                width: 5,
                height: 1,
                depth_or_array_layers: 2,
            },
        };

        let region = buffer_image_copy(
            &copy,
            &texture,
            4, // rgba8unorm bytes_per_pixel
            vk::ImageAspectFlags::COLOR,
        )
        .expect("single-row multi-slice copy must not error");

        assert_eq!(region.buffer_row_length, 64);
        assert_eq!(region.buffer_image_height, 1);
    }

    /// A multi-row copy with a texel-aligned bytesPerRow must compute
    /// bufferRowLength exactly as before (regression guard).
    #[test]
    fn buffer_image_copy_multi_row_aligned_bytes_per_row_computes_row_length() {
        // 256 bytes / 4 bytes-per-pixel = 64 texels wide.
        let (copy, texture) = make_copy(256, 4, 4, 4, 1);
        let region = buffer_image_copy(
            &copy,
            &texture,
            4, // rgba8unorm bytes_per_pixel
            vk::ImageAspectFlags::COLOR,
        )
        .expect("multi-row aligned copy must not error");
        // 256 / 4 = 64, block_width = 1 for rgba8unorm, so bufferRowLength = 64.
        assert_eq!(
            region.buffer_row_length, 64,
            "multi-row copy must compute texel-stride bufferRowLength"
        );
    }

    /// A multi-row copy with a non-texel-aligned bytesPerRow must still error
    /// (the divisibility check is only skipped for single-row copies).
    #[test]
    fn buffer_image_copy_multi_row_non_aligned_bytes_per_row_errors() {
        let (copy, texture) = make_copy(257, 4, 4, 4, 1);
        let result = buffer_image_copy(
            &copy,
            &texture,
            4, // rgba8unorm bytes_per_pixel
            vk::ImageAspectFlags::COLOR,
        );
        assert!(
            result.is_err(),
            "multi-row copy with non-aligned bytesPerRow must return an error"
        );
    }

    #[test]
    fn render_attachment_descriptions_preserve_contents_for_load_ops() {
        let color_target = HalRenderColorTarget {
            texture: HalTexture::Vulkan(dummy_vulkan_texture(
                HalTextureDimension::D2,
                HalTextureFormat::Rgba8Unorm,
            )),
            view_format: HalTextureFormat::Rgba8Unorm,
            resolve_target: None,
            resolve_view_format: None,
            mip_level: 0,
            array_layer: 0,
            depth_slice: 0,
            resolve_mip_level: 0,
            resolve_array_layer: 0,
            load_op: HalRenderLoadOp::Load,
            store: false,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        };
        let color =
            vk_color_attachment_description(HalTextureFormat::Rgba8Unorm, &color_target, false)
                .expect("color attachment description");
        assert_eq!(color.load_op, vk::AttachmentLoadOp::LOAD);
        assert_eq!(color.store_op, vk::AttachmentStoreOp::DONT_CARE);
        assert_eq!(
            color.initial_layout,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );

        let depth_stencil = HalRenderDepthStencilAttachment {
            texture: HalTexture::Vulkan(dummy_vulkan_texture(
                HalTextureDimension::D2,
                HalTextureFormat::Depth32FloatStencil8,
            )),
            format: HalTextureFormat::Depth32FloatStencil8,
            mip_level: 0,
            array_layer: 0,
            depth_load_op: HalRenderLoadOp::Load,
            depth_store: true,
            depth_clear_value: 0.5,
            depth_read_only: false,
            stencil_load_op: HalRenderLoadOp::Load,
            stencil_store: false,
            stencil_clear_value: 0,
            stencil_read_only: false,
        };
        let depth =
            vk_render_depth_stencil_attachment_description(&depth_stencil).expect("depth desc");
        assert_eq!(depth.load_op, vk::AttachmentLoadOp::LOAD);
        assert_eq!(depth.store_op, vk::AttachmentStoreOp::STORE);
        assert_eq!(depth.stencil_load_op, vk::AttachmentLoadOp::LOAD);
        assert_eq!(depth.stencil_store_op, vk::AttachmentStoreOp::DONT_CARE);
        assert_eq!(
            depth.initial_layout,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
    }

    #[test]
    fn render_attachment_subresource_ranges_scope_mip_layer_and_aspect() {
        let color_target = HalRenderColorTarget {
            texture: dummy_texture(HalTextureFormat::Rgba8Unorm),
            view_format: HalTextureFormat::Rgba8Unorm,
            resolve_target: None,
            resolve_view_format: None,
            mip_level: 2,
            array_layer: 1,
            depth_slice: 0,
            resolve_mip_level: 0,
            resolve_array_layer: 0,
            load_op: HalRenderLoadOp::Clear,
            store: true,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        };
        let color_texture =
            dummy_vulkan_texture(HalTextureDimension::D2, HalTextureFormat::Rgba8Unorm);
        let color = color_attachment_subresource_range(&color_texture, &color_target);
        assert_eq!(color.aspect_mask, vk::ImageAspectFlags::COLOR);
        assert_eq!(color.base_mip_level, 2);
        assert_eq!(color.level_count, 1);
        assert_eq!(color.base_array_layer, 1);
        assert_eq!(color.layer_count, 1);

        let color_3d_texture =
            dummy_vulkan_texture(HalTextureDimension::D3, HalTextureFormat::Rgba8Unorm);
        let color_3d = color_attachment_subresource_range(
            &color_3d_texture,
            &HalRenderColorTarget {
                depth_slice: 7,
                ..color_target.clone()
            },
        );
        assert_eq!(color_3d.base_array_layer, 7);

        let depth = HalRenderDepthStencilAttachment {
            texture: dummy_texture(HalTextureFormat::Depth32Float),
            format: HalTextureFormat::Depth32Float,
            mip_level: 3,
            array_layer: 4,
            depth_load_op: HalRenderLoadOp::Clear,
            depth_store: true,
            depth_clear_value: 0.5,
            depth_read_only: false,
            stencil_load_op: HalRenderLoadOp::Clear,
            stencil_store: false,
            stencil_clear_value: 0,
            stencil_read_only: false,
        };
        let depth_range = depth_stencil_attachment_subresource_range(&depth);
        assert_eq!(depth_range.aspect_mask, vk::ImageAspectFlags::DEPTH);
        assert_eq!(depth_range.base_mip_level, 3);
        assert_eq!(depth_range.level_count, 1);
        assert_eq!(depth_range.base_array_layer, 4);
        assert_eq!(depth_range.layer_count, 1);

        let packed = HalRenderDepthStencilAttachment {
            texture: dummy_texture(HalTextureFormat::Depth32FloatStencil8),
            format: HalTextureFormat::Depth32FloatStencil8,
            mip_level: 1,
            array_layer: 2,
            depth_load_op: HalRenderLoadOp::Clear,
            depth_store: true,
            depth_clear_value: 0.5,
            depth_read_only: false,
            stencil_load_op: HalRenderLoadOp::Clear,
            stencil_store: true,
            stencil_clear_value: 0,
            stencil_read_only: false,
        };
        let packed_range = depth_stencil_attachment_subresource_range(&packed);
        assert_eq!(
            packed_range.aspect_mask,
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(packed_range.base_mip_level, 1);
        assert_eq!(packed_range.level_count, 1);
        assert_eq!(packed_range.base_array_layer, 2);
        assert_eq!(packed_range.layer_count, 1);
    }

    #[test]
    fn render_attachment_image_view_usage_is_limited_to_attachment_role() {
        assert_eq!(
            color_attachment_image_view_usage(),
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT
        );
        assert_eq!(
            depth_stencil_attachment_image_view_usage(),
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        );
    }

    #[test]
    fn mip_extent_uses_attachment_mip_level_size_with_floor() {
        assert_eq!(mip_extent(16, 16, 0), (16, 16));
        assert_eq!(mip_extent(16, 16, 2), (4, 4));
        assert_eq!(mip_extent(24, 10, 2), (6, 2));
        assert_eq!(mip_extent(1, 1, 8), (1, 1));
    }

    /// Block 104 R6: the clear path is chosen from the format's own aspects and
    /// block layout, narrowed by the requested aspect — not from the requested
    /// aspect alone.
    #[test]
    fn clear_kind_rejects_missing_transfer_destination_usage() {
        for format in [
            HalTextureFormat::Rgba8Unorm,
            HalTextureFormat::Depth32Float,
            HalTextureFormat::Bc1RgbaUnorm,
        ] {
            let error = clear_kind(
                format,
                HalTextureAspect::All,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("texture clear requires TRANSFER_DST image usage"));
        }
    }

    #[test]
    fn clear_kind_selects_depth_stencil_compressed_and_color_paths() {
        assert_eq!(
            clear_kind(
                HalTextureFormat::Rgba8Unorm,
                HalTextureAspect::All,
                vk::ImageUsageFlags::TRANSFER_DST
            )
            .unwrap(),
            ClearKind::Color
        );
        assert_eq!(
            clear_kind(
                HalTextureFormat::Depth32Float,
                HalTextureAspect::All,
                vk::ImageUsageFlags::TRANSFER_DST
            )
            .unwrap(),
            ClearKind::DepthStencil(vk::ImageAspectFlags::DEPTH)
        );
        assert_eq!(
            clear_kind(
                HalTextureFormat::Stencil8,
                HalTextureAspect::All,
                vk::ImageUsageFlags::TRANSFER_DST
            )
            .unwrap(),
            ClearKind::DepthStencil(vk::ImageAspectFlags::STENCIL)
        );
        assert_eq!(
            clear_kind(
                HalTextureFormat::Depth24PlusStencil8,
                HalTextureAspect::All,
                vk::ImageUsageFlags::TRANSFER_DST
            )
            .unwrap(),
            ClearKind::DepthStencil(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
        );
        assert_eq!(
            clear_kind(
                HalTextureFormat::Depth32FloatStencil8,
                HalTextureAspect::DepthOnly,
                vk::ImageUsageFlags::TRANSFER_DST
            )
            .unwrap(),
            ClearKind::DepthStencil(vk::ImageAspectFlags::DEPTH)
        );
        assert_eq!(
            clear_kind(
                HalTextureFormat::Depth32FloatStencil8,
                HalTextureAspect::StencilOnly,
                vk::ImageUsageFlags::TRANSFER_DST
            )
            .unwrap(),
            ClearKind::DepthStencil(vk::ImageAspectFlags::STENCIL)
        );
        // A stencil request on a depth-only format selects no plane, so the
        // clear becomes a no-op instead of an invalid colour clear of a depth
        // image.
        assert_eq!(
            clear_kind(
                HalTextureFormat::Depth32Float,
                HalTextureAspect::StencilOnly,
                vk::ImageUsageFlags::TRANSFER_DST
            )
            .unwrap(),
            ClearKind::DepthStencil(vk::ImageAspectFlags::empty())
        );
        assert_eq!(
            clear_kind(
                HalTextureFormat::Bc1RgbaUnorm,
                HalTextureAspect::All,
                vk::ImageUsageFlags::TRANSFER_DST
            )
            .unwrap(),
            ClearKind::Compressed {
                block_bytes: 8,
                block_width: 4,
                block_height: 4,
            }
        );
        assert_eq!(
            clear_kind(
                HalTextureFormat::Astc6x5Unorm,
                HalTextureAspect::All,
                vk::ImageUsageFlags::TRANSFER_DST
            )
            .unwrap(),
            ClearKind::Compressed {
                block_bytes: 16,
                block_width: 6,
                block_height: 5,
            }
        );
    }

    /// Block 104 R6: the zeroed staging bytes of a compressed clear count whole
    /// blocks, so a mip edge that does not fill its last block still gets a
    /// full block row and a full block column.
    #[test]
    fn compressed_clear_layout_counts_whole_blocks_at_the_mip_edge() {
        let bc1 = (8, 4, 4);
        assert_eq!(
            compressed_clear_layout(8, 8, bc1).expect("bc1 8x8"),
            (16, 32)
        );
        assert_eq!(
            compressed_clear_layout(9, 9, bc1).expect("bc1 9x9"),
            (24, 72)
        );
        assert_eq!(compressed_clear_layout(4, 4, bc1).expect("bc1 4x4"), (8, 8));
        assert_eq!(compressed_clear_layout(1, 1, bc1).expect("bc1 1x1"), (8, 8));
        let astc6x5 = (16, 6, 5);
        assert_eq!(
            compressed_clear_layout(7, 6, astc6x5).expect("astc6x5 7x6"),
            (32, 64)
        );
        assert!(compressed_clear_layout(4, 4, (8, 0, 4)).is_err());
        assert!(compressed_clear_layout(4, 4, (8, 4, 0)).is_err());
    }

    /// Block 104 R6: a compressed clear region covers the physical (block
    /// rounded) mip so `vkCmdCopyBufferToImage` transfers whole blocks.
    #[test]
    fn block_aligned_extent_rounds_up_and_rejects_a_zero_block() {
        assert_eq!(block_aligned_extent(9, 4).expect("9 texels"), 12);
        assert_eq!(block_aligned_extent(8, 4).expect("8 texels"), 8);
        assert_eq!(block_aligned_extent(1, 4).expect("1 texel"), 4);
        assert_eq!(block_aligned_extent(7, 6).expect("7 texels"), 12);
        assert!(block_aligned_extent(4, 0).is_err());
    }

    /// Returns the first Vulkan adapter together with a device created on it,
    /// so a real-device test can skip on an optional format the adapter does
    /// not advertise.
    #[cfg(feature = "vulkan")]
    fn vulkan_adapter_and_device() -> (VulkanAdapter, VulkanDevice) {
        let instance = VulkanInstance::new().expect("create Vulkan instance");
        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Vulkan adapter");
        let device = adapter.create_device().expect("create Vulkan device");
        (adapter, device)
    }

    /// Returns the usage of a texture that is rendered to and copied both ways.
    #[cfg(feature = "vulkan")]
    fn clear_test_attachment_usage() -> HalTextureUsage {
        HalTextureUsage {
            copy_src: true,
            copy_dst: true,
            texture_binding: false,
            storage_binding: false,
            render_attachment: true,
            transient: false,
        }
    }

    /// Returns the usage of a texture that is only copied both ways.
    #[cfg(feature = "vulkan")]
    fn clear_test_copy_usage() -> HalTextureUsage {
        HalTextureUsage {
            copy_src: true,
            copy_dst: true,
            texture_binding: false,
            storage_binding: false,
            render_attachment: false,
            transient: false,
        }
    }

    /// Returns the usage of a host-written upload buffer.
    #[cfg(feature = "vulkan")]
    fn upload_usage() -> HalBufferUsage {
        HalBufferUsage {
            copy_src: true,
            map_write: true,
            ..HalBufferUsage::default()
        }
    }

    /// Reads `count` little-endian `f32` texels back from a buffer.
    #[cfg(feature = "vulkan")]
    fn read_f32s(buffer: &VulkanBuffer, count: u64) -> Vec<f32> {
        buffer
            .read(0, count * 4)
            .expect("read float texels")
            .chunks_exact(4)
            .map(|chunk| {
                let mut value = [0; 4];
                value.copy_from_slice(chunk);
                f32::from_le_bytes(value)
            })
            .collect()
    }

    /// Returns a buffer/texture copy of one full square 2D mip subresource of
    /// `size` texels a side, tightly packed at `bytes_per_block` bytes per
    /// texel (per compressed block for a block format).
    #[cfg(feature = "vulkan")]
    fn clear_test_copy(
        buffer: &VulkanBuffer,
        texture: &VulkanTexture,
        format: HalTextureFormat,
        aspect: HalTextureAspect,
        mip_level: u32,
        size: u32,
        bytes_per_block: u32,
    ) -> HalBufferTextureCopy {
        let (_, block_width, block_height) = format.compressed_block_info().unwrap_or((1, 1, 1));
        HalBufferTextureCopy {
            buffer: HalBuffer::Vulkan(buffer.clone()),
            buffer_layout: HalBufferTextureLayout {
                offset: 0,
                bytes_per_row: size.div_ceil(block_width) * bytes_per_block,
                rows_per_image: size.div_ceil(block_height),
            },
            texture: HalTexture::Vulkan(texture.clone()),
            format,
            aspect,
            mip_level,
            origin: HalOrigin3d { x: 0, y: 0, z: 0 },
            extent: HalExtent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        }
    }

    /// Returns a whole-subresource clear of mip `mip_level`, layer 0.
    #[cfg(feature = "vulkan")]
    fn clear_test_clear(
        texture: &VulkanTexture,
        format: HalTextureFormat,
        aspect: HalTextureAspect,
        mip_level: u32,
    ) -> HalCopy {
        HalCopy::ClearTexture(HalTextureClear {
            texture: HalTexture::Vulkan(texture.clone()),
            format,
            aspect,
            mip_level,
            base_array_layer: 0,
            array_layer_count: 1,
        })
    }

    /// Block 104 R6: `ClearTexture` zeroes a `depth32float` depth aspect on a
    /// real device. The canary read back first proves the zeros come from the
    /// clear and not from a freshly allocated (already zero) image.
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_clear_texture_zeroes_depth32float_after_canary() {
        let device = vulkan_device();
        let format = HalTextureFormat::Depth32Float;
        let texture = device
            .create_texture(&HalTextureDescriptor {
                dimension: HalTextureDimension::D2,
                format,
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: clear_test_copy_usage(),
            })
            .expect("create depth32float texture");
        let upload = device
            .create_buffer(64, upload_usage())
            .expect("create upload buffer");
        let readback = device
            .create_buffer(64, readback_usage())
            .expect("create readback buffer");
        let canary: Vec<u8> = (0..16).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        upload.write(0, &canary).expect("write depth canary");
        let depth_copy = |buffer: &VulkanBuffer| {
            clear_test_copy(
                buffer,
                &texture,
                format,
                HalTextureAspect::DepthOnly,
                0,
                4,
                4,
            )
        };

        device
            .queue()
            .submit_copies(&[
                HalCopy::BufferToTexture(depth_copy(&upload)),
                HalCopy::TextureToBuffer(depth_copy(&readback)),
            ])
            .expect("submit depth canary");
        device.queue().wait_idle().expect("wait idle");
        assert_eq!(
            read_f32s(&readback, 16),
            vec![1.0; 16],
            "the canary must land before the clear is measured"
        );

        device
            .queue()
            .submit_copies(&[
                clear_test_clear(&texture, format, HalTextureAspect::DepthOnly, 0),
                HalCopy::TextureToBuffer(depth_copy(&readback)),
            ])
            .expect("submit depth clear");
        device.queue().wait_idle().expect("wait idle");

        assert_eq!(read_f32s(&readback, 16), vec![0.0; 16]);
    }

    /// Block 104 R6: a `StencilOnly` clear of a combined depth-stencil texture
    /// zeroes the stencil plane and leaves the depth plane's canary intact.
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_clear_texture_zeroes_stencil_aspect_only() {
        let (adapter, device) = vulkan_adapter_and_device();
        if !adapter.supports_depth32float_stencil8() {
            eprintln!("skipping: adapter does not advertise depth32float-stencil8");
            return;
        }
        let format = HalTextureFormat::Depth32FloatStencil8;
        let texture = device
            .create_texture(&HalTextureDescriptor {
                dimension: HalTextureDimension::D2,
                format,
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: clear_test_attachment_usage(),
            })
            .expect("create depth32float-stencil8 texture");
        let depth_readback = device
            .create_buffer(64, readback_usage())
            .expect("create depth readback buffer");
        let stencil_readback = device
            .create_buffer(16, readback_usage())
            .expect("create stencil readback buffer");
        // The canary is written by an empty render pass that clears depth to
        // 1.0 and stencil to 0xff, the only way to write both planes at once.
        let canary_pass = HalCopy::RenderPassCommandStream(HalRenderPassCommandStream {
            color_targets: Vec::new(),
            framebuffer_fetch_color_slots: Vec::new(),
            depth_stencil_attachment: Some(HalRenderDepthStencilAttachment {
                texture: HalTexture::Vulkan(texture.clone()),
                format,
                mip_level: 0,
                array_layer: 0,
                depth_load_op: HalRenderLoadOp::Clear,
                depth_store: true,
                depth_clear_value: 1.0,
                depth_read_only: false,
                stencil_load_op: HalRenderLoadOp::Clear,
                stencil_store: true,
                stencil_clear_value: 0xff,
                stencil_read_only: false,
            }),
            occlusion_query_set: None,
            commands: Vec::new(),
        });
        let depth_copy = HalCopy::TextureToBuffer(clear_test_copy(
            &depth_readback,
            &texture,
            format,
            HalTextureAspect::DepthOnly,
            0,
            4,
            4,
        ));
        let stencil_copy = HalCopy::TextureToBuffer(clear_test_copy(
            &stencil_readback,
            &texture,
            format,
            HalTextureAspect::StencilOnly,
            0,
            4,
            1,
        ));

        // Both aspect readbacks ride in the same submission as the render pass
        // that wrote them. The first readback transitions both image aspects;
        // the second reuses that subresource layout without narrowing it.
        device
            .queue()
            .submit_copies(&[canary_pass, depth_copy.clone(), stencil_copy.clone()])
            .expect("submit depth-stencil canary");
        device.queue().wait_idle().expect("wait idle");
        assert_eq!(read_f32s(&depth_readback, 16), vec![1.0; 16]);
        assert_eq!(
            stencil_readback.read(0, 16).expect("read stencil canary"),
            vec![0xff; 16],
            "the stencil canary must land before the clear is measured"
        );

        device
            .queue()
            .submit_copies(&[
                clear_test_clear(&texture, format, HalTextureAspect::StencilOnly, 0),
                stencil_copy,
                depth_copy,
            ])
            .expect("submit stencil clear");
        device.queue().wait_idle().expect("wait idle");

        assert_eq!(
            stencil_readback.read(0, 16).expect("read cleared stencil"),
            vec![0; 16]
        );
        assert_eq!(
            read_f32s(&depth_readback, 16),
            vec![1.0; 16],
            "a stencil-only clear must not touch the depth plane"
        );
    }

    /// Block 104 R6: `ClearTexture` zeroes every sample of a 4x MSAA colour
    /// texture, observed through a resolve into a single-sample texture.
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_clear_texture_zeroes_multisampled_color() {
        let device = vulkan_device();
        let format = HalTextureFormat::Rgba8Unorm;
        let multisampled = device
            .create_texture(&HalTextureDescriptor {
                dimension: HalTextureDimension::D2,
                format,
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 4,
                usage: HalTextureUsage {
                    copy_src: false,
                    copy_dst: false,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("create 4x MSAA texture");
        let resolved = device
            .create_texture(&HalTextureDescriptor {
                dimension: HalTextureDimension::D2,
                format,
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: clear_test_attachment_usage(),
            })
            .expect("create resolve target");
        let readback = device
            .create_buffer(64, readback_usage())
            .expect("create readback buffer");
        let resolve_pass = |load_op, clear_color| {
            HalCopy::RenderPassCommandStream(HalRenderPassCommandStream {
                color_targets: vec![Some(HalRenderColorTarget {
                    texture: HalTexture::Vulkan(multisampled.clone()),
                    view_format: format,
                    resolve_target: Some(HalTexture::Vulkan(resolved.clone())),
                    resolve_view_format: Some(format),
                    mip_level: 0,
                    array_layer: 0,
                    depth_slice: 0,
                    resolve_mip_level: 0,
                    resolve_array_layer: 0,
                    load_op,
                    store: true,
                    clear_color,
                })],
                framebuffer_fetch_color_slots: Vec::new(),
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                commands: Vec::new(),
            })
        };
        let resolved_copy = HalCopy::TextureToBuffer(clear_test_copy(
            &readback,
            &resolved,
            format,
            HalTextureAspect::All,
            0,
            4,
            4,
        ));

        device
            .queue()
            .submit_copies(&[
                resolve_pass(HalRenderLoadOp::Clear, [1.0, 0.0, 0.0, 1.0]),
                resolved_copy.clone(),
            ])
            .expect("submit MSAA canary");
        device.queue().wait_idle().expect("wait idle");
        assert_eq!(
            readback.read(0, 64).expect("read MSAA canary"),
            [255, 0, 0, 255].repeat(16),
            "the canary must land before the clear is measured"
        );

        device
            .queue()
            .submit_copies(&[
                clear_test_clear(&multisampled, format, HalTextureAspect::All, 0),
                resolve_pass(HalRenderLoadOp::Load, [0.0; 4]),
                resolved_copy,
            ])
            .expect("submit MSAA clear");
        device.queue().wait_idle().expect("wait idle");

        assert_eq!(
            readback.read(0, 64).expect("read cleared MSAA"),
            vec![0; 64]
        );
    }

    /// Block 104 R6: `ClearTexture` of one mip of a block-compressed texture
    /// zeroes that mip's blocks and leaves the other mip's canary intact.
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_clear_texture_zeroes_bc1_mip_and_keeps_canary_mip() {
        let (adapter, device) = vulkan_adapter_and_device();
        if !adapter.supports_texture_compression_bc() {
            eprintln!("skipping: adapter does not advertise BC texture compression");
            return;
        }
        let format = HalTextureFormat::Bc1RgbaUnorm;
        let texture = device
            .create_texture(&HalTextureDescriptor {
                dimension: HalTextureDimension::D2,
                format,
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
                mip_level_count: 2,
                sample_count: 1,
                usage: clear_test_copy_usage(),
            })
            .expect("create bc1-rgba-unorm texture");
        // 8x8 is 2x2 blocks of 8 bytes; mip 1 is 4x4, a single block.
        let upload = device
            .create_buffer(32, upload_usage())
            .expect("create upload buffer");
        let mip0_readback = device
            .create_buffer(32, readback_usage())
            .expect("create mip 0 readback buffer");
        let mip1_readback = device
            .create_buffer(8, readback_usage())
            .expect("create mip 1 readback buffer");
        let canary: Vec<u8> = (0..32u8).map(|byte| byte.wrapping_add(1)).collect();
        upload.write(0, &canary).expect("write block canary");
        let mip0_copy = |buffer: &VulkanBuffer| {
            clear_test_copy(buffer, &texture, format, HalTextureAspect::All, 0, 8, 8)
        };
        let mip1_copy = |buffer: &VulkanBuffer| {
            clear_test_copy(buffer, &texture, format, HalTextureAspect::All, 1, 4, 8)
        };

        // Both mips carry a canary, so the mip 1 zero assertion cannot pass on
        // a freshly allocated (already zero) image.
        device
            .queue()
            .submit_copies(&[
                HalCopy::BufferToTexture(mip0_copy(&upload)),
                HalCopy::BufferToTexture(mip1_copy(&upload)),
                HalCopy::TextureToBuffer(mip0_copy(&mip0_readback)),
                HalCopy::TextureToBuffer(mip1_copy(&mip1_readback)),
            ])
            .expect("submit block canary");
        device.queue().wait_idle().expect("wait idle");
        assert_eq!(
            mip0_readback.read(0, 32).expect("read mip 0 canary"),
            canary
        );
        assert_eq!(
            mip1_readback.read(0, 8).expect("read mip 1 canary"),
            canary[..8],
            "the canary must land before the clear is measured"
        );

        device
            .queue()
            .submit_copies(&[
                clear_test_clear(&texture, format, HalTextureAspect::All, 1),
                HalCopy::TextureToBuffer(mip1_copy(&mip1_readback)),
                HalCopy::TextureToBuffer(mip0_copy(&mip0_readback)),
            ])
            .expect("submit block clear");
        device.queue().wait_idle().expect("wait idle");

        assert_eq!(
            mip1_readback.read(0, 8).expect("read cleared mip 1"),
            vec![0; 8]
        );
        assert_eq!(
            mip0_readback.read(0, 32).expect("read retained mip 0"),
            canary,
            "clearing mip 1 must not touch mip 0"
        );
    }
}
