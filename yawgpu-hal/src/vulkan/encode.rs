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

/// Records submit into the command stream.
pub(super) fn submit_copies(queue: &VulkanQueueInner, copies: &[HalCopy]) -> Result<(), HalError> {
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
) -> Result<(), HalError> {
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
                    encode_texture_clear(&queue.device.device, command_buffer, clear)?;
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
                    encode_texture_to_texture(&queue.device.device, command_buffer, copy)?;
                }
                HalCopy::ComputePass(pass) => {
                    let temps = encode_compute_pass(&queue.device.device, command_buffer, pass)?;
                    if let Some(pool) = temps.descriptor_pool {
                        descriptor_pools.push(pool);
                    }
                    image_views.extend(temps.image_views);
                }
                HalCopy::RenderPass(pass) => {
                    let temps = encode_render_pass(&queue.device, command_buffer, pass)?;
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
            queue
                .device
                .device
                .queue_submit(queue.queue, &[submit_info], created_fence)
                .map_err(|error| queue_submission_error("vkQueueSubmit", error))?;
            fence = None;
            let retire_fence = created_fence;
            let retained = collect_retained_resources(copies);
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
                pending_state.retire.retire(
                    &queue.device.device,
                    retire_fence,
                    cleanup,
                    retained,
                    true,
                )?;
            } else {
                let mut retire = queue
                    .retire
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                retire.retire(&queue.device.device, retire_fence, cleanup, retained, true)?;
            }
        }
        Ok(())
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
        );
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
            queue
                .inner
                .device
                .device
                .queue_submit(queue.inner.queue, &[submit_info], fence)
                .map_err(|_| HalError::PresentFailed {
                    backend: BACKEND,
                    message: "queue submit failed",
                })?;
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
    let mut retained = Vec::new();
    for copy in copies {
        retain_copy_resources(copy, &mut retained);
    }
    retained
}

fn retain_copy_resources(copy: &HalCopy, retained: &mut Vec<RetainedResource>) {
    match copy {
        HalCopy::Buffer(copy) => {
            retain_hal_buffer(&copy.source, retained);
            retain_hal_buffer(&copy.destination, retained);
        }
        HalCopy::BufferClear(clear) => retain_hal_buffer(&clear.buffer, retained),
        HalCopy::ClearTexture(clear) => retain_hal_texture(&clear.texture, retained),
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
        HalCopy::RenderPass(pass) => {
            if let Some(pipeline) = &pass.pipeline {
                retain_hal_render_pipeline(pipeline, retained);
            }
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
            for bound in &pass.vertex_buffers {
                retain_hal_buffer(&bound.buffer, retained);
            }
            if let Some(index_buffer) = &pass.index_buffer {
                retain_hal_buffer(&index_buffer.buffer, retained);
            }
            if let Some(indirect_buffer) = &pass.indirect_buffer {
                retain_hal_buffer(&indirect_buffer.buffer, retained);
            }
        }
        #[cfg(feature = "tiled")]
        HalCopy::SubpassRenderPass(pass) => retain_subpass_resources(pass, retained),
    }
}

fn retain_hal_buffer(buffer: &crate::HalBuffer, retained: &mut Vec<RetainedResource>) {
    let crate::HalBuffer::Vulkan(buffer) = buffer else {
        return;
    };
    if let Some(inner) = &buffer.inner {
        retained.push(RetainedResource::Buffer {
            _inner: Arc::clone(inner),
        });
    }
}

fn retain_hal_texture(texture: &HalTexture, retained: &mut Vec<RetainedResource>) {
    let HalTexture::Vulkan(texture) = texture else {
        return;
    };
    if let Some(inner) = &texture.inner {
        retained.push(RetainedResource::Texture {
            _inner: Arc::clone(inner),
        });
    }
}

fn retain_hal_sampler(sampler: &HalSampler, retained: &mut Vec<RetainedResource>) {
    let HalSampler::Vulkan(sampler) = sampler else {
        return;
    };
    if let Some(inner) = &sampler._inner {
        retained.push(RetainedResource::Sampler {
            _inner: Arc::clone(inner),
        });
    }
}

fn retain_hal_query_set(query_set: &HalQuerySet, retained: &mut Vec<RetainedResource>) {
    let HalQuerySet::Vulkan(query_set) = query_set else {
        return;
    };
    retained.push(RetainedResource::QuerySet {
        _inner: Arc::clone(&query_set.inner),
    });
}

fn retain_hal_compute_pipeline(
    pipeline: &crate::HalComputePipeline,
    retained: &mut Vec<RetainedResource>,
) {
    let crate::HalComputePipeline::Vulkan(pipeline) = pipeline else {
        return;
    };
    retained.push(RetainedResource::ComputePipeline {
        _inner: Arc::clone(&pipeline.inner),
    });
}

fn retain_hal_render_pipeline(
    pipeline: &crate::HalRenderPipeline,
    retained: &mut Vec<RetainedResource>,
) {
    let crate::HalRenderPipeline::Vulkan(pipeline) = pipeline else {
        return;
    };
    retained.push(RetainedResource::RenderPipeline {
        _inner: Arc::clone(&pipeline.inner),
    });
}

fn retain_hal_external_texture(
    texture: &crate::HalBoundExternalTexture,
    retained: &mut Vec<RetainedResource>,
) {
    retain_hal_texture(&texture.plane0, retained);
    retain_hal_texture(&texture.plane1, retained);
    retain_hal_buffer(&texture.params, retained);
}

#[cfg(feature = "tiled")]
fn retain_subpass_resources(
    pass: &HalSubpassRenderPassCommand,
    retained: &mut Vec<RetainedResource>,
) {
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
    retained: &mut Vec<RetainedResource>,
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
        | HalCopy::ResolveQuerySet(_)
        | HalCopy::ComputePass(_) => None,
        #[cfg(feature = "tiled")]
        HalCopy::SubpassRenderPass(pass) => surface_pending_from_subpass(pass),
        HalCopy::BufferToTexture(copy) | HalCopy::TextureToBuffer(copy) => {
            surface_pending_from_hal_texture(&copy.texture)
        }
        HalCopy::TextureToTexture(copy) => surface_pending_from_hal_texture(&copy.source)
            .or_else(|| surface_pending_from_hal_texture(&copy.destination)),
        HalCopy::RenderPass(pass) => pass.color_targets.iter().flatten().find_map(|target| {
            surface_pending_from_hal_texture(&target.texture).or_else(|| {
                target
                    .resolve_target
                    .as_ref()
                    .and_then(surface_pending_from_hal_texture)
            })
        }),
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

/// Records texture clear encode into the command stream.
pub(super) fn encode_texture_clear(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    clear: &HalTextureClear,
) -> Result<(), HalError> {
    let crate::HalTexture::Vulkan(texture) = &clear.texture else {
        return Err(texture_error("texture is not Vulkan-backed"));
    };
    validate_mip_level(texture, clear.mip_level)?;
    let texture_inner = texture.inner()?;
    let aspect = buffer_texture_copy_aspect_flags(clear.format, clear.aspect);
    transition_image_aspect(
        device,
        command_buffer,
        texture_inner,
        aspect,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_DST,
    );
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .base_mip_level(clear.mip_level)
        .level_count(1)
        .base_array_layer(match texture.dimension {
            HalTextureDimension::D3 => 0,
            HalTextureDimension::D1 | HalTextureDimension::D2 => clear.base_array_layer,
        })
        .layer_count(match texture.dimension {
            HalTextureDimension::D3 => 1,
            HalTextureDimension::D1 | HalTextureDimension::D2 => clear.array_layer_count,
        });
    let value = unsafe { vulkan_color_clear_value(clear.format, [0.0; 4]).color };
    unsafe {
        device.cmd_clear_color_image(
            command_buffer,
            texture_inner.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &value,
            &[range],
        );
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(texture_inner.image)
            .subresource_range(range)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::TRANSFER_READ);
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
    }
    Ok(())
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
    transition_image_aspect(
        device,
        command_buffer,
        texture_inner,
        aspect,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_DST,
    );
    let region = buffer_image_copy(copy, texture, texture_bytes_per_pixel(copy)?, aspect)?;
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
    transition_image_aspect(
        device,
        command_buffer,
        texture_inner,
        aspect,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_SRC,
    );
    let region = buffer_image_copy(copy, texture, texture_bytes_per_pixel(copy)?, aspect)?;
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
    transition_image_aspect(
        device,
        command_buffer,
        source_inner,
        aspect,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_SRC,
    );
    transition_image_aspect(
        device,
        command_buffer,
        destination_inner,
        aspect,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_DST,
    );
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
        .extent(texture_copy_extent(source.dimension, copy.extent));
    unsafe {
        device.cmd_copy_image(
            command_buffer,
            source_inner.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            destination_inner.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    Ok(())
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
        match allocate_compute_descriptor_sets(device, pool, pipeline) {
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
    transition_storage_textures(device, command_buffer, &pass.bind_textures)?;
    transfer_to_compute_barrier(device, command_buffer);
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
                device.cmd_dispatch(command_buffer, workgroups.0, workgroups.1, workgroups.2);
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
    compute_to_transfer_barrier(device, command_buffer);
    Ok(ComputePassTemps {
        descriptor_pool,
        image_views,
    })
}

fn transition_storage_textures(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    textures: &[HalBoundTexture],
) -> Result<(), HalError> {
    for bound in textures
        .iter()
        .filter(|bound| bound.storage_access.is_some())
    {
        let crate::HalTexture::Vulkan(texture) = &bound.texture else {
            return Err(texture_error("storage texture is not Vulkan-backed"));
        };
        transition_image_aspect(
            device,
            command_buffer,
            texture.inner()?,
            buffer_texture_copy_aspect_flags(bound.format, bound.aspect),
            vk::ImageLayout::GENERAL,
            IMAGE_LAYOUT_GENERAL,
        );
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
    let (views, persistent_textures) = subpass_attachment_views(pass)?;
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
        );
    }
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
        texture.inner()?.layout.store(
            subpass_color_tracked_layout(texture.transient),
            AtomicOrdering::Relaxed,
        );
    }
    Ok(RenderPassTemps {
        descriptor_pools,
        framebuffer,
        image_views,
        render_pass: None,
    })
}

#[cfg(feature = "tiled")]
struct SubpassDrawTemps {
    descriptor_pool: Option<vk::DescriptorPool>,
    image_views: Vec<vk::ImageView>,
}

#[cfg(feature = "tiled")]
type SubpassAttachmentViews = (Vec<vk::ImageView>, Vec<(u32, VulkanTexture)>);

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
        match allocate_render_descriptor_sets(device, pool, pipeline) {
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
    if let Some(depth) = &pass.depth_stencil_attachment {
        let (view, texture) = subpass_attachment_view(&depth.resource)?;
        views.push(view);
        persistent_textures.push((u32::MAX, texture));
    }
    Ok((views, persistent_textures))
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

/// Records encode into the command stream.
pub(super) fn encode_render_pass(
    device: &VulkanDeviceInner,
    command_buffer: vk::CommandBuffer,
    pass: &HalRenderPass,
) -> Result<RenderPassTemps, HalError> {
    let vk_device = &device.device;
    let color_textures = vulkan_render_color_textures(pass)?;
    let resolve_textures = vulkan_render_resolve_textures(pass)?;
    let depth_stencil_texture = vulkan_render_depth_stencil_texture(pass)?;
    if !color_textures.iter().any(Option::is_some) && depth_stencil_texture.is_none() {
        return Err(shader_error("Vulkan render pass requires an attachment"));
    }
    let active_query = vulkan_active_occlusion_query(pass)?;
    if let Some((query_set, query_index)) = active_query {
        unsafe {
            vk_device.cmd_reset_query_pool(command_buffer, query_set.pool(), query_index, 1);
        }
    }
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
        transition_image(
            vk_device,
            command_buffer,
            texture.inner()?,
            layout,
            layout_id,
        );
    }
    for texture in resolve_textures.iter().flatten() {
        transition_image(
            vk_device,
            command_buffer,
            texture.inner()?,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            IMAGE_LAYOUT_COLOR_ATTACHMENT,
        );
    }
    if let (Some(texture), Some(attachment)) =
        (depth_stencil_texture, &pass.depth_stencil_attachment)
    {
        transition_image_aspect(
            vk_device,
            command_buffer,
            texture.inner()?,
            depth_stencil_aspect_flags(attachment.format),
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT,
        );
    }
    let color_formats = render_pass_color_formats(&pass.color_targets);
    let resolve_formats = render_pass_resolve_formats(&pass.color_targets)?;
    let render_pass = match &pass.pipeline {
        Some(crate::HalRenderPipeline::Vulkan(_)) | None => create_render_pass_for_targets(
            vk_device,
            &color_formats,
            &resolve_formats,
            &pass.color_targets,
            pass.depth_stencil_attachment.as_ref(),
            &pass.framebuffer_fetch_color_slots,
        )?,
        Some(_) => return Err(shader_error("render pipeline is not Vulkan-backed")),
    };
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
    let mut image_views = framebuffer_resources.image_views;
    let color_attachment_views = framebuffer_resources.color_attachment_views;
    let mut descriptor_pool = None;
    let mut descriptor_sets = Vec::new();
    if let Some(crate::HalRenderPipeline::Vulkan(pipeline)) = &pass.pipeline {
        descriptor_pool = create_render_descriptor_pool(vk_device, pipeline)?;
        descriptor_sets = if let Some(pool) = descriptor_pool {
            match allocate_render_descriptor_sets(vk_device, pool, pipeline) {
                Ok(sets) => sets,
                Err(error) => {
                    unsafe {
                        vk_device.destroy_descriptor_pool(pool, None);
                        vk_device.destroy_framebuffer(framebuffer, None);
                        destroy_image_views(vk_device, &image_views);
                        if let Some(render_pass) = temporary_render_pass {
                            vk_device.destroy_render_pass(render_pass, None);
                        }
                    }
                    return Err(error);
                }
            }
        } else {
            Vec::new()
        };
        match update_render_descriptor_sets(
            vk_device,
            pipeline,
            pass,
            &color_attachment_views,
            &descriptor_sets,
        ) {
            Ok(descriptor_image_views) => image_views.extend(descriptor_image_views),
            Err(error) => {
                unsafe {
                    if let Some(pool) = descriptor_pool {
                        vk_device.destroy_descriptor_pool(pool, None);
                    }
                    vk_device.destroy_framebuffer(framebuffer, None);
                    destroy_image_views(vk_device, &image_views);
                    if let Some(render_pass) = temporary_render_pass {
                        vk_device.destroy_render_pass(render_pass, None);
                    }
                }
                return Err(error);
            }
        }
    }
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
        if let Some((query_set, query_index)) = active_query {
            vk_device.cmd_begin_query(
                command_buffer,
                query_set.pool(),
                query_index,
                if device.occlusion_query_precise {
                    vk::QueryControlFlags::PRECISE
                } else {
                    vk::QueryControlFlags::empty()
                },
            );
        }
    }
    if let (Some(crate::HalRenderPipeline::Vulkan(pipeline)), Some(draw)) =
        (&pass.pipeline, pass.draw)
    {
        unsafe {
            vk_device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.inner.pipeline,
            );
            // Flip clip-space Y with a negative-height viewport (the wgpu/Dawn
            // convention: origin at `y + height`, height negated). WebGPU clip space
            // is Y-up while the Vulkan framebuffer is Y-down; doing the flip here
            // means the shader frontend (Tint) never has to flip Y in-shader, so
            // generated SPIR-V agrees with the fixed-function state. Requires
            // Vulkan 1.1 / VK_KHR_maintenance1 (MoltenVK supports it).
            let viewport = pass.viewport.map_or(
                vk::Viewport {
                    x: 0.0,
                    y: height as f32,
                    width: width as f32,
                    height: -(height as f32),
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
            let scissor = pass.scissor_rect.map_or(render_area, |rect| vk::Rect2D {
                offset: vk::Offset2D {
                    x: rect.x as i32,
                    y: rect.y as i32,
                },
                extent: vk::Extent2D {
                    width: rect.width,
                    height: rect.height,
                },
            });
            vk_device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            vk_device.cmd_set_scissor(command_buffer, 0, &[scissor]);
            // Deliver the combined immediates push-constant block (Block 94
            // S3): the pass's user immediate bytes first, then -- for the
            // `@builtin(position)` pixel-center polyfill -- the viewport
            // depth-range pair (min/max f32s) at `depth_range_offset`. The
            // pipeline layout declared a matching range over the whole
            // block; a polyfill-only pipeline (no user immediates) composes
            // exactly the bare 8-byte pair delivered before Block 94.
            if let Some(immediates) = pipeline.inner.immediates {
                let block = crate::immediates::compose_immediates_block(
                    &pass.immediate_data,
                    immediates.block_size,
                    immediates.depth_range_offset,
                    [viewport.min_depth, viewport.max_depth],
                );
                vk_device.cmd_push_constants(
                    command_buffer,
                    pipeline.inner.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &block,
                );
            }
            vk_device.cmd_set_blend_constants(command_buffer, &pass.blend_constant);
            vk_device.cmd_set_stencil_reference(
                command_buffer,
                vk::StencilFaceFlags::FRONT_AND_BACK,
                pass.stencil_reference,
            );
            bind_render_descriptor_sets(vk_device, command_buffer, pipeline, &descriptor_sets);
        }
        bind_vertex_buffers(vk_device, command_buffer, pass)?;
        encode_render_draw(vk_device, command_buffer, pass, draw)?;
    }
    unsafe {
        if let Some((query_set, query_index)) = active_query {
            vk_device.cmd_end_query(command_buffer, query_set.pool(), query_index);
        }
        vk_device.cmd_end_render_pass(command_buffer);
    }
    for texture in color_textures.iter().flatten() {
        texture
            .inner()?
            .layout
            .store(IMAGE_LAYOUT_TRANSFER_SRC, AtomicOrdering::Relaxed);
    }
    for texture in resolve_textures.iter().flatten() {
        texture
            .inner()?
            .layout
            .store(IMAGE_LAYOUT_TRANSFER_SRC, AtomicOrdering::Relaxed);
    }
    for (texture, target) in color_textures
        .iter()
        .copied()
        .zip(pass.color_targets.iter())
        .filter_map(|(texture, target)| texture.zip(target.as_ref()))
        .filter(|(_, target)| !target.store)
    {
        let inner = texture.inner()?;
        transition_image(
            vk_device,
            command_buffer,
            inner,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            IMAGE_LAYOUT_TRANSFER_DST,
        );
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
        transition_image(
            vk_device,
            command_buffer,
            inner,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            IMAGE_LAYOUT_TRANSFER_SRC,
        );
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
            let depth_stencil_aspects = depth_stencil_aspect_flags(attachment.format);
            let inner = texture.inner()?;
            transition_image_aspect(
                vk_device,
                command_buffer,
                inner,
                depth_stencil_aspects,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                IMAGE_LAYOUT_TRANSFER_DST,
            );
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
            transition_image_aspect(
                vk_device,
                command_buffer,
                inner,
                depth_stencil_aspects,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT,
            );
        }
    }
    Ok(RenderPassTemps {
        descriptor_pools: descriptor_pool.into_iter().collect(),
        framebuffer,
        image_views,
        render_pass: temporary_render_pass,
    })
}

fn encode_render_draw(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pass: &HalRenderPass,
    draw: HalDraw,
) -> Result<(), HalError> {
    match draw {
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
        HalDraw::Indexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        } => {
            bind_render_index_buffer(device, command_buffer, pass)?;
            unsafe {
                device.cmd_draw_indexed(
                    command_buffer,
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                );
            }
            Ok(())
        }
        HalDraw::Indirect { offset } => {
            let buffer = vulkan_indirect_buffer(pass)?;
            unsafe {
                device.cmd_draw_indirect(command_buffer, buffer.inner()?.buffer, offset, 1, 16);
            }
            Ok(())
        }
        HalDraw::IndexedIndirect { offset } => {
            bind_render_index_buffer(device, command_buffer, pass)?;
            let buffer = vulkan_indirect_buffer(pass)?;
            unsafe {
                device.cmd_draw_indexed_indirect(
                    command_buffer,
                    buffer.inner()?.buffer,
                    offset,
                    1,
                    20,
                );
            }
            Ok(())
        }
    }
}

fn vulkan_active_occlusion_query(
    pass: &HalRenderPass,
) -> Result<Option<(&VulkanQuerySet, u32)>, HalError> {
    let Some(query_index) = pass.occlusion_query_index else {
        return Ok(None);
    };
    let Some(query_set) = &pass.occlusion_query_set else {
        return Err(buffer_error("active occlusion query has no query set"));
    };
    let HalQuerySet::Vulkan(query_set) = query_set else {
        return Err(buffer_error("occlusion query set is not Vulkan-backed"));
    };
    query_set.validate_query(query_index)?;
    Ok(Some((query_set, query_index)))
}

fn bind_render_index_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pass: &HalRenderPass,
) -> Result<(), HalError> {
    let bound = pass
        .index_buffer
        .as_ref()
        .ok_or_else(|| buffer_error("render index buffer is missing"))?;
    let crate::HalBuffer::Vulkan(buffer) = &bound.buffer else {
        return Err(buffer_error("render index buffer is not Vulkan-backed"));
    };
    buffer.validate_range(bound.offset, bound.size)?;
    unsafe {
        device.cmd_bind_index_buffer(
            command_buffer,
            buffer.inner()?.buffer,
            bound.offset,
            vk_index_type(bound.format),
        );
    }
    Ok(())
}

fn vulkan_indirect_buffer(pass: &HalRenderPass) -> Result<&VulkanBuffer, HalError> {
    let bound = pass
        .indirect_buffer
        .as_ref()
        .ok_or_else(|| buffer_error("render indirect buffer is missing"))?;
    let crate::HalBuffer::Vulkan(buffer) = &bound.buffer else {
        return Err(buffer_error("render indirect buffer is not Vulkan-backed"));
    };
    Ok(buffer)
}

fn vk_index_type(format: HalIndexFormat) -> vk::IndexType {
    match format {
        HalIndexFormat::Uint16 => vk::IndexType::UINT16,
        HalIndexFormat::Uint32 => vk::IndexType::UINT32,
    }
}

fn vulkan_render_color_textures(
    pass: &HalRenderPass,
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
    pass: &HalRenderPass,
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
    pass: &HalRenderPass,
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

fn render_pass_clear_values(pass: &HalRenderPass) -> Vec<vk::ClearValue> {
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
    let depth_stencil = depth_stencil_aspect_flags(format);
    if depth_stencil.is_empty() {
        vk::ImageAspectFlags::COLOR
    } else {
        depth_stencil
    }
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
        texture.inner()?.image,
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
        texture.inner()?.image,
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
        texture.inner()?.image,
        format,
        depth_stencil_attachment_subresource_range(attachment),
        depth_stencil_attachment_image_view_usage(),
    )
}

fn create_attachment_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    subresource_range: vk::ImageSubresourceRange,
    usage: vk::ImageUsageFlags,
) -> Result<vk::ImageView, HalError> {
    let mut view_usage_info = vk::ImageViewUsageCreateInfo::default().usage(usage);
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(subresource_range)
        .push_next(&mut view_usage_info);
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
    let (_, block_width, block_height) = texture_block_info(copy);
    let width_blocks = div_ceil_u32(copy.extent.width, block_width);
    let height_blocks = div_ceil_u32(copy.extent.height, block_height);
    let rows = u64::from(height_blocks.saturating_sub(1));
    let last_row = rows
        .checked_mul(u64::from(copy.buffer_layout.bytes_per_row))
        .ok_or_else(|| buffer_error("buffer texture row range overflows"))?;
    let images = u64::from(copy.extent.depth_or_array_layers.saturating_sub(1));
    let last_image = images
        .checked_mul(u64::from(copy.buffer_layout.bytes_per_row))
        .and_then(|bytes| bytes.checked_mul(u64::from(copy.buffer_layout.rows_per_image)))
        .ok_or_else(|| buffer_error("buffer texture image range overflows"))?;
    let row_bytes = u64::from(width_blocks)
        .checked_mul(u64::from(texture_bytes_per_pixel(copy)?))
        .ok_or_else(|| buffer_error("buffer texture row bytes overflow"))?;
    let required = copy
        .buffer_layout
        .offset
        .checked_add(last_image)
        .and_then(|offset| offset.checked_add(last_row))
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or_else(|| buffer_error("buffer texture range overflows"))?;
    if required > buffer.size() {
        return Err(buffer_error("buffer texture range exceeds buffer size"));
    }
    Ok(())
}

/// Returns texture bytes per pixel.
pub(super) fn texture_bytes_per_pixel(copy: &HalBufferTextureCopy) -> Result<u32, HalError> {
    let crate::HalTexture::Vulkan(texture) = &copy.texture else {
        return Err(texture_error("texture is not Vulkan-backed"));
    };
    aspect_bytes_per_pixel(copy.format, copy.aspect, texture.bytes_per_pixel)
}

fn aspect_bytes_per_pixel(
    format: HalTextureFormat,
    aspect: HalTextureAspect,
    full_bytes_per_pixel: u32,
) -> Result<u32, HalError> {
    match aspect {
        HalTextureAspect::StencilOnly => Ok(1),
        HalTextureAspect::DepthOnly => match format {
            HalTextureFormat::Depth16Unorm => Ok(2),
            HalTextureFormat::Depth32Float | HalTextureFormat::Depth32FloatStencil8 => Ok(4),
            _ => full_texture_bytes_per_pixel(full_bytes_per_pixel),
        },
        HalTextureAspect::All => full_texture_bytes_per_pixel(full_bytes_per_pixel),
    }
}

fn full_texture_bytes_per_pixel(bytes_per_pixel: u32) -> Result<u32, HalError> {
    if bytes_per_pixel == 0 {
        Err(texture_error("unsupported texture format"))
    } else {
        Ok(bytes_per_pixel)
    }
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
        .image_extent(texture_copy_extent(texture.dimension, copy.extent)))
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
        HalComputePipeline, HalIndexFormat, HalRenderPass, HalSampler, HalShaderSource, HalTexture,
        HalTextureComponentSwizzle, HalTextureViewDimension,
    };
    #[cfg(feature = "tiled")]
    use crate::{
        HalSubpassAttachmentLayout, HalSubpassAttachmentResource, HalSubpassColorAttachment,
        HalSubpassDependency, HalSubpassDepthStencilAttachment, HalSubpassInputAttachment,
        HalSubpassLayout, HalSubpassPassLayout,
    };

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
        let pass = HalRenderPass {
            pipeline: None,
            color_targets: Vec::new(),
            framebuffer_fetch_color_slots: Vec::new(),
            depth_stencil_attachment: None,
            bind_buffers: Vec::new(),
            bind_textures: vec![bound_texture(HalTexture::Vulkan(texture))],
            bind_samplers: vec![bound_sampler(HalSampler::Vulkan(sampler))],
            bind_external_textures: vec![bound_external_texture(
                HalTexture::Vulkan(external_plane0),
                HalTexture::Vulkan(external_plane1),
                HalBuffer::Vulkan(params),
            )],
            vertex_buffers: Vec::new(),
            index_buffer: Some(Box::new(HalBoundIndexBuffer {
                buffer: HalBuffer::Vulkan(index),
                format: HalIndexFormat::Uint16,
                offset: 0,
                size: 16,
            })),
            indirect_buffer: Some(Box::new(HalBoundIndirectBuffer {
                buffer: HalBuffer::Vulkan(indirect),
                offset: 0,
            })),
            viewport: None,
            scissor_rect: None,
            blend_constant: [0.0; 4],
            stencil_reference: 0,
            occlusion_query_set: None,
            occlusion_query_index: None,
            draw: None,
            immediate_data: Vec::new(),
        };

        let retained = collect_retained_resources(&[HalCopy::RenderPass(pass)]);

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
    fn aspect_bytes_per_pixel_uses_copied_depth_stencil_plane_size() {
        assert_eq!(
            aspect_bytes_per_pixel(
                HalTextureFormat::Depth24PlusStencil8,
                HalTextureAspect::StencilOnly,
                5,
            )
            .expect("packed stencil byte size"),
            1
        );
        assert_eq!(
            aspect_bytes_per_pixel(
                HalTextureFormat::Depth32FloatStencil8,
                HalTextureAspect::StencilOnly,
                5,
            )
            .expect("packed stencil byte size"),
            1
        );
        assert_eq!(
            aspect_bytes_per_pixel(HalTextureFormat::Stencil8, HalTextureAspect::StencilOnly, 4,)
                .expect("stencil8 byte size"),
            1
        );
        assert_eq!(
            aspect_bytes_per_pixel(
                HalTextureFormat::Depth16Unorm,
                HalTextureAspect::DepthOnly,
                2,
            )
            .expect("depth16 byte size"),
            2
        );
        assert_eq!(
            aspect_bytes_per_pixel(
                HalTextureFormat::Depth32Float,
                HalTextureAspect::DepthOnly,
                4,
            )
            .expect("depth32 byte size"),
            4
        );
        assert_eq!(
            aspect_bytes_per_pixel(
                HalTextureFormat::Depth32FloatStencil8,
                HalTextureAspect::DepthOnly,
                5,
            )
            .expect("packed depth byte size"),
            4
        );
        assert_eq!(
            aspect_bytes_per_pixel(
                HalTextureFormat::Depth32FloatStencil8,
                HalTextureAspect::All,
                5
            )
            .expect("whole-format byte size"),
            5
        );
        assert!(
            aspect_bytes_per_pixel(HalTextureFormat::Unsupported, HalTextureAspect::All, 0)
                .is_err()
        );
    }

    #[test]
    fn buffer_image_copy_converts_compressed_layout_from_blocks_to_texels() {
        let device = noop::NoopDevice::new();
        let mut texture =
            dummy_vulkan_texture(HalTextureDimension::D2, HalTextureFormat::Bc1RgbaUnorm);
        texture.bytes_per_pixel = 8;
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
}
