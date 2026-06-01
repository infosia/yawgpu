use super::*;
#[cfg(feature = "tiled")]
use crate::format::{format_has_depth_aspect, format_has_stencil_aspect};
#[cfg(feature = "tiled")]
use crate::{HalSubpassAttachmentLayout, HalSubpassDepthStencilAttachment};

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
    .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
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
    let mut render_passes = Vec::new();
    let surface_pending = find_surface_pending(copies);
    let result = (|| {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers =
            unsafe { queue.device.device.allocate_command_buffers(&allocate_info) }
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
        let Some(&command_buffer) = command_buffers.first() else {
            return Err(HalError::QueueSubmissionFailed { backend: BACKEND });
        };
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            queue
                .device
                .device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
        }
        for copy in copies {
            match copy {
                HalCopy::Buffer(copy) => {
                    encode_buffer_copy(&queue.device.device, command_buffer, copy)?;
                }
                HalCopy::BufferClear(clear) => {
                    encode_buffer_clear(&queue.device.device, command_buffer, clear)?;
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
                    if let Some(pool) =
                        encode_compute_pass(&queue.device.device, command_buffer, pass)?
                    {
                        descriptor_pools.push(pool);
                    }
                }
                HalCopy::RenderPass(pass) => {
                    let temps = encode_render_pass(&queue.device.device, command_buffer, pass)?;
                    descriptor_pools.extend(temps.descriptor_pools);
                    framebuffers.push(temps.framebuffer);
                    if let Some(render_pass) = temps.render_pass {
                        render_passes.push(render_pass);
                    }
                }
                #[cfg(feature = "tiled")]
                HalCopy::SubpassRenderPass(pass) => {
                    let temps = encode_subpass_render_pass(&queue.device, command_buffer, pass)?;
                    descriptor_pools.extend(temps.descriptor_pools);
                    framebuffers.push(temps.framebuffer);
                }
            }
        }
        unsafe {
            queue
                .device
                .device
                .end_command_buffer(command_buffer)
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
            let command_buffers = [command_buffer];
            let mut wait_semaphores = Vec::new();
            let mut wait_stages = Vec::new();
            let mut signal_semaphores = Vec::new();
            let mut surface_retire = None;
            if let Some(pending_state) = surface_pending.as_ref() {
                let mut state = pending_state
                    .lock()
                    .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
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
            let fence = queue
                .device
                .device
                .create_fence(&fence_info, None)
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            queue
                .device
                .device
                .queue_submit(queue.queue, &[submit_info], fence)
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
            let cleanup = retire_ops(command_pool, descriptor_pools, framebuffers, render_passes);
            if let Some(pending_state) = surface_retire {
                pending_state
                    .lock()
                    .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?
                    .retire
                    .retire(&queue.device.device, fence, cleanup, true)?;
            } else {
                queue
                    .retire
                    .lock()
                    .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?
                    .retire(&queue.device.device, fence, cleanup, true)?;
            }
        }
        Ok(())
    })();
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
        let mut state = pending_state.lock().map_err(|_| HalError::PresentFailed {
            backend: BACKEND,
            message: "surface pending state lock failed",
        })?;
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
                .map_err(|_| HalError::PresentFailed {
                    backend: BACKEND,
                    message: "surface pending state lock failed",
                })?
                .retire
                .retire(
                    &queue.inner.device.device,
                    fence,
                    vec![RetireOp::CommandBuffer {
                        pool: command_pool,
                        buffer: command_buffer,
                    }],
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
    render_passes: Vec<vk::RenderPass>,
) -> Vec<RetireOp> {
    let mut cleanup = Vec::new();
    cleanup.push(RetireOp::CommandPool(command_pool));
    cleanup.extend(descriptor_pools.into_iter().map(RetireOp::DescriptorPool));
    cleanup.extend(framebuffers.into_iter().map(RetireOp::Framebuffer));
    cleanup.extend(render_passes.into_iter().map(RetireOp::RenderPass));
    cleanup
}

fn find_surface_pending(copies: &[HalCopy]) -> Option<Arc<Mutex<SurfacePendingState>>> {
    copies.iter().find_map(surface_pending_from_copy)
}

fn surface_pending_from_copy(copy: &HalCopy) -> Option<Arc<Mutex<SurfacePendingState>>> {
    match copy {
        HalCopy::Buffer(_) | HalCopy::BufferClear(_) | HalCopy::ComputePass(_) => None,
        HalCopy::BufferToTexture(copy) | HalCopy::TextureToBuffer(copy) => {
            surface_pending_from_hal_texture(&copy.texture)
        }
        HalCopy::TextureToTexture(copy) => surface_pending_from_hal_texture(&copy.source)
            .or_else(|| surface_pending_from_hal_texture(&copy.destination)),
        HalCopy::RenderPass(pass) => surface_pending_from_hal_texture(&pass.color_target.texture),
        #[cfg(feature = "tiled")]
        HalCopy::SubpassRenderPass(pass) => surface_pending_from_subpass(pass),
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
    pass: &HalSubpassRenderPassCommand,
) -> Option<Arc<Mutex<SurfacePendingState>>> {
    pass.color_attachments
        .iter()
        .find_map(|attachment| surface_pending_from_attachment_resource(&attachment.resource))
        .or_else(|| {
            pass.depth_stencil_attachment
                .as_ref()
                .and_then(|attachment| {
                    surface_pending_from_attachment_resource(&attachment.resource)
                })
        })
}

#[cfg(feature = "tiled")]
fn surface_pending_from_attachment_resource(
    resource: &HalSubpassAttachmentResource,
) -> Option<Arc<Mutex<SurfacePendingState>>> {
    match resource {
        HalSubpassAttachmentResource::Persistent {
            texture,
            resolve_target,
        } => surface_pending_from_hal_texture(texture).or_else(|| {
            resolve_target
                .as_ref()
                .and_then(surface_pending_from_hal_texture)
        }),
        HalSubpassAttachmentResource::Transient(_) => None,
    }
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
    validate_mip_level(copy.mip_level)?;
    texture.validate_origin_extent(copy.origin, copy.extent)?;
    validate_buffer_texture_range(buffer, copy)?;
    let buffer = buffer.inner()?;
    let texture_inner = texture.inner()?;
    transition_image(
        device,
        command_buffer,
        texture_inner,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_DST,
    );
    let region = buffer_image_copy(copy, texture.bytes_per_pixel)?;
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
    validate_mip_level(copy.mip_level)?;
    texture.validate_origin_extent(copy.origin, copy.extent)?;
    validate_buffer_texture_range(buffer, copy)?;
    let buffer = buffer.inner()?;
    let texture_inner = texture.inner()?;
    transition_image(
        device,
        command_buffer,
        texture_inner,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_SRC,
    );
    let region = buffer_image_copy(copy, texture.bytes_per_pixel)?;
    unsafe {
        device.cmd_copy_image_to_buffer(
            command_buffer,
            texture_inner.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer.buffer,
            &[region],
        );
    }
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
    validate_mip_level(copy.source_mip_level)?;
    validate_mip_level(copy.destination_mip_level)?;
    source.validate_origin_extent(copy.source_origin, copy.extent)?;
    destination.validate_origin_extent(copy.destination_origin, copy.extent)?;
    let source_inner = source.inner()?;
    let destination_inner = destination.inner()?;
    transition_image(
        device,
        command_buffer,
        source_inner,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_SRC,
    );
    transition_image(
        device,
        command_buffer,
        destination_inner,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_DST,
    );
    let region = vk::ImageCopy::default()
        .src_subresource(image_subresource_layers())
        .src_offset(to_image_offset(
            copy.source_origin.x,
            copy.source_origin.y,
            copy.source_origin.z,
        )?)
        .dst_subresource(image_subresource_layers())
        .dst_offset(to_image_offset(
            copy.destination_origin.x,
            copy.destination_origin.y,
            copy.destination_origin.z,
        )?)
        .extent(to_image_extent(copy.extent));
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
) -> Result<Option<vk::DescriptorPool>, HalError> {
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
    if let Err(error) = update_compute_descriptor_sets(device, pipeline, pass, &descriptor_sets) {
        if let Some(pool) = descriptor_pool {
            unsafe {
                device.destroy_descriptor_pool(pool, None);
            }
        }
        return Err(error);
    }
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
        device.cmd_dispatch(
            command_buffer,
            pass.workgroups.0,
            pass.workgroups.1,
            pass.workgroups.2,
        );
    }
    compute_to_transfer_barrier(device, command_buffer);
    Ok(descriptor_pool)
}

/// Stores render pass temps data used by validation and backend submission.
pub(super) struct RenderPassTemps {
    descriptor_pools: Vec<vk::DescriptorPool>,
    framebuffer: vk::Framebuffer,
    render_pass: Option<vk::RenderPass>,
}

/// Records a tiled subpass render pass into the command stream.
#[cfg(feature = "tiled")]
pub(super) fn encode_subpass_render_pass(
    device: &VulkanDeviceInner,
    command_buffer: vk::CommandBuffer,
    pass: &HalSubpassRenderPassCommand,
) -> Result<RenderPassTemps, HalError> {
    let render_pass = cached_subpass_render_pass(device, pass)?;
    let (views, persistent_textures) = subpass_attachment_views(pass)?;
    for texture in &persistent_textures {
        transition_image(
            &device.device,
            command_buffer,
            texture.inner()?,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            IMAGE_LAYOUT_COLOR_ATTACHMENT,
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
    for subpass_index in 0..pass.layout.subpasses.len() {
        for draw in pass
            .draws
            .iter()
            .filter(|draw| draw.subpass_index as usize == subpass_index)
        {
            if let Some(pool) =
                encode_subpass_draw(&device.device, command_buffer, pass, draw, &views)?
            {
                descriptor_pools.push(pool);
            }
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
    for texture in persistent_textures {
        texture
            .inner()?
            .layout
            .store(IMAGE_LAYOUT_TRANSFER_SRC, AtomicOrdering::Relaxed);
    }
    Ok(RenderPassTemps {
        descriptor_pools,
        framebuffer,
        render_pass: None,
    })
}

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
    views: &[vk::ImageView],
) -> Result<Option<vk::DescriptorPool>, HalError> {
    let HalRenderPipeline::Vulkan(pipeline) = &draw.pipeline else {
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
    if let Err(error) =
        update_subpass_descriptor_sets(device, pipeline, pass, draw, &descriptor_sets, views)
    {
        if let Some(pool) = descriptor_pool {
            unsafe {
                device.destroy_descriptor_pool(pool, None);
            }
        }
        return Err(error);
    }
    unsafe {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.inner.pipeline,
        );
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: pass.extent.width as f32,
            height: pass.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: pass.extent.width,
                height: pass.extent.height,
            },
        };
        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
        device.cmd_set_scissor(command_buffer, 0, &[scissor]);
        bind_render_descriptor_sets(device, command_buffer, pipeline, &descriptor_sets);
    }
    bind_subpass_vertex_buffers(device, command_buffer, draw)?;
    unsafe {
        device.cmd_draw(
            command_buffer,
            draw.draw.vertex_count,
            draw.draw.instance_count,
            draw.draw.first_vertex,
            draw.draw.first_instance,
        );
    }
    Ok(descriptor_pool)
}

#[cfg(feature = "tiled")]
fn update_subpass_descriptor_sets(
    device: &ash::Device,
    pipeline: &VulkanRenderPipeline,
    pass: &HalSubpassRenderPassCommand,
    draw: &HalSubpassDraw,
    descriptor_sets: &[vk::DescriptorSet],
    views: &[vk::ImageView],
) -> Result<(), HalError> {
    if pipeline.inner.descriptor_bindings.is_empty() {
        return Ok(());
    }
    let subpass_inputs = pass
        .layout
        .subpasses
        .get(draw.subpass_index as usize)
        .map(|subpass| subpass.input_attachments.as_slice())
        .unwrap_or(&[]);
    // Tracks whether a write references `buffer_infos` or `image_infos`; both Vecs
    // are fully built before any `WriteDescriptorSet` borrows into them.
    enum DescriptorInfo {
        Buffer(usize),
        Image(usize),
    }
    let mut buffer_infos = Vec::new();
    let mut image_infos = Vec::new();
    let mut write_specs = Vec::new();
    for descriptor in &pipeline.inner.descriptor_bindings {
        match descriptor.kind {
            HalBufferBindingKind::Uniform | HalBufferBindingKind::Storage => {
                let bound = draw
                    .bind_buffers
                    .iter()
                    .find(|bound| {
                        bound.group == descriptor.group && bound.binding == descriptor.binding
                    })
                    .ok_or_else(|| shader_error("subpass descriptor binding is missing"))?;
                buffer_infos.push(descriptor_buffer_info(bound)?);
                write_specs.push((
                    DescriptorInfo::Buffer(buffer_infos.len() - 1),
                    descriptor.group,
                    descriptor.binding,
                    descriptor_type(descriptor.kind),
                ));
            }
            HalBufferBindingKind::InputAttachment => {
                let input = subpass_inputs
                    .iter()
                    .find(|input| {
                        input.group == descriptor.group && input.binding == descriptor.binding
                    })
                    .ok_or_else(|| shader_error("subpass input attachment mapping is missing"))?;
                let (view_index, image_layout) = if input.source_attachment == u32::MAX {
                    (
                        pass.layout.color_attachments.len(),
                        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    )
                } else {
                    (
                        input.source_attachment as usize,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    )
                };
                let view = views
                    .get(view_index)
                    .copied()
                    .ok_or_else(|| shader_error("subpass input attachment view is missing"))?;
                image_infos.push(
                    vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(image_layout),
                );
                write_specs.push((
                    DescriptorInfo::Image(image_infos.len() - 1),
                    descriptor.group,
                    descriptor.binding,
                    vk::DescriptorType::INPUT_ATTACHMENT,
                ));
            }
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
        let inner = buffer.inner()?;
        validate_bound_buffer_range(bound)?;
        let buffers = [inner.buffer];
        let offsets = [bound.offset];
        unsafe {
            device.cmd_bind_vertex_buffers(command_buffer, bound.binding, &buffers, &offsets);
        }
    }
    Ok(())
}

#[cfg(feature = "tiled")]
fn subpass_attachment_views(
    pass: &HalSubpassRenderPassCommand,
) -> Result<(Vec<vk::ImageView>, Vec<VulkanTexture>), HalError> {
    let mut views = Vec::new();
    let mut persistent_textures = Vec::new();
    for attachment in &pass.color_attachments {
        let (view, persistent) = subpass_attachment_view(&attachment.resource)?;
        views.push(view);
        if let Some(texture) = persistent {
            persistent_textures.push(texture);
        }
    }
    if let Some(depth) = &pass.depth_stencil_attachment {
        let (view, persistent) = subpass_attachment_view(&depth.resource)?;
        views.push(view);
        if let Some(texture) = persistent {
            persistent_textures.push(texture);
        }
    }
    Ok((views, persistent_textures))
}

#[cfg(feature = "tiled")]
fn subpass_attachment_view(
    resource: &HalSubpassAttachmentResource,
) -> Result<(vk::ImageView, Option<VulkanTexture>), HalError> {
    match resource {
        HalSubpassAttachmentResource::Persistent { texture, .. } => {
            let HalTexture::Vulkan(texture) = texture else {
                return Err(texture_error("subpass attachment is not Vulkan-backed"));
            };
            Ok((texture.inner()?.view, Some(texture.clone())))
        }
        HalSubpassAttachmentResource::Transient(attachment) => {
            let HalTransientAttachment::Vulkan(attachment) = attachment else {
                return Err(texture_error("subpass transient is not Vulkan-backed"));
            };
            Ok((attachment._inner.view, None))
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
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk_sample_count(layout.sample_count)?)
                .load_op(vk_load_op(binding.load_op))
                .store_op(if binding.store {
                    vk::AttachmentStoreOp::STORE
                } else {
                    vk::AttachmentStoreOp::DONT_CARE
                })
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        );
    }
    if let Some(layout) = pass.layout.depth_stencil_attachment {
        let binding = pass
            .depth_stencil_attachment
            .as_ref()
            .ok_or_else(|| shader_error("subpass depth-stencil attachment binding missing"))?;
        attachments.push(vk_depth_stencil_attachment_description(layout, binding)?);
    }
    let depth_index = pass.layout.color_attachments.len() as u32;
    let color_refs = pass
        .layout
        .subpasses
        .iter()
        .map(|subpass| {
            subpass
                .color_attachment_indices
                .iter()
                .map(|&attachment| {
                    vk::AttachmentReference::default()
                        .attachment(attachment)
                        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let input_refs = pass
        .layout
        .subpasses
        .iter()
        .map(|subpass| {
            subpass
                .input_attachments
                .iter()
                .map(|input| {
                    let (attachment, layout) = if input.source_attachment == u32::MAX {
                        (
                            depth_index,
                            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                        )
                    } else {
                        (
                            input.source_attachment,
                            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        )
                    };
                    vk::AttachmentReference::default()
                        .attachment(attachment)
                        .layout(layout)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let depth_refs = pass
        .layout
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
    for (index, subpass) in pass.layout.subpasses.iter().enumerate() {
        let mut description = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs[index])
            .input_attachments(&input_refs[index]);
        if let Some(depth_ref) = depth_refs[index].as_ref() {
            description = description.depth_stencil_attachment(depth_ref);
        }
        let _ = subpass;
        subpasses.push(description);
    }
    let dependencies = subpass_dependencies(pass);
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    unsafe { device.create_render_pass(&render_pass_info, None) }
        .map_err(|_| shader_error("subpass render pass creation failed"))
}

#[cfg(feature = "tiled")]
fn subpass_dependencies(pass: &HalSubpassRenderPassCommand) -> Vec<vk::SubpassDependency> {
    let mut dependencies = vec![vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
    dependencies.extend(pass.layout.dependencies.iter().map(|dependency| {
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
            .src_subpass(pass.layout.subpasses.len().saturating_sub(1) as u32)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
    );
    dependencies
}

#[cfg(feature = "tiled")]
fn subpass_clear_values(pass: &HalSubpassRenderPassCommand) -> Vec<vk::ClearValue> {
    let mut values = pass
        .color_attachments
        .iter()
        .map(|attachment| vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [
                    attachment.clear_color[0] as f32,
                    attachment.clear_color[1] as f32,
                    attachment.clear_color[2] as f32,
                    attachment.clear_color[3] as f32,
                ],
            },
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
fn vk_load_op(load_op: HalRenderLoadOp) -> vk::AttachmentLoadOp {
    match load_op {
        HalRenderLoadOp::Load => vk::AttachmentLoadOp::LOAD,
        HalRenderLoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
    }
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

#[cfg(all(test, feature = "tiled"))]
mod tiled_tests {
    use super::*;
    use crate::HalSubpassAttachmentResource;

    fn dummy_depth_binding() -> HalSubpassDepthStencilAttachment {
        HalSubpassDepthStencilAttachment {
            resource: HalSubpassAttachmentResource::Transient(HalTransientAttachment::Noop(
                crate::noop::NoopTransientAttachment,
            )),
            depth_load_op: HalRenderLoadOp::Load,
            depth_store: true,
            depth_clear_value: 1.0,
            stencil_load_op: HalRenderLoadOp::Clear,
            stencil_store: true,
            stencil_clear_value: 7,
        }
    }

    #[test]
    fn depth_stencil_attachment_description_uses_binding_ops_by_aspect() {
        let depth_only = vk_depth_stencil_attachment_description(
            HalSubpassAttachmentLayout {
                format: HalTextureFormat::Depth32Float,
                sample_count: 1,
            },
            &dummy_depth_binding(),
        )
        .expect("depth-only description");
        assert_eq!(depth_only.load_op, vk::AttachmentLoadOp::LOAD);
        assert_eq!(depth_only.store_op, vk::AttachmentStoreOp::STORE);
        assert_eq!(depth_only.stencil_load_op, vk::AttachmentLoadOp::DONT_CARE);
        assert_eq!(
            depth_only.stencil_store_op,
            vk::AttachmentStoreOp::DONT_CARE
        );

        let depth_stencil = vk_depth_stencil_attachment_description(
            HalSubpassAttachmentLayout {
                format: HalTextureFormat::Depth24PlusStencil8,
                sample_count: 1,
            },
            &dummy_depth_binding(),
        )
        .expect("depth-stencil description");
        assert_eq!(depth_stencil.load_op, vk::AttachmentLoadOp::LOAD);
        assert_eq!(depth_stencil.store_op, vk::AttachmentStoreOp::STORE);
        assert_eq!(depth_stencil.stencil_load_op, vk::AttachmentLoadOp::CLEAR);
        assert_eq!(depth_stencil.stencil_store_op, vk::AttachmentStoreOp::STORE);
    }
}

#[cfg(feature = "tiled")]
fn vk_sample_count(sample_count: u32) -> Result<vk::SampleCountFlags, HalError> {
    match sample_count {
        1 => Ok(vk::SampleCountFlags::TYPE_1),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        _ => Err(texture_error("unsupported subpass sample count")),
    }
}

/// Records encode into the command stream.
pub(super) fn encode_render_pass(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pass: &HalRenderPass,
) -> Result<RenderPassTemps, HalError> {
    let crate::HalTexture::Vulkan(texture) = &pass.color_target.texture else {
        return Err(texture_error("render target is not Vulkan-backed"));
    };
    if !matches!(pass.color_target.load_op, HalRenderLoadOp::Clear) {
        return Err(shader_error("Vulkan render pass load op is unsupported"));
    }
    if !pass.color_target.store {
        return Err(shader_error(
            "Vulkan render pass discard store op is unsupported",
        ));
    }
    let texture_inner = texture.inner()?;
    transition_image(
        device,
        command_buffer,
        texture_inner,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        IMAGE_LAYOUT_COLOR_ATTACHMENT,
    );
    let render_pass = match &pass.pipeline {
        Some(crate::HalRenderPipeline::Vulkan(pipeline)) => pipeline.inner.render_pass,
        Some(_) => return Err(shader_error("render pipeline is not Vulkan-backed")),
        None => create_render_pass_for_format(device, texture.format)?,
    };
    let temporary_render_pass = pass.pipeline.is_none().then_some(render_pass);
    let framebuffer = create_framebuffer(device, render_pass, texture)?;
    let mut descriptor_pool = None;
    let mut descriptor_sets = Vec::new();
    if let Some(crate::HalRenderPipeline::Vulkan(pipeline)) = &pass.pipeline {
        descriptor_pool = create_render_descriptor_pool(device, pipeline)?;
        descriptor_sets = if let Some(pool) = descriptor_pool {
            match allocate_render_descriptor_sets(device, pool, pipeline) {
                Ok(sets) => sets,
                Err(error) => {
                    unsafe {
                        device.destroy_descriptor_pool(pool, None);
                        device.destroy_framebuffer(framebuffer, None);
                        if let Some(render_pass) = temporary_render_pass {
                            device.destroy_render_pass(render_pass, None);
                        }
                    }
                    return Err(error);
                }
            }
        } else {
            Vec::new()
        };
        if let Err(error) = update_render_descriptor_sets(device, pipeline, pass, &descriptor_sets)
        {
            unsafe {
                if let Some(pool) = descriptor_pool {
                    device.destroy_descriptor_pool(pool, None);
                }
                device.destroy_framebuffer(framebuffer, None);
                if let Some(render_pass) = temporary_render_pass {
                    device.destroy_render_pass(render_pass, None);
                }
            }
            return Err(error);
        }
    }
    let clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [
                pass.color_target.clear_color[0] as f32,
                pass.color_target.clear_color[1] as f32,
                pass.color_target.clear_color[2] as f32,
                pass.color_target.clear_color[3] as f32,
            ],
        },
    }];
    let render_area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: texture.width,
            height: texture.height,
        },
    };
    let begin_info = vk::RenderPassBeginInfo::default()
        .render_pass(render_pass)
        .framebuffer(framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);
    unsafe {
        device.cmd_begin_render_pass(command_buffer, &begin_info, vk::SubpassContents::INLINE);
    }
    if let (Some(crate::HalRenderPipeline::Vulkan(pipeline)), Some(draw)) =
        (&pass.pipeline, pass.draw)
    {
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.inner.pipeline,
            );
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: texture.width as f32,
                height: texture.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[render_area]);
            bind_render_descriptor_sets(device, command_buffer, pipeline, &descriptor_sets);
        }
        bind_vertex_buffers(device, command_buffer, pass)?;
        unsafe {
            device.cmd_draw(
                command_buffer,
                draw.vertex_count,
                draw.instance_count,
                draw.first_vertex,
                draw.first_instance,
            );
        }
    }
    unsafe {
        device.cmd_end_render_pass(command_buffer);
    }
    texture_inner
        .layout
        .store(IMAGE_LAYOUT_TRANSFER_SRC, AtomicOrdering::Relaxed);
    Ok(RenderPassTemps {
        descriptor_pools: descriptor_pool.into_iter().collect(),
        framebuffer,
        render_pass: temporary_render_pass,
    })
}

/// Creates framebuffer and reports validation errors through the owning device.
pub(super) fn create_framebuffer(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    texture: &VulkanTexture,
) -> Result<vk::Framebuffer, HalError> {
    let inner = texture.inner()?;
    let attachments = [inner.view];
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(&attachments)
        .width(texture.width)
        .height(texture.height)
        .layers(1);
    unsafe { device.create_framebuffer(&framebuffer_info, None) }
        .map_err(|_| shader_error("framebuffer creation failed"))
}

/// Validates buffer texture range and returns a descriptive error on failure.
pub(super) fn validate_buffer_texture_range(
    buffer: &VulkanBuffer,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let rows = u64::from(copy.extent.height.saturating_sub(1));
    let last_row = rows
        .checked_mul(u64::from(copy.buffer_layout.bytes_per_row))
        .ok_or_else(|| buffer_error("buffer texture row range overflows"))?;
    let row_bytes = u64::from(copy.extent.width)
        .checked_mul(u64::from(texture_bytes_per_pixel(copy)?))
        .ok_or_else(|| buffer_error("buffer texture row bytes overflow"))?;
    let required = copy
        .buffer_layout
        .offset
        .checked_add(last_row)
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
    if texture.bytes_per_pixel == 0 {
        return Err(texture_error("unsupported texture format"));
    }
    Ok(texture.bytes_per_pixel)
}

/// Returns buffer image copy.
pub(super) fn buffer_image_copy(
    copy: &HalBufferTextureCopy,
    bytes_per_pixel: u32,
) -> Result<vk::BufferImageCopy, HalError> {
    let buffer_row_length = buffer_row_length(copy.buffer_layout.bytes_per_row, bytes_per_pixel)?;
    Ok(vk::BufferImageCopy::default()
        .buffer_offset(copy.buffer_layout.offset)
        .buffer_row_length(buffer_row_length)
        .buffer_image_height(copy.buffer_layout.rows_per_image)
        .image_subresource(image_subresource_layers())
        .image_offset(to_image_offset(
            copy.origin.x,
            copy.origin.y,
            copy.origin.z,
        )?)
        .image_extent(to_image_extent(copy.extent)))
}

/// Validates mip level and returns a descriptive error on failure.
pub(super) fn validate_mip_level(mip_level: u32) -> Result<(), HalError> {
    if mip_level != 0 {
        return Err(texture_error("unsupported texture mip level"));
    }
    Ok(())
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
