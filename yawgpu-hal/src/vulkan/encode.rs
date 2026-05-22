use super::*;

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
    let result = record_and_submit_copies(queue, command_pool, copies);
    unsafe {
        queue.device.device.destroy_command_pool(command_pool, None);
    }
    result
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
                    if let Some(pool) = temps.descriptor_pool {
                        descriptor_pools.push(pool);
                    }
                    framebuffers.push(temps.framebuffer);
                    if let Some(render_pass) = temps.render_pass {
                        render_passes.push(render_pass);
                    }
                }
                #[cfg(feature = "tiled")]
                HalCopy::SubpassRenderPass(pass) => {
                    let temps = encode_subpass_render_pass(&queue.device, command_buffer, pass)?;
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
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            queue
                .device
                .device
                .queue_submit(queue.queue, &[submit_info], vk::Fence::null())
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
            queue
                .device
                .device
                .queue_wait_idle(queue.queue)
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
        }
        Ok(())
    })();
    unsafe {
        for framebuffer in framebuffers {
            queue.device.device.destroy_framebuffer(framebuffer, None);
        }
        for render_pass in render_passes {
            queue.device.device.destroy_render_pass(render_pass, None);
        }
        for pool in descriptor_pools {
            queue.device.device.destroy_descriptor_pool(pool, None);
        }
    }
    result
}

/// Returns transition swapchain image to present.
pub(super) fn transition_swapchain_image_to_present(
    queue: &VulkanQueue,
    texture: &VulkanTexture,
) -> Result<(), HalError> {
    let inner = texture.inner()?;
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
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            queue
                .inner
                .device
                .device
                .queue_submit(queue.inner.queue, &[submit_info], vk::Fence::null())
                .map_err(|_| HalError::PresentFailed {
                    backend: BACKEND,
                    message: "queue submit failed",
                })?;
            queue
                .inner
                .device
                .device
                .queue_wait_idle(queue.inner.queue)
                .map_err(|_| HalError::PresentFailed {
                    backend: BACKEND,
                    message: "queue wait failed",
                })?;
        }
        Ok(())
    })();
    unsafe {
        queue
            .inner
            .device
            .device
            .destroy_command_pool(command_pool, None);
    }
    result
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
    descriptor_pool: Option<vk::DescriptorPool>,
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
        for _ in 1..pass.layout.subpasses.len() {
            device
                .device
                .cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);
        }
        device.device.cmd_end_render_pass(command_buffer);
    }
    for texture in persistent_textures {
        texture
            .inner()?
            .layout
            .store(IMAGE_LAYOUT_TRANSFER_SRC, AtomicOrdering::Relaxed);
    }
    Ok(RenderPassTemps {
        descriptor_pool: None,
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
                .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        );
    }
    if let Some(layout) = pass.layout.depth_stencil_attachment {
        let (format, _) = map_texture_format(layout.format)?;
        attachments.push(
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk_sample_count(layout.sample_count)?)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
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
        descriptor_pool,
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
