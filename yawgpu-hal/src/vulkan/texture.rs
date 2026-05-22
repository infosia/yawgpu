use super::*;

/// Constant value for image layout undefined.
pub(super) const IMAGE_LAYOUT_UNDEFINED: u8 = 0;
/// Constant value for image layout transfer dst.
pub(super) const IMAGE_LAYOUT_TRANSFER_DST: u8 = 1;
/// Constant value for image layout transfer src.
pub(super) const IMAGE_LAYOUT_TRANSFER_SRC: u8 = 2;
/// Constant value for image layout color attachment.
pub(super) const IMAGE_LAYOUT_COLOR_ATTACHMENT: u8 = 3;
/// Constant value for image layout present.
pub(super) const IMAGE_LAYOUT_PRESENT: u8 = 4;

/// Stores vulkan texture data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct VulkanTexture {
    pub(super) inner: Option<Arc<VulkanTextureInner>>,
    pub(super) swapchain: Option<Arc<VulkanSwapchainInner>>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) depth_or_array_layers: u32,
    pub(super) bytes_per_pixel: u32,
    pub(super) format: HalTextureFormat,
}

/// Stores vulkan transient attachment data used by tiled rendering.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub struct VulkanTransientAttachment {
    pub(super) _inner: Arc<VulkanTextureInner>,
    pub(super) _format: HalTextureFormat,
    pub(super) _width: u32,
    pub(super) _height: u32,
    pub(super) _sample_count: u32,
    pub(super) _lazily_allocated: bool,
}

impl VulkanTexture {
    /// Returns the backing texture state, or an error if creation failed.
    pub(super) fn inner(&self) -> Result<&VulkanTextureInner, HalError> {
        self.inner
            .as_deref()
            .ok_or_else(|| texture_error("texture allocation failed or unsupported descriptor"))
    }

    /// Validates origin extent and returns a descriptive error on failure.
    pub(super) fn validate_origin_extent(
        &self,
        origin: crate::HalOrigin3d,
        extent: HalExtent3d,
    ) -> Result<(), HalError> {
        let x_end = origin
            .x
            .checked_add(extent.width)
            .ok_or_else(|| texture_error("texture x range overflows"))?;
        let y_end = origin
            .y
            .checked_add(extent.height)
            .ok_or_else(|| texture_error("texture y range overflows"))?;
        let z_end = origin
            .z
            .checked_add(extent.depth_or_array_layers)
            .ok_or_else(|| texture_error("texture z range overflows"))?;
        if x_end > self.width || y_end > self.height || z_end > self.depth_or_array_layers {
            return Err(texture_error("texture range exceeds texture size"));
        }
        Ok(())
    }
}

/// Holds shared state for the vulkan texture handle.
#[derive(Debug)]
pub(super) struct VulkanTextureInner {
    pub(super) device: Arc<VulkanDeviceInner>,
    pub(super) image: vk::Image,
    pub(super) view: vk::ImageView,
    pub(super) memory: Option<vk::DeviceMemory>,
    pub(super) owns_image: bool,
    pub(super) layout: AtomicU8,
}

impl Drop for VulkanTextureInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_image_view(self.view, None);
            if self.owns_image {
                self.device.device.destroy_image(self.image, None);
            }
            if let Some(memory) = self.memory {
                self.device.device.free_memory(memory, None);
            }
        }
    }
}

/// Stores vulkan sampler data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct VulkanSampler {
    pub(super) _inner: Option<Arc<VulkanSamplerInner>>,
}

/// Holds shared state for the vulkan sampler handle.
#[derive(Debug)]
pub(super) struct VulkanSamplerInner {
    pub(super) device: Arc<VulkanDeviceInner>,
    pub(super) sampler: vk::Sampler,
}

impl Drop for VulkanSamplerInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_sampler(self.sampler, None);
        }
    }
}

/// Creates texture and reports validation errors through the owning device.
pub(super) fn create_texture(
    device: Arc<VulkanDeviceInner>,
    descriptor: &HalTextureDescriptor,
) -> Result<(VulkanTextureInner, u32), HalError> {
    if descriptor.depth_or_array_layers != 1
        || descriptor.mip_level_count != 1
        || descriptor.sample_count != 1
    {
        return Err(texture_error("unsupported texture descriptor"));
    }
    let (format, bytes_per_pixel) = map_texture_format(descriptor.format)?;
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: descriptor.width,
            height: descriptor.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(map_texture_usage(descriptor.usage))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.device.create_image(&image_info, None) }
        .map_err(|_| texture_error("image creation failed"))?;
    let requirements = unsafe { device.device.get_image_memory_requirements(image) };
    let memory_type_index = find_memory_type_index(
        &device.memory_properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        unsafe {
            device.device.destroy_image(image, None);
        }
        texture_error("compatible image memory type not found")
    })?;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = unsafe { device.device.allocate_memory(&allocate_info, None) }.map_err(|_| {
        unsafe {
            device.device.destroy_image(image, None);
        }
        texture_error("image memory allocation failed")
    })?;
    if let Err(error) = unsafe { device.device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.device.destroy_image(image, None);
            device.device.free_memory(memory, None);
        }
        return Err(map_texture_error(error, "image memory bind failed"));
    }
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(color_subresource_range());
    let view = unsafe { device.device.create_image_view(&view_info, None) }.map_err(|_| {
        unsafe {
            device.device.destroy_image(image, None);
            device.device.free_memory(memory, None);
        }
        texture_error("image view creation failed")
    })?;
    Ok((
        VulkanTextureInner {
            device,
            image,
            view,
            memory: Some(memory),
            owns_image: true,
            layout: AtomicU8::new(IMAGE_LAYOUT_UNDEFINED),
        },
        bytes_per_pixel,
    ))
}

/// Creates a transient attachment image and view.
#[cfg(feature = "tiled")]
pub(super) fn create_transient_attachment(
    device: Arc<VulkanDeviceInner>,
    descriptor: &HalTransientAttachmentDescriptor,
) -> Result<VulkanTransientAttachment, HalError> {
    let (format, _) = map_texture_format(descriptor.format)?;
    let samples = sample_count_flags(descriptor.sample_count)?;
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: descriptor.width,
            height: descriptor.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(samples)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                | vk::ImageUsageFlags::INPUT_ATTACHMENT
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.device.create_image(&image_info, None) }
        .map_err(|_| texture_error("transient image creation failed"))?;
    let requirements = unsafe { device.device.get_image_memory_requirements(image) };
    let lazy_memory = find_memory_type_index(
        &device.memory_properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
    );
    let lazily_allocated = lazy_memory.is_some();
    let memory_type_index = lazy_memory
        .or_else(|| {
            find_memory_type_index(
                &device.memory_properties,
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
        })
        .ok_or_else(|| {
            unsafe {
                device.device.destroy_image(image, None);
            }
            texture_error("compatible transient image memory type not found")
        })?;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = unsafe { device.device.allocate_memory(&allocate_info, None) }.map_err(|_| {
        unsafe {
            device.device.destroy_image(image, None);
        }
        texture_error("transient image memory allocation failed")
    })?;
    if let Err(error) = unsafe { device.device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.device.destroy_image(image, None);
            device.device.free_memory(memory, None);
        }
        return Err(map_texture_error(
            error,
            "transient image memory bind failed",
        ));
    }
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(color_subresource_range());
    let view = unsafe { device.device.create_image_view(&view_info, None) }.map_err(|_| {
        unsafe {
            device.device.destroy_image(image, None);
            device.device.free_memory(memory, None);
        }
        texture_error("transient image view creation failed")
    })?;
    Ok(VulkanTransientAttachment {
        _inner: Arc::new(VulkanTextureInner {
            device,
            image,
            view,
            memory: Some(memory),
            owns_image: true,
            layout: AtomicU8::new(IMAGE_LAYOUT_UNDEFINED),
        }),
        _format: descriptor.format,
        _width: descriptor.width,
        _height: descriptor.height,
        _sample_count: descriptor.sample_count,
        _lazily_allocated: lazily_allocated,
    })
}

/// Creates sampler and reports validation errors through the owning device.
pub(super) fn create_sampler(
    device: Arc<VulkanDeviceInner>,
    descriptor: &HalSamplerDescriptor,
) -> Result<VulkanSamplerInner, HalError> {
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(map_filter_mode(descriptor.mag_filter))
        .min_filter(map_filter_mode(descriptor.min_filter))
        .mipmap_mode(map_mipmap_filter_mode(descriptor.mipmap_filter))
        .address_mode_u(map_address_mode(descriptor.address_mode_u))
        .address_mode_v(map_address_mode(descriptor.address_mode_v))
        .address_mode_w(map_address_mode(descriptor.address_mode_w))
        .mip_lod_bias(0.0)
        .anisotropy_enable(descriptor.max_anisotropy > 1)
        .max_anisotropy(f32::from(descriptor.max_anisotropy))
        .compare_enable(descriptor.compare.is_some())
        .compare_op(
            descriptor
                .compare
                .map_or(vk::CompareOp::ALWAYS, map_compare_function),
        )
        .min_lod(descriptor.lod_min_clamp)
        .max_lod(descriptor.lod_max_clamp)
        .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
        .unnormalized_coordinates(false);
    let sampler = unsafe { device.device.create_sampler(&sampler_info, None) }
        .map_err(|_| texture_error("sampler creation failed"))?;
    Ok(VulkanSamplerInner { device, sampler })
}

#[cfg(feature = "tiled")]
fn sample_count_flags(sample_count: u32) -> Result<vk::SampleCountFlags, HalError> {
    match sample_count {
        1 => Ok(vk::SampleCountFlags::TYPE_1),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        _ => Err(texture_error("unsupported transient sample count")),
    }
}

/// Returns transition image.
pub(super) fn transition_image(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    texture: &VulkanTextureInner,
    new_layout: vk::ImageLayout,
    new_state: u8,
) {
    let old_state = texture.layout.swap(new_state, AtomicOrdering::Relaxed);
    let old_layout = image_layout(old_state);
    if old_layout == new_layout {
        return;
    }
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(texture.image)
        .subresource_range(color_subresource_range())
        .src_access_mask(access_mask_for_layout(old_layout))
        .dst_access_mask(access_mask_for_layout(new_layout));
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            stage_mask_for_layout(old_layout),
            stage_mask_for_layout(new_layout),
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

/// Returns transfer to compute barrier.
pub(super) fn transfer_to_compute_barrier(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// Returns compute to transfer barrier.
pub(super) fn compute_to_transfer_barrier(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::TRANSFER_READ
                | vk::AccessFlags::TRANSFER_WRITE
                | vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE,
        );
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// Returns image layout.
pub(super) fn image_layout(state: u8) -> vk::ImageLayout {
    match state {
        IMAGE_LAYOUT_TRANSFER_DST => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_SRC => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        IMAGE_LAYOUT_COLOR_ATTACHMENT => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        IMAGE_LAYOUT_PRESENT => vk::ImageLayout::PRESENT_SRC_KHR,
        _ => vk::ImageLayout::UNDEFINED,
    }
}

/// Returns access mask for layout.
pub(super) fn access_mask_for_layout(layout: vk::ImageLayout) -> vk::AccessFlags {
    match layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::AccessFlags::TRANSFER_READ,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        vk::ImageLayout::PRESENT_SRC_KHR => vk::AccessFlags::empty(),
        _ => vk::AccessFlags::empty(),
    }
}

/// Returns stage mask for layout.
pub(super) fn stage_mask_for_layout(layout: vk::ImageLayout) -> vk::PipelineStageFlags {
    match layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL | vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
            vk::PipelineStageFlags::TRANSFER
        }
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        }
        vk::ImageLayout::PRESENT_SRC_KHR => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        _ => vk::PipelineStageFlags::TOP_OF_PIPE,
    }
}

/// Returns image subresource layers.
pub(super) fn image_subresource_layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
}

/// Returns color subresource range.
pub(super) fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}
