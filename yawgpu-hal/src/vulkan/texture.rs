use super::*;
use crate::HalTextureDimension;

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
/// Constant value for image layout depth-stencil attachment.
pub(super) const IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT: u8 = 5;
/// Constant value for image layout general.
pub(super) const IMAGE_LAYOUT_GENERAL: u8 = 6;

/// Stores vulkan texture data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct VulkanTexture {
    pub(super) inner: Option<Arc<VulkanTextureInner>>,
    pub(super) swapchain: Option<Arc<VulkanSwapchainInner>>,
    pub(super) surface_pending: Option<Arc<Mutex<SurfacePendingState>>>,
    pub(super) dimension: HalTextureDimension,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) depth_or_array_layers: u32,
    pub(super) sample_count: u32,
    pub(super) bytes_per_pixel: u32,
    pub(super) format: HalTextureFormat,
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
    pub(super) mip_level_count: u32,
    pub(super) array_layers: u32,
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

/// True iff `usage` requires a `VkImageView` to be created alongside the
/// `VkImage`. View-compatible bits per VUID-VkImageViewCreateInfo-image-04441
/// (SAMPLED / STORAGE / *_ATTACHMENT). yawgpu's render_attachment maps to
/// COLOR_ATTACHMENT, so the three caller-facing usage bits cover all
/// view-compatible image-usage flags map_texture_usage can emit.
fn texture_usage_needs_view(usage: HalTextureUsage) -> bool {
    usage.texture_binding || usage.storage_binding || usage.render_attachment
}

/// Creates texture and reports validation errors through the owning device.
pub(super) fn create_texture(
    device: Arc<VulkanDeviceInner>,
    descriptor: &HalTextureDescriptor,
) -> Result<(VulkanTextureInner, u32), HalError> {
    let (format, bytes_per_pixel) = map_texture_format(descriptor.format)?;
    let samples = sample_count_flags(descriptor.sample_count)?;
    let image_type = match descriptor.dimension {
        HalTextureDimension::D1 => vk::ImageType::TYPE_1D,
        HalTextureDimension::D2 => vk::ImageType::TYPE_2D,
        HalTextureDimension::D3 => vk::ImageType::TYPE_3D,
    };
    let extent = vk::Extent3D {
        width: descriptor.width,
        height: match descriptor.dimension {
            HalTextureDimension::D1 => 1,
            HalTextureDimension::D2 | HalTextureDimension::D3 => descriptor.height,
        },
        depth: match descriptor.dimension {
            HalTextureDimension::D3 => descriptor.depth_or_array_layers,
            HalTextureDimension::D1 | HalTextureDimension::D2 => 1,
        },
    };
    let array_layers = match descriptor.dimension {
        HalTextureDimension::D2 => descriptor.depth_or_array_layers,
        HalTextureDimension::D1 | HalTextureDimension::D3 => 1,
    };
    let image_flags = match descriptor.dimension {
        HalTextureDimension::D3 => vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE,
        HalTextureDimension::D1 | HalTextureDimension::D2 => vk::ImageCreateFlags::empty(),
    };
    let image_info = vk::ImageCreateInfo::default()
        .flags(image_flags)
        .image_type(image_type)
        .format(format)
        .extent(extent)
        .mip_levels(descriptor.mip_level_count)
        .array_layers(array_layers)
        .samples(samples)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(map_texture_usage(descriptor.usage))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.device.create_image(&image_info, None) }
        .map_err(|error| map_texture_error(error, "image creation failed"))?;
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
    // Proactive guard: reject allocations that exceed the backing heap capacity.
    // MoltenVK defers real Metal allocation so vkAllocateMemory may return
    // VK_SUCCESS for impossible sizes; comparing against the heap size catches
    // those before the call and produces a deterministic OutOfMemory error.
    if requirements.size > memory_heap_size(&device.memory_properties, memory_type_index) {
        unsafe {
            device.device.destroy_image(image, None);
        }
        return Err(HalError::OutOfMemory {
            backend: BACKEND,
            resource: "texture",
        });
    }
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory =
        unsafe { device.device.allocate_memory(&allocate_info, None) }.map_err(|error| {
            unsafe {
                device.device.destroy_image(image, None);
            }
            map_texture_error(error, "image memory allocation failed")
        })?;
    if let Err(error) = unsafe { device.device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.device.destroy_image(image, None);
            device.device.free_memory(memory, None);
        }
        return Err(map_texture_error(error, "image memory bind failed"));
    }
    let view = if texture_usage_needs_view(descriptor.usage) {
        let view_type = match descriptor.dimension {
            HalTextureDimension::D1 => vk::ImageViewType::TYPE_1D,
            HalTextureDimension::D2 if descriptor.depth_or_array_layers > 1 => {
                vk::ImageViewType::TYPE_2D_ARRAY
            }
            HalTextureDimension::D2 => vk::ImageViewType::TYPE_2D,
            HalTextureDimension::D3 => vk::ImageViewType::TYPE_3D,
        };
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(format)
            .subresource_range(color_subresource_range(
                descriptor.mip_level_count,
                array_layers,
            ));
        unsafe { device.device.create_image_view(&view_info, None) }.map_err(|_| {
            unsafe {
                device.device.destroy_image(image, None);
                device.device.free_memory(memory, None);
            }
            texture_error("image view creation failed")
        })?
    } else {
        vk::ImageView::null()
    };
    Ok((
        VulkanTextureInner {
            device,
            image,
            view,
            memory: Some(memory),
            owns_image: true,
            mip_level_count: descriptor.mip_level_count,
            array_layers,
            layout: AtomicU8::new(IMAGE_LAYOUT_UNDEFINED),
        },
        bytes_per_pixel,
    ))
}

/// Computes the effective anisotropy enable flag and clamped max value.
///
/// WebGPU semantics: values above the platform maximum are clamped, never an
/// error. `anisotropy_enable` must only be set to `true` when the
/// `samplerAnisotropy` device feature is enabled; otherwise the Vulkan spec
/// (VUID-VkSamplerCreateInfo-anisotropyEnable-01070) is violated and MoltenVK
/// produces an error command buffer.
///
/// Returns `(anisotropy_enable, max_anisotropy)` for use in
/// `VkSamplerCreateInfo`.
pub(super) fn effective_anisotropy(
    requested: u16,
    feature_enabled: bool,
    device_max: f32,
) -> (bool, f32) {
    if !feature_enabled {
        // Feature absent: anisotropic filtering is unavailable; fall back to 1.
        return (false, 1.0);
    }
    let clamped = f32::from(requested).clamp(1.0, device_max);
    // anisotropyEnable must be false when the effective value is 1.0 to avoid
    // VUID violations on implementations that clamp internally.
    (clamped > 1.0, clamped)
}

/// Creates sampler and reports validation errors through the owning device.
pub(super) fn create_sampler(
    device: Arc<VulkanDeviceInner>,
    descriptor: &HalSamplerDescriptor,
) -> Result<VulkanSamplerInner, HalError> {
    let (anisotropy_enable, max_anisotropy) = effective_anisotropy(
        descriptor.max_anisotropy,
        device.sampler_anisotropy,
        device.max_sampler_anisotropy,
    );
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(map_filter_mode(descriptor.mag_filter))
        .min_filter(map_filter_mode(descriptor.min_filter))
        .mipmap_mode(map_mipmap_filter_mode(descriptor.mipmap_filter))
        .address_mode_u(map_address_mode(descriptor.address_mode_u))
        .address_mode_v(map_address_mode(descriptor.address_mode_v))
        .address_mode_w(map_address_mode(descriptor.address_mode_w))
        .mip_lod_bias(0.0)
        .anisotropy_enable(anisotropy_enable)
        .max_anisotropy(max_anisotropy)
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

fn sample_count_flags(sample_count: u32) -> Result<vk::SampleCountFlags, HalError> {
    match sample_count {
        1 => Ok(vk::SampleCountFlags::TYPE_1),
        4 => Ok(vk::SampleCountFlags::TYPE_4),
        _ => Err(texture_error("unsupported texture sample count")),
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
        .subresource_range(color_subresource_range(
            texture.mip_level_count,
            texture.array_layers,
        ))
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

/// Returns transition image for the requested aspect range.
pub(super) fn transition_image_aspect(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    texture: &VulkanTextureInner,
    aspect: vk::ImageAspectFlags,
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
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(texture.mip_level_count)
                .base_array_layer(0)
                .layer_count(texture.array_layers),
        )
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

pub(super) fn buffer_write_read_barrier_dst_access_mask() -> vk::AccessFlags {
    vk::AccessFlags::SHADER_READ
        | vk::AccessFlags::SHADER_WRITE
        | vk::AccessFlags::TRANSFER_READ
        | vk::AccessFlags::TRANSFER_WRITE
        | vk::AccessFlags::INDIRECT_COMMAND_READ
        | vk::AccessFlags::INDEX_READ
}

pub(super) fn buffer_write_read_barrier_dst_stage_mask() -> vk::PipelineStageFlags {
    vk::PipelineStageFlags::COMPUTE_SHADER
        | vk::PipelineStageFlags::VERTEX_SHADER
        | vk::PipelineStageFlags::FRAGMENT_SHADER
        | vk::PipelineStageFlags::TRANSFER
        | vk::PipelineStageFlags::DRAW_INDIRECT
        | vk::PipelineStageFlags::VERTEX_INPUT
}

/// Records a barrier from transfer writes to later buffer reads or writes.
pub(super) fn transfer_to_compute_barrier(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(buffer_write_read_barrier_dst_access_mask());
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            buffer_write_read_barrier_dst_stage_mask(),
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// Records a barrier from shader writes to later buffer reads or writes.
pub(super) fn compute_to_transfer_barrier(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(buffer_write_read_barrier_dst_access_mask());
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            buffer_write_read_barrier_dst_stage_mask(),
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
        IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        IMAGE_LAYOUT_GENERAL => vk::ImageLayout::GENERAL,
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
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
        }
        vk::ImageLayout::GENERAL => vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
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
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
        }
        vk::ImageLayout::GENERAL => {
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER
        }
        vk::ImageLayout::PRESENT_SRC_KHR => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        _ => vk::PipelineStageFlags::TOP_OF_PIPE,
    }
}

/// Returns image subresource layers.
pub(super) fn image_subresource_layers(
    aspect: vk::ImageAspectFlags,
    mip_level: u32,
    base_array_layer: u32,
    layer_count: u32,
) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(aspect)
        .mip_level(mip_level)
        .base_array_layer(base_array_layer)
        .layer_count(layer_count)
}

/// Returns color subresource range.
pub(super) fn color_subresource_range(
    level_count: u32,
    layer_count: u32,
) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(level_count)
        .base_array_layer(0)
        .layer_count(layer_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_anisotropy_clamps_above_device_max_to_device_max() {
        // (1024, true, 16.0) → (true, 16.0)
        let (enable, max) = effective_anisotropy(1024, true, 16.0);
        assert!(enable, "anisotropy should be enabled");
        assert_eq!(max, 16.0);
    }

    #[test]
    fn effective_anisotropy_passes_through_exact_device_max() {
        // (16, true, 16.0) → (true, 16.0)
        let (enable, max) = effective_anisotropy(16, true, 16.0);
        assert!(enable, "anisotropy should be enabled");
        assert_eq!(max, 16.0);
    }

    #[test]
    fn effective_anisotropy_disables_when_requested_is_one() {
        // (1, true, 16.0) → (false, 1.0)
        let (enable, max) = effective_anisotropy(1, true, 16.0);
        assert!(
            !enable,
            "anisotropy should be disabled for max_anisotropy=1"
        );
        assert_eq!(max, 1.0);
    }

    #[test]
    fn buffer_write_read_barrier_covers_indirect_index_and_copy_source_reads() {
        let access = buffer_write_read_barrier_dst_access_mask();
        assert!(access.contains(vk::AccessFlags::INDIRECT_COMMAND_READ));
        assert!(access.contains(vk::AccessFlags::INDEX_READ));
        assert!(access.contains(vk::AccessFlags::TRANSFER_READ));

        let stages = buffer_write_read_barrier_dst_stage_mask();
        assert!(stages.contains(vk::PipelineStageFlags::DRAW_INDIRECT));
        assert!(stages.contains(vk::PipelineStageFlags::VERTEX_INPUT));
        assert!(stages.contains(vk::PipelineStageFlags::TRANSFER));
    }

    #[test]
    fn effective_anisotropy_disables_when_feature_absent() {
        // (1024, false, 16.0) → (false, 1.0)
        let (enable, max) = effective_anisotropy(1024, false, 16.0);
        assert!(
            !enable,
            "anisotropy must be disabled when feature is not enabled"
        );
        assert_eq!(max, 1.0, "max should fall back to 1.0 when feature absent");
    }

    /// Builds a minimal `vk::PhysicalDeviceMemoryProperties` with a single
    /// device-local heap of the given capacity for pure-logic tests.
    #[allow(clippy::field_reassign_with_default)] // array elements cannot be set in struct literal
    fn device_local_memory_properties(heap_size: u64) -> vk::PhysicalDeviceMemoryProperties {
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        props.memory_type_count = 1;
        props.memory_types[0].heap_index = 0;
        props.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        props.memory_heap_count = 1;
        props.memory_heaps[0].size = heap_size;
        props
    }

    #[test]
    fn texture_heap_guard_rejects_requirement_exceeding_heap() {
        let props = device_local_memory_properties(1024);
        let heap = memory_heap_size(&props, 0);
        // 64 GiB (CTS oversized texture) must exceed a 1 KiB synthetic heap
        let oversized: u64 = 64 * 1024 * 1024 * 1024;
        assert!(oversized > heap, "oversized requirement should exceed heap");
    }

    #[test]
    fn texture_heap_guard_accepts_requirement_within_heap() {
        let props = device_local_memory_properties(u64::MAX);
        let heap = memory_heap_size(&props, 0);
        let normal: u64 = 4 * 4 * 4; // tiny 4×4 rgba8 texture
        assert!(normal <= heap, "small requirement should fit in heap");
    }

    fn texture_usage(
        texture_binding: bool,
        storage_binding: bool,
        render_attachment: bool,
    ) -> HalTextureUsage {
        HalTextureUsage {
            copy_src: false,
            copy_dst: false,
            texture_binding,
            storage_binding,
            render_attachment,
        }
    }

    #[test]
    fn texture_usage_needs_view_returns_true_for_render_attachment() {
        assert!(texture_usage_needs_view(texture_usage(false, false, true)));
    }

    #[test]
    fn texture_usage_needs_view_returns_true_for_texture_binding() {
        assert!(texture_usage_needs_view(texture_usage(true, false, false)));
    }

    #[test]
    fn texture_usage_needs_view_returns_true_for_storage_binding() {
        assert!(texture_usage_needs_view(texture_usage(false, true, false)));
    }

    #[test]
    fn texture_usage_needs_view_returns_false_for_copy_only() {
        assert!(!texture_usage_needs_view(HalTextureUsage {
            copy_src: true,
            copy_dst: true,
            texture_binding: false,
            storage_binding: false,
            render_attachment: false,
        }));
    }
}
