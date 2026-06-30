use super::*;

/// Holds shared state for the vulkan device handle.
pub(super) struct VulkanDeviceInner {
    pub(super) _instance: Arc<VulkanInstanceInner>,
    pub(super) device: ash::Device,
    pub(super) physical_device: vk::PhysicalDevice,
    pub(super) memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(super) queue_family_index: u32,
    pub(super) occlusion_query_precise: bool,
    /// Whether viewport depth clamp and primitive depth clipping can be controlled separately.
    pub(super) depth_clip_control: bool,
    /// Whether the `samplerAnisotropy` device feature was enabled at device creation.
    pub(super) sampler_anisotropy: bool,
    /// Whether `VK_EXT_shader_demote_to_helper_invocation` /
    /// `shaderDemoteToHelperInvocation` was enabled at device creation. When
    /// true, SPIR-V `OpDemoteToHelperInvocation` (which naga emits for WGSL
    /// `discard`) is legal to execute, so derivatives (`fwidth`/`dpdx`/`dpdy`)
    /// after a non-uniform `discard` are well-defined (CTS finding F-129).
    pub(super) shader_demote_to_helper_invocation: bool,
    /// Whether `VK_KHR_shader_float16_int8` / `shaderFloat16` was enabled.
    pub(super) shader_float16: bool,
    /// Whether `VK_KHR_vulkan_memory_model` / `vulkanMemoryModel` was enabled.
    pub(super) vulkan_memory_model: bool,
    /// Whether `VkImageFormatListCreateInfo` is available for mutable image views.
    pub(super) image_format_list: bool,
    /// Whether `VK_KHR_16bit_storage` / `storageBuffer16BitAccess` was enabled.
    pub(super) storage_buffer16_bit_access: bool,
    /// Whether `VK_KHR_16bit_storage` / `uniformAndStorageBuffer16BitAccess` was enabled.
    pub(super) uniform_and_storage_buffer16_bit_access: bool,
    /// Whether `VK_KHR_16bit_storage` / `storageInputOutput16` was enabled.
    pub(super) storage_input_output16: bool,
    /// Whether `VK_KHR_16bit_storage` / `storagePushConstant16` was enabled.
    pub(super) storage_push_constant16: bool,
    /// `VkPhysicalDeviceLimits.maxSamplerAnisotropy` — the hardware ceiling for
    /// anisotropic filtering. Used to clamp `VkSamplerCreateInfo.maxAnisotropy`
    /// per WebGPU semantics (clamp, never error).
    pub(super) max_sampler_anisotropy: f32,
    #[cfg(feature = "tiled")]
    pub(super) subpass_render_pass_cache: Mutex<BTreeMap<HalSubpassPassLayout, vk::RenderPass>>,
    pub(super) allocations: AtomicU64,
}

impl fmt::Debug for VulkanDeviceInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanDeviceInner")
            .field(
                "shader_demote_to_helper_invocation",
                &self.shader_demote_to_helper_invocation,
            )
            .field("shader_float16", &self.shader_float16)
            .field("vulkan_memory_model", &self.vulkan_memory_model)
            .field("image_format_list", &self.image_format_list)
            .field(
                "storage_buffer16_bit_access",
                &self.storage_buffer16_bit_access,
            )
            .field(
                "uniform_and_storage_buffer16_bit_access",
                &self.uniform_and_storage_buffer16_bit_access,
            )
            .field("storage_input_output16", &self.storage_input_output16)
            .field("storage_push_constant16", &self.storage_push_constant16)
            .finish_non_exhaustive()
    }
}

impl Drop for VulkanDeviceInner {
    fn drop(&mut self) {
        unsafe {
            #[cfg(feature = "tiled")]
            if let Ok(mut cache) = self.subpass_render_pass_cache.lock() {
                for (_, render_pass) in std::mem::take(&mut *cache) {
                    self.device.destroy_render_pass(render_pass, None);
                }
            }
            self.device.destroy_device(None);
        }
    }
}

/// Stores vulkan device data used by validation and backend submission.
#[derive(Debug)]
pub struct VulkanDevice {
    pub(super) inner: Arc<VulkanDeviceInner>,
    pub(super) queue: VulkanQueue,
}

impl VulkanDevice {
    /// Returns the allocation count.
    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.inner.allocations.load(Ordering::Relaxed)
    }

    /// Returns the queue.
    #[must_use]
    pub fn queue(&self) -> &VulkanQueue {
        &self.queue
    }

    /// Returns true when `VK_KHR_vulkan_memory_model` / `vulkanMemoryModel` was
    /// enabled at device creation.
    #[must_use]
    pub fn vulkan_memory_model(&self) -> bool {
        self.inner.vulkan_memory_model
    }

    /// Allocates a buffer of the given size on this device.
    pub fn create_buffer(
        &self,
        size: u64,
        usage: HalBufferUsage,
    ) -> Result<VulkanBuffer, HalError> {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        let inner = create_buffer(Arc::clone(&self.inner), size, usage)?;
        Ok(VulkanBuffer {
            inner: Some(Arc::new(inner)),
            size,
        })
    }

    /// Creates a texture matching the given descriptor.
    pub fn create_texture(
        &self,
        descriptor: &HalTextureDescriptor,
    ) -> Result<VulkanTexture, HalError> {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        let (inner, bytes_per_pixel) = create_texture(Arc::clone(&self.inner), descriptor)?;
        Ok(VulkanTexture {
            inner: Some(Arc::new(inner)),
            swapchain: None,
            surface_pending: None,
            dimension: descriptor.dimension,
            width: descriptor.width,
            height: descriptor.height,
            depth_or_array_layers: descriptor.depth_or_array_layers,
            sample_count: descriptor.sample_count,
            bytes_per_pixel,
            format: descriptor.format,
        })
    }

    /// Creates a query set matching the given kind and count.
    pub fn create_query_set(
        &self,
        kind: HalQueryKind,
        count: u32,
    ) -> Result<VulkanQuerySet, HalError> {
        match kind {
            HalQueryKind::Occlusion => {
                self.inner.allocations.fetch_add(1, Ordering::Relaxed);
                VulkanQuerySet::new(Arc::clone(&self.inner), count)
            }
        }
    }

    /// Creates a sampler matching the given descriptor.
    #[must_use]
    pub fn create_sampler(&self, descriptor: &HalSamplerDescriptor) -> VulkanSampler {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        VulkanSampler {
            _inner: create_sampler(Arc::clone(&self.inner), descriptor)
                .ok()
                .map(Arc::new),
        }
    }

    /// Creates a compute pipeline from the given shader, entry point, and bindings.
    pub fn create_compute_pipeline(
        &self,
        shader: HalShaderSource,
        entry_point: &str,
        _workgroup_size: (u32, u32, u32),
        bindings: &[HalDescriptorBinding],
    ) -> Result<VulkanComputePipeline, HalError> {
        create_compute_pipeline(Arc::clone(&self.inner), shader, entry_point, bindings)
    }

    /// Creates a render pipeline from the given shaders, vertex layout, and color targets.
    pub fn create_render_pipeline(
        &self,
        shader: HalShaderSource,
        vertex_entry_point: &str,
        fragment_entry_point: Option<&str>,
        descriptor: &HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
    ) -> Result<VulkanRenderPipeline, HalError> {
        create_render_pipeline(
            Arc::clone(&self.inner),
            shader,
            vertex_entry_point,
            fragment_entry_point,
            descriptor,
            bindings,
        )
    }

    /// Creates a subpass-compatible render pipeline from the given shaders and pass layout.
    #[cfg(feature = "tiled")]
    #[allow(clippy::too_many_arguments)]
    pub fn create_subpass_render_pipeline(
        &self,
        shader: HalShaderSource,
        vertex_entry_point: &str,
        fragment_entry_point: Option<&str>,
        descriptor: &HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
        pass_layout: &HalSubpassPassLayout,
        subpass_index: u32,
    ) -> Result<VulkanRenderPipeline, HalError> {
        create_subpass_render_pipeline(
            Arc::clone(&self.inner),
            shader,
            vertex_entry_point,
            fragment_entry_point,
            descriptor,
            bindings,
            pass_layout,
            subpass_index,
        )
    }
}

/// Returns physical device name.
pub(super) fn physical_device_name(properties: vk::PhysicalDeviceProperties) -> Option<String> {
    properties
        .device_name_as_c_str()
        .ok()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::super::*;
    use super::*;

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_allocation_count_tracks_created_resources() {
        let device = vulkan_device();
        assert_eq!(device.allocation_count(), 0);
        let _buffer = device.create_buffer(4, HalBufferUsage::default());
        let _texture = device.create_texture(&texture_descriptor());
        let _sampler = device.create_sampler(&sampler_descriptor());
        assert_eq!(device.allocation_count(), 3);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_queue_returns_same_reference() {
        let device = vulkan_device();
        assert!(std::ptr::eq(device.queue(), device.queue()));
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_create_buffer_records_size_and_maps_memory() {
        let device = vulkan_device();
        let buffer = device
            .create_buffer(16, HalBufferUsage::default())
            .expect("Vulkan buffer allocation should succeed");
        assert_eq!(buffer.size(), 16);
        assert!(buffer.mapped_ptr().is_some());
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_create_texture_records_descriptor_shape() {
        let device = vulkan_device();
        let texture = device
            .create_texture(&texture_descriptor())
            .expect("Vulkan texture allocation should succeed");
        assert_eq!(texture.width, 4);
        assert_eq!(texture.height, 4);
        assert_eq!(texture.depth_or_array_layers, 1);
        assert_eq!(texture.bytes_per_pixel, 4);
        assert!(matches!(texture.format, HalTextureFormat::Rgba8Unorm));
        assert!(texture.inner.is_some());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_create_sampler_returns_sampler() {
        let device = vulkan_device();
        let sampler = device.create_sampler(&sampler_descriptor());
        assert!(sampler._inner.is_some());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_create_compute_pipeline_accepts_spirv() {
        let device = vulkan_device();
        let pipeline = device
            .create_compute_pipeline(
                HalShaderSource::SpirV(compute_spirv()),
                "main",
                (1, 1, 1),
                &[],
            )
            .expect("create compute pipeline");
        assert_ne!(pipeline.inner.pipeline, vk::Pipeline::null());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_create_render_pipeline_accepts_spirv_stages() {
        let device = vulkan_device();
        let pipeline = device
            .create_render_pipeline(
                HalShaderSource::SpirVStages {
                    vertex: vertex_spirv(),
                    fragment: Some(fragment_spirv()),
                },
                "main",
                Some("main"),
                &render_descriptor(),
                &[],
            )
            .expect("create render pipeline");
        assert_ne!(pipeline.inner.pipeline, vk::Pipeline::null());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_oversized_allocations_return_out_of_memory() {
        // u64::MAX exceeds every Vulkan heap size, so vkAllocateMemory returns
        // VK_ERROR_OUT_OF_DEVICE_MEMORY → OutOfMemory.
        // For the texture, mirror the WebGPU CTS OOM recipe (error_test.ts):
        // query the device's own maxImageDimension2D / maxImageArrayLayers so
        // the descriptor stays legal on any device, and use a 16-byte texel
        // format so the total byte size (1 TiB even at spec-minimum limits)
        // cannot be satisfied, causing allocation to fail with OutOfMemory.
        let device = vulkan_device();
        assert!(matches!(
            device.create_buffer(u64::MAX, HalBufferUsage::default()),
            Err(HalError::OutOfMemory { .. })
        ));

        let limits = unsafe {
            device
                .inner
                ._instance
                .instance
                .get_physical_device_properties(device.inner.physical_device)
                .limits
        };
        let mut descriptor = texture_descriptor();
        descriptor.format = crate::HalTextureFormat::Rgba32Float;
        descriptor.width = limits.max_image_dimension2_d;
        descriptor.height = limits.max_image_dimension2_d;
        descriptor.depth_or_array_layers = limits.max_image_array_layers;
        assert!(matches!(
            device.create_texture(&descriptor),
            Err(HalError::OutOfMemory { .. })
        ));
    }
}
