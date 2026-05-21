use super::*;

/// Holds shared state for the vulkan device handle.
pub(super) struct VulkanDeviceInner {
    pub(super) _instance: Arc<VulkanInstanceInner>,
    pub(super) device: ash::Device,
    pub(super) physical_device: vk::PhysicalDevice,
    pub(super) memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(super) queue_family_index: u32,
    pub(super) allocations: AtomicU64,
}

impl fmt::Debug for VulkanDeviceInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanDeviceInner")
            .finish_non_exhaustive()
    }
}

impl Drop for VulkanDeviceInner {
    fn drop(&mut self) {
        unsafe {
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

    /// Allocates a buffer of the given size on this device.
    #[must_use]
    pub fn create_buffer(&self, size: u64) -> VulkanBuffer {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        match create_buffer(Arc::clone(&self.inner), size) {
            Ok(inner) => VulkanBuffer {
                inner: Some(Arc::new(inner)),
                size,
            },
            Err(_) => VulkanBuffer { inner: None, size },
        }
    }

    /// Creates a texture matching the given descriptor.
    #[must_use]
    pub fn create_texture(&self, descriptor: &HalTextureDescriptor) -> VulkanTexture {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        match create_texture(Arc::clone(&self.inner), descriptor) {
            Ok((inner, bytes_per_pixel)) => VulkanTexture {
                inner: Some(Arc::new(inner)),
                swapchain: None,
                width: descriptor.width,
                height: descriptor.height,
                depth_or_array_layers: descriptor.depth_or_array_layers,
                bytes_per_pixel,
                format: descriptor.format,
            },
            Err(_) => VulkanTexture {
                inner: None,
                swapchain: None,
                width: descriptor.width,
                height: descriptor.height,
                depth_or_array_layers: descriptor.depth_or_array_layers,
                bytes_per_pixel: 0,
                format: descriptor.format,
            },
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
        fragment_entry_point: &str,
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
        let _buffer = device.create_buffer(4);
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
        let buffer = device.create_buffer(16);
        assert_eq!(buffer.size(), 16);
        assert!(buffer.mapped_ptr().is_some());
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_create_texture_records_descriptor_shape() {
        let device = vulkan_device();
        let texture = device.create_texture(&texture_descriptor());
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
                    fragment: fragment_spirv(),
                },
                "main",
                "main",
                &render_descriptor(),
                &[],
            )
            .expect("create render pipeline");
        assert_ne!(pipeline.inner.pipeline, vk::Pipeline::null());
    }
}
