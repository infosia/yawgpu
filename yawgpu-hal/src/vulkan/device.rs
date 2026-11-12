use super::*;

/// Holds shared state for the vulkan device handle.
pub(super) struct VulkanDeviceInner {
    /// Actual extension names submitted to vkCreateDevice, retained for the manual test.
    #[cfg(test)]
    pub(super) spirv_1_4_extensions_enabled: [bool; 2],
    pub(super) shader_workarounds: crate::HalShaderWorkarounds,
    /// Whether the pipeline robustness extension and feature were enabled.
    pub(super) pipeline_robustness: bool,
    /// Reconvergence features enabled at device creation, independent of Tint choice.
    pub(super) reconvergence_features: ReconvergenceFeatures,
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
    /// true, Tint selects SPIR-V `OpDemoteToHelperInvocation` for WGSL
    /// `discard`; otherwise it emulates discard. Derivatives (`fwidth`/`dpdx`/`dpdy`)
    /// after a non-uniform `discard` are well-defined (CTS finding F-129).
    pub(super) shader_demote_to_helper_invocation: bool,
    /// Whether `VK_KHR_vulkan_memory_model` / `vulkanMemoryModel` was enabled.
    pub(super) vulkan_memory_model: bool,
    /// Whether `VkImageFormatListCreateInfo` is available for mutable image views.
    pub(super) image_format_list: bool,
    /// Whether the swapchain mutable format extension and its dependencies are enabled.
    pub(super) swapchain_mutable_format: bool,
    /// Whether `VK_EXT_subgroup_size_control` with `subgroupSizeControl`
    /// was enabled (Block 108 R5). Compute pipelines
    /// need it for a required subgroup size and for
    /// `ALLOW_VARYING_SUBGROUP_SIZE`.
    pub(super) subgroup_size_control: bool,
    /// Required size for compute pipelines without an explicit WGSL subgroup size.
    pub(super) default_compute_subgroup_size: Option<u32>,
    /// Whether `VK_KHR_16bit_storage` / `storageInputOutput16` was enabled.
    pub(super) storage_input_output16: bool,
    /// `VkPhysicalDeviceLimits.maxSamplerAnisotropy` — the hardware ceiling for
    /// anisotropic filtering. Used to clamp `VkSamplerCreateInfo.maxAnisotropy`
    /// per WebGPU semantics (clamp, never error).
    pub(super) max_sampler_anisotropy: f32,
    /// Configured surfaces whose next acquire must be waited on by the queue.
    pub(super) surfaces: Mutex<Vec<std::sync::Weak<Mutex<SurfacePendingState>>>>,
    pub(super) allocations: AtomicU64,
}

impl fmt::Debug for VulkanDeviceInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanDeviceInner")
            .field("pipeline_robustness", &self.pipeline_robustness)
            .field("reconvergence_features", &self.reconvergence_features)
            .field(
                "shader_demote_to_helper_invocation",
                &self.shader_demote_to_helper_invocation,
            )
            .field("vulkan_memory_model", &self.vulkan_memory_model)
            .field("image_format_list", &self.image_format_list)
            .field("subgroup_size_control", &self.subgroup_size_control)
            .field(
                "default_compute_subgroup_size",
                &self.default_compute_subgroup_size,
            )
            .field("storage_input_output16", &self.storage_input_output16)
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
    pub(crate) fn shader_workarounds(&self) -> crate::HalShaderWorkarounds {
        self.inner.shader_workarounds
    }

    pub(crate) fn storage_input_output16(&self) -> bool {
        self.inner.storage_input_output16
    }

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
        let image_usage = inner.usage;
        Ok(VulkanTexture {
            inner: Some(Arc::new(inner)),
            swapchain: None,
            _surface_pending: None,
            dimension: descriptor.dimension,
            width: descriptor.width,
            height: descriptor.height,
            depth_or_array_layers: descriptor.depth_or_array_layers,
            sample_count: descriptor.sample_count,
            bytes_per_pixel,
            format: descriptor.format,
            image_usage,
        })
    }

    /// Creates a query set matching the given kind and count.
    #[must_use = "query set creation can fail"]
    pub fn create_query_set(
        &self,
        kind: HalQueryKind,
        count: u32,
    ) -> Result<VulkanQuerySet, HalError> {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        VulkanQuerySet::new(Arc::clone(&self.inner), kind, count)
    }

    /// Creates a sampler matching the given descriptor, reporting backend allocation errors.
    #[must_use = "sampler creation can fail"]
    pub fn create_sampler(
        &self,
        descriptor: &HalSamplerDescriptor,
    ) -> Result<VulkanSampler, HalError> {
        let sampler = create_sampler(Arc::clone(&self.inner), descriptor)?;
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(VulkanSampler {
            _inner: Arc::new(sampler),
        })
    }

    /// Creates a compute pipeline from the given shader, entry point, and bindings.
    ///
    /// `user_immediate_size` is the pipeline layout's reserved user-immediate
    /// byte budget (Block 94 S3); it sizes the compute push-constant range.
    ///
    /// `required_subgroup_size` is `Some(S)` when the entry point declares WGSL
    /// `@subgroup_size(S)` (Block 108): the stage requires subgroup size `S`
    /// with full subgroups. It returns a [`HalError`] when the device did not
    /// enable `VK_EXT_subgroup_size_control`. Without an explicit size the
    /// device-selected default is required, when present, without full subgroups.
    pub fn create_compute_pipeline(
        &self,
        shader: HalShaderSource,
        entry_point: &str,
        _workgroup_size: (u32, u32, u32),
        bindings: &[HalDescriptorBinding],
        user_immediate_size: u32,
        required_subgroup_size: Option<u32>,
    ) -> Result<VulkanComputePipeline, HalError> {
        create_compute_pipeline(
            Arc::clone(&self.inner),
            shader,
            entry_point,
            bindings,
            user_immediate_size,
            required_subgroup_size,
        )
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
    use super::*;

    /// Block 102 R3: both query kinds allocate a real pool and each counts as
    /// one device allocation; the kind reaches the query set unchanged.
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_create_query_set_creates_pools_for_both_kinds() {
        let device = vulkan_device();

        let timestamp = device
            .create_query_set(HalQueryKind::Timestamp, 4)
            .expect("timestamp query set should allocate");
        let occlusion = device
            .create_query_set(HalQueryKind::Occlusion, 2)
            .expect("occlusion query set should allocate");

        assert_eq!(timestamp.kind(), HalQueryKind::Timestamp);
        assert_eq!(timestamp.count(), 4);
        assert_eq!(occlusion.kind(), HalQueryKind::Occlusion);
        assert_eq!(occlusion.count(), 2);
        assert_eq!(device.allocation_count(), 2);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_allocation_count_tracks_created_resources() {
        let device = vulkan_device();
        assert_eq!(device.allocation_count(), 0);
        let _buffer = device.create_buffer(4, HalBufferUsage::default());
        let _texture = device.create_texture(&texture_descriptor());
        let _sampler = device
            .create_sampler(&sampler_descriptor())
            .expect("create sampler");
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
        let before = device.allocation_count();
        let sampler = device
            .create_sampler(&sampler_descriptor())
            .expect("create sampler");
        assert_ne!(sampler._inner.sampler, vk::Sampler::null());
        assert_eq!(device.allocation_count(), before + 1);
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
                0,
                None,
            )
            .expect("create compute pipeline");
        assert_ne!(pipeline.inner.pipeline, vk::Pipeline::null());
    }

    #[test]
    #[ignore = "requires real Vulkan with VK_LAYER_KHRONOS_validation; Block 127 S2 manual run"]
    fn vulkan_device_reconvergence_and_default_compute_pipeline() {
        let instance = VulkanInstance::new().expect("create Vulkan instance");
        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Vulkan adapter");
        let query = adapter.subgroup_query();
        let device = adapter.create_device().expect("create Vulkan device");
        let flags = device.shader_workarounds();
        let maximal_extension =
            adapter.has_device_extension(vk::KHR_SHADER_MAXIMAL_RECONVERGENCE_NAME);
        let uniform_extension =
            adapter.has_device_extension(vk::KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_NAME);
        let mut maximal_features =
            vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default();
        let mut uniform_features =
            vk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR::default();
        let mut features = vk::PhysicalDeviceFeatures2::default();
        if maximal_extension {
            features = features.push_next(&mut maximal_features);
        }
        if uniform_extension {
            features = features.push_next(&mut uniform_features);
        }
        unsafe {
            instance
                .inner
                .instance
                .get_physical_device_features2(adapter.physical_device, &mut features);
        }
        let expected_enabled = enabled_reconvergence_features(
            maximal_extension,
            maximal_features.shader_maximal_reconvergence,
            uniform_extension,
            uniform_features.shader_subgroup_uniform_control_flow,
        );
        assert_eq!(device.inner.reconvergence_features, expected_enabled);
        let properties = unsafe {
            instance
                .inner
                .instance
                .get_physical_device_properties(adapter.physical_device)
        };
        let expected_tint = select_reconvergence(
            expected_enabled,
            query.supported_stages,
            cfg!(target_os = "android"),
            properties.vendor_id,
        );
        assert_eq!(
            flags.use_maximal_reconvergence,
            expected_tint == Reconvergence::Maximal
        );
        assert_eq!(
            flags.use_subgroup_uniform_control_flow,
            expected_tint == Reconvergence::UniformControlFlow
        );
        eprintln!(
            "reconvergence extensions: maximal={maximal_extension}, subgroup_uniform_control_flow={uniform_extension}; enabled features={:?}",
            device.inner.reconvergence_features,
        );
        eprintln!(
            "Tint reconvergence: maximal={}, subgroup_uniform_control_flow={}; default compute subgroup size={:?}",
            flags.use_maximal_reconvergence,
            flags.use_subgroup_uniform_control_flow,
            device.inner.default_compute_subgroup_size,
        );
        eprintln!(
            "subgroupSizeControl enabled={}; requiredSubgroupSizeStages={:?}",
            device.inner.subgroup_size_control, query.required_subgroup_size_stages,
        );
        if !device.inner.subgroup_size_control
            || !query
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
        {
            assert_eq!(device.inner.default_compute_subgroup_size, None);
        }
        if let Some(default_size) = device.inner.default_compute_subgroup_size {
            let subgroup_budget =
                u64::from(query.max_compute_workgroup_subgroups) * u64::from(default_size);
            assert!(subgroup_budget >= 256);
            assert!(
                u64::from(adapter.limits().max_compute_invocations_per_workgroup)
                    <= subgroup_budget
            );
        }
        assert!(!(flags.use_maximal_reconvergence && flags.use_subgroup_uniform_control_flow));
        let program = yawgpu_tint::Program::parse(
            "@compute @workgroup_size(1) fn main() {}",
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            &[],
        )
        .expect("parse compute shader without @subgroup_size");
        let mut writer = yawgpu_tint::SpirvWorkarounds::default();
        writer.use_maximal_reconvergence = flags.use_maximal_reconvergence;
        writer.use_subgroup_uniform_control_flow = flags.use_subgroup_uniform_control_flow;
        let words = program
            .generate_spirv(
                "main",
                &Default::default(),
                &[],
                true,
                false,
                0,
                false,
                None,
                false,
                0,
                true,
                false,
                writer,
            )
            .expect("generate compute SPIR-V");
        // A one-invocation workgroup also checks that defaults do not require full subgroups.
        let pipeline = device
            .create_compute_pipeline(
                HalShaderSource::SpirV(words),
                "main",
                (1, 1, 1),
                &[],
                0,
                None,
            )
            .expect("create compute pipeline without @subgroup_size");
        assert_ne!(pipeline.inner.pipeline, vk::Pipeline::null());
    }

    /// Block 108 R2 + R5: on an adapter advertising `subgroup-size-control`,
    /// the device enables the extension and a compute pipeline requiring
    /// `min_size` with a `(min_size, 1, 1)` workgroup creates.
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_device_create_compute_pipeline_honors_required_subgroup_size() {
        let instance = VulkanInstance::new().expect("create Vulkan instance");
        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Vulkan adapter");
        let Some(caps) = adapter.subgroup_size_control_caps() else {
            eprintln!("skipping: adapter does not advertise subgroup-size-control");
            return;
        };
        eprintln!("subgroup-size-control caps: {caps:?}");
        assert_eq!(
            adapter.subgroup_size_range(),
            Some((caps.min_size, caps.max_size))
        );
        let device = adapter.create_device().expect("create Vulkan device");
        assert!(device.inner.subgroup_size_control);

        let pipeline = device
            .create_compute_pipeline(
                HalShaderSource::SpirV(compute_spirv_with_workgroup_size_x(caps.min_size)),
                "main",
                (caps.min_size, 1, 1),
                &[],
                0,
                Some(caps.min_size),
            )
            .expect("create compute pipeline with a required subgroup size");
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
