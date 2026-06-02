use super::*;

/// Stores metal device data used by validation and backend submission.
pub struct MetalDevice {
    pub(super) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(super) allocations: AtomicU64,
    pub(super) queue: MetalQueue,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDevice")
            .field("allocations", &self.allocation_count())
            .finish()
    }
}

impl MetalDevice {
    /// Creates a new instance.
    pub fn new() -> Result<Self, HalError> {
        let adapter = MetalInstance::new()?
            .enumerate_adapters()
            .into_iter()
            .next()
            .ok_or(HalError::BackendUnavailable { backend: BACKEND })?;
        adapter.create_device()
    }

    /// Returns the allocation count.
    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }

    /// Returns the queue.
    #[must_use]
    pub fn queue(&self) -> &MetalQueue {
        &self.queue
    }

    /// Allocates a buffer of the given size on this device.
    #[must_use]
    pub fn create_buffer(&self, size: u64, _usage: HalBufferUsage) -> MetalBuffer {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        // MTLBuffer has no per-usage validation; the parameter is accepted
        // for HAL symmetry.
        let buffer = self.device.newBufferWithLength_options(
            usize::try_from(size).unwrap_or(usize::MAX),
            MTLResourceOptions::StorageModeShared,
        );
        let mapped_ptr = buffer.as_ref().map(|buffer| buffer.contents().cast::<u8>());
        MetalBuffer {
            inner: buffer,
            mapped_ptr,
            size,
        }
    }

    /// Creates a texture matching the given descriptor.
    #[must_use]
    pub fn create_texture(&self, descriptor: &HalTextureDescriptor) -> MetalTexture {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        match create_texture(&self.device, descriptor) {
            Ok((inner, bytes_per_pixel)) => MetalTexture {
                inner: Some(inner),
                dimension: descriptor.dimension,
                width: descriptor.width,
                height: descriptor.height,
                depth_or_array_layers: descriptor.depth_or_array_layers,
                bytes_per_pixel,
            },
            Err(_) => MetalTexture {
                inner: None,
                dimension: descriptor.dimension,
                width: descriptor.width,
                height: descriptor.height,
                depth_or_array_layers: descriptor.depth_or_array_layers,
                bytes_per_pixel: 0,
            },
        }
    }

    /// Creates a transient attachment matching the given concrete descriptor.
    #[cfg(feature = "tiled")]
    pub fn create_transient_attachment(
        &self,
        descriptor: &HalTransientAttachmentDescriptor,
    ) -> Result<MetalTransientAttachment, HalError> {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        create_transient_attachment(&self.device, descriptor)
    }

    /// Creates a sampler matching the given descriptor.
    #[must_use]
    pub fn create_sampler(&self, descriptor: &HalSamplerDescriptor) -> MetalSampler {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        MetalSampler {
            _inner: create_sampler(&self.device, descriptor).ok(),
        }
    }

    /// Creates a compute pipeline from the given shader, entry point, and bindings.
    pub fn create_compute_pipeline(
        &self,
        shader: HalShaderSource,
        entry_point: &str,
        workgroup_size: (u32, u32, u32),
        _bindings: &[HalDescriptorBinding],
    ) -> Result<MetalComputePipeline, HalError> {
        let HalShaderSource::Msl(msl_source) = shader else {
            return Err(shader_error(
                "Metal compute pipeline requires MSL".to_owned(),
            ));
        };
        create_compute_pipeline(&self.device, &msl_source, entry_point, workgroup_size)
    }

    /// Creates a render pipeline from the given shaders, vertex layout, and color targets.
    pub fn create_render_pipeline(
        &self,
        shader: HalShaderSource,
        vertex_entry_point: &str,
        fragment_entry_point: &str,
        descriptor: &HalRenderPipelineDescriptor,
        _bindings: &[HalDescriptorBinding],
    ) -> Result<MetalRenderPipeline, HalError> {
        let HalShaderSource::Msl(msl_source) = shader else {
            return Err(shader_error(
                "Metal render pipeline requires MSL".to_owned(),
            ));
        };
        create_render_pipeline(
            &self.device,
            &msl_source,
            vertex_entry_point,
            fragment_entry_point,
            descriptor,
        )
    }

    /// Creates a subpass-compatible render pipeline.
    ///
    /// The MTL pipeline gets one `colorAttachments[i].pixelFormat` per slot in
    /// the pass layout (not just the slots this subpass writes), so the
    /// pipeline's color-attachment layout matches the encoder's
    /// `MTLRenderPassDescriptor` slot-for-slot. The WGSL fragment's `@location(N)`
    /// — emitted by naga as MSL `[[color(N)]]` — lands in MTL slot N (the
    /// global flat numbering; see e2e shader comments mirroring mgpu's
    /// `hello_deferred`).
    ///
    /// When the caller's `descriptor.depth_stencil` is `None` but the pass
    /// layout declares a depth-stencil attachment, the MTL pipeline still
    /// needs `setDepthAttachmentPixelFormat` to match the encoder's bound
    /// depth attachment (Apple requires the format to match per
    /// `MTLRenderPipelineDescriptor`). Synthesize a no-op
    /// `HalDepthStencilState` carrying the layout's format so the pipeline
    /// declares the right `depthAttachmentPixelFormat`; the actual depth
    /// test/write stays disabled via the no-op `MTLDepthStencilState` that
    /// `create_render_pipeline` falls back to (depthCompare=Always,
    /// depthWrite=false, no stencil).
    // Each parameter here represents an orthogonal concern (shader source,
    // two entry points, the base render descriptor, bindings, the pass-level
    // layout, the subpass index) and matches the shapes the
    // `HalSubpassRenderPipelineDescriptor` carries from `yawgpu-core`. Folding
    // them into a struct would just be re-spelling the same eight values, so
    // accept the clippy warning at this site.
    #[cfg(feature = "tiled")]
    #[allow(clippy::too_many_arguments)]
    pub fn create_subpass_render_pipeline(
        &self,
        shader: HalShaderSource,
        vertex_entry_point: &str,
        fragment_entry_point: &str,
        descriptor: &HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
        pass_layout: &HalSubpassPassLayout,
        _subpass_index: u32,
    ) -> Result<MetalRenderPipeline, HalError> {
        let _ = bindings; // Metal subpass inputs use the color-slot map, not descriptors.
        let mut adjusted = descriptor.clone();
        adjusted.color_formats = pass_layout
            .color_attachments
            .iter()
            .map(|a| a.format)
            .collect();
        if adjusted.depth_stencil.is_none() {
            if let Some(layout_ds) = pass_layout.depth_stencil_attachment.as_ref() {
                let stencil_disabled = HalStencilFaceState {
                    compare: HalCompareFunction::Always,
                    fail_op: HalStencilOperation::Keep,
                    depth_fail_op: HalStencilOperation::Keep,
                    pass_op: HalStencilOperation::Keep,
                };
                adjusted.depth_stencil = Some(HalDepthStencilState {
                    format: layout_ds.format,
                    depth_write_enabled: false,
                    depth_compare: HalCompareFunction::Always,
                    stencil_front: stencil_disabled,
                    stencil_back: stencil_disabled,
                    stencil_read_mask: 0,
                    stencil_write_mask: 0,
                    depth_bias: 0,
                    depth_bias_slope_scale: 0.0,
                    depth_bias_clamp: 0.0,
                });
            }
        }
        self.create_render_pipeline(
            shader,
            vertex_entry_point,
            fragment_entry_point,
            &adjusted,
            &[],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_new_starts_with_zero_allocations() {
        let device = MetalDevice::new().expect("create Metal device");
        assert_eq!(device.allocation_count(), 0);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_allocation_count_tracks_created_resources() {
        let device = metal_device();
        assert_eq!(device.allocation_count(), 0);
        let _buffer = device.create_buffer(4, HalBufferUsage::default());
        let _texture = device.create_texture(&texture_descriptor());
        let _sampler = device.create_sampler(&sampler_descriptor());
        assert_eq!(device.allocation_count(), 3);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_queue_returns_same_reference() {
        let device = metal_device();
        assert!(std::ptr::eq(device.queue(), device.queue()));
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_create_buffer_records_size_and_maps_memory() {
        let device = metal_device();
        let buffer = device.create_buffer(16, HalBufferUsage::default());
        assert_eq!(buffer.size(), 16);
        assert!(buffer.mapped_ptr().is_some());
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_create_texture_records_descriptor_shape() {
        let device = metal_device();
        let texture = device.create_texture(&texture_descriptor());
        assert_eq!(texture.width, 4);
        assert_eq!(texture.height, 4);
        assert_eq!(texture.depth_or_array_layers, 1);
        assert_eq!(texture.bytes_per_pixel, 4);
        assert!(texture.inner.is_some());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_create_sampler_returns_sampler() {
        let device = metal_device();
        let sampler = device.create_sampler(&sampler_descriptor());
        assert!(sampler._inner.is_some());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_create_compute_pipeline_accepts_msl() {
        let device = metal_device();
        let pipeline = device
            .create_compute_pipeline(compute_msl(), "main0", (1, 1, 1), &[])
            .expect("create compute pipeline");
        assert_eq!(pipeline.workgroup_size, (1, 1, 1));
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_create_render_pipeline_accepts_msl() {
        let device = metal_device();
        let pipeline = device
            .create_render_pipeline(
                render_msl(),
                "vs_main",
                "fs_main",
                &render_descriptor(),
                &[],
            )
            .expect("create render pipeline");
        assert!(matches!(
            pipeline.primitive_topology,
            HalPrimitiveTopology::TriangleList
        ));
    }
}
