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
    pub fn create_buffer(
        &self,
        size: u64,
        _usage: HalBufferUsage,
    ) -> Result<MetalBuffer, HalError> {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        // MTLBuffer has no per-usage validation; the parameter is accepted
        // for HAL symmetry.
        //
        // Metal returns nil for `newBufferWithLength(0)`, but WebGPU permits
        // zero-size buffers. Allocate at least 1 byte so Metal always returns
        // a valid MTLBuffer; the *logical* size field keeps the requested
        // value so all map/read/write bounds validate against 0 (mirrors the
        // wgpu Metal backend).
        let alloc_size = usize::try_from(size).unwrap_or(usize::MAX).max(1);
        let buffer = self.device.newBufferWithLength_options(
            alloc_size,
            MTLResourceOptions::StorageModeShared,
        );
        let Some(buffer) = buffer else {
            return Err(HalError::OutOfMemory {
                backend: BACKEND,
                resource: "buffer",
            });
        };
        let mapped_ptr = Some(buffer.contents().cast::<u8>());
        Ok(MetalBuffer {
            inner: Some(buffer),
            mapped_ptr,
            size,
        })
    }

    /// Creates a texture matching the given descriptor.
    pub fn create_texture(
        &self,
        descriptor: &HalTextureDescriptor,
    ) -> Result<MetalTexture, HalError> {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let (inner, bytes_per_pixel) = create_texture(&self.device, descriptor)?;
        Ok(MetalTexture {
            inner: Some(inner),
            dimension: descriptor.dimension,
            width: descriptor.width,
            height: descriptor.height,
            depth_or_array_layers: descriptor.depth_or_array_layers,
            sample_count: descriptor.sample_count,
            bytes_per_pixel,
        })
    }

    /// Creates a query set matching the given kind and count.
    pub fn create_query_set(
        &self,
        kind: HalQueryKind,
        count: u32,
    ) -> Result<MetalQuerySet, HalError> {
        match kind {
            HalQueryKind::Occlusion => {
                self.allocations.fetch_add(1, Ordering::Relaxed);
                MetalQuerySet::new(&self.device, count)
            }
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
        let (msl_source, buffer_sizes_slot, buffer_size_bindings) = match shader {
            HalShaderSource::Msl(source) => (source, None, Vec::new()),
            HalShaderSource::MslWithBufferSizes {
                source,
                buffer_sizes_slot,
                buffer_size_bindings,
            } => (source, buffer_sizes_slot, buffer_size_bindings),
            _ => {
                return Err(shader_error(
                    "Metal compute pipeline requires MSL".to_owned(),
                ));
            }
        };
        create_compute_pipeline(
            &self.device,
            &msl_source,
            entry_point,
            workgroup_size,
            buffer_sizes_slot,
            buffer_size_bindings,
        )
    }

    /// Creates a render pipeline from the given shaders, vertex layout, and color targets.
    pub fn create_render_pipeline(
        &self,
        shader: HalShaderSource,
        vertex_entry_point: &str,
        fragment_entry_point: Option<&str>,
        descriptor: &HalRenderPipelineDescriptor,
        _bindings: &[HalDescriptorBinding],
    ) -> Result<MetalRenderPipeline, HalError> {
        create_render_pipeline(
            &self.device,
            shader,
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
        adjusted.color_targets = pass_layout
            .color_attachments
            .iter()
            .enumerate()
            .map(|(index, attachment)| {
                let state = descriptor.color_targets.get(index).copied().flatten();
                Some(HalColorTargetState {
                    format: attachment.format,
                    blend: state.and_then(|state| state.blend),
                    write_mask: state.map_or(0xf, |state| state.write_mask),
                })
            })
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
            Some(fragment_entry_point),
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
        let buffer = device
            .create_buffer(16, HalBufferUsage::default())
            .expect("Metal buffer allocation should succeed");
        assert_eq!(buffer.size(), 16);
        assert!(buffer.mapped_ptr().is_some());
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_create_zero_size_buffer_succeeds_with_logical_size_zero() {
        // F-072: Metal returns nil for newBufferWithLength(0); we allocate 1
        // byte internally but expose size=0 so all map/range checks see 0.
        let device = metal_device();
        let buffer = device
            .create_buffer(0, HalBufferUsage::default())
            .expect("zero-size Metal buffer should succeed (F-072)");
        assert_eq!(buffer.size(), 0, "logical size must be 0");
        assert!(
            buffer.mapped_ptr().is_some(),
            "mapped_ptr must be Some (1-byte backing allocation)"
        );
        assert_eq!(
            buffer.read(0, 0).expect("zero-length read should succeed"),
            Vec::<u8>::new(),
        );
        buffer.write(0, &[]).expect("zero-length write should succeed");
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_create_texture_records_descriptor_shape() {
        let device = metal_device();
        let texture = device
            .create_texture(&texture_descriptor())
            .expect("Metal texture allocation should succeed");
        assert_eq!(texture.width, 4);
        assert_eq!(texture.height, 4);
        assert_eq!(texture.depth_or_array_layers, 1);
        assert_eq!(texture.bytes_per_pixel, 4);
        assert!(texture.inner.is_some());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_oversized_allocations_return_out_of_memory() {
        // Dimensions must stay within Metal's MTLTextureDescriptor hard limits
        // (width/height ≤ 16384, array_length ≤ 2048) or Metal will ASSERT before
        // newTextureWithDescriptor can return nil.  Use the maximum legal dimensions
        // so the total byte size (16384×16384×4 bytes × 2048 layers = 2 TiB) is
        // impossible to satisfy, causing newTextureWithDescriptor to return nil → OutOfMemory.
        let device = metal_device();
        assert!(matches!(
            device.create_buffer(u64::MAX, HalBufferUsage::default()),
            Err(HalError::OutOfMemory {
                backend: BACKEND,
                resource: "buffer"
            })
        ));

        let mut descriptor = texture_descriptor();
        descriptor.width = 16384;
        descriptor.height = 16384;
        descriptor.depth_or_array_layers = 2048;
        assert!(matches!(
            device.create_texture(&descriptor),
            Err(HalError::OutOfMemory {
                backend: BACKEND,
                resource: "texture"
            })
        ));
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
                Some("fs_main"),
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
