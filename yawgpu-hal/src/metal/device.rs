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
        let buffer = self
            .device
            .newBufferWithLength_options(alloc_size, MTLResourceOptions::StorageModeShared);
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
            format: descriptor.format,
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
        let (msl_source, buffer_sizes_slot, buffer_size_bindings, workgroup_memory_sizes) =
            match shader {
                HalShaderSource::Msl(source) => (source, None, Vec::new(), Vec::new()),
                HalShaderSource::MslWithBufferSizes {
                    source,
                    buffer_sizes_slot,
                    buffer_size_bindings,
                    workgroup_memory_sizes,
                } => (
                    source,
                    buffer_sizes_slot,
                    buffer_size_bindings,
                    workgroup_memory_sizes,
                ),
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
            workgroup_memory_sizes,
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
        buffer
            .write(0, &[])
            .expect("zero-length write should succeed");
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
        // A pipeline created from plain Msl (no workgroup vars) has empty sizes.
        assert!(pipeline.workgroup_memory_sizes.is_empty());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_device_create_compute_pipeline_stores_workgroup_memory_sizes() {
        // Build the MSL source that naga would emit for a shader with two
        // var<workgroup> arguments: [[threadgroup(0)]] and [[threadgroup(1)]].
        // The sizes are pre-rounded (32, 16) as yawgpu-core would supply them.
        let source = r#"
#include <metal_stdlib>
using namespace metal;
kernel void cs(
    uint3 gid [[thread_position_in_grid]],
    threadgroup uint* tg_a [[threadgroup(0)]],
    threadgroup float* tg_b [[threadgroup(1)]])
{
    tg_a[0] = 1u;
    *tg_b = 2.0;
}
"#
        .to_owned();
        let shader = HalShaderSource::MslWithBufferSizes {
            source,
            buffer_sizes_slot: None,
            buffer_size_bindings: Vec::new(),
            workgroup_memory_sizes: vec![32, 16],
        };
        let device = metal_device();
        let pipeline = device
            .create_compute_pipeline(shader, "cs", (1, 1, 1), &[])
            .expect("workgroup memory pipeline should compile");
        // Verify the sizes are stored on the pipeline, ready for
        // setThreadgroupMemoryLength:atIndex: at dispatch time.
        assert_eq!(
            pipeline.workgroup_memory_sizes,
            vec![32, 16],
            "workgroup_memory_sizes must be stored on the Metal compute pipeline"
        );
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
