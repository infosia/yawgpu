use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{
    HalBufferUsage, HalCopy, HalError, HalQueryKind, HalTextureDescriptor, HalTextureDimension,
};

/// Stores noop instance data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopInstance;

impl NoopInstance {
    /// Creates a new instance.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns adapters exposed by this instance.
    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<NoopAdapter> {
        vec![NoopAdapter::synthetic()]
    }
}

impl Default for NoopInstance {
    fn default() -> Self {
        Self::new()
    }
}

/// Stores noop adapter data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopAdapter {
    name: &'static str,
}

impl NoopAdapter {
    /// Builds the single synthetic adapter the Noop backend exposes.
    #[must_use]
    pub fn synthetic() -> Self {
        Self {
            name: "yawgpu Noop Adapter",
        }
    }

    /// Returns the name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns true when WGSL `shader-f16` is supported.
    #[must_use]
    pub(super) fn supports_shader_float16(&self) -> bool {
        true
    }

    /// Creates a device (and its default queue) on this adapter.
    pub fn create_device(&self) -> Result<NoopDevice, HalError> {
        Ok(NoopDevice::new())
    }
}

/// Stores noop device data used by validation and backend submission.
#[derive(Debug)]
pub struct NoopDevice {
    allocations: AtomicU64,
    queue: NoopQueue,
}

impl NoopDevice {
    /// Creates a new instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            allocations: AtomicU64::new(0),
            queue: NoopQueue::new(),
        }
    }

    /// Returns the allocation count.
    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }

    /// Returns the queue.
    #[must_use]
    pub fn queue(&self) -> &NoopQueue {
        &self.queue
    }

    /// Allocates a buffer of the given size on this device.
    pub fn create_buffer(&self, size: u64, _usage: HalBufferUsage) -> Result<NoopBuffer, HalError> {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(NoopBuffer::new(size))
    }

    /// Creates a query set of the given kind and count.
    #[must_use]
    pub fn create_query_set(&self, _kind: HalQueryKind, count: u32) -> u32 {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        count
    }

    /// Creates a texture matching the given descriptor.
    pub fn create_texture(
        &self,
        descriptor: &HalTextureDescriptor,
    ) -> Result<NoopTexture, HalError> {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(NoopTexture {
            dimension: descriptor.dimension,
            width: descriptor.width,
            height: descriptor.height,
            depth_or_array_layers: descriptor.depth_or_array_layers,
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
        })
    }

    /// Creates a sampler matching the given descriptor.
    #[must_use]
    pub fn create_sampler(&self) -> NoopSampler {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        NoopSampler
    }
}

impl Default for NoopDevice {
    fn default() -> Self {
        Self::new()
    }
}

/// Stores noop queue data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopQueue {
    submitted_copies: Arc<Mutex<Vec<HalCopy>>>,
}

impl NoopQueue {
    /// Creates a new instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            submitted_copies: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Records submitted copy commands for Noop unit-test inspection.
    ///
    /// `HalCopy::Buffer` copies are executed eagerly so that subsequent
    /// map-reads on the destination buffer observe the written bytes (mirrors
    /// the real-GPU semantics where the copy completes before any following
    /// `mapAsync` resolves).
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
        for copy in copies {
            match copy {
                HalCopy::Buffer(buf_copy) => {
                    // Read from source, write into destination in order to
                    // make the data visible for subsequent map-reads.
                    let data = buf_copy
                        .source
                        .read(buf_copy.source_offset, buf_copy.size)?;
                    buf_copy
                        .destination
                        .write(buf_copy.destination_offset, &data)?;
                }
                HalCopy::ResolveQuerySet(resolve) => {
                    let byte_count = resolve_query_byte_count(resolve.query_count)?;
                    let zeros = vec![0; byte_count];
                    resolve
                        .destination
                        .write(resolve.destination_offset, &zeros)?;
                }
                HalCopy::ClearTexture(_) => {}
                _ => {}
            }
        }
        self.submitted_copies
            .lock()
            .map_err(|_| HalError::QueueSubmissionFailed { backend: "noop" })?
            .extend(copies.iter().cloned());
        Ok(())
    }

    /// Waits until all submitted queue work has completed.
    pub fn wait_idle(&self) -> Result<(), HalError> {
        Ok(())
    }

    /// Returns submitted copy commands recorded by this queue.
    #[must_use]
    pub fn submitted_copies(&self) -> Vec<HalCopy> {
        self.submitted_copies
            .lock()
            .map(|copies| copies.clone())
            .unwrap_or_default()
    }
}

impl Default for NoopQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Stores noop buffer data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopBuffer {
    size: u64,
    data: Arc<Mutex<Vec<u8>>>,
}

impl NoopBuffer {
    /// Creates a new noop buffer with zero-initialized storage.
    #[must_use]
    pub fn new(size: u64) -> Self {
        let len = usize::try_from(size).unwrap_or(0);
        Self {
            size,
            data: Arc::new(Mutex::new(vec![0; len])),
        }
    }

    /// Returns the size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Writes bytes into the buffer.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), HalError> {
        let end = validate_noop_buffer_range(self.size, offset, data.len() as u64)?;
        let offset = usize::try_from(offset).map_err(|_| HalError::BufferOperationFailed {
            backend: "noop",
            message: "buffer offset is too large",
        })?;
        let mut storage = self
            .data
            .lock()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: "noop",
                message: "buffer storage lock failed",
            })?;
        if end > storage.len() {
            return Err(HalError::BufferOperationFailed {
                backend: "noop",
                message: "buffer storage is too small for range",
            });
        }
        storage[offset..end].copy_from_slice(data);
        Ok(())
    }

    /// Reads bytes from the buffer.
    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        let end = validate_noop_buffer_range(self.size, offset, len)?;
        let offset = usize::try_from(offset).map_err(|_| HalError::BufferOperationFailed {
            backend: "noop",
            message: "buffer offset is too large",
        })?;
        let storage = self
            .data
            .lock()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: "noop",
                message: "buffer storage lock failed",
            })?;
        if end > storage.len() {
            return Err(HalError::BufferOperationFailed {
                backend: "noop",
                message: "buffer storage is too small for range",
            });
        }
        Ok(storage[offset..end].to_vec())
    }

    /// Returns mapped ptr.
    #[must_use]
    pub fn mapped_ptr(&self) -> Option<std::ptr::NonNull<u8>> {
        None
    }
}

fn resolve_query_byte_count(query_count: u32) -> Result<usize, HalError> {
    usize::try_from(u64::from(query_count) * 8).map_err(|_| HalError::BufferOperationFailed {
        backend: "noop",
        message: "query resolve byte count is too large",
    })
}

fn validate_noop_buffer_range(size: u64, offset: u64, len: u64) -> Result<usize, HalError> {
    let end = offset
        .checked_add(len)
        .ok_or(HalError::BufferOperationFailed {
            backend: "noop",
            message: "buffer range overflows",
        })?;
    if end > size {
        return Err(HalError::BufferOperationFailed {
            backend: "noop",
            message: "buffer range exceeds buffer size",
        });
    }
    usize::try_from(end).map_err(|_| HalError::BufferOperationFailed {
        backend: "noop",
        message: "buffer range is too large",
    })
}

/// Stores noop texture data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopTexture {
    dimension: HalTextureDimension,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    mip_level_count: u32,
    sample_count: u32,
}

impl NoopTexture {
    /// Returns the texture dimension.
    #[must_use]
    pub fn dimension(&self) -> HalTextureDimension {
        self.dimension
    }

    /// Returns the texture width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the texture height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the texture depth or array layer count.
    #[must_use]
    pub fn depth_or_array_layers(&self) -> u32 {
        self.depth_or_array_layers
    }

    /// Returns the mip level count.
    #[must_use]
    pub fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }

    /// Returns the texture sample count.
    #[must_use]
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }
}

/// Stores noop sampler data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopSampler;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HalTextureFormat, HalTextureUsage};

    fn texture_descriptor() -> HalTextureDescriptor {
        HalTextureDescriptor {
            dimension: HalTextureDimension::D2,
            format: HalTextureFormat::Rgba8Unorm,
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
            usage: HalTextureUsage {
                copy_src: true,
                copy_dst: true,
                texture_binding: false,
                storage_binding: false,
                render_attachment: true,
                transient: false,
            },
        }
    }

    #[test]
    fn noop_instance_new_constructs() {
        let instance = NoopInstance::new();

        assert_eq!(instance.enumerate_adapters().len(), 1);
    }

    #[test]
    fn noop_instance_enumerate_adapters_returns_synthetic_adapter() {
        let instance = NoopInstance::new();
        let adapters = instance.enumerate_adapters();

        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].name(), "yawgpu Noop Adapter");
    }

    #[test]
    fn noop_adapter_synthetic_exposes_documented_name() {
        let adapter = NoopAdapter::synthetic();

        assert_eq!(adapter.name(), "yawgpu Noop Adapter");
    }

    #[test]
    fn noop_adapter_name_returns_fixed_string() {
        let adapter = NoopAdapter::synthetic();

        assert_eq!(adapter.name(), "yawgpu Noop Adapter");
    }

    #[test]
    fn noop_adapter_supports_shader_float16_returns_true() {
        let adapter = NoopAdapter::synthetic();

        assert!(adapter.supports_shader_float16());
    }

    #[test]
    fn noop_adapter_create_device_returns_zero_allocation_device() {
        let adapter = NoopAdapter::synthetic();
        let device = adapter
            .create_device()
            .expect("Noop device creation succeeds");

        assert_eq!(device.allocation_count(), 0);
    }

    #[test]
    fn noop_device_new_starts_with_zero_allocations() {
        let device = NoopDevice::new();

        assert_eq!(device.allocation_count(), 0);
    }

    #[test]
    fn noop_device_allocation_count_tracks_created_resources() {
        let device = NoopDevice::new();

        assert_eq!(device.allocation_count(), 0);
        let _buffer = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("Noop buffer allocation should succeed");
        assert_eq!(device.allocation_count(), 1);
        let _texture = device
            .create_texture(&texture_descriptor())
            .expect("Noop texture allocation should succeed");
        assert_eq!(device.allocation_count(), 2);
        let _sampler = device.create_sampler();
        assert_eq!(device.allocation_count(), 3);
    }

    #[test]
    fn noop_device_queue_returns_same_reference() {
        let device = NoopDevice::new();

        assert!(std::ptr::eq(device.queue(), device.queue()));
    }

    #[test]
    fn noop_device_create_buffer_records_size_and_increments_allocation_count() {
        let device = NoopDevice::new();
        let buffer = device
            .create_buffer(64, HalBufferUsage::default())
            .expect("Noop buffer allocation should succeed");

        assert_eq!(buffer.size(), 64);
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn noop_device_create_texture_increments_allocation_count() {
        let device = NoopDevice::new();
        let _texture = device
            .create_texture(&texture_descriptor())
            .expect("Noop texture allocation should succeed");

        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn noop_device_create_texture_records_array_3d_and_mip_shape() {
        let device = NoopDevice::new();
        let mut descriptor = texture_descriptor();
        descriptor.dimension = HalTextureDimension::D3;
        descriptor.width = 8;
        descriptor.height = 4;
        descriptor.depth_or_array_layers = 3;
        descriptor.mip_level_count = 4;

        let texture = device
            .create_texture(&descriptor)
            .expect("Noop texture allocation should succeed");

        assert_eq!(texture.dimension(), HalTextureDimension::D3);
        assert_eq!(texture.width(), 8);
        assert_eq!(texture.height(), 4);
        assert_eq!(texture.depth_or_array_layers(), 3);
        assert_eq!(texture.mip_level_count(), 4);
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn noop_device_create_texture_accepts_multisample_descriptor() {
        let device = NoopDevice::new();
        let mut descriptor = texture_descriptor();
        descriptor.sample_count = 4;

        let texture = device
            .create_texture(&descriptor)
            .expect("Noop texture allocation should succeed");

        assert_eq!(texture.sample_count(), 4);
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn noop_device_create_sampler_increments_allocation_count() {
        let device = NoopDevice::new();
        let _sampler = device.create_sampler();

        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    #[allow(clippy::default_constructed_unit_structs)]
    fn noop_queue_new_matches_default_smoke() {
        let _queue = NoopQueue::new();
        let _default_queue = NoopQueue::default();
    }

    #[test]
    fn noop_buffer_size_returns_created_size() {
        let device = NoopDevice::new();

        assert_eq!(
            device
                .create_buffer(0, HalBufferUsage::default())
                .expect("Noop buffer allocation should succeed")
                .size(),
            0
        );
        assert_eq!(
            device
                .create_buffer(4096, HalBufferUsage::default())
                .expect("Noop buffer allocation should succeed")
                .size(),
            4096
        );
    }

    #[test]
    fn noop_buffer_mapped_ptr_returns_none() {
        let device = NoopDevice::new();
        let buffer = device
            .create_buffer(128, HalBufferUsage::default())
            .expect("Noop buffer allocation should succeed");

        assert!(buffer.mapped_ptr().is_none());
    }
}
