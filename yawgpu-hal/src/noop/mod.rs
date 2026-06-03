use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{HalBufferUsage, HalCopy, HalError, HalTextureDescriptor, HalTextureDimension};

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
    #[must_use]
    pub fn create_buffer(&self, size: u64, _usage: HalBufferUsage) -> NoopBuffer {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        NoopBuffer { size }
    }

    /// Creates a texture matching the given descriptor.
    #[must_use]
    pub fn create_texture(&self, descriptor: &HalTextureDescriptor) -> NoopTexture {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        NoopTexture {
            dimension: descriptor.dimension,
            width: descriptor.width,
            height: descriptor.height,
            depth_or_array_layers: descriptor.depth_or_array_layers,
            mip_level_count: descriptor.mip_level_count,
        }
    }

    /// Creates a transient attachment matching the given descriptor.
    #[cfg(feature = "tiled")]
    #[must_use]
    pub fn create_transient_attachment(&self) -> NoopTransientAttachment {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        NoopTransientAttachment
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
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
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
}

impl NoopBuffer {
    /// Returns the size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns mapped ptr.
    #[must_use]
    pub fn mapped_ptr(&self) -> Option<std::ptr::NonNull<u8>> {
        None
    }
}

/// Stores noop texture data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopTexture {
    dimension: HalTextureDimension,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    mip_level_count: u32,
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
}

/// Stores noop transient attachment data used by validation and backend submission.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
pub struct NoopTransientAttachment;

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
        let _buffer = device.create_buffer(4, HalBufferUsage::default());
        assert_eq!(device.allocation_count(), 1);
        let _texture = device.create_texture(&texture_descriptor());
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
        let buffer = device.create_buffer(64, HalBufferUsage::default());

        assert_eq!(buffer.size(), 64);
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn noop_device_create_texture_increments_allocation_count() {
        let device = NoopDevice::new();
        let _texture = device.create_texture(&texture_descriptor());

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

        let texture = device.create_texture(&descriptor);

        assert_eq!(texture.dimension(), HalTextureDimension::D3);
        assert_eq!(texture.width(), 8);
        assert_eq!(texture.height(), 4);
        assert_eq!(texture.depth_or_array_layers(), 3);
        assert_eq!(texture.mip_level_count(), 4);
        assert_eq!(device.allocation_count(), 1);
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn noop_device_create_transient_attachment_increments_allocation_count() {
        let device = NoopDevice::new();
        let _attachment = device.create_transient_attachment();

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

        assert_eq!(device.create_buffer(0, HalBufferUsage::default()).size(), 0);
        assert_eq!(
            device.create_buffer(4096, HalBufferUsage::default()).size(),
            4096
        );
    }

    #[test]
    fn noop_buffer_mapped_ptr_returns_none() {
        let device = NoopDevice::new();
        let buffer = device.create_buffer(128, HalBufferUsage::default());

        assert!(buffer.mapped_ptr().is_none());
    }
}
