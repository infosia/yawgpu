use std::sync::atomic::{AtomicU64, Ordering};

use crate::HalError;

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
    pub fn create_buffer(&self, size: u64) -> NoopBuffer {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        NoopBuffer { size }
    }

    /// Creates a texture matching the given descriptor.
    #[must_use]
    pub fn create_texture(&self) -> NoopTexture {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        NoopTexture
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
pub struct NoopQueue;

impl NoopQueue {
    /// Creates a new instance.
    #[must_use]
    pub fn new() -> Self {
        Self
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
pub struct NoopTexture;

/// Stores noop sampler data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopSampler;

#[cfg(test)]
mod tests {
    use super::*;

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
        let _buffer = device.create_buffer(4);
        assert_eq!(device.allocation_count(), 1);
        let _texture = device.create_texture();
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
        let buffer = device.create_buffer(64);

        assert_eq!(buffer.size(), 64);
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn noop_device_create_texture_increments_allocation_count() {
        let device = NoopDevice::new();
        let _texture = device.create_texture();

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

        assert_eq!(device.create_buffer(0).size(), 0);
        assert_eq!(device.create_buffer(4096).size(), 4096);
    }

    #[test]
    fn noop_buffer_mapped_ptr_returns_none() {
        let device = NoopDevice::new();
        let buffer = device.create_buffer(128);

        assert!(buffer.mapped_ptr().is_none());
    }
}
