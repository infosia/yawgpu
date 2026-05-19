use std::sync::atomic::{AtomicU64, Ordering};

use crate::HalError;

const BACKEND: &str = "metal";

#[derive(Debug)]
pub struct MetalInstance;

impl MetalInstance {
    pub fn new() -> Result<Self, HalError> {
        Ok(Self)
    }

    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<MetalAdapter> {
        metal::Device::all()
            .into_iter()
            .map(MetalAdapter::new)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct MetalAdapter {
    device: metal::Device,
    name: String,
}

impl MetalAdapter {
    #[must_use]
    pub fn new(device: metal::Device) -> Self {
        let name = device.name().to_owned();
        Self { device, name }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn create_device(&self) -> Result<MetalDevice, HalError> {
        let queue = self.device.new_command_queue();
        Ok(MetalDevice {
            _device: self.device.clone(),
            allocations: AtomicU64::new(0),
            queue: MetalQueue { inner: queue },
        })
    }
}

#[derive(Debug)]
pub struct MetalDevice {
    _device: metal::Device,
    allocations: AtomicU64,
    queue: MetalQueue,
}

impl MetalDevice {
    pub fn new() -> Result<Self, HalError> {
        let device = metal::Device::system_default()
            .ok_or(HalError::BackendUnavailable { backend: BACKEND })?;
        MetalAdapter::new(device).create_device()
    }

    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn queue(&self) -> &MetalQueue {
        &self.queue
    }

    #[must_use]
    pub fn create_buffer(&self, size: u64) -> MetalBuffer {
        // P7.2: allocate a real MTLBuffer.
        self.allocations.fetch_add(1, Ordering::Relaxed);
        MetalBuffer { size }
    }

    #[must_use]
    pub fn create_texture(&self) -> MetalTexture {
        // P7.3: allocate a real MTLTexture.
        self.allocations.fetch_add(1, Ordering::Relaxed);
        MetalTexture
    }

    #[must_use]
    pub fn create_sampler(&self) -> MetalSampler {
        // P7.3: allocate a real MTLSamplerState.
        self.allocations.fetch_add(1, Ordering::Relaxed);
        MetalSampler
    }
}

#[derive(Debug, Clone)]
pub struct MetalQueue {
    inner: metal::CommandQueue,
}

impl MetalQueue {
    pub fn new() -> Result<Self, HalError> {
        let device = metal::Device::system_default()
            .ok_or(HalError::BackendUnavailable { backend: BACKEND })?;
        let inner = device.new_command_queue();
        Ok(Self { inner })
    }

    pub fn submit_empty(&self) -> Result<(), HalError> {
        let command_buffer = self.inner.new_command_buffer();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MetalBuffer {
    size: u64,
}

impl MetalBuffer {
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }
}

#[derive(Debug, Clone)]
pub struct MetalTexture;

#[derive(Debug, Clone)]
pub struct MetalSampler;
