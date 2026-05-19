use std::sync::atomic::{AtomicU64, Ordering};

use crate::HalError;

use metal as _;

const BACKEND: &str = "metal";

#[derive(Debug)]
pub struct MetalInstance;

impl MetalInstance {
    pub fn new() -> Result<Self, HalError> {
        Err(HalError::BackendUnavailable { backend: BACKEND })
    }

    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<MetalAdapter> {
        Vec::new()
    }
}

#[derive(Debug, Clone)]
pub struct MetalAdapter {
    name: &'static str,
}

impl MetalAdapter {
    #[must_use]
    pub fn unavailable() -> Self {
        Self {
            name: "yawgpu Metal Adapter (unavailable)",
        }
    }

    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn create_device(&self) -> Result<MetalDevice, HalError> {
        Err(HalError::BackendUnavailable { backend: BACKEND })
    }
}

#[derive(Debug)]
pub struct MetalDevice {
    allocations: AtomicU64,
    queue: MetalQueue,
}

impl MetalDevice {
    pub fn new() -> Result<Self, HalError> {
        Err(HalError::BackendUnavailable { backend: BACKEND })
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
        self.allocations.fetch_add(1, Ordering::Relaxed);
        MetalBuffer { size }
    }

    #[must_use]
    pub fn create_texture(&self) -> MetalTexture {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        MetalTexture
    }

    #[must_use]
    pub fn create_sampler(&self) -> MetalSampler {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        MetalSampler
    }
}

#[derive(Debug, Clone)]
pub struct MetalQueue;

impl MetalQueue {
    pub fn new() -> Result<Self, HalError> {
        Err(HalError::BackendUnavailable { backend: BACKEND })
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
