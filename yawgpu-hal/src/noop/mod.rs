use std::sync::atomic::{AtomicU64, Ordering};

use crate::HalError;

#[derive(Debug, Clone)]
pub struct NoopInstance;

impl NoopInstance {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

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

#[derive(Debug, Clone)]
pub struct NoopAdapter {
    name: &'static str,
}

impl NoopAdapter {
    #[must_use]
    pub fn synthetic() -> Self {
        Self {
            name: "yawgpu Noop Adapter",
        }
    }

    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn create_device(&self) -> Result<NoopDevice, HalError> {
        Ok(NoopDevice::new())
    }
}

#[derive(Debug)]
pub struct NoopDevice {
    allocations: AtomicU64,
    queue: NoopQueue,
}

impl NoopDevice {
    #[must_use]
    pub fn new() -> Self {
        Self {
            allocations: AtomicU64::new(0),
            queue: NoopQueue::new(),
        }
    }

    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn queue(&self) -> &NoopQueue {
        &self.queue
    }

    #[must_use]
    pub fn create_buffer(&self, size: u64) -> NoopBuffer {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        NoopBuffer { size }
    }
}

impl Default for NoopDevice {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct NoopQueue;

impl NoopQueue {
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

#[derive(Debug, Clone)]
pub struct NoopBuffer {
    size: u64,
}

impl NoopBuffer {
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }
}
