use std::sync::atomic::{AtomicU64, Ordering};

use crate::{HalBuffer, HalBufferCopy, HalError};

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
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let buffer = self
            ._device
            .new_buffer(size, metal::MTLResourceOptions::StorageModeShared);
        MetalBuffer {
            inner: Some(buffer),
            size,
        }
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

    pub fn submit_buffer_copies(&self, copies: &[HalBufferCopy]) -> Result<(), HalError> {
        if copies.is_empty() {
            return Ok(());
        }

        let command_buffer = self.inner.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        for copy in copies {
            let HalBuffer::Metal(source) = &copy.source else {
                return Err(buffer_error("source buffer is not Metal-backed"));
            };
            let HalBuffer::Metal(destination) = &copy.destination else {
                return Err(buffer_error("destination buffer is not Metal-backed"));
            };
            source.validate_range(copy.source_offset, copy.size)?;
            destination.validate_range(copy.destination_offset, copy.size)?;
            let source_buffer = source.inner()?;
            let destination_buffer = destination.inner()?;
            blit.copy_from_buffer(
                source_buffer,
                to_ns(copy.source_offset)?,
                destination_buffer,
                to_ns(copy.destination_offset)?,
                to_ns(copy.size)?,
            );
        }
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MetalBuffer {
    inner: Option<metal::Buffer>,
    size: u64,
}

impl MetalBuffer {
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), HalError> {
        let len = u64::try_from(data.len()).map_err(|_| buffer_error("write size is too large"))?;
        self.validate_range(offset, len)?;
        if data.is_empty() {
            return Ok(());
        }
        let buffer = self.inner()?;
        let contents = buffer.contents();
        if contents.is_null() {
            return Err(buffer_error("buffer contents are unavailable"));
        }
        let offset = usize::try_from(offset).map_err(|_| buffer_error("offset is too large"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                contents.cast::<u8>().add(offset),
                data.len(),
            );
        }
        buffer.did_modify_range(metal::NSRange::new(
            to_ns(u64::try_from(offset).map_err(|_| buffer_error("offset is too large"))?)?,
            to_ns(len)?,
        ));
        Ok(())
    }

    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        self.validate_range(offset, len)?;
        let len = usize::try_from(len).map_err(|_| buffer_error("read length is too large"))?;
        let mut data = vec![0; len];
        if data.is_empty() {
            return Ok(data);
        }
        let buffer = self.inner()?;
        let contents = buffer.contents();
        if contents.is_null() {
            return Err(buffer_error("buffer contents are unavailable"));
        }
        let offset = usize::try_from(offset).map_err(|_| buffer_error("offset is too large"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents.cast::<u8>().add(offset),
                data.as_mut_ptr(),
                len,
            );
        }
        Ok(data)
    }

    fn inner(&self) -> Result<&metal::BufferRef, HalError> {
        self.inner
            .as_deref()
            .ok_or_else(|| buffer_error("buffer allocation failed"))
    }

    fn validate_range(&self, offset: u64, len: u64) -> Result<(), HalError> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| buffer_error("buffer range overflows"))?;
        if end > self.size {
            return Err(buffer_error("buffer range exceeds buffer size"));
        }
        Ok(())
    }
}

fn to_ns(value: u64) -> Result<metal::NSUInteger, HalError> {
    metal::NSUInteger::try_from(value).map_err(|_| buffer_error("value is too large"))
}

fn buffer_error(message: &'static str) -> HalError {
    HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    }
}

#[derive(Debug, Clone)]
pub struct MetalTexture;

#[derive(Debug, Clone)]
pub struct MetalSampler;
