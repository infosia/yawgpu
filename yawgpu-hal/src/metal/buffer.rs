use super::*;

/// Stores metal buffer data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalBuffer {
    pub(super) inner: Option<Retained<ProtocolObject<dyn MTLBufferTrait>>>,
    pub(super) mapped_ptr: Option<NonNull<u8>>,
    pub(super) size: u64,
}

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl std::fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("size", &self.size)
            .field("mapped", &self.mapped_ptr.is_some())
            .finish()
    }
}

impl MetalBuffer {
    /// Returns the size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Records a write command.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), HalError> {
        let len = u64::try_from(data.len()).map_err(|_| buffer_error("write size is too large"))?;
        self.validate_range(offset, len)?;
        if data.is_empty() {
            return Ok(());
        }
        let mapped_ptr = self
            .mapped_ptr
            .ok_or_else(|| buffer_error("buffer contents are unavailable"))?;
        let offset = usize::try_from(offset).map_err(|_| buffer_error("offset is too large"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                mapped_ptr.as_ptr().add(offset),
                data.len(),
            );
        }
        Ok(())
    }

    /// Reads `len` bytes at `offset` back from the buffer into host memory.
    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        self.validate_range(offset, len)?;
        let len = usize::try_from(len).map_err(|_| buffer_error("read length is too large"))?;
        let mut data = vec![0; len];
        if data.is_empty() {
            return Ok(data);
        }
        let mapped_ptr = self
            .mapped_ptr
            .ok_or_else(|| buffer_error("buffer contents are unavailable"))?;
        let offset = usize::try_from(offset).map_err(|_| buffer_error("offset is too large"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(mapped_ptr.as_ptr().add(offset), data.as_mut_ptr(), len);
        }
        Ok(data)
    }

    /// Returns mapped ptr.
    #[must_use]
    pub fn mapped_ptr(&self) -> Option<NonNull<u8>> {
        self.mapped_ptr
    }

    /// Returns the backing `MTLBuffer`, or an error if allocation failed.
    pub(super) fn inner(&self) -> Result<&ProtocolObject<dyn MTLBufferTrait>, HalError> {
        self.inner
            .as_deref()
            .ok_or_else(|| buffer_error("buffer allocation failed"))
    }

    /// Validates range and returns a descriptive error on failure.
    pub(super) fn validate_range(&self, offset: u64, len: u64) -> Result<(), HalError> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| buffer_error("buffer range overflows"))?;
        if end > self.size {
            return Err(buffer_error("buffer range exceeds buffer size"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_buffer_size_returns_created_size() {
        let buffer = metal_device()
            .create_buffer(32, HalBufferUsage::default())
            .expect("Metal buffer allocation should succeed");
        assert_eq!(buffer.size(), 32);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_buffer_write_updates_mapped_memory() {
        let buffer = metal_device()
            .create_buffer(4, HalBufferUsage::default())
            .expect("Metal buffer allocation should succeed");
        buffer.write(0, &[5, 6, 7, 8]).expect("write buffer");
        assert_eq!(buffer.read(0, 4).expect("read buffer"), [5, 6, 7, 8]);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_buffer_read_returns_written_bytes() {
        let buffer = metal_device()
            .create_buffer(4, HalBufferUsage::default())
            .expect("Metal buffer allocation should succeed");
        buffer.write(1, &[9, 10]).expect("write buffer");
        assert_eq!(buffer.read(1, 2).expect("read buffer"), [9, 10]);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_buffer_mapped_ptr_returns_non_null_pointer() {
        let buffer = metal_device()
            .create_buffer(4, HalBufferUsage::default())
            .expect("Metal buffer allocation should succeed");
        assert!(buffer.mapped_ptr().is_some());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_buffer_zero_size_read_returns_empty_and_write_is_noop() {
        // F-072: zero-size buffers must be usable on Metal; the 1-byte
        // backing allocation must not surface to callers.
        let buffer = metal_device()
            .create_buffer(0, HalBufferUsage::default())
            .expect("zero-size Metal buffer must succeed");
        assert_eq!(buffer.size(), 0);
        assert!(buffer.mapped_ptr().is_some());
        assert_eq!(
            buffer.read(0, 0).expect("zero-length read must succeed"),
            Vec::<u8>::new(),
        );
        buffer
            .write(0, &[])
            .expect("zero-length write must succeed");
    }
}
