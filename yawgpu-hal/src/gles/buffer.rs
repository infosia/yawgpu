use std::ptr::NonNull;

use super::unavailable;
use crate::HalError;

/// Stores GLES buffer data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesBuffer {
    size: u64,
}

impl std::fmt::Debug for GlesBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesBuffer")
            .field("size", &self.size)
            .finish()
    }
}

impl GlesBuffer {
    pub(super) fn new(size: u64) -> Self {
        Self { size }
    }

    /// Returns the size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Records a write command.
    pub fn write(&self, _offset: u64, _data: &[u8]) -> Result<(), HalError> {
        unavailable()
    }

    /// Reads `len` bytes at `offset` back from the buffer into host memory.
    pub fn read(&self, _offset: u64, _len: u64) -> Result<Vec<u8>, HalError> {
        unavailable()
    }

    /// Returns mapped ptr.
    #[must_use]
    pub fn mapped_ptr(&self) -> Option<NonNull<u8>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_backend_unavailable<T>(result: Result<T, HalError>) {
        assert!(matches!(
            result,
            Err(HalError::BackendUnavailable { backend: "gles" })
        ));
    }

    #[test]
    fn gles_buffer_size_matches_creation_size() {
        let buffer = GlesBuffer::new(256);

        assert_eq!(buffer.size(), 256);
    }

    #[test]
    fn gles_buffer_write_returns_unavailable() {
        let buffer = GlesBuffer::new(16);

        assert_backend_unavailable(buffer.write(0, &[1, 2, 3, 4]));
    }

    #[test]
    fn gles_buffer_read_returns_unavailable() {
        let buffer = GlesBuffer::new(16);

        assert_backend_unavailable(buffer.read(0, 4));
    }

    #[test]
    fn gles_buffer_mapped_ptr_returns_none() {
        let buffer = GlesBuffer::new(16);

        assert!(buffer.mapped_ptr().is_none());
    }
}
