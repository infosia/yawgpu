use std::ptr::NonNull;
use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::BACKEND;
use crate::{HalBufferUsage, HalError};

pub(super) struct GlesBufferInner {
    device: Arc<GlesDeviceInner>,
    buffer: Result<glow::Buffer, HalError>,
    size: u64,
}

impl Drop for GlesBufferInner {
    fn drop(&mut self) {
        if let Ok(buffer) = self.buffer.as_ref() {
            let buffer = *buffer;
            let _ = self.device.with_current_context(|gl| unsafe {
                gl.delete_buffer(buffer);
            });
        }
    }
}

/// Stores GLES buffer data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesBuffer {
    inner: Arc<GlesBufferInner>,
}

// SAFETY: `GlesBuffer` accesses GL state only through `GlesDeviceInner`, whose
// make-current lock serializes all GL commands.
unsafe impl Send for GlesBuffer {}
// SAFETY: See the `Send` impl; shared operations are synchronized by the
// owning device inner.
unsafe impl Sync for GlesBuffer {}

impl std::fmt::Debug for GlesBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesBuffer")
            .field("size", &self.inner.size)
            .finish()
    }
}

impl GlesBuffer {
    pub(super) fn new(device: Arc<GlesDeviceInner>, size: u64, _usage: HalBufferUsage) -> Self {
        let buffer = allocate_buffer(&device, size);
        Self {
            inner: Arc::new(GlesBufferInner {
                device,
                buffer,
                size,
            }),
        }
    }

    /// Returns the size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.inner.size
    }

    /// Records a write command.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), HalError> {
        if data.is_empty() {
            return Ok(());
        }
        check_range(offset, data.len() as u64, self.inner.size, "buffer write")?;
        let offset = i32::try_from(offset).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer write offset exceeds GLES limit",
        })?;
        let buffer = self.raw_or_err()?;
        self.inner.device.with_current_context(|gl| unsafe {
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
            gl.buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, offset, data);
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
        })
    }

    /// Reads `len` bytes at `offset` back from the buffer into host memory.
    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        if len == 0 {
            return Ok(Vec::new());
        }
        check_range(offset, len, self.inner.size, "buffer read")?;
        let offset = i32::try_from(offset).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer read offset exceeds GLES limit",
        })?;
        let len_i32 = i32::try_from(len).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer read length exceeds GLES limit",
        })?;
        let len_usize = usize::try_from(len).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer read length exceeds host limit",
        })?;
        let buffer = self.raw_or_err()?;
        self.inner.device.with_current_context(|gl| unsafe {
            gl.bind_buffer(glow::COPY_READ_BUFFER, Some(buffer));
            let ptr =
                gl.map_buffer_range(glow::COPY_READ_BUFFER, offset, len_i32, glow::MAP_READ_BIT);
            if ptr.is_null() {
                gl.bind_buffer(glow::COPY_READ_BUFFER, None);
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "glMapBufferRange failed",
                });
            }
            let out = std::slice::from_raw_parts(ptr, len_usize).to_vec();
            gl.unmap_buffer(glow::COPY_READ_BUFFER);
            gl.bind_buffer(glow::COPY_READ_BUFFER, None);
            Ok(out)
        })?
    }

    /// Returns mapped ptr.
    #[must_use]
    pub fn mapped_ptr(&self) -> Option<NonNull<u8>> {
        None
    }

    pub(super) fn raw_or_err(&self) -> Result<glow::Buffer, HalError> {
        self.inner
            .buffer
            .as_ref()
            .copied()
            .map_err(rebuild_hal_error)
    }
}

fn allocate_buffer(device: &Arc<GlesDeviceInner>, size: u64) -> Result<glow::Buffer, HalError> {
    device
        .with_current_context(|gl| unsafe {
            let buffer = gl
                .create_buffer()
                .map_err(|_| HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "glCreateBuffer failed",
                })?;
            let size = i32::try_from(size).map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "buffer size exceeds GLES limit",
            })?;
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
            gl.buffer_data_size(glow::COPY_WRITE_BUFFER, size, glow::DYNAMIC_DRAW);
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            Ok(buffer)
        })
        .and_then(|result| result)
}

fn rebuild_hal_error(error: &HalError) -> HalError {
    match error {
        HalError::BackendUnavailable { backend } => HalError::BackendUnavailable { backend },
        HalError::DeviceCreationFailed { backend } => HalError::DeviceCreationFailed { backend },
        HalError::QueueSubmissionFailed { backend } => HalError::QueueSubmissionFailed { backend },
        HalError::BufferOperationFailed { backend, message } => {
            HalError::BufferOperationFailed { backend, message }
        }
        HalError::ShaderCompilationFailed { backend, message } => {
            HalError::ShaderCompilationFailed {
                backend,
                message: message.clone(),
            }
        }
        HalError::SwapchainCreationFailed { backend, message } => {
            HalError::SwapchainCreationFailed { backend, message }
        }
        HalError::AcquireFailed { backend, message } => {
            HalError::AcquireFailed { backend, message }
        }
        HalError::PresentFailed { backend, message } => {
            HalError::PresentFailed { backend, message }
        }
    }
}

fn check_range(offset: u64, len: u64, size: u64, operation: &'static str) -> Result<(), HalError> {
    let end = offset
        .checked_add(len)
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer operation range overflow",
        })?;
    if end > size {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: operation_range_error(operation),
        });
    }
    Ok(())
}

fn operation_range_error(operation: &'static str) -> &'static str {
    match operation {
        "buffer write" => "buffer write range exceeds buffer size",
        "buffer read" => "buffer read range exceeds buffer size",
        _ => "buffer operation range exceeds buffer size",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_range_accepts_empty_and_in_bounds_ranges() {
        assert!(check_range(0, 0, 16, "buffer read").is_ok());
        assert!(check_range(4, 8, 16, "buffer write").is_ok());
        assert!(check_range(16, 0, 16, "buffer read").is_ok());
    }

    #[test]
    fn check_range_rejects_overflow_and_out_of_bounds_ranges() {
        let overflow =
            check_range(u64::MAX, 1, 16, "buffer read").expect_err("overflow must be rejected");
        assert!(matches!(
            overflow,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "buffer operation range overflow",
            }
        ));

        let out_of_bounds =
            check_range(12, 8, 16, "buffer write").expect_err("OOB must be rejected");
        assert!(matches!(
            out_of_bounds,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "buffer write range exceeds buffer size",
            }
        ));
    }
}
