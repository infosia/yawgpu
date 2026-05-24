use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::BACKEND;
use crate::{HalBuffer, HalBufferCopy, HalCopy, HalError};

/// Stores GLES queue data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesQueue {
    inner: Arc<GlesDeviceInner>,
}

// SAFETY: Queue submission calls into `GlesDeviceInner::with_current_context`,
// which serializes context binding and GL commands.
unsafe impl Send for GlesQueue {}
// SAFETY: See the `Send` impl; shared submission is synchronized by the device
// inner lock.
unsafe impl Sync for GlesQueue {}

impl std::fmt::Debug for GlesQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesQueue").finish()
    }
}

impl GlesQueue {
    pub(super) fn new(inner: Arc<GlesDeviceInner>) -> Self {
        Self { inner }
    }

    /// Submits an empty command buffer to flush the queue.
    pub fn submit_empty(&self) -> Result<(), HalError> {
        self.inner.with_current_context(|gl| unsafe {
            gl.flush();
        })
    }

    /// Records and submits the given buffer/texture copy operations.
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
        if copies.is_empty() {
            return Ok(());
        }

        self.inner
            .with_current_context(|gl| -> Result<(), HalError> {
                for copy in copies {
                    match copy {
                        HalCopy::Buffer(copy) => submit_buffer_copy(gl, copy)?,
                        _ => {
                            return Err(HalError::BufferOperationFailed {
                                backend: BACKEND,
                                message:
                                    "GLES backend supports only buffer-to-buffer copies in P15.2",
                            });
                        }
                    }
                }
                unsafe {
                    gl.flush();
                }
                Ok(())
            })?
    }
}

fn submit_buffer_copy(gl: &glow::Context, copy: &HalBufferCopy) -> Result<(), HalError> {
    let HalBuffer::Gles(source) = &copy.source else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer copy source is not a GLES buffer",
        });
    };
    let HalBuffer::Gles(destination) = &copy.destination else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer copy destination is not a GLES buffer",
        });
    };

    let source_buffer = source.raw_or_err()?;
    let destination_buffer = destination.raw_or_err()?;
    let source_offset =
        i32::try_from(copy.source_offset).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer copy source offset exceeds GLES limit",
        })?;
    let destination_offset =
        i32::try_from(copy.destination_offset).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer copy destination offset exceeds GLES limit",
        })?;
    let size = i32::try_from(copy.size).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message: "buffer copy size exceeds GLES limit",
    })?;

    unsafe {
        gl.bind_buffer(glow::COPY_READ_BUFFER, Some(source_buffer));
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(destination_buffer));
        gl.copy_buffer_sub_data(
            glow::COPY_READ_BUFFER,
            glow::COPY_WRITE_BUFFER,
            source_offset,
            destination_offset,
            size,
        );
        gl.bind_buffer(glow::COPY_READ_BUFFER, None);
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
    }

    Ok(())
}
