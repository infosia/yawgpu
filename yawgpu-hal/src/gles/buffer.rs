use std::ptr::NonNull;
use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::{rebuild_hal_error, BACKEND};
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
        self.inner
            .device
            .with_current_context(|gl| self.write_with_gl(gl, offset, data))?
    }

    /// Writes `data` at `offset` using an already-current GL context.
    ///
    /// The caller must already be inside
    /// `GlesDeviceInner::with_current_context` (i.e. hold the device's
    /// non-reentrant make-current lock); this fn does not re-acquire it.
    /// Queue submission uses this from within its own context closure —
    /// calling `write` there instead would self-deadlock (T-G4).
    pub(super) fn write_with_gl(
        &self,
        gl: &glow::Context,
        offset: u64,
        data: &[u8],
    ) -> Result<(), HalError> {
        if data.is_empty() {
            return Ok(());
        }
        check_range(offset, data.len() as u64, self.inner.size, "buffer write")?;
        let offset = i32::try_from(offset).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer write offset exceeds GLES limit",
        })?;
        let buffer = self.raw_or_err()?;
        unsafe {
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
            gl.buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, offset, data);
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
        }
        Ok(())
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
            // Validate the size fits the `i32` GLES limit (`glBufferData`
            // takes `GLsizeiptr`); the checked value is derived from the same
            // `u64` used for the zero vector below.
            i32::try_from(size).map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "buffer size exceeds GLES limit",
            })?;
            let size_as_usize =
                usize::try_from(size).map_err(|_| HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "buffer size exceeds host limit",
                })?;
            // WebGPU requires buffers to behave as zero-initialized on
            // creation. Unlike fresh Vulkan/Metal allocations (whose OS pages
            // come back zeroed), GL recycles freed-buffer memory within the
            // process, so `glBufferData(size, NULL)` would expose stale bytes
            // from a previously-destroyed buffer. Uploading a host zero vector
            // both allocates and initializes the storage to zero.
            let zeros = vec![0u8; size_as_usize];
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
            gl.buffer_data_u8_slice(glow::COPY_WRITE_BUFFER, &zeros, glow::DYNAMIC_DRAW);
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            Ok(buffer)
        })
        .and_then(|result| result)
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

    /// Creates a real EGL-backed device, self-skipping (returning `None`) when
    /// no GLES backend/adapter/device is available so the suite stays green on
    /// GPU-less CI (Noop-first).
    fn gles_device_or_skip(label: &str) -> Option<super::super::device::GlesDevice> {
        let instance = match super::super::instance::GlesInstance::new() {
            Ok(instance) => instance,
            Err(error) => {
                eprintln!("skipping {label}; backend unavailable: {error:?}");
                return None;
            }
        };
        let Some(adapter) = instance.enumerate_adapters().into_iter().next() else {
            eprintln!("skipping {label}; no adapter available");
            return None;
        };
        match adapter.create_device() {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping {label}; device unavailable: {error:?}");
                None
            }
        }
    }

    #[test]
    fn new_buffer_reads_back_all_zeros_even_after_allocation_recycling() {
        // WebGPU requires buffers to behave as zero-initialized on creation.
        // GL recycles freed-buffer memory within the process, so without the
        // zero-init in `allocate_buffer` a freshly created buffer can expose
        // stale bytes from a previously destroyed one. To make the pre-fix
        // state reliably non-zero, first fill a buffer with a 0xAB sentinel and
        // destroy it so its GL allocation is available for recycling, then
        // create a NEW buffer of the same size with NO writes and assert every
        // byte reads back as zero. This FAILS before the fix (stale 0xAB /
        // garbage) and PASSES after.
        let label = "GLES buffer zero-init test";
        let Some(device) = gles_device_or_skip(label) else {
            return;
        };

        const SIZE: u64 = 256;
        let usage = crate::HalBufferUsage::default();

        // Prime the GL allocator with a sentinel-filled buffer, then drop it so
        // the freed storage becomes a candidate for recycling.
        {
            let primer = device
                .create_buffer(SIZE, usage)
                .expect("primer buffer creation must succeed");
            primer
                .write(0, &vec![0xABu8; SIZE as usize])
                .expect("primer sentinel write must succeed");
        }

        // Create the buffer under test with no writes and read it back.
        let fresh = device
            .create_buffer(SIZE, usage)
            .expect("fresh buffer creation must succeed");
        let contents = fresh.read(0, SIZE).expect("fresh buffer read must succeed");

        assert_eq!(contents.len(), SIZE as usize);
        assert!(
            contents.iter().all(|&byte| byte == 0),
            "a freshly created buffer must read back all zeros (WebGPU zero-init); \
             found non-zero bytes indicating recycled, uninitialized GL storage"
        );
    }
}
