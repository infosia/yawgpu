use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::unavailable;
use crate::{HalCopy, HalError};

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
    pub fn submit_copies(&self, _copies: &[HalCopy]) -> Result<(), HalError> {
        unavailable()
    }
}
