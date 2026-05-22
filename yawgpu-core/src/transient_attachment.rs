use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{HalTransientAttachment, HalTransientAttachmentDescriptor};

use crate::format::*;
use crate::texture::hal_texture_format;

/// Enumerates transient attachment size modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransientSizeMode {
    /// Resolve the attachment size from the subpass render target at pass begin.
    MatchTarget,
    /// Use an explicit attachment size.
    Explicit {
        /// Width.
        width: u32,
        /// Height.
        height: u32,
    },
}

/// Describes a transient attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransientAttachmentDescriptor {
    /// Format.
    pub format: TextureFormat,
    /// Size mode.
    pub size: TransientSizeMode,
    /// Sample count.
    pub sample_count: u32,
}

/// Stores transient attachment data used by tiled render passes.
#[derive(Debug, Clone)]
pub struct TransientAttachment {
    pub(crate) inner: Arc<TransientAttachmentInner>,
}

/// Holds shared state for the transient attachment handle.
#[derive(Debug)]
pub(crate) struct TransientAttachmentInner {
    pub(crate) descriptor: TransientAttachmentDescriptor,
    pub(crate) hal: Mutex<Option<HalTransientAttachment>>,
    pub(crate) is_error: bool,
}

impl TransientAttachment {
    /// Creates a new transient attachment.
    #[must_use]
    pub(crate) fn new(
        descriptor: TransientAttachmentDescriptor,
        hal: Option<HalTransientAttachment>,
        is_error: bool,
    ) -> Self {
        Self {
            inner: Arc::new(TransientAttachmentInner {
                descriptor,
                hal: Mutex::new(hal),
                is_error,
            }),
        }
    }

    /// Returns the descriptor.
    #[must_use]
    pub fn descriptor(&self) -> TransientAttachmentDescriptor {
        self.inner.descriptor
    }

    /// Returns the HAL attachment, if it has already been allocated.
    #[must_use]
    pub fn hal(&self) -> Option<HalTransientAttachment> {
        self.inner.hal.lock().clone()
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Stores the HAL allocation resolved at pass begin for match-target attachments.
    pub(crate) fn set_hal_for_match_target(&self, hal: HalTransientAttachment) {
        let mut current = self.inner.hal.lock();
        if current.is_none() {
            *current = Some(hal);
        }
    }
}

/// Validates a transient attachment descriptor.
pub(crate) fn validate_transient_attachment_descriptor(
    descriptor: &TransientAttachmentDescriptor,
) -> Option<String> {
    if let TransientSizeMode::Explicit { width, height } = descriptor.size {
        if width == 0 || height == 0 {
            return Some("explicit transient attachment size must be non-zero".to_owned());
        }
    }
    None
}

/// Returns HAL transient attachment descriptor for a concrete attachment size.
pub(crate) fn hal_transient_attachment_descriptor(
    descriptor: &TransientAttachmentDescriptor,
    width: u32,
    height: u32,
) -> HalTransientAttachmentDescriptor {
    HalTransientAttachmentDescriptor {
        format: hal_texture_format(descriptor.format),
        width,
        height,
        sample_count: descriptor.sample_count,
    }
}
