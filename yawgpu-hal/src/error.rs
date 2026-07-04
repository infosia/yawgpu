/// Enumerates HAL error values.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HalError {
    #[error("HAL backend is unavailable: {backend}")]
    /// Backend unavailable variant.
    BackendUnavailable {
        /// Backend name.
        backend: &'static str,
    },
    #[error("HAL device creation failed: {backend}")]
    /// Device creation failed variant.
    DeviceCreationFailed {
        /// Backend name.
        backend: &'static str,
    },
    #[error("HAL out of memory: {backend}: {resource}")]
    /// Out-of-memory variant.
    OutOfMemory {
        /// Backend name.
        backend: &'static str,
        /// Resource name.
        resource: &'static str,
    },
    #[error("HAL queue submission failed: {backend}: {message}")]
    /// Queue submission failed variant.
    QueueSubmissionFailed {
        /// Backend name.
        backend: &'static str,
        /// Message variant.
        message: String,
    },
    #[error("HAL buffer operation failed: {backend}: {message}")]
    /// Buffer operation failed variant.
    BufferOperationFailed {
        /// Backend variant.
        backend: &'static str,
        /// Message variant.
        message: &'static str,
    },
    #[error("HAL shader compilation failed: {backend}: {message}")]
    /// Shader compilation failed variant.
    ShaderCompilationFailed {
        /// Backend variant.
        backend: &'static str,
        /// Message variant.
        message: String,
    },
    #[error("HAL swapchain creation failed: {backend}: {message}")]
    /// Swapchain creation failed variant.
    SwapchainCreationFailed {
        /// Backend variant.
        backend: &'static str,
        /// Message variant.
        message: &'static str,
    },
    #[error("HAL surface acquire failed: {backend}: {message}")]
    /// Acquire failed variant.
    AcquireFailed {
        /// Backend variant.
        backend: &'static str,
        /// Message variant.
        message: &'static str,
    },
    #[error("HAL surface present failed: {backend}: {message}")]
    /// Present failed variant.
    PresentFailed {
        /// Backend variant.
        backend: &'static str,
        /// Message variant.
        message: &'static str,
    },
    #[error("HAL texture creation failed: {backend}: {message}")]
    /// Texture creation failed variant. Carries a dynamic message so the
    /// error can name the concrete descriptor values (e.g. the format and
    /// sample count rejected by a device-capability check).
    TextureCreationFailed {
        /// Backend variant.
        backend: &'static str,
        /// Message describing the rejected descriptor.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_submission_failed_display_includes_backend_and_message() {
        let error = HalError::QueueSubmissionFailed {
            backend: "vulkan",
            message: "vkQueueSubmit failed: ERROR_OUT_OF_DEVICE_MEMORY".to_string(),
        };

        assert_eq!(
            error.to_string(),
            "HAL queue submission failed: vulkan: vkQueueSubmit failed: ERROR_OUT_OF_DEVICE_MEMORY"
        );
    }

    #[test]
    fn texture_creation_failed_display_includes_backend_and_message() {
        let error = HalError::TextureCreationFailed {
            backend: "vulkan",
            message: "sample count 4 not supported for R8Sint on this device".to_string(),
        };

        assert_eq!(
            error.to_string(),
            "HAL texture creation failed: vulkan: \
             sample count 4 not supported for R8Sint on this device"
        );
    }
}
