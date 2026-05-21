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
    #[error("HAL queue submission failed: {backend}")]
    /// Queue submission failed variant.
    QueueSubmissionFailed {
        /// Backend name.
        backend: &'static str,
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
}
