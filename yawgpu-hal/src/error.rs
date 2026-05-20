#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HalError {
    #[error("HAL backend is unavailable: {backend}")]
    BackendUnavailable { backend: &'static str },
    #[error("HAL device creation failed: {backend}")]
    DeviceCreationFailed { backend: &'static str },
    #[error("HAL queue submission failed: {backend}")]
    QueueSubmissionFailed { backend: &'static str },
    #[error("HAL buffer operation failed: {backend}: {message}")]
    BufferOperationFailed {
        backend: &'static str,
        message: &'static str,
    },
    #[error("HAL shader compilation failed: {backend}: {message}")]
    ShaderCompilationFailed {
        backend: &'static str,
        message: String,
    },
    #[error("HAL swapchain creation failed: {backend}: {message}")]
    SwapchainCreationFailed {
        backend: &'static str,
        message: &'static str,
    },
    #[error("HAL surface acquire failed: {backend}: {message}")]
    AcquireFailed {
        backend: &'static str,
        message: &'static str,
    },
    #[error("HAL surface present failed: {backend}: {message}")]
    PresentFailed {
        backend: &'static str,
        message: &'static str,
    },
}
