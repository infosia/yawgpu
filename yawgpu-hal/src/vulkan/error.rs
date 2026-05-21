use super::*;

/// Returns buffer error.
pub(super) fn buffer_error(message: &'static str) -> HalError {
    HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    }
}

/// Converts buffer error into the corresponding yawgpu representation.
pub(super) fn map_buffer_error(_error: vk::Result, message: &'static str) -> HalError {
    buffer_error(message)
}

/// Returns texture error.
pub(super) fn texture_error(message: &'static str) -> HalError {
    HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    }
}

/// Converts texture error into the corresponding yawgpu representation.
pub(super) fn map_texture_error(_error: vk::Result, message: &'static str) -> HalError {
    texture_error(message)
}

/// Returns shader error.
pub(super) fn shader_error(message: &'static str) -> HalError {
    HalError::ShaderCompilationFailed {
        backend: BACKEND,
        message: message.to_owned(),
    }
}
