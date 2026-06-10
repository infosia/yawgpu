use super::*;

/// Returns buffer error.
pub(super) fn buffer_error(message: &'static str) -> HalError {
    HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    }
}

/// Converts buffer error into the corresponding yawgpu representation.
pub(super) fn map_buffer_error(error: vk::Result, message: &'static str) -> HalError {
    if matches!(
        error,
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY
    ) {
        return HalError::OutOfMemory {
            backend: BACKEND,
            resource: "buffer",
        };
    }
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
pub(super) fn map_texture_error(error: vk::Result, message: &'static str) -> HalError {
    if matches!(
        error,
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | vk::Result::ERROR_OUT_OF_HOST_MEMORY
    ) {
        return HalError::OutOfMemory {
            backend: BACKEND,
            resource: "texture",
        };
    }
    texture_error(message)
}

/// Returns shader error.
pub(super) fn shader_error(message: &'static str) -> HalError {
    HalError::ShaderCompilationFailed {
        backend: BACKEND,
        message: message.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "vulkan")]
    fn vulkan_out_of_memory_results_map_to_hal_out_of_memory() {
        assert!(matches!(
            map_buffer_error(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY, "buffer failed"),
            HalError::OutOfMemory {
                backend: BACKEND,
                resource: "buffer"
            }
        ));
        assert!(matches!(
            map_texture_error(vk::Result::ERROR_OUT_OF_HOST_MEMORY, "texture failed"),
            HalError::OutOfMemory {
                backend: BACKEND,
                resource: "texture"
            }
        ));
        assert!(matches!(
            map_texture_error(vk::Result::ERROR_INITIALIZATION_FAILED, "texture failed"),
            HalError::BufferOperationFailed { .. }
        ));
    }
}
