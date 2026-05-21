use super::*;

/// Converts vertex format into the corresponding yawgpu representation.
#[must_use]
pub fn map_vertex_format(value: native::WGPUVertexFormat) -> core::VertexFormat {
    value.into()
}

/// Converts vertex format to native into the corresponding yawgpu representation.
#[must_use]
pub fn map_vertex_format_to_native(value: core::VertexFormat) -> native::WGPUVertexFormat {
    value.into()
}

/// Converts texture format into the corresponding yawgpu representation.
#[must_use]
pub fn map_texture_format(value: native::WGPUTextureFormat) -> core::TextureFormat {
    value.into()
}

/// Converts texture format to native into the corresponding yawgpu representation.
#[must_use]
pub fn map_texture_format_to_native(value: core::TextureFormat) -> native::WGPUTextureFormat {
    value.into()
}
