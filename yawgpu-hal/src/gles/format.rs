use super::BACKEND;
use crate::{HalError, HalPrimitiveTopology, HalTextureFormat, HalVertexFormat};

#[derive(Clone, Copy, Debug)]
pub(super) struct GlesFormat {
    pub(super) internal: u32,
    pub(super) format: u32,
    pub(super) ty: u32,
    pub(super) bytes_per_pixel: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GlesVertexFormat {
    pub(super) components: i32,
    pub(super) ty: u32,
    pub(super) normalized: bool,
}

pub(super) fn map_texture_format(format: HalTextureFormat) -> Result<GlesFormat, HalError> {
    match format {
        HalTextureFormat::Rgba8Unorm | HalTextureFormat::Bgra8Unorm => Ok(rgba8_unorm()),
        HalTextureFormat::Rgba8Uint => Ok(rgba8_uint()),
        _ => Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture format not supported on GLES (P15.3)",
        }),
    }
}

pub(super) fn map_vertex_format(format: HalVertexFormat) -> Result<GlesVertexFormat, HalError> {
    match format {
        HalVertexFormat::Float32 => Ok(GlesVertexFormat {
            components: 1,
            ty: glow::FLOAT,
            normalized: false,
        }),
        HalVertexFormat::Float32x2 => Ok(GlesVertexFormat {
            components: 2,
            ty: glow::FLOAT,
            normalized: false,
        }),
        HalVertexFormat::Float32x3 => Ok(GlesVertexFormat {
            components: 3,
            ty: glow::FLOAT,
            normalized: false,
        }),
        HalVertexFormat::Float32x4 => Ok(GlesVertexFormat {
            components: 4,
            ty: glow::FLOAT,
            normalized: false,
        }),
        HalVertexFormat::Unsupported => Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "Unsupported vertex format requested",
        }),
    }
}

pub(super) fn map_primitive_topology(topology: HalPrimitiveTopology) -> u32 {
    match topology {
        HalPrimitiveTopology::PointList => glow::POINTS,
        HalPrimitiveTopology::LineList => glow::LINES,
        HalPrimitiveTopology::LineStrip => glow::LINE_STRIP,
        HalPrimitiveTopology::TriangleList => glow::TRIANGLES,
        HalPrimitiveTopology::TriangleStrip => glow::TRIANGLE_STRIP,
    }
}

pub(super) fn fallback_format() -> GlesFormat {
    rgba8_unorm()
}

fn rgba8_unorm() -> GlesFormat {
    GlesFormat {
        internal: glow::RGBA8,
        format: glow::RGBA,
        ty: glow::UNSIGNED_BYTE,
        bytes_per_pixel: 4,
    }
}

fn rgba8_uint() -> GlesFormat {
    GlesFormat {
        internal: glow::RGBA8UI,
        format: glow::RGBA_INTEGER,
        ty: glow::UNSIGNED_BYTE,
        bytes_per_pixel: 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_texture_format_rgba8_and_bgra8_return_rgba8_triplet() {
        for texture_format in [HalTextureFormat::Rgba8Unorm, HalTextureFormat::Bgra8Unorm] {
            let format = map_texture_format(texture_format).expect("RGBA/BGRA8 is supported");

            assert_eq!(format.internal, glow::RGBA8);
            assert_eq!(format.format, glow::RGBA);
            assert_eq!(format.ty, glow::UNSIGNED_BYTE);
            assert_eq!(format.bytes_per_pixel, 4);
        }
    }

    #[test]
    fn map_texture_format_unsupported_returns_error() {
        let error = map_texture_format(HalTextureFormat::R8Unorm)
            .expect_err("unsupported format must error");

        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "texture format not supported on GLES (P15.3)",
            }
        ));
    }

    #[test]
    fn map_texture_format_rgba8_uint_returns_integer_triplet() {
        let format = map_texture_format(HalTextureFormat::Rgba8Uint).expect("RGBA8Uint supported");

        assert_eq!(format.internal, glow::RGBA8UI);
        assert_eq!(format.format, glow::RGBA_INTEGER);
        assert_eq!(format.ty, glow::UNSIGNED_BYTE);
        assert_eq!(format.bytes_per_pixel, 4);
    }

    #[test]
    fn map_vertex_format_table() {
        let float = map_vertex_format(HalVertexFormat::Float32).expect("Float32 supported");
        assert_eq!(float.components, 1);
        assert_eq!(float.ty, glow::FLOAT);
        assert!(!float.normalized);

        assert_eq!(
            map_vertex_format(HalVertexFormat::Float32x2)
                .expect("Float32x2 supported")
                .components,
            2
        );
        assert_eq!(
            map_vertex_format(HalVertexFormat::Float32x3)
                .expect("Float32x3 supported")
                .components,
            3
        );
        assert_eq!(
            map_vertex_format(HalVertexFormat::Float32x4)
                .expect("Float32x4 supported")
                .components,
            4
        );

        let error = map_vertex_format(HalVertexFormat::Unsupported)
            .expect_err("unsupported vertex format must error");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "Unsupported vertex format requested",
            }
        ));
    }

    #[test]
    fn map_primitive_topology_table() {
        assert_eq!(
            map_primitive_topology(HalPrimitiveTopology::PointList),
            glow::POINTS
        );
        assert_eq!(
            map_primitive_topology(HalPrimitiveTopology::LineList),
            glow::LINES
        );
        assert_eq!(
            map_primitive_topology(HalPrimitiveTopology::LineStrip),
            glow::LINE_STRIP
        );
        assert_eq!(
            map_primitive_topology(HalPrimitiveTopology::TriangleList),
            glow::TRIANGLES
        );
        assert_eq!(
            map_primitive_topology(HalPrimitiveTopology::TriangleStrip),
            glow::TRIANGLE_STRIP
        );
    }
}
