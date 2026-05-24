use super::BACKEND;
use crate::{HalError, HalTextureFormat};

#[derive(Clone, Copy, Debug)]
pub(super) struct GlesFormat {
    pub(super) internal: u32,
    pub(super) format: u32,
    pub(super) ty: u32,
    pub(super) bytes_per_pixel: u32,
}

pub(super) fn map_texture_format(format: HalTextureFormat) -> Result<GlesFormat, HalError> {
    match format {
        HalTextureFormat::Rgba8Unorm => Ok(rgba8_unorm()),
        _ => Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture format not supported on GLES (P15.3)",
        }),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_texture_format_rgba8unorm_returns_rgba8_triplet() {
        let format =
            map_texture_format(HalTextureFormat::Rgba8Unorm).expect("RGBA8Unorm is supported");

        assert_eq!(format.internal, glow::RGBA8);
        assert_eq!(format.format, glow::RGBA);
        assert_eq!(format.ty, glow::UNSIGNED_BYTE);
        assert_eq!(format.bytes_per_pixel, 4);
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
}
