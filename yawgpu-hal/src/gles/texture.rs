use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::format::{fallback_format, map_texture_format, GlesFormat};
use super::{rebuild_hal_error, BACKEND};
use crate::{HalError, HalTextureDescriptor, HalTextureDimension};

pub(super) struct GlesTextureMeta {
    pub(super) format: GlesFormat,
    pub(super) dimension: HalTextureDimension,
    pub(super) target: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) depth_or_array_layers: u32,
    pub(super) mip_level_count: u32,
}

pub(super) struct GlesTextureInner {
    device: Arc<GlesDeviceInner>,
    texture: Result<glow::Texture, HalError>,
    meta: GlesTextureMeta,
}

impl Drop for GlesTextureInner {
    fn drop(&mut self) {
        if let Ok(texture) = self.texture.as_ref() {
            let texture = *texture;
            let _ = self.device.with_current_context(|gl| unsafe {
                gl.delete_texture(texture);
            });
        }
    }
}

/// Stores GLES texture data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesTexture {
    inner: Arc<GlesTextureInner>,
}

// SAFETY: `GlesTexture` accesses GL state only through `GlesDeviceInner`, whose
// make-current lock serializes all GL commands.
unsafe impl Send for GlesTexture {}
// SAFETY: See the `Send` impl; shared operations are synchronized by the
// owning device inner.
unsafe impl Sync for GlesTexture {}

impl std::fmt::Debug for GlesTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesTexture")
            .field("dimension", &self.inner.meta.dimension)
            .field("width", &self.inner.meta.width)
            .field("height", &self.inner.meta.height)
            .field(
                "depth_or_array_layers",
                &self.inner.meta.depth_or_array_layers,
            )
            .field("mip_level_count", &self.inner.meta.mip_level_count)
            .finish()
    }
}

impl GlesTexture {
    pub(super) fn new(device: Arc<GlesDeviceInner>, descriptor: &HalTextureDescriptor) -> Self {
        let texture = allocate_texture(&device, descriptor);
        let meta = derive_meta(descriptor);
        Self {
            inner: Arc::new(GlesTextureInner {
                device,
                texture,
                meta,
            }),
        }
    }

    pub(super) fn raw_or_err(&self) -> Result<glow::Texture, HalError> {
        self.inner
            .texture
            .as_ref()
            .copied()
            .map_err(rebuild_hal_error)
    }

    pub(super) fn meta(&self) -> &GlesTextureMeta {
        &self.inner.meta
    }
}

fn derive_meta(descriptor: &HalTextureDescriptor) -> GlesTextureMeta {
    let format = match map_texture_format(descriptor.format) {
        Ok(format) => format,
        Err(_) => fallback_format(),
    };
    GlesTextureMeta {
        format,
        dimension: descriptor.dimension,
        target: texture_target(descriptor),
        width: descriptor.width,
        height: descriptor.height,
        depth_or_array_layers: descriptor.depth_or_array_layers,
        mip_level_count: descriptor.mip_level_count,
    }
}

fn allocate_texture(
    device: &Arc<GlesDeviceInner>,
    descriptor: &HalTextureDescriptor,
) -> Result<glow::Texture, HalError> {
    if descriptor.sample_count != 1 {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "multisampled textures not supported on GLES (P15.3)",
        });
    }
    if descriptor.mip_level_count == 0 {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture mip level count must be non-zero",
        });
    }
    let format = map_texture_format(descriptor.format)?;
    let width = i32::try_from(descriptor.width).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message: "texture width exceeds GLES limit",
    })?;
    let height = i32::try_from(descriptor.height).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message: "texture height exceeds GLES limit",
    })?;
    let depth = i32::try_from(descriptor.depth_or_array_layers).map_err(|_| {
        HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture depth or layer count exceeds GLES limit",
        }
    })?;
    let mip_level_count =
        i32::try_from(descriptor.mip_level_count).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture mip level count exceeds GLES limit",
        })?;
    let target = texture_target(descriptor);

    device
        .with_current_context(|gl| unsafe {
            let texture = gl
                .create_texture()
                .map_err(|_| HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "glCreateTexture failed",
                })?;
            gl.bind_texture(target, Some(texture));
            match descriptor.dimension {
                HalTextureDimension::D1 => {
                    gl.tex_storage_2d(target, mip_level_count, format.internal, width, 1);
                }
                HalTextureDimension::D2 if descriptor.depth_or_array_layers == 1 => {
                    gl.tex_storage_2d(target, mip_level_count, format.internal, width, height);
                }
                HalTextureDimension::D2 | HalTextureDimension::D3 => {
                    gl.tex_storage_3d(
                        target,
                        mip_level_count,
                        format.internal,
                        width,
                        height,
                        depth,
                    );
                }
            }
            gl.bind_texture(target, None);
            Ok(texture)
        })
        .and_then(|result| result)
}

fn texture_target(descriptor: &HalTextureDescriptor) -> u32 {
    match descriptor.dimension {
        HalTextureDimension::D1 => glow::TEXTURE_2D,
        HalTextureDimension::D2 if descriptor.depth_or_array_layers == 1 => glow::TEXTURE_2D,
        HalTextureDimension::D2 => glow::TEXTURE_2D_ARRAY,
        HalTextureDimension::D3 => glow::TEXTURE_3D,
    }
}
