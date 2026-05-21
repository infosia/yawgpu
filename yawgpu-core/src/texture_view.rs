use std::sync::Arc;

use crate::extent::*;
use crate::format::*;
use crate::texture::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureAspect {
    All,
    DepthOnly,
    StencilOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureViewDescriptor {
    pub format: Option<TextureFormat>,
    pub dimension: Option<TextureViewDimension>,
    pub base_mip_level: u32,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<u32>,
    pub aspect: Option<TextureAspect>,
}

/// A `TextureViewDescriptor` with every defaulted/inferred field already
/// filled in by `Texture::resolve_view_descriptor`. Validation and view
/// construction take this so an unresolved descriptor can't be validated
/// or stored by mistake.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedTextureViewDescriptor {
    pub(crate) format: TextureFormat,
    pub(crate) dimension: TextureViewDimension,
    pub(crate) base_mip_level: u32,
    pub(crate) mip_level_count: u32,
    pub(crate) base_array_layer: u32,
    pub(crate) array_layer_count: u32,
    pub(crate) aspect: TextureAspect,
}

#[derive(Debug, Clone)]
pub struct TextureView {
    pub(crate) inner: Arc<TextureViewInner>,
}

#[derive(Debug)]
pub(crate) struct TextureViewInner {
    pub(crate) texture: Texture,
    pub(crate) format: TextureFormat,
    pub(crate) dimension: TextureViewDimension,
    pub(crate) base_mip_level: u32,
    pub(crate) mip_level_count: u32,
    pub(crate) base_array_layer: u32,
    pub(crate) array_layer_count: u32,
    pub(crate) aspect: TextureAspect,
    pub(crate) is_error: bool,
}

impl TextureView {
    pub(crate) fn new(
        texture: Texture,
        descriptor: ResolvedTextureViewDescriptor,
        is_error: bool,
    ) -> Self {
        Self {
            inner: Arc::new(TextureViewInner {
                texture,
                format: descriptor.format,
                dimension: descriptor.dimension,
                base_mip_level: descriptor.base_mip_level,
                mip_level_count: descriptor.mip_level_count,
                base_array_layer: descriptor.base_array_layer,
                array_layer_count: descriptor.array_layer_count,
                aspect: descriptor.aspect,
                is_error,
            }),
        }
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub(crate) fn texture(&self) -> Texture {
        self.inner.texture.clone()
    }

    #[must_use]
    pub fn format(&self) -> TextureFormat {
        self.inner.format
    }

    #[must_use]
    pub fn dimension(&self) -> TextureViewDimension {
        self.inner.dimension
    }

    #[must_use]
    pub(crate) fn base_mip_level(&self) -> u32 {
        self.inner.base_mip_level
    }

    #[must_use]
    pub fn mip_level_count(&self) -> u32 {
        self.inner.mip_level_count
    }

    #[must_use]
    pub fn base_array_layer(&self) -> u32 {
        self.inner.base_array_layer
    }

    #[must_use]
    pub(crate) fn array_layer_count(&self) -> u32 {
        self.inner.array_layer_count
    }

    #[must_use]
    pub fn aspect(&self) -> TextureAspect {
        self.inner.aspect
    }

    #[must_use]
    pub(crate) fn render_extent(&self) -> Extent3d {
        let subresource = self.texture().subresource_size(self.base_mip_level());
        Extent3d {
            width: subresource.width,
            height: subresource.height,
            depth_or_array_layers: 1,
        }
    }
}

pub(crate) fn validate_texture_view_descriptor(
    texture: &Texture,
    descriptor: &ResolvedTextureViewDescriptor,
) -> Option<&'static str> {
    let ResolvedTextureViewDescriptor {
        format,
        dimension,
        mip_level_count,
        array_layer_count,
        aspect,
        ..
    } = *descriptor;

    if mip_level_count == 0 {
        return Some("texture view mipLevelCount must be greater than zero");
    }
    if array_layer_count == 0 {
        return Some("texture view arrayLayerCount must be greater than zero");
    }
    let Some(mip_end) = descriptor.base_mip_level.checked_add(mip_level_count) else {
        return Some("texture view mip range overflows");
    };
    if mip_end > texture.mip_level_count() {
        return Some("texture view mip range exceeds texture mip levels");
    }

    let texture_layers = texture.size().depth_or_array_layers;
    let Some(layer_end) = descriptor.base_array_layer.checked_add(array_layer_count) else {
        return Some("texture view array layer range overflows");
    };
    if texture.dimension() != TextureDimension::D3 && layer_end > texture_layers {
        return Some("texture view array layer range exceeds texture layers");
    }

    match texture.dimension() {
        TextureDimension::D1 if dimension != TextureViewDimension::D1 => {
            return Some("1D textures require 1D views");
        }
        TextureDimension::D3 if dimension != TextureViewDimension::D3 => {
            return Some("3D textures require 3D views");
        }
        TextureDimension::D2 => match dimension {
            TextureViewDimension::D2 if array_layer_count != 1 => {
                return Some("2D texture views require exactly one array layer");
            }
            TextureViewDimension::D2Array => {}
            TextureViewDimension::Cube if array_layer_count != 6 => {
                return Some("cube texture views require exactly six array layers");
            }
            TextureViewDimension::CubeArray if !array_layer_count.is_multiple_of(6) => {
                return Some(
                    "cube-array texture views require a layer count that is a multiple of six",
                );
            }
            TextureViewDimension::CubeArray => {}
            TextureViewDimension::D1 | TextureViewDimension::D3 => {
                return Some("2D textures require 2D-compatible views");
            }
            TextureViewDimension::D2 => {}
            _ => return Some("texture view dimension is unsupported"),
        },
        _ => {}
    }

    if !texture.is_view_format_compatible(format) {
        return Some("texture view format is not compatible with the texture");
    }

    let Some(format_caps) = format.caps() else {
        return Some("texture view format must not be Undefined");
    };
    match aspect {
        TextureAspect::All => {}
        TextureAspect::DepthOnly if !format_caps.aspects.depth => {
            return Some("DepthOnly texture views require a depth format");
        }
        TextureAspect::StencilOnly if !format_caps.aspects.stencil => {
            return Some("StencilOnly texture views require a stencil format");
        }
        TextureAspect::DepthOnly | TextureAspect::StencilOnly => {}
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

    #[test]
    fn texture_view_descriptor_fields_round_trip() {
        let texture = noop_device().create_texture(layered_mipped_texture_descriptor());

        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: Some(rgba8_unorm()),
            dimension: Some(TextureViewDimension::D2Array),
            base_mip_level: 1,
            mip_level_count: Some(1),
            base_array_layer: 1,
            array_layer_count: Some(2),
            aspect: Some(TextureAspect::All),
        });

        assert_eq!(error, None);
        assert!(!view.is_error());
        assert_eq!(view.format(), rgba8_unorm());
        assert_eq!(view.dimension(), TextureViewDimension::D2Array);
        assert_eq!(view.mip_level_count(), 1);
        assert_eq!(view.base_array_layer(), 1);
        assert_eq!(view.aspect(), TextureAspect::All);
    }
}
