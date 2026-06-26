use std::sync::Arc;

use crate::adapter::Feature;
use crate::extent::*;
use crate::format::*;
use crate::texture::*;

/// Enumerates texture component swizzle values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ComponentSwizzle {
    /// Zero constant variant.
    Zero,
    /// One constant variant.
    One,
    /// Red channel variant.
    R,
    /// Green channel variant.
    G,
    /// Blue channel variant.
    B,
    /// Alpha channel variant.
    A,
}

/// Describes texture component swizzle values for a view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureComponentSwizzle {
    /// Red output channel source.
    pub r: ComponentSwizzle,
    /// Green output channel source.
    pub g: ComponentSwizzle,
    /// Blue output channel source.
    pub b: ComponentSwizzle,
    /// Alpha output channel source.
    pub a: ComponentSwizzle,
}

impl Default for TextureComponentSwizzle {
    fn default() -> Self {
        Self {
            r: ComponentSwizzle::R,
            g: ComponentSwizzle::G,
            b: ComponentSwizzle::B,
            a: ComponentSwizzle::A,
        }
    }
}

impl TextureComponentSwizzle {
    /// Returns true when this swizzle leaves every channel unchanged.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        *self == Self::default()
    }
}

/// Enumerates texture view dimension values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureViewDimension {
    /// D1 variant.
    D1,
    /// D2 variant.
    D2,
    /// D2 array variant.
    D2Array,
    /// Cube variant.
    Cube,
    /// Cube array variant.
    CubeArray,
    /// D3 variant.
    D3,
}

/// Enumerates texture aspect values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureAspect {
    /// All variant.
    All,
    /// Depth only variant.
    DepthOnly,
    /// Stencil only variant.
    StencilOnly,
}

/// Describes texture view descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureViewDescriptor {
    /// Format.
    pub format: Option<TextureFormat>,
    /// Dimension.
    pub dimension: Option<TextureViewDimension>,
    /// Base mip level.
    pub base_mip_level: u32,
    /// Mip level count.
    pub mip_level_count: Option<u32>,
    /// Base array layer.
    pub base_array_layer: u32,
    /// Array layer count.
    pub array_layer_count: Option<u32>,
    /// Aspect.
    pub aspect: Option<TextureAspect>,
    /// Usage override for this view. `None` inherits the texture usage.
    pub usage: Option<TextureUsage>,
    /// Component swizzle override for this view.
    pub swizzle: Option<TextureComponentSwizzle>,
}

/// A `TextureViewDescriptor` with every defaulted/inferred field already
/// filled in by `Texture::resolve_view_descriptor`. Validation and view
/// construction take this so an unresolved descriptor can't be validated
/// or stored by mistake.
/// Describes resolved texture view descriptor.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedTextureViewDescriptor {
    pub(crate) format: TextureFormat,
    pub(crate) dimension: TextureViewDimension,
    pub(crate) base_mip_level: u32,
    pub(crate) mip_level_count: u32,
    pub(crate) base_array_layer: u32,
    pub(crate) array_layer_count: u32,
    pub(crate) aspect: TextureAspect,
    pub(crate) usage: TextureUsage,
    pub(crate) swizzle: TextureComponentSwizzle,
}

/// Stores texture view data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct TextureView {
    pub(crate) inner: Arc<TextureViewInner>,
}

/// Holds shared state for the texture view handle.
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
    pub(crate) usage: TextureUsage,
    pub(crate) swizzle: TextureComponentSwizzle,
    pub(crate) is_error: bool,
}

impl TextureView {
    /// Creates a new instance.
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
                usage: descriptor.usage,
                swizzle: descriptor.swizzle,
                is_error,
            }),
        }
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns the texture.
    #[must_use]
    pub(crate) fn texture(&self) -> Texture {
        self.inner.texture.clone()
    }

    /// Returns the format.
    #[must_use]
    pub fn format(&self) -> TextureFormat {
        self.inner.format
    }

    /// Returns the view's dimensionality (1D / 2D / 2D-array / cube / 3D).
    #[must_use]
    pub fn dimension(&self) -> TextureViewDimension {
        self.inner.dimension
    }

    /// Returns the first mip level the view exposes.
    #[must_use]
    pub(crate) fn base_mip_level(&self) -> u32 {
        self.inner.base_mip_level
    }

    /// Returns the number of mip levels the view exposes.
    #[must_use]
    pub fn mip_level_count(&self) -> u32 {
        self.inner.mip_level_count
    }

    /// Returns the first array layer the view exposes.
    #[must_use]
    pub fn base_array_layer(&self) -> u32 {
        self.inner.base_array_layer
    }

    /// Returns the number of array layers the view exposes.
    #[must_use]
    pub(crate) fn array_layer_count(&self) -> u32 {
        self.inner.array_layer_count
    }

    /// Returns which aspect (color / depth / stencil) the view targets.
    #[must_use]
    pub fn aspect(&self) -> TextureAspect {
        self.inner.aspect
    }

    /// Returns the usage visible through this texture view.
    #[must_use]
    pub fn usage(&self) -> TextureUsage {
        self.inner.usage
    }

    /// Returns the component swizzle visible through this texture view.
    #[must_use]
    pub fn swizzle(&self) -> TextureComponentSwizzle {
        self.inner.swizzle
    }

    /// Returns render extent.
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

/// Validates texture view descriptor and returns a descriptive error on failure.
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
        usage,
        swizzle,
        ..
    } = *descriptor;

    if !swizzle.is_identity()
        && !texture
            .inner
            .features
            .contains(&Feature::TextureComponentSwizzle)
    {
        return Some("texture component swizzle requires the texture-component-swizzle feature");
    }

    if usage.bits() == 0 {
        return Some("texture view usage must be non-zero");
    }
    if usage.bits() & !TextureUsage::ALL.bits() != 0 {
        return Some("texture view usage contains unknown bits");
    }
    if usage.bits() & !texture.usage().bits() != 0 {
        return Some("texture view usage must be a subset of the texture usage");
    }

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
    if texture.dimension() == TextureDimension::D3 {
        if layer_end > 1 {
            return Some("3D texture view array layer range exceeds the single layer");
        }
    } else if layer_end > texture_layers {
        return Some("texture view array layer range exceeds texture layers");
    }
    if matches!(
        dimension,
        TextureViewDimension::Cube | TextureViewDimension::CubeArray
    ) && texture.size().width != texture.size().height
    {
        return Some("cube texture views require square faces");
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
            TextureViewDimension::Cube => {}
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
        },
        _ => {}
    }

    // A view whose `aspect` selects a single aspect of a combined
    // depth-stencil texture may use the aspect-specific format (e.g. a
    // `DepthOnly` view of `depth24plus-stencil8` may declare `depth24plus`),
    // in addition to the texture's own format and its `viewFormats`. This
    // matches the WebGPU spec (and Dawn) and is relied on by the CTS
    // `textureLoad` tests on combined depth-stencil textures.
    let aspect_format_ok = texture
        .format()
        .aspect_view_format(aspect)
        .is_some_and(|aspect_format| aspect_format == format);
    if !aspect_format_ok && !texture.is_view_format_compatible(format) {
        return Some("texture view format is not compatible with the texture");
    }

    let Some(format_caps) = texture.view_format_caps(format) else {
        return Some("texture view format must not be Undefined");
    };
    if usage.contains(TextureUsage::RENDER_ATTACHMENT) && !format_caps.renderable {
        return Some("RenderAttachment texture view format must be renderable");
    }
    if usage.contains(TextureUsage::STORAGE_BINDING) && !format_caps.storage_capable {
        return Some("StorageBinding texture view format must support storage usage");
    }
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
    fn texture_component_swizzle_identity_detection() {
        assert!(TextureComponentSwizzle::default().is_identity());
        assert!(TextureComponentSwizzle {
            r: ComponentSwizzle::R,
            g: ComponentSwizzle::G,
            b: ComponentSwizzle::B,
            a: ComponentSwizzle::A,
        }
        .is_identity());
        assert!(!TextureComponentSwizzle {
            r: ComponentSwizzle::G,
            ..TextureComponentSwizzle::default()
        }
        .is_identity());
        assert!(!TextureComponentSwizzle {
            a: ComponentSwizzle::Zero,
            ..TextureComponentSwizzle::default()
        }
        .is_identity());
    }

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
            usage: Some(TextureUsage::COPY_SRC),
            swizzle: Some(TextureComponentSwizzle::default()),
        });

        assert_eq!(error, None);
        assert!(!view.is_error());
        assert_eq!(view.format(), rgba8_unorm());
        assert_eq!(view.dimension(), TextureViewDimension::D2Array);
        assert_eq!(view.mip_level_count(), 1);
        assert_eq!(view.base_array_layer(), 1);
        assert_eq!(view.aspect(), TextureAspect::All);
        assert_eq!(view.usage(), TextureUsage::COPY_SRC);
        assert!(view.swizzle().is_identity());
    }

    #[test]
    fn texture_component_swizzle_requires_feature_when_non_identity() {
        let texture = noop_device().create_texture(layered_mipped_texture_descriptor());

        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);

        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: None,
            swizzle: Some(TextureComponentSwizzle::default()),
        });
        assert_eq!(error, None);

        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: None,
            swizzle: Some(TextureComponentSwizzle {
                r: ComponentSwizzle::G,
                ..TextureComponentSwizzle::default()
            }),
        });
        assert_eq!(
            error,
            Some("texture component swizzle requires the texture-component-swizzle feature")
        );
    }

    #[test]
    fn texture_view_usage_must_be_known_and_subset_of_texture_usage() {
        let texture = noop_device().create_texture(TextureDescriptor {
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
            ..layered_mipped_texture_descriptor()
        });

        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: Some(TextureUsage::STORAGE_BINDING),
            swizzle: None,
        });
        assert_eq!(
            error,
            Some("texture view usage must be a subset of the texture usage")
        );

        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
            usage: Some(TextureUsage::from_bits_retain(
                TextureUsage::TEXTURE_BINDING.bits() | (1_u64 << 40),
            )),
            swizzle: None,
        });
        assert_eq!(error, Some("texture view usage contains unknown bits"));
    }

    #[test]
    fn texture_view_3d_array_layer_range_is_single_layer() {
        let texture = noop_device().create_texture(TextureDescriptor {
            dimension: TextureDimension::D3,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 4,
            },
            ..layered_mipped_texture_descriptor()
        });

        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D3),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(1),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);

        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D3),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(2),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(
            error,
            Some("3D texture view array layer range exceeds the single layer")
        );

        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::D3),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 1,
            array_layer_count: Some(1),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(
            error,
            Some("3D texture view array layer range exceeds the single layer")
        );
    }

    #[test]
    fn cube_views_require_six_layers_and_square_faces() {
        let square = noop_device().create_texture(TextureDescriptor {
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 12,
            },
            ..layered_mipped_texture_descriptor()
        });
        let (_, error) = square.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::Cube),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(6),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, None);

        let non_square = noop_device().create_texture(TextureDescriptor {
            size: Extent3d {
                width: 8,
                height: 4,
                depth_or_array_layers: 12,
            },
            ..layered_mipped_texture_descriptor()
        });
        let (_, error) = non_square.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::Cube),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(6),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, Some("cube texture views require square faces"));

        let (_, error) = non_square.create_view(TextureViewDescriptor {
            format: None,
            dimension: Some(TextureViewDimension::CubeArray),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(12),
            aspect: None,
            usage: None,
            swizzle: None,
        });
        assert_eq!(error, Some("cube texture views require square faces"));
    }

    /// Builds a 4x4 single-layer 2D depth-stencil texture usable as a sampled
    /// + render-attachment, on a device that enables `depth32float-stencil8`.
    fn depth_stencil_texture(format_raw: u32) -> Texture {
        let device = noop_adapter()
            .create_device(None, &[Feature::Depth32FloatStencil8], "", "")
            .expect("Noop device creation");
        device.create_texture(TextureDescriptor {
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: TextureFormat::from_raw(format_raw),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        })
    }

    #[test]
    fn depth_only_view_accepts_aspect_specific_format() {
        // depth24plus-stencil8 → DepthOnly view with format=depth24plus is valid.
        let texture = depth_stencil_texture(TextureFormat::DEPTH24_PLUS_STENCIL8);
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: Some(TextureFormat::from_raw(TextureFormat::DEPTH24_PLUS)),
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: Some(TextureAspect::DepthOnly),
            usage: Some(TextureUsage::TEXTURE_BINDING),
            swizzle: None,
        });
        assert_eq!(error, None);
        assert!(!view.is_error());

        // depth32float-stencil8 → DepthOnly view with format=depth32float is valid.
        let texture = depth_stencil_texture(TextureFormat::DEPTH32_FLOAT_STENCIL8);
        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: Some(TextureFormat::from_raw(TextureFormat::DEPTH32_FLOAT)),
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: Some(TextureAspect::DepthOnly),
            usage: Some(TextureUsage::TEXTURE_BINDING),
            swizzle: None,
        });
        assert_eq!(error, None);
    }

    #[test]
    fn stencil_only_view_accepts_stencil8_aspect_format() {
        for combined in [
            TextureFormat::DEPTH24_PLUS_STENCIL8,
            TextureFormat::DEPTH32_FLOAT_STENCIL8,
        ] {
            let texture = depth_stencil_texture(combined);
            let (_, error) = texture.create_view(TextureViewDescriptor {
                format: Some(TextureFormat::from_raw(TextureFormat::STENCIL8)),
                dimension: None,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
                aspect: Some(TextureAspect::StencilOnly),
                usage: Some(TextureUsage::TEXTURE_BINDING),
                swizzle: None,
            });
            assert_eq!(error, None);
        }
    }

    #[test]
    fn aspect_view_still_rejects_incompatible_format() {
        let texture = depth_stencil_texture(TextureFormat::DEPTH24_PLUS_STENCIL8);

        // An unrelated color format is still rejected.
        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: Some(rgba8_unorm()),
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: Some(TextureAspect::DepthOnly),
            usage: Some(TextureUsage::TEXTURE_BINDING),
            swizzle: None,
        });
        assert_eq!(
            error,
            Some("texture view format is not compatible with the texture")
        );

        // A depth format that is not this texture's depth aspect is still rejected.
        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: Some(TextureFormat::from_raw(TextureFormat::DEPTH16_UNORM)),
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: Some(TextureAspect::DepthOnly),
            usage: Some(TextureUsage::TEXTURE_BINDING),
            swizzle: None,
        });
        assert_eq!(
            error,
            Some("texture view format is not compatible with the texture")
        );
    }

    #[test]
    fn aspect_all_and_undefined_format_still_accept_texture_format() {
        let texture = depth_stencil_texture(TextureFormat::DEPTH24_PLUS_STENCIL8);

        // aspect=All with the texture's own combined format is accepted.
        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: Some(TextureFormat::from_raw(
                TextureFormat::DEPTH24_PLUS_STENCIL8,
            )),
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: Some(TextureAspect::All),
            usage: Some(TextureUsage::TEXTURE_BINDING),
            swizzle: None,
        });
        assert_eq!(error, None);

        // format=Undefined on a DepthOnly view still resolves to the texture's
        // own format and is accepted (unchanged behavior).
        let (_, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: Some(TextureAspect::DepthOnly),
            usage: Some(TextureUsage::TEXTURE_BINDING),
            swizzle: None,
        });
        assert_eq!(error, None);
    }
}
