use std::sync::Arc;

#[cfg(feature = "tiled")]
use yawgpu_hal::FramebufferFetchPath;
use yawgpu_hal::{HalAdapter, HalBackend};

use crate::device::*;
use crate::error::*;
use crate::limits::*;

/// Stores adapter data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct Adapter {
    pub(crate) inner: Arc<AdapterInner>,
}

/// Holds shared state for the adapter handle.
#[derive(Debug)]
pub(crate) struct AdapterInner {
    pub(crate) hal: HalAdapter,
    pub(crate) feature_level: FeatureLevel,
}

impl Adapter {
    /// Constructs this object from the backend HAL object.
    #[must_use]
    pub fn from_hal(hal: HalAdapter) -> Self {
        Self::from_hal_with_feature_level(hal, FeatureLevel::Core)
    }

    /// Constructs this object from hal with feature level.
    #[must_use]
    pub(crate) fn from_hal_with_feature_level(
        hal: HalAdapter,
        feature_level: FeatureLevel,
    ) -> Self {
        Self {
            inner: Arc::new(AdapterInner { hal, feature_level }),
        }
    }

    /// Returns the name.
    #[must_use]
    pub fn name(&self) -> String {
        self.inner.hal.name()
    }

    /// Returns the backend.
    #[must_use]
    pub fn backend(&self) -> HalBackend {
        self.inner.hal.backend()
    }

    /// Returns the limits.
    #[must_use]
    pub fn limits(&self) -> Limits {
        // Block 00: the synthetic Noop adapter's supported limits are the
        // WebGPU spec defaults by design.
        Limits::DEFAULT
    }

    /// Returns the feature level.
    #[must_use]
    pub(crate) fn feature_level(&self) -> FeatureLevel {
        self.inner.feature_level
    }

    /// Returns the features.
    #[must_use]
    pub fn features(&self) -> FeatureSet {
        #[cfg(not(feature = "tiled"))]
        {
            supported_features()
        }

        #[cfg(feature = "tiled")]
        {
            let mut features = supported_features();
            add_tiled_features(
                &mut features,
                self.backend(),
                self.inner.hal.framebuffer_fetch_path(),
            );
            features
        }
    }

    /// Returns true when this object has the requested feature.
    #[must_use]
    pub fn has_feature(&self, feature: Feature) -> bool {
        self.features().contains(&feature)
    }

    /// Creates a device and its queue from this adapter, honoring the requested limits and features.
    pub fn create_device(
        &self,
        required_limits: Option<&Limits>,
        required_features: &[Feature],
        label: impl Into<String>,
        queue_label: impl Into<String>,
    ) -> Result<Device, Error> {
        let limits = self
            .limits()
            .validate_required_limits(required_limits)
            .map_err(Error::Validation)?;
        let features = self.resolve_features(required_features)?;
        let hal = self.inner.hal.create_device()?;
        Ok(Device::from_hal(hal, limits, features, label, queue_label))
    }

    /// Resolves the requested feature list against what this adapter supports.
    pub(crate) fn resolve_features(
        &self,
        required_features: &[Feature],
    ) -> Result<FeatureSet, Error> {
        let supported = self.features();
        let mut resolved = FeatureSet::new();

        if self.feature_level() == FeatureLevel::Core {
            resolved.insert(Feature::CoreFeaturesAndLimits);
        }

        for feature in required_features {
            if !supported.contains(feature) {
                return Err(Error::Validation(format!(
                    "required feature {feature:?} is not supported"
                )));
            }
            resolved.insert(*feature);
        }

        apply_feature_implications(&mut resolved);
        Ok(resolved)
    }
}

/// Enumerates feature level values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FeatureLevel {
    /// Core variant.
    Core,
    /// Compatibility variant.
    Compatibility,
}

/// Enumerates feature values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Feature {
    /// Core features and limits variant.
    CoreFeaturesAndLimits,
    /// Rg11b10 ufloat renderable variant.
    Rg11b10UfloatRenderable,
    /// Timestamp query variant.
    TimestampQuery,
    /// Texture formats tier1 variant.
    TextureFormatsTier1,
    /// Texture formats tier2 variant.
    TextureFormatsTier2,
    /// Multi-subpass render pass support.
    #[cfg(feature = "tiled")]
    MultiSubpass,
    /// Transient attachment support.
    #[cfg(feature = "tiled")]
    TransientAttachments,
    /// Shader framebuffer fetch support.
    #[cfg(feature = "tiled")]
    ShaderFramebufferFetch,
    /// Programmable tile dispatch support.
    #[cfg(feature = "tiled")]
    ProgrammableTileDispatch,
    /// Other variant.
    Other(u32),
}

/// Stores tiled rendering limits exposed by the adapter.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledCapabilities {
    /// Maximum number of subpasses in one tiled render pass.
    pub max_subpasses: u32,
    /// Maximum number of color attachments in a subpass.
    pub max_subpass_color_attachments: u32,
    /// Maximum number of input attachments in a subpass.
    pub max_input_attachments: u32,
    /// Estimated tile memory budget, in bytes.
    pub estimated_tile_memory_bytes: u32,
}

/// Returns supported features.
#[must_use]
pub(crate) fn supported_features() -> FeatureSet {
    [
        Feature::CoreFeaturesAndLimits,
        Feature::Rg11b10UfloatRenderable,
        Feature::TimestampQuery,
        Feature::TextureFormatsTier1,
        Feature::TextureFormatsTier2,
    ]
    .into_iter()
    .collect()
}

/// Returns true when tiled rendering features are supported by `backend`.
#[cfg(feature = "tiled")]
#[must_use]
pub(crate) fn tiled_features_supported(backend: HalBackend) -> bool {
    matches!(backend, HalBackend::Metal | HalBackend::Vulkan)
}

/// Returns true when shader framebuffer fetch is supported by backend/path.
#[cfg(feature = "tiled")]
#[must_use]
pub(crate) fn framebuffer_fetch_supported(backend: HalBackend, path: FramebufferFetchPath) -> bool {
    match backend {
        HalBackend::Metal => true,
        HalBackend::Vulkan => !matches!(path, FramebufferFetchPath::Disabled),
        HalBackend::Noop => false,
        _ => false,
    }
}

#[cfg(feature = "tiled")]
fn add_tiled_features(features: &mut FeatureSet, backend: HalBackend, path: FramebufferFetchPath) {
    if !tiled_features_supported(backend) {
        return;
    }
    features.insert(Feature::MultiSubpass);
    features.insert(Feature::TransientAttachments);
    if framebuffer_fetch_supported(backend, path) {
        features.insert(Feature::ShaderFramebufferFetch);
    }
    features.insert(Feature::ProgrammableTileDispatch);
}

#[cfg(feature = "tiled")]
impl Adapter {
    /// Returns tiled rendering capabilities for this adapter.
    #[must_use]
    pub fn tiled_capabilities(&self) -> TiledCapabilities {
        if !tiled_features_supported(self.backend()) {
            return TiledCapabilities {
                max_subpasses: 0,
                max_subpass_color_attachments: 0,
                max_input_attachments: 0,
                estimated_tile_memory_bytes: 0,
            };
        }

        let limits = self.limits();
        TiledCapabilities {
            max_subpasses: 4,
            max_subpass_color_attachments: limits.max_color_attachments,
            max_input_attachments: limits.max_color_attachments,
            estimated_tile_memory_bytes: 256 * 1024,
        }
    }
}

/// Returns apply feature implications.
pub(crate) fn apply_feature_implications(features: &mut FeatureSet) {
    if features.contains(&Feature::TextureFormatsTier2) {
        features.insert(Feature::TextureFormatsTier1);
    }
    if features.contains(&Feature::TextureFormatsTier1) {
        features.insert(Feature::Rg11b10UfloatRenderable);
    }
}

/// Constant value for max query count.
pub(crate) const MAX_QUERY_COUNT: u32 = 4096;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

    #[test]
    fn adapter_from_hal_wraps_noop_hal_adapter() {
        let adapter = Adapter::from_hal(hal_noop_adapter());

        assert!(!adapter.name().is_empty());
    }

    #[test]
    fn adapter_name_backend_limits_and_features_match_noop_contract() {
        let adapter = noop_adapter();

        assert!(adapter.name().contains("Noop"));
        assert_eq!(adapter.backend(), yawgpu_hal::HalBackend::Noop);
        assert_eq!(
            adapter.limits().max_bind_groups,
            Limits::DEFAULT.max_bind_groups
        );
        assert_eq!(
            adapter.limits().max_texture_dimension_2d,
            Limits::DEFAULT.max_texture_dimension_2d
        );
        assert!(adapter.features().contains(&Feature::CoreFeaturesAndLimits));
        assert!(adapter.has_feature(Feature::TimestampQuery));
        assert!(!adapter.has_feature(Feature::Other(7)));
    }

    #[test]
    fn adapter_create_device_rejects_unsupported_required_feature() {
        let adapter = noop_adapter();

        let error = adapter
            .create_device(None, &[Feature::Other(7)], "", "")
            .expect_err("unsupported features must reject device creation");

        assert!(matches!(error, Error::Validation(message) if message.contains("not supported")));
    }

    #[test]
    fn adapter_create_device_applies_labels_and_core_feature() {
        let adapter = noop_adapter();

        let device = adapter
            .create_device(None, &[], "device label", "queue label")
            .expect("Noop device creation should succeed");

        assert_eq!(device.label(), "device label");
        assert!(device.has_feature(Feature::CoreFeaturesAndLimits));
        assert_eq!(device.queue().label(), "queue label");
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn tiled_features_supported_is_backend_aware_and_noop_does_not_advertise() {
        assert!(!tiled_features_supported(HalBackend::Noop));
        assert!(tiled_features_supported(HalBackend::Metal));
        assert!(tiled_features_supported(HalBackend::Vulkan));

        let adapter = noop_adapter();
        assert!(!adapter.has_feature(Feature::MultiSubpass));
        assert!(!adapter.has_feature(Feature::TransientAttachments));
        assert!(!adapter.has_feature(Feature::ShaderFramebufferFetch));
        assert!(!adapter.has_feature(Feature::ProgrammableTileDispatch));
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn framebuffer_fetch_support_is_backend_and_path_aware() {
        assert!(framebuffer_fetch_supported(
            HalBackend::Metal,
            FramebufferFetchPath::Disabled
        ));
        assert!(framebuffer_fetch_supported(
            HalBackend::Vulkan,
            FramebufferFetchPath::TileImage
        ));
        assert!(framebuffer_fetch_supported(
            HalBackend::Vulkan,
            FramebufferFetchPath::RasterOrderAttachmentAccess
        ));
        assert!(!framebuffer_fetch_supported(
            HalBackend::Vulkan,
            FramebufferFetchPath::Disabled
        ));
        assert!(!framebuffer_fetch_supported(
            HalBackend::Noop,
            FramebufferFetchPath::TileImage
        ));
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn tiled_feature_advertise_gates_shader_framebuffer_fetch() {
        let mut vulkan_disabled = supported_features();
        add_tiled_features(
            &mut vulkan_disabled,
            HalBackend::Vulkan,
            FramebufferFetchPath::Disabled,
        );
        assert!(vulkan_disabled.contains(&Feature::MultiSubpass));
        assert!(vulkan_disabled.contains(&Feature::TransientAttachments));
        assert!(!vulkan_disabled.contains(&Feature::ShaderFramebufferFetch));
        assert!(vulkan_disabled.contains(&Feature::ProgrammableTileDispatch));

        let mut vulkan_tile_image = supported_features();
        add_tiled_features(
            &mut vulkan_tile_image,
            HalBackend::Vulkan,
            FramebufferFetchPath::TileImage,
        );
        assert!(vulkan_tile_image.contains(&Feature::ShaderFramebufferFetch));

        let mut vulkan_roaa = supported_features();
        add_tiled_features(
            &mut vulkan_roaa,
            HalBackend::Vulkan,
            FramebufferFetchPath::RasterOrderAttachmentAccess,
        );
        assert!(vulkan_roaa.contains(&Feature::ShaderFramebufferFetch));

        let mut metal = supported_features();
        add_tiled_features(
            &mut metal,
            HalBackend::Metal,
            FramebufferFetchPath::Disabled,
        );
        assert!(metal.contains(&Feature::ShaderFramebufferFetch));

        let mut noop = supported_features();
        add_tiled_features(&mut noop, HalBackend::Noop, FramebufferFetchPath::TileImage);
        assert!(!noop.contains(&Feature::MultiSubpass));
        assert!(!noop.contains(&Feature::ShaderFramebufferFetch));
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn tiled_capabilities_are_zero_on_noop() {
        let adapter = noop_adapter();

        assert_eq!(
            adapter.tiled_capabilities(),
            TiledCapabilities {
                max_subpasses: 0,
                max_subpass_color_attachments: 0,
                max_input_attachments: 0,
                estimated_tile_memory_bytes: 0,
            }
        );
    }
}
