use std::sync::Arc;

use yawgpu_hal::{HalAdapter, HalBackend};

use parking_lot::Mutex;

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
    pub(crate) consumed: Mutex<bool>,
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
            inner: Arc::new(AdapterInner {
                hal,
                feature_level,
                consumed: Mutex::new(false),
            }),
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
        let mut features = supported_features();
        add_texture_compression_features(&mut features, &self.inner.hal);
        add_shader_float16_feature(&mut features, &self.inner.hal);
        add_subgroups_feature(&mut features, &self.inner.hal);
        add_depth_clip_control_feature(&mut features, &self.inner.hal);
        add_float32_blendable_feature(&mut features, &self.inner.hal);
        add_dual_source_blending_feature(&mut features, &self.inner.hal);
        add_indirect_first_instance_feature(&mut features, &self.inner.hal);
        #[cfg(feature = "tiled")]
        add_tiled_features(&mut features, self.backend());

        {
            features
        }
    }

    /// Returns true when this object has the requested feature.
    #[must_use]
    pub fn has_feature(&self, feature: Feature) -> bool {
        self.features().contains(&feature)
    }

    /// Returns the minimum subgroup size, or zero when subgroups are unsupported.
    #[must_use]
    pub fn subgroup_min_size(&self) -> u32 {
        self.inner
            .hal
            .subgroup_size_range()
            .map_or(0, |(min, _)| min)
    }

    /// Returns the maximum subgroup size, or zero when subgroups are unsupported.
    #[must_use]
    pub fn subgroup_max_size(&self) -> u32 {
        self.inner
            .hal
            .subgroup_size_range()
            .map_or(0, |(_, max)| max)
    }

    /// Creates a device and its queue from this adapter, honoring the requested limits and features.
    pub fn create_device(
        &self,
        required_limits: Option<&Limits>,
        required_features: &[Feature],
        label: impl Into<String>,
        queue_label: impl Into<String>,
    ) -> Result<Device, Error> {
        let mut consumed = self.inner.consumed.lock();
        if *consumed {
            return Err(Error::Validation("adapter is consumed".to_owned()));
        }
        let limits = self
            .limits()
            .validate_required_limits(required_limits)
            .map_err(Error::Validation)?;
        let features = self.resolve_features(required_features)?;
        let hal = self.inner.hal.create_device()?;
        *consumed = true;
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
    /// BC texture compression support.
    TextureCompressionBc,
    /// BC compressed 3D texture support.
    TextureCompressionBcSliced3d,
    /// ETC2/EAC texture compression support.
    TextureCompressionEtc2,
    /// ASTC texture compression support.
    TextureCompressionAstc,
    /// ASTC compressed 3D texture support.
    TextureCompressionAstcSliced3d,
    /// Depth32FloatStencil8 texture format support.
    Depth32FloatStencil8,
    /// Rg11b10 ufloat renderable variant.
    Rg11b10UfloatRenderable,
    /// Bgra8Unorm storage texture support.
    Bgra8UnormStorage,
    /// Float32 filterable texture binding support.
    Float32Filterable,
    /// Timestamp query variant.
    TimestampQuery,
    /// WGSL `shader-f16` support.
    ShaderF16,
    /// WGSL `subgroups` support.
    Subgroups,
    /// Depth clip control support.
    DepthClipControl,
    /// Float32 color target blend support.
    Float32Blendable,
    /// Dual-source blending support.
    DualSourceBlending,
    /// Non-zero first instance support in indirect draws.
    IndirectFirstInstance,
    /// Texture component swizzle support.
    TextureComponentSwizzle,
    /// Texture formats tier1 variant.
    TextureFormatsTier1,
    /// Texture formats tier2 variant.
    TextureFormatsTier2,
    /// Multi-subpass render pass support.
    #[cfg(feature = "tiled")]
    MultiSubpass,
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
        Feature::Depth32FloatStencil8,
        Feature::Rg11b10UfloatRenderable,
        Feature::Bgra8UnormStorage,
        Feature::Float32Filterable,
        Feature::TimestampQuery,
        Feature::TextureFormatsTier1,
        Feature::TextureFormatsTier2,
    ]
    .into_iter()
    .collect()
}

fn add_texture_compression_features(features: &mut FeatureSet, hal: &HalAdapter) {
    if hal.supports_texture_compression_bc() {
        features.insert(Feature::TextureCompressionBc);
        if hal.supports_texture_compression_bc_sliced_3d() {
            features.insert(Feature::TextureCompressionBcSliced3d);
        }
    }
    if hal.supports_texture_compression_etc2() {
        features.insert(Feature::TextureCompressionEtc2);
    }
    if hal.supports_texture_compression_astc() {
        features.insert(Feature::TextureCompressionAstc);
        if hal.supports_texture_compression_astc_sliced_3d() {
            features.insert(Feature::TextureCompressionAstcSliced3d);
        }
    }
}

fn add_shader_float16_feature(features: &mut FeatureSet, hal: &HalAdapter) {
    if hal.supports_shader_float16() {
        features.insert(Feature::ShaderF16);
    }
}

fn add_subgroups_feature(features: &mut FeatureSet, hal: &HalAdapter) {
    if hal.supports_subgroups() {
        features.insert(Feature::Subgroups);
    }
}

fn add_depth_clip_control_feature(features: &mut FeatureSet, hal: &HalAdapter) {
    if hal.supports_depth_clip_control() {
        features.insert(Feature::DepthClipControl);
    }
}

fn add_float32_blendable_feature(features: &mut FeatureSet, hal: &HalAdapter) {
    if hal.supports_float32_blendable() {
        features.insert(Feature::Float32Blendable);
    }
}

fn add_dual_source_blending_feature(features: &mut FeatureSet, hal: &HalAdapter) {
    if hal.supports_dual_source_blending() {
        features.insert(Feature::DualSourceBlending);
    }
}

fn add_indirect_first_instance_feature(features: &mut FeatureSet, hal: &HalAdapter) {
    if hal.supports_indirect_first_instance() {
        features.insert(Feature::IndirectFirstInstance);
    }
}

/// Returns true when tiled rendering features are supported by `backend`.
#[cfg(feature = "tiled")]
#[must_use]
pub(crate) fn tiled_features_supported(backend: HalBackend) -> bool {
    matches!(backend, HalBackend::Metal | HalBackend::Vulkan)
}

#[cfg(feature = "tiled")]
fn add_tiled_features(features: &mut FeatureSet, backend: HalBackend) {
    if tiled_features_supported(backend) {
        features.insert(Feature::MultiSubpass);
    }
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
    if features.contains(&Feature::TextureCompressionBcSliced3d) {
        features.insert(Feature::TextureCompressionBc);
    }
    if features.contains(&Feature::TextureCompressionAstcSliced3d) {
        features.insert(Feature::TextureCompressionAstc);
    }
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
    fn compression_features_are_adapter_gated_and_noop_preserves_coverage() {
        let base = supported_features();
        assert!(!base.contains(&Feature::TextureCompressionBc));
        assert!(!base.contains(&Feature::TextureCompressionBcSliced3d));
        assert!(!base.contains(&Feature::TextureCompressionEtc2));
        assert!(!base.contains(&Feature::TextureCompressionAstc));
        assert!(!base.contains(&Feature::TextureCompressionAstcSliced3d));

        let features = noop_adapter().features();
        assert!(features.contains(&Feature::TextureCompressionBc));
        assert!(features.contains(&Feature::TextureCompressionBcSliced3d));
        assert!(features.contains(&Feature::TextureCompressionEtc2));
        assert!(features.contains(&Feature::TextureCompressionAstc));
        assert!(features.contains(&Feature::TextureCompressionAstcSliced3d));
    }

    #[test]
    fn shader_f16_feature_is_adapter_gated_and_noop_advertises() {
        let base = supported_features();
        assert!(!base.contains(&Feature::ShaderF16));

        let features = noop_adapter().features();
        assert!(features.contains(&Feature::ShaderF16));
    }

    #[test]
    fn subgroups_feature_is_adapter_gated_and_noop_advertises() {
        let base = supported_features();
        assert!(!base.contains(&Feature::Subgroups));

        let adapter = noop_adapter();
        let features = adapter.features();
        assert!(features.contains(&Feature::Subgroups));
        assert_eq!(adapter.subgroup_min_size(), 4);
        assert_eq!(adapter.subgroup_max_size(), 4);
    }

    #[test]
    fn depth_clip_control_feature_is_adapter_gated_and_noop_advertises() {
        let base = supported_features();
        assert!(!base.contains(&Feature::DepthClipControl));

        let features = noop_adapter().features();
        assert!(features.contains(&Feature::DepthClipControl));
    }

    #[test]
    fn float32_blendable_feature_is_adapter_gated_and_noop_advertises() {
        let base = supported_features();
        assert!(!base.contains(&Feature::Float32Blendable));

        let features = noop_adapter().features();
        assert!(features.contains(&Feature::Float32Blendable));
    }

    #[test]
    fn dual_source_blending_feature_is_adapter_gated_and_noop_advertises() {
        let base = supported_features();
        assert!(!base.contains(&Feature::DualSourceBlending));

        let features = noop_adapter().features();
        assert!(features.contains(&Feature::DualSourceBlending));
    }

    #[test]
    fn indirect_first_instance_feature_is_adapter_gated_and_noop_advertises() {
        let base = supported_features();
        assert!(!base.contains(&Feature::IndirectFirstInstance));

        let features = noop_adapter().features();
        assert!(features.contains(&Feature::IndirectFirstInstance));
    }

    #[test]
    fn adapter_create_device_accepts_noop_shader_f16_feature() {
        let adapter = noop_adapter();

        let device = adapter
            .create_device(None, &[Feature::ShaderF16], "", "")
            .expect("Noop device should accept shader-f16");

        assert!(device.has_feature(Feature::ShaderF16));
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

    #[test]
    fn adapter_create_device_consumes_adapter_only_after_success() {
        let adapter = noop_adapter();

        let error = adapter
            .create_device(None, &[Feature::Other(7)], "", "")
            .expect_err("failed requestDevice should reject");
        assert!(matches!(error, Error::Validation(_)));

        let device = adapter
            .create_device(None, &[], "device label", "queue label")
            .expect("first valid requestDevice should succeed");
        assert_eq!(device.label(), "device label");

        let error = adapter
            .create_device(None, &[], "", "")
            .expect_err("successful requestDevice should consume adapter");
        assert!(matches!(error, Error::Validation(message) if message.contains("consumed")));
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
