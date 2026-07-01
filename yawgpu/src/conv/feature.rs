use super::*;

/// Converts feature into the corresponding yawgpu representation.
#[must_use]
#[allow(clippy::unnecessary_cast)]
pub fn map_feature(value: native::WGPUFeatureName) -> core::Feature {
    match value {
        native::WGPUFeatureName_CoreFeaturesAndLimits => core::Feature::CoreFeaturesAndLimits,
        native::WGPUFeatureName_TextureCompressionBC => core::Feature::TextureCompressionBc,
        native::WGPUFeatureName_TextureCompressionBCSliced3D => {
            core::Feature::TextureCompressionBcSliced3d
        }
        native::WGPUFeatureName_TextureCompressionETC2 => core::Feature::TextureCompressionEtc2,
        native::WGPUFeatureName_TextureCompressionASTC => core::Feature::TextureCompressionAstc,
        native::WGPUFeatureName_TextureCompressionASTCSliced3D => {
            core::Feature::TextureCompressionAstcSliced3d
        }
        native::WGPUFeatureName_Depth32FloatStencil8 => core::Feature::Depth32FloatStencil8,
        native::WGPUFeatureName_RG11B10UfloatRenderable => core::Feature::Rg11b10UfloatRenderable,
        native::WGPUFeatureName_BGRA8UnormStorage => core::Feature::Bgra8UnormStorage,
        native::WGPUFeatureName_Float32Filterable => core::Feature::Float32Filterable,
        native::WGPUFeatureName_TimestampQuery => core::Feature::TimestampQuery,
        native::WGPUFeatureName_ShaderF16 => core::Feature::ShaderF16,
        native::WGPUFeatureName_Subgroups => core::Feature::Subgroups,
        native::WGPUFeatureName_TextureComponentSwizzle => core::Feature::TextureComponentSwizzle,
        native::WGPUFeatureName_TextureFormatsTier1 => core::Feature::TextureFormatsTier1,
        native::WGPUFeatureName_TextureFormatsTier2 => core::Feature::TextureFormatsTier2,
        #[cfg(feature = "tiled")]
        crate::YaWGPUFeatureName_MultiSubpass => core::Feature::MultiSubpass,
        other => core::Feature::Other(other as u32),
    }
}

/// Converts feature to native into the corresponding yawgpu representation.
#[must_use]
#[allow(clippy::unnecessary_cast)]
pub fn map_feature_to_native(value: core::Feature) -> native::WGPUFeatureName {
    match value {
        core::Feature::CoreFeaturesAndLimits => native::WGPUFeatureName_CoreFeaturesAndLimits,
        core::Feature::TextureCompressionBc => native::WGPUFeatureName_TextureCompressionBC,
        core::Feature::TextureCompressionBcSliced3d => {
            native::WGPUFeatureName_TextureCompressionBCSliced3D
        }
        core::Feature::TextureCompressionEtc2 => native::WGPUFeatureName_TextureCompressionETC2,
        core::Feature::TextureCompressionAstc => native::WGPUFeatureName_TextureCompressionASTC,
        core::Feature::TextureCompressionAstcSliced3d => {
            native::WGPUFeatureName_TextureCompressionASTCSliced3D
        }
        core::Feature::Depth32FloatStencil8 => native::WGPUFeatureName_Depth32FloatStencil8,
        core::Feature::Rg11b10UfloatRenderable => native::WGPUFeatureName_RG11B10UfloatRenderable,
        core::Feature::Bgra8UnormStorage => native::WGPUFeatureName_BGRA8UnormStorage,
        core::Feature::Float32Filterable => native::WGPUFeatureName_Float32Filterable,
        core::Feature::TimestampQuery => native::WGPUFeatureName_TimestampQuery,
        core::Feature::ShaderF16 => native::WGPUFeatureName_ShaderF16,
        core::Feature::Subgroups => native::WGPUFeatureName_Subgroups,
        core::Feature::TextureComponentSwizzle => native::WGPUFeatureName_TextureComponentSwizzle,
        core::Feature::TextureFormatsTier1 => native::WGPUFeatureName_TextureFormatsTier1,
        core::Feature::TextureFormatsTier2 => native::WGPUFeatureName_TextureFormatsTier2,
        #[cfg(feature = "tiled")]
        core::Feature::MultiSubpass => crate::YaWGPUFeatureName_MultiSubpass,
        core::Feature::Other(value) => value as native::WGPUFeatureName,
        // exhaustive as of core::Feature @ 2026-05-17
        _ => native::WGPUFeatureName_Force32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_subgroups_feature_round_trip() {
        assert_eq!(
            map_feature(native::WGPUFeatureName_Subgroups),
            core::Feature::Subgroups
        );
        assert_eq!(
            map_feature_to_native(core::Feature::Subgroups),
            native::WGPUFeatureName_Subgroups
        );
    }
}

/// Stores info metadata.
#[derive(Debug, Clone, Copy)]
pub struct DeviceLostCallbackInfo {
    /// Mode.
    pub mode: native::WGPUCallbackMode,
    /// Callback.
    pub callback: native::WGPUDeviceLostCallback,
    /// Userdata1.
    pub userdata1: usize,
    /// Userdata2.
    pub userdata2: usize,
}

/// Stores uncaptured error callback metadata.
#[derive(Debug, Clone, Copy)]
pub struct UncapturedErrorCallbackInfo {
    /// Callback.
    pub callback: native::WGPUUncapturedErrorCallback,
    /// Userdata1.
    pub userdata1: usize,
    /// Userdata2.
    pub userdata2: usize,
}

/// Converts device lost callback info into the corresponding yawgpu representation.
#[must_use]
pub fn map_device_lost_callback_info(
    value: native::WGPUDeviceLostCallbackInfo,
) -> DeviceLostCallbackInfo {
    DeviceLostCallbackInfo {
        mode: value.mode,
        callback: value.callback,
        userdata1: value.userdata1 as usize,
        userdata2: value.userdata2 as usize,
    }
}

/// Converts uncaptured error callback info into the corresponding yawgpu representation.
#[must_use]
pub fn map_uncaptured_error_callback_info(
    value: native::WGPUUncapturedErrorCallbackInfo,
) -> UncapturedErrorCallbackInfo {
    UncapturedErrorCallbackInfo {
        callback: value.callback,
        userdata1: value.userdata1 as usize,
        userdata2: value.userdata2 as usize,
    }
}

/// Converts device lost reason into the corresponding yawgpu representation.
#[must_use]
pub fn map_device_lost_reason(reason: core::DeviceLostReason) -> native::WGPUDeviceLostReason {
    match reason {
        core::DeviceLostReason::Unknown => native::WGPUDeviceLostReason_Unknown,
        core::DeviceLostReason::Destroyed => native::WGPUDeviceLostReason_Destroyed,
        core::DeviceLostReason::CallbackCancelled => native::WGPUDeviceLostReason_CallbackCancelled,
        core::DeviceLostReason::FailedCreation => native::WGPUDeviceLostReason_FailedCreation,
        // exhaustive as of core::DeviceLostReason @ 2026-05-17
        _ => native::WGPUDeviceLostReason_Unknown,
    }
}
