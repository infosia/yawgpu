use super::*;

#[must_use]
#[allow(clippy::unnecessary_cast)]
pub fn map_feature(value: native::WGPUFeatureName) -> core::Feature {
    match value {
        native::WGPUFeatureName_CoreFeaturesAndLimits => core::Feature::CoreFeaturesAndLimits,
        native::WGPUFeatureName_RG11B10UfloatRenderable => core::Feature::Rg11b10UfloatRenderable,
        native::WGPUFeatureName_TimestampQuery => core::Feature::TimestampQuery,
        native::WGPUFeatureName_TextureFormatsTier1 => core::Feature::TextureFormatsTier1,
        native::WGPUFeatureName_TextureFormatsTier2 => core::Feature::TextureFormatsTier2,
        other => core::Feature::Other(other as u32),
    }
}

#[must_use]
#[allow(clippy::unnecessary_cast)]
pub fn map_feature_to_native(value: core::Feature) -> native::WGPUFeatureName {
    match value {
        core::Feature::CoreFeaturesAndLimits => native::WGPUFeatureName_CoreFeaturesAndLimits,
        core::Feature::Rg11b10UfloatRenderable => native::WGPUFeatureName_RG11B10UfloatRenderable,
        core::Feature::TimestampQuery => native::WGPUFeatureName_TimestampQuery,
        core::Feature::TextureFormatsTier1 => native::WGPUFeatureName_TextureFormatsTier1,
        core::Feature::TextureFormatsTier2 => native::WGPUFeatureName_TextureFormatsTier2,
        core::Feature::Other(value) => value as native::WGPUFeatureName,
        // exhaustive as of core::Feature @ 2026-05-17
        _ => native::WGPUFeatureName_Force32,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceLostCallbackInfo {
    pub mode: native::WGPUCallbackMode,
    pub callback: native::WGPUDeviceLostCallback,
    pub userdata1: usize,
    pub userdata2: usize,
}

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
