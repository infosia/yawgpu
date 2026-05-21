use super::*;

pub fn map_features_to_native(features: &core::FeatureSet) -> native::WGPUSupportedFeatures {
    let features = features
        .iter()
        .copied()
        .map(map_feature_to_native)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let feature_count = features.len();
    let features = Box::into_raw(features);

    native::WGPUSupportedFeatures {
        featureCount: feature_count,
        features: features.cast(),
    }
}

/// Frees the feature array allocated by `map_features_to_native`.
///
/// # Safety
///
/// `features.features`, when non-null, must be a pointer previously returned
/// by `map_features_to_native` with the same `featureCount`.
pub unsafe fn free_supported_features(features: native::WGPUSupportedFeatures) {
    if features.features.is_null() {
        return;
    }
    let slice =
        std::ptr::slice_from_raw_parts_mut(features.features.cast_mut(), features.featureCount);
    drop(Box::from_raw(slice));
}
