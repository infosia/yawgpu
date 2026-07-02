//! WGSL language feature support shared by API reporting and shader parsing.

/// `WGPUWGSLLanguageFeatureName_ReadonlyAndReadwriteStorageTextures`.
pub const WGSL_LANGUAGE_FEATURE_READONLY_AND_READWRITE_STORAGE_TEXTURES: u32 = 1;
/// `WGPUWGSLLanguageFeatureName_Packed4x8IntegerDotProduct`.
pub const WGSL_LANGUAGE_FEATURE_PACKED_4X8_INTEGER_DOT_PRODUCT: u32 = 2;
/// `WGPUWGSLLanguageFeatureName_UnrestrictedPointerParameters`.
pub const WGSL_LANGUAGE_FEATURE_UNRESTRICTED_POINTER_PARAMETERS: u32 = 3;
/// `WGPUWGSLLanguageFeatureName_PointerCompositeAccess`.
pub const WGSL_LANGUAGE_FEATURE_POINTER_COMPOSITE_ACCESS: u32 = 4;
/// `WGPUWGSLLanguageFeatureName_UniformBufferStandardLayout`.
pub const WGSL_LANGUAGE_FEATURE_UNIFORM_BUFFER_STANDARD_LAYOUT: u32 = 5;
/// `WGPUWGSLLanguageFeatureName_SubgroupId`.
pub const WGSL_LANGUAGE_FEATURE_SUBGROUP_ID: u32 = 6;
/// `WGPUWGSLLanguageFeatureName_TextureAndSamplerLet`.
pub const WGSL_LANGUAGE_FEATURE_TEXTURE_AND_SAMPLER_LET: u32 = 7;
/// `WGPUWGSLLanguageFeatureName_SubgroupUniformity`.
pub const WGSL_LANGUAGE_FEATURE_SUBGROUP_UNIFORMITY: u32 = 8;
/// `WGPUWGSLLanguageFeatureName_TextureFormatsTier1`.
pub const WGSL_LANGUAGE_FEATURE_TEXTURE_FORMATS_TIER1: u32 = 9;
/// `WGPUWGSLLanguageFeatureName_LinearIndexing`.
pub const WGSL_LANGUAGE_FEATURE_LINEAR_INDEXING: u32 = 10;
/// `WGPUWGSLLanguageFeatureName_ImmediateAddressSpace`.
pub const WGSL_LANGUAGE_FEATURE_IMMEDIATE_ADDRESS_SPACE: u32 = 11;

/// WGSL language features yawgpu's frontend supports (compiles and executes),
/// reported via `wgpuInstanceGetWGSLLanguageFeatures` and allowed by the Tint shim.
pub const SUPPORTED_WGSL_LANGUAGE_FEATURES: &[u32] = &[
    WGSL_LANGUAGE_FEATURE_READONLY_AND_READWRITE_STORAGE_TEXTURES,
    WGSL_LANGUAGE_FEATURE_PACKED_4X8_INTEGER_DOT_PRODUCT,
    WGSL_LANGUAGE_FEATURE_UNRESTRICTED_POINTER_PARAMETERS,
    WGSL_LANGUAGE_FEATURE_POINTER_COMPOSITE_ACCESS,
    WGSL_LANGUAGE_FEATURE_UNIFORM_BUFFER_STANDARD_LAYOUT,
    WGSL_LANGUAGE_FEATURE_SUBGROUP_ID,
    WGSL_LANGUAGE_FEATURE_TEXTURE_AND_SAMPLER_LET,
    WGSL_LANGUAGE_FEATURE_SUBGROUP_UNIFORMITY,
    WGSL_LANGUAGE_FEATURE_TEXTURE_FORMATS_TIER1,
    WGSL_LANGUAGE_FEATURE_LINEAR_INDEXING,
    WGSL_LANGUAGE_FEATURE_IMMEDIATE_ADDRESS_SPACE,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_wgsl_language_features_match_canonical_api_values() {
        assert_eq!(
            SUPPORTED_WGSL_LANGUAGE_FEATURES,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        );
        assert!(SUPPORTED_WGSL_LANGUAGE_FEATURES.contains(&6));
        assert!(SUPPORTED_WGSL_LANGUAGE_FEATURES.contains(&8));
    }
}
