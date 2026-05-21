// device_info — dumps the capabilities of the selected adapter and device.
//
// WebGPU exposes two capability surfaces:
//   * Limits   — numeric maxima/minima (texture sizes, binding counts,
//                workgroup dimensions, alignment requirements, ...).
//   * Features — optional, named capabilities a device may or may not
//                support (timestamp queries, texture compression, f16, ...).
// Both are reported separately by the *adapter* (what the hardware can do)
// and by the *device* (what was actually requested/enabled). This program
// prints all four. It is the C analogue of Dawn's `DawnInfo` sample.

#include "framework.h"

// Limits are printed as a name → value table; u32 and u64 fields use
// separate helpers because of their different format specifiers.
static void print_limit_u32(const char *name, uint32_t value) {
    printf("  %-42s %u\n", name, value);
}

static void print_limit_u64(const char *name, uint64_t value) {
    printf("  %-42s %llu\n", name, (unsigned long long)value);
}

// Maps a feature enum to a human-readable name for printing.
static const char *feature_name(WGPUFeatureName feature) {
    switch (feature) {
    case WGPUFeatureName_CoreFeaturesAndLimits:
        return "CoreFeaturesAndLimits";
    case WGPUFeatureName_DepthClipControl:
        return "DepthClipControl";
    case WGPUFeatureName_Depth32FloatStencil8:
        return "Depth32FloatStencil8";
    case WGPUFeatureName_TimestampQuery:
        return "TimestampQuery";
    case WGPUFeatureName_TextureCompressionBC:
        return "TextureCompressionBC";
    case WGPUFeatureName_TextureCompressionBCSliced3D:
        return "TextureCompressionBCSliced3D";
    case WGPUFeatureName_TextureCompressionETC2:
        return "TextureCompressionETC2";
    case WGPUFeatureName_TextureCompressionASTC:
        return "TextureCompressionASTC";
    case WGPUFeatureName_TextureCompressionASTCSliced3D:
        return "TextureCompressionASTCSliced3D";
    case WGPUFeatureName_IndirectFirstInstance:
        return "IndirectFirstInstance";
    case WGPUFeatureName_ShaderF16:
        return "ShaderF16";
    case WGPUFeatureName_RG11B10UfloatRenderable:
        return "RG11B10UfloatRenderable";
    case WGPUFeatureName_BGRA8UnormStorage:
        return "BGRA8UnormStorage";
    case WGPUFeatureName_Float32Filterable:
        return "Float32Filterable";
    case WGPUFeatureName_Float32Blendable:
        return "Float32Blendable";
    case WGPUFeatureName_ClipDistances:
        return "ClipDistances";
    case WGPUFeatureName_DualSourceBlending:
        return "DualSourceBlending";
    case WGPUFeatureName_Subgroups:
        return "Subgroups";
    case WGPUFeatureName_TextureFormatsTier1:
        return "TextureFormatsTier1";
    case WGPUFeatureName_TextureFormatsTier2:
        return "TextureFormatsTier2";
    case WGPUFeatureName_PrimitiveIndex:
        return "PrimitiveIndex";
    case WGPUFeatureName_TextureComponentSwizzle:
        return "TextureComponentSwizzle";
    default:
        return "UnknownFeature";
    }
}

static void print_limits(const WGPULimits *limits) {
    printf("Limits\n");
    print_limit_u32("maxTextureDimension1D", limits->maxTextureDimension1D);
    print_limit_u32("maxTextureDimension2D", limits->maxTextureDimension2D);
    print_limit_u32("maxTextureDimension3D", limits->maxTextureDimension3D);
    print_limit_u32("maxTextureArrayLayers", limits->maxTextureArrayLayers);
    print_limit_u32("maxBindGroups", limits->maxBindGroups);
    print_limit_u32("maxBindGroupsPlusVertexBuffers", limits->maxBindGroupsPlusVertexBuffers);
    print_limit_u32("maxBindingsPerBindGroup", limits->maxBindingsPerBindGroup);
    print_limit_u32("maxDynamicUniformBuffersPerPipelineLayout",
                    limits->maxDynamicUniformBuffersPerPipelineLayout);
    print_limit_u32("maxDynamicStorageBuffersPerPipelineLayout",
                    limits->maxDynamicStorageBuffersPerPipelineLayout);
    print_limit_u32("maxSampledTexturesPerShaderStage", limits->maxSampledTexturesPerShaderStage);
    print_limit_u32("maxSamplersPerShaderStage", limits->maxSamplersPerShaderStage);
    print_limit_u32("maxStorageBuffersPerShaderStage", limits->maxStorageBuffersPerShaderStage);
    print_limit_u32("maxStorageTexturesPerShaderStage", limits->maxStorageTexturesPerShaderStage);
    print_limit_u32("maxUniformBuffersPerShaderStage", limits->maxUniformBuffersPerShaderStage);
    print_limit_u64("maxUniformBufferBindingSize", limits->maxUniformBufferBindingSize);
    print_limit_u64("maxStorageBufferBindingSize", limits->maxStorageBufferBindingSize);
    print_limit_u32("minUniformBufferOffsetAlignment", limits->minUniformBufferOffsetAlignment);
    print_limit_u32("minStorageBufferOffsetAlignment", limits->minStorageBufferOffsetAlignment);
    print_limit_u32("maxVertexBuffers", limits->maxVertexBuffers);
    print_limit_u64("maxBufferSize", limits->maxBufferSize);
    print_limit_u32("maxVertexAttributes", limits->maxVertexAttributes);
    print_limit_u32("maxVertexBufferArrayStride", limits->maxVertexBufferArrayStride);
    print_limit_u32("maxInterStageShaderVariables", limits->maxInterStageShaderVariables);
    print_limit_u32("maxColorAttachments", limits->maxColorAttachments);
    print_limit_u32("maxColorAttachmentBytesPerSample", limits->maxColorAttachmentBytesPerSample);
    print_limit_u32("maxComputeWorkgroupStorageSize", limits->maxComputeWorkgroupStorageSize);
    print_limit_u32("maxComputeInvocationsPerWorkgroup", limits->maxComputeInvocationsPerWorkgroup);
    print_limit_u32("maxComputeWorkgroupSizeX", limits->maxComputeWorkgroupSizeX);
    print_limit_u32("maxComputeWorkgroupSizeY", limits->maxComputeWorkgroupSizeY);
    print_limit_u32("maxComputeWorkgroupSizeZ", limits->maxComputeWorkgroupSizeZ);
    print_limit_u32("maxComputeWorkgroupsPerDimension", limits->maxComputeWorkgroupsPerDimension);
    print_limit_u32("maxImmediateSize", limits->maxImmediateSize);
}

static void print_features(WGPUSupportedFeatures features) {
    printf("Features (%zu)\n", features.featureCount);
    for (size_t i = 0; i < features.featureCount; ++i) {
        printf("  %s (%u)\n", feature_name(features.features[i]), features.features[i]);
    }
}

int main(void) {
    YawgpuContext context = yawgpu_context_create();
    if (!context.instance || !context.adapter || !context.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        yawgpu_context_release(&context);
        return EXIT_FAILURE;
    }

    // vendor / architecture / device / backend type.
    yawgpu_print_adapter_info(context.adapter);

    // Limits are written into a caller-owned struct (an out-parameter); a
    // Success status means it was populated.
    WGPULimits adapter_limits = {0};
    if (wgpuAdapterGetLimits(context.adapter, &adapter_limits) == WGPUStatus_Success) {
        printf("\nAdapter ");
        print_limits(&adapter_limits);
    }

    WGPULimits device_limits = {0};
    if (wgpuDeviceGetLimits(context.device, &device_limits) == WGPUStatus_Success) {
        printf("\nDevice ");
        print_limits(&device_limits);
    }

    // Feature queries allocate an array the caller must free with
    // wgpuSupportedFeaturesFreeMembers once done reading it.
    WGPUSupportedFeatures adapter_features = {0};
    wgpuAdapterGetFeatures(context.adapter, &adapter_features);
    printf("\nAdapter ");
    print_features(adapter_features);
    wgpuSupportedFeaturesFreeMembers(adapter_features);

    WGPUSupportedFeatures device_features = {0};
    wgpuDeviceGetFeatures(context.device, &device_features);
    printf("\nDevice ");
    print_features(device_features);
    wgpuSupportedFeaturesFreeMembers(device_features);

    yawgpu_context_release(&context);
    return EXIT_SUCCESS;
}
