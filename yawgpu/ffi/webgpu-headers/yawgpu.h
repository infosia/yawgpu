// yawgpu.h - yawgpu vendor extensions for webgpu.h.
//
// Naming convention:
// - functions use yawgpu* names.
// - types, structs, enums, and handles use YaWGPU* names.
// - constants, macros, and SType tags use YAWGPU_* / YAWGPU_STYPE_* names.
// - feature names use YaWGPUFeatureName_* names.
// Standard webgpu.h types keep their WGPU* names.

#ifndef YAWGPU_H_
#define YAWGPU_H_

#include "webgpu.h"

#define YAWGPU_STYPE_INSTANCE_BACKEND_SELECT ((WGPUSType)0x70000001u)

enum {
    YAWGPU_INSTANCE_BACKEND_NOOP = 0,
    YAWGPU_INSTANCE_BACKEND_METAL = 1,
    YAWGPU_INSTANCE_BACKEND_VULKAN = 2,
};

typedef struct YaWGPUInstanceBackendSelect {
    WGPUChainedStruct chain;
    uint32_t backend;
} YaWGPUInstanceBackendSelect;

#if defined(YAWGPU_HAS_SHADER_PASSTHROUGH)
typedef struct YaWGPUShaderModuleSpirVDescriptor {
    WGPUChainedStruct const* nextInChain;
    WGPUStringView label;
    uint32_t codeSize;
    uint32_t const* code;
} YaWGPUShaderModuleSpirVDescriptor;

#define YAWGPU_SHADER_MODULE_SPIRV_DESCRIPTOR_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUShaderModuleSpirVDescriptor, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.label=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.codeSize=*/0 _wgpu_COMMA \
    /*.code=*/NULL _wgpu_COMMA \
})

typedef struct YaWGPUMslEntryPoint {
    WGPUStringView name;
    WGPUShaderStage stage;
    uint32_t workgroupSize[3];
} YaWGPUMslEntryPoint;

#define YAWGPU_MSL_ENTRY_POINT_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUMslEntryPoint, { \
    /*.name=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.stage=*/WGPUShaderStage_None _wgpu_COMMA \
    /*.workgroupSize=*/{0, 0, 0} _wgpu_COMMA \
})

typedef struct YaWGPUShaderModuleMslDescriptor {
    WGPUChainedStruct const* nextInChain;
    WGPUStringView label;
    WGPUStringView code;
    size_t entryPointCount;
    YaWGPUMslEntryPoint const* entryPoints;
} YaWGPUShaderModuleMslDescriptor;

#define YAWGPU_SHADER_MODULE_MSL_DESCRIPTOR_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUShaderModuleMslDescriptor, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.label=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.code=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.entryPointCount=*/0 _wgpu_COMMA \
    /*.entryPoints=*/NULL _wgpu_COMMA \
})

/*
 * Metal binding-index mapping for MSL passthrough
 * ------------------------------------------------
 * MSL modules are not reflected by yawgpu, so the caller's MSL source must use
 * the same Metal resource indices yawgpu derives from the explicit pipeline
 * layout.
 *
 * For buffer bindings in all bind groups:
 *   1. Walk the explicit pipeline layout's bind group layouts in group order.
 *   2. Collect only buffer entries (uniform, storage, and read-only storage).
 *   3. Sort the collected entries by (group, binding).
 *   4. Assign dense Metal buffer indices starting at zero.
 *
 * Therefore the first collected buffer uses [[buffer(0)]], the second uses
 * [[buffer(1)]], and so on, independent of the WebGPU binding number. Render
 * vertex buffers are assigned after these bind-group buffer indices: vertex
 * slot N uses [[buffer(buffer_binding_count + N)]].
 *
 * Texture and sampler declarations are not remapped by this buffer mapping;
 * write them with their WebGPU binding number as the Metal index:
 * [[texture(binding)]] and [[sampler(binding)]]. This mirrors the current
 * WGSL-to-MSL path, which only supplies explicit BindingMap entries for
 * buffers.
 *
 * Worked example for an explicit pipeline layout:
 *   group 0 binding 0: uniform buffer
 *   group 0 binding 1: sampled texture
 *   group 0 binding 2: sampler
 *   group 1 binding 0: storage buffer
 *
 * The buffer entries sorted by (group, binding) are (0,0), then (1,0), so the
 * MSL indices are:
 *
 *   struct Uniforms { float4 tint; };
 *   struct Data { device float4* values; };
 *
 *   fragment float4 fs_main(
 *       constant Uniforms& uniforms [[buffer(0)]],
 *       texture2d<float> image [[texture(1)]],
 *       sampler imageSampler [[sampler(2)]],
 *       device Data& data [[buffer(1)]])
 *   {
 *       return image.sample(imageSampler, float2(0.5)) + uniforms.tint + data.values[0];
 *   }
 */

WGPUShaderModule yawgpuDeviceCreateShaderModuleSpirV(
    WGPUDevice device,
    YaWGPUShaderModuleSpirVDescriptor const* descriptor);

WGPUShaderModule yawgpuDeviceCreateShaderModuleMsl(
    WGPUDevice device,
    YaWGPUShaderModuleMslDescriptor const* descriptor);
#endif

#if defined(YAWGPU_HAS_TILED)
#define YaWGPUFeatureName_MultiSubpass ((WGPUFeatureName)0x70010001u)
#define YaWGPUFeatureName_TransientAttachments ((WGPUFeatureName)0x70010002u)
#define YaWGPUFeatureName_ShaderFramebufferFetch ((WGPUFeatureName)0x70010003u)
#define YaWGPUFeatureName_ProgrammableTileDispatch ((WGPUFeatureName)0x70010004u)

typedef struct YaWGPUTiledCapabilities {
    WGPUChainedStruct const* nextInChain;
    uint32_t maxSubpasses;
    uint32_t maxSubpassColorAttachments;
    uint32_t maxInputAttachments;
    uint32_t estimatedTileMemoryBytes;
} YaWGPUTiledCapabilities;

#define YAWGPU_TILED_CAPABILITIES_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUTiledCapabilities, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.maxSubpasses=*/0 _wgpu_COMMA \
    /*.maxSubpassColorAttachments=*/0 _wgpu_COMMA \
    /*.maxInputAttachments=*/0 _wgpu_COMMA \
    /*.estimatedTileMemoryBytes=*/0 _wgpu_COMMA \
})

WGPUStatus yawgpuAdapterGetTiledCapabilities(
    WGPUAdapter adapter,
    YaWGPUTiledCapabilities* capabilities);

typedef struct YaWGPUTransientAttachmentImpl* YaWGPUTransientAttachment;

typedef enum YaWGPUTransientSizeMode {
    YaWGPUTransientSizeMode_MatchTarget = 0x00000000,
    YaWGPUTransientSizeMode_Explicit = 0x00000001,
    YaWGPUTransientSizeMode_Force32 = 0x7FFFFFFF
} YaWGPUTransientSizeMode;

typedef struct YaWGPUTransientAttachmentDescriptor {
    WGPUChainedStruct const* nextInChain;
    WGPUStringView label;
    WGPUTextureFormat format;
    YaWGPUTransientSizeMode sizeMode;
    uint32_t width;
    uint32_t height;
    uint32_t sampleCount;
} YaWGPUTransientAttachmentDescriptor;

#define YAWGPU_TRANSIENT_ATTACHMENT_DESCRIPTOR_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUTransientAttachmentDescriptor, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.label=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.format=*/WGPUTextureFormat_Undefined _wgpu_COMMA \
    /*.sizeMode=*/YaWGPUTransientSizeMode_MatchTarget _wgpu_COMMA \
    /*.width=*/0 _wgpu_COMMA \
    /*.height=*/0 _wgpu_COMMA \
    /*.sampleCount=*/1 _wgpu_COMMA \
})

YaWGPUTransientAttachment yawgpuDeviceCreateTransientAttachment(
    WGPUDevice device,
    YaWGPUTransientAttachmentDescriptor const* descriptor);

void yawgpuTransientAttachmentAddRef(YaWGPUTransientAttachment attachment);
void yawgpuTransientAttachmentRelease(YaWGPUTransientAttachment attachment);

/* B3+ adds subpass surface here. */
#endif

#endif
