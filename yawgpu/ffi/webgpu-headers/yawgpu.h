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

#define YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT ((WGPUSType)0x70000010u)

typedef struct YaWGPUInputAttachmentBindingLayout {
    WGPUChainedStruct chain;
    WGPUTextureSampleType sampleType;
    WGPUBool multisampled;
} YaWGPUInputAttachmentBindingLayout;

#define YAWGPU_INPUT_ATTACHMENT_BINDING_LAYOUT_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUInputAttachmentBindingLayout, { \
    /*.chain=*/{NULL, YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT} _wgpu_COMMA \
    /*.sampleType=*/WGPUTextureSampleType_Float _wgpu_COMMA \
    /*.multisampled=*/0 _wgpu_COMMA \
})

/* Input attachment resources are auto-wired from the subpass pass layout. */

typedef struct YaWGPUSubpassPassLayoutImpl* YaWGPUSubpassPassLayout;
typedef struct YaWGPUSubpassRenderPassEncoderImpl* YaWGPUSubpassRenderPassEncoder;

typedef struct YaWGPUAttachmentLayout {
    WGPUTextureFormat format;
    uint32_t sampleCount;
} YaWGPUAttachmentLayout;

#define YAWGPU_ATTACHMENT_LAYOUT_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUAttachmentLayout, { \
    /*.format=*/WGPUTextureFormat_Undefined _wgpu_COMMA \
    /*.sampleCount=*/1 _wgpu_COMMA \
})

typedef enum YaWGPUSubpassDependencyType {
    YaWGPUSubpassDependencyType_ColorToInput = 0x00000000,
    YaWGPUSubpassDependencyType_DepthToInput = 0x00000001,
    YaWGPUSubpassDependencyType_ColorDepthToInput = 0x00000002,
    YaWGPUSubpassDependencyType_Force32 = 0x7FFFFFFF
} YaWGPUSubpassDependencyType;

typedef struct YaWGPUSubpassDependency {
    uint32_t srcSubpass;
    uint32_t dstSubpass;
    YaWGPUSubpassDependencyType dependencyType;
    WGPUBool byRegion;
} YaWGPUSubpassDependency;

#define YAWGPU_DEPTH_STENCIL_ATTACHMENT_INDEX 0xFFFFFFFFu

typedef struct YaWGPUSubpassInputAttachment {
    uint32_t group;
    uint32_t binding;
    uint32_t sourceSubpass;
    uint32_t sourceAttachment;
} YaWGPUSubpassInputAttachment;

typedef struct YaWGPUSubpassLayoutDesc {
    uint32_t const* colorAttachmentIndices;
    size_t colorAttachmentIndexCount;
    WGPUBool usesDepthStencil;
    YaWGPUSubpassInputAttachment const* inputAttachments;
    size_t inputAttachmentCount;
} YaWGPUSubpassLayoutDesc;

#define YAWGPU_SUBPASS_LAYOUT_DESC_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUSubpassLayoutDesc, { \
    /*.colorAttachmentIndices=*/NULL _wgpu_COMMA \
    /*.colorAttachmentIndexCount=*/0 _wgpu_COMMA \
    /*.usesDepthStencil=*/0 _wgpu_COMMA \
    /*.inputAttachments=*/NULL _wgpu_COMMA \
    /*.inputAttachmentCount=*/0 _wgpu_COMMA \
})

typedef struct YaWGPUSubpassPassLayoutDescriptor {
    WGPUChainedStruct const* nextInChain;
    WGPUStringView label;
    YaWGPUAttachmentLayout const* colorAttachments;
    size_t colorAttachmentCount;
    YaWGPUAttachmentLayout depthStencilAttachment;
    YaWGPUSubpassLayoutDesc const* subpasses;
    size_t subpassCount;
    YaWGPUSubpassDependency const* dependencies;
    size_t dependencyCount;
} YaWGPUSubpassPassLayoutDescriptor;

#define YAWGPU_SUBPASS_PASS_LAYOUT_DESCRIPTOR_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUSubpassPassLayoutDescriptor, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.label=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.colorAttachments=*/NULL _wgpu_COMMA \
    /*.colorAttachmentCount=*/0 _wgpu_COMMA \
    /*.depthStencilAttachment=*/{WGPUTextureFormat_Undefined, 1} _wgpu_COMMA \
    /*.subpasses=*/NULL _wgpu_COMMA \
    /*.subpassCount=*/0 _wgpu_COMMA \
    /*.dependencies=*/NULL _wgpu_COMMA \
    /*.dependencyCount=*/0 _wgpu_COMMA \
})

YaWGPUSubpassPassLayout yawgpuDeviceCreateSubpassPassLayout(
    WGPUDevice device,
    YaWGPUSubpassPassLayoutDescriptor const* descriptor);
void yawgpuSubpassPassLayoutAddRef(YaWGPUSubpassPassLayout layout);
void yawgpuSubpassPassLayoutRelease(YaWGPUSubpassPassLayout layout);

typedef struct YaWGPUSubpassRenderPipelineDescriptor {
    WGPUChainedStruct const* nextInChain;
    WGPURenderPipelineDescriptor base;
    YaWGPUSubpassPassLayout passLayout;
    uint32_t subpassIndex;
} YaWGPUSubpassRenderPipelineDescriptor;

#define YAWGPU_SUBPASS_RENDER_PIPELINE_DESCRIPTOR_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUSubpassRenderPipelineDescriptor, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.base=*/WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT _wgpu_COMMA \
    /*.passLayout=*/NULL _wgpu_COMMA \
    /*.subpassIndex=*/0 _wgpu_COMMA \
})

WGPURenderPipeline yawgpuDeviceCreateSubpassRenderPipeline(
    WGPUDevice device,
    YaWGPUSubpassRenderPipelineDescriptor const* descriptor);

typedef enum YaWGPUSubpassAttachmentKind {
    YaWGPUSubpassAttachmentKind_Persistent = 0x00000000,
    YaWGPUSubpassAttachmentKind_Transient = 0x00000001,
    YaWGPUSubpassAttachmentKind_Force32 = 0x7FFFFFFF
} YaWGPUSubpassAttachmentKind;

typedef struct YaWGPUColorAttachmentBinding {
    YaWGPUSubpassAttachmentKind kind;
    WGPUTextureView view;
    WGPUTextureView resolveTarget;
    YaWGPUTransientAttachment transient;
    WGPULoadOp loadOp;
    WGPUStoreOp storeOp;
    WGPUColor clearValue;
} YaWGPUColorAttachmentBinding;

#define YAWGPU_COLOR_ATTACHMENT_BINDING_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUColorAttachmentBinding, { \
    /*.kind=*/YaWGPUSubpassAttachmentKind_Persistent _wgpu_COMMA \
    /*.view=*/NULL _wgpu_COMMA \
    /*.resolveTarget=*/NULL _wgpu_COMMA \
    /*.transient=*/NULL _wgpu_COMMA \
    /*.loadOp=*/WGPULoadOp_Load _wgpu_COMMA \
    /*.storeOp=*/WGPUStoreOp_Store _wgpu_COMMA \
    /*.clearValue=*/{0, 0, 0, 0} _wgpu_COMMA \
})

typedef struct YaWGPUDepthStencilAttachmentBinding {
    YaWGPUSubpassAttachmentKind kind;
    WGPUTextureView view;
    YaWGPUTransientAttachment transient;
    WGPULoadOp depthLoadOp;
    WGPUStoreOp depthStoreOp;
    float depthClearValue;
    WGPULoadOp stencilLoadOp;
    WGPUStoreOp stencilStoreOp;
    uint32_t stencilClearValue;
} YaWGPUDepthStencilAttachmentBinding;

#define YAWGPU_DEPTH_STENCIL_ATTACHMENT_BINDING_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUDepthStencilAttachmentBinding, { \
    /*.kind=*/YaWGPUSubpassAttachmentKind_Persistent _wgpu_COMMA \
    /*.view=*/NULL _wgpu_COMMA \
    /*.transient=*/NULL _wgpu_COMMA \
    /*.depthLoadOp=*/WGPULoadOp_Load _wgpu_COMMA \
    /*.depthStoreOp=*/WGPUStoreOp_Store _wgpu_COMMA \
    /*.depthClearValue=*/1.0f _wgpu_COMMA \
    /*.stencilLoadOp=*/WGPULoadOp_Load _wgpu_COMMA \
    /*.stencilStoreOp=*/WGPUStoreOp_Store _wgpu_COMMA \
    /*.stencilClearValue=*/0 _wgpu_COMMA \
})

typedef struct YaWGPUSubpassRenderPassDescriptor {
    WGPUChainedStruct const* nextInChain;
    WGPUStringView label;
    YaWGPUSubpassPassLayout passLayout;
    WGPUExtent3D extent;
    YaWGPUColorAttachmentBinding const* colorAttachments;
    size_t colorAttachmentCount;
    YaWGPUDepthStencilAttachmentBinding const* depthStencilAttachment;
} YaWGPUSubpassRenderPassDescriptor;

#define YAWGPU_SUBPASS_RENDER_PASS_DESCRIPTOR_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUSubpassRenderPassDescriptor, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.label=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.passLayout=*/NULL _wgpu_COMMA \
    /*.extent=*/{0, 0, 1} _wgpu_COMMA \
    /*.colorAttachments=*/NULL _wgpu_COMMA \
    /*.colorAttachmentCount=*/0 _wgpu_COMMA \
    /*.depthStencilAttachment=*/NULL _wgpu_COMMA \
})

YaWGPUSubpassRenderPassEncoder yawgpuCommandEncoderBeginSubpassRenderPass(
    WGPUCommandEncoder encoder,
    YaWGPUSubpassRenderPassDescriptor const* descriptor);
void yawgpuSubpassRenderPassEncoderNextSubpass(YaWGPUSubpassRenderPassEncoder encoder);
void yawgpuSubpassRenderPassEncoderEnd(YaWGPUSubpassRenderPassEncoder encoder);
void yawgpuSubpassRenderPassEncoderSetPipeline(YaWGPUSubpassRenderPassEncoder encoder, WGPURenderPipeline pipeline);
void yawgpuSubpassRenderPassEncoderSetBindGroup(YaWGPUSubpassRenderPassEncoder encoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const* dynamicOffsets);
void yawgpuSubpassRenderPassEncoderSetVertexBuffer(YaWGPUSubpassRenderPassEncoder encoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size);
void yawgpuSubpassRenderPassEncoderSetIndexBuffer(YaWGPUSubpassRenderPassEncoder encoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size);
void yawgpuSubpassRenderPassEncoderDraw(YaWGPUSubpassRenderPassEncoder encoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance);
void yawgpuSubpassRenderPassEncoderDrawIndexed(YaWGPUSubpassRenderPassEncoder encoder, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance);
void yawgpuSubpassRenderPassEncoderSetViewport(YaWGPUSubpassRenderPassEncoder encoder, float x, float y, float width, float height, float minDepth, float maxDepth);
void yawgpuSubpassRenderPassEncoderSetScissorRect(YaWGPUSubpassRenderPassEncoder encoder, uint32_t x, uint32_t y, uint32_t width, uint32_t height);
void yawgpuSubpassRenderPassEncoderAddRef(YaWGPUSubpassRenderPassEncoder encoder);
void yawgpuSubpassRenderPassEncoderRelease(YaWGPUSubpassRenderPassEncoder encoder);

/* B4b+ adds real backend subpass execution here. */
#endif

#endif
