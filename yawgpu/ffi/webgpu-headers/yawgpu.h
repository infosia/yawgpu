/**
 * @file
 * yawgpu vendor extensions for `webgpu.h`.
 *
 * This header layers a small set of vendor-prefixed types, constants, and
 * functions on top of the standard `webgpu.h` shipped alongside it. It is
 * always safe to include after `webgpu.h`; the feature-gated sections only
 * expose their symbols when their compile-time macro is defined.
 *
 * \par Naming convention
 * - Functions use `yawgpu*` names.
 * - Types, structs, enums, and handles use `YaWGPU*` names.
 * - Constants, macros, and `SType` tags use `YAWGPU_*` / `YAWGPU_STYPE_*` names.
 * - Feature names use `YaWGPUFeatureName_*` names.
 * - Standard `webgpu.h` types keep their `WGPU*` names.
 *
 * \par Sections
 * - **Instance backend select** (always available): explicit backend
 *   pick-list, chained into `WGPUInstanceDescriptor`.
 * - **Shader passthrough** (`YAWGPU_HAS_SHADER_PASSTHROUGH`): vendor entry
 *   points that consume pre-compiled SPIR-V or Metal Shading Language
 *   modules instead of WGSL.
 * - **Tiled / multi-subpass rendering** (`YAWGPU_HAS_TILED`): subpass pass
 *   layouts, transient attachments, input-attachment bind groups, and the
 *   `YaWGPUSubpassRenderPassEncoder` command stream.
 *
 * For all behavior not redefined here, yawgpu mirrors `webgpu.h`.
 */

#ifndef YAWGPU_H_
#define YAWGPU_H_

#include "webgpu.h"

/**
 * \defgroup InstanceBackendSelect Instance Backend Select
 * \brief Extension chain entry that forces a specific HAL backend at
 * instance creation.
 *
 * Chain a @ref YaWGPUInstanceBackendSelect onto `WGPUInstanceDescriptor.nextInChain`
 * before calling `wgpuCreateInstance` to pin the backend instead of letting
 * yawgpu auto-select. Useful for tests and platform-specific tooling.
 *
 * @{
 */

/**
 * `WGPUSType` tag identifying a @ref YaWGPUInstanceBackendSelect in a
 * chained-struct list.
 */
#define YAWGPU_STYPE_INSTANCE_BACKEND_SELECT ((WGPUSType)0x70000001u)

/**
 * Identifiers for the HAL backends a yawgpu instance can be pinned to via
 * @ref YaWGPUInstanceBackendSelect::backend.
 */
enum {
    /** Software / validation-only backend; always available, no GPU required. */
    YAWGPU_INSTANCE_BACKEND_NOOP = 0,
    /** Apple Metal backend. Available on macOS / iOS builds. */
    YAWGPU_INSTANCE_BACKEND_METAL = 1,
    /** Khronos Vulkan backend. Available on Linux / Windows / Android / MoltenVK builds. */
    YAWGPU_INSTANCE_BACKEND_VULKAN = 2,
    /** OpenGL ES 3.1+ backend (Tier 2 / experimental). Available on Android (native EGL)
     *  and Windows ANGLE. Requires the `gles` cargo feature to be compiled in;
     *  otherwise instance creation falls back to Noop. */
    YAWGPU_INSTANCE_BACKEND_GLES = 3,
};

/**
 * Chained extension that selects a specific HAL backend.
 *
 * `chain.sType` must be set to @ref YAWGPU_STYPE_INSTANCE_BACKEND_SELECT and
 * `backend` to one of the `YAWGPU_INSTANCE_BACKEND_*` constants. If the
 * requested backend is not compiled in or not available on the host, instance
 * creation falls back to the standard yawgpu selection policy.
 */
typedef struct YaWGPUInstanceBackendSelect {
    /** Chain header. `sType` must be @ref YAWGPU_STYPE_INSTANCE_BACKEND_SELECT. */
    WGPUChainedStruct chain;
    /** One of the `YAWGPU_INSTANCE_BACKEND_*` constants. */
    uint32_t backend;
} YaWGPUInstanceBackendSelect;

/** @} */

/**
 * \defgroup GlesContextBackend GLES Context Backend
 * \brief Extension chain entry that selects the GLES context backend (EGL vs
 * WGL) when the instance backend resolves to GLES.
 *
 * Chain a @ref YaWGPUGlesContextBackend onto the same
 * `WGPUInstanceDescriptor.nextInChain` list that may also contain a
 * @ref YaWGPUInstanceBackendSelect. This entry is additive: it only controls
 * the GLES context binding backend, while @ref YaWGPUInstanceBackendSelect
 * controls which HAL backend is requested.
 *
 * Resolution order is:
 * - A non-default @ref YaWGPUGlesContextBackend::contextBackend value wins.
 * - `YAWGPU_GLES_CONTEXT_BACKEND_DEFAULT` or no chain entry defers to the
 *   `YAWGPU_GLES_BACKEND` environment variable.
 * - If neither selects a backend, yawgpu uses EGL.
 *
 * `YAWGPU_GLES_CONTEXT_BACKEND_WGL` is Windows-only; on non-Windows hosts it
 * falls back to EGL. The entry is ignored when the resolved instance backend is
 * not GLES.
 *
 * Example two-entry chain:
 * \code{.c}
 * YaWGPUGlesContextBackend context = YAWGPU_GLES_CONTEXT_BACKEND_INIT;
 * context.contextBackend = YAWGPU_GLES_CONTEXT_BACKEND_EGL;
 *
 * YaWGPUInstanceBackendSelect backend = {
 *     { &context.chain, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT },
 *     YAWGPU_INSTANCE_BACKEND_GLES,
 * };
 *
 * WGPUInstanceDescriptor desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
 * desc.nextInChain = &backend.chain;
 * WGPUInstance instance = wgpuCreateInstance(&desc);
 * \endcode
 *
 * @{
 */

/**
 * `WGPUSType` tag identifying a @ref YaWGPUGlesContextBackend in a
 * chained-struct list.
 */
#define YAWGPU_STYPE_GLES_CONTEXT_BACKEND ((WGPUSType)0x70000002u)

/**
 * Identifiers for the GLES context backend selected via
 * @ref YaWGPUGlesContextBackend::contextBackend.
 */
enum {
    /** Defer to `YAWGPU_GLES_BACKEND`, then the default EGL backend. */
    YAWGPU_GLES_CONTEXT_BACKEND_DEFAULT = 0,
    /** Force EGL (`libEGL` / ANGLE on Windows, native EGL elsewhere). */
    YAWGPU_GLES_CONTEXT_BACKEND_EGL = 1,
    /** Force WGL on Windows; falls back to EGL on non-Windows hosts. */
    YAWGPU_GLES_CONTEXT_BACKEND_WGL = 2,
};

/**
 * Chained extension that selects the GLES context backend.
 *
 * `chain.sType` must be set to @ref YAWGPU_STYPE_GLES_CONTEXT_BACKEND and
 * `contextBackend` to one of the `YAWGPU_GLES_CONTEXT_BACKEND_*` constants.
 * Unknown values fall back to EGL, matching the environment-variable parser.
 */
typedef struct YaWGPUGlesContextBackend {
    /** Chain header. `sType` must be @ref YAWGPU_STYPE_GLES_CONTEXT_BACKEND. */
    WGPUChainedStruct chain;
    /** One of the `YAWGPU_GLES_CONTEXT_BACKEND_*` constants. */
    uint32_t contextBackend;
} YaWGPUGlesContextBackend;

/** Default initializer for @ref YaWGPUGlesContextBackend. */
#define YAWGPU_GLES_CONTEXT_BACKEND_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUGlesContextBackend, { \
    /*.chain=*/{NULL _wgpu_COMMA YAWGPU_STYPE_GLES_CONTEXT_BACKEND} _wgpu_COMMA \
    /*.contextBackend=*/YAWGPU_GLES_CONTEXT_BACKEND_DEFAULT _wgpu_COMMA \
})

/** @} */

#if defined(YAWGPU_HAS_SHADER_PASSTHROUGH)
/**
 * \defgroup ShaderPassthrough Shader passthrough (SPIR-V / MSL)
 * \brief Vendor entry points that bypass yawgpu's WGSL pipeline and consume
 * pre-compiled backend-native shader modules.
 *
 * Use these when:
 * - You ship a build-time SPIR-V toolchain (`glslangValidator`, `shaderc`,
 *   `naga`-cli) and want to skip the WGSL-to-SPIR-V step at runtime.
 * - You hand-author Metal Shading Language and need control over Metal-only
 *   features (function constants, argument buffers in a custom layout, etc.).
 * - You want a backend-specific reference path for performance comparison
 *   against the standard `wgpuDeviceCreateShaderModule` (WGSL) path.
 *
 * The returned `WGPUShaderModule` is interchangeable with one created via
 * the standard entry point: it can be referenced from a `WGPUProgrammableStage`
 * inside any pipeline descriptor.
 *
 * @{
 */

/**
 * Descriptor for a SPIR-V shader module passed verbatim to the Vulkan
 * backend.
 *
 * The Vulkan backend uploads `code[0..codeSize]` directly to
 * `vkCreateShaderModule`. The Metal backend rejects this descriptor — use
 * @ref YaWGPUShaderModuleMslDescriptor instead.
 *
 * Default values can be set using @ref YAWGPU_SHADER_MODULE_SPIRV_DESCRIPTOR_INIT
 * as initializer.
 */
typedef struct YaWGPUShaderModuleSpirVDescriptor {
    /** Chain pointer, currently unused. The `INIT` macro sets this to `NULL`. */
    WGPUChainedStruct const* nextInChain;
    /** Optional debug label. The `INIT` macro sets this to @ref WGPU_STRING_VIEW_INIT. */
    WGPUStringView label;
    /**
     * Number of 32-bit SPIR-V words in @ref code. Must match the byte length
     * of the SPIR-V blob divided by four.
     *
     * The `INIT` macro sets this to `0`.
     */
    uint32_t codeSize;
    /**
     * Pointer to the SPIR-V word stream. The pointer is only read during the
     * call to @ref yawgpuDeviceCreateShaderModuleSpirV; ownership is not
     * transferred. The caller must ensure the buffer is 4-byte aligned.
     *
     * The `INIT` macro sets this to `NULL`.
     */
    uint32_t const* code;
} YaWGPUShaderModuleSpirVDescriptor;

/**
 * Initializer for @ref YaWGPUShaderModuleSpirVDescriptor.
 */
#define YAWGPU_SHADER_MODULE_SPIRV_DESCRIPTOR_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUShaderModuleSpirVDescriptor, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.label=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.codeSize=*/0 _wgpu_COMMA \
    /*.code=*/NULL _wgpu_COMMA \
})

/**
 * One MSL entry point declared by a @ref YaWGPUShaderModuleMslDescriptor.
 *
 * A single MSL source can host multiple entry points (e.g. one vertex + one
 * fragment function), so the descriptor takes an array of these.
 *
 * Default values can be set using @ref YAWGPU_MSL_ENTRY_POINT_INIT as
 * initializer.
 */
typedef struct YaWGPUMslEntryPoint {
    /**
     * The MSL function name as it appears in the source (no Metal name
     * mangling). The `INIT` macro sets this to @ref WGPU_STRING_VIEW_INIT.
     */
    WGPUStringView name;
    /**
     * Single-bit shader stage selector. Exactly one of
     * `WGPUShaderStage_Vertex`, `WGPUShaderStage_Fragment`, or
     * `WGPUShaderStage_Compute` must be set.
     *
     * The `INIT` macro sets this to `WGPUShaderStage_None`.
     */
    WGPUShaderStage stage;
    /**
     * Compute workgroup size `(x, y, z)`. Ignored for vertex/fragment entry
     * points; required for compute. The `INIT` macro sets this to
     * `{0, 0, 0}`.
     */
    uint32_t workgroupSize[3];
} YaWGPUMslEntryPoint;

/**
 * Initializer for @ref YaWGPUMslEntryPoint.
 */
#define YAWGPU_MSL_ENTRY_POINT_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUMslEntryPoint, { \
    /*.name=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.stage=*/WGPUShaderStage_None _wgpu_COMMA \
    /*.workgroupSize=*/{0 _wgpu_COMMA 0 _wgpu_COMMA 0} _wgpu_COMMA \
})

/**
 * Descriptor for an MSL shader module passed verbatim to the Metal backend.
 *
 * `code` carries the MSL source text and `entryPoints` declares which
 * functions inside that source yawgpu may bind into pipeline stages. The
 * Vulkan backend rejects this descriptor — use
 * @ref YaWGPUShaderModuleSpirVDescriptor instead.
 *
 * Because yawgpu does not reflect MSL, the caller must arrange Metal
 * resource indices to match what yawgpu derives from the explicit pipeline
 * layout — see the "Metal binding-index mapping for MSL passthrough"
 * comment block below.
 *
 * Default values can be set using @ref YAWGPU_SHADER_MODULE_MSL_DESCRIPTOR_INIT
 * as initializer.
 */
typedef struct YaWGPUShaderModuleMslDescriptor {
    /** Chain pointer, currently unused. The `INIT` macro sets this to `NULL`. */
    WGPUChainedStruct const* nextInChain;
    /** Optional debug label. The `INIT` macro sets this to @ref WGPU_STRING_VIEW_INIT. */
    WGPUStringView label;
    /**
     * MSL source text. UTF-8, NUL-terminator not required. The pointer is
     * only read during the call. The `INIT` macro sets this to
     * @ref WGPU_STRING_VIEW_INIT.
     */
    WGPUStringView code;
    /** Number of elements in @ref entryPoints. The `INIT` macro sets this to `0`. */
    size_t entryPointCount;
    /**
     * Array of @ref YaWGPUMslEntryPoint describing the functions yawgpu may
     * bind out of @ref code. At least one entry point is required.
     *
     * The `INIT` macro sets this to `NULL`.
     */
    YaWGPUMslEntryPoint const* entryPoints;
} YaWGPUShaderModuleMslDescriptor;

/**
 * Initializer for @ref YaWGPUShaderModuleMslDescriptor.
 */
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

/**
 * Creates a `WGPUShaderModule` from a pre-compiled SPIR-V binary.
 *
 * Only the Vulkan backend implements this entry point. On the Metal or Noop
 * backends, the call routes an error to the device error sink and returns
 * `NULL`.
 *
 * @param device
 * The device that owns the resulting shader module.
 *
 * @param descriptor
 * Must be non-`NULL`. The pointed-to descriptor is only read during the
 * call; the SPIR-V bytes referenced by `descriptor->code` are copied
 * into the backend driver.
 *
 * @returns
 * A new `WGPUShaderModule` on success, or `NULL` if creation failed; in the
 * `NULL` case an error has been pushed to the device's error scope.
 */
WGPUShaderModule yawgpuDeviceCreateShaderModuleSpirV(
    WGPUDevice device,
    YaWGPUShaderModuleSpirVDescriptor const* descriptor);

/**
 * Creates a `WGPUShaderModule` from Metal Shading Language source.
 *
 * Only the Metal backend implements this entry point. On the Vulkan or Noop
 * backends, the call routes an error to the device error sink and returns
 * `NULL`.
 *
 * @param device
 * The device that owns the resulting shader module.
 *
 * @param descriptor
 * Must be non-`NULL`. The pointed-to descriptor and the MSL source it
 * references are only read during the call; the source text is compiled
 * into a `MTLLibrary` retained by the returned module.
 *
 * @returns
 * A new `WGPUShaderModule` on success, or `NULL` if compilation failed; in
 * the `NULL` case an error (including any MSL compiler diagnostics) has
 * been pushed to the device's error scope.
 */
WGPUShaderModule yawgpuDeviceCreateShaderModuleMsl(
    WGPUDevice device,
    YaWGPUShaderModuleMslDescriptor const* descriptor);

/** @} */
#endif

#if defined(YAWGPU_HAS_TILED)
/**
 * \defgroup Tiled Tiled / multi-subpass rendering
 * \brief Subpass pass layouts, transient attachments, and
 * input-attachment bind groups for tile-based deferred rendering on
 * Vulkan (`VK_KHR_create_renderpass2`) and Metal (programmable blending
 * / tile shading).
 *
 * Workflow:
 *   1. Query @ref yawgpuAdapterGetTiledCapabilities to discover backend
 *      limits.
 *   2. Build a @ref YaWGPUSubpassPassLayout describing the attachment
 *      formats, the per-subpass color/input attachment lists, and the
 *      subpass dependency graph.
 *   3. Create transient attachments (@ref yawgpuDeviceCreateTransientAttachment)
 *      for any G-buffer-like images that never need to leave tile memory.
 *   4. Create one or more subpass render pipelines
 *      (@ref yawgpuDeviceCreateSubpassRenderPipeline) bound to the layout.
 *   5. Begin a subpass render pass with
 *      @ref yawgpuCommandEncoderBeginSubpassRenderPass, encode draws with
 *      the @ref YaWGPUSubpassRenderPassEncoder API, advance subpasses with
 *      @ref yawgpuSubpassRenderPassEncoderNextSubpass, and end with
 *      @ref yawgpuSubpassRenderPassEncoderEnd.
 *
 * @{
 */

/**
 * `WGPUFeatureName` for multi-subpass render passes via
 * @ref YaWGPUSubpassPassLayout. Adapters that lack tile-based primitives
 * (most desktop discrete GPUs on certain drivers) may still expose this
 * with reduced limits, but the entry points work uniformly.
 */
#define YaWGPUFeatureName_MultiSubpass ((WGPUFeatureName)0x70010001u)
/**
 * `WGPUFeatureName` for transient (tile-memory-only) attachments via
 * @ref yawgpuDeviceCreateTransientAttachment.
 */
#define YaWGPUFeatureName_TransientAttachments ((WGPUFeatureName)0x70010002u)
/**
 * `WGPUFeatureName` for framebuffer-fetch / subpass-input shader access
 * (Vulkan input attachments / Metal `[[color(N)]]` in-tile reads).
 */
#define YaWGPUFeatureName_ShaderFramebufferFetch ((WGPUFeatureName)0x70010003u)

/**
 * Backend-reported tiled-rendering capabilities for an adapter.
 *
 * Fields are 0 when the backend does not support tiled rendering at all;
 * otherwise they reflect the smaller of the API limit and any
 * yawgpu-imposed maximum.
 *
 * Default values can be set using @ref YAWGPU_TILED_CAPABILITIES_INIT as
 * initializer.
 */
typedef struct YaWGPUTiledCapabilities {
    /** Chain pointer, currently unused. The `INIT` macro sets this to `NULL`. */
    WGPUChainedStruct const* nextInChain;
    /** Maximum subpasses allowed in a single @ref YaWGPUSubpassPassLayout. */
    uint32_t maxSubpasses;
    /** Maximum color attachments any single subpass may write. */
    uint32_t maxSubpassColorAttachments;
    /** Maximum input attachments any single subpass may read. */
    uint32_t maxInputAttachments;
    /**
     * Implementation-estimated tile-memory budget in bytes. Informational
     * only; large G-buffer layouts that exceed this may force the backend
     * out of tile mode but will still execute correctly.
     */
    uint32_t estimatedTileMemoryBytes;
} YaWGPUTiledCapabilities;

/**
 * Initializer for @ref YaWGPUTiledCapabilities.
 */
#define YAWGPU_TILED_CAPABILITIES_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUTiledCapabilities, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.maxSubpasses=*/0 _wgpu_COMMA \
    /*.maxSubpassColorAttachments=*/0 _wgpu_COMMA \
    /*.maxInputAttachments=*/0 _wgpu_COMMA \
    /*.estimatedTileMemoryBytes=*/0 _wgpu_COMMA \
})

/**
 * Populates @p capabilities with the adapter's tiled-rendering limits.
 *
 * @param adapter
 * The adapter to query.
 *
 * @param capabilities
 * Must be non-`NULL`. All fields are overwritten on success.
 *
 * @returns
 * `WGPUStatus_Success` on success; `WGPUStatus_Error` if the adapter does
 * not expose any of the `YaWGPUFeatureName_*` tiled features.
 */
WGPUStatus yawgpuAdapterGetTiledCapabilities(
    WGPUAdapter adapter,
    YaWGPUTiledCapabilities* capabilities);

/**
 * Opaque handle to a transient attachment (an image that may live entirely
 * in tile memory). Created via @ref yawgpuDeviceCreateTransientAttachment,
 * destroyed via @ref yawgpuTransientAttachmentRelease.
 */
typedef struct YaWGPUTransientAttachmentImpl* YaWGPUTransientAttachment;

/**
 * How a transient attachment derives its width/height. `MatchTarget` defers
 * sizing until the attachment is used inside a subpass render pass and the
 * pass extent is known.
 */
typedef enum YaWGPUTransientSizeMode {
    /** Inherit width/height from the pass extent at begin-pass time. */
    YaWGPUTransientSizeMode_MatchTarget = 0x00000000,
    /** Use the explicit `width`/`height` from the descriptor. */
    YaWGPUTransientSizeMode_Explicit = 0x00000001,
    /** Internal: forces the enum to 32 bits. */
    YaWGPUTransientSizeMode_Force32 = 0x7FFFFFFF
} YaWGPUTransientSizeMode;

/**
 * Descriptor for a transient attachment.
 *
 * A transient attachment has tile-memory-only semantics: backends may
 * choose to back it by lazily-allocated memory (Vulkan
 * `VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`) or a Metal memoryless
 * `MTLStorageMode::Memoryless` texture. Its contents are guaranteed only
 * for the lifetime of a single subpass render pass.
 *
 * Default values can be set using @ref YAWGPU_TRANSIENT_ATTACHMENT_DESCRIPTOR_INIT
 * as initializer.
 */
typedef struct YaWGPUTransientAttachmentDescriptor {
    /** Chain pointer, currently unused. The `INIT` macro sets this to `NULL`. */
    WGPUChainedStruct const* nextInChain;
    /** Optional debug label. */
    WGPUStringView label;
    /** Pixel format. Must match the format declared in the pass layout slot. */
    WGPUTextureFormat format;
    /** How width/height are determined; see @ref YaWGPUTransientSizeMode. */
    YaWGPUTransientSizeMode sizeMode;
    /** Width in pixels; only consulted when @ref sizeMode is `Explicit`. */
    uint32_t width;
    /** Height in pixels; only consulted when @ref sizeMode is `Explicit`. */
    uint32_t height;
    /** Sample count. `1` is the only widely supported value today. */
    uint32_t sampleCount;
} YaWGPUTransientAttachmentDescriptor;

/**
 * Initializer for @ref YaWGPUTransientAttachmentDescriptor.
 */
#define YAWGPU_TRANSIENT_ATTACHMENT_DESCRIPTOR_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUTransientAttachmentDescriptor, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.label=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.format=*/WGPUTextureFormat_Undefined _wgpu_COMMA \
    /*.sizeMode=*/YaWGPUTransientSizeMode_MatchTarget _wgpu_COMMA \
    /*.width=*/0 _wgpu_COMMA \
    /*.height=*/0 _wgpu_COMMA \
    /*.sampleCount=*/1 _wgpu_COMMA \
})

/**
 * Creates a transient attachment on @p device.
 *
 * @param device
 * The device that owns the attachment.
 *
 * @param descriptor
 * Must be non-`NULL`.
 *
 * @returns
 * A new transient attachment handle on success, or `NULL` on failure (with
 * an error pushed to the device's error scope).
 */
YaWGPUTransientAttachment yawgpuDeviceCreateTransientAttachment(
    WGPUDevice device,
    YaWGPUTransientAttachmentDescriptor const* descriptor);

/** Increments the refcount of a transient attachment. */
void yawgpuTransientAttachmentAddRef(YaWGPUTransientAttachment attachment);
/**
 * Decrements the refcount of a transient attachment. When the final
 * reference is released the backing tile-memory image is freed.
 */
void yawgpuTransientAttachmentRelease(YaWGPUTransientAttachment attachment);

/**
 * `WGPUSType` tag identifying a @ref YaWGPUInputAttachmentBindingLayout in
 * a bind-group-layout entry's chained-struct list.
 */
#define YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT ((WGPUSType)0x70000010u)

/**
 * Chained extension that promotes a `WGPUBindGroupLayoutEntry` into an
 * input-attachment binding (Vulkan `VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT` /
 * Metal subpass color read).
 *
 * The entry's `texture` field is otherwise ignored; the binding's source
 * attachment is taken from the subpass layout when a bind group is created
 * for this layout.
 *
 * Default values can be set using @ref YAWGPU_INPUT_ATTACHMENT_BINDING_LAYOUT_INIT
 * as initializer.
 */
typedef struct YaWGPUInputAttachmentBindingLayout {
    /** Chain header. `sType` must be @ref YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT. */
    WGPUChainedStruct chain;
    /** Texture sample type matching the producing attachment's format. */
    WGPUTextureSampleType sampleType;
    /** Non-zero when the source attachment is multisampled. */
    WGPUBool multisampled;
} YaWGPUInputAttachmentBindingLayout;

/**
 * Initializer for @ref YaWGPUInputAttachmentBindingLayout.
 */
#define YAWGPU_INPUT_ATTACHMENT_BINDING_LAYOUT_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUInputAttachmentBindingLayout, { \
    /*.chain=*/{NULL _wgpu_COMMA YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT} _wgpu_COMMA \
    /*.sampleType=*/WGPUTextureSampleType_Float _wgpu_COMMA \
    /*.multisampled=*/0 _wgpu_COMMA \
})

/*
 * Input attachment resources are auto-wired from the subpass pass layout:
 * a `WGPUBindGroupEntry` whose layout entry carries
 * @ref YaWGPUInputAttachmentBindingLayout looks up its source view from
 * the subpass's input-attachment table rather than from the bind-group
 * entry's `textureView` field, which is ignored.
 */

/**
 * Opaque handle to a subpass pass layout — the static description of an
 * entire multi-subpass render pass (attachment formats, per-subpass color
 * and input attachment lists, and subpass dependencies). Created via
 * @ref yawgpuDeviceCreateSubpassPassLayout.
 */
typedef struct YaWGPUSubpassPassLayoutImpl* YaWGPUSubpassPassLayout;
/**
 * Opaque handle to an in-flight subpass render pass. Created via
 * @ref yawgpuCommandEncoderBeginSubpassRenderPass and ended via
 * @ref yawgpuSubpassRenderPassEncoderEnd.
 */
typedef struct YaWGPUSubpassRenderPassEncoderImpl* YaWGPUSubpassRenderPassEncoder;

/**
 * One attachment slot in a subpass pass layout.
 *
 * Used for both color and depth-stencil slots; depth-stencil slots are
 * additionally selected via @ref YaWGPUSubpassPassLayoutDescriptor::depthStencilAttachment.
 *
 * Default values can be set using @ref YAWGPU_ATTACHMENT_LAYOUT_INIT as
 * initializer.
 */
typedef struct YaWGPUAttachmentLayout {
    /** Attachment pixel format. */
    WGPUTextureFormat format;
    /** Sample count. `1` for single-sample, `>1` for MSAA. */
    uint32_t sampleCount;
} YaWGPUAttachmentLayout;

/**
 * Initializer for @ref YaWGPUAttachmentLayout.
 */
#define YAWGPU_ATTACHMENT_LAYOUT_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUAttachmentLayout, { \
    /*.format=*/WGPUTextureFormat_Undefined _wgpu_COMMA \
    /*.sampleCount=*/1 _wgpu_COMMA \
})

/**
 * Kind of memory/synchronization dependency between two subpasses.
 *
 * Maps to a Vulkan `VkSubpassDependency` with appropriate src/dst
 * access masks; Metal collapses this into the natural per-attachment
 * tile-memory ordering.
 */
typedef enum YaWGPUSubpassDependencyType {
    /** Color attachment write in `srcSubpass` is read as input in `dstSubpass`. */
    YaWGPUSubpassDependencyType_ColorToInput = 0x00000000,
    /** Depth-stencil write in `srcSubpass` is read as input in `dstSubpass`. */
    YaWGPUSubpassDependencyType_DepthToInput = 0x00000001,
    /** Both color and depth-stencil writes in `srcSubpass` are read as input in `dstSubpass`. */
    YaWGPUSubpassDependencyType_ColorDepthToInput = 0x00000002,
    /** Internal: forces the enum to 32 bits. */
    YaWGPUSubpassDependencyType_Force32 = 0x7FFFFFFF
} YaWGPUSubpassDependencyType;

/**
 * One subpass-to-subpass synchronization edge in the pass dependency
 * graph.
 *
 * Provide one entry per producer→consumer pair the shader actually
 * reads; the backend uses these to insert pipeline barriers / tile
 * fences.
 */
typedef struct YaWGPUSubpassDependency {
    /** Index of the producing subpass. */
    uint32_t srcSubpass;
    /** Index of the consuming subpass. */
    uint32_t dstSubpass;
    /** Which attachment category crosses the boundary. */
    YaWGPUSubpassDependencyType dependencyType;
    /**
     * Non-zero when the dependency is by-region (Vulkan
     * `VK_DEPENDENCY_BY_REGION_BIT`). Required for input-attachment reads
     * in a tile-based pass.
     */
    WGPUBool byRegion;
} YaWGPUSubpassDependency;

/**
 * Sentinel `sourceAttachment` value indicating that an input attachment
 * is sourced from the pass's depth-stencil slot, not a color slot.
 */
#define YAWGPU_DEPTH_STENCIL_ATTACHMENT_INDEX 0xFFFFFFFFu

/**
 * One input-attachment binding declared by a subpass.
 *
 * `(group, binding)` names the slot in the consuming subpass's bind group
 * layout (the entry must carry @ref YaWGPUInputAttachmentBindingLayout).
 * `(sourceSubpass, sourceAttachment)` names the producing subpass and the
 * pass-level color attachment index it wrote to (or
 * @ref YAWGPU_DEPTH_STENCIL_ATTACHMENT_INDEX for the depth-stencil slot).
 */
typedef struct YaWGPUSubpassInputAttachment {
    /** Bind group index where the input attachment is bound. */
    uint32_t group;
    /** Binding number inside the bind group. */
    uint32_t binding;
    /** Subpass that produced the attachment being read. */
    uint32_t sourceSubpass;
    /**
     * Pass-level color attachment index, or
     * @ref YAWGPU_DEPTH_STENCIL_ATTACHMENT_INDEX for depth-stencil.
     */
    uint32_t sourceAttachment;
} YaWGPUSubpassInputAttachment;

/**
 * Static description of a single subpass inside a
 * @ref YaWGPUSubpassPassLayout.
 *
 * Default values can be set using @ref YAWGPU_SUBPASS_LAYOUT_DESC_INIT as
 * initializer.
 */
typedef struct YaWGPUSubpassLayoutDesc {
    /**
     * Indices into @ref YaWGPUSubpassPassLayoutDescriptor::colorAttachments
     * naming the color attachments this subpass writes. Each index appears
     * at most once.
     */
    uint32_t const* colorAttachmentIndices;
    /** Number of elements in @ref colorAttachmentIndices. */
    size_t colorAttachmentIndexCount;
    /** Non-zero when this subpass reads or writes the pass depth-stencil slot. */
    WGPUBool usesDepthStencil;
    /**
     * Input-attachment bindings this subpass reads. Entries reference
     * attachments produced by earlier subpasses in the same pass.
     */
    YaWGPUSubpassInputAttachment const* inputAttachments;
    /** Number of elements in @ref inputAttachments. */
    size_t inputAttachmentCount;
} YaWGPUSubpassLayoutDesc;

/**
 * Initializer for @ref YaWGPUSubpassLayoutDesc.
 */
#define YAWGPU_SUBPASS_LAYOUT_DESC_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUSubpassLayoutDesc, { \
    /*.colorAttachmentIndices=*/NULL _wgpu_COMMA \
    /*.colorAttachmentIndexCount=*/0 _wgpu_COMMA \
    /*.usesDepthStencil=*/0 _wgpu_COMMA \
    /*.inputAttachments=*/NULL _wgpu_COMMA \
    /*.inputAttachmentCount=*/0 _wgpu_COMMA \
})

/**
 * Full descriptor for a @ref YaWGPUSubpassPassLayout — the static plan
 * for an entire multi-subpass render pass.
 *
 * The same layout is shared by every render pass instance and every
 * subpass pipeline that participates in the pass. At begin-pass time the
 * caller binds concrete textures / transient attachments matching the
 * formats declared here.
 *
 * Default values can be set using @ref YAWGPU_SUBPASS_PASS_LAYOUT_DESCRIPTOR_INIT
 * as initializer.
 */
typedef struct YaWGPUSubpassPassLayoutDescriptor {
    /** Chain pointer, currently unused. */
    WGPUChainedStruct const* nextInChain;
    /** Optional debug label. */
    WGPUStringView label;
    /**
     * Pass-level color attachment table. Each subpass references a subset
     * of these via @ref YaWGPUSubpassLayoutDesc::colorAttachmentIndices.
     */
    YaWGPUAttachmentLayout const* colorAttachments;
    /** Number of elements in @ref colorAttachments. */
    size_t colorAttachmentCount;
    /**
     * Optional depth-stencil slot. `format == WGPUTextureFormat_Undefined`
     * means the pass has no depth-stencil attachment.
     */
    YaWGPUAttachmentLayout depthStencilAttachment;
    /** Per-subpass static descriptions. The array length is the subpass count. */
    YaWGPUSubpassLayoutDesc const* subpasses;
    /** Number of elements in @ref subpasses. Must be `≥ 1`. */
    size_t subpassCount;
    /**
     * Producer→consumer synchronization edges. May be empty when the
     * dependency graph is implicit (e.g. linear chain of subpasses
     * reading each predecessor's color outputs by region).
     */
    YaWGPUSubpassDependency const* dependencies;
    /** Number of elements in @ref dependencies. */
    size_t dependencyCount;
} YaWGPUSubpassPassLayoutDescriptor;

/**
 * Initializer for @ref YaWGPUSubpassPassLayoutDescriptor.
 */
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

/**
 * Creates a subpass pass layout on @p device.
 *
 * The layout is immutable after creation and can be shared between many
 * pipelines and many begin-pass calls.
 *
 * @param device
 * The device that owns the layout.
 *
 * @param descriptor
 * Must be non-`NULL` and well-formed (subpass indices in `dependencies`
 * and `inputAttachments` must reference declared subpasses /
 * attachments).
 *
 * @returns
 * A new layout handle on success, or `NULL` on failure (with an error
 * pushed to the device's error scope).
 */
YaWGPUSubpassPassLayout yawgpuDeviceCreateSubpassPassLayout(
    WGPUDevice device,
    YaWGPUSubpassPassLayoutDescriptor const* descriptor);
/** Increments the refcount of a subpass pass layout. */
void yawgpuSubpassPassLayoutAddRef(YaWGPUSubpassPassLayout layout);
/** Decrements the refcount of a subpass pass layout. */
void yawgpuSubpassPassLayoutRelease(YaWGPUSubpassPassLayout layout);

/**
 * Descriptor for a render pipeline bound to a specific subpass of a
 * @ref YaWGPUSubpassPassLayout.
 *
 * `base` is the standard WGPU render pipeline descriptor; the subpass
 * extension forces the pipeline to match the subpass's color/input
 * attachment layout and depth-stencil presence. Pipelines created with
 * this descriptor are usable only inside a subpass render pass whose
 * layout matches @ref passLayout and whose current subpass index matches
 * @ref subpassIndex.
 *
 * Default values can be set using @ref YAWGPU_SUBPASS_RENDER_PIPELINE_DESCRIPTOR_INIT
 * as initializer.
 */
typedef struct YaWGPUSubpassRenderPipelineDescriptor {
    /** Chain pointer, currently unused. */
    WGPUChainedStruct const* nextInChain;
    /**
     * Standard WGPU render pipeline descriptor. The descriptor's
     * `fragment.targets` and `depthStencil` slots must be consistent with
     * the subpass's color attachments and depth-stencil use — the backend
     * will derive its pipeline color-target layout from @ref passLayout,
     * not from `base.fragment.targets`.
     */
    WGPURenderPipelineDescriptor base;
    /** The pass layout this pipeline participates in. Must be non-`NULL`. */
    YaWGPUSubpassPassLayout passLayout;
    /** Zero-based index into the pass layout's subpass array. */
    uint32_t subpassIndex;
} YaWGPUSubpassRenderPipelineDescriptor;

/**
 * Initializer for @ref YaWGPUSubpassRenderPipelineDescriptor.
 */
#define YAWGPU_SUBPASS_RENDER_PIPELINE_DESCRIPTOR_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUSubpassRenderPipelineDescriptor, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.base=*/WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT _wgpu_COMMA \
    /*.passLayout=*/NULL _wgpu_COMMA \
    /*.subpassIndex=*/0 _wgpu_COMMA \
})

/**
 * Creates a render pipeline targeting a single subpass.
 *
 * @param device
 * The device that owns the pipeline.
 *
 * @param descriptor
 * Must be non-`NULL`.
 *
 * @returns
 * A new `WGPURenderPipeline` on success, or `NULL` on failure (with an
 * error pushed to the device's error scope).
 */
WGPURenderPipeline yawgpuDeviceCreateSubpassRenderPipeline(
    WGPUDevice device,
    YaWGPUSubpassRenderPipelineDescriptor const* descriptor);

/**
 * Whether an attachment slot is backed by a persistent texture or by a
 * @ref YaWGPUTransientAttachment.
 */
typedef enum YaWGPUSubpassAttachmentKind {
    /** Backed by a `WGPUTextureView` (`view`). */
    YaWGPUSubpassAttachmentKind_Persistent = 0x00000000,
    /** Backed by a @ref YaWGPUTransientAttachment (`transient`). */
    YaWGPUSubpassAttachmentKind_Transient = 0x00000001,
    /** Internal: forces the enum to 32 bits. */
    YaWGPUSubpassAttachmentKind_Force32 = 0x7FFFFFFF
} YaWGPUSubpassAttachmentKind;

/**
 * Concrete binding for one color attachment slot at begin-pass time.
 *
 * Exactly one of `view` (when @ref kind is `Persistent`) or `transient`
 * (when @ref kind is `Transient`) must be set. `resolveTarget`, when
 * non-`NULL`, declares an MSAA resolve target consistent with the slot's
 * sample count.
 *
 * Default values can be set using @ref YAWGPU_COLOR_ATTACHMENT_BINDING_INIT
 * as initializer.
 */
typedef struct YaWGPUColorAttachmentBinding {
    /** Selects which of `view` or `transient` carries the binding. */
    YaWGPUSubpassAttachmentKind kind;
    /** Persistent texture view. Used when @ref kind is `Persistent`. */
    WGPUTextureView view;
    /** Optional MSAA resolve target. May be `NULL`. */
    WGPUTextureView resolveTarget;
    /** Transient attachment. Used when @ref kind is `Transient`. */
    YaWGPUTransientAttachment transient;
    /** Load op applied at the start of the first subpass that writes this slot. */
    WGPULoadOp loadOp;
    /** Store op applied at the end of the last subpass that writes this slot. */
    WGPUStoreOp storeOp;
    /** Clear color, used only when @ref loadOp is `WGPULoadOp_Clear`. */
    WGPUColor clearValue;
} YaWGPUColorAttachmentBinding;

/**
 * Initializer for @ref YaWGPUColorAttachmentBinding.
 */
#define YAWGPU_COLOR_ATTACHMENT_BINDING_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUColorAttachmentBinding, { \
    /*.kind=*/YaWGPUSubpassAttachmentKind_Persistent _wgpu_COMMA \
    /*.view=*/NULL _wgpu_COMMA \
    /*.resolveTarget=*/NULL _wgpu_COMMA \
    /*.transient=*/NULL _wgpu_COMMA \
    /*.loadOp=*/WGPULoadOp_Load _wgpu_COMMA \
    /*.storeOp=*/WGPUStoreOp_Store _wgpu_COMMA \
    /*.clearValue=*/{0, 0, 0, 0} _wgpu_COMMA \
})

/**
 * Concrete binding for the pass's depth-stencil slot at begin-pass time.
 *
 * Exactly one of `view` or `transient` must be set, matching @ref kind.
 * Stencil fields are ignored when the bound format has no stencil aspect.
 *
 * Default values can be set using @ref YAWGPU_DEPTH_STENCIL_ATTACHMENT_BINDING_INIT
 * as initializer.
 */
typedef struct YaWGPUDepthStencilAttachmentBinding {
    /** Selects which of `view` or `transient` carries the binding. */
    YaWGPUSubpassAttachmentKind kind;
    /** Persistent texture view. */
    WGPUTextureView view;
    /** Transient attachment. */
    YaWGPUTransientAttachment transient;
    /** Load op for the depth aspect. */
    WGPULoadOp depthLoadOp;
    /** Store op for the depth aspect. */
    WGPUStoreOp depthStoreOp;
    /** Clear depth value, used only when @ref depthLoadOp is `WGPULoadOp_Clear`. */
    float depthClearValue;
    /** Load op for the stencil aspect (ignored for depth-only formats). */
    WGPULoadOp stencilLoadOp;
    /** Store op for the stencil aspect (ignored for depth-only formats). */
    WGPUStoreOp stencilStoreOp;
    /** Clear stencil value, used only when @ref stencilLoadOp is `WGPULoadOp_Clear`. */
    uint32_t stencilClearValue;
} YaWGPUDepthStencilAttachmentBinding;

/**
 * Initializer for @ref YaWGPUDepthStencilAttachmentBinding.
 */
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

/**
 * Descriptor passed to @ref yawgpuCommandEncoderBeginSubpassRenderPass.
 *
 * The descriptor binds concrete attachments to the pass-level slots
 * declared in @ref passLayout. The number of `colorAttachments` entries
 * must equal the layout's color attachment count, and
 * `depthStencilAttachment` must be non-`NULL` iff the layout declares a
 * depth-stencil slot.
 *
 * Default values can be set using @ref YAWGPU_SUBPASS_RENDER_PASS_DESCRIPTOR_INIT
 * as initializer.
 */
typedef struct YaWGPUSubpassRenderPassDescriptor {
    /** Chain pointer, currently unused. */
    WGPUChainedStruct const* nextInChain;
    /** Optional debug label. */
    WGPUStringView label;
    /** The pass layout describing the pass topology. Must be non-`NULL`. */
    YaWGPUSubpassPassLayout passLayout;
    /**
     * Render area in pixels. `depthOrArrayLayers` is informational and
     * typically `1`.
     */
    WGPUExtent3D extent;
    /** Per-slot color attachment bindings, in pass-level slot order. */
    YaWGPUColorAttachmentBinding const* colorAttachments;
    /** Number of elements in @ref colorAttachments. */
    size_t colorAttachmentCount;
    /**
     * Depth-stencil binding, or `NULL` when the pass layout declares no
     * depth-stencil slot.
     */
    YaWGPUDepthStencilAttachmentBinding const* depthStencilAttachment;
} YaWGPUSubpassRenderPassDescriptor;

/**
 * Initializer for @ref YaWGPUSubpassRenderPassDescriptor.
 */
#define YAWGPU_SUBPASS_RENDER_PASS_DESCRIPTOR_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUSubpassRenderPassDescriptor, { \
    /*.nextInChain=*/NULL _wgpu_COMMA \
    /*.label=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.passLayout=*/NULL _wgpu_COMMA \
    /*.extent=*/{0, 0, 1} _wgpu_COMMA \
    /*.colorAttachments=*/NULL _wgpu_COMMA \
    /*.colorAttachmentCount=*/0 _wgpu_COMMA \
    /*.depthStencilAttachment=*/NULL _wgpu_COMMA \
})

/**
 * Begins a subpass render pass on the given command encoder.
 *
 * The returned encoder is positioned at subpass `0`. After recording draws
 * for subpass `i`, call @ref yawgpuSubpassRenderPassEncoderNextSubpass to
 * advance to subpass `i+1`. After the last subpass, call
 * @ref yawgpuSubpassRenderPassEncoderEnd to finalize the pass.
 *
 * @param encoder
 * The command encoder recording the pass.
 *
 * @param descriptor
 * Must be non-`NULL`.
 *
 * @returns
 * A new subpass render pass encoder on success, or `NULL` on failure
 * (with an error pushed to the device's error scope).
 */
YaWGPUSubpassRenderPassEncoder yawgpuCommandEncoderBeginSubpassRenderPass(
    WGPUCommandEncoder encoder,
    YaWGPUSubpassRenderPassDescriptor const* descriptor);
/**
 * Advances the encoder to the next subpass of the pass layout. Must be
 * called exactly `subpassCount - 1` times between begin and end.
 */
void yawgpuSubpassRenderPassEncoderNextSubpass(YaWGPUSubpassRenderPassEncoder encoder);
/**
 * Ends the subpass render pass. After this call the encoder still needs
 * to be released via @ref yawgpuSubpassRenderPassEncoderRelease.
 */
void yawgpuSubpassRenderPassEncoderEnd(YaWGPUSubpassRenderPassEncoder encoder);
/**
 * Binds a pipeline for subsequent draws in the current subpass. The
 * pipeline must have been created against the same pass layout and
 * subpass index as the current subpass.
 */
void yawgpuSubpassRenderPassEncoderSetPipeline(YaWGPUSubpassRenderPassEncoder encoder, WGPURenderPipeline pipeline);
/**
 * Binds a bind group at slot @p groupIndex for subsequent draws.
 *
 * @param dynamicOffsetCount
 * Number of entries in @p dynamicOffsets.
 *
 * @param dynamicOffsets
 * Byte offsets for any dynamic-offset bindings in the bind group,
 * in binding-number order.
 */
void yawgpuSubpassRenderPassEncoderSetBindGroup(YaWGPUSubpassRenderPassEncoder encoder, uint32_t groupIndex, WGPUBindGroup group, size_t dynamicOffsetCount, uint32_t const* dynamicOffsets);
/**
 * Binds a vertex buffer at slot @p slot for subsequent draws.
 *
 * @param size
 * Range size in bytes. `WGPU_WHOLE_SIZE` means `buffer.size - offset`.
 */
void yawgpuSubpassRenderPassEncoderSetVertexBuffer(YaWGPUSubpassRenderPassEncoder encoder, uint32_t slot, WGPUBuffer buffer, uint64_t offset, uint64_t size);
/**
 * Binds an index buffer for subsequent indexed draws.
 *
 * @param size
 * Range size in bytes. `WGPU_WHOLE_SIZE` means `buffer.size - offset`.
 */
void yawgpuSubpassRenderPassEncoderSetIndexBuffer(YaWGPUSubpassRenderPassEncoder encoder, WGPUBuffer buffer, WGPUIndexFormat format, uint64_t offset, uint64_t size);
/** Records a non-indexed draw call. */
void yawgpuSubpassRenderPassEncoderDraw(YaWGPUSubpassRenderPassEncoder encoder, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance);
/** Records an indexed draw call. */
void yawgpuSubpassRenderPassEncoderDrawIndexed(YaWGPUSubpassRenderPassEncoder encoder, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t baseVertex, uint32_t firstInstance);
/**
 * Sets the viewport for subsequent draws. `minDepth`/`maxDepth` are
 * clamped to `[0, 1]` per WebGPU; yawgpu does not apply Vulkan-style
 * y-flip on either backend.
 */
void yawgpuSubpassRenderPassEncoderSetViewport(YaWGPUSubpassRenderPassEncoder encoder, float x, float y, float width, float height, float minDepth, float maxDepth);
/** Sets the scissor rectangle for subsequent draws (in pixels). */
void yawgpuSubpassRenderPassEncoderSetScissorRect(YaWGPUSubpassRenderPassEncoder encoder, uint32_t x, uint32_t y, uint32_t width, uint32_t height);
/** Increments the refcount of a subpass render pass encoder. */
void yawgpuSubpassRenderPassEncoderAddRef(YaWGPUSubpassRenderPassEncoder encoder);
/** Decrements the refcount of a subpass render pass encoder. */
void yawgpuSubpassRenderPassEncoderRelease(YaWGPUSubpassRenderPassEncoder encoder);

/*
 * Reserved (do NOT reuse for unrelated features):
 *   - YaWGPUFeatureName_ProgrammableTileDispatch == (WGPUFeatureName)0x70010004
 *   - any future tile-dispatch C entry point name
 * The previous scaffold was removed because no backend implements it; the API
 * shape will be defined when a real implementation lands.
 */

/** @} */
#endif

#endif
