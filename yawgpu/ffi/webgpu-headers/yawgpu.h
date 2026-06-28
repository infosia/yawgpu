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
 * - **Shader passthrough** (compile-time macro): raw native shader source
 *   chained into `WGPUShaderModuleDescriptor`.
 * - **External texture creation** (always available): vendor creation API for
 *   WebGPU external textures.
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
     *  otherwise (per IB3 rules above) `wgpuCreateInstance` returns NULL when
     *  this backend is explicitly requested via @ref YaWGPUInstanceBackendSelect. */
    YAWGPU_INSTANCE_BACKEND_GLES = 3,
};

/**
 * Chained extension that selects a specific HAL backend.
 *
 * `chain.sType` must be set to @ref YAWGPU_STYPE_INSTANCE_BACKEND_SELECT and
 * `backend` to one of the `YAWGPU_INSTANCE_BACKEND_*` constants. Resolution
 * follows these rules (`wgpuCreateInstance` return value in parentheses):
 *
 * - **No chain entry present** (`nextInChain` does not contain a
 *   @ref YaWGPUInstanceBackendSelect): a Noop instance is returned
 *   (non-NULL).
 * - **`backend == YAWGPU_INSTANCE_BACKEND_NOOP`**: a Noop instance is
 *   returned (non-NULL).
 * - **`backend == YAWGPU_INSTANCE_BACKEND_{METAL, VULKAN, GLES}`**: strict.
 *   `wgpuCreateInstance` returns NULL when the matching cargo feature was
 *   not compiled into this yawgpu build, when the backend's instance
 *   creation fails, or when the backend exposes no adapters. A best-effort
 *   diagnostic line is written to `stderr` identifying which cause fired
 *   (the only in-band signal is the NULL return; webgpu.h does not provide
 *   an error callback on `wgpuCreateInstance`). Callers wanting to confirm
 *   which backend was selected may inspect
 *   `wgpuAdapterGetInfo().backendType` after `wgpuInstanceRequestAdapter`.
 * - **Unrecognised `backend` value** (anything outside the four constants
 *   above): treated as if no chain were present and returns a Noop instance
 *   (non-NULL). This keeps older yawgpu builds forward-compatible with
 *   descriptors produced from a newer header that may define additional
 *   backend constants.
 */
typedef struct YaWGPUInstanceBackendSelect {
    /** Chain header. `sType` must be @ref YAWGPU_STYPE_INSTANCE_BACKEND_SELECT. */
    WGPUChainedStruct chain;
    /** One of the `YAWGPU_INSTANCE_BACKEND_*` constants. */
    uint32_t backend;
} YaWGPUInstanceBackendSelect;

/** @} */

/**
 * \defgroup ShaderPassthrough Shader Passthrough
 * \brief Vendor shader-source chain entries for native shader code.
 *
 * @{
 */

/** Defined when yawgpu's shader-passthrough declarations are available. */
#define YAWGPU_HAS_SHADER_PASSTHROUGH 1

/**
 * `WGPUSType` tag identifying a @ref YaWGPUShaderSourceMSL in a
 * chained-struct list.
 */
#define YAWGPU_STYPE_SHADER_SOURCE_MSL ((WGPUSType)0x70000004u)

/** Entry point metadata for a raw MSL shader module. */
typedef struct YaWGPUMslEntryPoint {
    /** Entry point function name. */
    WGPUStringView name;
    /** Exactly one of Vertex, Fragment, or Compute. */
    WGPUShaderStage stage;
    /** Compute workgroup size; ignored for non-compute entries. */
    uint32_t workgroupSize[3];
} YaWGPUMslEntryPoint;

/** Raw MSL shader source chained onto `WGPUShaderModuleDescriptor`. */
typedef struct YaWGPUShaderSourceMSL {
    /** Chain header. `sType` must be @ref YAWGPU_STYPE_SHADER_SOURCE_MSL. */
    WGPUChainedStruct chain;
    /** MSL source code. */
    WGPUStringView code;
    /** Number of entries pointed to by `entryPoints`. */
    size_t entryPointCount;
    /** Caller-provided entry point metadata. */
    YaWGPUMslEntryPoint const* entryPoints;
} YaWGPUShaderSourceMSL;

/**
 * Metal binding-slot ABI for passthrough MSL.
 *
 * Shader passthrough requires an explicit pipeline layout. yawgpu derives Metal
 * argument indices from that layout, never from shader reflection, so the MSL
 * source must use the exact `[[buffer(n)]]`, `[[texture(n)]]`, and
 * `[[sampler(n)]]` indices described here.
 *
 * For compute pipelines, yawgpu gathers every binding layout entry from every
 * bind group layout, sorts entries by `(group, binding)`, then assigns
 * zero-based counters independently by resource kind in that order:
 * buffers use `[[buffer(0)]]`, `[[buffer(1)]]`, ...; sampled and storage
 * textures use `[[texture(0)]]`, ...; samplers use `[[sampler(0)]]`, ....
 * An external texture consumes two consecutive texture slots plus one buffer
 * slot for its params.
 *
 * For render pipelines, vertex and fragment stages each have their own
 * independent buffer/texture/sampler counters. Entries are still considered in
 * `(group, binding)` order, but only entries visible to a stage consume slots
 * in that stage's function. For example, a buffer visible only to fragment can
 * be `[[buffer(0)]]` in the fragment entry point even when the vertex entry
 * point also has its own `[[buffer(0)]]`.
 *
 * Example compute layout:
 * - group 0 binding 0: storage buffer  -> `device T* data [[buffer(0)]]`
 * - group 0 binding 1: sampled texture -> `texture2d<float> tex [[texture(0)]]`
 * - group 0 binding 2: sampler         -> `sampler samp [[sampler(0)]]`
 */

/** Default initializer for @ref YaWGPUMslEntryPoint. */
#define YAWGPU_MSL_ENTRY_POINT_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUMslEntryPoint, { \
    /*.name=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.stage=*/WGPUShaderStage_None _wgpu_COMMA \
    /*.workgroupSize=*/{0 _wgpu_COMMA 0 _wgpu_COMMA 0} _wgpu_COMMA \
})

/** Default initializer for @ref YaWGPUShaderSourceMSL. */
#define YAWGPU_SHADER_SOURCE_MSL_INIT _wgpu_MAKE_INIT_STRUCT(YaWGPUShaderSourceMSL, { \
    /*.chain=*/{NULL _wgpu_COMMA YAWGPU_STYPE_SHADER_SOURCE_MSL} _wgpu_COMMA \
    /*.code=*/WGPU_STRING_VIEW_INIT _wgpu_COMMA \
    /*.entryPointCount=*/0 _wgpu_COMMA \
    /*.entryPoints=*/NULL _wgpu_COMMA \
})

/** @} */

/**
 * \defgroup ExternalTextureCreate External Texture Create
 * \brief Vendor creation API for `WGPUExternalTexture`.
 *
 * The core `webgpu.h` declares the opaque `WGPUExternalTexture` handle and
 * bind-group layout/entry types, but does not standardize creation. yawgpu
 * exposes a Dawn-shaped creation descriptor so runtime parameter packing can
 * match Tint's multiplanar external-texture transform.
 *
 * @{
 */

/** `WGPUSType` tag reserved for future chained external-texture extensions. */
#define YAWGPU_STYPE_EXTERNAL_TEXTURE_DESCRIPTOR ((WGPUSType)0x70000003u)

/** External texture source format. */
typedef enum YaWGPUExternalTextureFormat {
    /** Single-plane RGBA passthrough. `plane1` must be NULL. */
    YaWGPUExternalTextureFormat_Rgba = 0,
    /** Two-plane NV12. `plane0` is Y, `plane1` is interleaved UV. */
    YaWGPUExternalTextureFormat_Nv12 = 1,
} YaWGPUExternalTextureFormat;

/** External texture sampling rotation. */
typedef enum YaWGPUExternalTextureRotation {
    /** No rotation. */
    YaWGPUExternalTextureRotation_Rotate0Degrees = 0,
    /** Rotate 90 degrees. */
    YaWGPUExternalTextureRotation_Rotate90Degrees = 1,
    /** Rotate 180 degrees. */
    YaWGPUExternalTextureRotation_Rotate180Degrees = 2,
    /** Rotate 270 degrees. */
    YaWGPUExternalTextureRotation_Rotate270Degrees = 3,
} YaWGPUExternalTextureRotation;

/** Two-dimensional origin used by yawgpu vendor descriptors. */
typedef struct YaWGPUOrigin2D {
    /** X coordinate in texels. */
    uint32_t x;
    /** Y coordinate in texels. */
    uint32_t y;
} YaWGPUOrigin2D;

/** Two-dimensional extent used by yawgpu vendor descriptors. */
typedef struct YaWGPUExtent2D {
    /** Width in texels. */
    uint32_t width;
    /** Height in texels. */
    uint32_t height;
} YaWGPUExtent2D;

/**
 * Descriptor for `yawgpuDeviceCreateExternalTexture`.
 *
 * This is a plain vendor struct passed directly to the create function, not a
 * chained `sType` entry. Matrix fields use the same flat float order as Dawn:
 * column-major `mat3x4` for YUV-to-RGB and column-major `mat3x3` for gamut
 * conversion.
 */
typedef struct YaWGPUExternalTextureDescriptor {
    /** First plane texture view. Required. */
    WGPUTextureView plane0;
    /** Second plane texture view. Required for NV12 and NULL for RGBA. */
    WGPUTextureView plane1;
    /** Source format. */
    YaWGPUExternalTextureFormat format;
    /** Crop origin in plane0 texels. */
    YaWGPUOrigin2D cropOrigin;
    /** Crop size in plane0 texels. */
    YaWGPUExtent2D cropSize;
    /** Shader-visible size returned by external texture dimensions. */
    YaWGPUExtent2D apparentSize;
    /** When true, shaders stop after YUV-to-RGB conversion. */
    WGPUBool doYuvToRgbConversionOnly;
    /** Column-major `mat3x4<f32>` YUV-to-RGB conversion matrix. */
    float yuvToRgbConversionMatrix[12];
    /** Source transfer function parameters: G, A, B, C, D, E, F. */
    float srcTransferFunctionParameters[7];
    /** Destination transfer function parameters: G, A, B, C, D, E, F. */
    float dstTransferFunctionParameters[7];
    /** Column-major `mat3x3<f32>` gamut conversion matrix. */
    float gamutConversionMatrix[9];
    /** Whether sampling should mirror horizontally. */
    WGPUBool mirrored;
    /** Sampling rotation. */
    YaWGPUExternalTextureRotation rotation;
} YaWGPUExternalTextureDescriptor;

/**
 * Creates a `WGPUExternalTexture`.
 *
 * Validation errors are routed through the device error sink and return NULL.
 */
WGPU_EXPORT WGPUExternalTexture yawgpuDeviceCreateExternalTexture(
    WGPUDevice device,
    YaWGPUExternalTextureDescriptor const * descriptor) WGPU_FUNCTION_ATTRIBUTE;

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


#endif
