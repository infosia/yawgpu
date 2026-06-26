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
