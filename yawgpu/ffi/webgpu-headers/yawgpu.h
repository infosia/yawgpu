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
/* Phase 13 A3 adds yawgpuDeviceCreateShaderModule{SpirV,Msl} here. */
#endif

#if defined(YAWGPU_HAS_TILED)
/* Phase 14 adds the tiled surface here. */
#endif

#endif
