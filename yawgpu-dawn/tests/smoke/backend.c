#include "yawgpu.h"

// Keep yawgpu.h's WebGPU declarations in their own translation unit.
// `backend` is a YAWGPU_INSTANCE_BACKEND_* value; anything unlisted selects Noop.
const void* smokeBackendChain(uint32_t backend) {
    static const YaWGPUInstanceBackendSelect noop = {
        {NULL, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT}, YAWGPU_INSTANCE_BACKEND_NOOP};
    static const YaWGPUInstanceBackendSelect metal = {
        {NULL, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT}, YAWGPU_INSTANCE_BACKEND_METAL};
    static const YaWGPUInstanceBackendSelect vulkan = {
        {NULL, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT}, YAWGPU_INSTANCE_BACKEND_VULKAN};
    switch (backend) {
        case YAWGPU_INSTANCE_BACKEND_METAL: return &metal.chain;
        case YAWGPU_INSTANCE_BACKEND_VULKAN: return &vulkan.chain;
        default: return &noop.chain;
    }
}
