#include "yawgpu.h"

// Keep yawgpu.h's WebGPU declarations in their own translation unit.
const void* smokeBackendChain(int metal) {
    static const YaWGPUInstanceBackendSelect noop = {
        {NULL, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT}, YAWGPU_INSTANCE_BACKEND_NOOP};
    static const YaWGPUInstanceBackendSelect metalBackend = {
        {NULL, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT}, YAWGPU_INSTANCE_BACKEND_METAL};
    return metal ? &metalBackend.chain : &noop.chain;
}
