// Use the yawgpu extension header separately from Dawn's generated WebGPU declarations.
#include "yawgpu.h"
#include "yawgpu_dawn.h"

#include <atomic>

namespace {
std::atomic<uint32_t> defaultBackend{YAWGPU_DAWN_DEFAULT_BACKEND};
}

extern "C" WGPUInstance yawgpuDawnCreateInstance(const WGPUInstanceDescriptor* descriptor) {
    if (descriptor) {
        for (const auto* node = descriptor->nextInChain; node; node = node->next) {
            if (node->sType == YAWGPU_STYPE_INSTANCE_BACKEND_SELECT) {
                return wgpuCreateInstance(descriptor);
            }
        }
    }

    WGPUInstanceDescriptor copy = descriptor ? *descriptor : WGPUInstanceDescriptor{};
    YaWGPUInstanceBackendSelect selector{
        {copy.nextInChain, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT},
        defaultBackend.load(std::memory_order_relaxed)};
    copy.nextInChain = &selector.chain;
    return wgpuCreateInstance(&copy);
}

extern "C" void yawgpuDawnSetDefaultBackend(uint32_t backend) {
    defaultBackend.store(backend, std::memory_order_relaxed);
}
