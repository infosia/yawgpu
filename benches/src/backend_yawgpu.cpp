#include "backend.h"

#include <cstdlib>
#include <cstring>

#include <webgpu-headers/yawgpu.h>

namespace bench {

WGPUInstance createInstance() {
    YaWGPUInstanceBackendSelect backendSelect = {};
    backendSelect.chain.next = nullptr;
    backendSelect.chain.sType = YAWGPU_STYPE_INSTANCE_BACKEND_SELECT;
#if defined(__APPLE__)
    backendSelect.backend = YAWGPU_INSTANCE_BACKEND_METAL;
#else
    backendSelect.backend = YAWGPU_INSTANCE_BACKEND_VULKAN;
#endif
    // Same override knob the CTS harness uses, so a benchmark run and a
    // conformance run can be pointed at the same HAL.
    const char* sel = std::getenv("YAWGPU_BENCH_BACKEND");
    if (sel != nullptr) {
        if (std::strcmp(sel, "metal") == 0) {
            backendSelect.backend = YAWGPU_INSTANCE_BACKEND_METAL;
        } else if (std::strcmp(sel, "vulkan") == 0) {
            backendSelect.backend = YAWGPU_INSTANCE_BACKEND_VULKAN;
        } else if (std::strcmp(sel, "gles") == 0) {
            backendSelect.backend = YAWGPU_INSTANCE_BACKEND_GLES;
        }
    }

    WGPUInstanceDescriptor descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    descriptor.nextInChain = &backendSelect.chain;
    return wgpuCreateInstance(&descriptor);
}

const char* backendName() {
    return "yawgpu";
}

const WGPURequestAdapterOptions* adapterOptions() {
    return nullptr;
}

} // namespace bench
