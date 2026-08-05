#include "backend.h"

#include <cstdlib>
#include <cstring>

namespace {

WGPUBackendType configuredBackendType() {
#if defined(__APPLE__)
    WGPUBackendType backendType = WGPUBackendType_Metal;
#else
    WGPUBackendType backendType = WGPUBackendType_Vulkan;
#endif
    const char* sel = std::getenv("YAWGPU_BENCH_BACKEND");
    if (sel != nullptr) {
        if (std::strcmp(sel, "metal") == 0) {
            backendType = WGPUBackendType_Metal;
        } else if (std::strcmp(sel, "vulkan") == 0) {
            backendType = WGPUBackendType_Vulkan;
        } else if (std::strcmp(sel, "opengles") == 0 || std::strcmp(sel, "gles") == 0) {
            backendType = WGPUBackendType_OpenGLES;
        }
    }
    return backendType;
}

} // namespace

namespace bench {

WGPUInstance createInstance() {
    WGPUInstanceDescriptor descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    return wgpuCreateInstance(&descriptor);
}

const char* backendName() {
    return "dawn";
}

const WGPURequestAdapterOptions* adapterOptions() {
    static WGPURequestAdapterOptions options = [] {
        WGPURequestAdapterOptions o = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
        o.backendType = configuredBackendType();
        return o;
    }();
    return &options;
}

} // namespace bench
