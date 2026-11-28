#include "dawn/dawn_proc.h"
#include "yawgpu_dawn.h"

#include <algorithm>

extern "C" bool smokeCheckTable(const DawnProcTable* table) {
    if (table != yawgpuDawnGetProcTable()) return false;
#include "check_members.inc"
    return std::ranges::equal(table->version, dawn::kDawnVersion);
}

extern "C" WGPUBackendType smokeAdapterBackend(const WGPUInstanceDescriptor* descriptor) {
    const auto* table = yawgpuDawnGetProcTable();
    WGPUInstance instance = table->createInstance(descriptor);
    if (!instance) return WGPUBackendType_Undefined;

    WGPUAdapter adapter = nullptr;
    WGPURequestAdapterCallbackInfo callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.userdata1 = &adapter;
    callback.callback = [](WGPURequestAdapterStatus, WGPUAdapter result, WGPUStringView,
                           void* userdata, void*) {
        *static_cast<WGPUAdapter*>(userdata) = result;
    };
    WGPUFutureWaitInfo wait{table->instanceRequestAdapter(instance, nullptr, callback), false};
    WGPUBackendType backend = WGPUBackendType_Undefined;
    if (table->instanceWaitAny(instance, 1, &wait, 0) == WGPUWaitStatus_Success && adapter) {
        WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
        if (table->adapterGetInfo(adapter, &info) == WGPUStatus_Success) backend = info.backendType;
        table->adapterInfoFreeMembers(info);
    }
    if (adapter) table->adapterRelease(adapter);
    table->instanceRelease(instance);
    return backend;
}

extern "C" bool smokeCallStub() {
    const auto* table = yawgpuDawnGetProcTable();
    if (table->getProcAddress({"wgpuAdapterGetFormatCapabilities", WGPU_STRLEN})) return false;
    return table->adapterGetFormatCapabilities(nullptr, WGPUTextureFormat_Undefined, nullptr)
        == WGPUStatus_Error;
}

extern "C" void smokeSetWrongVersion() {
    auto copy = *yawgpuDawnGetProcTable();
    copy.version[0] ^= 1;
    dawnProcSetProcs(&copy);
}
