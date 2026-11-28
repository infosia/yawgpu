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

extern "C" WGPUWaitStatus smokeTimedWait(const WGPUInstanceDescriptor* descriptor) {
    const auto* table = yawgpuDawnGetProcTable();
    WGPUInstance instance = table->createInstance(descriptor);
    if (!instance) return WGPUWaitStatus_Force32;
    auto status = table->instanceWaitAny(instance, 0, nullptr, 1);
    table->instanceRelease(instance);
    return status;
}

extern "C" bool smokeCheckProcAddress() {
    const auto* table = yawgpuDawnGetProcTable();
    auto create = reinterpret_cast<WGPUProc>(table->createInstance);
    auto lookup = reinterpret_cast<WGPUProcGetProcAddress>(
        table->getProcAddress({"wgpuGetProcAddress", WGPU_STRLEN}));
    return lookup == table->getProcAddress &&
           lookup({"wgpuCreateInstance", WGPU_STRLEN}) == create &&
           table->getProcAddress({"wgpuCreateInstance", WGPU_STRLEN}) == create &&
           table->getProcAddress({"wgpuCreateInstance!", sizeof("wgpuCreateInstance") - 1}) == create &&
           table->getProcAddress({"wgpuInstanceRelease", WGPU_STRLEN}) ==
               reinterpret_cast<WGPUProc>(table->instanceRelease) &&
           !table->getProcAddress({"wgpuAdapterGetFormatCapabilities", WGPU_STRLEN}) &&
           !table->getProcAddress({"wgpuCreateInstance!", WGPU_STRLEN});
}

extern "C" bool smokeCallStubs() {
    const auto* table = yawgpuDawnGetProcTable();
    if (table->adapterGetFormatCapabilities(nullptr, WGPUTextureFormat_Undefined, nullptr)
        != WGPUStatus_Error) return false;
    table->deviceTick(nullptr);
    return table->adapterGetInstance(nullptr) == nullptr &&
           table->resourceTableGetSize(nullptr) == 0 &&
           table->sharedBufferMemoryIsDeviceLost(nullptr) == 0;
}

extern "C" bool smokeUnconfiguredCreateInstance() {
    return wgpuCreateInstance(nullptr) == nullptr;
}

extern "C" void smokeSetWrongVersion() {
    auto copy = *yawgpuDawnGetProcTable();
    copy.version[0] ^= 1;
    dawnProcSetProcs(&copy);
}
