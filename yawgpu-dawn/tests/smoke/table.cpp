#include "dawn/dawn_proc.h"
#include "yawgpu_dawn.h"

#include <algorithm>

extern "C" bool smokeCheckTable(const DawnProcTable* table) {
    if (table != yawgpuDawnGetProcTable()) return false;
#include "check_members.inc"
    return std::ranges::equal(table->version, dawn::kDawnVersion);
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
