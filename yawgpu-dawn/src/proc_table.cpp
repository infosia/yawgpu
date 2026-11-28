#include "yawgpu_dawn.h"
#include "dawn/dawn_proc_table.h"
#include "dawn/dawn_version.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" decltype(wgpuCreateInstance) yawgpuDawnCreateInstance;

namespace {
void reportUnsupported(const char* name, std::atomic_flag& reported) {
    const char* setting = std::getenv("YAWGPU_DAWN_STRICT");
    const bool strict = setting && std::strcmp(setting, "1") == 0;
    if (!reported.test_and_set(std::memory_order_relaxed) || strict) {
        std::fprintf(stderr, "yawgpu-dawn: unsupported %s\n", name);
        std::fflush(stderr);
    }
    if (strict) std::abort();
}

constexpr DawnProcTable makeTable() {
    DawnProcTable table{};
    std::ranges::copy(dawn::kDawnVersion, table.version);
#include "proc_init.inc"
    return table;
}

constexpr DawnProcTable kTable = makeTable();
}  // namespace

extern "C" const DawnProcTable* yawgpuDawnGetProcTable() {
    return &kTable;
}
