// Backend selection seam for the cross-implementation benchmark harness.
//
// The benchmark body is written once against canonical `webgpu.h` and linked
// against either yawgpu or Dawn. Only instance creation differs between the
// two, exactly as in webgpu-native-cts's `src/common/webgpu/backend.h`.
#pragma once

#if defined(YAWGPU_BENCH_BACKEND_DAWN)
#include <webgpu/webgpu.h>
#else
#include <webgpu-headers/webgpu.h>
#endif

namespace bench {

// Creates the instance, selecting the platform backend the harness was
// configured for (Metal on macOS, Vulkan elsewhere).
WGPUInstance createInstance();

// Short identifier for the linked implementation, used in report headers.
const char* backendName();

// Adapter options required to pin the implementation to the intended backend,
// or nullptr when the implementation selects it at instance creation.
const WGPURequestAdapterOptions* adapterOptions();

} // namespace bench
