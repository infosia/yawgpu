// enumerate_adapters — the smallest possible yawgpu program.
//
// It walks the first two steps every WebGPU application takes:
//   1. create an *instance* (the entry point to the whole API), and
//   2. request an *adapter* from it (a handle to a physical GPU, or to
//      yawgpu's CPU-only "Noop" backend).
// It then prints what the adapter reports about itself and exits.
//
// The backend is chosen at runtime by the YAWGPU_BACKEND environment
// variable (noop | metal | vulkan); see framework.c for how that is
// turned into the instance descriptor.

#include "framework.h"

int main(void) {
    // An instance owns no GPU resources by itself — it is the factory
    // from which adapters (and everything else) are requested.
    WGPUInstance instance = yawgpu_instance_create();
    if (!instance) {
        fprintf(stderr, "failed to create instance\n");
        return EXIT_FAILURE;
    }

    // Requesting an adapter is asynchronous in WebGPU: the result arrives
    // through a callback. The framework helper drives the event loop until
    // the callback fires and then hands back the adapter (or NULL).
    WGPUAdapter adapter = yawgpu_request_adapter(instance);
    if (!adapter) {
        fprintf(stderr, "failed to request adapter\n");
        wgpuInstanceRelease(instance);
        return EXIT_FAILURE;
    }

    printf("Requested adapter for YAWGPU_BACKEND=%s\n",
           getenv("YAWGPU_BACKEND") ? getenv("YAWGPU_BACKEND") : "noop");
    // Prints vendor / architecture / device / backend type, etc.
    yawgpu_print_adapter_info(adapter);

    // Every handle obtained from the API is reference-counted. Release the
    // references we hold, in reverse order of acquisition, before exiting.
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return EXIT_SUCCESS;
}
