#include "framework.h"

int main(void) {
    WGPUInstance instance = yawgpu_instance_create();
    if (!instance) {
        fprintf(stderr, "failed to create instance\n");
        return EXIT_FAILURE;
    }

    WGPUAdapter adapter = yawgpu_request_adapter(instance);
    if (!adapter) {
        fprintf(stderr, "failed to request adapter\n");
        wgpuInstanceRelease(instance);
        return EXIT_FAILURE;
    }

    printf("Requested adapter for YAWGPU_BACKEND=%s\n",
           getenv("YAWGPU_BACKEND") ? getenv("YAWGPU_BACKEND") : "noop");
    yawgpu_print_adapter_info(adapter);

    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return EXIT_SUCCESS;
}
