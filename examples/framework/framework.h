#ifndef YAWGPU_EXAMPLES_FRAMEWORK_H
#define YAWGPU_EXAMPLES_FRAMEWORK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "webgpu.h"

#define YAWGPU_UNUSED(x) (void)(x)
#define YAWGPU_STYPE_YAWGPU_INSTANCE_BACKEND_SELECT ((WGPUSType)0x70000001u)

enum {
    YAWGPU_INSTANCE_BACKEND_NOOP = 0,
    YAWGPU_INSTANCE_BACKEND_METAL = 1,
    YAWGPU_INSTANCE_BACKEND_VULKAN = 2,
};

typedef struct WGPUYawgpuInstanceBackendSelect {
    WGPUChainedStruct chain;
    uint32_t backend;
} WGPUYawgpuInstanceBackendSelect;

typedef struct YawgpuContext {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
} YawgpuContext;

typedef struct YawgpuWindow YawgpuWindow;

typedef struct YawgpuBufferInitDescriptor {
    const char *label;
    WGPUBufferUsage usage;
    const void *contents;
    size_t size;
} YawgpuBufferInitDescriptor;

WGPUStringView yawgpu_string_view(const char *value);
void yawgpu_print_string_view(WGPUStringView value);

WGPUInstance yawgpu_instance_create(void);
WGPUAdapter yawgpu_request_adapter(WGPUInstance instance);
WGPUDevice yawgpu_request_device(WGPUInstance instance, WGPUAdapter adapter);
YawgpuContext yawgpu_context_create(void);
void yawgpu_context_release(YawgpuContext *context);

WGPUShaderModule yawgpu_load_wgsl_shader(WGPUDevice device, const char *path);
WGPUBuffer yawgpu_create_buffer_init(WGPUDevice device,
                                     const YawgpuBufferInitDescriptor *descriptor);
void yawgpu_wait_for_future(WGPUInstance instance, WGPUFuture future);
void yawgpu_print_adapter_info(WGPUAdapter adapter);

YawgpuWindow *yawgpu_window_create(int width, int height, const char *title);
void yawgpu_window_destroy(YawgpuWindow *window);
bool yawgpu_window_should_close(YawgpuWindow *window);
void yawgpu_window_poll_events(void);
WGPUSurface yawgpu_window_create_surface(WGPUInstance instance,
                                         YawgpuWindow *window,
                                         const char *label);
void *yawgpu_window_metal_layer(YawgpuWindow *window);
void yawgpu_window_framebuffer_size(YawgpuWindow *window, int *width, int *height);

#endif
