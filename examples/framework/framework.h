// framework.h — shared helpers for the yawgpu C examples.
//
// The examples are deliberately small, so the repetitive boilerplate
// (creating an instance, driving the async adapter/device requests, loading
// a WGSL file, opening a platform window and turning it into a surface) is
// collected here and declared in this header. Each example #includes it and
// focuses on the WebGPU feature it demonstrates. This is example scaffolding,
// not part of the yawgpu library API.

#ifndef YAWGPU_EXAMPLES_FRAMEWORK_H
#define YAWGPU_EXAMPLES_FRAMEWORK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "webgpu.h"
#include "yawgpu.h"

// Marks an intentionally-unused parameter (e.g. callback userdata we ignore).
#define YAWGPU_UNUSED(x) (void)(x)

// The three handles every example needs, acquired together by
// yawgpu_context_create() and released by yawgpu_context_release().
typedef struct YawgpuContext {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
} YawgpuContext;

// Opaque platform window (GLFW on macOS, Win32 on Windows); see the
// per-platform framework_*.{c,m} implementations.
typedef struct YawgpuWindow YawgpuWindow;

// Convenience descriptor for yawgpu_create_buffer_init: create a buffer and
// upload `contents` into it in one call.
typedef struct YawgpuBufferInitDescriptor {
    const char *label;
    WGPUBufferUsage usage;
    const void *contents;
    size_t size;
} YawgpuBufferInitDescriptor;

// Wraps a NUL-terminated C string as a WGPUStringView (data + length).
WGPUStringView yawgpu_string_view(const char *value);
// Prints a WGPUStringView to stdout, honoring its (possibly non-NUL) length.
void yawgpu_print_string_view(WGPUStringView value);
// Number of uncaptured device errors reported through the framework callback.
unsigned int yawgpu_uncaptured_error_count(void);

// Creates an instance with the backend chosen by the YAWGPU_BACKEND env var.
// Also chains a YaWGPUGlesContextBackend entry whose value comes from the
// optional YAWGPU_GLES_CONTEXT_BACKEND env var (egl / wgl / default); the
// library ignores it for non-GLES backends and treats DEFAULT (the unset
// case) as "defer to YAWGPU_GLES_BACKEND", preserving the existing
// behaviour byte-for-byte for callers that don't set the new var.
WGPUInstance yawgpu_instance_create(void);
// Drives wgpuInstanceRequestAdapter to completion and returns the adapter.
WGPUAdapter yawgpu_request_adapter(WGPUInstance instance);
// Drives wgpuAdapterRequestDevice to completion and returns the device.
WGPUDevice yawgpu_request_device(WGPUInstance instance, WGPUAdapter adapter);
// instance + adapter + device in one call (NULL fields on failure).
YawgpuContext yawgpu_context_create(void);
// Releases whatever handles a context holds; safe on a zeroed/partial one.
void yawgpu_context_release(YawgpuContext *context);

// Reads a WGSL file from `path` and compiles it into a shader module.
WGPUShaderModule yawgpu_load_wgsl_shader(WGPUDevice device, const char *path);
// Creates a buffer (mappedAtCreation) and memcpy's `contents` into it.
WGPUBuffer yawgpu_create_buffer_init(WGPUDevice device,
                                     const YawgpuBufferInitDescriptor *descriptor);
// Pumps the instance's event loop until `future` completes.
void yawgpu_wait_for_future(WGPUInstance instance, WGPUFuture future);
// Prints the adapter's vendor/architecture/device/backend info.
void yawgpu_print_adapter_info(WGPUAdapter adapter);

// --- Windowing (implemented per platform in framework_macos.m / _windows.c) ---

// Opens a window of the given size and title; NULL on failure.
YawgpuWindow *yawgpu_window_create(int width, int height, const char *title);
// Closes the window and frees its resources.
void yawgpu_window_destroy(YawgpuWindow *window);
// True once the user has asked to close the window.
bool yawgpu_window_should_close(YawgpuWindow *window);
// Processes pending window/input events; call once per frame.
void yawgpu_window_poll_events(void);
// Creates the WebGPU surface backed by this window's native handle.
WGPUSurface yawgpu_window_create_surface(WGPUInstance instance,
                                         YawgpuWindow *window,
                                         const char *label);
// Returns the window's CAMetalLayer pointer (macOS only; NULL elsewhere).
void *yawgpu_window_metal_layer(YawgpuWindow *window);
// Reports the window's framebuffer size in physical pixels.
void yawgpu_window_framebuffer_size(YawgpuWindow *window, int *width, int *height);

#endif
