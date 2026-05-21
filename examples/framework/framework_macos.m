// framework_macos.m — macOS implementation of the windowing helpers
// (Objective-C, compiled only on Apple). It uses GLFW for the window and
// attaches a CAMetalLayer to the window's content view: that layer is the
// native surface yawgpu renders into, on both the Metal and Vulkan
// (MoltenVK) backends.

#define GLFW_EXPOSE_NATIVE_COCOA

#include "framework.h"

#include <QuartzCore/CAMetalLayer.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

// Concrete definition of the opaque YawgpuWindow from framework.h.
struct YawgpuWindow {
    GLFWwindow *handle;
    CAMetalLayer *layer;
};

YawgpuWindow *yawgpu_window_create(int width, int height, const char *title) {
    if (!glfwInit()) {
        return NULL;
    }
    // GLFW would create an OpenGL context by default; NO_API disables that
    // since we drive the surface through Metal/Vulkan ourselves.
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *handle = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!handle) {
        glfwTerminate();
        return NULL;
    }

    // Reach the Cocoa NSWindow behind the GLFW window and make its content
    // view layer-backed, then install a CAMetalLayer as that layer.
    NSWindow *native_window = glfwGetCocoaWindow(handle);
    NSView *content_view = [native_window contentView];
    [content_view setWantsLayer:YES];

    CAMetalLayer *layer = [CAMetalLayer layer];
    [layer retain]; // we keep our own reference; released in window_destroy
    // framebufferOnly=NO allows the layer's drawable texture to be copied
    // from (needed for read-back / blit paths).
    [layer setFramebufferOnly:NO];
    [content_view setLayer:layer];

    YawgpuWindow *window = (YawgpuWindow *)calloc(1, sizeof(YawgpuWindow));
    if (!window) {
        [layer release];
        glfwDestroyWindow(handle);
        glfwTerminate();
        return NULL;
    }
    window->handle = handle;
    window->layer = layer;
    return window;
}

void yawgpu_window_destroy(YawgpuWindow *window) {
    if (!window) {
        return;
    }
    if (window->layer) {
        [window->layer release];
    }
    if (window->handle) {
        glfwDestroyWindow(window->handle);
    }
    free(window);
    glfwTerminate();
}

bool yawgpu_window_should_close(YawgpuWindow *window) {
    return !window || glfwWindowShouldClose(window->handle);
}

void yawgpu_window_poll_events(void) {
    glfwPollEvents();
}

void *yawgpu_window_metal_layer(YawgpuWindow *window) {
    return window ? window->layer : NULL;
}

// Wraps the window's CAMetalLayer as a WebGPU surface. The native handle is
// passed through a WGPUSurfaceSourceMetalLayer struct chained onto the
// surface descriptor — the platform-specific way to identify a surface.
WGPUSurface yawgpu_window_create_surface(WGPUInstance instance,
                                         YawgpuWindow *window,
                                         const char *label) {
    void *layer = yawgpu_window_metal_layer(window);
    if (!layer) {
        fprintf(stderr, "failed to get CAMetalLayer\n");
        return NULL;
    }

    WGPUSurfaceSourceMetalLayer metal_layer = {
        .chain = {
            .next = NULL,
            .sType = WGPUSType_SurfaceSourceMetalLayer,
        },
        .layer = layer,
    };
    WGPUSurface surface = wgpuInstanceCreateSurface(
        instance,
        &(WGPUSurfaceDescriptor){
            .nextInChain = &metal_layer.chain,
            .label = yawgpu_string_view(label),
        });
    if (!surface) {
        fprintf(stderr, "failed to create surface\n");
    }
    return surface;
}

// Reports the framebuffer size in physical pixels and keeps the metal
// layer's drawable size in sync with it (so the swapchain matches the
// window, including HiDPI scaling).
void yawgpu_window_framebuffer_size(YawgpuWindow *window, int *width, int *height) {
    int local_width = 0;
    int local_height = 0;
    if (window) {
        glfwGetFramebufferSize(window->handle, &local_width, &local_height);
        [window->layer setDrawableSize:CGSizeMake(local_width, local_height)];
    }
    if (width) {
        *width = local_width;
    }
    if (height) {
        *height = local_height;
    }
}
