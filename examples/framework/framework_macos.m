#define GLFW_EXPOSE_NATIVE_COCOA

#include "framework.h"

#include <QuartzCore/CAMetalLayer.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

struct YawgpuWindow {
    GLFWwindow *handle;
    CAMetalLayer *layer;
};

YawgpuWindow *yawgpu_window_create(int width, int height, const char *title) {
    if (!glfwInit()) {
        return NULL;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *handle = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!handle) {
        glfwTerminate();
        return NULL;
    }

    NSWindow *native_window = glfwGetCocoaWindow(handle);
    NSView *content_view = [native_window contentView];
    [content_view setWantsLayer:YES];

    CAMetalLayer *layer = [CAMetalLayer layer];
    [layer retain];
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
