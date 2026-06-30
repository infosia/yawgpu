// surface_smoke — the minimal windowed program: open a window and present.
//
// It is the windowing counterpart of `capture`: there is no pipeline and no
// draw call, only a clear-only render pass that fills each frame with a
// flat slate color. Use it to verify the window → surface → swapchain →
// present path in isolation, without shaders or vertex data in the way.
// It presents ~60 frames (or until the window is closed) and exits.

#include "framework.h"

// No pipeline/shader/buffer here — just the surface to present to.
typedef struct SurfaceSmokeApp {
    YawgpuContext context;
    WGPUQueue queue;
    YawgpuWindow *window;
    WGPUSurface surface;
} SurfaceSmokeApp;

static bool choose_surface_format(WGPUSurface surface,
                                  WGPUAdapter adapter,
                                  WGPUTextureFormat *format) {
    WGPUSurfaceCapabilities capabilities = {0};
    if (wgpuSurfaceGetCapabilities(surface, adapter, &capabilities) != WGPUStatus_Success) {
        fprintf(stderr, "failed to get surface capabilities\n");
        return false;
    }

    bool found_format = false;
    *format = WGPUTextureFormat_BGRA8Unorm;
    for (size_t i = 0; i < capabilities.formatCount; ++i) {
        if (capabilities.formats[i] == WGPUTextureFormat_BGRA8Unorm) {
            *format = WGPUTextureFormat_BGRA8Unorm;
            found_format = true;
            break;
        }
        if (capabilities.formats[i] == WGPUTextureFormat_RGBA8Unorm) {
            *format = WGPUTextureFormat_RGBA8Unorm;
            found_format = true;
        }
    }
    wgpuSurfaceCapabilitiesFreeMembers(capabilities);
    if (!found_format) {
        fprintf(stderr, "no supported surface format found\n");
        return false;
    }
    return true;
}

static void surface_smoke_app_destroy(SurfaceSmokeApp *app) {
    if (app->surface) {
        wgpuSurfaceUnconfigure(app->surface);
        wgpuSurfaceRelease(app->surface);
    }
    if (app->window) {
        yawgpu_window_destroy(app->window);
    }
    if (app->queue) {
        wgpuQueueRelease(app->queue);
    }
    yawgpu_context_release(&app->context);
    *app = (SurfaceSmokeApp){0};
}

static bool surface_smoke_app_init(SurfaceSmokeApp *app) {
    *app = (SurfaceSmokeApp){0};
    app->context = yawgpu_context_create();
    if (!app->context.instance || !app->context.adapter || !app->context.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        return false;
    }

    app->queue = wgpuDeviceGetQueue(app->context.device);
    if (!app->queue) {
        fprintf(stderr, "failed to get queue\n");
        return false;
    }

    app->window = yawgpu_window_create(800, 600, "yawgpu surface_smoke");
    if (!app->window) {
        fprintf(stderr, "failed to create window\n");
        return false;
    }
    app->surface = yawgpu_window_create_surface(
        app->context.instance, app->window, "surface_smoke surface");
    if (!app->surface) {
        return false;
    }

    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    if (!choose_surface_format(app->surface, app->context.adapter, &format)) {
        return false;
    }

    int width = 0;
    int height = 0;
    yawgpu_window_framebuffer_size(app->window, &width, &height);
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "invalid framebuffer size\n");
        return false;
    }
    wgpuSurfaceConfigure(
        app->surface,
        &(WGPUSurfaceConfiguration){
            .nextInChain = NULL,
            .device = app->context.device,
            .format = format,
            .usage = WGPUTextureUsage_RenderAttachment,
            .width = (uint32_t)width,
            .height = (uint32_t)height,
            .viewFormatCount = 0,
            .viewFormats = NULL,
            .alphaMode = WGPUCompositeAlphaMode_Opaque,
            .presentMode = WGPUPresentMode_Fifo,
        });
    return true;
}

// Acquire → clear → present, with all per-frame handles released before
// returning. No pipeline or draw is involved.
static bool surface_smoke_render_frame(const SurfaceSmokeApp *app) {
    WGPUSurfaceTexture current = {0};
    wgpuSurfaceGetCurrentTexture(app->surface, &current);
    // Noop has no real swapchain (Lost + no texture) — skip the frame.
    if (current.status == WGPUSurfaceGetCurrentTextureStatus_Lost && !current.texture) {
        return true;
    }
    if (current.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
        !current.texture) {
        fprintf(stderr, "failed to acquire surface texture, status=%u\n", current.status);
        return false;
    }

    WGPUTextureView view = wgpuTextureCreateView(current.texture, NULL);
    if (!view) {
        fprintf(stderr, "failed to create texture view\n");
        wgpuTextureRelease(current.texture);
        return false;
    }

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
        app->context.device,
        &(WGPUCommandEncoderDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("surface_smoke encoder"),
        });
    if (!encoder) {
        fprintf(stderr, "failed to create command encoder\n");
        wgpuTextureViewRelease(view);
        wgpuTextureRelease(current.texture);
        return false;
    }

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
        encoder,
        &(WGPURenderPassDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("surface_smoke clear pass"),
            .colorAttachmentCount = 1,
            .colorAttachments = (WGPURenderPassColorAttachment[]){
                {
                    .nextInChain = NULL,
                    .view = view,
                    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
                    .resolveTarget = NULL,
                    // Clear the whole frame to an opaque slate color; the
                    // clear is the only thing this pass does.
                    .loadOp = WGPULoadOp_Clear,
                    .storeOp = WGPUStoreOp_Store,
                    .clearValue = {
                        .r = 0.1,
                        .g = 0.2,
                        .b = 0.3,
                        .a = 1.0,
                    },
                },
            },
            .depthStencilAttachment = NULL,
            .occlusionQuerySet = NULL,
            .timestampWrites = NULL,
        });
    if (!pass) {
        fprintf(stderr, "failed to begin render pass\n");
        wgpuCommandEncoderRelease(encoder);
        wgpuTextureViewRelease(view);
        wgpuTextureRelease(current.texture);
        return false;
    }
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(
        encoder,
        &(WGPUCommandBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("surface_smoke commands"),
        });
    if (!commands) {
        fprintf(stderr, "failed to finish command encoder\n");
        wgpuCommandEncoderRelease(encoder);
        wgpuTextureViewRelease(view);
        wgpuTextureRelease(current.texture);
        return false;
    }
    wgpuQueueSubmit(app->queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);

    WGPUStatus present_status = wgpuSurfacePresent(app->surface);
    wgpuTextureViewRelease(view);
    wgpuTextureRelease(current.texture);
    if (present_status != WGPUStatus_Success) {
        fprintf(stderr, "surface present failed\n");
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    // Keep the window open until the user closes it; with --verify, auto-exit
    // after a few frames (for headless / CI runs).
    const bool verify = (argc > 1 && strcmp(argv[1], "--verify") == 0);
    SurfaceSmokeApp app = {0};
    if (!surface_smoke_app_init(&app)) {
        surface_smoke_app_destroy(&app);
        return EXIT_FAILURE;
    }

    for (uint32_t frame = 0;
         !yawgpu_window_should_close(app.window) && !(verify && frame >= 60); ++frame) {
        if (!surface_smoke_render_frame(&app)) {
            surface_smoke_app_destroy(&app);
            return EXIT_FAILURE;
        }
        yawgpu_window_poll_events();
    }

    surface_smoke_app_destroy(&app);
    return EXIT_SUCCESS;
}
