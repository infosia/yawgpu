#include "framework.h"

typedef struct TriangleApp {
    YawgpuContext context;
    WGPUQueue queue;
    YawgpuWindow *window;
    WGPUSurface surface;
    WGPUShaderModule shader;
    WGPUPipelineLayout pipeline_layout;
    WGPURenderPipeline pipeline;
} TriangleApp;

static bool triangle_choose_surface_format(WGPUSurface surface,
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

static void triangle_app_destroy(TriangleApp *app) {
    if (app->pipeline) {
        wgpuRenderPipelineRelease(app->pipeline);
    }
    if (app->pipeline_layout) {
        wgpuPipelineLayoutRelease(app->pipeline_layout);
    }
    if (app->shader) {
        wgpuShaderModuleRelease(app->shader);
    }
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
    *app = (TriangleApp){0};
}

static bool triangle_create_surface(TriangleApp *app) {
    void *layer = yawgpu_window_metal_layer(app->window);
    if (!layer) {
        fprintf(stderr, "failed to get CAMetalLayer\n");
        return false;
    }

    WGPUSurfaceSourceMetalLayer metal_layer = {
        .chain = {
            .next = NULL,
            .sType = WGPUSType_SurfaceSourceMetalLayer,
        },
        .layer = layer,
    };
    app->surface = wgpuInstanceCreateSurface(
        app->context.instance,
        &(WGPUSurfaceDescriptor){
            .nextInChain = &metal_layer.chain,
            .label = yawgpu_string_view("triangle surface"),
        });
    if (!app->surface) {
        fprintf(stderr, "failed to create surface\n");
        return false;
    }
    return true;
}

static bool triangle_create_pipeline(TriangleApp *app, WGPUTextureFormat format) {
    app->shader = yawgpu_load_wgsl_shader(app->context.device, "shader.wgsl");
    if (!app->shader) {
        fprintf(stderr, "failed to load shader.wgsl\n");
        return false;
    }

    app->pipeline_layout = wgpuDeviceCreatePipelineLayout(
        app->context.device,
        &(WGPUPipelineLayoutDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("triangle pipeline layout"),
            .bindGroupLayoutCount = 0,
            .bindGroupLayouts = NULL,
        });
    if (!app->pipeline_layout) {
        fprintf(stderr, "failed to create pipeline layout\n");
        return false;
    }

    app->pipeline = wgpuDeviceCreateRenderPipeline(
        app->context.device,
        &(WGPURenderPipelineDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("triangle pipeline"),
            .layout = app->pipeline_layout,
            .vertex = {
                .nextInChain = NULL,
                .module = app->shader,
                .entryPoint = yawgpu_string_view("vs_main"),
                .constantCount = 0,
                .constants = NULL,
                .bufferCount = 0,
                .buffers = NULL,
            },
            .primitive = {
                .nextInChain = NULL,
                .topology = WGPUPrimitiveTopology_TriangleList,
                .stripIndexFormat = WGPUIndexFormat_Undefined,
                .frontFace = WGPUFrontFace_CCW,
                .cullMode = WGPUCullMode_None,
            },
            .depthStencil = NULL,
            .multisample = {
                .nextInChain = NULL,
                .count = 1,
                .mask = 0xFFFFFFFFu,
                .alphaToCoverageEnabled = false,
            },
            .fragment = &(WGPUFragmentState){
                .nextInChain = NULL,
                .module = app->shader,
                .entryPoint = yawgpu_string_view("fs_main"),
                .constantCount = 0,
                .constants = NULL,
                .targetCount = 1,
                .targets = (WGPUColorTargetState[]){
                    {
                        .nextInChain = NULL,
                        .format = format,
                        .blend = NULL,
                        .writeMask = WGPUColorWriteMask_All,
                    },
                },
            },
        });
    if (!app->pipeline) {
        fprintf(stderr, "failed to create render pipeline\n");
        return false;
    }
    return true;
}

static bool triangle_app_init(TriangleApp *app) {
    *app = (TriangleApp){0};
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

    app->window = yawgpu_window_create(800, 600, "yawgpu triangle");
    if (!app->window) {
        fprintf(stderr, "failed to create window\n");
        return false;
    }
    if (!triangle_create_surface(app)) {
        return false;
    }

    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    if (!triangle_choose_surface_format(app->surface, app->context.adapter, &format)) {
        return false;
    }
    if (!triangle_create_pipeline(app, format)) {
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

static bool triangle_render_frame(const TriangleApp *app) {
    WGPUSurfaceTexture current = {0};
    wgpuSurfaceGetCurrentTexture(app->surface, &current);
    if (current.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
        !current.texture) {
        fprintf(stderr, "failed to acquire surface texture, status=%u\n", current.status);
        return false;
    }

    WGPUTextureView view = wgpuTextureCreateView(current.texture, NULL);
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
        app->context.device,
        &(WGPUCommandEncoderDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("triangle encoder"),
        });
    if (!view || !encoder) {
        fprintf(stderr, "failed to create frame resources\n");
        if (encoder) {
            wgpuCommandEncoderRelease(encoder);
        }
        if (view) {
            wgpuTextureViewRelease(view);
        }
        wgpuTextureRelease(current.texture);
        return false;
    }

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
        encoder,
        &(WGPURenderPassDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("triangle render pass"),
            .colorAttachmentCount = 1,
            .colorAttachments = (WGPURenderPassColorAttachment[]){
                {
                    .nextInChain = NULL,
                    .view = view,
                    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
                    .resolveTarget = NULL,
                    .loadOp = WGPULoadOp_Clear,
                    .storeOp = WGPUStoreOp_Store,
                    .clearValue = {
                        .r = 0.0,
                        .g = 0.0,
                        .b = 0.0,
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
    wgpuRenderPassEncoderSetPipeline(pass, app->pipeline);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(
        encoder,
        &(WGPUCommandBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("triangle commands"),
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

int main(void) {
    TriangleApp app = {0};
    if (!triangle_app_init(&app)) {
        triangle_app_destroy(&app);
        return EXIT_FAILURE;
    }

    for (uint32_t frame = 0; frame < 60 && !yawgpu_window_should_close(app.window); ++frame) {
        if (!triangle_render_frame(&app)) {
            triangle_app_destroy(&app);
            return EXIT_FAILURE;
        }
        yawgpu_window_poll_events();
    }

    triangle_app_destroy(&app);
    return EXIT_SUCCESS;
}
