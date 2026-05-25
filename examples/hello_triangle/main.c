// hello_triangle — a windowed triangle fed from a vertex buffer.
//
// This is a port of Dawn's HelloTriangle sample. It draws the same
// RGB-corner gradient triangle as the `triangle` example, but the
// difference is instructive: here the vertex positions AND colors come
// from a real *vertex buffer* uploaded to the GPU, instead of being
// generated inside the vertex shader from the built-in vertex index.
// That means this example additionally shows:
//   * uploading interleaved vertex data (pos + color) to a Vertex-usage
//     buffer,
//   * describing the buffer's layout to the pipeline (stride + two
//     attributes at different byte offsets / shader locations),
//   * binding the buffer each frame with SetVertexBuffer before drawing.
// Everything else (surface, swapchain, per-frame loop) mirrors `triangle`.

#include "framework.h"

typedef struct HelloTriangleApp {
    YawgpuContext context;
    WGPUQueue queue;
    YawgpuWindow *window;
    WGPUSurface surface;
    WGPUShaderModule shader;
    WGPUBuffer vertex_buffer; // holds the three clip-space vertices below
    WGPUPipelineLayout pipeline_layout;
    WGPURenderPipeline pipeline;
} HelloTriangleApp;

// Three vertices, each a vec4 clip-space position (x, y, z, w) followed
// by a vec3 RGB color (r, g, b) — top-center red, bottom-left green,
// bottom-right blue. The pipeline interpolates the color across the
// surface to produce a per-vertex gradient.
static const float vertices[21] = {
     0.0f,  0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, // top:    red
    -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // BL:     green
     0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, // BR:     blue
};

static bool hello_triangle_choose_surface_format(WGPUSurface surface,
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

static void hello_triangle_app_destroy(HelloTriangleApp *app) {
    if (app->pipeline) {
        wgpuRenderPipelineRelease(app->pipeline);
    }
    if (app->pipeline_layout) {
        wgpuPipelineLayoutRelease(app->pipeline_layout);
    }
    if (app->vertex_buffer) {
        wgpuBufferRelease(app->vertex_buffer);
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
    *app = (HelloTriangleApp){0};
}

static bool hello_triangle_create_surface(HelloTriangleApp *app) {
    app->surface = yawgpu_window_create_surface(
        app->context.instance, app->window, "hello_triangle surface");
    return app->surface != NULL;
}

static bool hello_triangle_create_pipeline(HelloTriangleApp *app, WGPUTextureFormat format) {
    app->shader = yawgpu_load_wgsl_shader(app->context.device, "shader.wgsl");
    // Upload the vertex data once. Vertex usage makes it bindable as a
    // vertex buffer; CopyDst lets the initial contents be written into it.
    app->vertex_buffer = yawgpu_create_buffer_init(
        app->context.device,
        &(YawgpuBufferInitDescriptor){
            .label = "hello_triangle vertices",
            .usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst,
            .contents = vertices,
            .size = sizeof(vertices),
        });
    if (!app->shader || !app->vertex_buffer) {
        fprintf(stderr, "failed to create shader or vertex buffer\n");
        return false;
    }

    app->pipeline_layout = wgpuDeviceCreatePipelineLayout(
        app->context.device,
        &(WGPUPipelineLayoutDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("hello_triangle pipeline layout"),
            .bindGroupLayoutCount = 0,
            .bindGroupLayouts = NULL,
        });
    if (!app->pipeline_layout) {
        fprintf(stderr, "failed to create pipeline layout\n");
        return false;
    }

    // Describe how the pipeline reads the vertex buffer. Two attributes:
    //   * a vec4<f32> position at byte offset 0   → @location(0)
    //   * a vec3<f32> color    at byte offset 16  → @location(1)
    WGPUVertexAttribute vertex_attributes[2] = {
        {
            .format = WGPUVertexFormat_Float32x4,
            .offset = 0,
            .shaderLocation = 0,
        },
        {
            .format = WGPUVertexFormat_Float32x3,
            .offset = 4 * sizeof(float),
            .shaderLocation = 1,
        },
    };
    // Each vertex is 4 floats of position + 3 floats of color = 7 floats;
    // stepMode=Vertex advances one stride per vertex.
    WGPUVertexBufferLayout vertex_buffer_layout = {
        .arrayStride = 7 * sizeof(float),
        .stepMode = WGPUVertexStepMode_Vertex,
        .attributeCount = 2,
        .attributes = vertex_attributes,
    };
    app->pipeline = wgpuDeviceCreateRenderPipeline(
        app->context.device,
        &(WGPURenderPipelineDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("hello_triangle pipeline"),
            .layout = app->pipeline_layout,
            .vertex = {
                .nextInChain = NULL,
                .module = app->shader,
                .entryPoint = yawgpu_string_view("vs_main"),
                .constantCount = 0,
                .constants = NULL,
                // Unlike `triangle`, this pipeline takes one vertex buffer
                // (described by vertex_buffer_layout above).
                .bufferCount = 1,
                .buffers = &vertex_buffer_layout,
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

static bool hello_triangle_app_init(HelloTriangleApp *app) {
    *app = (HelloTriangleApp){0};
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

    app->window = yawgpu_window_create(800, 600, "yawgpu hello_triangle");
    if (!app->window) {
        fprintf(stderr, "failed to create window\n");
        return false;
    }
    if (!hello_triangle_create_surface(app)) {
        return false;
    }

    WGPUTextureFormat format = WGPUTextureFormat_Undefined;
    if (!hello_triangle_choose_surface_format(app->surface, app->context.adapter, &format)) {
        return false;
    }
    if (!hello_triangle_create_pipeline(app, format)) {
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

static bool hello_triangle_render_frame(const HelloTriangleApp *app) {
    WGPUSurfaceTexture current = {0};
    wgpuSurfaceGetCurrentTexture(app->surface, &current);
    if (current.status == WGPUSurfaceGetCurrentTextureStatus_Lost && !current.texture) {
        return true;
    }
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
            .label = yawgpu_string_view("hello_triangle encoder"),
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
            .label = yawgpu_string_view("hello_triangle render pass"),
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
    // Bind the pipeline, then bind the vertex buffer to slot 0 (matching
    // bufferCount=1 in the pipeline) covering its whole length, then draw
    // the 3 vertices it contains.
    wgpuRenderPassEncoderSetPipeline(pass, app->pipeline);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, app->vertex_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(
        encoder,
        &(WGPUCommandBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("hello_triangle commands"),
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
    HelloTriangleApp app = {0};
    if (!hello_triangle_app_init(&app)) {
        hello_triangle_app_destroy(&app);
        return EXIT_FAILURE;
    }

    for (uint32_t frame = 0; frame < 60 && !yawgpu_window_should_close(app.window); ++frame) {
        if (!hello_triangle_render_frame(&app)) {
            hello_triangle_app_destroy(&app);
            return EXIT_FAILURE;
        }
        yawgpu_window_poll_events();
    }

    hello_triangle_app_destroy(&app);
    return EXIT_SUCCESS;
}
