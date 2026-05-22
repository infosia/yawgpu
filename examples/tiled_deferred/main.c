// tiled_deferred - exercise yawgpu's tiled subpass API from C.
//
// With YAWGPU_HAS_TILED this renders a red G-buffer in subpass 0, reads it
// as a subpass input in subpass 1, swizzles red to green, copies the offscreen
// output texture to a readback buffer, and writes tiled_deferred.png.

#include "framework.h"

#if !defined(YAWGPU_HAS_TILED)

int main(void) {
    puts("tiled extension not enabled");
    return EXIT_SUCCESS;
}

#else

#include "stb_image_write.h"

enum {
    IMAGE_WIDTH = 16,
    IMAGE_HEIGHT = 16,
    COPY_BYTES_PER_ROW_ALIGNMENT = 256,
    BYTES_PER_PIXEL = 4,
};

typedef struct BufferDimensions {
    uint32_t width;
    uint32_t height;
    uint32_t unpadded_bytes_per_row;
    uint32_t padded_bytes_per_row;
} BufferDimensions;

typedef struct MapState {
    WGPUMapAsyncStatus status;
    bool called;
} MapState;

typedef struct QueueWorkDoneState {
    WGPUQueueWorkDoneStatus status;
    bool called;
} QueueWorkDoneState;

typedef struct TiledDeferredApp {
    YawgpuContext context;
    WGPUQueue queue;
    WGPUBuffer output_buffer;
    WGPUTexture gbuffer_texture;
    WGPUTexture output_texture;
    WGPUTextureView gbuffer_view;
    WGPUTextureView output_view;
    WGPUShaderModule write_module;
    WGPUShaderModule load_module;
    YaWGPUSubpassPassLayout pass_layout;
    WGPURenderPipeline write_pipeline;
    WGPURenderPipeline load_pipeline;
    YaWGPUAttachmentLayout attachment_layouts[2];
    uint32_t subpass0_color;
    uint32_t subpass1_color;
    YaWGPUSubpassInputAttachment input_attachment;
    YaWGPUSubpassLayoutDesc subpass_layouts[2];
    YaWGPUSubpassDependency dependency;
    YaWGPUColorAttachmentBinding color_bindings[2];
    YaWGPUSubpassRenderPassDescriptor pass_descriptor;
    WGPUColorTargetState color_targets[2];
    WGPUFragmentState fragment_states[2];
    WGPURenderPipelineDescriptor pipeline_bases[2];
    YaWGPUSubpassRenderPipelineDescriptor pipeline_descriptors[2];
    BufferDimensions dimensions;
    uint64_t buffer_size;
    bool output_buffer_mapped;
    unsigned int initial_error_count;
} TiledDeferredApp;

static const char *WRITE_WGSL =
    "struct VertexOut {\n"
    "    @builtin(position) position: vec4<f32>,\n"
    "}\n"
    "\n"
    "@vertex\n"
    "fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOut {\n"
    "    let positions = array<vec2<f32>, 3>(\n"
    "        vec2<f32>(-1.0, -1.0),\n"
    "        vec2<f32>(3.0, -1.0),\n"
    "        vec2<f32>(-1.0, 3.0),\n"
    "    );\n"
    "    var out: VertexOut;\n"
    "    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);\n"
    "    return out;\n"
    "}\n"
    "\n"
    "@fragment\n"
    "fn fs() -> @location(0) vec4<f32> {\n"
    "    return vec4<f32>(1.0, 0.0, 0.0, 1.0);\n"
    "}\n";

static const char *LOAD_WGSL =
    "struct VertexOut {\n"
    "    @builtin(position) position: vec4<f32>,\n"
    "}\n"
    "\n"
    "@group(0) @binding(0) var gbuffer: subpass_input<f32>;\n"
    "\n"
    "@vertex\n"
    "fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOut {\n"
    "    let positions = array<vec2<f32>, 3>(\n"
    "        vec2<f32>(-1.0, -1.0),\n"
    "        vec2<f32>(3.0, -1.0),\n"
    "        vec2<f32>(-1.0, 3.0),\n"
    "    );\n"
    "    var out: VertexOut;\n"
    "    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);\n"
    "    return out;\n"
    "}\n"
    "\n"
    "@fragment\n"
    "fn fs() -> @location(0) vec4<f32> {\n"
    "    let loaded = subpassLoad(gbuffer);\n"
    "    return vec4<f32>(loaded.g, loaded.r, loaded.b, 1.0);\n"
    "}\n";

static uint32_t align_up_u32(uint32_t value, uint32_t alignment) {
    uint32_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    return value + alignment - remainder;
}

static BufferDimensions buffer_dimensions_create(uint32_t width, uint32_t height) {
    return (BufferDimensions){
        .width = width,
        .height = height,
        .unpadded_bytes_per_row = width * BYTES_PER_PIXEL,
        .padded_bytes_per_row = align_up_u32(width * BYTES_PER_PIXEL,
                                             COPY_BYTES_PER_ROW_ALIGNMENT),
    };
}

static WGPUStringView sized_string_view(const char *value) {
    return (WGPUStringView){
        .data = value,
        .length = value ? strlen(value) : 0,
    };
}

static void map_callback(WGPUMapAsyncStatus status,
                         WGPUStringView message,
                         void *userdata1,
                         void *userdata2) {
    YAWGPU_UNUSED(userdata2);
    MapState *state = (MapState *)userdata1;
    state->status = status;
    state->called = true;
    if (status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "buffer map failed: ");
        yawgpu_print_string_view(message);
        fprintf(stderr, "\n");
    }
}

static void queue_work_done_callback(WGPUQueueWorkDoneStatus status,
                                     WGPUStringView message,
                                     void *userdata1,
                                     void *userdata2) {
    YAWGPU_UNUSED(userdata2);
    QueueWorkDoneState *state = (QueueWorkDoneState *)userdata1;
    state->status = status;
    state->called = true;
    if (status != WGPUQueueWorkDoneStatus_Success) {
        fprintf(stderr, "queue work failed: ");
        yawgpu_print_string_view(message);
        fprintf(stderr, "\n");
    }
}

static WGPUShaderModule create_wgsl_module(WGPUDevice device,
                                           const char *label,
                                           const char *source) {
    WGPUShaderSourceWGSL wgsl = {
        .chain = {
            .next = NULL,
            .sType = WGPUSType_ShaderSourceWGSL,
        },
        .code = sized_string_view(source),
    };
    WGPUShaderModuleDescriptor descriptor = {
        .nextInChain = &wgsl.chain,
        .label = yawgpu_string_view(label),
    };
    return wgpuDeviceCreateShaderModule(device, &descriptor);
}

static YaWGPUSubpassPassLayout create_pass_layout(TiledDeferredApp *app) {
    app->attachment_layouts[0] = (YaWGPUAttachmentLayout){
        .format = WGPUTextureFormat_RGBA8Unorm,
        .sampleCount = 1,
    };
    app->attachment_layouts[1] = (YaWGPUAttachmentLayout){
        .format = WGPUTextureFormat_RGBA8Unorm,
        .sampleCount = 1,
    };
    app->subpass0_color = 0;
    app->subpass1_color = 1;
    app->input_attachment = (YaWGPUSubpassInputAttachment){
        .group = 0,
        .binding = 0,
        .sourceSubpass = 0,
        .sourceAttachment = 0,
    };
    app->subpass_layouts[0] = (YaWGPUSubpassLayoutDesc){
        .colorAttachmentIndices = &app->subpass0_color,
        .colorAttachmentIndexCount = 1,
        .usesDepthStencil = false,
        .inputAttachments = NULL,
        .inputAttachmentCount = 0,
    };
    app->subpass_layouts[1] = (YaWGPUSubpassLayoutDesc){
        .colorAttachmentIndices = &app->subpass1_color,
        .colorAttachmentIndexCount = 1,
        .usesDepthStencil = false,
        .inputAttachments = &app->input_attachment,
        .inputAttachmentCount = 1,
    };
    app->dependency = (YaWGPUSubpassDependency){
        .srcSubpass = 0,
        .dstSubpass = 1,
        .dependencyType = YaWGPUSubpassDependencyType_ColorToInput,
        .byRegion = true,
    };
    YaWGPUSubpassPassLayoutDescriptor descriptor = {
        .nextInChain = NULL,
        .label = yawgpu_string_view("tiled deferred pass layout"),
        .colorAttachments = app->attachment_layouts,
        .colorAttachmentCount = 2,
        .depthStencilAttachment = {
            .format = WGPUTextureFormat_Undefined,
            .sampleCount = 1,
        },
        .subpasses = app->subpass_layouts,
        .subpassCount = 2,
        .dependencies = &app->dependency,
        .dependencyCount = 1,
    };
    return yawgpuDeviceCreateSubpassPassLayout(app->context.device, &descriptor);
}

static WGPURenderPipeline create_subpass_pipeline(TiledDeferredApp *app,
                                                  uint32_t subpass_index,
                                                  WGPUShaderModule module,
                                                  const char *label) {
    app->color_targets[subpass_index] = (WGPUColorTargetState){
        .nextInChain = NULL,
        .format = WGPUTextureFormat_RGBA8Unorm,
        .blend = NULL,
        .writeMask = WGPUColorWriteMask_All,
    };
    app->fragment_states[subpass_index] = (WGPUFragmentState){
        .nextInChain = NULL,
        .module = module,
        .entryPoint = sized_string_view("fs"),
        .constantCount = 0,
        .constants = NULL,
        .targetCount = 1,
        .targets = &app->color_targets[subpass_index],
    };
    app->pipeline_bases[subpass_index] = (WGPURenderPipelineDescriptor){
        .nextInChain = NULL,
        .label = yawgpu_string_view(label),
        .layout = NULL,
        .vertex = {
            .nextInChain = NULL,
            .module = module,
            .entryPoint = sized_string_view("vs"),
            .constantCount = 0,
            .constants = NULL,
            .bufferCount = 0,
            .buffers = NULL,
        },
        .primitive = {
            .nextInChain = NULL,
            .topology = WGPUPrimitiveTopology_TriangleList,
            .stripIndexFormat = WGPUIndexFormat_Undefined,
            .frontFace = WGPUFrontFace_Undefined,
            .cullMode = WGPUCullMode_Undefined,
            .unclippedDepth = false,
        },
        .depthStencil = NULL,
        .multisample = WGPU_MULTISAMPLE_STATE_INIT,
        .fragment = &app->fragment_states[subpass_index],
    };
    app->pipeline_descriptors[subpass_index] = (YaWGPUSubpassRenderPipelineDescriptor){
        .nextInChain = NULL,
        .base = app->pipeline_bases[subpass_index],
        .passLayout = app->pass_layout,
        .subpassIndex = subpass_index,
    };
    return yawgpuDeviceCreateSubpassRenderPipeline(app->context.device,
                                                   &app->pipeline_descriptors[subpass_index]);
}

static YaWGPUColorAttachmentBinding subpass_color_binding(WGPUTextureView view) {
    return (YaWGPUColorAttachmentBinding){
        .kind = YaWGPUSubpassAttachmentKind_Persistent,
        .view = view,
        .resolveTarget = NULL,
        .transient = NULL,
        .loadOp = WGPULoadOp_Clear,
        .storeOp = WGPUStoreOp_Store,
        .clearValue = {
            .r = 0.0,
            .g = 0.0,
            .b = 0.0,
            .a = 1.0,
        },
    };
}

static void tiled_deferred_app_destroy(TiledDeferredApp *app) {
    if (app->output_buffer_mapped) {
        wgpuBufferUnmap(app->output_buffer);
    }
    if (app->load_pipeline) {
        wgpuRenderPipelineRelease(app->load_pipeline);
    }
    if (app->write_pipeline) {
        wgpuRenderPipelineRelease(app->write_pipeline);
    }
    if (app->pass_layout) {
        yawgpuSubpassPassLayoutRelease(app->pass_layout);
    }
    if (app->load_module) {
        wgpuShaderModuleRelease(app->load_module);
    }
    if (app->write_module) {
        wgpuShaderModuleRelease(app->write_module);
    }
    if (app->output_view) {
        wgpuTextureViewRelease(app->output_view);
    }
    if (app->gbuffer_view) {
        wgpuTextureViewRelease(app->gbuffer_view);
    }
    if (app->output_texture) {
        wgpuTextureRelease(app->output_texture);
    }
    if (app->gbuffer_texture) {
        wgpuTextureRelease(app->gbuffer_texture);
    }
    if (app->output_buffer) {
        wgpuBufferRelease(app->output_buffer);
    }
    if (app->queue) {
        wgpuQueueRelease(app->queue);
    }
    yawgpu_context_release(&app->context);
    *app = (TiledDeferredApp){0};
}

static bool create_textures(TiledDeferredApp *app, WGPUExtent3D texture_size) {
    WGPUTextureDescriptor gbuffer_descriptor = {
        .nextInChain = NULL,
        .label = yawgpu_string_view("tiled deferred gbuffer"),
        .usage = WGPUTextureUsage_RenderAttachment,
        .dimension = WGPUTextureDimension_2D,
        .size = texture_size,
        .format = WGPUTextureFormat_RGBA8Unorm,
        .mipLevelCount = 1,
        .sampleCount = 1,
        .viewFormatCount = 0,
        .viewFormats = NULL,
    };
    WGPUTextureDescriptor output_descriptor = gbuffer_descriptor;
    output_descriptor.label = yawgpu_string_view("tiled deferred output");
    output_descriptor.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;

    app->gbuffer_texture = wgpuDeviceCreateTexture(app->context.device, &gbuffer_descriptor);
    app->output_texture = wgpuDeviceCreateTexture(app->context.device, &output_descriptor);
    if (!app->gbuffer_texture || !app->output_texture) {
        fprintf(stderr, "failed to create tiled deferred textures\n");
        return false;
    }

    app->gbuffer_view = wgpuTextureCreateView(app->gbuffer_texture, NULL);
    app->output_view = wgpuTextureCreateView(app->output_texture, NULL);
    if (!app->gbuffer_view || !app->output_view) {
        fprintf(stderr, "failed to create tiled deferred texture views\n");
        return false;
    }

    return true;
}

static bool value_near_u8(uint8_t actual, uint8_t expected) {
    int delta = (int)actual - (int)expected;
    return delta >= -1 && delta <= 1;
}

static bool verify_center_pixel(const TiledDeferredApp *app, const uint8_t *pixels) {
    uint32_t x = app->dimensions.width / 2;
    uint32_t y = app->dimensions.height / 2;
    size_t offset = (size_t)y * app->dimensions.padded_bytes_per_row +
                    (size_t)x * BYTES_PER_PIXEL;
    uint8_t rgba[4] = {
        pixels[offset],
        pixels[offset + 1],
        pixels[offset + 2],
        pixels[offset + 3],
    };
    bool ok = value_near_u8(rgba[0], 0) && value_near_u8(rgba[1], 255) &&
              value_near_u8(rgba[2], 0) && value_near_u8(rgba[3], 255);
    printf("tiled_deferred: center pixel RGBA=(%u,%u,%u,%u) %s\n",
           (unsigned int)rgba[0],
           (unsigned int)rgba[1],
           (unsigned int)rgba[2],
           (unsigned int)rgba[3],
           ok ? "OK" : "FAILED");
    return ok;
}

static bool require_tiled_backend(WGPUAdapter adapter) {
    WGPUAdapterInfo info = {0};
    if (wgpuAdapterGetInfo(adapter, &info) != WGPUStatus_Success) {
        fprintf(stderr, "failed to query adapter info\n");
        return false;
    }
    WGPUBackendType backend = info.backendType;
    wgpuAdapterInfoFreeMembers(info);
    if (backend != WGPUBackendType_Metal && backend != WGPUBackendType_Vulkan) {
        fprintf(stderr, "tiled_deferred requires Metal or native Vulkan; selected backend type=%u\n",
                (unsigned int)backend);
        return false;
    }
    if (!wgpuAdapterHasFeature(adapter, YaWGPUFeatureName_MultiSubpass)) {
        fprintf(stderr, "tiled_deferred requires YaWGPUFeatureName_MultiSubpass\n");
        return false;
    }
    return true;
}

static bool tiled_deferred_app_init(TiledDeferredApp *app) {
    *app = (TiledDeferredApp){0};
    app->context = yawgpu_context_create();
    if (!app->context.instance || !app->context.adapter || !app->context.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        return false;
    }
    app->initial_error_count = yawgpu_uncaptured_error_count();
    if (!require_tiled_backend(app->context.adapter)) {
        return false;
    }

    app->queue = wgpuDeviceGetQueue(app->context.device);
    if (!app->queue) {
        fprintf(stderr, "failed to get device queue\n");
        return false;
    }

    app->dimensions = buffer_dimensions_create(IMAGE_WIDTH, IMAGE_HEIGHT);
    app->buffer_size = (uint64_t)app->dimensions.padded_bytes_per_row * app->dimensions.height;
    app->output_buffer = wgpuDeviceCreateBuffer(
        app->context.device,
        &(WGPUBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("tiled deferred output buffer"),
            .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
            .size = app->buffer_size,
            .mappedAtCreation = false,
        });
    if (!app->output_buffer) {
        fprintf(stderr, "failed to create tiled deferred output buffer\n");
        return false;
    }

    WGPUExtent3D texture_size = {
        .width = app->dimensions.width,
        .height = app->dimensions.height,
        .depthOrArrayLayers = 1,
    };
    if (!create_textures(app, texture_size)) {
        return false;
    }

    app->pass_layout = create_pass_layout(app);
    app->write_module = create_wgsl_module(app->context.device,
                                           "tiled deferred write shader",
                                           WRITE_WGSL);
    app->load_module = create_wgsl_module(app->context.device,
                                          "tiled deferred load shader",
                                          LOAD_WGSL);
    if (!app->pass_layout || !app->write_module || !app->load_module) {
        fprintf(stderr, "failed to create tiled deferred layout or shaders\n");
        return false;
    }

    app->write_pipeline = create_subpass_pipeline(app,
                                                  0,
                                                  app->write_module,
                                                  "tiled deferred gbuffer pipeline");
    app->load_pipeline = create_subpass_pipeline(app,
                                                 1,
                                                 app->load_module,
                                                 "tiled deferred output pipeline");
    if (!app->write_pipeline || !app->load_pipeline) {
        fprintf(stderr, "failed to create tiled deferred pipelines\n");
        return false;
    }

    return true;
}

static bool record_tiled_pass(TiledDeferredApp *app, WGPUCommandEncoder encoder) {
    app->color_bindings[0] = subpass_color_binding(app->gbuffer_view);
    app->color_bindings[1] = subpass_color_binding(app->output_view);
    app->pass_descriptor = (YaWGPUSubpassRenderPassDescriptor){
        .nextInChain = NULL,
        .label = yawgpu_string_view("tiled deferred pass"),
        .passLayout = app->pass_layout,
        .extent = {
            .width = app->dimensions.width,
            .height = app->dimensions.height,
            .depthOrArrayLayers = 1,
        },
        .colorAttachments = app->color_bindings,
        .colorAttachmentCount = 2,
        .depthStencilAttachment = NULL,
    };
    YaWGPUSubpassRenderPassEncoder pass =
        yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &app->pass_descriptor);
    if (!pass) {
        fprintf(stderr, "failed to begin tiled deferred pass\n");
        return false;
    }

    yawgpuSubpassRenderPassEncoderSetPipeline(pass, app->write_pipeline);
    yawgpuSubpassRenderPassEncoderSetViewport(pass,
                                              0.0f,
                                              0.0f,
                                              (float)app->dimensions.width,
                                              (float)app->dimensions.height,
                                              0.0f,
                                              1.0f);
    yawgpuSubpassRenderPassEncoderSetScissorRect(pass,
                                                 0,
                                                 0,
                                                 app->dimensions.width,
                                                 app->dimensions.height);
    yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpuSubpassRenderPassEncoderNextSubpass(pass);
    yawgpuSubpassRenderPassEncoderSetPipeline(pass, app->load_pipeline);
    yawgpuSubpassRenderPassEncoderSetViewport(pass,
                                              0.0f,
                                              0.0f,
                                              (float)app->dimensions.width,
                                              (float)app->dimensions.height,
                                              0.0f,
                                              1.0f);
    yawgpuSubpassRenderPassEncoderSetScissorRect(pass,
                                                 0,
                                                 0,
                                                 app->dimensions.width,
                                                 app->dimensions.height);
    yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpuSubpassRenderPassEncoderEnd(pass);
    yawgpuSubpassRenderPassEncoderRelease(pass);
    return true;
}

static bool readback_verify_and_write_png(TiledDeferredApp *app) {
    MapState map_state = {0};
    WGPUFuture map_future = wgpuBufferMapAsync(
        app->output_buffer,
        WGPUMapMode_Read,
        0,
        app->buffer_size,
        (WGPUBufferMapCallbackInfo){
            .nextInChain = NULL,
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = map_callback,
            .userdata1 = &map_state,
            .userdata2 = NULL,
        });
    yawgpu_wait_for_future(app->context.instance, map_future);
    if (!map_state.called || map_state.status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "readback map did not complete successfully\n");
        return false;
    }
    app->output_buffer_mapped = true;

    const uint8_t *pixels =
        (const uint8_t *)wgpuBufferGetConstMappedRange(app->output_buffer, 0, app->buffer_size);
    if (!pixels) {
        fprintf(stderr, "readback mapped range is null\n");
        return false;
    }

    bool pixel_ok = verify_center_pixel(app, pixels);
    bool wrote_png = stbi_write_png("tiled_deferred.png",
                                    (int)app->dimensions.width,
                                    (int)app->dimensions.height,
                                    BYTES_PER_PIXEL,
                                    pixels,
                                    (int)app->dimensions.padded_bytes_per_row) != 0;
    if (!wrote_png) {
        fprintf(stderr, "failed to write tiled_deferred.png\n");
        return false;
    }

    printf("wrote tiled_deferred.png (%ux%u, expected opaque green, bytesPerRow=%u padded to %u)\n",
           app->dimensions.width,
           app->dimensions.height,
           app->dimensions.unpadded_bytes_per_row,
           app->dimensions.padded_bytes_per_row);
    if (!pixel_ok) {
        return false;
    }
    if (yawgpu_uncaptured_error_count() != app->initial_error_count) {
        fprintf(stderr, "tiled_deferred: FAILED due to uncaptured device error\n");
        return false;
    }
    return true;
}

static bool tiled_deferred_app_run(TiledDeferredApp *app) {
    WGPUExtent3D texture_size = {
        .width = app->dimensions.width,
        .height = app->dimensions.height,
        .depthOrArrayLayers = 1,
    };
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(app->context.device, NULL);
    if (!encoder) {
        fprintf(stderr, "failed to create command encoder\n");
        return false;
    }

    if (!record_tiled_pass(app, encoder)) {
        wgpuCommandEncoderRelease(encoder);
        return false;
    }

    wgpuCommandEncoderCopyTextureToBuffer(
        encoder,
        &(WGPUTexelCopyTextureInfo){
            .texture = app->output_texture,
            .mipLevel = 0,
            .origin = {
                .x = 0,
                .y = 0,
                .z = 0,
            },
            .aspect = WGPUTextureAspect_All,
        },
        &(WGPUTexelCopyBufferInfo){
            .buffer = app->output_buffer,
            .layout = {
                .offset = 0,
                .bytesPerRow = app->dimensions.padded_bytes_per_row,
                .rowsPerImage = app->dimensions.height,
            },
        },
        &texture_size);

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    if (!commands) {
        fprintf(stderr, "failed to finish command encoder\n");
        wgpuCommandEncoderRelease(encoder);
        return false;
    }

    wgpuQueueSubmit(app->queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);
    QueueWorkDoneState queue_state = {0};
    WGPUFuture queue_future = wgpuQueueOnSubmittedWorkDone(
        app->queue,
        (WGPUQueueWorkDoneCallbackInfo){
            .nextInChain = NULL,
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = queue_work_done_callback,
            .userdata1 = &queue_state,
            .userdata2 = NULL,
        });
    yawgpu_wait_for_future(app->context.instance, queue_future);
    if (!queue_state.called || queue_state.status != WGPUQueueWorkDoneStatus_Success) {
        fprintf(stderr, "submitted work did not complete successfully\n");
        return false;
    }
    return readback_verify_and_write_png(app);
}

int main(void) {
    TiledDeferredApp app = {0};
    if (!tiled_deferred_app_init(&app)) {
        tiled_deferred_app_destroy(&app);
        return EXIT_FAILURE;
    }
    if (!tiled_deferred_app_run(&app)) {
        tiled_deferred_app_destroy(&app);
        return EXIT_FAILURE;
    }

    tiled_deferred_app_destroy(&app);
    return EXIT_SUCCESS;
}

#endif
