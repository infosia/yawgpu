// tiled_deferred — multi-subpass deferred shading with the yawgpu `tiled` vendor
// extension (TBDR).
//
// Two subpasses in one render pass, sharing tile memory:
//   * subpass 0 (G-buffer): a full-screen triangle writes albedo (color 0) and a
//     packed normal (color 1).
//   * subpass 1 (lighting): reads albedo + normal back as INPUT ATTACHMENTS
//     (Metal `[[color(N)]]` programmable-blend tile reads; Vulkan `SubpassData`
//     INPUT_ATTACHMENT descriptors) and writes the shaded result to the final
//     target (color 2).
//
// On a TBDR GPU the G-buffer never leaves tile memory between the subpasses — the
// bandwidth win of deferred shading.
//
//   * Default: opens a window and presents the shaded result every frame.
//   * `--verify`: renders one frame offscreen, reads it back, writes
//     `tiled_deferred.png`, and prints the center pixel.
//
// Backend via the YAWGPU_BACKEND env var (metal / vulkan). Requires libyawgpu
// built with the `tiled` cargo feature (CMake: -DYAWGPU_TILED=ON).

#include "framework.h"
#include "stb_image_write.h"

#include <stdint.h>

enum {
    BYTES_PER_PIXEL = 4,
    ROW_ALIGN = 256, // CopyTextureToBuffer requires 256-byte-aligned rows.
};

static WGPUStringView sv(const char *s) { return yawgpu_string_view(s); }
static WGPUStringView sv_empty(void) {
    WGPUStringView v = {.data = NULL, .length = 0};
    return v;
}

// A G-buffer + the pass layout + the two subpass pipelines, all sized/typed for a
// given render extent and final-target format. Rebuilt if the window resizes.
typedef struct {
    WGPUTexture albedo, normal;
    WGPUTextureView albedo_view, normal_view;
    YaWGPUSubpassPassLayout layout;
    WGPURenderPipeline gbuffer_pipeline, lighting_pipeline;
    uint32_t width, height;
} Deferred;

static WGPUTexture make_target(WGPUDevice device, uint32_t w, uint32_t h,
                               WGPUTextureFormat format, WGPUTextureUsage usage) {
    WGPUTextureDescriptor d = {
        .label = sv_empty(),
        .usage = usage,
        .dimension = WGPUTextureDimension_2D,
        .size = {.width = w, .height = h, .depthOrArrayLayers = 1},
        .format = format,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    WGPUTexture t = wgpuDeviceCreateTexture(device, &d);
    if (!t) {
        fprintf(stderr, "failed to create render target\n");
        exit(1);
    }
    return t;
}

// Builds a global-slot-indexed color-target array. `written` lists the GLOBAL
// color slots this subpass writes; every other slot up to the max is disabled.
// Input-attachment slots are NOT listed — the core supplies them per backend
// (the portable contract). The written slots all use `format`.
static size_t build_targets(const uint32_t *written, size_t n_written,
                            WGPUTextureFormat format, WGPUColorTargetState *out) {
    uint32_t max_slot = 0;
    for (size_t i = 0; i < n_written; i++) {
        if (written[i] > max_slot) max_slot = written[i];
    }
    size_t count = (size_t)max_slot + 1;
    for (size_t slot = 0; slot < count; slot++) {
        bool is_written = false;
        for (size_t i = 0; i < n_written; i++) {
            if (written[i] == slot) is_written = true;
        }
        out[slot] = (WGPUColorTargetState){
            .format = is_written ? format : WGPUTextureFormat_Undefined,
            .blend = NULL,
            .writeMask = is_written ? WGPUColorWriteMask_All : WGPUColorWriteMask_None,
        };
    }
    return count;
}

static WGPURenderPipeline make_pipeline(WGPUDevice device,
                                        YaWGPUSubpassPassLayout layout,
                                        uint32_t subpass_index, WGPUShaderModule module,
                                        const uint32_t *written, size_t n_written,
                                        WGPUTextureFormat written_format) {
    WGPUColorTargetState targets[8];
    size_t n_targets = build_targets(written, n_written, written_format, targets);
    WGPUFragmentState fragment = {
        .module = module,
        .entryPoint = sv("fs"),
        .targetCount = n_targets,
        .targets = targets,
    };
    WGPURenderPipelineDescriptor base = {
        .label = sv_empty(),
        .layout = NULL, // auto layout (input attachments derived from reflection)
        .vertex = {.module = module, .entryPoint = sv("vs")},
        .primitive = {.topology = WGPUPrimitiveTopology_TriangleList},
        .multisample = {.count = 1, .mask = 0xFFFFFFFFu},
        .fragment = &fragment,
    };
    YaWGPUSubpassRenderPipelineDescriptor d = {
        .base = base, .passLayout = layout, .subpassIndex = subpass_index};
    WGPURenderPipeline p = yawgpuDeviceCreateSubpassRenderPipeline(device, &d);
    if (!p) {
        fprintf(stderr, "failed to create subpass pipeline %u\n", subpass_index);
        exit(1);
    }
    return p;
}

// Builds the G-buffer textures, the 2-subpass pass layout, and both subpass
// pipelines for the given extent + final-target format.
static Deferred deferred_create(WGPUDevice device, uint32_t w, uint32_t h,
                                WGPUTextureFormat final_format,
                                WGPUShaderModule gbuffer_module,
                                WGPUShaderModule lighting_module) {
    Deferred df = {.width = w, .height = h};
    // The albedo + normal G-buffers are pure subpass intermediates: written in
    // subpass 0, consumed as input attachments in subpass 1, never stored. Mark
    // them TransientAttachment so the backend keeps them on-tile / memoryless
    // (Metal MTLStorageMode::Memoryless, Vulkan LAZILY_ALLOCATED) — the TBDR
    // bandwidth payoff: they never spill to DRAM.
    df.albedo = make_target(device, w, h, WGPUTextureFormat_RGBA8Unorm,
                            WGPUTextureUsage_RenderAttachment |
                                WGPUTextureUsage_TransientAttachment);
    df.normal = make_target(device, w, h, WGPUTextureFormat_RGBA8Unorm,
                            WGPUTextureUsage_RenderAttachment |
                                WGPUTextureUsage_TransientAttachment);
    df.albedo_view = wgpuTextureCreateView(df.albedo, NULL);
    df.normal_view = wgpuTextureCreateView(df.normal, NULL);

    YaWGPUAttachmentLayout colors[3] = {
        {.format = WGPUTextureFormat_RGBA8Unorm, .sampleCount = 1},
        {.format = WGPUTextureFormat_RGBA8Unorm, .sampleCount = 1},
        {.format = final_format, .sampleCount = 1},
    };
    uint32_t gbuffer_colors[2] = {0, 1};
    uint32_t lighting_colors[1] = {2};
    YaWGPUSubpassInputAttachment inputs[2] = {
        {.group = 0, .binding = 0, .sourceSubpass = 0, .sourceAttachment = 0},
        {.group = 0, .binding = 1, .sourceSubpass = 0, .sourceAttachment = 1},
    };
    YaWGPUSubpassLayout subpasses[2] = {
        {.colorAttachmentIndices = gbuffer_colors,
         .colorAttachmentIndexCount = 2,
         .inputAttachments = NULL,
         .inputAttachmentCount = 0},
        {.colorAttachmentIndices = lighting_colors,
         .colorAttachmentIndexCount = 1,
         .inputAttachments = inputs,
         .inputAttachmentCount = 2},
    };
    YaWGPUSubpassDependency dependency = {
        .srcSubpass = 0,
        .dstSubpass = 1,
        .dependencyType = YaWGPUSubpassDependencyType_ColorToInput,
        .byRegion = 1,
    };
    YaWGPUSubpassPassLayoutDescriptor layout_desc = {
        .label = sv_empty(),
        .colorAttachments = colors,
        .colorAttachmentCount = 3,
        .subpasses = subpasses,
        .subpassCount = 2,
        .dependencies = &dependency,
        .dependencyCount = 1,
    };
    df.layout = yawgpuDeviceCreateSubpassPassLayout(device, &layout_desc);
    if (!df.layout) {
        fprintf(stderr, "failed to create subpass pass layout\n");
        exit(1);
    }

    uint32_t gbuffer_written[2] = {0, 1};
    uint32_t lighting_written[1] = {2};
    df.gbuffer_pipeline = make_pipeline(device, df.layout, 0, gbuffer_module,
                                        gbuffer_written, 2, WGPUTextureFormat_RGBA8Unorm);
    df.lighting_pipeline = make_pipeline(device, df.layout, 1, lighting_module,
                                         lighting_written, 1, final_format);
    return df;
}

static void deferred_destroy(Deferred *df) {
    wgpuRenderPipelineRelease(df->lighting_pipeline);
    wgpuRenderPipelineRelease(df->gbuffer_pipeline);
    yawgpuSubpassPassLayoutRelease(df->layout);
    wgpuTextureViewRelease(df->normal_view);
    wgpuTextureViewRelease(df->albedo_view);
    wgpuTextureRelease(df->normal);
    wgpuTextureRelease(df->albedo);
}

// Records the two-subpass deferred pass into `encoder`, writing the final result
// into `final_view`.
static void record_deferred(WGPUCommandEncoder encoder, const Deferred *df,
                            WGPUTextureView final_view) {
    YaWGPUSubpassColorAttachment attachments[3] = {
        // Transient G-buffers: consumed in-pass by subpass 1, so storeOp = Discard
        // (a memoryless attachment has no DRAM to store into).
        {.view = df->albedo_view,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Discard,
         .clearValue = {0, 0, 0, 1}},
        {.view = df->normal_view,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Discard,
         .clearValue = {0, 0, 0, 1}},
        {.view = final_view,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Store,
         .clearValue = {0, 0, 0, 1}},
    };
    YaWGPUSubpassRenderPassDescriptor pass_desc = {
        .label = sv_empty(),
        .passLayout = df->layout,
        .extent = {.width = df->width, .height = df->height, .depthOrArrayLayers = 1},
        .colorAttachments = attachments,
        .colorAttachmentCount = 3,
    };
    YaWGPUSubpassRenderPassEncoder pass =
        yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &pass_desc);
    yawgpuSubpassRenderPassEncoderSetPipeline(pass, df->gbuffer_pipeline);
    yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpuSubpassRenderPassEncoderNextSubpass(pass);
    yawgpuSubpassRenderPassEncoderSetPipeline(pass, df->lighting_pipeline);
    // The input attachments (group 0) are bound implicitly by the pass.
    yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpuSubpassRenderPassEncoderEnd(pass);
    yawgpuSubpassRenderPassEncoderRelease(pass);
}

typedef struct {
    bool called;
    WGPUMapAsyncStatus status;
} MapState;

static void map_callback(WGPUMapAsyncStatus status, WGPUStringView message,
                         void *ud1, void *ud2) {
    YAWGPU_UNUSED(message);
    YAWGPU_UNUSED(ud2);
    MapState *s = (MapState *)ud1;
    s->called = true;
    s->status = status;
}

// --- offscreen: render one frame to an RGBA8 texture, read back, write PNG ----
static int run_offscreen(YawgpuContext *ctx, WGPUQueue queue,
                         WGPUShaderModule gbuffer_module,
                         WGPUShaderModule lighting_module) {
    const uint32_t W = 256, H = 256;
    Deferred df = deferred_create(ctx->device, W, H, WGPUTextureFormat_RGBA8Unorm,
                                  gbuffer_module, lighting_module);
    WGPUTexture final = make_target(
        ctx->device, W, H, WGPUTextureFormat_RGBA8Unorm,
        WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);
    WGPUTextureView final_view = wgpuTextureCreateView(final, NULL);

    uint32_t unpadded = W * BYTES_PER_PIXEL;
    uint32_t padded_row = unpadded + (ROW_ALIGN - (unpadded % ROW_ALIGN)) % ROW_ALIGN;
    uint64_t buffer_size = (uint64_t)padded_row * H;
    WGPUBufferDescriptor bd = {
        .label = sv_empty(),
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        .size = buffer_size,
    };
    WGPUBuffer readback = wgpuDeviceCreateBuffer(ctx->device, &bd);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);
    record_deferred(encoder, &df, final_view);
    WGPUTexelCopyTextureInfo src = {.texture = final, .aspect = WGPUTextureAspect_All};
    WGPUTexelCopyBufferInfo dstb = {
        .layout = {.bytesPerRow = padded_row, .rowsPerImage = H}, .buffer = readback};
    WGPUExtent3D extent = {.width = W, .height = H, .depthOrArrayLayers = 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dstb, &extent);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(queue, 1, &cmd);

    MapState ms = {0};
    WGPUBufferMapCallbackInfo cb = {.mode = WGPUCallbackMode_AllowProcessEvents,
                                    .callback = map_callback,
                                    .userdata1 = &ms};
    WGPUFuture fut = wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0,
                                        (size_t)buffer_size, cb);
    yawgpu_wait_for_future(ctx->instance, fut);
    if (!ms.called || ms.status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "buffer map failed\n");
        return 1;
    }
    const uint8_t *mapped =
        (const uint8_t *)wgpuBufferGetConstMappedRange(readback, 0, (size_t)buffer_size);
    uint8_t *pixels = (uint8_t *)malloc((size_t)W * H * BYTES_PER_PIXEL);
    for (uint32_t y = 0; y < H; y++) {
        memcpy(pixels + (size_t)y * W * BYTES_PER_PIXEL,
               mapped + (size_t)y * padded_row, (size_t)W * BYTES_PER_PIXEL);
    }
    const uint8_t *center = pixels + ((size_t)(H / 2) * W + (W / 2)) * BYTES_PER_PIXEL;
    printf("center pixel (shaded from the input-attachment G-buffer read): "
           "(%u, %u, %u, %u)\n",
           center[0], center[1], center[2], center[3]);
    stbi_write_png("tiled_deferred.png", W, H, BYTES_PER_PIXEL, pixels,
                   W * BYTES_PER_PIXEL);
    printf("wrote tiled_deferred.png (%ux%u)\n", W, H);

    free(pixels);
    wgpuBufferUnmap(readback);
    wgpuBufferRelease(readback);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
    wgpuTextureViewRelease(final_view);
    wgpuTextureRelease(final);
    deferred_destroy(&df);
    return (int)yawgpu_uncaptured_error_count();
}

// --- windowed: present the shaded result every frame -------------------------
static bool choose_surface_format(WGPUSurface surface, WGPUAdapter adapter,
                                  WGPUTextureFormat *out) {
    WGPUSurfaceCapabilities caps = {0};
    if (wgpuSurfaceGetCapabilities(surface, adapter, &caps) != WGPUStatus_Success ||
        caps.formatCount == 0) {
        return false;
    }
    *out = caps.formats[0];
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    return true;
}

static int run_windowed(YawgpuContext *ctx, WGPUQueue queue,
                        WGPUShaderModule gbuffer_module,
                        WGPUShaderModule lighting_module) {
    YawgpuWindow *window =
        yawgpu_window_create(640, 640, "yawgpu tiled_deferred (TBDR deferred shading)");
    if (!window) {
        fprintf(stderr, "failed to create window (windowed examples need GLFW)\n");
        return 1;
    }
    WGPUSurface surface =
        yawgpu_window_create_surface(ctx->instance, window, "tiled_deferred surface");
    WGPUTextureFormat format;
    if (!surface || !choose_surface_format(surface, ctx->adapter, &format)) {
        fprintf(stderr, "failed to create/inspect surface\n");
        return 1;
    }
    int w = 0, h = 0;
    yawgpu_window_framebuffer_size(window, &w, &h);
    wgpuSurfaceConfigure(
        surface, &(WGPUSurfaceConfiguration){
                     .device = ctx->device,
                     .format = format,
                     .usage = WGPUTextureUsage_RenderAttachment,
                     .width = (uint32_t)w,
                     .height = (uint32_t)h,
                     .alphaMode = WGPUCompositeAlphaMode_Opaque,
                     .presentMode = WGPUPresentMode_Fifo,
                 });

    Deferred df = deferred_create(ctx->device, (uint32_t)w, (uint32_t)h, format,
                                  gbuffer_module, lighting_module);

    printf("presenting deferred-shaded frames; close the window to exit\n");
    while (!yawgpu_window_should_close(window)) {
        WGPUSurfaceTexture current = {0};
        wgpuSurfaceGetCurrentTexture(surface, &current);
        if (current.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
            !current.texture) {
            yawgpu_window_poll_events();
            continue;
        }
        WGPUTextureView view = wgpuTextureCreateView(current.texture, NULL);
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);
        record_deferred(encoder, &df, view);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuSurfacePresent(surface);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
        wgpuTextureViewRelease(view);
        wgpuTextureRelease(current.texture);
        yawgpu_window_poll_events();
    }

    deferred_destroy(&df);
    wgpuSurfaceRelease(surface);
    yawgpu_window_destroy(window);
    return (int)yawgpu_uncaptured_error_count();
}

int main(int argc, char **argv) {
    if (argc > 0) yawgpu_set_argv0(argv[0]);
    bool verify = (argc > 1 && strcmp(argv[1], "--verify") == 0);

    YawgpuContext ctx = yawgpu_context_create();
    if (!ctx.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        return 1;
    }
    WGPUQueue queue = wgpuDeviceGetQueue(ctx.device);
    WGPUShaderModule gbuffer_module = yawgpu_load_wgsl_shader(ctx.device, "gbuffer.wgsl");
    WGPUShaderModule lighting_module = yawgpu_load_wgsl_shader(ctx.device, "lighting.wgsl");
    if (!gbuffer_module || !lighting_module) {
        fprintf(stderr, "failed to load shaders\n");
        return 1;
    }

    int rc = verify ? run_offscreen(&ctx, queue, gbuffer_module, lighting_module)
                    : run_windowed(&ctx, queue, gbuffer_module, lighting_module);

    wgpuShaderModuleRelease(lighting_module);
    wgpuShaderModuleRelease(gbuffer_module);
    wgpuQueueRelease(queue);
    yawgpu_context_release(&ctx);
    if (rc != 0) fprintf(stderr, "%d uncaptured device error(s)\n", rc);
    return rc;
}
