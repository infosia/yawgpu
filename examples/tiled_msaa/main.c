// tiled_msaa — per-sample MSAA subpass input with the yawgpu `tiled` vendor
// extension (Vulkan-only).
//
// Three subpasses in one render pass, all on-tile:
//   * subpass 0 (scene, 4x MSAA): draws a centred triangle whose diagonal edges
//     alias without MSAA, into a multisampled color attachment.
//   * subpass 1 (per-sample, 4x MSAA): reads the scene attachment PER SAMPLE via
//     `inputAttachmentLoad(scene, @builtin(sample_index))` — SampleId promotes the
//     fragment to per-sample invocation (Vulkan `sampleRateShading`) — applies a
//     per-sample tint, and writes a 4x MSAA intermediate.
//   * subpass 2 (resolve, single-sampled): reads the intermediate as a
//     multisampled input and averages its four samples in-shader (a custom
//     resolve), writing the single-sample output — anti-aliased, no hardware
//     resolve attachment.
//
// The multisampled input attachments are declared via an EXPLICIT pipeline layout
// whose input-attachment binding is `multisampled` (WGSL cannot express an input
// attachment's multisampled-ness in the type). MSAA subpass input is a Vulkan-only
// vendor surface; this example errors on other backends.
//
//   * Default: opens a window and presents the anti-aliased triangle every frame.
//   * `--verify`: renders one frame offscreen, reads it back, writes
//     `tiled_msaa.png`, and prints the center pixel.
//
// Backend via the YAWGPU_BACKEND env var (must be vulkan). Requires libyawgpu
// built with the `tiled` cargo feature (CMake: -DYAWGPU_TILED=ON).

#include "framework.h"
#include "stb_image_write.h"

#include <stdint.h>

enum {
    BYTES_PER_PIXEL = 4,
    ROW_ALIGN = 256, // CopyTextureToBuffer requires 256-byte-aligned rows.
    SAMPLE_COUNT = 4,
};

static WGPUStringView sv(const char *s) { return yawgpu_string_view(s); }
static WGPUStringView sv_empty(void) {
    WGPUStringView v = {.data = NULL, .length = 0};
    return v;
}

// Scene + intermediate (both 4x MSAA), the 3-subpass pass layout, the explicit
// input-attachment pipeline layout, and the three subpass pipelines. Rebuilt on
// resize.
typedef struct {
    WGPUTexture scene, hdr; // 4x MSAA subpass intermediates
    WGPUTextureView scene_view, hdr_view;
    YaWGPUSubpassPassLayout layout;
    WGPUPipelineLayout input_layout; // explicit layout: multisampled input attachment
    WGPURenderPipeline scene_pipeline, persample_pipeline, resolve_pipeline;
    uint32_t width, height;
} Msaa;

static WGPUTexture make_target(WGPUDevice device, uint32_t w, uint32_t h,
                               WGPUTextureFormat format, WGPUTextureUsage usage,
                               uint32_t sample_count) {
    WGPUTextureDescriptor d = {
        .label = sv_empty(),
        .usage = usage,
        .dimension = WGPUTextureDimension_2D,
        .size = {.width = w, .height = h, .depthOrArrayLayers = 1},
        .format = format,
        .mipLevelCount = 1,
        .sampleCount = sample_count,
    };
    WGPUTexture t = wgpuDeviceCreateTexture(device, &d);
    if (!t) {
        fprintf(stderr, "failed to create render target\n");
        exit(1);
    }
    return t;
}

// An explicit pipeline layout with a single fragment input-attachment binding at
// @group(0) @binding(0), declared `multisampled`. Required for MSAA subpass input:
// the reflection cannot know an input attachment is multisampled, so the layout is
// the authority (and it drives the module-wide MSAA SPIR-V option).
static WGPUPipelineLayout make_input_attachment_layout(WGPUDevice device) {
    YaWGPUInputAttachmentBindingLayout ia = {
        .chain = {.sType = YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT},
        .sampleType = WGPUTextureSampleType_Float,
        .multisampled = 1,
    };
    WGPUBindGroupLayoutEntry entry = {
        .nextInChain = &ia.chain,
        .binding = 0,
        .visibility = WGPUShaderStage_Fragment,
        .buffer = {.type = WGPUBufferBindingType_BindingNotUsed},
        .sampler = {.type = WGPUSamplerBindingType_BindingNotUsed},
        .texture = {.sampleType = WGPUTextureSampleType_BindingNotUsed,
                    .viewDimension = WGPUTextureViewDimension_Undefined},
        .storageTexture = {.access = WGPUStorageTextureAccess_BindingNotUsed,
                           .format = WGPUTextureFormat_Undefined,
                           .viewDimension = WGPUTextureViewDimension_Undefined},
    };
    WGPUBindGroupLayoutDescriptor bgld = {
        .label = sv_empty(), .entryCount = 1, .entries = &entry};
    WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgld);
    if (!bgl) {
        fprintf(stderr, "failed to create input-attachment bind group layout\n");
        exit(1);
    }
    WGPUPipelineLayoutDescriptor pld = {
        .label = sv_empty(), .bindGroupLayoutCount = 1, .bindGroupLayouts = &bgl};
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(device, &pld);
    wgpuBindGroupLayoutRelease(bgl);
    if (!pl) {
        fprintf(stderr, "failed to create input-attachment pipeline layout\n");
        exit(1);
    }
    return pl;
}

// Builds a flat-slot-indexed color-target array: `written_slot` uses `format`,
// every lower slot is disabled (input attachments are supplied by the core).
static WGPURenderPipeline make_pipeline(WGPUDevice device,
                                        YaWGPUSubpassPassLayout layout,
                                        uint32_t subpass_index, WGPUShaderModule module,
                                        uint32_t written_slot, WGPUTextureFormat format,
                                        uint32_t sample_count,
                                        WGPUPipelineLayout pipeline_layout) {
    WGPUColorTargetState targets[8];
    size_t n_targets = (size_t)written_slot + 1;
    for (size_t slot = 0; slot < n_targets; slot++) {
        bool w = (slot == written_slot);
        targets[slot] = (WGPUColorTargetState){
            .format = w ? format : WGPUTextureFormat_Undefined,
            .blend = NULL,
            .writeMask = w ? WGPUColorWriteMask_All : WGPUColorWriteMask_None,
        };
    }
    WGPUFragmentState fragment = {
        .module = module,
        .entryPoint = sv("fs"),
        .targetCount = n_targets,
        .targets = targets,
    };
    WGPURenderPipelineDescriptor base = {
        .label = sv_empty(),
        .layout = pipeline_layout, // NULL = auto (subpass 0); explicit for the reads
        .vertex = {.module = module, .entryPoint = sv("vs")},
        .primitive = {.topology = WGPUPrimitiveTopology_TriangleList},
        .multisample = {.count = sample_count, .mask = 0xFFFFFFFFu},
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

static Msaa msaa_create(WGPUDevice device, uint32_t w, uint32_t h,
                        WGPUTextureFormat final_format, WGPUShaderModule scene_module,
                        WGPUShaderModule persample_module,
                        WGPUShaderModule resolve_module) {
    Msaa m = {.width = w, .height = h};
    // The scene + intermediate are 4x MSAA subpass intermediates: written, consumed
    // in-pass as multisampled input attachments, and never stored (storeOp =
    // Discard below). Mark them TransientAttachment so the backend keeps them
    // on-tile / memoryless (Vulkan LAZILY_ALLOCATED) — the TBDR bandwidth payoff:
    // the MSAA samples never spill to DRAM.
    WGPUTextureUsage attach_usage =
        WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TransientAttachment;
    m.scene = make_target(device, w, h, WGPUTextureFormat_RGBA8Unorm, attach_usage, SAMPLE_COUNT);
    m.hdr = make_target(device, w, h, WGPUTextureFormat_RGBA8Unorm, attach_usage, SAMPLE_COUNT);
    m.scene_view = wgpuTextureCreateView(m.scene, NULL);
    m.hdr_view = wgpuTextureCreateView(m.hdr, NULL);

    YaWGPUAttachmentLayout colors[3] = {
        {.format = WGPUTextureFormat_RGBA8Unorm, .sampleCount = SAMPLE_COUNT},
        {.format = WGPUTextureFormat_RGBA8Unorm, .sampleCount = SAMPLE_COUNT},
        {.format = final_format, .sampleCount = 1},
    };
    uint32_t scene_colors[1] = {0};
    uint32_t persample_colors[1] = {1};
    uint32_t resolve_colors[1] = {2};
    YaWGPUSubpassInputAttachment persample_input = {
        .group = 0, .binding = 0, .sourceSubpass = 0, .sourceAttachment = 0};
    YaWGPUSubpassInputAttachment resolve_input = {
        .group = 0, .binding = 0, .sourceSubpass = 1, .sourceAttachment = 1};
    YaWGPUSubpassLayout subpasses[3] = {
        {.colorAttachmentIndices = scene_colors,
         .colorAttachmentIndexCount = 1,
         .inputAttachments = NULL,
         .inputAttachmentCount = 0},
        {.colorAttachmentIndices = persample_colors,
         .colorAttachmentIndexCount = 1,
         .inputAttachments = &persample_input,
         .inputAttachmentCount = 1},
        {.colorAttachmentIndices = resolve_colors,
         .colorAttachmentIndexCount = 1,
         .inputAttachments = &resolve_input,
         .inputAttachmentCount = 1},
    };
    YaWGPUSubpassDependency dependencies[2] = {
        {.srcSubpass = 0,
         .dstSubpass = 1,
         .dependencyType = YaWGPUSubpassDependencyType_ColorToInput,
         .byRegion = 1},
        {.srcSubpass = 1,
         .dstSubpass = 2,
         .dependencyType = YaWGPUSubpassDependencyType_ColorToInput,
         .byRegion = 1},
    };
    YaWGPUSubpassPassLayoutDescriptor layout_desc = {
        .label = sv_empty(),
        .colorAttachments = colors,
        .colorAttachmentCount = 3,
        .subpasses = subpasses,
        .subpassCount = 3,
        .dependencies = dependencies,
        .dependencyCount = 2,
    };
    m.layout = yawgpuDeviceCreateSubpassPassLayout(device, &layout_desc);
    if (!m.layout) {
        fprintf(stderr, "failed to create subpass pass layout\n");
        exit(1);
    }

    m.input_layout = make_input_attachment_layout(device);
    m.scene_pipeline = make_pipeline(device, m.layout, 0, scene_module, 0,
                                     WGPUTextureFormat_RGBA8Unorm, SAMPLE_COUNT, NULL);
    m.persample_pipeline = make_pipeline(device, m.layout, 1, persample_module, 1,
                                         WGPUTextureFormat_RGBA8Unorm, SAMPLE_COUNT,
                                         m.input_layout);
    m.resolve_pipeline = make_pipeline(device, m.layout, 2, resolve_module, 2,
                                       final_format, 1, m.input_layout);
    return m;
}

static void msaa_destroy(Msaa *m) {
    wgpuRenderPipelineRelease(m->resolve_pipeline);
    wgpuRenderPipelineRelease(m->persample_pipeline);
    wgpuRenderPipelineRelease(m->scene_pipeline);
    wgpuPipelineLayoutRelease(m->input_layout);
    yawgpuSubpassPassLayoutRelease(m->layout);
    wgpuTextureViewRelease(m->hdr_view);
    wgpuTextureViewRelease(m->scene_view);
    wgpuTextureRelease(m->hdr);
    wgpuTextureRelease(m->scene);
}

static void record_msaa(WGPUCommandEncoder encoder, const Msaa *m,
                        WGPUTextureView final_view) {
    YaWGPUSubpassColorAttachment attachments[3] = {
        // Transient MSAA intermediates: consumed in-pass, so storeOp = Discard.
        {.view = m->scene_view,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Discard,
         .clearValue = {0.02, 0.02, 0.05, 1}},
        {.view = m->hdr_view,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Discard,
         .clearValue = {0.02, 0.02, 0.05, 1}},
        {.view = final_view,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Store,
         .clearValue = {0.02, 0.02, 0.05, 1}},
    };
    YaWGPUSubpassRenderPassDescriptor pass_desc = {
        .label = sv_empty(),
        .passLayout = m->layout,
        .extent = {.width = m->width, .height = m->height, .depthOrArrayLayers = 1},
        .colorAttachments = attachments,
        .colorAttachmentCount = 3,
    };
    YaWGPUSubpassRenderPassEncoder pass =
        yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &pass_desc);
    yawgpuSubpassRenderPassEncoderSetPipeline(pass, m->scene_pipeline);
    yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpuSubpassRenderPassEncoderNextSubpass(pass);
    yawgpuSubpassRenderPassEncoderSetPipeline(pass, m->persample_pipeline);
    yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpuSubpassRenderPassEncoderNextSubpass(pass);
    yawgpuSubpassRenderPassEncoderSetPipeline(pass, m->resolve_pipeline);
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

static int run_offscreen(YawgpuContext *ctx, WGPUQueue queue,
                         WGPUShaderModule scene_module,
                         WGPUShaderModule persample_module,
                         WGPUShaderModule resolve_module) {
    const uint32_t W = 256, H = 256;
    Msaa m = msaa_create(ctx->device, W, H, WGPUTextureFormat_RGBA8Unorm, scene_module,
                         persample_module, resolve_module);
    WGPUTexture final = make_target(
        ctx->device, W, H, WGPUTextureFormat_RGBA8Unorm,
        WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc, 1);
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
    record_msaa(encoder, &m, final_view);
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
    // Center pixel is inside the triangle → the tinted+resolved orange.
    const uint8_t *center = pixels + ((size_t)(H / 2) * W + (W / 2)) * BYTES_PER_PIXEL;
    printf("center pixel (per-sample MSAA input, in-shader resolved): (%u, %u, %u, %u)\n",
           center[0], center[1], center[2], center[3]);
    stbi_write_png("tiled_msaa.png", W, H, BYTES_PER_PIXEL, pixels, W * BYTES_PER_PIXEL);
    printf("wrote tiled_msaa.png (%ux%u)\n", W, H);

    free(pixels);
    wgpuBufferUnmap(readback);
    wgpuBufferRelease(readback);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
    wgpuTextureViewRelease(final_view);
    wgpuTextureRelease(final);
    msaa_destroy(&m);
    return (int)yawgpu_uncaptured_error_count();
}

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
                        WGPUShaderModule scene_module,
                        WGPUShaderModule persample_module,
                        WGPUShaderModule resolve_module) {
    YawgpuWindow *window =
        yawgpu_window_create(640, 640, "yawgpu tiled_msaa (per-sample MSAA subpass input)");
    if (!window) {
        fprintf(stderr, "failed to create window (windowed examples need GLFW)\n");
        return 1;
    }
    WGPUSurface surface =
        yawgpu_window_create_surface(ctx->instance, window, "tiled_msaa surface");
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

    Msaa m = msaa_create(ctx->device, (uint32_t)w, (uint32_t)h, format, scene_module,
                         persample_module, resolve_module);

    printf("presenting an MSAA-resolved triangle; close the window to exit\n");
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
        record_msaa(encoder, &m, view);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuSurfacePresent(surface);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(encoder);
        wgpuTextureViewRelease(view);
        wgpuTextureRelease(current.texture);
        yawgpu_window_poll_events();
    }

    msaa_destroy(&m);
    wgpuSurfaceRelease(surface);
    yawgpu_window_destroy(window);
    return (int)yawgpu_uncaptured_error_count();
}

// MSAA subpass input is a Vulkan-only vendor surface. Refuse other backends with a
// clear message instead of a confusing pipeline-creation failure.
static bool require_vulkan(WGPUAdapter adapter) {
    WGPUAdapterInfo info = {0};
    if (wgpuAdapterGetInfo(adapter, &info) != WGPUStatus_Success) return true;
    bool ok = info.backendType == WGPUBackendType_Vulkan;
    if (!ok) {
        fprintf(stderr,
                "tiled_msaa needs the Vulkan backend (MSAA subpass input is "
                "Vulkan-only); set YAWGPU_BACKEND=vulkan\n");
    }
    wgpuAdapterInfoFreeMembers(info);
    return ok;
}

int main(int argc, char **argv) {
    if (argc > 0) yawgpu_set_argv0(argv[0]);
    bool verify = (argc > 1 && strcmp(argv[1], "--verify") == 0);

    YawgpuContext ctx = yawgpu_context_create();
    if (!ctx.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        return 1;
    }
    if (!require_vulkan(ctx.adapter)) {
        yawgpu_context_release(&ctx);
        return 1;
    }
    WGPUQueue queue = wgpuDeviceGetQueue(ctx.device);
    WGPUShaderModule scene_module = yawgpu_load_wgsl_shader(ctx.device, "scene.wgsl");
    WGPUShaderModule persample_module =
        yawgpu_load_wgsl_shader(ctx.device, "persample.wgsl");
    WGPUShaderModule resolve_module = yawgpu_load_wgsl_shader(ctx.device, "resolve.wgsl");
    if (!scene_module || !persample_module || !resolve_module) {
        fprintf(stderr, "failed to load shaders\n");
        return 1;
    }

    int rc = verify ? run_offscreen(&ctx, queue, scene_module, persample_module,
                                    resolve_module)
                    : run_windowed(&ctx, queue, scene_module, persample_module,
                                   resolve_module);

    wgpuShaderModuleRelease(resolve_module);
    wgpuShaderModuleRelease(persample_module);
    wgpuShaderModuleRelease(scene_module);
    wgpuQueueRelease(queue);
    yawgpu_context_release(&ctx);
    if (rc != 0) fprintf(stderr, "%d uncaptured device error(s)\n", rc);
    return rc;
}
