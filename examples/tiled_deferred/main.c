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
// bandwidth win of deferred shading. The example renders offscreen, reads the
// final target back, writes `tiled_deferred.png`, and prints the center pixel.
//
// Backend via the YAWGPU_BACKEND env var (metal / vulkan). Requires libyawgpu
// built with the `tiled` cargo feature (CMake: -DYAWGPU_TILED=ON).

#include "framework.h"
#include "stb_image_write.h"

#include <stdint.h>

enum {
    WIDTH = 256,
    HEIGHT = 256,
    BYTES_PER_PIXEL = 4,
    ROW_ALIGN = 256, // CopyTextureToBuffer requires 256-byte-aligned rows.
};

static WGPUStringView sv(const char *s) { return yawgpu_string_view(s); }
static WGPUStringView sv_empty(void) {
    WGPUStringView v = {.data = NULL, .length = 0};
    return v;
}

static uint32_t padded_bytes_per_row(void) {
    uint32_t unpadded = WIDTH * BYTES_PER_PIXEL;
    uint32_t pad = (ROW_ALIGN - (unpadded % ROW_ALIGN)) % ROW_ALIGN;
    return unpadded + pad;
}

static WGPUTexture create_target(WGPUDevice device, WGPUTextureFormat format,
                                 WGPUTextureUsage usage) {
    WGPUTextureDescriptor d = {
        .label = sv_empty(),
        .usage = usage,
        .dimension = WGPUTextureDimension_2D,
        .size = {.width = WIDTH, .height = HEIGHT, .depthOrArrayLayers = 1},
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
// color slots this subpass writes (each gets an RGBA8 target); every other slot
// up to the max is a disabled placeholder. Input-attachment slots are NOT listed
// here — the core supplies them per backend (the portable contract).
static size_t build_targets(const uint32_t *written, size_t n_written,
                            WGPUColorTargetState *out) {
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
            .format = is_written ? WGPUTextureFormat_RGBA8Unorm
                                 : WGPUTextureFormat_Undefined,
            .blend = NULL,
            .writeMask = is_written ? WGPUColorWriteMask_All : WGPUColorWriteMask_None,
        };
    }
    return count;
}

static WGPURenderPipeline create_subpass_pipeline(WGPUDevice device,
                                                  YaWGPUSubpassPassLayout layout,
                                                  uint32_t subpass_index,
                                                  WGPUShaderModule module,
                                                  const uint32_t *written,
                                                  size_t n_written) {
    WGPUColorTargetState targets[8];
    size_t n_targets = build_targets(written, n_written, targets);

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
        .base = base,
        .passLayout = layout,
        .subpassIndex = subpass_index,
    };
    WGPURenderPipeline p = yawgpuDeviceCreateSubpassRenderPipeline(device, &d);
    if (!p) {
        fprintf(stderr, "failed to create subpass pipeline %u\n", subpass_index);
        exit(1);
    }
    return p;
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

int main(int argc, char **argv) {
    if (argc > 0) yawgpu_set_argv0(argv[0]);

    YawgpuContext ctx = yawgpu_context_create();
    if (!ctx.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        return 1;
    }
    WGPUDevice device = ctx.device;
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    // G-buffer + final targets. The final target is also a copy source.
    WGPUTexture albedo = create_target(device, WGPUTextureFormat_RGBA8Unorm,
                                       WGPUTextureUsage_RenderAttachment);
    WGPUTexture normal = create_target(device, WGPUTextureFormat_RGBA8Unorm,
                                       WGPUTextureUsage_RenderAttachment);
    WGPUTexture final = create_target(
        device, WGPUTextureFormat_RGBA8Unorm,
        WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc);
    WGPUTextureView albedo_view = wgpuTextureCreateView(albedo, NULL);
    WGPUTextureView normal_view = wgpuTextureCreateView(normal, NULL);
    WGPUTextureView final_view = wgpuTextureCreateView(final, NULL);

    // Pass layout: 3 color attachments, 2 subpasses. Subpass 1 reads attachments
    // 0 (albedo) and 1 (normal) written by subpass 0 as input attachments.
    YaWGPUAttachmentLayout colors[3] = {
        {.format = WGPUTextureFormat_RGBA8Unorm, .sampleCount = 1},
        {.format = WGPUTextureFormat_RGBA8Unorm, .sampleCount = 1},
        {.format = WGPUTextureFormat_RGBA8Unorm, .sampleCount = 1},
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
         .usesDepthStencil = 0,
         .inputAttachments = NULL,
         .inputAttachmentCount = 0},
        {.colorAttachmentIndices = lighting_colors,
         .colorAttachmentIndexCount = 1,
         .usesDepthStencil = 0,
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
        .depthStencilAttachment = NULL,
        .subpasses = subpasses,
        .subpassCount = 2,
        .dependencies = &dependency,
        .dependencyCount = 1,
    };
    YaWGPUSubpassPassLayout layout =
        yawgpuDeviceCreateSubpassPassLayout(device, &layout_desc);
    if (!layout) {
        fprintf(stderr, "failed to create subpass pass layout\n");
        return 1;
    }

    WGPUShaderModule gbuffer_module = yawgpu_load_wgsl_shader(device, "gbuffer.wgsl");
    WGPUShaderModule lighting_module = yawgpu_load_wgsl_shader(device, "lighting.wgsl");
    if (!gbuffer_module || !lighting_module) {
        fprintf(stderr, "failed to load shaders\n");
        return 1;
    }

    uint32_t gbuffer_written[2] = {0, 1};
    uint32_t lighting_written[1] = {2};
    WGPURenderPipeline gbuffer_pipeline = create_subpass_pipeline(
        device, layout, 0, gbuffer_module, gbuffer_written, 2);
    WGPURenderPipeline lighting_pipeline = create_subpass_pipeline(
        device, layout, 1, lighting_module, lighting_written, 1);

    // Record the two-subpass pass.
    YaWGPUSubpassColorAttachment attachments[3] = {
        {.view = albedo_view,
         .resolveTarget = NULL,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Store,
         .clearValue = {0, 0, 0, 1}},
        {.view = normal_view,
         .resolveTarget = NULL,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Store,
         .clearValue = {0, 0, 0, 1}},
        {.view = final_view,
         .resolveTarget = NULL,
         .loadOp = WGPULoadOp_Clear,
         .storeOp = WGPUStoreOp_Store,
         .clearValue = {0, 0, 0, 1}},
    };
    YaWGPUSubpassRenderPassDescriptor pass_desc = {
        .label = sv_empty(),
        .passLayout = layout,
        .extent = {.width = WIDTH, .height = HEIGHT, .depthOrArrayLayers = 1},
        .colorAttachments = attachments,
        .colorAttachmentCount = 3,
        .depthStencilAttachment = NULL,
    };

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    YaWGPUSubpassRenderPassEncoder pass =
        yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &pass_desc);
    yawgpuSubpassRenderPassEncoderSetPipeline(pass, gbuffer_pipeline);
    yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpuSubpassRenderPassEncoderNextSubpass(pass);
    yawgpuSubpassRenderPassEncoderSetPipeline(pass, lighting_pipeline);
    // The input attachments (group 0) are bound implicitly by the pass — no bind
    // group is needed.
    yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpuSubpassRenderPassEncoderEnd(pass);
    yawgpuSubpassRenderPassEncoderRelease(pass);

    // Copy the final target into a readback buffer (256-byte-aligned rows).
    uint32_t padded_row = padded_bytes_per_row();
    uint64_t buffer_size = (uint64_t)padded_row * HEIGHT;
    WGPUBufferDescriptor buffer_desc = {
        .label = sv_empty(),
        .usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead,
        .size = buffer_size,
        .mappedAtCreation = 0,
    };
    WGPUBuffer readback = wgpuDeviceCreateBuffer(device, &buffer_desc);

    WGPUTexelCopyTextureInfo src = {
        .texture = final,
        .mipLevel = 0,
        .origin = {0, 0, 0},
        .aspect = WGPUTextureAspect_All,
    };
    WGPUTexelCopyBufferInfo dst = {
        .layout = {.offset = 0, .bytesPerRow = padded_row, .rowsPerImage = HEIGHT},
        .buffer = readback,
    };
    WGPUExtent3D extent = {.width = WIDTH, .height = HEIGHT, .depthOrArrayLayers = 1};
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(queue, 1, &cmd);

    // Map + read back.
    MapState map_state = {0};
    WGPUBufferMapCallbackInfo cb = {
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = map_callback,
        .userdata1 = &map_state,
    };
    WGPUFuture future =
        wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, (size_t)buffer_size, cb);
    yawgpu_wait_for_future(ctx.instance, future);
    if (!map_state.called || map_state.status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "buffer map failed\n");
        return 1;
    }

    const uint8_t *mapped =
        (const uint8_t *)wgpuBufferGetConstMappedRange(readback, 0, (size_t)buffer_size);

    // Unpad rows into a tight WIDTH*HEIGHT*4 image.
    uint8_t *pixels = (uint8_t *)malloc((size_t)WIDTH * HEIGHT * BYTES_PER_PIXEL);
    for (uint32_t y = 0; y < HEIGHT; y++) {
        memcpy(pixels + (size_t)y * WIDTH * BYTES_PER_PIXEL,
               mapped + (size_t)y * padded_row, (size_t)WIDTH * BYTES_PER_PIXEL);
    }

    const uint8_t *center =
        pixels + ((size_t)(HEIGHT / 2) * WIDTH + (WIDTH / 2)) * BYTES_PER_PIXEL;
    printf("center pixel (shaded from the input-attachment G-buffer read): "
           "(%u, %u, %u, %u)\n",
           center[0], center[1], center[2], center[3]);

    if (!stbi_write_png("tiled_deferred.png", WIDTH, HEIGHT, BYTES_PER_PIXEL, pixels,
                        WIDTH * BYTES_PER_PIXEL)) {
        fprintf(stderr, "failed to write tiled_deferred.png\n");
        return 1;
    }
    printf("wrote tiled_deferred.png (%dx%d)\n", WIDTH, HEIGHT);

    unsigned int errors = yawgpu_uncaptured_error_count();
    if (errors != 0) {
        fprintf(stderr, "%u uncaptured device error(s)\n", errors);
    }

    free(pixels);
    wgpuBufferUnmap(readback);
    wgpuBufferRelease(readback);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(encoder);
    wgpuRenderPipelineRelease(lighting_pipeline);
    wgpuRenderPipelineRelease(gbuffer_pipeline);
    wgpuShaderModuleRelease(lighting_module);
    wgpuShaderModuleRelease(gbuffer_module);
    yawgpuSubpassPassLayoutRelease(layout);
    wgpuTextureViewRelease(final_view);
    wgpuTextureViewRelease(normal_view);
    wgpuTextureViewRelease(albedo_view);
    wgpuTextureRelease(final);
    wgpuTextureRelease(normal);
    wgpuTextureRelease(albedo);
    wgpuQueueRelease(queue);
    yawgpu_context_release(&ctx);
    return errors == 0 ? 0 : 1;
}
