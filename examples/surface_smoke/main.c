#include "framework.h"

int main(void) {
    int exit_code = EXIT_FAILURE;
    YawgpuWindow *window = NULL;
    WGPUSurface surface = NULL;
    WGPUQueue queue = NULL;
    WGPUTextureView view = NULL;
    WGPUCommandEncoder encoder = NULL;
    WGPURenderPassEncoder pass = NULL;
    WGPUCommandBuffer commands = NULL;
    WGPUTexture surface_texture = NULL;

    YawgpuContext context = yawgpu_context_create();
    if (!context.instance || !context.adapter || !context.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        goto cleanup;
    }
    queue = wgpuDeviceGetQueue(context.device);
    if (!queue) {
        fprintf(stderr, "failed to get queue\n");
        goto cleanup;
    }

    window = yawgpu_window_create(800, 600, "yawgpu surface_smoke");
    if (!window) {
        fprintf(stderr, "failed to create window\n");
        goto cleanup;
    }
    void *layer = yawgpu_window_metal_layer(window);
    if (!layer) {
        fprintf(stderr, "failed to get CAMetalLayer\n");
        goto cleanup;
    }

    WGPUSurfaceSourceMetalLayer metal_layer = {
        .chain = {
            .next = NULL,
            .sType = WGPUSType_SurfaceSourceMetalLayer,
        },
        .layer = layer,
    };
    surface = wgpuInstanceCreateSurface(
        context.instance,
        &(WGPUSurfaceDescriptor){
            .nextInChain = &metal_layer.chain,
            .label = yawgpu_string_view("surface_smoke surface"),
        });
    if (!surface) {
        fprintf(stderr, "failed to create surface\n");
        goto cleanup;
    }

    WGPUSurfaceCapabilities capabilities = {0};
    if (wgpuSurfaceGetCapabilities(surface, context.adapter, &capabilities) != WGPUStatus_Success) {
        fprintf(stderr, "failed to get surface capabilities\n");
        goto cleanup;
    }
    WGPUTextureFormat format = WGPUTextureFormat_BGRA8Unorm;
    bool found_format = false;
    for (size_t i = 0; i < capabilities.formatCount; ++i) {
        if (capabilities.formats[i] == WGPUTextureFormat_BGRA8Unorm) {
            format = WGPUTextureFormat_BGRA8Unorm;
            found_format = true;
            break;
        }
        if (capabilities.formats[i] == WGPUTextureFormat_RGBA8Unorm) {
            format = WGPUTextureFormat_RGBA8Unorm;
            found_format = true;
        }
    }
    wgpuSurfaceCapabilitiesFreeMembers(capabilities);
    if (!found_format) {
        fprintf(stderr, "no supported surface format found\n");
        goto cleanup;
    }

    int width = 0;
    int height = 0;
    yawgpu_window_framebuffer_size(window, &width, &height);
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "invalid framebuffer size\n");
        goto cleanup;
    }
    wgpuSurfaceConfigure(
        surface,
        &(WGPUSurfaceConfiguration){
            .nextInChain = NULL,
            .device = context.device,
            .format = format,
            .usage = WGPUTextureUsage_RenderAttachment,
            .width = (uint32_t)width,
            .height = (uint32_t)height,
            .viewFormatCount = 0,
            .viewFormats = NULL,
            .alphaMode = WGPUCompositeAlphaMode_Opaque,
            .presentMode = WGPUPresentMode_Fifo,
        });

    for (uint32_t frame = 0; frame < 60 && !yawgpu_window_should_close(window); ++frame) {
        WGPUSurfaceTexture current = {0};
        wgpuSurfaceGetCurrentTexture(surface, &current);
        if (current.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
            !current.texture) {
            fprintf(stderr, "failed to acquire surface texture, status=%u\n", current.status);
            goto cleanup;
        }
        surface_texture = current.texture;
        view = wgpuTextureCreateView(surface_texture, NULL);
        encoder = wgpuDeviceCreateCommandEncoder(
            context.device,
            &(WGPUCommandEncoderDescriptor){
                .nextInChain = NULL,
                .label = yawgpu_string_view("surface_smoke encoder"),
            });
        pass = wgpuCommandEncoderBeginRenderPass(
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
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
        pass = NULL;
        commands = wgpuCommandEncoderFinish(
            encoder,
            &(WGPUCommandBufferDescriptor){
                .nextInChain = NULL,
                .label = yawgpu_string_view("surface_smoke commands"),
            });
        wgpuQueueSubmit(queue, 1, &commands);
        if (wgpuSurfacePresent(surface) != WGPUStatus_Success) {
            fprintf(stderr, "surface present failed\n");
            goto cleanup;
        }
        wgpuCommandBufferRelease(commands);
        commands = NULL;
        wgpuCommandEncoderRelease(encoder);
        encoder = NULL;
        wgpuTextureViewRelease(view);
        view = NULL;
        wgpuTextureRelease(surface_texture);
        surface_texture = NULL;
        yawgpu_window_poll_events();
    }
    exit_code = EXIT_SUCCESS;

cleanup:
    if (commands) {
        wgpuCommandBufferRelease(commands);
    }
    if (pass) {
        wgpuRenderPassEncoderRelease(pass);
    }
    if (encoder) {
        wgpuCommandEncoderRelease(encoder);
    }
    if (view) {
        wgpuTextureViewRelease(view);
    }
    if (surface_texture) {
        wgpuTextureRelease(surface_texture);
    }
    if (surface) {
        wgpuSurfaceUnconfigure(surface);
        wgpuSurfaceRelease(surface);
    }
    if (window) {
        yawgpu_window_destroy(window);
    }
    if (queue) {
        wgpuQueueRelease(queue);
    }
    yawgpu_context_release(&context);
    return exit_code;
}
