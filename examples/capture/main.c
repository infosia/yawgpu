#include "framework.h"
#include "stb_image_write.h"

#include <stdint.h>

enum {
    IMAGE_WIDTH = 100,
    IMAGE_HEIGHT = 200,
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

static uint32_t align_up_u32(uint32_t value, uint32_t alignment) {
    uint32_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    return value + alignment - remainder;
}

static BufferDimensions buffer_dimensions_create(uint32_t width, uint32_t height) {
    BufferDimensions dimensions = {
        .width = width,
        .height = height,
        .unpadded_bytes_per_row = width * BYTES_PER_PIXEL,
        .padded_bytes_per_row = align_up_u32(width * BYTES_PER_PIXEL,
                                             COPY_BYTES_PER_ROW_ALIGNMENT),
    };
    return dimensions;
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

int main(void) {
    int exit_code = EXIT_FAILURE;
    YawgpuContext context = {0};
    WGPUQueue queue = NULL;
    WGPUBuffer output_buffer = NULL;
    WGPUTexture texture = NULL;
    WGPUTextureView texture_view = NULL;
    WGPUCommandEncoder encoder = NULL;
    WGPUCommandBuffer commands = NULL;
    bool output_buffer_mapped = false;

    context = yawgpu_context_create();
    if (!context.instance || !context.adapter || !context.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        goto cleanup;
    }

    queue = wgpuDeviceGetQueue(context.device);
    if (!queue) {
        fprintf(stderr, "failed to get device queue\n");
        goto cleanup;
    }

    BufferDimensions dimensions = buffer_dimensions_create(IMAGE_WIDTH, IMAGE_HEIGHT);
    uint64_t buffer_size = (uint64_t)dimensions.padded_bytes_per_row * dimensions.height;
    WGPUExtent3D texture_size = {
        .width = dimensions.width,
        .height = dimensions.height,
        .depthOrArrayLayers = 1,
    };

    output_buffer = wgpuDeviceCreateBuffer(
        context.device,
        &(WGPUBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("capture output buffer"),
            .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
            .size = buffer_size,
            .mappedAtCreation = false,
        });
    texture = wgpuDeviceCreateTexture(
        context.device,
        &(WGPUTextureDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("capture texture"),
            .usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc,
            .dimension = WGPUTextureDimension_2D,
            .size = texture_size,
            .format = WGPUTextureFormat_RGBA8Unorm,
            .mipLevelCount = 1,
            .sampleCount = 1,
            .viewFormatCount = 0,
            .viewFormats = NULL,
        });
    if (!output_buffer || !texture) {
        fprintf(stderr, "failed to create capture resources\n");
        goto cleanup;
    }

    texture_view = wgpuTextureCreateView(texture, NULL);
    encoder = wgpuDeviceCreateCommandEncoder(
        context.device,
        &(WGPUCommandEncoderDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("capture encoder"),
        });
    if (!texture_view || !encoder) {
        fprintf(stderr, "failed to create texture view or command encoder\n");
        goto cleanup;
    }

    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
        encoder,
        &(WGPURenderPassDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("capture clear pass"),
            .colorAttachmentCount = 1,
            .colorAttachments = (WGPURenderPassColorAttachment[]){
                {
                    .nextInChain = NULL,
                    .view = texture_view,
                    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
                    .resolveTarget = NULL,
                    .loadOp = WGPULoadOp_Clear,
                    .storeOp = WGPUStoreOp_Store,
                    .clearValue = {
                        .r = 1.0,
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
        goto cleanup;
    }
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    wgpuCommandEncoderCopyTextureToBuffer(
        encoder,
        &(WGPUTexelCopyTextureInfo){
            .texture = texture,
            .mipLevel = 0,
            .origin = {
                .x = 0,
                .y = 0,
                .z = 0,
            },
            .aspect = WGPUTextureAspect_All,
        },
        &(WGPUTexelCopyBufferInfo){
            .buffer = output_buffer,
            .layout = {
                .offset = 0,
                .bytesPerRow = dimensions.padded_bytes_per_row,
                .rowsPerImage = WGPU_COPY_STRIDE_UNDEFINED,
            },
        },
        &texture_size);

    commands = wgpuCommandEncoderFinish(
        encoder,
        &(WGPUCommandBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("capture commands"),
        });
    if (!commands) {
        fprintf(stderr, "failed to finish command encoder\n");
        goto cleanup;
    }
    wgpuQueueSubmit(queue, 1, &commands);

    MapState map_state = {0};
    WGPUFuture map_future = wgpuBufferMapAsync(
        output_buffer,
        WGPUMapMode_Read,
        0,
        buffer_size,
        (WGPUBufferMapCallbackInfo){
            .nextInChain = NULL,
            .mode = WGPUCallbackMode_AllowProcessEvents,
            .callback = map_callback,
            .userdata1 = &map_state,
            .userdata2 = NULL,
        });
    yawgpu_wait_for_future(context.instance, map_future);
    if (!map_state.called || map_state.status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "readback map did not complete successfully\n");
        goto cleanup;
    }
    output_buffer_mapped = true;

    const uint8_t *pixels =
        (const uint8_t *)wgpuBufferGetConstMappedRange(output_buffer, 0, buffer_size);
    if (!pixels) {
        fprintf(stderr, "readback mapped range is null\n");
        goto cleanup;
    }

    if (!stbi_write_png("red.png",
                        (int)dimensions.width,
                        (int)dimensions.height,
                        BYTES_PER_PIXEL,
                        pixels,
                        (int)dimensions.padded_bytes_per_row)) {
        fprintf(stderr, "failed to write red.png\n");
        goto cleanup;
    }

    printf("wrote red.png (%ux%u, bytesPerRow=%u padded to %u)\n",
           dimensions.width,
           dimensions.height,
           dimensions.unpadded_bytes_per_row,
           dimensions.padded_bytes_per_row);
    exit_code = EXIT_SUCCESS;

cleanup:
    if (output_buffer_mapped) {
        wgpuBufferUnmap(output_buffer);
    }
    if (commands) {
        wgpuCommandBufferRelease(commands);
    }
    if (encoder) {
        wgpuCommandEncoderRelease(encoder);
    }
    if (texture_view) {
        wgpuTextureViewRelease(texture_view);
    }
    if (texture) {
        wgpuTextureRelease(texture);
    }
    if (output_buffer) {
        wgpuBufferRelease(output_buffer);
    }
    if (queue) {
        wgpuQueueRelease(queue);
    }
    yawgpu_context_release(&context);
    return exit_code;
}
