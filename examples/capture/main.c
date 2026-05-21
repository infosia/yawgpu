// capture — render to an offscreen texture and save it as a PNG.
//
// There is no window here. The program:
//   1. creates a texture used as a render target (RenderAttachment) that
//      can also be copied from (CopySrc),
//   2. runs a render pass that just clears it to solid red,
//   3. copies the texture into a CPU-mappable buffer, and
//   4. maps that buffer and writes the pixels out as red.png.
//
// The one subtlety worth studying is the **256-byte row alignment**
// required by texture-to-buffer copies (see BufferDimensions below).

#include "framework.h"
#include "stb_image_write.h"

#include <stdint.h>

enum {
    IMAGE_WIDTH = 100,
    IMAGE_HEIGHT = 200,
    // WebGPU requires each row of a texture→buffer copy to start at a
    // 256-byte boundary, so the destination buffer's stride is padded.
    COPY_BYTES_PER_ROW_ALIGNMENT = 256,
    BYTES_PER_PIXEL = 4, // RGBA8Unorm = 4 bytes/pixel
};

// Tracks both the natural row size (`unpadded`, width * 4) and the padded
// row stride the GPU copy actually uses. The PNG writer is told the padded
// stride so it skips the per-row padding bytes.
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

// All the long-lived resources, grouped so a single _destroy() can release
// them in one place regardless of which init step failed.
typedef struct CaptureApp {
    YawgpuContext context;
    WGPUQueue queue;
    WGPUBuffer output_buffer;    // CPU-mappable copy destination
    WGPUTexture texture;         // the offscreen render target
    WGPUTextureView texture_view;
    BufferDimensions dimensions;
    uint64_t buffer_size;
    bool output_buffer_mapped;   // so _destroy knows whether to unmap
} CaptureApp;

// Rounds `value` up to the next multiple of `alignment`.
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

static void capture_app_destroy(CaptureApp *app) {
    if (app->output_buffer_mapped) {
        wgpuBufferUnmap(app->output_buffer);
    }
    if (app->texture_view) {
        wgpuTextureViewRelease(app->texture_view);
    }
    if (app->texture) {
        wgpuTextureRelease(app->texture);
    }
    if (app->output_buffer) {
        wgpuBufferRelease(app->output_buffer);
    }
    if (app->queue) {
        wgpuQueueRelease(app->queue);
    }
    yawgpu_context_release(&app->context);
    *app = (CaptureApp){0};
}

static bool capture_app_init(CaptureApp *app) {
    *app = (CaptureApp){0};
    app->context = yawgpu_context_create();
    if (!app->context.instance || !app->context.adapter || !app->context.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        return false;
    }

    app->queue = wgpuDeviceGetQueue(app->context.device);
    if (!app->queue) {
        fprintf(stderr, "failed to get device queue\n");
        return false;
    }

    app->dimensions = buffer_dimensions_create(IMAGE_WIDTH, IMAGE_HEIGHT);
    app->buffer_size = (uint64_t)app->dimensions.padded_bytes_per_row * app->dimensions.height;
    WGPUExtent3D texture_size = {
        .width = app->dimensions.width,
        .height = app->dimensions.height,
        .depthOrArrayLayers = 1,
    };

    // Destination buffer: MapRead so the CPU can read it, CopyDst so the
    // texture copy can write into it. Sized for the padded row stride.
    app->output_buffer = wgpuDeviceCreateBuffer(
        app->context.device,
        &(WGPUBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("capture output buffer"),
            .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
            .size = app->buffer_size,
            .mappedAtCreation = false,
        });
    // Render target: RenderAttachment so a render pass can draw into it,
    // CopySrc so it can be copied out to the buffer afterwards.
    app->texture = wgpuDeviceCreateTexture(
        app->context.device,
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
    if (!app->output_buffer || !app->texture) {
        fprintf(stderr, "failed to create capture resources\n");
        return false;
    }

    // A render pass attaches a *view* of the texture, not the texture
    // itself. NULL requests a default view covering the whole texture.
    app->texture_view = wgpuTextureCreateView(app->texture, NULL);
    if (!app->texture_view) {
        fprintf(stderr, "failed to create texture view\n");
        return false;
    }

    return true;
}

// Records and submits the GPU work, then reads the result back and writes
// the PNG. Returns false on the first failure.
static bool capture_app_run(CaptureApp *app) {
    WGPUExtent3D texture_size = {
        .width = app->dimensions.width,
        .height = app->dimensions.height,
        .depthOrArrayLayers = 1,
    };
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
        app->context.device,
        &(WGPUCommandEncoderDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("capture encoder"),
        });
    if (!encoder) {
        fprintf(stderr, "failed to create command encoder\n");
        return false;
    }

    // A "clear-only" render pass: loadOp=Clear fills the attachment with
    // clearValue (opaque red) at the start, storeOp=Store keeps it. No
    // pipeline or draw call is needed — the clear alone produces the image.
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
        encoder,
        &(WGPURenderPassDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("capture clear pass"),
            .colorAttachmentCount = 1,
            .colorAttachments = (WGPURenderPassColorAttachment[]){
                {
                    .nextInChain = NULL,
                    .view = app->texture_view,
                    .depthSlice = WGPU_DEPTH_SLICE_UNDEFINED,
                    .resolveTarget = NULL,
                    .loadOp = WGPULoadOp_Clear,
                    .storeOp = WGPUStoreOp_Store,
                    .clearValue = {
                        .r = 1.0, // opaque red
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
        return false;
    }
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    // Copy the rendered texture into the readback buffer. `bytesPerRow`
    // must be the 256-byte-aligned padded stride, not width * 4.
    wgpuCommandEncoderCopyTextureToBuffer(
        encoder,
        &(WGPUTexelCopyTextureInfo){
            .texture = app->texture,
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
                .rowsPerImage = WGPU_COPY_STRIDE_UNDEFINED,
            },
        },
        &texture_size);

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(
        encoder,
        &(WGPUCommandBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("capture commands"),
        });
    if (!commands) {
        fprintf(stderr, "failed to finish command encoder\n");
        wgpuCommandEncoderRelease(encoder);
        return false;
    }
    wgpuQueueSubmit(app->queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);

    // Map the buffer for reading once the GPU work completes (async; we
    // pump the event loop until the callback reports the status).
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

    // The mapped pointer addresses the padded rows directly; stb is given
    // the padded stride so it reads width*4 bytes and skips the padding.
    const uint8_t *pixels =
        (const uint8_t *)wgpuBufferGetConstMappedRange(app->output_buffer, 0, app->buffer_size);
    if (!pixels) {
        fprintf(stderr, "readback mapped range is null\n");
        return false;
    }

    if (!stbi_write_png("red.png",
                        (int)app->dimensions.width,
                        (int)app->dimensions.height,
                        BYTES_PER_PIXEL,
                        pixels,
                        (int)app->dimensions.padded_bytes_per_row)) {
        fprintf(stderr, "failed to write red.png\n");
        return false;
    }

    printf("wrote red.png (%ux%u, bytesPerRow=%u padded to %u)\n",
           app->dimensions.width,
           app->dimensions.height,
           app->dimensions.unpadded_bytes_per_row,
           app->dimensions.padded_bytes_per_row);
    return true;
}

int main(void) {
    // Two-phase structure: init() acquires resources, run() does the work.
    // Either way _destroy() releases everything, so there is no goto-based
    // cleanup ladder.
    CaptureApp app = {0};
    if (!capture_app_init(&app)) {
        capture_app_destroy(&app);
        return EXIT_FAILURE;
    }
    if (!capture_app_run(&app)) {
        capture_app_destroy(&app);
        return EXIT_FAILURE;
    }

    capture_app_destroy(&app);
    return EXIT_SUCCESS;
}
