#include "framework.h"

typedef struct MapState {
    WGPUMapAsyncStatus status;
    bool called;
} MapState;

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
    uint32_t input[] = {1, 2, 3, 4};
    const size_t byte_size = sizeof(input);
    const uint32_t element_count = (uint32_t)(byte_size / sizeof(input[0]));

    YawgpuContext context = yawgpu_context_create();
    if (!context.instance || !context.adapter || !context.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        yawgpu_context_release(&context);
        return EXIT_FAILURE;
    }

    WGPUQueue queue = wgpuDeviceGetQueue(context.device);
    WGPUShaderModule shader = yawgpu_load_wgsl_shader(context.device, "shader.wgsl");
    if (!queue || !shader) {
        fprintf(stderr, "failed to create queue or shader\n");
        yawgpu_context_release(&context);
        return EXIT_FAILURE;
    }

    WGPUBuffer storage = yawgpu_create_buffer_init(
        context.device,
        &(YawgpuBufferInitDescriptor){
            .label = "storage",
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
            .contents = input,
            .size = byte_size,
        });
    WGPUBuffer readback = wgpuDeviceCreateBuffer(
        context.device,
        &(WGPUBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("readback"),
            .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
            .size = byte_size,
            .mappedAtCreation = false,
        });
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(
        context.device,
        &(WGPUComputePipelineDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("collatz pipeline"),
            .layout = NULL,
            .compute = {
                .nextInChain = NULL,
                .module = shader,
                .entryPoint = yawgpu_string_view("main"),
                .constantCount = 0,
                .constants = NULL,
            },
        });
    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(
        context.device,
        &(WGPUBindGroupDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("bind group"),
            .layout = layout,
            .entryCount = 1,
            .entries = (WGPUBindGroupEntry[]){
                {
                    .nextInChain = NULL,
                    .binding = 0,
                    .buffer = storage,
                    .offset = 0,
                    .size = byte_size,
                    .sampler = NULL,
                    .textureView = NULL,
                },
            },
        });

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
        context.device,
        &(WGPUCommandEncoderDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("compute encoder"),
        });
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(
        encoder,
        &(WGPUComputePassDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("compute pass"),
            .timestampWrites = NULL,
        });
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, element_count, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, readback, 0, byte_size);
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(
        encoder,
        &(WGPUCommandBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("compute commands"),
        });
    wgpuQueueSubmit(queue, 1, &commands);

    MapState map_state = {0};
    WGPUFuture map_future = wgpuBufferMapAsync(
        readback,
        WGPUMapMode_Read,
        0,
        byte_size,
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
        return EXIT_FAILURE;
    }

    const uint32_t *result =
        (const uint32_t *)wgpuBufferGetConstMappedRange(readback, 0, byte_size);
    if (!result) {
        fprintf(stderr, "readback mapped range is null\n");
        return EXIT_FAILURE;
    }

    printf("collatz readback: [%u, %u, %u, %u]\n",
           result[0], result[1], result[2], result[3]);

    WGPUAdapterInfo adapter_info = {0};
    if (wgpuAdapterGetInfo(context.adapter, &adapter_info) == WGPUStatus_Success) {
        if (adapter_info.backendType == WGPUBackendType_Null) {
            printf("Noop validates the compute path but does not execute GPU compute.\n");
        }
        wgpuAdapterInfoFreeMembers(adapter_info);
    }

    wgpuBufferUnmap(readback);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
    wgpuBindGroupLayoutRelease(layout);
    wgpuComputePipelineRelease(pipeline);
    wgpuBufferRelease(readback);
    wgpuBufferRelease(storage);
    wgpuShaderModuleRelease(shader);
    wgpuQueueRelease(queue);
    yawgpu_context_release(&context);
    return EXIT_SUCCESS;
}
