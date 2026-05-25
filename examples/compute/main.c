// compute — a headless GPU compute dispatch with CPU readback.
//
// It runs one step of the Collatz sequence on each element of a small
// input array entirely on the GPU, then copies the result back to the CPU
// and prints it. The end-to-end flow demonstrated here is the canonical
// WebGPU compute pipeline:
//
//   storage buffer (input)  ──set as bind group──┐
//   compute pipeline (shader.wgsl)               │
//        │ dispatch one workgroup per element     │
//        ▼                                         │
//   storage buffer (mutated in place) ──copy──► readback buffer
//                                                   │ map for reading
//                                                   ▼
//                                              CPU reads the results
//
// On the Noop backend the whole path is *validated* but no actual GPU
// computation happens, so the readback stays at the input values.

#include "framework.h"

// wgpuBufferMapAsync delivers its result through this callback. We stash
// the status in `userdata1` so main() can check it after pumping events.
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

int main(int argc, char **argv) {
    // Let the framework resolve `shader.wgsl` next to this binary regardless
    // of cwd. Safe to call with NULL.
    yawgpu_set_argv0(argc > 0 ? argv[0] : NULL);

    uint32_t input[] = {1, 2, 3, 4};
    const size_t byte_size = sizeof(input);
    const uint32_t element_count = (uint32_t)(byte_size / sizeof(input[0]));

    // YawgpuContext bundles the instance + adapter + device acquisition
    // (the boilerplate every example shares) into one call.
    YawgpuContext context = yawgpu_context_create();
    if (!context.instance || !context.adapter || !context.device) {
        fprintf(stderr, "failed to create yawgpu context\n");
        yawgpu_context_release(&context);
        return EXIT_FAILURE;
    }

    // The queue is where finished command buffers are submitted for
    // execution. The shader module is compiled from WGSL at this point.
    WGPUQueue queue = wgpuDeviceGetQueue(context.device);
    WGPUShaderModule shader = yawgpu_load_wgsl_shader(context.device, "shader.wgsl");
    if (!queue || !shader) {
        fprintf(stderr, "failed to create queue or shader\n");
        yawgpu_context_release(&context);
        return EXIT_FAILURE;
    }

    // The `storage` buffer holds the data the shader reads and writes.
    // Usage flags declare every way the buffer will be used, up front:
    //   Storage  — bindable as a read_write storage buffer in the shader,
    //   CopyDst  — can receive the initial upload,
    //   CopySrc  — can be the source of the copy into the readback buffer.
    WGPUBuffer storage = yawgpu_create_buffer_init(
        context.device,
        &(YawgpuBufferInitDescriptor){
            .label = "storage",
            .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
            .contents = input,
            .size = byte_size,
        });
    // The GPU cannot map a storage buffer for CPU reading directly, so we
    // need a separate `readback` buffer with MapRead | CopyDst usage: the
    // shader's results are copied here, then this buffer is mapped.
    WGPUBuffer readback = wgpuDeviceCreateBuffer(
        context.device,
        &(WGPUBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("readback"),
            .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
            .size = byte_size,
            .mappedAtCreation = false,
        });
    // Compiling the compute pipeline binds the shader's `main` entry point.
    // Passing layout = NULL asks the implementation to infer the bind group
    // layout from the shader's declared bindings ("auto layout").
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
    // A bind group binds concrete resources to the shader's binding slots.
    // We fetch the auto-inferred layout for group 0 and bind `storage` at
    // binding 0, matching `@group(0) @binding(0)` in shader.wgsl.
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

    // GPU work is recorded into a command encoder, then finished into an
    // immutable command buffer and submitted. Nothing runs until submit.
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(
        context.device,
        &(WGPUCommandEncoderDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("compute encoder"),
        });
    // A compute pass is the scope in which dispatches happen.
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(
        encoder,
        &(WGPUComputePassDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("compute pass"),
            .timestampWrites = NULL,
        });
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    // Dispatch one workgroup per array element. shader.wgsl uses a
    // workgroup size of 1, so each invocation handles one element.
    wgpuComputePassEncoderDispatchWorkgroups(pass, element_count, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    // Copy the computed results out of the storage buffer (which the CPU
    // cannot map) into the mappable readback buffer.
    wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, readback, 0, byte_size);
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(
        encoder,
        &(WGPUCommandBufferDescriptor){
            .nextInChain = NULL,
            .label = yawgpu_string_view("compute commands"),
        });
    wgpuQueueSubmit(queue, 1, &commands);

    // Mapping is asynchronous: the buffer becomes CPU-accessible only once
    // the GPU work that produced its contents has completed. We register a
    // callback and pump the instance's event loop until it fires.
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

    int exit_status = EXIT_SUCCESS;
    bool mapped = false;
    if (!map_state.called || map_state.status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "readback map did not complete successfully\n");
        exit_status = EXIT_FAILURE;
        goto cleanup;
    }
    mapped = true;

    // Once mapped, GetConstMappedRange yields a CPU pointer into the
    // buffer's bytes that stays valid until wgpuBufferUnmap is called.
    const uint32_t *result =
        (const uint32_t *)wgpuBufferGetConstMappedRange(readback, 0, byte_size);
    if (!result) {
        fprintf(stderr, "readback mapped range is null\n");
        exit_status = EXIT_FAILURE;
        goto cleanup;
    }

    // On a real backend this prints the Collatz step of each input
    // (e.g. [0, 1, 7, 2]); on Noop it echoes the unmodified input.
    printf("collatz readback: [%u, %u, %u, %u]\n",
           result[0], result[1], result[2], result[3]);

    WGPUAdapterInfo adapter_info = {0};
    if (wgpuAdapterGetInfo(context.adapter, &adapter_info) == WGPUStatus_Success) {
        if (adapter_info.backendType == WGPUBackendType_Null) {
            printf("Noop validates the compute path but does not execute GPU compute.\n");
        }
        wgpuAdapterInfoFreeMembers(adapter_info);
    }

cleanup:
    // Unmap (if still mapped) invalidates `result`, then release every handle
    // (reverse order) and tear down the context. Reached from both the
    // success path and the early-failure `goto`s above.
    if (mapped) {
        wgpuBufferUnmap(readback);
    }
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
    return exit_status;
}
