// Cross-implementation CPU-overhead benchmark for the WebGPU C ABI.
//
// The same translation unit is compiled twice — once linked against yawgpu,
// once against Dawn — so every measurement crosses the identical `webgpu.h`
// entry points. What is measured is the *CPU* cost of the implementation:
// validation, object construction, command recording, submission bookkeeping.
// GPU execution time is deliberately excluded except in the two `submit/*wait`
// cases, which are labelled as such.
//
// Method: each case runs `warmup` untimed iterations, then `reps` batches of
// `iters` timed iterations. The reported figure is the *minimum* per-op time
// across batches (the standard microbenchmark statistic — the least
// noise-contaminated estimate of true cost), with the median alongside so a
// large min/median gap flags an unstable case.

#include "backend.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

WGPUStringView sv(const char* s) {
    WGPUStringView view;
    view.data = s;
    view.length = WGPU_STRLEN;
    return view;
}

std::string toString(WGPUStringView view) {
    if (view.data == nullptr) {
        return {};
    }
    if (view.length == WGPU_STRLEN) {
        return std::string(view.data);
    }
    return std::string(view.data, view.length);
}

[[noreturn]] void fail(const std::string& message) {
    std::fprintf(stderr, "bench: fatal: %s\n", message.c_str());
    std::exit(1);
}

// Busy-poll rather than the CTS harness's 1ms sleep: a sleeping poll loop would
// dominate every latency figure it appears in.
bool pumpUntil(WGPUInstance instance, const std::function<bool()>& done, uint64_t timeoutNs = 5'000'000'000) {
    const auto start = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::nanoseconds(timeoutNs);
    while (!done()) {
        wgpuInstanceProcessEvents(instance);
        if (done()) {
            return true;
        }
        if (std::chrono::steady_clock::now() - start >= timeout) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Device bring-up
// ---------------------------------------------------------------------------

struct AdapterState {
    bool completed = false;
    WGPURequestAdapterStatus status = WGPURequestAdapterStatus_Error;
    WGPUAdapter adapter = nullptr;
    std::string message;
};

struct DeviceState {
    bool completed = false;
    WGPURequestDeviceStatus status = WGPURequestDeviceStatus_Error;
    WGPUDevice device = nullptr;
    std::string message;
};

void onRequestAdapter(
    WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void*) {
    auto* state = static_cast<AdapterState*>(userdata1);
    state->completed = true;
    state->status = status;
    state->adapter = adapter;
    state->message = toString(message);
}

void onRequestDevice(
    WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void*) {
    auto* state = static_cast<DeviceState*>(userdata1);
    state->completed = true;
    state->status = status;
    state->device = device;
    state->message = toString(message);
}

void onUncapturedError(const WGPUDevice*, WGPUErrorType type, WGPUStringView message, void*, void*) {
    // An operation that fails validation is cheap; letting one go unnoticed
    // would silently turn a bug into a benchmark "win".
    std::fprintf(stderr, "bench: uncaptured device error (type %d): %s\n",
                 static_cast<int>(type), toString(message).c_str());
    std::exit(1);
}

void onDeviceLost(const WGPUDevice*, WGPUDeviceLostReason reason, WGPUStringView message, void*, void*) {
    if (reason == WGPUDeviceLostReason_Destroyed) {
        return;
    }
    std::fprintf(stderr, "bench: device lost (reason %d): %s\n",
                 static_cast<int>(reason), toString(message).c_str());
    std::exit(1);
}

// ---------------------------------------------------------------------------
// Timing core
// ---------------------------------------------------------------------------

struct Result {
    std::string name;
    uint64_t iters = 0;
    double minNsPerOp = 0.0;
    double medianNsPerOp = 0.0;
};

struct Options {
    uint64_t reps = 7;
    double scale = 1.0;
    std::string filter;
    bool tsv = false;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if (n == 0) {
        return 0.0;
    }
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

// `body(i)` performs exactly one operation of the case under test.
Result run(const Options& options, const std::string& name, uint64_t baseIters, const std::function<void(uint64_t)>& body) {
    uint64_t iters = static_cast<uint64_t>(static_cast<double>(baseIters) * options.scale);
    if (iters == 0) {
        iters = 1;
    }

    const uint64_t warmup = std::max<uint64_t>(1, iters / 10);
    for (uint64_t i = 0; i < warmup; ++i) {
        body(i);
    }

    std::vector<double> perOp;
    perOp.reserve(options.reps);
    for (uint64_t rep = 0; rep < options.reps; ++rep) {
        const auto start = std::chrono::steady_clock::now();
        for (uint64_t i = 0; i < iters; ++i) {
            body(i);
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        perOp.push_back(ns / static_cast<double>(iters));
    }

    Result result;
    result.name = name;
    result.iters = iters;
    result.minNsPerOp = *std::min_element(perOp.begin(), perOp.end());
    result.medianNsPerOp = median(perOp);
    return result;
}

// ---------------------------------------------------------------------------
// Shader sources
// ---------------------------------------------------------------------------

const char* kComputeWgsl = R"(
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    data[gid.x] = data[gid.x] + 1u;
}
)";

// A unique-per-iteration variant, so the implementation's shader/pipeline
// dedup cache cannot serve the request and a real Tint compile is measured.
std::string uniqueComputeWgsl(uint64_t i) {
    return "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n"
           "@compute @workgroup_size(64)\n"
           "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
           "    data[gid.x] = data[gid.x] + " + std::to_string(i) + "u;\n"
           "}\n";
}

const char* kRenderWgsl = R"(
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    return vec4f(f32(i) * 0.1, 0.0, 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
)";

std::string uniqueRenderWgsl(uint64_t i) {
    return "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
           "    return vec4f(f32(i) * " + std::to_string(0.1 + static_cast<double>(i)) + ", 0.0, 0.0, 1.0);\n"
           "}\n"
           "@fragment fn fs() -> @location(0) vec4f {\n"
           "    return vec4f(1.0, 0.0, 0.0, 1.0);\n"
           "}\n";
}

WGPUShaderModule makeShaderModule(WGPUDevice device, const char* source) {
    WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
    wgsl.code = sv(source);
    WGPUShaderModuleDescriptor descriptor = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    descriptor.nextInChain = &wgsl.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &descriptor);
    if (module == nullptr) {
        fail("wgpuDeviceCreateShaderModule returned NULL");
    }
    return module;
}

// ---------------------------------------------------------------------------
// Fixture: the persistent objects the individual cases operate against
// ---------------------------------------------------------------------------

struct Fixture {
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;

    WGPUBuffer storageBuffer = nullptr;   // 64 KiB, Storage | CopyDst
    WGPUBuffer uniformBuffer = nullptr;   // 256 B, Uniform | CopyDst
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPUBindGroup bindGroup = nullptr;
    WGPUShaderModule computeModule = nullptr;
    WGPUShaderModule renderModule = nullptr;
    WGPUComputePipeline computePipeline = nullptr;
    WGPURenderPipeline renderPipeline = nullptr;
    WGPUTexture colorTexture = nullptr;
    WGPUTextureView colorView = nullptr;

    std::vector<uint8_t> uploadData;
};

WGPUComputePipelineDescriptor computePipelineDescriptor(WGPUPipelineLayout layout, WGPUShaderModule module) {
    WGPUComputePipelineDescriptor descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    descriptor.layout = layout;
    descriptor.compute.module = module;
    descriptor.compute.entryPoint = sv("main");
    return descriptor;
}

void buildFixture(Fixture& f) {
    f.instance = bench::createInstance();
    if (f.instance == nullptr) {
        fail("wgpuCreateInstance returned NULL");
    }

    AdapterState adapterState;
    WGPURequestAdapterCallbackInfo adapterCb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    adapterCb.mode = WGPUCallbackMode_AllowProcessEvents;
    adapterCb.callback = onRequestAdapter;
    adapterCb.userdata1 = &adapterState;
    (void)wgpuInstanceRequestAdapter(f.instance, bench::adapterOptions(), adapterCb);
    if (!pumpUntil(f.instance, [&] { return adapterState.completed; })) {
        fail("requestAdapter timed out");
    }
    if (adapterState.status != WGPURequestAdapterStatus_Success || adapterState.adapter == nullptr) {
        fail("requestAdapter failed: " + adapterState.message);
    }
    f.adapter = adapterState.adapter;

    DeviceState deviceState;
    WGPUDeviceDescriptor deviceDescriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    deviceDescriptor.uncapturedErrorCallbackInfo.callback = onUncapturedError;
    deviceDescriptor.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowProcessEvents;
    deviceDescriptor.deviceLostCallbackInfo.callback = onDeviceLost;
    WGPURequestDeviceCallbackInfo deviceCb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    deviceCb.mode = WGPUCallbackMode_AllowProcessEvents;
    deviceCb.callback = onRequestDevice;
    deviceCb.userdata1 = &deviceState;
    (void)wgpuAdapterRequestDevice(f.adapter, &deviceDescriptor, deviceCb);
    if (!pumpUntil(f.instance, [&] { return deviceState.completed; })) {
        fail("requestDevice timed out");
    }
    if (deviceState.status != WGPURequestDeviceStatus_Success || deviceState.device == nullptr) {
        fail("requestDevice failed: " + deviceState.message);
    }
    f.device = deviceState.device;
    f.queue = wgpuDeviceGetQueue(f.device);

    WGPUBufferDescriptor storageDescriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    storageDescriptor.size = 64 * 1024;
    storageDescriptor.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    f.storageBuffer = wgpuDeviceCreateBuffer(f.device, &storageDescriptor);

    WGPUBufferDescriptor uniformDescriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    uniformDescriptor.size = 256;
    uniformDescriptor.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    f.uniformBuffer = wgpuDeviceCreateBuffer(f.device, &uniformDescriptor);

    WGPUBindGroupLayoutEntry layoutEntry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    layoutEntry.binding = 0;
    layoutEntry.visibility = WGPUShaderStage_Compute;
    layoutEntry.buffer.type = WGPUBufferBindingType_Storage;
    WGPUBindGroupLayoutDescriptor layoutDescriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    layoutDescriptor.entryCount = 1;
    layoutDescriptor.entries = &layoutEntry;
    f.bindGroupLayout = wgpuDeviceCreateBindGroupLayout(f.device, &layoutDescriptor);

    WGPUPipelineLayoutDescriptor pipelineLayoutDescriptor = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipelineLayoutDescriptor.bindGroupLayoutCount = 1;
    pipelineLayoutDescriptor.bindGroupLayouts = &f.bindGroupLayout;
    f.pipelineLayout = wgpuDeviceCreatePipelineLayout(f.device, &pipelineLayoutDescriptor);

    WGPUBindGroupEntry bindEntry = WGPU_BIND_GROUP_ENTRY_INIT;
    bindEntry.binding = 0;
    bindEntry.buffer = f.storageBuffer;
    bindEntry.offset = 0;
    bindEntry.size = 64 * 1024;
    WGPUBindGroupDescriptor bindGroupDescriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bindGroupDescriptor.layout = f.bindGroupLayout;
    bindGroupDescriptor.entryCount = 1;
    bindGroupDescriptor.entries = &bindEntry;
    f.bindGroup = wgpuDeviceCreateBindGroup(f.device, &bindGroupDescriptor);

    f.computeModule = makeShaderModule(f.device, kComputeWgsl);
    f.renderModule = makeShaderModule(f.device, kRenderWgsl);

    WGPUComputePipelineDescriptor computeDescriptor = computePipelineDescriptor(f.pipelineLayout, f.computeModule);
    f.computePipeline = wgpuDeviceCreateComputePipeline(f.device, &computeDescriptor);

    WGPUColorTargetState colorTarget = WGPU_COLOR_TARGET_STATE_INIT;
    colorTarget.format = WGPUTextureFormat_RGBA8Unorm;
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = f.renderModule;
    fragment.entryPoint = sv("fs");
    fragment.targetCount = 1;
    fragment.targets = &colorTarget;
    WGPURenderPipelineDescriptor renderDescriptor = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    renderDescriptor.layout = nullptr; // auto layout
    renderDescriptor.vertex.module = f.renderModule;
    renderDescriptor.vertex.entryPoint = sv("vs");
    renderDescriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    renderDescriptor.fragment = &fragment;
    f.renderPipeline = wgpuDeviceCreateRenderPipeline(f.device, &renderDescriptor);

    WGPUTextureDescriptor textureDescriptor = WGPU_TEXTURE_DESCRIPTOR_INIT;
    textureDescriptor.dimension = WGPUTextureDimension_2D;
    textureDescriptor.size = {256, 256, 1};
    textureDescriptor.format = WGPUTextureFormat_RGBA8Unorm;
    textureDescriptor.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    f.colorTexture = wgpuDeviceCreateTexture(f.device, &textureDescriptor);
    f.colorView = wgpuTextureCreateView(f.colorTexture, nullptr);

    f.uploadData.assign(4096, 0x5a);

    if (f.storageBuffer == nullptr || f.uniformBuffer == nullptr || f.bindGroupLayout == nullptr ||
        f.pipelineLayout == nullptr || f.bindGroup == nullptr || f.computePipeline == nullptr ||
        f.renderPipeline == nullptr || f.colorTexture == nullptr || f.colorView == nullptr) {
        fail("fixture construction produced a NULL object");
    }
}

void destroyFixture(Fixture& f) {
    wgpuTextureViewRelease(f.colorView);
    wgpuTextureRelease(f.colorTexture);
    wgpuRenderPipelineRelease(f.renderPipeline);
    wgpuComputePipelineRelease(f.computePipeline);
    wgpuShaderModuleRelease(f.renderModule);
    wgpuShaderModuleRelease(f.computeModule);
    wgpuBindGroupRelease(f.bindGroup);
    wgpuPipelineLayoutRelease(f.pipelineLayout);
    wgpuBindGroupLayoutRelease(f.bindGroupLayout);
    wgpuBufferRelease(f.uniformBuffer);
    wgpuBufferRelease(f.storageBuffer);
    wgpuQueueRelease(f.queue);
    wgpuDeviceRelease(f.device);
    wgpuAdapterRelease(f.adapter);
    wgpuInstanceRelease(f.instance);
}

// ---------------------------------------------------------------------------
// Reusable command-recording bodies
// ---------------------------------------------------------------------------

WGPURenderPassEncoder beginColorPass(WGPUCommandEncoder encoder, WGPUTextureView view) {
    WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    attachment.view = view;
    attachment.loadOp = WGPULoadOp_Clear;
    attachment.storeOp = WGPUStoreOp_Store;
    attachment.clearValue = {0.0, 0.0, 0.0, 1.0};
    WGPURenderPassDescriptor descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    descriptor.colorAttachmentCount = 1;
    descriptor.colorAttachments = &attachment;
    return wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
}

struct QueueDoneState {
    bool completed = false;
};

void onQueueWorkDone(WGPUQueueWorkDoneStatus, WGPUStringView, void* userdata1, void*) {
    static_cast<QueueDoneState*>(userdata1)->completed = true;
}

void waitForQueue(Fixture& f) {
    QueueDoneState state;
    WGPUQueueWorkDoneCallbackInfo info = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    info.mode = WGPUCallbackMode_AllowProcessEvents;
    info.callback = onQueueWorkDone;
    info.userdata1 = &state;
    (void)wgpuQueueOnSubmittedWorkDone(f.queue, info);
    if (!pumpUntil(f.instance, [&] { return state.completed; })) {
        fail("onSubmittedWorkDone timed out");
    }
}

// ---------------------------------------------------------------------------
// The cases
// ---------------------------------------------------------------------------

struct Case {
    const char* name;
    uint64_t iters;
    std::function<void(Fixture&, uint64_t)> body;
};

// `kDrawsPerPass` draws are recorded per timed iteration in the per-draw case;
// the reported figure is divided by it so the column reads as cost-per-draw.
constexpr uint64_t kDrawsPerPass = 100;

std::vector<Case> makeCases() {
    std::vector<Case> cases;

    cases.push_back({"buffer/create_destroy", 20000, [](Fixture& f, uint64_t) {
        WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
        d.size = 256;
        d.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        WGPUBuffer b = wgpuDeviceCreateBuffer(f.device, &d);
        wgpuBufferRelease(b);
    }});

    cases.push_back({"buffer/create_mapped_unmap", 10000, [](Fixture& f, uint64_t) {
        WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
        d.size = 256;
        d.usage = WGPUBufferUsage_CopySrc;
        d.mappedAtCreation = true;
        WGPUBuffer b = wgpuDeviceCreateBuffer(f.device, &d);
        void* p = wgpuBufferGetMappedRange(b, 0, 256);
        if (p == nullptr) {
            fail("getMappedRange returned NULL for a mappedAtCreation buffer");
        }
        wgpuBufferUnmap(b);
        wgpuBufferRelease(b);
    }});

    cases.push_back({"bindgroup/create_destroy", 20000, [](Fixture& f, uint64_t) {
        WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
        e.binding = 0;
        e.buffer = f.storageBuffer;
        e.offset = 0;
        e.size = 64 * 1024;
        WGPUBindGroupDescriptor d = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        d.layout = f.bindGroupLayout;
        d.entryCount = 1;
        d.entries = &e;
        WGPUBindGroup g = wgpuDeviceCreateBindGroup(f.device, &d);
        wgpuBindGroupRelease(g);
    }});

    cases.push_back({"queue/write_buffer_4kb", 20000, [](Fixture& f, uint64_t) {
        wgpuQueueWriteBuffer(f.queue, f.storageBuffer, 0, f.uploadData.data(), f.uploadData.size());
    }});

    // `queue/write_buffer_4kb` alone is not a like-for-like comparison: an
    // implementation is free to defer the upload into the next submit, moving
    // the cost out of the timed region. These two drain the queue, so whatever
    // was deferred is paid for inside the measurement.
    cases.push_back({"queue/write_buffer_then_wait", 500, [](Fixture& f, uint64_t) {
        wgpuQueueWriteBuffer(f.queue, f.storageBuffer, 0, f.uploadData.data(), f.uploadData.size());
        waitForQueue(f);
    }});

    cases.push_back({"frame/10writes_dispatch_submit_wait", 300, [](Fixture& f, uint64_t) {
        for (int w = 0; w < 10; ++w) {
            wgpuQueueWriteBuffer(f.queue, f.storageBuffer, static_cast<uint64_t>(w) * 4096,
                                 f.uploadData.data(), f.uploadData.size());
        }
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(f.device, nullptr);
        WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(e, nullptr);
        wgpuComputePassEncoderSetPipeline(p, f.computePipeline);
        wgpuComputePassEncoderSetBindGroup(p, 0, f.bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(p, 16, 1, 1);
        wgpuComputePassEncoderEnd(p);
        wgpuComputePassEncoderRelease(p);
        WGPUCommandBuffer c = wgpuCommandEncoderFinish(e, nullptr);
        wgpuQueueSubmit(f.queue, 1, &c);
        wgpuCommandBufferRelease(c);
        wgpuCommandEncoderRelease(e);
        waitForQueue(f);
    }});

    cases.push_back({"shader/create_cached", 5000, [](Fixture& f, uint64_t) {
        WGPUShaderModule m = makeShaderModule(f.device, kComputeWgsl);
        wgpuShaderModuleRelease(m);
    }});

    cases.push_back({"shader/create_unique", 200, [](Fixture& f, uint64_t i) {
        const std::string source = uniqueComputeWgsl(i);
        WGPUShaderModule m = makeShaderModule(f.device, source.c_str());
        wgpuShaderModuleRelease(m);
    }});

    cases.push_back({"pipeline/compute_cached", 5000, [](Fixture& f, uint64_t) {
        WGPUComputePipelineDescriptor d = computePipelineDescriptor(f.pipelineLayout, f.computeModule);
        WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(f.device, &d);
        wgpuComputePipelineRelease(p);
    }});

    cases.push_back({"pipeline/compute_unique", 100, [](Fixture& f, uint64_t i) {
        const std::string source = uniqueComputeWgsl(i);
        WGPUShaderModule m = makeShaderModule(f.device, source.c_str());
        WGPUComputePipelineDescriptor d = computePipelineDescriptor(f.pipelineLayout, m);
        WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(f.device, &d);
        if (p == nullptr) {
            fail("createComputePipeline returned NULL");
        }
        wgpuComputePipelineRelease(p);
        wgpuShaderModuleRelease(m);
    }});

    cases.push_back({"pipeline/render_unique", 60, [](Fixture& f, uint64_t i) {
        const std::string source = uniqueRenderWgsl(i);
        WGPUShaderModule m = makeShaderModule(f.device, source.c_str());
        WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
        target.format = WGPUTextureFormat_RGBA8Unorm;
        WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
        fragment.module = m;
        fragment.entryPoint = sv("fs");
        fragment.targetCount = 1;
        fragment.targets = &target;
        WGPURenderPipelineDescriptor d = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
        d.layout = nullptr;
        d.vertex.module = m;
        d.vertex.entryPoint = sv("vs");
        d.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        d.fragment = &fragment;
        WGPURenderPipeline p = wgpuDeviceCreateRenderPipeline(f.device, &d);
        if (p == nullptr) {
            fail("createRenderPipeline returned NULL");
        }
        wgpuRenderPipelineRelease(p);
        wgpuShaderModuleRelease(m);
    }});

    cases.push_back({"encode/empty_encoder_finish", 20000, [](Fixture& f, uint64_t) {
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(f.device, nullptr);
        WGPUCommandBuffer c = wgpuCommandEncoderFinish(e, nullptr);
        wgpuCommandBufferRelease(c);
        wgpuCommandEncoderRelease(e);
    }});

    cases.push_back({"encode/compute_1_dispatch", 20000, [](Fixture& f, uint64_t) {
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(f.device, nullptr);
        WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(e, nullptr);
        wgpuComputePassEncoderSetPipeline(p, f.computePipeline);
        wgpuComputePassEncoderSetBindGroup(p, 0, f.bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(p, 1, 1, 1);
        wgpuComputePassEncoderEnd(p);
        wgpuComputePassEncoderRelease(p);
        WGPUCommandBuffer c = wgpuCommandEncoderFinish(e, nullptr);
        wgpuCommandBufferRelease(c);
        wgpuCommandEncoderRelease(e);
    }});

    cases.push_back({"encode/render_pass_empty", 20000, [](Fixture& f, uint64_t) {
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(f.device, nullptr);
        WGPURenderPassEncoder p = beginColorPass(e, f.colorView);
        wgpuRenderPassEncoderEnd(p);
        wgpuRenderPassEncoderRelease(p);
        WGPUCommandBuffer c = wgpuCommandEncoderFinish(e, nullptr);
        wgpuCommandBufferRelease(c);
        wgpuCommandEncoderRelease(e);
    }});

    cases.push_back({"encode/render_draw", 2000, [](Fixture& f, uint64_t) {
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(f.device, nullptr);
        WGPURenderPassEncoder p = beginColorPass(e, f.colorView);
        wgpuRenderPassEncoderSetPipeline(p, f.renderPipeline);
        for (uint64_t d = 0; d < kDrawsPerPass; ++d) {
            wgpuRenderPassEncoderDraw(p, 3, 1, 0, 0);
        }
        wgpuRenderPassEncoderEnd(p);
        wgpuRenderPassEncoderRelease(p);
        WGPUCommandBuffer c = wgpuCommandEncoderFinish(e, nullptr);
        wgpuCommandBufferRelease(c);
        wgpuCommandEncoderRelease(e);
    }});

    cases.push_back({"submit/empty", 10000, [](Fixture& f, uint64_t) {
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(f.device, nullptr);
        WGPUCommandBuffer c = wgpuCommandEncoderFinish(e, nullptr);
        wgpuQueueSubmit(f.queue, 1, &c);
        wgpuCommandBufferRelease(c);
        wgpuCommandEncoderRelease(e);
    }});

    cases.push_back({"submit/compute_1_dispatch", 5000, [](Fixture& f, uint64_t) {
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(f.device, nullptr);
        WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(e, nullptr);
        wgpuComputePassEncoderSetPipeline(p, f.computePipeline);
        wgpuComputePassEncoderSetBindGroup(p, 0, f.bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(p, 1, 1, 1);
        wgpuComputePassEncoderEnd(p);
        wgpuComputePassEncoderRelease(p);
        WGPUCommandBuffer c = wgpuCommandEncoderFinish(e, nullptr);
        wgpuQueueSubmit(f.queue, 1, &c);
        wgpuCommandBufferRelease(c);
        wgpuCommandEncoderRelease(e);
    }});

    // Executes a render pass containing `kDrawsPerPass` draws. Unlike
    // `encode/render_draw` this submits and drains, so it prices what the
    // implementation actually asks the GPU to do for a multi-draw pass — an
    // implementation that starts a fresh GPU render pass per draw pays for an
    // attachment load/store each time, which recording alone cannot reveal.
    cases.push_back({"submit/render_100_draws_wait", 200, [](Fixture& f, uint64_t) {
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(f.device, nullptr);
        WGPURenderPassEncoder p = beginColorPass(e, f.colorView);
        wgpuRenderPassEncoderSetPipeline(p, f.renderPipeline);
        for (uint64_t d = 0; d < kDrawsPerPass; ++d) {
            wgpuRenderPassEncoderDraw(p, 3, 1, 0, 0);
        }
        wgpuRenderPassEncoderEnd(p);
        wgpuRenderPassEncoderRelease(p);
        WGPUCommandBuffer c = wgpuCommandEncoderFinish(e, nullptr);
        wgpuQueueSubmit(f.queue, 1, &c);
        wgpuCommandBufferRelease(c);
        wgpuCommandEncoderRelease(e);
        waitForQueue(f);
    }});

    // Includes the GPU round trip and the event-loop wakeup: a latency figure,
    // not a pure CPU-overhead one.
    cases.push_back({"submit/compute_wait_idle", 500, [](Fixture& f, uint64_t) {
        WGPUCommandEncoder e = wgpuDeviceCreateCommandEncoder(f.device, nullptr);
        WGPUComputePassEncoder p = wgpuCommandEncoderBeginComputePass(e, nullptr);
        wgpuComputePassEncoderSetPipeline(p, f.computePipeline);
        wgpuComputePassEncoderSetBindGroup(p, 0, f.bindGroup, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(p, 1, 1, 1);
        wgpuComputePassEncoderEnd(p);
        wgpuComputePassEncoderRelease(p);
        WGPUCommandBuffer c = wgpuCommandEncoderFinish(e, nullptr);
        wgpuQueueSubmit(f.queue, 1, &c);
        wgpuCommandBufferRelease(c);
        wgpuCommandEncoderRelease(e);
        waitForQueue(f);
    }});

    return cases;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fail("missing value for " + arg);
            }
            return argv[++i];
        };
        if (arg == "--reps") {
            options.reps = std::strtoull(next().c_str(), nullptr, 10);
        } else if (arg == "--scale") {
            options.scale = std::strtod(next().c_str(), nullptr);
        } else if (arg == "--filter") {
            options.filter = next();
        } else if (arg == "--tsv") {
            options.tsv = true;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("usage: bench [--reps N] [--scale F] [--filter SUBSTR] [--tsv]\n");
            std::exit(0);
        } else {
            fail("unknown argument: " + arg);
        }
    }
    if (options.reps == 0) {
        fail("--reps must be >= 1");
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    const Options options = parseOptions(argc, argv);

    Fixture fixture;
    buildFixture(fixture);

    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    if (wgpuAdapterGetInfo(fixture.adapter, &info) == WGPUStatus_Success) {
        std::fprintf(stderr, "bench: %s on %s (%s)\n", bench::backendName(),
                     toString(info.device).c_str(), toString(info.description).c_str());
        wgpuAdapterInfoFreeMembers(info);
    }

    std::vector<Result> results;
    for (const Case& c : makeCases()) {
        const std::string name = c.name;
        if (!options.filter.empty() && name.find(options.filter) == std::string::npos) {
            continue;
        }
        if (!c.body) {
            continue;
        }
        Result result = run(options, name, c.iters, [&](uint64_t i) { c.body(fixture, i); });
        if (name == "encode/render_draw") {
            result.minNsPerOp /= static_cast<double>(kDrawsPerPass);
            result.medianNsPerOp /= static_cast<double>(kDrawsPerPass);
            result.iters *= kDrawsPerPass;
        }
        results.push_back(result);
        std::fflush(stdout);
    }

    // The queue is left with in-flight work by the submit cases; drain it before
    // tearing the fixture down so destruction cost is not attributed elsewhere.
    waitForQueue(fixture);

    if (options.tsv) {
        std::printf("backend\tcase\titers\tmin_ns\tmedian_ns\n");
        for (const Result& r : results) {
            std::printf("%s\t%s\t%" PRIu64 "\t%.1f\t%.1f\n", bench::backendName(), r.name.c_str(), r.iters,
                        r.minNsPerOp, r.medianNsPerOp);
        }
    } else {
        std::printf("\n%-32s %10s %12s %12s\n", "case", "iters", "min ns/op", "med ns/op");
        std::printf("%-32s %10s %12s %12s\n", "--------------------------------", "----------", "------------",
                    "------------");
        for (const Result& r : results) {
            std::printf("%-32s %10" PRIu64 " %12.1f %12.1f\n", r.name.c_str(), r.iters, r.minNsPerOp,
                        r.medianNsPerOp);
        }
    }

    destroyFixture(fixture);
    return 0;
}
