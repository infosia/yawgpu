#include "dawn/dawn_proc.h"
#include "webgpu/webgpu_cpp.h"
#include "yawgpu_dawn.h"

#include <atomic>
#include <cstring>
#include <iostream>
#include <iterator>

extern "C" const void* smokeBackendChain(int metal);

namespace {
constexpr uint64_t kTimeoutNs = 10'000'000'000;
constexpr size_t kBytes = 4 * sizeof(uint32_t);

const char* runCompute(int metal, uint8_t* bytes, std::atomic_bool& deviceError) {
    wgpu::Adapter adapter;
    wgpu::Device device;
    bool mapped = false;
    constexpr auto timedWait = wgpu::InstanceFeatureName::TimedWaitAny;
    wgpu::InstanceDescriptor instanceDesc{
        .nextInChain = static_cast<const wgpu::ChainedStruct*>(smokeBackendChain(metal)),
        .requiredFeatureCount = 1, .requiredFeatures = &timedWait};
    auto instance = wgpu::CreateInstance(&instanceDesc);
    if (!instance) return "CreateInstance";

    const char* adapterNames[] = {"use_dxc", "allow_unsafe_apis", "decompose_uniform_buffers",
                                  "use_vulkan_memory_model", "vulkan_enable_f16_on_nvidia"};
    wgpu::DawnTogglesDescriptor adapterToggles;
    adapterToggles.enabledToggleCount = std::size(adapterNames);
    adapterToggles.enabledToggles = adapterNames;
    wgpu::RequestAdapterOptions options{.nextInChain = &adapterToggles};
    constexpr auto mode = wgpu::CallbackMode::WaitAnyOnly;
    auto adapterFuture = instance.RequestAdapter(&options, mode,
        [&](wgpu::RequestAdapterStatus status, wgpu::Adapter result, wgpu::StringView message) {
            if (status == wgpu::RequestAdapterStatus::Success) adapter = std::move(result);
            else std::cerr << std::string_view(message) << '\n';
        });
    if (instance.WaitAny(adapterFuture, kTimeoutNs) != wgpu::WaitStatus::Success || !adapter)
        return "RequestAdapter / WaitAny";

    const char* enabled[] = {"skip_validation", "d3d_disable_ieee_strictness",
                             "disable_robustness", "lazy_clear_resource_on_first_use"};
    const char* disabled[] = {"timestamp_quantization"};
    wgpu::DawnTogglesDescriptor deviceToggles;
    deviceToggles.enabledToggleCount = std::size(enabled);
    deviceToggles.enabledToggles = enabled;
    deviceToggles.disabledToggleCount = std::size(disabled);
    deviceToggles.disabledToggles = disabled;
    wgpu::DeviceDescriptor deviceDesc;
    deviceDesc.nextInChain = &deviceToggles;
    deviceDesc.SetUncapturedErrorCallback(
        [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView message, std::atomic_bool* failed) {
            failed->store(true);
            std::cerr << "Device error: " << std::string_view(message) << '\n';
        }, &deviceError);
    auto deviceFuture = adapter.RequestDevice(&deviceDesc, mode,
        [&](wgpu::RequestDeviceStatus status, wgpu::Device result, wgpu::StringView message) {
            if (status == wgpu::RequestDeviceStatus::Success) device = std::move(result);
            else std::cerr << std::string_view(message) << '\n';
        });
    if (instance.WaitAny(deviceFuture, kTimeoutNs) != wgpu::WaitStatus::Success || !device)
        return "RequestDevice / WaitAny";

    wgpu::BufferDescriptor storageDesc{
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, .size = kBytes};
    wgpu::BufferDescriptor readbackDesc{
        .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst, .size = kBytes};
    auto storage = device.CreateBuffer(&storageDesc);
    auto readback = device.CreateBuffer(&readbackDesc);
    if (!storage || !readback) return "CreateBuffer";

    wgpu::ShaderSourceWGSL wgsl;
    wgsl.code = R"(
        @group(0) @binding(0) var<storage, read_write> words: array<u32>;
        @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
            words[id.x] = 0x17100000u + id.x;
        })";
    wgpu::ShaderModuleDescriptor shaderDesc{.nextInChain = &wgsl};
    auto shader = device.CreateShaderModule(&shaderDesc);
    if (!shader) return "CreateShaderModule";
    wgpu::ComputePipelineDescriptor pipelineDesc;
    pipelineDesc.compute.module = shader;
    pipelineDesc.compute.entryPoint = "main";
    auto pipeline = device.CreateComputePipeline(&pipelineDesc);
    if (!pipeline) return "CreateComputePipeline";
    auto layout = pipeline.GetBindGroupLayout(0);
    if (!layout) return "GetBindGroupLayout";
    wgpu::BindGroupEntry entry{.binding = 0, .buffer = storage, .size = kBytes};
    wgpu::BindGroupDescriptor groupDesc{.layout = layout, .entryCount = 1, .entries = &entry};
    auto group = device.CreateBindGroup(&groupDesc);
    auto encoder = device.CreateCommandEncoder();
    if (!group || !encoder) return "CreateBindGroup / CreateCommandEncoder";
    auto pass = encoder.BeginComputePass();
    if (!pass) return "BeginComputePass";
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, group);
    pass.DispatchWorkgroups(4);
    pass.End();
    encoder.CopyBufferToBuffer(storage, 0, readback, 0, kBytes);
    auto commands = encoder.Finish();
    auto queue = device.GetQueue();
    if (!commands || !queue) return "Finish / GetQueue";
    queue.Submit(1, &commands);

    auto mapFuture = readback.MapAsync(wgpu::MapMode::Read, 0, kBytes, mode,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            mapped = status == wgpu::MapAsyncStatus::Success;
            if (!mapped) std::cerr << std::string_view(message) << '\n';
        });
    if (instance.WaitAny(mapFuture, kTimeoutNs) != wgpu::WaitStatus::Success || !mapped)
        return "MapAsync / WaitAny";
    const void* data = readback.GetConstMappedRange(0, kBytes);
    if (!data) return "GetConstMappedRange";
    std::memcpy(bytes, data, kBytes);
    readback.Unmap();
    return nullptr;
}
}  // namespace

extern "C" const char* smokeCompute(int metal, uint8_t* bytes, size_t size) {
    if (size != kBytes) return "wrong readback size";
    dawnProcSetProcs(yawgpuDawnGetProcTable());
    std::atomic_bool deviceError{false};
    const char* error = runCompute(metal, bytes, deviceError);
    dawnProcSetProcs(nullptr);
    return deviceError.load() ? "uncaptured device error" : error;
}
