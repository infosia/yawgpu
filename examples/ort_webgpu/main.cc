// ONNX Runtime's WebGPU execution provider running on yawgpu through the Dawn
// compatibility layer (Block 171).
//
// ONNX Runtime is built in its "external Dawn" mode against yawgpu's Dawn API package and
// receives yawgpu's DawnProcTable through the `dawnProcTable` provider option. This host
// runs an Abs model with CPU fallback disabled and checks the result.
//
//   ort_webgpu                 ONNX Runtime creates the instance and device
//   ort_webgpu --host_device   the host creates them and passes the handles
//   ort_webgpu --noop          select the Noop backend (session setup only, no result check)
//   ort_webgpu --model <file>  run <file> on WebGPU and on the CPU provider with the same
//                              generated float inputs and compare the outputs
//
// Build: see build.sh next to this file.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"

#include "webgpu.h"
#include "yawgpu.h"
#include "yawgpu_dawn.h"

namespace {

// https://github.com/onnx/onnx/blob/v1.15.0/onnx/backend/test/data/node/test_abs/model.onnx
constexpr uint8_t kAbsModel[] = {
    8,   7,  18, 12,  98,  97,  99,  107, 101, 110, 100, 45,  116, 101, 115, 116, 58, 73, 10,
    11,  10, 1,  120, 18,  1,   121, 34,  3,   65,  98,  115, 18,  8,   116, 101, 115, 116, 95,
    97,  98, 115, 90, 23,  10,  1,   120, 18,  18,  10,  16,  8,   1,   18,  12,  10,  2,   8,
    3,   10, 2,  8,   4,   10,  2,   8,   5,   98,  23,  10,  1,   121, 18,  18,  10,  16,  8,
    1,   18, 12, 10,  2,   8,   3,   10,  2,   8,   4,   10,  2,   8,   5,   66,  4,   10,  0,
    16,  13};

std::string PointerOption(const void* pointer) {
  return std::to_string(reinterpret_cast<uintptr_t>(pointer));
}

struct HostDevice {
  WGPUInstance instance = nullptr;
  WGPUDevice device = nullptr;
};

// Creates an instance and a device the way a host that owns its GPU context would: the
// device must enable ImplicitDeviceSynchronization for ONNX Runtime to accept it.
HostDevice CreateHostDevice(uint32_t backend) {
  HostDevice host;

  YaWGPUInstanceBackendSelect select = {};
  select.chain.sType = (WGPUSType)YAWGPU_STYPE_INSTANCE_BACKEND_SELECT;
  select.backend = backend;
  const WGPUInstanceFeatureName instance_feature = WGPUInstanceFeatureName_TimedWaitAny;
  WGPUInstanceDescriptor instance_descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
  instance_descriptor.nextInChain = &select.chain;
  instance_descriptor.requiredFeatureCount = 1;
  instance_descriptor.requiredFeatures = &instance_feature;
  host.instance = wgpuCreateInstance(&instance_descriptor);
  if (host.instance == nullptr) {
    throw std::runtime_error("wgpuCreateInstance failed");
  }

  WGPUAdapter adapter = nullptr;
  WGPURequestAdapterCallbackInfo adapter_callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
  adapter_callback.mode = WGPUCallbackMode_WaitAnyOnly;
  adapter_callback.callback = [](WGPURequestAdapterStatus status, WGPUAdapter result, WGPUStringView,
                                 void* userdata1, void*) {
    if (status == WGPURequestAdapterStatus_Success) {
      *static_cast<WGPUAdapter*>(userdata1) = result;
    }
  };
  adapter_callback.userdata1 = &adapter;
  WGPUFutureWaitInfo wait = {};
  wait.future = wgpuInstanceRequestAdapter(host.instance, nullptr, adapter_callback);
  wgpuInstanceWaitAny(host.instance, 1, &wait, UINT64_MAX);
  if (adapter == nullptr) {
    throw std::runtime_error("no adapter");
  }
  if (!wgpuAdapterHasFeature(adapter, WGPUFeatureName_ImplicitDeviceSynchronization)) {
    throw std::runtime_error("adapter lacks ImplicitDeviceSynchronization");
  }

  const WGPUFeatureName device_feature = WGPUFeatureName_ImplicitDeviceSynchronization;
  WGPUDeviceDescriptor device_descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
  device_descriptor.requiredFeatureCount = 1;
  device_descriptor.requiredFeatures = &device_feature;
  device_descriptor.uncapturedErrorCallbackInfo.callback =
      [](WGPUDevice const*, WGPUErrorType type, WGPUStringView message, void*, void*) {
        std::cerr << "host device error (" << static_cast<int>(type) << "): "
                  << std::string(message.data, message.length) << std::endl;
      };
  WGPURequestDeviceCallbackInfo device_callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
  device_callback.mode = WGPUCallbackMode_WaitAnyOnly;
  device_callback.callback = [](WGPURequestDeviceStatus status, WGPUDevice result, WGPUStringView,
                                void* userdata1, void*) {
    if (status == WGPURequestDeviceStatus_Success) {
      *static_cast<WGPUDevice*>(userdata1) = result;
    }
  };
  device_callback.userdata1 = &host.device;
  wait = {};
  wait.future = wgpuAdapterRequestDevice(adapter, &device_descriptor, device_callback);
  wgpuInstanceWaitAny(host.instance, 1, &wait, UINT64_MAX);
  wgpuAdapterRelease(adapter);
  if (host.device == nullptr) {
    throw std::runtime_error("no device");
  }
  return host;
}

// Runs `model` on `session_options` with deterministic inputs. Supports float32 inputs;
// a dynamic dimension is taken as 1.
std::vector<std::vector<float>> RunModel(Ort::Env& env, const std::string& model,
                                         Ort::SessionOptions& session_options) {
  // ORTCHAR_T is wchar_t on Windows.
  const std::basic_string<ORTCHAR_T> model_path(model.begin(), model.end());
  Ort::Session session{env, model_path.c_str(), session_options};
  Ort::AllocatorWithDefaultOptions allocator;
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<std::string> input_names, output_names;
  std::vector<std::vector<float>> input_data;
  std::vector<std::vector<int64_t>> input_shapes;
  for (size_t index = 0; index < session.GetInputCount(); ++index) {
    input_names.emplace_back(session.GetInputNameAllocated(index, allocator).get());
    // The tensor info is a view into the type info, which must outlive it.
    auto type_info = session.GetInputTypeInfo(index);
    auto info = type_info.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("input " + input_names.back() + " is not float32");
    }
    auto shape = info.GetShape();
    size_t count = 1;
    for (auto& dimension : shape) {
      if (dimension < 0) {
        dimension = 1;
      }
      count *= static_cast<size_t>(dimension);
    }
    std::vector<float> data(count);
    for (size_t element = 0; element < count; ++element) {
      data[element] = std::sin(static_cast<float>(element + 7 * index)) * 2.0f;
    }
    input_data.push_back(std::move(data));
    input_shapes.push_back(std::move(shape));
  }
  for (size_t index = 0; index < session.GetOutputCount(); ++index) {
    output_names.emplace_back(session.GetOutputNameAllocated(index, allocator).get());
  }

  std::vector<Ort::Value> inputs;
  std::vector<const char*> input_name_pointers, output_name_pointers;
  for (size_t index = 0; index < input_names.size(); ++index) {
    inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, input_data[index].data(),
                                                     input_data[index].size(),
                                                     input_shapes[index].data(),
                                                     input_shapes[index].size()));
    input_name_pointers.push_back(input_names[index].c_str());
  }
  for (const auto& name : output_names) {
    output_name_pointers.push_back(name.c_str());
  }
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_name_pointers.data(), inputs.data(),
                             inputs.size(), output_name_pointers.data(),
                             output_name_pointers.size());
  std::vector<std::vector<float>> results;
  for (auto& output : outputs) {
    auto info = output.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("an output is not float32");
    }
    const float* data = output.GetTensorData<float>();
    results.emplace_back(data, data + info.GetElementCount());
  }
  return results;
}

}  // namespace

int main(int argc, char* argv[]) {
  bool host_device_mode = false;
  bool noop = false;
  std::string model;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--host_device") {
      host_device_mode = true;
    } else if (argument == "--noop") {
      noop = true;
    } else if (argument == "--model" && index + 1 < argc) {
      model = argv[++index];
    } else {
      std::cerr << "unknown argument: " << argument << std::endl;
      return 2;
    }
  }

  try {
    const DawnProcTable* procs = yawgpuDawnGetProcTable();
    const uint32_t backend = noop ? YAWGPU_INSTANCE_BACKEND_NOOP :
#if defined(__APPLE__)
                                  YAWGPU_INSTANCE_BACKEND_METAL;
#else
                                  YAWGPU_INSTANCE_BACKEND_VULKAN;
#endif
    // The instance ONNX Runtime creates through the table carries no backend selector.
    yawgpuDawnSetDefaultBackend(backend);

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ort_webgpu"};
    Ort::SessionOptions session_options;
    session_options.DisableMemPattern();
    session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");

    std::unordered_map<std::string, std::string> provider_options;
    provider_options["dawnProcTable"] = PointerOption(procs);

    HostDevice host;
    if (host_device_mode) {
      host = CreateHostDevice(backend);
      provider_options["deviceId"] = "1";
      provider_options["webgpuInstance"] = PointerOption(host.instance);
      provider_options["webgpuDevice"] = PointerOption(host.device);
      provider_options["preserveDevice"] = "1";
    }
    session_options.AppendExecutionProvider("WebGPU", provider_options);

    if (!model.empty()) {
      Ort::SessionOptions cpu_options;
      const auto expected = RunModel(env, model, cpu_options);
      const auto actual = RunModel(env, model, session_options);
      if (noop) {
        std::cout << model << ": ran on the Noop backend (no result to check)." << std::endl;
        return 0;
      }
      float worst = 0.0f;
      size_t elements = 0;
      for (size_t output = 0; output < expected.size(); ++output) {
        if (actual[output].size() != expected[output].size()) {
          throw std::runtime_error("output size differs from the CPU provider");
        }
        for (size_t element = 0; element < expected[output].size(); ++element) {
          const float want = expected[output][element];
          const float error = std::abs(actual[output][element] - want) / (1.0f + std::abs(want));
          if (!(error <= 1e-3f)) {
            std::cerr << model << ": output " << output << " element " << element << ": got "
                      << actual[output][element] << ", CPU provider gives " << want << std::endl;
            return 1;
          }
          worst = std::max(worst, error);
          ++elements;
        }
      }
      std::cout << model << ": " << elements << " output elements match the CPU provider (worst "
                << "relative error " << worst << ")." << std::endl;
      return 0;
    }
    Ort::Session session{env, kAbsModel, sizeof(kAbsModel), session_options};

    const std::array<int64_t, 3> shape{3, 4, 5};
    std::array<float, 60> input_data;
    for (size_t index = 0; index < input_data.size(); ++index) {
      input_data[index] = static_cast<float>(index) - 30.0f;
    }
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                 shape.data(), shape.size());
    const char* input_names[] = {"x"};
    const char* output_names[] = {"y"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
    if (outputs.size() != 1 ||
        outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != input_data.size()) {
      throw std::runtime_error("unexpected Abs output shape");
    }
    if (noop) {
      std::cout << "WebGPU Abs session ran on the Noop backend (no result to check)." << std::endl;
      return 0;
    }
    const float* output_data = outputs[0].GetTensorData<float>();
    for (size_t index = 0; index < input_data.size(); ++index) {
      if (output_data[index] != std::abs(input_data[index])) {
        std::cerr << "mismatch at " << index << ": got " << output_data[index] << ", want "
                  << std::abs(input_data[index]) << std::endl;
        return 1;
      }
    }
    std::cout << "WebGPU Abs inference passed on yawgpu with CPU fallback disabled ("
              << (host_device_mode ? "host-owned device" : "ONNX Runtime-owned device") << ")."
              << std::endl;
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "failed: " << exception.what() << std::endl;
    return 1;
  }
}
