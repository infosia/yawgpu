use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{wait, ValidationTest};

#[derive(Default)]
struct RequestDeviceResult {
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    message: String,
}

#[test]
fn no_required_limits_success_reports_defaults() {
    let test = ValidationTest::new();

    unsafe {
        let result = request_device(test.instance(), test.adapter(), None);
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Success);
        assert!(!result.device.is_null());
        assert!(result.message.is_empty());

        let limits = get_device_limits(result.device);
        assert_eq!(limits.maxBindGroups, 4);

        yawgpu::wgpuDeviceRelease(result.device);
    }
}

#[test]
fn required_limits_all_defaults_success_reports_defaults() {
    let test = ValidationTest::new();

    unsafe {
        let required = get_adapter_limits(test.adapter());
        let result = request_device(test.instance(), test.adapter(), Some(&required));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Success);
        assert!(!result.device.is_null());
        assert!(result.message.is_empty());

        let limits = get_device_limits(result.device);
        assert_eq!(limits.maxTextureArrayLayers, 256);
        assert_eq!(limits.maxBindGroups, 4);

        yawgpu::wgpuDeviceRelease(result.device);
    }
}

#[test]
fn maximum_limit_above_supported_fails_and_worse_than_default_clamps() {
    let test = ValidationTest::new();

    unsafe {
        let supported = get_adapter_limits(test.adapter());

        let mut required = undefined_limits();
        required.maxBindGroups = supported.maxBindGroups + 1;
        let result = request_device(test.instance(), test.adapter(), Some(&required));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Error);
        assert!(result.device.is_null());
        assert!(!result.message.is_empty());

        required.maxBindGroups = 3;
        let result = request_device(test.instance(), test.adapter(), Some(&required));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Success);
        assert!(!result.device.is_null());
        assert!(result.message.is_empty());

        let limits = get_device_limits(result.device);
        assert_eq!(limits.maxBindGroups, 4);

        yawgpu::wgpuDeviceRelease(result.device);
    }
}

#[test]
fn alignment_limit_below_supported_fails_and_worse_than_default_clamps() {
    let test = ValidationTest::new();

    unsafe {
        let supported = get_adapter_limits(test.adapter());

        let mut required = undefined_limits();
        required.minUniformBufferOffsetAlignment = supported.minUniformBufferOffsetAlignment / 2;
        let result = request_device(test.instance(), test.adapter(), Some(&required));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Error);
        assert!(result.device.is_null());
        assert!(!result.message.is_empty());

        required.minUniformBufferOffsetAlignment = 512;
        let result = request_device(test.instance(), test.adapter(), Some(&required));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Success);
        assert!(!result.device.is_null());
        assert!(result.message.is_empty());

        let limits = get_device_limits(result.device);
        assert_eq!(limits.minUniformBufferOffsetAlignment, 256);

        yawgpu::wgpuDeviceRelease(result.device);
    }
}

#[test]
fn max_immediate_size_always_uses_supported_limit() {
    let test = ValidationTest::new();

    unsafe {
        let supported = get_adapter_limits(test.adapter());
        assert!(supported.maxImmediateSize > 16);

        let mut required = undefined_limits();
        required.maxImmediateSize = 16;
        let result = request_device(test.instance(), test.adapter(), Some(&required));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Success);
        assert!(!result.device.is_null());
        assert!(result.message.is_empty());

        let limits = get_device_limits(result.device);
        assert_eq!(limits.maxImmediateSize, supported.maxImmediateSize);

        yawgpu::wgpuDeviceRelease(result.device);
    }
}

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    required_limits: Option<&native::WGPULimits>,
) -> RequestDeviceResult {
    let mut result = RequestDeviceResult::default();
    let mut descriptor: native::WGPUDeviceDescriptor = std::mem::zeroed();
    if let Some(required_limits) = required_limits {
        descriptor.requiredLimits = required_limits;
    }

    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut result as *mut RequestDeviceResult).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let descriptor_ptr = if required_limits.is_some() {
        &descriptor
    } else {
        std::ptr::null()
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, descriptor_ptr, callback_info);
    wait(instance, future);
    result
}

unsafe fn get_adapter_limits(adapter: native::WGPUAdapter) -> native::WGPULimits {
    let mut limits = undefined_limits();
    let status = yawgpu::wgpuAdapterGetLimits(adapter, &mut limits);
    assert_eq!(status, native::WGPUStatus_Success);
    limits
}

unsafe fn get_device_limits(device: native::WGPUDevice) -> native::WGPULimits {
    let mut limits = undefined_limits();
    let status = yawgpu::wgpuDeviceGetLimits(device, &mut limits);
    assert_eq!(status, native::WGPUStatus_Success);
    limits
}

fn undefined_limits() -> native::WGPULimits {
    native::WGPULimits {
        nextInChain: std::ptr::null_mut(),
        maxTextureDimension1D: native::WGPU_LIMIT_U32_UNDEFINED,
        maxTextureDimension2D: native::WGPU_LIMIT_U32_UNDEFINED,
        maxTextureDimension3D: native::WGPU_LIMIT_U32_UNDEFINED,
        maxTextureArrayLayers: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBindGroups: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBindGroupsPlusVertexBuffers: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBindingsPerBindGroup: native::WGPU_LIMIT_U32_UNDEFINED,
        maxDynamicUniformBuffersPerPipelineLayout: native::WGPU_LIMIT_U32_UNDEFINED,
        maxDynamicStorageBuffersPerPipelineLayout: native::WGPU_LIMIT_U32_UNDEFINED,
        maxSampledTexturesPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxSamplersPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxStorageBuffersPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxStorageTexturesPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxUniformBuffersPerShaderStage: native::WGPU_LIMIT_U32_UNDEFINED,
        maxUniformBufferBindingSize: native::WGPU_LIMIT_U64_UNDEFINED,
        maxStorageBufferBindingSize: native::WGPU_LIMIT_U64_UNDEFINED,
        minUniformBufferOffsetAlignment: native::WGPU_LIMIT_U32_UNDEFINED,
        minStorageBufferOffsetAlignment: native::WGPU_LIMIT_U32_UNDEFINED,
        maxVertexBuffers: native::WGPU_LIMIT_U32_UNDEFINED,
        maxBufferSize: native::WGPU_LIMIT_U64_UNDEFINED,
        maxVertexAttributes: native::WGPU_LIMIT_U32_UNDEFINED,
        maxVertexBufferArrayStride: native::WGPU_LIMIT_U32_UNDEFINED,
        maxInterStageShaderVariables: native::WGPU_LIMIT_U32_UNDEFINED,
        maxColorAttachments: native::WGPU_LIMIT_U32_UNDEFINED,
        maxColorAttachmentBytesPerSample: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupStorageSize: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeInvocationsPerWorkgroup: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupSizeX: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupSizeY: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupSizeZ: native::WGPU_LIMIT_U32_UNDEFINED,
        maxComputeWorkgroupsPerDimension: native::WGPU_LIMIT_U32_UNDEFINED,
        maxImmediateSize: native::WGPU_LIMIT_U32_UNDEFINED,
    }
}

unsafe extern "C" fn request_device_callback(
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let result = &mut *(userdata1 as *mut RequestDeviceResult);
    result.status = status;
    result.device = device;
    result.message = string_view_to_string(message);
}

unsafe fn string_view_to_string(value: native::WGPUStringView) -> String {
    if value.data.is_null() {
        return String::new();
    }
    let bytes = if value.length == usize::MAX {
        std::ffi::CStr::from_ptr(value.data).to_bytes()
    } else {
        std::slice::from_raw_parts(value.data.cast::<u8>(), value.length)
    };
    String::from_utf8_lossy(bytes).into_owned()
}
