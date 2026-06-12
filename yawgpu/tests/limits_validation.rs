use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::wait;

#[derive(Default)]
struct RequestDeviceResult {
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    message: String,
}

#[test]
fn no_required_limits_success_reports_defaults() {
    let test = AdapterFixture::new();

    unsafe {
        let result = request_device(test.instance(), test.adapter(), None);
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Success);
        assert!(!result.device.is_null());
        assert!(result.message.is_empty());

        let limits = get_device_limits(result.device);
        assert_eq!(limits.maxBindGroups, 4);
        assert_eq!(limits.maxTextureDimension1D, 8192);
        assert_eq!(limits.maxTextureDimension2D, 8192);
        assert_eq!(limits.maxUniformBufferBindingSize, 65_536);
        assert_eq!(limits.maxInterStageShaderVariables, 16);
        assert_eq!(limits.maxColorAttachments, 8);
        assert_eq!(limits.maxComputeInvocationsPerWorkgroup, 256);
        assert_eq!(limits.maxComputeWorkgroupSizeX, 256);
        assert_eq!(limits.maxComputeWorkgroupSizeY, 256);
        assert_eq!(limits.maxImmediateSize, 0);

        yawgpu::wgpuDeviceRelease(result.device);
    }
}

#[test]
fn required_limits_all_defaults_success_reports_defaults() {
    let test = AdapterFixture::new();

    unsafe {
        let required = get_adapter_limits(test.adapter());
        let result = request_device(test.instance(), test.adapter(), Some(&required));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Success);
        assert!(!result.device.is_null());
        assert!(result.message.is_empty());

        let limits = get_device_limits(result.device);
        assert_eq!(limits.maxTextureArrayLayers, 256);
        assert_eq!(limits.maxBindGroups, 4);
        assert_eq!(limits.maxTextureDimension2D, 8192);
        assert_eq!(limits.maxColorAttachments, 8);

        yawgpu::wgpuDeviceRelease(result.device);
    }
}

#[test]
fn maximum_limit_above_supported_fails_and_worse_than_default_clamps() {
    let test = AdapterFixture::new();

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
    let test = AdapterFixture::new();

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
fn adapter_request_device_is_single_use_after_success_only() {
    let test = AdapterFixture::new();

    unsafe {
        let unsupported = 0x7FFF_FFFE as native::WGPUFeatureName;
        let invalid_descriptor = native::WGPUDeviceDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            requiredFeatureCount: 1,
            requiredFeatures: &unsupported,
            requiredLimits: std::ptr::null(),
            defaultQueue: native::WGPUQueueDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
            },
            deviceLostCallbackInfo: native::WGPUDeviceLostCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: 0,
                callback: None,
                userdata1: std::ptr::null_mut(),
                userdata2: std::ptr::null_mut(),
            },
            uncapturedErrorCallbackInfo: native::WGPUUncapturedErrorCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                callback: None,
                userdata1: std::ptr::null_mut(),
                userdata2: std::ptr::null_mut(),
            },
        };
        let invalid =
            request_device_with_descriptor(test.instance(), test.adapter(), &invalid_descriptor);
        assert_eq!(invalid.status, native::WGPURequestDeviceStatus_Error);
        assert!(invalid.device.is_null());

        let first = request_device(test.instance(), test.adapter(), None);
        assert_eq!(first.status, native::WGPURequestDeviceStatus_Success);
        assert!(!first.device.is_null());

        let second = request_device(test.instance(), test.adapter(), None);
        assert_eq!(second.status, native::WGPURequestDeviceStatus_Error);
        assert!(second.device.is_null());
        assert!(second.message.contains("consumed"));

        yawgpu::wgpuDeviceRelease(first.device);
    }
}

#[test]
fn max_immediate_size_always_uses_supported_limit() {
    let test = AdapterFixture::new();

    unsafe {
        let supported = get_adapter_limits(test.adapter());
        assert_eq!(supported.maxImmediateSize, 0);

        let required = undefined_limits();
        let result = request_device(test.instance(), test.adapter(), Some(&required));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Success);
        assert!(!result.device.is_null());
        assert!(result.message.is_empty());

        let limits = get_device_limits(result.device);
        assert_eq!(limits.maxImmediateSize, 0);

        yawgpu::wgpuDeviceRelease(result.device);
    }
}

#[test]
fn per_stage_limits_defaults_reported_via_compat_chain() {
    // The adapter and device must populate a chained WGPUCompatibilityModeLimits
    // with spec-default per-stage values (CTS kLimitInfos defaults).
    let test = AdapterFixture::new();

    unsafe {
        // Adapter limits via compat chain.
        let mut compat = native::WGPUCompatibilityModeLimits {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_CompatibilityModeLimits,
            },
            maxStorageBuffersInVertexStage: 0,
            maxStorageTexturesInVertexStage: 0,
            maxStorageBuffersInFragmentStage: 0,
            maxStorageTexturesInFragmentStage: 0,
        };
        let mut adapter_limits = undefined_limits();
        adapter_limits.nextInChain = (&mut compat.chain) as *mut native::WGPUChainedStruct;
        let status = yawgpu::wgpuAdapterGetLimits(test.adapter(), &mut adapter_limits);
        assert_eq!(status, native::WGPUStatus_Success);
        // CTS table: maxStorageBuffersIn{Vertex,Fragment}Stage default = 8,
        // maxStorageTexturesIn{Vertex,Fragment}Stage default = 4.
        assert_eq!(compat.maxStorageBuffersInVertexStage, 8);
        assert_eq!(compat.maxStorageBuffersInFragmentStage, 8);
        assert_eq!(compat.maxStorageTexturesInVertexStage, 4);
        assert_eq!(compat.maxStorageTexturesInFragmentStage, 4);

        // Device limits via compat chain.
        let result = request_device(test.instance(), test.adapter(), None);
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Success);
        assert!(!result.device.is_null());

        let mut dev_compat = native::WGPUCompatibilityModeLimits {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_CompatibilityModeLimits,
            },
            maxStorageBuffersInVertexStage: 0,
            maxStorageTexturesInVertexStage: 0,
            maxStorageBuffersInFragmentStage: 0,
            maxStorageTexturesInFragmentStage: 0,
        };
        let mut dev_limits = undefined_limits();
        dev_limits.nextInChain = (&mut dev_compat.chain) as *mut native::WGPUChainedStruct;
        let dev_status = yawgpu::wgpuDeviceGetLimits(result.device, &mut dev_limits);
        assert_eq!(dev_status, native::WGPUStatus_Success);
        assert_eq!(dev_compat.maxStorageBuffersInVertexStage, 8);
        assert_eq!(dev_compat.maxStorageBuffersInFragmentStage, 8);
        assert_eq!(dev_compat.maxStorageTexturesInVertexStage, 4);
        assert_eq!(dev_compat.maxStorageTexturesInFragmentStage, 4);

        yawgpu::wgpuDeviceRelease(result.device);
    }
}

#[test]
fn per_stage_limits_above_supported_rejected() {
    // Requesting a per-stage limit above the supported value must fail.
    let test = AdapterFixture::new();

    unsafe {
        // maxStorageBuffersInVertexStage supported = 8; request 9 (one above).
        let mut compat = native::WGPUCompatibilityModeLimits {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_CompatibilityModeLimits,
            },
            maxStorageBuffersInVertexStage: 9,
            maxStorageTexturesInVertexStage: native::WGPU_LIMIT_U32_UNDEFINED,
            maxStorageBuffersInFragmentStage: native::WGPU_LIMIT_U32_UNDEFINED,
            maxStorageTexturesInFragmentStage: native::WGPU_LIMIT_U32_UNDEFINED,
        };
        let mut req = undefined_limits();
        req.nextInChain = (&mut compat.chain) as *mut native::WGPUChainedStruct;

        let result = request_device(test.instance(), test.adapter(), Some(&req));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Error);
        assert!(result.device.is_null());
        assert!(!result.message.is_empty());
    }
}

#[test]
fn per_stage_limits_at_supported_delivered() {
    // Requesting per-stage limits at the supported value must succeed and the
    // device must report those exact values back.
    let test = AdapterFixture::new();

    unsafe {
        let mut req_compat = native::WGPUCompatibilityModeLimits {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_CompatibilityModeLimits,
            },
            // Supported == default == 8 and 4.
            maxStorageBuffersInVertexStage: 8,
            maxStorageTexturesInVertexStage: 4,
            maxStorageBuffersInFragmentStage: 8,
            maxStorageTexturesInFragmentStage: 4,
        };
        let mut req = undefined_limits();
        req.nextInChain = (&mut req_compat.chain) as *mut native::WGPUChainedStruct;

        let result = request_device(test.instance(), test.adapter(), Some(&req));
        assert_eq!(result.status, native::WGPURequestDeviceStatus_Success);
        assert!(!result.device.is_null());
        assert!(result.message.is_empty());

        let mut dev_compat = native::WGPUCompatibilityModeLimits {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_CompatibilityModeLimits,
            },
            maxStorageBuffersInVertexStage: 0,
            maxStorageTexturesInVertexStage: 0,
            maxStorageBuffersInFragmentStage: 0,
            maxStorageTexturesInFragmentStage: 0,
        };
        let mut dev_limits = undefined_limits();
        dev_limits.nextInChain = (&mut dev_compat.chain) as *mut native::WGPUChainedStruct;
        let status = yawgpu::wgpuDeviceGetLimits(result.device, &mut dev_limits);
        assert_eq!(status, native::WGPUStatus_Success);
        assert_eq!(dev_compat.maxStorageBuffersInVertexStage, 8);
        assert_eq!(dev_compat.maxStorageBuffersInFragmentStage, 8);
        assert_eq!(dev_compat.maxStorageTexturesInVertexStage, 4);
        assert_eq!(dev_compat.maxStorageTexturesInFragmentStage, 4);

        yawgpu::wgpuDeviceRelease(result.device);
    }
}

struct AdapterFixture {
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
}

impl AdapterFixture {
    fn new() -> Self {
        unsafe {
            let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
            assert!(!instance.is_null());

            let mut adapter: native::WGPUAdapter = std::ptr::null();
            let callback_info = native::WGPURequestAdapterCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(request_adapter_callback),
                userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let future =
                yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info);
            wait(instance, future);
            assert!(!adapter.is_null());

            Self { instance, adapter }
        }
    }

    fn instance(&self) -> native::WGPUInstance {
        self.instance
    }

    fn adapter(&self) -> native::WGPUAdapter {
        self.adapter
    }
}

impl Drop for AdapterFixture {
    fn drop(&mut self) {
        unsafe {
            yawgpu::wgpuAdapterRelease(self.adapter);
            yawgpu::wgpuInstanceRelease(self.instance);
        }
    }
}

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    required_limits: Option<&native::WGPULimits>,
) -> RequestDeviceResult {
    let mut descriptor: native::WGPUDeviceDescriptor = std::mem::zeroed();
    if let Some(required_limits) = required_limits {
        descriptor.requiredLimits = required_limits;
    }
    let descriptor_ptr = if required_limits.is_some() {
        &descriptor
    } else {
        std::ptr::null()
    };
    request_device_with_descriptor(instance, adapter, descriptor_ptr)
}

unsafe fn request_device_with_descriptor(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    descriptor: *const native::WGPUDeviceDescriptor,
) -> RequestDeviceResult {
    let mut result = RequestDeviceResult::default();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut result as *mut RequestDeviceResult).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, descriptor, callback_info);
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

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

unsafe extern "C" fn request_adapter_callback(
    status: native::WGPURequestAdapterStatus,
    adapter: native::WGPUAdapter,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestAdapterStatus_Success);
    *(userdata1 as *mut native::WGPUAdapter) = adapter;
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
