use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_test::{assert_device_error, wait, ValidationTest};

#[test]
fn chromium_disable_uniformity_analysis_is_rejected() {
    let test = ValidationTest::new();
    unsafe {
        assert_wgsl_ok(&test, "@compute @workgroup_size(1) fn main() {}");
        assert_wgsl_error(
            &test,
            r"
enable chromium_disable_uniformity_analysis;

@compute @workgroup_size(8) fn main(@builtin(local_invocation_id) id: vec3u) {
    if (id.x == 0u) {
        workgroupBarrier();
    }
}
",
        );
    }
}

#[test]
fn bind_group_layout_entry_array_size_is_rejected_above_one() {
    let test = ValidationTest::new();
    unsafe {
        let mut entry = texture_entry(0);
        entry.bindingArraySize = 0;
        assert_layout_ok(&test, &[entry]);

        entry.bindingArraySize = 1;
        assert_layout_ok(&test, &[entry]);

        entry.bindingArraySize = 2;
        assert_layout_error(&test, &[entry]);
    }
}

#[test]
fn static_binding_array_in_wgsl_is_rejected() {
    let test = ValidationTest::new();
    unsafe {
        assert_wgsl_error(
            &test,
            r"
@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 3>;
@fragment fn fs() -> @location(0) u32 {
    let _ = textures[0];
    return 0u;
}
",
        );

        assert_wgsl_error(
            &test,
            r"
@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 1>;
@fragment fn fs() -> @location(0) u32 {
    let _ = textures[0];
    return 0u;
}
",
        );
    }
}

#[test]
fn write_timestamp_requires_timestamp_query_feature() {
    let test = ValidationTest::new();
    unsafe {
        let mut query_set = std::ptr::null();
        assert_device_error!({
            query_set = create_query_set(test.device(), native::WGPUQueryType_Timestamp, 1);
        });
        assert!(!query_set.is_null());
        test.clear_errors();

        let encoder = create_encoder(test.device());
        yawgpu::wgpuCommandEncoderWriteTimestamp(encoder, query_set, 0);
        finish_encoder_error(&test, encoder);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn write_timestamp_with_timestamp_query_feature_is_allowed() {
    unsafe {
        let fixture = FeatureDeviceFixture::new(&[native::WGPUFeatureName_TimestampQuery]);
        let query_set = create_query_set(fixture.device, native::WGPUQueryType_Timestamp, 1);
        assert!(fixture.errors().is_empty());

        let encoder = create_encoder(fixture.device);
        yawgpu::wgpuCommandEncoderWriteTimestamp(encoder, query_set, 0);
        finish_feature_encoder_ok(&fixture, encoder);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

unsafe fn assert_wgsl_ok(test: &ValidationTest, source: &str) {
    test.clear_errors();
    let module = create_wgsl_module(test.device(), source);
    assert!(!module.is_null());
    assert!(test.errors().is_empty());
    yawgpu::wgpuShaderModuleRelease(module);
}

unsafe fn assert_wgsl_error(test: &ValidationTest, source: &str) {
    let mut module = std::ptr::null();
    assert_device_error!({
        module = create_wgsl_module(test.device(), source);
    });
    assert!(!module.is_null());
    yawgpu::wgpuShaderModuleRelease(module);
}

unsafe fn create_wgsl_module(device: native::WGPUDevice, source: &str) -> native::WGPUShaderModule {
    let mut wgsl = native::WGPUShaderSourceWGSL {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ShaderSourceWGSL,
        },
        code: string_view(source),
    };
    let descriptor = native::WGPUShaderModuleDescriptor {
        nextInChain: (&mut wgsl.chain) as *mut _,
        label: empty_string_view(),
    };
    yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor)
}

unsafe fn assert_layout_ok(test: &ValidationTest, entries: &[native::WGPUBindGroupLayoutEntry]) {
    test.clear_errors();
    let layout = create_layout(test.device(), entries);
    assert!(!layout.is_null());
    assert!(test.errors().is_empty());
    yawgpu::wgpuBindGroupLayoutRelease(layout);
}

unsafe fn assert_layout_error(test: &ValidationTest, entries: &[native::WGPUBindGroupLayoutEntry]) {
    let mut layout = std::ptr::null();
    assert_device_error!({
        layout = create_layout(test.device(), entries);
    });
    assert!(!layout.is_null());
    yawgpu::wgpuBindGroupLayoutRelease(layout);
}

unsafe fn create_layout(
    device: native::WGPUDevice,
    entries: &[native::WGPUBindGroupLayoutEntry],
) -> native::WGPUBindGroupLayout {
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor)
}

fn texture_entry(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility: native::WGPUShaderStage_Fragment,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_BindingNotUsed,
            hasDynamicOffset: 0,
            minBindingSize: 0,
        },
        sampler: native::WGPUSamplerBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUSamplerBindingType_BindingNotUsed,
        },
        texture: native::WGPUTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            sampleType: native::WGPUTextureSampleType_Float,
            viewDimension: native::WGPUTextureViewDimension_2D,
            multisampled: 0,
        },
        storageTexture: native::WGPUStorageTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            access: native::WGPUStorageTextureAccess_BindingNotUsed,
            format: native::WGPUTextureFormat_Undefined,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
        },
    }
}

unsafe fn create_query_set(
    device: native::WGPUDevice,
    query_type: native::WGPUQueryType,
    count: u32,
) -> native::WGPUQuerySet {
    let descriptor = native::WGPUQuerySetDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        type_: query_type,
        count,
    };
    yawgpu::wgpuDeviceCreateQuerySet(device, &descriptor)
}

unsafe fn create_encoder(device: native::WGPUDevice) -> native::WGPUCommandEncoder {
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    assert!(!encoder.is_null());
    encoder
}

unsafe fn finish_encoder_error(test: &ValidationTest, encoder: native::WGPUCommandEncoder) {
    let mut command_buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        },
        None,
    );
    assert!(!command_buffer.is_null());
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn finish_feature_encoder_ok(
    fixture: &FeatureDeviceFixture,
    encoder: native::WGPUCommandEncoder,
) {
    fixture.clear_errors();
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    assert!(fixture.errors().is_empty());
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

struct FeatureDeviceFixture {
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    device: native::WGPUDevice,
    errors: Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
}

impl FeatureDeviceFixture {
    unsafe fn new(required_features: &[native::WGPUFeatureName]) -> Self {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        assert!(!instance.is_null());

        let mut adapter: native::WGPUAdapter = std::ptr::null();
        let adapter_callback_info = native::WGPURequestAdapterCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(request_adapter_callback),
            userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future =
            yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), adapter_callback_info);
        wait(instance, future);
        assert!(!adapter.is_null());

        let descriptor = native::WGPUDeviceDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            requiredFeatureCount: required_features.len(),
            requiredFeatures: required_features.as_ptr(),
            requiredLimits: std::ptr::null(),
            defaultQueue: queue_descriptor_init(),
            deviceLostCallbackInfo: unsafe { std::mem::zeroed() },
            uncapturedErrorCallbackInfo: unsafe { std::mem::zeroed() },
        };
        let mut device: native::WGPUDevice = std::ptr::null();
        let device_callback_info = native::WGPURequestDeviceCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(request_device_callback),
            userdata1: (&mut device as *mut native::WGPUDevice).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future = yawgpu::wgpuAdapterRequestDevice(adapter, &descriptor, device_callback_info);
        wait(instance, future);
        assert!(!device.is_null());

        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );

        Self {
            instance,
            adapter,
            device,
            errors,
        }
    }

    fn clear_errors(&self) {
        self.errors.lock().expect("error lock").clear();
    }

    fn errors(&self) -> Vec<yawgpu_core::DeviceError> {
        self.errors.lock().expect("error lock").clone()
    }
}

impl Drop for FeatureDeviceFixture {
    fn drop(&mut self) {
        unsafe {
            yawgpu::wgpuDeviceRelease(self.device);
            yawgpu::wgpuAdapterRelease(self.adapter);
            yawgpu::wgpuInstanceRelease(self.instance);
        }
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
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestDeviceStatus_Success);
    *(userdata1 as *mut native::WGPUDevice) = device;
}

fn queue_descriptor_init() -> native::WGPUQueueDescriptor {
    native::WGPUQueueDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
    }
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
