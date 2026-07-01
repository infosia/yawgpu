use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::wait;

#[derive(Default)]
struct RequestAdapterResult {
    status: native::WGPURequestAdapterStatus,
    adapter: native::WGPUAdapter,
    message: String,
}

#[derive(Default)]
struct RequestDeviceResult {
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    message: String,
}

#[test]
fn texture_formats_tier1_implies_rg11b10ufloat_renderable() {
    unsafe {
        let fixture = Fixture::new(native::WGPUFeatureLevel_Core);
        let device = request_device(
            fixture.instance,
            fixture.adapter,
            &[native::WGPUFeatureName_TextureFormatsTier1],
        );

        assert_device_success(&device);
        assert_has_feature(
            device.device,
            native::WGPUFeatureName_TextureFormatsTier1,
            true,
        );
        assert_has_feature(
            device.device,
            native::WGPUFeatureName_RG11B10UfloatRenderable,
            true,
        );

        yawgpu::wgpuDeviceRelease(device.device);
    }
}

#[test]
fn texture_formats_tier2_implies_tier1_and_rg11b10ufloat_renderable() {
    unsafe {
        let fixture = Fixture::new(native::WGPUFeatureLevel_Core);
        let device = request_device(
            fixture.instance,
            fixture.adapter,
            &[native::WGPUFeatureName_TextureFormatsTier2],
        );

        assert_device_success(&device);
        assert_has_feature(
            device.device,
            native::WGPUFeatureName_TextureFormatsTier2,
            true,
        );
        assert_has_feature(
            device.device,
            native::WGPUFeatureName_TextureFormatsTier1,
            true,
        );
        assert_has_feature(
            device.device,
            native::WGPUFeatureName_RG11B10UfloatRenderable,
            true,
        );

        yawgpu::wgpuDeviceRelease(device.device);
    }
}

#[test]
fn core_adapter_explicit_core_features_and_limits_has_feature() {
    unsafe {
        let fixture = Fixture::new(native::WGPUFeatureLevel_Core);
        let device = request_device(
            fixture.instance,
            fixture.adapter,
            &[native::WGPUFeatureName_CoreFeaturesAndLimits],
        );

        assert_device_success(&device);
        assert_has_feature(
            device.device,
            native::WGPUFeatureName_CoreFeaturesAndLimits,
            true,
        );

        yawgpu::wgpuDeviceRelease(device.device);
    }
}

#[test]
fn core_adapter_implicitly_enables_core_features_and_limits() {
    unsafe {
        let fixture = Fixture::new(native::WGPUFeatureLevel_Core);
        let device = request_device(fixture.instance, fixture.adapter, &[]);

        assert_device_success(&device);
        assert_has_feature(
            device.device,
            native::WGPUFeatureName_CoreFeaturesAndLimits,
            true,
        );

        yawgpu::wgpuDeviceRelease(device.device);
    }
}

#[test]
fn compatibility_adapter_does_not_implicitly_enable_core_features_and_limits() {
    unsafe {
        let fixture = Fixture::new(native::WGPUFeatureLevel_Compatibility);
        let device = request_device(fixture.instance, fixture.adapter, &[]);

        assert_device_success(&device);
        assert_has_feature(
            device.device,
            native::WGPUFeatureName_CoreFeaturesAndLimits,
            false,
        );

        yawgpu::wgpuDeviceRelease(device.device);
    }
}

#[test]
fn compatibility_adapter_can_explicitly_request_core_features_and_limits() {
    unsafe {
        let fixture = Fixture::new(native::WGPUFeatureLevel_Compatibility);
        let device = request_device(
            fixture.instance,
            fixture.adapter,
            &[native::WGPUFeatureName_CoreFeaturesAndLimits],
        );

        assert_device_success(&device);
        assert_has_feature(
            device.device,
            native::WGPUFeatureName_CoreFeaturesAndLimits,
            true,
        );

        yawgpu::wgpuDeviceRelease(device.device);
    }
}

#[test]
fn get_features_round_trip_and_free_members() {
    unsafe {
        let fixture = Fixture::new(native::WGPUFeatureLevel_Core);

        let mut adapter_features = supported_features_init();
        yawgpu::wgpuAdapterGetFeatures(fixture.adapter, &mut adapter_features);
        let adapter_features_slice =
            std::slice::from_raw_parts(adapter_features.features, adapter_features.featureCount);
        assert!(adapter_features_slice.contains(&native::WGPUFeatureName_TextureFormatsTier2));
        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(
                fixture.adapter,
                native::WGPUFeatureName_TextureFormatsTier1
            ),
            1
        );
        yawgpu::wgpuSupportedFeaturesFreeMembers(adapter_features);

        let device = request_device(
            fixture.instance,
            fixture.adapter,
            &[native::WGPUFeatureName_TextureFormatsTier2],
        );
        assert_device_success(&device);

        let mut device_features = supported_features_init();
        yawgpu::wgpuDeviceGetFeatures(device.device, &mut device_features);
        let device_features_slice =
            std::slice::from_raw_parts(device_features.features, device_features.featureCount);
        assert!(device_features_slice.contains(&native::WGPUFeatureName_TextureFormatsTier2));
        assert!(device_features_slice.contains(&native::WGPUFeatureName_TextureFormatsTier1));
        assert!(device_features_slice.contains(&native::WGPUFeatureName_RG11B10UfloatRenderable));
        yawgpu::wgpuSupportedFeaturesFreeMembers(device_features);

        yawgpu::wgpuDeviceRelease(device.device);
    }
}

#[test]
fn unsupported_required_feature_fails_request_device() {
    unsafe {
        let fixture = Fixture::new(native::WGPUFeatureLevel_Core);
        // A standard WebGPU feature yawgpu does not yet implement, so the Noop
        // adapter does not advertise it and requesting it must fail.
        let device = request_device(
            fixture.instance,
            fixture.adapter,
            &[native::WGPUFeatureName_ClipDistances],
        );

        assert_eq!(device.status, native::WGPURequestDeviceStatus_Error);
        assert!(device.device.is_null());
        assert!(!device.message.is_empty());
    }
}

struct Fixture {
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
}

impl Fixture {
    unsafe fn new(feature_level: native::WGPUFeatureLevel) -> Self {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        assert!(!instance.is_null());

        let adapter = request_adapter(instance, feature_level);
        assert_eq!(adapter.status, native::WGPURequestAdapterStatus_Success);
        assert!(!adapter.adapter.is_null());
        assert!(adapter.message.is_empty());

        Self {
            instance,
            adapter: adapter.adapter,
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        unsafe {
            yawgpu::wgpuAdapterRelease(self.adapter);
            yawgpu::wgpuInstanceRelease(self.instance);
        }
    }
}

unsafe fn request_adapter(
    instance: native::WGPUInstance,
    feature_level: native::WGPUFeatureLevel,
) -> RequestAdapterResult {
    let mut result = RequestAdapterResult::default();
    let options = native::WGPURequestAdapterOptions {
        nextInChain: std::ptr::null_mut(),
        featureLevel: feature_level,
        powerPreference: native::WGPUPowerPreference_Undefined,
        forceFallbackAdapter: 0,
        backendType: native::WGPUBackendType_Undefined,
        compatibleSurface: std::ptr::null(),
    };
    let callback_info = native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_adapter_callback),
        userdata1: (&mut result as *mut RequestAdapterResult).cast(),
        userdata2: std::ptr::null_mut(),
    };

    let future = yawgpu::wgpuInstanceRequestAdapter(instance, &options, callback_info);
    wait(instance, future);
    result
}

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    required_features: &[native::WGPUFeatureName],
) -> RequestDeviceResult {
    let mut result = RequestDeviceResult::default();
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
        requiredFeatureCount: required_features.len(),
        requiredFeatures: if required_features.is_empty() {
            std::ptr::null()
        } else {
            required_features.as_ptr()
        },
        requiredLimits: std::ptr::null(),
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: native::WGPUStringView {
                data: std::ptr::null(),
                length: 0,
            },
        },
        deviceLostCallbackInfo: unsafe { std::mem::zeroed() },
        uncapturedErrorCallbackInfo: unsafe { std::mem::zeroed() },
    };
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut result as *mut RequestDeviceResult).cast(),
        userdata2: std::ptr::null_mut(),
    };

    let descriptor_ptr = if required_features.is_empty() {
        std::ptr::null()
    } else {
        &descriptor
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, descriptor_ptr, callback_info);
    wait(instance, future);
    result
}

fn supported_features_init() -> native::WGPUSupportedFeatures {
    native::WGPUSupportedFeatures {
        featureCount: 0,
        features: std::ptr::null(),
    }
}

unsafe fn assert_has_feature(
    device: native::WGPUDevice,
    feature: native::WGPUFeatureName,
    expected: bool,
) {
    assert_eq!(yawgpu::wgpuDeviceHasFeature(device, feature) != 0, expected);
}

fn assert_device_success(result: &RequestDeviceResult) {
    assert_eq!(result.status, native::WGPURequestDeviceStatus_Success);
    assert!(!result.device.is_null());
    assert!(result.message.is_empty());
}

unsafe extern "C" fn request_adapter_callback(
    status: native::WGPURequestAdapterStatus,
    adapter: native::WGPUAdapter,
    message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let result = &mut *(userdata1 as *mut RequestAdapterResult);
    result.status = status;
    result.adapter = adapter;
    result.message = string_view_to_string(message);
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
