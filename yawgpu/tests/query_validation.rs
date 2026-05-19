use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_test::{assert_device_error, wait, ValidationTest};

const MAX_QUERY_COUNT: u32 = 4096;

#[test]
fn occlusion_query_set_creation_and_reflection() {
    let test = ValidationTest::new();
    unsafe {
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 18);
        assert!(!query_set.is_null());
        assert_eq!(
            yawgpu::wgpuQuerySetGetType(query_set),
            native::WGPUQueryType_Occlusion
        );
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 18);
        yawgpu::wgpuQuerySetRelease(query_set);

        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
        assert!(!query_set.is_null());
        assert_eq!(
            yawgpu::wgpuQuerySetGetType(query_set),
            native::WGPUQueryType_Occlusion
        );
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 1);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn query_set_count_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let query_set = create_query_set(
            test.device(),
            native::WGPUQueryType_Occlusion,
            MAX_QUERY_COUNT,
        );
        assert!(!query_set.is_null());
        yawgpu::wgpuQuerySetRelease(query_set);

        let mut query_set = std::ptr::null();
        assert_device_error!({
            query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 0);
        });
        assert!(!query_set.is_null());
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 0);
        yawgpu::wgpuQuerySetRelease(query_set);

        let mut query_set = std::ptr::null();
        assert_device_error!({
            query_set = create_query_set(
                test.device(),
                native::WGPUQueryType_Occlusion,
                MAX_QUERY_COUNT + 1,
            );
        });
        assert!(!query_set.is_null());
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), MAX_QUERY_COUNT + 1);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn query_set_type_is_validated_and_reflected_for_error_query_sets() {
    let test = ValidationTest::new();
    unsafe {
        let invalid_type = 0xFFFF;
        let mut query_set = std::ptr::null();
        assert_device_error!({
            query_set = create_query_set(test.device(), invalid_type, 76);
        });
        assert!(!query_set.is_null());
        assert_eq!(yawgpu::wgpuQuerySetGetType(query_set), invalid_type);
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 76);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn timestamp_query_requires_feature() {
    let test = ValidationTest::new();
    unsafe {
        let mut query_set = std::ptr::null();
        assert_device_error!({
            query_set = create_query_set(test.device(), native::WGPUQueryType_Timestamp, 1);
        });
        assert!(!query_set.is_null());
        assert_eq!(
            yawgpu::wgpuQuerySetGetType(query_set),
            native::WGPUQueryType_Timestamp
        );
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 1);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn timestamp_query_succeeds_when_feature_is_requested() {
    unsafe {
        let fixture = FeatureDeviceFixture::new(&[native::WGPUFeatureName_TimestampQuery]);
        let query_set = create_query_set(fixture.device, native::WGPUQueryType_Timestamp, 1);

        assert!(!query_set.is_null());
        assert_eq!(
            yawgpu::wgpuQuerySetGetType(query_set),
            native::WGPUQueryType_Timestamp
        );
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 1);
        assert!(fixture.errors());
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn destroy_is_idempotent_and_reflection_stays_available() {
    let test = ValidationTest::new();
    unsafe {
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 2);
        yawgpu::wgpuQuerySetDestroy(query_set);
        yawgpu::wgpuQuerySetDestroy(query_set);
        yawgpu::wgpuQuerySetSetLabel(query_set, string_view("destroyed query set"));

        assert_eq!(
            yawgpu::wgpuQuerySetGetType(query_set),
            native::WGPUQueryType_Occlusion
        );
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 2);
        assert!(test.errors().is_empty());
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn query_set_add_ref_release_balances() {
    let test = ValidationTest::new();
    unsafe {
        let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
        yawgpu::wgpuQuerySetAddRef(query_set);
        yawgpu::wgpuQuerySetRelease(query_set);
        assert_eq!(yawgpu::wgpuQuerySetGetCount(query_set), 1);
        yawgpu::wgpuQuerySetRelease(query_set);
    }
}

#[test]
fn timestamp_query_feature_plumbing() {
    unsafe {
        let default = FeatureDeviceFixture::new(&[]);
        assert_eq!(
            yawgpu::wgpuDeviceHasFeature(default.device, native::WGPUFeatureName_TimestampQuery),
            0
        );

        let timestamp = FeatureDeviceFixture::new(&[native::WGPUFeatureName_TimestampQuery]);
        assert_eq!(
            yawgpu::wgpuDeviceHasFeature(timestamp.device, native::WGPUFeatureName_TimestampQuery),
            1
        );

        let mut features = supported_features_init();
        yawgpu::wgpuDeviceGetFeatures(timestamp.device, &mut features);
        let features_slice = std::slice::from_raw_parts(features.features, features.featureCount);
        assert!(features_slice.contains(&native::WGPUFeatureName_TimestampQuery));
        yawgpu::wgpuSupportedFeaturesFreeMembers(features);
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

    fn errors(&self) -> bool {
        self.errors.lock().expect("error lock").is_empty()
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

fn supported_features_init() -> native::WGPUSupportedFeatures {
    native::WGPUSupportedFeatures {
        featureCount: 0,
        features: std::ptr::null(),
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
