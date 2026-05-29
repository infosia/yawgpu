use std::cell::RefCell;
use std::ffi::CStr;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_core::{DeviceError, ErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RealBackend {
    Gles,
    Metal,
    Vulkan,
}

impl RealBackend {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gles => "gles",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        }
    }
}

#[must_use]
pub fn real_backend_available(backend: RealBackend) -> bool {
    match backend {
        RealBackend::Gles => gles_backend_available(),
        RealBackend::Metal => metal_backend_available(),
        RealBackend::Vulkan => vulkan_backend_available(),
    }
}

#[must_use]
pub fn real_backend_skip_reason(backend: RealBackend) -> Option<String> {
    (!real_backend_available(backend)).then(|| {
        format!(
            "{} backend is unavailable in the real-backend gated harness",
            backend.name()
        )
    })
}

#[cfg(feature = "gles")]
fn gles_backend_available() -> bool {
    let Ok(instance) = yawgpu_hal::gles::GlesInstance::new() else {
        return false;
    };
    let adapters = instance.enumerate_adapters();
    let Some(adapter) = adapters.into_iter().next() else {
        return false;
    };
    adapter.create_device().is_ok()
}

#[cfg(not(feature = "gles"))]
fn gles_backend_available() -> bool {
    false
}

#[cfg(feature = "metal")]
fn metal_backend_available() -> bool {
    !objc2_metal::MTLCopyAllDevices().is_empty()
}

#[cfg(not(feature = "metal"))]
fn metal_backend_available() -> bool {
    false
}

#[cfg(feature = "vulkan")]
fn vulkan_backend_available() -> bool {
    let Ok(entry) = (unsafe { ash::Entry::load() }) else {
        return false;
    };
    let extension_names = [ash::vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr()];
    let create_info = ash::vk::InstanceCreateInfo::default()
        .flags(ash::vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
        .enabled_extension_names(&extension_names);
    let instance = unsafe { entry.create_instance(&create_info, None) };
    let Ok(instance) = instance else {
        return false;
    };
    let available =
        unsafe { instance.enumerate_physical_devices() }.is_ok_and(|devices| !devices.is_empty());
    unsafe {
        instance.destroy_instance(None);
    }
    available
}

#[cfg(not(feature = "vulkan"))]
fn vulkan_backend_available() -> bool {
    false
}

thread_local! {
    static CURRENT_ERRORS: RefCell<Option<Arc<Mutex<Vec<DeviceError>>>>> = const { RefCell::new(None) };
}

pub struct ValidationTest {
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    device: native::WGPUDevice,
    errors: Arc<Mutex<Vec<DeviceError>>>,
}

impl ValidationTest {
    #[must_use]
    pub fn new() -> Self {
        Self::with_device_descriptor(std::ptr::null())
    }

    /// Creates a validation test device with the requested features enabled.
    #[must_use]
    pub fn with_features(features: &[native::WGPUFeatureName]) -> Self {
        let descriptor = device_descriptor(features, std::ptr::null());
        Self::with_device_descriptor(&descriptor)
    }

    /// Creates a validation test device with the requested limits.
    #[must_use]
    pub fn with_limits(limits: native::WGPULimits) -> Self {
        let descriptor = device_descriptor(&[], &limits);
        Self::with_device_descriptor(&descriptor)
    }

    fn with_device_descriptor(descriptor: *const native::WGPUDeviceDescriptor) -> Self {
        unsafe {
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
            let future = yawgpu::wgpuInstanceRequestAdapter(
                instance,
                std::ptr::null(),
                adapter_callback_info,
            );
            wait(instance, future);
            assert!(!adapter.is_null());

            let mut state = RequestDeviceState::default();
            let device_callback_info = native::WGPURequestDeviceCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(request_device_callback),
                userdata1: (&mut state as *mut RequestDeviceState).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let future =
                yawgpu::wgpuAdapterRequestDevice(adapter, descriptor, device_callback_info);
            wait(instance, future);
            assert_eq!(
                state.status,
                native::WGPURequestDeviceStatus_Success,
                "request device failed: {}",
                state.message
            );
            assert!(!state.device.is_null());

            let errors = Arc::new(Mutex::new(Vec::new()));
            let captured_errors = Arc::clone(&errors);
            yawgpu::testing_set_uncaptured_error_callback(
                state.device,
                Some(move |error| captured_errors.lock().expect("error lock").push(error)),
            );
            CURRENT_ERRORS.with(|current| *current.borrow_mut() = Some(Arc::clone(&errors)));

            Self {
                instance,
                adapter,
                device: state.device,
                errors,
            }
        }
    }

    #[must_use]
    pub fn instance(&self) -> native::WGPUInstance {
        self.instance
    }

    #[must_use]
    pub fn adapter(&self) -> native::WGPUAdapter {
        self.adapter
    }

    #[must_use]
    pub fn device(&self) -> native::WGPUDevice {
        self.device
    }

    pub fn clear_errors(&self) {
        self.errors.lock().expect("error lock").clear();
    }

    #[must_use]
    pub fn errors(&self) -> Vec<DeviceError> {
        self.errors.lock().expect("error lock").clone()
    }

    pub fn inject_device_error(&self, message: impl Into<String>) {
        unsafe {
            yawgpu::testing_dispatch_device_error(self.device, ErrorKind::Validation, message);
        }
    }

    pub fn assert_device_error_after<F>(&self, action: F, substring: Option<&str>)
    where
        F: FnOnce(),
    {
        self.clear_errors();
        action();
        let errors = self.errors();
        assert_eq!(
            errors.len(),
            1,
            "expected exactly one device error, got {}: {:?}",
            errors.len(),
            errors
        );
        if let Some(substring) = substring {
            assert!(
                errors[0].message.contains(substring),
                "expected device error message {:?} to contain {:?}",
                errors[0].message,
                substring
            );
        }
    }

    /// Runs `action` and asserts that the device error sink stays empty.
    pub fn expect_no_validation_error<F>(&self, action: F)
    where
        F: FnOnce(),
    {
        self.clear_errors();
        action();
        let errors = self.errors();
        assert!(
            errors.is_empty(),
            "expected no device errors, got {}: {:?}",
            errors.len(),
            errors
        );
    }
}

impl Default for ValidationTest {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ValidationTest {
    fn drop(&mut self) {
        CURRENT_ERRORS.with(|current| *current.borrow_mut() = None);
        unsafe {
            yawgpu::wgpuDeviceRelease(self.device);
            yawgpu::wgpuAdapterRelease(self.adapter);
            yawgpu::wgpuInstanceRelease(self.instance);
        }
    }
}

pub fn assert_current_device_error_after<F>(action: F, substring: Option<&str>)
where
    F: FnOnce(),
{
    let errors = CURRENT_ERRORS.with(|current| {
        current
            .borrow()
            .as_ref()
            .cloned()
            .expect("ValidationTest::new must be called before assert_device_error!")
    });
    errors.lock().expect("error lock").clear();
    action();

    let captured = errors.lock().expect("error lock").clone();
    assert_eq!(
        captured.len(),
        1,
        "expected exactly one device error, got {}: {:?}",
        captured.len(),
        captured
    );
    if let Some(substring) = substring {
        assert!(
            captured[0].message.contains(substring),
            "expected device error message {:?} to contain {:?}",
            captured[0].message,
            substring
        );
    }
}

/// Runs `action` and asserts that the current validation test error sink stays empty.
pub fn expect_no_validation_error<F>(action: F)
where
    F: FnOnce(),
{
    let errors = CURRENT_ERRORS.with(|current| {
        current
            .borrow()
            .as_ref()
            .cloned()
            .expect("ValidationTest::new must be called before expect_no_validation_error")
    });
    errors.lock().expect("error lock").clear();
    action();

    let captured = errors.lock().expect("error lock").clone();
    assert!(
        captured.is_empty(),
        "expected no device errors, got {}: {:?}",
        captured.len(),
        captured
    );
}

/// Returns the Cartesian product of two slices as owned tuples.
#[must_use]
pub fn cartesian2<A, B>(a: &[A], b: &[B]) -> Vec<(A, B)>
where
    A: Clone,
    B: Clone,
{
    a.iter()
        .flat_map(|item_a| b.iter().map(move |item_b| (item_a.clone(), item_b.clone())))
        .collect()
}

/// Returns the Cartesian product of three slices as owned tuples.
#[must_use]
pub fn cartesian3<A, B, C>(a: &[A], b: &[B], c: &[C]) -> Vec<(A, B, C)>
where
    A: Clone,
    B: Clone,
    C: Clone,
{
    a.iter()
        .flat_map(|item_a| {
            b.iter().flat_map(move |item_b| {
                c.iter()
                    .map(move |item_c| (item_a.clone(), item_b.clone(), item_c.clone()))
            })
        })
        .collect()
}

#[macro_export]
macro_rules! assert_device_error {
    ($expr:expr $(,)?) => {{
        $crate::assert_current_device_error_after(|| $expr, None);
    }};
    ($expr:expr, $substring:expr $(,)?) => {{
        $crate::assert_current_device_error_after(|| $expr, Some($substring));
    }};
}

/// Drives an instance until the provided future completes or reports an error.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle.
pub unsafe fn wait_for_future(instance: native::WGPUInstance, future: native::WGPUFuture) {
    unsafe {
        yawgpu::wgpuInstanceProcessEvents(instance);
        let mut wait_info = native::WGPUFutureWaitInfo {
            future,
            completed: 0,
        };
        let status = yawgpu::wgpuInstanceWaitAny(instance, 1, &mut wait_info, 0);
        assert_ne!(status, native::WGPUWaitStatus_Error);
    }
}

/// Drives an instance until the provided future has had a chance to fire.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle.
pub unsafe fn wait(instance: native::WGPUInstance, future: native::WGPUFuture) {
    unsafe { wait_for_future(instance, future) }
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
    let state = &mut *(userdata1 as *mut RequestDeviceState);
    state.status = status;
    state.device = device;
    state.message = string_view_to_string(message);
}

#[derive(Debug)]
struct RequestDeviceState {
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    message: String,
}

impl Default for RequestDeviceState {
    fn default() -> Self {
        Self {
            status: native::WGPURequestDeviceStatus_Error,
            device: std::ptr::null(),
            message: String::new(),
        }
    }
}

fn device_descriptor(
    features: &[native::WGPUFeatureName],
    limits: *const native::WGPULimits,
) -> native::WGPUDeviceDescriptor {
    native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: string_view_init(),
        requiredFeatureCount: features.len(),
        requiredFeatures: features.as_ptr(),
        requiredLimits: limits,
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: string_view_init(),
        },
        deviceLostCallbackInfo: native::WGPUDeviceLostCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowSpontaneous,
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
    }
}

fn string_view_init() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

fn string_view_to_string(value: native::WGPUStringView) -> String {
    if value.data.is_null() {
        return String::new();
    }
    if value.length == native::WGPU_STRLEN {
        unsafe { CStr::from_ptr(value.data) }
            .to_string_lossy()
            .into_owned()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(value.data.cast::<u8>(), value.length) };
        String::from_utf8_lossy(bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_buffer(
        device: native::WGPUDevice,
        usage: native::WGPUBufferUsage,
    ) -> native::WGPUBuffer {
        let descriptor = native::WGPUBufferDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: string_view_init(),
            usage,
            size: 4,
            mappedAtCreation: 0,
        };
        unsafe { yawgpu::wgpuDeviceCreateBuffer(device, &descriptor) }
    }

    #[test]
    fn expect_no_validation_error_method_accepts_clean_action() {
        let test = ValidationTest::new();
        let mut buffer = std::ptr::null();

        test.expect_no_validation_error(|| {
            buffer = create_buffer(test.device(), native::WGPUBufferUsage_CopySrc);
        });

        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }

    #[test]
    #[should_panic(expected = "expected no device errors")]
    fn expect_no_validation_error_method_rejects_captured_error() {
        let test = ValidationTest::new();
        test.expect_no_validation_error(|| {
            let buffer = create_buffer(test.device(), native::WGPUBufferUsage_None);
            unsafe { yawgpu::wgpuBufferRelease(buffer) };
        });
    }

    #[test]
    fn expect_no_validation_error_free_function_uses_current_test() {
        let test = ValidationTest::new();
        let mut buffer = std::ptr::null();

        expect_no_validation_error(|| {
            buffer = create_buffer(test.device(), native::WGPUBufferUsage_CopySrc);
        });

        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }

    #[test]
    fn cartesian2_returns_owned_pairs_in_nested_loop_order() {
        assert_eq!(
            cartesian2(&[1, 2], &["a", "b"]),
            vec![(1, "a"), (1, "b"), (2, "a"), (2, "b")]
        );
    }

    #[test]
    fn cartesian3_returns_owned_triples_in_nested_loop_order() {
        assert_eq!(
            cartesian3(&[1], &[2, 3], &[4, 5]),
            vec![(1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5)]
        );
    }

    #[test]
    fn with_features_enables_requested_supported_feature() {
        let test = ValidationTest::with_features(&[native::WGPUFeatureName_TimestampQuery]);

        assert_eq!(
            unsafe {
                yawgpu::wgpuDeviceHasFeature(test.device(), native::WGPUFeatureName_TimestampQuery)
            },
            1
        );
    }

    #[test]
    fn with_limits_accepts_requested_supported_limits() {
        let baseline = ValidationTest::new();
        let mut limits = unsafe { std::mem::zeroed::<native::WGPULimits>() };
        assert_eq!(
            unsafe { yawgpu::wgpuAdapterGetLimits(baseline.adapter(), &mut limits) },
            native::WGPUStatus_Success
        );

        let test = ValidationTest::with_limits(limits);
        let mut actual = unsafe { std::mem::zeroed::<native::WGPULimits>() };
        assert_eq!(
            unsafe { yawgpu::wgpuDeviceGetLimits(test.device(), &mut actual) },
            native::WGPUStatus_Success
        );
        assert_eq!(actual.maxBindGroups, limits.maxBindGroups);
    }

    #[test]
    fn wait_for_future_completes_wait_any_only_request() {
        let instance = unsafe { yawgpu::wgpuCreateInstance(std::ptr::null()) };
        let mut adapter: native::WGPUAdapter = std::ptr::null();
        let callback_info = native::WGPURequestAdapterCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_WaitAnyOnly,
            callback: Some(request_adapter_callback),
            userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future = unsafe {
            yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info)
        };

        unsafe { wait_for_future(instance, future) };

        assert!(!adapter.is_null());
        unsafe {
            yawgpu::wgpuAdapterRelease(adapter);
            yawgpu::wgpuInstanceRelease(instance);
        }
    }
}
