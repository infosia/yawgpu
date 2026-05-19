use std::cell::RefCell;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_core::{DeviceError, ErrorKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RealBackend {
    Metal,
    Vulkan,
}

impl RealBackend {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        }
    }
}

#[must_use]
pub fn real_backend_available(_backend: RealBackend) -> bool {
    false
}

#[must_use]
pub fn real_backend_skip_reason(backend: RealBackend) -> Option<String> {
    (!real_backend_available(backend)).then(|| {
        format!(
            "{} backend is unavailable in the P7.0 gated harness",
            backend.name()
        )
    })
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

            let mut device: native::WGPUDevice = std::ptr::null();
            let device_callback_info = native::WGPURequestDeviceCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(request_device_callback),
                userdata1: (&mut device as *mut native::WGPUDevice).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let future =
                yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), device_callback_info);
            wait(instance, future);
            assert!(!device.is_null());

            let errors = Arc::new(Mutex::new(Vec::new()));
            let captured_errors = Arc::clone(&errors);
            yawgpu::testing_set_uncaptured_error_callback(
                device,
                Some(move |error| captured_errors.lock().expect("error lock").push(error)),
            );
            CURRENT_ERRORS.with(|current| *current.borrow_mut() = Some(Arc::clone(&errors)));

            Self {
                instance,
                adapter,
                device,
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

#[macro_export]
macro_rules! assert_device_error {
    ($expr:expr $(,)?) => {{
        $crate::assert_current_device_error_after(|| $expr, None);
    }};
    ($expr:expr, $substring:expr $(,)?) => {{
        $crate::assert_current_device_error_after(|| $expr, Some($substring));
    }};
}

/// Drives an instance until the provided future has had a chance to fire.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle.
pub unsafe fn wait(instance: native::WGPUInstance, future: native::WGPUFuture) {
    unsafe {
        yawgpu::wgpuInstanceProcessEvents(instance);
        let mut wait_info = native::WGPUFutureWaitInfo {
            future,
            completed: 0,
        };
        let _ = yawgpu::wgpuInstanceWaitAny(instance, 1, &mut wait_info, 0);
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
