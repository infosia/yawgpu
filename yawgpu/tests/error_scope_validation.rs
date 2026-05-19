use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_core::ErrorKind;
use yawgpu_test::{assert_device_error, wait, ValidationTest};

#[derive(Default)]
struct PopState {
    calls: Vec<PopCall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PopCall {
    status: native::WGPUPopErrorScopeStatus,
    error_type: native::WGPUErrorType,
    message: String,
}

#[test]
fn success_returns_no_error() {
    let test = ValidationTest::new();
    unsafe {
        let mut state = PopState::default();
        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_Validation);
        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        wait(test.instance(), future);

        assert_eq!(
            state.calls,
            vec![PopCall {
                status: native::WGPUPopErrorScopeStatus_Success,
                error_type: native::WGPUErrorType_NoError,
                message: String::new(),
            }]
        );
    }
}

#[test]
fn validation_scope_catches_validation_error() {
    let test = ValidationTest::new();
    unsafe {
        let mut state = PopState::default();
        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_Validation);
        let buffer = create_invalid_buffer(test.device());
        yawgpu::wgpuBufferRelease(buffer);
        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        wait(test.instance(), future);

        assert_eq!(state.calls.len(), 1);
        assert_eq!(
            state.calls[0].status,
            native::WGPUPopErrorScopeStatus_Success
        );
        assert_eq!(state.calls[0].error_type, native::WGPUErrorType_Validation);
        assert!(test.errors().is_empty());
    }
}

#[test]
fn all_filters_map_to_matching_error_types() {
    let test = ValidationTest::new();
    unsafe {
        assert_matching_filter(
            &test,
            native::WGPUErrorFilter_Validation,
            ErrorKind::Validation,
            native::WGPUErrorType_Validation,
        );
        assert_matching_filter(
            &test,
            native::WGPUErrorFilter_OutOfMemory,
            ErrorKind::OutOfMemory,
            native::WGPUErrorType_OutOfMemory,
        );
        assert_matching_filter(
            &test,
            native::WGPUErrorFilter_Internal,
            ErrorKind::Internal,
            native::WGPUErrorType_Internal,
        );
    }
}

#[test]
fn pop_without_open_scope_reports_error_status() {
    let test = ValidationTest::new();
    unsafe {
        let mut state = PopState::default();
        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        wait(test.instance(), future);

        assert_eq!(state.calls.len(), 1);
        assert_eq!(state.calls[0].status, native::WGPUPopErrorScopeStatus_Error);
        assert_eq!(state.calls[0].error_type, native::WGPUErrorType_NoError);
        assert!(!state.calls[0].message.is_empty());
    }
}

#[test]
fn nested_non_matching_scope_bubbles_to_outer_matching_scope() {
    let test = ValidationTest::new();
    unsafe {
        let mut inner = PopState::default();
        let mut outer = PopState::default();
        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_Validation);
        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_OutOfMemory);

        let buffer = create_invalid_buffer(test.device());
        yawgpu::wgpuBufferRelease(buffer);

        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut inner,
        );
        wait(test.instance(), future);
        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut outer,
        );
        wait(test.instance(), future);

        assert_eq!(inner.calls[0].error_type, native::WGPUErrorType_NoError);
        assert_eq!(outer.calls[0].error_type, native::WGPUErrorType_Validation);
        assert!(test.errors().is_empty());
    }
}

#[test]
fn nested_inner_matching_scope_stops_bubbling() {
    let test = ValidationTest::new();
    unsafe {
        let mut inner = PopState::default();
        let mut outer = PopState::default();
        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_OutOfMemory);
        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_Validation);

        let buffer = create_invalid_buffer(test.device());
        yawgpu::wgpuBufferRelease(buffer);

        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut inner,
        );
        wait(test.instance(), future);
        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut outer,
        );
        wait(test.instance(), future);

        assert_eq!(inner.calls[0].error_type, native::WGPUErrorType_Validation);
        assert_eq!(outer.calls[0].error_type, native::WGPUErrorType_NoError);
        assert!(test.errors().is_empty());
    }
}

#[test]
fn caught_error_does_not_fire_uncaptured_callback_but_unmatched_error_does() {
    let test = ValidationTest::new();
    unsafe {
        let mut state = PopState::default();
        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_Validation);
        let buffer = create_invalid_buffer(test.device());
        yawgpu::wgpuBufferRelease(buffer);
        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        wait(test.instance(), future);
        assert!(test.errors().is_empty());

        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_OutOfMemory);
        assert_device_error!({
            let buffer = create_invalid_buffer(test.device());
            yawgpu::wgpuBufferRelease(buffer);
        });
        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        wait(test.instance(), future);
        assert_eq!(state.calls[1].error_type, native::WGPUErrorType_NoError);
    }
}

#[test]
fn matching_scope_keeps_first_error() {
    let test = ValidationTest::new();
    unsafe {
        let mut state = PopState::default();
        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_Validation);
        yawgpu::testing_dispatch_device_error(test.device(), ErrorKind::Validation, "first");
        yawgpu::testing_dispatch_device_error(test.device(), ErrorKind::Validation, "second");
        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        wait(test.instance(), future);

        assert_eq!(state.calls[0].error_type, native::WGPUErrorType_Validation);
        assert_eq!(state.calls[0].message, "first");
    }
}

#[test]
fn pop_error_scope_wait_any_only_is_async_and_fires_once() {
    let test = ValidationTest::new();
    unsafe {
        let mut state = PopState::default();
        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_Validation);
        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_WaitAnyOnly,
            &mut state,
        );

        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert!(state.calls.is_empty());

        let mut wait_info = native::WGPUFutureWaitInfo {
            future,
            completed: 0,
        };
        assert_eq!(
            yawgpu::wgpuInstanceWaitAny(test.instance(), 1, &mut wait_info, 0),
            native::WGPUWaitStatus_Success
        );
        assert_eq!(wait_info.completed, 1);
        assert_eq!(state.calls.len(), 1);

        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.calls.len(), 1);
    }
}

#[test]
fn destroyed_device_pop_resolves_success_no_error() {
    let test = ValidationTest::new();
    unsafe {
        let mut state = PopState::default();
        yawgpu::wgpuDevicePushErrorScope(test.device(), native::WGPUErrorFilter_Validation);
        yawgpu::wgpuDeviceDestroy(test.device());
        let future = pop_error_scope(
            test.device(),
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        wait(test.instance(), future);

        assert_eq!(
            state.calls,
            vec![PopCall {
                status: native::WGPUPopErrorScopeStatus_Success,
                error_type: native::WGPUErrorType_NoError,
                message: String::new(),
            }]
        );
    }
}

unsafe fn assert_matching_filter(
    test: &ValidationTest,
    filter: native::WGPUErrorFilter,
    kind: ErrorKind,
    expected_type: native::WGPUErrorType,
) {
    let mut state = PopState::default();
    yawgpu::wgpuDevicePushErrorScope(test.device(), filter);
    yawgpu::testing_dispatch_device_error(test.device(), kind, "matched");
    let future = pop_error_scope(
        test.device(),
        native::WGPUCallbackMode_AllowProcessEvents,
        &mut state,
    );
    wait(test.instance(), future);
    assert_eq!(
        state.calls[0].status,
        native::WGPUPopErrorScopeStatus_Success
    );
    assert_eq!(state.calls[0].error_type, expected_type);
    assert_eq!(state.calls[0].message, "matched");
}

unsafe fn pop_error_scope(
    device: native::WGPUDevice,
    mode: native::WGPUCallbackMode,
    state: &mut PopState,
) -> native::WGPUFuture {
    let callback_info = native::WGPUPopErrorScopeCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode,
        callback: Some(pop_error_scope_callback),
        userdata1: (state as *mut PopState).cast(),
        userdata2: std::ptr::null_mut(),
    };
    yawgpu::wgpuDevicePopErrorScope(device, callback_info)
}

unsafe fn create_invalid_buffer(device: native::WGPUDevice) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: u64::MAX,
        size: 4,
        mappedAtCreation: 0,
    };
    yawgpu::wgpuDeviceCreateBuffer(device, &descriptor)
}

unsafe extern "C" fn pop_error_scope_callback(
    status: native::WGPUPopErrorScopeStatus,
    error_type: native::WGPUErrorType,
    message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let state = &mut *(userdata1 as *mut PopState);
    state.calls.push(PopCall {
        status,
        error_type,
        message: string_view_to_string(message),
    });
}

unsafe fn string_view_to_string(value: native::WGPUStringView) -> String {
    if value.data.is_null() || value.length == 0 {
        return String::new();
    }
    let bytes = std::slice::from_raw_parts(value.data.cast::<u8>(), value.length);
    String::from_utf8_lossy(bytes).into_owned()
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
