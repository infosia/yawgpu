use std::os::raw::c_void;

use yawgpu::native;

#[derive(Default)]
struct AdapterCallbackState {
    fired: u32,
    adapter: native::WGPUAdapter,
}

unsafe extern "C" fn request_adapter_callback(
    status: native::WGPURequestAdapterStatus,
    adapter: native::WGPUAdapter,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestAdapterStatus_Success);
    let state = &mut *(userdata1 as *mut AdapterCallbackState);
    state.fired += 1;
    state.adapter = adapter;
}

fn request_adapter(
    instance: native::WGPUInstance,
    mode: native::WGPUCallbackMode,
    state: &mut AdapterCallbackState,
) -> native::WGPUFuture {
    unsafe {
        let callback_info = native::WGPURequestAdapterCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode,
            callback: Some(request_adapter_callback),
            userdata1: (state as *mut AdapterCallbackState).cast(),
            userdata2: std::ptr::null_mut(),
        };
        yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info)
    }
}

fn wait_info(future: native::WGPUFuture) -> native::WGPUFutureWaitInfo {
    native::WGPUFutureWaitInfo {
        future,
        completed: 0,
    }
}

#[test]
fn wait_any_only_callback_fires_only_in_wait_any() {
    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        let mut state = AdapterCallbackState::default();
        let future = request_adapter(instance, native::WGPUCallbackMode_WaitAnyOnly, &mut state);

        yawgpu::wgpuInstanceProcessEvents(instance);
        assert_eq!(state.fired, 0);
        assert!(state.adapter.is_null());

        let mut info = wait_info(future);
        let status = yawgpu::wgpuInstanceWaitAny(instance, 1, &mut info, 0);
        assert_eq!(status, native::WGPUWaitStatus_Success);
        assert_eq!(info.completed, 1);
        assert_eq!(state.fired, 1);
        assert!(!state.adapter.is_null());

        yawgpu::wgpuAdapterRelease(state.adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
fn allow_process_events_callback_fires_in_process_events() {
    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        let mut state = AdapterCallbackState::default();
        let _future = request_adapter(
            instance,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );

        yawgpu::wgpuInstanceProcessEvents(instance);
        assert_eq!(state.fired, 1);
        assert!(!state.adapter.is_null());

        yawgpu::wgpuAdapterRelease(state.adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
fn repeated_wait_any_on_completed_future_keeps_reporting_success() {
    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        let mut state = AdapterCallbackState::default();
        let future = request_adapter(instance, native::WGPUCallbackMode_WaitAnyOnly, &mut state);

        let mut first = wait_info(future);
        let first_status = yawgpu::wgpuInstanceWaitAny(instance, 1, &mut first, 0);
        assert_eq!(first_status, native::WGPUWaitStatus_Success);
        assert_eq!(first.completed, 1);
        assert_eq!(state.fired, 1);
        assert!(!state.adapter.is_null());

        let mut second = wait_info(future);
        let second_status = yawgpu::wgpuInstanceWaitAny(instance, 1, &mut second, 0);
        assert_eq!(second_status, native::WGPUWaitStatus_Success);
        assert_eq!(second.completed, 1);
        assert_eq!(state.fired, 1);

        yawgpu::wgpuAdapterRelease(state.adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
fn wait_any_poll_zero_count_and_null_futures_statuses() {
    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        let mut state = AdapterCallbackState::default();
        let future = request_adapter(instance, native::WGPUCallbackMode_WaitAnyOnly, &mut state);

        let zero_count_status = yawgpu::wgpuInstanceWaitAny(instance, 0, std::ptr::null_mut(), 0);
        assert_eq!(zero_count_status, native::WGPUWaitStatus_TimedOut);

        let null_futures_status = yawgpu::wgpuInstanceWaitAny(instance, 1, std::ptr::null_mut(), 0);
        assert_eq!(null_futures_status, native::WGPUWaitStatus_Error);

        let mut info = wait_info(future);
        let poll_status = yawgpu::wgpuInstanceWaitAny(instance, 1, &mut info, 0);
        assert_eq!(poll_status, native::WGPUWaitStatus_Success);
        assert_eq!(info.completed, 1);

        yawgpu::wgpuAdapterRelease(state.adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
fn timed_wait_any_requires_instance_feature() {
    unsafe {
        let descriptor_without_timed_wait_any = native::WGPUInstanceDescriptor {
            nextInChain: std::ptr::null_mut(),
            requiredFeatureCount: 0,
            requiredFeatures: std::ptr::null(),
            requiredLimits: std::ptr::null(),
        };
        let instance = yawgpu::wgpuCreateInstance(&descriptor_without_timed_wait_any);
        let mut state = AdapterCallbackState::default();
        let future = request_adapter(instance, native::WGPUCallbackMode_WaitAnyOnly, &mut state);
        let mut info = wait_info(future);

        let status = yawgpu::wgpuInstanceWaitAny(instance, 1, &mut info, 1);
        assert_eq!(status, native::WGPUWaitStatus_Error);
        assert_eq!(state.fired, 0);

        let poll_status = yawgpu::wgpuInstanceWaitAny(instance, 1, &mut info, 0);
        assert_eq!(poll_status, native::WGPUWaitStatus_Success);
        yawgpu::wgpuAdapterRelease(state.adapter);
        yawgpu::wgpuInstanceRelease(instance);

        let features = [native::WGPUInstanceFeatureName_TimedWaitAny];
        let descriptor_with_timed_wait_any = native::WGPUInstanceDescriptor {
            nextInChain: std::ptr::null_mut(),
            requiredFeatureCount: features.len(),
            requiredFeatures: features.as_ptr(),
            requiredLimits: std::ptr::null(),
        };
        let instance = yawgpu::wgpuCreateInstance(&descriptor_with_timed_wait_any);
        let mut state = AdapterCallbackState::default();
        let future = request_adapter(instance, native::WGPUCallbackMode_WaitAnyOnly, &mut state);
        let mut info = wait_info(future);

        let status = yawgpu::wgpuInstanceWaitAny(instance, 1, &mut info, 1);
        assert_eq!(status, native::WGPUWaitStatus_Success);
        assert_eq!(info.completed, 1);
        assert_eq!(state.fired, 1);

        yawgpu::wgpuAdapterRelease(state.adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}
