use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[derive(Default)]
struct MapState {
    statuses: Vec<native::WGPUMapAsyncStatus>,
}

struct RetryState {
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    statuses: Vec<native::WGPUMapAsyncStatus>,
}

struct UnmapState {
    buffer: native::WGPUBuffer,
    statuses: Vec<native::WGPUMapAsyncStatus>,
}

#[test]
fn wrong_usage_and_mode_vs_usage_callbacks_error() {
    let test = ValidationTest::new();
    unsafe {
        let copy_src = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc, false);
        let mut state = MapState::default();
        assert_device_error!({
            map_async(
                copy_src,
                native::WGPUMapMode_Read,
                0,
                4,
                native::WGPUCallbackMode_AllowProcessEvents,
                &mut state,
            );
        });
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);
        yawgpu::wgpuBufferRelease(copy_src);

        let map_read = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
        let mut state = MapState::default();
        assert_device_error!({
            map_async(
                map_read,
                native::WGPUMapMode_Write,
                0,
                4,
                native::WGPUCallbackMode_AllowProcessEvents,
                &mut state,
            );
        });
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);
        yawgpu::wgpuBufferRelease(map_read);
    }
}

#[test]
fn offset_and_size_alignment_validation() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);

        let mut state = MapState::default();
        assert_device_error!({
            map_async(
                buffer,
                native::WGPUMapMode_Read,
                4,
                4,
                native::WGPUCallbackMode_AllowProcessEvents,
                &mut state,
            );
        });
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);

        let mut state = MapState::default();
        assert_device_error!({
            map_async(
                buffer,
                native::WGPUMapMode_Read,
                0,
                2,
                native::WGPUCallbackMode_AllowProcessEvents,
                &mut state,
            );
        });
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);

        let mut state = MapState::default();
        map_async(
            buffer,
            native::WGPUMapMode_Read,
            8,
            4,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Success]);
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn offset_size_oob_and_overflow_callbacks_error() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);

        let mut state = MapState::default();
        assert_device_error!({
            map_async(
                buffer,
                native::WGPUMapMode_Read,
                16,
                4,
                native::WGPUCallbackMode_AllowProcessEvents,
                &mut state,
            );
        });
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);

        let mut state = MapState::default();
        assert_device_error!({
            map_async(
                buffer,
                native::WGPUMapMode_Read,
                usize::MAX - 7,
                8,
                native::WGPUCallbackMode_AllowProcessEvents,
                &mut state,
            );
        });
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn already_mapped_wrong_mode_unsupported_mode_destroyed_and_pending_error() {
    let test = ValidationTest::new();
    unsafe {
        let mapped = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc, true);
        let mut state = MapState::default();
        assert_device_error!({
            map_async(
                mapped,
                native::WGPUMapMode_Read,
                0,
                4,
                native::WGPUCallbackMode_AllowProcessEvents,
                &mut state,
            );
        });
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);
        yawgpu::wgpuBufferRelease(mapped);

        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
        for mode in [
            native::WGPUMapMode_None,
            native::WGPUMapMode_Read | native::WGPUMapMode_Write,
            native::WGPUMapMode_Read | 4,
        ] {
            let mut state = MapState::default();
            assert_device_error!({
                map_async(
                    buffer,
                    mode,
                    0,
                    4,
                    native::WGPUCallbackMode_AllowProcessEvents,
                    &mut state,
                );
            });
            yawgpu::wgpuInstanceProcessEvents(test.instance());
            assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);
        }

        yawgpu::wgpuBufferDestroy(buffer);
        let mut state = MapState::default();
        assert_device_error!({
            map_async(
                buffer,
                native::WGPUMapMode_Read,
                0,
                4,
                native::WGPUCallbackMode_AllowProcessEvents,
                &mut state,
            );
        });
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);
        yawgpu::wgpuBufferUnmap(buffer);
        let mut state = MapState::default();
        assert_device_error!({
            map_async(
                buffer,
                native::WGPUMapMode_Read,
                0,
                4,
                native::WGPUCallbackMode_AllowProcessEvents,
                &mut state,
            );
        });
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);
        yawgpu::wgpuBufferRelease(buffer);

        let pending = create_buffer(test.device(), 32, native::WGPUBufferUsage_MapRead, false);
        let mut first = MapState::default();
        map_async(
            pending,
            native::WGPUMapMode_Read,
            0,
            4,
            native::WGPUCallbackMode_WaitAnyOnly,
            &mut first,
        );
        assert_eq!(
            yawgpu::wgpuBufferGetMapState(pending),
            native::WGPUBufferMapState_Pending
        );
        for (offset, size) in [(0, 4), (8, 4)] {
            let mut state = MapState::default();
            assert_device_error!({
                map_async(
                    pending,
                    native::WGPUMapMode_Read,
                    offset,
                    size,
                    native::WGPUCallbackMode_AllowProcessEvents,
                    &mut state,
                );
            });
            yawgpu::wgpuInstanceProcessEvents(test.instance());
            assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Error]);
        }
        yawgpu::wgpuBufferRelease(pending);
    }
}

#[test]
fn valid_map_transitions_pending_then_mapped_success() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
        let mut state = MapState::default();
        map_async(
            buffer,
            native::WGPUMapMode_Read,
            0,
            native::WGPU_WHOLE_MAP_SIZE as usize,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        assert_eq!(
            yawgpu::wgpuBufferGetMapState(buffer),
            native::WGPUBufferMapState_Pending
        );
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Success]);
        assert_eq!(
            yawgpu::wgpuBufferGetMapState(buffer),
            native::WGPUBufferMapState_Mapped
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn unmap_destroy_and_release_before_drain_abort_once() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
        let mut state = MapState::default();
        map_async(
            buffer,
            native::WGPUMapMode_Read,
            0,
            4,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        yawgpu::wgpuBufferUnmap(buffer);
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Aborted]);
        yawgpu::wgpuBufferRelease(buffer);

        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
        let mut state = MapState::default();
        map_async(
            buffer,
            native::WGPUMapMode_Read,
            0,
            4,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        yawgpu::wgpuBufferDestroy(buffer);
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Aborted]);
        yawgpu::wgpuBufferRelease(buffer);

        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
        let mut state = MapState::default();
        map_async(
            buffer,
            native::WGPUMapMode_Read,
            0,
            4,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Aborted]);
    }
}

#[test]
fn error_callback_can_retry_valid_map_to_success() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
        let mut retry = RetryState {
            instance: test.instance(),
            buffer,
            statuses: Vec::new(),
        };
        assert_device_error!({
            map_async_with_callback(
                buffer,
                native::WGPUMapMode_Read,
                4,
                4,
                native::WGPUCallbackMode_AllowProcessEvents,
                Some(retry_callback),
                (&mut retry as *mut RetryState).cast(),
            );
        });
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(
            retry.statuses,
            vec![
                native::WGPUMapAsyncStatus_Error,
                native::WGPUMapAsyncStatus_Success,
            ]
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn unmap_inside_success_callback_fires_once() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
        let mut state = UnmapState {
            buffer,
            statuses: Vec::new(),
        };
        map_async_with_callback(
            buffer,
            native::WGPUMapMode_Read,
            0,
            4,
            native::WGPUCallbackMode_AllowProcessEvents,
            Some(unmap_callback),
            (&mut state as *mut UnmapState).cast(),
        );
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Success]);
        assert_eq!(
            yawgpu::wgpuBufferGetMapState(buffer),
            native::WGPUBufferMapState_Unmapped
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn wait_any_only_map_callback_waits_for_wait_any() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
        let mut state = MapState::default();
        let future = map_async(
            buffer,
            native::WGPUMapMode_Read,
            0,
            4,
            native::WGPUCallbackMode_WaitAnyOnly,
            &mut state,
        );

        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert!(state.statuses.is_empty());

        let mut wait_info = native::WGPUFutureWaitInfo {
            future,
            completed: 0,
        };
        assert_eq!(
            yawgpu::wgpuInstanceWaitAny(test.instance(), 1, &mut wait_info, 0),
            native::WGPUWaitStatus_Success
        );
        assert_eq!(wait_info.completed, 1);
        assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Success]);
        yawgpu::wgpuBufferRelease(buffer);
    }
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
    mapped_at_creation: bool,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
        usage,
        size,
        mappedAtCreation: u32::from(mapped_at_creation),
    };
    yawgpu::wgpuDeviceCreateBuffer(device, &descriptor)
}

unsafe fn map_async(
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
    callback_mode: native::WGPUCallbackMode,
    state: &mut MapState,
) -> native::WGPUFuture {
    map_async_with_callback(
        buffer,
        mode,
        offset,
        size,
        callback_mode,
        Some(map_callback),
        (state as *mut MapState).cast(),
    )
}

unsafe fn map_async_with_callback(
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
    callback_mode: native::WGPUCallbackMode,
    callback: native::WGPUBufferMapCallback,
    userdata1: *mut c_void,
) -> native::WGPUFuture {
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: callback_mode,
        callback,
        userdata1,
        userdata2: std::ptr::null_mut(),
    };
    yawgpu::wgpuBufferMapAsync(buffer, mode, offset, size, callback_info)
}

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    (*(userdata1 as *mut MapState)).statuses.push(status);
}

unsafe extern "C" fn retry_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let state = &mut *(userdata1 as *mut RetryState);
    state.statuses.push(status);
    if status == native::WGPUMapAsyncStatus_Error {
        map_async_with_callback(
            state.buffer,
            native::WGPUMapMode_Read,
            0,
            4,
            native::WGPUCallbackMode_AllowProcessEvents,
            Some(retry_callback),
            userdata1,
        );
        yawgpu::wgpuInstanceProcessEvents(state.instance);
    }
}

unsafe extern "C" fn unmap_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let state = &mut *(userdata1 as *mut UnmapState);
    state.statuses.push(status);
    yawgpu::wgpuBufferUnmap(state.buffer);
}
