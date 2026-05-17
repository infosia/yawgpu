use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[derive(Default)]
struct WorkDoneState {
    statuses: Vec<native::WGPUQueueWorkDoneStatus>,
}

#[derive(Default)]
struct MapState {
    statuses: Vec<native::WGPUMapAsyncStatus>,
}

#[test]
fn write_buffer_usage_alignment_and_bounds_validation() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let data = [0_u8; 16];

        let wrong_usage = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc, false);
        assert_device_error!({
            yawgpu::wgpuQueueWriteBuffer(queue, wrong_usage, 0, data.as_ptr().cast(), 4);
        });
        yawgpu::wgpuBufferRelease(wrong_usage);

        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, false);
        assert_device_error!({
            yawgpu::wgpuQueueWriteBuffer(queue, buffer, 0, data.as_ptr().cast(), 2);
        });
        assert_device_error!({
            yawgpu::wgpuQueueWriteBuffer(queue, buffer, 2, data.as_ptr().cast(), 4);
        });
        assert_device_error!({
            yawgpu::wgpuQueueWriteBuffer(queue, buffer, 12, data.as_ptr().cast(), 8);
        });
        assert_device_error!({
            yawgpu::wgpuQueueWriteBuffer(queue, buffer, u64::MAX - 3, data.as_ptr().cast(), 8);
        });

        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn write_buffer_rejects_mapped_pending_and_destroyed_buffers() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let data = [0_u8; 16];

        let mapped = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, true);
        assert_device_error!({
            yawgpu::wgpuQueueWriteBuffer(queue, mapped, 0, data.as_ptr().cast(), 4);
        });
        yawgpu::wgpuBufferRelease(mapped);

        let pending = create_buffer(
            test.device(),
            16,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
            false,
        );
        let mut map_state = MapState::default();
        map_async(
            pending,
            native::WGPUMapMode_Read,
            0,
            4,
            native::WGPUCallbackMode_WaitAnyOnly,
            &mut map_state,
        );
        assert_device_error!({
            yawgpu::wgpuQueueWriteBuffer(queue, pending, 0, data.as_ptr().cast(), 4);
        });
        yawgpu::wgpuBufferRelease(pending);

        let destroyed = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, false);
        yawgpu::wgpuBufferDestroy(destroyed);
        assert_device_error!({
            yawgpu::wgpuQueueWriteBuffer(queue, destroyed, 0, data.as_ptr().cast(), 4);
        });
        yawgpu::wgpuBufferRelease(destroyed);

        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn write_buffer_success_and_submit_argument_validation() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, false);
        let data = [1_u8, 2, 3, 4];

        test.clear_errors();
        yawgpu::wgpuQueueWriteBuffer(queue, buffer, 0, data.as_ptr().cast(), data.len());
        assert!(test.errors().is_empty());

        yawgpu::wgpuQueueSubmit(queue, 0, std::ptr::null());
        assert!(test.errors().is_empty());

        assert_device_error!({
            yawgpu::wgpuQueueSubmit(queue, 1, std::ptr::null());
        });

        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn on_submitted_work_done_before_submit_fires_success_on_process_events() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let mut state = WorkDoneState::default();
        on_submitted_work_done(
            queue,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );

        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(
            state.statuses,
            vec![native::WGPUQueueWorkDoneStatus_Success]
        );
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn on_submitted_work_done_wait_any_only_waits_for_wait_any() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let mut state = WorkDoneState::default();
        let future =
            on_submitted_work_done(queue, native::WGPUCallbackMode_WaitAnyOnly, &mut state);

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
        assert_eq!(
            state.statuses,
            vec![native::WGPUQueueWorkDoneStatus_Success]
        );
        yawgpu::wgpuQueueRelease(queue);
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

unsafe fn on_submitted_work_done(
    queue: native::WGPUQueue,
    callback_mode: native::WGPUCallbackMode,
    state: &mut WorkDoneState,
) -> native::WGPUFuture {
    let callback_info = native::WGPUQueueWorkDoneCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: callback_mode,
        callback: Some(work_done_callback),
        userdata1: (state as *mut WorkDoneState).cast(),
        userdata2: std::ptr::null_mut(),
    };
    yawgpu::wgpuQueueOnSubmittedWorkDone(queue, callback_info)
}

unsafe fn map_async(
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
    callback_mode: native::WGPUCallbackMode,
    state: &mut MapState,
) -> native::WGPUFuture {
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: callback_mode,
        callback: Some(map_callback),
        userdata1: (state as *mut MapState).cast(),
        userdata2: std::ptr::null_mut(),
    };
    yawgpu::wgpuBufferMapAsync(buffer, mode, offset, size, callback_info)
}

unsafe extern "C" fn work_done_callback(
    status: native::WGPUQueueWorkDoneStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    (*(userdata1 as *mut WorkDoneState)).statuses.push(status);
}

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    (*(userdata1 as *mut MapState)).statuses.push(status);
}
