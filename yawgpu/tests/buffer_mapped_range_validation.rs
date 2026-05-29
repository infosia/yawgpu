use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::ValidationTest;

#[derive(Default)]
struct MapState {
    statuses: Vec<native::WGPUMapAsyncStatus>,
}

#[test]
fn unmapped_buffers_return_null() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapWrite, false);
        assert_mapped_ranges_null(buffer, 0, 4);
        yawgpu::wgpuBufferRelease(buffer);

        let mapped = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc, true);
        assert!(!yawgpu::wgpuBufferGetMappedRange(mapped, 0, 4).is_null());
        yawgpu::wgpuBufferUnmap(mapped);
        assert_mapped_ranges_null(mapped, 0, 4);
        yawgpu::wgpuBufferRelease(mapped);

        assert!(test.errors().is_empty());
    }
}

#[test]
fn const_and_non_const_access_follow_map_mode() {
    let test = ValidationTest::new();
    unsafe {
        let read = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapRead, false);
        map_and_wait(test.instance(), read, native::WGPUMapMode_Read, 0, 4);
        assert!(yawgpu::wgpuBufferGetMappedRange(read, 0, 4).is_null());
        assert!(!yawgpu::wgpuBufferGetConstMappedRange(read, 0, 4).is_null());
        yawgpu::wgpuBufferRelease(read);

        let write = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapWrite, false);
        map_and_wait(test.instance(), write, native::WGPUMapMode_Write, 0, 4);
        assert!(!yawgpu::wgpuBufferGetMappedRange(write, 0, 4).is_null());
        yawgpu::wgpuBufferRelease(write);

        let write = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapWrite, false);
        map_and_wait(test.instance(), write, native::WGPUMapMode_Write, 0, 4);
        assert!(!yawgpu::wgpuBufferGetConstMappedRange(write, 0, 4).is_null());
        yawgpu::wgpuBufferRelease(write);

        let mapped = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc, true);
        assert!(!yawgpu::wgpuBufferGetMappedRange(mapped, 0, 4).is_null());
        yawgpu::wgpuBufferRelease(mapped);

        assert!(test.errors().is_empty());
    }
}

#[test]
fn offset_beyond_mapped_end_returns_null() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 64, native::WGPUBufferUsage_MapWrite, false);
        map_and_wait(test.instance(), buffer, native::WGPUMapMode_Write, 16, 16);

        assert!(yawgpu::wgpuBufferGetMappedRange(buffer, 33, 0).is_null());
        assert!(yawgpu::wgpuBufferGetMappedRange(buffer, 40, 4).is_null());
        yawgpu::wgpuBufferRelease(buffer);

        assert!(test.errors().is_empty());
    }
}

#[test]
fn offset_plus_size_beyond_mapped_end_returns_null() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 64, native::WGPUBufferUsage_MapWrite, false);
        map_and_wait(test.instance(), buffer, native::WGPUMapMode_Write, 16, 16);

        assert!(yawgpu::wgpuBufferGetMappedRange(buffer, 28, 8).is_null());
        assert!(yawgpu::wgpuBufferGetMappedRange(buffer, usize::MAX - 3, 8).is_null());
        yawgpu::wgpuBufferRelease(buffer);

        assert!(test.errors().is_empty());
    }
}

#[test]
fn whole_map_size_uses_buffer_size_then_validates_against_active_mapped_range() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 64, native::WGPUBufferUsage_MapWrite, false);
        map_and_wait(test.instance(), buffer, native::WGPUMapMode_Write, 16, 16);

        assert!(
            yawgpu::wgpuBufferGetMappedRange(buffer, 16, native::WGPU_WHOLE_MAP_SIZE).is_null()
        );
        assert!(
            yawgpu::wgpuBufferGetMappedRange(buffer, 20, native::WGPU_WHOLE_MAP_SIZE).is_null()
        );
        assert!(yawgpu::wgpuBufferGetMappedRange(buffer, 8, native::WGPU_WHOLE_MAP_SIZE).is_null());
        assert!(!yawgpu::wgpuBufferGetMappedRange(buffer, 16, 16).is_null());
        yawgpu::wgpuBufferRelease(buffer);

        assert!(test.errors().is_empty());
    }
}

#[test]
fn offset_before_mapped_start_returns_null() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 64, native::WGPUBufferUsage_MapWrite, false);
        map_and_wait(test.instance(), buffer, native::WGPUMapMode_Write, 16, 16);

        for size in [0, 4, native::WGPU_WHOLE_MAP_SIZE] {
            assert!(yawgpu::wgpuBufferGetMappedRange(buffer, 8, size).is_null());
        }
        yawgpu::wgpuBufferRelease(buffer);

        assert!(test.errors().is_empty());
    }
}

#[test]
fn destroyed_mapped_buffer_returns_null() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapWrite, false);
        map_and_wait(test.instance(), buffer, native::WGPUMapMode_Write, 0, 4);
        yawgpu::wgpuBufferDestroy(buffer);

        assert_mapped_ranges_null(buffer, 0, 4);
        yawgpu::wgpuBufferRelease(buffer);

        assert!(test.errors().is_empty());
    }
}

#[test]
fn write_mapped_range_points_into_stable_host_backing() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), 16, native::WGPUBufferUsage_MapWrite, false);
        map_and_wait(test.instance(), buffer, native::WGPUMapMode_Write, 8, 4);

        let ptr = yawgpu::wgpuBufferGetMappedRange(buffer, 8, 4).cast::<u8>();
        assert!(!ptr.is_null());
        ptr.write(11);
        ptr.add(1).write(22);
        ptr.add(2).write(33);
        ptr.add(3).write(44);

        assert_eq!(
            std::slice::from_raw_parts(ptr.cast_const(), 4),
            &[11, 22, 33, 44]
        );
        yawgpu::wgpuBufferRelease(buffer);

        assert!(test.errors().is_empty());
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

unsafe fn map_and_wait(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
) {
    let mut state = MapState::default();
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut state as *mut MapState).cast(),
        userdata2: std::ptr::null_mut(),
    };
    yawgpu::wgpuBufferMapAsync(buffer, mode, offset, size, callback_info);
    yawgpu::wgpuInstanceProcessEvents(instance);
    assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Success]);
}

unsafe fn assert_mapped_ranges_null(buffer: native::WGPUBuffer, offset: usize, size: usize) {
    assert!(yawgpu::wgpuBufferGetMappedRange(buffer, offset, size).is_null());
    assert!(yawgpu::wgpuBufferGetConstMappedRange(buffer, offset, size).is_null());
}

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    (*(userdata1 as *mut MapState)).statuses.push(status);
}
