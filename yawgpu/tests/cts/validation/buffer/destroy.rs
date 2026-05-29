//! Ports `$CTS/src/webgpu/api/validation/buffer/destroy.spec.ts`.

use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{assert_device_error, cartesian2, ValidationTest};

const BUFFER_USAGES: &[native::WGPUBufferUsage] = &[
    native::WGPUBufferUsage_MapRead,
    native::WGPUBufferUsage_MapWrite,
    native::WGPUBufferUsage_CopySrc,
    native::WGPUBufferUsage_CopyDst,
    native::WGPUBufferUsage_Index,
    native::WGPUBufferUsage_Vertex,
    native::WGPUBufferUsage_Uniform,
    native::WGPUBufferUsage_Storage,
    native::WGPUBufferUsage_Indirect,
    native::WGPUBufferUsage_QueryResolve,
];

#[derive(Default)]
struct MapState {
    statuses: Vec<native::WGPUMapAsyncStatus>,
}

#[test]
fn all_usages() {
    let test = ValidationTest::new();

    for &usage in BUFFER_USAGES {
        let buffer = create_buffer(test.device(), 4, usage, false);
        test.expect_no_validation_error(|| unsafe {
            yawgpu::wgpuBufferDestroy(buffer);
        });
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn error_buffer() {
    let test = ValidationTest::new();
    let mut buffer = std::ptr::null();

    assert_device_error!({
        buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_None, false);
    });
    test.expect_no_validation_error(|| unsafe {
        yawgpu::wgpuBufferDestroy(buffer);
    });
    unsafe { yawgpu::wgpuBufferRelease(buffer) };
}

#[test]
fn twice() {
    let test = ValidationTest::new();
    let descriptors = [
        (4, native::WGPUBufferUsage_CopySrc),
        (
            4,
            native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_CopySrc,
        ),
        (
            4,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        ),
    ];

    for (mapped_at_creation, (size, usage)) in cartesian2(&[false, true], &descriptors) {
        let buffer = create_buffer(test.device(), size, usage, mapped_at_creation);
        test.expect_no_validation_error(|| unsafe {
            yawgpu::wgpuBufferDestroy(buffer);
            yawgpu::wgpuBufferDestroy(buffer);
        });
        unsafe { yawgpu::wgpuBufferRelease(buffer) };
    }
}

#[test]
fn while_mapped() {
    let test = ValidationTest::new();
    let descriptors = [
        (native::WGPUBufferUsage_CopySrc, None),
        (
            native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_CopySrc,
            None,
        ),
        (
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
            None,
        ),
        (
            native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_CopySrc,
            Some(native::WGPUMapMode_Write),
        ),
        (
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
            Some(native::WGPUMapMode_Read),
        ),
    ];

    for (mapped_at_creation, unmap_before_destroy) in cartesian2(&[false, true], &[false, true]) {
        for &(usage, map_mode) in &descriptors {
            if !mapped_at_creation && map_mode.is_none() {
                continue;
            }

            let buffer = create_buffer(test.device(), 4, usage, mapped_at_creation);
            if let Some(map_mode) = map_mode {
                if mapped_at_creation {
                    unsafe { yawgpu::wgpuBufferUnmap(buffer) };
                }
                assert_map_success(&test, buffer, map_mode);
            }
            if unmap_before_destroy {
                unsafe { yawgpu::wgpuBufferUnmap(buffer) };
            }

            test.expect_no_validation_error(|| unsafe {
                yawgpu::wgpuBufferDestroy(buffer);
            });
            unsafe { yawgpu::wgpuBufferRelease(buffer) };
        }
    }
}

fn create_buffer(
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
    unsafe { yawgpu::wgpuDeviceCreateBuffer(device, &descriptor) }
}

fn assert_map_success(
    test: &ValidationTest,
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
) {
    let mut state = MapState::default();
    let future = map_async(
        test,
        buffer,
        mode,
        0,
        native::WGPU_WHOLE_MAP_SIZE,
        &mut state,
    );
    unsafe { yawgpu_test::wait_for_future(test.instance(), future) };
    assert_eq!(state.statuses, vec![native::WGPUMapAsyncStatus_Success]);
}

fn map_async(
    _test: &ValidationTest,
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
    state: &mut MapState,
) -> native::WGPUFuture {
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_WaitAnyOnly,
        callback: Some(map_callback),
        userdata1: (state as *mut MapState).cast(),
        userdata2: std::ptr::null_mut(),
    };
    unsafe { yawgpu::wgpuBufferMapAsync(buffer, mode, offset, size, callback_info) }
}

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    (*(userdata1 as *mut MapState)).statuses.push(status);
}
