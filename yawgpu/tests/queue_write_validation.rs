use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{wait, ValidationTest};

#[test]
fn write_buffer_then_map_async_without_submit_reads_written_bytes() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let descriptor = native::WGPUBufferDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: native::WGPUStringView {
                data: std::ptr::null(),
                length: 0,
            },
            usage: native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
            size: 8,
            mappedAtCreation: 0,
        };
        let buffer = yawgpu::wgpuDeviceCreateBuffer(test.device(), &descriptor);
        assert!(!buffer.is_null());

        let expected = [0x10_u8, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80];
        yawgpu::wgpuQueueWriteBuffer(
            queue,
            buffer,
            0,
            expected.as_ptr().cast::<c_void>(),
            expected.len(),
        );

        let mut map_status = native::WGPUMapAsyncStatus_Error;
        let callback_info = native::WGPUBufferMapCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(map_callback),
            userdata1: (&mut map_status as *mut native::WGPUMapAsyncStatus).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future = yawgpu::wgpuBufferMapAsync(
            buffer,
            native::WGPUMapMode_Read,
            0,
            expected.len(),
            callback_info,
        );
        wait(test.instance(), future);

        assert_eq!(map_status, native::WGPUMapAsyncStatus_Success);
        let mapped = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, expected.len());
        assert!(!mapped.is_null());
        // Safety: the successful map exposes exactly `expected.len()` bytes
        // until the buffer is unmapped below.
        assert_eq!(
            std::slice::from_raw_parts(mapped.cast::<u8>(), expected.len()),
            expected
        );
        assert!(test.errors().is_empty());

        yawgpu::wgpuBufferUnmap(buffer);
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuQueueRelease(queue);
    }
}

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
}
