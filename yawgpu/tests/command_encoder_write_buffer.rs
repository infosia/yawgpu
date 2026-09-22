//! Block 100: `wgpuCommandEncoderWriteBuffer` executes on the Noop backend.
//!
//! The encoder-side write is snapshotted at encode time and replayed at
//! submit in command order; the Noop HAL executes the staged copy eagerly
//! into host storage, so the destination can be mapped and read back.

use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{wait, ValidationTest};

const EMPTY_LABEL: native::WGPUStringView = native::WGPUStringView {
    data: std::ptr::null(),
    length: 0,
};

unsafe fn create_buffer(
    device: native::WGPUDevice,
    usage: native::WGPUBufferUsage,
    size: u64,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: EMPTY_LABEL,
        usage,
        size,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

/// Maps `buffer` for reading, copies out `size` bytes, and unmaps it.
unsafe fn read_back(test: &ValidationTest, buffer: native::WGPUBuffer, size: usize) -> Vec<u8> {
    let mut map_status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut map_status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, size, callback_info);
    wait(test.instance(), future);
    assert_eq!(map_status, native::WGPUMapAsyncStatus_Success);
    let mapped = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, size);
    assert!(!mapped.is_null());
    // Safety: the successful map exposes exactly `size` bytes until unmap.
    let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), size).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

#[test]
fn encoder_write_buffer_writes_bytes_at_offset_and_leaves_the_rest_zero() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let buffer = create_buffer(
            test.device(),
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
            32,
        );

        // The source bytes are mutated after the call to prove the encoder
        // took a copy (R1).
        let mut data: Vec<u8> = (0x10..0x20).collect();
        let expected_data = data.clone();
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
        yawgpu::wgpuCommandEncoderWriteBuffer(
            encoder,
            buffer,
            4,
            data.as_ptr().cast::<c_void>(),
            data.len(),
        );
        data.fill(0xff);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        assert!(!command_buffer.is_null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);

        let bytes = read_back(&test, buffer, 32);
        let mut expected = vec![0_u8; 32];
        expected[4..20].copy_from_slice(&expected_data);
        assert_eq!(bytes, expected);
        assert!(test.errors().is_empty(), "{:?}", test.errors());

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn encoder_write_buffer_then_copy_in_one_encoder_orders_the_write_first() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let a = create_buffer(
            test.device(),
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
            16,
        );
        let b = create_buffer(
            test.device(),
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
            16,
        );

        let data: Vec<u8> = (1..=16).collect();
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
        yawgpu::wgpuCommandEncoderWriteBuffer(
            encoder,
            a,
            0,
            data.as_ptr().cast::<c_void>(),
            data.len(),
        );
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, a, 0, b, 0, 16);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        assert!(!command_buffer.is_null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);

        assert_eq!(read_back(&test, b, 16), data);
        assert!(test.errors().is_empty(), "{:?}", test.errors());

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuBufferRelease(b);
        yawgpu::wgpuBufferRelease(a);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn encoder_write_buffer_null_data_with_size_reports_a_validation_error() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_CopyDst, 16);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
        test.assert_device_error_after(
            || yawgpu::wgpuCommandEncoderWriteBuffer(encoder, buffer, 0, std::ptr::null(), 4),
            Some("command encoder write buffer data must not be null"),
        );
        // Null with size 0 is silent.
        test.expect_no_validation_error(|| {
            yawgpu::wgpuCommandEncoderWriteBuffer(encoder, buffer, 0, std::ptr::null(), 0);
        });
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuBufferRelease(buffer);
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
