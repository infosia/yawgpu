use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn submit_error_command_buffer_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let encoder = create_encoder(&test);
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_CopySrc, 16, false);
        yawgpu::wgpuCommandEncoderClearBuffer(encoder, buffer, 0, 16);
        let command_buffer = finish_error(&test, encoder);

        assert_device_error!({
            submit(queue, &[command_buffer]);
        });

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn submit_command_buffer_twice_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let encoder = create_encoder(&test);
        let command_buffer = finish_ok(&test, encoder);

        test.clear_errors();
        submit(queue, &[command_buffer]);
        assert!(test.errors().is_empty());

        assert_device_error!({
            submit(queue, &[command_buffer]);
        });

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn submit_referencing_mapped_buffer_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_CopyDst, 16, true);
        let encoder = create_encoder(&test);
        yawgpu::wgpuCommandEncoderClearBuffer(encoder, buffer, 0, 16);
        let command_buffer = finish_ok(&test, encoder);

        assert_device_error!({
            submit(queue, &[command_buffer]);
        });

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn submit_referencing_destroyed_buffer_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_CopyDst, 16, false);
        let encoder = create_encoder(&test);
        yawgpu::wgpuCommandEncoderClearBuffer(encoder, buffer, 0, 16);
        let command_buffer = finish_ok(&test, encoder);
        yawgpu::wgpuBufferDestroy(buffer);

        assert_device_error!({
            submit(queue, &[command_buffer]);
        });

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn clean_submit_has_no_error() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_CopyDst, 16, false);
        let encoder = create_encoder(&test);
        yawgpu::wgpuCommandEncoderClearBuffer(encoder, buffer, 0, 16);
        let command_buffer = finish_ok(&test, encoder);

        test.clear_errors();
        submit(queue, &[command_buffer]);
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuQueueRelease(queue);
    }
}

unsafe fn submit(queue: native::WGPUQueue, command_buffers: &[native::WGPUCommandBuffer]) {
    yawgpu::wgpuQueueSubmit(queue, command_buffers.len(), command_buffers.as_ptr());
}

unsafe fn create_encoder(test: &ValidationTest) -> native::WGPUCommandEncoder {
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
    assert!(!encoder.is_null());
    encoder
}

unsafe fn finish_ok(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    test.clear_errors();
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    command_buffer
}

unsafe fn finish_error(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    let mut command_buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        },
        None,
    );
    assert!(!command_buffer.is_null());
    command_buffer
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    usage: native::WGPUBufferUsage,
    size: u64,
    mapped_at_creation: bool,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: u32::from(mapped_at_creation),
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
