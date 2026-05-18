use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn copy_buffer_to_buffer_bounds_alignment_and_zero_size_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let source = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc, false);
        let destination = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, false);

        assert_copy_ok(&test, source, 0, destination, 0, 16);
        assert_copy_ok(&test, source, 8, destination, 0, 8);
        assert_copy_ok(&test, source, 0, destination, 8, 8);
        assert_copy_ok(&test, source, 16, destination, 16, 0);

        assert_copy_error(&test, source, 8, destination, 0, 12);
        assert_copy_error(&test, source, 0, destination, 8, 12);
        assert_copy_error(&test, source, u64::MAX - 3, destination, 0, 8);
        assert_copy_error(&test, source, 9, destination, 0, 4);
        assert_copy_error(&test, source, 8, destination, 1, 4);
        assert_copy_error(&test, source, 8, destination, 0, 1);

        yawgpu::wgpuBufferRelease(destination);
        yawgpu::wgpuBufferRelease(source);
    }
}

#[test]
fn copy_buffer_to_buffer_usage_and_buffer_state_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let source = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc, false);
        let destination = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, false);
        let vertex = create_buffer(test.device(), 16, native::WGPUBufferUsage_Vertex, false);

        assert_copy_error(&test, vertex, 0, destination, 0, 16);
        assert_copy_error(&test, source, 0, vertex, 0, 16);

        let error = create_error_buffer(&test, native::WGPUBufferUsage_CopySrc);
        assert_copy_error(&test, error, 0, destination, 0, 4);
        assert_copy_error(&test, source, 0, error, 0, 4);

        let destroyed_src =
            create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc, false);
        yawgpu::wgpuBufferDestroy(destroyed_src);
        assert_copy_error(&test, destroyed_src, 0, destination, 0, 4);

        let destroyed_dst =
            create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, false);
        yawgpu::wgpuBufferDestroy(destroyed_dst);
        assert_copy_error(&test, source, 0, destroyed_dst, 0, 4);

        yawgpu::wgpuBufferRelease(destroyed_dst);
        yawgpu::wgpuBufferRelease(destroyed_src);
        yawgpu::wgpuBufferRelease(error);
        yawgpu::wgpuBufferRelease(vertex);
        yawgpu::wgpuBufferRelease(destination);
        yawgpu::wgpuBufferRelease(source);
    }
}

#[test]
fn copy_buffer_to_buffer_same_buffer_matches_dawn() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(
            test.device(),
            16,
            native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
            false,
        );

        assert_copy_error(&test, buffer, 0, buffer, 4, 8);
        assert_copy_error(&test, buffer, 0, buffer, 8, 8);
        assert_copy_error(&test, buffer, 4, buffer, 0, 8);
        assert_copy_error(&test, buffer, 8, buffer, 0, 8);
        assert_copy_ok(&test, buffer, 0, buffer, 0, 0);

        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn clear_buffer_validation_is_deferred_to_finish() {
    let test = ValidationTest::new();
    unsafe {
        let destination = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, false);
        let vertex = create_buffer(test.device(), 16, native::WGPUBufferUsage_Vertex, false);

        assert_clear_ok(&test, destination, 0, 16);
        assert_clear_ok(&test, destination, 8, 8);
        assert_clear_ok(&test, destination, 0, u64::MAX);
        assert_clear_ok(&test, destination, 8, u64::MAX);

        assert_clear_error(&test, vertex, 0, 16);
        assert_clear_error(&test, destination, 2, 4);
        assert_clear_error(&test, destination, 0, 2);
        assert_clear_error(&test, destination, 8, 12);
        assert_clear_error(&test, destination, 20, 0);

        let error = create_error_buffer(&test, native::WGPUBufferUsage_CopyDst);
        assert_clear_error(&test, error, 0, 4);

        let destroyed = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, false);
        yawgpu::wgpuBufferDestroy(destroyed);
        assert_clear_error(&test, destroyed, 0, 4);

        yawgpu::wgpuBufferRelease(destroyed);
        yawgpu::wgpuBufferRelease(error);
        yawgpu::wgpuBufferRelease(vertex);
        yawgpu::wgpuBufferRelease(destination);
    }
}

#[test]
fn command_encoder_write_buffer_validation_is_deferred_to_finish() {
    let test = ValidationTest::new();
    unsafe {
        let destination = create_buffer(test.device(), 64, native::WGPUBufferUsage_CopyDst, false);
        let source_only = create_buffer(test.device(), 64, native::WGPUBufferUsage_CopySrc, false);
        let data = [0_u8; 64];

        assert_write_ok(&test, destination, 0, data.as_ptr().cast(), 64);
        assert_write_ok(&test, destination, 4, data.as_ptr().cast(), 60);
        assert_write_ok(&test, destination, 40, data.as_ptr().cast(), 24);

        assert_write_error(&test, source_only, 0, data.as_ptr().cast(), 64);
        assert_write_error(&test, destination, 1, data.as_ptr().cast(), 4);
        assert_write_error(&test, destination, 4, data.as_ptr().cast(), 2);
        assert_write_error(&test, destination, 0, data.as_ptr().cast(), 68);
        assert_write_error(&test, destination, 60, data.as_ptr().cast(), 8);
        assert_write_error(&test, destination, u64::MAX - 3, data.as_ptr().cast(), 8);

        let error = create_error_buffer(&test, native::WGPUBufferUsage_CopyDst);
        assert_write_error(&test, error, 0, data.as_ptr().cast(), 4);

        let destroyed = create_buffer(test.device(), 64, native::WGPUBufferUsage_CopyDst, false);
        yawgpu::wgpuBufferDestroy(destroyed);
        assert_write_error(&test, destroyed, 0, data.as_ptr().cast(), 4);

        yawgpu::wgpuBufferRelease(destroyed);
        yawgpu::wgpuBufferRelease(error);
        yawgpu::wgpuBufferRelease(source_only);
        yawgpu::wgpuBufferRelease(destination);
    }
}

#[test]
fn copy_commands_after_finish_are_immediate_errors() {
    let test = ValidationTest::new();
    unsafe {
        let source = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopySrc, false);
        let destination = create_buffer(test.device(), 16, native::WGPUBufferUsage_CopyDst, false);
        let data = [0_u8; 4];
        let encoder = create_encoder(&test);
        let command_buffer = finish_ok(&test, encoder);

        test.assert_device_error_after(
            || {
                yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, source, 0, destination, 0, 4);
            },
            None,
        );
        test.assert_device_error_after(
            || {
                yawgpu::wgpuCommandEncoderClearBuffer(encoder, destination, 0, 4);
            },
            None,
        );
        test.assert_device_error_after(
            || {
                yawgpu::wgpuCommandEncoderWriteBuffer(
                    encoder,
                    destination,
                    0,
                    data.as_ptr().cast(),
                    data.len(),
                );
            },
            None,
        );

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuBufferRelease(destination);
        yawgpu::wgpuBufferRelease(source);
    }
}

unsafe fn assert_copy_ok(
    test: &ValidationTest,
    source: native::WGPUBuffer,
    source_offset: u64,
    destination: native::WGPUBuffer,
    destination_offset: u64,
    size: u64,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(
        encoder,
        source,
        source_offset,
        destination,
        destination_offset,
        size,
    );
    assert!(test.errors().is_empty());
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_copy_error(
    test: &ValidationTest,
    source: native::WGPUBuffer,
    source_offset: u64,
    destination: native::WGPUBuffer,
    destination_offset: u64,
    size: u64,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(
        encoder,
        source,
        source_offset,
        destination,
        destination_offset,
        size,
    );
    assert!(test.errors().is_empty());
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_clear_ok(
    test: &ValidationTest,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    yawgpu::wgpuCommandEncoderClearBuffer(encoder, buffer, offset, size);
    assert!(test.errors().is_empty());
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_clear_error(
    test: &ValidationTest,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    yawgpu::wgpuCommandEncoderClearBuffer(encoder, buffer, offset, size);
    assert!(test.errors().is_empty());
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_write_ok(
    test: &ValidationTest,
    buffer: native::WGPUBuffer,
    offset: u64,
    data: *const std::ffi::c_void,
    size: usize,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    yawgpu::wgpuCommandEncoderWriteBuffer(encoder, buffer, offset, data, size);
    assert!(test.errors().is_empty());
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_write_error(
    test: &ValidationTest,
    buffer: native::WGPUBuffer,
    offset: u64,
    data: *const std::ffi::c_void,
    size: usize,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    yawgpu::wgpuCommandEncoderWriteBuffer(encoder, buffer, offset, data, size);
    assert!(test.errors().is_empty());
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
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
    size: u64,
    usage: native::WGPUBufferUsage,
    mapped_at_creation: bool,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: mapped_at_creation.into(),
    };
    yawgpu::wgpuDeviceCreateBuffer(device, &descriptor)
}

unsafe fn create_error_buffer(
    test: &ValidationTest,
    extra_usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let mut buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            buffer = create_buffer(
                test.device(),
                4,
                native::WGPUBufferUsage_MapRead
                    | native::WGPUBufferUsage_CopySrc
                    | native::WGPUBufferUsage_CopyDst
                    | extra_usage,
                false,
            );
        },
        None,
    );
    assert!(!buffer.is_null());
    buffer
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
