//! CTS port of `webgpu/api/validation/queue/buffer_mapped.spec.ts`.

use yawgpu::native;
use yawgpu_test::{assert_device_error, expect_no_validation_error, wait, ValidationTest};

use super::common::*;

unsafe fn run_buffer_dependency_test<F>(
    test: &ValidationTest,
    usage: native::WGPUBufferUsage,
    mut callback: F,
) where
    F: FnMut(native::WGPUBuffer, bool),
{
    unsafe {
        let map_mode = if usage & native::WGPUBufferUsage_MapRead != 0 {
            native::WGPUMapMode_Read
        } else {
            native::WGPUMapMode_Write
        };
        let mappable = create_buffer(test.device(), 4096, usage, false);
        let unmapped = create_buffer(test.device(), 4096, usage, false);

        callback(mappable, true);

        let mut state = MapState::default();
        let future = map_async(
            mappable,
            map_mode,
            native::WGPUCallbackMode_WaitAnyOnly,
            &mut state,
        );
        callback(mappable, false);
        callback(unmapped, true);

        wait(test.instance(), future);
        callback(mappable, false);

        let _range = yawgpu::wgpuBufferGetMappedRange(mappable, 0, 4);
        callback(mappable, false);

        yawgpu::wgpuBufferUnmap(mappable);
        callback(mappable, true);

        let mapped_at_creation = create_buffer(test.device(), 4096, usage, true);
        callback(mapped_at_creation, false);
        callback(unmapped, true);
        yawgpu::wgpuBufferUnmap(mapped_at_creation);
        callback(mapped_at_creation, true);

        yawgpu::wgpuBufferRelease(mapped_at_creation);
        yawgpu::wgpuBufferRelease(unmapped);
        yawgpu::wgpuBufferRelease(mappable);
    }
}

#[test]
fn write_buffer() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        run_buffer_dependency_test(
            &test,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
            |buffer, success| {
                if success {
                    expect_no_validation_error(|| queue_write_buffer(q, buffer, 0, 4));
                } else {
                    assert_device_error!({
                        queue_write_buffer(q, buffer, 0, 4);
                    });
                }
            },
        );
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn copy_buffer_to_buffer() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        let source = create_buffer(test.device(), 4096, native::WGPUBufferUsage_CopySrc, false);
        let destination =
            create_buffer(test.device(), 4096, native::WGPUBufferUsage_CopyDst, false);

        run_buffer_dependency_test(
            &test,
            native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_CopySrc,
            |buffer, success| {
                let cb = encode_copy_buffer_to_buffer(test.device(), buffer, destination);
                expect_submit(&test, q, &[cb], success);
                yawgpu::wgpuCommandBufferRelease(cb);
            },
        );
        run_buffer_dependency_test(
            &test,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
            |buffer, success| {
                let cb = encode_copy_buffer_to_buffer(test.device(), source, buffer);
                expect_submit(&test, q, &[cb], success);
                yawgpu::wgpuCommandBufferRelease(cb);
            },
        );

        yawgpu::wgpuBufferRelease(destination);
        yawgpu::wgpuBufferRelease(source);
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn copy_buffer_to_texture() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        let texture = create_texture(test.device(), native::WGPUTextureUsage_CopyDst);

        run_buffer_dependency_test(
            &test,
            native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_CopySrc,
            |buffer, success| {
                let cb = encode_copy_buffer_to_texture(test.device(), buffer, texture);
                expect_submit(&test, q, &[cb], success);
                yawgpu::wgpuCommandBufferRelease(cb);
            },
        );

        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn copy_texture_to_buffer() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        let texture = create_texture(test.device(), native::WGPUTextureUsage_CopySrc);

        run_buffer_dependency_test(
            &test,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
            |buffer, success| {
                let cb = encode_copy_texture_to_buffer(test.device(), texture, buffer);
                expect_submit(&test, q, &[cb], success);
                yawgpu::wgpuCommandBufferRelease(cb);
            },
        );

        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn map_command_recording_order() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for (order, mapped_at_creation, should_error) in [
            ("record,map,unmap,finish,submit", false, false),
            ("record,map,finish,unmap,submit", false, false),
            ("record,finish,map,unmap,submit", false, false),
            ("map,record,unmap,finish,submit", false, false),
            ("map,record,finish,unmap,submit", false, false),
            ("map,record,finish,submit,unmap", false, true),
            ("record,map,finish,submit,unmap", false, true),
            ("record,finish,map,submit,unmap", false, true),
            ("record,unmap,finish,submit", true, false),
            ("record,finish,unmap,submit", true, false),
            ("record,finish,submit,unmap", true, true),
        ] {
            let buffer = create_buffer(
                test.device(),
                4,
                native::WGPUBufferUsage_MapWrite | native::WGPUBufferUsage_CopySrc,
                mapped_at_creation,
            );
            let target = create_buffer(test.device(), 4, native::WGPUBufferUsage_CopyDst, false);
            let encoder = create_encoder(test.device());
            let mut command_buffer = std::ptr::null();
            let mut map_state = MapState::default();

            for step in order.split(',') {
                match step {
                    "record" => {
                        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(
                            encoder, buffer, 0, target, 0, 4,
                        );
                    }
                    "map" => {
                        wait(
                            test.instance(),
                            map_async(
                                buffer,
                                native::WGPUMapMode_Write,
                                native::WGPUCallbackMode_AllowProcessEvents,
                                &mut map_state,
                            ),
                        );
                    }
                    "unmap" => yawgpu::wgpuBufferUnmap(buffer),
                    "finish" => {
                        command_buffer =
                            yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
                        assert!(!command_buffer.is_null());
                    }
                    "submit" => {
                        expect_submit(&test, q, &[command_buffer], !should_error);
                    }
                    _ => unreachable!(),
                }
            }

            yawgpu::wgpuCommandBufferRelease(command_buffer);
            yawgpu::wgpuCommandEncoderRelease(encoder);
            yawgpu::wgpuBufferRelease(target);
            yawgpu::wgpuBufferRelease(buffer);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}
