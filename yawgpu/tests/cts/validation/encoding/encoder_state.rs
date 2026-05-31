//! Ports `$CTS/src/webgpu/api/validation/encoding/encoder_state.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_compute_pass, begin_render_pass, color_attachment, create_encoder, create_view,
    empty_string_view, expect_finish, finish_ok, release_view, render_pass_descriptor,
};

#[derive(Clone, Copy)]
enum PassType {
    Compute,
    Render,
}

#[test]
fn pass_end_invalid_order() {
    let test = ValidationTest::new();
    unsafe {
        for pass0_type in [PassType::Compute, PassType::Render] {
            for pass1_type in [PassType::Compute, PassType::Render] {
                for first_pass_end in [true, false] {
                    let end_pass_tables: &[&[usize]] = &[&[], &[0], &[1], &[0, 1], &[1, 0]];
                    for end_passes in end_pass_tables {
                        let view = create_view(
                            test.device(),
                            native::WGPUTextureFormat_RGBA8Unorm,
                            native::WGPUTextureUsage_RenderAttachment,
                            1,
                        );
                        let attachment = color_attachment(view.view);
                        let attachments = [attachment];
                        let descriptor = render_pass_descriptor(&attachments, None);
                        let encoder = create_encoder(test.device());

                        let first = begin_pass(encoder, pass0_type, &descriptor);
                        if first_pass_end {
                            end_pass(first);
                        }
                        let second = begin_pass(encoder, pass1_type, &descriptor);
                        let passes = [first, second];

                        for &index in *end_passes {
                            let valid_end =
                                (index == 0 && !first_pass_end) || (index == 1 && first_pass_end);
                            if valid_end {
                                test.clear_errors();
                                end_pass(passes[index]);
                                assert!(
                                    test.errors().is_empty(),
                                    "unexpected end error for pass {index}: {:?}",
                                    test.errors()
                                );
                            } else {
                                test.assert_device_error_after(|| end_pass(passes[index]), None);
                            }
                        }

                        let valid_finish = first_pass_end && end_passes.contains(&1);
                        let command_buffer = expect_finish(&test, encoder, valid_finish);
                        release_pass(first);
                        release_pass(second);
                        yawgpu::wgpuCommandBufferRelease(command_buffer);
                        yawgpu::wgpuCommandEncoderRelease(encoder);
                        release_view(view);
                    }
                }
            }
        }
    }
}

#[test]
fn call_after_successful_finish() {
    let test = ValidationTest::new();
    unsafe {
        for call_cmd in [
            "beginComputePass",
            "beginRenderPass",
            "finishAndSubmitFirst",
            "finishAndSubmitSecond",
            "insertDebugMarker",
        ] {
            for pre_pass_type in [None, Some(PassType::Compute), Some(PassType::Render)] {
                for is_encoder_finished in [false, true] {
                    let view = create_view(
                        test.device(),
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureUsage_RenderAttachment,
                        1,
                    );
                    let attachment = color_attachment(view.view);
                    let attachments = [attachment];
                    let descriptor = render_pass_descriptor(&attachments, None);
                    let encoder = create_encoder(test.device());

                    if let Some(pass_type) = pre_pass_type {
                        let pass = begin_pass(encoder, pass_type, &descriptor);
                        end_pass(pass);
                        release_pass(pass);
                    }

                    let first_buffer = if is_encoder_finished {
                        Some(finish_ok(&test, encoder))
                    } else {
                        None
                    };

                    match call_cmd {
                        "beginComputePass" => {
                            let mut pass = std::ptr::null();
                            expect_device_error(&test, is_encoder_finished, || {
                                pass = yawgpu::wgpuCommandEncoderBeginComputePass(
                                    encoder,
                                    std::ptr::null(),
                                );
                            });
                            assert!(!pass.is_null());
                            expect_device_error(&test, is_encoder_finished, || {
                                yawgpu::wgpuComputePassEncoderEnd(pass);
                            });
                            yawgpu::wgpuComputePassEncoderRelease(pass);
                        }
                        "beginRenderPass" => {
                            let mut pass = std::ptr::null();
                            expect_device_error(&test, is_encoder_finished, || {
                                pass =
                                    yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
                            });
                            assert!(!pass.is_null());
                            expect_device_error(&test, is_encoder_finished, || {
                                yawgpu::wgpuRenderPassEncoderEnd(pass);
                            });
                            yawgpu::wgpuRenderPassEncoderRelease(pass);
                        }
                        "finishAndSubmitFirst" => {
                            let second_buffer = expect_finish(&test, encoder, !is_encoder_finished);
                            yawgpu::wgpuCommandBufferRelease(second_buffer);
                        }
                        "finishAndSubmitSecond" => {
                            let second_buffer = expect_finish(&test, encoder, !is_encoder_finished);
                            let queue = yawgpu::wgpuDeviceGetQueue(test.device());
                            expect_device_error(&test, is_encoder_finished, || {
                                yawgpu::wgpuQueueSubmit(queue, 1, &second_buffer);
                            });
                            yawgpu::wgpuCommandBufferRelease(second_buffer);
                        }
                        "insertDebugMarker" => {
                            expect_device_error(&test, is_encoder_finished, || {
                                yawgpu::wgpuCommandEncoderInsertDebugMarker(
                                    encoder,
                                    empty_string_view(),
                                );
                            });
                        }
                        _ => unreachable!(),
                    }

                    if !is_encoder_finished && !call_cmd.starts_with("finish") {
                        let command_buffer = finish_ok(&test, encoder);
                        yawgpu::wgpuCommandBufferRelease(command_buffer);
                    }
                    if let Some(command_buffer) = first_buffer {
                        yawgpu::wgpuCommandBufferRelease(command_buffer);
                    }
                    yawgpu::wgpuCommandEncoderRelease(encoder);
                    release_view(view);
                }
            }
        }
    }
}

#[test]
fn pass_end_none() {
    let test = ValidationTest::new();
    unsafe {
        for pass_type in [PassType::Compute, PassType::Render] {
            for end_count in [0, 1] {
                let view = create_view(
                    test.device(),
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureUsage_RenderAttachment,
                    1,
                );
                let attachment = color_attachment(view.view);
                let attachments = [attachment];
                let descriptor = render_pass_descriptor(&attachments, None);
                let encoder = create_encoder(test.device());
                let pass = begin_pass(encoder, pass_type, &descriptor);
                for _ in 0..end_count {
                    end_pass(pass);
                }
                let command_buffer = expect_finish(&test, encoder, end_count == 1);
                yawgpu::wgpuCommandBufferRelease(command_buffer);
                release_pass(pass);
                yawgpu::wgpuCommandEncoderRelease(encoder);
                release_view(view);
            }
        }
    }
}

#[test]
fn pass_end_twice_basic() {
    let test = ValidationTest::new();
    unsafe {
        for pass_type in [PassType::Compute, PassType::Render] {
            for end_twice in [false, true] {
                for second_end_in_another_pass in
                    [None, Some(PassType::Compute), Some(PassType::Render)]
                {
                    if !end_twice && second_end_in_another_pass.is_some() {
                        continue;
                    }
                    let view = create_view(
                        test.device(),
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureUsage_RenderAttachment,
                        1,
                    );
                    let attachment = color_attachment(view.view);
                    let attachments = [attachment];
                    let descriptor = render_pass_descriptor(&attachments, None);
                    let encoder = create_encoder(test.device());
                    let pass = begin_pass(encoder, pass_type, &descriptor);
                    end_pass(pass);

                    let other = second_end_in_another_pass.map(|other_type| {
                        let other = begin_pass(encoder, other_type, &descriptor);
                        test.assert_device_error_after(|| end_pass(pass), None);
                        end_pass(other);
                        other
                    });
                    if second_end_in_another_pass.is_none() && end_twice {
                        test.assert_device_error_after(|| end_pass(pass), None);
                    }

                    let command_buffer = finish_ok(&test, encoder);
                    yawgpu::wgpuCommandBufferRelease(command_buffer);
                    if let Some(other) = other {
                        release_pass(other);
                    }
                    release_pass(pass);
                    yawgpu::wgpuCommandEncoderRelease(encoder);
                    release_view(view);
                }
            }
        }
    }
}

#[test]
fn pass_end_twice_render_pass_invalid() {
    let test = ValidationTest::new();
    unsafe {
        for end_twice in [false, true] {
            let encoder = create_encoder(test.device());
            let attachments = [];
            let descriptor = render_pass_descriptor(&attachments, None);
            let pass = begin_render_pass(encoder, &descriptor);

            yawgpu::wgpuRenderPassEncoderEnd(pass);
            if end_twice {
                test.assert_device_error_after(
                    || {
                        yawgpu::wgpuRenderPassEncoderEnd(pass);
                    },
                    None,
                );
            }
            let command_buffer = expect_finish(&test, encoder, false);
            yawgpu::wgpuCommandBufferRelease(command_buffer);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
            yawgpu::wgpuCommandEncoderRelease(encoder);
        }
    }
}

#[test]
fn pass_begin_invalid_encoder() {
    let test = ValidationTest::new();
    unsafe {
        for pass0_type in [PassType::Compute, PassType::Render] {
            for pass1_type in [PassType::Compute, PassType::Render] {
                for first_pass_invalid in [false, true] {
                    let view = create_view(
                        test.device(),
                        native::WGPUTextureFormat_RGBA8Unorm,
                        native::WGPUTextureUsage_RenderAttachment,
                        1,
                    );
                    let attachment = color_attachment(view.view);
                    let attachments = [attachment];
                    let descriptor = render_pass_descriptor(&attachments, None);
                    let encoder = create_encoder(test.device());

                    let first = begin_pass(encoder, pass0_type, &descriptor);
                    if first_pass_invalid {
                        test.clear_errors();
                        pop_pass_debug_group(first);
                        assert!(test.errors().is_empty());
                    }
                    end_pass(first);

                    let second = begin_pass(encoder, pass1_type, &descriptor);
                    end_pass(second);
                    let command_buffer = expect_finish(&test, encoder, !first_pass_invalid);

                    yawgpu::wgpuCommandBufferRelease(command_buffer);
                    release_pass(second);
                    release_pass(first);
                    yawgpu::wgpuCommandEncoderRelease(encoder);
                    release_view(view);
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Pass {
    Compute(native::WGPUComputePassEncoder),
    Render(native::WGPURenderPassEncoder),
}

unsafe fn begin_pass(
    encoder: native::WGPUCommandEncoder,
    pass_type: PassType,
    render_descriptor: &native::WGPURenderPassDescriptor,
) -> Pass {
    match pass_type {
        PassType::Compute => Pass::Compute(begin_compute_pass(encoder, None)),
        PassType::Render => Pass::Render(begin_render_pass(encoder, render_descriptor)),
    }
}

unsafe fn end_pass(pass: Pass) {
    match pass {
        Pass::Compute(pass) => unsafe { yawgpu::wgpuComputePassEncoderEnd(pass) },
        Pass::Render(pass) => unsafe { yawgpu::wgpuRenderPassEncoderEnd(pass) },
    }
}

unsafe fn pop_pass_debug_group(pass: Pass) {
    match pass {
        Pass::Compute(pass) => unsafe { yawgpu::wgpuComputePassEncoderPopDebugGroup(pass) },
        Pass::Render(pass) => unsafe { yawgpu::wgpuRenderPassEncoderPopDebugGroup(pass) },
    }
}

unsafe fn release_pass(pass: Pass) {
    match pass {
        Pass::Compute(pass) => unsafe { yawgpu::wgpuComputePassEncoderRelease(pass) },
        Pass::Render(pass) => unsafe { yawgpu::wgpuRenderPassEncoderRelease(pass) },
    }
}

fn expect_device_error<F>(test: &ValidationTest, should_error: bool, action: F)
where
    F: FnOnce(),
{
    if should_error {
        test.assert_device_error_after(action, None);
    } else {
        test.clear_errors();
        action();
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );
    }
}
