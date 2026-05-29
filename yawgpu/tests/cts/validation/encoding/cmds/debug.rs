//! CTS port of `webgpu/api/validation/encoding/cmds/debug.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_compute_pass, begin_render_pass, color_attachment, compute_pass_descriptor,
    create_bundle_encoder, create_encoder, create_render_target, empty_string_view,
    expect_command_buffer, render_pass_descriptor, string_view, CommandExpectation,
};

#[test]
fn debug_group_balanced() {
    let test = ValidationTest::new();
    unsafe {
        for encoder_type in EncoderType::all() {
            for push_count in 0..=2 {
                for pop_count in 0..=2 {
                    expect_debug_commands(
                        &test,
                        encoder_type,
                        if push_count == pop_count {
                            CommandExpectation::Success
                        } else {
                            CommandExpectation::FinishError
                        },
                        |encoder| {
                            for index in 0..push_count {
                                push_debug_group(encoder, &index.to_string());
                            }
                            for _ in 0..pop_count {
                                pop_debug_group(encoder);
                            }
                        },
                    );
                }
            }
        }
    }
}

#[test]
fn debug_group() {
    let test = ValidationTest::new();
    unsafe {
        for encoder_type in EncoderType::all() {
            for label in [
                "",
                "group",
                "null\0in\0group\0label",
                "\0null at beginning",
                "\u{1f31e}\u{1f446}",
            ] {
                expect_debug_commands(
                    &test,
                    encoder_type,
                    CommandExpectation::Success,
                    |encoder| {
                        push_debug_group(encoder, label);
                        pop_debug_group(encoder);
                    },
                );
            }
        }
    }
}

#[test]
fn debug_marker() {
    let test = ValidationTest::new();
    unsafe {
        for encoder_type in EncoderType::all() {
            for label in [
                "",
                "marker",
                "null\0in\0marker",
                "\0null at beginning",
                "\u{1f31e}\u{1f446}",
            ] {
                expect_debug_commands(
                    &test,
                    encoder_type,
                    CommandExpectation::Success,
                    |encoder| {
                        insert_debug_marker(encoder, label);
                    },
                );
            }
        }
    }
}

#[derive(Clone, Copy)]
enum EncoderType {
    Command,
    ComputePass,
    RenderPass,
    RenderBundle,
}

impl EncoderType {
    const fn all() -> [Self; 4] {
        [
            Self::Command,
            Self::ComputePass,
            Self::RenderPass,
            Self::RenderBundle,
        ]
    }
}

#[derive(Clone, Copy)]
enum DebugEncoder {
    Command(native::WGPUCommandEncoder),
    ComputePass(native::WGPUComputePassEncoder),
    RenderPass(native::WGPURenderPassEncoder),
    RenderBundle(native::WGPURenderBundleEncoder),
}

unsafe fn expect_debug_commands<F>(
    test: &ValidationTest,
    encoder_type: EncoderType,
    expectation: CommandExpectation,
    commands: F,
) where
    F: FnOnce(DebugEncoder),
{
    unsafe {
        match encoder_type {
            EncoderType::Command => {
                let encoder = create_encoder(test.device());
                commands(DebugEncoder::Command(encoder));
                expect_command_buffer(test, encoder, expectation);
                yawgpu::wgpuCommandEncoderRelease(encoder);
            }
            EncoderType::ComputePass => {
                let encoder = create_encoder(test.device());
                let descriptor = compute_pass_descriptor(None);
                let pass = begin_compute_pass(encoder, Some(&descriptor));
                commands(DebugEncoder::ComputePass(pass));
                yawgpu::wgpuComputePassEncoderEnd(pass);
                expect_command_buffer(test, encoder, expectation);
                yawgpu::wgpuComputePassEncoderRelease(pass);
                yawgpu::wgpuCommandEncoderRelease(encoder);
            }
            EncoderType::RenderPass => {
                let encoder = create_encoder(test.device());
                let target =
                    create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1);
                let attachment = color_attachment(target.view);
                let descriptor = render_pass_descriptor(&[attachment], None);
                let pass = begin_render_pass(encoder, &descriptor);
                commands(DebugEncoder::RenderPass(pass));
                yawgpu::wgpuRenderPassEncoderEnd(pass);
                expect_command_buffer(test, encoder, expectation);
                yawgpu::wgpuRenderPassEncoderRelease(pass);
                yawgpu::wgpuTextureViewRelease(target.view);
                yawgpu::wgpuTextureRelease(target.texture);
                yawgpu::wgpuCommandEncoderRelease(encoder);
            }
            EncoderType::RenderBundle => {
                let encoder = create_bundle_encoder(test);
                commands(DebugEncoder::RenderBundle(encoder));
                match expectation {
                    CommandExpectation::Success => {
                        test.clear_errors();
                        let bundle =
                            yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
                        assert!(!bundle.is_null());
                        assert!(
                            test.errors().is_empty(),
                            "unexpected errors: {:?}",
                            test.errors()
                        );
                        yawgpu::wgpuRenderBundleRelease(bundle);
                    }
                    CommandExpectation::FinishError => {
                        let mut bundle = std::ptr::null();
                        test.assert_device_error_after(
                            || {
                                bundle = yawgpu::wgpuRenderBundleEncoderFinish(
                                    encoder,
                                    std::ptr::null(),
                                );
                            },
                            None,
                        );
                        assert!(!bundle.is_null());
                        yawgpu::wgpuRenderBundleRelease(bundle);
                    }
                    CommandExpectation::SubmitError => unreachable!(),
                }
                yawgpu::wgpuRenderBundleEncoderRelease(encoder);
            }
        }
    }
}

unsafe fn push_debug_group(encoder: DebugEncoder, label: &str) {
    unsafe {
        match encoder {
            DebugEncoder::Command(encoder) => {
                yawgpu::wgpuCommandEncoderPushDebugGroup(encoder, string_view(label));
            }
            DebugEncoder::ComputePass(pass) => {
                yawgpu::wgpuComputePassEncoderPushDebugGroup(pass, string_view(label));
            }
            DebugEncoder::RenderPass(pass) => {
                yawgpu::wgpuRenderPassEncoderPushDebugGroup(pass, string_view(label));
            }
            DebugEncoder::RenderBundle(encoder) => {
                yawgpu::wgpuRenderBundleEncoderPushDebugGroup(encoder, string_view(label));
            }
        }
    }
}

unsafe fn pop_debug_group(encoder: DebugEncoder) {
    unsafe {
        match encoder {
            DebugEncoder::Command(encoder) => yawgpu::wgpuCommandEncoderPopDebugGroup(encoder),
            DebugEncoder::ComputePass(pass) => yawgpu::wgpuComputePassEncoderPopDebugGroup(pass),
            DebugEncoder::RenderPass(pass) => yawgpu::wgpuRenderPassEncoderPopDebugGroup(pass),
            DebugEncoder::RenderBundle(encoder) => {
                yawgpu::wgpuRenderBundleEncoderPopDebugGroup(encoder);
            }
        }
    }
}

unsafe fn insert_debug_marker(encoder: DebugEncoder, label: &str) {
    unsafe {
        match encoder {
            DebugEncoder::Command(encoder) => {
                yawgpu::wgpuCommandEncoderInsertDebugMarker(encoder, string_view(label));
            }
            DebugEncoder::ComputePass(pass) => {
                yawgpu::wgpuComputePassEncoderInsertDebugMarker(pass, string_view(label));
            }
            DebugEncoder::RenderPass(pass) => {
                yawgpu::wgpuRenderPassEncoderInsertDebugMarker(pass, string_view(label));
            }
            DebugEncoder::RenderBundle(encoder) => {
                yawgpu::wgpuRenderBundleEncoderInsertDebugMarker(encoder, string_view(label));
            }
        }
    }
}

fn _unused_empty_string_view_reference() -> native::WGPUStringView {
    empty_string_view()
}
