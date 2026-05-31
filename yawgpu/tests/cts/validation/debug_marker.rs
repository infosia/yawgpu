//! CTS port of `webgpu/api/validation/debugMarker.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_compute_pass, begin_render_pass, color_attachment, compute_pass_descriptor,
    create_encoder, create_render_target, expect_command_buffer, render_pass_descriptor,
    string_view, CommandExpectation,
};

#[test]
fn push_pop_call_count_unbalance_command_encoder() {
    let test = ValidationTest::new();
    unsafe {
        for push_count in 1..=3 {
            for pop_count in 1..=3 {
                let encoder = create_encoder(test.device());
                for _ in 0..push_count {
                    yawgpu::wgpuCommandEncoderPushDebugGroup(encoder, string_view("EventStart"));
                }
                yawgpu::wgpuCommandEncoderInsertDebugMarker(encoder, string_view("Marker"));
                for _ in 0..pop_count {
                    yawgpu::wgpuCommandEncoderPopDebugGroup(encoder);
                }
                expect_command_buffer(
                    &test,
                    encoder,
                    if push_count == pop_count {
                        CommandExpectation::Success
                    } else {
                        CommandExpectation::FinishError
                    },
                );
                yawgpu::wgpuCommandEncoderRelease(encoder);
            }
        }
    }
}

#[test]
fn push_pop_call_count_unbalance_render_compute_pass() {
    let test = ValidationTest::new();
    unsafe {
        for pass_type in [PassType::Compute, PassType::Render] {
            for push_count in 1..=3 {
                for pop_count in 1..=3 {
                    expect_pass_debug_balance(
                        &test,
                        pass_type,
                        push_count,
                        pop_count,
                        if push_count == pop_count {
                            CommandExpectation::Success
                        } else {
                            CommandExpectation::FinishError
                        },
                    );
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum PassType {
    Compute,
    Render,
}

unsafe fn expect_pass_debug_balance(
    test: &ValidationTest,
    pass_type: PassType,
    push_count: u32,
    pop_count: u32,
    expectation: CommandExpectation,
) {
    unsafe {
        let encoder = create_encoder(test.device());
        match pass_type {
            PassType::Compute => {
                let descriptor = compute_pass_descriptor(None);
                let pass = begin_compute_pass(encoder, Some(&descriptor));
                for _ in 0..push_count {
                    yawgpu::wgpuComputePassEncoderPushDebugGroup(pass, string_view("EventStart"));
                }
                yawgpu::wgpuComputePassEncoderInsertDebugMarker(pass, string_view("Marker"));
                for _ in 0..pop_count {
                    yawgpu::wgpuComputePassEncoderPopDebugGroup(pass);
                }
                yawgpu::wgpuComputePassEncoderEnd(pass);
                expect_command_buffer(test, encoder, expectation);
                yawgpu::wgpuComputePassEncoderRelease(pass);
            }
            PassType::Render => {
                let target =
                    create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1);
                let attachment = color_attachment(target.view);
                let attachments = [attachment];
                let descriptor = render_pass_descriptor(&attachments, None);
                let pass = begin_render_pass(encoder, &descriptor);
                for _ in 0..push_count {
                    yawgpu::wgpuRenderPassEncoderPushDebugGroup(pass, string_view("EventStart"));
                }
                yawgpu::wgpuRenderPassEncoderInsertDebugMarker(pass, string_view("Marker"));
                for _ in 0..pop_count {
                    yawgpu::wgpuRenderPassEncoderPopDebugGroup(pass);
                }
                yawgpu::wgpuRenderPassEncoderEnd(pass);
                expect_command_buffer(test, encoder, expectation);
                yawgpu::wgpuRenderPassEncoderRelease(pass);
                yawgpu::wgpuTextureViewRelease(target.view);
                yawgpu::wgpuTextureRelease(target.texture);
            }
        }
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}
