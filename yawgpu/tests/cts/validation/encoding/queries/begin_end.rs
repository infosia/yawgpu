//! CTS port of `webgpu/api/validation/encoding/queries/begin_end.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_render_pass, color_attachment, create_encoder, create_query_set, create_render_target,
    expect_command_buffer, finish_error, finish_ok, render_pass_descriptor, CommandExpectation,
    RenderTarget,
};

#[test]
fn occlusion_query_begin_end_balance() {
    let test = ValidationTest::new();
    unsafe {
        let cases = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)];
        for (begin, end) in cases {
            let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 2);
            expect_occlusion_commands(
                &test,
                Some(query_set),
                if begin == end {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
                |pass| {
                    for index in 0..begin {
                        yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, index);
                    }
                    for _ in 0..end {
                        yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
                    }
                },
            );
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[test]
fn occlusion_query_begin_end_invalid_nesting() {
    let test = ValidationTest::new();
    unsafe {
        let cases: &[(&[Call], bool)] = &[
            (
                &[Call::Begin(0), Call::End, Call::Begin(1), Call::End],
                true,
            ),
            (
                &[Call::Begin(0), Call::Begin(0), Call::End, Call::End],
                false,
            ),
            (
                &[Call::Begin(0), Call::Begin(1), Call::End, Call::End],
                false,
            ),
        ];
        for (calls, valid) in cases {
            let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 2);
            expect_occlusion_commands(
                &test,
                Some(query_set),
                if *valid {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
                |pass| {
                    for call in *calls {
                        match *call {
                            Call::Begin(index) => {
                                yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, index);
                            }
                            Call::End => yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass),
                        }
                    }
                },
            );
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[test]
fn occlusion_query_disjoint_queries_with_same_query_index() {
    let test = ValidationTest::new();
    unsafe {
        for same_pass in [false, true] {
            let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
            let encoder = create_encoder(test.device());
            let mut targets = Vec::new();
            if same_pass {
                let target =
                    create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1);
                let attachment = color_attachment(target.view);
                let mut descriptor = render_pass_descriptor(&[attachment], None);
                descriptor.occlusionQuerySet = query_set;
                let pass = begin_render_pass(encoder, &descriptor);
                yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
                yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
                yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
                yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
                yawgpu::wgpuRenderPassEncoderEnd(pass);
                yawgpu::wgpuRenderPassEncoderRelease(pass);
                targets.push(target);
            } else {
                targets.push(encode_one_occlusion_pass(&test, encoder, query_set, 0));
                targets.push(encode_one_occlusion_pass(&test, encoder, query_set, 0));
            }
            let command_buffer = if same_pass {
                finish_error(&test, encoder)
            } else {
                finish_ok(&test, encoder)
            };
            yawgpu::wgpuCommandBufferRelease(command_buffer);
            for target in targets {
                yawgpu::wgpuTextureViewRelease(target.view);
                yawgpu::wgpuTextureRelease(target.texture);
            }
            yawgpu::wgpuCommandEncoderRelease(encoder);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
    }
}

#[test]
#[ignore = "CTS marks the query nesting case unimplemented; timestamp nesting subcases require timestamp-query coverage not defined by this CTS test yet"]
fn nesting() {
    // The source CTS `g.test('nesting')` is `.unimplemented()`.
}

#[derive(Clone, Copy)]
enum Call {
    Begin(u32),
    End,
}

unsafe fn expect_occlusion_commands<F>(
    test: &ValidationTest,
    query_set: Option<native::WGPUQuerySet>,
    expectation: CommandExpectation,
    commands: F,
) where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    unsafe {
        let encoder = create_encoder(test.device());
        let target = create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1);
        let attachment = color_attachment(target.view);
        let mut descriptor = render_pass_descriptor(&[attachment], None);
        descriptor.occlusionQuerySet = query_set.unwrap_or(std::ptr::null());
        let pass = begin_render_pass(encoder, &descriptor);
        commands(pass);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        expect_command_buffer(test, encoder, expectation);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        yawgpu::wgpuTextureViewRelease(target.view);
        yawgpu::wgpuTextureRelease(target.texture);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

unsafe fn encode_one_occlusion_pass(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
    query_set: native::WGPUQuerySet,
    query_index: u32,
) -> RenderTarget {
    unsafe {
        let target = create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1);
        let attachment = color_attachment(target.view);
        let mut descriptor = render_pass_descriptor(&[attachment], None);
        descriptor.occlusionQuerySet = query_set;
        let pass = begin_render_pass(encoder, &descriptor);
        yawgpu::wgpuRenderPassEncoderBeginOcclusionQuery(pass, query_index);
        yawgpu::wgpuRenderPassEncoderEndOcclusionQuery(pass);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        target
    }
}
