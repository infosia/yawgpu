//! CTS port of `webgpu/api/validation/encoding/render_bundle.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    begin_render_pass, color_attachment, create_encoder, create_render_target,
    depth_stencil_attachment, expect_command_buffer, render_pass_descriptor, CommandExpectation,
};

#[test]
fn empty_bundle_list() {
    let test = ValidationTest::new();
    unsafe {
        expect_execute_bundles(
            &test,
            &[native::WGPUTextureFormat_RGBA8Unorm],
            native::WGPUTextureFormat_Undefined,
            1,
            &[],
            CommandExpectation::Success,
        );
    }
}

#[test]
fn device_mismatch() {
    let test = ValidationTest::new();
    let other = ValidationTest::new();
    unsafe {
        for (bundle0_mismatched, bundle1_mismatched) in
            [(false, false), (true, false), (false, true)]
        {
            let bundle0_device = if bundle0_mismatched {
                other.device()
            } else {
                test.device()
            };
            let bundle1_device = if bundle1_mismatched {
                other.device()
            } else {
                test.device()
            };
            let bundle0 = create_bundle(
                bundle0_device,
                &[native::WGPUTextureFormat_RGBA8Unorm],
                native::WGPUTextureFormat_Undefined,
                1,
                false,
                false,
            );
            let bundle1 = create_bundle(
                bundle1_device,
                &[native::WGPUTextureFormat_RGBA8Unorm],
                native::WGPUTextureFormat_Undefined,
                1,
                false,
                false,
            );
            expect_execute_bundles(
                &test,
                &[native::WGPUTextureFormat_RGBA8Unorm],
                native::WGPUTextureFormat_Undefined,
                1,
                &[bundle0, bundle1],
                if bundle0_mismatched || bundle1_mismatched {
                    CommandExpectation::FinishError
                } else {
                    CommandExpectation::Success
                },
            );
            yawgpu::wgpuRenderBundleRelease(bundle1);
            yawgpu::wgpuRenderBundleRelease(bundle0);
        }
    }
}

#[test]
fn color_formats_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        let cases: &[(
            &[native::WGPUTextureFormat],
            &[native::WGPUTextureFormat],
            bool,
        )] = &[
            (
                &[
                    native::WGPUTextureFormat_BGRA8Unorm,
                    native::WGPUTextureFormat_RGBA8Unorm,
                ],
                &[
                    native::WGPUTextureFormat_BGRA8Unorm,
                    native::WGPUTextureFormat_RGBA8Unorm,
                ],
                true,
            ),
            (
                &[
                    native::WGPUTextureFormat_BGRA8Unorm,
                    native::WGPUTextureFormat_RGBA8Unorm,
                ],
                &[
                    native::WGPUTextureFormat_BGRA8Unorm,
                    native::WGPUTextureFormat_BGRA8Unorm,
                ],
                false,
            ),
            (
                &[
                    native::WGPUTextureFormat_BGRA8Unorm,
                    native::WGPUTextureFormat_RGBA8Unorm,
                ],
                &[
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureFormat_BGRA8Unorm,
                ],
                false,
            ),
            (
                &[
                    native::WGPUTextureFormat_BGRA8Unorm,
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureFormat_RGBA16Float,
                ],
                &[
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureFormat_BGRA8Unorm,
                ],
                false,
            ),
            (
                &[
                    native::WGPUTextureFormat_BGRA8Unorm,
                    native::WGPUTextureFormat_RGBA8Unorm,
                ],
                &[
                    native::WGPUTextureFormat_RGBA8Unorm,
                    native::WGPUTextureFormat_BGRA8Unorm,
                    native::WGPUTextureFormat_RGBA16Float,
                ],
                false,
            ),
        ];
        for (bundle_formats, pass_formats, compatible) in cases {
            let bundle = create_bundle(
                test.device(),
                bundle_formats,
                native::WGPUTextureFormat_Undefined,
                1,
                false,
                false,
            );
            expect_execute_bundles(
                &test,
                pass_formats,
                native::WGPUTextureFormat_Undefined,
                1,
                &[bundle],
                if *compatible {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuRenderBundleRelease(bundle);
        }
    }
}

#[test]
fn depth_stencil_formats_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        for (bundle_format, pass_format) in [
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUTextureFormat_Depth24Plus,
            ),
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUTextureFormat_Depth16Unorm,
            ),
            (
                native::WGPUTextureFormat_Depth24Plus,
                native::WGPUTextureFormat_Depth24PlusStencil8,
            ),
            (
                native::WGPUTextureFormat_Stencil8,
                native::WGPUTextureFormat_Depth24PlusStencil8,
            ),
        ] {
            let bundle = create_bundle(test.device(), &[], bundle_format, 1, false, false);
            expect_execute_bundles(
                &test,
                &[],
                pass_format,
                1,
                &[bundle],
                if bundle_format == pass_format {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuRenderBundleRelease(bundle);
        }
    }
}

#[test]
#[ignore = "render bundle attachment signatures do not include depthReadOnly/stencilReadOnly; CTS expects executeBundles to validate readonly compatibility"]
fn depth_stencil_readonly_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        for depth_read_only in [false, true] {
            for stencil_read_only in [false, true] {
                let bundle = create_bundle(
                    test.device(),
                    &[],
                    native::WGPUTextureFormat_Depth24PlusStencil8,
                    1,
                    depth_read_only,
                    stencil_read_only,
                );
                expect_execute_bundles(
                    &test,
                    &[],
                    native::WGPUTextureFormat_Depth24PlusStencil8,
                    1,
                    &[bundle],
                    CommandExpectation::Success,
                );
                yawgpu::wgpuRenderBundleRelease(bundle);
            }
        }
    }
}

#[test]
fn sample_count_mismatch() {
    let test = ValidationTest::new();
    unsafe {
        for (bundle_samples, pass_samples) in [(1, 1), (4, 4), (4, 1), (1, 4)] {
            let bundle = create_bundle(
                test.device(),
                &[native::WGPUTextureFormat_RGBA8Unorm],
                native::WGPUTextureFormat_Undefined,
                bundle_samples,
                false,
                false,
            );
            expect_execute_bundles(
                &test,
                &[native::WGPUTextureFormat_RGBA8Unorm],
                native::WGPUTextureFormat_Undefined,
                pass_samples,
                &[bundle],
                if bundle_samples == pass_samples {
                    CommandExpectation::Success
                } else {
                    CommandExpectation::FinishError
                },
            );
            yawgpu::wgpuRenderBundleRelease(bundle);
        }
    }
}

unsafe fn create_bundle(
    device: native::WGPUDevice,
    color_formats: &[native::WGPUTextureFormat],
    depth_stencil_format: native::WGPUTextureFormat,
    sample_count: u32,
    depth_read_only: bool,
    stencil_read_only: bool,
) -> native::WGPURenderBundle {
    unsafe {
        let descriptor = native::WGPURenderBundleEncoderDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: crate::common::empty_string_view(),
            colorFormatCount: color_formats.len(),
            colorFormats: color_formats.as_ptr(),
            depthStencilFormat: depth_stencil_format,
            sampleCount: sample_count,
            depthReadOnly: u32::from(depth_read_only),
            stencilReadOnly: u32::from(stencil_read_only),
        };
        let encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(device, &descriptor);
        assert!(!encoder.is_null());
        let bundle = yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
        assert!(!bundle.is_null());
        yawgpu::wgpuRenderBundleEncoderRelease(encoder);
        bundle
    }
}

unsafe fn expect_execute_bundles(
    test: &ValidationTest,
    color_formats: &[native::WGPUTextureFormat],
    depth_stencil_format: native::WGPUTextureFormat,
    sample_count: u32,
    bundles: &[native::WGPURenderBundle],
    expectation: CommandExpectation,
) {
    unsafe {
        let encoder = create_encoder(test.device());
        let mut targets = Vec::new();
        let mut color_attachments = Vec::new();
        for format in color_formats {
            let target = create_render_target(test.device(), *format, sample_count);
            color_attachments.push(color_attachment(target.view));
            targets.push(target);
        }
        let depth_target = if depth_stencil_format == native::WGPUTextureFormat_Undefined {
            None
        } else {
            Some(create_render_target(
                test.device(),
                depth_stencil_format,
                sample_count,
            ))
        };
        let depth_attachment = depth_target
            .as_ref()
            .map(|target| depth_stencil_attachment(target.view));
        let descriptor = render_pass_descriptor(&color_attachments, depth_attachment.as_ref());
        let pass = begin_render_pass(encoder, &descriptor);
        yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, bundles.len(), bundles.as_ptr());
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        expect_command_buffer(test, encoder, expectation);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        if let Some(target) = depth_target {
            yawgpu::wgpuTextureViewRelease(target.view);
            yawgpu::wgpuTextureRelease(target.texture);
        }
        for target in targets {
            yawgpu::wgpuTextureViewRelease(target.view);
            yawgpu::wgpuTextureRelease(target.texture);
        }
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}
