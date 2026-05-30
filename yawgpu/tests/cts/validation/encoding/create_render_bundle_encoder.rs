//! CTS port of `webgpu/api/validation/encoding/createRenderBundleEncoder.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn attachment_state_limits_max_color_attachments() {
    let test = ValidationTest::new();
    unsafe {
        let limits = device_limits(test.device());
        for color_format_count in 1..=limits.maxColorAttachments {
            let formats = vec![native::WGPUTextureFormat_R8Unorm; color_format_count as usize];
            expect_create_encoder(
                &test,
                &formats,
                native::WGPUTextureFormat_Undefined,
                1,
                true,
            );
        }
        let too_many =
            vec![native::WGPUTextureFormat_R8Unorm; limits.maxColorAttachments as usize + 1];
        expect_create_encoder(
            &test,
            &too_many,
            native::WGPUTextureFormat_Undefined,
            1,
            false,
        );
    }
}

#[test]
fn attachment_state_limits_max_color_attachment_bytes_per_sample_aligned() {
    let test = ValidationTest::new();
    unsafe {
        let formats = vec![native::WGPUTextureFormat_RGBA32Float; 3];
        expect_create_encoder(
            &test,
            &formats,
            native::WGPUTextureFormat_Undefined,
            1,
            false,
        );
    }
}

#[test]
fn attachment_state_limits_max_color_attachment_bytes_per_sample_unaligned() {
    let test = ValidationTest::new();
    unsafe {
        let formats = [
            native::WGPUTextureFormat_R8Unorm,
            native::WGPUTextureFormat_R32Float,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_RGBA32Float,
            native::WGPUTextureFormat_RGBA32Float,
            native::WGPUTextureFormat_R8Unorm,
        ];
        expect_create_encoder(
            &test,
            &formats,
            native::WGPUTextureFormat_Undefined,
            1,
            false,
        );
    }
}

#[test]
fn attachment_state_empty_color_formats() {
    let test = ValidationTest::new();
    unsafe {
        expect_create_encoder(&test, &[], native::WGPUTextureFormat_Undefined, 1, false);
        expect_create_encoder(
            &test,
            &[],
            native::WGPUTextureFormat_Depth24PlusStencil8,
            1,
            true,
        );
    }
}

#[test]
fn valid_texture_formats() {
    let test = ValidationTest::new();
    unsafe {
        for (format, color_renderable, depth_stencil) in [
            (native::WGPUTextureFormat_RGBA8Unorm, true, false),
            (native::WGPUTextureFormat_BGRA8Unorm, true, false),
            (native::WGPUTextureFormat_R8Snorm, false, false),
            (native::WGPUTextureFormat_Depth24Plus, false, true),
            (native::WGPUTextureFormat_Depth24PlusStencil8, false, true),
            (native::WGPUTextureFormat_Stencil8, false, true),
        ] {
            expect_create_encoder(
                &test,
                &[format],
                native::WGPUTextureFormat_Undefined,
                1,
                color_renderable,
            );
            expect_create_encoder(&test, &[], format, 1, depth_stencil);
        }
    }
}

#[test]
fn depth_stencil_readonly() {
    let test = ValidationTest::new();
    unsafe {
        for format in [
            native::WGPUTextureFormat_Depth24Plus,
            native::WGPUTextureFormat_Stencil8,
            native::WGPUTextureFormat_Depth24PlusStencil8,
        ] {
            for depth_read_only in [false, true] {
                for stencil_read_only in [false, true] {
                    let descriptor =
                        bundle_descriptor(&[], format, 1, depth_read_only, stencil_read_only);
                    test.clear_errors();
                    let encoder =
                        yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
                    assert!(!encoder.is_null());
                    assert!(
                        test.errors().is_empty(),
                        "unexpected errors: {:?}",
                        test.errors()
                    );
                    yawgpu::wgpuRenderBundleEncoderRelease(encoder);
                }
            }
        }
    }
}

unsafe fn expect_create_encoder(
    test: &ValidationTest,
    color_formats: &[native::WGPUTextureFormat],
    depth_stencil_format: native::WGPUTextureFormat,
    sample_count: u32,
    success: bool,
) {
    unsafe {
        let descriptor = bundle_descriptor(
            color_formats,
            depth_stencil_format,
            sample_count,
            false,
            false,
        );
        if success {
            test.clear_errors();
            let encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
            assert!(!encoder.is_null());
            assert!(
                test.errors().is_empty(),
                "unexpected errors: {:?}",
                test.errors()
            );
            yawgpu::wgpuRenderBundleEncoderRelease(encoder);
        } else {
            let mut encoder = std::ptr::null();
            test.assert_device_error_after(
                || {
                    encoder =
                        yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
                },
                None,
            );
            assert!(!encoder.is_null());
            yawgpu::wgpuRenderBundleEncoderRelease(encoder);
        }
    }
}

fn bundle_descriptor(
    color_formats: &[native::WGPUTextureFormat],
    depth_stencil_format: native::WGPUTextureFormat,
    sample_count: u32,
    depth_read_only: bool,
    stencil_read_only: bool,
) -> native::WGPURenderBundleEncoderDescriptor {
    native::WGPURenderBundleEncoderDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: crate::common::empty_string_view(),
        colorFormatCount: color_formats.len(),
        colorFormats: color_formats.as_ptr(),
        depthStencilFormat: depth_stencil_format,
        sampleCount: sample_count,
        depthReadOnly: u32::from(depth_read_only),
        stencilReadOnly: u32::from(stencil_read_only),
    }
}

unsafe fn device_limits(device: native::WGPUDevice) -> native::WGPULimits {
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(device, &mut limits),
            native::WGPUStatus_Success
        );
        limits
    }
}
