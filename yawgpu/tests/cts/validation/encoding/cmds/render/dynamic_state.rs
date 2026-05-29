//! Ports `$CTS/src/webgpu/api/validation/encoding/cmds/render/dynamic_state.spec.ts`.

use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    color_attachment, create_encoder, expect_command_buffer, release_render_target,
    render_pass_descriptor, CommandExpectation,
};

#[test]
fn set_viewport_width_height_nonnegative() {
    let test = ValidationTest::new();
    unsafe {
        for (x, y, width, height, success) in [
            (0.0, 0.0, 0.0, 0.0, true),
            (0.0, 0.0, -1.0, 0.0, false),
            (0.0, 0.0, 0.0, -1.0, false),
            (1.0, 0.0, -1.0, 0.0, false),
            (0.0, 1.0, 0.0, -1.0, false),
        ] {
            expect_viewport(
                &test,
                extent(1, 1),
                Viewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
                success,
            );
        }
    }
}

#[test]
fn set_viewport_exceeds_attachment_size() {
    let test = ValidationTest::new();
    unsafe {
        for (attachment_width, attachment_height) in [(3, 3), (1024, 1024)] {
            expect_viewport(
                &test,
                extent(attachment_width, attachment_height),
                Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: attachment_width as f32 + 1.0,
                    height: attachment_height as f32 + 1.0,
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
                true,
            );
        }
    }
}

#[test]
#[ignore = "core does not validate viewport rectangle against maximum viewport bounds; CTS expects out-of-bounds viewport rectangles to fail"]
fn set_viewport_xy_rect_contained_in_bounds() {
    let test = ValidationTest::new();
    unsafe {
        let max_viewport_size = device_limits(test.device()).maxTextureDimension2D as f32;
        for (x, y, width, height, success) in [
            (0.0, 0.0, max_viewport_size, max_viewport_size, true),
            (-1.0, 0.0, max_viewport_size, max_viewport_size, true),
            (
                max_viewport_size,
                0.0,
                max_viewport_size,
                max_viewport_size,
                false,
            ),
            (0.0, 0.0, max_viewport_size + 1.0, max_viewport_size, false),
        ] {
            expect_viewport(
                &test,
                extent(1, 1),
                Viewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
                success,
            );
        }
    }
}

#[test]
fn set_viewport_depth_range_and_order() {
    let test = ValidationTest::new();
    unsafe {
        for (min_depth, max_depth, success) in [
            (0.0, 1.0, true),
            (-0.0, -0.0, true),
            (1.0, 1.0, true),
            (0.3, 0.7, true),
            (0.7, 0.7, true),
            (0.3, 0.3, true),
            (-0.1, 1.0, false),
            (0.0, 1.1, false),
            (0.5, 0.49999, false),
        ] {
            expect_viewport(
                &test,
                extent(1, 1),
                Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: 1.0,
                    height: 1.0,
                    min_depth,
                    max_depth,
                },
                success,
            );
        }
    }
}

#[test]
#[ignore = "C webgpu.h setScissorRect parameters are unsigned integers; CTS expects JavaScript TypeError for negative x/y/width/height"]
fn set_scissor_rect_x_y_width_height_nonnegative() {}

#[test]
#[ignore = "core does not validate scissor rectangle against render attachment size; CTS expects rectangles outside the attachment to fail"]
fn set_scissor_rect_xy_rect_contained_in_attachment() {
    let test = ValidationTest::new();
    unsafe {
        for (attachment_width, attachment_height) in [(3, 5), (5, 3), (1024, 1), (1, 1024)] {
            for (dx, dy, dw, dh) in [
                (0, 0, 0, 0),
                (1, 0, -1, 0),
                (0, 1, 0, -1),
                (0, 0, -1, 0),
                (0, 0, 0, -1),
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            ] {
                let width = (attachment_width as i32 + dw) as u32;
                let height = (attachment_width as i32 + dh) as u32;
                let success = dx + width <= attachment_width && dy + height <= attachment_height;
                expect_scissor(
                    &test,
                    extent(attachment_width, attachment_height),
                    Scissor {
                        x: dx,
                        y: dy,
                        width,
                        height,
                    },
                    success,
                );
            }
        }
    }
}

#[test]
fn set_blend_constant() {
    let test = ValidationTest::new();
    unsafe {
        for color in [
            color(1.0, 1.0, 1.0, 1.0),
            color(-1.0, -1.0, -1.0, -1.0),
            color(
                9_007_199_254_740_991.0,
                -9_007_199_254_740_991.0,
                -0.0,
                100000.0,
            ),
        ] {
            expect_dynamic_command(&test, CommandExpectation::Success, |pass| {
                yawgpu::wgpuRenderPassEncoderSetBlendConstant(pass, &color);
            });
        }
    }
}

#[test]
fn set_stencil_reference() {
    let test = ValidationTest::new();
    unsafe {
        for value in [1, 0, 1000, 0xffff_ffff] {
            expect_dynamic_command(&test, CommandExpectation::Success, |pass| {
                yawgpu::wgpuRenderPassEncoderSetStencilReference(pass, value);
            });
        }
    }
}

#[derive(Clone, Copy)]
struct Viewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
}

#[derive(Clone, Copy)]
struct Scissor {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn extent(width: u32, height: u32) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width,
        height,
        depthOrArrayLayers: 1,
    }
}

fn color(r: f64, g: f64, b: f64, a: f64) -> native::WGPUColor {
    native::WGPUColor { r, g, b, a }
}

unsafe fn expect_viewport(
    test: &ValidationTest,
    attachment_size: native::WGPUExtent3D,
    viewport: Viewport,
    success: bool,
) {
    unsafe {
        expect_dynamic_command_with_attachment(
            test,
            attachment_size,
            if success {
                CommandExpectation::Success
            } else {
                CommandExpectation::FinishError
            },
            |pass| {
                yawgpu::wgpuRenderPassEncoderSetViewport(
                    pass,
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    viewport.min_depth,
                    viewport.max_depth,
                );
            },
        );
    }
}

unsafe fn expect_scissor(
    test: &ValidationTest,
    attachment_size: native::WGPUExtent3D,
    scissor: Scissor,
    success: bool,
) {
    unsafe {
        expect_dynamic_command_with_attachment(
            test,
            attachment_size,
            if success {
                CommandExpectation::Success
            } else {
                CommandExpectation::FinishError
            },
            |pass| {
                yawgpu::wgpuRenderPassEncoderSetScissorRect(
                    pass,
                    scissor.x,
                    scissor.y,
                    scissor.width,
                    scissor.height,
                );
            },
        );
    }
}

unsafe fn expect_dynamic_command<F>(
    test: &ValidationTest,
    expectation: CommandExpectation,
    command: F,
) where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    unsafe {
        expect_dynamic_command_with_attachment(test, extent(1, 1), expectation, command);
    }
}

unsafe fn expect_dynamic_command_with_attachment<F>(
    test: &ValidationTest,
    attachment_size: native::WGPUExtent3D,
    expectation: CommandExpectation,
    command: F,
) where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    unsafe {
        let encoder = create_encoder(test.device());
        let target = create_render_target_with_size(test.device(), attachment_size);
        let attachment = color_attachment(target.view);
        let descriptor = render_pass_descriptor(&[attachment], None);
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        assert!(!pass.is_null());
        command(pass);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        expect_command_buffer(test, encoder, expectation);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        release_render_target(target);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

unsafe fn create_render_target_with_size(
    device: native::WGPUDevice,
    size: native::WGPUExtent3D,
) -> crate::common::RenderTarget {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: crate::common::empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment,
        dimension: native::WGPUTextureDimension_2D,
        size,
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = unsafe { yawgpu::wgpuDeviceCreateTexture(device, &descriptor) };
    assert!(!texture.is_null());
    let view = unsafe { yawgpu::wgpuTextureCreateView(texture, std::ptr::null()) };
    assert!(!view.is_null());
    crate::common::RenderTarget { texture, view }
}

unsafe fn device_limits(device: native::WGPUDevice) -> native::WGPULimits {
    let mut limits = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { yawgpu::wgpuDeviceGetLimits(device, &mut limits) },
        native::WGPUStatus_Success
    );
    limits
}
