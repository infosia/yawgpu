//! CTS port of `webgpu/api/validation/queue/destroyed/texture.spec.ts`.

use yawgpu::native;
use yawgpu_test::{assert_device_error, expect_no_validation_error, ValidationTest};

use super::common::*;

#[test]
fn write_texture() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for destroyed in [false, true] {
            let texture = create_texture(test.device(), native::WGPUTextureUsage_CopyDst);
            if destroyed {
                yawgpu::wgpuTextureDestroy(texture);
            }
            if destroyed {
                assert_device_error!({
                    queue_write_texture(q, texture, 4);
                });
            } else {
                expect_no_validation_error(|| queue_write_texture(q, texture, 4));
            }
            yawgpu::wgpuTextureRelease(texture);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn copy_texture_to_texture() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for destroyed in ["none", "src", "dst", "both"] {
            let src = create_texture(test.device(), native::WGPUTextureUsage_CopySrc);
            let dst = create_texture(test.device(), native::WGPUTextureUsage_CopyDst);
            let cb = encode_copy_texture_to_texture(test.device(), src, dst);
            if destroyed == "src" || destroyed == "both" {
                yawgpu::wgpuTextureDestroy(src);
            }
            if destroyed == "dst" || destroyed == "both" {
                yawgpu::wgpuTextureDestroy(dst);
            }
            expect_submit(&test, q, &[cb], destroyed == "none");
            yawgpu::wgpuCommandBufferRelease(cb);
            yawgpu::wgpuTextureRelease(dst);
            yawgpu::wgpuTextureRelease(src);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn copy_buffer_to_texture() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for destroyed in [false, true] {
            let buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_CopySrc, false);
            let texture = create_texture(test.device(), native::WGPUTextureUsage_CopyDst);
            let cb = encode_copy_buffer_to_texture(test.device(), buffer, texture);
            if destroyed {
                yawgpu::wgpuTextureDestroy(texture);
            }
            expect_submit(&test, q, &[cb], !destroyed);
            yawgpu::wgpuCommandBufferRelease(cb);
            yawgpu::wgpuTextureRelease(texture);
            yawgpu::wgpuBufferRelease(buffer);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn copy_texture_to_buffer() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for destroyed in [false, true] {
            let texture = create_texture(test.device(), native::WGPUTextureUsage_CopySrc);
            let buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_CopyDst, false);
            let cb = encode_copy_texture_to_buffer(test.device(), texture, buffer);
            if destroyed {
                yawgpu::wgpuTextureDestroy(texture);
            }
            expect_submit(&test, q, &[cb], !destroyed);
            yawgpu::wgpuCommandBufferRelease(cb);
            yawgpu::wgpuBufferRelease(buffer);
            yawgpu::wgpuTextureRelease(texture);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn set_bind_group() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for destroyed in [false, true] {
            for kind in [
                EncoderKind::ComputePass,
                EncoderKind::RenderPass,
                EncoderKind::RenderBundle,
            ] {
                for storage in [false, true] {
                    let texture = create_texture(
                        test.device(),
                        native::WGPUTextureUsage_TextureBinding
                            | native::WGPUTextureUsage_StorageBinding,
                    );
                    let view = create_texture_view(texture);
                    let (layout, bind_group) =
                        create_texture_bind_group(test.device(), view, storage);
                    let cb = record_bind_group_use(&test, kind, bind_group);
                    if destroyed {
                        yawgpu::wgpuTextureDestroy(texture);
                    }
                    expect_submit(&test, q, &[cb], !destroyed);
                    yawgpu::wgpuCommandBufferRelease(cb);
                    yawgpu::wgpuBindGroupRelease(bind_group);
                    yawgpu::wgpuBindGroupLayoutRelease(layout);
                    yawgpu::wgpuTextureViewRelease(view);
                    yawgpu::wgpuTextureRelease(texture);
                }
            }
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn begin_render_pass() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for texture_to_destroy in [
            "none",
            "colorAttachment",
            "resolveAttachment",
            "depthStencilAttachment",
        ] {
            let color = create_texture_with_descriptor(
                test.device(),
                native::WGPUTextureDescriptor {
                    usage: native::WGPUTextureUsage_RenderAttachment,
                    sampleCount: 4,
                    ..texture_descriptor(native::WGPUTextureUsage_RenderAttachment)
                },
            );
            let resolve = create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
            let depth = create_texture_with_descriptor(
                test.device(),
                native::WGPUTextureDescriptor {
                    usage: native::WGPUTextureUsage_RenderAttachment,
                    format: native::WGPUTextureFormat_Depth32Float,
                    sampleCount: 4,
                    ..texture_descriptor(native::WGPUTextureUsage_RenderAttachment)
                },
            );
            let cb = encode_render_pass_with_attachments(test.device(), color, resolve, depth);
            match texture_to_destroy {
                "colorAttachment" => yawgpu::wgpuTextureDestroy(color),
                "resolveAttachment" => yawgpu::wgpuTextureDestroy(resolve),
                "depthStencilAttachment" => yawgpu::wgpuTextureDestroy(depth),
                "none" => {}
                _ => unreachable!(),
            }
            expect_submit(&test, q, &[cb], texture_to_destroy == "none");
            yawgpu::wgpuCommandBufferRelease(cb);
            yawgpu::wgpuTextureRelease(depth);
            yawgpu::wgpuTextureRelease(resolve);
            yawgpu::wgpuTextureRelease(color);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}
