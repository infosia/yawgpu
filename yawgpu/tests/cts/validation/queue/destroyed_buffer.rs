//! CTS port of `webgpu/api/validation/queue/destroyed/buffer.spec.ts`.

use yawgpu::native;
use yawgpu_test::{assert_device_error, expect_no_validation_error, ValidationTest};

use super::common::*;

#[test]
fn write_buffer() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for destroyed in [false, true] {
            let buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_CopyDst, false);
            if destroyed {
                yawgpu::wgpuBufferDestroy(buffer);
            }
            if destroyed {
                assert_device_error!({
                    queue_write_buffer(q, buffer, 0, 4);
                });
            } else {
                expect_no_validation_error(|| queue_write_buffer(q, buffer, 0, 4));
            }
            yawgpu::wgpuBufferRelease(buffer);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn copy_buffer_to_buffer() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for destroyed in ["none", "src", "dst", "both"] {
            let src = create_buffer(test.device(), 4, native::WGPUBufferUsage_CopySrc, false);
            let dst = create_buffer(test.device(), 4, native::WGPUBufferUsage_CopyDst, false);
            let cb = encode_copy_buffer_to_buffer(test.device(), src, dst);
            if destroyed == "src" || destroyed == "both" {
                yawgpu::wgpuBufferDestroy(src);
            }
            if destroyed == "dst" || destroyed == "both" {
                yawgpu::wgpuBufferDestroy(dst);
            }
            expect_submit(&test, q, &[cb], destroyed == "none");
            yawgpu::wgpuCommandBufferRelease(cb);
            yawgpu::wgpuBufferRelease(dst);
            yawgpu::wgpuBufferRelease(src);
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
                yawgpu::wgpuBufferDestroy(buffer);
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
                yawgpu::wgpuBufferDestroy(buffer);
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
                let buffer =
                    create_buffer(test.device(), 4, native::WGPUBufferUsage_Uniform, false);
                let (layout, bind_group) = create_buffer_bind_group(test.device(), buffer);
                let cb = record_bind_group_use(&test, kind, bind_group);
                if destroyed {
                    yawgpu::wgpuBufferDestroy(buffer);
                }
                expect_submit(&test, q, &[cb], !destroyed);
                yawgpu::wgpuCommandBufferRelease(cb);
                yawgpu::wgpuBindGroupRelease(bind_group);
                yawgpu::wgpuBindGroupLayoutRelease(layout);
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn set_vertex_buffer() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for destroyed in [false, true] {
            for kind in [EncoderKind::RenderPass, EncoderKind::RenderBundle] {
                let buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_Vertex, false);
                let cb = record_vertex_buffer_use(&test, kind, buffer);
                if destroyed {
                    yawgpu::wgpuBufferDestroy(buffer);
                }
                expect_submit(&test, q, &[cb], !destroyed);
                yawgpu::wgpuCommandBufferRelease(cb);
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn set_index_buffer() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for destroyed in [false, true] {
            for kind in [EncoderKind::RenderPass, EncoderKind::RenderBundle] {
                let buffer = create_buffer(test.device(), 4, native::WGPUBufferUsage_Index, false);
                let cb = record_index_buffer_use(&test, kind, buffer);
                if destroyed {
                    yawgpu::wgpuBufferDestroy(buffer);
                }
                expect_submit(&test, q, &[cb], !destroyed);
                yawgpu::wgpuCommandBufferRelease(cb);
                yawgpu::wgpuBufferRelease(buffer);
            }
        }
        yawgpu::wgpuQueueRelease(q);
    }
}

#[test]
fn resolve_query_set() {
    let test = ValidationTest::new();
    unsafe {
        let q = queue(test.device());
        for destroyed in [false, true] {
            let query_set = create_query_set(test.device(), native::WGPUQueryType_Occlusion, 1);
            let buffer = create_buffer(
                test.device(),
                8,
                native::WGPUBufferUsage_QueryResolve,
                false,
            );
            let cb = encode_resolve_query_set(test.device(), query_set, buffer);
            if destroyed {
                yawgpu::wgpuBufferDestroy(buffer);
            }
            expect_submit(&test, q, &[cb], !destroyed);
            yawgpu::wgpuCommandBufferRelease(cb);
            yawgpu::wgpuBufferRelease(buffer);
            yawgpu::wgpuQuerySetRelease(query_set);
        }
        yawgpu::wgpuQueueRelease(q);
    }
}
