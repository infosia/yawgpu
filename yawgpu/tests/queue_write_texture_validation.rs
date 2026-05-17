use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[derive(Clone, Copy)]
struct WriteArgs {
    texture: native::WGPUTexture,
    mip_level: u32,
    origin: native::WGPUOrigin3D,
    write_size: native::WGPUExtent3D,
    aspect: native::WGPUTextureAspect,
    data_layout: native::WGPUTexelCopyBufferLayout,
    data_size: u64,
}

#[test]
fn destination_usage_lifetime_sample_count_and_mip_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());

        let wrong_usage = create_texture(test.device(), native::WGPUTextureUsage_TextureBinding);
        assert_write_error(
            &test,
            queue,
            wrong_usage,
            0,
            origin(0, 0, 0),
            extent(1, 1, 1),
        );
        yawgpu::wgpuTextureRelease(wrong_usage);

        let destroyed = create_texture(test.device(), native::WGPUTextureUsage_CopyDst);
        yawgpu::wgpuTextureDestroy(destroyed);
        assert_write_error(&test, queue, destroyed, 0, origin(0, 0, 0), extent(1, 1, 1));
        yawgpu::wgpuTextureRelease(destroyed);

        let multisampled = create_texture_with_descriptor(
            test.device(),
            native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_RenderAttachment,
                sampleCount: 4,
                ..texture_descriptor(native::WGPUTextureUsage_CopyDst)
            },
        );
        assert_write_error(
            &test,
            queue,
            multisampled,
            0,
            origin(0, 0, 0),
            extent(1, 1, 1),
        );
        yawgpu::wgpuTextureRelease(multisampled);

        let texture = create_texture_with_descriptor(
            test.device(),
            native::WGPUTextureDescriptor {
                mipLevelCount: 2,
                ..texture_descriptor(native::WGPUTextureUsage_CopyDst)
            },
        );
        assert_write_error(&test, queue, texture, 2, origin(0, 0, 0), extent(1, 1, 1));
        yawgpu::wgpuTextureRelease(texture);

        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn origin_and_extent_must_fit_subresource() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let texture = create_texture_with_descriptor(
            test.device(),
            native::WGPUTextureDescriptor {
                size: extent(4, 2, 1),
                mipLevelCount: 3,
                ..texture_descriptor(native::WGPUTextureUsage_CopyDst)
            },
        );

        assert_write_ok(&test, queue, texture, 1, origin(0, 0, 0), extent(2, 1, 1));
        assert_write_error(&test, queue, texture, 1, origin(1, 0, 0), extent(2, 1, 1));
        assert_write_error(
            &test,
            queue,
            texture,
            0,
            origin(u32::MAX, 0, 0),
            extent(1, 1, 1),
        );

        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn writes_to_2d_textures_must_have_one_layer() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let texture = create_texture_with_descriptor(
            test.device(),
            native::WGPUTextureDescriptor {
                size: extent(4, 4, 2),
                ..texture_descriptor(native::WGPUTextureUsage_CopyDst)
            },
        );

        assert_write_error(&test, queue, texture, 0, origin(0, 0, 0), extent(1, 1, 2));

        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn bytes_per_row_uses_queue_row_size_without_256_alignment() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let texture = create_texture_with_descriptor(
            test.device(),
            native::WGPUTextureDescriptor {
                size: extent(3, 7, 1),
                ..texture_descriptor(native::WGPUTextureUsage_CopyDst)
            },
        );

        assert_write_with_layout_error(
            &test,
            queue,
            texture,
            extent(3, 7, 1),
            layout(0, 11, 7),
            128,
        );
        assert_write_with_layout_ok(&test, queue, texture, extent(3, 7, 1), layout(0, 12, 7), 84);
        assert_write_with_layout_error(
            &test,
            queue,
            texture,
            extent(3, 7, 1),
            layout_undefined(0),
            84,
        );
        assert_write_with_layout_ok(
            &test,
            queue,
            texture,
            extent(3, 1, 1),
            layout_undefined(0),
            12,
        );

        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn rows_per_image_and_required_data_size_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let texture_3d = create_texture_with_descriptor(
            test.device(),
            native::WGPUTextureDescriptor {
                dimension: native::WGPUTextureDimension_3D,
                size: extent(4, 4, 2),
                ..texture_descriptor(native::WGPUTextureUsage_CopyDst)
            },
        );

        assert_write_with_layout_error(
            &test,
            queue,
            texture_3d,
            extent(4, 4, 2),
            layout(0, 16, 3),
            128,
        );
        assert_write_with_layout_ok(
            &test,
            queue,
            texture_3d,
            extent(4, 4, 2),
            layout(0, 16, 4),
            128,
        );

        yawgpu::wgpuTextureRelease(texture_3d);

        let texture = create_texture(test.device(), native::WGPUTextureUsage_CopyDst);
        assert_write_with_layout_error(
            &test,
            queue,
            texture,
            extent(4, 4, 1),
            layout(0, 16, 4),
            63,
        );
        assert_write_with_layout_ok(&test, queue, texture, extent(4, 4, 1), layout(0, 16, 4), 64);
        assert_write_with_layout_error(
            &test,
            queue,
            texture,
            extent(1, 1, 1),
            layout_undefined(u64::MAX),
            u64::MAX,
        );

        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn aspect_must_match_texture_format() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());

        let color = create_texture(test.device(), native::WGPUTextureUsage_CopyDst);
        assert_write_with_aspect_error(
            &test,
            queue,
            color,
            native::WGPUTextureAspect_DepthOnly,
            layout(0, 16, 4),
            64,
        );
        assert_write_with_aspect_error(
            &test,
            queue,
            color,
            native::WGPUTextureAspect_StencilOnly,
            layout(0, 4, 4),
            16,
        );
        yawgpu::wgpuTextureRelease(color);

        let depth = create_texture_with_descriptor(
            test.device(),
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_Depth24Plus,
                ..texture_descriptor(native::WGPUTextureUsage_CopyDst)
            },
        );
        assert_write_with_aspect_ok(
            &test,
            queue,
            depth,
            native::WGPUTextureAspect_DepthOnly,
            layout(0, 16, 4),
            64,
        );
        yawgpu::wgpuTextureRelease(depth);

        let stencil = create_texture_with_descriptor(
            test.device(),
            native::WGPUTextureDescriptor {
                format: native::WGPUTextureFormat_Depth24PlusStencil8,
                ..texture_descriptor(native::WGPUTextureUsage_CopyDst)
            },
        );
        assert_write_with_aspect_ok(
            &test,
            queue,
            stencil,
            native::WGPUTextureAspect_StencilOnly,
            layout(0, 4, 4),
            16,
        );
        yawgpu::wgpuTextureRelease(stencil);

        yawgpu::wgpuQueueRelease(queue);
    }
}

#[test]
fn valid_copy_dst_write_has_no_error() {
    let test = ValidationTest::new();
    unsafe {
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let texture = create_texture(test.device(), native::WGPUTextureUsage_CopyDst);

        assert_write_with_layout_ok(&test, queue, texture, extent(4, 4, 1), layout(0, 16, 4), 64);

        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
    }
}

unsafe fn assert_write_error(
    test: &ValidationTest,
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    mip_level: u32,
    origin: native::WGPUOrigin3D,
    write_size: native::WGPUExtent3D,
) {
    assert_write_full_error(
        test,
        queue,
        WriteArgs {
            texture,
            mip_level,
            origin,
            write_size,
            aspect: native::WGPUTextureAspect_Undefined,
            data_layout: layout(0, 4, 1),
            data_size: 4,
        },
    );
}

unsafe fn assert_write_ok(
    test: &ValidationTest,
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    mip_level: u32,
    origin: native::WGPUOrigin3D,
    write_size: native::WGPUExtent3D,
) {
    assert_write_full_ok(
        test,
        queue,
        WriteArgs {
            texture,
            mip_level,
            origin,
            write_size,
            aspect: native::WGPUTextureAspect_Undefined,
            data_layout: layout_undefined(0),
            data_size: u64::from(write_size.width) * 4,
        },
    );
}

unsafe fn assert_write_with_layout_error(
    test: &ValidationTest,
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    write_size: native::WGPUExtent3D,
    data_layout: native::WGPUTexelCopyBufferLayout,
    data_size: u64,
) {
    assert_write_full_error(
        test,
        queue,
        WriteArgs {
            texture,
            mip_level: 0,
            origin: origin(0, 0, 0),
            write_size,
            aspect: native::WGPUTextureAspect_Undefined,
            data_layout,
            data_size,
        },
    );
}

unsafe fn assert_write_with_layout_ok(
    test: &ValidationTest,
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    write_size: native::WGPUExtent3D,
    data_layout: native::WGPUTexelCopyBufferLayout,
    data_size: u64,
) {
    assert_write_full_ok(
        test,
        queue,
        WriteArgs {
            texture,
            mip_level: 0,
            origin: origin(0, 0, 0),
            write_size,
            aspect: native::WGPUTextureAspect_Undefined,
            data_layout,
            data_size,
        },
    );
}

unsafe fn assert_write_with_aspect_error(
    test: &ValidationTest,
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    aspect: native::WGPUTextureAspect,
    data_layout: native::WGPUTexelCopyBufferLayout,
    data_size: u64,
) {
    assert_write_full_error(
        test,
        queue,
        WriteArgs {
            texture,
            mip_level: 0,
            origin: origin(0, 0, 0),
            write_size: extent(4, 4, 1),
            aspect,
            data_layout,
            data_size,
        },
    );
}

unsafe fn assert_write_with_aspect_ok(
    test: &ValidationTest,
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    aspect: native::WGPUTextureAspect,
    data_layout: native::WGPUTexelCopyBufferLayout,
    data_size: u64,
) {
    assert_write_full_ok(
        test,
        queue,
        WriteArgs {
            texture,
            mip_level: 0,
            origin: origin(0, 0, 0),
            write_size: extent(4, 4, 1),
            aspect,
            data_layout,
            data_size,
        },
    );
}

unsafe fn assert_write_full_error(
    test: &ValidationTest,
    queue: native::WGPUQueue,
    args: WriteArgs,
) {
    assert_device_error!({
        queue_write_texture(queue, args);
    });
    assert_eq!(test.errors().len(), 1);
}

unsafe fn assert_write_full_ok(test: &ValidationTest, queue: native::WGPUQueue, args: WriteArgs) {
    test.clear_errors();
    queue_write_texture(queue, args);
    assert!(test.errors().is_empty());
}

unsafe fn queue_write_texture(queue: native::WGPUQueue, args: WriteArgs) {
    let destination = native::WGPUTexelCopyTextureInfo {
        texture: args.texture,
        mipLevel: args.mip_level,
        origin: args.origin,
        aspect: args.aspect,
    };
    let data_size = usize::try_from(args.data_size).unwrap_or(usize::MAX);
    yawgpu::wgpuQueueWriteTexture(
        queue,
        &destination,
        std::ptr::null::<c_void>(),
        data_size,
        &args.data_layout,
        &args.write_size,
    );
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    create_texture_with_descriptor(device, texture_descriptor(usage))
}

unsafe fn create_texture_with_descriptor(
    device: native::WGPUDevice,
    descriptor: native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

fn texture_descriptor(usage: native::WGPUTextureUsage) -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: extent(4, 4, 1),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    }
}

fn layout(
    offset: u64,
    bytes_per_row: u32,
    rows_per_image: u32,
) -> native::WGPUTexelCopyBufferLayout {
    native::WGPUTexelCopyBufferLayout {
        offset,
        bytesPerRow: bytes_per_row,
        rowsPerImage: rows_per_image,
    }
}

fn layout_undefined(offset: u64) -> native::WGPUTexelCopyBufferLayout {
    native::WGPUTexelCopyBufferLayout {
        offset,
        bytesPerRow: native::WGPU_COPY_STRIDE_UNDEFINED,
        rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
    }
}

fn origin(x: u32, y: u32, z: u32) -> native::WGPUOrigin3D {
    native::WGPUOrigin3D { x, y, z }
}

fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width,
        height,
        depthOrArrayLayers: depth_or_array_layers,
    }
}
