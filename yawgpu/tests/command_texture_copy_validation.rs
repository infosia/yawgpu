use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn valid_texture_copy_commands_finish_successfully() {
    let test = ValidationTest::new();
    unsafe {
        let copy_src_buffer = create_buffer(test.device(), 1024, native::WGPUBufferUsage_CopySrc);
        let copy_dst_buffer = create_buffer(test.device(), 1024, native::WGPUBufferUsage_CopyDst);
        let copy_src_texture = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopySrc,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                1,
            ),
        );
        let copy_dst_texture = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                1,
            ),
        );

        assert_b2t_ok(&test, copy_src_buffer, copy_dst_texture, default_b2t_args());
        assert_t2b_ok(&test, copy_src_texture, copy_dst_buffer, default_t2b_args());
        assert_t2t_ok(
            &test,
            copy_src_texture,
            copy_dst_texture,
            default_t2t_args(),
        );

        yawgpu::wgpuTextureRelease(copy_dst_texture);
        yawgpu::wgpuTextureRelease(copy_src_texture);
        yawgpu::wgpuBufferRelease(copy_dst_buffer);
        yawgpu::wgpuBufferRelease(copy_src_buffer);
    }
}

#[test]
fn buffer_texture_usage_sample_count_and_layout_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let copy_src_buffer = create_buffer(test.device(), 4096, native::WGPUBufferUsage_CopySrc);
        let copy_dst_buffer = create_buffer(test.device(), 4096, native::WGPUBufferUsage_CopyDst);
        let vertex_buffer = create_buffer(test.device(), 4096, native::WGPUBufferUsage_Vertex);
        let copy_src_texture = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopySrc,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(128, 16, 1),
                1,
                1,
            ),
        );
        let copy_dst_texture = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(128, 16, 1),
                1,
                1,
            ),
        );
        let sampled_texture = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_TextureBinding,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(128, 16, 1),
                1,
                1,
            ),
        );
        let multisampled_dst = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_RenderAttachment,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                4,
            ),
        );
        let multisampled_src = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_RenderAttachment,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                4,
            ),
        );

        assert_b2t_error(&test, vertex_buffer, copy_dst_texture, default_b2t_args());
        assert_b2t_error(&test, copy_src_buffer, sampled_texture, default_b2t_args());
        assert_t2b_error(&test, sampled_texture, copy_dst_buffer, default_t2b_args());
        assert_t2b_error(&test, copy_src_texture, vertex_buffer, default_t2b_args());
        assert_b2t_error(&test, copy_src_buffer, multisampled_dst, default_b2t_args());
        assert_t2b_error(&test, multisampled_src, copy_dst_buffer, default_t2b_args());

        let error_buffer = create_error_buffer(&test, native::WGPUBufferUsage_CopySrc);
        let error_texture = create_error_texture(
            &test,
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(0, 4, 1),
                1,
                1,
            ),
        );
        assert_b2t_error(&test, error_buffer, copy_dst_texture, default_b2t_args());
        assert_b2t_error(&test, copy_src_buffer, error_texture, default_b2t_args());

        let destroyed_buffer = create_buffer(test.device(), 4096, native::WGPUBufferUsage_CopySrc);
        let destroyed_texture = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(128, 16, 1),
                1,
                1,
            ),
        );
        yawgpu::wgpuBufferDestroy(destroyed_buffer);
        yawgpu::wgpuTextureDestroy(destroyed_texture);
        assert_b2t_submit_error(
            &test,
            destroyed_buffer,
            copy_dst_texture,
            default_b2t_args(),
        );
        assert_b2t_submit_error(
            &test,
            copy_src_buffer,
            destroyed_texture,
            default_b2t_args(),
        );

        let mut args = default_b2t_args();
        args.layout = layout(0, 16, 2);
        args.copy_size = extent(4, 2, 1);
        assert_b2t_error(&test, copy_src_buffer, copy_dst_texture, args);

        let mut args = default_b2t_args();
        args.layout = layout(1, 256, 4);
        assert_b2t_error(&test, copy_src_buffer, copy_dst_texture, args);

        let mut args = default_t2b_args();
        args.layout = layout(1, 256, 4);
        assert_t2b_error(&test, copy_src_texture, copy_dst_buffer, args);

        let mut args = default_b2t_args();
        args.layout = layout(0, 256, 1);
        args.copy_size = extent(65, 1, 1);
        assert_b2t_error(&test, copy_src_buffer, copy_dst_texture, args);

        let texture_3d = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_3D,
                extent(4, 4, 4),
                1,
                1,
            ),
        );
        let mut args = default_b2t_args();
        args.layout = layout(0, 256, 1);
        args.copy_size = extent(4, 4, 2);
        assert_b2t_error(&test, copy_src_buffer, texture_3d, args);

        yawgpu::wgpuTextureRelease(texture_3d);
        yawgpu::wgpuTextureRelease(destroyed_texture);
        yawgpu::wgpuBufferRelease(destroyed_buffer);
        yawgpu::wgpuTextureRelease(error_texture);
        yawgpu::wgpuBufferRelease(error_buffer);
        yawgpu::wgpuTextureRelease(multisampled_src);
        yawgpu::wgpuTextureRelease(multisampled_dst);
        yawgpu::wgpuTextureRelease(sampled_texture);
        yawgpu::wgpuTextureRelease(copy_dst_texture);
        yawgpu::wgpuTextureRelease(copy_src_texture);
        yawgpu::wgpuBufferRelease(vertex_buffer);
        yawgpu::wgpuBufferRelease(copy_dst_buffer);
        yawgpu::wgpuBufferRelease(copy_src_buffer);
    }
}

#[test]
fn buffer_texture_bounds_mips_and_aspects_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let copy_src_buffer = create_buffer(test.device(), 1024, native::WGPUBufferUsage_CopySrc);
        let copy_dst_buffer = create_buffer(test.device(), 8, native::WGPUBufferUsage_CopyDst);
        let color_dst = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                1,
            ),
        );
        let color_src = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopySrc,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                1,
            ),
        );
        let depth_dst = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_Depth16Unorm,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 2),
                1,
                1,
            ),
        );

        let mut args = default_b2t_args();
        args.mip_level = 1;
        assert_b2t_error(&test, copy_src_buffer, color_dst, args);

        let mut args = default_b2t_args();
        args.origin = origin(2, 0, 0);
        assert_b2t_error(&test, copy_src_buffer, color_dst, args);

        assert_t2b_error(&test, color_src, copy_dst_buffer, default_t2b_args());

        let mut args = default_b2t_args();
        args.aspect = native::WGPUTextureAspect_DepthOnly;
        assert_b2t_error(&test, copy_src_buffer, color_dst, args);

        let mut args = default_b2t_args();
        args.aspect = native::WGPUTextureAspect_StencilOnly;
        assert_b2t_error(&test, copy_src_buffer, depth_dst, args);

        let mut args = default_b2t_args();
        args.aspect = native::WGPUTextureAspect_DepthOnly;
        args.copy_size = extent(4, 4, 2);
        assert_b2t_error(&test, copy_src_buffer, depth_dst, args);

        yawgpu::wgpuTextureRelease(depth_dst);
        yawgpu::wgpuTextureRelease(color_src);
        yawgpu::wgpuTextureRelease(color_dst);
        yawgpu::wgpuBufferRelease(copy_dst_buffer);
        yawgpu::wgpuBufferRelease(copy_src_buffer);
    }
}

#[test]
fn texture_to_texture_rules_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let copy_src = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopySrc,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(16, 16, 2),
                2,
                1,
            ),
        );
        let copy_dst = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(16, 16, 2),
                2,
                1,
            ),
        );
        let sampled = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_TextureBinding,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(16, 16, 2),
                2,
                1,
            ),
        );
        let uint_dst = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8Uint,
                native::WGPUTextureDimension_2D,
                extent(16, 16, 2),
                2,
                1,
            ),
        );
        let srgb_dst = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8UnormSrgb,
                native::WGPUTextureDimension_2D,
                extent(16, 16, 2),
                2,
                1,
            ),
        );

        assert_t2t_error(&test, copy_src, uint_dst, default_t2t_args());
        assert_t2t_ok(&test, copy_src, srgb_dst, default_t2t_args());
        assert_t2t_error(&test, sampled, copy_dst, default_t2t_args());
        assert_t2t_error(&test, copy_src, sampled, default_t2t_args());

        let mut args = default_t2t_args();
        args.source_origin = origin(15, 0, 0);
        args.copy_size = extent(2, 1, 1);
        assert_t2t_error(&test, copy_src, copy_dst, args);

        let mut args = default_t2t_args();
        args.destination_origin = origin(15, 0, 0);
        args.copy_size = extent(2, 1, 1);
        assert_t2t_error(&test, copy_src, copy_dst, args);

        let depth_src = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopySrc,
                native::WGPUTextureFormat_Depth16Unorm,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                1,
            ),
        );
        let depth_dst = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_Depth16Unorm,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                1,
            ),
        );
        let mut args = default_t2t_args();
        args.source_aspect = native::WGPUTextureAspect_DepthOnly;
        args.destination_aspect = native::WGPUTextureAspect_DepthOnly;
        assert_t2t_error(&test, depth_src, depth_dst, args);

        let multisampled_dst = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_RenderAttachment,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(16, 16, 1),
                1,
                4,
            ),
        );
        assert_t2t_error(&test, copy_src, multisampled_dst, default_t2t_args());

        let same_texture = create_texture(
            test.device(),
            texture_descriptor(
                native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                native::WGPUTextureFormat_RGBA8Unorm,
                native::WGPUTextureDimension_2D,
                extent(16, 16, 4),
                2,
                1,
            ),
        );
        let mut args = default_t2t_args();
        args.destination_origin = origin(2, 2, 0);
        args.copy_size = extent(1, 1, 1);
        assert_t2t_error(&test, same_texture, same_texture, args);

        let mut args = default_t2t_args();
        args.destination_origin = origin(0, 0, 2);
        args.copy_size = extent(1, 1, 2);
        assert_t2t_ok(&test, same_texture, same_texture, args);

        let mut args = default_t2t_args();
        args.destination_mip_level = 1;
        args.destination_origin = origin(1, 1, 0);
        args.copy_size = extent(1, 1, 1);
        assert_t2t_ok(&test, same_texture, same_texture, args);

        yawgpu::wgpuTextureRelease(same_texture);
        yawgpu::wgpuTextureRelease(multisampled_dst);
        yawgpu::wgpuTextureRelease(depth_dst);
        yawgpu::wgpuTextureRelease(depth_src);
        yawgpu::wgpuTextureRelease(srgb_dst);
        yawgpu::wgpuTextureRelease(uint_dst);
        yawgpu::wgpuTextureRelease(sampled);
        yawgpu::wgpuTextureRelease(copy_dst);
        yawgpu::wgpuTextureRelease(copy_src);
    }
}

#[derive(Clone, Copy)]
struct B2TArgs {
    layout: native::WGPUTexelCopyBufferLayout,
    mip_level: u32,
    origin: native::WGPUOrigin3D,
    aspect: native::WGPUTextureAspect,
    copy_size: native::WGPUExtent3D,
}

#[derive(Clone, Copy)]
struct T2BArgs {
    mip_level: u32,
    origin: native::WGPUOrigin3D,
    aspect: native::WGPUTextureAspect,
    layout: native::WGPUTexelCopyBufferLayout,
    copy_size: native::WGPUExtent3D,
}

#[derive(Clone, Copy)]
struct T2TArgs {
    source_mip_level: u32,
    source_origin: native::WGPUOrigin3D,
    source_aspect: native::WGPUTextureAspect,
    destination_mip_level: u32,
    destination_origin: native::WGPUOrigin3D,
    destination_aspect: native::WGPUTextureAspect,
    copy_size: native::WGPUExtent3D,
}

unsafe fn assert_b2t_ok(
    test: &ValidationTest,
    source: native::WGPUBuffer,
    destination: native::WGPUTexture,
    args: B2TArgs,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    copy_buffer_to_texture(encoder, source, destination, args);
    assert!(test.errors().is_empty());
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_b2t_error(
    test: &ValidationTest,
    source: native::WGPUBuffer,
    destination: native::WGPUTexture,
    args: B2TArgs,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    copy_buffer_to_texture(encoder, source, destination, args);
    assert!(test.errors().is_empty());
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_b2t_submit_error(
    test: &ValidationTest,
    source: native::WGPUBuffer,
    destination: native::WGPUTexture,
    args: B2TArgs,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    copy_buffer_to_texture(encoder, source, destination, args);
    assert!(test.errors().is_empty());
    let command_buffer = finish_ok(test, encoder);
    submit_error(test, command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_t2b_ok(
    test: &ValidationTest,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
    args: T2BArgs,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    copy_texture_to_buffer(encoder, source, destination, args);
    assert!(test.errors().is_empty());
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_t2b_error(
    test: &ValidationTest,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
    args: T2BArgs,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    copy_texture_to_buffer(encoder, source, destination, args);
    assert!(test.errors().is_empty());
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_t2t_ok(
    test: &ValidationTest,
    source: native::WGPUTexture,
    destination: native::WGPUTexture,
    args: T2TArgs,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    copy_texture_to_texture(encoder, source, destination, args);
    assert!(test.errors().is_empty());
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_t2t_error(
    test: &ValidationTest,
    source: native::WGPUTexture,
    destination: native::WGPUTexture,
    args: T2TArgs,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    copy_texture_to_texture(encoder, source, destination, args);
    assert!(test.errors().is_empty());
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn copy_buffer_to_texture(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUBuffer,
    destination: native::WGPUTexture,
    args: B2TArgs,
) {
    let source = native::WGPUTexelCopyBufferInfo {
        layout: args.layout,
        buffer: source,
    };
    let destination = native::WGPUTexelCopyTextureInfo {
        texture: destination,
        mipLevel: args.mip_level,
        origin: args.origin,
        aspect: args.aspect,
    };
    yawgpu::wgpuCommandEncoderCopyBufferToTexture(encoder, &source, &destination, &args.copy_size);
}

unsafe fn copy_texture_to_buffer(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
    args: T2BArgs,
) {
    let source = native::WGPUTexelCopyTextureInfo {
        texture: source,
        mipLevel: args.mip_level,
        origin: args.origin,
        aspect: args.aspect,
    };
    let destination = native::WGPUTexelCopyBufferInfo {
        layout: args.layout,
        buffer: destination,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &args.copy_size);
}

unsafe fn copy_texture_to_texture(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUTexture,
    args: T2TArgs,
) {
    let source = native::WGPUTexelCopyTextureInfo {
        texture: source,
        mipLevel: args.source_mip_level,
        origin: args.source_origin,
        aspect: args.source_aspect,
    };
    let destination = native::WGPUTexelCopyTextureInfo {
        texture: destination,
        mipLevel: args.destination_mip_level,
        origin: args.destination_origin,
        aspect: args.destination_aspect,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToTexture(encoder, &source, &destination, &args.copy_size);
}

unsafe fn create_encoder(test: &ValidationTest) -> native::WGPUCommandEncoder {
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
    assert!(!encoder.is_null());
    encoder
}

unsafe fn finish_ok(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    test.clear_errors();
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    command_buffer
}

unsafe fn finish_error(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    let mut command_buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        },
        None,
    );
    assert!(!command_buffer.is_null());
    command_buffer
}

unsafe fn submit_error(test: &ValidationTest, command_buffer: native::WGPUCommandBuffer) {
    let queue = yawgpu::wgpuDeviceGetQueue(test.device());
    test.assert_device_error_after(
        || {
            yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        },
        None,
    );
    yawgpu::wgpuQueueRelease(queue);
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: false.into(),
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

unsafe fn create_error_buffer(
    test: &ValidationTest,
    extra_usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let mut buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            buffer = create_buffer(
                test.device(),
                4,
                native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst | extra_usage,
            );
        },
        None,
    );
    assert!(!buffer.is_null());
    buffer
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    descriptor: native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

unsafe fn create_error_texture(
    test: &ValidationTest,
    descriptor: native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    let mut texture = std::ptr::null();
    test.assert_device_error_after(
        || {
            texture = yawgpu::wgpuDeviceCreateTexture(test.device(), &descriptor);
        },
        None,
    );
    assert!(!texture.is_null());
    texture
}

fn texture_descriptor(
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
    dimension: native::WGPUTextureDimension,
    size: native::WGPUExtent3D,
    mip_level_count: u32,
    sample_count: u32,
) -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension,
        size,
        format,
        mipLevelCount: mip_level_count,
        sampleCount: sample_count,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    }
}

fn default_b2t_args() -> B2TArgs {
    B2TArgs {
        layout: layout(0, 256, 4),
        mip_level: 0,
        origin: origin(0, 0, 0),
        aspect: native::WGPUTextureAspect_All,
        copy_size: extent(4, 4, 1),
    }
}

fn default_t2b_args() -> T2BArgs {
    T2BArgs {
        mip_level: 0,
        origin: origin(0, 0, 0),
        aspect: native::WGPUTextureAspect_All,
        layout: layout(0, 256, 4),
        copy_size: extent(4, 4, 1),
    }
}

fn default_t2t_args() -> T2TArgs {
    T2TArgs {
        source_mip_level: 0,
        source_origin: origin(0, 0, 0),
        source_aspect: native::WGPUTextureAspect_All,
        destination_mip_level: 0,
        destination_origin: origin(0, 0, 0),
        destination_aspect: native::WGPUTextureAspect_All,
        copy_size: extent(4, 4, 1),
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

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
