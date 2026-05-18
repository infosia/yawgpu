use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn command_after_successful_finish_is_an_immediate_error() {
    let test = ValidationTest::new();
    unsafe {
        let encoder = create_encoder(&test);
        let command_buffer = finish_ok(&test, encoder);
        assert!(!command_buffer.is_null());

        let mut pass = std::ptr::null();
        test.assert_device_error_after(
            || {
                pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
            },
            None,
        );
        assert!(!pass.is_null());

        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

#[test]
fn finish_with_open_pass_is_an_error_command_buffer() {
    let test = ValidationTest::new();
    unsafe {
        let render_encoder = create_encoder(&test);
        let render_descriptor = render_pass_descriptor();
        let render_pass =
            yawgpu::wgpuCommandEncoderBeginRenderPass(render_encoder, &render_descriptor);
        assert!(!render_pass.is_null());
        let render_buffer = finish_error(&test, render_encoder);
        assert!(!render_buffer.is_null());

        yawgpu::wgpuCommandBufferRelease(render_buffer);
        yawgpu::wgpuRenderPassEncoderRelease(render_pass);
        yawgpu::wgpuCommandEncoderRelease(render_encoder);

        let compute_encoder = create_encoder(&test);
        let compute_pass =
            yawgpu::wgpuCommandEncoderBeginComputePass(compute_encoder, std::ptr::null());
        assert!(!compute_pass.is_null());
        let compute_buffer = finish_error(&test, compute_encoder);
        assert!(!compute_buffer.is_null());

        yawgpu::wgpuCommandBufferRelease(compute_buffer);
        yawgpu::wgpuComputePassEncoderRelease(compute_pass);
        yawgpu::wgpuCommandEncoderRelease(compute_encoder);
    }
}

#[test]
fn pass_end_twice_and_command_after_end_are_immediate_errors() {
    let test = ValidationTest::new();
    unsafe {
        let encoder = create_encoder(&test);
        let descriptor = render_pass_descriptor();
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        assert!(!pass.is_null());

        yawgpu::wgpuRenderPassEncoderEnd(pass);
        test.assert_device_error_after(
            || {
                yawgpu::wgpuRenderPassEncoderEnd(pass);
            },
            None,
        );
        test.assert_device_error_after(
            || {
                yawgpu::wgpuRenderPassEncoderInsertDebugMarker(pass, empty_string_view());
            },
            None,
        );

        let command_buffer = finish_error(&test, encoder);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

#[test]
fn beginning_a_pass_while_another_is_open_is_deferred_to_finish() {
    let test = ValidationTest::new();
    unsafe {
        let encoder = create_encoder(&test);
        let compute_pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        assert!(!compute_pass.is_null());
        let render_descriptor = render_pass_descriptor();

        test.clear_errors();
        let render_pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &render_descriptor);
        assert!(!render_pass.is_null());
        assert!(test.errors().is_empty());

        yawgpu::wgpuComputePassEncoderEnd(compute_pass);
        let command_buffer = finish_error(&test, encoder);

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuRenderPassEncoderRelease(render_pass);
        yawgpu::wgpuComputePassEncoderRelease(compute_pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

#[test]
fn begin_compute_pass_accepts_null_descriptor() {
    let test = ValidationTest::new();
    unsafe {
        let encoder = create_encoder(&test);
        test.clear_errors();
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        assert!(!pass.is_null());
        assert!(test.errors().is_empty());

        yawgpu::wgpuComputePassEncoderEnd(pass);
        let command_buffer = finish_ok(&test, encoder);

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

#[test]
fn encoder_debug_group_balance_is_checked_at_finish() {
    let test = ValidationTest::new();
    unsafe {
        let balanced = create_encoder(&test);
        yawgpu::wgpuCommandEncoderPushDebugGroup(balanced, empty_string_view());
        yawgpu::wgpuCommandEncoderPopDebugGroup(balanced);
        let balanced_buffer = finish_ok(&test, balanced);
        yawgpu::wgpuCommandBufferRelease(balanced_buffer);
        yawgpu::wgpuCommandEncoderRelease(balanced);

        let unbalanced = create_encoder(&test);
        yawgpu::wgpuCommandEncoderPushDebugGroup(unbalanced, empty_string_view());
        let unbalanced_buffer = finish_error(&test, unbalanced);
        yawgpu::wgpuCommandBufferRelease(unbalanced_buffer);
        yawgpu::wgpuCommandEncoderRelease(unbalanced);

        let pop_without_push = create_encoder(&test);
        test.clear_errors();
        yawgpu::wgpuCommandEncoderPopDebugGroup(pop_without_push);
        assert!(test.errors().is_empty());
        let pop_buffer = finish_error(&test, pop_without_push);
        yawgpu::wgpuCommandBufferRelease(pop_buffer);
        yawgpu::wgpuCommandEncoderRelease(pop_without_push);
    }
}

#[test]
fn pass_after_parent_finish_is_an_immediate_error() {
    let test = ValidationTest::new();
    unsafe {
        let encoder = create_encoder(&test);
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderEnd(pass);
        let command_buffer = finish_ok(&test, encoder);

        test.assert_device_error_after(
            || {
                yawgpu::wgpuComputePassEncoderInsertDebugMarker(pass, empty_string_view());
            },
            None,
        );

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

#[test]
fn command_encoder_from_destroyed_device_is_safe_error() {
    let test = ValidationTest::new();
    unsafe {
        yawgpu::wgpuDeviceDestroy(test.device());
        let encoder = create_encoder(&test);
        let command_buffer = finish_error(&test, encoder);
        assert!(!command_buffer.is_null());

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

#[test]
fn command_objects_add_ref_release_are_safe() {
    let test = ValidationTest::new();
    unsafe {
        let encoder = create_encoder(&test);
        yawgpu::wgpuCommandEncoderAddRef(encoder);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let compute_pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderAddRef(compute_pass);
        yawgpu::wgpuComputePassEncoderRelease(compute_pass);
        yawgpu::wgpuComputePassEncoderEnd(compute_pass);

        let command_buffer = finish_ok(&test, encoder);
        yawgpu::wgpuCommandBufferAddRef(command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuComputePassEncoderRelease(compute_pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let render_encoder = create_encoder(&test);
        let render_texture = create_render_texture(test.device());
        let render_view = yawgpu::wgpuTextureCreateView(render_texture, std::ptr::null());
        assert!(!render_view.is_null());
        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: render_view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Load,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            },
        };
        let descriptor = native::WGPURenderPassDescriptor {
            colorAttachmentCount: 1,
            colorAttachments: &color_attachment,
            ..render_pass_descriptor()
        };
        let render_pass = yawgpu::wgpuCommandEncoderBeginRenderPass(render_encoder, &descriptor);
        yawgpu::wgpuRenderPassEncoderAddRef(render_pass);
        yawgpu::wgpuRenderPassEncoderRelease(render_pass);
        yawgpu::wgpuRenderPassEncoderEnd(render_pass);
        let render_buffer = finish_ok(&test, render_encoder);

        yawgpu::wgpuCommandBufferRelease(render_buffer);
        yawgpu::wgpuRenderPassEncoderRelease(render_pass);
        yawgpu::wgpuTextureViewRelease(render_view);
        yawgpu::wgpuTextureRelease(render_texture);
        yawgpu::wgpuCommandEncoderRelease(render_encoder);

        let error_encoder = create_encoder(&test);
        let error_pass =
            yawgpu::wgpuCommandEncoderBeginComputePass(error_encoder, std::ptr::null());
        let error_buffer = finish_error(&test, error_encoder);
        yawgpu::wgpuCommandBufferAddRef(error_buffer);
        yawgpu::wgpuCommandBufferRelease(error_buffer);

        yawgpu::wgpuCommandBufferRelease(error_buffer);
        yawgpu::wgpuComputePassEncoderRelease(error_pass);
        yawgpu::wgpuCommandEncoderRelease(error_encoder);
    }
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

fn render_pass_descriptor() -> native::WGPURenderPassDescriptor {
    native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: 0,
        colorAttachments: std::ptr::null(),
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    }
}

unsafe fn create_render_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 1,
            height: 1,
            depthOrArrayLayers: 1,
        },
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
