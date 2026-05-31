use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[test]
fn base() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(test.device(), native::WGPUTextureUsage_TextureBinding);
        test.expect_no_validation_error(|| yawgpu::wgpuTextureDestroy(texture));
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn twice() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(test.device(), native::WGPUTextureUsage_TextureBinding);
        test.expect_no_validation_error(|| {
            yawgpu::wgpuTextureDestroy(texture);
            yawgpu::wgpuTextureDestroy(texture);
        });
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn invalid_texture() {
    let test = ValidationTest::new();
    unsafe {
        let mut texture = std::ptr::null();
        assert_device_error!({
            texture = yawgpu::wgpuDeviceCreateTexture(
                test.device(),
                &native::WGPUTextureDescriptor {
                    usage: native::WGPUTextureUsage_None,
                    ..texture_descriptor(native::WGPUTextureUsage_TextureBinding)
                },
            );
        });
        assert!(!texture.is_null());
        test.expect_no_validation_error(|| yawgpu::wgpuTextureDestroy(texture));
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn submit_a_destroyed_texture_as_attachment() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        assert!(!view.is_null());

        let attachment = color_attachment(view);
        let attachments = [attachment];
        let pass_descriptor = render_pass_descriptor(&attachments);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        assert!(!command_buffer.is_null());

        yawgpu::wgpuTextureDestroy(texture);
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        test.assert_device_error_after(
            || yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer),
            Some("destroyed"),
        );

        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &texture_descriptor(usage));
    assert!(!texture.is_null());
    texture
}

fn texture_descriptor(usage: native::WGPUTextureUsage) -> native::WGPUTextureDescriptor {
    native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
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

fn render_pass_descriptor(
    attachments: &[native::WGPURenderPassColorAttachment],
) -> native::WGPURenderPassDescriptor {
    native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: attachments.len(),
        colorAttachments: attachments.as_ptr(),
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    }
}

fn color_attachment(view: native::WGPUTextureView) -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
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
    }
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
