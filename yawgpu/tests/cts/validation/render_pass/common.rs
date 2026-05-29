use yawgpu::native;
use yawgpu_test::ValidationTest;

#[derive(Clone, Copy)]
pub struct ViewResource {
    pub texture: native::WGPUTexture,
    pub view: native::WGPUTextureView,
}

pub unsafe fn expect_render_pass(
    test: &ValidationTest,
    success: bool,
    descriptor: &native::WGPURenderPassDescriptor,
) {
    let encoder = unsafe { create_encoder(test.device()) };
    test.clear_errors();
    let pass = unsafe { yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, descriptor) };
    assert!(!pass.is_null());
    assert!(
        test.errors().is_empty(),
        "beginRenderPass should defer descriptor validation to finish: {:?}",
        test.errors()
    );
    unsafe {
        yawgpu::wgpuRenderPassEncoderEnd(pass);
    }
    if success {
        unsafe { finish_ok(test, encoder) };
    } else {
        unsafe { finish_error(test, encoder) };
    }
    unsafe {
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

pub unsafe fn expect_render_pass_with_commands<F>(
    test: &ValidationTest,
    success: bool,
    descriptor: &native::WGPURenderPassDescriptor,
    commands: F,
) where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = unsafe { create_encoder(test.device()) };
    test.clear_errors();
    let pass = unsafe { yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, descriptor) };
    assert!(!pass.is_null());
    assert!(
        test.errors().is_empty(),
        "beginRenderPass should defer descriptor validation to finish: {:?}",
        test.errors()
    );
    commands(pass);
    assert!(
        test.errors().is_empty(),
        "render pass commands should defer validation to finish: {:?}",
        test.errors()
    );
    unsafe {
        yawgpu::wgpuRenderPassEncoderEnd(pass);
    }
    if success {
        unsafe { finish_ok(test, encoder) };
    } else {
        unsafe { finish_error(test, encoder) };
    }
    unsafe {
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

pub unsafe fn expect_render_passes(
    test: &ValidationTest,
    success: bool,
    descriptors: &[native::WGPURenderPassDescriptor],
) {
    let encoder = unsafe { create_encoder(test.device()) };
    test.clear_errors();
    for descriptor in descriptors {
        let pass = unsafe { yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, descriptor) };
        assert!(!pass.is_null());
        assert!(test.errors().is_empty());
        unsafe {
            yawgpu::wgpuRenderPassEncoderEnd(pass);
            yawgpu::wgpuRenderPassEncoderRelease(pass);
        }
    }
    if success {
        unsafe { finish_ok(test, encoder) };
    } else {
        unsafe { finish_error(test, encoder) };
    }
    unsafe {
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

pub unsafe fn finish_ok(test: &ValidationTest, encoder: native::WGPUCommandEncoder) {
    test.clear_errors();
    let command_buffer = unsafe { yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null()) };
    assert!(!command_buffer.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    unsafe {
        yawgpu::wgpuCommandBufferRelease(command_buffer);
    }
}

pub unsafe fn finish_error(test: &ValidationTest, encoder: native::WGPUCommandEncoder) {
    let mut command_buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            command_buffer = unsafe { yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null()) };
        },
        None,
    );
    assert!(!command_buffer.is_null());
    unsafe {
        yawgpu::wgpuCommandBufferRelease(command_buffer);
    }
}

pub unsafe fn create_encoder(device: native::WGPUDevice) -> native::WGPUCommandEncoder {
    let encoder = unsafe { yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null()) };
    assert!(!encoder.is_null());
    encoder
}

pub unsafe fn create_query_set(
    device: native::WGPUDevice,
    query_type: native::WGPUQueryType,
    count: u32,
) -> native::WGPUQuerySet {
    let descriptor = native::WGPUQuerySetDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        type_: query_type,
        count,
    };
    let query_set = unsafe { yawgpu::wgpuDeviceCreateQuerySet(device, &descriptor) };
    assert!(!query_set.is_null());
    query_set
}

pub unsafe fn create_view(
    device: native::WGPUDevice,
    options: TextureOptions,
    view_descriptor: Option<native::WGPUTextureViewDescriptor>,
) -> ViewResource {
    let texture = unsafe { create_texture(device, options) };
    let view = unsafe {
        yawgpu::wgpuTextureCreateView(
            texture,
            view_descriptor
                .as_ref()
                .map_or(std::ptr::null(), std::ptr::from_ref),
        )
    };
    assert!(!view.is_null());
    ViewResource { texture, view }
}

pub unsafe fn create_texture(
    device: native::WGPUDevice,
    options: TextureOptions,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: options.usage,
        dimension: options.dimension,
        size: native::WGPUExtent3D {
            width: options.width,
            height: options.height,
            depthOrArrayLayers: options.depth_or_array_layers,
        },
        format: options.format,
        mipLevelCount: options.mip_level_count,
        sampleCount: options.sample_count,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = unsafe { yawgpu::wgpuDeviceCreateTexture(device, &descriptor) };
    assert!(!texture.is_null());
    texture
}

#[derive(Clone, Copy)]
pub struct TextureOptions {
    pub usage: native::WGPUTextureUsage,
    pub format: native::WGPUTextureFormat,
    pub dimension: native::WGPUTextureDimension,
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
    pub mip_level_count: u32,
    pub sample_count: u32,
}

impl TextureOptions {
    pub fn color() -> Self {
        Self {
            usage: native::WGPUTextureUsage_RenderAttachment,
            format: native::WGPUTextureFormat_RGBA8Unorm,
            dimension: native::WGPUTextureDimension_2D,
            width: 16,
            height: 16,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
        }
    }

    pub fn depth_stencil() -> Self {
        Self {
            format: native::WGPUTextureFormat_Depth24PlusStencil8,
            ..Self::color()
        }
    }
}

pub unsafe fn release_view(resource: ViewResource) {
    unsafe {
        yawgpu::wgpuTextureViewRelease(resource.view);
        yawgpu::wgpuTextureRelease(resource.texture);
    }
}

pub fn render_pass_descriptor(
    color_attachments: &[native::WGPURenderPassColorAttachment],
    depth_stencil_attachment: Option<&native::WGPURenderPassDepthStencilAttachment>,
) -> native::WGPURenderPassDescriptor {
    native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: color_attachments.len(),
        colorAttachments: color_attachments.as_ptr(),
        depthStencilAttachment: depth_stencil_attachment
            .map_or(std::ptr::null(), std::ptr::from_ref),
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    }
}

pub fn color_attachment(view: native::WGPUTextureView) -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: color(0.0, 0.0, 0.0, 0.0),
    }
}

pub fn color_attachment_with_resolve(
    view: native::WGPUTextureView,
    resolve_target: native::WGPUTextureView,
) -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        resolveTarget: resolve_target,
        ..color_attachment(view)
    }
}

pub fn sparse_color_attachment() -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        view: std::ptr::null(),
        ..color_attachment(std::ptr::null())
    }
}

pub fn depth_stencil_attachment(
    view: native::WGPUTextureView,
) -> native::WGPURenderPassDepthStencilAttachment {
    native::WGPURenderPassDepthStencilAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthLoadOp: native::WGPULoadOp_Clear,
        depthStoreOp: native::WGPUStoreOp_Store,
        depthClearValue: 1.0,
        depthReadOnly: 0,
        stencilLoadOp: native::WGPULoadOp_Clear,
        stencilStoreOp: native::WGPUStoreOp_Store,
        stencilClearValue: 0,
        stencilReadOnly: 0,
    }
}

pub fn view_descriptor(
    dimension: native::WGPUTextureViewDimension,
    base_mip_level: u32,
    mip_level_count: u32,
    base_array_layer: u32,
    array_layer_count: u32,
    format: native::WGPUTextureFormat,
) -> native::WGPUTextureViewDescriptor {
    native::WGPUTextureViewDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        format,
        dimension,
        baseMipLevel: base_mip_level,
        mipLevelCount: mip_level_count,
        baseArrayLayer: base_array_layer,
        arrayLayerCount: array_layer_count,
        aspect: native::WGPUTextureAspect_All,
        usage: native::WGPUTextureUsage_None,
    }
}

pub fn color(r: f64, g: f64, b: f64, a: f64) -> native::WGPUColor {
    native::WGPUColor { r, g, b, a }
}

pub fn timestamp_writes(
    query_set: native::WGPUQuerySet,
    beginning_index: u32,
    end_index: u32,
) -> native::WGPUPassTimestampWrites {
    native::WGPUPassTimestampWrites {
        nextInChain: std::ptr::null_mut(),
        querySet: query_set,
        beginningOfPassWriteIndex: beginning_index,
        endOfPassWriteIndex: end_index,
    }
}

pub fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
