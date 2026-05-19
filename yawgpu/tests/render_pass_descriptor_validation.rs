use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn empty_depth_only_color_only_and_occlusion_query_fields_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 1),
            1,
            1,
            None,
        );
        let depth = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_Depth24Plus,
            extent(4, 4, 1),
            1,
            1,
            None,
        );

        assert_render_pass_error(&test, &render_pass_descriptor(&[], None));

        assert_render_pass_ok(
            &test,
            &render_pass_descriptor(&[make_color_attachment(color.view)], None),
        );

        let depth_attachment = make_depth_attachment(depth.view);
        assert_render_pass_ok(&test, &render_pass_descriptor(&[], Some(&depth_attachment)));

        let query_set_descriptor = native::WGPUQuerySetDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            type_: native::WGPUQueryType_Occlusion,
            count: 1,
        };
        let query_set = yawgpu::wgpuDeviceCreateQuerySet(test.device(), &query_set_descriptor);
        assert!(!query_set.is_null());
        let mut descriptor = render_pass_descriptor(&[make_color_attachment(color.view)], None);
        descriptor.occlusionQuerySet = query_set;
        assert_render_pass_ok(&test, &descriptor);

        yawgpu::wgpuQuerySetRelease(query_set);
        release_view(color);
        release_view(depth);
    }
}

#[test]
fn color_count_usage_sparse_and_format_rules_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let mut limits = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuDeviceGetLimits(test.device(), &mut limits),
            native::WGPUStatus_Success
        );

        let color = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 1),
            1,
            1,
            None,
        );
        let sampled = create_view(
            test.device(),
            native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 1),
            1,
            1,
            None,
        );
        let depth = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_Depth24Plus,
            extent(4, 4, 1),
            1,
            1,
            None,
        );
        let non_renderable = create_view(
            test.device(),
            native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureFormat_R8Snorm,
            extent(4, 4, 1),
            1,
            1,
            None,
        );

        let too_many: Vec<_> = (0..=limits.maxColorAttachments)
            .map(|_| make_color_attachment(color.view))
            .collect();
        assert_render_pass_error(&test, &render_pass_descriptor(&too_many, None));

        assert_render_pass_error(
            &test,
            &render_pass_descriptor(&[make_color_attachment(sampled.view)], None),
        );
        assert_render_pass_error(
            &test,
            &render_pass_descriptor(&[make_color_attachment(non_renderable.view)], None),
        );
        assert_render_pass_error(
            &test,
            &render_pass_descriptor(&[make_color_attachment(depth.view)], None),
        );

        let sparse = [
            make_sparse_color_attachment(),
            make_color_attachment(color.view),
            make_sparse_color_attachment(),
        ];
        assert_render_pass_ok(&test, &render_pass_descriptor(&sparse, None));

        let all_null = [
            make_sparse_color_attachment(),
            make_sparse_color_attachment(),
        ];
        assert_render_pass_error(&test, &render_pass_descriptor(&all_null, None));

        release_view(non_renderable);
        release_view(depth);
        release_view(sampled);
        release_view(color);
    }
}

#[test]
fn depth_stencil_size_layer_and_sample_rules_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 1),
            1,
            1,
            None,
        );
        let color_large = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(8, 8, 1),
            1,
            1,
            None,
        );
        let color_array_view = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 2),
            1,
            1,
            Some(view_descriptor(
                native::WGPUTextureViewDimension_2DArray,
                0,
                1,
                0,
                2,
                native::WGPUTextureFormat_Undefined,
            )),
        );
        let depth = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_Depth24PlusStencil8,
            extent(4, 4, 1),
            1,
            1,
            None,
        );
        let color_as_depth = make_depth_attachment(color.view);
        assert_render_pass_error(&test, &render_pass_descriptor(&[], Some(&color_as_depth)));

        assert_render_pass_error(
            &test,
            &render_pass_descriptor(
                &[
                    make_color_attachment(color.view),
                    make_color_attachment(color_large.view),
                ],
                None,
            ),
        );

        let color_array_attachment = make_color_attachment(color_array_view.view);
        assert_render_pass_error(
            &test,
            &render_pass_descriptor(&[color_array_attachment], None),
        );

        assert_render_pass_ok(
            &test,
            &render_pass_descriptor(
                &[make_color_attachment(color.view)],
                Some(&make_depth_attachment(depth.view)),
            ),
        );

        let color_msaa = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 1),
            1,
            4,
            None,
        );
        assert_render_pass_error(
            &test,
            &render_pass_descriptor(
                &[make_color_attachment(color_msaa.view)],
                Some(&make_depth_attachment(depth.view)),
            ),
        );

        release_view(color_msaa);
        release_view(depth);
        release_view(color_array_view);
        release_view(color_large);
        release_view(color);
    }
}

#[test]
fn resolve_target_rules_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let color_msaa = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 1),
            1,
            4,
            None,
        );
        let resolve = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 1),
            1,
            1,
            None,
        );
        let resolve_msaa = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 1),
            1,
            4,
            None,
        );
        let resolve_format = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_BGRA8Unorm,
            extent(4, 4, 1),
            1,
            1,
            None,
        );
        let resolve_large = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(8, 8, 1),
            1,
            1,
            None,
        );
        let resolve_sampled = create_view(
            test.device(),
            native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 1),
            1,
            1,
            None,
        );
        let resolve_array = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 2),
            1,
            1,
            Some(view_descriptor(
                native::WGPUTextureViewDimension_2DArray,
                0,
                1,
                0,
                2,
                native::WGPUTextureFormat_Undefined,
            )),
        );

        assert_render_pass_ok(
            &test,
            &render_pass_descriptor(
                &[make_color_attachment_with_resolve(
                    color_msaa.view,
                    resolve.view,
                )],
                None,
            ),
        );
        assert_render_pass_error(
            &test,
            &render_pass_descriptor(
                &[make_color_attachment_with_resolve(
                    color_msaa.view,
                    resolve_msaa.view,
                )],
                None,
            ),
        );
        assert_render_pass_error(
            &test,
            &render_pass_descriptor(
                &[make_color_attachment_with_resolve(
                    color_msaa.view,
                    resolve_format.view,
                )],
                None,
            ),
        );
        assert_render_pass_error(
            &test,
            &render_pass_descriptor(
                &[make_color_attachment_with_resolve(
                    color_msaa.view,
                    resolve_large.view,
                )],
                None,
            ),
        );
        assert_render_pass_error(
            &test,
            &render_pass_descriptor(
                &[make_color_attachment_with_resolve(
                    color_msaa.view,
                    resolve_sampled.view,
                )],
                None,
            ),
        );
        assert_render_pass_error(
            &test,
            &render_pass_descriptor(
                &[make_color_attachment_with_resolve(
                    color_msaa.view,
                    resolve_array.view,
                )],
                None,
            ),
        );

        release_view(resolve_array);
        release_view(resolve_sampled);
        release_view(resolve_large);
        release_view(resolve_format);
        release_view(resolve_msaa);
        release_view(resolve);
        release_view(color_msaa);
    }
}

#[test]
fn load_store_and_clear_values_are_validated() {
    let test = ValidationTest::new();
    unsafe {
        let color = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
            extent(4, 4, 1),
            1,
            1,
            None,
        );
        let depth_stencil = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_Depth24PlusStencil8,
            extent(4, 4, 1),
            1,
            1,
            None,
        );
        let depth = create_view(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_Depth24Plus,
            extent(4, 4, 1),
            1,
            1,
            None,
        );

        let mut color_attachment = make_color_attachment(color.view);
        color_attachment.loadOp = native::WGPULoadOp_Undefined;
        assert_render_pass_error(&test, &render_pass_descriptor(&[color_attachment], None));

        let mut color_attachment = make_color_attachment(color.view);
        color_attachment.storeOp = native::WGPUStoreOp_Undefined;
        assert_render_pass_error(&test, &render_pass_descriptor(&[color_attachment], None));

        let mut color_attachment = make_color_attachment(color.view);
        color_attachment.loadOp = native::WGPULoadOp_Clear;
        color_attachment.clearValue.r = f64::NAN;
        assert_render_pass_error(&test, &render_pass_descriptor(&[color_attachment], None));

        let mut color_attachment = make_color_attachment(color.view);
        color_attachment.loadOp = native::WGPULoadOp_Clear;
        color_attachment.clearValue.g = f64::INFINITY;
        assert_render_pass_error(&test, &render_pass_descriptor(&[color_attachment], None));

        let mut depth_attachment = make_depth_attachment(depth.view);
        depth_attachment.depthLoadOp = native::WGPULoadOp_Undefined;
        assert_render_pass_error(&test, &render_pass_descriptor(&[], Some(&depth_attachment)));

        let mut depth_attachment = make_depth_attachment(depth.view);
        depth_attachment.depthStoreOp = native::WGPUStoreOp_Undefined;
        assert_render_pass_error(&test, &render_pass_descriptor(&[], Some(&depth_attachment)));

        let mut depth_attachment = make_depth_attachment(depth.view);
        depth_attachment.depthLoadOp = native::WGPULoadOp_Clear;
        depth_attachment.depthClearValue = 2.0;
        assert_render_pass_error(&test, &render_pass_descriptor(&[], Some(&depth_attachment)));

        let mut depth_attachment = make_depth_attachment(depth.view);
        depth_attachment.depthLoadOp = native::WGPULoadOp_Clear;
        depth_attachment.depthClearValue = 0.5;
        depth_attachment.stencilLoadOp = native::WGPULoadOp_Undefined;
        depth_attachment.stencilStoreOp = native::WGPUStoreOp_Undefined;
        assert_render_pass_ok(&test, &render_pass_descriptor(&[], Some(&depth_attachment)));

        let mut depth_stencil_attachment = make_depth_attachment(depth_stencil.view);
        depth_stencil_attachment.depthStoreOp = native::WGPUStoreOp_Discard;
        depth_stencil_attachment.stencilStoreOp = native::WGPUStoreOp_Store;
        assert_render_pass_ok(
            &test,
            &render_pass_descriptor(&[], Some(&depth_stencil_attachment)),
        );

        release_view(depth);
        release_view(depth_stencil);
        release_view(color);
    }
}

#[derive(Clone, Copy)]
struct ViewResource {
    texture: native::WGPUTexture,
    view: native::WGPUTextureView,
}

unsafe fn assert_render_pass_ok(
    test: &ValidationTest,
    descriptor: &native::WGPURenderPassDescriptor,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, descriptor);
    assert!(!pass.is_null());
    assert!(test.errors().is_empty());
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_render_pass_error(
    test: &ValidationTest,
    descriptor: &native::WGPURenderPassDescriptor,
) {
    let encoder = create_encoder(test);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, descriptor);
    assert!(!pass.is_null());
    assert!(test.errors().is_empty());
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
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

unsafe fn create_view(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
    size: native::WGPUExtent3D,
    mip_level_count: u32,
    sample_count: u32,
    view_descriptor: Option<native::WGPUTextureViewDescriptor>,
) -> ViewResource {
    let texture = create_texture(device, usage, format, size, mip_level_count, sample_count);
    let view = yawgpu::wgpuTextureCreateView(
        texture,
        view_descriptor
            .as_ref()
            .map_or(std::ptr::null(), std::ptr::from_ref),
    );
    assert!(!view.is_null());
    ViewResource { texture, view }
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
    size: native::WGPUExtent3D,
    mip_level_count: u32,
    sample_count: u32,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size,
        format,
        mipLevelCount: mip_level_count,
        sampleCount: sample_count,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

unsafe fn release_view(resource: ViewResource) {
    yawgpu::wgpuTextureViewRelease(resource.view);
    yawgpu::wgpuTextureRelease(resource.texture);
}

fn render_pass_descriptor(
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

fn make_color_attachment(view: native::WGPUTextureView) -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Load,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: color(0.0, 0.0, 0.0, 0.0),
    }
}

fn make_color_attachment_with_resolve(
    view: native::WGPUTextureView,
    resolve_target: native::WGPUTextureView,
) -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        resolveTarget: resolve_target,
        ..make_color_attachment(view)
    }
}

fn make_sparse_color_attachment() -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        view: std::ptr::null(),
        ..make_color_attachment(std::ptr::null())
    }
}

fn make_depth_attachment(
    view: native::WGPUTextureView,
) -> native::WGPURenderPassDepthStencilAttachment {
    native::WGPURenderPassDepthStencilAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthLoadOp: native::WGPULoadOp_Load,
        depthStoreOp: native::WGPUStoreOp_Store,
        depthClearValue: 1.0,
        depthReadOnly: false.into(),
        stencilLoadOp: native::WGPULoadOp_Load,
        stencilStoreOp: native::WGPUStoreOp_Store,
        stencilClearValue: 0,
        stencilReadOnly: false.into(),
    }
}

fn view_descriptor(
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

fn color(r: f64, g: f64, b: f64, a: f64) -> native::WGPUColor {
    native::WGPUColor { r, g, b, a }
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
