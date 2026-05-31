use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn render_bundle_encoder_descriptor_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        assert_bundle_encoder_error(
            &test,
            bundle_descriptor(&[], native::WGPUTextureFormat_Undefined, 1),
        );
        let color_formats = [];
        let invalid_descriptor =
            bundle_descriptor(&color_formats, native::WGPUTextureFormat_Undefined, 1);
        let mut invalid_encoder = std::ptr::null();
        test.assert_device_error_after(
            || {
                invalid_encoder =
                    yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &invalid_descriptor);
            },
            None,
        );
        assert!(!invalid_encoder.is_null());
        test.clear_errors();
        let error_bundle = yawgpu::wgpuRenderBundleEncoderFinish(invalid_encoder, std::ptr::null());
        assert!(!error_bundle.is_null());
        assert!(
            test.errors().is_empty(),
            "invalid descriptor finish must not redispatch: {:?}",
            test.errors()
        );
        yawgpu::wgpuRenderBundleRelease(error_bundle);
        yawgpu::wgpuRenderBundleEncoderRelease(invalid_encoder);

        let limits = device_limits(test.device());
        let too_many =
            vec![native::WGPUTextureFormat_RGBA8Unorm; limits.maxColorAttachments as usize + 1];
        assert_bundle_encoder_error(
            &test,
            bundle_descriptor(&too_many, native::WGPUTextureFormat_Undefined, 1),
        );

        assert_bundle_encoder_error(
            &test,
            bundle_descriptor(
                &[native::WGPUTextureFormat_R8Snorm],
                native::WGPUTextureFormat_Undefined,
                1,
            ),
        );
        assert_bundle_encoder_error(
            &test,
            bundle_descriptor(
                &[native::WGPUTextureFormat_RGBA8Unorm],
                native::WGPUTextureFormat_Undefined,
                2,
            ),
        );
        let encoder = create_bundle_encoder_ok(
            &test,
            bundle_descriptor(
                &[native::WGPUTextureFormat_RGBA8Unorm],
                native::WGPUTextureFormat_Undefined,
                1,
            ),
        );
        yawgpu::wgpuRenderBundleEncoderRelease(encoder);
    }
}

#[test]
fn bundle_pipeline_attachment_compatibility_is_validated_at_finish() {
    let test = ValidationTest::new();
    unsafe {
        let rgba8 = [native::WGPUTextureFormat_RGBA8Unorm];
        let desc = bundle_descriptor(&rgba8, native::WGPUTextureFormat_Undefined, 1);
        let matching = create_render_pipeline(
            &test,
            render_no_input(),
            None,
            &[],
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
        );
        let mismatched_format = create_render_pipeline(
            &test,
            render_no_input(),
            None,
            &[],
            native::WGPUTextureFormat_BGRA8Unorm,
            1,
        );
        let mismatched_sample = create_render_pipeline(
            &test,
            render_no_input(),
            None,
            &[],
            native::WGPUTextureFormat_RGBA8Unorm,
            4,
        );

        assert_bundle_finish_ok(&test, desc, |encoder| {
            yawgpu::wgpuRenderBundleEncoderSetPipeline(encoder, matching);
        });
        assert_bundle_finish_error(&test, desc, |encoder| {
            yawgpu::wgpuRenderBundleEncoderSetPipeline(encoder, mismatched_format);
        });
        assert_bundle_finish_error(&test, desc, |encoder| {
            yawgpu::wgpuRenderBundleEncoderSetPipeline(encoder, mismatched_sample);
        });

        yawgpu::wgpuRenderPipelineRelease(mismatched_sample);
        yawgpu::wgpuRenderPipelineRelease(mismatched_format);
        yawgpu::wgpuRenderPipelineRelease(matching);
    }
}

#[test]
fn valid_bundle_records_draw_commands() {
    let test = ValidationTest::new();
    unsafe {
        let uniform = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 1024);
        let vertex = create_buffer(test.device(), native::WGPUBufferUsage_Vertex, 256);
        let index = create_buffer(test.device(), native::WGPUBufferUsage_Index, 256);
        let indirect = create_buffer(test.device(), native::WGPUBufferUsage_Indirect, 64);
        let bgl = create_bind_group_layout(
            test.device(),
            &[uniform_layout(0, native::WGPUShaderStage_Vertex)],
        );
        let bind_group =
            create_bind_group(test.device(), bgl, &[buffer_binding(0, uniform, 0, 256)]);
        let layout = create_pipeline_layout(test.device(), &[bgl]);
        let attr = [vertex_attribute(native::WGPUVertexFormat_Float32x2, 0, 0)];
        let vb = [vertex_buffer(8, &attr)];
        let pipeline = create_render_pipeline(
            &test,
            render_uniform_vertex_input(),
            Some(layout),
            &vb,
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
        );
        let rgba8 = [native::WGPUTextureFormat_RGBA8Unorm];
        let desc = bundle_descriptor(&rgba8, native::WGPUTextureFormat_Undefined, 1);

        let bundle = assert_bundle_finish_ok(&test, desc, |encoder| {
            yawgpu::wgpuRenderBundleEncoderSetPipeline(encoder, pipeline);
            yawgpu::wgpuRenderBundleEncoderSetBindGroup(
                encoder,
                0,
                bind_group,
                0,
                std::ptr::null(),
            );
            yawgpu::wgpuRenderBundleEncoderSetVertexBuffer(encoder, 0, vertex, 0, 256);
            yawgpu::wgpuRenderBundleEncoderSetIndexBuffer(
                encoder,
                index,
                native::WGPUIndexFormat_Uint16,
                0,
                256,
            );
            yawgpu::wgpuRenderBundleEncoderDraw(encoder, 3, 1, 0, 0);
            yawgpu::wgpuRenderBundleEncoderDrawIndexed(encoder, 3, 1, 0, 0, 0);
            yawgpu::wgpuRenderBundleEncoderDrawIndirect(encoder, indirect, 0);
        });

        yawgpu::wgpuRenderBundleRelease(bundle);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuBufferRelease(indirect);
        yawgpu::wgpuBufferRelease(index);
        yawgpu::wgpuBufferRelease(vertex);
        yawgpu::wgpuBufferRelease(uniform);
    }
}

#[test]
fn render_bundle_encoder_finish_state_is_validated() {
    let test = ValidationTest::new();
    unsafe {
        let rgba8 = [native::WGPUTextureFormat_RGBA8Unorm];
        let desc = bundle_descriptor(&rgba8, native::WGPUTextureFormat_Undefined, 1);
        let encoder = create_bundle_encoder_ok(&test, desc);
        let bundle = finish_bundle_ok(&test, encoder);
        let second = finish_bundle_error(&test, encoder);
        yawgpu::wgpuRenderBundleRelease(second);
        yawgpu::wgpuRenderBundleRelease(bundle);

        let encoder = create_bundle_encoder_ok(&test, desc);
        let bundle = finish_bundle_ok(&test, encoder);
        test.assert_device_error_after(
            || {
                yawgpu::wgpuRenderBundleEncoderDraw(encoder, 0, 0, 0, 0);
            },
            None,
        );
        yawgpu::wgpuRenderBundleRelease(bundle);
        yawgpu::wgpuRenderBundleEncoderRelease(encoder);
    }
}

#[test]
fn execute_bundles_validates_signature_error_bundles_state_clear_and_multi_execute() {
    let test = ValidationTest::new();
    unsafe {
        let rgba8 = [native::WGPUTextureFormat_RGBA8Unorm];
        let bgra8 = [native::WGPUTextureFormat_BGRA8Unorm];
        let matching_desc = bundle_descriptor(&rgba8, native::WGPUTextureFormat_Undefined, 1);
        let mismatched_desc = bundle_descriptor(&bgra8, native::WGPUTextureFormat_Undefined, 1);
        let sample_mismatch_desc =
            bundle_descriptor(&rgba8, native::WGPUTextureFormat_Undefined, 4);
        let pipeline = create_render_pipeline(
            &test,
            render_no_input(),
            None,
            &[],
            native::WGPUTextureFormat_RGBA8Unorm,
            1,
        );
        let bundle_a = assert_bundle_finish_ok(&test, matching_desc, |_| {});
        let bundle_b = assert_bundle_finish_ok(&test, matching_desc, |_| {});
        let mismatched = assert_bundle_finish_ok(&test, mismatched_desc, |_| {});
        let sample_mismatched = assert_bundle_finish_ok(&test, sample_mismatch_desc, |_| {});
        let error_bundle = assert_bundle_finish_error(&test, matching_desc, |encoder| {
            yawgpu::wgpuRenderBundleEncoderDraw(encoder, 1, 1, 0, 0);
        });

        assert_render_pass_ok(&test, 1, |pass| {
            yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, [bundle_a].as_ptr());
        });
        assert_render_pass_error(&test, 1, |pass| {
            yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, [mismatched].as_ptr());
        });
        assert_render_pass_error(&test, 1, |pass| {
            yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, [sample_mismatched].as_ptr());
        });
        assert_render_pass_error(&test, 1, |pass| {
            yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, [error_bundle].as_ptr());
        });
        assert_render_pass_error(&test, 1, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, [bundle_a].as_ptr());
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
        assert_render_pass_ok(&test, 1, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, [bundle_a].as_ptr());
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        });
        assert_render_pass_ok(&test, 1, |pass| {
            yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 2, [bundle_a, bundle_b].as_ptr());
            yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, [bundle_a].as_ptr());
        });

        yawgpu::wgpuRenderBundleRelease(error_bundle);
        yawgpu::wgpuRenderBundleRelease(sample_mismatched);
        yawgpu::wgpuRenderBundleRelease(mismatched);
        yawgpu::wgpuRenderBundleRelease(bundle_b);
        yawgpu::wgpuRenderBundleRelease(bundle_a);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
    }
}

unsafe fn assert_bundle_encoder_error(
    test: &ValidationTest,
    descriptor: native::WGPURenderBundleEncoderDescriptor,
) {
    let mut encoder = std::ptr::null();
    test.assert_device_error_after(
        || {
            encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
        },
        None,
    );
    assert!(!encoder.is_null());
    yawgpu::wgpuRenderBundleEncoderRelease(encoder);
}

unsafe fn create_bundle_encoder_ok(
    test: &ValidationTest,
    descriptor: native::WGPURenderBundleEncoderDescriptor,
) -> native::WGPURenderBundleEncoder {
    test.clear_errors();
    let encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
    assert!(!encoder.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    encoder
}

unsafe fn assert_bundle_finish_ok<F>(
    test: &ValidationTest,
    descriptor: native::WGPURenderBundleEncoderDescriptor,
    commands: F,
) -> native::WGPURenderBundle
where
    F: FnOnce(native::WGPURenderBundleEncoder),
{
    let encoder = create_bundle_encoder_ok(test, descriptor);
    commands(encoder);
    assert!(test.errors().is_empty());
    let bundle = finish_bundle_ok(test, encoder);
    yawgpu::wgpuRenderBundleEncoderRelease(encoder);
    bundle
}

unsafe fn assert_bundle_finish_error<F>(
    test: &ValidationTest,
    descriptor: native::WGPURenderBundleEncoderDescriptor,
    commands: F,
) -> native::WGPURenderBundle
where
    F: FnOnce(native::WGPURenderBundleEncoder),
{
    let encoder = create_bundle_encoder_ok(test, descriptor);
    commands(encoder);
    assert!(test.errors().is_empty());
    let bundle = finish_bundle_error(test, encoder);
    yawgpu::wgpuRenderBundleEncoderRelease(encoder);
    bundle
}

unsafe fn finish_bundle_ok(
    test: &ValidationTest,
    encoder: native::WGPURenderBundleEncoder,
) -> native::WGPURenderBundle {
    test.clear_errors();
    let bundle = yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
    assert!(!bundle.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    bundle
}

unsafe fn finish_bundle_error(
    test: &ValidationTest,
    encoder: native::WGPURenderBundleEncoder,
) -> native::WGPURenderBundle {
    let mut bundle = std::ptr::null();
    test.assert_device_error_after(
        || {
            bundle = yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
        },
        None,
    );
    assert!(!bundle.is_null());
    bundle
}

unsafe fn assert_render_pass_ok<F>(test: &ValidationTest, sample_count: u32, commands: F)
where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = create_encoder(test);
    let target = create_render_target(
        test.device(),
        native::WGPUTextureFormat_RGBA8Unorm,
        sample_count,
    );
    let attachment = color_attachment(target.view);
    let attachments = [attachment];
    let descriptor = render_pass_descriptor(&attachments);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    commands(pass);
    assert!(test.errors().is_empty());
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    release_render_target(target);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn assert_render_pass_error<F>(test: &ValidationTest, sample_count: u32, commands: F)
where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = create_encoder(test);
    let target = create_render_target(
        test.device(),
        native::WGPUTextureFormat_RGBA8Unorm,
        sample_count,
    );
    let attachment = color_attachment(target.view);
    let attachments = [attachment];
    let descriptor = render_pass_descriptor(&attachments);
    test.clear_errors();
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    commands(pass);
    assert!(test.errors().is_empty());
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    release_render_target(target);
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

unsafe fn create_render_pipeline(
    test: &ValidationTest,
    vertex_source: &str,
    layout: Option<native::WGPUPipelineLayout>,
    vertex_buffers: &[native::WGPUVertexBufferLayout],
    format: native::WGPUTextureFormat,
    sample_count: u32,
) -> native::WGPURenderPipeline {
    let vertex_module = create_wgsl_module(test.device(), vertex_source);
    let fragment_module = create_wgsl_module(test.device(), fragment_source());
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment_module,
        entryPoint: empty_string_view(),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &color_target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: layout.unwrap_or(std::ptr::null()),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex_module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: vertex_buffers.len(),
            buffers: vertex_buffers.as_ptr(),
        },
        primitive: default_primitive(),
        depthStencil: std::ptr::null(),
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: sample_count,
            mask: 0xFFFF_FFFF,
            alphaToCoverageEnabled: 0,
        },
        fragment: &fragment,
    };
    test.clear_errors();
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor);
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuShaderModuleRelease(fragment_module);
    yawgpu::wgpuShaderModuleRelease(vertex_module);
    pipeline
}

unsafe fn create_wgsl_module(device: native::WGPUDevice, source: &str) -> native::WGPUShaderModule {
    let mut wgsl = native::WGPUShaderSourceWGSL {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ShaderSourceWGSL,
        },
        code: string_view(source),
    };
    let descriptor = native::WGPUShaderModuleDescriptor {
        nextInChain: (&mut wgsl.chain) as *mut _,
        label: empty_string_view(),
    };
    yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor)
}

unsafe fn create_bind_group_layout(
    device: native::WGPUDevice,
    entries: &[native::WGPUBindGroupLayoutEntry],
) -> native::WGPUBindGroupLayout {
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let layout = yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn create_pipeline_layout(
    device: native::WGPUDevice,
    layouts: &[native::WGPUBindGroupLayout],
) -> native::WGPUPipelineLayout {
    let descriptor = native::WGPUPipelineLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        bindGroupLayoutCount: layouts.len(),
        bindGroupLayouts: layouts.as_ptr(),
        immediateSize: 0,
    };
    let layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    entries: &[native::WGPUBindGroupEntry],
) -> native::WGPUBindGroup {
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!group.is_null());
    group
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    usage: native::WGPUBufferUsage,
    size: u64,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

struct RenderTarget {
    texture: native::WGPUTexture,
    view: native::WGPUTextureView,
}

unsafe fn create_render_target(
    device: native::WGPUDevice,
    format: native::WGPUTextureFormat,
    sample_count: u32,
) -> RenderTarget {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        },
        format,
        mipLevelCount: 1,
        sampleCount: sample_count,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
    assert!(!view.is_null());
    RenderTarget { texture, view }
}

unsafe fn release_render_target(target: RenderTarget) {
    yawgpu::wgpuTextureViewRelease(target.view);
    yawgpu::wgpuTextureRelease(target.texture);
}

unsafe fn device_limits(device: native::WGPUDevice) -> native::WGPULimits {
    let mut limits = std::mem::zeroed();
    assert_eq!(
        yawgpu::wgpuDeviceGetLimits(device, &mut limits),
        native::WGPUStatus_Success
    );
    limits
}

fn bundle_descriptor(
    formats: &[native::WGPUTextureFormat],
    depth_stencil_format: native::WGPUTextureFormat,
    sample_count: u32,
) -> native::WGPURenderBundleEncoderDescriptor {
    native::WGPURenderBundleEncoderDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorFormatCount: formats.len(),
        colorFormats: formats.as_ptr(),
        depthStencilFormat: depth_stencil_format,
        sampleCount: sample_count,
        depthReadOnly: 0,
        stencilReadOnly: 0,
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

fn buffer_binding(
    binding: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer,
        offset,
        size,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    }
}

fn uniform_layout(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_bind_group_layout_entry(binding, visibility);
    entry.buffer.type_ = native::WGPUBufferBindingType_Uniform;
    entry.buffer.minBindingSize = 16;
    entry
}

fn default_bind_group_layout_entry(
    binding: u32,
    visibility: native::WGPUShaderStage,
) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_BindingNotUsed,
            hasDynamicOffset: 0,
            minBindingSize: 0,
        },
        sampler: native::WGPUSamplerBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUSamplerBindingType_BindingNotUsed,
        },
        texture: native::WGPUTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            sampleType: native::WGPUTextureSampleType_BindingNotUsed,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
            multisampled: 0,
        },
        storageTexture: native::WGPUStorageTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            access: native::WGPUStorageTextureAccess_BindingNotUsed,
            format: native::WGPUTextureFormat_Undefined,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
        },
    }
}

fn vertex_buffer(
    array_stride: u64,
    attributes: &[native::WGPUVertexAttribute],
) -> native::WGPUVertexBufferLayout {
    native::WGPUVertexBufferLayout {
        nextInChain: std::ptr::null_mut(),
        stepMode: native::WGPUVertexStepMode_Vertex,
        arrayStride: array_stride,
        attributeCount: attributes.len(),
        attributes: attributes.as_ptr(),
    }
}

fn vertex_attribute(
    format: native::WGPUVertexFormat,
    offset: u64,
    shader_location: u32,
) -> native::WGPUVertexAttribute {
    native::WGPUVertexAttribute {
        nextInChain: std::ptr::null_mut(),
        format,
        offset,
        shaderLocation: shader_location,
    }
}

fn default_primitive() -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        nextInChain: std::ptr::null_mut(),
        topology: native::WGPUPrimitiveTopology_TriangleList,
        stripIndexFormat: native::WGPUIndexFormat_Undefined,
        frontFace: native::WGPUFrontFace_Undefined,
        cullMode: native::WGPUCullMode_Undefined,
        unclippedDepth: 0,
    }
}

fn render_uniform_vertex_input() -> &'static str {
    "struct U { value: vec4f }
     @group(0) @binding(0) var<uniform> u: U;
     @vertex fn vs(@location(0) pos: vec2f) -> @builtin(position) vec4f {
         return vec4f(pos, 0.0, 1.0) + u.value * 0.0;
     }"
}

fn render_no_input() -> &'static str {
    "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }"
}

fn fragment_source() -> &'static str {
    "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }"
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
