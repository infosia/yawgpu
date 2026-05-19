use yawgpu::native;
use yawgpu_test::ValidationTest;

#[test]
fn storage_buffer_write_write_aliasing_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Storage, 512);
        assert_compute_usage_error(
            &test,
            compute_two_storage_buffers(),
            &[storage_layout(0), storage_layout(1)],
            &[
                buffer_binding(0, buffer, 0, 32),
                buffer_binding(1, buffer, 0, 32),
            ],
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn storage_buffer_disjoint_write_ranges_are_allowed() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Storage, 512);
        assert_compute_usage_ok(
            &test,
            compute_two_storage_buffers(),
            &[storage_layout(0), storage_layout(1)],
            &[
                buffer_binding(0, buffer, 0, 256),
                buffer_binding(1, buffer, 256, 256),
            ],
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn buffer_read_write_aliasing_in_render_draw_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(
            test.device(),
            native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_Storage,
            64,
        );
        assert_render_usage_error(
            &test,
            render_uniform_and_storage(),
            &[uniform_layout(0), vertex_storage_layout(1)],
            &[
                buffer_binding(0, buffer, 0, 32),
                buffer_binding(1, buffer, 0, 32),
            ],
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn buffer_read_only_aliasing_is_allowed() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Storage, 512);
        assert_compute_usage_ok(
            &test,
            compute_two_readonly_storage_buffers(),
            &[readonly_storage_layout(0), readonly_storage_layout(1)],
            &[
                buffer_binding(0, buffer, 0, 32),
                buffer_binding(1, buffer, 256, 32),
            ],
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

#[test]
fn storage_texture_write_write_aliasing_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(test.device(), native::WGPUTextureUsage_StorageBinding);
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        assert!(!view.is_null());
        assert_compute_usage_error(
            &test,
            compute_two_storage_textures(),
            &[storage_texture_layout(0), storage_texture_layout(1)],
            &[texture_binding(0, view), texture_binding(1, view)],
        );
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn different_storage_textures_are_allowed() {
    let test = ValidationTest::new();
    unsafe {
        let texture_a = create_texture(test.device(), native::WGPUTextureUsage_StorageBinding);
        let texture_b = create_texture(test.device(), native::WGPUTextureUsage_StorageBinding);
        let view_a = yawgpu::wgpuTextureCreateView(texture_a, std::ptr::null());
        let view_b = yawgpu::wgpuTextureCreateView(texture_b, std::ptr::null());
        assert!(!view_a.is_null());
        assert!(!view_b.is_null());
        assert_compute_usage_ok(
            &test,
            compute_two_storage_textures(),
            &[storage_texture_layout(0), storage_texture_layout(1)],
            &[texture_binding(0, view_a), texture_binding(1, view_b)],
        );
        yawgpu::wgpuTextureViewRelease(view_b);
        yawgpu::wgpuTextureViewRelease(view_a);
        yawgpu::wgpuTextureRelease(texture_b);
        yawgpu::wgpuTextureRelease(texture_a);
    }
}

#[test]
fn render_attachment_sampled_in_same_pass_is_an_error() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_TextureBinding,
        );
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        assert!(!view.is_null());
        assert_render_usage_error_with_attachment(
            &test,
            render_sampled_texture(),
            &[sampled_texture_layout(0)],
            &[texture_binding(0, view)],
            view,
        );
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

#[test]
fn sampled_texture_different_from_attachment_is_allowed() {
    let test = ValidationTest::new();
    unsafe {
        let attachment = create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
        let sampled = create_texture(test.device(), native::WGPUTextureUsage_TextureBinding);
        let attachment_view = yawgpu::wgpuTextureCreateView(attachment, std::ptr::null());
        let sampled_view = yawgpu::wgpuTextureCreateView(sampled, std::ptr::null());
        assert!(!attachment_view.is_null());
        assert!(!sampled_view.is_null());
        assert_render_usage_ok_with_attachment(
            &test,
            render_sampled_texture(),
            &[sampled_texture_layout(0)],
            &[texture_binding(0, sampled_view)],
            attachment_view,
        );
        yawgpu::wgpuTextureViewRelease(sampled_view);
        yawgpu::wgpuTextureViewRelease(attachment_view);
        yawgpu::wgpuTextureRelease(sampled);
        yawgpu::wgpuTextureRelease(attachment);
    }
}

unsafe fn assert_compute_usage_ok(
    test: &ValidationTest,
    source: &str,
    layout_entries: &[native::WGPUBindGroupLayoutEntry],
    bind_group_entries: &[native::WGPUBindGroupEntry],
) {
    let (layout, bind_group, pipeline_layout) =
        create_bind_group_stack(test, layout_entries, bind_group_entries);
    let pipeline = create_compute_pipeline(test, source, pipeline_layout);
    let encoder = create_encoder(test);
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    assert!(!pass.is_null());
    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
}

unsafe fn assert_compute_usage_error(
    test: &ValidationTest,
    source: &str,
    layout_entries: &[native::WGPUBindGroupLayoutEntry],
    bind_group_entries: &[native::WGPUBindGroupEntry],
) {
    let (layout, bind_group, pipeline_layout) =
        create_bind_group_stack(test, layout_entries, bind_group_entries);
    let pipeline = create_compute_pipeline(test, source, pipeline_layout);
    let encoder = create_encoder(test);
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    assert!(!pass.is_null());
    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
}

unsafe fn assert_render_usage_error(
    test: &ValidationTest,
    vertex_source: &str,
    layout_entries: &[native::WGPUBindGroupLayoutEntry],
    bind_group_entries: &[native::WGPUBindGroupEntry],
) {
    let target = create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
    let target_view = yawgpu::wgpuTextureCreateView(target, std::ptr::null());
    assert_render_usage_error_with_attachment(
        test,
        vertex_source,
        layout_entries,
        bind_group_entries,
        target_view,
    );
    yawgpu::wgpuTextureViewRelease(target_view);
    yawgpu::wgpuTextureRelease(target);
}

unsafe fn assert_render_usage_error_with_attachment(
    test: &ValidationTest,
    vertex_source: &str,
    layout_entries: &[native::WGPUBindGroupLayoutEntry],
    bind_group_entries: &[native::WGPUBindGroupEntry],
    attachment_view: native::WGPUTextureView,
) {
    let (layout, bind_group, pipeline_layout) =
        create_bind_group_stack(test, layout_entries, bind_group_entries);
    let pipeline = create_render_pipeline(test, vertex_source, pipeline_layout);
    let encoder = create_encoder(test);
    let attachment = color_attachment(attachment_view);
    let descriptor = render_pass_descriptor(&[attachment]);
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = finish_error(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuRenderPipelineRelease(pipeline);
    yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
}

unsafe fn assert_render_usage_ok_with_attachment(
    test: &ValidationTest,
    vertex_source: &str,
    layout_entries: &[native::WGPUBindGroupLayoutEntry],
    bind_group_entries: &[native::WGPUBindGroupEntry],
    attachment_view: native::WGPUTextureView,
) {
    let (layout, bind_group, pipeline_layout) =
        create_bind_group_stack(test, layout_entries, bind_group_entries);
    let pipeline = create_render_pipeline(test, vertex_source, pipeline_layout);
    let encoder = create_encoder(test);
    let attachment = color_attachment(attachment_view);
    let descriptor = render_pass_descriptor(&[attachment]);
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = finish_ok(test, encoder);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuRenderPipelineRelease(pipeline);
    yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
}

unsafe fn create_bind_group_stack(
    test: &ValidationTest,
    layout_entries: &[native::WGPUBindGroupLayoutEntry],
    bind_group_entries: &[native::WGPUBindGroupEntry],
) -> (
    native::WGPUBindGroupLayout,
    native::WGPUBindGroup,
    native::WGPUPipelineLayout,
) {
    let layout = create_bind_group_layout(test.device(), layout_entries);
    test.clear_errors();
    let bind_group = create_bind_group(test.device(), layout, bind_group_entries);
    assert!(
        test.errors().is_empty(),
        "unexpected bind group errors: {:?}",
        test.errors()
    );
    let pipeline_layout = create_pipeline_layout(test.device(), &[layout]);
    (layout, bind_group, pipeline_layout)
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
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!bind_group.is_null());
    bind_group
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

unsafe fn create_compute_pipeline(
    test: &ValidationTest,
    source: &str,
    layout: native::WGPUPipelineLayout,
) -> native::WGPUComputePipeline {
    let module = create_wgsl_module(test.device(), source);
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    test.clear_errors();
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(test.device(), &descriptor);
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    yawgpu::wgpuShaderModuleRelease(module);
    pipeline
}

unsafe fn create_render_pipeline(
    test: &ValidationTest,
    vertex_source: &str,
    layout: native::WGPUPipelineLayout,
) -> native::WGPURenderPipeline {
    let vertex_module = create_wgsl_module(test.device(), vertex_source);
    let fragment_module = create_wgsl_module(test.device(), fragment_source());
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
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
        layout,
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex_module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: default_primitive(),
        depthStencil: std::ptr::null(),
        multisample: default_multisample(),
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

unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 4,
            height: 4,
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

fn texture_binding(binding: u32, view: native::WGPUTextureView) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer: std::ptr::null(),
        offset: 0,
        size: 0,
        sampler: std::ptr::null(),
        textureView: view,
    }
}

fn uniform_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_bind_group_layout_entry(binding);
    entry.visibility = native::WGPUShaderStage_Vertex;
    entry.buffer.type_ = native::WGPUBufferBindingType_Uniform;
    entry.buffer.minBindingSize = 16;
    entry
}

fn storage_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_bind_group_layout_entry(binding);
    entry.buffer.type_ = native::WGPUBufferBindingType_Storage;
    entry.buffer.minBindingSize = 16;
    entry
}

fn vertex_storage_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = storage_layout(binding);
    entry.visibility = native::WGPUShaderStage_Vertex;
    entry
}

fn readonly_storage_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_bind_group_layout_entry(binding);
    entry.buffer.type_ = native::WGPUBufferBindingType_ReadOnlyStorage;
    entry.buffer.minBindingSize = 16;
    entry
}

fn sampled_texture_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_bind_group_layout_entry(binding);
    entry.visibility = native::WGPUShaderStage_Vertex;
    entry.texture.sampleType = native::WGPUTextureSampleType_UnfilterableFloat;
    entry.texture.viewDimension = native::WGPUTextureViewDimension_2D;
    entry
}

fn storage_texture_layout(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    let mut entry = default_bind_group_layout_entry(binding);
    entry.storageTexture.access = native::WGPUStorageTextureAccess_WriteOnly;
    entry.storageTexture.format = native::WGPUTextureFormat_RGBA8Unorm;
    entry.storageTexture.viewDimension = native::WGPUTextureViewDimension_2D;
    entry
}

fn default_bind_group_layout_entry(binding: u32) -> native::WGPUBindGroupLayoutEntry {
    native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        visibility: native::WGPUShaderStage_Compute,
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

fn default_multisample() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 1,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
}

fn compute_two_storage_buffers() -> &'static str {
    "struct Data { value: array<u32> }
     @group(0) @binding(0) var<storage, read_write> a: Data;
     @group(0) @binding(1) var<storage, read_write> b: Data;
     @compute @workgroup_size(1) fn main() { a.value[0] = b.value[0]; }"
}

fn compute_two_readonly_storage_buffers() -> &'static str {
    "struct Data { value: array<u32> }
     @group(0) @binding(0) var<storage, read> a: Data;
     @group(0) @binding(1) var<storage, read> b: Data;
     @compute @workgroup_size(1) fn main() { _ = a.value[0] + b.value[0]; }"
}

fn compute_two_storage_textures() -> &'static str {
    "@group(0) @binding(0) var a: texture_storage_2d<rgba8unorm, write>;
     @group(0) @binding(1) var b: texture_storage_2d<rgba8unorm, write>;
     @compute @workgroup_size(1) fn main() {
         textureStore(a, vec2i(0, 0), vec4f());
         textureStore(b, vec2i(0, 0), vec4f());
     }"
}

fn render_uniform_and_storage() -> &'static str {
    "struct Data { value: vec4f }
     @group(0) @binding(0) var<uniform> u: Data;
     @group(0) @binding(1) var<storage, read_write> s: Data;
     @vertex fn vs() -> @builtin(position) vec4f {
         s.value = u.value;
         return u.value;
     }"
}

fn render_sampled_texture() -> &'static str {
    "@group(0) @binding(0) var t: texture_2d<f32>;
     @vertex fn vs() -> @builtin(position) vec4f {
         return textureLoad(t, vec2i(0, 0), 0);
     }"
}

fn fragment_source() -> &'static str {
    "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }"
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}
