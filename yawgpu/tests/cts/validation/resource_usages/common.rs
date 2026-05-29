use yawgpu::native;
use yawgpu_test::ValidationTest;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    Ok,
    Error,
}

pub unsafe fn create_buffer(
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
    let buffer = unsafe { yawgpu::wgpuDeviceCreateBuffer(device, &descriptor) };
    assert!(!buffer.is_null());
    buffer
}

pub unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
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
        format,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = unsafe { yawgpu::wgpuDeviceCreateTexture(device, &descriptor) };
    assert!(!texture.is_null());
    texture
}

pub unsafe fn create_view(texture: native::WGPUTexture) -> native::WGPUTextureView {
    let view = unsafe { yawgpu::wgpuTextureCreateView(texture, std::ptr::null()) };
    assert!(!view.is_null());
    view
}

pub unsafe fn create_encoder(test: &ValidationTest) -> native::WGPUCommandEncoder {
    let encoder =
        unsafe { yawgpu::wgpuDeviceCreateCommandEncoder(test.device(), std::ptr::null()) };
    assert!(!encoder.is_null());
    encoder
}

pub unsafe fn finish_expect(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
    expect: Expect,
) -> native::WGPUCommandBuffer {
    match expect {
        Expect::Ok => {
            test.clear_errors();
            let command_buffer =
                unsafe { yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null()) };
            assert!(!command_buffer.is_null());
            assert!(
                test.errors().is_empty(),
                "unexpected errors: {:?}",
                test.errors()
            );
            command_buffer
        }
        Expect::Error => {
            let mut command_buffer = std::ptr::null();
            test.assert_device_error_after(
                || {
                    command_buffer =
                        unsafe { yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null()) };
                },
                None,
            );
            assert!(!command_buffer.is_null());
            command_buffer
        }
    }
}

pub unsafe fn assert_compute_buffer_alias(expect: Expect) {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Storage, 512);
        let entries = [storage_layout(0), storage_layout(1)];
        let bindings = [
            buffer_binding(0, buffer, 0, 32),
            buffer_binding(1, buffer, 0, 32),
        ];
        encode_compute_with_bind_group(
            &test,
            compute_two_storage_buffers(),
            &entries,
            &bindings,
            expect,
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

pub unsafe fn assert_compute_buffer_read_only_ok() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_Storage, 512);
        let entries = [readonly_storage_layout(0), readonly_storage_layout(1)];
        let bindings = [
            buffer_binding(0, buffer, 0, 32),
            buffer_binding(1, buffer, 0, 32),
        ];
        encode_compute_with_bind_group(
            &test,
            compute_two_readonly_storage_buffers(),
            &entries,
            &bindings,
            Expect::Ok,
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

pub unsafe fn assert_render_buffer_read_write_alias(expect: Expect) {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(
            test.device(),
            native::WGPUBufferUsage_Uniform | native::WGPUBufferUsage_Storage,
            64,
        );
        let entries = [uniform_layout(0), vertex_storage_layout(1)];
        let bindings = [
            buffer_binding(0, buffer, 0, 32),
            buffer_binding(1, buffer, 0, 32),
        ];
        encode_render_with_bind_group(
            &test,
            render_uniform_and_storage(),
            &entries,
            &bindings,
            None,
            expect,
        );
        yawgpu::wgpuBufferRelease(buffer);
    }
}

pub unsafe fn assert_storage_texture_alias(expect: Expect) {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(
            test.device(),
            native::WGPUTextureUsage_StorageBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
        );
        let view = create_view(texture);
        let entries = [storage_texture_layout(0), storage_texture_layout(1)];
        let bindings = [texture_binding(0, view), texture_binding(1, view)];
        encode_compute_with_bind_group(
            &test,
            compute_two_storage_textures(),
            &entries,
            &bindings,
            expect,
        );
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

pub unsafe fn assert_render_attachment_sampled_alias(expect: Expect) {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(
            test.device(),
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_TextureBinding,
            native::WGPUTextureFormat_RGBA8Unorm,
        );
        let view = create_view(texture);
        let entries = [sampled_texture_layout(0)];
        let bindings = [texture_binding(0, view)];
        encode_render_with_bind_group(
            &test,
            render_sampled_texture(),
            &entries,
            &bindings,
            Some(view),
            expect,
        );
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

pub unsafe fn assert_copy_and_pass_buffer_ok() {
    let test = ValidationTest::new();
    unsafe {
        let buffer = create_buffer(
            test.device(),
            native::WGPUBufferUsage_CopySrc
                | native::WGPUBufferUsage_CopyDst
                | native::WGPUBufferUsage_Storage,
            256,
        );
        let other = create_buffer(
            test.device(),
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
            256,
        );
        let encoder = create_encoder(&test);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, buffer, 0, other, 0, 4);
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        assert!(!pass.is_null());
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        let command_buffer = finish_expect(&test, encoder, Expect::Ok);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuBufferRelease(other);
        yawgpu::wgpuBufferRelease(buffer);
    }
}

pub unsafe fn assert_copy_and_render_texture_ok() {
    let test = ValidationTest::new();
    unsafe {
        let texture = create_texture(
            test.device(),
            native::WGPUTextureUsage_CopySrc
                | native::WGPUTextureUsage_CopyDst
                | native::WGPUTextureUsage_RenderAttachment,
            native::WGPUTextureFormat_RGBA8Unorm,
        );
        let view = create_view(texture);
        let buffer = create_buffer(test.device(), native::WGPUBufferUsage_CopyDst, 256);
        let encoder = create_encoder(&test);
        let source = texture_copy(texture);
        let destination = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: 256,
                rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
            },
            buffer,
        };
        let size = native::WGPUExtent3D {
            width: 1,
            height: 1,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &size);
        let attachment = color_attachment(view);
        let descriptor = render_pass_descriptor(&[attachment]);
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        assert!(!pass.is_null());
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        let command_buffer = finish_expect(&test, encoder, Expect::Ok);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
    }
}

unsafe fn encode_compute_with_bind_group(
    test: &ValidationTest,
    source: &str,
    layout_entries: &[native::WGPUBindGroupLayoutEntry],
    bind_group_entries: &[native::WGPUBindGroupEntry],
    expect: Expect,
) {
    unsafe {
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
        let command_buffer = finish_expect(test, encoder, expect);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
    }
}

unsafe fn encode_render_with_bind_group(
    test: &ValidationTest,
    vertex_source: &str,
    layout_entries: &[native::WGPUBindGroupLayoutEntry],
    bind_group_entries: &[native::WGPUBindGroupEntry],
    attachment_view: Option<native::WGPUTextureView>,
    expect: Expect,
) {
    unsafe {
        let target = if attachment_view.is_none() {
            let texture = create_texture(
                test.device(),
                native::WGPUTextureUsage_RenderAttachment,
                native::WGPUTextureFormat_RGBA8Unorm,
            );
            Some((texture, create_view(texture)))
        } else {
            None
        };
        let view = attachment_view.unwrap_or_else(|| target.expect("target").1);
        let (layout, bind_group, pipeline_layout) =
            create_bind_group_stack(test, layout_entries, bind_group_entries);
        let pipeline = create_render_pipeline(test, vertex_source, pipeline_layout);
        let encoder = create_encoder(test);
        let attachment = color_attachment(view);
        let descriptor = render_pass_descriptor(&[attachment]);
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        assert!(!pass.is_null());
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        let command_buffer = finish_expect(test, encoder, expect);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        if let Some((texture, target_view)) = target {
            yawgpu::wgpuTextureViewRelease(target_view);
            yawgpu::wgpuTextureRelease(texture);
        }
    }
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
    unsafe {
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
    let layout = unsafe { yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor) };
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
    let bind_group = unsafe { yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor) };
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
    let layout = unsafe { yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor) };
    assert!(!layout.is_null());
    layout
}

unsafe fn create_compute_pipeline(
    test: &ValidationTest,
    source: &str,
    layout: native::WGPUPipelineLayout,
) -> native::WGPUComputePipeline {
    unsafe {
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
}

unsafe fn create_render_pipeline(
    test: &ValidationTest,
    vertex_source: &str,
    layout: native::WGPUPipelineLayout,
) -> native::WGPURenderPipeline {
    unsafe {
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
    unsafe { yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor) }
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

fn texture_copy(texture: native::WGPUTexture) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
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
