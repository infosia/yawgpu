use yawgpu::native;
use yawgpu_test::ValidationTest;

#[derive(Clone, Copy)]
pub struct ViewResource {
    pub texture: native::WGPUTexture,
    pub view: native::WGPUTextureView,
}

pub unsafe fn create_encoder(device: native::WGPUDevice) -> native::WGPUCommandEncoder {
    let encoder = unsafe { yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null()) };
    assert!(!encoder.is_null());
    encoder
}

pub unsafe fn finish_ok(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    test.clear_errors();
    let command_buffer = unsafe { yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null()) };
    assert!(!command_buffer.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    command_buffer
}

pub unsafe fn finish_error(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    let mut command_buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            command_buffer = unsafe { yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null()) };
        },
        None,
    );
    assert!(!command_buffer.is_null());
    command_buffer
}

pub unsafe fn create_view(
    device: native::WGPUDevice,
    format: native::WGPUTextureFormat,
    usage: native::WGPUTextureUsage,
    sample_count: u32,
) -> ViewResource {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: extent(4, 4, 1),
        format,
        mipLevelCount: 1,
        sampleCount: sample_count,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = unsafe { yawgpu::wgpuDeviceCreateTexture(device, &descriptor) };
    assert!(!texture.is_null());
    let view = unsafe { yawgpu::wgpuTextureCreateView(texture, std::ptr::null()) };
    assert!(!view.is_null());
    ViewResource { texture, view }
}

pub unsafe fn release_view(resource: ViewResource) {
    unsafe {
        yawgpu::wgpuTextureViewRelease(resource.view);
        yawgpu::wgpuTextureRelease(resource.texture);
    }
}

pub unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
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

pub fn color_attachment(view: native::WGPUTextureView) -> native::WGPURenderPassColorAttachment {
    native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        },
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

pub fn depth_stencil_attachment(
    view: native::WGPUTextureView,
) -> native::WGPURenderPassDepthStencilAttachment {
    native::WGPURenderPassDepthStencilAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthLoadOp: native::WGPULoadOp_Clear,
        depthStoreOp: native::WGPUStoreOp_Store,
        depthClearValue: 0.0,
        depthReadOnly: 0,
        stencilLoadOp: native::WGPULoadOp_Clear,
        stencilStoreOp: native::WGPUStoreOp_Store,
        stencilClearValue: 0,
        stencilReadOnly: 0,
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

pub fn compute_pass_descriptor(
    timestamp_writes: Option<&native::WGPUPassTimestampWrites>,
) -> native::WGPUComputePassDescriptor {
    native::WGPUComputePassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        timestampWrites: timestamp_writes.map_or(std::ptr::null(), std::ptr::from_ref),
    }
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

pub unsafe fn begin_render_pass(
    encoder: native::WGPUCommandEncoder,
    descriptor: &native::WGPURenderPassDescriptor,
) -> native::WGPURenderPassEncoder {
    let pass = unsafe { yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, descriptor) };
    assert!(!pass.is_null());
    pass
}

pub unsafe fn begin_compute_pass(
    encoder: native::WGPUCommandEncoder,
    descriptor: Option<&native::WGPUComputePassDescriptor>,
) -> native::WGPUComputePassEncoder {
    let pass = unsafe {
        yawgpu::wgpuCommandEncoderBeginComputePass(
            encoder,
            descriptor.map_or(std::ptr::null(), std::ptr::from_ref),
        )
    };
    assert!(!pass.is_null());
    pass
}

pub unsafe fn expect_finish(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
    success: bool,
) -> native::WGPUCommandBuffer {
    if success {
        unsafe { finish_ok(test, encoder) }
    } else {
        unsafe { finish_error(test, encoder) }
    }
}

pub fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width,
        height,
        depthOrArrayLayers: depth_or_array_layers,
    }
}

pub fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

pub fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

pub unsafe fn create_wgsl_module(
    device: native::WGPUDevice,
    source: &str,
) -> native::WGPUShaderModule {
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
    let module = unsafe { yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor) };
    assert!(!module.is_null());
    module
}

pub unsafe fn create_compute_pipeline(test: &ValidationTest) -> native::WGPUComputePipeline {
    let module =
        unsafe { create_wgsl_module(test.device(), "@compute @workgroup_size(1) fn main() {}") };
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    test.clear_errors();
    let pipeline = unsafe { yawgpu::wgpuDeviceCreateComputePipeline(test.device(), &descriptor) };
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected compute pipeline errors: {:?}",
        test.errors()
    );
    unsafe {
        yawgpu::wgpuShaderModuleRelease(module);
    }
    pipeline
}

pub unsafe fn create_render_pipeline(test: &ValidationTest) -> native::WGPURenderPipeline {
    let vertex = unsafe {
        create_wgsl_module(
            test.device(),
            "@vertex fn main() -> @builtin(position) vec4f { return vec4f(0.0); }",
        )
    };
    let fragment = unsafe { create_wgsl_module(test.device(), "@fragment fn main() {}") };
    let target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_None,
    };
    let fragment_state = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment,
        entryPoint: empty_string_view(),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: native::WGPUFrontFace_Undefined,
            cullMode: native::WGPUCullMode_Undefined,
            unclippedDepth: 0,
        },
        depthStencil: std::ptr::null(),
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: 1,
            mask: 0xFFFF_FFFF,
            alphaToCoverageEnabled: 0,
        },
        fragment: &fragment_state,
    };
    test.clear_errors();
    let pipeline = unsafe { yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor) };
    assert!(!pipeline.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected render pipeline errors: {:?}",
        test.errors()
    );
    unsafe {
        yawgpu::wgpuShaderModuleRelease(fragment);
        yawgpu::wgpuShaderModuleRelease(vertex);
    }
    pipeline
}
