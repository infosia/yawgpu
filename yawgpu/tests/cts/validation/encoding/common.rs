use yawgpu::native;
use yawgpu_test::ValidationTest;

#[derive(Clone, Copy)]
pub struct ViewResource {
    pub texture: native::WGPUTexture,
    pub view: native::WGPUTextureView,
}

#[derive(Clone, Copy)]
pub struct RenderTarget {
    pub texture: native::WGPUTexture,
    pub view: native::WGPUTextureView,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CommandExpectation {
    Success,
    FinishError,
    SubmitError,
}

#[derive(Clone, Copy)]
pub enum RenderEncodeType {
    RenderPass,
    RenderBundle,
}

pub enum RenderEncoder {
    RenderPass(native::WGPURenderPassEncoder),
    RenderBundle(native::WGPURenderBundleEncoder),
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

pub unsafe fn create_error_buffer(test: &ValidationTest) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: 0,
        size: 4,
        mappedAtCreation: 0,
    };
    let mut buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            buffer = unsafe { yawgpu::wgpuDeviceCreateBuffer(test.device(), &descriptor) };
        },
        None,
    );
    assert!(!buffer.is_null());
    buffer
}

pub unsafe fn create_texture(
    device: native::WGPUDevice,
    descriptor: native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    let texture = unsafe { yawgpu::wgpuDeviceCreateTexture(device, &descriptor) };
    assert!(!texture.is_null());
    texture
}

pub unsafe fn create_render_target(
    device: native::WGPUDevice,
    format: native::WGPUTextureFormat,
    sample_count: u32,
) -> RenderTarget {
    let texture = unsafe {
        create_texture(
            device,
            texture_descriptor(
                native::WGPUTextureUsage_RenderAttachment,
                format,
                native::WGPUTextureDimension_2D,
                extent(4, 4, 1),
                1,
                sample_count,
            ),
        )
    };
    let view = unsafe { yawgpu::wgpuTextureCreateView(texture, std::ptr::null()) };
    assert!(!view.is_null());
    RenderTarget { texture, view }
}

pub unsafe fn release_render_target(target: RenderTarget) {
    unsafe {
        yawgpu::wgpuTextureViewRelease(target.view);
        yawgpu::wgpuTextureRelease(target.texture);
    }
}

pub unsafe fn create_error_texture(test: &ValidationTest) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        usage: native::WGPUTextureUsage_None,
        ..texture_descriptor(
            native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureDimension_2D,
            extent(4, 4, 1),
            1,
            1,
        )
    };
    let mut texture = std::ptr::null();
    test.assert_device_error_after(
        || {
            texture = unsafe { yawgpu::wgpuDeviceCreateTexture(test.device(), &descriptor) };
        },
        None,
    );
    assert!(!texture.is_null());
    texture
}

pub fn texture_descriptor(
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

pub fn bundle_descriptor(
    formats: &[native::WGPUTextureFormat],
) -> native::WGPURenderBundleEncoderDescriptor {
    native::WGPURenderBundleEncoderDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorFormatCount: formats.len(),
        colorFormats: formats.as_ptr(),
        depthStencilFormat: native::WGPUTextureFormat_Undefined,
        sampleCount: 1,
        depthReadOnly: 0,
        stencilReadOnly: 0,
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

pub unsafe fn create_bundle_encoder(test: &ValidationTest) -> native::WGPURenderBundleEncoder {
    let formats = [native::WGPUTextureFormat_RGBA8Unorm];
    let descriptor = bundle_descriptor(&formats);
    test.clear_errors();
    let encoder =
        unsafe { yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor) };
    assert!(!encoder.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
    encoder
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

pub fn origin(x: u32, y: u32, z: u32) -> native::WGPUOrigin3D {
    native::WGPUOrigin3D { x, y, z }
}

pub fn texture_info(
    texture: native::WGPUTexture,
    mip_level: u32,
    origin: native::WGPUOrigin3D,
    aspect: native::WGPUTextureAspect,
) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: mip_level,
        origin,
        aspect,
    }
}

pub fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

pub unsafe fn expect_command_buffer(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
    expectation: CommandExpectation,
) {
    match expectation {
        CommandExpectation::Success => {
            let command_buffer = unsafe { finish_ok(test, encoder) };
            unsafe {
                let queue = yawgpu::wgpuDeviceGetQueue(test.device());
                test.clear_errors();
                yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
                assert!(
                    test.errors().is_empty(),
                    "unexpected submit errors: {:?}",
                    test.errors()
                );
                yawgpu::wgpuQueueRelease(queue);
                yawgpu::wgpuCommandBufferRelease(command_buffer);
            }
        }
        CommandExpectation::FinishError => {
            let command_buffer = unsafe { finish_error(test, encoder) };
            unsafe {
                yawgpu::wgpuCommandBufferRelease(command_buffer);
            }
        }
        CommandExpectation::SubmitError => {
            let command_buffer = unsafe { finish_ok(test, encoder) };
            let queue = unsafe { yawgpu::wgpuDeviceGetQueue(test.device()) };
            test.assert_device_error_after(
                || unsafe {
                    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
                },
                None,
            );
            unsafe {
                yawgpu::wgpuQueueRelease(queue);
                yawgpu::wgpuCommandBufferRelease(command_buffer);
            }
        }
    }
}

pub unsafe fn expect_render_pass_commands<F>(
    test: &ValidationTest,
    expectation: CommandExpectation,
    commands: F,
) where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = unsafe { create_encoder(test.device()) };
    let target =
        unsafe { create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1) };
    let attachment = color_attachment(target.view);
    let attachments = [attachment];
    let descriptor = render_pass_descriptor(&attachments, None);
    let pass = unsafe { begin_render_pass(encoder, &descriptor) };
    commands(pass);
    unsafe {
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        expect_command_buffer(test, encoder, expectation);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        release_render_target(target);
        yawgpu::wgpuCommandEncoderRelease(encoder);
    }
}

pub unsafe fn expect_render_bundle_commands<F>(
    test: &ValidationTest,
    expectation: CommandExpectation,
    commands: F,
) where
    F: FnOnce(native::WGPURenderBundleEncoder),
{
    let encoder = unsafe { create_bundle_encoder(test) };
    commands(encoder);
    match expectation {
        CommandExpectation::Success => {
            test.clear_errors();
            let bundle =
                unsafe { yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null()) };
            assert!(!bundle.is_null());
            assert!(
                test.errors().is_empty(),
                "unexpected errors: {:?}",
                test.errors()
            );
            unsafe { yawgpu::wgpuRenderBundleRelease(bundle) };
        }
        CommandExpectation::FinishError => {
            let mut bundle = std::ptr::null();
            test.assert_device_error_after(
                || {
                    bundle =
                        unsafe { yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null()) };
                },
                None,
            );
            assert!(!bundle.is_null());
            unsafe { yawgpu::wgpuRenderBundleRelease(bundle) };
        }
        CommandExpectation::SubmitError => unreachable!("render bundle is not queue-submitted"),
    }
    unsafe { yawgpu::wgpuRenderBundleEncoderRelease(encoder) };
}

pub unsafe fn expect_render_bundle_submit_error<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPURenderBundleEncoder),
{
    let bundle_encoder = unsafe { create_bundle_encoder(test) };
    commands(bundle_encoder);
    test.clear_errors();
    let bundle = unsafe { yawgpu::wgpuRenderBundleEncoderFinish(bundle_encoder, std::ptr::null()) };
    assert!(!bundle.is_null());
    assert!(
        test.errors().is_empty(),
        "unexpected bundle finish errors: {:?}",
        test.errors()
    );

    let command_encoder = unsafe { create_encoder(test.device()) };
    let target =
        unsafe { create_render_target(test.device(), native::WGPUTextureFormat_RGBA8Unorm, 1) };
    let attachment = color_attachment(target.view);
    let attachments = [attachment];
    let descriptor = render_pass_descriptor(&attachments, None);
    let pass = unsafe { begin_render_pass(command_encoder, &descriptor) };
    unsafe {
        yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, &bundle);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        expect_command_buffer(test, command_encoder, CommandExpectation::SubmitError);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        release_render_target(target);
        yawgpu::wgpuCommandEncoderRelease(command_encoder);
        yawgpu::wgpuRenderBundleRelease(bundle);
        yawgpu::wgpuRenderBundleEncoderRelease(bundle_encoder);
    }
}

pub unsafe fn expect_render_commands<F>(
    test: &ValidationTest,
    encode_type: RenderEncodeType,
    expectation: CommandExpectation,
    commands: F,
) where
    F: FnOnce(RenderEncoder),
{
    match encode_type {
        RenderEncodeType::RenderPass => unsafe {
            expect_render_pass_commands(test, expectation, |pass| {
                commands(RenderEncoder::RenderPass(pass));
            });
        },
        RenderEncodeType::RenderBundle => unsafe {
            if expectation == CommandExpectation::SubmitError {
                expect_render_bundle_submit_error(test, |bundle| {
                    commands(RenderEncoder::RenderBundle(bundle));
                });
            } else {
                expect_render_bundle_commands(test, expectation, |bundle| {
                    commands(RenderEncoder::RenderBundle(bundle));
                });
            }
        },
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

pub unsafe fn create_render_pipeline_with(
    test: &ValidationTest,
    device: native::WGPUDevice,
    vertex_source: &str,
    vertex_buffers: &[native::WGPUVertexBufferLayout],
    primitive: native::WGPUPrimitiveState,
) -> native::WGPURenderPipeline {
    let vertex = unsafe { create_wgsl_module(device, vertex_source) };
    let fragment = unsafe { create_wgsl_module(device, fragment_source()) };
    let target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_None,
    };
    let fragment_state = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fragment,
        entryPoint: string_view("fs"),
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
            entryPoint: string_view("vs"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: vertex_buffers.len(),
            buffers: vertex_buffers.as_ptr(),
        },
        primitive,
        depthStencil: std::ptr::null(),
        multisample: default_multisample(),
        fragment: &fragment_state,
    };
    if device == test.device() {
        test.clear_errors();
    }
    let pipeline = unsafe { yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor) };
    assert!(!pipeline.is_null());
    if device == test.device() {
        assert!(
            test.errors().is_empty(),
            "unexpected render pipeline errors: {:?}",
            test.errors()
        );
    }
    unsafe {
        yawgpu::wgpuShaderModuleRelease(fragment);
        yawgpu::wgpuShaderModuleRelease(vertex);
    }
    pipeline
}

pub unsafe fn create_error_render_pipeline(test: &ValidationTest) -> native::WGPURenderPipeline {
    let vertex = unsafe { create_wgsl_module(test.device(), vertex_no_input()) };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex,
            entryPoint: string_view("vs"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: invalid_strip_primitive(),
        depthStencil: std::ptr::null(),
        multisample: default_multisample(),
        fragment: std::ptr::null(),
    };
    let mut pipeline = std::ptr::null();
    test.assert_device_error_after(
        || {
            pipeline =
                unsafe { yawgpu::wgpuDeviceCreateRenderPipeline(test.device(), &descriptor) };
        },
        None,
    );
    assert!(!pipeline.is_null());
    unsafe { yawgpu::wgpuShaderModuleRelease(vertex) };
    pipeline
}

pub fn vertex_no_input() -> &'static str {
    "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }"
}

pub fn fragment_source() -> &'static str {
    "@fragment fn fs() {}"
}

pub fn vertex_input_shader(location: u32, ty: &str) -> String {
    format!(
        "@vertex fn vs(@location({location}) value: {ty}) -> @builtin(position) vec4f {{
            _ = value;
            return vec4f();
        }}"
    )
}

pub fn vertex_buffer_layout(
    step_mode: native::WGPUVertexStepMode,
    array_stride: u64,
    attributes: &[native::WGPUVertexAttribute],
) -> native::WGPUVertexBufferLayout {
    native::WGPUVertexBufferLayout {
        nextInChain: std::ptr::null_mut(),
        stepMode: step_mode,
        arrayStride: array_stride,
        attributeCount: attributes.len(),
        attributes: attributes.as_ptr(),
    }
}

pub fn vertex_attribute(
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

pub fn default_primitive() -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        nextInChain: std::ptr::null_mut(),
        topology: native::WGPUPrimitiveTopology_TriangleList,
        stripIndexFormat: native::WGPUIndexFormat_Undefined,
        frontFace: native::WGPUFrontFace_Undefined,
        cullMode: native::WGPUCullMode_Undefined,
        unclippedDepth: 0,
    }
}

pub fn strip_primitive(
    topology: native::WGPUPrimitiveTopology,
    format: native::WGPUIndexFormat,
) -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        topology,
        stripIndexFormat: format,
        ..default_primitive()
    }
}

pub fn invalid_strip_primitive() -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        stripIndexFormat: native::WGPUIndexFormat_Uint16,
        ..default_primitive()
    }
}

pub fn default_multisample() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 1,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
}

pub unsafe fn set_pipeline(encoder: &RenderEncoder, pipeline: native::WGPURenderPipeline) {
    unsafe {
        match *encoder {
            RenderEncoder::RenderPass(pass) => {
                yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            }
            RenderEncoder::RenderBundle(bundle) => {
                yawgpu::wgpuRenderBundleEncoderSetPipeline(bundle, pipeline);
            }
        }
    }
}

pub unsafe fn set_vertex_buffer(
    encoder: &RenderEncoder,
    slot: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    unsafe {
        match *encoder {
            RenderEncoder::RenderPass(pass) => {
                yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, slot, buffer, offset, size);
            }
            RenderEncoder::RenderBundle(bundle) => {
                yawgpu::wgpuRenderBundleEncoderSetVertexBuffer(bundle, slot, buffer, offset, size);
            }
        }
    }
}

pub unsafe fn set_index_buffer(
    encoder: &RenderEncoder,
    buffer: native::WGPUBuffer,
    format: native::WGPUIndexFormat,
    offset: u64,
    size: u64,
) {
    unsafe {
        match *encoder {
            RenderEncoder::RenderPass(pass) => {
                yawgpu::wgpuRenderPassEncoderSetIndexBuffer(pass, buffer, format, offset, size);
            }
            RenderEncoder::RenderBundle(bundle) => {
                yawgpu::wgpuRenderBundleEncoderSetIndexBuffer(bundle, buffer, format, offset, size);
            }
        }
    }
}

pub unsafe fn draw_with_offsets(
    encoder: &RenderEncoder,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    unsafe {
        match *encoder {
            RenderEncoder::RenderPass(pass) => yawgpu::wgpuRenderPassEncoderDraw(
                pass,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            ),
            RenderEncoder::RenderBundle(bundle) => yawgpu::wgpuRenderBundleEncoderDraw(
                bundle,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            ),
        }
    }
}

pub unsafe fn draw_indexed(
    encoder: &RenderEncoder,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    unsafe {
        match *encoder {
            RenderEncoder::RenderPass(pass) => yawgpu::wgpuRenderPassEncoderDrawIndexed(
                pass,
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            ),
            RenderEncoder::RenderBundle(bundle) => yawgpu::wgpuRenderBundleEncoderDrawIndexed(
                bundle,
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            ),
        }
    }
}

pub unsafe fn draw_indirect(encoder: &RenderEncoder, indirect: native::WGPUBuffer) {
    unsafe {
        match *encoder {
            RenderEncoder::RenderPass(pass) => {
                yawgpu::wgpuRenderPassEncoderDrawIndirect(pass, indirect, 0);
            }
            RenderEncoder::RenderBundle(bundle) => {
                yawgpu::wgpuRenderBundleEncoderDrawIndirect(bundle, indirect, 0);
            }
        }
    }
}

pub unsafe fn draw_indexed_indirect(encoder: &RenderEncoder, indirect: native::WGPUBuffer) {
    unsafe {
        match *encoder {
            RenderEncoder::RenderPass(pass) => {
                yawgpu::wgpuRenderPassEncoderDrawIndexedIndirect(pass, indirect, 0);
            }
            RenderEncoder::RenderBundle(bundle) => {
                yawgpu::wgpuRenderBundleEncoderDrawIndexedIndirect(bundle, indirect, 0);
            }
        }
    }
}
