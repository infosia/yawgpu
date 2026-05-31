use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{wait, ValidationTest};

#[derive(Default)]
struct ComputeAsyncState {
    calls: u32,
    statuses: Vec<native::WGPUCreatePipelineAsyncStatus>,
    pipelines: Vec<native::WGPUComputePipeline>,
}

#[test]
fn bind_group_resources_must_belong_to_the_bind_group_device() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let local = create_buffer(test.device(), native::WGPUBufferUsage_Uniform, 256);
        let foreign = create_buffer(other, native::WGPUBufferUsage_Uniform, 256);
        let layout = create_bind_group_layout(test.device());

        let valid = create_bind_group(test.device(), layout, local);
        assert_no_device_error(&test);

        let mut error_group = std::ptr::null();
        test.assert_device_error_after(
            || {
                error_group = create_bind_group(test.device(), layout, foreign);
            },
            None,
        );
        assert!(!error_group.is_null());

        yawgpu::wgpuBindGroupRelease(error_group);
        yawgpu::wgpuBindGroupRelease(valid);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuBufferRelease(foreign);
        yawgpu::wgpuBufferRelease(local);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn command_encoder_copy_rejects_wrong_device_buffers_at_finish() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let local_src = create_buffer(test.device(), native::WGPUBufferUsage_CopySrc, 16);
        let local_dst = create_buffer(test.device(), native::WGPUBufferUsage_CopyDst, 16);
        let foreign_src = create_buffer(other, native::WGPUBufferUsage_CopySrc, 16);

        let valid_encoder = create_encoder(test.device());
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(valid_encoder, local_src, 0, local_dst, 0, 16);
        finish_ok(&test, valid_encoder);

        let error_encoder = create_encoder(test.device());
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(
            error_encoder,
            foreign_src,
            0,
            local_dst,
            0,
            16,
        );
        finish_error(&test, error_encoder);

        yawgpu::wgpuBufferRelease(foreign_src);
        yawgpu::wgpuBufferRelease(local_dst);
        yawgpu::wgpuBufferRelease(local_src);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn render_pass_rejects_wrong_device_state_objects_at_finish() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let foreign_buffer = create_buffer(
            other,
            native::WGPUBufferUsage_Vertex | native::WGPUBufferUsage_Index,
            16,
        );
        let foreign_bgl = create_bind_group_layout(other);
        let foreign_uniform = create_buffer(other, native::WGPUBufferUsage_Uniform, 256);
        let foreign_bind_group = create_bind_group(other, foreign_bgl, foreign_uniform);
        let foreign_pipeline = create_render_pipeline(other, std::ptr::null());
        let foreign_bundle = create_render_bundle(other);

        assert_render_pass_ok(&test, |_| {});
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetBindGroup(
                pass,
                0,
                foreign_bind_group,
                0,
                std::ptr::null(),
            );
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, foreign_buffer, 0, 16);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                pass,
                foreign_buffer,
                native::WGPUIndexFormat_Uint16,
                0,
                16,
            );
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderSetPipeline(pass, foreign_pipeline);
        });
        assert_render_pass_error(&test, |pass| {
            yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, &foreign_bundle);
        });

        yawgpu::wgpuRenderBundleRelease(foreign_bundle);
        yawgpu::wgpuRenderPipelineRelease(foreign_pipeline);
        yawgpu::wgpuBindGroupRelease(foreign_bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(foreign_bgl);
        yawgpu::wgpuBufferRelease(foreign_uniform);
        yawgpu::wgpuBufferRelease(foreign_buffer);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn compute_pass_rejects_wrong_device_state_objects_at_finish() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let foreign_bgl = create_bind_group_layout(other);
        let foreign_buffer = create_buffer(other, native::WGPUBufferUsage_Uniform, 256);
        let foreign_bind_group = create_bind_group(other, foreign_bgl, foreign_buffer);
        let foreign_pipeline = create_compute_pipeline(other, std::ptr::null());

        assert_compute_pass_ok(&test, |_| {});
        assert_compute_pass_error(&test, |pass| {
            yawgpu::wgpuComputePassEncoderSetBindGroup(
                pass,
                0,
                foreign_bind_group,
                0,
                std::ptr::null(),
            );
        });
        assert_compute_pass_error(&test, |pass| {
            yawgpu::wgpuComputePassEncoderSetPipeline(pass, foreign_pipeline);
        });

        yawgpu::wgpuComputePipelineRelease(foreign_pipeline);
        yawgpu::wgpuBindGroupRelease(foreign_bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(foreign_bgl);
        yawgpu::wgpuBufferRelease(foreign_buffer);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn queue_operations_reject_wrong_device_objects() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let queue = yawgpu::wgpuDeviceGetQueue(test.device());
        let foreign_buffer = create_buffer(other, native::WGPUBufferUsage_CopyDst, 16);

        test.assert_device_error_after(
            || {
                let data = [0_u8; 4];
                yawgpu::wgpuQueueWriteBuffer(queue, foreign_buffer, 0, data.as_ptr().cast(), 4);
            },
            None,
        );

        let foreign_encoder = create_encoder(other);
        let foreign_command_buffer =
            yawgpu::wgpuCommandEncoderFinish(foreign_encoder, std::ptr::null());
        assert!(!foreign_command_buffer.is_null());
        test.assert_device_error_after(
            || {
                yawgpu::wgpuQueueSubmit(queue, 1, &foreign_command_buffer);
            },
            None,
        );

        yawgpu::wgpuCommandBufferRelease(foreign_command_buffer);
        yawgpu::wgpuCommandEncoderRelease(foreign_encoder);
        yawgpu::wgpuBufferRelease(foreign_buffer);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(other);
    }
}

#[test]
fn pipeline_creation_rejects_wrong_device_shader_and_layout() {
    let test = ValidationTest::new();
    unsafe {
        let other = request_device(test.instance(), test.adapter());
        let foreign_module = create_wgsl_module(other, compute_source());
        let local_module = create_wgsl_module(test.device(), compute_source());
        let foreign_vertex_module = create_wgsl_module(other, vertex_source());
        let local_vertex_module = create_wgsl_module(test.device(), vertex_source());
        let local_fragment_module = create_wgsl_module(test.device(), fragment_source());
        let foreign_bgl = create_bind_group_layout(other);
        let foreign_layout = create_pipeline_layout(other, &[foreign_bgl]);

        let valid = create_compute_pipeline(test.device(), std::ptr::null());
        assert_no_device_error(&test);

        let mut sync_error = std::ptr::null();
        test.assert_device_error_after(
            || {
                sync_error = create_compute_pipeline_with_module(
                    test.device(),
                    std::ptr::null(),
                    foreign_module,
                );
            },
            None,
        );
        assert!(!sync_error.is_null());

        let mut layout_error = std::ptr::null();
        test.assert_device_error_after(
            || {
                layout_error = create_compute_pipeline_with_module(
                    test.device(),
                    foreign_layout,
                    local_module,
                );
            },
            None,
        );
        assert!(!layout_error.is_null());

        let mut render_shader_error = std::ptr::null();
        test.assert_device_error_after(
            || {
                render_shader_error = create_render_pipeline_with_modules(
                    test.device(),
                    std::ptr::null(),
                    foreign_vertex_module,
                    local_fragment_module,
                );
            },
            None,
        );
        assert!(!render_shader_error.is_null());

        let mut render_layout_error = std::ptr::null();
        test.assert_device_error_after(
            || {
                render_layout_error = create_render_pipeline_with_modules(
                    test.device(),
                    foreign_layout,
                    local_vertex_module,
                    local_fragment_module,
                );
            },
            None,
        );
        assert!(!render_layout_error.is_null());

        let mut pipeline_layout_error = std::ptr::null();
        test.assert_device_error_after(
            || {
                pipeline_layout_error = create_pipeline_layout(test.device(), &[foreign_bgl]);
            },
            None,
        );
        assert!(!pipeline_layout_error.is_null());

        let mut state = ComputeAsyncState::default();
        let descriptor = compute_pipeline_descriptor(std::ptr::null(), foreign_module);
        test.clear_errors();
        yawgpu::wgpuDeviceCreateComputePipelineAsync(
            test.device(),
            &descriptor,
            native::WGPUCreateComputePipelineAsyncCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(compute_pipeline_callback),
                userdata1: (&mut state as *mut ComputeAsyncState).cast(),
                userdata2: std::ptr::null_mut(),
            },
        );
        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.calls, 1);
        assert_eq!(
            state.statuses,
            vec![native::WGPUCreatePipelineAsyncStatus_ValidationError]
        );
        assert!(state.pipelines[0].is_null());
        assert_no_device_error(&test);

        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout_error);
        yawgpu::wgpuRenderPipelineRelease(render_layout_error);
        yawgpu::wgpuRenderPipelineRelease(render_shader_error);
        yawgpu::wgpuComputePipelineRelease(layout_error);
        yawgpu::wgpuComputePipelineRelease(sync_error);
        yawgpu::wgpuComputePipelineRelease(valid);
        yawgpu::wgpuPipelineLayoutRelease(foreign_layout);
        yawgpu::wgpuBindGroupLayoutRelease(foreign_bgl);
        yawgpu::wgpuShaderModuleRelease(local_fragment_module);
        yawgpu::wgpuShaderModuleRelease(local_vertex_module);
        yawgpu::wgpuShaderModuleRelease(foreign_vertex_module);
        yawgpu::wgpuShaderModuleRelease(local_module);
        yawgpu::wgpuShaderModuleRelease(foreign_module);
        yawgpu::wgpuDeviceRelease(other);
    }
}

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let mut device: native::WGPUDevice = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), callback_info);
    wait(instance, future);
    assert!(!device.is_null());
    device
}

unsafe extern "C" fn request_device_callback(
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestDeviceStatus_Success);
    *(userdata1 as *mut native::WGPUDevice) = device;
}

unsafe extern "C" fn compute_pipeline_callback(
    status: native::WGPUCreatePipelineAsyncStatus,
    pipeline: native::WGPUComputePipeline,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let state = &mut *(userdata1 as *mut ComputeAsyncState);
    state.calls += 1;
    state.statuses.push(status);
    state.pipelines.push(pipeline);
}

unsafe fn assert_render_pass_ok<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = create_encoder(test.device());
    let target = create_render_target(test.device());
    let attachment = color_attachment(target.view);
    let attachments = [attachment];
    let descriptor = render_pass_descriptor(&attachments);
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    commands(pass);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    finish_ok(test, encoder);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    release_render_target(target);
}

unsafe fn assert_render_pass_error<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPURenderPassEncoder),
{
    let encoder = create_encoder(test.device());
    let target = create_render_target(test.device());
    let attachment = color_attachment(target.view);
    let attachments = [attachment];
    let descriptor = render_pass_descriptor(&attachments);
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    commands(pass);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    finish_error(test, encoder);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    release_render_target(target);
}

unsafe fn assert_compute_pass_ok<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPUComputePassEncoder),
{
    let encoder = create_encoder(test.device());
    let descriptor = native::WGPUComputePassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        timestampWrites: std::ptr::null(),
    };
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, &descriptor);
    commands(pass);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    finish_ok(test, encoder);
    yawgpu::wgpuComputePassEncoderRelease(pass);
}

unsafe fn assert_compute_pass_error<F>(test: &ValidationTest, commands: F)
where
    F: FnOnce(native::WGPUComputePassEncoder),
{
    let encoder = create_encoder(test.device());
    let descriptor = native::WGPUComputePassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        timestampWrites: std::ptr::null(),
    };
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, &descriptor);
    commands(pass);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    finish_error(test, encoder);
    yawgpu::wgpuComputePassEncoderRelease(pass);
}

unsafe fn finish_ok(test: &ValidationTest, encoder: native::WGPUCommandEncoder) {
    test.clear_errors();
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert!(!command_buffer.is_null());
    assert_no_device_error(test);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn finish_error(test: &ValidationTest, encoder: native::WGPUCommandEncoder) {
    let mut command_buffer = std::ptr::null();
    test.assert_device_error_after(
        || {
            command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        },
        None,
    );
    assert!(!command_buffer.is_null());
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

fn assert_no_device_error(test: &ValidationTest) {
    assert!(
        test.errors().is_empty(),
        "unexpected errors: {:?}",
        test.errors()
    );
}

unsafe fn create_encoder(device: native::WGPUDevice) -> native::WGPUCommandEncoder {
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    assert!(!encoder.is_null());
    encoder
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

unsafe fn create_bind_group_layout(device: native::WGPUDevice) -> native::WGPUBindGroupLayout {
    let buffer = native::WGPUBufferBindingLayout {
        nextInChain: std::ptr::null_mut(),
        type_: native::WGPUBufferBindingType_Uniform,
        hasDynamicOffset: 0,
        minBindingSize: 0,
    };
    let entry = native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        visibility: native::WGPUShaderStage_Compute | native::WGPUShaderStage_Vertex,
        bindingArraySize: 0,
        buffer,
        sampler: zero_sampler_layout(),
        texture: zero_texture_layout(),
        storageTexture: zero_storage_texture_layout(),
    };
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: 1,
        entries: &entry,
    };
    let layout = yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    buffer: native::WGPUBuffer,
) -> native::WGPUBindGroup {
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer,
        offset: 0,
        size: 256,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    };
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: 1,
        entries: &entry,
    };
    let group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!group.is_null());
    group
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
    device: native::WGPUDevice,
    layout: native::WGPUPipelineLayout,
) -> native::WGPUComputePipeline {
    let module = create_wgsl_module(device, compute_source());
    let pipeline = create_compute_pipeline_with_module(device, layout, module);
    yawgpu::wgpuShaderModuleRelease(module);
    pipeline
}

unsafe fn create_compute_pipeline_with_module(
    device: native::WGPUDevice,
    layout: native::WGPUPipelineLayout,
    module: native::WGPUShaderModule,
) -> native::WGPUComputePipeline {
    let descriptor = compute_pipeline_descriptor(layout, module);
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

fn compute_pipeline_descriptor(
    layout: native::WGPUPipelineLayout,
    module: native::WGPUShaderModule,
) -> native::WGPUComputePipelineDescriptor {
    native::WGPUComputePipelineDescriptor {
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
    }
}

unsafe fn create_render_pipeline(
    device: native::WGPUDevice,
    layout: native::WGPUPipelineLayout,
) -> native::WGPURenderPipeline {
    let vertex = create_wgsl_module(device, vertex_source());
    let fragment_module = create_wgsl_module(device, fragment_source());
    let pipeline = create_render_pipeline_with_modules(device, layout, vertex, fragment_module);
    yawgpu::wgpuShaderModuleRelease(fragment_module);
    yawgpu::wgpuShaderModuleRelease(vertex);
    pipeline
}

unsafe fn create_render_pipeline_with_modules(
    device: native::WGPUDevice,
    layout: native::WGPUPipelineLayout,
    vertex: native::WGPUShaderModule,
    fragment_module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let target = native::WGPUColorTargetState {
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
        targets: &target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
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
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

unsafe fn create_render_bundle(device: native::WGPUDevice) -> native::WGPURenderBundle {
    let formats = [native::WGPUTextureFormat_RGBA8Unorm];
    let descriptor = native::WGPURenderBundleEncoderDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorFormatCount: formats.len(),
        colorFormats: formats.as_ptr(),
        depthStencilFormat: native::WGPUTextureFormat_Undefined,
        sampleCount: 1,
        depthReadOnly: 0,
        stencilReadOnly: 0,
    };
    let encoder = yawgpu::wgpuDeviceCreateRenderBundleEncoder(device, &descriptor);
    assert!(!encoder.is_null());
    let bundle = yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
    assert!(!bundle.is_null());
    yawgpu::wgpuRenderBundleEncoderRelease(encoder);
    bundle
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
    let module = yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor);
    assert!(!module.is_null());
    module
}

struct RenderTarget {
    texture: native::WGPUTexture,
    view: native::WGPUTextureView,
}

unsafe fn create_render_target(device: native::WGPUDevice) -> RenderTarget {
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
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
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

fn compute_source() -> &'static str {
    "@group(0) @binding(0) var<uniform> u: vec4f;
     @compute @workgroup_size(1) fn main() { _ = u; }"
}

fn vertex_source() -> &'static str {
    "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }"
}

fn fragment_source() -> &'static str {
    "@fragment fn fs() -> @location(0) vec4f { return vec4f(); }"
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

fn zero_sampler_layout() -> native::WGPUSamplerBindingLayout {
    native::WGPUSamplerBindingLayout {
        nextInChain: std::ptr::null_mut(),
        type_: native::WGPUSamplerBindingType_BindingNotUsed,
    }
}

fn zero_texture_layout() -> native::WGPUTextureBindingLayout {
    native::WGPUTextureBindingLayout {
        nextInChain: std::ptr::null_mut(),
        sampleType: native::WGPUTextureSampleType_BindingNotUsed,
        viewDimension: native::WGPUTextureViewDimension_Undefined,
        multisampled: 0,
    }
}

fn zero_storage_texture_layout() -> native::WGPUStorageTextureBindingLayout {
    native::WGPUStorageTextureBindingLayout {
        nextInChain: std::ptr::null_mut(),
        access: native::WGPUStorageTextureAccess_BindingNotUsed,
        format: native::WGPUTextureFormat_Undefined,
        viewDimension: native::WGPUTextureViewDimension_Undefined,
    }
}
