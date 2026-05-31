use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{assert_device_error, expect_no_validation_error, wait, ValidationTest};

const RGBA8_FORMATS: [native::WGPUTextureFormat; 1] = [native::WGPUTextureFormat_RGBA8Unorm];

#[derive(Default)]
pub struct MapState {
    pub statuses: Vec<native::WGPUMapAsyncStatus>,
}

#[derive(Clone, Copy)]
pub enum EncoderKind {
    ComputePass,
    RenderPass,
    RenderBundle,
}

pub unsafe fn queue(device: native::WGPUDevice) -> native::WGPUQueue {
    unsafe { yawgpu::wgpuDeviceGetQueue(device) }
}

pub unsafe fn submit(queue: native::WGPUQueue, command_buffers: &[native::WGPUCommandBuffer]) {
    unsafe {
        yawgpu::wgpuQueueSubmit(queue, command_buffers.len(), command_buffers.as_ptr());
    }
}

pub unsafe fn expect_submit(
    test: &ValidationTest,
    queue: native::WGPUQueue,
    command_buffers: &[native::WGPUCommandBuffer],
    success: bool,
) {
    if success {
        expect_no_validation_error(|| unsafe { submit(queue, command_buffers) });
    } else {
        assert_device_error!({
            unsafe { submit(queue, command_buffers) };
        });
    }
    let _ = test;
}

pub unsafe fn create_encoder(device: native::WGPUDevice) -> native::WGPUCommandEncoder {
    unsafe {
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        assert!(!encoder.is_null());
        encoder
    }
}

pub unsafe fn finish_ok(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    test.expect_no_validation_error(|| {});
    unsafe {
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        assert!(!command_buffer.is_null());
        assert!(
            test.errors().is_empty(),
            "unexpected errors: {:?}",
            test.errors()
        );
        command_buffer
    }
}

pub unsafe fn finish_error(
    test: &ValidationTest,
    encoder: native::WGPUCommandEncoder,
) -> native::WGPUCommandBuffer {
    let mut command_buffer = std::ptr::null();
    test.assert_device_error_after(
        || unsafe {
            command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        },
        None,
    );
    assert!(!command_buffer.is_null());
    command_buffer
}

pub unsafe fn create_empty_command_buffer(
    test: &ValidationTest,
    device: native::WGPUDevice,
    valid: bool,
) -> native::WGPUCommandBuffer {
    unsafe {
        let encoder = create_encoder(device);
        if !valid {
            yawgpu::wgpuCommandEncoderPopDebugGroup(encoder);
            finish_error(test, encoder)
        } else {
            finish_ok(test, encoder)
        }
    }
}

pub unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
    mapped_at_creation: bool,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: u32::from(mapped_at_creation),
    };
    unsafe {
        let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
        assert!(!buffer.is_null());
        buffer
    }
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
        || unsafe {
            buffer = yawgpu::wgpuDeviceCreateBuffer(test.device(), &descriptor);
        },
        None,
    );
    assert!(!buffer.is_null());
    buffer
}

pub unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    unsafe { create_texture_with_descriptor(device, texture_descriptor(usage)) }
}

pub unsafe fn create_texture_with_descriptor(
    device: native::WGPUDevice,
    descriptor: native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    unsafe {
        let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
        assert!(!texture.is_null());
        texture
    }
}

pub unsafe fn create_error_texture(test: &ValidationTest) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        usage: 0,
        ..texture_descriptor(native::WGPUTextureUsage_CopyDst)
    };
    let mut texture = std::ptr::null();
    test.assert_device_error_after(
        || unsafe {
            texture = yawgpu::wgpuDeviceCreateTexture(test.device(), &descriptor);
        },
        None,
    );
    assert!(!texture.is_null());
    texture
}

pub fn texture_descriptor(usage: native::WGPUTextureUsage) -> native::WGPUTextureDescriptor {
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

pub unsafe fn create_texture_view(texture: native::WGPUTexture) -> native::WGPUTextureView {
    unsafe {
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        assert!(!view.is_null());
        view
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
    unsafe {
        let query_set = yawgpu::wgpuDeviceCreateQuerySet(device, &descriptor);
        assert!(!query_set.is_null());
        query_set
    }
}

pub unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    #[derive(Default)]
    struct State {
        device: native::WGPUDevice,
    }
    unsafe extern "C" fn callback(
        status: native::WGPURequestDeviceStatus,
        device: native::WGPUDevice,
        _message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        assert_eq!(status, native::WGPURequestDeviceStatus_Success);
        unsafe {
            (*(userdata1 as *mut State)).device = device;
        }
    }

    let mut state = State::default();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(callback),
        userdata1: (&mut state as *mut State).cast(),
        userdata2: std::ptr::null_mut(),
    };
    unsafe {
        let future = yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), callback_info);
        wait(instance, future);
    }
    assert!(!state.device.is_null());
    state.device
}

pub unsafe fn map_async(
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    callback_mode: native::WGPUCallbackMode,
    state: &mut MapState,
) -> native::WGPUFuture {
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: callback_mode,
        callback: Some(map_callback),
        userdata1: (state as *mut MapState).cast(),
        userdata2: std::ptr::null_mut(),
    };
    unsafe { yawgpu::wgpuBufferMapAsync(buffer, mode, 0, 4, callback_info) }
}

pub unsafe fn queue_write_buffer(
    queue: native::WGPUQueue,
    buffer: native::WGPUBuffer,
    offset: u64,
    data_size: usize,
) {
    let data = [0_u8; 64];
    unsafe {
        yawgpu::wgpuQueueWriteBuffer(queue, buffer, offset, data.as_ptr().cast(), data_size);
    }
}

pub unsafe fn queue_write_texture(
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    data_size: u64,
) {
    let data = [0_u8; 64];
    let destination = native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: origin(0, 0, 0),
        aspect: native::WGPUTextureAspect_Undefined,
    };
    let layout = native::WGPUTexelCopyBufferLayout {
        offset: 0,
        bytesPerRow: native::WGPU_COPY_STRIDE_UNDEFINED,
        rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
    };
    unsafe {
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &destination,
            data.as_ptr().cast(),
            data_size as usize,
            &layout,
            &extent(1, 1, 1),
        );
    }
}

pub unsafe fn encode_copy_buffer_to_buffer(
    device: native::WGPUDevice,
    source: native::WGPUBuffer,
    destination: native::WGPUBuffer,
) -> native::WGPUCommandBuffer {
    unsafe {
        let encoder = create_encoder(device);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, source, 0, destination, 0, 4);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuCommandEncoderRelease(encoder);
        assert!(!command_buffer.is_null());
        command_buffer
    }
}

pub unsafe fn encode_copy_buffer_to_texture(
    device: native::WGPUDevice,
    buffer: native::WGPUBuffer,
    texture: native::WGPUTexture,
) -> native::WGPUCommandBuffer {
    unsafe {
        let encoder = create_encoder(device);
        let source = native::WGPUTexelCopyBufferInfo {
            layout: texture_layout(),
            buffer,
        };
        let destination = texture_info(texture);
        yawgpu::wgpuCommandEncoderCopyBufferToTexture(
            encoder,
            &source,
            &destination,
            &extent(1, 1, 1),
        );
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuCommandEncoderRelease(encoder);
        assert!(!command_buffer.is_null());
        command_buffer
    }
}

pub unsafe fn encode_copy_texture_to_buffer(
    device: native::WGPUDevice,
    texture: native::WGPUTexture,
    buffer: native::WGPUBuffer,
) -> native::WGPUCommandBuffer {
    unsafe {
        let encoder = create_encoder(device);
        let source = texture_info(texture);
        let destination = native::WGPUTexelCopyBufferInfo {
            layout: texture_layout(),
            buffer,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &source,
            &destination,
            &extent(1, 1, 1),
        );
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuCommandEncoderRelease(encoder);
        assert!(!command_buffer.is_null());
        command_buffer
    }
}

pub unsafe fn encode_copy_texture_to_texture(
    device: native::WGPUDevice,
    source_texture: native::WGPUTexture,
    destination_texture: native::WGPUTexture,
) -> native::WGPUCommandBuffer {
    unsafe {
        let encoder = create_encoder(device);
        let source = texture_info(source_texture);
        let destination = texture_info(destination_texture);
        yawgpu::wgpuCommandEncoderCopyTextureToTexture(
            encoder,
            &source,
            &destination,
            &extent(1, 1, 1),
        );
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuCommandEncoderRelease(encoder);
        assert!(!command_buffer.is_null());
        command_buffer
    }
}

pub unsafe fn encode_resolve_query_set(
    device: native::WGPUDevice,
    query_set: native::WGPUQuerySet,
    buffer: native::WGPUBuffer,
) -> native::WGPUCommandBuffer {
    unsafe {
        let encoder = create_encoder(device);
        yawgpu::wgpuCommandEncoderResolveQuerySet(encoder, query_set, 0, 1, buffer, 0);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuCommandEncoderRelease(encoder);
        assert!(!command_buffer.is_null());
        command_buffer
    }
}

pub unsafe fn encode_render_pass_with_attachments(
    device: native::WGPUDevice,
    color: native::WGPUTexture,
    resolve: native::WGPUTexture,
    depth: native::WGPUTexture,
) -> native::WGPUCommandBuffer {
    unsafe {
        let color_view = create_texture_view(color);
        let resolve_view = create_texture_view(resolve);
        let depth_view = create_texture_view(depth);
        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: color_view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: resolve_view,
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            },
        };
        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: 0,
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: 0,
        };
        let descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 1,
            colorAttachments: &color_attachment,
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = create_encoder(device);
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureViewRelease(resolve_view);
        yawgpu::wgpuTextureViewRelease(color_view);
        assert!(!command_buffer.is_null());
        command_buffer
    }
}

pub unsafe fn record_bind_group_use(
    test: &ValidationTest,
    kind: EncoderKind,
    bind_group: native::WGPUBindGroup,
) -> native::WGPUCommandBuffer {
    unsafe {
        match kind {
            EncoderKind::ComputePass => {
                let encoder = create_encoder(test.device());
                let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
                yawgpu::wgpuComputePassEncoderSetBindGroup(
                    pass,
                    0,
                    bind_group,
                    0,
                    std::ptr::null(),
                );
                yawgpu::wgpuComputePassEncoderEnd(pass);
                yawgpu::wgpuComputePassEncoderRelease(pass);
                let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
                yawgpu::wgpuCommandEncoderRelease(encoder);
                assert!(!command_buffer.is_null());
                command_buffer
            }
            EncoderKind::RenderPass => {
                let color =
                    create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
                let color_view = create_texture_view(color);
                let color_attachment = color_attachment(color_view);
                let color_attachments = [color_attachment];
                let descriptor = render_pass_descriptor(&color_attachments, std::ptr::null());
                let encoder = create_encoder(test.device());
                let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
                yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
                yawgpu::wgpuRenderPassEncoderEnd(pass);
                yawgpu::wgpuRenderPassEncoderRelease(pass);
                let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
                yawgpu::wgpuCommandEncoderRelease(encoder);
                yawgpu::wgpuTextureViewRelease(color_view);
                assert!(!command_buffer.is_null());
                command_buffer
            }
            EncoderKind::RenderBundle => {
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
                let encoder =
                    yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
                assert!(!encoder.is_null());
                yawgpu::wgpuRenderBundleEncoderSetBindGroup(
                    encoder,
                    0,
                    bind_group,
                    0,
                    std::ptr::null(),
                );
                let bundle = yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
                yawgpu::wgpuRenderBundleEncoderRelease(encoder);
                let color =
                    create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
                let color_view = create_texture_view(color);
                let color_attachment = color_attachment(color_view);
                let color_attachments = [color_attachment];
                let descriptor = render_pass_descriptor(&color_attachments, std::ptr::null());
                let command_encoder = create_encoder(test.device());
                let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(command_encoder, &descriptor);
                yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, &bundle);
                yawgpu::wgpuRenderPassEncoderEnd(pass);
                yawgpu::wgpuRenderPassEncoderRelease(pass);
                yawgpu::wgpuRenderBundleRelease(bundle);
                let command_buffer =
                    yawgpu::wgpuCommandEncoderFinish(command_encoder, std::ptr::null());
                yawgpu::wgpuCommandEncoderRelease(command_encoder);
                yawgpu::wgpuTextureViewRelease(color_view);
                assert!(!command_buffer.is_null());
                command_buffer
            }
        }
    }
}

pub unsafe fn record_vertex_buffer_use(
    test: &ValidationTest,
    kind: EncoderKind,
    buffer: native::WGPUBuffer,
) -> native::WGPUCommandBuffer {
    unsafe {
        match kind {
            EncoderKind::RenderPass => {
                let color =
                    create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
                let color_view = create_texture_view(color);
                let color_attachment = color_attachment(color_view);
                let color_attachments = [color_attachment];
                let descriptor = render_pass_descriptor(&color_attachments, std::ptr::null());
                let encoder = create_encoder(test.device());
                let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
                yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, buffer, 0, 4);
                yawgpu::wgpuRenderPassEncoderEnd(pass);
                yawgpu::wgpuRenderPassEncoderRelease(pass);
                let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
                yawgpu::wgpuCommandEncoderRelease(encoder);
                yawgpu::wgpuTextureViewRelease(color_view);
                assert!(!command_buffer.is_null());
                command_buffer
            }
            EncoderKind::RenderBundle => {
                let descriptor = bundle_descriptor();
                let encoder =
                    yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
                assert!(!encoder.is_null());
                yawgpu::wgpuRenderBundleEncoderSetVertexBuffer(encoder, 0, buffer, 0, 4);
                let bundle = yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
                yawgpu::wgpuRenderBundleEncoderRelease(encoder);
                render_bundle_to_command_buffer(test, bundle)
            }
            EncoderKind::ComputePass => unreachable!("compute pass has no vertex buffers"),
        }
    }
}

pub unsafe fn record_index_buffer_use(
    test: &ValidationTest,
    kind: EncoderKind,
    buffer: native::WGPUBuffer,
) -> native::WGPUCommandBuffer {
    unsafe {
        match kind {
            EncoderKind::RenderPass => {
                let color =
                    create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
                let color_view = create_texture_view(color);
                let color_attachment = color_attachment(color_view);
                let color_attachments = [color_attachment];
                let descriptor = render_pass_descriptor(&color_attachments, std::ptr::null());
                let encoder = create_encoder(test.device());
                let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
                yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                    pass,
                    buffer,
                    native::WGPUIndexFormat_Uint16,
                    0,
                    4,
                );
                yawgpu::wgpuRenderPassEncoderEnd(pass);
                yawgpu::wgpuRenderPassEncoderRelease(pass);
                let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
                yawgpu::wgpuCommandEncoderRelease(encoder);
                yawgpu::wgpuTextureViewRelease(color_view);
                assert!(!command_buffer.is_null());
                command_buffer
            }
            EncoderKind::RenderBundle => {
                let descriptor = bundle_descriptor();
                let encoder =
                    yawgpu::wgpuDeviceCreateRenderBundleEncoder(test.device(), &descriptor);
                assert!(!encoder.is_null());
                yawgpu::wgpuRenderBundleEncoderSetIndexBuffer(
                    encoder,
                    buffer,
                    native::WGPUIndexFormat_Uint16,
                    0,
                    4,
                );
                let bundle = yawgpu::wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
                yawgpu::wgpuRenderBundleEncoderRelease(encoder);
                render_bundle_to_command_buffer(test, bundle)
            }
            EncoderKind::ComputePass => unreachable!("compute pass has no index buffers"),
        }
    }
}

pub unsafe fn render_bundle_to_command_buffer(
    test: &ValidationTest,
    bundle: native::WGPURenderBundle,
) -> native::WGPUCommandBuffer {
    unsafe {
        let color = create_texture(test.device(), native::WGPUTextureUsage_RenderAttachment);
        let color_view = create_texture_view(color);
        let color_attachment = color_attachment(color_view);
        let color_attachments = [color_attachment];
        let descriptor = render_pass_descriptor(&color_attachments, std::ptr::null());
        let encoder = create_encoder(test.device());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        yawgpu::wgpuRenderPassEncoderExecuteBundles(pass, 1, &bundle);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        yawgpu::wgpuRenderBundleRelease(bundle);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuCommandEncoderRelease(encoder);
        yawgpu::wgpuTextureViewRelease(color_view);
        assert!(!command_buffer.is_null());
        command_buffer
    }
}

pub unsafe fn create_buffer_bind_group(
    device: native::WGPUDevice,
    buffer: native::WGPUBuffer,
) -> (native::WGPUBindGroupLayout, native::WGPUBindGroup) {
    let layout_entry = native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        visibility: native::WGPUShaderStage_Compute | native::WGPUShaderStage_Vertex,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_Uniform,
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
    };
    unsafe { create_bind_group(device, &layout_entry, buffer_binding(buffer)) }
}

pub unsafe fn create_texture_bind_group(
    device: native::WGPUDevice,
    view: native::WGPUTextureView,
    storage: bool,
) -> (native::WGPUBindGroupLayout, native::WGPUBindGroup) {
    let layout_entry = native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        visibility: if storage {
            native::WGPUShaderStage_Compute | native::WGPUShaderStage_Fragment
        } else {
            native::WGPUShaderStage_Compute | native::WGPUShaderStage_Vertex
        },
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
            sampleType: if storage {
                native::WGPUTextureSampleType_BindingNotUsed
            } else {
                native::WGPUTextureSampleType_Float
            },
            viewDimension: if storage {
                native::WGPUTextureViewDimension_Undefined
            } else {
                native::WGPUTextureViewDimension_2D
            },
            multisampled: 0,
        },
        storageTexture: native::WGPUStorageTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            access: if storage {
                native::WGPUStorageTextureAccess_WriteOnly
            } else {
                native::WGPUStorageTextureAccess_BindingNotUsed
            },
            format: if storage {
                native::WGPUTextureFormat_RGBA8Unorm
            } else {
                native::WGPUTextureFormat_Undefined
            },
            viewDimension: if storage {
                native::WGPUTextureViewDimension_2D
            } else {
                native::WGPUTextureViewDimension_Undefined
            },
        },
    };
    unsafe { create_bind_group(device, &layout_entry, texture_binding(view)) }
}

unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout_entry: &native::WGPUBindGroupLayoutEntry,
    entry: native::WGPUBindGroupEntry,
) -> (native::WGPUBindGroupLayout, native::WGPUBindGroup) {
    let layout_descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: 1,
        entries: layout_entry,
    };
    unsafe {
        let layout = yawgpu::wgpuDeviceCreateBindGroupLayout(device, &layout_descriptor);
        assert!(!layout.is_null());
        let descriptor = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout,
            entryCount: 1,
            entries: &entry,
        };
        let group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
        assert!(!group.is_null());
        (layout, group)
    }
}

fn buffer_binding(buffer: native::WGPUBuffer) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer,
        offset: 0,
        size: 4,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    }
}

fn texture_binding(view: native::WGPUTextureView) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer: std::ptr::null(),
        offset: 0,
        size: 0,
        sampler: std::ptr::null(),
        textureView: view,
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
        clearValue: native::WGPUColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        },
    }
}

pub fn render_pass_descriptor(
    color_attachments: &[native::WGPURenderPassColorAttachment],
    occlusion_query_set: native::WGPUQuerySet,
) -> native::WGPURenderPassDescriptor {
    native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: color_attachments.len(),
        colorAttachments: color_attachments.as_ptr(),
        depthStencilAttachment: std::ptr::null(),
        occlusionQuerySet: occlusion_query_set,
        timestampWrites: std::ptr::null(),
    }
}

fn bundle_descriptor() -> native::WGPURenderBundleEncoderDescriptor {
    native::WGPURenderBundleEncoderDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorFormatCount: RGBA8_FORMATS.len(),
        colorFormats: RGBA8_FORMATS.as_ptr(),
        depthStencilFormat: native::WGPUTextureFormat_Undefined,
        sampleCount: 1,
        depthReadOnly: 0,
        stencilReadOnly: 0,
    }
}

pub fn texture_info(texture: native::WGPUTexture) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: origin(0, 0, 0),
        aspect: native::WGPUTextureAspect_Undefined,
    }
}

pub fn texture_layout() -> native::WGPUTexelCopyBufferLayout {
    native::WGPUTexelCopyBufferLayout {
        offset: 0,
        bytesPerRow: native::WGPU_COPY_STRIDE_UNDEFINED,
        rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
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

pub fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    unsafe {
        (*(userdata1 as *mut MapState)).statuses.push(status);
    }
}
