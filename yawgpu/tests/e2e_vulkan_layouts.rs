#![cfg(feature = "vulkan")]
use std::os::raw::c_void;
use std::sync::Mutex;
use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};
/// Row pitch satisfying texture-to-buffer copy alignment.
const BYTES_PER_ROW: u32 = 256;

/// Creates an instance selecting the real Vulkan backend.
unsafe fn create_vulkan_instance() -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: YAWGPU_INSTANCE_BACKEND_VULKAN,
    };
    let descriptor = native::WGPUInstanceDescriptor {
        nextInChain: (&mut backend.chain) as *mut native::WGPUChainedStruct,
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
    };
    let instance = yawgpu::wgpuCreateInstance(&descriptor);
    assert!(!instance.is_null());
    instance
}

/// Requests a Vulkan adapter and waits for its callback.
unsafe fn request_adapter(instance: native::WGPUInstance) -> native::WGPUAdapter {
    let mut adapter: native::WGPUAdapter = std::ptr::null();
    let callback_info = native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_adapter_callback),
        userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info);
    wait(instance, future);
    assert!(!adapter.is_null());
    adapter
}

/// Requests a device with native uncaptured-error capture.
unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    errors: &Mutex<Vec<String>>,
) -> native::WGPUDevice {
    let mut device: native::WGPUDevice = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let mut descriptor: native::WGPUDeviceDescriptor = std::mem::zeroed();
    descriptor.uncapturedErrorCallbackInfo.callback = Some(error_callback);
    descriptor.uncapturedErrorCallbackInfo.userdata1 =
        (errors as *const Mutex<Vec<String>>).cast_mut().cast();
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, &descriptor, callback_info);
    wait(instance, future);
    assert!(!device.is_null());
    device
}

/// Stores the successfully requested adapter.
unsafe extern "C" fn request_adapter_callback(
    status: native::WGPURequestAdapterStatus,
    adapter: native::WGPUAdapter,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestAdapterStatus_Success);
    *(userdata1 as *mut native::WGPUAdapter) = adapter;
}

/// Stores the successfully requested device.
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

/// Stores the asynchronous mapping status.
unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
}

/// Maps a buffer, copies its bytes, and unmaps it.
unsafe fn read_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    offset: u64,
    len: usize,
) -> Vec<u8> {
    let mut status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuBufferMapAsync(
        buffer,
        native::WGPUMapMode_Read,
        usize::try_from(offset).expect("test offset fits in usize"),
        len,
        callback_info,
    );
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);

    let ptr = yawgpu::wgpuBufferGetConstMappedRange(
        buffer,
        usize::try_from(offset).expect("test offset fits in usize"),
        len,
    );
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

/// Creates a shader module through the WGSL C descriptor.
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

/// Borrows a Rust string as a native string view.
fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

/// Returns an empty native string view.
fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

/// Creates a compute pipeline with an inferred binding layout.
unsafe fn create_compute_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPUComputePipeline {
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("main"),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

/// Creates a full-screen triangle pipeline with no vertex buffers.
unsafe fn create_render_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &color_target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("vs"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: primitive_state(),
        depthStencil: std::ptr::null(),
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

/// Returns triangle-list primitive state.
fn primitive_state() -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        nextInChain: std::ptr::null_mut(),
        topology: native::WGPUPrimitiveTopology_TriangleList,
        stripIndexFormat: native::WGPUIndexFormat_Undefined,
        frontFace: native::WGPUFrontFace_Undefined,
        cullMode: native::WGPUCullMode_Undefined,
        unclippedDepth: 0,
    }
}

/// Returns single-sample rasterization state.
fn multisample_state() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 1,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
}

/// Captures errors through the public C callback, retaining their messages.
unsafe extern "C" fn error_callback(
    _device: *const native::WGPUDevice,
    error_type: native::WGPUErrorType,
    message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let errors = &*userdata1.cast::<Mutex<Vec<String>>>();
    let bytes = std::slice::from_raw_parts(message.data.cast::<u8>(), message.length);
    errors
        .lock()
        .expect("error lock")
        .push(format!("{error_type}: {}", String::from_utf8_lossy(bytes)));
}

/// Creates one RGBA8 texture for the selected mip/layer scenario.
unsafe fn create_texture(
    device: native::WGPUDevice,
    width: u32,
    mips: u32,
    layers: u32,
    render: bool,
) -> native::WGPUTexture {
    let mut descriptor: native::WGPUTextureDescriptor = std::mem::zeroed();
    descriptor.usage = native::WGPUTextureUsage_TextureBinding
        | native::WGPUTextureUsage_CopySrc
        | native::WGPUTextureUsage_CopyDst;
    if render {
        descriptor.usage |= native::WGPUTextureUsage_RenderAttachment;
    }
    descriptor.dimension = native::WGPUTextureDimension_2D;
    descriptor.size = extent(width, layers);
    descriptor.format = native::WGPUTextureFormat_RGBA8Unorm;
    descriptor.mipLevelCount = mips;
    descriptor.sampleCount = 1;
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

/// Selects exactly one mip and array layer as a 2D view.
unsafe fn create_view(
    texture: native::WGPUTexture,
    mip: u32,
    layer: u32,
) -> native::WGPUTextureView {
    let mut descriptor: native::WGPUTextureViewDescriptor = std::mem::zeroed();
    descriptor.format = native::WGPUTextureFormat_RGBA8Unorm;
    descriptor.dimension = native::WGPUTextureViewDimension_2D;
    descriptor.baseMipLevel = mip;
    descriptor.mipLevelCount = 1;
    descriptor.baseArrayLayer = layer;
    descriptor.arrayLayerCount = 1;
    descriptor.aspect = native::WGPUTextureAspect_All;
    let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
    assert!(!view.is_null());
    view
}

/// Creates a buffer of the requested size and usage.
unsafe fn create_buffer(
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
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

/// Describes a square mip or one array layer.
fn extent(width: u32, layers: u32) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width,
        height: width,
        depthOrArrayLayers: layers,
    }
}

/// Names a single mip and starting array layer for a texture copy.
fn texture_copy(
    texture: native::WGPUTexture,
    mip: u32,
    layer: u32,
) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: mip,
        origin: native::WGPUOrigin3D {
            x: 0,
            y: 0,
            z: layer,
        },
        aspect: native::WGPUTextureAspect_All,
    }
}

/// Uploads pixels through a 256-byte-row staging layout.
unsafe fn write_pixels(
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    mip: u32,
    layer: u32,
    width: u32,
    pixels: &[u8],
) {
    let mut padded = vec![0; BYTES_PER_ROW as usize * width as usize];
    for (source, destination) in pixels
        .chunks_exact(width as usize * 4)
        .zip(padded.chunks_exact_mut(BYTES_PER_ROW as usize))
    {
        destination[..source.len()].copy_from_slice(source);
    }
    let layout = native::WGPUTexelCopyBufferLayout {
        offset: 0,
        bytesPerRow: BYTES_PER_ROW,
        rowsPerImage: width,
    };
    yawgpu::wgpuQueueWriteTexture(
        queue,
        &texture_copy(texture, mip, layer),
        padded.as_ptr().cast(),
        padded.len(),
        &layout,
        &extent(width, 1),
    );
}

/// Records a mip-zero/layer-zero readback with padded rows.
unsafe fn copy_pixels(
    encoder: native::WGPUCommandEncoder,
    texture: native::WGPUTexture,
    buffer: native::WGPUBuffer,
    width: u32,
) {
    let destination = native::WGPUTexelCopyBufferInfo {
        buffer,
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: width,
        },
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
        encoder,
        &texture_copy(texture, 0, 0),
        &destination,
        &extent(width, 1),
    );
}

/// Reads pixels without the row padding.
unsafe fn read_pixels(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    width: u32,
) -> Vec<u8> {
    read_buffer(instance, buffer, 0, BYTES_PER_ROW as usize * width as usize)
        .chunks_exact(BYTES_PER_ROW as usize)
        .flat_map(|row| row[..width as usize * 4].iter().copied())
        .collect()
}

/// Creates the sampled texture binding and optional compute output binding.
unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    view: native::WGPUTextureView,
    output: Option<native::WGPUBuffer>,
) -> native::WGPUBindGroup {
    let mut entry: native::WGPUBindGroupEntry = std::mem::zeroed();
    entry.textureView = view;
    let mut entries = vec![entry];
    if let Some(buffer) = output {
        let mut entry: native::WGPUBindGroupEntry = std::mem::zeroed();
        entry.binding = 1;
        entry.buffer = buffer;
        entry.size = 4;
        entries.push(entry);
    }
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

/// Finishes and submits one command buffer, releasing its encoder.
unsafe fn submit_encoder(queue: native::WGPUQueue, encoder: native::WGPUCommandEncoder) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

/// Sampling a sibling mip must not be skipped because the image is attached.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_render_to_mip0_while_sampling_mip1_reads_sampled_color() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let errors = Mutex::new(Vec::new());
        let device = request_device(instance, adapter, &errors);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let color = [0x10, 0x20, 0x30, 0xFF];
        let texture = create_texture(device, 8, 2, 1, true);
        write_pixels(queue, texture, 1, 0, 4, &color.repeat(16));
        let sampled = create_view(texture, 1, 0);
        let attachment = create_view(texture, 0, 0);
        let module = create_wgsl_module(
            device,
            r#"
@group(0) @binding(0) var t: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let positions = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(positions[i], 0, 1);
}
@fragment fn fs() -> @location(0) vec4f { return textureLoad(t, vec2i(0, 0), 0); }
"#,
        );
        let pipeline = create_render_pipeline(device, module);
        let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
        let group = create_bind_group(device, layout, sampled, None);
        let readback = create_buffer(
            device,
            u64::from(BYTES_PER_ROW * 8),
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let mut color_attachment: native::WGPURenderPassColorAttachment = std::mem::zeroed();
        color_attachment.view = attachment;
        color_attachment.depthSlice = native::WGPU_DEPTH_SLICE_UNDEFINED;
        color_attachment.loadOp = native::WGPULoadOp_Clear;
        color_attachment.storeOp = native::WGPUStoreOp_Store;
        let mut descriptor: native::WGPURenderPassDescriptor = std::mem::zeroed();
        descriptor.colorAttachmentCount = 1;
        descriptor.colorAttachments = &color_attachment;
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        copy_pixels(encoder, texture, readback, 8);
        submit_encoder(queue, encoder);
        assert_eq!(read_pixels(instance, readback, 8), color.repeat(64));
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBindGroupRelease(group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(attachment);
        yawgpu::wgpuTextureViewRelease(sampled);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Sampling one array layer preserves the other layer's copy state and pixels.
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_copy_to_layer0_then_sample_layer1_then_copy_layer0_round_trips() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let errors = Mutex::new(Vec::new());
        let device = request_device(instance, adapter, &errors);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let texture = create_texture(device, 4, 1, 2, false);
        let pattern: Vec<u8> = (0..64).map(|i| (i * 3 + 7) as u8).collect();
        let color = [0x40, 0x80, 0xC0, 0xFF];
        write_pixels(queue, texture, 0, 0, 4, &pattern);
        write_pixels(queue, texture, 0, 1, 4, &color.repeat(16));
        let view = create_view(texture, 0, 1);
        let output = create_buffer(
            device,
            4,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        // WebGPU forbids MapRead on storage buffers, so read through CopyDst.
        let output_readback = create_buffer(
            device,
            4,
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        let readback = create_buffer(
            device,
            u64::from(BYTES_PER_ROW * 4),
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        let module = create_wgsl_module(
            device,
            r#"
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn main() { out[0] = pack4x8unorm(textureLoad(t, vec2i(0, 0), 0)); }
"#,
        );
        let pipeline = create_compute_pipeline(device, module);
        let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        let group = create_bind_group(device, layout, view, Some(output));
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        copy_pixels(encoder, texture, readback, 4);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, output_readback, 0, 4);
        submit_encoder(queue, encoder);
        assert_eq!(read_pixels(instance, readback, 4), pattern);
        let packed = read_buffer(instance, output_readback, 0, 4);
        let value = u32::from_ne_bytes(packed.try_into().expect("four bytes"));
        assert_eq!(value.to_le_bytes(), color);
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuBindGroupRelease(group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(output_readback);
        yawgpu::wgpuBufferRelease(output);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// A tiled draw transitions copied sampled textures and preserves depth layout for readback.
#[cfg(feature = "tiled")]
#[test]
#[ignore = "manual real-backend test"]
fn vulkan_tiled_subpass_samples_texture_written_by_copy() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let errors = Mutex::new(Vec::new());
        let device = request_device(instance, adapter, &errors);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let color = [0x10, 0x20, 0x30, 0xFF];
        let mut texture_descriptor: native::WGPUTextureDescriptor = std::mem::zeroed();
        texture_descriptor.usage =
            native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst;
        texture_descriptor.dimension = native::WGPUTextureDimension_2D;
        texture_descriptor.size = extent(4, 1);
        texture_descriptor.format = native::WGPUTextureFormat_RGBA8Unorm;
        texture_descriptor.mipLevelCount = 1;
        texture_descriptor.sampleCount = 1;
        let texture = yawgpu::wgpuDeviceCreateTexture(device, &texture_descriptor);
        assert!(!texture.is_null());
        write_pixels(queue, texture, 0, 0, 4, &color.repeat(16));
        let sampled = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        assert!(!sampled.is_null());
        texture_descriptor.usage =
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc;
        let target = yawgpu::wgpuDeviceCreateTexture(device, &texture_descriptor);
        assert!(!target.is_null());
        let attachment = yawgpu::wgpuTextureCreateView(target, std::ptr::null());
        assert!(!attachment.is_null());

        texture_descriptor.format = native::WGPUTextureFormat_Depth32Float;
        let depth = yawgpu::wgpuDeviceCreateTexture(device, &texture_descriptor);
        assert!(!depth.is_null());
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());
        assert!(!depth_view.is_null());
        let depth_layout = yawgpu::YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_Depth32Float,
            sampleCount: 1,
        };

        let color_layout = yawgpu::YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sampleCount: 1,
        };
        let color_slot = 0u32;
        let subpass_layout = yawgpu::YaWGPUSubpassLayout {
            colorAttachmentIndices: &color_slot,
            colorAttachmentIndexCount: 1,
            usesDepthStencil: 1,
            inputAttachments: std::ptr::null(),
            inputAttachmentCount: 0,
        };
        let layout_descriptor = yawgpu::YaWGPUSubpassPassLayoutDescriptor {
            nextInChain: std::ptr::null(),
            label: empty_string_view(),
            colorAttachments: &color_layout,
            colorAttachmentCount: 1,
            depthStencilAttachment: &depth_layout,
            subpasses: &subpass_layout,
            subpassCount: 1,
            dependencies: std::ptr::null(),
            dependencyCount: 0,
        };
        let pass_layout = yawgpu::yawgpuDeviceCreateSubpassPassLayout(device, &layout_descriptor);
        assert!(!pass_layout.is_null());
        let module = create_wgsl_module(
            device,
            r#"
@group(0) @binding(0) var t: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let positions = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(positions[i], 0, 1);
}
@fragment fn fs() -> @location(0) vec4f { return textureLoad(t, vec2i(0, 0), 0); }
"#,
        );
        let color_target = native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_RGBA8Unorm,
            blend: std::ptr::null(),
            writeMask: native::WGPUColorWriteMask_All,
        };
        let mut fragment: native::WGPUFragmentState = std::mem::zeroed();
        fragment.module = module;
        fragment.entryPoint = string_view("fs");
        fragment.targetCount = 1;
        fragment.targets = &color_target;
        let mut base: native::WGPURenderPipelineDescriptor = std::mem::zeroed();
        base.vertex.module = module;
        base.vertex.entryPoint = string_view("vs");
        base.primitive = primitive_state();
        base.multisample = multisample_state();
        base.fragment = &fragment;
        let mut depth_state: native::WGPUDepthStencilState = std::mem::zeroed();
        depth_state.format = native::WGPUTextureFormat_Depth32Float;
        depth_state.depthWriteEnabled = native::WGPUOptionalBool_True;
        depth_state.depthCompare = native::WGPUCompareFunction_Always;
        base.depthStencil = &depth_state;
        let pipeline_descriptor = yawgpu::YaWGPUSubpassRenderPipelineDescriptor {
            nextInChain: std::ptr::null(),
            base,
            passLayout: pass_layout,
            subpassIndex: 0,
        };
        let pipeline =
            yawgpu::yawgpuDeviceCreateSubpassRenderPipeline(device, &pipeline_descriptor);
        assert!(!pipeline.is_null());
        let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
        let group = create_bind_group(device, layout, sampled, None);
        let readback = create_buffer(
            device,
            u64::from(BYTES_PER_ROW * 4),
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        let depth_readback = create_buffer(
            device,
            u64::from(BYTES_PER_ROW * 4),
            native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
        );
        let depth_attachment = yawgpu::YaWGPUSubpassDepthStencilAttachment {
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 1.0,
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
        };
        let color_attachment = yawgpu::YaWGPUSubpassColorAttachment {
            view: attachment,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        };
        let descriptor = yawgpu::YaWGPUSubpassRenderPassDescriptor {
            nextInChain: std::ptr::null(),
            label: empty_string_view(),
            passLayout: pass_layout,
            extent: extent(4, 1),
            colorAttachments: &color_attachment,
            colorAttachmentCount: 1,
            depthStencilAttachment: &depth_attachment,
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &descriptor);
        assert!(!pass.is_null());
        yawgpu::yawgpuSubpassRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::yawgpuSubpassRenderPassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
        yawgpu::yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::yawgpuSubpassRenderPassEncoderEnd(pass);
        copy_pixels(encoder, target, readback, 4);
        let mut depth_copy = texture_copy(depth, 0, 0);
        depth_copy.aspect = native::WGPUTextureAspect_DepthOnly;
        let depth_destination = native::WGPUTexelCopyBufferInfo {
            buffer: depth_readback,
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: BYTES_PER_ROW,
                rowsPerImage: 4,
            },
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &depth_copy,
            &depth_destination,
            &extent(4, 1),
        );
        submit_encoder(queue, encoder);
        assert_eq!(read_pixels(instance, readback, 4), color.repeat(16));
        for pixel in read_pixels(instance, depth_readback, 4).chunks_exact(4) {
            assert_eq!(
                f32::from_ne_bytes(pixel.try_into().expect("depth texel")),
                0.0
            );
        }
        assert!(errors.lock().expect("error lock").is_empty(), "{errors:?}");
        yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);
        yawgpu::wgpuBufferRelease(depth_readback);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBindGroupRelease(group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::yawgpuSubpassPassLayoutRelease(pass_layout);
        yawgpu::wgpuTextureViewRelease(attachment);
        yawgpu::wgpuTextureViewRelease(sampled);
        yawgpu::wgpuTextureRelease(target);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}
