#![cfg(all(feature = "metal", feature = "shader-passthrough"))]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::{
    native, YaWGPUInstanceBackendSelect, YaWGPUMslEntryPoint, YaWGPUShaderModuleMslDescriptor,
    YAWGPU_INSTANCE_BACKEND_METAL, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const ELEMENTS: usize = 8;
const BUFFER_SIZE: u64 = (ELEMENTS * std::mem::size_of::<u32>()) as u64;
const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const BYTES_PER_PIXEL: usize = 4;
const BYTES_PER_ROW: u32 = 256;
const ROW_BYTES: usize = WIDTH as usize * BYTES_PER_PIXEL;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;
const TRIANGLE_VERTICES: [f32; 6] = [-0.9, -0.9, 0.2, -0.9, -0.9, 0.2];

#[test]
#[ignore = "manual real-backend test"]
fn metal_msl_compute_fills_storage_buffer() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let source = r#"
#include <metal_stdlib>
using namespace metal;

struct Data {
    uint values[8];
};

kernel void cs(device Data& out_data [[buffer(0)]],
               uint3 id [[thread_position_in_grid]]) {
    uint i = id.x;
    if (i < 8) {
        out_data.values[i] = i * i;
    }
}
"#;
        let module = create_msl_module(
            device,
            source,
            &[msl_entry("cs", native::WGPUShaderStage_Compute, [4, 1, 1])],
        );
        let readback = run_compute_submit(device, module);
        let actual = read_u32_buffer(instance, readback);
        let expected = (0..ELEMENTS as u32)
            .map(|value| value * value)
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_msl_render_draws_constant_color_triangle() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let source = r#"
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut vs_main(const device packed_float2* positions [[buffer(0)]],
                         uint vertex_id [[vertex_id]]) {
    VertexOut out;
    out.position = float4(positions[vertex_id], 0.0, 1.0);
    return out;
}

fragment float4 fs_main() {
    return float4(1.0, 0.0, 0.0, 1.0);
}
"#;
        let module = create_msl_module(
            device,
            source,
            &[
                msl_entry("vs_main", native::WGPUShaderStage_Vertex, [0, 0, 0]),
                msl_entry("fs_main", native::WGPUShaderStage_Fragment, [0, 0, 0]),
            ],
        );
        let readback = run_render_submit(device, module);
        let pixels = read_unpacked_texture_buffer(instance, readback);

        assert!(contains_pixel(&pixels, [255, 0, 0, 255]));
        assert!(contains_pixel(&pixels, [26, 51, 77, 255]));
        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

fn msl_entry(
    name: &str,
    stage: native::WGPUShaderStage,
    workgroup_size: [u32; 3],
) -> YaWGPUMslEntryPoint {
    YaWGPUMslEntryPoint {
        name: string_view(name),
        stage,
        workgroupSize: workgroup_size,
    }
}

unsafe fn create_msl_module(
    device: native::WGPUDevice,
    source: &str,
    entry_points: &[YaWGPUMslEntryPoint],
) -> native::WGPUShaderModule {
    let descriptor = YaWGPUShaderModuleMslDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        code: string_view(source),
        entryPointCount: entry_points.len(),
        entryPoints: entry_points.as_ptr(),
    };
    let module = yawgpu::yawgpuDeviceCreateShaderModuleMsl(device, &descriptor);
    assert!(!module.is_null());
    module
}

unsafe fn run_compute_submit(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let output = create_buffer(
        device,
        BUFFER_SIZE,
        native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
    );
    let readback = create_buffer(
        device,
        BUFFER_SIZE,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let bind_group_layout = create_storage_bind_group_layout(device);
    let pipeline_layout = create_pipeline_layout(device, &[bind_group_layout]);
    let pipeline = create_compute_pipeline(device, module, pipeline_layout);
    let bind_group = create_storage_bind_group(device, bind_group_layout, output);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 2, 1, 1);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, BUFFER_SIZE);
    submit_encoder(queue, encoder);

    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
    yawgpu::wgpuBindGroupLayoutRelease(bind_group_layout);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuBufferRelease(output);
    yawgpu::wgpuQueueRelease(queue);
    readback
}

unsafe fn create_storage_bind_group_layout(
    device: native::WGPUDevice,
) -> native::WGPUBindGroupLayout {
    let entry = native::WGPUBindGroupLayoutEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        visibility: native::WGPUShaderStage_Compute,
        bindingArraySize: 0,
        buffer: native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_Storage,
            hasDynamicOffset: 0,
            minBindingSize: BUFFER_SIZE,
        },
        sampler: unsafe { std::mem::zeroed() },
        texture: unsafe { std::mem::zeroed() },
        storageTexture: unsafe { std::mem::zeroed() },
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
    module: native::WGPUShaderModule,
    layout: native::WGPUPipelineLayout,
) -> native::WGPUComputePipeline {
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("cs"),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

unsafe fn create_storage_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    output: native::WGPUBuffer,
) -> native::WGPUBindGroup {
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer: output,
        offset: 0,
        size: BUFFER_SIZE,
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
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!bind_group.is_null());
    bind_group
}

unsafe fn run_render_submit(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let vertex = create_buffer(
        device,
        vertex_buffer_size(),
        native::WGPUBufferUsage_Vertex | native::WGPUBufferUsage_CopyDst,
    );
    write_f32_buffer(queue, vertex, &TRIANGLE_VERTICES);
    let texture = create_texture(
        device,
        native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
    );
    let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
    let readback = create_buffer(
        device,
        READBACK_SIZE as u64,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let pipeline_layout = create_pipeline_layout(device, &[]);
    let pipeline = create_render_pipeline(device, module, pipeline_layout);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    record_render_pass(encoder, pipeline, vertex, view);
    record_t2b(encoder, texture, readback);
    submit_encoder(queue, encoder);

    yawgpu::wgpuRenderPipelineRelease(pipeline);
    yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuTextureViewRelease(view);
    yawgpu::wgpuTextureRelease(texture);
    yawgpu::wgpuBufferRelease(vertex);
    yawgpu::wgpuQueueRelease(queue);
    readback
}

unsafe fn create_render_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    layout: native::WGPUPipelineLayout,
) -> native::WGPURenderPipeline {
    let vertex_buffer = native::WGPUVertexBufferLayout {
        nextInChain: std::ptr::null_mut(),
        arrayStride: 8,
        stepMode: native::WGPUVertexStepMode_Vertex,
        attributeCount: 0,
        attributes: std::ptr::null(),
    };
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view("fs_main"),
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
            module,
            entryPoint: string_view("vs_main"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 1,
            buffers: &vertex_buffer,
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

unsafe fn record_render_pass(
    encoder: native::WGPUCommandEncoder,
    pipeline: native::WGPURenderPipeline,
    vertex: native::WGPUBuffer,
    view: native::WGPUTextureView,
) {
    let attachment = color_attachment(view);
    let descriptor = render_pass_descriptor(&[attachment]);
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 0, vertex_buffer_size());
    yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
}

unsafe fn record_t2b(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
) {
    let source = texture_copy_info(source);
    let destination = buffer_copy_info(destination);
    let size = texture_extent();
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &size);
}

unsafe fn submit_encoder(queue: native::WGPUQueue, encoder: native::WGPUCommandEncoder) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

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

unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: texture_extent(),
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

unsafe fn write_f32_buffer(queue: native::WGPUQueue, buffer: native::WGPUBuffer, values: &[f32]) {
    yawgpu::wgpuQueueWriteBuffer(
        queue,
        buffer,
        0,
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
    );
}

unsafe fn read_u32_buffer(instance: native::WGPUInstance, buffer: native::WGPUBuffer) -> Vec<u32> {
    read_buffer(instance, buffer, 0, BUFFER_SIZE as usize)
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|bytes| u32::from_ne_bytes(bytes.try_into().expect("u32 chunk")))
        .collect()
}

unsafe fn read_unpacked_texture_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
) -> Vec<u8> {
    let mapped = read_buffer(instance, buffer, 0, READBACK_SIZE);
    let mut pixels = vec![0; ROW_BYTES * HEIGHT as usize];
    for row in 0..HEIGHT as usize {
        let pixel_offset = row * ROW_BYTES;
        let padded_offset = row * BYTES_PER_ROW as usize;
        pixels[pixel_offset..pixel_offset + ROW_BYTES]
            .copy_from_slice(&mapped[padded_offset..padded_offset + ROW_BYTES]);
    }
    pixels
}

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
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.1,
            g: 0.2,
            b: 0.3,
            a: 1.0,
        },
    }
}

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

fn multisample_state() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 1,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
}

fn buffer_copy_info(buffer: native::WGPUBuffer) -> native::WGPUTexelCopyBufferInfo {
    native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        },
        buffer,
    }
}

fn texture_copy_info(texture: native::WGPUTexture) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    }
}

fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
    }
}

fn contains_pixel(pixels: &[u8], rgba: [u8; 4]) -> bool {
    pixels
        .chunks_exact(BYTES_PER_PIXEL)
        .any(|pixel| pixel == rgba)
}

fn vertex_buffer_size() -> u64 {
    std::mem::size_of_val(&TRIANGLE_VERTICES) as u64
}

unsafe fn create_metal_instance() -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: YAWGPU_INSTANCE_BACKEND_METAL,
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

unsafe fn install_error_capture(
    device: native::WGPUDevice,
) -> Arc<Mutex<Vec<yawgpu_core::DeviceError>>> {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let captured_errors = Arc::clone(&errors);
    yawgpu::testing_set_uncaptured_error_callback(
        device,
        Some(move |error| captured_errors.lock().expect("error lock").push(error)),
    );
    errors
}

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

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
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
