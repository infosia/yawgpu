#![cfg(feature = "gles")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu_test::{real_backend_available, wait, RealBackend};

mod common;
use common::create_gles_instance;

const ELEMENTS: usize = 8;
const BUFFER_SIZE: u64 = (ELEMENTS * std::mem::size_of::<u32>()) as u64;

#[test]
#[ignore = "manual real-backend test"]
fn gles_compute_fills_storage_buffer() {
    assert!(
        real_backend_available(RealBackend::Gles),
        "GLES backend not available; install an ANGLE build with ES 3.1 support \
         (Chrome / Edge ANGLE caps at ES 3.0; see specs/blocks/67-gles-backend.md). \
         Without a real GLES adapter this test cannot verify the GLES execution path."
    );

    unsafe {
        let instance = create_gles_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let shader = r#"
struct Data {
    values: array<u32, 8>,
}

@group(0) @binding(0) var<storage, read_write> out_data: Data;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < 8u) {
        out_data.values[i] = i * i;
    }
}
"#;
        let readback = run_compute_submit(device, shader, &[], &[0]);
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
fn gles_compute_reads_input_and_writes_output_storage_buffers() {
    assert!(
        real_backend_available(RealBackend::Gles),
        "GLES backend not available; install an ANGLE build with ES 3.1 support \
         (Chrome / Edge ANGLE caps at ES 3.0; see specs/blocks/67-gles-backend.md). \
         Without a real GLES adapter this test cannot verify the GLES execution path."
    );

    unsafe {
        let instance = create_gles_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let shader = r#"
struct Data {
    values: array<u32, 8>,
}

@group(0) @binding(0) var<storage, read> in_data: Data;
@group(0) @binding(1) var<storage, read_write> out_data: Data;

@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < 8u) {
        out_data.values[i] = in_data.values[i] + 1u;
    }
}
"#;
        let input = (0..ELEMENTS as u32)
            .map(|value| value * 3 + 2)
            .collect::<Vec<_>>();
        let readback = run_compute_submit(device, shader, &input, &[0, 1]);
        let actual = read_u32_buffer(instance, readback);
        let expected = input.iter().map(|value| value + 1).collect::<Vec<_>>();

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
fn gles_compute_reads_depth_stencil_aspect_textures() {
    assert!(
        real_backend_available(RealBackend::Gles),
        "GLES backend not available; install an ANGLE build with ES 3.1 support \
         (Chrome / Edge ANGLE caps at ES 3.0; see specs/blocks/67-gles-backend.md). \
         Without a real GLES adapter this test cannot verify the GLES execution path."
    );

    unsafe {
        let instance = create_gles_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let texture = create_depth_stencil_texture(device);
        clear_depth_stencil_texture(device, texture);

        let depth_view = create_aspect_view(texture, native::WGPUTextureAspect_DepthOnly);
        let depth_readback = run_texture_readback_compute(
            device,
            depth_view,
            r#"
@group(0) @binding(0) var src: texture_depth_2d;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(1)
fn main() {
    dst[0] = textureLoad(src, vec2<i32>(0, 0), 0);
}
"#,
        );
        let depth = f32::from_ne_bytes(
            read_buffer(instance, depth_readback, 0, 4)
                .try_into()
                .expect("depth readback is four bytes"),
        );
        assert!(
            (0.0..=1.0).contains(&depth),
            "depth readback should be normalized, got {depth}"
        );

        let stencil_view = create_aspect_view(texture, native::WGPUTextureAspect_StencilOnly);
        let stencil_readback = run_texture_readback_compute(
            device,
            stencil_view,
            r#"
@group(0) @binding(0) var src: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(1)
fn main() {
    dst[0] = textureLoad(src, vec2<i32>(0, 0), 0).r;
}
"#,
        );
        let stencil = read_u32_buffer(instance, stencil_readback)[0];
        assert_eq!(stencil, 7);

        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuBufferRelease(stencil_readback);
        yawgpu::wgpuTextureViewRelease(stencil_view);
        yawgpu::wgpuBufferRelease(depth_readback);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn gles_compute_reads_stencil_after_render_pass_clear() {
    assert!(
        real_backend_available(RealBackend::Gles),
        "GLES backend not available; install an ANGLE build with ES 3.1 support \
         (Chrome / Edge ANGLE caps at ES 3.0; see specs/blocks/67-gles-backend.md). \
         Without a real GLES adapter this test cannot verify the GLES execution path."
    );

    unsafe {
        let instance = create_gles_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let texture = create_depth_stencil_texture(device);
        clear_depth_stencil_texture(device, texture);
        let stencil_view = create_aspect_view(texture, native::WGPUTextureAspect_StencilOnly);
        let stencil_readback = run_texture_readback_compute(
            device,
            stencil_view,
            r#"
@group(0) @binding(0) var src: texture_2d<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(1)
fn main() {
    dst[0] = textureLoad(src, vec2<i32>(0, 0), 0).r;
}
"#,
        );

        let stencil = read_u32_buffer(instance, stencil_readback)[0];
        assert_eq!(stencil, 7, "stencil value after render-pass clear");
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBufferRelease(stencil_readback);
        yawgpu::wgpuTextureViewRelease(stencil_view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn run_compute_submit(
    device: native::WGPUDevice,
    shader: &str,
    input_values: &[u32],
    binding_order: &[u32],
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let input = (!input_values.is_empty()).then(|| {
        create_buffer(
            device,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopyDst,
        )
    });
    let output = create_buffer(
        device,
        native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
    );
    let readback = create_buffer(
        device,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    if let Some(input) = input {
        write_u32_buffer(queue, input, input_values);
    }

    let module = create_wgsl_module(device, shader);
    let pipeline = create_compute_pipeline(device, module);
    let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    let bind_group = create_bind_group(device, layout, input, output, binding_order);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 2, 1, 1);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, BUFFER_SIZE);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);

    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuBufferRelease(output);
    if let Some(input) = input {
        yawgpu::wgpuBufferRelease(input);
    }
    yawgpu::wgpuQueueRelease(queue);
    readback
}

unsafe fn run_texture_readback_compute(
    device: native::WGPUDevice,
    view: native::WGPUTextureView,
    shader: &str,
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let output = create_buffer(
        device,
        native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
    );
    let readback = create_buffer(
        device,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let module = create_wgsl_module(device, shader);
    let pipeline = create_compute_pipeline(device, module);
    let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    let bind_group = create_texture_readback_bind_group(device, layout, view, output);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, BUFFER_SIZE);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);

    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuBufferRelease(output);
    yawgpu::wgpuQueueRelease(queue);
    readback
}

unsafe fn create_texture_readback_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    view: native::WGPUTextureView,
    output: native::WGPUBuffer,
) -> native::WGPUBindGroup {
    let entries = [
        native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 0,
            buffer: std::ptr::null(),
            offset: 0,
            size: 0,
            sampler: std::ptr::null(),
            textureView: view,
        },
        native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 1,
            buffer: output,
            offset: 0,
            size: BUFFER_SIZE,
            sampler: std::ptr::null(),
            textureView: std::ptr::null(),
        },
    ];
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

unsafe fn create_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    input: Option<native::WGPUBuffer>,
    output: native::WGPUBuffer,
    binding_order: &[u32],
) -> native::WGPUBindGroup {
    let entries = binding_order
        .iter()
        .map(|binding| native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: *binding,
            buffer: match (input, *binding) {
                (Some(input), 0) => input,
                _ => output,
            },
            offset: 0,
            size: BUFFER_SIZE,
            sampler: std::ptr::null(),
            textureView: std::ptr::null(),
        })
        .collect::<Vec<_>>();
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

unsafe fn create_buffer(
    device: native::WGPUDevice,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size: BUFFER_SIZE,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

unsafe fn create_depth_stencil_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_TextureBinding,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        },
        format: native::WGPUTextureFormat_Depth24PlusStencil8,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

unsafe fn create_aspect_view(
    texture: native::WGPUTexture,
    aspect: native::WGPUTextureAspect,
) -> native::WGPUTextureView {
    let descriptor = native::WGPUTextureViewDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        format: native::WGPUTextureFormat_Undefined,
        dimension: native::WGPUTextureViewDimension_Undefined,
        baseMipLevel: 0,
        mipLevelCount: 1,
        baseArrayLayer: 0,
        arrayLayerCount: 1,
        aspect,
        usage: native::WGPUTextureUsage_TextureBinding,
    };
    let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
    assert!(!view.is_null());
    view
}

unsafe fn clear_depth_stencil_texture(device: native::WGPUDevice, texture: native::WGPUTexture) {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
    assert!(!view.is_null());
    let attachment = native::WGPURenderPassDepthStencilAttachment {
        nextInChain: std::ptr::null_mut(),
        view,
        depthLoadOp: native::WGPULoadOp_Clear,
        depthStoreOp: native::WGPUStoreOp_Store,
        depthReadOnly: 0,
        depthClearValue: 0.5,
        stencilLoadOp: native::WGPULoadOp_Clear,
        stencilStoreOp: native::WGPUStoreOp_Store,
        stencilReadOnly: 0,
        stencilClearValue: 7,
    };
    let descriptor = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: 0,
        colorAttachments: std::ptr::null(),
        depthStencilAttachment: &attachment,
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuRenderPassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuTextureViewRelease(view);
    yawgpu::wgpuQueueRelease(queue);
}

unsafe fn write_u32_buffer(queue: native::WGPUQueue, buffer: native::WGPUBuffer, values: &[u32]) {
    yawgpu::wgpuQueueWriteBuffer(
        queue,
        buffer,
        0,
        values.as_ptr().cast(),
        std::mem::size_of_val(values),
    );
}

unsafe fn read_u32_buffer(instance: native::WGPUInstance, buffer: native::WGPUBuffer) -> Vec<u32> {
    /// Width of one `u32` in the readback byte stream, as a const-generic argument.
    const U32_BYTES: usize = std::mem::size_of::<u32>();

    let bytes = read_buffer(instance, buffer, 0, BUFFER_SIZE as usize);
    bytes
        .as_chunks::<U32_BYTES>()
        .0
        .iter()
        .map(|chunk| u32::from_ne_bytes(*chunk))
        .collect()
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

unsafe fn request_adapter(instance: native::WGPUInstance) -> native::WGPUAdapter {
    let options = native::WGPURequestAdapterOptions {
        nextInChain: std::ptr::null_mut(),
        featureLevel: native::WGPUFeatureLevel_Undefined,
        powerPreference: native::WGPUPowerPreference_Undefined,
        forceFallbackAdapter: 0,
        backendType: native::WGPUBackendType_Undefined,
        compatibleSurface: std::ptr::null(),
    };
    let mut adapter: native::WGPUAdapter = std::ptr::null();
    let callback_info = native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_adapter_callback),
        userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuInstanceRequestAdapter(instance, &options, callback_info);
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

/// Propagation audit 2026-11 candidate 08: two sampled views of one texture
/// that differ only in mip range share one GL texture name, so each view must
/// still read its own level within a single dispatch.
#[test]
#[ignore = "manual real-backend test"]
fn gles_compute_reads_two_mip_views_of_one_texture() {
    assert!(
        real_backend_available(RealBackend::Gles),
        "GLES backend not available; this test needs a real GLES 3.1 adapter."
    );

    const SHADER: &str = r#"
@group(0) @binding(0) var first: texture_2d<f32>;
@group(0) @binding(1) var second: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(1)
fn main() {
    dst[0] = u32(textureLoad(first, vec2<i32>(0, 0), 0).r * 255.0 + 0.5);
    dst[1] = u32(textureLoad(second, vec2<i32>(0, 0), 0).r * 255.0 + 0.5);
    dst[2] = textureDimensions(first).x;
    dst[3] = textureDimensions(second).x;
}
"#;
    const MIP0_RED: u8 = 10;
    const MIP1_RED: u8 = 200;

    unsafe {
        let instance = create_gles_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let texture = create_two_mip_texture(device);
        write_mip(queue, texture, 0, 2, MIP0_RED);
        write_mip(queue, texture, 1, 1, MIP1_RED);
        let mip0 = create_single_mip_view(texture, 0);
        let mip1 = create_single_mip_view(texture, 1);

        // Both binding orders: whichever view binds last must not win for both.
        for (views, expected) in [
            (
                [mip0, mip1],
                [u32::from(MIP0_RED), u32::from(MIP1_RED), 2, 1],
            ),
            (
                [mip1, mip0],
                [u32::from(MIP1_RED), u32::from(MIP0_RED), 1, 2],
            ),
        ] {
            let readback = run_two_view_readback_compute(device, views, SHADER);
            if lacks_texture_view(&errors) {
                eprintln!("skipping: this GLES host has no glTextureView");
                return;
            }
            let actual = read_u32_buffer(instance, readback);
            assert_eq!(
                actual[..4],
                expected,
                "each view must read its own mip level (red, red, width, width)"
            );
            yawgpu::wgpuBufferRelease(readback);
        }

        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuTextureViewRelease(mip1);
        yawgpu::wgpuTextureViewRelease(mip0);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Whether a submit failed with the Tier 2 `HalError` for a host without
/// `glTextureView` (ES 3.1 without `OES`/`EXT_texture_view`), where a second
/// mip range of one texture cannot be bound.
fn lacks_texture_view(errors: &Mutex<Vec<yawgpu_core::DeviceError>>) -> bool {
    errors
        .lock()
        .expect("error lock")
        .iter()
        .any(|error| error.message.contains("glTextureView"))
}

unsafe fn create_two_mip_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    create_two_mip_texture_with_usage(
        device,
        native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_TextureBinding,
    )
}

unsafe fn create_two_mip_texture_with_usage(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 2,
            height: 2,
            depthOrArrayLayers: 1,
        },
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 2,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

unsafe fn write_mip(
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    mip_level: u32,
    size: u32,
    red: u8,
) {
    let pixels: Vec<u8> = (0..size * size).flat_map(|_| [red, 0, 0, 255]).collect();
    let destination = native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: mip_level,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let layout = native::WGPUTexelCopyBufferLayout {
        offset: 0,
        bytesPerRow: size * 4,
        rowsPerImage: size,
    };
    let extent = native::WGPUExtent3D {
        width: size,
        height: size,
        depthOrArrayLayers: 1,
    };
    yawgpu::wgpuQueueWriteTexture(
        queue,
        &destination,
        pixels.as_ptr().cast(),
        pixels.len(),
        &layout,
        &extent,
    );
}

unsafe fn create_single_mip_view(
    texture: native::WGPUTexture,
    base_mip_level: u32,
) -> native::WGPUTextureView {
    let descriptor = native::WGPUTextureViewDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        format: native::WGPUTextureFormat_Undefined,
        dimension: native::WGPUTextureViewDimension_Undefined,
        baseMipLevel: base_mip_level,
        mipLevelCount: 1,
        baseArrayLayer: 0,
        arrayLayerCount: 1,
        aspect: native::WGPUTextureAspect_All,
        // Inherit the texture's usage, so one view serves sampled and storage bindings.
        usage: native::WGPUTextureUsage_None,
    };
    let view = yawgpu::wgpuTextureCreateView(texture, &descriptor);
    assert!(!view.is_null());
    view
}

unsafe fn run_two_view_readback_compute(
    device: native::WGPUDevice,
    views: [native::WGPUTextureView; 2],
    shader: &str,
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let output = create_buffer(
        device,
        native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
    );
    let readback = create_buffer(
        device,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let module = create_wgsl_module(device, shader);
    let pipeline = create_compute_pipeline(device, module);
    let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    let texture_entry = |binding: u32, view| native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer: std::ptr::null(),
        offset: 0,
        size: 0,
        sampler: std::ptr::null(),
        textureView: view,
    };
    let entries = [
        texture_entry(0, views[0]),
        texture_entry(1, views[1]),
        native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 2,
            buffer: output,
            offset: 0,
            size: BUFFER_SIZE,
            sampler: std::ptr::null(),
            textureView: std::ptr::null(),
        },
    ];
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!bind_group.is_null());

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, BUFFER_SIZE);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);

    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuBufferRelease(output);
    yawgpu::wgpuQueueRelease(queue);
    readback
}

/// Block 169 S2: a storage write to a mip level outside the range of a sampled
/// view of the same texture must land, both when the sampled view is bound in
/// the same dispatch and when an earlier dispatch left its mip bounds behind.
#[test]
#[ignore = "manual real-backend test"]
fn gles_compute_stores_to_mip_outside_sampled_view_range() {
    assert!(
        real_backend_available(RealBackend::Gles),
        "GLES backend not available; this test needs a real GLES 3.1 adapter."
    );

    const COPY_SHADER: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1)
fn main() {
    textureStore(dst, vec2<i32>(0, 0), textureLoad(src, vec2<i32>(0, 0), 0));
}
"#;
    const SAMPLE_SHADER: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(1)
fn main() {
    dst[0] = u32(textureLoad(src, vec2<i32>(0, 0), 0).r * 255.0 + 0.5);
}
"#;
    const STORE_SHADER: &str = r#"
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1)
fn main() {
    textureStore(dst, vec2<i32>(0, 0), vec4<f32>(1.0, 0.0, 1.0, 1.0));
}
"#;
    const USAGE: native::WGPUTextureUsage = native::WGPUTextureUsage_CopyDst
        | native::WGPUTextureUsage_CopySrc
        | native::WGPUTextureUsage_TextureBinding
        | native::WGPUTextureUsage_StorageBinding;

    unsafe {
        let instance = create_gles_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // Separate dispatches: sample mip 0 first, then only store to mip 1. This
        // half needs no `glTextureView`, so it runs before the skip below.
        let texture = create_two_mip_texture_with_usage(device, USAGE);
        write_mip(queue, texture, 0, 2, 10);
        write_mip(queue, texture, 1, 1, 200);
        let mip0 = create_single_mip_view(texture, 0);
        let mip1 = create_single_mip_view(texture, 1);
        let sampled = run_texture_readback_compute(device, mip0, SAMPLE_SHADER);
        assert_eq!(read_u32_buffer(instance, sampled)[0], 10);
        run_store_compute(device, STORE_SHADER, None, mip1);
        assert_eq!(
            read_mip1_pixel(instance, device, texture),
            [255, 0, 255, 255],
            "a store to mip 1 after an earlier dispatch sampled mip 0 must land"
        );
        yawgpu::wgpuBufferRelease(sampled);
        yawgpu::wgpuTextureViewRelease(mip1);
        yawgpu::wgpuTextureViewRelease(mip0);
        yawgpu::wgpuTextureRelease(texture);

        // Same dispatch: sample mip 0, store it to mip 1.
        let texture = create_two_mip_texture_with_usage(device, USAGE);
        write_mip(queue, texture, 0, 2, 10);
        write_mip(queue, texture, 1, 1, 200);
        let mip0 = create_single_mip_view(texture, 0);
        let mip1 = create_single_mip_view(texture, 1);
        run_store_compute(device, COPY_SHADER, Some(mip0), mip1);
        if lacks_texture_view(&errors) {
            eprintln!("skipping: this GLES host has no glTextureView");
            return;
        }
        assert_eq!(
            read_mip1_pixel(instance, device, texture),
            [10, 0, 0, 255],
            "a store to mip 1 next to a sampled mip-0 view must land"
        );
        yawgpu::wgpuTextureViewRelease(mip1);
        yawgpu::wgpuTextureViewRelease(mip0);
        yawgpu::wgpuTextureRelease(texture);

        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Runs one dispatch with an optional sampled view at binding 0 and a storage
/// view at binding 1.
unsafe fn run_store_compute(
    device: native::WGPUDevice,
    shader: &str,
    sampled: Option<native::WGPUTextureView>,
    storage: native::WGPUTextureView,
) {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let module = create_wgsl_module(device, shader);
    let pipeline = create_compute_pipeline(device, module);
    let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    let entries: Vec<_> = sampled
        .into_iter()
        .map(|view| (0, view))
        .chain([(1, storage)])
        .map(|(binding, view)| native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding,
            buffer: std::ptr::null(),
            offset: 0,
            size: 0,
            sampler: std::ptr::null(),
            textureView: view,
        })
        .collect();
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!bind_group.is_null());

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);

    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuQueueRelease(queue);
}

/// Copies the single texel of mip 1 to a buffer and returns it.
unsafe fn read_mip1_pixel(
    instance: native::WGPUInstance,
    device: native::WGPUDevice,
    texture: native::WGPUTexture,
) -> [u8; 4] {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let readback = create_buffer(
        device,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let source = native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 1,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let destination = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: 256,
            rowsPerImage: 1,
        },
        buffer: readback,
    };
    let extent = native::WGPUExtent3D {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
    };
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &extent);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuQueueRelease(queue);

    let pixel = read_buffer(instance, readback, 0, 4)
        .try_into()
        .expect("one RGBA8 texel");
    yawgpu::wgpuBufferRelease(readback);
    pixel
}
