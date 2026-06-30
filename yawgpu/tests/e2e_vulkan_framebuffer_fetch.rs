//! Real-Vulkan e2e for `@color(N)` framebuffer fetch (TBDR re-home, Slice 1.4).
//!
//! Vulkan lowers `@color(0)` to a SubpassData input-attachment self-read; the HAL
//! wires a VkRenderPass self-dependency + INPUT_ATTACHMENT descriptor. The
//! attachment is cleared to (0.1, 0.2, 0.3, 1.0); on the triangle the fragment
//! must observe that cleared value and produce (0.6, 0.2, 0.3, 1.0).
//!
//! Gated on `vulkan` + `tiled`; run with:
//!   cargo test -p yawgpu --features vulkan,tiled --test e2e_vulkan_framebuffer_fetch -- --ignored

#![cfg(all(feature = "vulkan", feature = "tiled"))]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const BYTES_PER_PIXEL: usize = 4;
const BYTES_PER_ROW: u32 = 256;
const ROW_BYTES: usize = WIDTH as usize * BYTES_PER_PIXEL;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;

const TRIANGLE_VERTICES: [f32; 6] = [-0.9, -0.9, 0.2, -0.9, -0.9, 0.2];

const FRAMEBUFFER_FETCH_SHADER: &str = r#"
enable chromium_experimental_framebuffer_fetch;

@vertex
fn vs(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 0.0, 1.0);
}

@fragment
fn fs(@color(0) prev: vec4<f32>) -> @location(0) vec4<f32> {
    return prev + vec4<f32>(0.5, 0.0, 0.0, 0.0);
}
"#;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_framebuffer_fetch_reads_cleared_attachment() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        // MoltenVK (the only Vulkan ICD on macOS) does not map an input-attachment
        // self-read (the `@color(N)` framebuffer-fetch pattern: color attachment ==
        // input attachment, by-region self-dependency) to Metal's tile read; it reads
        // zero. The HAL wiring is spec-valid (validation-clean) and Metal's native
        // [[color(N)]] path is verified by e2e_metal_framebuffer_fetch — so this case
        // is left for native Vulkan hardware (Linux/Windows), where this test runs and
        // verifies. See specs/tracking/tbdr-tint-rehome-plan.md (Slice 1.4 MoltenVK gap).
        if cfg!(target_os = "macos") {
            eprintln!(
                "skipping: MoltenVK does not execute input-attachment framebuffer fetch \
                 (needs native Vulkan HW)"
            );
            return;
        }

        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let readback = run_render_submit(device, FRAMEBUFFER_FETCH_SHADER);
        let pixels = read_unpacked_texture_buffer(instance, readback);

        // Background = clear color (0.1, 0.2, 0.3, 1.0).
        assert!(
            contains_pixel(&pixels, [26, 51, 77, 255]),
            "expected cleared background pixel"
        );
        // Triangle = prev + 0.5 red = (0.6, 0.2, 0.3, 1.0): proves the fragment
        // read the cleared attachment value via @color(0) input-attachment.
        assert!(
            contains_pixel(&pixels, [153, 51, 77, 255]),
            "expected framebuffer-fetch (clear + red) pixel; got {:?}",
            distinct_pixels(&pixels)
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "device errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn run_render_submit(device: native::WGPUDevice, shader: &str) -> native::WGPUBuffer {
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
    let module = create_wgsl_module(device, shader);
    let pipeline = create_render_pipeline(device, module);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    record_render_pass(encoder, pipeline, vertex, view);
    record_t2b(encoder, texture, readback);
    submit_encoder(queue, encoder);

    yawgpu::wgpuRenderPipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuTextureViewRelease(view);
    yawgpu::wgpuTextureRelease(texture);
    yawgpu::wgpuBufferRelease(vertex);
    yawgpu::wgpuQueueRelease(queue);
    readback
}

unsafe fn record_render_pass(
    encoder: native::WGPUCommandEncoder,
    pipeline: native::WGPURenderPipeline,
    vertex: native::WGPUBuffer,
    view: native::WGPUTextureView,
) {
    let attachment = color_attachment(view);
    let attachments = [attachment];
    let descriptor = render_pass_descriptor(&attachments);
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

unsafe fn create_render_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let attribute = native::WGPUVertexAttribute {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUVertexFormat_Float32x2,
        offset: 0,
        shaderLocation: 0,
    };
    let vertex_buffer = native::WGPUVertexBufferLayout {
        nextInChain: std::ptr::null_mut(),
        arrayStride: 8,
        stepMode: native::WGPUVertexStepMode_Vertex,
        attributeCount: 1,
        attributes: &attribute,
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
            bufferCount: 1,
            buffers: &vertex_buffer,
        },
        primitive: primitive_state(),
        depthStencil: std::ptr::null(),
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null(), "framebuffer-fetch pipeline creation failed");
    pipeline
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
    assert!(!module.is_null(), "framebuffer-fetch shader module creation failed");
    module
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
    pixels.chunks_exact(BYTES_PER_PIXEL).any(|pixel| pixel == rgba)
}

fn distinct_pixels(pixels: &[u8]) -> Vec<[u8; 4]> {
    let mut seen: Vec<[u8; 4]> = Vec::new();
    for chunk in pixels.chunks_exact(BYTES_PER_PIXEL) {
        let px = [chunk[0], chunk[1], chunk[2], chunk[3]];
        if !seen.contains(&px) {
            seen.push(px);
        }
    }
    seen
}

fn vertex_buffer_size() -> u64 {
    std::mem::size_of_val(&TRIANGLE_VERTICES) as u64
}

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
