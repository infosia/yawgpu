//! Real-Metal regression for Block 33 slice B1: raw **MSL passthrough** compute.
//!
//! A `WGPUShaderModule` is created from hand-written MSL (no WGSL, no Tint) via
//! the vendor `YaWGPUShaderSourceMSL` chain, then used in a compute pipeline with
//! an **explicit** pipeline layout. The kernel doubles every element of a storage
//! buffer bound at Metal `[[buffer(0)]]` — the slot yawgpu's deterministic
//! `metal_buffer_binding_map` assigns to `(group 0, binding 0)` for a compute
//! layout. The compute workgroup size is supplied through the entry-point
//! metadata (there is no reflection to recover it from). Proves MP4 end-to-end.

#![cfg(all(feature = "metal", feature = "shader-passthrough"))]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YaWGPUMslEntryPoint, YaWGPUShaderSourceMSL,
    YAWGPU_INSTANCE_BACKEND_METAL, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
    YAWGPU_STYPE_SHADER_SOURCE_MSL,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const ELEMENTS: usize = 8;
const BUFFER_SIZE: u64 = (ELEMENTS * std::mem::size_of::<u32>()) as u64;

// Hand-written MSL: double each u32 in the storage buffer at [[buffer(0)]].
// One workgroup of 8 threads covers the 8 elements (no bounds check needed).
const DOUBLE_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;

kernel void double_values(device uint* data [[buffer(0)]],
                          uint gid [[thread_position_in_grid]]) {
    data[gid] = data[gid] * 2u;
}
"#;

#[test]
#[ignore = "manual real-backend test"]
fn metal_msl_passthrough_compute_doubles_storage_buffer() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let input: Vec<u32> = (1..=ELEMENTS as u32).collect();
        let buffer = create_buffer_sized(
            device,
            BUFFER_SIZE,
            native::WGPUBufferUsage_Storage
                | native::WGPUBufferUsage_CopySrc
                | native::WGPUBufferUsage_CopyDst,
        );
        write_u32_buffer(queue, buffer, &input);

        let bgl = create_storage_bgl(device);
        let pipeline_layout = create_pipeline_layout(device, bgl);
        let module =
            create_msl_module(device, DOUBLE_MSL, "double_values", [ELEMENTS as u32, 1, 1]);
        let pipeline =
            create_compute_pipeline_with_layout(device, module, pipeline_layout, "double_values");
        let bind_group = create_single_storage_bind_group(device, bgl, buffer);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);

        let readback = create_buffer_sized(
            device,
            BUFFER_SIZE,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, buffer, 0, readback, 0, BUFFER_SIZE);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let actual = read_u32_buffer(instance, readback);
        let expected: Vec<u32> = input.iter().map(|v| v * 2).collect();
        assert_eq!(
            actual, expected,
            "raw MSL passthrough compute did not double the storage buffer"
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected device errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuBufferRelease(buffer);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

const RENDER_W: u32 = 4;
const RENDER_H: u32 = 4;
const RENDER_BPR: u32 = 256;
const RENDER_READBACK: usize = RENDER_BPR as usize * RENDER_H as usize;

// Hand-written MSL render: a full-screen triangle from [[vertex_id]] (no vertex
// buffers, no bind groups) whose fragment outputs a solid color. Two entry
// points in one module — vmain (vertex) + fmain (fragment).
const SOLID_RENDER_MSL: &str = r#"#include <metal_stdlib>
using namespace metal;

struct VOut { float4 position [[position]]; };

vertex VOut vmain(uint vid [[vertex_id]]) {
    float2 p;
    switch (vid) {
        case 0: p = float2(-1.0, -1.0); break;
        case 1: p = float2(3.0, -1.0); break;
        default: p = float2(-1.0, 3.0); break;
    }
    VOut o;
    o.position = float4(p, 0.0, 1.0);
    return o;
}

fragment float4 fmain() {
    return float4(0.25, 0.5, 0.75, 1.0);
}
"#;

#[test]
#[ignore = "manual real-backend test"]
fn metal_msl_passthrough_render_draws_solid_color() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let color = create_color_texture(device);
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());
        let readback = create_buffer_sized(
            device,
            RENDER_READBACK as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let module = create_msl_render_module(device, SOLID_RENDER_MSL);
        let layout = create_empty_pipeline_layout(device);
        let pipeline = create_render_pipeline_passthrough(device, module, layout);

        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: color_view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        };
        let attachments = [color_attachment];
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: attachments.len(),
            colorAttachments: attachments.as_ptr(),
            depthStencilAttachment: std::ptr::null(),
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        let src = native::WGPUTexelCopyTextureInfo {
            texture: color,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let dst = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: RENDER_BPR,
                rowsPerImage: RENDER_H,
            },
            buffer: readback,
        };
        let extent = native::WGPUExtent3D {
            width: RENDER_W,
            height: RENDER_H,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let pixels = read_bytes(instance, readback, RENDER_READBACK);
        // 0.25, 0.5, 0.75, 1.0 → RGBA8 (64, 128, 191, 255); allow ±2 rounding.
        let near = |a: u8, b: u8| (i32::from(a) - i32::from(b)).abs() <= 2;
        assert!(
            near(pixels[0], 64) && near(pixels[1], 128) && near(pixels[2], 191) && pixels[3] == 255,
            "raw MSL passthrough render produced the wrong pixel: {:?}",
            &pixels[0..4]
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected device errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn create_msl_render_module(
    device: native::WGPUDevice,
    source: &str,
) -> native::WGPUShaderModule {
    let entries = [
        YaWGPUMslEntryPoint {
            name: string_view("vmain"),
            stage: native::WGPUShaderStage_Vertex,
            workgroupSize: [1, 1, 1],
        },
        YaWGPUMslEntryPoint {
            name: string_view("fmain"),
            stage: native::WGPUShaderStage_Fragment,
            workgroupSize: [1, 1, 1],
        },
    ];
    let mut msl = YaWGPUShaderSourceMSL {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_SHADER_SOURCE_MSL,
        },
        code: string_view(source),
        entryPointCount: entries.len(),
        entryPoints: entries.as_ptr(),
    };
    let descriptor = native::WGPUShaderModuleDescriptor {
        nextInChain: (&mut msl.chain) as *mut _,
        label: empty_string_view(),
    };
    let module = yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor);
    assert!(!module.is_null());
    module
}

unsafe fn create_empty_pipeline_layout(device: native::WGPUDevice) -> native::WGPUPipelineLayout {
    let descriptor = native::WGPUPipelineLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        bindGroupLayoutCount: 0,
        bindGroupLayouts: std::ptr::null(),
        immediateSize: 0,
    };
    let layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn create_render_pipeline_passthrough(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    layout: native::WGPUPipelineLayout,
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
        entryPoint: string_view("fmain"),
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
            entryPoint: string_view("vmain"),
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

unsafe fn create_color_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: RENDER_W,
            height: RENDER_H,
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

unsafe fn read_bytes(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
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
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, len, callback_info);
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, len);
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

unsafe fn create_msl_module(
    device: native::WGPUDevice,
    source: &str,
    entry: &str,
    workgroup_size: [u32; 3],
) -> native::WGPUShaderModule {
    let entry_point = YaWGPUMslEntryPoint {
        name: string_view(entry),
        stage: native::WGPUShaderStage_Compute,
        workgroupSize: workgroup_size,
    };
    let mut msl = YaWGPUShaderSourceMSL {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_SHADER_SOURCE_MSL,
        },
        code: string_view(source),
        entryPointCount: 1,
        entryPoints: &entry_point,
    };
    let descriptor = native::WGPUShaderModuleDescriptor {
        nextInChain: (&mut msl.chain) as *mut _,
        label: empty_string_view(),
    };
    let module = yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor);
    assert!(!module.is_null());
    module
}

unsafe fn create_storage_bgl(device: native::WGPUDevice) -> native::WGPUBindGroupLayout {
    let mut entry: native::WGPUBindGroupLayoutEntry = std::mem::zeroed();
    entry.binding = 0;
    entry.visibility = native::WGPUShaderStage_Compute;
    entry.buffer.type_ = native::WGPUBufferBindingType_Storage;
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: 1,
        entries: &entry,
    };
    let bgl = yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor);
    assert!(!bgl.is_null());
    bgl
}

unsafe fn create_pipeline_layout(
    device: native::WGPUDevice,
    bgl: native::WGPUBindGroupLayout,
) -> native::WGPUPipelineLayout {
    let descriptor = native::WGPUPipelineLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        bindGroupLayoutCount: 1,
        bindGroupLayouts: &bgl,
        immediateSize: 0,
    };
    let layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn create_compute_pipeline_with_layout(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
    layout: native::WGPUPipelineLayout,
    entry: &str,
) -> native::WGPUComputePipeline {
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view(entry),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

unsafe fn create_single_storage_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    buffer: native::WGPUBuffer,
) -> native::WGPUBindGroup {
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer,
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

unsafe fn create_buffer_sized(
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
        0,
        BUFFER_SIZE as usize,
        callback_info,
    );
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, BUFFER_SIZE as usize);
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), BUFFER_SIZE as usize).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_ne_bytes(chunk.try_into().expect("chunk is four bytes")))
        .collect()
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
