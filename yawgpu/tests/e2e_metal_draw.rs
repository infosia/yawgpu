//! Real-GPU Metal regression for F-034: `drawIndexed`, `drawIndirect`, and
//! `drawIndexedIndirect` must actually execute — both rasterize and run their
//! fragment-stage `read_write` storage write. Before F-034 these were
//! validation-only stubs that emitted no HAL command, so nothing ran (the
//! fragment storage write read back 0). Each probe renders a full-screen quad
//! whose fragment writes `result = 1u` and outputs green, then reads back BOTH
//! the storage buffer (the write happened) and the colour (it rasterized).
//!
//! The F-034 root cause is in the HAL-agnostic core draw path, so a Metal probe
//! exercises the shared recording + a Tier-1 backend; the CTS `rendering/draw`
//! port covers Metal and Vulkan. Gated on the `metal` feature.
#![cfg(feature = "metal")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
#[cfg(feature = "metal")]
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::wait;
#[cfg(feature = "metal")]
use yawgpu_test::{real_backend_skip_reason, RealBackend};

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const BYTES_PER_ROW: u32 = 256;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;

// A full-screen quad (6 vertices from the builtin index) whose fragment writes a
// `read_write` storage buffer and outputs green. The storage write is the F-034
// signature; the green output proves the draw rasterized.
const DRAW_SHADER: &str = r#"
struct Result { value: u32 }
@group(0) @binding(0) var<storage, read_write> result: Result;
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    result.value = 1u;
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
"#;

/// `drawIndexed` must rasterize + run the fragment storage write.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_draw_indexed_writes_storage() {
    run_draw_variant(DrawVariant::Indexed);
}

/// `drawIndirect` (non-indexed) must rasterize + run the fragment storage write.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_draw_indirect_writes_storage() {
    run_draw_variant(DrawVariant::Indirect);
}

/// `drawIndexedIndirect` must rasterize + run the fragment storage write.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_draw_indexed_indirect_writes_storage() {
    run_draw_variant(DrawVariant::IndexedIndirect);
}

#[cfg(feature = "metal")]
#[derive(Clone, Copy)]
enum DrawVariant {
    Indexed,
    Indirect,
    IndexedIndirect,
}

#[cfg(feature = "metal")]
fn run_draw_variant(variant: DrawVariant) {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // Storage buffer (init 0), colour target, and readbacks.
        let storage = create_buffer(
            device,
            4,
            native::WGPUBufferUsage_Storage
                | native::WGPUBufferUsage_CopySrc
                | native::WGPUBufferUsage_CopyDst,
        );
        let zero = [0u8; 4];
        yawgpu::wgpuQueueWriteBuffer(queue, storage, 0, zero.as_ptr().cast(), zero.len());
        let storage_readback = create_buffer(
            device,
            4,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let color = create_color_texture(device);
        let color_readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let module = create_wgsl_module(device, DRAW_SHADER);
        let pipeline = create_draw_pipeline(device, module);
        let bind_group = create_storage_bind_group(device, pipeline, storage);
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());

        // 6 indices [0..6) for the indexed variants; indirect args per variant.
        let indices: [u32; 6] = [0, 1, 2, 3, 4, 5];
        let index_buffer = create_buffer(
            device,
            std::mem::size_of_val(&indices) as u64,
            native::WGPUBufferUsage_Index | native::WGPUBufferUsage_CopyDst,
        );
        yawgpu::wgpuQueueWriteBuffer(
            queue,
            index_buffer,
            0,
            indices.as_ptr().cast(),
            std::mem::size_of_val(&indices),
        );
        // drawIndirect args: [vertexCount, instanceCount, firstVertex, firstInstance]
        // drawIndexedIndirect args: [indexCount, instanceCount, firstIndex, baseVertex, firstInstance]
        let indirect_args: [u32; 5] = [6, 1, 0, 0, 0];
        let indirect_buffer = create_buffer(
            device,
            std::mem::size_of_val(&indirect_args) as u64,
            native::WGPUBufferUsage_Indirect | native::WGPUBufferUsage_CopyDst,
        );
        yawgpu::wgpuQueueWriteBuffer(
            queue,
            indirect_buffer,
            0,
            indirect_args.as_ptr().cast(),
            std::mem::size_of_val(&indirect_args),
        );

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
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        match variant {
            DrawVariant::Indexed => {
                yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                    pass,
                    index_buffer,
                    native::WGPUIndexFormat_Uint32,
                    0,
                    std::mem::size_of_val(&indices) as u64,
                );
                yawgpu::wgpuRenderPassEncoderDrawIndexed(pass, 6, 1, 0, 0, 0);
            }
            DrawVariant::Indirect => {
                yawgpu::wgpuRenderPassEncoderDrawIndirect(pass, indirect_buffer, 0);
            }
            DrawVariant::IndexedIndirect => {
                yawgpu::wgpuRenderPassEncoderSetIndexBuffer(
                    pass,
                    index_buffer,
                    native::WGPUIndexFormat_Uint32,
                    0,
                    std::mem::size_of_val(&indices) as u64,
                );
                yawgpu::wgpuRenderPassEncoderDrawIndexedIndirect(pass, indirect_buffer, 0);
            }
        }
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        // Read back the storage write and the colour in the same submission.
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, storage_readback, 0, 4);
        let color_src = native::WGPUTexelCopyTextureInfo {
            texture: color,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let color_dst = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: BYTES_PER_ROW,
                rowsPerImage: HEIGHT,
            },
            buffer: color_readback,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &color_src,
            &color_dst,
            &texture_extent(),
        );
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let storage_bytes = read_buffer(instance, storage_readback, 0, 4);
        let storage_value = u32::from_le_bytes([
            storage_bytes[0],
            storage_bytes[1],
            storage_bytes[2],
            storage_bytes[3],
        ]);
        assert_eq!(
            storage_value, 1,
            "fragment storage write did not take effect on the draw variant"
        );

        let pixels = read_buffer(instance, color_readback, 0, READBACK_SIZE);
        assert!(
            pixels[0] == 0 && pixels[1] == 255 && pixels[2] == 0 && pixels[3] == 255,
            "draw variant did not rasterize green (first pixel): {:?}",
            &pixels[0..4]
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBufferRelease(indirect_buffer);
        yawgpu::wgpuBufferRelease(index_buffer);
        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuBufferRelease(color_readback);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuBufferRelease(storage_readback);
        yawgpu::wgpuBufferRelease(storage);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[cfg(feature = "metal")]
unsafe fn create_draw_pipeline(
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

#[cfg(feature = "metal")]
unsafe fn create_storage_bind_group(
    device: native::WGPUDevice,
    pipeline: native::WGPURenderPipeline,
    storage: native::WGPUBuffer,
) -> native::WGPUBindGroup {
    let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer: storage,
        offset: 0,
        size: 4,
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
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    bind_group
}

#[cfg(feature = "metal")]
unsafe fn create_color_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
    }
}

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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
