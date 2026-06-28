//! Real-Vulkan regression for Block 33 slice B2: raw **SPIR-V passthrough** compute.
//!
//! A `WGPUShaderModule` is created from precompiled SPIR-V (no WGSL, no Tint) via
//! the standard `WGPUShaderSourceSPIRV` chain, then used in a compute pipeline
//! with an **explicit** pipeline layout. The kernel doubles every element of a
//! storage buffer at `set = 0, binding = 0` (yawgpu maps WebGPU group→set,
//! binding→binding). No reflection — the words pass to the driver verbatim and
//! the workgroup `LocalSize` is baked into the SPIR-V. Proves SP2 end-to-end.
//!
//! The SPIR-V below was compiled offline with `glslangValidator -V` from:
//! ```glsl
//! #version 450
//! layout(local_size_x = 8) in;
//! layout(set = 0, binding = 0) buffer Data { uint data[]; };
//! void main() { uint i = gl_GlobalInvocationID.x; data[i] = data[i] * 2u; }
//! ```

#![cfg(all(feature = "vulkan", feature = "shader-passthrough"))]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const ELEMENTS: usize = 8;
const BUFFER_SIZE: u64 = (ELEMENTS * std::mem::size_of::<u32>()) as u64;

// `double.comp` → SPIR-V (220 words). Magic word 0x07230203 = 119734787.
#[rustfmt::skip]
const DOUBLE_SPIRV: &[u32] = &[
    119734787, 65536, 524299, 33, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
    808793134, 0, 196622, 0, 1, 393231, 5, 4, 1852399981, 0, 11, 393232, 4, 17, 8, 1,
    1, 196611, 2, 450, 262149, 4, 1852399981, 0, 196613, 8, 105, 524293, 11,
    1197436007, 1633841004, 1986939244, 1952539503, 1231974249, 68, 262149, 17,
    1635017028, 0, 327686, 17, 0, 1635017060, 0, 196613, 19, 0, 262215, 11, 11, 28,
    262215, 16, 6, 4, 196679, 17, 3, 327752, 17, 0, 35, 0, 262215, 19, 33, 0, 262215,
    19, 34, 0, 262215, 32, 11, 25, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0, 262176,
    7, 7, 6, 262167, 9, 6, 3, 262176, 10, 1, 9, 262203, 10, 11, 1, 262187, 6, 12, 0,
    262176, 13, 1, 6, 196637, 16, 6, 196638, 17, 16, 262176, 18, 2, 17, 262203, 18, 19,
    2, 262165, 20, 32, 1, 262187, 20, 21, 0, 262176, 24, 2, 6, 262187, 6, 27, 2,
    262187, 6, 30, 8, 262187, 6, 31, 1, 393260, 9, 32, 30, 31, 31, 327734, 2, 4, 0, 3,
    131320, 5, 262203, 7, 8, 7, 327745, 13, 14, 11, 12, 262205, 6, 15, 14, 196670, 8,
    15, 262205, 6, 22, 8, 262205, 6, 23, 8, 393281, 24, 25, 19, 21, 23, 262205, 6, 26,
    25, 327812, 6, 28, 26, 27, 393281, 24, 29, 19, 21, 22, 196670, 29, 28, 65789,
    65592,
];

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_spirv_passthrough_compute_doubles_storage_buffer() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
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
        let module = create_spirv_module(device, DOUBLE_SPIRV);
        let pipeline = create_compute_pipeline_with_layout(device, module, pipeline_layout, "main");
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
            "raw SPIR-V passthrough compute did not double the storage buffer"
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

// `tri.vert` → SPIR-V: a full-screen triangle from gl_VertexIndex (no inputs).
#[rustfmt::skip]
const TRIANGLE_VERT_SPIRV: &[u32] = &[
    119734787, 65536, 524299, 46, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134,
    0, 196622, 0, 1, 458767, 0, 4, 1852399981, 0, 8, 37, 196611, 2, 450, 262149, 4, 1852399981,
    0, 393221, 8, 1449094247, 1702130277, 1684949368, 30821, 196613, 18, 112, 393221, 35,
    1348430951, 1700164197, 2019914866, 0, 393222, 35, 0, 1348430951, 1953067887, 7237481,
    458758, 35, 1, 1348430951, 1953393007, 1702521171, 0, 458758, 35, 2, 1130327143,
    1148217708, 1635021673, 6644590, 458758, 35, 3, 1130327143, 1147956341, 1635021673,
    6644590, 196613, 37, 0, 262215, 8, 11, 42, 196679, 35, 2, 327752, 35, 0, 11, 0, 327752, 35,
    1, 11, 1, 327752, 35, 2, 11, 3, 327752, 35, 3, 11, 4, 131091, 2, 196641, 3, 2, 262165, 6,
    32, 1, 262176, 7, 1, 6, 262203, 7, 8, 1, 262187, 6, 10, 0, 131092, 11, 196630, 15, 32,
    262167, 16, 15, 2, 262176, 17, 7, 16, 262187, 15, 19, 3212836864, 327724, 16, 20, 19, 19,
    262187, 6, 23, 1, 262187, 15, 27, 1077936128, 327724, 16, 28, 27, 19, 327724, 16, 30, 19,
    27, 262167, 31, 15, 4, 262165, 32, 32, 0, 262187, 32, 33, 1, 262172, 34, 15, 33, 393246,
    35, 31, 15, 34, 34, 262176, 36, 3, 35, 262203, 36, 37, 3, 262187, 15, 39, 0, 262187, 15,
    40, 1065353216, 262176, 44, 3, 31, 327734, 2, 4, 0, 3, 131320, 5, 262203, 17, 18, 7,
    262205, 6, 9, 8, 327850, 11, 12, 9, 10, 196855, 14, 0, 262394, 12, 13, 21, 131320, 13,
    196670, 18, 20, 131321, 14, 131320, 21, 262205, 6, 22, 8, 327850, 11, 24, 22, 23, 196855,
    26, 0, 262394, 24, 25, 29, 131320, 25, 196670, 18, 28, 131321, 26, 131320, 29, 196670, 18,
    30, 131321, 26, 131320, 26, 131321, 14, 131320, 14, 262205, 16, 38, 18, 327761, 15, 41, 38,
    0, 327761, 15, 42, 38, 1, 458832, 31, 43, 41, 42, 39, 40, 327745, 44, 45, 37, 10, 196670,
    45, 43, 65789, 65592,
];

// `tri.frag` → SPIR-V: solid color out (0.25, 0.5, 0.75, 1.0) at location 0.
#[rustfmt::skip]
const TRIANGLE_FRAG_SPIRV: &[u32] = &[
    119734787, 65536, 524299, 15, 0, 131089, 1, 393227, 1, 1280527431, 1685353262, 808793134,
    0, 196622, 0, 1, 393231, 4, 4, 1852399981, 0, 9, 196624, 4, 7, 196611, 2, 450, 262149, 4,
    1852399981, 0, 262149, 9, 1869377379, 114, 262215, 9, 30, 0, 131091, 2, 196641, 3, 2,
    196630, 6, 32, 262167, 7, 6, 4, 262176, 8, 3, 7, 262203, 8, 9, 3, 262187, 6, 10,
    1048576000, 262187, 6, 11, 1056964608, 262187, 6, 12, 1061158912, 262187, 6, 13,
    1065353216, 458796, 7, 14, 10, 11, 12, 13, 327734, 2, 4, 0, 3, 131320, 5, 196670, 9, 14,
    65789, 65592,
];

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_spirv_passthrough_render_draws_solid_color() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
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

        let vertex = create_spirv_module(device, TRIANGLE_VERT_SPIRV);
        let fragment = create_spirv_module(device, TRIANGLE_FRAG_SPIRV);
        let layout = create_empty_pipeline_layout(device);
        let pipeline = create_render_pipeline_passthrough(device, vertex, fragment, layout);

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
            "raw SPIR-V passthrough render produced the wrong pixel: {:?}",
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
        yawgpu::wgpuShaderModuleRelease(fragment);
        yawgpu::wgpuShaderModuleRelease(vertex);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
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
    vertex_module: native::WGPUShaderModule,
    fragment_module: native::WGPUShaderModule,
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
        module: fragment_module,
        entryPoint: string_view("main"),
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
            module: vertex_module,
            entryPoint: string_view("main"),
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

unsafe fn create_spirv_module(
    device: native::WGPUDevice,
    words: &[u32],
) -> native::WGPUShaderModule {
    let mut spirv = native::WGPUShaderSourceSPIRV {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ShaderSourceSPIRV,
        },
        codeSize: u32::try_from(words.len()).expect("spirv word count fits in u32"),
        code: words.as_ptr(),
    };
    let descriptor = native::WGPUShaderModuleDescriptor {
        nextInChain: (&mut spirv.chain) as *mut _,
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
