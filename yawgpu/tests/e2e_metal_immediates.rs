//! Real-Metal e2e for Block 94 immediates: `wgpu*SetImmediates` user data
//! must reach `var<immediate>` shader variables, update between dispatches,
//! and compose with the internal frag-depth-clamp immediate block.
//!
//! Run manually on a Metal machine:
//!   cargo test -p yawgpu --features metal --test e2e_metal_immediates -- --ignored

#[cfg(feature = "metal")]
use std::os::raw::c_void;
#[cfg(feature = "metal")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "metal")]
use yawgpu::native;
#[cfg(feature = "metal")]
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
#[cfg(feature = "metal")]
use yawgpu_test::wait;
#[cfg(feature = "metal")]
use yawgpu_test::{real_backend_skip_reason, RealBackend};

#[cfg(feature = "metal")]
const WIDTH: u32 = 16;
#[cfg(feature = "metal")]
const HEIGHT: u32 = 16;
#[cfg(feature = "metal")]
const BYTES_PER_ROW: u32 = 256;
#[cfg(feature = "metal")]
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;

/// A compute entry point reading a `var<immediate>` vec4u must observe the
/// bytes written by `wgpuComputePassEncoderSetImmediates`.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_compute_immediates_reach_shader() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let shader = r#"
requires immediate_address_space;

var<immediate> imm : vec4u;

@group(0) @binding(0) var<storage, read_write> out_data : vec4u;

@compute @workgroup_size(1)
fn main() {
    out_data = imm;
}
"#;
        let output = create_buffer_sized(
            device,
            16,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let readback = create_buffer_sized(
            device,
            16,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let bgl = create_storage_bgl(device);
        let layout = create_pipeline_layout_with_immediates(device, &bgl, 1, 16);
        let module = create_wgsl_module(device, shader);
        let pipeline = create_compute_pipeline_with_layout(device, module, layout);
        let bind_group = create_single_storage_bind_group(device, bgl, output, 16);

        let values: [u32; 4] = [10, 20, 30, 40];
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetImmediates(
            pass,
            0,
            values.as_ptr().cast(),
            std::mem::size_of_val(&values),
        );
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, 16);
        submit_encoder(queue, encoder);
        yawgpu::wgpuComputePassEncoderRelease(pass);

        let actual = read_u32s(instance, readback, 4);
        assert_eq!(
            actual,
            vec![10, 20, 30, 40],
            "var<immediate> did not observe SetImmediates data"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(output);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Immediates are pass-scoped state: a partial `SetImmediates` between two
/// dispatches must update only the written range for the second dispatch
/// (per-draw snapshot delivery), leaving the other bytes intact.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_compute_immediates_partial_update_between_dispatches() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let shader = r#"
requires immediate_address_space;

var<immediate> imm : vec4u;

@group(0) @binding(0) var<storage, read_write> out_data : vec4u;

@compute @workgroup_size(1)
fn main() {
    out_data = imm;
}
"#;
        let output_a = create_buffer_sized(
            device,
            16,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let output_b = create_buffer_sized(
            device,
            16,
            native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
        );
        let readback = create_buffer_sized(
            device,
            32,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );
        let bgl = create_storage_bgl(device);
        let layout = create_pipeline_layout_with_immediates(device, &bgl, 1, 16);
        let module = create_wgsl_module(device, shader);
        let pipeline = create_compute_pipeline_with_layout(device, module, layout);
        let bind_group_a = create_single_storage_bind_group(device, bgl, output_a, 16);
        let bind_group_b = create_single_storage_bind_group(device, bgl, output_b, 16);

        let first: [u32; 4] = [1, 2, 3, 4];
        // Overwrites only bytes [4, 12) — imm.y and imm.z — before dispatch 2.
        let partial: [u32; 2] = [200, 300];
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetImmediates(
            pass,
            0,
            first.as_ptr().cast(),
            std::mem::size_of_val(&first),
        );
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group_a, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderSetImmediates(
            pass,
            4,
            partial.as_ptr().cast(),
            std::mem::size_of_val(&partial),
        );
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group_b, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output_a, 0, readback, 0, 16);
        yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output_b, 0, readback, 16, 16);
        submit_encoder(queue, encoder);
        yawgpu::wgpuComputePassEncoderRelease(pass);

        let actual = read_u32s(instance, readback, 8);
        assert_eq!(
            &actual[0..4],
            &[1, 2, 3, 4],
            "first dispatch saw wrong immediates"
        );
        assert_eq!(
            &actual[4..8],
            &[1, 200, 300, 4],
            "second dispatch must see the partial update overlaid on the first write"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBindGroupRelease(bind_group_b);
        yawgpu::wgpuBindGroupRelease(bind_group_a);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuBufferRelease(output_b);
        yawgpu::wgpuBufferRelease(output_a);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// The internal frag-depth-clamp immediate must compose with user immediates:
/// a fragment entry point that BOTH reads a `var<immediate>` and writes
/// `@builtin(frag_depth)` gets the clamp range appended after the user
/// prefix (Dawn `RenderImmediates` layout). The written depth comes from the
/// user immediate, so a collision between the two regions would corrupt it.
#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_frag_depth_from_immediate_composes_with_clamp() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let vs = r#"
@vertex
fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0));
    return vec4<f32>(pos[idx], 0.0, 1.0);
}
"#;
        let fs = r#"
requires immediate_address_space;

var<immediate> imm : vec4f;

@fragment
fn fs() -> @builtin(frag_depth) f32 {
    return imm.x;
}
"#;
        let layout = create_pipeline_layout_with_immediates(device, std::ptr::null(), 0, 16);
        let vs_module = create_wgsl_module(device, vs);
        let fs_module = create_wgsl_module(device, fs);
        let pipeline = create_frag_depth_pipeline_with_layout(device, vs_module, fs_module, layout);
        let depth = create_depth_texture(device);
        let depth_view = yawgpu::wgpuTextureCreateView(depth, std::ptr::null());
        let readback = create_buffer_sized(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let values: [f32; 4] = [0.625, 0.0, 0.0, 0.0];
        let depth_attachment = native::WGPURenderPassDepthStencilAttachment {
            nextInChain: std::ptr::null_mut(),
            view: depth_view,
            depthLoadOp: native::WGPULoadOp_Clear,
            depthStoreOp: native::WGPUStoreOp_Store,
            depthClearValue: 0.0,
            depthReadOnly: false.into(),
            stencilLoadOp: native::WGPULoadOp_Undefined,
            stencilStoreOp: native::WGPUStoreOp_Undefined,
            stencilClearValue: 0,
            stencilReadOnly: false.into(),
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 0,
            colorAttachments: std::ptr::null(),
            depthStencilAttachment: &depth_attachment,
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetImmediates(
            pass,
            0,
            values.as_ptr().cast(),
            std::mem::size_of_val(&values),
        );
        yawgpu::wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_depth_t2b(encoder, depth, readback);
        submit_encoder(queue, encoder);

        let bytes = read_bytes(instance, readback, READBACK_SIZE);
        let mut mismatches = Vec::new();
        for row in 0..HEIGHT as usize {
            for col in 0..WIDTH as usize {
                let off = row * BYTES_PER_ROW as usize + col * 4;
                let got = f32::from_ne_bytes(bytes[off..off + 4].try_into().expect("four bytes"));
                if (got - 0.625).abs() > 1e-5 {
                    mismatches.push((col, row, got));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "frag_depth from immediate wrong (col,row,got expected 0.625): {mismatches:?}"
        );
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureViewRelease(depth_view);
        yawgpu::wgpuTextureRelease(depth);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(fs_module);
        yawgpu::wgpuShaderModuleRelease(vs_module);
        yawgpu::wgpuPipelineLayoutRelease(layout);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// ---------------------------------------------------------------------------
// Helpers (per-file, mirroring e2e_metal_compute.rs / e2e_metal_depth.rs)
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
unsafe fn create_pipeline_layout_with_immediates(
    device: native::WGPUDevice,
    bgls: *const native::WGPUBindGroupLayout,
    bgl_count: usize,
    immediate_size: u32,
) -> native::WGPUPipelineLayout {
    let descriptor = native::WGPUPipelineLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        bindGroupLayoutCount: bgl_count,
        bindGroupLayouts: bgls,
        immediateSize: immediate_size,
    };
    let layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
unsafe fn create_compute_pipeline_with_layout(
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
            entryPoint: string_view("main"),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
unsafe fn create_single_storage_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    buffer: native::WGPUBuffer,
    size: u64,
) -> native::WGPUBindGroup {
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer,
        offset: 0,
        size,
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

#[cfg(feature = "metal")]
unsafe fn create_frag_depth_pipeline_with_layout(
    device: native::WGPUDevice,
    vs_module: native::WGPUShaderModule,
    fs_module: native::WGPUShaderModule,
    layout: native::WGPUPipelineLayout,
) -> native::WGPURenderPipeline {
    let depth_stencil = native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_Depth32Float,
        depthWriteEnabled: native::WGPUOptionalBool_True,
        depthCompare: native::WGPUCompareFunction_Always,
        stencilFront: stencil_face(),
        stencilBack: stencil_face(),
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    };
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: fs_module,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 0,
        targets: std::ptr::null(),
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vs_module,
            entryPoint: string_view("vs"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: primitive_state(),
        depthStencil: &depth_stencil,
        multisample: multisample_state(),
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

#[cfg(feature = "metal")]
unsafe fn create_depth_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        dimension: native::WGPUTextureDimension_2D,
        size: texture_extent(),
        format: native::WGPUTextureFormat_Depth32Float,
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
unsafe fn record_depth_t2b(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
) {
    let source = native::WGPUTexelCopyTextureInfo {
        texture: source,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_DepthOnly,
    };
    let destination = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        },
        buffer: destination,
    };
    let size = texture_extent();
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &size);
}

#[cfg(feature = "metal")]
fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
    }
}

#[cfg(feature = "metal")]
fn stencil_face() -> native::WGPUStencilFaceState {
    native::WGPUStencilFaceState {
        compare: native::WGPUCompareFunction_Always,
        failOp: native::WGPUStencilOperation_Keep,
        depthFailOp: native::WGPUStencilOperation_Keep,
        passOp: native::WGPUStencilOperation_Keep,
    }
}

#[cfg(feature = "metal")]
fn primitive_state() -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        nextInChain: std::ptr::null_mut(),
        topology: native::WGPUPrimitiveTopology_TriangleList,
        stripIndexFormat: native::WGPUIndexFormat_Undefined,
        frontFace: native::WGPUFrontFace_CCW,
        cullMode: native::WGPUCullMode_None,
        unclippedDepth: 0,
    }
}

#[cfg(feature = "metal")]
fn multisample_state() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 1,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
}

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
unsafe fn submit_encoder(queue: native::WGPUQueue, encoder: native::WGPUCommandEncoder) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

#[cfg(feature = "metal")]
unsafe fn read_u32s(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    count: usize,
) -> Vec<u32> {
    let bytes = read_bytes(instance, buffer, count * std::mem::size_of::<u32>());
    bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_ne_bytes(chunk.try_into().expect("chunk is four bytes")))
        .collect()
}

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
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

#[cfg(feature = "metal")]
unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
}

#[cfg(feature = "metal")]
fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

#[cfg(feature = "metal")]
fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
