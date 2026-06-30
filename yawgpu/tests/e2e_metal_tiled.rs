//! Real-Metal e2e for TBDR multi-subpass deferred rendering (Slice 2.7a).
//!
//! Two color attachments, two subpasses. Subpass 0 writes attachment 0 (the
//! "g-buffer") = (0.5, 0.0, 0.0, 1.0). Subpass 1 reads attachment 0 as an
//! `input_attachment` (Metal `[[color(0)]]` programmable-blend tile read) and
//! writes attachment 1 (the final target) = loaded + (0.0, 0.25, 0.0, 0.0). The
//! fragment writes its subpass's GLOBAL color slot via `@location(N)`. Reading the
//! input requires real GPU execution + readback, so the value (128, 64, 0, 255) at
//! the final attachment proves the input_attachment color-slot map + Metal
//! programmable blending work end to end.
//!
//! Gated on `metal` + `tiled`; run with:
//! `cargo test -p yawgpu --features metal,tiled --test e2e_metal_tiled -- --ignored`

#![cfg(all(feature = "metal", feature = "tiled"))]

use std::os::raw::c_void;

use yawgpu::{
    native, YaWGPUAttachmentLayout, YaWGPUInstanceBackendSelect, YaWGPUSubpassColorAttachment,
    YaWGPUSubpassDependency, YaWGPUSubpassDependencyType_ColorToInput, YaWGPUSubpassInputAttachment,
    YaWGPUSubpassLayout, YaWGPUSubpassPassLayout, YaWGPUSubpassPassLayoutDescriptor,
    YaWGPUSubpassRenderPassDescriptor, YaWGPUSubpassRenderPipelineDescriptor,
    YAWGPU_INSTANCE_BACKEND_METAL, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const BYTES_PER_PIXEL: usize = 4;
const BYTES_PER_ROW: u32 = 256;
const ROW_BYTES: usize = WIDTH as usize * BYTES_PER_PIXEL;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;

// Fullscreen triangle covering the 16x16 target (clip-space, no vertex buffer).
const WRITE_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(0.5, 0.0, 0.0, 1.0);
}
"#;

const LOAD_SHADER: &str = r#"
enable chromium_internal_input_attachments;

@group(0) @binding(0) @input_attachment_index(0) var gbuffer: input_attachment<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  return vec4<f32>(p[i], 0.0, 1.0);
}

@fragment
fn fs() -> @location(1) vec4<f32> {
  return inputAttachmentLoad(gbuffer) + vec4<f32>(0.0, 0.25, 0.0, 0.0);
}
"#;

#[test]
#[ignore = "requires a real Metal device"]
fn metal_tiled_deferred_reads_input_attachment() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        eprintln!("skipping: no real Metal device");
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = std::sync::Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured.lock().expect("lock").push(format!("{error:?}"))),
        );
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let readback = run_deferred(device, queue);
        let pixels = read_unpacked_texture_buffer(instance, readback);

        // Final attachment = g-buffer (0.5, 0, 0) + (0, 0.25, 0) = (0.5, 0.25, 0, 1).
        // Reading the g-buffer back as an input attachment proves the color-slot map
        // + Metal programmable-blend tile read work end to end.
        let expected = [128u8, 64, 0, 255];
        assert!(
            contains_pixel(&pixels, expected),
            "expected {:?} from the input-attachment read; distinct = {:?}",
            expected,
            distinct_pixels(&pixels)
        );
        assert!(
            errors.lock().expect("lock").is_empty(),
            "device errors: {:?}",
            errors.lock().expect("lock")
        );

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
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

unsafe fn run_deferred(
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
) -> native::WGPUBuffer {
    let layout = create_two_subpass_input_layout(device);
    let pipeline0 = create_subpass_pipeline(device, layout, 0, WRITE_SHADER, 0, None);
    let pipeline1 = create_subpass_pipeline(device, layout, 1, LOAD_SHADER, 1, Some(0));

    let gbuffer = create_texture(device, native::WGPUTextureUsage_RenderAttachment);
    let final_tex = create_texture(
        device,
        native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
    );
    let gbuffer_view = yawgpu::wgpuTextureCreateView(gbuffer, std::ptr::null());
    let final_view = yawgpu::wgpuTextureCreateView(final_tex, std::ptr::null());

    let attachments = [
        subpass_color_attachment(gbuffer_view),
        subpass_color_attachment(final_view),
    ];
    let pass_descriptor = YaWGPUSubpassRenderPassDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        passLayout: layout,
        extent: texture_extent(),
        colorAttachments: attachments.as_ptr(),
        colorAttachmentCount: attachments.len(),
        depthStencilAttachment: std::ptr::null(),
    };

    let readback = create_buffer(
        device,
        READBACK_SIZE,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &pass_descriptor);
    assert!(!pass.is_null());
    yawgpu::yawgpuSubpassRenderPassEncoderSetPipeline(pass, pipeline0);
    yawgpu::yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpu::yawgpuSubpassRenderPassEncoderNextSubpass(pass);
    yawgpu::yawgpuSubpassRenderPassEncoderSetPipeline(pass, pipeline1);
    // The input attachment (group 0 / binding 0) is bound implicitly by the pass's
    // color attachment 0 — the bind group omits it, so an empty group satisfies the
    // pipeline's group-0 layout.
    let bgl = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline1, 0);
    let bind_group = create_empty_bind_group(device, bgl);
    yawgpu::yawgpuSubpassRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpu::yawgpuSubpassRenderPassEncoderEnd(pass);
    yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);

    record_t2b(encoder, final_tex, readback);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);

    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(bgl);
    yawgpu::wgpuTextureViewRelease(final_view);
    yawgpu::wgpuTextureViewRelease(gbuffer_view);
    yawgpu::wgpuTextureRelease(final_tex);
    yawgpu::wgpuTextureRelease(gbuffer);
    yawgpu::wgpuRenderPipelineRelease(pipeline1);
    yawgpu::wgpuRenderPipelineRelease(pipeline0);
    yawgpu::yawgpuSubpassPassLayoutRelease(layout);
    readback
}

unsafe fn create_two_subpass_input_layout(device: native::WGPUDevice) -> YaWGPUSubpassPassLayout {
    let colors = [
        YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sampleCount: 1,
        },
        YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sampleCount: 1,
        },
    ];
    let subpass0_colors = [0u32];
    let subpass1_colors = [1u32];
    let input = YaWGPUSubpassInputAttachment {
        group: 0,
        binding: 0,
        sourceSubpass: 0,
        sourceAttachment: 0,
    };
    let subpasses = [
        YaWGPUSubpassLayout {
            colorAttachmentIndices: subpass0_colors.as_ptr(),
            colorAttachmentIndexCount: subpass0_colors.len(),
            usesDepthStencil: 0,
            inputAttachments: std::ptr::null(),
            inputAttachmentCount: 0,
        },
        YaWGPUSubpassLayout {
            colorAttachmentIndices: subpass1_colors.as_ptr(),
            colorAttachmentIndexCount: subpass1_colors.len(),
            usesDepthStencil: 0,
            inputAttachments: &input,
            inputAttachmentCount: 1,
        },
    ];
    let dependency = YaWGPUSubpassDependency {
        srcSubpass: 0,
        dstSubpass: 1,
        dependencyType: YaWGPUSubpassDependencyType_ColorToInput,
        byRegion: 1,
    };
    let descriptor = YaWGPUSubpassPassLayoutDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        colorAttachments: colors.as_ptr(),
        colorAttachmentCount: colors.len(),
        depthStencilAttachment: std::ptr::null(),
        subpasses: subpasses.as_ptr(),
        subpassCount: subpasses.len(),
        dependencies: &dependency,
        dependencyCount: 1,
    };
    let layout = yawgpu::yawgpuDeviceCreateSubpassPassLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn create_subpass_pipeline(
    device: native::WGPUDevice,
    layout: YaWGPUSubpassPassLayout,
    subpass_index: u32,
    shader_src: &str,
    write_slot: u32,
    read_slot: Option<u32>,
) -> native::WGPURenderPipeline {
    let shader = create_wgsl_module(device, shader_src);
    // Targets are indexed by GLOBAL color slot. The written slot is active; an
    // input-attachment-read slot needs the format declared (so Metal knows the
    // `[[color(N)]]` input's pixel format) but writeMask None; any other slot is
    // disabled (format Undefined).
    let max_slot = read_slot.map_or(write_slot, |r| r.max(write_slot));
    let targets: Vec<native::WGPUColorTargetState> = (0..=max_slot)
        .map(|slot| {
            if slot == write_slot {
                native::WGPUColorTargetState {
                    nextInChain: std::ptr::null_mut(),
                    format: native::WGPUTextureFormat_RGBA8Unorm,
                    blend: std::ptr::null(),
                    writeMask: native::WGPUColorWriteMask_All,
                }
            } else if read_slot == Some(slot) {
                native::WGPUColorTargetState {
                    nextInChain: std::ptr::null_mut(),
                    format: native::WGPUTextureFormat_RGBA8Unorm,
                    blend: std::ptr::null(),
                    writeMask: native::WGPUColorWriteMask_None,
                }
            } else {
                native::WGPUColorTargetState {
                    nextInChain: std::ptr::null_mut(),
                    format: native::WGPUTextureFormat_Undefined,
                    blend: std::ptr::null(),
                    writeMask: native::WGPUColorWriteMask_None,
                }
            }
        })
        .collect();
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module: shader,
        entryPoint: string_view("fs"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: targets.len(),
        targets: targets.as_ptr(),
    };
    let base = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: shader,
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
    let descriptor = YaWGPUSubpassRenderPipelineDescriptor {
        nextInChain: std::ptr::null(),
        base,
        passLayout: layout,
        subpassIndex: subpass_index,
    };
    let pipeline = yawgpu::yawgpuDeviceCreateSubpassRenderPipeline(device, &descriptor);
    yawgpu::wgpuShaderModuleRelease(shader);
    assert!(!pipeline.is_null(), "subpass pipeline {subpass_index} creation failed");
    pipeline
}

unsafe fn record_t2b(
    encoder: native::WGPUCommandEncoder,
    texture: native::WGPUTexture,
    buffer: native::WGPUBuffer,
) {
    let source = texture_copy_info(texture);
    let destination = buffer_copy_info(buffer);
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &texture_extent());
}

fn subpass_color_attachment(view: native::WGPUTextureView) -> YaWGPUSubpassColorAttachment {
    YaWGPUSubpassColorAttachment {
        view,
        resolveTarget: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.0,
            g: 0.0,
            b: 0.0,
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

unsafe fn create_empty_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
) -> native::WGPUBindGroup {
    let descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: 0,
        entries: std::ptr::null(),
    };
    let group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert!(!group.is_null());
    group
}

unsafe fn create_buffer(
    device: native::WGPUDevice,
    size: usize,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size: size as u64,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
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
    assert!(!module.is_null(), "shader module creation failed");
    module
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
        usize::try_from(offset).expect("offset fits usize"),
        len,
        callback_info,
    );
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(
        buffer,
        usize::try_from(offset).expect("offset fits usize"),
        len,
    );
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut std::ffi::c_void,
    _userdata2: *mut std::ffi::c_void,
) {
    unsafe {
        *(userdata1.cast::<native::WGPUMapAsyncStatus>()) = status;
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

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

fn contains_pixel(pixels: &[u8], rgba: [u8; 4]) -> bool {
    pixels.chunks_exact(BYTES_PER_PIXEL).any(|pixel| pixel == rgba)
}

fn distinct_pixels(pixels: &[u8]) -> Vec<[u8; 4]> {
    let mut seen: Vec<[u8; 4]> = Vec::new();
    for pixel in pixels.chunks_exact(BYTES_PER_PIXEL) {
        let rgba = [pixel[0], pixel[1], pixel[2], pixel[3]];
        if !seen.contains(&rgba) {
            seen.push(rgba);
        }
    }
    seen
}
