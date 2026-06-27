//! Real-Metal e2e for external-texture Slice D / R10.
//!
//! External textures are implemented on Metal only (mirroring wgpu). Slice A
//! lowered `texture_external` to multiplanar MSL via Tint; Slices B–C added the
//! vendor `yawgpuDeviceCreateExternalTexture` resource + bind-group binding;
//! Slice D wires the plane/params bindings into the Metal HAL. This test closes
//! R10: sampling an `Rgba` external texture in a fragment shader and reading the
//! rendered result back must reproduce the source colour (identity passthrough).
//!
//! Run manually on a real Metal GPU:
//!   cargo test -p yawgpu --features metal --test e2e_metal_external_texture -- --ignored --nocapture

#![cfg(feature = "metal")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUExtent2D, YaWGPUExternalTextureDescriptor, YaWGPUInstanceBackendSelect, YaWGPUOrigin2D,
    YAWGPU_EXTERNAL_TEXTURE_FORMAT_NV12, YAWGPU_EXTERNAL_TEXTURE_FORMAT_RGBA,
    YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_0_DEGREES, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const WIDTH: u32 = 1;
const HEIGHT: u32 = 1;
const BYTES_PER_ROW: u32 = 256;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize;

// Known source colour written into the plane-0 texture; identity passthrough
// must reproduce it at the rendered output (within rounding tolerance).
const SRC_RGBA: [u8; 4] = [64, 128, 192, 255];

const EXTERNAL_TEXTURE_SHADER: &str = "@vertex\n\
fn vertexMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n\
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));\n\
  return vec4f(p[i], 0.0, 1.0);\n\
}\n\
\n\
@group(0) @binding(0) var ext: texture_external;\n\
\n\
@fragment\n\
fn fragmentMain() -> @location(0) vec4f {\n\
  return textureLoad(ext, vec2u(0u, 0u));\n\
}\n";

#[test]
#[ignore = "manual real-backend test"]
fn metal_external_texture_rgba_passthrough_round_trips() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // 1. Source plane-0 texture, written with the known colour.
        let source = create_texture(
            device,
            native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
            native::WGPUTextureFormat_RGBA8Unorm,
        );
        write_solid_pixel(queue, source, &SRC_RGBA);
        let plane0 = create_default_view(source);

        // 2. External texture (Rgba, identity params).
        let descriptor = YaWGPUExternalTextureDescriptor {
            plane0,
            plane1: std::ptr::null_mut(),
            format: YAWGPU_EXTERNAL_TEXTURE_FORMAT_RGBA,
            cropOrigin: YaWGPUOrigin2D { x: 0, y: 0 },
            cropSize: YaWGPUExtent2D {
                width: WIDTH,
                height: HEIGHT,
            },
            apparentSize: YaWGPUExtent2D {
                width: WIDTH,
                height: HEIGHT,
            },
            // RGBA passthrough: stop after (the no-op) YUV step so Tint skips the
            // gamut matrix + src/dst gamma transfer functions entirely. With the
            // full color-management path (doYuvToRgbConversionOnly=0) the zero-filled
            // transfer params apply a non-identity curve (Metal pow(0,0)=0) and zero
            // the RGB. numPlanes=1 means no YUV matrix is applied either, so the
            // result is the raw plane0 texel.
            doYuvToRgbConversionOnly: 1,
            // Unused for Rgba; identity-ish fill.
            yuvToRgbConversionMatrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            srcTransferFunctionParameters: [0.0; 7],
            dstTransferFunctionParameters: [0.0; 7],
            gamutConversionMatrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            mirrored: 0,
            rotation: YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_0_DEGREES,
        };
        let external = yawgpu::yawgpuDeviceCreateExternalTexture(device, &descriptor);
        assert!(
            !external.is_null(),
            "yawgpuDeviceCreateExternalTexture returned null: {:?}",
            errors.lock().expect("error lock")
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "external-texture creation raised a device error: {:?}",
            errors.lock().expect("error lock")
        );

        // 3. Output render target.
        let output = create_texture(
            device,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            native::WGPUTextureFormat_RGBA8Unorm,
        );
        let output_view = create_default_view(output);

        // 4. Render pipeline (auto layout) sampling the external texture.
        let module = create_wgsl_module(device, EXTERNAL_TEXTURE_SHADER);
        let pipeline = create_render_pipeline(device, module);

        // 5. Bind group: external texture at binding 0 via the chained entry.
        let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
        assert!(!layout.is_null());
        let mut ext_entry = native::WGPUExternalTextureBindingEntry {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_ExternalTextureBindingEntry,
            },
            externalTexture: external,
        };
        let bind_entry = native::WGPUBindGroupEntry {
            nextInChain: (&mut ext_entry.chain) as *mut native::WGPUChainedStruct,
            binding: 0,
            buffer: std::ptr::null_mut(),
            offset: 0,
            size: 0,
            sampler: std::ptr::null_mut(),
            textureView: std::ptr::null_mut(),
        };
        let entries = [bind_entry];
        let bind_group_descriptor = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout,
            entryCount: entries.len(),
            entries: entries.as_ptr(),
        };
        let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &bind_group_descriptor);
        assert!(
            !bind_group.is_null(),
            "bind group creation failed: {:?}",
            errors.lock().expect("error lock")
        );

        // 6. Render + copy output to a readback buffer.
        let readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: output_view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null_mut(),
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
            occlusionQuerySet: std::ptr::null_mut(),
            timestampWrites: std::ptr::null(),
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

        let color_src = native::WGPUTexelCopyTextureInfo {
            texture: output,
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
            buffer: readback,
        };
        let extent = native::WGPUExtent3D {
            width: WIDTH,
            height: HEIGHT,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &color_src, &color_dst, &extent);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let pixels = read_buffer(instance, readback, 0, 4);
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "device error during external-texture sampling: {:?}",
            errors.lock().expect("error lock")
        );
        for (channel, (&got, &expected)) in pixels.iter().zip(SRC_RGBA.iter()).enumerate() {
            let delta = i32::from(got) - i32::from(expected);
            assert!(
                delta.abs() <= 2,
                "channel {channel}: external-texture passthrough got {got}, expected ~{expected} (full {pixels:?})"
            );
        }

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuBindGroupLayoutRelease(layout);
        yawgpu::wgpuRenderPipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuExternalTextureRelease(external);
        yawgpu::wgpuTextureViewRelease(plane0);
        yawgpu::wgpuTextureViewRelease(output_view);
        yawgpu::wgpuTextureRelease(source);
        yawgpu::wgpuTextureRelease(output);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Renders a fullscreen triangle whose fragment shader `textureLoad`s the given
/// external texture into an RGBA8Unorm target, then reads back texel (0,0).
/// Releases only the resources it creates (output, pipeline, bind group, readback);
/// the caller owns `external` and the planes.
unsafe fn render_external_texture(
    instance: native::WGPUInstance,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    external: native::WGPUExternalTexture,
    errors: &Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
) -> [u8; 4] {
    let output = create_texture(
        device,
        native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        native::WGPUTextureFormat_RGBA8Unorm,
    );
    let output_view = create_default_view(output);
    let module = create_wgsl_module(device, EXTERNAL_TEXTURE_SHADER);
    let pipeline = create_render_pipeline(device, module);

    let layout = yawgpu::wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
    assert!(!layout.is_null());
    let mut ext_entry = native::WGPUExternalTextureBindingEntry {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ExternalTextureBindingEntry,
        },
        externalTexture: external,
    };
    let bind_entry = native::WGPUBindGroupEntry {
        nextInChain: (&mut ext_entry.chain) as *mut native::WGPUChainedStruct,
        binding: 0,
        buffer: std::ptr::null_mut(),
        offset: 0,
        size: 0,
        sampler: std::ptr::null_mut(),
        textureView: std::ptr::null_mut(),
    };
    let entries = [bind_entry];
    let bind_group_descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &bind_group_descriptor);
    assert!(
        !bind_group.is_null(),
        "bind group creation failed: {:?}",
        errors.lock().expect("error lock")
    );

    let readback = create_buffer(
        device,
        READBACK_SIZE as u64,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let color_attachment = native::WGPURenderPassColorAttachment {
        nextInChain: std::ptr::null_mut(),
        view: output_view,
        depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
        resolveTarget: std::ptr::null_mut(),
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
        occlusionQuerySet: std::ptr::null_mut(),
        timestampWrites: std::ptr::null(),
    };
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
    yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpu::wgpuRenderPassEncoderEnd(pass);
    yawgpu::wgpuRenderPassEncoderRelease(pass);

    let color_src = native::WGPUTexelCopyTextureInfo {
        texture: output,
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
        buffer: readback,
    };
    let extent = native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &color_src, &color_dst, &extent);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);

    let pixels = read_buffer(instance, readback, 0, 4);

    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    yawgpu::wgpuRenderPipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuTextureViewRelease(output_view);
    yawgpu::wgpuTextureRelease(output);
    yawgpu::wgpuBufferRelease(readback);

    [pixels[0], pixels[1], pixels[2], pixels[3]]
}

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
        entryPoint: string_view("fragmentMain"),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: 1,
        targets: &color_target,
    };
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null_mut(),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("vertexMain"),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: 0,
            buffers: std::ptr::null(),
        },
        primitive: native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: native::WGPUFrontFace_CCW,
            cullMode: native::WGPUCullMode_None,
            unclippedDepth: 0,
        },
        depthStencil: std::ptr::null(),
        multisample: native::WGPUMultisampleState {
            nextInChain: std::ptr::null_mut(),
            count: 1,
            mask: u32::MAX,
            alphaToCoverageEnabled: 0,
        },
        fragment: &fragment,
    };
    let pipeline = yawgpu::wgpuDeviceCreateRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
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
        nextInChain: (&mut wgsl.chain) as *mut native::WGPUChainedStruct,
        label: empty_string_view(),
    };
    let module = yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor);
    assert!(!module.is_null());
    module
}

unsafe fn create_default_view(texture: native::WGPUTexture) -> native::WGPUTextureView {
    let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
    assert!(!view.is_null());
    view
}

unsafe fn write_solid_pixel(
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    rgba: &[u8; 4],
) {
    let destination = native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let layout = native::WGPUTexelCopyBufferLayout {
        offset: 0,
        bytesPerRow: 4,
        rowsPerImage: HEIGHT,
    };
    let extent = native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
    };
    yawgpu::wgpuQueueWriteTexture(
        queue,
        &destination,
        rgba.as_ptr().cast(),
        rgba.len(),
        &layout,
        &extent,
    );
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    usage: native::WGPUTextureUsage,
    format: native::WGPUTextureFormat,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: WIDTH,
            height: HEIGHT,
            depthOrArrayLayers: 1,
        },
        format,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
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

unsafe fn read_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    offset: u64,
    len: usize,
) -> Vec<u8> {
    let mapped_len = len.next_multiple_of(4);
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
        mapped_len,
        callback_info,
    );
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(
        buffer,
        usize::try_from(offset).expect("test offset fits in usize"),
        mapped_len,
    );
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), mapped_len)[..len].to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
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
    let mut adapter: native::WGPUAdapter = std::ptr::null_mut();
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
    let mut device: native::WGPUDevice = std::ptr::null_mut();
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        },
        deviceLostCallbackInfo: std::mem::zeroed(),
        uncapturedErrorCallbackInfo: std::mem::zeroed(),
    };
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, &descriptor, callback_info);
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

// ── Nv12 (two-plane YUV→RGB) ──────────────────────────────────────────────────
//
// R10 second half: a 2-plane Nv12 external texture (R8Unorm luma + RG8Unorm
// chroma) sampled via `textureLoad` must combine plane0.r + plane1.rg and apply
// `yuvToRgbConversionMatrix`. Using an *identity* YUV matrix (R=Y, G=U, B=V) with
// `doYuvToRgbConversionOnly=1` (skip gamut/transfer) makes the conversion exactly
// the plane recombination, so the readback equals the written (Y,U,V) directly —
// proving the two-plane load path and the mat3x4 multiply are wired correctly.
#[test]
#[ignore = "manual real-backend test"]
fn metal_external_texture_nv12_two_plane_round_trips() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    // Y, U, V written into the planes; identity YUV matrix → expected RGBA.
    const Y: u8 = 64;
    const U: u8 = 128;
    const V: u8 = 192;
    const EXPECTED: [u8; 4] = [Y, U, V, 255];

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // plane0 = R8Unorm luma, plane1 = RG8Unorm chroma.
        let luma = create_texture(
            device,
            native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
            native::WGPUTextureFormat_R8Unorm,
        );
        write_solid_bytes(queue, luma, &[Y], 1);
        let plane0 = create_default_view(luma);

        let chroma = create_texture(
            device,
            native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
            native::WGPUTextureFormat_RG8Unorm,
        );
        write_solid_bytes(queue, chroma, &[U, V], 2);
        let plane1 = create_default_view(chroma);

        let descriptor = YaWGPUExternalTextureDescriptor {
            plane0,
            plane1,
            format: YAWGPU_EXTERNAL_TEXTURE_FORMAT_NV12,
            cropOrigin: YaWGPUOrigin2D { x: 0, y: 0 },
            cropSize: YaWGPUExtent2D {
                width: WIDTH,
                height: HEIGHT,
            },
            apparentSize: YaWGPUExtent2D {
                width: WIDTH,
                height: HEIGHT,
            },
            doYuvToRgbConversionOnly: 1,
            // Identity YUV→RGB: R=Y, G=U, B=V (column-major mat3x4).
            yuvToRgbConversionMatrix: [
                1.0, 0.0, 0.0, 0.0, // col 0 → R = Y
                0.0, 1.0, 0.0, 0.0, // col 1 → G = U
                0.0, 0.0, 1.0, 0.0, // col 2 → B = V
            ],
            srcTransferFunctionParameters: [0.0; 7],
            dstTransferFunctionParameters: [0.0; 7],
            gamutConversionMatrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            mirrored: 0,
            rotation: YAWGPU_EXTERNAL_TEXTURE_ROTATION_ROTATE_0_DEGREES,
        };
        let external = yawgpu::yawgpuDeviceCreateExternalTexture(device, &descriptor);
        assert!(
            !external.is_null(),
            "Nv12 create returned null: {:?}",
            errors.lock().expect("error lock")
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "Nv12 create raised a device error: {:?}",
            errors.lock().expect("error lock")
        );

        let result = render_external_texture(instance, device, queue, external, &errors);
        for (i, (&got, &want)) in result.iter().zip(EXPECTED.iter()).enumerate() {
            let diff = (got as i16 - want as i16).abs();
            assert!(
                diff <= 2,
                "channel {i}: Nv12 got {got}, expected ~{want} (full {result:?})"
            );
        }

        yawgpu::wgpuExternalTextureRelease(external);
        yawgpu::wgpuTextureRelease(chroma);
        yawgpu::wgpuTextureRelease(luma);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn write_solid_bytes(
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    texel: &[u8],
    bytes_per_row: u32,
) {
    let destination = native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    let layout = native::WGPUTexelCopyBufferLayout {
        offset: 0,
        bytesPerRow: bytes_per_row,
        rowsPerImage: HEIGHT,
    };
    let extent = native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
    };
    yawgpu::wgpuQueueWriteTexture(
        queue,
        &destination,
        texel.as_ptr().cast(),
        texel.len(),
        &layout,
        &extent,
    );
}
