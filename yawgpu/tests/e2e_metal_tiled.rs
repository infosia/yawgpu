#![cfg(all(feature = "metal", feature = "tiled"))]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::{
    native, YaWGPUInstanceBackendSelect, YaWGPUTiledCapabilities, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const WIDTH: u32 = 16;
const HEIGHT: u32 = 16;
const BYTES_PER_PIXEL: usize = 4;
const BYTES_PER_ROW: u32 = 256;
const ROW_BYTES: usize = WIDTH as usize * BYTES_PER_PIXEL;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;

#[test]
#[ignore = "manual real-backend test"]
fn metal_tiled_features_and_capabilities_are_advertised() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        assert!(!adapter.is_null());

        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(adapter, yawgpu::YaWGPUFeatureName_MultiSubpass),
            1
        );
        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(adapter, yawgpu::YaWGPUFeatureName_TransientAttachments,),
            1
        );
        assert_eq!(
            yawgpu::wgpuAdapterHasFeature(
                adapter,
                yawgpu::YaWGPUFeatureName_ShaderFramebufferFetch,
            ),
            1
        );
        let mut capabilities = zeroed_tiled_capabilities();
        assert_eq!(
            yawgpu::yawgpuAdapterGetTiledCapabilities(adapter, &mut capabilities),
            native::WGPUStatus_Success
        );
        assert!(capabilities.maxSubpasses > 0);
        assert!(capabilities.maxSubpassColorAttachments > 0);

        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_explicit_transient_attachment_allocates_without_device_error() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );
        let descriptor = yawgpu::YaWGPUTransientAttachmentDescriptor {
            nextInChain: std::ptr::null(),
            label: native::WGPUStringView {
                data: std::ptr::null(),
                length: 0,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sizeMode: yawgpu::YaWGPUTransientSizeMode_Explicit,
            width: 16,
            height: 16,
            sampleCount: 1,
        };

        let attachment = yawgpu::yawgpuDeviceCreateTransientAttachment(device, &descriptor);
        assert!(!attachment.is_null());
        assert!(errors.lock().expect("error lock").is_empty());

        yawgpu::yawgpuTransientAttachmentRelease(attachment);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_clear_only_subpass_pass_submits_without_device_error() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );

        let texture = create_color_texture(device);
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        let layout = create_single_color_subpass_layout(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        record_clear_subpass_pass(encoder, layout, view);
        submit_encoder(queue, encoder);

        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::yawgpuSubpassPassLayoutRelease(layout);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_clear_only_subpass_pass_accepts_memoryless_transient_color() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );

        let transient = create_match_target_transient(device);
        let layout = create_single_color_subpass_layout(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        record_clear_transient_subpass_pass(encoder, layout, transient);
        submit_encoder(queue, encoder);

        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::yawgpuSubpassPassLayoutRelease(layout);
        yawgpu::yawgpuTransientAttachmentRelease(transient);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_two_subpass_draw_subpass_load_readback() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = Arc::new(Mutex::new(Vec::new()));
        let captured_errors = Arc::clone(&errors);
        yawgpu::testing_set_uncaptured_error_callback(
            device,
            Some(move |error| captured_errors.lock().expect("error lock").push(error)),
        );

        let readback = run_two_subpass_draw_readback(device);
        let pixels = read_unpacked_texture_buffer(instance, readback);

        assert_center_pixel_approx(&pixels, [0, 255, 0, 255], 1);
        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuBufferRelease(readback);
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
    yawgpu::wgpuCreateInstance(&descriptor)
}

unsafe fn run_two_subpass_draw_readback(device: native::WGPUDevice) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let gbuffer = create_texture(device, native::WGPUTextureUsage_RenderAttachment);
    let output = create_texture(
        device,
        native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
    );
    let gbuffer_view = yawgpu::wgpuTextureCreateView(gbuffer, std::ptr::null());
    let output_view = yawgpu::wgpuTextureCreateView(output, std::ptr::null());
    let readback = create_buffer(
        device,
        READBACK_SIZE as u64,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let layout = create_two_subpass_input_layout(device);
    let write_module = create_wgsl_module(device, SUBPASS_WRITE_SHADER);
    let load_module = create_wgsl_module(device, SUBPASS_LOAD_SHADER);
    let rgba_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        blend: std::ptr::null(),
        writeMask: native::WGPUColorWriteMask_All,
    };
    let targets = [rgba_target];
    // Subpass 1 on Metal writes to MTL color attachment 1 (the output slot)
    // via `fs_metal`'s `@location(1)`. The pipeline target array has one entry
    // (matching the subpass's single `color_attachment_indices=[1]`); the HAL
    // backs every MTL `colorAttachments[i]` from the pass layout's flat
    // attachment list so the pipeline is format-compatible with the encoder.
    // (Mirrors mgpu's hello_deferred — naga MSL doesn't subpass-remap output
    // locations; the shader author writes the global flat index directly.)
    let write_pipeline = create_subpass_pipeline(device, layout, 0, write_module, "fs", &targets);
    let load_pipeline =
        create_subpass_pipeline(device, layout, 1, load_module, "fs_metal", &targets);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    record_two_subpass_draw(
        encoder,
        layout,
        gbuffer_view,
        output_view,
        write_pipeline,
        load_pipeline,
    );
    record_t2b(encoder, output, readback);
    submit_encoder(queue, encoder);

    yawgpu::wgpuRenderPipelineRelease(load_pipeline);
    yawgpu::wgpuRenderPipelineRelease(write_pipeline);
    yawgpu::wgpuShaderModuleRelease(load_module);
    yawgpu::wgpuShaderModuleRelease(write_module);
    yawgpu::yawgpuSubpassPassLayoutRelease(layout);
    yawgpu::wgpuTextureViewRelease(output_view);
    yawgpu::wgpuTextureViewRelease(gbuffer_view);
    yawgpu::wgpuTextureRelease(output);
    yawgpu::wgpuTextureRelease(gbuffer);
    yawgpu::wgpuQueueRelease(queue);
    readback
}

unsafe fn create_two_subpass_input_layout(
    device: native::WGPUDevice,
) -> yawgpu::YaWGPUSubpassPassLayout {
    let attachments = [
        yawgpu::YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sampleCount: 1,
        },
        yawgpu::YaWGPUAttachmentLayout {
            format: native::WGPUTextureFormat_RGBA8Unorm,
            sampleCount: 1,
        },
    ];
    let subpass_0_color = 0_u32;
    let subpass_1_color = 1_u32;
    let input = yawgpu::YaWGPUSubpassInputAttachment {
        group: 0,
        binding: 0,
        sourceSubpass: 0,
        sourceAttachment: 0,
    };
    let subpasses = [
        yawgpu::YaWGPUSubpassLayoutDesc {
            colorAttachmentIndices: &subpass_0_color,
            colorAttachmentIndexCount: 1,
            usesDepthStencil: 0,
            inputAttachments: std::ptr::null(),
            inputAttachmentCount: 0,
        },
        yawgpu::YaWGPUSubpassLayoutDesc {
            colorAttachmentIndices: &subpass_1_color,
            colorAttachmentIndexCount: 1,
            usesDepthStencil: 0,
            inputAttachments: &input,
            inputAttachmentCount: 1,
        },
    ];
    let dependency = yawgpu::YaWGPUSubpassDependency {
        srcSubpass: 0,
        dstSubpass: 1,
        dependencyType: yawgpu::YaWGPUSubpassDependencyType_ColorToInput,
        byRegion: 1,
    };
    let descriptor = yawgpu::YaWGPUSubpassPassLayoutDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        colorAttachments: attachments.as_ptr(),
        colorAttachmentCount: attachments.len(),
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
    pass_layout: yawgpu::YaWGPUSubpassPassLayout,
    subpass_index: u32,
    module: native::WGPUShaderModule,
    fragment_entry: &str,
    targets: &[native::WGPUColorTargetState],
) -> native::WGPURenderPipeline {
    let fragment = native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: string_view(fragment_entry),
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
    let descriptor = yawgpu::YaWGPUSubpassRenderPipelineDescriptor {
        nextInChain: std::ptr::null(),
        base,
        passLayout: pass_layout,
        subpassIndex: subpass_index,
    };
    let pipeline = yawgpu::yawgpuDeviceCreateSubpassRenderPipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
}

unsafe fn record_two_subpass_draw(
    encoder: native::WGPUCommandEncoder,
    layout: yawgpu::YaWGPUSubpassPassLayout,
    gbuffer_view: native::WGPUTextureView,
    output_view: native::WGPUTextureView,
    write_pipeline: native::WGPURenderPipeline,
    load_pipeline: native::WGPURenderPipeline,
) {
    let attachments = [
        subpass_color_binding(gbuffer_view),
        subpass_color_binding(output_view),
    ];
    let descriptor = yawgpu::YaWGPUSubpassRenderPassDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        passLayout: layout,
        extent: texture_extent(),
        colorAttachments: attachments.as_ptr(),
        colorAttachmentCount: attachments.len(),
        depthStencilAttachment: std::ptr::null(),
    };
    let pass = yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    yawgpu::yawgpuSubpassRenderPassEncoderSetPipeline(pass, write_pipeline);
    yawgpu::yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpu::yawgpuSubpassRenderPassEncoderNextSubpass(pass);
    yawgpu::yawgpuSubpassRenderPassEncoderSetPipeline(pass, load_pipeline);
    yawgpu::yawgpuSubpassRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    yawgpu::yawgpuSubpassRenderPassEncoderEnd(pass);
    yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);
}

fn subpass_color_binding(view: native::WGPUTextureView) -> yawgpu::YaWGPUColorAttachmentBinding {
    yawgpu::YaWGPUColorAttachmentBinding {
        kind: yawgpu::YaWGPUSubpassAttachmentKind_Persistent,
        view,
        resolveTarget: std::ptr::null(),
        transient: std::ptr::null(),
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

unsafe fn request_adapter(instance: native::WGPUInstance) -> native::WGPUAdapter {
    let mut adapter = std::ptr::null();
    let callback_info = native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_adapter_callback),
        userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info);
    wait(instance, future);
    adapter
}

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let mut device = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), callback_info);
    wait(instance, future);
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

fn zeroed_tiled_capabilities() -> YaWGPUTiledCapabilities {
    YaWGPUTiledCapabilities {
        nextInChain: std::ptr::null(),
        maxSubpasses: 0,
        maxSubpassColorAttachments: 0,
        maxInputAttachments: 0,
        estimatedTileMemoryBytes: 0,
    }
}

unsafe fn create_single_color_subpass_layout(
    device: native::WGPUDevice,
) -> yawgpu::YaWGPUSubpassPassLayout {
    let color = yawgpu::YaWGPUAttachmentLayout {
        format: native::WGPUTextureFormat_RGBA8Unorm,
        sampleCount: 1,
    };
    let color_index = 0_u32;
    let subpass = yawgpu::YaWGPUSubpassLayoutDesc {
        colorAttachmentIndices: &color_index,
        colorAttachmentIndexCount: 1,
        usesDepthStencil: 0,
        inputAttachments: std::ptr::null(),
        inputAttachmentCount: 0,
    };
    let descriptor = yawgpu::YaWGPUSubpassPassLayoutDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        colorAttachments: &color,
        colorAttachmentCount: 1,
        depthStencilAttachment: std::ptr::null(),
        subpasses: &subpass,
        subpassCount: 1,
        dependencies: std::ptr::null(),
        dependencyCount: 0,
    };
    let layout = yawgpu::yawgpuDeviceCreateSubpassPassLayout(device, &descriptor);
    assert!(!layout.is_null());
    layout
}

unsafe fn record_clear_subpass_pass(
    encoder: native::WGPUCommandEncoder,
    layout: yawgpu::YaWGPUSubpassPassLayout,
    view: native::WGPUTextureView,
) {
    let color = color_binding_for_view(view);
    let descriptor = subpass_render_pass_descriptor(layout, &color);
    let pass = yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    yawgpu::yawgpuSubpassRenderPassEncoderEnd(pass);
    yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);
}

unsafe fn record_clear_transient_subpass_pass(
    encoder: native::WGPUCommandEncoder,
    layout: yawgpu::YaWGPUSubpassPassLayout,
    transient: yawgpu::YaWGPUTransientAttachment,
) {
    let color = yawgpu::YaWGPUColorAttachmentBinding {
        kind: yawgpu::YaWGPUSubpassAttachmentKind_Transient,
        view: std::ptr::null(),
        resolveTarget: std::ptr::null(),
        transient,
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Discard,
        clearValue: native::WGPUColor {
            r: 0.25,
            g: 0.5,
            b: 0.75,
            a: 1.0,
        },
    };
    let descriptor = subpass_render_pass_descriptor(layout, &color);
    let pass = yawgpu::yawgpuCommandEncoderBeginSubpassRenderPass(encoder, &descriptor);
    assert!(!pass.is_null());
    yawgpu::yawgpuSubpassRenderPassEncoderEnd(pass);
    yawgpu::yawgpuSubpassRenderPassEncoderRelease(pass);
}

fn subpass_render_pass_descriptor(
    layout: yawgpu::YaWGPUSubpassPassLayout,
    color: &yawgpu::YaWGPUColorAttachmentBinding,
) -> yawgpu::YaWGPUSubpassRenderPassDescriptor {
    yawgpu::YaWGPUSubpassRenderPassDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        passLayout: layout,
        extent: texture_extent(),
        colorAttachments: color,
        colorAttachmentCount: 1,
        depthStencilAttachment: std::ptr::null(),
    }
}

fn color_binding_for_view(view: native::WGPUTextureView) -> yawgpu::YaWGPUColorAttachmentBinding {
    yawgpu::YaWGPUColorAttachmentBinding {
        kind: yawgpu::YaWGPUSubpassAttachmentKind_Persistent,
        view,
        resolveTarget: std::ptr::null(),
        transient: std::ptr::null(),
        loadOp: native::WGPULoadOp_Clear,
        storeOp: native::WGPUStoreOp_Store,
        clearValue: native::WGPUColor {
            r: 0.25,
            g: 0.5,
            b: 0.75,
            a: 1.0,
        },
    }
}

unsafe fn create_match_target_transient(
    device: native::WGPUDevice,
) -> yawgpu::YaWGPUTransientAttachment {
    let descriptor = yawgpu::YaWGPUTransientAttachmentDescriptor {
        nextInChain: std::ptr::null(),
        label: empty_string_view(),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        sizeMode: yawgpu::YaWGPUTransientSizeMode_MatchTarget,
        width: 0,
        height: 0,
        sampleCount: 1,
    };
    let transient = yawgpu::yawgpuDeviceCreateTransientAttachment(device, &descriptor);
    assert!(!transient.is_null());
    transient
}

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

unsafe fn record_t2b(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
) {
    let source = native::WGPUTexelCopyTextureInfo {
        texture: source,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
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

unsafe fn submit_encoder(queue: native::WGPUQueue, encoder: native::WGPUCommandEncoder) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
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

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
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

fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: 16,
        height: 16,
        depthOrArrayLayers: 1,
    }
}

fn assert_center_pixel_approx(pixels: &[u8], rgba: [u8; 4], tolerance: u8) {
    let center_x = WIDTH as usize / 2;
    let center_y = HEIGHT as usize / 2;
    let offset = center_y * ROW_BYTES + center_x * BYTES_PER_PIXEL;
    let pixel = &pixels[offset..offset + BYTES_PER_PIXEL];
    for (&actual, &expected) in pixel.iter().zip(rgba.iter()) {
        assert!(
            actual.abs_diff(expected) <= tolerance,
            "center pixel {:?} did not match {:?} within {}",
            pixel,
            rgba,
            tolerance
        );
    }
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

const SUBPASS_WRITE_SHADER: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#;

const SUBPASS_LOAD_SHADER: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4<f32>,
}

@group(0) @binding(0) var gbuffer: subpass_input<f32>;

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

// Two fragment entry points mirror the Vulkan / Metal attachment-numbering
// difference (same idea as mgpu's hello_deferred): naga MSL doesn't remap a
// subpass's `@location(N)` to the global MTL color-attachment index, so on
// Metal we need a separate entry whose location matches the MTL slot the
// lighting subpass writes (here: slot 1, the output). On Vulkan the
// subpass-local `@location(0)` is remapped by `VkRenderPass`.
@fragment
fn fs() -> @location(0) vec4<f32> {
    let loaded = subpassLoad(gbuffer);
    return vec4<f32>(loaded.g, loaded.r, loaded.b, 1.0);
}

@fragment
fn fs_metal() -> @location(1) vec4<f32> {
    let loaded = subpassLoad(gbuffer);
    return vec4<f32>(loaded.g, loaded.r, loaded.b, 1.0);
}
"#;
