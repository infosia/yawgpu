//! F-115 repro: `textureLoad` on the depth aspect of a COMBINED depth+stencil
//! format errors on Metal, while depth-ONLY formats pass.
//!
//! For each format, build a compute pipeline that does
//! `textureLoad(t: texture_depth_2d, ...)`, create the texture + a depth-aspect
//! view + a bind group, dispatch, and submit — printing how many device errors
//! were captured AFTER EACH STEP so the failing step (view / bind group /
//! pipeline / submit) is pinpointed. Manual real-backend diagnostic.
//!
//! Run:
//!   cargo test -p yawgpu --features metal --test e2e_metal_f115 \
//!     f115_texture_load_depth_aspect -- --ignored --exact --nocapture

#![cfg(feature = "metal")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, RealBackend};

// Mirrors the CTS structural sampling shader: a sampler is DECLARED at binding 1
// (unused for textureLoad) and an explicit bind group layout (tex+sampler+buffer)
// is used — matching how CTS sets up textureLoad.
const LOAD_SHADER: &str = "\
@group(0) @binding(0) var t: texture_depth_2d;\n\
@group(0) @binding(1) var s: sampler;\n\
@group(0) @binding(2) var<storage, read_write> out: f32;\n\
@compute @workgroup_size(1)\n\
fn main() {\n\
    out = textureLoad(t, vec2<i32>(0, 0), 0);\n\
}\n";

#[test]
#[ignore = "manual real-backend F-115 repro"]
fn f115_texture_load_depth_aspect() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        eprintln!("f115: skipped (no real Metal device)");
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        // Explicit bind group layout (texture=Depth + sampler + storage buffer),
        // mirroring CTS. Pipeline is format-independent; build once.
        let bgl = create_explicit_bgl(device);
        let pl_desc = native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            bindGroupLayoutCount: 1,
            bindGroupLayouts: &bgl,
            immediateSize: 0,
        };
        let pipeline_layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &pl_desc);
        assert!(!pipeline_layout.is_null());
        let module = create_wgsl_module(device, LOAD_SHADER);
        let pipeline = make_compute_pipeline(device, module, pipeline_layout);
        eprintln!(
            "F-115 pipeline build: null={} errors={}",
            pipeline.is_null(),
            errors.lock().expect("lock").len()
        );
        let sampler_desc = native::WGPUSamplerDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            addressModeU: native::WGPUAddressMode_ClampToEdge,
            addressModeV: native::WGPUAddressMode_ClampToEdge,
            addressModeW: native::WGPUAddressMode_ClampToEdge,
            magFilter: native::WGPUFilterMode_Nearest,
            minFilter: native::WGPUFilterMode_Nearest,
            mipmapFilter: native::WGPUMipmapFilterMode_Nearest,
            lodMinClamp: 0.0,
            lodMaxClamp: 32.0,
            compare: native::WGPUCompareFunction_Undefined,
            maxAnisotropy: 1,
        };
        let sampler = yawgpu::wgpuDeviceCreateSampler(device, &sampler_desc);
        assert!(!sampler.is_null());

        for (label, format) in [
            ("Depth32Float       (depth-only)", native::WGPUTextureFormat_Depth32Float),
            ("Depth32FloatStencil8 (combined)", native::WGPUTextureFormat_Depth32FloatStencil8),
            ("Depth24PlusStencil8  (combined)", native::WGPUTextureFormat_Depth24PlusStencil8),
        ] {
            errors.lock().expect("lock").clear();
            eprintln!("--- F-115 format: {label} ---");
            run_one(device, queue, pipeline, bgl, sampler, format, &errors);
            let errs = errors.lock().expect("lock");
            for e in errs.iter() {
                eprintln!("    error: {e:?}");
            }
        }

        yawgpu::wgpuSamplerRelease(sampler);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn run_one(
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    pipeline: native::WGPUComputePipeline,
    bgl: native::WGPUBindGroupLayout,
    sampler: native::WGPUSampler,
    format: native::WGPUTextureFormat,
    errors: &Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
) {
    let snap = |tag: &str| eprintln!("    after {tag}: errors={}", errors.lock().expect("lock").len());

    let tex_desc = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_TextureBinding,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        },
        format,
        mipLevelCount: 3,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &tex_desc);
    snap("createTexture");
    assert!(!texture.is_null());

    // initializeDepthTexture (mirrors CTS): render pass with a depth-stencil
    // attachment (aspect=All view) clearing depth, and stencil too for combined
    // formats. This is the step the earlier minimal repro skipped.
    let has_stencil = matches!(
        format,
        native::WGPUTextureFormat_Depth32FloatStencil8 | native::WGPUTextureFormat_Depth24PlusStencil8
    );
    let att_view_desc = native::WGPUTextureViewDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        // CTS sets the init attachment view format to the texture's own format
        // (line 1672). This is what triggers F-115 for combined formats.
        format,
        dimension: native::WGPUTextureViewDimension_2D,
        baseMipLevel: 0,
        mipLevelCount: 1,
        baseArrayLayer: 0,
        arrayLayerCount: 1,
        aspect: native::WGPUTextureAspect_All,
        usage: native::WGPUTextureUsage_RenderAttachment,
    };
    let att_view = yawgpu::wgpuTextureCreateView(texture, &att_view_desc);
    let ds = native::WGPURenderPassDepthStencilAttachment {
        nextInChain: std::ptr::null_mut(),
        view: att_view,
        depthLoadOp: native::WGPULoadOp_Clear,
        depthStoreOp: native::WGPUStoreOp_Store,
        depthClearValue: 0.5,
        depthReadOnly: 0,
        stencilLoadOp: if has_stencil {
            native::WGPULoadOp_Clear
        } else {
            native::WGPULoadOp_Undefined
        },
        stencilStoreOp: if has_stencil {
            native::WGPUStoreOp_Store
        } else {
            native::WGPUStoreOp_Undefined
        },
        stencilClearValue: 0,
        stencilReadOnly: 0,
    };
    let init_pass_desc = native::WGPURenderPassDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        colorAttachmentCount: 0,
        colorAttachments: std::ptr::null(),
        depthStencilAttachment: &ds,
        occlusionQuerySet: std::ptr::null(),
        timestampWrites: std::ptr::null(),
    };
    let init_enc = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let init_rp = yawgpu::wgpuCommandEncoderBeginRenderPass(init_enc, &init_pass_desc);
    yawgpu::wgpuRenderPassEncoderEnd(init_rp);
    yawgpu::wgpuRenderPassEncoderRelease(init_rp);
    let init_cb = yawgpu::wgpuCommandEncoderFinish(init_enc, std::ptr::null());
    snap("initEncoderFinish");
    yawgpu::wgpuQueueSubmit(queue, 1, &init_cb);
    snap("initSubmit");
    yawgpu::wgpuCommandBufferRelease(init_cb);
    yawgpu::wgpuCommandEncoderRelease(init_enc);
    yawgpu::wgpuTextureViewRelease(att_view);

    // CTS sets the depth-only view format to the aspect-specific format
    // (viewFormatForAspect): depth24plus-stencil8 -> depth24plus,
    // depth32float-stencil8 -> depth32float. This is what F-115 rejects.
    let aspect_format = match format {
        native::WGPUTextureFormat_Depth32FloatStencil8 => native::WGPUTextureFormat_Depth32Float,
        native::WGPUTextureFormat_Depth24PlusStencil8 => native::WGPUTextureFormat_Depth24Plus,
        other => other,
    };
    let view_desc = native::WGPUTextureViewDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        format: aspect_format,
        dimension: native::WGPUTextureViewDimension_2D,
        baseMipLevel: 0,
        mipLevelCount: 3,
        baseArrayLayer: 0,
        arrayLayerCount: 1,
        aspect: native::WGPUTextureAspect_DepthOnly,
        usage: native::WGPUTextureUsage_TextureBinding,
    };
    let view = yawgpu::wgpuTextureCreateView(texture, &view_desc);
    snap("createView(DepthOnly)");

    let out = create_buffer(
        device,
        4,
        native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
    );
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
            buffer: std::ptr::null(),
            offset: 0,
            size: 0,
            sampler,
            textureView: std::ptr::null(),
        },
        native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 2,
            buffer: out,
            offset: 0,
            size: 4,
            sampler: std::ptr::null(),
            textureView: std::ptr::null(),
        },
    ];
    let bg_desc = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: bgl,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(device, &bg_desc);
    snap("createBindGroup");

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    snap("encoderFinish");
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    snap("queueSubmit");
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);

    // Cleanup
    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuTextureViewRelease(view);
    yawgpu::wgpuBufferRelease(out);
    yawgpu::wgpuTextureRelease(texture);
    let _ = queue;
}

unsafe fn create_explicit_bgl(device: native::WGPUDevice) -> native::WGPUBindGroupLayout {
    let mut tex: native::WGPUBindGroupLayoutEntry = std::mem::zeroed();
    tex.binding = 0;
    tex.visibility = native::WGPUShaderStage_Compute;
    tex.texture.sampleType = native::WGPUTextureSampleType_Depth;
    tex.texture.viewDimension = native::WGPUTextureViewDimension_2D;
    let mut samp: native::WGPUBindGroupLayoutEntry = std::mem::zeroed();
    samp.binding = 1;
    samp.visibility = native::WGPUShaderStage_Compute;
    samp.sampler.type_ = native::WGPUSamplerBindingType_Filtering;
    let mut buf: native::WGPUBindGroupLayoutEntry = std::mem::zeroed();
    buf.binding = 2;
    buf.visibility = native::WGPUShaderStage_Compute;
    buf.buffer.type_ = native::WGPUBufferBindingType_Storage;
    let entries = [tex, samp, buf];
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let bgl = yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor);
    assert!(!bgl.is_null());
    bgl
}

unsafe fn make_compute_pipeline(
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
    yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor)
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
    yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor)
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

unsafe fn install_error_capture(
    device: native::WGPUDevice,
) -> Arc<Mutex<Vec<yawgpu_core::DeviceError>>> {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&errors);
    yawgpu::testing_set_uncaptured_error_callback(
        device,
        Some(move |error| captured.lock().expect("lock").push(error)),
    );
    errors
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
    yawgpu_test::wait(instance, future);
    assert!(!adapter.is_null());
    adapter
}

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let mut device: native::WGPUDevice = std::ptr::null();
    // depth32float-stencil8 is an optional WebGPU feature; enable it so the
    // combined-format texture can be created.
    let required_features = [native::WGPUFeatureName_Depth32FloatStencil8];
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        requiredFeatureCount: required_features.len(),
        requiredFeatures: required_features.as_ptr(),
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
    yawgpu_test::wait(instance, future);
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
