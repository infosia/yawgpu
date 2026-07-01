//! Real-GPU verification of the WebGPU `float32-blendable` feature on Vulkan.
//!
//! Blending onto a 32-bit-float color target is only permitted with the feature
//! enabled. This renders two full-screen triangles into an `rgba32float` target
//! with additive blending (`src=One, dst=One`) over a zero clear: the first draw
//! writes color A, the second adds color B, so the read-back pixel is `A + B`.
//! The colors use values > 1.0 (only representable on a float32 target) and the
//! sum is exactly representable in binary32, so an exact readback proves the
//! blend actually executed on the float32 attachment through the new feature.
#![cfg(feature = "vulkan")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
#[cfg(feature = "vulkan")]
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::wait;
#[cfg(feature = "vulkan")]
use yawgpu_test::{real_backend_skip_reason, RealBackend};

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;
const BYTES_PER_ROW: u32 = 256;
const READBACK_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;

// color A = (0.5, 1.0, 1.5, 2.0); color B = (0.25, 0.5, 0.75, 1.0).
// Additive blend over a zero clear → A + B = (0.75, 1.5, 2.25, 3.0), all exact
// in binary32.
const EXPECTED: [f32; 4] = [0.75, 1.5, 2.25, 3.0];

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_adapter_advertises_float32_blendable() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        assert!(
            yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_Float32Blendable) != 0,
            "Vulkan adapter must advertise float32-blendable"
        );
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "vulkan")]
fn vulkan_float32_target_blends_additively() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device_with_float32_blendable(instance, adapter);
        assert!(
            yawgpu::wgpuDeviceHasFeature(device, native::WGPUFeatureName_Float32Blendable) != 0,
            "device must report float32-blendable after requesting it"
        );
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let module_a = create_wgsl_module(device, &full_shader("vec4<f32>(0.5, 1.0, 1.5, 2.0)"));
        let module_b = create_wgsl_module(device, &full_shader("vec4<f32>(0.25, 0.5, 0.75, 1.0)"));
        let pipeline_a = create_blend_pipeline(device, module_a);
        let pipeline_b = create_blend_pipeline(device, module_b);

        let color = create_color_texture(device);
        let color_view = yawgpu::wgpuTextureCreateView(color, std::ptr::null());
        let color_readback = create_buffer(
            device,
            READBACK_SIZE as u64,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let color_attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view: color_view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
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
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline_a);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderSetPipeline(pass, pipeline_b);
        yawgpu::wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);

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
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &color_src, &color_dst, &texture_extent());
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let bytes = read_buffer(instance, color_readback, 0, READBACK_SIZE);
        let actual = [
            f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            f32::from_ne_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            f32::from_ne_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            f32::from_ne_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
        ];
        for (i, (a, e)) in actual.iter().zip(EXPECTED.iter()).enumerate() {
            assert!(
                (a - e).abs() < 1e-5,
                "channel {i}: additive blend on rgba32float gave {a}, expected {e} (A+B)"
            );
        }
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "float32 blend path raised a device error"
        );

        yawgpu::wgpuBufferRelease(color_readback);
        yawgpu::wgpuTextureViewRelease(color_view);
        yawgpu::wgpuTextureRelease(color);
        yawgpu::wgpuRenderPipelineRelease(pipeline_b);
        yawgpu::wgpuRenderPipelineRelease(pipeline_a);
        yawgpu::wgpuShaderModuleRelease(module_b);
        yawgpu::wgpuShaderModuleRelease(module_a);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

// ---- helpers ----

fn full_shader(color: &str) -> String {
    format!(
        "@vertex\n\
         fn vs(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {{\n\
             var pos = array<vec2<f32>, 3>(\n\
                 vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));\n\
             return vec4<f32>(pos[idx], 0.0, 1.0);\n\
         }}\n\
         @fragment\n\
         fn fs() -> @location(0) vec4<f32> {{ return {color}; }}\n"
    )
}

#[cfg(feature = "vulkan")]
unsafe fn create_blend_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPURenderPipeline {
    let additive = native::WGPUBlendComponent {
        operation: native::WGPUBlendOperation_Add,
        srcFactor: native::WGPUBlendFactor_One,
        dstFactor: native::WGPUBlendFactor_One,
    };
    let blend = native::WGPUBlendState {
        color: additive,
        alpha: additive,
    };
    let color_target = native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format: native::WGPUTextureFormat_RGBA32Float,
        blend: &blend,
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

#[cfg(feature = "vulkan")]
unsafe fn create_color_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
        dimension: native::WGPUTextureDimension_2D,
        size: texture_extent(),
        format: native::WGPUTextureFormat_RGBA32Float,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: 0,
        viewFormats: std::ptr::null(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

#[cfg(feature = "vulkan")]
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

#[cfg(feature = "vulkan")]
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

#[cfg(feature = "vulkan")]
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

#[cfg(feature = "vulkan")]
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

#[cfg(feature = "vulkan")]
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

#[cfg(feature = "vulkan")]
unsafe fn request_device_with_float32_blendable(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let features = [native::WGPUFeatureName_Float32Blendable];
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        requiredFeatureCount: features.len(),
        requiredFeatures: features.as_ptr(),
        requiredLimits: std::ptr::null(),
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        },
        deviceLostCallbackInfo: std::mem::zeroed(),
        uncapturedErrorCallbackInfo: std::mem::zeroed(),
    };
    let mut device: native::WGPUDevice = std::ptr::null();
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

#[cfg(feature = "vulkan")]
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
