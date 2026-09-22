//! Native Vulkan verification of texture-component-swizzle and the depth base.
#![cfg(feature = "vulkan")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};
use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const SHADER: &str = r#"
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4f>;
@compute @workgroup_size(1) fn main() { out[0] = textureLoad(t, vec2i(0, 0), 0); }
"#;

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_swizzle_remaps_color_channels() {
    if let Some(reason) = real_backend_skip_reason(RealBackend::Vulkan) {
        eprintln!("Skipping Vulkan swizzle: {reason}");
        return;
    }
    unsafe {
        run_read_case(
            "color (B, G, R, One)",
            native::WGPUTextureFormat_RGBA8Unorm,
            &[0x10, 0x20, 0x30, 0x40],
            Some(native::WGPUTextureComponentSwizzle {
                r: native::WGPUComponentSwizzle_B,
                g: native::WGPUComponentSwizzle_G,
                b: native::WGPUComponentSwizzle_R,
                a: native::WGPUComponentSwizzle_One,
            }),
            [48.0 / 255.0, 32.0 / 255.0, 16.0 / 255.0, 1.0],
            1.0 / 255.0,
        );
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_swizzle_composes_over_the_depth_base() {
    if let Some(reason) = real_backend_skip_reason(RealBackend::Vulkan) {
        eprintln!("Skipping Vulkan swizzle: {reason}");
        return;
    }
    unsafe {
        run_read_case(
            "depth (G, R, A, B)",
            native::WGPUTextureFormat_Depth16Unorm,
            &[0x00, 0x40],
            Some(native::WGPUTextureComponentSwizzle {
                r: native::WGPUComponentSwizzle_G,
                g: native::WGPUComponentSwizzle_R,
                b: native::WGPUComponentSwizzle_A,
                a: native::WGPUComponentSwizzle_B,
            }),
            [0.0, 16384.0 / 65535.0, 1.0, 0.0],
            1e-3,
        );
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_swizzle_identity_on_depth_reads_d001() {
    if let Some(reason) = real_backend_skip_reason(RealBackend::Vulkan) {
        eprintln!("Skipping Vulkan swizzle: {reason}");
        return;
    }
    unsafe {
        run_read_case(
            "depth identity (no chain)",
            native::WGPUTextureFormat_Depth16Unorm,
            &[0x00, 0x40],
            None,
            [16384.0 / 65535.0, 0.0, 0.0, 1.0],
            1e-3,
        );
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_non_identity_swizzle_without_feature_is_a_device_error() {
    if let Some(reason) = real_backend_skip_reason(RealBackend::Vulkan) {
        eprintln!("Skipping Vulkan swizzle: {reason}");
        return;
    }
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
        assert_eq!(
            yawgpu::wgpuDeviceHasFeature(device, native::WGPUFeatureName_TextureComponentSwizzle),
            0
        );
        let errors = install_error_capture(device);
        let texture = create_texture(device, native::WGPUTextureFormat_RGBA8Unorm);
        assert_no_errors(&errors, "texture creation without feature");
        let view = create_view(
            texture,
            native::WGPUTextureAspect_All,
            Some(native::WGPUTextureComponentSwizzle {
                r: native::WGPUComponentSwizzle_G,
                g: native::WGPUComponentSwizzle_R,
                b: native::WGPUComponentSwizzle_B,
                a: native::WGPUComponentSwizzle_A,
            }),
        );
        {
            let captured = errors.lock().expect("error lock");
            for error in captured.iter() {
                println!(
                    "swizzle without feature: {:?}: {}",
                    error.kind, error.message
                );
            }
            assert_eq!(captured.len(), 1, "{captured:?}");
            assert_eq!(captured[0].kind, yawgpu_core::ErrorKind::Validation);
            let message = captured[0].message.to_ascii_lowercase();
            assert!(
                message.contains("swizzle") || message.contains("feature"),
                "{message}"
            );
        }
        yawgpu::wgpuTextureViewRelease(view);
        assert_eq!(errors.lock().expect("error lock").len(), 1);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn run_read_case(
    name: &str,
    format: native::WGPUTextureFormat,
    texel: &[u8],
    swizzle: Option<native::WGPUTextureComponentSwizzle>,
    expected: [f32; 4],
    tolerance: f32,
) {
    let instance = create_vulkan_instance();
    let adapter = request_adapter(instance);
    if yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_TextureComponentSwizzle) == 0
    {
        eprintln!("Skipping {name}: Vulkan adapter does not advertise texture-component-swizzle");
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
        return;
    }
    let device = request_device(
        instance,
        adapter,
        &[native::WGPUFeatureName_TextureComponentSwizzle],
    );
    assert_ne!(
        yawgpu::wgpuDeviceHasFeature(device, native::WGPUFeatureName_TextureComponentSwizzle),
        0
    );
    let errors = install_error_capture(device);
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let texture = create_texture(device, format);
    let aspect = if format == native::WGPUTextureFormat_Depth16Unorm {
        native::WGPUTextureAspect_DepthOnly
    } else {
        native::WGPUTextureAspect_All
    };
    let destination = native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect,
    };
    let layout = native::WGPUTexelCopyBufferLayout {
        offset: 0,
        bytesPerRow: 256,
        rowsPerImage: 1,
    };
    let mut padded = [0_u8; 256];
    padded[..texel.len()].copy_from_slice(texel);
    yawgpu::wgpuQueueWriteTexture(
        queue,
        &destination,
        padded.as_ptr().cast(),
        padded.len(),
        &layout,
        &extent(),
    );
    assert_no_errors(&errors, "writeTexture");
    let view = create_view(texture, aspect, swizzle);
    assert_no_errors(&errors, "texture view creation");
    assert!(!view.is_null());
    let actual = load_texel(instance, device, queue, view, &errors);
    println!("{name}: {actual:?}");
    if tolerance == 0.0 {
        assert_eq!(actual, expected, "{name}");
    } else {
        for (channel, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{name}, channel {channel}: {actual} != {expected}"
            );
        }
    }
    assert_no_errors(&errors, name);
    yawgpu::wgpuTextureViewRelease(view);
    yawgpu::wgpuTextureRelease(texture);
    yawgpu::wgpuQueueRelease(queue);
    yawgpu::wgpuDeviceRelease(device);
    yawgpu::wgpuAdapterRelease(adapter);
    yawgpu::wgpuInstanceRelease(instance);
}

fn assert_no_errors(errors: &Arc<Mutex<Vec<yawgpu_core::DeviceError>>>, stage: &str) {
    let captured = errors.lock().expect("error lock");
    for error in captured.iter() {
        eprintln!("{stage}: {:?}: {}", error.kind, error.message);
    }
    assert!(captured.is_empty(), "{stage}: {captured:?}");
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    format: native::WGPUTextureFormat,
) -> native::WGPUTexture {
    let mut descriptor: native::WGPUTextureDescriptor = std::mem::zeroed();
    descriptor.usage = native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst;
    descriptor.dimension = native::WGPUTextureDimension_2D;
    descriptor.size = extent();
    descriptor.format = format;
    descriptor.mipLevelCount = 1;
    descriptor.sampleCount = 1;
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

unsafe fn create_view(
    texture: native::WGPUTexture,
    aspect: native::WGPUTextureAspect,
    swizzle: Option<native::WGPUTextureComponentSwizzle>,
) -> native::WGPUTextureView {
    let mut chain = swizzle.map(|swizzle| native::WGPUTextureComponentSwizzleDescriptor {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_TextureComponentSwizzleDescriptor,
        },
        swizzle,
    });
    let mut descriptor: native::WGPUTextureViewDescriptor = std::mem::zeroed();
    descriptor.dimension = native::WGPUTextureViewDimension_2D;
    descriptor.mipLevelCount = 1;
    descriptor.arrayLayerCount = 1;
    descriptor.aspect = aspect;
    if let Some(chain) = chain.as_mut() {
        descriptor.nextInChain = &mut chain.chain;
    }
    yawgpu::wgpuTextureCreateView(texture, &descriptor)
}

fn extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: 1,
        height: 1,
        depthOrArrayLayers: 1,
    }
}

unsafe fn load_texel(
    instance: native::WGPUInstance,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    view: native::WGPUTextureView,
    errors: &Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
) -> [f32; 4] {
    let output = create_buffer(
        device,
        16,
        native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopySrc,
    );
    let readback = create_buffer(
        device,
        16,
        native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst,
    );
    let module = create_wgsl_module(device, SHADER);
    // An explicit unfilterable-float binding accepts the depth texture for textureLoad.
    let mut entries: [native::WGPUBindGroupLayoutEntry; 2] = std::mem::zeroed();
    entries[0].binding = 0;
    entries[0].visibility = native::WGPUShaderStage_Compute;
    entries[0].texture.sampleType = native::WGPUTextureSampleType_UnfilterableFloat;
    entries[0].texture.viewDimension = native::WGPUTextureViewDimension_2D;
    entries[1].binding = 1;
    entries[1].visibility = native::WGPUShaderStage_Compute;
    entries[1].buffer.type_ = native::WGPUBufferBindingType_Storage;
    entries[1].buffer.minBindingSize = 16;
    let mut group_descriptor: native::WGPUBindGroupLayoutDescriptor = std::mem::zeroed();
    group_descriptor.entryCount = entries.len();
    group_descriptor.entries = entries.as_ptr();
    let group_layout = yawgpu::wgpuDeviceCreateBindGroupLayout(device, &group_descriptor);
    assert_no_errors(errors, "bind group layout creation");
    let mut layout_descriptor: native::WGPUPipelineLayoutDescriptor = std::mem::zeroed();
    layout_descriptor.bindGroupLayoutCount = 1;
    layout_descriptor.bindGroupLayouts = &group_layout;
    let pipeline_layout = yawgpu::wgpuDeviceCreatePipelineLayout(device, &layout_descriptor);
    let mut descriptor: native::WGPUComputePipelineDescriptor = std::mem::zeroed();
    descriptor.layout = pipeline_layout;
    descriptor.compute.module = module;
    descriptor.compute.entryPoint = string_view("main");
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
    assert_no_errors(errors, "compute pipeline creation");
    assert!(!pipeline.is_null());
    let mut bindings: [native::WGPUBindGroupEntry; 2] = std::mem::zeroed();
    bindings[0].binding = 0;
    bindings[0].textureView = view;
    bindings[1].binding = 1;
    bindings[1].buffer = output;
    bindings[1].size = 16;
    let mut descriptor: native::WGPUBindGroupDescriptor = std::mem::zeroed();
    descriptor.layout = group_layout;
    descriptor.entryCount = bindings.len();
    descriptor.entries = bindings.as_ptr();
    let group = yawgpu::wgpuDeviceCreateBindGroup(device, &descriptor);
    assert_no_errors(errors, "texture_2d<f32> UnfilterableFloat binding");
    assert!(!group.is_null());
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, std::ptr::null());
    yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(encoder, output, 0, readback, 0, 16);
    let commands = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    assert_no_errors(errors, "compute command encoding");
    yawgpu::wgpuQueueSubmit(queue, 1, &commands);
    assert_no_errors(errors, "compute submission");
    yawgpu::wgpuCommandBufferRelease(commands);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    let bytes = read_buffer(instance, readback, 0, 16);
    let values = std::array::from_fn(|i| {
        f32::from_ne_bytes(bytes[i * 4..i * 4 + 4].try_into().expect("four bytes"))
    });
    yawgpu::wgpuBindGroupRelease(group);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
    yawgpu::wgpuBindGroupLayoutRelease(group_layout);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuBufferRelease(readback);
    yawgpu::wgpuBufferRelease(output);
    values
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
unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    features: &[native::WGPUFeatureName],
) -> native::WGPUDevice {
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

#[cfg(feature = "vulkan")]
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
