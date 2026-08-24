//! Real-Vulkan e2e for Block 72 Slice 2 (`specs/blocks/72-texture-formats-tier2.md`):
//! a device without `texture-formats-tier2` rejects a `read-write`
//! `rgba8unorm` storage-texture layout with a message naming the format and
//! the tier, and a device with the feature executes `read_write` compute
//! passes on `rgba8unorm` and on the Vulkan *extended* storage format
//! `r8unorm` (the latter proves `shaderStorageImageExtendedFormats` is
//! enabled on the logical device).

#![cfg(feature = "vulkan")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const WIDTH: u32 = 2;
const HEIGHT: u32 = 2;
const PADDED_BYTES_PER_ROW: u32 = 256;
const READBACK_SIZE: u64 = PADDED_BYTES_PER_ROW as u64 * HEIGHT as u64;

/// Swaps red and green and forces blue to 1.0 through a single read-write
/// storage texture binding — every texel is both loaded and stored.
const RGBA8_READ_WRITE_SHADER: &str = r#"
@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, read_write>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let coord = vec2<i32>(id.xy);
    let texel = textureLoad(tex, coord);
    textureStore(tex, coord, vec4<f32>(texel.g, texel.r, 1.0, texel.a));
}
"#;

/// Inverts the single channel of an `r8unorm` read-write storage texture.
const R8_READ_WRITE_SHADER: &str = r#"
@group(0) @binding(0) var tex: texture_storage_2d<r8unorm, read_write>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let coord = vec2<i32>(id.xy);
    let texel = textureLoad(tex, coord);
    textureStore(tex, coord, vec4<f32>(1.0 - texel.r, 0.0, 0.0, 1.0));
}
"#;

struct RoundTrip {
    format: native::WGPUTextureFormat,
    bytes_per_pixel: usize,
    shader: &'static str,
    initial: Vec<u8>,
    expected: Vec<u8>,
}

fn rgba8unorm_round_trip() -> RoundTrip {
    let mut initial = Vec::new();
    for y in 0..HEIGHT as u8 {
        for x in 0..WIDTH as u8 {
            initial.extend_from_slice(&[x * 80 + 10, y * 80 + 20, 0, 255]);
        }
    }
    let expected = initial
        .chunks_exact(4)
        .flat_map(|texel| [texel[1], texel[0], 255, texel[3]])
        .collect();
    RoundTrip {
        format: native::WGPUTextureFormat_RGBA8Unorm,
        bytes_per_pixel: 4,
        shader: RGBA8_READ_WRITE_SHADER,
        initial,
        expected,
    }
}

fn r8unorm_round_trip() -> RoundTrip {
    let initial: Vec<u8> = (0..(WIDTH * HEIGHT) as u8).map(|i| i * 60 + 5).collect();
    let expected = initial.iter().map(|value| 255 - value).collect();
    RoundTrip {
        format: native::WGPUTextureFormat_R8Unorm,
        bytes_per_pixel: 1,
        shader: R8_READ_WRITE_SHADER,
        initial,
        expected,
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_device_without_tier2_rejects_rgba8unorm_read_write_layout() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter, &[]);
        let errors = install_error_capture(device);

        let bgl = create_read_write_bgl(device, native::WGPUTextureFormat_RGBA8Unorm);
        let captured = errors.lock().expect("error lock");
        assert_eq!(
            captured.len(),
            1,
            "expected exactly one validation error, got {captured:?}"
        );
        let message = &captured[0].message;
        assert!(
            message.contains("rgba8unorm"),
            "message must name the format: {message}"
        );
        assert!(
            message.contains("texture-formats-tier2"),
            "message must name the tier: {message}"
        );
        drop(captured);

        if !bgl.is_null() {
            yawgpu::wgpuBindGroupLayoutRelease(bgl);
        }
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_device_with_tier2_executes_rgba8unorm_read_write_storage() {
    run_round_trip_if_advertised(rgba8unorm_round_trip());
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_device_with_tier2_executes_r8unorm_extended_format_read_write_storage() {
    run_round_trip_if_advertised(r8unorm_round_trip());
}

fn run_round_trip_if_advertised(case: RoundTrip) {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }

    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        if yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_TextureFormatsTier2) == 0
        {
            eprintln!("skipping: this Vulkan adapter does not advertise texture-formats-tier2");
            yawgpu::wgpuAdapterRelease(adapter);
            yawgpu::wgpuInstanceRelease(instance);
            return;
        }
        let device = request_device(
            instance,
            adapter,
            &[native::WGPUFeatureName_TextureFormatsTier2],
        );
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let texture = create_storage_texture(device, case.format);
        write_texture_pixels(queue, texture, &case.initial, case.bytes_per_pixel);
        let view = yawgpu::wgpuTextureCreateView(texture, std::ptr::null());
        let readback = create_buffer(
            device,
            READBACK_SIZE,
            native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        );

        let bgl = create_read_write_bgl(device, case.format);
        assert!(!bgl.is_null());
        let pipeline_layout = create_pipeline_layout(device, bgl);
        let module = create_wgsl_module(device, case.shader);
        let pipeline = create_compute_pipeline(device, module, pipeline_layout);
        let bind_group = create_texture_bind_group(device, bgl, view);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
        yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
        yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
        yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, WIDTH, HEIGHT, 1);
        yawgpu::wgpuComputePassEncoderEnd(pass);
        record_texture_to_buffer(encoder, texture, readback);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuComputePassEncoderRelease(pass);
        yawgpu::wgpuCommandEncoderRelease(encoder);

        let actual = read_unpadded_pixels(instance, readback, case.bytes_per_pixel);
        assert_eq!(
            actual, case.expected,
            "read-write storage texture did not round-trip through the compute pass"
        );
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuBindGroupRelease(bind_group);
        yawgpu::wgpuComputePipelineRelease(pipeline);
        yawgpu::wgpuShaderModuleRelease(module);
        yawgpu::wgpuPipelineLayoutRelease(pipeline_layout);
        yawgpu::wgpuBindGroupLayoutRelease(bgl);
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn create_read_write_bgl(
    device: native::WGPUDevice,
    format: native::WGPUTextureFormat,
) -> native::WGPUBindGroupLayout {
    let mut entry: native::WGPUBindGroupLayoutEntry = std::mem::zeroed();
    entry.binding = 0;
    entry.visibility = native::WGPUShaderStage_Compute;
    entry.storageTexture.access = native::WGPUStorageTextureAccess_ReadWrite;
    entry.storageTexture.format = format;
    entry.storageTexture.viewDimension = native::WGPUTextureViewDimension_2D;
    let descriptor = native::WGPUBindGroupLayoutDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        entryCount: 1,
        entries: &entry,
    };
    yawgpu::wgpuDeviceCreateBindGroupLayout(device, &descriptor)
}

unsafe fn create_storage_texture(
    device: native::WGPUDevice,
    format: native::WGPUTextureFormat,
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_StorageBinding
            | native::WGPUTextureUsage_CopyDst
            | native::WGPUTextureUsage_CopySrc,
        dimension: native::WGPUTextureDimension_2D,
        size: texture_extent(),
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

fn texture_extent() -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: WIDTH,
        height: HEIGHT,
        depthOrArrayLayers: 1,
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

unsafe fn write_texture_pixels(
    queue: native::WGPUQueue,
    texture: native::WGPUTexture,
    pixels: &[u8],
    bytes_per_pixel: usize,
) {
    let destination = texture_copy_info(texture);
    let layout = native::WGPUTexelCopyBufferLayout {
        offset: 0,
        bytesPerRow: (WIDTH as usize * bytes_per_pixel) as u32,
        rowsPerImage: HEIGHT,
    };
    yawgpu::wgpuQueueWriteTexture(
        queue,
        &destination,
        pixels.as_ptr().cast(),
        pixels.len(),
        &layout,
        &texture_extent(),
    );
}

unsafe fn record_texture_to_buffer(
    encoder: native::WGPUCommandEncoder,
    texture: native::WGPUTexture,
    buffer: native::WGPUBuffer,
) {
    let source = texture_copy_info(texture);
    let destination = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: PADDED_BYTES_PER_ROW,
            rowsPerImage: HEIGHT,
        },
        buffer,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
        encoder,
        &source,
        &destination,
        &texture_extent(),
    );
}

unsafe fn read_unpadded_pixels(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    bytes_per_pixel: usize,
) -> Vec<u8> {
    let row_bytes = WIDTH as usize * bytes_per_pixel;
    let mapped = read_buffer(instance, buffer, READBACK_SIZE as usize);
    let mut pixels = Vec::with_capacity(row_bytes * HEIGHT as usize);
    for row in 0..HEIGHT as usize {
        let start = row * PADDED_BYTES_PER_ROW as usize;
        pixels.extend_from_slice(&mapped[start..start + row_bytes]);
    }
    pixels
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

unsafe fn create_compute_pipeline(
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

unsafe fn create_texture_bind_group(
    device: native::WGPUDevice,
    layout: native::WGPUBindGroupLayout,
    view: native::WGPUTextureView,
) -> native::WGPUBindGroup {
    let entry = native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding: 0,
        buffer: std::ptr::null(),
        offset: 0,
        size: 0,
        sampler: std::ptr::null(),
        textureView: view,
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

unsafe fn read_buffer(
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

fn string_view(text: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: text.as_ptr().cast(),
        length: text.len(),
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
