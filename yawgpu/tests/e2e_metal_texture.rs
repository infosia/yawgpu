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
const BYTES_PER_PIXEL: usize = 4;
const BYTES_PER_ROW: u32 = 256;
const ROW_PIXELS_BYTES: usize = WIDTH as usize * BYTES_PER_PIXEL;
const BUFFER_SIZE: usize = BYTES_PER_ROW as usize * HEIGHT as usize;

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_buffer_texture_buffer_round_trip() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let pixels = source_pixels();
        let destination = run_buffer_texture_buffer_submit(device, &pixels);
        let actual = read_unpacked_texture_buffer(instance, destination);

        assert_eq!(actual, pixels);
        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuBufferRelease(destination);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_texture_texture_round_trip() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let pixels = source_pixels();
        let destination = run_texture_texture_submit(device, &pixels);
        let actual = read_unpacked_texture_buffer(instance, destination);

        assert_eq!(actual, pixels);
        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuBufferRelease(destination);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
#[cfg(feature = "metal")]
fn metal_sampler_creation_has_no_device_error() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let sampler = create_sampler(device);

        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuSamplerRelease(sampler);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn default_noop_texture_and_sampler_path_has_no_device_error() {
    unsafe {
        let instance = yawgpu::wgpuCreateInstance(std::ptr::null());
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        let pixels = source_pixels();
        let b2t_destination = run_buffer_texture_buffer_submit(device, &pixels);
        let t2t_destination = run_texture_texture_submit(device, &pixels);
        let sampler = create_sampler(device);

        assert!(errors.lock().expect("error lock").is_empty());
        yawgpu::wgpuSamplerRelease(sampler);
        yawgpu::wgpuBufferRelease(t2t_destination);
        yawgpu::wgpuBufferRelease(b2t_destination);
        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn run_buffer_texture_buffer_submit(
    device: native::WGPUDevice,
    pixels: &[u8],
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let source = create_buffer(
        device,
        native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
    );
    let destination = create_buffer(
        device,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let texture = create_texture(
        device,
        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
    );
    write_padded_pixels(queue, source, pixels);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    record_b2t(encoder, source, texture);
    record_t2b(encoder, texture, destination);
    submit_encoder(queue, encoder);

    yawgpu::wgpuTextureRelease(texture);
    yawgpu::wgpuBufferRelease(source);
    yawgpu::wgpuQueueRelease(queue);
    destination
}

unsafe fn run_texture_texture_submit(
    device: native::WGPUDevice,
    pixels: &[u8],
) -> native::WGPUBuffer {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let source = create_buffer(
        device,
        native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
    );
    let destination = create_buffer(
        device,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let texture_a = create_texture(
        device,
        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
    );
    let texture_b = create_texture(
        device,
        native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
    );
    write_padded_pixels(queue, source, pixels);

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    record_b2t(encoder, source, texture_a);
    record_t2t(encoder, texture_a, texture_b);
    record_t2b(encoder, texture_b, destination);
    submit_encoder(queue, encoder);

    yawgpu::wgpuTextureRelease(texture_b);
    yawgpu::wgpuTextureRelease(texture_a);
    yawgpu::wgpuBufferRelease(source);
    yawgpu::wgpuQueueRelease(queue);
    destination
}

unsafe fn record_b2t(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUBuffer,
    destination: native::WGPUTexture,
) {
    let source = buffer_copy_info(source);
    let destination = texture_copy_info(destination);
    let size = texture_extent();
    yawgpu::wgpuCommandEncoderCopyBufferToTexture(encoder, &source, &destination, &size);
}

unsafe fn record_t2b(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUBuffer,
) {
    let source = texture_copy_info(source);
    let destination = buffer_copy_info(destination);
    let size = texture_extent();
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &size);
}

unsafe fn record_t2t(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUTexture,
) {
    let source = texture_copy_info(source);
    let destination = texture_copy_info(destination);
    let size = texture_extent();
    yawgpu::wgpuCommandEncoderCopyTextureToTexture(encoder, &source, &destination, &size);
}

unsafe fn submit_encoder(queue: native::WGPUQueue, encoder: native::WGPUCommandEncoder) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn write_padded_pixels(queue: native::WGPUQueue, buffer: native::WGPUBuffer, pixels: &[u8]) {
    let mut padded = vec![0; BUFFER_SIZE];
    for row in 0..HEIGHT as usize {
        let pixel_offset = row * ROW_PIXELS_BYTES;
        let padded_offset = row * BYTES_PER_ROW as usize;
        padded[padded_offset..padded_offset + ROW_PIXELS_BYTES]
            .copy_from_slice(&pixels[pixel_offset..pixel_offset + ROW_PIXELS_BYTES]);
    }
    yawgpu::wgpuQueueWriteBuffer(queue, buffer, 0, padded.as_ptr().cast(), padded.len());
}

#[cfg(feature = "metal")]
unsafe fn read_unpacked_texture_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
) -> Vec<u8> {
    let mapped = read_buffer(instance, buffer, 0, BUFFER_SIZE);
    let mut pixels = vec![0; ROW_PIXELS_BYTES * HEIGHT as usize];
    for row in 0..HEIGHT as usize {
        let pixel_offset = row * ROW_PIXELS_BYTES;
        let padded_offset = row * BYTES_PER_ROW as usize;
        pixels[pixel_offset..pixel_offset + ROW_PIXELS_BYTES]
            .copy_from_slice(&mapped[padded_offset..padded_offset + ROW_PIXELS_BYTES]);
    }
    pixels
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

unsafe fn create_buffer(
    device: native::WGPUDevice,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size: BUFFER_SIZE as u64,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
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

unsafe fn create_sampler(device: native::WGPUDevice) -> native::WGPUSampler {
    let descriptor = native::WGPUSamplerDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        addressModeU: native::WGPUAddressMode_ClampToEdge,
        addressModeV: native::WGPUAddressMode_ClampToEdge,
        addressModeW: native::WGPUAddressMode_ClampToEdge,
        magFilter: native::WGPUFilterMode_Linear,
        minFilter: native::WGPUFilterMode_Linear,
        mipmapFilter: native::WGPUMipmapFilterMode_Nearest,
        lodMinClamp: 0.0,
        lodMaxClamp: 32.0,
        compare: native::WGPUCompareFunction_Undefined,
        maxAnisotropy: 1,
    };
    let sampler = yawgpu::wgpuDeviceCreateSampler(device, &descriptor);
    assert!(!sampler.is_null());
    sampler
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

fn source_pixels() -> Vec<u8> {
    (0..ROW_PIXELS_BYTES * HEIGHT as usize)
        .map(|value| u8::try_from((value * 17 + 3) % 251).expect("test byte fits in u8"))
        .collect()
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

#[cfg(feature = "metal")]
unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
