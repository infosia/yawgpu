//! Real-Metal e2e for Block 111 S1 (`specs/blocks/111-code-review-hal-majors.md`):
//! R1 — a color attachment uses the view format (a clear through an sRGB view
//! of a unorm texture stores sRGB-encoded bytes), and R2 — 2D-array ↔ 3D
//! texture copies move every layer / slice.

#![cfg(feature = "metal")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const SIZE: u32 = 4;
const LAYERS: u32 = 3;
const BYTES_PER_ROW: u32 = 256;
const ROW_BYTES: usize = SIZE as usize * 4;
const LAYER_BYTES: usize = BYTES_PER_ROW as usize * SIZE as usize;

/// R1: clearing a `rgba8unorm` texture through its `rgba8unorm-srgb` view
/// with linear 0.5 must store the sRGB encoding (~188), not 128.
#[test]
#[ignore = "manual real-backend test"]
fn metal_srgb_view_clear_stores_srgb_encoded_bytes() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let (instance, adapter, device) = metal_device();
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let view_formats = [native::WGPUTextureFormat_RGBA8UnormSrgb];
        let texture = create_texture(
            device,
            native::WGPUTextureDimension_2D,
            1,
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            &view_formats,
        );
        let view_descriptor = native::WGPUTextureViewDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            format: native::WGPUTextureFormat_RGBA8UnormSrgb,
            dimension: native::WGPUTextureViewDimension_2D,
            baseMipLevel: 0,
            mipLevelCount: 1,
            baseArrayLayer: 0,
            arrayLayerCount: 1,
            aspect: native::WGPUTextureAspect_All,
            usage: native::WGPUTextureUsage_None,
        };
        let view = yawgpu::wgpuTextureCreateView(texture, &view_descriptor);
        let readback = create_buffer(device, LAYER_BYTES as u64);

        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.5,
                g: 0.5,
                b: 0.5,
                a: 1.0,
            },
        };
        let pass_descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 1,
            colorAttachments: &attachment,
            depthStencilAttachment: std::ptr::null(),
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let pass = yawgpu::wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        yawgpu::wgpuRenderPassEncoderEnd(pass);
        yawgpu::wgpuRenderPassEncoderRelease(pass);
        record_t2b(encoder, texture, 0, readback, 1);
        submit(queue, encoder);

        let bytes = read_buffer(instance, readback, LAYER_BYTES);
        let texel = &bytes[..4];
        for channel in &texel[..3] {
            assert!(
                channel.abs_diff(188) <= 1,
                "sRGB-encoded 0.5 must be ~188, got texel {texel:?}"
            );
        }
        assert_eq!(texel[3], 255);
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureViewRelease(view);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
        release(instance, adapter, device);
    }
}

/// R2: copy a 3-layer 2D-array texture into a 3D texture (depth 3) and back
/// into a second 2D-array texture; every layer must survive the round trip.
#[test]
#[ignore = "manual real-backend test"]
fn metal_2d_array_to_3d_and_back_copies_every_layer() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let (instance, adapter, device) = metal_device();
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);

        let usage = native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst;
        let source = create_texture(device, native::WGPUTextureDimension_2D, LAYERS, usage, &[]);
        let volume = create_texture(device, native::WGPUTextureDimension_3D, LAYERS, usage, &[]);
        let destination =
            create_texture(device, native::WGPUTextureDimension_2D, LAYERS, usage, &[]);

        let mut expected = vec![0u8; LAYER_BYTES * LAYERS as usize];
        for layer in 0..LAYERS as usize {
            for row in 0..SIZE as usize {
                for byte in 0..ROW_BYTES {
                    expected[layer * LAYER_BYTES + row * BYTES_PER_ROW as usize + byte] =
                        u8::try_from((layer * 61 + row * 13 + byte * 7 + 1) % 251)
                            .expect("byte fits");
                }
            }
        }
        let destination_info = native::WGPUTexelCopyTextureInfo {
            texture: source,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: SIZE,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &destination_info,
            expected.as_ptr().cast(),
            expected.len(),
            &layout,
            &extent(LAYERS),
        );

        let readback = create_buffer(device, (LAYER_BYTES * LAYERS as usize) as u64);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        record_t2t(encoder, source, volume);
        record_t2t(encoder, volume, destination);
        record_t2b(encoder, destination, 0, readback, LAYERS);
        submit(queue, encoder);

        let bytes = read_buffer(instance, readback, LAYER_BYTES * LAYERS as usize);
        for layer in 0..LAYERS as usize {
            for row in 0..SIZE as usize {
                let start = layer * LAYER_BYTES + row * BYTES_PER_ROW as usize;
                assert_eq!(
                    &bytes[start..start + ROW_BYTES],
                    &expected[start..start + ROW_BYTES],
                    "layer {layer} row {row} must survive the 2D-array -> 3D -> 2D-array round trip"
                );
            }
        }
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected errors: {:?}",
            errors.lock().expect("error lock")
        );

        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(destination);
        yawgpu::wgpuTextureRelease(volume);
        yawgpu::wgpuTextureRelease(source);
        yawgpu::wgpuQueueRelease(queue);
        release(instance, adapter, device);
    }
}

unsafe fn record_t2t(
    encoder: native::WGPUCommandEncoder,
    source: native::WGPUTexture,
    destination: native::WGPUTexture,
) {
    let info = |texture| native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
        aspect: native::WGPUTextureAspect_All,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToTexture(
        encoder,
        &info(source),
        &info(destination),
        &extent(LAYERS),
    );
}

unsafe fn record_t2b(
    encoder: native::WGPUCommandEncoder,
    texture: native::WGPUTexture,
    layer: u32,
    buffer: native::WGPUBuffer,
    layers: u32,
) {
    let source = native::WGPUTexelCopyTextureInfo {
        texture,
        mipLevel: 0,
        origin: native::WGPUOrigin3D {
            x: 0,
            y: 0,
            z: layer,
        },
        aspect: native::WGPUTextureAspect_All,
    };
    let destination = native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: BYTES_PER_ROW,
            rowsPerImage: SIZE,
        },
        buffer,
    };
    yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &extent(layers));
}

unsafe fn submit(queue: native::WGPUQueue, encoder: native::WGPUCommandEncoder) {
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
}

unsafe fn create_texture(
    device: native::WGPUDevice,
    dimension: native::WGPUTextureDimension,
    depth_or_layers: u32,
    usage: native::WGPUTextureUsage,
    view_formats: &[native::WGPUTextureFormat],
) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        dimension,
        size: extent(depth_or_layers),
        format: native::WGPUTextureFormat_RGBA8Unorm,
        mipLevelCount: 1,
        sampleCount: 1,
        viewFormatCount: view_formats.len(),
        viewFormats: view_formats.as_ptr(),
    };
    let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
    assert!(!texture.is_null());
    texture
}

unsafe fn create_buffer(device: native::WGPUDevice, size: u64) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        size,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

fn extent(depth_or_layers: u32) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: SIZE,
        height: SIZE,
        depthOrArrayLayers: depth_or_layers,
    }
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

unsafe fn metal_device() -> (
    native::WGPUInstance,
    native::WGPUAdapter,
    native::WGPUDevice,
) {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: YAWGPU_INSTANCE_BACKEND_METAL,
    };
    let instance_descriptor = native::WGPUInstanceDescriptor {
        nextInChain: (&mut backend.chain) as *mut native::WGPUChainedStruct,
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
    };
    let instance = yawgpu::wgpuCreateInstance(&instance_descriptor);
    assert!(!instance.is_null());

    let mut adapter: native::WGPUAdapter = std::ptr::null();
    let adapter_callback = native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_adapter_callback),
        userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), adapter_callback);
    wait(instance, future);
    assert!(!adapter.is_null());

    let mut device: native::WGPUDevice = std::ptr::null();
    let device_descriptor = native::WGPUDeviceDescriptor {
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
    let device_callback = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, &device_descriptor, device_callback);
    wait(instance, future);
    assert!(!device.is_null());
    (instance, adapter, device)
}

unsafe fn release(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    device: native::WGPUDevice,
) {
    yawgpu::wgpuDeviceRelease(device);
    yawgpu::wgpuAdapterRelease(adapter);
    yawgpu::wgpuInstanceRelease(instance);
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

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
