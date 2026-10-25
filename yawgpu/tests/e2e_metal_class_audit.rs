//! Real-Metal e2e for Block 115 S1 (`specs/blocks/115-class-audit-metal-gles-tiled.md`):
//! T5 — compressed-texture copies at a mip whose real size is not a
//! multiple of the block size must be clamped to the texture edge (run
//! under `MTL_DEBUG_LAYER=1`; the unclamped copy is a Metal API violation).

#![cfg(feature = "metal")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

/// T5: a 12x12 BC1 texture's mip 1 is 6x6 (physically 8x8 in blocks).
/// writeTexture + copyTextureToBuffer with the WebGPU block-rounded 8x8
/// extent must round-trip the two block rows exactly.
#[test]
#[ignore = "manual real-backend test"]
fn metal_bc1_mip1_unaligned_round_trip() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let Some((instance, adapter, device)) =
            metal_device(&[native::WGPUFeatureName_TextureCompressionBC])
        else {
            return;
        };
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let descriptor = native::WGPUTextureDescriptor {
            usage: native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
            dimension: native::WGPUTextureDimension_2D,
            size: native::WGPUExtent3D {
                width: 12,
                height: 12,
                depthOrArrayLayers: 1,
            },
            format: native::WGPUTextureFormat_BC1RGBAUnorm,
            mipLevelCount: 2,
            sampleCount: 1,
            ..std::mem::zeroed()
        };
        let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
        assert!(!texture.is_null());
        // 2x2 blocks of 8 bytes; bytesPerRow 256 for the copy alignment.
        let bytes_per_row = 256u32;
        let mut data = vec![0u8; bytes_per_row as usize * 2];
        for row in 0..2usize {
            for byte in 0..16usize {
                data[row * bytes_per_row as usize + byte] =
                    u8::try_from((row * 16 + byte) * 9 % 251 + 1).expect("byte");
            }
        }
        let info = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: 1,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: bytes_per_row,
            rowsPerImage: 2,
        };
        let extent = native::WGPUExtent3D {
            width: 8,
            height: 8,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &info,
            data.as_ptr().cast(),
            data.len(),
            &layout,
            &extent,
        );
        let readback = create_buffer(device, data.len() as u64);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let destination = native::WGPUTexelCopyBufferInfo {
            layout,
            buffer: readback,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &info, &destination, &extent);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        let bytes = read_buffer(instance, readback, data.len());
        for row in 0..2usize {
            let start = row * bytes_per_row as usize;
            assert_eq!(
                &bytes[start..start + 16],
                &data[start..start + 16],
                "block row {row}"
            );
        }
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected errors: {:?}",
            errors.lock().expect("error lock")
        );
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuQueueRelease(queue);
        release(instance, adapter, device);
    }
}

/// Block 115 Phase Review M1: BC1 texture->texture copy between mips whose
/// real sizes differ (12x12 mip 1 = 6x6, 20x20 mip 2 = 5x5; both physically
/// 8x8). The WebGPU copy size 8x8 must reach Metal as a valid size for both
/// sides (run under `MTL_DEBUG_LAYER=1`) and round-trip the two block rows.
#[test]
#[ignore = "manual real-backend test"]
fn metal_bc1_texture_to_texture_mismatched_mips_round_trip() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let Some((instance, adapter, device)) =
            metal_device(&[native::WGPUFeatureName_TextureCompressionBC])
        else {
            return;
        };
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let make = |size: u32| {
            let descriptor = native::WGPUTextureDescriptor {
                usage: native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                dimension: native::WGPUTextureDimension_2D,
                size: native::WGPUExtent3D {
                    width: size,
                    height: size,
                    depthOrArrayLayers: 1,
                },
                format: native::WGPUTextureFormat_BC1RGBAUnorm,
                mipLevelCount: 3,
                sampleCount: 1,
                ..std::mem::zeroed()
            };
            let texture = yawgpu::wgpuDeviceCreateTexture(device, &descriptor);
            assert!(!texture.is_null());
            texture
        };
        let source = make(12);
        let destination = make(20);
        let bytes_per_row = 256u32;
        let mut data = vec![0u8; bytes_per_row as usize * 2];
        for row in 0..2usize {
            for byte in 0..16usize {
                data[row * bytes_per_row as usize + byte] =
                    u8::try_from((row * 16 + byte) * 7 % 251 + 3).expect("byte");
            }
        }
        let info = |texture, mip_level| native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: mip_level,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let layout = native::WGPUTexelCopyBufferLayout {
            offset: 0,
            bytesPerRow: bytes_per_row,
            rowsPerImage: 2,
        };
        let extent = native::WGPUExtent3D {
            width: 8,
            height: 8,
            depthOrArrayLayers: 1,
        };
        yawgpu::wgpuQueueWriteTexture(
            queue,
            &info(source, 1),
            data.as_ptr().cast(),
            data.len(),
            &layout,
            &extent,
        );
        let readback = create_buffer(device, data.len() as u64);
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        yawgpu::wgpuCommandEncoderCopyTextureToTexture(
            encoder,
            &info(source, 1),
            &info(destination, 2),
            &extent,
        );
        let target = native::WGPUTexelCopyBufferInfo {
            layout,
            buffer: readback,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(
            encoder,
            &info(destination, 2),
            &target,
            &extent,
        );
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        let bytes = read_buffer(instance, readback, data.len());
        for row in 0..2usize {
            let start = row * bytes_per_row as usize;
            assert_eq!(
                &bytes[start..start + 16],
                &data[start..start + 16],
                "block row {row}"
            );
        }
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected errors: {:?}",
            errors.lock().expect("error lock")
        );
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(destination);
        yawgpu::wgpuTextureRelease(source);
        yawgpu::wgpuQueueRelease(queue);
        release(instance, adapter, device);
    }
}

/// Block 115 T8 / Phase Review m1: a buffer->texture copy whose
/// `bytesPerRow` exceeds Metal's per-row limit, from a buffer sized exactly
/// to the WebGPU footprint (`bytesPerRow * (rows - 1) + rowBytes`).
#[test]
#[ignore = "manual real-backend test"]
fn metal_buffer_to_texture_with_oversized_bytes_per_row() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let Some((instance, adapter, device)) = metal_device(&[]) else {
            return;
        };
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let wide = 70_144u32; // > 16384 texels * 4 bytes, multiple of 256
        let rows = 4usize;
        let row_bytes = 16usize;
        let total = wide as usize * (rows - 1) + row_bytes;
        let mut data = vec![0u8; total];
        for row in 0..rows {
            for byte in 0..row_bytes {
                data[row * wide as usize + byte] =
                    u8::try_from((row * 31 + byte * 5) % 251 + 1).expect("byte");
            }
        }
        let upload_descriptor = native::WGPUBufferDescriptor {
            usage: native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
            size: u64::try_from(total.next_multiple_of(4)).expect("size"),
            ..std::mem::zeroed()
        };
        let upload = yawgpu::wgpuDeviceCreateBuffer(device, &upload_descriptor);
        assert!(!upload.is_null());
        yawgpu::wgpuQueueWriteBuffer(
            queue,
            upload,
            0,
            data.as_ptr().cast(),
            total.next_multiple_of(4).min(data.len()) / 4 * 4,
        );
        let texture_descriptor = native::WGPUTextureDescriptor {
            usage: native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
            dimension: native::WGPUTextureDimension_2D,
            size: native::WGPUExtent3D {
                width: 4,
                height: 4,
                depthOrArrayLayers: 1,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm,
            mipLevelCount: 1,
            sampleCount: 1,
            ..std::mem::zeroed()
        };
        let texture = yawgpu::wgpuDeviceCreateTexture(device, &texture_descriptor);
        assert!(!texture.is_null());
        let readback = create_buffer(device, 256 * 4);
        let texture_info = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: 0,
            origin: native::WGPUOrigin3D { x: 0, y: 0, z: 0 },
            aspect: native::WGPUTextureAspect_All,
        };
        let extent = native::WGPUExtent3D {
            width: 4,
            height: 4,
            depthOrArrayLayers: 1,
        };
        let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let source = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: wide,
                rowsPerImage: 4,
            },
            buffer: upload,
        };
        yawgpu::wgpuCommandEncoderCopyBufferToTexture(encoder, &source, &texture_info, &extent);
        let target = native::WGPUTexelCopyBufferInfo {
            layout: native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: 256,
                rowsPerImage: 4,
            },
            buffer: readback,
        };
        yawgpu::wgpuCommandEncoderCopyTextureToBuffer(encoder, &texture_info, &target, &extent);
        let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
        yawgpu::wgpuCommandBufferRelease(command_buffer);
        yawgpu::wgpuCommandEncoderRelease(encoder);
        let bytes = read_buffer(instance, readback, 256 * 4);
        for row in 0..rows {
            assert_eq!(
                &bytes[row * 256..row * 256 + row_bytes],
                &data[row * wide as usize..row * wide as usize + row_bytes],
                "row {row}"
            );
        }
        assert!(
            errors.lock().expect("error lock").is_empty(),
            "unexpected errors: {:?}",
            errors.lock().expect("error lock")
        );
        yawgpu::wgpuBufferRelease(readback);
        yawgpu::wgpuTextureRelease(texture);
        yawgpu::wgpuBufferRelease(upload);
        yawgpu::wgpuQueueRelease(queue);
        release(instance, adapter, device);
    }
}

unsafe fn create_buffer(device: native::WGPUDevice, size: u64) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        usage: native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
        size,
        ..std::mem::zeroed()
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
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

unsafe fn metal_device(
    features: &[native::WGPUFeatureName],
) -> Option<(
    native::WGPUInstance,
    native::WGPUAdapter,
    native::WGPUDevice,
)> {
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
    for &feature in features {
        if yawgpu::wgpuAdapterHasFeature(adapter, feature) == 0 {
            eprintln!("SKIP: adapter lacks feature {feature}");
            yawgpu::wgpuAdapterRelease(adapter);
            yawgpu::wgpuInstanceRelease(instance);
            return None;
        }
    }

    let mut device: native::WGPUDevice = std::ptr::null();
    let device_descriptor = native::WGPUDeviceDescriptor {
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
    Some((instance, adapter, device))
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
