//! Scratch repro for the HANDOFF "yawgpu retains released GPU resources" leak.
//!
//! Drives a create -> use -> release loop on a real Metal device and lets an
//! external `/usr/bin/time -l` observe peak RSS. Attribution is done by running
//! the same test with different `YAWGPU_LEAK_MODE` / `YAWGPU_LEAK_ITERS` and
//! comparing peak RSS deltas across iteration counts.
//!
//! Run:
//!   YAWGPU_LEAK_MODE=cmdsubmit YAWGPU_LEAK_ITERS=20000 \
//!   /usr/bin/time -l cargo test -p yawgpu --features metal \
//!     --test e2e_metal_leak metal_leak_loop -- --ignored --exact --nocapture
//!
//! Modes: buffer | texture | cmdsubmit | map | all
//!
//! NOT a regression test — manual diagnostic only.

#![cfg(any(feature = "metal", feature = "vulkan"))]

use std::os::raw::c_void;

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

#[test]
#[ignore = "manual real-backend leak repro"]
fn metal_leak_loop() {
    let backend = std::env::var("YAWGPU_LEAK_BACKEND").unwrap_or_else(|_| "metal".to_owned());
    let (backend_id, real_backend) = match backend.as_str() {
        "vulkan" => (YAWGPU_INSTANCE_BACKEND_VULKAN, RealBackend::Vulkan),
        "metal" => (YAWGPU_INSTANCE_BACKEND_METAL, RealBackend::Metal),
        other => panic!("unknown YAWGPU_LEAK_BACKEND: {other}"),
    };
    if real_backend_skip_reason(real_backend).is_some() {
        eprintln!("metal_leak_loop: skipped (no real {backend} device)");
        return;
    }

    let mode = std::env::var("YAWGPU_LEAK_MODE").unwrap_or_else(|_| "all".to_owned());
    let iters: u64 = std::env::var("YAWGPU_LEAK_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    eprintln!("metal_leak_loop: mode={mode} iters={iters}");

    unsafe {
        let instance = create_instance(backend_id);
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);

        for i in 0..iters {
            match mode.as_str() {
                "buffer" => iter_buffer(device),
                "texture" => iter_texture(device),
                "cmdsubmit" => iter_cmdsubmit(device),
                "map" => iter_map(instance, device),
                "write" => iter_write(device),
                "maponly" => iter_maponly(instance, device),
                "mapfail" => iter_mapfail(instance, device),
                "all" => {
                    iter_buffer(device);
                    iter_texture(device);
                    iter_cmdsubmit(device);
                    iter_map(instance, device);
                }
                other => panic!("unknown YAWGPU_LEAK_MODE: {other}"),
            }
            if i % 5000 == 0 {
                eprintln!("  rss after {i}: {} KB", current_rss_kb());
            }
        }
        eprintln!("metal_leak_loop: final rss {} KB", current_rss_kb());

        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Create + release a small buffer (no submit, no map).
unsafe fn iter_buffer(device: native::WGPUDevice) {
    let buffer = create_buffer(
        device,
        256,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
    );
    yawgpu::wgpuBufferRelease(buffer);
}

/// Create + release a small 8x8 rgba8unorm texture (no submit).
unsafe fn iter_texture(device: native::WGPUDevice) {
    let texture = create_texture(device);
    yawgpu::wgpuTextureRelease(texture);
}

/// Create encoder, record a clearBuffer, finish, submit, release everything.
/// Exercises the CommandBuffer referenced-resource / command_ops retention.
unsafe fn iter_cmdsubmit(device: native::WGPUDevice) {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let buffer = create_buffer(
        device,
        256,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
    );
    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
    yawgpu::wgpuCommandEncoderClearBuffer(encoder, buffer, 0, 256);
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuCommandEncoderRelease(encoder);
    yawgpu::wgpuBufferRelease(buffer);
    yawgpu::wgpuQueueRelease(queue);
}

/// Full write -> map -> readback -> unmap -> release. Exercises the future
/// registry / pending-callback path on every iteration.
unsafe fn iter_map(instance: native::WGPUInstance, device: native::WGPUDevice) {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let buffer = create_buffer(
        device,
        256,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let data = [7u8; 256];
    yawgpu::wgpuQueueWriteBuffer(queue, buffer, 0, data.as_ptr().cast(), data.len());

    let mut status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, 256, callback_info);
    wait(instance, future);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, 256);
    assert!(!ptr.is_null());
    yawgpu::wgpuBufferUnmap(buffer);
    yawgpu::wgpuBufferRelease(buffer);
    yawgpu::wgpuQueueRelease(queue);
}

/// Only `wgpuQueueWriteBuffer` (staging + submit), no map.
unsafe fn iter_write(device: native::WGPUDevice) {
    let queue = yawgpu::wgpuDeviceGetQueue(device);
    let buffer = create_buffer(
        device,
        256,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
    );
    let data = [7u8; 256];
    yawgpu::wgpuQueueWriteBuffer(queue, buffer, 0, data.as_ptr().cast(), data.len());
    yawgpu::wgpuBufferRelease(buffer);
    yawgpu::wgpuQueueRelease(queue);
}

/// Only mapAsync + wait + getMappedRange + unmap (no writeBuffer).
unsafe fn iter_maponly(instance: native::WGPUInstance, device: native::WGPUDevice) {
    let size: u64 = std::env::var("YAWGPU_LEAK_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let buffer = create_buffer(
        device,
        size,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let mut status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, 256, callback_info);
    wait(instance, future);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, 256);
    assert!(!ptr.is_null());
    yawgpu::wgpuBufferUnmap(buffer);
    yawgpu::wgpuBufferRelease(buffer);
}

/// mapAsync on a buffer WITHOUT MapRead usage: validation fails, so the
/// pending callback retains no buffer Arc, but a future is still registered.
/// Isolates the FutureRegistry registration machinery from the data path.
unsafe fn iter_mapfail(instance: native::WGPUInstance, device: native::WGPUDevice) {
    let buffer = create_buffer(
        device,
        256,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_CopySrc,
    );
    let mut status = native::WGPUMapAsyncStatus_Success;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, 256, callback_info);
    wait(instance, future);
    yawgpu::wgpuBufferRelease(buffer);
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

unsafe fn create_texture(device: native::WGPUDevice) -> native::WGPUTexture {
    let descriptor = native::WGPUTextureDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage: native::WGPUTextureUsage_TextureBinding | native::WGPUTextureUsage_CopyDst,
        dimension: native::WGPUTextureDimension_2D,
        size: native::WGPUExtent3D {
            width: 8,
            height: 8,
            depthOrArrayLayers: 1,
        },
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

unsafe fn create_instance(backend_id: u32) -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: backend_id,
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

/// Current resident set size in KB via `mach_task_self` / `task_info`.
fn current_rss_kb() -> u64 {
    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [i32; 2],
        system_time: [i32; 2],
        policy: i32,
        suspend_count: i32,
    }
    // MACH_TASK_BASIC_INFO_COUNT = sizeof(info)/sizeof(natural_t=u32)
    const COUNT: u32 = (std::mem::size_of::<MachTaskBasicInfo>() / 4) as u32;
    const MACH_TASK_BASIC_INFO: i32 = 20;
    extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(task: u32, flavor: i32, info: *mut i32, count: *mut u32) -> i32;
    }
    unsafe {
        let mut info = std::mem::zeroed::<MachTaskBasicInfo>();
        let mut count = COUNT;
        let kr = task_info(
            mach_task_self(),
            MACH_TASK_BASIC_INFO,
            (&mut info as *mut MachTaskBasicInfo).cast(),
            &mut count,
        );
        if kr != 0 {
            return 0;
        }
        info.resident_size / 1024
    }
}
