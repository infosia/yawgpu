//! Shared C API workload for adapters advertising implicit device synchronization.

use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Barrier;
use std::time::{Duration, Instant};
use yawgpu::{native::*, *};

#[allow(dead_code)]
#[path = "indirect_dispatch.rs"]
mod indirect_dispatch;
use indirect_dispatch::{create_buffer, create_wgsl_module, request_adapter, string_view};

const ITERATIONS: u32 = 32;
const COMPUTE_SHADER: &str = "@compute @workgroup_size(1) fn main() {}";
const WRITE_ERRORS: [(u64, usize, &str); 4] = [
    (1, 4, "queue write offset must be 4-byte aligned"),
    (0, 1, "queue write size must be 4-byte aligned"),
    (4, 4, "queue write range exceeds buffer size"),
    (u64::MAX - 3, 4, "queue write range overflows"),
];

/// Exercises a single shared device selected by a YAWGPU_INSTANCE_BACKEND constant.
pub(super) fn run(backend: u32) {
    unsafe {
        let mut select = YaWGPUInstanceBackendSelect {
            chain: WGPUChainedStruct {
                sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
                next: ptr::null_mut(),
            },
            backend,
        };
        let instance = wgpuCreateInstance(&WGPUInstanceDescriptor {
            nextInChain: &mut select.chain,
            requiredFeatureCount: 1,
            requiredFeatures: &WGPUInstanceFeatureName_TimedWaitAny,
            ..std::mem::zeroed()
        });
        assert!(!instance.is_null());
        let adapter = request_adapter(instance);
        let feature = WGPUFeatureName_ImplicitDeviceSynchronization;
        assert_ne!(wgpuAdapterHasFeature(adapter, feature), 0);
        let errors = AtomicU32::new(0);
        let mut device: WGPUDevice = ptr::null();
        let future = wgpuAdapterRequestDevice(
            adapter,
            &WGPUDeviceDescriptor {
                requiredFeatureCount: 1,
                requiredFeatures: &feature,
                uncapturedErrorCallbackInfo: WGPUUncapturedErrorCallbackInfo {
                    callback: Some(uncaptured),
                    userdata1: ptr::from_ref(&errors).cast_mut().cast(),
                    ..std::mem::zeroed()
                },
                ..std::mem::zeroed()
            },
            WGPURequestDeviceCallbackInfo {
                mode: WGPUCallbackMode_WaitAnyOnly,
                callback: Some(device_ready),
                userdata1: ptr::from_mut(&mut device).cast(),
                ..std::mem::zeroed()
            },
        );
        wait(instance, future, true);
        assert!(!device.is_null());
        assert_ne!(wgpuDeviceHasFeature(device, feature), 0);
        let have_tint = has_tint(instance, device);
        let queue = wgpuDeviceGetQueue(device);
        // These C handles are owned until every scoped worker has joined.
        let handles = (instance as usize, device as usize, queue as usize);
        let start = Barrier::new(WRITE_ERRORS.len());
        std::thread::scope(|scope| {
            for (worker, &error) in WRITE_ERRORS.iter().enumerate() {
                let start = &start;
                scope.spawn(move || {
                    for iteration in 0..ITERATIONS {
                        wgpuDevicePushErrorScope(handles.1 as _, WGPUErrorFilter_Validation);
                        if iteration == 0 {
                            // All four scope stacks are live before the first error is reported.
                            start.wait();
                        }
                        let word = 1 + worker as u32 * ITERATIONS + iteration;
                        round(
                            handles.0 as _,
                            handles.1 as _,
                            handles.2 as _,
                            word,
                            error,
                            have_tint,
                        );
                    }
                });
            }
        });
        wgpuQueueRelease(queue);
        wgpuDeviceRelease(device);
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(instance);
        assert_eq!(
            errors.load(Ordering::Relaxed),
            0,
            "uncaptured device errors"
        );
    }
}

unsafe extern "C" fn device_ready(
    status: WGPURequestDeviceStatus,
    device: WGPUDevice,
    _: WGPUStringView,
    data: *mut c_void,
    _: *mut c_void,
) {
    if status == WGPURequestDeviceStatus_Success {
        *data.cast::<WGPUDevice>() = device;
    }
}

unsafe extern "C" fn uncaptured(
    _: *const WGPUDevice,
    _: WGPUErrorType,
    _: WGPUStringView,
    data: *mut c_void,
    _: *mut c_void,
) {
    (*data.cast::<AtomicU32>()).fetch_add(1, Ordering::Relaxed);
}

unsafe fn wait(instance: WGPUInstance, future: WGPUFuture, timed: bool) {
    let mut info = WGPUFutureWaitInfo {
        future,
        completed: 0,
    };
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let status = wgpuInstanceWaitAny(
            instance,
            1,
            &mut info,
            if timed { 2_000_000_000 } else { 0 },
        );
        if status == WGPUWaitStatus_Success {
            assert_eq!(info.completed, 1);
            return;
        }
        assert_eq!(status, WGPUWaitStatus_TimedOut);
        assert_eq!(info.completed, 0);
        assert!(Instant::now() < deadline, "future {} timed out", future.id);
        std::thread::yield_now();
    }
}

unsafe fn has_tint(instance: WGPUInstance, device: WGPUDevice) -> bool {
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    wgpuShaderModuleRelease(create_wgsl_module(device, COMPUTE_SHADER));
    let (kind, message) = pop_scope(instance, device);
    if kind == WGPUErrorType_NoError {
        return true;
    }
    assert_eq!(kind, WGPUErrorType_Validation);
    assert_eq!(
        message,
        "yawgpu-tint was built without Tint (YAWGPU_DAWN_DIR unset)"
    );
    false
}

unsafe fn round(
    instance: WGPUInstance,
    device: WGPUDevice,
    queue: WGPUQueue,
    word: u32,
    error: (u64, usize, &str),
    have_tint: bool,
) {
    let source = create_buffer(device, 4, WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
    let readback = create_buffer(device, 4, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    let data = word.to_ne_bytes();
    wgpuQueueWriteBuffer(queue, source, 0, data.as_ptr().cast(), data.len());
    let encoder = wgpuDeviceCreateCommandEncoder(device, ptr::null());
    if have_tint {
        encode_compute(device, encoder);
    }
    wgpuCommandEncoderCopyBufferToBuffer(encoder, source, 0, readback, 0, 4);
    let commands = wgpuCommandEncoderFinish(encoder, ptr::null());
    wgpuQueueSubmit(queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);

    let status = AtomicU32::new(WGPUMapAsyncStatus_Error);
    let future = wgpuBufferMapAsync(
        readback,
        WGPUMapMode_Read,
        0,
        4,
        WGPUBufferMapCallbackInfo {
            mode: WGPUCallbackMode_AllowProcessEvents,
            callback: Some(mapped),
            userdata1: ptr::from_ref(&status).cast_mut().cast(),
            ..std::mem::zeroed()
        },
    );
    if word.is_multiple_of(2) {
        wgpuInstanceProcessEvents(instance);
    }
    wait(instance, future, word.is_multiple_of(2));
    assert_eq!(status.load(Ordering::Acquire), WGPUMapAsyncStatus_Success);
    let range = wgpuBufferGetConstMappedRange(readback, 0, 4);
    assert!(!range.is_null());
    assert_eq!(std::slice::from_raw_parts(range.cast::<u8>(), 4), data);
    wgpuBufferUnmap(readback);

    wgpuQueueWriteBuffer(queue, source, error.0, data.as_ptr().cast(), error.1);
    let (kind, message) = pop_scope(instance, device);
    assert_eq!(kind, WGPUErrorType_Validation);
    assert_eq!(message, error.2);
    wgpuBufferRelease(readback);
    wgpuBufferRelease(source);
}

unsafe fn pop_scope(instance: WGPUInstance, device: WGPUDevice) -> (WGPUErrorType, String) {
    let mut captured: Option<(WGPUPopErrorScopeStatus, WGPUErrorType, String)> = None;
    let future = wgpuDevicePopErrorScope(
        device,
        WGPUPopErrorScopeCallbackInfo {
            mode: WGPUCallbackMode_WaitAnyOnly,
            callback: Some(popped),
            userdata1: ptr::from_mut(&mut captured).cast(),
            ..std::mem::zeroed()
        },
    );
    wait(instance, future, false);
    let (status, kind, message) = captured.expect("error scope callback");
    assert_eq!(status, WGPUPopErrorScopeStatus_Success);
    (kind, message)
}

unsafe fn encode_compute(device: WGPUDevice, encoder: WGPUCommandEncoder) {
    let layout = wgpuDeviceCreateBindGroupLayout(device, &std::mem::zeroed());
    let group = wgpuDeviceCreateBindGroup(
        device,
        &WGPUBindGroupDescriptor {
            layout,
            ..std::mem::zeroed()
        },
    );
    let pipeline_layout = wgpuDeviceCreatePipelineLayout(
        device,
        &WGPUPipelineLayoutDescriptor {
            bindGroupLayoutCount: 1,
            bindGroupLayouts: &layout,
            ..std::mem::zeroed()
        },
    );
    let module = create_wgsl_module(device, COMPUTE_SHADER);
    let pipeline = wgpuDeviceCreateComputePipeline(
        device,
        &WGPUComputePipelineDescriptor {
            layout: pipeline_layout,
            compute: WGPUComputeState {
                module,
                entryPoint: string_view("main"),
                ..std::mem::zeroed()
            },
            ..std::mem::zeroed()
        },
    );
    let pass = wgpuCommandEncoderBeginComputePass(encoder, ptr::null());
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, ptr::null());
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuComputePipelineRelease(pipeline);
    wgpuShaderModuleRelease(module);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuBindGroupRelease(group);
    wgpuBindGroupLayoutRelease(layout);
}

unsafe extern "C" fn mapped(
    status: WGPUMapAsyncStatus,
    _: WGPUStringView,
    data: *mut c_void,
    _: *mut c_void,
) {
    (*data.cast::<AtomicU32>()).store(status, Ordering::Release);
}

unsafe extern "C" fn popped(
    status: WGPUPopErrorScopeStatus,
    kind: WGPUErrorType,
    message: WGPUStringView,
    data: *mut c_void,
    _: *mut c_void,
) {
    let bytes = if message.length == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(message.data.cast(), message.length)
    };
    *data.cast::<Option<(WGPUPopErrorScopeStatus, WGPUErrorType, String)>>() =
        Some((status, kind, String::from_utf8_lossy(bytes).into_owned()));
}
