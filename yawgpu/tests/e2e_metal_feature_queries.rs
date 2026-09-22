//! Real-Metal e2e for Block 99 (`specs/blocks/99-feature-advertisement-queries.md`):
//! the features whose advertisement Dawn gates on a Metal device query are
//! advertised through the C ABI iff the same query, made directly against
//! `MTLCreateSystemDefaultDevice`, says so.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::os::raw::c_void;

use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLCommonCounterTimestamp, MTLCounter, MTLCounterSamplingPoint,
    MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDevice,
};
use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

/// Dawn's `IsGPUCounterSupported(timestamp)` rule, evaluated straight from
/// Metal as the oracle: the `timestamp` counter set lists the `timestamp`
/// counter, and the device samples at a stage boundary or at every command
/// boundary.
fn metal_device_supports_timestamp_query() -> bool {
    let device = MTLCreateSystemDefaultDevice().expect("system default Metal device");
    // SAFETY: reading immutable Metal.framework string constants.
    let (set_name, counter_name) = unsafe {
        (
            MTLCommonCounterSetTimestamp.to_string().to_lowercase(),
            MTLCommonCounterTimestamp.to_string().to_lowercase(),
        )
    };
    let has_counter = device.counterSets().is_some_and(|sets| {
        sets.iter()
            .find(|set| set.name().to_string().to_lowercase() == set_name)
            .is_some_and(|set| {
                set.counters()
                    .iter()
                    .any(|counter| counter.name().to_string().to_lowercase() == counter_name)
            })
    });
    let stage = device.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary);
    let command = device.supportsCounterSampling(MTLCounterSamplingPoint::AtDrawBoundary)
        && device.supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary)
        && device.supportsCounterSampling(MTLCounterSamplingPoint::AtBlitBoundary);
    has_counter && (stage || command)
}

fn metal_device_supports_float32_filterable() -> bool {
    let device = MTLCreateSystemDefaultDevice().expect("system default Metal device");
    device.supports32BitFloatFiltering()
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_adapter_advertises_timestamp_query_iff_device_supports_counter_sampling() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let advertised =
            yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_TimestampQuery) != 0;
        assert_eq!(
            advertised,
            metal_device_supports_timestamp_query(),
            "timestamp-query advertisement must follow Dawn's IsGPUCounterSupported rule"
        );
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_adapter_advertises_float32_filterable_iff_device_supports_32bit_float_filtering() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let advertised =
            yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_Float32Filterable) != 0;
        assert_eq!(
            advertised,
            metal_device_supports_float32_filterable(),
            "float32-filterable advertisement must follow MTLDevice.supports32BitFloatFiltering"
        );
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
