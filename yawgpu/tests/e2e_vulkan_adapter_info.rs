//! Real-Vulkan e2e for Block 135 (`specs/blocks/135-adapter-info-parity.md`):
//! `wgpuAdapterGetInfo` reports the GPU, as Dawn does.

#![cfg(feature = "vulkan")]

#[allow(dead_code)]
#[path = "common/indirect_dispatch.rs"]
mod indirect_dispatch;

use indirect_dispatch::{create_instance, request_adapter};
use yawgpu::native;
use yawgpu::YAWGPU_INSTANCE_BACKEND_VULKAN;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

unsafe fn text(view: native::WGPUStringView) -> String {
    if view.data.is_null() {
        return String::new();
    }
    String::from_utf8_lossy(std::slice::from_raw_parts(
        view.data.cast::<u8>(),
        view.length,
    ))
    .into_owned()
}

#[test]
#[ignore = "manual real-backend test"]
fn adapter_info_reports_the_gpu() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    unsafe {
        let instance = create_instance(YAWGPU_INSTANCE_BACKEND_VULKAN);
        let adapter = request_adapter(instance);
        let mut info: native::WGPUAdapterInfo = std::mem::zeroed();
        assert_eq!(
            yawgpu::wgpuAdapterGetInfo(adapter, &mut info),
            native::WGPUStatus_Success
        );
        let vendor = text(info.vendor);
        let architecture = text(info.architecture);
        let device = text(info.device);
        let description = text(info.description);
        eprintln!(
            "vendor={vendor:?} architecture={architecture:?} device={device:?} description={description:?} \
             vendorID={:#x} deviceID={:#x} adapterType={} backendType={}",
            info.vendorID, info.deviceID, info.adapterType, info.backendType
        );

        assert_eq!(info.backendType, native::WGPUBackendType_Vulkan);
        assert_ne!(info.vendorID, 0, "a real Vulkan device has a vendor id");
        assert!(!device.is_empty());
        assert!(!description.is_empty());
        assert_ne!(info.adapterType, native::WGPUAdapterType_Unknown);

        // The NVIDIA host: Dawn reports vendor "nvidia", a discrete adapter, and an
        // architecture name from gpu_info.json, and the driver name in the description.
        if device.to_ascii_uppercase().contains("NVIDIA") {
            assert_eq!(info.vendorID, 0x10de);
            assert_eq!(vendor, "nvidia");
            assert_eq!(info.adapterType, native::WGPUAdapterType_DiscreteGPU);
            // RTX 5060 Ti is 0x2d04: gpu_info.json mask 0xFFFFFF00 -> 0x2d00, Blackwell.
            if info.deviceID & 0xffff_ff00 == 0x2d00 {
                assert_eq!(architecture, "blackwell");
            }
            assert!(
                !architecture.is_empty(),
                "no architecture for {:#x}",
                info.deviceID
            );
            assert!(description.contains("NVIDIA"), "{description}");
        }

        yawgpu::wgpuAdapterInfoFreeMembers(info);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

/// Adapter types the Vulkan loader reports, read without the HAL.
fn loader_adapter_types() -> Vec<ash::vk::PhysicalDeviceType> {
    unsafe {
        let entry = ash::Entry::load().expect("Vulkan loader");
        let instance = entry
            .create_instance(&ash::vk::InstanceCreateInfo::default(), None)
            .expect("Vulkan instance");
        let types = instance
            .enumerate_physical_devices()
            .expect("physical devices")
            .into_iter()
            .map(|device| instance.get_physical_device_properties(device).device_type)
            .collect();
        instance.destroy_instance(None);
        types
    }
}

unsafe extern "C" fn selection_callback(
    status: native::WGPURequestAdapterStatus,
    adapter: native::WGPUAdapter,
    _message: native::WGPUStringView,
    userdata1: *mut std::os::raw::c_void,
    _userdata2: *mut std::os::raw::c_void,
) {
    *(userdata1 as *mut (native::WGPURequestAdapterStatus, native::WGPUAdapter)) =
        (status, adapter);
}

/// Requests an adapter and returns its type and device name, or `None` when unavailable.
unsafe fn request_adapter_type(
    instance: native::WGPUInstance,
    power_preference: native::WGPUPowerPreference,
    force_fallback: bool,
) -> Option<(native::WGPUAdapterType, String)> {
    let options = native::WGPURequestAdapterOptions {
        nextInChain: std::ptr::null_mut(),
        featureLevel: native::WGPUFeatureLevel_Core,
        powerPreference: power_preference,
        forceFallbackAdapter: u32::from(force_fallback),
        backendType: native::WGPUBackendType_Vulkan,
        compatibleSurface: std::ptr::null(),
    };
    let mut result: (native::WGPURequestAdapterStatus, native::WGPUAdapter) =
        (native::WGPURequestAdapterStatus_Error, std::ptr::null());
    let callback_info = native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(selection_callback),
        userdata1: (&mut result as *mut (native::WGPURequestAdapterStatus, native::WGPUAdapter))
            .cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuInstanceRequestAdapter(instance, &options, callback_info);
    yawgpu_test::wait(instance, future);
    let (status, adapter) = result;
    if adapter.is_null() {
        assert_eq!(status, native::WGPURequestAdapterStatus_Unavailable);
        return None;
    }
    assert_eq!(status, native::WGPURequestAdapterStatus_Success);
    let mut info: native::WGPUAdapterInfo = std::mem::zeroed();
    assert_eq!(
        yawgpu::wgpuAdapterGetInfo(adapter, &mut info),
        native::WGPUStatus_Success
    );
    let selected = (info.adapterType, text(info.device));
    yawgpu::wgpuAdapterInfoFreeMembers(info);
    yawgpu::wgpuAdapterRelease(adapter);
    Some(selected)
}

/// Block 166 R6/R7 on hardware: `powerPreference` ranks the enumerated adapters
/// and `forceFallbackAdapter` keeps only CPU adapters. The ranking is observable
/// only on a host with both a discrete and an integrated GPU.
#[test]
#[ignore = "manual real-backend test"]
fn request_adapter_ranks_by_power_preference() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    let types = loader_adapter_types();
    let has = |kind| types.contains(&kind);
    let discrete = has(ash::vk::PhysicalDeviceType::DISCRETE_GPU);
    let integrated = has(ash::vk::PhysicalDeviceType::INTEGRATED_GPU);
    let cpu = has(ash::vk::PhysicalDeviceType::CPU);
    unsafe {
        let instance = create_instance(YAWGPU_INSTANCE_BACKEND_VULKAN);
        let high =
            request_adapter_type(instance, native::WGPUPowerPreference_HighPerformance, false);
        let low = request_adapter_type(instance, native::WGPUPowerPreference_LowPower, false);
        let undefined =
            request_adapter_type(instance, native::WGPUPowerPreference_Undefined, false);
        let fallback =
            request_adapter_type(instance, native::WGPUPowerPreference_HighPerformance, true);
        eprintln!(
            "loader types={types:?}\n high={high:?}\n low={low:?}\n undefined={undefined:?}\n fallback={fallback:?}"
        );

        if discrete {
            assert_eq!(
                high.as_ref().map(|selected| selected.0),
                Some(native::WGPUAdapterType_DiscreteGPU)
            );
        }
        if integrated {
            assert_eq!(
                low.as_ref().map(|selected| selected.0),
                Some(native::WGPUAdapterType_IntegratedGPU)
            );
        }
        if discrete || integrated {
            // A CPU adapter ranks after every GPU under any preference.
            assert_ne!(
                undefined.as_ref().map(|selected| selected.0),
                Some(native::WGPUAdapterType_CPU)
            );
        }
        match fallback {
            Some((kind, _)) => assert_eq!(kind, native::WGPUAdapterType_CPU),
            None => assert!(!cpu, "a CPU adapter exists but the fallback request failed"),
        }
        yawgpu::wgpuInstanceRelease(instance);
    }
}
