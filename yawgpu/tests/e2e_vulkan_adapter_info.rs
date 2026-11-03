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
