//! Real-Vulkan e2e for Block 99 (`specs/blocks/99-feature-advertisement-queries.md`):
//! the features whose advertisement Dawn gates on a physical-device query are
//! advertised through the C ABI iff the same query, made directly through
//! `ash` against the first physical device, says so.

#![cfg(feature = "vulkan")]

use std::os::raw::c_void;

use ash::vk;
use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_VULKAN,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

/// The five Dawn rules (`PhysicalDeviceVk.cpp`), evaluated straight from
/// `ash` as the oracle against the first enumerated physical device — the
/// same device yawgpu's Vulkan instance picks.
struct VulkanOracle {
    timestamp_query: bool,
    depth32float_stencil8: bool,
    rg11b10ufloat_renderable: bool,
    bgra8unorm_storage: bool,
    float32_filterable: bool,
}

fn vulkan_oracle() -> VulkanOracle {
    // SAFETY: plain loader / instance creation and read-only property queries;
    // the instance is destroyed before returning.
    unsafe {
        let entry = ash::Entry::load().expect("load the Vulkan loader");
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
        // MoltenVK is a portability ICD: without VK_KHR_portability_enumeration
        // the loader hides it and reports ERROR_INCOMPATIBLE_DRIVER, so mirror
        // what yawgpu's own instance does when the extension is available.
        let portability = entry
            .enumerate_instance_extension_properties(None)
            .expect("enumerate instance extensions")
            .iter()
            .any(|extension| {
                extension.extension_name_as_c_str().ok()
                    == Some(vk::KHR_PORTABILITY_ENUMERATION_NAME)
            });
        let extension_names = if portability {
            vec![vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr()]
        } else {
            Vec::new()
        };
        let flags = if portability {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .flags(flags);
        let instance = entry
            .create_instance(&create_info, None)
            .expect("create a Vulkan instance");
        let physical_device = instance
            .enumerate_physical_devices()
            .expect("enumerate physical devices")
            .into_iter()
            .next()
            .expect("at least one physical device");
        let optimal = |format: vk::Format| {
            instance
                .get_physical_device_format_properties(physical_device, format)
                .optimal_tiling_features
        };
        let limits = instance
            .get_physical_device_properties(physical_device)
            .limits;
        let linear = vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
        let oracle = VulkanOracle {
            timestamp_query: limits.timestamp_compute_and_graphics == vk::TRUE,
            depth32float_stencil8: optimal(vk::Format::D32_SFLOAT_S8_UINT)
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT),
            rg11b10ufloat_renderable: optimal(vk::Format::B10G11R11_UFLOAT_PACK32).contains(
                vk::FormatFeatureFlags::COLOR_ATTACHMENT
                    | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND,
            ),
            bgra8unorm_storage: optimal(vk::Format::B8G8R8A8_UNORM)
                .contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
            float32_filterable: optimal(vk::Format::R32_SFLOAT).contains(linear)
                && optimal(vk::Format::R32G32_SFLOAT).contains(linear)
                && optimal(vk::Format::R32G32B32A32_SFLOAT).contains(linear),
        };
        instance.destroy_instance(None);
        oracle
    }
}

#[test]
#[ignore = "manual real-backend test"]
fn vulkan_adapter_advertises_device_gated_features_iff_physical_device_reports_them() {
    if real_backend_skip_reason(RealBackend::Vulkan).is_some() {
        return;
    }
    let oracle = vulkan_oracle();
    unsafe {
        let instance = create_vulkan_instance();
        let adapter = request_adapter(instance);
        let has =
            |feature: native::WGPUFeatureName| yawgpu::wgpuAdapterHasFeature(adapter, feature) != 0;
        assert_eq!(
            has(native::WGPUFeatureName_TimestampQuery),
            oracle.timestamp_query,
            "timestamp-query must follow timestampComputeAndGraphics"
        );
        assert_eq!(
            has(native::WGPUFeatureName_Depth32FloatStencil8),
            oracle.depth32float_stencil8,
            "depth32float-stencil8 must follow D32_SFLOAT_S8_UINT attachment support"
        );
        assert_eq!(
            has(native::WGPUFeatureName_RG11B10UfloatRenderable),
            oracle.rg11b10ufloat_renderable,
            "rg11b10ufloat-renderable must follow B10G11R11 attachment+blend support"
        );
        assert_eq!(
            has(native::WGPUFeatureName_BGRA8UnormStorage),
            oracle.bgra8unorm_storage,
            "bgra8unorm-storage must follow B8G8R8A8_UNORM storage-image support"
        );
        assert_eq!(
            has(native::WGPUFeatureName_Float32Filterable),
            oracle.float32_filterable,
            "float32-filterable must follow linear-filter support on R32/RG32/RGBA32"
        );
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
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
