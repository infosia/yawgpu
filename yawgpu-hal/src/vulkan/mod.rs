use std::collections::VecDeque;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ash::vk;

use crate::{
    HalAddressMode, HalBlendFactor, HalBlendOperation, HalBoundBuffer, HalBoundIndexBuffer,
    HalBoundSampler, HalBoundTexture, HalBuffer, HalBufferBindingKind, HalBufferClear,
    HalBufferCopy, HalBufferTextureCopy, HalBufferUsage, HalColorClearKind, HalColorTargetState,
    HalCompareFunction, HalComputeDispatch, HalComputePass, HalCopy, HalCullMode,
    HalDepthStencilState, HalDescriptorBinding, HalDescriptorBindingKind, HalError, HalExtent3d,
    HalFilterMode, HalFrontFace, HalIndexFormat, HalLimits, HalMipmapFilterMode,
    HalPrimitiveTopology, HalQueryKind, HalQuerySet, HalRenderLoadOp, HalRenderPassCommand,
    HalRenderPassCommandStream, HalRenderPipelineDescriptor, HalResolveQuerySet, HalSampler,
    HalSamplerDescriptor, HalShaderSource, HalStencilOperation, HalSurfaceConfiguration,
    HalTexture, HalTextureCopy, HalTextureDescriptor, HalTextureFormat, HalTextureUsage,
    HalVertexFormat, HalVertexStepMode, HalWriteTimestamp, SubmissionIndex,
};
#[cfg(feature = "tiled")]
use crate::{HalDraw, HalSubpassPassLayout};

const BACKEND: &str = "vulkan";
const ASSUMED_MAX_BUFFER_SIZE: u64 = 2 * 1024 * 1024 * 1024;
/// Minimum Vulkan API version yawgpu requests at vkCreateInstance.
/// Documented in specs/blocks/60-real-backends.md § Minimum Vulkan version.
const YAWGPU_VULKAN_API_VERSION: u32 = vk::API_VERSION_1_1;

static VULKAN_ENTRY: OnceLock<ash::Entry> = OnceLock::new();
static VULKAN_ENTRY_INIT: Mutex<()> = Mutex::new(());

fn queue_submission_error(call: &'static str, error: vk::Result) -> HalError {
    HalError::QueueSubmissionFailed {
        backend: BACKEND,
        message: format!("{call} failed: {error:?}"),
    }
}

fn shared_entry() -> Result<&'static ash::Entry, HalError> {
    if let Some(entry) = VULKAN_ENTRY.get() {
        return Ok(entry);
    }

    let _guard = VULKAN_ENTRY_INIT
        .lock()
        .map_err(|_| HalError::BackendUnavailable { backend: BACKEND })?;
    if let Some(entry) = VULKAN_ENTRY.get() {
        return Ok(entry);
    }

    let entry = unsafe { ash::Entry::load() }
        .map_err(|_| HalError::BackendUnavailable { backend: BACKEND })?;
    if VULKAN_ENTRY.set(entry).is_err() {
        return VULKAN_ENTRY
            .get()
            .ok_or(HalError::BackendUnavailable { backend: BACKEND });
    }
    VULKAN_ENTRY
        .get()
        .ok_or(HalError::BackendUnavailable { backend: BACKEND })
}

/// Stores vulkan instance data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct VulkanInstance {
    inner: Arc<VulkanInstanceInner>,
}

impl VulkanInstance {
    /// Creates a new instance.
    pub fn new() -> Result<Self, HalError> {
        let entry = shared_entry()?;
        let available_extensions =
            unsafe { entry.enumerate_instance_extension_properties(None) }
                .map_err(|_| HalError::BackendUnavailable { backend: BACKEND })?;
        let available_extension_names = available_extensions
            .iter()
            .filter_map(|extension| extension.extension_name_as_c_str().ok())
            .collect::<Vec<_>>();
        let Some((extension_names, flags)) = instance_extension_config(&available_extension_names)
        else {
            return Err(HalError::BackendUnavailable { backend: BACKEND });
        };
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"yawgpu")
            .engine_name(c"yawgpu")
            .api_version(YAWGPU_VULKAN_API_VERSION);
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .flags(flags)
            .enabled_extension_names(&extension_names);
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .map_err(|_| HalError::DeviceCreationFailed { backend: BACKEND })?;
        Ok(Self {
            inner: Arc::new(VulkanInstanceInner {
                _entry: entry,
                instance,
            }),
        })
    }

    /// Returns adapters exposed by this instance.
    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<VulkanAdapter> {
        let physical_devices = unsafe { self.inner.instance.enumerate_physical_devices() };
        let Ok(physical_devices) = physical_devices else {
            return Vec::new();
        };
        physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                VulkanAdapter::new(Arc::clone(&self.inner), physical_device)
            })
            .collect()
    }

    /// # Safety
    ///
    /// `layer` must be a valid, non-dangling `CAMetalLayer` instance pointer.
    pub unsafe fn create_surface_from_metal_layer(
        &self,
        layer: *mut c_void,
    ) -> Result<VulkanSurface, HalError> {
        if layer.is_null() {
            return Err(HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "surface layer is null",
            });
        }
        let loader =
            ash::ext::metal_surface::Instance::new(self.inner._entry, &self.inner.instance);
        let create_info = vk::MetalSurfaceCreateInfoEXT::default().layer(layer);
        let surface = unsafe { loader.create_metal_surface(&create_info, None) }.map_err(|_| {
            HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "vkCreateMetalSurfaceEXT failed",
            }
        })?;
        let surface_inner = Arc::new(VulkanSurfaceInner::new(Arc::clone(&self.inner), surface));
        Ok(VulkanSurface {
            suboptimal: false,
            present_error: None,
            surface,
            surface_inner,
            swapchain: None,
            config: None,
            current_image_index: None,
            pending_state: Arc::new(Mutex::new(SurfacePendingState::new())),
            image_acquired_semaphores: Vec::new(),
            render_finished_semaphores: Vec::new(),
            present_ready_semaphores: Vec::new(),
            in_flight_fences: Vec::new(),
            next_sync_index: 0,
        })
    }

    /// # Safety
    ///
    /// `hwnd` must be a valid Win32 window handle and `hinstance` the module
    /// instance that registered its window class; both must outlive the surface.
    pub unsafe fn create_surface_from_windows_hwnd(
        &self,
        hinstance: *mut c_void,
        hwnd: *mut c_void,
    ) -> Result<VulkanSurface, HalError> {
        if hwnd.is_null() {
            return Err(HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "surface hwnd is null",
            });
        }
        let loader =
            ash::khr::win32_surface::Instance::new(self.inner._entry, &self.inner.instance);
        let create_info = vk::Win32SurfaceCreateInfoKHR::default()
            .hinstance(hinstance as _)
            .hwnd(hwnd as _);
        let surface = unsafe { loader.create_win32_surface(&create_info, None) }.map_err(|_| {
            HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "vkCreateWin32SurfaceKHR failed",
            }
        })?;
        let surface_inner = Arc::new(VulkanSurfaceInner::new(Arc::clone(&self.inner), surface));
        Ok(VulkanSurface {
            suboptimal: false,
            present_error: None,
            surface,
            surface_inner,
            swapchain: None,
            config: None,
            current_image_index: None,
            pending_state: Arc::new(Mutex::new(SurfacePendingState::new())),
            image_acquired_semaphores: Vec::new(),
            render_finished_semaphores: Vec::new(),
            present_ready_semaphores: Vec::new(),
            in_flight_fences: Vec::new(),
            next_sync_index: 0,
        })
    }
}

fn instance_extension_config(
    available_extensions: &[&CStr],
) -> Option<(Vec<*const c_char>, vk::InstanceCreateFlags)> {
    if !has_instance_extension(available_extensions, vk::KHR_SURFACE_NAME) {
        return None;
    }

    let mut extension_names = vec![vk::KHR_SURFACE_NAME.as_ptr()];
    if has_instance_extension(available_extensions, vk::EXT_METAL_SURFACE_NAME) {
        extension_names.push(vk::EXT_METAL_SURFACE_NAME.as_ptr());
    }
    if has_instance_extension(available_extensions, vk::KHR_WIN32_SURFACE_NAME) {
        extension_names.push(vk::KHR_WIN32_SURFACE_NAME.as_ptr());
    }

    let portability_enumeration =
        has_instance_extension(available_extensions, vk::KHR_PORTABILITY_ENUMERATION_NAME);
    if portability_enumeration {
        extension_names.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
    }
    let flags = if portability_enumeration {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::default()
    };

    Some((extension_names, flags))
}

fn has_instance_extension(available_extensions: &[&CStr], name: &CStr) -> bool {
    available_extensions.contains(&name)
}

fn is_supported_api_version(api_version: u32) -> bool {
    let major = vk::api_version_major(api_version);
    let minor = vk::api_version_minor(api_version);
    (major, minor) >= (1, 1)
}

struct VulkanInstanceInner {
    _entry: &'static ash::Entry,
    instance: ash::Instance,
}

impl fmt::Debug for VulkanInstanceInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanInstanceInner")
            .finish_non_exhaustive()
    }
}

impl Drop for VulkanInstanceInner {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

/// Stores vulkan adapter data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct VulkanAdapter {
    instance: Arc<VulkanInstanceInner>,
    physical_device: vk::PhysicalDevice,
    name: String,
    astc_sliced_3d_support: Arc<OnceLock<bool>>,
    subgroup_query: Arc<OnceLock<SubgroupQuery>>,
}

fn is_fallback_device_type(device_type: vk::PhysicalDeviceType) -> bool {
    device_type == vk::PhysicalDeviceType::CPU
}

impl VulkanAdapter {
    pub(crate) fn info(&self) -> crate::HalAdapterInfo {
        let properties = unsafe {
            self.instance
                .instance
                .get_physical_device_properties(self.physical_device)
        };
        let version = crate::adapter_info::vulkan_driver_version(
            properties.vendor_id,
            properties.driver_version,
            cfg!(target_os = "windows"),
        );
        let driver = query_driver_properties(&self.instance, self.physical_device);
        let strings = driver.as_ref().map(|driver| {
            let name = driver
                .driver_name_as_c_str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let info = driver
                .driver_info_as_c_str()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            (name, info)
        });
        crate::HalAdapterInfo::new(
            properties.vendor_id,
            properties.device_id,
            adapter_type(properties.device_type),
            crate::adapter_info::vulkan_driver_description(
                &version,
                strings
                    .as_ref()
                    .map(|(name, info)| (name.as_str(), info.as_str())),
            ),
        )
    }

    pub(crate) fn is_fallback(&self) -> bool {
        let properties = unsafe {
            self.instance
                .instance
                .get_physical_device_properties(self.physical_device)
        };
        is_fallback_device_type(properties.device_type)
    }

    fn new(
        instance: Arc<VulkanInstanceInner>,
        physical_device: vk::PhysicalDevice,
    ) -> Option<Self> {
        let properties = unsafe {
            instance
                .instance
                .get_physical_device_properties(physical_device)
        };
        if !is_supported_api_version(properties.api_version) {
            return None;
        }
        let name = physical_device_name(properties)?;
        Some(Self {
            instance,
            physical_device,
            name,
            astc_sliced_3d_support: Arc::new(OnceLock::new()),
            subgroup_query: Arc::new(OnceLock::new()),
        })
    }

    /// Returns the name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the backend-reported supported limits.
    #[must_use]
    pub(crate) fn limits(&self) -> HalLimits {
        let properties = unsafe {
            self.instance
                .instance
                .get_physical_device_properties(self.physical_device)
        };
        let vk = properties.limits;
        let max_buffer_size = self.max_buffer_size();

        // Block 94 S3: the combined immediates push-constant block is at
        // most 64 user bytes + the 8-byte pixel-center depth-range pair =
        // 72 bytes. Every conformant Vulkan implementation satisfies this
        // by a wide margin -- the spec-required minimum for
        // `maxPushConstantsSize` is 128 (Vulkan 1.3 spec, "Limit Requirements"
        // table 53) -- so this is a debug-only tripwire against a broken
        // driver report, not a runtime capability gate.
        debug_assert!(
            vk.max_push_constants_size >= 72,
            "Vulkan maxPushConstantsSize {} < 72 (64 user immediates + 8 depth-range); \
             the spec minimum is 128",
            vk.max_push_constants_size
        );

        let mut limits = hal_limits_from_vk(vk, max_buffer_size, properties.vendor_id);
        let subgroup_query = self.subgroup_query();
        limits.max_compute_invocations_per_workgroup = clamp_compute_invocations_per_workgroup(
            limits.max_compute_invocations_per_workgroup,
            subgroup_query.max_compute_workgroup_subgroups,
            subgroup_query.default_compute_subgroup_size(),
        );
        limits
    }

    fn max_buffer_size(&self) -> u64 {
        let mut maintenance4 = vk::PhysicalDeviceMaintenance4Properties::default();
        let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut maintenance4)
            .push_next(&mut maintenance3);
        unsafe {
            self.instance
                .instance
                .get_physical_device_properties2(self.physical_device, &mut properties2);
        }
        select_max_buffer_size(
            maintenance4.max_buffer_size,
            maintenance3.max_memory_allocation_size,
        )
    }

    /// Returns true when BC texture compression is supported by this physical device.
    #[must_use]
    pub fn supports_texture_compression_bc(&self) -> bool {
        unsafe {
            self.instance
                .instance
                .get_physical_device_features(self.physical_device)
                .texture_compression_bc
                == vk::TRUE
        }
    }

    /// Returns true when 3D (sliced) BC textures are supported; on Vulkan this equals base BC support (Dawn parity).
    #[must_use]
    pub fn supports_texture_compression_bc_sliced_3d(&self) -> bool {
        self.supports_texture_compression_bc()
    }

    /// Returns true when ETC2/EAC texture compression is supported by this physical device.
    #[must_use]
    pub fn supports_texture_compression_etc2(&self) -> bool {
        unsafe {
            self.instance
                .instance
                .get_physical_device_features(self.physical_device)
                .texture_compression_etc2
                == vk::TRUE
        }
    }

    /// Returns true when ASTC LDR texture compression is supported by this physical device.
    #[must_use]
    pub fn supports_texture_compression_astc(&self) -> bool {
        unsafe {
            self.instance
                .instance
                .get_physical_device_features(self.physical_device)
                .texture_compression_astc_ldr
                == vk::TRUE
        }
    }

    /// Returns true when ASTC LDR is supported and every mapped ASTC LDR format
    /// supports sampled, optimally tiled 3D images with no additional image flags.
    #[must_use]
    pub fn supports_texture_compression_astc_sliced_3d(&self) -> bool {
        *self.astc_sliced_3d_support.get_or_init(|| {
            astc_sliced_3d_supported(self.supports_texture_compression_astc(), |format| unsafe {
                self.instance
                    .instance
                    .get_physical_device_image_format_properties(
                        self.physical_device,
                        format,
                        vk::ImageType::TYPE_3D,
                        vk::ImageTiling::OPTIMAL,
                        vk::ImageUsageFlags::SAMPLED,
                        vk::ImageCreateFlags::empty(),
                    )
                    .is_ok()
            })
        })
    }

    /// Returns true when texture view component swizzling is supported by this
    /// physical device. Dawn enables this unconditionally on Vulkan, so a
    /// non-portability driver always reports `true`. A portability
    /// implementation (`VK_KHR_portability_subset`, e.g. MoltenVK) may not
    /// support a non-identity `VkComponentMapping` at all, so there the
    /// advertisement is gated on its `imageViewFormatSwizzle` feature
    /// (`VUID-VkImageViewCreateInfo-imageViewFormatSwizzle-04465`).
    #[must_use]
    pub fn supports_texture_component_swizzle(&self) -> bool {
        let portability_subset_present = self.has_device_extension(vk::KHR_PORTABILITY_SUBSET_NAME);
        if !portability_subset_present {
            return true;
        }
        let mut portability_features = vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default();
        let mut features2 =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut portability_features);
        unsafe {
            self.instance
                .instance
                .get_physical_device_features2(self.physical_device, &mut features2);
        }
        texture_component_swizzle_supported(
            portability_subset_present,
            portability_features.image_view_format_swizzle,
        )
    }

    /// Returns the optimal-tiling format feature flags the physical device
    /// reports for `format` (`vkGetPhysicalDeviceFormatProperties`).
    fn optimal_tiling_features(&self, format: vk::Format) -> vk::FormatFeatureFlags {
        unsafe {
            self.instance
                .instance
                .get_physical_device_format_properties(self.physical_device, format)
        }
        .optimal_tiling_features
    }

    fn supports_texture_formats_tiers(&self) -> bool {
        let features = unsafe {
            self.instance
                .instance
                .get_physical_device_features(self.physical_device)
        };
        if features.shader_storage_image_extended_formats != vk::TRUE {
            return false;
        }

        let required_features = vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND;
        [
            vk::Format::R16_UNORM,
            vk::Format::R16_SNORM,
            vk::Format::R16G16_UNORM,
            vk::Format::R16G16_SNORM,
            vk::Format::R16G16B16A16_UNORM,
            vk::Format::R16G16B16A16_SNORM,
            vk::Format::R8_SNORM,
            vk::Format::R8G8_SNORM,
            vk::Format::R8G8B8A8_SNORM,
            vk::Format::B10G11R11_UFLOAT_PACK32,
        ]
        .into_iter()
        .all(|format| {
            let props = unsafe {
                self.instance
                    .instance
                    .get_physical_device_format_properties(self.physical_device, format)
            };
            props.optimal_tiling_features.contains(required_features)
        })
    }

    /// Returns true when Vulkan supports extended storage-image formats and all
    /// formats Dawn requires are color-attachment renderable and blendable. Vulkan
    /// advertises texture format tiers 1 and 2 together under this rule.
    #[must_use]
    pub(super) fn supports_texture_formats_tier1(&self) -> bool {
        self.supports_texture_formats_tiers()
    }

    /// Returns true when Vulkan supports extended storage-image formats and all
    /// formats Dawn requires are color-attachment renderable and blendable. Vulkan
    /// advertises texture format tiers 1 and 2 together under this rule.
    #[must_use]
    pub(super) fn supports_texture_formats_tier2(&self) -> bool {
        self.supports_texture_formats_tiers()
    }

    /// Returns true when `Rg11b10Ufloat` is renderable by this physical device:
    /// `B10G11R11_UFLOAT_PACK32` optimal-tiling features contain
    /// `COLOR_ATTACHMENT | COLOR_ATTACHMENT_BLEND` (Block 99 R4, Dawn
    /// `PhysicalDeviceVk.cpp` `InitializeSupportedFeaturesImpl`).
    #[must_use]
    pub(super) fn supports_rg11b10ufloat_renderable(&self) -> bool {
        rg11b10ufloat_renderable_from_flags(
            self.optimal_tiling_features(vk::Format::B10G11R11_UFLOAT_PACK32),
        )
    }

    /// Returns true when BGRA8 unorm storage textures are supported by this
    /// physical device: `B8G8R8A8_UNORM` optimal-tiling features contain
    /// `STORAGE_IMAGE` (Block 99 R4, Dawn `InitializeSupportedFeaturesImpl`).
    #[must_use]
    pub(super) fn supports_bgra8unorm_storage(&self) -> bool {
        bgra8unorm_storage_from_flags(self.optimal_tiling_features(vk::Format::B8G8R8A8_UNORM))
    }

    /// Returns true when 32-bit float textures are filterable by this physical
    /// device: `R32_SFLOAT`, `R32G32_SFLOAT` and `R32G32B32A32_SFLOAT`
    /// optimal-tiling features all contain `SAMPLED_IMAGE_FILTER_LINEAR`
    /// (Block 99 R4, Dawn `InitializeSupportedFeaturesImpl`).
    #[must_use]
    pub(super) fn supports_float32_filterable(&self) -> bool {
        float32_filterable_from_flags(
            self.optimal_tiling_features(vk::Format::R32_SFLOAT),
            self.optimal_tiling_features(vk::Format::R32G32_SFLOAT),
            self.optimal_tiling_features(vk::Format::R32G32B32A32_SFLOAT),
        )
    }

    /// Returns nanoseconds per timestamp tick: `limits.timestampPeriod`
    /// (Block 102 R3, Dawn `DeviceVk.cpp` `GetTimestampPeriodInNS`).
    #[must_use]
    pub(crate) fn timestamp_period(&self) -> f32 {
        let properties = unsafe {
            self.instance
                .instance
                .get_physical_device_properties(self.physical_device)
        };
        timestamp_period_from_limits(properties.limits.timestamp_period)
    }

    /// Returns true when timestamp queries are supported by this physical
    /// device: `limits.timestampComputeAndGraphics == VK_TRUE` (Block 99 R3,
    /// Dawn `InitializeSupportedFeaturesImpl`).
    #[must_use]
    pub(super) fn supports_timestamp_query(&self) -> bool {
        let properties = unsafe {
            self.instance
                .instance
                .get_physical_device_properties(self.physical_device)
        };
        timestamp_query_from_limits(properties.limits.timestamp_compute_and_graphics)
    }

    /// Returns true when Depth32FloatStencil8 textures are supported by this
    /// physical device: `D32_SFLOAT_S8_UINT` optimal-tiling features contain
    /// `DEPTH_STENCIL_ATTACHMENT` (Block 99 R4, Dawn
    /// `IsDepthStencilFormatSupported`).
    #[must_use]
    pub(super) fn supports_depth32float_stencil8(&self) -> bool {
        depth32float_stencil8_from_flags(
            self.optimal_tiling_features(vk::Format::D32_SFLOAT_S8_UINT),
        )
    }

    /// Returns true when WGSL `shader-f16` is supported by this physical device.
    #[must_use]
    pub(super) fn supports_shader_float16(&self) -> bool {
        let extension_present = self.has_device_extension(vk::KHR_SHADER_FLOAT16_INT8_NAME);
        if !extension_present {
            return false;
        }

        let mut features = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let properties = unsafe {
            self.instance
                .instance
                .get_physical_device_properties(self.physical_device)
        };
        let storage_available = storage_16bit_available(
            YAWGPU_VULKAN_API_VERSION.min(properties.api_version),
            self.has_device_extension(vk::KHR_16BIT_STORAGE_NAME),
        );
        let mut storage = vk::PhysicalDevice16BitStorageFeatures::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features);
        if storage_available {
            features2 = features2.push_next(&mut storage);
        }
        unsafe {
            self.instance
                .instance
                .get_physical_device_features2(self.physical_device, &mut features2);
        }
        let shader_int16 = features2.features.shader_int16;
        shader_float16_supported(
            extension_present,
            features.shader_float16,
            shader_int16,
            storage.storage_buffer16_bit_access,
            storage.uniform_and_storage_buffer16_bit_access,
        )
    }

    /// Returns true when WGSL `subgroups` is supported by this physical device.
    #[must_use]
    pub(super) fn supports_subgroups(&self) -> bool {
        self.subgroup_size_range().is_some()
    }

    /// Returns true when depth clip control is supported by this physical device.
    #[must_use]
    pub(super) fn supports_depth_clip_control(&self) -> bool {
        // WebGPU unclippedDepth maps to core Vulkan depthClampEnable, which
        // implicitly disables depth clipping; VK_EXT_depth_clip_enable is only
        // an optional enhancement for independent clip control.
        (unsafe {
            self.instance
                .instance
                .get_physical_device_features(self.physical_device)
                .depth_clamp
        }) == vk::TRUE
    }

    /// Returns true when dual-source blending is supported by this physical device.
    #[must_use]
    pub(super) fn supports_dual_source_blending(&self) -> bool {
        (unsafe {
            self.instance
                .instance
                .get_physical_device_features(self.physical_device)
                .dual_src_blend
        }) == vk::TRUE
    }

    /// Returns true when WGSL clip distances are supported by this physical device.
    #[must_use]
    pub(super) fn supports_clip_distances(&self) -> bool {
        (unsafe {
            self.instance
                .instance
                .get_physical_device_features(self.physical_device)
                .shader_clip_distance
        }) == vk::TRUE
    }

    /// Returns true when WGSL primitive index is supported by this physical device.
    #[must_use]
    pub(super) fn supports_primitive_index(&self) -> bool {
        (unsafe {
            self.instance
                .instance
                .get_physical_device_features(self.physical_device)
                .geometry_shader
        }) == vk::TRUE
    }

    /// Returns true when indirect draws support non-zero first instance values.
    #[must_use]
    pub(super) fn supports_indirect_first_instance(&self) -> bool {
        (unsafe {
            self.instance
                .instance
                .get_physical_device_features(self.physical_device)
                .draw_indirect_first_instance
        }) == vk::TRUE
    }

    /// Returns true when float32 color target blending is supported by this physical device.
    #[must_use]
    pub(super) fn supports_float32_blendable(&self) -> bool {
        let blendable = |format: vk::Format| {
            let props = unsafe {
                self.instance
                    .instance
                    .get_physical_device_format_properties(self.physical_device, format)
            };
            props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND)
        };
        blendable(vk::Format::R32_SFLOAT)
            && blendable(vk::Format::R32G32_SFLOAT)
            && blendable(vk::Format::R32G32B32A32_SFLOAT)
    }

    /// Returns the supported subgroup size range for this physical device.
    #[must_use]
    pub(super) fn subgroup_size_range(&self) -> Option<(u32, u32)> {
        self.subgroup_query().range
    }

    /// Returns the explicit compute subgroup size capabilities.
    ///
    /// `Some` iff WGSL `subgroups` is supported and `VK_EXT_subgroup_size_control`
    /// is present with `subgroupSizeControl`, `computeFullSubgroups`, and
    /// `COMPUTE` in `requiredSubgroupSizeStages` (Block 108 R2). The extension
    /// is required even on a Vulkan 1.3 device because yawgpu runs at
    /// `YAWGPU_VULKAN_API_VERSION` (1.1). The size range is
    /// [`Self::subgroup_size_range`]'s, so the two always agree.
    #[must_use]
    pub(super) fn subgroup_size_control_caps(&self) -> Option<crate::HalSubgroupSizeControlCaps> {
        self.subgroup_query().caps()
    }

    /// Returns the subgroup data of this physical device, queried on first use
    /// and shared by every clone of the adapter.
    fn subgroup_query(&self) -> &SubgroupQuery {
        self.subgroup_query
            .get_or_init(|| self.query_subgroup_properties())
    }

    /// Queries the subgroup properties and `VK_EXT_subgroup_size_control`
    /// features backing [`SubgroupQuery`].
    fn query_subgroup_properties(&self) -> SubgroupQuery {
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut size_control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup)
            .push_next(&mut size_control);
        unsafe {
            self.instance
                .instance
                .get_physical_device_properties2(self.physical_device, &mut properties2);
        }
        let api_version = unsafe {
            self.instance
                .instance
                .get_physical_device_properties(self.physical_device)
                .api_version
        };
        let extension_present = self.has_device_extension(vk::EXT_SUBGROUP_SIZE_CONTROL_NAME);
        let range = subgroup_size_range_from_properties(
            subgroups_supported(subgroup.supported_operations, subgroup.supported_stages),
            subgroup_size_control_available(api_version, extension_present),
            subgroup.subgroup_size,
            size_control.min_subgroup_size,
            size_control.max_subgroup_size,
        );
        let mut features = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
        if extension_present {
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features);
            unsafe {
                self.instance
                    .instance
                    .get_physical_device_features2(self.physical_device, &mut features2);
            }
        }
        SubgroupQuery {
            supported_stages: subgroup.supported_stages,
            range,
            extension_present,
            subgroup_size_control: features.subgroup_size_control,
            compute_full_subgroups: features.compute_full_subgroups,
            required_subgroup_size_stages: if extension_present {
                size_control.required_subgroup_size_stages
            } else {
                vk::ShaderStageFlags::empty()
            },
            max_compute_workgroup_subgroups: if extension_present {
                size_control.max_compute_workgroup_subgroups
            } else {
                0
            },
        }
    }

    /// Creates a device (and its default queue) on this adapter.
    pub fn create_device(&self) -> Result<VulkanDevice, HalError> {
        let queue_family_index = self
            .queue_family_index()
            .ok_or(HalError::DeviceCreationFailed { backend: BACKEND })?;
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);
        let queue_create_infos = [queue_create_info];
        let mut extension_names = Vec::new();
        // A portability implementation (VK_KHR_portability_subset, e.g. MoltenVK)
        // reports every optional subset feature it supports through
        // VkPhysicalDevicePortabilitySubsetFeaturesKHR, and each one must be
        // enabled *explicitly* at device creation: enabling the extension alone
        // leaves them all VK_FALSE, so the device behaves as if none were
        // supported. Without this chain the Khronos validation layer rejects
        // e.g. every non-identity VkComponentMapping with
        // VUID-VkImageViewCreateInfo-imageViewFormatSwizzle-04465, although
        // MoltenVK reports imageViewFormatSwizzle = VK_TRUE. Bits the driver
        // does not report stay VK_FALSE (enabling an unsupported feature is
        // itself invalid), so the queried bits are mirrored unmodified.
        let portability_subset_extension =
            self.has_device_extension(vk::KHR_PORTABILITY_SUBSET_NAME);
        if portability_subset_extension {
            extension_names.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }
        let mut portability_subset_supported_features =
            vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default();
        if portability_subset_extension {
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut portability_subset_supported_features);
            unsafe {
                self.instance
                    .instance
                    .get_physical_device_features2(self.physical_device, &mut features2);
            }
        }
        let portability_subset_features = enabled_portability_subset_features(
            portability_subset_extension,
            portability_subset_supported_features,
        );
        if self.has_device_extension(vk::KHR_SWAPCHAIN_NAME) {
            extension_names.push(vk::KHR_SWAPCHAIN_NAME.as_ptr());
        }
        let depth_clip_enable_extension = self.has_device_extension(vk::EXT_DEPTH_CLIP_ENABLE_NAME);
        let shader_float16_int8_extension =
            self.has_device_extension(vk::KHR_SHADER_FLOAT16_INT8_NAME);
        let storage_16bit_extension = self.has_device_extension(vk::KHR_16BIT_STORAGE_NAME);
        let vulkan_memory_model_extension =
            self.has_device_extension(vk::KHR_VULKAN_MEMORY_MODEL_NAME);
        let image_format_list_extension = self.has_device_extension(vk::KHR_IMAGE_FORMAT_LIST_NAME);
        let device_properties = unsafe {
            self.instance
                .instance
                .get_physical_device_properties(self.physical_device)
        };
        let spirv_extension = self.has_device_extension(vk::KHR_SPIRV_1_4_NAME);
        let float_controls_extension =
            self.has_device_extension(vk::KHR_SHADER_FLOAT_CONTROLS_NAME);
        let use_spirv_1_4 = crate::shader_workarounds::use_spirv_1_4(
            spirv_extension,
            float_controls_extension,
            cfg!(target_os = "android"),
            device_properties.vendor_id,
            device_properties.device_id,
        );
        append_spirv_1_4_extensions(
            &mut extension_names,
            spirv_extension,
            float_controls_extension,
        );
        // Promotion-to-core decisions must use the *effective enabled* Vulkan
        // version, not the physical device's *maximum supported* version
        // (`device_properties.api_version`). The effective version is
        // `min(YAWGPU_VULKAN_API_VERSION, device_max)`: yawgpu requests 1.1 at
        // vkCreateInstance, so even a 1.3-capable device runs as 1.1 here and a
        // feature promoted to core in 1.2 (e.g. vulkanMemoryModel,
        // VkImageFormatListCreateInfo) is NOT core — its extension name must still
        // be enabled. Deciding on device_max would chain the feature struct into
        // VkDeviceCreateInfo.pNext without enabling the parent extension
        // (VUID-VkDeviceCreateInfo-pNext-pNext).
        let effective_api_version = YAWGPU_VULKAN_API_VERSION.min(device_properties.api_version);
        let robustness2_enabled = self.has_device_extension(vk::EXT_ROBUSTNESS2_NAME);
        let pipeline_robustness_extension =
            self.has_device_extension(vk::EXT_PIPELINE_ROBUSTNESS_NAME);
        let mut robustness2_features = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        let mut pipeline_robustness_features =
            vk::PhysicalDevicePipelineRobustnessFeaturesEXT::default();
        let mut robustness_features = vk::PhysicalDeviceFeatures2::default();
        if robustness2_enabled {
            extension_names.push(vk::EXT_ROBUSTNESS2_NAME.as_ptr());
            robustness_features = robustness_features.push_next(&mut robustness2_features);
        }
        if pipeline_robustness_extension {
            robustness_features = robustness_features.push_next(&mut pipeline_robustness_features);
        }
        unsafe {
            self.instance
                .instance
                .get_physical_device_features2(self.physical_device, &mut robustness_features);
        }
        let pipeline_robustness = pipeline_robustness_extension
            && pipeline_robustness_features.pipeline_robustness == vk::TRUE;
        if pipeline_robustness {
            extension_names.push(vk::EXT_PIPELINE_ROBUSTNESS_NAME.as_ptr());
        }
        // Match Dawn's DeviceExt::DescriptorIndexing: only the advertised
        // extension permits this query, with no core-version promotion.
        // No descriptor-indexing features are enabled here.
        let descriptor_indexing_available = descriptor_indexing_available(
            device_properties.api_version,
            self.has_device_extension(vk::EXT_DESCRIPTOR_INDEXING_NAME),
        );
        let mut descriptor_indexing_properties =
            vk::PhysicalDeviceDescriptorIndexingProperties::default();
        if descriptor_indexing_available {
            let mut properties2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut descriptor_indexing_properties);
            unsafe {
                self.instance
                    .instance
                    .get_physical_device_properties2(self.physical_device, &mut properties2);
            }
        }
        let image_robust_access2 = use_image_robust_access2(
            robustness2_enabled,
            robustness2_features.robust_image_access2 == vk::TRUE,
        );
        // Reuse the queried bits without the query chain links.
        robustness2_features.p_next = std::ptr::null_mut();
        pipeline_robustness_features.p_next = std::ptr::null_mut();
        // Match Tint's native instructions to features actually enabled below.
        // Without demote support Tint retains its discard emulation.
        let shader_demote_extension =
            self.has_device_extension(vk::EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_NAME);
        let integer_dot_extension =
            self.has_device_extension(vk::KHR_SHADER_INTEGER_DOT_PRODUCT_NAME);
        let zero_initialize_extension =
            self.has_device_extension(vk::KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_NAME);
        let mut demote_features =
            vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures::default();
        let mut integer_dot_features = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut zero_initialize_features =
            vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures::default();
        let maximal_reconvergence_extension =
            self.has_device_extension(vk::KHR_SHADER_MAXIMAL_RECONVERGENCE_NAME);
        let uniform_control_flow_extension =
            self.has_device_extension(vk::KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_NAME);
        let mut maximal_reconvergence_features =
            vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default();
        let mut uniform_control_flow_features =
            vk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR::default();
        let mut writer_features = vk::PhysicalDeviceFeatures2::default();
        if maximal_reconvergence_extension {
            writer_features = writer_features.push_next(&mut maximal_reconvergence_features);
        }
        if uniform_control_flow_extension {
            writer_features = writer_features.push_next(&mut uniform_control_flow_features);
        }
        if writer_feature_available(effective_api_version, shader_demote_extension) {
            writer_features = writer_features.push_next(&mut demote_features);
        }
        if writer_feature_available(effective_api_version, integer_dot_extension) {
            writer_features = writer_features.push_next(&mut integer_dot_features);
        }
        if writer_feature_available(effective_api_version, zero_initialize_extension) {
            writer_features = writer_features.push_next(&mut zero_initialize_features);
        }
        unsafe {
            self.instance
                .instance
                .get_physical_device_features2(self.physical_device, &mut writer_features);
        }
        let shader_demote_to_helper_invocation = writer_feature_enabled(
            effective_api_version,
            shader_demote_extension,
            demote_features.shader_demote_to_helper_invocation,
        );
        let shader_integer_dot_product = writer_feature_enabled(
            effective_api_version,
            integer_dot_extension,
            integer_dot_features.shader_integer_dot_product,
        );
        let zero_initialize_workgroup_memory = zero_initialize_workgroup_memory_enabled(
            effective_api_version,
            zero_initialize_extension,
            zero_initialize_features.shader_zero_initialize_workgroup_memory,
            device_properties.vendor_id,
        );
        let reconvergence_features = enabled_reconvergence_features(
            maximal_reconvergence_extension,
            maximal_reconvergence_features.shader_maximal_reconvergence,
            uniform_control_flow_extension,
            uniform_control_flow_features.shader_subgroup_uniform_control_flow,
        );
        let reconvergence = select_reconvergence(
            reconvergence_features,
            self.subgroup_query().supported_stages,
            cfg!(target_os = "android"),
            device_properties.vendor_id,
        );
        if reconvergence_features.maximal {
            extension_names.push(vk::KHR_SHADER_MAXIMAL_RECONVERGENCE_NAME.as_ptr());
        }
        if reconvergence_features.uniform_control_flow {
            extension_names.push(vk::KHR_SHADER_SUBGROUP_UNIFORM_CONTROL_FLOW_NAME.as_ptr());
        }
        // These features were promoted in Vulkan 1.3. Use the effective API
        // version, so a 1.1 instance still enables the extension names.
        if effective_api_version < vk::API_VERSION_1_3 {
            if shader_integer_dot_product {
                extension_names.push(vk::KHR_SHADER_INTEGER_DOT_PRODUCT_NAME.as_ptr());
            }
            if zero_initialize_workgroup_memory {
                extension_names.push(vk::KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_NAME.as_ptr());
            }
        }
        let vulkan_memory_model_available =
            vulkan_memory_model_available(effective_api_version, vulkan_memory_model_extension);
        let mut vulkan_memory_model_supported_features =
            vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
        if vulkan_memory_model_available {
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut vulkan_memory_model_supported_features);
            unsafe {
                self.instance
                    .instance
                    .get_physical_device_features2(self.physical_device, &mut features2);
            }
        }
        let vulkan_memory_model = vulkan_memory_model_enabled(
            vulkan_memory_model_available,
            vulkan_memory_model_supported_features.vulkan_memory_model,
            vulkan_memory_model_supported_features.vulkan_memory_model_device_scope,
        );
        let vulkan_memory_model_device_scope = vulkan_memory_model;
        let image_format_list =
            image_format_list_available(effective_api_version, image_format_list_extension);
        let swapchain_mutable_format = image_format_list
            && self.has_device_extension(vk::KHR_SWAPCHAIN_NAME)
            && self.has_device_extension(vk::KHR_SWAPCHAIN_MUTABLE_FORMAT_NAME);
        if swapchain_mutable_format {
            extension_names.push(vk::KHR_SWAPCHAIN_MUTABLE_FORMAT_NAME.as_ptr());
        }
        let mut shader_float16_int8_features =
            vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut storage_16bit_supported_features =
            vk::PhysicalDevice16BitStorageFeatures::default();
        let storage_available =
            storage_16bit_available(effective_api_version, storage_16bit_extension);
        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        if storage_available {
            features2 = features2.push_next(&mut storage_16bit_supported_features);
        }
        if shader_float16_int8_extension {
            features2 = features2.push_next(&mut shader_float16_int8_features);
        }
        unsafe {
            self.instance
                .instance
                .get_physical_device_features2(self.physical_device, &mut features2);
        }
        let supported_features = features2.features;
        let shader_float16 = shader_float16_supported(
            shader_float16_int8_extension,
            shader_float16_int8_features.shader_float16,
            supported_features.shader_int16,
            storage_16bit_supported_features.storage_buffer16_bit_access,
            storage_16bit_supported_features.uniform_and_storage_buffer16_bit_access,
        );
        let storage_16bit_features =
            enabled_16bit_storage_features(storage_available, storage_16bit_supported_features);
        let occlusion_query_precise = supported_features.occlusion_query_precise == vk::TRUE;
        // Enable samplerAnisotropy when the physical device supports it.
        // Setting anisotropyEnable = true without this feature enabled is a VUID
        // violation and causes MoltenVK to produce error command buffers.
        let sampler_anisotropy = supported_features.sampler_anisotropy == vk::TRUE;
        // WebGPU requires OOB vertex-attribute fetches to be clamped/zeroed; Vulkan
        // robustBufferAccess guarantees bounded behaviour for vertex buffer reads,
        // covering both direct and indirect draws.  Enable it whenever the physical
        // device reports support (the spec mandates every Vulkan 1.0 device exposes
        // this feature, so the guard is defensive rather than required).
        let robust_buffer_access = supported_features.robust_buffer_access == vk::TRUE;
        // robustBufferAccess2 requires robustBufferAccess to be enabled
        // (VUID-VkPhysicalDeviceRobustness2FeaturesEXT-robustBufferAccess2-04000).
        if !robust_buffer_access {
            robustness2_features.robust_buffer_access2 = vk::FALSE;
        }
        let buffer_robust_access2 = use_buffer_robust_access2(
            robustness2_enabled,
            robustness2_features.robust_buffer_access2 == vk::TRUE,
            robust_buffer_access,
            descriptor_indexing_available,
            descriptor_indexing_properties.robust_buffer_access_update_after_bind == vk::TRUE,
        );
        // WebGPU render pipelines can configure blend/write masks per color target.
        // Vulkan requires independentBlend for differing per-attachment blend state.
        let independent_blend = supported_features.independent_blend == vk::TRUE;
        // Tier 1/2 storage formats are Vulkan extended storage-image formats. Dawn
        // enables this for TextureFormatsTier1; HAL has no requested feature set,
        // so enable it whenever the physical device supports it, like its neighbours.
        let shader_storage_image_extended_formats =
            supported_features.shader_storage_image_extended_formats == vk::TRUE;
        let dual_src_blend = supported_features.dual_src_blend == vk::TRUE;
        let shader_clip_distance = supported_features.shader_clip_distance == vk::TRUE;
        let geometry_shader = supported_features.geometry_shader == vk::TRUE;
        let draw_indirect_first_instance =
            supported_features.draw_indirect_first_instance == vk::TRUE;
        let depth_clamp = supported_features.depth_clamp == vk::TRUE;
        // VK_EXT_depth_clip_enable gives independent clip control; without it,
        // core depthClampEnable already yields WebGPU unclippedDepth semantics.
        let depth_clip_control = depth_clamp && depth_clip_enable_extension;
        // Tint emits `OpCapability SampleRateShading` whenever a shader uses
        // `@builtin(sample_mask)` or `@builtin(sample_index)` (the latter also
        // backs per-sample MSAA subpass input reads on tiled builds, where
        // SPIR-V `SampleId` auto-promotes the fragment shader to per-sample
        // execution); Vulkan requires sampleRateShading for that capability.
        let sample_rate_shading = supported_features.sample_rate_shading == vk::TRUE;
        // Tint-generated fragment shaders may write storage buffers/textures;
        // Vulkan requires fragmentStoresAndAtomics for fragment-stage storage
        // writes (VUID-RuntimeSpirv-NonWritable-06340).
        let fragment_stores_and_atomics =
            supported_features.fragment_stores_and_atomics == vk::TRUE;
        let mut enabled_features = enabled_texture_compression_features(&supported_features);
        if shader_float16 {
            enabled_features.shader_int16 = vk::TRUE;
        }
        if occlusion_query_precise {
            enabled_features.occlusion_query_precise = vk::TRUE;
        }
        if depth_clamp {
            enabled_features.depth_clamp = vk::TRUE;
        }
        if depth_clip_control {
            extension_names.push(vk::EXT_DEPTH_CLIP_ENABLE_NAME.as_ptr());
        }
        if sampler_anisotropy {
            enabled_features.sampler_anisotropy = vk::TRUE;
        }
        if robust_buffer_access {
            enabled_features.robust_buffer_access = vk::TRUE;
        }
        if independent_blend {
            enabled_features.independent_blend = vk::TRUE;
        }
        if shader_storage_image_extended_formats {
            enabled_features.shader_storage_image_extended_formats = vk::TRUE;
        }
        if dual_src_blend {
            enabled_features.dual_src_blend = vk::TRUE;
        }
        if shader_clip_distance {
            enabled_features.shader_clip_distance = vk::TRUE;
        }
        if geometry_shader {
            enabled_features.geometry_shader = vk::TRUE;
        }
        if draw_indirect_first_instance {
            enabled_features.draw_indirect_first_instance = vk::TRUE;
        }
        if sample_rate_shading {
            enabled_features.sample_rate_shading = vk::TRUE;
        }
        if fragment_stores_and_atomics {
            enabled_features.fragment_stores_and_atomics = vk::TRUE;
        }
        if shader_demote_to_helper_invocation && effective_api_version < vk::API_VERSION_1_3 {
            extension_names.push(vk::EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_NAME.as_ptr());
        }
        if shader_float16 {
            extension_names.push(vk::KHR_SHADER_FLOAT16_INT8_NAME.as_ptr());
        }
        if storage_16bit_features.enabled && storage_16bit_extension {
            extension_names.push(vk::KHR_16BIT_STORAGE_NAME.as_ptr());
        }
        if vulkan_memory_model
            && vulkan_memory_model_extension_required(
                effective_api_version,
                vulkan_memory_model_extension,
            )
        {
            extension_names.push(vk::KHR_VULKAN_MEMORY_MODEL_NAME.as_ptr());
        }
        if image_format_list_extension {
            extension_names.push(vk::KHR_IMAGE_FORMAT_LIST_NAME.as_ptr());
        }
        // Block 108 R5: like Dawn's device-level knob, enable
        // VK_EXT_subgroup_size_control whenever WGSL subgroups are supported
        // and the driver reports `subgroupSizeControl`, independent of the
        // requested WebGPU features and of whether `subgroup-size-control`
        // is advertised. Pipelines without `@subgroup_size` use the selected
        // default size or ALLOW_VARYING_SUBGROUP_SIZE; REQUIRE_FULL_SUBGROUPS
        // additionally needs `computeFullSubgroups`, which is enabled only
        // when the driver reports it (the WebGPU feature requires it).
        let subgroup_query = *self.subgroup_query();
        let subgroup_size_control = subgroup_query.extension_usable();
        if subgroup_size_control {
            extension_names.push(vk::EXT_SUBGROUP_SIZE_CONTROL_NAME.as_ptr());
        }
        let mut subgroup_size_control_features =
            vk::PhysicalDeviceSubgroupSizeControlFeatures::default()
                .subgroup_size_control(true)
                .compute_full_subgroups(subgroup_query.compute_full_subgroups == vk::TRUE);
        let default_compute_subgroup_size = subgroup_query.default_compute_subgroup_size();
        let mut maximal_reconvergence_enable_features =
            vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default()
                .shader_maximal_reconvergence(true);
        let mut uniform_control_flow_enable_features =
            vk::PhysicalDeviceShaderSubgroupUniformControlFlowFeaturesKHR::default()
                .shader_subgroup_uniform_control_flow(true);
        let mut depth_clip_enable_features =
            vk::PhysicalDeviceDepthClipEnableFeaturesEXT::default().depth_clip_enable(true);
        let mut shader_demote_features =
            vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures::default()
                .shader_demote_to_helper_invocation(true);
        // Keep query chains separate: reusing a queried struct would retain
        // its pNext links and append duplicate feature structures at creation.
        let mut integer_dot_enable_features =
            vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
                .shader_integer_dot_product(true);
        let mut zero_initialize_enable_features =
            vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures::default()
                .shader_zero_initialize_workgroup_memory(true);
        let mut shader_float16_int8_enable_features =
            vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
        let mut vulkan_memory_model_enable_features =
            vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
                .vulkan_memory_model(true)
                .vulkan_memory_model_device_scope(vulkan_memory_model_device_scope);
        let mut storage_16bit_enable_features = storage_16bit_features.to_vk();
        let mut portability_subset_enable_features = portability_subset_features.to_vk();
        let mut create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names)
            .enabled_features(&enabled_features);
        if robustness2_enabled {
            create_info = create_info.push_next(&mut robustness2_features);
        }
        if pipeline_robustness {
            create_info = create_info.push_next(&mut pipeline_robustness_features);
        }
        if depth_clip_control {
            create_info = create_info.push_next(&mut depth_clip_enable_features);
        }
        if shader_demote_to_helper_invocation {
            create_info = create_info.push_next(&mut shader_demote_features);
        }
        if shader_integer_dot_product {
            create_info = create_info.push_next(&mut integer_dot_enable_features);
        }
        if zero_initialize_workgroup_memory {
            create_info = create_info.push_next(&mut zero_initialize_enable_features);
        }
        if shader_float16 {
            create_info = create_info.push_next(&mut shader_float16_int8_enable_features);
        }
        if vulkan_memory_model {
            create_info = create_info.push_next(&mut vulkan_memory_model_enable_features);
        }
        if storage_16bit_features.enabled {
            create_info = create_info.push_next(&mut storage_16bit_enable_features);
        }
        if portability_subset_features.enabled {
            create_info = create_info.push_next(&mut portability_subset_enable_features);
        }
        if subgroup_size_control {
            create_info = create_info.push_next(&mut subgroup_size_control_features);
        }
        if reconvergence_features.maximal {
            create_info = create_info.push_next(&mut maximal_reconvergence_enable_features);
        }
        if reconvergence_features.uniform_control_flow {
            create_info = create_info.push_next(&mut uniform_control_flow_enable_features);
        }
        let device = unsafe {
            self.instance
                .instance
                .create_device(self.physical_device, &create_info, None)
        }
        .map_err(|_| HalError::DeviceCreationFailed { backend: BACKEND })?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let memory_properties = unsafe {
            self.instance
                .instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        // Query the device property limit for anisotropy so create_sampler can clamp.
        let max_sampler_anisotropy = device_properties.limits.max_sampler_anisotropy;
        let driver_id = query_driver_id(&self.instance, self.physical_device);
        let forced = std::env::var("YAWGPU_FORCE_SHADER_WORKAROUNDS").ok();
        let shader_workarounds = crate::HalShaderWorkarounds::for_vulkan(
            device_properties.vendor_id,
            device_properties.driver_version,
            device_properties.device_id,
            driver_id,
            cfg!(target_os = "android"),
            cfg!(target_os = "windows"),
        )
        .with_forced(crate::HalShaderWorkarounds::parse_forced(forced.as_deref()));
        let mut shader_workarounds = vulkan_writer_flags(
            shader_workarounds,
            shader_demote_to_helper_invocation,
            shader_integer_dot_product,
            zero_initialize_workgroup_memory,
            reconvergence,
            image_robust_access2,
            buffer_robust_access2,
        );
        shader_workarounds.use_spirv_1_4 = use_spirv_1_4;
        let inner = Arc::new(VulkanDeviceInner {
            #[cfg(test)]
            spirv_1_4_extensions_enabled: [
                vk::KHR_SPIRV_1_4_NAME,
                vk::KHR_SHADER_FLOAT_CONTROLS_NAME,
            ]
            .map(|name| {
                extension_names
                    .iter()
                    .any(|&enabled| (unsafe { CStr::from_ptr(enabled) }) == name)
            }),
            shader_workarounds,
            reconvergence_features,
            pipeline_robustness,
            _instance: Arc::clone(&self.instance),
            device,
            physical_device: self.physical_device,
            memory_properties,
            queue_family_index,
            occlusion_query_precise,
            depth_clip_control,
            sampler_anisotropy,
            shader_demote_to_helper_invocation,
            shader_float16,
            vulkan_memory_model,
            image_format_list,
            swapchain_mutable_format,
            subgroup_size_control,
            default_compute_subgroup_size,
            storage_buffer16_bit_access: storage_16bit_features.storage_buffer16_bit_access,
            uniform_and_storage_buffer16_bit_access: storage_16bit_features
                .uniform_and_storage_buffer16_bit_access,
            storage_input_output16: storage_16bit_features.storage_input_output16,
            storage_push_constant16: storage_16bit_features.storage_push_constant16,
            max_sampler_anisotropy,
            surfaces: Mutex::new(Vec::new()),
            allocations: AtomicU64::new(0),
        });
        Ok(VulkanDevice {
            inner: Arc::clone(&inner),
            queue: VulkanQueue {
                inner: Arc::new(VulkanQueueInner {
                    device: inner,
                    queue,
                    queue_access: Mutex::new(()),
                    retire: Mutex::new(SubmissionRetireQueue::default()),
                    recycler: Arc::new(Mutex::new(SubmitRecycler::default())),
                    submissions: Arc::new(Mutex::new(SubmissionTracker::new())),
                }),
            },
        })
    }

    fn queue_family_index(&self) -> Option<u32> {
        let families = unsafe {
            self.instance
                .instance
                .get_physical_device_queue_family_properties(self.physical_device)
        };
        families.iter().enumerate().find_map(|(index, family)| {
            let flags = family.queue_flags;
            (flags.contains(vk::QueueFlags::GRAPHICS)
                && flags.contains(vk::QueueFlags::COMPUTE)
                && family.queue_count > 0)
                .then(|| u32::try_from(index).ok())
                .flatten()
        })
    }

    fn has_device_extension(&self, name: &CStr) -> bool {
        has_device_extension_for_physical_device(&self.instance, self.physical_device, name)
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Enabled16BitStorageFeatures {
    enabled: bool,
    storage_buffer16_bit_access: bool,
    uniform_and_storage_buffer16_bit_access: bool,
    storage_input_output16: bool,
    storage_push_constant16: bool,
}

impl Enabled16BitStorageFeatures {
    fn to_vk(self) -> vk::PhysicalDevice16BitStorageFeatures<'static> {
        vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(self.storage_buffer16_bit_access)
            .uniform_and_storage_buffer16_bit_access(self.uniform_and_storage_buffer16_bit_access)
            .storage_input_output16(self.storage_input_output16)
            .storage_push_constant16(self.storage_push_constant16)
    }
}

/// The `VK_KHR_portability_subset` features to enable at device creation, as a
/// pointer-free mirror of `VkPhysicalDevicePortabilitySubsetFeaturesKHR` (the
/// queried struct carries a `pNext` that belongs to the query chain, so the
/// bits are copied out and rebuilt rather than re-chained in place).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EnabledPortabilitySubsetFeatures {
    /// True when `VK_KHR_portability_subset` is present, i.e. the feature
    /// struct must be chained into `VkDeviceCreateInfo.pNext`.
    enabled: bool,
    constant_alpha_color_blend_factors: bool,
    events: bool,
    image_view_format_reinterpretation: bool,
    image_view_format_swizzle: bool,
    image_view2_d_on3_d_image: bool,
    multisample_array_image: bool,
    mutable_comparison_samplers: bool,
    point_polygons: bool,
    sampler_mip_lod_bias: bool,
    separate_stencil_mask_ref: bool,
    shader_sample_rate_interpolation_functions: bool,
    tessellation_isolines: bool,
    tessellation_point_mode: bool,
    triangle_fans: bool,
    vertex_attribute_access_beyond_stride: bool,
}

impl EnabledPortabilitySubsetFeatures {
    fn to_vk(self) -> vk::PhysicalDevicePortabilitySubsetFeaturesKHR<'static> {
        vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default()
            .constant_alpha_color_blend_factors(self.constant_alpha_color_blend_factors)
            .events(self.events)
            .image_view_format_reinterpretation(self.image_view_format_reinterpretation)
            .image_view_format_swizzle(self.image_view_format_swizzle)
            .image_view2_d_on3_d_image(self.image_view2_d_on3_d_image)
            .multisample_array_image(self.multisample_array_image)
            .mutable_comparison_samplers(self.mutable_comparison_samplers)
            .point_polygons(self.point_polygons)
            .sampler_mip_lod_bias(self.sampler_mip_lod_bias)
            .separate_stencil_mask_ref(self.separate_stencil_mask_ref)
            .shader_sample_rate_interpolation_functions(
                self.shader_sample_rate_interpolation_functions,
            )
            .tessellation_isolines(self.tessellation_isolines)
            .tessellation_point_mode(self.tessellation_point_mode)
            .triangle_fans(self.triangle_fans)
            .vertex_attribute_access_beyond_stride(self.vertex_attribute_access_beyond_stride)
    }
}

/// Maps all ASTC LDR variants through the texture creation format mapping.
fn astc_ldr_formats() -> impl Iterator<Item = Result<vk::Format, HalError>> {
    use HalTextureFormat::*;
    [
        Astc4x4Unorm,
        Astc4x4UnormSrgb,
        Astc5x4Unorm,
        Astc5x4UnormSrgb,
        Astc5x5Unorm,
        Astc5x5UnormSrgb,
        Astc6x5Unorm,
        Astc6x5UnormSrgb,
        Astc6x6Unorm,
        Astc6x6UnormSrgb,
        Astc8x5Unorm,
        Astc8x5UnormSrgb,
        Astc8x6Unorm,
        Astc8x6UnormSrgb,
        Astc8x8Unorm,
        Astc8x8UnormSrgb,
        Astc10x5Unorm,
        Astc10x5UnormSrgb,
        Astc10x6Unorm,
        Astc10x6UnormSrgb,
        Astc10x8Unorm,
        Astc10x8UnormSrgb,
        Astc10x10Unorm,
        Astc10x10UnormSrgb,
        Astc12x10Unorm,
        Astc12x10UnormSrgb,
        Astc12x12Unorm,
        Astc12x12UnormSrgb,
    ]
    .into_iter()
    .map(|format| format::map_texture_format(format).map(|(format, _)| format))
}

/// Requires base ASTC support and successful 3D probes for every mapped format.
fn astc_sliced_3d_supported(
    astc_supported: bool,
    mut probe: impl FnMut(vk::Format) -> bool,
) -> bool {
    astc_supported && astc_ldr_formats().all(|format| format.is_ok_and(&mut probe))
}

/// Enables each supported compression family; HAL has no requested-feature set.
fn enabled_texture_compression_features(
    supported: &vk::PhysicalDeviceFeatures,
) -> vk::PhysicalDeviceFeatures {
    let mut enabled = vk::PhysicalDeviceFeatures::default();
    if supported.texture_compression_bc == vk::TRUE {
        enabled.texture_compression_bc = vk::TRUE;
    }
    if supported.texture_compression_etc2 == vk::TRUE {
        enabled.texture_compression_etc2 = vk::TRUE;
    }
    if supported.texture_compression_astc_ldr == vk::TRUE {
        enabled.texture_compression_astc_ldr = vk::TRUE;
    }
    enabled
}

fn storage_16bit_available(api_version: u32, extension_present: bool) -> bool {
    api_version >= vk::API_VERSION_1_1 || extension_present
}

fn shader_float16_supported(
    extension_present: bool,
    shader_float16: vk::Bool32,
    shader_int16: vk::Bool32,
    storage_buffer16: vk::Bool32,
    uniform_and_storage_buffer16: vk::Bool32,
) -> bool {
    extension_present
        && shader_float16 == vk::TRUE
        && shader_int16 == vk::TRUE
        && storage_buffer16 == vk::TRUE
        && uniform_and_storage_buffer16 == vk::TRUE
}

/// Block 106: WebGPU `texture-component-swizzle` needs a non-identity
/// `VkComponentMapping` on the texture view. Dawn advertises the feature
/// unconditionally on Vulkan, which holds for every non-portability driver. A
/// portability implementation may not support it at all, so there the
/// advertisement follows the reported `imageViewFormatSwizzle` bit.
fn texture_component_swizzle_supported(
    portability_subset_present: bool,
    image_view_format_swizzle: vk::Bool32,
) -> bool {
    !portability_subset_present || image_view_format_swizzle == vk::TRUE
}

/// Block 99 R3: `timestamp-query` iff `limits.timestampComputeAndGraphics == VK_TRUE`.
fn timestamp_query_from_limits(timestamp_compute_and_graphics: vk::Bool32) -> bool {
    timestamp_compute_and_graphics == vk::TRUE
}

/// Block 102 R3: the reported timestamp period is nanoseconds per tick, so it
/// must be finite and strictly positive. A driver reporting `0` or a
/// non-finite value would make the nanosecond conversion pass produce zeroes
/// or NaNs, so such values fall back to one nanosecond per tick.
fn timestamp_period_from_limits(period: f32) -> f32 {
    if period.is_finite() && period > 0.0 {
        period
    } else {
        1.0
    }
}

/// Block 99 R4: `depth32float-stencil8` iff `D32_SFLOAT_S8_UINT` optimal-tiling
/// features contain `DEPTH_STENCIL_ATTACHMENT`.
fn depth32float_stencil8_from_flags(flags: vk::FormatFeatureFlags) -> bool {
    flags.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
}

/// Block 99 R4: `rg11b10ufloat-renderable` iff `B10G11R11_UFLOAT_PACK32`
/// optimal-tiling features contain `COLOR_ATTACHMENT | COLOR_ATTACHMENT_BLEND`.
fn rg11b10ufloat_renderable_from_flags(flags: vk::FormatFeatureFlags) -> bool {
    flags.contains(
        vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND,
    )
}

/// Block 99 R4: `bgra8unorm-storage` iff `B8G8R8A8_UNORM` optimal-tiling
/// features contain `STORAGE_IMAGE`.
fn bgra8unorm_storage_from_flags(flags: vk::FormatFeatureFlags) -> bool {
    flags.contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
}

/// Block 99 R4: `float32-filterable` iff every 32-bit float format
/// (`R32_SFLOAT`, `R32G32_SFLOAT`, `R32G32B32A32_SFLOAT`) has
/// `SAMPLED_IMAGE_FILTER_LINEAR` in its optimal-tiling features.
fn float32_filterable_from_flags(
    r32: vk::FormatFeatureFlags,
    rg32: vk::FormatFeatureFlags,
    rgba32: vk::FormatFeatureFlags,
) -> bool {
    [r32, rg32, rgba32]
        .into_iter()
        .all(|flags| flags.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR))
}

fn subgroups_supported(
    supported_operations: vk::SubgroupFeatureFlags,
    supported_stages: vk::ShaderStageFlags,
) -> bool {
    let required_operations = vk::SubgroupFeatureFlags::BASIC
        | vk::SubgroupFeatureFlags::BALLOT
        | vk::SubgroupFeatureFlags::SHUFFLE
        | vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE
        | vk::SubgroupFeatureFlags::ARITHMETIC
        | vk::SubgroupFeatureFlags::QUAD;
    let required_stages = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT;
    // Deviation from Dawn: subgroup advertisement does not require
    // VK_EXT_subgroup_size_control. When the extension is enabled (Block 108
    // R5: present with `subgroupSizeControl`, see
    // `subgroup_size_control_extension_usable`), compute pipelines without
    // `@subgroup_size` require the selected default size when sizes vary,
    // otherwise ALLOW_VARYING_SUBGROUP_SIZE; without the extension they use
    // the driver's default.
    supported_operations.contains(required_operations) && supported_stages.contains(required_stages)
}

fn validated_subgroup_size_range(min: u32, max: u32) -> Option<(u32, u32)> {
    if min < 4 || max > 128 {
        return None;
    }
    Some((min, max))
}

/// Selects the Vulkan-backed WebGPU `maxBufferSize`.
///
/// Dawn prefers `VkPhysicalDeviceMaintenance4Properties::maxBufferSize`
/// (`PhysicalDeviceVk.cpp:888`) because NVIDIA can report the Maintenance3
/// `maxMemoryAllocationSize` as `u64::MAX`, which is not a usable finite
/// WebGPU buffer limit.
fn select_max_buffer_size(maintenance4: u64, maintenance3: u64) -> u64 {
    if maintenance4 != 0 {
        maintenance4
    } else if maintenance3 != 0 {
        maintenance3
    } else {
        ASSUMED_MAX_BUFFER_SIZE
    }
}

/// Maps a Vulkan-reported `VkPhysicalDeviceLimits` (plus the separately queried
/// `maxBufferSize` and vendor ID) onto the HAL's supported-limits view.
///
/// Split out of [`VulkanAdapter::limits`] so the driver-value mapping is
/// testable without a live device (Block 92 P92.4).
fn hal_limits_from_vk(
    vk: vk::PhysicalDeviceLimits,
    max_buffer_size: u64,
    vendor_id: u32,
) -> HalLimits {
    // Ported from Dawn's PhysicalDeviceVk.cpp InitializeSupportedLimitsImpl
    // and Limits.cpp NormalizeLimits.
    let max_storage_buffer_binding_size = if vendor_id == 0x10de {
        // NVIDIA's OpArrayLength can return zero for bindings of 2 GiB or more.
        u64::from(vk.max_storage_buffer_range)
            .min(max_buffer_size)
            .min((1_u64 << 31) - 4)
    } else {
        u64::from(vk.max_storage_buffer_range)
    };
    let max_storage_buffer_binding_size =
        max_storage_buffer_binding_size.min(max_buffer_size) / 4 * 4;
    let mut limits = HalLimits {
        max_texture_dimension_1d: vk.max_image_dimension1_d,
        max_texture_dimension_2d: vk
            .max_image_dimension2_d
            .min(vk.max_image_dimension_cube)
            .min(vk.max_framebuffer_width)
            .min(vk.max_framebuffer_height)
            .min(vk.max_viewport_dimensions[0])
            .min(vk.max_viewport_dimensions[1]),
        max_texture_dimension_3d: vk.max_image_dimension3_d,
        max_texture_array_layers: vk.max_image_array_layers,
        max_bind_groups: vk.max_bound_descriptor_sets.min(4),
        // Dawn advertised tier max (Limits.cpp:74-75), not the internal
        // kMax*=16 array ceiling; keeping <= maxUniformBuffersPerShaderStage
        // (12) keeps the CTS atMaximum single-stage case consistent.
        max_dynamic_uniform_buffers_per_pipeline_layout: vk
            .max_descriptor_set_uniform_buffers_dynamic
            .min(10),
        max_dynamic_storage_buffers_per_pipeline_layout: vk
            .max_descriptor_set_storage_buffers_dynamic
            .min(8),
        max_sampled_textures_per_shader_stage: vk.max_per_stage_descriptor_sampled_images.min(48),
        max_samplers_per_shader_stage: vk.max_per_stage_descriptor_samplers.min(16),
        max_storage_buffers_per_shader_stage: vk.max_per_stage_descriptor_storage_buffers.min(16),
        max_storage_textures_per_shader_stage: vk.max_per_stage_descriptor_storage_images.min(8),
        max_uniform_buffers_per_shader_stage: vk.max_per_stage_descriptor_uniform_buffers.min(12),
        max_uniform_buffer_binding_size: u64::from(vk.max_uniform_buffer_range / 16 * 16)
            .min(max_buffer_size),
        max_storage_buffer_binding_size,
        min_uniform_buffer_offset_alignment: vk.min_uniform_buffer_offset_alignment as u32,
        min_storage_buffer_offset_alignment: vk.min_storage_buffer_offset_alignment as u32,
        max_vertex_buffers: vk.max_vertex_input_bindings.min(8),
        max_buffer_size,
        max_vertex_attributes: vk.max_vertex_input_attributes.min(30),
        max_vertex_buffer_array_stride: vk
            .max_vertex_input_binding_stride
            // Block 92 P92.4: `maxVertexInputAttributeOffset` has no
            // spec-mandated upper bound and RADV reports `u32::MAX`, so an
            // unchecked `+ 1` overflows (abort under overflow checks, wrap to
            // 0 in release). Saturating keeps the following `.min(2048)` correct.
            .min(vk.max_vertex_input_attribute_offset.saturating_add(1))
            .min(2048),
        max_inter_stage_shader_variables: (vk
            .max_vertex_output_components
            .min(vk.max_fragment_input_components)
            / 4)
        .saturating_sub(2),
        max_color_attachments: vk.max_color_attachments.min(8),
        max_color_attachment_bytes_per_sample: 32_u32.saturating_mul(vk.max_color_attachments),
        max_compute_workgroup_storage_size: vk.max_compute_shared_memory_size,
        max_compute_invocations_per_workgroup: vk.max_compute_work_group_invocations,
        max_compute_workgroup_size_x: vk.max_compute_work_group_size[0],
        max_compute_workgroup_size_y: vk.max_compute_work_group_size[1],
        max_compute_workgroup_size_z: vk.max_compute_work_group_size[2],
        max_compute_workgroups_per_dimension: vk.max_compute_work_group_count[0]
            .min(vk.max_compute_work_group_count[1])
            .min(vk.max_compute_work_group_count[2]),
        // Block 94 S3: Vulkan now executes SetImmediates (push-constant
        // delivery in encode.rs), so it advertises Dawn's base-tier
        // `maxImmediateSize` (`kMaxImmediateDataBytes`,
        // dawn/common/Constants.h). The Vulkan-minimum
        // `maxPushConstantsSize >= 128` always covers the 64-byte user
        // region plus the 8-byte internal depth-range constants. GLES
        // (Tier 2) stays 0 via `HalLimits::DEFAULT`.
        max_immediate_size: 64,
        ..HalLimits::DEFAULT
    };
    let base = HalLimits::DEFAULT;
    let minimum = base.max_storage_buffers_per_shader_stage
        + base.max_storage_textures_per_shader_stage
        + base.max_color_attachments;
    let maximum = limits.max_storage_buffers_per_shader_stage
        + limits.max_storage_textures_per_shader_stage
        + limits.max_color_attachments;
    let combined = vk.max_fragment_combined_output_resources;
    if combined > minimum && combined < maximum {
        let mut extra = combined - minimum;
        limits.max_color_attachments =
            (base.max_color_attachments + extra / 3).min(vk.max_color_attachments);
        // Saturating subtraction also tolerates synthetic, below-default driver facts.
        extra -= limits
            .max_color_attachments
            .saturating_sub(base.max_color_attachments);
        limits.max_storage_textures_per_shader_stage = (base.max_storage_textures_per_shader_stage
            + extra / 2)
            .min(vk.max_per_stage_descriptor_storage_images);
        extra -= limits
            .max_storage_textures_per_shader_stage
            .saturating_sub(base.max_storage_textures_per_shader_stage);
        limits.max_storage_buffers_per_shader_stage = (base.max_storage_buffers_per_shader_stage
            + extra)
            .min(vk.max_per_stage_descriptor_storage_buffers);
    }
    limits.max_color_attachments = limits.max_color_attachments.min(8);
    limits.max_storage_textures_per_shader_stage =
        limits.max_storage_textures_per_shader_stage.min(8);
    limits.max_storage_buffers_per_shader_stage =
        limits.max_storage_buffers_per_shader_stage.min(16);
    limits
}

/// Matches Dawn's extension-only availability, regardless of the device API version.
fn descriptor_indexing_available(_device_api_version: u32, extension_present: bool) -> bool {
    extension_present
}

/// Uses driver image robustness only when robustness2 and its image bit are enabled.
fn use_image_robust_access2(robustness2_enabled: bool, robust_image_access2: bool) -> bool {
    robustness2_enabled && robust_image_access2
}

/// Matches Dawn's conservative descriptor-indexing exception for buffer robustness.
fn use_buffer_robust_access2(
    robustness2_enabled: bool,
    robust_buffer_access2: bool,
    robust_buffer_access_enabled: bool,
    descriptor_indexing_available: bool,
    robust_buffer_access_update_after_bind: bool,
) -> bool {
    robustness2_enabled
        && robust_buffer_access2
        && robust_buffer_access_enabled
        && (!descriptor_indexing_available || robust_buffer_access_update_after_bind)
}

/// Appends both SPIR-V 1.4 dependencies when both are reported, regardless of codegen.
fn append_spirv_1_4_extensions(
    extension_names: &mut Vec<*const c_char>,
    spirv_extension: bool,
    float_controls_extension: bool,
) {
    if spirv_extension && float_controls_extension {
        extension_names.push(vk::KHR_SPIRV_1_4_NAME.as_ptr());
        extension_names.push(vk::KHR_SHADER_FLOAT_CONTROLS_NAME.as_ptr());
    }
}

/// Adds enabled writer features without changing device-selected driver workarounds.
fn vulkan_writer_flags(
    workarounds: crate::HalShaderWorkarounds,
    demote_enabled: bool,
    integer_dot_enabled: bool,
    zero_init_enabled: bool,
    reconvergence: Reconvergence,
    image_robust_access2: bool,
    buffer_robust_access2: bool,
) -> crate::HalShaderWorkarounds {
    crate::HalShaderWorkarounds {
        disable_image_robustness: image_robust_access2,
        disable_storage_runtime_sized_array_index_clamping: buffer_robust_access2,
        use_demote_to_helper_invocation: demote_enabled,
        native_dot_4x8_packed: integer_dot_enabled,
        use_zero_initialize_workgroup_memory: zero_init_enabled,
        use_maximal_reconvergence: reconvergence == Reconvergence::Maximal,
        use_subgroup_uniform_control_flow: reconvergence == Reconvergence::UniformControlFlow,
        ..workarounds
    }
}

/// Reconvergence features enabled independently for generated and passthrough SPIR-V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReconvergenceFeatures {
    maximal: bool,
    uniform_control_flow: bool,
}

/// Dawn enables each supported device feature independently of its Tint toggles.
fn enabled_reconvergence_features(
    maximal_extension: bool,
    maximal_feature: vk::Bool32,
    uniform_extension: bool,
    uniform_feature: vk::Bool32,
) -> ReconvergenceFeatures {
    ReconvergenceFeatures {
        maximal: maximal_extension && maximal_feature == vk::TRUE,
        uniform_control_flow: uniform_extension && uniform_feature == vk::TRUE,
    }
}

/// Mutually exclusive Tint execution modes; device features may both be enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reconvergence {
    None,
    Maximal,
    UniformControlFlow,
}

/// Dawn prefers maximal reconvergence, then uniform control flow in both stages.
fn select_reconvergence(
    enabled: ReconvergenceFeatures,
    supported_stages: vk::ShaderStageFlags,
    is_android: bool,
    vendor_id: u32,
) -> Reconvergence {
    // Dawn's IsAndroidImgTec tests the vendor regardless of driver identity.
    if is_android && vendor_id == 0x1010 {
        Reconvergence::None
    } else if enabled.maximal {
        Reconvergence::Maximal
    } else if enabled.uniform_control_flow
        && supported_stages.contains(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT)
    {
        Reconvergence::UniformControlFlow
    } else {
        Reconvergence::None
    }
}

/// Dawn's FindDefaultComputeSubgroupSize heuristic, independent of WebGPU requests.
fn find_default_compute_subgroup_size(
    size_control_enabled: bool,
    required_subgroup_size_stages: vk::ShaderStageFlags,
    min_size: u32,
    max_size: u32,
    max_compute_workgroup_subgroups: u32,
) -> Option<u32> {
    // Dawn's FindDefaultComputeSubgroupSize lacks this stage check. Deliberately
    // deviate to avoid VUID-02755 on drivers such as MoltenVK.
    if !size_control_enabled
        || !required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE)
        || min_size == max_size
    {
        return None;
    }
    let size = match min_size.checked_mul(2) {
        Some(size) if size <= max_size => size,
        _ => min_size,
    };
    // Core raises the invocation limit to at least 256. A smaller subgroup
    // budget would make a valid 256-invocation workgroup violate VUID-02756.
    if u64::from(max_compute_workgroup_subgroups) * u64::from(size) < 256 {
        return None;
    }
    Some(size)
}

/// Required compute sizes constrain invocations even without full subgroups.
fn clamp_compute_invocations_per_workgroup(
    invocations: u32,
    max_workgroup_subgroups: u32,
    default_size: Option<u32>,
) -> u32 {
    default_size.map_or(invocations, |size| {
        invocations.min(max_workgroup_subgroups.saturating_mul(size))
    })
}

/// Whether a Vulkan 1.3 writer feature can be queried or enabled.
fn writer_feature_available(api_version: u32, extension_present: bool) -> bool {
    // The Vulkan 1.3 branch is dormant while YAWGPU_VULKAN_API_VERSION is 1.1.
    api_version >= vk::API_VERSION_1_3 || extension_present
}

/// Native writer instructions require both availability and the feature bit.
fn writer_feature_enabled(api_version: u32, extension_present: bool, feature: vk::Bool32) -> bool {
    writer_feature_available(api_version, extension_present) && feature == vk::TRUE
}

/// Dawn disables native workgroup zero initialization on ARM and Imagination.
fn zero_initialize_workgroup_memory_enabled(
    api_version: u32,
    extension_present: bool,
    feature: vk::Bool32,
    vendor_id: u32,
) -> bool {
    // Dawn enables the feature on ARM/Imagination and gates only Tint; yawgpu gates both, with no behavioural difference.
    writer_feature_enabled(api_version, extension_present, feature)
        && !matches!(vendor_id, 0x13b5 | 0x1010)
}

/// Tint always emits device-scope capability when using the Vulkan memory model.
fn vulkan_memory_model_enabled(
    available: bool,
    model: vk::Bool32,
    device_scope: vk::Bool32,
) -> bool {
    available && model == vk::TRUE && device_scope == vk::TRUE
}

fn vulkan_memory_model_available(api_version: u32, extension_present: bool) -> bool {
    let major = vk::api_version_major(api_version);
    let minor = vk::api_version_minor(api_version);
    (major, minor) >= (1, 2) || extension_present
}

fn vulkan_memory_model_extension_required(api_version: u32, extension_present: bool) -> bool {
    let major = vk::api_version_major(api_version);
    let minor = vk::api_version_minor(api_version);
    (major, minor) < (1, 2) && extension_present
}

fn image_format_list_available(api_version: u32, extension_present: bool) -> bool {
    let major = vk::api_version_major(api_version);
    let minor = vk::api_version_minor(api_version);
    (major, minor) >= (1, 2) || extension_present
}

/// Block 108 R5: device creation enables `VK_EXT_subgroup_size_control` iff
/// WGSL `subgroups` is supported, the extension is present, and the driver
/// reports `subgroupSizeControl`. Unlike [`subgroup_size_control_supported`]
/// this needs neither `computeFullSubgroups` nor `COMPUTE` in
/// `requiredSubgroupSizeStages`: `ALLOW_VARYING_SUBGROUP_SIZE` only needs
/// `subgroupSizeControl` (Dawn `ComputePipelineVk.cpp` parity).
fn subgroup_size_control_extension_usable(
    subgroups_supported: bool,
    extension_present: bool,
    subgroup_size_control: vk::Bool32,
) -> bool {
    subgroups_supported && extension_present && subgroup_size_control == vk::TRUE
}

/// Subgroup data of one physical device, queried once per [`VulkanAdapter`]
/// so feature enumeration and device creation read the same snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SubgroupQuery {
    /// Stages supporting subgroups, used by the reconvergence decision.
    supported_stages: vk::ShaderStageFlags,
    /// Validated WGSL subgroup size range; `None` when WGSL `subgroups` is
    /// unsupported.
    range: Option<(u32, u32)>,
    /// Whether `VK_EXT_subgroup_size_control` is a device extension.
    extension_present: bool,
    /// `VkPhysicalDeviceSubgroupSizeControlFeatures::subgroupSizeControl`
    /// (`FALSE` when the extension is absent).
    subgroup_size_control: vk::Bool32,
    /// `VkPhysicalDeviceSubgroupSizeControlFeatures::computeFullSubgroups`
    /// (`FALSE` when the extension is absent).
    compute_full_subgroups: vk::Bool32,
    /// `requiredSubgroupSizeStages` (empty when the extension is absent).
    required_subgroup_size_stages: vk::ShaderStageFlags,
    /// `maxComputeWorkgroupSubgroups` (0 when the extension is absent).
    max_compute_workgroup_subgroups: u32,
}

impl SubgroupQuery {
    /// Default required size for compute stages without a WGSL attribute.
    fn default_compute_subgroup_size(&self) -> Option<u32> {
        let (min, max) = self.range?;
        find_default_compute_subgroup_size(
            self.extension_usable(),
            self.required_subgroup_size_stages,
            min,
            max,
            self.max_compute_workgroup_subgroups,
        )
    }

    /// Whether device creation enables `VK_EXT_subgroup_size_control`.
    fn extension_usable(&self) -> bool {
        subgroup_size_control_extension_usable(
            self.range.is_some(),
            self.extension_present,
            self.subgroup_size_control,
        )
    }

    /// The WebGPU `subgroup-size-control` capabilities, `Some` iff the full
    /// Block 108 R2 predicate holds.
    fn caps(&self) -> Option<crate::HalSubgroupSizeControlCaps> {
        if !subgroup_size_control_supported(
            self.range.is_some(),
            self.extension_present,
            self.subgroup_size_control,
            self.compute_full_subgroups,
            self.required_subgroup_size_stages,
        ) {
            return None;
        }
        let (min_size, max_size) = self.range?;
        Some(crate::HalSubgroupSizeControlCaps::new(
            min_size,
            max_size,
            self.max_compute_workgroup_subgroups,
        ))
    }
}

/// Derives the validated WGSL subgroup size range: the size-control
/// min/max when available and non-zero, else the fixed `subgroupSize`.
fn subgroup_size_range_from_properties(
    subgroups_supported: bool,
    size_control_available: bool,
    subgroup_size: u32,
    min_subgroup_size: u32,
    max_subgroup_size: u32,
) -> Option<(u32, u32)> {
    if !subgroups_supported {
        return None;
    }
    let range = if size_control_available && min_subgroup_size != 0 && max_subgroup_size != 0 {
        Some((min_subgroup_size, max_subgroup_size))
    } else if subgroup_size != 0 {
        Some((subgroup_size, subgroup_size))
    } else {
        None
    }?;
    validated_subgroup_size_range(range.0, range.1)
}

fn subgroup_size_control_available(api_version: u32, extension_present: bool) -> bool {
    let major = vk::api_version_major(api_version);
    let minor = vk::api_version_minor(api_version);
    (major, minor) >= (1, 3) || extension_present
}

/// Block 108 R2: WebGPU `subgroup-size-control` needs WGSL `subgroups`,
/// the `VK_EXT_subgroup_size_control` device extension (the Vulkan 1.3 core
/// promotion is unusable at yawgpu's 1.1 API version), and both
/// `subgroupSizeControl` and `computeFullSubgroups` (Dawn
/// `hasComputeFullSubgroups`). Stricter than Dawn, `requiredSubgroupSizeStages`
/// must also contain `COMPUTE`, because a required size on a compute stage is
/// otherwise invalid (VUID-VkPipelineShaderStageCreateInfo-pNext-02755).
/// Implies [`subgroup_size_control_extension_usable`], so an advertising
/// adapter always enables the extension at device creation.
fn subgroup_size_control_supported(
    subgroups_supported: bool,
    extension_present: bool,
    subgroup_size_control: vk::Bool32,
    compute_full_subgroups: vk::Bool32,
    required_subgroup_size_stages: vk::ShaderStageFlags,
) -> bool {
    subgroups_supported
        && extension_present
        && subgroup_size_control == vk::TRUE
        && compute_full_subgroups == vk::TRUE
        && required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE)
}

/// Mirrors every portability-subset feature the physical device reports, so the
/// whole set can be enabled explicitly at device creation. Features the driver
/// does not report stay disabled; without the extension nothing is chained.
fn enabled_portability_subset_features(
    extension_present: bool,
    supported: vk::PhysicalDevicePortabilitySubsetFeaturesKHR<'_>,
) -> EnabledPortabilitySubsetFeatures {
    if !extension_present {
        return EnabledPortabilitySubsetFeatures::default();
    }
    EnabledPortabilitySubsetFeatures {
        enabled: true,
        constant_alpha_color_blend_factors: supported.constant_alpha_color_blend_factors
            == vk::TRUE,
        events: supported.events == vk::TRUE,
        image_view_format_reinterpretation: supported.image_view_format_reinterpretation
            == vk::TRUE,
        image_view_format_swizzle: supported.image_view_format_swizzle == vk::TRUE,
        image_view2_d_on3_d_image: supported.image_view2_d_on3_d_image == vk::TRUE,
        multisample_array_image: supported.multisample_array_image == vk::TRUE,
        mutable_comparison_samplers: supported.mutable_comparison_samplers == vk::TRUE,
        point_polygons: supported.point_polygons == vk::TRUE,
        sampler_mip_lod_bias: supported.sampler_mip_lod_bias == vk::TRUE,
        separate_stencil_mask_ref: supported.separate_stencil_mask_ref == vk::TRUE,
        shader_sample_rate_interpolation_functions: supported
            .shader_sample_rate_interpolation_functions
            == vk::TRUE,
        tessellation_isolines: supported.tessellation_isolines == vk::TRUE,
        tessellation_point_mode: supported.tessellation_point_mode == vk::TRUE,
        triangle_fans: supported.triangle_fans == vk::TRUE,
        vertex_attribute_access_beyond_stride: supported.vertex_attribute_access_beyond_stride
            == vk::TRUE,
    }
}

fn enabled_16bit_storage_features(
    extension_present: bool,
    supported: vk::PhysicalDevice16BitStorageFeatures<'_>,
) -> Enabled16BitStorageFeatures {
    if !extension_present {
        return Enabled16BitStorageFeatures::default();
    }
    let storage_buffer16_bit_access = supported.storage_buffer16_bit_access == vk::TRUE;
    let uniform_and_storage_buffer16_bit_access =
        supported.uniform_and_storage_buffer16_bit_access == vk::TRUE;
    let storage_input_output16 = supported.storage_input_output16 == vk::TRUE;
    let storage_push_constant16 = supported.storage_push_constant16 == vk::TRUE;
    let enabled = storage_buffer16_bit_access
        || uniform_and_storage_buffer16_bit_access
        || storage_input_output16
        || storage_push_constant16;
    if !enabled {
        return Enabled16BitStorageFeatures::default();
    }
    Enabled16BitStorageFeatures {
        enabled,
        storage_buffer16_bit_access,
        uniform_and_storage_buffer16_bit_access,
        storage_input_output16,
        storage_push_constant16,
    }
}

/// Queries driver properties only when the physical device advertises the extension.
fn query_driver_id(
    instance: &Arc<VulkanInstanceInner>,
    physical_device: vk::PhysicalDevice,
) -> Option<u32> {
    query_driver_properties(instance, physical_device)
        .map(|driver| driver.driver_id.as_raw() as u32)
}

fn query_driver_properties(
    instance: &Arc<VulkanInstanceInner>,
    physical_device: vk::PhysicalDevice,
) -> Option<vk::PhysicalDeviceDriverProperties<'static>> {
    let extension_present = has_device_extension_for_physical_device(
        instance,
        physical_device,
        vk::KHR_DRIVER_PROPERTIES_NAME,
    );
    // Dawn's DeviceExt::DriverProperties is extension-name only, with no 1.2 promotion.
    if !extension_present {
        return None;
    }
    let mut driver = vk::PhysicalDeviceDriverProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
    // SAFETY: physical_device belongs to instance; driver is a supported pNext
    // structure and both output structures stay alive throughout the call.
    unsafe {
        instance
            .instance
            .get_physical_device_properties2(physical_device, &mut properties);
    }
    Some(driver)
}

fn adapter_type(device_type: vk::PhysicalDeviceType) -> crate::HalAdapterType {
    match device_type {
        vk::PhysicalDeviceType::INTEGRATED_GPU => crate::HalAdapterType::Integrated,
        vk::PhysicalDeviceType::DISCRETE_GPU | vk::PhysicalDeviceType::VIRTUAL_GPU => {
            crate::HalAdapterType::Discrete
        }
        vk::PhysicalDeviceType::CPU => crate::HalAdapterType::Cpu,
        _ => crate::HalAdapterType::Unknown,
    }
}

fn has_device_extension_for_physical_device(
    instance: &Arc<VulkanInstanceInner>,
    physical_device: vk::PhysicalDevice,
    name: &CStr,
) -> bool {
    let extensions = unsafe {
        instance
            .instance
            .enumerate_device_extension_properties(physical_device)
    };
    let Ok(extensions) = extensions else {
        return false;
    };
    extensions.iter().any(|extension| {
        extension
            .extension_name_as_c_str()
            .is_ok_and(|extension_name| extension_name == name)
    })
}

mod buffer;
mod device;
mod encode;
mod error;
mod format;
mod layout;
mod pipeline;
mod query_set;
mod queue;
mod surface;
use self::buffer::*;
use self::device::*;
use self::encode::*;
use self::error::*;
use self::format::*;
use self::layout::*;
use self::pipeline::*;
use self::query_set::*;
use self::queue::*;
use self::surface::*;
use self::texture::*;
#[cfg(test)]
mod test_helpers;
mod texture;

pub use buffer::VulkanBuffer;
pub use device::VulkanDevice;
pub use pipeline::{VulkanComputePipeline, VulkanRenderPipeline};
pub use query_set::VulkanQuerySet;
pub use queue::VulkanQueue;
pub use surface::VulkanSurface;
pub use texture::{VulkanSampler, VulkanTexture};

#[cfg(test)]
mod tests {
    #[test]
    fn adapter_type_maps_vulkan_hardware_classes() {
        for (input, expected) in [
            (
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                crate::HalAdapterType::Integrated,
            ),
            (
                vk::PhysicalDeviceType::DISCRETE_GPU,
                crate::HalAdapterType::Discrete,
            ),
            (
                vk::PhysicalDeviceType::VIRTUAL_GPU,
                crate::HalAdapterType::Discrete,
            ),
            (vk::PhysicalDeviceType::CPU, crate::HalAdapterType::Cpu),
            (
                vk::PhysicalDeviceType::OTHER,
                crate::HalAdapterType::Unknown,
            ),
            (
                vk::PhysicalDeviceType::from_raw(12345),
                crate::HalAdapterType::Unknown,
            ),
        ] {
            assert_eq!(adapter_type(input), expected);
        }
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU; Block 135 adapter identity"]
    fn adapter_info_matches_physical_device_properties() {
        let Ok(instance) = VulkanInstance::new() else {
            eprintln!("skipping: Vulkan unavailable");
            return;
        };
        let Some(adapter) = instance.enumerate_adapters().into_iter().next() else {
            eprintln!("skipping: no Vulkan adapter");
            return;
        };
        let properties = unsafe {
            adapter
                .instance
                .instance
                .get_physical_device_properties(adapter.physical_device)
        };
        let info = crate::HalAdapter::Vulkan(adapter).info();
        eprintln!("{info:?}");
        assert_eq!(info.vendor_id, properties.vendor_id);
        assert_eq!(info.device_id, properties.device_id);
        assert_eq!(info.adapter_type, adapter_type(properties.device_type));
        assert!(!info.driver_description.is_empty());
    }

    #[test]
    #[ignore = "requires a real Vulkan GPU; Block 131 S2 device-fact logging"]
    fn vulkan_driver_workaround_predicates() {
        let device = test_helpers::vulkan_device();
        let instance = &device.inner._instance;
        let physical = device.inner.physical_device;
        let properties = unsafe { instance.instance.get_physical_device_properties(physical) };
        let driver_id = query_driver_id(instance, physical);
        let predicates = crate::shader_workarounds::VulkanDriverPredicates::new(
            properties.vendor_id,
            properties.device_id,
            driver_id,
            cfg!(target_os = "android"),
            cfg!(target_os = "windows"),
        );
        // Debug includes every R3 predicate, even when none of the gates fires.
        eprintln!(
            "vendorID={:#06x}, deviceID={:#06x}, driverID={driver_id:?}, {predicates:?}",
            properties.vendor_id, properties.device_id
        );
    }

    #[test]
    #[ignore = "manual real-backend test"]
    fn forced_shader_workarounds_reach_vulkan_device() {
        let _lock = crate::shader_workarounds::FORCE_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var(
            "YAWGPU_FORCE_SHADER_WORKAROUNDS",
            "replace_unsigned_compare_zero",
        );
        let device = test_helpers::vulkan_device_or_skip();
        std::env::remove_var("YAWGPU_FORCE_SHADER_WORKAROUNDS");
        let Some(device) = device else { return };
        let device = crate::HalDevice::Vulkan(device);
        assert!(device.shader_workarounds().replace_unsigned_compare_zero);
    }

    #[test]
    fn robustness2_decisions_cover_all_device_facts() {
        for (enabled, supported, expected) in [
            (false, false, false),
            (false, true, false),
            (true, false, false),
            (true, true, true),
        ] {
            assert_eq!(use_image_robust_access2(enabled, supported), expected);
            for (robust_buffer_access, indexing, update_after_bind, expected_buffer) in [
                (true, false, false, expected),
                (true, false, true, expected),
                (true, true, false, false),
                (true, true, true, expected),
                (false, false, false, false),
                (false, false, true, false),
                (false, true, false, false),
                (false, true, true, false),
            ] {
                assert_eq!(
                    use_buffer_robust_access2(
                        enabled,
                        supported,
                        robust_buffer_access,
                        indexing,
                        update_after_bind,
                    ),
                    expected_buffer,
                );
            }
        }
    }

    #[test]
    fn descriptor_indexing_availability_requires_advertised_extension() {
        for version in [
            vk::API_VERSION_1_1,
            vk::API_VERSION_1_2,
            vk::API_VERSION_1_3,
        ] {
            assert!(!descriptor_indexing_available(version, false));
            assert!(descriptor_indexing_available(version, true));
        }
    }

    #[test]
    #[ignore = "requires a real Vulkan device and validation layer; S2 manual run"]
    fn vulkan_robustness2_decisions_and_render_pipeline() {
        let device = test_helpers::vulkan_device();
        let inner = &device.inner;
        let instance = &inner._instance;
        let physical = inner.physical_device;
        let robustness2 =
            has_device_extension_for_physical_device(instance, physical, vk::EXT_ROBUSTNESS2_NAME);
        let pipeline_extension = has_device_extension_for_physical_device(
            instance,
            physical,
            vk::EXT_PIPELINE_ROBUSTNESS_NAME,
        );
        let mut robust = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        let mut pipeline = vk::PhysicalDevicePipelineRobustnessFeaturesEXT::default();
        let mut features = vk::PhysicalDeviceFeatures2::default();
        if robustness2 {
            features = features.push_next(&mut robust);
        }
        if pipeline_extension {
            features = features.push_next(&mut pipeline);
        }
        unsafe {
            instance
                .instance
                .get_physical_device_features2(physical, &mut features);
        }
        let robust_buffer_access_enabled = features.features.robust_buffer_access == vk::TRUE;
        let properties = unsafe { instance.instance.get_physical_device_properties(physical) };
        let indexing = descriptor_indexing_available(
            properties.api_version,
            has_device_extension_for_physical_device(
                instance,
                physical,
                vk::EXT_DESCRIPTOR_INDEXING_NAME,
            ),
        );
        let mut indexing_properties = vk::PhysicalDeviceDescriptorIndexingProperties::default();
        if indexing {
            let mut properties2 =
                vk::PhysicalDeviceProperties2::default().push_next(&mut indexing_properties);
            unsafe {
                instance
                    .instance
                    .get_physical_device_properties2(physical, &mut properties2);
            }
        }
        let flags = device.shader_workarounds();
        eprintln!("robustness2={robustness2}, robustBufferAccess={robust_buffer_access_enabled}, robustBufferAccess2={}, robustImageAccess2={}, nullDescriptor={}, descriptorIndexing={indexing}, robustBufferAccessUpdateAfterBind={}, pipelineRobustness={}, imageDecision={}, bufferDecision={}",
            robust.robust_buffer_access2, robust.robust_image_access2, robust.null_descriptor,
            indexing_properties.robust_buffer_access_update_after_bind, inner.pipeline_robustness,
            flags.disable_image_robustness, flags.disable_storage_runtime_sized_array_index_clamping);
        assert_eq!(
            flags.disable_image_robustness,
            use_image_robust_access2(robustness2, robust.robust_image_access2 == vk::TRUE,)
        );
        assert_eq!(
            flags.disable_storage_runtime_sized_array_index_clamping,
            use_buffer_robust_access2(
                robustness2,
                robust.robust_buffer_access2 == vk::TRUE,
                robust_buffer_access_enabled,
                indexing,
                indexing_properties.robust_buffer_access_update_after_bind == vk::TRUE,
            )
        );
        assert_eq!(
            inner.pipeline_robustness,
            pipeline_extension && pipeline.pipeline_robustness == vk::TRUE
        );
        let source = "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); } @fragment fn fs() -> @location(0) vec4f { return vec4f(1); }";
        let shader = crate::HalShaderSource::SpirVStages {
            vertex: compile_writer_test_spirv(&device, source, "vs"),
            fragment: Some(compile_writer_test_spirv(&device, source, "fs")),
        };
        device
            .create_render_pipeline(
                shader,
                "vs",
                Some("fs"),
                &test_helpers::render_descriptor(),
                &[],
            )
            .expect("create render pipeline with enabled robustness state");
    }

    #[test]
    fn vulkan_writer_flags_preserve_workarounds_and_map_enabled_features() {
        let workarounds = crate::HalShaderWorkarounds {
            replace_unsigned_compare_zero: true,
            polyfill_bool_vec_dynamic_store: true,
            disable_image_robustness: true,
            disable_storage_runtime_sized_array_index_clamping: true,
            // Stale writer flags must be replaced by the enabled-device facts.
            use_demote_to_helper_invocation: true,
            native_dot_4x8_packed: true,
            use_zero_initialize_workgroup_memory: true,
            use_maximal_reconvergence: true,
            use_subgroup_uniform_control_flow: true,
            ..Default::default()
        };
        for enabled in [
            [false, false, false],
            [true, false, false],
            [false, true, false],
            [false, false, true],
            [true, true, false],
            [true, false, true],
            [false, true, true],
            [true, true, true],
        ] {
            let flags = vulkan_writer_flags(
                workarounds,
                enabled[0],
                enabled[1],
                enabled[2],
                Reconvergence::None,
                enabled[0],
                enabled[1],
            );
            assert_eq!(flags.disable_image_robustness, enabled[0]);
            assert_eq!(
                flags.disable_storage_runtime_sized_array_index_clamping,
                enabled[1]
            );
            assert!(!flags.use_maximal_reconvergence);
            assert!(!flags.use_subgroup_uniform_control_flow);
            assert_eq!(
                [
                    flags.use_demote_to_helper_invocation,
                    flags.native_dot_4x8_packed,
                    flags.use_zero_initialize_workgroup_memory
                ],
                enabled
            );
            assert!(flags.replace_unsigned_compare_zero);
            assert!(flags.polyfill_bool_vec_dynamic_store);
        }
    }

    #[test]
    fn vulkan_reconvergence_device_features_enable_independently() {
        let present = (true, vk::TRUE);
        let absent = (false, vk::FALSE);
        for (maximal, uniform, expected) in [
            (present, present, (true, true)),
            (present, absent, (true, false)),
            (absent, present, (false, true)),
            (absent, absent, (false, false)),
            ((true, vk::FALSE), present, (false, true)),
            ((false, vk::TRUE), present, (false, true)),
            (present, (true, vk::FALSE), (true, false)),
            (present, (false, vk::TRUE), (true, false)),
        ] {
            let enabled =
                enabled_reconvergence_features(maximal.0, maximal.1, uniform.0, uniform.1);
            assert_eq!((enabled.maximal, enabled.uniform_control_flow), expected);
            // The Android/Imagination restriction applies only to Tint's mode.
            assert_eq!(
                select_reconvergence(
                    enabled,
                    vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT,
                    true,
                    0x1010
                ),
                Reconvergence::None
            );
        }
    }

    #[test]
    fn vulkan_reconvergence_selection_matches_dawn_branches() {
        use Reconvergence::{Maximal, None as Disabled, UniformControlFlow as Uniform};
        let both = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT;
        let present = (true, vk::TRUE);
        let absent = (false, vk::FALSE);
        for (maximal, uniform, expected) in [
            (present, present, Maximal),
            (present, absent, Maximal),
            (absent, present, Uniform),
            ((true, vk::FALSE), present, Uniform),
            ((false, vk::TRUE), present, Uniform),
            (absent, (true, vk::FALSE), Disabled),
            (absent, (false, vk::TRUE), Disabled),
            (absent, absent, Disabled),
        ] {
            assert_eq!(
                select_reconvergence(
                    enabled_reconvergence_features(maximal.0, maximal.1, uniform.0, uniform.1),
                    both,
                    false,
                    0x8086
                ),
                expected,
                "{maximal:?}/{uniform:?}"
            );
        }
        let enabled_both = ReconvergenceFeatures {
            maximal: true,
            uniform_control_flow: true,
        };
        let uniform_only = ReconvergenceFeatures {
            maximal: false,
            uniform_control_flow: true,
        };
        for (stages, expected) in [
            (vk::ShaderStageFlags::empty(), Disabled),
            (vk::ShaderStageFlags::COMPUTE, Disabled),
            (vk::ShaderStageFlags::FRAGMENT, Disabled),
            (both, Uniform),
            (both | vk::ShaderStageFlags::VERTEX, Uniform),
        ] {
            assert_eq!(
                select_reconvergence(uniform_only, stages, false, 0x8086),
                expected
            );
            // Maximal reconvergence does not need the uniform-flow stage check.
            assert_eq!(
                select_reconvergence(enabled_both, stages, false, 0x8086),
                Maximal
            );
        }
        // Android/Imagination excludes both Tint modes, leaving device features enabled.
        for (android, vendor, maximal, uniform) in [
            (true, 0x1010, Disabled, Disabled),
            (false, 0x1010, Maximal, Uniform),
            (true, 0x8086, Maximal, Uniform),
            (false, 0x8086, Maximal, Uniform),
        ] {
            assert_eq!(
                select_reconvergence(enabled_both, both, android, vendor),
                maximal
            );
            assert_eq!(
                select_reconvergence(uniform_only, both, android, vendor),
                uniform
            );
        }
        for (selected, maximal, uniform) in [
            (Disabled, false, false),
            (Maximal, true, false),
            (Uniform, false, true),
        ] {
            let flags = vulkan_writer_flags(
                Default::default(),
                false,
                false,
                false,
                selected,
                false,
                false,
            );
            assert_eq!(flags.use_maximal_reconvergence, maximal);
            assert_eq!(flags.use_subgroup_uniform_control_flow, uniform);
        }
    }

    #[test]
    fn vulkan_default_compute_subgroup_size_matches_dawn_heuristic() {
        let compute = vk::ShaderStageFlags::COMPUTE;
        assert_eq!(
            find_default_compute_subgroup_size(false, compute, 8, 32, 32),
            None
        );
        assert_eq!(
            find_default_compute_subgroup_size(true, compute, 32, 32, 32),
            None
        );
        assert_eq!(
            find_default_compute_subgroup_size(true, compute, 8, 32, 32),
            Some(16)
        );
        assert_eq!(
            find_default_compute_subgroup_size(true, compute, 8, 16, 32),
            Some(16)
        );
        assert_eq!(
            find_default_compute_subgroup_size(true, compute, 8, 12, 32),
            Some(8)
        );
        assert_eq!(
            find_default_compute_subgroup_size(true, compute, u32::MAX - 1, u32::MAX, 32),
            Some(u32::MAX - 1)
        );
        let query = full_subgroup_query();
        assert_eq!(query.default_compute_subgroup_size(), Some(32));
        // Defaults do not require computeFullSubgroups, unlike explicit WGSL sizes.
        assert_eq!(
            SubgroupQuery {
                compute_full_subgroups: vk::FALSE,
                ..query
            }
            .default_compute_subgroup_size(),
            Some(32)
        );
        assert_eq!(
            SubgroupQuery {
                extension_present: false,
                ..query
            }
            .default_compute_subgroup_size(),
            None
        );
    }

    #[test]
    fn vulkan_default_compute_subgroup_size_respects_core_invocation_floor() {
        let compute = vk::ShaderStageFlags::COMPUTE;
        assert_eq!(
            find_default_compute_subgroup_size(true, compute, 8, 32, 15),
            None
        );
        assert_eq!(
            find_default_compute_subgroup_size(true, compute, 8, 32, 16),
            Some(16)
        );
        let query = SubgroupQuery {
            range: Some((8, 32)),
            max_compute_workgroup_subgroups: 15,
            ..full_subgroup_query()
        };
        let default_size = query.default_compute_subgroup_size();
        assert_eq!(default_size, None);
        assert_eq!(
            clamp_compute_invocations_per_workgroup(
                1024,
                query.max_compute_workgroup_subgroups,
                default_size
            ),
            1024
        );
    }

    #[test]
    fn vulkan_default_compute_subgroup_size_requires_enabled_compute_support() {
        for (enabled, stages) in [
            (false, vk::ShaderStageFlags::COMPUTE),
            (true, vk::ShaderStageFlags::empty()),
            (true, vk::ShaderStageFlags::FRAGMENT),
        ] {
            assert_eq!(
                find_default_compute_subgroup_size(enabled, stages, 8, 32, 32),
                None
            );
        }
        for query in [
            SubgroupQuery {
                subgroup_size_control: vk::FALSE,
                ..full_subgroup_query()
            },
            SubgroupQuery {
                required_subgroup_size_stages: vk::ShaderStageFlags::empty(),
                ..full_subgroup_query()
            },
            SubgroupQuery {
                required_subgroup_size_stages: vk::ShaderStageFlags::FRAGMENT,
                ..full_subgroup_query()
            },
        ] {
            let default_size = query.default_compute_subgroup_size();
            assert_eq!(default_size, None);
            assert_eq!(
                clamp_compute_invocations_per_workgroup(
                    1024,
                    query.max_compute_workgroup_subgroups,
                    default_size,
                ),
                1024
            );
        }
    }

    #[test]
    fn vulkan_compute_invocations_clamp_uses_default_subgroup_size() {
        assert_eq!(
            clamp_compute_invocations_per_workgroup(1024, 32, Some(16)),
            512
        );
        assert_eq!(
            clamp_compute_invocations_per_workgroup(256, 32, Some(16)),
            256
        );
        assert_eq!(
            clamp_compute_invocations_per_workgroup(1024, 32, None),
            1024
        );
        assert_eq!(
            clamp_compute_invocations_per_workgroup(1024, u32::MAX, Some(16)),
            1024
        );
    }

    #[test]
    fn vulkan_writer_native_features_require_availability_and_feature_bits() {
        // A reported feature alone is insufficient at the Vulkan 1.1 baseline.
        assert!(!writer_feature_available(vk::API_VERSION_1_1, false));
        assert!(!writer_feature_enabled(
            vk::API_VERSION_1_1,
            false,
            vk::TRUE
        ));
        // R1 demote and R2 integer dot use the same promoted-feature gate.
        for (api, extension, feature, expected) in [
            (vk::API_VERSION_1_1, false, vk::TRUE, false),
            (vk::API_VERSION_1_1, true, vk::TRUE, true),
            (vk::API_VERSION_1_1, true, vk::FALSE, false),
            (vk::API_VERSION_1_2, false, vk::TRUE, false),
            (vk::API_VERSION_1_2, true, vk::TRUE, true),
            (vk::API_VERSION_1_3, false, vk::TRUE, true),
            (vk::API_VERSION_1_3, false, vk::FALSE, false),
        ] {
            assert_eq!(writer_feature_enabled(api, extension, feature), expected);
        }
        for (api, extension, feature, vendor, expected) in [
            (vk::API_VERSION_1_1, false, vk::TRUE, 0x8086, false),
            (vk::API_VERSION_1_1, true, vk::FALSE, 0x8086, false),
            (vk::API_VERSION_1_1, true, vk::TRUE, 0x8086, true),
            (vk::API_VERSION_1_1, true, vk::TRUE, 0x13b5, false),
            (vk::API_VERSION_1_1, true, vk::TRUE, 0x1010, false),
            (vk::API_VERSION_1_3, false, vk::TRUE, 0x10de, true),
            (vk::API_VERSION_1_3, false, vk::TRUE, 0x13b5, false),
            (vk::API_VERSION_1_3, false, vk::TRUE, 0x1010, false),
        ] {
            assert_eq!(
                zero_initialize_workgroup_memory_enabled(api, extension, feature, vendor),
                expected
            );
        }
    }

    #[test]
    fn vulkan_memory_model_requires_model_and_device_scope() {
        for (available, model, scope, expected) in [
            (true, vk::TRUE, vk::FALSE, false),
            (true, vk::TRUE, vk::TRUE, true),
            (true, vk::FALSE, vk::TRUE, false),
            (true, vk::FALSE, vk::FALSE, false),
            (false, vk::TRUE, vk::TRUE, false),
            (false, vk::TRUE, vk::FALSE, false),
            (false, vk::FALSE, vk::TRUE, false),
            (false, vk::FALSE, vk::FALSE, false),
        ] {
            assert_eq!(
                vulkan_memory_model_enabled(available, model, scope),
                expected
            );
        }
    }

    #[test]
    #[ignore = "requires a real Vulkan device and validation layer; S2 manual run"]
    fn vulkan_writer_features_report_enabled_device_features() {
        let instance = VulkanInstance::new().expect("create Vulkan instance");
        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Vulkan adapter");
        let device = adapter.create_device().expect("create Vulkan device");
        let inner = &device.inner;
        let flags = device.shader_workarounds();
        eprintln!(
            "enabled Vulkan writer features: {flags:?}, memory model: {}",
            inner.vulkan_memory_model
        );
        let properties = unsafe {
            inner
                ._instance
                .instance
                .get_physical_device_properties(inner.physical_device)
        };
        let api = YAWGPU_VULKAN_API_VERSION.min(properties.api_version);
        let demote_available = writer_feature_available(
            api,
            adapter.has_device_extension(vk::EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_NAME),
        );
        let dot_available = writer_feature_available(
            api,
            adapter.has_device_extension(vk::KHR_SHADER_INTEGER_DOT_PRODUCT_NAME),
        );
        let zero_available = writer_feature_available(
            api,
            adapter.has_device_extension(vk::KHR_ZERO_INITIALIZE_WORKGROUP_MEMORY_NAME),
        );
        let model_available = vulkan_memory_model_available(
            api,
            adapter.has_device_extension(vk::KHR_VULKAN_MEMORY_MODEL_NAME),
        );
        let mut demote = vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures::default();
        let mut dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
        let mut zero = vk::PhysicalDeviceZeroInitializeWorkgroupMemoryFeatures::default();
        let mut model = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
        let mut features = vk::PhysicalDeviceFeatures2::default();
        if demote_available {
            features = features.push_next(&mut demote);
        }
        if dot_available {
            features = features.push_next(&mut dot);
        }
        if zero_available {
            features = features.push_next(&mut zero);
        }
        if model_available {
            features = features.push_next(&mut model);
        }
        unsafe {
            inner
                ._instance
                .instance
                .get_physical_device_features2(inner.physical_device, &mut features);
        }
        assert_eq!(
            flags.use_demote_to_helper_invocation,
            inner.shader_demote_to_helper_invocation
        );
        assert_eq!(
            flags.use_demote_to_helper_invocation,
            demote_available && demote.shader_demote_to_helper_invocation == vk::TRUE
        );
        assert_eq!(
            flags.native_dot_4x8_packed,
            dot_available && dot.shader_integer_dot_product == vk::TRUE
        );
        assert_eq!(
            flags.use_zero_initialize_workgroup_memory,
            zero_available
                && zero.shader_zero_initialize_workgroup_memory == vk::TRUE
                && !matches!(properties.vendor_id, 0x13b5 | 0x1010)
        );
        assert_eq!(
            inner.vulkan_memory_model,
            model_available
                && model.vulkan_memory_model == vk::TRUE
                && model.vulkan_memory_model_device_scope == vk::TRUE
        );
    }

    #[test]
    fn spirv_1_4_extensions_follow_availability() {
        for spirv in [false, true] {
            for controls in [false, true] {
                let enabled = spirv && controls;
                let mut names = vec![vk::KHR_SWAPCHAIN_NAME.as_ptr()];
                append_spirv_1_4_extensions(&mut names, spirv, controls);
                let names = extension_names_from_pointers(&names);
                assert_eq!(names.contains(&vk::KHR_SPIRV_1_4_NAME), enabled);
                assert_eq!(names.contains(&vk::KHR_SHADER_FLOAT_CONTROLS_NAME), enabled);
                assert_eq!(names.len(), if enabled { 3 } else { 1 });
                assert_eq!(names[0], vk::KHR_SWAPCHAIN_NAME);
            }
        }
        for (android, vendor, device) in [(true, 0x13b5, 0), (false, 0x5143, 0x07000001)] {
            let decision =
                crate::shader_workarounds::use_spirv_1_4(true, true, android, vendor, device);
            let mut names = Vec::new();
            append_spirv_1_4_extensions(&mut names, true, true);
            let names = extension_names_from_pointers(&names);
            assert_eq!(
                names,
                [vk::KHR_SPIRV_1_4_NAME, vk::KHR_SHADER_FLOAT_CONTROLS_NAME]
            );
            assert!(!decision);
        }
    }

    #[test]
    #[ignore = "manual real-backend test"]
    fn vulkan_spirv_1_4_decision_and_pipelines() {
        let Some(device) = test_helpers::vulkan_device_or_skip() else {
            return;
        };
        let inner = &device.inner;
        let properties = unsafe {
            inner
                ._instance
                .instance
                .get_physical_device_properties(inner.physical_device)
        };
        let has_extension = |name| {
            has_device_extension_for_physical_device(&inner._instance, inner.physical_device, name)
        };
        let spirv_extension = has_extension(vk::KHR_SPIRV_1_4_NAME);
        let float_controls_extension = has_extension(vk::KHR_SHADER_FLOAT_CONTROLS_NAME);
        let expected = crate::shader_workarounds::use_spirv_1_4(
            spirv_extension,
            float_controls_extension,
            cfg!(target_os = "android"),
            properties.vendor_id,
            properties.device_id,
        );
        let decision = device.shader_workarounds().use_spirv_1_4;
        let [spirv_enabled, controls_enabled] = inner.spirv_1_4_extensions_enabled;
        eprintln!("SPIR-V 1.4: vendorID={:#x}, deviceID={:#x}, decision={decision}, VK_KHR_spirv_1_4 enabled={spirv_enabled}, VK_KHR_shader_float_controls enabled={controls_enabled}", properties.vendor_id, properties.device_id);
        assert_eq!(decision, expected);
        assert_eq!(
            [spirv_enabled, controls_enabled],
            [spirv_extension && float_controls_extension; 2]
        );
        let source = "@compute @workgroup_size(1) fn cs() {} @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0, 0, 0, 1); } @fragment fn fs() -> @location(0) vec4f { return vec4f(1); }";
        let compile = |entry| {
            let words = compile_writer_test_spirv(&device, source, entry);
            assert_eq!(words[1], if decision { 0x00010400 } else { 0x00010300 });
            words
        };
        device
            .create_compute_pipeline(
                crate::HalShaderSource::SpirV(compile("cs")),
                "cs",
                (1, 1, 1),
                &[],
                0,
                None,
            )
            .expect("create compute pipeline with selected SPIR-V version");
        device
            .create_render_pipeline(
                crate::HalShaderSource::SpirVStages {
                    vertex: compile("vs"),
                    fragment: Some(compile("fs")),
                },
                "vs",
                Some("fs"),
                &test_helpers::render_descriptor(),
                &[],
            )
            .expect("create render pipeline with selected SPIR-V version");
    }

    fn compile_writer_test_spirv(device: &VulkanDevice, source: &str, entry: &str) -> Vec<u32> {
        let flags = device.shader_workarounds();
        let mut options = yawgpu_tint::SpirvWorkarounds::default();
        options.use_spirv_1_4 = flags.use_spirv_1_4;
        options.use_demote_to_helper_invocation = flags.use_demote_to_helper_invocation;
        options.native_dot_4x8_packed = flags.native_dot_4x8_packed;
        options.disable_image_robustness = flags.disable_image_robustness;
        options.disable_storage_runtime_sized_array_index_clamping =
            flags.disable_storage_runtime_sized_array_index_clamping;
        options.use_zero_initialize_workgroup_memory = flags.use_zero_initialize_workgroup_memory;
        yawgpu_tint::Program::parse(source, false, false, false, false, false, false, &[])
            .expect("parse writer test WGSL")
            .generate_spirv(
                entry,
                &yawgpu_tint::Bindings::default(),
                &[],
                true,
                device.vulkan_memory_model(),
                0,
                false,
                None,
                false,
                0,
                device.storage_input_output16(),
                false,
                options,
            )
            .expect("compile writer test SPIR-V")
    }

    #[test]
    #[ignore = "requires a real Vulkan device and validation layer; S2 manual run"]
    fn vulkan_dot4_u8_packed_compiles_and_runs() {
        use crate::{
            HalBoundBuffer, HalBuffer, HalBufferBindingKind, HalBufferUsage, HalComputeDispatch,
            HalComputePass, HalComputePipeline, HalCopy, HalDescriptorBinding,
            HalDescriptorBindingKind, HalShaderSource,
        };
        let device = test_helpers::vulkan_device();
        let source = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32, 3>;
@compute @workgroup_size(1) fn cs() { data[0] = dot4U8Packed(data[1], data[2]); }
"#;
        let spirv = compile_writer_test_spirv(&device, source, "cs");
        let pipeline = device
            .create_compute_pipeline(
                HalShaderSource::SpirV(spirv),
                "cs",
                (1, 1, 1),
                &[HalDescriptorBinding {
                    group: 0,
                    binding: 0,
                    kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage),
                }],
                0,
                None,
            )
            .unwrap();
        let buffer = device
            .create_buffer(
                12,
                HalBufferUsage {
                    storage: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let input: Vec<u8> = [0u32, 0x04030201, 0x01010101]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        buffer.write(0, &input).unwrap();
        let pass = HalComputePass {
            pipeline: HalComputePipeline::Vulkan(pipeline),
            bind_buffers: vec![HalBoundBuffer {
                group: 0,
                binding: 0,
                metal_index: 0,
                vertex_metal_index: None,
                fragment_metal_index: None,
                buffer: HalBuffer::Vulkan(buffer.clone()),
                offset: 0,
                size: 12,
            }],
            bind_textures: Vec::new(),
            bind_samplers: Vec::new(),
            bind_external_textures: Vec::new(),
            immediate_data: Vec::new(),
            dispatch: HalComputeDispatch::Direct {
                workgroups: (1, 1, 1),
            },
        };
        let submission = device
            .queue()
            .submit_copies(&[HalCopy::ComputePass(pass)])
            .unwrap();
        device.queue().wait_for_submission(submission).unwrap();
        assert_eq!(buffer.read(0, 4).unwrap(), 10u32.to_le_bytes());
    }

    #[test]
    #[ignore = "requires a real Vulkan device and validation layer; S2 manual run"]
    fn vulkan_discard_then_derivative_compiles_and_runs() {
        use crate::{
            HalBuffer, HalBufferTextureCopy, HalBufferTextureLayout, HalBufferUsage, HalCopy,
            HalExtent3d, HalOrigin3d, HalRenderColorTarget, HalRenderLoadOp, HalRenderPassCommand,
            HalRenderPassCommandStream, HalRenderPipeline, HalShaderSource, HalTexture,
            HalTextureAspect, HalTextureFormat,
        };
        let device = test_helpers::vulkan_device();
        let source = r#"
diagnostic(off, derivative_uniformity);
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(p[i], 0, 1);
}
@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
    if (p.x < 2.0) { discard; }
    return vec4f(dpdx(p.x));
}
"#;
        let shader = HalShaderSource::SpirVStages {
            vertex: compile_writer_test_spirv(&device, source, "vs"),
            fragment: Some(compile_writer_test_spirv(&device, source, "fs")),
        };
        let pipeline = device
            .create_render_pipeline(
                shader,
                "vs",
                Some("fs"),
                &test_helpers::render_descriptor(),
                &[],
            )
            .unwrap();
        let texture = device
            .create_texture(&test_helpers::texture_descriptor())
            .unwrap();
        let pass = HalRenderPassCommandStream {
            color_targets: vec![Some(HalRenderColorTarget {
                texture: HalTexture::Vulkan(texture.clone()),
                view_format: crate::HalTextureFormat::Rgba8Unorm,
                resolve_target: None,
                resolve_view_format: None,
                mip_level: 0,
                array_layer: 0,
                depth_slice: 0,
                resolve_mip_level: 0,
                resolve_array_layer: 0,
                load_op: HalRenderLoadOp::Clear,
                store: true,
                clear_color: [0.0; 4],
            })],
            commands: vec![
                HalRenderPassCommand::SetPipeline(HalRenderPipeline::Vulkan(pipeline)),
                HalRenderPassCommand::Draw {
                    vertex_count: 3,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
            ],
            ..Default::default()
        };
        let readback = device
            .create_buffer(
                8,
                HalBufferUsage {
                    copy_dst: true,
                    map_read: true,
                    ..Default::default()
                },
            )
            .unwrap();
        readback.write(0, &[0x7f; 8]).unwrap();
        // Adjacent pixels at x=1 and x=2 straddle the p.x < 2.0 discard boundary.
        let copy = HalCopy::TextureToBuffer(HalBufferTextureCopy {
            buffer: HalBuffer::Vulkan(readback.clone()),
            buffer_layout: HalBufferTextureLayout {
                offset: 0,
                bytes_per_row: 8,
                rows_per_image: 1,
            },
            texture: HalTexture::Vulkan(texture),
            format: HalTextureFormat::Rgba8Unorm,
            aspect: HalTextureAspect::All,
            mip_level: 0,
            origin: HalOrigin3d { x: 1, y: 1, z: 0 },
            extent: HalExtent3d {
                width: 2,
                height: 1,
                depth_or_array_layers: 1,
            },
        });
        let submission = device
            .queue()
            .submit_copies(&[HalCopy::RenderPassCommandStream(pass), copy])
            .unwrap();
        device.queue().wait_for_submission(submission).unwrap();
        let pixels = readback.read(0, 8).unwrap();
        assert_eq!(
            &pixels[..4],
            &[0, 0, 0, 0],
            "discarded pixel must keep the clear value"
        );
        assert_eq!(
            &pixels[4..],
            &[255, 255, 255, 255],
            "dpdx(p.x) must shade the surviving pixel white"
        );
    }

    #[test]
    fn audit_e11_only_cpu_physical_devices_are_fallback_adapters() {
        assert!(is_fallback_device_type(vk::PhysicalDeviceType::CPU));
        for kind in [
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            vk::PhysicalDeviceType::DISCRETE_GPU,
            vk::PhysicalDeviceType::VIRTUAL_GPU,
            vk::PhysicalDeviceType::OTHER,
        ] {
            assert!(!is_fallback_device_type(kind));
        }
    }

    use super::*;

    #[test]
    fn hal_limits_from_vk_caps_nvidia_storage_bindings() {
        let cap = (1_u64 << 31) - 4;
        for (vendor_id, range, buffer_size, expected) in [
            (0x10de, u32::MAX, u64::from(u32::MAX), cap),
            (
                0x1002,
                u32::MAX,
                u64::from(u32::MAX),
                u64::from(u32::MAX) - 3,
            ),
            (0x10de, u32::MAX, 1 << 30, 1 << 30),
            (0x10de, u32::MAX, (1 << 30) + 3, 1 << 30),
            (0x1002, u32::MAX, (1 << 30) + 3, 1 << 30),
            (0x10de, 1 << 29, 1 << 30, 1 << 29),
        ] {
            let raw = vk::PhysicalDeviceLimits {
                max_storage_buffer_range: range,
                ..Default::default()
            };
            let limits = hal_limits_from_vk(raw, buffer_size, vendor_id);
            assert_eq!(limits.max_storage_buffer_binding_size, expected);
        }
    }

    #[test]
    fn hal_limits_from_vk_caps_uniform_bindings_to_buffer_size() {
        let raw = vk::PhysicalDeviceLimits {
            max_uniform_buffer_range: 65536,
            ..Default::default()
        };
        for vendor_id in [0x10de, 0x1002] {
            let limits = hal_limits_from_vk(raw, 32771, vendor_id);
            assert_eq!(limits.max_uniform_buffer_binding_size, 32771);
        }
    }

    #[test]
    fn hal_limits_from_vk_saturates_color_attachment_bytes_per_sample() {
        let raw = vk::PhysicalDeviceLimits {
            max_color_attachments: u32::MAX,
            ..Default::default()
        };
        let limits = hal_limits_from_vk(raw, ASSUMED_MAX_BUFFER_SIZE, 0);
        assert_eq!(limits.max_color_attachment_bytes_per_sample, u32::MAX);
        assert_eq!(limits.max_color_attachments, 8);
    }

    #[test]
    fn hal_limits_from_vk_ignores_nonbinding_combined_output_limits() {
        // The already-clamped maximum is 16 + 8 + 8 = 32; the default is 20.
        for combined in [0, 19, 20, 32, 33, u32::MAX] {
            let raw = vk::PhysicalDeviceLimits {
                max_color_attachments: 16,
                max_per_stage_descriptor_storage_images: 32,
                max_per_stage_descriptor_storage_buffers: 32,
                max_fragment_combined_output_resources: combined,
                ..Default::default()
            };
            let limits = hal_limits_from_vk(raw, ASSUMED_MAX_BUFFER_SIZE, 0);
            assert_eq!(limits.max_color_attachments, 8);
            assert_eq!(limits.max_storage_textures_per_shader_stage, 8);
            assert_eq!(limits.max_storage_buffers_per_shader_stage, 16);
            assert_eq!(limits.max_color_attachment_bytes_per_sample, 512);
        }
    }

    #[test]
    fn hal_limits_from_vk_redistributes_combined_outputs_in_dawn_order() {
        // Each raw clamp is reached, and the one-extra case checks rounding.
        for (colors, textures, buffers, combined, expected) in [
            (8, 8, 16, 21, (8, 4, 9)),
            (8, 8, 16, 26, (8, 7, 11)),
            (9, 8, 16, 30, (8, 8, 13)),
            (8, 5, 16, 28, (8, 5, 15)),
            (8, 8, 9, 24, (8, 6, 9)),
            (16, 32, 32, 26, (8, 6, 10)),
            // Redistribution produces 9 textures; only the final clamp lowers it to 8.
            (8, 32, 32, 31, (8, 8, 14)),
        ] {
            let raw = vk::PhysicalDeviceLimits {
                max_color_attachments: colors,
                max_per_stage_descriptor_storage_images: textures,
                max_per_stage_descriptor_storage_buffers: buffers,
                max_fragment_combined_output_resources: combined,
                ..Default::default()
            };
            let limits = hal_limits_from_vk(raw, ASSUMED_MAX_BUFFER_SIZE, 0);
            assert_eq!(
                (
                    limits.max_color_attachments,
                    limits.max_storage_textures_per_shader_stage,
                    limits.max_storage_buffers_per_shader_stage,
                ),
                expected,
                "raw=({colors}, {textures}, {buffers}), combined={combined}"
            );
            assert_eq!(limits.max_color_attachment_bytes_per_sample, 32 * colors);
        }
    }

    #[test]
    #[ignore = "manual real-backend test"]
    fn vulkan_adapter_limits_match_raw_storage_color_and_combined_outputs() {
        let instance = VulkanInstance::new().expect("Vulkan instance");
        let adapters = instance.enumerate_adapters();
        assert!(!adapters.is_empty(), "no Vulkan adapter");
        for adapter in adapters {
            let properties = unsafe {
                adapter
                    .instance
                    .instance
                    .get_physical_device_properties(adapter.physical_device)
            };
            let raw = properties.limits;
            let limits = adapter.limits();
            eprintln!(
                "{}: storageBindingSize={}, colorBytesPerSample={}, combinedOutputs={} (colors={}, storageTextures={}, storageBuffers={})",
                adapter.name(),
                limits.max_storage_buffer_binding_size,
                limits.max_color_attachment_bytes_per_sample,
                raw.max_fragment_combined_output_resources,
                limits.max_color_attachments,
                limits.max_storage_textures_per_shader_stage,
                limits.max_storage_buffers_per_shader_stage,
            );
            let expected_storage = if properties.vendor_id == 0x10de {
                u64::from(raw.max_storage_buffer_range)
                    .min(adapter.max_buffer_size())
                    .min((1_u64 << 31) - 4)
            } else {
                u64::from(raw.max_storage_buffer_range)
            };
            let expected_storage = expected_storage.min(adapter.max_buffer_size()) / 4 * 4;
            assert_eq!(limits.max_storage_buffer_binding_size, expected_storage);
            assert_eq!(
                limits.max_uniform_buffer_binding_size,
                u64::from(raw.max_uniform_buffer_range / 16 * 16).min(adapter.max_buffer_size())
            );
            assert_eq!(
                limits.max_color_attachment_bytes_per_sample,
                32_u32.saturating_mul(raw.max_color_attachments)
            );

            let base = HalLimits::DEFAULT;
            let minimum = base
                .max_color_attachments
                .saturating_add(base.max_storage_textures_per_shader_stage)
                .saturating_add(base.max_storage_buffers_per_shader_stage);
            let maximum = raw
                .max_color_attachments
                .min(8)
                .saturating_add(raw.max_per_stage_descriptor_storage_images.min(8))
                .saturating_add(raw.max_per_stage_descriptor_storage_buffers.min(16));
            let combined = raw.max_fragment_combined_output_resources;
            let mut expected = (
                raw.max_color_attachments.min(8),
                raw.max_per_stage_descriptor_storage_images.min(8),
                raw.max_per_stage_descriptor_storage_buffers.min(16),
            );
            if combined > minimum && combined < maximum {
                let mut extra = combined.saturating_sub(minimum);
                expected.0 = base
                    .max_color_attachments
                    .saturating_add(extra / 3)
                    .min(raw.max_color_attachments);
                extra = extra.saturating_sub(expected.0.saturating_sub(base.max_color_attachments));
                expected.1 = base
                    .max_storage_textures_per_shader_stage
                    .saturating_add(extra / 2)
                    .min(raw.max_per_stage_descriptor_storage_images);
                extra = extra.saturating_sub(
                    expected
                        .1
                        .saturating_sub(base.max_storage_textures_per_shader_stage),
                );
                expected.2 = base
                    .max_storage_buffers_per_shader_stage
                    .saturating_add(extra)
                    .min(raw.max_per_stage_descriptor_storage_buffers);
            }
            expected.0 = expected.0.min(8);
            expected.1 = expected.1.min(8);
            expected.2 = expected.2.min(16);
            assert_eq!(
                (
                    limits.max_color_attachments,
                    limits.max_storage_textures_per_shader_stage,
                    limits.max_storage_buffers_per_shader_stage
                ),
                expected
            );
        }
    }

    fn extension_names_from_pointers(extension_names: &[*const c_char]) -> Vec<&CStr> {
        extension_names
            .iter()
            .map(|name| unsafe { CStr::from_ptr(*name) })
            .collect()
    }

    /// Block 92 P92.4: RADV reports `maxVertexInputAttributeOffset == u32::MAX`,
    /// so the `+ 1` used to compute `maxVertexBufferArrayStride` overflowed --
    /// aborting the process under overflow checks and wrapping to 0 in release.
    /// The saturating add must yield the intended 2048 instead.
    #[test]
    fn hal_limits_from_vk_saturates_max_vertex_input_attribute_offset() {
        let vk = vk::PhysicalDeviceLimits {
            max_vertex_input_attribute_offset: u32::MAX,
            max_vertex_input_binding_stride: 2048,
            ..Default::default()
        };

        let limits = hal_limits_from_vk(vk, ASSUMED_MAX_BUFFER_SIZE, 0);

        assert_eq!(limits.max_vertex_buffer_array_stride, 2048);
    }

    /// A driver-reported offset below the 2048 cap still clamps the stride to
    /// `offset + 1`, so the saturating add did not change the normal path.
    #[test]
    fn hal_limits_from_vk_clamps_stride_to_attribute_offset_plus_one() {
        let vk = vk::PhysicalDeviceLimits {
            max_vertex_input_attribute_offset: 2047,
            max_vertex_input_binding_stride: 4096,
            ..Default::default()
        };

        let limits = hal_limits_from_vk(vk, ASSUMED_MAX_BUFFER_SIZE, 0);

        assert_eq!(limits.max_vertex_buffer_array_stride, 2048);

        let vk = vk::PhysicalDeviceLimits {
            max_vertex_input_attribute_offset: 1023,
            max_vertex_input_binding_stride: 4096,
            ..Default::default()
        };

        let limits = hal_limits_from_vk(vk, ASSUMED_MAX_BUFFER_SIZE, 0);

        assert_eq!(limits.max_vertex_buffer_array_stride, 1024);
    }

    #[test]
    fn yawgpu_vulkan_api_version_is_at_least_1_1() {
        let major = vk::api_version_major(YAWGPU_VULKAN_API_VERSION);
        let minor = vk::api_version_minor(YAWGPU_VULKAN_API_VERSION);

        assert!((major, minor) >= (1, 1));
    }

    #[test]
    fn is_supported_api_version_accepts_1_1_and_above() {
        assert!(is_supported_api_version(vk::API_VERSION_1_1));
        assert!(is_supported_api_version(vk::API_VERSION_1_2));
        assert!(is_supported_api_version(vk::API_VERSION_1_3));
    }

    #[test]
    fn is_supported_api_version_rejects_1_0() {
        assert!(!is_supported_api_version(vk::API_VERSION_1_0));
        assert!(!is_supported_api_version(vk::make_api_version(0, 1, 0, 0)));
    }

    #[test]
    fn vulkan_instance_extension_config_requires_khr_surface() {
        assert!(instance_extension_config(&[vk::KHR_WIN32_SURFACE_NAME]).is_none());
    }

    #[test]
    fn vulkan_instance_extension_config_enables_available_optional_extensions() {
        let (extension_names, flags) = instance_extension_config(&[
            vk::KHR_SURFACE_NAME,
            vk::EXT_METAL_SURFACE_NAME,
            vk::KHR_WIN32_SURFACE_NAME,
            vk::KHR_PORTABILITY_ENUMERATION_NAME,
        ])
        .expect("KHR_surface should allow instance extension configuration");

        let extension_names = extension_names_from_pointers(&extension_names);
        assert_eq!(
            extension_names,
            vec![
                vk::KHR_SURFACE_NAME,
                vk::EXT_METAL_SURFACE_NAME,
                vk::KHR_WIN32_SURFACE_NAME,
                vk::KHR_PORTABILITY_ENUMERATION_NAME,
            ]
        );
        assert_eq!(flags, vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
    }

    #[test]
    fn vulkan_instance_extension_config_skips_absent_optional_extensions() {
        let (extension_names, flags) =
            instance_extension_config(&[vk::KHR_SURFACE_NAME, vk::KHR_WIN32_SURFACE_NAME])
                .expect("KHR_surface should allow instance extension configuration");

        let extension_names = extension_names_from_pointers(&extension_names);
        assert_eq!(
            extension_names,
            vec![vk::KHR_SURFACE_NAME, vk::KHR_WIN32_SURFACE_NAME]
        );
        assert_eq!(flags, vk::InstanceCreateFlags::default());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_instance_new_constructs() {
        VulkanInstance::new().expect("create Vulkan instance");
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_instance_enumerate_adapters_returns_devices() {
        let adapters = VulkanInstance::new()
            .expect("create Vulkan instance")
            .enumerate_adapters();
        assert!(!adapters.is_empty());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_instance_create_surface_from_metal_layer_rejects_null_layer() {
        let instance = VulkanInstance::new().expect("create Vulkan instance");
        let error = unsafe { instance.create_surface_from_metal_layer(std::ptr::null_mut()) }
            .expect_err("null layer must fail");
        assert!(matches!(
            error,
            HalError::SwapchainCreationFailed {
                backend: "vulkan",
                message: "surface layer is null"
            }
        ));
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_instance_create_surface_from_windows_hwnd_rejects_null_hwnd() {
        let instance = VulkanInstance::new().expect("create Vulkan instance");
        let error = unsafe {
            instance.create_surface_from_windows_hwnd(std::ptr::null_mut(), std::ptr::null_mut())
        }
        .expect_err("null hwnd must fail");
        assert!(matches!(
            error,
            HalError::SwapchainCreationFailed {
                backend: "vulkan",
                message: "surface hwnd is null"
            }
        ));
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_adapter_name_returns_non_empty_name() {
        let adapter = VulkanInstance::new()
            .expect("create Vulkan instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Vulkan adapter");
        assert!(!adapter.name().is_empty());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_adapter_limits_reports_real_device_limits() {
        let adapter = VulkanInstance::new()
            .expect("create Vulkan instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Vulkan adapter");
        let limits = adapter.limits();

        assert!(limits.max_texture_dimension_2d >= 8192);
        assert!(limits.max_compute_invocations_per_workgroup >= 256);
        // Block 94 S3: Vulkan now executes SetImmediates, so it advertises
        // Dawn's base-tier maxImmediateSize.
        assert_eq!(limits.max_immediate_size, 64);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_adapter_sliced_3d_compression_matches_base_support_and_is_cached() {
        let Ok(instance) = VulkanInstance::new() else {
            eprintln!("SKIP: Vulkan instance unavailable");
            return;
        };
        let Some(adapter) = instance.enumerate_adapters().into_iter().next() else {
            eprintln!("SKIP: no Vulkan adapter available");
            return;
        };
        let cloned_adapter = adapter.clone();
        assert!(Arc::ptr_eq(
            &adapter.astc_sliced_3d_support,
            &cloned_adapter.astc_sliced_3d_support,
        ));
        assert!(adapter.astc_sliced_3d_support.get().is_none());
        assert_eq!(
            adapter.supports_texture_compression_bc_sliced_3d(),
            adapter.supports_texture_compression_bc(),
        );
        let astc_sliced_3d = adapter.supports_texture_compression_astc_sliced_3d();
        assert!(!astc_sliced_3d || adapter.supports_texture_compression_astc());
        assert_eq!(
            adapter.supports_texture_compression_astc_sliced_3d(),
            astc_sliced_3d,
        );
        assert_eq!(
            cloned_adapter.astc_sliced_3d_support.get(),
            Some(&astc_sliced_3d)
        );
        assert_eq!(
            cloned_adapter.supports_texture_compression_astc_sliced_3d(),
            astc_sliced_3d,
        );
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_adapter_supports_texture_component_swizzle_follows_portability_bit() {
        let adapter = VulkanInstance::new()
            .expect("create Vulkan instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Vulkan adapter");

        if adapter.has_device_extension(vk::KHR_PORTABILITY_SUBSET_NAME) {
            // Portability implementation (MoltenVK): the advertisement mirrors
            // the reported imageViewFormatSwizzle bit, which MoltenVK sets.
            let mut portability = vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default();
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut portability);
            unsafe {
                adapter
                    .instance
                    .instance
                    .get_physical_device_features2(adapter.physical_device, &mut features2);
            }
            assert_eq!(
                adapter.supports_texture_component_swizzle(),
                portability.image_view_format_swizzle == vk::TRUE
            );
        } else {
            assert!(adapter.supports_texture_component_swizzle());
        }
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_adapter_texture_formats_tiers_match_dawn_rule() {
        let adapter = VulkanInstance::new()
            .expect("create Vulkan instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Vulkan adapter");
        let features = unsafe {
            adapter
                .instance
                .instance
                .get_physical_device_features(adapter.physical_device)
        };
        let required_features = vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND;
        let rule = features.shader_storage_image_extended_formats == vk::TRUE
            && [
                vk::Format::R16_UNORM,
                vk::Format::R16_SNORM,
                vk::Format::R16G16_UNORM,
                vk::Format::R16G16_SNORM,
                vk::Format::R16G16B16A16_UNORM,
                vk::Format::R16G16B16A16_SNORM,
                vk::Format::R8_SNORM,
                vk::Format::R8G8_SNORM,
                vk::Format::R8G8B8A8_SNORM,
                vk::Format::B10G11R11_UFLOAT_PACK32,
            ]
            .into_iter()
            .all(|format| {
                let props = unsafe {
                    adapter
                        .instance
                        .instance
                        .get_physical_device_format_properties(adapter.physical_device, format)
                };
                props.optimal_tiling_features.contains(required_features)
            });

        assert_eq!(adapter.supports_texture_formats_tier1(), rule);
        assert_eq!(adapter.supports_texture_formats_tier2(), rule);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_adapter_create_device_returns_zero_allocation_device() {
        let adapter = VulkanInstance::new()
            .expect("create Vulkan instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Vulkan adapter");
        let device = adapter.create_device().expect("create Vulkan device");
        assert_eq!(device.allocation_count(), 0);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_instance_new_uses_shared_entry() {
        let first = VulkanInstance::new().expect("create first Vulkan instance");
        let second = VulkanInstance::new().expect("create second Vulkan instance");

        assert!(std::ptr::eq(first.inner._entry, second.inner._entry));
        assert!(std::ptr::eq(
            shared_entry().expect("shared Vulkan entry"),
            first.inner._entry
        ));
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_instance_device_creation_churn_survives_shared_entry() {
        const ITERATIONS: usize = 160;

        for iteration in 0..ITERATIONS {
            let instance = VulkanInstance::new().unwrap_or_else(|error| {
                panic!("create Vulkan instance at iteration {iteration}: {error:?}")
            });
            let adapter = instance
                .enumerate_adapters()
                .into_iter()
                .next()
                .unwrap_or_else(|| panic!("at least one Vulkan adapter at iteration {iteration}"));
            let device = adapter.create_device().unwrap_or_else(|error| {
                panic!("create Vulkan device at iteration {iteration}: {error:?}")
            });

            drop(device);
            drop(adapter);
            drop(instance);
        }
    }

    #[test]
    fn astc_ldr_formats_cover_all_mapped_blocks() {
        let formats = astc_ldr_formats().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(formats.len(), 28);
        let unique = formats
            .iter()
            .map(|format| format.as_raw())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique.len(), 28);
        for format in formats {
            let name = format!("{format:?}");
            assert!(name.starts_with("ASTC_"), "{name}");
            assert!(
                name.ends_with("_UNORM_BLOCK") || name.ends_with("_SRGB_BLOCK"),
                "{name}"
            );
        }
    }

    #[test]
    fn astc_sliced_3d_requires_base_support_without_probing() {
        let mut probes = 0;
        assert!(!astc_sliced_3d_supported(false, |_| {
            probes += 1;
            true
        }));
        assert_eq!(probes, 0);
    }

    #[test]
    fn astc_sliced_3d_rejects_each_failed_format_probe() {
        for unsupported in astc_ldr_formats() {
            let unsupported = unsupported.unwrap();
            assert!(!astc_sliced_3d_supported(true, |format| format != unsupported));
        }
    }

    #[test]
    fn astc_sliced_3d_accepts_all_successful_format_probes() {
        let mut probes = Vec::new();
        assert!(astc_sliced_3d_supported(true, |format| {
            probes.push(format);
            true
        }));
        assert_eq!(
            probes,
            astc_ldr_formats().collect::<Result<Vec<_>, _>>().unwrap()
        );
    }

    #[test]
    fn texture_compression_features_are_enabled_independently() {
        for bc in [vk::FALSE, vk::TRUE] {
            for etc2 in [vk::FALSE, vk::TRUE] {
                for astc in [vk::FALSE, vk::TRUE] {
                    let supported = vk::PhysicalDeviceFeatures {
                        texture_compression_bc: bc,
                        texture_compression_etc2: etc2,
                        texture_compression_astc_ldr: astc,
                        ..Default::default()
                    };
                    let enabled = enabled_texture_compression_features(&supported);
                    assert_eq!(enabled.texture_compression_bc, bc);
                    assert_eq!(enabled.texture_compression_etc2, etc2);
                    assert_eq!(enabled.texture_compression_astc_ldr, astc);
                }
            }
        }
    }

    /// Core feature-enable logic forwards each available feature into
    /// `enabled_features` and leaves each unavailable feature FALSE.
    #[test]
    fn vulkan_create_device_enables_supported_core_features() {
        // Simulate the feature-enable logic in create_device without a real GPU.
        for (supported, expected_enabled) in [(vk::TRUE, vk::TRUE), (vk::FALSE, vk::FALSE)] {
            let supported_features = vk::PhysicalDeviceFeatures {
                robust_buffer_access: supported,
                independent_blend: supported,
                shader_storage_image_extended_formats: supported,
                dual_src_blend: supported,
                shader_clip_distance: supported,
                geometry_shader: supported,
                draw_indirect_first_instance: supported,
                sample_rate_shading: supported,
                fragment_stores_and_atomics: supported,
                texture_compression_bc: supported,
                texture_compression_etc2: supported,
                texture_compression_astc_ldr: supported,
                ..Default::default()
            };
            let robust_buffer_access = supported_features.robust_buffer_access == vk::TRUE;
            let independent_blend = supported_features.independent_blend == vk::TRUE;
            let shader_storage_image_extended_formats =
                supported_features.shader_storage_image_extended_formats == vk::TRUE;
            let dual_src_blend = supported_features.dual_src_blend == vk::TRUE;
            let shader_clip_distance = supported_features.shader_clip_distance == vk::TRUE;
            let geometry_shader = supported_features.geometry_shader == vk::TRUE;
            let draw_indirect_first_instance =
                supported_features.draw_indirect_first_instance == vk::TRUE;
            let sample_rate_shading = supported_features.sample_rate_shading == vk::TRUE;
            let fragment_stores_and_atomics =
                supported_features.fragment_stores_and_atomics == vk::TRUE;
            let mut enabled_features = enabled_texture_compression_features(&supported_features);
            if robust_buffer_access {
                enabled_features.robust_buffer_access = vk::TRUE;
            }
            if independent_blend {
                enabled_features.independent_blend = vk::TRUE;
            }
            if shader_storage_image_extended_formats {
                enabled_features.shader_storage_image_extended_formats = vk::TRUE;
            }
            if dual_src_blend {
                enabled_features.dual_src_blend = vk::TRUE;
            }
            if shader_clip_distance {
                enabled_features.shader_clip_distance = vk::TRUE;
            }
            if geometry_shader {
                enabled_features.geometry_shader = vk::TRUE;
            }
            if draw_indirect_first_instance {
                enabled_features.draw_indirect_first_instance = vk::TRUE;
            }
            if sample_rate_shading {
                enabled_features.sample_rate_shading = vk::TRUE;
            }
            if fragment_stores_and_atomics {
                enabled_features.fragment_stores_and_atomics = vk::TRUE;
            }
            assert_eq!(enabled_features.texture_compression_bc, expected_enabled);
            assert_eq!(enabled_features.texture_compression_etc2, expected_enabled);
            assert_eq!(
                enabled_features.texture_compression_astc_ldr,
                expected_enabled
            );
            assert_eq!(
                enabled_features.robust_buffer_access, expected_enabled,
                "robust_buffer_access should be {expected_enabled} when supported={supported}"
            );
            assert_eq!(
                enabled_features.independent_blend, expected_enabled,
                "independent_blend should be {expected_enabled} when supported={supported}"
            );
            assert_eq!(
                enabled_features.shader_storage_image_extended_formats, expected_enabled,
                "shader_storage_image_extended_formats should be {expected_enabled} when supported={supported}"
            );
            assert_eq!(
                enabled_features.dual_src_blend, expected_enabled,
                "dual_src_blend should be {expected_enabled} when supported={supported}"
            );
            assert_eq!(
                enabled_features.shader_clip_distance, expected_enabled,
                "shader_clip_distance should be {expected_enabled} when supported={supported}"
            );
            assert_eq!(
                enabled_features.geometry_shader, expected_enabled,
                "geometry_shader should be {expected_enabled} when supported={supported}"
            );
            assert_eq!(
                enabled_features.draw_indirect_first_instance, expected_enabled,
                "draw_indirect_first_instance should be {expected_enabled} when supported={supported}"
            );
            assert_eq!(
                enabled_features.sample_rate_shading, expected_enabled,
                "sample_rate_shading should be {expected_enabled} when supported={supported}"
            );
            assert_eq!(
                enabled_features.fragment_stores_and_atomics, expected_enabled,
                "fragment_stores_and_atomics should be {expected_enabled} when supported={supported}"
            );
        }
    }

    #[test]
    fn vulkan_memory_model_is_available_from_core_1_2_or_extension() {
        assert!(vulkan_memory_model_available(vk::API_VERSION_1_2, false));
        assert!(vulkan_memory_model_available(vk::API_VERSION_1_1, true));
        assert!(!vulkan_memory_model_available(vk::API_VERSION_1_1, false));
    }

    #[test]
    fn vulkan_memory_model_extension_is_required_only_before_core_1_2() {
        assert!(!vulkan_memory_model_extension_required(
            vk::API_VERSION_1_2,
            true
        ));
        assert!(vulkan_memory_model_extension_required(
            vk::API_VERSION_1_1,
            true
        ));
        assert!(!vulkan_memory_model_extension_required(
            vk::API_VERSION_1_1,
            false
        ));
    }

    /// Regression: promotion-to-core must be decided on the *effective enabled*
    /// version `min(YAWGPU_VULKAN_API_VERSION, device_max)`, not the device's
    /// *maximum supported* version. A 1.3-capable device runs as 1.1 here, so the
    /// memory model is never core and its extension name must be enabled whenever
    /// the device exposes the extension + feature — otherwise the feature struct is
    /// chained into VkDeviceCreateInfo.pNext without the parent extension
    /// (VUID-VkDeviceCreateInfo-pNext-pNext). Mirrors the create_device decision and
    /// the `extension_names_from_pointers` test style; no real GPU required.
    #[test]
    fn vulkan_memory_model_extension_name_pushed_at_yawgpu_baseline() {
        let device_max = vk::API_VERSION_1_3;
        let extension_present = true;
        let vulkan_memory_model = true; // available + feature reported TRUE
        let effective_api_version = YAWGPU_VULKAN_API_VERSION.min(device_max);

        // Deciding on the effective version pushes the extension name.
        let mut extension_names: Vec<*const c_char> = Vec::new();
        if vulkan_memory_model
            && vulkan_memory_model_extension_required(effective_api_version, extension_present)
        {
            extension_names.push(vk::KHR_VULKAN_MEMORY_MODEL_NAME.as_ptr());
        }
        assert!(
            extension_names_from_pointers(&extension_names)
                .contains(&vk::KHR_VULKAN_MEMORY_MODEL_NAME),
            "extension name must be enabled at the 1.1 baseline"
        );

        // Deciding on the raw device-max would wrongly skip it (the original bug).
        let mut wrong: Vec<*const c_char> = Vec::new();
        if vulkan_memory_model
            && vulkan_memory_model_extension_required(device_max, extension_present)
        {
            wrong.push(vk::KHR_VULKAN_MEMORY_MODEL_NAME.as_ptr());
        }
        assert!(
            extension_names_from_pointers(&wrong).is_empty(),
            "device-max (1.3) wrongly treats the memory model as core"
        );

        // VkImageFormatListCreateInfo shares the same >= (1,2) promotion shape and
        // the same latent bug: available at the 1.1 baseline only via the extension.
        assert!(image_format_list_available(
            effective_api_version,
            extension_present
        ));
    }

    #[test]
    fn vulkan_memory_model_enablement_requires_available_reported_feature() {
        for (api_version, extension_present, feature_supported, expected) in [
            (vk::API_VERSION_1_2, false, true, true),
            (vk::API_VERSION_1_2, false, false, false),
            (vk::API_VERSION_1_1, true, true, true),
            (vk::API_VERSION_1_1, true, false, false),
            (vk::API_VERSION_1_1, false, true, false),
        ] {
            let available = vulkan_memory_model_available(api_version, extension_present);
            let features = if available {
                vk::PhysicalDeviceVulkanMemoryModelFeatures {
                    vulkan_memory_model: if feature_supported {
                        vk::TRUE
                    } else {
                        vk::FALSE
                    },
                    vulkan_memory_model_device_scope: vk::TRUE,
                    ..Default::default()
                }
            } else {
                vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
            };
            let vulkan_memory_model = vulkan_memory_model_enabled(
                available,
                features.vulkan_memory_model,
                features.vulkan_memory_model_device_scope,
            );
            let vulkan_memory_model_device_scope =
                vulkan_memory_model && features.vulkan_memory_model_device_scope == vk::TRUE;

            assert_eq!(
                vulkan_memory_model, expected,
                "vulkan_memory_model should be {expected} when api={api_version:#x} extension={extension_present} feature={feature_supported}"
            );
            assert_eq!(vulkan_memory_model_device_scope, expected);
        }
    }

    #[test]
    fn phase_review_storage16_requires_core_version_or_extension() {
        for (api, extension, expected) in [
            (vk::API_VERSION_1_0, false, false),
            (vk::API_VERSION_1_0, true, true),
            (vk::API_VERSION_1_1, false, true),
            (vk::API_VERSION_1_1, true, true),
        ] {
            assert_eq!(storage_16bit_available(api, extension), expected);
            let supported = vk::PhysicalDevice16BitStorageFeatures::default()
                .storage_buffer16_bit_access(true)
                .uniform_and_storage_buffer16_bit_access(true);
            let enabled =
                enabled_16bit_storage_features(storage_16bit_available(api, extension), supported);
            assert_eq!(enabled.enabled, expected);
        }
    }

    #[test]
    fn phase_review_f16_requires_uniform_storage16_access() {
        assert!(!shader_float16_supported(
            true,
            vk::TRUE,
            vk::TRUE,
            vk::TRUE,
            vk::FALSE
        ));
        assert!(shader_float16_supported(
            true,
            vk::TRUE,
            vk::TRUE,
            vk::TRUE,
            vk::TRUE
        ));
    }

    /// Shader f16 requires float arithmetic, integer decomposition and buffer storage.
    /// Pure logic test; no real GPU required.
    #[test]
    fn audit_t4_f16_requires_float_int_and_storage_features() {
        for mask in 0..32 {
            let bit = |n| {
                if mask & (1 << n) != 0 {
                    vk::TRUE
                } else {
                    vk::FALSE
                }
            };
            assert_eq!(
                shader_float16_supported(mask & 1 != 0, bit(1), bit(2), bit(3), bit(4)),
                mask == 31
            );
        }
    }

    #[test]
    fn vulkan_supports_shader_float16_requires_extension_and_feature() {
        for (extension_present, shader_float16, expected) in [
            (true, vk::TRUE, true),
            (true, vk::FALSE, false),
            (false, vk::TRUE, false),
            (false, vk::FALSE, false),
        ] {
            assert_eq!(
                shader_float16_supported(
                    extension_present,
                    shader_float16,
                    vk::TRUE,
                    vk::TRUE,
                    vk::TRUE
                ),
                expected
            );
        }
    }

    /// Block 99 R3: `timestamp-query` follows `timestampComputeAndGraphics`.
    #[test]
    fn vulkan_timestamp_query_from_limits_follows_timestamp_compute_and_graphics() {
        assert!(timestamp_query_from_limits(vk::TRUE));
        assert!(!timestamp_query_from_limits(vk::FALSE));
    }

    /// Block 102 R3: a finite, positive `timestampPeriod` is passed through;
    /// zero, negative and non-finite reports fall back to 1.0 ns per tick.
    #[test]
    fn timestamp_period_from_limits_passes_through_positive_finite_periods() {
        assert_eq!(timestamp_period_from_limits(1.0), 1.0);
        assert_eq!(timestamp_period_from_limits(41.666_668), 41.666_668);
        assert_eq!(
            timestamp_period_from_limits(f32::MIN_POSITIVE),
            f32::MIN_POSITIVE
        );
    }

    /// Block 102 R3: the fallback keeps the conversion pass usable when a
    /// driver reports a period that cannot scale ticks to nanoseconds.
    #[test]
    fn timestamp_period_from_limits_falls_back_on_non_positive_or_non_finite() {
        for period in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0] {
            assert_eq!(timestamp_period_from_limits(period), 1.0);
        }
    }

    /// Block 102 R3: the adapter reports the physical device's own
    /// `limits.timestampPeriod` (Dawn `GetTimestampPeriodInNS`).
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_adapter_timestamp_period_matches_physical_device_limits() {
        let adapter = VulkanInstance::new()
            .expect("create Vulkan instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Vulkan adapter");
        let properties = unsafe {
            adapter
                .instance
                .instance
                .get_physical_device_properties(adapter.physical_device)
        };

        let period = adapter.timestamp_period();

        assert_eq!(
            period,
            timestamp_period_from_limits(properties.limits.timestamp_period)
        );
        assert!(period.is_finite() && period > 0.0);
    }

    /// Block 99 R4: `depth32float-stencil8` needs `DEPTH_STENCIL_ATTACHMENT`.
    #[test]
    fn vulkan_depth32float_stencil8_from_flags_requires_depth_stencil_attachment() {
        assert!(depth32float_stencil8_from_flags(
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::FormatFeatureFlags::SAMPLED_IMAGE
        ));
        assert!(!depth32float_stencil8_from_flags(
            vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::TRANSFER_DST
        ));
        assert!(!depth32float_stencil8_from_flags(
            vk::FormatFeatureFlags::empty()
        ));
    }

    /// Block 99 R4: `rg11b10ufloat-renderable` needs both `COLOR_ATTACHMENT`
    /// and `COLOR_ATTACHMENT_BLEND`; either alone is not enough.
    #[test]
    fn vulkan_rg11b10ufloat_renderable_from_flags_requires_attachment_and_blend() {
        assert!(rg11b10ufloat_renderable_from_flags(
            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
                | vk::FormatFeatureFlags::SAMPLED_IMAGE
        ));
        assert!(!rg11b10ufloat_renderable_from_flags(
            vk::FormatFeatureFlags::COLOR_ATTACHMENT
        ));
        assert!(!rg11b10ufloat_renderable_from_flags(
            vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
        ));
        assert!(!rg11b10ufloat_renderable_from_flags(
            vk::FormatFeatureFlags::empty()
        ));
    }

    /// Block 99 R4: `bgra8unorm-storage` needs `STORAGE_IMAGE`.
    #[test]
    fn vulkan_bgra8unorm_storage_from_flags_requires_storage_image() {
        assert!(bgra8unorm_storage_from_flags(
            vk::FormatFeatureFlags::STORAGE_IMAGE | vk::FormatFeatureFlags::COLOR_ATTACHMENT
        ));
        assert!(!bgra8unorm_storage_from_flags(
            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
                | vk::FormatFeatureFlags::SAMPLED_IMAGE
        ));
        assert!(!bgra8unorm_storage_from_flags(
            vk::FormatFeatureFlags::empty()
        ));
    }

    /// Block 99 R4: `float32-filterable` needs `SAMPLED_IMAGE_FILTER_LINEAR`
    /// on all three 32-bit float formats; one missing format rejects.
    #[test]
    fn vulkan_float32_filterable_from_flags_requires_linear_filter_on_all_formats() {
        let linear = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR;
        let nearest_only = vk::FormatFeatureFlags::SAMPLED_IMAGE;

        assert!(float32_filterable_from_flags(linear, linear, linear));
        assert!(!float32_filterable_from_flags(nearest_only, linear, linear));
        assert!(!float32_filterable_from_flags(linear, nearest_only, linear));
        assert!(!float32_filterable_from_flags(linear, linear, nearest_only));
        assert!(!float32_filterable_from_flags(
            vk::FormatFeatureFlags::empty(),
            vk::FormatFeatureFlags::empty(),
            vk::FormatFeatureFlags::empty()
        ));
    }

    /// Block 99: each gated `supports_*` entry equals a fresh
    /// `vkGetPhysicalDeviceFormatProperties` /
    /// `vkGetPhysicalDeviceProperties` evaluation on the adapter's physical
    /// device, applying the same Dawn rule.
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_adapter_feature_queries_match_fresh_physical_device_queries() {
        let adapter = VulkanInstance::new()
            .expect("create Vulkan instance")
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("at least one Vulkan adapter");
        let optimal = |format: vk::Format| {
            unsafe {
                adapter
                    .instance
                    .instance
                    .get_physical_device_format_properties(adapter.physical_device, format)
            }
            .optimal_tiling_features
        };
        let properties = unsafe {
            adapter
                .instance
                .instance
                .get_physical_device_properties(adapter.physical_device)
        };

        assert_eq!(
            adapter.supports_timestamp_query(),
            properties.limits.timestamp_compute_and_graphics == vk::TRUE
        );
        assert_eq!(
            adapter.supports_depth32float_stencil8(),
            optimal(vk::Format::D32_SFLOAT_S8_UINT)
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        );
        assert_eq!(
            adapter.supports_rg11b10ufloat_renderable(),
            optimal(vk::Format::B10G11R11_UFLOAT_PACK32).contains(
                vk::FormatFeatureFlags::COLOR_ATTACHMENT
                    | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND
            )
        );
        assert_eq!(
            adapter.supports_bgra8unorm_storage(),
            optimal(vk::Format::B8G8R8A8_UNORM).contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
        );
        assert_eq!(
            adapter.supports_float32_filterable(),
            [
                vk::Format::R32_SFLOAT,
                vk::Format::R32G32_SFLOAT,
                vk::Format::R32G32B32A32_SFLOAT,
            ]
            .into_iter()
            .all(|format| optimal(format)
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR))
        );
    }

    #[test]
    fn vulkan_subgroups_require_webgpu_operations_and_compute_fragment_stages() {
        let required_operations = vk::SubgroupFeatureFlags::BASIC
            | vk::SubgroupFeatureFlags::BALLOT
            | vk::SubgroupFeatureFlags::SHUFFLE
            | vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE
            | vk::SubgroupFeatureFlags::ARITHMETIC
            | vk::SubgroupFeatureFlags::QUAD;
        let required_stages = vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT;

        assert!(subgroups_supported(required_operations, required_stages));
        assert!(!subgroups_supported(
            required_operations & !vk::SubgroupFeatureFlags::SHUFFLE,
            required_stages
        ));
        assert!(!subgroups_supported(
            required_operations & !vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE,
            required_stages
        ));
        assert!(!subgroups_supported(
            required_operations & !vk::SubgroupFeatureFlags::QUAD,
            required_stages
        ));
        assert!(!subgroups_supported(
            required_operations,
            required_stages & !vk::ShaderStageFlags::COMPUTE
        ));
        assert!(!subgroups_supported(
            required_operations,
            required_stages & !vk::ShaderStageFlags::FRAGMENT
        ));
    }

    #[test]
    fn vulkan_image_format_list_is_available_from_core_1_2_or_extension() {
        assert!(image_format_list_available(vk::API_VERSION_1_2, false));
        assert!(image_format_list_available(vk::API_VERSION_1_1, true));
        assert!(!image_format_list_available(vk::API_VERSION_1_1, false));
    }

    #[test]
    fn vulkan_subgroup_size_control_is_available_from_core_1_3_or_extension() {
        assert!(subgroup_size_control_available(vk::API_VERSION_1_3, false));
        assert!(subgroup_size_control_available(vk::API_VERSION_1_1, true));
        assert!(!subgroup_size_control_available(vk::API_VERSION_1_2, false));
    }

    #[test]
    fn vulkan_subgroup_size_control_supported_requires_every_condition() {
        let compute = vk::ShaderStageFlags::COMPUTE;
        let compute_and_fragment = compute | vk::ShaderStageFlags::FRAGMENT;
        assert!(subgroup_size_control_supported(
            true,
            true,
            vk::TRUE,
            vk::TRUE,
            compute
        ));
        assert!(subgroup_size_control_supported(
            true,
            true,
            vk::TRUE,
            vk::TRUE,
            compute_and_fragment
        ));
        // Each missing condition alone disables the feature.
        assert!(!subgroup_size_control_supported(
            false,
            true,
            vk::TRUE,
            vk::TRUE,
            compute
        ));
        assert!(!subgroup_size_control_supported(
            true,
            false,
            vk::TRUE,
            vk::TRUE,
            compute
        ));
        assert!(!subgroup_size_control_supported(
            true,
            true,
            vk::FALSE,
            vk::TRUE,
            compute
        ));
        assert!(!subgroup_size_control_supported(
            true,
            true,
            vk::TRUE,
            vk::FALSE,
            compute
        ));
        assert!(!subgroup_size_control_supported(
            true,
            true,
            vk::TRUE,
            vk::TRUE,
            vk::ShaderStageFlags::FRAGMENT
        ));
        assert!(!subgroup_size_control_supported(
            true,
            true,
            vk::TRUE,
            vk::TRUE,
            vk::ShaderStageFlags::empty()
        ));
        assert!(!subgroup_size_control_supported(
            false,
            false,
            vk::FALSE,
            vk::FALSE,
            vk::ShaderStageFlags::empty()
        ));
    }

    #[test]
    fn vulkan_subgroup_size_control_extension_usable_truth_table() {
        for subgroups in [false, true] {
            for present in [false, true] {
                for control in [vk::FALSE, vk::TRUE] {
                    assert_eq!(
                        subgroup_size_control_extension_usable(subgroups, present, control),
                        subgroups && present && control == vk::TRUE,
                        "{subgroups}/{present}/{control}"
                    );
                }
            }
        }
    }

    /// A snapshot that satisfies the full Block 108 R2 predicate.
    fn full_subgroup_query() -> SubgroupQuery {
        SubgroupQuery {
            supported_stages: vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT,
            range: Some((16, 64)),
            extension_present: true,
            subgroup_size_control: vk::TRUE,
            compute_full_subgroups: vk::TRUE,
            required_subgroup_size_stages: vk::ShaderStageFlags::COMPUTE,
            max_compute_workgroup_subgroups: 32,
        }
    }

    #[test]
    fn vulkan_subgroup_query_full_support_enables_extension_and_advertises_feature() {
        let query = full_subgroup_query();
        assert!(query.extension_usable());
        assert_eq!(
            query.caps(),
            Some(crate::HalSubgroupSizeControlCaps::new(16, 64, 32))
        );
    }

    /// Block 108 R5 (m1): without `computeFullSubgroups` the extension is
    /// still enabled (so ALLOW_VARYING applies) but the WebGPU feature is not
    /// advertised.
    #[test]
    fn vulkan_subgroup_query_without_compute_full_subgroups_enables_extension_only() {
        let query = SubgroupQuery {
            compute_full_subgroups: vk::FALSE,
            ..full_subgroup_query()
        };
        assert!(query.extension_usable());
        assert_eq!(query.caps(), None);

        let query = SubgroupQuery {
            required_subgroup_size_stages: vk::ShaderStageFlags::FRAGMENT,
            ..full_subgroup_query()
        };
        assert!(query.extension_usable());
        assert_eq!(query.caps(), None);
    }

    #[test]
    fn vulkan_subgroup_query_disables_extension_and_feature_without_prerequisites() {
        for query in [
            SubgroupQuery {
                range: None,
                ..full_subgroup_query()
            },
            SubgroupQuery {
                extension_present: false,
                ..full_subgroup_query()
            },
            SubgroupQuery {
                subgroup_size_control: vk::FALSE,
                ..full_subgroup_query()
            },
        ] {
            assert!(!query.extension_usable(), "{query:?}");
            assert_eq!(query.caps(), None, "{query:?}");
        }
    }

    #[test]
    fn vulkan_subgroup_size_range_from_properties_prefers_size_control_range() {
        assert_eq!(
            subgroup_size_range_from_properties(true, true, 32, 16, 64),
            Some((16, 64))
        );
        // Size control unavailable or unreported: fall back to subgroupSize.
        assert_eq!(
            subgroup_size_range_from_properties(true, false, 32, 16, 64),
            Some((32, 32))
        );
        assert_eq!(
            subgroup_size_range_from_properties(true, true, 32, 0, 64),
            Some((32, 32))
        );
        assert_eq!(
            subgroup_size_range_from_properties(true, true, 0, 0, 0),
            None
        );
        // WGSL subgroups unsupported or the range outside WebGPU bounds.
        assert_eq!(
            subgroup_size_range_from_properties(false, true, 32, 16, 64),
            None
        );
        assert_eq!(
            subgroup_size_range_from_properties(true, true, 32, 1, 64),
            None
        );
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_adapter_subgroup_query_is_computed_once_per_adapter() {
        let Ok(instance) = VulkanInstance::new() else {
            eprintln!("SKIP: Vulkan instance unavailable");
            return;
        };
        let Some(adapter) = instance.enumerate_adapters().into_iter().next() else {
            eprintln!("SKIP: no Vulkan adapter available");
            return;
        };
        let cloned_adapter = adapter.clone();
        assert!(Arc::ptr_eq(
            &adapter.subgroup_query,
            &cloned_adapter.subgroup_query
        ));
        assert!(adapter.subgroup_query.get().is_none());
        let caps = adapter.subgroup_size_control_caps();
        let snapshot = adapter.subgroup_query.get().copied();
        assert!(snapshot.is_some());
        assert_eq!(
            cloned_adapter.subgroup_size_range(),
            snapshot.and_then(|q| q.range)
        );
        assert_eq!(cloned_adapter.subgroup_size_control_caps(), caps);
        assert_eq!(adapter.subgroup_query.get().copied(), snapshot);
    }

    #[test]
    fn vulkan_select_max_buffer_size_prefers_maintenance4_then_maintenance3_then_default() {
        let finite_m4 = 8 * 1024 * 1024 * 1024;
        let finite_m3 = 4 * 1024 * 1024 * 1024;

        assert_eq!(select_max_buffer_size(finite_m4, u64::MAX), finite_m4);
        assert_eq!(select_max_buffer_size(0, finite_m3), finite_m3);
        assert_eq!(select_max_buffer_size(0, 0), ASSUMED_MAX_BUFFER_SIZE);
    }

    #[test]
    fn vulkan_subgroup_size_range_rejects_values_outside_webgpu_bounds() {
        assert_eq!(validated_subgroup_size_range(4, 64), Some((4, 64)));
        assert_eq!(validated_subgroup_size_range(1, 64), None);
        assert_eq!(validated_subgroup_size_range(4, 256), None);
    }

    /// Device creation enables only the `VK_KHR_16bit_storage` sub-features
    /// reported by the physical device. Pure-logic test, no real GPU required.
    #[test]
    fn vulkan_16bit_storage_enablement_mirrors_reported_subfeatures() {
        let supported = vk::PhysicalDevice16BitStorageFeatures {
            storage_buffer16_bit_access: vk::TRUE,
            uniform_and_storage_buffer16_bit_access: vk::TRUE,
            storage_input_output16: vk::FALSE,
            storage_push_constant16: vk::TRUE,
            ..Default::default()
        };

        let enabled = enabled_16bit_storage_features(true, supported);

        assert!(enabled.enabled);
        assert!(enabled.storage_buffer16_bit_access);
        assert!(enabled.uniform_and_storage_buffer16_bit_access);
        assert!(!enabled.storage_input_output16);
        assert!(enabled.storage_push_constant16);

        let disabled = enabled_16bit_storage_features(false, supported);
        assert_eq!(disabled, Enabled16BitStorageFeatures::default());

        let all_false =
            enabled_16bit_storage_features(true, vk::PhysicalDevice16BitStorageFeatures::default());
        assert_eq!(all_false, Enabled16BitStorageFeatures::default());
    }

    /// A portability implementation requires every supported subset feature to
    /// be enabled explicitly, so device creation mirrors the queried struct
    /// bit-for-bit — and chains nothing at all without the extension.
    /// Pure-logic test, no real GPU required.
    #[test]
    fn vulkan_portability_subset_enablement_mirrors_reported_features() {
        let supported = vk::PhysicalDevicePortabilitySubsetFeaturesKHR {
            constant_alpha_color_blend_factors: vk::TRUE,
            events: vk::TRUE,
            image_view_format_reinterpretation: vk::TRUE,
            image_view_format_swizzle: vk::TRUE,
            image_view2_d_on3_d_image: vk::FALSE,
            multisample_array_image: vk::TRUE,
            mutable_comparison_samplers: vk::TRUE,
            point_polygons: vk::FALSE,
            sampler_mip_lod_bias: vk::FALSE,
            separate_stencil_mask_ref: vk::TRUE,
            shader_sample_rate_interpolation_functions: vk::FALSE,
            tessellation_isolines: vk::FALSE,
            tessellation_point_mode: vk::FALSE,
            triangle_fans: vk::TRUE,
            vertex_attribute_access_beyond_stride: vk::TRUE,
            ..Default::default()
        };

        let enabled = enabled_portability_subset_features(true, supported);

        assert!(enabled.enabled);
        assert!(enabled.constant_alpha_color_blend_factors);
        assert!(enabled.events);
        assert!(enabled.image_view_format_reinterpretation);
        assert!(enabled.image_view_format_swizzle);
        assert!(!enabled.image_view2_d_on3_d_image);
        assert!(enabled.multisample_array_image);
        assert!(enabled.mutable_comparison_samplers);
        assert!(!enabled.point_polygons);
        assert!(!enabled.sampler_mip_lod_bias);
        assert!(enabled.separate_stencil_mask_ref);
        assert!(!enabled.shader_sample_rate_interpolation_functions);
        assert!(!enabled.tessellation_isolines);
        assert!(!enabled.tessellation_point_mode);
        assert!(enabled.triangle_fans);
        assert!(enabled.vertex_attribute_access_beyond_stride);

        // No portability extension: nothing is chained into VkDeviceCreateInfo.
        let absent = enabled_portability_subset_features(false, supported);
        assert_eq!(absent, EnabledPortabilitySubsetFeatures::default());
        assert!(!absent.enabled);

        // Present but reporting nothing: still chained (a zeroed struct is the
        // explicit "enable none of them"), with every bit left disabled.
        let none_reported = enabled_portability_subset_features(
            true,
            vk::PhysicalDevicePortabilitySubsetFeaturesKHR::default(),
        );
        assert_eq!(
            none_reported,
            EnabledPortabilitySubsetFeatures {
                enabled: true,
                ..Default::default()
            }
        );
    }

    /// The chained `VkPhysicalDevicePortabilitySubsetFeaturesKHR` carries the
    /// mirrored bits, the right `sType`, and a null `pNext` (the query chain's
    /// `pNext` must not leak into device creation).
    #[test]
    fn vulkan_portability_subset_to_vk_rebuilds_the_feature_struct() {
        let features = EnabledPortabilitySubsetFeatures {
            enabled: true,
            image_view_format_swizzle: true,
            triangle_fans: true,
            ..Default::default()
        };

        let vk_features = features.to_vk();

        assert_eq!(
            vk_features.s_type,
            vk::StructureType::PHYSICAL_DEVICE_PORTABILITY_SUBSET_FEATURES_KHR
        );
        assert!(vk_features.p_next.is_null());
        assert_eq!(vk_features.image_view_format_swizzle, vk::TRUE);
        assert_eq!(vk_features.triangle_fans, vk::TRUE);
        assert_eq!(vk_features.image_view2_d_on3_d_image, vk::FALSE);
        assert_eq!(vk_features.sampler_mip_lod_bias, vk::FALSE);
        assert_eq!(
            enabled_portability_subset_features(true, vk_features),
            features
        );
    }

    /// `texture-component-swizzle` is unconditional on a normal Vulkan driver
    /// (Dawn's rule) and follows `imageViewFormatSwizzle` on a portability one.
    /// Pure-logic test, no real GPU required.
    #[test]
    fn vulkan_texture_component_swizzle_gates_on_portability_feature() {
        assert!(texture_component_swizzle_supported(false, vk::FALSE));
        assert!(texture_component_swizzle_supported(false, vk::TRUE));
        assert!(texture_component_swizzle_supported(true, vk::TRUE));
        assert!(!texture_component_swizzle_supported(true, vk::FALSE));
    }
}
