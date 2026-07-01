#[cfg(feature = "tiled")]
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::atomic::{AtomicU8, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock};

use ash::vk;

#[cfg(feature = "tiled")]
use crate::HalSubpassPassLayout;
use crate::{
    HalAddressMode, HalBlendFactor, HalBlendOperation, HalBoundBuffer, HalBoundSampler,
    HalBoundTexture, HalBuffer, HalBufferBindingKind, HalBufferClear, HalBufferCopy,
    HalBufferTextureCopy, HalBufferUsage, HalColorClearKind, HalColorTargetState,
    HalCompareFunction, HalComputeDispatch, HalComputePass, HalCopy, HalCullMode,
    HalDepthStencilState, HalDescriptorBinding, HalDescriptorBindingKind, HalDraw, HalError,
    HalExtent3d, HalFilterMode, HalFrontFace, HalIndexFormat, HalMipmapFilterMode,
    HalPrimitiveTopology, HalQueryKind, HalQuerySet, HalRenderLoadOp, HalRenderPass,
    HalRenderPipelineDescriptor, HalResolveQuerySet, HalSampler, HalSamplerDescriptor,
    HalShaderSource, HalStencilOperation, HalSurfaceConfiguration, HalTexture, HalTextureCopy,
    HalTextureDescriptor, HalTextureFormat, HalTextureUsage, HalVertexFormat, HalVertexStepMode,
};

const BACKEND: &str = "vulkan";
/// Minimum Vulkan API version yawgpu requests at vkCreateInstance.
/// Documented in specs/blocks/60-real-backends.md § Minimum Vulkan version.
const YAWGPU_VULKAN_API_VERSION: u32 = vk::API_VERSION_1_1;
const IMAGE_LAYOUT_UNDEFINED: u8 = 0;
const IMAGE_LAYOUT_TRANSFER_DST: u8 = 1;
const IMAGE_LAYOUT_TRANSFER_SRC: u8 = 2;
const IMAGE_LAYOUT_COLOR_ATTACHMENT: u8 = 3;
const IMAGE_LAYOUT_PRESENT: u8 = 4;

static VULKAN_ENTRY: OnceLock<ash::Entry> = OnceLock::new();
static VULKAN_ENTRY_INIT: Mutex<()> = Mutex::new(());

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
}

impl VulkanAdapter {
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
        })
    }

    /// Returns the name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
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

    /// Returns true when 3D BC texture compression is supported by this physical device.
    #[must_use]
    pub fn supports_texture_compression_bc_sliced_3d(&self) -> bool {
        false
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

    /// Returns true when 3D ASTC texture compression is supported by this physical device.
    #[must_use]
    pub fn supports_texture_compression_astc_sliced_3d(&self) -> bool {
        false
    }

    /// Returns true when WGSL `shader-f16` is supported by this physical device.
    #[must_use]
    pub(super) fn supports_shader_float16(&self) -> bool {
        let extension_present = self.has_device_extension(vk::KHR_SHADER_FLOAT16_INT8_NAME);
        if !extension_present {
            return false;
        }

        let mut features = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features);
        unsafe {
            self.instance
                .instance
                .get_physical_device_features2(self.physical_device, &mut features2);
        }
        shader_float16_supported(extension_present, features.shader_float16)
    }

    /// Returns true when WGSL `subgroups` is supported by this physical device.
    #[must_use]
    pub(super) fn supports_subgroups(&self) -> bool {
        self.subgroup_size_range().is_some()
    }

    /// Returns the supported subgroup size range for this physical device.
    #[must_use]
    pub(super) fn subgroup_size_range(&self) -> Option<(u32, u32)> {
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

        if !subgroups_supported(subgroup.supported_operations, subgroup.supported_stages) {
            return None;
        }

        let api_version = unsafe {
            self.instance
                .instance
                .get_physical_device_properties(self.physical_device)
                .api_version
        };
        let size_control_available = subgroup_size_control_available(
            api_version,
            self.has_device_extension(vk::EXT_SUBGROUP_SIZE_CONTROL_NAME),
        );
        let range = if size_control_available
            && size_control.min_subgroup_size != 0
            && size_control.max_subgroup_size != 0
        {
            Some((
                size_control.min_subgroup_size,
                size_control.max_subgroup_size,
            ))
        } else if subgroup.subgroup_size != 0 {
            Some((subgroup.subgroup_size, subgroup.subgroup_size))
        } else {
            None
        }?;
        validated_subgroup_size_range(range.0, range.1)
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
        if self.has_device_extension(vk::KHR_PORTABILITY_SUBSET_NAME) {
            extension_names.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }
        if self.has_device_extension(vk::KHR_SWAPCHAIN_NAME) {
            extension_names.push(vk::KHR_SWAPCHAIN_NAME.as_ptr());
        }
        let depth_clip_enable_extension = self.has_device_extension(vk::EXT_DEPTH_CLIP_ENABLE_NAME);
        // CTS finding F-129(1): naga lowers WGSL `discard` to SPIR-V
        // `OpDemoteToHelperInvocation` so that derivatives (`fwidth`/`dpdx`/`dpdy`)
        // after a non-uniform `discard` stay well-defined. Executing that opcode
        // requires `shaderDemoteToHelperInvocation`. Enable it whenever the device
        // supports it; if absent, degrade gracefully (device creation still
        // succeeds — shaders without `discard` are unaffected, and a `discard`
        // shader would fail pipeline validation rather than crash here).
        let shader_demote_extension =
            self.has_device_extension(vk::EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_NAME);
        let shader_demote_to_helper_invocation = if shader_demote_extension {
            let mut demote_features =
                vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures::default();
            let mut features2 =
                vk::PhysicalDeviceFeatures2::default().push_next(&mut demote_features);
            unsafe {
                self.instance
                    .instance
                    .get_physical_device_features2(self.physical_device, &mut features2);
            }
            demote_features.shader_demote_to_helper_invocation == vk::TRUE
        } else {
            false
        };
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
        let vulkan_memory_model = vulkan_memory_model_available
            && vulkan_memory_model_supported_features.vulkan_memory_model == vk::TRUE;
        let vulkan_memory_model_device_scope = vulkan_memory_model
            && vulkan_memory_model_supported_features.vulkan_memory_model_device_scope == vk::TRUE;
        let image_format_list =
            image_format_list_available(effective_api_version, image_format_list_extension);
        let mut shader_float16_int8_features =
            vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut storage_16bit_supported_features =
            vk::PhysicalDevice16BitStorageFeatures::default();
        if shader_float16_int8_extension || storage_16bit_extension {
            let mut features2 = vk::PhysicalDeviceFeatures2::default();
            if shader_float16_int8_extension {
                features2 = features2.push_next(&mut shader_float16_int8_features);
            }
            if storage_16bit_extension {
                features2 = features2.push_next(&mut storage_16bit_supported_features);
            }
            unsafe {
                self.instance
                    .instance
                    .get_physical_device_features2(self.physical_device, &mut features2);
            }
        }
        let shader_float16 = shader_float16_supported(
            shader_float16_int8_extension,
            shader_float16_int8_features.shader_float16,
        );
        let storage_16bit_features = enabled_16bit_storage_features(
            storage_16bit_extension,
            storage_16bit_supported_features,
        );
        let supported_features = unsafe {
            self.instance
                .instance
                .get_physical_device_features(self.physical_device)
        };
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
        // WebGPU render pipelines can configure blend/write masks per color target.
        // Vulkan requires independentBlend for differing per-attachment blend state.
        let independent_blend = supported_features.independent_blend == vk::TRUE;
        let depth_clamp = supported_features.depth_clamp == vk::TRUE;
        let depth_clip_control = depth_clamp && depth_clip_enable_extension;
        // Per-sample MSAA subpass input reads `@builtin(sample_index)`, which
        // lowers to SPIR-V `SampleId` and auto-promotes the fragment shader to
        // per-sample execution; Vulkan requires sampleRateShading for that.
        #[cfg(feature = "tiled")]
        let sample_rate_shading = supported_features.sample_rate_shading == vk::TRUE;
        let mut enabled_features = vk::PhysicalDeviceFeatures::default();
        if occlusion_query_precise {
            enabled_features.occlusion_query_precise = vk::TRUE;
        }
        if depth_clip_control {
            enabled_features.depth_clamp = vk::TRUE;
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
        #[cfg(feature = "tiled")]
        if sample_rate_shading {
            enabled_features.sample_rate_shading = vk::TRUE;
        }
        if shader_demote_to_helper_invocation {
            extension_names.push(vk::EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_NAME.as_ptr());
        }
        if shader_float16 {
            extension_names.push(vk::KHR_SHADER_FLOAT16_INT8_NAME.as_ptr());
        }
        if storage_16bit_features.enabled {
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
        let mut depth_clip_enable_features =
            vk::PhysicalDeviceDepthClipEnableFeaturesEXT::default().depth_clip_enable(true);
        let mut shader_demote_features =
            vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures::default()
                .shader_demote_to_helper_invocation(true);
        let mut shader_float16_int8_enable_features =
            vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
        let mut vulkan_memory_model_enable_features =
            vk::PhysicalDeviceVulkanMemoryModelFeatures::default()
                .vulkan_memory_model(true)
                .vulkan_memory_model_device_scope(vulkan_memory_model_device_scope);
        let mut storage_16bit_enable_features = storage_16bit_features.to_vk();
        let mut create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names)
            .enabled_features(&enabled_features);
        if depth_clip_control {
            create_info = create_info.push_next(&mut depth_clip_enable_features);
        }
        if shader_demote_to_helper_invocation {
            create_info = create_info.push_next(&mut shader_demote_features);
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
        let inner = Arc::new(VulkanDeviceInner {
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
            storage_buffer16_bit_access: storage_16bit_features.storage_buffer16_bit_access,
            uniform_and_storage_buffer16_bit_access: storage_16bit_features
                .uniform_and_storage_buffer16_bit_access,
            storage_input_output16: storage_16bit_features.storage_input_output16,
            storage_push_constant16: storage_16bit_features.storage_push_constant16,
            max_sampler_anisotropy,
            #[cfg(feature = "tiled")]
            subpass_render_pass_cache: Mutex::new(BTreeMap::new()),
            allocations: AtomicU64::new(0),
        });
        Ok(VulkanDevice {
            inner: Arc::clone(&inner),
            queue: VulkanQueue {
                inner: Arc::new(VulkanQueueInner {
                    device: inner,
                    queue,
                    retire: Mutex::new(RetireRing::new(RETIRE_RING_SIZE)),
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

fn shader_float16_supported(extension_present: bool, shader_float16: vk::Bool32) -> bool {
    extension_present && shader_float16 == vk::TRUE
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
    // Deviation from Dawn: yawgpu does not require subgroup size control until it creates varying-size subgroup pipelines.
    supported_operations.contains(required_operations) && supported_stages.contains(required_stages)
}

fn validated_subgroup_size_range(min: u32, max: u32) -> Option<(u32, u32)> {
    if min < 4 || max > 128 {
        return None;
    }
    Some((min, max))
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

fn subgroup_size_control_available(api_version: u32, extension_present: bool) -> bool {
    let major = vk::api_version_major(api_version);
    let minor = vk::api_version_minor(api_version);
    (major, minor) >= (1, 3) || extension_present
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
mod pipeline;
mod query_set;
mod queue;
mod surface;
use self::buffer::*;
use self::device::*;
use self::encode::*;
use self::error::*;
use self::format::*;
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
    use super::*;

    fn extension_names_from_pointers(extension_names: &[*const c_char]) -> Vec<&CStr> {
        extension_names
            .iter()
            .map(|name| unsafe { CStr::from_ptr(*name) })
            .collect()
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

    /// `robust_buffer_access` feature-enable logic: when the physical device
    /// reports the feature as available (vk::TRUE), it must be forwarded into
    /// `enabled_features`; when absent, it must remain FALSE.
    #[test]
    fn vulkan_create_device_enables_supported_core_features() {
        // Simulate the feature-enable logic in create_device without a real GPU.
        for (supported, expected_enabled) in [(vk::TRUE, vk::TRUE), (vk::FALSE, vk::FALSE)] {
            let supported_features = vk::PhysicalDeviceFeatures {
                robust_buffer_access: supported,
                independent_blend: supported,
                sample_rate_shading: supported,
                ..Default::default()
            };
            let robust_buffer_access = supported_features.robust_buffer_access == vk::TRUE;
            let independent_blend = supported_features.independent_blend == vk::TRUE;
            #[cfg(feature = "tiled")]
            let sample_rate_shading = supported_features.sample_rate_shading == vk::TRUE;
            let mut enabled_features = vk::PhysicalDeviceFeatures::default();
            if robust_buffer_access {
                enabled_features.robust_buffer_access = vk::TRUE;
            }
            if independent_blend {
                enabled_features.independent_blend = vk::TRUE;
            }
            #[cfg(feature = "tiled")]
            if sample_rate_shading {
                enabled_features.sample_rate_shading = vk::TRUE;
            }
            assert_eq!(
                enabled_features.robust_buffer_access, expected_enabled,
                "robust_buffer_access should be {expected_enabled} when supported={supported}"
            );
            assert_eq!(
                enabled_features.independent_blend, expected_enabled,
                "independent_blend should be {expected_enabled} when supported={supported}"
            );
            #[cfg(feature = "tiled")]
            assert_eq!(
                enabled_features.sample_rate_shading, expected_enabled,
                "sample_rate_shading should be {expected_enabled} when supported={supported}"
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
            let vulkan_memory_model = available && features.vulkan_memory_model == vk::TRUE;
            let vulkan_memory_model_device_scope =
                vulkan_memory_model && features.vulkan_memory_model_device_scope == vk::TRUE;

            assert_eq!(
                vulkan_memory_model, expected,
                "vulkan_memory_model should be {expected} when api={api_version:#x} extension={extension_present} feature={feature_supported}"
            );
            assert_eq!(vulkan_memory_model_device_scope, expected);
        }
    }

    /// `supports_shader_float16` advertises support only when the
    /// `VK_KHR_shader_float16_int8` extension is present and `shaderFloat16`
    /// reports TRUE. Pure-logic test, no real GPU required.
    #[test]
    fn vulkan_supports_shader_float16_requires_extension_and_feature() {
        for (extension_present, shader_float16, expected) in [
            (true, vk::TRUE, true),
            (true, vk::FALSE, false),
            (false, vk::TRUE, false),
            (false, vk::FALSE, false),
        ] {
            assert_eq!(
                shader_float16_supported(extension_present, shader_float16),
                expected
            );
        }
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
}
