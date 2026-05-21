use std::ffi::{c_char, c_void, CStr, CString};
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::atomic::{AtomicU8, Ordering as AtomicOrdering};
use std::sync::Arc;

use ash::vk;

use crate::{
    HalAddressMode, HalBoundBuffer, HalBufferBindingKind, HalBufferCopy, HalBufferTextureCopy,
    HalCompareFunction, HalComputePass, HalCopy, HalDescriptorBinding, HalError, HalExtent3d,
    HalFilterMode, HalMipmapFilterMode, HalPrimitiveTopology, HalRenderLoadOp, HalRenderPass,
    HalRenderPipelineDescriptor, HalSamplerDescriptor, HalShaderSource, HalSurfaceConfiguration,
    HalTextureCopy, HalTextureDescriptor, HalTextureFormat, HalTextureUsage, HalVertexFormat,
    HalVertexStepMode,
};

const BACKEND: &str = "vulkan";
const IMAGE_LAYOUT_UNDEFINED: u8 = 0;
const IMAGE_LAYOUT_TRANSFER_DST: u8 = 1;
const IMAGE_LAYOUT_TRANSFER_SRC: u8 = 2;
const IMAGE_LAYOUT_COLOR_ATTACHMENT: u8 = 3;
const IMAGE_LAYOUT_PRESENT: u8 = 4;

/// Stores vulkan instance data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct VulkanInstance {
    inner: Arc<VulkanInstanceInner>,
}

impl VulkanInstance {
    /// Creates a new instance.
    pub fn new() -> Result<Self, HalError> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|_| HalError::BackendUnavailable { backend: BACKEND })?;
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
        let create_info = vk::InstanceCreateInfo::default()
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
            ash::ext::metal_surface::Instance::new(&self.inner._entry, &self.inner.instance);
        let create_info = vk::MetalSurfaceCreateInfoEXT::default().layer(layer);
        let surface = unsafe { loader.create_metal_surface(&create_info, None) }.map_err(|_| {
            HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "vkCreateMetalSurfaceEXT failed",
            }
        })?;
        Ok(VulkanSurface {
            instance: Arc::clone(&self.inner),
            surface,
            swapchain: None,
            config: None,
            current_image_index: None,
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
            ash::khr::win32_surface::Instance::new(&self.inner._entry, &self.inner.instance);
        let create_info = vk::Win32SurfaceCreateInfoKHR::default()
            .hinstance(hinstance as _)
            .hwnd(hwnd as _);
        let surface = unsafe { loader.create_win32_surface(&create_info, None) }.map_err(|_| {
            HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "vkCreateWin32SurfaceKHR failed",
            }
        })?;
        Ok(VulkanSurface {
            instance: Arc::clone(&self.inner),
            surface,
            swapchain: None,
            config: None,
            current_image_index: None,
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

struct VulkanInstanceInner {
    _entry: ash::Entry,
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
        Some(Self {
            instance,
            physical_device,
            name: physical_device_name(properties)?,
        })
    }

    /// Returns the name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
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
        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&extension_names);
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
        let inner = Arc::new(VulkanDeviceInner {
            _instance: Arc::clone(&self.instance),
            device,
            physical_device: self.physical_device,
            memory_properties,
            queue_family_index,
            allocations: AtomicU64::new(0),
        });
        Ok(VulkanDevice {
            inner: Arc::clone(&inner),
            queue: VulkanQueue {
                inner: Arc::new(VulkanQueueInner {
                    device: inner,
                    queue,
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
        let extensions = unsafe {
            self.instance
                .instance
                .enumerate_device_extension_properties(self.physical_device)
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
}

mod buffer;
mod device;
mod encode;
mod error;
mod format;
mod pipeline;
mod queue;
mod surface;
use self::buffer::*;
use self::device::*;
use self::encode::*;
use self::error::*;
use self::format::*;
use self::pipeline::*;
use self::queue::*;
use self::surface::*;
use self::texture::*;
#[cfg(test)]
mod test_helpers;
mod texture;

pub use buffer::VulkanBuffer;
pub use device::VulkanDevice;
pub use pipeline::{VulkanComputePipeline, VulkanRenderPipeline};
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
}
