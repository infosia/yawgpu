use std::ffi::CStr;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ash::vk;

use crate::{HalCopy, HalError};

const BACKEND: &str = "vulkan";

#[derive(Debug, Clone)]
pub struct VulkanInstance {
    inner: Arc<VulkanInstanceInner>,
}

impl VulkanInstance {
    pub fn new() -> Result<Self, HalError> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|_| HalError::BackendUnavailable { backend: BACKEND })?;
        let extension_names = [vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr()];
        let create_info = vk::InstanceCreateInfo::default()
            .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
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

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

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
        let inner = Arc::new(VulkanDeviceInner {
            _instance: Arc::clone(&self.instance),
            device,
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

struct VulkanDeviceInner {
    _instance: Arc<VulkanInstanceInner>,
    device: ash::Device,
    allocations: AtomicU64,
}

impl fmt::Debug for VulkanDeviceInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanDeviceInner")
            .finish_non_exhaustive()
    }
}

impl Drop for VulkanDeviceInner {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

#[derive(Debug)]
pub struct VulkanDevice {
    inner: Arc<VulkanDeviceInner>,
    queue: VulkanQueue,
}

impl VulkanDevice {
    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.inner.allocations.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn queue(&self) -> &VulkanQueue {
        &self.queue
    }
}

#[derive(Debug, Clone)]
pub struct VulkanQueue {
    inner: Arc<VulkanQueueInner>,
}

#[derive(Debug)]
struct VulkanQueueInner {
    device: Arc<VulkanDeviceInner>,
    queue: vk::Queue,
}

impl VulkanQueue {
    pub fn submit_empty(&self) -> Result<(), HalError> {
        unsafe {
            self.inner
                .device
                .device
                .queue_submit(self.inner.queue, &[], vk::Fence::null())
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
            self.inner
                .device
                .device
                .queue_wait_idle(self.inner.queue)
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
        }
        Ok(())
    }

    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
        if copies.is_empty() {
            self.submit_empty()
        } else {
            Err(HalError::BackendUnavailable { backend: BACKEND })
        }
    }
}

#[derive(Debug, Clone)]
pub struct VulkanBuffer;

#[derive(Debug, Clone)]
pub struct VulkanTexture;

#[derive(Debug, Clone)]
pub struct VulkanSampler;

#[derive(Debug, Clone)]
pub struct VulkanComputePipeline;

#[derive(Debug, Clone)]
pub struct VulkanRenderPipeline;

fn physical_device_name(properties: vk::PhysicalDeviceProperties) -> Option<String> {
    properties
        .device_name_as_c_str()
        .ok()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
}
