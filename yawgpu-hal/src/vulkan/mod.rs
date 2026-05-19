use std::ffi::CStr;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ash::vk;

use crate::{HalBufferCopy, HalCopy, HalError};

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
        let memory_properties = unsafe {
            self.instance
                .instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let inner = Arc::new(VulkanDeviceInner {
            _instance: Arc::clone(&self.instance),
            device,
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

struct VulkanDeviceInner {
    _instance: Arc<VulkanInstanceInner>,
    device: ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_family_index: u32,
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

    #[must_use]
    pub fn create_buffer(&self, size: u64) -> VulkanBuffer {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        match create_buffer(Arc::clone(&self.inner), size) {
            Ok(inner) => VulkanBuffer {
                inner: Some(Arc::new(inner)),
                size,
            },
            Err(_) => VulkanBuffer { inner: None, size },
        }
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
            return self.submit_empty();
        }
        submit_buffer_copies(&self.inner, copies)
    }
}

#[derive(Debug, Clone)]
pub struct VulkanBuffer {
    inner: Option<Arc<VulkanBufferInner>>,
    size: u64,
}

impl VulkanBuffer {
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), HalError> {
        let len = u64::try_from(data.len()).map_err(|_| buffer_error("write size is too large"))?;
        self.validate_range(offset, len)?;
        if data.is_empty() {
            return Ok(());
        }
        let inner = self.inner()?;
        if inner.mapped.is_null() {
            return Err(buffer_error("buffer memory is not mapped"));
        }
        let offset = usize::try_from(offset).map_err(|_| buffer_error("offset is too large"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), inner.mapped.add(offset), data.len());
        }
        Ok(())
    }

    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        let len = usize::try_from(len).map_err(|_| buffer_error("read length is too large"))?;
        self.validate_range(
            offset,
            u64::try_from(len).map_err(|_| buffer_error("read length is too large"))?,
        )?;
        let mut data = vec![0; len];
        if len == 0 {
            return Ok(data);
        }
        let inner = self.inner()?;
        if inner.mapped.is_null() {
            return Err(buffer_error("buffer memory is not mapped"));
        }
        let offset = usize::try_from(offset).map_err(|_| buffer_error("offset is too large"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(inner.mapped.add(offset), data.as_mut_ptr(), len);
        }
        Ok(data)
    }

    fn inner(&self) -> Result<&VulkanBufferInner, HalError> {
        self.inner
            .as_deref()
            .ok_or_else(|| buffer_error("buffer allocation failed"))
    }

    fn validate_range(&self, offset: u64, len: u64) -> Result<(), HalError> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| buffer_error("buffer range overflows"))?;
        if end > self.size {
            return Err(buffer_error("buffer range exceeds buffer size"));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct VulkanBufferInner {
    device: Arc<VulkanDeviceInner>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
}

unsafe impl Send for VulkanBufferInner {}
unsafe impl Sync for VulkanBufferInner {}

impl Drop for VulkanBufferInner {
    fn drop(&mut self) {
        unsafe {
            if !self.mapped.is_null() {
                self.device.device.unmap_memory(self.memory);
            }
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}

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

fn create_buffer(device: Arc<VulkanDeviceInner>, size: u64) -> Result<VulkanBufferInner, HalError> {
    let allocation_size = size.max(1);
    let create_info = vk::BufferCreateInfo::default()
        .size(allocation_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.device.create_buffer(&create_info, None) }
        .map_err(|_| buffer_error("buffer creation failed"))?;
    let requirements = unsafe { device.device.get_buffer_memory_requirements(buffer) };
    let memory_type_index = find_memory_type_index(
        &device.memory_properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or_else(|| {
        unsafe {
            device.device.destroy_buffer(buffer, None);
        }
        buffer_error("compatible buffer memory type not found")
    })?;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = unsafe { device.device.allocate_memory(&allocate_info, None) }.map_err(|_| {
        unsafe {
            device.device.destroy_buffer(buffer, None);
        }
        buffer_error("buffer memory allocation failed")
    })?;
    if let Err(error) = unsafe { device.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.device.destroy_buffer(buffer, None);
            device.device.free_memory(memory, None);
        }
        return Err(map_buffer_error(error, "buffer memory bind failed"));
    }
    let mapped = match unsafe {
        device
            .device
            .map_memory(memory, 0, requirements.size, vk::MemoryMapFlags::empty())
    } {
        Ok(mapped) => mapped.cast::<u8>(),
        Err(error) => {
            unsafe {
                device.device.destroy_buffer(buffer, None);
                device.device.free_memory(memory, None);
            }
            return Err(map_buffer_error(error, "buffer memory map failed"));
        }
    };
    Ok(VulkanBufferInner {
        device,
        buffer,
        memory,
        mapped,
    })
}

fn find_memory_type_index(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    properties.memory_types[..usize::try_from(properties.memory_type_count).ok()?]
        .iter()
        .enumerate()
        .find_map(|(index, memory_type)| {
            let index = u32::try_from(index).ok()?;
            let supported = (type_bits & (1 << index)) != 0;
            (supported && memory_type.property_flags.contains(required)).then_some(index)
        })
}

fn submit_buffer_copies(queue: &VulkanQueueInner, copies: &[HalCopy]) -> Result<(), HalError> {
    let buffer_copies = copies
        .iter()
        .map(|copy| match copy {
            HalCopy::Buffer(copy) => Ok(copy),
            _ => Err(HalError::BackendUnavailable { backend: BACKEND }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(queue.device.queue_family_index);
    let command_pool = unsafe {
        queue
            .device
            .device
            .create_command_pool(&command_pool_info, None)
    }
    .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
    let result = record_and_submit_buffer_copies(queue, command_pool, &buffer_copies);
    unsafe {
        queue.device.device.destroy_command_pool(command_pool, None);
    }
    result
}

fn record_and_submit_buffer_copies(
    queue: &VulkanQueueInner,
    command_pool: vk::CommandPool,
    copies: &[&HalBufferCopy],
) -> Result<(), HalError> {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffers = unsafe { queue.device.device.allocate_command_buffers(&allocate_info) }
        .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
    let Some(&command_buffer) = command_buffers.first() else {
        return Err(HalError::QueueSubmissionFailed { backend: BACKEND });
    };
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        queue
            .device
            .device
            .begin_command_buffer(command_buffer, &begin_info)
            .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
    }
    for copy in copies {
        encode_buffer_copy(&queue.device.device, command_buffer, copy)?;
    }
    unsafe {
        queue
            .device
            .device
            .end_command_buffer(command_buffer)
            .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        queue
            .device
            .device
            .queue_submit(queue.queue, &[submit_info], vk::Fence::null())
            .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
        queue
            .device
            .device
            .queue_wait_idle(queue.queue)
            .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
    }
    Ok(())
}

fn encode_buffer_copy(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    copy: &HalBufferCopy,
) -> Result<(), HalError> {
    let crate::HalBuffer::Vulkan(source) = &copy.source else {
        return Err(buffer_error("source buffer is not Vulkan-backed"));
    };
    let crate::HalBuffer::Vulkan(destination) = &copy.destination else {
        return Err(buffer_error("destination buffer is not Vulkan-backed"));
    };
    source.validate_range(copy.source_offset, copy.size)?;
    destination.validate_range(copy.destination_offset, copy.size)?;
    if copy.size == 0 {
        return Ok(());
    }
    let source = source.inner()?;
    let destination = destination.inner()?;
    let region = vk::BufferCopy::default()
        .src_offset(copy.source_offset)
        .dst_offset(copy.destination_offset)
        .size(copy.size);
    unsafe {
        device.cmd_copy_buffer(command_buffer, source.buffer, destination.buffer, &[region]);
    }
    Ok(())
}

fn buffer_error(message: &'static str) -> HalError {
    HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    }
}

fn map_buffer_error(_error: vk::Result, message: &'static str) -> HalError {
    buffer_error(message)
}
