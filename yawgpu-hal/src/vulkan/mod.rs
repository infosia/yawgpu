use std::ffi::CStr;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::atomic::{AtomicU8, Ordering as AtomicOrdering};
use std::sync::Arc;

use ash::vk;

use crate::{
    HalAddressMode, HalBufferCopy, HalBufferTextureCopy, HalCompareFunction, HalCopy, HalError,
    HalExtent3d, HalFilterMode, HalMipmapFilterMode, HalSamplerDescriptor, HalTextureCopy,
    HalTextureDescriptor, HalTextureFormat, HalTextureUsage,
};

const BACKEND: &str = "vulkan";
const IMAGE_LAYOUT_UNDEFINED: u8 = 0;
const IMAGE_LAYOUT_TRANSFER_DST: u8 = 1;
const IMAGE_LAYOUT_TRANSFER_SRC: u8 = 2;

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

    #[must_use]
    pub fn create_texture(&self, descriptor: &HalTextureDescriptor) -> VulkanTexture {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        match create_texture(Arc::clone(&self.inner), descriptor) {
            Ok((inner, bytes_per_pixel)) => VulkanTexture {
                inner: Some(Arc::new(inner)),
                width: descriptor.width,
                height: descriptor.height,
                depth_or_array_layers: descriptor.depth_or_array_layers,
                bytes_per_pixel,
            },
            Err(_) => VulkanTexture {
                inner: None,
                width: descriptor.width,
                height: descriptor.height,
                depth_or_array_layers: descriptor.depth_or_array_layers,
                bytes_per_pixel: 0,
            },
        }
    }

    #[must_use]
    pub fn create_sampler(&self, descriptor: &HalSamplerDescriptor) -> VulkanSampler {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        VulkanSampler {
            _inner: create_sampler(Arc::clone(&self.inner), descriptor)
                .ok()
                .map(Arc::new),
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
        submit_copies(&self.inner, copies)
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
pub struct VulkanTexture {
    inner: Option<Arc<VulkanTextureInner>>,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    bytes_per_pixel: u32,
}

impl VulkanTexture {
    fn inner(&self) -> Result<&VulkanTextureInner, HalError> {
        self.inner
            .as_deref()
            .ok_or_else(|| texture_error("texture allocation failed or unsupported descriptor"))
    }

    fn validate_origin_extent(
        &self,
        origin: crate::HalOrigin3d,
        extent: HalExtent3d,
    ) -> Result<(), HalError> {
        let x_end = origin
            .x
            .checked_add(extent.width)
            .ok_or_else(|| texture_error("texture x range overflows"))?;
        let y_end = origin
            .y
            .checked_add(extent.height)
            .ok_or_else(|| texture_error("texture y range overflows"))?;
        let z_end = origin
            .z
            .checked_add(extent.depth_or_array_layers)
            .ok_or_else(|| texture_error("texture z range overflows"))?;
        if x_end > self.width || y_end > self.height || z_end > self.depth_or_array_layers {
            return Err(texture_error("texture range exceeds texture size"));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct VulkanTextureInner {
    device: Arc<VulkanDeviceInner>,
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    layout: AtomicU8,
}

impl Drop for VulkanTextureInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_image_view(self.view, None);
            self.device.device.destroy_image(self.image, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}

#[derive(Debug, Clone)]
pub struct VulkanSampler {
    _inner: Option<Arc<VulkanSamplerInner>>,
}

#[derive(Debug)]
struct VulkanSamplerInner {
    device: Arc<VulkanDeviceInner>,
    sampler: vk::Sampler,
}

impl Drop for VulkanSamplerInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_sampler(self.sampler, None);
        }
    }
}

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

fn create_texture(
    device: Arc<VulkanDeviceInner>,
    descriptor: &HalTextureDescriptor,
) -> Result<(VulkanTextureInner, u32), HalError> {
    if descriptor.depth_or_array_layers != 1
        || descriptor.mip_level_count != 1
        || descriptor.sample_count != 1
    {
        return Err(texture_error("unsupported texture descriptor"));
    }
    let (format, bytes_per_pixel) = map_texture_format(descriptor.format)?;
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: descriptor.width,
            height: descriptor.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(map_texture_usage(descriptor.usage))
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.device.create_image(&image_info, None) }
        .map_err(|_| texture_error("image creation failed"))?;
    let requirements = unsafe { device.device.get_image_memory_requirements(image) };
    let memory_type_index = find_memory_type_index(
        &device.memory_properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| {
        unsafe {
            device.device.destroy_image(image, None);
        }
        texture_error("compatible image memory type not found")
    })?;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory = unsafe { device.device.allocate_memory(&allocate_info, None) }.map_err(|_| {
        unsafe {
            device.device.destroy_image(image, None);
        }
        texture_error("image memory allocation failed")
    })?;
    if let Err(error) = unsafe { device.device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.device.destroy_image(image, None);
            device.device.free_memory(memory, None);
        }
        return Err(map_texture_error(error, "image memory bind failed"));
    }
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(color_subresource_range());
    let view = unsafe { device.device.create_image_view(&view_info, None) }.map_err(|_| {
        unsafe {
            device.device.destroy_image(image, None);
            device.device.free_memory(memory, None);
        }
        texture_error("image view creation failed")
    })?;
    Ok((
        VulkanTextureInner {
            device,
            image,
            view,
            memory,
            layout: AtomicU8::new(IMAGE_LAYOUT_UNDEFINED),
        },
        bytes_per_pixel,
    ))
}

fn create_sampler(
    device: Arc<VulkanDeviceInner>,
    descriptor: &HalSamplerDescriptor,
) -> Result<VulkanSamplerInner, HalError> {
    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(map_filter_mode(descriptor.mag_filter))
        .min_filter(map_filter_mode(descriptor.min_filter))
        .mipmap_mode(map_mipmap_filter_mode(descriptor.mipmap_filter))
        .address_mode_u(map_address_mode(descriptor.address_mode_u))
        .address_mode_v(map_address_mode(descriptor.address_mode_v))
        .address_mode_w(map_address_mode(descriptor.address_mode_w))
        .mip_lod_bias(0.0)
        .anisotropy_enable(descriptor.max_anisotropy > 1)
        .max_anisotropy(f32::from(descriptor.max_anisotropy))
        .compare_enable(descriptor.compare.is_some())
        .compare_op(
            descriptor
                .compare
                .map_or(vk::CompareOp::ALWAYS, map_compare_function),
        )
        .min_lod(descriptor.lod_min_clamp)
        .max_lod(descriptor.lod_max_clamp)
        .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
        .unnormalized_coordinates(false);
    let sampler = unsafe { device.device.create_sampler(&sampler_info, None) }
        .map_err(|_| texture_error("sampler creation failed"))?;
    Ok(VulkanSamplerInner { device, sampler })
}

fn submit_copies(queue: &VulkanQueueInner, copies: &[HalCopy]) -> Result<(), HalError> {
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
    let result = record_and_submit_copies(queue, command_pool, copies);
    unsafe {
        queue.device.device.destroy_command_pool(command_pool, None);
    }
    result
}

fn record_and_submit_copies(
    queue: &VulkanQueueInner,
    command_pool: vk::CommandPool,
    copies: &[HalCopy],
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
        match copy {
            HalCopy::Buffer(copy) => {
                encode_buffer_copy(&queue.device.device, command_buffer, copy)?;
            }
            HalCopy::BufferToTexture(copy) => {
                encode_buffer_to_texture(&queue.device.device, command_buffer, copy)?;
            }
            HalCopy::TextureToBuffer(copy) => {
                encode_texture_to_buffer(&queue.device.device, command_buffer, copy)?;
            }
            HalCopy::TextureToTexture(copy) => {
                encode_texture_to_texture(&queue.device.device, command_buffer, copy)?;
            }
            HalCopy::ComputePass(_) | HalCopy::RenderPass(_) => {
                return Err(HalError::BackendUnavailable { backend: BACKEND });
            }
        }
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

fn encode_buffer_to_texture(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let crate::HalBuffer::Vulkan(buffer) = &copy.buffer else {
        return Err(buffer_error("buffer is not Vulkan-backed"));
    };
    let crate::HalTexture::Vulkan(texture) = &copy.texture else {
        return Err(texture_error("texture is not Vulkan-backed"));
    };
    validate_mip_level(copy.mip_level)?;
    texture.validate_origin_extent(copy.origin, copy.extent)?;
    validate_buffer_texture_range(buffer, copy)?;
    let buffer = buffer.inner()?;
    let texture_inner = texture.inner()?;
    transition_image(
        device,
        command_buffer,
        texture_inner,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_DST,
    );
    let region = buffer_image_copy(copy, texture.bytes_per_pixel)?;
    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer.buffer,
            texture_inner.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    Ok(())
}

fn encode_texture_to_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let crate::HalBuffer::Vulkan(buffer) = &copy.buffer else {
        return Err(buffer_error("buffer is not Vulkan-backed"));
    };
    let crate::HalTexture::Vulkan(texture) = &copy.texture else {
        return Err(texture_error("texture is not Vulkan-backed"));
    };
    validate_mip_level(copy.mip_level)?;
    texture.validate_origin_extent(copy.origin, copy.extent)?;
    validate_buffer_texture_range(buffer, copy)?;
    let buffer = buffer.inner()?;
    let texture_inner = texture.inner()?;
    transition_image(
        device,
        command_buffer,
        texture_inner,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_SRC,
    );
    let region = buffer_image_copy(copy, texture.bytes_per_pixel)?;
    unsafe {
        device.cmd_copy_image_to_buffer(
            command_buffer,
            texture_inner.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer.buffer,
            &[region],
        );
    }
    Ok(())
}

fn encode_texture_to_texture(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    copy: &HalTextureCopy,
) -> Result<(), HalError> {
    let crate::HalTexture::Vulkan(source) = &copy.source else {
        return Err(texture_error("source texture is not Vulkan-backed"));
    };
    let crate::HalTexture::Vulkan(destination) = &copy.destination else {
        return Err(texture_error("destination texture is not Vulkan-backed"));
    };
    validate_mip_level(copy.source_mip_level)?;
    validate_mip_level(copy.destination_mip_level)?;
    source.validate_origin_extent(copy.source_origin, copy.extent)?;
    destination.validate_origin_extent(copy.destination_origin, copy.extent)?;
    let source_inner = source.inner()?;
    let destination_inner = destination.inner()?;
    transition_image(
        device,
        command_buffer,
        source_inner,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_SRC,
    );
    transition_image(
        device,
        command_buffer,
        destination_inner,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_DST,
    );
    let region = vk::ImageCopy::default()
        .src_subresource(image_subresource_layers())
        .src_offset(to_image_offset(
            copy.source_origin.x,
            copy.source_origin.y,
            copy.source_origin.z,
        )?)
        .dst_subresource(image_subresource_layers())
        .dst_offset(to_image_offset(
            copy.destination_origin.x,
            copy.destination_origin.y,
            copy.destination_origin.z,
        )?)
        .extent(to_image_extent(copy.extent));
    unsafe {
        device.cmd_copy_image(
            command_buffer,
            source_inner.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            destination_inner.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    Ok(())
}

fn transition_image(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    texture: &VulkanTextureInner,
    new_layout: vk::ImageLayout,
    new_state: u8,
) {
    let old_state = texture.layout.swap(new_state, AtomicOrdering::Relaxed);
    let old_layout = image_layout(old_state);
    if old_layout == new_layout {
        return;
    }
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(texture.image)
        .subresource_range(color_subresource_range())
        .src_access_mask(access_mask_for_layout(old_layout))
        .dst_access_mask(access_mask_for_layout(new_layout));
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn image_layout(state: u8) -> vk::ImageLayout {
    match state {
        IMAGE_LAYOUT_TRANSFER_DST => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_SRC => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        _ => vk::ImageLayout::UNDEFINED,
    }
}

fn access_mask_for_layout(layout: vk::ImageLayout) -> vk::AccessFlags {
    match layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::AccessFlags::TRANSFER_READ,
        _ => vk::AccessFlags::empty(),
    }
}

fn validate_buffer_texture_range(
    buffer: &VulkanBuffer,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let rows = u64::from(copy.extent.height.saturating_sub(1));
    let last_row = rows
        .checked_mul(u64::from(copy.buffer_layout.bytes_per_row))
        .ok_or_else(|| buffer_error("buffer texture row range overflows"))?;
    let row_bytes = u64::from(copy.extent.width)
        .checked_mul(u64::from(texture_bytes_per_pixel(copy)?))
        .ok_or_else(|| buffer_error("buffer texture row bytes overflow"))?;
    let required = copy
        .buffer_layout
        .offset
        .checked_add(last_row)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or_else(|| buffer_error("buffer texture range overflows"))?;
    if required > buffer.size() {
        return Err(buffer_error("buffer texture range exceeds buffer size"));
    }
    Ok(())
}

fn texture_bytes_per_pixel(copy: &HalBufferTextureCopy) -> Result<u32, HalError> {
    let crate::HalTexture::Vulkan(texture) = &copy.texture else {
        return Err(texture_error("texture is not Vulkan-backed"));
    };
    if texture.bytes_per_pixel == 0 {
        return Err(texture_error("unsupported texture format"));
    }
    Ok(texture.bytes_per_pixel)
}

fn buffer_image_copy(
    copy: &HalBufferTextureCopy,
    bytes_per_pixel: u32,
) -> Result<vk::BufferImageCopy, HalError> {
    let buffer_row_length = buffer_row_length(copy.buffer_layout.bytes_per_row, bytes_per_pixel)?;
    Ok(vk::BufferImageCopy::default()
        .buffer_offset(copy.buffer_layout.offset)
        .buffer_row_length(buffer_row_length)
        .buffer_image_height(copy.buffer_layout.rows_per_image)
        .image_subresource(image_subresource_layers())
        .image_offset(to_image_offset(
            copy.origin.x,
            copy.origin.y,
            copy.origin.z,
        )?)
        .image_extent(to_image_extent(copy.extent)))
}

fn validate_mip_level(mip_level: u32) -> Result<(), HalError> {
    if mip_level != 0 {
        return Err(texture_error("unsupported texture mip level"));
    }
    Ok(())
}

fn buffer_row_length(bytes_per_row: u32, bytes_per_pixel: u32) -> Result<u32, HalError> {
    if bytes_per_row == 0 {
        return Ok(0);
    }
    if bytes_per_pixel == 0 || !bytes_per_row.is_multiple_of(bytes_per_pixel) {
        return Err(buffer_error(
            "buffer texture bytes per row is not texel-aligned",
        ));
    }
    Ok(bytes_per_row / bytes_per_pixel)
}

fn image_subresource_layers() -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
}

fn to_image_offset(x: u32, y: u32, z: u32) -> Result<vk::Offset3D, HalError> {
    Ok(vk::Offset3D {
        x: i32::try_from(x).map_err(|_| texture_error("texture x offset is too large"))?,
        y: i32::try_from(y).map_err(|_| texture_error("texture y offset is too large"))?,
        z: i32::try_from(z).map_err(|_| texture_error("texture z offset is too large"))?,
    })
}

fn to_image_extent(extent: HalExtent3d) -> vk::Extent3D {
    vk::Extent3D {
        width: extent.width,
        height: extent.height,
        depth: extent.depth_or_array_layers,
    }
}

fn map_texture_format(format: HalTextureFormat) -> Result<(vk::Format, u32), HalError> {
    match format {
        HalTextureFormat::R8Unorm => Ok((vk::Format::R8_UNORM, 1)),
        HalTextureFormat::Rgba8Unorm => Ok((vk::Format::R8G8B8A8_UNORM, 4)),
        HalTextureFormat::Bgra8Unorm => Ok((vk::Format::B8G8R8A8_UNORM, 4)),
        HalTextureFormat::Unsupported => Err(texture_error("unsupported texture format")),
    }
}

fn map_texture_usage(usage: HalTextureUsage) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    if usage.texture_binding {
        flags |= vk::ImageUsageFlags::SAMPLED;
    }
    if usage.storage_binding {
        flags |= vk::ImageUsageFlags::STORAGE;
    }
    if usage.render_attachment {
        flags |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
    }
    flags
}

fn map_address_mode(mode: HalAddressMode) -> vk::SamplerAddressMode {
    match mode {
        HalAddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        HalAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        HalAddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

fn map_filter_mode(mode: HalFilterMode) -> vk::Filter {
    match mode {
        HalFilterMode::Nearest => vk::Filter::NEAREST,
        HalFilterMode::Linear => vk::Filter::LINEAR,
    }
}

fn map_mipmap_filter_mode(mode: HalMipmapFilterMode) -> vk::SamplerMipmapMode {
    match mode {
        HalMipmapFilterMode::Nearest => vk::SamplerMipmapMode::NEAREST,
        HalMipmapFilterMode::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

fn map_compare_function(compare: HalCompareFunction) -> vk::CompareOp {
    match compare {
        HalCompareFunction::Never => vk::CompareOp::NEVER,
        HalCompareFunction::Less => vk::CompareOp::LESS,
        HalCompareFunction::Equal => vk::CompareOp::EQUAL,
        HalCompareFunction::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        HalCompareFunction::Greater => vk::CompareOp::GREATER,
        HalCompareFunction::NotEqual => vk::CompareOp::NOT_EQUAL,
        HalCompareFunction::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        HalCompareFunction::Always => vk::CompareOp::ALWAYS,
    }
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

fn texture_error(message: &'static str) -> HalError {
    HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    }
}

fn map_texture_error(_error: vk::Result, message: &'static str) -> HalError {
    texture_error(message)
}
