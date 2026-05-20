use std::ffi::{CStr, CString};
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
    HalRenderPipelineDescriptor, HalSamplerDescriptor, HalShaderSource, HalTextureCopy,
    HalTextureDescriptor, HalTextureFormat, HalTextureUsage, HalVertexFormat, HalVertexStepMode,
};

const BACKEND: &str = "vulkan";
const IMAGE_LAYOUT_UNDEFINED: u8 = 0;
const IMAGE_LAYOUT_TRANSFER_DST: u8 = 1;
const IMAGE_LAYOUT_TRANSFER_SRC: u8 = 2;
const IMAGE_LAYOUT_COLOR_ATTACHMENT: u8 = 3;

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

    pub fn create_compute_pipeline(
        &self,
        shader: HalShaderSource,
        entry_point: &str,
        _workgroup_size: (u32, u32, u32),
        bindings: &[HalDescriptorBinding],
    ) -> Result<VulkanComputePipeline, HalError> {
        create_compute_pipeline(Arc::clone(&self.inner), shader, entry_point, bindings)
    }

    pub fn create_render_pipeline(
        &self,
        shader: HalShaderSource,
        vertex_entry_point: &str,
        fragment_entry_point: &str,
        descriptor: &HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
    ) -> Result<VulkanRenderPipeline, HalError> {
        create_render_pipeline(
            Arc::clone(&self.inner),
            shader,
            vertex_entry_point,
            fragment_entry_point,
            descriptor,
            bindings,
        )
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

    #[must_use]
    pub fn mapped_ptr(&self) -> Option<NonNull<u8>> {
        self.inner
            .as_ref()
            .and_then(|inner| NonNull::new(inner.mapped))
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
pub struct VulkanComputePipeline {
    inner: Arc<VulkanComputePipelineInner>,
}

#[derive(Debug)]
struct VulkanComputePipelineInner {
    device: Arc<VulkanDeviceInner>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_bindings: Vec<HalDescriptorBinding>,
    shader_module: vk::ShaderModule,
}

impl Drop for VulkanComputePipelineInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            for layout in &self.descriptor_set_layouts {
                self.device
                    .device
                    .destroy_descriptor_set_layout(*layout, None);
            }
            self.device
                .device
                .destroy_shader_module(self.shader_module, None);
        }
    }
}

#[derive(Debug, Clone)]
pub struct VulkanRenderPipeline {
    inner: Arc<VulkanRenderPipelineInner>,
}

#[derive(Debug)]
struct VulkanRenderPipelineInner {
    device: Arc<VulkanDeviceInner>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_bindings: Vec<HalDescriptorBinding>,
    vertex_shader_module: vk::ShaderModule,
    fragment_shader_module: vk::ShaderModule,
}

impl Drop for VulkanRenderPipelineInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(self.pipeline, None);
            self.device
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .device
                .destroy_render_pass(self.render_pass, None);
            for layout in &self.descriptor_set_layouts {
                self.device
                    .device
                    .destroy_descriptor_set_layout(*layout, None);
            }
            self.device
                .device
                .destroy_shader_module(self.fragment_shader_module, None);
            self.device
                .device
                .destroy_shader_module(self.vertex_shader_module, None);
        }
    }
}

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

fn create_compute_pipeline(
    device: Arc<VulkanDeviceInner>,
    shader: HalShaderSource,
    entry_point: &str,
    bindings: &[HalDescriptorBinding],
) -> Result<VulkanComputePipeline, HalError> {
    let HalShaderSource::SpirV(code) = shader else {
        return Err(shader_error("Vulkan compute pipeline requires SPIR-V"));
    };
    let entry_point =
        CString::new(entry_point).map_err(|_| shader_error("compute entry point contains NUL"))?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(&code);
    let shader_module = unsafe { device.device.create_shader_module(&shader_info, None) }
        .map_err(|_| shader_error("shader module creation failed"))?;
    let descriptor_set_layouts =
        match create_descriptor_set_layouts(&device, bindings, vk::ShaderStageFlags::COMPUTE) {
            Ok(layouts) => layouts,
            Err(error) => {
                unsafe {
                    device.device.destroy_shader_module(shader_module, None);
                }
                return Err(error);
            }
        };
    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);
    let pipeline_layout = match unsafe {
        device
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)
    } {
        Ok(layout) => layout,
        Err(_) => {
            unsafe {
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                device.device.destroy_shader_module(shader_module, None);
            }
            return Err(shader_error("pipeline layout creation failed"));
        }
    };
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_point);
    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);
    let pipelines = match unsafe {
        device
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    } {
        Ok(pipelines) => pipelines,
        Err((pipelines, _)) => {
            unsafe {
                for pipeline in pipelines {
                    device.device.destroy_pipeline(pipeline, None);
                }
                device.device.destroy_pipeline_layout(pipeline_layout, None);
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                device.device.destroy_shader_module(shader_module, None);
            }
            return Err(shader_error("compute pipeline creation failed"));
        }
    };
    let Some(&pipeline) = pipelines.first() else {
        unsafe {
            device.device.destroy_pipeline_layout(pipeline_layout, None);
            destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
            device.device.destroy_shader_module(shader_module, None);
        }
        return Err(shader_error(
            "compute pipeline creation returned no pipeline",
        ));
    };
    Ok(VulkanComputePipeline {
        inner: Arc::new(VulkanComputePipelineInner {
            device,
            pipeline,
            pipeline_layout,
            descriptor_set_layouts,
            descriptor_bindings: bindings.to_vec(),
            shader_module,
        }),
    })
}

fn create_render_pipeline(
    device: Arc<VulkanDeviceInner>,
    shader: HalShaderSource,
    vertex_entry_point: &str,
    fragment_entry_point: &str,
    descriptor: &HalRenderPipelineDescriptor,
    bindings: &[HalDescriptorBinding],
) -> Result<VulkanRenderPipeline, HalError> {
    let HalShaderSource::SpirVStages { vertex, fragment } = shader else {
        return Err(shader_error(
            "Vulkan render pipeline requires vertex and fragment SPIR-V",
        ));
    };
    let vertex_entry = CString::new(vertex_entry_point)
        .map_err(|_| shader_error("vertex entry point contains NUL"))?;
    let fragment_entry = CString::new(fragment_entry_point)
        .map_err(|_| shader_error("fragment entry point contains NUL"))?;
    let vertex_shader_module = create_shader_module(&device, &vertex)?;
    let fragment_shader_module = match create_shader_module(&device, &fragment) {
        Ok(module) => module,
        Err(error) => {
            unsafe {
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(error);
        }
    };
    let descriptor_set_layouts = match create_descriptor_set_layouts(
        &device,
        bindings,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
    ) {
        Ok(layouts) => layouts,
        Err(error) => {
            unsafe {
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(error);
        }
    };
    let pipeline_layout_info =
        vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);
    let pipeline_layout = match unsafe {
        device
            .device
            .create_pipeline_layout(&pipeline_layout_info, None)
    } {
        Ok(layout) => layout,
        Err(_) => {
            unsafe {
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(shader_error("render pipeline layout creation failed"));
        }
    };
    let render_pass = match create_render_pass(&device, descriptor) {
        Ok(render_pass) => render_pass,
        Err(error) => {
            unsafe {
                device.device.destroy_pipeline_layout(pipeline_layout, None);
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(error);
        }
    };
    let pipeline = match create_graphics_pipeline(
        &device,
        descriptor,
        pipeline_layout,
        render_pass,
        vertex_shader_module,
        fragment_shader_module,
        &vertex_entry,
        &fragment_entry,
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            unsafe {
                device.device.destroy_render_pass(render_pass, None);
                device.device.destroy_pipeline_layout(pipeline_layout, None);
                destroy_descriptor_set_layouts(&device.device, &descriptor_set_layouts);
                device
                    .device
                    .destroy_shader_module(fragment_shader_module, None);
                device
                    .device
                    .destroy_shader_module(vertex_shader_module, None);
            }
            return Err(error);
        }
    };
    Ok(VulkanRenderPipeline {
        inner: Arc::new(VulkanRenderPipelineInner {
            device,
            pipeline,
            pipeline_layout,
            render_pass,
            descriptor_set_layouts,
            descriptor_bindings: bindings.to_vec(),
            vertex_shader_module,
            fragment_shader_module,
        }),
    })
}

fn create_shader_module(
    device: &VulkanDeviceInner,
    code: &[u32],
) -> Result<vk::ShaderModule, HalError> {
    let shader_info = vk::ShaderModuleCreateInfo::default().code(code);
    unsafe { device.device.create_shader_module(&shader_info, None) }
        .map_err(|_| shader_error("shader module creation failed"))
}

fn create_render_pass(
    device: &VulkanDeviceInner,
    descriptor: &HalRenderPipelineDescriptor,
) -> Result<vk::RenderPass, HalError> {
    let color_format = descriptor
        .color_formats
        .first()
        .copied()
        .ok_or_else(|| shader_error("render pipeline requires a color target"))?;
    let (format, _) = map_texture_format(color_format)?;
    let attachment = vk::AttachmentDescription::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    let color_reference = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_references = [color_reference];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_references);
    let dependency_in = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
    let dependency_out = vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    let attachments = [attachment];
    let subpasses = [subpass];
    let dependencies = [dependency_in, dependency_out];
    let render_pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    unsafe { device.device.create_render_pass(&render_pass_info, None) }
        .map_err(|_| shader_error("render pass creation failed"))
}

#[allow(clippy::too_many_arguments)]
fn create_graphics_pipeline(
    device: &VulkanDeviceInner,
    descriptor: &HalRenderPipelineDescriptor,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vertex_shader_module: vk::ShaderModule,
    fragment_shader_module: vk::ShaderModule,
    vertex_entry: &CStr,
    fragment_entry: &CStr,
) -> Result<vk::Pipeline, HalError> {
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(vertex_entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(fragment_entry),
    ];
    let binding_descriptions = descriptor
        .vertex_buffers
        .iter()
        .enumerate()
        .map(|(slot, layout)| {
            let slot =
                u32::try_from(slot).map_err(|_| shader_error("vertex buffer slot is too large"))?;
            Ok(vk::VertexInputBindingDescription::default()
                .binding(slot)
                .stride(
                    u32::try_from(layout.array_stride)
                        .map_err(|_| shader_error("vertex array stride is too large"))?,
                )
                .input_rate(match layout.step_mode {
                    HalVertexStepMode::Vertex => vk::VertexInputRate::VERTEX,
                    HalVertexStepMode::Instance => vk::VertexInputRate::INSTANCE,
                }))
        })
        .collect::<Result<Vec<_>, HalError>>()?;
    let mut attribute_descriptions = Vec::new();
    for (slot, layout) in descriptor.vertex_buffers.iter().enumerate() {
        let slot =
            u32::try_from(slot).map_err(|_| shader_error("vertex buffer slot is too large"))?;
        for attribute in &layout.attributes {
            attribute_descriptions.push(
                vk::VertexInputAttributeDescription::default()
                    .location(attribute.shader_location)
                    .binding(slot)
                    .format(map_vertex_format(attribute.format)?)
                    .offset(
                        u32::try_from(attribute.offset)
                            .map_err(|_| shader_error("vertex attribute offset is too large"))?,
                    ),
            );
        }
    }
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(map_primitive_topology(descriptor.primitive_topology))
        .primitive_restart_enable(false);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false);
    let color_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        );
    let color_attachments = [color_attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&color_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipelines = unsafe {
        device
            .device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    };
    match pipelines {
        Ok(pipelines) => pipelines
            .first()
            .copied()
            .ok_or_else(|| shader_error("graphics pipeline creation returned no pipeline")),
        Err((pipelines, _)) => {
            unsafe {
                for pipeline in pipelines {
                    device.device.destroy_pipeline(pipeline, None);
                }
            }
            Err(shader_error("graphics pipeline creation failed"))
        }
    }
}

fn create_descriptor_set_layouts(
    device: &VulkanDeviceInner,
    bindings: &[HalDescriptorBinding],
    stage_flags: vk::ShaderStageFlags,
) -> Result<Vec<vk::DescriptorSetLayout>, HalError> {
    let Some(max_group) = bindings.iter().map(|binding| binding.group).max() else {
        return Ok(Vec::new());
    };
    let mut layouts = Vec::new();
    for group in 0..=max_group {
        let layout_bindings = bindings
            .iter()
            .filter(|binding| binding.group == group)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding.binding)
                    .descriptor_type(descriptor_type(binding.kind))
                    .descriptor_count(1)
                    .stage_flags(stage_flags)
            })
            .collect::<Vec<_>>();
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        match unsafe {
            device
                .device
                .create_descriptor_set_layout(&layout_info, None)
        } {
            Ok(layout) => layouts.push(layout),
            Err(_) => {
                unsafe {
                    destroy_descriptor_set_layouts(&device.device, &layouts);
                }
                return Err(shader_error("descriptor set layout creation failed"));
            }
        }
    }
    Ok(layouts)
}

unsafe fn destroy_descriptor_set_layouts(
    device: &ash::Device,
    layouts: &[vk::DescriptorSetLayout],
) {
    for layout in layouts {
        device.destroy_descriptor_set_layout(*layout, None);
    }
}

fn descriptor_type(kind: HalBufferBindingKind) -> vk::DescriptorType {
    match kind {
        HalBufferBindingKind::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
        HalBufferBindingKind::Storage => vk::DescriptorType::STORAGE_BUFFER,
    }
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
    let mut descriptor_pools = Vec::new();
    let mut framebuffers = Vec::new();
    let result = (|| {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffers =
            unsafe { queue.device.device.allocate_command_buffers(&allocate_info) }
                .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
        let Some(&command_buffer) = command_buffers.first() else {
            return Err(HalError::QueueSubmissionFailed { backend: BACKEND });
        };
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
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
                HalCopy::ComputePass(pass) => {
                    if let Some(pool) =
                        encode_compute_pass(&queue.device.device, command_buffer, pass)?
                    {
                        descriptor_pools.push(pool);
                    }
                }
                HalCopy::RenderPass(pass) => {
                    let temps = encode_render_pass(&queue.device.device, command_buffer, pass)?;
                    if let Some(pool) = temps.descriptor_pool {
                        descriptor_pools.push(pool);
                    }
                    framebuffers.push(temps.framebuffer);
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
    })();
    unsafe {
        for framebuffer in framebuffers {
            queue.device.device.destroy_framebuffer(framebuffer, None);
        }
        for pool in descriptor_pools {
            queue.device.device.destroy_descriptor_pool(pool, None);
        }
    }
    result
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
    transfer_to_compute_barrier(device, command_buffer);
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

fn encode_compute_pass(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pass: &HalComputePass,
) -> Result<Option<vk::DescriptorPool>, HalError> {
    let crate::HalComputePipeline::Vulkan(pipeline) = &pass.pipeline else {
        return Err(shader_error("compute pipeline is not Vulkan-backed"));
    };
    let descriptor_pool = create_compute_descriptor_pool(device, pipeline)?;
    let descriptor_sets = if let Some(pool) = descriptor_pool {
        match allocate_compute_descriptor_sets(device, pool, pipeline) {
            Ok(sets) => sets,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                }
                return Err(error);
            }
        }
    } else {
        Vec::new()
    };
    if let Err(error) = update_compute_descriptor_sets(device, pipeline, pass, &descriptor_sets) {
        if let Some(pool) = descriptor_pool {
            unsafe {
                device.destroy_descriptor_pool(pool, None);
            }
        }
        return Err(error);
    }
    unsafe {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline.inner.pipeline,
        );
        if !descriptor_sets.is_empty() {
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.inner.pipeline_layout,
                0,
                &descriptor_sets,
                &[],
            );
        }
        device.cmd_dispatch(
            command_buffer,
            pass.workgroups.0,
            pass.workgroups.1,
            pass.workgroups.2,
        );
    }
    compute_to_transfer_barrier(device, command_buffer);
    Ok(descriptor_pool)
}

struct RenderPassTemps {
    descriptor_pool: Option<vk::DescriptorPool>,
    framebuffer: vk::Framebuffer,
}

fn encode_render_pass(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pass: &HalRenderPass,
) -> Result<RenderPassTemps, HalError> {
    let crate::HalRenderPipeline::Vulkan(pipeline) = &pass.pipeline else {
        return Err(shader_error("render pipeline is not Vulkan-backed"));
    };
    let crate::HalTexture::Vulkan(texture) = &pass.color_target.texture else {
        return Err(texture_error("render target is not Vulkan-backed"));
    };
    if !matches!(pass.color_target.load_op, HalRenderLoadOp::Clear) {
        return Err(shader_error("Vulkan render pass load op is unsupported"));
    }
    if !pass.color_target.store {
        return Err(shader_error(
            "Vulkan render pass discard store op is unsupported",
        ));
    }
    let texture_inner = texture.inner()?;
    transition_image(
        device,
        command_buffer,
        texture_inner,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        IMAGE_LAYOUT_COLOR_ATTACHMENT,
    );
    let framebuffer = create_framebuffer(device, pipeline, texture)?;
    let descriptor_pool = create_render_descriptor_pool(device, pipeline)?;
    let descriptor_sets = if let Some(pool) = descriptor_pool {
        match allocate_render_descriptor_sets(device, pool, pipeline) {
            Ok(sets) => sets,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_pool(pool, None);
                    device.destroy_framebuffer(framebuffer, None);
                }
                return Err(error);
            }
        }
    } else {
        Vec::new()
    };
    if let Err(error) = update_render_descriptor_sets(device, pipeline, pass, &descriptor_sets) {
        unsafe {
            if let Some(pool) = descriptor_pool {
                device.destroy_descriptor_pool(pool, None);
            }
            device.destroy_framebuffer(framebuffer, None);
        }
        return Err(error);
    }
    let clear_values = [vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [
                pass.color_target.clear_color[0] as f32,
                pass.color_target.clear_color[1] as f32,
                pass.color_target.clear_color[2] as f32,
                pass.color_target.clear_color[3] as f32,
            ],
        },
    }];
    let render_area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: texture.width,
            height: texture.height,
        },
    };
    let begin_info = vk::RenderPassBeginInfo::default()
        .render_pass(pipeline.inner.render_pass)
        .framebuffer(framebuffer)
        .render_area(render_area)
        .clear_values(&clear_values);
    unsafe {
        device.cmd_begin_render_pass(command_buffer, &begin_info, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.inner.pipeline,
        );
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: texture.width as f32,
            height: texture.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
        device.cmd_set_scissor(command_buffer, 0, &[render_area]);
        bind_render_descriptor_sets(device, command_buffer, pipeline, &descriptor_sets);
    }
    bind_vertex_buffers(device, command_buffer, pass)?;
    unsafe {
        device.cmd_draw(
            command_buffer,
            pass.draw.vertex_count,
            pass.draw.instance_count,
            pass.draw.first_vertex,
            pass.draw.first_instance,
        );
        device.cmd_end_render_pass(command_buffer);
    }
    texture_inner
        .layout
        .store(IMAGE_LAYOUT_TRANSFER_SRC, AtomicOrdering::Relaxed);
    Ok(RenderPassTemps {
        descriptor_pool,
        framebuffer,
    })
}

fn create_framebuffer(
    device: &ash::Device,
    pipeline: &VulkanRenderPipeline,
    texture: &VulkanTexture,
) -> Result<vk::Framebuffer, HalError> {
    let inner = texture.inner()?;
    let attachments = [inner.view];
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(pipeline.inner.render_pass)
        .attachments(&attachments)
        .width(texture.width)
        .height(texture.height)
        .layers(1);
    unsafe { device.create_framebuffer(&framebuffer_info, None) }
        .map_err(|_| shader_error("framebuffer creation failed"))
}

fn create_compute_descriptor_pool(
    device: &ash::Device,
    pipeline: &VulkanComputePipeline,
) -> Result<Option<vk::DescriptorPool>, HalError> {
    if pipeline.inner.descriptor_set_layouts.is_empty() {
        return Ok(None);
    }
    let uniform_count = pipeline
        .inner
        .descriptor_bindings
        .iter()
        .filter(|binding| matches!(binding.kind, HalBufferBindingKind::Uniform))
        .count();
    let storage_count = pipeline
        .inner
        .descriptor_bindings
        .iter()
        .filter(|binding| matches!(binding.kind, HalBufferBindingKind::Storage))
        .count();
    let mut pool_sizes = Vec::new();
    if uniform_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(
                    u32::try_from(uniform_count)
                        .map_err(|_| shader_error("uniform descriptor count is too large"))?,
                ),
        );
    }
    if storage_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(
                    u32::try_from(storage_count)
                        .map_err(|_| shader_error("storage descriptor count is too large"))?,
                ),
        );
    }
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(
            u32::try_from(pipeline.inner.descriptor_set_layouts.len())
                .map_err(|_| shader_error("descriptor set count is too large"))?,
        )
        .pool_sizes(&pool_sizes);
    let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .map_err(|_| shader_error("descriptor pool creation failed"))?;
    Ok(Some(pool))
}

fn allocate_compute_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    pipeline: &VulkanComputePipeline,
) -> Result<Vec<vk::DescriptorSet>, HalError> {
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&pipeline.inner.descriptor_set_layouts);
    unsafe { device.allocate_descriptor_sets(&allocate_info) }
        .map_err(|_| shader_error("descriptor set allocation failed"))
}

fn update_compute_descriptor_sets(
    device: &ash::Device,
    pipeline: &VulkanComputePipeline,
    pass: &HalComputePass,
    descriptor_sets: &[vk::DescriptorSet],
) -> Result<(), HalError> {
    if pipeline.inner.descriptor_bindings.is_empty() {
        return Ok(());
    }
    let mut buffer_infos = Vec::new();
    let mut write_specs = Vec::new();
    for descriptor in &pipeline.inner.descriptor_bindings {
        let bound = pass
            .bind_buffers
            .iter()
            .find(|bound| bound.group == descriptor.group && bound.binding == descriptor.binding)
            .ok_or_else(|| shader_error("compute descriptor binding is missing"))?;
        let buffer_info = descriptor_buffer_info(bound)?;
        buffer_infos.push(buffer_info);
        write_specs.push((
            buffer_infos.len() - 1,
            descriptor.group,
            descriptor.binding,
            descriptor_type(descriptor.kind),
        ));
    }
    let writes = write_specs
        .iter()
        .map(|(info_index, group, binding, descriptor_type)| {
            let group = usize::try_from(*group)
                .map_err(|_| shader_error("descriptor group index is too large"))?;
            let descriptor_set = descriptor_sets
                .get(group)
                .copied()
                .ok_or_else(|| shader_error("descriptor set is missing"))?;
            Ok(vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(*binding)
                .descriptor_type(*descriptor_type)
                .buffer_info(std::slice::from_ref(&buffer_infos[*info_index])))
        })
        .collect::<Result<Vec<_>, HalError>>()?;
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
    Ok(())
}

fn create_render_descriptor_pool(
    device: &ash::Device,
    pipeline: &VulkanRenderPipeline,
) -> Result<Option<vk::DescriptorPool>, HalError> {
    if pipeline.inner.descriptor_set_layouts.is_empty() {
        return Ok(None);
    }
    create_descriptor_pool(
        device,
        pipeline.inner.descriptor_set_layouts.len(),
        &pipeline.inner.descriptor_bindings,
    )
}

fn allocate_render_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    pipeline: &VulkanRenderPipeline,
) -> Result<Vec<vk::DescriptorSet>, HalError> {
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&pipeline.inner.descriptor_set_layouts);
    unsafe { device.allocate_descriptor_sets(&allocate_info) }
        .map_err(|_| shader_error("descriptor set allocation failed"))
}

fn update_render_descriptor_sets(
    device: &ash::Device,
    pipeline: &VulkanRenderPipeline,
    pass: &HalRenderPass,
    descriptor_sets: &[vk::DescriptorSet],
) -> Result<(), HalError> {
    if pipeline.inner.descriptor_bindings.is_empty() {
        return Ok(());
    }
    let mut buffer_infos = Vec::new();
    let mut write_specs = Vec::new();
    for descriptor in &pipeline.inner.descriptor_bindings {
        let bound = pass
            .bind_buffers
            .iter()
            .find(|bound| bound.group == descriptor.group && bound.binding == descriptor.binding)
            .ok_or_else(|| shader_error("render descriptor binding is missing"))?;
        let buffer_info = descriptor_buffer_info(bound)?;
        buffer_infos.push(buffer_info);
        write_specs.push((
            buffer_infos.len() - 1,
            descriptor.group,
            descriptor.binding,
            descriptor_type(descriptor.kind),
        ));
    }
    let writes = write_specs
        .iter()
        .map(|(info_index, group, binding, descriptor_type)| {
            let group = usize::try_from(*group)
                .map_err(|_| shader_error("descriptor group index is too large"))?;
            let descriptor_set = descriptor_sets
                .get(group)
                .copied()
                .ok_or_else(|| shader_error("descriptor set is missing"))?;
            Ok(vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(*binding)
                .descriptor_type(*descriptor_type)
                .buffer_info(std::slice::from_ref(&buffer_infos[*info_index])))
        })
        .collect::<Result<Vec<_>, HalError>>()?;
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
    Ok(())
}

fn bind_render_descriptor_sets(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline: &VulkanRenderPipeline,
    descriptor_sets: &[vk::DescriptorSet],
) {
    if descriptor_sets.is_empty() {
        return;
    }
    unsafe {
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.inner.pipeline_layout,
            0,
            descriptor_sets,
            &[],
        );
    }
}

fn bind_vertex_buffers(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pass: &HalRenderPass,
) -> Result<(), HalError> {
    for bound in &pass.vertex_buffers {
        let crate::HalBuffer::Vulkan(buffer) = &bound.buffer else {
            return Err(buffer_error("vertex buffer is not Vulkan-backed"));
        };
        let inner = buffer.inner()?;
        validate_bound_buffer_range(bound)?;
        let buffers = [inner.buffer];
        let offsets = [bound.offset];
        unsafe {
            device.cmd_bind_vertex_buffers(command_buffer, bound.binding, &buffers, &offsets);
        }
    }
    Ok(())
}

fn validate_bound_buffer_range(bound: &HalBoundBuffer) -> Result<(), HalError> {
    let crate::HalBuffer::Vulkan(buffer) = &bound.buffer else {
        return Err(buffer_error("buffer is not Vulkan-backed"));
    };
    bound_buffer_range(bound, buffer.size()).map(|_| ())
}

fn bound_buffer_range(bound: &HalBoundBuffer, buffer_size: u64) -> Result<u64, HalError> {
    if bound.offset > buffer_size {
        return Err(buffer_error("buffer offset exceeds buffer size"));
    }
    let range = if bound.size == u64::MAX {
        buffer_size
            .checked_sub(bound.offset)
            .ok_or_else(|| buffer_error("buffer range exceeds buffer size"))?
    } else {
        bound.size
    };
    let end = bound
        .offset
        .checked_add(range)
        .ok_or_else(|| buffer_error("buffer range overflows"))?;
    if end > buffer_size {
        return Err(buffer_error("buffer range exceeds buffer size"));
    }
    Ok(range)
}

fn create_descriptor_pool(
    device: &ash::Device,
    descriptor_set_count: usize,
    bindings: &[HalDescriptorBinding],
) -> Result<Option<vk::DescriptorPool>, HalError> {
    if descriptor_set_count == 0 {
        return Ok(None);
    }
    let uniform_count = bindings
        .iter()
        .filter(|binding| matches!(binding.kind, HalBufferBindingKind::Uniform))
        .count();
    let storage_count = bindings
        .iter()
        .filter(|binding| matches!(binding.kind, HalBufferBindingKind::Storage))
        .count();
    let mut pool_sizes = Vec::new();
    if uniform_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(
                    u32::try_from(uniform_count)
                        .map_err(|_| shader_error("uniform descriptor count is too large"))?,
                ),
        );
    }
    if storage_count > 0 {
        pool_sizes.push(
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(
                    u32::try_from(storage_count)
                        .map_err(|_| shader_error("storage descriptor count is too large"))?,
                ),
        );
    }
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(
            u32::try_from(descriptor_set_count)
                .map_err(|_| shader_error("descriptor set count is too large"))?,
        )
        .pool_sizes(&pool_sizes);
    let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .map_err(|_| shader_error("descriptor pool creation failed"))?;
    Ok(Some(pool))
}

fn descriptor_buffer_info(bound: &HalBoundBuffer) -> Result<vk::DescriptorBufferInfo, HalError> {
    let crate::HalBuffer::Vulkan(buffer) = &bound.buffer else {
        return Err(buffer_error("buffer is not Vulkan-backed"));
    };
    let inner = buffer.inner()?;
    let range = bound_buffer_range(bound, buffer.size())?;
    Ok(vk::DescriptorBufferInfo::default()
        .buffer(inner.buffer)
        .offset(bound.offset)
        .range(range))
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
            stage_mask_for_layout(old_layout),
            stage_mask_for_layout(new_layout),
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn transfer_to_compute_barrier(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

fn compute_to_transfer_barrier(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(
            vk::AccessFlags::TRANSFER_READ
                | vk::AccessFlags::TRANSFER_WRITE
                | vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE,
        );
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

fn image_layout(state: u8) -> vk::ImageLayout {
    match state {
        IMAGE_LAYOUT_TRANSFER_DST => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        IMAGE_LAYOUT_TRANSFER_SRC => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        IMAGE_LAYOUT_COLOR_ATTACHMENT => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        _ => vk::ImageLayout::UNDEFINED,
    }
}

fn access_mask_for_layout(layout: vk::ImageLayout) -> vk::AccessFlags {
    match layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::AccessFlags::TRANSFER_READ,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        _ => vk::AccessFlags::empty(),
    }
}

fn stage_mask_for_layout(layout: vk::ImageLayout) -> vk::PipelineStageFlags {
    match layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL | vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
            vk::PipelineStageFlags::TRANSFER
        }
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
        }
        _ => vk::PipelineStageFlags::TOP_OF_PIPE,
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

fn map_vertex_format(format: HalVertexFormat) -> Result<vk::Format, HalError> {
    match format {
        HalVertexFormat::Float32 => Ok(vk::Format::R32_SFLOAT),
        HalVertexFormat::Float32x2 => Ok(vk::Format::R32G32_SFLOAT),
        HalVertexFormat::Float32x3 => Ok(vk::Format::R32G32B32_SFLOAT),
        HalVertexFormat::Float32x4 => Ok(vk::Format::R32G32B32A32_SFLOAT),
        HalVertexFormat::Unsupported => Err(shader_error("unsupported vertex format")),
    }
}

fn map_primitive_topology(topology: HalPrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        HalPrimitiveTopology::PointList => vk::PrimitiveTopology::POINT_LIST,
        HalPrimitiveTopology::LineList => vk::PrimitiveTopology::LINE_LIST,
        HalPrimitiveTopology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
        HalPrimitiveTopology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
        HalPrimitiveTopology::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
    }
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

fn shader_error(message: &'static str) -> HalError {
    HalError::ShaderCompilationFailed {
        backend: BACKEND,
        message: message.to_owned(),
    }
}
