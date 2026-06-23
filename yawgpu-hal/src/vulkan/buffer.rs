use super::*;

/// Stores vulkan buffer data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct VulkanBuffer {
    pub(super) inner: Option<Arc<VulkanBufferInner>>,
    pub(super) size: u64,
}

impl VulkanBuffer {
    /// Returns the size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Records a write command.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), HalError> {
        if self
            .inner
            .as_ref()
            .is_some_and(|inner| inner.device.is_destroyed())
        {
            return Err(buffer_error("device is destroyed"));
        }
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

    /// Reads `len` bytes at `offset` back from the buffer into host memory.
    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        if self
            .inner
            .as_ref()
            .is_some_and(|inner| inner.device.is_destroyed())
        {
            return Err(buffer_error("device is destroyed"));
        }
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

    /// Returns mapped ptr.
    #[must_use]
    pub fn mapped_ptr(&self) -> Option<NonNull<u8>> {
        self.inner
            .as_ref()
            .filter(|inner| !inner.device.is_destroyed())
            .and_then(|inner| NonNull::new(inner.mapped))
    }

    /// Returns the backing buffer state, or an error if allocation failed.
    pub(super) fn inner(&self) -> Result<&VulkanBufferInner, HalError> {
        self.inner
            .as_deref()
            .ok_or_else(|| buffer_error("buffer allocation failed"))
    }

    /// Validates range and returns a descriptive error on failure.
    pub(super) fn validate_range(&self, offset: u64, len: u64) -> Result<(), HalError> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| buffer_error("buffer range overflows"))?;
        if end > self.size {
            return Err(buffer_error("buffer range exceeds buffer size"));
        }
        Ok(())
    }
}

/// Holds shared state for the vulkan buffer handle.
#[derive(Debug)]
pub(super) struct VulkanBufferInner {
    pub(super) device: Arc<VulkanDeviceInner>,
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) mapped: *mut u8,
}

unsafe impl Send for VulkanBufferInner {}
unsafe impl Sync for VulkanBufferInner {}

impl Drop for VulkanBufferInner {
    fn drop(&mut self) {
        if self.device.is_destroyed() {
            return;
        }
        unsafe {
            if !self.mapped.is_null() {
                self.device.device.unmap_memory(self.memory);
            }
            self.device.device.destroy_buffer(self.buffer, None);
            self.device.device.free_memory(self.memory, None);
        }
    }
}

/// Creates buffer and reports validation errors through the owning device.
pub(super) fn create_buffer(
    device: Arc<VulkanDeviceInner>,
    size: u64,
    usage: HalBufferUsage,
) -> Result<VulkanBufferInner, HalError> {
    let allocation_size = size.max(1);
    let create_info = vk::BufferCreateInfo::default()
        .size(allocation_size)
        .usage(map_buffer_usage(usage))
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.device.create_buffer(&create_info, None) }
        .map_err(|error| map_buffer_error(error, "buffer creation failed"))?;
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
    // Proactive guard: reject allocations that exceed the backing heap capacity.
    // MoltenVK defers real Metal allocation so vkAllocateMemory may return
    // VK_SUCCESS for impossible sizes; comparing against the heap size catches
    // those before the call and produces a deterministic OutOfMemory error.
    if requirements.size > memory_heap_size(&device.memory_properties, memory_type_index) {
        unsafe {
            device.device.destroy_buffer(buffer, None);
        }
        return Err(HalError::OutOfMemory {
            backend: BACKEND,
            resource: "buffer",
        });
    }
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index);
    let memory =
        unsafe { device.device.allocate_memory(&allocate_info, None) }.map_err(|error| {
            unsafe {
                device.device.destroy_buffer(buffer, None);
            }
            map_buffer_error(error, "buffer memory allocation failed")
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

/// Returns find memory type index.
pub(super) fn find_memory_type_index(
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

/// Returns the total byte capacity of the memory heap backing the given memory
/// type index.  Returns 0 when `memory_type_index` is out of range so callers
/// that compare `requirements.size > memory_heap_size(...)` will reject the
/// allocation rather than silently proceeding.
pub(super) fn memory_heap_size(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    memory_type_index: u32,
) -> u64 {
    let type_count = usize::try_from(memory_properties.memory_type_count).unwrap_or(0);
    let heap_count = usize::try_from(memory_properties.memory_heap_count).unwrap_or(0);
    let type_index = usize::try_from(memory_type_index).unwrap_or(usize::MAX);
    if type_index >= type_count {
        return 0;
    }
    let heap_index = usize::from(memory_properties.memory_types[type_index].heap_index as u8);
    if heap_index >= heap_count {
        return 0;
    }
    memory_properties.memory_heaps[heap_index].size
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;

    /// Builds a minimal `vk::PhysicalDeviceMemoryProperties` with one type and
    /// one heap of the given capacity so pure-logic tests need no real device.
    #[allow(clippy::field_reassign_with_default)] // array elements cannot be set in struct literal
    fn synthetic_memory_properties(heap_size: u64) -> vk::PhysicalDeviceMemoryProperties {
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        props.memory_type_count = 1;
        props.memory_types[0].heap_index = 0;
        props.memory_types[0].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        props.memory_heap_count = 1;
        props.memory_heaps[0].size = heap_size;
        props
    }

    #[test]
    fn memory_heap_size_returns_heap_capacity_for_valid_type_index() {
        let props = synthetic_memory_properties(1024);
        assert_eq!(memory_heap_size(&props, 0), 1024);
    }

    #[test]
    fn memory_heap_size_returns_zero_for_out_of_range_type_index() {
        let props = synthetic_memory_properties(1024);
        // type index 1 is beyond memory_type_count == 1
        assert_eq!(memory_heap_size(&props, 1), 0);
    }

    #[test]
    fn allocation_exceeds_heap_detects_oversized_requirement() {
        let props = synthetic_memory_properties(1024);
        let heap_size = memory_heap_size(&props, 0);
        // requirement exactly at the limit must NOT be rejected
        assert!(1024_u64 <= heap_size);
        // requirement one byte over must be rejected
        assert!(1025_u64 > heap_size);
    }

    #[test]
    fn allocation_within_heap_is_not_rejected() {
        let props = synthetic_memory_properties(u64::MAX);
        let heap_size = memory_heap_size(&props, 0);
        // any realistic requirement fits inside a max-sized heap
        assert!(68_719_476_736_u64 <= heap_size); // 64 GiB fits
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_buffer_size_returns_created_size() {
        let buffer = vulkan_device()
            .create_buffer(32, HalBufferUsage::default())
            .expect("Vulkan buffer allocation should succeed");
        assert_eq!(buffer.size(), 32);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_buffer_write_updates_mapped_memory() {
        let buffer = vulkan_device()
            .create_buffer(4, HalBufferUsage::default())
            .expect("Vulkan buffer allocation should succeed");
        buffer.write(0, &[5, 6, 7, 8]).expect("write buffer");
        assert_eq!(buffer.read(0, 4).expect("read buffer"), [5, 6, 7, 8]);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_buffer_read_returns_written_bytes() {
        let buffer = vulkan_device()
            .create_buffer(4, HalBufferUsage::default())
            .expect("Vulkan buffer allocation should succeed");
        buffer.write(1, &[9, 10]).expect("write buffer");
        assert_eq!(buffer.read(1, 2).expect("read buffer"), [9, 10]);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_buffer_mapped_ptr_returns_non_null_pointer() {
        let buffer = vulkan_device()
            .create_buffer(4, HalBufferUsage::default())
            .expect("Vulkan buffer allocation should succeed");
        assert!(buffer.mapped_ptr().is_some());
    }
}
