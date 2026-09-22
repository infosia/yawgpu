use super::*;

/// Stores Vulkan query-set resources.
#[derive(Clone)]
pub struct VulkanQuerySet {
    pub(super) inner: Arc<VulkanQuerySetInner>,
    kind: HalQueryKind,
    count: u32,
}

impl fmt::Debug for VulkanQuerySet {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanQuerySet")
            .field("kind", &self.kind)
            .field("count", &self.count)
            .finish()
    }
}

impl VulkanQuerySet {
    /// Creates a new Vulkan query set backed by a query pool of `kind`.
    pub(super) fn new(
        device: Arc<VulkanDeviceInner>,
        kind: HalQueryKind,
        count: u32,
    ) -> Result<Self, HalError> {
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vulkan_query_type(kind))
            .query_count(count.max(1));
        let pool = unsafe { device.device.create_query_pool(&create_info, None) }
            .map_err(|_| buffer_error("query-pool creation failed"))?;
        Ok(Self {
            inner: Arc::new(VulkanQuerySetInner { device, pool }),
            kind,
            count,
        })
    }

    /// Returns the kind of query this set holds.
    #[must_use]
    pub fn kind(&self) -> HalQueryKind {
        self.kind
    }

    /// Returns the number of queries in this set.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Returns the backing query pool.
    #[must_use]
    pub(super) fn pool(&self) -> vk::QueryPool {
        self.inner.pool
    }

    /// Validates a single query index.
    pub(super) fn validate_query(&self, index: u32) -> Result<(), HalError> {
        if index >= self.count {
            return Err(buffer_error("query index exceeds query-set count"));
        }
        Ok(())
    }

    /// Validates a query range.
    pub(super) fn validate_range(&self, first: u32, count: u32) -> Result<(), HalError> {
        let end = first
            .checked_add(count)
            .ok_or_else(|| buffer_error("query range overflows"))?;
        if end > self.count {
            return Err(buffer_error("query range exceeds query-set count"));
        }
        Ok(())
    }
}

/// Maps a HAL query kind to its Vulkan query type, as Dawn's
/// `VulkanQueryType` (`QuerySetVk.cpp`) does.
fn vulkan_query_type(kind: HalQueryKind) -> vk::QueryType {
    match kind {
        HalQueryKind::Occlusion => vk::QueryType::OCCLUSION,
        HalQueryKind::Timestamp => vk::QueryType::TIMESTAMP,
    }
}

/// Holds shared state for the Vulkan query-set handle.
pub(super) struct VulkanQuerySetInner {
    pub(super) device: Arc<VulkanDeviceInner>,
    pool: vk::QueryPool,
}

impl fmt::Debug for VulkanQuerySetInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanQuerySetInner")
            .finish_non_exhaustive()
    }
}

unsafe impl Send for VulkanQuerySetInner {}
unsafe impl Sync for VulkanQuerySetInner {}

impl Drop for VulkanQuerySetInner {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_query_pool(self.pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;

    /// Block 102 R3: a timestamp set is a `VK_QUERY_TYPE_TIMESTAMP` pool while
    /// occlusion stays `VK_QUERY_TYPE_OCCLUSION`. Pure mapping, no GPU needed.
    #[test]
    fn vulkan_query_type_maps_timestamp_and_occlusion_kinds() {
        assert_eq!(
            vulkan_query_type(HalQueryKind::Timestamp),
            vk::QueryType::TIMESTAMP
        );
        assert_eq!(
            vulkan_query_type(HalQueryKind::Occlusion),
            vk::QueryType::OCCLUSION
        );
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_query_set_reports_count_and_query_pool() {
        let query_set = vulkan_device()
            .create_query_set(HalQueryKind::Occlusion, 4)
            .expect("occlusion query set should allocate");
        assert_eq!(query_set.kind(), HalQueryKind::Occlusion);
        assert_eq!(query_set.count(), 4);
        assert_ne!(query_set.pool(), ash::vk::QueryPool::null());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_query_set_count_zero_still_provides_a_pool() {
        let query_set = vulkan_device()
            .create_query_set(HalQueryKind::Occlusion, 0)
            .expect("zero-count occlusion query set should allocate");
        assert_eq!(query_set.count(), 0);
        assert_ne!(query_set.pool(), ash::vk::QueryPool::null());
    }

    /// Block 102 R3: a timestamp query set allocates a timestamp pool of
    /// `max(count, 1)` queries and reports its kind back.
    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_query_set_timestamp_creates_a_timestamp_pool() {
        let device = vulkan_device();
        let query_set = device
            .create_query_set(HalQueryKind::Timestamp, 4)
            .expect("timestamp query set should allocate");
        assert_eq!(query_set.kind(), HalQueryKind::Timestamp);
        assert_eq!(query_set.count(), 4);
        assert_ne!(query_set.pool(), ash::vk::QueryPool::null());

        let empty = device
            .create_query_set(HalQueryKind::Timestamp, 0)
            .expect("zero-count timestamp query set should allocate");
        assert_eq!(empty.kind(), HalQueryKind::Timestamp);
        assert_eq!(empty.count(), 0);
        assert_ne!(empty.pool(), ash::vk::QueryPool::null());
        assert!(empty.validate_query(0).is_err());
    }
}
