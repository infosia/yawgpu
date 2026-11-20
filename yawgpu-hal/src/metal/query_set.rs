use super::counter_sample_pool::CounterSampleAllocation;
use super::*;

/// Stores Metal query-set resources.
#[derive(Clone)]
pub struct MetalQuerySet {
    storage: QueryStorage,
    count: u32,
}

#[derive(Clone)]
enum QueryStorage {
    Occlusion { buffer: MetalBuffer },
    Timestamp(Arc<CounterSampleAllocation>),
}

impl std::fmt::Debug for MetalQuerySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalQuerySet")
            .field("count", &self.count)
            .finish()
    }
}

impl MetalQuerySet {
    /// Creates a new Metal query set backed by visibility results or timestamp counter samples.
    #[must_use = "query set creation can fail"]
    pub(super) fn new(
        device: &MetalDevice,
        kind: HalQueryKind,
        count: u32,
    ) -> Result<Self, HalError> {
        if kind == HalQueryKind::Timestamp {
            let allocation = device.counter_sample_pool.allocate(&device.device, count)?;
            return Ok(Self {
                storage: QueryStorage::Timestamp(Arc::new(allocation)),
                count,
            });
        }
        let size = u64::from(count)
            .checked_mul(8)
            .ok_or_else(|| buffer_error("query-set buffer size overflows"))?
            .max(8);
        let buffer = device
            .device
            .newBufferWithLength_options(to_ns(size)?, MTLResourceOptions::StorageModePrivate)
            .ok_or(HalError::OutOfMemory {
                backend: BACKEND,
                resource: "occlusion visibility buffer",
            })?;
        let buffer = MetalBuffer {
            inner: Some(buffer),
            mapped_ptr: None,
            size,
        };
        Ok(Self {
            storage: QueryStorage::Occlusion { buffer },
            count,
        })
    }

    /// Returns the number of queries in this set.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Returns the visibility-result buffer.
    #[must_use = "timestamp sets have no visibility buffer"]
    pub(super) fn buffer(&self) -> Result<&ProtocolObject<dyn MTLBufferTrait>, HalError> {
        match &self.storage {
            QueryStorage::Occlusion { buffer } => buffer.inner(),
            QueryStorage::Timestamp(_) => {
                Err(buffer_error("timestamp query set has no visibility buffer"))
            }
        }
    }
    /// Returns timestamp storage, or None for an occlusion query set.
    #[must_use]
    pub(super) fn sample_buffer(
        &self,
    ) -> Option<&ProtocolObject<dyn objc2_metal::MTLCounterSampleBuffer>> {
        match &self.storage {
            QueryStorage::Timestamp(allocation) => allocation.sample_buffer(),
            QueryStorage::Occlusion { .. } => None,
        }
    }

    /// Translates a validated query index to its timestamp sample buffer index.
    pub(super) fn sample_index(&self, query_index: u32) -> Result<usize, HalError> {
        let QueryStorage::Timestamp(allocation) = &self.storage else {
            return Err(buffer_error("query set is not a timestamp set"));
        };
        allocation
            .base_index()
            .checked_add(to_ns(u64::from(query_index))?)
            .ok_or_else(|| buffer_error("timestamp sample index overflows"))
    }
}

#[cfg(test)]
mod tests {
    use objc2_metal::MTLCounterSampleBuffer;

    use super::super::test_helpers::*;
    use crate::HalQueryKind;

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_query_set_timestamp_allocates_a_counter_sample_buffer() {
        let device = metal_device();
        for count in [4, 0] {
            let set = device
                .create_query_set(HalQueryKind::Timestamp, count)
                .expect("timestamp allocation");
            assert_eq!(set.count(), count);
            assert!(set.sample_buffer().is_some());
            assert!(set.buffer().is_err());
        }
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    fn metal_timestamp_pool_supports_64_live_query_sets() {
        let device = metal_device();
        let sets: Vec<_> = (0..64)
            .map(|_| {
                device
                    .create_query_set(HalQueryKind::Timestamp, 2)
                    .expect("timestamps")
            })
            .collect();
        let buffer = sets[0].sample_buffer().expect("sample buffer");
        assert_eq!(buffer.sampleCount(), 4096);
        for (index, set) in sets.iter().enumerate() {
            assert!(std::ptr::eq(
                set.sample_buffer().expect("sample buffer"),
                buffer
            ));
            assert_eq!(set.sample_index(0).expect("base index"), index * 2);
            assert_eq!(set.sample_index(1).expect("second index"), index * 2 + 1);
        }
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    fn metal_timestamp_pool_reuses_slots_after_the_last_clone_drops() {
        let device = metal_device();
        let create_set = || {
            device
                .create_query_set(HalQueryKind::Timestamp, 2)
                .expect("timestamps")
        };
        let first = create_set();
        let second = create_set();
        let clone = first.clone();
        drop(first);
        let third = create_set();
        assert_eq!(third.sample_index(0).expect("third base index"), 4);

        drop(clone);
        let reused = create_set();
        assert!(std::ptr::eq(
            reused.sample_buffer().expect("sample buffer"),
            second.sample_buffer().expect("sample buffer"),
        ));
        assert_eq!(reused.sample_index(0).expect("reused base index"), 0);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    fn metal_timestamp_allocation_outlives_its_device() {
        let query_set = metal_device()
            .create_query_set(HalQueryKind::Timestamp, 2)
            .expect("timestamps");
        assert_eq!(
            query_set
                .sample_buffer()
                .expect("sample buffer")
                .sampleCount(),
            4096
        );
        drop(query_set);
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_query_set_occlusion_still_reports_visibility_buffer() {
        let query_set = metal_device()
            .create_query_set(HalQueryKind::Occlusion, 4)
            .expect("occlusion query set should allocate");
        assert_eq!(query_set.count(), 4);
        assert!(query_set.buffer().is_ok());
        assert!(query_set.sample_buffer().is_none());
    }

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_query_set_count_zero_still_provides_a_buffer() {
        // Even a zero-count occlusion set provides a backing buffer to the HAL.
        let query_set = metal_device()
            .create_query_set(HalQueryKind::Occlusion, 0)
            .expect("zero-count occlusion query set should allocate");
        assert_eq!(query_set.count(), 0);
        assert!(query_set.buffer().is_ok());
        assert!(query_set.sample_buffer().is_none());
    }
}
