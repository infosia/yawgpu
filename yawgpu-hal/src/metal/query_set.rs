use super::*;

/// Stores Metal query-set resources.
#[derive(Clone)]
pub struct MetalQuerySet {
    storage: QueryStorage,
    count: u32,
}

// SAFETY: Metal query resources support thread-safe retain/release and command
// encoding. Storage and count are immutable after construction; private sample
// memory is accessed only by ordered GPU commands, never mapped by the CPU.
unsafe impl Send for MetalQuerySet {}
unsafe impl Sync for MetalQuerySet {}

#[derive(Clone)]
enum QueryStorage {
    Occlusion {
        buffer: MetalBuffer,
    },
    Timestamp {
        sample_buffer: Retained<ProtocolObject<dyn objc2_metal::MTLCounterSampleBuffer>>,
    },
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
        device: &ProtocolObject<dyn MTLDevice>,
        kind: HalQueryKind,
        count: u32,
    ) -> Result<Self, HalError> {
        if kind == HalQueryKind::Timestamp {
            let counter_set = metal_timestamp_counter_set(device)
                .ok_or_else(|| buffer_error("Metal device has no timestamp counter set"))?;
            let descriptor = objc2_metal::MTLCounterSampleBufferDescriptor::new();
            descriptor.setCounterSet(Some(&counter_set));
            // SAFETY: Allocate at least one sample, with a representable sample count.
            unsafe {
                descriptor.setSampleCount(to_ns(u64::from(count.max(1)))?);
            }
            descriptor.setStorageMode(MTLStorageMode::Private);
            let sample_buffer = device
                .newCounterSampleBufferWithDescriptor_error(&descriptor)
                .map_err(|_| HalError::OutOfMemory {
                    backend: BACKEND,
                    resource: "timestamp counter sample buffer",
                })?;
            return Ok(Self {
                storage: QueryStorage::Timestamp { sample_buffer },
                count,
            });
        }
        let size = u64::from(count)
            .checked_mul(8)
            .ok_or_else(|| buffer_error("query-set buffer size overflows"))?
            .max(8);
        let buffer = device
            .newBufferWithLength_options(to_ns(size)?, MTLResourceOptions::StorageModePrivate);
        let buffer = MetalBuffer {
            inner: buffer,
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
            QueryStorage::Timestamp { .. } => {
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
            QueryStorage::Timestamp { sample_buffer } => Some(sample_buffer),
            QueryStorage::Occlusion { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
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
        // A zero-count occlusion set can never have an in-range query, but it may
        // still be attached as a render pass `occlusionQuerySet`; its visibility
        // buffer must stay valid so a normal draw in that pass does not fail at submit.
        let query_set = metal_device()
            .create_query_set(HalQueryKind::Occlusion, 0)
            .expect("zero-count occlusion query set should allocate");
        assert_eq!(query_set.count(), 0);
        assert!(query_set.buffer().is_ok());
        assert!(query_set.sample_buffer().is_none());
    }
}
