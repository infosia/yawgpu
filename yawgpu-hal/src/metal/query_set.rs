use super::*;

/// Stores Metal query-set resources.
#[derive(Clone)]
pub struct MetalQuerySet {
    pub(super) buffer: MetalBuffer,
    count: u32,
}

impl std::fmt::Debug for MetalQuerySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalQuerySet")
            .field("count", &self.count)
            .finish()
    }
}

impl MetalQuerySet {
    /// Creates a new Metal query set backed by a visibility-result buffer.
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        count: u32,
    ) -> Result<Self, HalError> {
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
        Ok(Self { buffer, count })
    }

    /// Returns the number of queries in this set.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Returns the visibility-result buffer.
    pub(super) fn buffer(&self) -> Result<&ProtocolObject<dyn MTLBufferTrait>, HalError> {
        self.buffer.inner()
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use crate::HalQueryKind;

    #[test]
    #[ignore = "manual real Metal backend test"]
    #[cfg(feature = "metal")]
    fn metal_query_set_reports_count_and_visibility_buffer() {
        let query_set = metal_device()
            .create_query_set(HalQueryKind::Occlusion, 4)
            .expect("occlusion query set should allocate");
        assert_eq!(query_set.count(), 4);
        assert!(query_set.buffer().is_ok());
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
    }
}
