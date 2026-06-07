use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::HalQuerySet;

use crate::adapter::*;
use crate::device::*;

/// Enumerates query type values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryType {
    /// Occlusion variant.
    Occlusion,
    /// Timestamp variant.
    Timestamp,
    /// Unknown variant.
    Unknown(u32),
}

impl From<u32> for QueryType {
    fn from(value: u32) -> Self {
        match value {
            1 => Self::Occlusion,
            2 => Self::Timestamp,
            other => Self::Unknown(other),
        }
    }
}

impl From<i32> for QueryType {
    fn from(value: i32) -> Self {
        Self::from(value as u32)
    }
}

impl From<QueryType> for u32 {
    fn from(value: QueryType) -> Self {
        match value {
            QueryType::Occlusion => 1,
            QueryType::Timestamp => 2,
            QueryType::Unknown(raw) => raw,
        }
    }
}

impl From<QueryType> for i32 {
    fn from(value: QueryType) -> Self {
        u32::from(value) as i32
    }
}

/// Describes query set descriptor.
#[derive(Debug, Clone)]
pub struct QuerySetDescriptor {
    /// Label.
    pub label: String,
    /// Kind.
    pub kind: QueryType,
    /// Count.
    pub count: u32,
}

/// Stores query set data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct QuerySet {
    pub(crate) inner: Arc<QuerySetInner>,
}

/// Holds shared state for the query set handle.
#[derive(Debug)]
pub(crate) struct QuerySetInner {
    pub(crate) label: Mutex<String>,
    pub(crate) kind: QueryType,
    pub(crate) count: u32,
    pub(crate) hal: Option<HalQuerySet>,
    pub(crate) state: Mutex<QuerySetState>,
}

/// Tracks the lifecycle state for query set.
#[derive(Debug)]
pub(crate) struct QuerySetState {
    pub(crate) is_error: bool,
    pub(crate) is_destroyed: bool,
}

impl QuerySet {
    /// Creates a new instance.
    pub(crate) fn new(
        descriptor: QuerySetDescriptor,
        is_error: bool,
        hal: Option<HalQuerySet>,
    ) -> Self {
        Self {
            inner: Arc::new(QuerySetInner {
                label: Mutex::new(descriptor.label),
                kind: descriptor.kind,
                count: descriptor.count,
                hal,
                state: Mutex::new(QuerySetState {
                    is_error,
                    is_destroyed: false,
                }),
            }),
        }
    }

    /// Returns the query type (Occlusion or Timestamp) this set holds.
    #[must_use]
    pub fn kind(&self) -> QueryType {
        self.inner.kind
    }

    /// Returns the count.
    #[must_use]
    pub fn count(&self) -> u32 {
        self.inner.count
    }

    /// Returns the HAL query set.
    pub(crate) fn hal(&self) -> Option<HalQuerySet> {
        self.inner.hal.clone()
    }

    /// Sets label on this object or encoder.
    pub fn set_label(&self, label: &str) {
        *self.inner.label.lock() = label.to_owned();
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.state.lock().is_error
    }

    /// Returns true when this object is destroyed.
    #[must_use]
    pub(crate) fn is_destroyed(&self) -> bool {
        self.inner.state.lock().is_destroyed
    }

    /// Returns true when both handles share the same backing object.
    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Destroys this object and releases backend resources.
    pub fn destroy(&self) {
        self.inner.state.lock().is_destroyed = true;
    }
}

/// Validates query set descriptor and returns a descriptive error on failure.
pub(crate) fn validate_query_set_descriptor(
    descriptor: &QuerySetDescriptor,
    features: &FeatureSet,
) -> Option<&'static str> {
    if descriptor.count > MAX_QUERY_COUNT {
        return Some("query set count exceeds the maximum query count");
    }
    match descriptor.kind {
        QueryType::Occlusion => None,
        QueryType::Timestamp => (!features.contains(&Feature::TimestampQuery))
            .then_some("timestamp query set requires the timestamp-query feature"),
        QueryType::Unknown(_) => Some("query set type is invalid"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;

    #[test]
    fn query_set_accessors_pin_kind_count_label_is_error_same_destroy() {
        assert_eq!(QueryType::from(1_u32), QueryType::Occlusion);
        assert_eq!(QueryType::from(2_i32), QueryType::Timestamp);
        assert_eq!(QueryType::from(0xFFFF_u32), QueryType::Unknown(0xFFFF));
        assert_eq!(u32::from(QueryType::Occlusion), 1);
        assert_eq!(i32::from(QueryType::Timestamp), 2);
        assert_eq!(u32::from(QueryType::Unknown(0xFFFF)), 0xFFFF);

        let device = noop_device();
        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "queries".to_owned(),
            kind: QueryType::Occlusion,
            count: 4,
        });
        let clone = query_set.clone();
        let (other, other_error) = device.create_query_set(QuerySetDescriptor {
            label: "other".to_owned(),
            kind: QueryType::Occlusion,
            count: 2,
        });

        assert_eq!(error, None);
        assert_eq!(other_error, None);
        assert_eq!(query_set.kind(), QueryType::Occlusion);
        assert_eq!(query_set.count(), 4);
        assert!(query_set.hal().is_some());
        assert!(!query_set.is_error());
        assert!(query_set.same(&clone));
        assert!(!query_set.same(&other));
        assert_eq!(query_set.inner.label.lock().as_str(), "queries");

        query_set.set_label("renamed queries");
        assert_eq!(query_set.inner.label.lock().as_str(), "renamed queries");
        assert!(!query_set.is_destroyed());
        query_set.destroy();
        query_set.destroy();
        assert!(query_set.is_destroyed());

        let (error_query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "bad".to_owned(),
            kind: QueryType::Occlusion,
            count: MAX_QUERY_COUNT + 1,
        });
        assert!(error_query_set.is_error());
        assert!(error_query_set.hal().is_none());
        assert_eq!(
            error,
            Some("query set count exceeds the maximum query count".to_owned())
        );
    }
}
