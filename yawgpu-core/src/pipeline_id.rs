use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_PIPELINE_ID: AtomicU64 = AtomicU64::new(1);

/// Returns a process-unique pipeline identifier.
pub(crate) fn next_pipeline_id() -> u64 {
    NEXT_PIPELINE_ID.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_pipeline_id_returns_distinct_values() {
        let first = next_pipeline_id();
        let second = next_pipeline_id();

        assert_ne!(first, second);
    }
}
