use std::collections::BTreeMap;

use parking_lot::Mutex;

/// Identifies id used for cache lookup or callback tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FutureId(u64);

impl FutureId {
    /// Returns the raw 64-bit future identifier.
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }

    /// Constructs this object from raw.
    #[must_use]
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

/// Stores future registry data used by validation and backend submission.
#[derive(Debug, Default)]
pub struct FutureRegistry {
    pub(crate) inner: Mutex<FutureRegistryInner>,
}

/// Holds shared state for the future registry handle.
#[derive(Debug)]
pub(crate) struct FutureRegistryInner {
    pub(crate) next_id: u64,
    pub(crate) futures: BTreeMap<FutureId, FutureEntry>,
}

impl Default for FutureRegistryInner {
    fn default() -> Self {
        Self {
            next_id: 1,
            futures: BTreeMap::new(),
        }
    }
}

/// Enumerates future state values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FutureState {
    /// Pending variant.
    Pending,
    /// Complete variant.
    Complete,
}

/// Enumerates future callback mode values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FutureCallbackMode {
    /// Wait any only variant.
    WaitAnyOnly,
    /// Allow process events variant.
    AllowProcessEvents,
    /// Allow spontaneous variant.
    AllowSpontaneous,
}

/// Enumerates wait any status values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WaitAnyStatus {
    /// Success variant.
    Success,
    /// Timed out variant.
    TimedOut,
    /// Error variant.
    Error,
}

/// Stores wait any result data used by validation and backend submission.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WaitAnyResult {
    /// Status.
    pub status: WaitAnyStatus,
    /// Completed.
    pub completed: Vec<FutureId>,
    /// Callbacks to fire.
    pub callbacks_to_fire: Vec<FutureId>,
}

/// Stores entry metadata.
#[derive(Debug)]
pub(crate) struct FutureEntry {
    pub(crate) mode: FutureCallbackMode,
    pub(crate) state: FutureState,
    pub(crate) callback_fired: bool,
}

impl FutureRegistry {
    /// Creates a new instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new pending future with the given callback mode and returns its id.
    #[must_use]
    pub fn register(&self, mode: FutureCallbackMode) -> FutureId {
        let mut inner = self.inner.lock();
        let id = FutureId(inner.next_id);
        inner.next_id = inner.next_id.saturating_add(1);
        inner.futures.insert(
            id,
            FutureEntry {
                mode,
                state: FutureState::Pending,
                callback_fired: false,
            },
        );
        id
    }

    /// Marks the future with the given id as complete.
    pub fn complete(&self, id: FutureId) {
        if let Some(entry) = self.inner.lock().futures.get_mut(&id) {
            entry.state = FutureState::Complete;
        }
    }

    /// Returns process events.
    #[must_use]
    pub fn process_events(&self) -> Vec<FutureId> {
        let mut inner = self.inner.lock();
        inner
            .futures
            .iter_mut()
            .filter_map(|(id, entry)| {
                let can_fire = entry.state == FutureState::Complete
                    && !entry.callback_fired
                    && matches!(
                        entry.mode,
                        FutureCallbackMode::AllowProcessEvents
                            | FutureCallbackMode::AllowSpontaneous
                    );
                if can_fire {
                    entry.callback_fired = true;
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns wait any.
    #[must_use]
    pub fn wait_any(&self, ids: &[FutureId]) -> WaitAnyResult {
        if ids.is_empty() {
            return WaitAnyResult {
                status: WaitAnyStatus::TimedOut,
                completed: Vec::new(),
                callbacks_to_fire: Vec::new(),
            };
        }

        let mut inner = self.inner.lock();
        let mut completed = Vec::new();
        let mut callbacks_to_fire = Vec::new();

        for id in ids {
            let Some(entry) = inner.futures.get_mut(id) else {
                continue;
            };
            if entry.state == FutureState::Complete {
                completed.push(*id);
                if !entry.callback_fired {
                    entry.callback_fired = true;
                    callbacks_to_fire.push(*id);
                }
            }
        }

        let status = if completed.is_empty() {
            WaitAnyStatus::TimedOut
        } else {
            WaitAnyStatus::Success
        };

        WaitAnyResult {
            status,
            completed,
            callbacks_to_fire,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn future_id_get_and_from_raw_round_trip() {
        assert_eq!(FutureId::from_raw(42).get(), 42);
        assert_eq!(FutureId::from_raw(0).get(), 0);
        assert_eq!(FutureId::from_raw(u64::MAX).get(), u64::MAX);
    }

    #[test]
    fn future_registry_process_events_respects_callback_mode() {
        let registry = FutureRegistry::new();
        let first = registry.register(FutureCallbackMode::WaitAnyOnly);
        let second = registry.register(FutureCallbackMode::AllowProcessEvents);
        registry.complete(first);
        registry.complete(second);

        assert_eq!(registry.process_events(), vec![second]);
        assert!(registry.process_events().is_empty());

        let result = registry.wait_any(&[first, second]);
        assert_eq!(result.status, WaitAnyStatus::Success);
        assert_eq!(result.completed, vec![first, second]);
        assert_eq!(result.callbacks_to_fire, vec![first]);

        let result = registry.wait_any(&[first, second]);
        assert_eq!(result.status, WaitAnyStatus::Success);
        assert_eq!(result.completed, vec![first, second]);
        assert!(result.callbacks_to_fire.is_empty());
    }
}
