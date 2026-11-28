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
    /// The operation is ready for callback delivery.
    Ready,
    /// A callback is executing on this thread.
    Running(std::thread::ThreadId),
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
    /// Ready IDs registered for external delivery with `register`.
    pub callbacks_to_fire: Vec<FutureId>,
}

/// Stores entry metadata.
pub(crate) struct FutureEntry {
    mode: FutureCallbackMode,
    state: FutureState,
    callback: Option<Box<dyn FnOnce(FutureId) + Send>>,
}

impl std::fmt::Debug for FutureEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FutureEntry")
            .field("mode", &self.mode)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

enum Poll {
    Pending,
    Complete,
    Delivered,
    ExternalCallback,
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
        self.insert(mode, None)
    }

    /// Registers a pending future whose completion includes the return of its callback.
    #[must_use]
    pub fn register_with_callback(
        &self,
        mode: FutureCallbackMode,
        callback: impl FnOnce(FutureId) + Send + 'static,
    ) -> FutureId {
        self.insert(mode, Some(Box::new(callback)))
    }

    fn insert(
        &self,
        mode: FutureCallbackMode,
        callback: Option<Box<dyn FnOnce(FutureId) + Send>>,
    ) -> FutureId {
        let mut inner = self.inner.lock();
        let id = FutureId(inner.next_id);
        inner.next_id = inner.next_id.saturating_add(1);
        inner.futures.insert(
            id,
            FutureEntry {
                mode,
                state: FutureState::Pending,
                callback,
            },
        );
        id
    }

    /// Marks the future with the given id as complete.
    pub fn complete(&self, id: FutureId) {
        if let Some(entry) = self.inner.lock().futures.get_mut(&id) {
            if entry.state == FutureState::Pending {
                entry.state = FutureState::Ready;
            }
        }
    }

    /// Delivers ready event-processing callbacks and returns the IDs claimed by this call.
    #[must_use]
    pub fn process_events(&self) -> Vec<FutureId> {
        let mut ready = self
            .inner
            .lock()
            .futures
            .iter()
            .filter_map(|(&id, entry)| {
                (entry.state == FutureState::Ready && entry.mode != FutureCallbackMode::WaitAnyOnly)
                    .then_some(id)
            })
            .collect::<Vec<_>>();
        ready.retain(|&id| matches!(self.poll(id), Poll::Delivered | Poll::ExternalCallback));
        ready
    }

    /// Polls the requested futures. Registered callbacks execute with no registry lock held.
    #[must_use]
    pub fn wait_any(&self, ids: &[FutureId]) -> WaitAnyResult {
        let mut completed = Vec::new();
        let mut callbacks_to_fire = Vec::new();
        for &id in ids {
            match self.poll(id) {
                Poll::Pending => continue,
                Poll::ExternalCallback => callbacks_to_fire.push(id),
                Poll::Delivered | Poll::Complete => {}
            }
            completed.push(id);
        }
        WaitAnyResult {
            status: if completed.is_empty() {
                WaitAnyStatus::TimedOut
            } else {
                WaitAnyStatus::Success
            },
            completed,
            callbacks_to_fire,
        }
    }

    fn poll(&self, id: FutureId) -> Poll {
        let mut inner = self.inner.lock();
        let Some(entry) = inner.futures.get_mut(&id) else {
            return Poll::Complete;
        };
        let owner = std::thread::current().id();
        match entry.state {
            FutureState::Running(thread) if thread == owner => return Poll::Complete,
            FutureState::Pending | FutureState::Running(_) => return Poll::Pending,
            FutureState::Ready => {}
        }
        let Some(callback) = entry.callback.take() else {
            inner.futures.remove(&id);
            return Poll::ExternalCallback;
        };
        entry.state = FutureState::Running(owner);
        drop(inner);
        // Retire the entry when the callback leaves the stack, including a Rust unwind.
        struct Delivery<'a>(&'a FutureRegistry, FutureId);
        impl Drop for Delivery<'_> {
            fn drop(&mut self) {
                self.0.inner.lock().futures.remove(&self.1);
            }
        }
        let _delivery = Delivery(self, id);
        callback(id);
        Poll::Delivered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl FutureRegistry {
        fn len(&self) -> usize {
            self.inner.lock().futures.len()
        }
    }

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

    #[test]
    fn future_registry_process_events_removes_fired_entries() {
        let registry = FutureRegistry::new();
        let id = registry.register(FutureCallbackMode::AllowProcessEvents);
        registry.complete(id);

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.process_events(), vec![id]);
        assert_eq!(registry.len(), 0);
        assert!(registry.process_events().is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn future_registry_wait_any_reports_removed_process_events_future_completed() {
        let registry = FutureRegistry::new();
        let id = registry.register(FutureCallbackMode::AllowProcessEvents);
        registry.complete(id);

        assert_eq!(registry.process_events(), vec![id]);
        assert_eq!(registry.len(), 0);

        let result = registry.wait_any(&[id]);
        assert_eq!(result.status, WaitAnyStatus::Success);
        assert_eq!(result.completed, vec![id]);
        assert!(result.callbacks_to_fire.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn future_registry_wait_any_removes_directly_fired_wait_any_only_future() {
        let registry = FutureRegistry::new();
        let id = registry.register(FutureCallbackMode::WaitAnyOnly);
        registry.complete(id);

        let result = registry.wait_any(&[id]);
        assert_eq!(result.status, WaitAnyStatus::Success);
        assert_eq!(result.completed, vec![id]);
        assert_eq!(result.callbacks_to_fire, vec![id]);
        assert_eq!(registry.len(), 0);

        let result = registry.wait_any(&[id]);
        assert_eq!(result.status, WaitAnyStatus::Success);
        assert_eq!(result.completed, vec![id]);
        assert!(result.callbacks_to_fire.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn waiters_observe_callback_return_and_reentrant_waits_can_deliver_other_futures() {
        for process_events in [false, true] {
            let registry = std::sync::Arc::new(FutureRegistry::new());
            let (entered, running) = std::sync::mpsc::channel();
            let (release, resume) = std::sync::mpsc::channel();
            let nested = std::sync::Arc::clone(&registry);
            let (delivered, delivery) = std::sync::mpsc::channel();
            let second =
                registry.register_with_callback(FutureCallbackMode::WaitAnyOnly, move |id| {
                    delivered.send(id).unwrap();
                });
            let first = registry.register_with_callback(
                FutureCallbackMode::AllowProcessEvents,
                move |id| {
                    let result = nested.wait_any(&[id, second]);
                    assert_eq!(result.completed, [id, second]);
                    assert!(result.callbacks_to_fire.is_empty());
                    assert_eq!(delivery.try_recv().unwrap(), second);
                    entered.send(()).unwrap();
                    resume.recv().unwrap();
                },
            );
            registry.complete(first);
            registry.complete(second);
            std::thread::scope(|scope| {
                let worker = scope.spawn(|| {
                    if process_events {
                        assert_eq!(registry.process_events(), [first]);
                    } else {
                        assert_eq!(
                            registry.wait_any(&[first, second]).completed,
                            [first, second]
                        );
                    }
                });
                running.recv().unwrap();
                registry.complete(first);
                let result = registry.wait_any(&[first]);
                release.send(()).unwrap();
                assert_eq!(result.status, WaitAnyStatus::TimedOut);
                assert!(result.completed.is_empty());
                worker.join().unwrap();
            });
            assert_eq!(registry.wait_any(&[first]).completed, [first]);
        }
    }

    #[test]
    fn future_registry_retains_registered_pending_future() {
        let registry = FutureRegistry::new();
        let id = registry.register(FutureCallbackMode::AllowProcessEvents);

        assert_eq!(registry.len(), 1);
        assert!(registry.process_events().is_empty());

        let result = registry.wait_any(&[id]);
        assert_eq!(result.status, WaitAnyStatus::TimedOut);
        assert!(result.completed.is_empty());
        assert!(result.callbacks_to_fire.is_empty());
        assert_eq!(registry.len(), 1);
    }
}
