use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{HalAdapter, HalDevice, HalError, HalInstance, HalQueue};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Hal(#[from] HalError),
}

#[derive(Debug, Clone)]
pub struct Instance {
    inner: Arc<InstanceInner>,
}

#[derive(Debug)]
struct InstanceInner {
    hal: HalInstance,
    futures: FutureRegistry,
}

impl Instance {
    #[must_use]
    pub fn new_noop() -> Self {
        Self::from_hal(HalInstance::new_noop())
    }

    #[must_use]
    pub fn from_hal(hal: HalInstance) -> Self {
        Self {
            inner: Arc::new(InstanceInner {
                hal,
                futures: FutureRegistry::new(),
            }),
        }
    }

    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<Adapter> {
        self.inner
            .hal
            .enumerate_adapters()
            .into_iter()
            .map(Adapter::from_hal)
            .collect()
    }

    #[must_use]
    pub fn future_registry(&self) -> &FutureRegistry {
        &self.inner.futures
    }
}

#[derive(Debug, Clone)]
pub struct Adapter {
    inner: Arc<AdapterInner>,
}

#[derive(Debug)]
struct AdapterInner {
    hal: HalAdapter,
}

impl Adapter {
    #[must_use]
    pub fn from_hal(hal: HalAdapter) -> Self {
        Self {
            inner: Arc::new(AdapterInner { hal }),
        }
    }

    pub fn create_device(&self) -> Result<Device, Error> {
        let hal = self.inner.hal.create_device()?;
        Ok(Device::from_hal(hal))
    }
}

#[derive(Debug, Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

#[derive(Debug)]
struct DeviceInner {
    hal: HalDevice,
    queue: Queue,
    error_sink: Mutex<ErrorSink>,
}

impl Device {
    #[must_use]
    pub fn from_hal(hal: HalDevice) -> Self {
        let queue = Queue::from_hal(hal.queue());
        Self {
            inner: Arc::new(DeviceInner {
                hal,
                queue,
                error_sink: Mutex::new(ErrorSink::default()),
            }),
        }
    }

    #[must_use]
    pub fn queue(&self) -> Queue {
        self.inner.queue.clone()
    }

    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.inner.hal.allocation_count()
    }

    pub fn set_uncaptured_error_callback<F>(&self, callback: Option<F>)
    where
        F: Fn(DeviceError) + Send + Sync + 'static,
    {
        self.inner.error_sink.lock().uncaptured_error_callback = callback.map(|f| Arc::new(f) as _);
    }

    pub fn clear_uncaptured_error_callback(&self) {
        self.inner.error_sink.lock().uncaptured_error_callback = None;
    }

    pub fn push_error_scope(&self) {
        self.inner
            .error_sink
            .lock()
            .scopes
            .push(ErrorScope::default());
    }

    #[must_use]
    pub fn pop_error_scope(&self) -> Option<DeviceError> {
        self.inner
            .error_sink
            .lock()
            .scopes
            .pop()
            .and_then(|scope| scope.error)
    }

    pub fn dispatch_error(&self, kind: ErrorKind, msg: impl Into<String>) {
        let error = DeviceError::new(kind, msg);
        let callback = {
            let mut sink = self.inner.error_sink.lock();
            if let Some(scope) = sink.scopes.last_mut() {
                if scope.error.is_none() {
                    scope.error = Some(error);
                }
                return;
            }
            sink.uncaptured_error_callback.clone()
        };

        if let Some(callback) = callback {
            callback(error);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Queue {
    inner: Arc<QueueInner>,
}

#[derive(Debug)]
struct QueueInner {
    hal: HalQueue,
}

impl Queue {
    #[must_use]
    pub fn from_hal(hal: HalQueue) -> Self {
        Self {
            inner: Arc::new(QueueInner { hal }),
        }
    }

    #[must_use]
    pub fn hal(&self) -> &HalQueue {
        &self.inner.hal
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    Validation,
    OutOfMemory,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceError {
    pub kind: ErrorKind,
    pub message: String,
}

impl DeviceError {
    #[must_use]
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

type UncapturedErrorCallback = Arc<dyn Fn(DeviceError) + Send + Sync>;

#[derive(Default)]
struct ErrorSink {
    uncaptured_error_callback: Option<UncapturedErrorCallback>,
    scopes: Vec<ErrorScope>,
}

#[derive(Default)]
struct ErrorScope {
    error: Option<DeviceError>,
}

impl std::fmt::Debug for ErrorSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorSink")
            .field(
                "uncaptured_error_callback",
                &self.uncaptured_error_callback.is_some(),
            )
            .field("scopes", &self.scopes)
            .finish()
    }
}

impl std::fmt::Debug for ErrorScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ErrorScope")
            .field("error", &self.error)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FutureId(u64);

impl FutureId {
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Default)]
pub struct FutureRegistry {
    inner: Mutex<FutureRegistryInner>,
}

#[derive(Debug)]
struct FutureRegistryInner {
    next_id: u64,
    futures: BTreeMap<FutureId, FutureState>,
}

impl Default for FutureRegistryInner {
    fn default() -> Self {
        Self {
            next_id: 1,
            futures: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FutureState {
    Pending,
    Complete,
}

impl FutureRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn register(&self) -> FutureId {
        let mut inner = self.inner.lock();
        let id = FutureId(inner.next_id);
        inner.next_id = inner.next_id.saturating_add(1);
        inner.futures.insert(id, FutureState::Pending);
        id
    }

    pub fn complete(&self, id: FutureId) {
        if let Some(state) = self.inner.lock().futures.get_mut(&id) {
            *state = FutureState::Complete;
        }
    }

    #[must_use]
    pub fn poll_all(&self) -> Vec<FutureId> {
        let mut inner = self.inner.lock();
        let completed: Vec<_> = inner.futures.keys().copied().collect();
        inner.futures.clear();
        completed
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::{ErrorKind, FutureRegistry, Instance};

    #[test]
    fn creates_noop_device_and_queue() {
        let instance = Instance::new_noop();
        let adapters = instance.enumerate_adapters();
        assert_eq!(adapters.len(), 1);

        let device = adapters[0]
            .create_device()
            .expect("Noop device should be created");
        assert_eq!(device.allocation_count(), 0);

        let _queue = device.queue();
    }

    #[test]
    fn scoped_error_captures_without_uncaptured_callback() {
        let instance = Instance::new_noop();
        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter should exist");
        let device = adapter
            .create_device()
            .expect("Noop device should be created");
        let uncaptured_count = Arc::new(AtomicUsize::new(0));
        let callback_count = uncaptured_count.clone();

        device.set_uncaptured_error_callback(Some(move |_| {
            callback_count.fetch_add(1, Ordering::Relaxed);
        }));
        device.push_error_scope();
        device.dispatch_error(ErrorKind::Validation, "scoped validation error");

        let error = device
            .pop_error_scope()
            .expect("scope should contain an error");
        assert_eq!(error.kind, ErrorKind::Validation);
        assert_eq!(error.message, "scoped validation error");
        assert_eq!(uncaptured_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn uncaptured_error_routes_to_callback_without_scope() {
        let instance = Instance::new_noop();
        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter should exist");
        let device = adapter
            .create_device()
            .expect("Noop device should be created");
        let uncaptured_count = Arc::new(AtomicUsize::new(0));
        let callback_count = uncaptured_count.clone();

        device.set_uncaptured_error_callback(Some(move |error: super::DeviceError| {
            assert_eq!(error.kind, ErrorKind::Internal);
            callback_count.fetch_add(1, Ordering::Relaxed);
        }));
        device.dispatch_error(ErrorKind::Internal, "uncaptured internal error");

        assert_eq!(uncaptured_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn future_registry_completes_synchronously_on_poll() {
        let registry = FutureRegistry::new();
        let first = registry.register();
        let second = registry.register();
        registry.complete(first);

        assert_eq!(registry.poll_all(), vec![first, second]);
        assert!(registry.poll_all().is_empty());
    }
}
