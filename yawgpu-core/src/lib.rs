use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{HalAdapter, HalDevice, HalError, HalInstance, HalQueue};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error(transparent)]
    Hal(#[from] HalError),
    #[error("{0}")]
    Validation(String),
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
        self.enumerate_adapters_with_feature_level(FeatureLevel::Core)
    }

    #[must_use]
    pub fn enumerate_adapters_with_feature_level(
        &self,
        feature_level: FeatureLevel,
    ) -> Vec<Adapter> {
        self.inner
            .hal
            .enumerate_adapters()
            .into_iter()
            .map(|hal| Adapter::from_hal_with_feature_level(hal, feature_level))
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
    feature_level: FeatureLevel,
}

impl Adapter {
    #[must_use]
    pub fn from_hal(hal: HalAdapter) -> Self {
        Self::from_hal_with_feature_level(hal, FeatureLevel::Core)
    }

    #[must_use]
    pub fn from_hal_with_feature_level(hal: HalAdapter, feature_level: FeatureLevel) -> Self {
        Self {
            inner: Arc::new(AdapterInner { hal, feature_level }),
        }
    }

    #[must_use]
    pub fn limits(&self) -> Limits {
        Limits::DEFAULT
    }

    #[must_use]
    pub fn feature_level(&self) -> FeatureLevel {
        self.inner.feature_level
    }

    #[must_use]
    pub fn features(&self) -> FeatureSet {
        supported_features()
    }

    #[must_use]
    pub fn has_feature(&self, feature: Feature) -> bool {
        self.features().contains(&feature)
    }

    pub fn create_device(
        &self,
        required_limits: Option<&Limits>,
        required_features: &[Feature],
    ) -> Result<Device, Error> {
        let limits = self
            .limits()
            .validate_required_limits(required_limits)
            .map_err(Error::Validation)?;
        let features = self.resolve_features(required_features)?;
        let hal = self.inner.hal.create_device()?;
        Ok(Device::from_hal(hal, limits, features))
    }

    fn resolve_features(&self, required_features: &[Feature]) -> Result<FeatureSet, Error> {
        let supported = self.features();
        let mut resolved = FeatureSet::new();

        if self.feature_level() == FeatureLevel::Core {
            resolved.insert(Feature::CoreFeaturesAndLimits);
        }

        for feature in required_features {
            if !supported.contains(feature) {
                return Err(Error::Validation(format!(
                    "required feature {feature:?} is not supported"
                )));
            }
            resolved.insert(*feature);
        }

        apply_feature_implications(&mut resolved);
        Ok(resolved)
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
    lost: Mutex<DeviceLostState>,
    limits: Limits,
    features: FeatureSet,
}

impl Device {
    #[must_use]
    pub fn from_hal(hal: HalDevice, limits: Limits, features: FeatureSet) -> Self {
        let queue = Queue::from_hal(hal.queue());
        Self {
            inner: Arc::new(DeviceInner {
                hal,
                queue,
                error_sink: Mutex::new(ErrorSink::default()),
                lost: Mutex::new(DeviceLostState::default()),
                limits,
                features,
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

    #[must_use]
    pub fn limits(&self) -> Limits {
        self.inner.limits
    }

    #[must_use]
    pub fn features(&self) -> FeatureSet {
        self.inner.features.clone()
    }

    #[must_use]
    pub fn has_feature(&self, feature: Feature) -> bool {
        self.inner.features.contains(&feature)
    }

    pub fn destroy(&self) -> Option<DeviceLostReason> {
        self.lose(DeviceLostReason::Destroyed)
    }

    pub fn lose(&self, reason: DeviceLostReason) -> Option<DeviceLostReason> {
        let mut lost = self.inner.lost.lock();
        if lost.reason.is_some() {
            return None;
        }
        lost.reason = Some(reason);
        Some(reason)
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

#[derive(Debug, Default)]
struct DeviceLostState {
    reason: Option<DeviceLostReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceLostReason {
    Unknown,
    Destroyed,
    CallbackCancelled,
    FailedCreation,
}

pub type FeatureSet = BTreeSet<Feature>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FeatureLevel {
    Core,
    Compatibility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Feature {
    CoreFeaturesAndLimits,
    Rg11b10UfloatRenderable,
    TextureFormatsTier1,
    TextureFormatsTier2,
    Other(u32),
}

#[must_use]
pub fn supported_features() -> FeatureSet {
    [
        Feature::CoreFeaturesAndLimits,
        Feature::Rg11b10UfloatRenderable,
        Feature::TextureFormatsTier1,
        Feature::TextureFormatsTier2,
    ]
    .into_iter()
    .collect()
}

fn apply_feature_implications(features: &mut FeatureSet) {
    if features.contains(&Feature::TextureFormatsTier2) {
        features.insert(Feature::TextureFormatsTier1);
    }
    if features.contains(&Feature::TextureFormatsTier1) {
        features.insert(Feature::Rg11b10UfloatRenderable);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Limits {
    pub max_texture_dimension_1d: u32,
    pub max_texture_dimension_2d: u32,
    pub max_texture_dimension_3d: u32,
    pub max_texture_array_layers: u32,
    pub max_bind_groups: u32,
    pub max_bind_groups_plus_vertex_buffers: u32,
    pub max_bindings_per_bind_group: u32,
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,
    pub max_sampled_textures_per_shader_stage: u32,
    pub max_samplers_per_shader_stage: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_storage_textures_per_shader_stage: u32,
    pub max_uniform_buffers_per_shader_stage: u32,
    pub max_uniform_buffer_binding_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub min_uniform_buffer_offset_alignment: u32,
    pub min_storage_buffer_offset_alignment: u32,
    pub max_vertex_buffers: u32,
    pub max_buffer_size: u64,
    pub max_vertex_attributes: u32,
    pub max_vertex_buffer_array_stride: u32,
    pub max_inter_stage_shader_variables: u32,
    pub max_color_attachments: u32,
    pub max_color_attachment_bytes_per_sample: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_immediate_size: u32,
}

impl Limits {
    pub const DEFAULT: Self = Self {
        max_texture_dimension_1d: 4096,
        max_texture_dimension_2d: 4096,
        max_texture_dimension_3d: 2048,
        max_texture_array_layers: 256,
        max_bind_groups: 4,
        max_bind_groups_plus_vertex_buffers: 24,
        max_bindings_per_bind_group: 1000,
        max_dynamic_uniform_buffers_per_pipeline_layout: 8,
        max_dynamic_storage_buffers_per_pipeline_layout: 4,
        max_sampled_textures_per_shader_stage: 16,
        max_samplers_per_shader_stage: 16,
        max_storage_buffers_per_shader_stage: 8,
        max_storage_textures_per_shader_stage: 4,
        max_uniform_buffers_per_shader_stage: 12,
        max_uniform_buffer_binding_size: 16_384,
        max_storage_buffer_binding_size: 128 * 1024 * 1024,
        min_uniform_buffer_offset_alignment: 256,
        min_storage_buffer_offset_alignment: 256,
        max_vertex_buffers: 8,
        max_buffer_size: 256 * 1024 * 1024,
        max_vertex_attributes: 16,
        max_vertex_buffer_array_stride: 2048,
        max_inter_stage_shader_variables: 15,
        max_color_attachments: 4,
        max_color_attachment_bytes_per_sample: 32,
        max_compute_workgroup_storage_size: 16_384,
        max_compute_invocations_per_workgroup: 128,
        max_compute_workgroup_size_x: 128,
        max_compute_workgroup_size_y: 128,
        max_compute_workgroup_size_z: 64,
        max_compute_workgroups_per_dimension: 65_535,
        max_immediate_size: 64,
    };

    fn validate_required_limits(self, required: Option<&Self>) -> Result<Self, String> {
        let required = required.copied().unwrap_or(Self::DEFAULT);
        let default = Self::DEFAULT;
        let mut effective = default;

        macro_rules! maximum {
            ($field:ident) => {
                if required.$field > self.$field {
                    return Err(format!(
                        "required limit {}={} exceeds supported {}",
                        stringify!($field),
                        required.$field,
                        self.$field
                    ));
                }
                effective.$field = required.$field.max(default.$field);
            };
        }

        macro_rules! alignment {
            ($field:ident) => {
                if required.$field < self.$field {
                    return Err(format!(
                        "required limit {}={} is below supported {}",
                        stringify!($field),
                        required.$field,
                        self.$field
                    ));
                }
                effective.$field = required.$field.min(default.$field);
            };
        }

        maximum!(max_texture_dimension_1d);
        maximum!(max_texture_dimension_2d);
        maximum!(max_texture_dimension_3d);
        maximum!(max_texture_array_layers);
        maximum!(max_bind_groups);
        maximum!(max_bind_groups_plus_vertex_buffers);
        maximum!(max_bindings_per_bind_group);
        maximum!(max_dynamic_uniform_buffers_per_pipeline_layout);
        maximum!(max_dynamic_storage_buffers_per_pipeline_layout);
        maximum!(max_sampled_textures_per_shader_stage);
        maximum!(max_samplers_per_shader_stage);
        maximum!(max_storage_buffers_per_shader_stage);
        maximum!(max_storage_textures_per_shader_stage);
        maximum!(max_uniform_buffers_per_shader_stage);
        maximum!(max_uniform_buffer_binding_size);
        maximum!(max_storage_buffer_binding_size);
        alignment!(min_uniform_buffer_offset_alignment);
        alignment!(min_storage_buffer_offset_alignment);
        maximum!(max_vertex_buffers);
        maximum!(max_buffer_size);
        maximum!(max_vertex_attributes);
        maximum!(max_vertex_buffer_array_stride);
        maximum!(max_inter_stage_shader_variables);
        maximum!(max_color_attachments);
        maximum!(max_color_attachment_bytes_per_sample);
        maximum!(max_compute_workgroup_storage_size);
        maximum!(max_compute_invocations_per_workgroup);
        maximum!(max_compute_workgroup_size_x);
        maximum!(max_compute_workgroup_size_y);
        maximum!(max_compute_workgroup_size_z);
        maximum!(max_compute_workgroups_per_dimension);

        if required.max_immediate_size > self.max_immediate_size {
            return Err(format!(
                "required limit max_immediate_size={} exceeds supported {}",
                required.max_immediate_size, self.max_immediate_size
            ));
        }
        effective.max_immediate_size = self.max_immediate_size;

        Ok(effective)
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

    #[must_use]
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

#[derive(Debug, Default)]
pub struct FutureRegistry {
    inner: Mutex<FutureRegistryInner>,
}

#[derive(Debug)]
struct FutureRegistryInner {
    next_id: u64,
    futures: BTreeMap<FutureId, FutureEntry>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FutureCallbackMode {
    WaitAnyOnly,
    AllowProcessEvents,
    AllowSpontaneous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WaitAnyStatus {
    Success,
    TimedOut,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct WaitAnyResult {
    pub status: WaitAnyStatus,
    pub completed: Vec<FutureId>,
    pub callbacks_to_fire: Vec<FutureId>,
}

#[derive(Debug)]
struct FutureEntry {
    mode: FutureCallbackMode,
    state: FutureState,
    callback_fired: bool,
}

impl FutureRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn complete(&self, id: FutureId) {
        if let Some(entry) = self.inner.lock().futures.get_mut(&id) {
            entry.state = FutureState::Complete;
        }
    }

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

    #[must_use]
    pub fn wait_any(&self, ids: &[FutureId], _poll_only: bool) -> WaitAnyResult {
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::{ErrorKind, FutureCallbackMode, FutureRegistry, Instance, WaitAnyStatus};

    #[test]
    fn creates_noop_device_and_queue() {
        let instance = Instance::new_noop();
        let adapters = instance.enumerate_adapters();
        assert_eq!(adapters.len(), 1);

        let device = adapters[0]
            .create_device(None, &[])
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
            .create_device(None, &[])
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
            .create_device(None, &[])
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
    fn future_registry_process_events_respects_callback_mode() {
        let registry = FutureRegistry::new();
        let first = registry.register(FutureCallbackMode::WaitAnyOnly);
        let second = registry.register(FutureCallbackMode::AllowProcessEvents);
        registry.complete(first);
        registry.complete(second);

        assert_eq!(registry.process_events(), vec![second]);
        assert!(registry.process_events().is_empty());

        let result = registry.wait_any(&[first, second], true);
        assert_eq!(result.status, WaitAnyStatus::Success);
        assert_eq!(result.completed, vec![first, second]);
        assert_eq!(result.callbacks_to_fire, vec![first]);

        let result = registry.wait_any(&[first, second], true);
        assert_eq!(result.status, WaitAnyStatus::Success);
        assert_eq!(result.completed, vec![first, second]);
        assert!(result.callbacks_to_fire.is_empty());
    }
}
