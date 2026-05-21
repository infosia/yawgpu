use std::collections::BTreeSet;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::HalDevice;

use crate::adapter::*;
use crate::bind_group::*;
use crate::bind_group_layout::*;
use crate::buffer::*;
use crate::command_encoder::*;
use crate::compute_pipeline::*;
use crate::error::*;
use crate::limits::*;
use crate::pipeline_layout::*;
use crate::query_set::*;
use crate::queue::*;
use crate::render_pipeline::*;
use crate::sampler::*;
use crate::shader::*;
use crate::texture::*;

#[derive(Debug, Clone)]
pub struct Device {
    pub(crate) inner: Arc<DeviceInner>,
}

#[derive(Debug)]
pub(crate) struct DeviceInner {
    pub(crate) hal: HalDevice,
    pub(crate) queue: Queue,
    pub(crate) error_sink: Mutex<ErrorSink>,
    pub(crate) lost: Mutex<DeviceLostState>,
    pub(crate) label: Mutex<String>,
    pub(crate) limits: Limits,
    pub(crate) features: FeatureSet,
}

impl Device {
    #[must_use]
    pub fn from_hal(
        hal: HalDevice,
        limits: Limits,
        features: FeatureSet,
        label: impl Into<String>,
        queue_label: impl Into<String>,
    ) -> Self {
        let queue = Queue::from_hal(hal.queue(), queue_label);
        Self {
            inner: Arc::new(DeviceInner {
                hal,
                queue,
                error_sink: Mutex::new(ErrorSink::default()),
                lost: Mutex::new(DeviceLostState::default()),
                label: Mutex::new(label.into()),
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
    pub fn hal(&self) -> &HalDevice {
        &self.inner.hal
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

    #[must_use]
    pub fn create_query_set(&self, descriptor: QuerySetDescriptor) -> (QuerySet, Option<String>) {
        if self.is_lost() {
            return (QuerySet::new(descriptor, true), None);
        }
        let error = validate_query_set_descriptor(&descriptor, &self.inner.features);
        let is_error = error.is_some();
        (
            QuerySet::new(descriptor, is_error),
            error.map(str::to_owned),
        )
    }

    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn set_label(&self, label: &str) {
        *self.inner.label.lock() = label.to_owned();
    }

    #[must_use]
    pub fn label(&self) -> String {
        self.inner.label.lock().clone()
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

    #[must_use]
    pub fn is_lost(&self) -> bool {
        self.inner.lost.lock().reason.is_some()
    }

    #[must_use]
    pub fn lost_reason(&self) -> Option<DeviceLostReason> {
        self.inner.lost.lock().reason
    }

    pub fn set_uncaptured_error_callback<F>(&self, callback: Option<F>)
    where
        F: Fn(DeviceError) + Send + Sync + 'static,
    {
        self.inner.error_sink.lock().uncaptured_error_callback = callback.map(|f| Arc::new(f) as _);
    }

    pub fn push_error_scope(&self, filter: ErrorFilter) {
        self.inner.error_sink.lock().scopes.push(ErrorScope {
            filter,
            error: None,
        });
    }

    pub fn pop_error_scope(&self) -> Result<Option<DeviceError>, PopErrorScopeError> {
        self.inner
            .error_sink
            .lock()
            .scopes
            .pop()
            .map(|scope| scope.error)
            .ok_or(PopErrorScopeError::EmptyStack)
    }

    pub fn dispatch_error(&self, kind: ErrorKind, msg: impl Into<String>) {
        let error = DeviceError::new(kind, msg);
        let callback = {
            let mut sink = self.inner.error_sink.lock();
            for scope in sink.scopes.iter_mut().rev() {
                if scope.filter.matches(error.kind) {
                    if scope.error.is_none() {
                        scope.error = Some(error);
                    }
                    return;
                }
            }
            sink.uncaptured_error_callback.clone()
        };

        if let Some(callback) = callback {
            callback(error);
        }
    }

    #[must_use]
    pub fn create_buffer(&self, descriptor: BufferDescriptor) -> Buffer {
        if self.is_lost() {
            return Buffer::new(descriptor, None, true);
        }
        let error = validate_buffer_descriptor(&descriptor, self.limits());
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }

        let hal = if is_error {
            None
        } else {
            Some(self.inner.hal.create_buffer(descriptor.size))
        };

        Buffer::new(descriptor, hal, is_error)
    }

    #[must_use]
    pub fn create_texture(&self, descriptor: TextureDescriptor) -> Texture {
        if self.is_lost() {
            return Texture::new(descriptor, None, true);
        }
        let error = validate_texture_descriptor(&descriptor, self.limits());
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }

        let hal = if is_error {
            None
        } else {
            Some(
                self.inner
                    .hal
                    .create_texture(&hal_texture_descriptor(&descriptor)),
            )
        };

        Texture::new(descriptor, hal, is_error)
    }

    #[must_use]
    pub fn create_sampler(&self, descriptor: SamplerDescriptor) -> Sampler {
        let resolved = ResolvedSamplerDescriptor::from_descriptor(descriptor);
        if self.is_lost() {
            return Sampler::new(resolved, None, true);
        }
        let error = validate_sampler_descriptor(&resolved);
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }

        let hal = if is_error {
            None
        } else {
            Some(
                self.inner
                    .hal
                    .create_sampler(&hal_sampler_descriptor(&resolved)),
            )
        };

        Sampler::new(resolved, hal, is_error)
    }

    #[must_use]
    pub fn create_shader_module(&self, source: ShaderModuleSource) -> ShaderModule {
        if self.is_lost() {
            return ShaderModule::new(
                ShaderModuleSourceKind::Invalid,
                Some("device is lost".to_owned()),
            );
        }
        let (inner, error) = match source {
            ShaderModuleSource::Wgsl(source) => match ShaderModule::from_wgsl(source) {
                Ok(inner) => (inner, None),
                Err(message) => (ShaderModuleSourceKind::Invalid, Some(message)),
            },
            ShaderModuleSource::Spirv(words) => {
                (ShaderModuleSourceKind::Spirv { _words: words }, None)
            }
            ShaderModuleSource::Invalid(message) => {
                (ShaderModuleSourceKind::Invalid, Some(message))
            }
        };

        let diagnostic = error.clone();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        ShaderModule::new(inner, diagnostic)
    }

    #[must_use]
    pub fn create_bind_group_layout(
        &self,
        descriptor: BindGroupLayoutDescriptor,
    ) -> BindGroupLayout {
        if self.is_lost() {
            return BindGroupLayout::new(descriptor.entries, true, false);
        }
        let error = descriptor.error.clone().or_else(|| {
            crate::bind_group_layout::validate_bind_group_layout_descriptor(
                &descriptor.entries,
                self.limits(),
            )
        });
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        BindGroupLayout::new(descriptor.entries, is_error, false)
    }

    #[must_use]
    pub fn create_bind_group(
        &self,
        layout: Arc<BindGroupLayout>,
        entries: Vec<BindGroupEntry>,
    ) -> BindGroup {
        if self.is_lost() {
            return BindGroup::new(layout, entries, true);
        }
        let error = validate_bind_group_descriptor(self, &layout, &entries, self.limits());
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        BindGroup::new(layout, entries, is_error)
    }

    #[must_use]
    pub fn create_pipeline_layout(&self, descriptor: PipelineLayoutDescriptor) -> PipelineLayout {
        if self.is_lost() {
            return PipelineLayout::new(
                descriptor.bind_group_layouts,
                descriptor.immediate_size,
                true,
            );
        }
        let error = descriptor.error.clone().or_else(|| {
            validate_pipeline_layout_descriptor(
                &descriptor.bind_group_layouts,
                descriptor.immediate_size,
                self.limits(),
            )
        });
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        PipelineLayout::new(
            descriptor.bind_group_layouts,
            descriptor.immediate_size,
            is_error,
        )
    }

    #[must_use]
    pub fn create_command_encoder(&self) -> CommandEncoder {
        if self.is_lost() {
            CommandEncoder::new_error("command encoder device is lost")
        } else {
            CommandEncoder::new()
        }
    }

    #[must_use]
    pub fn create_compute_pipeline(
        &self,
        descriptor: ComputePipelineDescriptor,
    ) -> ComputePipeline {
        if self.is_lost() {
            return ComputePipeline::new(descriptor, true, self.limits(), None).0;
        }
        let error = descriptor
            .error
            .clone()
            .or_else(|| validate_compute_pipeline_descriptor(&descriptor, self.limits()));
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        let (pipeline, backend_error) =
            ComputePipeline::new(descriptor, is_error, self.limits(), Some(&self.inner.hal));
        if let Some(message) = backend_error {
            self.dispatch_error(ErrorKind::Internal, message);
        }
        pipeline
    }

    #[must_use]
    pub fn create_compute_pipeline_without_error_dispatch(
        &self,
        descriptor: ComputePipelineDescriptor,
    ) -> ComputePipeline {
        if self.is_lost() {
            return ComputePipeline::new(descriptor, true, self.limits(), None).0;
        }
        let error = descriptor
            .error
            .clone()
            .or_else(|| validate_compute_pipeline_descriptor(&descriptor, self.limits()));
        ComputePipeline::new(
            descriptor,
            error.is_some(),
            self.limits(),
            Some(&self.inner.hal),
        )
        .0
    }

    #[must_use]
    pub fn create_render_pipeline(&self, descriptor: RenderPipelineDescriptor) -> RenderPipeline {
        if self.is_lost() {
            return RenderPipeline::new(descriptor, true, self.limits(), None).0;
        }
        let error = descriptor
            .error
            .clone()
            .or_else(|| validate_render_pipeline_descriptor(&descriptor, self.limits()));
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        let (pipeline, backend_error) =
            RenderPipeline::new(descriptor, is_error, self.limits(), Some(&self.inner.hal));
        if let Some(message) = backend_error {
            self.dispatch_error(ErrorKind::Internal, message);
        }
        pipeline
    }

    #[must_use]
    pub fn create_render_pipeline_without_error_dispatch(
        &self,
        descriptor: RenderPipelineDescriptor,
    ) -> RenderPipeline {
        if self.is_lost() {
            return RenderPipeline::new(descriptor, true, self.limits(), None).0;
        }
        let error = descriptor
            .error
            .clone()
            .or_else(|| validate_render_pipeline_descriptor(&descriptor, self.limits()));
        RenderPipeline::new(
            descriptor,
            error.is_some(),
            self.limits(),
            Some(&self.inner.hal),
        )
        .0
    }
}

#[derive(Debug, Default)]
pub(crate) struct DeviceLostState {
    pub(crate) reason: Option<DeviceLostReason>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn device_from_hal_wraps_noop_hal_device() {
        let device = Device::from_hal(
            hal_noop_device(),
            Limits::DEFAULT,
            FeatureSet::new(),
            "",
            "",
        );

        assert!(matches!(device.hal(), yawgpu_hal::HalDevice::Noop(_)));
    }

    #[test]
    fn device_hal_limits_and_features_match_noop_contract() {
        let device = noop_device();

        assert!(matches!(device.hal(), yawgpu_hal::HalDevice::Noop(_)));
        assert_eq!(
            device.limits().max_bind_groups,
            Limits::DEFAULT.max_bind_groups
        );
        assert_eq!(
            device.limits().max_buffer_size,
            Limits::DEFAULT.max_buffer_size
        );
        assert!(device.features().contains(&Feature::CoreFeaturesAndLimits));
        assert!(device.has_feature(Feature::CoreFeaturesAndLimits));
        assert!(!device.has_feature(Feature::Other(99)));
    }

    #[test]
    fn device_create_query_set_validates_count_and_creates_happy_path() {
        let device = noop_device();

        let (error_query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "bad".to_owned(),
            kind: QueryType::Occlusion,
            count: 0,
        });
        assert!(error_query_set.is_error());
        assert_eq!(
            error,
            Some("query set count must be greater than zero".to_owned())
        );

        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "good".to_owned(),
            kind: QueryType::Occlusion,
            count: 4,
        });
        assert!(error.is_none());
        assert!(!query_set.is_error());
        assert_eq!(query_set.count(), 4);
    }

    #[test]
    fn device_error_new_constructs_with_kind_and_message() {
        let validation = DeviceError::new(ErrorKind::Validation, "validation message");
        assert_eq!(validation.kind, ErrorKind::Validation);
        assert_eq!(validation.message, "validation message");

        let internal = DeviceError::new(ErrorKind::Internal, String::from("internal message"));
        assert_eq!(internal.kind, ErrorKind::Internal);
        assert_eq!(internal.message, "internal message");
    }

    #[test]
    fn device_same_distinguishes_clone_from_distinct_device() {
        let device = noop_device();
        let clone = device.clone();
        let other = noop_device();

        assert!(device.same(&clone));
        assert!(!device.same(&other));
    }

    #[test]
    fn device_label_defaults_empty_and_set_label_updates_it() {
        let device = noop_device();

        assert_eq!(device.label(), "");
        device.set_label("renamed");
        assert_eq!(device.label(), "renamed");
    }

    #[test]
    fn device_destroy_lose_is_lost_and_lost_reason_are_idempotent() {
        let device = noop_device();

        assert!(!device.is_lost());
        assert_eq!(device.lost_reason(), None);
        assert_eq!(
            device.lose(DeviceLostReason::Unknown),
            Some(DeviceLostReason::Unknown)
        );
        assert!(device.is_lost());
        assert_eq!(device.lost_reason(), Some(DeviceLostReason::Unknown));
        assert_eq!(device.destroy(), None);

        let destroyed = noop_device();
        assert_eq!(destroyed.destroy(), Some(DeviceLostReason::Destroyed));
        assert_eq!(destroyed.destroy(), None);
        assert_eq!(destroyed.lost_reason(), Some(DeviceLostReason::Destroyed));
    }

    #[test]
    fn device_create_buffer_increments_allocation_count() {
        let device = noop_device();
        let before = device.allocation_count();

        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        });

        assert!(!buffer.is_error());
        assert_eq!(buffer.size(), 4);
        assert_eq!(buffer.usage(), BufferUsage::COPY_DST);
        assert_eq!(device.allocation_count(), before + 1);
    }

    #[test]
    fn device_create_texture_happy_path_and_invalid_size_scope_error() {
        let device = noop_device();
        let before = device.allocation_count();

        let texture = device.create_texture(valid_texture_descriptor());

        assert!(!texture.is_error());
        assert_eq!(texture.size().width, 1);
        assert_eq!(device.allocation_count(), before + 1);

        let mut invalid = valid_texture_descriptor();
        invalid.size.width = 0;
        device.push_error_scope(ErrorFilter::Validation);
        let error_texture = device.create_texture(invalid);
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid texture should be scoped");

        assert!(error_texture.is_error());
        assert_eq!(error.kind, ErrorKind::Validation);
        assert_eq!(error.message, "2D texture width is out of range");
    }

    #[test]
    fn device_create_sampler_uses_default_descriptor() {
        let device = noop_device();

        let sampler = device.create_sampler(SamplerDescriptor::default());

        assert!(!sampler.is_error());
        assert_eq!(
            sampler.descriptor().address_mode_u,
            AddressMode::ClampToEdge
        );
        assert_eq!(sampler.descriptor().mag_filter, FilterMode::Nearest);
    }

    #[test]
    fn device_create_shader_module_accepts_minimal_compute_wgsl() {
        let device = noop_device();

        let shader = device.create_shader_module(ShaderModuleSource::Wgsl(
            "@compute @workgroup_size(1) fn cs() {}".to_owned(),
        ));

        assert!(!shader.is_error());
        assert_eq!(shader.diagnostic(), None);
    }

    #[test]
    fn device_create_bind_group_layout_bind_group_and_pipeline_layout_empty() {
        let device = noop_device();

        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: Vec::new(),
            error: None,
        }));
        let bind_group = device.create_bind_group(layout.clone(), Vec::new());
        let pipeline_layout = device.create_pipeline_layout(PipelineLayoutDescriptor {
            bind_group_layouts: vec![layout.clone()],
            immediate_size: 0,
            error: None,
        });

        assert!(!layout.is_error());
        assert!(layout.entries().is_empty());
        assert!(!bind_group.is_error());
        assert!(bind_group.entries().is_empty());
        assert!(!pipeline_layout.is_error());
        assert_eq!(pipeline_layout.bind_group_layouts().len(), 1);
    }

    #[test]
    fn device_create_command_encoder_finishes_empty_encoder() {
        let device = noop_device();

        let encoder = device.create_command_encoder();
        let (command_buffer, error) = encoder.finish();

        assert!(error.is_none());
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn device_create_compute_pipeline_happy_path_and_error_scope() {
        let device = noop_device();
        let module = compute_shader_module(&device);

        let pipeline = device.create_compute_pipeline(compute_pipeline_descriptor(module.clone()));
        assert!(!pipeline.is_error());
        assert_eq!(pipeline.entry_name(), "cs");

        let mut invalid = compute_pipeline_descriptor(module);
        invalid.error = Some("forced compute pipeline error".to_owned());
        device.push_error_scope(ErrorFilter::Validation);
        let error_pipeline = device.create_compute_pipeline(invalid);
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid compute pipeline should be scoped");

        assert!(error_pipeline.is_error());
        assert_eq!(error.kind, ErrorKind::Validation);
        assert_eq!(error.message, "forced compute pipeline error");
    }

    #[test]
    fn device_create_compute_pipeline_without_error_dispatch_keeps_scope_empty() {
        let device = noop_device();
        let module = compute_shader_module(&device);
        let mut descriptor = compute_pipeline_descriptor(module);
        descriptor.error = Some("forced compute pipeline error".to_owned());

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline_without_error_dispatch(descriptor);
        let scoped = device.pop_error_scope().expect("scope should exist");

        assert!(pipeline.is_error());
        assert!(scoped.is_none());
    }

    #[test]
    fn device_create_render_pipeline_happy_path_and_error_scope() {
        let device = noop_device();
        let module = render_shader_module(&device);

        let pipeline = device.create_render_pipeline(render_pipeline_descriptor(module.clone()));
        assert!(!pipeline.is_error());
        assert_eq!(pipeline.vertex_entry_name(), "vs");
        assert_eq!(pipeline.fragment_entry_name(), Some("fs"));

        let mut invalid = render_pipeline_descriptor(module);
        invalid.error = Some("forced render pipeline error".to_owned());
        device.push_error_scope(ErrorFilter::Validation);
        let error_pipeline = device.create_render_pipeline(invalid);
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid render pipeline should be scoped");

        assert!(error_pipeline.is_error());
        assert_eq!(error.kind, ErrorKind::Validation);
        assert_eq!(error.message, "forced render pipeline error");
    }

    #[test]
    fn device_create_render_pipeline_without_error_dispatch_keeps_scope_empty() {
        let device = noop_device();
        let module = render_shader_module(&device);
        let mut descriptor = render_pipeline_descriptor(module);
        descriptor.error = Some("forced render pipeline error".to_owned());

        device.push_error_scope(ErrorFilter::Validation);
        let pipeline = device.create_render_pipeline_without_error_dispatch(descriptor);
        let scoped = device.pop_error_scope().expect("scope should exist");

        assert!(pipeline.is_error());
        assert!(scoped.is_none());
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
            .create_device(None, &[], "", "")
            .expect("Noop device should be created");
        let uncaptured_count = Arc::new(AtomicUsize::new(0));
        let callback_count = uncaptured_count.clone();

        device.set_uncaptured_error_callback(Some(move |_| {
            callback_count.fetch_add(1, Ordering::Relaxed);
        }));
        device.push_error_scope(super::ErrorFilter::Validation);
        device.dispatch_error(ErrorKind::Validation, "scoped validation error");

        let error = device
            .pop_error_scope()
            .expect("scope should exist")
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
            .create_device(None, &[], "", "")
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
}
