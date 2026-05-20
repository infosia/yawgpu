use std::cell::UnsafeCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{
    HalAdapter, HalAddressMode, HalBackend, HalBoundBuffer, HalBuffer, HalBufferBindingKind,
    HalBufferCopy, HalBufferTextureCopy, HalBufferTextureLayout, HalCompareFunction,
    HalComputePass, HalComputePipeline, HalCopy, HalDescriptorBinding, HalDevice, HalDraw,
    HalError, HalExtent3d, HalFilterMode, HalInstance, HalMipmapFilterMode, HalOrigin3d,
    HalPrimitiveTopology, HalQueue, HalRenderColorTarget, HalRenderLoadOp, HalRenderPass,
    HalRenderPipeline, HalRenderPipelineDescriptor, HalSampler, HalSamplerDescriptor,
    HalShaderSource, HalSurface, HalTexture, HalTextureCopy, HalTextureDescriptor,
    HalTextureFormat, HalTextureUsage, HalVertexAttribute, HalVertexBufferLayout, HalVertexFormat,
    HalVertexStepMode,
};

pub(crate) mod shader_naga;

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

    #[must_use]
    pub fn hal(&self) -> &HalInstance {
        &self.inner.hal
    }

    /// # Safety
    ///
    /// `layer` must be a valid, non-dangling `CAMetalLayer` instance pointer.
    pub unsafe fn create_surface_from_metal_layer(
        &self,
        layer: *mut std::ffi::c_void,
    ) -> Result<HalSurface, Error> {
        unsafe {
            self.inner
                .hal
                .create_surface_from_metal_layer(layer)
                .map_err(Error::Hal)
        }
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
    pub(crate) fn from_hal_with_feature_level(
        hal: HalAdapter,
        feature_level: FeatureLevel,
    ) -> Self {
        Self {
            inner: Arc::new(AdapterInner { hal, feature_level }),
        }
    }

    #[must_use]
    pub fn name(&self) -> String {
        self.inner.hal.name()
    }

    #[must_use]
    pub fn backend(&self) -> HalBackend {
        self.inner.hal.backend()
    }

    #[must_use]
    pub fn limits(&self) -> Limits {
        // Block 00: the synthetic Noop adapter's supported limits are the
        // WebGPU spec defaults by design.
        Limits::DEFAULT
    }

    #[must_use]
    pub(crate) fn feature_level(&self) -> FeatureLevel {
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
        label: impl Into<String>,
        queue_label: impl Into<String>,
    ) -> Result<Device, Error> {
        let limits = self
            .limits()
            .validate_required_limits(required_limits)
            .map_err(Error::Validation)?;
        let features = self.resolve_features(required_features)?;
        let hal = self.inner.hal.create_device()?;
        Ok(Device::from_hal(hal, limits, features, label, queue_label))
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
    label: Mutex<String>,
    limits: Limits,
    features: FeatureSet,
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
        let error = descriptor
            .error
            .clone()
            .or_else(|| validate_bind_group_layout_descriptor(&descriptor.entries, self.limits()));
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDescriptor {
    pub usage: BufferUsage,
    pub size: u64,
    pub mapped_at_creation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MapMode {
    Read,
    Write,
}

impl MapMode {
    pub fn from_bits(bits: u32) -> Result<Self, &'static str> {
        const READ: u32 = 1;
        const WRITE: u32 = 2;
        const ALLOWED: u32 = READ | WRITE;

        if bits & !ALLOWED != 0 {
            return Err("map mode has unsupported bits");
        }
        match bits {
            READ => Ok(Self::Read),
            WRITE => Ok(Self::Write),
            _ => Err("map mode must be exactly Read or Write"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MapAsyncStatus {
    Success,
    CallbackCancelled,
    Error,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueueWorkDoneStatus {
    Success,
    CallbackCancelled,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferUsage(u64);

impl BufferUsage {
    pub const NONE: Self = Self(0);
    pub const MAP_READ: Self = Self(1);
    pub const MAP_WRITE: Self = Self(2);
    pub const COPY_SRC: Self = Self(4);
    pub const COPY_DST: Self = Self(8);
    pub const INDEX: Self = Self(16);
    pub const VERTEX: Self = Self(32);
    pub const UNIFORM: Self = Self(64);
    pub const STORAGE: Self = Self(128);
    pub const INDIRECT: Self = Self(256);
    pub const QUERY_RESOLVE: Self = Self(512);

    #[must_use]
    pub fn from_bits_retain(bits: u64) -> Self {
        Self(bits)
    }

    #[must_use]
    pub fn bits(self) -> u64 {
        self.0
    }

    #[must_use]
    pub(crate) fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for BufferUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureUsage(u64);

impl TextureUsage {
    pub const NONE: Self = Self(0);
    pub const COPY_SRC: Self = Self(1);
    pub const COPY_DST: Self = Self(2);
    pub const TEXTURE_BINDING: Self = Self(4);
    pub const STORAGE_BINDING: Self = Self(8);
    pub const RENDER_ATTACHMENT: Self = Self(16);
    pub const TRANSIENT_ATTACHMENT: Self = Self(32);

    #[must_use]
    pub fn from_bits_retain(bits: u64) -> Self {
        Self(bits)
    }

    #[must_use]
    pub fn bits(self) -> u64 {
        self.0
    }

    #[must_use]
    pub(crate) fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for TextureUsage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureDimension {
    D1,
    D2,
    D3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureFormat(u32);

impl TextureFormat {
    const UNDEFINED: u32 = 0x00;
    const R8_UNORM: u32 = 0x01;
    const R8_SNORM: u32 = 0x02;
    const R8_UINT: u32 = 0x03;
    const R8_SINT: u32 = 0x04;
    const RG8_UNORM: u32 = 0x0A;
    const RG8_SNORM: u32 = 0x0B;
    const RG8_UINT: u32 = 0x0C;
    const RG8_SINT: u32 = 0x0D;
    const R32_FLOAT: u32 = 0x0E;
    const R32_UINT: u32 = 0x0F;
    const R32_SINT: u32 = 0x10;
    const RGBA8_UNORM: u32 = 0x16;
    const RGBA8_UNORM_SRGB: u32 = 0x17;
    const RGBA8_SNORM: u32 = 0x18;
    const RGBA8_UINT: u32 = 0x19;
    const RGBA8_SINT: u32 = 0x1A;
    const RG11B10_UFLOAT: u32 = 0x1F;
    const RGB9E5_UFLOAT: u32 = 0x20;
    const RG32_FLOAT: u32 = 0x21;
    const RG32_UINT: u32 = 0x22;
    const RG32_SINT: u32 = 0x23;
    const RGBA16_UNORM: u32 = 0x24;
    const RGBA16_SNORM: u32 = 0x25;
    const RGBA16_UINT: u32 = 0x26;
    const RGBA16_SINT: u32 = 0x27;
    const RGBA16_FLOAT: u32 = 0x28;
    const RGBA32_FLOAT: u32 = 0x29;
    const RGBA32_UINT: u32 = 0x2A;
    const RGBA32_SINT: u32 = 0x2B;
    const STENCIL8: u32 = 0x2C;
    const DEPTH16_UNORM: u32 = 0x2D;
    const DEPTH24_PLUS: u32 = 0x2E;
    const DEPTH24_PLUS_STENCIL8: u32 = 0x2F;
    const DEPTH32_FLOAT: u32 = 0x30;
    const DEPTH32_FLOAT_STENCIL8: u32 = 0x31;
    const BC1_RGBA_UNORM: u32 = 0x32;
    const BC1_RGBA_UNORM_SRGB: u32 = 0x33;
    const BGRA8_UNORM: u32 = 0x1B;
    const BGRA8_UNORM_SRGB: u32 = 0x1C;
    const BC7_RGBA_UNORM: u32 = 0x3E;
    const BC7_RGBA_UNORM_SRGB: u32 = 0x3F;

    #[must_use]
    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    #[must_use]
    pub fn raw(self) -> u32 {
        self.0
    }

    #[must_use]
    pub(crate) fn is_undefined(self) -> bool {
        self.0 == Self::UNDEFINED
    }

    #[must_use]
    pub fn caps(self) -> Option<FormatCaps> {
        if self.is_undefined() {
            return None;
        }

        let caps = match self.0 {
            Self::R8_UNORM => FormatCaps::float_color(1, 1)
                .blendable()
                .renderable()
                .multisample(),
            Self::R8_SNORM => FormatCaps::float_color(1, 1).blendable(),
            Self::R8_UINT => FormatCaps::uint_color(1, 1).renderable().multisample(),
            Self::R8_SINT => FormatCaps::sint_color(1, 1).renderable().multisample(),
            Self::RG8_UNORM => FormatCaps::float_color(2, 2)
                .blendable()
                .renderable()
                .multisample(),
            Self::RG8_SNORM => FormatCaps::float_color(2, 2).blendable(),
            Self::RG8_UINT => FormatCaps::uint_color(2, 2).renderable().multisample(),
            Self::RG8_SINT => FormatCaps::sint_color(2, 2).renderable().multisample(),
            Self::R32_FLOAT => FormatCaps::float_color(4, 1)
                .renderable()
                .multisample()
                .storage(),
            Self::R32_UINT => FormatCaps::uint_color(4, 1)
                .renderable()
                .multisample()
                .storage(),
            Self::R32_SINT => FormatCaps::sint_color(4, 1)
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA8_UNORM => FormatCaps::float_color(4, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA8_UNORM_SRGB => FormatCaps::float_color(4, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample(),
            Self::BGRA8_UNORM | Self::BGRA8_UNORM_SRGB => FormatCaps::float_color(4, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample(),
            // snorm formats are NOT storage-capable (Dawn `Format.cpp`).
            Self::RGBA8_SNORM => FormatCaps::float_color(4, 4).alpha().blendable(),
            Self::RGBA8_UINT => FormatCaps::uint_color(4, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA8_SINT => FormatCaps::sint_color(4, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            Self::RG11B10_UFLOAT | Self::RGB9E5_UFLOAT => FormatCaps::float_color(4, 3).blendable(),
            Self::RG32_FLOAT => FormatCaps::float_color(8, 2).renderable().storage(),
            Self::RG32_UINT => FormatCaps::uint_color(8, 2).renderable().storage(),
            Self::RG32_SINT => FormatCaps::sint_color(8, 2).renderable().storage(),
            Self::RGBA16_UNORM => FormatCaps::float_color(8, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample()
                .storage(),
            // snorm formats are NOT storage-capable (Dawn `Format.cpp`); the
            // remaining `*16` renderable/multisample approximation stays a
            // tracked note (block 20 → P4/P5).
            Self::RGBA16_SNORM => FormatCaps::float_color(8, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample(),
            Self::RGBA16_UINT => FormatCaps::uint_color(8, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA16_SINT => FormatCaps::sint_color(8, 4)
                .alpha()
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA16_FLOAT => FormatCaps::float_color(8, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample()
                .storage(),
            Self::RGBA32_FLOAT => FormatCaps::float_color(16, 4)
                .alpha()
                .renderable()
                .storage(),
            Self::RGBA32_UINT => FormatCaps::uint_color(16, 4).alpha().renderable().storage(),
            Self::RGBA32_SINT => FormatCaps::sint_color(16, 4).alpha().renderable().storage(),
            Self::STENCIL8 => FormatCaps::stencil(1).renderable().multisample(),
            Self::DEPTH16_UNORM => FormatCaps::depth(2).renderable().multisample(),
            Self::DEPTH24_PLUS => FormatCaps::depth(4).renderable().multisample(),
            Self::DEPTH24_PLUS_STENCIL8 => FormatCaps::depth_stencil(4).renderable().multisample(),
            Self::DEPTH32_FLOAT => FormatCaps::depth(4).renderable().multisample(),
            Self::DEPTH32_FLOAT_STENCIL8 => FormatCaps::depth_stencil(5).renderable().multisample(),
            Self::BC1_RGBA_UNORM | Self::BC1_RGBA_UNORM_SRGB => {
                FormatCaps::compressed_color(8, 4, 4)
            }
            Self::BC7_RGBA_UNORM | Self::BC7_RGBA_UNORM_SRGB => {
                FormatCaps::compressed_color(16, 4, 4)
            }
            // Unknown defined formats are unsupported until explicitly modeled.
            _ => return None,
        };
        Some(caps)
    }

    #[must_use]
    pub(crate) fn srgb_pair(self) -> Option<Self> {
        let pair = match self.0 {
            Self::RGBA8_UNORM => Self::RGBA8_UNORM_SRGB,
            Self::RGBA8_UNORM_SRGB => Self::RGBA8_UNORM,
            Self::BGRA8_UNORM => Self::BGRA8_UNORM_SRGB,
            Self::BGRA8_UNORM_SRGB => Self::BGRA8_UNORM,
            Self::BC1_RGBA_UNORM => Self::BC1_RGBA_UNORM_SRGB,
            Self::BC1_RGBA_UNORM_SRGB => Self::BC1_RGBA_UNORM,
            Self::BC7_RGBA_UNORM => Self::BC7_RGBA_UNORM_SRGB,
            Self::BC7_RGBA_UNORM_SRGB => Self::BC7_RGBA_UNORM,
            _ => return None,
        };
        Some(Self(pair))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatAspects {
    pub color: bool,
    pub depth: bool,
    pub stencil: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatCaps {
    pub aspects: FormatAspects,
    pub renderable: bool,
    pub multisample_capable: bool,
    pub storage_capable: bool,
    pub output_class: Option<FormatOutputClass>,
    pub color_components: u8,
    pub is_blendable: bool,
    pub has_alpha: bool,
    pub is_compressed: bool,
    pub texel_block_size: u32,
    pub block_w: u32,
    pub block_h: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatOutputClass {
    Float,
    Sint,
    Uint,
}

impl FormatCaps {
    const fn float_color(texel_block_size: u32, color_components: u8) -> Self {
        Self::color(texel_block_size, color_components, FormatOutputClass::Float)
    }

    const fn sint_color(texel_block_size: u32, color_components: u8) -> Self {
        Self::color(texel_block_size, color_components, FormatOutputClass::Sint)
    }

    const fn uint_color(texel_block_size: u32, color_components: u8) -> Self {
        Self::color(texel_block_size, color_components, FormatOutputClass::Uint)
    }

    const fn color(
        texel_block_size: u32,
        color_components: u8,
        output_class: FormatOutputClass,
    ) -> Self {
        Self::new(
            FormatAspects {
                color: true,
                depth: false,
                stencil: false,
            },
            texel_block_size,
            1,
            1,
            false,
            Some(output_class),
            color_components,
        )
    }

    const fn depth(texel_block_size: u32) -> Self {
        Self::new(
            FormatAspects {
                color: false,
                depth: true,
                stencil: false,
            },
            texel_block_size,
            1,
            1,
            false,
            None,
            0,
        )
    }

    const fn stencil(texel_block_size: u32) -> Self {
        Self::new(
            FormatAspects {
                color: false,
                depth: false,
                stencil: true,
            },
            texel_block_size,
            1,
            1,
            false,
            None,
            0,
        )
    }

    const fn depth_stencil(texel_block_size: u32) -> Self {
        Self::new(
            FormatAspects {
                color: false,
                depth: true,
                stencil: true,
            },
            texel_block_size,
            1,
            1,
            false,
            None,
            0,
        )
    }

    const fn compressed_color(texel_block_size: u32, block_w: u32, block_h: u32) -> Self {
        Self::new(
            FormatAspects {
                color: true,
                depth: false,
                stencil: false,
            },
            texel_block_size,
            block_w,
            block_h,
            true,
            Some(FormatOutputClass::Float),
            4,
        )
    }

    const fn new(
        aspects: FormatAspects,
        texel_block_size: u32,
        block_w: u32,
        block_h: u32,
        is_compressed: bool,
        output_class: Option<FormatOutputClass>,
        color_components: u8,
    ) -> Self {
        Self {
            aspects,
            renderable: false,
            multisample_capable: false,
            storage_capable: false,
            output_class,
            color_components,
            is_blendable: false,
            has_alpha: false,
            is_compressed,
            texel_block_size,
            block_w,
            block_h,
        }
    }

    const fn renderable(mut self) -> Self {
        self.renderable = true;
        self
    }

    const fn multisample(mut self) -> Self {
        self.multisample_capable = true;
        self
    }

    const fn storage(mut self) -> Self {
        self.storage_capable = true;
        self
    }

    const fn blendable(mut self) -> Self {
        self.is_blendable = true;
        self
    }

    const fn alpha(mut self) -> Self {
        self.has_alpha = true;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Origin3d {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TexelCopyBufferLayout {
    pub offset: u64,
    pub bytes_per_row: Option<u32>,
    pub rows_per_image: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct TexelCopyBufferInfo {
    pub buffer: Arc<Buffer>,
    pub layout: TexelCopyBufferLayout,
}

#[derive(Debug, Clone)]
pub struct TexelCopyTextureInfo {
    pub texture: Arc<Texture>,
    pub mip_level: u32,
    pub origin: Origin3d,
    pub aspect: TextureAspect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOp {
    Undefined,
    Load,
    Clear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOp {
    Undefined,
    Store,
    Discard,
}

#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

#[derive(Debug, Clone)]
pub struct RenderPassDescriptor {
    pub max_color_attachments: u32,
    pub color_attachments: Vec<Option<RenderPassColorAttachment>>,
    pub depth_stencil_attachment: Option<RenderPassDepthStencilAttachment>,
    pub occlusion_query_set: Option<QuerySet>,
    pub timestamp_writes: Option<RenderPassTimestampWrites>,
}

#[derive(Debug, Clone)]
pub struct RenderPassTimestampWrites {
    pub query_set: QuerySet,
    pub beginning_index: Option<u32>,
    pub end_index: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct RenderPassColorAttachment {
    pub view: Arc<TextureView>,
    pub resolve_target: Option<Arc<TextureView>>,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_value: Color,
}

#[derive(Debug, Clone)]
pub struct RenderPassDepthStencilAttachment {
    pub view: Arc<TextureView>,
    pub depth_load_op: LoadOp,
    pub depth_store_op: StoreOp,
    pub depth_clear_value: f32,
    pub stencil_load_op: LoadOp,
    pub stencil_store_op: StoreOp,
}

#[derive(Debug, Clone)]
pub struct RenderBundleEncoderDescriptor {
    pub max_color_attachments: u32,
    pub color_formats: Vec<Option<TextureFormat>>,
    pub depth_stencil_format: Option<TextureFormat>,
    pub sample_count: u32,
    pub depth_read_only: bool,
    pub stencil_read_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AttachmentSignature {
    color_formats: Vec<Option<TextureFormat>>,
    depth_stencil_format: Option<TextureFormat>,
    sample_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureDescriptor {
    pub usage: TextureUsage,
    pub dimension: TextureDimension,
    pub size: Extent3d,
    pub format: TextureFormat,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub view_formats: Vec<TextureFormat>,
}

#[derive(Debug, Clone)]
pub struct Texture {
    inner: Arc<TextureInner>,
}

#[derive(Debug)]
struct TextureInner {
    hal: Option<HalTexture>,
    usage: TextureUsage,
    dimension: TextureDimension,
    size: Extent3d,
    format: TextureFormat,
    mip_level_count: u32,
    sample_count: u32,
    view_formats: Vec<TextureFormat>,
    state: Mutex<TextureState>,
}

#[derive(Debug)]
struct TextureState {
    is_error: bool,
    is_destroyed: bool,
}

impl Texture {
    fn new(descriptor: TextureDescriptor, hal: Option<HalTexture>, is_error: bool) -> Self {
        Self {
            inner: Arc::new(TextureInner {
                hal,
                usage: descriptor.usage,
                dimension: descriptor.dimension,
                size: descriptor.size,
                format: descriptor.format,
                mip_level_count: descriptor.mip_level_count,
                sample_count: descriptor.sample_count,
                view_formats: descriptor.view_formats,
                state: Mutex::new(TextureState {
                    is_error,
                    is_destroyed: false,
                }),
            }),
        }
    }

    #[must_use]
    pub fn from_hal(descriptor: TextureDescriptor, hal: HalTexture) -> Self {
        Self::new(descriptor, Some(hal), false)
    }

    #[must_use]
    pub fn usage(&self) -> TextureUsage {
        self.inner.usage
    }

    #[must_use]
    pub fn dimension(&self) -> TextureDimension {
        self.inner.dimension
    }

    #[must_use]
    pub fn size(&self) -> Extent3d {
        self.inner.size
    }

    #[must_use]
    pub fn format(&self) -> TextureFormat {
        self.inner.format
    }

    #[must_use]
    pub fn mip_level_count(&self) -> u32 {
        self.inner.mip_level_count
    }

    #[must_use]
    pub fn sample_count(&self) -> u32 {
        self.inner.sample_count
    }

    #[must_use]
    pub(crate) fn view_formats(&self) -> &[TextureFormat] {
        &self.inner.view_formats
    }

    /// A view format is compatible only when it equals the texture's format
    /// or is explicitly listed in the texture's `viewFormats`. There is no
    /// implicit sRGB-counterpart allowance — that mirrors Dawn
    /// `Texture.cpp` `ValidateCanViewTextureAs`.
    #[must_use]
    pub(crate) fn is_view_format_compatible(&self, view_format: TextureFormat) -> bool {
        view_format == self.format() || self.view_formats().contains(&view_format)
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.state.lock().is_error
    }

    #[must_use]
    pub(crate) fn is_destroyed(&self) -> bool {
        self.inner.state.lock().is_destroyed
    }

    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn hal(&self) -> Option<HalTexture> {
        self.inner.hal.clone()
    }

    pub fn destroy(&self) {
        self.inner.state.lock().is_destroyed = true;
    }

    #[must_use]
    pub fn create_view(
        &self,
        descriptor: TextureViewDescriptor,
    ) -> (TextureView, Option<&'static str>) {
        let resolved = self.resolve_view_descriptor(descriptor);
        let error = if self.is_error() {
            Some("cannot create a view from an error texture")
        } else {
            validate_texture_view_descriptor(self, &resolved)
        };
        let is_error = error.is_some();
        (TextureView::new(self.clone(), resolved, is_error), error)
    }

    fn resolve_view_descriptor(
        &self,
        descriptor: TextureViewDescriptor,
    ) -> ResolvedTextureViewDescriptor {
        let base_mip_level = descriptor.base_mip_level;
        let base_array_layer = descriptor.base_array_layer;
        let mip_level_count = descriptor
            .mip_level_count
            .unwrap_or_else(|| self.mip_level_count().saturating_sub(base_mip_level));
        let array_layer_count =
            descriptor
                .array_layer_count
                .unwrap_or_else(|| match self.dimension() {
                    TextureDimension::D1 => 1,
                    TextureDimension::D2 => self
                        .size()
                        .depth_or_array_layers
                        .saturating_sub(base_array_layer),
                    TextureDimension::D3 => self.size().depth_or_array_layers,
                });
        let dimension = descriptor
            .dimension
            .unwrap_or_else(|| match self.dimension() {
                TextureDimension::D1 => TextureViewDimension::D1,
                TextureDimension::D3 => TextureViewDimension::D3,
                TextureDimension::D2 if array_layer_count == 1 => TextureViewDimension::D2,
                TextureDimension::D2 => TextureViewDimension::D2Array,
            });

        ResolvedTextureViewDescriptor {
            format: descriptor.format.unwrap_or_else(|| self.format()),
            dimension,
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            aspect: descriptor.aspect.unwrap_or(TextureAspect::All),
        }
    }

    pub fn validate_queue_write(
        &self,
        mip_level: u32,
        origin: Origin3d,
        write_size: Extent3d,
        aspect: TextureAspect,
        layout: TexelCopyBufferLayout,
        data_size: u64,
    ) -> Result<(), String> {
        validate_queue_write_texture(
            self, mip_level, origin, write_size, aspect, layout, data_size,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureAspect {
    All,
    DepthOnly,
    StencilOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureViewDescriptor {
    pub format: Option<TextureFormat>,
    pub dimension: Option<TextureViewDimension>,
    pub base_mip_level: u32,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: u32,
    pub array_layer_count: Option<u32>,
    pub aspect: Option<TextureAspect>,
}

/// A `TextureViewDescriptor` with every defaulted/inferred field already
/// filled in by `Texture::resolve_view_descriptor`. Validation and view
/// construction take this so an unresolved descriptor can't be validated
/// or stored by mistake.
#[derive(Debug, Clone, Copy)]
struct ResolvedTextureViewDescriptor {
    format: TextureFormat,
    dimension: TextureViewDimension,
    base_mip_level: u32,
    mip_level_count: u32,
    base_array_layer: u32,
    array_layer_count: u32,
    aspect: TextureAspect,
}

#[derive(Debug, Clone)]
pub struct TextureView {
    inner: Arc<TextureViewInner>,
}

#[derive(Debug)]
struct TextureViewInner {
    texture: Texture,
    format: TextureFormat,
    dimension: TextureViewDimension,
    base_mip_level: u32,
    mip_level_count: u32,
    base_array_layer: u32,
    array_layer_count: u32,
    aspect: TextureAspect,
    is_error: bool,
}

impl TextureView {
    fn new(texture: Texture, descriptor: ResolvedTextureViewDescriptor, is_error: bool) -> Self {
        Self {
            inner: Arc::new(TextureViewInner {
                texture,
                format: descriptor.format,
                dimension: descriptor.dimension,
                base_mip_level: descriptor.base_mip_level,
                mip_level_count: descriptor.mip_level_count,
                base_array_layer: descriptor.base_array_layer,
                array_layer_count: descriptor.array_layer_count,
                aspect: descriptor.aspect,
                is_error,
            }),
        }
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub(crate) fn texture(&self) -> Texture {
        self.inner.texture.clone()
    }

    #[must_use]
    pub fn format(&self) -> TextureFormat {
        self.inner.format
    }

    #[must_use]
    pub fn dimension(&self) -> TextureViewDimension {
        self.inner.dimension
    }

    #[must_use]
    pub(crate) fn base_mip_level(&self) -> u32 {
        self.inner.base_mip_level
    }

    #[must_use]
    pub fn mip_level_count(&self) -> u32 {
        self.inner.mip_level_count
    }

    #[must_use]
    pub fn base_array_layer(&self) -> u32 {
        self.inner.base_array_layer
    }

    #[must_use]
    pub(crate) fn array_layer_count(&self) -> u32 {
        self.inner.array_layer_count
    }

    #[must_use]
    pub fn aspect(&self) -> TextureAspect {
        self.inner.aspect
    }

    #[must_use]
    pub(crate) fn render_extent(&self) -> Extent3d {
        let subresource = self.texture().subresource_size(self.base_mip_level());
        Extent3d {
            width: subresource.width,
            height: subresource.height,
            depth_or_array_layers: 1,
        }
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ShaderModuleSource {
    Wgsl(String),
    Spirv(Vec<u32>),
    Invalid(String),
}

#[derive(Debug, Clone)]
pub struct ShaderModule {
    inner: Arc<ShaderModuleInner>,
}

#[derive(Debug)]
struct ShaderModuleInner {
    _source: ShaderModuleSourceKind,
    diagnostic: Option<String>,
    is_error: bool,
}

#[derive(Debug)]
enum ShaderModuleSourceKind {
    Wgsl {
        _source: String,
        validated: Box<shader_naga::ValidatedWgslModule>,
    },
    Spirv {
        _words: Vec<u32>,
    },
    Invalid,
}

impl ShaderModule {
    fn new(source: ShaderModuleSourceKind, diagnostic: Option<String>) -> Self {
        Self {
            inner: Arc::new(ShaderModuleInner {
                is_error: diagnostic.is_some(),
                _source: source,
                diagnostic,
            }),
        }
    }

    fn from_wgsl(source: String) -> Result<ShaderModuleSourceKind, String> {
        let validated = shader_naga::parse_and_validate_wgsl(&source)?;
        validate_wgsl_module_limits(&validated.module)?;
        Ok(ShaderModuleSourceKind::Wgsl {
            _source: source,
            validated: Box::new(validated),
        })
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub fn diagnostic(&self) -> Option<&str> {
        self.inner.diagnostic.as_deref()
    }

    #[must_use]
    fn validated_wgsl(&self) -> Option<&shader_naga::ValidatedWgslModule> {
        match &self.inner._source {
            ShaderModuleSourceKind::Wgsl { validated, .. } => Some(validated),
            _ => None,
        }
    }
}

fn validate_wgsl_module_limits(module: &naga::Module) -> Result<(), String> {
    let mut ids = BTreeSet::new();
    for (_, override_) in module.overrides.iter() {
        if let Some(id) = override_.id {
            if !ids.insert(id) {
                return Err(format!("duplicate shader override id {id}"));
            }
        }
    }

    for (_, global) in module.global_variables.iter() {
        if let Some(binding) = global.binding {
            if binding.binding >= 1000 {
                return Err(format!(
                    "shader resource binding {} exceeds the maximum binding number",
                    binding.binding
                ));
            }
        }
    }

    Ok(())
}

fn validate_bind_group_layout_descriptor(
    entries: &[BindGroupLayoutEntry],
    limits: Limits,
) -> Option<String> {
    if entries.len() > 1000 {
        return Some("bind group layout entry count exceeds 1000".to_owned());
    }

    let mut bindings = BTreeSet::new();
    let mut dynamic_uniform_buffers = 0_u32;
    let mut dynamic_storage_buffers = 0_u32;
    let mut stage_counts = [StageResourceCounts::default(); 3];

    for entry in entries {
        if entry.binding >= 1000 {
            return Some("bind group layout binding must be less than 1000".to_owned());
        }
        if !bindings.insert(entry.binding) {
            return Some("bind group layout bindings must be unique".to_owned());
        }
        if entry.binding_array_size > 1 {
            return Some(
                "bind group layout bindingArraySize greater than one is not supported".to_owned(),
            );
        }

        let Some(kind) = entry.kind else {
            return Some("bind group layout entry must set exactly one binding layout".to_owned());
        };

        match kind {
            BindingLayoutKind::Buffer {
                ty,
                has_dynamic_offset,
                ..
            } => {
                match ty {
                    BufferBindingType::Uniform if has_dynamic_offset => {
                        dynamic_uniform_buffers += 1;
                    }
                    BufferBindingType::Storage | BufferBindingType::ReadOnlyStorage
                        if has_dynamic_offset =>
                    {
                        dynamic_storage_buffers += 1;
                    }
                    _ => {}
                }
                if dynamic_uniform_buffers > limits.max_dynamic_uniform_buffers_per_pipeline_layout
                {
                    return Some(
                        "too many dynamic uniform buffers in bind group layout".to_owned(),
                    );
                }
                if dynamic_storage_buffers > limits.max_dynamic_storage_buffers_per_pipeline_layout
                {
                    return Some(
                        "too many dynamic storage buffers in bind group layout".to_owned(),
                    );
                }
            }
            BindingLayoutKind::Texture {
                view_dimension,
                multisampled,
                ..
            } => {
                if multisampled && view_dimension != TextureViewDimension::D2 {
                    return Some(
                        "multisampled texture bindings require 2D view dimension".to_owned(),
                    );
                }
            }
            BindingLayoutKind::StorageTexture {
                format,
                view_dimension,
                ..
            } => {
                if view_dimension == TextureViewDimension::D1 {
                    return Some(
                        "storage texture bindings must not use 1D view dimension".to_owned(),
                    );
                }
                let Some(caps) = format.caps() else {
                    return Some("storage texture binding format must not be Undefined".to_owned());
                };
                if !caps.storage_capable {
                    return Some(
                        "storage texture binding format must support storage usage".to_owned(),
                    );
                }
            }
            BindingLayoutKind::Sampler { .. } => {}
        }

        for stage in visible_stages(entry.visibility) {
            stage_counts[stage].add(kind);
            if stage_counts[stage].sampled_textures > limits.max_sampled_textures_per_shader_stage {
                return Some("too many sampled textures for one shader stage".to_owned());
            }
            if stage_counts[stage].samplers > limits.max_samplers_per_shader_stage {
                return Some("too many samplers for one shader stage".to_owned());
            }
            if stage_counts[stage].storage_buffers > limits.max_storage_buffers_per_shader_stage {
                return Some("too many storage buffers for one shader stage".to_owned());
            }
            if stage_counts[stage].storage_textures > limits.max_storage_textures_per_shader_stage {
                return Some("too many storage textures for one shader stage".to_owned());
            }
            if stage_counts[stage].uniform_buffers > limits.max_uniform_buffers_per_shader_stage {
                return Some("too many uniform buffers for one shader stage".to_owned());
            }
        }
    }

    None
}

#[derive(Debug, Clone, Copy, Default)]
struct StageResourceCounts {
    sampled_textures: u32,
    samplers: u32,
    storage_buffers: u32,
    storage_textures: u32,
    uniform_buffers: u32,
}

impl StageResourceCounts {
    fn add(&mut self, kind: BindingLayoutKind) {
        match kind {
            BindingLayoutKind::Buffer {
                ty: BufferBindingType::Uniform,
                ..
            } => self.uniform_buffers += 1,
            BindingLayoutKind::Buffer {
                ty: BufferBindingType::Storage | BufferBindingType::ReadOnlyStorage,
                ..
            } => self.storage_buffers += 1,
            BindingLayoutKind::Sampler { .. } => self.samplers += 1,
            BindingLayoutKind::Texture { .. } => self.sampled_textures += 1,
            BindingLayoutKind::StorageTexture { .. } => self.storage_textures += 1,
        }
    }
}

fn visible_stages(visibility: u64) -> impl Iterator<Item = usize> {
    const VERTEX: u64 = 1;
    const FRAGMENT: u64 = 2;
    const COMPUTE: u64 = 4;
    [VERTEX, FRAGMENT, COMPUTE]
        .into_iter()
        .enumerate()
        .filter_map(move |(index, bit)| (visibility & bit != 0).then_some(index))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindGroupLayoutDescriptor {
    pub entries: Vec<BindGroupLayoutEntry>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: u64,
    pub binding_array_size: u32,
    pub kind: Option<BindingLayoutKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BindingLayoutKind {
    Buffer {
        ty: BufferBindingType,
        has_dynamic_offset: bool,
        min_binding_size: u64,
    },
    Sampler {
        ty: SamplerBindingType,
    },
    Texture {
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        access: StorageTextureAccess,
        format: TextureFormat,
        view_dimension: TextureViewDimension,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferBindingType {
    Uniform,
    Storage,
    ReadOnlyStorage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SamplerBindingType {
    Filtering,
    NonFiltering,
    Comparison,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TextureSampleType {
    Float,
    UnfilterableFloat,
    Depth,
    Sint,
    Uint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StorageTextureAccess {
    WriteOnly,
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub struct BindGroupLayout {
    inner: Arc<BindGroupLayoutInner>,
}

#[derive(Debug)]
struct BindGroupLayoutInner {
    entries: Vec<BindGroupLayoutEntry>,
    is_error: bool,
    is_default: bool,
}

impl BindGroupLayout {
    fn new(entries: Vec<BindGroupLayoutEntry>, is_error: bool, is_default: bool) -> Self {
        Self {
            inner: Arc::new(BindGroupLayoutInner {
                entries,
                is_error,
                is_default,
            }),
        }
    }

    #[must_use]
    pub fn error() -> Self {
        Self::new(Vec::new(), true, false)
    }

    #[must_use]
    pub fn entries(&self) -> &[BindGroupLayoutEntry] {
        &self.inner.entries
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub(crate) fn is_default(&self) -> bool {
        self.inner.is_default
    }

    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

#[derive(Debug, Clone)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub resource: BindGroupResource,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BindGroupResource {
    Buffer {
        buffer: Arc<Buffer>,
        device: Arc<Device>,
        offset: u64,
        size: u64,
    },
    Sampler {
        sampler: Arc<Sampler>,
        device: Arc<Device>,
    },
    TextureView {
        texture_view: Arc<TextureView>,
        device: Arc<Device>,
    },
    Invalid(String),
}

#[derive(Debug, Clone)]
pub struct BindGroup {
    inner: Arc<BindGroupInner>,
}

#[derive(Debug)]
struct BindGroupInner {
    _layout: Arc<BindGroupLayout>,
    _entries: Vec<BindGroupEntry>,
    is_error: bool,
}

impl BindGroup {
    fn new(layout: Arc<BindGroupLayout>, entries: Vec<BindGroupEntry>, is_error: bool) -> Self {
        Self {
            inner: Arc::new(BindGroupInner {
                _layout: layout,
                _entries: entries,
                is_error,
            }),
        }
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub(crate) fn layout(&self) -> &Arc<BindGroupLayout> {
        &self.inner._layout
    }

    #[must_use]
    pub fn entries(&self) -> &[BindGroupEntry] {
        &self.inner._entries
    }
}

fn validate_bind_group_descriptor(
    device: &Device,
    layout: &BindGroupLayout,
    entries: &[BindGroupEntry],
    limits: Limits,
) -> Option<String> {
    if layout.is_error() {
        return Some("cannot create bind group from an error bind group layout".to_owned());
    }
    if entries.len() != layout.entries().len() {
        return Some("bind group entry count must match bind group layout".to_owned());
    }

    let layout_entries = layout
        .entries()
        .iter()
        .map(|entry| (entry.binding, entry))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();

    for entry in entries {
        if !seen.insert(entry.binding) {
            return Some("bind group binding must not be set more than once".to_owned());
        }
        let Some(layout_entry) = layout_entries.get(&entry.binding).copied() else {
            return Some("bind group entry binding is not present in the layout".to_owned());
        };
        let Some(kind) = layout_entry.kind else {
            return Some("cannot create bind group from an invalid bind group layout".to_owned());
        };
        if let Some(message) = validate_bind_group_entry(device, entry, kind, limits) {
            return Some(message);
        }
    }

    for layout_entry in layout.entries() {
        if !seen.contains(&layout_entry.binding) {
            return Some("bind group is missing a layout binding".to_owned());
        }
    }

    None
}

fn validate_bind_group_entry(
    device: &Device,
    entry: &BindGroupEntry,
    kind: BindingLayoutKind,
    limits: Limits,
) -> Option<String> {
    match (&entry.resource, kind) {
        (
            BindGroupResource::Buffer {
                buffer,
                device: resource_device,
                offset,
                size,
            },
            BindingLayoutKind::Buffer {
                ty,
                min_binding_size,
                ..
            },
        ) => validate_bind_group_buffer(
            device,
            resource_device,
            BindGroupBufferValidation {
                buffer,
                offset: *offset,
                size: *size,
                ty,
                min_binding_size,
                limits,
            },
        ),
        (
            BindGroupResource::Sampler {
                sampler,
                device: resource_device,
            },
            BindingLayoutKind::Sampler { .. },
        ) => {
            if !device.same(resource_device) {
                Some("bind group sampler must belong to the same device".to_owned())
            } else if sampler.is_error() {
                Some("bind group sampler must not be an error sampler".to_owned())
            } else {
                None
            }
        }
        (
            BindGroupResource::TextureView {
                texture_view,
                device: resource_device,
            },
            BindingLayoutKind::Texture {
                sample_type,
                view_dimension,
                multisampled,
            },
        ) => validate_bind_group_texture(
            device,
            resource_device,
            texture_view,
            sample_type,
            view_dimension,
            multisampled,
        ),
        (
            BindGroupResource::TextureView {
                texture_view,
                device: resource_device,
            },
            BindingLayoutKind::StorageTexture { view_dimension, .. },
        ) => validate_bind_group_storage_texture(
            device,
            resource_device,
            texture_view,
            view_dimension,
        ),
        (BindGroupResource::Invalid(message), _) => Some(message.clone()),
        _ => Some("bind group entry resource kind must match the layout".to_owned()),
    }
}

fn validate_bind_group_buffer(
    device: &Device,
    resource_device: &Device,
    validation: BindGroupBufferValidation<'_>,
) -> Option<String> {
    if !device.same(resource_device) {
        return Some("bind group buffer must belong to the same device".to_owned());
    }
    let BindGroupBufferValidation {
        buffer,
        offset,
        size,
        ty,
        min_binding_size,
        limits,
    } = validation;
    if buffer.is_error() {
        return Some("bind group buffer must not be an error buffer".to_owned());
    }

    let (required_usage, alignment, max_binding_size) = match ty {
        BufferBindingType::Uniform => (
            BufferUsage::UNIFORM,
            u64::from(limits.min_uniform_buffer_offset_alignment),
            limits.max_uniform_buffer_binding_size,
        ),
        BufferBindingType::Storage | BufferBindingType::ReadOnlyStorage => (
            BufferUsage::STORAGE,
            u64::from(limits.min_storage_buffer_offset_alignment),
            limits.max_storage_buffer_binding_size,
        ),
    };

    if !buffer.usage().contains(required_usage) {
        return Some("bind group buffer usage does not satisfy the layout".to_owned());
    }
    if alignment != 0 && !offset.is_multiple_of(alignment) {
        return Some("bind group buffer offset is not correctly aligned".to_owned());
    }

    let effective_size = if size == u64::MAX {
        let Some(remaining) = buffer.size().checked_sub(offset) else {
            return Some("bind group buffer offset exceeds buffer size".to_owned());
        };
        remaining
    } else {
        size
    };
    if effective_size == 0 {
        return Some("bind group buffer binding size must be greater than zero".to_owned());
    }
    if offset
        .checked_add(effective_size)
        .is_none_or(|end| end > buffer.size())
    {
        return Some("bind group buffer binding range exceeds buffer size".to_owned());
    }
    if min_binding_size != 0 && effective_size < min_binding_size {
        return Some("bind group buffer binding size is below the layout minimum".to_owned());
    }
    if effective_size > max_binding_size {
        return Some("bind group buffer binding size exceeds the device limit".to_owned());
    }

    None
}

#[derive(Debug, Clone, Copy)]
struct BindGroupBufferValidation<'a> {
    buffer: &'a Buffer,
    offset: u64,
    size: u64,
    ty: BufferBindingType,
    min_binding_size: u64,
    limits: Limits,
}

fn validate_bind_group_texture(
    device: &Device,
    resource_device: &Device,
    texture_view: &TextureView,
    sample_type: TextureSampleType,
    view_dimension: TextureViewDimension,
    multisampled: bool,
) -> Option<String> {
    if !device.same(resource_device) {
        return Some("bind group texture view must belong to the same device".to_owned());
    }
    if texture_view.is_error() {
        return Some("bind group texture view must not be an error texture view".to_owned());
    }
    let texture = texture_view.texture();
    if !texture.usage().contains(TextureUsage::TEXTURE_BINDING) {
        return Some("bind group texture usage does not satisfy the layout".to_owned());
    }
    if texture_view.dimension() != view_dimension {
        return Some("bind group texture view dimension must match the layout".to_owned());
    }
    if (texture.sample_count() > 1) != multisampled {
        return Some("bind group texture multisampling must match the layout".to_owned());
    }
    if texture_view.format().caps().is_some_and(|caps| {
        (caps.aspects.depth || caps.aspects.stencil) && sample_type == TextureSampleType::Float
    }) {
        return Some("depth or stencil texture bindings must not use Float sample type".to_owned());
    }

    None
}

fn validate_bind_group_storage_texture(
    device: &Device,
    resource_device: &Device,
    texture_view: &TextureView,
    view_dimension: TextureViewDimension,
) -> Option<String> {
    if !device.same(resource_device) {
        return Some("bind group texture view must belong to the same device".to_owned());
    }
    if texture_view.is_error() {
        return Some("bind group texture view must not be an error texture view".to_owned());
    }
    let texture = texture_view.texture();
    if !texture.usage().contains(TextureUsage::STORAGE_BINDING) {
        return Some("bind group texture usage does not satisfy the layout".to_owned());
    }
    if texture_view.dimension() != view_dimension {
        return Some("bind group texture view dimension must match the layout".to_owned());
    }
    if texture_view.array_layer_count() != 1 {
        return Some("storage texture bindings require a single array layer".to_owned());
    }

    None
}

#[derive(Debug, Clone)]
pub struct PipelineLayoutDescriptor {
    pub bind_group_layouts: Vec<Arc<BindGroupLayout>>,
    pub immediate_size: u32,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PipelineLayout {
    inner: Arc<PipelineLayoutInner>,
}

#[derive(Debug)]
struct PipelineLayoutInner {
    _bind_group_layouts: Vec<Arc<BindGroupLayout>>,
    _immediate_size: u32,
    is_error: bool,
}

impl PipelineLayout {
    fn new(
        bind_group_layouts: Vec<Arc<BindGroupLayout>>,
        immediate_size: u32,
        is_error: bool,
    ) -> Self {
        Self {
            inner: Arc::new(PipelineLayoutInner {
                _bind_group_layouts: bind_group_layouts,
                _immediate_size: immediate_size,
                is_error,
            }),
        }
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub fn bind_group_layouts(&self) -> &[Arc<BindGroupLayout>] {
        &self.inner._bind_group_layouts
    }
}

fn validate_pipeline_layout_descriptor(
    bind_group_layouts: &[Arc<BindGroupLayout>],
    immediate_size: u32,
    limits: Limits,
) -> Option<String> {
    if bind_group_layouts.len() > limits.max_bind_groups as usize {
        return Some("pipeline layout bindGroupLayoutCount exceeds the device limit".to_owned());
    }
    if bind_group_layouts.iter().any(|layout| layout.is_error()) {
        return Some("pipeline layout cannot contain an error bind group layout".to_owned());
    }
    if bind_group_layouts.iter().any(|layout| layout.is_default()) {
        return Some("pipeline layout cannot contain a default bind group layout".to_owned());
    }
    if immediate_size > limits.max_immediate_size {
        return Some("pipeline layout immediateSize exceeds the device limit".to_owned());
    }

    None
}

#[derive(Debug, Clone)]
pub struct ComputePipelineDescriptor {
    pub layout: ComputePipelineLayout,
    pub shader_module: Arc<ShaderModule>,
    pub entry_point: Option<String>,
    pub constants: Vec<PipelineConstant>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ComputePipelineLayout {
    Auto,
    Explicit(Arc<PipelineLayout>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineConstant {
    pub key: String,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct ComputePipeline {
    inner: Arc<ComputePipelineInner>,
}

#[derive(Debug)]
struct ComputePipelineInner {
    _layout: ComputePipelineLayout,
    _shader_module: Arc<ShaderModule>,
    entry_name: String,
    _bindings: Vec<shader_naga::ReflectedResourceBinding>,
    metal_bindings: Vec<MetalBufferBinding>,
    hal: Option<HalComputePipeline>,
    bind_group_layouts: Vec<Arc<BindGroupLayout>>,
    is_error: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResolvedComputeWorkgroup {
    size: [u32; 3],
    storage_size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalBufferBinding {
    group: u32,
    binding: u32,
    metal_index: u32,
    ty: BufferBindingType,
}

impl ComputePipeline {
    fn new(
        descriptor: ComputePipelineDescriptor,
        is_error: bool,
        limits: Limits,
        hal_device: Option<&HalDevice>,
    ) -> (Self, Option<String>) {
        let resolved = if is_error {
            None
        } else {
            resolve_compute_pipeline_descriptor(&descriptor, limits).ok()
        };
        let (entry_name, bindings, workgroup, bind_group_layouts) = resolved.unwrap_or_else(|| {
            (
                descriptor.entry_point.clone().unwrap_or_default(),
                Vec::new(),
                None,
                Vec::new(),
            )
        });
        let metal_bindings = metal_buffer_binding_map(&bind_group_layouts);
        let (hal, backend_error) = if is_error {
            (None, None)
        } else {
            create_hal_compute_pipeline(
                hal_device,
                &descriptor.shader_module,
                &entry_name,
                workgroup,
                &metal_bindings,
            )
        };
        let is_error = is_error || backend_error.is_some();
        (
            Self {
                inner: Arc::new(ComputePipelineInner {
                    _layout: descriptor.layout,
                    _shader_module: descriptor.shader_module,
                    entry_name,
                    _bindings: bindings,
                    metal_bindings,
                    hal,
                    bind_group_layouts,
                    is_error,
                }),
            },
            backend_error,
        )
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub fn entry_name(&self) -> &str {
        &self.inner.entry_name
    }

    #[must_use]
    pub fn bind_group_layouts(&self) -> &[Arc<BindGroupLayout>] {
        &self.inner.bind_group_layouts
    }

    fn hal(&self) -> Option<HalComputePipeline> {
        self.inner.hal.clone()
    }

    fn metal_bindings(&self) -> &[MetalBufferBinding] {
        &self.inner.metal_bindings
    }
}

type ResolvedPipelineParts = (
    String,
    Vec<shader_naga::ReflectedResourceBinding>,
    Option<ResolvedComputeWorkgroup>,
    Vec<Arc<BindGroupLayout>>,
);

fn create_hal_compute_pipeline(
    hal_device: Option<&HalDevice>,
    shader_module: &ShaderModule,
    entry_name: &str,
    workgroup: Option<ResolvedComputeWorkgroup>,
    metal_bindings: &[MetalBufferBinding],
) -> (Option<HalComputePipeline>, Option<String>) {
    let Some(hal_device) = hal_device else {
        return (None, None);
    };
    let Some(module) = shader_module.validated_wgsl() else {
        return (
            None,
            Some("compute pipeline requires a valid WGSL shader module".to_owned()),
        );
    };
    let Some(workgroup) = workgroup else {
        return (
            None,
            Some("compute pipeline workgroup size reflection failed".to_owned()),
        );
    };
    let (shader, entry_point, descriptor_bindings) = match hal_device.backend() {
        HalBackend::Metal => {
            let msl_binding_map = shader_naga::MslBindingMap {
                buffers: metal_bindings
                    .iter()
                    .map(|binding| shader_naga::MslBufferBinding {
                        group: binding.group,
                        binding: binding.binding,
                        metal_index: binding.metal_index,
                    })
                    .collect(),
            };
            let generated = match module.generate_msl(entry_name, &msl_binding_map) {
                Ok(generated) => generated,
                Err(message) => return (None, Some(message)),
            };
            (
                HalShaderSource::Msl(generated.source),
                generated.entry_point,
                Vec::new(),
            )
        }
        HalBackend::Vulkan => {
            let spirv = match module.generate_spirv(entry_name, naga::ShaderStage::Compute) {
                Ok(spirv) => spirv,
                Err(message) => return (None, Some(message)),
            };
            (
                HalShaderSource::SpirV(spirv),
                entry_name.to_owned(),
                hal_descriptor_bindings(metal_bindings),
            )
        }
        HalBackend::Noop => return (None, None),
        _ => return (None, None),
    };
    match hal_device.create_compute_pipeline(
        shader,
        &entry_point,
        (workgroup.size[0], workgroup.size[1], workgroup.size[2]),
        &descriptor_bindings,
    ) {
        Ok(pipeline) => (Some(pipeline), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

fn hal_descriptor_bindings(bindings: &[MetalBufferBinding]) -> Vec<HalDescriptorBinding> {
    bindings
        .iter()
        .map(|binding| HalDescriptorBinding {
            group: binding.group,
            binding: binding.binding,
            kind: match binding.ty {
                BufferBindingType::Uniform => HalBufferBindingKind::Uniform,
                BufferBindingType::Storage | BufferBindingType::ReadOnlyStorage => {
                    HalBufferBindingKind::Storage
                }
            },
        })
        .collect()
}

fn metal_buffer_binding_map(layouts: &[Arc<BindGroupLayout>]) -> Vec<MetalBufferBinding> {
    let mut bindings = Vec::new();
    let mut metal_index = 0u32;
    for (group_index, layout) in layouts.iter().enumerate() {
        let Ok(group) = u32::try_from(group_index) else {
            break;
        };
        for entry in layout.entries() {
            if let Some(BindingLayoutKind::Buffer { ty, .. }) = entry.kind {
                bindings.push(MetalBufferBinding {
                    group,
                    binding: entry.binding,
                    metal_index,
                    ty,
                });
                metal_index = metal_index.saturating_add(1);
            }
        }
    }
    bindings.sort_by_key(|binding| (binding.group, binding.binding));
    for (index, binding) in bindings.iter_mut().enumerate() {
        binding.metal_index = u32::try_from(index).unwrap_or(u32::MAX);
    }
    bindings
}

fn validate_compute_pipeline_descriptor(
    descriptor: &ComputePipelineDescriptor,
    limits: Limits,
) -> Option<String> {
    resolve_compute_pipeline_descriptor(descriptor, limits).err()
}

fn resolve_compute_pipeline_descriptor(
    descriptor: &ComputePipelineDescriptor,
    limits: Limits,
) -> Result<ResolvedPipelineParts, String> {
    if descriptor.shader_module.is_error() {
        return Err("compute pipeline shader module must not be an error module".to_owned());
    }
    let Some(module) = descriptor.shader_module.validated_wgsl() else {
        return Err("compute pipeline requires a valid WGSL shader module".to_owned());
    };
    let entry_name = resolve_compute_entry(module, descriptor.entry_point.as_deref())?;
    let overrides = module.overrides();
    let constants = resolve_pipeline_constants(&overrides, &descriptor.constants)?;
    let workgroup = resolve_compute_workgroup(module, &entry_name, &overrides, &constants, limits)?;
    let bindings = module.resource_bindings_for_entry(&entry_name)?;
    validate_compute_pipeline_layout(&descriptor.layout, &bindings)?;
    let bind_group_layouts =
        effective_compute_bind_group_layouts(&descriptor.layout, &bindings, limits)?;
    Ok((entry_name, bindings, Some(workgroup), bind_group_layouts))
}

fn resolve_compute_entry(
    module: &shader_naga::ValidatedWgslModule,
    entry_point: Option<&str>,
) -> Result<String, String> {
    let entries = module.entry_points();
    let compute_entries = entries
        .iter()
        .filter(|entry| entry.stage == shader_naga::ReflectedShaderStage::Compute)
        .collect::<Vec<_>>();

    match entry_point {
        None => match compute_entries.as_slice() {
            [entry] => Ok(entry.name.clone()),
            [] => Err("compute pipeline shader module has no compute entry point".to_owned()),
            _ => Err(
                "compute pipeline entryPoint is required when multiple compute entries exist"
                    .to_owned(),
            ),
        },
        Some(name) => {
            if compute_entries.iter().any(|entry| entry.name == name) {
                Ok(name.to_owned())
            } else {
                Err("compute pipeline entryPoint must name a compute entry point".to_owned())
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ResolvedOverrideConstant {
    index: usize,
    value: f64,
}

fn resolve_pipeline_constants(
    overrides: &[shader_naga::ReflectedOverride],
    constants: &[PipelineConstant],
) -> Result<Vec<ResolvedOverrideConstant>, String> {
    let mut seen_keys = BTreeSet::new();
    let mut resolved = Vec::new();

    for constant in constants {
        if !seen_keys.insert(constant.key.as_str()) {
            return Err("pipeline constant keys must be unique".to_owned());
        }
        let index = resolve_pipeline_constant_key(overrides, &constant.key)?;
        validate_pipeline_constant_value(&overrides[index], constant.value)?;
        resolved.push(ResolvedOverrideConstant {
            index,
            value: constant.value,
        });
    }

    for (index, override_) in overrides.iter().enumerate() {
        if !override_.has_default && !resolved.iter().any(|constant| constant.index == index) {
            return Err("pipeline constant is required for override without a default".to_owned());
        }
    }

    Ok(resolved)
}

fn resolve_pipeline_constant_key(
    overrides: &[shader_naga::ReflectedOverride],
    key: &str,
) -> Result<usize, String> {
    if let Ok(id) = key.parse::<u16>() {
        return overrides
            .iter()
            .position(|override_| override_.id == Some(id))
            .ok_or_else(|| "pipeline constant key does not match a shader override".to_owned());
    }

    if overrides
        .iter()
        .any(|override_| override_.id.is_some() && override_.name.as_deref() == Some(key))
    {
        return Err("pipeline constant key must use numeric id for @id overrides".to_owned());
    }

    overrides
        .iter()
        .position(|override_| override_.id.is_none() && override_.name.as_deref() == Some(key))
        .ok_or_else(|| "pipeline constant key does not match a shader override".to_owned())
}

fn validate_pipeline_constant_value(
    override_: &shader_naga::ReflectedOverride,
    value: f64,
) -> Result<(), String> {
    if !value.is_finite() {
        return Err("pipeline constant value must be finite".to_owned());
    }
    if override_.ty.components != 1 {
        return Err("pipeline override constants must be scalar".to_owned());
    }

    match override_.ty.scalar {
        shader_naga::ReflectedTypeScalarClass::Float => {
            let max = if override_.ty.width == 2 {
                65_504.0
            } else {
                f64::from(f32::MAX)
            };
            if value.abs() > max {
                return Err("pipeline constant value is outside the override type range".to_owned());
            }
        }
        shader_naga::ReflectedTypeScalarClass::Sint => {
            if value.fract() != 0.0 || value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
                return Err("pipeline constant value is outside the override type range".to_owned());
            }
        }
        shader_naga::ReflectedTypeScalarClass::Uint => {
            if value.fract() != 0.0 || value < 0.0 || value > f64::from(u32::MAX) {
                return Err("pipeline constant value is outside the override type range".to_owned());
            }
        }
        shader_naga::ReflectedTypeScalarClass::Bool => {}
    }
    Ok(())
}

fn resolve_compute_workgroup(
    module: &shader_naga::ValidatedWgslModule,
    entry_name: &str,
    overrides: &[shader_naga::ReflectedOverride],
    constants: &[ResolvedOverrideConstant],
    limits: Limits,
) -> Result<ResolvedComputeWorkgroup, String> {
    let workgroup = module
        .compute_workgroup_size(entry_name)?
        .ok_or_else(|| "compute entry point workgroup size reflection failed".to_owned())?;
    let mut size = workgroup.literal_size;
    for (axis, key) in workgroup.override_keys.iter().enumerate() {
        if let Some(key) = key {
            let index = resolve_override_key(overrides, key)?;
            let value = constants
                .iter()
                .find(|constant| constant.index == index)
                .map(|constant| constant.value)
                .or_else(|| default_override_number(&overrides[index]))
                .ok_or_else(|| "workgroup size override has no value".to_owned())?;
            if value.fract() != 0.0 || value < 0.0 || value > f64::from(u32::MAX) {
                return Err("workgroup size override must resolve to a u32 value".to_owned());
            }
            size[axis] = value as u32;
        }
    }

    if size[0] > limits.max_compute_workgroup_size_x {
        return Err("compute workgroup x size exceeds the device limit".to_owned());
    }
    if size[1] > limits.max_compute_workgroup_size_y {
        return Err("compute workgroup y size exceeds the device limit".to_owned());
    }
    if size[2] > limits.max_compute_workgroup_size_z {
        return Err("compute workgroup z size exceeds the device limit".to_owned());
    }
    let invocations = size[0]
        .checked_mul(size[1])
        .and_then(|xy| xy.checked_mul(size[2]))
        .ok_or_else(|| "compute workgroup invocation count overflows".to_owned())?;
    if invocations > limits.max_compute_invocations_per_workgroup {
        return Err("compute workgroup invocation count exceeds the device limit".to_owned());
    }
    if workgroup.workgroup_storage_size > u64::from(limits.max_compute_workgroup_storage_size) {
        return Err("compute workgroup storage size exceeds the device limit".to_owned());
    }

    Ok(ResolvedComputeWorkgroup {
        size,
        storage_size: workgroup.workgroup_storage_size,
    })
}

fn resolve_override_key(
    overrides: &[shader_naga::ReflectedOverride],
    key: &shader_naga::ReflectedOverrideKey,
) -> Result<usize, String> {
    overrides
        .iter()
        .position(|override_| {
            if let Some(id) = key.id {
                override_.id == Some(id)
            } else {
                override_.name == key.name
            }
        })
        .ok_or_else(|| "workgroup size override key does not match a shader override".to_owned())
}

fn default_override_number(override_: &shader_naga::ReflectedOverride) -> Option<f64> {
    match override_.default_value {
        Some(shader_naga::ReflectedOverrideValue::Number(value)) => Some(value),
        Some(shader_naga::ReflectedOverrideValue::Bool(value)) => Some(f64::from(value as u8)),
        None => None,
    }
}

fn validate_compute_pipeline_layout(
    layout: &ComputePipelineLayout,
    bindings: &[shader_naga::ReflectedResourceBinding],
) -> Result<(), String> {
    let ComputePipelineLayout::Explicit(layout) = layout else {
        return Ok(());
    };
    if layout.is_error() {
        return Err("compute pipeline layout must not be an error pipeline layout".to_owned());
    }
    let requirements = bindings
        .iter()
        .cloned()
        .map(|binding| StageResourceBinding {
            stage: PipelineShaderStage::Compute,
            binding,
        })
        .collect::<Vec<_>>();
    validate_pipeline_layout_stage_bindings(layout, &requirements)
}

fn effective_compute_bind_group_layouts(
    layout: &ComputePipelineLayout,
    bindings: &[shader_naga::ReflectedResourceBinding],
    limits: Limits,
) -> Result<Vec<Arc<BindGroupLayout>>, String> {
    match layout {
        ComputePipelineLayout::Explicit(layout) => Ok(layout.bind_group_layouts().to_vec()),
        ComputePipelineLayout::Auto => derive_bind_group_layouts(
            bindings
                .iter()
                .cloned()
                .map(|binding| StageResourceBinding {
                    stage: PipelineShaderStage::Compute,
                    binding,
                }),
            limits,
        ),
    }
}

#[derive(Debug, Clone)]
struct StageResourceBinding {
    stage: PipelineShaderStage,
    binding: shader_naga::ReflectedResourceBinding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipelineShaderStage {
    Vertex,
    Fragment,
    Compute,
}

fn validate_pipeline_layout_stage_bindings(
    layout: &PipelineLayout,
    requirements: &[StageResourceBinding],
) -> Result<(), String> {
    for requirement in requirements {
        let binding = &requirement.binding;
        if !binding.statically_used {
            continue;
        }
        let group = usize::try_from(binding.group)
            .map_err(|_| "shader binding group index is too large".to_owned())?;
        let Some(group_layout) = layout.bind_group_layouts().get(group) else {
            return Err("pipeline layout is missing a shader bind group".to_owned());
        };
        let Some(layout_entry) = group_layout
            .entries()
            .iter()
            .find(|entry| entry.binding == binding.binding)
        else {
            return Err("pipeline layout is missing a shader binding".to_owned());
        };
        if layout_entry.visibility & pipeline_stage_visibility_bit(requirement.stage) == 0 {
            return Err(
                "pipeline layout binding visibility does not include the shader stage".to_owned(),
            );
        }
        let Some(kind) = layout_entry.kind else {
            return Err("pipeline layout binding must be valid".to_owned());
        };
        validate_shader_binding_compat(binding, kind)?;
    }

    Ok(())
}

fn derive_bind_group_layouts<I>(
    requirements: I,
    limits: Limits,
) -> Result<Vec<Arc<BindGroupLayout>>, String>
where
    I: IntoIterator<Item = StageResourceBinding>,
{
    let mut groups = BTreeMap::<u32, BTreeMap<u32, BindGroupLayoutEntry>>::new();
    for requirement in requirements {
        let binding = requirement.binding;
        if !binding.statically_used {
            continue;
        }
        let group = groups.entry(binding.group).or_default();
        let visibility = pipeline_stage_visibility_bit(requirement.stage);
        let derived = reflected_bind_group_layout_entry(&binding, visibility)?;
        match group.get_mut(&binding.binding) {
            Some(existing) => merge_bind_group_layout_entry(existing, derived)?,
            None => {
                group.insert(binding.binding, derived);
            }
        }
    }

    let Some(max_group) = groups.keys().next_back().copied() else {
        return Ok(Vec::new());
    };
    let group_count = usize::try_from(max_group)
        .ok()
        .and_then(|group| group.checked_add(1))
        .ok_or_else(|| "pipeline bind group index is too large".to_owned())?;
    if group_count > limits.max_bind_groups as usize {
        return Err("pipeline auto layout bind group count exceeds the device limit".to_owned());
    }

    let mut layouts = Vec::with_capacity(group_count);
    for group_index in 0..=max_group {
        let entries = groups
            .remove(&group_index)
            .map(|entries| entries.into_values().collect::<Vec<_>>())
            .unwrap_or_default();
        if let Some(message) = validate_bind_group_layout_descriptor(&entries, limits) {
            return Err(message);
        }
        layouts.push(Arc::new(BindGroupLayout::new(entries, false, true)));
    }
    Ok(layouts)
}

fn reflected_bind_group_layout_entry(
    binding: &shader_naga::ReflectedResourceBinding,
    visibility: u64,
) -> Result<BindGroupLayoutEntry, String> {
    Ok(BindGroupLayoutEntry {
        binding: binding.binding,
        visibility,
        binding_array_size: 0,
        kind: Some(reflected_binding_layout_kind(binding)?),
    })
}

fn reflected_binding_layout_kind(
    binding: &shader_naga::ReflectedResourceBinding,
) -> Result<BindingLayoutKind, String> {
    match &binding.kind {
        shader_naga::ReflectedResourceBindingKind::Buffer(ty) => Ok(BindingLayoutKind::Buffer {
            ty: match ty {
                shader_naga::ReflectedBufferType::Uniform => BufferBindingType::Uniform,
                shader_naga::ReflectedBufferType::Storage => BufferBindingType::Storage,
                shader_naga::ReflectedBufferType::ReadOnlyStorage => {
                    BufferBindingType::ReadOnlyStorage
                }
            },
            has_dynamic_offset: false,
            min_binding_size: binding.min_binding_size,
        }),
        shader_naga::ReflectedResourceBindingKind::Sampler { comparison } => {
            Ok(BindingLayoutKind::Sampler {
                ty: if *comparison {
                    SamplerBindingType::Comparison
                } else {
                    SamplerBindingType::Filtering
                },
            })
        }
        shader_naga::ReflectedResourceBindingKind::Texture {
            sampled,
            sample_kind,
            sample_usage,
            view_dimension,
            multisampled,
        } => Ok(BindingLayoutKind::Texture {
            sample_type: reflected_texture_sample_type(*sampled, *sample_kind, *sample_usage)?,
            view_dimension: reflected_texture_view_dimension(*view_dimension),
            multisampled: *multisampled,
        }),
        shader_naga::ReflectedResourceBindingKind::StorageTexture {
            format,
            access,
            view_dimension,
        } => Ok(BindingLayoutKind::StorageTexture {
            access: reflected_storage_texture_access(access),
            format: reflected_storage_texture_format(format)?,
            view_dimension: reflected_texture_view_dimension(*view_dimension),
        }),
    }
}

fn reflected_texture_sample_type(
    sampled: bool,
    sample_kind: Option<shader_naga::ReflectedTypeScalarClass>,
    sample_usage: shader_naga::ReflectedTextureSampleUsage,
) -> Result<TextureSampleType, String> {
    if !sampled {
        return Ok(TextureSampleType::Depth);
    }
    match sample_kind {
        Some(shader_naga::ReflectedTypeScalarClass::Float) => Ok(match sample_usage {
            shader_naga::ReflectedTextureSampleUsage::Sample => TextureSampleType::Float,
            shader_naga::ReflectedTextureSampleUsage::Load => TextureSampleType::UnfilterableFloat,
        }),
        Some(shader_naga::ReflectedTypeScalarClass::Sint) => Ok(TextureSampleType::Sint),
        Some(shader_naga::ReflectedTypeScalarClass::Uint) => Ok(TextureSampleType::Uint),
        _ => Err("pipeline texture binding sample type is unsupported".to_owned()),
    }
}

fn reflected_texture_view_dimension(
    dimension: shader_naga::ReflectedTextureViewDimension,
) -> TextureViewDimension {
    match dimension {
        shader_naga::ReflectedTextureViewDimension::D1 => TextureViewDimension::D1,
        shader_naga::ReflectedTextureViewDimension::D2 => TextureViewDimension::D2,
        shader_naga::ReflectedTextureViewDimension::D2Array => TextureViewDimension::D2Array,
        shader_naga::ReflectedTextureViewDimension::Cube => TextureViewDimension::Cube,
        shader_naga::ReflectedTextureViewDimension::CubeArray => TextureViewDimension::CubeArray,
        shader_naga::ReflectedTextureViewDimension::D3 => TextureViewDimension::D3,
    }
}

fn reflected_storage_texture_access(
    access: &shader_naga::ReflectedStorageTextureAccess,
) -> StorageTextureAccess {
    match (access.read, access.write) {
        (true, true) => StorageTextureAccess::ReadWrite,
        (true, false) => StorageTextureAccess::ReadOnly,
        _ => StorageTextureAccess::WriteOnly,
    }
}

fn reflected_storage_texture_format(format: &str) -> Result<TextureFormat, String> {
    let raw = match format {
        "Rgba8Unorm" => 0x0000_0016,
        "Rgba8Snorm" => 0x0000_0018,
        "Rgba8Uint" => 0x0000_0019,
        "Rgba8Sint" => 0x0000_001A,
        "Rgba16Uint" => 0x0000_0026,
        "Rgba16Sint" => 0x0000_0027,
        "Rgba16Float" => 0x0000_0028,
        "R32Uint" => 0x0000_000F,
        "R32Sint" => 0x0000_0010,
        "R32Float" => 0x0000_000E,
        "Rg32Uint" => 0x0000_0022,
        "Rg32Sint" => 0x0000_0023,
        "Rg32Float" => 0x0000_0021,
        "Rgba32Uint" => 0x0000_002A,
        "Rgba32Sint" => 0x0000_002B,
        "Rgba32Float" => 0x0000_0029,
        _ => return Err("pipeline auto layout storage texture format is unsupported".to_owned()),
    };
    Ok(TextureFormat::from_raw(raw))
}

fn merge_bind_group_layout_entry(
    existing: &mut BindGroupLayoutEntry,
    incoming: BindGroupLayoutEntry,
) -> Result<(), String> {
    existing.visibility |= incoming.visibility;
    match (&mut existing.kind, incoming.kind) {
        (
            Some(BindingLayoutKind::Buffer {
                ty,
                min_binding_size,
                ..
            }),
            Some(BindingLayoutKind::Buffer {
                ty: incoming_ty,
                min_binding_size: incoming_min_binding_size,
                ..
            }),
        ) if *ty == incoming_ty => {
            *min_binding_size = (*min_binding_size).max(incoming_min_binding_size);
            Ok(())
        }
        (
            Some(BindingLayoutKind::Texture { sample_type, .. }),
            Some(BindingLayoutKind::Texture {
                sample_type: incoming_sample_type,
                ..
            }),
        ) if *sample_type == incoming_sample_type
            || matches!(
                (*sample_type, incoming_sample_type),
                (
                    TextureSampleType::Float,
                    TextureSampleType::UnfilterableFloat
                ) | (
                    TextureSampleType::UnfilterableFloat,
                    TextureSampleType::Float
                )
            ) =>
        {
            if incoming_sample_type == TextureSampleType::Float {
                *sample_type = TextureSampleType::Float;
            }
            Ok(())
        }
        (Some(existing_kind), Some(incoming_kind)) if *existing_kind == incoming_kind => Ok(()),
        _ => Err("pipeline auto layout has incompatible shader bindings".to_owned()),
    }
}

fn pipeline_stage_visibility_bit(stage: PipelineShaderStage) -> u64 {
    match stage {
        PipelineShaderStage::Vertex => 1,
        PipelineShaderStage::Fragment => 2,
        PipelineShaderStage::Compute => 4,
    }
}

fn validate_shader_binding_compat(
    binding: &shader_naga::ReflectedResourceBinding,
    layout_kind: BindingLayoutKind,
) -> Result<(), String> {
    match (&binding.kind, layout_kind) {
        (
            shader_naga::ReflectedResourceBindingKind::Buffer(shader_ty),
            BindingLayoutKind::Buffer {
                ty,
                min_binding_size,
                ..
            },
        ) => {
            if !buffer_binding_types_compatible(*shader_ty, ty) {
                return Err(
                    "compute pipeline layout buffer binding type is incompatible".to_owned(),
                );
            }
            if min_binding_size < binding.min_binding_size {
                return Err("compute pipeline layout buffer minBindingSize is too small".to_owned());
            }
            Ok(())
        }
        (
            shader_naga::ReflectedResourceBindingKind::Sampler { .. },
            BindingLayoutKind::Sampler { .. },
        )
        | (
            shader_naga::ReflectedResourceBindingKind::Texture { .. },
            BindingLayoutKind::Texture { .. },
        )
        | (
            shader_naga::ReflectedResourceBindingKind::StorageTexture { .. },
            BindingLayoutKind::StorageTexture { .. },
        ) => {
            let expected = reflected_binding_layout_kind(binding)?;
            if shader_binding_layout_kinds_compatible(expected, layout_kind) {
                Ok(())
            } else {
                Err(
                    "pipeline layout binding kind is incompatible with the shader binding"
                        .to_owned(),
                )
            }
        }
        _ => Err("compute pipeline layout binding type is incompatible".to_owned()),
    }
}

fn shader_binding_layout_kinds_compatible(
    expected: BindingLayoutKind,
    actual: BindingLayoutKind,
) -> bool {
    match (expected, actual) {
        (
            BindingLayoutKind::Sampler { ty: expected },
            BindingLayoutKind::Sampler { ty: actual },
        ) => expected == actual,
        (
            BindingLayoutKind::Texture {
                sample_type,
                view_dimension,
                multisampled,
            },
            BindingLayoutKind::Texture {
                sample_type: actual_sample_type,
                view_dimension: actual_view_dimension,
                multisampled: actual_multisampled,
            },
        ) => {
            sample_type == actual_sample_type
                && view_dimension == actual_view_dimension
                && multisampled == actual_multisampled
        }
        (
            BindingLayoutKind::StorageTexture {
                access,
                format,
                view_dimension,
            },
            BindingLayoutKind::StorageTexture {
                access: actual_access,
                format: actual_format,
                view_dimension: actual_view_dimension,
            },
        ) => {
            access == actual_access
                && format == actual_format
                && view_dimension == actual_view_dimension
        }
        _ => false,
    }
}

fn buffer_binding_types_compatible(
    shader_ty: shader_naga::ReflectedBufferType,
    layout_ty: BufferBindingType,
) -> bool {
    matches!(
        (shader_ty, layout_ty),
        (
            shader_naga::ReflectedBufferType::Uniform,
            BufferBindingType::Uniform
        ) | (
            shader_naga::ReflectedBufferType::Storage,
            BufferBindingType::Storage
        ) | (
            shader_naga::ReflectedBufferType::ReadOnlyStorage,
            BufferBindingType::ReadOnlyStorage
        )
    )
}

#[derive(Debug, Clone)]
pub struct RenderPipelineDescriptor {
    pub layout: RenderPipelineLayout,
    pub vertex: RenderPipelineVertexState,
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub multisample: MultisampleState,
    pub fragment: Option<RenderPipelineFragmentState>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RenderPipelineLayout {
    Auto,
    Explicit(Arc<PipelineLayout>),
}

#[derive(Debug, Clone)]
pub struct RenderPipelineVertexState {
    pub shader: RenderPipelineShaderStage,
    pub buffer_count: usize,
    pub buffers: Vec<VertexBufferLayout>,
}

#[derive(Debug, Clone)]
pub struct VertexBufferLayout {
    pub array_stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: Vec<VertexAttribute>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexStepMode {
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u64,
    pub shader_location: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexFormat(u32);

impl VertexFormat {
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    fn info(self) -> VertexFormatInfo {
        match self.0 {
            0x0000_0001 => VertexFormatInfo::new(1, FormatOutputClass::Uint),
            0x0000_0002 => VertexFormatInfo::new(2, FormatOutputClass::Uint),
            0x0000_0003 => VertexFormatInfo::new(4, FormatOutputClass::Uint),
            0x0000_0004 => VertexFormatInfo::new(1, FormatOutputClass::Sint),
            0x0000_0005 => VertexFormatInfo::new(2, FormatOutputClass::Sint),
            0x0000_0006 => VertexFormatInfo::new(4, FormatOutputClass::Sint),
            0x0000_0007 | 0x0000_000A => VertexFormatInfo::new(1, FormatOutputClass::Float),
            0x0000_0008 | 0x0000_000B => VertexFormatInfo::new(2, FormatOutputClass::Float),
            0x0000_0009 | 0x0000_000C => VertexFormatInfo::new(4, FormatOutputClass::Float),
            0x0000_000D => VertexFormatInfo::new(2, FormatOutputClass::Uint),
            0x0000_000E => VertexFormatInfo::new(4, FormatOutputClass::Uint),
            0x0000_000F => VertexFormatInfo::new(8, FormatOutputClass::Uint),
            0x0000_0010 => VertexFormatInfo::new(2, FormatOutputClass::Sint),
            0x0000_0011 => VertexFormatInfo::new(4, FormatOutputClass::Sint),
            0x0000_0012 => VertexFormatInfo::new(8, FormatOutputClass::Sint),
            0x0000_0013 | 0x0000_0016 | 0x0000_0019 => {
                VertexFormatInfo::new(2, FormatOutputClass::Float)
            }
            0x0000_0014 | 0x0000_0017 | 0x0000_001A => {
                VertexFormatInfo::new(4, FormatOutputClass::Float)
            }
            0x0000_0015 | 0x0000_0018 | 0x0000_001B => {
                VertexFormatInfo::new(8, FormatOutputClass::Float)
            }
            0x0000_001C => VertexFormatInfo::new(4, FormatOutputClass::Float),
            0x0000_001D => VertexFormatInfo::new(8, FormatOutputClass::Float),
            0x0000_001E => VertexFormatInfo::new(12, FormatOutputClass::Float),
            0x0000_001F => VertexFormatInfo::new(16, FormatOutputClass::Float),
            0x0000_0020 => VertexFormatInfo::new(4, FormatOutputClass::Uint),
            0x0000_0021 => VertexFormatInfo::new(8, FormatOutputClass::Uint),
            0x0000_0022 => VertexFormatInfo::new(12, FormatOutputClass::Uint),
            0x0000_0023 => VertexFormatInfo::new(16, FormatOutputClass::Uint),
            0x0000_0024 => VertexFormatInfo::new(4, FormatOutputClass::Sint),
            0x0000_0025 => VertexFormatInfo::new(8, FormatOutputClass::Sint),
            0x0000_0026 => VertexFormatInfo::new(12, FormatOutputClass::Sint),
            0x0000_0027 => VertexFormatInfo::new(16, FormatOutputClass::Sint),
            0x0000_0028 | 0x0000_0029 => VertexFormatInfo::new(4, FormatOutputClass::Float),
            // Keep unknown future values conservative instead of guessing a smaller footprint.
            _ => VertexFormatInfo::new(16, FormatOutputClass::Float),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct VertexFormatInfo {
    byte_size: u64,
    output_class: FormatOutputClass,
}

impl VertexFormatInfo {
    const fn new(byte_size: u64, output_class: FormatOutputClass) -> Self {
        Self {
            byte_size,
            output_class,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderPipelineFragmentState {
    pub shader: RenderPipelineShaderStage,
    pub target_count: usize,
    pub targets: Vec<ColorTargetState>,
}

#[derive(Debug, Clone)]
pub struct RenderPipelineShaderStage {
    pub module: Arc<ShaderModule>,
    pub entry_point: Option<String>,
    pub constants: Vec<PipelineConstant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorTargetState {
    pub format: TextureFormat,
    pub blend: bool,
    pub write_mask: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub strip_index_format: Option<IndexFormat>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth_write_enabled: Option<bool>,
    pub depth_compare: Option<CompareFunction>,
    pub stencil_front: StencilFaceState,
    pub stencil_back: StencilFaceState,
    pub stencil_read_mask: u32,
    pub stencil_write_mask: u32,
    pub depth_bias: i32,
    pub depth_bias_slope_scale: f32,
    pub depth_bias_clamp: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilFaceState {
    pub compare: CompareFunction,
    pub fail_op: StencilOperation,
    pub depth_fail_op: StencilOperation,
    pub pass_op: StencilOperation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StencilOperation {
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultisampleState {
    pub count: u32,
    pub mask: u32,
    pub alpha_to_coverage_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct RenderPipeline {
    inner: Arc<RenderPipelineInner>,
}

#[derive(Debug)]
struct RenderPipelineInner {
    _layout: RenderPipelineLayout,
    _vertex: RenderPipelineVertexState,
    _primitive: PrimitiveState,
    _depth_stencil: Option<DepthStencilState>,
    _multisample: MultisampleState,
    _fragment: Option<RenderPipelineFragmentState>,
    vertex_entry_name: String,
    fragment_entry_name: Option<String>,
    metal_bindings: Vec<MetalBufferBinding>,
    vertex_buffer_bindings: Vec<MetalVertexBufferBinding>,
    hal: Option<HalRenderPipeline>,
    bind_group_layouts: Vec<Arc<BindGroupLayout>>,
    is_error: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalVertexBufferBinding {
    slot: u32,
    metal_index: u32,
}

impl RenderPipeline {
    fn new(
        descriptor: RenderPipelineDescriptor,
        is_error: bool,
        limits: Limits,
        hal_device: Option<&HalDevice>,
    ) -> (Self, Option<String>) {
        let resolved = if is_error {
            None
        } else {
            resolve_render_pipeline_descriptor(&descriptor, limits).ok()
        };
        let (vertex_entry_name, fragment_entry_name, bind_group_layouts) =
            resolved.unwrap_or_else(|| {
                (
                    descriptor
                        .vertex
                        .shader
                        .entry_point
                        .clone()
                        .unwrap_or_default(),
                    descriptor
                        .fragment
                        .as_ref()
                        .and_then(|fragment| fragment.shader.entry_point.clone()),
                    Vec::new(),
                )
            });
        let metal_bindings = metal_buffer_binding_map(&bind_group_layouts);
        let vertex_buffer_bindings =
            metal_vertex_buffer_binding_map(descriptor.vertex.buffer_count, &metal_bindings);
        let (hal, backend_error) = if is_error {
            (None, None)
        } else {
            create_hal_render_pipeline(
                hal_device,
                &descriptor,
                &vertex_entry_name,
                fragment_entry_name.as_deref(),
                &metal_bindings,
                &vertex_buffer_bindings,
            )
        };
        let is_error = is_error || backend_error.is_some();
        (
            Self {
                inner: Arc::new(RenderPipelineInner {
                    _layout: descriptor.layout,
                    _vertex: descriptor.vertex,
                    _primitive: descriptor.primitive,
                    _depth_stencil: descriptor.depth_stencil,
                    _multisample: descriptor.multisample,
                    _fragment: descriptor.fragment,
                    vertex_entry_name,
                    fragment_entry_name,
                    metal_bindings,
                    vertex_buffer_bindings,
                    hal,
                    bind_group_layouts,
                    is_error,
                }),
            },
            backend_error,
        )
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub fn vertex_entry_name(&self) -> &str {
        &self.inner.vertex_entry_name
    }

    #[must_use]
    pub fn fragment_entry_name(&self) -> Option<&str> {
        self.inner.fragment_entry_name.as_deref()
    }

    #[must_use]
    pub fn bind_group_layouts(&self) -> &[Arc<BindGroupLayout>] {
        &self.inner.bind_group_layouts
    }

    fn hal(&self) -> Option<HalRenderPipeline> {
        self.inner.hal.clone()
    }

    fn metal_bindings(&self) -> &[MetalBufferBinding] {
        &self.inner.metal_bindings
    }

    fn vertex_buffer_bindings(&self) -> &[MetalVertexBufferBinding] {
        &self.inner.vertex_buffer_bindings
    }

    #[must_use]
    pub(crate) fn required_vertex_buffer_count(&self) -> usize {
        self.inner._vertex.buffer_count
    }

    #[must_use]
    fn vertex_buffer_layouts(&self) -> &[VertexBufferLayout] {
        &self.inner._vertex.buffers
    }

    #[must_use]
    fn primitive_state(&self) -> PrimitiveState {
        self.inner._primitive
    }

    #[must_use]
    fn attachment_signature(&self) -> AttachmentSignature {
        AttachmentSignature {
            color_formats: self
                .inner
                ._fragment
                .as_ref()
                .map(|fragment| {
                    fragment
                        .targets
                        .iter()
                        .map(|target| (!target.format.is_undefined()).then_some(target.format))
                        .collect()
                })
                .unwrap_or_default(),
            depth_stencil_format: self.inner._depth_stencil.map(|depth| depth.format),
            sample_count: self.inner._multisample.count,
        }
    }
}

fn validate_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    limits: Limits,
) -> Option<String> {
    resolve_render_pipeline_descriptor(descriptor, limits).err()
}

type ResolvedRenderPipelineParts = (String, Option<String>, Vec<Arc<BindGroupLayout>>);

fn create_hal_render_pipeline(
    hal_device: Option<&HalDevice>,
    descriptor: &RenderPipelineDescriptor,
    vertex_entry_name: &str,
    fragment_entry_name: Option<&str>,
    metal_bindings: &[MetalBufferBinding],
    vertex_buffer_bindings: &[MetalVertexBufferBinding],
) -> (Option<HalRenderPipeline>, Option<String>) {
    let Some(hal_device) = hal_device else {
        return (None, None);
    };
    if matches!(hal_device.backend(), HalBackend::Noop) {
        return (None, None);
    }
    if descriptor.depth_stencil.is_some()
        || descriptor.multisample.count != 1
        || descriptor
            .fragment
            .as_ref()
            .map_or(0, |fragment| fragment.target_count)
            != 1
    {
        return (
            None,
            Some(
                "real render pipeline currently supports one single-sampled color target only"
                    .to_owned(),
            ),
        );
    }
    let Some(fragment) = &descriptor.fragment else {
        return (
            None,
            Some("Metal render pipeline requires a fragment stage".to_owned()),
        );
    };
    let Some(fragment_entry_name) = fragment_entry_name else {
        return (
            None,
            Some("real render pipeline requires a fragment entry point".to_owned()),
        );
    };
    let (shader, vertex_entry_point, fragment_entry_point, descriptor_bindings) = match hal_device
        .backend()
    {
        HalBackend::Metal => {
            if !Arc::ptr_eq(&descriptor.vertex.shader.module, &fragment.shader.module) {
                return (
                    None,
                    Some(
                        "Metal render pipeline requires vertex and fragment entries in the same WGSL module"
                            .to_owned(),
                    ),
                );
            }
            let Some(module) = descriptor.vertex.shader.module.validated_wgsl() else {
                return (
                    None,
                    Some("render pipeline requires a valid WGSL shader module".to_owned()),
                );
            };
            let msl_binding_map = shader_naga::MslBindingMap {
                buffers: metal_bindings
                    .iter()
                    .map(|binding| shader_naga::MslBufferBinding {
                        group: binding.group,
                        binding: binding.binding,
                        metal_index: binding.metal_index,
                    })
                    .collect(),
            };
            let msl_vertex_buffers = match msl_vertex_buffer_bindings(
                &descriptor.vertex.buffers,
                vertex_buffer_bindings,
            ) {
                Ok(bindings) => bindings,
                Err(message) => return (None, Some(message)),
            };
            let generated = match module.generate_render_msl(
                vertex_entry_name,
                fragment_entry_name,
                &msl_binding_map,
                &msl_vertex_buffers,
            ) {
                Ok(generated) => generated,
                Err(message) => return (None, Some(message)),
            };
            (
                HalShaderSource::Msl(generated.source),
                generated.vertex_entry_point,
                generated.fragment_entry_point,
                Vec::new(),
            )
        }
        HalBackend::Vulkan => {
            let Some(vertex_module) = descriptor.vertex.shader.module.validated_wgsl() else {
                return (
                    None,
                    Some("render pipeline requires a valid WGSL vertex shader module".to_owned()),
                );
            };
            let Some(fragment_module) = fragment.shader.module.validated_wgsl() else {
                return (
                    None,
                    Some("render pipeline requires a valid WGSL fragment shader module".to_owned()),
                );
            };
            let vertex =
                match vertex_module.generate_spirv(vertex_entry_name, naga::ShaderStage::Vertex) {
                    Ok(spirv) => spirv,
                    Err(message) => return (None, Some(message)),
                };
            let fragment = match fragment_module
                .generate_spirv(fragment_entry_name, naga::ShaderStage::Fragment)
            {
                Ok(spirv) => spirv,
                Err(message) => return (None, Some(message)),
            };
            (
                HalShaderSource::SpirVStages { vertex, fragment },
                vertex_entry_name.to_owned(),
                fragment_entry_name.to_owned(),
                hal_descriptor_bindings(metal_bindings),
            )
        }
        _ => return (None, None),
    };
    let hal_descriptor = match hal_render_pipeline_descriptor(descriptor, vertex_buffer_bindings) {
        Ok(descriptor) => descriptor,
        Err(message) => return (None, Some(message)),
    };
    match hal_device.create_render_pipeline(
        shader,
        &vertex_entry_point,
        &fragment_entry_point,
        &hal_descriptor,
        &descriptor_bindings,
    ) {
        Ok(pipeline) => (Some(pipeline), None),
        Err(error) => (None, Some(error.to_string())),
    }
}

fn metal_vertex_buffer_binding_map(
    vertex_buffer_count: usize,
    metal_bindings: &[MetalBufferBinding],
) -> Vec<MetalVertexBufferBinding> {
    let start = metal_bindings.len();
    (0..vertex_buffer_count)
        .filter_map(|slot| {
            Some(MetalVertexBufferBinding {
                slot: u32::try_from(slot).ok()?,
                metal_index: u32::try_from(start.checked_add(slot)?).ok()?,
            })
        })
        .collect()
}

fn msl_vertex_buffer_bindings(
    layouts: &[VertexBufferLayout],
    bindings: &[MetalVertexBufferBinding],
) -> Result<Vec<shader_naga::MslVertexBufferBinding>, String> {
    layouts
        .iter()
        .zip(bindings)
        .map(|(layout, binding)| {
            Ok(shader_naga::MslVertexBufferBinding {
                slot: binding.slot,
                metal_index: binding.metal_index,
                array_stride: layout.array_stride,
                step_mode: match layout.step_mode {
                    VertexStepMode::Vertex => shader_naga::MslVertexStepMode::Vertex,
                    VertexStepMode::Instance => shader_naga::MslVertexStepMode::Instance,
                },
                attributes: layout
                    .attributes
                    .iter()
                    .map(|attribute| {
                        Ok(shader_naga::MslVertexAttribute {
                            shader_location: attribute.shader_location,
                            offset: attribute.offset,
                            format: msl_vertex_format(attribute.format)?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            })
        })
        .collect()
}

fn hal_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    bindings: &[MetalVertexBufferBinding],
) -> Result<HalRenderPipelineDescriptor, String> {
    let color_formats = descriptor
        .fragment
        .as_ref()
        .map(|fragment| {
            fragment
                .targets
                .iter()
                .map(|target| hal_texture_format(target.format))
                .collect()
        })
        .unwrap_or_default();
    let vertex_buffers = descriptor
        .vertex
        .buffers
        .iter()
        .zip(bindings)
        .map(|(layout, binding)| {
            Ok(HalVertexBufferLayout {
                array_stride: layout.array_stride,
                step_mode: match layout.step_mode {
                    VertexStepMode::Vertex => HalVertexStepMode::Vertex,
                    VertexStepMode::Instance => HalVertexStepMode::Instance,
                },
                attributes: layout
                    .attributes
                    .iter()
                    .map(|attribute| {
                        Ok(HalVertexAttribute {
                            format: hal_vertex_format(attribute.format),
                            offset: attribute.offset,
                            shader_location: attribute.shader_location,
                            metal_buffer_index: binding.metal_index,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(HalRenderPipelineDescriptor {
        color_formats,
        vertex_buffers,
        primitive_topology: hal_primitive_topology(descriptor.primitive.topology),
    })
}

fn msl_vertex_format(format: VertexFormat) -> Result<shader_naga::MslVertexFormat, String> {
    match format.0 {
        0x0000_001C => Ok(shader_naga::MslVertexFormat::Float32),
        0x0000_001D => Ok(shader_naga::MslVertexFormat::Float32x2),
        0x0000_001E => Ok(shader_naga::MslVertexFormat::Float32x3),
        0x0000_001F => Ok(shader_naga::MslVertexFormat::Float32x4),
        _ => Err("Metal render pipeline currently supports Float32 vertex formats only".to_owned()),
    }
}

fn hal_vertex_format(format: VertexFormat) -> HalVertexFormat {
    match format.0 {
        0x0000_001C => HalVertexFormat::Float32,
        0x0000_001D => HalVertexFormat::Float32x2,
        0x0000_001E => HalVertexFormat::Float32x3,
        0x0000_001F => HalVertexFormat::Float32x4,
        _ => HalVertexFormat::Unsupported,
    }
}

fn hal_primitive_topology(topology: PrimitiveTopology) -> HalPrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => HalPrimitiveTopology::PointList,
        PrimitiveTopology::LineList => HalPrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip => HalPrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList => HalPrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip => HalPrimitiveTopology::TriangleStrip,
    }
}

fn resolve_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    limits: Limits,
) -> Result<ResolvedRenderPipelineParts, String> {
    if let RenderPipelineLayout::Explicit(layout) = &descriptor.layout {
        if layout.is_error() {
            return Err("render pipeline layout must not be an error pipeline layout".to_owned());
        }
    }

    let vertex_entry = resolve_render_entry(
        &descriptor.vertex.shader,
        shader_naga::ReflectedShaderStage::Vertex,
        "vertex",
    )?;
    let fragment_entry = if let Some(fragment) = &descriptor.fragment {
        Some(resolve_render_entry(
            &fragment.shader,
            shader_naga::ReflectedShaderStage::Fragment,
            "fragment",
        )?)
    } else {
        None
    };

    validate_render_constants(&descriptor.vertex.shader)?;
    if let Some(fragment) = &descriptor.fragment {
        validate_render_constants(&fragment.shader)?;
    }
    validate_vertex_state(&descriptor.vertex, &vertex_entry, limits)?;
    validate_render_presence(descriptor)?;
    validate_primitive_state(descriptor.primitive)?;
    if let Some(depth_stencil) = descriptor.depth_stencil {
        validate_depth_bias_state(descriptor.primitive.topology, depth_stencil)?;
        validate_depth_stencil_aspects(depth_stencil)?;
    }
    validate_fragment_depth_output(descriptor, fragment_entry.as_deref())?;
    validate_color_targets(descriptor, fragment_entry.as_deref(), limits)?;
    validate_render_pipeline_layout(descriptor, &vertex_entry, fragment_entry.as_deref())?;
    validate_multisample_state(descriptor, fragment_entry.as_deref())?;
    let bind_group_layouts = effective_render_bind_group_layouts(
        descriptor,
        &vertex_entry,
        fragment_entry.as_deref(),
        limits,
    )?;

    Ok((vertex_entry, fragment_entry, bind_group_layouts))
}

fn validate_vertex_state(
    vertex: &RenderPipelineVertexState,
    vertex_entry: &str,
    limits: Limits,
) -> Result<(), String> {
    if vertex.buffer_count > limits.max_vertex_buffers as usize {
        return Err("render pipeline vertex buffer count exceeds the device limit".to_owned());
    }
    if vertex.buffers.len() != vertex.buffer_count {
        return Err("render pipeline vertex buffer count does not match buffers".to_owned());
    }

    let attribute_count = vertex
        .buffers
        .iter()
        .map(|buffer| buffer.attributes.len())
        .try_fold(0usize, |sum, count| {
            sum.checked_add(count)
                .ok_or_else(|| "render pipeline vertex attribute count overflows".to_owned())
        })?;
    if attribute_count > limits.max_vertex_attributes as usize {
        return Err("render pipeline vertex attribute count exceeds the device limit".to_owned());
    }

    let mut locations = BTreeSet::new();
    let mut attribute_classes = BTreeMap::new();
    for buffer in &vertex.buffers {
        if buffer.array_stride != 0 && buffer.array_stride % 4 != 0 {
            return Err(
                "render pipeline vertex buffer arrayStride must be a multiple of 4".to_owned(),
            );
        }
        if buffer.array_stride > u64::from(limits.max_vertex_buffer_array_stride) {
            return Err(
                "render pipeline vertex buffer arrayStride exceeds the device limit".to_owned(),
            );
        }

        for attribute in &buffer.attributes {
            let info = attribute.format.info();
            let alignment = info.byte_size.min(4);
            if attribute.offset % alignment != 0 {
                return Err(
                    "render pipeline vertex attribute offset is not properly aligned".to_owned(),
                );
            }
            let end = attribute
                .offset
                .checked_add(info.byte_size)
                .ok_or_else(|| {
                    "render pipeline vertex attribute byte range overflows".to_owned()
                })?;
            let upper_bound = if buffer.array_stride == 0 {
                u64::from(limits.max_vertex_buffer_array_stride)
            } else {
                buffer.array_stride
            };
            if end > upper_bound {
                return Err(
                    "render pipeline vertex attribute byte range exceeds the buffer arrayStride"
                        .to_owned(),
                );
            }
            if !locations.insert(attribute.shader_location) {
                return Err(
                    "render pipeline vertex attributes must not duplicate shaderLocation"
                        .to_owned(),
                );
            }
            if attribute.shader_location >= limits.max_vertex_attributes {
                return Err(
                    "render pipeline vertex attribute shaderLocation exceeds the device limit"
                        .to_owned(),
                );
            }
            attribute_classes.insert(attribute.shader_location, info.output_class);
        }
    }

    for (location, input) in vertex_inputs(vertex, vertex_entry)? {
        let Some(attribute_class) = attribute_classes.get(&location) else {
            return Err(
                "render pipeline vertex shader input has no matching vertex attribute".to_owned(),
            );
        };
        let input_class = match input.scalar {
            shader_naga::ReflectedTypeScalarClass::Float => FormatOutputClass::Float,
            shader_naga::ReflectedTypeScalarClass::Sint => FormatOutputClass::Sint,
            shader_naga::ReflectedTypeScalarClass::Uint => FormatOutputClass::Uint,
            shader_naga::ReflectedTypeScalarClass::Bool => {
                return Err("render pipeline vertex shader input type is incompatible".to_owned());
            }
        };
        if *attribute_class != input_class {
            return Err("render pipeline vertex shader input type is incompatible".to_owned());
        }
    }

    Ok(())
}

fn vertex_inputs(
    vertex: &RenderPipelineVertexState,
    vertex_entry: &str,
) -> Result<BTreeMap<u32, shader_naga::ReflectedTypeClass>, String> {
    let Some(module) = vertex.shader.module.validated_wgsl() else {
        return Err("vertex module reflection failed".to_owned());
    };
    Ok(module
        .entry_point_io()
        .into_iter()
        .find(|io| io.entry_point == vertex_entry)
        .map(|io| {
            io.inputs
                .into_iter()
                .map(|input| (input.location, input.ty))
                .collect()
        })
        .unwrap_or_default())
}

fn resolve_render_entry(
    stage: &RenderPipelineShaderStage,
    expected_stage: shader_naga::ReflectedShaderStage,
    label: &str,
) -> Result<String, String> {
    if stage.module.is_error() {
        return Err(format!(
            "render pipeline {label} shader module must not be an error module"
        ));
    }
    let Some(module) = stage.module.validated_wgsl() else {
        return Err(format!(
            "render pipeline {label} stage requires a valid WGSL shader module"
        ));
    };
    let entries = module.entry_points();
    let matching_entries = entries
        .iter()
        .filter(|entry| entry.stage == expected_stage)
        .collect::<Vec<_>>();

    match stage.entry_point.as_deref() {
        None => match matching_entries.as_slice() {
            [entry] => Ok(entry.name.clone()),
            [] => Err(format!(
                "render pipeline {label} shader module has no matching entry point"
            )),
            _ => Err(format!(
                "render pipeline {label} entryPoint is required when multiple matching entries exist"
            )),
        },
        Some(name) => matching_entries
            .iter()
            .any(|entry| entry.name == name)
            .then(|| name.to_owned())
            .ok_or_else(|| {
                format!("render pipeline {label} entryPoint must name a matching entry point")
            }),
    }
}

fn validate_render_presence(descriptor: &RenderPipelineDescriptor) -> Result<(), String> {
    if descriptor.fragment.is_none() && descriptor.depth_stencil.is_none() {
        return Err("render pipeline requires a fragment state or depthStencil state".to_owned());
    }
    if descriptor
        .fragment
        .as_ref()
        .is_some_and(|fragment| fragment.target_count == 0)
    {
        return Err("render pipeline fragment targetCount must be at least one".to_owned());
    }
    Ok(())
}

fn validate_render_constants(stage: &RenderPipelineShaderStage) -> Result<(), String> {
    let Some(module) = stage.module.validated_wgsl() else {
        return Err("render pipeline stage requires a valid WGSL shader module".to_owned());
    };
    resolve_pipeline_constants(&module.overrides(), &stage.constants)?;
    Ok(())
}

fn validate_primitive_state(primitive: PrimitiveState) -> Result<(), String> {
    if primitive.strip_index_format.is_some()
        && !matches!(
            primitive.topology,
            PrimitiveTopology::LineStrip | PrimitiveTopology::TriangleStrip
        )
    {
        return Err(
            "render pipeline stripIndexFormat requires a strip primitive topology".to_owned(),
        );
    }
    Ok(())
}

fn validate_depth_bias_state(
    topology: PrimitiveTopology,
    depth_stencil: DepthStencilState,
) -> Result<(), String> {
    if !depth_stencil.depth_bias_slope_scale.is_finite()
        || !depth_stencil.depth_bias_clamp.is_finite()
    {
        return Err("render pipeline depth bias values must be finite".to_owned());
    }

    let has_non_zero_bias = depth_stencil.depth_bias != 0
        || depth_stencil.depth_bias_slope_scale != 0.0
        || depth_stencil.depth_bias_clamp != 0.0;
    if has_non_zero_bias
        && !matches!(
            topology,
            PrimitiveTopology::TriangleList | PrimitiveTopology::TriangleStrip
        )
    {
        return Err("render pipeline non-zero depth bias requires triangle topology".to_owned());
    }
    Ok(())
}

fn validate_depth_stencil_aspects(depth_stencil: DepthStencilState) -> Result<(), String> {
    let caps = depth_stencil.format.caps();
    let has_depth = caps.is_some_and(|caps| caps.aspects.depth);
    let has_stencil = caps.is_some_and(|caps| caps.aspects.stencil);

    if (depth_stencil.depth_compare.is_some() || depth_stencil.depth_write_enabled == Some(true))
        && !has_depth
    {
        return Err("render pipeline depth test or write requires a depth format".to_owned());
    }

    if has_depth
        && (depth_stencil.depth_compare.is_none() || depth_stencil.depth_write_enabled.is_none())
    {
        return Err(
            "render pipeline depth format requires depthCompare and depthWriteEnabled".to_owned(),
        );
    }

    if depth_stencil_uses_stencil(depth_stencil) && !has_stencil {
        return Err("render pipeline stencil state requires a stencil format".to_owned());
    }

    Ok(())
}

fn depth_stencil_uses_stencil(depth_stencil: DepthStencilState) -> bool {
    stencil_face_uses_stencil(depth_stencil.stencil_front)
        || stencil_face_uses_stencil(depth_stencil.stencil_back)
        || depth_stencil.stencil_read_mask != u32::MAX
        || depth_stencil.stencil_write_mask != u32::MAX
}

fn stencil_face_uses_stencil(face: StencilFaceState) -> bool {
    face.compare != CompareFunction::Always
        || face.fail_op != StencilOperation::Keep
        || face.depth_fail_op != StencilOperation::Keep
        || face.pass_op != StencilOperation::Keep
}

fn validate_fragment_depth_output(
    descriptor: &RenderPipelineDescriptor,
    fragment_entry: Option<&str>,
) -> Result<(), String> {
    let Some(fragment) = &descriptor.fragment else {
        return Ok(());
    };
    let Some(entry_name) = fragment_entry else {
        return Ok(());
    };
    let Some(module) = fragment.shader.module.validated_wgsl() else {
        return Err("fragment module reflection failed".to_owned());
    };
    let outputs_frag_depth = module
        .fragment_builtins()
        .into_iter()
        .any(|builtins| builtins.entry_point == entry_name && builtins.frag_depth);
    if outputs_frag_depth
        && !descriptor
            .depth_stencil
            .and_then(|state| state.format.caps())
            .is_some_and(|caps| caps.aspects.depth)
    {
        return Err("render pipeline frag_depth output requires a depth attachment".to_owned());
    }
    Ok(())
}

fn validate_color_targets(
    descriptor: &RenderPipelineDescriptor,
    fragment_entry: Option<&str>,
    limits: Limits,
) -> Result<(), String> {
    let Some(fragment) = &descriptor.fragment else {
        return Ok(());
    };
    if fragment.targets.len() != fragment.target_count {
        return Err("render pipeline fragment target array must match targetCount".to_owned());
    }

    let outputs = fragment_outputs(fragment, fragment_entry)?;
    let mut color_bytes = 0_u32;
    let mut has_alpha_to_coverage_target = false;
    for (index, target) in fragment.targets.iter().enumerate() {
        if target.format.is_undefined() {
            if target.blend {
                return Err("render pipeline undefined color target must not have blend".to_owned());
            }
            continue;
        }

        let caps = target
            .format
            .caps()
            .ok_or_else(|| "render pipeline color target format must be defined".to_owned())?;
        if !caps.renderable {
            return Err("render pipeline color target format must be renderable".to_owned());
        }
        if target.blend && !caps.is_blendable {
            return Err("render pipeline color target format must be blendable".to_owned());
        }
        if descriptor.multisample.alpha_to_coverage_enabled && caps.is_blendable && caps.has_alpha {
            has_alpha_to_coverage_target = true;
        }

        match outputs.get(&(index as u32)) {
            Some(output) => validate_fragment_output_compat(*output, caps)?,
            None if target.write_mask != 0 => {
                return Err(
                    "render pipeline color target without shader output must use writeMask 0"
                        .to_owned(),
                );
            }
            None => {}
        }

        color_bytes = color_bytes
            .checked_add(caps.texel_block_size)
            .ok_or_else(|| "render pipeline color target byte count overflows".to_owned())?;
    }

    if descriptor.multisample.alpha_to_coverage_enabled && !has_alpha_to_coverage_target {
        return Err(
            "render pipeline alphaToCoverage requires an alpha blendable color target".to_owned(),
        );
    }
    if color_bytes > limits.max_color_attachment_bytes_per_sample {
        return Err(
            "render pipeline color target bytes per sample exceed the device limit".to_owned(),
        );
    }

    Ok(())
}

fn fragment_outputs(
    fragment: &RenderPipelineFragmentState,
    fragment_entry: Option<&str>,
) -> Result<BTreeMap<u32, shader_naga::ReflectedTypeClass>, String> {
    let Some(entry_name) = fragment_entry else {
        return Ok(BTreeMap::new());
    };
    let Some(module) = fragment.shader.module.validated_wgsl() else {
        return Err("fragment module reflection failed".to_owned());
    };
    Ok(module
        .entry_point_io()
        .into_iter()
        .find(|io| io.entry_point == entry_name)
        .map(|io| {
            io.outputs
                .into_iter()
                .map(|output| (output.location, output.ty))
                .collect()
        })
        .unwrap_or_default())
}

fn validate_fragment_output_compat(
    output: shader_naga::ReflectedTypeClass,
    caps: FormatCaps,
) -> Result<(), String> {
    let Some(format_class) = caps.output_class else {
        return Err("render pipeline color target format has no output class".to_owned());
    };
    let output_class = match output.scalar {
        shader_naga::ReflectedTypeScalarClass::Float => FormatOutputClass::Float,
        shader_naga::ReflectedTypeScalarClass::Sint => FormatOutputClass::Sint,
        shader_naga::ReflectedTypeScalarClass::Uint => FormatOutputClass::Uint,
        shader_naga::ReflectedTypeScalarClass::Bool => {
            return Err("render pipeline fragment output type is incompatible".to_owned());
        }
    };
    if output_class != format_class || output.components < caps.color_components {
        return Err("render pipeline fragment output type is incompatible".to_owned());
    }
    Ok(())
}

fn validate_render_pipeline_layout(
    descriptor: &RenderPipelineDescriptor,
    vertex_entry: &str,
    fragment_entry: Option<&str>,
) -> Result<(), String> {
    let RenderPipelineLayout::Explicit(layout) = &descriptor.layout else {
        return Ok(());
    };
    if layout.is_error() {
        return Err("render pipeline layout must not be an error pipeline layout".to_owned());
    }

    let mut requirements = stage_resource_bindings(
        &descriptor.vertex.shader,
        vertex_entry,
        PipelineShaderStage::Vertex,
    )?;
    if let Some(fragment) = &descriptor.fragment {
        if let Some(fragment_entry) = fragment_entry {
            requirements.extend(stage_resource_bindings(
                &fragment.shader,
                fragment_entry,
                PipelineShaderStage::Fragment,
            )?);
        }
    }
    validate_pipeline_layout_stage_bindings(layout, &requirements)
}

fn effective_render_bind_group_layouts(
    descriptor: &RenderPipelineDescriptor,
    vertex_entry: &str,
    fragment_entry: Option<&str>,
    limits: Limits,
) -> Result<Vec<Arc<BindGroupLayout>>, String> {
    match &descriptor.layout {
        RenderPipelineLayout::Explicit(layout) => Ok(layout.bind_group_layouts().to_vec()),
        RenderPipelineLayout::Auto => {
            let mut requirements = stage_resource_bindings(
                &descriptor.vertex.shader,
                vertex_entry,
                PipelineShaderStage::Vertex,
            )?;
            if let Some(fragment) = &descriptor.fragment {
                if let Some(fragment_entry) = fragment_entry {
                    requirements.extend(stage_resource_bindings(
                        &fragment.shader,
                        fragment_entry,
                        PipelineShaderStage::Fragment,
                    )?);
                }
            }
            derive_bind_group_layouts(requirements, limits)
        }
    }
}

fn stage_resource_bindings(
    stage: &RenderPipelineShaderStage,
    entry_point: &str,
    pipeline_stage: PipelineShaderStage,
) -> Result<Vec<StageResourceBinding>, String> {
    let Some(module) = stage.module.validated_wgsl() else {
        return Err("render pipeline stage requires a valid WGSL shader module".to_owned());
    };
    Ok(module
        .resource_bindings_for_entry(entry_point)?
        .into_iter()
        .map(|binding| StageResourceBinding {
            stage: pipeline_stage,
            binding,
        })
        .collect())
}

fn validate_multisample_state(
    descriptor: &RenderPipelineDescriptor,
    fragment_entry: Option<&str>,
) -> Result<(), String> {
    let multisample = descriptor.multisample;
    if !matches!(multisample.count, 1 | 4) {
        return Err("render pipeline multisample count must be 1 or 4".to_owned());
    }
    if multisample.alpha_to_coverage_enabled && multisample.count != 4 {
        return Err("render pipeline alphaToCoverage requires multisample count 4".to_owned());
    }
    if multisample.alpha_to_coverage_enabled {
        if let (Some(fragment), Some(entry_name)) = (&descriptor.fragment, fragment_entry) {
            let module = fragment
                .shader
                .module
                .validated_wgsl()
                .ok_or_else(|| "fragment module reflection failed".to_owned())?;
            if module
                .fragment_builtins()
                .into_iter()
                .any(|builtins| builtins.entry_point == entry_name && builtins.sample_mask)
            {
                return Err(
                    "render pipeline alphaToCoverage conflicts with fragment sample_mask output"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MipmapFilterMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerDescriptor {
    pub address_mode_u: Option<AddressMode>,
    pub address_mode_v: Option<AddressMode>,
    pub address_mode_w: Option<AddressMode>,
    pub mag_filter: Option<FilterMode>,
    pub min_filter: Option<FilterMode>,
    pub mipmap_filter: Option<MipmapFilterMode>,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare: Option<CompareFunction>,
    pub max_anisotropy: u16,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self {
            address_mode_u: None,
            address_mode_v: None,
            address_mode_w: None,
            mag_filter: None,
            min_filter: None,
            mipmap_filter: None,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            max_anisotropy: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedSamplerDescriptor {
    pub address_mode_u: AddressMode,
    pub address_mode_v: AddressMode,
    pub address_mode_w: AddressMode,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub mipmap_filter: MipmapFilterMode,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare: Option<CompareFunction>,
    pub max_anisotropy: u16,
}

impl ResolvedSamplerDescriptor {
    fn from_descriptor(descriptor: SamplerDescriptor) -> Self {
        Self {
            address_mode_u: descriptor
                .address_mode_u
                .unwrap_or(AddressMode::ClampToEdge),
            address_mode_v: descriptor
                .address_mode_v
                .unwrap_or(AddressMode::ClampToEdge),
            address_mode_w: descriptor
                .address_mode_w
                .unwrap_or(AddressMode::ClampToEdge),
            mag_filter: descriptor.mag_filter.unwrap_or(FilterMode::Nearest),
            min_filter: descriptor.min_filter.unwrap_or(FilterMode::Nearest),
            mipmap_filter: descriptor
                .mipmap_filter
                .unwrap_or(MipmapFilterMode::Nearest),
            lod_min_clamp: descriptor.lod_min_clamp,
            lod_max_clamp: descriptor.lod_max_clamp,
            compare: descriptor.compare,
            max_anisotropy: descriptor.max_anisotropy,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    inner: Arc<SamplerInner>,
}

#[derive(Debug)]
struct SamplerInner {
    _hal: Option<HalSampler>,
    descriptor: ResolvedSamplerDescriptor,
    is_error: bool,
}

impl Sampler {
    fn new(descriptor: ResolvedSamplerDescriptor, hal: Option<HalSampler>, is_error: bool) -> Self {
        Self {
            inner: Arc::new(SamplerInner {
                _hal: hal,
                descriptor,
                is_error,
            }),
        }
    }

    #[must_use]
    pub fn descriptor(&self) -> ResolvedSamplerDescriptor {
        self.inner.descriptor
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferMapState {
    Unmapped,
    Pending,
    Mapped,
}

#[derive(Debug, Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
}

#[derive(Debug)]
struct BufferInner {
    hal: Option<HalBuffer>,
    usage: BufferUsage,
    size: u64,
    host: HostBuffer,
    state: Mutex<BufferState>,
}

#[derive(Debug)]
struct BufferState {
    map_state: BufferMapState,
    is_error: bool,
    is_destroyed: bool,
    pending_map: Option<PendingMap>,
    active_map: Option<ActiveMap>,
}

#[derive(Debug)]
struct PendingMap {
    mode: MapMode,
    offset: u64,
    size: u64,
    outcome: MapAsyncStatus,
}

#[derive(Debug, Clone, Copy)]
struct ActiveMap {
    mode: MapMode,
    offset: u64,
    size: u64,
}

struct HostBuffer {
    bytes: Box<[UnsafeCell<u8>]>,
}

impl HostBuffer {
    fn new(size: u64) -> Self {
        debug_assert!(
            usize::try_from(size).is_ok(),
            "buffer sizes above usize::MAX must be rejected before allocation"
        );
        let len = match usize::try_from(size) {
            Ok(len) => len,
            Err(_) => usize::MAX,
        };
        let bytes = (0..len)
            .map(|_| UnsafeCell::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { bytes }
    }

    fn ptr_at(&self, offset: u64) -> Option<*mut u8> {
        let offset = usize::try_from(offset).ok()?;
        if offset > self.bytes.len() {
            return None;
        }
        // One-past-the-end is valid for zero-sized mapped ranges.
        Some(unsafe { self.bytes.as_ptr().add(offset).cast::<u8>().cast_mut() })
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<(), String> {
        let offset = usize::try_from(offset).map_err(|_| "host buffer offset is too large")?;
        let end = offset
            .checked_add(data.len())
            .ok_or("host buffer write range overflows")?;
        if end > self.bytes.len() {
            return Err("host buffer write range exceeds buffer size".to_owned());
        }
        for (cell, byte) in self.bytes[offset..end].iter().zip(data) {
            unsafe {
                *cell.get() = *byte;
            }
        }
        Ok(())
    }

    fn read(&self, offset: u64, size: u64) -> Result<Vec<u8>, String> {
        let offset = usize::try_from(offset).map_err(|_| "host buffer offset is too large")?;
        let size = usize::try_from(size).map_err(|_| "host buffer read size is too large")?;
        let end = offset
            .checked_add(size)
            .ok_or("host buffer read range overflows")?;
        if end > self.bytes.len() {
            return Err("host buffer read range exceeds buffer size".to_owned());
        }
        let mut data = Vec::with_capacity(size);
        for cell in &self.bytes[offset..end] {
            data.push(unsafe { *cell.get() });
        }
        Ok(data)
    }
}

impl fmt::Debug for HostBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostBuffer")
            .field("len", &self.bytes.len())
            .finish()
    }
}

// Mapped ranges expose raw pointers whose synchronization is governed by the
// WebGPU map/unmap state machine rather than Rust references.
unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl Buffer {
    fn new(descriptor: BufferDescriptor, hal: Option<HalBuffer>, is_error: bool) -> Self {
        let map_state = if descriptor.mapped_at_creation && !is_error {
            BufferMapState::Mapped
        } else {
            BufferMapState::Unmapped
        };
        let active_map = if descriptor.mapped_at_creation && !is_error {
            Some(ActiveMap {
                mode: MapMode::Write,
                offset: 0,
                size: descriptor.size,
            })
        } else {
            None
        };
        Self {
            inner: Arc::new(BufferInner {
                hal,
                usage: descriptor.usage,
                size: descriptor.size,
                host: HostBuffer::new(if is_error { 0 } else { descriptor.size }),
                state: Mutex::new(BufferState {
                    map_state,
                    is_error,
                    is_destroyed: false,
                    pending_map: None,
                    active_map,
                }),
            }),
        }
    }

    #[must_use]
    pub fn size(&self) -> u64 {
        self.inner.size
    }

    #[must_use]
    pub fn usage(&self) -> BufferUsage {
        self.inner.usage
    }

    #[must_use]
    pub fn map_state(&self) -> BufferMapState {
        self.inner.state.lock().map_state
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.state.lock().is_error
    }

    #[must_use]
    pub(crate) fn is_destroyed(&self) -> bool {
        self.inner.state.lock().is_destroyed
    }

    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Marks any pending map as aborted without draining `pending_map`.
    ///
    /// The transient invariant is `map_state == Unmapped` while
    /// `pending_map.is_some()` until the callback consumes it through
    /// `resolve_pending_map`.
    pub fn destroy(&self) {
        let mut state = self.inner.state.lock();
        state.is_destroyed = true;
        if let Some(pending) = state.pending_map.as_mut() {
            pending.outcome = MapAsyncStatus::Aborted;
        }
        state.map_state = BufferMapState::Unmapped;
        state.active_map = None;
    }

    /// Marks any pending map as aborted without draining `pending_map`.
    ///
    /// The transient invariant is `map_state == Unmapped` while
    /// `pending_map.is_some()` until the callback consumes it through
    /// `resolve_pending_map`.
    pub fn unmap(&self) -> Option<DeviceError> {
        let mut state = self.inner.state.lock();
        let active_map = state.active_map;
        if let Some(pending) = state.pending_map.as_mut() {
            pending.outcome = MapAsyncStatus::Aborted;
        }
        state.map_state = BufferMapState::Unmapped;
        state.active_map = None;
        drop(state);

        let active_map = active_map?;
        if active_map.mode != MapMode::Write {
            return None;
        }
        let Some(hal) = &self.inner.hal else {
            return None;
        };
        if hal.mapped_ptr().is_some() {
            return None;
        }
        let data = match self.inner.host.read(active_map.offset, active_map.size) {
            Ok(data) => data,
            Err(message) => return Some(DeviceError::internal(message)),
        };
        hal.write(active_map.offset, &data)
            .err()
            .map(|error| DeviceError::internal(error.to_string()))
    }

    pub fn begin_map(&self, mode: MapMode, offset: u64, size: u64) -> Result<(), &'static str> {
        let mut state = self.inner.state.lock();
        if state.is_error {
            return Err("cannot map an error buffer");
        }
        if state.is_destroyed {
            return Err("cannot map a destroyed buffer");
        }
        if state.map_state == BufferMapState::Mapped {
            return Err("buffer is already mapped");
        }
        if state.map_state == BufferMapState::Pending {
            return Err("buffer already has a pending map");
        }

        match mode {
            MapMode::Read if !self.inner.usage.contains(BufferUsage::MAP_READ) => {
                return Err("read mapping requires MapRead usage");
            }
            MapMode::Write if !self.inner.usage.contains(BufferUsage::MAP_WRITE) => {
                return Err("write mapping requires MapWrite usage");
            }
            _ => {}
        }

        if !offset.is_multiple_of(8) {
            return Err("map offset must be 8-byte aligned");
        }
        if !size.is_multiple_of(4) {
            return Err("map size must be 4-byte aligned");
        }
        let Some(end) = offset.checked_add(size) else {
            return Err("map range overflows");
        };
        if offset > self.inner.size || end > self.inner.size {
            return Err("map range exceeds buffer size");
        }

        state.map_state = BufferMapState::Pending;
        state.pending_map = Some(PendingMap {
            mode,
            offset,
            size,
            outcome: MapAsyncStatus::Success,
        });
        state.active_map = None;
        Ok(())
    }

    #[must_use]
    pub fn resolve_pending_map(&self) -> MapAsyncStatus {
        let mut state = self.inner.state.lock();
        let pending = state.pending_map.take();
        let mut outcome = pending
            .as_ref()
            .map(|pending| pending.outcome)
            .unwrap_or(MapAsyncStatus::Aborted);
        if outcome == MapAsyncStatus::Success {
            if let Some(pending) = pending.as_ref() {
                if pending.mode == MapMode::Read {
                    if let Some(hal) = &self.inner.hal {
                        if hal.mapped_ptr().is_none() {
                            outcome = match hal.read(pending.offset, pending.size) {
                                Ok(bytes)
                                    if self.inner.host.write(pending.offset, &bytes).is_ok() =>
                                {
                                    MapAsyncStatus::Success
                                }
                                _ => MapAsyncStatus::Error,
                            };
                        }
                    }
                }
            }
        }
        state.map_state = if outcome == MapAsyncStatus::Success {
            state.active_map = pending.map(|pending| ActiveMap {
                mode: pending.mode,
                offset: pending.offset,
                size: pending.size,
            });
            BufferMapState::Mapped
        } else {
            state.active_map = None;
            BufferMapState::Unmapped
        };
        outcome
    }

    /// Marks a pending map as aborted without draining `pending_map`.
    ///
    /// The transient invariant is `map_state == Unmapped` while
    /// `pending_map.is_some()` until the callback consumes it through
    /// `resolve_pending_map`.
    pub fn abort_pending_map(&self) {
        let mut state = self.inner.state.lock();
        if let Some(pending) = state.pending_map.as_mut() {
            pending.outcome = MapAsyncStatus::Aborted;
            state.map_state = BufferMapState::Unmapped;
            state.active_map = None;
        }
    }

    #[must_use]
    pub fn mapped_range(
        &self,
        const_access: bool,
        offset: u64,
        size: Option<u64>,
    ) -> Option<*mut u8> {
        let state = self.inner.state.lock();
        if state.is_destroyed || state.map_state != BufferMapState::Mapped {
            return None;
        }
        let active = state.active_map?;
        if !const_access && active.mode == MapMode::Read {
            return None;
        }
        let map_end = active.offset.checked_add(active.size)?;
        let size = size.unwrap_or_else(|| map_end.saturating_sub(offset));
        if offset < active.offset || offset > map_end {
            return None;
        }
        let end = offset.checked_add(size)?;
        if end > map_end {
            return None;
        }
        drop(state);
        if let Some(mapped_ptr) = self.inner.hal.as_ref().and_then(HalBuffer::mapped_ptr) {
            let offset = usize::try_from(offset).ok()?;
            return Some(unsafe { mapped_ptr.as_ptr().add(offset) });
        }
        self.inner.host.ptr_at(offset)
    }

    pub(crate) fn write_from_queue(&self, offset: u64, data: &[u8]) -> Option<DeviceError> {
        let size = match u64::try_from(data.len()) {
            Ok(size) => size,
            Err(_) => {
                return Some(DeviceError::validation("queue write size is too large"));
            }
        };
        if let Err(message) = self.validate_queue_write(offset, size) {
            return Some(DeviceError::validation(message));
        }
        if let Some(hal) = &self.inner.hal {
            if let Err(error) = hal.write(offset, data) {
                return Some(DeviceError::internal(error.to_string()));
            }
        }
        None
    }

    pub fn hal(&self) -> Option<HalBuffer> {
        self.inner.hal.clone()
    }

    pub fn validate_queue_write(&self, offset: u64, size: u64) -> Result<(), &'static str> {
        let state = self.inner.state.lock();
        if state.is_error {
            return Err("cannot write to an error buffer");
        }
        if state.is_destroyed {
            return Err("cannot write to a destroyed buffer");
        }
        if state.map_state != BufferMapState::Unmapped {
            return Err("cannot write to a mapped buffer");
        }
        if !self.inner.usage.contains(BufferUsage::COPY_DST) {
            return Err("queue write requires CopyDst usage");
        }
        if !offset.is_multiple_of(4) {
            return Err("queue write offset must be 4-byte aligned");
        }
        if !size.is_multiple_of(4) {
            return Err("queue write size must be 4-byte aligned");
        }
        let Some(end) = offset.checked_add(size) else {
            return Err("queue write range overflows");
        };
        if end > self.inner.size {
            return Err("queue write range exceeds buffer size");
        }
        Ok(())
    }
}

fn validate_buffer_descriptor(
    descriptor: &BufferDescriptor,
    limits: Limits,
) -> Option<&'static str> {
    let usage = descriptor.usage;
    if usage.bits() == 0 {
        return Some("buffer usage must be non-zero");
    }
    if usage.contains(BufferUsage::MAP_READ) {
        let allowed = (BufferUsage::MAP_READ | BufferUsage::COPY_DST).bits();
        if usage.bits() & !allowed != 0 {
            return Some("MapRead buffers may only combine with CopyDst");
        }
    }
    if usage.contains(BufferUsage::MAP_WRITE) {
        let allowed = (BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC).bits();
        if usage.bits() & !allowed != 0 {
            return Some("MapWrite buffers may only combine with CopySrc");
        }
    }
    if descriptor.size > limits.max_buffer_size {
        return Some("buffer size exceeds device limit");
    }
    if descriptor.mapped_at_creation && !descriptor.size.is_multiple_of(4) {
        return Some("mappedAtCreation buffer size must be 4-byte aligned");
    }
    None
}

fn validate_texture_descriptor(
    descriptor: &TextureDescriptor,
    limits: Limits,
) -> Option<&'static str> {
    let usage = descriptor.usage;
    let size = descriptor.size;
    let multisampled = descriptor.sample_count > 1;

    if usage.bits() == 0 {
        return Some("texture usage must be non-zero");
    }
    if descriptor.sample_count != 1 && descriptor.sample_count != 4 {
        return Some("texture sample count must be 1 or 4");
    }
    if multisampled && descriptor.mip_level_count != 1 {
        return Some("multisampled textures must have exactly one mip level");
    }
    if multisampled && descriptor.dimension != TextureDimension::D2 {
        return Some("multisampled textures must be 2D");
    }
    if multisampled && size.depth_or_array_layers != 1 {
        return Some("multisampled textures must have one array layer");
    }
    if multisampled && usage.contains(TextureUsage::STORAGE_BINDING) {
        return Some("multisampled textures cannot use StorageBinding");
    }
    if multisampled && !usage.contains(TextureUsage::RENDER_ATTACHMENT) {
        return Some("multisampled textures must use RenderAttachment");
    }
    if descriptor.mip_level_count == 0 {
        return Some("texture mipLevelCount must be at least 1");
    }
    if descriptor.mip_level_count > max_texture_mips(size, descriptor.dimension) {
        return Some("texture mipLevelCount exceeds the texture size");
    }
    if descriptor.dimension == TextureDimension::D1 && descriptor.mip_level_count != 1 {
        return Some("1D textures must have exactly one mip level");
    }
    if descriptor.dimension == TextureDimension::D2
        && size.depth_or_array_layers > limits.max_texture_array_layers
    {
        return Some("texture array layers exceed device limit");
    }
    match descriptor.dimension {
        TextureDimension::D1 => {
            if size.width == 0 || size.width > limits.max_texture_dimension_1d {
                return Some("1D texture width is out of range");
            }
            if size.height != 1 {
                return Some("1D texture height must be 1");
            }
            if size.depth_or_array_layers != 1 {
                return Some("1D texture depthOrArrayLayers must be 1");
            }
        }
        TextureDimension::D2 => {
            if size.width == 0 || size.width > limits.max_texture_dimension_2d {
                return Some("2D texture width is out of range");
            }
            if size.height == 0 || size.height > limits.max_texture_dimension_2d {
                return Some("2D texture height is out of range");
            }
            if size.depth_or_array_layers == 0 {
                return Some("2D texture depthOrArrayLayers must be at least 1");
            }
        }
        TextureDimension::D3 => {
            if size.width == 0 || size.width > limits.max_texture_dimension_3d {
                return Some("3D texture width is out of range");
            }
            if size.height == 0 || size.height > limits.max_texture_dimension_3d {
                return Some("3D texture height is out of range");
            }
            if size.depth_or_array_layers == 0
                || size.depth_or_array_layers > limits.max_texture_dimension_3d
            {
                return Some("3D texture depth is out of range");
            }
        }
    }
    if usage.contains(TextureUsage::RENDER_ATTACHMENT)
        && descriptor.dimension != TextureDimension::D2
    {
        return Some("RenderAttachment textures must be 2D");
    }
    let Some(format_caps) = descriptor.format.caps() else {
        return Some("texture format must not be Undefined");
    };
    if multisampled && !format_caps.multisample_capable {
        return Some("multisampled texture format must support multisampling");
    }
    if (format_caps.aspects.depth || format_caps.aspects.stencil)
        && descriptor.dimension != TextureDimension::D2
    {
        return Some("depth/stencil texture formats must be 2D");
    }
    if usage.contains(TextureUsage::RENDER_ATTACHMENT) && !format_caps.renderable {
        return Some("RenderAttachment texture format must be renderable");
    }
    if usage.contains(TextureUsage::STORAGE_BINDING) && !format_caps.storage_capable {
        return Some("StorageBinding texture format must support storage usage");
    }
    None
}

fn validate_texture_view_descriptor(
    texture: &Texture,
    descriptor: &ResolvedTextureViewDescriptor,
) -> Option<&'static str> {
    let ResolvedTextureViewDescriptor {
        format,
        dimension,
        mip_level_count,
        array_layer_count,
        aspect,
        ..
    } = *descriptor;

    if mip_level_count == 0 {
        return Some("texture view mipLevelCount must be greater than zero");
    }
    if array_layer_count == 0 {
        return Some("texture view arrayLayerCount must be greater than zero");
    }
    let Some(mip_end) = descriptor.base_mip_level.checked_add(mip_level_count) else {
        return Some("texture view mip range overflows");
    };
    if mip_end > texture.mip_level_count() {
        return Some("texture view mip range exceeds texture mip levels");
    }

    let texture_layers = texture.size().depth_or_array_layers;
    let Some(layer_end) = descriptor.base_array_layer.checked_add(array_layer_count) else {
        return Some("texture view array layer range overflows");
    };
    if texture.dimension() != TextureDimension::D3 && layer_end > texture_layers {
        return Some("texture view array layer range exceeds texture layers");
    }

    match texture.dimension() {
        TextureDimension::D1 if dimension != TextureViewDimension::D1 => {
            return Some("1D textures require 1D views");
        }
        TextureDimension::D3 if dimension != TextureViewDimension::D3 => {
            return Some("3D textures require 3D views");
        }
        TextureDimension::D2 => match dimension {
            TextureViewDimension::D2 if array_layer_count != 1 => {
                return Some("2D texture views require exactly one array layer");
            }
            TextureViewDimension::D2Array => {}
            TextureViewDimension::Cube if array_layer_count != 6 => {
                return Some("cube texture views require exactly six array layers");
            }
            TextureViewDimension::CubeArray if !array_layer_count.is_multiple_of(6) => {
                return Some(
                    "cube-array texture views require a layer count that is a multiple of six",
                );
            }
            TextureViewDimension::CubeArray => {}
            TextureViewDimension::D1 | TextureViewDimension::D3 => {
                return Some("2D textures require 2D-compatible views");
            }
            TextureViewDimension::D2 => {}
            _ => return Some("texture view dimension is unsupported"),
        },
        _ => {}
    }

    if !texture.is_view_format_compatible(format) {
        return Some("texture view format is not compatible with the texture");
    }

    let Some(format_caps) = format.caps() else {
        return Some("texture view format must not be Undefined");
    };
    match aspect {
        TextureAspect::All => {}
        TextureAspect::DepthOnly if !format_caps.aspects.depth => {
            return Some("DepthOnly texture views require a depth format");
        }
        TextureAspect::StencilOnly if !format_caps.aspects.stencil => {
            return Some("StencilOnly texture views require a stencil format");
        }
        TextureAspect::DepthOnly | TextureAspect::StencilOnly => {}
    }

    None
}

fn validate_queue_write_texture(
    texture: &Texture,
    mip_level: u32,
    origin: Origin3d,
    write_size: Extent3d,
    aspect: TextureAspect,
    layout: TexelCopyBufferLayout,
    data_size: u64,
) -> Result<(), String> {
    if !texture.usage().contains(TextureUsage::COPY_DST) {
        return Err("queue texture write destination usage must include CopyDst".to_owned());
    }
    if texture.is_error() || texture.is_destroyed() {
        return Err("queue texture write destination must be a valid live texture".to_owned());
    }
    if texture.sample_count() != 1 {
        return Err("queue texture write destination sampleCount must be one".to_owned());
    }
    if mip_level >= texture.mip_level_count() {
        return Err("queue texture write mipLevel is out of range".to_owned());
    }

    let Some(format_caps) = texture.format().caps() else {
        return Err("queue texture write format must not be Undefined".to_owned());
    };
    match aspect {
        TextureAspect::All => {}
        TextureAspect::DepthOnly if !format_caps.aspects.depth => {
            return Err("DepthOnly texture writes require a depth format".to_owned());
        }
        TextureAspect::StencilOnly if !format_caps.aspects.stencil => {
            return Err("StencilOnly texture writes require a stencil format".to_owned());
        }
        TextureAspect::DepthOnly | TextureAspect::StencilOnly => {}
    }

    let subresource = texture.subresource_size(mip_level);
    if origin
        .x
        .checked_add(write_size.width)
        .is_none_or(|end| end > subresource.width)
        || origin
            .y
            .checked_add(write_size.height)
            .is_none_or(|end| end > subresource.height)
        || origin
            .z
            .checked_add(write_size.depth_or_array_layers)
            .is_none_or(|end| end > subresource.depth_or_array_layers)
    {
        return Err("queue texture write range exceeds the texture subresource".to_owned());
    }
    if texture.dimension() == TextureDimension::D2 && write_size.depth_or_array_layers != 1 {
        return Err(
            "queue texture writes to 2D textures require depthOrArrayLayers to be one".to_owned(),
        );
    }

    let required_bytes = validate_texel_copy_layout(
        format_caps,
        aspect,
        write_size,
        layout,
        "queue texture write",
        false,
    )?;
    let required_end = layout
        .offset
        .checked_add(required_bytes)
        .ok_or("queue texture write data range overflows")?;
    if required_end > data_size {
        return Err("queue texture write dataSize is too small".to_owned());
    }

    Ok(())
}

impl Texture {
    fn subresource_size(&self, mip_level: u32) -> Extent3d {
        let size = self.size();
        let mip = |value: u32| value.checked_shr(mip_level).unwrap_or(0).max(1);
        Extent3d {
            width: mip(size.width),
            height: match self.dimension() {
                TextureDimension::D1 => 1,
                TextureDimension::D2 | TextureDimension::D3 => mip(size.height),
            },
            depth_or_array_layers: match self.dimension() {
                TextureDimension::D1 => 1,
                TextureDimension::D2 => size.depth_or_array_layers,
                TextureDimension::D3 => mip(size.depth_or_array_layers),
            },
        }
    }
}

fn validate_texel_copy_layout(
    format_caps: FormatCaps,
    aspect: TextureAspect,
    write_size: Extent3d,
    layout: TexelCopyBufferLayout,
    label: &str,
    require_bytes_per_row_alignment: bool,
) -> Result<u64, String> {
    let width_blocks = div_ceil_u32(write_size.width, format_caps.block_w);
    let height_blocks = div_ceil_u32(write_size.height, format_caps.block_h);
    let depth = write_size.depth_or_array_layers;
    let block_size = texel_copy_block_size(format_caps, aspect);
    let last_row_bytes = u64::from(width_blocks)
        .checked_mul(u64::from(block_size))
        .ok_or_else(|| format!("{label} row byte size overflows"))?;

    if let Some(bytes_per_row) = layout.bytes_per_row {
        if require_bytes_per_row_alignment && !bytes_per_row.is_multiple_of(256) {
            return Err(format!("{label} bytesPerRow must be 256-byte aligned"));
        }
        if u64::from(bytes_per_row) < last_row_bytes {
            return Err(format!("{label} bytesPerRow is too small"));
        }
    } else if height_blocks > 1 || depth > 1 {
        return Err(format!(
            "{label} bytesPerRow is required for multi-row copies"
        ));
    }

    if let Some(rows_per_image) = layout.rows_per_image {
        if rows_per_image < height_blocks {
            return Err(format!("{label} rowsPerImage is too small"));
        }
    } else if depth > 1 {
        return Err(format!(
            "{label} rowsPerImage is required for multi-image copies"
        ));
    }

    required_bytes_in_texel_copy(
        layout.bytes_per_row,
        layout.rows_per_image,
        height_blocks,
        depth,
        last_row_bytes,
        label,
    )
}

fn required_bytes_in_texel_copy(
    bytes_per_row: Option<u32>,
    rows_per_image: Option<u32>,
    height_blocks: u32,
    depth: u32,
    last_row_bytes: u64,
    label: &str,
) -> Result<u64, String> {
    if last_row_bytes == 0 || height_blocks == 0 || depth == 0 {
        return Ok(0);
    }

    let bytes_per_row = u64::from(bytes_per_row.unwrap_or(0));
    let rows_per_image = u64::from(rows_per_image.unwrap_or(height_blocks));
    let image_offset_rows = rows_per_image
        .checked_mul(u64::from(depth.saturating_sub(1)))
        .ok_or_else(|| format!("{label} required byte size overflows"))?;
    let row_offset_rows = u64::from(height_blocks.saturating_sub(1));
    let offset_rows = image_offset_rows
        .checked_add(row_offset_rows)
        .ok_or_else(|| format!("{label} required byte size overflows"))?;
    let offset_bytes = bytes_per_row
        .checked_mul(offset_rows)
        .ok_or_else(|| format!("{label} required byte size overflows"))?;
    offset_bytes
        .checked_add(last_row_bytes)
        .ok_or_else(|| format!("{label} required byte size overflows"))
}

fn texel_copy_block_size(format_caps: FormatCaps, aspect: TextureAspect) -> u32 {
    if aspect == TextureAspect::StencilOnly {
        1
    } else {
        format_caps.texel_block_size
    }
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    if value == 0 {
        0
    } else {
        u64::from(value).div_ceil(u64::from(divisor)) as u32
    }
}

fn hal_texture_descriptor(descriptor: &TextureDescriptor) -> HalTextureDescriptor {
    HalTextureDescriptor {
        format: hal_texture_format(descriptor.format),
        width: descriptor.size.width,
        height: descriptor.size.height,
        depth_or_array_layers: descriptor.size.depth_or_array_layers,
        mip_level_count: descriptor.mip_level_count,
        sample_count: descriptor.sample_count,
        usage: hal_texture_usage(descriptor.usage),
    }
}

fn hal_texture_format(format: TextureFormat) -> HalTextureFormat {
    match format.raw() {
        TextureFormat::R8_UNORM => HalTextureFormat::R8Unorm,
        TextureFormat::RGBA8_UNORM => HalTextureFormat::Rgba8Unorm,
        TextureFormat::BGRA8_UNORM => HalTextureFormat::Bgra8Unorm,
        _ => HalTextureFormat::Unsupported,
    }
}

fn hal_texture_usage(usage: TextureUsage) -> HalTextureUsage {
    HalTextureUsage {
        copy_src: usage.contains(TextureUsage::COPY_SRC),
        copy_dst: usage.contains(TextureUsage::COPY_DST),
        texture_binding: usage.contains(TextureUsage::TEXTURE_BINDING),
        storage_binding: usage.contains(TextureUsage::STORAGE_BINDING),
        render_attachment: usage.contains(TextureUsage::RENDER_ATTACHMENT),
    }
}

fn hal_sampler_descriptor(descriptor: &ResolvedSamplerDescriptor) -> HalSamplerDescriptor {
    HalSamplerDescriptor {
        address_mode_u: hal_address_mode(descriptor.address_mode_u),
        address_mode_v: hal_address_mode(descriptor.address_mode_v),
        address_mode_w: hal_address_mode(descriptor.address_mode_w),
        mag_filter: hal_filter_mode(descriptor.mag_filter),
        min_filter: hal_filter_mode(descriptor.min_filter),
        mipmap_filter: hal_mipmap_filter_mode(descriptor.mipmap_filter),
        lod_min_clamp: descriptor.lod_min_clamp,
        lod_max_clamp: descriptor.lod_max_clamp,
        compare: descriptor.compare.map(hal_compare_function),
        max_anisotropy: descriptor.max_anisotropy,
    }
}

fn hal_address_mode(mode: AddressMode) -> HalAddressMode {
    match mode {
        AddressMode::ClampToEdge => HalAddressMode::ClampToEdge,
        AddressMode::Repeat => HalAddressMode::Repeat,
        AddressMode::MirrorRepeat => HalAddressMode::MirrorRepeat,
    }
}

fn hal_filter_mode(mode: FilterMode) -> HalFilterMode {
    match mode {
        FilterMode::Nearest => HalFilterMode::Nearest,
        FilterMode::Linear => HalFilterMode::Linear,
    }
}

fn hal_mipmap_filter_mode(mode: MipmapFilterMode) -> HalMipmapFilterMode {
    match mode {
        MipmapFilterMode::Nearest => HalMipmapFilterMode::Nearest,
        MipmapFilterMode::Linear => HalMipmapFilterMode::Linear,
    }
}

fn hal_compare_function(compare: CompareFunction) -> HalCompareFunction {
    match compare {
        CompareFunction::Never => HalCompareFunction::Never,
        CompareFunction::Less => HalCompareFunction::Less,
        CompareFunction::Equal => HalCompareFunction::Equal,
        CompareFunction::LessEqual => HalCompareFunction::LessEqual,
        CompareFunction::Greater => HalCompareFunction::Greater,
        CompareFunction::NotEqual => HalCompareFunction::NotEqual,
        CompareFunction::GreaterEqual => HalCompareFunction::GreaterEqual,
        CompareFunction::Always => HalCompareFunction::Always,
    }
}

fn hal_origin(origin: Origin3d) -> HalOrigin3d {
    HalOrigin3d {
        x: origin.x,
        y: origin.y,
        z: origin.z,
    }
}

fn hal_extent(extent: Extent3d) -> HalExtent3d {
    HalExtent3d {
        width: extent.width,
        height: extent.height,
        depth_or_array_layers: extent.depth_or_array_layers,
    }
}

fn hal_buffer_texture_layout(
    layout: TexelCopyBufferLayout,
    texture: &Texture,
    copy_size: Extent3d,
) -> Option<HalBufferTextureLayout> {
    let format_caps = texture.format().caps()?;
    let width_blocks = div_ceil_u32(copy_size.width, format_caps.block_w);
    let height_blocks = div_ceil_u32(copy_size.height, format_caps.block_h);
    let row_bytes = width_blocks.checked_mul(format_caps.texel_block_size)?;
    Some(HalBufferTextureLayout {
        offset: layout.offset,
        bytes_per_row: layout.bytes_per_row.unwrap_or(row_bytes),
        rows_per_image: layout.rows_per_image.unwrap_or(height_blocks),
    })
}

fn hal_command_execution(op: &CommandExecution) -> Option<HalCopy> {
    match op {
        CommandExecution::BufferCopy(copy) => {
            let source = copy.source.hal()?;
            let destination = copy.destination.hal()?;
            Some(HalCopy::Buffer(HalBufferCopy {
                source,
                source_offset: copy.source_offset,
                destination,
                destination_offset: copy.destination_offset,
                size: copy.size,
            }))
        }
        CommandExecution::TextureCopy(copy) => hal_texture_copy_execution(copy),
        CommandExecution::ComputePass(pass) => hal_compute_pass_execution(pass),
        CommandExecution::RenderPass(pass) => hal_render_pass_execution(pass),
    }
}

fn hal_texture_copy_execution(copy: &TextureCopyCommand) -> Option<HalCopy> {
    match copy {
        TextureCopyCommand::BufferToTexture {
            source,
            destination,
            copy_size,
        } => {
            let buffer = source.buffer.hal()?;
            let texture = destination.texture.hal()?;
            let buffer_layout =
                hal_buffer_texture_layout(source.layout, &destination.texture, *copy_size)?;
            Some(HalCopy::BufferToTexture(HalBufferTextureCopy {
                buffer,
                buffer_layout,
                texture,
                mip_level: destination.mip_level,
                origin: hal_origin(destination.origin),
                extent: hal_extent(*copy_size),
            }))
        }
        TextureCopyCommand::TextureToBuffer {
            source,
            destination,
            copy_size,
        } => {
            let buffer = destination.buffer.hal()?;
            let texture = source.texture.hal()?;
            let buffer_layout =
                hal_buffer_texture_layout(destination.layout, &source.texture, *copy_size)?;
            Some(HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer,
                buffer_layout,
                texture,
                mip_level: source.mip_level,
                origin: hal_origin(source.origin),
                extent: hal_extent(*copy_size),
            }))
        }
        TextureCopyCommand::TextureToTexture {
            source,
            destination,
            copy_size,
        } => {
            let source_texture = source.texture.hal()?;
            let destination_texture = destination.texture.hal()?;
            Some(HalCopy::TextureToTexture(HalTextureCopy {
                source: source_texture,
                source_mip_level: source.mip_level,
                source_origin: hal_origin(source.origin),
                destination: destination_texture,
                destination_mip_level: destination.mip_level,
                destination_origin: hal_origin(destination.origin),
                extent: hal_extent(*copy_size),
            }))
        }
    }
}

fn hal_compute_pass_execution(pass: &ComputePassCommand) -> Option<HalCopy> {
    let pipeline = pass.pipeline.hal()?;
    let mut bind_buffers = Vec::new();
    for binding in pass.pipeline.metal_bindings() {
        let bound = pass.bind_groups.get(&binding.group)?;
        let entry = bound
            .group
            .entries()
            .iter()
            .find(|entry| entry.binding == binding.binding)?;
        let BindGroupResource::Buffer {
            buffer,
            offset,
            size,
            ..
        } = &entry.resource
        else {
            return None;
        };
        let dynamic_offset = dynamic_offset_for_binding(
            pass.pipeline.bind_group_layouts(),
            binding.group,
            binding.binding,
            &bound.dynamic_offsets,
        )?;
        let offset = offset.checked_add(dynamic_offset)?;
        bind_buffers.push(HalBoundBuffer {
            group: binding.group,
            binding: binding.binding,
            metal_index: binding.metal_index,
            buffer: buffer.hal()?,
            offset,
            size: *size,
        });
    }
    Some(HalCopy::ComputePass(HalComputePass {
        pipeline,
        bind_buffers,
        workgroups: pass.workgroups,
    }))
}

fn hal_render_pass_execution(pass: &RenderPassCommand) -> Option<HalCopy> {
    let (pipeline, bind_buffers, vertex_buffers, draw) =
        if let (Some(pipeline), Some(draw)) = (&pass.pipeline, pass.draw) {
            let bind_buffers = hal_bind_buffers(
                pipeline.bind_group_layouts(),
                pipeline.metal_bindings(),
                &pass.bind_groups,
            )?;
            let mut vertex_buffers = Vec::new();
            for binding in pipeline.vertex_buffer_bindings() {
                let bound = pass.vertex_buffers.get(&binding.slot)?;
                vertex_buffers.push(HalBoundBuffer {
                    group: 0,
                    binding: binding.slot,
                    metal_index: binding.metal_index,
                    buffer: bound.buffer.hal()?,
                    offset: bound.offset,
                    size: bound.size,
                });
            }
            (
                Some(pipeline.hal()?),
                bind_buffers,
                vertex_buffers,
                Some(HalDraw {
                    vertex_count: draw.vertex_count,
                    instance_count: draw.instance_count,
                    first_vertex: draw.first_vertex,
                    first_instance: draw.first_instance,
                }),
            )
        } else {
            (None, Vec::new(), Vec::new(), None)
        };
    Some(HalCopy::RenderPass(HalRenderPass {
        pipeline,
        color_target: HalRenderColorTarget {
            texture: pass.color_attachment.texture.hal()?,
            load_op: match pass.color_attachment.load_op {
                LoadOp::Load => HalRenderLoadOp::Load,
                LoadOp::Clear | LoadOp::Undefined => HalRenderLoadOp::Clear,
            },
            store: matches!(pass.color_attachment.store_op, StoreOp::Store),
            clear_color: [
                pass.color_attachment.clear_value.r,
                pass.color_attachment.clear_value.g,
                pass.color_attachment.clear_value.b,
                pass.color_attachment.clear_value.a,
            ],
        },
        bind_buffers,
        vertex_buffers,
        draw,
    }))
}

fn hal_bind_buffers(
    layouts: &[Arc<BindGroupLayout>],
    metal_bindings: &[MetalBufferBinding],
    bind_groups: &BTreeMap<u32, BoundBindGroup>,
) -> Option<Vec<HalBoundBuffer>> {
    let mut bind_buffers = Vec::new();
    for binding in metal_bindings {
        let bound = bind_groups.get(&binding.group)?;
        let entry = bound
            .group
            .entries()
            .iter()
            .find(|entry| entry.binding == binding.binding)?;
        let BindGroupResource::Buffer {
            buffer,
            offset,
            size,
            ..
        } = &entry.resource
        else {
            return None;
        };
        let dynamic_offset = dynamic_offset_for_binding(
            layouts,
            binding.group,
            binding.binding,
            &bound.dynamic_offsets,
        )?;
        let offset = offset.checked_add(dynamic_offset)?;
        bind_buffers.push(HalBoundBuffer {
            group: binding.group,
            binding: binding.binding,
            metal_index: binding.metal_index,
            buffer: buffer.hal()?,
            offset,
            size: *size,
        });
    }
    Some(bind_buffers)
}

fn dynamic_offset_for_binding(
    layouts: &[Arc<BindGroupLayout>],
    group: u32,
    binding: u32,
    dynamic_offsets: &[u32],
) -> Option<u64> {
    let layout = layouts.get(usize::try_from(group).ok()?)?;
    let mut dynamic_index = 0usize;
    for entry in layout.entries() {
        let is_dynamic = matches!(
            entry.kind,
            Some(BindingLayoutKind::Buffer {
                has_dynamic_offset: true,
                ..
            })
        );
        if entry.binding == binding {
            return if is_dynamic {
                dynamic_offsets.get(dynamic_index).copied().map(u64::from)
            } else {
                Some(0)
            };
        }
        if is_dynamic {
            dynamic_index = dynamic_index.checked_add(1)?;
        }
    }
    None
}

fn validate_sampler_descriptor(descriptor: &ResolvedSamplerDescriptor) -> Option<&'static str> {
    if !descriptor.lod_min_clamp.is_finite() {
        return Some("sampler lodMinClamp must be finite");
    }
    if !descriptor.lod_max_clamp.is_finite() {
        return Some("sampler lodMaxClamp must be finite");
    }
    if descriptor.max_anisotropy == 0 {
        return Some("sampler maxAnisotropy must be at least one");
    }
    if descriptor.max_anisotropy > 1
        && (descriptor.mag_filter != FilterMode::Linear
            || descriptor.min_filter != FilterMode::Linear
            || descriptor.mipmap_filter != MipmapFilterMode::Linear)
    {
        return Some("anisotropic samplers require all filters to be Linear");
    }
    None
}

fn max_texture_mips(size: Extent3d, dimension: TextureDimension) -> u32 {
    let mut max_extent = size.width;
    if matches!(dimension, TextureDimension::D2 | TextureDimension::D3) {
        max_extent = max_extent.max(size.height);
    }
    if dimension == TextureDimension::D3 {
        max_extent = max_extent.max(size.depth_or_array_layers);
    }

    let mut levels = 0;
    while max_extent > 0 {
        levels += 1;
        max_extent /= 2;
    }
    levels
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
    TimestampQuery,
    TextureFormatsTier1,
    TextureFormatsTier2,
    Other(u32),
}

#[must_use]
pub(crate) fn supported_features() -> FeatureSet {
    [
        Feature::CoreFeaturesAndLimits,
        Feature::Rg11b10UfloatRenderable,
        Feature::TimestampQuery,
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

const MAX_QUERY_COUNT: u32 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueryType {
    Occlusion,
    Timestamp,
    Unknown(u32),
}

#[derive(Debug, Clone)]
pub struct QuerySetDescriptor {
    pub label: String,
    pub kind: QueryType,
    pub count: u32,
}

#[derive(Debug, Clone)]
pub struct QuerySet {
    inner: Arc<QuerySetInner>,
}

#[derive(Debug)]
struct QuerySetInner {
    label: Mutex<String>,
    kind: QueryType,
    count: u32,
    state: Mutex<QuerySetState>,
}

#[derive(Debug)]
struct QuerySetState {
    is_error: bool,
    is_destroyed: bool,
}

impl QuerySet {
    fn new(descriptor: QuerySetDescriptor, is_error: bool) -> Self {
        Self {
            inner: Arc::new(QuerySetInner {
                label: Mutex::new(descriptor.label),
                kind: descriptor.kind,
                count: descriptor.count,
                state: Mutex::new(QuerySetState {
                    is_error,
                    is_destroyed: false,
                }),
            }),
        }
    }

    #[must_use]
    pub fn kind(&self) -> QueryType {
        self.inner.kind
    }

    #[must_use]
    pub fn count(&self) -> u32 {
        self.inner.count
    }

    pub fn set_label(&self, label: &str) {
        *self.inner.label.lock() = label.to_owned();
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.state.lock().is_error
    }

    #[must_use]
    pub(crate) fn is_destroyed(&self) -> bool {
        self.inner.state.lock().is_destroyed
    }

    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn destroy(&self) {
        self.inner.state.lock().is_destroyed = true;
    }
}

fn validate_query_set_descriptor(
    descriptor: &QuerySetDescriptor,
    features: &FeatureSet,
) -> Option<&'static str> {
    if descriptor.count == 0 {
        return Some("query set count must be greater than zero");
    }
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
        // Block 00: for the synthetic Noop adapter, supported limits equal
        // the WebGPU spec defaults, so comparisons against `self` collapse to
        // comparisons against `DEFAULT` intentionally.
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
pub struct CommandEncoder {
    inner: Arc<CommandEncoderInner>,
}

#[derive(Debug)]
struct CommandEncoderInner {
    state: Mutex<CommandEncoderState>,
}

#[derive(Debug)]
struct CommandEncoderState {
    lifecycle: CommandEncoderLifecycle,
    open_pass: Option<PassToken>,
    next_pass_id: u64,
    first_error: Option<String>,
    debug_group_depth: u32,
    referenced_buffers: Vec<Arc<Buffer>>,
    command_ops: Vec<CommandExecution>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommandEncoderLifecycle {
    Recording,
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassKind {
    Render,
    Compute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PassToken {
    kind: PassKind,
    id: u64,
}

#[derive(Debug, Clone)]
pub struct CommandBuffer {
    inner: Arc<CommandBufferInner>,
}

#[derive(Debug)]
struct CommandBufferInner {
    is_error: bool,
    referenced_buffers: Vec<Arc<Buffer>>,
    command_ops: Vec<CommandExecution>,
    submitted: Mutex<bool>,
}

#[derive(Debug, Clone)]
struct BufferCopyCommand {
    source: Arc<Buffer>,
    source_offset: u64,
    destination: Arc<Buffer>,
    destination_offset: u64,
    size: u64,
}

#[derive(Debug, Clone)]
enum TextureCopyCommand {
    BufferToTexture {
        source: TexelCopyBufferInfo,
        destination: TexelCopyTextureInfo,
        copy_size: Extent3d,
    },
    TextureToBuffer {
        source: TexelCopyTextureInfo,
        destination: TexelCopyBufferInfo,
        copy_size: Extent3d,
    },
    TextureToTexture {
        source: TexelCopyTextureInfo,
        destination: TexelCopyTextureInfo,
        copy_size: Extent3d,
    },
}

#[derive(Debug, Clone)]
enum CommandExecution {
    BufferCopy(BufferCopyCommand),
    TextureCopy(TextureCopyCommand),
    ComputePass(ComputePassCommand),
    RenderPass(RenderPassCommand),
}

#[derive(Debug, Clone)]
struct ComputePassCommand {
    pipeline: Arc<ComputePipeline>,
    bind_groups: BTreeMap<u32, BoundBindGroup>,
    workgroups: (u32, u32, u32),
}

#[derive(Debug, Clone)]
struct RenderPassCommand {
    pipeline: Option<Arc<RenderPipeline>>,
    color_attachment: RenderPassColorExecution,
    bind_groups: BTreeMap<u32, BoundBindGroup>,
    vertex_buffers: BTreeMap<u32, BoundVertexBuffer>,
    draw: Option<RenderDrawExecution>,
}

#[derive(Debug, Clone)]
struct RenderPassColorExecution {
    texture: Texture,
    load_op: LoadOp,
    store_op: StoreOp,
    clear_value: Color,
}

#[derive(Debug, Clone, Copy)]
struct RenderDrawExecution {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

#[derive(Debug, Clone)]
pub struct RenderPassEncoder {
    inner: Arc<PassEncoderInner>,
}

#[derive(Debug, Clone)]
pub struct ComputePassEncoder {
    inner: Arc<PassEncoderInner>,
}

#[derive(Debug, Clone)]
pub struct RenderBundleEncoder {
    inner: Arc<RenderBundleEncoderInner>,
}

#[derive(Debug, Clone)]
pub struct RenderBundle {
    inner: Arc<RenderBundleInner>,
}

#[derive(Debug)]
struct PassEncoderInner {
    parent: CommandEncoder,
    token: PassToken,
    state: Mutex<PassEncoderState>,
}

#[derive(Debug)]
struct PassEncoderState {
    ended: bool,
    debug_group_depth: u32,
    render_pipeline: Option<Arc<RenderPipeline>>,
    compute_pipeline: Option<Arc<ComputePipeline>>,
    bind_groups: BTreeMap<u32, BoundBindGroup>,
    vertex_buffers: BTreeMap<u32, BoundVertexBuffer>,
    index_buffer: Option<BoundIndexBuffer>,
    attachment_signature: Option<AttachmentSignature>,
    attachment_textures: Vec<Texture>,
    render_color_attachment: Option<RenderPassColorExecution>,
    render_pass_recorded: bool,
    occlusion_query_set: Option<QuerySet>,
    open_occlusion_query: Option<u32>,
    used_occlusion_queries: BTreeSet<u32>,
}

impl PassEncoderState {
    fn new(
        attachment_signature: Option<AttachmentSignature>,
        attachment_textures: Vec<Texture>,
        render_color_attachment: Option<RenderPassColorExecution>,
        occlusion_query_set: Option<QuerySet>,
    ) -> Self {
        Self {
            ended: false,
            debug_group_depth: 0,
            render_pipeline: None,
            compute_pipeline: None,
            bind_groups: BTreeMap::new(),
            vertex_buffers: BTreeMap::new(),
            index_buffer: None,
            attachment_signature,
            attachment_textures,
            render_color_attachment,
            render_pass_recorded: false,
            occlusion_query_set,
            open_occlusion_query: None,
            used_occlusion_queries: BTreeSet::new(),
        }
    }

    fn clear_render_state(&mut self) {
        self.render_pipeline = None;
        self.bind_groups.clear();
        self.vertex_buffers.clear();
        self.index_buffer = None;
    }
}

#[derive(Debug)]
struct RenderBundleEncoderInner {
    descriptor: RenderBundleEncoderDescriptor,
    state: Mutex<RenderBundleEncoderState>,
}

#[derive(Debug)]
struct RenderBundleEncoderState {
    lifecycle: RenderBundleEncoderLifecycle,
    first_error: Option<String>,
    pass_state: PassEncoderState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderBundleEncoderLifecycle {
    Recording,
    Errored,
    Finished,
}

#[derive(Debug)]
struct RenderBundleInner {
    is_error: bool,
    attachment_signature: AttachmentSignature,
}

#[derive(Debug, Clone)]
struct BoundBindGroup {
    group: Arc<BindGroup>,
    dynamic_offsets: Vec<u32>,
}

#[derive(Debug, Clone)]
struct BoundVertexBuffer {
    buffer: Arc<Buffer>,
    offset: u64,
    size: u64,
}

#[derive(Debug, Clone)]
struct BoundIndexBuffer {
    buffer: Arc<Buffer>,
    format: IndexFormat,
    offset: u64,
    size: u64,
}

impl CommandEncoder {
    fn new() -> Self {
        Self {
            inner: Arc::new(CommandEncoderInner {
                state: Mutex::new(CommandEncoderState {
                    lifecycle: CommandEncoderLifecycle::Recording,
                    open_pass: None,
                    next_pass_id: 0,
                    first_error: None,
                    debug_group_depth: 0,
                    referenced_buffers: Vec::new(),
                    command_ops: Vec::new(),
                }),
            }),
        }
    }

    fn new_error(message: impl Into<String>) -> Self {
        let encoder = Self::new();
        encoder.record_first_error(message);
        encoder
    }

    #[must_use]
    pub fn begin_render_pass(
        &self,
        descriptor: &RenderPassDescriptor,
    ) -> (RenderPassEncoder, Option<String>) {
        let (token, immediate_error) = self.begin_pass(PassKind::Render);
        let attachment_signature = render_pass_attachment_signature(descriptor).ok();
        if immediate_error.is_none() {
            if let Err(message) = validate_render_pass_descriptor(descriptor) {
                self.record_first_error(message);
            }
        }
        (
            RenderPassEncoder {
                inner: Arc::new(PassEncoderInner::new(
                    self.clone(),
                    token,
                    attachment_signature,
                    render_pass_attachment_textures(descriptor),
                    render_pass_color_execution(descriptor),
                    descriptor.occlusion_query_set.clone(),
                )),
            },
            immediate_error,
        )
    }

    #[must_use]
    pub fn begin_compute_pass(&self) -> (ComputePassEncoder, Option<String>) {
        let (token, immediate_error) = self.begin_pass(PassKind::Compute);
        (
            ComputePassEncoder {
                inner: Arc::new(PassEncoderInner::new(
                    self.clone(),
                    token,
                    None,
                    Vec::new(),
                    None,
                    None,
                )),
            },
            immediate_error,
        )
    }

    fn begin_pass(&self, kind: PassKind) -> (PassToken, Option<String>) {
        let mut state = self.inner.state.lock();
        let token = PassToken {
            kind,
            id: state.next_pass_id,
        };
        state.next_pass_id = state.next_pass_id.saturating_add(1);

        if state.lifecycle != CommandEncoderLifecycle::Recording {
            return (
                token,
                Some("command encoder cannot record after finish".to_owned()),
            );
        }

        if state.open_pass.is_some() {
            record_first_error_locked(
                &mut state,
                "command encoder cannot begin a pass while another pass is open",
            );
            return (token, None);
        }

        state.open_pass = Some(token);
        (token, None)
    }

    pub fn insert_debug_marker(&self) -> Option<String> {
        self.record_encoder_command()
    }

    pub fn record_validation_error(&self, message: impl Into<String>) -> Option<String> {
        self.record_buffer_command(Vec::new(), None, None, || Err(message.into()))
    }

    pub fn copy_buffer_to_buffer(
        &self,
        source: Arc<Buffer>,
        source_offset: u64,
        destination: Arc<Buffer>,
        destination_offset: u64,
        size: u64,
    ) -> Option<String> {
        let copy = BufferCopyCommand {
            source: Arc::clone(&source),
            source_offset,
            destination: Arc::clone(&destination),
            destination_offset,
            size,
        };
        self.record_buffer_command(
            vec![Arc::clone(&source), Arc::clone(&destination)],
            Some(copy),
            None,
            || {
                validate_copy_buffer_to_buffer(
                    &source,
                    source_offset,
                    &destination,
                    destination_offset,
                    size,
                )
            },
        )
    }

    pub fn clear_buffer(&self, buffer: Arc<Buffer>, offset: u64, size: u64) -> Option<String> {
        self.record_buffer_command(vec![Arc::clone(&buffer)], None, None, || {
            validate_clear_buffer(&buffer, offset, size)
        })
    }

    pub fn write_buffer(&self, buffer: Arc<Buffer>, offset: u64, size: u64) -> Option<String> {
        self.record_buffer_command(vec![Arc::clone(&buffer)], None, None, || {
            validate_encoder_write_buffer(&buffer, offset, size)
        })
    }

    pub fn write_timestamp(&self, query_set: Arc<QuerySet>, query_index: u32) -> Option<String> {
        self.record_buffer_command(Vec::new(), None, None, || {
            validate_timestamp_query_set(&query_set, "write timestamp")?;
            validate_query_index(&query_set, query_index, "write timestamp query index")
        })
    }

    pub fn resolve_query_set(
        &self,
        query_set: Arc<QuerySet>,
        first_query: u32,
        query_count: u32,
        destination: Arc<Buffer>,
        destination_offset: u64,
    ) -> Option<String> {
        self.record_buffer_command(vec![Arc::clone(&destination)], None, None, || {
            validate_resolve_query_set(
                &query_set,
                first_query,
                query_count,
                &destination,
                destination_offset,
            )
        })
    }

    pub fn copy_buffer_to_texture(
        &self,
        source: TexelCopyBufferInfo,
        destination: TexelCopyTextureInfo,
        copy_size: Extent3d,
    ) -> Option<String> {
        let copy = TextureCopyCommand::BufferToTexture {
            source: source.clone(),
            destination: destination.clone(),
            copy_size,
        };
        self.record_buffer_command(vec![Arc::clone(&source.buffer)], None, Some(copy), || {
            validate_buffer_texture_copy(
                source,
                BufferUsage::COPY_SRC,
                destination,
                TextureUsage::COPY_DST,
                copy_size,
                "copy buffer to texture",
            )
        })
    }

    pub fn copy_texture_to_buffer(
        &self,
        source: TexelCopyTextureInfo,
        destination: TexelCopyBufferInfo,
        copy_size: Extent3d,
    ) -> Option<String> {
        let copy = TextureCopyCommand::TextureToBuffer {
            source: source.clone(),
            destination: destination.clone(),
            copy_size,
        };
        self.record_buffer_command(
            vec![Arc::clone(&destination.buffer)],
            None,
            Some(copy),
            || {
                validate_buffer_texture_copy(
                    destination,
                    BufferUsage::COPY_DST,
                    source,
                    TextureUsage::COPY_SRC,
                    copy_size,
                    "copy texture to buffer",
                )
            },
        )
    }

    pub fn copy_texture_to_texture(
        &self,
        source: TexelCopyTextureInfo,
        destination: TexelCopyTextureInfo,
        copy_size: Extent3d,
    ) -> Option<String> {
        let copy = TextureCopyCommand::TextureToTexture {
            source: source.clone(),
            destination: destination.clone(),
            copy_size,
        };
        self.record_buffer_command(Vec::new(), None, Some(copy), || {
            validate_texture_to_texture_copy(source, destination, copy_size)
        })
    }

    pub fn push_debug_group(&self) -> Option<String> {
        let mut state = self.inner.state.lock();
        if state.lifecycle != CommandEncoderLifecycle::Recording {
            return Some("command encoder cannot record after finish".to_owned());
        }
        if state.open_pass.is_some() {
            record_first_error_locked(
                &mut state,
                "command encoder command cannot be recorded while a pass is open",
            );
            return None;
        }
        state.debug_group_depth = state.debug_group_depth.saturating_add(1);
        None
    }

    pub fn pop_debug_group(&self) -> Option<String> {
        let mut state = self.inner.state.lock();
        if state.lifecycle != CommandEncoderLifecycle::Recording {
            return Some("command encoder cannot record after finish".to_owned());
        }
        if state.open_pass.is_some() {
            record_first_error_locked(
                &mut state,
                "command encoder command cannot be recorded while a pass is open",
            );
            return None;
        }
        if state.debug_group_depth == 0 {
            record_first_error_locked(&mut state, "command encoder debug group stack is empty");
        } else {
            state.debug_group_depth -= 1;
        }
        None
    }

    pub(crate) fn record_command_guard(&self) -> Result<(), String> {
        let state = self.inner.state.lock();
        if state.lifecycle != CommandEncoderLifecycle::Recording {
            return Err("command encoder cannot record after finish".to_owned());
        }
        if state.open_pass.is_some() {
            return Err(
                "command encoder command cannot be recorded while a pass is open".to_owned(),
            );
        }
        Ok(())
    }

    fn record_encoder_command(&self) -> Option<String> {
        match self.record_command_guard() {
            Ok(()) => None,
            Err(message) => {
                let mut state = self.inner.state.lock();
                if state.lifecycle == CommandEncoderLifecycle::Recording {
                    record_first_error_locked(&mut state, message);
                    None
                } else {
                    Some(message)
                }
            }
        }
    }

    fn record_buffer_command<F>(
        &self,
        referenced_buffers: Vec<Arc<Buffer>>,
        buffer_copy: Option<BufferCopyCommand>,
        texture_copy: Option<TextureCopyCommand>,
        validate: F,
    ) -> Option<String>
    where
        F: FnOnce() -> Result<(), String>,
    {
        if let Err(message) = self.record_command_guard() {
            let mut state = self.inner.state.lock();
            if state.lifecycle == CommandEncoderLifecycle::Recording {
                record_first_error_locked(&mut state, message);
                return None;
            }
            return Some(message);
        }

        if let Err(message) = validate() {
            self.record_first_error(message);
        } else {
            let mut state = self.inner.state.lock();
            state.referenced_buffers.extend(referenced_buffers);
            if let Some(copy) = buffer_copy {
                state
                    .command_ops
                    .push(CommandExecution::BufferCopy(copy.clone()));
            }
            if let Some(copy) = texture_copy {
                state
                    .command_ops
                    .push(CommandExecution::TextureCopy(copy.clone()));
            }
        }
        None
    }

    #[must_use]
    pub fn finish(&self) -> (CommandBuffer, Option<String>) {
        let mut state = self.inner.state.lock();
        if state.lifecycle != CommandEncoderLifecycle::Recording {
            return (
                CommandBuffer::new(true, Vec::new(), Vec::new()),
                Some("command encoder cannot be finished more than once".to_owned()),
            );
        }
        state.lifecycle = CommandEncoderLifecycle::Finished;

        let finish_error = state
            .first_error
            .clone()
            .or_else(|| {
                state
                    .open_pass
                    .is_some()
                    .then(|| "command encoder cannot finish while a pass is open".to_owned())
            })
            .or_else(|| {
                (state.debug_group_depth != 0)
                    .then(|| "command encoder debug group stack is unbalanced".to_owned())
            });
        let referenced_buffers = if finish_error.is_some() {
            Vec::new()
        } else {
            std::mem::take(&mut state.referenced_buffers)
        };
        let command_ops = if finish_error.is_some() {
            Vec::new()
        } else {
            std::mem::take(&mut state.command_ops)
        };
        (
            CommandBuffer::new(finish_error.is_some(), referenced_buffers, command_ops),
            finish_error,
        )
    }

    fn end_pass(&self, token: PassToken) {
        let mut state = self.inner.state.lock();
        if state.open_pass == Some(token) {
            state.open_pass = None;
        }
    }

    fn record_first_error(&self, message: impl Into<String>) {
        let mut state = self.inner.state.lock();
        record_first_error_locked(&mut state, message);
    }

    fn record_referenced_buffer(&self, buffer: Arc<Buffer>) {
        self.inner.state.lock().referenced_buffers.push(buffer);
    }

    fn record_referenced_buffers(&self, buffers: Vec<Arc<Buffer>>) {
        self.inner.state.lock().referenced_buffers.extend(buffers);
    }

    fn record_compute_pass(&self, command: ComputePassCommand) {
        self.inner
            .state
            .lock()
            .command_ops
            .push(CommandExecution::ComputePass(command));
    }

    fn record_render_pass(&self, command: RenderPassCommand) {
        self.inner
            .state
            .lock()
            .command_ops
            .push(CommandExecution::RenderPass(command));
    }

    fn is_finished(&self) -> bool {
        self.inner.state.lock().lifecycle == CommandEncoderLifecycle::Finished
    }
}

fn validate_copy_buffer_to_buffer(
    source: &Buffer,
    source_offset: u64,
    destination: &Buffer,
    destination_offset: u64,
    size: u64,
) -> Result<(), String> {
    if source.is_error() || destination.is_error() {
        return Err("copy buffer command cannot use an error buffer".to_owned());
    }
    if source.is_destroyed() || destination.is_destroyed() {
        return Err("copy buffer command cannot use a destroyed buffer".to_owned());
    }
    if !source.usage().contains(BufferUsage::COPY_SRC) {
        return Err("copy source buffer must have CopySrc usage".to_owned());
    }
    if !destination.usage().contains(BufferUsage::COPY_DST) {
        return Err("copy destination buffer must have CopyDst usage".to_owned());
    }
    if !source_offset.is_multiple_of(4) {
        return Err("copy source offset must be 4-byte aligned".to_owned());
    }
    if !destination_offset.is_multiple_of(4) {
        return Err("copy destination offset must be 4-byte aligned".to_owned());
    }
    if !size.is_multiple_of(4) {
        return Err("copy size must be 4-byte aligned".to_owned());
    }
    validate_buffer_range(source_offset, size, source.size(), "copy source range")?;
    validate_buffer_range(
        destination_offset,
        size,
        destination.size(),
        "copy destination range",
    )?;
    if size > 0 && source.same(destination) {
        return Err("copy source and destination ranges must not use the same buffer".to_owned());
    }
    Ok(())
}

fn validate_clear_buffer(buffer: &Buffer, offset: u64, size: u64) -> Result<(), String> {
    if buffer.is_error() {
        return Err("clear buffer command cannot use an error buffer".to_owned());
    }
    if buffer.is_destroyed() {
        return Err("clear buffer command cannot use a destroyed buffer".to_owned());
    }
    if !buffer.usage().contains(BufferUsage::COPY_DST) {
        return Err("clear buffer requires CopyDst usage".to_owned());
    }
    if offset > buffer.size() {
        return Err("clear buffer offset exceeds buffer size".to_owned());
    }
    let resolved_size = if size == u64::MAX {
        buffer.size() - offset
    } else {
        size
    };
    if !offset.is_multiple_of(4) {
        return Err("clear buffer offset must be 4-byte aligned".to_owned());
    }
    if !resolved_size.is_multiple_of(4) {
        return Err("clear buffer size must be 4-byte aligned".to_owned());
    }
    validate_buffer_range(offset, resolved_size, buffer.size(), "clear buffer range")
}

fn validate_encoder_write_buffer(buffer: &Buffer, offset: u64, size: u64) -> Result<(), String> {
    if buffer.is_error() {
        return Err("command encoder write buffer cannot use an error buffer".to_owned());
    }
    if buffer.is_destroyed() {
        return Err("command encoder write buffer cannot use a destroyed buffer".to_owned());
    }
    if !buffer.usage().contains(BufferUsage::COPY_DST) {
        return Err("command encoder write buffer requires CopyDst usage".to_owned());
    }
    if !offset.is_multiple_of(4) {
        return Err("command encoder write buffer offset must be 4-byte aligned".to_owned());
    }
    if !size.is_multiple_of(4) {
        return Err("command encoder write buffer size must be 4-byte aligned".to_owned());
    }
    validate_buffer_range(
        offset,
        size,
        buffer.size(),
        "command encoder write buffer range",
    )
}

fn validate_render_pass_descriptor(descriptor: &RenderPassDescriptor) -> Result<(), String> {
    render_pass_attachment_signature(descriptor)?;
    if let Some(query_set) = &descriptor.occlusion_query_set {
        validate_occlusion_query_set(query_set, "render pass occlusion query set")?;
    }
    if let Some(timestamp_writes) = &descriptor.timestamp_writes {
        validate_render_pass_timestamp_writes(timestamp_writes)?;
    }
    Ok(())
}

fn validate_render_pass_timestamp_writes(
    timestamp_writes: &RenderPassTimestampWrites,
) -> Result<(), String> {
    validate_timestamp_query_set(
        &timestamp_writes.query_set,
        "render pass timestamp writes query set",
    )?;
    if timestamp_writes.beginning_index.is_none() && timestamp_writes.end_index.is_none() {
        return Err("render pass timestamp writes requires at least one query index".to_owned());
    }
    if let Some(index) = timestamp_writes.beginning_index {
        validate_query_index(
            &timestamp_writes.query_set,
            index,
            "render pass beginning timestamp query index",
        )?;
    }
    if let Some(index) = timestamp_writes.end_index {
        validate_query_index(
            &timestamp_writes.query_set,
            index,
            "render pass end timestamp query index",
        )?;
    }
    if timestamp_writes.beginning_index == timestamp_writes.end_index {
        return Err("render pass timestamp write indices must be distinct".to_owned());
    }
    Ok(())
}

fn validate_occlusion_query_set(query_set: &QuerySet, usage: &str) -> Result<(), String> {
    validate_query_set_alive(query_set, usage)?;
    if query_set.kind() != QueryType::Occlusion {
        return Err(format!("{usage} requires an occlusion query set"));
    }
    Ok(())
}

fn validate_timestamp_query_set(query_set: &QuerySet, usage: &str) -> Result<(), String> {
    validate_query_set_alive(query_set, usage)?;
    if query_set.kind() != QueryType::Timestamp {
        return Err(format!("{usage} requires a timestamp query set"));
    }
    Ok(())
}

fn validate_query_set_alive(query_set: &QuerySet, usage: &str) -> Result<(), String> {
    if query_set.is_error() {
        return Err(format!("{usage} cannot use an error query set"));
    }
    if query_set.is_destroyed() {
        return Err(format!("{usage} cannot use a destroyed query set"));
    }
    Ok(())
}

fn validate_query_index(query_set: &QuerySet, index: u32, name: &str) -> Result<(), String> {
    if index >= query_set.count() {
        return Err(format!("{name} exceeds query set count"));
    }
    Ok(())
}

fn validate_resolve_query_set(
    query_set: &QuerySet,
    first_query: u32,
    query_count: u32,
    destination: &Buffer,
    destination_offset: u64,
) -> Result<(), String> {
    validate_query_set_alive(query_set, "resolve query set")?;
    if query_count == 0 {
        return Err("resolve query count must be greater than zero".to_owned());
    }
    let end_query = first_query
        .checked_add(query_count)
        .ok_or_else(|| "resolve query range overflows".to_owned())?;
    if end_query > query_set.count() {
        return Err("resolve query range exceeds query set count".to_owned());
    }
    if destination.is_error() {
        return Err("resolve query set cannot use an error destination buffer".to_owned());
    }
    if destination.is_destroyed() {
        return Err("resolve query set cannot use a destroyed destination buffer".to_owned());
    }
    if !destination.usage().contains(BufferUsage::QUERY_RESOLVE) {
        return Err("resolve query set destination requires QueryResolve usage".to_owned());
    }
    if !destination_offset.is_multiple_of(256) {
        return Err("resolve query set destination offset must be 256-byte aligned".to_owned());
    }
    let byte_count = u64::from(query_count)
        .checked_mul(8)
        .ok_or_else(|| "resolve query byte count overflows".to_owned())?;
    validate_buffer_range(
        destination_offset,
        byte_count,
        destination.size(),
        "resolve query destination range",
    )
}

fn render_pass_attachment_signature(
    descriptor: &RenderPassDescriptor,
) -> Result<AttachmentSignature, String> {
    if descriptor.color_attachments.len() > descriptor.max_color_attachments as usize {
        return Err("render pass colorAttachmentCount exceeds the device limit".to_owned());
    }

    let mut has_attachment = false;
    let mut render_extent = None;
    let mut sample_count = None;
    let mut color_formats = Vec::with_capacity(descriptor.color_attachments.len());

    for attachment in &descriptor.color_attachments {
        if let Some(attachment) = attachment {
            has_attachment = true;
            validate_color_attachment(attachment)?;
            validate_render_attachment_common(
                &attachment.view,
                &mut render_extent,
                &mut sample_count,
                "render pass color attachment",
            )?;
            if let Some(resolve_target) = &attachment.resolve_target {
                validate_resolve_target(&attachment.view, resolve_target)?;
            }
            color_formats.push(Some(attachment.view.format()));
        } else {
            color_formats.push(None);
        }
    }

    let mut depth_stencil_format = None;
    if let Some(attachment) = &descriptor.depth_stencil_attachment {
        has_attachment = true;
        validate_depth_stencil_attachment(attachment)?;
        depth_stencil_format = Some(attachment.view.format());
        validate_render_attachment_common(
            &attachment.view,
            &mut render_extent,
            &mut sample_count,
            "render pass depth-stencil attachment",
        )?;
    }

    if !has_attachment {
        return Err("render pass requires at least one attachment".to_owned());
    }
    Ok(AttachmentSignature {
        color_formats,
        depth_stencil_format,
        sample_count: sample_count.unwrap_or(1),
    })
}

fn render_pass_attachment_textures(descriptor: &RenderPassDescriptor) -> Vec<Texture> {
    let mut textures = Vec::new();
    for attachment in descriptor.color_attachments.iter().flatten() {
        textures.push(attachment.view.texture());
        if let Some(resolve_target) = &attachment.resolve_target {
            textures.push(resolve_target.texture());
        }
    }
    if let Some(attachment) = &descriptor.depth_stencil_attachment {
        textures.push(attachment.view.texture());
    }
    textures
}

fn render_pass_color_execution(
    descriptor: &RenderPassDescriptor,
) -> Option<RenderPassColorExecution> {
    descriptor
        .color_attachments
        .iter()
        .flatten()
        .next()
        .map(|attachment| RenderPassColorExecution {
            texture: attachment.view.texture(),
            load_op: attachment.load_op,
            store_op: attachment.store_op,
            clear_value: attachment.clear_value,
        })
}

fn validate_color_attachment(attachment: &RenderPassColorAttachment) -> Result<(), String> {
    let texture = attachment.view.texture();
    let Some(format_caps) = attachment.view.format().caps() else {
        return Err("render pass color attachment format must be supported".to_owned());
    };
    if !texture.usage().contains(TextureUsage::RENDER_ATTACHMENT) {
        return Err("render pass color attachment requires RenderAttachment usage".to_owned());
    }
    if !format_caps.aspects.color || !format_caps.renderable {
        return Err("render pass color attachment format must be color-renderable".to_owned());
    }
    if attachment.load_op == LoadOp::Undefined {
        return Err("render pass color attachment loadOp must be set".to_owned());
    }
    if attachment.store_op == StoreOp::Undefined {
        return Err("render pass color attachment storeOp must be set".to_owned());
    }
    if attachment.load_op == LoadOp::Clear
        && ![
            attachment.clear_value.r,
            attachment.clear_value.g,
            attachment.clear_value.b,
            attachment.clear_value.a,
        ]
        .into_iter()
        .all(f64::is_finite)
    {
        return Err("render pass color clearValue components must be finite".to_owned());
    }
    Ok(())
}

fn validate_depth_stencil_attachment(
    attachment: &RenderPassDepthStencilAttachment,
) -> Result<(), String> {
    let texture = attachment.view.texture();
    let Some(format_caps) = attachment.view.format().caps() else {
        return Err("render pass depth-stencil attachment format must be supported".to_owned());
    };
    if !texture.usage().contains(TextureUsage::RENDER_ATTACHMENT) {
        return Err(
            "render pass depth-stencil attachment requires RenderAttachment usage".to_owned(),
        );
    }
    if !format_caps.aspects.depth && !format_caps.aspects.stencil {
        return Err(
            "render pass depth-stencil attachment format must have depth or stencil aspect"
                .to_owned(),
        );
    }
    if format_caps.aspects.depth {
        if attachment.depth_load_op == LoadOp::Undefined {
            return Err("render pass depth loadOp must be set".to_owned());
        }
        if attachment.depth_store_op == StoreOp::Undefined {
            return Err("render pass depth storeOp must be set".to_owned());
        }
        if attachment.depth_load_op == LoadOp::Clear
            && (!attachment.depth_clear_value.is_finite()
                || !(0.0..=1.0).contains(&attachment.depth_clear_value))
        {
            return Err("render pass depth clear value must be finite and in [0, 1]".to_owned());
        }
    }
    if format_caps.aspects.stencil {
        if attachment.stencil_load_op == LoadOp::Undefined {
            return Err("render pass stencil loadOp must be set".to_owned());
        }
        if attachment.stencil_store_op == StoreOp::Undefined {
            return Err("render pass stencil storeOp must be set".to_owned());
        }
    }
    Ok(())
}

fn validate_render_attachment_common(
    view: &TextureView,
    render_extent: &mut Option<(u32, u32)>,
    sample_count: &mut Option<u32>,
    label: &str,
) -> Result<(), String> {
    if view.is_error() {
        return Err(format!("{label} view must not be an error view"));
    }
    if view.array_layer_count() != 1 {
        return Err(format!("{label} view arrayLayerCount must be one"));
    }
    let extent = view.render_extent();
    let size = (extent.width, extent.height);
    if let Some(expected) = *render_extent {
        if expected != size {
            return Err("render pass attachments must have matching sizes".to_owned());
        }
    } else {
        *render_extent = Some(size);
    }

    let view_sample_count = view.texture().sample_count();
    if let Some(expected) = *sample_count {
        if expected != view_sample_count {
            return Err("render pass attachments must have matching sample counts".to_owned());
        }
    } else {
        *sample_count = Some(view_sample_count);
    }
    Ok(())
}

fn validate_resolve_target(
    color_view: &TextureView,
    resolve_target: &TextureView,
) -> Result<(), String> {
    let color_texture = color_view.texture();
    let resolve_texture = resolve_target.texture();
    if color_texture.sample_count() <= 1 {
        return Err(
            "render pass resolveTarget requires a multisampled color attachment".to_owned(),
        );
    }
    if resolve_target.is_error() {
        return Err("render pass resolveTarget view must not be an error view".to_owned());
    }
    if !resolve_texture
        .usage()
        .contains(TextureUsage::RENDER_ATTACHMENT)
    {
        return Err("render pass resolveTarget requires RenderAttachment usage".to_owned());
    }
    if resolve_texture.sample_count() != 1 {
        return Err("render pass resolveTarget sampleCount must be one".to_owned());
    }
    if color_view.format() != resolve_target.format() {
        return Err("render pass resolveTarget format must match the color attachment".to_owned());
    }
    if resolve_target.array_layer_count() != 1 {
        return Err("render pass resolveTarget view arrayLayerCount must be one".to_owned());
    }
    if color_view.render_extent() != resolve_target.render_extent() {
        return Err("render pass resolveTarget size must match the color attachment".to_owned());
    }
    Ok(())
}

fn validate_buffer_texture_copy(
    buffer_copy: TexelCopyBufferInfo,
    required_buffer_usage: BufferUsage,
    texture_copy: TexelCopyTextureInfo,
    required_texture_usage: TextureUsage,
    copy_size: Extent3d,
    label: &str,
) -> Result<(), String> {
    let buffer = buffer_copy.buffer;
    let texture = texture_copy.texture;
    if buffer.is_error() || texture.is_error() {
        return Err(format!("{label} cannot use an error resource"));
    }
    if buffer.is_destroyed() || texture.is_destroyed() {
        return Err(format!("{label} cannot use a destroyed resource"));
    }
    if !buffer.usage().contains(required_buffer_usage) {
        return Err(format!("{label} buffer has invalid usage"));
    }
    if !texture.usage().contains(required_texture_usage) {
        return Err(format!("{label} texture has invalid usage"));
    }
    if texture.sample_count() != 1 {
        return Err(format!("{label} texture sampleCount must be one"));
    }

    let format_caps = validate_texture_copy_subresource(
        &texture,
        texture_copy.mip_level,
        texture_copy.origin,
        copy_size,
        texture_copy.aspect,
        label,
        true,
    )?;
    if !buffer_copy.layout.offset.is_multiple_of(4) {
        return Err(format!("{label} buffer offset must be 4-byte aligned"));
    }
    let required_bytes = validate_texel_copy_layout(
        format_caps,
        texture_copy.aspect,
        copy_size,
        buffer_copy.layout,
        label,
        true,
    )?;
    validate_buffer_range(
        buffer_copy.layout.offset,
        required_bytes,
        buffer.size(),
        label,
    )
}

fn validate_texture_to_texture_copy(
    source_copy: TexelCopyTextureInfo,
    destination_copy: TexelCopyTextureInfo,
    copy_size: Extent3d,
) -> Result<(), String> {
    let source = source_copy.texture;
    let destination = destination_copy.texture;
    if source.is_error() || destination.is_error() {
        return Err("copy texture to texture cannot use an error texture".to_owned());
    }
    if source.is_destroyed() || destination.is_destroyed() {
        return Err("copy texture to texture cannot use a destroyed texture".to_owned());
    }
    if !source.usage().contains(TextureUsage::COPY_SRC) {
        return Err("copy texture source must have CopySrc usage".to_owned());
    }
    if !destination.usage().contains(TextureUsage::COPY_DST) {
        return Err("copy texture destination must have CopyDst usage".to_owned());
    }
    if source_copy.aspect != destination_copy.aspect {
        return Err("copy texture aspects must match".to_owned());
    }
    if !texture_formats_copy_compatible(source.format(), destination.format()) {
        return Err("copy texture formats are not copy-compatible".to_owned());
    }
    if source.sample_count() != destination.sample_count() {
        return Err("copy texture sample counts must match".to_owned());
    }

    let source_caps = validate_texture_copy_subresource(
        &source,
        source_copy.mip_level,
        source_copy.origin,
        copy_size,
        source_copy.aspect,
        "copy texture source",
        false,
    )?;
    validate_texture_copy_subresource(
        &destination,
        destination_copy.mip_level,
        destination_copy.origin,
        copy_size,
        destination_copy.aspect,
        "copy texture destination",
        false,
    )?;

    if (source_caps.aspects.depth || source_caps.aspects.stencil)
        && source_copy.aspect != TextureAspect::All
    {
        return Err("copy texture to texture depth/stencil copies require All aspect".to_owned());
    }
    if source.sample_count() > 1
        && (!origin_is_zero(source_copy.origin)
            || !origin_is_zero(destination_copy.origin)
            || copy_size != source.subresource_size(source_copy.mip_level)
            || copy_size != destination.subresource_size(destination_copy.mip_level))
    {
        return Err("copy texture multisampled copies must cover the full subresource".to_owned());
    }
    if source.same(&destination) {
        validate_same_texture_copy(
            &source,
            source_copy.mip_level,
            source_copy.origin,
            destination_copy.mip_level,
            destination_copy.origin,
            copy_size,
        )?;
    }

    Ok(())
}

fn validate_texture_copy_subresource(
    texture: &Texture,
    mip_level: u32,
    origin: Origin3d,
    copy_size: Extent3d,
    aspect: TextureAspect,
    label: &str,
    require_2d_single_layer: bool,
) -> Result<FormatCaps, String> {
    if mip_level >= texture.mip_level_count() {
        return Err(format!("{label} mipLevel is out of range"));
    }

    let Some(format_caps) = texture.format().caps() else {
        return Err(format!("{label} format must not be Undefined"));
    };
    validate_copy_aspect(format_caps, aspect, label)?;

    let subresource = texture.subresource_size(mip_level);
    if origin
        .x
        .checked_add(copy_size.width)
        .is_none_or(|end| end > subresource.width)
        || origin
            .y
            .checked_add(copy_size.height)
            .is_none_or(|end| end > subresource.height)
        || origin
            .z
            .checked_add(copy_size.depth_or_array_layers)
            .is_none_or(|end| end > subresource.depth_or_array_layers)
    {
        return Err(format!("{label} range exceeds the texture subresource"));
    }
    if require_2d_single_layer
        && texture.dimension() == TextureDimension::D2
        && copy_size.depth_or_array_layers != 1
    {
        return Err(format!(
            "{label} 2D copies require depthOrArrayLayers to be one"
        ));
    }
    if (format_caps.aspects.depth || format_caps.aspects.stencil)
        && (texture.dimension() != TextureDimension::D2 || copy_size.depth_or_array_layers != 1)
    {
        return Err(format!(
            "{label} depth/stencil copies require a single 2D layer"
        ));
    }

    Ok(format_caps)
}

fn validate_copy_aspect(
    format_caps: FormatCaps,
    aspect: TextureAspect,
    label: &str,
) -> Result<(), String> {
    match aspect {
        TextureAspect::All => Ok(()),
        TextureAspect::DepthOnly if format_caps.aspects.depth => Ok(()),
        TextureAspect::StencilOnly if format_caps.aspects.stencil => Ok(()),
        TextureAspect::DepthOnly => {
            Err(format!("{label} DepthOnly aspect requires a depth format"))
        }
        TextureAspect::StencilOnly => Err(format!(
            "{label} StencilOnly aspect requires a stencil format"
        )),
    }
}

fn texture_formats_copy_compatible(source: TextureFormat, destination: TextureFormat) -> bool {
    source == destination || source.srgb_pair() == Some(destination)
}

fn origin_is_zero(origin: Origin3d) -> bool {
    origin.x == 0 && origin.y == 0 && origin.z == 0
}

fn validate_same_texture_copy(
    texture: &Texture,
    source_mip_level: u32,
    source_origin: Origin3d,
    destination_mip_level: u32,
    destination_origin: Origin3d,
    copy_size: Extent3d,
) -> Result<(), String> {
    if copy_size.width == 0 || copy_size.height == 0 || copy_size.depth_or_array_layers == 0 {
        return Ok(());
    }
    if source_mip_level != destination_mip_level {
        return Ok(());
    }
    if texture.dimension() == TextureDimension::D3 {
        return Err(
            "copy texture to texture cannot copy within the same 3D texture mip".to_owned(),
        );
    }

    let source_end = source_origin
        .z
        .saturating_add(copy_size.depth_or_array_layers);
    let destination_end = destination_origin
        .z
        .saturating_add(copy_size.depth_or_array_layers);
    if source_origin.z < destination_end && destination_origin.z < source_end {
        return Err(
            "copy texture to texture same-texture array layers must not overlap".to_owned(),
        );
    }
    Ok(())
}

fn validate_buffer_range(
    offset: u64,
    size: u64,
    buffer_size: u64,
    label: &str,
) -> Result<(), String> {
    let Some(end) = offset.checked_add(size) else {
        return Err(format!("{label} overflows"));
    };
    if offset > buffer_size || end > buffer_size {
        return Err(format!("{label} exceeds buffer size"));
    }
    Ok(())
}

fn record_first_error_locked(state: &mut CommandEncoderState, message: impl Into<String>) {
    if state.first_error.is_none() {
        state.first_error = Some(message.into());
    }
}

fn record_first_error_option(first_error: &mut Option<String>, message: impl Into<String>) {
    if first_error.is_none() {
        *first_error = Some(message.into());
    }
}

impl CommandBuffer {
    fn new(
        is_error: bool,
        referenced_buffers: Vec<Arc<Buffer>>,
        command_ops: Vec<CommandExecution>,
    ) -> Self {
        Self {
            inner: Arc::new(CommandBufferInner {
                is_error,
                referenced_buffers,
                command_ops,
                submitted: Mutex::new(false),
            }),
        }
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    fn referenced_buffers(&self) -> &[Arc<Buffer>] {
        &self.inner.referenced_buffers
    }

    fn command_ops(&self) -> &[CommandExecution] {
        &self.inner.command_ops
    }

    fn mark_submitted(&self) -> Result<(), String> {
        let mut submitted = self.inner.submitted.lock();
        if *submitted {
            Err("command buffer cannot be submitted more than once".to_owned())
        } else {
            *submitted = true;
            Ok(())
        }
    }

    fn is_submitted(&self) -> bool {
        *self.inner.submitted.lock()
    }

    fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl PassEncoderInner {
    fn new(
        parent: CommandEncoder,
        token: PassToken,
        attachment_signature: Option<AttachmentSignature>,
        attachment_textures: Vec<Texture>,
        render_color_attachment: Option<RenderPassColorExecution>,
        occlusion_query_set: Option<QuerySet>,
    ) -> Self {
        Self {
            parent,
            token,
            state: Mutex::new(PassEncoderState::new(
                attachment_signature,
                attachment_textures,
                render_color_attachment,
                occlusion_query_set,
            )),
        }
    }

    fn end(&self) -> Option<String> {
        let mut state = self.state.lock();
        if state.ended {
            let message = "pass encoder cannot be ended more than once".to_owned();
            self.parent.record_first_error(message.clone());
            return Some(message);
        }
        if self.parent.is_finished() {
            let message = "pass encoder cannot be used after parent encoder finish".to_owned();
            self.parent.record_first_error(message.clone());
            return Some(message);
        }
        state.ended = true;
        let unbalanced_debug_groups = state.debug_group_depth != 0;
        let open_occlusion_query = state.open_occlusion_query.is_some();
        let render_pass_command = if !state.render_pass_recorded {
            state
                .render_color_attachment
                .clone()
                .map(|color_attachment| {
                    state.render_pass_recorded = true;
                    RenderPassCommand {
                        pipeline: state.render_pipeline.clone(),
                        color_attachment,
                        bind_groups: state.bind_groups.clone(),
                        vertex_buffers: state.vertex_buffers.clone(),
                        draw: None,
                    }
                })
        } else {
            None
        };
        drop(state);

        if let Some(command) = render_pass_command {
            self.parent.record_render_pass(command);
        }
        self.parent.end_pass(self.token);
        if unbalanced_debug_groups {
            let message = "pass encoder debug group stack is unbalanced".to_owned();
            self.parent.record_first_error(message);
            None
        } else if open_occlusion_query {
            self.parent
                .record_first_error("render pass occlusion query is still open");
            None
        } else {
            None
        }
    }

    fn insert_debug_marker(&self) -> Option<String> {
        self.pass_command_guard().err()
    }

    fn push_debug_group(&self) -> Option<String> {
        if let Err(message) = self.pass_command_guard() {
            return Some(message);
        }
        let mut state = self.state.lock();
        state.debug_group_depth = state.debug_group_depth.saturating_add(1);
        None
    }

    fn pop_debug_group(&self) -> Option<String> {
        if let Err(message) = self.pass_command_guard() {
            return Some(message);
        }
        let mut state = self.state.lock();
        if state.debug_group_depth == 0 {
            let message = "pass encoder debug group stack is empty".to_owned();
            self.parent.record_first_error(message);
            None
        } else {
            state.debug_group_depth -= 1;
            None
        }
    }

    fn record_pass_command<F>(&self, command: F) -> Option<String>
    where
        F: FnOnce(&mut PassEncoderState) -> Result<(), String>,
    {
        if let Err(message) = self.pass_command_guard() {
            return Some(message);
        }
        let mut state = self.state.lock();
        if let Err(message) = command(&mut state) {
            self.parent.record_first_error(message);
        }
        None
    }

    fn pass_command_guard(&self) -> Result<(), String> {
        if self.parent.is_finished() {
            let message = "pass encoder cannot be used after parent encoder finish".to_owned();
            self.parent.record_first_error(message.clone());
            return Err(message);
        }
        if self.state.lock().ended {
            let message = "pass encoder cannot be used after end".to_owned();
            self.parent.record_first_error(message.clone());
            return Err(message);
        }
        Ok(())
    }
}

impl RenderPassEncoder {
    pub fn end(&self) -> Option<String> {
        self.inner.end()
    }

    pub fn insert_debug_marker(&self) -> Option<String> {
        self.inner.insert_debug_marker()
    }

    pub fn push_debug_group(&self) -> Option<String> {
        self.inner.push_debug_group()
    }

    pub fn pop_debug_group(&self) -> Option<String> {
        self.inner.pop_debug_group()
    }

    pub fn set_pipeline(&self, pipeline: Arc<RenderPipeline>) -> Option<String> {
        self.inner.record_pass_command(|state| {
            state.render_pipeline = Some(pipeline);
            Ok(())
        })
    }

    pub fn record_validation_error(&self, message: impl Into<String>) -> Option<String> {
        self.inner.record_pass_command(|_| Err(message.into()))
    }

    pub fn set_bind_group(
        &self,
        index: u32,
        group: Option<Arc<BindGroup>>,
        dynamic_offsets: Vec<u32>,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            if let Some(group) = group {
                self.inner
                    .parent
                    .record_referenced_buffers(bind_group_buffer_resources(&group));
                state.bind_groups.insert(
                    index,
                    BoundBindGroup {
                        group,
                        dynamic_offsets,
                    },
                );
            } else {
                state.bind_groups.remove(&index);
            }
            Ok(())
        })
    }

    pub fn set_vertex_buffer(
        &self,
        slot: u32,
        buffer: Option<Arc<Buffer>>,
        offset: u64,
        size: u64,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_vertex_buffer_slot(slot, limits)?;
            if let Some(buffer) = buffer {
                let size = validate_set_vertex_buffer(&buffer, offset, size)?;
                self.inner
                    .parent
                    .record_referenced_buffer(Arc::clone(&buffer));
                state.vertex_buffers.insert(
                    slot,
                    BoundVertexBuffer {
                        buffer,
                        offset,
                        size,
                    },
                );
            } else {
                validate_clear_vertex_buffer(offset, size)?;
                state.vertex_buffers.remove(&slot);
            }
            Ok(())
        })
    }

    pub fn set_index_buffer(
        &self,
        buffer: Arc<Buffer>,
        format: Option<IndexFormat>,
        offset: u64,
        size: u64,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            let format = format.ok_or_else(|| "render pass index format is invalid".to_owned())?;
            let size = validate_set_index_buffer(&buffer, format, offset, size)?;
            self.inner
                .parent
                .record_referenced_buffer(Arc::clone(&buffer));
            state.index_buffer = Some(BoundIndexBuffer {
                buffer,
                format,
                offset,
                size,
            });
            Ok(())
        })
    }

    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_render_draw_state(
                state,
                RenderDrawKind::Direct {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                },
                limits,
            )?;
            let pipeline = state
                .render_pipeline
                .as_ref()
                .ok_or_else(|| "render pass requires a render pipeline".to_owned())?;
            let color_attachment = state
                .render_color_attachment
                .clone()
                .ok_or_else(|| "render pass requires a color attachment".to_owned())?;
            self.inner.parent.record_render_pass(RenderPassCommand {
                pipeline: Some(Arc::clone(pipeline)),
                color_attachment,
                bind_groups: state.bind_groups.clone(),
                vertex_buffers: state.vertex_buffers.clone(),
                draw: Some(RenderDrawExecution {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                }),
            });
            state.render_pass_recorded = true;
            Ok(())
        })
    }

    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        _base_vertex: i32,
        first_instance: u32,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_render_draw_state(
                state,
                RenderDrawKind::IndexedDirect {
                    index_count,
                    instance_count,
                    first_index,
                    first_instance,
                },
                limits,
            )
        })
    }

    pub fn draw_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::Indirect, limits)?;
            validate_indirect_buffer(&indirect_buffer, indirect_offset, 16, "draw indirect")?;
            self.inner.parent.record_referenced_buffer(indirect_buffer);
            Ok(())
        })
    }

    pub fn draw_indexed_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::IndexedIndirect, limits)?;
            validate_indirect_buffer(
                &indirect_buffer,
                indirect_offset,
                20,
                "draw indexed indirect",
            )?;
            self.inner.parent.record_referenced_buffer(indirect_buffer);
            Ok(())
        })
    }

    pub fn set_viewport(
        &self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Option<String> {
        self.inner
            .record_pass_command(|_| validate_viewport(x, y, width, height, min_depth, max_depth))
    }

    pub fn set_scissor_rect(&self, x: u32, y: u32, width: u32, height: u32) -> Option<String> {
        self.inner.record_pass_command(|_| {
            x.checked_add(width)
                .ok_or_else(|| "render pass scissor rectangle width overflows".to_owned())?;
            y.checked_add(height)
                .ok_or_else(|| "render pass scissor rectangle height overflows".to_owned())?;
            Ok(())
        })
    }

    pub fn set_blend_constant(&self, color: Color) -> Option<String> {
        self.inner.record_pass_command(|_| {
            if [color.r, color.g, color.b, color.a]
                .into_iter()
                .all(f64::is_finite)
            {
                Ok(())
            } else {
                Err("render pass blend constant components must be finite".to_owned())
            }
        })
    }

    pub fn set_stencil_reference(&self, _reference: u32) -> Option<String> {
        self.inner.record_pass_command(|_| Ok(()))
    }

    pub fn execute_bundles(&self, bundles: &[Arc<RenderBundle>]) -> Option<String> {
        self.inner.record_pass_command(|state| {
            let pass_signature = state
                .attachment_signature
                .as_ref()
                .ok_or_else(|| "render pass has no attachment signature".to_owned())?;
            for bundle in bundles {
                if bundle.is_error() {
                    return Err("render pass cannot execute an error render bundle".to_owned());
                }
                if bundle.attachment_signature() != pass_signature {
                    return Err(
                        "render bundle attachment signature is incompatible with the render pass"
                            .to_owned(),
                    );
                }
            }
            state.clear_render_state();
            Ok(())
        })
    }

    pub fn begin_occlusion_query(&self, query_index: u32) -> Option<String> {
        self.inner.record_pass_command(|state| {
            let query_set = state
                .occlusion_query_set
                .as_ref()
                .ok_or_else(|| "render pass has no occlusion query set".to_owned())?;
            validate_occlusion_query_set(query_set, "render pass occlusion query")?;
            validate_query_index(query_set, query_index, "occlusion query index")?;
            if state.open_occlusion_query.is_some() {
                return Err("render pass occlusion query is already open".to_owned());
            }
            if !state.used_occlusion_queries.insert(query_index) {
                return Err("render pass occlusion query index was already used".to_owned());
            }
            state.open_occlusion_query = Some(query_index);
            Ok(())
        })
    }

    pub fn end_occlusion_query(&self) -> Option<String> {
        self.inner.record_pass_command(|state| {
            if state.open_occlusion_query.take().is_none() {
                return Err("render pass has no open occlusion query".to_owned());
            }
            Ok(())
        })
    }
}

impl ComputePassEncoder {
    pub fn end(&self) -> Option<String> {
        self.inner.end()
    }

    pub fn insert_debug_marker(&self) -> Option<String> {
        self.inner.insert_debug_marker()
    }

    pub fn push_debug_group(&self) -> Option<String> {
        self.inner.push_debug_group()
    }

    pub fn pop_debug_group(&self) -> Option<String> {
        self.inner.pop_debug_group()
    }

    pub fn set_pipeline(&self, pipeline: Arc<ComputePipeline>) -> Option<String> {
        self.inner.record_pass_command(|state| {
            state.compute_pipeline = Some(pipeline);
            Ok(())
        })
    }

    pub fn record_validation_error(&self, message: impl Into<String>) -> Option<String> {
        self.inner.record_pass_command(|_| Err(message.into()))
    }

    pub fn set_bind_group(
        &self,
        index: u32,
        group: Option<Arc<BindGroup>>,
        dynamic_offsets: Vec<u32>,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            if let Some(group) = group {
                self.inner
                    .parent
                    .record_referenced_buffers(bind_group_buffer_resources(&group));
                state.bind_groups.insert(
                    index,
                    BoundBindGroup {
                        group,
                        dynamic_offsets,
                    },
                );
            } else {
                state.bind_groups.remove(&index);
            }
            Ok(())
        })
    }

    pub fn dispatch_workgroups(&self, x: u32, y: u32, z: u32, limits: Limits) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_compute_dispatch_state(state, limits)?;
            if x > limits.max_compute_workgroups_per_dimension
                || y > limits.max_compute_workgroups_per_dimension
                || z > limits.max_compute_workgroups_per_dimension
            {
                return Err("compute dispatch workgroup count exceeds the device limit".to_owned());
            }
            let pipeline = state
                .compute_pipeline
                .as_ref()
                .ok_or_else(|| "compute dispatch requires a compute pipeline".to_owned())?;
            self.inner.parent.record_compute_pass(ComputePassCommand {
                pipeline: Arc::clone(pipeline),
                bind_groups: state.bind_groups.clone(),
                workgroups: (x, y, z),
            });
            Ok(())
        })
    }

    pub fn dispatch_workgroups_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.inner.record_pass_command(|state| {
            validate_compute_dispatch_state(state, limits)?;
            validate_indirect_buffer(
                &indirect_buffer,
                indirect_offset,
                12,
                "dispatch workgroups indirect",
            )?;
            self.inner.parent.record_referenced_buffer(indirect_buffer);
            Ok(())
        })
    }
}

impl RenderBundleEncoder {
    #[must_use]
    pub fn new(
        descriptor: RenderBundleEncoderDescriptor,
        limits: Limits,
    ) -> (Self, Option<String>) {
        let descriptor_error = validate_render_bundle_encoder_descriptor(&descriptor, limits).err();
        let attachment_signature = descriptor.attachment_signature();
        (
            Self {
                inner: Arc::new(RenderBundleEncoderInner {
                    descriptor,
                    state: Mutex::new(RenderBundleEncoderState {
                        lifecycle: if descriptor_error.is_some() {
                            RenderBundleEncoderLifecycle::Errored
                        } else {
                            RenderBundleEncoderLifecycle::Recording
                        },
                        first_error: None,
                        pass_state: PassEncoderState::new(
                            Some(attachment_signature),
                            Vec::new(),
                            None,
                            None,
                        ),
                    }),
                }),
            },
            descriptor_error,
        )
    }

    pub fn finish(&self) -> (RenderBundle, Option<String>) {
        let mut state = self.inner.state.lock();
        match state.lifecycle {
            RenderBundleEncoderLifecycle::Errored => {
                state.lifecycle = RenderBundleEncoderLifecycle::Finished;
                return (
                    RenderBundle::new(self.inner.descriptor.attachment_signature(), true),
                    None,
                );
            }
            RenderBundleEncoderLifecycle::Finished => {
                return (
                    RenderBundle::new(self.inner.descriptor.attachment_signature(), true),
                    Some("render bundle encoder cannot be finished more than once".to_owned()),
                );
            }
            RenderBundleEncoderLifecycle::Recording => {}
        }
        state.lifecycle = RenderBundleEncoderLifecycle::Finished;
        if state.first_error.is_none() && state.pass_state.debug_group_depth != 0 {
            record_first_error_option(
                &mut state.first_error,
                "render bundle debug group stack is unbalanced".to_owned(),
            );
        }
        let error = state.first_error.clone();
        (
            RenderBundle::new(
                self.inner.descriptor.attachment_signature(),
                error.is_some(),
            ),
            error,
        )
    }

    pub fn insert_debug_marker(&self) -> Option<String> {
        self.record_bundle_command(|_| Ok(()))
    }

    pub fn push_debug_group(&self) -> Option<String> {
        self.record_bundle_command(|state| {
            state.debug_group_depth = state.debug_group_depth.saturating_add(1);
            Ok(())
        })
    }

    pub fn pop_debug_group(&self) -> Option<String> {
        self.record_bundle_command(|state| {
            if state.debug_group_depth == 0 {
                Err("render bundle debug group stack is empty".to_owned())
            } else {
                state.debug_group_depth -= 1;
                Ok(())
            }
        })
    }

    pub fn set_pipeline(&self, pipeline: Arc<RenderPipeline>) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_bundle_pipeline(&self.inner.descriptor, &pipeline)?;
            state.render_pipeline = Some(pipeline);
            Ok(())
        })
    }

    pub fn set_bind_group(
        &self,
        index: u32,
        group: Option<Arc<BindGroup>>,
        dynamic_offsets: Vec<u32>,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            if let Some(group) = group {
                state.bind_groups.insert(
                    index,
                    BoundBindGroup {
                        group,
                        dynamic_offsets,
                    },
                );
            } else {
                state.bind_groups.remove(&index);
            }
            Ok(())
        })
    }

    pub fn set_vertex_buffer(
        &self,
        slot: u32,
        buffer: Option<Arc<Buffer>>,
        offset: u64,
        size: u64,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_vertex_buffer_slot(slot, limits)?;
            if let Some(buffer) = buffer {
                let size = validate_set_vertex_buffer(&buffer, offset, size)?;
                state.vertex_buffers.insert(
                    slot,
                    BoundVertexBuffer {
                        buffer,
                        offset,
                        size,
                    },
                );
            } else {
                validate_clear_vertex_buffer(offset, size)?;
                state.vertex_buffers.remove(&slot);
            }
            Ok(())
        })
    }

    pub fn set_index_buffer(
        &self,
        buffer: Arc<Buffer>,
        format: Option<IndexFormat>,
        offset: u64,
        size: u64,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            let format = format.ok_or_else(|| "render pass index format is invalid".to_owned())?;
            let size = validate_set_index_buffer(&buffer, format, offset, size)?;
            state.index_buffer = Some(BoundIndexBuffer {
                buffer,
                format,
                offset,
                size,
            });
            Ok(())
        })
    }

    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_draw_state(
                state,
                RenderDrawKind::Direct {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                },
                limits,
            )
        })
    }

    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        _base_vertex: i32,
        first_instance: u32,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_draw_state(
                state,
                RenderDrawKind::IndexedDirect {
                    index_count,
                    instance_count,
                    first_index,
                    first_instance,
                },
                limits,
            )
        })
    }

    pub fn draw_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::Indirect, limits)?;
            validate_indirect_buffer(&indirect_buffer, indirect_offset, 16, "draw indirect")
        })
    }

    pub fn draw_indexed_indirect(
        &self,
        indirect_buffer: Arc<Buffer>,
        indirect_offset: u64,
        limits: Limits,
    ) -> Option<String> {
        self.record_bundle_command(|state| {
            validate_render_draw_state(state, RenderDrawKind::IndexedIndirect, limits)?;
            validate_indirect_buffer(
                &indirect_buffer,
                indirect_offset,
                20,
                "draw indexed indirect",
            )
        })
    }

    fn record_bundle_command<F>(&self, command: F) -> Option<String>
    where
        F: FnOnce(&mut PassEncoderState) -> Result<(), String>,
    {
        let mut state = self.inner.state.lock();
        match state.lifecycle {
            RenderBundleEncoderLifecycle::Recording => {}
            RenderBundleEncoderLifecycle::Errored => return None,
            RenderBundleEncoderLifecycle::Finished => {
                return Some("render bundle encoder cannot record after finish".to_owned());
            }
        }
        if let Err(message) = command(&mut state.pass_state) {
            record_first_error_option(&mut state.first_error, message);
        }
        None
    }
}

impl RenderBundle {
    fn new(attachment_signature: AttachmentSignature, is_error: bool) -> Self {
        Self {
            inner: Arc::new(RenderBundleInner {
                is_error,
                attachment_signature,
            }),
        }
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    fn attachment_signature(&self) -> &AttachmentSignature {
        &self.inner.attachment_signature
    }
}

#[derive(Debug, Clone, Copy)]
enum RenderDrawKind {
    Direct {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    IndexedDirect {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        first_instance: u32,
    },
    Indirect,
    IndexedIndirect,
}

fn validate_render_draw_state(
    state: &PassEncoderState,
    kind: RenderDrawKind,
    limits: Limits,
) -> Result<(), String> {
    let pipeline = validate_render_draw_base_state(state, limits, kind.is_indexed())?;
    validate_usage_scope(
        pipeline.bind_group_layouts(),
        &state.bind_groups,
        Some(&state.attachment_textures),
    )?;
    validate_strip_index_format(pipeline, state, kind.is_indexed())?;
    match kind {
        RenderDrawKind::Direct {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        } => validate_vertex_buffer_oob(
            pipeline,
            state,
            Some((first_vertex, vertex_count)),
            first_instance,
            instance_count,
        ),
        RenderDrawKind::IndexedDirect {
            index_count,
            instance_count,
            first_index,
            first_instance,
        } => {
            validate_index_buffer_oob(state, first_index, index_count)?;
            validate_vertex_buffer_oob(pipeline, state, None, first_instance, instance_count)
        }
        RenderDrawKind::Indirect | RenderDrawKind::IndexedIndirect => Ok(()),
    }
}

impl RenderDrawKind {
    fn is_indexed(self) -> bool {
        matches!(
            self,
            RenderDrawKind::IndexedDirect { .. } | RenderDrawKind::IndexedIndirect
        )
    }
}

impl RenderBundleEncoderDescriptor {
    fn attachment_signature(&self) -> AttachmentSignature {
        AttachmentSignature {
            color_formats: self.color_formats.clone(),
            depth_stencil_format: self.depth_stencil_format,
            sample_count: self.sample_count,
        }
    }
}

fn validate_render_bundle_encoder_descriptor(
    descriptor: &RenderBundleEncoderDescriptor,
    _limits: Limits,
) -> Result<(), String> {
    if descriptor.color_formats.len() > descriptor.max_color_attachments as usize {
        return Err("render bundle colorFormatCount exceeds the device limit".to_owned());
    }
    if descriptor.sample_count != 1 && descriptor.sample_count != 4 {
        return Err("render bundle sampleCount must be 1 or 4".to_owned());
    }

    let mut has_attachment = descriptor.depth_stencil_format.is_some();
    for color_format in descriptor.color_formats.iter().flatten().copied() {
        has_attachment = true;
        let Some(caps) = color_format.caps() else {
            return Err("render bundle color format must be defined".to_owned());
        };
        if !caps.aspects.color || !caps.renderable {
            return Err("render bundle color format must be color-renderable".to_owned());
        }
    }
    if let Some(depth_format) = descriptor.depth_stencil_format {
        let Some(caps) = depth_format.caps() else {
            return Err("render bundle depthStencilFormat must be defined".to_owned());
        };
        if !caps.aspects.depth && !caps.aspects.stencil {
            return Err(
                "render bundle depthStencilFormat must have depth or stencil aspect".to_owned(),
            );
        }
    }
    if !has_attachment {
        return Err("render bundle requires at least one attachment format".to_owned());
    }
    Ok(())
}

fn validate_render_bundle_pipeline(
    descriptor: &RenderBundleEncoderDescriptor,
    pipeline: &RenderPipeline,
) -> Result<(), String> {
    if pipeline.is_error() {
        return Err("render bundle requires a valid render pipeline".to_owned());
    }
    if pipeline.attachment_signature() != descriptor.attachment_signature() {
        return Err("render bundle pipeline attachment signature is incompatible".to_owned());
    }
    Ok(())
}

fn validate_render_draw_base_state(
    state: &PassEncoderState,
    limits: Limits,
    indexed: bool,
) -> Result<&Arc<RenderPipeline>, String> {
    let Some(pipeline) = &state.render_pipeline else {
        return Err("render pass draw requires a render pipeline".to_owned());
    };
    if pipeline.is_error() {
        return Err("render pass draw requires a valid render pipeline".to_owned());
    }
    validate_pipeline_bind_groups(pipeline.bind_group_layouts(), &state.bind_groups, limits)?;
    for slot in 0..pipeline.required_vertex_buffer_count() {
        let slot = u32::try_from(slot)
            .map_err(|_| "render pipeline vertex buffer slot is too large".to_owned())?;
        if !state.vertex_buffers.contains_key(&slot) {
            return Err(
                "render pass draw requires all declared vertex buffers to be set".to_owned(),
            );
        }
    }
    if indexed && state.index_buffer.is_none() {
        return Err("render pass indexed draw requires an index buffer".to_owned());
    }
    Ok(pipeline)
}

fn validate_set_index_buffer(
    buffer: &Buffer,
    format: IndexFormat,
    offset: u64,
    size: u64,
) -> Result<u64, String> {
    if buffer.is_error() {
        return Err("render pass index buffer must not be an error buffer".to_owned());
    }
    if buffer.is_destroyed() {
        return Err("render pass index buffer must not be destroyed".to_owned());
    }
    if !buffer.usage().contains(BufferUsage::INDEX) {
        return Err("render pass index buffer requires Index usage".to_owned());
    }
    let format_size = index_format_size(format);
    if !offset.is_multiple_of(format_size) {
        return Err("render pass index buffer offset is not aligned".to_owned());
    }
    resolve_buffer_binding_size(
        offset,
        size,
        buffer.size(),
        "render pass index buffer range",
    )
}

fn validate_set_vertex_buffer(buffer: &Buffer, offset: u64, size: u64) -> Result<u64, String> {
    if buffer.is_error() {
        return Err("render pass vertex buffer must not be an error buffer".to_owned());
    }
    if buffer.is_destroyed() {
        return Err("render pass vertex buffer must not be destroyed".to_owned());
    }
    if !buffer.usage().contains(BufferUsage::VERTEX) {
        return Err("render pass vertex buffer requires Vertex usage".to_owned());
    }
    if !offset.is_multiple_of(4) {
        return Err("render pass vertex buffer offset must be 4-byte aligned".to_owned());
    }
    resolve_buffer_binding_size(
        offset,
        size,
        buffer.size(),
        "render pass vertex buffer range",
    )
}

fn validate_vertex_buffer_slot(slot: u32, limits: Limits) -> Result<(), String> {
    if slot >= limits.max_vertex_buffers {
        return Err("render pass vertex buffer slot exceeds the device limit".to_owned());
    }
    Ok(())
}

fn validate_clear_vertex_buffer(offset: u64, size: u64) -> Result<(), String> {
    if offset != 0 || size != 0 {
        return Err("render pass null vertex buffer requires zero offset and size".to_owned());
    }
    Ok(())
}

fn resolve_buffer_binding_size(
    offset: u64,
    size: u64,
    buffer_size: u64,
    label: &str,
) -> Result<u64, String> {
    if offset > buffer_size {
        return Err(format!("{label} exceeds buffer size"));
    }
    let resolved_size = if size == u64::MAX {
        buffer_size - offset
    } else {
        size
    };
    validate_buffer_range(offset, resolved_size, buffer_size, label)?;
    Ok(resolved_size)
}

fn validate_strip_index_format(
    pipeline: &RenderPipeline,
    state: &PassEncoderState,
    indexed: bool,
) -> Result<(), String> {
    if !indexed {
        return Ok(());
    }
    let primitive = pipeline.primitive_state();
    if !matches!(
        primitive.topology,
        PrimitiveTopology::LineStrip | PrimitiveTopology::TriangleStrip
    ) {
        return Ok(());
    }
    let Some(strip_format) = primitive.strip_index_format else {
        return Err("render pass strip indexed draw requires pipeline stripIndexFormat".to_owned());
    };
    let index_buffer = state
        .index_buffer
        .as_ref()
        .ok_or_else(|| "render pass indexed draw requires an index buffer".to_owned())?;
    if index_buffer.format != strip_format {
        return Err(
            "render pass index buffer format must match pipeline stripIndexFormat".to_owned(),
        );
    }
    Ok(())
}

fn validate_vertex_buffer_oob(
    pipeline: &RenderPipeline,
    state: &PassEncoderState,
    vertex_draw: Option<(u32, u32)>,
    first_instance: u32,
    instance_count: u32,
) -> Result<(), String> {
    for (slot, layout) in pipeline.vertex_buffer_layouts().iter().enumerate() {
        if layout.array_stride == 0 {
            continue;
        }
        let stride_count = match layout.step_mode {
            VertexStepMode::Vertex => {
                let Some((first_vertex, vertex_count)) = vertex_draw else {
                    continue;
                };
                first_vertex
                    .checked_add(vertex_count)
                    .ok_or_else(|| "render pass draw vertex count overflows".to_owned())?
            }
            VertexStepMode::Instance => first_instance
                .checked_add(instance_count)
                .ok_or_else(|| "render pass draw instance count overflows".to_owned())?,
        };
        let required_size = layout
            .array_stride
            .checked_mul(u64::from(stride_count))
            .ok_or_else(|| "render pass vertex buffer required size overflows".to_owned())?;
        let slot = u32::try_from(slot)
            .map_err(|_| "render pipeline vertex buffer slot is too large".to_owned())?;
        let bound = state.vertex_buffers.get(&slot).ok_or_else(|| {
            "render pass draw requires all declared vertex buffers to be set".to_owned()
        })?;
        let required_end = bound
            .offset
            .checked_add(required_size)
            .ok_or_else(|| "render pass vertex buffer required range overflows".to_owned())?;
        let bound_end = bound
            .offset
            .checked_add(bound.size)
            .ok_or_else(|| "render pass vertex buffer bound range overflows".to_owned())?;
        if required_end > bound_end || required_end > bound.buffer.size() {
            return Err("render pass draw vertex buffer range exceeds the bound buffer".to_owned());
        }
    }
    Ok(())
}

fn validate_index_buffer_oob(
    state: &PassEncoderState,
    first_index: u32,
    index_count: u32,
) -> Result<(), String> {
    let index_buffer = state
        .index_buffer
        .as_ref()
        .ok_or_else(|| "render pass indexed draw requires an index buffer".to_owned())?;
    let required_indices = first_index
        .checked_add(index_count)
        .ok_or_else(|| "render pass indexed draw index count overflows".to_owned())?;
    let required_size = u64::from(required_indices)
        .checked_mul(index_format_size(index_buffer.format))
        .ok_or_else(|| "render pass indexed draw index buffer size overflows".to_owned())?;
    let required_end = index_buffer
        .offset
        .checked_add(required_size)
        .ok_or_else(|| "render pass indexed draw index buffer range overflows".to_owned())?;
    let bound_end = index_buffer
        .offset
        .checked_add(index_buffer.size)
        .ok_or_else(|| "render pass indexed draw index buffer bound range overflows".to_owned())?;
    if required_end > bound_end || required_end > index_buffer.buffer.size() {
        return Err(
            "render pass indexed draw index buffer range exceeds the bound buffer".to_owned(),
        );
    }
    Ok(())
}

fn validate_indirect_buffer(
    buffer: &Buffer,
    indirect_offset: u64,
    args_size: u64,
    label: &str,
) -> Result<(), String> {
    if buffer.is_error() {
        return Err(format!("{label} buffer must not be an error buffer"));
    }
    if buffer.is_destroyed() {
        return Err(format!("{label} buffer must not be destroyed"));
    }
    if !buffer.usage().contains(BufferUsage::INDIRECT) {
        return Err(format!("{label} buffer requires Indirect usage"));
    }
    if !indirect_offset.is_multiple_of(4) {
        return Err(format!("{label} offset must be 4-byte aligned"));
    }
    validate_buffer_range(indirect_offset, args_size, buffer.size(), label)
}

const fn index_format_size(format: IndexFormat) -> u64 {
    match format {
        IndexFormat::Uint16 => 2,
        IndexFormat::Uint32 => 4,
    }
}

fn validate_compute_dispatch_state(state: &PassEncoderState, limits: Limits) -> Result<(), String> {
    let Some(pipeline) = &state.compute_pipeline else {
        return Err("compute dispatch requires a compute pipeline".to_owned());
    };
    if pipeline.is_error() {
        return Err("compute dispatch requires a valid compute pipeline".to_owned());
    }
    validate_pipeline_bind_groups(pipeline.bind_group_layouts(), &state.bind_groups, limits)?;
    validate_usage_scope(pipeline.bind_group_layouts(), &state.bind_groups, None)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResourceAccess {
    Read,
    Write,
}

#[derive(Debug)]
struct BufferScopeUse {
    buffer: Arc<Buffer>,
    offset: u64,
    size: u64,
    access: ResourceAccess,
}

#[derive(Debug)]
struct TextureScopeUse {
    texture: Texture,
    access: ResourceAccess,
}

fn validate_usage_scope(
    required_layouts: &[Arc<BindGroupLayout>],
    bound_groups: &BTreeMap<u32, BoundBindGroup>,
    attachment_textures: Option<&[Texture]>,
) -> Result<(), String> {
    let mut buffer_uses = Vec::new();
    let mut texture_uses = Vec::new();

    for (index, layout) in required_layouts.iter().enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| "pipeline bind group index is too large".to_owned())?;
        let Some(bound) = bound_groups.get(&index) else {
            continue;
        };
        collect_bind_group_usage(layout, bound, &mut buffer_uses, &mut texture_uses)?;
    }

    validate_buffer_usage_scope(&buffer_uses)?;
    validate_texture_usage_scope(&texture_uses)?;
    if let Some(attachment_textures) = attachment_textures {
        for texture_use in &texture_uses {
            if attachment_textures
                .iter()
                .any(|attachment| attachment.same(&texture_use.texture))
            {
                return Err(
                    "render pass attachment texture cannot be used through a bind group".to_owned(),
                );
            }
        }
    }
    Ok(())
}

fn collect_bind_group_usage(
    layout: &BindGroupLayout,
    bound: &BoundBindGroup,
    buffer_uses: &mut Vec<BufferScopeUse>,
    texture_uses: &mut Vec<TextureScopeUse>,
) -> Result<(), String> {
    let layout_entries = layout
        .entries()
        .iter()
        .map(|entry| (entry.binding, entry))
        .collect::<BTreeMap<_, _>>();
    let dynamic_entries = layout
        .entries()
        .iter()
        .filter(|entry| {
            matches!(
                entry.kind,
                Some(BindingLayoutKind::Buffer {
                    has_dynamic_offset: true,
                    ..
                })
            )
        })
        .map(|entry| entry.binding)
        .collect::<Vec<_>>();

    for entry in bound.group.entries() {
        let Some(layout_entry) = layout_entries.get(&entry.binding).copied() else {
            continue;
        };
        let Some(kind) = layout_entry.kind else {
            continue;
        };
        let access = match kind {
            BindingLayoutKind::Buffer {
                ty: BufferBindingType::Uniform | BufferBindingType::ReadOnlyStorage,
                ..
            }
            | BindingLayoutKind::Texture { .. }
            | BindingLayoutKind::StorageTexture {
                access: StorageTextureAccess::ReadOnly,
                ..
            } => Some(ResourceAccess::Read),
            BindingLayoutKind::Buffer {
                ty: BufferBindingType::Storage,
                ..
            }
            | BindingLayoutKind::StorageTexture {
                access: StorageTextureAccess::WriteOnly | StorageTextureAccess::ReadWrite,
                ..
            } => Some(ResourceAccess::Write),
            BindingLayoutKind::Sampler { .. } => None,
        };
        let Some(access) = access else {
            continue;
        };

        match (&entry.resource, kind) {
            (
                BindGroupResource::Buffer {
                    buffer,
                    offset,
                    size,
                    ..
                },
                BindingLayoutKind::Buffer { .. },
            ) => {
                let dynamic_offset = dynamic_entries
                    .iter()
                    .position(|binding| *binding == entry.binding)
                    .and_then(|dynamic_index| bound.dynamic_offsets.get(dynamic_index))
                    .copied()
                    .unwrap_or(0);
                let offset = offset
                    .checked_add(u64::from(dynamic_offset))
                    .ok_or_else(|| "usage scope buffer offset overflows".to_owned())?;
                let size = if *size == u64::MAX {
                    buffer.size().saturating_sub(offset)
                } else {
                    size.saturating_sub(u64::from(dynamic_offset))
                };
                buffer_uses.push(BufferScopeUse {
                    buffer: Arc::clone(buffer),
                    offset,
                    size,
                    access,
                });
            }
            (
                BindGroupResource::TextureView { texture_view, .. },
                BindingLayoutKind::Texture { .. } | BindingLayoutKind::StorageTexture { .. },
            ) => texture_uses.push(TextureScopeUse {
                texture: texture_view.texture(),
                access,
            }),
            _ => {}
        }
    }
    Ok(())
}

fn validate_buffer_usage_scope(buffer_uses: &[BufferScopeUse]) -> Result<(), String> {
    for (index, current) in buffer_uses.iter().enumerate() {
        for previous in &buffer_uses[..index] {
            if !current.buffer.same(&previous.buffer) || !buffer_ranges_overlap(current, previous) {
                continue;
            }
            if current.access == ResourceAccess::Write || previous.access == ResourceAccess::Write {
                return Err(
                    "usage scope cannot read and write or write the same buffer range twice"
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

fn validate_texture_usage_scope(texture_uses: &[TextureScopeUse]) -> Result<(), String> {
    for (index, current) in texture_uses.iter().enumerate() {
        for previous in &texture_uses[..index] {
            if !current.texture.same(&previous.texture) {
                continue;
            }
            if current.access == ResourceAccess::Write || previous.access == ResourceAccess::Write {
                return Err(
                    "usage scope cannot read and write or write the same texture twice".to_owned(),
                );
            }
        }
    }
    Ok(())
}

fn buffer_ranges_overlap(a: &BufferScopeUse, b: &BufferScopeUse) -> bool {
    let a_end = a.offset.saturating_add(a.size);
    let b_end = b.offset.saturating_add(b.size);
    a.offset < b_end && b.offset < a_end
}

fn bind_group_buffer_resources(group: &BindGroup) -> Vec<Arc<Buffer>> {
    group
        .entries()
        .iter()
        .filter_map(|entry| match &entry.resource {
            BindGroupResource::Buffer { buffer, .. } => Some(Arc::clone(buffer)),
            _ => None,
        })
        .collect()
}

fn validate_pipeline_bind_groups(
    required_layouts: &[Arc<BindGroupLayout>],
    bound_groups: &BTreeMap<u32, BoundBindGroup>,
    limits: Limits,
) -> Result<(), String> {
    for (index, required_layout) in required_layouts.iter().enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| "pipeline bind group index is too large".to_owned())?;
        let Some(bound) = bound_groups.get(&index) else {
            return Err("pipeline requires a missing bind group".to_owned());
        };
        if bound.group.is_error() {
            return Err("pipeline cannot use an error bind group".to_owned());
        }
        if !bind_group_layouts_compatible(required_layout, bound.group.layout()) {
            return Err("pipeline bind group layout is incompatible".to_owned());
        }
        validate_dynamic_offsets(
            required_layout,
            &bound.group,
            &bound.dynamic_offsets,
            limits,
        )?;
    }
    Ok(())
}

fn bind_group_layouts_compatible(
    required: &Arc<BindGroupLayout>,
    actual: &Arc<BindGroupLayout>,
) -> bool {
    if required.is_default() || actual.is_default() {
        return required.same(actual);
    }
    required.entries() == actual.entries()
}

fn validate_dynamic_offsets(
    layout: &BindGroupLayout,
    group: &BindGroup,
    dynamic_offsets: &[u32],
    limits: Limits,
) -> Result<(), String> {
    let dynamic_entries = layout
        .entries()
        .iter()
        .filter(|entry| {
            matches!(
                entry.kind,
                Some(BindingLayoutKind::Buffer {
                    has_dynamic_offset: true,
                    ..
                })
            )
        })
        .collect::<Vec<_>>();
    if dynamic_offsets.len() != dynamic_entries.len() {
        return Err("bind group dynamic offset count is invalid".to_owned());
    }

    for (layout_entry, dynamic_offset) in dynamic_entries.into_iter().zip(dynamic_offsets.iter()) {
        let Some(group_entry) = group
            .entries()
            .iter()
            .find(|entry| entry.binding == layout_entry.binding)
        else {
            return Err("bind group dynamic offset binding is missing".to_owned());
        };
        let Some(BindingLayoutKind::Buffer { ty, .. }) = layout_entry.kind else {
            continue;
        };
        let alignment = match ty {
            BufferBindingType::Uniform => limits.min_uniform_buffer_offset_alignment,
            BufferBindingType::Storage | BufferBindingType::ReadOnlyStorage => {
                limits.min_storage_buffer_offset_alignment
            }
        };
        if *dynamic_offset % alignment != 0 {
            return Err("bind group dynamic offset is not aligned".to_owned());
        }
        let BindGroupResource::Buffer {
            buffer,
            offset,
            size,
            ..
        } = &group_entry.resource
        else {
            return Err("bind group dynamic offset requires a buffer binding".to_owned());
        };
        let dynamic_offset = u64::from(*dynamic_offset);
        let base = offset
            .checked_add(dynamic_offset)
            .ok_or_else(|| "bind group dynamic offset range overflows".to_owned())?;
        if base > buffer.size() {
            return Err("bind group dynamic offset exceeds buffer size".to_owned());
        }
        if *size != u64::MAX && dynamic_offset > *size {
            return Err("bind group dynamic offset exceeds binding size".to_owned());
        }
    }

    Ok(())
}

fn validate_viewport(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
) -> Result<(), String> {
    if ![x, y, width, height, min_depth, max_depth]
        .into_iter()
        .all(f32::is_finite)
    {
        return Err("render pass viewport values must be finite".to_owned());
    }
    if width < 0.0 || height < 0.0 {
        return Err("render pass viewport width and height must be non-negative".to_owned());
    }
    if !(0.0..=1.0).contains(&min_depth)
        || !(0.0..=1.0).contains(&max_depth)
        || min_depth > max_depth
    {
        return Err("render pass viewport depth range is invalid".to_owned());
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Queue {
    inner: Arc<QueueInner>,
}

#[derive(Debug)]
struct QueueInner {
    hal: HalQueue,
    label: Mutex<String>,
}

impl Queue {
    #[must_use]
    pub fn from_hal(hal: HalQueue, label: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(QueueInner {
                hal,
                label: Mutex::new(label.into()),
            }),
        }
    }

    #[must_use]
    pub fn hal(&self) -> &HalQueue {
        &self.inner.hal
    }

    pub fn set_label(&self, label: &str) {
        *self.inner.label.lock() = label.to_owned();
    }

    #[must_use]
    pub fn label(&self) -> String {
        self.inner.label.lock().clone()
    }

    pub fn write_buffer(&self, buffer: &Buffer, offset: u64, data: &[u8]) -> Option<DeviceError> {
        buffer.write_from_queue(offset, data)
    }

    pub fn submit(&self, command_buffers: &[Arc<CommandBuffer>]) -> Option<DeviceError> {
        for (index, command_buffer) in command_buffers.iter().enumerate() {
            if command_buffer.is_error() {
                return Some(DeviceError::validation(
                    "queue submit cannot use an error command buffer",
                ));
            }
            if command_buffer.is_submitted() {
                return Some(DeviceError::validation(
                    "command buffer cannot be submitted more than once",
                ));
            }
            if command_buffers[..index]
                .iter()
                .any(|previous| previous.same(command_buffer))
            {
                return Some(DeviceError::validation(
                    "command buffer cannot be submitted more than once",
                ));
            }
            for buffer in command_buffer.referenced_buffers() {
                if buffer.map_state() != BufferMapState::Unmapped {
                    return Some(DeviceError::validation(
                        "queue submit cannot use a mapped buffer",
                    ));
                }
                if buffer.is_destroyed() {
                    return Some(DeviceError::validation(
                        "queue submit cannot use a destroyed buffer",
                    ));
                }
            }
        }
        for command_buffer in command_buffers {
            if let Err(message) = command_buffer.mark_submitted() {
                return Some(DeviceError::validation(message));
            }
        }
        if command_buffers.is_empty() {
            if let Err(error) = self.inner.hal.submit_empty() {
                return Some(DeviceError::internal(error.to_string()));
            }
            return None;
        }
        let mut copies = Vec::new();
        for command_buffer in command_buffers {
            for op in command_buffer.command_ops() {
                if let Some(copy) = hal_command_execution(op) {
                    copies.push(copy);
                }
            }
        }
        if let Err(error) = self.inner.hal.submit_copies(&copies) {
            return Some(DeviceError::internal(error.to_string()));
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    Validation,
    OutOfMemory,
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorFilter {
    Validation,
    OutOfMemory,
    Internal,
}

impl ErrorFilter {
    #[must_use]
    pub(crate) fn matches(self, kind: ErrorKind) -> bool {
        matches!(
            (self, kind),
            (Self::Validation, ErrorKind::Validation)
                | (Self::OutOfMemory, ErrorKind::OutOfMemory)
                | (Self::Internal, ErrorKind::Internal)
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PopErrorScopeError {
    EmptyStack,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceError {
    pub kind: ErrorKind,
    pub message: String,
}

impl DeviceError {
    #[must_use]
    pub(crate) fn validation(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Validation,
            message: message.into(),
        }
    }

    #[must_use]
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::Internal,
            message: message.into(),
        }
    }
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

struct ErrorScope {
    filter: ErrorFilter,
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
            .field("filter", &self.filter)
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;

    fn noop_adapter() -> Adapter {
        Instance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter must exist")
    }

    fn noop_device() -> Device {
        let instance = Instance::new_noop();
        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter must exist");
        adapter
            .create_device(None, &[], "", "")
            .expect("Noop device creation")
    }

    fn hal_noop_adapter() -> yawgpu_hal::HalAdapter {
        yawgpu_hal::HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop HAL adapter must exist")
    }

    fn hal_noop_device() -> yawgpu_hal::HalDevice {
        hal_noop_adapter()
            .create_device()
            .expect("Noop HAL device creation")
    }

    fn hal_noop_queue() -> yawgpu_hal::HalQueue {
        hal_noop_device().queue()
    }

    fn rgba8_unorm() -> TextureFormat {
        TextureFormat::from_raw(0x0000_0016)
    }

    fn valid_texture_descriptor() -> TextureDescriptor {
        TextureDescriptor {
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }
    }

    fn texture_descriptor_4x4() -> TextureDescriptor {
        TextureDescriptor {
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }
    }

    fn layered_mipped_texture_descriptor() -> TextureDescriptor {
        TextureDescriptor {
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 3,
            },
            format: rgba8_unorm(),
            mip_level_count: 3,
            sample_count: 1,
            view_formats: Vec::new(),
        }
    }

    fn noop_texture() -> Texture {
        noop_device().create_texture(texture_descriptor_4x4())
    }

    fn noop_buffer(size: u64, usage: BufferUsage) -> Buffer {
        noop_device().create_buffer(BufferDescriptor {
            usage,
            size,
            mapped_at_creation: false,
        })
    }

    fn noop_render_attachment(device: &Device) -> Arc<TextureView> {
        let texture = device.create_texture(TextureDescriptor {
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::COPY_SRC,
            dimension: TextureDimension::D2,
            size: Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: rgba8_unorm(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
        });
        assert_eq!(error, None);
        Arc::new(view)
    }

    fn noop_render_pass_descriptor(
        view: Arc<TextureView>,
        occlusion_query_set: Option<QuerySet>,
    ) -> RenderPassDescriptor {
        RenderPassDescriptor {
            max_color_attachments: Limits::DEFAULT.max_color_attachments,
            color_attachments: vec![Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                load_op: LoadOp::Clear,
                store_op: StoreOp::Store,
                clear_value: Color {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set,
            timestamp_writes: None,
        }
    }

    fn noop_compute_pipeline(device: &Device) -> Arc<ComputePipeline> {
        Arc::new(
            device.create_compute_pipeline(compute_pipeline_descriptor(compute_shader_module(
                device,
            ))),
        )
    }

    fn noop_render_pipeline(device: &Device) -> Arc<RenderPipeline> {
        Arc::new(
            device.create_render_pipeline(render_pipeline_descriptor(render_shader_module(device))),
        )
    }

    fn empty_bind_group(device: &Device) -> Arc<BindGroup> {
        let layout = Arc::new(device.create_bind_group_layout(BindGroupLayoutDescriptor {
            entries: Vec::new(),
            error: None,
        }));
        Arc::new(device.create_bind_group(layout, Vec::new()))
    }

    fn noop_indirect_buffer(device: &Device) -> Arc<Buffer> {
        Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::INDIRECT | BufferUsage::COPY_DST,
            size: 20,
            mapped_at_creation: false,
        }))
    }

    fn render_bundle_encoder_descriptor() -> RenderBundleEncoderDescriptor {
        RenderBundleEncoderDescriptor {
            max_color_attachments: Limits::DEFAULT.max_color_attachments,
            color_formats: vec![Some(rgba8_unorm())],
            depth_stencil_format: None,
            sample_count: 1,
            depth_read_only: false,
            stencil_read_only: false,
        }
    }

    fn compute_shader_module(device: &Device) -> Arc<ShaderModule> {
        Arc::new(device.create_shader_module(ShaderModuleSource::Wgsl(
            "@compute @workgroup_size(1) fn cs() {}".to_owned(),
        )))
    }

    fn compute_pipeline_descriptor(module: Arc<ShaderModule>) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            layout: ComputePipelineLayout::Auto,
            shader_module: module,
            entry_point: Some("cs".to_owned()),
            constants: Vec::new(),
            error: None,
        }
    }

    fn render_shader_module(device: &Device) -> Arc<ShaderModule> {
        Arc::new(
            device.create_shader_module(ShaderModuleSource::Wgsl(
                r"
@vertex
fn vs() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"
                .to_owned(),
            )),
        )
    }

    fn render_pipeline_descriptor(module: Arc<ShaderModule>) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            layout: RenderPipelineLayout::Auto,
            vertex: RenderPipelineVertexState {
                shader: RenderPipelineShaderStage {
                    module: module.clone(),
                    entry_point: Some("vs".to_owned()),
                    constants: Vec::new(),
                },
                buffer_count: 0,
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: u32::MAX,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(RenderPipelineFragmentState {
                shader: RenderPipelineShaderStage {
                    module,
                    entry_point: Some("fs".to_owned()),
                    constants: Vec::new(),
                },
                target_count: 1,
                targets: vec![ColorTargetState {
                    format: rgba8_unorm(),
                    blend: false,
                    write_mask: 0xF,
                }],
            }),
            error: None,
        }
    }

    #[test]
    fn instance_from_hal_wraps_noop_hal() {
        let instance = Instance::from_hal(yawgpu_hal::HalInstance::new_noop());

        assert_eq!(instance.enumerate_adapters().len(), 1);
    }

    #[test]
    fn instance_enumerate_adapters_with_feature_level_sets_adapter_feature_level() {
        let instance = Instance::new_noop();
        let core = instance.enumerate_adapters_with_feature_level(FeatureLevel::Core);
        let compatibility =
            instance.enumerate_adapters_with_feature_level(FeatureLevel::Compatibility);

        assert_eq!(core.len(), 1);
        assert_eq!(core[0].feature_level(), FeatureLevel::Core);
        assert_eq!(compatibility.len(), 1);
        assert_eq!(
            compatibility[0].feature_level(),
            FeatureLevel::Compatibility
        );
    }

    #[test]
    fn instance_future_registry_process_events_is_empty_without_futures() {
        let instance = Instance::new_noop();

        assert!(instance.future_registry().process_events().is_empty());
    }

    #[test]
    fn instance_hal_returns_noop_hal_instance() {
        let instance = Instance::new_noop();

        assert!(matches!(instance.hal(), yawgpu_hal::HalInstance::Noop(_)));
    }

    #[test]
    fn instance_create_surface_from_metal_layer_noop_returns_noop_surface() {
        let instance = Instance::new_noop();

        let surface = unsafe { instance.create_surface_from_metal_layer(std::ptr::null_mut()) }
            .expect("Noop surface creation should ignore the layer pointer");

        assert!(matches!(surface, yawgpu_hal::HalSurface::Noop));
    }

    #[test]
    fn adapter_from_hal_wraps_noop_hal_adapter() {
        let adapter = Adapter::from_hal(hal_noop_adapter());

        assert!(!adapter.name().is_empty());
    }

    #[test]
    fn adapter_name_backend_limits_and_features_match_noop_contract() {
        let adapter = noop_adapter();

        assert!(adapter.name().contains("Noop"));
        assert_eq!(adapter.backend(), yawgpu_hal::HalBackend::Noop);
        assert_eq!(
            adapter.limits().max_bind_groups,
            Limits::DEFAULT.max_bind_groups
        );
        assert_eq!(
            adapter.limits().max_texture_dimension_2d,
            Limits::DEFAULT.max_texture_dimension_2d
        );
        assert!(adapter.features().contains(&Feature::CoreFeaturesAndLimits));
        assert!(adapter.has_feature(Feature::TimestampQuery));
        assert!(!adapter.has_feature(Feature::Other(7)));
    }

    #[test]
    fn adapter_create_device_rejects_unsupported_required_feature() {
        let adapter = noop_adapter();

        let error = adapter
            .create_device(None, &[Feature::Other(7)], "", "")
            .expect_err("unsupported features must reject device creation");

        assert!(matches!(error, Error::Validation(message) if message.contains("not supported")));
    }

    #[test]
    fn adapter_create_device_applies_labels_and_core_feature() {
        let adapter = noop_adapter();

        let device = adapter
            .create_device(None, &[], "device label", "queue label")
            .expect("Noop device creation should succeed");

        assert_eq!(device.label(), "device label");
        assert!(device.has_feature(Feature::CoreFeaturesAndLimits));
        assert_eq!(device.queue().label(), "queue label");
    }

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
    fn queue_from_hal_hal_label_and_set_label_round_trip() {
        let queue = Queue::from_hal(hal_noop_queue(), "initial");

        assert!(matches!(queue.hal(), yawgpu_hal::HalQueue::Noop(_)));
        assert_eq!(queue.label(), "initial");
        queue.set_label("renamed");
        assert_eq!(queue.label(), "renamed");
    }

    #[test]
    fn queue_write_buffer_and_submit_empty_succeed() {
        let device = noop_device();
        let queue = device.queue();
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 4,
            mapped_at_creation: false,
        });

        assert_eq!(queue.write_buffer(&buffer, 0, &[1, 2, 3, 4]), None);
        assert_eq!(queue.submit(&[]), None);
    }

    #[test]
    fn buffer_usage_from_bits_retain_round_trips_known_and_unknown_bits() {
        let raw = (BufferUsage::MAP_READ | BufferUsage::COPY_DST).bits() | (1_u64 << 40);
        let usage = BufferUsage::from_bits_retain(raw);

        assert_eq!(usage.bits(), raw);
    }

    #[test]
    fn texture_usage_from_bits_retain_round_trips_known_and_unknown_bits() {
        let raw = (TextureUsage::COPY_SRC | TextureUsage::RENDER_ATTACHMENT).bits() | (1_u64 << 40);
        let usage = TextureUsage::from_bits_retain(raw);

        assert_eq!(usage.bits(), raw);
    }

    #[test]
    fn texture_format_from_raw_raw_and_caps_pin_rgba8_unorm_and_undefined() {
        let format = TextureFormat::from_raw(0x0000_0016);

        assert_eq!(format.raw(), 0x0000_0016);

        let caps = format.caps().expect("RGBA8Unorm caps");
        assert_eq!(
            caps.aspects,
            FormatAspects {
                color: true,
                depth: false,
                stencil: false,
            }
        );
        assert_eq!(caps.texel_block_size, 4);
        assert_eq!(caps.block_w, 1);
        assert_eq!(caps.block_h, 1);
        assert_eq!(caps.output_class, Some(FormatOutputClass::Float));
        assert_eq!(caps.color_components, 4);
        assert!(caps.renderable);
        assert!(caps.multisample_capable);
        assert!(caps.storage_capable);
        assert!(caps.is_blendable);
        assert!(caps.has_alpha);
        assert!(!caps.is_compressed);

        assert_eq!(TextureFormat::from_raw(0).caps(), None);
    }

    #[test]
    fn texture_from_hal_and_descriptor_accessors_round_trip() {
        let descriptor = texture_descriptor_4x4();
        let texture = Texture::from_hal(
            descriptor.clone(),
            yawgpu_hal::HalTexture::Noop(yawgpu_hal::noop::NoopTexture),
        );

        assert_eq!(texture.usage(), descriptor.usage);
        assert_eq!(texture.dimension(), descriptor.dimension);
        assert_eq!(texture.size(), descriptor.size);
        assert_eq!(texture.format(), descriptor.format);
        assert_eq!(texture.mip_level_count(), descriptor.mip_level_count);
        assert_eq!(texture.sample_count(), descriptor.sample_count);
        assert!(!texture.is_error());
    }

    #[test]
    fn texture_is_error_same_destroy_create_view_and_validate_queue_write() {
        let texture = noop_texture();
        let other = noop_texture();
        let clone = texture.clone();

        assert!(texture.same(&clone));
        assert!(!texture.same(&other));

        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
        });
        assert_eq!(error, None);
        assert!(!view.is_error());
        assert_eq!(view.format(), texture.format());

        assert_eq!(
            texture.validate_queue_write(
                0,
                Origin3d { x: 0, y: 0, z: 0 },
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                TextureAspect::All,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
                4,
            ),
            Ok(())
        );
        assert_eq!(
            texture.validate_queue_write(
                0,
                Origin3d { x: 4, y: 0, z: 0 },
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                TextureAspect::All,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
                4,
            ),
            Err("queue texture write range exceeds the texture subresource".to_owned())
        );

        texture.destroy();
        texture.destroy();
        assert_eq!(
            texture.validate_queue_write(
                0,
                Origin3d { x: 0, y: 0, z: 0 },
                Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                TextureAspect::All,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
                4,
            ),
            Err("queue texture write destination must be a valid live texture".to_owned())
        );
    }

    #[test]
    fn texture_error_texture_reports_is_error_and_error_view() {
        let device = noop_device();
        let mut invalid = texture_descriptor_4x4();
        invalid.size.width = 0;

        device.push_error_scope(ErrorFilter::Validation);
        let texture = device.create_texture(invalid);
        let scoped = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid texture should be scoped");
        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
        });

        assert_eq!(scoped.kind, ErrorKind::Validation);
        assert_eq!(scoped.message, "2D texture width is out of range");
        assert!(texture.is_error());
        assert!(view.is_error());
        assert_eq!(error, Some("cannot create a view from an error texture"));
    }

    #[test]
    fn texture_view_descriptor_fields_round_trip() {
        let texture = noop_device().create_texture(layered_mipped_texture_descriptor());

        let (view, error) = texture.create_view(TextureViewDescriptor {
            format: Some(rgba8_unorm()),
            dimension: Some(TextureViewDimension::D2Array),
            base_mip_level: 1,
            mip_level_count: Some(1),
            base_array_layer: 1,
            array_layer_count: Some(2),
            aspect: Some(TextureAspect::All),
        });

        assert_eq!(error, None);
        assert!(!view.is_error());
        assert_eq!(view.format(), rgba8_unorm());
        assert_eq!(view.dimension(), TextureViewDimension::D2Array);
        assert_eq!(view.mip_level_count(), 1);
        assert_eq!(view.base_array_layer(), 1);
        assert_eq!(view.aspect(), TextureAspect::All);
    }

    #[test]
    fn sampler_descriptor_and_is_error_pin_valid_and_invalid_descriptors() {
        let device = noop_device();
        let descriptor = SamplerDescriptor {
            address_mode_u: Some(AddressMode::Repeat),
            address_mode_v: Some(AddressMode::MirrorRepeat),
            address_mode_w: Some(AddressMode::ClampToEdge),
            mag_filter: Some(FilterMode::Linear),
            min_filter: Some(FilterMode::Linear),
            mipmap_filter: Some(MipmapFilterMode::Linear),
            lod_min_clamp: 0.5,
            lod_max_clamp: 12.0,
            compare: Some(CompareFunction::LessEqual),
            max_anisotropy: 2,
        };

        let sampler = device.create_sampler(descriptor);
        assert!(!sampler.is_error());
        assert_eq!(sampler.descriptor().address_mode_u, AddressMode::Repeat);
        assert_eq!(
            sampler.descriptor().address_mode_v,
            AddressMode::MirrorRepeat
        );
        assert_eq!(sampler.descriptor().mipmap_filter, MipmapFilterMode::Linear);
        assert_eq!(
            sampler.descriptor().compare,
            Some(CompareFunction::LessEqual)
        );
        assert_eq!(sampler.descriptor().max_anisotropy, 2);

        let invalid = SamplerDescriptor {
            max_anisotropy: 2,
            ..SamplerDescriptor::default()
        };
        device.push_error_scope(ErrorFilter::Validation);
        let error_sampler = device.create_sampler(invalid);
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid sampler should be scoped");
        assert!(error_sampler.is_error());
        assert_eq!(
            error.message,
            "anisotropic samplers require all filters to be Linear"
        );
    }

    #[test]
    fn buffer_accessors_error_same_destroy_hal_and_validate_queue_write() {
        let device = noop_device();
        let buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 16,
            mapped_at_creation: false,
        });
        let clone = buffer.clone();
        let other = noop_buffer(16, BufferUsage::COPY_DST);

        assert_eq!(buffer.size(), 16);
        assert_eq!(buffer.usage(), BufferUsage::COPY_DST);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
        assert!(!buffer.is_error());
        assert!(buffer.same(&clone));
        assert!(!buffer.same(&other));
        assert!(matches!(buffer.hal(), Some(yawgpu_hal::HalBuffer::Noop(_))));
        assert_eq!(buffer.validate_queue_write(0, 4), Ok(()));
        assert_eq!(
            buffer.validate_queue_write(12, 8),
            Err("queue write range exceeds buffer size")
        );

        buffer.destroy();
        buffer.destroy();
        assert_eq!(
            buffer.validate_queue_write(0, 4),
            Err("cannot write to a destroyed buffer")
        );

        device.push_error_scope(ErrorFilter::Validation);
        let error_buffer = device.create_buffer(BufferDescriptor {
            usage: BufferUsage::NONE,
            size: 16,
            mapped_at_creation: false,
        });
        let error = device
            .pop_error_scope()
            .expect("scope should exist")
            .expect("invalid buffer should be scoped");

        assert!(error_buffer.is_error());
        assert_eq!(error.message, "buffer usage must be non-zero");
    }

    #[test]
    fn buffer_map_state_machine_transitions_and_mapped_range_bounds() {
        let mapped = noop_device().create_buffer(BufferDescriptor {
            usage: BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
            size: 16,
            mapped_at_creation: true,
        });
        assert_eq!(mapped.map_state(), BufferMapState::Mapped);
        assert_eq!(
            mapped.begin_map(MapMode::Write, 0, 4),
            Err("buffer is already mapped")
        );
        assert_eq!(mapped.unmap(), None);
        assert_eq!(mapped.map_state(), BufferMapState::Unmapped);
        assert_eq!(mapped.unmap(), None);

        let buffer = noop_buffer(16, BufferUsage::MAP_READ | BufferUsage::COPY_DST);
        assert_eq!(buffer.begin_map(MapMode::Read, 0, 8), Ok(()));
        assert_eq!(buffer.map_state(), BufferMapState::Pending);
        assert_eq!(buffer.resolve_pending_map(), MapAsyncStatus::Success);
        assert_eq!(buffer.map_state(), BufferMapState::Mapped);
        assert!(buffer.mapped_range(true, 0, Some(8)).is_some());
        assert_eq!(buffer.mapped_range(false, 0, Some(8)), None);
        assert_eq!(buffer.mapped_range(true, 12, Some(8)), None);
        assert_eq!(buffer.unmap(), None);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
    }

    #[test]
    fn buffer_abort_pending_map_returns_unmapped_and_resolve_reports_aborted() {
        let buffer = noop_buffer(16, BufferUsage::MAP_READ | BufferUsage::COPY_DST);

        assert_eq!(buffer.begin_map(MapMode::Read, 0, 8), Ok(()));
        assert_eq!(buffer.map_state(), BufferMapState::Pending);
        buffer.abort_pending_map();
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
        assert_eq!(buffer.resolve_pending_map(), MapAsyncStatus::Aborted);
        assert_eq!(buffer.map_state(), BufferMapState::Unmapped);
    }

    #[test]
    fn command_encoder_create_finish_idempotent_and_command_buffer_is_error_false() {
        let encoder = noop_device().create_command_encoder();

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert!(command_buffer.command_ops().is_empty());

        let (second, error) = encoder.finish();
        assert!(second.is_error());
        assert_eq!(
            error,
            Some("command encoder cannot be finished more than once".to_owned())
        );
    }

    #[test]
    fn command_encoder_debug_markers_and_validation_error() {
        let encoder = noop_device().create_command_encoder();

        assert_eq!(encoder.push_debug_group(), None);
        assert_eq!(encoder.insert_debug_marker(), None);
        assert_eq!(encoder.pop_debug_group(), None);
        assert_eq!(
            encoder.record_validation_error("forced encoder validation"),
            None
        );

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(error, Some("forced encoder validation".to_owned()));
    }

    #[test]
    fn command_encoder_buffer_copies_clear_and_write_validate_offsets() {
        let device = noop_device();
        let source = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC,
            size: 32,
            mapped_at_creation: false,
        }));
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_DST,
            size: 32,
            mapped_at_creation: false,
        }));
        let encoder = device.create_command_encoder();

        assert_eq!(
            encoder.copy_buffer_to_buffer(source.clone(), 0, destination.clone(), 0, 16),
            None
        );
        assert_eq!(encoder.clear_buffer(destination.clone(), 0, 16), None);
        assert_eq!(encoder.write_buffer(destination.clone(), 0, 16), None);
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 1);

        let invalid = device.create_command_encoder();
        assert_eq!(
            invalid.copy_buffer_to_buffer(source, 2, destination, 0, 4),
            None
        );
        let (command_buffer, error) = invalid.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("copy source offset must be 4-byte aligned".to_owned())
        );
    }

    #[test]
    fn command_encoder_texture_copies_record_copy_commands() {
        let device = noop_device();
        let texture_a = Arc::new(device.create_texture(texture_descriptor_4x4()));
        let texture_b = Arc::new(device.create_texture(texture_descriptor_4x4()));
        let buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            size: 1024,
            mapped_at_creation: false,
        }));
        let layout = TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(256),
            rows_per_image: None,
        };
        let size = Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        };
        let texture_info_a = TexelCopyTextureInfo {
            texture: texture_a,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::All,
        };
        let texture_info_b = TexelCopyTextureInfo {
            texture: texture_b,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::All,
        };
        let buffer_info = TexelCopyBufferInfo { buffer, layout };
        let encoder = device.create_command_encoder();

        assert_eq!(
            encoder.copy_buffer_to_texture(buffer_info.clone(), texture_info_a.clone(), size),
            None
        );
        assert_eq!(
            encoder.copy_texture_to_buffer(texture_info_a.clone(), buffer_info, size),
            None
        );
        assert_eq!(
            encoder.copy_texture_to_texture(texture_info_a, texture_info_b, size),
            None
        );

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 3);
    }

    #[test]
    fn command_encoder_query_and_timestamps_pin_validation_and_resolve() {
        let device = noop_device();
        let (timestamp_query, _) = device.create_query_set(QuerySetDescriptor {
            label: "bad timestamp".to_owned(),
            kind: QueryType::Timestamp,
            count: 2,
        });
        let timestamp_encoder = device.create_command_encoder();
        assert_eq!(
            timestamp_encoder.write_timestamp(Arc::new(timestamp_query), 0),
            None
        );
        let (command_buffer, error) = timestamp_encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(
            error,
            Some("write timestamp cannot use an error query set".to_owned())
        );

        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "occlusion".to_owned(),
            kind: QueryType::Occlusion,
            count: 2,
        });
        assert_eq!(error, None);
        let destination = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::QUERY_RESOLVE,
            size: 256,
            mapped_at_creation: false,
        }));
        let encoder = device.create_command_encoder();
        assert_eq!(
            encoder.resolve_query_set(Arc::new(query_set), 0, 2, destination, 0),
            None
        );
        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn render_pass_encoder_lifecycle_and_debug_markers() {
        let device = noop_device();
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.push_debug_group(), None);
        assert_eq!(pass.insert_debug_marker(), None);
        assert_eq!(pass.pop_debug_group(), None);
        assert_eq!(
            pass.record_validation_error("forced render pass error"),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(error, Some("forced render pass error".to_owned()));
    }

    #[test]
    fn render_pass_encoder_set_pipeline_bind_group_buffers_and_draw() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let bind_group = empty_bind_group(&device);
        let vertex_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::VERTEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let index_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::INDEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.set_bind_group(0, Some(bind_group), Vec::new()), None);
        assert_eq!(
            pass.set_vertex_buffer(0, Some(vertex_buffer), 0, 16, device.limits()),
            None
        );
        assert_eq!(
            pass.set_index_buffer(index_buffer, Some(IndexFormat::Uint16), 0, 16),
            None
        );
        assert_eq!(pass.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 1);
    }

    #[test]
    fn render_pass_encoder_indexed_and_indirect_draws() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let index_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::INDEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let indirect = noop_indirect_buffer(&device);
        let view = noop_render_attachment(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, None));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(
            pass.set_index_buffer(index_buffer, Some(IndexFormat::Uint16), 0, 16),
            None
        );
        assert_eq!(pass.draw_indexed(3, 1, 0, 0, 0, device.limits()), None);
        assert_eq!(
            pass.draw_indirect(indirect.clone(), 0, device.limits()),
            None
        );
        assert_eq!(
            pass.draw_indexed_indirect(indirect, 0, device.limits()),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn render_pass_encoder_state_setters_occlusion_query_and_execute_bundles() {
        let device = noop_device();
        let view = noop_render_attachment(&device);
        let (query_set, error) = device.create_query_set(QuerySetDescriptor {
            label: "occlusion".to_owned(),
            kind: QueryType::Occlusion,
            count: 2,
        });
        assert_eq!(error, None);
        let (bundle_encoder, error) =
            RenderBundleEncoder::new(render_bundle_encoder_descriptor(), device.limits());
        assert_eq!(error, None);
        let (bundle, error) = bundle_encoder.finish();
        assert_eq!(error, None);
        assert!(!bundle.is_error());
        let bundle = Arc::new(bundle);

        let encoder = device.create_command_encoder();
        let (pass, begin_error) =
            encoder.begin_render_pass(&noop_render_pass_descriptor(view, Some(query_set)));
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_viewport(0.0, 0.0, 4.0, 4.0, 0.0, 1.0), None);
        assert_eq!(pass.set_scissor_rect(0, 0, 4, 4), None);
        assert_eq!(
            pass.set_blend_constant(Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            }),
            None
        );
        assert_eq!(pass.set_stencil_reference(1), None);
        assert_eq!(pass.begin_occlusion_query(0), None);
        assert_eq!(pass.end_occlusion_query(), None);
        assert_eq!(pass.execute_bundles(&[bundle]), None);
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
    }

    #[test]
    fn compute_pass_encoder_lifecycle_and_debug_markers() {
        let encoder = noop_device().create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.push_debug_group(), None);
        assert_eq!(pass.insert_debug_marker(), None);
        assert_eq!(pass.pop_debug_group(), None);
        assert_eq!(
            pass.record_validation_error("forced compute pass error"),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert!(command_buffer.is_error());
        assert_eq!(error, Some("forced compute pass error".to_owned()));
    }

    #[test]
    fn compute_pass_encoder_pipeline_bind_group_and_dispatch() {
        let device = noop_device();
        let pipeline = noop_compute_pipeline(&device);
        let bind_group = empty_bind_group(&device);
        let indirect = noop_indirect_buffer(&device);
        let encoder = device.create_command_encoder();
        let (pass, begin_error) = encoder.begin_compute_pass();
        assert_eq!(begin_error, None);

        assert_eq!(pass.set_pipeline(pipeline), None);
        assert_eq!(pass.set_bind_group(0, Some(bind_group), Vec::new()), None);
        assert_eq!(pass.dispatch_workgroups(1, 1, 1, device.limits()), None);
        assert_eq!(
            pass.dispatch_workgroups_indirect(indirect, 0, device.limits()),
            None
        );
        assert_eq!(pass.end(), None);

        let (command_buffer, error) = encoder.finish();
        assert_eq!(error, None);
        assert!(!command_buffer.is_error());
        assert_eq!(command_buffer.command_ops().len(), 1);
    }

    #[test]
    fn render_bundle_encoder_lifecycle_set_pipeline_buffers_and_draws() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let bind_group = empty_bind_group(&device);
        let vertex_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::VERTEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let index_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::INDEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let (bundle_encoder, error) =
            RenderBundleEncoder::new(render_bundle_encoder_descriptor(), device.limits());
        assert_eq!(error, None);

        assert_eq!(bundle_encoder.insert_debug_marker(), None);
        assert_eq!(bundle_encoder.push_debug_group(), None);
        assert_eq!(bundle_encoder.pop_debug_group(), None);
        assert_eq!(bundle_encoder.set_pipeline(pipeline), None);
        assert_eq!(
            bundle_encoder.set_bind_group(0, Some(bind_group), Vec::new()),
            None
        );
        assert_eq!(
            bundle_encoder.set_vertex_buffer(0, Some(vertex_buffer), 0, 16, device.limits()),
            None
        );
        assert_eq!(
            bundle_encoder.set_index_buffer(index_buffer, Some(IndexFormat::Uint16), 0, 16),
            None
        );
        assert_eq!(bundle_encoder.draw(3, 1, 0, 0, device.limits()), None);
        assert_eq!(
            bundle_encoder.draw_indexed(3, 1, 0, 0, 0, device.limits()),
            None
        );
        let (bundle, error) = bundle_encoder.finish();
        assert_eq!(error, None);
        assert!(!bundle.is_error());
    }

    #[test]
    fn render_bundle_encoder_indirect_draws() {
        let device = noop_device();
        let pipeline = noop_render_pipeline(&device);
        let index_buffer = Arc::new(device.create_buffer(BufferDescriptor {
            usage: BufferUsage::INDEX,
            size: 16,
            mapped_at_creation: false,
        }));
        let indirect = noop_indirect_buffer(&device);
        let (bundle_encoder, error) =
            RenderBundleEncoder::new(render_bundle_encoder_descriptor(), device.limits());
        assert_eq!(error, None);

        assert_eq!(bundle_encoder.set_pipeline(pipeline), None);
        assert_eq!(
            bundle_encoder.set_index_buffer(index_buffer, Some(IndexFormat::Uint16), 0, 16),
            None
        );
        assert_eq!(
            bundle_encoder.draw_indirect(indirect.clone(), 0, device.limits()),
            None
        );
        assert_eq!(
            bundle_encoder.draw_indexed_indirect(indirect, 0, device.limits()),
            None
        );
        let (bundle, error) = bundle_encoder.finish();
        assert_eq!(error, None);
        assert!(!bundle.is_error());
    }

    #[test]
    fn creates_noop_device_and_queue() {
        let instance = Instance::new_noop();
        let adapters = instance.enumerate_adapters();
        assert_eq!(adapters.len(), 1);

        let device = adapters[0]
            .create_device(None, &[], "", "")
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
    fn map_mode_from_bits_rejects_none_both_and_unsupported_bits() {
        assert_eq!(MapMode::from_bits(1), Ok(MapMode::Read));
        assert_eq!(MapMode::from_bits(2), Ok(MapMode::Write));
        assert_eq!(
            MapMode::from_bits(0),
            Err("map mode must be exactly Read or Write")
        );
        assert_eq!(
            MapMode::from_bits(1 | 2),
            Err("map mode must be exactly Read or Write")
        );
        assert_eq!(MapMode::from_bits(4), Err("map mode has unsupported bits"));
    }
}
