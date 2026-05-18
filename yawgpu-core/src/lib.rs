use std::cell::UnsafeCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use yawgpu_hal::{
    HalAdapter, HalBuffer, HalDevice, HalError, HalInstance, HalQueue, HalSampler, HalTexture,
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
        // Block 00: the synthetic Noop adapter's supported limits are the
        // WebGPU spec defaults by design.
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

    #[must_use]
    pub fn create_buffer(&self, descriptor: BufferDescriptor) -> Buffer {
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
        let error = validate_texture_descriptor(&descriptor, self.limits());
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }

        let hal = if is_error {
            None
        } else {
            Some(self.inner.hal.create_texture())
        };

        Texture::new(descriptor, hal, is_error)
    }

    #[must_use]
    pub fn create_sampler(&self, descriptor: SamplerDescriptor) -> Sampler {
        let resolved = ResolvedSamplerDescriptor::from_descriptor(descriptor);
        let error = validate_sampler_descriptor(&resolved);
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }

        let hal = if is_error {
            None
        } else {
            Some(self.inner.hal.create_sampler())
        };

        Sampler::new(resolved, hal, is_error)
    }

    #[must_use]
    pub fn create_shader_module(&self, source: ShaderModuleSource) -> ShaderModule {
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
        let error = descriptor
            .error
            .clone()
            .or_else(|| validate_bind_group_layout_descriptor(&descriptor.entries, self.limits()));
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        BindGroupLayout::new(descriptor.entries, is_error)
    }

    #[must_use]
    pub fn create_bind_group(
        &self,
        layout: Arc<BindGroupLayout>,
        entries: Vec<BindGroupEntry>,
    ) -> BindGroup {
        let error = validate_bind_group_descriptor(self, &layout, &entries, self.limits());
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        BindGroup::new(layout, entries, is_error)
    }

    #[must_use]
    pub fn create_pipeline_layout(&self, descriptor: PipelineLayoutDescriptor) -> PipelineLayout {
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
    pub fn create_compute_pipeline(
        &self,
        descriptor: ComputePipelineDescriptor,
    ) -> ComputePipeline {
        let error = descriptor
            .error
            .clone()
            .or_else(|| validate_compute_pipeline_descriptor(&descriptor, self.limits()));
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        ComputePipeline::new(descriptor, is_error)
    }

    #[must_use]
    pub fn create_render_pipeline(&self, descriptor: RenderPipelineDescriptor) -> RenderPipeline {
        let error = descriptor
            .error
            .clone()
            .or_else(|| validate_render_pipeline_descriptor(&descriptor, self.limits()));
        let is_error = error.is_some();
        if let Some(message) = error {
            self.dispatch_error(ErrorKind::Validation, message);
        }
        RenderPipeline::new(descriptor, is_error)
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
    pub fn contains(self, other: Self) -> bool {
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
    pub fn contains(self, other: Self) -> bool {
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
    pub fn is_undefined(self) -> bool {
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
            // Unknown WebGPU formats stay conservative for the carried W5
            // approximation: plain renderable float color with alpha.
            _ => FormatCaps::float_color(4, 4)
                .alpha()
                .blendable()
                .renderable()
                .multisample(),
        };
        Some(caps)
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
    _hal: Option<HalTexture>,
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
                _hal: hal,
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
    pub fn view_formats(&self) -> &[TextureFormat] {
        &self.inner.view_formats
    }

    /// A view format is compatible only when it equals the texture's format
    /// or is explicitly listed in the texture's `viewFormats`. There is no
    /// implicit sRGB-counterpart allowance — that mirrors Dawn
    /// `Texture.cpp` `ValidateCanViewTextureAs`.
    #[must_use]
    pub fn is_view_format_compatible(&self, view_format: TextureFormat) -> bool {
        view_format == self.format() || self.view_formats().contains(&view_format)
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.state.lock().is_error
    }

    #[must_use]
    pub fn is_destroyed(&self) -> bool {
        self.inner.state.lock().is_destroyed
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
    ) -> Result<(), &'static str> {
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
    pub fn texture(&self) -> Texture {
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
    pub fn base_mip_level(&self) -> u32 {
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
    pub fn array_layer_count(&self) -> u32 {
        self.inner.array_layer_count
    }

    #[must_use]
    pub fn aspect(&self) -> TextureAspect {
        self.inner.aspect
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
}

impl BindGroupLayout {
    fn new(entries: Vec<BindGroupLayoutEntry>, is_error: bool) -> Self {
        Self {
            inner: Arc::new(BindGroupLayoutInner { entries, is_error }),
        }
    }

    #[must_use]
    pub fn entries(&self) -> &[BindGroupLayoutEntry] {
        &self.inner.entries
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
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
    _workgroup: Option<ResolvedComputeWorkgroup>,
    is_error: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResolvedComputeWorkgroup {
    size: [u32; 3],
    storage_size: u32,
}

impl ComputePipeline {
    fn new(descriptor: ComputePipelineDescriptor, is_error: bool) -> Self {
        let resolved = if is_error {
            None
        } else {
            resolve_compute_pipeline(&descriptor).ok()
        };
        let (entry_name, bindings, workgroup) = resolved.unwrap_or_else(|| {
            (
                descriptor.entry_point.clone().unwrap_or_default(),
                Vec::new(),
                None,
            )
        });
        Self {
            inner: Arc::new(ComputePipelineInner {
                _layout: descriptor.layout,
                _shader_module: descriptor.shader_module,
                entry_name,
                _bindings: bindings,
                _workgroup: workgroup,
                is_error,
            }),
        }
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    #[must_use]
    pub fn entry_name(&self) -> &str {
        &self.inner.entry_name
    }
}

type ResolvedPipelineParts = (
    String,
    Vec<shader_naga::ReflectedResourceBinding>,
    Option<ResolvedComputeWorkgroup>,
);

fn validate_compute_pipeline_descriptor(
    descriptor: &ComputePipelineDescriptor,
    limits: Limits,
) -> Option<String> {
    resolve_compute_pipeline_descriptor(descriptor, limits).err()
}

fn resolve_compute_pipeline(
    descriptor: &ComputePipelineDescriptor,
) -> Result<ResolvedPipelineParts, String> {
    resolve_compute_pipeline_descriptor(descriptor, Limits::DEFAULT)
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
    let bindings = module.resource_bindings();
    validate_compute_pipeline_layout(&descriptor.layout, &bindings)?;
    Ok((entry_name, bindings, Some(workgroup)))
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
    if workgroup.workgroup_storage_size > limits.max_compute_workgroup_storage_size {
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
        (shader_naga::ReflectedResourceBindingKind::Sampler, BindingLayoutKind::Sampler { .. }) => {
            Ok(())
        }
        (
            shader_naga::ReflectedResourceBindingKind::Texture { .. },
            BindingLayoutKind::Texture { .. },
        ) => Ok(()),
        (
            shader_naga::ReflectedResourceBindingKind::StorageTexture { .. },
            BindingLayoutKind::StorageTexture { .. },
        ) => Ok(()),
        _ => Err("compute pipeline layout binding type is incompatible".to_owned()),
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
    is_error: bool,
}

impl RenderPipeline {
    fn new(descriptor: RenderPipelineDescriptor, is_error: bool) -> Self {
        let resolved = if is_error {
            None
        } else {
            resolve_render_pipeline_descriptor(&descriptor, Limits::DEFAULT).ok()
        };
        let (vertex_entry_name, fragment_entry_name) = resolved.unwrap_or_else(|| {
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
            )
        });
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
                is_error,
            }),
        }
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
}

fn validate_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    limits: Limits,
) -> Option<String> {
    resolve_render_pipeline_descriptor(descriptor, limits).err()
}

fn resolve_render_pipeline_descriptor(
    descriptor: &RenderPipelineDescriptor,
    limits: Limits,
) -> Result<(String, Option<String>), String> {
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
    validate_render_pipeline_layout(descriptor)?;
    validate_multisample_state(descriptor, fragment_entry.as_deref())?;

    Ok((vertex_entry, fragment_entry))
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

fn validate_render_pipeline_layout(descriptor: &RenderPipelineDescriptor) -> Result<(), String> {
    let RenderPipelineLayout::Explicit(layout) = &descriptor.layout else {
        return Ok(());
    };
    if layout.is_error() {
        return Err("render pipeline layout must not be an error pipeline layout".to_owned());
    }

    let mut requirements =
        stage_resource_bindings(&descriptor.vertex.shader, PipelineShaderStage::Vertex)?;
    if let Some(fragment) = &descriptor.fragment {
        requirements.extend(stage_resource_bindings(
            &fragment.shader,
            PipelineShaderStage::Fragment,
        )?);
    }
    validate_pipeline_layout_stage_bindings(layout, &requirements)
}

fn stage_resource_bindings(
    stage: &RenderPipelineShaderStage,
    pipeline_stage: PipelineShaderStage,
) -> Result<Vec<StageResourceBinding>, String> {
    let Some(module) = stage.module.validated_wgsl() else {
        return Err("render pipeline stage requires a valid WGSL shader module".to_owned());
    };
    Ok(module
        .resource_bindings()
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
    _hal: Option<HalBuffer>,
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
                _hal: hal,
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
    pub fn unmap(&self) {
        let mut state = self.inner.state.lock();
        if let Some(pending) = state.pending_map.as_mut() {
            pending.outcome = MapAsyncStatus::Aborted;
        }
        state.map_state = BufferMapState::Unmapped;
        state.active_map = None;
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
        let outcome = pending
            .as_ref()
            .map(|pending| pending.outcome)
            .unwrap_or(MapAsyncStatus::Aborted);
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
        self.inner.host.ptr_at(offset)
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
) -> Result<(), &'static str> {
    if !texture.usage().contains(TextureUsage::COPY_DST) {
        return Err("queue texture write destination usage must include CopyDst");
    }
    if texture.is_error() || texture.is_destroyed() {
        return Err("queue texture write destination must be a valid live texture");
    }
    if texture.sample_count() != 1 {
        return Err("queue texture write destination sampleCount must be one");
    }
    if mip_level >= texture.mip_level_count() {
        return Err("queue texture write mipLevel is out of range");
    }

    let Some(format_caps) = texture.format().caps() else {
        return Err("queue texture write format must not be Undefined");
    };
    match aspect {
        TextureAspect::All => {}
        TextureAspect::DepthOnly if !format_caps.aspects.depth => {
            return Err("DepthOnly texture writes require a depth format");
        }
        TextureAspect::StencilOnly if !format_caps.aspects.stencil => {
            return Err("StencilOnly texture writes require a stencil format");
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
        return Err("queue texture write range exceeds the texture subresource");
    }
    if texture.dimension() == TextureDimension::D2 && write_size.depth_or_array_layers != 1 {
        return Err("queue texture writes to 2D textures require depthOrArrayLayers to be one");
    }

    validate_texel_copy_layout(format_caps, aspect, write_size, layout, data_size)
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
    data_size: u64,
) -> Result<(), &'static str> {
    let width_blocks = div_ceil_u32(write_size.width, format_caps.block_w);
    let height_blocks = div_ceil_u32(write_size.height, format_caps.block_h);
    let depth = write_size.depth_or_array_layers;
    let block_size = texel_copy_block_size(format_caps, aspect);
    let last_row_bytes = u64::from(width_blocks)
        .checked_mul(u64::from(block_size))
        .ok_or("queue texture write row byte size overflows")?;

    if let Some(bytes_per_row) = layout.bytes_per_row {
        if u64::from(bytes_per_row) < last_row_bytes {
            return Err("queue texture write bytesPerRow is too small");
        }
    } else if height_blocks > 1 || depth > 1 {
        return Err("queue texture write bytesPerRow is required for multi-row copies");
    }

    if let Some(rows_per_image) = layout.rows_per_image {
        if rows_per_image < height_blocks {
            return Err("queue texture write rowsPerImage is too small");
        }
    } else if depth > 1 {
        return Err("queue texture write rowsPerImage is required for multi-image copies");
    }

    let required_bytes = required_bytes_in_texel_copy(
        layout.bytes_per_row,
        layout.rows_per_image,
        height_blocks,
        depth,
        last_row_bytes,
    )?;
    let required_end = layout
        .offset
        .checked_add(required_bytes)
        .ok_or("queue texture write data range overflows")?;
    if required_end > data_size {
        return Err("queue texture write dataSize is too small");
    }

    Ok(())
}

fn required_bytes_in_texel_copy(
    bytes_per_row: Option<u32>,
    rows_per_image: Option<u32>,
    height_blocks: u32,
    depth: u32,
    last_row_bytes: u64,
) -> Result<u64, &'static str> {
    if last_row_bytes == 0 || height_blocks == 0 || depth == 0 {
        return Ok(0);
    }

    let bytes_per_row = u64::from(bytes_per_row.unwrap_or(0));
    let rows_per_image = u64::from(rows_per_image.unwrap_or(height_blocks));
    let image_offset_rows = rows_per_image
        .checked_mul(u64::from(depth.saturating_sub(1)))
        .ok_or("queue texture write required byte size overflows")?;
    let row_offset_rows = u64::from(height_blocks.saturating_sub(1));
    let offset_rows = image_offset_rows
        .checked_add(row_offset_rows)
        .ok_or("queue texture write required byte size overflows")?;
    let offset_bytes = bytes_per_row
        .checked_mul(offset_rows)
        .ok_or("queue texture write required byte size overflows")?;
    offset_bytes
        .checked_add(last_row_bytes)
        .ok_or("queue texture write required byte size overflows")
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

    use super::{ErrorKind, FutureCallbackMode, FutureRegistry, Instance, MapMode, WaitAnyStatus};

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
