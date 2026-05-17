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
            Self::R8_UNORM => FormatCaps::color(1).renderable().multisample(),
            Self::R8_SNORM => FormatCaps::color(1),
            Self::R8_UINT | Self::R8_SINT => FormatCaps::color(1).renderable().multisample(),
            Self::RG8_UNORM => FormatCaps::color(2).renderable().multisample(),
            Self::RG8_SNORM => FormatCaps::color(2),
            Self::RG8_UINT | Self::RG8_SINT => FormatCaps::color(2).renderable().multisample(),
            Self::R32_FLOAT | Self::R32_UINT | Self::R32_SINT => {
                FormatCaps::color(4).renderable().multisample().storage()
            }
            Self::RGBA8_UNORM => FormatCaps::color(4).renderable().multisample().storage(),
            Self::RGBA8_UNORM_SRGB => FormatCaps::color(4).renderable().multisample(),
            Self::BGRA8_UNORM | Self::BGRA8_UNORM_SRGB => {
                FormatCaps::color(4).renderable().multisample()
            }
            // snorm formats are NOT storage-capable (Dawn `Format.cpp`).
            Self::RGBA8_SNORM => FormatCaps::color(4),
            Self::RGBA8_UINT | Self::RGBA8_SINT => {
                FormatCaps::color(4).renderable().multisample().storage()
            }
            Self::RG11B10_UFLOAT | Self::RGB9E5_UFLOAT => FormatCaps::color(4),
            Self::RG32_FLOAT | Self::RG32_UINT | Self::RG32_SINT => {
                FormatCaps::color(8).renderable().storage()
            }
            Self::RGBA16_UNORM => FormatCaps::color(8).renderable().multisample().storage(),
            // snorm formats are NOT storage-capable (Dawn `Format.cpp`); the
            // remaining `*16` renderable/multisample approximation stays a
            // tracked note (block 20 → P4/P5).
            Self::RGBA16_SNORM => FormatCaps::color(8).renderable().multisample(),
            Self::RGBA16_UINT | Self::RGBA16_SINT | Self::RGBA16_FLOAT => {
                FormatCaps::color(8).renderable().multisample().storage()
            }
            Self::RGBA32_FLOAT | Self::RGBA32_UINT | Self::RGBA32_SINT => {
                FormatCaps::color(16).renderable().storage()
            }
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
            // Unknown WebGPU formats are treated conservatively as plain
            // renderable color until a later phase needs exact capabilities.
            _ => FormatCaps::color(4).renderable().multisample(),
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
    pub is_compressed: bool,
    pub texel_block_size: u32,
    pub block_w: u32,
    pub block_h: u32,
}

impl FormatCaps {
    const fn color(texel_block_size: u32) -> Self {
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
        )
    }

    const fn new(
        aspects: FormatAspects,
        texel_block_size: u32,
        block_w: u32,
        block_h: u32,
        is_compressed: bool,
    ) -> Self {
        Self {
            aspects,
            renderable: false,
            multisample_capable: false,
            storage_capable: false,
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
        (TextureView::new(resolved, is_error), error)
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
    fn new(descriptor: ResolvedTextureViewDescriptor, is_error: bool) -> Self {
        Self {
            inner: Arc::new(TextureViewInner {
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
        _module: Box<naga::Module>,
        _info: Box<naga::valid::ModuleInfo>,
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
            _module: Box::new(validated.module),
            _info: Box::new(validated.info),
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
