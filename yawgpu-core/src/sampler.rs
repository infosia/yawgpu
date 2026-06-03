use std::sync::Arc;

use yawgpu_hal::{
    HalAddressMode, HalCompareFunction, HalFilterMode, HalMipmapFilterMode, HalSampler,
    HalSamplerDescriptor,
};

/// Enumerates address mode values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AddressMode {
    /// Clamp to edge variant.
    ClampToEdge,
    /// Repeat variant.
    Repeat,
    /// Mirror repeat variant.
    MirrorRepeat,
}

/// Enumerates filter mode values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FilterMode {
    /// Nearest variant.
    Nearest,
    /// Linear variant.
    Linear,
}

/// Enumerates mipmap filter mode values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MipmapFilterMode {
    /// Nearest variant.
    Nearest,
    /// Linear variant.
    Linear,
}

/// Enumerates compare function values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompareFunction {
    /// Never variant.
    Never,
    /// Less variant.
    Less,
    /// Equal variant.
    Equal,
    /// Less equal variant.
    LessEqual,
    /// Greater variant.
    Greater,
    /// Not equal variant.
    NotEqual,
    /// Greater equal variant.
    GreaterEqual,
    /// Always variant.
    Always,
}

/// Describes sampler descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerDescriptor {
    /// Address mode u.
    pub address_mode_u: Option<AddressMode>,
    /// Address mode v.
    pub address_mode_v: Option<AddressMode>,
    /// Address mode w.
    pub address_mode_w: Option<AddressMode>,
    /// Mag filter.
    pub mag_filter: Option<FilterMode>,
    /// Min filter.
    pub min_filter: Option<FilterMode>,
    /// Mipmap filter.
    pub mipmap_filter: Option<MipmapFilterMode>,
    /// Lod min clamp.
    pub lod_min_clamp: f32,
    /// Lod max clamp.
    pub lod_max_clamp: f32,
    /// Compare.
    pub compare: Option<CompareFunction>,
    /// Max anisotropy.
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

/// Describes resolved sampler descriptor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedSamplerDescriptor {
    /// Address mode u.
    pub address_mode_u: AddressMode,
    /// Address mode v.
    pub address_mode_v: AddressMode,
    /// Address mode w.
    pub address_mode_w: AddressMode,
    /// Mag filter.
    pub mag_filter: FilterMode,
    /// Min filter.
    pub min_filter: FilterMode,
    /// Mipmap filter.
    pub mipmap_filter: MipmapFilterMode,
    /// Lod min clamp.
    pub lod_min_clamp: f32,
    /// Lod max clamp.
    pub lod_max_clamp: f32,
    /// Compare.
    pub compare: Option<CompareFunction>,
    /// Max anisotropy.
    pub max_anisotropy: u16,
}

impl ResolvedSamplerDescriptor {
    /// Constructs this object from descriptor.
    pub(crate) fn from_descriptor(descriptor: SamplerDescriptor) -> Self {
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

/// Stores sampler data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct Sampler {
    pub(crate) inner: Arc<SamplerInner>,
}

/// Holds shared state for the sampler handle.
#[derive(Debug)]
pub(crate) struct SamplerInner {
    pub(crate) _hal: Option<HalSampler>,
    pub(crate) descriptor: ResolvedSamplerDescriptor,
    pub(crate) is_error: bool,
}

impl Sampler {
    /// Creates a new instance.
    pub(crate) fn new(
        descriptor: ResolvedSamplerDescriptor,
        hal: Option<HalSampler>,
        is_error: bool,
    ) -> Self {
        Self {
            inner: Arc::new(SamplerInner {
                _hal: hal,
                descriptor,
                is_error,
            }),
        }
    }

    /// Returns the descriptor.
    #[must_use]
    pub fn descriptor(&self) -> ResolvedSamplerDescriptor {
        self.inner.descriptor
    }

    /// Returns true when this object is error.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }

    /// Returns the HAL sampler.
    pub(crate) fn hal(&self) -> Option<HalSampler> {
        self.inner._hal.clone()
    }
}

/// Returns HAL sampler descriptor.
pub(crate) fn hal_sampler_descriptor(
    descriptor: &ResolvedSamplerDescriptor,
) -> HalSamplerDescriptor {
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

/// Returns HAL address mode.
pub(crate) fn hal_address_mode(mode: AddressMode) -> HalAddressMode {
    match mode {
        AddressMode::ClampToEdge => HalAddressMode::ClampToEdge,
        AddressMode::Repeat => HalAddressMode::Repeat,
        AddressMode::MirrorRepeat => HalAddressMode::MirrorRepeat,
    }
}

/// Returns HAL filter mode.
pub(crate) fn hal_filter_mode(mode: FilterMode) -> HalFilterMode {
    match mode {
        FilterMode::Nearest => HalFilterMode::Nearest,
        FilterMode::Linear => HalFilterMode::Linear,
    }
}

/// Returns HAL mipmap filter mode.
pub(crate) fn hal_mipmap_filter_mode(mode: MipmapFilterMode) -> HalMipmapFilterMode {
    match mode {
        MipmapFilterMode::Nearest => HalMipmapFilterMode::Nearest,
        MipmapFilterMode::Linear => HalMipmapFilterMode::Linear,
    }
}

/// Returns HAL compare function.
pub(crate) fn hal_compare_function(compare: CompareFunction) -> HalCompareFunction {
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

/// Validates sampler descriptor and returns a descriptive error on failure.
pub(crate) fn validate_sampler_descriptor(
    descriptor: &ResolvedSamplerDescriptor,
) -> Option<&'static str> {
    if !descriptor.lod_min_clamp.is_finite() {
        return Some("sampler lodMinClamp must be finite");
    }
    if !descriptor.lod_max_clamp.is_finite() {
        return Some("sampler lodMaxClamp must be finite");
    }
    if descriptor.lod_min_clamp < 0.0 {
        return Some("sampler lodMinClamp must be non-negative");
    }
    if descriptor.lod_max_clamp < 0.0 {
        return Some("sampler lodMaxClamp must be non-negative");
    }
    if descriptor.lod_min_clamp > descriptor.lod_max_clamp {
        return Some("sampler lodMinClamp must not exceed lodMaxClamp");
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::*;
    use crate::*;

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

        let invalid = SamplerDescriptor {
            lod_min_clamp: -1.0,
            ..SamplerDescriptor::default()
        };
        let resolved = ResolvedSamplerDescriptor::from_descriptor(invalid);
        assert_eq!(
            validate_sampler_descriptor(&resolved),
            Some("sampler lodMinClamp must be non-negative")
        );
    }
}
