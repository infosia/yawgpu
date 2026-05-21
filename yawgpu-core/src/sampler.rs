use std::sync::Arc;

use yawgpu_hal::{
    HalAddressMode, HalCompareFunction, HalFilterMode, HalMipmapFilterMode, HalSampler,
    HalSamplerDescriptor,
};

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

#[derive(Debug, Clone)]
pub struct Sampler {
    pub(crate) inner: Arc<SamplerInner>,
}

#[derive(Debug)]
pub(crate) struct SamplerInner {
    pub(crate) _hal: Option<HalSampler>,
    pub(crate) descriptor: ResolvedSamplerDescriptor,
    pub(crate) is_error: bool,
}

impl Sampler {
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

    #[must_use]
    pub fn descriptor(&self) -> ResolvedSamplerDescriptor {
        self.inner.descriptor
    }

    #[must_use]
    pub fn is_error(&self) -> bool {
        self.inner.is_error
    }
}

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

pub(crate) fn hal_address_mode(mode: AddressMode) -> HalAddressMode {
    match mode {
        AddressMode::ClampToEdge => HalAddressMode::ClampToEdge,
        AddressMode::Repeat => HalAddressMode::Repeat,
        AddressMode::MirrorRepeat => HalAddressMode::MirrorRepeat,
    }
}

pub(crate) fn hal_filter_mode(mode: FilterMode) -> HalFilterMode {
    match mode {
        FilterMode::Nearest => HalFilterMode::Nearest,
        FilterMode::Linear => HalFilterMode::Linear,
    }
}

pub(crate) fn hal_mipmap_filter_mode(mode: MipmapFilterMode) -> HalMipmapFilterMode {
    match mode {
        MipmapFilterMode::Nearest => HalMipmapFilterMode::Nearest,
        MipmapFilterMode::Linear => HalMipmapFilterMode::Linear,
    }
}

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

pub(crate) fn validate_sampler_descriptor(
    descriptor: &ResolvedSamplerDescriptor,
) -> Option<&'static str> {
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
    }
}
