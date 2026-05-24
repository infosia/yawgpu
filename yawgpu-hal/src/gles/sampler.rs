use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::{rebuild_hal_error, BACKEND};
use crate::{
    HalAddressMode, HalCompareFunction, HalError, HalFilterMode, HalMipmapFilterMode,
    HalSamplerDescriptor,
};

pub(super) struct GlesSamplerInner {
    device: Arc<GlesDeviceInner>,
    sampler: Result<glow::Sampler, HalError>,
}

impl Drop for GlesSamplerInner {
    fn drop(&mut self) {
        if let Ok(sampler) = self.sampler.as_ref() {
            let sampler = *sampler;
            let _ = self.device.with_current_context(|gl| unsafe {
                gl.delete_sampler(sampler);
            });
        }
    }
}

/// Stores GLES sampler data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesSampler {
    inner: Arc<GlesSamplerInner>,
}

// SAFETY: `GlesSampler` accesses GL state only through `GlesDeviceInner`, whose
// make-current lock serializes all GL commands.
unsafe impl Send for GlesSampler {}
// SAFETY: See the `Send` impl; shared operations are synchronized by the
// owning device inner.
unsafe impl Sync for GlesSampler {}

impl std::fmt::Debug for GlesSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesSampler").finish()
    }
}

impl GlesSampler {
    pub(super) fn new(device: Arc<GlesDeviceInner>, descriptor: &HalSamplerDescriptor) -> Self {
        let sampler = allocate_sampler(&device, descriptor);
        Self {
            inner: Arc::new(GlesSamplerInner { device, sampler }),
        }
    }

    #[allow(dead_code)]
    pub(super) fn raw_or_err(&self) -> Result<glow::Sampler, HalError> {
        self.inner
            .sampler
            .as_ref()
            .copied()
            .map_err(rebuild_hal_error)
    }
}

fn allocate_sampler(
    device: &Arc<GlesDeviceInner>,
    descriptor: &HalSamplerDescriptor,
) -> Result<glow::Sampler, HalError> {
    let address_u = map_address_mode(descriptor.address_mode_u);
    let address_v = map_address_mode(descriptor.address_mode_v);
    let address_w = map_address_mode(descriptor.address_mode_w);
    let mag_filter = map_filter_mode(descriptor.mag_filter);
    let min_filter = map_min_filter(descriptor.min_filter, descriptor.mipmap_filter);
    let compare = descriptor.compare.map(map_compare_function);

    device
        .with_current_context(|gl| unsafe {
            let sampler = gl
                .create_sampler()
                .map_err(|_| HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "glCreateSampler failed",
                })?;
            gl.sampler_parameter_i32(sampler, glow::TEXTURE_WRAP_S, address_u);
            gl.sampler_parameter_i32(sampler, glow::TEXTURE_WRAP_T, address_v);
            gl.sampler_parameter_i32(sampler, glow::TEXTURE_WRAP_R, address_w);
            gl.sampler_parameter_i32(sampler, glow::TEXTURE_MAG_FILTER, mag_filter);
            gl.sampler_parameter_i32(sampler, glow::TEXTURE_MIN_FILTER, min_filter);
            gl.sampler_parameter_f32(sampler, glow::TEXTURE_MIN_LOD, descriptor.lod_min_clamp);
            gl.sampler_parameter_f32(sampler, glow::TEXTURE_MAX_LOD, descriptor.lod_max_clamp);
            if let Some(compare_func) = compare {
                gl.sampler_parameter_i32(
                    sampler,
                    glow::TEXTURE_COMPARE_MODE,
                    glow::COMPARE_REF_TO_TEXTURE as i32,
                );
                gl.sampler_parameter_i32(sampler, glow::TEXTURE_COMPARE_FUNC, compare_func);
            }
            if descriptor.max_anisotropy > 1
                && gl
                    .supported_extensions()
                    .contains("GL_EXT_texture_filter_anisotropic")
            {
                gl.sampler_parameter_f32(
                    sampler,
                    glow::TEXTURE_MAX_ANISOTROPY_EXT,
                    f32::from(descriptor.max_anisotropy),
                );
            }
            Ok(sampler)
        })
        .and_then(|result| result)
}

fn map_filter_mode(mode: HalFilterMode) -> i32 {
    match mode {
        HalFilterMode::Nearest => glow::NEAREST as i32,
        HalFilterMode::Linear => glow::LINEAR as i32,
    }
}

fn map_address_mode(mode: HalAddressMode) -> i32 {
    match mode {
        HalAddressMode::ClampToEdge => glow::CLAMP_TO_EDGE as i32,
        HalAddressMode::Repeat => glow::REPEAT as i32,
        HalAddressMode::MirrorRepeat => glow::MIRRORED_REPEAT as i32,
    }
}

fn map_min_filter(min_filter: HalFilterMode, mipmap_filter: HalMipmapFilterMode) -> i32 {
    match (min_filter, mipmap_filter) {
        (HalFilterMode::Nearest, HalMipmapFilterMode::Nearest) => {
            glow::NEAREST_MIPMAP_NEAREST as i32
        }
        (HalFilterMode::Nearest, HalMipmapFilterMode::Linear) => glow::NEAREST_MIPMAP_LINEAR as i32,
        (HalFilterMode::Linear, HalMipmapFilterMode::Nearest) => glow::LINEAR_MIPMAP_NEAREST as i32,
        (HalFilterMode::Linear, HalMipmapFilterMode::Linear) => glow::LINEAR_MIPMAP_LINEAR as i32,
    }
}

fn map_compare_function(function: HalCompareFunction) -> i32 {
    match function {
        HalCompareFunction::Never => glow::NEVER as i32,
        HalCompareFunction::Less => glow::LESS as i32,
        HalCompareFunction::Equal => glow::EQUAL as i32,
        HalCompareFunction::LessEqual => glow::LEQUAL as i32,
        HalCompareFunction::Greater => glow::GREATER as i32,
        HalCompareFunction::NotEqual => glow::NOTEQUAL as i32,
        HalCompareFunction::GreaterEqual => glow::GEQUAL as i32,
        HalCompareFunction::Always => glow::ALWAYS as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_filter_mode_maps_known_values() {
        assert_eq!(
            map_filter_mode(HalFilterMode::Nearest),
            glow::NEAREST as i32
        );
        assert_eq!(map_filter_mode(HalFilterMode::Linear), glow::LINEAR as i32);
    }

    #[test]
    fn map_address_mode_maps_known_values() {
        assert_eq!(
            map_address_mode(HalAddressMode::ClampToEdge),
            glow::CLAMP_TO_EDGE as i32
        );
        assert_eq!(
            map_address_mode(HalAddressMode::Repeat),
            glow::REPEAT as i32
        );
        assert_eq!(
            map_address_mode(HalAddressMode::MirrorRepeat),
            glow::MIRRORED_REPEAT as i32
        );
    }

    #[test]
    fn map_min_filter_combines_min_and_mipmap_filters() {
        assert_eq!(
            map_min_filter(HalFilterMode::Nearest, HalMipmapFilterMode::Nearest),
            glow::NEAREST_MIPMAP_NEAREST as i32
        );
        assert_eq!(
            map_min_filter(HalFilterMode::Nearest, HalMipmapFilterMode::Linear),
            glow::NEAREST_MIPMAP_LINEAR as i32
        );
        assert_eq!(
            map_min_filter(HalFilterMode::Linear, HalMipmapFilterMode::Nearest),
            glow::LINEAR_MIPMAP_NEAREST as i32
        );
        assert_eq!(
            map_min_filter(HalFilterMode::Linear, HalMipmapFilterMode::Linear),
            glow::LINEAR_MIPMAP_LINEAR as i32
        );
    }

    #[test]
    fn map_compare_function_maps_known_values() {
        assert_eq!(
            map_compare_function(HalCompareFunction::Never),
            glow::NEVER as i32
        );
        assert_eq!(
            map_compare_function(HalCompareFunction::Less),
            glow::LESS as i32
        );
        assert_eq!(
            map_compare_function(HalCompareFunction::Equal),
            glow::EQUAL as i32
        );
        assert_eq!(
            map_compare_function(HalCompareFunction::LessEqual),
            glow::LEQUAL as i32
        );
        assert_eq!(
            map_compare_function(HalCompareFunction::Greater),
            glow::GREATER as i32
        );
        assert_eq!(
            map_compare_function(HalCompareFunction::NotEqual),
            glow::NOTEQUAL as i32
        );
        assert_eq!(
            map_compare_function(HalCompareFunction::GreaterEqual),
            glow::GEQUAL as i32
        );
        assert_eq!(
            map_compare_function(HalCompareFunction::Always),
            glow::ALWAYS as i32
        );
    }
}
