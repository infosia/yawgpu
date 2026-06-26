use super::*;
use crate::HalTextureDimension;

/// Stores metal texture data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalTexture {
    pub(super) inner: Option<Retained<ProtocolObject<dyn MTLTextureTrait>>>,
    pub(super) format: HalTextureFormat,
    pub(super) dimension: HalTextureDimension,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) depth_or_array_layers: u32,
    pub(super) sample_count: u32,
    pub(super) bytes_per_pixel: u32,
}

unsafe impl Send for MetalTexture {}
unsafe impl Sync for MetalTexture {}

impl std::fmt::Debug for MetalTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalTexture")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("depth_or_array_layers", &self.depth_or_array_layers)
            .field("sample_count", &self.sample_count)
            .finish()
    }
}

impl MetalTexture {
    /// Returns the backing `MTLTexture`, or an error if creation failed.
    pub(super) fn inner(&self) -> Result<&ProtocolObject<dyn MTLTextureTrait>, HalError> {
        self.inner
            .as_deref()
            .ok_or_else(|| texture_error("texture allocation failed or unsupported descriptor"))
    }

    /// Validates origin extent and returns a descriptive error on failure.
    pub(super) fn validate_origin_extent(
        &self,
        origin: crate::HalOrigin3d,
        extent: HalExtent3d,
    ) -> Result<(), HalError> {
        let x_end = origin
            .x
            .checked_add(extent.width)
            .ok_or_else(|| texture_error("texture x range overflows"))?;
        let y_end = origin
            .y
            .checked_add(extent.height)
            .ok_or_else(|| texture_error("texture y range overflows"))?;
        let z_end = origin
            .z
            .checked_add(extent.depth_or_array_layers)
            .ok_or_else(|| texture_error("texture z range overflows"))?;
        if x_end > self.width || y_end > self.height || z_end > self.depth_or_array_layers {
            return Err(texture_error("texture range exceeds texture size"));
        }
        Ok(())
    }
}

/// Stores metal sampler data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalSampler {
    pub(super) _inner: Option<Retained<ProtocolObject<dyn MTLSamplerState>>>,
}

unsafe impl Send for MetalSampler {}
unsafe impl Sync for MetalSampler {}

impl std::fmt::Debug for MetalSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalSampler").finish_non_exhaustive()
    }
}

/// Creates texture and reports validation errors through the owning device.
pub(super) fn create_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    descriptor: &HalTextureDescriptor,
) -> Result<(Retained<ProtocolObject<dyn MTLTextureTrait>>, u32), HalError> {
    if descriptor.sample_count != 1 && descriptor.sample_count != 4 {
        return Err(texture_error("unsupported texture sample count"));
    }
    if descriptor.sample_count > 1
        && (descriptor.dimension != HalTextureDimension::D2
            || descriptor.depth_or_array_layers != 1)
    {
        return Err(texture_error(
            "multisample texture must be a single-layer 2D texture",
        ));
    }
    let (pixel_format, bytes_per_pixel) = map_texture_format(descriptor.format)?;
    let texture_descriptor = MTLTextureDescriptor::new();
    texture_descriptor.setTextureType(match descriptor.dimension {
        HalTextureDimension::D2 if descriptor.sample_count > 1 => MTLTextureType::Type2DMultisample,
        HalTextureDimension::D1 => MTLTextureType::Type1D,
        HalTextureDimension::D2 if descriptor.depth_or_array_layers > 1 => {
            MTLTextureType::Type2DArray
        }
        HalTextureDimension::D2 => MTLTextureType::Type2D,
        HalTextureDimension::D3 => MTLTextureType::Type3D,
    });
    texture_descriptor.setPixelFormat(pixel_format);
    unsafe {
        texture_descriptor.setWidth(to_ns(u64::from(descriptor.width))?);
        texture_descriptor.setHeight(to_ns(u64::from(match descriptor.dimension {
            HalTextureDimension::D1 => 1,
            HalTextureDimension::D2 | HalTextureDimension::D3 => descriptor.height,
        }))?);
        texture_descriptor.setDepth(to_ns(u64::from(match descriptor.dimension {
            HalTextureDimension::D3 => descriptor.depth_or_array_layers,
            HalTextureDimension::D1 | HalTextureDimension::D2 => 1,
        }))?);
        texture_descriptor.setArrayLength(to_ns(u64::from(match descriptor.dimension {
            HalTextureDimension::D2 if descriptor.sample_count == 1 => {
                descriptor.depth_or_array_layers
            }
            HalTextureDimension::D1 | HalTextureDimension::D3 => 1,
            HalTextureDimension::D2 => 1,
        }))?);
        texture_descriptor.setMipmapLevelCount(to_ns(u64::from(descriptor.mip_level_count))?);
        texture_descriptor.setSampleCount(to_ns(u64::from(descriptor.sample_count))?);
    }
    texture_descriptor.setStorageMode(if descriptor.sample_count > 1 {
        MTLStorageMode::Private
    } else {
        MTLStorageMode::Shared
    });
    let mut texture_usage = map_texture_usage(descriptor.usage);
    if is_combined_depth_stencil(descriptor.format) {
        // Allow a stencil-only reinterpret view (`X32_Stencil8`) for sampling.
        texture_usage |= MTLTextureUsage::PixelFormatView;
    }
    texture_descriptor.setUsage(texture_usage);
    let texture =
        device
            .newTextureWithDescriptor(&texture_descriptor)
            .ok_or(HalError::OutOfMemory {
                backend: BACKEND,
                resource: "texture",
            })?;
    Ok((texture, bytes_per_pixel))
}

/// Metal's documented maximum for `setMaxAnisotropy` (API range [1, 16]).
/// Values above this are clamped — WebGPU requires clamping, not an error.
const METAL_MAX_ANISOTROPY: u16 = 16;

/// Clamps a requested anisotropy value to Metal's documented [1, 16] range.
/// WebGPU semantics: values above the platform maximum are silently clamped.
pub(super) fn clamp_anisotropy(requested: u16) -> u16 {
    requested.clamp(1, METAL_MAX_ANISOTROPY)
}

/// Creates sampler and reports validation errors through the owning device.
pub(super) fn create_sampler(
    device: &ProtocolObject<dyn MTLDevice>,
    descriptor: &HalSamplerDescriptor,
) -> Result<Retained<ProtocolObject<dyn MTLSamplerState>>, HalError> {
    let sampler_descriptor = MTLSamplerDescriptor::new();
    sampler_descriptor.setSAddressMode(map_address_mode(descriptor.address_mode_u));
    sampler_descriptor.setTAddressMode(map_address_mode(descriptor.address_mode_v));
    sampler_descriptor.setRAddressMode(map_address_mode(descriptor.address_mode_w));
    sampler_descriptor.setMagFilter(map_filter_mode(descriptor.mag_filter));
    sampler_descriptor.setMinFilter(map_filter_mode(descriptor.min_filter));
    sampler_descriptor.setMipFilter(map_mipmap_filter_mode(descriptor.mipmap_filter));
    sampler_descriptor.setLodMinClamp(descriptor.lod_min_clamp);
    sampler_descriptor.setLodMaxClamp(descriptor.lod_max_clamp);
    // Clamp to Metal's [1, 16] range; WebGPU requires clamping, not an error.
    let clamped = clamp_anisotropy(descriptor.max_anisotropy);
    sampler_descriptor.setMaxAnisotropy(to_ns(u64::from(clamped))?);
    if let Some(compare) = descriptor.compare {
        sampler_descriptor.setCompareFunction(map_compare_function(compare));
    }
    device
        .newSamplerStateWithDescriptor(&sampler_descriptor)
        .ok_or_else(|| texture_error("sampler allocation failed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_anisotropy_clamps_above_max_to_16() {
        assert_eq!(clamp_anisotropy(1024), 16);
        assert_eq!(clamp_anisotropy(17), 16);
    }

    #[test]
    fn clamp_anisotropy_passes_through_values_within_range() {
        assert_eq!(clamp_anisotropy(16), 16);
        assert_eq!(clamp_anisotropy(8), 8);
        assert_eq!(clamp_anisotropy(1), 1);
    }

    #[test]
    fn clamp_anisotropy_clamps_zero_to_one() {
        // core validation rejects 0, but clamp_anisotropy is a pure helper;
        // verify it doesn't panic and returns the floor.
        assert_eq!(clamp_anisotropy(0), 1);
    }
}
