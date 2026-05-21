use super::*;

/// Stores metal texture data used by validation and backend submission.
#[derive(Clone)]
pub struct MetalTexture {
    pub(super) inner: Option<Retained<ProtocolObject<dyn MTLTextureTrait>>>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) depth_or_array_layers: u32,
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
    if descriptor.depth_or_array_layers != 1
        || descriptor.mip_level_count != 1
        || descriptor.sample_count != 1
    {
        return Err(texture_error("unsupported texture descriptor"));
    }
    let (pixel_format, bytes_per_pixel) = map_texture_format(descriptor.format)?;
    let texture_descriptor = MTLTextureDescriptor::new();
    texture_descriptor.setTextureType(MTLTextureType::Type2D);
    texture_descriptor.setPixelFormat(pixel_format);
    unsafe {
        texture_descriptor.setWidth(to_ns(u64::from(descriptor.width))?);
        texture_descriptor.setHeight(to_ns(u64::from(descriptor.height))?);
        texture_descriptor.setDepth(1);
        texture_descriptor.setArrayLength(1);
        texture_descriptor.setMipmapLevelCount(1);
        texture_descriptor.setSampleCount(1);
    }
    texture_descriptor.setStorageMode(MTLStorageMode::Shared);
    texture_descriptor.setUsage(map_texture_usage(descriptor.usage));
    let texture = device
        .newTextureWithDescriptor(&texture_descriptor)
        .ok_or_else(|| texture_error("texture allocation failed"))?;
    Ok((texture, bytes_per_pixel))
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
    sampler_descriptor.setMaxAnisotropy(to_ns(u64::from(descriptor.max_anisotropy))?);
    if let Some(compare) = descriptor.compare {
        sampler_descriptor.setCompareFunction(map_compare_function(compare));
    }
    device
        .newSamplerStateWithDescriptor(&sampler_descriptor)
        .ok_or_else(|| texture_error("sampler allocation failed"))
}
