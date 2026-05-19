use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    HalAddressMode, HalBoundBuffer, HalBuffer, HalBufferTextureCopy, HalCompareFunction,
    HalComputePass, HalCopy, HalDescriptorBinding, HalDraw, HalError, HalExtent3d, HalFilterMode,
    HalMipmapFilterMode, HalPrimitiveTopology, HalRenderLoadOp, HalRenderPass,
    HalRenderPipelineDescriptor, HalSamplerDescriptor, HalShaderSource, HalTexture, HalTextureCopy,
    HalTextureDescriptor, HalTextureFormat, HalTextureUsage, HalVertexFormat, HalVertexStepMode,
};

const BACKEND: &str = "metal";

#[derive(Debug)]
pub struct MetalInstance;

impl MetalInstance {
    pub fn new() -> Result<Self, HalError> {
        Ok(Self)
    }

    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<MetalAdapter> {
        metal::Device::all()
            .into_iter()
            .map(MetalAdapter::new)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct MetalAdapter {
    device: metal::Device,
    name: String,
}

impl MetalAdapter {
    #[must_use]
    pub fn new(device: metal::Device) -> Self {
        let name = device.name().to_owned();
        Self { device, name }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn create_device(&self) -> Result<MetalDevice, HalError> {
        let queue = self.device.new_command_queue();
        Ok(MetalDevice {
            _device: self.device.clone(),
            allocations: AtomicU64::new(0),
            queue: MetalQueue { inner: queue },
        })
    }
}

#[derive(Debug)]
pub struct MetalDevice {
    _device: metal::Device,
    allocations: AtomicU64,
    queue: MetalQueue,
}

impl MetalDevice {
    pub fn new() -> Result<Self, HalError> {
        let device = metal::Device::system_default()
            .ok_or(HalError::BackendUnavailable { backend: BACKEND })?;
        MetalAdapter::new(device).create_device()
    }

    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn queue(&self) -> &MetalQueue {
        &self.queue
    }

    #[must_use]
    pub fn create_buffer(&self, size: u64) -> MetalBuffer {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let buffer = self
            ._device
            .new_buffer(size, metal::MTLResourceOptions::StorageModeShared);
        MetalBuffer {
            inner: Some(buffer),
            size,
        }
    }

    #[must_use]
    pub fn create_texture(&self, descriptor: &HalTextureDescriptor) -> MetalTexture {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        match create_texture(&self._device, descriptor) {
            Ok((inner, bytes_per_pixel)) => MetalTexture {
                inner: Some(inner),
                width: descriptor.width,
                height: descriptor.height,
                depth_or_array_layers: descriptor.depth_or_array_layers,
                bytes_per_pixel,
            },
            Err(_) => MetalTexture {
                inner: None,
                width: descriptor.width,
                height: descriptor.height,
                depth_or_array_layers: descriptor.depth_or_array_layers,
                bytes_per_pixel: 0,
            },
        }
    }

    #[must_use]
    pub fn create_sampler(&self, descriptor: &HalSamplerDescriptor) -> MetalSampler {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        MetalSampler {
            _inner: create_sampler(&self._device, descriptor).ok(),
        }
    }

    pub fn create_compute_pipeline(
        &self,
        shader: HalShaderSource,
        entry_point: &str,
        workgroup_size: (u32, u32, u32),
        _bindings: &[HalDescriptorBinding],
    ) -> Result<MetalComputePipeline, HalError> {
        let HalShaderSource::Msl(msl_source) = shader else {
            return Err(shader_error(
                "Metal compute pipeline requires MSL".to_owned(),
            ));
        };
        create_compute_pipeline(&self._device, &msl_source, entry_point, workgroup_size)
    }

    pub fn create_render_pipeline(
        &self,
        shader: HalShaderSource,
        vertex_entry_point: &str,
        fragment_entry_point: &str,
        descriptor: &HalRenderPipelineDescriptor,
        _bindings: &[HalDescriptorBinding],
    ) -> Result<MetalRenderPipeline, HalError> {
        let HalShaderSource::Msl(msl_source) = shader else {
            return Err(shader_error(
                "Metal render pipeline requires MSL".to_owned(),
            ));
        };
        create_render_pipeline(
            &self._device,
            &msl_source,
            vertex_entry_point,
            fragment_entry_point,
            descriptor,
        )
    }
}

#[derive(Debug, Clone)]
pub struct MetalQueue {
    inner: metal::CommandQueue,
}

impl MetalQueue {
    pub fn new() -> Result<Self, HalError> {
        let device = metal::Device::system_default()
            .ok_or(HalError::BackendUnavailable { backend: BACKEND })?;
        let inner = device.new_command_queue();
        Ok(Self { inner })
    }

    pub fn submit_empty(&self) -> Result<(), HalError> {
        let command_buffer = self.inner.new_command_buffer();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }

    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
        if copies.is_empty() {
            return Ok(());
        }

        let command_buffer = self.inner.new_command_buffer();
        for copy in copies {
            match copy {
                HalCopy::Buffer(copy) => {
                    let blit = command_buffer.new_blit_command_encoder();
                    encode_buffer_copy(blit, copy)?;
                    blit.end_encoding();
                }
                HalCopy::BufferToTexture(copy) => {
                    let blit = command_buffer.new_blit_command_encoder();
                    encode_buffer_to_texture(blit, copy)?;
                    blit.end_encoding();
                }
                HalCopy::TextureToBuffer(copy) => {
                    let blit = command_buffer.new_blit_command_encoder();
                    encode_texture_to_buffer(blit, copy)?;
                    blit.end_encoding();
                }
                HalCopy::TextureToTexture(copy) => {
                    let blit = command_buffer.new_blit_command_encoder();
                    encode_texture_to_texture(blit, copy)?;
                    blit.end_encoding();
                }
                HalCopy::ComputePass(pass) => {
                    let encoder = command_buffer.new_compute_command_encoder();
                    encode_compute_pass(encoder, pass)?;
                    encoder.end_encoding();
                }
                HalCopy::RenderPass(pass) => {
                    let descriptor = render_pass_descriptor(pass)?;
                    let encoder = command_buffer.new_render_command_encoder(descriptor);
                    encode_render_pass(encoder, pass)?;
                    encoder.end_encoding();
                }
            }
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MetalBuffer {
    inner: Option<metal::Buffer>,
    size: u64,
}

impl MetalBuffer {
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), HalError> {
        let len = u64::try_from(data.len()).map_err(|_| buffer_error("write size is too large"))?;
        self.validate_range(offset, len)?;
        if data.is_empty() {
            return Ok(());
        }
        let buffer = self.inner()?;
        let contents = buffer.contents();
        if contents.is_null() {
            return Err(buffer_error("buffer contents are unavailable"));
        }
        let offset = usize::try_from(offset).map_err(|_| buffer_error("offset is too large"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                contents.cast::<u8>().add(offset),
                data.len(),
            );
        }
        buffer.did_modify_range(metal::NSRange::new(
            to_ns(u64::try_from(offset).map_err(|_| buffer_error("offset is too large"))?)?,
            to_ns(len)?,
        ));
        Ok(())
    }

    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        self.validate_range(offset, len)?;
        let len = usize::try_from(len).map_err(|_| buffer_error("read length is too large"))?;
        let mut data = vec![0; len];
        if data.is_empty() {
            return Ok(data);
        }
        let buffer = self.inner()?;
        let contents = buffer.contents();
        if contents.is_null() {
            return Err(buffer_error("buffer contents are unavailable"));
        }
        let offset = usize::try_from(offset).map_err(|_| buffer_error("offset is too large"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents.cast::<u8>().add(offset),
                data.as_mut_ptr(),
                len,
            );
        }
        Ok(data)
    }

    fn inner(&self) -> Result<&metal::BufferRef, HalError> {
        self.inner
            .as_deref()
            .ok_or_else(|| buffer_error("buffer allocation failed"))
    }

    fn validate_range(&self, offset: u64, len: u64) -> Result<(), HalError> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| buffer_error("buffer range overflows"))?;
        if end > self.size {
            return Err(buffer_error("buffer range exceeds buffer size"));
        }
        Ok(())
    }
}

fn to_ns(value: u64) -> Result<metal::NSUInteger, HalError> {
    metal::NSUInteger::try_from(value).map_err(|_| buffer_error("value is too large"))
}

fn buffer_error(message: &'static str) -> HalError {
    HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    }
}

#[derive(Debug, Clone)]
pub struct MetalTexture {
    inner: Option<metal::Texture>,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    bytes_per_pixel: u32,
}

impl MetalTexture {
    fn inner(&self) -> Result<&metal::TextureRef, HalError> {
        self.inner
            .as_deref()
            .ok_or_else(|| texture_error("texture allocation failed or unsupported descriptor"))
    }

    fn validate_origin_extent(
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

#[derive(Debug, Clone)]
pub struct MetalSampler {
    _inner: Option<metal::SamplerState>,
}

#[derive(Debug, Clone)]
pub struct MetalComputePipeline {
    inner: metal::ComputePipelineState,
    workgroup_size: (u32, u32, u32),
}

#[derive(Debug, Clone)]
pub struct MetalRenderPipeline {
    inner: metal::RenderPipelineState,
    primitive_topology: HalPrimitiveTopology,
}

fn create_compute_pipeline(
    device: &metal::DeviceRef,
    msl_source: &str,
    entry_point: &str,
    workgroup_size: (u32, u32, u32),
) -> Result<MetalComputePipeline, HalError> {
    let options = metal::CompileOptions::new();
    let library = device
        .new_library_with_source(msl_source, &options)
        .map_err(shader_error)?;
    let function = library
        .get_function(entry_point, None)
        .map_err(shader_error)?;
    let inner = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(shader_error)?;
    Ok(MetalComputePipeline {
        inner,
        workgroup_size,
    })
}

fn create_render_pipeline(
    device: &metal::DeviceRef,
    msl_source: &str,
    vertex_entry_point: &str,
    fragment_entry_point: &str,
    descriptor: &HalRenderPipelineDescriptor,
) -> Result<MetalRenderPipeline, HalError> {
    let color_format = descriptor
        .color_formats
        .first()
        .copied()
        .ok_or_else(|| shader_error("render pipeline requires a color target".to_owned()))?;
    let options = metal::CompileOptions::new();
    let library = device
        .new_library_with_source(msl_source, &options)
        .map_err(shader_error)?;
    let vertex_function = library
        .get_function(vertex_entry_point, None)
        .map_err(shader_error)?;
    let fragment_function = library
        .get_function(fragment_entry_point, None)
        .map_err(shader_error)?;
    let pipeline_descriptor = metal::RenderPipelineDescriptor::new();
    pipeline_descriptor.set_vertex_function(Some(&vertex_function));
    pipeline_descriptor.set_fragment_function(Some(&fragment_function));
    let (pixel_format, _) = map_texture_format(color_format)?;
    let color = pipeline_descriptor
        .color_attachments()
        .object_at(0)
        .ok_or_else(|| shader_error("render color attachment slot 0 is unavailable".to_owned()))?;
    color.set_pixel_format(pixel_format);
    let vertex_descriptor = metal::VertexDescriptor::new();
    for buffer in &descriptor.vertex_buffers {
        let metal_index = buffer
            .attributes
            .first()
            .map(|attribute| attribute.metal_buffer_index)
            .unwrap_or(0);
        let layout = vertex_descriptor
            .layouts()
            .object_at(to_ns(u64::from(metal_index))?)
            .ok_or_else(|| shader_error("vertex buffer layout slot is unavailable".to_owned()))?;
        layout.set_stride(to_ns(buffer.array_stride)?);
        layout.set_step_function(match buffer.step_mode {
            HalVertexStepMode::Vertex => metal::MTLVertexStepFunction::PerVertex,
            HalVertexStepMode::Instance => metal::MTLVertexStepFunction::PerInstance,
        });
        layout.set_step_rate(1);
        for attribute in &buffer.attributes {
            let attr = vertex_descriptor
                .attributes()
                .object_at(to_ns(u64::from(attribute.shader_location))?)
                .ok_or_else(|| {
                    shader_error("vertex attribute descriptor slot is unavailable".to_owned())
                })?;
            attr.set_format(map_vertex_format(attribute.format)?);
            attr.set_offset(to_ns(attribute.offset)?);
            attr.set_buffer_index(to_ns(u64::from(attribute.metal_buffer_index))?);
        }
    }
    pipeline_descriptor.set_vertex_descriptor(Some(vertex_descriptor));
    let inner = device
        .new_render_pipeline_state(&pipeline_descriptor)
        .map_err(shader_error)?;
    Ok(MetalRenderPipeline {
        inner,
        primitive_topology: descriptor.primitive_topology,
    })
}

fn create_texture(
    device: &metal::DeviceRef,
    descriptor: &HalTextureDescriptor,
) -> Result<(metal::Texture, u32), HalError> {
    if descriptor.depth_or_array_layers != 1
        || descriptor.mip_level_count != 1
        || descriptor.sample_count != 1
    {
        return Err(texture_error("unsupported texture descriptor"));
    }
    let (pixel_format, bytes_per_pixel) = map_texture_format(descriptor.format)?;
    let texture_descriptor = metal::TextureDescriptor::new();
    texture_descriptor.set_texture_type(metal::MTLTextureType::D2);
    texture_descriptor.set_pixel_format(pixel_format);
    texture_descriptor.set_width(to_ns(u64::from(descriptor.width))?);
    texture_descriptor.set_height(to_ns(u64::from(descriptor.height))?);
    texture_descriptor.set_depth(1);
    texture_descriptor.set_array_length(1);
    texture_descriptor.set_mipmap_level_count(1);
    texture_descriptor.set_sample_count(1);
    texture_descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
    texture_descriptor.set_usage(map_texture_usage(descriptor.usage));
    Ok((device.new_texture(&texture_descriptor), bytes_per_pixel))
}

fn create_sampler(
    device: &metal::DeviceRef,
    descriptor: &HalSamplerDescriptor,
) -> Result<metal::SamplerState, HalError> {
    let sampler_descriptor = metal::SamplerDescriptor::new();
    sampler_descriptor.set_address_mode_s(map_address_mode(descriptor.address_mode_u));
    sampler_descriptor.set_address_mode_t(map_address_mode(descriptor.address_mode_v));
    sampler_descriptor.set_address_mode_r(map_address_mode(descriptor.address_mode_w));
    sampler_descriptor.set_mag_filter(map_filter_mode(descriptor.mag_filter));
    sampler_descriptor.set_min_filter(map_filter_mode(descriptor.min_filter));
    sampler_descriptor.set_mip_filter(map_mipmap_filter_mode(descriptor.mipmap_filter));
    sampler_descriptor.set_lod_min_clamp(descriptor.lod_min_clamp);
    sampler_descriptor.set_lod_max_clamp(descriptor.lod_max_clamp);
    sampler_descriptor.set_max_anisotropy(to_ns(u64::from(descriptor.max_anisotropy))?);
    if let Some(compare) = descriptor.compare {
        sampler_descriptor.set_compare_function(map_compare_function(compare));
    }
    Ok(device.new_sampler(&sampler_descriptor))
}

fn encode_buffer_copy(
    blit: &metal::BlitCommandEncoderRef,
    copy: &crate::HalBufferCopy,
) -> Result<(), HalError> {
    let HalBuffer::Metal(source) = &copy.source else {
        return Err(buffer_error("source buffer is not Metal-backed"));
    };
    let HalBuffer::Metal(destination) = &copy.destination else {
        return Err(buffer_error("destination buffer is not Metal-backed"));
    };
    source.validate_range(copy.source_offset, copy.size)?;
    destination.validate_range(copy.destination_offset, copy.size)?;
    blit.copy_from_buffer(
        source.inner()?,
        to_ns(copy.source_offset)?,
        destination.inner()?,
        to_ns(copy.destination_offset)?,
        to_ns(copy.size)?,
    );
    Ok(())
}

fn encode_buffer_to_texture(
    blit: &metal::BlitCommandEncoderRef,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &copy.buffer else {
        return Err(buffer_error("buffer is not Metal-backed"));
    };
    let HalTexture::Metal(texture) = &copy.texture else {
        return Err(texture_error("texture is not Metal-backed"));
    };
    texture.validate_origin_extent(copy.origin, copy.extent)?;
    validate_buffer_texture_range(buffer, copy)?;
    blit.copy_from_buffer_to_texture(
        buffer.inner()?,
        to_ns(copy.buffer_layout.offset)?,
        to_ns(u64::from(copy.buffer_layout.bytes_per_row))?,
        buffer_texture_bytes_per_image(copy)?,
        to_mtl_size(copy.extent)?,
        texture.inner()?,
        to_ns(u64::from(copy.origin.z))?,
        to_ns(u64::from(copy.mip_level))?,
        to_mtl_origin(copy.origin.x, copy.origin.y, 0)?,
        metal::MTLBlitOption::None,
    );
    Ok(())
}

fn encode_texture_to_buffer(
    blit: &metal::BlitCommandEncoderRef,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &copy.buffer else {
        return Err(buffer_error("buffer is not Metal-backed"));
    };
    let HalTexture::Metal(texture) = &copy.texture else {
        return Err(texture_error("texture is not Metal-backed"));
    };
    texture.validate_origin_extent(copy.origin, copy.extent)?;
    validate_buffer_texture_range(buffer, copy)?;
    blit.copy_from_texture_to_buffer(
        texture.inner()?,
        to_ns(u64::from(copy.origin.z))?,
        to_ns(u64::from(copy.mip_level))?,
        to_mtl_origin(copy.origin.x, copy.origin.y, 0)?,
        to_mtl_size(copy.extent)?,
        buffer.inner()?,
        to_ns(copy.buffer_layout.offset)?,
        to_ns(u64::from(copy.buffer_layout.bytes_per_row))?,
        buffer_texture_bytes_per_image(copy)?,
        metal::MTLBlitOption::None,
    );
    Ok(())
}

fn encode_texture_to_texture(
    blit: &metal::BlitCommandEncoderRef,
    copy: &HalTextureCopy,
) -> Result<(), HalError> {
    let HalTexture::Metal(source) = &copy.source else {
        return Err(texture_error("source texture is not Metal-backed"));
    };
    let HalTexture::Metal(destination) = &copy.destination else {
        return Err(texture_error("destination texture is not Metal-backed"));
    };
    source.validate_origin_extent(copy.source_origin, copy.extent)?;
    destination.validate_origin_extent(copy.destination_origin, copy.extent)?;
    blit.copy_from_texture(
        source.inner()?,
        to_ns(u64::from(copy.source_origin.z))?,
        to_ns(u64::from(copy.source_mip_level))?,
        to_mtl_origin(copy.source_origin.x, copy.source_origin.y, 0)?,
        to_mtl_size(copy.extent)?,
        destination.inner()?,
        to_ns(u64::from(copy.destination_origin.z))?,
        to_ns(u64::from(copy.destination_mip_level))?,
        to_mtl_origin(copy.destination_origin.x, copy.destination_origin.y, 0)?,
    );
    Ok(())
}

fn encode_compute_pass(
    encoder: &metal::ComputeCommandEncoderRef,
    pass: &HalComputePass,
) -> Result<(), HalError> {
    let crate::HalComputePipeline::Metal(pipeline) = &pass.pipeline else {
        return Err(shader_error(
            "compute pipeline is not Metal-backed".to_owned(),
        ));
    };
    encoder.set_compute_pipeline_state(&pipeline.inner);
    for binding in &pass.bind_buffers {
        encode_compute_buffer(encoder, binding)?;
    }
    encoder.dispatch_thread_groups(
        to_mtl_dispatch_size(pass.workgroups)?,
        to_mtl_workgroup_size(pipeline.workgroup_size)?,
    );
    Ok(())
}

fn encode_compute_buffer(
    encoder: &metal::ComputeCommandEncoderRef,
    binding: &HalBoundBuffer,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &binding.buffer else {
        return Err(buffer_error("compute buffer is not Metal-backed"));
    };
    if binding.offset > buffer.size() {
        return Err(buffer_error("compute buffer offset exceeds buffer size"));
    }
    encoder.set_buffer(
        to_ns(u64::from(binding.metal_index))?,
        Some(buffer.inner()?),
        to_ns(binding.offset)?,
    );
    Ok(())
}

fn render_pass_descriptor(
    pass: &HalRenderPass,
) -> Result<&metal::RenderPassDescriptorRef, HalError> {
    let HalTexture::Metal(texture) = &pass.color_target.texture else {
        return Err(texture_error("render target is not Metal-backed"));
    };
    let descriptor = metal::RenderPassDescriptor::new();
    let color = descriptor
        .color_attachments()
        .object_at(0)
        .ok_or_else(|| texture_error("render color attachment slot 0 is unavailable"))?;
    color.set_texture(Some(texture.inner()?));
    color.set_load_action(match pass.color_target.load_op {
        HalRenderLoadOp::Load => metal::MTLLoadAction::Load,
        HalRenderLoadOp::Clear => metal::MTLLoadAction::Clear,
    });
    color.set_store_action(if pass.color_target.store {
        metal::MTLStoreAction::Store
    } else {
        metal::MTLStoreAction::DontCare
    });
    let [r, g, b, a] = pass.color_target.clear_color;
    color.set_clear_color(metal::MTLClearColor::new(r, g, b, a));
    Ok(descriptor)
}

fn encode_render_pass(
    encoder: &metal::RenderCommandEncoderRef,
    pass: &HalRenderPass,
) -> Result<(), HalError> {
    let crate::HalRenderPipeline::Metal(pipeline) = &pass.pipeline else {
        return Err(shader_error(
            "render pipeline is not Metal-backed".to_owned(),
        ));
    };
    encoder.set_render_pipeline_state(&pipeline.inner);
    for binding in &pass.bind_buffers {
        encode_render_bind_buffer(encoder, binding)?;
    }
    for binding in &pass.vertex_buffers {
        encode_render_vertex_buffer(encoder, binding)?;
    }
    draw_primitives(encoder, pipeline.primitive_topology, pass.draw)?;
    Ok(())
}

fn encode_render_bind_buffer(
    encoder: &metal::RenderCommandEncoderRef,
    binding: &HalBoundBuffer,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &binding.buffer else {
        return Err(buffer_error("render bind buffer is not Metal-backed"));
    };
    if binding.offset > buffer.size() {
        return Err(buffer_error(
            "render bind buffer offset exceeds buffer size",
        ));
    }
    let index = to_ns(u64::from(binding.metal_index))?;
    let offset = to_ns(binding.offset)?;
    encoder.set_vertex_buffer(index, Some(buffer.inner()?), offset);
    encoder.set_fragment_buffer(index, Some(buffer.inner()?), offset);
    Ok(())
}

fn encode_render_vertex_buffer(
    encoder: &metal::RenderCommandEncoderRef,
    binding: &HalBoundBuffer,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &binding.buffer else {
        return Err(buffer_error("render vertex buffer is not Metal-backed"));
    };
    if binding.offset > buffer.size() {
        return Err(buffer_error(
            "render vertex buffer offset exceeds buffer size",
        ));
    }
    encoder.set_vertex_buffer(
        to_ns(u64::from(binding.metal_index))?,
        Some(buffer.inner()?),
        to_ns(binding.offset)?,
    );
    Ok(())
}

fn draw_primitives(
    encoder: &metal::RenderCommandEncoderRef,
    topology: HalPrimitiveTopology,
    draw: HalDraw,
) -> Result<(), HalError> {
    encoder.draw_primitives_instanced_base_instance(
        map_primitive_topology(topology),
        to_ns(u64::from(draw.first_vertex))?,
        to_ns(u64::from(draw.vertex_count))?,
        to_ns(u64::from(draw.instance_count))?,
        to_ns(u64::from(draw.first_instance))?,
    );
    Ok(())
}

fn validate_buffer_texture_range(
    buffer: &MetalBuffer,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let rows = u64::from(copy.extent.height.saturating_sub(1));
    let last_row = rows
        .checked_mul(u64::from(copy.buffer_layout.bytes_per_row))
        .ok_or_else(|| buffer_error("buffer texture row range overflows"))?;
    let row_bytes = u64::from(copy.extent.width)
        .checked_mul(u64::from(texture_bytes_per_pixel(copy)?))
        .ok_or_else(|| buffer_error("buffer texture row bytes overflow"))?;
    let required = copy
        .buffer_layout
        .offset
        .checked_add(last_row)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or_else(|| buffer_error("buffer texture range overflows"))?;
    if required > buffer.size() {
        return Err(buffer_error("buffer texture range exceeds buffer size"));
    }
    Ok(())
}

fn texture_bytes_per_pixel(copy: &HalBufferTextureCopy) -> Result<u32, HalError> {
    let HalTexture::Metal(texture) = &copy.texture else {
        return Err(texture_error("texture is not Metal-backed"));
    };
    if texture.bytes_per_pixel == 0 {
        return Err(texture_error("unsupported texture format"));
    }
    Ok(texture.bytes_per_pixel)
}

fn buffer_texture_bytes_per_image(
    copy: &HalBufferTextureCopy,
) -> Result<metal::NSUInteger, HalError> {
    let bytes = u64::from(copy.buffer_layout.bytes_per_row)
        .checked_mul(u64::from(copy.buffer_layout.rows_per_image))
        .ok_or_else(|| buffer_error("buffer texture bytes per image overflows"))?;
    to_ns(bytes)
}

fn to_mtl_origin(x: u32, y: u32, z: u32) -> Result<metal::MTLOrigin, HalError> {
    Ok(metal::MTLOrigin {
        x: to_ns(u64::from(x))?,
        y: to_ns(u64::from(y))?,
        z: to_ns(u64::from(z))?,
    })
}

fn to_mtl_size(extent: HalExtent3d) -> Result<metal::MTLSize, HalError> {
    Ok(metal::MTLSize::new(
        to_ns(u64::from(extent.width))?,
        to_ns(u64::from(extent.height))?,
        to_ns(u64::from(extent.depth_or_array_layers))?,
    ))
}

fn to_mtl_dispatch_size(size: (u32, u32, u32)) -> Result<metal::MTLSize, HalError> {
    Ok(metal::MTLSize::new(
        to_ns(u64::from(size.0))?,
        to_ns(u64::from(size.1))?,
        to_ns(u64::from(size.2))?,
    ))
}

fn to_mtl_workgroup_size(size: (u32, u32, u32)) -> Result<metal::MTLSize, HalError> {
    to_mtl_dispatch_size(size)
}

fn map_texture_format(format: HalTextureFormat) -> Result<(metal::MTLPixelFormat, u32), HalError> {
    match format {
        HalTextureFormat::R8Unorm => Ok((metal::MTLPixelFormat::R8Unorm, 1)),
        HalTextureFormat::Rgba8Unorm => Ok((metal::MTLPixelFormat::RGBA8Unorm, 4)),
        HalTextureFormat::Bgra8Unorm => Ok((metal::MTLPixelFormat::BGRA8Unorm, 4)),
        HalTextureFormat::Unsupported => Err(texture_error("unsupported texture format")),
    }
}

fn map_texture_usage(usage: HalTextureUsage) -> metal::MTLTextureUsage {
    let mut metal_usage = metal::MTLTextureUsage::Unknown;
    if usage.copy_src || usage.texture_binding {
        metal_usage |= metal::MTLTextureUsage::ShaderRead;
    }
    if usage.copy_dst || usage.storage_binding {
        metal_usage |= metal::MTLTextureUsage::ShaderWrite;
    }
    if usage.render_attachment {
        metal_usage |= metal::MTLTextureUsage::RenderTarget;
    }
    metal_usage
}

fn map_address_mode(mode: HalAddressMode) -> metal::MTLSamplerAddressMode {
    match mode {
        HalAddressMode::ClampToEdge => metal::MTLSamplerAddressMode::ClampToEdge,
        HalAddressMode::Repeat => metal::MTLSamplerAddressMode::Repeat,
        HalAddressMode::MirrorRepeat => metal::MTLSamplerAddressMode::MirrorRepeat,
    }
}

fn map_filter_mode(mode: HalFilterMode) -> metal::MTLSamplerMinMagFilter {
    match mode {
        HalFilterMode::Nearest => metal::MTLSamplerMinMagFilter::Nearest,
        HalFilterMode::Linear => metal::MTLSamplerMinMagFilter::Linear,
    }
}

fn map_mipmap_filter_mode(mode: HalMipmapFilterMode) -> metal::MTLSamplerMipFilter {
    match mode {
        HalMipmapFilterMode::Nearest => metal::MTLSamplerMipFilter::Nearest,
        HalMipmapFilterMode::Linear => metal::MTLSamplerMipFilter::Linear,
    }
}

fn map_compare_function(compare: HalCompareFunction) -> metal::MTLCompareFunction {
    match compare {
        HalCompareFunction::Never => metal::MTLCompareFunction::Never,
        HalCompareFunction::Less => metal::MTLCompareFunction::Less,
        HalCompareFunction::Equal => metal::MTLCompareFunction::Equal,
        HalCompareFunction::LessEqual => metal::MTLCompareFunction::LessEqual,
        HalCompareFunction::Greater => metal::MTLCompareFunction::Greater,
        HalCompareFunction::NotEqual => metal::MTLCompareFunction::NotEqual,
        HalCompareFunction::GreaterEqual => metal::MTLCompareFunction::GreaterEqual,
        HalCompareFunction::Always => metal::MTLCompareFunction::Always,
    }
}

fn map_vertex_format(format: HalVertexFormat) -> Result<metal::MTLVertexFormat, HalError> {
    match format {
        HalVertexFormat::Float32 => Ok(metal::MTLVertexFormat::Float),
        HalVertexFormat::Float32x2 => Ok(metal::MTLVertexFormat::Float2),
        HalVertexFormat::Float32x3 => Ok(metal::MTLVertexFormat::Float3),
        HalVertexFormat::Float32x4 => Ok(metal::MTLVertexFormat::Float4),
        HalVertexFormat::Unsupported => Err(shader_error(
            "unsupported vertex format for Metal".to_owned(),
        )),
    }
}

fn map_primitive_topology(topology: HalPrimitiveTopology) -> metal::MTLPrimitiveType {
    match topology {
        HalPrimitiveTopology::PointList => metal::MTLPrimitiveType::Point,
        HalPrimitiveTopology::LineList => metal::MTLPrimitiveType::Line,
        HalPrimitiveTopology::LineStrip => metal::MTLPrimitiveType::LineStrip,
        HalPrimitiveTopology::TriangleList => metal::MTLPrimitiveType::Triangle,
        HalPrimitiveTopology::TriangleStrip => metal::MTLPrimitiveType::TriangleStrip,
    }
}

fn texture_error(message: &'static str) -> HalError {
    HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    }
}

fn shader_error(message: String) -> HalError {
    HalError::ShaderCompilationFailed {
        backend: BACKEND,
        message,
    }
}
