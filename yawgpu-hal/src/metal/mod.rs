use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::CGSize;
use objc2_foundation::{NSArray, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer as MTLBufferTrait, MTLClearColor, MTLCommandBuffer,
    MTLCommandEncoder, MTLCommandQueue, MTLCompareFunction, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCopyAllDevices, MTLDevice, MTLDrawable, MTLLibrary, MTLLoadAction,
    MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLSamplerState, MTLSize,
    MTLStorageMode, MTLStoreAction, MTLTexture as MTLTextureTrait, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage, MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use crate::{
    HalAddressMode, HalBoundBuffer, HalBuffer, HalBufferTextureCopy, HalCompareFunction,
    HalComputePass, HalCopy, HalDescriptorBinding, HalDraw, HalError, HalExtent3d, HalFilterMode,
    HalMipmapFilterMode, HalPrimitiveTopology, HalRenderLoadOp, HalRenderPass,
    HalRenderPipelineDescriptor, HalSamplerDescriptor, HalShaderSource, HalSurfaceConfiguration,
    HalTexture, HalTextureCopy, HalTextureDescriptor, HalTextureFormat, HalTextureUsage,
    HalVertexFormat, HalVertexStepMode,
};

const BACKEND: &str = "metal";

pub struct MetalInstance;

impl std::fmt::Debug for MetalInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalInstance").finish()
    }
}

impl MetalInstance {
    pub fn new() -> Result<Self, HalError> {
        Ok(Self)
    }

    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<MetalAdapter> {
        autoreleasepool(|_| {
            let devices: Retained<NSArray<ProtocolObject<dyn MTLDevice>>> = MTLCopyAllDevices();
            devices.into_iter().map(MetalAdapter::new).collect()
        })
    }
}

#[derive(Clone)]
pub struct MetalAdapter {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    name: String,
}

impl std::fmt::Debug for MetalAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalAdapter")
            .field("name", &self.name)
            .finish()
    }
}

impl MetalAdapter {
    #[must_use]
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        let name = device.name().to_string();
        Self { device, name }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn create_device(&self) -> Result<MetalDevice, HalError> {
        let queue = self
            .device
            .newCommandQueue()
            .ok_or(HalError::DeviceCreationFailed { backend: BACKEND })?;
        Ok(MetalDevice {
            device: self.device.clone(),
            allocations: AtomicU64::new(0),
            queue: MetalQueue { inner: queue },
        })
    }
}

pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    allocations: AtomicU64,
    queue: MetalQueue,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDevice")
            .field("allocations", &self.allocation_count())
            .finish()
    }
}

impl MetalDevice {
    pub fn new() -> Result<Self, HalError> {
        let adapter = MetalInstance::new()?
            .enumerate_adapters()
            .into_iter()
            .next()
            .ok_or(HalError::BackendUnavailable { backend: BACKEND })?;
        adapter.create_device()
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
        let buffer = self.device.newBufferWithLength_options(
            usize::try_from(size).unwrap_or(usize::MAX),
            MTLResourceOptions::StorageModeShared,
        );
        let mapped_ptr = buffer.as_ref().map(|buffer| buffer.contents().cast::<u8>());
        MetalBuffer {
            inner: buffer,
            mapped_ptr,
            size,
        }
    }

    #[must_use]
    pub fn create_texture(&self, descriptor: &HalTextureDescriptor) -> MetalTexture {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        match create_texture(&self.device, descriptor) {
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
            _inner: create_sampler(&self.device, descriptor).ok(),
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
        create_compute_pipeline(&self.device, &msl_source, entry_point, workgroup_size)
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
            &self.device,
            &msl_source,
            vertex_entry_point,
            fragment_entry_point,
            descriptor,
        )
    }
}

#[derive(Debug)]
pub struct MetalSurface {
    layer: Retained<CAMetalLayer>,
    current_drawable: Option<Retained<ProtocolObject<dyn CAMetalDrawable>>>,
    config: Option<HalSurfaceConfiguration>,
}

unsafe impl Send for MetalSurface {}
unsafe impl Sync for MetalSurface {}

impl MetalSurface {
    pub fn from_layer(layer: *mut c_void) -> Result<Self, HalError> {
        let layer = unsafe { Retained::retain(layer.cast::<CAMetalLayer>()) }.ok_or(
            HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "surface layer is null",
            },
        )?;
        Ok(Self {
            layer,
            current_drawable: None,
            config: None,
        })
    }

    pub fn configure(
        &mut self,
        device: &MetalDevice,
        config: HalSurfaceConfiguration,
    ) -> Result<(), HalError> {
        let (pixel_format, _) = map_texture_format(config.format)?;
        self.layer.setDevice(Some(&device.device));
        self.layer.setPixelFormat(pixel_format);
        self.layer.setFramebufferOnly(false);
        self.layer.setDrawableSize(CGSize {
            width: f64::from(config.width),
            height: f64::from(config.height),
        });
        let _ = config.usage;
        let _ = config.present_mode;
        self.current_drawable = None;
        self.config = Some(config);
        Ok(())
    }

    pub fn unconfigure(&mut self) {
        self.current_drawable = None;
        self.config = None;
    }

    pub fn acquire_next_texture(&mut self) -> Result<MetalTexture, HalError> {
        let config = self.config.ok_or(HalError::AcquireFailed {
            backend: BACKEND,
            message: "surface is not configured",
        })?;
        let drawable = self.layer.nextDrawable().ok_or(HalError::AcquireFailed {
            backend: BACKEND,
            message: "nextDrawable returned null",
        })?;
        let texture = drawable.texture();
        self.current_drawable = Some(drawable);
        let (_, bytes_per_pixel) = map_texture_format(config.format)?;
        Ok(MetalTexture {
            inner: Some(texture),
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
            bytes_per_pixel,
        })
    }

    pub fn present(&mut self, queue: &MetalQueue) -> Result<(), HalError> {
        let drawable = self
            .current_drawable
            .take()
            .ok_or(HalError::PresentFailed {
                backend: BACKEND,
                message: "no acquired drawable to present",
            })?;
        let _ = queue;
        drawable.present();
        Ok(())
    }
}

#[derive(Clone)]
pub struct MetalQueue {
    inner: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl std::fmt::Debug for MetalQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalQueue").finish_non_exhaustive()
    }
}

impl MetalQueue {
    pub fn new() -> Result<Self, HalError> {
        Ok(MetalDevice::new()?.queue().clone())
    }

    pub fn submit_empty(&self) -> Result<(), HalError> {
        let command_buffer = self
            .inner
            .commandBuffer()
            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        Ok(())
    }

    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
        if copies.is_empty() {
            return Ok(());
        }

        autoreleasepool(|_| {
            let command_buffer = self
                .inner
                .commandBuffer()
                .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
            for copy in copies {
                match copy {
                    HalCopy::Buffer(copy) => {
                        let blit = command_buffer
                            .blitCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        encode_buffer_copy(&blit, copy)?;
                        blit.endEncoding();
                    }
                    HalCopy::BufferToTexture(copy) => {
                        let blit = command_buffer
                            .blitCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        encode_buffer_to_texture(&blit, copy)?;
                        blit.endEncoding();
                    }
                    HalCopy::TextureToBuffer(copy) => {
                        let blit = command_buffer
                            .blitCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        encode_texture_to_buffer(&blit, copy)?;
                        blit.endEncoding();
                    }
                    HalCopy::TextureToTexture(copy) => {
                        let blit = command_buffer
                            .blitCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        encode_texture_to_texture(&blit, copy)?;
                        blit.endEncoding();
                    }
                    HalCopy::ComputePass(pass) => {
                        let encoder = command_buffer
                            .computeCommandEncoder()
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        encode_compute_pass(&encoder, pass)?;
                        encoder.endEncoding();
                    }
                    HalCopy::RenderPass(pass) => {
                        let descriptor = render_pass_descriptor(pass)?;
                        let encoder = command_buffer
                            .renderCommandEncoderWithDescriptor(&descriptor)
                            .ok_or(HalError::QueueSubmissionFailed { backend: BACKEND })?;
                        encode_render_pass(&encoder, pass)?;
                        encoder.endEncoding();
                    }
                }
            }
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            Ok(())
        })
    }
}

#[derive(Clone)]
pub struct MetalBuffer {
    inner: Option<Retained<ProtocolObject<dyn MTLBufferTrait>>>,
    mapped_ptr: Option<NonNull<u8>>,
    size: u64,
}

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl std::fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("size", &self.size)
            .field("mapped", &self.mapped_ptr.is_some())
            .finish()
    }
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
        let mapped_ptr = self
            .mapped_ptr
            .ok_or_else(|| buffer_error("buffer contents are unavailable"))?;
        let offset = usize::try_from(offset).map_err(|_| buffer_error("offset is too large"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                mapped_ptr.as_ptr().add(offset),
                data.len(),
            );
        }
        Ok(())
    }

    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        self.validate_range(offset, len)?;
        let len = usize::try_from(len).map_err(|_| buffer_error("read length is too large"))?;
        let mut data = vec![0; len];
        if data.is_empty() {
            return Ok(data);
        }
        let mapped_ptr = self
            .mapped_ptr
            .ok_or_else(|| buffer_error("buffer contents are unavailable"))?;
        let offset = usize::try_from(offset).map_err(|_| buffer_error("offset is too large"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(mapped_ptr.as_ptr().add(offset), data.as_mut_ptr(), len);
        }
        Ok(data)
    }

    #[must_use]
    pub fn mapped_ptr(&self) -> Option<NonNull<u8>> {
        self.mapped_ptr
    }

    fn inner(&self) -> Result<&ProtocolObject<dyn MTLBufferTrait>, HalError> {
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

fn to_ns(value: u64) -> Result<usize, HalError> {
    usize::try_from(value).map_err(|_| buffer_error("value is too large"))
}

fn buffer_error(message: &'static str) -> HalError {
    HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    }
}

#[derive(Clone)]
pub struct MetalTexture {
    inner: Option<Retained<ProtocolObject<dyn MTLTextureTrait>>>,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    bytes_per_pixel: u32,
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
    fn inner(&self) -> Result<&ProtocolObject<dyn MTLTextureTrait>, HalError> {
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

#[derive(Clone)]
pub struct MetalSampler {
    _inner: Option<Retained<ProtocolObject<dyn MTLSamplerState>>>,
}

unsafe impl Send for MetalSampler {}
unsafe impl Sync for MetalSampler {}

impl std::fmt::Debug for MetalSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalSampler").finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct MetalComputePipeline {
    inner: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    workgroup_size: (u32, u32, u32),
}

unsafe impl Send for MetalComputePipeline {}
unsafe impl Sync for MetalComputePipeline {}

impl std::fmt::Debug for MetalComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalComputePipeline")
            .field("workgroup_size", &self.workgroup_size)
            .finish()
    }
}

#[derive(Clone)]
pub struct MetalRenderPipeline {
    inner: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    primitive_topology: HalPrimitiveTopology,
}

unsafe impl Send for MetalRenderPipeline {}
unsafe impl Sync for MetalRenderPipeline {}

impl std::fmt::Debug for MetalRenderPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalRenderPipeline")
            .field("primitive_topology", &self.primitive_topology)
            .finish()
    }
}

fn create_compute_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    msl_source: &str,
    entry_point: &str,
    workgroup_size: (u32, u32, u32),
) -> Result<MetalComputePipeline, HalError> {
    let source = NSString::from_str(msl_source);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|error| shader_error(error.localizedDescription().to_string()))?;
    let function = library
        .newFunctionWithName(&NSString::from_str(entry_point))
        .ok_or_else(|| shader_error(format!("compute function '{entry_point}' not found")))?;
    let inner = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| shader_error(error.localizedDescription().to_string()))?;
    Ok(MetalComputePipeline {
        inner,
        workgroup_size,
    })
}

fn create_render_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
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
    let source = NSString::from_str(msl_source);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|error| shader_error(error.localizedDescription().to_string()))?;
    let vertex_function = library
        .newFunctionWithName(&NSString::from_str(vertex_entry_point))
        .ok_or_else(|| shader_error(format!("vertex function '{vertex_entry_point}' not found")))?;
    let fragment_function = library
        .newFunctionWithName(&NSString::from_str(fragment_entry_point))
        .ok_or_else(|| {
            shader_error(format!(
                "fragment function '{fragment_entry_point}' not found"
            ))
        })?;
    let pipeline_descriptor = MTLRenderPipelineDescriptor::new();
    pipeline_descriptor.setVertexFunction(Some(&vertex_function));
    pipeline_descriptor.setFragmentFunction(Some(&fragment_function));
    let (pixel_format, _) = map_texture_format(color_format)?;
    let color_attachments = pipeline_descriptor.colorAttachments();
    let color = unsafe { color_attachments.objectAtIndexedSubscript(0) };
    color.setPixelFormat(pixel_format);
    let vertex_descriptor = MTLVertexDescriptor::new();
    for buffer in &descriptor.vertex_buffers {
        let metal_index = buffer
            .attributes
            .first()
            .map(|attribute| attribute.metal_buffer_index)
            .unwrap_or(0);
        let layouts = vertex_descriptor.layouts();
        let layout = unsafe { layouts.objectAtIndexedSubscript(to_ns(u64::from(metal_index))?) };
        unsafe {
            layout.setStride(to_ns(buffer.array_stride)?);
            layout.setStepRate(1);
        }
        layout.setStepFunction(match buffer.step_mode {
            HalVertexStepMode::Vertex => MTLVertexStepFunction::PerVertex,
            HalVertexStepMode::Instance => MTLVertexStepFunction::PerInstance,
        });
        for attribute in &buffer.attributes {
            let attributes = vertex_descriptor.attributes();
            let attr = unsafe {
                attributes.objectAtIndexedSubscript(to_ns(u64::from(attribute.shader_location))?)
            };
            attr.setFormat(map_vertex_format(attribute.format)?);
            unsafe {
                attr.setOffset(to_ns(attribute.offset)?);
                attr.setBufferIndex(to_ns(u64::from(attribute.metal_buffer_index))?);
            }
        }
    }
    pipeline_descriptor.setVertexDescriptor(Some(&vertex_descriptor));
    let inner = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_descriptor)
        .map_err(|error| shader_error(error.localizedDescription().to_string()))?;
    Ok(MetalRenderPipeline {
        inner,
        primitive_topology: descriptor.primitive_topology,
    })
}

fn create_texture(
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

fn create_sampler(
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

fn encode_buffer_copy(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
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
    unsafe {
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
            source.inner()?,
            to_ns(copy.source_offset)?,
            destination.inner()?,
            to_ns(copy.destination_offset)?,
            to_ns(copy.size)?,
        );
    }
    Ok(())
}

fn encode_buffer_to_texture(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
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
    unsafe {
        blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
            buffer.inner()?,
            to_ns(copy.buffer_layout.offset)?,
            to_ns(u64::from(copy.buffer_layout.bytes_per_row))?,
            buffer_texture_bytes_per_image(copy)?,
            to_mtl_size(copy.extent)?,
            texture.inner()?,
            to_ns(u64::from(copy.origin.z))?,
            to_ns(u64::from(copy.mip_level))?,
            to_mtl_origin(copy.origin.x, copy.origin.y, 0)?,
        );
    }
    Ok(())
}

fn encode_texture_to_buffer(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
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
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
            texture.inner()?,
            to_ns(u64::from(copy.origin.z))?,
            to_ns(u64::from(copy.mip_level))?,
            to_mtl_origin(copy.origin.x, copy.origin.y, 0)?,
            to_mtl_size(copy.extent)?,
            buffer.inner()?,
            to_ns(copy.buffer_layout.offset)?,
            to_ns(u64::from(copy.buffer_layout.bytes_per_row))?,
            buffer_texture_bytes_per_image(copy)?,
        );
    }
    Ok(())
}

fn encode_texture_to_texture(
    blit: &ProtocolObject<dyn MTLBlitCommandEncoder>,
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
    unsafe {
        blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
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
    }
    Ok(())
}

fn encode_compute_pass(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pass: &HalComputePass,
) -> Result<(), HalError> {
    let crate::HalComputePipeline::Metal(pipeline) = &pass.pipeline else {
        return Err(shader_error(
            "compute pipeline is not Metal-backed".to_owned(),
        ));
    };
    encoder.setComputePipelineState(&pipeline.inner);
    for binding in &pass.bind_buffers {
        encode_compute_buffer(encoder, binding)?;
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        to_mtl_dispatch_size(pass.workgroups)?,
        to_mtl_workgroup_size(pipeline.workgroup_size)?,
    );
    Ok(())
}

fn encode_compute_buffer(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    binding: &HalBoundBuffer,
) -> Result<(), HalError> {
    let HalBuffer::Metal(buffer) = &binding.buffer else {
        return Err(buffer_error("compute buffer is not Metal-backed"));
    };
    if binding.offset > buffer.size() {
        return Err(buffer_error("compute buffer offset exceeds buffer size"));
    }
    unsafe {
        encoder.setBuffer_offset_atIndex(
            Some(buffer.inner()?),
            to_ns(binding.offset)?,
            to_ns(u64::from(binding.metal_index))?,
        );
    }
    Ok(())
}

fn render_pass_descriptor(
    pass: &HalRenderPass,
) -> Result<Retained<MTLRenderPassDescriptor>, HalError> {
    let HalTexture::Metal(texture) = &pass.color_target.texture else {
        return Err(texture_error("render target is not Metal-backed"));
    };
    let descriptor = MTLRenderPassDescriptor::renderPassDescriptor();
    let color_attachments = descriptor.colorAttachments();
    let color = unsafe { color_attachments.objectAtIndexedSubscript(0) };
    color.setTexture(Some(texture.inner()?));
    color.setLoadAction(match pass.color_target.load_op {
        HalRenderLoadOp::Load => MTLLoadAction::Load,
        HalRenderLoadOp::Clear => MTLLoadAction::Clear,
    });
    color.setStoreAction(if pass.color_target.store {
        MTLStoreAction::Store
    } else {
        MTLStoreAction::DontCare
    });
    let [r, g, b, a] = pass.color_target.clear_color;
    color.setClearColor(MTLClearColor {
        red: r,
        green: g,
        blue: b,
        alpha: a,
    });
    Ok(descriptor)
}

fn encode_render_pass(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    pass: &HalRenderPass,
) -> Result<(), HalError> {
    let (Some(pipeline), Some(draw)) = (&pass.pipeline, pass.draw) else {
        return Ok(());
    };
    let crate::HalRenderPipeline::Metal(pipeline) = pipeline else {
        return Err(shader_error(
            "render pipeline is not Metal-backed".to_owned(),
        ));
    };
    encoder.setRenderPipelineState(&pipeline.inner);
    for binding in &pass.bind_buffers {
        encode_render_bind_buffer(encoder, binding)?;
    }
    for binding in &pass.vertex_buffers {
        encode_render_vertex_buffer(encoder, binding)?;
    }
    draw_primitives(encoder, pipeline.primitive_topology, draw)?;
    Ok(())
}

fn encode_render_bind_buffer(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
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
    unsafe {
        encoder.setVertexBuffer_offset_atIndex(Some(buffer.inner()?), offset, index);
        encoder.setFragmentBuffer_offset_atIndex(Some(buffer.inner()?), offset, index);
    }
    Ok(())
}

fn encode_render_vertex_buffer(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
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
    unsafe {
        encoder.setVertexBuffer_offset_atIndex(
            Some(buffer.inner()?),
            to_ns(binding.offset)?,
            to_ns(u64::from(binding.metal_index))?,
        );
    }
    Ok(())
}

fn draw_primitives(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    topology: HalPrimitiveTopology,
    draw: HalDraw,
) -> Result<(), HalError> {
    unsafe {
        encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
            map_primitive_topology(topology),
            to_ns(u64::from(draw.first_vertex))?,
            to_ns(u64::from(draw.vertex_count))?,
            to_ns(u64::from(draw.instance_count))?,
            to_ns(u64::from(draw.first_instance))?,
        );
    }
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

fn buffer_texture_bytes_per_image(copy: &HalBufferTextureCopy) -> Result<usize, HalError> {
    let bytes = u64::from(copy.buffer_layout.bytes_per_row)
        .checked_mul(u64::from(copy.buffer_layout.rows_per_image))
        .ok_or_else(|| buffer_error("buffer texture bytes per image overflows"))?;
    to_ns(bytes)
}

fn to_mtl_origin(x: u32, y: u32, z: u32) -> Result<MTLOrigin, HalError> {
    Ok(MTLOrigin {
        x: to_ns(u64::from(x))?,
        y: to_ns(u64::from(y))?,
        z: to_ns(u64::from(z))?,
    })
}

fn to_mtl_size(extent: HalExtent3d) -> Result<MTLSize, HalError> {
    Ok(MTLSize {
        width: to_ns(u64::from(extent.width))?,
        height: to_ns(u64::from(extent.height))?,
        depth: to_ns(u64::from(extent.depth_or_array_layers))?,
    })
}

fn to_mtl_dispatch_size(size: (u32, u32, u32)) -> Result<MTLSize, HalError> {
    Ok(MTLSize {
        width: to_ns(u64::from(size.0))?,
        height: to_ns(u64::from(size.1))?,
        depth: to_ns(u64::from(size.2))?,
    })
}

fn to_mtl_workgroup_size(size: (u32, u32, u32)) -> Result<MTLSize, HalError> {
    to_mtl_dispatch_size(size)
}

fn map_texture_format(format: HalTextureFormat) -> Result<(MTLPixelFormat, u32), HalError> {
    match format {
        HalTextureFormat::R8Unorm => Ok((MTLPixelFormat::R8Unorm, 1)),
        HalTextureFormat::Rgba8Unorm => Ok((MTLPixelFormat::RGBA8Unorm, 4)),
        HalTextureFormat::Bgra8Unorm => Ok((MTLPixelFormat::BGRA8Unorm, 4)),
        HalTextureFormat::Unsupported => Err(texture_error("unsupported texture format")),
    }
}

fn map_texture_usage(usage: HalTextureUsage) -> MTLTextureUsage {
    let mut metal_usage = MTLTextureUsage::Unknown;
    if usage.copy_src || usage.texture_binding {
        metal_usage |= MTLTextureUsage::ShaderRead;
    }
    if usage.copy_dst || usage.storage_binding {
        metal_usage |= MTLTextureUsage::ShaderWrite;
    }
    if usage.render_attachment {
        metal_usage |= MTLTextureUsage::RenderTarget;
    }
    metal_usage
}

fn map_address_mode(mode: HalAddressMode) -> MTLSamplerAddressMode {
    match mode {
        HalAddressMode::ClampToEdge => MTLSamplerAddressMode::ClampToEdge,
        HalAddressMode::Repeat => MTLSamplerAddressMode::Repeat,
        HalAddressMode::MirrorRepeat => MTLSamplerAddressMode::MirrorRepeat,
    }
}

fn map_filter_mode(mode: HalFilterMode) -> MTLSamplerMinMagFilter {
    match mode {
        HalFilterMode::Nearest => MTLSamplerMinMagFilter::Nearest,
        HalFilterMode::Linear => MTLSamplerMinMagFilter::Linear,
    }
}

fn map_mipmap_filter_mode(mode: HalMipmapFilterMode) -> MTLSamplerMipFilter {
    match mode {
        HalMipmapFilterMode::Nearest => MTLSamplerMipFilter::Nearest,
        HalMipmapFilterMode::Linear => MTLSamplerMipFilter::Linear,
    }
}

fn map_compare_function(compare: HalCompareFunction) -> MTLCompareFunction {
    match compare {
        HalCompareFunction::Never => MTLCompareFunction::Never,
        HalCompareFunction::Less => MTLCompareFunction::Less,
        HalCompareFunction::Equal => MTLCompareFunction::Equal,
        HalCompareFunction::LessEqual => MTLCompareFunction::LessEqual,
        HalCompareFunction::Greater => MTLCompareFunction::Greater,
        HalCompareFunction::NotEqual => MTLCompareFunction::NotEqual,
        HalCompareFunction::GreaterEqual => MTLCompareFunction::GreaterEqual,
        HalCompareFunction::Always => MTLCompareFunction::Always,
    }
}

fn map_vertex_format(format: HalVertexFormat) -> Result<MTLVertexFormat, HalError> {
    match format {
        HalVertexFormat::Float32 => Ok(MTLVertexFormat::Float),
        HalVertexFormat::Float32x2 => Ok(MTLVertexFormat::Float2),
        HalVertexFormat::Float32x3 => Ok(MTLVertexFormat::Float3),
        HalVertexFormat::Float32x4 => Ok(MTLVertexFormat::Float4),
        HalVertexFormat::Unsupported => Err(shader_error(
            "unsupported vertex format for Metal".to_owned(),
        )),
    }
}

fn map_primitive_topology(topology: HalPrimitiveTopology) -> MTLPrimitiveType {
    match topology {
        HalPrimitiveTopology::PointList => MTLPrimitiveType::Point,
        HalPrimitiveTopology::LineList => MTLPrimitiveType::Line,
        HalPrimitiveTopology::LineStrip => MTLPrimitiveType::LineStrip,
        HalPrimitiveTopology::TriangleList => MTLPrimitiveType::Triangle,
        HalPrimitiveTopology::TriangleStrip => MTLPrimitiveType::TriangleStrip,
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
