#![warn(missing_docs)]
//! Backend abstraction layer for yawgpu GPU implementations.

use std::ffi::c_void;
use std::ptr::NonNull;

mod command;
mod descriptors;
mod error;
mod format;
mod present;
mod shader;

pub use command::{
    HalBoundBuffer, HalBufferBindingKind, HalBufferCopy, HalBufferTextureCopy,
    HalBufferTextureLayout, HalComputePass, HalCopy, HalDescriptorBinding, HalDraw,
    HalRenderColorTarget, HalRenderLoadOp, HalRenderPass, HalTextureCopy,
};
pub use descriptors::{
    HalExtent3d, HalOrigin3d, HalRenderPipelineDescriptor, HalSamplerDescriptor,
    HalTextureDescriptor, HalVertexAttribute, HalVertexBufferLayout,
};
pub use error::HalError;
pub use format::{
    HalAddressMode, HalCompareFunction, HalFilterMode, HalMipmapFilterMode, HalPrimitiveTopology,
    HalTextureFormat, HalTextureUsage, HalVertexFormat, HalVertexStepMode,
};
pub use present::{HalPresentMode, HalSurfaceConfiguration};
pub use shader::HalShaderSource;

/// Noop module.
#[cfg(feature = "noop")]
pub mod noop;

/// Metal module.
#[cfg(feature = "metal")]
pub mod metal;

/// Vulkan module.
#[cfg(feature = "vulkan")]
pub mod vulkan;

/// Enumerates HAL instance values.
#[derive(Debug)]
#[non_exhaustive]
pub enum HalInstance {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop(noop::NoopInstance),
    #[cfg(feature = "vulkan")]
    /// Vulkan variant.
    Vulkan(vulkan::VulkanInstance),
    #[cfg(feature = "metal")]
    /// Metal variant.
    Metal(metal::MetalInstance),
}

impl HalInstance {
    /// Returns new noop.
    #[cfg(feature = "noop")]
    #[must_use]
    pub fn new_noop() -> Self {
        Self::Noop(noop::NoopInstance::new())
    }

    /// Returns adapters exposed by this instance.
    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<HalAdapter> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(instance) => instance
                .enumerate_adapters()
                .into_iter()
                .map(HalAdapter::Noop)
                .collect(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(instance) => instance
                .enumerate_adapters()
                .into_iter()
                .map(HalAdapter::Vulkan)
                .collect(),
            #[cfg(feature = "metal")]
            Self::Metal(instance) => instance
                .enumerate_adapters()
                .into_iter()
                .map(HalAdapter::Metal)
                .collect(),
        }
    }

    /// # Safety
    ///
    /// `layer` must be a valid, non-dangling `CAMetalLayer` instance pointer.
    pub unsafe fn create_surface_from_metal_layer(
        &self,
        layer: *mut c_void,
    ) -> Result<HalSurface, HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = layer;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(HalSurface::Noop),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(instance) => unsafe {
                instance
                    .create_surface_from_metal_layer(layer)
                    .map(HalSurface::Vulkan)
            },
            #[cfg(feature = "metal")]
            Self::Metal(_) => unsafe {
                metal::MetalSurface::from_layer(layer).map(HalSurface::Metal)
            },
        }
    }

    /// # Safety
    ///
    /// `hwnd` must be a valid Win32 window handle and `hinstance` its module
    /// instance; both must outlive the surface. Ignored by the Noop backend.
    pub unsafe fn create_surface_from_windows_hwnd(
        &self,
        hinstance: *mut c_void,
        hwnd: *mut c_void,
    ) -> Result<HalSurface, HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = (hinstance, hwnd);
        #[cfg(all(feature = "metal", not(feature = "vulkan")))]
        let _ = (hinstance, hwnd);
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(HalSurface::Noop),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(instance) => unsafe {
                instance
                    .create_surface_from_windows_hwnd(hinstance, hwnd)
                    .map(HalSurface::Vulkan)
            },
            #[cfg(feature = "metal")]
            Self::Metal(_) => Err(HalError::SwapchainCreationFailed {
                backend: "metal",
                message: "HWND surface is not supported on Metal",
            }),
        }
    }
}

/// Enumerates HAL adapter values.
#[derive(Debug)]
#[non_exhaustive]
pub enum HalAdapter {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop(noop::NoopAdapter),
    #[cfg(feature = "vulkan")]
    /// Vulkan variant.
    Vulkan(vulkan::VulkanAdapter),
    #[cfg(feature = "metal")]
    /// Metal variant.
    Metal(metal::MetalAdapter),
}

impl HalAdapter {
    /// Returns the name.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.name().to_owned(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.name().to_owned(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.name().to_owned(),
        }
    }

    /// Returns the backend.
    #[must_use]
    pub fn backend(&self) -> HalBackend {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => HalBackend::Noop,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => HalBackend::Vulkan,
            #[cfg(feature = "metal")]
            Self::Metal(_) => HalBackend::Metal,
        }
    }

    /// Creates a device (and its default queue) on this adapter.
    pub fn create_device(&self) -> Result<HalDevice, HalError> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.create_device().map(HalDevice::Noop),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.create_device().map(HalDevice::Vulkan),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.create_device().map(HalDevice::Metal),
        }
    }
}

/// Enumerates HAL backend values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalBackend {
    /// Noop variant.
    Noop,
    /// Vulkan variant.
    Vulkan,
    /// Metal variant.
    Metal,
}

/// Enumerates HAL device values.
#[derive(Debug)]
#[non_exhaustive]
pub enum HalDevice {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop(noop::NoopDevice),
    #[cfg(feature = "vulkan")]
    /// Vulkan variant.
    Vulkan(vulkan::VulkanDevice),
    #[cfg(feature = "metal")]
    /// Metal variant.
    Metal(metal::MetalDevice),
}

impl HalDevice {
    /// Returns the backend.
    #[must_use]
    pub fn backend(&self) -> HalBackend {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => HalBackend::Noop,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => HalBackend::Vulkan,
            #[cfg(feature = "metal")]
            Self::Metal(_) => HalBackend::Metal,
        }
    }

    /// Returns the allocation count.
    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => device.allocation_count(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.allocation_count(),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.allocation_count(),
        }
    }

    /// Returns the queue.
    #[must_use]
    pub fn queue(&self) -> HalQueue {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalQueue::Noop(device.queue().clone()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => HalQueue::Vulkan(device.queue().clone()),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalQueue::Metal(device.queue().clone()),
        }
    }

    /// Allocates a buffer of the given size on this device.
    #[must_use]
    pub fn create_buffer(&self, size: u64) -> HalBuffer {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalBuffer::Noop(device.create_buffer(size)),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => HalBuffer::Vulkan(device.create_buffer(size)),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalBuffer::Metal(device.create_buffer(size)),
        }
    }

    /// Creates a texture matching the given descriptor.
    #[must_use]
    pub fn create_texture(&self, descriptor: &HalTextureDescriptor) -> HalTexture {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = descriptor;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalTexture::Noop(device.create_texture()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => HalTexture::Vulkan(device.create_texture(descriptor)),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalTexture::Metal(device.create_texture(descriptor)),
        }
    }

    /// Creates a sampler matching the given descriptor.
    #[must_use]
    pub fn create_sampler(&self, descriptor: &HalSamplerDescriptor) -> HalSampler {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = descriptor;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalSampler::Noop(device.create_sampler()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => HalSampler::Vulkan(device.create_sampler(descriptor)),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalSampler::Metal(device.create_sampler(descriptor)),
        }
    }

    /// Creates a compute pipeline from the given shader, entry point, and bindings.
    pub fn create_compute_pipeline(
        &self,
        shader: HalShaderSource,
        entry_point: &str,
        workgroup_size: (u32, u32, u32),
        bindings: &[HalDescriptorBinding],
    ) -> Result<HalComputePipeline, HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = (shader, entry_point, workgroup_size, bindings);
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(HalComputePipeline::Noop),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device
                .create_compute_pipeline(shader, entry_point, workgroup_size, bindings)
                .map(HalComputePipeline::Vulkan),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device
                .create_compute_pipeline(shader, entry_point, workgroup_size, bindings)
                .map(HalComputePipeline::Metal),
        }
    }

    /// Creates a render pipeline from the given shaders, vertex layout, and color targets.
    pub fn create_render_pipeline(
        &self,
        shader: HalShaderSource,
        vertex_entry_point: &str,
        fragment_entry_point: &str,
        descriptor: &HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
    ) -> Result<HalRenderPipeline, HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = (
            shader,
            vertex_entry_point,
            fragment_entry_point,
            descriptor,
            bindings,
        );
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(HalRenderPipeline::Noop),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device
                .create_render_pipeline(
                    shader,
                    vertex_entry_point,
                    fragment_entry_point,
                    descriptor,
                    bindings,
                )
                .map(HalRenderPipeline::Vulkan),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device
                .create_render_pipeline(
                    shader,
                    vertex_entry_point,
                    fragment_entry_point,
                    descriptor,
                    bindings,
                )
                .map(HalRenderPipeline::Metal),
        }
    }
}

/// Enumerates HAL surface values.
#[derive(Debug)]
#[non_exhaustive]
pub enum HalSurface {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop,
    #[cfg(feature = "vulkan")]
    /// Vulkan variant.
    Vulkan(vulkan::VulkanSurface),
    #[cfg(feature = "metal")]
    /// Metal variant.
    Metal(metal::MetalSurface),
}

impl HalSurface {
    /// Configures the surface's swapchain for the given format, size, and present mode.
    pub fn configure(
        &mut self,
        device: &HalDevice,
        config: HalSurfaceConfiguration,
    ) -> Result<(), HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = config;
        #[allow(unreachable_patterns)]
        match (self, device) {
            #[cfg(feature = "noop")]
            (Self::Noop, _) => Ok(()),
            #[cfg(feature = "vulkan")]
            (Self::Vulkan(surface), HalDevice::Vulkan(device)) => surface.configure(device, config),
            #[cfg(feature = "metal")]
            (Self::Metal(surface), HalDevice::Metal(device)) => surface.configure(device, config),
            _ => Err(HalError::SwapchainCreationFailed {
                backend: "surface",
                message: "surface and device backends do not match",
            }),
        }
    }

    /// Tears down the surface's swapchain.
    pub fn unconfigure(&mut self) {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop => {}
            #[cfg(feature = "vulkan")]
            Self::Vulkan(surface) => surface.unconfigure(),
            #[cfg(feature = "metal")]
            Self::Metal(surface) => surface.unconfigure(),
        }
    }

    /// Returns acquire next texture.
    pub fn acquire_next_texture(&mut self) -> Result<HalTexture, HalError> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop => Err(HalError::AcquireFailed {
                backend: "noop",
                message: "Noop surfaces do not provide swapchain textures",
            }),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(surface) => surface.acquire_next_texture().map(HalTexture::Vulkan),
            #[cfg(feature = "metal")]
            Self::Metal(surface) => surface.acquire_next_texture().map(HalTexture::Metal),
        }
    }

    /// Presents the most recently acquired surface texture.
    pub fn present(&mut self, queue: &HalQueue) -> Result<(), HalError> {
        #[allow(unreachable_patterns)]
        match (self, queue) {
            #[cfg(feature = "noop")]
            (Self::Noop, _) => Ok(()),
            #[cfg(feature = "vulkan")]
            (Self::Vulkan(surface), HalQueue::Vulkan(queue)) => surface.present(queue),
            #[cfg(feature = "metal")]
            (Self::Metal(surface), HalQueue::Metal(queue)) => surface.present(queue),
            _ => Err(HalError::PresentFailed {
                backend: "surface",
                message: "surface and queue backends do not match",
            }),
        }
    }
}

/// Enumerates HAL queue values.
#[derive(Debug)]
#[non_exhaustive]
pub enum HalQueue {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop(noop::NoopQueue),
    #[cfg(feature = "vulkan")]
    /// Vulkan variant.
    Vulkan(vulkan::VulkanQueue),
    #[cfg(feature = "metal")]
    /// Metal variant.
    Metal(metal::MetalQueue),
}

impl HalQueue {
    /// Submits an empty command buffer to flush the queue.
    pub fn submit_empty(&self) -> Result<(), HalError> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(queue) => queue.submit_empty(),
            #[cfg(feature = "metal")]
            Self::Metal(queue) => queue.submit_empty(),
        }
    }

    /// Records and submits the given buffer/texture copy operations.
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = copies;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(queue) => queue.submit_copies(copies),
            #[cfg(feature = "metal")]
            Self::Metal(queue) => queue.submit_copies(copies),
        }
    }
}

/// Enumerates HAL buffer values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalBuffer {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop(noop::NoopBuffer),
    #[cfg(feature = "vulkan")]
    /// Vulkan variant.
    Vulkan(vulkan::VulkanBuffer),
    #[cfg(feature = "metal")]
    /// Metal variant.
    Metal(metal::MetalBuffer),
}

impl HalBuffer {
    /// Returns the size.
    #[must_use]
    pub fn size(&self) -> u64 {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(buffer) => buffer.size(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(buffer) => buffer.size(),
            #[cfg(feature = "metal")]
            Self::Metal(buffer) => buffer.size(),
        }
    }

    /// Records a write command.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = (offset, data);
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(()),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(buffer) => buffer.write(offset, data),
            #[cfg(feature = "metal")]
            Self::Metal(buffer) => buffer.write(offset, data),
        }
    }

    /// Reads `len` bytes at `offset` back from the buffer into host memory.
    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = offset;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => usize::try_from(len).map_or_else(
                |_| {
                    Err(HalError::BufferOperationFailed {
                        backend: "noop",
                        message: "read length is too large",
                    })
                },
                |len| Ok(vec![0; len]),
            ),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(buffer) => buffer.read(offset, len),
            #[cfg(feature = "metal")]
            Self::Metal(buffer) => buffer.read(offset, len),
        }
    }

    /// Returns mapped ptr.
    #[must_use]
    pub fn mapped_ptr(&self) -> Option<NonNull<u8>> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(buffer) => buffer.mapped_ptr(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(buffer) => buffer.mapped_ptr(),
            #[cfg(feature = "metal")]
            Self::Metal(buffer) => buffer.mapped_ptr(),
        }
    }
}

/// Enumerates HAL texture values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalTexture {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop(noop::NoopTexture),
    #[cfg(feature = "vulkan")]
    /// Vulkan variant.
    Vulkan(vulkan::VulkanTexture),
    #[cfg(feature = "metal")]
    /// Metal variant.
    Metal(metal::MetalTexture),
}

/// Enumerates HAL sampler values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalSampler {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop(noop::NoopSampler),
    #[cfg(feature = "vulkan")]
    /// Vulkan variant.
    Vulkan(vulkan::VulkanSampler),
    #[cfg(feature = "metal")]
    /// Metal variant.
    Metal(metal::MetalSampler),
}

/// Enumerates HAL compute pipeline values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalComputePipeline {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop,
    #[cfg(feature = "vulkan")]
    /// Vulkan variant.
    Vulkan(vulkan::VulkanComputePipeline),
    #[cfg(feature = "metal")]
    /// Metal variant.
    Metal(metal::MetalComputePipeline),
}

/// Enumerates HAL render pipeline values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalRenderPipeline {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop,
    #[cfg(feature = "vulkan")]
    /// Vulkan variant.
    Vulkan(vulkan::VulkanRenderPipeline),
    #[cfg(feature = "metal")]
    /// Metal variant.
    Metal(metal::MetalRenderPipeline),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop_device() -> Result<HalDevice, HalError> {
        let instance = HalInstance::new_noop();
        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop instance yields one adapter");
        adapter.create_device()
    }

    fn texture_descriptor() -> HalTextureDescriptor {
        HalTextureDescriptor {
            format: HalTextureFormat::Rgba8Unorm,
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
            usage: HalTextureUsage {
                copy_src: true,
                copy_dst: true,
                texture_binding: false,
                storage_binding: false,
                render_attachment: true,
            },
        }
    }

    fn sampler_descriptor() -> HalSamplerDescriptor {
        HalSamplerDescriptor {
            address_mode_u: HalAddressMode::ClampToEdge,
            address_mode_v: HalAddressMode::ClampToEdge,
            address_mode_w: HalAddressMode::ClampToEdge,
            mag_filter: HalFilterMode::Nearest,
            min_filter: HalFilterMode::Nearest,
            mipmap_filter: HalMipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            max_anisotropy: 1,
        }
    }

    fn surface_configuration() -> HalSurfaceConfiguration {
        HalSurfaceConfiguration::new(
            HalTextureFormat::Bgra8Unorm,
            HalTextureUsage {
                copy_src: false,
                copy_dst: false,
                texture_binding: false,
                storage_binding: false,
                render_attachment: true,
            },
            640,
            480,
            HalPresentMode::Fifo,
        )
    }

    fn render_pipeline_descriptor() -> HalRenderPipelineDescriptor {
        HalRenderPipelineDescriptor {
            color_formats: vec![HalTextureFormat::Rgba8Unorm],
            vertex_buffers: Vec::new(),
            primitive_topology: HalPrimitiveTopology::TriangleList,
        }
    }

    #[test]
    fn noop_creates_device_with_zero_allocations() -> Result<(), HalError> {
        let instance = HalInstance::new_noop();
        let adapters = instance.enumerate_adapters();
        assert_eq!(adapters.len(), 1);

        let device = adapters[0].create_device()?;
        assert_eq!(device.allocation_count(), 0);

        Ok(())
    }

    #[test]
    fn create_surface_from_metal_layer_noop_ignores_layer_pointer() -> Result<(), HalError> {
        let instance = HalInstance::new_noop();
        let dangling = 0xdead_beefusize as *mut c_void;

        // SAFETY: Noop arm does not dereference the layer pointer.
        let surface = unsafe { instance.create_surface_from_metal_layer(dangling)? };

        assert!(matches!(surface, HalSurface::Noop));
        Ok(())
    }

    #[test]
    fn create_surface_from_windows_hwnd_noop_ignores_pointers() -> Result<(), HalError> {
        let instance = HalInstance::new_noop();
        let hwnd = 0xdead_beefusize as *mut c_void;

        // SAFETY: Noop arm does not dereference the pointers.
        let surface =
            unsafe { instance.create_surface_from_windows_hwnd(std::ptr::null_mut(), hwnd)? };

        assert!(matches!(surface, HalSurface::Noop));
        Ok(())
    }

    #[test]
    fn hal_adapter_name_noop_returns_fixed_string() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter exists");

        assert_eq!(adapter.name(), "yawgpu Noop Adapter");
    }

    #[test]
    fn hal_adapter_backend_noop_returns_noop() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter exists");

        assert_eq!(adapter.backend(), HalBackend::Noop);
    }

    #[test]
    fn hal_device_backend_noop_returns_noop() -> Result<(), HalError> {
        let device = noop_device()?;

        assert_eq!(device.backend(), HalBackend::Noop);
        Ok(())
    }

    #[test]
    fn hal_device_queue_noop_returns_queue_that_submits_empty() -> Result<(), HalError> {
        let device = noop_device()?;
        let queue = device.queue();

        assert!(matches!(queue, HalQueue::Noop(_)));
        queue.submit_empty()
    }

    #[test]
    fn hal_device_create_buffer_noop_records_requested_size() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(256);

        assert!(matches!(buffer, HalBuffer::Noop(_)));
        assert_eq!(buffer.size(), 256);
        Ok(())
    }

    #[test]
    fn hal_device_create_texture_noop_returns_texture_and_increments_allocations(
    ) -> Result<(), HalError> {
        let device = noop_device()?;
        let texture = device.create_texture(&texture_descriptor());

        assert!(matches!(texture, HalTexture::Noop(_)));
        assert_eq!(device.allocation_count(), 1);
        Ok(())
    }

    #[test]
    fn hal_device_create_sampler_noop_returns_sampler_and_increments_allocations(
    ) -> Result<(), HalError> {
        let device = noop_device()?;
        let sampler = device.create_sampler(&sampler_descriptor());

        assert!(matches!(sampler, HalSampler::Noop(_)));
        assert_eq!(device.allocation_count(), 1);
        Ok(())
    }

    #[test]
    fn hal_device_create_compute_pipeline_noop_accepts_empty_shader() -> Result<(), HalError> {
        let device = noop_device()?;
        let pipeline = device.create_compute_pipeline(
            HalShaderSource::Msl(String::new()),
            "main",
            (1, 1, 1),
            &[],
        )?;

        assert!(matches!(pipeline, HalComputePipeline::Noop));
        Ok(())
    }

    #[test]
    fn hal_device_create_render_pipeline_noop_accepts_empty_shader() -> Result<(), HalError> {
        let device = noop_device()?;
        let pipeline = device.create_render_pipeline(
            HalShaderSource::Msl(String::new()),
            "vs_main",
            "fs_main",
            &render_pipeline_descriptor(),
            &[],
        )?;

        assert!(matches!(pipeline, HalRenderPipeline::Noop));
        Ok(())
    }

    #[test]
    fn hal_surface_configure_noop_returns_ok() -> Result<(), HalError> {
        let device = noop_device()?;
        let mut surface = HalSurface::Noop;

        surface.configure(&device, surface_configuration())
    }

    #[test]
    fn hal_surface_unconfigure_noop_is_idempotent() {
        let mut surface = HalSurface::Noop;

        surface.unconfigure();
        surface.unconfigure();
    }

    #[test]
    fn hal_surface_acquire_next_texture_noop_returns_acquire_failed() {
        let mut surface = HalSurface::Noop;

        let error = surface
            .acquire_next_texture()
            .expect_err("Noop surface has no swapchain texture");
        match error {
            HalError::AcquireFailed { backend, .. } => assert_eq!(backend, "noop"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn hal_surface_present_noop_returns_ok_without_acquire() -> Result<(), HalError> {
        let device = noop_device()?;
        let queue = device.queue();
        let mut surface = HalSurface::Noop;

        surface.present(&queue)
    }

    #[test]
    fn hal_queue_submit_empty_noop_returns_ok() -> Result<(), HalError> {
        let device = noop_device()?;
        let queue = device.queue();

        queue.submit_empty()
    }

    #[test]
    fn hal_queue_submit_copies_noop_accepts_empty_and_buffer_copy() -> Result<(), HalError> {
        let device = noop_device()?;
        let queue = device.queue();
        let source = device.create_buffer(8);
        let destination = device.create_buffer(8);
        let copy = HalCopy::Buffer(HalBufferCopy {
            source,
            source_offset: 0,
            destination,
            destination_offset: 0,
            size: 8,
        });

        queue.submit_copies(&[])?;
        queue.submit_copies(&[copy])
    }

    #[test]
    fn hal_buffer_size_noop_matches_creation_size() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(4096);

        assert_eq!(buffer.size(), 4096);
        Ok(())
    }

    #[test]
    fn hal_buffer_write_noop_accepts_empty_and_non_empty_data() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(16);

        buffer.write(0, &[])?;
        buffer.write(4, &[1, 2, 3, 4])
    }

    #[test]
    fn hal_buffer_read_noop_returns_zeroed_vector() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(16);

        assert_eq!(buffer.read(0, 0)?, Vec::<u8>::new());
        assert_eq!(buffer.read(4, 4)?, vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn hal_buffer_mapped_ptr_noop_returns_none() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(16);

        assert!(buffer.mapped_ptr().is_none());
        Ok(())
    }
}
