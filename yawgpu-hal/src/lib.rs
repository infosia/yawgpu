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
    HalBoundBuffer, HalBoundIndexBuffer, HalBoundIndirectBuffer, HalBoundSampler, HalBoundTexture,
    HalBufferBindingKind, HalBufferClear, HalBufferCopy, HalBufferTextureCopy,
    HalBufferTextureLayout, HalComputePass, HalCopy, HalDescriptorBinding,
    HalDescriptorBindingKind, HalDraw, HalIndexFormat, HalRenderColorTarget,
    HalRenderDepthStencilAttachment, HalRenderLoadOp, HalRenderPass, HalTextureAspect,
    HalTextureCopy, HalTextureViewDimension,
};
#[cfg(feature = "tiled")]
pub use command::{
    HalSubpassAttachmentLayout, HalSubpassAttachmentResource, HalSubpassColorAttachment,
    HalSubpassDependency, HalSubpassDependencyType, HalSubpassDepthStencilAttachment,
    HalSubpassDraw, HalSubpassInputAttachment, HalSubpassLayout, HalSubpassPassLayout,
    HalSubpassRenderPass, HalSubpassRenderPassCommand,
};
#[cfg(feature = "tiled")]
pub use descriptors::HalTransientAttachmentDescriptor;
pub use descriptors::{
    HalBlendComponent, HalBlendFactor, HalBlendOperation, HalBlendState, HalColorTargetState,
    HalDepthStencilState, HalExtent3d, HalOrigin3d, HalRenderPipelineDescriptor,
    HalSamplerDescriptor, HalStencilFaceState, HalTextureDescriptor, HalTextureDimension,
    HalVertexAttribute, HalVertexBufferLayout,
};
pub use error::HalError;
pub use format::{
    HalAddressMode, HalBufferUsage, HalCompareFunction, HalFilterMode, HalMipmapFilterMode,
    HalPrimitiveTopology, HalStencilOperation, HalTextureFormat, HalTextureUsage, HalVertexFormat,
    HalVertexStepMode,
};
pub use present::{HalPresentMode, HalSurfaceConfiguration};
pub use shader::{HalShaderSource, HalShaderStage};

/// Enumerates backend paths for shader framebuffer fetch.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FramebufferFetchPath {
    /// Shader framebuffer fetch is unavailable.
    Disabled,
    /// Vulkan `VK_EXT_shader_tile_image` path.
    TileImage,
    /// Vulkan raster-order attachment access path.
    RasterOrderAttachmentAccess,
}

/// Noop module.
#[cfg(feature = "noop")]
pub mod noop;

/// Metal module.
#[cfg(feature = "metal")]
pub mod metal;

/// GLES module.
#[cfg(feature = "gles")]
pub mod gles;

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
    #[cfg(feature = "gles")]
    /// GLES variant.
    Gles(gles::GlesInstance),
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
            #[cfg(feature = "gles")]
            Self::Gles(instance) => instance
                .enumerate_adapters()
                .into_iter()
                .map(HalAdapter::Gles)
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
            #[cfg(feature = "gles")]
            Self::Gles(_) => Err(HalError::SwapchainCreationFailed {
                backend: "gles",
                message: "Metal layer surface is not supported on GLES",
            }),
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
        #[cfg(not(feature = "vulkan"))]
        let _ = hinstance;
        #[cfg(not(any(feature = "gles", feature = "vulkan")))]
        let _ = hwnd;
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
            #[cfg(feature = "gles")]
            Self::Gles(instance) => unsafe {
                instance
                    .create_surface_from_windows_hwnd(hwnd)
                    .map(HalSurface::Gles)
            },
        }
    }

    /// # Safety
    ///
    /// `window` must be a valid `ANativeWindow*` from the Android NDK and
    /// must outlive the resulting surface. Ignored by the Noop backend.
    pub unsafe fn create_surface_from_android_native_window(
        &self,
        window: *mut c_void,
    ) -> Result<HalSurface, HalError> {
        #[cfg(not(feature = "gles"))]
        let _ = window;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(HalSurface::Noop),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => Err(HalError::SwapchainCreationFailed {
                backend: "vulkan",
                message: "Android native window surface not implemented",
            }),
            #[cfg(feature = "metal")]
            Self::Metal(_) => Err(HalError::SwapchainCreationFailed {
                backend: "metal",
                message: "Android native window surface is not supported on Metal",
            }),
            #[cfg(feature = "gles")]
            Self::Gles(instance) => unsafe {
                instance
                    .create_surface_from_android_native_window(window)
                    .map(HalSurface::Gles)
            },
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
    #[cfg(feature = "gles")]
    /// GLES variant.
    Gles(gles::GlesAdapter),
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
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.name().to_owned(),
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
            #[cfg(feature = "gles")]
            Self::Gles(_) => HalBackend::Gles,
        }
    }

    /// Returns the detected shader framebuffer-fetch path.
    #[cfg(feature = "tiled")]
    #[must_use]
    pub fn framebuffer_fetch_path(&self) -> FramebufferFetchPath {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => FramebufferFetchPath::Disabled,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.framebuffer_fetch_path(),
            #[cfg(feature = "metal")]
            Self::Metal(_) => FramebufferFetchPath::TileImage,
            #[cfg(feature = "gles")]
            Self::Gles(_) => FramebufferFetchPath::Disabled,
        }
    }

    /// Returns true when shader framebuffer fetch is supported.
    #[cfg(feature = "tiled")]
    #[must_use]
    pub fn supports_shader_framebuffer_fetch(&self) -> bool {
        !matches!(
            self.framebuffer_fetch_path(),
            FramebufferFetchPath::Disabled
        )
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
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.create_device().map(HalDevice::Gles),
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
    /// GLES variant.
    Gles,
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
    #[cfg(feature = "gles")]
    /// GLES variant.
    Gles(gles::GlesDevice),
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
            #[cfg(feature = "gles")]
            Self::Gles(_) => HalBackend::Gles,
        }
    }

    /// Returns the detected shader framebuffer-fetch path.
    #[cfg(feature = "tiled")]
    #[must_use]
    pub fn framebuffer_fetch_path(&self) -> FramebufferFetchPath {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => FramebufferFetchPath::Disabled,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.framebuffer_fetch_path(),
            #[cfg(feature = "metal")]
            Self::Metal(_) => FramebufferFetchPath::TileImage,
            #[cfg(feature = "gles")]
            Self::Gles(_) => FramebufferFetchPath::Disabled,
        }
    }

    /// Returns true when shader framebuffer fetch is supported.
    #[cfg(feature = "tiled")]
    #[must_use]
    pub fn supports_shader_framebuffer_fetch(&self) -> bool {
        !matches!(
            self.framebuffer_fetch_path(),
            FramebufferFetchPath::Disabled
        )
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
            #[cfg(feature = "gles")]
            Self::Gles(device) => device.allocation_count(),
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
            #[cfg(feature = "gles")]
            Self::Gles(device) => HalQueue::Gles(device.queue().clone()),
        }
    }

    /// Allocates a buffer of the given size on this device.
    #[must_use]
    pub fn create_buffer(&self, size: u64, usage: HalBufferUsage) -> HalBuffer {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalBuffer::Noop(device.create_buffer(size, usage)),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => HalBuffer::Vulkan(device.create_buffer(size, usage)),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalBuffer::Metal(device.create_buffer(size, usage)),
            #[cfg(feature = "gles")]
            Self::Gles(device) => HalBuffer::Gles(device.create_buffer(size, usage)),
        }
    }

    /// Creates a texture matching the given descriptor.
    #[must_use]
    pub fn create_texture(&self, descriptor: &HalTextureDescriptor) -> HalTexture {
        #[cfg(not(any(feature = "metal", feature = "vulkan", feature = "gles")))]
        let _ = descriptor;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalTexture::Noop(device.create_texture(descriptor)),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => HalTexture::Vulkan(device.create_texture(descriptor)),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalTexture::Metal(device.create_texture(descriptor)),
            #[cfg(feature = "gles")]
            Self::Gles(device) => HalTexture::Gles(device.create_texture(descriptor)),
        }
    }

    /// Creates a transient attachment matching the given concrete descriptor.
    #[cfg(feature = "tiled")]
    pub fn create_transient_attachment(
        &self,
        descriptor: &HalTransientAttachmentDescriptor,
    ) -> Result<HalTransientAttachment, HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = descriptor;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => Ok(HalTransientAttachment::Noop(
                device.create_transient_attachment(),
            )),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device
                .create_transient_attachment(descriptor)
                .map(HalTransientAttachment::Vulkan),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device
                .create_transient_attachment(descriptor)
                .map(HalTransientAttachment::Metal),
            #[cfg(feature = "gles")]
            Self::Gles(_) => Err(HalError::BackendUnavailable { backend: "gles" }),
        }
    }

    /// Begins a subpass render pass.
    #[cfg(feature = "tiled")]
    pub fn begin_subpass_render_pass(&self) -> Result<HalSubpassRenderPass, HalError> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(HalSubpassRenderPass::Noop(
                command::HalNoopSubpassRenderPass::new(),
            )),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => Ok(HalSubpassRenderPass::Vulkan),
            #[cfg(feature = "metal")]
            Self::Metal(_) => Ok(HalSubpassRenderPass::Metal),
            #[cfg(feature = "gles")]
            Self::Gles(_) => Err(HalError::BackendUnavailable { backend: "gles" }),
        }
    }

    /// Advances a subpass render pass.
    #[cfg(feature = "tiled")]
    pub fn next_subpass_render_pass(
        &self,
        pass: &mut HalSubpassRenderPass,
    ) -> Result<(), HalError> {
        #[allow(unreachable_patterns)]
        match (self, pass) {
            #[cfg(feature = "noop")]
            (Self::Noop(_), HalSubpassRenderPass::Noop(pass)) => {
                pass.next_subpass();
                Ok(())
            }
            #[cfg(feature = "vulkan")]
            (Self::Vulkan(_), HalSubpassRenderPass::Vulkan) => Ok(()),
            #[cfg(feature = "metal")]
            (Self::Metal(_), HalSubpassRenderPass::Metal) => Ok(()),
            _ => Err(HalError::BufferOperationFailed {
                backend: "subpass",
                message: "subpass pass backend does not match device",
            }),
        }
    }

    /// Ends a subpass render pass.
    #[cfg(feature = "tiled")]
    pub fn end_subpass_render_pass(&self, pass: HalSubpassRenderPass) -> Result<(), HalError> {
        #[allow(unreachable_patterns)]
        match (self, pass) {
            #[cfg(feature = "noop")]
            (Self::Noop(_), HalSubpassRenderPass::Noop(_)) => Ok(()),
            #[cfg(feature = "vulkan")]
            (Self::Vulkan(_), HalSubpassRenderPass::Vulkan) => Ok(()),
            #[cfg(feature = "metal")]
            (Self::Metal(_), HalSubpassRenderPass::Metal) => Ok(()),
            _ => Err(HalError::BufferOperationFailed {
                backend: "subpass",
                message: "subpass pass backend does not match device",
            }),
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
            #[cfg(feature = "gles")]
            Self::Gles(device) => HalSampler::Gles(device.create_sampler(descriptor)),
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
        #[cfg(not(any(feature = "gles", feature = "metal", feature = "vulkan")))]
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
            #[cfg(feature = "gles")]
            Self::Gles(device) => device
                .create_compute_pipeline(shader, entry_point, workgroup_size, bindings)
                .map(HalComputePipeline::Gles),
        }
    }

    /// Creates a render pipeline from the given shaders, vertex layout, and color targets.
    pub fn create_render_pipeline(
        &self,
        shader: HalShaderSource,
        vertex_entry_point: &str,
        fragment_entry_point: Option<&str>,
        descriptor: &HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
    ) -> Result<HalRenderPipeline, HalError> {
        #[cfg(not(any(feature = "gles", feature = "metal", feature = "vulkan")))]
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
            #[cfg(feature = "gles")]
            Self::Gles(device) => device
                .create_render_pipeline(
                    shader,
                    vertex_entry_point,
                    fragment_entry_point,
                    descriptor,
                    bindings,
                )
                .map(HalRenderPipeline::Gles),
        }
    }
    /// Creates a subpass-compatible render pipeline from the given shaders and pass layout.
    #[cfg(feature = "tiled")]
    #[allow(clippy::too_many_arguments)]
    pub fn create_subpass_render_pipeline(
        &self,
        shader: HalShaderSource,
        vertex_entry_point: &str,
        fragment_entry_point: &str,
        descriptor: &HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
        pass_layout: &HalSubpassPassLayout,
        subpass_index: u32,
    ) -> Result<HalRenderPipeline, HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan")))]
        let _ = (
            shader,
            vertex_entry_point,
            fragment_entry_point,
            descriptor,
            bindings,
            pass_layout,
            subpass_index,
        );
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => Ok(HalRenderPipeline::Noop),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device
                .create_subpass_render_pipeline(
                    shader,
                    vertex_entry_point,
                    fragment_entry_point,
                    descriptor,
                    bindings,
                    pass_layout,
                    subpass_index,
                )
                .map(HalRenderPipeline::Vulkan),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device
                .create_subpass_render_pipeline(
                    shader,
                    vertex_entry_point,
                    fragment_entry_point,
                    descriptor,
                    bindings,
                    pass_layout,
                    subpass_index,
                )
                .map(HalRenderPipeline::Metal),
            #[cfg(feature = "gles")]
            Self::Gles(_) => Err(HalError::BackendUnavailable { backend: "gles" }),
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
    #[cfg(feature = "gles")]
    /// GLES variant.
    Gles(gles::GlesSurface),
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
            #[cfg(feature = "gles")]
            (Self::Gles(surface), HalDevice::Gles(device)) => surface.configure(device, config),
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
            #[cfg(feature = "gles")]
            Self::Gles(surface) => surface.unconfigure(),
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
            #[cfg(feature = "gles")]
            Self::Gles(surface) => surface.acquire_next_texture().map(HalTexture::Gles),
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
            #[cfg(feature = "gles")]
            (Self::Gles(surface), HalQueue::Gles(queue)) => surface.present(queue),
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
    #[cfg(feature = "gles")]
    /// GLES variant.
    Gles(gles::GlesQueue),
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
            #[cfg(feature = "gles")]
            Self::Gles(queue) => queue.submit_empty(),
        }
    }

    /// Waits until all submitted queue work has completed.
    pub fn wait_idle(&self) -> Result<(), HalError> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(queue) => queue.wait_idle(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(queue) => queue.wait_idle(),
            #[cfg(feature = "metal")]
            Self::Metal(queue) => queue.wait_idle(),
            #[cfg(feature = "gles")]
            Self::Gles(queue) => queue.wait_idle(),
        }
    }

    /// Records and submits the given buffer/texture copy operations.
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
        #[cfg(not(any(feature = "noop", feature = "metal", feature = "vulkan")))]
        let _ = copies;
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(queue) => queue.submit_copies(copies),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(queue) => queue.submit_copies(copies),
            #[cfg(feature = "metal")]
            Self::Metal(queue) => queue.submit_copies(copies),
            #[cfg(feature = "gles")]
            Self::Gles(queue) => queue.submit_copies(copies),
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
    #[cfg(feature = "gles")]
    /// GLES variant.
    Gles(gles::GlesBuffer),
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
            #[cfg(feature = "gles")]
            Self::Gles(buffer) => buffer.size(),
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
            #[cfg(feature = "gles")]
            Self::Gles(buffer) => buffer.write(offset, data),
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
            #[cfg(feature = "gles")]
            Self::Gles(buffer) => buffer.read(offset, len),
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
            #[cfg(feature = "gles")]
            Self::Gles(buffer) => buffer.mapped_ptr(),
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
    #[cfg(feature = "gles")]
    /// GLES variant.
    Gles(gles::GlesTexture),
}

/// Enumerates HAL transient attachment values.
#[cfg(feature = "tiled")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalTransientAttachment {
    #[cfg(feature = "noop")]
    /// Noop variant.
    Noop(noop::NoopTransientAttachment),
    #[cfg(feature = "vulkan")]
    /// Vulkan variant.
    Vulkan(vulkan::VulkanTransientAttachment),
    #[cfg(feature = "metal")]
    /// Metal variant.
    Metal(metal::MetalTransientAttachment),
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
    #[cfg(feature = "gles")]
    /// GLES variant.
    Gles(gles::GlesSampler),
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
    #[cfg(feature = "gles")]
    /// GLES variant.
    Gles(gles::GlesComputePipeline),
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
    #[cfg(feature = "gles")]
    /// GLES variant.
    Gles(gles::GlesRenderPipeline),
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
            dimension: HalTextureDimension::D2,
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

    fn depth_texture_descriptor() -> HalTextureDescriptor {
        HalTextureDescriptor {
            dimension: HalTextureDimension::D2,
            format: HalTextureFormat::Depth32Float,
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
            color_targets: vec![HalColorTargetState {
                format: HalTextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: 0xf,
            }],
            depth_stencil: None,
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
    fn create_surface_from_android_native_window_noop_ignores_window_pointer(
    ) -> Result<(), HalError> {
        let instance = HalInstance::new_noop();
        let window = 0xdead_beefusize as *mut c_void;

        // SAFETY: Noop arm does not dereference the window pointer.
        let surface = unsafe { instance.create_surface_from_android_native_window(window)? };

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
        let buffer = device.create_buffer(256, HalBufferUsage::default());

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
            Some("fs_main"),
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
    fn hal_queue_wait_idle_noop_returns_ok() -> Result<(), HalError> {
        let device = noop_device()?;
        let queue = device.queue();

        queue.wait_idle()
    }

    #[test]
    fn hal_queue_submit_copies_noop_accepts_empty_and_buffer_copy() -> Result<(), HalError> {
        let device = noop_device()?;
        let queue = device.queue();
        let source = device.create_buffer(8, HalBufferUsage::default());
        let destination = device.create_buffer(8, HalBufferUsage::default());
        let clear_buffer = device.create_buffer(8, HalBufferUsage::default());
        let copy = HalCopy::Buffer(HalBufferCopy {
            source,
            source_offset: 0,
            destination,
            destination_offset: 0,
            size: 8,
        });
        let clear = HalCopy::BufferClear(HalBufferClear {
            buffer: clear_buffer,
            offset: 0,
            size: 8,
        });

        queue.submit_copies(&[])?;
        queue.submit_copies(&[copy, clear])
    }

    #[test]
    fn hal_queue_submit_copies_noop_records_depth_only_render_pass() -> Result<(), HalError> {
        let device = noop_device()?;
        let queue = device.queue();
        let depth = device.create_texture(&depth_texture_descriptor());

        queue.submit_copies(&[HalCopy::RenderPass(HalRenderPass {
            pipeline: None,
            color_targets: Vec::new(),
            depth_stencil_attachment: Some(HalRenderDepthStencilAttachment {
                texture: depth,
                format: HalTextureFormat::Depth32Float,
                mip_level: 0,
                array_layer: 0,
                depth_load_op: HalRenderLoadOp::Clear,
                depth_store: true,
                depth_clear_value: 0.25,
                depth_read_only: false,
                stencil_load_op: HalRenderLoadOp::Clear,
                stencil_store: false,
                stencil_clear_value: 3,
                stencil_read_only: true,
            }),
            bind_buffers: Vec::new(),
            bind_textures: Vec::new(),
            bind_samplers: Vec::new(),
            vertex_buffers: Vec::new(),
            index_buffer: None,
            indirect_buffer: None,
            blend_constant: [0.0; 4],
            stencil_reference: 0,
            draw: None,
        })])?;

        let submitted = match &queue {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            #[cfg(any(feature = "vulkan", feature = "metal", feature = "gles"))]
            _ => Vec::new(),
        };
        assert!(matches!(
            submitted.as_slice(),
            [HalCopy::RenderPass(pass)]
                if pass.color_targets.is_empty()
                    && pass.depth_stencil_attachment.as_ref().is_some_and(|attachment|
                        attachment.format == HalTextureFormat::Depth32Float
                            && (attachment.depth_clear_value - 0.25).abs() < f32::EPSILON
                            && attachment.stencil_clear_value == 3
                    )
        ));
        Ok(())
    }

    #[test]
    fn hal_buffer_size_noop_matches_creation_size() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(4096, HalBufferUsage::default());

        assert_eq!(buffer.size(), 4096);
        Ok(())
    }

    #[test]
    fn hal_buffer_write_noop_accepts_empty_and_non_empty_data() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(16, HalBufferUsage::default());

        buffer.write(0, &[])?;
        buffer.write(4, &[1, 2, 3, 4])
    }

    #[test]
    fn hal_buffer_read_noop_returns_zeroed_vector() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(16, HalBufferUsage::default());

        assert_eq!(buffer.read(0, 0)?, Vec::<u8>::new());
        assert_eq!(buffer.read(4, 4)?, vec![0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn hal_buffer_mapped_ptr_noop_returns_none() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(16, HalBufferUsage::default());

        assert!(buffer.mapped_ptr().is_none());
        Ok(())
    }
}
