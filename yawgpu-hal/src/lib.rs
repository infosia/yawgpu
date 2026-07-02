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
    HalBoundBuffer, HalBoundExternalTexture, HalBoundIndexBuffer, HalBoundIndirectBuffer,
    HalBoundSampler, HalBoundTexture, HalBufferBindingKind, HalBufferClear, HalBufferCopy,
    HalBufferTextureCopy, HalBufferTextureLayout, HalComputeDispatch, HalComputePass, HalCopy,
    HalDescriptorBinding, HalDescriptorBindingKind, HalDraw, HalIndexFormat, HalRenderColorTarget,
    HalRenderDepthStencilAttachment, HalRenderLoadOp, HalRenderPass, HalResolveQuerySet,
    HalScissorRect, HalStorageTextureAccess, HalTextureAspect, HalTextureClear, HalTextureCopy,
    HalTextureViewDimension, HalViewport,
};
#[cfg(feature = "tiled")]
pub use command::{
    HalSubpassAttachmentLayout, HalSubpassAttachmentResource, HalSubpassColorAttachment,
    HalSubpassDependency, HalSubpassDependencyType, HalSubpassDepthStencilAttachment,
    HalSubpassDraw, HalSubpassInputAttachment, HalSubpassLayout, HalSubpassPassLayout,
    HalSubpassRenderPass, HalSubpassRenderPassCommand,
};
pub use descriptors::{
    HalBlendComponent, HalBlendFactor, HalBlendOperation, HalBlendState, HalColorTargetState,
    HalComponentSwizzle, HalCullMode, HalDepthStencilState, HalExtent3d, HalFrontFace, HalOrigin3d,
    HalRenderPipelineDescriptor, HalSamplerDescriptor, HalStencilFaceState,
    HalTextureComponentSwizzle, HalTextureDescriptor, HalTextureDimension, HalVertexAttribute,
    HalVertexBufferLayout,
};
pub use error::HalError;
pub use format::{
    HalAddressMode, HalBufferUsage, HalColorClearKind, HalCompareFunction, HalFilterMode,
    HalMipmapFilterMode, HalPrimitiveTopology, HalStencilOperation, HalTextureFormat,
    HalTextureUsage, HalVertexFormat, HalVertexStepMode,
};
pub use present::{HalPresentMode, HalSurfaceConfiguration};
pub use shader::{HalMslBufferSizeBinding, HalShaderSource, HalShaderStage};

/// Stores backend-reported supported adapter limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct HalLimits {
    /// Max texture dimension 1d.
    pub max_texture_dimension_1d: u32,
    /// Max texture dimension 2d.
    pub max_texture_dimension_2d: u32,
    /// Max texture dimension 3d.
    pub max_texture_dimension_3d: u32,
    /// Max texture array layers.
    pub max_texture_array_layers: u32,
    /// Max bind groups.
    pub max_bind_groups: u32,
    /// Max bind groups plus vertex buffers.
    pub max_bind_groups_plus_vertex_buffers: u32,
    /// Max bindings per bind group.
    pub max_bindings_per_bind_group: u32,
    /// Max dynamic uniform buffers per pipeline layout.
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    /// Max dynamic storage buffers per pipeline layout.
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,
    /// Max sampled textures per shader stage.
    pub max_sampled_textures_per_shader_stage: u32,
    /// Max samplers per shader stage.
    pub max_samplers_per_shader_stage: u32,
    /// Max storage buffers per shader stage.
    pub max_storage_buffers_per_shader_stage: u32,
    /// Max storage textures per shader stage.
    pub max_storage_textures_per_shader_stage: u32,
    /// Max storage buffers usable in the vertex stage.
    pub max_storage_buffers_in_vertex_stage: u32,
    /// Max storage buffers usable in the fragment stage.
    pub max_storage_buffers_in_fragment_stage: u32,
    /// Max storage textures usable in the vertex stage.
    pub max_storage_textures_in_vertex_stage: u32,
    /// Max storage textures usable in the fragment stage.
    pub max_storage_textures_in_fragment_stage: u32,
    /// Max uniform buffers per shader stage.
    pub max_uniform_buffers_per_shader_stage: u32,
    /// Max uniform buffer binding size.
    pub max_uniform_buffer_binding_size: u64,
    /// Max storage buffer binding size.
    pub max_storage_buffer_binding_size: u64,
    /// Min uniform buffer offset alignment.
    pub min_uniform_buffer_offset_alignment: u32,
    /// Min storage buffer offset alignment.
    pub min_storage_buffer_offset_alignment: u32,
    /// Max vertex buffers.
    pub max_vertex_buffers: u32,
    /// Max buffer size.
    pub max_buffer_size: u64,
    /// Max vertex attributes.
    pub max_vertex_attributes: u32,
    /// Max vertex buffer array stride.
    pub max_vertex_buffer_array_stride: u32,
    /// Max inter stage shader variables.
    pub max_inter_stage_shader_variables: u32,
    /// Max color attachments.
    pub max_color_attachments: u32,
    /// Max color attachment bytes per sample.
    pub max_color_attachment_bytes_per_sample: u32,
    /// Max compute workgroup storage size.
    pub max_compute_workgroup_storage_size: u32,
    /// Max compute invocations per workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// Max compute workgroup size x.
    pub max_compute_workgroup_size_x: u32,
    /// Max compute workgroup size y.
    pub max_compute_workgroup_size_y: u32,
    /// Max compute workgroup size z.
    pub max_compute_workgroup_size_z: u32,
    /// Max compute workgroups per dimension.
    pub max_compute_workgroups_per_dimension: u32,
    /// Max immediate size.
    pub max_immediate_size: u32,
}

impl HalLimits {
    /// Constant value for default.
    pub const DEFAULT: Self = Self {
        max_texture_dimension_1d: 8192,
        max_texture_dimension_2d: 8192,
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
        max_storage_buffers_in_vertex_stage: 8,
        max_storage_buffers_in_fragment_stage: 8,
        max_storage_textures_in_vertex_stage: 4,
        max_storage_textures_in_fragment_stage: 4,
        max_uniform_buffers_per_shader_stage: 12,
        max_uniform_buffer_binding_size: 65_536,
        max_storage_buffer_binding_size: 128 * 1024 * 1024,
        min_uniform_buffer_offset_alignment: 256,
        min_storage_buffer_offset_alignment: 256,
        max_vertex_buffers: 8,
        max_buffer_size: 256 * 1024 * 1024,
        max_vertex_attributes: 16,
        max_vertex_buffer_array_stride: 2048,
        max_inter_stage_shader_variables: 16,
        max_color_attachments: 8,
        max_color_attachment_bytes_per_sample: 32,
        max_compute_workgroup_storage_size: 16_384,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_size_y: 256,
        max_compute_workgroup_size_z: 64,
        max_compute_workgroups_per_dimension: 65_535,
        max_immediate_size: 0,
    };
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

    /// Returns the backend-reported supported limits.
    #[must_use]
    pub fn limits(&self) -> HalLimits {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.limits(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.limits(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.limits(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.limits(),
        }
    }

    /// Returns true when BC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_bc(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => true,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_texture_compression_bc(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_texture_compression_bc(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_texture_compression_bc(),
        }
    }

    /// Returns true when 3D BC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_bc_sliced_3d(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => true,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_texture_compression_bc_sliced_3d(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_texture_compression_bc_sliced_3d(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_texture_compression_bc_sliced_3d(),
        }
    }

    /// Returns true when ETC2/EAC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_etc2(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => true,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_texture_compression_etc2(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_texture_compression_etc2(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_texture_compression_etc2(),
        }
    }

    /// Returns true when ASTC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_astc(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => true,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_texture_compression_astc(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_texture_compression_astc(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_texture_compression_astc(),
        }
    }

    /// Returns true when 3D ASTC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_astc_sliced_3d(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => true,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_texture_compression_astc_sliced_3d(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_texture_compression_astc_sliced_3d(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_texture_compression_astc_sliced_3d(),
        }
    }

    /// Returns true when texture view component swizzling is supported.
    #[must_use]
    pub fn supports_texture_component_swizzle(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.supports_texture_component_swizzle(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_texture_component_swizzle(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_texture_component_swizzle(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_texture_component_swizzle(),
        }
    }

    /// Returns true when WGSL `shader-f16` is supported.
    #[must_use]
    pub fn supports_shader_float16(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.supports_shader_float16(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_shader_float16(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_shader_float16(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_shader_float16(),
        }
    }

    /// Returns true when WGSL `subgroups` is supported.
    #[must_use]
    pub fn supports_subgroups(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.supports_subgroups(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_subgroups(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_subgroups(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_subgroups(),
        }
    }

    /// Returns true when depth clip control is supported.
    #[must_use]
    pub fn supports_depth_clip_control(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.supports_depth_clip_control(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_depth_clip_control(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_depth_clip_control(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_depth_clip_control(),
        }
    }

    /// Returns true when float32 color target blending is supported.
    #[must_use]
    pub fn supports_float32_blendable(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.supports_float32_blendable(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_float32_blendable(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_float32_blendable(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_float32_blendable(),
        }
    }

    /// Returns true when dual-source blending is supported.
    #[must_use]
    pub fn supports_dual_source_blending(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.supports_dual_source_blending(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_dual_source_blending(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_dual_source_blending(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_dual_source_blending(),
        }
    }

    /// Returns true when WGSL clip distances are supported.
    #[must_use]
    pub fn supports_clip_distances(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.supports_clip_distances(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_clip_distances(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_clip_distances(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_clip_distances(),
        }
    }

    /// Returns true when WGSL primitive index is supported.
    #[must_use]
    pub fn supports_primitive_index(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.supports_primitive_index(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_primitive_index(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_primitive_index(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_primitive_index(),
        }
    }

    /// Returns true when indirect draws support non-zero first instance values.
    #[must_use]
    pub fn supports_indirect_first_instance(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.supports_indirect_first_instance(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.supports_indirect_first_instance(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.supports_indirect_first_instance(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.supports_indirect_first_instance(),
        }
    }

    /// Returns the supported subgroup size range, or `None` when unsupported.
    #[must_use]
    pub fn subgroup_size_range(&self) -> Option<(u32, u32)> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(adapter) => adapter.subgroup_size_range(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(adapter) => adapter.subgroup_size_range(),
            #[cfg(feature = "metal")]
            Self::Metal(adapter) => adapter.subgroup_size_range(),
            #[cfg(feature = "gles")]
            Self::Gles(adapter) => adapter.subgroup_size_range(),
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

/// Enumerates HAL query set kind values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalQueryKind {
    /// Occlusion query set.
    Occlusion,
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

    /// Returns true when `VK_KHR_vulkan_memory_model` / `vulkanMemoryModel` was
    /// enabled at device creation (Vulkan only).
    #[must_use]
    pub fn vulkan_memory_model(&self) -> bool {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(_) => false,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.vulkan_memory_model(),
            #[cfg(feature = "metal")]
            Self::Metal(_) => false,
            #[cfg(feature = "gles")]
            Self::Gles(_) => false,
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
    pub fn create_buffer(&self, size: u64, usage: HalBufferUsage) -> Result<HalBuffer, HalError> {
        Ok(match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalBuffer::Noop(device.create_buffer(size, usage)?),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => HalBuffer::Vulkan(device.create_buffer(size, usage)?),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalBuffer::Metal(device.create_buffer(size, usage)?),
            #[cfg(feature = "gles")]
            Self::Gles(device) => HalBuffer::Gles(device.create_buffer(size, usage)?),
        })
    }

    /// Creates a texture matching the given descriptor.
    pub fn create_texture(
        &self,
        descriptor: &HalTextureDescriptor,
    ) -> Result<HalTexture, HalError> {
        #[cfg(not(any(feature = "metal", feature = "vulkan", feature = "gles")))]
        let _ = descriptor;
        Ok(match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => HalTexture::Noop(device.create_texture(descriptor)?),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => HalTexture::Vulkan(device.create_texture(descriptor)?),
            #[cfg(feature = "metal")]
            Self::Metal(device) => HalTexture::Metal(device.create_texture(descriptor)?),
            #[cfg(feature = "gles")]
            Self::Gles(device) => HalTexture::Gles(device.create_texture(descriptor)?),
        })
    }

    /// Creates a query set matching the given kind and count.
    pub fn create_query_set(
        &self,
        kind: HalQueryKind,
        count: u32,
    ) -> Result<HalQuerySet, HalError> {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop(device) => Ok(HalQuerySet::Noop {
                count: device.create_query_set(kind, count),
            }),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device
                .create_query_set(kind, count)
                .map(HalQuerySet::Vulkan),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.create_query_set(kind, count).map(HalQuerySet::Metal),
            #[cfg(feature = "gles")]
            Self::Gles(_) => Ok(HalQuerySet::Gles { count }),
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

    /// Creates a subpass-compatible render pipeline.
    #[cfg(feature = "tiled")]
    #[allow(clippy::too_many_arguments)]
    pub fn create_subpass_render_pipeline(
        &self,
        shader: HalShaderSource,
        vertex_entry_point: &str,
        fragment_entry_point: Option<&str>,
        descriptor: &HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
        pass_layout: &HalSubpassPassLayout,
        subpass_index: u32,
    ) -> Result<HalRenderPipeline, HalError> {
        #[cfg(not(any(feature = "gles", feature = "metal", feature = "vulkan")))]
        let _ = (
            shader,
            vertex_entry_point,
            fragment_entry_point,
            descriptor,
            bindings,
            pass_layout,
            subpass_index,
        );
        let _ = (pass_layout, subpass_index);
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
                .create_render_pipeline(
                    shader,
                    vertex_entry_point,
                    fragment_entry_point,
                    descriptor,
                    bindings,
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
            Self::Noop(buffer) => buffer.write(offset, data),
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
            Self::Noop(buffer) => buffer.read(offset, len),
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

/// Enumerates HAL query-set values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalQuerySet {
    #[cfg(feature = "noop")]
    /// Noop query-set variant.
    Noop {
        /// Number of queries in the set.
        count: u32,
    },
    #[cfg(feature = "vulkan")]
    /// Vulkan query-set variant.
    Vulkan(vulkan::VulkanQuerySet),
    #[cfg(feature = "metal")]
    /// Metal query-set variant.
    Metal(metal::MetalQuerySet),
    #[cfg(feature = "gles")]
    /// GLES placeholder query-set variant.
    Gles {
        /// Number of queries in the set.
        count: u32,
    },
}

impl HalQuerySet {
    /// Returns the number of queries in this set.
    #[must_use]
    pub fn count(&self) -> u32 {
        match self {
            #[cfg(feature = "noop")]
            Self::Noop { count } => *count,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(query_set) => query_set.count(),
            #[cfg(feature = "metal")]
            Self::Metal(query_set) => query_set.count(),
            #[cfg(feature = "gles")]
            Self::Gles { count } => *count,
        }
    }
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
                transient: false,
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
                transient: false,
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
                transient: false,
            },
            640,
            480,
            HalPresentMode::Fifo,
        )
    }

    fn render_pipeline_descriptor() -> HalRenderPipelineDescriptor {
        HalRenderPipelineDescriptor {
            sample_count: 1,
            sample_mask: u32::MAX,
            alpha_to_coverage_enabled: false,
            color_targets: vec![Some(HalColorTargetState {
                format: HalTextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: 0xf,
            })],
            depth_stencil: None,
            vertex_buffers: Vec::new(),
            primitive_topology: HalPrimitiveTopology::TriangleList,
            front_face: HalFrontFace::Ccw,
            cull_mode: HalCullMode::None,
            unclipped_depth: false,
            needs_frag_depth_range_push_constant: false,
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
    fn hal_adapter_supports_shader_float16_noop_returns_true() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter exists");

        assert!(adapter.supports_shader_float16());
    }

    #[test]
    fn hal_adapter_supports_subgroups_noop_returns_nominal_range() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter exists");

        assert!(adapter.supports_subgroups());
        assert_eq!(adapter.subgroup_size_range(), Some((4, 4)));
    }

    #[test]
    fn hal_adapter_supports_depth_clip_control_noop_returns_true() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter exists");

        assert!(adapter.supports_depth_clip_control());
    }

    #[test]
    fn hal_adapter_supports_float32_blendable_noop_returns_true() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter exists");

        assert!(adapter.supports_float32_blendable());
    }

    #[test]
    fn hal_adapter_supports_dual_source_blending_noop_returns_true() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter exists");

        assert!(adapter.supports_dual_source_blending());
    }

    #[test]
    fn hal_adapter_supports_clip_distances_noop_returns_true() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter exists");

        assert!(adapter.supports_clip_distances());
    }

    #[test]
    fn hal_adapter_supports_primitive_index_noop_returns_true() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter exists");

        assert!(adapter.supports_primitive_index());
    }

    #[test]
    fn hal_adapter_supports_indirect_first_instance_noop_returns_true() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter exists");

        assert!(adapter.supports_indirect_first_instance());
    }

    #[test]
    fn hal_adapter_supports_texture_component_swizzle_noop_returns_true() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter exists");

        assert!(adapter.supports_texture_component_swizzle());
    }

    #[test]
    fn hal_device_backend_noop_returns_noop() -> Result<(), HalError> {
        let device = noop_device()?;

        assert_eq!(device.backend(), HalBackend::Noop);
        Ok(())
    }

    #[test]
    fn hal_device_vulkan_memory_model_noop_returns_false() -> Result<(), HalError> {
        let device = noop_device()?;

        assert!(!device.vulkan_memory_model());
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
        let buffer = device.create_buffer(256, HalBufferUsage::default())?;

        assert!(matches!(buffer, HalBuffer::Noop(_)));
        assert_eq!(buffer.size(), 256);
        Ok(())
    }

    #[test]
    fn hal_device_create_texture_noop_returns_texture_and_increments_allocations(
    ) -> Result<(), HalError> {
        let device = noop_device()?;
        let texture = device.create_texture(&texture_descriptor())?;

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

    #[cfg(feature = "tiled")]
    #[test]
    fn hal_device_subpass_render_pass_noop_advances_and_ends() -> Result<(), HalError> {
        let device = noop_device()?;
        let mut pass = device.begin_subpass_render_pass()?;

        match &pass {
            HalSubpassRenderPass::Noop(pass) => assert_eq!(pass.active_subpass(), 0),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            _ => panic!("unexpected non-Noop subpass pass"),
        }
        device.next_subpass_render_pass(&mut pass)?;
        match &pass {
            HalSubpassRenderPass::Noop(pass) => assert_eq!(pass.active_subpass(), 1),
            #[cfg(any(feature = "vulkan", feature = "metal"))]
            _ => panic!("unexpected non-Noop subpass pass"),
        }
        device.end_subpass_render_pass(pass)
    }

    #[cfg(feature = "tiled")]
    #[test]
    fn hal_subpass_render_pass_command_constructs_with_empty_draws() {
        let command = HalSubpassRenderPassCommand {
            layout: HalSubpassPassLayout {
                color_attachments: vec![HalSubpassAttachmentLayout {
                    format: HalTextureFormat::Rgba8Unorm,
                    sample_count: 1,
                }],
                depth_stencil_attachment: None,
                subpasses: vec![HalSubpassLayout {
                    color_attachment_indices: vec![0],
                    uses_depth_stencil: false,
                    input_attachments: Vec::new(),
                }],
                dependencies: Vec::new(),
            },
            extent: HalExtent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            color_attachments: Vec::new(),
            depth_stencil_attachment: None,
            draws: Vec::new(),
        };

        assert_eq!(command.layout.subpasses.len(), 1);
        assert!(command.draws.is_empty());
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
        let source = device.create_buffer(8, HalBufferUsage::default())?;
        let destination = device.create_buffer(8, HalBufferUsage::default())?;
        let clear_buffer = device.create_buffer(8, HalBufferUsage::default())?;
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
    fn hal_query_set_noop_creates_and_resolves_zeroes() -> Result<(), HalError> {
        let device = noop_device()?;
        let queue = device.queue();
        let query_set = device.create_query_set(HalQueryKind::Occlusion, 2)?;
        let destination = device.create_buffer(16, HalBufferUsage::default())?;

        assert_eq!(query_set.count(), 2);
        destination.write(0, &[1; 16])?;
        queue.submit_copies(&[HalCopy::ResolveQuerySet(HalResolveQuerySet {
            query_set,
            first_query: 0,
            query_count: 2,
            written_queries: Vec::new(),
            destination: destination.clone(),
            destination_offset: 0,
        })])?;

        assert_eq!(destination.read(0, 16)?, [0; 16]);
        Ok(())
    }

    #[test]
    fn hal_queue_submit_copies_noop_records_depth_only_render_pass() -> Result<(), HalError> {
        let device = noop_device()?;
        let queue = device.queue();
        let depth = device.create_texture(&depth_texture_descriptor())?;

        queue.submit_copies(&[HalCopy::RenderPass(HalRenderPass {
            pipeline: None,
            color_targets: Vec::new(),
            framebuffer_fetch_color_slots: Vec::new(),
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
            bind_external_textures: Vec::new(),
            vertex_buffers: Vec::new(),
            index_buffer: None,
            indirect_buffer: None,
            viewport: None,
            scissor_rect: None,
            blend_constant: [0.0; 4],
            stencil_reference: 0,
            occlusion_query_set: None,
            occlusion_query_index: None,
            draw: None,
            immediate_data: Vec::new(),
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

    /// Block 94 S1: the Noop backend "accepts and records, executes as
    /// no-op" for `HalRenderPass::immediate_data` -- the field round-trips
    /// through `submit_copies` without error or mutation, but Noop performs
    /// no actual delivery (that lands in S2/S3 for Metal/Vulkan).
    #[test]
    fn hal_queue_submit_copies_noop_records_render_pass_immediate_data() -> Result<(), HalError> {
        let device = noop_device()?;
        let queue = device.queue();
        let depth = device.create_texture(&depth_texture_descriptor())?;

        queue.submit_copies(&[HalCopy::RenderPass(HalRenderPass {
            pipeline: None,
            color_targets: Vec::new(),
            framebuffer_fetch_color_slots: Vec::new(),
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
            bind_external_textures: Vec::new(),
            vertex_buffers: Vec::new(),
            index_buffer: None,
            indirect_buffer: None,
            viewport: None,
            scissor_rect: None,
            blend_constant: [0.0; 4],
            stencil_reference: 0,
            occlusion_query_set: None,
            occlusion_query_index: None,
            draw: None,
            immediate_data: vec![1, 2, 3, 4],
        })])?;

        let submitted = match &queue {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            #[cfg(any(feature = "vulkan", feature = "metal", feature = "gles"))]
            _ => Vec::new(),
        };
        assert!(matches!(
            submitted.as_slice(),
            [HalCopy::RenderPass(pass)] if pass.immediate_data == vec![1, 2, 3, 4]
        ));
        Ok(())
    }

    /// Block 94 S1: same Noop "accepts and no-ops" contract as above, for
    /// `HalComputePass::immediate_data`.
    #[test]
    fn hal_queue_submit_copies_noop_records_compute_pass_immediate_data() -> Result<(), HalError> {
        let device = noop_device()?;
        let queue = device.queue();
        let pipeline = device.create_compute_pipeline(
            HalShaderSource::Msl(String::new()),
            "main",
            (1, 1, 1),
            &[],
        )?;

        queue.submit_copies(&[HalCopy::ComputePass(HalComputePass {
            pipeline,
            bind_buffers: Vec::new(),
            bind_textures: Vec::new(),
            bind_samplers: Vec::new(),
            bind_external_textures: Vec::new(),
            immediate_data: vec![5, 6, 7, 8],
            dispatch: HalComputeDispatch::Direct {
                workgroups: (1, 1, 1),
            },
        })])?;

        let submitted = match &queue {
            HalQueue::Noop(queue) => queue.submitted_copies(),
            #[cfg(any(feature = "vulkan", feature = "metal", feature = "gles"))]
            _ => Vec::new(),
        };
        assert!(matches!(
            submitted.as_slice(),
            [HalCopy::ComputePass(pass)] if pass.immediate_data == vec![5, 6, 7, 8]
        ));
        Ok(())
    }

    #[test]
    fn hal_buffer_size_noop_matches_creation_size() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(4096, HalBufferUsage::default())?;

        assert_eq!(buffer.size(), 4096);
        Ok(())
    }

    #[test]
    fn hal_limits_default_matches_webgpu_core_floor() {
        let limits = HalLimits::DEFAULT;

        assert_eq!(limits.max_bind_groups, 4);
        assert_eq!(limits.max_texture_dimension_2d, 8192);
        assert_eq!(limits.max_uniform_buffer_binding_size, 65_536);
        assert_eq!(limits.max_storage_buffer_binding_size, 128 * 1024 * 1024);
        assert_eq!(limits.min_uniform_buffer_offset_alignment, 256);
        assert_eq!(limits.max_immediate_size, 0);
    }

    #[test]
    fn hal_adapter_limits_noop_reports_max_immediate_size_64_else_default() {
        let adapter = HalInstance::new_noop()
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("noop adapter");

        // Block 94 S1: Noop is the first backend to flip `max_immediate_size`
        // to Dawn's `kMaxImmediateDataBytes` (64); every other limit stays at
        // the shared `HalLimits::DEFAULT` floor.
        assert_eq!(
            adapter.limits(),
            HalLimits {
                max_immediate_size: 64,
                ..HalLimits::DEFAULT
            }
        );
    }

    #[test]
    fn hal_buffer_write_noop_accepts_empty_and_non_empty_data() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(16, HalBufferUsage::default())?;

        buffer.write(0, &[])?;
        buffer.write(4, &[1, 2, 3, 4])
    }

    #[test]
    fn hal_buffer_noop_round_trips_written_bytes() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(16, HalBufferUsage::default())?;

        assert_eq!(buffer.read(0, 0)?, Vec::<u8>::new());
        assert_eq!(buffer.read(4, 4)?, vec![0, 0, 0, 0]);
        buffer.write(5, &[9, 8, 7])?;
        assert_eq!(buffer.read(4, 5)?, vec![0, 9, 8, 7, 0]);
        Ok(())
    }

    #[test]
    fn hal_buffer_mapped_ptr_noop_returns_none() -> Result<(), HalError> {
        let device = noop_device()?;
        let buffer = device.create_buffer(16, HalBufferUsage::default())?;

        assert!(buffer.mapped_ptr().is_none());
        Ok(())
    }
}
