use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use glow::HasContext;
use parking_lot::Mutex;
use parking_lot::MutexGuard;

use super::buffer::GlesBuffer;
use super::egl::{EglContext, EglSurface};
use super::format::GlesColorRenderCaps;
use super::instance::{EglInstanceState, GlesInstanceInner};
use super::pipeline::{GlesComputePipeline, GlesPipelineResourceBindings, GlesRenderPipeline};
use super::queue::GlesQueue;
use super::sampler::{create_nearest_placeholder_sampler, GlesSampler};
use super::texture::GlesTexture;
use super::{rebuild_hal_error, BACKEND};
use crate::{
    HalBufferUsage, HalDescriptorBinding, HalError, HalRenderPipelineDescriptor,
    HalSamplerDescriptor, HalShaderSource, HalShaderStage, HalTextureDescriptor,
};

pub(super) type GlesSampleMaskIFn = unsafe extern "system" fn(u32, u32);
pub(super) type GlesTextureViewFn =
    unsafe extern "system" fn(u32, u32, u32, u32, u32, u32, u32, u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum TextureToBufferComputeEncoding {
    R8Snorm,
    Rg8Snorm,
    Rgba8Snorm,
    R16Unorm,
    R16Snorm,
    Rg16Unorm,
    Rg16Snorm,
    Rgba16Unorm,
    Rgba16Snorm,
    Rgb9e5Ufloat,
    Depth16Unorm,
    Depth24Plus,
    Depth32Float,
}

impl TextureToBufferComputeEncoding {
    pub(super) fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::R8Snorm => 1,
            Self::Rg8Snorm | Self::R16Unorm | Self::R16Snorm | Self::Depth16Unorm => 2,
            Self::Rgba8Snorm
            | Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rgb9e5Ufloat
            | Self::Depth24Plus
            | Self::Depth32Float => 4,
            Self::Rgba16Unorm | Self::Rgba16Snorm => 8,
        }
    }

    pub(super) fn shader_store(self) -> &'static str {
        match self {
            Self::R8Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeByte(base, packSnorm4x8(vec4(value.r, 0.0, 0.0, 0.0)) & 0xffu);"
            }
            Self::Rg8Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU16(base, packSnorm4x8(vec4(value.rg, 0.0, 0.0)) & 0xffffu);"
            }
            Self::Rgba8Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packSnorm4x8(value));"
            }
            Self::R16Unorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU16(base, packUnorm16(value.r));"
            }
            Self::R16Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU16(base, packSnorm2x16(vec2(value.r, 0.0)) & 0xffffu);"
            }
            Self::Rg16Unorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packUnorm16(value.r) | (packUnorm16(value.g) << 16));"
            }
            Self::Rg16Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packSnorm2x16(value.rg));"
            }
            Self::Rgba16Unorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packUnorm16(value.r) | (packUnorm16(value.g) << 16));\n\
                 writeU32(base + 4u, packUnorm16(value.b) | (packUnorm16(value.a) << 16));"
            }
            Self::Rgba16Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packSnorm2x16(value.rg));\n\
                 writeU32(base + 4u, packSnorm2x16(value.ba));"
            }
            Self::Rgb9e5Ufloat => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packRgb9e5(value.rgb));"
            }
            Self::Depth16Unorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU16(base, packUnorm16(value.r));"
            }
            Self::Depth24Plus => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, uint(round(clamp(value.r, 0.0, 1.0) * 16777215.0)));"
            }
            Self::Depth32Float => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, floatBitsToUint(value.r));"
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct TextureToBufferComputeProgramKey {
    pub(super) target: u32,
    pub(super) encoding: TextureToBufferComputeEncoding,
}

#[derive(Clone)]
pub(super) struct TextureToBufferComputeProgram {
    pub(super) program: glow::Program,
    pub(super) u_texture: Option<glow::UniformLocation>,
    pub(super) u_mip: Option<glow::UniformLocation>,
    pub(super) u_origin: Option<glow::UniformLocation>,
    pub(super) u_extent: Option<glow::UniformLocation>,
}

pub(super) struct GlesDeviceCaps {
    pub(super) supports_base_vertex: bool,
    pub(super) color_render_caps: GlesColorRenderCaps,
    pub(super) supports_vertex_array_bgra: bool,
    pub(super) max_samples: i32,
    pub(super) sample_mask_i: Option<GlesSampleMaskIFn>,
    pub(super) supports_texture_view: bool,
    pub(super) supports_cube_map_array: bool,
    pub(super) texture_view: Option<GlesTextureViewFn>,
}

pub(super) enum GlesDeviceInner {
    Egl(EglDeviceState),
    #[cfg(windows)]
    Wgl(super::wgl::WglDeviceState),
}

pub(super) struct EglDeviceState {
    pub(super) instance: Arc<GlesInstanceInner>,
    pub(super) context: EglContext,
    pub(super) surface: EglSurface,
    pub(super) gl: glow::Context,
    current_lock: Mutex<()>,
    pub(super) allocations: AtomicU64,
    /// Whether the context supports the base-vertex indexed-draw entry
    /// points (GLES 3.2 core or `GL_OES/EXT_draw_elements_base_vertex`);
    /// detected once at device creation (T-G11).
    pub(super) supports_base_vertex: bool,
    /// Extension-gated float color-renderability caps
    /// (`GL_EXT_color_buffer_float` / `GL_EXT_color_buffer_half_float`);
    /// detected once at device creation (T-G12).
    pub(super) color_render_caps: GlesColorRenderCaps,
    /// Whether `glVertexAttribPointer(size = GL_BGRA, type = GL_UNSIGNED_BYTE,
    /// normalized = GL_TRUE)` is supported for BGRA-order vertex fetch.
    pub(super) supports_vertex_array_bgra: bool,
    /// Maximum sample count reported by `GL_MAX_SAMPLES`.
    pub(super) max_samples: i32,
    /// GLES 3.1 core `glSampleMaski`; cached because glow 0.14 does not expose
    /// a public wrapper on `HasContext`.
    pub(super) sample_mask_i: Option<GlesSampleMaskIFn>,
    /// Whether `glTextureView` is supported by GLES 3.2 or
    /// `GL_OES/EXT_texture_view`, and the entry point loaded successfully.
    pub(super) supports_texture_view: bool,
    /// Whether cube-array texture targets are supported by GLES 3.2 or the
    /// cube-map-array extension.
    pub(super) supports_cube_map_array: bool,
    /// Manually loaded `glTextureView`; glow 0.14 has no `HasContext` wrapper
    /// for it.
    pub(super) texture_view: Option<GlesTextureViewFn>,
    /// Internal NEAREST sampler used for Tint placeholder combined samplers
    /// emitted for samplerless textureLoad. Integer/stencil textures are
    /// incomplete with the texture object's default LINEAR filtering.
    pub(super) placeholder_sampler: Result<glow::Sampler, HalError>,
    pub(super) texture_to_buffer_compute_programs:
        Mutex<HashMap<TextureToBufferComputeProgramKey, TextureToBufferComputeProgram>>,
}

// SAFETY: All access to the EGL context and `glow::Context` goes through
// `with_current_context`, which holds `current_lock` while making the context
// current and executing GL commands.
unsafe impl Send for GlesDeviceInner {}
// SAFETY: See the `Send` impl; shared references are synchronized by
// `current_lock`, and resource teardown only runs after the final `Arc` drops.
unsafe impl Sync for GlesDeviceInner {}

impl Drop for EglDeviceState {
    fn drop(&mut self) {
        if let GlesInstanceInner::Egl(egl_state) = self.instance.as_ref() {
            let _ = egl_state.egl.make_current(
                egl_state.display,
                Some(self.surface),
                Some(self.surface),
                Some(self.context),
            );
            if let Ok(sampler) = self.placeholder_sampler.as_ref() {
                unsafe {
                    self.gl.delete_sampler(*sampler);
                }
            }
            for cached in self.texture_to_buffer_compute_programs.get_mut().drain() {
                unsafe {
                    self.gl.delete_program(cached.1.program);
                }
            }
            let _ = egl_state
                .egl
                .make_current(egl_state.display, None, None, None);
            let _ = egl_state
                .egl
                .destroy_surface(egl_state.display, self.surface);
            let _ = egl_state
                .egl
                .destroy_context(egl_state.display, self.context);
        }
    }
}

impl GlesDeviceInner {
    pub(super) fn current_lock_acquire(&self) -> MutexGuard<'_, ()> {
        match self {
            Self::Egl(state) => state.current_lock.lock(),
            #[cfg(windows)]
            Self::Wgl(state) => state.current_lock_acquire(),
        }
    }

    pub(super) fn with_current_context<R>(
        &self,
        f: impl FnOnce(&glow::Context) -> R,
    ) -> Result<R, HalError> {
        match self {
            Self::Egl(state) => state.with_current_context(f),
            #[cfg(windows)]
            Self::Wgl(state) => state.with_current_context(f),
        }
    }

    pub(super) fn egl_state(&self) -> Option<&EglDeviceState> {
        match self {
            Self::Egl(state) => Some(state),
            #[cfg(windows)]
            Self::Wgl(_) => None,
        }
    }

    /// Whether the context supports the base-vertex indexed-draw entry
    /// points (GLES 3.2 core or `GL_OES/EXT_draw_elements_base_vertex`);
    /// detected once at device creation (T-G11).
    pub(super) fn supports_base_vertex(&self) -> bool {
        match self {
            Self::Egl(state) => state.supports_base_vertex,
            #[cfg(windows)]
            Self::Wgl(state) => state.supports_base_vertex,
        }
    }

    /// Extension-gated float color-renderability caps
    /// (`GL_EXT_color_buffer_float` / `GL_EXT_color_buffer_half_float`);
    /// detected once at device creation (T-G12).
    pub(super) fn color_render_caps(&self) -> GlesColorRenderCaps {
        match self {
            Self::Egl(state) => state.color_render_caps,
            #[cfg(windows)]
            Self::Wgl(state) => state.color_render_caps,
        }
    }

    pub(super) fn supports_vertex_array_bgra(&self) -> bool {
        match self {
            Self::Egl(state) => state.supports_vertex_array_bgra,
            #[cfg(windows)]
            Self::Wgl(state) => state.supports_vertex_array_bgra,
        }
    }

    pub(super) fn max_samples(&self) -> i32 {
        match self {
            Self::Egl(state) => state.max_samples,
            #[cfg(windows)]
            Self::Wgl(state) => state.max_samples,
        }
    }

    pub(super) fn sample_mask_i(&self) -> Option<GlesSampleMaskIFn> {
        match self {
            Self::Egl(state) => state.sample_mask_i,
            #[cfg(windows)]
            Self::Wgl(state) => state.sample_mask_i,
        }
    }

    pub(super) fn supports_texture_view(&self) -> bool {
        match self {
            Self::Egl(state) => state.supports_texture_view,
            #[cfg(windows)]
            Self::Wgl(state) => state.supports_texture_view,
        }
    }

    pub(super) fn supports_cube_map_array(&self) -> bool {
        match self {
            Self::Egl(state) => state.supports_cube_map_array,
            #[cfg(windows)]
            Self::Wgl(state) => state.supports_cube_map_array,
        }
    }

    pub(super) fn texture_view(&self) -> Option<GlesTextureViewFn> {
        match self {
            Self::Egl(state) => state.texture_view,
            #[cfg(windows)]
            Self::Wgl(state) => state.texture_view,
        }
    }

    pub(super) fn placeholder_sampler(&self) -> Result<glow::Sampler, HalError> {
        match self {
            Self::Egl(state) => state
                .placeholder_sampler
                .as_ref()
                .copied()
                .map_err(rebuild_hal_error),
            #[cfg(windows)]
            Self::Wgl(state) => state
                .placeholder_sampler
                .as_ref()
                .copied()
                .map_err(rebuild_hal_error),
        }
    }

    pub(super) fn with_texture_to_buffer_compute_program<R>(
        &self,
        gl: &glow::Context,
        key: TextureToBufferComputeProgramKey,
        create: impl FnOnce(
            &glow::Context,
            TextureToBufferComputeProgramKey,
        ) -> Result<TextureToBufferComputeProgram, HalError>,
        use_cached: impl FnOnce(&TextureToBufferComputeProgram) -> R,
    ) -> Result<R, HalError> {
        let cache = match self {
            Self::Egl(state) => &state.texture_to_buffer_compute_programs,
            #[cfg(windows)]
            Self::Wgl(state) => &state.texture_to_buffer_compute_programs,
        };
        let mut cache = cache.lock();
        let program = match cache.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(create(gl, key)?),
        };
        Ok(use_cached(program))
    }

    fn allocation_count(&self) -> u64 {
        match self {
            Self::Egl(state) => state.allocations.load(Ordering::Relaxed),
            #[cfg(windows)]
            Self::Wgl(state) => state.allocations.load(Ordering::Relaxed),
        }
    }

    fn allocation_increment(&self) {
        match self {
            Self::Egl(state) => {
                state.allocations.fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(windows)]
            Self::Wgl(state) => {
                state.allocations.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl EglDeviceState {
    fn egl_instance(&self) -> Result<&EglInstanceState, HalError> {
        let GlesInstanceInner::Egl(state) = self.instance.as_ref() else {
            return Err(HalError::QueueSubmissionFailed {
                backend: BACKEND,
                message: "EGL device used with non-EGL instance".to_string(),
            });
        };
        Ok(state)
    }

    fn with_current_context<R>(&self, f: impl FnOnce(&glow::Context) -> R) -> Result<R, HalError> {
        let _guard = self.current_lock.lock();
        let instance = self.egl_instance()?;
        instance
            .egl
            .make_current(
                instance.display,
                Some(self.surface),
                Some(self.surface),
                Some(self.context),
            )
            .map_err(|_| HalError::QueueSubmissionFailed {
                backend: BACKEND,
                message: "eglMakeCurrent failed".to_string(),
            })?;
        Ok(f(&self.gl))
    }
}

/// Stores GLES device data used by validation and backend submission.
pub struct GlesDevice {
    inner: Arc<GlesDeviceInner>,
    queue: GlesQueue,
}

// SAFETY: `GlesDevice` delegates all GL/EGL context access to
// `GlesDeviceInner::with_current_context`, which serializes access.
unsafe impl Send for GlesDevice {}
// SAFETY: See the `Send` impl; shared operations are synchronized by the inner
// make-current lock.
unsafe impl Sync for GlesDevice {}

impl std::fmt::Debug for GlesDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesDevice")
            .field("allocations", &self.allocation_count())
            .finish()
    }
}

impl GlesDevice {
    pub(super) fn from_egl(
        instance: Arc<GlesInstanceInner>,
        context: EglContext,
        surface: EglSurface,
        gl: glow::Context,
        caps: GlesDeviceCaps,
    ) -> Self {
        let placeholder_sampler = unsafe { create_nearest_placeholder_sampler(&gl) };
        let inner = Arc::new(GlesDeviceInner::Egl(EglDeviceState {
            instance,
            context,
            surface,
            gl,
            current_lock: Mutex::new(()),
            allocations: AtomicU64::new(0),
            supports_base_vertex: caps.supports_base_vertex,
            color_render_caps: caps.color_render_caps,
            supports_vertex_array_bgra: caps.supports_vertex_array_bgra,
            max_samples: caps.max_samples,
            sample_mask_i: caps.sample_mask_i,
            supports_texture_view: caps.supports_texture_view,
            supports_cube_map_array: caps.supports_cube_map_array,
            texture_view: caps.texture_view,
            placeholder_sampler,
            texture_to_buffer_compute_programs: Mutex::new(HashMap::new()),
        }));
        let queue = GlesQueue::new(Arc::clone(&inner));
        Self { inner, queue }
    }

    #[cfg(windows)]
    pub(super) fn from_wgl(state: super::wgl::WglDeviceState) -> Self {
        let inner = Arc::new(GlesDeviceInner::Wgl(state));
        let queue = GlesQueue::new(Arc::clone(&inner));
        Self { inner, queue }
    }

    /// Returns the allocation count.
    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.inner.allocation_count()
    }

    /// Returns the queue.
    #[must_use]
    pub fn queue(&self) -> &GlesQueue {
        &self.queue
    }

    pub(super) fn inner_clone(&self) -> Arc<GlesDeviceInner> {
        Arc::clone(&self.inner)
    }

    /// Allocates a buffer of the given size on this device.
    pub fn create_buffer(&self, size: u64, usage: HalBufferUsage) -> Result<GlesBuffer, HalError> {
        self.inner.allocation_increment();
        Ok(GlesBuffer::new(Arc::clone(&self.inner), size, usage))
    }

    /// Creates a texture matching the given descriptor.
    pub fn create_texture(
        &self,
        descriptor: &HalTextureDescriptor,
    ) -> Result<GlesTexture, HalError> {
        self.inner.allocation_increment();
        let texture = GlesTexture::new(Arc::clone(&self.inner), descriptor);
        texture.raw_or_err()?;
        Ok(texture)
    }

    /// Creates a sampler matching the given descriptor.
    #[must_use]
    pub fn create_sampler(&self, descriptor: &HalSamplerDescriptor) -> GlesSampler {
        self.inner.allocation_increment();
        GlesSampler::new(Arc::clone(&self.inner), descriptor)
    }

    /// Creates a compute pipeline from the given shader, entry point, and bindings.
    pub fn create_compute_pipeline(
        &self,
        shader: HalShaderSource,
        _entry_point: &str,
        workgroup_size: (u32, u32, u32),
        bindings: &[HalDescriptorBinding],
    ) -> Result<GlesComputePipeline, HalError> {
        let HalShaderSource::Glsl {
            source,
            stage: HalShaderStage::Compute,
            combined_samplers,
            texture_metadata_slots,
            binding_remaps,
            texture_metadata_ubo_binding,
        } = shader
        else {
            return Err(HalError::ShaderCompilationFailed {
                backend: BACKEND,
                message: "GLES compute pipeline requires compute GLSL source".to_owned(),
            });
        };
        GlesComputePipeline::new(
            Arc::clone(&self.inner),
            source,
            workgroup_size,
            bindings,
            GlesPipelineResourceBindings {
                combined_samplers,
                texture_metadata_slots,
                binding_remaps,
                texture_metadata_ubo_binding,
            },
        )
    }

    /// Creates a render pipeline from the given shaders, vertex layout, and color targets.
    pub fn create_render_pipeline(
        &self,
        shader: HalShaderSource,
        _vertex_entry_point: &str,
        _fragment_entry_point: Option<&str>,
        descriptor: &HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
    ) -> Result<GlesRenderPipeline, HalError> {
        let HalShaderSource::GlslStages {
            vertex,
            fragment,
            combined_samplers,
            texture_metadata_slots,
            binding_remaps,
            texture_metadata_ubo_binding,
        } = shader
        else {
            return Err(HalError::ShaderCompilationFailed {
                backend: BACKEND,
                message: "GLES render pipeline requires GlslStages shader source".to_owned(),
            });
        };
        GlesRenderPipeline::new(
            Arc::clone(&self.inner),
            vertex,
            fragment,
            descriptor.clone(),
            bindings,
            GlesPipelineResourceBindings {
                combined_samplers,
                texture_metadata_slots,
                binding_remaps,
                texture_metadata_ubo_binding,
            },
        )
    }
}
