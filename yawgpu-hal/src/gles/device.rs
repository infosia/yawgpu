use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use parking_lot::MutexGuard;

use super::buffer::GlesBuffer;
use super::egl::{EglContext, EglSurface};
use super::instance::{EglInstanceState, GlesInstanceInner};
use super::pipeline::{GlesComputePipeline, GlesRenderPipeline};
use super::queue::GlesQueue;
use super::sampler::GlesSampler;
use super::texture::GlesTexture;
use super::BACKEND;
use crate::{
    HalBufferUsage, HalDescriptorBinding, HalError, HalRenderPipelineDescriptor,
    HalSamplerDescriptor, HalShaderSource, HalShaderStage, HalTextureDescriptor,
};

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
        supports_base_vertex: bool,
    ) -> Self {
        let inner = Arc::new(GlesDeviceInner::Egl(EglDeviceState {
            instance,
            context,
            surface,
            gl,
            current_lock: Mutex::new(()),
            allocations: AtomicU64::new(0),
            supports_base_vertex,
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
        Ok(GlesTexture::new(Arc::clone(&self.inner), descriptor))
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
        } = shader
        else {
            return Err(HalError::ShaderCompilationFailed {
                backend: BACKEND,
                message: "GLES compute pipeline requires compute GLSL source".to_owned(),
            });
        };
        GlesComputePipeline::new(Arc::clone(&self.inner), source, workgroup_size, bindings)
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
        let HalShaderSource::GlslStages { vertex, fragment } = shader else {
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
        )
    }
}
