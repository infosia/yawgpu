use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use super::buffer::GlesBuffer;
use super::egl::{EglContext, EglSurface};
use super::instance::GlesInstanceInner;
use super::pipeline::{GlesComputePipeline, GlesRenderPipeline};
use super::queue::GlesQueue;
use super::sampler::GlesSampler;
use super::texture::GlesTexture;
use super::{unavailable, BACKEND};
use crate::{
    HalBufferUsage, HalDescriptorBinding, HalError, HalRenderPipelineDescriptor,
    HalSamplerDescriptor, HalShaderSource, HalTextureDescriptor,
};

pub(super) struct GlesDeviceInner {
    pub(super) instance: Arc<GlesInstanceInner>,
    pub(super) context: EglContext,
    pub(super) surface: EglSurface,
    pub(super) gl: glow::Context,
    current_lock: Mutex<()>,
    allocations: AtomicU64,
}

// SAFETY: All access to the EGL context and `glow::Context` goes through
// `with_current_context`, which holds `current_lock` while making the context
// current and executing GL commands.
unsafe impl Send for GlesDeviceInner {}
// SAFETY: See the `Send` impl; shared references are synchronized by
// `current_lock`, and resource teardown only runs after the final `Arc` drops.
unsafe impl Sync for GlesDeviceInner {}

impl Drop for GlesDeviceInner {
    fn drop(&mut self) {
        let _ = self
            .instance
            .egl
            .make_current(self.instance.display, None, None, None);
        let _ = self
            .instance
            .egl
            .destroy_surface(self.instance.display, self.surface);
        let _ = self
            .instance
            .egl
            .destroy_context(self.instance.display, self.context);
    }
}

impl GlesDeviceInner {
    pub(super) fn with_current_context<R>(
        &self,
        f: impl FnOnce(&glow::Context) -> R,
    ) -> Result<R, HalError> {
        let _guard = self.current_lock.lock();
        self.instance
            .egl
            .make_current(
                self.instance.display,
                Some(self.surface),
                Some(self.surface),
                Some(self.context),
            )
            .map_err(|_| HalError::QueueSubmissionFailed { backend: BACKEND })?;
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
    pub(super) fn from_parts(
        instance: Arc<GlesInstanceInner>,
        context: EglContext,
        surface: EglSurface,
        gl: glow::Context,
    ) -> Self {
        let inner = Arc::new(GlesDeviceInner {
            instance,
            context,
            surface,
            gl,
            current_lock: Mutex::new(()),
            allocations: AtomicU64::new(0),
        });
        let queue = GlesQueue::new(Arc::clone(&inner));
        Self { inner, queue }
    }

    /// Returns the allocation count.
    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.inner.allocations.load(Ordering::Relaxed)
    }

    /// Returns the queue.
    #[must_use]
    pub fn queue(&self) -> &GlesQueue {
        &self.queue
    }

    /// Allocates a buffer of the given size on this device.
    #[must_use]
    pub fn create_buffer(&self, size: u64, _usage: HalBufferUsage) -> GlesBuffer {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        GlesBuffer::new(size)
    }

    /// Creates a texture matching the given descriptor.
    #[must_use]
    pub fn create_texture(&self, _descriptor: &HalTextureDescriptor) -> GlesTexture {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        GlesTexture
    }

    /// Creates a sampler matching the given descriptor.
    #[must_use]
    pub fn create_sampler(&self, _descriptor: &HalSamplerDescriptor) -> GlesSampler {
        self.inner.allocations.fetch_add(1, Ordering::Relaxed);
        GlesSampler
    }

    /// Creates a compute pipeline from the given shader, entry point, and bindings.
    pub fn create_compute_pipeline(
        &self,
        _shader: HalShaderSource,
        _entry_point: &str,
        _workgroup_size: (u32, u32, u32),
        _bindings: &[HalDescriptorBinding],
    ) -> Result<GlesComputePipeline, HalError> {
        unavailable()
    }

    /// Creates a render pipeline from the given shaders, vertex layout, and color targets.
    pub fn create_render_pipeline(
        &self,
        _shader: HalShaderSource,
        _vertex_entry_point: &str,
        _fragment_entry_point: &str,
        _descriptor: &HalRenderPipelineDescriptor,
        _bindings: &[HalDescriptorBinding],
    ) -> Result<GlesRenderPipeline, HalError> {
        unavailable()
    }
}
