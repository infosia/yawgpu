//! GLES backend (Tier 2 / experimental).
//!
//! This module is a compile-time scaffold (P15.0). All public entry points
//! return [`HalError::BackendUnavailable`]; no real EGL / `glow` /
//! `libloading` calls are issued. Real bring-up lands P15.1 onward
//! (`specs/blocks/67-gles-backend.md`).

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use glow as _;
use khronos_egl as _;
use libloading as _;

use crate::{
    HalBufferUsage, HalCopy, HalDescriptorBinding, HalError, HalRenderPipelineDescriptor,
    HalSamplerDescriptor, HalShaderSource, HalSurfaceConfiguration, HalTextureDescriptor,
};

const BACKEND: &str = "gles";

fn unavailable<T>() -> Result<T, HalError> {
    Err(HalError::BackendUnavailable { backend: BACKEND })
}

/// Stores GLES instance data used by validation and backend submission.
pub struct GlesInstance;

impl std::fmt::Debug for GlesInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesInstance").finish()
    }
}

impl GlesInstance {
    /// Creates a new GLES instance scaffold.
    pub fn new() -> Result<Self, HalError> {
        Ok(Self)
    }

    /// Returns adapters exposed by this instance.
    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<GlesAdapter> {
        Vec::new()
    }
}

/// Stores GLES adapter data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesAdapter;

impl std::fmt::Debug for GlesAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesAdapter").finish()
    }
}

impl GlesAdapter {
    #[cfg(test)]
    fn new() -> Self {
        Self
    }

    /// Returns the adapter name.
    #[must_use]
    pub fn name(&self) -> &str {
        "yawgpu GLES Adapter (unavailable)"
    }

    /// Creates a device (and its default queue) on this adapter.
    pub fn create_device(&self) -> Result<GlesDevice, HalError> {
        unavailable()
    }
}

/// Stores GLES device data used by validation and backend submission.
pub struct GlesDevice {
    allocations: Arc<AtomicU64>,
    queue: GlesQueue,
}

impl std::fmt::Debug for GlesDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesDevice")
            .field("allocations", &self.allocation_count())
            .finish()
    }
}

impl GlesDevice {
    #[cfg(test)]
    fn new_for_scaffold() -> Self {
        Self {
            allocations: Arc::new(AtomicU64::new(0)),
            queue: GlesQueue,
        }
    }

    /// Returns the allocation count.
    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }

    /// Returns the queue.
    #[must_use]
    pub fn queue(&self) -> &GlesQueue {
        &self.queue
    }

    /// Allocates a buffer of the given size on this device.
    #[must_use]
    pub fn create_buffer(&self, size: u64, _usage: HalBufferUsage) -> GlesBuffer {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        GlesBuffer { size }
    }

    /// Creates a texture matching the given descriptor.
    #[must_use]
    pub fn create_texture(&self, _descriptor: &HalTextureDescriptor) -> GlesTexture {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        GlesTexture
    }

    /// Creates a sampler matching the given descriptor.
    #[must_use]
    pub fn create_sampler(&self, _descriptor: &HalSamplerDescriptor) -> GlesSampler {
        self.allocations.fetch_add(1, Ordering::Relaxed);
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

/// Stores GLES queue data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesQueue;

impl std::fmt::Debug for GlesQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesQueue").finish()
    }
}

impl GlesQueue {
    /// Submits an empty command buffer to flush the queue.
    pub fn submit_empty(&self) -> Result<(), HalError> {
        unavailable()
    }

    /// Records and submits the given buffer/texture copy operations.
    pub fn submit_copies(&self, _copies: &[HalCopy]) -> Result<(), HalError> {
        unavailable()
    }
}

/// Stores GLES buffer data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesBuffer {
    size: u64,
}

impl std::fmt::Debug for GlesBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesBuffer")
            .field("size", &self.size)
            .finish()
    }
}

impl GlesBuffer {
    /// Returns the size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Records a write command.
    pub fn write(&self, _offset: u64, _data: &[u8]) -> Result<(), HalError> {
        unavailable()
    }

    /// Reads `len` bytes at `offset` back from the buffer into host memory.
    pub fn read(&self, _offset: u64, _len: u64) -> Result<Vec<u8>, HalError> {
        unavailable()
    }

    /// Returns mapped ptr.
    #[must_use]
    pub fn mapped_ptr(&self) -> Option<NonNull<u8>> {
        None
    }
}

/// Stores GLES texture data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesTexture;

impl std::fmt::Debug for GlesTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesTexture").finish()
    }
}

/// Stores GLES sampler data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesSampler;

impl std::fmt::Debug for GlesSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesSampler").finish()
    }
}

/// Stores GLES surface data used by validation and backend submission.
pub struct GlesSurface;

impl std::fmt::Debug for GlesSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesSurface").finish()
    }
}

impl GlesSurface {
    /// Creates a GLES surface scaffold from an Android native window pointer.
    ///
    /// # Safety
    ///
    /// `window` must be a valid `ANativeWindow*` from the Android NDK and
    /// must outlive the resulting surface.
    pub unsafe fn from_android_native_window(_window: *mut c_void) -> Result<Self, HalError> {
        unavailable()
    }

    /// Configures the surface's swapchain for the given format, size, and present mode.
    pub fn configure(
        &mut self,
        _device: &GlesDevice,
        _config: HalSurfaceConfiguration,
    ) -> Result<(), HalError> {
        unavailable()
    }

    /// Tears down the surface's swapchain.
    pub fn unconfigure(&mut self) {}

    /// Returns acquire next texture.
    pub fn acquire_next_texture(&mut self) -> Result<GlesTexture, HalError> {
        unavailable()
    }

    /// Presents the most recently acquired surface texture.
    pub fn present(&mut self, _queue: &GlesQueue) -> Result<(), HalError> {
        unavailable()
    }
}

/// Stores GLES compute pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesComputePipeline;

impl std::fmt::Debug for GlesComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesComputePipeline").finish()
    }
}

/// Stores GLES render pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesRenderPipeline;

impl std::fmt::Debug for GlesRenderPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesRenderPipeline").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HalAddressMode, HalFilterMode, HalMipmapFilterMode, HalPresentMode, HalTextureFormat,
        HalTextureUsage,
    };

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

    fn assert_backend_unavailable<T>(result: Result<T, HalError>) {
        assert!(matches!(
            result,
            Err(HalError::BackendUnavailable { backend: "gles" })
        ));
    }

    #[test]
    fn gles_instance_new_returns_ok() {
        assert!(GlesInstance::new().is_ok());
    }

    #[test]
    fn gles_instance_enumerate_adapters_is_empty() {
        let adapters = GlesInstance::new()
            .expect("create GLES instance")
            .enumerate_adapters();

        assert!(adapters.is_empty());
    }

    #[test]
    fn gles_adapter_new_reports_unavailable_name() {
        let adapter = GlesAdapter::new();

        assert_eq!(adapter.name(), "yawgpu GLES Adapter (unavailable)");
    }

    #[test]
    fn gles_adapter_create_device_returns_unavailable() {
        assert_backend_unavailable(GlesAdapter::new().create_device());
    }

    #[test]
    fn gles_device_allocation_count_tracks_stub_allocations() {
        let device = GlesDevice::new_for_scaffold();

        assert_eq!(device.allocation_count(), 0);
        let _buffer = device.create_buffer(4, HalBufferUsage::default());
        let _texture = device.create_texture(&texture_descriptor());
        let _sampler = device.create_sampler(&sampler_descriptor());
        assert_eq!(device.allocation_count(), 3);
    }

    #[test]
    fn gles_device_queue_returns_queue() {
        let device = GlesDevice::new_for_scaffold();

        assert_backend_unavailable(device.queue().submit_empty());
    }

    #[test]
    fn gles_device_create_buffer_records_requested_size() {
        let device = GlesDevice::new_for_scaffold();
        let buffer = device.create_buffer(512, HalBufferUsage::default());

        assert_eq!(buffer.size(), 512);
    }

    #[test]
    fn gles_device_create_texture_increments_allocations() {
        let device = GlesDevice::new_for_scaffold();

        let _texture = device.create_texture(&texture_descriptor());

        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn gles_device_create_sampler_increments_allocations() {
        let device = GlesDevice::new_for_scaffold();

        let _sampler = device.create_sampler(&sampler_descriptor());

        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn gles_device_create_compute_pipeline_returns_unavailable() {
        let device = GlesDevice::new_for_scaffold();

        assert_backend_unavailable(device.create_compute_pipeline(
            HalShaderSource::Msl(String::new()),
            "main",
            (1, 1, 1),
            &[],
        ));
    }

    #[test]
    fn gles_device_create_render_pipeline_returns_unavailable() {
        let device = GlesDevice::new_for_scaffold();
        let descriptor = HalRenderPipelineDescriptor {
            color_formats: vec![HalTextureFormat::Rgba8Unorm],
            depth_stencil: None,
            vertex_buffers: Vec::new(),
            primitive_topology: crate::HalPrimitiveTopology::TriangleList,
        };

        assert_backend_unavailable(device.create_render_pipeline(
            HalShaderSource::Msl(String::new()),
            "vs_main",
            "fs_main",
            &descriptor,
            &[],
        ));
    }

    #[test]
    fn gles_queue_submit_empty_returns_unavailable() {
        assert_backend_unavailable(GlesQueue.submit_empty());
    }

    #[test]
    fn gles_queue_submit_copies_returns_unavailable() {
        assert_backend_unavailable(GlesQueue.submit_copies(&[]));
    }

    #[test]
    fn gles_buffer_size_matches_creation_size() {
        let buffer = GlesDevice::new_for_scaffold().create_buffer(256, HalBufferUsage::default());

        assert_eq!(buffer.size(), 256);
    }

    #[test]
    fn gles_buffer_write_returns_unavailable() {
        let buffer = GlesDevice::new_for_scaffold().create_buffer(16, HalBufferUsage::default());

        assert_backend_unavailable(buffer.write(0, &[1, 2, 3, 4]));
    }

    #[test]
    fn gles_buffer_read_returns_unavailable() {
        let buffer = GlesDevice::new_for_scaffold().create_buffer(16, HalBufferUsage::default());

        assert_backend_unavailable(buffer.read(0, 4));
    }

    #[test]
    fn gles_buffer_mapped_ptr_returns_none() {
        let buffer = GlesDevice::new_for_scaffold().create_buffer(16, HalBufferUsage::default());

        assert!(buffer.mapped_ptr().is_none());
    }

    #[test]
    fn gles_surface_from_android_native_window_returns_unavailable() {
        let window = 0xdead_beefusize as *mut c_void;

        // SAFETY: The scaffold does not dereference the window pointer.
        assert_backend_unavailable(unsafe { GlesSurface::from_android_native_window(window) });
    }

    #[test]
    fn gles_surface_configure_returns_unavailable() {
        let device = GlesDevice::new_for_scaffold();
        let mut surface = GlesSurface;

        assert_backend_unavailable(surface.configure(&device, surface_configuration()));
    }

    #[test]
    fn gles_surface_unconfigure_is_noop() {
        let mut surface = GlesSurface;

        surface.unconfigure();
        surface.unconfigure();
    }

    #[test]
    fn gles_surface_acquire_next_texture_returns_unavailable() {
        let mut surface = GlesSurface;

        assert_backend_unavailable(surface.acquire_next_texture());
    }

    #[test]
    fn gles_surface_present_returns_unavailable() {
        let mut surface = GlesSurface;

        assert_backend_unavailable(surface.present(&GlesQueue));
    }
}
