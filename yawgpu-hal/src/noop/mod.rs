use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{
    HalBufferUsage, HalCopy, HalError, HalLimits, HalQueryKind, HalRenderPassCommandStream,
    HalTextureDescriptor, HalTextureDimension, SubmissionIndex,
};

/// Stores noop instance data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopInstance;

impl NoopInstance {
    /// Creates a new instance.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Returns adapters exposed by this instance.
    #[must_use]
    pub fn enumerate_adapters(&self) -> Vec<NoopAdapter> {
        vec![NoopAdapter::synthetic()]
    }
}

impl Default for NoopInstance {
    fn default() -> Self {
        Self::new()
    }
}

/// Stores noop adapter data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopAdapter {
    name: &'static str,
}

impl NoopAdapter {
    /// Builds the single synthetic adapter the Noop backend exposes.
    #[must_use]
    pub fn synthetic() -> Self {
        Self {
            name: "yawgpu Noop Adapter",
        }
    }

    /// Returns the name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the backend-reported supported limits.
    ///
    /// `max_immediate_size` is `64` (Dawn's `kMaxImmediateDataBytes`,
    /// `dawn/common/Constants.h:58`) starting Block 94 slice S1: the Noop
    /// backend "executes" `SetImmediates` trivially (it records and no-ops),
    /// so it clears the "advertise only what compiles AND executes" bar
    /// first. Metal/Vulkan/GLES stay at the `HalLimits::DEFAULT` floor of
    /// `0` until their own delivery slices (S2/S3; GLES stays `0`
    /// permanently, see `specs/blocks/67-gles-backend.md`).
    #[must_use]
    pub(crate) fn limits(&self) -> HalLimits {
        HalLimits {
            max_immediate_size: 64,
            ..HalLimits::DEFAULT
        }
    }

    /// Returns true when WebGPU texture format tier 1 is supported.
    #[must_use]
    pub(super) fn supports_texture_formats_tier1(&self) -> bool {
        true
    }

    /// Returns true when WebGPU texture format tier 2 is supported.
    #[must_use]
    pub(super) fn supports_texture_formats_tier2(&self) -> bool {
        true
    }

    /// Returns true when `Rg11b10Ufloat` is renderable.
    #[must_use]
    pub(super) fn supports_rg11b10ufloat_renderable(&self) -> bool {
        true
    }

    /// Returns true when BGRA8 unorm storage textures are supported.
    #[must_use]
    pub(super) fn supports_bgra8unorm_storage(&self) -> bool {
        true
    }

    /// Returns true when 32-bit float textures are filterable.
    #[must_use]
    pub(super) fn supports_float32_filterable(&self) -> bool {
        true
    }

    /// Returns true when timestamp queries are supported.
    #[must_use]
    pub(super) fn supports_timestamp_query(&self) -> bool {
        true
    }

    /// Returns true when Depth32FloatStencil8 textures are supported.
    #[must_use]
    pub(super) fn supports_depth32float_stencil8(&self) -> bool {
        true
    }

    /// Returns true when WGSL `shader-f16` is supported.
    #[must_use]
    pub(super) fn supports_shader_float16(&self) -> bool {
        true
    }

    /// Returns true when WGSL `subgroups` is supported.
    #[must_use]
    pub(super) fn supports_subgroups(&self) -> bool {
        true
    }

    /// Returns true when depth clip control is supported.
    #[must_use]
    pub(super) fn supports_depth_clip_control(&self) -> bool {
        true
    }

    /// Returns true when float32 color target blending is supported.
    #[must_use]
    pub(super) fn supports_float32_blendable(&self) -> bool {
        true
    }

    /// Returns true when dual-source blending is supported.
    #[must_use]
    pub(super) fn supports_dual_source_blending(&self) -> bool {
        true
    }

    /// Returns true when WGSL clip distances are supported.
    #[must_use]
    pub(super) fn supports_clip_distances(&self) -> bool {
        true
    }

    /// Returns true when WGSL primitive index is supported.
    #[must_use]
    pub(super) fn supports_primitive_index(&self) -> bool {
        true
    }

    /// Returns true when indirect draws support non-zero first instance values.
    #[must_use]
    pub(super) fn supports_indirect_first_instance(&self) -> bool {
        true
    }

    /// Returns true when texture view component swizzling is supported.
    #[must_use]
    pub(super) fn supports_texture_component_swizzle(&self) -> bool {
        true
    }

    /// Returns the supported subgroup size range.
    #[must_use]
    pub(super) fn subgroup_size_range(&self) -> Option<(u32, u32)> {
        Some((4, 4))
    }

    /// Creates a device (and its default queue) on this adapter.
    pub fn create_device(&self) -> Result<NoopDevice, HalError> {
        Ok(NoopDevice::new())
    }
}

/// Stores noop device data used by validation and backend submission.
#[derive(Debug)]
pub struct NoopDevice {
    allocations: AtomicU64,
    queue: NoopQueue,
}

impl NoopDevice {
    /// Creates a new instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            allocations: AtomicU64::new(0),
            queue: NoopQueue::new(),
        }
    }

    /// Returns the allocation count.
    #[must_use]
    pub fn allocation_count(&self) -> u64 {
        self.allocations.load(Ordering::Relaxed)
    }

    /// Returns the queue.
    #[must_use]
    pub fn queue(&self) -> &NoopQueue {
        &self.queue
    }

    /// Allocates a buffer of the given size on this device.
    pub fn create_buffer(&self, size: u64, _usage: HalBufferUsage) -> Result<NoopBuffer, HalError> {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(NoopBuffer::new(size))
    }

    /// Creates a query set of the given kind and count.
    #[must_use]
    pub fn create_query_set(&self, _kind: HalQueryKind, count: u32) -> u32 {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        count
    }

    /// Creates a texture matching the given descriptor.
    pub fn create_texture(
        &self,
        descriptor: &HalTextureDescriptor,
    ) -> Result<NoopTexture, HalError> {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(NoopTexture {
            dimension: descriptor.dimension,
            width: descriptor.width,
            height: descriptor.height,
            depth_or_array_layers: descriptor.depth_or_array_layers,
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
        })
    }

    /// Creates a sampler matching the given descriptor.
    #[must_use]
    pub fn create_sampler(&self) -> NoopSampler {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        NoopSampler
    }
}

impl Default for NoopDevice {
    fn default() -> Self {
        Self::new()
    }
}

/// Stores noop queue data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopQueue {
    submitted_copies: Arc<Mutex<Vec<HalCopy>>>,
    submitted_render_pass_command_streams: Arc<Mutex<Vec<HalRenderPassCommandStream>>>,
    last_submission_index: Arc<AtomicU64>,
}

impl NoopQueue {
    /// Creates a new instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            submitted_copies: Arc::new(Mutex::new(Vec::new())),
            submitted_render_pass_command_streams: Arc::new(Mutex::new(Vec::new())),
            last_submission_index: Arc::new(AtomicU64::new(SubmissionIndex::NONE.0)),
        }
    }

    /// Submits an empty command buffer to flush the queue.
    pub fn submit_empty(&self) -> Result<SubmissionIndex, HalError> {
        crate::next_submission_index(&self.last_submission_index, "noop")
    }

    /// Records submitted copy commands for Noop unit-test inspection.
    ///
    /// `HalCopy::Buffer` copies are executed eagerly so that subsequent
    /// map-reads on the destination buffer observe the written bytes (mirrors
    /// the real-GPU semantics where the copy completes before any following
    /// `mapAsync` resolves).
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<SubmissionIndex, HalError> {
        for copy in copies {
            match copy {
                HalCopy::Buffer(buf_copy) => {
                    // Read from source, write into destination in order to
                    // make the data visible for subsequent map-reads.
                    let data = buf_copy
                        .source
                        .read(buf_copy.source_offset, buf_copy.size)?;
                    buf_copy
                        .destination
                        .write(buf_copy.destination_offset, &data)?;
                }
                HalCopy::ResolveQuerySet(resolve) => {
                    let byte_count = resolve_query_byte_count(resolve.query_count)?;
                    let zeros = vec![0; byte_count];
                    resolve
                        .destination
                        .write(resolve.destination_offset, &zeros)?;
                }
                HalCopy::ClearTexture(_) => {}
                HalCopy::RenderPassCommandStream(pass) => {
                    self.submitted_render_pass_command_streams
                        .lock()
                        .map_err(|_| HalError::QueueSubmissionFailed {
                            backend: "noop",
                            message: "submitted render pass command streams lock poisoned"
                                .to_string(),
                        })?
                        .push(pass.clone());
                }
                _ => {}
            }
        }
        self.submitted_copies
            .lock()
            .map_err(|_| HalError::QueueSubmissionFailed {
                backend: "noop",
                message: "submitted copies lock poisoned".to_string(),
            })?
            .extend(copies.iter().cloned());
        crate::next_submission_index(&self.last_submission_index, "noop")
    }

    /// Records render-pass command streams for Noop unit-test inspection.
    pub fn submit_render_pass_command_streams(
        &self,
        passes: &[HalRenderPassCommandStream],
    ) -> Result<SubmissionIndex, HalError> {
        self.submitted_render_pass_command_streams
            .lock()
            .map_err(|_| HalError::QueueSubmissionFailed {
                backend: "noop",
                message: "submitted render pass command streams lock poisoned".to_string(),
            })?
            .extend(passes.iter().cloned());
        crate::next_submission_index(&self.last_submission_index, "noop")
    }

    /// Returns the highest submission index proven complete without blocking.
    pub fn completed_submission_index(&self) -> Result<SubmissionIndex, HalError> {
        Ok(SubmissionIndex(
            self.last_submission_index.load(Ordering::Acquire),
        ))
    }

    /// Blocks until the requested submission index has completed.
    pub fn wait_for_submission(&self, index: SubmissionIndex) -> Result<(), HalError> {
        let completed = self.completed_submission_index()?;
        if index <= completed {
            Ok(())
        } else {
            Err(HalError::QueueSubmissionFailed {
                backend: "noop",
                message: "submission index has not been issued".to_string(),
            })
        }
    }

    /// Waits until all submitted queue work has completed.
    pub fn wait_idle(&self) -> Result<(), HalError> {
        Ok(())
    }

    /// Returns submitted copy commands recorded by this queue.
    #[must_use]
    pub fn submitted_copies(&self) -> Vec<HalCopy> {
        self.submitted_copies
            .lock()
            .map(|copies| copies.clone())
            .unwrap_or_default()
    }

    /// Returns render-pass command streams recorded by this queue.
    #[must_use]
    pub fn submitted_render_pass_command_streams(&self) -> Vec<HalRenderPassCommandStream> {
        self.submitted_render_pass_command_streams
            .lock()
            .map(|passes| passes.clone())
            .unwrap_or_default()
    }
}

impl Default for NoopQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Stores noop buffer data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopBuffer {
    size: u64,
    data: Arc<Mutex<Vec<u8>>>,
}

impl NoopBuffer {
    /// Creates a new noop buffer with zero-initialized storage.
    #[must_use]
    pub fn new(size: u64) -> Self {
        let len = usize::try_from(size).unwrap_or(0);
        Self {
            size,
            data: Arc::new(Mutex::new(vec![0; len])),
        }
    }

    /// Returns the size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Writes bytes into the buffer.
    pub fn write(&self, offset: u64, data: &[u8]) -> Result<(), HalError> {
        let end = validate_noop_buffer_range(self.size, offset, data.len() as u64)?;
        let offset = usize::try_from(offset).map_err(|_| HalError::BufferOperationFailed {
            backend: "noop",
            message: "buffer offset is too large",
        })?;
        let mut storage = self
            .data
            .lock()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: "noop",
                message: "buffer storage lock failed",
            })?;
        if end > storage.len() {
            return Err(HalError::BufferOperationFailed {
                backend: "noop",
                message: "buffer storage is too small for range",
            });
        }
        storage[offset..end].copy_from_slice(data);
        Ok(())
    }

    /// Reads bytes from the buffer.
    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>, HalError> {
        let end = validate_noop_buffer_range(self.size, offset, len)?;
        let offset = usize::try_from(offset).map_err(|_| HalError::BufferOperationFailed {
            backend: "noop",
            message: "buffer offset is too large",
        })?;
        let storage = self
            .data
            .lock()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: "noop",
                message: "buffer storage lock failed",
            })?;
        if end > storage.len() {
            return Err(HalError::BufferOperationFailed {
                backend: "noop",
                message: "buffer storage is too small for range",
            });
        }
        Ok(storage[offset..end].to_vec())
    }

    /// Returns mapped ptr.
    #[must_use]
    pub fn mapped_ptr(&self) -> Option<std::ptr::NonNull<u8>> {
        None
    }
}

fn resolve_query_byte_count(query_count: u32) -> Result<usize, HalError> {
    usize::try_from(u64::from(query_count) * 8).map_err(|_| HalError::BufferOperationFailed {
        backend: "noop",
        message: "query resolve byte count is too large",
    })
}

fn validate_noop_buffer_range(size: u64, offset: u64, len: u64) -> Result<usize, HalError> {
    let end = offset
        .checked_add(len)
        .ok_or(HalError::BufferOperationFailed {
            backend: "noop",
            message: "buffer range overflows",
        })?;
    if end > size {
        return Err(HalError::BufferOperationFailed {
            backend: "noop",
            message: "buffer range exceeds buffer size",
        });
    }
    usize::try_from(end).map_err(|_| HalError::BufferOperationFailed {
        backend: "noop",
        message: "buffer range is too large",
    })
}

/// Stores noop texture data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopTexture {
    dimension: HalTextureDimension,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    mip_level_count: u32,
    sample_count: u32,
}

impl NoopTexture {
    /// Returns the texture dimension.
    #[must_use]
    pub fn dimension(&self) -> HalTextureDimension {
        self.dimension
    }

    /// Returns the texture width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the texture height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns the texture depth or array layer count.
    #[must_use]
    pub fn depth_or_array_layers(&self) -> u32 {
        self.depth_or_array_layers
    }

    /// Returns the mip level count.
    #[must_use]
    pub fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }

    /// Returns the texture sample count.
    #[must_use]
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }
}

/// Stores noop sampler data used by validation and backend submission.
#[derive(Debug, Clone)]
pub struct NoopSampler;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        HalBoundBuffer, HalBoundIndexBuffer, HalBoundIndirectBuffer, HalBuffer, HalIndexFormat,
        HalQuerySet, HalRenderBundle, HalRenderColorTarget, HalRenderDepthStencilAttachment,
        HalRenderLoadOp, HalRenderPassCommand, HalRenderPipeline, HalScissorRect, HalTexture,
        HalTextureFormat, HalTextureUsage, HalViewport,
    };

    fn texture_descriptor() -> HalTextureDescriptor {
        HalTextureDescriptor {
            dimension: HalTextureDimension::D2,
            format: HalTextureFormat::Rgba8Unorm,
            width: 4,
            height: 4,
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

    #[test]
    fn noop_instance_new_constructs() {
        let instance = NoopInstance::new();

        assert_eq!(instance.enumerate_adapters().len(), 1);
    }

    #[test]
    fn noop_instance_enumerate_adapters_returns_synthetic_adapter() {
        let instance = NoopInstance::new();
        let adapters = instance.enumerate_adapters();

        assert_eq!(adapters.len(), 1);
        assert_eq!(adapters[0].name(), "yawgpu Noop Adapter");
    }

    #[test]
    fn noop_adapter_synthetic_exposes_documented_name() {
        let adapter = NoopAdapter::synthetic();

        assert_eq!(adapter.name(), "yawgpu Noop Adapter");
    }

    #[test]
    fn noop_adapter_name_returns_fixed_string() {
        let adapter = NoopAdapter::synthetic();

        assert_eq!(adapter.name(), "yawgpu Noop Adapter");
    }

    #[test]
    fn noop_adapter_supports_shader_float16_returns_true() {
        let adapter = NoopAdapter::synthetic();

        assert!(adapter.supports_shader_float16());
    }

    #[test]
    fn noop_adapter_supports_subgroups_returns_nominal_range() {
        let adapter = NoopAdapter::synthetic();

        assert!(adapter.supports_subgroups());
        assert_eq!(adapter.subgroup_size_range(), Some((4, 4)));
    }

    #[test]
    fn noop_adapter_supports_depth_clip_control_returns_true() {
        let adapter = NoopAdapter::synthetic();

        assert!(adapter.supports_depth_clip_control());
    }

    #[test]
    fn noop_adapter_supports_float32_blendable_returns_true() {
        let adapter = NoopAdapter::synthetic();

        assert!(adapter.supports_float32_blendable());
    }

    #[test]
    fn noop_adapter_supports_dual_source_blending_returns_true() {
        let adapter = NoopAdapter::synthetic();

        assert!(adapter.supports_dual_source_blending());
    }

    #[test]
    fn noop_adapter_supports_clip_distances_returns_true() {
        let adapter = NoopAdapter::synthetic();

        assert!(adapter.supports_clip_distances());
    }

    #[test]
    fn noop_adapter_supports_primitive_index_returns_true() {
        let adapter = NoopAdapter::synthetic();

        assert!(adapter.supports_primitive_index());
    }

    #[test]
    fn noop_adapter_supports_indirect_first_instance_returns_true() {
        let adapter = NoopAdapter::synthetic();

        assert!(adapter.supports_indirect_first_instance());
    }

    #[test]
    fn noop_adapter_supports_texture_component_swizzle_returns_true() {
        let adapter = NoopAdapter::synthetic();

        assert!(adapter.supports_texture_component_swizzle());
    }

    #[test]
    fn noop_adapter_create_device_returns_zero_allocation_device() {
        let adapter = NoopAdapter::synthetic();
        let device = adapter
            .create_device()
            .expect("Noop device creation succeeds");

        assert_eq!(device.allocation_count(), 0);
    }

    #[test]
    fn noop_device_new_starts_with_zero_allocations() {
        let device = NoopDevice::new();

        assert_eq!(device.allocation_count(), 0);
    }

    #[test]
    fn noop_device_allocation_count_tracks_created_resources() {
        let device = NoopDevice::new();

        assert_eq!(device.allocation_count(), 0);
        let _buffer = device
            .create_buffer(4, HalBufferUsage::default())
            .expect("Noop buffer allocation should succeed");
        assert_eq!(device.allocation_count(), 1);
        let _texture = device
            .create_texture(&texture_descriptor())
            .expect("Noop texture allocation should succeed");
        assert_eq!(device.allocation_count(), 2);
        let _sampler = device.create_sampler();
        assert_eq!(device.allocation_count(), 3);
    }

    #[test]
    fn noop_device_queue_returns_same_reference() {
        let device = NoopDevice::new();

        assert!(std::ptr::eq(device.queue(), device.queue()));
    }

    #[test]
    fn noop_device_create_buffer_records_size_and_increments_allocation_count() {
        let device = NoopDevice::new();
        let buffer = device
            .create_buffer(64, HalBufferUsage::default())
            .expect("Noop buffer allocation should succeed");

        assert_eq!(buffer.size(), 64);
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn noop_device_create_texture_increments_allocation_count() {
        let device = NoopDevice::new();
        let _texture = device
            .create_texture(&texture_descriptor())
            .expect("Noop texture allocation should succeed");

        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn noop_device_create_texture_records_array_3d_and_mip_shape() {
        let device = NoopDevice::new();
        let mut descriptor = texture_descriptor();
        descriptor.dimension = HalTextureDimension::D3;
        descriptor.width = 8;
        descriptor.height = 4;
        descriptor.depth_or_array_layers = 3;
        descriptor.mip_level_count = 4;

        let texture = device
            .create_texture(&descriptor)
            .expect("Noop texture allocation should succeed");

        assert_eq!(texture.dimension(), HalTextureDimension::D3);
        assert_eq!(texture.width(), 8);
        assert_eq!(texture.height(), 4);
        assert_eq!(texture.depth_or_array_layers(), 3);
        assert_eq!(texture.mip_level_count(), 4);
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn noop_device_create_texture_accepts_multisample_descriptor() {
        let device = NoopDevice::new();
        let mut descriptor = texture_descriptor();
        descriptor.sample_count = 4;

        let texture = device
            .create_texture(&descriptor)
            .expect("Noop texture allocation should succeed");

        assert_eq!(texture.sample_count(), 4);
        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    fn noop_device_create_sampler_increments_allocation_count() {
        let device = NoopDevice::new();
        let _sampler = device.create_sampler();

        assert_eq!(device.allocation_count(), 1);
    }

    #[test]
    #[allow(clippy::default_constructed_unit_structs)]
    fn noop_queue_new_matches_default_smoke() {
        let _queue = NoopQueue::new();
        let _default_queue = NoopQueue::default();
    }

    #[test]
    fn noop_queue_submission_indices_are_monotonic() -> Result<(), HalError> {
        let queue = NoopQueue::new();

        assert_eq!(queue.completed_submission_index()?, SubmissionIndex::NONE);
        let first = queue.submit_empty()?;
        let second = queue.submit_copies(&[])?;
        let third = queue.submit_empty()?;

        assert!(SubmissionIndex::NONE < first);
        assert!(first < second);
        assert!(second < third);
        Ok(())
    }

    #[test]
    fn noop_queue_submit_render_pass_command_streams_records_sparse_state_and_command_order(
    ) -> Result<(), HalError> {
        let device = NoopDevice::new();
        let queue = NoopQueue::new();
        let buffer = HalBuffer::Noop(device.create_buffer(256, HalBufferUsage::default())?);
        let color_texture = HalTexture::Noop(device.create_texture(&texture_descriptor())?);
        let mut depth_descriptor = texture_descriptor();
        depth_descriptor.format = HalTextureFormat::Depth32Float;
        let depth_texture = HalTexture::Noop(device.create_texture(&depth_descriptor)?);
        let color_target = HalRenderColorTarget {
            texture: color_texture,
            view_format: HalTextureFormat::Rgba8Unorm,
            resolve_target: None,
            resolve_view_format: None,
            mip_level: 0,
            array_layer: 0,
            depth_slice: 0,
            resolve_mip_level: 0,
            resolve_array_layer: 0,
            load_op: HalRenderLoadOp::Clear,
            store: true,
            clear_color: [0.25, 0.5, 0.75, 1.0],
        };
        let vertex_buffer = HalBoundBuffer {
            group: 0,
            binding: 3,
            metal_index: 7,
            vertex_metal_index: None,
            fragment_metal_index: None,
            buffer: buffer.clone(),
            offset: 16,
            size: 128,
        };
        let index_buffer = HalBoundIndexBuffer {
            buffer: buffer.clone(),
            format: HalIndexFormat::Uint32,
            offset: 32,
            size: 96,
        };
        let indirect_buffer = HalBoundIndirectBuffer { buffer, offset: 64 };
        let bundle = HalRenderBundle {
            commands: vec![
                HalRenderPassCommand::SetPipeline(HalRenderPipeline::Noop),
                HalRenderPassCommand::Draw {
                    vertex_count: 1,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
            ],
        };
        let pass = HalRenderPassCommandStream {
            color_targets: vec![Some(color_target.clone()), None, Some(color_target)],
            framebuffer_fetch_color_slots: vec![2],
            depth_stencil_attachment: Some(HalRenderDepthStencilAttachment {
                texture: depth_texture,
                format: HalTextureFormat::Depth32Float,
                mip_level: 0,
                array_layer: 0,
                depth_load_op: HalRenderLoadOp::Load,
                depth_store: true,
                depth_clear_value: 1.0,
                depth_read_only: false,
                stencil_load_op: HalRenderLoadOp::Load,
                stencil_store: false,
                stencil_clear_value: 0,
                stencil_read_only: true,
            }),
            occlusion_query_set: Some(HalQuerySet::Noop { count: 4 }),
            commands: vec![
                HalRenderPassCommand::SetPipeline(HalRenderPipeline::Noop),
                HalRenderPassCommand::SetBindGroup {
                    index: 1,
                    buffers: Vec::new(),
                    textures: Vec::new(),
                    samplers: Vec::new(),
                    external_textures: Vec::new(),
                },
                HalRenderPassCommand::SetVertexBuffer {
                    slot: 3,
                    buffer: Some(vertex_buffer),
                },
                HalRenderPassCommand::SetIndexBuffer(index_buffer),
                HalRenderPassCommand::SetViewport(HalViewport {
                    x: 1.0,
                    y: 2.0,
                    width: 3.0,
                    height: 4.0,
                    min_depth: 0.25,
                    max_depth: 0.75,
                }),
                HalRenderPassCommand::SetScissorRect(HalScissorRect {
                    x: 1,
                    y: 2,
                    width: 3,
                    height: 4,
                }),
                HalRenderPassCommand::SetBlendConstant([0.1, 0.2, 0.3, 0.4]),
                HalRenderPassCommand::SetStencilReference(0x7f),
                HalRenderPassCommand::SetImmediates {
                    offset: 4,
                    data: vec![1, 2, 3, 4],
                },
                HalRenderPassCommand::BeginOcclusionQuery { index: 2 },
                HalRenderPassCommand::Draw {
                    vertex_count: 3,
                    instance_count: 2,
                    first_vertex: 1,
                    first_instance: 4,
                },
                HalRenderPassCommand::DrawIndexed {
                    index_count: 6,
                    instance_count: 2,
                    first_index: 1,
                    base_vertex: -2,
                    first_instance: 3,
                },
                HalRenderPassCommand::DrawIndirect {
                    indirect_buffer: indirect_buffer.clone(),
                },
                HalRenderPassCommand::DrawIndexedIndirect { indirect_buffer },
                HalRenderPassCommand::ExecuteRenderBundle(bundle),
                HalRenderPassCommand::EndOcclusionQuery,
            ],
        };

        let empty_submission = queue.submit_render_pass_command_streams(&[])?;
        let pass_submission = queue.submit_render_pass_command_streams(&[pass])?;
        assert!(empty_submission < pass_submission);

        let recorded = queue.submitted_render_pass_command_streams();
        assert_eq!(recorded.len(), 1);
        let recorded = &recorded[0];
        assert_eq!(recorded.color_targets.len(), 3);
        assert!(recorded.color_targets[0].is_some());
        assert!(recorded.color_targets[1].is_none());
        assert!(recorded.color_targets[2].is_some());
        assert_eq!(recorded.framebuffer_fetch_color_slots, vec![2]);
        assert!(recorded.depth_stencil_attachment.is_some());
        assert!(recorded.occlusion_query_set.is_some());
        assert_eq!(recorded.commands.len(), 16);
        assert!(matches!(
            recorded.commands[0],
            HalRenderPassCommand::SetPipeline(HalRenderPipeline::Noop)
        ));
        assert!(matches!(
            recorded.commands[1],
            HalRenderPassCommand::SetBindGroup { index: 1, .. }
        ));
        assert!(matches!(
            recorded.commands[2],
            HalRenderPassCommand::SetVertexBuffer {
                slot: 3,
                buffer: Some(_)
            }
        ));
        assert!(matches!(
            recorded.commands[3],
            HalRenderPassCommand::SetIndexBuffer(_)
        ));
        assert!(matches!(
            recorded.commands[4],
            HalRenderPassCommand::SetViewport(_)
        ));
        assert!(matches!(
            recorded.commands[5],
            HalRenderPassCommand::SetScissorRect(_)
        ));
        assert!(matches!(
            recorded.commands[6],
            HalRenderPassCommand::SetBlendConstant(_)
        ));
        assert!(matches!(
            recorded.commands[7],
            HalRenderPassCommand::SetStencilReference(0x7f)
        ));
        assert!(matches!(
            &recorded.commands[8],
            HalRenderPassCommand::SetImmediates { offset: 4, data }
                if data == &[1, 2, 3, 4]
        ));
        assert!(matches!(
            recorded.commands[9],
            HalRenderPassCommand::BeginOcclusionQuery { index: 2 }
        ));
        assert!(matches!(
            recorded.commands[10],
            HalRenderPassCommand::Draw {
                vertex_count: 3,
                ..
            }
        ));
        assert!(matches!(
            recorded.commands[11],
            HalRenderPassCommand::DrawIndexed { index_count: 6, .. }
        ));
        assert!(matches!(
            recorded.commands[12],
            HalRenderPassCommand::DrawIndirect { .. }
        ));
        assert!(matches!(
            recorded.commands[13],
            HalRenderPassCommand::DrawIndexedIndirect { .. }
        ));
        assert!(matches!(
            &recorded.commands[14],
            HalRenderPassCommand::ExecuteRenderBundle(bundle) if bundle.commands.len() == 2
        ));
        assert!(matches!(
            recorded.commands[15],
            HalRenderPassCommand::EndOcclusionQuery
        ));
        Ok(())
    }

    #[test]
    fn noop_queue_submit_render_pass_command_streams_reports_poisoned_recorder() {
        let queue = NoopQueue::new();
        let recorder = Arc::clone(&queue.submitted_render_pass_command_streams);
        let poisoned = std::thread::spawn(move || {
            let _guard = recorder.lock().expect("recorder lock starts healthy");
            panic!("poison render pass command stream recorder");
        });
        assert!(poisoned.join().is_err());

        let error = queue
            .submit_render_pass_command_streams(&[])
            .expect_err("poisoned recorder must reject submission");
        assert!(matches!(
            error,
            HalError::QueueSubmissionFailed {
                backend: "noop",
                message
            } if message == "submitted render pass command streams lock poisoned"
        ));
        assert!(queue.submitted_render_pass_command_streams().is_empty());
    }

    #[test]
    fn noop_queue_completed_submission_index_tracks_completed_submission() -> Result<(), HalError> {
        let queue = NoopQueue::new();
        let submitted = queue.submit_empty()?;

        assert_eq!(queue.completed_submission_index()?, submitted);
        Ok(())
    }

    #[test]
    fn noop_queue_wait_for_already_completed_submission_returns_promptly() -> Result<(), HalError> {
        let queue = NoopQueue::new();
        let submitted = queue.submit_empty()?;
        let started = std::time::Instant::now();

        queue.wait_for_submission(submitted)?;

        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        Ok(())
    }

    #[test]
    fn noop_queue_wait_for_unissued_submission_returns_error() -> Result<(), HalError> {
        let queue = NoopQueue::new();
        let unissued = queue.submit_empty()?;
        let unissued = SubmissionIndex(unissued.0 + 1);

        assert!(queue.wait_for_submission(unissued).is_err());
        Ok(())
    }

    #[test]
    fn noop_buffer_size_returns_created_size() {
        let device = NoopDevice::new();

        assert_eq!(
            device
                .create_buffer(0, HalBufferUsage::default())
                .expect("Noop buffer allocation should succeed")
                .size(),
            0
        );
        assert_eq!(
            device
                .create_buffer(4096, HalBufferUsage::default())
                .expect("Noop buffer allocation should succeed")
                .size(),
            4096
        );
    }

    #[test]
    fn noop_buffer_mapped_ptr_returns_none() {
        let device = NoopDevice::new();
        let buffer = device
            .create_buffer(128, HalBufferUsage::default())
            .expect("Noop buffer allocation should succeed");

        assert!(buffer.mapped_ptr().is_none());
    }
}
