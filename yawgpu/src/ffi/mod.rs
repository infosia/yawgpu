/// Adapter module.
pub mod adapter;
/// Bindings module.
pub mod bindings;
/// Buffer module.
pub mod buffer;
/// Bundle module.
pub mod bundle;
/// Command buffer module.
pub mod command_buffer;
/// Compute pass module.
pub mod compute_pass;
/// Device module.
pub mod device;
/// Encoder module.
pub mod encoder;
/// Instance module.
pub mod instance;
/// Pipelines module.
pub mod pipelines;
/// Query module.
pub mod query;
/// Queue module.
pub mod queue;
/// Render pass module.
pub mod render_pass;
/// Sampler module.
pub mod sampler;
/// Shader module.
pub mod shader;
/// Surface module.
pub mod surface;
/// Texture module.
pub mod texture;

#[cfg(test)]
use adapter::*;
#[cfg(test)]
use bindings::*;
#[cfg(test)]
use buffer::*;
#[cfg(test)]
use bundle::*;
#[cfg(test)]
use command_buffer::*;
#[cfg(test)]
use compute_pass::*;
#[cfg(test)]
use device::*;
#[cfg(test)]
use encoder::*;
#[cfg(test)]
use instance::*;
#[cfg(test)]
use pipelines::*;
#[cfg(test)]
use query::*;
#[cfg(test)]
use queue::*;
#[cfg(test)]
use render_pass::*;
#[cfg(test)]
use sampler::*;
#[cfg(test)]
use shader::*;
#[cfg(test)]
use surface::*;
#[cfg(test)]
use texture::*;

use crate::{
    native, YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_INSTANCE_BACKEND_VULKAN, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use std::collections::{BTreeMap, HashMap};
use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu_core as core;

use crate::conv::{
    add_ref_handle, arc_to_handle, borrow_handle, clone_handle, free_supported_features,
    label_from_string_view, map_bind_group_entries, map_bind_group_layout_descriptor,
    map_buffer_descriptor, map_buffer_map_state, map_buffer_usage_to_native, map_color,
    map_compilation_info_request_status_success, map_compilation_message_type_error,
    map_compute_pipeline_descriptor, map_device_lost_callback_info, map_device_lost_reason,
    map_error_filter, map_error_type, map_extent_3d, map_feature, map_feature_level,
    map_features_to_native, map_limits, map_limits_to_native, map_map_async_status, map_map_mode,
    map_origin_3d, map_pipeline_layout_descriptor, map_pop_error_scope_status_error,
    map_pop_error_scope_status_success, map_query_set_descriptor, map_query_type_to_native,
    map_queue_work_done_status, map_render_bundle_encoder_descriptor, map_render_pass_descriptor,
    map_render_pipeline_descriptor, map_sampler_descriptor, map_shader_module_descriptor,
    map_texel_copy_buffer_layout, map_texel_copy_texture_info_parts, map_texture_aspect,
    map_texture_descriptor, map_texture_dimension_to_native, map_texture_format_to_native,
    map_texture_usage, map_texture_usage_to_native, map_texture_view_descriptor, release_handle,
    string_view, string_view_to_str, DeviceLostCallbackInfo,
};
use yawgpu_hal::{
    HalInstance, HalPresentMode, HalSurface, HalSurfaceConfiguration, HalTextureFormat,
    HalTextureUsage,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstanceBackendSelection {
    Noop,
    Metal,
    Vulkan,
}

/// Owns the core object and retained handles for the WGPU Adapter handle.
pub struct WGPUAdapterImpl {
    pub(crate) core: Arc<core::Adapter>,
    pub(crate) instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU Buffer handle.
pub struct WGPUBufferImpl {
    pub(crate) core: Arc<core::Buffer>,
    pub(crate) device: Arc<core::Device>,
    pub(crate) instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU BindGroupLayout handle.
pub struct WGPUBindGroupLayoutImpl {
    pub(crate) _core: Arc<core::BindGroupLayout>,
    pub(crate) _device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU BindGroup handle.
pub struct WGPUBindGroupImpl {
    pub(crate) _core: Arc<core::BindGroup>,
    pub(crate) _layout: Arc<core::BindGroupLayout>,
    pub(crate) _device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU Device handle.
pub struct WGPUDeviceImpl {
    pub(crate) core: Arc<core::Device>,
    pub(crate) instance: Arc<WGPUInstanceImpl>,
    pub(crate) device_lost_callback: DeviceLostCallbackInfo,
    pub(crate) device_lost_futures: Mutex<Vec<u64>>,
    pub(crate) default_queue: Mutex<Option<Arc<WGPUQueueImpl>>>,
    pub(crate) shader_module_cache: Mutex<HashMap<ShaderModuleCacheKey, Arc<WGPUShaderModuleImpl>>>,
    pub(crate) pipeline_layout_cache:
        Mutex<HashMap<PipelineLayoutCacheKey, Arc<WGPUPipelineLayoutImpl>>>,
    pub(crate) compute_pipeline_cache:
        Mutex<HashMap<ComputePipelineCacheKey, Arc<WGPUComputePipelineImpl>>>,
    pub(crate) render_pipeline_cache:
        Mutex<HashMap<RenderPipelineCacheKey, Arc<WGPURenderPipelineImpl>>>,
}

/// Owns the core object and retained handles for the WGPU Instance handle.
pub struct WGPUInstanceImpl {
    pub(crate) core: Arc<core::Instance>,
    pub(crate) timed_wait_any_enabled: bool,
    pub(crate) pending_callbacks: Mutex<BTreeMap<u64, PendingCallback>>,
}

/// Owns the core object and retained handles for the WGPU Queue handle.
pub struct WGPUQueueImpl {
    pub(crate) core: core::Queue,
    pub(crate) device: Arc<core::Device>,
    pub(crate) instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU Texture handle.
pub struct WGPUTextureImpl {
    pub(crate) core: Arc<core::Texture>,
    pub(crate) device: Arc<core::Device>,
    pub(crate) instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU TextureView handle.
pub struct WGPUTextureViewImpl {
    pub(crate) _core: Arc<core::TextureView>,
    pub(crate) _texture: Arc<core::Texture>,
    pub(crate) _device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU Sampler handle.
pub struct WGPUSamplerImpl {
    pub(crate) _core: Arc<core::Sampler>,
    pub(crate) _device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU ShaderModule handle.
pub struct WGPUShaderModuleImpl {
    pub(crate) _core: Arc<core::ShaderModule>,
    pub(crate) _device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU PipelineLayout handle.
pub struct WGPUPipelineLayoutImpl {
    pub(crate) _core: Arc<core::PipelineLayout>,
    pub(crate) _device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU ComputePipeline handle.
pub struct WGPUComputePipelineImpl {
    pub(crate) _core: Arc<core::ComputePipeline>,
    pub(crate) _device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
    pub(crate) bind_group_layout_handles: Mutex<Vec<Option<Arc<WGPUBindGroupLayoutImpl>>>>,
}

/// Owns the core object and retained handles for the WGPU RenderPipeline handle.
pub struct WGPURenderPipelineImpl {
    pub(crate) _core: Arc<core::RenderPipeline>,
    pub(crate) _device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
    pub(crate) bind_group_layout_handles: Mutex<Vec<Option<Arc<WGPUBindGroupLayoutImpl>>>>,
}

/// Enumerates shader module cache key values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ShaderModuleCacheKey {
    /// Wgsl variant.
    Wgsl(String),
    /// Spirv variant.
    Spirv(Vec<u32>),
}

/// Identifies pipeline layout cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PipelineLayoutCacheKey {
    bind_group_layouts: Vec<usize>,
    immediate_size: u32,
}

/// Enumerates pipeline layout identity values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum PipelineLayoutIdentity {
    /// Auto variant.
    Auto,
    /// Explicit variant.
    Explicit(usize),
}

/// Identifies pipeline constant cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PipelineConstantCacheKey {
    key: String,
    value_bits: u64,
}

/// Identifies compute pipeline cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ComputePipelineCacheKey {
    module: usize,
    entry_point: Option<String>,
    constants: Vec<PipelineConstantCacheKey>,
    layout: PipelineLayoutIdentity,
}

/// Identifies render pipeline cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RenderPipelineCacheKey {
    layout: PipelineLayoutIdentity,
    vertex: RenderStageCacheKey,
    vertex_buffers: Vec<VertexBufferLayoutCacheKey>,
    primitive: PrimitiveStateCacheKey,
    depth_stencil: Option<DepthStencilStateCacheKey>,
    multisample: MultisampleStateCacheKey,
    fragment: Option<FragmentStateCacheKey>,
}

/// Identifies render stage cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RenderStageCacheKey {
    module: usize,
    entry_point: Option<String>,
    constants: Vec<PipelineConstantCacheKey>,
}

/// Identifies fragment state cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FragmentStateCacheKey {
    stage: RenderStageCacheKey,
    target_count: usize,
    targets: Vec<ColorTargetCacheKey>,
}

/// Identifies color target cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ColorTargetCacheKey {
    format: native::WGPUTextureFormat,
    blend: Option<BlendStateCacheKey>,
    write_mask: native::WGPUColorWriteMask,
}

/// Identifies blend state cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct BlendStateCacheKey {
    color: BlendComponentCacheKey,
    alpha: BlendComponentCacheKey,
}

/// Identifies blend component cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct BlendComponentCacheKey {
    operation: native::WGPUBlendOperation,
    src_factor: native::WGPUBlendFactor,
    dst_factor: native::WGPUBlendFactor,
}

/// Identifies vertex buffer layout cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct VertexBufferLayoutCacheKey {
    step_mode: native::WGPUVertexStepMode,
    array_stride: u64,
    attributes: Vec<VertexAttributeCacheKey>,
}

/// Identifies vertex attribute cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct VertexAttributeCacheKey {
    format: native::WGPUVertexFormat,
    offset: u64,
    shader_location: u32,
}

/// Identifies primitive state cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PrimitiveStateCacheKey {
    topology: native::WGPUPrimitiveTopology,
    strip_index_format: native::WGPUIndexFormat,
    front_face: native::WGPUFrontFace,
    cull_mode: native::WGPUCullMode,
    unclipped_depth: native::WGPUBool,
}

/// Identifies depth stencil state cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct DepthStencilStateCacheKey {
    format: native::WGPUTextureFormat,
    depth_write_enabled: native::WGPUOptionalBool,
    depth_compare: native::WGPUCompareFunction,
    stencil_front: StencilFaceStateCacheKey,
    stencil_back: StencilFaceStateCacheKey,
    stencil_read_mask: u32,
    stencil_write_mask: u32,
    depth_bias: i32,
    depth_bias_slope_scale_bits: u32,
    depth_bias_clamp_bits: u32,
}

/// Identifies stencil face state cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct StencilFaceStateCacheKey {
    compare: native::WGPUCompareFunction,
    fail_op: native::WGPUStencilOperation,
    depth_fail_op: native::WGPUStencilOperation,
    pass_op: native::WGPUStencilOperation,
}

/// Identifies multisample state cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MultisampleStateCacheKey {
    count: u32,
    mask: u32,
    alpha_to_coverage_enabled: native::WGPUBool,
}

/// Owns the core object and retained handles for the WGPU Surface handle.
pub struct WGPUSurfaceImpl {
    pub(crate) label: Mutex<String>,
    pub(crate) configured: Mutex<Option<SurfaceConfigurationState>>,
    pub(crate) hal: Mutex<Option<HalSurface>>,
    pub(crate) is_error: bool,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Tracks the lifecycle state for surface configuration.
#[derive(Debug, Clone)]
pub(crate) struct SurfaceConfigurationState {
    device: Arc<core::Device>,
    format: native::WGPUTextureFormat,
    usage: native::WGPUTextureUsage,
    width: u32,
    height: u32,
    view_formats: Vec<native::WGPUTextureFormat>,
    _present_mode: native::WGPUPresentMode,
    _alpha_mode: native::WGPUCompositeAlphaMode,
}

/// Owns the core object and retained handles for the WGPU QuerySet handle.
pub struct WGPUQuerySetImpl {
    pub(crate) core: Arc<core::QuerySet>,
    pub(crate) _device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU CommandEncoder handle.
pub struct WGPUCommandEncoderImpl {
    pub(crate) core: Arc<core::CommandEncoder>,
    pub(crate) device: Arc<core::Device>,
    pub(crate) instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU CommandBuffer handle.
pub struct WGPUCommandBufferImpl {
    pub(crate) core: Arc<core::CommandBuffer>,
    pub(crate) _device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU RenderPassEncoder handle.
pub struct WGPURenderPassEncoderImpl {
    pub(crate) core: Arc<core::RenderPassEncoder>,
    pub(crate) device: Arc<core::Device>,
    pub(crate) _parent: Arc<core::CommandEncoder>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU ComputePassEncoder handle.
pub struct WGPUComputePassEncoderImpl {
    pub(crate) core: Arc<core::ComputePassEncoder>,
    pub(crate) device: Arc<core::Device>,
    pub(crate) _parent: Arc<core::CommandEncoder>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU RenderBundleEncoder handle.
pub struct WGPURenderBundleEncoderImpl {
    pub(crate) core: Arc<core::RenderBundleEncoder>,
    pub(crate) device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

/// Owns the core object and retained handles for the WGPU RenderBundle handle.
pub struct WGPURenderBundleImpl {
    pub(crate) core: Arc<core::RenderBundle>,
    pub(crate) _device: Arc<core::Device>,
    pub(crate) _instance: Arc<WGPUInstanceImpl>,
}

impl WGPUInstanceImpl {
    fn new_noop(timed_wait_any_enabled: bool) -> Arc<Self> {
        Self::from_core(core::Instance::new_noop(), timed_wait_any_enabled)
    }

    fn from_core(core: core::Instance, timed_wait_any_enabled: bool) -> Arc<Self> {
        Arc::new(Self {
            core: Arc::new(core),
            timed_wait_any_enabled,
            pending_callbacks: Mutex::new(BTreeMap::new()),
        })
    }

    fn register_callback(&self, callback: PendingCallback) -> native::WGPUFuture {
        let future = self.register_pending_callback(callback);
        self.complete_future(future);
        future
    }

    fn register_pending_callback(&self, callback: PendingCallback) -> native::WGPUFuture {
        let future = self
            .core
            .future_registry()
            .register(callback.callback_mode());
        self.pending_callbacks
            .lock()
            .expect("pending callback lock is not poisoned")
            .insert(future.get(), callback);
        native::WGPUFuture { id: future.get() }
    }

    fn complete_future(&self, future: native::WGPUFuture) {
        self.core
            .future_registry()
            .complete(core::FutureId::from_raw(future.id));
    }

    fn abort_pending_device_callbacks(&self, device: &core::Device) {
        let mut callbacks = self
            .pending_callbacks
            .lock()
            .expect("pending callback lock is not poisoned");
        for callback in callbacks.values_mut() {
            match callback {
                PendingCallback::BufferMap {
                    device: callback_device,
                    buffer,
                    status,
                    ..
                } if callback_device.same(device) => {
                    if let Some(buffer) = buffer.take() {
                        buffer.abort_pending_map();
                    }
                    *status = core::MapAsyncStatus::Aborted;
                }
                PendingCallback::QueueWorkDone {
                    device: callback_device,
                    status,
                    ..
                } if callback_device.same(device) => {
                    *status = core::QueueWorkDoneStatus::Error;
                }
                _ => {}
            }
        }
    }

    fn process_callbacks(&self) -> usize {
        let ready = self.core.future_registry().process_events();
        let mut callbacks = self
            .pending_callbacks
            .lock()
            .expect("pending callback lock is not poisoned");
        let callbacks_to_fire = ready
            .into_iter()
            .filter_map(|id| callbacks.remove(&id.get()))
            .collect::<Vec<_>>();
        drop(callbacks);

        let count = callbacks_to_fire.len();
        for callback in callbacks_to_fire {
            unsafe {
                callback.fire();
            }
        }
        count
    }

    fn wait_any(&self, future_ids: &[core::FutureId]) -> core::WaitAnyResult {
        let result = self.core.future_registry().wait_any(future_ids);

        let mut callbacks = self
            .pending_callbacks
            .lock()
            .expect("pending callback lock is not poisoned");
        let callbacks_to_fire = result
            .callbacks_to_fire
            .iter()
            .filter_map(|id| callbacks.remove(&id.get()))
            .collect::<Vec<_>>();
        drop(callbacks);

        for callback in callbacks_to_fire {
            unsafe {
                callback.fire();
            }
        }

        result
    }
}

impl Drop for WGPUBufferImpl {
    fn drop(&mut self) {
        self.core.abort_pending_map();
        self.core.destroy();
    }
}

impl Drop for WGPUTextureImpl {
    fn drop(&mut self) {
        self.core.destroy();
    }
}

impl Drop for WGPUDeviceImpl {
    fn drop(&mut self) {
        self.implicit_destroy_on_last_release();
    }
}

impl WGPUDeviceImpl {
    fn implicit_destroy_on_last_release(&self) {
        self.schedule_device_lost(std::ptr::null(), core::DeviceLostReason::Destroyed);
    }

    fn schedule_device_lost(
        &self,
        device: native::WGPUDevice,
        reason: core::DeviceLostReason,
    ) -> Option<native::WGPUFuture> {
        let reason = self.core.lose(reason)?;
        self.instance.abort_pending_device_callbacks(&self.core);
        for future_id in self
            .device_lost_futures
            .lock()
            .expect("device lost future lock is not poisoned")
            .drain(..)
        {
            self.instance
                .complete_future(native::WGPUFuture { id: future_id });
        }
        if !self.device_lost_callback.has_callback() {
            return None;
        }
        Some(
            self.instance
                .register_callback(PendingCallback::DeviceLost {
                    mode: self.device_lost_callback.mode,
                    callback: self.device_lost_callback.callback,
                    device: device as usize,
                    reason,
                    userdata1: self.device_lost_callback.userdata1,
                    userdata2: self.device_lost_callback.userdata2,
                }),
        )
    }

    fn get_lost_future(&self, device: native::WGPUDevice) -> native::WGPUFuture {
        let reason = self
            .core
            .lost_reason()
            .unwrap_or(core::DeviceLostReason::Unknown);
        let future = self
            .instance
            .register_pending_callback(PendingCallback::DeviceLost {
                mode: 0,
                callback: None,
                device: device as usize,
                reason,
                userdata1: 0,
                userdata2: 0,
            });
        if self.core.is_lost() {
            self.instance.complete_future(future);
        } else {
            self.device_lost_futures
                .lock()
                .expect("device lost future lock is not poisoned")
                .push(future.id);
        }
        future
    }

    /// Sets uncaptured error callback on this object or encoder.
    #[doc(hidden)]
    pub fn set_uncaptured_error_callback<F>(&self, callback: Option<F>)
    where
        F: Fn(core::DeviceError) + Send + Sync + 'static,
    {
        self.core.set_uncaptured_error_callback(callback);
    }

    /// Dispatches the pending error callback.
    #[doc(hidden)]
    pub fn dispatch_error(&self, kind: core::ErrorKind, message: impl Into<String>) {
        self.core.dispatch_error(kind, message);
    }

    fn default_queue(&self) -> Arc<WGPUQueueImpl> {
        let mut queue = self
            .default_queue
            .lock()
            .expect("default queue lock is not poisoned");
        Arc::clone(queue.get_or_insert_with(|| {
            Arc::new(WGPUQueueImpl {
                core: self.core.queue(),
                device: Arc::clone(&self.core),
                instance: Arc::clone(&self.instance),
            })
        }))
    }
}

// yawgpu's FFI cache dedups by a structural descriptor key whose sub-objects
// are identified by C-handle identity. This matches the webgpu.h-observable
// pointer equality in Dawn's ObjectCaching tests; deeper engine-internal dedup
// of content-equal but handle-distinct sub-objects is intentionally out of scope.
fn cache_handle<T>(
    cache: &Mutex<HashMap<T, Arc<T::Handle>>>,
    key: T,
    handle: Arc<T::Handle>,
) -> Arc<T::Handle>
where
    T: CacheKey,
{
    let mut cache = cache
        .lock()
        .expect("device object cache lock must not poison");
    if let Some(cached) = cache.get(&key) {
        return Arc::clone(cached);
    }
    cache.insert(key, Arc::clone(&handle));
    handle
}

trait CacheKey: Eq + std::hash::Hash {
    type Handle;
}

impl CacheKey for ShaderModuleCacheKey {
    type Handle = WGPUShaderModuleImpl;
}

impl CacheKey for PipelineLayoutCacheKey {
    type Handle = WGPUPipelineLayoutImpl;
}

impl CacheKey for ComputePipelineCacheKey {
    type Handle = WGPUComputePipelineImpl;
}

impl CacheKey for RenderPipelineCacheKey {
    type Handle = WGPURenderPipelineImpl;
}

fn shader_module_cache_key(source: &core::ShaderModuleSource) -> Option<ShaderModuleCacheKey> {
    match source {
        core::ShaderModuleSource::Wgsl(source) => Some(ShaderModuleCacheKey::Wgsl(source.clone())),
        core::ShaderModuleSource::Spirv(words) => Some(ShaderModuleCacheKey::Spirv(words.clone())),
        core::ShaderModuleSource::Invalid(_) => None,
        _ => None,
    }
}

unsafe fn pipeline_layout_cache_key(
    descriptor: &native::WGPUPipelineLayoutDescriptor,
) -> Option<PipelineLayoutCacheKey> {
    let bind_group_layouts = if descriptor.bindGroupLayoutCount == 0 {
        Vec::new()
    } else if descriptor.bindGroupLayouts.is_null() {
        return None;
    } else {
        std::slice::from_raw_parts(descriptor.bindGroupLayouts, descriptor.bindGroupLayoutCount)
            .iter()
            .copied()
            .map(|layout| (!layout.is_null()).then_some(layout as usize))
            .collect::<Option<Vec<_>>>()?
    };
    Some(PipelineLayoutCacheKey {
        bind_group_layouts,
        immediate_size: descriptor.immediateSize,
    })
}

fn layout_identity(layout: native::WGPUPipelineLayout) -> PipelineLayoutIdentity {
    if layout.is_null() {
        PipelineLayoutIdentity::Auto
    } else {
        PipelineLayoutIdentity::Explicit(layout as usize)
    }
}

unsafe fn compute_pipeline_cache_key(
    descriptor: &native::WGPUComputePipelineDescriptor,
) -> Option<ComputePipelineCacheKey> {
    if descriptor.compute.module.is_null() {
        return None;
    }
    Some(ComputePipelineCacheKey {
        module: descriptor.compute.module as usize,
        entry_point: cache_string_view(descriptor.compute.entryPoint),
        constants: pipeline_constant_cache_keys(
            descriptor.compute.constantCount,
            descriptor.compute.constants,
        )?,
        layout: layout_identity(descriptor.layout),
    })
}

unsafe fn render_pipeline_cache_key(
    descriptor: &native::WGPURenderPipelineDescriptor,
) -> Option<RenderPipelineCacheKey> {
    if descriptor.vertex.module.is_null() {
        return None;
    }
    Some(RenderPipelineCacheKey {
        layout: layout_identity(descriptor.layout),
        vertex: RenderStageCacheKey {
            module: descriptor.vertex.module as usize,
            entry_point: cache_string_view(descriptor.vertex.entryPoint),
            constants: pipeline_constant_cache_keys(
                descriptor.vertex.constantCount,
                descriptor.vertex.constants,
            )?,
        },
        vertex_buffers: vertex_buffer_cache_keys(
            descriptor.vertex.bufferCount,
            descriptor.vertex.buffers,
        )?,
        primitive: primitive_state_cache_key(descriptor.primitive),
        depth_stencil: descriptor
            .depthStencil
            .as_ref()
            .map(depth_stencil_state_cache_key),
        multisample: multisample_state_cache_key(descriptor.multisample),
        fragment: descriptor
            .fragment
            .as_ref()
            .and_then(|fragment| fragment_state_cache_key(fragment)),
    })
}

unsafe fn validate_pipeline_layout_devices(
    device: &WGPUDeviceImpl,
    descriptor: &native::WGPUPipelineLayoutDescriptor,
) -> Option<String> {
    if descriptor.bindGroupLayoutCount == 0 || descriptor.bindGroupLayouts.is_null() {
        return None;
    }
    for layout in
        std::slice::from_raw_parts(descriptor.bindGroupLayouts, descriptor.bindGroupLayoutCount)
    {
        if layout.is_null() {
            continue;
        }
        let layout = clone_handle::<WGPUBindGroupLayoutImpl>(*layout, "WGPUBindGroupLayout");
        if !layout._device.same(&device.core) {
            return Some("pipeline layout bind group layout must belong to the same device".into());
        }
    }
    None
}

unsafe fn validate_compute_pipeline_devices(
    device: &WGPUDeviceImpl,
    descriptor: &native::WGPUComputePipelineDescriptor,
) -> Option<String> {
    let module =
        clone_handle::<WGPUShaderModuleImpl>(descriptor.compute.module, "WGPUShaderModule");
    if !module._device.same(&device.core) {
        return Some("compute pipeline shader module must belong to the same device".into());
    }
    if !descriptor.layout.is_null() {
        let layout =
            clone_handle::<WGPUPipelineLayoutImpl>(descriptor.layout, "WGPUPipelineLayout");
        if !layout._device.same(&device.core) {
            return Some("compute pipeline layout must belong to the same device".into());
        }
    }
    None
}

unsafe fn validate_render_pipeline_devices(
    device: &WGPUDeviceImpl,
    descriptor: &native::WGPURenderPipelineDescriptor,
) -> Option<String> {
    let vertex_module =
        clone_handle::<WGPUShaderModuleImpl>(descriptor.vertex.module, "WGPUShaderModule");
    if !vertex_module._device.same(&device.core) {
        return Some("render pipeline vertex shader module must belong to the same device".into());
    }
    if let Some(fragment) = descriptor.fragment.as_ref() {
        let fragment_module =
            clone_handle::<WGPUShaderModuleImpl>(fragment.module, "WGPUShaderModule");
        if !fragment_module._device.same(&device.core) {
            return Some(
                "render pipeline fragment shader module must belong to the same device".into(),
            );
        }
    }
    if !descriptor.layout.is_null() {
        let layout =
            clone_handle::<WGPUPipelineLayoutImpl>(descriptor.layout, "WGPUPipelineLayout");
        if !layout._device.same(&device.core) {
            return Some("render pipeline layout must belong to the same device".into());
        }
    }
    None
}

const SURFACE_FORMATS: [native::WGPUTextureFormat; 2] = [
    native::WGPUTextureFormat_BGRA8Unorm,
    native::WGPUTextureFormat_RGBA8Unorm,
];
const SURFACE_PRESENT_MODES: [native::WGPUPresentMode; 1] = [native::WGPUPresentMode_Fifo];
const SURFACE_ALPHA_MODES: [native::WGPUCompositeAlphaMode; 1] =
    [native::WGPUCompositeAlphaMode_Opaque];
const SURFACE_USAGES: native::WGPUTextureUsage = native::WGPUTextureUsage_RenderAttachment;

fn is_supported_surface_source(s_type: native::WGPUSType) -> bool {
    matches!(
        s_type,
        native::WGPUSType_SurfaceSourceMetalLayer
            | native::WGPUSType_SurfaceSourceWindowsHWND
            | native::WGPUSType_SurfaceSourceXlibWindow
            | native::WGPUSType_SurfaceSourceWaylandSurface
            | native::WGPUSType_SurfaceSourceXCBWindow
            | native::WGPUSType_SurfaceSourceAndroidNativeWindow
    )
}

unsafe fn has_supported_surface_source(mut chain: *const native::WGPUChainedStruct) -> bool {
    while let Some(link) = chain.as_ref() {
        if is_supported_surface_source(link.sType) {
            return true;
        }
        chain = link.next;
    }
    false
}

unsafe fn find_metal_layer_source(
    mut chain: *const native::WGPUChainedStruct,
) -> Option<*mut c_void> {
    while let Some(link) = chain.as_ref() {
        if link.sType == native::WGPUSType_SurfaceSourceMetalLayer {
            let source = (link as *const native::WGPUChainedStruct)
                .cast::<native::WGPUSurfaceSourceMetalLayer>();
            return source.as_ref().map(|source| source.layer);
        }
        chain = link.next;
    }
    None
}

unsafe fn find_windows_hwnd_source(
    mut chain: *const native::WGPUChainedStruct,
) -> Option<(*mut c_void, *mut c_void)> {
    while let Some(link) = chain.as_ref() {
        if link.sType == native::WGPUSType_SurfaceSourceWindowsHWND {
            let source = (link as *const native::WGPUChainedStruct)
                .cast::<native::WGPUSurfaceSourceWindowsHWND>();
            return source
                .as_ref()
                .map(|source| (source.hinstance, source.hwnd));
        }
        chain = link.next;
    }
    None
}

fn real_hal_surface(surface: HalSurface) -> Option<HalSurface> {
    match surface {
        HalSurface::Noop => None,
        other => Some(other),
    }
}

fn is_real_hal_instance(instance: &HalInstance) -> bool {
    #[allow(unreachable_patterns)]
    match instance {
        #[cfg(feature = "vulkan")]
        HalInstance::Vulkan(_) => true,
        #[cfg(feature = "metal")]
        HalInstance::Metal(_) => true,
        _ => false,
    }
}

fn hal_surface_format(format: native::WGPUTextureFormat) -> HalTextureFormat {
    match format {
        native::WGPUTextureFormat_RGBA8Unorm => HalTextureFormat::Rgba8Unorm,
        native::WGPUTextureFormat_BGRA8Unorm => HalTextureFormat::Bgra8Unorm,
        _ => HalTextureFormat::Unsupported,
    }
}

fn hal_surface_usage(usage: native::WGPUTextureUsage) -> HalTextureUsage {
    HalTextureUsage {
        copy_src: usage & native::WGPUTextureUsage_CopySrc != 0,
        copy_dst: usage & native::WGPUTextureUsage_CopyDst != 0,
        texture_binding: usage & native::WGPUTextureUsage_TextureBinding != 0,
        storage_binding: usage & native::WGPUTextureUsage_StorageBinding != 0,
        render_attachment: usage & native::WGPUTextureUsage_RenderAttachment != 0,
    }
}

fn hal_present_mode(mode: native::WGPUPresentMode) -> HalPresentMode {
    match mode {
        native::WGPUPresentMode_Immediate => HalPresentMode::Immediate,
        native::WGPUPresentMode_Mailbox => HalPresentMode::Mailbox,
        _ => HalPresentMode::Fifo,
    }
}

fn surface_configuration_error(
    device: &WGPUDeviceImpl,
    config: &native::WGPUSurfaceConfiguration,
) -> Option<&'static str> {
    if device.core.is_lost() {
        return Some("surface configuration device is lost");
    }
    if !SURFACE_FORMATS.contains(&config.format) {
        return Some("surface configuration format is not supported");
    }
    if config.usage == native::WGPUTextureUsage_None || config.usage & !SURFACE_USAGES != 0 {
        return Some("surface configuration usage is not supported");
    }
    if config.width == 0 || config.height == 0 {
        return Some("surface configuration size must be non-zero");
    }
    if !SURFACE_PRESENT_MODES.contains(&config.presentMode) {
        return Some("surface configuration present mode is not supported");
    }
    if !SURFACE_ALPHA_MODES.contains(&config.alphaMode) {
        return Some("surface configuration alpha mode is not supported");
    }
    None
}

fn cache_string_view(value: native::WGPUStringView) -> Option<String> {
    unsafe { string_view_to_str(value).map(ToOwned::to_owned) }
}

unsafe fn pipeline_constant_cache_keys(
    count: usize,
    constants: *const native::WGPUConstantEntry,
) -> Option<Vec<PipelineConstantCacheKey>> {
    if count == 0 {
        return Some(Vec::new());
    }
    if constants.is_null() {
        return None;
    }
    let mut keys = std::slice::from_raw_parts(constants, count)
        .iter()
        .map(|constant| PipelineConstantCacheKey {
            key: cache_string_view(constant.key).unwrap_or_default(),
            value_bits: canonical_f64_bits(constant.value),
        })
        .collect::<Vec<_>>();
    keys.sort_by(|a, b| {
        a.key
            .cmp(&b.key)
            .then_with(|| a.value_bits.cmp(&b.value_bits))
    });
    Some(keys)
}

fn canonical_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else if value.is_nan() {
        f64::NAN.to_bits()
    } else {
        value.to_bits()
    }
}

fn canonical_f32_bits(value: f32) -> u32 {
    if value == 0.0 {
        0.0f32.to_bits()
    } else if value.is_nan() {
        f32::NAN.to_bits()
    } else {
        value.to_bits()
    }
}

unsafe fn vertex_buffer_cache_keys(
    count: usize,
    buffers: *const native::WGPUVertexBufferLayout,
) -> Option<Vec<VertexBufferLayoutCacheKey>> {
    if count == 0 {
        return Some(Vec::new());
    }
    if buffers.is_null() {
        return None;
    }
    std::slice::from_raw_parts(buffers, count)
        .iter()
        .map(|buffer| vertex_buffer_cache_key(buffer))
        .collect()
}

unsafe fn vertex_buffer_cache_key(
    buffer: &native::WGPUVertexBufferLayout,
) -> Option<VertexBufferLayoutCacheKey> {
    let attributes = if buffer.attributeCount == 0 {
        Vec::new()
    } else if buffer.attributes.is_null() {
        return None;
    } else {
        std::slice::from_raw_parts(buffer.attributes, buffer.attributeCount)
            .iter()
            .map(|attribute| VertexAttributeCacheKey {
                format: attribute.format,
                offset: attribute.offset,
                shader_location: attribute.shaderLocation,
            })
            .collect()
    };
    Some(VertexBufferLayoutCacheKey {
        step_mode: buffer.stepMode,
        array_stride: buffer.arrayStride,
        attributes,
    })
}

unsafe fn fragment_state_cache_key(
    fragment: &native::WGPUFragmentState,
) -> Option<FragmentStateCacheKey> {
    if fragment.module.is_null() {
        return None;
    }
    Some(FragmentStateCacheKey {
        stage: RenderStageCacheKey {
            module: fragment.module as usize,
            entry_point: cache_string_view(fragment.entryPoint),
            constants: pipeline_constant_cache_keys(fragment.constantCount, fragment.constants)?,
        },
        target_count: fragment.targetCount,
        targets: color_target_cache_keys(fragment.targetCount, fragment.targets)?,
    })
}

unsafe fn color_target_cache_keys(
    count: usize,
    targets: *const native::WGPUColorTargetState,
) -> Option<Vec<ColorTargetCacheKey>> {
    if count == 0 {
        return Some(Vec::new());
    }
    if targets.is_null() {
        return None;
    }
    Some(
        std::slice::from_raw_parts(targets, count)
            .iter()
            .map(|target| ColorTargetCacheKey {
                format: target.format,
                blend: target.blend.as_ref().map(|blend| BlendStateCacheKey {
                    color: blend_component_cache_key(blend.color),
                    alpha: blend_component_cache_key(blend.alpha),
                }),
                write_mask: target.writeMask,
            })
            .collect(),
    )
}

fn blend_component_cache_key(component: native::WGPUBlendComponent) -> BlendComponentCacheKey {
    BlendComponentCacheKey {
        operation: component.operation,
        src_factor: component.srcFactor,
        dst_factor: component.dstFactor,
    }
}

fn primitive_state_cache_key(primitive: native::WGPUPrimitiveState) -> PrimitiveStateCacheKey {
    PrimitiveStateCacheKey {
        topology: primitive.topology,
        strip_index_format: primitive.stripIndexFormat,
        front_face: primitive.frontFace,
        cull_mode: primitive.cullMode,
        unclipped_depth: primitive.unclippedDepth,
    }
}

fn depth_stencil_state_cache_key(
    depth_stencil: &native::WGPUDepthStencilState,
) -> DepthStencilStateCacheKey {
    DepthStencilStateCacheKey {
        format: depth_stencil.format,
        depth_write_enabled: depth_stencil.depthWriteEnabled,
        depth_compare: depth_stencil.depthCompare,
        stencil_front: stencil_face_state_cache_key(depth_stencil.stencilFront),
        stencil_back: stencil_face_state_cache_key(depth_stencil.stencilBack),
        stencil_read_mask: depth_stencil.stencilReadMask,
        stencil_write_mask: depth_stencil.stencilWriteMask,
        depth_bias: depth_stencil.depthBias,
        depth_bias_slope_scale_bits: canonical_f32_bits(depth_stencil.depthBiasSlopeScale),
        depth_bias_clamp_bits: canonical_f32_bits(depth_stencil.depthBiasClamp),
    }
}

fn stencil_face_state_cache_key(face: native::WGPUStencilFaceState) -> StencilFaceStateCacheKey {
    StencilFaceStateCacheKey {
        compare: face.compare,
        fail_op: face.failOp,
        depth_fail_op: face.depthFailOp,
        pass_op: face.passOp,
    }
}

fn multisample_state_cache_key(
    multisample: native::WGPUMultisampleState,
) -> MultisampleStateCacheKey {
    MultisampleStateCacheKey {
        count: multisample.count,
        mask: multisample.mask,
        alpha_to_coverage_enabled: multisample.alphaToCoverageEnabled,
    }
}

/// Enumerates pending callback values.
pub(crate) enum PendingCallback {
    /// Request adapter variant.
    RequestAdapter {
        /// Mode variant.
        mode: native::WGPUCallbackMode,
        /// Callback variant.
        callback: native::WGPURequestAdapterCallback,
        /// Adapter variant.
        adapter: Arc<WGPUAdapterImpl>,
        /// Userdata1 variant.
        userdata1: usize,
        /// Userdata2 variant.
        userdata2: usize,
    },
    /// Request device variant.
    RequestDevice {
        /// Mode variant.
        mode: native::WGPUCallbackMode,
        /// Callback variant.
        callback: native::WGPURequestDeviceCallback,
        /// Result variant.
        result: Result<Arc<WGPUDeviceImpl>, String>,
        /// Userdata1 variant.
        userdata1: usize,
        /// Userdata2 variant.
        userdata2: usize,
    },
    /// Device lost variant.
    DeviceLost {
        /// Mode variant.
        mode: native::WGPUCallbackMode,
        /// Callback variant.
        callback: native::WGPUDeviceLostCallback,
        /// Device variant.
        device: usize,
        /// Reason variant.
        reason: core::DeviceLostReason,
        /// Userdata1 variant.
        userdata1: usize,
        /// Userdata2 variant.
        userdata2: usize,
    },
    /// Buffer map variant.
    BufferMap {
        /// Mode variant.
        mode: native::WGPUCallbackMode,
        /// Callback variant.
        callback: native::WGPUBufferMapCallback,
        /// Device variant.
        device: Arc<core::Device>,
        /// Buffer variant.
        buffer: Option<core::Buffer>,
        /// Status variant.
        status: core::MapAsyncStatus,
        /// Userdata1 variant.
        userdata1: usize,
        /// Userdata2 variant.
        userdata2: usize,
    },
    /// Queue work done variant.
    QueueWorkDone {
        /// Mode variant.
        mode: native::WGPUCallbackMode,
        /// Callback variant.
        callback: native::WGPUQueueWorkDoneCallback,
        /// Device variant.
        device: Arc<core::Device>,
        /// Status variant.
        status: core::QueueWorkDoneStatus,
        /// Userdata1 variant.
        userdata1: usize,
        /// Userdata2 variant.
        userdata2: usize,
    },
    /// Compilation info variant.
    CompilationInfo {
        /// Mode variant.
        mode: native::WGPUCallbackMode,
        /// Callback variant.
        callback: native::WGPUCompilationInfoCallback,
        /// Shader module variant.
        shader_module: Arc<core::ShaderModule>,
        /// Userdata1 variant.
        userdata1: usize,
        /// Userdata2 variant.
        userdata2: usize,
    },
    /// Create compute pipeline async variant.
    CreateComputePipelineAsync {
        /// Mode variant.
        mode: native::WGPUCallbackMode,
        /// Callback variant.
        callback: native::WGPUCreateComputePipelineAsyncCallback,
        /// Pipeline variant.
        pipeline: Arc<WGPUComputePipelineImpl>,
        /// Userdata1 variant.
        userdata1: usize,
        /// Userdata2 variant.
        userdata2: usize,
    },
    /// Create render pipeline async variant.
    CreateRenderPipelineAsync {
        /// Mode variant.
        mode: native::WGPUCallbackMode,
        /// Callback variant.
        callback: native::WGPUCreateRenderPipelineAsyncCallback,
        /// Pipeline variant.
        pipeline: Arc<WGPURenderPipelineImpl>,
        /// Userdata1 variant.
        userdata1: usize,
        /// Userdata2 variant.
        userdata2: usize,
    },
    /// Pop error scope variant.
    PopErrorScope {
        /// Mode variant.
        mode: native::WGPUCallbackMode,
        /// Callback variant.
        callback: native::WGPUPopErrorScopeCallback,
        /// Status variant.
        status: native::WGPUPopErrorScopeStatus,
        /// Error variant.
        error: Option<core::DeviceError>,
        /// Message variant.
        message: String,
        /// Userdata1 variant.
        userdata1: usize,
        /// Userdata2 variant.
        userdata2: usize,
    },
}

impl PendingCallback {
    fn callback_mode(&self) -> core::FutureCallbackMode {
        let mode = match self {
            Self::RequestAdapter { mode, .. }
            | Self::RequestDevice { mode, .. }
            | Self::DeviceLost { mode, .. }
            | Self::BufferMap { mode, .. }
            | Self::QueueWorkDone { mode, .. }
            | Self::CompilationInfo { mode, .. }
            | Self::CreateComputePipelineAsync { mode, .. }
            | Self::CreateRenderPipelineAsync { mode, .. }
            | Self::PopErrorScope { mode, .. } => *mode,
        };
        match mode {
            native::WGPUCallbackMode_AllowProcessEvents => {
                core::FutureCallbackMode::AllowProcessEvents
            }
            native::WGPUCallbackMode_AllowSpontaneous => core::FutureCallbackMode::AllowSpontaneous,
            _ => core::FutureCallbackMode::WaitAnyOnly,
        }
    }

    unsafe fn fire(self) {
        match self {
            Self::RequestAdapter {
                callback,
                adapter,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    callback(
                        native::WGPURequestAdapterStatus_Success,
                        arc_to_handle(adapter),
                        string_view(b""),
                        userdata1 as *mut c_void,
                        userdata2 as *mut c_void,
                    );
                }
            }
            Self::RequestDevice {
                callback,
                result,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    match result {
                        Ok(device) => callback(
                            native::WGPURequestDeviceStatus_Success,
                            arc_to_handle(device),
                            string_view(b""),
                            userdata1 as *mut c_void,
                            userdata2 as *mut c_void,
                        ),
                        Err(message) => callback(
                            native::WGPURequestDeviceStatus_Error,
                            std::ptr::null(),
                            string_view(message.as_bytes()),
                            userdata1 as *mut c_void,
                            userdata2 as *mut c_void,
                        ),
                    }
                }
            }
            Self::DeviceLost {
                callback,
                device,
                reason,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    let device = device as native::WGPUDevice;
                    callback(
                        &device,
                        map_device_lost_reason(reason),
                        string_view(device_lost_message(reason).as_bytes()),
                        userdata1 as *mut c_void,
                        userdata2 as *mut c_void,
                    );
                }
            }
            Self::BufferMap {
                callback,
                buffer,
                status,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    let status = buffer
                        .as_ref()
                        .map(core::Buffer::resolve_pending_map)
                        .unwrap_or(status);
                    callback(
                        map_map_async_status(status),
                        string_view(map_async_message(status).as_bytes()),
                        userdata1 as *mut c_void,
                        userdata2 as *mut c_void,
                    );
                }
            }
            Self::QueueWorkDone {
                callback,
                status,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    callback(
                        map_queue_work_done_status(status),
                        string_view(queue_work_done_message(status).as_bytes()),
                        userdata1 as *mut c_void,
                        userdata2 as *mut c_void,
                    );
                }
            }
            Self::CompilationInfo {
                callback,
                shader_module,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    let diagnostic = shader_module.diagnostic();
                    let message = diagnostic.map(|message| native::WGPUCompilationMessage {
                        nextInChain: std::ptr::null_mut(),
                        message: string_view(message.as_bytes()),
                        type_: map_compilation_message_type_error(),
                        lineNum: 0,
                        linePos: 0,
                        offset: 0,
                        length: 0,
                    });
                    let messages = message.as_ref().map_or(std::ptr::null(), |message| message);
                    let info = native::WGPUCompilationInfo {
                        nextInChain: std::ptr::null_mut(),
                        messageCount: usize::from(message.is_some()),
                        messages,
                    };
                    callback(
                        map_compilation_info_request_status_success(),
                        &info,
                        userdata1 as *mut c_void,
                        userdata2 as *mut c_void,
                    );
                }
            }
            Self::CreateComputePipelineAsync {
                callback,
                pipeline,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    if pipeline._core.is_error() {
                        callback(
                            native::WGPUCreatePipelineAsyncStatus_ValidationError,
                            std::ptr::null(),
                            string_view(b"Pipeline creation failed validation"),
                            userdata1 as *mut c_void,
                            userdata2 as *mut c_void,
                        );
                    } else {
                        callback(
                            native::WGPUCreatePipelineAsyncStatus_Success,
                            arc_to_handle(pipeline),
                            string_view(b""),
                            userdata1 as *mut c_void,
                            userdata2 as *mut c_void,
                        );
                    }
                }
            }
            Self::CreateRenderPipelineAsync {
                callback,
                pipeline,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    if pipeline._core.is_error() {
                        callback(
                            native::WGPUCreatePipelineAsyncStatus_ValidationError,
                            std::ptr::null(),
                            string_view(b"Pipeline creation failed validation"),
                            userdata1 as *mut c_void,
                            userdata2 as *mut c_void,
                        );
                    } else {
                        callback(
                            native::WGPUCreatePipelineAsyncStatus_Success,
                            arc_to_handle(pipeline),
                            string_view(b""),
                            userdata1 as *mut c_void,
                            userdata2 as *mut c_void,
                        );
                    }
                }
            }
            Self::PopErrorScope {
                callback,
                status,
                error,
                message,
                userdata1,
                userdata2,
                ..
            } => {
                if let Some(callback) = callback {
                    let error_type = if status == native::WGPUPopErrorScopeStatus_Success {
                        error
                            .as_ref()
                            .map_or(native::WGPUErrorType_NoError, |error| {
                                map_error_type(error.kind)
                            })
                    } else {
                        native::WGPUErrorType_NoError
                    };
                    let message = error
                        .as_ref()
                        .map_or(message.as_bytes(), |error| error.message.as_bytes());
                    callback(
                        status,
                        error_type,
                        string_view(message),
                        userdata1 as *mut c_void,
                        userdata2 as *mut c_void,
                    );
                }
            }
        }
    }
}

fn queue_work_done_message(status: core::QueueWorkDoneStatus) -> &'static str {
    match status {
        core::QueueWorkDoneStatus::Success => "",
        core::QueueWorkDoneStatus::CallbackCancelled => "Queue work done callback was cancelled",
        core::QueueWorkDoneStatus::Error => "Queue work done failed",
        _ => "Queue work done failed",
    }
}

fn map_async_message(status: core::MapAsyncStatus) -> &'static str {
    match status {
        core::MapAsyncStatus::Success => "",
        core::MapAsyncStatus::Aborted => "Buffer map was aborted",
        core::MapAsyncStatus::CallbackCancelled => "Buffer map callback was cancelled",
        core::MapAsyncStatus::Error => "Buffer map failed",
        _ => "Buffer map failed",
    }
}

fn device_lost_message(reason: core::DeviceLostReason) -> &'static str {
    match reason {
        core::DeviceLostReason::Destroyed => "Device was destroyed",
        core::DeviceLostReason::FailedCreation => "Device creation failed",
        core::DeviceLostReason::CallbackCancelled => "Device lost callback was cancelled",
        core::DeviceLostReason::Unknown => "Device was lost",
        _ => "Device was lost",
    }
}

unsafe fn create_compute_pipeline_handle(
    device: &WGPUDeviceImpl,
    descriptor: &native::WGPUComputePipelineDescriptor,
    dispatch_errors: bool,
) -> Arc<WGPUComputePipelineImpl> {
    let key = compute_pipeline_cache_key(descriptor);
    let device_error = validate_compute_pipeline_devices(device, descriptor);
    let mut descriptor = map_compute_pipeline_descriptor(descriptor);
    if descriptor.error.is_none() {
        descriptor.error = device_error;
    }
    let pipeline = if dispatch_errors {
        device.core.create_compute_pipeline(descriptor)
    } else {
        device
            .core
            .create_compute_pipeline_without_error_dispatch(descriptor)
    };
    let handle = Arc::new(WGPUComputePipelineImpl {
        _core: Arc::new(pipeline),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
        bind_group_layout_handles: Mutex::new(Vec::new()),
    });
    if !handle._core.is_error() {
        if let Some(key) = key {
            cache_handle(&device.compute_pipeline_cache, key, handle)
        } else {
            handle
        }
    } else {
        handle
    }
}

unsafe fn create_render_pipeline_handle(
    device: &WGPUDeviceImpl,
    descriptor: &native::WGPURenderPipelineDescriptor,
    dispatch_errors: bool,
) -> Arc<WGPURenderPipelineImpl> {
    let key = render_pipeline_cache_key(descriptor);
    let device_error = validate_render_pipeline_devices(device, descriptor);
    let mut descriptor = map_render_pipeline_descriptor(descriptor);
    if descriptor.error.is_none() {
        descriptor.error = device_error;
    }
    let pipeline = if dispatch_errors {
        device.core.create_render_pipeline(descriptor)
    } else {
        device
            .core
            .create_render_pipeline_without_error_dispatch(descriptor)
    };
    let handle = Arc::new(WGPURenderPipelineImpl {
        _core: Arc::new(pipeline),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
        bind_group_layout_handles: Mutex::new(Vec::new()),
    });
    if !handle._core.is_error() {
        if let Some(key) = key {
            cache_handle(&device.render_pipeline_cache, key, handle)
        } else {
            handle
        }
    } else {
        handle
    }
}

fn dispatch_optional_error(device: &core::Device, error: Option<String>) {
    if let Some(message) = error {
        device.dispatch_error(core::ErrorKind::Validation, message);
    }
}

fn dispatch_optional_device_error(device: &core::Device, error: Option<core::DeviceError>) {
    if let Some(error) = error {
        device.dispatch_error(error.kind, error.message);
    }
}

fn adapter_info_from_core(adapter: &core::Adapter) -> native::WGPUAdapterInfo {
    let (backend_type, adapter_type) = match adapter.backend() {
        yawgpu_hal::HalBackend::Noop => (native::WGPUBackendType_Null, native::WGPUAdapterType_CPU),
        yawgpu_hal::HalBackend::Vulkan => (
            native::WGPUBackendType_Vulkan,
            native::WGPUAdapterType_Unknown,
        ),
        yawgpu_hal::HalBackend::Metal => (
            native::WGPUBackendType_Metal,
            native::WGPUAdapterType_Unknown,
        ),
        _ => (
            native::WGPUBackendType_Undefined,
            native::WGPUAdapterType_Unknown,
        ),
    };
    native::WGPUAdapterInfo {
        nextInChain: std::ptr::null_mut(),
        vendor: owned_string_view("yawgpu"),
        architecture: owned_string_view(""),
        device: owned_string_view(&adapter.name()),
        description: owned_string_view(""),
        backendType: backend_type,
        adapterType: adapter_type,
        vendorID: 0,
        deviceID: 0,
        subgroupMinSize: 0,
        subgroupMaxSize: 0,
    }
}

fn owned_string_view(value: &str) -> native::WGPUStringView {
    if value.is_empty() {
        return native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        };
    }
    let bytes = value.as_bytes().to_vec().into_boxed_slice();
    let length = bytes.len();
    let data = Box::into_raw(bytes).cast::<std::os::raw::c_char>();
    native::WGPUStringView { data, length }
}

unsafe fn free_owned_string_view(value: native::WGPUStringView) {
    if value.data.is_null() {
        return;
    }
    let slice =
        std::ptr::slice_from_raw_parts_mut(value.data.cast_mut().cast::<u8>(), value.length);
    drop(Box::from_raw(slice));
}

unsafe fn dynamic_offsets_slice(count: usize, offsets: *const u32) -> Vec<u32> {
    if count == 0 {
        return Vec::new();
    }
    assert!(
        !offsets.is_null(),
        "dynamicOffsets must not be null when count is non-zero"
    );
    std::slice::from_raw_parts(offsets, count).to_vec()
}

unsafe fn render_bundle_slice(
    count: usize,
    bundles: *const native::WGPURenderBundle,
) -> Vec<Arc<WGPURenderBundleImpl>> {
    if count == 0 {
        return Vec::new();
    }
    assert!(
        !bundles.is_null(),
        "bundles must not be null when count is non-zero"
    );
    std::slice::from_raw_parts(bundles, count)
        .iter()
        .map(|bundle| clone_handle::<WGPURenderBundleImpl>(*bundle, "WGPURenderBundle"))
        .collect()
}

fn map_index_format(format: native::WGPUIndexFormat) -> Option<core::IndexFormat> {
    match format {
        native::WGPUIndexFormat_Uint16 => Some(core::IndexFormat::Uint16),
        native::WGPUIndexFormat_Uint32 => Some(core::IndexFormat::Uint32),
        _ => None,
    }
}

fn get_pipeline_bind_group_layout(
    layouts: &[Arc<core::BindGroupLayout>],
    device: &Arc<core::Device>,
    instance: &Arc<WGPUInstanceImpl>,
    handles: &Mutex<Vec<Option<Arc<WGPUBindGroupLayoutImpl>>>>,
    group_index: u32,
) -> native::WGPUBindGroupLayout {
    let Ok(index) = usize::try_from(group_index) else {
        device.dispatch_error(
            core::ErrorKind::Validation,
            "pipeline bind group layout index is out of range",
        );
        return error_bind_group_layout_handle(device, instance);
    };
    let Some(layout) = layouts.get(index) else {
        device.dispatch_error(
            core::ErrorKind::Validation,
            "pipeline bind group layout index is out of range",
        );
        return error_bind_group_layout_handle(device, instance);
    };

    let mut handles = handles
        .lock()
        .expect("pipeline BGL cache lock must not poison");
    if handles.len() <= index {
        handles.resize_with(index + 1, || None);
    }
    let handle = handles[index].get_or_insert_with(|| {
        Arc::new(WGPUBindGroupLayoutImpl {
            _core: Arc::clone(layout),
            _device: Arc::clone(device),
            _instance: Arc::clone(instance),
        })
    });
    arc_to_handle(Arc::clone(handle))
}

fn error_bind_group_layout_handle(
    device: &Arc<core::Device>,
    instance: &Arc<WGPUInstanceImpl>,
) -> native::WGPUBindGroupLayout {
    arc_to_handle(Arc::new(WGPUBindGroupLayoutImpl {
        _core: Arc::new(core::BindGroupLayout::error()),
        _device: Arc::clone(device),
        _instance: Arc::clone(instance),
    }))
}

unsafe fn instance_has_timed_wait_any(descriptor: *const native::WGPUInstanceDescriptor) -> bool {
    let Some(descriptor) = descriptor.as_ref() else {
        return true;
    };
    if descriptor.requiredFeatureCount == 0 {
        return false;
    }
    if descriptor.requiredFeatures.is_null() {
        return false;
    }
    let features =
        std::slice::from_raw_parts(descriptor.requiredFeatures, descriptor.requiredFeatureCount);
    features.contains(&native::WGPUInstanceFeatureName_TimedWaitAny)
}

unsafe fn instance_backend_selection(
    descriptor: *const native::WGPUInstanceDescriptor,
) -> InstanceBackendSelection {
    let Some(descriptor) = descriptor.as_ref() else {
        return InstanceBackendSelection::Noop;
    };
    let mut chain = descriptor.nextInChain;
    while let Some(node) = chain.as_ref() {
        if node.sType == YAWGPU_STYPE_INSTANCE_BACKEND_SELECT {
            let selection =
                &*(node as *const native::WGPUChainedStruct as *const YaWGPUInstanceBackendSelect);
            return match selection.backend {
                YAWGPU_INSTANCE_BACKEND_METAL => InstanceBackendSelection::Metal,
                YAWGPU_INSTANCE_BACKEND_VULKAN => InstanceBackendSelection::Vulkan,
                _ => InstanceBackendSelection::Noop,
            };
        }
        chain = node.next;
    }
    InstanceBackendSelection::Noop
}

unsafe fn required_features_from_descriptor(
    descriptor: &native::WGPUDeviceDescriptor,
) -> Vec<core::Feature> {
    if descriptor.requiredFeatureCount == 0 {
        return Vec::new();
    }
    if descriptor.requiredFeatures.is_null() {
        return Vec::new();
    }
    let features =
        std::slice::from_raw_parts(descriptor.requiredFeatures, descriptor.requiredFeatureCount);
    features.iter().copied().map(map_feature).collect()
}

fn validate_map_async(
    buffer: &WGPUBufferImpl,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
) -> Result<(core::MapMode, u64, u64), &'static str> {
    let mode = map_map_mode(mode)?;
    let offset = u64::try_from(offset).map_err(|_| "map offset is too large")?;
    let size = if size == native::WGPU_WHOLE_MAP_SIZE {
        buffer
            .core
            .size()
            .checked_sub(offset)
            .ok_or("map offset exceeds buffer size")?
    } else {
        u64::try_from(size).map_err(|_| "map size is too large")?
    };
    Ok((mode, offset, size))
}

unsafe fn mapped_range_ptr(
    buffer: native::WGPUBuffer,
    const_access: bool,
    offset: usize,
    size: usize,
) -> Option<*mut u8> {
    let buffer = borrow_handle(buffer, "WGPUBuffer");
    let offset = u64::try_from(offset).ok()?;
    let size = if size == native::WGPU_WHOLE_MAP_SIZE {
        None
    } else {
        Some(u64::try_from(size).ok()?)
    };
    buffer.core.mapped_range(const_access, offset, size)
}

/// Installs a Rust-side uncaptured-error callback for test harnesses.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[doc(hidden)]
pub unsafe fn testing_set_uncaptured_error_callback<F>(
    device: native::WGPUDevice,
    callback: Option<F>,
) where
    F: Fn(core::DeviceError) + Send + Sync + 'static,
{
    borrow_handle(device, "WGPUDevice").set_uncaptured_error_callback(callback);
}

/// Dispatches a Rust-side device error for test harnesses.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[doc(hidden)]
pub unsafe fn testing_dispatch_device_error(
    device: native::WGPUDevice,
    kind: core::ErrorKind,
    message: impl Into<String>,
) {
    borrow_handle(device, "WGPUDevice").dispatch_error(kind, message);
}

/// Returns a bind group layout entry's visibility for validation tests.
///
/// # Safety
///
/// `layout` must be a non-null live yawgpu bind group layout handle.
#[doc(hidden)]
pub unsafe fn testing_bind_group_layout_entry_visibility(
    layout: native::WGPUBindGroupLayout,
    binding: u32,
) -> Option<u64> {
    borrow_handle(layout, "WGPUBindGroupLayout")
        ._core
        .entries()
        .iter()
        .find(|entry| entry.binding == binding)
        .map(|entry| entry.visibility)
}

/// Returns the device label for validation tests.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[doc(hidden)]
pub unsafe fn testing_get_device_label(device: native::WGPUDevice) -> String {
    borrow_handle(device, "WGPUDevice").core.label()
}

/// Returns the queue label for validation tests.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle.
#[doc(hidden)]
pub unsafe fn testing_get_queue_label(queue: native::WGPUQueue) -> String {
    borrow_handle(queue, "WGPUQueue").core.label()
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)]

    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::YAWGPU_INSTANCE_BACKEND_NOOP;

    #[derive(Default)]
    struct RequestAdapterState {
        fired: u32,
        status: native::WGPURequestAdapterStatus,
        adapter: native::WGPUAdapter,
    }

    #[derive(Default)]
    struct RequestDeviceState {
        fired: u32,
        status: native::WGPURequestDeviceStatus,
        device: native::WGPUDevice,
    }

    #[derive(Default)]
    struct PopErrorScopeState {
        fired: u32,
        status: native::WGPUPopErrorScopeStatus,
        error_type: native::WGPUErrorType,
        message: String,
    }

    #[derive(Default)]
    struct QueueWorkDoneState {
        fired: u32,
        status: native::WGPUQueueWorkDoneStatus,
        message: String,
    }

    #[derive(Default)]
    struct ComputePipelineAsyncState {
        fired: u32,
        status: native::WGPUCreatePipelineAsyncStatus,
        pipeline: native::WGPUComputePipeline,
        message: String,
    }

    #[derive(Default)]
    struct RenderPipelineAsyncState {
        fired: u32,
        status: native::WGPUCreatePipelineAsyncStatus,
        pipeline: native::WGPURenderPipeline,
        message: String,
    }

    #[derive(Default)]
    struct BufferMapAsyncState {
        fired: u32,
        status: native::WGPUMapAsyncStatus,
        message: String,
    }

    #[derive(Default)]
    struct CompilationInfoState {
        fired: u32,
        status: native::WGPUCompilationInfoRequestStatus,
        message_count: usize,
        error_messages: Vec<String>,
    }

    unsafe extern "C" fn request_adapter_callback(
        status: native::WGPURequestAdapterStatus,
        adapter: native::WGPUAdapter,
        _message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        let state = &mut *(userdata1 as *mut RequestAdapterState);
        state.fired += 1;
        state.status = status;
        state.adapter = adapter;
    }

    unsafe extern "C" fn request_device_callback(
        status: native::WGPURequestDeviceStatus,
        device: native::WGPUDevice,
        _message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        let state = &mut *(userdata1 as *mut RequestDeviceState);
        state.fired += 1;
        state.status = status;
        state.device = device;
    }

    unsafe extern "C" fn pop_error_scope_callback(
        status: native::WGPUPopErrorScopeStatus,
        error_type: native::WGPUErrorType,
        message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        let state = &mut *(userdata1 as *mut PopErrorScopeState);
        state.fired += 1;
        state.status = status;
        state.error_type = error_type;
        state.message = string_view_to_string(message);
    }

    unsafe extern "C" fn queue_work_done_callback(
        status: native::WGPUQueueWorkDoneStatus,
        message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        let state = &mut *(userdata1 as *mut QueueWorkDoneState);
        state.fired += 1;
        state.status = status;
        state.message = string_view_to_string(message);
    }

    unsafe extern "C" fn compute_pipeline_async_callback(
        status: native::WGPUCreatePipelineAsyncStatus,
        pipeline: native::WGPUComputePipeline,
        message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        let state = &mut *(userdata1 as *mut ComputePipelineAsyncState);
        state.fired += 1;
        state.status = status;
        state.pipeline = pipeline;
        state.message = string_view_to_string(message);
    }

    unsafe extern "C" fn render_pipeline_async_callback(
        status: native::WGPUCreatePipelineAsyncStatus,
        pipeline: native::WGPURenderPipeline,
        message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        let state = &mut *(userdata1 as *mut RenderPipelineAsyncState);
        state.fired += 1;
        state.status = status;
        state.pipeline = pipeline;
        state.message = string_view_to_string(message);
    }

    unsafe extern "C" fn buffer_map_async_callback(
        status: native::WGPUMapAsyncStatus,
        message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        let state = &mut *(userdata1 as *mut BufferMapAsyncState);
        state.fired += 1;
        state.status = status;
        state.message = string_view_to_string(message);
    }

    unsafe extern "C" fn compilation_info_callback(
        status: native::WGPUCompilationInfoRequestStatus,
        info: *const native::WGPUCompilationInfo,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        let state = &mut *(userdata1 as *mut CompilationInfoState);
        state.fired += 1;
        state.status = status;
        let info = info.as_ref().expect("compilation info must not be null");
        state.message_count = info.messageCount;
        for index in 0..info.messageCount {
            let message = &*info.messages.add(index);
            if message.type_ == native::WGPUCompilationMessageType_Error {
                state
                    .error_messages
                    .push(string_view_to_string(message.message));
            }
        }
    }

    unsafe fn make_noop_instance() -> native::WGPUInstance {
        let mut chain = YaWGPUInstanceBackendSelect {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
            },
            backend: YAWGPU_INSTANCE_BACKEND_NOOP,
        };
        let descriptor = native::WGPUInstanceDescriptor {
            nextInChain: (&mut chain.chain) as *mut native::WGPUChainedStruct,
            requiredFeatureCount: 0,
            requiredFeatures: std::ptr::null(),
            requiredLimits: std::ptr::null(),
        };
        wgpuCreateInstance(&descriptor)
    }

    unsafe fn request_noop_adapter(instance: native::WGPUInstance) -> native::WGPUAdapter {
        let mut state = RequestAdapterState::default();
        let callback_info = native::WGPURequestAdapterCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(request_adapter_callback),
            userdata1: (&mut state as *mut RequestAdapterState).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future = wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info);
        assert_ne!(future.id, 0);

        for _ in 0..8 {
            if state.fired != 0 {
                break;
            }
            wgpuInstanceProcessEvents(instance);
        }

        assert_eq!(state.fired, 1);
        assert_eq!(state.status, native::WGPURequestAdapterStatus_Success);
        assert!(!state.adapter.is_null());
        state.adapter
    }

    unsafe fn request_noop_device(
        instance: native::WGPUInstance,
        adapter: native::WGPUAdapter,
    ) -> native::WGPUDevice {
        let mut state = RequestDeviceState::default();
        let callback_info = native::WGPURequestDeviceCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(request_device_callback),
            userdata1: (&mut state as *mut RequestDeviceState).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future = wgpuAdapterRequestDevice(adapter, std::ptr::null(), callback_info);
        assert_ne!(future.id, 0);

        for _ in 0..8 {
            if state.fired != 0 {
                break;
            }
            wgpuInstanceProcessEvents(instance);
        }

        assert_eq!(state.fired, 1);
        assert_eq!(state.status, native::WGPURequestDeviceStatus_Success);
        assert!(!state.device.is_null());
        state.device
    }

    unsafe fn release_handles(
        instance: native::WGPUInstance,
        adapter: native::WGPUAdapter,
        device: native::WGPUDevice,
    ) {
        if !device.is_null() {
            wgpuDeviceRelease(device);
        }
        if !adapter.is_null() {
            wgpuAdapterRelease(adapter);
        }
        if !instance.is_null() {
            wgpuInstanceRelease(instance);
        }
    }

    unsafe fn noop_chain() -> (
        native::WGPUInstance,
        native::WGPUAdapter,
        native::WGPUDevice,
    ) {
        let instance = make_noop_instance();
        let adapter = request_noop_adapter(instance);
        let device = request_noop_device(instance, adapter);
        (instance, adapter, device)
    }

    #[test]
    fn WGPUDeviceImpl_set_uncaptured_error_callback_records_callback_for_dispatch() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let device_impl = clone_handle(device, "WGPUDevice");
            let counter = Arc::new(AtomicUsize::new(0));
            let callback_counter = Arc::clone(&counter);
            device_impl.set_uncaptured_error_callback(Some(move |_error| {
                callback_counter.fetch_add(1, Ordering::Relaxed);
            }));

            device_impl.dispatch_error(core::ErrorKind::Internal, "direct dispatch");

            assert_eq!(counter.load(Ordering::Relaxed), 1);
            drop(device_impl);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn WGPUDeviceImpl_dispatch_error_routes_to_uncaptured_callback() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let device_impl = clone_handle(device, "WGPUDevice");
            let counter = Arc::new(AtomicUsize::new(0));
            let callback_counter = Arc::clone(&counter);
            device_impl.set_uncaptured_error_callback(Some(move |error: core::DeviceError| {
                assert!(matches!(error.kind, core::ErrorKind::Validation));
                assert_eq!(error.message, "validation dispatch");
                callback_counter.fetch_add(1, Ordering::Relaxed);
            }));

            device_impl.dispatch_error(core::ErrorKind::Validation, "validation dispatch");

            assert_eq!(counter.load(Ordering::Relaxed), 1);
            drop(device_impl);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn testing_set_uncaptured_error_callback_installs_callback_for_dispatch() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let counter = Arc::new(AtomicUsize::new(0));
            let callback_counter = Arc::clone(&counter);
            testing_set_uncaptured_error_callback(
                device,
                Some(move |_error| {
                    callback_counter.fetch_add(1, Ordering::Relaxed);
                }),
            );

            testing_dispatch_device_error(device, core::ErrorKind::Internal, "helper dispatch");

            assert_eq!(counter.load(Ordering::Relaxed), 1);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn testing_dispatch_device_error_routes_to_uncaptured_callback() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let counter = Arc::new(AtomicUsize::new(0));
            let callback_counter = Arc::clone(&counter);
            testing_set_uncaptured_error_callback(
                device,
                Some(move |error: core::DeviceError| {
                    assert!(matches!(error.kind, core::ErrorKind::Validation));
                    assert_eq!(error.message, "helper validation");
                    callback_counter.fetch_add(1, Ordering::Relaxed);
                }),
            );

            testing_dispatch_device_error(device, core::ErrorKind::Validation, "helper validation");

            assert_eq!(counter.load(Ordering::Relaxed), 1);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn testing_bind_group_layout_entry_visibility_returns_entry_visibility_and_none() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let visibility = native::WGPUShaderStage_Vertex | native::WGPUShaderStage_Fragment;
            let entry = native::WGPUBindGroupLayoutEntry {
                nextInChain: std::ptr::null_mut(),
                binding: 7,
                visibility,
                bindingArraySize: 0,
                buffer: native::WGPUBufferBindingLayout {
                    nextInChain: std::ptr::null_mut(),
                    type_: native::WGPUBufferBindingType_Uniform,
                    hasDynamicOffset: 0,
                    minBindingSize: 16,
                },
                sampler: native::WGPUSamplerBindingLayout {
                    nextInChain: std::ptr::null_mut(),
                    type_: native::WGPUSamplerBindingType_BindingNotUsed,
                },
                texture: native::WGPUTextureBindingLayout {
                    nextInChain: std::ptr::null_mut(),
                    sampleType: native::WGPUTextureSampleType_BindingNotUsed,
                    viewDimension: native::WGPUTextureViewDimension_Undefined,
                    multisampled: 0,
                },
                storageTexture: native::WGPUStorageTextureBindingLayout {
                    nextInChain: std::ptr::null_mut(),
                    access: native::WGPUStorageTextureAccess_BindingNotUsed,
                    format: native::WGPUTextureFormat_Undefined,
                    viewDimension: native::WGPUTextureViewDimension_Undefined,
                },
            };
            let descriptor = native::WGPUBindGroupLayoutDescriptor {
                nextInChain: std::ptr::null_mut(),
                label: empty_string_view(),
                entryCount: 1,
                entries: &entry,
            };
            let layout = wgpuDeviceCreateBindGroupLayout(device, &descriptor);

            assert_eq!(
                testing_bind_group_layout_entry_visibility(layout, 7),
                Some(visibility)
            );
            assert_eq!(testing_bind_group_layout_entry_visibility(layout, 9), None);

            wgpuBindGroupLayoutRelease(layout);
            release_handles(instance, adapter, device);
        }
    }

    fn empty_string_view() -> native::WGPUStringView {
        native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        }
    }

    fn zeroed_adapter_info() -> native::WGPUAdapterInfo {
        native::WGPUAdapterInfo {
            nextInChain: std::ptr::null_mut(),
            vendor: empty_string_view(),
            architecture: empty_string_view(),
            device: empty_string_view(),
            description: empty_string_view(),
            backendType: native::WGPUBackendType_Undefined,
            adapterType: native::WGPUAdapterType_Unknown,
            vendorID: 0,
            deviceID: 0,
            subgroupMinSize: 0,
            subgroupMaxSize: 0,
        }
    }

    fn zeroed_limits() -> native::WGPULimits {
        native::WGPULimits {
            nextInChain: std::ptr::null_mut(),
            maxTextureDimension1D: 0,
            maxTextureDimension2D: 0,
            maxTextureDimension3D: 0,
            maxTextureArrayLayers: 0,
            maxBindGroups: 0,
            maxBindGroupsPlusVertexBuffers: 0,
            maxBindingsPerBindGroup: 0,
            maxDynamicUniformBuffersPerPipelineLayout: 0,
            maxDynamicStorageBuffersPerPipelineLayout: 0,
            maxSampledTexturesPerShaderStage: 0,
            maxSamplersPerShaderStage: 0,
            maxStorageBuffersPerShaderStage: 0,
            maxStorageTexturesPerShaderStage: 0,
            maxUniformBuffersPerShaderStage: 0,
            maxUniformBufferBindingSize: 0,
            maxStorageBufferBindingSize: 0,
            minUniformBufferOffsetAlignment: 0,
            minStorageBufferOffsetAlignment: 0,
            maxVertexBuffers: 0,
            maxBufferSize: 0,
            maxVertexAttributes: 0,
            maxVertexBufferArrayStride: 0,
            maxInterStageShaderVariables: 0,
            maxColorAttachments: 0,
            maxColorAttachmentBytesPerSample: 0,
            maxComputeWorkgroupStorageSize: 0,
            maxComputeInvocationsPerWorkgroup: 0,
            maxComputeWorkgroupSizeX: 0,
            maxComputeWorkgroupSizeY: 0,
            maxComputeWorkgroupSizeZ: 0,
            maxComputeWorkgroupsPerDimension: 0,
            maxImmediateSize: 0,
        }
    }

    unsafe fn string_view_to_string(value: native::WGPUStringView) -> String {
        string_view_to_str(value).unwrap_or_default().to_owned()
    }

    fn label_view(value: &str) -> native::WGPUStringView {
        native::WGPUStringView {
            data: value.as_ptr().cast(),
            length: value.len(),
        }
    }

    fn buffer_descriptor(
        usage: native::WGPUBufferUsage,
        size: u64,
    ) -> native::WGPUBufferDescriptor {
        native::WGPUBufferDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage,
            size,
            mappedAtCreation: 0,
        }
    }

    fn mapped_buffer_descriptor(
        usage: native::WGPUBufferUsage,
        size: u64,
    ) -> native::WGPUBufferDescriptor {
        native::WGPUBufferDescriptor {
            mappedAtCreation: 1,
            ..buffer_descriptor(usage, size)
        }
    }

    fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> native::WGPUExtent3D {
        native::WGPUExtent3D {
            width,
            height,
            depthOrArrayLayers: depth_or_array_layers,
        }
    }

    fn origin(x: u32, y: u32, z: u32) -> native::WGPUOrigin3D {
        native::WGPUOrigin3D { x, y, z }
    }

    fn texture_descriptor(
        usage: native::WGPUTextureUsage,
        width: u32,
    ) -> native::WGPUTextureDescriptor {
        native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage,
            dimension: native::WGPUTextureDimension_2D,
            size: extent(width, 1, 1),
            format: native::WGPUTextureFormat_RGBA8Unorm,
            mipLevelCount: 1,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        }
    }

    fn texture_descriptor_3d(
        usage: native::WGPUTextureUsage,
        size: native::WGPUExtent3D,
        mip_level_count: u32,
    ) -> native::WGPUTextureDescriptor {
        native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage,
            dimension: native::WGPUTextureDimension_3D,
            size,
            format: native::WGPUTextureFormat_RGBA8Unorm,
            mipLevelCount: mip_level_count,
            sampleCount: 1,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
        }
    }

    fn texture_view_descriptor_with_format(
        format: native::WGPUTextureFormat,
    ) -> native::WGPUTextureViewDescriptor {
        native::WGPUTextureViewDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            format,
            dimension: native::WGPUTextureViewDimension_Undefined,
            baseMipLevel: 0,
            mipLevelCount: native::WGPU_MIP_LEVEL_COUNT_UNDEFINED,
            baseArrayLayer: 0,
            arrayLayerCount: native::WGPU_ARRAY_LAYER_COUNT_UNDEFINED,
            aspect: native::WGPUTextureAspect_Undefined,
            usage: native::WGPUTextureUsage_None,
        }
    }

    fn default_sampler_descriptor() -> native::WGPUSamplerDescriptor {
        native::WGPUSamplerDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            addressModeU: native::WGPUAddressMode_Undefined,
            addressModeV: native::WGPUAddressMode_Undefined,
            addressModeW: native::WGPUAddressMode_Undefined,
            magFilter: native::WGPUFilterMode_Undefined,
            minFilter: native::WGPUFilterMode_Undefined,
            mipmapFilter: native::WGPUMipmapFilterMode_Undefined,
            lodMinClamp: 0.0,
            lodMaxClamp: 32.0,
            compare: native::WGPUCompareFunction_Undefined,
            maxAnisotropy: 1,
        }
    }

    fn query_set_descriptor(count: u32) -> native::WGPUQuerySetDescriptor {
        native::WGPUQuerySetDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            type_: native::WGPUQueryType_Occlusion,
            count,
        }
    }

    unsafe fn create_wgsl_module(
        device: native::WGPUDevice,
        source: &str,
    ) -> native::WGPUShaderModule {
        let mut wgsl = native::WGPUShaderSourceWGSL {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_ShaderSourceWGSL,
            },
            code: label_view(source),
        };
        let descriptor = native::WGPUShaderModuleDescriptor {
            nextInChain: (&mut wgsl.chain) as *mut _,
            label: empty_string_view(),
        };
        wgpuDeviceCreateShaderModule(device, &descriptor)
    }

    fn bind_group_layout_descriptor() -> native::WGPUBindGroupLayoutDescriptor {
        native::WGPUBindGroupLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            entryCount: 0,
            entries: std::ptr::null(),
        }
    }

    fn bind_group_descriptor(
        layout: native::WGPUBindGroupLayout,
    ) -> native::WGPUBindGroupDescriptor {
        native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout,
            entryCount: 0,
            entries: std::ptr::null(),
        }
    }

    fn pipeline_layout_descriptor(
        layouts: &[native::WGPUBindGroupLayout],
    ) -> native::WGPUPipelineLayoutDescriptor {
        native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            bindGroupLayoutCount: layouts.len(),
            bindGroupLayouts: layouts.as_ptr(),
            immediateSize: 0,
        }
    }

    fn compute_pipeline_descriptor(
        module: native::WGPUShaderModule,
        layout: native::WGPUPipelineLayout,
    ) -> native::WGPUComputePipelineDescriptor {
        native::WGPUComputePipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout,
            compute: native::WGPUComputeState {
                nextInChain: std::ptr::null_mut(),
                module,
                entryPoint: label_view("cs"),
                constantCount: 0,
                constants: std::ptr::null(),
            },
        }
    }

    fn render_pipeline_descriptor(
        vertex_module: native::WGPUShaderModule,
        fragment_module: native::WGPUShaderModule,
        layout: native::WGPUPipelineLayout,
    ) -> native::WGPURenderPipelineDescriptor {
        let color_target = Box::leak(Box::new(native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_RGBA8Unorm,
            blend: std::ptr::null(),
            writeMask: native::WGPUColorWriteMask_All,
        }));
        let fragment = Box::leak(Box::new(native::WGPUFragmentState {
            nextInChain: std::ptr::null_mut(),
            module: fragment_module,
            entryPoint: label_view("fs"),
            constantCount: 0,
            constants: std::ptr::null(),
            targetCount: 1,
            targets: color_target,
        }));
        native::WGPURenderPipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout,
            vertex: native::WGPUVertexState {
                nextInChain: std::ptr::null_mut(),
                module: vertex_module,
                entryPoint: label_view("vs"),
                constantCount: 0,
                constants: std::ptr::null(),
                bufferCount: 0,
                buffers: std::ptr::null(),
            },
            primitive: native::WGPUPrimitiveState {
                nextInChain: std::ptr::null_mut(),
                topology: native::WGPUPrimitiveTopology_TriangleList,
                stripIndexFormat: native::WGPUIndexFormat_Undefined,
                frontFace: native::WGPUFrontFace_Undefined,
                cullMode: native::WGPUCullMode_Undefined,
                unclippedDepth: 0,
            },
            depthStencil: std::ptr::null(),
            multisample: native::WGPUMultisampleState {
                nextInChain: std::ptr::null_mut(),
                count: 1,
                mask: 0xFFFF_FFFF,
                alphaToCoverageEnabled: 0,
            },
            fragment,
        }
    }

    fn render_bundle_encoder_descriptor(
        formats: &[native::WGPUTextureFormat],
    ) -> native::WGPURenderBundleEncoderDescriptor {
        native::WGPURenderBundleEncoderDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorFormatCount: formats.len(),
            colorFormats: formats.as_ptr(),
            depthStencilFormat: native::WGPUTextureFormat_Undefined,
            sampleCount: 1,
            depthReadOnly: 0,
            stencilReadOnly: 0,
        }
    }

    unsafe fn pop_error_scope(
        instance: native::WGPUInstance,
        device: native::WGPUDevice,
        state: &mut PopErrorScopeState,
    ) -> native::WGPUFuture {
        let callback_info = native::WGPUPopErrorScopeCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(pop_error_scope_callback),
            userdata1: (state as *mut PopErrorScopeState).cast(),
            userdata2: std::ptr::null_mut(),
        };
        let future = wgpuDevicePopErrorScope(device, callback_info);
        wgpuInstanceProcessEvents(instance);
        future
    }

    unsafe fn assert_validation_error_contains(
        instance: native::WGPUInstance,
        device: native::WGPUDevice,
        expected: &str,
    ) {
        let mut state = PopErrorScopeState::default();
        let future = pop_error_scope(instance, device, &mut state);
        assert_ne!(future.id, 0);
        assert_eq!(state.fired, 1);
        assert_eq!(state.status, native::WGPUPopErrorScopeStatus_Success);
        assert_eq!(state.error_type, native::WGPUErrorType_Validation);
        assert!(state.message.contains(expected), "{}", state.message);
    }

    unsafe fn map_buffer_async(
        buffer: native::WGPUBuffer,
        mode: native::WGPUMapMode,
        offset: usize,
        size: usize,
        state: &mut BufferMapAsyncState,
    ) -> native::WGPUFuture {
        let callback_info = native::WGPUBufferMapCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(buffer_map_async_callback),
            userdata1: (state as *mut BufferMapAsyncState).cast(),
            userdata2: std::ptr::null_mut(),
        };
        wgpuBufferMapAsync(buffer, mode, offset, size, callback_info)
    }

    unsafe fn process_events_until_buffer_map_fires(
        instance: native::WGPUInstance,
        state: &BufferMapAsyncState,
    ) {
        for _ in 0..8 {
            if state.fired != 0 {
                break;
            }
            wgpuInstanceProcessEvents(instance);
        }
    }

    fn render_pass_color_attachment(
        view: native::WGPUTextureView,
    ) -> native::WGPURenderPassColorAttachment {
        native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
        }
    }

    fn noop_render_pass_descriptor(
        attachments: &[native::WGPURenderPassColorAttachment],
        occlusion_query_set: native::WGPUQuerySet,
    ) -> native::WGPURenderPassDescriptor {
        native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: attachments.len(),
            colorAttachments: attachments.as_ptr(),
            depthStencilAttachment: std::ptr::null(),
            occlusionQuerySet: occlusion_query_set,
            timestampWrites: std::ptr::null(),
        }
    }

    unsafe fn noop_render_attachment(
        device: native::WGPUDevice,
    ) -> (native::WGPUTexture, native::WGPUTextureView) {
        let texture_desc = texture_descriptor(
            native::WGPUTextureUsage_RenderAttachment | native::WGPUTextureUsage_CopySrc,
            4,
        );
        let texture = wgpuDeviceCreateTexture(device, &texture_desc);
        let view = wgpuTextureCreateView(texture, std::ptr::null());
        (texture, view)
    }

    unsafe fn noop_compute_pipeline(device: native::WGPUDevice) -> native::WGPUComputePipeline {
        let module = create_wgsl_module(device, "@compute @workgroup_size(1) fn cs() {}");
        let layout_desc = pipeline_layout_descriptor(&[]);
        let layout = wgpuDeviceCreatePipelineLayout(device, &layout_desc);
        let pipeline_desc = compute_pipeline_descriptor(module, layout);
        let pipeline = wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
        wgpuPipelineLayoutRelease(layout);
        wgpuShaderModuleRelease(module);
        pipeline
    }

    unsafe fn noop_compute_pipeline_with_layout(
        device: native::WGPUDevice,
        bind_group_layout: native::WGPUBindGroupLayout,
    ) -> (native::WGPUPipelineLayout, native::WGPUComputePipeline) {
        let layouts = [bind_group_layout];
        let pipeline_layout_desc = pipeline_layout_descriptor(&layouts);
        let pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_desc);
        let module = create_wgsl_module(device, "@compute @workgroup_size(1) fn cs() {}");
        let pipeline_desc = compute_pipeline_descriptor(module, pipeline_layout);
        let pipeline = wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
        wgpuShaderModuleRelease(module);
        (pipeline_layout, pipeline)
    }

    unsafe fn noop_render_pipeline(device: native::WGPUDevice) -> native::WGPURenderPipeline {
        let module = create_wgsl_module(
            device,
            "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
             @fragment fn fs() -> @location(0) vec4f { return vec4f(); }",
        );
        let layout_desc = pipeline_layout_descriptor(&[]);
        let layout = wgpuDeviceCreatePipelineLayout(device, &layout_desc);
        let pipeline_desc = render_pipeline_descriptor(module, module, layout);
        let pipeline = wgpuDeviceCreateRenderPipeline(device, &pipeline_desc);
        wgpuPipelineLayoutRelease(layout);
        wgpuShaderModuleRelease(module);
        pipeline
    }

    unsafe fn noop_render_pipeline_with_layout(
        device: native::WGPUDevice,
        bind_group_layout: native::WGPUBindGroupLayout,
    ) -> (native::WGPUPipelineLayout, native::WGPURenderPipeline) {
        let layouts = [bind_group_layout];
        let pipeline_layout_desc = pipeline_layout_descriptor(&layouts);
        let pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_desc);
        let module = create_wgsl_module(
            device,
            "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
             @fragment fn fs() -> @location(0) vec4f { return vec4f(); }",
        );
        let pipeline_desc = render_pipeline_descriptor(module, module, pipeline_layout);
        let pipeline = wgpuDeviceCreateRenderPipeline(device, &pipeline_desc);
        wgpuShaderModuleRelease(module);
        (pipeline_layout, pipeline)
    }

    unsafe fn noop_bind_group(
        device: native::WGPUDevice,
    ) -> (native::WGPUBindGroupLayout, native::WGPUBindGroup) {
        let layout_desc = bind_group_layout_descriptor();
        let layout = wgpuDeviceCreateBindGroupLayout(device, &layout_desc);
        let bind_group_desc = bind_group_descriptor(layout);
        let bind_group = wgpuDeviceCreateBindGroup(device, &bind_group_desc);
        (layout, bind_group)
    }

    unsafe fn noop_indirect_buffer(device: native::WGPUDevice) -> native::WGPUBuffer {
        let desc = buffer_descriptor(
            native::WGPUBufferUsage_Indirect | native::WGPUBufferUsage_CopyDst,
            20,
        );
        wgpuDeviceCreateBuffer(device, &desc)
    }

    unsafe fn get_compilation_info(
        module: native::WGPUShaderModule,
        state: &mut CompilationInfoState,
    ) -> native::WGPUFuture {
        let callback_info = native::WGPUCompilationInfoCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(compilation_info_callback),
            userdata1: (state as *mut CompilationInfoState).cast(),
            userdata2: std::ptr::null_mut(),
        };
        wgpuShaderModuleGetCompilationInfo(module, callback_info)
    }

    unsafe fn process_events_until_compilation_info_fires(
        instance: native::WGPUInstance,
        state: &CompilationInfoState,
    ) {
        for _ in 0..8 {
            if state.fired != 0 {
                break;
            }
            wgpuInstanceProcessEvents(instance);
        }
    }

    unsafe fn create_noop_surface(instance: native::WGPUInstance) -> native::WGPUSurface {
        let mut source = native::WGPUSurfaceSourceMetalLayer {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_SurfaceSourceMetalLayer,
            },
            layer: std::ptr::dangling_mut(),
        };
        let descriptor = native::WGPUSurfaceDescriptor {
            nextInChain: (&mut source.chain) as *mut _,
            label: empty_string_view(),
        };
        wgpuInstanceCreateSurface(instance, &descriptor)
    }

    fn valid_surface_config(device: native::WGPUDevice) -> native::WGPUSurfaceConfiguration {
        native::WGPUSurfaceConfiguration {
            nextInChain: std::ptr::null_mut(),
            device,
            format: native::WGPUTextureFormat_BGRA8Unorm,
            usage: native::WGPUTextureUsage_RenderAttachment,
            width: 640,
            height: 480,
            viewFormatCount: 0,
            viewFormats: std::ptr::null(),
            alphaMode: native::WGPUCompositeAlphaMode_Opaque,
            presentMode: native::WGPUPresentMode_Fifo,
        }
    }

    fn empty_surface_capabilities() -> native::WGPUSurfaceCapabilities {
        native::WGPUSurfaceCapabilities {
            nextInChain: std::ptr::null_mut(),
            usages: native::WGPUTextureUsage_None,
            formatCount: 0,
            formats: std::ptr::null(),
            presentModeCount: 0,
            presentModes: std::ptr::null(),
            alphaModeCount: 0,
            alphaModes: std::ptr::null(),
        }
    }

    fn empty_surface_texture() -> native::WGPUSurfaceTexture {
        native::WGPUSurfaceTexture {
            nextInChain: std::ptr::null_mut(),
            texture: std::ptr::null(),
            status: native::WGPUSurfaceGetCurrentTextureStatus_Error,
        }
    }

    #[test]
    fn wgpuCreateInstance_noop_backend_and_null_descriptor_return_instances() {
        unsafe {
            let noop_instance = make_noop_instance();
            assert!(!noop_instance.is_null());
            assert!(matches!(
                borrow_handle(noop_instance, "WGPUInstance").core.hal(),
                HalInstance::Noop(_)
            ));

            let default_instance = wgpuCreateInstance(std::ptr::null());
            assert!(!default_instance.is_null());
            assert!(matches!(
                borrow_handle(default_instance, "WGPUInstance").core.hal(),
                HalInstance::Noop(_)
            ));

            wgpuInstanceRelease(default_instance);
            wgpuInstanceRelease(noop_instance);
        }
    }

    #[test]
    fn wgpuInstanceAddRef_and_wgpuInstanceRelease_balance_owned_refs() {
        unsafe {
            let instance = make_noop_instance();
            let borrowed_arc = clone_handle(instance, "WGPUInstance");
            assert_eq!(Arc::strong_count(&borrowed_arc), 2);

            wgpuInstanceAddRef(instance);
            assert_eq!(Arc::strong_count(&borrowed_arc), 3);

            wgpuInstanceRelease(instance);
            assert_eq!(Arc::strong_count(&borrowed_arc), 2);

            drop(borrowed_arc);
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpuInstanceCreateSurface_accepts_noop_metal_layer_source() {
        unsafe {
            let instance = make_noop_instance();
            let mut source = native::WGPUSurfaceSourceMetalLayer {
                chain: native::WGPUChainedStruct {
                    next: std::ptr::null_mut(),
                    sType: native::WGPUSType_SurfaceSourceMetalLayer,
                },
                layer: std::ptr::dangling_mut(),
            };
            let descriptor = native::WGPUSurfaceDescriptor {
                nextInChain: (&mut source.chain) as *mut native::WGPUChainedStruct,
                label: empty_string_view(),
            };

            let surface = wgpuInstanceCreateSurface(instance, &descriptor);
            assert!(!surface.is_null());

            wgpuSurfaceRelease(surface);
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpuInstanceCreateSurface_accepts_noop_windows_hwnd_source() {
        unsafe {
            let instance = make_noop_instance();
            let mut source = native::WGPUSurfaceSourceWindowsHWND {
                chain: native::WGPUChainedStruct {
                    next: std::ptr::null_mut(),
                    sType: native::WGPUSType_SurfaceSourceWindowsHWND,
                },
                hinstance: std::ptr::null_mut(),
                hwnd: std::ptr::dangling_mut(),
            };
            let descriptor = native::WGPUSurfaceDescriptor {
                nextInChain: (&mut source.chain) as *mut native::WGPUChainedStruct,
                label: empty_string_view(),
            };

            let surface = wgpuInstanceCreateSurface(instance, &descriptor);
            assert!(!surface.is_null());

            wgpuSurfaceRelease(surface);
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpuInstanceRequestAdapter_process_events_returns_success_adapter() {
        unsafe {
            let instance = make_noop_instance();
            let mut state = RequestAdapterState::default();
            let adapter_callback_info = native::WGPURequestAdapterCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(request_adapter_callback),
                userdata1: (&mut state as *mut RequestAdapterState).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let future =
                wgpuInstanceRequestAdapter(instance, std::ptr::null(), adapter_callback_info);
            assert_ne!(future.id, 0);
            assert_eq!(state.fired, 0);

            wgpuInstanceProcessEvents(instance);
            assert_eq!(state.fired, 1);
            assert_eq!(state.status, native::WGPURequestAdapterStatus_Success);
            assert!(!state.adapter.is_null());

            wgpuAdapterRelease(state.adapter);
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpuInstanceProcessEvents_without_registered_futures_is_noop() {
        unsafe {
            let instance = make_noop_instance();
            wgpuInstanceProcessEvents(instance);
            wgpuInstanceProcessEvents(instance);
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpuInstanceWaitAny_empty_list_returns_timed_out_and_null_list_errors() {
        unsafe {
            let instance = make_noop_instance();

            assert_eq!(
                wgpuInstanceWaitAny(instance, 0, std::ptr::null_mut(), 0),
                native::WGPUWaitStatus_TimedOut
            );
            assert_eq!(
                wgpuInstanceWaitAny(instance, 1, std::ptr::null_mut(), 0),
                native::WGPUWaitStatus_Error
            );

            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpuInstanceWaitAny_wait_any_only_request_adapter_fires_callback() {
        unsafe {
            let instance = make_noop_instance();
            let mut state = RequestAdapterState::default();
            let callback_info = native::WGPURequestAdapterCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_WaitAnyOnly,
                callback: Some(request_adapter_callback),
                userdata1: (&mut state as *mut RequestAdapterState).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let future = wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info);
            let mut wait_info = native::WGPUFutureWaitInfo {
                future,
                completed: 0,
            };

            wgpuInstanceProcessEvents(instance);
            assert_eq!(state.fired, 0);

            assert_eq!(
                wgpuInstanceWaitAny(instance, 1, &mut wait_info, 0),
                native::WGPUWaitStatus_Success
            );
            assert_eq!(wait_info.completed, 1);
            assert_eq!(state.fired, 1);
            assert_eq!(state.status, native::WGPURequestAdapterStatus_Success);
            assert!(!state.adapter.is_null());

            wgpuAdapterRelease(state.adapter);
            wgpuInstanceRelease(instance);
        }
    }

    #[test]
    fn wgpuAdapterAddRef_and_wgpuAdapterRelease_balance_owned_refs() {
        unsafe {
            let instance = make_noop_instance();
            let adapter = request_noop_adapter(instance);
            let borrowed_arc = clone_handle(adapter, "WGPUAdapter");
            assert_eq!(Arc::strong_count(&borrowed_arc), 2);

            wgpuAdapterAddRef(adapter);
            assert_eq!(Arc::strong_count(&borrowed_arc), 3);

            wgpuAdapterRelease(adapter);
            assert_eq!(Arc::strong_count(&borrowed_arc), 2);

            drop(borrowed_arc);
            release_handles(instance, adapter, std::ptr::null());
        }
    }

    #[test]
    fn wgpuAdapterGetLimits_populates_noop_defaults_and_rejects_null_out() {
        unsafe {
            let instance = make_noop_instance();
            let adapter = request_noop_adapter(instance);
            let mut limits = zeroed_limits();

            assert_eq!(
                wgpuAdapterGetLimits(adapter, &mut limits),
                native::WGPUStatus_Success
            );
            assert_eq!(
                limits.maxTextureDimension2D,
                core::Limits::DEFAULT.max_texture_dimension_2d
            );
            assert_eq!(limits.maxBindGroups, core::Limits::DEFAULT.max_bind_groups);
            assert_eq!(limits.maxBufferSize, core::Limits::DEFAULT.max_buffer_size);
            assert_eq!(
                wgpuAdapterGetLimits(adapter, std::ptr::null_mut()),
                native::WGPUStatus_Error
            );

            release_handles(instance, adapter, std::ptr::null());
        }
    }

    #[test]
    fn wgpuAdapterGetFeatures_populates_supported_features_and_free_members() {
        unsafe {
            let instance = make_noop_instance();
            let adapter = request_noop_adapter(instance);
            let mut features = native::WGPUSupportedFeatures {
                featureCount: 0,
                features: std::ptr::null(),
            };

            wgpuAdapterGetFeatures(adapter, &mut features);
            assert_eq!(features.featureCount, 5);
            let values = std::slice::from_raw_parts(features.features, features.featureCount);
            assert!(values.contains(&native::WGPUFeatureName_CoreFeaturesAndLimits));
            assert!(values.contains(&native::WGPUFeatureName_RG11B10UfloatRenderable));
            assert!(values.contains(&native::WGPUFeatureName_TimestampQuery));
            assert!(values.contains(&native::WGPUFeatureName_TextureFormatsTier1));
            assert!(values.contains(&native::WGPUFeatureName_TextureFormatsTier2));

            wgpuSupportedFeaturesFreeMembers(features);
            release_handles(instance, adapter, std::ptr::null());
        }
    }

    #[test]
    fn wgpuSupportedFeaturesFreeMembers_accepts_empty_features() {
        unsafe {
            wgpuSupportedFeaturesFreeMembers(native::WGPUSupportedFeatures {
                featureCount: 0,
                features: std::ptr::null(),
            });
        }
    }

    #[test]
    fn wgpuAdapterHasFeature_reports_supported_and_unknown_features() {
        unsafe {
            let instance = make_noop_instance();
            let adapter = request_noop_adapter(instance);

            assert_eq!(
                wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_TimestampQuery),
                1
            );
            assert_eq!(
                wgpuAdapterHasFeature(adapter, 0xFFFF_FFFFu32 as native::WGPUFeatureName),
                0
            );

            release_handles(instance, adapter, std::ptr::null());
        }
    }

    #[test]
    fn wgpuAdapterGetInfo_populates_noop_info_and_free_members() {
        unsafe {
            let instance = make_noop_instance();
            let adapter = request_noop_adapter(instance);
            let mut info = zeroed_adapter_info();

            assert_eq!(
                wgpuAdapterGetInfo(adapter, &mut info),
                native::WGPUStatus_Success
            );
            assert_eq!(string_view_to_string(info.vendor), "yawgpu");
            assert_eq!(string_view_to_string(info.architecture), "");
            assert_eq!(string_view_to_string(info.device), "yawgpu Noop Adapter");
            assert_eq!(string_view_to_string(info.description), "");
            assert_eq!(info.backendType, native::WGPUBackendType_Null);
            assert_eq!(info.adapterType, native::WGPUAdapterType_CPU);
            assert_eq!(info.vendorID, 0);
            assert_eq!(info.deviceID, 0);
            assert_eq!(
                wgpuAdapterGetInfo(adapter, std::ptr::null_mut()),
                native::WGPUStatus_Error
            );

            wgpuAdapterInfoFreeMembers(info);
            release_handles(instance, adapter, std::ptr::null());
        }
    }

    #[test]
    fn wgpuAdapterInfoFreeMembers_accepts_empty_members() {
        unsafe {
            wgpuAdapterInfoFreeMembers(zeroed_adapter_info());
        }
    }

    #[test]
    fn wgpuAdapterRequestDevice_process_events_returns_success_device() {
        unsafe {
            let instance = make_noop_instance();
            let adapter = request_noop_adapter(instance);
            let mut state = RequestDeviceState::default();
            let device_callback_info = native::WGPURequestDeviceCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(request_device_callback),
                userdata1: (&mut state as *mut RequestDeviceState).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let future = wgpuAdapterRequestDevice(adapter, std::ptr::null(), device_callback_info);
            assert_ne!(future.id, 0);
            assert_eq!(state.fired, 0);

            wgpuInstanceProcessEvents(instance);
            assert_eq!(state.fired, 1);
            assert_eq!(state.status, native::WGPURequestDeviceStatus_Success);
            assert!(!state.device.is_null());

            let queue = wgpuDeviceGetQueue(state.device);
            assert!(!queue.is_null());

            wgpuQueueRelease(queue);
            release_handles(instance, adapter, state.device);
        }
    }

    #[test]
    fn request_noop_device_helper_returns_live_device() {
        unsafe {
            let instance = make_noop_instance();
            let adapter = request_noop_adapter(instance);
            let device = request_noop_device(instance, adapter);

            assert!(!device.is_null());

            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuDeviceAddRef_and_wgpuDeviceRelease_balance_owned_refs() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let borrowed_arc = clone_handle(device, "WGPUDevice");
            assert_eq!(Arc::strong_count(&borrowed_arc), 2);

            wgpuDeviceAddRef(device);
            assert_eq!(Arc::strong_count(&borrowed_arc), 3);

            wgpuDeviceRelease(device);
            assert_eq!(Arc::strong_count(&borrowed_arc), 2);
            let queue = wgpuDeviceGetQueue(device);
            assert!(!queue.is_null());
            wgpuQueueRelease(queue);

            drop(borrowed_arc);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuDeviceDestroy_and_wgpuDeviceGetLostFuture_complete_loss() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let lost = wgpuDeviceGetLostFuture(device);
            assert_ne!(lost.id, 0);
            let mut wait_info = native::WGPUFutureWaitInfo {
                future: lost,
                completed: 0,
            };

            assert_eq!(
                wgpuInstanceWaitAny(instance, 1, &mut wait_info, 0),
                native::WGPUWaitStatus_TimedOut
            );
            wgpuDeviceDestroy(device);
            wgpuDeviceDestroy(device);
            assert_eq!(
                wgpuInstanceWaitAny(instance, 1, &mut wait_info, 0),
                native::WGPUWaitStatus_Success
            );
            assert_eq!(wait_info.completed, 1);

            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuDevicePushErrorScope_and_wgpuDevicePopErrorScope_capture_and_empty_stack() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let bad_query_descriptor = query_set_descriptor(0);
            let bad_query = wgpuDeviceCreateQuerySet(device, &bad_query_descriptor);
            assert!(!bad_query.is_null());

            assert_validation_error_contains(instance, device, "query set count");
            wgpuQuerySetRelease(bad_query);

            let mut empty_state = PopErrorScopeState::default();
            let future = pop_error_scope(instance, device, &mut empty_state);
            assert_ne!(future.id, 0);
            assert_eq!(empty_state.fired, 1);
            assert_eq!(empty_state.status, native::WGPUPopErrorScopeStatus_Error);
            assert_eq!(empty_state.error_type, native::WGPUErrorType_NoError);
            assert_eq!(empty_state.message, "No error scopes are open");

            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuDeviceSetLabel_limits_features_and_has_feature_pin_noop_device() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            wgpuDeviceSetLabel(device, label_view("device label"));
            assert_eq!(testing_get_device_label(device), "device label");

            let mut limits = zeroed_limits();
            assert_eq!(
                wgpuDeviceGetLimits(device, &mut limits),
                native::WGPUStatus_Success
            );
            assert_eq!(limits.maxBindGroups, core::Limits::DEFAULT.max_bind_groups);
            assert_eq!(limits.maxBufferSize, core::Limits::DEFAULT.max_buffer_size);
            assert_eq!(
                wgpuDeviceGetLimits(device, std::ptr::null_mut()),
                native::WGPUStatus_Error
            );

            let mut features = native::WGPUSupportedFeatures {
                featureCount: 0,
                features: std::ptr::null(),
            };
            wgpuDeviceGetFeatures(device, &mut features);
            assert_eq!(features.featureCount, 1);
            let values = std::slice::from_raw_parts(features.features, features.featureCount);
            assert_eq!(values, &[native::WGPUFeatureName_CoreFeaturesAndLimits]);
            assert_eq!(
                wgpuDeviceHasFeature(device, native::WGPUFeatureName_CoreFeaturesAndLimits),
                1
            );
            assert_eq!(
                wgpuDeviceHasFeature(device, native::WGPUFeatureName_TimestampQuery),
                0
            );
            wgpuSupportedFeaturesFreeMembers(features);

            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuDeviceCreate_resources_and_invalid_descriptors_report_errors() {
        unsafe {
            let (instance, adapter, device) = noop_chain();

            let buffer_desc = buffer_descriptor(native::WGPUBufferUsage_CopyDst, 4);
            let buffer = wgpuDeviceCreateBuffer(device, &buffer_desc);
            assert!(!buffer.is_null());
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let bad_buffer_desc = buffer_descriptor(native::WGPUBufferUsage_None, 4);
            let bad_buffer = wgpuDeviceCreateBuffer(device, &bad_buffer_desc);
            assert_validation_error_contains(instance, device, "buffer usage");

            let texture_desc = texture_descriptor(
                native::WGPUTextureUsage_CopyDst | native::WGPUTextureUsage_CopySrc,
                1,
            );
            let texture = wgpuDeviceCreateTexture(device, &texture_desc);
            assert!(!texture.is_null());
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let bad_texture_desc = texture_descriptor(native::WGPUTextureUsage_CopyDst, 0);
            let bad_texture = wgpuDeviceCreateTexture(device, &bad_texture_desc);
            assert_validation_error_contains(instance, device, "width is out of range");

            let sampler_desc = default_sampler_descriptor();
            let sampler = wgpuDeviceCreateSampler(device, &sampler_desc);
            assert!(!sampler.is_null());

            let query_desc = query_set_descriptor(4);
            let query_set = wgpuDeviceCreateQuerySet(device, &query_desc);
            assert!(!query_set.is_null());
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let bad_query_desc = query_set_descriptor(0);
            let bad_query = wgpuDeviceCreateQuerySet(device, &bad_query_desc);
            assert_validation_error_contains(instance, device, "query set count");

            let compute_module =
                create_wgsl_module(device, "@compute @workgroup_size(1) fn cs() {}");
            assert!(!compute_module.is_null());
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let bad_shader = create_wgsl_module(device, "not wgsl");
            assert_validation_error_contains(instance, device, "expected global item");

            let bgl_desc = bind_group_layout_descriptor();
            let bind_group_layout = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);
            assert!(!bind_group_layout.is_null());
            let bg_desc = bind_group_descriptor(bind_group_layout);
            let bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);
            assert!(!bind_group.is_null());
            let layouts = [bind_group_layout];
            let pipeline_layout_desc = pipeline_layout_descriptor(&layouts);
            let pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_desc);
            assert!(!pipeline_layout.is_null());

            let compute_desc = compute_pipeline_descriptor(compute_module, pipeline_layout);
            let compute_pipeline = wgpuDeviceCreateComputePipeline(device, &compute_desc);
            assert!(!compute_pipeline.is_null());

            let render_module = create_wgsl_module(
                device,
                "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
                 @fragment fn fs() -> @location(0) vec4f { return vec4f(); }",
            );
            let render_desc =
                render_pipeline_descriptor(render_module, render_module, pipeline_layout);
            let render_pipeline = wgpuDeviceCreateRenderPipeline(device, &render_desc);
            assert!(!render_pipeline.is_null());

            let command_encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            assert!(!command_encoder.is_null());
            let formats = [native::WGPUTextureFormat_RGBA8Unorm];
            let bundle_desc = render_bundle_encoder_descriptor(&formats);
            let bundle_encoder = wgpuDeviceCreateRenderBundleEncoder(device, &bundle_desc);
            assert!(!bundle_encoder.is_null());

            wgpuRenderBundleEncoderRelease(bundle_encoder);
            wgpuCommandEncoderRelease(command_encoder);
            wgpuRenderPipelineRelease(render_pipeline);
            wgpuShaderModuleRelease(render_module);
            wgpuComputePipelineRelease(compute_pipeline);
            wgpuPipelineLayoutRelease(pipeline_layout);
            wgpuBindGroupRelease(bind_group);
            wgpuBindGroupLayoutRelease(bind_group_layout);
            wgpuShaderModuleRelease(bad_shader);
            wgpuShaderModuleRelease(compute_module);
            wgpuQuerySetRelease(bad_query);
            wgpuQuerySetRelease(query_set);
            wgpuSamplerRelease(sampler);
            wgpuTextureRelease(bad_texture);
            wgpuTextureRelease(texture);
            wgpuBufferRelease(bad_buffer);
            wgpuBufferRelease(buffer);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuDeviceCreateComputePipelineAsync_and_render_async_fire_success_callbacks() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let layout_desc = pipeline_layout_descriptor(&[]);
            let pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &layout_desc);
            let compute_module =
                create_wgsl_module(device, "@compute @workgroup_size(1) fn cs() {}");
            let compute_desc = compute_pipeline_descriptor(compute_module, pipeline_layout);
            let mut compute_state = ComputePipelineAsyncState::default();
            let compute_info = native::WGPUCreateComputePipelineAsyncCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(compute_pipeline_async_callback),
                userdata1: (&mut compute_state as *mut ComputePipelineAsyncState).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let compute_future =
                wgpuDeviceCreateComputePipelineAsync(device, &compute_desc, compute_info);
            assert_ne!(compute_future.id, 0);

            let render_module = create_wgsl_module(
                device,
                "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }
                 @fragment fn fs() -> @location(0) vec4f { return vec4f(); }",
            );
            let render_desc =
                render_pipeline_descriptor(render_module, render_module, pipeline_layout);
            let mut render_state = RenderPipelineAsyncState::default();
            let render_info = native::WGPUCreateRenderPipelineAsyncCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(render_pipeline_async_callback),
                userdata1: (&mut render_state as *mut RenderPipelineAsyncState).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let render_future =
                wgpuDeviceCreateRenderPipelineAsync(device, &render_desc, render_info);
            assert_ne!(render_future.id, 0);

            wgpuInstanceProcessEvents(instance);
            assert_eq!(compute_state.fired, 1);
            assert_eq!(
                compute_state.status,
                native::WGPUCreatePipelineAsyncStatus_Success
            );
            assert!(compute_state.message.is_empty());
            assert!(!compute_state.pipeline.is_null());
            wgpuInstanceProcessEvents(instance);
            assert_eq!(render_state.fired, 1);
            assert_eq!(
                render_state.status,
                native::WGPUCreatePipelineAsyncStatus_Success
            );
            assert!(render_state.message.is_empty());
            assert!(!render_state.pipeline.is_null());

            wgpuRenderPipelineRelease(render_state.pipeline);
            wgpuComputePipelineRelease(compute_state.pipeline);
            wgpuShaderModuleRelease(render_module);
            wgpuShaderModuleRelease(compute_module);
            wgpuPipelineLayoutRelease(pipeline_layout);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuDeviceGetQueue_queue_add_ref_release_and_set_label_pin_identity() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let queue = wgpuDeviceGetQueue(device);
            let second = wgpuDeviceGetQueue(device);
            assert!(!queue.is_null());
            assert_eq!(queue, second);

            let borrowed_arc = clone_handle(queue, "WGPUQueue");
            assert_eq!(Arc::strong_count(&borrowed_arc), 4);
            wgpuQueueAddRef(queue);
            assert_eq!(Arc::strong_count(&borrowed_arc), 5);
            wgpuQueueRelease(queue);
            assert_eq!(Arc::strong_count(&borrowed_arc), 4);
            wgpuQueueSetLabel(queue, label_view("queue label"));
            assert_eq!(testing_get_queue_label(queue), "queue label");

            drop(borrowed_arc);
            wgpuQueueRelease(second);
            wgpuQueueRelease(queue);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuQueueOnSubmittedWorkDone_and_wgpuQueueSubmit_cover_empty_and_command_buffer() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let queue = wgpuDeviceGetQueue(device);
            let mut state = QueueWorkDoneState::default();
            let callback_info = native::WGPUQueueWorkDoneCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(queue_work_done_callback),
                userdata1: (&mut state as *mut QueueWorkDoneState).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let future = wgpuQueueOnSubmittedWorkDone(queue, callback_info);
            assert_ne!(future.id, 0);

            wgpuQueueSubmit(queue, 0, std::ptr::null());
            wgpuInstanceProcessEvents(instance);
            assert_eq!(state.fired, 1);
            assert_eq!(state.status, native::WGPUQueueWorkDoneStatus_Success);
            assert!(state.message.is_empty());

            let encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            wgpuQueueSubmit(queue, 1, &command_buffer);
            wgpuCommandBufferRelease(command_buffer);
            wgpuCommandEncoderRelease(encoder);
            wgpuQueueRelease(queue);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuQueueWriteBuffer_and_wgpuQueueWriteTexture_validate_happy_and_error_paths() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let queue = wgpuDeviceGetQueue(device);
            let buffer_desc = buffer_descriptor(native::WGPUBufferUsage_CopyDst, 4);
            let buffer = wgpuDeviceCreateBuffer(device, &buffer_desc);
            let bytes = [1_u8, 2, 3, 4];
            wgpuQueueWriteBuffer(queue, buffer, 0, bytes.as_ptr().cast(), bytes.len());

            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            wgpuQueueWriteBuffer(queue, buffer, 4, bytes.as_ptr().cast(), bytes.len());
            assert_validation_error_contains(
                instance,
                device,
                "queue write range exceeds buffer size",
            );

            let texture_desc = texture_descriptor(native::WGPUTextureUsage_CopyDst, 1);
            let texture = wgpuDeviceCreateTexture(device, &texture_desc);
            let destination = native::WGPUTexelCopyTextureInfo {
                texture,
                mipLevel: 0,
                origin: origin(0, 0, 0),
                aspect: native::WGPUTextureAspect_Undefined,
            };
            let layout = native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: native::WGPU_COPY_STRIDE_UNDEFINED,
                rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
            };
            let write_size = extent(1, 1, 1);
            wgpuQueueWriteTexture(
                queue,
                &destination,
                bytes.as_ptr().cast(),
                bytes.len(),
                &layout,
                &write_size,
            );

            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            wgpuQueueWriteTexture(
                queue,
                &destination,
                bytes.as_ptr().cast(),
                0,
                &layout,
                &write_size,
            );
            assert_validation_error_contains(instance, device, "dataSize is too small");

            wgpuTextureRelease(texture);
            wgpuBufferRelease(buffer);
            wgpuQueueRelease(queue);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuBuffer_destroy_unmap_release_addref_lifecycle() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let mapped_desc = mapped_buffer_descriptor(native::WGPUBufferUsage_CopyDst, 16);
            let mapped = wgpuDeviceCreateBuffer(device, &mapped_desc);
            assert!(!mapped.is_null());
            assert_eq!(
                wgpuBufferGetMapState(mapped),
                native::WGPUBufferMapState_Mapped
            );
            wgpuBufferUnmap(mapped);
            assert_eq!(
                wgpuBufferGetMapState(mapped),
                native::WGPUBufferMapState_Unmapped
            );

            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            wgpuBufferUnmap(mapped);
            let mut empty_unmap = PopErrorScopeState::default();
            pop_error_scope(instance, device, &mut empty_unmap);
            assert_eq!(empty_unmap.fired, 1);
            assert_eq!(empty_unmap.status, native::WGPUPopErrorScopeStatus_Success);
            assert_eq!(empty_unmap.error_type, native::WGPUErrorType_NoError);
            assert!(empty_unmap.message.is_empty());

            let buffer_desc = buffer_descriptor(native::WGPUBufferUsage_CopyDst, 16);
            let buffer = wgpuDeviceCreateBuffer(device, &buffer_desc);
            let borrowed_arc = clone_handle(buffer, "WGPUBuffer");
            assert_eq!(Arc::strong_count(&borrowed_arc), 2);
            wgpuBufferAddRef(buffer);
            assert_eq!(Arc::strong_count(&borrowed_arc), 3);
            wgpuBufferRelease(buffer);
            assert_eq!(Arc::strong_count(&borrowed_arc), 2);

            wgpuBufferDestroy(buffer);
            wgpuBufferDestroy(buffer);
            let queue = wgpuDeviceGetQueue(device);
            let bytes = [0_u8; 4];
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            wgpuQueueWriteBuffer(queue, buffer, 0, bytes.as_ptr().cast(), bytes.len());
            assert_validation_error_contains(instance, device, "destroyed buffer");

            drop(borrowed_arc);
            wgpuQueueRelease(queue);
            wgpuBufferRelease(buffer);
            wgpuBufferRelease(mapped);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuBuffer_map_async_and_mapped_range_walk_state_machine() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let write_desc = buffer_descriptor(native::WGPUBufferUsage_MapWrite, 16);
            let write_buffer = wgpuDeviceCreateBuffer(device, &write_desc);
            assert_eq!(
                wgpuBufferGetMapState(write_buffer),
                native::WGPUBufferMapState_Unmapped
            );

            let mut write_state = BufferMapAsyncState::default();
            let write_future = map_buffer_async(
                write_buffer,
                native::WGPUMapMode_Write,
                0,
                16,
                &mut write_state,
            );
            assert_ne!(write_future.id, 0);
            assert_eq!(
                wgpuBufferGetMapState(write_buffer),
                native::WGPUBufferMapState_Pending
            );
            process_events_until_buffer_map_fires(instance, &write_state);
            assert_eq!(write_state.fired, 1);
            assert_eq!(write_state.status, native::WGPUMapAsyncStatus_Success);
            assert!(write_state.message.is_empty());
            assert_eq!(
                wgpuBufferGetMapState(write_buffer),
                native::WGPUBufferMapState_Mapped
            );
            assert!(!wgpuBufferGetMappedRange(write_buffer, 0, 4).is_null());
            assert!(wgpuBufferGetMappedRange(write_buffer, 16, 4).is_null());
            wgpuBufferUnmap(write_buffer);
            assert_eq!(
                wgpuBufferGetMapState(write_buffer),
                native::WGPUBufferMapState_Unmapped
            );

            let read_desc = buffer_descriptor(native::WGPUBufferUsage_MapRead, 16);
            let read_buffer = wgpuDeviceCreateBuffer(device, &read_desc);
            let mut read_state = BufferMapAsyncState::default();
            let read_future = map_buffer_async(
                read_buffer,
                native::WGPUMapMode_Read,
                0,
                16,
                &mut read_state,
            );
            assert_ne!(read_future.id, 0);
            assert_eq!(
                wgpuBufferGetMapState(read_buffer),
                native::WGPUBufferMapState_Pending
            );
            process_events_until_buffer_map_fires(instance, &read_state);
            assert_eq!(read_state.fired, 1);
            assert_eq!(read_state.status, native::WGPUMapAsyncStatus_Success);
            assert!(read_state.message.is_empty());
            assert_eq!(
                wgpuBufferGetMapState(read_buffer),
                native::WGPUBufferMapState_Mapped
            );
            assert!(!wgpuBufferGetConstMappedRange(read_buffer, 0, 4).is_null());
            assert!(wgpuBufferGetConstMappedRange(read_buffer, 16, 4).is_null());
            assert!(wgpuBufferGetMappedRange(read_buffer, 0, 4).is_null());
            wgpuBufferUnmap(read_buffer);

            wgpuBufferRelease(read_buffer);
            wgpuBufferRelease(write_buffer);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuBuffer_size_and_usage_accessors_match_descriptor() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let usage = native::WGPUBufferUsage_MapRead | native::WGPUBufferUsage_CopyDst;
            let desc = buffer_descriptor(usage, 64);
            let buffer = wgpuDeviceCreateBuffer(device, &desc);

            assert_eq!(wgpuBufferGetSize(buffer), 64);
            assert_eq!(wgpuBufferGetUsage(buffer), usage);

            wgpuBufferRelease(buffer);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuTexture_accessors_match_descriptor() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let usage = native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst;
            let desc = texture_descriptor_3d(usage, extent(4, 3, 2), 2);
            let texture = wgpuDeviceCreateTexture(device, &desc);

            assert_eq!(
                wgpuTextureGetFormat(texture),
                native::WGPUTextureFormat_RGBA8Unorm
            );
            assert_eq!(
                wgpuTextureGetDimension(texture),
                native::WGPUTextureDimension_3D
            );
            assert_eq!(wgpuTextureGetWidth(texture), 4);
            assert_eq!(wgpuTextureGetHeight(texture), 3);
            assert_eq!(wgpuTextureGetDepthOrArrayLayers(texture), 2);
            assert_eq!(wgpuTextureGetMipLevelCount(texture), 2);
            assert_eq!(wgpuTextureGetSampleCount(texture), 1);
            assert_eq!(wgpuTextureGetUsage(texture), usage);

            wgpuTextureRelease(texture);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuTexture_create_view_and_destroy_release_addref() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let texture_desc = texture_descriptor(native::WGPUTextureUsage_TextureBinding, 1);
            let texture = wgpuDeviceCreateTexture(device, &texture_desc);
            let texture_arc = clone_handle(texture, "WGPUTexture");
            assert_eq!(Arc::strong_count(&texture_arc), 2);
            wgpuTextureAddRef(texture);
            assert_eq!(Arc::strong_count(&texture_arc), 3);
            wgpuTextureRelease(texture);
            assert_eq!(Arc::strong_count(&texture_arc), 2);

            let view = wgpuTextureCreateView(texture, std::ptr::null());
            assert!(!view.is_null());
            let view_arc = clone_handle(view, "WGPUTextureView");
            assert_eq!(Arc::strong_count(&view_arc), 2);
            wgpuTextureViewAddRef(view);
            assert_eq!(Arc::strong_count(&view_arc), 3);
            wgpuTextureViewRelease(view);
            assert_eq!(Arc::strong_count(&view_arc), 2);
            drop(view_arc);
            wgpuTextureViewRelease(view);

            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let bad_view_desc =
                texture_view_descriptor_with_format(native::WGPUTextureFormat_R8Unorm);
            let bad_view = wgpuTextureCreateView(texture, &bad_view_desc);
            assert!(!bad_view.is_null());
            assert_validation_error_contains(instance, device, "view format");
            wgpuTextureViewRelease(bad_view);

            wgpuTextureDestroy(texture);
            wgpuTextureDestroy(texture);
            drop(texture_arc);
            wgpuTextureRelease(texture);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuSampler_release_and_addref_lifecycle() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let sampler_desc = default_sampler_descriptor();
            let sampler = wgpuDeviceCreateSampler(device, &sampler_desc);
            let sampler_arc = clone_handle(sampler, "WGPUSampler");
            assert_eq!(Arc::strong_count(&sampler_arc), 2);
            wgpuSamplerAddRef(sampler);
            assert_eq!(Arc::strong_count(&sampler_arc), 3);
            wgpuSamplerRelease(sampler);
            assert_eq!(Arc::strong_count(&sampler_arc), 2);

            drop(sampler_arc);
            wgpuSamplerRelease(sampler);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuCommandEncoder_lifecycle_release_addref_finish() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            let encoder_arc = clone_handle(encoder, "WGPUCommandEncoder");
            assert_eq!(Arc::strong_count(&encoder_arc), 2);
            wgpuCommandEncoderAddRef(encoder);
            assert_eq!(Arc::strong_count(&encoder_arc), 3);
            wgpuCommandEncoderRelease(encoder);
            assert_eq!(Arc::strong_count(&encoder_arc), 2);

            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());
            let command_buffer_arc = clone_handle(command_buffer, "WGPUCommandBuffer");
            assert_eq!(Arc::strong_count(&command_buffer_arc), 2);
            wgpuCommandBufferAddRef(command_buffer);
            assert_eq!(Arc::strong_count(&command_buffer_arc), 3);
            wgpuCommandBufferRelease(command_buffer);
            assert_eq!(Arc::strong_count(&command_buffer_arc), 2);

            drop(command_buffer_arc);
            drop(encoder_arc);
            wgpuCommandBufferRelease(command_buffer);
            wgpuCommandEncoderRelease(encoder);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuCommandEncoder_debug_markers_insert_push_pop() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());

            wgpuCommandEncoderPushDebugGroup(encoder, label_view("encoder group"));
            wgpuCommandEncoderInsertDebugMarker(encoder, label_view("encoder marker"));
            wgpuCommandEncoderPopDebugGroup(encoder);
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());

            wgpuCommandBufferRelease(command_buffer);
            wgpuCommandEncoderRelease(encoder);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuCommandEncoder_buffer_copies_and_clear_and_write() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let source_desc = buffer_descriptor(native::WGPUBufferUsage_CopySrc, 32);
            let source = wgpuDeviceCreateBuffer(device, &source_desc);
            let destination_desc = buffer_descriptor(native::WGPUBufferUsage_CopyDst, 32);
            let destination = wgpuDeviceCreateBuffer(device, &destination_desc);
            let encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            let bytes = [0_u8; 16];

            wgpuCommandEncoderCopyBufferToBuffer(encoder, source, 0, destination, 0, 16);
            wgpuCommandEncoderClearBuffer(encoder, destination, 0, 16);
            wgpuCommandEncoderWriteBuffer(
                encoder,
                destination,
                0,
                bytes.as_ptr().cast(),
                bytes.len(),
            );
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());
            wgpuCommandBufferRelease(command_buffer);
            wgpuCommandEncoderRelease(encoder);

            let invalid_copy = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            wgpuCommandEncoderCopyBufferToBuffer(invalid_copy, source, 2, destination, 0, 4);
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let bad_copy_buffer = wgpuCommandEncoderFinish(invalid_copy, std::ptr::null());
            assert_validation_error_contains(instance, device, "copy source offset");
            wgpuCommandBufferRelease(bad_copy_buffer);
            wgpuCommandEncoderRelease(invalid_copy);

            let invalid_write = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            wgpuCommandEncoderWriteBuffer(
                invalid_write,
                destination,
                32,
                bytes.as_ptr().cast(),
                bytes.len(),
            );
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let bad_write_buffer = wgpuCommandEncoderFinish(invalid_write, std::ptr::null());
            assert_validation_error_contains(instance, device, "write buffer range");
            wgpuCommandBufferRelease(bad_write_buffer);
            wgpuCommandEncoderRelease(invalid_write);

            wgpuBufferRelease(destination);
            wgpuBufferRelease(source);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuCommandEncoder_texture_copies_walk() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let texture_desc = texture_descriptor(
                native::WGPUTextureUsage_CopySrc | native::WGPUTextureUsage_CopyDst,
                4,
            );
            let texture_a = wgpuDeviceCreateTexture(device, &texture_desc);
            let texture_b = wgpuDeviceCreateTexture(device, &texture_desc);
            let buffer_desc = buffer_descriptor(
                native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_CopyDst,
                1024,
            );
            let buffer = wgpuDeviceCreateBuffer(device, &buffer_desc);
            let layout = native::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: 256,
                rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
            };
            let buffer_info = native::WGPUTexelCopyBufferInfo { buffer, layout };
            let texture_info_a = native::WGPUTexelCopyTextureInfo {
                texture: texture_a,
                mipLevel: 0,
                origin: origin(0, 0, 0),
                aspect: native::WGPUTextureAspect_Undefined,
            };
            let texture_info_b = native::WGPUTexelCopyTextureInfo {
                texture: texture_b,
                mipLevel: 0,
                origin: origin(0, 0, 0),
                aspect: native::WGPUTextureAspect_Undefined,
            };
            let copy_size = extent(4, 4, 1);
            let encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());

            wgpuCommandEncoderCopyBufferToTexture(
                encoder,
                &buffer_info,
                &texture_info_a,
                &copy_size,
            );
            wgpuCommandEncoderCopyTextureToBuffer(
                encoder,
                &texture_info_a,
                &buffer_info,
                &copy_size,
            );
            wgpuCommandEncoderCopyTextureToTexture(
                encoder,
                &texture_info_a,
                &texture_info_b,
                &copy_size,
            );
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());

            wgpuCommandBufferRelease(command_buffer);
            wgpuCommandEncoderRelease(encoder);
            wgpuBufferRelease(buffer);
            wgpuTextureRelease(texture_b);
            wgpuTextureRelease(texture_a);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuCommandEncoder_query_and_timestamps() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let timestamp_desc = native::WGPUQuerySetDescriptor {
                type_: native::WGPUQueryType_Timestamp,
                count: 2,
                ..query_set_descriptor(2)
            };
            let timestamp_query = wgpuDeviceCreateQuerySet(device, &timestamp_desc);
            let timestamp_encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            wgpuCommandEncoderWriteTimestamp(timestamp_encoder, timestamp_query, 0);
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let timestamp_buffer = wgpuCommandEncoderFinish(timestamp_encoder, std::ptr::null());
            assert_validation_error_contains(instance, device, "timestamp");
            wgpuCommandBufferRelease(timestamp_buffer);
            wgpuCommandEncoderRelease(timestamp_encoder);

            let query_desc = query_set_descriptor(2);
            let query_set = wgpuDeviceCreateQuerySet(device, &query_desc);
            let destination_desc = buffer_descriptor(native::WGPUBufferUsage_QueryResolve, 256);
            let destination = wgpuDeviceCreateBuffer(device, &destination_desc);
            let encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            wgpuCommandEncoderResolveQuerySet(encoder, query_set, 0, 2, destination, 0);
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());

            wgpuCommandBufferRelease(command_buffer);
            wgpuCommandEncoderRelease(encoder);
            wgpuBufferRelease(destination);
            wgpuQuerySetRelease(query_set);
            wgpuQuerySetRelease(timestamp_query);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuRenderPassEncoder_lifecycle_release_addref_end_with_debug_markers() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let (texture, view) = noop_render_attachment(device);
            let attachment = render_pass_color_attachment(view);
            let attachments = [attachment];
            let descriptor = noop_render_pass_descriptor(&attachments, std::ptr::null());
            let encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            let pass = wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
            assert!(!pass.is_null());

            let pass_arc = clone_handle(pass, "WGPURenderPassEncoder");
            assert_eq!(Arc::strong_count(&pass_arc), 2);
            wgpuRenderPassEncoderAddRef(pass);
            assert_eq!(Arc::strong_count(&pass_arc), 3);
            wgpuRenderPassEncoderRelease(pass);
            assert_eq!(Arc::strong_count(&pass_arc), 2);
            wgpuRenderPassEncoderPushDebugGroup(pass, label_view("render group"));
            wgpuRenderPassEncoderInsertDebugMarker(pass, label_view("render marker"));
            wgpuRenderPassEncoderPopDebugGroup(pass);
            wgpuRenderPassEncoderEnd(pass);
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());

            drop(pass_arc);
            wgpuCommandBufferRelease(command_buffer);
            wgpuRenderPassEncoderRelease(pass);
            wgpuCommandEncoderRelease(encoder);
            wgpuTextureViewRelease(view);
            wgpuTextureRelease(texture);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuRenderPassEncoder_set_pipeline_bind_group_buffers_and_draws() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let pipeline = noop_render_pipeline(device);
            let (bind_group_layout, bind_group) = noop_bind_group(device);
            let vertex_desc = buffer_descriptor(native::WGPUBufferUsage_Vertex, 16);
            let vertex = wgpuDeviceCreateBuffer(device, &vertex_desc);
            let index_desc = buffer_descriptor(native::WGPUBufferUsage_Index, 16);
            let index = wgpuDeviceCreateBuffer(device, &index_desc);
            let indirect = noop_indirect_buffer(device);
            let (texture, view) = noop_render_attachment(device);
            let attachment = render_pass_color_attachment(view);
            let attachments = [attachment];
            let descriptor = noop_render_pass_descriptor(&attachments, std::ptr::null());
            let encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            let pass = wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);

            wgpuRenderPassEncoderSetPipeline(pass, pipeline);
            wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex, 0, 16);
            wgpuRenderPassEncoderSetIndexBuffer(pass, index, native::WGPUIndexFormat_Uint16, 0, 16);
            wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
            wgpuRenderPassEncoderDrawIndexed(pass, 3, 1, 0, 0, 0);
            wgpuRenderPassEncoderDrawIndirect(pass, indirect, 0);
            wgpuRenderPassEncoderDrawIndexedIndirect(pass, indirect, 0);
            wgpuRenderPassEncoderEnd(pass);
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());

            wgpuCommandBufferRelease(command_buffer);
            wgpuRenderPassEncoderRelease(pass);
            wgpuCommandEncoderRelease(encoder);
            wgpuTextureViewRelease(view);
            wgpuTextureRelease(texture);
            wgpuBufferRelease(indirect);
            wgpuBufferRelease(index);
            wgpuBufferRelease(vertex);
            wgpuBindGroupRelease(bind_group);
            wgpuBindGroupLayoutRelease(bind_group_layout);
            wgpuRenderPipelineRelease(pipeline);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuRenderPassEncoder_state_setters_occlusion_and_execute_bundles() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let (texture, view) = noop_render_attachment(device);
            let query_desc = query_set_descriptor(2);
            let query_set = wgpuDeviceCreateQuerySet(device, &query_desc);
            let formats = [native::WGPUTextureFormat_RGBA8Unorm];
            let bundle_desc = render_bundle_encoder_descriptor(&formats);
            let bundle_encoder = wgpuDeviceCreateRenderBundleEncoder(device, &bundle_desc);
            let bundle = wgpuRenderBundleEncoderFinish(bundle_encoder, std::ptr::null());
            let attachment = render_pass_color_attachment(view);
            let attachments = [attachment];
            let descriptor = noop_render_pass_descriptor(&attachments, query_set);
            let encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            let pass = wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);

            wgpuRenderPassEncoderSetViewport(pass, 0.0, 0.0, 4.0, 4.0, 0.0, 1.0);
            wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, 4, 4);
            let blend = native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            };
            wgpuRenderPassEncoderSetBlendConstant(pass, &blend);
            wgpuRenderPassEncoderSetStencilReference(pass, 1);
            wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
            wgpuRenderPassEncoderEndOcclusionQuery(pass);
            wgpuRenderPassEncoderExecuteBundles(pass, 1, &bundle);
            wgpuRenderPassEncoderEnd(pass);
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());

            wgpuCommandBufferRelease(command_buffer);
            wgpuRenderPassEncoderRelease(pass);
            wgpuCommandEncoderRelease(encoder);
            wgpuRenderBundleRelease(bundle);
            wgpuRenderBundleEncoderRelease(bundle_encoder);
            wgpuQuerySetRelease(query_set);
            wgpuTextureViewRelease(view);
            wgpuTextureRelease(texture);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuComputePassEncoder_lifecycle_release_addref_end_with_debug_markers() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            let pass = wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
            assert!(!pass.is_null());

            let pass_arc = clone_handle(pass, "WGPUComputePassEncoder");
            assert_eq!(Arc::strong_count(&pass_arc), 2);
            wgpuComputePassEncoderAddRef(pass);
            assert_eq!(Arc::strong_count(&pass_arc), 3);
            wgpuComputePassEncoderRelease(pass);
            assert_eq!(Arc::strong_count(&pass_arc), 2);
            wgpuComputePassEncoderPushDebugGroup(pass, label_view("compute group"));
            wgpuComputePassEncoderInsertDebugMarker(pass, label_view("compute marker"));
            wgpuComputePassEncoderPopDebugGroup(pass);
            wgpuComputePassEncoderEnd(pass);
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());

            drop(pass_arc);
            wgpuCommandBufferRelease(command_buffer);
            wgpuComputePassEncoderRelease(pass);
            wgpuCommandEncoderRelease(encoder);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuComputePassEncoder_set_pipeline_bind_group_and_dispatch() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let pipeline = noop_compute_pipeline(device);
            let (bind_group_layout, bind_group) = noop_bind_group(device);
            let indirect = noop_indirect_buffer(device);
            let encoder = wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
            let pass = wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());

            wgpuComputePassEncoderSetPipeline(pass, pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
            wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
            wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, indirect, 0);
            wgpuComputePassEncoderEnd(pass);
            let command_buffer = wgpuCommandEncoderFinish(encoder, std::ptr::null());
            assert!(!command_buffer.is_null());

            wgpuCommandBufferRelease(command_buffer);
            wgpuComputePassEncoderRelease(pass);
            wgpuCommandEncoderRelease(encoder);
            wgpuBufferRelease(indirect);
            wgpuBindGroupRelease(bind_group);
            wgpuBindGroupLayoutRelease(bind_group_layout);
            wgpuComputePipelineRelease(pipeline);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuRenderBundleEncoder_lifecycle_finish_and_debug_markers_and_release_addref() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let formats = [native::WGPUTextureFormat_RGBA8Unorm];
            let descriptor = render_bundle_encoder_descriptor(&formats);
            let encoder = wgpuDeviceCreateRenderBundleEncoder(device, &descriptor);
            let encoder_arc = clone_handle(encoder, "WGPURenderBundleEncoder");
            assert_eq!(Arc::strong_count(&encoder_arc), 2);
            wgpuRenderBundleEncoderAddRef(encoder);
            assert_eq!(Arc::strong_count(&encoder_arc), 3);
            wgpuRenderBundleEncoderRelease(encoder);
            assert_eq!(Arc::strong_count(&encoder_arc), 2);

            wgpuRenderBundleEncoderPushDebugGroup(encoder, label_view("bundle group"));
            wgpuRenderBundleEncoderInsertDebugMarker(encoder, label_view("bundle marker"));
            wgpuRenderBundleEncoderPopDebugGroup(encoder);
            let bundle = wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
            assert!(!bundle.is_null());

            let bundle_arc = clone_handle(bundle, "WGPURenderBundle");
            assert_eq!(Arc::strong_count(&bundle_arc), 2);
            wgpuRenderBundleAddRef(bundle);
            assert_eq!(Arc::strong_count(&bundle_arc), 3);
            wgpuRenderBundleRelease(bundle);
            assert_eq!(Arc::strong_count(&bundle_arc), 2);

            drop(bundle_arc);
            drop(encoder_arc);
            wgpuRenderBundleRelease(bundle);
            wgpuRenderBundleEncoderRelease(encoder);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuRenderBundleEncoder_set_pipeline_bind_group_buffers_and_draws() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let pipeline = noop_render_pipeline(device);
            let (bind_group_layout, bind_group) = noop_bind_group(device);
            let vertex_desc = buffer_descriptor(native::WGPUBufferUsage_Vertex, 16);
            let vertex = wgpuDeviceCreateBuffer(device, &vertex_desc);
            let index_desc = buffer_descriptor(native::WGPUBufferUsage_Index, 16);
            let index = wgpuDeviceCreateBuffer(device, &index_desc);
            let indirect = noop_indirect_buffer(device);
            let formats = [native::WGPUTextureFormat_RGBA8Unorm];
            let descriptor = render_bundle_encoder_descriptor(&formats);
            let encoder = wgpuDeviceCreateRenderBundleEncoder(device, &descriptor);

            wgpuRenderBundleEncoderSetPipeline(encoder, pipeline);
            wgpuRenderBundleEncoderSetBindGroup(encoder, 0, bind_group, 0, std::ptr::null());
            wgpuRenderBundleEncoderSetVertexBuffer(encoder, 0, vertex, 0, 16);
            wgpuRenderBundleEncoderSetIndexBuffer(
                encoder,
                index,
                native::WGPUIndexFormat_Uint16,
                0,
                16,
            );
            wgpuRenderBundleEncoderDraw(encoder, 3, 1, 0, 0);
            wgpuRenderBundleEncoderDrawIndexed(encoder, 3, 1, 0, 0, 0);
            wgpuRenderBundleEncoderDrawIndirect(encoder, indirect, 0);
            wgpuRenderBundleEncoderDrawIndexedIndirect(encoder, indirect, 0);
            let bundle = wgpuRenderBundleEncoderFinish(encoder, std::ptr::null());
            assert!(!bundle.is_null());

            wgpuRenderBundleRelease(bundle);
            wgpuRenderBundleEncoderRelease(encoder);
            wgpuBufferRelease(indirect);
            wgpuBufferRelease(index);
            wgpuBufferRelease(vertex);
            wgpuBindGroupRelease(bind_group);
            wgpuBindGroupLayoutRelease(bind_group_layout);
            wgpuRenderPipelineRelease(pipeline);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuComputePipeline_get_bind_group_layout_release_addref() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let bgl_desc = bind_group_layout_descriptor();
            let bind_group_layout = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);
            let (pipeline_layout, pipeline) =
                noop_compute_pipeline_with_layout(device, bind_group_layout);
            let pipeline_arc = clone_handle(pipeline, "WGPUComputePipeline");
            let pipeline_count = Arc::strong_count(&pipeline_arc);
            wgpuComputePipelineAddRef(pipeline);
            assert_eq!(Arc::strong_count(&pipeline_arc), pipeline_count + 1);
            wgpuComputePipelineRelease(pipeline);
            assert_eq!(Arc::strong_count(&pipeline_arc), pipeline_count);

            let layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
            assert!(!layout.is_null());
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let bad_layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 1);
            assert_validation_error_contains(instance, device, "bind group layout index");

            wgpuBindGroupLayoutRelease(bad_layout);
            wgpuBindGroupLayoutRelease(layout);
            drop(pipeline_arc);
            wgpuComputePipelineRelease(pipeline);
            wgpuPipelineLayoutRelease(pipeline_layout);
            wgpuBindGroupLayoutRelease(bind_group_layout);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuRenderPipeline_get_bind_group_layout_release_addref() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let bgl_desc = bind_group_layout_descriptor();
            let bind_group_layout = wgpuDeviceCreateBindGroupLayout(device, &bgl_desc);
            let (pipeline_layout, pipeline) =
                noop_render_pipeline_with_layout(device, bind_group_layout);
            let pipeline_arc = clone_handle(pipeline, "WGPURenderPipeline");
            let pipeline_count = Arc::strong_count(&pipeline_arc);
            wgpuRenderPipelineAddRef(pipeline);
            assert_eq!(Arc::strong_count(&pipeline_arc), pipeline_count + 1);
            wgpuRenderPipelineRelease(pipeline);
            assert_eq!(Arc::strong_count(&pipeline_arc), pipeline_count);

            let layout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
            assert!(!layout.is_null());
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let bad_layout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 1);
            assert_validation_error_contains(instance, device, "bind group layout index");

            wgpuBindGroupLayoutRelease(bad_layout);
            wgpuBindGroupLayoutRelease(layout);
            drop(pipeline_arc);
            wgpuRenderPipelineRelease(pipeline);
            wgpuPipelineLayoutRelease(pipeline_layout);
            wgpuBindGroupLayoutRelease(bind_group_layout);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuBindGroupLayout_and_BindGroup_and_PipelineLayout_release_addref() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let (bind_group_layout, bind_group) = noop_bind_group(device);
            let layouts = [bind_group_layout];
            let pipeline_layout_desc = pipeline_layout_descriptor(&layouts);
            let pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_desc);

            let bgl_arc = clone_handle(bind_group_layout, "WGPUBindGroupLayout");
            let bgl_count = Arc::strong_count(&bgl_arc);
            wgpuBindGroupLayoutAddRef(bind_group_layout);
            assert_eq!(Arc::strong_count(&bgl_arc), bgl_count + 1);
            wgpuBindGroupLayoutRelease(bind_group_layout);
            assert_eq!(Arc::strong_count(&bgl_arc), bgl_count);

            let bg_arc = clone_handle(bind_group, "WGPUBindGroup");
            let bg_count = Arc::strong_count(&bg_arc);
            wgpuBindGroupAddRef(bind_group);
            assert_eq!(Arc::strong_count(&bg_arc), bg_count + 1);
            wgpuBindGroupRelease(bind_group);
            assert_eq!(Arc::strong_count(&bg_arc), bg_count);

            let pl_arc = clone_handle(pipeline_layout, "WGPUPipelineLayout");
            let pl_count = Arc::strong_count(&pl_arc);
            wgpuPipelineLayoutAddRef(pipeline_layout);
            assert_eq!(Arc::strong_count(&pl_arc), pl_count + 1);
            wgpuPipelineLayoutRelease(pipeline_layout);
            assert_eq!(Arc::strong_count(&pl_arc), pl_count);

            drop(pl_arc);
            drop(bg_arc);
            drop(bgl_arc);
            wgpuPipelineLayoutRelease(pipeline_layout);
            wgpuBindGroupRelease(bind_group);
            wgpuBindGroupLayoutRelease(bind_group_layout);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuShaderModule_get_compilation_info_and_release_addref() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let module = create_wgsl_module(device, "@compute @workgroup_size(1) fn cs() {}");
            let module_arc = clone_handle(module, "WGPUShaderModule");
            let module_count = Arc::strong_count(&module_arc);
            wgpuShaderModuleAddRef(module);
            assert_eq!(Arc::strong_count(&module_arc), module_count + 1);
            wgpuShaderModuleRelease(module);
            assert_eq!(Arc::strong_count(&module_arc), module_count);

            let mut valid_state = CompilationInfoState::default();
            let future = get_compilation_info(module, &mut valid_state);
            assert_ne!(future.id, 0);
            process_events_until_compilation_info_fires(instance, &valid_state);
            assert_eq!(valid_state.fired, 1);
            assert_eq!(
                valid_state.status,
                native::WGPUCompilationInfoRequestStatus_Success
            );
            assert_eq!(valid_state.message_count, 0);
            assert!(valid_state.error_messages.is_empty());

            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            let invalid = create_wgsl_module(device, "not wgsl");
            assert_validation_error_contains(instance, device, "expected global item");
            let mut invalid_state = CompilationInfoState::default();
            let invalid_future = get_compilation_info(invalid, &mut invalid_state);
            assert_ne!(invalid_future.id, 0);
            process_events_until_compilation_info_fires(instance, &invalid_state);
            assert_eq!(invalid_state.fired, 1);
            assert_eq!(
                invalid_state.status,
                native::WGPUCompilationInfoRequestStatus_Success
            );
            assert_eq!(invalid_state.message_count, 1);
            assert_eq!(invalid_state.error_messages.len(), 1);
            assert!(!invalid_state.error_messages[0].is_empty());

            wgpuShaderModuleRelease(invalid);
            drop(module_arc);
            wgpuShaderModuleRelease(module);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuQuerySet_accessors_lifecycle_pin_type_count_label_destroy_release_addref() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let descriptor = query_set_descriptor(4);
            let query_set = wgpuDeviceCreateQuerySet(device, &descriptor);
            let query_arc = clone_handle(query_set, "WGPUQuerySet");
            assert_eq!(Arc::strong_count(&query_arc), 2);
            wgpuQuerySetAddRef(query_set);
            assert_eq!(Arc::strong_count(&query_arc), 3);
            wgpuQuerySetRelease(query_set);
            assert_eq!(Arc::strong_count(&query_arc), 2);

            assert_eq!(
                wgpuQuerySetGetType(query_set),
                native::WGPUQueryType_Occlusion
            );
            assert_eq!(wgpuQuerySetGetCount(query_set), 4);
            wgpuQuerySetSetLabel(query_set, label_view("query label"));
            wgpuQuerySetDestroy(query_set);
            wgpuQuerySetDestroy(query_set);

            drop(query_arc);
            wgpuQuerySetRelease(query_set);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuSurface_get_capabilities_capabilities_free_members_and_lifecycle() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let surface = create_noop_surface(instance);
            let surface_arc = clone_handle(surface, "WGPUSurface");
            assert_eq!(Arc::strong_count(&surface_arc), 2);
            wgpuSurfaceAddRef(surface);
            assert_eq!(Arc::strong_count(&surface_arc), 3);
            wgpuSurfaceRelease(surface);
            assert_eq!(Arc::strong_count(&surface_arc), 2);
            wgpuSurfaceSetLabel(surface, label_view("surface label"));
            assert_eq!(
                borrow_handle(surface, "WGPUSurface")
                    .label
                    .lock()
                    .expect("surface label lock is not poisoned")
                    .as_str(),
                "surface label"
            );

            let mut capabilities = empty_surface_capabilities();
            assert_eq!(
                wgpuSurfaceGetCapabilities(surface, adapter, &mut capabilities),
                native::WGPUStatus_Success
            );
            assert_eq!(
                capabilities.usages,
                native::WGPUTextureUsage_RenderAttachment
            );
            let formats =
                std::slice::from_raw_parts(capabilities.formats, capabilities.formatCount);
            assert_eq!(
                formats,
                &[
                    native::WGPUTextureFormat_BGRA8Unorm,
                    native::WGPUTextureFormat_RGBA8Unorm
                ]
            );
            let present_modes = std::slice::from_raw_parts(
                capabilities.presentModes,
                capabilities.presentModeCount,
            );
            assert_eq!(present_modes, &[native::WGPUPresentMode_Fifo]);
            let alpha_modes =
                std::slice::from_raw_parts(capabilities.alphaModes, capabilities.alphaModeCount);
            assert_eq!(alpha_modes, &[native::WGPUCompositeAlphaMode_Opaque]);
            wgpuSurfaceCapabilitiesFreeMembers(capabilities);

            drop(surface_arc);
            wgpuSurfaceRelease(surface);
            release_handles(instance, adapter, device);
        }
    }

    #[test]
    fn wgpuSurface_configure_unconfigure_get_current_texture_present_noop_contract() {
        unsafe {
            let (instance, adapter, device) = noop_chain();
            let surface = create_noop_surface(instance);
            let config = valid_surface_config(device);
            wgpuSurfaceConfigure(surface, &config);

            let mut surface_texture = empty_surface_texture();
            wgpuSurfaceGetCurrentTexture(surface, &mut surface_texture);
            assert_eq!(
                surface_texture.status,
                native::WGPUSurfaceGetCurrentTextureStatus_Lost
            );
            assert!(surface_texture.texture.is_null());
            assert_eq!(wgpuSurfacePresent(surface), native::WGPUStatus_Success);

            wgpuSurfaceUnconfigure(surface);
            wgpuSurfaceUnconfigure(surface);
            let mut unconfigured_texture = empty_surface_texture();
            wgpuSurfaceGetCurrentTexture(surface, &mut unconfigured_texture);
            assert_eq!(
                unconfigured_texture.status,
                native::WGPUSurfaceGetCurrentTextureStatus_Error
            );
            assert!(unconfigured_texture.texture.is_null());

            let mut bad_config = valid_surface_config(device);
            bad_config.width = 0;
            wgpuDevicePushErrorScope(device, native::WGPUErrorFilter_Validation);
            wgpuSurfaceConfigure(surface, &bad_config);
            assert_validation_error_contains(instance, device, "size must be non-zero");

            wgpuSurfaceRelease(surface);
            release_handles(instance, adapter, device);
        }
    }
}
