pub mod conv;

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
    map_texture_usage_to_native, map_texture_view_descriptor, release_handle, string_view,
    string_view_to_str, DeviceLostCallbackInfo,
};

pub const WGPU_YAWGPU_INSTANCE_BACKEND_NOOP: u32 = 0;
pub const WGPU_YAWGPU_INSTANCE_BACKEND_METAL: u32 = 1;
pub const WGPU_YAWGPU_INSTANCE_BACKEND_VULKAN: u32 = 2;
pub const WGPU_STYPE_YAWGPU_INSTANCE_BACKEND_SELECT: native::WGPUSType = 0x7000_0001;

/// yawgpu vendor extension for selecting a backend at instance creation.
///
/// Chain this from `WGPUInstanceDescriptor::nextInChain` with
/// `WGPU_STYPE_YAWGPU_INSTANCE_BACKEND_SELECT`. This is intentionally outside
/// webgpu.h and mirrors native-only backend selection extensions.
#[repr(C)]
pub struct WGPUYawgpuInstanceBackendSelect {
    pub chain: native::WGPUChainedStruct,
    pub backend: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstanceBackendSelection {
    Noop,
    Metal,
    Vulkan,
}

pub struct WGPUAdapterImpl {
    core: Arc<core::Adapter>,
    instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUBufferImpl {
    core: Arc<core::Buffer>,
    device: Arc<core::Device>,
    instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUBindGroupLayoutImpl {
    _core: Arc<core::BindGroupLayout>,
    _device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUBindGroupImpl {
    _core: Arc<core::BindGroup>,
    _layout: Arc<core::BindGroupLayout>,
    _device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUDeviceImpl {
    core: Arc<core::Device>,
    instance: Arc<WGPUInstanceImpl>,
    device_lost_callback: DeviceLostCallbackInfo,
    device_lost_futures: Mutex<Vec<u64>>,
    default_queue: Mutex<Option<Arc<WGPUQueueImpl>>>,
    shader_module_cache: Mutex<HashMap<ShaderModuleCacheKey, Arc<WGPUShaderModuleImpl>>>,
    pipeline_layout_cache: Mutex<HashMap<PipelineLayoutCacheKey, Arc<WGPUPipelineLayoutImpl>>>,
    compute_pipeline_cache: Mutex<HashMap<ComputePipelineCacheKey, Arc<WGPUComputePipelineImpl>>>,
    render_pipeline_cache: Mutex<HashMap<RenderPipelineCacheKey, Arc<WGPURenderPipelineImpl>>>,
}

pub struct WGPUInstanceImpl {
    core: Arc<core::Instance>,
    timed_wait_any_enabled: bool,
    pending_callbacks: Mutex<BTreeMap<u64, PendingCallback>>,
}

pub struct WGPUQueueImpl {
    core: core::Queue,
    device: Arc<core::Device>,
    instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUTextureImpl {
    core: Arc<core::Texture>,
    device: Arc<core::Device>,
    instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUTextureViewImpl {
    _core: Arc<core::TextureView>,
    _texture: Arc<core::Texture>,
    _device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUSamplerImpl {
    _core: Arc<core::Sampler>,
    _device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUShaderModuleImpl {
    _core: Arc<core::ShaderModule>,
    _device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUPipelineLayoutImpl {
    _core: Arc<core::PipelineLayout>,
    _device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUComputePipelineImpl {
    _core: Arc<core::ComputePipeline>,
    _device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
    bind_group_layout_handles: Mutex<Vec<Option<Arc<WGPUBindGroupLayoutImpl>>>>,
}

pub struct WGPURenderPipelineImpl {
    _core: Arc<core::RenderPipeline>,
    _device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
    bind_group_layout_handles: Mutex<Vec<Option<Arc<WGPUBindGroupLayoutImpl>>>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ShaderModuleCacheKey {
    Wgsl(String),
    Spirv(Vec<u32>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineLayoutCacheKey {
    bind_group_layouts: Vec<usize>,
    immediate_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PipelineLayoutIdentity {
    Auto,
    Explicit(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineConstantCacheKey {
    key: String,
    value_bits: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ComputePipelineCacheKey {
    module: usize,
    entry_point: Option<String>,
    constants: Vec<PipelineConstantCacheKey>,
    layout: PipelineLayoutIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RenderPipelineCacheKey {
    layout: PipelineLayoutIdentity,
    vertex: RenderStageCacheKey,
    vertex_buffers: Vec<VertexBufferLayoutCacheKey>,
    primitive: PrimitiveStateCacheKey,
    depth_stencil: Option<DepthStencilStateCacheKey>,
    multisample: MultisampleStateCacheKey,
    fragment: Option<FragmentStateCacheKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RenderStageCacheKey {
    module: usize,
    entry_point: Option<String>,
    constants: Vec<PipelineConstantCacheKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FragmentStateCacheKey {
    stage: RenderStageCacheKey,
    target_count: usize,
    targets: Vec<ColorTargetCacheKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ColorTargetCacheKey {
    format: native::WGPUTextureFormat,
    blend: Option<BlendStateCacheKey>,
    write_mask: native::WGPUColorWriteMask,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BlendStateCacheKey {
    color: BlendComponentCacheKey,
    alpha: BlendComponentCacheKey,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BlendComponentCacheKey {
    operation: native::WGPUBlendOperation,
    src_factor: native::WGPUBlendFactor,
    dst_factor: native::WGPUBlendFactor,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VertexBufferLayoutCacheKey {
    step_mode: native::WGPUVertexStepMode,
    array_stride: u64,
    attributes: Vec<VertexAttributeCacheKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct VertexAttributeCacheKey {
    format: native::WGPUVertexFormat,
    offset: u64,
    shader_location: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PrimitiveStateCacheKey {
    topology: native::WGPUPrimitiveTopology,
    strip_index_format: native::WGPUIndexFormat,
    front_face: native::WGPUFrontFace,
    cull_mode: native::WGPUCullMode,
    unclipped_depth: native::WGPUBool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DepthStencilStateCacheKey {
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StencilFaceStateCacheKey {
    compare: native::WGPUCompareFunction,
    fail_op: native::WGPUStencilOperation,
    depth_fail_op: native::WGPUStencilOperation,
    pass_op: native::WGPUStencilOperation,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct MultisampleStateCacheKey {
    count: u32,
    mask: u32,
    alpha_to_coverage_enabled: native::WGPUBool,
}

pub struct WGPUSurfaceImpl {
    label: Mutex<String>,
    configured: Mutex<Option<SurfaceConfigurationState>>,
    is_error: bool,
    _instance: Arc<WGPUInstanceImpl>,
}

#[derive(Debug, Clone, Copy)]
struct SurfaceConfigurationState {
    _format: native::WGPUTextureFormat,
    _usage: native::WGPUTextureUsage,
    _width: u32,
    _height: u32,
    _present_mode: native::WGPUPresentMode,
    _alpha_mode: native::WGPUCompositeAlphaMode,
}

pub struct WGPUQuerySetImpl {
    core: Arc<core::QuerySet>,
    _device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUCommandEncoderImpl {
    core: Arc<core::CommandEncoder>,
    device: Arc<core::Device>,
    instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUCommandBufferImpl {
    core: Arc<core::CommandBuffer>,
    _device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPURenderPassEncoderImpl {
    core: Arc<core::RenderPassEncoder>,
    device: Arc<core::Device>,
    _parent: Arc<core::CommandEncoder>,
    _instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPUComputePassEncoderImpl {
    core: Arc<core::ComputePassEncoder>,
    device: Arc<core::Device>,
    _parent: Arc<core::CommandEncoder>,
    _instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPURenderBundleEncoderImpl {
    core: Arc<core::RenderBundleEncoder>,
    device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
}

pub struct WGPURenderBundleImpl {
    core: Arc<core::RenderBundle>,
    _device: Arc<core::Device>,
    _instance: Arc<WGPUInstanceImpl>,
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

    #[doc(hidden)]
    pub fn set_uncaptured_error_callback<F>(&self, callback: Option<F>)
    where
        F: Fn(core::DeviceError) + Send + Sync + 'static,
    {
        self.core.set_uncaptured_error_callback(callback);
    }

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

enum PendingCallback {
    RequestAdapter {
        mode: native::WGPUCallbackMode,
        callback: native::WGPURequestAdapterCallback,
        adapter: Arc<WGPUAdapterImpl>,
        userdata1: usize,
        userdata2: usize,
    },
    RequestDevice {
        mode: native::WGPUCallbackMode,
        callback: native::WGPURequestDeviceCallback,
        result: Result<Arc<WGPUDeviceImpl>, String>,
        userdata1: usize,
        userdata2: usize,
    },
    DeviceLost {
        mode: native::WGPUCallbackMode,
        callback: native::WGPUDeviceLostCallback,
        device: usize,
        reason: core::DeviceLostReason,
        userdata1: usize,
        userdata2: usize,
    },
    BufferMap {
        mode: native::WGPUCallbackMode,
        callback: native::WGPUBufferMapCallback,
        device: Arc<core::Device>,
        buffer: Option<core::Buffer>,
        status: core::MapAsyncStatus,
        userdata1: usize,
        userdata2: usize,
    },
    QueueWorkDone {
        mode: native::WGPUCallbackMode,
        callback: native::WGPUQueueWorkDoneCallback,
        device: Arc<core::Device>,
        status: core::QueueWorkDoneStatus,
        userdata1: usize,
        userdata2: usize,
    },
    CompilationInfo {
        mode: native::WGPUCallbackMode,
        callback: native::WGPUCompilationInfoCallback,
        shader_module: Arc<core::ShaderModule>,
        userdata1: usize,
        userdata2: usize,
    },
    CreateComputePipelineAsync {
        mode: native::WGPUCallbackMode,
        callback: native::WGPUCreateComputePipelineAsyncCallback,
        pipeline: Arc<WGPUComputePipelineImpl>,
        userdata1: usize,
        userdata2: usize,
    },
    CreateRenderPipelineAsync {
        mode: native::WGPUCallbackMode,
        callback: native::WGPUCreateRenderPipelineAsyncCallback,
        pipeline: Arc<WGPURenderPipelineImpl>,
        userdata1: usize,
        userdata2: usize,
    },
    PopErrorScope {
        mode: native::WGPUCallbackMode,
        callback: native::WGPUPopErrorScopeCallback,
        status: native::WGPUPopErrorScopeStatus,
        error: Option<core::DeviceError>,
        message: String,
        userdata1: usize,
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

pub mod native {
    #![allow(
        dead_code,
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        improper_ctypes
    )]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// Creates a new WebGPU instance.
///
/// # Safety
///
/// `descriptor`, when non-null, must point to a valid `WGPUInstanceDescriptor`.
#[no_mangle]
pub unsafe extern "C" fn wgpuCreateInstance(
    descriptor: *const native::WGPUInstanceDescriptor,
) -> native::WGPUInstance {
    let timed_wait_any_enabled = instance_has_timed_wait_any(descriptor);
    let instance = match instance_backend_selection(descriptor) {
        InstanceBackendSelection::Noop => WGPUInstanceImpl::new_noop(timed_wait_any_enabled),
        InstanceBackendSelection::Metal => {
            #[cfg(feature = "metal")]
            {
                match yawgpu_hal::metal::MetalInstance::new() {
                    Ok(instance) => {
                        let hal_instance = yawgpu_hal::HalInstance::Metal(instance);
                        if hal_instance.enumerate_adapters().is_empty() {
                            WGPUInstanceImpl::new_noop(timed_wait_any_enabled)
                        } else {
                            WGPUInstanceImpl::from_core(
                                core::Instance::from_hal(hal_instance),
                                timed_wait_any_enabled,
                            )
                        }
                    }
                    Err(_) => WGPUInstanceImpl::new_noop(timed_wait_any_enabled),
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                WGPUInstanceImpl::new_noop(timed_wait_any_enabled)
            }
        }
        InstanceBackendSelection::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                match yawgpu_hal::vulkan::VulkanInstance::new() {
                    Ok(instance) => {
                        let hal_instance = yawgpu_hal::HalInstance::Vulkan(instance);
                        if hal_instance.enumerate_adapters().is_empty() {
                            WGPUInstanceImpl::new_noop(timed_wait_any_enabled)
                        } else {
                            WGPUInstanceImpl::from_core(
                                core::Instance::from_hal(hal_instance),
                                timed_wait_any_enabled,
                            )
                        }
                    }
                    Err(_) => WGPUInstanceImpl::new_noop(timed_wait_any_enabled),
                }
            }
            #[cfg(not(feature = "vulkan"))]
            {
                WGPUInstanceImpl::new_noop(timed_wait_any_enabled)
            }
        }
    };
    arc_to_handle(instance)
}

/// Releases one owned reference to an instance handle.
///
/// # Safety
///
/// `instance` must be a non-null handle previously returned by yawgpu and not
/// already fully released.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceRelease(instance: native::WGPUInstance) {
    release_handle(instance, "WGPUInstance");
}

/// Adds one owned reference to an instance handle.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceAddRef(instance: native::WGPUInstance) {
    add_ref_handle(instance, "WGPUInstance");
}

/// Creates a synthetic Noop surface from a recognized surface-source chain.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle. `descriptor`,
/// when non-null, must point to a valid `WGPUSurfaceDescriptor`.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceCreateSurface(
    instance: native::WGPUInstance,
    descriptor: *const native::WGPUSurfaceDescriptor,
) -> native::WGPUSurface {
    let instance = clone_handle(instance, "WGPUInstance");
    let (label, is_error) = if let Some(descriptor) = descriptor.as_ref() {
        (
            label_from_string_view(descriptor.label).unwrap_or_default(),
            !has_supported_surface_source(descriptor.nextInChain),
        )
    } else {
        (String::new(), true)
    };
    arc_to_handle(Arc::new(WGPUSurfaceImpl {
        label: Mutex::new(label),
        configured: Mutex::new(None),
        is_error,
        _instance: instance,
    }))
}

/// Requests a Noop adapter from an instance.
///
/// # Safety
///
/// `instance_handle` must be a non-null live yawgpu instance handle. `options`,
/// when non-null, must point to a valid `WGPURequestAdapterOptions`.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceRequestAdapter(
    instance_handle: native::WGPUInstance,
    options: *const native::WGPURequestAdapterOptions,
    callback_info: native::WGPURequestAdapterCallbackInfo,
) -> native::WGPUFuture {
    let instance = borrow_handle(instance_handle, "WGPUInstance");
    let feature_level = options
        .as_ref()
        .map(|options| map_feature_level(options.featureLevel))
        .unwrap_or(core::FeatureLevel::Core);
    let adapter = instance
        .core
        .enumerate_adapters_with_feature_level(feature_level)
        .into_iter()
        .next()
        .expect("Noop instance must expose an adapter");
    let adapter = Arc::new(WGPUAdapterImpl {
        core: Arc::new(adapter),
        instance: clone_handle(instance_handle, "WGPUInstance"),
    });

    instance.register_callback(PendingCallback::RequestAdapter {
        mode: callback_info.mode,
        callback: callback_info.callback,
        adapter,
        userdata1: callback_info.userdata1 as usize,
        userdata2: callback_info.userdata2 as usize,
    })
}

/// Releases one owned reference to an adapter handle.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterRelease(adapter: native::WGPUAdapter) {
    release_handle(adapter, "WGPUAdapter");
}

/// Adds one owned reference to an adapter handle.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterAddRef(adapter: native::WGPUAdapter) {
    add_ref_handle(adapter, "WGPUAdapter");
}

/// Gets the supported limits for an adapter.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle. `limits` must
/// point to writable `WGPULimits` storage.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterGetLimits(
    adapter: native::WGPUAdapter,
    limits: *mut native::WGPULimits,
) -> native::WGPUStatus {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    let Some(limits) = limits.as_mut() else {
        return native::WGPUStatus_Error;
    };
    *limits = map_limits_to_native(adapter.core.limits());
    native::WGPUStatus_Success
}

/// Gets the supported features for an adapter.
///
/// The returned `features` array is allocated by yawgpu and must be released
/// with `wgpuSupportedFeaturesFreeMembers`.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle. `features` must
/// point to writable `WGPUSupportedFeatures` storage.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterGetFeatures(
    adapter: native::WGPUAdapter,
    features: *mut native::WGPUSupportedFeatures,
) {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    let features = features
        .as_mut()
        .expect("WGPUSupportedFeatures must not be null");
    *features = map_features_to_native(&adapter.core.features());
}

/// Returns whether the adapter supports `feature`.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterHasFeature(
    adapter: native::WGPUAdapter,
    feature: native::WGPUFeatureName,
) -> native::WGPUBool {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    native::WGPUBool::from(adapter.core.has_feature(map_feature(feature)))
}

/// Gets identifying information for an adapter.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle. `info` must point
/// to writable `WGPUAdapterInfo` storage. String members must be released with
/// `wgpuAdapterInfoFreeMembers`.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterGetInfo(
    adapter: native::WGPUAdapter,
    info: *mut native::WGPUAdapterInfo,
) -> native::WGPUStatus {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    let Some(info) = info.as_mut() else {
        return native::WGPUStatus_Error;
    };
    *info = adapter_info_from_core(&adapter.core);
    native::WGPUStatus_Success
}

/// Frees string members allocated by `wgpuAdapterGetInfo`.
///
/// # Safety
///
/// Any non-null string member must have been returned by yawgpu.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterInfoFreeMembers(info: native::WGPUAdapterInfo) {
    free_owned_string_view(info.vendor);
    free_owned_string_view(info.architecture);
    free_owned_string_view(info.device);
    free_owned_string_view(info.description);
}

/// Requests a device from an adapter.
///
/// # Safety
///
/// `adapter` must be a non-null live yawgpu adapter handle. `descriptor`, when
/// non-null, must point to a valid `WGPUDeviceDescriptor`.
#[no_mangle]
pub unsafe extern "C" fn wgpuAdapterRequestDevice(
    adapter: native::WGPUAdapter,
    descriptor: *const native::WGPUDeviceDescriptor,
    callback_info: native::WGPURequestDeviceCallbackInfo,
) -> native::WGPUFuture {
    let adapter = borrow_handle(adapter, "WGPUAdapter");
    let required_limits = descriptor
        .as_ref()
        .and_then(|descriptor| descriptor.requiredLimits.as_ref())
        .map(map_limits);
    let required_features = descriptor
        .as_ref()
        .map(|descriptor| required_features_from_descriptor(descriptor))
        .unwrap_or_default();
    let label = descriptor
        .as_ref()
        .and_then(|descriptor| label_from_string_view(descriptor.label))
        .unwrap_or_default();
    let queue_label = descriptor
        .as_ref()
        .and_then(|descriptor| label_from_string_view(descriptor.defaultQueue.label))
        .unwrap_or_default();
    let device_lost_callback = descriptor
        .as_ref()
        .map(|descriptor| map_device_lost_callback_info(descriptor.deviceLostCallbackInfo))
        .unwrap_or(DeviceLostCallbackInfo {
            mode: 0,
            callback: None,
            userdata1: 0,
            userdata2: 0,
        });
    let result = adapter
        .core
        .create_device(
            required_limits.as_ref(),
            &required_features,
            label,
            queue_label,
        )
        .map(|device| {
            Arc::new(WGPUDeviceImpl {
                core: Arc::new(device),
                instance: Arc::clone(&adapter.instance),
                device_lost_callback,
                device_lost_futures: Mutex::new(Vec::new()),
                default_queue: Mutex::new(None),
                shader_module_cache: Mutex::new(HashMap::new()),
                pipeline_layout_cache: Mutex::new(HashMap::new()),
                compute_pipeline_cache: Mutex::new(HashMap::new()),
                render_pipeline_cache: Mutex::new(HashMap::new()),
            })
        })
        .map_err(|err| err.to_string());
    let failed = result.is_err();

    let future = adapter
        .instance
        .register_callback(PendingCallback::RequestDevice {
            mode: callback_info.mode,
            callback: callback_info.callback,
            result,
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        });

    if failed && device_lost_callback.has_callback() {
        adapter
            .instance
            .register_callback(PendingCallback::DeviceLost {
                mode: device_lost_callback.mode,
                callback: device_lost_callback.callback,
                device: 0,
                reason: core::DeviceLostReason::FailedCreation,
                userdata1: device_lost_callback.userdata1,
                userdata2: device_lost_callback.userdata2,
            });
    }

    future
}

/// Releases one owned reference to a device handle.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceRelease(device: native::WGPUDevice) {
    let device = device
        .as_ref()
        .map(|_| device)
        .unwrap_or_else(|| panic!("WGPUDevice must not be null"));
    let owned = Arc::from_raw(device);
    if Arc::strong_count(&owned) == 1 {
        owned.implicit_destroy_on_last_release();
    }
}

/// Adds one owned reference to a device handle.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceAddRef(device: native::WGPUDevice) {
    add_ref_handle(device, "WGPUDevice");
}

/// Destroys a device and fires its device-lost callback once.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceDestroy(device: native::WGPUDevice) {
    let device_impl = borrow_handle(device, "WGPUDevice");
    device_impl.schedule_device_lost(device, core::DeviceLostReason::Destroyed);
}

/// Returns a future that completes when the device is lost.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetLostFuture(device: native::WGPUDevice) -> native::WGPUFuture {
    let device_impl = borrow_handle(device, "WGPUDevice");
    device_impl.get_lost_future(device)
}

/// Pushes a device error scope for matching errors.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDevicePushErrorScope(
    device: native::WGPUDevice,
    filter: native::WGPUErrorFilter,
) {
    let device = borrow_handle(device, "WGPUDevice");
    let Some(filter) = map_error_filter(filter) else {
        device.dispatch_error(core::ErrorKind::Validation, "error scope filter is invalid");
        return;
    };
    device.core.push_error_scope(filter);
}

/// Pops the innermost device error scope and resolves through the callback future.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `callback_info`
/// userdata pointers must remain valid until the callback fires.
#[no_mangle]
pub unsafe extern "C" fn wgpuDevicePopErrorScope(
    device: native::WGPUDevice,
    callback_info: native::WGPUPopErrorScopeCallbackInfo,
) -> native::WGPUFuture {
    let device = borrow_handle(device, "WGPUDevice");
    let (status, error, message) = if device.core.is_lost() {
        (map_pop_error_scope_status_success(), None, String::new())
    } else {
        match device.core.pop_error_scope() {
            Ok(error) => (map_pop_error_scope_status_success(), error, String::new()),
            Err(core::PopErrorScopeError::EmptyStack) => (
                map_pop_error_scope_status_error(),
                None,
                "No error scopes are open".to_owned(),
            ),
            Err(_) => (
                map_pop_error_scope_status_error(),
                None,
                "Pop error scope failed".to_owned(),
            ),
        }
    };
    device
        .instance
        .register_callback(PendingCallback::PopErrorScope {
            mode: callback_info.mode,
            callback: callback_info.callback,
            status,
            error,
            message,
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        })
}

/// Sets the debug label for a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `label` must point
/// to valid string data according to `WGPUStringView` when non-empty.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceSetLabel(
    device: native::WGPUDevice,
    label: native::WGPUStringView,
) {
    let device = borrow_handle(device, "WGPUDevice");
    let label = label_from_string_view(label).unwrap_or_default();
    device.core.set_label(&label);
}

/// Creates a buffer on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUBufferDescriptor`.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBuffer(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUBufferDescriptor,
) -> native::WGPUBuffer {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUBufferDescriptor must not be null");
    let buffer = device.core.create_buffer(map_buffer_descriptor(descriptor));
    arc_to_handle(Arc::new(WGPUBufferImpl {
        core: Arc::new(buffer),
        device: Arc::clone(&device.core),
        instance: Arc::clone(&device.instance),
    }))
}

/// Creates a texture on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUTextureDescriptor`.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateTexture(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUTextureDescriptor,
) -> native::WGPUTexture {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUTextureDescriptor must not be null");
    let texture = device
        .core
        .create_texture(map_texture_descriptor(descriptor));
    arc_to_handle(Arc::new(WGPUTextureImpl {
        core: Arc::new(texture),
        device: Arc::clone(&device.core),
        instance: Arc::clone(&device.instance),
    }))
}

/// Creates a sampler on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor`, when
/// non-null, must point to a valid `WGPUSamplerDescriptor`.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateSampler(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUSamplerDescriptor,
) -> native::WGPUSampler {
    let device = borrow_handle(device, "WGPUDevice");
    let sampler = device
        .core
        .create_sampler(map_sampler_descriptor(descriptor.as_ref()));
    arc_to_handle(Arc::new(WGPUSamplerImpl {
        _core: Arc::new(sampler),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Creates a query set on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUQuerySetDescriptor`.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateQuerySet(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUQuerySetDescriptor,
) -> native::WGPUQuerySet {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUQuerySetDescriptor must not be null");
    let (query_set, error) = device
        .core
        .create_query_set(map_query_set_descriptor(descriptor));
    if let Some(message) = error {
        device.dispatch_error(core::ErrorKind::Validation, message);
    }
    arc_to_handle(Arc::new(WGPUQuerySetImpl {
        core: Arc::new(query_set),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Creates a shader module on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUShaderModuleDescriptor` and its extension chain must
/// contain exactly one recognized shader source.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateShaderModule(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUShaderModuleDescriptor,
) -> native::WGPUShaderModule {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUShaderModuleDescriptor must not be null");
    let source = map_shader_module_descriptor(descriptor);
    let key = shader_module_cache_key(&source);
    let shader_module = device.core.create_shader_module(source);
    let handle = Arc::new(WGPUShaderModuleImpl {
        _core: Arc::new(shader_module),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    });
    let handle = if !handle._core.is_error() {
        if let Some(key) = key {
            cache_handle(&device.shader_module_cache, key, handle)
        } else {
            handle
        }
    } else {
        handle
    };
    arc_to_handle(handle)
}

/// Creates a bind group layout on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUBindGroupLayoutDescriptor`.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBindGroupLayout(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUBindGroupLayoutDescriptor,
) -> native::WGPUBindGroupLayout {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUBindGroupLayoutDescriptor must not be null");
    let layout = device
        .core
        .create_bind_group_layout(map_bind_group_layout_descriptor(descriptor));
    arc_to_handle(Arc::new(WGPUBindGroupLayoutImpl {
        _core: Arc::new(layout),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Creates a bind group on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUBindGroupDescriptor`. `descriptor.layout` must be a
/// non-null live yawgpu bind group layout handle. `descriptor.entries`, when
/// non-null and `entryCount > 0`, must point to valid bind group entries.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateBindGroup(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUBindGroupDescriptor,
) -> native::WGPUBindGroup {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUBindGroupDescriptor must not be null");
    let layout = clone_handle(descriptor.layout, "WGPUBindGroupLayout");
    let bind_group = device.core.create_bind_group(
        Arc::clone(&layout._core),
        map_bind_group_entries(descriptor),
    );
    arc_to_handle(Arc::new(WGPUBindGroupImpl {
        _core: Arc::new(bind_group),
        _layout: Arc::clone(&layout._core),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Creates a pipeline layout on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUPipelineLayoutDescriptor`. Its `bindGroupLayouts`
/// array may be null only when `bindGroupLayoutCount` is zero.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreatePipelineLayout(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUPipelineLayoutDescriptor,
) -> native::WGPUPipelineLayout {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUPipelineLayoutDescriptor must not be null");
    let key = pipeline_layout_cache_key(descriptor);
    let device_error = validate_pipeline_layout_devices(device, descriptor);
    let mut descriptor = map_pipeline_layout_descriptor(descriptor);
    if descriptor.error.is_none() {
        descriptor.error = device_error;
    }
    let pipeline_layout = device.core.create_pipeline_layout(descriptor);
    let handle = Arc::new(WGPUPipelineLayoutImpl {
        _core: Arc::new(pipeline_layout),
        _device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    });
    let handle = if !handle._core.is_error() {
        if let Some(key) = key {
            cache_handle(&device.pipeline_layout_cache, key, handle)
        } else {
            handle
        }
    } else {
        handle
    };
    arc_to_handle(handle)
}

/// Creates a compute pipeline on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUComputePipelineDescriptor`. `descriptor.compute.module`
/// must be a non-null live yawgpu shader module handle. `descriptor.layout`
/// may be null to request automatic layout.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateComputePipeline(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUComputePipelineDescriptor,
) -> native::WGPUComputePipeline {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUComputePipelineDescriptor must not be null");
    arc_to_handle(create_compute_pipeline_handle(device, descriptor, true))
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

/// Creates a compute pipeline asynchronously on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPUComputePipelineDescriptor`. The callback info follows
/// the `webgpu.h` callback contract.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateComputePipelineAsync(
    device: native::WGPUDevice,
    descriptor: *const native::WGPUComputePipelineDescriptor,
    callback_info: native::WGPUCreateComputePipelineAsyncCallbackInfo,
) -> native::WGPUFuture {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPUComputePipelineDescriptor must not be null");
    let pipeline = create_compute_pipeline_handle(device, descriptor, false);
    device
        .instance
        .register_callback(PendingCallback::CreateComputePipelineAsync {
            mode: callback_info.mode,
            callback: callback_info.callback,
            pipeline,
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        })
}

/// Creates a render pipeline on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPURenderPipelineDescriptor`. `descriptor.vertex.module`
/// and optional `descriptor.fragment.module` must be non-null live yawgpu
/// shader module handles. `descriptor.layout`, `depthStencil`, and `fragment`
/// may be null where allowed by WebGPU.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateRenderPipeline(
    device: native::WGPUDevice,
    descriptor: *const native::WGPURenderPipelineDescriptor,
) -> native::WGPURenderPipeline {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPURenderPipelineDescriptor must not be null");
    arc_to_handle(create_render_pipeline_handle(device, descriptor, true))
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

/// Creates a render pipeline asynchronously on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` must
/// point to a valid `WGPURenderPipelineDescriptor`. The callback info follows
/// the `webgpu.h` callback contract.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateRenderPipelineAsync(
    device: native::WGPUDevice,
    descriptor: *const native::WGPURenderPipelineDescriptor,
    callback_info: native::WGPUCreateRenderPipelineAsyncCallbackInfo,
) -> native::WGPUFuture {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPURenderPipelineDescriptor must not be null");
    let pipeline = create_render_pipeline_handle(device, descriptor, false);
    device
        .instance
        .register_callback(PendingCallback::CreateRenderPipelineAsync {
            mode: callback_info.mode,
            callback: callback_info.callback,
            pipeline,
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        })
}

/// Creates a command encoder on a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `descriptor` may be
/// null; P6.1 stores no command encoder descriptor fields.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateCommandEncoder(
    device: native::WGPUDevice,
    _descriptor: *const native::WGPUCommandEncoderDescriptor,
) -> native::WGPUCommandEncoder {
    let device = borrow_handle(device, "WGPUDevice");
    arc_to_handle(Arc::new(WGPUCommandEncoderImpl {
        core: Arc::new(device.core.create_command_encoder()),
        device: Arc::clone(&device.core),
        instance: Arc::clone(&device.instance),
    }))
}

/// Creates a render bundle encoder on a device.
///
/// # Safety
///
/// `device` and `descriptor` must be non-null live yawgpu pointers.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceCreateRenderBundleEncoder(
    device: native::WGPUDevice,
    descriptor: *const native::WGPURenderBundleEncoderDescriptor,
) -> native::WGPURenderBundleEncoder {
    let device = borrow_handle(device, "WGPUDevice");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPURenderBundleEncoderDescriptor must not be null");
    let descriptor = map_render_bundle_encoder_descriptor(
        descriptor,
        device.core.limits().max_color_attachments,
    );
    let (encoder, error) = core::RenderBundleEncoder::new(descriptor, device.core.limits());
    dispatch_optional_error(&device.core, error);
    arc_to_handle(Arc::new(WGPURenderBundleEncoderImpl {
        core: Arc::new(encoder),
        device: Arc::clone(&device.core),
        _instance: Arc::clone(&device.instance),
    }))
}

/// Begins a render pass.
///
/// # Safety
///
/// `command_encoder` and `descriptor` must be non-null live yawgpu pointers.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderBeginRenderPass(
    command_encoder: native::WGPUCommandEncoder,
    descriptor: *const native::WGPURenderPassDescriptor,
) -> native::WGPURenderPassEncoder {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let descriptor = descriptor
        .as_ref()
        .expect("WGPURenderPassDescriptor must not be null");
    let descriptor =
        map_render_pass_descriptor(descriptor, encoder.device.limits().max_color_attachments);
    let (pass, error) = encoder.core.begin_render_pass(&descriptor);
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(WGPURenderPassEncoderImpl {
        core: Arc::new(pass),
        device: Arc::clone(&encoder.device),
        _parent: Arc::clone(&encoder.core),
        _instance: Arc::clone(&encoder.instance),
    }))
}

/// Begins a compute pass. The descriptor is nullable by `webgpu.h`; P6.1
/// tracks lifecycle only.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderBeginComputePass(
    command_encoder: native::WGPUCommandEncoder,
    _descriptor: *const native::WGPUComputePassDescriptor,
) -> native::WGPUComputePassEncoder {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let (pass, error) = encoder.core.begin_compute_pass();
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(WGPUComputePassEncoderImpl {
        core: Arc::new(pass),
        device: Arc::clone(&encoder.device),
        _parent: Arc::clone(&encoder.core),
        _instance: Arc::clone(&encoder.instance),
    }))
}

/// Finishes command encoding into a command buffer.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
/// `descriptor` may be null; P6.1 stores no command buffer descriptor fields.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderFinish(
    command_encoder: native::WGPUCommandEncoder,
    _descriptor: *const native::WGPUCommandBufferDescriptor,
) -> native::WGPUCommandBuffer {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let (command_buffer, error) = encoder.core.finish();
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(WGPUCommandBufferImpl {
        core: Arc::new(command_buffer),
        _device: Arc::clone(&encoder.device),
        _instance: Arc::clone(&encoder.instance),
    }))
}

/// Inserts an encoder debug marker.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderInsertDebugMarker(
    command_encoder: native::WGPUCommandEncoder,
    _marker_label: native::WGPUStringView,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.insert_debug_marker());
}

/// Pushes an encoder debug group.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderPushDebugGroup(
    command_encoder: native::WGPUCommandEncoder,
    _group_label: native::WGPUStringView,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.push_debug_group());
}

/// Pops an encoder debug group.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderPopDebugGroup(
    command_encoder: native::WGPUCommandEncoder,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.pop_debug_group());
}

/// Records a buffer-to-buffer copy command.
///
/// # Safety
///
/// `command_encoder`, `source`, and `destination` must be non-null live yawgpu
/// handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyBufferToBuffer(
    command_encoder: native::WGPUCommandEncoder,
    source: native::WGPUBuffer,
    source_offset: u64,
    destination: native::WGPUBuffer,
    destination_offset: u64,
    size: u64,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let source = clone_handle(source, "WGPUBuffer");
    let destination = clone_handle(destination, "WGPUBuffer");
    if !source.device.same(&encoder.device) || !destination.device.same(&encoder.device) {
        dispatch_optional_error(
            &encoder.device,
            encoder
                .core
                .record_validation_error("copy buffers must belong to the command encoder device"),
        );
        return;
    }
    dispatch_optional_error(
        &encoder.device,
        encoder.core.copy_buffer_to_buffer(
            Arc::clone(&source.core),
            source_offset,
            Arc::clone(&destination.core),
            destination_offset,
            size,
        ),
    );
}

/// Records a buffer clear command.
///
/// # Safety
///
/// `command_encoder` and `buffer` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderClearBuffer(
    command_encoder: native::WGPUCommandEncoder,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let buffer = clone_handle(buffer, "WGPUBuffer");
    dispatch_optional_error(
        &encoder.device,
        encoder
            .core
            .clear_buffer(Arc::clone(&buffer.core), offset, size),
    );
}

/// Records a host-to-buffer write command. Noop validation does not consume
/// the `data` bytes.
///
/// # Safety
///
/// `command_encoder` and `buffer` must be non-null live yawgpu handles. `data`
/// is not read by this P6.2 validation implementation.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderWriteBuffer(
    command_encoder: native::WGPUCommandEncoder,
    buffer: native::WGPUBuffer,
    buffer_offset: u64,
    _data: *const c_void,
    size: usize,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let buffer = clone_handle(buffer, "WGPUBuffer");
    let size = match u64::try_from(size) {
        Ok(size) => size,
        Err(_) => {
            dispatch_optional_error(
                &encoder.device,
                Some("command encoder write buffer size is too large".to_owned()),
            );
            return;
        }
    };
    dispatch_optional_error(
        &encoder.device,
        encoder
            .core
            .write_buffer(Arc::clone(&buffer.core), buffer_offset, size),
    );
}

/// Records a timestamp write command.
///
/// # Safety
///
/// `command_encoder` and `query_set` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderWriteTimestamp(
    command_encoder: native::WGPUCommandEncoder,
    query_set: native::WGPUQuerySet,
    query_index: u32,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let query_set = clone_handle(query_set, "WGPUQuerySet");
    dispatch_optional_error(
        &encoder.device,
        encoder
            .core
            .write_timestamp(Arc::clone(&query_set.core), query_index),
    );
}

/// Records a query set resolve command.
///
/// # Safety
///
/// `command_encoder`, `query_set`, and `destination` must be non-null live
/// yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderResolveQuerySet(
    command_encoder: native::WGPUCommandEncoder,
    query_set: native::WGPUQuerySet,
    first_query: u32,
    query_count: u32,
    destination: native::WGPUBuffer,
    destination_offset: u64,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let query_set = clone_handle(query_set, "WGPUQuerySet");
    let destination = clone_handle(destination, "WGPUBuffer");
    dispatch_optional_error(
        &encoder.device,
        encoder.core.resolve_query_set(
            Arc::clone(&query_set.core),
            first_query,
            query_count,
            Arc::clone(&destination.core),
            destination_offset,
        ),
    );
}

/// Records a buffer-to-texture copy command.
///
/// # Safety
///
/// `command_encoder`, `source`, `destination`, and `copy_size` must be
/// non-null. Nested buffer and texture handles must be non-null live yawgpu
/// handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyBufferToTexture(
    command_encoder: native::WGPUCommandEncoder,
    source: *const native::WGPUTexelCopyBufferInfo,
    destination: *const native::WGPUTexelCopyTextureInfo,
    copy_size: *const native::WGPUExtent3D,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let source = source
        .as_ref()
        .expect("wgpuCommandEncoderCopyBufferToTexture source must not be null");
    let destination = destination
        .as_ref()
        .expect("wgpuCommandEncoderCopyBufferToTexture destination must not be null");
    let copy_size = copy_size
        .as_ref()
        .expect("wgpuCommandEncoderCopyBufferToTexture copySize must not be null");
    let source_buffer = clone_handle(source.buffer, "WGPUBuffer");
    let destination_texture = clone_handle(destination.texture, "WGPUTexture");
    let (destination_mip_level, destination_origin, destination_aspect) =
        map_texel_copy_texture_info_parts(destination);

    dispatch_optional_error(
        &encoder.device,
        encoder.core.copy_buffer_to_texture(
            core::TexelCopyBufferInfo {
                buffer: Arc::clone(&source_buffer.core),
                layout: map_texel_copy_buffer_layout(source.layout),
            },
            core::TexelCopyTextureInfo {
                texture: Arc::clone(&destination_texture.core),
                mip_level: destination_mip_level,
                origin: destination_origin,
                aspect: destination_aspect,
            },
            map_extent_3d(*copy_size),
        ),
    );
}

/// Records a texture-to-buffer copy command.
///
/// # Safety
///
/// `command_encoder`, `source`, `destination`, and `copy_size` must be
/// non-null. Nested texture and buffer handles must be non-null live yawgpu
/// handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyTextureToBuffer(
    command_encoder: native::WGPUCommandEncoder,
    source: *const native::WGPUTexelCopyTextureInfo,
    destination: *const native::WGPUTexelCopyBufferInfo,
    copy_size: *const native::WGPUExtent3D,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let source = source
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToBuffer source must not be null");
    let destination = destination
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToBuffer destination must not be null");
    let copy_size = copy_size
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToBuffer copySize must not be null");
    let source_texture = clone_handle(source.texture, "WGPUTexture");
    let destination_buffer = clone_handle(destination.buffer, "WGPUBuffer");
    let (source_mip_level, source_origin, source_aspect) =
        map_texel_copy_texture_info_parts(source);

    dispatch_optional_error(
        &encoder.device,
        encoder.core.copy_texture_to_buffer(
            core::TexelCopyTextureInfo {
                texture: Arc::clone(&source_texture.core),
                mip_level: source_mip_level,
                origin: source_origin,
                aspect: source_aspect,
            },
            core::TexelCopyBufferInfo {
                buffer: Arc::clone(&destination_buffer.core),
                layout: map_texel_copy_buffer_layout(destination.layout),
            },
            map_extent_3d(*copy_size),
        ),
    );
}

/// Records a texture-to-texture copy command.
///
/// # Safety
///
/// `command_encoder`, `source`, `destination`, and `copy_size` must be
/// non-null. Nested texture handles must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderCopyTextureToTexture(
    command_encoder: native::WGPUCommandEncoder,
    source: *const native::WGPUTexelCopyTextureInfo,
    destination: *const native::WGPUTexelCopyTextureInfo,
    copy_size: *const native::WGPUExtent3D,
) {
    let encoder = borrow_handle(command_encoder, "WGPUCommandEncoder");
    let source = source
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToTexture source must not be null");
    let destination = destination
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToTexture destination must not be null");
    let copy_size = copy_size
        .as_ref()
        .expect("wgpuCommandEncoderCopyTextureToTexture copySize must not be null");
    let source_texture = clone_handle(source.texture, "WGPUTexture");
    let destination_texture = clone_handle(destination.texture, "WGPUTexture");
    let (source_mip_level, source_origin, source_aspect) =
        map_texel_copy_texture_info_parts(source);
    let (destination_mip_level, destination_origin, destination_aspect) =
        map_texel_copy_texture_info_parts(destination);

    dispatch_optional_error(
        &encoder.device,
        encoder.core.copy_texture_to_texture(
            core::TexelCopyTextureInfo {
                texture: Arc::clone(&source_texture.core),
                mip_level: source_mip_level,
                origin: source_origin,
                aspect: source_aspect,
            },
            core::TexelCopyTextureInfo {
                texture: Arc::clone(&destination_texture.core),
                mip_level: destination_mip_level,
                origin: destination_origin,
                aspect: destination_aspect,
            },
            map_extent_3d(*copy_size),
        ),
    );
}

/// Ends a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderEnd(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.end());
}

/// Begins an occlusion query in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderBeginOcclusionQuery(
    render_pass_encoder: native::WGPURenderPassEncoder,
    query_index: u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.begin_occlusion_query(query_index));
}

/// Ends the current occlusion query in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderEndOcclusionQuery(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.end_occlusion_query());
}

/// Inserts a render pass debug marker.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderInsertDebugMarker(
    render_pass_encoder: native::WGPURenderPassEncoder,
    _marker_label: native::WGPUStringView,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.insert_debug_marker());
}

/// Pushes a render pass debug group.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderPushDebugGroup(
    render_pass_encoder: native::WGPURenderPassEncoder,
    _group_label: native::WGPUStringView,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.push_debug_group());
}

/// Pops a render pass debug group.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderPopDebugGroup(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.pop_debug_group());
}

/// Sets the render pipeline for a render pass.
///
/// # Safety
///
/// `render_pass_encoder` and `pipeline` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetPipeline(
    render_pass_encoder: native::WGPURenderPassEncoder,
    pipeline: native::WGPURenderPipeline,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let pipeline = clone_handle(pipeline, "WGPURenderPipeline");
    if !pipeline._device.same(&pass.device) {
        dispatch_optional_error(
            &pass.device,
            pass.core
                .record_validation_error("render pipeline must belong to the render pass device"),
        );
        return;
    }
    dispatch_optional_error(
        &pass.device,
        pass.core.set_pipeline(Arc::clone(&pipeline._core)),
    );
}

/// Sets or clears a render pass bind group.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// `group` may be null to clear the slot. `dynamic_offsets` must point to
/// `dynamic_offset_count` elements when the count is non-zero.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetBindGroup(
    render_pass_encoder: native::WGPURenderPassEncoder,
    group_index: u32,
    group: native::WGPUBindGroup,
    dynamic_offset_count: usize,
    dynamic_offsets: *const u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let group =
        (!group.is_null()).then(|| clone_handle::<WGPUBindGroupImpl>(group, "WGPUBindGroup"));
    if let Some(group) = group.as_ref() {
        if !group._device.same(&pass.device) {
            dispatch_optional_error(
                &pass.device,
                pass.core
                    .record_validation_error("bind group must belong to the render pass device"),
            );
            return;
        }
    }
    let offsets = dynamic_offsets_slice(dynamic_offset_count, dynamic_offsets);
    dispatch_optional_error(
        &pass.device,
        pass.core.set_bind_group(
            group_index,
            group.map(|group| Arc::clone(&group._core)),
            offsets,
        ),
    );
}

/// Sets or clears a render pass vertex buffer.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// `buffer` may be null to clear the slot.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetVertexBuffer(
    render_pass_encoder: native::WGPURenderPassEncoder,
    slot: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let buffer = (!buffer.is_null()).then(|| clone_handle::<WGPUBufferImpl>(buffer, "WGPUBuffer"));
    if let Some(buffer) = buffer.as_ref() {
        if !buffer.device.same(&pass.device) {
            dispatch_optional_error(
                &pass.device,
                pass.core
                    .record_validation_error("vertex buffer must belong to the render pass device"),
            );
            return;
        }
    }
    dispatch_optional_error(
        &pass.device,
        pass.core.set_vertex_buffer(
            slot,
            buffer.map(|buffer| Arc::clone(&buffer.core)),
            offset,
            size,
            pass.device.limits(),
        ),
    );
}

/// Sets the render pass index buffer.
///
/// # Safety
///
/// `render_pass_encoder` and `buffer` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetIndexBuffer(
    render_pass_encoder: native::WGPURenderPassEncoder,
    buffer: native::WGPUBuffer,
    format: native::WGPUIndexFormat,
    offset: u64,
    size: u64,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let buffer = clone_handle::<WGPUBufferImpl>(buffer, "WGPUBuffer");
    if !buffer.device.same(&pass.device) {
        dispatch_optional_error(
            &pass.device,
            pass.core
                .record_validation_error("index buffer must belong to the render pass device"),
        );
        return;
    }
    dispatch_optional_error(
        &pass.device,
        pass.core.set_index_buffer(
            Arc::clone(&buffer.core),
            map_index_format(format),
            offset,
            size,
        ),
    );
}

/// Records a non-indexed draw in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDraw(
    render_pass_encoder: native::WGPURenderPassEncoder,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core.draw(
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
            pass.device.limits(),
        ),
    );
}

/// Records an indexed draw in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDrawIndexed(
    render_pass_encoder: native::WGPURenderPassEncoder,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core.draw_indexed(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
            pass.device.limits(),
        ),
    );
}

/// Records an indirect non-indexed draw in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` and `indirect_buffer` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDrawIndirect(
    render_pass_encoder: native::WGPURenderPassEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let indirect_buffer = clone_handle::<WGPUBufferImpl>(indirect_buffer, "WGPUBuffer");
    dispatch_optional_error(
        &pass.device,
        pass.core.draw_indirect(
            Arc::clone(&indirect_buffer.core),
            indirect_offset,
            pass.device.limits(),
        ),
    );
}

/// Records an indirect indexed draw in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` and `indirect_buffer` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderDrawIndexedIndirect(
    render_pass_encoder: native::WGPURenderPassEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let indirect_buffer = clone_handle::<WGPUBufferImpl>(indirect_buffer, "WGPUBuffer");
    dispatch_optional_error(
        &pass.device,
        pass.core.draw_indexed_indirect(
            Arc::clone(&indirect_buffer.core),
            indirect_offset,
            pass.device.limits(),
        ),
    );
}

/// Sets the render pass viewport.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetViewport(
    render_pass_encoder: native::WGPURenderPassEncoder,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core
            .set_viewport(x, y, width, height, min_depth, max_depth),
    );
}

/// Sets the render pass scissor rectangle.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetScissorRect(
    render_pass_encoder: native::WGPURenderPassEncoder,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core.set_scissor_rect(x, y, width, height),
    );
}

/// Sets the render pass blend constant.
///
/// # Safety
///
/// `render_pass_encoder` and `color` must be non-null live pointers.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetBlendConstant(
    render_pass_encoder: native::WGPURenderPassEncoder,
    color: *const native::WGPUColor,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let color = color
        .as_ref()
        .expect("WGPUColor for SetBlendConstant must not be null");
    dispatch_optional_error(
        &pass.device,
        pass.core.set_blend_constant(map_color(*color)),
    );
}

/// Sets the render pass stencil reference.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderSetStencilReference(
    render_pass_encoder: native::WGPURenderPassEncoder,
    reference: u32,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    dispatch_optional_error(&pass.device, pass.core.set_stencil_reference(reference));
}

/// Executes render bundles in a render pass.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
/// `bundles` must point to `bundle_count` live render bundle handles when the
/// count is non-zero.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderExecuteBundles(
    render_pass_encoder: native::WGPURenderPassEncoder,
    bundle_count: usize,
    bundles: *const native::WGPURenderBundle,
) {
    let pass = borrow_handle(render_pass_encoder, "WGPURenderPassEncoder");
    let bundle_handles = render_bundle_slice(bundle_count, bundles);
    if bundle_handles
        .iter()
        .any(|bundle| !bundle._device.same(&pass.device))
    {
        dispatch_optional_error(
            &pass.device,
            pass.core
                .record_validation_error("render bundle must belong to the render pass device"),
        );
        return;
    }
    let bundles = bundle_handles
        .iter()
        .map(|bundle| Arc::clone(&bundle.core))
        .collect::<Vec<_>>();
    dispatch_optional_error(&pass.device, pass.core.execute_bundles(&bundles));
}

/// Sets the render pipeline for a render bundle encoder.
///
/// # Safety
///
/// `render_bundle_encoder` and `pipeline` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetPipeline(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    pipeline: native::WGPURenderPipeline,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let pipeline = clone_handle(pipeline, "WGPURenderPipeline");
    dispatch_optional_error(
        &encoder.device,
        encoder.core.set_pipeline(Arc::clone(&pipeline._core)),
    );
}

/// Sets or clears a render bundle bind group.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// `group` may be null. `dynamic_offsets` must point to `dynamic_offset_count`
/// elements when the count is non-zero.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetBindGroup(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    group_index: u32,
    group: native::WGPUBindGroup,
    dynamic_offset_count: usize,
    dynamic_offsets: *const u32,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let group =
        (!group.is_null()).then(|| clone_handle::<WGPUBindGroupImpl>(group, "WGPUBindGroup"));
    let offsets = dynamic_offsets_slice(dynamic_offset_count, dynamic_offsets);
    dispatch_optional_error(
        &encoder.device,
        encoder.core.set_bind_group(
            group_index,
            group.map(|group| Arc::clone(&group._core)),
            offsets,
        ),
    );
}

/// Sets or clears a render bundle vertex buffer.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
/// `buffer` may be null.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetVertexBuffer(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    slot: u32,
    buffer: native::WGPUBuffer,
    offset: u64,
    size: u64,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let buffer = (!buffer.is_null()).then(|| clone_handle::<WGPUBufferImpl>(buffer, "WGPUBuffer"));
    dispatch_optional_error(
        &encoder.device,
        encoder.core.set_vertex_buffer(
            slot,
            buffer.map(|buffer| Arc::clone(&buffer.core)),
            offset,
            size,
            encoder.device.limits(),
        ),
    );
}

/// Sets a render bundle index buffer.
///
/// # Safety
///
/// `render_bundle_encoder` and `buffer` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderSetIndexBuffer(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    buffer: native::WGPUBuffer,
    format: native::WGPUIndexFormat,
    offset: u64,
    size: u64,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let buffer = clone_handle::<WGPUBufferImpl>(buffer, "WGPUBuffer");
    dispatch_optional_error(
        &encoder.device,
        encoder.core.set_index_buffer(
            Arc::clone(&buffer.core),
            map_index_format(format),
            offset,
            size,
        ),
    );
}

/// Records a non-indexed draw in a render bundle.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDraw(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    dispatch_optional_error(
        &encoder.device,
        encoder.core.draw(
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
            encoder.device.limits(),
        ),
    );
}

/// Records an indexed draw in a render bundle.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDrawIndexed(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    dispatch_optional_error(
        &encoder.device,
        encoder.core.draw_indexed(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
            encoder.device.limits(),
        ),
    );
}

/// Records an indirect non-indexed draw in a render bundle.
///
/// # Safety
///
/// `render_bundle_encoder` and `indirect_buffer` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDrawIndirect(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let indirect_buffer = clone_handle::<WGPUBufferImpl>(indirect_buffer, "WGPUBuffer");
    dispatch_optional_error(
        &encoder.device,
        encoder.core.draw_indirect(
            Arc::clone(&indirect_buffer.core),
            indirect_offset,
            encoder.device.limits(),
        ),
    );
}

/// Records an indirect indexed draw in a render bundle.
///
/// # Safety
///
/// `render_bundle_encoder` and `indirect_buffer` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderDrawIndexedIndirect(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let indirect_buffer = clone_handle::<WGPUBufferImpl>(indirect_buffer, "WGPUBuffer");
    dispatch_optional_error(
        &encoder.device,
        encoder.core.draw_indexed_indirect(
            Arc::clone(&indirect_buffer.core),
            indirect_offset,
            encoder.device.limits(),
        ),
    );
}

/// Inserts a render bundle debug marker.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderInsertDebugMarker(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    _marker_label: native::WGPUStringView,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.insert_debug_marker());
}

/// Pushes a render bundle debug group.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderPushDebugGroup(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    _group_label: native::WGPUStringView,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.push_debug_group());
}

/// Pops a render bundle debug group.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderPopDebugGroup(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
) {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    dispatch_optional_error(&encoder.device, encoder.core.pop_debug_group());
}

/// Finishes a render bundle encoder.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderFinish(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
    _descriptor: *const native::WGPURenderBundleDescriptor,
) -> native::WGPURenderBundle {
    let encoder = borrow_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
    let (bundle, error) = encoder.core.finish();
    dispatch_optional_error(&encoder.device, error);
    arc_to_handle(Arc::new(WGPURenderBundleImpl {
        core: Arc::new(bundle),
        _device: Arc::clone(&encoder.device),
        _instance: Arc::clone(&encoder._instance),
    }))
}

/// Ends a compute pass.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderEnd(
    compute_pass_encoder: native::WGPUComputePassEncoder,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    dispatch_optional_error(&pass.device, pass.core.end());
}

/// Inserts a compute pass debug marker.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderInsertDebugMarker(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    _marker_label: native::WGPUStringView,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    dispatch_optional_error(&pass.device, pass.core.insert_debug_marker());
}

/// Pushes a compute pass debug group.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderPushDebugGroup(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    _group_label: native::WGPUStringView,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    dispatch_optional_error(&pass.device, pass.core.push_debug_group());
}

/// Pops a compute pass debug group.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderPopDebugGroup(
    compute_pass_encoder: native::WGPUComputePassEncoder,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    dispatch_optional_error(&pass.device, pass.core.pop_debug_group());
}

/// Sets the compute pipeline for a compute pass.
///
/// # Safety
///
/// `compute_pass_encoder` and `pipeline` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderSetPipeline(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    pipeline: native::WGPUComputePipeline,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    let pipeline = clone_handle(pipeline, "WGPUComputePipeline");
    if !pipeline._device.same(&pass.device) {
        dispatch_optional_error(
            &pass.device,
            pass.core
                .record_validation_error("compute pipeline must belong to the compute pass device"),
        );
        return;
    }
    dispatch_optional_error(
        &pass.device,
        pass.core.set_pipeline(Arc::clone(&pipeline._core)),
    );
}

/// Sets or clears a compute pass bind group.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
/// `group` may be null to clear the slot. `dynamic_offsets` must point to
/// `dynamic_offset_count` elements when the count is non-zero.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderSetBindGroup(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    group_index: u32,
    group: native::WGPUBindGroup,
    dynamic_offset_count: usize,
    dynamic_offsets: *const u32,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    let group =
        (!group.is_null()).then(|| clone_handle::<WGPUBindGroupImpl>(group, "WGPUBindGroup"));
    if let Some(group) = group.as_ref() {
        if !group._device.same(&pass.device) {
            dispatch_optional_error(
                &pass.device,
                pass.core
                    .record_validation_error("bind group must belong to the compute pass device"),
            );
            return;
        }
    }
    let offsets = dynamic_offsets_slice(dynamic_offset_count, dynamic_offsets);
    dispatch_optional_error(
        &pass.device,
        pass.core.set_bind_group(
            group_index,
            group.map(|group| Arc::clone(&group._core)),
            offsets,
        ),
    );
}

/// Records a compute dispatch.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderDispatchWorkgroups(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    workgroup_count_x: u32,
    workgroup_count_y: u32,
    workgroup_count_z: u32,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    dispatch_optional_error(
        &pass.device,
        pass.core.dispatch_workgroups(
            workgroup_count_x,
            workgroup_count_y,
            workgroup_count_z,
            pass.device.limits(),
        ),
    );
}

/// Records an indirect compute dispatch.
///
/// # Safety
///
/// `compute_pass_encoder` and `indirect_buffer` must be non-null live yawgpu handles.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderDispatchWorkgroupsIndirect(
    compute_pass_encoder: native::WGPUComputePassEncoder,
    indirect_buffer: native::WGPUBuffer,
    indirect_offset: u64,
) {
    let pass = borrow_handle(compute_pass_encoder, "WGPUComputePassEncoder");
    let indirect_buffer = clone_handle::<WGPUBufferImpl>(indirect_buffer, "WGPUBuffer");
    dispatch_optional_error(
        &pass.device,
        pass.core.dispatch_workgroups_indirect(
            Arc::clone(&indirect_buffer.core),
            indirect_offset,
            pass.device.limits(),
        ),
    );
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

/// Gets a compute pipeline bind group layout.
///
/// # Safety
///
/// `compute_pipeline` must be a non-null live yawgpu compute pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineGetBindGroupLayout(
    compute_pipeline: native::WGPUComputePipeline,
    group_index: u32,
) -> native::WGPUBindGroupLayout {
    let pipeline = borrow_handle(compute_pipeline, "WGPUComputePipeline");
    get_pipeline_bind_group_layout(
        pipeline._core.bind_group_layouts(),
        &pipeline._device,
        &pipeline._instance,
        &pipeline.bind_group_layout_handles,
        group_index,
    )
}

/// Gets a render pipeline bind group layout.
///
/// # Safety
///
/// `render_pipeline` must be a non-null live yawgpu render pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineGetBindGroupLayout(
    render_pipeline: native::WGPURenderPipeline,
    group_index: u32,
) -> native::WGPUBindGroupLayout {
    let pipeline = borrow_handle(render_pipeline, "WGPURenderPipeline");
    get_pipeline_bind_group_layout(
        pipeline._core.bind_group_layouts(),
        &pipeline._device,
        &pipeline._instance,
        &pipeline.bind_group_layout_handles,
        group_index,
    )
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

/// Gets the effective limits for a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `limits` must point
/// to writable `WGPULimits` storage.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetLimits(
    device: native::WGPUDevice,
    limits: *mut native::WGPULimits,
) -> native::WGPUStatus {
    let device = borrow_handle(device, "WGPUDevice");
    let Some(limits) = limits.as_mut() else {
        return native::WGPUStatus_Error;
    };
    *limits = map_limits_to_native(device.core.limits());
    native::WGPUStatus_Success
}

/// Gets the resolved features for a device.
///
/// The returned `features` array is allocated by yawgpu and must be released
/// with `wgpuSupportedFeaturesFreeMembers`.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle. `features` must
/// point to writable `WGPUSupportedFeatures` storage.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetFeatures(
    device: native::WGPUDevice,
    features: *mut native::WGPUSupportedFeatures,
) {
    let device = borrow_handle(device, "WGPUDevice");
    let features = features
        .as_mut()
        .expect("WGPUSupportedFeatures must not be null");
    *features = map_features_to_native(&device.core.features());
}

/// Returns whether the device has `feature` enabled.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceHasFeature(
    device: native::WGPUDevice,
    feature: native::WGPUFeatureName,
) -> native::WGPUBool {
    let device = borrow_handle(device, "WGPUDevice");
    native::WGPUBool::from(device.core.has_feature(map_feature(feature)))
}

/// Gets the default queue for a device.
///
/// # Safety
///
/// `device` must be a non-null live yawgpu device handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuDeviceGetQueue(device: native::WGPUDevice) -> native::WGPUQueue {
    let device = borrow_handle(device, "WGPUDevice");
    arc_to_handle(device.default_queue())
}

/// Destroys a buffer. This operation is idempotent.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferDestroy(buffer: native::WGPUBuffer) {
    borrow_handle(buffer, "WGPUBuffer").core.destroy();
}

/// Unmaps a buffer. This is safe on unmapped, destroyed, and error buffers.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferUnmap(buffer: native::WGPUBuffer) {
    borrow_handle(buffer, "WGPUBuffer").core.unmap();
}

/// Asynchronously maps a buffer range.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle. `callback_info`
/// userdata pointers must remain valid until the callback fires.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferMapAsync(
    buffer: native::WGPUBuffer,
    mode: native::WGPUMapMode,
    offset: usize,
    size: usize,
    callback_info: native::WGPUBufferMapCallbackInfo,
) -> native::WGPUFuture {
    let buffer = borrow_handle(buffer, "WGPUBuffer");
    let map_result = validate_map_async(buffer, mode, offset, size);

    let pending = match map_result {
        Ok((mode, offset, size)) => match buffer.core.begin_map(mode, offset, size) {
            Ok(()) => PendingCallback::BufferMap {
                mode: callback_info.mode,
                callback: callback_info.callback,
                device: Arc::clone(&buffer.device),
                buffer: Some((*buffer.core).clone()),
                status: core::MapAsyncStatus::Success,
                userdata1: callback_info.userdata1 as usize,
                userdata2: callback_info.userdata2 as usize,
            },
            Err(message) => {
                buffer
                    .device
                    .dispatch_error(core::ErrorKind::Validation, message);
                PendingCallback::BufferMap {
                    mode: callback_info.mode,
                    callback: callback_info.callback,
                    device: Arc::clone(&buffer.device),
                    buffer: None,
                    status: core::MapAsyncStatus::Error,
                    userdata1: callback_info.userdata1 as usize,
                    userdata2: callback_info.userdata2 as usize,
                }
            }
        },
        Err(message) => {
            buffer
                .device
                .dispatch_error(core::ErrorKind::Validation, message);
            PendingCallback::BufferMap {
                mode: callback_info.mode,
                callback: callback_info.callback,
                device: Arc::clone(&buffer.device),
                buffer: None,
                status: core::MapAsyncStatus::Error,
                userdata1: callback_info.userdata1 as usize,
                userdata2: callback_info.userdata2 as usize,
            }
        }
    };

    buffer.instance.register_callback(pending)
}

/// Returns a mutable pointer to a mapped buffer range, or null on misuse.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle. The returned pointer
/// is valid only while the buffer remains mapped.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetMappedRange(
    buffer: native::WGPUBuffer,
    offset: usize,
    size: usize,
) -> *mut c_void {
    mapped_range_ptr(buffer, false, offset, size).map_or(std::ptr::null_mut(), |ptr| ptr.cast())
}

/// Returns a const pointer to a mapped buffer range, or null on misuse.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle. The returned pointer
/// is valid only while the buffer remains mapped.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetConstMappedRange(
    buffer: native::WGPUBuffer,
    offset: usize,
    size: usize,
) -> *const c_void {
    mapped_range_ptr(buffer, true, offset, size)
        .map_or(std::ptr::null(), |ptr| ptr.cast_const().cast())
}

/// Returns the buffer map state.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetMapState(
    buffer: native::WGPUBuffer,
) -> native::WGPUBufferMapState {
    map_buffer_map_state(borrow_handle(buffer, "WGPUBuffer").core.map_state())
}

/// Returns the descriptor size reflected by the buffer.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetSize(buffer: native::WGPUBuffer) -> u64 {
    borrow_handle(buffer, "WGPUBuffer").core.size()
}

/// Returns the descriptor usage reflected by the buffer.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferGetUsage(buffer: native::WGPUBuffer) -> native::WGPUBufferUsage {
    map_buffer_usage_to_native(borrow_handle(buffer, "WGPUBuffer").core.usage())
}

/// Releases one owned reference to a buffer handle.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferRelease(buffer: native::WGPUBuffer) {
    release_handle(buffer, "WGPUBuffer");
}

/// Adds one owned reference to a buffer handle.
///
/// # Safety
///
/// `buffer` must be a non-null live yawgpu buffer handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuBufferAddRef(buffer: native::WGPUBuffer) {
    add_ref_handle(buffer, "WGPUBuffer");
}

/// Releases one owned reference to a bind group layout handle.
///
/// # Safety
///
/// `bind_group_layout` must be a non-null live yawgpu bind group layout handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupLayoutRelease(
    bind_group_layout: native::WGPUBindGroupLayout,
) {
    release_handle(bind_group_layout, "WGPUBindGroupLayout");
}

/// Adds one owned reference to a bind group layout handle.
///
/// # Safety
///
/// `bind_group_layout` must be a non-null live yawgpu bind group layout handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupLayoutAddRef(bind_group_layout: native::WGPUBindGroupLayout) {
    add_ref_handle(bind_group_layout, "WGPUBindGroupLayout");
}

/// Releases one owned reference to a bind group handle.
///
/// # Safety
///
/// `bind_group` must be a non-null live yawgpu bind group handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupRelease(bind_group: native::WGPUBindGroup) {
    release_handle(bind_group, "WGPUBindGroup");
}

/// Adds one owned reference to a bind group handle.
///
/// # Safety
///
/// `bind_group` must be a non-null live yawgpu bind group handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuBindGroupAddRef(bind_group: native::WGPUBindGroup) {
    add_ref_handle(bind_group, "WGPUBindGroup");
}

/// Releases one owned reference to a pipeline layout handle.
///
/// # Safety
///
/// `pipeline_layout` must be a non-null live yawgpu pipeline layout handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuPipelineLayoutRelease(pipeline_layout: native::WGPUPipelineLayout) {
    release_handle(pipeline_layout, "WGPUPipelineLayout");
}

/// Adds one owned reference to a pipeline layout handle.
///
/// # Safety
///
/// `pipeline_layout` must be a non-null live yawgpu pipeline layout handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuPipelineLayoutAddRef(pipeline_layout: native::WGPUPipelineLayout) {
    add_ref_handle(pipeline_layout, "WGPUPipelineLayout");
}

/// Releases one owned reference to a compute pipeline handle.
///
/// # Safety
///
/// `compute_pipeline` must be a non-null live yawgpu compute pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineRelease(compute_pipeline: native::WGPUComputePipeline) {
    release_handle(compute_pipeline, "WGPUComputePipeline");
}

/// Adds one owned reference to a compute pipeline handle.
///
/// # Safety
///
/// `compute_pipeline` must be a non-null live yawgpu compute pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePipelineAddRef(compute_pipeline: native::WGPUComputePipeline) {
    add_ref_handle(compute_pipeline, "WGPUComputePipeline");
}

/// Releases one owned reference to a render pipeline handle.
///
/// # Safety
///
/// `render_pipeline` must be a non-null live yawgpu render pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineRelease(render_pipeline: native::WGPURenderPipeline) {
    release_handle(render_pipeline, "WGPURenderPipeline");
}

/// Adds one owned reference to a render pipeline handle.
///
/// # Safety
///
/// `render_pipeline` must be a non-null live yawgpu render pipeline handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPipelineAddRef(render_pipeline: native::WGPURenderPipeline) {
    add_ref_handle(render_pipeline, "WGPURenderPipeline");
}

/// Releases one owned reference to a command encoder handle.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderRelease(command_encoder: native::WGPUCommandEncoder) {
    release_handle(command_encoder, "WGPUCommandEncoder");
}

/// Adds one owned reference to a command encoder handle.
///
/// # Safety
///
/// `command_encoder` must be a non-null live yawgpu command encoder handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandEncoderAddRef(command_encoder: native::WGPUCommandEncoder) {
    add_ref_handle(command_encoder, "WGPUCommandEncoder");
}

/// Releases one owned reference to a command buffer handle.
///
/// # Safety
///
/// `command_buffer` must be a non-null live yawgpu command buffer handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandBufferRelease(command_buffer: native::WGPUCommandBuffer) {
    release_handle(command_buffer, "WGPUCommandBuffer");
}

/// Adds one owned reference to a command buffer handle.
///
/// # Safety
///
/// `command_buffer` must be a non-null live yawgpu command buffer handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuCommandBufferAddRef(command_buffer: native::WGPUCommandBuffer) {
    add_ref_handle(command_buffer, "WGPUCommandBuffer");
}

/// Releases one owned reference to a render pass encoder handle.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderRelease(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    release_handle(render_pass_encoder, "WGPURenderPassEncoder");
}

/// Adds one owned reference to a render pass encoder handle.
///
/// # Safety
///
/// `render_pass_encoder` must be a non-null live yawgpu render pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderPassEncoderAddRef(
    render_pass_encoder: native::WGPURenderPassEncoder,
) {
    add_ref_handle(render_pass_encoder, "WGPURenderPassEncoder");
}

/// Releases one owned reference to a compute pass encoder handle.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderRelease(
    compute_pass_encoder: native::WGPUComputePassEncoder,
) {
    release_handle(compute_pass_encoder, "WGPUComputePassEncoder");
}

/// Adds one owned reference to a compute pass encoder handle.
///
/// # Safety
///
/// `compute_pass_encoder` must be a non-null live yawgpu compute pass encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuComputePassEncoderAddRef(
    compute_pass_encoder: native::WGPUComputePassEncoder,
) {
    add_ref_handle(compute_pass_encoder, "WGPUComputePassEncoder");
}

/// Releases one owned reference to a render bundle encoder handle.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderRelease(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
) {
    release_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
}

/// Adds one owned reference to a render bundle encoder handle.
///
/// # Safety
///
/// `render_bundle_encoder` must be a non-null live yawgpu render bundle encoder.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleEncoderAddRef(
    render_bundle_encoder: native::WGPURenderBundleEncoder,
) {
    add_ref_handle(render_bundle_encoder, "WGPURenderBundleEncoder");
}

/// Releases one owned reference to a render bundle handle.
///
/// # Safety
///
/// `render_bundle` must be a non-null live yawgpu render bundle handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleRelease(render_bundle: native::WGPURenderBundle) {
    release_handle(render_bundle, "WGPURenderBundle");
}

/// Adds one owned reference to a render bundle handle.
///
/// # Safety
///
/// `render_bundle` must be a non-null live yawgpu render bundle handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuRenderBundleAddRef(render_bundle: native::WGPURenderBundle) {
    add_ref_handle(render_bundle, "WGPURenderBundle");
}

/// Destroys a query set. This operation is idempotent.
///
/// # Safety
///
/// `query_set` must be a non-null live yawgpu query set handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetDestroy(query_set: native::WGPUQuerySet) {
    borrow_handle(query_set, "WGPUQuerySet").core.destroy();
}

/// Returns the descriptor query type reflected by the query set.
///
/// # Safety
///
/// `query_set` must be a non-null live yawgpu query set handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetGetType(
    query_set: native::WGPUQuerySet,
) -> native::WGPUQueryType {
    map_query_type_to_native(borrow_handle(query_set, "WGPUQuerySet").core.kind())
}

/// Returns the descriptor count reflected by the query set.
///
/// # Safety
///
/// `query_set` must be a non-null live yawgpu query set handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetGetCount(query_set: native::WGPUQuerySet) -> u32 {
    borrow_handle(query_set, "WGPUQuerySet").core.count()
}

/// Sets the debug label for a query set.
///
/// # Safety
///
/// `query_set` must be a non-null live yawgpu query set handle. `label` must
/// point to valid string data according to `WGPUStringView` when non-empty.
#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetSetLabel(
    query_set: native::WGPUQuerySet,
    label: native::WGPUStringView,
) {
    let query_set = borrow_handle(query_set, "WGPUQuerySet");
    let label = label_from_string_view(label).unwrap_or_default();
    query_set.core.set_label(&label);
}

/// Releases one owned reference to a query set handle.
///
/// # Safety
///
/// `query_set` must be a non-null live yawgpu query set handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetRelease(query_set: native::WGPUQuerySet) {
    release_handle(query_set, "WGPUQuerySet");
}

/// Adds one owned reference to a query set handle.
///
/// # Safety
///
/// `query_set` must be a non-null live yawgpu query set handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetAddRef(query_set: native::WGPUQuerySet) {
    add_ref_handle(query_set, "WGPUQuerySet");
}

/// Returns deterministic Noop surface capabilities.
///
/// # Safety
///
/// `surface` and `adapter` must be non-null live yawgpu handles.
/// `capabilities`, when non-null, must point to writable memory.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceGetCapabilities(
    surface: native::WGPUSurface,
    adapter: native::WGPUAdapter,
    capabilities: *mut native::WGPUSurfaceCapabilities,
) -> native::WGPUStatus {
    let surface = borrow_handle(surface, "WGPUSurface");
    let _adapter = borrow_handle(adapter, "WGPUAdapter");
    let Some(capabilities) = capabilities.as_mut() else {
        return native::WGPUStatus_Error;
    };
    if surface.is_error {
        return native::WGPUStatus_Error;
    }
    capabilities.nextInChain = std::ptr::null_mut();
    capabilities.usages = SURFACE_USAGES;
    capabilities.formatCount = SURFACE_FORMATS.len();
    capabilities.formats = Box::leak(Box::new(SURFACE_FORMATS)).as_ptr();
    capabilities.presentModeCount = SURFACE_PRESENT_MODES.len();
    capabilities.presentModes = Box::leak(Box::new(SURFACE_PRESENT_MODES)).as_ptr();
    capabilities.alphaModeCount = SURFACE_ALPHA_MODES.len();
    capabilities.alphaModes = Box::leak(Box::new(SURFACE_ALPHA_MODES)).as_ptr();
    native::WGPUStatus_Success
}

/// Frees arrays allocated by `wgpuSurfaceGetCapabilities`.
///
/// # Safety
///
/// Any non-null array member must have been returned by yawgpu.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceCapabilitiesFreeMembers(
    capabilities: native::WGPUSurfaceCapabilities,
) {
    if !capabilities.formats.is_null() {
        drop(Box::from_raw(
            capabilities.formats as *mut [native::WGPUTextureFormat; SURFACE_FORMATS.len()],
        ));
    }
    if !capabilities.presentModes.is_null() {
        drop(Box::from_raw(
            capabilities.presentModes
                as *mut [native::WGPUPresentMode; SURFACE_PRESENT_MODES.len()],
        ));
    }
    if !capabilities.alphaModes.is_null() {
        drop(Box::from_raw(
            capabilities.alphaModes
                as *mut [native::WGPUCompositeAlphaMode; SURFACE_ALPHA_MODES.len()],
        ));
    }
}

/// Configures a surface after validating it against Noop capabilities.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle. `config`, when
/// non-null, must point to a valid `WGPUSurfaceConfiguration`.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceConfigure(
    surface: native::WGPUSurface,
    config: *const native::WGPUSurfaceConfiguration,
) {
    let surface = borrow_handle(surface, "WGPUSurface");
    let Some(config) = config.as_ref() else {
        return;
    };
    if config.device.is_null() {
        return;
    }
    let device = borrow_handle(config.device, "WGPUDevice");
    if surface.is_error {
        device.dispatch_error(core::ErrorKind::Validation, "surface is invalid");
        return;
    }
    if let Some(message) = surface_configuration_error(device, config) {
        device.dispatch_error(core::ErrorKind::Validation, message);
        return;
    }
    *surface
        .configured
        .lock()
        .expect("surface configuration lock is not poisoned") = Some(SurfaceConfigurationState {
        _format: config.format,
        _usage: config.usage,
        _width: config.width,
        _height: config.height,
        _present_mode: config.presentMode,
        _alpha_mode: config.alphaMode,
    });
}

/// Clears any stored surface configuration.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceUnconfigure(surface: native::WGPUSurface) {
    let surface = borrow_handle(surface, "WGPUSurface");
    *surface
        .configured
        .lock()
        .expect("surface configuration lock is not poisoned") = None;
}

/// Gets the current surface texture.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle. `surface_texture`,
/// when non-null, must point to writable memory.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceGetCurrentTexture(
    surface: native::WGPUSurface,
    surface_texture: *mut native::WGPUSurfaceTexture,
) {
    let surface = borrow_handle(surface, "WGPUSurface");
    let Some(surface_texture) = surface_texture.as_mut() else {
        return;
    };
    surface_texture.nextInChain = std::ptr::null_mut();
    surface_texture.texture = std::ptr::null();
    if surface.is_error
        || surface
            .configured
            .lock()
            .expect("surface configuration lock is not poisoned")
            .is_none()
    {
        surface_texture.status = native::WGPUSurfaceGetCurrentTextureStatus_Error;
        return;
    }
    // Noop has no native window/backbuffer, so a valid configuration still
    // cannot produce a swapchain image. This is the recorded SF3 N/A boundary.
    surface_texture.status = native::WGPUSurfaceGetCurrentTextureStatus_Lost;
}

/// Presents the current surface texture. Noop has no presentation backend.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfacePresent(surface: native::WGPUSurface) -> native::WGPUStatus {
    let _surface = borrow_handle(surface, "WGPUSurface");
    native::WGPUStatus_Success
}

/// Sets the debug label for a surface.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle. `label` must point
/// to valid string data according to `WGPUStringView` when non-empty.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceSetLabel(
    surface: native::WGPUSurface,
    label: native::WGPUStringView,
) {
    let surface = borrow_handle(surface, "WGPUSurface");
    *surface
        .label
        .lock()
        .expect("surface label lock is not poisoned") =
        label_from_string_view(label).unwrap_or_default();
}

/// Releases one owned reference to a surface handle.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceRelease(surface: native::WGPUSurface) {
    release_handle(surface, "WGPUSurface");
}

/// Adds one owned reference to a surface handle.
///
/// # Safety
///
/// `surface` must be a non-null live yawgpu surface handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuSurfaceAddRef(surface: native::WGPUSurface) {
    add_ref_handle(surface, "WGPUSurface");
}

/// Destroys a texture. This operation is idempotent.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureDestroy(texture: native::WGPUTexture) {
    borrow_handle(texture, "WGPUTexture").core.destroy();
}

/// Creates a view over a texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle. `descriptor`,
/// when non-null, must point to a valid `WGPUTextureViewDescriptor`.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureCreateView(
    texture: native::WGPUTexture,
    descriptor: *const native::WGPUTextureViewDescriptor,
) -> native::WGPUTextureView {
    let texture = borrow_handle(texture, "WGPUTexture");
    let descriptor = map_texture_view_descriptor(descriptor.as_ref());
    let (view, error) = texture.core.create_view(descriptor);
    if let Some(message) = error {
        texture
            .device
            .dispatch_error(core::ErrorKind::Validation, message);
    }
    arc_to_handle(Arc::new(WGPUTextureViewImpl {
        _core: Arc::new(view),
        _texture: Arc::clone(&texture.core),
        _device: Arc::clone(&texture.device),
        _instance: Arc::clone(&texture.instance),
    }))
}

/// Returns the descriptor format reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetFormat(
    texture: native::WGPUTexture,
) -> native::WGPUTextureFormat {
    map_texture_format_to_native(borrow_handle(texture, "WGPUTexture").core.format())
}

/// Returns the descriptor dimension reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetDimension(
    texture: native::WGPUTexture,
) -> native::WGPUTextureDimension {
    map_texture_dimension_to_native(borrow_handle(texture, "WGPUTexture").core.dimension())
}

/// Returns the descriptor width reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetWidth(texture: native::WGPUTexture) -> u32 {
    borrow_handle(texture, "WGPUTexture").core.size().width
}

/// Returns the descriptor height reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetHeight(texture: native::WGPUTexture) -> u32 {
    borrow_handle(texture, "WGPUTexture").core.size().height
}

/// Returns the descriptor depth/array-layer count reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetDepthOrArrayLayers(texture: native::WGPUTexture) -> u32 {
    borrow_handle(texture, "WGPUTexture")
        .core
        .size()
        .depth_or_array_layers
}

/// Returns the descriptor mip level count reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetMipLevelCount(texture: native::WGPUTexture) -> u32 {
    borrow_handle(texture, "WGPUTexture").core.mip_level_count()
}

/// Returns the descriptor sample count reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetSampleCount(texture: native::WGPUTexture) -> u32 {
    borrow_handle(texture, "WGPUTexture").core.sample_count()
}

/// Returns the descriptor usage reflected by the texture.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureGetUsage(
    texture: native::WGPUTexture,
) -> native::WGPUTextureUsage {
    map_texture_usage_to_native(borrow_handle(texture, "WGPUTexture").core.usage())
}

/// Releases one owned reference to a texture handle.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureRelease(texture: native::WGPUTexture) {
    release_handle(texture, "WGPUTexture");
}

/// Adds one owned reference to a texture handle.
///
/// # Safety
///
/// `texture` must be a non-null live yawgpu texture handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureAddRef(texture: native::WGPUTexture) {
    add_ref_handle(texture, "WGPUTexture");
}

/// Releases one owned reference to a texture view handle.
///
/// # Safety
///
/// `texture_view` must be a non-null live yawgpu texture view handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureViewRelease(texture_view: native::WGPUTextureView) {
    release_handle(texture_view, "WGPUTextureView");
}

/// Adds one owned reference to a texture view handle.
///
/// # Safety
///
/// `texture_view` must be a non-null live yawgpu texture view handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuTextureViewAddRef(texture_view: native::WGPUTextureView) {
    add_ref_handle(texture_view, "WGPUTextureView");
}

/// Releases one owned reference to a sampler handle.
///
/// # Safety
///
/// `sampler` must be a non-null live yawgpu sampler handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuSamplerRelease(sampler: native::WGPUSampler) {
    release_handle(sampler, "WGPUSampler");
}

/// Adds one owned reference to a sampler handle.
///
/// # Safety
///
/// `sampler` must be a non-null live yawgpu sampler handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuSamplerAddRef(sampler: native::WGPUSampler) {
    add_ref_handle(sampler, "WGPUSampler");
}

/// Requests compilation information for a shader module.
///
/// # Safety
///
/// `shader_module` must be a non-null live yawgpu shader module handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuShaderModuleGetCompilationInfo(
    shader_module: native::WGPUShaderModule,
    callback_info: native::WGPUCompilationInfoCallbackInfo,
) -> native::WGPUFuture {
    let shader_module = borrow_handle(shader_module, "WGPUShaderModule");
    shader_module
        ._instance
        .register_callback(PendingCallback::CompilationInfo {
            mode: callback_info.mode,
            callback: callback_info.callback,
            shader_module: Arc::clone(&shader_module._core),
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        })
}

/// Releases one owned reference to a shader module handle.
///
/// # Safety
///
/// `shader_module` must be a non-null live yawgpu shader module handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuShaderModuleRelease(shader_module: native::WGPUShaderModule) {
    release_handle(shader_module, "WGPUShaderModule");
}

/// Adds one owned reference to a shader module handle.
///
/// # Safety
///
/// `shader_module` must be a non-null live yawgpu shader module handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuShaderModuleAddRef(shader_module: native::WGPUShaderModule) {
    add_ref_handle(shader_module, "WGPUShaderModule");
}

/// Releases one owned reference to a queue handle.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueRelease(queue: native::WGPUQueue) {
    release_handle(queue, "WGPUQueue");
}

/// Adds one owned reference to a queue handle.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueAddRef(queue: native::WGPUQueue) {
    add_ref_handle(queue, "WGPUQueue");
}

/// Sets the debug label for a queue.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle. `label` must point to
/// valid string data according to `WGPUStringView` when non-empty.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueSetLabel(
    queue: native::WGPUQueue,
    label: native::WGPUStringView,
) {
    let queue = borrow_handle(queue, "WGPUQueue");
    let label = label_from_string_view(label).unwrap_or_default();
    queue.core.set_label(&label);
}

/// Schedules a callback once all submitted queue work is done.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle. `callback_info`
/// userdata pointers must remain valid until the callback fires.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueOnSubmittedWorkDone(
    queue: native::WGPUQueue,
    callback_info: native::WGPUQueueWorkDoneCallbackInfo,
) -> native::WGPUFuture {
    let queue = borrow_handle(queue, "WGPUQueue");
    queue
        .instance
        .register_callback(PendingCallback::QueueWorkDone {
            mode: callback_info.mode,
            callback: callback_info.callback,
            device: Arc::clone(&queue.device),
            status: core::QueueWorkDoneStatus::Success,
            userdata1: callback_info.userdata1 as usize,
            userdata2: callback_info.userdata2 as usize,
        })
}

/// Submits command buffers to a queue. Phase 2 validates only null arguments.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle. If `command_count` is
/// non-zero, `commands` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueSubmit(
    queue: native::WGPUQueue,
    command_count: usize,
    commands: *const native::WGPUCommandBuffer,
) {
    let queue = borrow_handle(queue, "WGPUQueue");
    if command_count > 0 && commands.is_null() {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue submit commands must not be null when commandCount is non-zero",
        );
        return;
    }
    let commands = if command_count == 0 {
        Some(Vec::new())
    } else {
        std::slice::from_raw_parts(commands, command_count)
            .iter()
            .map(|command| {
                let command = clone_handle::<WGPUCommandBufferImpl>(*command, "WGPUCommandBuffer");
                if !command._device.same(&queue.device) {
                    queue.device.dispatch_error(
                        core::ErrorKind::Validation,
                        "command buffer must belong to the queue device",
                    );
                    None
                } else {
                    Some(Arc::clone(&command.core))
                }
            })
            .collect::<Option<Vec<_>>>()
    };
    let Some(commands) = commands else {
        return;
    };
    dispatch_optional_device_error(&queue.device, queue.core.submit(&commands));
}

/// Writes CPU data into a buffer through the queue.
///
/// # Safety
///
/// `queue` and `buffer` must be non-null live yawgpu handles. `data` must
/// point to `size` bytes when `size` is non-zero.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueWriteBuffer(
    queue: native::WGPUQueue,
    buffer: native::WGPUBuffer,
    buffer_offset: u64,
    data: *const c_void,
    size: usize,
) {
    let queue = borrow_handle(queue, "WGPUQueue");
    let buffer = borrow_handle(buffer, "WGPUBuffer");
    if !buffer.device.same(&queue.device) {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue write buffer target must belong to the queue device",
        );
        return;
    }
    if size > 0 && data.is_null() {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue write data must not be null when size is non-zero",
        );
        return;
    }
    let data = std::slice::from_raw_parts(data.cast::<u8>(), size);
    dispatch_optional_device_error(
        &queue.device,
        queue.core.write_buffer(&buffer.core, buffer_offset, data),
    );
}

/// Validates a queue texture write. Noop does not copy bytes.
///
/// # Safety
///
/// `queue` must be a non-null live yawgpu queue handle. `destination`,
/// `data_layout`, and `write_size` must be non-null pointers to valid WebGPU
/// structs. `destination.texture` must be a non-null live yawgpu texture
/// handle. `data` is not read by the Noop validation implementation.
#[no_mangle]
pub unsafe extern "C" fn wgpuQueueWriteTexture(
    queue: native::WGPUQueue,
    destination: *const native::WGPUTexelCopyTextureInfo,
    _data: *const c_void,
    data_size: usize,
    data_layout: *const native::WGPUTexelCopyBufferLayout,
    write_size: *const native::WGPUExtent3D,
) {
    let queue = borrow_handle(queue, "WGPUQueue");
    let Some(destination) = destination.as_ref() else {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue write texture destination must not be null",
        );
        return;
    };
    let Some(data_layout) = data_layout.as_ref() else {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue write texture dataLayout must not be null",
        );
        return;
    };
    let Some(write_size) = write_size.as_ref() else {
        queue.device.dispatch_error(
            core::ErrorKind::Validation,
            "queue write texture writeSize must not be null",
        );
        return;
    };
    let data_size = match u64::try_from(data_size) {
        Ok(size) => size,
        Err(_) => {
            queue.device.dispatch_error(
                core::ErrorKind::Validation,
                "queue write texture dataSize is too large",
            );
            return;
        }
    };
    let texture = borrow_handle(destination.texture, "WGPUTexture");
    let aspect = map_texture_aspect(destination.aspect).unwrap_or(core::TextureAspect::All);

    if let Err(message) = texture.core.validate_queue_write(
        destination.mipLevel,
        map_origin_3d(destination.origin),
        map_extent_3d(*write_size),
        aspect,
        map_texel_copy_buffer_layout(*data_layout),
        data_size,
    ) {
        queue
            .device
            .dispatch_error(core::ErrorKind::Validation, message);
    }
}

/// Frees a feature array returned by `wgpuAdapterGetFeatures` or
/// `wgpuDeviceGetFeatures`.
///
/// # Safety
///
/// `supported_features.features`, when non-null, must be a pointer previously
/// returned by yawgpu from `wgpuAdapterGetFeatures` or
/// `wgpuDeviceGetFeatures`, paired with the same `featureCount`, and must not
/// be freed more than once.
#[no_mangle]
pub unsafe extern "C" fn wgpuSupportedFeaturesFreeMembers(
    supported_features: native::WGPUSupportedFeatures,
) {
    free_supported_features(supported_features);
}

/// Processes callbacks whose mode allows process-events delivery.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceProcessEvents(instance: native::WGPUInstance) {
    let instance = borrow_handle(instance, "WGPUInstance");
    instance.process_callbacks();
}

/// Waits for any listed future and fires callbacks for completed futures.
///
/// # Safety
///
/// `instance` must be a non-null live yawgpu instance handle. If
/// `future_count` is non-zero, `futures` must point to `future_count` valid
/// `WGPUFutureWaitInfo` entries.
#[no_mangle]
pub unsafe extern "C" fn wgpuInstanceWaitAny(
    instance: native::WGPUInstance,
    future_count: usize,
    futures: *mut native::WGPUFutureWaitInfo,
    timeout_ns: u64,
) -> native::WGPUWaitStatus {
    let instance = borrow_handle(instance, "WGPUInstance");
    if future_count > 0 && futures.is_null() {
        return native::WGPUWaitStatus_Error;
    }
    if future_count == 0 {
        return native::WGPUWaitStatus_TimedOut;
    }
    if timeout_ns > 0 && !instance.timed_wait_any_enabled {
        return native::WGPUWaitStatus_Error;
    }

    let wait_infos = std::slice::from_raw_parts_mut(futures, future_count);
    let future_ids = wait_infos
        .iter()
        .map(|info| core::FutureId::from_raw(info.future.id))
        .collect::<Vec<_>>();
    let result = instance.wait_any(&future_ids);

    for info in wait_infos {
        let id = core::FutureId::from_raw(info.future.id);
        info.completed = u32::from(result.completed.contains(&id));
    }

    match result.status {
        core::WaitAnyStatus::Success => native::WGPUWaitStatus_Success,
        core::WaitAnyStatus::TimedOut => native::WGPUWaitStatus_TimedOut,
        core::WaitAnyStatus::Error => native::WGPUWaitStatus_Error,
        _ => native::WGPUWaitStatus_Error,
    }
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
        if node.sType == WGPU_STYPE_YAWGPU_INSTANCE_BACKEND_SELECT {
            let selection = &*(node as *const native::WGPUChainedStruct
                as *const WGPUYawgpuInstanceBackendSelect);
            return match selection.backend {
                WGPU_YAWGPU_INSTANCE_BACKEND_METAL => InstanceBackendSelection::Metal,
                WGPU_YAWGPU_INSTANCE_BACKEND_VULKAN => InstanceBackendSelection::Vulkan,
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
    let size = if size == native::WGPU_WHOLE_MAP_SIZE as usize {
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
    let size = if size == native::WGPU_WHOLE_MAP_SIZE as usize {
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
    use super::*;

    unsafe extern "C" fn request_adapter_callback(
        status: native::WGPURequestAdapterStatus,
        adapter: native::WGPUAdapter,
        _message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        assert_eq!(status, native::WGPURequestAdapterStatus_Success);
        *(userdata1 as *mut native::WGPUAdapter) = adapter;
    }

    unsafe extern "C" fn request_device_callback(
        status: native::WGPURequestDeviceStatus,
        device: native::WGPUDevice,
        _message: native::WGPUStringView,
        userdata1: *mut c_void,
        _userdata2: *mut c_void,
    ) {
        assert_eq!(status, native::WGPURequestDeviceStatus_Success);
        *(userdata1 as *mut native::WGPUDevice) = device;
    }

    #[test]
    fn instance_add_ref_release_balances_core_arc() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());
            let core = Arc::clone(&borrow_handle(instance, "WGPUInstance").core);
            assert_eq!(Arc::strong_count(&core), 2);

            wgpuInstanceAddRef(instance);
            assert_eq!(Arc::strong_count(&core), 2);

            wgpuInstanceRelease(instance);
            assert_eq!(Arc::strong_count(&core), 2);

            wgpuInstanceRelease(instance);
            assert_eq!(Arc::strong_count(&core), 1);
        }
    }

    #[test]
    fn noop_request_adapter_request_device_process_events_round_trip() {
        unsafe {
            let instance = wgpuCreateInstance(std::ptr::null());
            let mut adapter: native::WGPUAdapter = std::ptr::null();

            let adapter_callback_info = native::WGPURequestAdapterCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(request_adapter_callback),
                userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let future =
                wgpuInstanceRequestAdapter(instance, std::ptr::null(), adapter_callback_info);
            assert_ne!(future.id, 0);
            assert!(adapter.is_null());

            wgpuInstanceProcessEvents(instance);
            assert!(!adapter.is_null());

            let mut device: native::WGPUDevice = std::ptr::null();
            let device_callback_info = native::WGPURequestDeviceCallbackInfo {
                nextInChain: std::ptr::null_mut(),
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: Some(request_device_callback),
                userdata1: (&mut device as *mut native::WGPUDevice).cast(),
                userdata2: std::ptr::null_mut(),
            };
            let future = wgpuAdapterRequestDevice(adapter, std::ptr::null(), device_callback_info);
            assert_ne!(future.id, 0);
            assert!(device.is_null());

            wgpuInstanceProcessEvents(instance);
            assert!(!device.is_null());

            let queue = wgpuDeviceGetQueue(device);
            assert!(!queue.is_null());

            wgpuQueueRelease(queue);
            wgpuDeviceRelease(device);
            wgpuAdapterRelease(adapter);
            wgpuInstanceRelease(instance);
        }
    }
}
