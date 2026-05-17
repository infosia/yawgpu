use std::ffi::CStr;
use std::sync::Arc;

use crate::native;
use yawgpu_core as core;

pub const WGPU_STRLEN: usize = usize::MAX;

/// Handle refcount contract:
/// - create/request functions return one owned C reference (+1) via `Arc::into_raw`.
/// - `wgpuXxxAddRef` borrows the handle, clones the `Arc`, and leaks that clone (+1).
/// - `wgpuXxxRelease` reconstructs one `Arc` with `Arc::from_raw` and drops it (-1).
#[must_use]
pub fn arc_to_handle<T>(value: Arc<T>) -> *const T {
    Arc::into_raw(value)
}

/// Drops one owned C reference for a yawgpu handle.
///
/// # Safety
///
/// `handle` must be a non-null pointer returned by `Arc::into_raw` for `T`.
/// It must represent one currently owned C reference.
pub unsafe fn release_handle<T>(handle: *const T, name: &str) {
    let handle = handle
        .as_ref()
        .map(|_| handle)
        .unwrap_or_else(|| panic!("{name} must not be null"));
    drop(Arc::from_raw(handle));
}

/// Clones one C handle reference without consuming the incoming handle.
///
/// # Safety
///
/// `handle` must be a non-null live pointer returned by `Arc::into_raw` for
/// `T`.
pub unsafe fn add_ref_handle<T>(handle: *const T, name: &str) {
    let handle = handle
        .as_ref()
        .map(|_| handle)
        .unwrap_or_else(|| panic!("{name} must not be null"));
    Arc::increment_strong_count(handle);
}

#[must_use]
/// Clones a C handle into a Rust `Arc`.
///
/// # Safety
///
/// `handle` must be a non-null live pointer returned by `Arc::into_raw` for
/// `T`.
pub unsafe fn clone_handle<T>(handle: *const T, name: &str) -> Arc<T> {
    let handle = handle
        .as_ref()
        .map(|_| handle)
        .unwrap_or_else(|| panic!("{name} must not be null"));
    Arc::increment_strong_count(handle);
    Arc::from_raw(handle)
}

/// Borrows a C handle without changing its reference count.
///
/// # Safety
///
/// `handle` must be a non-null live pointer returned by `Arc::into_raw` for
/// `T`, and the returned borrow must not outlive the owned C reference.
pub unsafe fn borrow_handle<'a, T>(handle: *const T, name: &str) -> &'a T {
    handle
        .as_ref()
        .unwrap_or_else(|| panic!("{name} must not be null"))
}

#[must_use]
pub fn string_view(data: &[u8]) -> native::WGPUStringView {
    native::WGPUStringView {
        data: data.as_ptr().cast(),
        length: data.len(),
    }
}

#[must_use]
/// Converts a `WGPUStringView` to UTF-8 text.
///
/// # Safety
///
/// `value.data`, when non-null, must point to a valid byte buffer for
/// `value.length` bytes, or to a valid NUL-terminated C string when
/// `value.length == WGPU_STRLEN`.
pub unsafe fn string_view_to_str<'a>(value: native::WGPUStringView) -> Option<&'a str> {
    if value.data.is_null() {
        return None;
    }

    let bytes = if value.length == WGPU_STRLEN {
        CStr::from_ptr(value.data).to_bytes()
    } else {
        std::slice::from_raw_parts(value.data.cast::<u8>(), value.length)
    };

    std::str::from_utf8(bytes).ok()
}

#[must_use]
/// Converts a label string view to an owned string.
///
/// # Safety
///
/// Same requirements as [`string_view_to_str`].
pub unsafe fn label_from_string_view(value: native::WGPUStringView) -> Option<String> {
    string_view_to_str(value).map(ToOwned::to_owned)
}

#[must_use]
pub fn map_feature(value: native::WGPUFeatureName) -> core::Feature {
    match value {
        native::WGPUFeatureName_CoreFeaturesAndLimits => core::Feature::CoreFeaturesAndLimits,
        native::WGPUFeatureName_RG11B10UfloatRenderable => core::Feature::Rg11b10UfloatRenderable,
        native::WGPUFeatureName_TextureFormatsTier1 => core::Feature::TextureFormatsTier1,
        native::WGPUFeatureName_TextureFormatsTier2 => core::Feature::TextureFormatsTier2,
        other => core::Feature::Other(other),
    }
}

#[must_use]
pub fn map_feature_to_native(value: core::Feature) -> native::WGPUFeatureName {
    match value {
        core::Feature::CoreFeaturesAndLimits => native::WGPUFeatureName_CoreFeaturesAndLimits,
        core::Feature::Rg11b10UfloatRenderable => native::WGPUFeatureName_RG11B10UfloatRenderable,
        core::Feature::TextureFormatsTier1 => native::WGPUFeatureName_TextureFormatsTier1,
        core::Feature::TextureFormatsTier2 => native::WGPUFeatureName_TextureFormatsTier2,
        core::Feature::Other(value) => value,
        // exhaustive as of core::Feature @ 2026-05-17
        _ => native::WGPUFeatureName_Force32,
    }
}

#[must_use]
pub fn map_feature_level(value: native::WGPUFeatureLevel) -> core::FeatureLevel {
    match value {
        native::WGPUFeatureLevel_Compatibility => core::FeatureLevel::Compatibility,
        _ => core::FeatureLevel::Core,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceLostCallbackInfo {
    pub mode: native::WGPUCallbackMode,
    pub callback: native::WGPUDeviceLostCallback,
    pub userdata1: usize,
    pub userdata2: usize,
}

impl DeviceLostCallbackInfo {
    #[must_use]
    pub fn has_callback(self) -> bool {
        self.callback.is_some()
    }
}

#[must_use]
pub fn map_device_lost_callback_info(
    value: native::WGPUDeviceLostCallbackInfo,
) -> DeviceLostCallbackInfo {
    DeviceLostCallbackInfo {
        mode: value.mode,
        callback: value.callback,
        userdata1: value.userdata1 as usize,
        userdata2: value.userdata2 as usize,
    }
}

#[must_use]
pub fn map_device_lost_reason(reason: core::DeviceLostReason) -> native::WGPUDeviceLostReason {
    match reason {
        core::DeviceLostReason::Unknown => native::WGPUDeviceLostReason_Unknown,
        core::DeviceLostReason::Destroyed => native::WGPUDeviceLostReason_Destroyed,
        core::DeviceLostReason::CallbackCancelled => native::WGPUDeviceLostReason_CallbackCancelled,
        core::DeviceLostReason::FailedCreation => native::WGPUDeviceLostReason_FailedCreation,
        // exhaustive as of core::DeviceLostReason @ 2026-05-17
        _ => native::WGPUDeviceLostReason_Unknown,
    }
}

#[must_use]
pub fn map_buffer_usage(value: native::WGPUBufferUsage) -> core::BufferUsage {
    core::BufferUsage::from_bits_retain(value)
}

#[must_use]
pub fn map_buffer_usage_to_native(value: core::BufferUsage) -> native::WGPUBufferUsage {
    value.bits()
}

#[must_use]
pub fn map_buffer_map_state(value: core::BufferMapState) -> native::WGPUBufferMapState {
    match value {
        core::BufferMapState::Unmapped => native::WGPUBufferMapState_Unmapped,
        core::BufferMapState::Pending => native::WGPUBufferMapState_Pending,
        core::BufferMapState::Mapped => native::WGPUBufferMapState_Mapped,
        // exhaustive as of core::BufferMapState @ 2026-05-17
        _ => native::WGPUBufferMapState_Force32,
    }
}

#[must_use]
pub fn map_map_async_status(value: core::MapAsyncStatus) -> native::WGPUMapAsyncStatus {
    match value {
        core::MapAsyncStatus::Success => native::WGPUMapAsyncStatus_Success,
        core::MapAsyncStatus::CallbackCancelled => native::WGPUMapAsyncStatus_CallbackCancelled,
        core::MapAsyncStatus::Error => native::WGPUMapAsyncStatus_Error,
        core::MapAsyncStatus::Aborted => native::WGPUMapAsyncStatus_Aborted,
        // exhaustive as of core::MapAsyncStatus @ 2026-05-17
        _ => native::WGPUMapAsyncStatus_Error,
    }
}

#[must_use]
pub fn map_queue_work_done_status(
    value: core::QueueWorkDoneStatus,
) -> native::WGPUQueueWorkDoneStatus {
    match value {
        core::QueueWorkDoneStatus::Success => native::WGPUQueueWorkDoneStatus_Success,
        core::QueueWorkDoneStatus::CallbackCancelled => {
            native::WGPUQueueWorkDoneStatus_CallbackCancelled
        }
        core::QueueWorkDoneStatus::Error => native::WGPUQueueWorkDoneStatus_Error,
        // exhaustive as of core::QueueWorkDoneStatus @ 2026-05-17
        _ => native::WGPUQueueWorkDoneStatus_Error,
    }
}

pub fn map_map_mode(value: native::WGPUMapMode) -> Result<core::MapMode, &'static str> {
    let bits = u32::try_from(value).map_err(|_| "map mode has unsupported bits")?;
    core::MapMode::from_bits(bits)
}

#[must_use]
pub fn map_buffer_descriptor(value: &native::WGPUBufferDescriptor) -> core::BufferDescriptor {
    core::BufferDescriptor {
        usage: map_buffer_usage(value.usage),
        size: value.size,
        mapped_at_creation: value.mappedAtCreation != 0,
    }
}

#[must_use]
pub fn map_texture_usage(value: native::WGPUTextureUsage) -> core::TextureUsage {
    core::TextureUsage::from_bits_retain(value)
}

#[must_use]
pub fn map_texture_usage_to_native(value: core::TextureUsage) -> native::WGPUTextureUsage {
    value.bits()
}

#[must_use]
pub fn map_texture_dimension(value: native::WGPUTextureDimension) -> core::TextureDimension {
    match value {
        native::WGPUTextureDimension_1D => core::TextureDimension::D1,
        native::WGPUTextureDimension_3D => core::TextureDimension::D3,
        _ => core::TextureDimension::D2,
    }
}

#[must_use]
pub fn map_texture_dimension_to_native(
    value: core::TextureDimension,
) -> native::WGPUTextureDimension {
    match value {
        core::TextureDimension::D1 => native::WGPUTextureDimension_1D,
        core::TextureDimension::D2 => native::WGPUTextureDimension_2D,
        core::TextureDimension::D3 => native::WGPUTextureDimension_3D,
        // exhaustive as of core::TextureDimension @ 2026-05-17
        _ => native::WGPUTextureDimension_2D,
    }
}

#[must_use]
pub fn map_texture_format(value: native::WGPUTextureFormat) -> core::TextureFormat {
    core::TextureFormat::from_raw(value)
}

#[must_use]
pub fn map_texture_format_to_native(value: core::TextureFormat) -> native::WGPUTextureFormat {
    value.raw()
}

#[must_use]
pub fn map_extent_3d(value: native::WGPUExtent3D) -> core::Extent3d {
    core::Extent3d {
        width: value.width,
        height: value.height,
        depth_or_array_layers: value.depthOrArrayLayers,
    }
}

#[must_use]
pub fn map_texture_descriptor(value: &native::WGPUTextureDescriptor) -> core::TextureDescriptor {
    core::TextureDescriptor {
        usage: map_texture_usage(value.usage),
        dimension: map_texture_dimension(value.dimension),
        size: map_extent_3d(value.size),
        format: map_texture_format(value.format),
        mip_level_count: value.mipLevelCount,
        sample_count: value.sampleCount,
    }
}

pub fn map_features_to_native(features: &core::FeatureSet) -> native::WGPUSupportedFeatures {
    let features = features
        .iter()
        .copied()
        .map(map_feature_to_native)
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let feature_count = features.len();
    let features = Box::into_raw(features);

    native::WGPUSupportedFeatures {
        featureCount: feature_count,
        features: features.cast(),
    }
}

/// Frees the feature array allocated by `map_features_to_native`.
///
/// # Safety
///
/// `features.features`, when non-null, must be a pointer previously returned
/// by `map_features_to_native` with the same `featureCount`.
pub unsafe fn free_supported_features(features: native::WGPUSupportedFeatures) {
    if features.features.is_null() {
        return;
    }
    let slice =
        std::ptr::slice_from_raw_parts_mut(features.features.cast_mut(), features.featureCount);
    drop(Box::from_raw(slice));
}

#[must_use]
pub fn map_limits_to_native(limits: core::Limits) -> native::WGPULimits {
    native::WGPULimits {
        nextInChain: std::ptr::null_mut(),
        maxTextureDimension1D: limits.max_texture_dimension_1d,
        maxTextureDimension2D: limits.max_texture_dimension_2d,
        maxTextureDimension3D: limits.max_texture_dimension_3d,
        maxTextureArrayLayers: limits.max_texture_array_layers,
        maxBindGroups: limits.max_bind_groups,
        maxBindGroupsPlusVertexBuffers: limits.max_bind_groups_plus_vertex_buffers,
        maxBindingsPerBindGroup: limits.max_bindings_per_bind_group,
        maxDynamicUniformBuffersPerPipelineLayout: limits
            .max_dynamic_uniform_buffers_per_pipeline_layout,
        maxDynamicStorageBuffersPerPipelineLayout: limits
            .max_dynamic_storage_buffers_per_pipeline_layout,
        maxSampledTexturesPerShaderStage: limits.max_sampled_textures_per_shader_stage,
        maxSamplersPerShaderStage: limits.max_samplers_per_shader_stage,
        maxStorageBuffersPerShaderStage: limits.max_storage_buffers_per_shader_stage,
        maxStorageTexturesPerShaderStage: limits.max_storage_textures_per_shader_stage,
        maxUniformBuffersPerShaderStage: limits.max_uniform_buffers_per_shader_stage,
        maxUniformBufferBindingSize: limits.max_uniform_buffer_binding_size,
        maxStorageBufferBindingSize: limits.max_storage_buffer_binding_size,
        minUniformBufferOffsetAlignment: limits.min_uniform_buffer_offset_alignment,
        minStorageBufferOffsetAlignment: limits.min_storage_buffer_offset_alignment,
        maxVertexBuffers: limits.max_vertex_buffers,
        maxBufferSize: limits.max_buffer_size,
        maxVertexAttributes: limits.max_vertex_attributes,
        maxVertexBufferArrayStride: limits.max_vertex_buffer_array_stride,
        maxInterStageShaderVariables: limits.max_inter_stage_shader_variables,
        maxColorAttachments: limits.max_color_attachments,
        maxColorAttachmentBytesPerSample: limits.max_color_attachment_bytes_per_sample,
        maxComputeWorkgroupStorageSize: limits.max_compute_workgroup_storage_size,
        maxComputeInvocationsPerWorkgroup: limits.max_compute_invocations_per_workgroup,
        maxComputeWorkgroupSizeX: limits.max_compute_workgroup_size_x,
        maxComputeWorkgroupSizeY: limits.max_compute_workgroup_size_y,
        maxComputeWorkgroupSizeZ: limits.max_compute_workgroup_size_z,
        maxComputeWorkgroupsPerDimension: limits.max_compute_workgroups_per_dimension,
        maxImmediateSize: limits.max_immediate_size,
    }
}

#[must_use]
pub fn map_limits(value: &native::WGPULimits) -> core::Limits {
    let default = core::Limits::DEFAULT;
    let mut limits = default;
    limits.max_texture_dimension_1d = limit_u32(
        value.maxTextureDimension1D,
        default.max_texture_dimension_1d,
    );
    limits.max_texture_dimension_2d = limit_u32(
        value.maxTextureDimension2D,
        default.max_texture_dimension_2d,
    );
    limits.max_texture_dimension_3d = limit_u32(
        value.maxTextureDimension3D,
        default.max_texture_dimension_3d,
    );
    limits.max_texture_array_layers = limit_u32(
        value.maxTextureArrayLayers,
        default.max_texture_array_layers,
    );
    limits.max_bind_groups = limit_u32(value.maxBindGroups, default.max_bind_groups);
    limits.max_bind_groups_plus_vertex_buffers = limit_u32(
        value.maxBindGroupsPlusVertexBuffers,
        default.max_bind_groups_plus_vertex_buffers,
    );
    limits.max_bindings_per_bind_group = limit_u32(
        value.maxBindingsPerBindGroup,
        default.max_bindings_per_bind_group,
    );
    limits.max_dynamic_uniform_buffers_per_pipeline_layout = limit_u32(
        value.maxDynamicUniformBuffersPerPipelineLayout,
        default.max_dynamic_uniform_buffers_per_pipeline_layout,
    );
    limits.max_dynamic_storage_buffers_per_pipeline_layout = limit_u32(
        value.maxDynamicStorageBuffersPerPipelineLayout,
        default.max_dynamic_storage_buffers_per_pipeline_layout,
    );
    limits.max_sampled_textures_per_shader_stage = limit_u32(
        value.maxSampledTexturesPerShaderStage,
        default.max_sampled_textures_per_shader_stage,
    );
    limits.max_samplers_per_shader_stage = limit_u32(
        value.maxSamplersPerShaderStage,
        default.max_samplers_per_shader_stage,
    );
    limits.max_storage_buffers_per_shader_stage = limit_u32(
        value.maxStorageBuffersPerShaderStage,
        default.max_storage_buffers_per_shader_stage,
    );
    limits.max_storage_textures_per_shader_stage = limit_u32(
        value.maxStorageTexturesPerShaderStage,
        default.max_storage_textures_per_shader_stage,
    );
    limits.max_uniform_buffers_per_shader_stage = limit_u32(
        value.maxUniformBuffersPerShaderStage,
        default.max_uniform_buffers_per_shader_stage,
    );
    limits.max_uniform_buffer_binding_size = limit_u64(
        value.maxUniformBufferBindingSize,
        default.max_uniform_buffer_binding_size,
    );
    limits.max_storage_buffer_binding_size = limit_u64(
        value.maxStorageBufferBindingSize,
        default.max_storage_buffer_binding_size,
    );
    limits.min_uniform_buffer_offset_alignment = limit_u32(
        value.minUniformBufferOffsetAlignment,
        default.min_uniform_buffer_offset_alignment,
    );
    limits.min_storage_buffer_offset_alignment = limit_u32(
        value.minStorageBufferOffsetAlignment,
        default.min_storage_buffer_offset_alignment,
    );
    limits.max_vertex_buffers = limit_u32(value.maxVertexBuffers, default.max_vertex_buffers);
    limits.max_buffer_size = limit_u64(value.maxBufferSize, default.max_buffer_size);
    limits.max_vertex_attributes =
        limit_u32(value.maxVertexAttributes, default.max_vertex_attributes);
    limits.max_vertex_buffer_array_stride = limit_u32(
        value.maxVertexBufferArrayStride,
        default.max_vertex_buffer_array_stride,
    );
    limits.max_inter_stage_shader_variables = limit_u32(
        value.maxInterStageShaderVariables,
        default.max_inter_stage_shader_variables,
    );
    limits.max_color_attachments =
        limit_u32(value.maxColorAttachments, default.max_color_attachments);
    limits.max_color_attachment_bytes_per_sample = limit_u32(
        value.maxColorAttachmentBytesPerSample,
        default.max_color_attachment_bytes_per_sample,
    );
    limits.max_compute_workgroup_storage_size = limit_u32(
        value.maxComputeWorkgroupStorageSize,
        default.max_compute_workgroup_storage_size,
    );
    limits.max_compute_invocations_per_workgroup = limit_u32(
        value.maxComputeInvocationsPerWorkgroup,
        default.max_compute_invocations_per_workgroup,
    );
    limits.max_compute_workgroup_size_x = limit_u32(
        value.maxComputeWorkgroupSizeX,
        default.max_compute_workgroup_size_x,
    );
    limits.max_compute_workgroup_size_y = limit_u32(
        value.maxComputeWorkgroupSizeY,
        default.max_compute_workgroup_size_y,
    );
    limits.max_compute_workgroup_size_z = limit_u32(
        value.maxComputeWorkgroupSizeZ,
        default.max_compute_workgroup_size_z,
    );
    limits.max_compute_workgroups_per_dimension = limit_u32(
        value.maxComputeWorkgroupsPerDimension,
        default.max_compute_workgroups_per_dimension,
    );
    limits.max_immediate_size = limit_u32(value.maxImmediateSize, default.max_immediate_size);
    limits
}

#[must_use]
fn limit_u32(value: u32, default: u32) -> u32 {
    if value == native::WGPU_LIMIT_U32_UNDEFINED {
        default
    } else {
        value
    }
}

#[must_use]
fn limit_u64(value: u64, default: u64) -> u64 {
    if value == native::WGPU_LIMIT_U64_UNDEFINED as u64 {
        default
    } else {
        value
    }
}
