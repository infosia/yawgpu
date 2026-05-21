use super::*;

pub fn map_query_type(value: native::WGPUQueryType) -> core::QueryType {
    value.into()
}

#[must_use]
pub fn map_query_type_to_native(value: core::QueryType) -> native::WGPUQueryType {
    value.into()
}

#[must_use]
pub fn map_feature_level(value: native::WGPUFeatureLevel) -> core::FeatureLevel {
    match value {
        native::WGPUFeatureLevel_Compatibility => core::FeatureLevel::Compatibility,
        _ => core::FeatureLevel::Core,
    }
}

impl DeviceLostCallbackInfo {
    #[must_use]
    pub fn has_callback(self) -> bool {
        self.callback.is_some()
    }
}

#[must_use]
pub fn map_error_filter(value: native::WGPUErrorFilter) -> Option<core::ErrorFilter> {
    match value {
        native::WGPUErrorFilter_Validation => Some(core::ErrorFilter::Validation),
        native::WGPUErrorFilter_OutOfMemory => Some(core::ErrorFilter::OutOfMemory),
        native::WGPUErrorFilter_Internal => Some(core::ErrorFilter::Internal),
        _ => None,
    }
}

#[must_use]
pub fn map_error_type(kind: core::ErrorKind) -> native::WGPUErrorType {
    match kind {
        core::ErrorKind::Validation => native::WGPUErrorType_Validation,
        core::ErrorKind::OutOfMemory => native::WGPUErrorType_OutOfMemory,
        core::ErrorKind::Internal => native::WGPUErrorType_Internal,
        _ => native::WGPUErrorType_Unknown,
    }
}

#[must_use]
pub fn map_pop_error_scope_status_error() -> native::WGPUPopErrorScopeStatus {
    native::WGPUPopErrorScopeStatus_Error
}

#[must_use]
pub fn map_pop_error_scope_status_success() -> native::WGPUPopErrorScopeStatus {
    native::WGPUPopErrorScopeStatus_Success
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

#[must_use]
pub fn map_compilation_info_request_status_success() -> native::WGPUCompilationInfoRequestStatus {
    native::WGPUCompilationInfoRequestStatus_Success
}

#[must_use]
pub fn map_compilation_message_type_error() -> native::WGPUCompilationMessageType {
    native::WGPUCompilationMessageType_Error
}

pub fn map_map_mode(value: native::WGPUMapMode) -> Result<core::MapMode, &'static str> {
    let bits = u32::try_from(value).map_err(|_| "map mode has unsupported bits")?;
    core::MapMode::from_bits(bits)
}

#[must_use]
pub fn map_address_mode(value: native::WGPUAddressMode) -> Option<core::AddressMode> {
    match value {
        native::WGPUAddressMode_Undefined => None,
        native::WGPUAddressMode_ClampToEdge => Some(core::AddressMode::ClampToEdge),
        native::WGPUAddressMode_Repeat => Some(core::AddressMode::Repeat),
        native::WGPUAddressMode_MirrorRepeat => Some(core::AddressMode::MirrorRepeat),
        _ => None,
    }
}

#[must_use]
pub fn map_filter_mode(value: native::WGPUFilterMode) -> Option<core::FilterMode> {
    match value {
        native::WGPUFilterMode_Undefined => None,
        native::WGPUFilterMode_Nearest => Some(core::FilterMode::Nearest),
        native::WGPUFilterMode_Linear => Some(core::FilterMode::Linear),
        _ => None,
    }
}

#[must_use]
pub fn map_mipmap_filter_mode(
    value: native::WGPUMipmapFilterMode,
) -> Option<core::MipmapFilterMode> {
    match value {
        native::WGPUMipmapFilterMode_Undefined => None,
        native::WGPUMipmapFilterMode_Nearest => Some(core::MipmapFilterMode::Nearest),
        native::WGPUMipmapFilterMode_Linear => Some(core::MipmapFilterMode::Linear),
        _ => None,
    }
}

#[must_use]
pub fn map_compare_function(value: native::WGPUCompareFunction) -> Option<core::CompareFunction> {
    match value {
        native::WGPUCompareFunction_Undefined => None,
        native::WGPUCompareFunction_Never => Some(core::CompareFunction::Never),
        native::WGPUCompareFunction_Less => Some(core::CompareFunction::Less),
        native::WGPUCompareFunction_Equal => Some(core::CompareFunction::Equal),
        native::WGPUCompareFunction_LessEqual => Some(core::CompareFunction::LessEqual),
        native::WGPUCompareFunction_Greater => Some(core::CompareFunction::Greater),
        native::WGPUCompareFunction_NotEqual => Some(core::CompareFunction::NotEqual),
        native::WGPUCompareFunction_GreaterEqual => Some(core::CompareFunction::GreaterEqual),
        native::WGPUCompareFunction_Always => Some(core::CompareFunction::Always),
        _ => None,
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
pub fn map_query_index(value: u32) -> Option<u32> {
    (value != native::WGPU_QUERY_SET_INDEX_UNDEFINED).then_some(value)
}

#[must_use]
pub fn map_load_op(value: native::WGPULoadOp) -> core::LoadOp {
    match value {
        native::WGPULoadOp_Load => core::LoadOp::Load,
        native::WGPULoadOp_Clear => core::LoadOp::Clear,
        _ => core::LoadOp::Undefined,
    }
}

#[must_use]
pub fn map_store_op(value: native::WGPUStoreOp) -> core::StoreOp {
    match value {
        native::WGPUStoreOp_Store => core::StoreOp::Store,
        native::WGPUStoreOp_Discard => core::StoreOp::Discard,
        _ => core::StoreOp::Undefined,
    }
}

#[must_use]
pub fn map_texture_view_dimension(
    value: native::WGPUTextureViewDimension,
) -> Option<core::TextureViewDimension> {
    match value {
        native::WGPUTextureViewDimension_Undefined => None,
        native::WGPUTextureViewDimension_1D => Some(core::TextureViewDimension::D1),
        native::WGPUTextureViewDimension_2D => Some(core::TextureViewDimension::D2),
        native::WGPUTextureViewDimension_2DArray => Some(core::TextureViewDimension::D2Array),
        native::WGPUTextureViewDimension_Cube => Some(core::TextureViewDimension::Cube),
        native::WGPUTextureViewDimension_CubeArray => Some(core::TextureViewDimension::CubeArray),
        native::WGPUTextureViewDimension_3D => Some(core::TextureViewDimension::D3),
        _ => None,
    }
}

#[must_use]
pub fn map_texture_aspect(value: native::WGPUTextureAspect) -> Option<core::TextureAspect> {
    match value {
        native::WGPUTextureAspect_Undefined => None,
        native::WGPUTextureAspect_All => Some(core::TextureAspect::All),
        native::WGPUTextureAspect_DepthOnly => Some(core::TextureAspect::DepthOnly),
        native::WGPUTextureAspect_StencilOnly => Some(core::TextureAspect::StencilOnly),
        _ => None,
    }
}
