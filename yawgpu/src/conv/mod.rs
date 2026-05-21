#![allow(dead_code, unused_imports)]

mod bind;
mod descriptors;
mod enums;
mod feature;
mod features;
mod format;
mod limits;
mod pipeline;
mod shader;
mod strings;

pub use bind::*;
pub use descriptors::*;
pub use enums::*;
pub use feature::*;
pub use features::*;
pub use format::*;
pub use limits::*;
pub use pipeline::*;
pub use shader::*;
pub use strings::*;

use std::sync::Arc;

use yawgpu_core as core;

use crate::native;

pub const WGPU_STRLEN: usize = usize::MAX;

fn set_first_error(error: &mut Option<String>, message: &str) {
    if error.is_none() {
        *error = Some(message.to_owned());
    }
}

use crate::{
    WGPUBindGroupLayoutImpl, WGPUBufferImpl, WGPUPipelineLayoutImpl, WGPUQuerySetImpl,
    WGPUSamplerImpl, WGPUShaderModuleImpl, WGPUTextureViewImpl,
};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        WGPUBindGroupLayoutImpl, WGPUBufferImpl, WGPUDeviceImpl, WGPUInstanceImpl,
        WGPUPipelineLayoutImpl, WGPUShaderModuleImpl, WGPUTextureImpl, WGPUTextureViewImpl,
    };
    use std::collections::{BTreeMap, HashSet};
    use std::ffi::{c_void, CString};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn empty_string_view() -> native::WGPUStringView {
        native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        }
    }

    fn instance_impl() -> Arc<WGPUInstanceImpl> {
        Arc::new(WGPUInstanceImpl {
            core: Arc::new(core::Instance::new_noop()),
            timed_wait_any_enabled: false,
            pending_callbacks: Mutex::new(BTreeMap::new()),
        })
    }

    fn device_impl() -> Arc<WGPUDeviceImpl> {
        let instance = instance_impl();
        let adapter = instance
            .core
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("Noop adapter");
        let device = adapter
            .create_device(None, &[], "device", "queue")
            .expect("Noop device");
        Arc::new(WGPUDeviceImpl {
            core: Arc::new(device),
            instance,
            device_lost_callback: DeviceLostCallbackInfo {
                mode: native::WGPUCallbackMode_AllowProcessEvents,
                callback: None,
                userdata1: 0,
                userdata2: 0,
            },
            device_lost_futures: Mutex::new(Vec::new()),
            default_queue: Mutex::new(None),
            shader_module_cache: Mutex::new(std::collections::HashMap::new()),
            pipeline_layout_cache: Mutex::new(std::collections::HashMap::new()),
            compute_pipeline_cache: Mutex::new(std::collections::HashMap::new()),
            render_pipeline_cache: Mutex::new(std::collections::HashMap::new()),
        })
    }

    fn buffer_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUBuffer {
        let buffer = device.core.create_buffer(core::BufferDescriptor {
            usage: core::BufferUsage::COPY_SRC | core::BufferUsage::COPY_DST,
            size: 64,
            mapped_at_creation: false,
        });
        arc_to_handle(Arc::new(WGPUBufferImpl {
            core: Arc::new(buffer),
            device: Arc::clone(&device.core),
            instance: Arc::clone(&device.instance),
        }))
    }

    fn shader_module_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUShaderModule {
        let shader = device
            .core
            .create_shader_module(core::ShaderModuleSource::Spirv(Vec::new()));
        arc_to_handle(Arc::new(WGPUShaderModuleImpl {
            _core: Arc::new(shader),
            _device: Arc::clone(&device.core),
            _instance: Arc::clone(&device.instance),
        }))
    }

    fn bind_group_layout_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUBindGroupLayout {
        let layout = device
            .core
            .create_bind_group_layout(core::BindGroupLayoutDescriptor {
                entries: vec![core::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: native::WGPUShaderStage_Compute,
                    binding_array_size: 0,
                    kind: Some(core::BindingLayoutKind::Buffer {
                        ty: core::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: 4,
                    }),
                }],
                error: None,
            });
        arc_to_handle(Arc::new(WGPUBindGroupLayoutImpl {
            _core: Arc::new(layout),
            _device: Arc::clone(&device.core),
            _instance: Arc::clone(&device.instance),
        }))
    }

    fn pipeline_layout_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUPipelineLayout {
        let layout = device
            .core
            .create_pipeline_layout(core::PipelineLayoutDescriptor {
                bind_group_layouts: Vec::new(),
                immediate_size: 0,
                error: None,
            });
        arc_to_handle(Arc::new(WGPUPipelineLayoutImpl {
            _core: Arc::new(layout),
            _device: Arc::clone(&device.core),
            _instance: Arc::clone(&device.instance),
        }))
    }

    fn texture_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUTexture {
        let texture = device.core.create_texture(core::TextureDescriptor {
            usage: core::TextureUsage::TEXTURE_BINDING | core::TextureUsage::RENDER_ATTACHMENT,
            dimension: core::TextureDimension::D2,
            size: core::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm.into(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        });
        arc_to_handle(Arc::new(WGPUTextureImpl {
            core: Arc::new(texture),
            device: Arc::clone(&device.core),
            instance: Arc::clone(&device.instance),
        }))
    }

    fn texture_view_handle(device: &Arc<WGPUDeviceImpl>) -> native::WGPUTextureView {
        let texture = Arc::new(device.core.create_texture(core::TextureDescriptor {
            usage: core::TextureUsage::TEXTURE_BINDING | core::TextureUsage::RENDER_ATTACHMENT,
            dimension: core::TextureDimension::D2,
            size: core::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm.into(),
            mip_level_count: 1,
            sample_count: 1,
            view_formats: Vec::new(),
        }));
        let (view, _) = texture.create_view(core::TextureViewDescriptor {
            format: None,
            dimension: None,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: None,
        });
        arc_to_handle(Arc::new(WGPUTextureViewImpl {
            _core: Arc::new(view),
            _texture: texture,
            _device: Arc::clone(&device.core),
            _instance: Arc::clone(&device.instance),
        }))
    }

    #[derive(Debug)]
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn arc_to_handle_round_trips_with_clone_handle_refcount_math() {
        let value = Arc::new(7_u32);
        let handle = arc_to_handle(Arc::clone(&value));
        assert_eq!(Arc::strong_count(&value), 2);
        let cloned = unsafe { clone_handle(handle, "u32") };
        assert_eq!(*cloned, 7);
        assert_eq!(Arc::strong_count(&value), 3);
        drop(cloned);
        assert_eq!(Arc::strong_count(&value), 2);
        unsafe { release_handle(handle, "u32") };
        assert_eq!(Arc::strong_count(&value), 1);
    }

    #[test]
    fn release_handle_drops_owned_reference_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let handle = arc_to_handle(Arc::new(DropCounter(Arc::clone(&drops))));
        unsafe { release_handle(handle, "DropCounter") };
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn add_ref_handle_increments_refcount_for_later_release() {
        let value = Arc::new(11_u32);
        let handle = arc_to_handle(Arc::clone(&value));
        unsafe { add_ref_handle(handle, "u32") };
        assert_eq!(Arc::strong_count(&value), 3);
        unsafe { release_handle(handle, "u32") };
        assert_eq!(Arc::strong_count(&value), 2);
        unsafe { release_handle(handle, "u32") };
        assert_eq!(Arc::strong_count(&value), 1);
    }

    #[test]
    #[should_panic(expected = "WGPUInstance must not be null")]
    fn release_handle_null_panics_with_contract_message() {
        unsafe {
            release_handle::<core::Instance>(std::ptr::null(), "WGPUInstance");
        }
    }

    #[test]
    #[should_panic(expected = "WGPUInstance must not be null")]
    fn add_ref_handle_null_panics_with_contract_message() {
        unsafe {
            add_ref_handle::<core::Instance>(std::ptr::null(), "WGPUInstance");
        }
    }

    #[test]
    fn clone_handle_leaves_original_handle_valid() {
        let value = Arc::new(13_u32);
        let handle = arc_to_handle(Arc::clone(&value));
        let cloned = unsafe { clone_handle(handle, "u32") };
        assert_eq!(unsafe { *borrow_handle::<u32>(handle, "u32") }, 13);
        drop(cloned);
        unsafe { release_handle(handle, "u32") };
    }

    #[test]
    fn borrow_handle_returns_reference_without_consuming_arc() {
        let value = Arc::new(17_u32);
        let handle = arc_to_handle(Arc::clone(&value));
        let borrowed = unsafe { borrow_handle::<u32>(handle, "u32") };
        assert_eq!(*borrowed, 17);
        assert!(std::ptr::eq(borrowed, Arc::as_ptr(&value)));
        assert_eq!(Arc::strong_count(&value), 2);
        unsafe { release_handle(handle, "u32") };
    }

    #[test]
    #[should_panic(expected = "WGPUBuffer must not be null")]
    fn clone_handle_null_panics_with_contract_message() {
        let _ = unsafe { clone_handle::<WGPUBufferImpl>(std::ptr::null(), "WGPUBuffer") };
    }

    #[test]
    #[should_panic(expected = "WGPUBuffer must not be null")]
    fn borrow_handle_null_panics_with_contract_message() {
        let _ = unsafe { borrow_handle::<WGPUBufferImpl>(std::ptr::null(), "WGPUBuffer") };
    }

    #[test]
    fn string_view_round_trips_data_and_empty_slice() {
        let view = string_view(b"hello");
        assert_eq!(view.length, 5);
        assert_eq!(unsafe { string_view_to_str(view) }, Some("hello"));
        let empty = string_view(b"");
        assert_eq!(empty.length, 0);
    }

    #[test]
    fn string_view_to_str_handles_explicit_strlen_and_null_data() {
        let direct = string_view(b"abc");
        assert_eq!(unsafe { string_view_to_str(direct) }, Some("abc"));
        let c_string = CString::new("auto").expect("CString");
        let auto = native::WGPUStringView {
            data: c_string.as_ptr(),
            length: WGPU_STRLEN,
        };
        assert_eq!(unsafe { string_view_to_str(auto) }, Some("auto"));
        assert_eq!(unsafe { string_view_to_str(empty_string_view()) }, None);
    }

    #[test]
    fn label_from_string_view_returns_owned_label_or_none() {
        assert_eq!(
            unsafe { label_from_string_view(string_view(b"label")) },
            Some("label".to_owned())
        );
        assert_eq!(unsafe { label_from_string_view(empty_string_view()) }, None);
    }

    #[test]
    fn map_feature_round_trips_defined_and_other_variants() {
        for value in [
            native::WGPUFeatureName_CoreFeaturesAndLimits,
            native::WGPUFeatureName_RG11B10UfloatRenderable,
            native::WGPUFeatureName_TimestampQuery,
            native::WGPUFeatureName_TextureFormatsTier1,
            native::WGPUFeatureName_TextureFormatsTier2,
            0xCAFE,
        ] {
            assert_eq!(map_feature_to_native(map_feature(value)), value);
        }
    }

    #[test]
    fn map_query_type_round_trips_defined_and_unknown_variants() {
        for value in [
            native::WGPUQueryType_Occlusion,
            native::WGPUQueryType_Timestamp,
            0xCAFE,
        ] {
            assert_eq!(map_query_type_to_native(map_query_type(value)), value);
        }
    }

    #[test]
    fn from_native_query_type_round_trips_known_and_unknown_variants() {
        let known = core::QueryType::from(native::WGPUQueryType_Occlusion);
        assert_eq!(known, core::QueryType::Occlusion);
        assert_eq!(
            Into::<native::WGPUQueryType>::into(known),
            native::WGPUQueryType_Occlusion
        );

        let unknown_native = 0xFFFF_u32 as native::WGPUQueryType;
        let unknown = core::QueryType::from(unknown_native);
        assert_eq!(unknown, core::QueryType::Unknown(0xFFFF));
        assert_eq!(Into::<native::WGPUQueryType>::into(unknown), unknown_native);
    }

    #[test]
    fn map_buffer_usage_round_trips_bitmask() {
        let usage = native::WGPUBufferUsage_MapRead
            | native::WGPUBufferUsage_CopyDst
            | native::WGPUBufferUsage_Uniform
            | 0x8000_0000;
        assert_eq!(map_buffer_usage_to_native(map_buffer_usage(usage)), usage);
    }

    #[test]
    fn map_texture_usage_round_trips_bitmask() {
        let usage = native::WGPUTextureUsage_CopySrc
            | native::WGPUTextureUsage_TextureBinding
            | native::WGPUTextureUsage_RenderAttachment
            | 0x8000_0000;
        assert_eq!(map_texture_usage_to_native(map_texture_usage(usage)), usage);
    }

    #[test]
    fn map_texture_dimension_round_trips_defined_variants() {
        for (native_value, core_value) in [
            (native::WGPUTextureDimension_1D, core::TextureDimension::D1),
            (native::WGPUTextureDimension_2D, core::TextureDimension::D2),
            (native::WGPUTextureDimension_3D, core::TextureDimension::D3),
        ] {
            assert_eq!(map_texture_dimension(native_value), core_value);
            assert_eq!(map_texture_dimension_to_native(core_value), native_value);
        }
        assert_eq!(
            map_texture_dimension(native::WGPUTextureDimension_Undefined),
            core::TextureDimension::D2
        );
    }

    #[test]
    fn map_texture_format_round_trips_defined_and_unknown_raw_values() {
        for value in [
            native::WGPUTextureFormat_Undefined,
            native::WGPUTextureFormat_R8Unorm,
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_BGRA8Unorm,
            0xCAFE,
        ] {
            assert_eq!(
                map_texture_format_to_native(map_texture_format(value)),
                value
            );
        }
    }

    #[test]
    fn from_native_texture_format_round_trips_known_and_unknown_variants() {
        let known = core::TextureFormat::from(native::WGPUTextureFormat_RGBA8Unorm);
        assert_eq!(known, core::TextureFormat::from_raw(0x16));
        assert_eq!(
            Into::<native::WGPUTextureFormat>::into(known),
            native::WGPUTextureFormat_RGBA8Unorm
        );

        let unknown_native = 0xFFFF_u32 as native::WGPUTextureFormat;
        let unknown = core::TextureFormat::from(unknown_native);
        assert_eq!(unknown.raw(), 0xFFFF);
        assert_eq!(
            Into::<native::WGPUTextureFormat>::into(unknown),
            unknown_native
        );
    }

    #[test]
    fn from_native_vertex_format_round_trips_known_and_unknown_variants() {
        let known = core::VertexFormat::from(native::WGPUVertexFormat_Float32x2);
        assert_eq!(known, core::VertexFormat::from_raw(0x1D));
        assert_eq!(map_vertex_format(native::WGPUVertexFormat_Float32x2), known);
        assert_eq!(
            Into::<native::WGPUVertexFormat>::into(known),
            native::WGPUVertexFormat_Float32x2
        );
        assert_eq!(
            map_vertex_format_to_native(known),
            native::WGPUVertexFormat_Float32x2
        );

        let unknown_native = 0xFFFF_u32 as native::WGPUVertexFormat;
        let unknown = core::VertexFormat::from(unknown_native);
        assert_eq!(unknown.raw(), 0xFFFF);
        assert_eq!(
            Into::<native::WGPUVertexFormat>::into(unknown),
            unknown_native
        );
    }

    #[test]
    fn map_feature_level_maps_compatibility_and_default_core() {
        assert_eq!(
            map_feature_level(native::WGPUFeatureLevel_Compatibility),
            core::FeatureLevel::Compatibility
        );
        assert_eq!(
            map_feature_level(native::WGPUFeatureLevel_Core),
            core::FeatureLevel::Core
        );
        assert_eq!(
            map_feature_level(native::WGPUFeatureLevel_Undefined),
            core::FeatureLevel::Core
        );
    }

    #[test]
    fn map_device_lost_reason_maps_every_core_variant() {
        assert_eq!(
            map_device_lost_reason(core::DeviceLostReason::Unknown),
            native::WGPUDeviceLostReason_Unknown
        );
        assert_eq!(
            map_device_lost_reason(core::DeviceLostReason::Destroyed),
            native::WGPUDeviceLostReason_Destroyed
        );
        assert_eq!(
            map_device_lost_reason(core::DeviceLostReason::CallbackCancelled),
            native::WGPUDeviceLostReason_CallbackCancelled
        );
        assert_eq!(
            map_device_lost_reason(core::DeviceLostReason::FailedCreation),
            native::WGPUDeviceLostReason_FailedCreation
        );
    }

    #[test]
    fn map_error_filter_maps_known_values_and_rejects_unknown() {
        assert_eq!(
            map_error_filter(native::WGPUErrorFilter_Validation),
            Some(core::ErrorFilter::Validation)
        );
        assert_eq!(
            map_error_filter(native::WGPUErrorFilter_OutOfMemory),
            Some(core::ErrorFilter::OutOfMemory)
        );
        assert_eq!(
            map_error_filter(native::WGPUErrorFilter_Internal),
            Some(core::ErrorFilter::Internal)
        );
        assert_eq!(map_error_filter(0xCAFE), None);
    }

    #[test]
    fn map_error_type_maps_every_core_variant() {
        assert_eq!(
            map_error_type(core::ErrorKind::Validation),
            native::WGPUErrorType_Validation
        );
        assert_eq!(
            map_error_type(core::ErrorKind::OutOfMemory),
            native::WGPUErrorType_OutOfMemory
        );
        assert_eq!(
            map_error_type(core::ErrorKind::Internal),
            native::WGPUErrorType_Internal
        );
    }

    #[test]
    fn map_pop_error_scope_status_error_returns_error() {
        assert_eq!(
            map_pop_error_scope_status_error(),
            native::WGPUPopErrorScopeStatus_Error
        );
    }

    #[test]
    fn map_pop_error_scope_status_success_returns_success() {
        assert_eq!(
            map_pop_error_scope_status_success(),
            native::WGPUPopErrorScopeStatus_Success
        );
    }

    #[test]
    fn map_buffer_map_state_maps_every_core_variant() {
        assert_eq!(
            map_buffer_map_state(core::BufferMapState::Unmapped),
            native::WGPUBufferMapState_Unmapped
        );
        assert_eq!(
            map_buffer_map_state(core::BufferMapState::Pending),
            native::WGPUBufferMapState_Pending
        );
        assert_eq!(
            map_buffer_map_state(core::BufferMapState::Mapped),
            native::WGPUBufferMapState_Mapped
        );
    }

    #[test]
    fn map_map_async_status_maps_every_core_variant() {
        assert_eq!(
            map_map_async_status(core::MapAsyncStatus::Success),
            native::WGPUMapAsyncStatus_Success
        );
        assert_eq!(
            map_map_async_status(core::MapAsyncStatus::CallbackCancelled),
            native::WGPUMapAsyncStatus_CallbackCancelled
        );
        assert_eq!(
            map_map_async_status(core::MapAsyncStatus::Error),
            native::WGPUMapAsyncStatus_Error
        );
        assert_eq!(
            map_map_async_status(core::MapAsyncStatus::Aborted),
            native::WGPUMapAsyncStatus_Aborted
        );
    }

    #[test]
    fn map_queue_work_done_status_maps_every_core_variant() {
        assert_eq!(
            map_queue_work_done_status(core::QueueWorkDoneStatus::Success),
            native::WGPUQueueWorkDoneStatus_Success
        );
        assert_eq!(
            map_queue_work_done_status(core::QueueWorkDoneStatus::CallbackCancelled),
            native::WGPUQueueWorkDoneStatus_CallbackCancelled
        );
        assert_eq!(
            map_queue_work_done_status(core::QueueWorkDoneStatus::Error),
            native::WGPUQueueWorkDoneStatus_Error
        );
    }

    #[test]
    fn map_compilation_info_request_status_success_returns_success() {
        assert_eq!(
            map_compilation_info_request_status_success(),
            native::WGPUCompilationInfoRequestStatus_Success
        );
    }

    #[test]
    fn map_compilation_message_type_error_returns_error() {
        assert_eq!(
            map_compilation_message_type_error(),
            native::WGPUCompilationMessageType_Error
        );
    }

    #[test]
    fn map_map_mode_accepts_single_modes_and_rejects_invalid_combinations() {
        assert_eq!(
            map_map_mode(native::WGPUMapMode_Read),
            Ok(core::MapMode::Read)
        );
        assert_eq!(
            map_map_mode(native::WGPUMapMode_Write),
            Ok(core::MapMode::Write)
        );
        assert!(map_map_mode(native::WGPUMapMode_Read | native::WGPUMapMode_Write).is_err());
        assert!(map_map_mode(native::WGPUMapMode_None).is_err());
        assert!(map_map_mode(0x8000_0000).is_err());
    }

    #[test]
    fn map_address_mode_maps_known_values_and_rejects_unknown() {
        assert_eq!(map_address_mode(native::WGPUAddressMode_Undefined), None);
        assert_eq!(
            map_address_mode(native::WGPUAddressMode_ClampToEdge),
            Some(core::AddressMode::ClampToEdge)
        );
        assert_eq!(
            map_address_mode(native::WGPUAddressMode_Repeat),
            Some(core::AddressMode::Repeat)
        );
        assert_eq!(
            map_address_mode(native::WGPUAddressMode_MirrorRepeat),
            Some(core::AddressMode::MirrorRepeat)
        );
        assert_eq!(map_address_mode(0xCAFE), None);
    }

    #[test]
    fn map_filter_mode_maps_known_values_and_rejects_unknown() {
        assert_eq!(map_filter_mode(native::WGPUFilterMode_Undefined), None);
        assert_eq!(
            map_filter_mode(native::WGPUFilterMode_Nearest),
            Some(core::FilterMode::Nearest)
        );
        assert_eq!(
            map_filter_mode(native::WGPUFilterMode_Linear),
            Some(core::FilterMode::Linear)
        );
        assert_eq!(map_filter_mode(0xCAFE), None);
    }

    #[test]
    fn map_mipmap_filter_mode_maps_known_values_and_rejects_unknown() {
        assert_eq!(
            map_mipmap_filter_mode(native::WGPUMipmapFilterMode_Undefined),
            None
        );
        assert_eq!(
            map_mipmap_filter_mode(native::WGPUMipmapFilterMode_Nearest),
            Some(core::MipmapFilterMode::Nearest)
        );
        assert_eq!(
            map_mipmap_filter_mode(native::WGPUMipmapFilterMode_Linear),
            Some(core::MipmapFilterMode::Linear)
        );
        assert_eq!(map_mipmap_filter_mode(0xCAFE), None);
    }

    #[test]
    fn map_compare_function_maps_known_values_and_rejects_undefined() {
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Undefined),
            None
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Never),
            Some(core::CompareFunction::Never)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Less),
            Some(core::CompareFunction::Less)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Equal),
            Some(core::CompareFunction::Equal)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_LessEqual),
            Some(core::CompareFunction::LessEqual)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Greater),
            Some(core::CompareFunction::Greater)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_NotEqual),
            Some(core::CompareFunction::NotEqual)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_GreaterEqual),
            Some(core::CompareFunction::GreaterEqual)
        );
        assert_eq!(
            map_compare_function(native::WGPUCompareFunction_Always),
            Some(core::CompareFunction::Always)
        );
        assert_eq!(map_compare_function(0xCAFE), None);
    }

    #[test]
    fn map_texture_view_dimension_maps_known_values_and_rejects_unknown() {
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_Undefined),
            None
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_1D),
            Some(core::TextureViewDimension::D1)
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_2D),
            Some(core::TextureViewDimension::D2)
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_2DArray),
            Some(core::TextureViewDimension::D2Array)
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_Cube),
            Some(core::TextureViewDimension::Cube)
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_CubeArray),
            Some(core::TextureViewDimension::CubeArray)
        );
        assert_eq!(
            map_texture_view_dimension(native::WGPUTextureViewDimension_3D),
            Some(core::TextureViewDimension::D3)
        );
        assert_eq!(map_texture_view_dimension(0xCAFE), None);
    }

    #[test]
    fn map_texture_aspect_maps_known_values_and_rejects_undefined() {
        assert_eq!(
            map_texture_aspect(native::WGPUTextureAspect_Undefined),
            None
        );
        assert_eq!(
            map_texture_aspect(native::WGPUTextureAspect_All),
            Some(core::TextureAspect::All)
        );
        assert_eq!(
            map_texture_aspect(native::WGPUTextureAspect_DepthOnly),
            Some(core::TextureAspect::DepthOnly)
        );
        assert_eq!(
            map_texture_aspect(native::WGPUTextureAspect_StencilOnly),
            Some(core::TextureAspect::StencilOnly)
        );
        assert_eq!(map_texture_aspect(0xCAFE), None);
    }

    #[test]
    fn map_load_op_maps_defined_values_and_undefined_fallback() {
        assert_eq!(map_load_op(native::WGPULoadOp_Load), core::LoadOp::Load);
        assert_eq!(map_load_op(native::WGPULoadOp_Clear), core::LoadOp::Clear);
        assert_eq!(
            map_load_op(native::WGPULoadOp_Undefined),
            core::LoadOp::Undefined
        );
    }

    #[test]
    fn map_store_op_maps_defined_values_and_undefined_fallback() {
        assert_eq!(
            map_store_op(native::WGPUStoreOp_Store),
            core::StoreOp::Store
        );
        assert_eq!(
            map_store_op(native::WGPUStoreOp_Discard),
            core::StoreOp::Discard
        );
        assert_eq!(
            map_store_op(native::WGPUStoreOp_Undefined),
            core::StoreOp::Undefined
        );
    }

    #[test]
    fn map_query_index_maps_defined_values_and_undefined_to_none() {
        assert_eq!(map_query_index(3), Some(3));
        assert_eq!(
            map_query_index(native::WGPU_QUERY_SET_INDEX_UNDEFINED),
            None
        );
    }

    #[test]
    fn has_callback_detects_present_and_absent_device_lost_callbacks() {
        unsafe extern "C" fn callback(
            _device: *const native::WGPUDevice,
            _reason: native::WGPUDeviceLostReason,
            _message: native::WGPUStringView,
            _userdata1: *mut c_void,
            _userdata2: *mut c_void,
        ) {
        }

        let with_callback = DeviceLostCallbackInfo {
            mode: native::WGPUCallbackMode_AllowProcessEvents,
            callback: Some(callback),
            userdata1: 1,
            userdata2: 2,
        };
        let without_callback = DeviceLostCallbackInfo {
            callback: None,
            ..with_callback
        };
        assert!(with_callback.has_callback());
        assert!(!without_callback.has_callback());
    }

    #[test]
    fn map_buffer_descriptor_round_trips_fields() {
        let descriptor = native::WGPUBufferDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: string_view(b"buffer"),
            usage: native::WGPUBufferUsage_CopySrc | native::WGPUBufferUsage_Uniform,
            size: 4096,
            mappedAtCreation: 1,
        };
        let mapped = map_buffer_descriptor(&descriptor);
        assert_eq!(mapped.usage.bits(), descriptor.usage);
        assert_eq!(mapped.size, 4096);
        assert!(mapped.mapped_at_creation);
    }

    #[test]
    fn map_sampler_descriptor_round_trips_fields_with_undefined_compare() {
        let descriptor = native::WGPUSamplerDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            addressModeU: native::WGPUAddressMode_ClampToEdge,
            addressModeV: native::WGPUAddressMode_Repeat,
            addressModeW: native::WGPUAddressMode_MirrorRepeat,
            magFilter: native::WGPUFilterMode_Linear,
            minFilter: native::WGPUFilterMode_Nearest,
            mipmapFilter: native::WGPUMipmapFilterMode_Linear,
            lodMinClamp: 1.0,
            lodMaxClamp: 8.0,
            compare: native::WGPUCompareFunction_Undefined,
            maxAnisotropy: 4,
        };
        let mapped = map_sampler_descriptor(Some(&descriptor));
        assert_eq!(mapped.address_mode_u, Some(core::AddressMode::ClampToEdge));
        assert_eq!(mapped.address_mode_v, Some(core::AddressMode::Repeat));
        assert_eq!(mapped.address_mode_w, Some(core::AddressMode::MirrorRepeat));
        assert_eq!(mapped.mag_filter, Some(core::FilterMode::Linear));
        assert_eq!(mapped.min_filter, Some(core::FilterMode::Nearest));
        assert_eq!(mapped.mipmap_filter, Some(core::MipmapFilterMode::Linear));
        assert_eq!(mapped.lod_min_clamp, 1.0);
        assert_eq!(mapped.lod_max_clamp, 8.0);
        assert_eq!(mapped.compare, None);
        assert_eq!(mapped.max_anisotropy, 4);
    }

    #[test]
    fn map_extent_3d_round_trips_fields() {
        let mapped = map_extent_3d(native::WGPUExtent3D {
            width: 1,
            height: 2,
            depthOrArrayLayers: 3,
        });
        assert_eq!(
            mapped,
            core::Extent3d {
                width: 1,
                height: 2,
                depth_or_array_layers: 3
            }
        );
    }

    #[test]
    fn map_origin_3d_round_trips_fields() {
        assert_eq!(
            map_origin_3d(native::WGPUOrigin3D { x: 4, y: 5, z: 6 }),
            core::Origin3d { x: 4, y: 5, z: 6 }
        );
    }

    #[test]
    fn map_color_round_trips_float_bits_including_nan() {
        let nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let mapped = map_color(native::WGPUColor {
            r: 1.0,
            g: -2.0,
            b: nan,
            a: 4.0,
        });
        assert_eq!(mapped.r.to_bits(), 1.0f64.to_bits());
        assert_eq!(mapped.g.to_bits(), (-2.0f64).to_bits());
        assert_eq!(mapped.b.to_bits(), nan.to_bits());
        assert_eq!(mapped.a.to_bits(), 4.0f64.to_bits());
    }

    #[test]
    fn map_texel_copy_buffer_layout_round_trips_fields_and_undefined_strides() {
        let mapped = map_texel_copy_buffer_layout(native::WGPUTexelCopyBufferLayout {
            offset: 64,
            bytesPerRow: 256,
            rowsPerImage: native::WGPU_COPY_STRIDE_UNDEFINED,
        });
        assert_eq!(mapped.offset, 64);
        assert_eq!(mapped.bytes_per_row, Some(256));
        assert_eq!(mapped.rows_per_image, None);
    }

    #[test]
    fn map_texel_copy_texture_info_parts_round_trips_fields() {
        let device = device_impl();
        let texture = texture_handle(&device);
        let value = native::WGPUTexelCopyTextureInfo {
            texture,
            mipLevel: 2,
            origin: native::WGPUOrigin3D { x: 1, y: 2, z: 3 },
            aspect: native::WGPUTextureAspect_DepthOnly,
        };
        let (mip_level, origin, aspect) = map_texel_copy_texture_info_parts(&value);
        assert_eq!(mip_level, 2);
        assert_eq!(origin, core::Origin3d { x: 1, y: 2, z: 3 });
        assert_eq!(aspect, core::TextureAspect::DepthOnly);
        unsafe { release_handle(texture, "WGPUTexture") };
    }

    fn distinct_limits() -> native::WGPULimits {
        native::WGPULimits {
            nextInChain: std::ptr::null_mut(),
            maxTextureDimension1D: 101,
            maxTextureDimension2D: 102,
            maxTextureDimension3D: 103,
            maxTextureArrayLayers: 104,
            maxBindGroups: 105,
            maxBindGroupsPlusVertexBuffers: 106,
            maxBindingsPerBindGroup: 107,
            maxDynamicUniformBuffersPerPipelineLayout: 108,
            maxDynamicStorageBuffersPerPipelineLayout: 109,
            maxSampledTexturesPerShaderStage: 110,
            maxSamplersPerShaderStage: 111,
            maxStorageBuffersPerShaderStage: 112,
            maxStorageTexturesPerShaderStage: 113,
            maxUniformBuffersPerShaderStage: 114,
            maxUniformBufferBindingSize: 115,
            maxStorageBufferBindingSize: 116,
            minUniformBufferOffsetAlignment: 117,
            minStorageBufferOffsetAlignment: 118,
            maxVertexBuffers: 119,
            maxBufferSize: 120,
            maxVertexAttributes: 121,
            maxVertexBufferArrayStride: 122,
            maxInterStageShaderVariables: 123,
            maxColorAttachments: 124,
            maxColorAttachmentBytesPerSample: 125,
            maxComputeWorkgroupStorageSize: 126,
            maxComputeInvocationsPerWorkgroup: 127,
            maxComputeWorkgroupSizeX: 128,
            maxComputeWorkgroupSizeY: 129,
            maxComputeWorkgroupSizeZ: 130,
            maxComputeWorkgroupsPerDimension: 131,
            maxImmediateSize: 132,
        }
    }

    #[test]
    fn map_limits_round_trips_every_field_from_native() {
        let mapped = map_limits(&distinct_limits());
        assert_eq!(mapped.max_texture_dimension_1d, 101);
        assert_eq!(mapped.max_texture_dimension_2d, 102);
        assert_eq!(mapped.max_texture_dimension_3d, 103);
        assert_eq!(mapped.max_texture_array_layers, 104);
        assert_eq!(mapped.max_bind_groups, 105);
        assert_eq!(mapped.max_bind_groups_plus_vertex_buffers, 106);
        assert_eq!(mapped.max_bindings_per_bind_group, 107);
        assert_eq!(mapped.max_dynamic_uniform_buffers_per_pipeline_layout, 108);
        assert_eq!(mapped.max_dynamic_storage_buffers_per_pipeline_layout, 109);
        assert_eq!(mapped.max_sampled_textures_per_shader_stage, 110);
        assert_eq!(mapped.max_samplers_per_shader_stage, 111);
        assert_eq!(mapped.max_storage_buffers_per_shader_stage, 112);
        assert_eq!(mapped.max_storage_textures_per_shader_stage, 113);
        assert_eq!(mapped.max_uniform_buffers_per_shader_stage, 114);
        assert_eq!(mapped.max_uniform_buffer_binding_size, 115);
        assert_eq!(mapped.max_storage_buffer_binding_size, 116);
        assert_eq!(mapped.min_uniform_buffer_offset_alignment, 117);
        assert_eq!(mapped.min_storage_buffer_offset_alignment, 118);
        assert_eq!(mapped.max_vertex_buffers, 119);
        assert_eq!(mapped.max_buffer_size, 120);
        assert_eq!(mapped.max_vertex_attributes, 121);
        assert_eq!(mapped.max_vertex_buffer_array_stride, 122);
        assert_eq!(mapped.max_inter_stage_shader_variables, 123);
        assert_eq!(mapped.max_color_attachments, 124);
        assert_eq!(mapped.max_color_attachment_bytes_per_sample, 125);
        assert_eq!(mapped.max_compute_workgroup_storage_size, 126);
        assert_eq!(mapped.max_compute_invocations_per_workgroup, 127);
        assert_eq!(mapped.max_compute_workgroup_size_x, 128);
        assert_eq!(mapped.max_compute_workgroup_size_y, 129);
        assert_eq!(mapped.max_compute_workgroup_size_z, 130);
        assert_eq!(mapped.max_compute_workgroups_per_dimension, 131);
        assert_eq!(mapped.max_immediate_size, 132);
    }

    #[test]
    fn map_limits_to_native_round_trips_through_map_limits() {
        let limits = map_limits(&distinct_limits());
        let native = map_limits_to_native(limits);
        assert_eq!(map_limits(&native), limits);
    }

    #[test]
    fn map_features_to_native_allocates_feature_array_and_free_supported_features_releases_it() {
        let features = [
            core::Feature::TimestampQuery,
            core::Feature::TextureFormatsTier1,
        ]
        .into_iter()
        .collect::<core::FeatureSet>();
        let native_features = map_features_to_native(&features);
        assert_eq!(native_features.featureCount, 2);
        let slice = unsafe {
            std::slice::from_raw_parts(native_features.features, native_features.featureCount)
        };
        let found = slice.iter().copied().collect::<HashSet<_>>();
        assert!(found.contains(&native::WGPUFeatureName_TimestampQuery));
        assert!(found.contains(&native::WGPUFeatureName_TextureFormatsTier1));
        unsafe { free_supported_features(native_features) };
    }

    #[test]
    fn free_supported_features_accepts_null_feature_array() {
        unsafe {
            free_supported_features(native::WGPUSupportedFeatures {
                featureCount: 0,
                features: std::ptr::null(),
            })
        };
    }

    #[test]
    fn map_shader_module_descriptor_decodes_wgsl_source_and_missing_source_error() {
        let mut wgsl = native::WGPUShaderSourceWGSL {
            chain: native::WGPUChainedStruct {
                next: std::ptr::null_mut(),
                sType: native::WGPUSType_ShaderSourceWGSL,
            },
            code: string_view(b"@compute @workgroup_size(1) fn main() {}"),
        };
        let descriptor = native::WGPUShaderModuleDescriptor {
            nextInChain: (&mut wgsl.chain) as *mut native::WGPUChainedStruct,
            label: empty_string_view(),
        };
        match unsafe { map_shader_module_descriptor(&descriptor) } {
            core::ShaderModuleSource::Wgsl(source) => assert!(source.contains("fn main")),
            other => panic!("unexpected source: {other:?}"),
        }
        let missing = native::WGPUShaderModuleDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        };
        match unsafe { map_shader_module_descriptor(&missing) } {
            core::ShaderModuleSource::Invalid(message) => {
                assert!(message.contains("exactly one shader source"));
            }
            other => panic!("unexpected source: {other:?}"),
        }
    }

    fn buffer_binding_layout() -> native::WGPUBufferBindingLayout {
        native::WGPUBufferBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUBufferBindingType_Uniform,
            hasDynamicOffset: 1,
            minBindingSize: 16,
        }
    }

    fn unused_sampler_layout() -> native::WGPUSamplerBindingLayout {
        native::WGPUSamplerBindingLayout {
            nextInChain: std::ptr::null_mut(),
            type_: native::WGPUSamplerBindingType_BindingNotUsed,
        }
    }

    fn unused_texture_layout() -> native::WGPUTextureBindingLayout {
        native::WGPUTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            sampleType: native::WGPUTextureSampleType_BindingNotUsed,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
            multisampled: 0,
        }
    }

    fn unused_storage_texture_layout() -> native::WGPUStorageTextureBindingLayout {
        native::WGPUStorageTextureBindingLayout {
            nextInChain: std::ptr::null_mut(),
            access: native::WGPUStorageTextureAccess_BindingNotUsed,
            format: native::WGPUTextureFormat_Undefined,
            viewDimension: native::WGPUTextureViewDimension_Undefined,
        }
    }

    #[test]
    fn map_bind_group_layout_descriptor_decodes_buffer_entry_and_null_entries_error() {
        let entry = native::WGPUBindGroupLayoutEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 7,
            visibility: native::WGPUShaderStage_Vertex | native::WGPUShaderStage_Compute,
            bindingArraySize: 2,
            buffer: buffer_binding_layout(),
            sampler: unused_sampler_layout(),
            texture: unused_texture_layout(),
            storageTexture: unused_storage_texture_layout(),
        };
        let descriptor = native::WGPUBindGroupLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            entryCount: 1,
            entries: &entry,
        };
        let mapped = unsafe { map_bind_group_layout_descriptor(&descriptor) };
        assert_eq!(mapped.error, None);
        assert_eq!(mapped.entries.len(), 1);
        assert_eq!(mapped.entries[0].binding, 7);
        assert_eq!(mapped.entries[0].visibility, entry.visibility);
        assert!(matches!(
            mapped.entries[0].kind,
            Some(core::BindingLayoutKind::Buffer {
                ty: core::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: 16
            })
        ));

        let invalid = native::WGPUBindGroupLayoutDescriptor {
            entryCount: 1,
            entries: std::ptr::null(),
            ..descriptor
        };
        assert!(unsafe { map_bind_group_layout_descriptor(&invalid) }
            .error
            .expect("error")
            .contains("must not be null"));
    }

    #[test]
    fn map_bind_group_entries_decodes_buffer_entry_and_null_entries_error() {
        let device = device_impl();
        let buffer = buffer_handle(&device);
        let entry = native::WGPUBindGroupEntry {
            nextInChain: std::ptr::null_mut(),
            binding: 3,
            buffer,
            offset: 4,
            size: 8,
            sampler: std::ptr::null(),
            textureView: std::ptr::null(),
        };
        let descriptor = native::WGPUBindGroupDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null(),
            entryCount: 1,
            entries: &entry,
        };
        let mapped = unsafe { map_bind_group_entries(&descriptor) };
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].binding, 3);
        assert!(matches!(
            mapped[0].resource,
            core::BindGroupResource::Buffer {
                offset: 4,
                size: 8,
                ..
            }
        ));

        let invalid = native::WGPUBindGroupDescriptor {
            entryCount: 1,
            entries: std::ptr::null(),
            ..descriptor
        };
        assert!(matches!(
            unsafe { map_bind_group_entries(&invalid) }[0].resource,
            core::BindGroupResource::Invalid(_)
        ));
        unsafe { release_handle(buffer, "WGPUBuffer") };
    }

    #[test]
    fn map_pipeline_layout_descriptor_decodes_layouts_and_null_array_error() {
        let device = device_impl();
        let layout = bind_group_layout_handle(&device);
        let layouts = [layout];
        let descriptor = native::WGPUPipelineLayoutDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            bindGroupLayoutCount: 1,
            bindGroupLayouts: layouts.as_ptr(),
            immediateSize: 32,
        };
        let mapped = unsafe { map_pipeline_layout_descriptor(&descriptor) };
        assert_eq!(mapped.bind_group_layouts.len(), 1);
        assert_eq!(mapped.immediate_size, 32);
        assert_eq!(mapped.error, None);

        let invalid = native::WGPUPipelineLayoutDescriptor {
            bindGroupLayouts: std::ptr::null(),
            ..descriptor
        };
        assert!(unsafe { map_pipeline_layout_descriptor(&invalid) }
            .error
            .expect("error")
            .contains("must not be null"));
        unsafe { release_handle(layout, "WGPUBindGroupLayout") };
    }

    #[test]
    #[should_panic(expected = "WGPUShaderModule must not be null")]
    fn map_compute_pipeline_descriptor_null_module_panics() {
        let descriptor = native::WGPUComputePipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null(),
            compute: native::WGPUComputeState {
                nextInChain: std::ptr::null_mut(),
                module: std::ptr::null(),
                entryPoint: string_view(b"main"),
                constantCount: 0,
                constants: std::ptr::null(),
            },
        };
        let _ = unsafe { map_compute_pipeline_descriptor(&descriptor) };
    }

    #[test]
    fn map_compute_pipeline_descriptor_decodes_module_entry_layout_and_constants() {
        let device = device_impl();
        let shader = shader_module_handle(&device);
        let layout = pipeline_layout_handle(&device);
        let constant = native::WGPUConstantEntry {
            nextInChain: std::ptr::null_mut(),
            key: string_view(b"X"),
            value: 2.5,
        };
        let descriptor = native::WGPUComputePipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout,
            compute: native::WGPUComputeState {
                nextInChain: std::ptr::null_mut(),
                module: shader,
                entryPoint: string_view(b"main"),
                constantCount: 1,
                constants: &constant,
            },
        };
        let mapped = unsafe { map_compute_pipeline_descriptor(&descriptor) };
        assert!(matches!(
            mapped.layout,
            core::ComputePipelineLayout::Explicit(_)
        ));
        assert_eq!(mapped.entry_point.as_deref(), Some("main"));
        assert_eq!(mapped.constants.len(), 1);
        assert_eq!(mapped.constants[0].key, "X");
        assert_eq!(mapped.constants[0].value, 2.5);
        assert_eq!(mapped.error, None);
        unsafe {
            release_handle(shader, "WGPUShaderModule");
            release_handle(layout, "WGPUPipelineLayout");
        }
    }

    fn primitive_state() -> native::WGPUPrimitiveState {
        native::WGPUPrimitiveState {
            nextInChain: std::ptr::null_mut(),
            topology: native::WGPUPrimitiveTopology_TriangleList,
            stripIndexFormat: native::WGPUIndexFormat_Undefined,
            frontFace: native::WGPUFrontFace_CCW,
            cullMode: native::WGPUCullMode_None,
            unclippedDepth: 0,
        }
    }

    #[test]
    fn map_render_pipeline_descriptor_decodes_vertex_fragment_and_error_path() {
        let device = device_impl();
        let shader = shader_module_handle(&device);
        let attribute = native::WGPUVertexAttribute {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUVertexFormat_Float32x2,
            offset: 0,
            shaderLocation: 1,
        };
        let vertex_buffer = native::WGPUVertexBufferLayout {
            nextInChain: std::ptr::null_mut(),
            stepMode: native::WGPUVertexStepMode_Vertex,
            arrayStride: 8,
            attributeCount: 1,
            attributes: &attribute,
        };
        let color_target = native::WGPUColorTargetState {
            nextInChain: std::ptr::null_mut(),
            format: native::WGPUTextureFormat_RGBA8Unorm,
            blend: std::ptr::null(),
            writeMask: native::WGPUColorWriteMask_All,
        };
        let fragment = native::WGPUFragmentState {
            nextInChain: std::ptr::null_mut(),
            module: shader,
            entryPoint: string_view(b"fs_main"),
            constantCount: 0,
            constants: std::ptr::null(),
            targetCount: 1,
            targets: &color_target,
        };
        let descriptor = native::WGPURenderPipelineDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            layout: std::ptr::null(),
            vertex: native::WGPUVertexState {
                nextInChain: std::ptr::null_mut(),
                module: shader,
                entryPoint: string_view(b"vs_main"),
                constantCount: 0,
                constants: std::ptr::null(),
                bufferCount: 1,
                buffers: &vertex_buffer,
            },
            primitive: primitive_state(),
            depthStencil: std::ptr::null(),
            multisample: native::WGPUMultisampleState {
                nextInChain: std::ptr::null_mut(),
                count: 1,
                mask: u32::MAX,
                alphaToCoverageEnabled: 0,
            },
            fragment: &fragment,
        };
        let mapped = unsafe { map_render_pipeline_descriptor(&descriptor) };
        assert_eq!(mapped.vertex.shader.entry_point.as_deref(), Some("vs_main"));
        assert_eq!(mapped.vertex.buffer_count, 1);
        assert_eq!(mapped.vertex.buffers[0].array_stride, 8);
        assert_eq!(mapped.fragment.as_ref().expect("fragment").target_count, 1);
        assert_eq!(mapped.error, None);

        let invalid_vertex = native::WGPUVertexState {
            bufferCount: 1,
            buffers: std::ptr::null(),
            ..descriptor.vertex
        };
        let invalid = native::WGPURenderPipelineDescriptor {
            vertex: invalid_vertex,
            ..descriptor
        };
        assert!(unsafe { map_render_pipeline_descriptor(&invalid) }
            .error
            .expect("error")
            .contains("vertex buffers"));
        unsafe { release_handle(shader, "WGPUShaderModule") };
    }

    #[test]
    fn map_query_set_descriptor_decodes_type_count_label() {
        let descriptor = native::WGPUQuerySetDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: string_view(b"query-set"),
            type_: native::WGPUQueryType_Timestamp,
            count: 4,
        };
        let mapped = unsafe { map_query_set_descriptor(&descriptor) };
        assert_eq!(mapped.label, "query-set");
        assert_eq!(mapped.kind, core::QueryType::Timestamp);
        assert_eq!(mapped.count, 4);
    }

    #[test]
    fn map_render_pass_descriptor_decodes_color_attachment_and_sparse_null_view() {
        let device = device_impl();
        let view = texture_view_handle(&device);
        let attachment = native::WGPURenderPassColorAttachment {
            nextInChain: std::ptr::null_mut(),
            view,
            depthSlice: native::WGPU_DEPTH_SLICE_UNDEFINED,
            resolveTarget: std::ptr::null(),
            loadOp: native::WGPULoadOp_Clear,
            storeOp: native::WGPUStoreOp_Store,
            clearValue: native::WGPUColor {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 0.4,
            },
        };
        let descriptor = native::WGPURenderPassDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorAttachmentCount: 1,
            colorAttachments: &attachment,
            depthStencilAttachment: std::ptr::null(),
            occlusionQuerySet: std::ptr::null(),
            timestampWrites: std::ptr::null(),
        };
        let mapped = unsafe { map_render_pass_descriptor(&descriptor, 4) };
        let color = mapped.color_attachments[0].as_ref().expect("color");
        assert_eq!(color.load_op, core::LoadOp::Clear);
        assert_eq!(color.store_op, core::StoreOp::Store);
        assert_eq!(color.clear_value.g, 0.2);

        let null_attachment = native::WGPURenderPassColorAttachment {
            view: std::ptr::null(),
            ..attachment
        };
        let null_descriptor = native::WGPURenderPassDescriptor {
            colorAttachments: &null_attachment,
            ..descriptor
        };
        assert!(
            unsafe { map_render_pass_descriptor(&null_descriptor, 4) }.color_attachments[0]
                .is_none()
        );
        unsafe { release_handle(view, "WGPUTextureView") };
    }

    #[test]
    fn map_render_bundle_encoder_descriptor_decodes_formats_and_null_format_array() {
        let formats = [
            native::WGPUTextureFormat_RGBA8Unorm,
            native::WGPUTextureFormat_Undefined,
        ];
        let descriptor = native::WGPURenderBundleEncoderDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            colorFormatCount: 2,
            colorFormats: formats.as_ptr(),
            depthStencilFormat: native::WGPUTextureFormat_Depth24Plus,
            sampleCount: 4,
            depthReadOnly: 1,
            stencilReadOnly: 0,
        };
        let mapped = unsafe { map_render_bundle_encoder_descriptor(&descriptor, 4) };
        assert_eq!(mapped.color_formats.len(), 2);
        assert_eq!(
            mapped.color_formats[0],
            Some(native::WGPUTextureFormat_RGBA8Unorm.into())
        );
        assert_eq!(mapped.color_formats[1], None);
        assert_eq!(
            mapped.depth_stencil_format,
            Some(native::WGPUTextureFormat_Depth24Plus.into())
        );
        assert_eq!(mapped.sample_count, 4);
        assert!(mapped.depth_read_only);

        let null_descriptor = native::WGPURenderBundleEncoderDescriptor {
            colorFormats: std::ptr::null(),
            ..descriptor
        };
        assert_eq!(
            unsafe { map_render_bundle_encoder_descriptor(&null_descriptor, 4) }.color_formats,
            vec![None, None]
        );
    }

    #[test]
    fn map_texture_descriptor_decodes_usage_format_dimension_size_and_view_formats() {
        let view_formats = [native::WGPUTextureFormat_RGBA8UnormSrgb];
        let descriptor = native::WGPUTextureDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            usage: native::WGPUTextureUsage_TextureBinding
                | native::WGPUTextureUsage_RenderAttachment,
            dimension: native::WGPUTextureDimension_3D,
            size: native::WGPUExtent3D {
                width: 8,
                height: 9,
                depthOrArrayLayers: 10,
            },
            format: native::WGPUTextureFormat_RGBA8Unorm,
            mipLevelCount: 3,
            sampleCount: 1,
            viewFormatCount: 1,
            viewFormats: view_formats.as_ptr(),
        };
        let mapped = unsafe { map_texture_descriptor(&descriptor) };
        assert_eq!(mapped.usage.bits(), descriptor.usage);
        assert_eq!(mapped.dimension, core::TextureDimension::D3);
        assert_eq!(mapped.size.width, 8);
        assert_eq!(mapped.format, native::WGPUTextureFormat_RGBA8Unorm.into());
        assert_eq!(mapped.view_formats.len(), 1);

        let null_view_formats = native::WGPUTextureDescriptor {
            viewFormats: std::ptr::null(),
            ..descriptor
        };
        assert!(unsafe { map_texture_descriptor(&null_view_formats) }
            .view_formats
            .is_empty());
    }

    #[test]
    fn map_texture_view_descriptor_decodes_fields_and_none_defaults() {
        let descriptor = native::WGPUTextureViewDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
            format: native::WGPUTextureFormat_RGBA8Unorm,
            dimension: native::WGPUTextureViewDimension_2DArray,
            baseMipLevel: 2,
            mipLevelCount: 3,
            baseArrayLayer: 4,
            arrayLayerCount: 5,
            aspect: native::WGPUTextureAspect_All,
            usage: native::WGPUTextureUsage_TextureBinding,
        };
        let mapped = map_texture_view_descriptor(Some(&descriptor));
        assert_eq!(
            mapped.format,
            Some(native::WGPUTextureFormat_RGBA8Unorm.into())
        );
        assert_eq!(mapped.dimension, Some(core::TextureViewDimension::D2Array));
        assert_eq!(mapped.base_mip_level, 2);
        assert_eq!(mapped.mip_level_count, Some(3));
        assert_eq!(mapped.base_array_layer, 4);
        assert_eq!(mapped.array_layer_count, Some(5));
        assert_eq!(mapped.aspect, Some(core::TextureAspect::All));

        let defaulted = map_texture_view_descriptor(None);
        assert_eq!(defaulted.format, None);
        assert_eq!(defaulted.dimension, None);
        assert_eq!(defaulted.mip_level_count, None);
        assert_eq!(defaulted.array_layer_count, None);
    }

    #[test]
    fn map_device_lost_callback_info_round_trips_present_and_absent_callback() {
        unsafe extern "C" fn callback(
            _device: *const native::WGPUDevice,
            _reason: native::WGPUDeviceLostReason,
            _message: native::WGPUStringView,
            _userdata1: *mut c_void,
            _userdata2: *mut c_void,
        ) {
        }

        let native_info = native::WGPUDeviceLostCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_WaitAnyOnly,
            callback: Some(callback),
            userdata1: 0x1234usize as *mut c_void,
            userdata2: 0x5678usize as *mut c_void,
        };
        let mapped = map_device_lost_callback_info(native_info);
        assert_eq!(mapped.mode, native::WGPUCallbackMode_WaitAnyOnly);
        assert!(mapped.has_callback());
        assert_eq!(mapped.userdata1, 0x1234);
        assert_eq!(mapped.userdata2, 0x5678);

        let absent = map_device_lost_callback_info(native::WGPUDeviceLostCallbackInfo {
            callback: None,
            ..native_info
        });
        assert!(!absent.has_callback());
    }
}
