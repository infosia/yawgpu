use super::*;

wgpu_handle_exports!(
    refcount:
    WGPUQuerySetImpl,
    native::WGPUQuerySet,
    "WGPUQuerySet",
    wgpuQuerySetAddRef,
    wgpuQuerySetRelease
);

/// Destroys a query set. This operation is idempotent.
///
/// # Safety
///
/// `query_set` must be a non-null live yawgpu query set handle.
/// Returns WGPU query set destroy.
#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetDestroy(query_set: native::WGPUQuerySet) {
    borrow_handle(query_set, "WGPUQuerySet").core.destroy();
}

/// Returns the descriptor query type reflected by the query set.
///
/// # Safety
///
/// `query_set` must be a non-null live yawgpu query set handle.
/// Returns WGPU query set get type.
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
/// Returns WGPU query set get count.
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
/// Returns WGPU query set set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuQuerySetSetLabel(
    query_set: native::WGPUQuerySet,
    label: native::WGPUStringView,
) {
    let query_set = borrow_handle(query_set, "WGPUQuerySet");
    let label = string_view_to_str(label).unwrap_or_default();
    query_set.core.set_label(label);
}
