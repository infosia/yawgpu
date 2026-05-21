use super::*;

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
