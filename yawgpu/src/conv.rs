use std::ffi::CStr;
use std::sync::Arc;

use crate::native;

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
