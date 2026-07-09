use super::*;
use std::ffi::CStr;

/// Returns string view.
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
    // Labels preserve `{NULL, 0}` as the empty string, while `string_view_to_str`
    // keeps treating it as absent for non-label defaults such as entry points.
    if value.length == 0 {
        return Some(String::new());
    }
    if value.data.is_null() {
        return None;
    }

    let bytes = if value.length == WGPU_STRLEN {
        CStr::from_ptr(value.data).to_bytes()
    } else {
        std::slice::from_raw_parts(value.data.cast::<u8>(), value.length)
    };

    std::str::from_utf8(bytes).ok().map(ToOwned::to_owned)
}
