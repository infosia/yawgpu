use super::*;

/// Sets a sampler label.
///
/// # Safety
///
/// `sampler` must be a non-null live yawgpu sampler handle. `label` must point
/// to valid string data according to `WGPUStringView` when non-empty.
/// Returns WGPU sampler set label.
#[no_mangle]
pub unsafe extern "C" fn wgpuSamplerSetLabel(
    sampler: native::WGPUSampler,
    label: native::WGPUStringView,
) {
    let sampler = borrow_handle(sampler, "WGPUSampler");
    *sampler.label.lock().expect("label lock must not poison") = label_from_string_view(label);
}

/// Releases one owned reference to a sampler handle.
///
/// # Safety
///
/// `sampler` must be a non-null live yawgpu sampler handle.
/// Returns WGPU sampler release.
#[no_mangle]
pub unsafe extern "C" fn wgpuSamplerRelease(sampler: native::WGPUSampler) {
    release_handle(sampler, "WGPUSampler");
}

/// Adds one owned reference to a sampler handle.
///
/// # Safety
///
/// `sampler` must be a non-null live yawgpu sampler handle.
/// Returns WGPU sampler add ref.
#[no_mangle]
pub unsafe extern "C" fn wgpuSamplerAddRef(sampler: native::WGPUSampler) {
    add_ref_handle(sampler, "WGPUSampler");
}
