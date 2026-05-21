use super::*;

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
