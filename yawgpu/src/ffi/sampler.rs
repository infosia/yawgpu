use super::*;

wgpu_handle_exports!(
    refcount_and_label:
    WGPUSamplerImpl,
    native::WGPUSampler,
    "WGPUSampler",
    wgpuSamplerAddRef,
    wgpuSamplerRelease,
    wgpuSamplerSetLabel
);
