pub mod conv;
mod ffi;

pub use ffi::adapter::*;
pub use ffi::bindings::*;
pub use ffi::buffer::*;
pub use ffi::bundle::*;
pub use ffi::command_buffer::*;
pub use ffi::compute_pass::*;
pub use ffi::device::*;
pub use ffi::encoder::*;
pub use ffi::instance::*;
pub use ffi::pipelines::*;
pub use ffi::query::*;
pub use ffi::queue::*;
pub use ffi::render_pass::*;
pub use ffi::sampler::*;
pub use ffi::shader::*;
pub use ffi::surface::*;
pub use ffi::texture::*;
pub use ffi::*;

pub const WGPU_YAWGPU_INSTANCE_BACKEND_NOOP: u32 = 0;
pub const WGPU_YAWGPU_INSTANCE_BACKEND_METAL: u32 = 1;
pub const WGPU_YAWGPU_INSTANCE_BACKEND_VULKAN: u32 = 2;
pub const WGPU_STYPE_YAWGPU_INSTANCE_BACKEND_SELECT: native::WGPUSType = 0x7000_0001;

/// yawgpu vendor extension for selecting a backend at instance creation.
///
/// Chain this from `WGPUInstanceDescriptor::nextInChain` with
/// `WGPU_STYPE_YAWGPU_INSTANCE_BACKEND_SELECT`. This is intentionally outside
/// webgpu.h and mirrors native-only backend selection extensions.
#[repr(C)]
pub struct WGPUYawgpuInstanceBackendSelect {
    pub chain: native::WGPUChainedStruct,
    pub backend: u32,
}

pub mod native {
    #![allow(
        dead_code,
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        improper_ctypes
    )]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
