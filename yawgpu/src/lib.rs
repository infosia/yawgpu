#![warn(missing_docs)]
//! C WebGPU API surface backed by yawgpu-core and yawgpu-hal.

/// Conv module.
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

/// Constant value for the yawgpu Noop instance backend.
pub const YAWGPU_INSTANCE_BACKEND_NOOP: u32 = 0;
/// Constant value for the yawgpu Metal instance backend.
pub const YAWGPU_INSTANCE_BACKEND_METAL: u32 = 1;
/// Constant value for the yawgpu Vulkan instance backend.
pub const YAWGPU_INSTANCE_BACKEND_VULKAN: u32 = 2;
/// Constant value for the yawgpu GLES (Tier 2 / experimental) instance backend.
/// Requires the `gles` cargo feature; otherwise (per the IB3 rules on
/// `YaWGPUInstanceBackendSelect`) `wgpuCreateInstance` returns NULL when this
/// backend is explicitly requested.
pub const YAWGPU_INSTANCE_BACKEND_GLES: u32 = 3;
/// SType value for `YaWGPUInstanceBackendSelect`.
pub const YAWGPU_STYPE_INSTANCE_BACKEND_SELECT: native::WGPUSType = 0x7000_0001;
/// Constant value that defers GLES context backend selection to the env var.
pub const YAWGPU_GLES_CONTEXT_BACKEND_DEFAULT: u32 = 0;
/// Constant value for the EGL GLES context backend.
pub const YAWGPU_GLES_CONTEXT_BACKEND_EGL: u32 = 1;
/// Constant value for the Windows WGL GLES context backend.
pub const YAWGPU_GLES_CONTEXT_BACKEND_WGL: u32 = 2;
/// SType value for `YaWGPUGlesContextBackend`.
pub const YAWGPU_STYPE_GLES_CONTEXT_BACKEND: native::WGPUSType = 0x7000_0002;

/// yawgpu vendor extension for selecting a backend at instance creation.
///
/// Chain this from `WGPUInstanceDescriptor::nextInChain` with
/// `YAWGPU_STYPE_INSTANCE_BACKEND_SELECT`. This mirrors the declaration in
/// yawgpu.h and native-only backend selection extensions. Resolution rules
/// applied by `wgpuCreateInstance` (return value in parentheses):
///
/// - **No chain entry present** (`nextInChain` does not contain a
///   `YaWGPUInstanceBackendSelect`): a Noop instance is returned (non-NULL).
/// - **`backend == YAWGPU_INSTANCE_BACKEND_NOOP`**: a Noop instance is
///   returned (non-NULL).
/// - **`backend == YAWGPU_INSTANCE_BACKEND_{METAL, VULKAN, GLES}`**: strict.
///   `wgpuCreateInstance` returns NULL when the matching cargo feature was
///   not compiled into this yawgpu build, when the backend's instance
///   creation fails, or when the backend exposes no adapters. A best-effort
///   diagnostic line is written to `stderr` identifying which cause fired
///   (the only in-band signal is the NULL return; webgpu.h does not provide
///   an error callback on `wgpuCreateInstance`). Callers wanting to confirm
///   which backend was selected may inspect
///   `wgpuAdapterGetInfo().backendType` after `wgpuInstanceRequestAdapter`.
/// - **Unrecognised `backend` value** (anything outside the four constants
///   above): treated as if no chain were present and returns a Noop instance
///   (non-NULL). This keeps older yawgpu builds forward-compatible with
///   descriptors produced from a newer header that may define additional
///   backend constants.
#[repr(C)]
pub struct YaWGPUInstanceBackendSelect {
    /// Chain.
    pub chain: native::WGPUChainedStruct,
    /// Backend.
    pub backend: u32,
}

/// yawgpu vendor extension for selecting the GLES context backend.
///
/// Chain this from `WGPUInstanceDescriptor::nextInChain` with
/// `YAWGPU_STYPE_GLES_CONTEXT_BACKEND`. The value is only consumed when the
/// resolved instance backend is GLES.
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUGlesContextBackend {
    /// Extension chain.
    pub chain: native::WGPUChainedStruct,
    /// GLES context backend.
    pub contextBackend: u32,
}

/// Native module.
pub mod native {
    #![allow(
        dead_code,
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        improper_ctypes,
        missing_docs,
        rustdoc::broken_intra_doc_links
    )]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
