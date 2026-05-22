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
/// SType value for `YaWGPUInstanceBackendSelect`.
pub const YAWGPU_STYPE_INSTANCE_BACKEND_SELECT: native::WGPUSType = 0x7000_0001;
/// Feature value for tiled multi-subpass render passes.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUFeatureName_MultiSubpass: native::WGPUFeatureName = 0x7001_0001;
/// Feature value for tiled transient attachments.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUFeatureName_TransientAttachments: native::WGPUFeatureName = 0x7001_0002;
/// Feature value for tiled shader framebuffer fetch.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUFeatureName_ShaderFramebufferFetch: native::WGPUFeatureName = 0x7001_0003;
/// Feature value for tiled programmable tile dispatch.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUFeatureName_ProgrammableTileDispatch: native::WGPUFeatureName = 0x7001_0004;

/// yawgpu vendor extension for selecting a backend at instance creation.
///
/// Chain this from `WGPUInstanceDescriptor::nextInChain` with
/// `YAWGPU_STYPE_INSTANCE_BACKEND_SELECT`. This mirrors the declaration in
/// yawgpu.h and native-only backend selection extensions.
#[repr(C)]
pub struct YaWGPUInstanceBackendSelect {
    /// Chain.
    pub chain: native::WGPUChainedStruct,
    /// Backend.
    pub backend: u32,
}

/// yawgpu vendor extension result for querying tiled rendering capabilities.
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUTiledCapabilities {
    /// Extension chain.
    pub nextInChain: *const native::WGPUChainedStruct,
    /// Maximum number of subpasses in one tiled render pass.
    pub maxSubpasses: u32,
    /// Maximum number of color attachments in a subpass.
    pub maxSubpassColorAttachments: u32,
    /// Maximum number of input attachments in a subpass.
    pub maxInputAttachments: u32,
    /// Estimated tile memory budget, in bytes.
    pub estimatedTileMemoryBytes: u32,
}

/// yawgpu transient attachment handle.
#[cfg(feature = "tiled")]
pub type YaWGPUTransientAttachment = *const YaWGPUTransientAttachmentImpl;
/// Transient attachment size mode for tiled rendering.
#[cfg(feature = "tiled")]
pub type YaWGPUTransientSizeMode = u32;
/// Match the subpass render target size at pass begin.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUTransientSizeMode_MatchTarget: YaWGPUTransientSizeMode = 0;
/// Use the descriptor's explicit width and height.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUTransientSizeMode_Explicit: YaWGPUTransientSizeMode = 1;
/// Force this enum to 32 bits in C.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUTransientSizeMode_Force32: YaWGPUTransientSizeMode = 0x7FFF_FFFF;

/// yawgpu vendor extension descriptor for creating a transient attachment.
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUTransientAttachmentDescriptor {
    /// Extension chain.
    pub nextInChain: *const native::WGPUChainedStruct,
    /// Debug label.
    pub label: native::WGPUStringView,
    /// Attachment format.
    pub format: native::WGPUTextureFormat,
    /// Size mode.
    pub sizeMode: YaWGPUTransientSizeMode,
    /// Explicit width. Ignored for match-target attachments.
    pub width: u32,
    /// Explicit height. Ignored for match-target attachments.
    pub height: u32,
    /// Sample count.
    pub sampleCount: u32,
}

/// yawgpu vendor extension descriptor for creating a shader module from SPIR-V words.
#[cfg(feature = "shader-passthrough")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUShaderModuleSpirVDescriptor {
    /// Extension chain.
    pub nextInChain: *const native::WGPUChainedStruct,
    /// Debug label.
    pub label: native::WGPUStringView,
    /// Number of `u32` words in `code`.
    pub codeSize: u32,
    /// SPIR-V words.
    pub code: *const u32,
}

/// yawgpu vendor extension MSL entry-point metadata.
#[cfg(feature = "shader-passthrough")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUMslEntryPoint {
    /// Entry-point name.
    pub name: native::WGPUStringView,
    /// Standard `WGPUShaderStage` bitflag. Exactly one bit must be set.
    pub stage: native::WGPUShaderStage,
    /// Compute workgroup size. Ignored for non-compute stages.
    pub workgroupSize: [u32; 3],
}

/// yawgpu vendor extension descriptor for creating a shader module from MSL source.
#[cfg(feature = "shader-passthrough")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUShaderModuleMslDescriptor {
    /// Extension chain.
    pub nextInChain: *const native::WGPUChainedStruct,
    /// Debug label.
    pub label: native::WGPUStringView,
    /// MSL source code.
    pub code: native::WGPUStringView,
    /// Number of entries in `entryPoints`.
    pub entryPointCount: usize,
    /// Caller-supplied entry-point metadata.
    pub entryPoints: *const YaWGPUMslEntryPoint,
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
