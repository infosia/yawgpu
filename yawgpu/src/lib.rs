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
// 0x7001_0004 reserved — see `yawgpu.h` "Programmable tile dispatch — removed".
/// SType value for `YaWGPUInputAttachmentBindingLayout`.
#[cfg(feature = "tiled")]
pub const YAWGPU_STYPE_INPUT_ATTACHMENT_BINDING_LAYOUT: native::WGPUSType = 0x7000_0010;

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

/// yawgpu vendor extension bind group layout entry for input attachments.
///
/// Chain this from `WGPUBindGroupLayoutEntry::nextInChain`. When this chain is
/// present the enclosing entry's `buffer`/`sampler`/`texture`/`storageTexture`
/// fields must be left zero-initialized; setting any standard binding-layout
/// field alongside this chain is rejected at bind-group-layout creation. The
/// resource itself is auto-wired from the subpass pass layout, so the matching
/// `WGPUBindGroupEntry` must be omitted from the bind group's `entries` array
/// (supplying any entry for that binding is rejected at bind-group creation).
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUInputAttachmentBindingLayout {
    /// Extension chain.
    pub chain: native::WGPUChainedStruct,
    /// Input attachment sample type.
    pub sampleType: native::WGPUTextureSampleType,
    /// Whether the input attachment is multisampled.
    pub multisampled: native::WGPUBool,
}

/// yawgpu subpass pass layout handle.
#[cfg(feature = "tiled")]
pub type YaWGPUSubpassPassLayout = *const YaWGPUSubpassPassLayoutImpl;
/// yawgpu subpass render pass encoder handle.
#[cfg(feature = "tiled")]
pub type YaWGPUSubpassRenderPassEncoder = *const YaWGPUSubpassRenderPassEncoderImpl;
/// Depth-stencil source attachment sentinel.
#[cfg(feature = "tiled")]
pub const YAWGPU_DEPTH_STENCIL_ATTACHMENT_INDEX: u32 = u32::MAX;
/// Subpass dependency type.
#[cfg(feature = "tiled")]
pub type YaWGPUSubpassDependencyType = u32;
/// Color-to-input dependency.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUSubpassDependencyType_ColorToInput: YaWGPUSubpassDependencyType = 0;
/// Depth-to-input dependency.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUSubpassDependencyType_DepthToInput: YaWGPUSubpassDependencyType = 1;
/// Color-depth-to-input dependency.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUSubpassDependencyType_ColorDepthToInput: YaWGPUSubpassDependencyType = 2;
/// Subpass attachment kind.
#[cfg(feature = "tiled")]
pub type YaWGPUSubpassAttachmentKind = u32;
/// Persistent texture view attachment kind.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUSubpassAttachmentKind_Persistent: YaWGPUSubpassAttachmentKind = 0;
/// Transient attachment kind.
#[cfg(feature = "tiled")]
#[allow(non_upper_case_globals)]
pub const YaWGPUSubpassAttachmentKind_Transient: YaWGPUSubpassAttachmentKind = 1;

/// yawgpu subpass attachment layout.
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUAttachmentLayout {
    /// Format.
    pub format: native::WGPUTextureFormat,
    /// Sample count.
    pub sampleCount: u32,
}

/// yawgpu subpass dependency.
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUSubpassDependency {
    /// Source subpass.
    pub srcSubpass: u32,
    /// Destination subpass.
    pub dstSubpass: u32,
    /// Dependency type.
    pub dependencyType: YaWGPUSubpassDependencyType,
    /// Whether dependency is region-local.
    pub byRegion: native::WGPUBool,
}

/// yawgpu subpass input attachment source.
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUSubpassInputAttachment {
    /// Bind group.
    pub group: u32,
    /// Binding.
    pub binding: u32,
    /// Source subpass.
    pub sourceSubpass: u32,
    /// Source attachment.
    pub sourceAttachment: u32,
}

/// yawgpu subpass layout descriptor.
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUSubpassLayoutDesc {
    /// Color attachment indices.
    pub colorAttachmentIndices: *const u32,
    /// Color attachment index count.
    pub colorAttachmentIndexCount: usize,
    /// Uses depth-stencil.
    pub usesDepthStencil: native::WGPUBool,
    /// Input attachments.
    pub inputAttachments: *const YaWGPUSubpassInputAttachment,
    /// Input attachment count.
    pub inputAttachmentCount: usize,
}

/// yawgpu subpass pass layout descriptor.
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUSubpassPassLayoutDescriptor {
    /// Extension chain.
    pub nextInChain: *const native::WGPUChainedStruct,
    /// Label.
    pub label: native::WGPUStringView,
    /// Color attachments.
    pub colorAttachments: *const YaWGPUAttachmentLayout,
    /// Color attachment count.
    pub colorAttachmentCount: usize,
    /// Depth-stencil attachment. Undefined format means absent.
    pub depthStencilAttachment: YaWGPUAttachmentLayout,
    /// Subpasses.
    pub subpasses: *const YaWGPUSubpassLayoutDesc,
    /// Subpass count.
    pub subpassCount: usize,
    /// Dependencies.
    pub dependencies: *const YaWGPUSubpassDependency,
    /// Dependency count.
    pub dependencyCount: usize,
}

/// yawgpu subpass render pipeline descriptor.
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUSubpassRenderPipelineDescriptor {
    /// Extension chain.
    pub nextInChain: *const native::WGPUChainedStruct,
    /// Base render pipeline descriptor. `fragment.targets`,
    /// `multisample.count`, and `depthStencil` must match the subpass selected
    /// by `passLayout` / `subpassIndex` (count, per-target `format`, sample
    /// count, and depth-stencil format/presence are all validated at pipeline
    /// creation; mismatches return a NULL pipeline with an error pushed to
    /// the device's error scope).
    pub base: native::WGPURenderPipelineDescriptor,
    /// Compatible pass layout.
    pub passLayout: YaWGPUSubpassPassLayout,
    /// Compatible subpass index.
    pub subpassIndex: u32,
}

/// yawgpu color attachment binding.
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUColorAttachmentBinding {
    /// Attachment kind.
    pub kind: YaWGPUSubpassAttachmentKind,
    /// Persistent view.
    pub view: native::WGPUTextureView,
    /// Persistent resolve target.
    pub resolveTarget: native::WGPUTextureView,
    /// Transient attachment.
    pub transient: YaWGPUTransientAttachment,
    /// Load op.
    pub loadOp: native::WGPULoadOp,
    /// Store op.
    pub storeOp: native::WGPUStoreOp,
    /// Clear value.
    pub clearValue: native::WGPUColor,
}

/// yawgpu depth-stencil attachment binding.
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUDepthStencilAttachmentBinding {
    /// Attachment kind.
    pub kind: YaWGPUSubpassAttachmentKind,
    /// Persistent view.
    pub view: native::WGPUTextureView,
    /// Transient attachment.
    pub transient: YaWGPUTransientAttachment,
    /// Depth load op.
    pub depthLoadOp: native::WGPULoadOp,
    /// Depth store op.
    pub depthStoreOp: native::WGPUStoreOp,
    /// Depth clear value.
    pub depthClearValue: f32,
    /// Stencil load op.
    pub stencilLoadOp: native::WGPULoadOp,
    /// Stencil store op.
    pub stencilStoreOp: native::WGPUStoreOp,
    /// Stencil clear value.
    pub stencilClearValue: u32,
}

/// yawgpu subpass render pass descriptor.
#[cfg(feature = "tiled")]
#[allow(non_snake_case)]
#[repr(C)]
pub struct YaWGPUSubpassRenderPassDescriptor {
    /// Extension chain.
    pub nextInChain: *const native::WGPUChainedStruct,
    /// Label.
    pub label: native::WGPUStringView,
    /// Pass layout.
    pub passLayout: YaWGPUSubpassPassLayout,
    /// Extent.
    pub extent: native::WGPUExtent3D,
    /// Color attachments.
    pub colorAttachments: *const YaWGPUColorAttachmentBinding,
    /// Color attachment count.
    pub colorAttachmentCount: usize,
    /// Depth-stencil attachment.
    pub depthStencilAttachment: *const YaWGPUDepthStencilAttachmentBinding,
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
