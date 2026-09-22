//! `wgpuGetProcAddress` — name → exported entry-point lookup (Block 101).
//!
//! The export table below is the single place that enumerates every C
//! entry point the crate exports: all `wgpu*` functions declared in the
//! pinned `webgpu.h` (including the macro-generated `*AddRef` /
//! `*Release` / `*SetLabel` exports and `wgpuGetProcAddress` itself) and
//! the `yawgpu*` vendor entry points of `yawgpu.h`. Feature-gated vendor
//! exports carry the same `#[cfg]` as their definition, so they resolve
//! only when compiled in.

use super::*;

/// Converts a function's address into the `WGPUProc` erased fn-pointer
/// type.
///
/// Rust does not allow an `as` cast between fn-pointer types of different
/// signatures (E0605), so each table arm first casts the fn item to a
/// `*const ()` with `as` — which only compiles when the identifier names
/// a function — and this helper performs the single, fixed-type
/// pointer → `unsafe extern "C" fn()` conversion. The pointer is never
/// called through this type inside Rust; C callers cast it back to the
/// entry point's declared signature, exactly as with `dlsym`.
fn erase_fn_pointer(address: *const ()) -> unsafe extern "C" fn() {
    // SAFETY: `address` is the address of a real `extern "C"` function
    // (every table arm derives it from a fn item), so it is non-null and
    // a valid function pointer; fn pointers and `*const ()` share the
    // same size and representation on all supported targets.
    unsafe { std::mem::transmute::<*const (), unsafe extern "C" fn()>(address) }
}

/// Expands to the `&str` → `WGPUProc` lookup for the given identifier list.
///
/// Each arm is `stringify!(name) => Some(<address of name>)`; an
/// optional `#[cfg(...)]` before an identifier gates its arm.
macro_rules! wgpu_proc_table {
    ($name:ident; $( $(#[$meta:meta])* $proc:ident ),* $(,)?) => {
        match $name {
            $(
                $(#[$meta])*
                stringify!($proc) => Some(erase_fn_pointer(crate::$proc as *const ())),
            )*
            _ => None,
        }
    };
}

/// Resolves an exported entry point by its exact C symbol name.
fn lookup(name: &str) -> native::WGPUProc {
    wgpu_proc_table!(name;
        // --- webgpu.h: free functions ---
        wgpuCreateInstance,
        wgpuGetInstanceFeatures,
        wgpuGetInstanceLimits,
        wgpuGetProcAddress,
        wgpuHasInstanceFeature,
        // --- webgpu.h: *FreeMembers ---
        wgpuAdapterInfoFreeMembers,
        wgpuSupportedFeaturesFreeMembers,
        wgpuSupportedInstanceFeaturesFreeMembers,
        wgpuSupportedWGSLLanguageFeaturesFreeMembers,
        wgpuSurfaceCapabilitiesFreeMembers,
        // --- Adapter ---
        wgpuAdapterAddRef,
        wgpuAdapterGetFeatures,
        wgpuAdapterGetInfo,
        wgpuAdapterGetLimits,
        wgpuAdapterHasFeature,
        wgpuAdapterRelease,
        wgpuAdapterRequestDevice,
        // --- BindGroup / BindGroupLayout ---
        wgpuBindGroupAddRef,
        wgpuBindGroupRelease,
        wgpuBindGroupSetLabel,
        wgpuBindGroupLayoutAddRef,
        wgpuBindGroupLayoutRelease,
        wgpuBindGroupLayoutSetLabel,
        // --- Buffer ---
        wgpuBufferAddRef,
        wgpuBufferDestroy,
        wgpuBufferGetConstMappedRange,
        wgpuBufferGetMapState,
        wgpuBufferGetMappedRange,
        wgpuBufferGetSize,
        wgpuBufferGetUsage,
        wgpuBufferMapAsync,
        wgpuBufferReadMappedRange,
        wgpuBufferRelease,
        wgpuBufferSetLabel,
        wgpuBufferUnmap,
        wgpuBufferWriteMappedRange,
        // --- CommandBuffer ---
        wgpuCommandBufferAddRef,
        wgpuCommandBufferRelease,
        wgpuCommandBufferSetLabel,
        // --- CommandEncoder ---
        wgpuCommandEncoderAddRef,
        wgpuCommandEncoderBeginComputePass,
        wgpuCommandEncoderBeginRenderPass,
        wgpuCommandEncoderClearBuffer,
        wgpuCommandEncoderCopyBufferToBuffer,
        wgpuCommandEncoderCopyBufferToTexture,
        wgpuCommandEncoderCopyTextureToBuffer,
        wgpuCommandEncoderCopyTextureToTexture,
        wgpuCommandEncoderFinish,
        wgpuCommandEncoderInsertDebugMarker,
        wgpuCommandEncoderPopDebugGroup,
        wgpuCommandEncoderPushDebugGroup,
        wgpuCommandEncoderRelease,
        wgpuCommandEncoderResolveQuerySet,
        wgpuCommandEncoderSetLabel,
        wgpuCommandEncoderWriteBuffer,
        wgpuCommandEncoderWriteTimestamp,
        // --- ComputePassEncoder ---
        wgpuComputePassEncoderAddRef,
        wgpuComputePassEncoderDispatchWorkgroups,
        wgpuComputePassEncoderDispatchWorkgroupsIndirect,
        wgpuComputePassEncoderEnd,
        wgpuComputePassEncoderInsertDebugMarker,
        wgpuComputePassEncoderPopDebugGroup,
        wgpuComputePassEncoderPushDebugGroup,
        wgpuComputePassEncoderRelease,
        wgpuComputePassEncoderSetBindGroup,
        wgpuComputePassEncoderSetImmediates,
        wgpuComputePassEncoderSetLabel,
        wgpuComputePassEncoderSetPipeline,
        // --- ComputePipeline ---
        wgpuComputePipelineAddRef,
        wgpuComputePipelineGetBindGroupLayout,
        wgpuComputePipelineRelease,
        wgpuComputePipelineSetLabel,
        // --- Device ---
        wgpuDeviceAddRef,
        wgpuDeviceCreateBindGroup,
        wgpuDeviceCreateBindGroupLayout,
        wgpuDeviceCreateBuffer,
        wgpuDeviceCreateCommandEncoder,
        wgpuDeviceCreateComputePipeline,
        wgpuDeviceCreateComputePipelineAsync,
        wgpuDeviceCreatePipelineLayout,
        wgpuDeviceCreateQuerySet,
        wgpuDeviceCreateRenderBundleEncoder,
        wgpuDeviceCreateRenderPipeline,
        wgpuDeviceCreateRenderPipelineAsync,
        wgpuDeviceCreateSampler,
        wgpuDeviceCreateShaderModule,
        wgpuDeviceCreateTexture,
        wgpuDeviceDestroy,
        wgpuDeviceGetAdapterInfo,
        wgpuDeviceGetFeatures,
        wgpuDeviceGetLimits,
        wgpuDeviceGetLostFuture,
        wgpuDeviceGetQueue,
        wgpuDeviceHasFeature,
        wgpuDevicePopErrorScope,
        wgpuDevicePushErrorScope,
        wgpuDeviceRelease,
        wgpuDeviceSetLabel,
        // --- ExternalTexture ---
        wgpuExternalTextureAddRef,
        wgpuExternalTextureRelease,
        wgpuExternalTextureSetLabel,
        // --- Instance ---
        wgpuInstanceAddRef,
        wgpuInstanceCreateSurface,
        wgpuInstanceGetWGSLLanguageFeatures,
        wgpuInstanceHasWGSLLanguageFeature,
        wgpuInstanceProcessEvents,
        wgpuInstanceRelease,
        wgpuInstanceRequestAdapter,
        wgpuInstanceWaitAny,
        // --- PipelineLayout ---
        wgpuPipelineLayoutAddRef,
        wgpuPipelineLayoutRelease,
        wgpuPipelineLayoutSetLabel,
        // --- QuerySet ---
        wgpuQuerySetAddRef,
        wgpuQuerySetDestroy,
        wgpuQuerySetGetCount,
        wgpuQuerySetGetType,
        wgpuQuerySetRelease,
        wgpuQuerySetSetLabel,
        // --- Queue ---
        wgpuQueueAddRef,
        wgpuQueueOnSubmittedWorkDone,
        wgpuQueueRelease,
        wgpuQueueSetLabel,
        wgpuQueueSubmit,
        wgpuQueueWriteBuffer,
        wgpuQueueWriteTexture,
        // --- RenderBundle ---
        wgpuRenderBundleAddRef,
        wgpuRenderBundleRelease,
        wgpuRenderBundleSetLabel,
        // --- RenderBundleEncoder ---
        wgpuRenderBundleEncoderAddRef,
        wgpuRenderBundleEncoderDraw,
        wgpuRenderBundleEncoderDrawIndexed,
        wgpuRenderBundleEncoderDrawIndexedIndirect,
        wgpuRenderBundleEncoderDrawIndirect,
        wgpuRenderBundleEncoderFinish,
        wgpuRenderBundleEncoderInsertDebugMarker,
        wgpuRenderBundleEncoderPopDebugGroup,
        wgpuRenderBundleEncoderPushDebugGroup,
        wgpuRenderBundleEncoderRelease,
        wgpuRenderBundleEncoderSetBindGroup,
        wgpuRenderBundleEncoderSetImmediates,
        wgpuRenderBundleEncoderSetIndexBuffer,
        wgpuRenderBundleEncoderSetLabel,
        wgpuRenderBundleEncoderSetPipeline,
        wgpuRenderBundleEncoderSetVertexBuffer,
        // --- RenderPassEncoder ---
        wgpuRenderPassEncoderAddRef,
        wgpuRenderPassEncoderBeginOcclusionQuery,
        wgpuRenderPassEncoderDraw,
        wgpuRenderPassEncoderDrawIndexed,
        wgpuRenderPassEncoderDrawIndexedIndirect,
        wgpuRenderPassEncoderDrawIndirect,
        wgpuRenderPassEncoderEnd,
        wgpuRenderPassEncoderEndOcclusionQuery,
        wgpuRenderPassEncoderExecuteBundles,
        wgpuRenderPassEncoderInsertDebugMarker,
        wgpuRenderPassEncoderPopDebugGroup,
        wgpuRenderPassEncoderPushDebugGroup,
        wgpuRenderPassEncoderRelease,
        wgpuRenderPassEncoderSetBindGroup,
        wgpuRenderPassEncoderSetBlendConstant,
        wgpuRenderPassEncoderSetImmediates,
        wgpuRenderPassEncoderSetIndexBuffer,
        wgpuRenderPassEncoderSetLabel,
        wgpuRenderPassEncoderSetPipeline,
        wgpuRenderPassEncoderSetScissorRect,
        wgpuRenderPassEncoderSetStencilReference,
        wgpuRenderPassEncoderSetVertexBuffer,
        wgpuRenderPassEncoderSetViewport,
        // --- RenderPipeline ---
        wgpuRenderPipelineAddRef,
        wgpuRenderPipelineGetBindGroupLayout,
        wgpuRenderPipelineRelease,
        wgpuRenderPipelineSetLabel,
        // --- Sampler ---
        wgpuSamplerAddRef,
        wgpuSamplerRelease,
        wgpuSamplerSetLabel,
        // --- ShaderModule ---
        wgpuShaderModuleAddRef,
        wgpuShaderModuleGetCompilationInfo,
        wgpuShaderModuleRelease,
        wgpuShaderModuleSetLabel,
        // --- Surface ---
        wgpuSurfaceAddRef,
        wgpuSurfaceConfigure,
        wgpuSurfaceGetCapabilities,
        wgpuSurfaceGetCurrentTexture,
        wgpuSurfacePresent,
        wgpuSurfaceRelease,
        wgpuSurfaceSetLabel,
        wgpuSurfaceUnconfigure,
        // --- Texture ---
        wgpuTextureAddRef,
        wgpuTextureCreateView,
        wgpuTextureDestroy,
        wgpuTextureGetDepthOrArrayLayers,
        wgpuTextureGetDimension,
        wgpuTextureGetFormat,
        wgpuTextureGetHeight,
        wgpuTextureGetMipLevelCount,
        wgpuTextureGetSampleCount,
        wgpuTextureGetTextureBindingViewDimension,
        wgpuTextureGetUsage,
        wgpuTextureGetWidth,
        wgpuTextureRelease,
        wgpuTextureSetLabel,
        // --- TextureView ---
        wgpuTextureViewAddRef,
        wgpuTextureViewRelease,
        wgpuTextureViewSetLabel,
        // --- yawgpu.h: vendor entry points (always compiled) ---
        yawgpuDeviceCreateExternalTexture,
        // --- yawgpu.h: vendor entry points behind the `tiled` feature ---
        #[cfg(feature = "tiled")]
        yawgpuAdapterGetTiledCapabilities,
        #[cfg(feature = "tiled")]
        yawgpuCommandEncoderBeginSubpassRenderPass,
        #[cfg(feature = "tiled")]
        yawgpuDeviceCreateSubpassPassLayout,
        #[cfg(feature = "tiled")]
        yawgpuDeviceCreateSubpassRenderPipeline,
        #[cfg(feature = "tiled")]
        yawgpuSubpassPassLayoutAddRef,
        #[cfg(feature = "tiled")]
        yawgpuSubpassPassLayoutRelease,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderAddRef,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderDraw,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderDrawIndexed,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderEnd,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderNextSubpass,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderRelease,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderSetBindGroup,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderSetIndexBuffer,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderSetPipeline,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderSetScissorRect,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderSetVertexBuffer,
        #[cfg(feature = "tiled")]
        yawgpuSubpassRenderPassEncoderSetViewport,
    )
}

/// Returns the address of the exported entry point named `proc_name`, or
/// `NULL` when the crate exports no entry point of that exact name.
///
/// Every `wgpu*` function declared in the pinned `webgpu.h` resolves
/// (including this one), as does every `yawgpu*` vendor entry point in
/// `yawgpu.h` that is compiled in; feature-gated vendor entry points
/// resolve to `NULL` when their feature is not enabled. An empty, unknown,
/// prefix-only, or extended name, and a name that is not valid UTF-8,
/// yield `NULL`.
///
/// # Safety
///
/// `proc_name` must follow the `WGPUStringView` rules: when `data` is
/// non-null it must point to `length` valid bytes, or to a NUL-terminated
/// string when `length == WGPU_STRLEN`. A null `data` is accepted (it
/// denotes the empty / null string and yields `NULL`).
#[no_mangle]
pub unsafe extern "C" fn wgpuGetProcAddress(proc_name: native::WGPUStringView) -> native::WGPUProc {
    // `string_view_to_str` returns `None` for null data (the null / empty
    // string) and for non-UTF-8 bytes; neither names an export.
    let name = string_view_to_str(proc_name)?;
    lookup(name)
}

#[cfg(test)]
// The test names mirror the C entry point they cover, as required by
// `specs/blocks/101-get-proc-address.md`.
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::conv::{string_view, WGPU_STRLEN};

    /// Returns the address a resolved `WGPUProc` points at.
    fn address_of(proc_: native::WGPUProc) -> Option<usize> {
        proc_.map(|f| f as *const () as usize)
    }

    /// Resolves `name` through the C entry point using the explicit-length
    /// string-view form.
    fn resolve(name: &str) -> native::WGPUProc {
        unsafe { wgpuGetProcAddress(string_view(name.as_bytes())) }
    }

    /// Extracts the `wgpu*` identifier from a `WGPU_EXPORT` declaration
    /// line of `webgpu.h` (the identifier immediately before the first
    /// `(`), without a regex crate.
    fn declared_name(line: &str) -> Option<&str> {
        let line = line.trim_start();
        if !line.starts_with("WGPU_EXPORT") {
            return None;
        }
        let paren = line.find('(')?;
        let head = &line[..paren];
        let start = head
            .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .map_or(0, |i| i + 1);
        let name = &head[start..];
        name.starts_with("wgpu").then_some(name)
    }

    #[test]
    fn wgpuGetProcAddress_resolves_every_header_declaration() {
        let header = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/ffi/webgpu-headers/webgpu.h"
        ));
        let declared: Vec<&str> = header.lines().filter_map(declared_name).collect();
        assert_eq!(
            declared.len(),
            202,
            "webgpu.h declaration count changed; extend the table and this count"
        );

        let unresolved: Vec<&str> = declared
            .iter()
            .copied()
            .filter(|name| resolve(name).is_none())
            .collect();
        assert!(
            unresolved.is_empty(),
            "header-declared entry points missing from the table: {unresolved:?}"
        );
    }

    #[test]
    fn wgpuGetProcAddress_returns_the_exported_symbol_address() {
        let expected: [(&str, usize); 3] = [
            (
                "wgpuCreateInstance",
                crate::wgpuCreateInstance as *const () as usize,
            ),
            (
                "wgpuDeviceRelease",
                crate::wgpuDeviceRelease as *const () as usize,
            ),
            (
                "wgpuGetProcAddress",
                wgpuGetProcAddress as *const () as usize,
            ),
        ];
        for (name, address) in expected {
            assert_eq!(
                address_of(resolve(name)),
                Some(address),
                "{name} resolved to a different address"
            );
        }
    }

    #[test]
    fn wgpuGetProcAddress_resolves_vendor_exports() {
        assert_eq!(
            address_of(resolve("yawgpuDeviceCreateExternalTexture")),
            Some(crate::yawgpuDeviceCreateExternalTexture as *const () as usize)
        );

        #[cfg(feature = "tiled")]
        assert_eq!(
            address_of(resolve("yawgpuAdapterGetTiledCapabilities")),
            Some(crate::yawgpuAdapterGetTiledCapabilities as *const () as usize)
        );
        #[cfg(not(feature = "tiled"))]
        assert!(resolve("yawgpuAdapterGetTiledCapabilities").is_none());
    }

    #[test]
    fn wgpuGetProcAddress_returns_null_for_unknown_prefix_and_extension_names() {
        for name in [
            "wgpuCreate",
            "wgpuCreateInstanceX",
            "WGPUCreateInstance",
            "",
        ] {
            assert!(resolve(name).is_none(), "{name:?} must not resolve");
        }
    }

    #[test]
    fn wgpuGetProcAddress_handles_string_view_forms() {
        let expected = Some(crate::wgpuCreateInstance as *const () as usize);

        // NUL-terminated form (`length == WGPU_STRLEN`).
        let nul_terminated = native::WGPUStringView {
            data: c"wgpuCreateInstance".as_ptr(),
            length: WGPU_STRLEN,
        };
        assert_eq!(
            address_of(unsafe { wgpuGetProcAddress(nul_terminated) }),
            expected
        );

        // Explicit-length form: only the first `length` bytes count.
        let explicit = native::WGPUStringView {
            data: c"wgpuCreateInstanceX".as_ptr(),
            length: "wgpuCreateInstance".len(),
        };
        assert_eq!(
            address_of(unsafe { wgpuGetProcAddress(explicit) }),
            expected
        );

        // Empty string with non-null data.
        let empty = native::WGPUStringView {
            data: c"wgpuCreateInstance".as_ptr(),
            length: 0,
        };
        assert!(unsafe { wgpuGetProcAddress(empty) }.is_none());

        // Null data with `length == 0` and with `WGPU_STRLEN` (the null string).
        for length in [0, WGPU_STRLEN] {
            let null = native::WGPUStringView {
                data: std::ptr::null(),
                length,
            };
            assert!(unsafe { wgpuGetProcAddress(null) }.is_none());
        }

        // Non-UTF-8 bytes never name an export.
        let invalid = [b'w', b'g', b'p', b'u', 0xFF, 0xFE];
        assert!(unsafe { wgpuGetProcAddress(string_view(&invalid)) }.is_none());
    }
}
