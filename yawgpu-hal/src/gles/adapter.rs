use std::sync::Arc;

use glow::HasContext;
use khronos_egl as egl;

use super::device::{GlesDevice, GlesDeviceCaps, GlesSampleMaskIFn, GlesTextureViewFn};
use super::egl::{EglConfig, EglContext, EglSurface};
use super::format::GlesColorRenderCaps;
use super::instance::{EglInstanceState, GlesInstanceInner};
use super::BACKEND;
use crate::{HalError, HalLimits};

/// Stores GLES adapter data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesAdapter {
    inner: GlesAdapterInner,
}

#[derive(Clone)]
enum GlesAdapterInner {
    Egl {
        instance: Arc<GlesInstanceInner>,
        config: EglConfig,
        caps: GlesAdapterCaps,
    },
    #[cfg(windows)]
    Wgl {
        instance: Arc<GlesInstanceInner>,
        caps: GlesAdapterCaps,
    },
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GlesAdapterCaps {
    limits: HalLimits,
    color_render_caps: GlesColorRenderCaps,
    supports_float32_filterable: bool,
}

// SAFETY: The adapter is an immutable handle to an EGL config plus the shared
// instance. Context creation uses EGL calls and returns errors on failure; no
// Rust-managed mutable state is shared through this type.
unsafe impl Send for GlesAdapter {}
// SAFETY: See the `Send` impl; all fields are immutable after construction.
unsafe impl Sync for GlesAdapter {}

impl std::fmt::Debug for GlesAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesAdapter").finish()
    }
}

impl GlesAdapter {
    pub(super) fn new_egl(
        instance: Arc<GlesInstanceInner>,
        config: EglConfig,
    ) -> Result<Self, HalError> {
        let GlesInstanceInner::Egl(egl_state) = instance.as_ref() else {
            return Err(HalError::BackendUnavailable { backend: BACKEND });
        };
        let caps = query_egl_adapter_caps(egl_state, config)?;
        Ok(Self {
            inner: GlesAdapterInner::Egl {
                instance,
                config,
                caps,
            },
        })
    }

    #[cfg(windows)]
    pub(super) fn new_wgl(instance: Arc<GlesInstanceInner>) -> Result<Self, HalError> {
        let GlesInstanceInner::Wgl(wgl_state) = instance.as_ref() else {
            return Err(HalError::BackendUnavailable { backend: BACKEND });
        };
        let caps = super::wgl::query_adapter_caps(Arc::clone(&instance), wgl_state)?;
        Ok(Self {
            inner: GlesAdapterInner::Wgl { instance, caps },
        })
    }

    /// Returns the adapter name.
    #[must_use]
    pub fn name(&self) -> &str {
        match &self.inner {
            GlesAdapterInner::Egl { .. } => "yawgpu GLES Adapter (EGL)",
            #[cfg(windows)]
            GlesAdapterInner::Wgl { .. } => "yawgpu GLES Adapter (WGL)",
        }
    }

    /// Returns the backend-reported supported limits.
    #[must_use]
    pub(crate) fn limits(&self) -> HalLimits {
        match &self.inner {
            GlesAdapterInner::Egl { caps, .. } => caps.limits,
            #[cfg(windows)]
            GlesAdapterInner::Wgl { caps, .. } => caps.limits,
        }
    }

    /// Returns true when WebGPU texture format tier 1 is supported.
    #[must_use]
    pub(crate) fn supports_texture_formats_tier1(&self) -> bool {
        false
    }

    /// Returns true when WebGPU texture format tier 2 is supported.
    #[must_use]
    pub(crate) fn supports_texture_formats_tier2(&self) -> bool {
        false
    }

    /// Returns true when `Rg11b10Ufloat` is renderable.
    #[must_use]
    pub(crate) fn supports_rg11b10ufloat_renderable(&self) -> bool {
        match &self.inner {
            GlesAdapterInner::Egl { caps, .. } => caps.color_render_caps.color_buffer_float,
            #[cfg(windows)]
            GlesAdapterInner::Wgl { caps, .. } => caps.color_render_caps.color_buffer_float,
        }
    }

    /// Returns true when BGRA8 unorm storage textures are supported.
    #[must_use]
    pub(crate) fn supports_bgra8unorm_storage(&self) -> bool {
        false
    }

    /// Returns true when 32-bit float textures are filterable.
    #[must_use]
    pub(crate) fn supports_float32_filterable(&self) -> bool {
        match &self.inner {
            GlesAdapterInner::Egl { caps, .. } => caps.supports_float32_filterable,
            #[cfg(windows)]
            GlesAdapterInner::Wgl { caps, .. } => caps.supports_float32_filterable,
        }
    }

    /// Returns true when timestamp queries are supported.
    #[must_use]
    pub(crate) fn supports_timestamp_query(&self) -> bool {
        false
    }

    /// Returns true when Depth32FloatStencil8 textures are supported.
    #[must_use]
    pub(crate) fn supports_depth32float_stencil8(&self) -> bool {
        true
    }

    /// Returns true when BC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_bc(&self) -> bool {
        false
    }

    /// Returns true when 3D BC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_bc_sliced_3d(&self) -> bool {
        false
    }

    /// Returns true when ETC2/EAC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_etc2(&self) -> bool {
        false
    }

    /// Returns true when ASTC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_astc(&self) -> bool {
        false
    }

    /// Returns true when 3D ASTC texture compression is supported.
    #[must_use]
    pub fn supports_texture_compression_astc_sliced_3d(&self) -> bool {
        false
    }

    /// Returns true when texture view component swizzling is supported.
    #[must_use]
    pub fn supports_texture_component_swizzle(&self) -> bool {
        false
    }

    /// Returns true when WGSL `shader-f16` is supported.
    #[must_use]
    pub(crate) fn supports_shader_float16(&self) -> bool {
        false
    }

    /// Returns true when WGSL `subgroups` is supported.
    #[must_use]
    pub(crate) fn supports_subgroups(&self) -> bool {
        false
    }

    /// Returns true when depth clip control is supported.
    #[must_use]
    pub(crate) fn supports_depth_clip_control(&self) -> bool {
        false
    }

    /// Returns true when float32 color target blending is supported.
    #[must_use]
    pub(crate) fn supports_float32_blendable(&self) -> bool {
        false
    }

    /// Returns true when dual-source blending is supported.
    #[must_use]
    pub(crate) fn supports_dual_source_blending(&self) -> bool {
        false
    }

    /// Returns true when WGSL clip distances are supported.
    #[must_use]
    pub(crate) fn supports_clip_distances(&self) -> bool {
        false
    }

    /// Returns true when WGSL primitive index is supported.
    #[must_use]
    pub(crate) fn supports_primitive_index(&self) -> bool {
        false
    }

    /// Returns true when indirect draws support non-zero first instance values.
    #[must_use]
    pub(crate) fn supports_indirect_first_instance(&self) -> bool {
        false
    }

    /// Returns the supported subgroup size range.
    #[must_use]
    pub(crate) fn subgroup_size_range(&self) -> Option<(u32, u32)> {
        None
    }

    /// Creates a device (and its default queue) on this adapter.
    pub fn create_device(&self) -> Result<GlesDevice, HalError> {
        match &self.inner {
            GlesAdapterInner::Egl {
                instance, config, ..
            } => {
                let GlesInstanceInner::Egl(egl_state) = instance.as_ref() else {
                    return Err(HalError::DeviceCreationFailed { backend: BACKEND });
                };
                create_egl_device(instance, egl_state, *config)
            }
            #[cfg(windows)]
            GlesAdapterInner::Wgl { instance } => {
                let GlesInstanceInner::Wgl(wgl_state) = instance.as_ref() else {
                    return Err(HalError::DeviceCreationFailed { backend: BACKEND });
                };
                let device_state =
                    super::wgl::WglDeviceState::create(Arc::clone(instance), wgl_state)?;
                Ok(GlesDevice::from_wgl(device_state))
            }
        }
    }
}

fn create_egl_device(
    instance: &Arc<GlesInstanceInner>,
    egl_state: &EglInstanceState,
    config: EglConfig,
) -> Result<GlesDevice, HalError> {
    // One-shot EGL display introspection so failures surface what ANGLE /
    // the host EGL stack actually reports. Cheap; called once per device.
    if let Ok(version) = egl_state
        .egl
        .query_string(Some(egl_state.display), egl::VERSION)
    {
        eprintln!("yawgpu-gles: EGL_VERSION={:?}", version.to_string_lossy());
    }
    if let Ok(vendor) = egl_state
        .egl
        .query_string(Some(egl_state.display), egl::VENDOR)
    {
        eprintln!("yawgpu-gles: EGL_VENDOR={:?}", vendor.to_string_lossy());
    }

    // Try ES 3.1 first (CONTEXT_MAJOR_VERSION=3 + CONTEXT_MINOR_VERSION=1).
    // If ANGLE rejects with BadMatch (some configs don't accept MINOR),
    // fall back to ES 3 (CLIENT_VERSION=3, no MINOR). The downstream
    // GL_VERSION check still enforces the >= 3.1 floor.
    let attribs_es31 = [
        egl::CONTEXT_MAJOR_VERSION,
        3,
        egl::CONTEXT_MINOR_VERSION,
        1,
        egl::NONE,
    ];
    let attribs_es3 = [egl::CONTEXT_CLIENT_VERSION, 3, egl::NONE];
    let context = match egl_state
        .egl
        .create_context(egl_state.display, config, None, &attribs_es31)
    {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!("yawgpu-gles: eglCreateContext(ES 3.1) failed: {err:?}; retrying with ES 3");
            egl_state
                .egl
                .create_context(egl_state.display, config, None, &attribs_es3)
                .map_err(|err2| {
                    eprintln!("yawgpu-gles: eglCreateContext(ES 3) failed: {err2:?}");
                    HalError::DeviceCreationFailed { backend: BACKEND }
                })?
        }
    };

    let pbuffer_attribs = [egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE];
    let surface =
        match egl_state
            .egl
            .create_pbuffer_surface(egl_state.display, config, &pbuffer_attribs)
        {
            Ok(surface) => surface,
            Err(err) => {
                eprintln!("yawgpu-gles: eglCreatePbufferSurface failed: {err:?}");
                destroy_context(egl_state, context);
                return Err(HalError::DeviceCreationFailed { backend: BACKEND });
            }
        };

    if let Err(err) = egl_state.egl.make_current(
        egl_state.display,
        Some(surface),
        Some(surface),
        Some(context),
    ) {
        eprintln!("yawgpu-gles: eglMakeCurrent(pbuffer) failed: {err:?}");
        destroy_surface(egl_state, surface);
        destroy_context(egl_state, context);
        return Err(HalError::DeviceCreationFailed { backend: BACKEND });
    }

    let gl = unsafe {
        glow::Context::from_loader_function(|name| {
            egl_state
                .egl
                .get_proc_address(name)
                .map(|proc| proc as *const _)
                .unwrap_or(std::ptr::null())
        })
    };
    let version = unsafe { gl.get_parameter_string(glow::VERSION) };
    let Some((major, minor)) = parse_gles_version(&version) else {
        eprintln!("yawgpu-gles: unable to parse GL_VERSION={version:?}");
        destroy_surface(egl_state, surface);
        destroy_context(egl_state, context);
        return Err(HalError::DeviceCreationFailed { backend: BACKEND });
    };
    if (major, minor) < (3, 1) {
        eprintln!(
            "yawgpu-gles: GLES {major}.{minor} below the required 3.1 (GL_VERSION={version:?})"
        );
        destroy_surface(egl_state, surface);
        destroy_context(egl_state, context);
        return Err(HalError::DeviceCreationFailed { backend: BACKEND });
    }

    let extensions = gl.supported_extensions();
    let texture_view = load_egl_proc::<GlesTextureViewFn>(egl_state, "glTextureView");
    let caps = GlesDeviceCaps {
        supports_base_vertex: detect_base_vertex_support((major, minor), extensions),
        color_render_caps: detect_color_render_caps(extensions),
        supports_vertex_array_bgra: detect_vertex_array_bgra_support(extensions),
        max_samples: unsafe { gl.get_parameter_i32(glow::MAX_SAMPLES) },
        sample_mask_i: load_egl_proc::<GlesSampleMaskIFn>(egl_state, "glSampleMaski"),
        supports_texture_view: detect_texture_view_support((major, minor), extensions)
            && texture_view.is_some(),
        supports_cube_map_array: detect_cube_map_array_support((major, minor), extensions),
        texture_view,
    };

    Ok(GlesDevice::from_egl(
        Arc::clone(instance),
        context,
        surface,
        gl,
        caps,
    ))
}

fn query_egl_adapter_caps(
    egl_state: &EglInstanceState,
    config: EglConfig,
) -> Result<GlesAdapterCaps, HalError> {
    let attribs_es31 = [
        egl::CONTEXT_MAJOR_VERSION,
        3,
        egl::CONTEXT_MINOR_VERSION,
        1,
        egl::NONE,
    ];
    let attribs_es3 = [egl::CONTEXT_CLIENT_VERSION, 3, egl::NONE];
    let context = match egl_state
        .egl
        .create_context(egl_state.display, config, None, &attribs_es31)
    {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!(
                "yawgpu-gles: eglCreateContext(limit probe ES 3.1) failed: {err:?}; retrying with ES 3"
            );
            egl_state
                .egl
                .create_context(egl_state.display, config, None, &attribs_es3)
                .map_err(|err2| {
                    eprintln!("yawgpu-gles: eglCreateContext(limit probe ES 3) failed: {err2:?}");
                    HalError::BackendUnavailable { backend: BACKEND }
                })?
        }
    };

    let pbuffer_attribs = [egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE];
    let surface =
        match egl_state
            .egl
            .create_pbuffer_surface(egl_state.display, config, &pbuffer_attribs)
        {
            Ok(surface) => surface,
            Err(err) => {
                eprintln!("yawgpu-gles: eglCreatePbufferSurface(limit probe) failed: {err:?}");
                destroy_context(egl_state, context);
                return Err(HalError::BackendUnavailable { backend: BACKEND });
            }
        };

    if let Err(err) = egl_state.egl.make_current(
        egl_state.display,
        Some(surface),
        Some(surface),
        Some(context),
    ) {
        eprintln!("yawgpu-gles: eglMakeCurrent(limit probe) failed: {err:?}");
        destroy_surface(egl_state, surface);
        destroy_context(egl_state, context);
        return Err(HalError::BackendUnavailable { backend: BACKEND });
    }

    let gl = unsafe {
        glow::Context::from_loader_function(|name| {
            egl_state
                .egl
                .get_proc_address(name)
                .map(|proc| proc as *const _)
                .unwrap_or(std::ptr::null())
        })
    };
    let version = unsafe { gl.get_parameter_string(glow::VERSION) };
    let Some((major, minor)) = parse_gles_version(&version) else {
        eprintln!("yawgpu-gles: unable to parse GL_VERSION during limit probe: {version:?}");
        let _ = egl_state
            .egl
            .make_current(egl_state.display, None, None, None);
        destroy_surface(egl_state, surface);
        destroy_context(egl_state, context);
        return Err(HalError::BackendUnavailable { backend: BACKEND });
    };
    if (major, minor) < (3, 1) {
        eprintln!(
            "yawgpu-gles: limit probe found GLES {major}.{minor} below the required 3.1 (GL_VERSION={version:?})"
        );
        let _ = egl_state
            .egl
            .make_current(egl_state.display, None, None, None);
        destroy_surface(egl_state, surface);
        destroy_context(egl_state, context);
        return Err(HalError::BackendUnavailable { backend: BACKEND });
    }
    let extensions = gl.supported_extensions();
    let caps = query_gles_adapter_caps(&gl, extensions);
    let _ = egl_state
        .egl
        .make_current(egl_state.display, None, None, None);
    destroy_surface(egl_state, surface);
    destroy_context(egl_state, context);
    Ok(caps)
}

pub(super) fn query_gles_adapter_caps(
    gl: &glow::Context,
    extensions: &std::collections::HashSet<String>,
) -> GlesAdapterCaps {
    GlesAdapterCaps {
        limits: query_gles_limits(gl),
        color_render_caps: detect_color_render_caps(extensions),
        supports_float32_filterable: detect_float32_filterable_support(extensions),
    }
}

pub(super) fn detect_texture_view_support(
    version: (u32, u32),
    extensions: &std::collections::HashSet<String>,
) -> bool {
    version >= (3, 2)
        || extensions.contains("GL_OES_texture_view")
        || extensions.contains("GL_EXT_texture_view")
        || extensions.contains("texture_view")
}

pub(super) fn detect_cube_map_array_support(
    version: (u32, u32),
    extensions: &std::collections::HashSet<String>,
) -> bool {
    version >= (3, 2)
        || extensions.contains("GL_OES_texture_cube_map_array")
        || extensions.contains("GL_EXT_texture_cube_map_array")
        || extensions.contains("texture_cube_map_array")
        || extensions.contains("cube_map_array")
}

pub(super) fn query_gles_limits(gl: &glow::Context) -> HalLimits {
    let default = HalLimits::DEFAULT;
    let texture_units = min3(
        query_u32(
            gl,
            glow::MAX_VERTEX_TEXTURE_IMAGE_UNITS,
            default.max_sampled_textures_per_shader_stage,
        ),
        query_u32(
            gl,
            glow::MAX_TEXTURE_IMAGE_UNITS,
            default.max_sampled_textures_per_shader_stage,
        ),
        query_u32(
            gl,
            glow::MAX_COMPUTE_TEXTURE_IMAGE_UNITS,
            default.max_sampled_textures_per_shader_stage,
        ),
    );
    let storage_buffers_per_stage = min3(
        query_u32(
            gl,
            glow::MAX_VERTEX_SHADER_STORAGE_BLOCKS,
            default.max_storage_buffers_per_shader_stage,
        ),
        query_u32(
            gl,
            glow::MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,
            default.max_storage_buffers_per_shader_stage,
        ),
        query_u32(
            gl,
            glow::MAX_COMPUTE_SHADER_STORAGE_BLOCKS,
            default.max_storage_buffers_per_shader_stage,
        ),
    );
    let storage_textures_per_stage = min3(
        query_u32(
            gl,
            glow::MAX_VERTEX_IMAGE_UNIFORMS,
            default.max_storage_textures_per_shader_stage,
        ),
        query_u32(
            gl,
            glow::MAX_FRAGMENT_IMAGE_UNIFORMS,
            default.max_storage_textures_per_shader_stage,
        ),
        query_u32(
            gl,
            glow::MAX_COMPUTE_IMAGE_UNIFORMS,
            default.max_storage_textures_per_shader_stage,
        ),
    );
    let uniform_buffers_per_stage = min3(
        query_u32(
            gl,
            glow::MAX_VERTEX_UNIFORM_BLOCKS,
            default.max_uniform_buffers_per_shader_stage,
        ),
        query_u32(
            gl,
            glow::MAX_FRAGMENT_UNIFORM_BLOCKS,
            default.max_uniform_buffers_per_shader_stage,
        ),
        query_u32(
            gl,
            glow::MAX_COMPUTE_UNIFORM_BLOCKS,
            default.max_uniform_buffers_per_shader_stage,
        ),
    );
    let max_color_attachments = query_u32(
        gl,
        glow::MAX_COLOR_ATTACHMENTS,
        default.max_color_attachments,
    )
    .min(query_u32(
        gl,
        glow::MAX_DRAW_BUFFERS,
        default.max_color_attachments,
    ));
    let max_compute_workgroups_per_dimension = min3(
        query_indexed_u32(
            gl,
            glow::MAX_COMPUTE_WORK_GROUP_COUNT,
            0,
            default.max_compute_workgroups_per_dimension,
        ),
        query_indexed_u32(
            gl,
            glow::MAX_COMPUTE_WORK_GROUP_COUNT,
            1,
            default.max_compute_workgroups_per_dimension,
        ),
        query_indexed_u32(
            gl,
            glow::MAX_COMPUTE_WORK_GROUP_COUNT,
            2,
            default.max_compute_workgroups_per_dimension,
        ),
    );

    // Tint emits `layout(binding = N)` using WGSL binding numbers directly
    // into GL's per-resource binding-point spaces. A WGSL binding is safe
    // only when every resource class that may use that number has a GL
    // binding point for it.
    let max_bindings_per_bind_group = [
        query_u32(
            gl,
            glow::MAX_UNIFORM_BUFFER_BINDINGS,
            default.max_bindings_per_bind_group,
        ),
        query_u32(
            gl,
            glow::MAX_SHADER_STORAGE_BUFFER_BINDINGS,
            default.max_bindings_per_bind_group,
        ),
        query_u32(
            gl,
            glow::MAX_COMBINED_TEXTURE_IMAGE_UNITS,
            default.max_bindings_per_bind_group,
        ),
        query_u32(
            gl,
            glow::MAX_IMAGE_UNITS,
            default.max_bindings_per_bind_group,
        ),
    ]
    .into_iter()
    .min()
    .unwrap_or(default.max_bindings_per_bind_group)
    .max(1);

    HalLimits {
        max_texture_dimension_1d: query_u32(
            gl,
            glow::MAX_TEXTURE_SIZE,
            default.max_texture_dimension_1d,
        ),
        max_texture_dimension_2d: query_u32(
            gl,
            glow::MAX_TEXTURE_SIZE,
            default.max_texture_dimension_2d,
        ),
        max_texture_dimension_3d: query_u32(
            gl,
            glow::MAX_3D_TEXTURE_SIZE,
            default.max_texture_dimension_3d,
        ),
        max_texture_array_layers: query_u32(
            gl,
            glow::MAX_ARRAY_TEXTURE_LAYERS,
            default.max_texture_array_layers,
        ),
        max_bind_groups: 4,
        max_bindings_per_bind_group,
        max_sampled_textures_per_shader_stage: texture_units,
        max_samplers_per_shader_stage: texture_units,
        max_storage_buffers_per_shader_stage: storage_buffers_per_stage,
        max_storage_textures_per_shader_stage: storage_textures_per_stage,
        max_storage_buffers_in_vertex_stage: query_u32(
            gl,
            glow::MAX_VERTEX_SHADER_STORAGE_BLOCKS,
            default.max_storage_buffers_in_vertex_stage,
        ),
        max_storage_buffers_in_fragment_stage: query_u32(
            gl,
            glow::MAX_FRAGMENT_SHADER_STORAGE_BLOCKS,
            default.max_storage_buffers_in_fragment_stage,
        ),
        max_storage_textures_in_vertex_stage: query_u32(
            gl,
            glow::MAX_VERTEX_IMAGE_UNIFORMS,
            default.max_storage_textures_in_vertex_stage,
        ),
        max_storage_textures_in_fragment_stage: query_u32(
            gl,
            glow::MAX_FRAGMENT_IMAGE_UNIFORMS,
            default.max_storage_textures_in_fragment_stage,
        ),
        max_uniform_buffers_per_shader_stage: uniform_buffers_per_stage,
        max_uniform_buffer_binding_size: u64::from(query_u32(
            gl,
            glow::MAX_UNIFORM_BLOCK_SIZE,
            default.max_uniform_buffer_binding_size as u32,
        )),
        max_storage_buffer_binding_size: u64::from(query_u32(
            gl,
            glow::MAX_SHADER_STORAGE_BLOCK_SIZE,
            default.max_storage_buffer_binding_size as u32,
        )),
        min_uniform_buffer_offset_alignment: query_u32(
            gl,
            glow::UNIFORM_BUFFER_OFFSET_ALIGNMENT,
            default.min_uniform_buffer_offset_alignment,
        ),
        min_storage_buffer_offset_alignment: query_u32(
            gl,
            glow::SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT,
            default.min_storage_buffer_offset_alignment,
        ),
        max_vertex_buffers: query_u32(
            gl,
            glow::MAX_VERTEX_ATTRIB_BINDINGS,
            default.max_vertex_buffers,
        )
        .min(default.max_vertex_buffers),
        max_vertex_attributes: query_u32(
            gl,
            glow::MAX_VERTEX_ATTRIBS,
            default.max_vertex_attributes,
        ),
        max_vertex_buffer_array_stride: query_u32(
            gl,
            glow::MAX_VERTEX_ATTRIB_STRIDE,
            default.max_vertex_buffer_array_stride,
        ),
        max_inter_stage_shader_variables: query_u32(
            gl,
            glow::MAX_VARYING_VECTORS,
            default.max_inter_stage_shader_variables,
        ),
        max_color_attachments,
        max_compute_workgroup_storage_size: query_u32(
            gl,
            glow::MAX_COMPUTE_SHARED_MEMORY_SIZE,
            default.max_compute_workgroup_storage_size,
        ),
        max_compute_invocations_per_workgroup: query_u32(
            gl,
            glow::MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
            default.max_compute_invocations_per_workgroup,
        ),
        max_compute_workgroup_size_x: query_indexed_u32(
            gl,
            glow::MAX_COMPUTE_WORK_GROUP_SIZE,
            0,
            default.max_compute_workgroup_size_x,
        ),
        max_compute_workgroup_size_y: query_indexed_u32(
            gl,
            glow::MAX_COMPUTE_WORK_GROUP_SIZE,
            1,
            default.max_compute_workgroup_size_y,
        ),
        max_compute_workgroup_size_z: query_indexed_u32(
            gl,
            glow::MAX_COMPUTE_WORK_GROUP_SIZE,
            2,
            default.max_compute_workgroup_size_z,
        ),
        max_compute_workgroups_per_dimension,
        max_immediate_size: 0,
        ..default
    }
}

fn query_u32(gl: &glow::Context, parameter: u32, fallback: u32) -> u32 {
    let value = unsafe { gl.get_parameter_i32(parameter) };
    u32::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or(fallback.max(1))
}

fn query_indexed_u32(gl: &glow::Context, parameter: u32, index: u32, fallback: u32) -> u32 {
    let value = unsafe { gl.get_parameter_indexed_i32(parameter, index) };
    u32::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or(fallback.max(1))
}

fn min3(a: u32, b: u32, c: u32) -> u32 {
    a.min(b).min(c).max(1)
}

fn load_egl_proc<T>(instance: &EglInstanceState, name: &str) -> Option<T> {
    let proc = instance.egl.get_proc_address(name)?;
    Some(unsafe { std::mem::transmute_copy(&proc) })
}

fn destroy_context(instance: &EglInstanceState, context: EglContext) {
    let _ = instance.egl.destroy_context(instance.display, context);
}

fn destroy_surface(instance: &EglInstanceState, surface: EglSurface) {
    let _ = instance.egl.destroy_surface(instance.display, surface);
}

pub(super) fn parse_gles_version(version: &str) -> Option<(u32, u32)> {
    let rest = version.strip_prefix("OpenGL ES ")?;
    let mut parts = rest.split_whitespace().next()?.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// Returns whether the context supports the base-vertex indexed-draw entry
/// points (`glDrawElementsBaseVertex` and friends): core in GLES 3.2, and
/// available on GLES 3.1 through `GL_OES_draw_elements_base_vertex` /
/// `GL_EXT_draw_elements_base_vertex` (T-G11). Pure function of the parsed
/// GL_VERSION tuple and the context's extension set, evaluated once at
/// device creation.
pub(super) fn detect_base_vertex_support(
    version: (u32, u32),
    extensions: &std::collections::HashSet<String>,
) -> bool {
    version >= (3, 2)
        || extensions.contains("GL_OES_draw_elements_base_vertex")
        || extensions.contains("GL_EXT_draw_elements_base_vertex")
}

/// Detects the extension-gated float color-renderability caps (T-G12):
/// `GL_EXT_color_buffer_float` makes the 16- and 32-bit float formats plus
/// `R11F_G11F_B10F` color-renderable, and `GL_EXT_color_buffer_half_float`
/// covers only the 16-bit float formats. Neither extension is core in any
/// GLES version (3.2 included), so this is a pure function of the context's
/// extension set, evaluated once at device creation.
pub(super) fn detect_color_render_caps(
    extensions: &std::collections::HashSet<String>,
) -> GlesColorRenderCaps {
    GlesColorRenderCaps {
        color_buffer_float: extensions.contains("GL_EXT_color_buffer_float"),
        color_buffer_half_float: extensions.contains("GL_EXT_color_buffer_half_float"),
    }
}

/// Detects BGRA-order vertex attribute fetch support. GLES exposes this via
/// `GL_EXT_vertex_array_bgra`; desktop GL contexts may expose the ARB spelling.
pub(super) fn detect_vertex_array_bgra_support(
    extensions: &std::collections::HashSet<String>,
) -> bool {
    extensions.contains("GL_EXT_vertex_array_bgra")
        || extensions.contains("GL_ARB_vertex_array_bgra")
}

/// Detects whether 32-bit float textures are filterable.
pub(super) fn detect_float32_filterable_support(
    extensions: &std::collections::HashSet<String>,
) -> bool {
    extensions.contains("GL_OES_texture_float_linear")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gles_version_accepts_es_versions() {
        assert_eq!(parse_gles_version("OpenGL ES 3.1"), Some((3, 1)));
        assert_eq!(
            parse_gles_version("OpenGL ES 3.2 ANGLE (Vulkan 1.3)"),
            Some((3, 2))
        );
        assert_eq!(parse_gles_version("OpenGL ES 3.0"), Some((3, 0)));
    }

    #[test]
    fn parse_gles_version_rejects_non_es_versions() {
        assert_eq!(parse_gles_version("OpenGL ES-CM 1.1"), None);
        assert_eq!(parse_gles_version(""), None);
        assert_eq!(parse_gles_version("OpenGL 4.5"), None);
    }

    #[test]
    fn detect_base_vertex_support_requires_gles_3_2_or_extension() {
        let empty = std::collections::HashSet::new();
        let oes: std::collections::HashSet<String> =
            ["GL_OES_draw_elements_base_vertex".to_owned()].into();
        let ext: std::collections::HashSet<String> =
            ["GL_EXT_draw_elements_base_vertex".to_owned()].into();
        let unrelated: std::collections::HashSet<String> = ["GL_EXT_copy_image".to_owned()].into();
        // Core in GLES 3.2 and later, regardless of extensions.
        assert!(detect_base_vertex_support((3, 2), &empty));
        assert!(detect_base_vertex_support((4, 0), &empty));
        // GLES 3.1 needs the OES or EXT extension.
        assert!(!detect_base_vertex_support((3, 1), &empty));
        assert!(!detect_base_vertex_support((3, 1), &unrelated));
        assert!(detect_base_vertex_support((3, 1), &oes));
        assert!(detect_base_vertex_support((3, 1), &ext));
    }

    #[test]
    fn detect_color_render_caps_checks_each_extension_independently() {
        let empty = std::collections::HashSet::new();
        let unrelated: std::collections::HashSet<String> = ["GL_EXT_copy_image".to_owned()].into();
        let float_only: std::collections::HashSet<String> =
            ["GL_EXT_color_buffer_float".to_owned()].into();
        let half_float_only: std::collections::HashSet<String> =
            ["GL_EXT_color_buffer_half_float".to_owned()].into();
        let both: std::collections::HashSet<String> = [
            "GL_EXT_color_buffer_float".to_owned(),
            "GL_EXT_color_buffer_half_float".to_owned(),
        ]
        .into();

        assert_eq!(
            detect_color_render_caps(&empty),
            GlesColorRenderCaps::default()
        );
        assert_eq!(
            detect_color_render_caps(&unrelated),
            GlesColorRenderCaps::default()
        );
        assert_eq!(
            detect_color_render_caps(&float_only),
            GlesColorRenderCaps {
                color_buffer_float: true,
                color_buffer_half_float: false,
            }
        );
        assert_eq!(
            detect_color_render_caps(&half_float_only),
            GlesColorRenderCaps {
                color_buffer_float: false,
                color_buffer_half_float: true,
            }
        );
        assert_eq!(
            detect_color_render_caps(&both),
            GlesColorRenderCaps {
                color_buffer_float: true,
                color_buffer_half_float: true,
            }
        );
    }

    #[test]
    fn detect_float32_filterable_support_requires_oes_texture_float_linear() {
        let empty = std::collections::HashSet::new();
        let unrelated: std::collections::HashSet<String> = ["GL_EXT_copy_image".to_owned()].into();
        let float_linear: std::collections::HashSet<String> =
            ["GL_OES_texture_float_linear".to_owned()].into();

        assert!(!detect_float32_filterable_support(&empty));
        assert!(!detect_float32_filterable_support(&unrelated));
        assert!(detect_float32_filterable_support(&float_linear));
    }

    #[test]
    fn detect_vertex_array_bgra_support_checks_ext_and_arb_names() {
        let empty = std::collections::HashSet::new();
        let unrelated: std::collections::HashSet<String> = ["GL_EXT_copy_image".to_owned()].into();
        let ext: std::collections::HashSet<String> = ["GL_EXT_vertex_array_bgra".to_owned()].into();
        let arb: std::collections::HashSet<String> = ["GL_ARB_vertex_array_bgra".to_owned()].into();

        assert!(!detect_vertex_array_bgra_support(&empty));
        assert!(!detect_vertex_array_bgra_support(&unrelated));
        assert!(detect_vertex_array_bgra_support(&ext));
        assert!(detect_vertex_array_bgra_support(&arb));
    }

    #[test]
    fn egl_adapter_limits_match_live_gl_context() {
        let instance = match super::super::instance::GlesInstance::new_with_choice(Some(
            super::super::instance::BackendChoice::Egl,
        )) {
            Ok(instance) => instance,
            Err(error) => {
                eprintln!("skipping GLES adapter limit test; EGL unavailable: {error:?}");
                return;
            }
        };
        let Some(adapter) = instance.enumerate_adapters().into_iter().next() else {
            eprintln!("skipping GLES adapter limit test; no EGL adapter");
            return;
        };
        let device = match adapter.create_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping GLES adapter limit test; device unavailable: {error:?}");
                return;
            }
        };

        let limits = adapter.limits();
        assert_positive_limits_except_immediates(limits);
        assert!(
            limits.max_bindings_per_bind_group < HalLimits::DEFAULT.max_bindings_per_bind_group,
            "GLES binding limit should be queried from GL, not the WebGPU default"
        );

        let queried = device
            .inner_clone()
            .with_current_context(|gl| {
                let texture_size = query_u32(
                    gl,
                    glow::MAX_TEXTURE_SIZE,
                    HalLimits::DEFAULT.max_texture_dimension_2d,
                );
                let binding_points = [
                    query_u32(
                        gl,
                        glow::MAX_UNIFORM_BUFFER_BINDINGS,
                        HalLimits::DEFAULT.max_bindings_per_bind_group,
                    ),
                    query_u32(
                        gl,
                        glow::MAX_SHADER_STORAGE_BUFFER_BINDINGS,
                        HalLimits::DEFAULT.max_bindings_per_bind_group,
                    ),
                    query_u32(
                        gl,
                        glow::MAX_COMBINED_TEXTURE_IMAGE_UNITS,
                        HalLimits::DEFAULT.max_bindings_per_bind_group,
                    ),
                    query_u32(
                        gl,
                        glow::MAX_IMAGE_UNITS,
                        HalLimits::DEFAULT.max_bindings_per_bind_group,
                    ),
                ];
                let min_binding_points = binding_points
                    .into_iter()
                    .min()
                    .unwrap_or(HalLimits::DEFAULT.max_bindings_per_bind_group)
                    .max(1);
                (texture_size, min_binding_points)
            })
            .expect("live GLES context should remain usable");

        assert_eq!(limits.max_texture_dimension_2d, queried.0);
        assert_eq!(limits.max_bindings_per_bind_group, queried.1);
    }

    fn assert_positive_limits_except_immediates(limits: HalLimits) {
        let values = [
            limits.max_texture_dimension_1d,
            limits.max_texture_dimension_2d,
            limits.max_texture_dimension_3d,
            limits.max_texture_array_layers,
            limits.max_bind_groups,
            limits.max_bind_groups_plus_vertex_buffers,
            limits.max_bindings_per_bind_group,
            limits.max_dynamic_uniform_buffers_per_pipeline_layout,
            limits.max_dynamic_storage_buffers_per_pipeline_layout,
            limits.max_sampled_textures_per_shader_stage,
            limits.max_samplers_per_shader_stage,
            limits.max_storage_buffers_per_shader_stage,
            limits.max_storage_textures_per_shader_stage,
            limits.max_storage_buffers_in_vertex_stage,
            limits.max_storage_buffers_in_fragment_stage,
            limits.max_storage_textures_in_vertex_stage,
            limits.max_storage_textures_in_fragment_stage,
            limits.max_uniform_buffers_per_shader_stage,
            limits.min_uniform_buffer_offset_alignment,
            limits.min_storage_buffer_offset_alignment,
            limits.max_vertex_buffers,
            limits.max_vertex_attributes,
            limits.max_vertex_buffer_array_stride,
            limits.max_inter_stage_shader_variables,
            limits.max_color_attachments,
            limits.max_color_attachment_bytes_per_sample,
            limits.max_compute_workgroup_storage_size,
            limits.max_compute_invocations_per_workgroup,
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
            limits.max_compute_workgroups_per_dimension,
        ];
        for value in values {
            assert!(value > 0);
        }
        assert!(limits.max_uniform_buffer_binding_size > 0);
        assert!(limits.max_storage_buffer_binding_size > 0);
        assert!(limits.max_buffer_size > 0);
        assert_eq!(limits.max_immediate_size, 0);
    }
}
