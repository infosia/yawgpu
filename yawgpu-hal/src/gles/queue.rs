use std::sync::Arc;

use glow::HasContext;

use super::device::{GlesDeviceInner, GlesTextureViewFn};
use super::format::{
    color_clear_kind, map_primitive_topology, map_vertex_format, storage_image_format,
    GlesClearKind,
};
use super::texture::GlesTextureMeta;
use super::BACKEND;
use crate::format::{format_has_depth_aspect, format_has_stencil_aspect};
use crate::{
    HalBlendFactor, HalBlendOperation, HalBoundSampler, HalBoundTexture, HalBuffer,
    HalBufferBindingKind, HalBufferClear, HalBufferCopy, HalBufferTextureCopy, HalColorTargetState,
    HalCompareFunction, HalComputeDispatch, HalComputePass, HalComputePipeline, HalCopy,
    HalCullMode, HalDepthStencilState, HalDescriptorBinding, HalDescriptorBindingKind, HalDraw,
    HalError, HalFrontFace, HalGlesBindingClass, HalGlesBindingRemap, HalIndexFormat,
    HalRenderLoadOp, HalRenderPass, HalRenderPipeline, HalSampler, HalStencilFaceState,
    HalStencilOperation, HalStorageTextureAccess, HalTexture, HalTextureAspect, HalTextureClear,
    HalTextureCopy, HalTextureFormat, HalTextureMetadataSlot, HalTextureViewDimension,
    HalVertexStepMode,
};

/// Stores GLES queue data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesQueue {
    inner: Arc<GlesDeviceInner>,
}

// SAFETY: Queue submission calls into `GlesDeviceInner::with_current_context`,
// which serializes context binding and GL commands.
unsafe impl Send for GlesQueue {}
// SAFETY: See the `Send` impl; shared submission is synchronized by the device
// inner lock.
unsafe impl Sync for GlesQueue {}

impl std::fmt::Debug for GlesQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesQueue").finish()
    }
}

impl GlesQueue {
    pub(super) fn new(inner: Arc<GlesDeviceInner>) -> Self {
        Self { inner }
    }

    /// Submits an empty command buffer to flush the queue.
    pub fn submit_empty(&self) -> Result<(), HalError> {
        self.inner.with_current_context(|gl| unsafe {
            gl.flush();
        })
    }

    /// Waits until all submitted queue work has completed.
    pub fn wait_idle(&self) -> Result<(), HalError> {
        self.inner.with_current_context(|gl| unsafe {
            gl.finish();
        })
    }

    /// Records and submits the given buffer/texture copy operations.
    pub fn submit_copies(&self, copies: &[HalCopy]) -> Result<(), HalError> {
        if copies.is_empty() {
            return Ok(());
        }

        let supports_base_vertex = self.inner.supports_base_vertex();
        let supports_vertex_array_bgra = self.inner.supports_vertex_array_bgra();
        let sample_mask_i = self.inner.sample_mask_i();
        let supports_texture_view = self.inner.supports_texture_view();
        let supports_cube_map_array = self.inner.supports_cube_map_array();
        let texture_view = self.inner.texture_view();
        let placeholder_sampler = self.inner.placeholder_sampler()?;
        self.inner
            .with_current_context(|gl| -> Result<(), HalError> {
                for copy in copies {
                    match copy {
                        HalCopy::Buffer(copy) => submit_buffer_copy(gl, copy)?,
                        HalCopy::BufferClear(clear) => submit_buffer_clear(gl, clear)?,
                        HalCopy::ClearTexture(clear) => submit_texture_clear(gl, clear)?,
                        HalCopy::ResolveQuerySet(resolve) => {
                            let HalBuffer::Gles(destination) = &resolve.destination else {
                                return Err(HalError::BufferOperationFailed {
                                    backend: BACKEND,
                                    message: "query resolve destination is not a GLES buffer",
                                });
                            };
                            let byte_count = usize::try_from(u64::from(resolve.query_count) * 8)
                                .map_err(|_| HalError::BufferOperationFailed {
                                    backend: BACKEND,
                                    message: "query resolve byte count is too large",
                                })?;
                            // The device's make-current lock is already held
                            // by this closure; `GlesBuffer::write` would
                            // re-acquire it and self-deadlock (T-G4), so use
                            // the lock-free variant with the current `gl`.
                            destination.write_with_gl(
                                gl,
                                resolve.destination_offset,
                                &vec![0; byte_count],
                            )?;
                        }
                        HalCopy::BufferToTexture(copy) => submit_buffer_to_texture(gl, copy)?,
                        HalCopy::TextureToBuffer(copy) => submit_texture_to_buffer(gl, copy)?,
                        HalCopy::TextureToTexture(copy) => submit_texture_to_texture(gl, copy)?,
                        HalCopy::ComputePass(pass) => {
                            submit_compute_pass(
                                gl,
                                pass,
                                placeholder_sampler,
                                TextureViewCaps {
                                    supports_texture_view,
                                    supports_cube_map_array,
                                    texture_view,
                                },
                            )?;
                        }
                        HalCopy::RenderPass(pass) => {
                            let render_caps = RenderDrawCaps {
                                supports_base_vertex,
                                supports_vertex_array_bgra,
                                sample_mask_i,
                            };
                            submit_render_pass(
                                gl,
                                pass,
                                render_caps,
                                placeholder_sampler,
                                TextureViewCaps {
                                    supports_texture_view,
                                    supports_cube_map_array,
                                    texture_view,
                                },
                            )?;
                        }
                        #[cfg(feature = "tiled")]
                        HalCopy::SubpassRenderPass(_) => {
                            return Err(HalError::QueueSubmissionFailed {
                                backend: BACKEND,
                                message: "GLES subpass render pass submission is unsupported"
                                    .to_string(),
                            });
                        }
                    }
                }
                unsafe {
                    gl.flush();
                }
                Ok(())
            })?
    }
}

fn submit_buffer_copy(gl: &glow::Context, copy: &HalBufferCopy) -> Result<(), HalError> {
    let HalBuffer::Gles(source) = &copy.source else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer copy source is not a GLES buffer",
        });
    };
    let HalBuffer::Gles(destination) = &copy.destination else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer copy destination is not a GLES buffer",
        });
    };

    let source_buffer = source.raw_or_err()?;
    let destination_buffer = destination.raw_or_err()?;
    let source_offset =
        i32::try_from(copy.source_offset).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer copy source offset exceeds GLES limit",
        })?;
    let destination_offset =
        i32::try_from(copy.destination_offset).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer copy destination offset exceeds GLES limit",
        })?;
    let size = i32::try_from(copy.size).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message: "buffer copy size exceeds GLES limit",
    })?;

    unsafe {
        gl.bind_buffer(glow::COPY_READ_BUFFER, Some(source_buffer));
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(destination_buffer));
        gl.copy_buffer_sub_data(
            glow::COPY_READ_BUFFER,
            glow::COPY_WRITE_BUFFER,
            source_offset,
            destination_offset,
            size,
        );
        gl.bind_buffer(glow::COPY_READ_BUFFER, None);
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
    }

    Ok(())
}

fn submit_buffer_clear(gl: &glow::Context, clear: &HalBufferClear) -> Result<(), HalError> {
    let HalBuffer::Gles(buffer) = &clear.buffer else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer clear target is not a GLES buffer",
        });
    };
    let end = clear
        .offset
        .checked_add(clear.size)
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer clear range overflow",
        })?;
    if end > buffer.size() {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer clear range exceeds buffer size",
        });
    }
    if clear.size == 0 {
        return Ok(());
    }

    let raw = buffer.raw_or_err()?;
    let base_offset = i32::try_from(clear.offset).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message: "buffer clear offset exceeds GLES limit",
    })?;
    let size = usize::try_from(clear.size).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message: "buffer clear size exceeds host limit",
    })?;
    const ZERO_CHUNK: usize = 4096;
    let zeros = [0_u8; ZERO_CHUNK];
    unsafe {
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(raw));
        let mut written = 0_usize;
        while written < size {
            let chunk = (size - written).min(ZERO_CHUNK);
            let offset = base_offset
                .checked_add(i32::try_from(written).map_err(|_| {
                    HalError::BufferOperationFailed {
                        backend: BACKEND,
                        message: "buffer clear offset exceeds GLES limit",
                    }
                })?)
                .ok_or(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "buffer clear offset exceeds GLES limit",
                })?;
            gl.buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, offset, &zeros[..chunk]);
            written += chunk;
        }
        gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
    }
    Ok(())
}

fn submit_compute_pass(
    gl: &glow::Context,
    pass: &HalComputePass,
    placeholder_sampler: glow::Sampler,
    texture_view_caps: TextureViewCaps,
) -> Result<(), HalError> {
    let HalComputePipeline::Gles(pipeline) = &pass.pipeline else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "compute pass pipeline is not a GLES pipeline",
        });
    };
    reject_external_texture_bindings(pass.bind_external_textures.len())?;
    let program = pipeline.raw_or_err()?;
    let bindings = pass
        .bind_buffers
        .iter()
        .map(|bound| {
            let HalBuffer::Gles(buffer) = &bound.buffer else {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "compute pass binding is not a GLES buffer",
                });
            };
            let (target, class) = binding_target(pipeline.bindings(), bound.group, bound.binding)?;
            let Some(flat_binding) =
                flat_binding(pipeline.binding_remaps(), bound.group, bound.binding, class)
            else {
                return Ok(None);
            };
            let bound_size = gles_bound_buffer_size(buffer, bound)?;
            let buffer = buffer.raw_or_err()?;
            let offset =
                i32::try_from(bound.offset).map_err(|_| HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "compute buffer binding offset exceeds GLES limit",
                })?;
            let size = i32::try_from(bound_size).map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "compute buffer binding size exceeds GLES limit",
            })?;
            Ok(Some((target, flat_binding, buffer, offset, size)))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    unsafe {
        gl.use_program(Some(program));
        for (target, binding, buffer, offset, size) in bindings {
            gl.bind_buffer_range(target, binding, Some(buffer), offset, size);
        }
        let texture_units = bind_combined_samplers(
            gl,
            pipeline.combined_samplers(),
            &pass.bind_textures,
            &pass.bind_samplers,
            placeholder_sampler,
            texture_view_caps,
        )?;
        let _texture_cleanup = TextureUnitCleanup {
            gl,
            texture_units: texture_units.units,
            texture_views: texture_units.texture_views,
        };
        let _texture_metadata_cleanup = bind_texture_metadata_ubo(
            gl,
            pipeline.texture_metadata_ubo_binding(),
            pipeline.texture_metadata_slots(),
            &pass.bind_textures,
        )?;
        bind_storage_textures(
            gl,
            pipeline.bindings(),
            pipeline.binding_remaps(),
            &pass.bind_textures,
        )?;
        match &pass.dispatch {
            HalComputeDispatch::Direct { workgroups } => {
                // WebGPU: a dispatch with any zero workgroup count does
                // nothing, so skip the API call. Indirect dispatches cannot
                // be pre-checked CPU-side and are left as-is.
                if workgroups.0 != 0 && workgroups.1 != 0 && workgroups.2 != 0 {
                    gl.dispatch_compute(workgroups.0, workgroups.1, workgroups.2);
                }
            }
            HalComputeDispatch::Indirect { buffer } => {
                let HalBuffer::Gles(indirect_buffer) = &buffer.buffer else {
                    return Err(HalError::BufferOperationFailed {
                        backend: BACKEND,
                        message: "compute indirect buffer is not a GLES buffer",
                    });
                };
                let offset =
                    i32::try_from(buffer.offset).map_err(|_| HalError::BufferOperationFailed {
                        backend: BACKEND,
                        message: "compute indirect buffer offset exceeds GLES limit",
                    })?;
                gl.bind_buffer(
                    glow::DISPATCH_INDIRECT_BUFFER,
                    Some(indirect_buffer.raw_or_err()?),
                );
                gl.dispatch_compute_indirect(offset);
                gl.bind_buffer(glow::DISPATCH_INDIRECT_BUFFER, None);
            }
        }
        gl.memory_barrier(glow::ALL_BARRIER_BITS);
        gl.use_program(None);
    }

    Ok(())
}

fn binding_target(
    bindings: &[HalDescriptorBinding],
    group: u32,
    binding: u32,
) -> Result<(u32, HalGlesBindingClass), HalError> {
    let descriptor = bindings
        .iter()
        .find(|descriptor| descriptor.group == group && descriptor.binding == binding)
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer binding is missing from pipeline layout",
        })?;
    match descriptor.kind {
        HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Uniform) => {
            Ok((glow::UNIFORM_BUFFER, HalGlesBindingClass::UniformBuffer))
        }
        HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage) => Ok((
            glow::SHADER_STORAGE_BUFFER,
            HalGlesBindingClass::StorageBuffer,
        )),
        HalDescriptorBindingKind::Texture
        | HalDescriptorBindingKind::StorageTexture { .. }
        | HalDescriptorBindingKind::Sampler
        | HalDescriptorBindingKind::InputAttachment { .. } => {
            Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "texture and sampler descriptors are not valid buffer bindings",
            })
        }
    }
}

fn flat_binding(
    remaps: &[HalGlesBindingRemap],
    group: u32,
    binding: u32,
    class: HalGlesBindingClass,
) -> Option<u32> {
    remaps
        .iter()
        .find(|remap| remap.group == group && remap.binding == binding && remap.class == class)
        .map(|remap| remap.flat_binding)
}

fn reject_external_texture_bindings(count: usize) -> Result<(), HalError> {
    if count != 0 {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES does not support external texture bindings",
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct TextureViewCaps {
    supports_texture_view: bool,
    supports_cube_map_array: bool,
    texture_view: Option<GlesTextureViewFn>,
}

struct BoundCombinedSamplers {
    units: Vec<u32>,
    texture_views: Vec<TransientTextureView>,
}

struct TransientTextureView {
    unit: u32,
    target: u32,
    texture: glow::Texture,
}

fn bind_combined_samplers(
    gl: &glow::Context,
    combined_samplers: &[super::pipeline::GlesResolvedCombinedSampler],
    textures: &[HalBoundTexture],
    samplers: &[HalBoundSampler],
    placeholder_sampler: glow::Sampler,
    texture_view_caps: TextureViewCaps,
) -> Result<BoundCombinedSamplers, HalError> {
    let mut units = Vec::with_capacity(combined_samplers.len());
    let mut texture_views = Vec::new();
    for combined in combined_samplers {
        let texture = textures
            .iter()
            .find(|texture| {
                texture.group == combined.texture_group
                    && texture.binding == combined.texture_binding
                    && texture.storage_access.is_none()
            })
            .ok_or_else(|| HalError::QueueSubmissionFailed {
                backend: BACKEND,
                message: format!(
                    "GLES sampled texture binding (group {}, binding {}) is missing for uniform {}",
                    combined.texture_group, combined.texture_binding, combined.uniform_name
                ),
            })?;
        let HalTexture::Gles(gles_texture) = &texture.texture else {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "sampled texture binding is not a GLES texture",
            });
        };
        let target = texture_view_target(
            texture.dimension,
            gles_texture.meta().target,
            texture_view_caps.supports_cube_map_array,
        )?;
        let raw_texture = gles_texture.raw_or_err()?;
        let base_level = i32_from_u32(
            texture.base_mip_level,
            "texture base mip level exceeds GLES limit",
        )?;
        let max_level = texture
            .base_mip_level
            .checked_add(texture.mip_level_count)
            .and_then(|end| end.checked_sub(1))
            .ok_or(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "texture mip level range is empty or overflows",
            })?;
        let max_level = i32_from_u32(max_level, "texture max mip level exceeds GLES limit")?;
        let unit = combined.unit;
        let (bound_texture, owns_bound_texture) = texture_handle_for_sampled_view(
            gl,
            raw_texture,
            target,
            gles_texture.meta(),
            texture,
            texture_view_caps,
        )?;
        unsafe {
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(target, Some(bound_texture));
            apply_depth_stencil_texture_mode(gl, target, gles_texture.meta(), texture.aspect);
            if target != glow::TEXTURE_2D_MULTISAMPLE {
                // This mutates texture-object state. GLES queue submission is
                // single-context serialized, so restoring per-view mip bounds
                // at each bind is deterministic for this backend.
                if owns_bound_texture {
                    gl.tex_parameter_i32(target, glow::TEXTURE_BASE_LEVEL, 0);
                    gl.tex_parameter_i32(
                        target,
                        glow::TEXTURE_MAX_LEVEL,
                        i32_from_u32(
                            texture.mip_level_count.saturating_sub(1),
                            "texture view mip level count exceeds GLES limit",
                        )?,
                    );
                } else {
                    gl.tex_parameter_i32(target, glow::TEXTURE_BASE_LEVEL, base_level);
                    gl.tex_parameter_i32(target, glow::TEXTURE_MAX_LEVEL, max_level);
                }
            }
        }
        if owns_bound_texture {
            texture_views.push(TransientTextureView {
                unit,
                target,
                texture: bound_texture,
            });
        }
        if combined.uses_placeholder_sampler {
            unsafe {
                gl.bind_sampler(unit, Some(placeholder_sampler));
            }
        } else {
            let sampler = samplers
                .iter()
                .find(|sampler| {
                    sampler.group == combined.sampler_group
                        && sampler.binding == combined.sampler_binding
                })
                .ok_or_else(|| HalError::QueueSubmissionFailed {
                    backend: BACKEND,
                    message: format!(
                        "GLES sampler binding (group {}, binding {}) is missing for uniform {}",
                        combined.sampler_group, combined.sampler_binding, combined.uniform_name
                    ),
                })?;
            let HalSampler::Gles(gles_sampler) = &sampler.sampler else {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "sampler binding is not a GLES sampler",
                });
            };
            unsafe {
                gl.bind_sampler(unit, Some(gles_sampler.raw_or_err()?));
            }
        }
        units.push(unit);
    }
    Ok(BoundCombinedSamplers {
        units,
        texture_views,
    })
}

fn texture_handle_for_sampled_view(
    gl: &glow::Context,
    raw_texture: glow::Texture,
    target: u32,
    meta: &GlesTextureMeta,
    texture: &HalBoundTexture,
    caps: TextureViewCaps,
) -> Result<(glow::Texture, bool), HalError> {
    if !requires_texture_view(meta, texture, target) {
        return Ok((raw_texture, false));
    }
    if !caps.supports_texture_view {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES cannot bind this texture view without glTextureView",
        });
    }
    let Some(texture_view) = caps.texture_view else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES texture-view support was reported but glTextureView is unavailable",
        });
    };
    create_transient_texture_view(gl, raw_texture, target, meta, texture, texture_view)
        .map(|view| (view, true))
}

fn requires_texture_view(
    meta: &GlesTextureMeta,
    texture: &HalBoundTexture,
    view_target: u32,
) -> bool {
    requires_texture_view_for(
        meta,
        view_target,
        texture.format,
        texture.aspect,
        texture.base_array_layer,
        texture.array_layer_count,
    )
}

fn requires_texture_view_for(
    meta: &GlesTextureMeta,
    view_target: u32,
    view_format: HalTextureFormat,
    aspect: HalTextureAspect,
    base_array_layer: u32,
    array_layer_count: u32,
) -> bool {
    view_target != meta.target
        || base_array_layer != 0
        || array_layer_count < meta.depth_or_array_layers
        || (view_format != meta.hal_format && !is_depth_or_stencil_hal_format(meta.hal_format))
        || (matches!(aspect, HalTextureAspect::StencilOnly)
            && is_packed_depth_stencil_format(meta.format.internal))
}

fn is_depth_or_stencil_hal_format(format: HalTextureFormat) -> bool {
    format_has_depth_aspect(format) || format_has_stencil_aspect(format)
}

fn create_transient_texture_view(
    gl: &glow::Context,
    raw_texture: glow::Texture,
    target: u32,
    meta: &GlesTextureMeta,
    texture: &HalBoundTexture,
    texture_view: GlesTextureViewFn,
) -> Result<glow::Texture, HalError> {
    let internal_format = texture_view_internal_format(meta, texture)?;
    let min_level = texture.base_mip_level;
    let num_levels = texture.mip_level_count;
    let min_layer = texture.base_array_layer;
    let num_layers = texture_view_layer_count(texture);
    unsafe {
        let view = gl
            .create_texture()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateTexture failed for GLES texture view",
            })?;
        // Simple and correct first: create a transient GL view for this submit
        // and delete it in `TextureUnitCleanup`. A later cache can move this
        // onto the texture/view object without changing binding semantics.
        texture_view(
            gl_texture_name(view),
            target,
            gl_texture_name(raw_texture),
            internal_format,
            min_level,
            num_levels,
            min_layer,
            num_layers,
        );
        let error = gl.get_error();
        if error != glow::NO_ERROR {
            gl.delete_texture(view);
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glTextureView failed for GLES sampled texture view",
            });
        }
        Ok(view)
    }
}

fn texture_view_internal_format(
    meta: &GlesTextureMeta,
    texture: &HalBoundTexture,
) -> Result<u32, HalError> {
    if is_packed_depth_stencil_format(meta.format.internal) {
        return Ok(meta.format.internal);
    }
    Ok(super::format::map_texture_format(texture.format)?.internal)
}

fn texture_view_layer_count(texture: &HalBoundTexture) -> u32 {
    match texture.dimension {
        HalTextureViewDimension::D1 | HalTextureViewDimension::D2 | HalTextureViewDimension::D3 => {
            1
        }
        HalTextureViewDimension::D2Array
        | HalTextureViewDimension::Cube
        | HalTextureViewDimension::CubeArray => texture.array_layer_count,
    }
}

fn is_packed_depth_stencil_format(internal_format: u32) -> bool {
    matches!(
        internal_format,
        glow::DEPTH24_STENCIL8 | glow::DEPTH32F_STENCIL8
    )
}

fn gl_texture_name(texture: glow::Texture) -> u32 {
    texture.0.get()
}

fn bind_storage_textures(
    gl: &glow::Context,
    descriptors: &[HalDescriptorBinding],
    remaps: &[HalGlesBindingRemap],
    textures: &[HalBoundTexture],
) -> Result<(), HalError> {
    for texture in textures
        .iter()
        .filter(|texture| texture.storage_access.is_some())
    {
        let descriptor = descriptors
            .iter()
            .find(|descriptor| {
                descriptor.group == texture.group && descriptor.binding == texture.binding
            })
            .ok_or(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "storage texture binding is missing from pipeline layout",
            })?;
        if !matches!(
            descriptor.kind,
            HalDescriptorBindingKind::StorageTexture { .. }
        ) {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "storage texture binding is not a storage texture descriptor",
            });
        }
        let Some(flat_binding) = flat_binding(
            remaps,
            texture.group,
            texture.binding,
            HalGlesBindingClass::StorageTexture,
        ) else {
            continue;
        };
        let HalTexture::Gles(gles_texture) = &texture.texture else {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "storage texture binding is not a GLES texture",
            });
        };
        let raw_texture = gles_texture.raw_or_err()?;
        let access = storage_texture_access(texture.storage_access.ok_or(
            HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "storage texture binding is missing access",
            },
        )?);
        let internal_format =
            storage_image_format(texture.format).ok_or(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "GLES image load/store does not support this storage format",
            })?;
        let base_level = i32_from_u32(
            texture.base_mip_level,
            "storage texture base mip level exceeds GLES limit",
        )?;
        if texture.mip_level_count != 1 {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "GLES storage texture views must expose exactly one mip level",
            });
        }
        let (layered, layer) = storage_image_layer_binding(
            texture,
            gles_texture.meta().target,
            gles_texture.meta().depth_or_array_layers,
        )?;
        unsafe {
            gl.bind_image_texture(
                flat_binding,
                raw_texture,
                base_level,
                layered,
                layer,
                access,
                internal_format,
            );
        }
    }
    Ok(())
}

fn storage_texture_access(access: HalStorageTextureAccess) -> u32 {
    match access {
        HalStorageTextureAccess::ReadOnly => glow::READ_ONLY,
        HalStorageTextureAccess::WriteOnly => glow::WRITE_ONLY,
        HalStorageTextureAccess::ReadWrite => glow::READ_WRITE,
    }
}

fn storage_image_layer_binding(
    texture: &HalBoundTexture,
    texture_target: u32,
    full_layer_count: u32,
) -> Result<(bool, i32), HalError> {
    match texture.dimension {
        HalTextureViewDimension::D2 => {
            let layer = if texture_target == glow::TEXTURE_2D {
                if texture.base_array_layer != 0 {
                    return Err(HalError::BufferOperationFailed {
                        backend: BACKEND,
                        message: "GLES cannot bind a non-zero layer of a plain 2D storage texture",
                    });
                }
                0
            } else {
                texture.base_array_layer
            };
            Ok((
                false,
                i32_from_u32(layer, "storage texture layer exceeds GLES limit")?,
            ))
        }
        HalTextureViewDimension::D2Array | HalTextureViewDimension::D3 => {
            if texture.base_array_layer == 0 && texture.array_layer_count == full_layer_count {
                return Ok((true, 0));
            }
            if texture.array_layer_count == 1 {
                return Ok((
                    false,
                    i32_from_u32(
                        texture.base_array_layer,
                        "storage texture layer exceeds GLES limit",
                    )?,
                ));
            }
            Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "GLES storage texture views must bind a whole layered view or one layer",
            })
        }
        HalTextureViewDimension::D1
        | HalTextureViewDimension::Cube
        | HalTextureViewDimension::CubeArray => Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES storage texture view dimension is unsupported",
        }),
    }
}

unsafe fn apply_depth_stencil_texture_mode(
    gl: &glow::Context,
    target: u32,
    meta: &GlesTextureMeta,
    aspect: HalTextureAspect,
) {
    if !matches!(
        meta.format.internal,
        glow::DEPTH24_STENCIL8 | glow::DEPTH32F_STENCIL8
    ) {
        return;
    }
    let mode = match aspect {
        HalTextureAspect::StencilOnly => glow::STENCIL_INDEX,
        _ => glow::DEPTH_COMPONENT,
    };
    gl.tex_parameter_i32(target, glow::DEPTH_STENCIL_TEXTURE_MODE, mode as i32);
}

fn bind_texture_metadata_ubo<'a>(
    gl: &'a glow::Context,
    binding: Option<u32>,
    metadata_slots: &[HalTextureMetadataSlot],
    textures: &[HalBoundTexture],
) -> Result<Option<TextureMetadataUboCleanup<'a>>, HalError> {
    let Some(binding) = binding else {
        return Ok(None);
    };
    if metadata_slots.is_empty() {
        return Ok(None);
    }
    let max_offset = metadata_slots
        .iter()
        .map(|slot| slot.offset)
        .max()
        .ok_or_else(|| HalError::QueueSubmissionFailed {
            backend: BACKEND,
            message: "GLES texture metadata UBO has no slots".to_owned(),
        })?;
    let value_count = usize::try_from(max_offset)
        .ok()
        .and_then(|offset| offset.checked_add(1))
        .ok_or_else(|| HalError::QueueSubmissionFailed {
            backend: BACKEND,
            message: "GLES texture metadata UBO size exceeds platform limits".to_owned(),
        })?;
    let mut values = vec![0; (value_count + 3) & !3];
    for slot in metadata_slots {
        let texture = textures
            .iter()
            .find(|texture| {
                texture.group == slot.texture_group
                    && texture.binding == slot.texture_binding
                    && texture.storage_access.is_none()
            })
            .ok_or_else(|| HalError::QueueSubmissionFailed {
                backend: BACKEND,
                message: format!(
                    "GLES texture binding (group {}, binding {}) is missing for metadata uniform",
                    slot.texture_group, slot.texture_binding
                ),
            })?;
        let HalTexture::Gles(gles_texture) = &texture.texture else {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "texture metadata binding is not a GLES texture",
            });
        };
        let value = if gles_texture.meta().target == glow::TEXTURE_2D_MULTISAMPLE {
            gles_texture.meta().sample_count
        } else {
            texture.mip_level_count
        };
        values[usize::try_from(slot.offset).map_err(|_| HalError::QueueSubmissionFailed {
            backend: BACKEND,
            message: "GLES texture metadata slot offset exceeds platform limits".to_owned(),
        })?] = value;
    }
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<u32>());
    for value in &values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    unsafe {
        let buffer = gl
            .create_buffer()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateBuffer failed for GLES texture metadata uniform",
            })?;
        gl.bind_buffer(glow::UNIFORM_BUFFER, Some(buffer));
        gl.buffer_data_u8_slice(glow::UNIFORM_BUFFER, &bytes, glow::DYNAMIC_DRAW);
        gl.bind_buffer_base(glow::UNIFORM_BUFFER, binding, Some(buffer));
        gl.bind_buffer(glow::UNIFORM_BUFFER, None);
        Ok(Some(TextureMetadataUboCleanup {
            gl,
            binding,
            buffer,
        }))
    }
}

struct TextureMetadataUboCleanup<'a> {
    gl: &'a glow::Context,
    binding: u32,
    buffer: glow::Buffer,
}

impl Drop for TextureMetadataUboCleanup<'_> {
    fn drop(&mut self) {
        unsafe {
            self.gl
                .bind_buffer_base(glow::UNIFORM_BUFFER, self.binding, None);
            self.gl.delete_buffer(self.buffer);
        }
    }
}

fn texture_view_target(
    dimension: HalTextureViewDimension,
    texture_target: u32,
    supports_cube_map_array: bool,
) -> Result<u32, HalError> {
    match dimension {
        HalTextureViewDimension::D1 => Ok(glow::TEXTURE_2D),
        HalTextureViewDimension::D2 if texture_target == glow::TEXTURE_2D_MULTISAMPLE => {
            Ok(glow::TEXTURE_2D_MULTISAMPLE)
        }
        HalTextureViewDimension::D2 => Ok(glow::TEXTURE_2D),
        HalTextureViewDimension::D2Array => Ok(glow::TEXTURE_2D_ARRAY),
        HalTextureViewDimension::Cube => Ok(glow::TEXTURE_CUBE_MAP),
        HalTextureViewDimension::CubeArray if supports_cube_map_array => {
            Ok(glow::TEXTURE_CUBE_MAP_ARRAY)
        }
        HalTextureViewDimension::CubeArray => Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES lacks cube-array textures",
        }),
        HalTextureViewDimension::D3 => Ok(glow::TEXTURE_3D),
    }
}

struct TextureUnitCleanup<'a> {
    gl: &'a glow::Context,
    texture_units: Vec<u32>,
    texture_views: Vec<TransientTextureView>,
}

impl Drop for TextureUnitCleanup<'_> {
    fn drop(&mut self) {
        unsafe {
            for view in &self.texture_views {
                self.gl.active_texture(glow::TEXTURE0 + view.unit);
                self.gl.bind_texture(view.target, None);
                self.gl.delete_texture(view.texture);
            }
            for unit in &self.texture_units {
                self.gl.bind_sampler(*unit, None);
            }
            self.gl.active_texture(glow::TEXTURE0);
        }
    }
}

#[derive(Clone, Copy)]
struct RenderDrawCaps {
    supports_base_vertex: bool,
    supports_vertex_array_bgra: bool,
    sample_mask_i: Option<super::device::GlesSampleMaskIFn>,
}

fn submit_render_pass(
    gl: &glow::Context,
    pass: &HalRenderPass,
    caps: RenderDrawCaps,
    placeholder_sampler: glow::Sampler,
    texture_view_caps: TextureViewCaps,
) -> Result<(), HalError> {
    reject_external_texture_bindings(pass.bind_external_textures.len())?;
    let fbo = create_render_fbo(gl, pass)?;
    let pipeline = match &pass.pipeline {
        None => {
            let _cleanup = RenderPassCleanup { gl, fbo, vao: None };
            return Ok(());
        }
        Some(HalRenderPipeline::Gles(pipeline)) => pipeline,
        Some(_) => {
            let _cleanup = RenderPassCleanup { gl, fbo, vao: None };
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "render pass pipeline is not a GLES pipeline",
            });
        }
    };
    let vao = unsafe {
        gl.create_vertex_array()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateVertexArray failed",
            })
    };
    let vao = match vao {
        Ok(vao) => vao,
        Err(error) => {
            let _cleanup = RenderPassCleanup { gl, fbo, vao: None };
            return Err(error);
        }
    };
    let _cleanup = RenderPassCleanup {
        gl,
        fbo,
        vao: Some(vao),
    };
    run_render_draw(
        gl,
        pass,
        pipeline,
        vao,
        caps,
        placeholder_sampler,
        texture_view_caps,
    )?;
    resolve_render_pass(gl, pass, fbo)
}

struct RenderPassCleanup<'a> {
    gl: &'a glow::Context,
    fbo: glow::Framebuffer,
    vao: Option<glow::VertexArray>,
}

impl Drop for RenderPassCleanup<'_> {
    fn drop(&mut self) {
        unsafe {
            if let Some(vao) = self.vao {
                self.gl.bind_vertex_array(None);
                self.gl.delete_vertex_array(vao);
            }
            self.gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            self.gl.delete_framebuffer(self.fbo);
            self.gl.use_program(None);
            self.gl.active_texture(glow::TEXTURE0);
            self.gl.color_mask(true, true, true, true);
            self.gl.disable(glow::BLEND);
            self.gl.disable(glow::STENCIL_TEST);
            self.gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE);
            self.gl.disable(glow::SAMPLE_MASK);
            self.gl.memory_barrier(glow::ALL_BARRIER_BITS);
        }
    }
}

fn create_render_fbo(
    gl: &glow::Context,
    pass: &HalRenderPass,
) -> Result<glow::Framebuffer, HalError> {
    // T-G8 (MRT): resolve every color slot up front -- `Some` slots must be
    // 2D GLES textures with a live GL name; `None` slots stay sparse and get
    // `GL_NONE` in the glDrawBuffers list below. Hoisting the per-target
    // checks before `glCreateFramebuffer` keeps every error path free of FBO
    // cleanup.
    let color_targets = pass
        .color_targets
        .iter()
        .map(|slot| {
            let Some(target) = slot else {
                return Ok(None);
            };
            let HalTexture::Gles(texture) = &target.texture else {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "render pass color target is not a GLES texture",
                });
            };
            if !matches!(
                texture.meta().target,
                glow::TEXTURE_2D | glow::TEXTURE_2D_MULTISAMPLE
            ) {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES render pass supports only 2D color attachments",
                });
            }
            Ok(Some((target, texture, texture.raw_or_err()?)))
        })
        .collect::<Result<Vec<_>, HalError>>()?;
    // Trailing `None` slots need no glDrawBuffers entry; truncating to the
    // last attachment keeps the list within the driver's draw-buffer limit.
    let attachment_count = color_targets
        .iter()
        .rposition(Option::is_some)
        .map_or(0, |last| last + 1);
    let depth_stencil_target = pass
        .depth_stencil_attachment
        .as_ref()
        .map(|attachment| match &attachment.texture {
            HalTexture::Gles(texture) => Ok(texture),
            _ => Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "render pass depth-stencil attachment is not a GLES texture",
            }),
        })
        .transpose()?;
    let size_texture = color_targets
        .iter()
        .flatten()
        .map(|(_, texture, _)| *texture)
        .next()
        .or(depth_stencil_target)
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "render pass requires an attachment",
        })?;
    let width = i32_from_u32(
        size_texture.meta().width,
        "render target width exceeds GLES limit",
    )?;
    let height = i32_from_u32(
        size_texture.meta().height,
        "render target height exceeds GLES limit",
    )?;

    unsafe {
        // T-G8 (MRT): GLES 3.1 guarantees at least 4 color attachments /
        // draw buffers; WebGPU's maxColorAttachments caps the useful range
        // at 8. Anything beyond the driver limit cannot be attached.
        let max_draw_buffers = gl
            .get_parameter_i32(glow::MAX_COLOR_ATTACHMENTS)
            .min(gl.get_parameter_i32(glow::MAX_DRAW_BUFFERS))
            .clamp(0, 8) as usize;
        if attachment_count > max_draw_buffers {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message:
                    "GLES render pass color attachment count exceeds the driver draw-buffer limit",
            });
        }
        let fbo = gl
            .create_framebuffer()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateFramebuffer failed (render)",
            })?;
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(fbo));
        let mut draw_buffer_list = Vec::with_capacity(attachment_count);
        for (index, slot) in color_targets.iter().take(attachment_count).enumerate() {
            let Some((target, texture, color_texture)) = slot else {
                // Sparse slot: nothing attached, fragment output discarded.
                draw_buffer_list.push(glow::NONE);
                continue;
            };
            let attachment = glow::COLOR_ATTACHMENT0 + index as u32;
            attach_2d_texture_to_framebuffer(
                gl,
                glow::DRAW_FRAMEBUFFER,
                attachment,
                texture.meta(),
                *color_texture,
                target.mip_level,
            );
            draw_buffer_list.push(attachment);
        }
        gl.draw_buffers(&draw_buffer_list);
        if let (Some(attachment), Some(target_texture)) =
            (&pass.depth_stencil_attachment, depth_stencil_target)
        {
            let depth_stencil_texture = target_texture.raw_or_err()?;
            let meta = target_texture.meta();
            let attachment_point = match (
                format_has_depth_aspect(attachment.format),
                format_has_stencil_aspect(attachment.format),
            ) {
                (true, true) => glow::DEPTH_STENCIL_ATTACHMENT,
                (true, false) => glow::DEPTH_ATTACHMENT,
                (false, true) => glow::STENCIL_ATTACHMENT,
                (false, false) => {
                    gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
                    gl.delete_framebuffer(fbo);
                    return Err(HalError::BufferOperationFailed {
                        backend: BACKEND,
                        message: "GLES render pass depth-stencil format has no supported aspect",
                    });
                }
            };
            attach_depth_stencil_texture_to_framebuffer(
                gl,
                glow::DRAW_FRAMEBUFFER,
                attachment_point,
                meta,
                depth_stencil_texture,
                attachment.mip_level,
                attachment.array_layer,
            )?;
        }
        if gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            gl.delete_framebuffer(fbo);
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "framebuffer incomplete for render pass",
            });
        }
        if let Some(viewport) = pass.viewport {
            gl.viewport(
                viewport.x as i32,
                viewport.y as i32,
                viewport.width as i32,
                viewport.height as i32,
            );
            gl.depth_range_f32(viewport.min_depth, viewport.max_depth);
        } else {
            gl.viewport(0, 0, width, height);
            gl.depth_range_f32(0.0, 1.0);
        }
        if let Some(rect) = pass.scissor_rect {
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(
                rect.x as i32,
                rect.y as i32,
                rect.width as i32,
                rect.height as i32,
            );
        } else {
            gl.disable(glow::SCISSOR_TEST);
        }
        let mut clear_mask = 0;
        // T-G8 (MRT): each attachment carries its own load op and clear
        // value, so clears go through the per-draw-buffer `glClearBuffer*`
        // entry points rather than a single global `glClearColor`+`glClear`.
        // Integer attachments additionally *require* the typed
        // `glClearBuffer{i,u}iv` variants (`glClear` is undefined for
        // integer formats, T-G7).
        for (index, slot) in color_targets.iter().enumerate() {
            let Some((color, _, _)) = slot else {
                continue;
            };
            if !matches!(color.load_op, HalRenderLoadOp::Clear) {
                continue;
            }
            let draw_buffer = index as u32;
            let [r, g, b, a] = color.clear_color;
            match color_clear_kind(color.view_format) {
                GlesClearKind::Float => {
                    gl.clear_buffer_f32_slice(
                        glow::COLOR,
                        draw_buffer,
                        &[r as f32, g as f32, b as f32, a as f32],
                    );
                }
                GlesClearKind::Sint => {
                    gl.clear_buffer_i32_slice(
                        glow::COLOR,
                        draw_buffer,
                        &[r as i32, g as i32, b as i32, a as i32],
                    );
                }
                GlesClearKind::Uint => {
                    gl.clear_buffer_u32_slice(
                        glow::COLOR,
                        draw_buffer,
                        &[r as u32, g as u32, b as u32, a as u32],
                    );
                }
            }
        }
        if let Some(depth_stencil) = &pass.depth_stencil_attachment {
            if !depth_stencil.depth_read_only
                && matches!(depth_stencil.depth_load_op, HalRenderLoadOp::Clear)
            {
                gl.depth_mask(true);
                gl.clear_depth_f32(depth_stencil.depth_clear_value);
                clear_mask |= glow::DEPTH_BUFFER_BIT;
            }
            if !depth_stencil.stencil_read_only
                && matches!(depth_stencil.stencil_load_op, HalRenderLoadOp::Clear)
            {
                let stencil = i32_from_u32(
                    depth_stencil.stencil_clear_value,
                    "stencil clear value exceeds GLES limit",
                )?;
                gl.stencil_mask(u32::MAX);
                gl.clear_stencil(stencil);
                clear_mask |= glow::STENCIL_BUFFER_BIT;
            }
        }
        if clear_mask != 0 {
            gl.clear(clear_mask);
        }
        Ok(fbo)
    }
}

unsafe fn attach_2d_texture_to_framebuffer(
    gl: &glow::Context,
    framebuffer_target: u32,
    attachment: u32,
    meta: &GlesTextureMeta,
    texture: glow::Texture,
    mip_level: u32,
) {
    unsafe {
        gl.framebuffer_texture_2d(
            framebuffer_target,
            attachment,
            meta.target,
            Some(texture),
            if meta.target == glow::TEXTURE_2D_MULTISAMPLE {
                0
            } else {
                mip_level as i32
            },
        );
    }
}

unsafe fn attach_depth_stencil_texture_to_framebuffer(
    gl: &glow::Context,
    framebuffer_target: u32,
    attachment: u32,
    meta: &GlesTextureMeta,
    texture: glow::Texture,
    mip_level: u32,
    array_layer: u32,
) -> Result<(), HalError> {
    unsafe {
        match meta.target {
            glow::TEXTURE_2D | glow::TEXTURE_2D_MULTISAMPLE => {
                attach_2d_texture_to_framebuffer(
                    gl,
                    framebuffer_target,
                    attachment,
                    meta,
                    texture,
                    mip_level,
                );
            }
            glow::TEXTURE_2D_ARRAY | glow::TEXTURE_3D => {
                gl.framebuffer_texture_layer(
                    framebuffer_target,
                    attachment,
                    Some(texture),
                    i32_from_u32(
                        mip_level,
                        "depth-stencil attachment mip level exceeds GLES limit",
                    )?,
                    i32_from_u32(
                        array_layer,
                        "depth-stencil attachment array layer exceeds GLES limit",
                    )?,
                );
            }
            _ => {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES render pass depth-stencil attachment target is unsupported",
                });
            }
        }
    }
    Ok(())
}

unsafe fn attach_resolve_texture_to_framebuffer(
    gl: &glow::Context,
    attachment: u32,
    meta: &GlesTextureMeta,
    texture: glow::Texture,
    mip_level: u32,
    array_layer: u32,
) -> Result<(), HalError> {
    unsafe {
        match meta.target {
            glow::TEXTURE_2D => {
                gl.framebuffer_texture_2d(
                    glow::DRAW_FRAMEBUFFER,
                    attachment,
                    glow::TEXTURE_2D,
                    Some(texture),
                    i32_from_u32(mip_level, "resolve target mip level exceeds GLES limit")?,
                );
            }
            glow::TEXTURE_2D_ARRAY => {
                gl.framebuffer_texture_layer(
                    glow::DRAW_FRAMEBUFFER,
                    attachment,
                    Some(texture),
                    i32_from_u32(mip_level, "resolve target mip level exceeds GLES limit")?,
                    i32_from_u32(array_layer, "resolve target array layer exceeds GLES limit")?,
                );
            }
            _ => {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES resolve target must be a single-sample 2D texture",
                });
            }
        }
    }
    Ok(())
}

fn resolve_render_pass(
    gl: &glow::Context,
    pass: &HalRenderPass,
    render_fbo: glow::Framebuffer,
) -> Result<(), HalError> {
    let resolves = pass
        .color_targets
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| slot.as_ref().map(|target| (index, target)))
        .filter(|(_, target)| target.resolve_target.is_some())
        .collect::<Vec<_>>();
    if resolves.is_empty() {
        return Ok(());
    }

    unsafe {
        let draw_fbo = gl
            .create_framebuffer()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateFramebuffer failed (resolve)",
            })?;
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(render_fbo));
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(draw_fbo));
        let mut result = Ok(());

        for (index, target) in resolves {
            let HalTexture::Gles(source) = &target.texture else {
                result = Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "resolve source is not a GLES texture",
                });
                break;
            };
            if source.meta().sample_count <= 1 {
                result = Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES resolve source must be multisampled",
                });
                break;
            }
            let Some(HalTexture::Gles(resolve)) = target.resolve_target.as_ref() else {
                result = Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "resolve target is not a GLES texture",
                });
                break;
            };
            if resolve.meta().sample_count != 1 {
                result = Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES resolve target must be single-sampled",
                });
                break;
            }
            let attachment = glow::COLOR_ATTACHMENT0 + index as u32;
            gl.read_buffer(attachment);
            attach_resolve_texture_to_framebuffer(
                gl,
                glow::COLOR_ATTACHMENT0,
                resolve.meta(),
                resolve.raw_or_err()?,
                target.resolve_mip_level,
                target.resolve_array_layer,
            )?;
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
            if gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                result = Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "framebuffer incomplete for render pass resolve",
                });
                break;
            }
            gl.blit_framebuffer(
                0,
                0,
                source.meta().width as i32,
                source.meta().height as i32,
                0,
                0,
                resolve.meta().width as i32,
                resolve.meta().height as i32,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            );
            gl.framebuffer_texture_2d(
                glow::DRAW_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                None,
                0,
            );
        }

        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(render_fbo));
        gl.delete_framebuffer(draw_fbo);
        result
    }
}

fn run_render_draw(
    gl: &glow::Context,
    pass: &HalRenderPass,
    pipeline: &super::pipeline::GlesRenderPipeline,
    vao: glow::VertexArray,
    caps: RenderDrawCaps,
    placeholder_sampler: glow::Sampler,
    texture_view_caps: TextureViewCaps,
) -> Result<(), HalError> {
    let program = pipeline.raw_or_err()?;
    unsafe {
        gl.use_program(Some(program));
    }
    bind_render_buffers(gl, pass, pipeline)?;
    let texture_units = bind_combined_samplers(
        gl,
        pipeline.combined_samplers(),
        &pass.bind_textures,
        &pass.bind_samplers,
        placeholder_sampler,
        texture_view_caps,
    )?;
    let _texture_cleanup = TextureUnitCleanup {
        gl,
        texture_units: texture_units.units,
        texture_views: texture_units.texture_views,
    };
    let _texture_metadata_cleanup = bind_texture_metadata_ubo(
        gl,
        pipeline.texture_metadata_ubo_binding(),
        pipeline.texture_metadata_slots(),
        &pass.bind_textures,
    )?;
    bind_storage_textures(
        gl,
        pipeline.bindings(),
        pipeline.binding_remaps(),
        &pass.bind_textures,
    )?;
    let first_instance = pass.draw.map(draw_first_instance).unwrap_or(0);
    bind_vertex_buffers(
        gl,
        pass,
        pipeline,
        vao,
        first_instance,
        caps.supports_vertex_array_bgra,
    )?;
    if let Some(draw) = pass.draw {
        apply_raster_state(gl, pipeline.front_face(), pipeline.cull_mode());
        apply_multisample_state(
            gl,
            pipeline.sample_mask(),
            pipeline.alpha_to_coverage_enabled(),
            caps.sample_mask_i,
        )?;
        apply_color_target_state(gl, pipeline.color_target(), pass.blend_constant);
        apply_stencil_state(gl, pipeline.depth_stencil(), pass.stencil_reference)?;
        if let Some(location) = pipeline.first_instance_location() {
            set_first_instance_uniform(gl, location, draw);
        }
        let topology = map_primitive_topology(pipeline.primitive_topology());
        run_gles_draw(gl, pass, topology, draw, caps.supports_base_vertex)?;
    }
    Ok(())
}

fn apply_raster_state(gl: &glow::Context, front_face: HalFrontFace, cull_mode: HalCullMode) {
    unsafe {
        gl.front_face(match front_face {
            HalFrontFace::Ccw => glow::CCW,
            HalFrontFace::Cw => glow::CW,
        });
        match cull_mode {
            HalCullMode::None => gl.disable(glow::CULL_FACE),
            HalCullMode::Front => {
                gl.enable(glow::CULL_FACE);
                gl.cull_face(glow::FRONT);
            }
            HalCullMode::Back => {
                gl.enable(glow::CULL_FACE);
                gl.cull_face(glow::BACK);
            }
        }
    }
}

fn apply_multisample_state(
    gl: &glow::Context,
    sample_mask: u32,
    alpha_to_coverage_enabled: bool,
    sample_mask_i: Option<super::device::GlesSampleMaskIFn>,
) -> Result<(), HalError> {
    unsafe {
        if alpha_to_coverage_enabled {
            gl.enable(glow::SAMPLE_ALPHA_TO_COVERAGE);
        } else {
            gl.disable(glow::SAMPLE_ALPHA_TO_COVERAGE);
        }
        if sample_mask == u32::MAX {
            gl.disable(glow::SAMPLE_MASK);
        } else {
            let Some(sample_mask_i) = sample_mask_i else {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES sample mask requires glSampleMaski",
                });
            };
            gl.enable(glow::SAMPLE_MASK);
            sample_mask_i(0, sample_mask);
        }
    }
    Ok(())
}

fn apply_stencil_state(
    gl: &glow::Context,
    depth_stencil: Option<HalDepthStencilState>,
    stencil_reference: u32,
) -> Result<(), HalError> {
    let Some(depth_stencil) = depth_stencil else {
        unsafe {
            gl.disable(glow::STENCIL_TEST);
        }
        return Ok(());
    };
    let reference = i32_from_u32(
        stencil_reference,
        "stencil reference value exceeds GLES limit",
    )?;
    unsafe {
        gl.enable(glow::STENCIL_TEST);
        apply_stencil_face_state(
            gl,
            glow::FRONT,
            depth_stencil.stencil_front,
            depth_stencil.stencil_read_mask,
            depth_stencil.stencil_write_mask,
            reference,
        );
        apply_stencil_face_state(
            gl,
            glow::BACK,
            depth_stencil.stencil_back,
            depth_stencil.stencil_read_mask,
            depth_stencil.stencil_write_mask,
            reference,
        );
    }
    Ok(())
}

unsafe fn apply_stencil_face_state(
    gl: &glow::Context,
    face: u32,
    state: HalStencilFaceState,
    read_mask: u32,
    write_mask: u32,
    reference: i32,
) {
    unsafe {
        gl.stencil_func_separate(
            face,
            gles_compare_function(state.compare),
            reference,
            read_mask,
        );
        gl.stencil_op_separate(
            face,
            gles_stencil_operation(state.fail_op),
            gles_stencil_operation(state.depth_fail_op),
            gles_stencil_operation(state.pass_op),
        );
        gl.stencil_mask_separate(face, write_mask);
    }
}

fn gles_compare_function(compare: HalCompareFunction) -> u32 {
    match compare {
        HalCompareFunction::Never => glow::NEVER,
        HalCompareFunction::Less => glow::LESS,
        HalCompareFunction::Equal => glow::EQUAL,
        HalCompareFunction::LessEqual => glow::LEQUAL,
        HalCompareFunction::Greater => glow::GREATER,
        HalCompareFunction::NotEqual => glow::NOTEQUAL,
        HalCompareFunction::GreaterEqual => glow::GEQUAL,
        HalCompareFunction::Always => glow::ALWAYS,
    }
}

fn gles_stencil_operation(operation: HalStencilOperation) -> u32 {
    match operation {
        HalStencilOperation::Keep => glow::KEEP,
        HalStencilOperation::Zero => glow::ZERO,
        HalStencilOperation::Replace => glow::REPLACE,
        HalStencilOperation::Invert => glow::INVERT,
        HalStencilOperation::IncrementClamp => glow::INCR,
        HalStencilOperation::DecrementClamp => glow::DECR,
        HalStencilOperation::IncrementWrap => glow::INCR_WRAP,
        HalStencilOperation::DecrementWrap => glow::DECR_WRAP,
    }
}

fn apply_color_target_state(
    gl: &glow::Context,
    color_target: Option<HalColorTargetState>,
    blend_constant: [f32; 4],
) {
    let Some(color_target) = color_target else {
        return;
    };
    unsafe {
        gl.color_mask(
            color_target.write_mask & 0x1 != 0,
            color_target.write_mask & 0x2 != 0,
            color_target.write_mask & 0x4 != 0,
            color_target.write_mask & 0x8 != 0,
        );
        if let Some(blend) = color_target.blend {
            gl.enable(glow::BLEND);
            gl.blend_color(
                blend_constant[0],
                blend_constant[1],
                blend_constant[2],
                blend_constant[3],
            );
            gl.blend_func_separate(
                gles_blend_factor(blend.color.src_factor, false),
                gles_blend_factor(blend.color.dst_factor, false),
                gles_blend_factor(blend.alpha.src_factor, true),
                gles_blend_factor(blend.alpha.dst_factor, true),
            );
            gl.blend_equation_separate(
                gles_blend_operation(blend.color.operation),
                gles_blend_operation(blend.alpha.operation),
            );
        } else {
            gl.disable(glow::BLEND);
        }
    }
}

fn gles_blend_operation(operation: HalBlendOperation) -> u32 {
    match operation {
        HalBlendOperation::Add => glow::FUNC_ADD,
        HalBlendOperation::Subtract => glow::FUNC_SUBTRACT,
        HalBlendOperation::ReverseSubtract => glow::FUNC_REVERSE_SUBTRACT,
        HalBlendOperation::Min => glow::MIN,
        HalBlendOperation::Max => glow::MAX,
    }
}

fn gles_blend_factor(factor: HalBlendFactor, alpha: bool) -> u32 {
    match factor {
        HalBlendFactor::Zero => glow::ZERO,
        HalBlendFactor::One => glow::ONE,
        HalBlendFactor::Src => {
            if alpha {
                glow::SRC_ALPHA
            } else {
                glow::SRC_COLOR
            }
        }
        HalBlendFactor::OneMinusSrc => {
            if alpha {
                glow::ONE_MINUS_SRC_ALPHA
            } else {
                glow::ONE_MINUS_SRC_COLOR
            }
        }
        HalBlendFactor::SrcAlpha => glow::SRC_ALPHA,
        HalBlendFactor::OneMinusSrcAlpha => glow::ONE_MINUS_SRC_ALPHA,
        HalBlendFactor::Dst => {
            if alpha {
                glow::DST_ALPHA
            } else {
                glow::DST_COLOR
            }
        }
        HalBlendFactor::OneMinusDst => {
            if alpha {
                glow::ONE_MINUS_DST_ALPHA
            } else {
                glow::ONE_MINUS_DST_COLOR
            }
        }
        HalBlendFactor::DstAlpha => glow::DST_ALPHA,
        HalBlendFactor::OneMinusDstAlpha => glow::ONE_MINUS_DST_ALPHA,
        HalBlendFactor::SrcAlphaSaturated => glow::SRC_ALPHA_SATURATE,
        HalBlendFactor::Constant => {
            if alpha {
                glow::CONSTANT_ALPHA
            } else {
                glow::CONSTANT_COLOR
            }
        }
        HalBlendFactor::OneMinusConstant => {
            if alpha {
                glow::ONE_MINUS_CONSTANT_ALPHA
            } else {
                glow::ONE_MINUS_CONSTANT_COLOR
            }
        }
        HalBlendFactor::Src1 => {
            if alpha {
                glow::SRC1_ALPHA
            } else {
                glow::SRC1_COLOR
            }
        }
        HalBlendFactor::OneMinusSrc1 => {
            if alpha {
                glow::ONE_MINUS_SRC1_ALPHA
            } else {
                glow::ONE_MINUS_SRC1_COLOR
            }
        }
        HalBlendFactor::Src1Alpha => glow::SRC1_ALPHA,
        HalBlendFactor::OneMinusSrc1Alpha => glow::ONE_MINUS_SRC1_ALPHA,
    }
}

/// Returns the `firstInstance` draw parameter carried by `draw`, or `0` for
/// the indirect variants, whose `firstInstance` (if any) is embedded in a
/// GPU-side buffer read at draw time and not observable here. GLES 3.1's
/// indirect-draw command structs (`DrawArraysIndirectCommand` /
/// `DrawElementsIndirectCommand`) have no `baseInstance` field at all (unlike
/// desktop GL's `ARB_base_instance`), so a non-zero indirect firstInstance
/// cannot be honored regardless -- consistent with
/// `GlesAdapter::supports_indirect_first_instance` always returning `false`
/// (`adapter.rs`), which keeps `Feature::IndirectFirstInstance` off this
/// backend's advertised feature set.
fn draw_first_instance(draw: HalDraw) -> u32 {
    match draw {
        HalDraw::Direct { first_instance, .. } | HalDraw::Indexed { first_instance, .. } => {
            first_instance
        }
        HalDraw::Indirect { .. } | HalDraw::IndexedIndirect { .. } => 0,
    }
}

fn set_first_instance_uniform(gl: &glow::Context, location: &glow::UniformLocation, draw: HalDraw) {
    unsafe {
        gl.uniform_1_u32(Some(location), draw_first_instance(draw));
    }
}

/// Computes the byte offset to add to an instance-stepped vertex buffer's
/// base offset so that instance 0 of the draw reads row `first_instance`
/// instead of row `0`.
///
/// GLES 3.1's `glDrawArraysInstanced`/`glDrawElementsInstanced` have no
/// `baseInstance` parameter (that is `ARB_base_instance`, a desktop-GL-only
/// extension), so per-instance vertex attributes always fetch starting at
/// row `0` regardless of the WebGPU `firstInstance` draw parameter. Dawn's
/// GL backend compensates by adding `firstInstance * arrayStride` to every
/// instance-step vertex buffer's attribute offset at draw time
/// (`VertexStateBufferBindingTracker::Apply`,
/// `third_party/dawn/src/dawn/native/opengl/CommandBufferGL.cpp:259-261`);
/// this mirrors that formula. Returns `Ok(0)` for `first_instance == 0`
/// (the common case) without doing arithmetic.
fn instance_step_offset(array_stride: u64, first_instance: u32) -> Result<i64, HalError> {
    if first_instance == 0 {
        return Ok(0);
    }
    let offset = u64::from(first_instance).checked_mul(array_stride).ok_or(
        HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "vertex buffer firstInstance offset exceeds GLES limit",
        },
    )?;
    i64::try_from(offset).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message: "vertex buffer firstInstance offset exceeds GLES limit",
    })
}

fn run_gles_draw(
    gl: &glow::Context,
    pass: &HalRenderPass,
    topology: u32,
    draw: HalDraw,
    supports_base_vertex: bool,
) -> Result<(), HalError> {
    match draw {
        HalDraw::Direct {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        } => {
            let first_vertex = i32_from_u32(first_vertex, "draw firstVertex exceeds GLES limit")?;
            let vertex_count = i32_from_u32(vertex_count, "draw vertexCount exceeds GLES limit")?;
            unsafe {
                if instance_count == 1 && first_instance == 0 {
                    gl.draw_arrays(topology, first_vertex, vertex_count);
                } else {
                    let instance_count =
                        i32_from_u32(instance_count, "draw instanceCount exceeds GLES limit")?;
                    gl.draw_arrays_instanced(topology, first_vertex, vertex_count, instance_count);
                }
            }
        }
        HalDraw::Indexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        } => {
            // T-G11: non-zero baseVertex maps to the base-vertex draw entry
            // points, which are core in GLES 3.2 and exposed on GLES 3.1
            // through `GL_OES/EXT_draw_elements_base_vertex`; they are used
            // opportunistically when the device reports support and the
            // draw is otherwise rejected (never silently mis-drawn).
            if base_vertex != 0 && !supports_base_vertex {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES indexed draw with non-zero baseVertex requires GLES 3.2 or OES/EXT_draw_elements_base_vertex",
                });
            }
            let (index_type, index_offset) = bind_gles_index_buffer(gl, pass, first_index)?;
            let index_count = i32_from_u32(index_count, "draw indexCount exceeds GLES limit")?;
            let instance_count =
                i32_from_u32(instance_count, "draw instanceCount exceeds GLES limit")?;
            // `first_instance` stays folded into the instance-stepped vertex
            // buffer offsets by `bind_vertex_buffers` (M2); it is orthogonal
            // to `base_vertex` and unaffected by the entry-point choice here.
            unsafe {
                if base_vertex != 0 {
                    if instance_count == 1 && first_instance == 0 {
                        gl.draw_elements_base_vertex(
                            topology,
                            index_count,
                            index_type,
                            index_offset,
                            base_vertex,
                        );
                    } else {
                        gl.draw_elements_instanced_base_vertex(
                            topology,
                            index_count,
                            index_type,
                            index_offset,
                            instance_count,
                            base_vertex,
                        );
                    }
                } else if instance_count == 1 && first_instance == 0 {
                    gl.draw_elements(topology, index_count, index_type, index_offset);
                } else {
                    gl.draw_elements_instanced(
                        topology,
                        index_count,
                        index_type,
                        index_offset,
                        instance_count,
                    );
                }
            }
        }
        HalDraw::Indirect { offset } => {
            bind_gles_indirect_buffer(gl, pass)?;
            let offset = i32_from_u64(offset, "draw indirect offset exceeds GLES limit")?;
            unsafe {
                gl.draw_arrays_indirect_offset(topology, offset);
            }
        }
        HalDraw::IndexedIndirect { offset } => {
            if pass
                .index_buffer
                .as_ref()
                .is_some_and(|bound| bound.offset != 0)
            {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES indexed indirect draw requires index buffer offset 0",
                });
            }
            let (index_type, _) = bind_gles_index_buffer(gl, pass, 0)?;
            bind_gles_indirect_buffer(gl, pass)?;
            let offset = i32_from_u64(offset, "draw indexed indirect offset exceeds GLES limit")?;
            unsafe {
                gl.draw_elements_indirect_offset(topology, index_type, offset);
            }
        }
    }
    Ok(())
}

fn bind_gles_index_buffer(
    gl: &glow::Context,
    pass: &HalRenderPass,
    first_index: u32,
) -> Result<(u32, i32), HalError> {
    let bound = pass
        .index_buffer
        .as_ref()
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "render index buffer is missing",
        })?;
    let HalBuffer::Gles(buffer) = &bound.buffer else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "render index buffer is not GLES-backed",
        });
    };
    let index_size = match bound.format {
        HalIndexFormat::Uint16 => 2,
        HalIndexFormat::Uint32 => 4,
    };
    let offset = bound
        .offset
        .checked_add(u64::from(first_index) * index_size)
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "render index buffer offset overflows",
        })?;
    let offset = i32_from_u64(offset, "render index buffer offset exceeds GLES limit")?;
    unsafe {
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(buffer.raw_or_err()?));
    }
    Ok((gles_index_type(bound.format), offset))
}

fn bind_gles_indirect_buffer(gl: &glow::Context, pass: &HalRenderPass) -> Result<(), HalError> {
    let bound = pass
        .indirect_buffer
        .as_ref()
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "render indirect buffer is missing",
        })?;
    let HalBuffer::Gles(buffer) = &bound.buffer else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "render indirect buffer is not GLES-backed",
        });
    };
    unsafe {
        gl.bind_buffer(glow::DRAW_INDIRECT_BUFFER, Some(buffer.raw_or_err()?));
    }
    Ok(())
}

fn gles_index_type(format: HalIndexFormat) -> u32 {
    match format {
        HalIndexFormat::Uint16 => glow::UNSIGNED_SHORT,
        HalIndexFormat::Uint32 => glow::UNSIGNED_INT,
    }
}

/// Resolves the effective byte size of a buffer binding.
///
/// A WebGPU bind-group entry with an unspecified size means "whole buffer from
/// `offset`", carried to the HAL as the sentinel `bound.size == u64::MAX`. This
/// mirrors the Metal backend's `bound_buffer_size`: it resolves the sentinel to
/// `buffer.size() - offset` and rejects an out-of-range offset/range.
fn gles_bound_buffer_size(
    buffer: &super::buffer::GlesBuffer,
    bound: &crate::HalBoundBuffer,
) -> Result<u64, HalError> {
    resolve_bound_buffer_size(buffer.size(), bound.offset, bound.size)
}

/// Pure resolution of a buffer binding's byte size against a known buffer size.
///
/// Split from [`gles_bound_buffer_size`] so the whole-size sentinel logic can be
/// unit-tested without a GL context. `bound_size == u64::MAX` means "whole
/// buffer from `offset`".
fn resolve_bound_buffer_size(
    buffer_size: u64,
    offset: u64,
    bound_size: u64,
) -> Result<u64, HalError> {
    if offset > buffer_size {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer binding offset exceeds buffer size",
        });
    }
    if bound_size == u64::MAX {
        buffer_size
            .checked_sub(offset)
            .ok_or(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "buffer binding range exceeds buffer size",
            })
    } else {
        Ok(bound_size)
    }
}

fn bind_render_buffers(
    gl: &glow::Context,
    pass: &HalRenderPass,
    pipeline: &super::pipeline::GlesRenderPipeline,
) -> Result<(), HalError> {
    for bound in &pass.bind_buffers {
        let HalBuffer::Gles(buffer) = &bound.buffer else {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "render pass binding is not a GLES buffer",
            });
        };
        let (target, class) = binding_target(pipeline.bindings(), bound.group, bound.binding)?;
        let Some(flat_binding) =
            flat_binding(pipeline.binding_remaps(), bound.group, bound.binding, class)
        else {
            continue;
        };
        let bound_size = gles_bound_buffer_size(buffer, bound)?;
        let buffer = buffer.raw_or_err()?;
        let offset = i32::try_from(bound.offset).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "render buffer binding offset exceeds GLES limit",
        })?;
        let size = i32::try_from(bound_size).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "render buffer binding size exceeds GLES limit",
        })?;
        unsafe {
            gl.bind_buffer_range(target, flat_binding, Some(buffer), offset, size);
        }
    }
    Ok(())
}

fn bind_vertex_buffers(
    gl: &glow::Context,
    pass: &HalRenderPass,
    pipeline: &super::pipeline::GlesRenderPipeline,
    vao: glow::VertexArray,
    first_instance: u32,
    supports_vertex_array_bgra: bool,
) -> Result<(), HalError> {
    unsafe {
        gl.bind_vertex_array(Some(vao));
    }
    for bound in &pass.vertex_buffers {
        let layout_index =
            usize::try_from(bound.binding).map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "vertex buffer binding index exceeds host limit",
            })?;
        // T-G5 (Finding G-5, specs/tracking/cts-gles-sweep-0705.md): per
        // WebGPU semantics, vertex buffers bound at slots beyond the
        // pipeline's declared vertex-buffer layouts are ignored at draw
        // time, so skip (rather than reject) bindings at slots the pipeline
        // does not declare.
        let Some(layout) = pipeline.vertex_buffers().get(layout_index) else {
            continue;
        };
        let HalBuffer::Gles(buffer) = &bound.buffer else {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "vertex buffer binding is not a GLES buffer",
            });
        };
        let buffer = buffer.raw_or_err()?;
        let stride =
            i32::try_from(layout.array_stride).map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "vertex buffer stride exceeds GLES limit",
            })?;
        let buffer_offset =
            i64::try_from(bound.offset).map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "vertex buffer offset exceeds GLES limit",
            })?;
        // M2 (GLES first-instance vertex fix): instance-stepped buffers get
        // `firstInstance * arrayStride` folded into the base offset here, on
        // every draw, because this function already re-specifies every
        // attribute pointer against a freshly created VAO for every single
        // `HalRenderPass` (see `submit_render_pass`'s
        // `gl.create_vertex_array()` -- one HAL render pass models exactly
        // one draw call, unlike Dawn's persistent-VAO GL backend). That
        // means there is no cross-draw GL state to go stale, so unlike
        // Dawn's `VertexStateBufferBindingTracker` (which dirty-tracks
        // `mFirstInstance` because its VAO persists across draws within a
        // render pass) this HAL needs no `first_instance` dirty bit: a
        // subsequent draw with a different (or zero) `first_instance`
        // recomputes this offset from scratch.
        let buffer_offset = if matches!(layout.step_mode, HalVertexStepMode::Instance) {
            buffer_offset
                .checked_add(instance_step_offset(layout.array_stride, first_instance)?)
                .ok_or(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "vertex buffer firstInstance offset exceeds GLES limit",
                })?
        } else {
            buffer_offset
        };
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(buffer));
        }
        for attribute in &layout.attributes {
            let format = map_vertex_format(attribute.format)?;
            let attribute_offset = buffer_offset
                .checked_add(i64::try_from(attribute.offset).map_err(|_| {
                    HalError::BufferOperationFailed {
                        backend: BACKEND,
                        message: "vertex attribute offset exceeds GLES limit",
                    }
                })?)
                .ok_or(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "vertex attribute offset exceeds GLES limit",
                })?;
            let attribute_offset =
                i32::try_from(attribute_offset).map_err(|_| HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "vertex attribute offset exceeds GLES limit",
                })?;
            unsafe {
                gl.enable_vertex_attrib_array(attribute.shader_location);
                if format.integer {
                    gl.vertex_attrib_pointer_i32(
                        attribute.shader_location,
                        format.components,
                        format.ty,
                        stride,
                        attribute_offset,
                    );
                } else {
                    let components = if format.bgra && supports_vertex_array_bgra {
                        debug_assert_eq!(format.components, 4);
                        debug_assert_eq!(format.ty, glow::UNSIGNED_BYTE);
                        debug_assert!(format.normalized);
                        i32::try_from(glow::BGRA).expect("GL_BGRA fits in i32")
                    } else {
                        format.components
                    };
                    gl.vertex_attrib_pointer_f32(
                        attribute.shader_location,
                        components,
                        format.ty,
                        format.normalized,
                        stride,
                        attribute_offset,
                    );
                }
                gl.vertex_attrib_divisor(
                    attribute.shader_location,
                    if matches!(layout.step_mode, HalVertexStepMode::Instance) {
                        1
                    } else {
                        0
                    },
                );
            }
        }
    }
    Ok(())
}

fn submit_texture_clear(gl: &glow::Context, clear: &HalTextureClear) -> Result<(), HalError> {
    let HalTexture::Gles(texture) = &clear.texture else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture clear target is not a GLES texture",
        });
    };
    if format_has_depth_aspect(clear.format) || format_has_stencil_aspect(clear.format) {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES texture clear supports only color formats",
        });
    }
    let raw_texture = texture.raw_or_err()?;
    let meta = texture.meta();
    reject_multisample_texture_copy(meta, "texture clear target is multisampled")?;
    let mip_level = i32_from_u32(
        clear.mip_level,
        "texture clear mip level exceeds GLES limit",
    )?;
    let mip_width = mip_dimension(meta.width, clear.mip_level);
    let mip_height = mip_dimension(meta.height, clear.mip_level);
    let layers = texture_clear_layers(meta, clear.mip_level, clear)?;
    if texture_to_buffer_compute_encoding(clear.format, HalTextureAspect::All).is_some() {
        return submit_texture_clear_zero_upload(gl, meta, raw_texture, mip_level, clear, &layers);
    }

    unsafe {
        let framebuffer = gl
            .create_framebuffer()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateFramebuffer failed",
            })?;
        normalize_texture_mip_bounds(gl, meta, raw_texture)?;
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(framebuffer));
        gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
        gl.disable(glow::SCISSOR_TEST);
        gl.viewport(
            0,
            0,
            i32_from_u32(mip_width, "texture clear width exceeds GLES limit")?,
            i32_from_u32(mip_height, "texture clear height exceeds GLES limit")?,
        );

        let mut result = Ok(());
        for layer in layers {
            attach_texture_clear_layer(gl, meta, raw_texture, mip_level, layer)?;
            if gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                result = Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "framebuffer incomplete for texture clear",
                });
                break;
            }
            match color_clear_kind(clear.format) {
                GlesClearKind::Float => {
                    gl.clear_buffer_f32_slice(glow::COLOR, 0, &[0.0, 0.0, 0.0, 0.0]);
                }
                GlesClearKind::Sint => {
                    gl.clear_buffer_i32_slice(glow::COLOR, 0, &[0, 0, 0, 0]);
                }
                GlesClearKind::Uint => {
                    gl.clear_buffer_u32_slice(glow::COLOR, 0, &[0, 0, 0, 0]);
                }
            }
        }

        gl.framebuffer_texture_2d(
            glow::DRAW_FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            None,
            0,
        );
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
        gl.delete_framebuffer(framebuffer);
        result
    }
}

fn submit_texture_clear_zero_upload(
    gl: &glow::Context,
    meta: &GlesTextureMeta,
    texture: glow::Texture,
    mip_level: i32,
    clear: &HalTextureClear,
    layers: &[i32],
) -> Result<(), HalError> {
    let mip_width = mip_dimension(meta.width, clear.mip_level);
    let mip_height = mip_dimension(meta.height, clear.mip_level);
    let layer_count = u32::try_from(layers.len()).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message: "texture clear layer count exceeds GLES limit",
    })?;
    let byte_count = u64::from(mip_width)
        .checked_mul(u64::from(mip_height))
        .and_then(|bytes| bytes.checked_mul(u64::from(layer_count)))
        .and_then(|bytes| bytes.checked_mul(u64::from(meta.format.bytes_per_pixel)))
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture clear zero upload size exceeds host limit",
        })?;
    let zeros = vec![
        0u8;
        usize::try_from(byte_count).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture clear zero upload size exceeds host limit",
        })?
    ];
    let width = i32_from_u32(mip_width, "texture clear width exceeds GLES limit")?;
    let height = i32_from_u32(mip_height, "texture clear height exceeds GLES limit")?;

    unsafe {
        normalize_texture_mip_bounds(gl, meta, texture)?;
        gl.bind_texture(meta.target, Some(texture));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
        gl.pixel_store_i32(glow::UNPACK_IMAGE_HEIGHT, 0);
        match meta.target {
            glow::TEXTURE_2D => {
                gl.tex_sub_image_2d(
                    meta.target,
                    mip_level,
                    0,
                    0,
                    width,
                    height,
                    meta.format.format,
                    meta.format.ty,
                    glow::PixelUnpackData::Slice(&zeros),
                );
            }
            glow::TEXTURE_2D_ARRAY | glow::TEXTURE_3D => {
                let first_layer = layers.first().copied().unwrap_or(0);
                gl.tex_sub_image_3d(
                    meta.target,
                    mip_level,
                    0,
                    0,
                    first_layer,
                    width,
                    height,
                    i32_from_u32(layer_count, "texture clear depth exceeds GLES limit")?,
                    meta.format.format,
                    meta.format.ty,
                    glow::PixelUnpackData::Slice(&zeros),
                );
            }
            _ => {
                gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
                gl.bind_texture(meta.target, None);
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "unsupported GLES texture clear target",
                });
            }
        }
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
        gl.bind_texture(meta.target, None);
    }
    Ok(())
}

unsafe fn attach_texture_clear_layer(
    gl: &glow::Context,
    meta: &GlesTextureMeta,
    texture: glow::Texture,
    mip_level: i32,
    layer: i32,
) -> Result<(), HalError> {
    unsafe {
        match meta.target {
            glow::TEXTURE_2D => {
                gl.framebuffer_texture_2d(
                    glow::DRAW_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::TEXTURE_2D,
                    Some(texture),
                    mip_level,
                );
            }
            glow::TEXTURE_2D_ARRAY | glow::TEXTURE_3D => {
                gl.framebuffer_texture_layer(
                    glow::DRAW_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    Some(texture),
                    mip_level,
                    layer,
                );
            }
            _ => {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "unsupported GLES texture clear target",
                });
            }
        }
    }
    Ok(())
}

fn texture_clear_layers(
    meta: &GlesTextureMeta,
    mip_level: u32,
    clear: &HalTextureClear,
) -> Result<Vec<i32>, HalError> {
    let (start, count) = match meta.target {
        glow::TEXTURE_2D => (0, 1),
        glow::TEXTURE_2D_ARRAY => (clear.base_array_layer, clear.array_layer_count),
        glow::TEXTURE_3D => (0, mip_dimension(meta.depth_or_array_layers, mip_level)),
        _ => {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "unsupported GLES texture clear target",
            });
        }
    };
    let end = start
        .checked_add(count)
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture clear layer range exceeds GLES limit",
        })?;
    let mut layers = Vec::with_capacity(count as usize);
    for layer in start..end {
        layers.push(i32_from_u32(
            layer,
            "texture clear layer index exceeds GLES limit",
        )?);
    }
    Ok(layers)
}

fn mip_dimension(size: u32, mip_level: u32) -> u32 {
    size.checked_shr(mip_level).unwrap_or(0).max(1)
}

fn submit_buffer_to_texture(
    gl: &glow::Context,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let HalBuffer::Gles(source) = &copy.buffer else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer-to-texture source is not a GLES buffer",
        });
    };
    let HalTexture::Gles(destination) = &copy.texture else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer-to-texture destination is not a GLES texture",
        });
    };

    let buffer = source.raw_or_err()?;
    let texture = destination.raw_or_err()?;
    let meta = destination.meta();
    reject_multisample_texture_copy(meta, "buffer-to-texture destination is multisampled")?;
    let row_pixels = pixels_per_row(
        copy.buffer_layout.bytes_per_row,
        meta.format.bytes_per_pixel,
    )?;
    let mip_level = i32_from_u32(copy.mip_level, "texture mip level exceeds GLES limit")?;
    let x = i32_from_u32(copy.origin.x, "texture x origin exceeds GLES limit")?;
    let y = i32_from_u32(copy.origin.y, "texture y origin exceeds GLES limit")?;
    let width = i32_from_u32(copy.extent.width, "texture copy width exceeds GLES limit")?;
    let height = i32_from_u32(copy.extent.height, "texture copy height exceeds GLES limit")?;
    let buffer_offset = u32_from_u64(
        copy.buffer_layout.offset,
        "buffer-to-texture offset exceeds GLES limit",
    )?;
    let uses_layered_target = meta.target != glow::TEXTURE_2D;
    let (z, depth, image_height) = if uses_layered_target {
        (
            i32_from_u32(copy.origin.z, "texture z origin exceeds GLES limit")?,
            i32_from_u32(
                copy.extent.depth_or_array_layers,
                "texture copy depth exceeds GLES limit",
            )?,
            i32_from_u32(
                copy.buffer_layout.rows_per_image,
                "rows per image exceeds GLES limit",
            )?,
        )
    } else {
        ensure_2d_target_copy(copy.extent.depth_or_array_layers, copy.origin.z)?;
        (0, 1, 0)
    };

    unsafe {
        gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(buffer));
        gl.bind_texture(meta.target, Some(texture));
        normalize_texture_mip_bounds(gl, meta, texture)?;
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, row_pixels);
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        if uses_layered_target {
            gl.pixel_store_i32(glow::UNPACK_IMAGE_HEIGHT, image_height);
            gl.tex_sub_image_3d(
                meta.target,
                mip_level,
                x,
                y,
                z,
                width,
                height,
                depth,
                meta.format.format,
                meta.format.ty,
                glow::PixelUnpackData::BufferOffset(buffer_offset),
            );
            gl.pixel_store_i32(glow::UNPACK_IMAGE_HEIGHT, 0);
        } else {
            gl.tex_sub_image_2d(
                meta.target,
                mip_level,
                x,
                y,
                width,
                height,
                meta.format.format,
                meta.format.ty,
                glow::PixelUnpackData::BufferOffset(buffer_offset),
            );
        }
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
        gl.bind_texture(meta.target, None);
        gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, None);
    }

    Ok(())
}

fn submit_texture_to_buffer(
    gl: &glow::Context,
    copy: &HalBufferTextureCopy,
) -> Result<(), HalError> {
    let HalTexture::Gles(source) = &copy.texture else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer source is not a GLES texture",
        });
    };
    let HalBuffer::Gles(destination) = &copy.buffer else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer destination is not a GLES buffer",
        });
    };

    let texture = source.raw_or_err()?;
    let buffer = destination.raw_or_err()?;
    let meta = source.meta();
    reject_multisample_texture_copy(meta, "texture-to-buffer source is multisampled")?;
    if matches!(copy.aspect, HalTextureAspect::StencilOnly)
        || (format_has_stencil_aspect(copy.format)
            && !matches!(copy.aspect, HalTextureAspect::DepthOnly))
    {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES cannot read back stencil formats",
        });
    }
    if let Some(encoding) = texture_to_buffer_compute_encoding(copy.format, copy.aspect) {
        return submit_texture_to_buffer_compute(gl, copy, encoding);
    }
    let mip_level = i32_from_u32(copy.mip_level, "texture mip level exceeds GLES limit")?;
    let x = i32_from_u32(copy.origin.x, "texture x origin exceeds GLES limit")?;
    let width = i32_from_u32(copy.extent.width, "texture copy width exceeds GLES limit")?;
    let uses_layered_target = meta.target != glow::TEXTURE_2D;
    if !uses_layered_target {
        ensure_2d_target_copy(copy.extent.depth_or_array_layers, copy.origin.z)?;
    }

    // Some drivers (observed on Mesa) resolve `glReadPixels` into a
    // PIXEL_PACK_BUFFER from a TEXTURE_3D layer attachment against layer 0
    // instead of the attached layer. Route 3D readbacks through client
    // memory + `glBufferSubData`; 2D and 2D-array targets keep the direct
    // pack-buffer path.
    let use_client_staging = meta.target == glow::TEXTURE_3D;
    let row_bytes = u64::from(copy.extent.width)
        .checked_mul(u64::from(meta.format.bytes_per_pixel))
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer row size exceeds GLES limit",
        })?;
    let staging_len = usize::try_from(row_bytes).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message: "texture-to-buffer row size exceeds host limit",
    })?;
    i32_from_u64(row_bytes, "texture-to-buffer row size exceeds GLES limit")?;
    if row_bytes == 0 || copy.extent.height == 0 || copy.extent.depth_or_array_layers == 0 {
        return Ok(());
    }
    if copy.extent.height > 1 && u64::from(copy.buffer_layout.bytes_per_row) < row_bytes {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "bytes_per_row is smaller than texture copy row",
        });
    }

    // Precompute every exact texel-row span so fallible arithmetic happens
    // before any GL state is touched. Reading one tight row at a time avoids
    // relying on PACK_ROW_LENGTH/PBO padding preservation, and keeps row,
    // image, pre-offset, and post-copy padding bytes untouched.
    let image_stride =
        u64::from(copy.buffer_layout.bytes_per_row) * u64::from(copy.buffer_layout.rows_per_image);
    let mut row_spans = Vec::with_capacity(
        copy.extent
            .depth_or_array_layers
            .saturating_mul(copy.extent.height) as usize,
    );
    for slice in 0..copy.extent.depth_or_array_layers {
        let layer = copy
            .origin
            .z
            .checked_add(slice)
            .ok_or(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "texture layer index exceeds GLES limit",
            })?;
        let layer = i32_from_u32(layer, "texture layer index exceeds GLES limit")?;
        let slice_offset = u64::from(slice)
            .checked_mul(image_stride)
            .and_then(|bytes| copy.buffer_layout.offset.checked_add(bytes))
            .ok_or(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "texture-to-buffer offset exceeds GLES limit",
            })?;
        for row in 0..copy.extent.height {
            let row_offset = u64::from(row)
                .checked_mul(u64::from(copy.buffer_layout.bytes_per_row))
                .and_then(|bytes| slice_offset.checked_add(bytes))
                .ok_or(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "texture-to-buffer offset exceeds GLES limit",
                })?;
            let row_end =
                row_offset
                    .checked_add(row_bytes)
                    .ok_or(HalError::BufferOperationFailed {
                        backend: BACKEND,
                        message: "texture-to-buffer range exceeds GLES limit",
                    })?;
            if row_end > destination.size() {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "texture-to-buffer range exceeds buffer size",
                });
            }
            // The staged path passes the offset to `glBufferSubData` (i32);
            // the pack-buffer path passes it to `glReadPixels` (u32). Validate
            // the stricter bound up front so the copy loop below cannot fail.
            if use_client_staging {
                i32_from_u64(row_offset, "texture-to-buffer offset exceeds GLES limit")?;
            } else {
                u32_from_u64(row_offset, "texture-to-buffer offset exceeds GLES limit")?;
            }
            let row_y = copy
                .origin
                .y
                .checked_add(row)
                .ok_or(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "texture row index exceeds GLES limit",
                })?;
            let row_y = i32_from_u32(row_y, "texture row index exceeds GLES limit")?;
            row_spans.push((layer, row_y, row_offset));
        }
    }

    unsafe {
        let framebuffer = gl
            .create_framebuffer()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateFramebuffer failed",
            })?;
        normalize_texture_mip_bounds(gl, meta, texture)?;
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(framebuffer));
        gl.read_buffer(glow::COLOR_ATTACHMENT0);
        if !use_client_staging {
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(buffer));
        }
        gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
        let mut result = Ok(());
        let mut attached_layer = None;
        for (layer, row_y, row_offset) in row_spans {
            let attachment_changed = if attached_layer == Some(layer) {
                // Reuse the framebuffer attachment across rows of the same
                // layer; `row_spans` is ordered by layer then row.
                false
            } else if uses_layered_target {
                gl.framebuffer_texture_layer(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    Some(texture),
                    mip_level,
                    layer,
                );
                attached_layer = Some(layer);
                true
            } else {
                gl.framebuffer_texture_2d(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    meta.target,
                    Some(texture),
                    mip_level,
                );
                attached_layer = Some(layer);
                true
            };
            if attachment_changed
                && gl.check_framebuffer_status(glow::READ_FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE
            {
                result = Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "framebuffer incomplete for texture-to-buffer copy",
                });
                break;
            }
            if use_client_staging {
                let mut staging = vec![0u8; staging_len];
                gl.read_pixels(
                    x,
                    row_y,
                    width,
                    1,
                    meta.format.format,
                    meta.format.ty,
                    glow::PixelPackData::Slice(&mut staging),
                );
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
                gl.buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, row_offset as i32, &staging);
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            } else {
                gl.read_pixels(
                    x,
                    row_y,
                    width,
                    1,
                    meta.format.format,
                    meta.format.ty,
                    glow::PixelPackData::BufferOffset(row_offset as u32),
                );
            }
        }
        gl.pixel_store_i32(glow::PACK_ALIGNMENT, 4);
        if !use_client_staging {
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
        }
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        gl.delete_framebuffer(framebuffer);
        result
    }
}

#[derive(Clone, Copy, Debug)]
enum TextureToBufferComputeEncoding {
    R8Snorm,
    Rg8Snorm,
    Rgba8Snorm,
    R16Unorm,
    R16Snorm,
    Rg16Unorm,
    Rg16Snorm,
    Rgba16Unorm,
    Rgba16Snorm,
    Rgb9e5Ufloat,
    Depth16Unorm,
    Depth24Plus,
    Depth32Float,
}

impl TextureToBufferComputeEncoding {
    fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::R8Snorm => 1,
            Self::Rg8Snorm | Self::R16Unorm | Self::R16Snorm | Self::Depth16Unorm => 2,
            Self::Rgba8Snorm
            | Self::Rg16Unorm
            | Self::Rg16Snorm
            | Self::Rgb9e5Ufloat
            | Self::Depth24Plus
            | Self::Depth32Float => 4,
            Self::Rgba16Unorm | Self::Rgba16Snorm => 8,
        }
    }

    fn shader_store(self) -> &'static str {
        match self {
            Self::R8Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeByte(base, packSnorm4x8(vec4(value.r, 0.0, 0.0, 0.0)) & 0xffu);"
            }
            Self::Rg8Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU16(base, packSnorm4x8(vec4(value.rg, 0.0, 0.0)) & 0xffffu);"
            }
            Self::Rgba8Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packSnorm4x8(value));"
            }
            Self::R16Unorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU16(base, packUnorm16(value.r));"
            }
            Self::R16Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU16(base, packSnorm2x16(vec2(value.r, 0.0)) & 0xffffu);"
            }
            Self::Rg16Unorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packUnorm16(value.r) | (packUnorm16(value.g) << 16));"
            }
            Self::Rg16Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packSnorm2x16(value.rg));"
            }
            Self::Rgba16Unorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packUnorm16(value.r) | (packUnorm16(value.g) << 16));\n\
                 writeU32(base + 4u, packUnorm16(value.b) | (packUnorm16(value.a) << 16));"
            }
            Self::Rgba16Snorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packSnorm2x16(value.rg));\n\
                 writeU32(base + 4u, packSnorm2x16(value.ba));"
            }
            Self::Rgb9e5Ufloat => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, packRgb9e5(value.rgb));"
            }
            Self::Depth16Unorm => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU16(base, packUnorm16(value.r));"
            }
            Self::Depth24Plus => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, uint(round(clamp(value.r, 0.0, 1.0) * 16777215.0)));"
            }
            Self::Depth32Float => {
                "vec4 value = texelFetch(u_texture, texelCoord(gid), u_mip);\n\
                 writeU32(base, floatBitsToUint(value.r));"
            }
        }
    }
}

fn texture_to_buffer_compute_encoding(
    format: crate::HalTextureFormat,
    aspect: HalTextureAspect,
) -> Option<TextureToBufferComputeEncoding> {
    match (format, aspect) {
        (crate::HalTextureFormat::R8Snorm, HalTextureAspect::All) => {
            Some(TextureToBufferComputeEncoding::R8Snorm)
        }
        (crate::HalTextureFormat::Rg8Snorm, HalTextureAspect::All) => {
            Some(TextureToBufferComputeEncoding::Rg8Snorm)
        }
        (crate::HalTextureFormat::Rgba8Snorm, HalTextureAspect::All) => {
            Some(TextureToBufferComputeEncoding::Rgba8Snorm)
        }
        (crate::HalTextureFormat::R16Unorm, HalTextureAspect::All) => {
            Some(TextureToBufferComputeEncoding::R16Unorm)
        }
        (crate::HalTextureFormat::R16Snorm, HalTextureAspect::All) => {
            Some(TextureToBufferComputeEncoding::R16Snorm)
        }
        (crate::HalTextureFormat::Rg16Unorm, HalTextureAspect::All) => {
            Some(TextureToBufferComputeEncoding::Rg16Unorm)
        }
        (crate::HalTextureFormat::Rg16Snorm, HalTextureAspect::All) => {
            Some(TextureToBufferComputeEncoding::Rg16Snorm)
        }
        (crate::HalTextureFormat::Rgba16Unorm, HalTextureAspect::All) => {
            Some(TextureToBufferComputeEncoding::Rgba16Unorm)
        }
        (crate::HalTextureFormat::Rgba16Snorm, HalTextureAspect::All) => {
            Some(TextureToBufferComputeEncoding::Rgba16Snorm)
        }
        (crate::HalTextureFormat::Rgb9e5Ufloat, HalTextureAspect::All) => {
            Some(TextureToBufferComputeEncoding::Rgb9e5Ufloat)
        }
        (
            crate::HalTextureFormat::Depth16Unorm,
            HalTextureAspect::All | HalTextureAspect::DepthOnly,
        ) => Some(TextureToBufferComputeEncoding::Depth16Unorm),
        (
            crate::HalTextureFormat::Depth24Plus,
            HalTextureAspect::All | HalTextureAspect::DepthOnly,
        ) => Some(TextureToBufferComputeEncoding::Depth24Plus),
        (crate::HalTextureFormat::Depth24PlusStencil8, HalTextureAspect::DepthOnly) => {
            Some(TextureToBufferComputeEncoding::Depth24Plus)
        }
        (
            crate::HalTextureFormat::Depth32Float,
            HalTextureAspect::All | HalTextureAspect::DepthOnly,
        ) => Some(TextureToBufferComputeEncoding::Depth32Float),
        (crate::HalTextureFormat::Depth32FloatStencil8, HalTextureAspect::DepthOnly) => {
            Some(TextureToBufferComputeEncoding::Depth32Float)
        }
        _ => None,
    }
}

fn submit_texture_to_buffer_compute(
    gl: &glow::Context,
    copy: &HalBufferTextureCopy,
    encoding: TextureToBufferComputeEncoding,
) -> Result<(), HalError> {
    let HalTexture::Gles(source) = &copy.texture else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer source is not a GLES texture",
        });
    };
    let HalBuffer::Gles(destination) = &copy.buffer else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer destination is not a GLES buffer",
        });
    };

    let texture = source.raw_or_err()?;
    let buffer = destination.raw_or_err()?;
    let meta = source.meta();
    let bytes_per_pixel = encoding.bytes_per_pixel();
    let row_bytes = u64::from(copy.extent.width)
        .checked_mul(u64::from(bytes_per_pixel))
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer row size exceeds GLES limit",
        })?;
    if row_bytes == 0 || copy.extent.height == 0 || copy.extent.depth_or_array_layers == 0 {
        return Ok(());
    }
    if copy.extent.height > 1 && u64::from(copy.buffer_layout.bytes_per_row) < row_bytes {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "bytes_per_row is smaller than texture copy row",
        });
    }
    if meta.target == glow::TEXTURE_2D {
        ensure_2d_target_copy(copy.extent.depth_or_array_layers, copy.origin.z)?;
    }

    let image_stride =
        u64::from(copy.buffer_layout.bytes_per_row) * u64::from(copy.buffer_layout.rows_per_image);
    let mut row_spans = Vec::with_capacity(
        copy.extent
            .depth_or_array_layers
            .saturating_mul(copy.extent.height) as usize,
    );
    for slice in 0..copy.extent.depth_or_array_layers {
        let slice_offset = u64::from(slice)
            .checked_mul(image_stride)
            .and_then(|bytes| copy.buffer_layout.offset.checked_add(bytes))
            .ok_or(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "texture-to-buffer offset exceeds GLES limit",
            })?;
        for row in 0..copy.extent.height {
            let row_offset = u64::from(row)
                .checked_mul(u64::from(copy.buffer_layout.bytes_per_row))
                .and_then(|bytes| slice_offset.checked_add(bytes))
                .ok_or(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "texture-to-buffer offset exceeds GLES limit",
                })?;
            let row_end =
                row_offset
                    .checked_add(row_bytes)
                    .ok_or(HalError::BufferOperationFailed {
                        backend: BACKEND,
                        message: "texture-to-buffer range exceeds GLES limit",
                    })?;
            if row_end > destination.size() {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "texture-to-buffer range exceeds buffer size",
                });
            }
            i32_from_u64(row_offset, "texture-to-buffer offset exceeds GLES limit")?;
            row_spans.push(row_offset);
        }
    }

    let staging_len = row_bytes
        .checked_mul(u64::from(copy.extent.height))
        .and_then(|bytes| bytes.checked_mul(u64::from(copy.extent.depth_or_array_layers)))
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer staging size exceeds GLES limit",
        })?;
    let staging_len =
        usize::try_from(staging_len).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer staging size exceeds host limit",
        })?;
    let staging_words = staging_len.div_ceil(4).max(1);
    let staging_bytes = staging_words
        .checked_mul(4)
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer staging size exceeds host limit",
        })?;
    let staging_size =
        i32::try_from(staging_bytes).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer staging size exceeds GLES limit",
        })?;
    let row_bytes_usize =
        usize::try_from(row_bytes).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer row size exceeds host limit",
        })?;

    unsafe {
        let program = create_texture_to_buffer_compute_program(gl, meta.target, encoding)?;
        let staging = gl
            .create_buffer()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateBuffer failed",
            })?;
        let zeros = vec![0u8; staging_bytes];
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(staging));
        gl.buffer_data_u8_slice(glow::SHADER_STORAGE_BUFFER, &zeros, glow::STREAM_READ);
        gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(staging));

        normalize_texture_mip_bounds(gl, meta, texture)?;
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(meta.target, Some(texture));
        apply_depth_stencil_texture_mode(gl, meta.target, meta, copy.aspect);
        gl.use_program(Some(program));
        if let Some(location) = gl.get_uniform_location(program, "u_texture") {
            gl.uniform_1_i32(Some(&location), 0);
        }
        if let Some(location) = gl.get_uniform_location(program, "u_mip") {
            gl.uniform_1_i32(
                Some(&location),
                i32_from_u32(copy.mip_level, "texture mip level exceeds GLES limit")?,
            );
        }
        if let Some(location) = gl.get_uniform_location(program, "u_origin") {
            gl.uniform_3_u32(Some(&location), copy.origin.x, copy.origin.y, copy.origin.z);
        }
        if let Some(location) = gl.get_uniform_location(program, "u_extent") {
            gl.uniform_3_u32(
                Some(&location),
                copy.extent.width,
                copy.extent.height,
                copy.extent.depth_or_array_layers,
            );
        }

        let groups_x = copy.extent.width.div_ceil(8);
        let groups_y = copy.extent.height.div_ceil(8);
        gl.dispatch_compute(groups_x, groups_y, copy.extent.depth_or_array_layers);
        gl.memory_barrier(glow::ALL_BARRIER_BITS);

        gl.bind_buffer(glow::COPY_READ_BUFFER, Some(staging));
        let ptr = gl.map_buffer_range(glow::COPY_READ_BUFFER, 0, staging_size, glow::MAP_READ_BIT);
        let result = if ptr.is_null() {
            Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glMapBufferRange failed",
            })
        } else {
            let staged = std::slice::from_raw_parts(ptr, staging_len);
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
            for (index, row_offset) in row_spans.iter().copied().enumerate() {
                let staged_offset = index * row_bytes_usize;
                gl.buffer_sub_data_u8_slice(
                    glow::COPY_WRITE_BUFFER,
                    row_offset as i32,
                    &staged[staged_offset..staged_offset + row_bytes_usize],
                );
            }
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            Ok(())
        };
        if !ptr.is_null() {
            gl.unmap_buffer(glow::COPY_READ_BUFFER);
        }
        gl.bind_buffer(glow::COPY_READ_BUFFER, None);
        gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, None);
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);
        gl.bind_texture(meta.target, None);
        gl.use_program(None);
        gl.delete_buffer(staging);
        gl.delete_program(program);
        result
    }
}

fn create_texture_to_buffer_compute_program(
    gl: &glow::Context,
    target: u32,
    encoding: TextureToBufferComputeEncoding,
) -> Result<glow::Program, HalError> {
    let sampler_type = match target {
        glow::TEXTURE_2D => "sampler2D",
        glow::TEXTURE_2D_ARRAY => "sampler2DArray",
        glow::TEXTURE_3D => "sampler3D",
        _ => {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "unsupported GLES texture target for compute readback",
            });
        }
    };
    let texel_coord_function = match target {
        glow::TEXTURE_2D => {
            "ivec2 texelCoord(uvec3 gid) {\n\
                 uvec2 coord = u_origin.xy + gid.xy;\n\
                 return ivec2(int(coord.x), int(coord.y));\n\
             }"
        }
        glow::TEXTURE_2D_ARRAY | glow::TEXTURE_3D => {
            "ivec3 texelCoord(uvec3 gid) {\n\
                 uvec3 coord = u_origin + gid;\n\
                 return ivec3(int(coord.x), int(coord.y), int(coord.z));\n\
             }"
        }
        _ => unreachable!(),
    };
    let source = format!(
        "#version 310 es\n\
         precision highp float;\n\
         precision highp int;\n\
         precision highp sampler2D;\n\
         precision highp sampler2DArray;\n\
         precision highp sampler3D;\n\
         layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n\
         uniform {sampler_type} u_texture;\n\
         uniform int u_mip;\n\
         uniform uvec3 u_origin;\n\
         uniform uvec3 u_extent;\n\
         layout(std430, binding = 0) buffer Readback {{ coherent uint data[]; }};\n\
         uint packUnorm16(float value) {{ return uint(round(clamp(value, 0.0, 1.0) * 65535.0)); }}\n\
         uint packRgb9e5(vec3 value) {{\n\
             vec3 color = clamp(value, vec3(0.0), vec3(65408.0));\n\
             float maxChannel = max(max(color.r, color.g), color.b);\n\
             if (maxChannel == 0.0) {{ return 0u; }}\n\
             float exponent = max(-16.0, floor(log2(maxChannel))) + 1.0;\n\
             uint sharedExponent = uint(exponent + 15.0);\n\
             float scale = exp2(exponent - 9.0);\n\
             uint maxMantissa = uint(floor(maxChannel / scale + 0.5));\n\
             if (maxMantissa == 512u) {{\n\
                 sharedExponent += 1u;\n\
                 scale *= 2.0;\n\
             }}\n\
             vec3 rounded = floor(color / scale + vec3(0.5));\n\
             uvec3 mantissa = min(uvec3(uint(rounded.r), uint(rounded.g), uint(rounded.b)), uvec3(511u));\n\
             return (sharedExponent << 27) | (mantissa.b << 18) | (mantissa.g << 9) | mantissa.r;\n\
         }}\n\
         void writeByte(uint offset, uint value) {{\n\
             uint word = offset >> 2;\n\
             uint shift = (offset & 3u) * 8u;\n\
             atomicOr(data[word], (value & 0xffu) << shift);\n\
         }}\n\
         void writeU16(uint offset, uint value) {{\n\
             uint word = offset >> 2;\n\
             uint shift = (offset & 2u) * 8u;\n\
             atomicOr(data[word], (value & 0xffffu) << shift);\n\
         }}\n\
         void writeU32(uint offset, uint value) {{ data[offset >> 2] = value; }}\n\
         uvec3 globalCoord() {{ return gl_GlobalInvocationID.xyz; }}\n\
         {texel_coord_function}\n\
         void main() {{\n\
             uvec3 gid = globalCoord();\n\
             if (any(greaterThanEqual(gid, u_extent))) {{ return; }}\n\
             uint linear = ((gid.z * u_extent.y + gid.y) * u_extent.x + gid.x);\n\
             uint base = linear * {bpp}u;\n\
             {store}\n\
         }}\n",
        bpp = encoding.bytes_per_pixel(),
        store = encoding.shader_store(),
    );

    unsafe {
        let shader = gl.create_shader(glow::COMPUTE_SHADER).map_err(|_| {
            HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateShader failed",
            }
        })?;
        gl.shader_source(shader, &source);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            let info = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            eprintln!("GLES texture-to-buffer compute shader compile log: {info}");
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "texture-to-buffer compute shader compilation failed",
            });
        }
        let program = gl
            .create_program()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateProgram failed",
            })?;
        gl.attach_shader(program, shader);
        gl.link_program(program);
        gl.detach_shader(program, shader);
        gl.delete_shader(shader);
        if !gl.get_program_link_status(program) {
            let info = gl.get_program_info_log(program);
            gl.delete_program(program);
            eprintln!("GLES texture-to-buffer compute program link log: {info}");
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "texture-to-buffer compute program linking failed",
            });
        }
        Ok(program)
    }
}

fn submit_texture_to_texture(gl: &glow::Context, copy: &HalTextureCopy) -> Result<(), HalError> {
    let HalTexture::Gles(source) = &copy.source else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-texture source is not a GLES texture",
        });
    };
    let HalTexture::Gles(destination) = &copy.destination else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-texture destination is not a GLES texture",
        });
    };

    let source_texture = source.raw_or_err()?;
    let destination_texture = destination.raw_or_err()?;
    reject_multisample_texture_copy(source.meta(), "texture-to-texture source is multisampled")?;
    reject_multisample_texture_copy(
        destination.meta(),
        "texture-to-texture destination is multisampled",
    )?;
    let source_mip_level = i32_from_u32(
        copy.source_mip_level,
        "source texture mip level exceeds GLES limit",
    )?;
    let destination_mip_level = i32_from_u32(
        copy.destination_mip_level,
        "destination texture mip level exceeds GLES limit",
    )?;
    let source_x = i32_from_u32(
        copy.source_origin.x,
        "source texture x origin exceeds GLES limit",
    )?;
    let source_y = i32_from_u32(
        copy.source_origin.y,
        "source texture y origin exceeds GLES limit",
    )?;
    let destination_x = i32_from_u32(
        copy.destination_origin.x,
        "destination texture x origin exceeds GLES limit",
    )?;
    let destination_y = i32_from_u32(
        copy.destination_origin.y,
        "destination texture y origin exceeds GLES limit",
    )?;
    let width = i32_from_u32(copy.extent.width, "texture copy width exceeds GLES limit")?;
    let height = i32_from_u32(copy.extent.height, "texture copy height exceeds GLES limit")?;
    let source_z = i32_from_u32(
        copy.source_origin.z,
        "source texture z origin exceeds GLES limit",
    )?;
    let destination_z = i32_from_u32(
        copy.destination_origin.z,
        "destination texture z origin exceeds GLES limit",
    )?;
    let depth = i32_from_u32(
        copy.extent.depth_or_array_layers,
        "texture copy depth exceeds GLES limit",
    )?;
    let source_target = source.meta().target;
    let destination_target = destination.meta().target;
    if source_target == glow::TEXTURE_2D {
        ensure_2d_target_copy(copy.extent.depth_or_array_layers, copy.source_origin.z)?;
    }
    if destination_target == glow::TEXTURE_2D {
        ensure_2d_target_copy(copy.extent.depth_or_array_layers, copy.destination_origin.z)?;
    }

    if supports_copy_image(gl) {
        unsafe {
            normalize_texture_mip_bounds(gl, source.meta(), source_texture)?;
            normalize_texture_mip_bounds(gl, destination.meta(), destination_texture)?;
            gl.copy_image_sub_data(
                source_texture,
                source_target,
                source_mip_level,
                source_x,
                source_y,
                source_z,
                destination_texture,
                destination_target,
                destination_mip_level,
                destination_x,
                destination_y,
                destination_z,
                width,
                height,
                depth,
            );
        }
        return Ok(());
    }

    if source_target == glow::TEXTURE_2D && destination_target == glow::TEXTURE_2D {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GL_EXT_copy_image required for texture-to-texture copies; not supported by this GLES driver",
        });
    }

    // No glCopyImageSubData: emulate per slice by attaching the source slice
    // to a read framebuffer and copying into the bound destination texture
    // with glCopyTexSubImage{2D,3D} (both core in ES 3.1).
    let mut slice_layers = Vec::with_capacity(depth as usize);
    for slice in 0..depth {
        let source_layer = source_z
            .checked_add(slice)
            .ok_or(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "source texture layer index exceeds GLES limit",
            })?;
        let destination_layer =
            destination_z
                .checked_add(slice)
                .ok_or(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "destination texture layer index exceeds GLES limit",
                })?;
        slice_layers.push((source_layer, destination_layer));
    }
    unsafe {
        let framebuffer = gl
            .create_framebuffer()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateFramebuffer failed",
            })?;
        normalize_texture_mip_bounds(gl, source.meta(), source_texture)?;
        normalize_texture_mip_bounds(gl, destination.meta(), destination_texture)?;
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(framebuffer));
        gl.read_buffer(glow::COLOR_ATTACHMENT0);
        gl.bind_texture(destination_target, Some(destination_texture));
        let mut result = Ok(());
        for (source_layer, destination_layer) in slice_layers {
            if source_target == glow::TEXTURE_2D {
                gl.framebuffer_texture_2d(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    source_target,
                    Some(source_texture),
                    source_mip_level,
                );
            } else {
                gl.framebuffer_texture_layer(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    Some(source_texture),
                    source_mip_level,
                    source_layer,
                );
            }
            if gl.check_framebuffer_status(glow::READ_FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
                result = Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "framebuffer incomplete for texture-to-texture copy",
                });
                break;
            }
            if destination_target == glow::TEXTURE_2D {
                gl.copy_tex_sub_image_2d(
                    destination_target,
                    destination_mip_level,
                    destination_x,
                    destination_y,
                    source_x,
                    source_y,
                    width,
                    height,
                );
            } else {
                gl.copy_tex_sub_image_3d(
                    destination_target,
                    destination_mip_level,
                    destination_x,
                    destination_y,
                    destination_layer,
                    source_x,
                    source_y,
                    width,
                    height,
                );
            }
        }
        gl.bind_texture(destination_target, None);
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        gl.delete_framebuffer(framebuffer);
        result
    }
}

fn supports_copy_image(gl: &glow::Context) -> bool {
    gl.supported_extensions().contains("GL_EXT_copy_image")
        || gl.supported_extensions().contains("GL_OES_copy_image")
        || unsafe { gles_version_at_least_3_2(&gl.get_parameter_string(glow::VERSION)) }
}

fn reject_multisample_texture_copy(
    meta: &GlesTextureMeta,
    message: &'static str,
) -> Result<(), HalError> {
    if meta.sample_count > 1 {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message,
        });
    }
    Ok(())
}

fn gles_version_at_least_3_2(version: &str) -> bool {
    let Some(version_start) = version.find(|ch: char| ch.is_ascii_digit()) else {
        return false;
    };
    let version = &version[version_start..];
    let mut parts = version
        .split(|ch: char| !ch.is_ascii_digit())
        .filter(|part| !part.is_empty());
    let major = parts.next().and_then(|part| part.parse::<u32>().ok());
    let minor = parts.next().and_then(|part| part.parse::<u32>().ok());
    matches!((major, minor), (Some(major), Some(minor)) if major > 3 || (major == 3 && minor >= 2))
}

fn ensure_2d_target_copy(depth_or_array_layers: u32, z: u32) -> Result<(), HalError> {
    if depth_or_array_layers != 1 || z != 0 {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "copy addressing layers or depth slices of a plain 2D texture on GLES",
        });
    }
    Ok(())
}

fn pixels_per_row(bytes_per_row: u32, bytes_per_pixel: u32) -> Result<i32, HalError> {
    if bytes_per_pixel == 0 || !bytes_per_row.is_multiple_of(bytes_per_pixel) {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "bytes_per_row is not a multiple of bytes_per_pixel",
        });
    }
    i32::try_from(bytes_per_row / bytes_per_pixel).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message: "row pixel count exceeds GLES limit",
    })
}

unsafe fn normalize_texture_mip_bounds(
    gl: &glow::Context,
    meta: &GlesTextureMeta,
    texture: glow::Texture,
) -> Result<(), HalError> {
    if meta.target == glow::TEXTURE_2D_MULTISAMPLE {
        return Ok(());
    }
    let max_level = meta
        .mip_level_count
        .checked_sub(1)
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture has no mip levels",
        })?;
    let max_level = i32_from_u32(max_level, "texture max mip level exceeds GLES limit")?;
    unsafe {
        gl.bind_texture(meta.target, Some(texture));
        gl.tex_parameter_i32(meta.target, glow::TEXTURE_BASE_LEVEL, 0);
        gl.tex_parameter_i32(meta.target, glow::TEXTURE_MAX_LEVEL, max_level);
    }
    Ok(())
}

fn i32_from_u32(value: u32, message: &'static str) -> Result<i32, HalError> {
    i32::try_from(value).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    })
}

fn i32_from_u64(value: u64, message: &'static str) -> Result<i32, HalError> {
    i32::try_from(value).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    })
}

fn u32_from_u64(value: u64, message: &'static str) -> Result<u32, HalError> {
    u32::try_from(value).map_err(|_| HalError::BufferOperationFailed {
        backend: BACKEND,
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_copies_resolve_query_set_completes_and_writes_zeroes() {
        // Regression test for T-G4: the ResolveQuerySet arm used to call
        // `GlesBuffer::write` from inside `with_current_context`, re-acquiring
        // the non-reentrant make-current lock and self-deadlocking. Under the
        // fixed code this submit returns promptly.
        let instance = match super::super::instance::GlesInstance::new() {
            Ok(instance) => instance,
            Err(error) => {
                eprintln!("skipping GLES resolve-query-set test; backend unavailable: {error:?}");
                return;
            }
        };
        let Some(adapter) = instance.enumerate_adapters().into_iter().next() else {
            eprintln!("skipping GLES resolve-query-set test; no adapter available");
            return;
        };
        let device = match adapter.create_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping GLES resolve-query-set test; device unavailable: {error:?}");
                return;
            }
        };

        let destination = device
            .create_buffer(
                16,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES buffer creation must succeed");
        destination
            .write(0, &[0xAB; 16])
            .expect("pre-filling the destination buffer must succeed");

        device
            .queue()
            .submit_copies(&[HalCopy::ResolveQuerySet(crate::HalResolveQuerySet {
                query_set: crate::HalQuerySet::Gles { count: 2 },
                first_query: 0,
                query_count: 2,
                written_queries: Vec::new(),
                destination: HalBuffer::Gles(destination.clone()),
                destination_offset: 0,
            })])
            .expect("submitting a ResolveQuerySet copy must complete without deadlocking");

        assert_eq!(
            destination
                .read(0, 16)
                .expect("reading back the resolved buffer must succeed"),
            [0; 16],
            "resolved query range must be zero-filled"
        );
    }

    #[test]
    fn submit_render_pass_ignores_vertex_buffer_at_undeclared_slot() {
        // Regression test for T-G5: a vertex buffer bound at a slot the
        // pipeline's vertex-buffer layouts do not declare used to fail the
        // whole submit with "vertex buffer binding is missing from pipeline
        // layout". WebGPU ignores such bindings at draw time, so the fixed
        // `bind_vertex_buffers` skips them and the submit succeeds.
        let instance = match super::super::instance::GlesInstance::new() {
            Ok(instance) => instance,
            Err(error) => {
                eprintln!(
                    "skipping GLES undeclared-vertex-slot test; backend unavailable: {error:?}"
                );
                return;
            }
        };
        let Some(adapter) = instance.enumerate_adapters().into_iter().next() else {
            eprintln!("skipping GLES undeclared-vertex-slot test; no adapter available");
            return;
        };
        let device = match adapter.create_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!(
                    "skipping GLES undeclared-vertex-slot test; device unavailable: {error:?}"
                );
                return;
            }
        };

        let color_texture = device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Rgba8Unorm,
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: false,
                    copy_dst: false,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("GLES texture creation must succeed");

        let pipeline = device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    vertex: "#version 310 es\n\
                             void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }\n"
                        .to_owned(),
                    fragment: Some(
                        "#version 310 es\n\
                         precision mediump float;\n\
                         layout(location = 0) out vec4 frag_color;\n\
                         void main() { frag_color = vec4(1.0); }\n"
                            .to_owned(),
                    ),
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: Vec::new(),
                    texture_metadata_ubo_binding: None,
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: vec![Some(HalColorTargetState {
                        format: crate::HalTextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 0xf,
                    })],
                    depth_stencil: None,
                    // Zero declared vertex-buffer layouts: slot 0 is undeclared.
                    vertex_buffers: Vec::new(),
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[],
            )
            .expect("GLES render pipeline creation must succeed");

        let vertex_buffer = device
            .create_buffer(
                16,
                crate::HalBufferUsage {
                    vertex: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES buffer creation must succeed");

        let mut pass = render_pass(vec![Some(crate::HalRenderColorTarget {
            texture: HalTexture::Gles(color_texture),
            view_format: crate::HalTextureFormat::Rgba8Unorm,
            resolve_target: None,
            resolve_view_format: None,
            mip_level: 0,
            array_layer: 0,
            depth_slice: 0,
            resolve_mip_level: 0,
            resolve_array_layer: 0,
            load_op: HalRenderLoadOp::Clear,
            store: true,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        })]);
        pass.pipeline = Some(HalRenderPipeline::Gles(pipeline));
        pass.vertex_buffers = vec![crate::HalBoundBuffer {
            group: 0,
            binding: 0,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: None,
            buffer: HalBuffer::Gles(vertex_buffer),
            offset: 0,
            size: 16,
        }];
        pass.draw = Some(HalDraw::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });

        device
            .queue()
            .submit_copies(&[HalCopy::RenderPass(pass)])
            .expect("submit must ignore a vertex buffer bound at an undeclared slot");
    }

    #[test]
    fn submit_render_pass_accepts_depth_attachment_array_layer() {
        let Some(device) = gles_device_or_skip("GLES depth array-layer attachment test") else {
            return;
        };

        let color_texture = device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: false,
                    copy_dst: false,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("GLES color render attachment creation must succeed");
        let depth_texture = device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Depth24Plus,
                width: 2,
                height: 2,
                depth_or_array_layers: 2,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: false,
                    copy_dst: false,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("GLES 2D-array depth attachment creation must succeed");

        let mut pass = render_pass(vec![Some(crate::HalRenderColorTarget {
            texture: HalTexture::Gles(color_texture),
            view_format: crate::HalTextureFormat::Rgba8Unorm,
            resolve_target: None,
            resolve_view_format: None,
            mip_level: 0,
            array_layer: 0,
            depth_slice: 0,
            resolve_mip_level: 0,
            resolve_array_layer: 0,
            load_op: HalRenderLoadOp::Clear,
            store: true,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        })]);
        pass.depth_stencil_attachment = Some(crate::HalRenderDepthStencilAttachment {
            texture: HalTexture::Gles(depth_texture),
            format: crate::HalTextureFormat::Depth24Plus,
            mip_level: 0,
            array_layer: 1,
            depth_load_op: HalRenderLoadOp::Clear,
            depth_store: true,
            depth_clear_value: 0.25,
            depth_read_only: false,
            stencil_load_op: HalRenderLoadOp::Load,
            stencil_store: false,
            stencil_clear_value: 0,
            stencil_read_only: true,
        });

        device
            .queue()
            .submit_copies(&[HalCopy::RenderPass(pass)])
            .expect("render pass must attach and clear depth layer 1");
    }

    #[test]
    fn create_render_pipeline_accepts_unorm8x4_bgra_vertex_attribute() {
        let Some(device) = gles_device_or_skip("GLES Unorm8x4Bgra vertex attribute test") else {
            return;
        };

        let color_texture = render_attachment_texture(&device, crate::HalTextureFormat::Rgba8Unorm);
        let pipeline = device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    vertex: "#version 310 es\n\
                             precision mediump float;\n\
                             layout(location = 0) in vec2 position;\n\
                             layout(location = 1) in vec4 color;\n\
                             out vec4 vertex_color;\n\
                             void main() {\n\
                                 gl_Position = vec4(position, 0.0, 1.0);\n\
                                 vertex_color = color;\n\
                             }\n"
                    .to_owned(),
                    fragment: Some(
                        "#version 310 es\n\
                         precision mediump float;\n\
                         in vec4 vertex_color;\n\
                         layout(location = 0) out vec4 frag_color;\n\
                         void main() { frag_color = vertex_color; }\n"
                            .to_owned(),
                    ),
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: Vec::new(),
                    texture_metadata_ubo_binding: None,
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: vec![Some(HalColorTargetState {
                        format: crate::HalTextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 0xf,
                    })],
                    depth_stencil: None,
                    vertex_buffers: vec![crate::HalVertexBufferLayout {
                        slot: 0,
                        array_stride: 12,
                        step_mode: HalVertexStepMode::Vertex,
                        attributes: vec![
                            crate::HalVertexAttribute {
                                format: crate::HalVertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                                metal_buffer_index: 0,
                            },
                            crate::HalVertexAttribute {
                                format: crate::HalVertexFormat::Unorm8x4Bgra,
                                offset: 8,
                                shader_location: 1,
                                metal_buffer_index: 0,
                            },
                        ],
                    }],
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[],
            )
            .expect("GLES render pipeline must accept Unorm8x4Bgra vertex attributes");

        if !device.inner_clone().supports_vertex_array_bgra() {
            eprintln!(
                "GLES Unorm8x4Bgra vertex attribute test ran acceptance path; device reports no GL_EXT/ARB_vertex_array_bgra support, skipping pixel assertion"
            );
            return;
        }
        eprintln!(
            "GLES Unorm8x4Bgra vertex attribute test running GL_BGRA vertex fetch pixel assertion"
        );

        let vertices = [(-1.0f32, -1.0f32), (3.0f32, -1.0f32), (-1.0f32, 3.0f32)];
        let mut vertex_bytes = Vec::with_capacity(36);
        for (x, y) in vertices {
            vertex_bytes.extend_from_slice(&x.to_ne_bytes());
            vertex_bytes.extend_from_slice(&y.to_ne_bytes());
            vertex_bytes.extend_from_slice(&[255, 0, 64, 255]);
        }
        let vertex_buffer = device
            .create_buffer(
                vertex_bytes.len() as u64,
                crate::HalBufferUsage {
                    vertex: true,
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES BGRA vertex buffer creation must succeed");
        vertex_buffer
            .write(0, &vertex_bytes)
            .expect("writing BGRA vertex data must succeed");

        let mut pass = render_pass(vec![Some(color_target_for(
            color_texture.clone(),
            crate::HalTextureFormat::Rgba8Unorm,
            [0.0, 0.0, 0.0, 0.0],
        ))]);
        pass.pipeline = Some(HalRenderPipeline::Gles(pipeline));
        pass.vertex_buffers = vec![crate::HalBoundBuffer {
            group: 0,
            binding: 0,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: None,
            buffer: HalBuffer::Gles(vertex_buffer),
            offset: 0,
            size: vertex_bytes.len() as u64,
        }];
        pass.draw = Some(HalDraw::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });

        device
            .queue()
            .submit_copies(&[HalCopy::RenderPass(pass)])
            .expect("GLES BGRA vertex attribute draw must succeed");

        assert_eq!(
            read_rgba8_1x1(&device, color_texture),
            [64, 0, 255, 255],
            "GL_BGRA vertex fetch must present BGRA bytes as RGBA shader components"
        );
    }

    #[test]
    fn submit_render_pass_clears_uint_color_attachment_with_integer_clear() {
        // T-G7: integer color attachments cannot be cleared through
        // `glClearColor`+`glClear` (undefined for integer formats); the clear
        // path must use `glClearBufferuiv`/`glClearBufferiv`. A clear-only
        // pass on an R32Uint attachment followed by a texture-to-buffer copy
        // must read back the integer clear value.
        let instance = match super::super::instance::GlesInstance::new() {
            Ok(instance) => instance,
            Err(error) => {
                eprintln!("skipping GLES uint-clear test; backend unavailable: {error:?}");
                return;
            }
        };
        let Some(adapter) = instance.enumerate_adapters().into_iter().next() else {
            eprintln!("skipping GLES uint-clear test; no adapter available");
            return;
        };
        let device = match adapter.create_device() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping GLES uint-clear test; device unavailable: {error:?}");
                return;
            }
        };

        let color_texture = device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::R32Uint,
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: true,
                    copy_dst: false,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("GLES R32Uint texture creation must succeed");

        let readback = device
            .create_buffer(
                4,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES readback buffer creation must succeed");

        let mut pass = render_pass(vec![Some(crate::HalRenderColorTarget {
            texture: HalTexture::Gles(color_texture.clone()),
            view_format: crate::HalTextureFormat::R32Uint,
            resolve_target: None,
            resolve_view_format: None,
            mip_level: 0,
            array_layer: 0,
            depth_slice: 0,
            resolve_mip_level: 0,
            resolve_array_layer: 0,
            load_op: HalRenderLoadOp::Clear,
            store: true,
            clear_color: [5.0, 0.0, 0.0, 1.0],
        })]);
        pass.pipeline = None;

        device
            .queue()
            .submit_copies(&[
                HalCopy::RenderPass(pass),
                HalCopy::TextureToBuffer(HalBufferTextureCopy {
                    buffer: HalBuffer::Gles(readback.clone()),
                    buffer_layout: crate::HalBufferTextureLayout {
                        offset: 0,
                        bytes_per_row: 4,
                        rows_per_image: 1,
                    },
                    texture: HalTexture::Gles(color_texture),
                    format: crate::HalTextureFormat::R32Uint,
                    aspect: crate::HalTextureAspect::All,
                    mip_level: 0,
                    origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                    extent: crate::HalExtent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                }),
            ])
            .expect("clear-only pass on an R32Uint attachment plus readback must succeed");

        let bytes = readback
            .read(0, 4)
            .expect("reading back the cleared texel must succeed");
        let value = u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(
            value, 5,
            "R32Uint attachment must hold the integer clear value"
        );
    }

    #[test]
    fn submit_render_pass_reads_disjoint_texture_metadata_per_stage() {
        // Cross-stage texture-metadata UBO regression (P1-surfaced regression of
        // c06e516): the vertex and fragment stages each query metadata
        // (textureNumLevels) on a DIFFERENT texture. yawgpu-core merges both
        // stages' metadata slots into one UBO, keyed by offset. The fixed shim
        // makes each offset a function of the resolved texture binding, so the
        // two stages get DISJOINT offsets (vtex -> offset 0, ftex -> offset 1)
        // and each stage reads its own texture's level count. If the offsets
        // collided (the pre-fix per-stage-from-0 packing), both stages would
        // read the same UBO slot and the two channels would be equal -- this
        // test fails in that case. This mirrors Dawn's per-pipeline
        // EmulatedTextureBuiltinRegistrar. A Noop backend cannot catch it (it is
        // a real GLES UBO layout bug), hence a real-EGL test.
        let Some(device) = gles_device_or_skip("GLES cross-stage metadata test") else {
            return;
        };

        // vtex has 3 mip levels (queried by the vertex stage); ftex has 1
        // (queried by the fragment stage). Distinct counts prove each stage
        // reads its own texture rather than a shared, colliding slot.
        let metadata_texture = |mip_level_count: u32| {
            device
                .create_texture(&crate::HalTextureDescriptor {
                    dimension: crate::HalTextureDimension::D2,
                    format: crate::HalTextureFormat::Rgba8Unorm,
                    width: 4,
                    height: 4,
                    depth_or_array_layers: 1,
                    mip_level_count,
                    sample_count: 1,
                    usage: crate::HalTextureUsage {
                        copy_src: false,
                        copy_dst: false,
                        texture_binding: true,
                        storage_binding: false,
                        render_attachment: false,
                        transient: false,
                    },
                })
                .expect("GLES metadata texture creation must succeed")
        };
        let vtex = metadata_texture(3);
        let ftex = metadata_texture(1);

        let color_texture = device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Rgba8Unorm,
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: true,
                    copy_dst: false,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("GLES color attachment creation must succeed");
        let readback = device
            .create_buffer(
                4,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES readback buffer creation must succeed");

        // Two distinct uniform blocks share metadata UBO binding 0 (as Tint
        // emits: v_/f_ prefixed block names, same binding). The vertex stage
        // reads slot 0 (vtex) and forwards it; the fragment stage reads slot 1
        // (ftex). Output encodes both level counts in separate channels.
        let vertex = "#version 310 es\n\
             layout(binding = 0, std140) uniform v_TintTextureUniformData_ubo {\n\
               uvec4 metadata[1];\n\
             } v;\n\
             flat out uint v_vertex_levels;\n\
             void main() {\n\
               v_vertex_levels = v.metadata[0u / 4u][0u % 4u];\n\
               float x = float((gl_VertexID & 1) << 2) - 1.0;\n\
               float y = float((gl_VertexID & 2) << 1) - 1.0;\n\
               gl_Position = vec4(x, y, 0.0, 1.0);\n\
             }\n"
            .to_owned();
        let fragment = "#version 310 es\n\
             precision highp float;\n\
             precision highp int;\n\
             layout(binding = 0, std140) uniform f_TintTextureUniformData_ubo {\n\
               uvec4 metadata[1];\n\
             } v;\n\
             flat in uint v_vertex_levels;\n\
             layout(location = 0) out vec4 frag_color;\n\
             void main() {\n\
               uint fragment_levels = v.metadata[1u / 4u][1u % 4u];\n\
               frag_color = vec4(float(v_vertex_levels) / 255.0,\n\
                                 float(fragment_levels) / 255.0, 0.0, 1.0);\n\
             }\n"
            .to_owned();

        let pipeline = device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    vertex,
                    fragment: Some(fragment),
                    combined_samplers: Vec::new(),
                    // Merged, disjoint slots exactly as core produces post-fix.
                    texture_metadata_slots: vec![
                        HalTextureMetadataSlot {
                            offset: 0,
                            texture_group: 0,
                            texture_binding: 0,
                        },
                        HalTextureMetadataSlot {
                            offset: 1,
                            texture_group: 0,
                            texture_binding: 1,
                        },
                    ],
                    binding_remaps: Vec::new(),
                    texture_metadata_ubo_binding: Some(0),
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: vec![Some(HalColorTargetState {
                        format: crate::HalTextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 0xf,
                    })],
                    depth_stencil: None,
                    vertex_buffers: Vec::new(),
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[
                    HalDescriptorBinding {
                        group: 0,
                        binding: 0,
                        kind: HalDescriptorBindingKind::Texture,
                    },
                    HalDescriptorBinding {
                        group: 0,
                        binding: 1,
                        kind: HalDescriptorBindingKind::Texture,
                    },
                ],
            )
            .expect("GLES cross-stage metadata pipeline creation must succeed");

        let bound_metadata_texture =
            |texture: super::super::texture::GlesTexture, binding: u32, mip_level_count: u32| {
                HalBoundTexture {
                    group: 0,
                    binding,
                    metal_index: 0,
                    vertex_metal_index: None,
                    fragment_metal_index: None,
                    texture: HalTexture::Gles(texture),
                    format: crate::HalTextureFormat::Rgba8Unorm,
                    dimension: HalTextureViewDimension::D2,
                    base_mip_level: 0,
                    mip_level_count,
                    base_array_layer: 0,
                    array_layer_count: 1,
                    aspect: crate::HalTextureAspect::All,
                    swizzle: crate::HalTextureComponentSwizzle::default(),
                    storage_access: None,
                }
            };

        let mut pass = render_pass(vec![Some(crate::HalRenderColorTarget {
            texture: HalTexture::Gles(color_texture.clone()),
            view_format: crate::HalTextureFormat::Rgba8Unorm,
            resolve_target: None,
            resolve_view_format: None,
            mip_level: 0,
            array_layer: 0,
            depth_slice: 0,
            resolve_mip_level: 0,
            resolve_array_layer: 0,
            load_op: HalRenderLoadOp::Clear,
            store: true,
            clear_color: [0.0, 0.0, 0.0, 1.0],
        })]);
        pass.pipeline = Some(HalRenderPipeline::Gles(pipeline));
        pass.bind_textures = vec![
            bound_metadata_texture(vtex, 0, 3),
            bound_metadata_texture(ftex, 1, 1),
        ];
        pass.draw = Some(HalDraw::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });

        device
            .queue()
            .submit_copies(&[
                HalCopy::RenderPass(pass),
                HalCopy::TextureToBuffer(HalBufferTextureCopy {
                    buffer: HalBuffer::Gles(readback.clone()),
                    buffer_layout: crate::HalBufferTextureLayout {
                        offset: 0,
                        bytes_per_row: 4,
                        rows_per_image: 1,
                    },
                    texture: HalTexture::Gles(color_texture),
                    format: crate::HalTextureFormat::Rgba8Unorm,
                    aspect: crate::HalTextureAspect::All,
                    mip_level: 0,
                    origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                    extent: crate::HalExtent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                }),
            ])
            .expect("cross-stage metadata render plus readback must succeed");

        let bytes = readback
            .read(0, 4)
            .expect("reading back the cross-stage metadata texel must succeed");
        // R = vertex stage's texture level count (3), G = fragment stage's (1).
        // A collision (shared offset) would make R == G; the fix keeps them
        // disjoint so each stage observes its own texture.
        assert_eq!(
            bytes[0], 3,
            "vertex stage must read vtex's 3 mip levels; got {bytes:?}"
        );
        assert_eq!(
            bytes[1], 1,
            "fragment stage must read ftex's 1 mip level; got {bytes:?}"
        );
    }

    /// Creates a 2x2 Rgba8Unorm texture with `depth_or_array_layers` slices of
    /// the given dimension, usable as both a copy source and destination.
    fn rgba8_copy_texture(
        device: &super::super::device::GlesDevice,
        dimension: crate::HalTextureDimension,
        depth_or_array_layers: u32,
    ) -> super::super::texture::GlesTexture {
        device
            .create_texture(&crate::HalTextureDescriptor {
                dimension,
                format: crate::HalTextureFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                depth_or_array_layers,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: true,
                    copy_dst: true,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: false,
                    transient: false,
                },
            })
            .expect("GLES Rgba8Unorm copy texture creation must succeed")
    }

    fn rgba8_copy_texture_2d(
        device: &super::super::device::GlesDevice,
        width: u32,
        height: u32,
        mip_level_count: u32,
    ) -> super::super::texture::GlesTexture {
        device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Rgba8Unorm,
                width,
                height,
                depth_or_array_layers: 1,
                mip_level_count,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: true,
                    copy_dst: true,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: false,
                    transient: false,
                },
            })
            .expect("GLES Rgba8Unorm 2D copy texture creation must succeed")
    }

    fn rgba8_tight_layout(width: u32, height: u32) -> crate::HalBufferTextureLayout {
        crate::HalBufferTextureLayout {
            offset: 0,
            bytes_per_row: width * 4,
            rows_per_image: height,
        }
    }

    fn upload_rgba8_region(
        device: &super::super::device::GlesDevice,
        texture: &super::super::texture::GlesTexture,
        mip_level: u32,
        origin: crate::HalOrigin3d,
        width: u32,
        height: u32,
        bytes: &[u8],
    ) {
        assert_eq!(bytes.len(), (width * height * 4) as usize);
        let upload = device
            .create_buffer(
                bytes.len() as u64,
                crate::HalBufferUsage {
                    copy_src: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES upload buffer creation must succeed");
        upload
            .write(0, bytes)
            .expect("writing the upload buffer must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::BufferToTexture(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(upload),
                buffer_layout: rgba8_tight_layout(width, height),
                texture: HalTexture::Gles(texture.clone()),
                format: crate::HalTextureFormat::Rgba8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level,
                origin,
                extent: crate::HalExtent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("buffer-to-texture copy of RGBA8 region must succeed");
    }

    fn read_back_rgba8_region(
        device: &super::super::device::GlesDevice,
        texture: &super::super::texture::GlesTexture,
        mip_level: u32,
        origin: crate::HalOrigin3d,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let byte_count = u64::from(width * height * 4);
        let readback = device
            .create_buffer(
                byte_count,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES readback buffer creation must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(readback.clone()),
                buffer_layout: rgba8_tight_layout(width, height),
                texture: HalTexture::Gles(texture.clone()),
                format: crate::HalTextureFormat::Rgba8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level,
                origin,
                extent: crate::HalExtent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("texture-to-buffer copy of RGBA8 region must succeed");
        readback
            .read(0, byte_count)
            .expect("reading back the RGBA8 region must succeed")
    }

    fn rgba8_pattern(width: u32, height: u32, base: u8) -> Vec<u8> {
        let mut bytes = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let v = base.wrapping_add((y * width + x) as u8);
                bytes.extend_from_slice(&[v, v.wrapping_add(1), v.wrapping_add(2), 255]);
            }
        }
        bytes
    }

    fn copy_rgba8_texel(
        destination: &mut [u8],
        destination_width: u32,
        destination_origin: (u32, u32),
        source: &[u8],
        source_width: u32,
        source_origin: (u32, u32),
    ) {
        let (destination_x, destination_y) = destination_origin;
        let (source_x, source_y) = source_origin;
        let destination_offset = ((destination_y * destination_width + destination_x) * 4) as usize;
        let source_offset = ((source_y * source_width + source_x) * 4) as usize;
        destination[destination_offset..destination_offset + 4]
            .copy_from_slice(&source[source_offset..source_offset + 4]);
    }

    fn r8_copy_texture_2d(
        device: &super::super::device::GlesDevice,
        width: u32,
        height: u32,
    ) -> super::super::texture::GlesTexture {
        device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::R8Unorm,
                width,
                height,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: true,
                    copy_dst: true,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: false,
                    transient: false,
                },
            })
            .expect("GLES R8Unorm 2D copy texture creation must succeed")
    }

    fn r8_layout(width: u32, height: u32) -> crate::HalBufferTextureLayout {
        crate::HalBufferTextureLayout {
            offset: 0,
            bytes_per_row: width,
            rows_per_image: height,
        }
    }

    fn upload_r8_region(
        device: &super::super::device::GlesDevice,
        texture: &super::super::texture::GlesTexture,
        origin: crate::HalOrigin3d,
        width: u32,
        height: u32,
        bytes: &[u8],
    ) {
        assert_eq!(bytes.len(), (width * height) as usize);
        let upload = device
            .create_buffer(
                bytes.len() as u64,
                crate::HalBufferUsage {
                    copy_src: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES R8 upload buffer creation must succeed");
        upload
            .write(0, bytes)
            .expect("writing the R8 upload buffer must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::BufferToTexture(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(upload),
                buffer_layout: r8_layout(width, height),
                texture: HalTexture::Gles(texture.clone()),
                format: crate::HalTextureFormat::R8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin,
                extent: crate::HalExtent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("buffer-to-texture copy of R8 region must succeed");
    }

    fn read_back_r8_region(
        device: &super::super::device::GlesDevice,
        texture: &super::super::texture::GlesTexture,
        origin: crate::HalOrigin3d,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let byte_count = u64::from(width * height);
        let readback = device
            .create_buffer(
                byte_count,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES R8 readback buffer creation must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(readback.clone()),
                buffer_layout: r8_layout(width, height),
                texture: HalTexture::Gles(texture.clone()),
                format: crate::HalTextureFormat::R8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin,
                extent: crate::HalExtent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("texture-to-buffer copy of R8 region must succeed");
        readback
            .read(0, byte_count)
            .expect("reading back the R8 region must succeed")
    }

    fn byte_copy_texture_2d(
        device: &super::super::device::GlesDevice,
        format: crate::HalTextureFormat,
        width: u32,
        height: u32,
    ) -> super::super::texture::GlesTexture {
        device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format,
                width,
                height,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: true,
                    copy_dst: true,
                    texture_binding: true,
                    storage_binding: false,
                    render_attachment: false,
                    transient: false,
                },
            })
            .expect("GLES byte-format 2D copy texture creation must succeed")
    }

    fn byte_copy_texture(
        device: &super::super::device::GlesDevice,
        dimension: crate::HalTextureDimension,
        format: crate::HalTextureFormat,
        width: u32,
        height: u32,
        depth_or_array_layers: u32,
    ) -> super::super::texture::GlesTexture {
        device
            .create_texture(&crate::HalTextureDescriptor {
                dimension,
                format,
                width,
                height,
                depth_or_array_layers,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: true,
                    copy_dst: true,
                    texture_binding: true,
                    storage_binding: false,
                    render_attachment: false,
                    transient: false,
                },
            })
            .expect("GLES byte-format copy texture creation must succeed")
    }

    fn byte_layout(width: u32, height: u32, bytes_per_pixel: u32) -> crate::HalBufferTextureLayout {
        crate::HalBufferTextureLayout {
            offset: 0,
            bytes_per_row: width * bytes_per_pixel,
            rows_per_image: height,
        }
    }

    fn upload_byte_region(
        device: &super::super::device::GlesDevice,
        texture: &super::super::texture::GlesTexture,
        format: crate::HalTextureFormat,
        bytes_per_pixel: u32,
        width: u32,
        height: u32,
        bytes: &[u8],
    ) {
        assert_eq!(bytes.len(), (width * height * bytes_per_pixel) as usize);
        let upload = device
            .create_buffer(
                bytes.len() as u64,
                crate::HalBufferUsage {
                    copy_src: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES byte-format upload buffer creation must succeed");
        upload
            .write(0, bytes)
            .expect("writing the byte-format upload buffer must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::BufferToTexture(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(upload),
                buffer_layout: byte_layout(width, height, bytes_per_pixel),
                texture: HalTexture::Gles(texture.clone()),
                format,
                aspect: if format_has_depth_aspect(format) {
                    crate::HalTextureAspect::DepthOnly
                } else {
                    crate::HalTextureAspect::All
                },
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("buffer-to-texture copy of byte-format region must succeed");
    }

    fn read_back_byte_region(
        device: &super::super::device::GlesDevice,
        texture: &super::super::texture::GlesTexture,
        format: crate::HalTextureFormat,
        bytes_per_pixel: u32,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let byte_count = u64::from(width * height * bytes_per_pixel);
        let readback = device
            .create_buffer(
                byte_count,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES byte-format readback buffer creation must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(readback.clone()),
                buffer_layout: byte_layout(width, height, bytes_per_pixel),
                texture: HalTexture::Gles(texture.clone()),
                format,
                aspect: if format_has_depth_aspect(format) {
                    crate::HalTextureAspect::DepthOnly
                } else {
                    crate::HalTextureAspect::All
                },
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("texture-to-buffer copy of byte-format region must succeed");
        readback
            .read(0, byte_count)
            .expect("reading back the byte-format region must succeed")
    }

    #[derive(Clone, Copy, Debug)]
    struct ByteMatrixFormat {
        format: crate::HalTextureFormat,
        name: &'static str,
        bytes_per_pixel: u32,
    }

    #[derive(Debug)]
    struct ByteMatrixCase {
        format: ByteMatrixFormat,
        dimension: crate::HalTextureDimension,
        offset: u64,
        origin: crate::HalOrigin3d,
        width: u32,
        height: u32,
        layers: u32,
    }

    fn matrix_texel_bytes(case: &ByteMatrixCase) -> Vec<u8> {
        let texel_count = case.width * case.height * case.layers;
        let mut bytes = Vec::with_capacity((texel_count * case.format.bytes_per_pixel) as usize);
        for texel in 0..texel_count {
            match case.format.format {
                crate::HalTextureFormat::R16Float => {
                    let values = [0x0000u16, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500];
                    bytes.extend_from_slice(&values[texel as usize % values.len()].to_ne_bytes());
                }
                crate::HalTextureFormat::Rgba16Float => {
                    let values = [
                        0x0000u16, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700,
                    ];
                    for component in 0..4 {
                        let index = (texel as usize * 4 + component) % values.len();
                        bytes.extend_from_slice(&values[index].to_ne_bytes());
                    }
                }
                crate::HalTextureFormat::Rgb9e5Ufloat => {
                    let base = texel as f32 + 1.0;
                    let packed = pack_rgb9e5_reference([base * 0.125, base * 0.1875, base * 0.25]);
                    bytes.extend_from_slice(&packed.to_ne_bytes());
                }
                _ => {
                    for byte in 0..case.format.bytes_per_pixel {
                        let value =
                            1u8.wrapping_add((texel * case.format.bytes_per_pixel + byte) as u8);
                        bytes.push(value);
                    }
                }
            }
        }
        bytes
    }

    fn pack_rgb9e5_reference(value: [f32; 3]) -> u32 {
        let color = [
            value[0].clamp(0.0, 65408.0),
            value[1].clamp(0.0, 65408.0),
            value[2].clamp(0.0, 65408.0),
        ];
        let max_channel = color[0].max(color[1]).max(color[2]);
        if max_channel == 0.0 {
            return 0;
        }
        let exponent = max_channel.log2().floor().max(-16.0) + 1.0;
        let mut shared_exponent = (exponent + 15.0) as u32;
        let mut scale = 2.0f32.powf(exponent - 9.0);
        let max_mantissa = (max_channel / scale + 0.5).floor() as u32;
        if max_mantissa == 512 {
            shared_exponent += 1;
            scale *= 2.0;
        }
        let r = ((color[0] / scale + 0.5).floor() as u32).min(511);
        let g = ((color[1] / scale + 0.5).floor() as u32).min(511);
        let b = ((color[2] / scale + 0.5).floor() as u32).min(511);
        (shared_exponent << 27) | (b << 18) | (g << 9) | r
    }

    fn write_matrix_texel_rows(
        destination: &mut [u8],
        case: &ByteMatrixCase,
        bytes_per_row: u32,
        rows_per_image: u32,
        texels: &[u8],
    ) {
        let row_bytes = (case.width * case.format.bytes_per_pixel) as usize;
        let mut source_offset = 0usize;
        for layer in 0..case.layers {
            for row in 0..case.height {
                let destination_offset = case.offset as usize
                    + (layer * rows_per_image * bytes_per_row + row * bytes_per_row) as usize;
                destination[destination_offset..destination_offset + row_bytes]
                    .copy_from_slice(&texels[source_offset..source_offset + row_bytes]);
                source_offset += row_bytes;
            }
        }
    }

    fn first_matrix_mismatch(actual: &[u8], expected: &[u8]) -> Option<String> {
        actual
            .iter()
            .zip(expected)
            .enumerate()
            .find_map(|(index, (actual, expected))| {
                (actual != expected)
                    .then(|| format!("byte {index}: expected {expected}, got {actual}"))
            })
    }

    fn submit_matrix_buffer_to_texture(
        device: &super::super::device::GlesDevice,
        texture: &super::super::texture::GlesTexture,
        case: &ByteMatrixCase,
        bytes_per_row: u32,
        rows_per_image: u32,
        bytes: &[u8],
    ) -> Result<(), crate::HalError> {
        let upload = device.create_buffer(
            bytes.len() as u64,
            crate::HalBufferUsage {
                copy_src: true,
                ..crate::HalBufferUsage::default()
            },
        )?;
        upload.write(0, bytes)?;
        device
            .queue()
            .submit_copies(&[HalCopy::BufferToTexture(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(upload),
                buffer_layout: crate::HalBufferTextureLayout {
                    offset: case.offset,
                    bytes_per_row,
                    rows_per_image,
                },
                texture: HalTexture::Gles(texture.clone()),
                format: case.format.format,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: case.origin,
                extent: crate::HalExtent3d {
                    width: case.width,
                    height: case.height,
                    depth_or_array_layers: case.layers,
                },
            })])
    }

    fn submit_matrix_texture_to_buffer(
        device: &super::super::device::GlesDevice,
        texture: &super::super::texture::GlesTexture,
        case: &ByteMatrixCase,
        bytes_per_row: u32,
        rows_per_image: u32,
        byte_count: u64,
    ) -> Result<Vec<u8>, crate::HalError> {
        let readback = device.create_buffer(
            byte_count,
            crate::HalBufferUsage {
                copy_dst: true,
                ..crate::HalBufferUsage::default()
            },
        )?;
        readback.write(0, &vec![0xab; byte_count as usize])?;
        device
            .queue()
            .submit_copies(&[HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(readback.clone()),
                buffer_layout: crate::HalBufferTextureLayout {
                    offset: case.offset,
                    bytes_per_row,
                    rows_per_image,
                },
                texture: HalTexture::Gles(texture.clone()),
                format: case.format.format,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: case.origin,
                extent: crate::HalExtent3d {
                    width: case.width,
                    height: case.height,
                    depth_or_array_layers: case.layers,
                },
            })])?;
        readback.read(0, byte_count)
    }

    /// One 2x2 Rgba8Unorm slice is 16 bytes; layout used by the multi-slice
    /// copy tests below (`bytes_per_row` 8, `rows_per_image` 2).
    const RGBA8_SLICE_BYTES: u32 = 16;

    fn rgba8_slice_layout(offset: u64) -> crate::HalBufferTextureLayout {
        crate::HalBufferTextureLayout {
            offset,
            bytes_per_row: 8,
            rows_per_image: 2,
        }
    }

    /// Uploads `slice_count` slices of distinct bytes (slice `i` holds bytes
    /// `i*16..(i+1)*16`) into the texture via one buffer-to-texture copy and
    /// returns the uploaded bytes.
    fn upload_rgba8_slices(
        device: &super::super::device::GlesDevice,
        texture: &super::super::texture::GlesTexture,
        slice_count: u32,
    ) -> Vec<u8> {
        let byte_count = u64::from(slice_count * RGBA8_SLICE_BYTES);
        let bytes: Vec<u8> = (0..byte_count).map(|byte| byte as u8).collect();
        let upload = device
            .create_buffer(
                byte_count,
                crate::HalBufferUsage {
                    copy_src: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES upload buffer creation must succeed");
        upload
            .write(0, &bytes)
            .expect("writing the upload buffer must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::BufferToTexture(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(upload),
                buffer_layout: rgba8_slice_layout(0),
                texture: HalTexture::Gles(texture.clone()),
                format: crate::HalTextureFormat::Rgba8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: slice_count,
                },
            })])
            .expect("buffer-to-texture copy across multiple slices must succeed");
        bytes
    }

    /// Reads `slice_count` slices starting at slice `z` back into a buffer via
    /// one texture-to-buffer copy and returns the bytes.
    fn read_back_rgba8_slices(
        device: &super::super::device::GlesDevice,
        texture: &super::super::texture::GlesTexture,
        z: u32,
        slice_count: u32,
    ) -> Vec<u8> {
        let byte_count = u64::from(slice_count * RGBA8_SLICE_BYTES);
        let readback = device
            .create_buffer(
                byte_count,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES readback buffer creation must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(readback.clone()),
                buffer_layout: rgba8_slice_layout(0),
                texture: HalTexture::Gles(texture.clone()),
                format: crate::HalTextureFormat::Rgba8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: slice_count,
                },
            })])
            .expect("texture-to-buffer copy of texture slices must succeed");
        readback
            .read(0, byte_count)
            .expect("reading back the copied slices must succeed")
    }

    fn sampled_rgba8_texture(
        device: &super::super::device::GlesDevice,
        pixels: &[u8; 16],
    ) -> super::super::texture::GlesTexture {
        let texture = device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: false,
                    copy_dst: true,
                    texture_binding: true,
                    storage_binding: false,
                    render_attachment: false,
                    transient: false,
                },
            })
            .expect("GLES sampled texture creation must succeed");
        let upload = device
            .create_buffer(
                pixels.len() as u64,
                crate::HalBufferUsage {
                    copy_src: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES sampled-texture upload buffer creation must succeed");
        upload
            .write(0, pixels)
            .expect("writing sampled-texture pixels must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::BufferToTexture(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(upload),
                buffer_layout: rgba8_slice_layout(0),
                texture: HalTexture::Gles(texture.clone()),
                format: crate::HalTextureFormat::Rgba8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("uploading sampled texture pixels must succeed");
        texture
    }

    fn sampled_rgba8_array_texture(
        device: &super::super::device::GlesDevice,
        layer_colors: &[[u8; 4]],
    ) -> super::super::texture::GlesTexture {
        let texture = device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                depth_or_array_layers: layer_colors.len() as u32,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: false,
                    copy_dst: true,
                    texture_binding: true,
                    storage_binding: false,
                    render_attachment: false,
                    transient: false,
                },
            })
            .expect("GLES sampled array texture creation must succeed");
        let mut pixels = Vec::with_capacity(layer_colors.len() * RGBA8_SLICE_BYTES as usize);
        for color in layer_colors {
            for _ in 0..4 {
                pixels.extend_from_slice(color);
            }
        }
        let upload = device
            .create_buffer(
                pixels.len() as u64,
                crate::HalBufferUsage {
                    copy_src: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES sampled array texture upload buffer creation must succeed");
        upload
            .write(0, &pixels)
            .expect("writing sampled array texture pixels must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::BufferToTexture(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(upload),
                buffer_layout: rgba8_slice_layout(0),
                texture: HalTexture::Gles(texture.clone()),
                format: crate::HalTextureFormat::Rgba8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: layer_colors.len() as u32,
                },
            })])
            .expect("uploading sampled array texture pixels must succeed");
        texture
    }

    fn bound_sampled_texture(
        texture: super::super::texture::GlesTexture,
        binding: u32,
    ) -> HalBoundTexture {
        HalBoundTexture {
            group: 0,
            binding,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: Some(0),
            texture: HalTexture::Gles(texture),
            format: crate::HalTextureFormat::Rgba8Unorm,
            dimension: HalTextureViewDimension::D2,
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
            aspect: crate::HalTextureAspect::All,
            swizzle: crate::HalTextureComponentSwizzle::default(),
            storage_access: None,
        }
    }

    fn bound_sampled_texture_view(
        texture: super::super::texture::GlesTexture,
        binding: u32,
        dimension: HalTextureViewDimension,
        base_array_layer: u32,
        array_layer_count: u32,
    ) -> HalBoundTexture {
        HalBoundTexture {
            group: 0,
            binding,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: Some(0),
            texture: HalTexture::Gles(texture),
            format: crate::HalTextureFormat::Rgba8Unorm,
            dimension,
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer,
            array_layer_count,
            aspect: crate::HalTextureAspect::All,
            swizzle: crate::HalTextureComponentSwizzle::default(),
            storage_access: None,
        }
    }

    fn nearest_sampler_binding(
        device: &super::super::device::GlesDevice,
        binding: u32,
    ) -> HalBoundSampler {
        let sampler = device.create_sampler(&crate::HalSamplerDescriptor {
            address_mode_u: crate::HalAddressMode::ClampToEdge,
            address_mode_v: crate::HalAddressMode::ClampToEdge,
            address_mode_w: crate::HalAddressMode::ClampToEdge,
            mag_filter: crate::HalFilterMode::Nearest,
            min_filter: crate::HalFilterMode::Nearest,
            mipmap_filter: crate::HalMipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 32.0,
            compare: None,
            max_anisotropy: 1,
        });
        HalBoundSampler {
            group: 0,
            binding,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: Some(0),
            sampler: HalSampler::Gles(sampler),
        }
    }

    #[test]
    fn submit_render_pass_samples_texture_with_sampler_binding() {
        let Some(device) = gles_device_or_skip("GLES render sampled-texture test") else {
            return;
        };
        let pixels: [u8; 16] = [
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ];
        let sampled = sampled_rgba8_texture(&device, &pixels);
        let target = device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: true,
                    copy_dst: false,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("GLES 2x2 render target creation must succeed");
        let readback = device
            .create_buffer(
                16,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES render sampled-texture readback buffer creation must succeed");

        let pipeline = device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    vertex: "#version 310 es\n\
                             void main() {\n\
                                 vec2 pos = vec2(float((gl_VertexID & 1) << 2) - 1.0,\n\
                                                 float((gl_VertexID & 2) << 1) - 1.0);\n\
                                 gl_Position = vec4(pos, 0.0, 1.0);\n\
                             }\n"
                    .to_owned(),
                    fragment: Some(
                        "#version 310 es\n\
                         precision mediump float;\n\
                         uniform sampler2D u_tex;\n\
                         layout(location = 0) out vec4 frag_color;\n\
                         void main() {\n\
                             frag_color = texture(u_tex, gl_FragCoord.xy * 0.5);\n\
                         }\n"
                        .to_owned(),
                    ),
                    combined_samplers: vec![crate::HalCombinedSampler {
                        glsl_uniform_name: "u_tex".to_owned(),
                        texture_group: 0,
                        texture_binding: 1,
                        sampler_group: 0,
                        sampler_binding: 2,
                        uses_placeholder_sampler: false,
                    }],
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: Vec::new(),
                    texture_metadata_ubo_binding: None,
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: vec![Some(HalColorTargetState {
                        format: crate::HalTextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 0xf,
                    })],
                    depth_stencil: None,
                    vertex_buffers: Vec::new(),
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[
                    HalDescriptorBinding {
                        group: 0,
                        binding: 1,
                        kind: HalDescriptorBindingKind::Texture,
                    },
                    HalDescriptorBinding {
                        group: 0,
                        binding: 2,
                        kind: HalDescriptorBindingKind::Sampler,
                    },
                ],
            )
            .expect("GLES sampled-texture render pipeline creation must succeed");

        let mut pass = render_pass(vec![Some(color_target_for(
            target.clone(),
            crate::HalTextureFormat::Rgba8Unorm,
            [0.0, 0.0, 0.0, 0.0],
        ))]);
        pass.pipeline = Some(HalRenderPipeline::Gles(pipeline));
        pass.bind_textures = vec![bound_sampled_texture(sampled, 1)];
        pass.bind_samplers = vec![nearest_sampler_binding(&device, 2)];
        pass.draw = Some(HalDraw::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });

        device
            .queue()
            .submit_copies(&[
                HalCopy::RenderPass(pass),
                HalCopy::TextureToBuffer(HalBufferTextureCopy {
                    buffer: HalBuffer::Gles(readback.clone()),
                    buffer_layout: rgba8_slice_layout(0),
                    texture: HalTexture::Gles(target),
                    format: crate::HalTextureFormat::Rgba8Unorm,
                    aspect: crate::HalTextureAspect::All,
                    mip_level: 0,
                    origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                    extent: crate::HalExtent3d {
                        width: 2,
                        height: 2,
                        depth_or_array_layers: 1,
                    },
                }),
            ])
            .expect("render pass sampling a texture plus readback must succeed");

        assert_eq!(
            readback
                .read(0, 16)
                .expect("reading back the sampled render target must succeed"),
            pixels,
            "each 2x2 output pixel must sample the matching source texel"
        );
    }

    // P2 depth raw-read (hardware verification for the shim IR transform). A
    // `texture_depth_2d` sampled by a non-comparison builtin now lowers to a
    // plain `sampler2D` + `textureLod(...).x` (see the yawgpu-tint
    // generate_glsl_depth_* tests). This test proves the hardware side of that
    // contract on crocus: a depth texture bound to a `sampler2D` through a
    // non-comparison sampler returns the RAW stored depth, not a 0/1
    // shadow-compare result. We clear a depth texture to 0.5, sample it, and
    // assert the output is mid-range (~128) rather than 0 or 255.
    #[test]
    fn submit_render_pass_samples_depth_texture_raw_not_shadow() {
        let Some(device) = gles_device_or_skip("GLES depth raw-sample test") else {
            return;
        };

        let depth = device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Depth32Float,
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: false,
                    copy_dst: false,
                    texture_binding: true,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("GLES depth texture creation must succeed");
        let scratch_color = render_attachment_texture(&device, crate::HalTextureFormat::Rgba8Unorm);
        let output = render_attachment_texture(&device, crate::HalTextureFormat::Rgba8Unorm);

        // Pass 1: clear the depth texture to 0.5 and store it.
        let mut clear_pass = render_pass(vec![Some(color_target_for(
            scratch_color,
            crate::HalTextureFormat::Rgba8Unorm,
            [0.0, 0.0, 0.0, 1.0],
        ))]);
        clear_pass.depth_stencil_attachment = Some(crate::HalRenderDepthStencilAttachment {
            texture: HalTexture::Gles(depth.clone()),
            format: crate::HalTextureFormat::Depth32Float,
            mip_level: 0,
            array_layer: 0,
            depth_load_op: HalRenderLoadOp::Clear,
            depth_store: true,
            depth_clear_value: 0.5,
            depth_read_only: false,
            stencil_load_op: HalRenderLoadOp::Load,
            stencil_store: false,
            stencil_clear_value: 0,
            stencil_read_only: true,
        });
        device
            .queue()
            .submit_copies(&[HalCopy::RenderPass(clear_pass)])
            .expect("clearing the depth texture to 0.5 must succeed");

        // Pass 2: sample the depth texture as a plain `sampler2D` (the shim's
        // lowering) and write the raw depth into the color output.
        let pipeline = device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    vertex: "#version 310 es\n\
                             void main() {\n\
                                 vec2 pos = vec2(float((gl_VertexID & 1) << 2) - 1.0,\n\
                                                 float((gl_VertexID & 2) << 1) - 1.0);\n\
                                 gl_Position = vec4(pos, 0.0, 1.0);\n\
                             }\n"
                    .to_owned(),
                    fragment: Some(
                        "#version 310 es\n\
                         precision highp float;\n\
                         uniform highp sampler2D u_depth;\n\
                         layout(location = 0) out vec4 frag_color;\n\
                         void main() {\n\
                             float d = textureLod(u_depth, vec2(0.5), 0.0).x;\n\
                             frag_color = vec4(d, d, d, 1.0);\n\
                         }\n"
                        .to_owned(),
                    ),
                    combined_samplers: vec![crate::HalCombinedSampler {
                        glsl_uniform_name: "u_depth".to_owned(),
                        texture_group: 0,
                        texture_binding: 1,
                        sampler_group: 0,
                        sampler_binding: 2,
                        uses_placeholder_sampler: false,
                    }],
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: Vec::new(),
                    texture_metadata_ubo_binding: None,
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: vec![Some(HalColorTargetState {
                        format: crate::HalTextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 0xf,
                    })],
                    depth_stencil: None,
                    vertex_buffers: Vec::new(),
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[
                    HalDescriptorBinding {
                        group: 0,
                        binding: 1,
                        kind: HalDescriptorBindingKind::Texture,
                    },
                    HalDescriptorBinding {
                        group: 0,
                        binding: 2,
                        kind: HalDescriptorBindingKind::Sampler,
                    },
                ],
            )
            .expect("GLES depth raw-sample render pipeline creation must succeed");

        let depth_binding = HalBoundTexture {
            group: 0,
            binding: 1,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: Some(0),
            texture: HalTexture::Gles(depth),
            format: crate::HalTextureFormat::Depth32Float,
            dimension: HalTextureViewDimension::D2,
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
            aspect: crate::HalTextureAspect::DepthOnly,
            swizzle: crate::HalTextureComponentSwizzle::default(),
            storage_access: None,
        };

        let mut pass = render_pass(vec![Some(color_target_for(
            output.clone(),
            crate::HalTextureFormat::Rgba8Unorm,
            [0.0, 0.0, 0.0, 0.0],
        ))]);
        pass.pipeline = Some(HalRenderPipeline::Gles(pipeline));
        pass.bind_textures = vec![depth_binding];
        pass.bind_samplers = vec![nearest_sampler_binding(&device, 2)];
        pass.draw = Some(HalDraw::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });
        device
            .queue()
            .submit_copies(&[HalCopy::RenderPass(pass)])
            .expect("sampling the depth texture as a raw sampler2D must succeed");

        let out = read_rgba8_1x1(&device, output);
        assert!(
            (100..=155).contains(&out[0]),
            "raw depth sample must be ~0.5 (got {out:?}); a ref-0 shadow compare would read 0 or 255"
        );
    }

    #[test]
    fn submit_render_pass_binds_uniform_and_sampler_texture_across_groups() {
        let Some(device) = gles_device_or_skip("GLES multi-group render binding test") else {
            return;
        };
        let sampled = sampled_rgba8_texture(
            &device,
            &[
                10, 20, 30, 255, 10, 20, 30, 255, 10, 20, 30, 255, 10, 20, 30, 255,
            ],
        );
        let target = device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: true,
                    copy_dst: false,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("GLES multi-group render target creation must succeed");
        let readback = device
            .create_buffer(
                16,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES multi-group render readback buffer creation must succeed");
        let uniform = device
            .create_buffer(
                16,
                crate::HalBufferUsage {
                    uniform: true,
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES multi-group render uniform buffer creation must succeed");
        let uniform_bytes = [2.0_f32, 3.0, 4.0, 1.0]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        uniform
            .write(0, &uniform_bytes)
            .expect("writing GLES multi-group render uniform data must succeed");

        let pipeline = device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    vertex: "#version 310 es\n\
                             void main() {\n\
                                 vec2 pos = vec2(float((gl_VertexID & 1) << 2) - 1.0,\n\
                                                 float((gl_VertexID & 2) << 1) - 1.0);\n\
                                 gl_Position = vec4(pos, 0.0, 1.0);\n\
                             }\n"
                    .to_owned(),
                    fragment: Some(
                        "#version 310 es\n\
                         precision mediump float;\n\
                         layout(std140, binding = 0) uniform Params { vec4 scale; } params;\n\
                         uniform sampler2D u_tex;\n\
                         layout(location = 0) out vec4 frag_color;\n\
                         void main() { frag_color = texture(u_tex, vec2(0.25)) * params.scale; }\n"
                            .to_owned(),
                    ),
                    combined_samplers: vec![crate::HalCombinedSampler {
                        glsl_uniform_name: "u_tex".to_owned(),
                        texture_group: 1,
                        texture_binding: 1,
                        sampler_group: 1,
                        sampler_binding: 2,
                        uses_placeholder_sampler: false,
                    }],
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: vec![crate::HalGlesBindingRemap::new(
                        0,
                        0,
                        crate::HalGlesBindingClass::UniformBuffer,
                        0,
                    )],
                    texture_metadata_ubo_binding: None,
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: vec![Some(HalColorTargetState {
                        format: crate::HalTextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 0xf,
                    })],
                    depth_stencil: None,
                    vertex_buffers: Vec::new(),
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[
                    HalDescriptorBinding {
                        group: 0,
                        binding: 0,
                        kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Uniform),
                    },
                    HalDescriptorBinding {
                        group: 1,
                        binding: 1,
                        kind: HalDescriptorBindingKind::Texture,
                    },
                    HalDescriptorBinding {
                        group: 1,
                        binding: 2,
                        kind: HalDescriptorBindingKind::Sampler,
                    },
                ],
            )
            .expect("GLES multi-group render pipeline creation must succeed");

        let mut texture_binding = bound_sampled_texture(sampled, 1);
        texture_binding.group = 1;
        let mut sampler_binding = nearest_sampler_binding(&device, 2);
        sampler_binding.group = 1;
        let mut pass = render_pass(vec![Some(color_target_for(
            target.clone(),
            crate::HalTextureFormat::Rgba8Unorm,
            [0.0, 0.0, 0.0, 0.0],
        ))]);
        pass.pipeline = Some(HalRenderPipeline::Gles(pipeline));
        pass.bind_buffers = vec![crate::HalBoundBuffer {
            group: 0,
            binding: 0,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: None,
            buffer: HalBuffer::Gles(uniform),
            offset: 0,
            size: 16,
        }];
        pass.bind_textures = vec![texture_binding];
        pass.bind_samplers = vec![sampler_binding];
        pass.draw = Some(HalDraw::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });

        device
            .queue()
            .submit_copies(&[
                HalCopy::RenderPass(pass),
                HalCopy::TextureToBuffer(HalBufferTextureCopy {
                    buffer: HalBuffer::Gles(readback.clone()),
                    buffer_layout: rgba8_slice_layout(0),
                    texture: HalTexture::Gles(target),
                    format: crate::HalTextureFormat::Rgba8Unorm,
                    aspect: crate::HalTextureAspect::All,
                    mip_level: 0,
                    origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                    extent: crate::HalExtent3d {
                        width: 2,
                        height: 2,
                        depth_or_array_layers: 1,
                    },
                }),
            ])
            .expect("GLES multi-group render submit plus readback must succeed");

        assert_eq!(
            readback
                .read(0, 16)
                .expect("reading GLES multi-group render output must succeed"),
            [20, 60, 120, 255].repeat(4)
        );
    }

    #[test]
    fn submit_compute_pass_samples_texture_with_placeholder_sampler() {
        let Some(device) = gles_device_or_skip("GLES compute sampled-texture test") else {
            return;
        };
        let pixels: [u8; 16] = [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let sampled = sampled_rgba8_texture(&device, &pixels);
        let output = device
            .create_buffer(
                16 * 4,
                crate::HalBufferUsage {
                    storage: true,
                    copy_src: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES compute sampled-texture output buffer creation must succeed");
        let pipeline = device
            .create_compute_pipeline(
                crate::HalShaderSource::Glsl {
                    source: "#version 310 es\n\
                             precision highp float;\n\
                             precision highp int;\n\
                             layout(local_size_x = 4) in;\n\
                             uniform sampler2D u_tex;\n\
                             layout(std430, binding = 0) buffer Out { uvec4 values[4]; } out_buf;\n\
                             void main() {\n\
                                 uint i = gl_GlobalInvocationID.x;\n\
                                 ivec2 xy = ivec2(int(i & 1u), int(i >> 1u));\n\
                                 out_buf.values[i] = uvec4(texelFetch(u_tex, xy, 0) * 255.0 + 0.5);\n\
                             }\n"
                    .to_owned(),
                    stage: crate::HalShaderStage::Compute,
                    combined_samplers: vec![crate::HalCombinedSampler {
                        glsl_uniform_name: "u_tex".to_owned(),
                        texture_group: 0,
                        texture_binding: 1,
                        sampler_group: u32::MAX,
                        sampler_binding: u32::MAX,
                        uses_placeholder_sampler: true,
                    }],
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: vec![crate::HalGlesBindingRemap::new(
                        0,
                        0,
                        crate::HalGlesBindingClass::StorageBuffer,
                        0,
                    )],
                    texture_metadata_ubo_binding: None,
                },
                "main",
                (4, 1, 1),
                &[
                    HalDescriptorBinding {
                        group: 0,
                        binding: 0,
                        kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage),
                    },
                    HalDescriptorBinding {
                        group: 0,
                        binding: 1,
                        kind: HalDescriptorBindingKind::Texture,
                    },
                ],
            )
            .expect("GLES sampled-texture compute pipeline creation must succeed");

        device
            .queue()
            .submit_copies(&[HalCopy::ComputePass(crate::HalComputePass {
                pipeline: HalComputePipeline::Gles(pipeline),
                bind_buffers: vec![crate::HalBoundBuffer {
                    group: 0,
                    binding: 0,
                    metal_index: 0,
                    vertex_metal_index: None,
                    fragment_metal_index: None,
                    buffer: HalBuffer::Gles(output.clone()),
                    offset: 0,
                    size: 16 * 4,
                }],
                bind_textures: vec![bound_sampled_texture(sampled, 1)],
                bind_samplers: Vec::new(),
                bind_external_textures: Vec::new(),
                immediate_data: Vec::new(),
                dispatch: HalComputeDispatch::Direct {
                    workgroups: (1, 1, 1),
                },
            })])
            .expect("compute pass sampling a texture must succeed");

        let bytes = output
            .read(0, 16 * 4)
            .expect("reading compute sampled-texture output must succeed");
        let actual: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        let expected: Vec<u32> = pixels.iter().map(|value| u32::from(*value)).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn submit_compute_pass_samples_cube_view_from_2d_array_texture_view() {
        let Some(device) = gles_device_or_skip("GLES cube texture-view sampling test") else {
            return;
        };
        if !device.inner_clone().supports_texture_view() {
            eprintln!("skipping GLES cube texture-view sampling test; glTextureView unavailable");
            return;
        }
        let face_colors = [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
            [255, 0, 255, 255],
            [0, 255, 255, 255],
        ];
        let sampled = sampled_rgba8_array_texture(&device, &face_colors);
        let output = device
            .create_buffer(
                6 * 16,
                crate::HalBufferUsage {
                    storage: true,
                    copy_src: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES cube-view output buffer creation must succeed");
        let pipeline = device
            .create_compute_pipeline(
                crate::HalShaderSource::Glsl {
                    source: "#version 310 es\n\
                             precision highp float;\n\
                             precision highp int;\n\
                             layout(local_size_x = 6) in;\n\
                             uniform samplerCube u_tex;\n\
                             layout(std430, binding = 0) buffer Out { uvec4 values[6]; } out_buf;\n\
                             const vec3 dirs[6] = vec3[6](\n\
                                 vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0),\n\
                                 vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0),\n\
                                 vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0));\n\
                             void main() {\n\
                                 uint i = gl_GlobalInvocationID.x;\n\
                                 out_buf.values[i] = uvec4(textureLod(u_tex, dirs[i], 0.0) * 255.0 + 0.5);\n\
                             }\n"
                    .to_owned(),
                    stage: crate::HalShaderStage::Compute,
                    combined_samplers: vec![crate::HalCombinedSampler {
                        glsl_uniform_name: "u_tex".to_owned(),
                        texture_group: 0,
                        texture_binding: 1,
                        sampler_group: u32::MAX,
                        sampler_binding: u32::MAX,
                        uses_placeholder_sampler: true,
                    }],
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: vec![crate::HalGlesBindingRemap::new(
                        0,
                        0,
                        crate::HalGlesBindingClass::StorageBuffer,
                        0,
                    )],
                    texture_metadata_ubo_binding: None,
                },
                "main",
                (6, 1, 1),
                &[
                    HalDescriptorBinding {
                        group: 0,
                        binding: 0,
                        kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage),
                    },
                    HalDescriptorBinding {
                        group: 0,
                        binding: 1,
                        kind: HalDescriptorBindingKind::Texture,
                    },
                ],
            )
            .expect("GLES cube-view compute pipeline creation must succeed");

        device
            .queue()
            .submit_copies(&[HalCopy::ComputePass(crate::HalComputePass {
                pipeline: HalComputePipeline::Gles(pipeline),
                bind_buffers: vec![crate::HalBoundBuffer {
                    group: 0,
                    binding: 0,
                    metal_index: 0,
                    vertex_metal_index: None,
                    fragment_metal_index: None,
                    buffer: HalBuffer::Gles(output.clone()),
                    offset: 0,
                    size: 6 * 16,
                }],
                bind_textures: vec![bound_sampled_texture_view(
                    sampled,
                    1,
                    HalTextureViewDimension::Cube,
                    0,
                    6,
                )],
                bind_samplers: Vec::new(),
                bind_external_textures: Vec::new(),
                immediate_data: Vec::new(),
                dispatch: HalComputeDispatch::Direct {
                    workgroups: (1, 1, 1),
                },
            })])
            .expect("compute pass sampling a cube texture view must succeed");

        let bytes = output
            .read(0, 6 * 16)
            .expect("reading cube-view compute output must succeed");
        let actual: Vec<[u32; 4]> = bytes
            .chunks_exact(16)
            .map(|chunk| {
                [
                    u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                    u32::from_ne_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                    u32::from_ne_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
                    u32::from_ne_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
                ]
            })
            .collect();
        let expected: Vec<[u32; 4]> = face_colors
            .iter()
            .map(|color| color.map(u32::from))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn submit_compute_pass_samples_array_layer_subrange_texture_view() {
        let Some(device) = gles_device_or_skip("GLES array-layer texture-view sampling test")
        else {
            return;
        };
        if !device.inner_clone().supports_texture_view() {
            eprintln!(
                "skipping GLES array-layer texture-view sampling test; glTextureView unavailable"
            );
            return;
        }
        let layer_colors = [[10, 20, 30, 255], [40, 50, 60, 255], [70, 80, 90, 255]];
        let sampled = sampled_rgba8_array_texture(&device, &layer_colors);
        let output = device
            .create_buffer(
                2 * 16,
                crate::HalBufferUsage {
                    storage: true,
                    copy_src: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES array-view output buffer creation must succeed");
        let pipeline = device
            .create_compute_pipeline(
                crate::HalShaderSource::Glsl {
                    source: "#version 310 es\n\
                             precision highp float;\n\
                             precision highp int;\n\
                             layout(local_size_x = 2) in;\n\
                             uniform highp sampler2DArray u_tex;\n\
                             layout(std430, binding = 0) buffer Out { uvec4 values[2]; } out_buf;\n\
                             void main() {\n\
                                 uint i = gl_GlobalInvocationID.x;\n\
                                 out_buf.values[i] = uvec4(texelFetch(u_tex, ivec3(0, 0, int(i)), 0) * 255.0 + 0.5);\n\
                             }\n"
                    .to_owned(),
                    stage: crate::HalShaderStage::Compute,
                    combined_samplers: vec![crate::HalCombinedSampler {
                        glsl_uniform_name: "u_tex".to_owned(),
                        texture_group: 0,
                        texture_binding: 1,
                        sampler_group: u32::MAX,
                        sampler_binding: u32::MAX,
                        uses_placeholder_sampler: true,
                    }],
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: vec![crate::HalGlesBindingRemap::new(
                        0,
                        0,
                        crate::HalGlesBindingClass::StorageBuffer,
                        0,
                    )],
                    texture_metadata_ubo_binding: None,
                },
                "main",
                (2, 1, 1),
                &[
                    HalDescriptorBinding {
                        group: 0,
                        binding: 0,
                        kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage),
                    },
                    HalDescriptorBinding {
                        group: 0,
                        binding: 1,
                        kind: HalDescriptorBindingKind::Texture,
                    },
                ],
            )
            .expect("GLES array-view compute pipeline creation must succeed");

        device
            .queue()
            .submit_copies(&[HalCopy::ComputePass(crate::HalComputePass {
                pipeline: HalComputePipeline::Gles(pipeline),
                bind_buffers: vec![crate::HalBoundBuffer {
                    group: 0,
                    binding: 0,
                    metal_index: 0,
                    vertex_metal_index: None,
                    fragment_metal_index: None,
                    buffer: HalBuffer::Gles(output.clone()),
                    offset: 0,
                    size: 2 * 16,
                }],
                bind_textures: vec![bound_sampled_texture_view(
                    sampled,
                    1,
                    HalTextureViewDimension::D2Array,
                    1,
                    2,
                )],
                bind_samplers: Vec::new(),
                bind_external_textures: Vec::new(),
                immediate_data: Vec::new(),
                dispatch: HalComputeDispatch::Direct {
                    workgroups: (1, 1, 1),
                },
            })])
            .expect("compute pass sampling an array-layer texture view must succeed");

        let bytes = output
            .read(0, 2 * 16)
            .expect("reading array-view compute output must succeed");
        let actual: Vec<[u32; 4]> = bytes
            .chunks_exact(16)
            .map(|chunk| {
                [
                    u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                    u32::from_ne_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                    u32::from_ne_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
                    u32::from_ne_bytes([chunk[12], chunk[13], chunk[14], chunk[15]]),
                ]
            })
            .collect();
        let expected = vec![
            layer_colors[1].map(u32::from),
            layer_colors[2].map(u32::from),
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn submit_compute_pass_binds_buffers_and_textures_across_groups() {
        let Some(device) = gles_device_or_skip("GLES multi-group compute binding test") else {
            return;
        };
        let sampled = sampled_rgba8_texture(
            &device,
            &[10, 0, 0, 255, 10, 0, 0, 255, 10, 0, 0, 255, 10, 0, 0, 255],
        );
        let output = device
            .create_buffer(
                16,
                crate::HalBufferUsage {
                    storage: true,
                    copy_src: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES multi-group output buffer creation must succeed");
        let uniform = device
            .create_buffer(
                16,
                crate::HalBufferUsage {
                    uniform: true,
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES multi-group uniform buffer creation must succeed");
        let mut uniform_bytes = vec![0_u8; 16];
        uniform_bytes[..4].copy_from_slice(&7_u32.to_ne_bytes());
        uniform
            .write(0, &uniform_bytes)
            .expect("writing GLES multi-group uniform data must succeed");

        let pipeline = device
            .create_compute_pipeline(
                crate::HalShaderSource::Glsl {
                    source: "#version 310 es\n\
                             precision highp float;\n\
                             precision highp int;\n\
                             layout(local_size_x = 1) in;\n\
                             layout(std430, binding = 0) buffer Out { uint value; } out_buf;\n\
                             layout(std140, binding = 0) uniform Params { uint addend; } params;\n\
                             uniform sampler2D u_tex;\n\
                             void main() {\n\
                                 uint texel = uint(texelFetch(u_tex, ivec2(0, 0), 0).r * 255.0 + 0.5);\n\
                                 out_buf.value = params.addend + texel;\n\
                             }\n"
                    .to_owned(),
                    stage: crate::HalShaderStage::Compute,
                    combined_samplers: vec![crate::HalCombinedSampler {
                        glsl_uniform_name: "u_tex".to_owned(),
                        texture_group: 1,
                        texture_binding: 2,
                        sampler_group: u32::MAX,
                        sampler_binding: u32::MAX,
                        uses_placeholder_sampler: true,
                    }],
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: vec![
                        crate::HalGlesBindingRemap::new(
                            1,
                            1,
                            crate::HalGlesBindingClass::UniformBuffer,
                            0,
                        ),
                        crate::HalGlesBindingRemap::new(
                            0,
                            0,
                            crate::HalGlesBindingClass::StorageBuffer,
                            0,
                        ),
                    ],
                    texture_metadata_ubo_binding: None,
                },
                "main",
                (1, 1, 1),
                &[
                    HalDescriptorBinding {
                        group: 0,
                        binding: 0,
                        kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage),
                    },
                    HalDescriptorBinding {
                        group: 1,
                        binding: 1,
                        kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Uniform),
                    },
                    HalDescriptorBinding {
                        group: 1,
                        binding: 2,
                        kind: HalDescriptorBindingKind::Texture,
                    },
                ],
            )
            .expect("GLES multi-group compute pipeline creation must succeed");
        let mut texture_binding = bound_sampled_texture(sampled, 2);
        texture_binding.group = 1;

        device
            .queue()
            .submit_copies(&[HalCopy::ComputePass(crate::HalComputePass {
                pipeline: HalComputePipeline::Gles(pipeline),
                bind_buffers: vec![
                    crate::HalBoundBuffer {
                        group: 0,
                        binding: 0,
                        metal_index: 0,
                        vertex_metal_index: None,
                        fragment_metal_index: None,
                        buffer: HalBuffer::Gles(output.clone()),
                        offset: 0,
                        size: 16,
                    },
                    crate::HalBoundBuffer {
                        group: 1,
                        binding: 1,
                        metal_index: 0,
                        vertex_metal_index: None,
                        fragment_metal_index: None,
                        buffer: HalBuffer::Gles(uniform),
                        offset: 0,
                        size: 16,
                    },
                ],
                bind_textures: vec![texture_binding],
                bind_samplers: Vec::new(),
                bind_external_textures: Vec::new(),
                immediate_data: Vec::new(),
                dispatch: HalComputeDispatch::Direct {
                    workgroups: (1, 1, 1),
                },
            })])
            .expect("GLES multi-group compute submit must succeed");

        let bytes = output
            .read(0, 4)
            .expect("reading GLES multi-group compute output must succeed");
        let bytes: [u8; 4] = bytes
            .try_into()
            .expect("GLES multi-group compute output must be one u32");
        assert_eq!(u32::from_ne_bytes(bytes), 17);
    }

    #[test]
    fn submit_copies_round_trips_2d_array_texture_layers() {
        // T-G9: buffer-to-texture / texture-to-buffer copies on a 2D-array
        // texture used to fail with "only 2D texture copies are supported on
        // GLES (P15.3)". A 3-layer round trip must preserve each layer's
        // distinct bytes.
        let Some(device) = gles_device_or_skip("GLES 2D-array copy round-trip test") else {
            return;
        };

        let texture = rgba8_copy_texture(&device, crate::HalTextureDimension::D2, 3);
        let bytes = upload_rgba8_slices(&device, &texture, 3);

        for layer in 0..3u32 {
            let expected = &bytes
                [(layer * RGBA8_SLICE_BYTES) as usize..((layer + 1) * RGBA8_SLICE_BYTES) as usize];
            assert_eq!(
                read_back_rgba8_slices(&device, &texture, layer, 1),
                expected,
                "layer {layer} must round-trip its distinct bytes"
            );
        }
    }

    #[test]
    fn submit_copies_round_trips_3d_texture_slices() {
        // T-G9: a 2x2x3 3D texture exercises `glTexSubImage3D` with depth > 1
        // on upload and the per-slice `glFramebufferTextureLayer` readback
        // with depth > 1 on the way back.
        let Some(device) = gles_device_or_skip("GLES 3D-texture copy round-trip test") else {
            return;
        };

        let texture = rgba8_copy_texture(&device, crate::HalTextureDimension::D3, 3);
        let bytes = upload_rgba8_slices(&device, &texture, 3);

        assert_eq!(
            read_back_rgba8_slices(&device, &texture, 0, 3),
            bytes,
            "all 3 depth slices must round-trip in one texture-to-buffer copy"
        );
    }

    #[test]
    fn submit_copies_texture_to_buffer_preserves_2d_padding_bytes() {
        let Some(device) = gles_device_or_skip("GLES 2D texture-to-buffer padding test") else {
            return;
        };

        let texture = rgba8_copy_texture(&device, crate::HalTextureDimension::D2, 1);
        let pixels = upload_rgba8_slices(&device, &texture, 1);
        let readback_size = 276u64;
        let readback = device
            .create_buffer(
                readback_size,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES padded readback buffer creation must succeed");
        readback
            .write(0, &vec![0xAB; readback_size as usize])
            .expect("pre-filling the padded readback buffer must succeed");

        device
            .queue()
            .submit_copies(&[HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(readback.clone()),
                buffer_layout: crate::HalBufferTextureLayout {
                    offset: 4,
                    bytes_per_row: 256,
                    rows_per_image: 2,
                },
                texture: HalTexture::Gles(texture),
                format: crate::HalTextureFormat::Rgba8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("padded 2D texture-to-buffer copy must succeed");

        let actual = readback
            .read(0, readback_size)
            .expect("reading the padded readback buffer must succeed");
        let mut expected = vec![0xAB; readback_size as usize];
        expected[4..12].copy_from_slice(&pixels[0..8]);
        expected[260..268].copy_from_slice(&pixels[8..16]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn submit_copies_texture_to_buffer_preserves_2d_array_padding_bytes() {
        let Some(device) = gles_device_or_skip("GLES 2D-array texture-to-buffer padding test")
        else {
            return;
        };

        let texture = rgba8_copy_texture(&device, crate::HalTextureDimension::D2, 2);
        let pixels = upload_rgba8_slices(&device, &texture, 2);
        let readback_size = 1300u64;
        let readback = device
            .create_buffer(
                readback_size,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES padded array readback buffer creation must succeed");
        readback
            .write(0, &vec![0xAB; readback_size as usize])
            .expect("pre-filling the padded array readback buffer must succeed");

        device
            .queue()
            .submit_copies(&[HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(readback.clone()),
                buffer_layout: crate::HalBufferTextureLayout {
                    offset: 4,
                    bytes_per_row: 256,
                    rows_per_image: 4,
                },
                texture: HalTexture::Gles(texture),
                format: crate::HalTextureFormat::Rgba8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 2,
                },
            })])
            .expect("padded 2D-array texture-to-buffer copy must succeed");

        let actual = readback
            .read(0, readback_size)
            .expect("reading the padded array readback buffer must succeed");
        let mut expected = vec![0xAB; readback_size as usize];
        expected[4..12].copy_from_slice(&pixels[0..8]);
        expected[260..268].copy_from_slice(&pixels[8..16]);
        expected[1028..1036].copy_from_slice(&pixels[16..24]);
        expected[1284..1292].copy_from_slice(&pixels[24..32]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn submit_copies_buffer_texture_respects_nonzero_2d_origins() {
        let Some(device) = gles_device_or_skip("GLES nonzero B2T/T2B origin test") else {
            return;
        };

        let texture = rgba8_copy_texture_2d(&device, 4, 4, 1);
        let zeros = vec![0; 4 * 4 * 4];
        upload_rgba8_region(
            &device,
            &texture,
            0,
            crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            4,
            4,
            &zeros,
        );

        let expected = rgba8_pattern(2, 2, 31);
        upload_rgba8_region(
            &device,
            &texture,
            0,
            crate::HalOrigin3d { x: 1, y: 1, z: 0 },
            2,
            2,
            &expected,
        );

        assert_eq!(
            read_back_rgba8_region(
                &device,
                &texture,
                0,
                crate::HalOrigin3d { x: 1, y: 1, z: 0 },
                2,
                2,
            ),
            expected,
            "B2T must write at destination origin (1,1), and T2B must read from source origin (1,1)"
        );
    }

    #[test]
    fn submit_copies_texture_to_texture_respects_nonzero_2d_origins() {
        let Some(device) = gles_device_or_skip("GLES nonzero T2T origin test") else {
            return;
        };

        let source = rgba8_copy_texture_2d(&device, 4, 4, 1);
        let source_bytes = rgba8_pattern(4, 4, 50);
        upload_rgba8_region(
            &device,
            &source,
            0,
            crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            4,
            4,
            &source_bytes,
        );
        let destination = rgba8_copy_texture_2d(&device, 4, 4, 1);
        let mut expected = vec![0; 4 * 4 * 4];
        upload_rgba8_region(
            &device,
            &destination,
            0,
            crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            4,
            4,
            &expected,
        );

        device
            .queue()
            .submit_copies(&[HalCopy::TextureToTexture(HalTextureCopy {
                source: HalTexture::Gles(source),
                source_mip_level: 0,
                source_origin: crate::HalOrigin3d { x: 2, y: 1, z: 0 },
                destination: HalTexture::Gles(destination.clone()),
                destination_mip_level: 0,
                destination_origin: crate::HalOrigin3d { x: 0, y: 1, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("texture-to-texture copy with nonzero origins must succeed");

        for y in 0..2 {
            for x in 0..2 {
                copy_rgba8_texel(
                    &mut expected,
                    4,
                    (x, y + 1),
                    &source_bytes,
                    4,
                    (x + 2, y + 1),
                );
            }
        }
        assert_eq!(
            read_back_rgba8_region(
                &device,
                &destination,
                0,
                crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                4,
                4,
            ),
            expected,
            "T2T must read source origin (2,1) and write destination origin (0,1)"
        );
    }

    #[test]
    fn submit_copies_texture_to_texture_respects_mip_level_one_origins() {
        let Some(device) = gles_device_or_skip("GLES mip-1 T2T origin test") else {
            return;
        };

        let source = rgba8_copy_texture_2d(&device, 8, 8, 2);
        let source_mip1 = rgba8_pattern(4, 4, 90);
        upload_rgba8_region(
            &device,
            &source,
            1,
            crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            4,
            4,
            &source_mip1,
        );
        let destination = rgba8_copy_texture_2d(&device, 8, 8, 2);
        let mut expected = vec![0; 4 * 4 * 4];
        upload_rgba8_region(
            &device,
            &destination,
            1,
            crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            4,
            4,
            &expected,
        );

        device
            .queue()
            .submit_copies(&[HalCopy::TextureToTexture(HalTextureCopy {
                source: HalTexture::Gles(source),
                source_mip_level: 1,
                source_origin: crate::HalOrigin3d { x: 1, y: 1, z: 0 },
                destination: HalTexture::Gles(destination.clone()),
                destination_mip_level: 1,
                destination_origin: crate::HalOrigin3d { x: 1, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("texture-to-texture copy at mip level 1 must succeed");

        for y in 0..2 {
            for x in 0..2 {
                copy_rgba8_texel(
                    &mut expected,
                    4,
                    (x + 1, y),
                    &source_mip1,
                    4,
                    (x + 1, y + 1),
                );
            }
        }
        assert_eq!(
            read_back_rgba8_region(
                &device,
                &destination,
                1,
                crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                4,
                4,
            ),
            expected,
            "T2T must honor source/destination origins at mip level 1"
        );
    }

    #[test]
    fn submit_copies_r8_odd_rows_round_trip_without_alignment_padding() {
        let Some(device) = gles_device_or_skip("GLES R8 odd-row B2T/T2B alignment test") else {
            return;
        };

        let texture = r8_copy_texture_2d(&device, 3, 3);
        let expected = vec![0, 17, 34, 51, 68, 85, 102, 119, 136];
        upload_r8_region(
            &device,
            &texture,
            crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            3,
            3,
            &expected,
        );

        assert_eq!(
            read_back_r8_region(
                &device,
                &texture,
                crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                3,
                3,
            ),
            expected,
            "R8 rows are 3 bytes wide; GL pack/unpack alignment must be 1"
        );
    }

    #[test]
    fn submit_copies_r8_texture_to_texture_nonzero_sub_box_preserves_zeroes() {
        let Some(device) = gles_device_or_skip("GLES R8 nonzero T2T sub-box test") else {
            return;
        };

        let source = r8_copy_texture_2d(&device, 5, 3);
        let source_bytes = vec![
            10, 11, 12, 13, 14, //
            20, 21, 22, 23, 24, //
            30, 31, 32, 33, 34,
        ];
        upload_r8_region(
            &device,
            &source,
            crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            5,
            3,
            &source_bytes,
        );

        let destination = r8_copy_texture_2d(&device, 5, 3);
        let mut expected = vec![0; 5 * 3];
        upload_r8_region(
            &device,
            &destination,
            crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            5,
            3,
            &expected,
        );

        device
            .queue()
            .submit_copies(&[HalCopy::TextureToTexture(HalTextureCopy {
                source: HalTexture::Gles(source),
                source_mip_level: 0,
                source_origin: crate::HalOrigin3d { x: 2, y: 1, z: 0 },
                destination: HalTexture::Gles(destination.clone()),
                destination_mip_level: 0,
                destination_origin: crate::HalOrigin3d { x: 1, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("R8 texture-to-texture copy with nonzero origins must succeed");

        for y in 0..2 {
            for x in 0..2 {
                let source_offset = ((y + 1) * 5 + (x + 2)) as usize;
                let destination_offset = (y * 5 + (x + 1)) as usize;
                expected[destination_offset] = source_bytes[source_offset];
            }
        }

        assert_eq!(
            read_back_r8_region(
                &device,
                &destination,
                crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                5,
                3,
            ),
            expected,
            "R8 T2T must copy only the 2x2 sub-box and leave the rest zero"
        );
    }

    #[test]
    fn submit_copies_r8snorm_reads_back_via_compute_fallback() {
        let Some(device) = gles_device_or_skip("GLES R8Snorm compute T2B fallback test") else {
            return;
        };

        let texture = byte_copy_texture_2d(&device, crate::HalTextureFormat::R8Snorm, 3, 3);
        let expected = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        upload_byte_region(
            &device,
            &texture,
            crate::HalTextureFormat::R8Snorm,
            1,
            3,
            3,
            &expected,
        );

        assert_eq!(
            read_back_byte_region(
                &device,
                &texture,
                crate::HalTextureFormat::R8Snorm,
                1,
                3,
                3,
            ),
            expected,
            "R8Snorm is not framebuffer-attachable on GLES and must read back byte-exactly through compute"
        );
    }

    #[test]
    fn submit_copies_rg16snorm_reads_back_via_compute_fallback() {
        let Some(device) = gles_device_or_skip("GLES RG16Snorm compute T2B fallback test") else {
            return;
        };

        let texture = byte_copy_texture_2d(&device, crate::HalTextureFormat::Rg16Snorm, 2, 2);
        let values: [i16; 8] = [0, 1, 2, 3, 1000, 2000, 3000, 4000];
        let mut expected = Vec::with_capacity(values.len() * 2);
        for value in values {
            expected.extend_from_slice(&value.to_ne_bytes());
        }
        upload_byte_region(
            &device,
            &texture,
            crate::HalTextureFormat::Rg16Snorm,
            4,
            2,
            2,
            &expected,
        );

        assert_eq!(
            read_back_byte_region(
                &device,
                &texture,
                crate::HalTextureFormat::Rg16Snorm,
                4,
                2,
                2,
            ),
            expected,
            "RG16Snorm readback must preserve the signed 16-bit channel bytes"
        );
    }

    #[test]
    fn submit_copies_depth24plus_reads_depth_aspect_via_compute_fallback() {
        let Some(device) = gles_device_or_skip("GLES Depth24Plus compute T2B fallback test") else {
            return;
        };

        let texture = byte_copy_texture_2d(&device, crate::HalTextureFormat::Depth24Plus, 2, 2);
        let mut upload = Vec::new();
        for value in [0u32, u32::MAX, u32::MAX, 0] {
            upload.extend_from_slice(&value.to_ne_bytes());
        }
        upload_byte_region(
            &device,
            &texture,
            crate::HalTextureFormat::Depth24Plus,
            4,
            2,
            2,
            &upload,
        );

        let readback = read_back_byte_region(
            &device,
            &texture,
            crate::HalTextureFormat::Depth24Plus,
            4,
            2,
            2,
        );
        let values: Vec<u32> = readback
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(values[0], 0, "zero depth must read back as zero");
        assert_eq!(values[3], 0, "zero depth must read back as zero");
        assert!(
            values[1] >= 0x00ff_0000 && values[2] >= 0x00ff_0000,
            "max depth values must read back near the top of the 24-bit depth range: {values:?}"
        );
    }

    #[test]
    fn submit_copies_matrix_preserves_padded_texture_buffer_layouts() {
        let Some(device) = gles_device_or_skip("GLES comprehensive B2T/T2B layout matrix") else {
            return;
        };

        let formats = [
            ByteMatrixFormat {
                format: crate::HalTextureFormat::R8Unorm,
                name: "r8unorm",
                bytes_per_pixel: 1,
            },
            ByteMatrixFormat {
                format: crate::HalTextureFormat::R8Uint,
                name: "r8uint",
                bytes_per_pixel: 1,
            },
            ByteMatrixFormat {
                format: crate::HalTextureFormat::Rg8Unorm,
                name: "rg8unorm",
                bytes_per_pixel: 2,
            },
            ByteMatrixFormat {
                format: crate::HalTextureFormat::R16Float,
                name: "r16float",
                bytes_per_pixel: 2,
            },
            ByteMatrixFormat {
                format: crate::HalTextureFormat::R16Uint,
                name: "r16uint",
                bytes_per_pixel: 2,
            },
            ByteMatrixFormat {
                format: crate::HalTextureFormat::Rgba8Unorm,
                name: "rgba8unorm",
                bytes_per_pixel: 4,
            },
            ByteMatrixFormat {
                format: crate::HalTextureFormat::Rgba16Float,
                name: "rgba16float",
                bytes_per_pixel: 8,
            },
            ByteMatrixFormat {
                format: crate::HalTextureFormat::R8Snorm,
                name: "r8snorm",
                bytes_per_pixel: 1,
            },
            ByteMatrixFormat {
                format: crate::HalTextureFormat::Rg16Snorm,
                name: "rg16snorm",
                bytes_per_pixel: 4,
            },
            ByteMatrixFormat {
                format: crate::HalTextureFormat::Rgb9e5Ufloat,
                name: "rgb9e5ufloat",
                bytes_per_pixel: 4,
            },
        ];
        let origins = [
            crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            crate::HalOrigin3d { x: 1, y: 1, z: 0 },
        ];
        let sizes = [(3, 2), (4, 4)];
        let bytes_per_row = 256;
        let mut failures = Vec::new();

        for format in formats {
            for offset_multiplier in [0, 1, 3] {
                let offset = u64::from(offset_multiplier * format.bytes_per_pixel);
                for origin in origins {
                    for (width, height) in sizes {
                        for layers in [1, 2] {
                            let texture_width = origin.x + width;
                            let texture_height = origin.y + height;
                            let rows_per_image = height + 1;
                            let byte_count = offset
                                + u64::from(bytes_per_row)
                                    * u64::from(rows_per_image)
                                    * u64::from(layers);
                            let case = ByteMatrixCase {
                                format,
                                dimension: crate::HalTextureDimension::D2,
                                offset,
                                origin,
                                width,
                                height,
                                layers,
                            };
                            let texture = byte_copy_texture(
                                &device,
                                case.dimension,
                                format.format,
                                texture_width,
                                texture_height,
                                layers,
                            );
                            let texels = matrix_texel_bytes(&case);
                            let mut upload_bytes = vec![0xee; byte_count as usize];
                            write_matrix_texel_rows(
                                &mut upload_bytes,
                                &case,
                                bytes_per_row,
                                rows_per_image,
                                &texels,
                            );
                            if let Err(error) = submit_matrix_buffer_to_texture(
                                &device,
                                &texture,
                                &case,
                                bytes_per_row,
                                rows_per_image,
                                &upload_bytes,
                            ) {
                                failures.push(format!(
                                    "{} offset={} origin=({}, {}) size={}x{} layers={} B2T error: {error:?}",
                                    format.name,
                                    offset,
                                    origin.x,
                                    origin.y,
                                    width,
                                    height,
                                    layers
                                ));
                                continue;
                            }

                            let readback = match submit_matrix_texture_to_buffer(
                                &device,
                                &texture,
                                &case,
                                bytes_per_row,
                                rows_per_image,
                                byte_count,
                            ) {
                                Ok(readback) => readback,
                                Err(error) => {
                                    failures.push(format!(
                                        "{} offset={} origin=({}, {}) size={}x{} layers={} T2B error: {error:?}",
                                        format.name,
                                        offset,
                                        origin.x,
                                        origin.y,
                                        width,
                                        height,
                                        layers
                                    ));
                                    continue;
                                }
                            };
                            let mut expected = vec![0xab; byte_count as usize];
                            write_matrix_texel_rows(
                                &mut expected,
                                &case,
                                bytes_per_row,
                                rows_per_image,
                                &texels,
                            );
                            if let Some(mismatch) = first_matrix_mismatch(&readback, &expected) {
                                failures.push(format!(
                                    "{} offset={} origin=({}, {}) size={}x{} layers={}: {mismatch}",
                                    format.name, offset, origin.x, origin.y, width, height, layers
                                ));
                            }
                        }
                    }
                }
            }
        }

        let format = ByteMatrixFormat {
            format: crate::HalTextureFormat::R8Snorm,
            name: "r8snorm-3d",
            bytes_per_pixel: 1,
        };
        let case = ByteMatrixCase {
            format,
            dimension: crate::HalTextureDimension::D3,
            offset: u64::from(3 * format.bytes_per_pixel),
            origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            width: 2,
            height: 2,
            layers: 2,
        };
        let rows_per_image = case.height + 1;
        let byte_count = case.offset
            + u64::from(bytes_per_row) * u64::from(rows_per_image) * u64::from(case.layers);
        let texture = byte_copy_texture(
            &device,
            case.dimension,
            format.format,
            case.width,
            case.height,
            case.layers,
        );
        let texels = matrix_texel_bytes(&case);
        let mut upload_bytes = vec![0xee; byte_count as usize];
        write_matrix_texel_rows(
            &mut upload_bytes,
            &case,
            bytes_per_row,
            rows_per_image,
            &texels,
        );
        if let Err(error) = submit_matrix_buffer_to_texture(
            &device,
            &texture,
            &case,
            bytes_per_row,
            rows_per_image,
            &upload_bytes,
        ) {
            failures.push(format!("{} B2T error: {error:?}", format.name));
        } else {
            match submit_matrix_texture_to_buffer(
                &device,
                &texture,
                &case,
                bytes_per_row,
                rows_per_image,
                byte_count,
            ) {
                Ok(readback) => {
                    let mut expected = vec![0xab; byte_count as usize];
                    write_matrix_texel_rows(
                        &mut expected,
                        &case,
                        bytes_per_row,
                        rows_per_image,
                        &texels,
                    );
                    if let Some(mismatch) = first_matrix_mismatch(&readback, &expected) {
                        failures.push(format!("{}: {mismatch}", format.name));
                    }
                }
                Err(error) => {
                    failures.push(format!("{} T2B error: {error:?}", format.name));
                }
            }
        }

        assert!(
            failures.is_empty(),
            "GLES B2T/T2B layout matrix failures:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn submit_copies_sampled_mip_subrange_then_reads_mip_one() {
        let Some(device) = gles_device_or_skip("GLES sampled mip clamp then T2B mip 1 test") else {
            return;
        };

        let sampled = rgba8_copy_texture_2d(&device, 4, 4, 2);
        upload_rgba8_region(
            &device,
            &sampled,
            0,
            crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            4,
            4,
            &[0; 4 * 4 * 4],
        );
        let mip_one = rgba8_pattern(2, 2, 170);
        upload_rgba8_region(
            &device,
            &sampled,
            1,
            crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            2,
            2,
            &mip_one,
        );

        let target = rgba8_copy_texture_2d(&device, 2, 2, 1);
        let pipeline = device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    vertex: "#version 310 es\n\
                             void main() {\n\
                                 vec2 pos = vec2(float((gl_VertexID & 1) << 2) - 1.0,\n\
                                                 float((gl_VertexID & 2) << 1) - 1.0);\n\
                                 gl_Position = vec4(pos, 0.0, 1.0);\n\
                             }\n"
                    .to_owned(),
                    fragment: Some(
                        "#version 310 es\n\
                         precision mediump float;\n\
                         uniform sampler2D u_tex;\n\
                         layout(location = 0) out vec4 frag_color;\n\
                         void main() { frag_color = texture(u_tex, vec2(0.25)); }\n"
                            .to_owned(),
                    ),
                    combined_samplers: vec![crate::HalCombinedSampler {
                        glsl_uniform_name: "u_tex".to_owned(),
                        texture_group: 0,
                        texture_binding: 1,
                        sampler_group: 0,
                        sampler_binding: 2,
                        uses_placeholder_sampler: false,
                    }],
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: Vec::new(),
                    texture_metadata_ubo_binding: None,
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: vec![Some(HalColorTargetState {
                        format: crate::HalTextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 0xf,
                    })],
                    depth_stencil: None,
                    vertex_buffers: Vec::new(),
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[
                    HalDescriptorBinding {
                        group: 0,
                        binding: 1,
                        kind: HalDescriptorBindingKind::Texture,
                    },
                    HalDescriptorBinding {
                        group: 0,
                        binding: 2,
                        kind: HalDescriptorBindingKind::Sampler,
                    },
                ],
            )
            .expect("GLES mip clamp render pipeline creation must succeed");

        let mut pass = render_pass(vec![Some(color_target_for(
            target,
            crate::HalTextureFormat::Rgba8Unorm,
            [0.0, 0.0, 0.0, 0.0],
        ))]);
        pass.pipeline = Some(HalRenderPipeline::Gles(pipeline));
        pass.bind_textures = vec![bound_sampled_texture(sampled.clone(), 1)];
        pass.bind_samplers = vec![nearest_sampler_binding(&device, 2)];
        pass.draw = Some(HalDraw::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });

        device
            .queue()
            .submit_copies(&[HalCopy::RenderPass(pass)])
            .expect("render pass with sampled mip subrange must succeed");

        assert_eq!(
            read_back_rgba8_region(
                &device,
                &sampled,
                1,
                crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                2,
                2,
            ),
            mip_one,
            "T2B of mip 1 must ignore prior sampled-view BASE/MAX_LEVEL clamp"
        );
    }

    #[test]
    fn submit_copies_texture_to_texture_across_array_layers() {
        // T-G9: texture-to-texture copies with array layers. Layers 1..3 of a
        // 3-layer source copy to layers 0..2 of a 2-layer destination; the
        // driver takes glCopyImageSubData when GL_EXT_copy_image / ES 3.2 is
        // available and the framebuffer_texture_layer + glCopyTexSubImage3D
        // fallback otherwise.
        let Some(device) = gles_device_or_skip("GLES array texture-to-texture copy test") else {
            return;
        };

        let copy_image = device
            .queue()
            .inner
            .with_current_context(supports_copy_image)
            .expect("querying GL_EXT_copy_image support must succeed");
        eprintln!(
            "GLES array texture-to-texture copy path: {}",
            if copy_image {
                "glCopyImageSubData (GL_EXT_copy_image / ES 3.2)"
            } else {
                "framebuffer_texture_layer + glCopyTexSubImage3D fallback"
            }
        );

        let source = rgba8_copy_texture(&device, crate::HalTextureDimension::D2, 3);
        let bytes = upload_rgba8_slices(&device, &source, 3);
        let destination = rgba8_copy_texture(&device, crate::HalTextureDimension::D2, 2);

        device
            .queue()
            .submit_copies(&[HalCopy::TextureToTexture(HalTextureCopy {
                source: HalTexture::Gles(source),
                source_mip_level: 0,
                source_origin: crate::HalOrigin3d { x: 0, y: 0, z: 1 },
                destination: HalTexture::Gles(destination.clone()),
                destination_mip_level: 0,
                destination_origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 2,
                },
            })])
            .expect("texture-to-texture copy across array layers must succeed");

        assert_eq!(
            read_back_rgba8_slices(&device, &destination, 0, 2),
            &bytes[RGBA8_SLICE_BYTES as usize..3 * RGBA8_SLICE_BYTES as usize],
            "destination layers 0..2 must hold source layers 1..3"
        );
    }

    fn gles_device_or_skip(label: &str) -> Option<super::super::device::GlesDevice> {
        let instance = match super::super::instance::GlesInstance::new() {
            Ok(instance) => instance,
            Err(error) => {
                eprintln!("skipping {label}; backend unavailable: {error:?}");
                return None;
            }
        };
        let Some(adapter) = instance.enumerate_adapters().into_iter().next() else {
            eprintln!("skipping {label}; no adapter available");
            return None;
        };
        match adapter.create_device() {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping {label}; device unavailable: {error:?}");
                None
            }
        }
    }

    #[test]
    fn submit_compute_pass_writes_rgba8_storage_texture() {
        let Some(device) = gles_device_or_skip("GLES compute storage-texture write test") else {
            return;
        };
        let texture = storage_texture_2x2(
            &device,
            crate::HalTextureFormat::Rgba8Unorm,
            crate::HalTextureUsage {
                copy_src: true,
                copy_dst: false,
                texture_binding: false,
                storage_binding: true,
                render_attachment: false,
                transient: false,
            },
        );
        let pipeline = device
            .create_compute_pipeline(
                crate::HalShaderSource::Glsl {
                    source: "#version 310 es\n\
                             layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n\
                             layout(binding = 0, rgba8) uniform highp writeonly image2D tex;\n\
                             void main() {\n\
                                 imageStore(tex, ivec2(gl_GlobalInvocationID.xy), vec4(0.25, 0.5, 0.75, 1.0));\n\
                             }\n"
                    .to_owned(),
                    stage: crate::HalShaderStage::Compute,
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: vec![crate::HalGlesBindingRemap::new(
                        0,
                        0,
                        crate::HalGlesBindingClass::StorageTexture,
                        0,
                    )],
                    texture_metadata_ubo_binding: None,
                },
                "main",
                (1, 1, 1),
                &[HalDescriptorBinding {
                    group: 0,
                    binding: 0,
                    kind: HalDescriptorBindingKind::StorageTexture {
                        access: HalStorageTextureAccess::WriteOnly,
                    },
                }],
            )
            .expect("GLES storage-texture write compute pipeline must create");

        device
            .queue()
            .submit_copies(&[HalCopy::ComputePass(HalComputePass {
                pipeline: HalComputePipeline::Gles(pipeline),
                bind_buffers: Vec::new(),
                bind_textures: vec![bound_storage_texture(
                    texture.clone(),
                    crate::HalTextureFormat::Rgba8Unorm,
                    HalStorageTextureAccess::WriteOnly,
                )],
                bind_samplers: Vec::new(),
                bind_external_textures: Vec::new(),
                immediate_data: Vec::new(),
                dispatch: HalComputeDispatch::Direct {
                    workgroups: (2, 2, 1),
                },
            })])
            .expect("GLES storage-texture write dispatch must succeed");

        assert_eq!(
            read_texture_bytes(
                &device,
                texture,
                crate::HalTextureFormat::Rgba8Unorm,
                16,
                8,
                2
            ),
            [64, 128, 191, 255].repeat(4)
        );
    }

    #[test]
    fn submit_compute_pass_read_write_r32uint_storage_texture() {
        let Some(device) = gles_device_or_skip("GLES compute read-write storage-texture test")
        else {
            return;
        };
        let texture = storage_texture_2x2(
            &device,
            crate::HalTextureFormat::R32Uint,
            crate::HalTextureUsage {
                copy_src: true,
                copy_dst: true,
                texture_binding: false,
                storage_binding: true,
                render_attachment: false,
                transient: false,
            },
        );
        let upload = device
            .create_buffer(
                16,
                crate::HalBufferUsage {
                    copy_src: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES R32Uint upload buffer creation must succeed");
        let initial = [1_u32, 2, 3, 4]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        upload
            .write(0, &initial)
            .expect("writing R32Uint upload data must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::BufferToTexture(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(upload),
                buffer_layout: crate::HalBufferTextureLayout {
                    offset: 0,
                    bytes_per_row: 8,
                    rows_per_image: 2,
                },
                texture: HalTexture::Gles(texture.clone()),
                format: crate::HalTextureFormat::R32Uint,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("uploading R32Uint storage texture must succeed");

        let pipeline = device
            .create_compute_pipeline(
                crate::HalShaderSource::Glsl {
                    source: "#version 310 es\n\
                             layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n\
                             layout(binding = 0, r32ui) uniform highp uimage2D tex;\n\
                             void main() {\n\
                                 ivec2 p = ivec2(gl_GlobalInvocationID.xy);\n\
                                 uvec4 v = imageLoad(tex, p);\n\
                                 imageStore(tex, p, uvec4(v.x + 1u, 0u, 0u, 0u));\n\
                             }\n"
                    .to_owned(),
                    stage: crate::HalShaderStage::Compute,
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: vec![crate::HalGlesBindingRemap::new(
                        0,
                        0,
                        crate::HalGlesBindingClass::StorageTexture,
                        0,
                    )],
                    texture_metadata_ubo_binding: None,
                },
                "main",
                (1, 1, 1),
                &[HalDescriptorBinding {
                    group: 0,
                    binding: 0,
                    kind: HalDescriptorBindingKind::StorageTexture {
                        access: HalStorageTextureAccess::ReadWrite,
                    },
                }],
            )
            .expect("GLES read-write storage-texture compute pipeline must create");

        device
            .queue()
            .submit_copies(&[HalCopy::ComputePass(HalComputePass {
                pipeline: HalComputePipeline::Gles(pipeline),
                bind_buffers: Vec::new(),
                bind_textures: vec![bound_storage_texture(
                    texture.clone(),
                    crate::HalTextureFormat::R32Uint,
                    HalStorageTextureAccess::ReadWrite,
                )],
                bind_samplers: Vec::new(),
                bind_external_textures: Vec::new(),
                immediate_data: Vec::new(),
                dispatch: HalComputeDispatch::Direct {
                    workgroups: (2, 2, 1),
                },
            })])
            .expect("GLES read-write storage-texture dispatch must succeed");

        let bytes =
            read_texture_bytes(&device, texture, crate::HalTextureFormat::R32Uint, 16, 8, 2);
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        assert_eq!(values, [2, 3, 4, 5]);
    }

    #[test]
    fn submit_compute_pass_rejects_unsupported_storage_texture_format() {
        let Some(device) = gles_device_or_skip("GLES unsupported storage-texture format test")
        else {
            return;
        };
        let texture = storage_texture_2x2(
            &device,
            crate::HalTextureFormat::Rg32Uint,
            crate::HalTextureUsage {
                copy_src: false,
                copy_dst: false,
                texture_binding: false,
                storage_binding: true,
                render_attachment: false,
                transient: false,
            },
        );
        let pipeline = device
            .create_compute_pipeline(
                crate::HalShaderSource::Glsl {
                    source: "#version 310 es\n\
                             layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n\
                             layout(binding = 0, rgba8) uniform highp writeonly image2D tex;\n\
                             void main() { imageStore(tex, ivec2(0), vec4(1.0)); }\n"
                        .to_owned(),
                    stage: crate::HalShaderStage::Compute,
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: vec![crate::HalGlesBindingRemap::new(
                        0,
                        0,
                        crate::HalGlesBindingClass::StorageTexture,
                        0,
                    )],
                    texture_metadata_ubo_binding: None,
                },
                "main",
                (1, 1, 1),
                &[HalDescriptorBinding {
                    group: 0,
                    binding: 0,
                    kind: HalDescriptorBindingKind::StorageTexture {
                        access: HalStorageTextureAccess::WriteOnly,
                    },
                }],
            )
            .expect("GLES unsupported-format storage-texture pipeline must create");

        let error = device
            .queue()
            .submit_copies(&[HalCopy::ComputePass(HalComputePass {
                pipeline: HalComputePipeline::Gles(pipeline),
                bind_buffers: Vec::new(),
                bind_textures: vec![bound_storage_texture(
                    texture,
                    crate::HalTextureFormat::Rg32Uint,
                    HalStorageTextureAccess::WriteOnly,
                )],
                bind_samplers: Vec::new(),
                bind_external_textures: Vec::new(),
                immediate_data: Vec::new(),
                dispatch: HalComputeDispatch::Direct {
                    workgroups: (1, 1, 1),
                },
            })])
            .expect_err("unsupported GLES storage image format must return HalError");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "GLES image load/store does not support this storage format",
            }
        ));
    }

    fn storage_texture_2x2(
        device: &super::super::device::GlesDevice,
        format: crate::HalTextureFormat,
        usage: crate::HalTextureUsage,
    ) -> super::super::texture::GlesTexture {
        device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format,
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage,
            })
            .expect("GLES storage texture creation must succeed")
    }

    fn bound_storage_texture(
        texture: super::super::texture::GlesTexture,
        format: crate::HalTextureFormat,
        access: HalStorageTextureAccess,
    ) -> HalBoundTexture {
        HalBoundTexture {
            group: 0,
            binding: 0,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: None,
            texture: HalTexture::Gles(texture),
            format,
            dimension: HalTextureViewDimension::D2,
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
            aspect: crate::HalTextureAspect::All,
            swizzle: crate::HalTextureComponentSwizzle::default(),
            storage_access: Some(access),
        }
    }

    fn read_texture_bytes(
        device: &super::super::device::GlesDevice,
        texture: super::super::texture::GlesTexture,
        format: crate::HalTextureFormat,
        byte_count: u64,
        bytes_per_row: u32,
        rows_per_image: u32,
    ) -> Vec<u8> {
        let readback = device
            .create_buffer(
                byte_count,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES storage texture readback buffer creation must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(readback.clone()),
                buffer_layout: crate::HalBufferTextureLayout {
                    offset: 0,
                    bytes_per_row,
                    rows_per_image,
                },
                texture: HalTexture::Gles(texture),
                format,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("GLES storage texture-to-buffer copy must succeed");
        readback
            .read(0, byte_count)
            .expect("reading storage texture readback buffer must succeed")
    }

    fn render_attachment_texture(
        device: &super::super::device::GlesDevice,
        format: crate::HalTextureFormat,
    ) -> super::super::texture::GlesTexture {
        device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format,
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: true,
                    copy_dst: false,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("GLES render-attachment texture creation must succeed")
    }

    fn rgba8_render_attachment_texture(
        device: &super::super::device::GlesDevice,
        sample_count: u32,
    ) -> super::super::texture::GlesTexture {
        device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count,
                usage: crate::HalTextureUsage {
                    copy_src: sample_count == 1,
                    copy_dst: false,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("GLES RGBA8 render-attachment texture creation must succeed")
    }

    fn fullscreen_rgba8_pipeline(
        device: &super::super::device::GlesDevice,
        sample_mask: u32,
        alpha_to_coverage_enabled: bool,
    ) -> super::super::pipeline::GlesRenderPipeline {
        device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    vertex: "#version 310 es\n\
                             void main() {\n\
                                 vec2 pos = vec2(float((gl_VertexID & 1) << 2) - 1.0,\n\
                                                 float((gl_VertexID & 2) << 1) - 1.0);\n\
                                 gl_Position = vec4(pos, 0.0, 1.0);\n\
                             }\n"
                    .to_owned(),
                    fragment: Some(
                        "#version 310 es\n\
                         precision mediump float;\n\
                         layout(location = 0) out vec4 frag_color;\n\
                         void main() { frag_color = vec4(1.0, 0.0, 0.0, 1.0); }\n"
                            .to_owned(),
                    ),
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: Vec::new(),
                    texture_metadata_ubo_binding: None,
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 4,
                    sample_mask,
                    alpha_to_coverage_enabled,
                    color_targets: vec![Some(HalColorTargetState {
                        format: crate::HalTextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 0xf,
                    })],
                    depth_stencil: None,
                    vertex_buffers: Vec::new(),
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[],
            )
            .expect("GLES MSAA render pipeline creation must succeed")
    }

    fn submit_msaa_resolve_draw(
        device: &super::super::device::GlesDevice,
        sample_mask: u32,
        alpha_to_coverage_enabled: bool,
        clear_color: [f64; 4],
    ) -> super::super::texture::GlesTexture {
        let msaa = rgba8_render_attachment_texture(device, 4);
        let resolve = rgba8_render_attachment_texture(device, 1);
        let pipeline = fullscreen_rgba8_pipeline(device, sample_mask, alpha_to_coverage_enabled);
        let mut target = color_target_for(msaa, crate::HalTextureFormat::Rgba8Unorm, clear_color);
        target.resolve_target = Some(HalTexture::Gles(resolve.clone()));
        target.resolve_view_format = Some(crate::HalTextureFormat::Rgba8Unorm);
        target.store = false;
        let mut pass = render_pass(vec![Some(target)]);
        pass.pipeline = Some(HalRenderPipeline::Gles(pipeline));
        pass.draw = Some(HalDraw::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });
        device
            .queue()
            .submit_copies(&[HalCopy::RenderPass(pass)])
            .expect("MSAA render pass plus resolve must succeed");
        resolve
    }

    fn read_rgba8_2x2(
        device: &super::super::device::GlesDevice,
        texture: super::super::texture::GlesTexture,
    ) -> Vec<u8> {
        let readback = device
            .create_buffer(
                16,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES MSAA resolve readback buffer creation must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(readback.clone()),
                buffer_layout: crate::HalBufferTextureLayout {
                    offset: 0,
                    bytes_per_row: 8,
                    rows_per_image: 2,
                },
                texture: HalTexture::Gles(texture),
                format: crate::HalTextureFormat::Rgba8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 2,
                    height: 2,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("resolved texture-to-buffer copy must succeed");
        readback
            .read(0, 16)
            .expect("reading resolved RGBA8 pixels must succeed")
    }

    fn read_rgba8_1x1(
        device: &super::super::device::GlesDevice,
        texture: super::super::texture::GlesTexture,
    ) -> [u8; 4] {
        let readback = device
            .create_buffer(
                4,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES RGBA8 readback buffer creation must succeed");
        device
            .queue()
            .submit_copies(&[HalCopy::TextureToBuffer(HalBufferTextureCopy {
                buffer: HalBuffer::Gles(readback.clone()),
                buffer_layout: crate::HalBufferTextureLayout {
                    offset: 0,
                    bytes_per_row: 4,
                    rows_per_image: 1,
                },
                texture: HalTexture::Gles(texture),
                format: crate::HalTextureFormat::Rgba8Unorm,
                aspect: crate::HalTextureAspect::All,
                mip_level: 0,
                origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                extent: crate::HalExtent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            })])
            .expect("RGBA8 texture-to-buffer copy must succeed");
        readback
            .read(0, 4)
            .expect("reading RGBA8 pixel must succeed")
            .try_into()
            .expect("RGBA8 readback must be exactly one pixel")
    }

    fn skip_if_msaa4_unavailable(device: &super::super::device::GlesDevice, label: &str) -> bool {
        if device.inner_clone().max_samples() < 4 {
            eprintln!(
                "skipping {label}; GL_MAX_SAMPLES={} is below 4",
                device.inner_clone().max_samples()
            );
            true
        } else {
            false
        }
    }

    fn color_target_for(
        texture: super::super::texture::GlesTexture,
        view_format: crate::HalTextureFormat,
        clear_color: [f64; 4],
    ) -> crate::HalRenderColorTarget {
        crate::HalRenderColorTarget {
            texture: HalTexture::Gles(texture),
            view_format,
            resolve_target: None,
            resolve_view_format: None,
            mip_level: 0,
            array_layer: 0,
            depth_slice: 0,
            resolve_mip_level: 0,
            resolve_array_layer: 0,
            load_op: HalRenderLoadOp::Clear,
            store: true,
            clear_color,
        }
    }

    fn texture_to_buffer_copy(
        texture: super::super::texture::GlesTexture,
        format: crate::HalTextureFormat,
        buffer: super::super::buffer::GlesBuffer,
        bytes_per_row: u32,
    ) -> HalCopy {
        HalCopy::TextureToBuffer(HalBufferTextureCopy {
            buffer: HalBuffer::Gles(buffer),
            buffer_layout: crate::HalBufferTextureLayout {
                offset: 0,
                bytes_per_row,
                rows_per_image: 1,
            },
            texture: HalTexture::Gles(texture),
            format,
            aspect: crate::HalTextureAspect::All,
            mip_level: 0,
            origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
            extent: crate::HalExtent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        })
    }

    #[test]
    fn submit_render_pass_clears_two_color_attachments_independently() {
        // T-G8 (MRT): a clear-only pass with two color attachments must
        // clear each attachment with its own clear value through the
        // per-draw-buffer `glClearBuffer*` path (Rgba8Unorm via
        // glClearBufferfv on draw buffer 0, R32Uint via glClearBufferuiv on
        // draw buffer 1).
        let Some(device) = gles_device_or_skip("GLES two-attachment clear test") else {
            return;
        };

        let rgba_texture = render_attachment_texture(&device, crate::HalTextureFormat::Rgba8Unorm);
        let uint_texture = render_attachment_texture(&device, crate::HalTextureFormat::R32Uint);
        let rgba_readback = device
            .create_buffer(
                4,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES readback buffer creation must succeed");
        let uint_readback = device
            .create_buffer(
                4,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES readback buffer creation must succeed");

        let pass = render_pass(vec![
            Some(color_target_for(
                rgba_texture.clone(),
                crate::HalTextureFormat::Rgba8Unorm,
                [1.0, 0.0, 0.0, 1.0],
            )),
            Some(color_target_for(
                uint_texture.clone(),
                crate::HalTextureFormat::R32Uint,
                [7.0, 0.0, 0.0, 1.0],
            )),
        ]);

        device
            .queue()
            .submit_copies(&[
                HalCopy::RenderPass(pass),
                texture_to_buffer_copy(
                    rgba_texture,
                    crate::HalTextureFormat::Rgba8Unorm,
                    rgba_readback.clone(),
                    4,
                ),
                texture_to_buffer_copy(
                    uint_texture,
                    crate::HalTextureFormat::R32Uint,
                    uint_readback.clone(),
                    4,
                ),
            ])
            .expect("clear-only pass with two color attachments plus readbacks must succeed");

        assert_eq!(
            rgba_readback
                .read(0, 4)
                .expect("reading back the Rgba8Unorm texel must succeed"),
            [255, 0, 0, 255],
            "attachment 0 must hold its own clear color"
        );
        let bytes = uint_readback
            .read(0, 4)
            .expect("reading back the R32Uint texel must succeed");
        assert_eq!(
            u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            7,
            "attachment 1 must hold its own integer clear value"
        );
    }

    #[test]
    fn submit_render_pass_draws_into_two_color_attachments() {
        // T-G8 (MRT): a render pipeline with two color targets sharing one
        // write mask / blend state must create, and a draw whose fragment
        // stage writes `layout(location = 0) out vec4` plus
        // `layout(location = 1) out uvec4` must land in both attachments.
        let Some(device) = gles_device_or_skip("GLES two-attachment draw test") else {
            return;
        };

        let rgba_texture = render_attachment_texture(&device, crate::HalTextureFormat::Rgba8Unorm);
        let uint_texture = render_attachment_texture(&device, crate::HalTextureFormat::R32Uint);
        let rgba_readback = device
            .create_buffer(
                4,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES readback buffer creation must succeed");
        let uint_readback = device
            .create_buffer(
                4,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES readback buffer creation must succeed");

        let pipeline = device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    // Full-viewport triangle from gl_VertexID; covers the
                    // 1x1 attachments without vertex buffers.
                    vertex: "#version 310 es\n\
                             void main() {\n\
                                 vec2 pos = vec2(float((gl_VertexID & 1) << 2) - 1.0,\n\
                                                 float((gl_VertexID & 2) << 1) - 1.0);\n\
                                 gl_Position = vec4(pos, 0.0, 1.0);\n\
                             }\n"
                    .to_owned(),
                    fragment: Some(
                        "#version 310 es\n\
                         precision mediump float;\n\
                         precision highp int;\n\
                         layout(location = 0) out vec4 frag_color0;\n\
                         layout(location = 1) out highp uvec4 frag_color1;\n\
                         void main() {\n\
                             frag_color0 = vec4(0.0, 1.0, 0.0, 1.0);\n\
                             frag_color1 = uvec4(9u, 0u, 0u, 1u);\n\
                         }\n"
                        .to_owned(),
                    ),
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: Vec::new(),
                    texture_metadata_ubo_binding: None,
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: vec![
                        Some(HalColorTargetState {
                            format: crate::HalTextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: 0xf,
                        }),
                        Some(HalColorTargetState {
                            format: crate::HalTextureFormat::R32Uint,
                            blend: None,
                            write_mask: 0xf,
                        }),
                    ],
                    depth_stencil: None,
                    vertex_buffers: Vec::new(),
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[],
            )
            .expect("GLES render pipeline with two color targets must create");

        let mut pass = render_pass(vec![
            Some(color_target_for(
                rgba_texture.clone(),
                crate::HalTextureFormat::Rgba8Unorm,
                [0.0, 0.0, 0.0, 0.0],
            )),
            Some(color_target_for(
                uint_texture.clone(),
                crate::HalTextureFormat::R32Uint,
                [0.0, 0.0, 0.0, 0.0],
            )),
        ]);
        pass.pipeline = Some(HalRenderPipeline::Gles(pipeline));
        pass.draw = Some(HalDraw::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });

        device
            .queue()
            .submit_copies(&[
                HalCopy::RenderPass(pass),
                texture_to_buffer_copy(
                    rgba_texture,
                    crate::HalTextureFormat::Rgba8Unorm,
                    rgba_readback.clone(),
                    4,
                ),
                texture_to_buffer_copy(
                    uint_texture,
                    crate::HalTextureFormat::R32Uint,
                    uint_readback.clone(),
                    4,
                ),
            ])
            .expect("draw into two color attachments plus readbacks must succeed");

        assert_eq!(
            rgba_readback
                .read(0, 4)
                .expect("reading back the Rgba8Unorm texel must succeed"),
            [0, 255, 0, 255],
            "fragment output 0 must land in attachment 0"
        );
        let bytes = uint_readback
            .read(0, 4)
            .expect("reading back the R32Uint texel must succeed");
        assert_eq!(
            u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            9,
            "fragment output 1 must land in attachment 1"
        );
    }

    #[test]
    fn submit_render_pass_msaa_draw_resolves_to_single_sample_texture() {
        let Some(device) = gles_device_or_skip("GLES MSAA resolve test") else {
            return;
        };
        if skip_if_msaa4_unavailable(&device, "GLES MSAA resolve test") {
            return;
        }
        let resolve = submit_msaa_resolve_draw(&device, u32::MAX, false, [0.0, 0.0, 0.0, 1.0]);
        let bytes = read_rgba8_2x2(&device, resolve);
        assert_eq!(
            bytes,
            [255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255]
        );
    }

    #[test]
    fn submit_render_pass_msaa_sample_mask_controls_resolved_color() {
        let Some(device) = gles_device_or_skip("GLES MSAA sample-mask test") else {
            return;
        };
        if skip_if_msaa4_unavailable(&device, "GLES MSAA sample-mask test") {
            return;
        }
        let masked = submit_msaa_resolve_draw(&device, 0, false, [0.0, 0.0, 1.0, 1.0]);
        let masked_bytes = read_rgba8_2x2(&device, masked);
        assert_eq!(
            masked_bytes,
            [0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255]
        );

        let unmasked = submit_msaa_resolve_draw(&device, u32::MAX, false, [0.0, 0.0, 1.0, 1.0]);
        let unmasked_bytes = read_rgba8_2x2(&device, unmasked);
        assert_eq!(
            unmasked_bytes,
            [255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255]
        );
    }

    #[test]
    fn submit_render_pass_msaa_alpha_to_coverage_draws_ok() {
        let Some(device) = gles_device_or_skip("GLES alpha-to-coverage MSAA smoke test") else {
            return;
        };
        if skip_if_msaa4_unavailable(&device, "GLES alpha-to-coverage MSAA smoke test") {
            return;
        }
        let resolve = submit_msaa_resolve_draw(&device, u32::MAX, true, [0.0, 0.0, 0.0, 1.0]);
        let _bytes = read_rgba8_2x2(&device, resolve);
    }

    #[test]
    fn submit_render_pass_indexed_draw_applies_base_vertex() {
        // T-G11: an indexed draw with baseVertex=1 must fetch vertices
        // 1..=3 through `glDrawElementsBaseVertex` when the device supports
        // the base-vertex entry points. The vertex buffer holds four vec2
        // positions where vertex 0 duplicates vertex 1, so drawing indices
        // [0, 1, 2] *without* the base-vertex offset yields a zero-area
        // triangle (no fragments; the clear color survives), while applying
        // baseVertex=1 selects the full-viewport triangle at vertices 1..=3
        // and writes green.
        let Some(device) = gles_device_or_skip("GLES base-vertex indexed draw test") else {
            return;
        };
        if !device.inner_clone().supports_base_vertex() {
            eprintln!(
                "skipping GLES base-vertex indexed draw test; device reports no GLES 3.2 / OES/EXT_draw_elements_base_vertex support"
            );
            return;
        }

        let color_texture = render_attachment_texture(&device, crate::HalTextureFormat::Rgba8Unorm);
        let readback = device
            .create_buffer(
                4,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES readback buffer creation must succeed");

        // Vertex 0 duplicates vertex 1; vertices 1..=3 form the
        // full-viewport triangle.
        let positions: [[f32; 2]; 4] = [[-1.0, -1.0], [-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];
        let vertex_bytes: Vec<u8> = positions
            .iter()
            .flatten()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let vertex_buffer = device
            .create_buffer(
                vertex_bytes.len() as u64,
                crate::HalBufferUsage {
                    vertex: true,
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES vertex buffer creation must succeed");
        vertex_buffer
            .write(0, &vertex_bytes)
            .expect("writing vertex data must succeed");

        let indices: [u16; 3] = [0, 1, 2];
        let index_bytes: Vec<u8> = indices
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let index_buffer = device
            .create_buffer(
                index_bytes.len() as u64,
                crate::HalBufferUsage {
                    index: true,
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES index buffer creation must succeed");
        index_buffer
            .write(0, &index_bytes)
            .expect("writing index data must succeed");

        let pipeline = device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    vertex: "#version 310 es\n\
                             layout(location = 0) in vec2 position;\n\
                             void main() { gl_Position = vec4(position, 0.0, 1.0); }\n"
                        .to_owned(),
                    fragment: Some(
                        "#version 310 es\n\
                         precision mediump float;\n\
                         layout(location = 0) out vec4 frag_color;\n\
                         void main() { frag_color = vec4(0.0, 1.0, 0.0, 1.0); }\n"
                            .to_owned(),
                    ),
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: Vec::new(),
                    texture_metadata_ubo_binding: None,
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: vec![Some(HalColorTargetState {
                        format: crate::HalTextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 0xf,
                    })],
                    depth_stencil: None,
                    vertex_buffers: vec![crate::HalVertexBufferLayout {
                        slot: 0,
                        array_stride: 8,
                        step_mode: HalVertexStepMode::Vertex,
                        attributes: vec![crate::HalVertexAttribute {
                            format: crate::HalVertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                            metal_buffer_index: 0,
                        }],
                    }],
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[],
            )
            .expect("GLES render pipeline creation must succeed");

        let mut pass = render_pass(vec![Some(color_target_for(
            color_texture.clone(),
            crate::HalTextureFormat::Rgba8Unorm,
            [0.0, 0.0, 0.0, 0.0],
        ))]);
        pass.pipeline = Some(HalRenderPipeline::Gles(pipeline));
        pass.vertex_buffers = vec![crate::HalBoundBuffer {
            group: 0,
            binding: 0,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: None,
            buffer: HalBuffer::Gles(vertex_buffer),
            offset: 0,
            size: vertex_bytes.len() as u64,
        }];
        pass.index_buffer = Some(Box::new(crate::HalBoundIndexBuffer {
            buffer: HalBuffer::Gles(index_buffer),
            format: HalIndexFormat::Uint16,
            offset: 0,
            size: index_bytes.len() as u64,
        }));
        pass.draw = Some(HalDraw::Indexed {
            index_count: 3,
            instance_count: 1,
            first_index: 0,
            base_vertex: 1,
            first_instance: 0,
        });

        device
            .queue()
            .submit_copies(&[
                HalCopy::RenderPass(pass),
                texture_to_buffer_copy(
                    color_texture,
                    crate::HalTextureFormat::Rgba8Unorm,
                    readback.clone(),
                    4,
                ),
            ])
            .expect("indexed draw with baseVertex 1 plus readback must succeed");

        assert_eq!(
            readback
                .read(0, 4)
                .expect("reading back the Rgba8Unorm texel must succeed"),
            [0, 255, 0, 255],
            "baseVertex 1 must offset index fetches to the full-viewport triangle at vertices 1..=3"
        );
    }

    #[test]
    fn submit_render_pass_clears_float_color_target_with_ext_color_buffer_float() {
        // T-G12: with `GL_EXT_color_buffer_float` the float formats are
        // color-renderable — a render pipeline with an Rgba16Float color
        // target must create, and a clear-only pass on an R32Float
        // attachment must land the exact f32 clear value in a readback
        // buffer. Self-skips (naming the absent extensions) on contexts
        // without the extension.
        let Some(device) = gles_device_or_skip("GLES float color target test") else {
            return;
        };
        let caps = device.inner_clone().color_render_caps();
        if !caps.color_buffer_float {
            let mut absent = vec!["GL_EXT_color_buffer_float"];
            if !caps.color_buffer_half_float {
                absent.push("GL_EXT_color_buffer_half_float");
            }
            eprintln!(
                "skipping GLES float color target test; absent extensions: {}",
                absent.join(", ")
            );
            return;
        }

        // Pipeline side: an Rgba16Float color target passes the caps-gated
        // renderability check and the pipeline creates.
        device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    // Full-viewport triangle from gl_VertexID; no vertex
                    // buffers needed.
                    vertex: "#version 310 es\n\
                             void main() {\n\
                                 vec2 pos = vec2(float((gl_VertexID & 1) << 2) - 1.0,\n\
                                                 float((gl_VertexID & 2) << 1) - 1.0);\n\
                                 gl_Position = vec4(pos, 0.0, 1.0);\n\
                             }\n"
                    .to_owned(),
                    fragment: Some(
                        "#version 310 es\n\
                         precision mediump float;\n\
                         layout(location = 0) out vec4 frag_color;\n\
                         void main() { frag_color = vec4(0.25, 0.5, 0.75, 1.0); }\n"
                            .to_owned(),
                    ),
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: Vec::new(),
                    texture_metadata_ubo_binding: None,
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: vec![Some(HalColorTargetState {
                        format: crate::HalTextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: 0xf,
                    })],
                    depth_stencil: None,
                    vertex_buffers: Vec::new(),
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[],
            )
            .expect("GLES render pipeline with an Rgba16Float color target must create");

        // Attachment side: clear an R32Float texture to 0.5 and read the
        // f32 bits back through a texture-to-buffer copy.
        let float_texture = render_attachment_texture(&device, crate::HalTextureFormat::R32Float);
        let readback = device
            .create_buffer(
                4,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES readback buffer creation must succeed");

        let pass = render_pass(vec![Some(color_target_for(
            float_texture.clone(),
            crate::HalTextureFormat::R32Float,
            [0.5, 0.0, 0.0, 1.0],
        ))]);
        device
            .queue()
            .submit_copies(&[
                HalCopy::RenderPass(pass),
                texture_to_buffer_copy(
                    float_texture,
                    crate::HalTextureFormat::R32Float,
                    readback.clone(),
                    4,
                ),
            ])
            .expect("clear-only pass on an R32Float attachment plus readback must succeed");

        let bytes = readback
            .read(0, 4)
            .expect("reading back the R32Float texel must succeed");
        assert_eq!(
            f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            0.5,
            "the R32Float attachment must hold the exact f32 clear value"
        );
    }

    fn render_pass(
        color_targets: Vec<Option<crate::HalRenderColorTarget>>,
    ) -> crate::HalRenderPass {
        crate::HalRenderPass {
            pipeline: None,
            color_targets,
            framebuffer_fetch_color_slots: Vec::new(),
            depth_stencil_attachment: None,
            bind_buffers: Vec::new(),
            bind_textures: Vec::new(),
            bind_samplers: Vec::new(),
            bind_external_textures: Vec::new(),
            vertex_buffers: Vec::new(),
            index_buffer: None,
            indirect_buffer: None,
            viewport: None,
            scissor_rect: None,
            blend_constant: [0.0; 4],
            stencil_reference: 0,
            occlusion_query_set: None,
            occlusion_query_index: None,
            draw: None,
            immediate_data: Vec::new(),
        }
    }

    #[test]
    fn pixels_per_row_accepts_aligned_and_zero_stride() {
        assert_eq!(pixels_per_row(256, 4).expect("aligned row"), 64);
        assert_eq!(pixels_per_row(0, 4).expect("zero row stride"), 0);
    }

    #[test]
    fn pixels_per_row_rejects_unaligned_and_zero_pixel_size() {
        let unaligned = pixels_per_row(255, 4).expect_err("unaligned row must error");
        assert!(matches!(
            unaligned,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "bytes_per_row is not a multiple of bytes_per_pixel",
            }
        ));

        let zero_pixel_size = pixels_per_row(8, 0).expect_err("zero pixel size must error");
        assert!(matches!(
            zero_pixel_size,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "bytes_per_row is not a multiple of bytes_per_pixel",
            }
        ));
    }

    #[test]
    fn i32_from_u32_accepts_in_range_and_rejects_overflow() {
        assert_eq!(i32_from_u32(0, "test").expect("zero is in range"), 0);
        assert_eq!(
            i32_from_u32(i32::MAX as u32, "test").expect("i32::MAX is in range"),
            i32::MAX
        );
        let overflow = i32_from_u32(i32::MAX as u32 + 1, "overflow message")
            .expect_err("overflow must be rejected");
        assert!(matches!(
            overflow,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "overflow message",
            }
        ));
    }

    #[test]
    fn u32_from_u64_accepts_in_range_and_rejects_overflow() {
        assert_eq!(u32_from_u64(0, "test").expect("zero is in range"), 0);
        assert_eq!(
            u32_from_u64(u64::from(u32::MAX), "test").expect("u32::MAX is in range"),
            u32::MAX
        );
        let overflow = u32_from_u64(u64::from(u32::MAX) + 1, "overflow message")
            .expect_err("overflow must be rejected");
        assert!(matches!(
            overflow,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "overflow message",
            }
        ));
    }

    #[test]
    fn ensure_2d_target_copy_accepts_layer_one_z_zero_only() {
        assert!(ensure_2d_target_copy(1, 0).is_ok());
        assert!(matches!(
            ensure_2d_target_copy(2, 0),
            Err(HalError::BufferOperationFailed {
                backend: "gles",
                ..
            })
        ));
        assert!(matches!(
            ensure_2d_target_copy(1, 1),
            Err(HalError::BufferOperationFailed {
                backend: "gles",
                ..
            })
        ));
        assert!(matches!(
            ensure_2d_target_copy(0, 0),
            Err(HalError::BufferOperationFailed {
                backend: "gles",
                ..
            })
        ));
    }

    #[test]
    fn gles_version_at_least_3_2_parses_common_strings() {
        assert!(gles_version_at_least_3_2("OpenGL ES 3.2 ANGLE"));
        assert!(gles_version_at_least_3_2("OpenGL ES 4.0"));
        assert!(!gles_version_at_least_3_2("OpenGL ES 3.1 ANGLE"));
        assert!(!gles_version_at_least_3_2("not a version"));
    }

    #[test]
    fn binding_target_maps_buffer_kinds() {
        let bindings = [
            HalDescriptorBinding {
                group: 0,
                binding: 0,
                kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Uniform),
            },
            HalDescriptorBinding {
                group: 0,
                binding: 1,
                kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage),
            },
        ];

        assert_eq!(
            binding_target(&bindings, 0, 0).expect("uniform binding"),
            (glow::UNIFORM_BUFFER, HalGlesBindingClass::UniformBuffer)
        );
        assert_eq!(
            binding_target(&bindings, 0, 1).expect("storage binding"),
            (
                glow::SHADER_STORAGE_BUFFER,
                HalGlesBindingClass::StorageBuffer
            )
        );
        let missing = binding_target(&bindings, 0, 2).expect_err("missing binding");
        assert!(matches!(
            missing,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "buffer binding is missing from pipeline layout",
            }
        ));
    }

    #[test]
    fn texture_view_target_maps_supported_dimensions_and_gates_cube_array() {
        assert_eq!(
            texture_view_target(HalTextureViewDimension::D1, glow::TEXTURE_2D, false)
                .expect("D1 target"),
            glow::TEXTURE_2D
        );
        assert_eq!(
            texture_view_target(HalTextureViewDimension::D2, glow::TEXTURE_2D, false)
                .expect("D2 target"),
            glow::TEXTURE_2D
        );
        assert_eq!(
            texture_view_target(
                HalTextureViewDimension::D2,
                glow::TEXTURE_2D_MULTISAMPLE,
                false,
            )
            .expect("multisample D2 target"),
            glow::TEXTURE_2D_MULTISAMPLE
        );
        assert_eq!(
            texture_view_target(
                HalTextureViewDimension::D2Array,
                glow::TEXTURE_2D_ARRAY,
                false
            )
            .expect("D2Array target"),
            glow::TEXTURE_2D_ARRAY
        );
        assert_eq!(
            texture_view_target(HalTextureViewDimension::D3, glow::TEXTURE_3D, false)
                .expect("D3 target"),
            glow::TEXTURE_3D
        );
        assert_eq!(
            texture_view_target(HalTextureViewDimension::Cube, glow::TEXTURE_CUBE_MAP, false)
                .expect("cube target"),
            glow::TEXTURE_CUBE_MAP
        );
        assert_eq!(
            texture_view_target(HalTextureViewDimension::CubeArray, glow::TEXTURE_2D, true)
                .expect("cube-array target"),
            glow::TEXTURE_CUBE_MAP_ARRAY
        );
        let cube_array =
            texture_view_target(HalTextureViewDimension::CubeArray, glow::TEXTURE_2D, false)
                .expect_err("cube array");
        assert!(matches!(
            cube_array,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "GLES lacks cube-array textures",
            }
        ));
    }

    #[test]
    fn sampled_texture_view_predicate_matches_reinterpretation_cases() {
        let meta = GlesTextureMeta {
            hal_format: crate::HalTextureFormat::Rgba8Unorm,
            format: super::super::format::map_texture_format(crate::HalTextureFormat::Rgba8Unorm)
                .expect("rgba8unorm format"),
            dimension: crate::HalTextureDimension::D2,
            target: glow::TEXTURE_2D_ARRAY,
            width: 2,
            height: 2,
            depth_or_array_layers: 6,
            mip_level_count: 1,
            sample_count: 1,
        };
        assert!(!requires_texture_view_for(
            &meta,
            glow::TEXTURE_2D_ARRAY,
            crate::HalTextureFormat::Rgba8Unorm,
            crate::HalTextureAspect::All,
            0,
            6,
        ));
        assert!(requires_texture_view_for(
            &meta,
            glow::TEXTURE_CUBE_MAP,
            crate::HalTextureFormat::Rgba8Unorm,
            crate::HalTextureAspect::All,
            0,
            6,
        ));
        assert!(requires_texture_view_for(
            &meta,
            glow::TEXTURE_2D_ARRAY,
            crate::HalTextureFormat::Rgba8Unorm,
            crate::HalTextureAspect::All,
            1,
            2,
        ));

        let stencil_meta = GlesTextureMeta {
            hal_format: crate::HalTextureFormat::Depth24PlusStencil8,
            format: super::super::format::map_texture_format(
                crate::HalTextureFormat::Depth24PlusStencil8,
            )
            .expect("packed depth-stencil format"),
            dimension: crate::HalTextureDimension::D2,
            target: glow::TEXTURE_2D,
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
        };
        assert!(requires_texture_view_for(
            &stencil_meta,
            glow::TEXTURE_2D,
            crate::HalTextureFormat::Stencil8,
            crate::HalTextureAspect::StencilOnly,
            0,
            1,
        ));
    }

    #[test]
    fn draw_first_instance_reads_direct_and_indexed_and_zeroes_indirect() {
        assert_eq!(
            draw_first_instance(HalDraw::Direct {
                vertex_count: 3,
                instance_count: 1,
                first_vertex: 0,
                first_instance: 7,
            }),
            7
        );
        assert_eq!(
            draw_first_instance(HalDraw::Indexed {
                index_count: 3,
                instance_count: 1,
                first_index: 0,
                base_vertex: 0,
                first_instance: 9,
            }),
            9
        );
        // Indirect variants carry firstInstance inside a GPU-side buffer,
        // not observable at record time; GLES 3.1's indirect-draw structs
        // additionally have no baseInstance field at all, so this is always
        // 0 (see `GlesAdapter::supports_indirect_first_instance`).
        assert_eq!(draw_first_instance(HalDraw::Indirect { offset: 0 }), 0);
        assert_eq!(
            draw_first_instance(HalDraw::IndexedIndirect { offset: 0 }),
            0
        );
    }

    #[test]
    fn instance_step_offset_is_zero_for_first_instance_zero() {
        assert_eq!(instance_step_offset(32, 0).expect("no overflow"), 0);
        assert_eq!(instance_step_offset(0, 5).expect("no overflow"), 0);
    }

    #[test]
    fn instance_step_offset_multiplies_stride_by_first_instance() {
        // Mirrors Dawn's `offset += mFirstInstance * vertexBuffer.arrayStride`
        // (CommandBufferGL.cpp:259-261).
        assert_eq!(instance_step_offset(16, 3).expect("no overflow"), 48);
        assert_eq!(
            instance_step_offset(1, u32::MAX).expect("no overflow"),
            i64::from(u32::MAX)
        );
    }

    #[test]
    fn instance_step_offset_rejects_overflow() {
        let error = instance_step_offset(u64::MAX, 2).expect_err("must overflow u64 multiply");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "vertex buffer firstInstance offset exceeds GLES limit",
            }
        ));

        // Product fits in u64 (2^64 - 2^32) but not i64 (glow's
        // pointer-offset APIs take `i32`, but the intermediate byte offset
        // here is `i64`).
        let error = instance_step_offset(u64::from(u32::MAX) + 1, u32::MAX)
            .expect_err("must overflow i64 conversion");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "vertex buffer firstInstance offset exceeds GLES limit",
            }
        ));
    }

    #[test]
    fn resolve_bound_buffer_size_resolves_whole_size_sentinel() {
        // Whole-size (`u64::MAX`) resolves to `buffer_size - offset`.
        assert_eq!(
            resolve_bound_buffer_size(256, 0, u64::MAX).expect("whole size from zero offset"),
            256
        );
        assert_eq!(
            resolve_bound_buffer_size(272, 256, u64::MAX).expect("whole size from nonzero offset"),
            16
        );
        // `offset == buffer_size` is allowed and yields an empty range.
        assert_eq!(
            resolve_bound_buffer_size(256, 256, u64::MAX).expect("empty whole-size range"),
            0
        );
    }

    #[test]
    fn resolve_bound_buffer_size_passes_through_explicit_size() {
        assert_eq!(
            resolve_bound_buffer_size(256, 0, 16).expect("explicit size"),
            16
        );
        assert_eq!(
            resolve_bound_buffer_size(256, 64, 32).expect("explicit size at offset"),
            32
        );
    }

    #[test]
    fn resolve_bound_buffer_size_rejects_offset_past_buffer() {
        let error = resolve_bound_buffer_size(256, 257, u64::MAX)
            .expect_err("offset past buffer size must error");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "buffer binding offset exceeds buffer size",
            }
        ));
    }

    #[test]
    fn submit_compute_pass_binds_whole_size_storage_buffer_at_offset() {
        let Some(device) = gles_device_or_skip("GLES compute whole-size storage-buffer test")
        else {
            return;
        };
        // Buffer of 256 (a universally safe SSBO offset alignment) + 16 bytes
        // for a 4-element `uint` payload bound whole-size from offset 256.
        let buffer = device
            .create_buffer(
                256 + 16,
                crate::HalBufferUsage {
                    storage: true,
                    copy_src: true,
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES whole-size storage buffer creation must succeed");
        // Prefill the whole buffer so the pre-offset region is a known sentinel.
        buffer
            .write(0, &[0x11_u8; 256 + 16])
            .expect("prefilling GLES whole-size storage buffer must succeed");
        let pipeline = device
            .create_compute_pipeline(
                crate::HalShaderSource::Glsl {
                    source: "#version 310 es\n\
                             layout(local_size_x = 4) in;\n\
                             layout(std430, binding = 0) buffer Out { uint values[4]; } out_buf;\n\
                             void main() {\n\
                                 uint i = gl_GlobalInvocationID.x;\n\
                                 out_buf.values[i] = i + 1u;\n\
                             }\n"
                    .to_owned(),
                    stage: crate::HalShaderStage::Compute,
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: vec![crate::HalGlesBindingRemap::new(
                        0,
                        0,
                        crate::HalGlesBindingClass::StorageBuffer,
                        0,
                    )],
                    texture_metadata_ubo_binding: None,
                },
                "main",
                (4, 1, 1),
                &[HalDescriptorBinding {
                    group: 0,
                    binding: 0,
                    kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage),
                }],
            )
            .expect("GLES whole-size storage compute pipeline creation must succeed");

        device
            .queue()
            .submit_copies(&[HalCopy::ComputePass(crate::HalComputePass {
                pipeline: HalComputePipeline::Gles(pipeline),
                bind_buffers: vec![crate::HalBoundBuffer {
                    group: 0,
                    binding: 0,
                    metal_index: 0,
                    vertex_metal_index: None,
                    fragment_metal_index: None,
                    buffer: HalBuffer::Gles(buffer.clone()),
                    offset: 256,
                    // Whole-buffer-from-offset sentinel (previously rejected on
                    // GLES with "binding size exceeds GLES limit").
                    size: u64::MAX,
                }],
                bind_textures: Vec::new(),
                bind_samplers: Vec::new(),
                bind_external_textures: Vec::new(),
                immediate_data: Vec::new(),
                dispatch: HalComputeDispatch::Direct {
                    workgroups: (1, 1, 1),
                },
            })])
            .expect("GLES whole-size storage buffer dispatch must succeed");

        let bytes = buffer
            .read(0, 256 + 16)
            .expect("reading GLES whole-size storage buffer must succeed");
        // The pre-offset region is untouched; the shader wrote the payload at
        // byte 256, proving the whole-size binding resolved to offset 256.
        assert_eq!(&bytes[0..256], &[0x11_u8; 256]);
        let payload: Vec<u32> = bytes[256..272]
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(payload, vec![1, 2, 3, 4]);
    }

    #[test]
    fn submit_render_pass_binds_whole_size_uniform_buffer_at_offset() {
        let Some(device) = gles_device_or_skip("GLES render whole-size uniform-buffer test") else {
            return;
        };
        let target = device
            .create_texture(&crate::HalTextureDescriptor {
                dimension: crate::HalTextureDimension::D2,
                format: crate::HalTextureFormat::Rgba8Unorm,
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
                mip_level_count: 1,
                sample_count: 1,
                usage: crate::HalTextureUsage {
                    copy_src: true,
                    copy_dst: false,
                    texture_binding: false,
                    storage_binding: false,
                    render_attachment: true,
                    transient: false,
                },
            })
            .expect("GLES whole-size uniform render target creation must succeed");
        let readback = device
            .create_buffer(
                16,
                crate::HalBufferUsage {
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES whole-size uniform readback buffer creation must succeed");
        // 256 (safe UBO offset alignment) + one `vec4` bound whole-size from 256.
        let uniform = device
            .create_buffer(
                256 + 16,
                crate::HalBufferUsage {
                    uniform: true,
                    copy_dst: true,
                    ..crate::HalBufferUsage::default()
                },
            )
            .expect("GLES whole-size uniform buffer creation must succeed");
        let color_bytes = [0.25_f32, 0.5, 0.75, 1.0]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        uniform
            .write(256, &color_bytes)
            .expect("writing GLES whole-size uniform color must succeed");

        let pipeline = device
            .create_render_pipeline(
                crate::HalShaderSource::GlslStages {
                    vertex: "#version 310 es\n\
                             void main() {\n\
                                 vec2 pos = vec2(float((gl_VertexID & 1) << 2) - 1.0,\n\
                                                 float((gl_VertexID & 2) << 1) - 1.0);\n\
                                 gl_Position = vec4(pos, 0.0, 1.0);\n\
                             }\n"
                    .to_owned(),
                    fragment: Some(
                        "#version 310 es\n\
                         precision mediump float;\n\
                         layout(std140, binding = 0) uniform Params { vec4 color; } params;\n\
                         layout(location = 0) out vec4 frag_color;\n\
                         void main() { frag_color = params.color; }\n"
                            .to_owned(),
                    ),
                    combined_samplers: Vec::new(),
                    texture_metadata_slots: Vec::new(),
                    binding_remaps: vec![crate::HalGlesBindingRemap::new(
                        0,
                        0,
                        crate::HalGlesBindingClass::UniformBuffer,
                        0,
                    )],
                    texture_metadata_ubo_binding: None,
                },
                "main",
                Some("main"),
                &crate::HalRenderPipelineDescriptor {
                    sample_count: 1,
                    sample_mask: u32::MAX,
                    alpha_to_coverage_enabled: false,
                    color_targets: vec![Some(HalColorTargetState {
                        format: crate::HalTextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: 0xf,
                    })],
                    depth_stencil: None,
                    vertex_buffers: Vec::new(),
                    primitive_topology: crate::HalPrimitiveTopology::TriangleList,
                    front_face: HalFrontFace::Ccw,
                    cull_mode: HalCullMode::None,
                    unclipped_depth: false,
                    needs_frag_depth_range_push_constant: false,
                    user_immediate_size: 0,
                },
                &[HalDescriptorBinding {
                    group: 0,
                    binding: 0,
                    kind: HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Uniform),
                }],
            )
            .expect("GLES whole-size uniform render pipeline creation must succeed");

        let mut pass = render_pass(vec![Some(color_target_for(
            target.clone(),
            crate::HalTextureFormat::Rgba8Unorm,
            [0.0, 0.0, 0.0, 0.0],
        ))]);
        pass.pipeline = Some(HalRenderPipeline::Gles(pipeline));
        pass.bind_buffers = vec![crate::HalBoundBuffer {
            group: 0,
            binding: 0,
            metal_index: 0,
            vertex_metal_index: None,
            fragment_metal_index: None,
            buffer: HalBuffer::Gles(uniform),
            offset: 256,
            // Whole-buffer-from-offset sentinel (previously rejected on GLES).
            size: u64::MAX,
        }];
        pass.draw = Some(HalDraw::Direct {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        });

        device
            .queue()
            .submit_copies(&[
                HalCopy::RenderPass(pass),
                HalCopy::TextureToBuffer(HalBufferTextureCopy {
                    buffer: HalBuffer::Gles(readback.clone()),
                    buffer_layout: rgba8_slice_layout(0),
                    texture: HalTexture::Gles(target),
                    format: crate::HalTextureFormat::Rgba8Unorm,
                    aspect: crate::HalTextureAspect::All,
                    mip_level: 0,
                    origin: crate::HalOrigin3d { x: 0, y: 0, z: 0 },
                    extent: crate::HalExtent3d {
                        width: 2,
                        height: 2,
                        depth_or_array_layers: 1,
                    },
                }),
            ])
            .expect("GLES whole-size uniform render submit plus readback must succeed");

        assert_eq!(
            readback
                .read(0, 16)
                .expect("reading GLES whole-size uniform output must succeed"),
            [64, 128, 191, 255].repeat(4)
        );
    }
}
