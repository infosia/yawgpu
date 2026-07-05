use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::format::{color_clear_kind, map_primitive_topology, map_vertex_format, GlesClearKind};
use super::BACKEND;
use crate::format::{format_has_depth_aspect, format_has_stencil_aspect};
use crate::{
    HalBlendFactor, HalBlendOperation, HalBuffer, HalBufferBindingKind, HalBufferClear,
    HalBufferCopy, HalBufferTextureCopy, HalColorTargetState, HalCompareFunction,
    HalComputeDispatch, HalComputePass, HalComputePipeline, HalCopy, HalCullMode,
    HalDepthStencilState, HalDescriptorBinding, HalDescriptorBindingKind, HalDraw, HalError,
    HalFrontFace, HalIndexFormat, HalRenderLoadOp, HalRenderPass, HalRenderPipeline,
    HalStencilFaceState, HalStencilOperation, HalTexture, HalTextureCopy, HalVertexStepMode,
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
        self.inner
            .with_current_context(|gl| -> Result<(), HalError> {
                for copy in copies {
                    match copy {
                        HalCopy::Buffer(copy) => submit_buffer_copy(gl, copy)?,
                        HalCopy::BufferClear(clear) => submit_buffer_clear(gl, clear)?,
                        HalCopy::ClearTexture(_) => {}
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
                        HalCopy::ComputePass(pass) => submit_compute_pass(gl, pass)?,
                        HalCopy::RenderPass(pass) => {
                            submit_render_pass(gl, pass, supports_base_vertex)?;
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

fn submit_compute_pass(gl: &glow::Context, pass: &HalComputePass) -> Result<(), HalError> {
    let HalComputePipeline::Gles(pipeline) = &pass.pipeline else {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "compute pass pipeline is not a GLES pipeline",
        });
    };
    if !pass.bind_textures.is_empty() {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES compute does not support texture bindings",
        });
    }
    let program = pipeline.raw_or_err()?;
    let bindings = pass
        .bind_buffers
        .iter()
        .map(|bound| {
            if bound.group != 0 {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES compute supports only bind group 0",
                });
            }
            let HalBuffer::Gles(buffer) = &bound.buffer else {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "compute pass binding is not a GLES buffer",
                });
            };
            let target = binding_target(pipeline.bindings(), bound.binding)?;
            let buffer = buffer.raw_or_err()?;
            let offset =
                i32::try_from(bound.offset).map_err(|_| HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "compute buffer binding offset exceeds GLES limit",
                })?;
            let size = i32::try_from(bound.size).map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "compute buffer binding size exceeds GLES limit",
            })?;
            Ok((target, bound.binding, buffer, offset, size))
        })
        .collect::<Result<Vec<_>, _>>()?;
    unsafe {
        gl.use_program(Some(program));
        for (target, binding, buffer, offset, size) in bindings {
            gl.bind_buffer_range(target, binding, Some(buffer), offset, size);
        }
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

fn binding_target(bindings: &[HalDescriptorBinding], binding: u32) -> Result<u32, HalError> {
    let descriptor = bindings
        .iter()
        .find(|descriptor| descriptor.group == 0 && descriptor.binding == binding)
        .ok_or(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "buffer binding is missing from pipeline layout",
        })?;
    match descriptor.kind {
        HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Uniform) => Ok(glow::UNIFORM_BUFFER),
        HalDescriptorBindingKind::Buffer(HalBufferBindingKind::Storage) => {
            Ok(glow::SHADER_STORAGE_BUFFER)
        }
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

fn submit_render_pass(
    gl: &glow::Context,
    pass: &HalRenderPass,
    supports_base_vertex: bool,
) -> Result<(), HalError> {
    reject_render_texture_sampler_bindings(pass)?;
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
    run_render_draw(gl, pass, pipeline, vao, supports_base_vertex)
}

fn reject_render_texture_sampler_bindings(pass: &HalRenderPass) -> Result<(), HalError> {
    reject_render_texture_sampler_binding_counts(pass.bind_textures.len(), pass.bind_samplers.len())
}

fn reject_render_texture_sampler_binding_counts(
    texture_count: usize,
    sampler_count: usize,
) -> Result<(), HalError> {
    if texture_count != 0 || sampler_count != 0 {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pass does not support texture/sampler bindings",
        });
    }
    Ok(())
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
            self.gl.color_mask(true, true, true, true);
            self.gl.disable(glow::BLEND);
            self.gl.disable(glow::STENCIL_TEST);
            self.gl.memory_barrier(glow::ALL_BARRIER_BITS);
        }
    }
}

fn create_render_fbo(
    gl: &glow::Context,
    pass: &HalRenderPass,
) -> Result<glow::Framebuffer, HalError> {
    if pass
        .color_targets
        .iter()
        .flatten()
        .any(|target| target.resolve_target.is_some())
    {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pass does not support multisample/resolve",
        });
    }
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
            if texture.meta().target != glow::TEXTURE_2D {
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
                message: "GLES render pass color attachment count exceeds the driver draw-buffer limit",
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
            let Some((_, _, color_texture)) = slot else {
                // Sparse slot: nothing attached, fragment output discarded.
                draw_buffer_list.push(glow::NONE);
                continue;
            };
            let attachment = glow::COLOR_ATTACHMENT0 + index as u32;
            gl.framebuffer_texture_2d(
                glow::DRAW_FRAMEBUFFER,
                attachment,
                glow::TEXTURE_2D,
                Some(*color_texture),
                0,
            );
            draw_buffer_list.push(attachment);
        }
        gl.draw_buffers(&draw_buffer_list);
        if let (Some(attachment), Some(target_texture)) =
            (&pass.depth_stencil_attachment, depth_stencil_target)
        {
            let depth_stencil_texture = target_texture.raw_or_err()?;
            let meta = target_texture.meta();
            if meta.target != glow::TEXTURE_2D {
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
                gl.delete_framebuffer(fbo);
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES render pass supports only 2D depth-stencil attachments",
                });
            }
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
            gl.framebuffer_texture_2d(
                glow::DRAW_FRAMEBUFFER,
                attachment_point,
                glow::TEXTURE_2D,
                Some(depth_stencil_texture),
                0,
            );
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

fn run_render_draw(
    gl: &glow::Context,
    pass: &HalRenderPass,
    pipeline: &super::pipeline::GlesRenderPipeline,
    vao: glow::VertexArray,
    supports_base_vertex: bool,
) -> Result<(), HalError> {
    let program = pipeline.raw_or_err()?;
    unsafe {
        gl.use_program(Some(program));
    }
    bind_render_buffers(gl, pass, pipeline)?;
    let first_instance = pass.draw.map(draw_first_instance).unwrap_or(0);
    bind_vertex_buffers(gl, pass, pipeline, vao, first_instance)?;
    if let Some(draw) = pass.draw {
        apply_raster_state(gl, pipeline.front_face(), pipeline.cull_mode());
        apply_color_target_state(gl, pipeline.color_target(), pass.blend_constant);
        apply_stencil_state(gl, pipeline.depth_stencil(), pass.stencil_reference)?;
        if let Some(location) = pipeline.first_instance_location() {
            set_first_instance_uniform(gl, location, draw);
        }
        let topology = map_primitive_topology(pipeline.primitive_topology());
        run_gles_draw(gl, pass, topology, draw, supports_base_vertex)?;
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

fn bind_render_buffers(
    gl: &glow::Context,
    pass: &HalRenderPass,
    pipeline: &super::pipeline::GlesRenderPipeline,
) -> Result<(), HalError> {
    for bound in &pass.bind_buffers {
        if bound.group != 0 {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "GLES render supports only bind group 0",
            });
        }
        let HalBuffer::Gles(buffer) = &bound.buffer else {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "render pass binding is not a GLES buffer",
            });
        };
        let target = binding_target(pipeline.bindings(), bound.binding)?;
        let buffer = buffer.raw_or_err()?;
        let offset = i32::try_from(bound.offset).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "render buffer binding offset exceeds GLES limit",
        })?;
        let size = i32::try_from(bound.size).map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "render buffer binding size exceeds GLES limit",
        })?;
        unsafe {
            gl.bind_buffer_range(target, bound.binding, Some(buffer), offset, size);
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
                    gl.vertex_attrib_pointer_f32(
                        attribute.shader_location,
                        format.components,
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
    let row_pixels = pixels_per_row(
        copy.buffer_layout.bytes_per_row,
        meta.format.bytes_per_pixel,
    )?;
    let mip_level = i32_from_u32(copy.mip_level, "texture mip level exceeds GLES limit")?;
    let x = i32_from_u32(copy.origin.x, "texture x origin exceeds GLES limit")?;
    let y = i32_from_u32(copy.origin.y, "texture y origin exceeds GLES limit")?;
    let width = i32_from_u32(copy.extent.width, "texture copy width exceeds GLES limit")?;
    let height = i32_from_u32(copy.extent.height, "texture copy height exceeds GLES limit")?;
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
    let staging_len = if copy.extent.height == 0 || copy.extent.width == 0 {
        0usize
    } else {
        let tail_row_bytes = u64::from(copy.extent.width) * u64::from(meta.format.bytes_per_pixel);
        let full_rows_bytes = u64::from(copy.extent.height - 1)
            .checked_mul(u64::from(copy.buffer_layout.bytes_per_row))
            .ok_or(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "texture-to-buffer slice size exceeds GLES limit",
            })?;
        usize::try_from(full_rows_bytes.checked_add(tail_row_bytes).ok_or(
            HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "texture-to-buffer slice size exceeds GLES limit",
            },
        )?)
        .map_err(|_| HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture-to-buffer slice size exceeds host limit",
        })?
    };

    // Precompute the (layer, buffer offset) pair for every slice so that all
    // fallible arithmetic happens before any GL state is touched.
    let image_stride =
        u64::from(copy.buffer_layout.bytes_per_row) * u64::from(copy.buffer_layout.rows_per_image);
    let mut slices = Vec::with_capacity(copy.extent.depth_or_array_layers as usize);
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
        // The staged path passes the offset to `glBufferSubData` (i32); the
        // pack-buffer path passes it to `glReadPixels` (u32). Validate the
        // stricter bound up front so the copy loop below cannot fail.
        if use_client_staging {
            i32_from_u64(slice_offset, "texture-to-buffer offset exceeds GLES limit")?;
        } else {
            u32_from_u64(slice_offset, "texture-to-buffer offset exceeds GLES limit")?;
        }
        slices.push((layer, slice_offset));
    }

    unsafe {
        let framebuffer = gl
            .create_framebuffer()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateFramebuffer failed",
            })?;
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(framebuffer));
        gl.read_buffer(glow::COLOR_ATTACHMENT0);
        if !use_client_staging {
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(buffer));
        }
        gl.pixel_store_i32(glow::PACK_ROW_LENGTH, row_pixels);
        gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
        let mut result = Ok(());
        for (layer, slice_offset) in slices {
            if uses_layered_target {
                gl.framebuffer_texture_layer(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    Some(texture),
                    mip_level,
                    layer,
                );
            } else {
                gl.framebuffer_texture_2d(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    meta.target,
                    Some(texture),
                    mip_level,
                );
            }
            if gl.check_framebuffer_status(glow::READ_FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
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
                    y,
                    width,
                    height,
                    meta.format.format,
                    meta.format.ty,
                    glow::PixelPackData::Slice(&mut staging),
                );
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
                gl.buffer_sub_data_u8_slice(glow::COPY_WRITE_BUFFER, slice_offset as i32, &staging);
                gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            } else {
                gl.read_pixels(
                    x,
                    y,
                    width,
                    height,
                    meta.format.format,
                    meta.format.ty,
                    glow::PixelPackData::BufferOffset(slice_offset as u32),
                );
            }
        }
        gl.pixel_store_i32(glow::PACK_ROW_LENGTH, 0);
        gl.pixel_store_i32(glow::PACK_ALIGNMENT, 4);
        if !use_client_staging {
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
        }
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        gl.delete_framebuffer(framebuffer);
        result
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
        || unsafe { gles_version_at_least_3_2(&gl.get_parameter_string(glow::VERSION)) }
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
            binding_target(&bindings, 0).expect("uniform binding"),
            glow::UNIFORM_BUFFER
        );
        assert_eq!(
            binding_target(&bindings, 1).expect("storage binding"),
            glow::SHADER_STORAGE_BUFFER
        );
        let missing = binding_target(&bindings, 2).expect_err("missing binding");
        assert!(matches!(
            missing,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "buffer binding is missing from pipeline layout",
            }
        ));
    }

    #[test]
    fn reject_render_texture_sampler_bindings_rejects_texture_or_sampler_counts() {
        assert!(reject_render_texture_sampler_binding_counts(0, 0).is_ok());
        let texture = reject_render_texture_sampler_binding_counts(1, 0)
            .expect_err("texture binding must be rejected");
        assert!(matches!(
            texture,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "GLES render pass does not support texture/sampler bindings",
            }
        ));
        let sampler = reject_render_texture_sampler_binding_counts(0, 1)
            .expect_err("sampler binding must be rejected");
        assert!(matches!(
            sampler,
            HalError::BufferOperationFailed {
                backend: "gles",
                message: "GLES render pass does not support texture/sampler bindings",
            }
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
}
