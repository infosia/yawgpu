use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::format::{map_primitive_topology, map_vertex_format};
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

        self.inner
            .with_current_context(|gl| -> Result<(), HalError> {
                for copy in copies {
                    match copy {
                        HalCopy::Buffer(copy) => submit_buffer_copy(gl, copy)?,
                        HalCopy::BufferClear(clear) => submit_buffer_clear(gl, clear)?,
                        HalCopy::BufferToTexture(copy) => submit_buffer_to_texture(gl, copy)?,
                        HalCopy::TextureToBuffer(copy) => submit_texture_to_buffer(gl, copy)?,
                        HalCopy::TextureToTexture(copy) => submit_texture_to_texture(gl, copy)?,
                        HalCopy::ComputePass(pass) => submit_compute_pass(gl, pass)?,
                        HalCopy::RenderPass(pass) => submit_render_pass(gl, pass)?,
                        #[cfg(feature = "tiled")]
                        HalCopy::SubpassRenderPass(_) => {
                            return Err(HalError::BufferOperationFailed {
                                backend: BACKEND,
                                message:
                                    "GLES backend supports only buffer, texture, compute, and render commands in P15.5",
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
                gl.dispatch_compute(workgroups.0, workgroups.1, workgroups.2);
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
        #[cfg(feature = "tiled")]
        HalDescriptorBindingKind::Buffer(HalBufferBindingKind::InputAttachment) => {
            Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "input attachments are not valid buffer bindings",
            })
        }
        HalDescriptorBindingKind::Texture
        | HalDescriptorBindingKind::StorageTexture { .. }
        | HalDescriptorBindingKind::Sampler => Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "texture and sampler descriptors are not valid buffer bindings",
        }),
    }
}

fn submit_render_pass(gl: &glow::Context, pass: &HalRenderPass) -> Result<(), HalError> {
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
    run_render_draw(gl, pass, pipeline, vao)
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
    if pass.color_targets.len() > 1 {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pass supports at most one color attachment",
        });
    }
    if pass
        .color_targets
        .iter()
        .any(|target| target.resolve_target.is_some())
    {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pass does not support multisample/resolve",
        });
    }
    let color_target = pass
        .color_targets
        .first()
        .as_ref()
        .map(|target| match &target.texture {
            HalTexture::Gles(texture) => Ok(texture),
            _ => Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "render pass color target is not a GLES texture",
            }),
        })
        .transpose()?;
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
    let size_texture =
        color_target
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
        let fbo = gl
            .create_framebuffer()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateFramebuffer failed (render)",
            })?;
        gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(fbo));
        if let Some(target_texture) = color_target {
            let color_texture = target_texture.raw_or_err()?;
            let meta = target_texture.meta();
            if meta.target != glow::TEXTURE_2D {
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
                gl.delete_framebuffer(fbo);
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES render pass supports only 2D color attachments",
                });
            }
            gl.framebuffer_texture_2d(
                glow::DRAW_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D,
                Some(color_texture),
                0,
            );
            gl.draw_buffers(&[glow::COLOR_ATTACHMENT0]);
        } else {
            gl.draw_buffers(&[]);
        }
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
        if let Some(color) = pass.color_targets.first() {
            let [r, g, b, a] = color.clear_color;
            gl.clear_color(r as f32, g as f32, b as f32, a as f32);
            if matches!(color.load_op, HalRenderLoadOp::Clear) {
                clear_mask |= glow::COLOR_BUFFER_BIT;
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
) -> Result<(), HalError> {
    let program = pipeline.raw_or_err()?;
    unsafe {
        gl.use_program(Some(program));
    }
    bind_render_buffers(gl, pass, pipeline)?;
    bind_vertex_buffers(gl, pass, pipeline, vao)?;
    if let Some(draw) = pass.draw {
        apply_raster_state(gl, pipeline.front_face(), pipeline.cull_mode());
        apply_color_target_state(gl, pipeline.color_target(), pass.blend_constant);
        apply_stencil_state(gl, pipeline.depth_stencil(), pass.stencil_reference)?;
        if let Some(location) = pipeline.first_instance_location() {
            set_first_instance_uniform(gl, location, draw);
        }
        let topology = map_primitive_topology(pipeline.primitive_topology());
        run_gles_draw(gl, pass, topology, draw)?;
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

fn set_first_instance_uniform(gl: &glow::Context, location: &glow::UniformLocation, draw: HalDraw) {
    let first_instance = match draw {
        HalDraw::Direct { first_instance, .. } | HalDraw::Indexed { first_instance, .. } => {
            first_instance
        }
        HalDraw::Indirect { .. } | HalDraw::IndexedIndirect { .. } => 0,
    };
    unsafe {
        gl.uniform_1_u32(Some(location), first_instance);
    }
}

fn run_gles_draw(
    gl: &glow::Context,
    pass: &HalRenderPass,
    topology: u32,
    draw: HalDraw,
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
            if base_vertex != 0 {
                return Err(HalError::BufferOperationFailed {
                    backend: BACKEND,
                    message: "GLES indexed draw requires baseVertex 0",
                });
            }
            let (index_type, index_offset) = bind_gles_index_buffer(gl, pass, first_index)?;
            let index_count = i32_from_u32(index_count, "draw indexCount exceeds GLES limit")?;
            let instance_count =
                i32_from_u32(instance_count, "draw instanceCount exceeds GLES limit")?;
            unsafe {
                if instance_count == 1 && first_instance == 0 {
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
        let Some(layout) = pipeline.vertex_buffers().get(layout_index) else {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "vertex buffer binding is missing from pipeline layout",
            });
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
                gl.vertex_attrib_pointer_f32(
                    attribute.shader_location,
                    format.components,
                    format.ty,
                    format.normalized,
                    stride,
                    attribute_offset,
                );
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

    ensure_2d_copy(copy.extent.depth_or_array_layers, copy.origin.z)?;
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

    unsafe {
        gl.bind_buffer(glow::PIXEL_UNPACK_BUFFER, Some(buffer));
        gl.bind_texture(meta.target, Some(texture));
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, row_pixels);
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
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

    ensure_2d_copy(copy.extent.depth_or_array_layers, copy.origin.z)?;
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
    let buffer_offset = u32_from_u64(
        copy.buffer_layout.offset,
        "texture-to-buffer offset exceeds GLES limit",
    )?;

    unsafe {
        let framebuffer = gl
            .create_framebuffer()
            .map_err(|_| HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "glCreateFramebuffer failed",
            })?;
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(framebuffer));
        gl.framebuffer_texture_2d(
            glow::READ_FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            meta.target,
            Some(texture),
            mip_level,
        );
        gl.read_buffer(glow::COLOR_ATTACHMENT0);
        if gl.check_framebuffer_status(glow::READ_FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
            gl.delete_framebuffer(framebuffer);
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "framebuffer incomplete for texture-to-buffer copy",
            });
        }
        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(buffer));
        gl.pixel_store_i32(glow::PACK_ROW_LENGTH, row_pixels);
        gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
        gl.read_pixels(
            x,
            y,
            width,
            height,
            meta.format.format,
            meta.format.ty,
            glow::PixelPackData::BufferOffset(buffer_offset),
        );
        gl.pixel_store_i32(glow::PACK_ROW_LENGTH, 0);
        gl.pixel_store_i32(glow::PACK_ALIGNMENT, 4);
        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
        gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        gl.delete_framebuffer(framebuffer);
    }

    Ok(())
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

    if !supports_copy_image(gl) {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GL_EXT_copy_image required for texture-to-texture copies; not supported by this GLES driver",
        });
    }

    ensure_2d_copy(copy.extent.depth_or_array_layers, copy.source_origin.z)?;
    ensure_2d_copy(copy.extent.depth_or_array_layers, copy.destination_origin.z)?;
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

    unsafe {
        gl.copy_image_sub_data(
            source_texture,
            source.meta().target,
            source_mip_level,
            source_x,
            source_y,
            0,
            destination_texture,
            destination.meta().target,
            destination_mip_level,
            destination_x,
            destination_y,
            0,
            width,
            height,
            1,
        );
    }

    Ok(())
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

fn ensure_2d_copy(depth_or_array_layers: u32, z: u32) -> Result<(), HalError> {
    if depth_or_array_layers != 1 || z != 0 {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "only 2D texture copies are supported on GLES (P15.3)",
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
    fn ensure_2d_copy_accepts_layer_one_z_zero_only() {
        assert!(ensure_2d_copy(1, 0).is_ok());
        assert!(matches!(
            ensure_2d_copy(2, 0),
            Err(HalError::BufferOperationFailed {
                backend: "gles",
                ..
            })
        ));
        assert!(matches!(
            ensure_2d_copy(1, 1),
            Err(HalError::BufferOperationFailed {
                backend: "gles",
                ..
            })
        ));
        assert!(matches!(
            ensure_2d_copy(0, 0),
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
}
