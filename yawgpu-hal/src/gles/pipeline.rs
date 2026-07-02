use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::format::map_vertex_format;
use super::{rebuild_hal_error, BACKEND};
use crate::{
    HalColorTargetState, HalCullMode, HalDepthStencilState, HalDescriptorBinding, HalError,
    HalFrontFace, HalPrimitiveTopology, HalRenderPipelineDescriptor, HalTextureFormat,
    HalVertexBufferLayout,
};

/// The GLSL uniform name Tint's GLSL writer emits for the first element of
/// its "immediate data" emulation array when `first_instance_offset` is
/// requested (`yawgpu-core`'s `tint_bindings_for_glsl`/`generate_glsl`
/// request offset `0`, and GLES has no other internal immediate sharing the
/// struct, so element `0` is always `tint_first_instance`). See
/// `tint_shim.h`'s `yawgpu_tint_generate_glsl` docs and the
/// `generate_glsl_first_instance_offset_only_applied_when_requested` pin in
/// `yawgpu-tint` for the exact GLSL text this name is queried against (F2,
/// specs/tracking/tint-integration-refactor.md slice R6).
const FIRST_INSTANCE_UNIFORM_NAME: &str = "tint_immediates[0]";

struct GlesComputePipelineInner {
    device: Arc<GlesDeviceInner>,
    program: Result<glow::Program, HalError>,
    workgroup_size: (u32, u32, u32),
    bindings: Vec<HalDescriptorBinding>,
}

impl Drop for GlesComputePipelineInner {
    fn drop(&mut self) {
        if let Ok(program) = self.program.as_ref() {
            let program = *program;
            let _ = self.device.with_current_context(|gl| unsafe {
                gl.delete_program(program);
            });
        }
    }
}

/// Stores GLES compute pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesComputePipeline {
    inner: Arc<GlesComputePipelineInner>,
}

// SAFETY: `GlesComputePipeline` owns only GL object names and uses the owning
// device inner to serialize all GL access.
unsafe impl Send for GlesComputePipeline {}
// SAFETY: See the `Send` impl; shared access is synchronized by the device.
unsafe impl Sync for GlesComputePipeline {}

impl std::fmt::Debug for GlesComputePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesComputePipeline")
            .field("workgroup_size", &self.inner.workgroup_size)
            .field("bindings", &self.inner.bindings)
            .finish()
    }
}

impl GlesComputePipeline {
    pub(super) fn new(
        device: Arc<GlesDeviceInner>,
        source: String,
        workgroup_size: (u32, u32, u32),
        bindings: &[HalDescriptorBinding],
    ) -> Result<Self, HalError> {
        let program = build_compute_program(&device, &source)?;
        Ok(Self {
            inner: Arc::new(GlesComputePipelineInner {
                device,
                program: Ok(program),
                workgroup_size,
                bindings: bindings.to_vec(),
            }),
        })
    }

    pub(super) fn raw_or_err(&self) -> Result<glow::Program, HalError> {
        self.inner
            .program
            .as_ref()
            .copied()
            .map_err(rebuild_hal_error)
    }

    #[must_use]
    pub(super) fn bindings(&self) -> &[HalDescriptorBinding] {
        &self.inner.bindings
    }
}

struct GlesRenderPipelineInner {
    device: Arc<GlesDeviceInner>,
    program: Result<glow::Program, HalError>,
    vertex_buffers: Vec<HalVertexBufferLayout>,
    color_target: Option<HalColorTargetState>,
    depth_stencil: Option<HalDepthStencilState>,
    primitive_topology: HalPrimitiveTopology,
    front_face: HalFrontFace,
    cull_mode: HalCullMode,
    bindings: Vec<HalDescriptorBinding>,
    first_instance_location: Option<glow::UniformLocation>,
}

impl Drop for GlesRenderPipelineInner {
    fn drop(&mut self) {
        if let Ok(program) = self.program.as_ref() {
            let program = *program;
            let _ = self.device.with_current_context(|gl| unsafe {
                gl.delete_program(program);
            });
        }
    }
}

/// Stores GLES render pipeline data used by validation and backend submission.
#[derive(Clone)]
pub struct GlesRenderPipeline {
    inner: Arc<GlesRenderPipelineInner>,
}

// SAFETY: `GlesRenderPipeline` owns only GL object names and uses the owning
// device inner to serialize all GL access.
unsafe impl Send for GlesRenderPipeline {}
// SAFETY: See the `Send` impl; shared access is synchronized by the device.
unsafe impl Sync for GlesRenderPipeline {}

impl std::fmt::Debug for GlesRenderPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesRenderPipeline")
            .field("vertex_buffers", &self.inner.vertex_buffers)
            .field("primitive_topology", &self.inner.primitive_topology)
            .field("bindings", &self.inner.bindings)
            .finish()
    }
}

impl GlesRenderPipeline {
    pub(super) fn new(
        device: Arc<GlesDeviceInner>,
        vertex_source: String,
        fragment_source: Option<String>,
        descriptor: HalRenderPipelineDescriptor,
        bindings: &[HalDescriptorBinding],
    ) -> Result<Self, HalError> {
        validate_render_pipeline_descriptor(&descriptor)?;
        let (program, first_instance_location) =
            build_render_program(&device, &vertex_source, fragment_source.as_deref())?;
        Ok(Self {
            inner: Arc::new(GlesRenderPipelineInner {
                device,
                program: Ok(program),
                vertex_buffers: descriptor.vertex_buffers,
                color_target: descriptor.color_targets.iter().copied().flatten().next(),
                depth_stencil: descriptor.depth_stencil,
                primitive_topology: descriptor.primitive_topology,
                front_face: descriptor.front_face,
                cull_mode: descriptor.cull_mode,
                bindings: bindings.to_vec(),
                first_instance_location,
            }),
        })
    }

    pub(super) fn raw_or_err(&self) -> Result<glow::Program, HalError> {
        self.inner
            .program
            .as_ref()
            .copied()
            .map_err(rebuild_hal_error)
    }

    #[must_use]
    pub(super) fn vertex_buffers(&self) -> &[HalVertexBufferLayout] {
        &self.inner.vertex_buffers
    }

    #[must_use]
    pub(super) fn color_target(&self) -> Option<HalColorTargetState> {
        self.inner.color_target
    }

    #[must_use]
    pub(super) fn depth_stencil(&self) -> Option<HalDepthStencilState> {
        self.inner.depth_stencil
    }

    #[must_use]
    pub(super) fn primitive_topology(&self) -> HalPrimitiveTopology {
        self.inner.primitive_topology
    }

    #[must_use]
    pub(super) fn front_face(&self) -> HalFrontFace {
        self.inner.front_face
    }

    #[must_use]
    pub(super) fn cull_mode(&self) -> HalCullMode {
        self.inner.cull_mode
    }

    #[must_use]
    pub(super) fn bindings(&self) -> &[HalDescriptorBinding] {
        &self.inner.bindings
    }

    #[must_use]
    pub(super) fn first_instance_location(&self) -> Option<&glow::UniformLocation> {
        self.inner.first_instance_location.as_ref()
    }
}

fn build_compute_program(
    device: &Arc<GlesDeviceInner>,
    source: &str,
) -> Result<glow::Program, HalError> {
    device
        .with_current_context(|gl| unsafe {
            let shader = gl.create_shader(glow::COMPUTE_SHADER).map_err(|_| {
                HalError::ShaderCompilationFailed {
                    backend: BACKEND,
                    message: "glCreateShader failed".to_owned(),
                }
            })?;
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let log = gl.get_shader_info_log(shader);
                gl.delete_shader(shader);
                return Err(HalError::ShaderCompilationFailed {
                    backend: BACKEND,
                    message: format!("GLES compute shader compilation failed: {log}"),
                });
            }

            let program = match gl.create_program() {
                Ok(program) => program,
                Err(_) => {
                    gl.delete_shader(shader);
                    return Err(HalError::ShaderCompilationFailed {
                        backend: BACKEND,
                        message: "glCreateProgram failed".to_owned(),
                    });
                }
            };
            gl.attach_shader(program, shader);
            gl.link_program(program);
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(HalError::ShaderCompilationFailed {
                    backend: BACKEND,
                    message: format!("GLES compute program link failed: {log}"),
                });
            }
            Ok(program)
        })
        .and_then(|result| result)
}

fn validate_render_pipeline_descriptor(
    descriptor: &HalRenderPipelineDescriptor,
) -> Result<(), HalError> {
    if descriptor.color_targets.iter().flatten().count() > 1 {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pipeline supports at most one color target",
        });
    }
    if descriptor
        .color_targets
        .iter()
        .position(Option::is_some)
        .is_some_and(|slot| slot > 0)
    {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pipeline does not support a color attachment at a non-zero slot",
        });
    }
    if descriptor.sample_count > 1 {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pass does not support multisample/resolve",
        });
    }
    if descriptor.sample_mask != u32::MAX {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pipeline does not support non-default multisample mask",
        });
    }
    if descriptor.alpha_to_coverage_enabled {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pipeline does not support alpha-to-coverage",
        });
    }
    if descriptor.unclipped_depth {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pipeline does not support unclipped depth",
        });
    }
    if let Some(target) = descriptor.color_targets.iter().flatten().next() {
        if !matches!(
            target.format,
            HalTextureFormat::Rgba8Unorm | HalTextureFormat::Bgra8Unorm
        ) {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message:
                    "GLES render pipeline supports only RGBA8Unorm or BGRA8Unorm color targets",
            });
        }
        if target.blend.is_some_and(gles_blend_uses_dual_source) {
            return Err(HalError::BufferOperationFailed {
                backend: BACKEND,
                message: "GLES render pipeline does not support dual-source blend factors",
            });
        }
    }
    if !descriptor.color_targets.iter().any(Option::is_some) && descriptor.depth_stencil.is_none() {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pipeline requires a color target or depth-stencil state",
        });
    }
    for layout in &descriptor.vertex_buffers {
        for attribute in &layout.attributes {
            map_vertex_format(attribute.format)?;
        }
    }
    Ok(())
}

fn gles_blend_uses_dual_source(blend: crate::HalBlendState) -> bool {
    gles_blend_component_uses_dual_source(blend.color)
        || gles_blend_component_uses_dual_source(blend.alpha)
}

fn gles_blend_component_uses_dual_source(component: crate::HalBlendComponent) -> bool {
    gles_blend_factor_is_dual_source(component.src_factor)
        || gles_blend_factor_is_dual_source(component.dst_factor)
}

fn gles_blend_factor_is_dual_source(factor: crate::HalBlendFactor) -> bool {
    matches!(
        factor,
        crate::HalBlendFactor::Src1
            | crate::HalBlendFactor::OneMinusSrc1
            | crate::HalBlendFactor::Src1Alpha
            | crate::HalBlendFactor::OneMinusSrc1Alpha
    )
}

fn build_render_program(
    device: &Arc<GlesDeviceInner>,
    vertex_source: &str,
    fragment_source: Option<&str>,
) -> Result<(glow::Program, Option<glow::UniformLocation>), HalError> {
    device
        .with_current_context(|gl| unsafe {
            let vertex = compile_shader(gl, glow::VERTEX_SHADER, vertex_source, "vertex")?;
            let fragment = match fragment_source
                .map(|source| compile_shader(gl, glow::FRAGMENT_SHADER, source, "fragment"))
                .transpose()
            {
                Ok(fragment) => fragment,
                Err(error) => {
                    gl.delete_shader(vertex);
                    return Err(error);
                }
            };
            let program = match gl.create_program() {
                Ok(program) => program,
                Err(_) => {
                    gl.delete_shader(vertex);
                    if let Some(fragment) = fragment {
                        gl.delete_shader(fragment);
                    }
                    return Err(HalError::ShaderCompilationFailed {
                        backend: BACKEND,
                        message: "glCreateProgram failed".to_owned(),
                    });
                }
            };
            gl.attach_shader(program, vertex);
            if let Some(fragment) = fragment {
                gl.attach_shader(program, fragment);
            }
            gl.link_program(program);
            gl.detach_shader(program, vertex);
            if let Some(fragment) = fragment {
                gl.detach_shader(program, fragment);
            }
            gl.delete_shader(vertex);
            if let Some(fragment) = fragment {
                gl.delete_shader(fragment);
            }
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(HalError::ShaderCompilationFailed {
                    backend: BACKEND,
                    message: format!("GLES render program link failed: {log}"),
                });
            }
            // Tint's GLSL writer emits buffer blocks with an explicit
            // `layout(binding = N)` matching the WGSL @binding number
            // directly (see `yawgpu-core`'s `tint_bindings_for_glsl`, which
            // supplies an identity remap so this always holds) -- GLES 3.1
            // core honors that at link time, so no
            // `glUniformBlockBinding`/`glShaderStorageBlockBinding` runtime
            // remap is needed (F2, specs/tracking/tint-integration-refactor.md
            // slice R6; this replaced a naga-era `_block_N`-suffix name
            // parser that no longer matched Tint's block-naming scheme and
            // would have silently mis-bound buffers).
            //
            // First-instance offset: when the vertex stage reads
            // `@builtin(instance_index)`, `yawgpu-core` requests Tint's
            // `first_instance_offset`, which emits a single
            // `layout(location = 0) uniform uint tint_immediates[1]` array
            // (see `tint_shim.h`'s `yawgpu_tint_generate_glsl` docs). Query
            // its location by the GLSL array-element name; `queue.rs`
            // writes the WebGPU `firstInstance` draw parameter there before
            // each draw.
            let first_instance_location =
                gl.get_uniform_location(program, FIRST_INSTANCE_UNIFORM_NAME);
            Ok((program, first_instance_location))
        })
        .and_then(|result| result)
}

fn compile_shader(
    gl: &glow::Context,
    stage: u32,
    source: &str,
    label: &'static str,
) -> Result<glow::Shader, HalError> {
    unsafe {
        let shader = gl
            .create_shader(stage)
            .map_err(|_| HalError::ShaderCompilationFailed {
                backend: BACKEND,
                message: "glCreateShader failed".to_owned(),
            })?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            return Err(HalError::ShaderCompilationFailed {
                backend: BACKEND,
                message: format!("GLES {label} shader compilation failed: {log}"),
            });
        }
        Ok(shader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn color_target() -> HalColorTargetState {
        HalColorTargetState {
            format: HalTextureFormat::Rgba8Unorm,
            blend: None,
            write_mask: 0xf,
        }
    }

    fn render_descriptor(
        color_targets: Vec<Option<HalColorTargetState>>,
    ) -> HalRenderPipelineDescriptor {
        HalRenderPipelineDescriptor {
            sample_count: 1,
            sample_mask: u32::MAX,
            alpha_to_coverage_enabled: false,
            color_targets,
            depth_stencil: None,
            vertex_buffers: Vec::new(),
            primitive_topology: crate::HalPrimitiveTopology::TriangleList,
            front_face: crate::HalFrontFace::Ccw,
            cull_mode: crate::HalCullMode::None,
            unclipped_depth: false,
            needs_frag_depth_range_push_constant: false,
            user_immediate_size: 0,
        }
    }

    #[test]
    fn validate_render_pipeline_descriptor_rejects_non_zero_color_slot() {
        assert!(validate_render_pipeline_descriptor(&render_descriptor(vec![
            Some(color_target())
        ]))
        .is_ok());
        assert!(validate_render_pipeline_descriptor(&render_descriptor(vec![
            Some(color_target()),
            None,
        ]))
        .is_ok());

        let error = validate_render_pipeline_descriptor(&render_descriptor(vec![
            None,
            Some(color_target()),
        ]))
        .expect_err("GLES must reject a real color target at slot 1");
        assert!(matches!(
            error,
            HalError::BufferOperationFailed {
                backend: "gles",
                message:
                    "GLES render pipeline does not support a color attachment at a non-zero slot",
            }
        ));
    }

    /// F2 (specs/tracking/tint-integration-refactor.md, slice R6): pins the
    /// GLSL uniform name this HAL queries against the exact array-element
    /// syntax Tint's GLSL writer emits (see
    /// `yawgpu-tint`'s `generate_glsl_first_instance_offset_only_applied_when_requested`,
    /// which pins the shim side of this same contract: `layout(location = 0)
    /// uniform uint tint_immediates[1];`).
    #[test]
    fn first_instance_uniform_name_matches_tint_immediate_array_element_zero() {
        assert_eq!(FIRST_INSTANCE_UNIFORM_NAME, "tint_immediates[0]");
    }
}
