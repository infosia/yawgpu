use std::sync::Arc;

use glow::HasContext;

use super::device::GlesDeviceInner;
use super::format::map_vertex_format;
use super::{rebuild_hal_error, BACKEND};
use crate::{
    HalColorTargetState, HalDescriptorBinding, HalError, HalPrimitiveTopology,
    HalRenderPipelineDescriptor, HalTextureFormat, HalVertexBufferLayout,
};

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
    primitive_topology: HalPrimitiveTopology,
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
                color_target: descriptor.color_targets.first().copied(),
                primitive_topology: descriptor.primitive_topology,
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
    pub(super) fn primitive_topology(&self) -> HalPrimitiveTopology {
        self.inner.primitive_topology
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
            bind_program_blocks_from_source(gl, program, source);
            Ok(program)
        })
        .and_then(|result| result)
}

fn validate_render_pipeline_descriptor(
    descriptor: &HalRenderPipelineDescriptor,
) -> Result<(), HalError> {
    if descriptor.color_targets.len() > 1 {
        return Err(HalError::BufferOperationFailed {
            backend: BACKEND,
            message: "GLES render pipeline supports at most one color target",
        });
    }
    if let Some(target) = descriptor.color_targets.first() {
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
    if descriptor.color_targets.is_empty() && descriptor.depth_stencil.is_none() {
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
            bind_program_blocks_from_source(gl, program, vertex_source);
            if let Some(fragment_source) = fragment_source {
                bind_program_blocks_from_source(gl, program, fragment_source);
            }
            let first_instance_location =
                gl.get_uniform_location(program, "naga_vs_first_instance");
            Ok((program, first_instance_location))
        })
        .and_then(|result| result)
}

fn bind_program_blocks_from_source(gl: &glow::Context, program: glow::Program, source: &str) {
    for line in source.lines().map(str::trim) {
        let words = line.split_whitespace().collect::<Vec<_>>();
        let Some((kind, name)) = block_kind_and_name(&words) else {
            continue;
        };
        let Some(binding) = block_binding_from_name(name) else {
            continue;
        };
        unsafe {
            match kind {
                BlockKind::Uniform => {
                    if let Some(index) = gl.get_uniform_block_index(program, name) {
                        gl.uniform_block_binding(program, index, binding);
                    }
                }
                BlockKind::Storage => {
                    if let Some(index) = gl.get_shader_storage_block_index(program, name) {
                        gl.shader_storage_block_binding(program, index, binding);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockKind {
    Uniform,
    Storage,
}

fn block_kind_and_name<'a>(words: &'a [&str]) -> Option<(BlockKind, &'a str)> {
    let (kind, index) = words
        .iter()
        .position(|word| *word == "uniform")
        .map(|index| (BlockKind::Uniform, index))
        .or_else(|| {
            words
                .iter()
                .position(|word| *word == "buffer")
                .map(|index| (BlockKind::Storage, index))
        })?;
    words
        .get(index + 1)
        .map(|name| (kind, name.trim_end_matches('{')))
}

fn block_binding_from_name(name: &str) -> Option<u32> {
    let rest = name.split_once("_block_")?.1;
    let digits = rest
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
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

    #[test]
    fn block_binding_from_name_extracts_naga_binding_suffix() {
        assert_eq!(block_binding_from_name("Data_block_0Compute"), Some(0));
        assert_eq!(block_binding_from_name("Color_block_12Fragment"), Some(12));
        assert_eq!(block_binding_from_name("Data"), None);
        assert_eq!(block_binding_from_name("Data_block_Compute"), None);
    }

    #[test]
    fn block_kind_and_name_parses_uniform_and_storage_lines() {
        assert_eq!(
            block_kind_and_name(&["layout(std140)", "uniform", "Color_block_1Fragment", "{"]),
            Some((BlockKind::Uniform, "Color_block_1Fragment"))
        );
        assert_eq!(
            block_kind_and_name(&[
                "layout(std430)",
                "readonly",
                "buffer",
                "Data_block_0Compute",
                "{",
            ]),
            Some((BlockKind::Storage, "Data_block_0Compute"))
        );
        assert_eq!(block_kind_and_name(&["void", "main()"]), None);
    }
}
